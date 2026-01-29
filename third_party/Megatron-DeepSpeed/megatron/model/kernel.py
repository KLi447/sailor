import torch
import triton
import triton.language as tl
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import deque
import time
import random


@dataclass  
class LoRAConfig:
    rank: int
    alpha: float
    batch_size: int
    adapter_id: int


@dataclass
class NanoFlowConfig:
    initial_nano_batch: int = 16384
    min_nano_batch: int = 1024
    max_nano_batch: int = 2e20

    slow_start_threshold: int = 16384
    multiplicative_increase: float = 2.0
    additive_increase: int = 8192
    multiplicative_decrease: float = 0.75

    num_streams: int = 3

    update_interval: int = 5
    warmup_iterations: int = 3


class AIMDController:    
    def __init__(self, config: NanoFlowConfig):
        self.config = config
        self.nano_batch_size = config.initial_nano_batch
        
        self.iteration = 0
        self.throughput_history: deque = deque(maxlen=5)
        self.best_throughput = 0.0
        self.best_batch_size = config.initial_nano_batch

        self.in_slow_start = True
        self.ssthresh = config.slow_start_threshold
        self.congestion_detected = False
    
    def get_nano_batch_size(self) -> int:
        return int(self.nano_batch_size)
    
    def record_iteration(self, tokens_processed: int, time_ms: float):
        self.iteration += 1
        
        throughput = tokens_processed / (time_ms + 1e-6)
        self.throughput_history.append(throughput)

        if self.iteration <= self.config.warmup_iterations:
            return

        if self.iteration % self.config.update_interval != 0:
            return
        
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history)

        if avg_throughput > self.best_throughput:
            self.best_throughput = avg_throughput
            self.best_batch_size = self.nano_batch_size
            self.congestion_detected = False

        if avg_throughput < 0.9 * self.best_throughput and not self.congestion_detected:
            self.ssthresh = max(
                self.nano_batch_size * self.config.multiplicative_decrease,
                self.config.min_nano_batch
            )
            self.nano_batch_size = self.ssthresh
            self.in_slow_start = False
            self.congestion_detected = True
            return
        
        # Increase phase
        if self.in_slow_start and self.nano_batch_size < self.ssthresh:
            self.nano_batch_size = min(
                self.nano_batch_size * self.config.multiplicative_increase,
                self.ssthresh,
                self.config.max_nano_batch
            )
        else:
            self.in_slow_start = False
            if avg_throughput >= 0.95 * self.best_throughput:
                self.nano_batch_size = min(
                    self.nano_batch_size + self.config.additive_increase,
                    self.config.max_nano_batch
                )
    
    def get_stats(self) -> Dict:
        return {
            'nano_batch_size': int(self.nano_batch_size),
            'best_batch_size': int(self.best_batch_size),
            'best_throughput': self.best_throughput,
            'iteration': self.iteration,
            'in_slow_start': self.in_slow_start,
            'ssthresh': int(self.ssthresh),
        }


class NanoFlowExecutor:    
    def __init__(self, config: NanoFlowConfig, device: torch.device):
        self.config = config
        self.device = device
        self.aimd = AIMDController(config)

        self.streams = [
            torch.cuda.Stream(device=device)
            for _ in range(config.num_streams)
        ]

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def get_nano_batch_boundaries(self, total_size: int) -> List[Tuple[int, int]]:
        nano_batch_size = self.aimd.get_nano_batch_size()
        
        if total_size <= nano_batch_size:
            return [(0, total_size)]
        
        boundaries = []
        start = 0
        while start < total_size:
            end = min(start + nano_batch_size, total_size)
            boundaries.append((start, end))
            start = end
        
        return boundaries
    
    def get_stream(self, idx: int) -> torch.cuda.Stream:
        return self.streams[idx % self.config.num_streams]
    
    def record_timing(self, tokens: int, time_ms: float):
        self.aimd.record_iteration(tokens, time_ms)
    
    def get_stats(self) -> Dict:
        return self.aimd.get_stats()
    
    def synchronize_all(self):
        for stream in self.streams:
            stream.synchronize()

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_R': 16, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_R': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'K', 'R'],
)
@triton.jit
def lora_stage1_kernel(
    x_ptr, a_ptr, inter_ptr, rows_ptr,
    M, K, R,
    stride_xm, stride_xk,
    stride_ar, stride_ak,
    stride_im, stride_ir,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_r_blocks = tl.cdiv(R, BLOCK_R)
    pid_m = pid // num_r_blocks
    pid_r = pid % num_r_blocks
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_k = tl.arange(0, BLOCK_K)
    
    row_idxs = tl.load(rows_ptr + offs_m, mask=offs_m < M, other=0)
    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    
    x_ptrs = x_ptr + row_idxs[:, None] * stride_xm + offs_k[None, :] * stride_xk
    a_ptrs = a_ptr + offs_r[None, :] * stride_ar + offs_k[:, None] * stride_ak
    
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0).to(tl.float32)
        a_tile = tl.load(a_ptrs, mask=k_mask[:, None] & (offs_r[None, :] < R), other=0.0).to(tl.float32)
        acc += tl.dot(x_tile, a_tile, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        a_ptrs += BLOCK_K * stride_ak
    
    inter_ptrs = inter_ptr + offs_m[:, None] * stride_im + offs_r[None, :] * stride_ir
    mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    tl.store(inter_ptrs, acc.to(tl.float16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_R': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_R': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_R': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'R', 'N'],
)
@triton.jit
def lora_stage2_add_kernel(
    inter_ptr, b_ptr, out_ptr, rows_ptr,
    M, R, N,
    stride_im, stride_ir,
    stride_br, stride_bn,
    stride_outm, stride_outn,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    
    row_idxs = tl.load(rows_ptr + offs_m, mask=offs_m < M, other=0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    inter_ptrs = inter_ptr + offs_m[:, None] * stride_im + offs_r[None, :] * stride_ir
    b_ptrs = b_ptr + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn
    
    for r_start in range(0, R, BLOCK_R):
        r_mask = (r_start + offs_r) < R
        inter_tile = tl.load(inter_ptrs, mask=(offs_m[:, None] < M) & r_mask[None, :], other=0.0).to(tl.float32)
        b_tile = tl.load(b_ptrs, mask=r_mask[:, None] & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(inter_tile, b_tile, allow_tf32=True)
        inter_ptrs += BLOCK_R * stride_ir
        b_ptrs += BLOCK_R * stride_br
    
    acc *= scale
    out_ptrs = out_ptr + row_idxs[:, None] * stride_outm + offs_n[None, :] * stride_outn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    current = tl.load(out_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    tl.store(out_ptrs, (current + acc).to(tl.float16), mask=out_mask)


class NanoFlowAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, batch_to_adapter, module, base_weight, *lora_params):
        num_adapters = module.num_adapters
        lora_a = lora_params[:num_adapters]
        lora_b = lora_params[num_adapters:]
        scales = module.scales

        ctx.save_for_backward(x, batch_to_adapter, base_weight, *lora_params)
        ctx.module = module
        
        x = x.contiguous()
        M, K = x.shape
        N = module.out_features

        executor = module._get_executor()

        executor.start_event.record()

        out = torch.empty(M, N, device=module.device, dtype=module.dtype)

        nano_batches = executor.get_nano_batch_boundaries(M)
        
        # If single nano-batch, use original path
        if len(nano_batches) == 1:
            adapter_indices = module._get_adapter_indices(batch_to_adapter)
            intermediates = []
            
            for adapter_id, indices in enumerate(adapter_indices):
                if len(indices) > 0:
                    inter = torch.empty(len(indices), module.adapter_ranks[adapter_id],
                                       device=module.device, dtype=module.dtype)
                else:
                    inter = None
                intermediates.append(inter)
            
            base_done = torch.cuda.Event()

            with torch.cuda.stream(module._streams[0]):
                torch.mm(x, base_weight.t(), out=out)
                base_done.record(module._streams[0])

            with torch.cuda.stream(module._streams[1]):
                for adapter_id, indices in enumerate(adapter_indices):
                    if len(indices) == 0:
                        continue
                    inter = intermediates[adapter_id]
                    module._launch_triton_stage1(x, lora_a[adapter_id], inter, indices,
                                               len(indices), K, module.adapter_ranks[adapter_id])
                
                module._streams[1].wait_event(base_done)

                for adapter_id, indices in enumerate(adapter_indices):
                    if len(indices) == 0:
                        continue
                    inter = intermediates[adapter_id]
                    module._launch_triton_stage2(inter, lora_b[adapter_id], out, indices,
                                               len(indices), module.adapter_ranks[adapter_id], N, scales[adapter_id])
            
            torch.cuda.synchronize()
        else:
            for nb_idx, (nb_start, nb_end) in enumerate(nano_batches):
                stream = executor.get_stream(nb_idx)
                
                with torch.cuda.stream(stream):
                    x_nb = x[nb_start:nb_end]
                    batch_to_adapter_nb = batch_to_adapter[nb_start:nb_end]
                    M_nb = nb_end - nb_start

                    adapter_indices_nb = [
                        torch.where(batch_to_adapter_nb == i)[0]
                        for i in range(num_adapters)
                    ]

                    adapter_indices_global = [
                        idx + nb_start for idx in adapter_indices_nb
                    ]

                    intermediates_nb = []
                    for adapter_id, local_indices in enumerate(adapter_indices_nb):
                        if len(local_indices) > 0:
                            inter = torch.empty(len(local_indices), module.adapter_ranks[adapter_id],
                                              device=module.device, dtype=module.dtype)
                        else:
                            inter = None
                        intermediates_nb.append(inter)

                    out_nb = out[nb_start:nb_end]
                    torch.mm(x_nb, base_weight.t(), out=out_nb)

                    for adapter_id, local_indices in enumerate(adapter_indices_nb):
                        if len(local_indices) == 0:
                            continue
                        inter = intermediates_nb[adapter_id]
                        module._launch_triton_stage1(
                            x_nb, lora_a[adapter_id], inter, local_indices,
                            len(local_indices), K, module.adapter_ranks[adapter_id]
                        )

                    for adapter_id, local_indices in enumerate(adapter_indices_nb):
                        if len(local_indices) == 0:
                            continue
                        inter = intermediates_nb[adapter_id]
                        module._launch_triton_stage2(
                            inter, lora_b[adapter_id], out_nb, local_indices,
                            len(local_indices), module.adapter_ranks[adapter_id], N, scales[adapter_id]
                        )

            executor.synchronize_all()

        executor.end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = executor.start_event.elapsed_time(executor.end_event)
        executor.record_timing(M, elapsed_ms)
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        x = saved[0]
        batch_to_adapter = saved[1]
        base_weight = saved[2]
        
        module = ctx.module
        num_adapters = module.num_adapters
        lora_a_list = saved[3 : 3 + num_adapters]
        lora_b_list = saved[3 + num_adapters :]
        scales = module.scales
        
        M, K = x.shape
        N = grad_output.shape[1]

        executor = module._get_executor()
        nano_batches = executor.get_nano_batch_boundaries(M)

        grad_base_weight = torch.mm(grad_output.t(), x)
        grad_x = torch.mm(grad_output, base_weight)

        grad_lora_a_list = [torch.zeros_like(a) for a in lora_a_list]
        grad_lora_b_list = [torch.zeros_like(b) for b in lora_b_list]

        for nb_idx, (nb_start, nb_end) in enumerate(nano_batches):
            stream = executor.get_stream(nb_idx)
            
            with torch.cuda.stream(stream):
                x_nb = x[nb_start:nb_end]
                grad_output_nb = grad_output[nb_start:nb_end]
                batch_to_adapter_nb = batch_to_adapter[nb_start:nb_end]

                adapter_indices_nb = [
                    torch.where(batch_to_adapter_nb == i)[0]
                    for i in range(num_adapters)
                ]
                
                for i in range(num_adapters):
                    local_indices = adapter_indices_nb[i]
                    if len(local_indices) == 0:
                        continue

                    x_slice = x_nb[local_indices]
                    d_out_slice = grad_output_nb[local_indices]
                    scale = scales[i]

                    inter = torch.mm(x_slice, lora_a_list[i].t())

                    grad_b = torch.mm(inter.t(), d_out_slice) * scale
                    grad_lora_b_list[i] += grad_b

                    d_inter = torch.mm(d_out_slice, lora_b_list[i].t()) * scale

                    grad_a = torch.mm(d_inter.t(), x_slice)
                    grad_lora_a_list[i] += grad_a

                    d_x_slice = torch.mm(d_inter, lora_a_list[i])

                    global_indices = nb_start + local_indices
                    grad_x[global_indices] += d_x_slice

        executor.synchronize_all()
        
        return (grad_x, None, None, grad_base_weight, *grad_lora_a_list, *grad_lora_b_list)
