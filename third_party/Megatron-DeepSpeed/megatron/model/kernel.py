import torch
import triton
import triton.language as tl
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    rank: int
    alpha: float
    batch_size: int
    adapter_id: int


# =============================================================================
# FUSED MULTI-ADAPTER FORWARD KERNEL
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scaling
    scale,
    # Whether to accumulate or overwrite
    accumulate: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    General matmul kernel: C = A @ B * scale (+ C if accumulate)
    Uses tl.dot for tensor core acceleration.
    """
    pid = tl.program_id(0)
    
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to first block
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=True)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Scale
    acc *= scale
    
    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    if accumulate:
        current = tl.load(c_ptrs, mask=mask, other=0.0)
        acc += current
    
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'R', 'K'],
)
@triton.jit
def lora_first_matmul_kernel(
    # X: [M, K], A: [R, K] -> intermediate: [M, R]
    x_ptr, a_ptr, out_ptr,
    M, K, R,
    stride_xm, stride_xk,
    stride_ar, stride_ak,
    stride_outm, stride_outr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # This is BLOCK_R
    BLOCK_K: tl.constexpr,
):
    """
    Compute X @ A^T -> [M, R]
    A is [R, K], so A^T is [K, R]
    """
    pid = tl.program_id(0)
    
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_r_blocks = tl.cdiv(R, BLOCK_N)
    
    pid_m = pid // num_r_blocks
    pid_r = pid % num_r_blocks
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = pid_r * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # X: [M, K] - load rows
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # A: [R, K] - load rows, but we want A^T so we'll load columns
    # A^T[k, r] = A[r, k]
    a_ptrs = a_ptr + offs_r[None, :] * stride_ar + offs_k[:, None] * stride_ak
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask = offs_k < k_remaining
        
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        # Load A^T: [BLOCK_K, BLOCK_R]
        a_t = tl.load(a_ptrs, mask=k_mask[:, None] & (offs_r[None, :] < R), other=0.0)
        
        acc += tl.dot(x, a_t, allow_tf32=True)
        
        x_ptrs += BLOCK_K * stride_xk
        a_ptrs += BLOCK_K * stride_ak
    
    # Store [M, R]
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_r[None, :] * stride_outr
    mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'R'],
)
@triton.jit
def lora_second_matmul_add_kernel(
    # intermediate: [M, R], B: [R, N] -> out += intermediate @ B * scale
    inter_ptr, b_ptr, out_ptr,
    M, R, N,
    stride_im, stride_ir,
    stride_br, stride_bn,
    stride_outm, stride_outn,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # This is BLOCK_R
):
    """
    Compute out += intermediate @ B * scale
    intermediate: [M, R], B: [R, N]
    """
    pid = tl.program_id(0)
    
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_K)
    
    inter_ptrs = inter_ptr + offs_m[:, None] * stride_im + offs_r[None, :] * stride_ir
    b_ptrs = b_ptr + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for r in range(0, R, BLOCK_K):
        r_remaining = R - r
        r_mask = offs_r < r_remaining
        
        inter = tl.load(inter_ptrs, mask=(offs_m[:, None] < M) & r_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=r_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        
        acc += tl.dot(inter, b, allow_tf32=True)
        
        inter_ptrs += BLOCK_K * stride_ir
        b_ptrs += BLOCK_K * stride_br
    
    # Scale and add to output
    acc *= scale
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    current = tl.load(out_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, (current + acc).to(tl.float16), mask=mask)


# =============================================================================
# BACKWARD KERNELS
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def lora_backward_dx_kernel(
    # grad_out: [M, N], B: [R, N], A: [R, K] -> grad_x += grad_out @ B^T @ A * scale
    # We compute: grad_out @ B^T -> [M, R], then [M, R] @ A -> [M, K]
    grad_out_ptr, b_ptr, a_ptr, grad_x_ptr,
    M, N, R, K,
    stride_gom, stride_gon,
    stride_br, stride_bn,
    stride_ar, stride_ak,
    stride_gxm, stride_gxk,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused backward for dx: grad_x += (grad_out @ B^T @ A) * scale
    """
    pid = tl.program_id(0)
    
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_k_blocks = tl.cdiv(K, BLOCK_N)  # Output is [M, K]
    
    pid_m = pid // num_k_blocks
    pid_k = pid % num_k_blocks
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # For each rank r, accumulate: grad_out[m, :] @ B[r, :]^T * A[r, k]
    # = sum_n grad_out[m, n] * B[r, n] * A[r, k]
    # = sum_r (sum_n grad_out[m, n] * B[r, n]) * A[r, k]
    
    offs_r = tl.arange(0, BLOCK_K)
    
    for r_start in range(0, R, BLOCK_K):
        r_offs = r_start + offs_r
        r_mask = r_offs < R
        
        # Compute grad_out @ B^T for this rank block -> [BLOCK_M, BLOCK_K]
        inter = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        
        offs_n = tl.arange(0, BLOCK_K)
        for n_start in range(0, N, BLOCK_K):
            n_offs = n_start + offs_n
            n_mask = n_offs < N
            
            # grad_out: [M, N]
            go = tl.load(
                grad_out_ptr + offs_m[:, None] * stride_gom + n_offs[None, :] * stride_gon,
                mask=(offs_m[:, None] < M) & n_mask[None, :],
                other=0.0
            )
            # B^T: [N, R] from B: [R, N]
            b_t = tl.load(
                b_ptr + r_offs[None, :] * stride_br + n_offs[:, None] * stride_bn,
                mask=n_mask[:, None] & r_mask[None, :],
                other=0.0
            )
            inter += tl.dot(go, b_t, allow_tf32=True)
        
        # Now inter is [BLOCK_M, BLOCK_K] containing grad_out @ B^T for ranks [r_start:r_start+BLOCK_K]
        # Multiply by A[r, k] for output columns
        # A: [R, K]
        a_block = tl.load(
            a_ptr + r_offs[:, None] * stride_ar + offs_k[None, :] * stride_ak,
            mask=r_mask[:, None] & (offs_k[None, :] < K),
            other=0.0
        )
        
        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(inter, a_block, allow_tf32=True)
    
    acc *= scale
    
    # Add to grad_x
    gx_ptrs = grad_x_ptr + offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    current = tl.load(gx_ptrs, mask=mask, other=0.0)
    tl.store(gx_ptrs, (current + acc).to(tl.float16), mask=mask)


# =============================================================================
# AUTOGRAD FUNCTION
# =============================================================================

class OptimizedLoRAFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        lora_a_list: List[torch.Tensor],
        lora_b_list: List[torch.Tensor],
        scales: List[float],
        batch_to_adapter: torch.Tensor,
        adapter_indices: List[torch.Tensor],
    ):
        M, K = x.shape
        N = base_weight.shape[0]
        
        # Base matmul - use PyTorch (cuBLAS)
        out = torch.nn.functional.linear(x, base_weight)
        
        # LoRA contributions using optimized two-matmul approach
        for adapter_id, indices in enumerate(adapter_indices):
            if len(indices) == 0:
                continue
            
            lora_a = lora_a_list[adapter_id]  # [R, K]
            lora_b = lora_b_list[adapter_id]  # [R, N]
            scale = scales[adapter_id]
            R = lora_a.shape[0]
            
            x_group = x[indices].contiguous()  # [M_group, K]
            M_group = x_group.shape[0]
            
            # Allocate intermediate [M_group, R]
            intermediate = torch.empty(M_group, R, device=x.device, dtype=x.dtype)
            
            # First matmul: X @ A^T -> [M_group, R]
            grid1 = lambda meta: (
                triton.cdiv(M_group, meta['BLOCK_M']) * triton.cdiv(R, meta['BLOCK_N']),
            )
            lora_first_matmul_kernel[grid1](
                x_group, lora_a, intermediate,
                M_group, K, R,
                x_group.stride(0), x_group.stride(1),
                lora_a.stride(0), lora_a.stride(1),
                intermediate.stride(0), intermediate.stride(1),
            )
            
            # Second matmul: intermediate @ B -> add to out[indices]
            out_group = out[indices].contiguous()
            
            grid2 = lambda meta: (
                triton.cdiv(M_group, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
            )
            lora_second_matmul_add_kernel[grid2](
                intermediate, lora_b, out_group,
                M_group, R, N,
                intermediate.stride(0), intermediate.stride(1),
                lora_b.stride(0), lora_b.stride(1),
                out_group.stride(0), out_group.stride(1),
                scale,
            )
            
            out[indices] = out_group
        
        ctx.save_for_backward(x, base_weight, batch_to_adapter)
        ctx.lora_a_list = lora_a_list
        ctx.lora_b_list = lora_b_list
        ctx.scales = scales
        ctx.adapter_indices = adapter_indices
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, base_weight, batch_to_adapter = ctx.saved_tensors
        lora_a_list = ctx.lora_a_list
        lora_b_list = ctx.lora_b_list
        scales = ctx.scales
        adapter_indices = ctx.adapter_indices
        
        grad_output = grad_output.contiguous()
        M, K = x.shape
        N = grad_output.shape[1]
        
        # Base gradient using cuBLAS
        grad_x = torch.nn.functional.linear(grad_output, base_weight.t())
        grad_base_weight = grad_output.t() @ x
        
        grad_lora_a_list = []
        grad_lora_b_list = []
        
        for adapter_id, indices in enumerate(adapter_indices):
            lora_a = lora_a_list[adapter_id]
            lora_b = lora_b_list[adapter_id]
            scale = scales[adapter_id]
            R = lora_a.shape[0]
            
            if len(indices) == 0:
                grad_lora_a_list.append(torch.zeros_like(lora_a))
                grad_lora_b_list.append(torch.zeros_like(lora_b))
                continue
            
            x_group = x[indices].contiguous()
            grad_out_group = grad_output[indices].contiguous()
            M_group = x_group.shape[0]
            
            # LoRA backward for grad_x using fused kernel
            grad_x_group = grad_x[indices].contiguous()
            
            grid = lambda meta: (
                triton.cdiv(M_group, meta['BLOCK_M']) * triton.cdiv(K, meta['BLOCK_N']),
            )
            lora_backward_dx_kernel[grid](
                grad_out_group, lora_b, lora_a, grad_x_group,
                M_group, N, R, K,
                grad_out_group.stride(0), grad_out_group.stride(1),
                lora_b.stride(0), lora_b.stride(1),
                lora_a.stride(0), lora_a.stride(1),
                grad_x_group.stride(0), grad_x_group.stride(1),
                scale,
            )
            grad_x[indices] = grad_x_group
            
            # LoRA weight gradients using PyTorch (cuBLAS is good here)
            # grad_A = scale * (grad_out @ B^T)^T @ x = scale * B @ grad_out^T @ x
            # grad_B = scale * (x @ A^T)^T @ grad_out = scale * A @ x^T @ grad_out
            
            # Intermediate: x @ A^T -> [M_group, R]
            inter = x_group @ lora_a.t()
            
            # grad_B = inter^T @ grad_out * scale = [R, M] @ [M, N] = [R, N]
            grad_b = (inter.t() @ grad_out_group) * scale
            
            # grad_A = (grad_out @ B^T)^T @ x * scale = [R, M] @ [M, K] = [R, K]
            # grad_out @ B^T = [M, N] @ [N, R] = [M, R]
            grad_a = ((grad_out_group @ lora_b.t()).t() @ x_group) * scale
            
            grad_lora_a_list.append(grad_a)
            grad_lora_b_list.append(grad_b)
        
        return grad_x, grad_base_weight, None, None, None, None, None