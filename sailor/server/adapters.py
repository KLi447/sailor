from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional
from uuid import uuid4
from datetime import datetime

class TaskStatus(str, Enum):
    QUEUED = "queued"
    RECEIVED = "received"
    FAILED = "failed"
    COMPLETED = "completed"

class LoRAConfig(BaseModel):
    r: int = Field(..., description="LoRA rank")
    alpha: int = Field(..., description="Scaling factor")
    target_modules: List[str] = Field(..., description="Target layers to inject LoRA into")
    model_name: str = Field(..., description="Base model name")

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.RECEIVED
    lora_config: LoRAConfig
    dataset_path: str
    dataset_name: str
