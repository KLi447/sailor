from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from .adapters import Task, LoRAConfig, TaskStatus
from .database import registry
import os
import shutil
from datetime import datetime
from pathlib import Path

app = FastAPI(title="LoRA Fine-tuning Server", version="0.2")

DATASET_DIR = Path(__file__).resolve().parent.parent / "server/datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/submit_job")
async def submit_job(
    r: int = Form(...),
    alpha: int = Form(...),
    model_name: str = Form(...),
    target_modules: str = Form(...),
    dataset: UploadFile = File(...)
):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{dataset.filename}"
    dataset_path = DATASET_DIR / safe_filename

    try:
        with open(dataset_path, "wb") as f:
            shutil.copyfileobj(dataset.file, f)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save dataset: {e}"}
        )

    lora_cfg = LoRAConfig(
        r=r,
        alpha=alpha,
        model_name=model_name,
        target_modules=[t.strip() for t in target_modules.split(",") if t.strip()]
    )

    task = Task(
        lora_config=lora_cfg,
        dataset_path=str(dataset_path),
        dataset_name=dataset.filename,
        status=TaskStatus.RECEIVED
    )
    registry.add_task(task)

    return {
        "job_id": task.id,
        "status": task.status,
        "dataset_stored_at": str(dataset_path)
    }


@app.get("/tasks")
async def list_tasks():
    return registry.list_tasks()


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = registry.get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return task
