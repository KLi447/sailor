from typing import Dict
from .adapters import Task

class TaskRegistry:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}

    def add_task(self, task: Task):
        self._tasks[task.id] = task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        return list(self._tasks.values())

registry = TaskRegistry()
