import json
import logging
from abc import ABC
from typing import Any, List, Tuple

logger = logging.getLogger("agent_eval")


class Task(ABC):
    """Base class for a task instance."""

    task_name: str = "base"

    def __init__(self, **kwargs) -> None:
        self.task_id: Any = kwargs.get("task_id", None)
        self.metadata = {}
        
    @classmethod
    def load_tasks(cls, path: str) -> Tuple[List["Task"], int]:
        """Load all the tasks from a given jsonl file."""
        assert path.endswith(".jsonl") or path.endswith(".json")
        with open(path, "r") as f:
            tasks = [cls(**json.loads(line)) for line in f.readlines()]
        logger.info(f"Loaded {len(tasks)} tasks from {path}")
        return tasks, len(tasks)

    def to_dict(self) -> dict:
        """Convert the task to a dictionary."""
        return {
            "task_name": self.task_name,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "reference": self.reference,
            "metadata": self.metadata,
        }
