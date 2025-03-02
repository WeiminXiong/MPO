import os
import json
import yaml
import logging
from typing import Iterable, List, Tuple

from scienceworld import ScienceWorldEnv

from tasks.base import Task

logger = logging.getLogger("agent_eval")


class SciWorldTask(Task):
    """ScienceWorld task instance."""
    
    task_name = "sciworld"
    
    def __init__(
        self,
        sub_task_name: str,
        variation_idx: int,
        workflow: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_task_name = sub_task_name
        self.variation_idx = variation_idx
        self.workflow = workflow
    
    @classmethod
    def load_tasks(cls, path: str, workflow_path: str = None, split: str = "test", part_num: int = 1, part_idx: int = -1):
        if split == 'train':
            task_idxs = json.load(open("data/sciworld/train_indices.json"))
        elif split == 'dev':
            task_idxs = json.load(open("data/sciworld/dev_indices.json"))
        elif split == 'test':
            task_idxs = json.load(open("data/sciworld/test_indices.json"))
        else:
            raise ValueError
        taskname2id = json.load(open("data/sciworld/taskname2id.json"))

        if workflow_path is not None:
            with open(workflow_path, "r") as fr:
                raw_workflows = fr.readlines()
                workflows = [json.loads(w) for w in raw_workflows]
        else:
            workflows = [None] * len(task_idxs)


        if part_num == 1:
            task_idxs = task_idxs
        else:
            assert part_idx != -1
            part_len = len(task_idxs) // part_num + 1
            task_idxs = task_idxs[part_len * part_idx: part_len * (part_idx + 1)]
            workflows = workflows[part_len * part_idx: part_len * (part_idx + 1)]
        N_TASKS = len(task_idxs)

        task_id_2_workflow = {}
        for item in workflows:
            if item is None:
                continue
            task_id = item['id']
            workflow = item["workflow"]
            task_id_2_workflow[task_id] = workflow

        
        def generator():
            for item in task_idxs:
                task_name = item[0]
                variation_idx = item[1]
                yield cls(
                    task_id=f"{taskname2id[task_name]}_{variation_idx}",
                    sub_task_name=task_name,
                    variation_idx=variation_idx,
                    workflow=task_id_2_workflow.get(f"{taskname2id[task_name]}_{variation_idx}", None),
                )
                    
        return generator(), N_TASKS
    