import os
import json
import yaml
import logging
from typing import Iterable, Tuple

import alfworld
import alfworld.agents.environment as envs

from tasks.base import Task


logger = logging.getLogger("agent_eval")

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """Alfworld task instance."""

    task_name = "alfworld"

    def __init__(
        self,
        task_name: str,
        env: envs.AlfredTWEnv,
        task_type: str,
        obs: str,
        workflow: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.task_type = task_type
        self.observation = obs
        self.workflow = workflow

        self.env = env

    
    @classmethod
    def load_tasks(
        cls, 
        path: str, 
        workflow_path: str = None,
        split: str = "test",
        part_num: int = 1,
        part_idx: int = -1,
    ) -> Tuple[Iterable[Task], int]:
        """Load alfworld data and prompts from a directory."""
        os.environ["ALFWORLD_DATA"] = path

        with open(os.path.join(path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)

        # Split following ReAct
        # https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb
        if split == 'train':
            split = "train"
            N_TASKS = 3553
        elif split == 'dev':
            split = "eval_in_distribution"
            N_TASKS = 140
        elif split == 'test':
            split = "eval_out_of_distribution"
            N_TASKS = 134

        env = getattr(alfworld.agents.environment, config["env"]["type"])(
            config, train_eval=split
        )
        assert isinstance(env, alfworld.agents.environment.AlfredTWEnv)
        env = env.init_env(batch_size=1)

        if workflow_path is not None:
            with open(workflow_path) as fr:
                raw_workflows = fr.readlines()
                workflows = [json.loads(w) for w in raw_workflows]
        else:
            workflows = [None] * N_TASKS

        if part_num > 1:
            assert part_idx != -1
            part_inst_num = [N_TASKS // part_num] * part_num
            part_inst_num[-1] += N_TASKS % part_num
            # jump to the start of the part
            env.skip(sum(part_inst_num[:part_idx]))
            workflows = workflows[sum(part_inst_num[:part_idx]):sum(part_inst_num[:part_idx+1])]
            N_TASKS = part_inst_num[part_idx]
            
        obs_2_workflow = {}
        for item in workflows:
            if item is None:
                continue
            obs = item['task']
            workflow = item['workflow']
            obs_2_workflow[obs] = workflow

        def generator():
            for idx in range(N_TASKS):
                obs, info = env.reset()
                obs = "\n".join(obs[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]
                
                name = "/".join(game_file.split("/")[-3:-1])

                task_type = None
                for _, (k, v) in enumerate(PREFIXES.items()):
                    if name.startswith(k):
                        task_type = k
                        break
                assert task_type is not None, f"Task type not found for {name}"

                yield cls(
                    task_id=idx,
                    task_name=name,
                    env=env,
                    task_type=task_type,
                    obs=obs,
                    workflow=obs_2_workflow.get(obs, None),
                )

        return generator(), N_TASKS
