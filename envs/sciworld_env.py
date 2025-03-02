import re
import json
import logging
from typing import Tuple

from scienceworld import ScienceWorldEnv

from envs import BaseEnv
from tasks import SciWorldTask
from prompt import prompt_with_icl
from utils.datatypes import State


logger = logging.getLogger("agent_frame")


class SciWorldEnv(BaseEnv):
    def __init__(
        self,
        task: SciWorldTask,
        env: ScienceWorldEnv,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: SciWorldTask = task
        self.env = env
        self.max_steps_dict = json.load(open("data/sciworld/max_steps.json"))

        self.max_error_step = 10
        
        self.state = State()
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
        except:
            observation = f"Observation: Invalid format. The input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.state.reward = 0
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state
        try:
            observation, _, done, info = self.env.step(action)
            reward = info['raw_score']

            if "No known action matches that input" in observation:
                self.state.error_step += 1
                if self.state.error_step >= self.max_error_step:
                    self.state.finished = True
                    self.state.success = False
                    self.state.terminate_reason = "max_error_steps"
            else:
                self.state.error_step = 0

            observation = f"Observation: {observation}"
            if self.state.reward is None or reward > self.state.reward:
                self.state.reward = reward
        except AssertionError:
            observation = 'Observation: Invalid action!'
            done = False

        if self.args.incorporation_type == "observation" and self.task.workflow:
            observation+=f"\n\nThis workflow maybe helpful for you to complete the task:\n{self.task.workflow}"

        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"

        return observation, self.state
    
    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.state.error_step = 0
        self.max_steps = self.max_steps_dict[self.task.sub_task_name]
        self.env.load(self.task.sub_task_name, self.task.variation_idx, simplificationStr="easy", generateGoldPath=False)
        obs, info = self.env.reset()
        cur_task = info['taskDesc']
        self.taskDesc = cur_task
        if self.args.incorporation_type == "query":
            observation, messages = prompt_with_icl(
                instruction=self.instruction,
                raw_icl=self.raw_icl,
                cur_task=cur_task,
                icl_num=1,
                workflow=self.task.workflow,
            )
        else:
            observation, messages = prompt_with_icl(
                instruction=self.instruction,
                raw_icl=self.raw_icl,
                cur_task=cur_task,
                icl_num=1,
                workflow=None,
            )

        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
