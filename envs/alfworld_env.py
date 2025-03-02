import re
import json
import logging
from typing import Any, Dict, List, Tuple

from envs import BaseEnv
from tasks import AlfWorldTask
from prompt import prompt_with_icl
from utils.datatypes import State


logger = logging.getLogger("agent_eval")


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


class AlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.state = State()
        self.bad_steps = 0
        self.max_bad_steps = 50

    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        put_action = re.findall(r"put\s+(.*)\s+[io]n\s+(.*)", action)
        if put_action:
            action = f"put {put_action[0][0]} in/on {put_action[0][1]}"
        assert action is not None
        return action
    
    def conduct_action(self, action: str):
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        return observation, reward, done
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
            observation, reward, done = self.conduct_action(action)
        except Exception as e:
            self.state.success = False
            self.state.finished = False
            self.state.reward=0
            observation = f"Observation: Error Input. Your input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.bad_steps += 1
            if self.state.steps >= self.max_steps or self.bad_steps >= self.max_bad_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state


        observation = f"Observation: {observation}"

        if self.args.incorporation_type == "observation" and self.task.workflow:
            observation+=f"\n\nThis workflow maybe helpful for you to complete the task:\n{self.task.workflow}"

        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps or self.bad_steps >= self.max_bad_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward

        return observation, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()
        cur_task = self.task.observation
        if self.args.incorporation_type == "query":
            observation, messages = prompt_with_icl(
                instruction=self.instruction, 
                raw_icl=self.raw_icl[self.task.task_type], 
                cur_task=cur_task, 
                icl_num=1,
                workflow=self.task.workflow,
            )
        else:
            observation, messages = prompt_with_icl(
                instruction=self.instruction, 
                raw_icl=self.raw_icl[self.task.task_type], 
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
