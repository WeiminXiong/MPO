import logging
import os

import backoff
import openai
from openai import OpenAI

from .base import BaseAgent

logger = logging.getLogger("agent_eval")


class OpenAIAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        self.client = OpenAI(
            base_url=config.get("api_base", None),
            api_key=config.get("api_key", os.environ.get("OPENAI_API_KEY")),
        )

    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.APIError,
            openai.Timeout,
            openai.RateLimitError,
            openai.APIConnectionError,
        ),
    )
    def __call__(self, messages) -> str:
        # Prepend the prompt with the system message
        response = self.client.chat.completions.create(
            model=self.config["model_name"],
            messages=messages,
            max_completion_tokens=self.config.get("max_completion_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
        )
        return response.choices[0].message.content
    
    def call_with_workflow(self, messages) -> str:
        # Prepend the prompt with the system message
        new_messages = messages + [{"role": "assistant", "content": self.workflow + "Thought: "}]
        response = self.client.chat.completions.create(
            model=self.config["model_name"],
            messages=new_messages,
            max_completion_tokens=self.config.get("max_completion_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False
            }
        )
        return response.choices[0].message.content
