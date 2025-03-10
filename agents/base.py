import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping

logger = logging.getLogger("agent_eval")


class BaseAgent(ABC):
    """Base class for an agent."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        logger.debug(f"Initialized {self.__class__.__name__} with config: {config}")
        # The agent should not generate observations or expert feedback
        self.stop_words = ["\nObservation:", "\nTask:", "\n---"]

    @abstractmethod
    def __call__(self) -> str:
        pass

    def add_system_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        # Prepend the prompt with the system message
        first_msg = messages[0]
        assert first_msg["role"] == "user"
        system, examples, task = first_msg["content"].split("\n---\n")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": examples + "\n---\n" + task},
        ] + messages[1:]
        return messages

    def set_workflow(self, workflow: str) -> None:
        self.workflow = workflow
    
