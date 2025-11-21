"""
Base Agent Class
Foundation for all trading agents with structured communication
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import json

from trading_agents.core.llm import BaseLLM


class BaseAgent(ABC):
    """
    Base class for all trading agents
    Defines interface and common functionality
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm: BaseLLM,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize agent

        Args:
            name: Agent name
            role: Agent role description
            llm: LLM instance
            temperature: Override LLM temperature
            max_tokens: Override max tokens
        """
        self.name = name
        self.role = role
        self.llm = llm

        # Override LLM settings if provided
        if temperature is not None:
            self.llm.temperature = temperature
        if max_tokens is not None:
            self.llm.max_tokens = max_tokens

        # Communication history
        self.message_history = []

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process information and generate output

        Args:
            context: Input context dictionary

        Returns:
            Output dictionary with agent's analysis/decision
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get agent-specific system prompt

        Returns:
            System prompt string
        """
        pass

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build prompt from context
        To be implemented by subclasses

        Args:
            context: Input context

        Returns:
            Formatted prompt string
        """
        return json.dumps(context, indent=2)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response
        Default implementation returns as-is

        Args:
            response: LLM response string

        Returns:
            Parsed response dictionary
        """
        return {"response": response}

    def generate_response(self, context: Dict[str, Any], use_json: bool = True) -> Dict[str, Any]:
        """
        Generate response using LLM

        Args:
            context: Input context
            use_json: Whether to request JSON output

        Returns:
            Agent response dictionary
        """
        system_prompt = self.get_system_prompt()
        prompt = self._build_prompt(context)

        # Log message
        self.message_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "input",
            "content": context
        })

        # Generate response
        if use_json:
            response = self.llm.generate_json(prompt, system_prompt)
        else:
            response_text = self.llm.generate(prompt, system_prompt)
            response = self._parse_response(response_text)

        # Add metadata
        response['agent_name'] = self.name
        response['agent_role'] = self.role
        response['timestamp'] = datetime.now().isoformat()

        # Log response
        self.message_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "output",
            "content": response
        })

        return response

    def get_message_history(self) -> list:
        """Get agent's message history"""
        return self.message_history

    def clear_history(self):
        """Clear message history"""
        self.message_history = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', role='{self.role}')"
