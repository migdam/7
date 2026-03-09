"""
LLM Interface Layer
Supports multiple LLM providers: OpenAI, Anthropic, local models via ollama
"""

import os
import re
import json
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE = "azure"


class BaseLLM(ABC):
    """Base class for LLM providers"""

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured JSON response"""
        pass

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from text that may contain non-JSON content."""
        # Try non-greedy match first (innermost object)
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: greedy match (outermost object)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _retry_call(self, fn, max_retries: int = 3):
        """Retry a callable with exponential backoff."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        raise last_error


class OpenAILLM(BaseLLM):
    """OpenAI API implementation"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 1000):
        super().__init__(model, temperature, max_tokens)
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        response = self._retry_call(_call)
        return response.choices[0].message.content

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured JSON using OpenAI API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

        response = self._retry_call(_call)
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            extracted = self._extract_json_from_text(content)
            if extracted is not None:
                return extracted
            return {"response": content, "error": "Failed to parse JSON"}


class AnthropicLLM(BaseLLM):
    """Anthropic Claude API implementation"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.7, max_tokens: int = 1000):
        super().__init__(model, temperature, max_tokens)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Anthropic API"""
        def _call():
            return self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )

        message = self._retry_call(_call)
        return message.content[0].text

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured JSON using Anthropic API"""
        json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."
        full_prompt = prompt + json_instruction

        response = self.generate(full_prompt, system_prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            extracted = self._extract_json_from_text(response)
            if extracted is not None:
                return extracted
            return {"response": response, "error": "Failed to parse JSON"}


class OllamaLLM(BaseLLM):
    """Local Ollama implementation"""

    def __init__(self, model: str = "llama3.1", temperature: float = 0.7, max_tokens: int = 1000, base_url: str = "http://localhost:11434"):
        super().__init__(model, temperature, max_tokens)
        self.base_url = base_url
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests package not installed. Install with: pip install requests")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama API"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        def _call():
            return self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
            )

        response = self._retry_call(_call)

        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama API error: {response.status_code}")

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate structured JSON using Ollama"""
        json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."
        full_prompt = prompt + json_instruction

        response = self.generate(full_prompt, system_prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            extracted = self._extract_json_from_text(response)
            if extracted is not None:
                return extracted
            return {"response": response, "error": "Failed to parse JSON"}


class LLMFactory:
    """Factory for creating LLM instances"""

    @staticmethod
    def create(
        provider: LLMProvider,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance based on provider

        Args:
            provider: LLM provider to use
            model: Model name (optional, uses defaults)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments

        Returns:
            BaseLLM instance
        """
        if provider == LLMProvider.OPENAI:
            return OpenAILLM(
                model=model or "gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicLLM(
                model=model or "claude-3-5-sonnet-20241022",
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == LLMProvider.OLLAMA:
            return OllamaLLM(
                model=model or "llama3.1",
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=kwargs.get("base_url", "http://localhost:11434")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")


# Convenience function for quick access
def get_llm(
    provider_name: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> BaseLLM:
    """
    Convenience function to get an LLM instance

    Args:
        provider_name: Name of provider ("openai", "anthropic", "ollama")
        model: Model name (optional)
        temperature: Generation temperature
        max_tokens: Maximum tokens
        **kwargs: Additional arguments

    Returns:
        BaseLLM instance
    """
    provider = LLMProvider(provider_name.lower())
    return LLMFactory.create(provider, model, temperature, max_tokens, **kwargs)
