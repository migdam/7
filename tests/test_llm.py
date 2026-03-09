"""Tests for trading_agents.core.llm"""

import pytest
import json
from trading_agents.core.llm import BaseLLM, LLMFactory, LLMProvider, get_llm


class TestExtractJsonFromText:

    def test_extracts_clean_json(self):
        text = '{"key": "value"}'
        result = BaseLLM._extract_json_from_text(text)
        assert result == {"key": "value"}

    def test_extracts_json_from_surrounding_text(self):
        text = 'Here is the result: {"action": "BUY", "size": 5} end.'
        result = BaseLLM._extract_json_from_text(text)
        assert result is not None
        assert result['action'] == 'BUY'

    def test_extracts_nested_json(self):
        obj = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        text = f"Response: {json.dumps(obj)}"
        result = BaseLLM._extract_json_from_text(text)
        assert result is not None
        assert 'outer' in result

    def test_returns_none_for_no_json(self):
        text = "This is just plain text with no JSON"
        result = BaseLLM._extract_json_from_text(text)
        assert result is None

    def test_returns_none_for_invalid_json(self):
        text = "{this is not valid json}"
        result = BaseLLM._extract_json_from_text(text)
        assert result is None


class TestRetryCall:

    def test_succeeds_on_first_try(self):
        # Create a concrete subclass for testing
        class TestLLM(BaseLLM):
            def generate(self, prompt, system_prompt=None):
                return "test"
            def generate_json(self, prompt, system_prompt=None):
                return {}

        llm = TestLLM(model="test")
        result = llm._retry_call(lambda: "success", max_retries=3)
        assert result == "success"

    def test_retries_on_failure(self):
        class TestLLM(BaseLLM):
            def generate(self, prompt, system_prompt=None):
                return "test"
            def generate_json(self, prompt, system_prompt=None):
                return {}

        llm = TestLLM(model="test")
        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("fail")
            return "success"

        result = llm._retry_call(failing_fn, max_retries=3)
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        class TestLLM(BaseLLM):
            def generate(self, prompt, system_prompt=None):
                return "test"
            def generate_json(self, prompt, system_prompt=None):
                return {}

        llm = TestLLM(model="test")

        with pytest.raises(Exception, match="always fail"):
            llm._retry_call(lambda: (_ for _ in ()).throw(Exception("always fail")),
                           max_retries=0)


class TestLLMFactory:

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            LLMProvider("nonexistent")

    def test_get_llm_creates_provider(self):
        # This will fail at runtime (no API key) but tests the factory path
        with pytest.raises((ImportError, Exception)):
            get_llm(provider_name="openai", model="gpt-4o-mini")
