"""
Tests for chatbot implementations.

This module tests the refactored chatbot architecture including:
- Factory function routing
- API key retrieval
- Tool result truncation
- Model-specific configurations
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Test model name constants - Provider prefixes
LLAMA_PROVIDER = "meta-llama"
ANTHROPIC_PROVIDER = "anthropic"
OPENAI_PROVIDER = "openai"
GOOGLE_PROVIDER = "google"

# Llama models
LLAMA_3_1_8B = f"{LLAMA_PROVIDER}/Llama-3.1-8B-Instruct"
LLAMA_3_2_3B = f"{LLAMA_PROVIDER}/Llama-3.2-3B-Instruct"
LLAMA_3_3_70B = f"{LLAMA_PROVIDER}/Llama-3.3-70B-Instruct"

# Claude models
CLAUDE_HAIKU = "claude-3-5-haiku"
CLAUDE_HAIKU_WITH_PROVIDER = f"{ANTHROPIC_PROVIDER}/{CLAUDE_HAIKU}"
CLAUDE_SONNET = "claude-sonnet-4-20250514"
CLAUDE_SONNET_WITH_PROVIDER = f"{ANTHROPIC_PROVIDER}/{CLAUDE_SONNET}"
CLAUDE_HAIKU_DATED = "claude-3-5-haiku-20241022"

# OpenAI models
GPT_4O_MINI = "gpt-4o-mini"
GPT_4O_MINI_WITH_PROVIDER = f"{OPENAI_PROVIDER}/{GPT_4O_MINI}"
GPT_4O = "gpt-4o"

# Google models
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_FLASH_WITH_PROVIDER = f"{GOOGLE_PROVIDER}/{GEMINI_FLASH}"
GEMINI_FLASH_EXP = "gemini-2.0-flash-exp"
GEMINI_FLASH_EXP_WITH_PROVIDER = f"{GOOGLE_PROVIDER}/{GEMINI_FLASH_EXP}"


@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor for testing (without Korrel8r)."""
    import asyncio

    executor = Mock()

    # Define the tools list
    tools_list = [
        {
            "name": "search_metrics",
            "description": "Search for metrics",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "execute_promql",
            "description": "Execute PromQL query",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "get_metric_metadata",
            "description": "Get metric metadata",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "get_label_values",
            "description": "Get label values",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "suggest_queries",
            "description": "Suggest PromQL queries",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "explain_results",
            "description": "Explain query results",
            "input_schema": {"type": "object", "properties": {}}
        }
    ]

    # Mock get_tools to return a NEW coroutine each time it's called
    async def mock_get_tools():
        return tools_list

    # Use side_effect to create a new coroutine on each call
    executor.get_tools.side_effect = lambda: mock_get_tools()
    executor.execute_tool.return_value = "Mock tool result"
    return executor


@pytest.fixture
def mock_tool_executor_with_korrel8r():
    """Create a mock tool executor with Korrel8r tool enabled."""
    import asyncio

    executor = Mock()

    # Define the tools list
    tools_list = [
        {
            "name": "search_metrics",
            "description": "Search for metrics",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "execute_promql",
            "description": "Execute PromQL query",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "get_metric_metadata",
            "description": "Get metric metadata",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "get_label_values",
            "description": "Get label values",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "suggest_queries",
            "description": "Suggest PromQL queries",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "explain_results",
            "description": "Explain query results",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "korrel8r_get_correlated",
            "description": "Get correlated observability data using Korrel8r",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Korrel8r query (e.g., 'alert:alert:{alertname=\"HighCPU\"}')"
                    }
                },
                "required": ["query"]
            }
        }
    ]

    # Mock get_tools to return a NEW coroutine each time it's called
    async def mock_get_tools():
        return tools_list

    # Use side_effect to create a new coroutine on each call
    executor.get_tools.side_effect = lambda: mock_get_tools()
    executor.execute_tool.return_value = "Mock korrel8r result"
    return executor


def test_chatbot_imports():
    """Test that all chatbot classes can be imported."""
    from mcp_server.chatbots import (
        BaseChatBot,
        AnthropicChatBot,
        OpenAIChatBot,
        GoogleChatBot,
        LlamaChatBot,
        DeterministicChatBot,
        create_chatbot
    )

    assert BaseChatBot is not None
    assert AnthropicChatBot is not None
    assert OpenAIChatBot is not None
    assert GoogleChatBot is not None
    assert LlamaChatBot is not None
    assert DeterministicChatBot is not None
    assert create_chatbot is not None


def test_factory_creates_llama_bot(mock_tool_executor):
    """Test that factory creates LlamaChatBot for Llama 3.1 models."""
    from mcp_server.chatbots import create_chatbot, LlamaChatBot

    bot = create_chatbot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
    assert isinstance(bot, LlamaChatBot)
    assert bot.model_name == LLAMA_3_1_8B


def test_factory_creates_deterministic_bot(mock_tool_executor):
    """Test that factory creates DeterministicChatBot for Llama 3.2 models."""
    from mcp_server.chatbots import create_chatbot, DeterministicChatBot

    bot = create_chatbot(LLAMA_3_2_3B, tool_executor=mock_tool_executor)
    assert isinstance(bot, DeterministicChatBot)
    assert bot.model_name == LLAMA_3_2_3B


def test_factory_creates_anthropic_bot(mock_tool_executor):
    """Test that factory creates AnthropicChatBot for Anthropic models."""
    from mcp_server.chatbots import create_chatbot, AnthropicChatBot

    # Factory determines bot type based on model name patterns
    bot = create_chatbot(CLAUDE_HAIKU_WITH_PROVIDER, api_key="test-key", tool_executor=mock_tool_executor)
    assert isinstance(bot, AnthropicChatBot)


def test_factory_creates_openai_bot(mock_tool_executor):
    """Test that factory creates OpenAIChatBot for OpenAI models."""
    from mcp_server.chatbots import create_chatbot, OpenAIChatBot

    # Factory determines bot type based on model name patterns
    bot = create_chatbot(GPT_4O_MINI_WITH_PROVIDER, api_key="test-key", tool_executor=mock_tool_executor)
    assert isinstance(bot, OpenAIChatBot)


def test_factory_creates_google_bot(mock_tool_executor):
    """Test that factory creates GoogleChatBot for Google models."""
    from mcp_server.chatbots import create_chatbot, GoogleChatBot

    # Factory determines bot type based on model name patterns
    bot = create_chatbot(GEMINI_FLASH_EXP_WITH_PROVIDER, api_key="test-key", tool_executor=mock_tool_executor)
    assert isinstance(bot, GoogleChatBot)


class TestAPIKeyRetrieval:
    """Test API key retrieval for all bot types."""

    def test_anthropic_bot_api_key_from_env(self, mock_tool_executor):
        """Test AnthropicChatBot gets API key from environment."""
        from mcp_server.chatbots import AnthropicChatBot

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-anthropic-key'}):
            bot = AnthropicChatBot(CLAUDE_HAIKU, tool_executor=mock_tool_executor)
            assert bot._get_api_key() == 'test-anthropic-key'
            assert bot.api_key == 'test-anthropic-key'

    def test_openai_bot_api_key_from_env(self, mock_tool_executor):
        """Test OpenAIChatBot gets API key from environment."""
        from mcp_server.chatbots import OpenAIChatBot

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'}):
            bot = OpenAIChatBot(GPT_4O_MINI, tool_executor=mock_tool_executor)
            assert bot._get_api_key() == 'test-openai-key'
            assert bot.api_key == 'test-openai-key'

    def test_google_bot_api_key_from_env(self, mock_tool_executor):
        """Test GoogleChatBot gets API key from environment."""
        from mcp_server.chatbots import GoogleChatBot

        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-google-key'}):
            bot = GoogleChatBot(GEMINI_FLASH, tool_executor=mock_tool_executor)
            assert bot._get_api_key() == 'test-google-key'
            assert bot.api_key == 'test-google-key'

    def test_llama_bot_no_api_key_needed(self, mock_tool_executor):
        """Test LlamaChatBot returns None for API key (local model)."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
        assert bot._get_api_key() is None
        assert bot.api_key is None

    def test_deterministic_bot_no_api_key_needed(self, mock_tool_executor):
        """Test DeterministicChatBot returns None for API key (local model)."""
        from mcp_server.chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_tool_executor)
        assert bot._get_api_key() is None
        assert bot.api_key is None

    def test_explicit_api_key_overrides_env(self, mock_tool_executor):
        """Test that explicitly passed API key overrides environment variable."""
        from mcp_server.chatbots import OpenAIChatBot

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            bot = OpenAIChatBot(GPT_4O_MINI, api_key="explicit-key", tool_executor=mock_tool_executor)
            assert bot.api_key == "explicit-key"

    def test_openai_bot_can_be_created_without_api_key(self, mock_tool_executor):
        """Test that OpenAIChatBot can be initialized without an API key."""
        from mcp_server.chatbots import OpenAIChatBot

        # Clear any environment variables
        with patch.dict(os.environ, {}, clear=True):
            bot = OpenAIChatBot(GPT_4O_MINI, tool_executor=mock_tool_executor)
            assert bot.api_key is None
            assert bot.client is None  # Client should not be created without API key

    def test_openai_bot_with_api_key_creates_client(self, mock_tool_executor):
        """Test that OpenAIChatBot creates client when API key is provided."""
        from mcp_server.chatbots import OpenAIChatBot

        with patch('openai.OpenAI') as mock_openai_class:
            bot = OpenAIChatBot(GPT_4O_MINI, api_key="test-key", tool_executor=mock_tool_executor)
            assert bot.api_key == "test-key"
            # Verify OpenAI client was instantiated with the API key
            mock_openai_class.assert_called_once_with(api_key="test-key")

    def test_openai_bot_without_api_key_does_not_create_client(self, mock_tool_executor):
        """Test that OpenAIChatBot does not create client when no API key is provided."""
        from mcp_server.chatbots import OpenAIChatBot

        with patch('openai.OpenAI') as mock_openai_class:
            with patch.dict(os.environ, {}, clear=True):
                bot = OpenAIChatBot(GPT_4O_MINI, tool_executor=mock_tool_executor)
                assert bot.api_key is None
                assert bot.client is None
                # Verify OpenAI client was NOT instantiated
                mock_openai_class.assert_not_called()


class TestToolResultTruncation:
    """Test tool result truncation for all bot types."""

    def test_anthropic_bot_max_length(self, mock_tool_executor):
        """Test AnthropicChatBot has correct max length (15K)."""
        from mcp_server.chatbots import AnthropicChatBot

        bot = AnthropicChatBot(CLAUDE_HAIKU, api_key="test", tool_executor=mock_tool_executor)
        assert bot._get_max_tool_result_length() == 15000

    def test_openai_bot_max_length(self, mock_tool_executor):
        """Test OpenAIChatBot has correct max length (10K)."""
        from mcp_server.chatbots import OpenAIChatBot

        bot = OpenAIChatBot(GPT_4O_MINI, api_key="test", tool_executor=mock_tool_executor)
        assert bot._get_max_tool_result_length() == 10000

    def test_google_bot_max_length(self, mock_tool_executor):
        """Test GoogleChatBot has correct max length (10K)."""
        from mcp_server.chatbots import GoogleChatBot

        bot = GoogleChatBot(GEMINI_FLASH, api_key="test", tool_executor=mock_tool_executor)
        assert bot._get_max_tool_result_length() == 10000

    def test_llama_bot_max_length(self, mock_tool_executor):
        """Test LlamaChatBot has correct max length (8K)."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
        assert bot._get_max_tool_result_length() == 8000

    def test_deterministic_bot_uses_base_max_length(self, mock_tool_executor):
        """Test DeterministicChatBot uses base class default (5K)."""
        from mcp_server.chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_tool_executor)
        assert bot._get_max_tool_result_length() == 5000

    def test_get_tool_result_truncates_large_results(self, mock_tool_executor):
        """Test that _get_tool_result properly truncates results exceeding max length."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Mock _route_tool_call_to_mcp to return a large result
        large_result = "x" * 10000  # 10K chars, exceeds Llama's 8K limit
        with patch.object(bot, '_route_tool_call_to_mcp', return_value=large_result):
            result = bot._get_tool_result("test_tool", {"arg": "value"})

            # Should be truncated to 8000 + truncation message
            assert len(result) == 8000 + len("\n... [Result truncated due to size]")
            assert result.endswith("\n... [Result truncated due to size]")
            assert result.startswith("x" * 100)  # Verify it starts with the original content

    def test_get_tool_result_does_not_truncate_small_results(self, mock_tool_executor):
        """Test that _get_tool_result doesn't truncate results within max length."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Mock _route_tool_call_to_mcp to return a small result
        small_result = "Small result"
        with patch.object(bot, '_route_tool_call_to_mcp', return_value=small_result):
            result = bot._get_tool_result("test_tool", {"arg": "value"})

            # Should NOT be truncated
            assert result == small_result
            assert "truncated" not in result.lower()

    def test_get_tool_result_calls_route_with_correct_args(self, mock_tool_executor):
        """Test that _get_tool_result calls _route_tool_call_to_mcp with correct arguments."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        with patch.object(bot, '_route_tool_call_to_mcp', return_value="result") as mock_route:
            tool_name = "execute_promql"
            tool_args = {"query": "up"}

            bot._get_tool_result(tool_name, tool_args)

            # Verify the method was called with correct args
            mock_route.assert_called_once_with(tool_name, tool_args)


class TestModelSpecificInstructions:
    """Test that each bot has model-specific instructions."""

    def test_anthropic_bot_has_specific_instructions(self, mock_tool_executor):
        """Test AnthropicChatBot has Claude-specific instructions."""
        from mcp_server.chatbots import AnthropicChatBot

        bot = AnthropicChatBot(CLAUDE_HAIKU, api_key="test", tool_executor=mock_tool_executor)
        instructions = bot._get_model_specific_instructions()

        assert "CLAUDE-SPECIFIC" in instructions
        assert len(instructions) > 0

    def test_openai_bot_has_specific_instructions(self, mock_tool_executor):
        """Test OpenAIChatBot has GPT-specific instructions."""
        from mcp_server.chatbots import OpenAIChatBot

        bot = OpenAIChatBot(GPT_4O_MINI, api_key="test", tool_executor=mock_tool_executor)
        instructions = bot._get_model_specific_instructions()

        assert "GPT-SPECIFIC" in instructions
        assert len(instructions) > 0

    def test_google_bot_has_specific_instructions(self, mock_tool_executor):
        """Test GoogleChatBot has Gemini-specific instructions."""
        from mcp_server.chatbots import GoogleChatBot

        bot = GoogleChatBot(GEMINI_FLASH, api_key="test", tool_executor=mock_tool_executor)
        instructions = bot._get_model_specific_instructions()

        assert "GEMINI-SPECIFIC" in instructions
        assert len(instructions) > 0

    def test_llama_bot_has_specific_instructions(self, mock_tool_executor):
        """Test LlamaChatBot has Llama-specific instructions."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
        instructions = bot._get_model_specific_instructions()

        assert "LLAMA-SPECIFIC" in instructions
        assert "Tool Calling Format" in instructions
        assert "PromQL Query Patterns" in instructions
        assert "Key PromQL Rules" in instructions


class TestModelNameExtraction:
    """Test model name extraction functionality."""

    def test_anthropic_extracts_model_name_with_provider(self, mock_tool_executor):
        """Test Anthropic bot extracts model name from provider/model format."""
        from mcp_server.chatbots import AnthropicChatBot

        bot = AnthropicChatBot(CLAUDE_SONNET_WITH_PROVIDER, api_key="test", tool_executor=mock_tool_executor)
        extracted = bot._extract_model_name()

        assert extracted == CLAUDE_SONNET

    def test_anthropic_keeps_model_name_without_provider(self, mock_tool_executor):
        """Test Anthropic bot keeps model name when no provider prefix."""
        from mcp_server.chatbots import AnthropicChatBot

        bot = AnthropicChatBot(CLAUDE_HAIKU_DATED, api_key="test", tool_executor=mock_tool_executor)
        extracted = bot._extract_model_name()

        assert extracted == CLAUDE_HAIKU_DATED

    def test_openai_extracts_model_name_with_provider(self, mock_tool_executor):
        """Test OpenAI bot extracts model name from provider/model format."""
        from mcp_server.chatbots import OpenAIChatBot

        bot = OpenAIChatBot(GPT_4O_MINI_WITH_PROVIDER, api_key="test", tool_executor=mock_tool_executor)
        extracted = bot._extract_model_name()

        assert extracted == GPT_4O_MINI

    def test_openai_keeps_model_name_without_provider(self, mock_tool_executor):
        """Test OpenAI bot keeps model name when no provider prefix."""
        from mcp_server.chatbots import OpenAIChatBot

        bot = OpenAIChatBot(GPT_4O, api_key="test", tool_executor=mock_tool_executor)
        extracted = bot._extract_model_name()

        assert extracted == GPT_4O

    def test_google_extracts_model_name_with_provider(self, mock_tool_executor):
        """Test Google bot extracts model name from provider/model format."""
        from mcp_server.chatbots import GoogleChatBot

        bot = GoogleChatBot(GEMINI_FLASH_EXP_WITH_PROVIDER, api_key="test", tool_executor=mock_tool_executor)
        extracted = bot._extract_model_name()

        assert extracted == GEMINI_FLASH_EXP

    def test_google_keeps_model_name_without_provider(self, mock_tool_executor):
        """Test Google bot keeps model name when no provider prefix."""
        from mcp_server.chatbots import GoogleChatBot

        bot = GoogleChatBot(GEMINI_FLASH, api_key="test", tool_executor=mock_tool_executor)
        extracted = bot._extract_model_name()

        assert extracted == GEMINI_FLASH

    def test_llama_uses_full_model_name(self, mock_tool_executor):
        """Test Llama bot uses full model name (doesn't strip provider for local models)."""
        from mcp_server.chatbots import LlamaChatBot

        # Llama uses the full model name including provider as it may be needed for local model paths
        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
        extracted = bot._extract_model_name()

        # For local models like Llama, the full path is preserved
        assert extracted == LLAMA_3_1_8B

    def test_base_extraction_with_slash(self, mock_tool_executor):
        """Test base class extraction splits on first slash for API models."""
        from mcp_server.chatbots import AnthropicChatBot

        # API-based models strip the provider prefix
        bot = AnthropicChatBot(CLAUDE_SONNET_WITH_PROVIDER, api_key="test", tool_executor=mock_tool_executor)
        extracted = bot._extract_model_name()

        assert extracted == CLAUDE_SONNET

    def test_extraction_preserves_original_model_name(self, mock_tool_executor):
        """Test that original model_name attribute is preserved."""
        from mcp_server.chatbots import AnthropicChatBot

        original_name = CLAUDE_SONNET_WITH_PROVIDER
        bot = AnthropicChatBot(original_name, api_key="test", tool_executor=mock_tool_executor)

        # Original should be preserved
        assert bot.model_name == original_name

        # Extracted should be without provider
        assert bot._extract_model_name() == CLAUDE_SONNET


class TestBaseChatBot:
    """Test BaseChatBot common functionality."""

    def test_base_chatbot_is_abstract(self, mock_tool_executor):
        """Test that BaseChatBot cannot be instantiated directly."""
        from mcp_server.chatbots.base import BaseChatBot

        # BaseChatBot is abstract and should raise TypeError
        with pytest.raises(TypeError):
            BaseChatBot("test-model")

    def test_tool_executor_is_required(self, mock_tool_executor):
        """Test that tool_executor is required for all chatbots."""
        from mcp_server.chatbots import LlamaChatBot

        # Creating chatbot without tool_executor should raise ValueError
        with pytest.raises(ValueError, match="tool_executor is required"):
            LlamaChatBot(LLAMA_3_1_8B)

    def test_factory_requires_tool_executor(self, mock_tool_executor):
        """Test that factory function requires tool_executor."""
        from mcp_server.chatbots import create_chatbot

        # Creating chatbot via factory without tool_executor should raise ValueError
        with pytest.raises(ValueError, match="tool_executor is required"):
            create_chatbot(LLAMA_3_1_8B)

    def test_get_mcp_tools_returns_list(self, mock_tool_executor):
        """Test that _get_mcp_tools returns a list of tool definitions."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
        tools = bot._get_mcp_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        # Check that tools have expected structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_create_system_prompt_includes_model_specific(self, mock_tool_executor):
        """Test that system prompt includes model-specific instructions."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
        prompt = bot._create_system_prompt(namespace="test-namespace")

        # Should include both base prompt and model-specific instructions
        assert "Kubernetes and Prometheus" in prompt  # Base prompt
        assert "LLAMA-SPECIFIC" in prompt  # Model-specific


class TestKorrel8rNormalization:
    """Test Korrel8r query normalization functionality in BaseChatBot."""

    def test_normalize_alert_query_missing_class(self, mock_tool_executor):
        """Test that alert queries without class get 'alert:alert:' prefix."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Missing class - should be normalized
        query = 'alert:{"alertname":"PodDisruptionBudgetAtLimit"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"PodDisruptionBudgetAtLimit"}'

    def test_normalize_alert_query_already_correct(self, mock_tool_executor):
        """Test that correctly formatted alert queries are not changed."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Already correct - should not change
        query = 'alert:alert:{"alertname":"HighCPU"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"HighCPU"}'

    def test_normalize_escaped_quotes(self, mock_tool_executor):
        """Test that escaped quotes are unescaped."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Escaped quotes should be unescaped
        query = 'alert:{\"alertname\":\"Test\"}'
        normalized = bot._normalize_korrel8r_query(query)

        # Should unescape quotes AND add missing class
        assert normalized == 'alert:alert:{"alertname":"Test"}'

    def test_normalize_k8s_alert_misclassification(self, mock_tool_executor):
        """Test that k8s:Alert: is corrected to alert:alert:."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Misclassified as k8s - should be corrected
        query = 'k8s:Alert:{"alertname":"PodDown"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"PodDown"}'

    def test_normalize_alert_unquoted_keys(self, mock_tool_executor):
        """Test that unquoted keys in alert selectors are quoted (JSON format)."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Unquoted key - should be quoted for alert domain
        query = 'alert:alert:{alertname="HighLatency"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"HighLatency"}'

    def test_normalize_alert_multiple_unquoted_keys(self, mock_tool_executor):
        """Test normalization with multiple unquoted keys."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Multiple unquoted keys
        query = 'alert:alert:{alertname="Test",severity="critical"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"Test","severity":"critical"}'

    def test_normalize_k8s_pod_query(self, mock_tool_executor):
        """Test normalization of k8s Pod queries (non-alert domain)."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # k8s domain uses := operator format
        query = 'k8s:Pod:{namespace="llm-serving"}'
        normalized = bot._normalize_korrel8r_query(query)

        # For non-alert domains, should use := operator
        assert normalized == 'k8s:Pod:{"namespace":="llm-serving"}'

    def test_normalize_loki_log_query(self, mock_tool_executor):
        """Test normalization of loki log queries."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Loki domain
        query = 'loki:log:{kubernetes.namespace_name="test"}'
        normalized = bot._normalize_korrel8r_query(query)

        # Should use := for non-alert domains
        assert 'kubernetes.namespace_name":=' in normalized

    def test_normalize_trace_span_query(self, mock_tool_executor):
        """Test normalization of trace span queries."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Trace domain - dots in key names need special handling
        query = 'trace:span:{k8s_namespace_name="llm-serving"}'
        normalized = bot._normalize_korrel8r_query(query)

        # Should use := for non-alert domains
        assert 'k8s_namespace_name":=' in normalized

    def test_normalize_empty_query(self, mock_tool_executor):
        """Test that empty queries are handled gracefully."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Empty query
        normalized = bot._normalize_korrel8r_query("")
        assert normalized == ""

    def test_normalize_none_query(self, mock_tool_executor):
        """Test that None queries are handled gracefully."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # None query - implementation converts to empty string
        normalized = bot._normalize_korrel8r_query(None)
        assert normalized == ""

    def test_normalize_malformed_query_doesnt_crash(self, mock_tool_executor):
        """Test that malformed queries don't crash the normalization."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Malformed query - should return original on error
        query = 'totally:invalid{{'
        normalized = bot._normalize_korrel8r_query(query)

        # Should return something (either original or partially normalized)
        assert normalized is not None

    def test_normalize_works_for_all_bot_types(self, mock_tool_executor):
        """Test that normalization is available to all chatbot types."""
        from mcp_server.chatbots import (
            LlamaChatBot,
            AnthropicChatBot,
            OpenAIChatBot,
            GoogleChatBot
        )

        query = 'alert:{"alertname":"Test"}'
        expected = 'alert:alert:{"alertname":"Test"}'

        # Test each bot type
        llama_bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
        assert llama_bot._normalize_korrel8r_query(query) == expected

        anthropic_bot = AnthropicChatBot(CLAUDE_HAIKU, api_key="test", tool_executor=mock_tool_executor)
        assert anthropic_bot._normalize_korrel8r_query(query) == expected

        openai_bot = OpenAIChatBot(GPT_4O_MINI, api_key="test", tool_executor=mock_tool_executor)
        assert openai_bot._normalize_korrel8r_query(query) == expected

        google_bot = GoogleChatBot(GEMINI_FLASH, api_key="test", tool_executor=mock_tool_executor)
        assert google_bot._normalize_korrel8r_query(query) == expected


class TestKorrel8rToolIntegration:
    """Test Korrel8r tool integration in routing."""

    def test_normalize_is_called_for_korrel8r_queries(self, mock_tool_executor):
        """Test that normalization is invoked for korrel8r queries."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)

        # Test that normalize method works correctly
        query = 'alert:{"alertname":"Test"}'
        normalized = bot._normalize_korrel8r_query(query)

        # Should be normalized
        assert normalized == 'alert:alert:{"alertname":"Test"}'

    def test_korrel8r_tool_not_in_base_tools_by_default(self, mock_tool_executor):
        """Test that korrel8r tool is NOT in tools when executor doesn't provide it."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor)
        tools = bot._get_mcp_tools()

        # Check if korrel8r tool is NOT in the list (mock doesn't include it)
        tool_names = [tool["name"] for tool in tools]

        # Verify base tools are present
        assert isinstance(tools, list)
        assert len(tools) == 6  # Base tools only
        assert "search_metrics" in tool_names
        assert "execute_promql" in tool_names
        assert "korrel8r_get_correlated" not in tool_names

    def test_korrel8r_tool_included_when_enabled(self, mock_tool_executor_with_korrel8r):
        """Test that korrel8r tool is included when MCP server provides it."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor_with_korrel8r)
        tools = bot._get_mcp_tools()

        # Check if korrel8r tool IS in the list
        tool_names = [tool["name"] for tool in tools]

        assert isinstance(tools, list)
        assert len(tools) == 7  # Base tools + Korrel8r
        assert "korrel8r_get_correlated" in tool_names

        # Verify the Korrel8r tool has correct structure
        korrel8r_tool = next(t for t in tools if t["name"] == "korrel8r_get_correlated")
        assert "description" in korrel8r_tool
        assert "input_schema" in korrel8r_tool
        assert "query" in korrel8r_tool["input_schema"]["properties"]

    def test_korrel8r_query_normalization_during_routing(self, mock_tool_executor_with_korrel8r):
        """Test that Korrel8r queries are normalized when routed to MCP."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor_with_korrel8r)

        # Mock the tool executor's execute_tool method
        with patch.object(mock_tool_executor_with_korrel8r, 'execute_tool', return_value="Korrel8r result") as mock_execute:
            # Call with a query that needs normalization
            bad_query = 'alert:{"alertname":"HighCPU"}'
            result = bot._route_tool_call_to_mcp("korrel8r_get_correlated", {"query": bad_query})

            # Verify execute_tool was called with normalized query
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[0][0] == "korrel8r_get_correlated"  # tool name

            # Check that the query was normalized
            actual_query = call_args[0][1]["query"]
            expected_query = 'alert:alert:{"alertname":"HighCPU"}'
            assert actual_query == expected_query

    def test_korrel8r_tool_works_with_all_bot_types(self, mock_tool_executor_with_korrel8r):
        """Test that Korrel8r tool integration works with all chatbot types."""
        from mcp_server.chatbots import (
            LlamaChatBot,
            AnthropicChatBot,
            OpenAIChatBot,
            GoogleChatBot,
            DeterministicChatBot
        )

        bots = [
            LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor_with_korrel8r),
            AnthropicChatBot(CLAUDE_HAIKU, api_key="test", tool_executor=mock_tool_executor_with_korrel8r),
            OpenAIChatBot(GPT_4O_MINI, api_key="test", tool_executor=mock_tool_executor_with_korrel8r),
            GoogleChatBot(GEMINI_FLASH, api_key="test", tool_executor=mock_tool_executor_with_korrel8r),
            DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_tool_executor_with_korrel8r)
        ]

        # All bots should have access to Korrel8r tool
        for bot in bots:
            tools = bot._get_mcp_tools()
            tool_names = [tool["name"] for tool in tools]
            assert "korrel8r_get_correlated" in tool_names, f"{bot.__class__.__name__} missing Korrel8r tool"

    def test_normalization_available_to_all_bots(self, mock_tool_executor):
        """Test that normalization method is available to all bot types."""
        from mcp_server.chatbots import (
            LlamaChatBot,
            AnthropicChatBot,
            OpenAIChatBot,
            GoogleChatBot,
            DeterministicChatBot
        )

        bots = [
            LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor),
            AnthropicChatBot(CLAUDE_HAIKU, api_key="test", tool_executor=mock_tool_executor),
            OpenAIChatBot(GPT_4O_MINI, api_key="test", tool_executor=mock_tool_executor),
            GoogleChatBot(GEMINI_FLASH, api_key="test", tool_executor=mock_tool_executor),
            DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_tool_executor)
        ]

        query = 'alert:{"alertname":"Test"}'
        expected = 'alert:alert:{"alertname":"Test"}'

        # All bots should have the method and it should work correctly
        for bot in bots:
            assert hasattr(bot, '_normalize_korrel8r_query')
            assert bot._normalize_korrel8r_query(query) == expected

    def test_korrel8r_normalization_with_different_query_formats(self, mock_tool_executor_with_korrel8r):
        """Test Korrel8r normalization with various query formats."""
        from mcp_server.chatbots import LlamaChatBot

        bot = LlamaChatBot(LLAMA_3_1_8B, tool_executor=mock_tool_executor_with_korrel8r)

        test_cases = [
            # (input_query, expected_normalized_query)
            ('alert:{"alertname":"Test"}', 'alert:alert:{"alertname":"Test"}'),
            ('alert:alert:{"alertname":"Test"}', 'alert:alert:{"alertname":"Test"}'),  # Already correct
            ('k8s:Alert:{"alertname":"PodDown"}', 'alert:alert:{"alertname":"PodDown"}'),  # Misclassified
            ('alert:{alertname="Test"}', 'alert:alert:{"alertname":"Test"}'),  # Unquoted keys
            ('k8s:Pod:{namespace="test"}', 'k8s:Pod:{"namespace":="test"}'),  # K8s domain
        ]

        with patch.object(mock_tool_executor_with_korrel8r, 'execute_tool', return_value="Result"):
            for input_query, expected_query in test_cases:
                # Call routing which should normalize
                bot._route_tool_call_to_mcp("korrel8r_get_correlated", {"query": input_query})

                # Get the last call args
                call_args = mock_tool_executor_with_korrel8r.execute_tool.call_args
                actual_query = call_args[0][1]["query"]

                assert actual_query == expected_query, f"Failed for input: {input_query}"


def test_no_claude_integration_references():
    """Test that no code references the deleted claude_integration module."""
    import subprocess

    # Search for references to PrometheusChatBot or claude_integration
    result = subprocess.run(
        ['grep', '-r', 'PrometheusChatBot', 'src/', '--include=*.py'],
        capture_output=True,
        text=True
    )

    # Should return non-zero (not found) or empty output
    assert result.returncode != 0 or len(result.stdout.strip()) == 0, \
        f"Found references to PrometheusChatBot: {result.stdout}"


if __name__ == "__main__":
    # Run with: python -m pytest tests/mcp_server/test_chatbots.py -v
    pytest.main([__file__, "-v"])
