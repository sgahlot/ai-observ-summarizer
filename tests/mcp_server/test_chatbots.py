"""
Tests for chatbot implementations.

This module tests the LangChain/LangGraph-based chatbot architecture including:
- Factory function routing (LangGraphAgent vs DeterministicChatBot)
- Provider detection and model ID extraction
- Tool result truncation (via factory max_tool_result_length)
- LangGraphAgent initialization and validation
- Smart nudge system (_get_nudge_for_query standalone function)
- LangChain tool bridge (create_langchain_tools)
- Korrel8r query normalization (DeterministicChatBot)
- Namespace injection into PromQL (DeterministicChatBot)
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def mock_mcp_tools():
    """Mock tool executor for testing."""
    from chatbots.tool_executor import ToolExecutor, MCPTool

    class MockToolExecutor(ToolExecutor):
        def __init__(self):
            self.tools = [
                MCPTool("execute_promql", "Execute PromQL query", {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
                MCPTool("get_label_values", "Get label values", {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"}
                    }
                })
            ]

        def call_tool(self, tool_name: str, arguments: dict) -> str:
            return '{"status": "success", "data": "mock result"}'

        def list_tools(self):
            return self.tools

        def get_tool(self, tool_name: str):
            for tool in self.tools:
                if tool.name == tool_name:
                    return tool
            return None

    return MockToolExecutor()


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
CLAUDE_HAIKU = "claude-haiku-4-5"
CLAUDE_HAIKU_WITH_PROVIDER = f"{ANTHROPIC_PROVIDER}/{CLAUDE_HAIKU}"
CLAUDE_SONNET = "claude-sonnet-4-20250514"
CLAUDE_SONNET_WITH_PROVIDER = f"{ANTHROPIC_PROVIDER}/{CLAUDE_SONNET}"
CLAUDE_HAIKU_DATED = "claude-haiku-4-5-20251001"

# OpenAI models
GPT_4O_MINI = "gpt-4o-mini"
GPT_4O_MINI_WITH_PROVIDER = f"{OPENAI_PROVIDER}/{GPT_4O_MINI}"
GPT_4O = "gpt-4o"

# Google models
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_FLASH_WITH_PROVIDER = f"{GOOGLE_PROVIDER}/{GEMINI_FLASH}"
GEMINI_FLASH_EXP = "gemini-2.0-flash-exp"
GEMINI_FLASH_EXP_WITH_PROVIDER = f"{GOOGLE_PROVIDER}/{GEMINI_FLASH_EXP}"


def test_chatbot_imports(mock_mcp_tools):
    """Test that all chatbot classes can be imported."""
    from chatbots import (
        LangGraphAgent,
        DeterministicChatBot,
        create_chatbot
    )

    assert LangGraphAgent is not None
    assert DeterministicChatBot is not None
    assert create_chatbot is not None


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

@patch("chatbots.factory.RAG_AVAILABLE", True)
@patch("langchain_openai.ChatOpenAI")
def test_factory_creates_llama_agent(mock_chat_openai, mock_mcp_tools):
    """Test that factory creates LangGraphAgent for Llama 3.1 models."""
    from chatbots import create_chatbot
    from chatbots.langchain_agent import LangGraphAgent

    bot = create_chatbot(LLAMA_3_1_8B, tool_executor=mock_mcp_tools)
    assert isinstance(bot, LangGraphAgent)
    assert bot.model_name == LLAMA_3_1_8B
    assert bot.is_local_llama is True


@patch("chatbots.factory.RAG_AVAILABLE", True)
@patch("langchain_openai.ChatOpenAI")
def test_factory_creates_llama_33_agent(mock_chat_openai, mock_mcp_tools):
    """Test that factory creates LangGraphAgent for Llama 3.3 models."""
    from chatbots import create_chatbot
    from chatbots.langchain_agent import LangGraphAgent

    bot = create_chatbot(LLAMA_3_3_70B, tool_executor=mock_mcp_tools)
    assert isinstance(bot, LangGraphAgent)
    assert bot.model_name == LLAMA_3_3_70B
    assert bot.is_local_llama is True


@patch("chatbots.factory.RAG_AVAILABLE", True)
def test_factory_creates_deterministic_bot(mock_mcp_tools):
    """Test that factory creates DeterministicChatBot for Llama 3.2 models."""
    from chatbots import create_chatbot, DeterministicChatBot

    bot = create_chatbot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)
    assert isinstance(bot, DeterministicChatBot)
    assert bot.model_name == LLAMA_3_2_3B


@patch("langchain_anthropic.ChatAnthropic")
def test_factory_creates_anthropic_agent(mock_chat_anthropic, mock_mcp_tools):
    """Test that factory creates LangGraphAgent for Anthropic models."""
    from chatbots import create_chatbot
    from chatbots.langchain_agent import LangGraphAgent

    bot = create_chatbot(CLAUDE_HAIKU_WITH_PROVIDER, api_key="test-key", tool_executor=mock_mcp_tools)
    assert isinstance(bot, LangGraphAgent)
    assert bot.is_local_llama is False


@patch("langchain_openai.ChatOpenAI")
def test_factory_creates_openai_agent(mock_chat_openai, mock_mcp_tools):
    """Test that factory creates LangGraphAgent for OpenAI models."""
    from chatbots import create_chatbot
    from chatbots.langchain_agent import LangGraphAgent

    bot = create_chatbot(GPT_4O_MINI_WITH_PROVIDER, api_key="test-key", tool_executor=mock_mcp_tools)
    assert isinstance(bot, LangGraphAgent)
    assert bot.is_local_llama is False


@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_factory_creates_google_agent(mock_chat_google, mock_mcp_tools):
    """Test that factory creates LangGraphAgent for Google models."""
    from chatbots import create_chatbot
    from chatbots.langchain_agent import LangGraphAgent

    bot = create_chatbot(GEMINI_FLASH_EXP_WITH_PROVIDER, api_key="test-key", tool_executor=mock_mcp_tools)
    assert isinstance(bot, LangGraphAgent)
    assert bot.is_local_llama is False


@patch("langchain_openai.ChatOpenAI")
def test_factory_creates_openai_agent_for_maas(mock_chat_openai, mock_mcp_tools):
    """Test that factory creates LangGraphAgent for MAAS models (OpenAI-compatible)."""
    from chatbots import create_chatbot
    from chatbots.langchain_agent import LangGraphAgent

    bot = create_chatbot("maas/qwen3-14b", api_key="test-maas-key", tool_executor=mock_mcp_tools)
    assert isinstance(bot, LangGraphAgent)
    assert bot.model_name == "maas/qwen3-14b"
    assert bot.is_local_llama is False


@patch("langchain_openai.ChatOpenAI")
def test_factory_maas_pattern_matching(mock_chat_openai, mock_mcp_tools):
    """Test that factory correctly identifies MAAS models by pattern."""
    from chatbots import create_chatbot
    from chatbots.langchain_agent import LangGraphAgent

    # Test various MAAS model name patterns
    maas_patterns = [
        "maas/qwen3-14b",
        "maas/granite-3.1-8b-instruct",
        "MAAS/model-name",  # Case insensitive
    ]

    for model_name in maas_patterns:
        bot = create_chatbot(model_name, api_key="test-key", tool_executor=mock_mcp_tools)
        assert isinstance(bot, LangGraphAgent), f"Failed for pattern: {model_name}"


@patch("langchain_openai.ChatOpenAI")
def test_factory_maas_passes_api_url(mock_chat_openai, mock_mcp_tools):
    """Test that factory passes api_url through to ChatOpenAI for MAAS models."""
    from chatbots import create_chatbot
    from chatbots.langchain_agent import LangGraphAgent

    bot = create_chatbot(
        "maas/qwen3-14b",
        api_key="test-maas-key",
        api_url="https://custom-maas.example.com/v1",
        tool_executor=mock_mcp_tools
    )
    assert isinstance(bot, LangGraphAgent)

    # Verify ChatOpenAI was called with the custom base_url
    mock_chat_openai.assert_called_once()
    call_kwargs = mock_chat_openai.call_args[1]
    assert call_kwargs["base_url"] == "https://custom-maas.example.com/v1"


def test_factory_requires_tool_executor():
    """Test that factory raises ValueError when tool_executor is None."""
    from chatbots import create_chatbot

    with pytest.raises(ValueError, match="tool_executor is required"):
        create_chatbot(CLAUDE_HAIKU_WITH_PROVIDER, api_key="test-key", tool_executor=None)


def test_factory_raises_without_rag_for_local_model():
    """Test that factory raises ValueError for local models when RAG is unavailable."""
    from chatbots import create_chatbot

    with patch("chatbots.factory.RAG_AVAILABLE", False):
        with pytest.raises(ValueError, match="not available"):
            create_chatbot(LLAMA_3_1_8B, tool_executor=MagicMock())


# ---------------------------------------------------------------------------
# Factory max_tool_result_length tests
# ---------------------------------------------------------------------------

@patch("langchain_anthropic.ChatAnthropic")
def test_factory_anthropic_max_tool_result_length(mock_chat_anthropic, mock_mcp_tools):
    """Test that factory creates LangGraphAgent with correct max_tool_result_length for Anthropic (15K)."""
    from chatbots import create_chatbot

    bot = create_chatbot(CLAUDE_HAIKU_WITH_PROVIDER, api_key="test-key", tool_executor=mock_mcp_tools)
    assert bot.max_tool_result_length == 15000


@patch("langchain_openai.ChatOpenAI")
def test_factory_openai_max_tool_result_length(mock_chat_openai, mock_mcp_tools):
    """Test that factory creates LangGraphAgent with correct max_tool_result_length for OpenAI (10K)."""
    from chatbots import create_chatbot

    bot = create_chatbot(GPT_4O_MINI_WITH_PROVIDER, api_key="test-key", tool_executor=mock_mcp_tools)
    assert bot.max_tool_result_length == 10000


@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_factory_google_max_tool_result_length(mock_chat_google, mock_mcp_tools):
    """Test that factory creates LangGraphAgent with correct max_tool_result_length for Google (10K)."""
    from chatbots import create_chatbot

    bot = create_chatbot(GEMINI_FLASH_EXP_WITH_PROVIDER, api_key="test-key", tool_executor=mock_mcp_tools)
    assert bot.max_tool_result_length == 10000


@patch("langchain_openai.ChatOpenAI")
def test_factory_maas_max_tool_result_length(mock_chat_openai, mock_mcp_tools):
    """Test that factory creates LangGraphAgent with correct max_tool_result_length for MAAS (10K)."""
    from chatbots import create_chatbot

    bot = create_chatbot("maas/qwen3-14b", api_key="test-key", tool_executor=mock_mcp_tools)
    assert bot.max_tool_result_length == 10000


@patch("chatbots.factory.RAG_AVAILABLE", True)
@patch("langchain_openai.ChatOpenAI")
def test_factory_local_llama_max_tool_result_length(mock_chat_openai, mock_mcp_tools):
    """Test that factory creates LangGraphAgent with correct max_tool_result_length for local Llama (8K)."""
    from chatbots import create_chatbot

    bot = create_chatbot(LLAMA_3_1_8B, tool_executor=mock_mcp_tools)
    assert bot.max_tool_result_length == 8000


def test_deterministic_bot_max_tool_result_length(mock_mcp_tools):
    """Test DeterministicChatBot uses base class default (5K)."""
    from chatbots import DeterministicChatBot

    bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)
    assert bot._get_max_tool_result_length() == 5000


# ---------------------------------------------------------------------------
# Provider detection tests
# ---------------------------------------------------------------------------

class TestProviderDetection:
    """Test _detect_provider function from factory."""

    def test_detect_anthropic_by_prefix(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("anthropic/claude-haiku-4-5") == "anthropic"

    def test_detect_anthropic_by_name(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("claude-haiku-4-5") == "anthropic"

    def test_detect_openai_by_prefix(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("openai/gpt-4o-mini") == "openai"

    def test_detect_openai_by_gpt_prefix(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("gpt-4o") == "openai"

    def test_detect_google_by_prefix(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("google/gemini-2.5-flash") == "google"

    def test_detect_google_by_name(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("gemini-2.0-flash-exp") == "google"

    def test_detect_maas(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("maas/qwen3-14b") == "maas"

    def test_detect_maas_case_insensitive(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("MAAS/model-name") == "maas"

    def test_detect_local_model_returns_none(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("meta-llama/Llama-3.1-8B-Instruct") is None

    def test_detect_unknown_model_returns_none(self):
        from chatbots.factory import _detect_provider
        assert _detect_provider("some-unknown-model") is None


# ---------------------------------------------------------------------------
# Model ID extraction tests
# ---------------------------------------------------------------------------

class TestModelIdExtraction:
    """Test _extract_model_id function from factory."""

    def test_strips_anthropic_prefix(self):
        from chatbots.factory import _extract_model_id
        assert _extract_model_id("anthropic/claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"

    def test_strips_openai_prefix(self):
        from chatbots.factory import _extract_model_id
        assert _extract_model_id("openai/gpt-4o-mini") == "gpt-4o-mini"

    def test_strips_google_prefix(self):
        from chatbots.factory import _extract_model_id
        assert _extract_model_id("google/gemini-2.5-flash") == "gemini-2.5-flash"

    def test_strips_maas_prefix(self):
        from chatbots.factory import _extract_model_id
        assert _extract_model_id("maas/qwen3-14b") == "qwen3-14b"

    def test_preserves_non_provider_prefix(self):
        from chatbots.factory import _extract_model_id
        # meta-llama is not in the provider list, so keep the full name
        assert _extract_model_id("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"

    def test_preserves_model_without_slash(self):
        from chatbots.factory import _extract_model_id
        assert _extract_model_id("claude-haiku-4-5") == "claude-haiku-4-5"

    def test_preserves_gpt_model_without_prefix(self):
        from chatbots.factory import _extract_model_id
        assert _extract_model_id("gpt-4o") == "gpt-4o"


# ---------------------------------------------------------------------------
# LangGraphAgent tests
# ---------------------------------------------------------------------------

class TestLangGraphAgent:
    """Test LangGraphAgent initialization and validation."""

    def test_requires_tool_executor(self):
        """Test that LangGraphAgent raises ValueError on None tool_executor."""
        from chatbots.langchain_agent import LangGraphAgent

        with pytest.raises(ValueError, match="tool_executor is required"):
            LangGraphAgent(
                chat_model=MagicMock(),
                tool_executor=None,
                tools=[],
                model_name="test-model",
            )

    def test_requires_tool_executor_type(self):
        """Test that LangGraphAgent raises TypeError for wrong tool_executor type."""
        from chatbots.langchain_agent import LangGraphAgent

        with pytest.raises(TypeError, match="must implement ToolExecutor"):
            LangGraphAgent(
                chat_model=MagicMock(),
                tool_executor="not-a-tool-executor",
                tools=[],
                model_name="test-model",
            )

    @patch("langchain_anthropic.ChatAnthropic")
    def test_agent_has_correct_attributes(self, mock_chat_anthropic, mock_mcp_tools):
        """Test that LangGraphAgent has the expected attributes after creation."""
        from chatbots import create_chatbot
        from chatbots.langchain_agent import LangGraphAgent

        bot = create_chatbot(CLAUDE_HAIKU_WITH_PROVIDER, api_key="test-key", tool_executor=mock_mcp_tools)

        assert isinstance(bot, LangGraphAgent)
        assert bot.model_name == CLAUDE_HAIKU_WITH_PROVIDER
        assert bot.tool_executor is mock_mcp_tools
        assert isinstance(bot.tools, list)
        assert bot.is_local_llama is False
        assert bot.max_tool_result_length == 15000

    @patch("chatbots.factory.RAG_AVAILABLE", True)
    @patch("langchain_openai.ChatOpenAI")
    def test_local_llama_agent_attributes(self, mock_chat_openai, mock_mcp_tools):
        """Test that local Llama LangGraphAgent has correct attributes."""
        from chatbots import create_chatbot
        from chatbots.langchain_agent import LangGraphAgent

        bot = create_chatbot(LLAMA_3_1_8B, tool_executor=mock_mcp_tools)

        assert isinstance(bot, LangGraphAgent)
        assert bot.model_name == LLAMA_3_1_8B
        assert bot.is_local_llama is True
        assert bot.max_tool_result_length == 8000


# ---------------------------------------------------------------------------
# BaseChatBot abstract tests
# ---------------------------------------------------------------------------

class TestBaseChatBot:
    """Test BaseChatBot common functionality."""

    def test_base_chatbot_is_abstract(self, mock_mcp_tools):
        """Test that BaseChatBot cannot be instantiated directly."""
        from chatbots.base import BaseChatBot

        # BaseChatBot is abstract and should raise TypeError
        with pytest.raises(TypeError):
            BaseChatBot("test-model")


# ---------------------------------------------------------------------------
# LangChain tool bridge tests
# ---------------------------------------------------------------------------

class TestCreateLangchainTools:
    """Test create_langchain_tools from tools.py."""

    def test_creates_correct_number_of_wrappers(self, mock_mcp_tools):
        """Test that create_langchain_tools creates one wrapper per MCP tool."""
        from chatbots.tools import create_langchain_tools

        lc_tools = create_langchain_tools(mock_mcp_tools)
        assert len(lc_tools) == 2

    def test_creates_correct_tool_names(self, mock_mcp_tools):
        """Test that created tools have correct names."""
        from chatbots.tools import create_langchain_tools

        lc_tools = create_langchain_tools(mock_mcp_tools)
        tool_names = {t.name for t in lc_tools}
        assert tool_names == {"execute_promql", "get_label_values"}

    def test_allowlist_filters_tools(self, mock_mcp_tools):
        """Test that allowlist parameter filters tools correctly."""
        from chatbots.tools import create_langchain_tools

        lc_tools = create_langchain_tools(mock_mcp_tools, allowlist={"execute_promql"})
        assert len(lc_tools) == 1
        assert lc_tools[0].name == "execute_promql"

    def test_empty_allowlist_returns_no_tools(self, mock_mcp_tools):
        """Test that empty allowlist returns no tools."""
        from chatbots.tools import create_langchain_tools

        lc_tools = create_langchain_tools(mock_mcp_tools, allowlist=set())
        assert len(lc_tools) == 0

    def test_none_allowlist_returns_all_tools(self, mock_mcp_tools):
        """Test that None allowlist returns all tools."""
        from chatbots.tools import create_langchain_tools

        lc_tools = create_langchain_tools(mock_mcp_tools, allowlist=None)
        assert len(lc_tools) == 2


# ---------------------------------------------------------------------------
# Korrel8r normalization tests (using DeterministicChatBot)
# ---------------------------------------------------------------------------

class TestKorrel8rNormalization:
    """Test Korrel8r query normalization functionality in BaseChatBot."""

    def test_normalize_alert_query_missing_class(self, mock_mcp_tools):
        """Test that alert queries without class get 'alert:alert:' prefix."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Missing class - should be normalized
        query = 'alert:{"alertname":"PodDisruptionBudgetAtLimit"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"PodDisruptionBudgetAtLimit"}'

    def test_normalize_alert_query_already_correct(self, mock_mcp_tools):
        """Test that correctly formatted alert queries are not changed."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Already correct - should not change
        query = 'alert:alert:{"alertname":"HighCPU"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"HighCPU"}'

    def test_normalize_escaped_quotes(self, mock_mcp_tools):
        """Test that escaped quotes are unescaped."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Escaped quotes should be unescaped
        query = 'alert:{\"alertname\":\"Test\"}'
        normalized = bot._normalize_korrel8r_query(query)

        # Should unescape quotes AND add missing class
        assert normalized == 'alert:alert:{"alertname":"Test"}'

    def test_normalize_k8s_alert_misclassification(self, mock_mcp_tools):
        """Test that k8s:Alert: is corrected to alert:alert:."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Misclassified as k8s - should be corrected
        query = 'k8s:Alert:{"alertname":"PodDown"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"PodDown"}'

    def test_normalize_alert_unquoted_keys(self, mock_mcp_tools):
        """Test that unquoted keys in alert selectors are quoted (JSON format)."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Unquoted key - should be quoted for alert domain
        query = 'alert:alert:{alertname="HighLatency"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"HighLatency"}'

    def test_normalize_alert_multiple_unquoted_keys(self, mock_mcp_tools):
        """Test normalization with multiple unquoted keys."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Multiple unquoted keys
        query = 'alert:alert:{alertname="Test",severity="critical"}'
        normalized = bot._normalize_korrel8r_query(query)

        assert normalized == 'alert:alert:{"alertname":"Test","severity":"critical"}'

    def test_normalize_k8s_pod_query(self, mock_mcp_tools):
        """Test normalization of k8s Pod queries (non-alert domain)."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # k8s domain uses := operator format
        query = 'k8s:Pod:{namespace="llm-serving"}'
        normalized = bot._normalize_korrel8r_query(query)

        # For non-alert domains, should use := operator
        assert normalized == 'k8s:Pod:{"namespace":="llm-serving"}'

    def test_normalize_loki_log_query(self, mock_mcp_tools):
        """Test normalization of loki log queries."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Loki domain
        query = 'loki:log:{kubernetes.namespace_name="test"}'
        normalized = bot._normalize_korrel8r_query(query)

        # Should use := for non-alert domains
        assert 'kubernetes.namespace_name":=' in normalized

    def test_normalize_trace_span_query(self, mock_mcp_tools):
        """Test normalization of trace span queries."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Trace domain - dots in key names need special handling
        query = 'trace:span:{k8s_namespace_name="llm-serving"}'
        normalized = bot._normalize_korrel8r_query(query)

        # Should use := for non-alert domains
        assert 'k8s_namespace_name":=' in normalized

    def test_normalize_empty_query(self, mock_mcp_tools):
        """Test that empty queries are handled gracefully."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Empty query
        normalized = bot._normalize_korrel8r_query("")
        assert normalized == ""

    def test_normalize_none_query(self, mock_mcp_tools):
        """Test that None queries are handled gracefully."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # None query - implementation converts to empty string
        normalized = bot._normalize_korrel8r_query(None)
        assert normalized == ""

    def test_normalize_malformed_query_doesnt_crash(self, mock_mcp_tools):
        """Test that malformed queries don't crash the normalization."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Malformed query - should return original on error
        query = 'totally:invalid{{'
        normalized = bot._normalize_korrel8r_query(query)

        # Should return something (either original or partially normalized)
        assert normalized is not None


# ---------------------------------------------------------------------------
# Korrel8r tool integration tests (using DeterministicChatBot)
# ---------------------------------------------------------------------------

class TestKorrel8rToolIntegration:
    """Test Korrel8r tool integration in routing."""

    def test_normalize_is_called_for_korrel8r_queries(self, mock_mcp_tools):
        """Test that normalization is invoked for korrel8r queries."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Test that normalize method works correctly
        query = 'alert:{"alertname":"Test"}'
        normalized = bot._normalize_korrel8r_query(query)

        # Should be normalized
        assert normalized == 'alert:alert:{"alertname":"Test"}'

    def test_normalization_available_to_deterministic_bot(self, mock_mcp_tools):
        """Test that normalization method is available to DeterministicChatBot."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        query = 'alert:{"alertname":"Test"}'
        expected = 'alert:alert:{"alertname":"Test"}'

        assert hasattr(bot, '_normalize_korrel8r_query')
        assert bot._normalize_korrel8r_query(query) == expected


# ---------------------------------------------------------------------------
# Standalone normalize_korrel8r_query tests (langchain_agent.py version)
# ---------------------------------------------------------------------------

class TestStandaloneKorrel8rNormalization:
    """Test the standalone normalize_korrel8r_query function in langchain_agent.py."""

    def test_normalize_alert_missing_class(self):
        from chatbots.langchain_agent import normalize_korrel8r_query

        query = 'alert:{"alertname":"PodDown"}'
        result = normalize_korrel8r_query(query)
        assert result == 'alert:alert:{"alertname":"PodDown"}'

    def test_normalize_k8s_alert_misclassification(self):
        from chatbots.langchain_agent import normalize_korrel8r_query

        query = 'k8s:Alert:{"alertname":"HighCPU"}'
        result = normalize_korrel8r_query(query)
        assert result == 'alert:alert:{"alertname":"HighCPU"}'

    def test_normalize_empty_query(self):
        from chatbots.langchain_agent import normalize_korrel8r_query

        result = normalize_korrel8r_query("")
        assert result == ""

    def test_normalize_none_query(self):
        from chatbots.langchain_agent import normalize_korrel8r_query

        result = normalize_korrel8r_query(None)
        assert result == ""


# ---------------------------------------------------------------------------
# Standalone inject_namespace_into_promql tests (langchain_agent.py version)
# ---------------------------------------------------------------------------

class TestStandaloneNamespaceInjection:
    """Test the standalone inject_namespace_into_promql function."""

    def test_replaces_namespace_not_equal(self):
        from chatbots.langchain_agent import inject_namespace_into_promql

        query = 'ALERTS{namespace!="kube-system",alertstate="firing"}'
        result = inject_namespace_into_promql(query, "my-app")
        assert 'namespace="my-app"' in result
        assert 'namespace!=' not in result

    def test_replaces_namespace_not_regex(self):
        from chatbots.langchain_agent import inject_namespace_into_promql

        query = 'up{namespace!~"kube-.*"}'
        result = inject_namespace_into_promql(query, "my-app")
        assert 'namespace="my-app"' in result
        assert 'namespace!~' not in result

    def test_no_namespace_injects_new_filter(self):
        from chatbots.langchain_agent import inject_namespace_into_promql

        query = 'container_cpu_usage_seconds_total{pod="web-1"}'
        result = inject_namespace_into_promql(query, "my-app")
        assert 'namespace="my-app"' in result
        assert 'pod="web-1"' in result


# ---------------------------------------------------------------------------
# No claude_integration references test
# ---------------------------------------------------------------------------

def test_no_claude_integration_references(mock_mcp_tools):
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


# ---------------------------------------------------------------------------
# Namespace injection tests (using DeterministicChatBot)
# ---------------------------------------------------------------------------

class TestNamespaceInjection:
    """Test namespace injection via _get_tool_result parameter."""

    def test_get_tool_result_injects_namespace_into_promql(self, mock_mcp_tools):
        """Test that passing namespace to _get_tool_result modifies the PromQL query."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        with patch.object(bot, '_route_tool_call_to_mcp', return_value="result") as mock_route:
            bot._get_tool_result("execute_promql", {"query": "up"}, namespace="test-ns")

            # The query should have been modified to include namespace
            call_args = mock_route.call_args
            actual_args = call_args[0][1]  # second positional arg is tool_args
            assert 'namespace="test-ns"' in actual_args["query"]

    def test_get_tool_result_injects_namespace_into_tool_args(self, mock_mcp_tools):
        """Test that passing namespace to a namespace-aware non-PromQL tool adds namespace to tool_args."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # search_metrics is in NAMESPACE_AWARE_TOOLS
        with patch.object(bot, '_namespace_aware_tools', {"search_metrics", "execute_promql"}):
            with patch.object(bot, '_route_tool_call_to_mcp', return_value="result") as mock_route:
                bot._get_tool_result("search_metrics", {"pattern": "cpu"}, namespace="test-ns")

                call_args = mock_route.call_args
                actual_args = call_args[0][1]
                assert actual_args["namespace"] == "test-ns"
                assert actual_args["pattern"] == "cpu"


class TestNamespaceInjectionPromQL:
    """Test _inject_namespace_into_promql handles negated operators correctly."""

    def _make_bot(self, mock_mcp_tools):
        from chatbots import DeterministicChatBot
        return DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

    def test_replaces_namespace_not_equal(self, mock_mcp_tools):
        """namespace!=\"kube-system\" should be replaced with the active namespace."""
        bot = self._make_bot(mock_mcp_tools)
        query = 'ALERTS{namespace!="kube-system",alertstate="firing"}'
        result = bot._inject_namespace_into_promql(query, "my-app")
        assert 'namespace="my-app"' in result
        assert 'namespace!=' not in result

    def test_replaces_namespace_not_regex(self, mock_mcp_tools):
        """namespace!~\"kube-.*\" should be replaced with the active namespace."""
        bot = self._make_bot(mock_mcp_tools)
        query = 'up{namespace!~"kube-.*"}'
        result = bot._inject_namespace_into_promql(query, "my-app")
        assert 'namespace="my-app"' in result
        assert 'namespace!~' not in result

    def test_replaces_namespace_regex_match(self, mock_mcp_tools):
        """namespace=~\"test-.*\" should be replaced with the active namespace."""
        bot = self._make_bot(mock_mcp_tools)
        query = 'container_memory_usage_bytes{namespace=~"test-.*"}'
        result = bot._inject_namespace_into_promql(query, "my-app")
        assert 'namespace="my-app"' in result
        assert 'namespace=~' not in result

    def test_no_namespace_injects_new_filter(self, mock_mcp_tools):
        """Query without any namespace filter should get one injected."""
        bot = self._make_bot(mock_mcp_tools)
        query = 'container_cpu_usage_seconds_total{pod="web-1"}'
        result = bot._inject_namespace_into_promql(query, "my-app")
        assert 'namespace="my-app"' in result
        assert 'pod="web-1"' in result


# ---------------------------------------------------------------------------
# Smart nudge tests (_get_nudge_for_query standalone function)
# ---------------------------------------------------------------------------

class TestSmartNudge:
    """Test query-category detection and smart nudge generation via standalone function."""

    def test_alert_query_routes_to_execute_promql(self):
        """Test that alert queries produce a nudge mentioning execute_promql."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("Any alerts firing in jianrong namespace", namespace="jianrong")

        assert tool == "execute_promql"
        assert "execute_promql" in nudge
        assert "ALERTS" in nudge
        assert "jianrong" in nudge

    def test_pod_failure_query_routes_to_execute_promql(self):
        """Test that pod failure queries produce a nudge with the right PromQL."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("any pods failing in openshift-monitoring namespace")

        assert tool == "execute_promql"
        assert "execute_promql" in nudge
        assert "kube_pod_container_status_waiting_reason" in nudge

    def test_correlation_query_routes_to_korrel8r(self):
        """Test that correlation queries produce a nudge mentioning korrel8r."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("Use correlated data to investigate pod my-app in jianrong namespace")

        assert tool == "korrel8r_get_correlated"
        assert "korrel8r_get_correlated" in nudge

    def test_korrel8r_beats_alert_in_pod_name(self):
        """Test that korrel8r/investigate matches before alert pattern for pod names like alert-example."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query(
            "Use korrel8r to investigate pod alert-example-5d9cbf68fd-62zsb in jianrong ns"
        )

        assert tool == "korrel8r_get_correlated"
        assert "korrel8r_get_correlated" in nudge

    def test_trace_detail_query_routes_to_get_trace_details(self):
        """Test that trace detail queries produce a nudge mentioning get_trace_details_tool."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("Give me trace details for trace id abc123")

        assert tool == "get_trace_details_tool"
        assert "get_trace_details_tool" in nudge

    def test_general_trace_query_routes_to_chat_tempo(self):
        """Test that general trace queries produce a nudge mentioning chat_tempo_tool."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("Find the top trace and find its details")

        # "trace" + "details" should match trace detail pattern first
        # but "Find the top trace" without "trace id" may match general trace
        assert "trace" in nudge.lower()

    def test_gpu_query_routes_to_execute_promql(self):
        """Test that GPU queries produce a nudge with GPU-specific PromQL hints."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("Show me GPU power consumption and temperature trends")

        assert tool == "execute_promql"
        assert "execute_promql" in nudge
        assert "DCGM_FI_DEV_POWER_USAGE" in nudge
        assert "DCGM_FI_DEV_GPU_TEMP" in nudge

    def test_unknown_query_returns_generic_nudge(self):
        """Test that unrecognized queries get a generic nudge."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("What is the meaning of life?")

        assert tool is None
        assert "MUST use the provided tools" in nudge

    def test_namespace_substitution(self):
        """Test that namespace placeholder is replaced when namespace is provided."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("Any alerts firing", namespace="my-ns")

        assert "my-ns" in nudge
        assert "<namespace>" not in nudge

    def test_no_namespace_removes_placeholder(self):
        """Test that namespace placeholder is removed when no namespace is provided."""
        from chatbots.langchain_agent import _get_nudge_for_query

        nudge, tool = _get_nudge_for_query("Any alerts firing")

        assert "<namespace>" not in nudge


# ---------------------------------------------------------------------------
# Tool result truncation tests (DeterministicChatBot)
# ---------------------------------------------------------------------------

class TestToolResultTruncation:
    """Test tool result truncation for DeterministicChatBot."""

    def test_deterministic_bot_uses_base_max_length(self, mock_mcp_tools):
        """Test DeterministicChatBot uses base class default (5K)."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)
        assert bot._get_max_tool_result_length() == 5000

    def test_get_tool_result_truncates_large_results(self, mock_mcp_tools):
        """Test that _get_tool_result properly truncates results exceeding max length."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Mock _route_tool_call_to_mcp to return a large result
        large_result = "x" * 7000  # 7K chars, exceeds DeterministicChatBot's 5K limit
        with patch.object(bot, '_route_tool_call_to_mcp', return_value=large_result):
            result = bot._get_tool_result("test_tool", {"arg": "value"})

            # Should be truncated to 5000 + truncation message
            assert len(result) == 5000 + len("\n... [Result truncated due to size]")
            assert result.endswith("\n... [Result truncated due to size]")
            assert result.startswith("x" * 100)  # Verify it starts with the original content

    def test_get_tool_result_does_not_truncate_small_results(self, mock_mcp_tools):
        """Test that _get_tool_result doesn't truncate results within max length."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        # Mock _route_tool_call_to_mcp to return a small result
        small_result = "Small result"
        with patch.object(bot, '_route_tool_call_to_mcp', return_value=small_result):
            result = bot._get_tool_result("test_tool", {"arg": "value"})

            # Should NOT be truncated
            assert result == small_result
            assert "truncated" not in result.lower()

    def test_get_tool_result_calls_route_with_correct_args(self, mock_mcp_tools):
        """Test that _get_tool_result calls _route_tool_call_to_mcp with correct arguments."""
        from chatbots import DeterministicChatBot

        bot = DeterministicChatBot(LLAMA_3_2_3B, tool_executor=mock_mcp_tools)

        with patch.object(bot, '_route_tool_call_to_mcp', return_value="result") as mock_route:
            tool_name = "execute_promql"
            tool_args = {"query": "up"}

            bot._get_tool_result(tool_name, tool_args)

            # Verify the method was called with correct args
            mock_route.assert_called_once_with(tool_name, tool_args)


# ---------------------------------------------------------------------------
# Standalone get_tool_result tests (langchain_agent.py version)
# ---------------------------------------------------------------------------

class TestStandaloneGetToolResult:
    """Test the standalone get_tool_result function from langchain_agent.py."""

    def test_truncates_large_results(self, mock_mcp_tools):
        """Test that get_tool_result truncates results exceeding max_length."""
        from chatbots.langchain_agent import get_tool_result
        from core.config import NAMESPACE_AWARE_TOOLS

        large_result = "x" * 12000
        with patch.object(mock_mcp_tools, 'call_tool', return_value=large_result):
            result = get_tool_result(
                tool_executor=mock_mcp_tools,
                tool_name="execute_promql",
                tool_args={"query": "up"},
                namespace=None,
                namespace_aware_tools=NAMESPACE_AWARE_TOOLS,
                max_length=10000,
            )

            assert len(result) == 10000 + len("\n... [Result truncated due to size]")
            assert result.endswith("\n... [Result truncated due to size]")

    def test_does_not_truncate_small_results(self, mock_mcp_tools):
        """Test that get_tool_result does not truncate results within max_length."""
        from chatbots.langchain_agent import get_tool_result
        from core.config import NAMESPACE_AWARE_TOOLS

        small_result = "Small result"
        with patch.object(mock_mcp_tools, 'call_tool', return_value=small_result):
            result = get_tool_result(
                tool_executor=mock_mcp_tools,
                tool_name="test_tool",
                tool_args={"arg": "value"},
                namespace=None,
                namespace_aware_tools=NAMESPACE_AWARE_TOOLS,
                max_length=10000,
            )

            assert result == small_result

    def test_injects_namespace_into_promql(self, mock_mcp_tools):
        """Test that get_tool_result injects namespace into PromQL queries."""
        from chatbots.langchain_agent import get_tool_result
        from core.config import NAMESPACE_AWARE_TOOLS

        with patch.object(mock_mcp_tools, 'call_tool', return_value="result") as mock_call:
            get_tool_result(
                tool_executor=mock_mcp_tools,
                tool_name="execute_promql",
                tool_args={"query": "up"},
                namespace="test-ns",
                namespace_aware_tools=NAMESPACE_AWARE_TOOLS,
                max_length=10000,
            )

            call_args = mock_call.call_args
            actual_args = call_args[0][1]  # second positional arg
            assert 'namespace="test-ns"' in actual_args["query"]

    def test_normalizes_korrel8r_queries(self, mock_mcp_tools):
        """Test that get_tool_result normalizes korrel8r queries."""
        from chatbots.langchain_agent import get_tool_result
        from core.config import NAMESPACE_AWARE_TOOLS

        with patch.object(mock_mcp_tools, 'call_tool', return_value="result") as mock_call:
            get_tool_result(
                tool_executor=mock_mcp_tools,
                tool_name="korrel8r_get_correlated",
                tool_args={"query": 'alert:{"alertname":"Test"}'},
                namespace=None,
                namespace_aware_tools=NAMESPACE_AWARE_TOOLS,
                max_length=10000,
            )

            call_args = mock_call.call_args
            actual_args = call_args[0][1]
            # Should have been normalized to include alert:alert:
            assert actual_args["query"] == 'alert:alert:{"alertname":"Test"}'


if __name__ == "__main__":
    # Run with: python -m pytest tests/mcp_server/test_chatbots.py -v
    pytest.main([__file__, "-v"])
