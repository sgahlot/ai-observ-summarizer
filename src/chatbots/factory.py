"""
Chat Bot Factory

Creates LangGraph-based chatbot agents backed by LangChain ChatModels.
Handles provider detection (Anthropic, OpenAI, Google, MAAS, local Llama)
and returns a LangGraphAgent with the appropriate ChatModel.
"""

import os
from typing import Optional

from .tool_executor import ToolExecutor
from .tools import create_langchain_tools
from .langchain_agent import LangGraphAgent
from .deterministic_bot import DeterministicChatBot
from common.pylogger import get_python_logger
from core.config import RAG_AVAILABLE, LLAMA_STACK_URL, LLM_API_TOKEN, LLM_TIMEOUT_SECONDS

logger = get_python_logger()

# Llama tool allowlist — controls which MCP tools are sent to
# context-constrained local models.  Kept here (not in the MCP server)
# because the constraint is specific to Llama's limited context window.
_LLAMA_TOOL_ALLOWLIST = {
    # Prometheus / metrics
    "execute_promql",
    "search_metrics",
    "get_metric_metadata",
    "get_label_values",
    "suggest_queries",
    "explain_results",
    "select_best_metric",
    "get_metrics_categories",
    "search_metrics_by_category",
    "get_category_metrics_detail",
    # Traces
    "chat_tempo_tool",
    "query_tempo_tool",
    "get_trace_details_tool",
    # Correlation & logs
    "korrel8r_get_correlated",
    "korrel8r_query_objects",
    "get_correlated_logs",
    # Infrastructure discovery
    "get_gpu_info",
    "get_deployment_info",
    "list_openshift_namespaces",
}


def _detect_provider(model_name: str) -> Optional[str]:
    """Detect the LLM provider from a model name string.

    Returns one of: 'maas', 'anthropic', 'openai', 'google', or None (local).
    """
    PROVIDER_PATTERNS = {
        "maas": [("maas/", True)],
        "anthropic": [("anthropic/", True), ("claude", False)],
        "openai": [("openai/", True), ("gpt-", True), ("o1-", True)],
        "google": [("google/", True), ("gemini", False)],
    }

    model_lower = model_name.lower()
    for prov, patterns in PROVIDER_PATTERNS.items():
        for pattern, is_startswith in patterns:
            if model_lower.startswith(pattern) if is_startswith else (pattern in model_lower):
                logger.info("Detected %s model from name: %s", prov, model_name)
                return prov
    return None


def _extract_model_id(model_name: str) -> str:
    """Strip provider prefix from model name for the API."""
    if "/" in model_name:
        parts = model_name.split("/", 1)
        # Keep MAAS-style prefixes where the first part is the provider routing hint
        # but the API model ID is the part after the slash
        if parts[0].lower() in ("anthropic", "openai", "google", "maas"):
            return parts[1]
    return model_name


def _create_chat_model(
    provider: str,
    model_name: str,
    api_key: Optional[str],
    api_url: Optional[str],
):
    """Create a LangChain ChatModel for the given provider."""

    model_id = _extract_model_id(model_name)
    timeout = LLM_TIMEOUT_SECONDS

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=model_id,
            api_key=key,
            timeout=timeout,
            temperature=0,
            max_tokens=4096,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model_id,
            api_key=key,
            timeout=timeout,
            temperature=0,
        )

    # OpenAI and MAAS both use the OpenAI-compatible interface
    from langchain_openai import ChatOpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    kwargs = dict(
        model=model_id,
        api_key=key or "not-needed",
        timeout=timeout,
        temperature=0,
    )
    if api_url:
        kwargs["base_url"] = api_url

    return ChatOpenAI(**kwargs)


def _create_local_chat_model(model_name: str):
    """Create a LangChain ChatModel for a local model via LlamaStack."""
    from langchain_openai import ChatOpenAI

    base_url = LLAMA_STACK_URL.removesuffix("/chat/completions")

    return ChatOpenAI(
        model=model_name,
        api_key=LLM_API_TOKEN or "dummy",
        base_url=base_url,
        timeout=LLM_TIMEOUT_SECONDS,
        temperature=0,
    )


# Per-provider max tool result lengths (mirrors the old bot classes)
_MAX_TOOL_RESULT_LENGTHS = {
    "anthropic": 15000,
    "openai": 10000,
    "maas": 10000,
    "google": 10000,
    "local_llama": 8000,
}


def create_chatbot(
    model_name: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    tool_executor: ToolExecutor = None
) -> LangGraphAgent:
    """
    Factory function to create a LangGraph-based chatbot agent.

    Args:
        model_name: Name of the model to use
        api_key: Optional API key for external models
        api_url: Optional API URL for custom endpoints (for DEV mode, MAAS)
        tool_executor: Tool executor for calling observability tools (required)

    Returns:
        LangGraphAgent instance with the appropriate ChatModel

    Raises:
        ValueError: If tool_executor is None or local model requested without RAG
    """
    if tool_executor is None:
        raise ValueError(
            "tool_executor is required. Pass a MCPServerAdapter instance "
            "from the MCP server context"
        )

    provider = _detect_provider(model_name)
    is_external = provider is not None

    if is_external:
        # External provider — create LangChain ChatModel
        chat_model = _create_chat_model(provider, model_name, api_key, api_url)
        tools = create_langchain_tools(tool_executor)
        max_length = _MAX_TOOL_RESULT_LENGTHS.get(provider, 10000)

        logger.info("Creating LangGraphAgent for %s model %s", provider, model_name)
        return LangGraphAgent(
            chat_model=chat_model,
            tool_executor=tool_executor,
            tools=tools,
            model_name=model_name,
            is_local_llama=False,
            max_tool_result_length=max_length,
        )

    # Local model — check RAG availability
    if not RAG_AVAILABLE:
        logger.error("Local model %s requested but RAG infrastructure not available", model_name)
        raise ValueError(
            f"Local model '{model_name}' is not available. "
            "RAG infrastructure is not installed or not accessible. "
            "Please use an external model (anthropic/claude, openai/gpt, google/gemini) instead."
        )

    # Local models — detect Llama version and capabilities
    model_lower = model_name.lower().replace("-", ".")

    # Llama 3.2 and unknown models → DeterministicChatBot (unchanged)
    if "llama.3.2" in model_lower:
        logger.info(
            "Creating DeterministicChatBot for %s (67%% tool calling accuracy)",
            model_name,
        )
        return DeterministicChatBot(model_name, api_key, tool_executor)

    # Llama 3.1/3.3 → LangGraph agent with local ChatModel + nudge system
    is_known_llama = "llama.3.1" in model_lower or "llama.3.3" in model_lower
    if is_known_llama:
        chat_model = _create_local_chat_model(model_name)
        tools = create_langchain_tools(tool_executor, allowlist=_LLAMA_TOOL_ALLOWLIST)

        logger.info("Creating LangGraphAgent for local Llama model %s", model_name)
        return LangGraphAgent(
            chat_model=chat_model,
            tool_executor=tool_executor,
            tools=tools,
            model_name=model_name,
            is_local_llama=True,
            max_tool_result_length=_MAX_TOOL_RESULT_LENGTHS["local_llama"],
        )

    # Unknown local model — use deterministic parsing for safety
    logger.info(
        "Creating DeterministicChatBot for %s (unknown capability)",
        model_name,
    )
    return DeterministicChatBot(model_name, api_key, tool_executor)
