"""
Chat Bot Factory

This module provides a factory function to create the appropriate chatbot
based on model capabilities and provider.
"""

from typing import Optional

from .base import BaseChatBot
from .tool_executor import ToolExecutor
from .anthropic_bot import AnthropicChatBot
from .openai_bot import OpenAIChatBot
from .google_bot import GoogleChatBot
from .llama_bot import LlamaChatBot
from .deterministic_bot import DeterministicChatBot
from common.pylogger import get_python_logger

logger = get_python_logger()


def create_chatbot(
    model_name: str,
    api_key: Optional[str] = None,
    tool_executor: ToolExecutor = None
) -> BaseChatBot:
    """
    Factory function to create the appropriate chatbot based on model capabilities.

    Args:
        model_name: Name of the model to use
        api_key: Optional API key for external models
        tool_executor: Tool executor for calling observability tools (required)

    Returns:
        Instance of the appropriate chatbot class

    Raises:
        ValueError: If tool_executor is None

    Examples:
        >>> from mcp_client_adapter import MCPClientAdapter
        >>> tool_executor = MCPClientAdapter()
        >>> chatbot = create_chatbot("gpt-4o-mini", api_key="sk-...", tool_executor=tool_executor)
        >>> response = chatbot.chat("What's the CPU usage?")

        >>> chatbot = create_chatbot("meta-llama/Llama-3.1-8B-Instruct", tool_executor=tool_executor)
        >>> response = chatbot.chat("Check memory usage")
    """
    if tool_executor is None:
        raise ValueError(
            "tool_executor is required. Please pass an implementation of ToolExecutor."
        )

    # Detect provider from model name pattern using dict mapping
    PROVIDER_PATTERNS = {
        "anthropic": [("anthropic/", False), ("claude", False)],
        "openai": [("openai/", False), ("gpt-", True), ("o1-", True)],
        "google": [("google/", False), ("gemini", False)]
    }

    model_lower = model_name.lower()
    provider = None
    for prov, patterns in PROVIDER_PATTERNS.items():
        for pattern, is_startswith in patterns:
            if model_lower.startswith(pattern) if is_startswith else (pattern in model_lower):
                provider = prov
                logger.info(f"Detected {prov} model from name: {model_name}")
                break
        if provider:
            break

    is_external = provider is not None

    # Route to appropriate implementation based on provider and capabilities
    if is_external:
        if provider == "anthropic":
            logger.info(f"Creating AnthropicChatBot for {model_name}")
            return AnthropicChatBot(model_name, api_key, tool_executor)
        elif provider == "openai":
            logger.info(f"Creating OpenAIChatBot for {model_name}")
            return OpenAIChatBot(model_name, api_key, tool_executor)
        elif provider == "google":
            logger.info(f"Creating GoogleChatBot for {model_name}")
            return GoogleChatBot(model_name, api_key, tool_executor)
        else:
            logger.warning(f"Unknown external provider {provider}, using OpenAI as fallback")
            return OpenAIChatBot(model_name, api_key, tool_executor)
    else:
        # Local models - detect Llama version and create appropriate bot
        LLAMA_MODEL_PATTERNS = {
            "llama.3.1": (LlamaChatBot, "tool calling capable", ["8b", "70b"]),
            "llama.3.3": (LlamaChatBot, "tool calling capable", ["70b"]),
            "llama.3.2": (DeterministicChatBot, "67% tool calling accuracy - using deterministic parsing", None),
        }
        # Local models - check if they support reliable tool calling
        model_lower = model_name.lower().replace("-", ".")  # Replace "-" with "."

        for model, patterns in LLAMA_MODEL_PATTERNS.items():
            bot_class, model_capability, sizes = patterns
            if model in model_lower:
                if sizes is None or any(size in model_lower for size in sizes):
                    logger.info(f"Creating {bot_class.__name__} for {model_name} ({model_capability})")
                    return bot_class(model_name, api_key, tool_executor)

        # Unknown local models - use deterministic parsing for safety
        logger.info(f"Creating DeterministicChatBot for {model_name} (unknown capability - using deterministic parsing)")
        return DeterministicChatBot(model_name, api_key, tool_executor)
