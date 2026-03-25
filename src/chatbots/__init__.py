"""
Chat Bots Package

This package provides LangGraph-based chatbot agents backed by LangChain ChatModels.

Usage:
    from chatbots import create_chatbot

    # Create a chatbot using the factory function
    chatbot = create_chatbot("gpt-4o-mini", api_key="sk-...", tool_executor=executor)
    response = chatbot.chat("What's the CPU usage?")

Architecture:
    - LangGraphAgent: Unified agent using LangChain ChatModels + LangGraph StateGraph
    - DeterministicChatBot: Rule-based fallback for small/unknown local models
    - create_chatbot(): Factory function to create appropriate agent
    - ToolExecutor: Interface for tool execution (MCPServerAdapter implements this)
"""

# Lazy imports to avoid loading SDKs until needed
def __getattr__(name):
    if name == 'LangGraphAgent':
        from .langchain_agent import LangGraphAgent
        return LangGraphAgent
    elif name == 'ToolExecutor':
        from .tool_executor import ToolExecutor
        return ToolExecutor
    elif name == 'DeterministicChatBot':
        from .deterministic_bot import DeterministicChatBot
        return DeterministicChatBot
    elif name == 'create_chatbot':
        from .factory import create_chatbot
        return create_chatbot
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'LangGraphAgent',
    'ToolExecutor',
    'DeterministicChatBot',
    'create_chatbot',
]

__version__ = '2.0.0'
