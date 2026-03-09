"""
Llama 70B Chat Bot Implementation (via LlamaStack)

This module provides a 70B-optimized Llama implementation using LlamaStack's
OpenAI-compatible API.  The 70B model is capable enough for reliable tool calling
without the extra guard / nudge logic that the 8B model needs.
"""

import json
import re
from typing import Optional, List, Dict, Any, Callable

from openai import OpenAI

from .base import BaseChatBot
from chatbots.tool_executor import ToolExecutor
from core.config import LLAMA_STACK_URL, LLM_API_TOKEN, LLM_TIMEOUT_SECONDS
from common.pylogger import get_python_logger

logger = get_python_logger()


class Llama70BChatBot(BaseChatBot):
    """70B Llama implementation — clean chat loop, no 8B guardrails."""

    def _get_api_key(self) -> Optional[str]:
        """Local models don't require API keys."""
        return None

    def _get_max_tool_result_length(self) -> int:
        """70B gets 10K per tool result.

        Lower than the theoretical max to leave room for the system prompt
        and formatting instructions after multiple tool results accumulate.
        """
        return 10000

    def _get_tool_allowlist(self) -> set:
        """70B gets all tools (no restriction)."""
        return None

    def _extract_model_name(self) -> str:
        """Resolve the model name registered in LlamaStack.

        LlamaStack 0.5.x registers models with a provider-prefixed ID
        (e.g., ``provider-key/meta-llama/Llama-3.3-70B-Instruct``).
        Query ``/v1/models`` once and cache the resolved name so the
        OpenAI SDK sends the correct model ID on each request.
        """
        if self._resolved_model_name is not None:
            return self._resolved_model_name
        return self.model_name

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        tool_executor: ToolExecutor = None
    ):
        super().__init__(model_name, api_key, tool_executor)

        self._resolved_model_name: Optional[str] = None
        base_url = LLAMA_STACK_URL.removesuffix("/chat/completions")
        self.client = OpenAI(
            base_url=base_url,
            api_key=LLM_API_TOKEN or "dummy"
        )

        # Resolve provider-prefixed model name from LlamaStack
        self._resolve_model_name(base_url)

    def _resolve_model_name(self, base_url: str) -> None:
        """Query LlamaStack /v1/models to find the provider-prefixed model ID.

        LlamaStack 0.5.x registers models as ``<provider-key>/<model-id>``
        (e.g., ``llama-3-3-70b-instruct-quantization-fp8/meta-llama/Llama-3.3-70B-Instruct``).
        This method finds the registered ID that contains ``self.model_name``
        and caches it for use in API calls.
        """
        try:
            import requests as req
            models_url = f"{base_url.rstrip('/')}/models"
            resp = req.get(models_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for model in data:
                    model_id = model.get("id", "")
                    if self.model_name in model_id:
                        self._resolved_model_name = model_id
                        logger.info(
                            "Resolved model name: %s -> %s",
                            self.model_name, model_id,
                        )
                        return
                logger.warning(
                    "Model '%s' not found in LlamaStack /v1/models. "
                    "Available: %s",
                    self.model_name,
                    [m.get("id") for m in data],
                )
        except Exception as e:
            logger.warning("Failed to resolve model name from LlamaStack: %s", e)

    def _get_model_specific_instructions(self) -> str:
        """70B-specific tool-calling and response format instructions."""
        return (
            "**LLAMA-SPECIFIC TOOL-CALLING RULES:**\n"
            "- ALWAYS use the function calling API — NEVER output tool calls as JSON text.\n"
            "- Call tools one at a time; wait for each result before deciding the next step.\n"
            "- If a tool returns an error, do NOT retry with identical arguments.\n\n"
            "**METRIC DISCOVERY (MANDATORY):**\n"
            "- Before calling execute_promql, ALWAYS call search_metrics or "
            "search_metrics_by_category to discover the exact metric name.\n"
            "- For AI/ML metrics (KV cache, tokens, latency, throughput, inference, "
            "batch size, TTFT, model serving, vLLM), use search_metrics_by_category "
            "with category='gpu_ai' and max_results=20.\n"
            "- NEVER guess metric names. NEVER use node_memory_* or generic OS metrics "
            "for AI/ML queries — use the vllm_* prefixed metrics discovered via search.\n\n"
            "**RESPONSE FORMAT (MANDATORY):**\n"
            "- Use markdown formatting (bold headers, bullet lists).\n"
            "- Include a **Technical Details** section with: the PromQL query used, "
            "metric source, and key data points.\n"
            "- Provide operational context: what the result means, whether it is "
            "normal or concerning, and actionable recommendations."
        )

    def _detect_text_tool_calls(self, text: str, tool_names: List[str]) -> bool:
        """Detect if the model described a tool call in text instead of using the function calling API.

        Llama models may output tool calls as plain text in several formats:
        - Tool Call: header (case-insensitive)
        - tool_name( function-call syntax
        - {"name": "tool_name"} JSON pattern

        Returns True if a text-based tool call pattern is detected.
        """
        if not text or not tool_names:
            return False

        # Check for "Tool Call:" header pattern
        if re.search(r'Tool Call:', text, re.IGNORECASE):
            return True

        # Check for tool_name( pattern — function invocation syntax
        for name in tool_names:
            pattern = r'\b' + re.escape(name) + r'\s*\('
            if re.search(pattern, text):
                return True

        # Check for JSON-style {"name": "tool_name"} pattern (Llama-specific)
        for name in tool_names:
            pattern = r'"name"\s*:\s*"' + re.escape(name) + r'"'
            if re.search(pattern, text):
                return True

        return False

    def _convert_tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format."""
        tools = self._get_mcp_tools()
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })
        return openai_tools

    def chat(
        self,
        user_question: str,
        namespace: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Chat with Llama 70B using LlamaStack OpenAI-compatible API.

        Clean loop modeled after OpenAIChatBot — no fabrication guard,
        no forced tool_choice, just straightforward tool calling.
        """
        try:
            # Create system prompt
            system_prompt = self._create_system_prompt(namespace)

            # LlamaStack expects the full model name (override preserves it)
            model_id = self._extract_model_name()

            # Prepare messages - start with system prompt
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history if provided
            if conversation_history:
                logger.info(f"📜 Adding {len(conversation_history)} messages from conversation history")
                messages.extend(conversation_history)

            # Add current user question
            messages.append({"role": "user", "content": user_question})

            # Convert tools to OpenAI format
            openai_tools = self._convert_tools_to_openai_format()

            # Extract tool names for text-tool-call detection
            tool_names = [t["function"]["name"] for t in openai_tools]

            # Iterative tool calling loop
            max_iterations = 30
            iteration = 0
            # Counter for text-tool-call nudges.  The 70B model often
            # outputs tool calls as text; tool_choice="required" forces
            # a real function call on the retry.  Allow multiple cycles
            # (text → force → real call → text → force → …) up to a cap.
            text_tool_call_retries = 0
            _MAX_TEXT_TOOL_CALL_RETRIES = 3
            tool_choice = "auto"
            consecutive_tool_tracker = {"name": None, "count": 0}

            while iteration < max_iterations:
                iteration += 1
                logger.info(f"🤖 Llama70B tool calling iteration {iteration}")

                if progress_callback:
                    progress_callback(f"🤖 Thinking... (iteration {iteration})")

                # Call LlamaStack via OpenAI SDK
                # parallel_tool_calls=False — vLLM rejects multi-tool-call requests.
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                    temperature=0,
                    timeout=LLM_TIMEOUT_SECONDS,
                )

                # Reset to auto after each call so forced tool calls don't persist
                tool_choice = "auto"

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                message = choice.message

                # Enforce single tool call — vLLM rejects multi-tool-call messages.
                if message.tool_calls and len(message.tool_calls) > 1:
                    logger.warning(
                        "Model returned %d tool calls — trimming to first only (%s)",
                        len(message.tool_calls),
                        message.tool_calls[0].function.name,
                    )
                    message.tool_calls = [message.tool_calls[0]]

                logger.debug(
                    "Llama70B iteration %d: finish_reason=%s, has_tool_calls=%s, content_len=%d",
                    iteration, finish_reason, bool(message.tool_calls),
                    len(message.content or '')
                )

                # Convert message to dict format for conversation history
                message_dict = {
                    "role": "assistant",
                    "content": message.content
                }

                # Add tool calls if present
                if message.tool_calls:
                    message_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]

                # Add assistant's response to conversation
                messages.append(message_dict)

                # If model wants to use tools, execute them.
                # Check tool_calls presence regardless of finish_reason —
                # LlamaStack may return finish_reason='stop' even with tool calls.
                if message.tool_calls:
                    logger.info(f"🤖 Llama70B requesting {len(message.tool_calls)} tool(s)")

                    tool_results = []
                    tool_loop_detected = False
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments
                        tool_id = tool_call.id

                        # Parse arguments
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            tool_args = {}

                        if self._check_tool_loop(tool_name, consecutive_tool_tracker):
                            tool_loop_detected = True
                            break

                        if progress_callback:
                            progress_callback(f"🔧 Using tool: {tool_name}")

                        # Get tool result with automatic truncation
                        tool_result = self._get_tool_result(tool_name, tool_args, namespace=namespace)

                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": tool_result
                        })

                    if tool_loop_detected:
                        return (
                            "I got stuck in a loop calling the same tool repeatedly. "
                            "Please try rephrasing your question or being more specific."
                        )

                    # Add tool results to conversation
                    messages.extend(tool_results)

                    # Truncate conversation safely (preserves tool-call/result pairs)
                    messages = self._truncate_messages(
                        messages, keep_system_prompt=True,
                        max_messages=40, target_messages=28,
                    )

                    # Continue loop
                    continue

                else:
                    final_response = message.content or ''

                    # Text-tool-call guard: model wrote a tool call as text
                    # instead of using the function calling API.  Force
                    # tool_choice="required" on the retry so the next response
                    # MUST be a proper function call.  Allow up to
                    # _MAX_TEXT_TOOL_CALL_RETRIES cycles (the model typically
                    # needs one force per tool in its chain: search → promql).
                    if self._detect_text_tool_calls(final_response, tool_names):
                        text_tool_call_retries += 1
                        if text_tool_call_retries > _MAX_TEXT_TOOL_CALL_RETRIES:
                            logger.warning(
                                "Llama70B exceeded %d text-tool-call retries "
                                "(iteration %d). Returning fallback.",
                                _MAX_TEXT_TOOL_CALL_RETRIES, iteration,
                            )
                            return (
                                "I wasn't able to retrieve data for this query. "
                                "Please try rephrasing your question or being more "
                                "specific about what information you need."
                            )
                        logger.warning(
                            "Llama70B output a tool call as text instead of using "
                            "the function calling API (iteration %d, retry %d/%d). "
                            "Forcing tool_choice=required.",
                            iteration, text_tool_call_retries,
                            _MAX_TEXT_TOOL_CALL_RETRIES,
                        )
                        # Remove the assistant message that contained the text
                        # tool call — it's noise that bloats context and buries
                        # the formatting instructions in the system prompt.
                        messages.pop()
                        tool_choice = "required"
                        continue

                    if not final_response.strip():
                        logger.warning(
                            "Llama70B returned empty response at iteration %d.",
                            iteration,
                        )
                        return (
                            "I wasn't able to retrieve data for this query. "
                            "Please try rephrasing your question or being more "
                            "specific about what information you need."
                        )

                    logger.info(f"Llama70B tool calling completed in {iteration} iterations")
                    return final_response

            # Hit max iterations
            logger.warning(f"Hit max iterations ({max_iterations})")
            return "Analysis incomplete. Please try a more specific question."

        except Exception as e:
            logger.error(f"Error in Llama70B chat: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error during Llama70B tool calling: {str(e)}"
