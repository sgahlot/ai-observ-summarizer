"""
Llama Chat Bot Implementation (via LlamaStack)

This module provides Llama-specific implementation using LlamaStack's OpenAI-compatible API.
"""

import json
import re
from typing import Optional, List, Dict, Any, Callable

from .base import BaseChatBot
from chatbots.tool_executor import ToolExecutor
from core.config import LLAMA_STACK_URL, LLM_API_TOKEN
from common.pylogger import get_python_logger

logger = get_python_logger()


class LlamaChatBot(BaseChatBot):
    """Llama implementation using LlamaStack with OpenAI-compatible API."""

    def _get_api_key(self) -> Optional[str]:
        """Local models don't require API keys."""
        return None

    def _get_max_tool_result_length(self) -> int:
        """Llama 3.1 supports 128K token context - 8K chars is reasonable."""
        return 8000

    # Tools that Llama should have access to. Excludes admin/config tools,
    # LLM-chaining tools (analyze_vllm, chat_openshift, etc.), and the
    # recursive `chat` tool to stay within the 14K token context limit.
    _TOOL_ALLOWLIST = {
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

    def _get_tool_allowlist(self) -> set:
        """Limit tools to reduce context usage for Llama's constrained context."""
        return self._TOOL_ALLOWLIST

    def _extract_model_name(self) -> str:
        """LlamaStack expects the full model name including provider prefix.

        Override the base class method to return the full name.
        """
        return self.model_name

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        tool_executor: ToolExecutor = None
    ):
        super().__init__(model_name, api_key, tool_executor)

        # Import OpenAI SDK (LlamaStack is OpenAI-compatible)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=f"{LLAMA_STACK_URL}/chat/completions".replace("/chat/completions", ""),
                api_key=LLM_API_TOKEN or "dummy"
            )
        except ImportError:
            logger.error("OpenAI SDK not installed. Install with: pip install openai")
            self.client = None

    def _get_model_specific_instructions(self) -> str:
        """No separate model-specific block — merged into _get_base_prompt."""
        return ""

    def _get_base_prompt(self, namespace=None) -> str:
        """Compact system prompt for Llama's constrained context window."""
        scope = self._format_scope_line(namespace)
        ns_directive = self._format_namespace_directive(namespace)

        return f"""You are an expert Kubernetes and Prometheus observability assistant.

**Scope:** {scope}
{ns_directive}
**CRITICAL — Tool Calling:**
- ALWAYS call tools via the function calling API to get real data.
- NEVER fabricate data or make up numbers.
- NEVER output tool calls as JSON text — use the function calling mechanism.

**Tool Selection:**
- Metrics/pods/GPU/CPU/memory → execute_promql (primary), search_metrics, suggest_queries
- Traces/spans/latency → chat_tempo_tool, query_tempo_tool, get_trace_details_tool
- Logs/errors → get_correlated_logs
- Alert investigation → execute_promql with ALERTS metric, then korrel8r_get_correlated
- Correlation → korrel8r_get_correlated, korrel8r_query_objects

**PromQL Patterns:**
- CPU: sum(rate(container_cpu_usage_seconds_total[5m])) by (pod, namespace)
- Memory: sum(container_memory_usage_bytes) by (pod, namespace)
- GPU temp: avg(DCGM_FI_DEV_GPU_TEMP) or avg(habanalabs_temperature_onchip)
- GPU util: avg(DCGM_FI_DEV_GPU_UTIL) or avg(habanalabs_utilization)
- GPU power: avg(DCGM_FI_DEV_POWER_USAGE) or avg(habanalabs_power_mW) / 1000
- Pod status: kube_pod_status_phase{{phase="Running"}} == 1
- Failing pods: kube_pod_container_status_waiting_reason{{reason=~"CrashLoopBackOff|ImagePullBackOff"}} == 1
- Always use aggregation (sum/avg/max) and group by (pod, namespace)
- Use rate() for counters, append == 1 for boolean metrics
- Use regex for pod names: pod=~"name.*" (pods have hash suffixes)

**Response Format:**
- Use markdown (bold, lists) — no code block wrappers
- Include PromQL used, metric source, and data points in a Technical Details section
- Provide operational context and actionable insights"""

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

        # Check for "Tool Call:" header pattern (shared with Google)
        if re.search(r'Tool Call:', text, re.IGNORECASE):
            return True

        # Check for tool_name( pattern — function invocation syntax (shared with Google)
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
        """Chat with Llama using LlamaStack OpenAI-compatible API."""
        if not self.client:
            return "Error: OpenAI SDK not installed. Please install it with: pip install openai"

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
            nudge_retried = False
            tool_choice = "auto"

            while iteration < max_iterations:
                iteration += 1
                logger.info(f"🤖 LlamaStack tool calling iteration {iteration}")

                if progress_callback:
                    progress_callback(f"🤖 Thinking... (iteration {iteration})")

                # Call LlamaStack via OpenAI SDK
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice=tool_choice,
                    temperature=0
                )

                # Reset to auto after each call so forced tool calls don't persist
                tool_choice = "auto"

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                message = choice.message

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

                # If model wants to use tools, execute them
                if finish_reason == 'tool_calls' and message.tool_calls:
                    logger.info(f"🤖 LlamaStack requesting {len(message.tool_calls)} tool(s)")

                    tool_results = []
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments
                        tool_id = tool_call.id

                        # Parse arguments
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            tool_args = {}

                        if progress_callback:
                            progress_callback(f"🔧 Using tool: {tool_name}")

                        # Get tool result with automatic truncation (logging handled in base class)
                        tool_result = self._get_tool_result(tool_name, tool_args, namespace=namespace)

                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": tool_result
                        })

                    # Add tool results to conversation
                    messages.extend(tool_results)

                    # Truncate conversation safely (preserves tool-call/result pairs)
                    messages = self._truncate_messages(messages, keep_system_prompt=True)

                    # Continue loop
                    continue

                else:
                    final_response = message.content or ''

                    if not nudge_retried:
                        # Guard 1: Fabrication — model returned stop on iteration 1
                        # without ever calling tools (likely fabricated response)
                        if iteration == 1:
                            logger.warning(
                                "Llama returned finish_reason=stop on iteration 1 "
                                "without calling any tools — possible fabrication. "
                                "Nudging model to use tools."
                            )
                            nudge_retried = True
                            tool_choice = "required"
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You MUST use the provided tools to answer this question. "
                                    "Do NOT fabricate data. Call the appropriate tool now."
                                )
                            })
                            continue

                        # Guard 2: Text tool calls — model wrote tool call as text
                        # instead of using the function calling API
                        if self._detect_text_tool_calls(final_response, tool_names):
                            logger.warning(
                                "Llama output a tool call as text instead of using "
                                "the function calling API. Nudging model to retry."
                            )
                            nudge_retried = True
                            tool_choice = "required"
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You wrote a tool call as text instead of using the "
                                    "function calling API. Do NOT output JSON tool calls "
                                    "as text. Use the function calling mechanism to invoke "
                                    "the tool."
                                )
                            })
                            continue

                    # Normal completion (or already nudged once)
                    logger.info(f"LlamaStack tool calling completed in {iteration} iterations")
                    return final_response

            # Hit max iterations
            logger.warning(f"Hit max iterations ({max_iterations})")
            return "Analysis incomplete. Please try a more specific question."

        except Exception as e:
            logger.error(f"Error in LlamaStack chat: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error during LlamaStack tool calling: {str(e)}"
