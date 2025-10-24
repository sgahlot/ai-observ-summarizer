"""
Generic Chat Bot - Multi-Provider Support

This module provides a provider-agnostic chat bot that can work with any model
from MODEL_CONFIG, supporting both external APIs (OpenAI, Google, Anthropic) 
and local models.

Key capabilities:
- Dynamic model selection based on MODEL_CONFIG
- Provider-specific authentication handling
- Tool calling support (convert MCP tools to provider-specific format)
- Conversation management
- Error handling and fallbacks
"""

import os
import json
import logging
import importlib.util
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

try:
    from .utils.pylogger import get_python_logger
    from .observability_mcp import ObservabilityMCPServer
    from ..core.llm_client import summarize_with_llm
    from ..core.config import MODEL_CONFIG, LLAMA_STACK_URL, LLAMA_STACK_CHAT_URL, LLM_API_TOKEN, VERIFY_SSL
    from ..core.response_validator import ResponseType
except ImportError:
    # Fallback for when imported directly
    import sys
    import importlib.util
    
    # Add src directory to path
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Load dependencies
    try:
        from common.pylogger import get_python_logger
    except ImportError:
        try:
            from mcp_server.utils.pylogger import get_python_logger
        except ImportError:
            # Fallback to basic logging
            def get_python_logger(name):
                return logging.getLogger(name)
    
    try:
        from mcp_server.observability_mcp import ObservabilityMCPServer
    except ImportError:
        mcp_path = os.path.join(os.path.dirname(__file__), 'observability_mcp.py')
        spec = importlib.util.spec_from_file_location("observability_mcp", mcp_path)
        obs_mcp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obs_mcp)
        ObservabilityMCPServer = obs_mcp.ObservabilityMCPServer
    
try:
    from core.llm_client import summarize_with_llm
    from core.config import MODEL_CONFIG, LLAMA_STACK_URL, LLAMA_STACK_CHAT_URL, LLM_API_TOKEN, VERIFY_SSL
    from core.response_validator import ResponseType
except ImportError:
    # Fallback imports
    def summarize_with_llm(*args, **kwargs):
        raise NotImplementedError("LLM client not available")

    # Try to get MODEL_CONFIG from environment
    import json
    try:
        model_config_str = os.getenv("MODEL_CONFIG", "{}")
        MODEL_CONFIG = json.loads(model_config_str)
    except Exception:
        MODEL_CONFIG = {}

    # Fallback config values
    LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL", "http://localhost:8321/v1/openai")
    LLAMA_STACK_CHAT_URL = os.getenv("LLAMA_STACK_CHAT_URL", "http://localhost:8321/v1")
    LLM_API_TOKEN = os.getenv("LLM_API_TOKEN", "")
    VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() in ("true", "1", "yes")

    # Create a simple ResponseType fallback
    class ResponseType:
        GENERAL_CHAT = "general_chat"
        OPENSHIFT_ANALYSIS = "openshift_analysis"
        VLLM_ANALYSIS = "vllm_analysis"

# Initialize logger
try:
    logger = get_python_logger(__name__)
    logger.setLevel(logging.INFO)
except Exception:
    # Fallback to standard logging if pylogger fails
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


class GenericChatBot:
    """Provider-agnostic chat bot that works with any model from MODEL_CONFIG."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize with model name and optional API key."""
        self.model_name = model_name
        self.model_config = self._get_model_config()
        self.is_external = self.model_config.get("external", False) if self.model_config else False
        self.provider = self.model_config.get("provider", "local") if self.model_config else "local"
        self.api_key = api_key or self._get_default_api_key()

        # Initialize MCP server (our tools)
        self.mcp_server = ObservabilityMCPServer()

        # Convert MCP tools to provider-specific format
        self.provider_tools = self._convert_mcp_tools_to_provider_format()

        logger.info(f"Generic Chat Bot initialized with model: {self.model_name} (provider: {self.provider}, external: {self.is_external})")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration from MODEL_CONFIG."""
        logger.debug(f"MODEL_CONFIG type: {type(MODEL_CONFIG)}, content: {MODEL_CONFIG}")
        logger.debug(f"Looking for model: {self.model_name}")
        
        if not isinstance(MODEL_CONFIG, dict):
            logger.warning("MODEL_CONFIG is not a dictionary")
            return {}
        
        model_config = MODEL_CONFIG.get(self.model_name, {})
        if not model_config:
            logger.warning(f"Model {self.model_name} not found in MODEL_CONFIG")
            logger.debug(f"Available models: {list(MODEL_CONFIG.keys())}")
            return {}
        
        logger.debug(f"Found model config: {model_config}")
        return model_config
    
    def _supports_reliable_tool_calling(self) -> bool:
        """
        Determine if the model has reliable tool calling support (>75% accuracy).

        Tool calling accuracy benchmarks (BFCL):
        - Llama 3.1-8B: 82.6%
        - Llama 3.1-70B: ~85%
        - Llama 3.3-70B: ~85%
        - Llama 3.2-3B: 67.0%
        - Llama 3.2-1B: 25.7%
        """
        # External models (OpenAI, Google, Anthropic) have good tool calling
        if self.is_external:
            return True

        # For local models, check the model name
        model_lower = self.model_name.lower()

        # Llama 3.1 series (8B+) has good tool calling
        if "llama-3.1" in model_lower or "llama-3-1" in model_lower:
            if any(size in model_lower for size in ["8b", "70b", "405b"]):
                return True

        # Llama 3.3 series (70B) has good tool calling
        if "llama-3.3" in model_lower or "llama-3-3" in model_lower:
            if "70b" in model_lower:
                return True

        # Llama 3.2 3B and smaller have poor tool calling - use deterministic parsing
        if "llama-3.2" in model_lower or "llama-3-2" in model_lower:
            return False

        # Unknown models - default to deterministic parsing for safety
        logger.warning(f"Unknown model tool calling capability for {self.model_name}, defaulting to deterministic parsing")
        return False

    def _get_default_api_key(self) -> Optional[str]:
        """Get default API key based on provider."""
        if not self.model_config:
            return None
        provider = self.model_config.get("provider", "openai")
        
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif provider == "google":
            return os.getenv("GOOGLE_API_KEY")
        elif provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        else:
            # For local models, no API key needed
            return None
    
    def _convert_mcp_tools_to_provider_format(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to provider-specific format."""
        # Get the base MCP tools
        mcp_tools = self._get_mcp_tools()
        
        if self.provider == "openai":
            return self._convert_to_openai_functions(mcp_tools)
        elif self.provider == "google":
            return self._convert_to_google_functions(mcp_tools)
        elif self.provider == "anthropic":
            return self._convert_to_anthropic_tools(mcp_tools)
        else:
            # For local models, use prompt-based approach
            return self._convert_to_prompt_based(mcp_tools)
    
    def _get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get the base MCP tools that we want to expose."""
        return [
            {
                "name": "search_metrics",
                "description": "Search for Prometheus metrics by pattern (regex supported). Essential for discovering relevant metrics.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern or regex for metric names (e.g., 'pod', 'gpu', 'memory')"
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "get_metric_metadata",
                "description": "Get detailed metadata about a specific metric including type, help text, available labels, and query examples.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "description": "Exact name of the metric to get metadata for"
                        }
                    },
                    "required": ["metric_name"]
                }
            },
            {
                "name": "execute_promql",
                "description": "Execute a PromQL query against Prometheus/Thanos and get results. Use this to get actual metric values.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": "Valid PromQL query to execute (use metrics discovered through search_metrics or find_best_metric tools)"
                        },
                        "time_range": {
                            "type": "string",
                            "description": "Optional time range (e.g., '5m', '1h', '1d')",
                            "default": "now"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_label_values", 
                "description": "Get all possible values for a specific label across metrics.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "label_name": {
                            "type": "string",
                            "description": "Name of the label to get values for (e.g., 'namespace', 'phase', 'job')"
                        }
                    },
                    "required": ["label_name"]
                }
            },
            {
                "name": "suggest_queries",
                "description": "Get PromQL query suggestions based on intent or description.",
                "input_schema": {
                    "type": "object", 
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "What you want to query about the infrastructure (describe in natural language)"
                        }
                    },
                    "required": ["intent"]
                }
            },
            {
                "name": "explain_results",
                "description": "Get human-readable explanation of query results and metrics data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "Query results or metrics data to explain"
                        }
                    },
                    "required": ["data"]
                }
            }
        ]
    
    def _convert_to_openai_functions(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format."""
        functions = []
        for tool in tools:
            function = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
            functions.append(function)
        return functions
    
    def _convert_to_google_functions(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Google function calling format."""
        # Google uses similar format to OpenAI
        return self._convert_to_openai_functions(tools)
    
    def _convert_to_anthropic_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic tool format."""
        # Anthropic uses the same format as our MCP tools
        return tools
    
    def _convert_to_prompt_based(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to prompt-based format for local models."""
        # For local models, we'll include tool descriptions in the prompt
        return tools
    
    def _route_tool_call_to_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Route tool call to our MCP server."""
        try:
            # Import MCP client helper to call our tools
            import sys
            ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui')
            if ui_path not in sys.path:
                sys.path.insert(0, ui_path)
            
            try:
                from mcp_client_helper import MCPClientHelper
            except ImportError:
                # Load mcp_client_helper directly
                mcp_helper_path = os.path.join(ui_path, 'mcp_client_helper.py')
                spec = importlib.util.spec_from_file_location("mcp_client_helper", mcp_helper_path)
                mcp_helper = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mcp_helper)
                MCPClientHelper = mcp_helper.MCPClientHelper
            
            mcp_client = MCPClientHelper()
            
            # Call the tool via MCP
            result = mcp_client.call_tool_sync(tool_name, arguments)
            
            if result and len(result) > 0:
                return result[0]['text']
            else:
                return f"No results returned from {tool_name}"
                
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"
    
    def chat(self, user_question: str, namespace: Optional[str] = None, scope: Optional[str] = None, progress_callback: Optional[Callable] = None) -> str:
        """
        Chat with the selected model using direct tool access.
        The model decides which tools to use, when, and how many times.
        """
        if not self.api_key and self.model_config and self.model_config.get("requiresApiKey", False):
            return f"API key required for {self.provider} model {self.model_name}. Please provide an API key."

        try:
            # Create system prompt
            system_prompt = self._create_system_prompt(namespace)

            # Route based on model capabilities
            # Use tool calling for models with good tool calling support
            # Use deterministic parsing for smaller models that struggle with tool calling

            if self._supports_reliable_tool_calling():
                # Models with good tool calling support (>75% accuracy)
                return self._chat_with_tools(user_question, system_prompt, progress_callback)
            else:
                # Smaller models - use deterministic parsing for reliability
                logger.info(f"Using deterministic parsing for {self.model_name}")
                return self._chat_with_prompt(user_question, system_prompt, progress_callback)

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error communicating with {self.provider} model: {str(e)}"
    
    def _chat_with_tools(self, user_question: str, system_prompt: str, progress_callback: Optional[Callable] = None) -> str:
        """Handle chat with native tool calling support (OpenAI-compatible or Anthropic)."""

        # For local models (non-external), use the chat completions endpoint with tools
        if not self.is_external:
            return self._chat_with_local_tools(user_question, system_prompt, progress_callback)

        # For external models, delegate to the appropriate provider
        if self.provider == "anthropic":
            # Anthropic has its own implementation in PrometheusChatBot
            # For now, fall back to simple LLM call
            messages = [{"role": "user", "content": user_question}]
            prompt = f"{system_prompt}\n\nUser Question: {user_question}"

            try:
                response = summarize_with_llm(
                    prompt=prompt,
                    summarize_model_id=self.model_name,
                    response_type=ResponseType.GENERAL_CHAT,
                    api_key=self.api_key,
                    messages=messages
                )
                return response
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                return f"Error generating response: {str(e)}"

        # For OpenAI and Google, implement tool calling
        elif self.provider in ["openai", "google"]:
            return self._chat_with_external_tools(user_question, system_prompt, progress_callback)

        else:
            # Fallback for unknown providers
            messages = [{"role": "user", "content": user_question}]
            prompt = f"{system_prompt}\n\nUser Question: {user_question}"

            try:
                response = summarize_with_llm(
                    prompt=prompt,
                    summarize_model_id=self.model_name,
                    response_type=ResponseType.GENERAL_CHAT,
                    api_key=self.api_key,
                    messages=messages
                )
                return response
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                return f"Error generating response: {str(e)}"

    def _chat_with_local_tools(self, user_question: str, system_prompt: str, progress_callback: Optional[Callable] = None) -> str:
        """Handle tool calling for local LlamaStack models using OpenAI-compatible API."""
        import requests
        # Config variables are imported at the top of the file

        # Prepare messages with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]

        # Convert tools to OpenAI format
        tools = self._convert_to_openai_functions(self._get_mcp_tools())
        openai_tools = [{"type": "function", "function": tool} for tool in tools]

        # Iterative tool calling loop (like Anthropic's implementation)
        max_iterations = 30  # Match Anthropic's limit for comprehensive analysis
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🤖 Tool calling iteration {iteration}")

            if progress_callback:
                progress_callback(f"🤖 Thinking... (iteration {iteration})")

            try:
                # Call LlamaStack chat completions endpoint with tools
                headers = {"Content-Type": "application/json"}
                if LLM_API_TOKEN:
                    headers["Authorization"] = f"Bearer {LLM_API_TOKEN}"

                # For chat completions, use the full MODEL_CONFIG key (e.g., "meta-llama/Llama-3.2-3B-Instruct")
                # For local models, the serviceName format is "llama-3-2-3b-instruct/meta-llama/Llama-3.2-3B-Instruct"
                # Construct it from serviceName and model_name
                service_name = self.model_config.get("serviceName", "")
                if service_name and "/" not in service_name:
                    # If serviceName doesn't have the full format, construct it
                    model_id = f"{service_name}/{self.model_name}"
                elif service_name:
                    # Already in full format
                    model_id = service_name
                else:
                    # Fall back to model_name
                    model_id = self.model_name

                payload = {
                    "model": model_id,
                    "messages": messages,
                    "tools": openai_tools,
                    "temperature": 0
                }

                # Use the dedicated chat completions URL
                chat_url = f"{LLAMA_STACK_CHAT_URL}/chat/completions"

                response = requests.post(
                    chat_url,
                    headers=headers,
                    json=payload,
                    verify=VERIFY_SSL
                )

                response.raise_for_status()
                result = response.json()

                choice = result['choices'][0]
                finish_reason = choice.get('finish_reason', '')
                message = choice['message']

                # Add assistant's response to conversation
                messages.append(message)

                # If model wants to use tools, execute them
                if finish_reason == 'tool_calls' and 'tool_calls' in message:
                    logger.info(f"Model is using {len(message['tool_calls'])} tool(s)")

                    tool_results = []
                    for tool_call in message['tool_calls']:
                        tool_name = tool_call['function']['name']
                        tool_args_str = tool_call['function']['arguments']
                        tool_id = tool_call['id']

                        logger.info(f"🔧 Calling tool: {tool_name}")
                        if progress_callback:
                            progress_callback(f"🔧 Using tool: {tool_name}")

                        # Parse arguments
                        import json
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            tool_args = {}

                        # Route to MCP server
                        tool_result = self._route_tool_call_to_mcp(tool_name, tool_args)

                        # Truncate large results
                        if isinstance(tool_result, str) and len(tool_result) > 3000:
                            tool_result = tool_result[:3000] + "\n... [Result truncated]"

                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": tool_result
                        })

                    # Add tool results to conversation
                    messages.extend(tool_results)

                    # Limit conversation history to prevent token overflow
                    if len(messages) > 10:
                        # Keep system message + last 8 messages
                        messages = [messages[0]] + messages[-8:]

                    # Continue loop to get final response
                    continue

                else:
                    # Model is done, return final response
                    final_response = message.get('content', '')
                    logger.info(f"Tool calling completed in {iteration} iterations")
                    return final_response

            except Exception as e:
                logger.error(f"Error in tool calling iteration {iteration}: {e}")
                return f"Error during tool calling: {str(e)}"

        # Hit max iterations
        logger.warning(f"Hit max iterations ({max_iterations})")
        return "Analysis incomplete. Please try a more specific question."

    def _chat_with_external_tools(self, user_question: str, system_prompt: str, progress_callback: Optional[Callable] = None) -> str:
        """Handle tool calling for external providers (OpenAI, Google)."""

        if self.provider == "openai":
            return self._chat_with_openai_tools(user_question, system_prompt, progress_callback)
        elif self.provider == "google":
            # TODO: Implement Google Gemini tool calling
            return self._chat_with_google_tools(user_question, system_prompt, progress_callback)
        else:
            # Fallback for unknown external providers
            messages = [{"role": "user", "content": user_question}]
            prompt = f"{system_prompt}\n\nUser Question: {user_question}"

            try:
                response = summarize_with_llm(
                    prompt=prompt,
                    summarize_model_id=self.model_name,
                    response_type=ResponseType.GENERAL_CHAT,
                    api_key=self.api_key,
                    messages=messages
                )
                return response
            except Exception as e:
                logger.error(f"Error calling external LLM: {e}")
                return f"Error generating response: {str(e)}"

    def _chat_with_openai_tools(self, user_question: str, system_prompt: str, progress_callback: Optional[Callable] = None) -> str:
        """Handle tool calling for OpenAI models using their official SDK."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("OpenAI SDK not installed. Install with: pip install openai")
            return "Error: OpenAI SDK not installed. Please install it with: pip install openai"

        # Initialize OpenAI client
        client = OpenAI(api_key=self.api_key)

        # Prepare messages with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]

        # Convert tools to OpenAI format
        tools = self._convert_to_openai_functions(self._get_mcp_tools())
        openai_tools = [{"type": "function", "function": tool} for tool in tools]

        # Get model name from config
        model_name = self.model_config.get("modelName", "gpt-4o-mini")

        # Iterative tool calling loop
        max_iterations = 30
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🤖 OpenAI tool calling iteration {iteration}")

            if progress_callback:
                progress_callback(f"🤖 Thinking... (iteration {iteration})")

            try:
                # Call OpenAI API using SDK
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=openai_tools,
                    temperature=0
                )

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
                    logger.info(f"OpenAI is using {len(message.tool_calls)} tool(s)")

                    tool_results = []
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments
                        tool_id = tool_call.id

                        logger.info(f"🔧 Calling tool: {tool_name}")
                        if progress_callback:
                            progress_callback(f"🔧 Using tool: {tool_name}")

                        # Parse arguments
                        import json
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            tool_args = {}

                        # Route to MCP server
                        tool_result = self._route_tool_call_to_mcp(tool_name, tool_args)

                        # Truncate large results
                        if isinstance(tool_result, str) and len(tool_result) > 3000:
                            tool_result = tool_result[:3000] + "\n... [Result truncated]"

                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": tool_result
                        })

                    # Add tool results to conversation
                    messages.extend(tool_results)

                    # Limit conversation history
                    if len(messages) > 10:
                        messages = [messages[0]] + messages[-8:]

                    # Continue loop
                    continue

                else:
                    # Model is done, return final response
                    final_response = message.content or ''

                    # Strip markdown code fences if OpenAI wrapped the response
                    # OpenAI sometimes wraps responses in ```markdown ... ``` for complex queries
                    if final_response.startswith('```') and final_response.endswith('```'):
                        # Remove opening fence (```markdown or ```)
                        lines = final_response.split('\n')
                        if lines[0].startswith('```'):
                            lines = lines[1:]
                        # Remove closing fence
                        if lines and lines[-1].strip() == '```':
                            lines = lines[:-1]
                        final_response = '\n'.join(lines).strip()

                    logger.info(f"OpenAI tool calling completed in {iteration} iterations")
                    return final_response

            except Exception as e:
                logger.error(f"Error in OpenAI tool calling iteration {iteration}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return f"Error during OpenAI tool calling: {str(e)}"

        # Hit max iterations
        logger.warning(f"Hit max iterations ({max_iterations})")
        return "Analysis incomplete. Please try a more specific question."

    def _chat_with_google_tools(self, user_question: str, system_prompt: str, progress_callback: Optional[Callable] = None) -> str:
        """Handle tool calling for Google Gemini using their official SDK."""
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error("Google Generative AI SDK not installed. Install with: pip install google-generativeai")
            return "Error: Google Generative AI SDK not installed. Please install it with: pip install google-generativeai"

        # Configure API key
        genai.configure(api_key=self.api_key)

        # Get model name from config
        model_name = self.model_config.get("modelName", "gemini-2.5-flash")

        # Convert MCP tools to Google Gemini format
        gemini_tools = self._convert_to_gemini_tools_sdk_format(self._get_mcp_tools())

        # Initialize the model with tools
        model = genai.GenerativeModel(
            model_name=model_name,
            tools=gemini_tools
        )

        # Start a chat session
        chat = model.start_chat(enable_automatic_function_calling=False)

        # Send initial message with system prompt
        initial_message = f"{system_prompt}\n\n{user_question}"

        # Iterative tool calling loop
        max_iterations = 30
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🤖 Google Gemini tool calling iteration {iteration}")

            if progress_callback:
                progress_callback(f"🤖 Thinking... (iteration {iteration})")

            try:
                # Send message to model
                if iteration == 1:
                    # First iteration - send the initial question
                    response = chat.send_message(
                        initial_message,
                        generation_config=genai.GenerationConfig(temperature=0)
                    )
                else:
                    # Subsequent iterations - send function responses
                    response = chat.send_message(
                        function_responses,
                        generation_config=genai.GenerationConfig(temperature=0)
                    )

                # Check if model wants to use tools
                if response.candidates[0].content.parts:
                    parts = response.candidates[0].content.parts

                    # Check for function calls
                    has_function_calls = any(hasattr(part, 'function_call') and part.function_call for part in parts)

                    if has_function_calls:
                        logger.info("Google Gemini is using tools")

                        # Build function responses for next iteration
                        function_responses = []
                        for part in parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                func_call = part.function_call
                                tool_name = func_call.name
                                tool_args = dict(func_call.args)

                                logger.info(f"🔧 Calling tool: {tool_name}")
                                if progress_callback:
                                    progress_callback(f"🔧 Using tool: {tool_name}")

                                # Route to MCP server
                                tool_result = self._route_tool_call_to_mcp(tool_name, tool_args)

                                # Truncate large results
                                if isinstance(tool_result, str) and len(tool_result) > 3000:
                                    tool_result = tool_result[:3000] + "\n... [Result truncated]"

                                # Create function response for Gemini SDK
                                function_responses.append(
                                    genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response={"content": tool_result}
                                        )
                                    )
                                )

                        # Continue loop to send function responses
                        continue

                    else:
                        # Model is done, extract final text response
                        final_response = ""
                        for part in parts:
                            if hasattr(part, 'text') and part.text:
                                final_response += part.text

                        logger.info(f"Google Gemini tool calling completed in {iteration} iterations")
                        return final_response
                else:
                    return "Error: No response from Google Gemini"

            except Exception as e:
                logger.error(f"Error in Google Gemini tool calling iteration {iteration}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return f"Error during Google Gemini tool calling: {str(e)}"

        # Hit max iterations
        logger.warning(f"Hit max iterations ({max_iterations})")
        return "Analysis incomplete. Please try a more specific question."

    def _convert_to_gemini_functions(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to Google Gemini function calling format (for REST API)."""
        all_declarations = []

        for tool in mcp_tools:
            # MCP tools use 'input_schema', Gemini uses 'parameters'
            parameters = tool.get("input_schema", tool.get("parameters", {}))

            gemini_function = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": parameters
            }
            all_declarations.append(gemini_function)

        # Gemini expects a single tools object with all function declarations
        if all_declarations:
            return [{"functionDeclarations": all_declarations}]

        return []

    def _convert_to_gemini_tools_sdk_format(self, mcp_tools: List[Dict[str, Any]]) -> List:
        """Convert MCP tools to Google Gemini SDK format."""
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error("Google Generative AI SDK not installed")
            return []

        # Convert to SDK tool format
        sdk_tools = []
        for tool in mcp_tools:
            # MCP tools use 'input_schema', Gemini SDK uses 'parameters'
            parameters = tool.get("input_schema", tool.get("parameters", {}))

            # Create a function declaration for the SDK
            sdk_tools.append(
                genai.protos.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            k: genai.protos.Schema(
                                type=self._json_type_to_gemini_type(v.get("type", "string")),
                                description=v.get("description", "")
                            )
                            for k, v in parameters.get("properties", {}).items()
                        },
                        required=parameters.get("required", [])
                    )
                )
            )

        return sdk_tools

    def _json_type_to_gemini_type(self, json_type: str):
        """Convert JSON schema type to Gemini proto type."""
        try:
            import google.generativeai as genai
        except ImportError:
            return None

        type_mapping = {
            "string": genai.protos.Type.STRING,
            "number": genai.protos.Type.NUMBER,
            "integer": genai.protos.Type.INTEGER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT
        }
        return type_mapping.get(json_type, genai.protos.Type.STRING)

    def _chat_with_prompt(self, user_question: str, system_prompt: str, progress_callback: Optional[Callable] = None) -> str:
        """Handle chat with prompt-based approach for local models."""
        # For local models without native tool calling, we manually orchestrate:
        # 1. Detect what tools are needed based on the question
        # 2. Execute those tools
        # 3. Pass tool results to the LLM for intelligent formatting

        if progress_callback:
            progress_callback("🔍 Analyzing your question...")

        # Extract key concepts from the question
        question_lower = user_question.lower()

        # Identify what type of metrics are being asked about
        tool_results = []
        query_type = None  # Track which type of query we're handling

        # Memory-related questions (check FIRST to avoid "usage" matching CPU)
        if any(term in question_lower for term in ['memory', 'mem', 'ram']):
            query_type = 'memory'
            try:
                if progress_callback:
                    progress_callback("📊 Querying memory metrics...")

                # Use similar queries to what Anthropic uses
                memory_query = "sum(container_memory_usage_bytes{}) / 1024 / 1024 / 1024"
                query_result = self._route_tool_call_to_mcp("execute_promql", {"query": memory_query})

                tool_results.append({
                    "tool": "execute_promql",
                    "query": memory_query,
                    "result": query_result
                })

            except Exception as e:
                logger.error(f"Error calling MCP tools: {e}")
                return f"Error analyzing memory metrics: {str(e)}"

        # CPU-related questions
        elif any(term in question_lower for term in ['cpu', 'usage', 'utilization']):
            query_type = 'cpu'
            try:
                if progress_callback:
                    progress_callback("📊 Querying CPU metrics...")

                # Use the better cluster-level CPU metric (like Anthropic does)
                cpu_query = "cluster:container_cpu_usage:ratio"
                query_result = self._route_tool_call_to_mcp("execute_promql", {"query": cpu_query})

                tool_results.append({
                    "tool": "execute_promql",
                    "query": cpu_query,
                    "result": query_result
                })

            except Exception as e:
                logger.error(f"Error calling MCP tools: {e}")
                return f"Error analyzing CPU metrics: {str(e)}"

        else:
            # For general questions, provide a helpful response
            return f"""I understand you're asking about: "{user_question}"

To provide accurate metrics and insights, please ask about specific aspects like:
- CPU usage or utilization
- Memory usage
- Pod counts or status
- Network metrics
- Storage metrics

I'll query the Prometheus metrics and provide detailed analysis."""

        # Now format the results - parse the data and create a clean response
        if tool_results:
            if progress_callback:
                progress_callback("✨ Formatting response...")

            # For CPU queries, extract the value and format nicely
            if query_type == 'cpu':
                try:
                    result_text = tool_results[0]['result']
                    query_used = tool_results[0]['query']

                    # Try to extract numeric value from the result
                    # The result might be JSON or text, try to parse it
                    import json
                    import re

                    cpu_value = None

                    # Try to find a numeric value in the result
                    # Look for patterns like "value": 0.0427 or just a number
                    if isinstance(result_text, str):
                        # Try JSON first
                        try:
                            result_json = json.loads(result_text)
                            if isinstance(result_json, dict):
                                # Look for value in different possible locations
                                if 'results' in result_json and len(result_json['results']) > 0:
                                    first_result = result_json['results'][0]
                                    if 'value' in first_result and len(first_result['value']) > 1:
                                        cpu_value = float(first_result['value'][1])
                        except (json.JSONDecodeError, ValueError, KeyError):
                            pass

                        # If JSON parsing failed, try regex
                        if cpu_value is None:
                            # Look for patterns like: "value": [timestamp, "0.0427"]
                            match = re.search(r'"value"\s*:\s*\[\s*[\d.]+\s*,\s*"([\d.]+)"', result_text)
                            if match:
                                cpu_value = float(match.group(1))
                            else:
                                # Try to find any decimal number
                                match = re.search(r'(\d+\.\d+)', result_text)
                                if match:
                                    cpu_value = float(match.group(1))

                    if cpu_value is not None:
                        # Format as percentage (the ratio is already 0-1, so multiply by 100)
                        cpu_percent = cpu_value * 100

                        # Determine usage level
                        if cpu_percent < 20:
                            usage_level = "low"
                            recommendation = "The cluster has plenty of available computational resources."
                        elif cpu_percent < 60:
                            usage_level = "moderate"
                            recommendation = "CPU utilization is within normal operating range."
                        elif cpu_percent < 80:
                            usage_level = "high"
                            recommendation = "Consider monitoring for potential performance impacts."
                        else:
                            usage_level = "very high"
                            recommendation = "CPU resources are heavily utilized. Consider scaling or optimization."

                        response = f"""🖥️ CPU Usage Overview

Current CPU Utilization: {cpu_percent:.2f}%

📊 Detailed Breakdown:
- The cluster is currently using approximately {cpu_percent:.2f}% of its total CPU capacity
- This indicates a {usage_level} level of CPU utilization
- The metric represents the ratio of CPU cores being used across the entire cluster

PromQL Used: `{query_used}`

Operational Insights:
- {recommendation}
- This measurement reflects cluster-wide container CPU usage"""

                        return response
                    else:
                        # Couldn't parse the value, show a simplified response
                        return f"""🖥️ CPU Usage Analysis

I retrieved the CPU metrics but had trouble parsing the exact value.

PromQL Used: `{query_used}`

Raw result: {result_text[:500]}

Please check the Prometheus query directly for detailed values."""

                except Exception as e:
                    logger.error(f"Error parsing CPU result: {e}")
                    return f"Error formatting CPU metrics: {str(e)}\n\nRaw result: {tool_results[0]['result'][:500]}"

            # For memory queries, similar approach
            elif query_type == 'memory':
                try:
                    result_text = tool_results[0]['result']
                    query_used = tool_results[0]['query']

                    # Try to extract numeric value from the result
                    import json
                    import re

                    memory_gb = None

                    # Try to find a numeric value in the result
                    if isinstance(result_text, str):
                        # Try JSON first
                        try:
                            result_json = json.loads(result_text)
                            if isinstance(result_json, dict):
                                # Look for value in different possible locations
                                if 'results' in result_json and len(result_json['results']) > 0:
                                    first_result = result_json['results'][0]
                                    if 'value' in first_result and len(first_result['value']) > 1:
                                        memory_gb = float(first_result['value'][1])
                        except (json.JSONDecodeError, ValueError, KeyError):
                            pass

                        # If JSON parsing failed, try regex
                        if memory_gb is None:
                            # Look for patterns like: "value": [timestamp, "1720.5"]
                            match = re.search(r'"value"\s*:\s*\[\s*[\d.]+\s*,\s*"([\d.]+)"', result_text)
                            if match:
                                memory_gb = float(match.group(1))
                            else:
                                # Try to find any decimal number
                                match = re.search(r'(\d+\.\d+)', result_text)
                                if match:
                                    memory_gb = float(match.group(1))

                    if memory_gb is not None:
                        # Provide context based on memory usage
                        if memory_gb < 100:
                            usage_level = "low"
                            recommendation = "Memory usage is minimal, plenty of capacity available."
                        elif memory_gb < 500:
                            usage_level = "moderate"
                            recommendation = "Memory usage is within normal operating range."
                        elif memory_gb < 1000:
                            usage_level = "high"
                            recommendation = "Memory usage is elevated. Monitor for potential issues."
                        else:
                            usage_level = "very high"
                            recommendation = "Substantial memory consumption. Review container memory requests and limits."

                        response = f"""🧠 Memory Usage Analysis

Total Memory Used: {memory_gb:.1f} GB

📊 Detailed Breakdown:
- Cluster containers are using approximately {memory_gb:.1f} GB of memory
- This indicates a {usage_level} level of memory utilization
- The metric represents total container memory usage across the cluster

PromQL Used: `{query_used}`

Operational Insights:
- {recommendation}
- This measurement reflects cluster-wide container memory consumption"""

                        return response
                    else:
                        # Couldn't parse the value, show a simplified response
                        return f"""🧠 Memory Usage Analysis

I retrieved the memory metrics but had trouble parsing the exact value.

PromQL Used: `{query_used}`

Raw result: {result_text[:500]}

Please check the Prometheus query directly for detailed values."""

                except Exception as e:
                    logger.error(f"Error parsing memory result: {e}")
                    return f"Error formatting memory metrics: {str(e)}\n\nRaw result: {tool_results[0]['result'][:500]}"

        return "No metrics found for your question. Please try asking about CPU, memory, pods, or other infrastructure metrics."
    
    def _get_tools_description(self) -> str:
        """Get a description of available tools for prompt-based models."""
        descriptions = []
        for tool in self.provider_tools:
            descriptions.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(descriptions)
    
    def _create_system_prompt(self, namespace: Optional[str] = None) -> str:
        """Create system prompt that makes the model behave like an expert observability assistant."""
        return f"""You are an expert Kubernetes and Prometheus observability assistant. 

🎯 **PRIMARY RULE: ANSWER ONLY WHAT THE USER ASKS. DO NOT EXPLORE BEYOND THEIR SPECIFIC QUESTION.**

You have access to monitoring tools and should provide focused, targeted responses.

**Your Environment:**
- Cluster: OpenShift with AI/ML workloads, GPUs, and comprehensive monitoring
- Scope: {namespace if namespace else 'Cluster-wide analysis'}
- Tools: Direct access to Prometheus/Thanos metrics via MCP tools

**Available Tools:**
- search_metrics: Pattern-based metric search - use for broad exploration
- execute_promql: Execute PromQL queries for actual data
- get_metric_metadata: Get detailed information about specific metrics
- get_label_values: Get available label values
- suggest_queries: Get PromQL suggestions based on user intent
- explain_results: Get human-readable explanation of query results

**🧠 Your Intelligence Style:**

1. **Rich Contextual Analysis**: Don't just report numbers - provide context, thresholds, and implications
   - For temperature metrics → compare against known safe operating ranges
   - For count metrics → provide health context and status interpretation

2. **Intelligent Grouping & Categorization**: 
   - Group related pods: "🤖 AI/ML Stack (2 pods): llama-3-2-3b-predictor, llamastack"
   - Categorize by function: "🔧 Infrastructure (3 pods)", "🗄️ Data Storage (2 pods)"

3. **Operational Intelligence**:
   - Provide health assessments: "indicates a healthy environment"
   - Suggest implications: "This level indicates substantial usage of AI infrastructure"
   - Add recommendations when relevant

4. **Always Show PromQL Queries**:
   - Include the PromQL query used in a technical details section
   - Format: "**PromQL Used:** `[the actual query you executed]`"

5. **Smart Follow-up Context**:
   - Cross-reference related metrics when helpful
   - Provide trend context: "stable over time", "increasing usage"
   - Add operational context: "typical for conversational AI workloads"

**CRITICAL: ANSWER ONLY WHAT THE USER ASKS - DON'T EXPLORE EVERYTHING**

**Your Workflow (FOCUSED & DIRECT):**
1. 🎯 **STOP AND THINK**: What exactly is the user asking for?
2. 🔍 **FIND ONCE**: Use search_metrics to find the specific metric
3. 📊 **QUERY ONCE**: Execute the PromQL query for that specific metric
4. 📋 **ANSWER**: Provide the specific answer to their question - DONE!

**STRICT RULES - FOLLOW FOR ANY QUESTION:**
1. Extract key search terms from their question
2. Call search_metrics with those terms to find relevant metrics  
3. Call execute_promql with the best metric found
4. Report the specific answer to their question - DONE!

**CORE PRINCIPLES:**
- **BE THOROUGH BUT FOCUSED**: Use as many tools as needed to answer comprehensively
- **STOP when you have enough data** to answer the question well
- **ANSWER ONLY** what they asked for
- **NO EXPLORATION** beyond their specific question
- **BE DIRECT** - don't analyze everything about a topic

**Response Format:**
```
🤖 [Emoji + Summary Title]
[Key Numbers & Summary]

[Rich contextual analysis with operational insights]

**Technical Details:**
- **PromQL Used:** `your_query_here`
- **Metric Source:** metric_name_here  
- **Data Points:** X samples over Y timeframe
```

**Critical Rules:**
- ALWAYS include the PromQL query in technical details
- ALWAYS use tools to get real data - never make up numbers
- Provide operational context and health assessments
- Use emojis and categorization for clarity
- Make responses informative and actionable
- Show conversational tool usage: "Let me check..." "I'll also look at..."

Begin by finding the perfect metric for the user's question, then provide comprehensive analysis."""
    
    def test_connection(self) -> bool:
        """Test if MCP tools and model are working."""
        try:
            # Test MCP server
            if hasattr(self.mcp_server, 'mcp') and hasattr(self.mcp_server.mcp, '_tool_manager'):
                tool_count = len(self.mcp_server.mcp._tool_manager._tools)
                if tool_count > 0:
                    logger.info(f"MCP server working with {tool_count} tools")
                    return True
                else:
                    logger.error("MCP server has no registered tools")
                    return False
            else:
                logger.error("MCP server not properly initialized")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
