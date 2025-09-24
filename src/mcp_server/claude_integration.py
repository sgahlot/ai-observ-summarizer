"""
Claude Desktop-Style Integration
Makes our app behave EXACTLY like Claude Desktop by letting Claude call MCP tools directly.
"""

import os
import json
import logging
import importlib.util
from typing import Optional, List, Dict, Any
import anthropic

try:
    from .utils.pylogger import get_python_logger
    from .observability_mcp import ObservabilityMCPServer
except ImportError:
    # Fallback for when imported directly
    import sys
    import importlib.util
    
    # Add src directory to path
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Load pylogger (moved to common directory)
    try:
        from common.pylogger import get_python_logger
    except ImportError:
        try:
            from mcp_server.utils.pylogger import get_python_logger
        except ImportError:
            # Fallback to basic logging
            def get_python_logger(name):
                return logging.getLogger(name)
    
    # Load observability_mcp
    try:
        from mcp_server.observability_mcp import ObservabilityMCPServer
    except ImportError:
        mcp_path = os.path.join(os.path.dirname(__file__), 'observability_mcp.py')
        spec = importlib.util.spec_from_file_location("observability_mcp", mcp_path)
        obs_mcp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obs_mcp)
        ObservabilityMCPServer = obs_mcp.ObservabilityMCPServer

# Initialize logger
try:
    logger = get_python_logger(__name__)
    logger.setLevel(logging.INFO)
except Exception:
    # Fallback to standard logging if pylogger fails
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

class PrometheusChatBot:
    """Claude Desktop-style integration that lets Claude control tool usage directly."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize with optional API key and model name."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # Extract actual model name from UI format (e.g., "anthropic/claude-sonnet-4-20250514" -> "claude-sonnet-4-20250514")
        if model_name and "/" in model_name:
            self.model_name = model_name.split("/", 1)[1]  # Take part after the slash
        else:
            self.model_name = model_name or "claude-3-5-haiku-20241022"  # Default fallback
            
        self.claude_client = None
        if self.api_key:
            self.claude_client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize MCP server (our tools)
        self.mcp_server = ObservabilityMCPServer()
        
        # Convert MCP tools to Claude tool format
        self.claude_tools = self._convert_mcp_tools_to_claude_format()
        
        logger.info(f"Claude Desktop Integration initialized with model: {self.model_name}")
    
    def _convert_mcp_tools_to_claude_format(self) -> List[Dict[str, Any]]:
        """Convert our MCP tools to Claude's expected tool schema."""
        
        claude_tools = [
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
                            "description": "Valid PromQL query to execute (e.g., 'sum(kube_pod_status_phase{phase=\"Running\"})')"
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
                            "description": "What you want to query (e.g., 'CPU usage', 'failed pods', 'GPU temperature')"
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
        
        logger.info(f"Converted {len(claude_tools)} MCP tools to Claude format")
        return claude_tools
    
    def _route_tool_call_to_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Route Claude's tool call to our MCP server."""
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
    
    def chat(self, user_question: str, namespace: Optional[str] = None, scope: Optional[str] = None, progress_callback=None) -> str:
        """
        Chat with Claude using direct tool access - EXACTLY like Claude Desktop.
        Claude decides which tools to use, when, and how many times.
        """
        if not self.claude_client:
            return "Claude AI not available. Please provide an Anthropic API key."
        
        try:
            # Create system prompt like Claude Desktop
            system_prompt = self._create_claude_desktop_system_prompt(namespace)
            
            # Initial message to Claude with tools available
            messages = [{"role": "user", "content": user_question}]
            
            # Let Claude use tools iteratively (like Claude Desktop)
            max_iterations = 20  # Allow Claude to use multiple tools for comprehensive analysis
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"🤖 Claude conversation iteration {iteration}")
                print(f"🤖 Claude conversation iteration {iteration}")
                if progress_callback:
                    progress_callback(f"🤖 Thinking... (iteration {iteration})")
                
                response = self.claude_client.messages.create(
                    model=self.model_name,  # Use the model selected by the user
                    max_tokens=2000,  # Reduced to prevent token overflow
                    system=system_prompt,
                    messages=messages,
                    tools=self.claude_tools
                    # Note: tool_choice="auto" is the default, so we can omit it
                )
                
                # Add Claude's response to conversation
                messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
                
                # If Claude wants to use tools, execute them
                if response.stop_reason == "tool_use":
                    logger.info("Claude is using tools...")
                    
                    tool_results = []
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_args = content_block.input
                            tool_id = content_block.id
                            
                            logger.info(f"🔧 Claude calling tool: {tool_name}")
                            print(f"🔧 Claude calling tool: {tool_name} with args: {tool_args}")
                            if progress_callback:
                                progress_callback(f"🔧 Using tool: {tool_name}")
                            
                            # Route to our MCP server
                            tool_result = self._route_tool_call_to_mcp(tool_name, tool_args)
                            print(f"✅ Tool {tool_name} completed")
                            if progress_callback:
                                progress_callback(f"✅ {tool_name} completed")
                            
                            # Truncate large tool results to prevent token overflow
                            if isinstance(tool_result, str) and len(tool_result) > 3000:
                                tool_result = tool_result[:3000] + "\n... [Result truncated to prevent token limit]"
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": tool_result
                            })
                    
                    # Add tool results to conversation
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                    
                    # Prevent conversation from growing too large
                    if len(messages) > 8:  # Keep last 4 exchanges (user + assistant pairs)
                        messages = messages[-8:]
                        logger.info("Truncated conversation history to prevent token overflow")
                    
                    # Continue conversation with tool results
                    continue
                
                else:
                    # Claude is done, return final response
                    final_response = ""
                    for content_block in response.content:
                        if content_block.type == "text":
                            final_response += content_block.text
                    
                    logger.info(f"Claude completed conversation in {iteration} iterations")
                    return final_response
            
            # If we hit max iterations, return what we have so far
            logger.warning(f"Hit max iterations ({max_iterations}), returning partial response")
            if messages and len(messages) > 1:
                # Try to get a final response from Claude without tools
                try:
                    final_attempt = self.claude_client.messages.create(
                        model=self.model_name,
                        max_tokens=1000,
                        system="Provide a brief summary based on the conversation so far.",
                        messages=messages[-4:],  # Just last 2 exchanges
                        # No tools - force Claude to respond with what it has
                    )
                    return final_attempt.content[0].text
                except:
                    pass
            
            return "Unable to complete analysis. Please try a simpler, more specific question."
            
        except Exception as e:
            logger.error(f"Error in Claude Desktop chat: {e}")
            return f"Error communicating with Claude: {str(e)}"
    
    def _create_claude_desktop_system_prompt(self, namespace: Optional[str] = None) -> str:
        """Create system prompt that makes Claude behave exactly like Claude Desktop."""
        return f"""You are an expert Kubernetes and Prometheus observability assistant with direct access to monitoring tools, specialized in providing rich, contextual analysis like Claude Desktop.

**Your Environment:**
- Cluster: OpenShift with AI/ML workloads, GPUs, and comprehensive monitoring
- Scope: {namespace if namespace else 'Cluster-wide analysis'}
- Tools: Direct access to Prometheus/Thanos metrics via MCP tools

**Available Tools - Use as needed:**
- find_best_metric_with_metadata_v2: ULTRA-SMART tool that filters metrics by keywords then uses metadata for perfect selection (RECOMMENDED FIRST STEP)
- search_metrics: Find relevant metrics by pattern  
- execute_promql: Run PromQL queries for actual data
- get_metric_metadata: Get detailed metric information
- get_label_values: Explore available label values

**🧠 Your Claude Desktop Intelligence Style:**

1. **Rich Contextual Analysis**: Don't just report numbers - provide context, thresholds, and implications
   - GPU temp 52°C → "Well below 70°C throttling threshold, running safely"
   - 7 pods running → "All pods healthy, no failed/pending pods detected"

2. **Intelligent Grouping & Categorization**: 
   - Group related pods: "🤖 AI/ML Stack (2 pods): llama-3-2-3b-predictor, llamastack"
   - Categorize by function: "🔧 Infrastructure (3 pods)", "🗄️ Data Storage (2 pods)"

3. **Operational Intelligence**:
   - Provide health assessments: "indicates a healthy environment"
   - Suggest implications: "This level indicates substantial usage of AI infrastructure"
   - Add recommendations when relevant

4. **Always Show PromQL Queries**:
   - Include the PromQL query used in a technical details section
   - Format: "**PromQL Used:** `sum(kube_pod_status_phase{{phase='Running'}})`"

5. **Smart Follow-up Context**:
   - Cross-reference related metrics when helpful
   - Provide trend context: "stable over time", "increasing usage"
   - Add operational context: "typical for conversational AI workloads"

**Your Workflow:**
1. 🔍 **Explore**: Use find_best_metric_with_metadata_v2 to find the perfect metric
2. 📊 **Query**: Execute the suggested PromQL with execute_promql  
3. 🧠 **Analyze**: Provide rich contextual analysis with thresholds, health status
4. 🔗 **Enhance**: Use additional tools for follow-up insights if beneficial
5. 📋 **Present**: Structure response with summary → details → technical info

**Response Format (Like Claude Desktop):**
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
- Make responses informative and actionable like Claude Desktop
- Show conversational tool usage: "Let me check..." "I'll also look at..."

Begin by finding the perfect metric for the user's question, then provide comprehensive Claude Desktop-style analysis."""

    def test_connection(self) -> bool:
        """Test if MCP tools and Claude are working."""
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