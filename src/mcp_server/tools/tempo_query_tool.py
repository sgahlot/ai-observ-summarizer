"""Tempo Query Tool - Query traces from Tempo instance in observability-hub namespace.

This module provides async MCP tools for interacting with Tempo traces:
- query_tempo_tool: Search traces by service, operation, time range
- get_trace_details_tool: Get detailed trace information by trace ID
- list_trace_services_tool: List available services with traces
- analyze_traces_tool: Analyze trace patterns and summarize using LLM
"""

import asyncio
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import logging

from .observability_vllm_tools import _resp, resolve_time_range
from core.llm_client import summarize_with_llm, build_prompt
from core.response_validator import ResponseType

logger = logging.getLogger(__name__)


class TempoQueryTool:
    """Tool for querying Tempo traces with async support."""

    def __init__(self):
        # Tempo configuration based on TMP/observability/tempo/values.yaml
        # Use environment variable for local development or OpenShift deployment
        import os
        self.tempo_url = os.getenv(
            "TEMPO_URL",
            "https://tempo-tempostack-gateway.observability-hub.svc.cluster.local:8080"
        )
        # Tenant ID not needed for standard Jaeger API endpoints
        self.tenant_id = os.getenv("TEMPO_TENANT_ID", "dev")
        self.namespace = "observability-hub"
        
    def _get_service_account_token(self) -> str:
        """Get the service account token for authentication."""
        try:
            with open('/var/run/secrets/kubernetes.io/serviceaccount/token', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback for local development
            return "dev-token"

    def _extract_root_service(self, trace: Dict[str, Any]) -> str:
        """Extract the root service name from a Jaeger trace."""
        if "processes" in trace and trace["processes"]:
            # Get the first process (usually the root service)
            first_process = list(trace["processes"].values())[0]
            return first_process.get("serviceName", "unknown")
        return "unknown"

    def _calculate_duration(self, trace: Dict[str, Any]) -> int:
        """Calculate trace duration in milliseconds from Jaeger trace."""
        if "spans" in trace and trace["spans"]:
            # Find the span with the earliest start time and latest end time
            min_start = float('inf')
            max_end = 0

            for span in trace["spans"]:
                start_time = span.get("startTime", 0)
                duration = span.get("duration", 0)
                end_time = start_time + duration

                min_start = min(min_start, start_time)
                max_end = max(max_end, end_time)

            if min_start != float('inf') and max_end > min_start:
                # Convert from microseconds to milliseconds
                return int((max_end - min_start) / 1000)

        return 0

    def _get_start_time(self, trace: Dict[str, Any]) -> int:
        """Get the start time of the trace from Jaeger trace."""
        if "spans" in trace and trace["spans"]:
            # Find the earliest start time
            min_start = float('inf')
            for span in trace["spans"]:
                start_time = span.get("startTime", 0)
                min_start = min(min_start, start_time)

            if min_start != float('inf'):
                return int(min_start)

        return 0
        
    async def query_traces(
        self,
        query: str,
        start_time: str,
        end_time: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Query traces from Tempo.

        Args:
            query: TraceQL query (e.g., "service.name=my-service")
            start_time: Start time in ISO format
            end_time: End time in ISO format
            limit: Maximum number of traces to return
        """
        try:
            # Convert times to Unix timestamps
            start_ts = int(datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp())
            end_ts = int(datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp())

            headers = {
                "X-Scope-OrgID": self.tenant_id,
                "Content-Type": "application/json"
            }

            # Add service account token if running in cluster
            try:
                token = self._get_service_account_token()
                if token and token != "dev-token":
                    headers["Authorization"] = f"Bearer {token}"
            except Exception as e:
                logger.debug(f"No service account token available: {e}")

            # Use Jaeger API format (working endpoint from template)
            search_url = f"{self.tempo_url}/api/traces/v1/{self.tenant_id}/api/traces"

            # Parse TraceQL query to extract service name
            # Simple parsing for service.name=value format
            service_name = None
            if "service.name=" in query:
                # Extract service name from TraceQL query like "service.name=my-service"
                parts = query.split("service.name=")
                if len(parts) > 1:
                    service_name = parts[1].split()[0].strip('"\'')
            elif "service=" in query:
                # Handle direct service= format
                parts = query.split("service=")
                if len(parts) > 1:
                    service_name = parts[1].split()[0].strip('"\'')

            # Build Jaeger API parameters
            params = {
                "start": start_ts * 1000000,  # Jaeger expects microseconds
                "end": end_ts * 1000000,
                "limit": limit
            }

            if service_name:
                params["service"] = service_name
            else:
                # If we can't parse service name, try a general search
                params["service"] = "llama-3-2-3b-instruct-predictor"  # fallback

            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                try:
                    logger.info(f"Querying Jaeger API: {search_url}")
                    response = await client.get(search_url, params=params, headers=headers)

                    if response.status_code == 200:
                        jaeger_data = response.json()

                        # Convert Jaeger format to our expected format
                        traces = []
                        if "data" in jaeger_data and jaeger_data["data"]:
                            for trace in jaeger_data["data"]:
                                # Extract basic trace info from Jaeger format
                                trace_info = {
                                    "traceID": trace.get("traceID", ""),
                                    "rootServiceName": self._extract_root_service(trace),
                                    "durationMs": self._calculate_duration(trace),
                                    "spanCount": len(trace.get("spans", [])),
                                    "startTime": self._get_start_time(trace)
                                }
                                traces.append(trace_info)

                        return {
                            "success": True,
                            "traces": traces,
                            "total": len(traces),
                            "query": query,
                            "time_range": f"{start_time} to {end_time}",
                            "api_endpoint": search_url,
                            "service_queried": service_name or "llama-3-2-3b-instruct-predictor"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Jaeger API query failed: HTTP {response.status_code} - {response.text}",
                            "query": query,
                            "api_endpoint": search_url
                        }

                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error querying Jaeger API: {str(e)}",
                        "query": query,
                        "api_endpoint": search_url
                    }

        except Exception as e:
            logger.error(f"Tempo query error: {e}")
            error_msg = str(e)

            # Provide helpful error message for common connection issues
            if "nodename nor servname provided" in error_msg or "Name or service not known" in error_msg:
                error_msg = f"Tempo service not reachable at {self.tempo_url}. This is expected when running locally. Deploy to OpenShift to access Tempo."
            elif "Connection refused" in error_msg:
                error_msg = f"Tempo service refused connection at {self.tempo_url}. Check if Tempo is running in the observability-hub namespace."

            return {
                "success": False,
                "error": error_msg,
                "query": query,
                "tempo_url": self.tempo_url
            }

    async def get_trace_details(self, trace_id: str) -> Dict[str, Any]:
        """Get detailed trace information."""
        try:
            trace_url = f"{self.tempo_url}/api/traces/v1/{self.tenant_id}/{trace_id}"
            
            headers = {
                "X-Scope-OrgID": self.tenant_id,
                "Content-Type": "application/json"
            }

            # Add service account token if running in cluster
            try:
                token = self._get_service_account_token()
                if token and token != "dev-token":
                    headers["Authorization"] = f"Bearer {token}"
            except Exception as e:
                logger.debug(f"No service account token available: {e}")
            
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(trace_url, headers=headers)
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "trace": response.json()
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Trace fetch failed: {response.status_code} - {response.text}"
                    }
                    
        except Exception as e:
            logger.error(f"Trace details error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# MCP Tool functions for FastMCP integration
async def query_tempo_tool(
    query: str,
    start_time: str,
    end_time: str,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    MCP tool function for querying Tempo traces.
    
    Args:
        query: TraceQL query string (e.g., "service.name=my-service" or "service=my-service")
        start_time: Start time in ISO format (e.g., "2024-01-01T00:00:00Z")
        end_time: End time in ISO format (e.g., "2024-01-01T23:59:59Z")
        limit: Maximum number of traces to return (default: 20)
    
    Returns:
        List of trace information
    """
    tempo_tool = TempoQueryTool()
    result = await tempo_tool.query_traces(query, start_time, end_time, limit)
    
    if result["success"]:
        content = f"🔍 **Tempo Query Results**\n\n"
        content += f"**Query**: `{result['query']}`\n"
        content += f"**Time Range**: {result['time_range']}\n"
        content += f"**Found**: {result['total']} traces\n\n"
        
        if result["traces"]:
            content += "**Traces**:\n"
            for i, trace in enumerate(result["traces"][:5], 1):  # Show first 5
                trace_id = trace.get("traceID", "unknown")
                service_name = trace.get("rootServiceName", "unknown")
                duration = trace.get("durationMs", 0)
                content += f"{i}. **{service_name}** - {trace_id} ({duration}ms)\n"
            
            if len(result["traces"]) > 5:
                content += f"... and {len(result['traces']) - 5} more traces\n"
        else:
            content += "No traces found matching the query.\n"
            
        return [{"type": "text", "text": content}]
    else:
        # Use the detailed error message from the tool if available
        error_content = result['error']

        # Add helpful deployment instructions for local development
        if "not reachable" in result['error'] or "not known" in result['error']:
            error_content += "\n\n💡 **Note**: To use Tempo queries, deploy the MCP server to OpenShift where Tempo is running.\n"
            error_content += "   Local development cannot access the Tempo service in the observability-hub namespace.\n"

        return [{"type": "text", "text": error_content}]


async def get_trace_details_tool(trace_id: str) -> List[Dict[str, Any]]:
    """
    MCP tool function for getting detailed trace information.
    
    Args:
        trace_id: The trace ID to retrieve details for
    
    Returns:
        Detailed trace information including spans
    """
    tempo_tool = TempoQueryTool()
    result = await tempo_tool.get_trace_details(trace_id)
    
    if result["success"]:
        trace_data = result["trace"]
        
        # Format trace details for display
        content = f"🔍 **Trace Details for {trace_id}**\n\n"
        
        if isinstance(trace_data, list) and trace_data:
            # Jaeger format - list of spans
            content += f"**Total Spans**: {len(trace_data)}\n\n"
            content += "**Spans**:\n"
            
            for i, span in enumerate(trace_data[:10], 1):  # Show first 10 spans
                span_id = span.get("spanID", "unknown")
                operation = span.get("operationName", "unknown")
                service = span.get("serviceName", "unknown")
                duration = span.get("duration", 0)
                start_time = span.get("startTime", 0)
                
                content += f"{i}. **{operation}** ({service})\n"
                content += f"   - Span ID: {span_id}\n"
                content += f"   - Duration: {duration}μs\n"
                content += f"   - Start Time: {start_time}\n"
                
                # Show tags if available
                tags = span.get("tags", [])
                if tags:
                    content += f"   - Tags: {len(tags)} tags\n"
                
                content += "\n"
            
            if len(trace_data) > 10:
                content += f"... and {len(trace_data) - 10} more spans\n"
        else:
            content += "No span data available for this trace.\n"
            
        return [{"type": "text", "text": content}]
    else:
        error_content = f"Failed to get trace details: {result['error']}"
        return [{"type": "text", "text": error_content}]


# Note: list_trace_services_tool removed because the /api/traces/v1/{tenant_id}/services endpoint
# is not available in this TempoStack deployment. Use query_tempo_tool to search for traces instead.

async def analyze_traces_tool(
    query: str,
    start_time: str,
    end_time: str,
    summarize_model_id: str = "llama-3.1-8b",
    api_key: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    MCP tool function for analyzing trace patterns and summarizing using LLM.
    
    Args:
        query: TraceQL query string
        start_time: Start time in ISO format
        end_time: End time in ISO format
        summarize_model_id: LLM model to use for analysis
        api_key: API key for LLM service
        limit: Maximum number of traces to analyze
    
    Returns:
        Analysis summary of trace patterns
    """
    try:
        # First, query traces
        tempo_tool = TempoQueryTool()
        result = await tempo_tool.query_traces(query, start_time, end_time, limit)
        
        if not result["success"]:
            return [{"type": "text", "text": f"Failed to query traces: {result['error']}"}]
        
        traces = result.get("traces", [])
        
        if not traces:
            return [{"type": "text", "text": "No traces found for analysis"}]
        
        # Analyze trace patterns
        analysis_data = {
            "total_traces": len(traces),
            "services": {},
            "operations": {},
            "duration_stats": {
                "min": float('inf'),
                "max": 0,
                "total": 0
            },
            "error_traces": 0
        }
        
        for trace in traces:
            service = trace.get("rootServiceName", "unknown")
            duration = trace.get("durationMs", 0)
            
            # Count services
            analysis_data["services"][service] = analysis_data["services"].get(service, 0) + 1
            
            # Duration statistics (convert from ms to μs for consistency)
            duration_us = duration * 1000
            if duration_us > 0:
                analysis_data["duration_stats"]["min"] = min(analysis_data["duration_stats"]["min"], duration_us)
                analysis_data["duration_stats"]["max"] = max(analysis_data["duration_stats"]["max"], duration_us)
                analysis_data["duration_stats"]["total"] += duration_us
        
        # Calculate average duration
        if analysis_data["duration_stats"]["total"] > 0:
            analysis_data["duration_stats"]["avg"] = analysis_data["duration_stats"]["total"] / len(traces)
        else:
            analysis_data["duration_stats"]["avg"] = 0
        
        # Build analysis prompt
        prompt = f"""
Analyze the following trace data and provide insights:

Trace Analysis Summary:
- Total traces: {analysis_data['total_traces']}
- Services involved: {list(analysis_data['services'].keys())}
- Duration stats: min={analysis_data['duration_stats']['min']:.2f}μs, max={analysis_data['duration_stats']['max']:.2f}μs, avg={analysis_data['duration_stats']['avg']:.2f}μs

Service distribution:
{json.dumps(analysis_data['services'], indent=2)}

Please provide:
1. Key performance insights
2. Service health assessment
3. Potential issues or bottlenecks
4. Recommendations for improvement
"""
        
        # Generate LLM summary
        summary = summarize_with_llm(
            prompt,
            summarize_model_id,
            ResponseType.VLLM_ANALYSIS,
            api_key
        )
        
        # Combine analysis data and summary
        content = f"📊 **Trace Analysis Results**\n\n{summary}\n\n**Detailed Analysis Data:**\n{json.dumps(analysis_data, indent=2)}"
        
        return [{"type": "text", "text": content}]
        
    except Exception as e:
        return [{"type": "text", "text": f"Failed to analyze traces: {str(e)}"}]