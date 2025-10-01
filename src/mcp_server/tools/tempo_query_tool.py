"""Tempo Query Tool - Query traces from Tempo instance in observability-hub namespace.

This module provides async MCP tools for interacting with Tempo traces:
- query_tempo_tool: Search traces by service, operation, time range
- get_trace_details_tool: Get detailed trace information by trace ID
- chat_tempo_tool: Conversational interface for Tempo trace analysis
"""

import httpx
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

# Note: Removed unused imports for analyze_traces_tool that was removed

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
            # Fallback for local development - use TEMPO_TOKEN if available
            import os
            tempo_token = os.getenv("TEMPO_TOKEN")
            if tempo_token:
                return tempo_token
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

    async def get_available_services(self) -> List[str]:
        """Get list of available services from Tempo/Jaeger."""
        try:
            services_url = f"{self.tempo_url}/api/traces/v1/{self.tenant_id}/api/services"
            
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
                logger.info(f"Getting available services from: {services_url}")
                response = await client.get(services_url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    services = data.get("data", [])
                    logger.info(f"Found {len(services)} available services: {services}")
                    return services
                else:
                    logger.error(f"Failed to get services: HTTP {response.status_code} - {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting available services: {e}")
            return []

    async def _query_single_service(self, search_url: str, params: Dict[str, Any], headers: Dict[str, str], 
                                   query: str, start_time: str, end_time: str, duration_filter: int) -> Dict[str, Any]:
        """Query traces from a single service."""
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            try:
                logger.info(f"Querying Jaeger API: {search_url}")
                logger.info(f"Query parameters: {params}")
                response = await client.get(search_url, params=params, headers=headers)

                if response.status_code == 200:
                    jaeger_data = response.json()
                    logger.info(f"Jaeger API response status: {response.status_code}")
                    if "data" in jaeger_data:
                        logger.info(f"Number of traces in response: {len(jaeger_data['data']) if jaeger_data['data'] else 0}")

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
                            
                            # Apply duration filter if specified
                            if duration_filter is None or trace_info["durationMs"] >= duration_filter:
                                traces.append(trace_info)

                    logger.info(f"Query results: {len(traces)} traces after filtering (duration_filter: {duration_filter}ms)")
                    if traces:
                        logger.info(f"Sample trace durations: {[t.get('durationMs', 0) for t in traces[:3]]}")
                    else:
                        logger.warning(f"No traces found. Raw response data: {jaeger_data}")
                    
                    return {
                        "success": True,
                        "traces": traces,
                        "total": len(traces),
                        "query": query,
                        "time_range": f"{start_time} to {end_time}",
                        "api_endpoint": search_url,
                        "service_queried": params.get("service", "unknown"),
                        "duration_filter_ms": duration_filter
                    }
                else:
                    logger.error(f"Jaeger API query failed: HTTP {response.status_code}")
                    logger.error(f"Response text: {response.text}")
                    return {
                        "success": False,
                        "error": f"Jaeger API query failed: HTTP {response.status_code} - {response.text}",
                        "query": query,
                        "api_endpoint": search_url,
                        "params": params,
                        "headers": {k: v for k, v in headers.items() if k != "Authorization"}
                    }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error querying Jaeger API: {str(e)}",
                    "query": query,
                    "api_endpoint": search_url
                }

    async def _query_all_services(self, search_url: str, params: Dict[str, Any], headers: Dict[str, str],
                                 query: str, start_time: str, end_time: str, duration_filter: int, limit: int) -> Dict[str, Any]:
        """Query traces from all available services."""
        available_services = await self.get_available_services()
        if not available_services:
            return {
                "success": False,
                "error": "No services available or could not retrieve service list",
                "query": query,
                "api_endpoint": search_url
            }

        logger.info(f"Querying all {len(available_services)} services for wildcard query")
        
        all_traces = []
        successful_services = []
        failed_services = []

        # Query each service
        for service in available_services:
            service_params = params.copy()
            service_params["service"] = service
            service_params["limit"] = min(limit, 50)  # Limit per service to avoid overwhelming
            
            result = await self._query_single_service(search_url, service_params, headers, query, start_time, end_time, duration_filter)
            
            if result["success"]:
                all_traces.extend(result["traces"])
                successful_services.append(service)
                logger.info(f"Service '{service}': {len(result['traces'])} traces")
            else:
                failed_services.append(service)
                logger.warning(f"Service '{service}': {result['error']}")

        # Sort all traces by duration (for fastest/slowest analysis)
        all_traces.sort(key=lambda x: x.get("durationMs", 0), reverse=True)
        
        # Limit total results
        if len(all_traces) > limit:
            all_traces = all_traces[:limit]

        logger.info(f"Combined results: {len(all_traces)} traces from {len(successful_services)} services")
        if failed_services:
            logger.warning(f"Failed to query {len(failed_services)} services: {failed_services}")

        return {
            "success": True,
            "traces": all_traces,
            "total": len(all_traces),
            "query": query,
            "time_range": f"{start_time} to {end_time}",
            "api_endpoint": search_url,
            "service_queried": f"all services ({len(successful_services)}/{len(available_services)})",
            "duration_filter_ms": duration_filter,
            "services_queried": successful_services,
            "failed_services": failed_services
        }
        
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

            # Parse TraceQL query to extract service name and duration filter
            # Simple parsing for service.name=value format
            service_name = None
            duration_filter = None
            
            if "service.name=" in query:
                # Extract service name from TraceQL query like "service.name=my-service"
                parts = query.split("service.name=")
                if len(parts) > 1:
                    extracted_name = parts[1].split()[0].strip('"\'')
                    # Handle wildcard queries - don't set service_name for wildcards
                    if extracted_name != "*" and extracted_name:
                        service_name = extracted_name
            elif "service=" in query:
                # Handle direct service= format
                parts = query.split("service=")
                if len(parts) > 1:
                    extracted_name = parts[1].split()[0].strip('"\'')
                    # Handle wildcard queries - don't set service_name for wildcards
                    if extracted_name != "*" and extracted_name:
                        service_name = extracted_name
            
            # Check for duration filter in query
            if "duration>" in query:
                # Extract duration filter like "duration>1s"
                import re
                duration_match = re.search(r'duration>(\d+)([smh]?)', query)
                if duration_match:
                    duration_value = int(duration_match.group(1))
                    duration_unit = duration_match.group(2) or 's'
                    
                    # Convert to milliseconds
                    if duration_unit == 's':
                        duration_filter = duration_value * 1000
                    elif duration_unit == 'm':
                        duration_filter = duration_value * 60 * 1000
                    elif duration_unit == 'h':
                        duration_filter = duration_value * 60 * 60 * 1000
                    else:
                        duration_filter = duration_value * 1000  # default to seconds

            # Build Jaeger API parameters
            params = {
                "start": start_ts * 1000000,  # Jaeger expects microseconds
                "end": end_ts * 1000000,
                "limit": limit
            }

            if service_name:
                params["service"] = service_name
                # Query single service
                return await self._query_single_service(search_url, params, headers, query, start_time, end_time, duration_filter)
            else:
                # For wildcard queries, query all available services
                return await self._query_all_services(search_url, params, headers, query, start_time, end_time, duration_filter, limit)


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
            trace_url = f"{self.tempo_url}/api/traces/v1/{self.tenant_id}/api/traces/{trace_id}"
            
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
        
        # Debug logging
        logger.info(f"Trace data type: {type(trace_data)}")
        if isinstance(trace_data, dict):
            logger.info(f"Trace data keys: {list(trace_data.keys())}")
        
        # Handle different Jaeger API response formats
        spans = []
        try:
            if isinstance(trace_data, dict):
                # Check if it's a single trace object with spans
                if "spans" in trace_data:
                    spans = trace_data["spans"]
                elif "data" in trace_data and isinstance(trace_data["data"], list) and trace_data["data"]:
                    # Check if data contains trace objects
                    first_trace = trace_data["data"][0]
                    if "spans" in first_trace:
                        spans = first_trace["spans"]
            elif isinstance(trace_data, list) and trace_data:
                # Direct list of spans
                spans = trace_data
        except Exception as e:
            logger.error(f"Error extracting spans from trace data: {e}")
            content += f"**Error**: Could not extract spans from trace data: {str(e)}\n\n"
            content += f"**Raw trace data**: {str(trace_data)[:500]}...\n\n"
            return [{"type": "text", "text": content}]

        if spans:
            content += f"**Total Spans**: {len(spans)}\n\n"
            content += "**Spans**:\n"
            
            for i, span in enumerate(spans[:10], 1):  # Show first 10 spans
                try:
                    span_id = span.get("spanID", "unknown")
                    operation = span.get("operationName", "unknown")
                    # Service name is in the process object for Jaeger format
                    service = span.get("process", {}).get("serviceName", "unknown")
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
                except Exception as e:
                    logger.error(f"Error processing span {i}: {e}")
                    content += f"{i}. **Error processing span**: {str(e)}\n"
                    content += f"   - Raw span data: {str(span)[:200]}...\n\n"
            
            if len(spans) > 10:
                content += f"... and {len(spans) - 10} more spans\n"
        else:
            content += "No span data available for this trace.\n"
            
        return [{"type": "text", "text": content}]
    else:
        error_content = f"Failed to get trace details: {result['error']}"
        return [{"type": "text", "text": error_content}]


def extract_time_range_from_question(question: str) -> str:
    """Extract time range from user question for trace analysis"""
    question_lower = question.lower()
    
    # Check for specific time ranges
    if "last 24 hours" in question_lower or "last 24h" in question_lower or "yesterday" in question_lower:
        return "last 24h"
    elif "last week" in question_lower or "last 7 days" in question_lower:
        return "last 7d"
    elif "last month" in question_lower or "last 30 days" in question_lower:
        return "last 30d"
    elif "last 2 hours" in question_lower or "last 2h" in question_lower:
        return "last 2h"
    elif "last 6 hours" in question_lower or "last 6h" in question_lower:
        return "last 6h"
    elif "last 12 hours" in question_lower or "last 12h" in question_lower:
        return "last 12h"
    elif "last hour" in question_lower or "last 1h" in question_lower:
        return "last 1h"
    elif "last 30 minutes" in question_lower or "last 30m" in question_lower:
        return "last 30m"
    elif "last 15 minutes" in question_lower or "last 15m" in question_lower:
        return "last 15m"
    elif "last 5 minutes" in question_lower or "last 5m" in question_lower:
        return "last 5m"
    elif "week" in question_lower or "7 days" in question_lower:
        # Catch references to week without "last"
        return "last 7d"
    elif "month" in question_lower or "30 days" in question_lower:
        # Catch references to month without "last"
        return "last 30d"
    elif "day" in question_lower or "24 hours" in question_lower:
        # Catch references to day without "last"
        return "last 24h"
    else:
        # For follow-up questions without explicit time, default to 7 days to maintain context
        # This helps when users ask follow-up questions about traces they previously queried
        return "last 7d"

async def chat_tempo_tool(question: str) -> List[Dict[str, Any]]:
    """
    MCP tool function for conversational Tempo trace analysis.
    
    This tool provides a conversational interface for analyzing traces, allowing users to ask
    questions about trace patterns, errors, performance, and service behavior. The tool automatically
    extracts time ranges from the question (e.g., "last 24 hours", "yesterday", "last week").
    
    Args:
        question: Natural language question about traces (e.g., "Show me traces with errors from last 24 hours", 
                 "What services are having performance issues this week?", "Find traces for user login yesterday")
    
    Returns:
        Conversational analysis of traces with insights and recommendations
    """
    tempo_tool = TempoQueryTool()
    
    try:
        # Extract time range from the question
        extracted_time_range = extract_time_range_from_question(question)
        logger.info(f"Extracted time range from question: {extracted_time_range}")
        
        # Parse time range to get start and end times
        now = datetime.now()
        if extracted_time_range.startswith("last "):
            duration_str = extracted_time_range[5:]  # Remove "last "
            if duration_str.endswith("h"):
                hours = int(duration_str[:-1])
                start_time = now - timedelta(hours=hours)
            elif duration_str.endswith("d"):
                days = int(duration_str[:-1])
                start_time = now - timedelta(days=days)
            elif duration_str.endswith("m"):
                minutes = int(duration_str[:-1])
                start_time = now - timedelta(minutes=minutes)
            else:
                # Default to 1 hour
                start_time = now - timedelta(hours=1)
        else:
            # Default to 1 hour
            start_time = now - timedelta(hours=1)
        
        end_time = now
        
        # Convert to ISO format
        start_iso = start_time.isoformat() + "Z"
        end_iso = end_time.isoformat() + "Z"
        
        # Analyze the question to determine appropriate query
        question_lower = question.lower()
        
        # Check if this is a specific trace ID query
        import re
        trace_id_pattern = r'\b[a-f0-9]{16,32}\b'
        trace_id_match = re.search(trace_id_pattern, question)
        
        if trace_id_match:
            # This is a specific trace ID query - get trace details
            trace_id = trace_id_match.group()
            logger.info(f"Detected specific trace ID query: {trace_id}")
            
            # Get trace details
            details_result = await tempo_tool.get_trace_details(trace_id)
            
            if details_result["success"]:
                trace_data = details_result["trace"]
                
                # Extract spans from the trace data (same logic as get_trace_details_tool)
                spans = []
                try:
                    if isinstance(trace_data, dict):
                        # Check if it's a single trace object with spans
                        if "spans" in trace_data:
                            spans = trace_data["spans"]
                        elif "data" in trace_data and isinstance(trace_data["data"], list) and trace_data["data"]:
                            # Check if data contains trace objects
                            first_trace = trace_data["data"][0]
                            if "spans" in first_trace:
                                spans = first_trace["spans"]
                    elif isinstance(trace_data, list) and trace_data:
                        # Direct list of spans
                        spans = trace_data
                except Exception as e:
                    logger.error(f"Error extracting spans from trace data: {e}")
                    content = f"🔍 **Trace Details Analysis**\n\n"
                    content += f"**Trace ID**: {trace_id}\n"
                    content += f"**Error**: Could not extract spans from trace data: {str(e)}\n\n"
                    content += f"**Raw trace data**: {str(trace_data)[:500]}...\n\n"
                    return [{"type": "text", "text": content}]
                
                content = f"🔍 **Trace Details for {trace_id}**\n\n"
                
                if spans:
                    content += f"**Total Spans**: {len(spans)}\n\n"
                    content += "**Spans**:\n"
                    
                    for i, span in enumerate(spans[:10], 1):  # Show first 10 spans
                        try:
                            span_id = span.get("spanID", "unknown")
                            operation = span.get("operationName", "unknown")
                            # Service name is in the process object for Jaeger format
                            service = span.get("process", {}).get("serviceName", "unknown")
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
                        except Exception as e:
                            logger.error(f"Error processing span {i}: {e}")
                            content += f"{i}. **Error processing span**: {str(e)}\n"
                            content += f"   - Raw span data: {str(span)[:200]}...\n\n"
                    
                    if len(spans) > 10:
                        content += f"... and {len(spans) - 10} more spans\n"
                else:
                    content += "No span data available for this trace.\n"
            else:
                content = f"❌ **Error retrieving trace details**: {details_result['error']}\n\n"
                content += "**Troubleshooting**:\n"
                content += "- Verify the trace ID is correct\n"
                content += "- Check if the trace exists in the specified time range\n"
                content += "- Ensure Tempo is accessible\n"
            
            return [{"type": "text", "text": content}]
        
        # Check if this is a detailed analysis request (top N, slowest, etc.)
        elif any(keyword in question_lower for keyword in ["top", "slowest", "fastest", "request flow", "detailed analysis"]):
            # This is a detailed analysis request - get traces and analyze them
            logger.info("Detected detailed analysis request")
            
            # Determine query based on question content
            if "error" in question_lower or "failed" in question_lower or "exception" in question_lower:
                query = "status=error"
            elif "slow" in question_lower and "fastest" not in question_lower:
                # Only apply duration filter if asking for slow traces but NOT fastest
                query = "duration>1s"
            else:
                # For fastest, slowest, or both - get all traces for analysis
                query = "service.name=*"
        else:
            # Determine query based on question content
            if "error" in question_lower or "failed" in question_lower or "exception" in question_lower:
                query = "status=error"
            elif "slow" in question_lower and "fastest" not in question_lower:
                # Only apply duration filter if asking for slow traces but NOT fastest
                query = "duration>1s"
            elif "fastest" in question_lower or "slowest" in question_lower:
                # For fastest/slowest analysis, get all traces
                query = "service.name=*"
            elif "performance" in question_lower or "latency" in question_lower:
                query = "duration>1s"
            elif "service" in question_lower and ("list" in question_lower or "show" in question_lower):
                query = "service.name=*"
            elif any(keyword in question_lower for keyword in ["show me", "what traces", "available traces", "all traces"]):
                # For general trace queries, don't apply duration filter
                query = "service.name=*"
            else:
                query = "service.name=*"
        
        # Query traces
        logger.info(f"Executing Tempo query: '{query}' for time range {start_iso} to {end_iso}")
        result = await tempo_tool.query_traces(query, start_iso, end_iso, limit=50)
        
        if result["success"]:
            traces = result["traces"]
            
            # Analyze traces for insights
            content = f"🔍 **Tempo Chat Analysis**\n\n"
            content += f"**Question**: {question}\n"
            content += f"**Time Range**: {extracted_time_range}\n"
            content += f"**Found**: {len(traces)} traces\n\n"
            
            if traces:
                # Analyze trace patterns
                services = {}
                error_traces = []
                slow_traces = []
                all_traces_with_duration = []
                
                for trace in traces:
                    service_name = trace.get("rootServiceName", "unknown")
                    
                    # Try different duration field names and formats
                    duration = 0
                    if "durationMs" in trace:
                        duration = trace.get("durationMs", 0)
                    elif "duration" in trace:
                        # Convert microseconds to milliseconds if needed
                        duration = trace.get("duration", 0) / 1000
                    elif "durationNanos" in trace:
                        # Convert nanoseconds to milliseconds
                        duration = trace.get("durationNanos", 0) / 1000000
                    
                    # Debug: Log duration information for first few traces
                    if len(all_traces_with_duration) < 3:
                        logger.info(f"Trace {len(all_traces_with_duration)+1} duration fields: {[k for k in trace.keys() if 'duration' in k.lower()]}, calculated duration: {duration}ms")
                    
                    # Count services
                    services[service_name] = services.get(service_name, 0) + 1
                    
                    # Store all traces with duration for analysis
                    trace_with_duration = trace.copy()
                    trace_with_duration["durationMs"] = duration
                    all_traces_with_duration.append(trace_with_duration)
                    
                    # Identify slow traces (>1 second)
                    if duration > 1000:
                        slow_traces.append(trace_with_duration)
                    
                    # Check for error traces (simplified - would need to query span details)
                    if "error" in str(trace).lower():
                        error_traces.append(trace_with_duration)
                
                # Generate insights
                content += "## 📊 **Analysis Results**\n\n"
                
                # Service distribution
                if services:
                    content += "**Services Activity**:\n"
                    for service, count in sorted(services.items(), key=lambda x: x[1], reverse=True)[:5]:
                        content += f"- {service}: {count} traces\n"
                    content += "\n"
                
                # Performance insights - analyze by service for fastest/slowest queries
                if any(keyword in question_lower for keyword in ["fastest", "slowest", "performance"]):
                    # Analyze service-level performance
                    service_performance = {}
                    
                    for trace in all_traces_with_duration:
                        service_name = trace.get("rootServiceName", "unknown")
                        duration = trace.get("durationMs", 0)
                        
                        if service_name not in service_performance:
                            service_performance[service_name] = {
                                "traces": [],
                                "total_duration": 0,
                                "count": 0,
                                "min_duration": float('inf'),
                                "max_duration": 0
                            }
                        
                        service_performance[service_name]["traces"].append(trace)
                        service_performance[service_name]["total_duration"] += duration
                        service_performance[service_name]["count"] += 1
                        service_performance[service_name]["min_duration"] = min(service_performance[service_name]["min_duration"], duration)
                        service_performance[service_name]["max_duration"] = max(service_performance[service_name]["max_duration"], duration)
                    
                    # Calculate average durations
                    for service_name, perf in service_performance.items():
                        perf["avg_duration"] = perf["total_duration"] / perf["count"] if perf["count"] > 0 else 0
                    
                    # Sort services by average duration
                    services_by_avg = sorted(service_performance.items(), key=lambda x: x[1]["avg_duration"])
                    
                    content += "## 🚀 **Service Performance Analysis**\n\n"
                    
                    if len(services_by_avg) == 1:
                        # Only one service - provide detailed analysis
                        service_name, perf = services_by_avg[0]
                        content += f"### 🎯 **Single Service Found: {service_name}**\n\n"
                        content += f"**⚠️ Note**: Only one service has traces in the specified time range. This service is both the fastest AND slowest by default.\n\n"
                        content += f"**Performance Summary**:\n"
                        content += f"- **Average Response Time**: {perf['avg_duration']:.2f}ms\n"
                        content += f"- **Response Time Range**: {perf['min_duration']:.2f}ms - {perf['max_duration']:.2f}ms\n"
                        content += f"- **Total Traces Analyzed**: {perf['count']}\n"
                        content += f"- **Performance Rating**: {'🏃‍♂️ Excellent' if perf['avg_duration'] < 100 else '⚠️ Good' if perf['avg_duration'] < 1000 else '🐌 Needs Improvement'}\n\n"
                        
                        # Analyze performance distribution
                        response_times = [trace.get('durationMs', 0) for trace in perf['traces']]
                        response_times.sort()
                        
                        # Calculate percentiles
                        p50 = response_times[len(response_times)//2] if response_times else 0
                        p90 = response_times[int(len(response_times)*0.9)] if response_times else 0
                        p95 = response_times[int(len(response_times)*0.95)] if response_times else 0
                        p99 = response_times[int(len(response_times)*0.99)] if response_times else 0
                        
                        content += f"**Performance Distribution**:\n"
                        content += f"- **P50 (Median)**: {p50:.2f}ms\n"
                        content += f"- **P90**: {p90:.2f}ms\n"
                        content += f"- **P95**: {p95:.2f}ms\n"
                        content += f"- **P99**: {p99:.2f}ms\n\n"
                        
                        # Performance insights
                        duration_range = perf['max_duration'] - perf['min_duration']
                        
                        if duration_range == 0:
                            content += f"🔍 **Performance Consistency**: All requests have identical duration ({perf['avg_duration']:.2f}ms)\n"
                            content += f"   - This could indicate very consistent performance or data rounding\n"
                            content += f"   - Consider checking if other services are generating traces\n\n"
                        elif duration_range > perf['avg_duration'] * 2:
                            content += f"⚠️ **Performance Variability**: High variability detected (range: {duration_range:.2f}ms)\n"
                            content += f"   - Consider investigating what causes the slower requests\n\n"
                        
                        if p95 > perf['avg_duration'] * 2:
                            content += f"⚠️ **Tail Latency**: 5% of requests are significantly slower than average\n"
                            content += f"   - P95 ({p95:.2f}ms) is {p95/perf['avg_duration']:.1f}x the average\n\n"
                        
                        # Show sample traces for analysis
                        content += f"**Sample Traces for Analysis**:\n"
                        sample_traces = sorted(perf['traces'], key=lambda x: x.get('durationMs', 0), reverse=True)[:3]
                        for i, trace in enumerate(sample_traces, 1):
                            trace_id = trace.get('traceID', 'unknown')
                            duration = trace.get('durationMs', 0)
                            content += f"{i}. **{trace_id}** - {duration:.2f}ms\n"
                        content += f"\n💡 **Tip**: Use `Get details for trace <trace_id>` to analyze specific requests\n\n"
                        
                        # Add recommendations for finding more services
                        content += f"## 🔍 **Recommendations for Better Analysis**\n\n"
                        content += f"**To get meaningful fastest/slowest service comparison:**\n"
                        content += f"1. **Check other services**: Query specific services that might be generating traces\n"
                        content += f"   - Try: `Query traces from service <service_name> from last 7 days`\n"
                        content += f"   - Try: `Show me traces from all services from last 24 hours`\n"
                        content += f"2. **Expand time range**: Try a longer time period to capture more services\n"
                        content += f"   - Try: `Show me fastest and slowest services from last 30 days`\n"
                        content += f"3. **Check service discovery**: Verify what services are available\n"
                        content += f"   - The system found only `{service_name}` in the current time range\n"
                        content += f"4. **Investigate trace generation**: Ensure other services are properly instrumented\n"
                        content += f"   - Check if other services have tracing enabled\n"
                        content += f"   - Verify trace sampling configuration\n\n"
                        
                        # Show what services were discovered but had no traces
                        if 'services_queried' in result and 'failed_services' in result:
                            total_services_discovered = len(result.get('services_queried', [])) + len(result.get('failed_services', []))
                            if total_services_discovered > 1:
                                content += f"**Service Discovery Results**:\n"
                                content += f"- **Services with traces**: {len(result.get('services_queried', []))}\n"
                                content += f"- **Services without traces**: {len(result.get('failed_services', []))}\n"
                                if result.get('failed_services'):
                                    content += f"- **Services found but no traces**: {', '.join(result['failed_services'][:5])}\n"
                                    if len(result['failed_services']) > 5:
                                        content += f"  ... and {len(result['failed_services']) - 5} more\n"
                                content += f"\n"
                        
                    elif len(services_by_avg) == 2:
                        # Two services - compare them
                        service1_name, perf1 = services_by_avg[0]
                        service2_name, perf2 = services_by_avg[1]
                        
                        content += f"### 🏃‍♂️ **Fastest Service**: {service1_name}\n"
                        content += f"- **Average**: {perf1['avg_duration']:.2f}ms\n"
                        content += f"- **Range**: {perf1['min_duration']:.2f}ms - {perf1['max_duration']:.2f}ms\n"
                        content += f"- **Traces**: {perf1['count']}\n\n"
                        
                        content += f"### 🐌 **Slowest Service**: {service2_name}\n"
                        content += f"- **Average**: {perf2['avg_duration']:.2f}ms\n"
                        content += f"- **Range**: {perf2['min_duration']:.2f}ms - {perf2['max_duration']:.2f}ms\n"
                        content += f"- **Traces**: {perf2['count']}\n\n"
                        
                        # Performance comparison
                        speed_diff = perf2['avg_duration'] - perf1['avg_duration']
                        speed_ratio = perf2['avg_duration'] / perf1['avg_duration'] if perf1['avg_duration'] > 0 else 1
                        
                        content += f"**Performance Comparison**:\n"
                        content += f"- **Speed Difference**: {service2_name} is {speed_diff:.2f}ms slower on average\n"
                        content += f"- **Speed Ratio**: {service2_name} is {speed_ratio:.1f}x slower than {service1_name}\n\n"
                        
                    else:
                        # Multiple services - show fastest and slowest
                        if "fastest" in question_lower or "slowest" in question_lower:
                            content += "### 🏃‍♂️ **Fastest Services** (by average response time):\n"
                            for i, (service_name, perf) in enumerate(services_by_avg[:3], 1):
                                content += f"{i}. **{service_name}**\n"
                                content += f"   - Average: {perf['avg_duration']:.2f}ms\n"
                                content += f"   - Min: {perf['min_duration']:.2f}ms\n"
                                content += f"   - Max: {perf['max_duration']:.2f}ms\n"
                                content += f"   - Traces: {perf['count']}\n\n"
                            
                            content += "### 🐌 **Slowest Services** (by average response time):\n"
                            for i, (service_name, perf) in enumerate(services_by_avg[-3:][::-1], 1):
                                content += f"{i}. **{service_name}**\n"
                                content += f"   - Average: {perf['avg_duration']:.2f}ms\n"
                                content += f"   - Min: {perf['min_duration']:.2f}ms\n"
                                content += f"   - Max: {perf['max_duration']:.2f}ms\n"
                                content += f"   - Traces: {perf['count']}\n\n"
                        else:
                            # Show all services sorted by performance
                            content += "### 📊 **All Services Performance** (sorted by average response time):\n"
                            for i, (service_name, perf) in enumerate(services_by_avg, 1):
                                performance_icon = "🏃‍♂️" if perf['avg_duration'] < 100 else "⚠️" if perf['avg_duration'] < 1000 else "🐌"
                                content += f"{i}. {performance_icon} **{service_name}**\n"
                                content += f"   - Average: {perf['avg_duration']:.2f}ms\n"
                                content += f"   - Min: {perf['min_duration']:.2f}ms\n"
                                content += f"   - Max: {perf['max_duration']:.2f}ms\n"
                                content += f"   - Traces: {perf['count']}\n\n"
                
                # Show individual trace details for detailed analysis requests
                elif any(keyword in question_lower for keyword in ["top", "request flow", "detailed analysis"]):
                    # For detailed analysis requests, show top traces by duration
                    if all_traces_with_duration:
                        # Sort all traces by duration and get top 3
                        top_traces = sorted(all_traces_with_duration, key=lambda x: x.get("durationMs", 0), reverse=True)[:3]
                        
                        content += "## 🔍 **Detailed Analysis**\n\n"
                        content += "**Request Flow Analysis** (Top 3 traces by duration):\n"
                        
                        for i, trace in enumerate(top_traces, 1):
                            trace_id = trace.get("traceID", "unknown")
                            service = trace.get("rootServiceName", "unknown")
                            duration = trace.get("durationMs", 0)
                            
                            content += f"\n### **Trace {i}: {trace_id}**\n"
                            content += f"- **Service**: {service}\n"
                            content += f"- **Duration**: {duration:.2f}ms\n"
                            content += f"- **Performance Impact**: {'🚨 Critical' if duration > 5000 else '⚠️ Slow' if duration > 1000 else '✅ Normal'}\n"
                            
                            # Get additional trace details for analysis
                            try:
                                details_result = await tempo_tool.get_trace_details(trace_id)
                                if details_result["success"] and details_result["trace"]:
                                    trace_data = details_result["trace"]
                                    # Extract spans from the trace data
                                    spans = []
                                    if isinstance(trace_data, dict):
                                        # Check if it's a single trace object with spans
                                        if "spans" in trace_data:
                                            spans = trace_data["spans"]
                                        elif "data" in trace_data and isinstance(trace_data["data"], list) and trace_data["data"]:
                                            # Check if data contains trace objects
                                            first_trace = trace_data["data"][0]
                                            if "spans" in first_trace:
                                                spans = first_trace["spans"]
                                    elif isinstance(trace_data, list) and trace_data:
                                        # Direct list of spans
                                        spans = trace_data
                                    
                                    if spans:
                                        content += f"- **Span Count**: {len(spans)}\n"
                                        
                                        # Analyze span hierarchy
                                        services_involved = set()
                                        for span in spans:
                                            service_name = span.get("process", {}).get("serviceName", "unknown")
                                            services_involved.add(service_name)
                                        
                                        if len(services_involved) > 1:
                                            content += f"- **Services Involved**: {', '.join(sorted(services_involved))}\n"
                                        
                                        # Show critical spans (longest duration)
                                        critical_spans = sorted(spans, key=lambda x: x.get("duration", 0), reverse=True)[:3]
                                        content += "- **Critical Spans**:\n"
                                        for span in critical_spans:
                                            operation = span.get("operationName", "unknown")
                                            span_duration = span.get("duration", 0)
                                            span_service = span.get("process", {}).get("serviceName", "unknown")
                                            content += f"  - {operation} ({span_service}): {span_duration/1000:.2f}ms\n"
                                    else:
                                        content += f"- **Note**: No spans found in trace details\n"
                                else:
                                    content += f"- **Note**: Could not retrieve trace details: {details_result.get('error', 'Unknown error')}\n"
                            except Exception as e:
                                logger.error(f"Error getting trace details for {trace_id}: {e}")
                                content += f"- **Note**: Could not retrieve detailed span information: {str(e)}\n"
                            
                            content += f"- **Action**: Use `Get details for trace {trace_id}` for complete analysis\n"
                        
                        content += "\n"
                
                # Show slow traces if any
                if slow_traces:
                    content += f"**⚠️ Performance Issues**: {len(slow_traces)} slow traces found (>1000ms)\n"
                    content += "Slowest traces:\n"
                    
                    # Sort by duration and get top traces
                    top_slow_traces = sorted(slow_traces, key=lambda x: x.get("durationMs", 0), reverse=True)[:3]
                    
                    for i, trace in enumerate(top_slow_traces, 1):
                        trace_id = trace.get("traceID", "unknown")
                        service = trace.get("rootServiceName", "unknown")
                        duration = trace.get("durationMs", 0)
                        content += f"{i}. **{service}**: {trace_id} ({duration:.2f}ms)\n"
                    
                    content += "\n"
                
                # Error insights
                if error_traces:
                    content += f"**🚨 Error Traces**: {len(error_traces)} error traces found\n"
                    content += "Recent error traces:\n"
                    for trace in error_traces[:3]:
                        trace_id = trace.get("traceID", "unknown")
                        service = trace.get("rootServiceName", "unknown")
                        content += f"- {service}: {trace_id}\n"
                    content += "\n"
                
                # Recommendations
                content += "## 💡 **Recommendations**\n\n"
                if slow_traces:
                    content += f"- **Investigate slow traces**: {len(slow_traces)} traces took >1 second\n"
                    content += f"- **Slowest trace**: {slow_traces[0]['traceID']} ({slow_traces[0]['durationMs']}ms)\n"
                    content += "- **Get trace details**: Use `get_trace_details_tool` with trace ID\n"
                if error_traces:
                    content += f"- **Check error traces**: {len(error_traces)} traces had errors\n"
                    content += f"- **Error trace**: {error_traces[0]['traceID']}\n"
                if len(services) > 5:
                    content += "- **Service consolidation**: Consider consolidating {len(services)} services\n"
                
                content += "- **Query specific traces**: Use `query_tempo_tool` for filtered searches\n"
                content += "- **Example queries**:\n"
                if traces:
                    content += f"  - `Get details for trace {traces[0]['traceID']}`\n"
                content += "  - `Query traces with duration > 5000ms from last week`\n"
                content += "  - `Show me traces with errors from last week`\n"
                
            else:
                content += "No traces found for the specified criteria.\n\n"
                content += "**Suggestions**:\n"
                content += "- Try a broader time range\n"
                content += "- Check if services are actively generating traces\n"
                content += "- Verify the query parameters\n"
            
            return [{"type": "text", "text": content}]
        else:
            error_content = f"Failed to analyze traces: {result['error']}\n\n"
            error_content += "**Troubleshooting**:\n"
            error_content += "- Check if Tempo is accessible\n"
            error_content += "- Verify authentication credentials\n"
            error_content += "- Try a different time range\n"
            
            return [{"type": "text", "text": error_content}]
            
    except Exception as e:
        logger.error(f"Tempo chat error: {e}")
        error_content = f"Error during Tempo chat analysis: {str(e)}\n\n"
        error_content += "**Troubleshooting**:\n"
        error_content += "- Check Tempo connectivity\n"
        error_content += "- Verify time range format\n"
        error_content += "- Try a simpler question\n"
        
        return [{"type": "text", "text": error_content}]


# Note: list_trace_services_tool removed because the /api/traces/v1/{tenant_id}/services endpoint
# is not available in this TempoStack deployment. Use query_tempo_tool to search for traces instead.

# Note: analyze_traces_tool removed for now - can be added back later when needed
# It requires LLM integration and complex analysis logic that may need refinement