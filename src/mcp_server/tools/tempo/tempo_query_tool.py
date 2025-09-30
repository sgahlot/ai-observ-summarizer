"""Tempo Query Tool - Query traces from Tempo instance in observability-hub namespace.

This module provides async MCP tools for interacting with Tempo traces:
- query_tempo_tool: Search traces by service, operation, time range
- get_trace_details_tool: Get detailed trace information by trace ID
- chat_tempo_tool: Conversational interface for Tempo trace analysis
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
import re

from .tempo_query import TempoQueryService
from .tempo_formatters import TempoTraceFormatter
from .tempo_analyzers import TempoTraceAnalyzer

logger = logging.getLogger(__name__)


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
    tempo_tool = TempoQueryService()
    result = await tempo_tool.query_traces(query, start_time, end_time, limit)
    
    if result["success"]:
        content = TempoTraceFormatter.format_query_results(result)
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
    tempo_tool = TempoQueryService()
    result = await tempo_tool.get_trace_details(trace_id)
    
    if result["success"]:
        content = TempoTraceFormatter.format_trace_details(trace_id, result["trace"])
        return [{"type": "text", "text": content}]
    else:
        error_content = f"Failed to get trace details: {result['error']}"
        return [{"type": "text", "text": error_content}]


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
    tempo_tool = TempoQueryService()
    
    try:
        # Extract time range from the question
        extracted_time_range = TempoTraceAnalyzer.extract_time_range_from_question(question)
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
        trace_id_pattern = r'\b[a-f0-9]{16,32}\b'
        trace_id_match = re.search(trace_id_pattern, question)
        
        if trace_id_match:
            # This is a specific trace ID query - get trace details
            trace_id = trace_id_match.group()
            logger.info(f"Detected specific trace ID query: {trace_id}")
            
            # Get trace details
            details_result = await tempo_tool.get_trace_details(trace_id)
            
            if details_result["success"]:
                content = TempoTraceFormatter.format_trace_details(trace_id, details_result["trace"])
            else:
                content = f"❌ **Error retrieving trace details**: {details_result['error']}\n\n"
                content += "**Troubleshooting**:\n"
                content += "- Verify the trace ID is correct\n"
                content += "- Check if the trace exists in the specified time range\n"
                content += "- Ensure Tempo is accessible\n"
            
            return [{"type": "text", "text": content}]
        
        # Determine query based on question content
        query = TempoTraceAnalyzer.determine_query_from_question(question)
        
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
                analysis = TempoTraceAnalyzer.analyze_trace_patterns(traces)
                services = analysis["services"]
                error_traces = analysis["error_traces"]
                slow_traces = analysis["slow_traces"]
                all_traces_with_duration = analysis["all_traces_with_duration"]
                
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
                    perf_analysis = TempoTraceAnalyzer.analyze_service_performance(all_traces_with_duration)
                    services_by_avg = perf_analysis["services_by_avg"]
                    
                    content += TempoTraceFormatter.format_service_performance_analysis(services_by_avg, question_lower)
                
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
                                    spans = TempoTraceFormatter.extract_spans_from_trace_data(trace_data)
                                    
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
