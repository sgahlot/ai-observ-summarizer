"""Formatters for Tempo trace data display.

This module contains all the formatting logic for displaying trace data
in various formats (detailed, summary, analysis, etc.).
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TempoTraceFormatter:
    """Formatter for Tempo trace data display."""

    @staticmethod
    def extract_spans_from_trace_data(trace_data: Any) -> List[Dict[str, Any]]:
        """Extract spans from trace data, handling different Jaeger API response formats."""
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
            raise e
        return spans

    @staticmethod
    def format_trace_details(trace_id: str, trace_data: Any) -> str:
        """Format detailed trace information for display."""
        content = f"🔍 **Trace Details for {trace_id}**\n\n"
        
        # Debug logging
        logger.info(f"Trace data type: {type(trace_data)}")
        if isinstance(trace_data, dict):
            logger.info(f"Trace data keys: {list(trace_data.keys())}")
        
        # Extract spans from the trace data
        spans = TempoTraceFormatter.extract_spans_from_trace_data(trace_data)
        
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
            
        return content

    @staticmethod
    def format_query_results(result: Dict[str, Any]) -> str:
        """Format query results for display."""
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
            
        return content

    @staticmethod
    def format_single_service_analysis(service_name: str, perf: Dict[str, Any], result: Dict[str, Any] = None) -> str:
        """Format analysis for a single service."""
        content = f"### 🎯 **Single Service Found: {service_name}**\n\n"
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
        if result and 'services_queried' in result and 'failed_services' in result:
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
        
        return content

    @staticmethod
    def format_service_performance_analysis(services_by_avg: List[tuple], question_lower: str) -> str:
        """Format service performance analysis."""
        content = "## 🚀 **Service Performance Analysis**\n\n"
        
        if len(services_by_avg) == 1:
            service_name, perf = services_by_avg[0]
            return TempoTraceFormatter.format_single_service_analysis(service_name, perf)
        elif len(services_by_avg) == 2:
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
        
        return content
