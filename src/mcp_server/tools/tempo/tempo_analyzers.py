"""Analyzers for Tempo trace data.

This module contains analysis logic for processing trace data and generating insights.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TempoTraceAnalyzer:
    """Analyzer for Tempo trace data."""

    @staticmethod
    def analyze_service_performance(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze service-level performance from traces."""
        service_performance = {}
        
        for trace in traces:
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
        
        return {
            "service_performance": service_performance,
            "services_by_avg": services_by_avg
        }

    @staticmethod
    def analyze_trace_patterns(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trace patterns and generate insights."""
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
        
        return {
            "services": services,
            "error_traces": error_traces,
            "slow_traces": slow_traces,
            "all_traces_with_duration": all_traces_with_duration
        }

    @staticmethod
    def determine_query_from_question(question: str) -> str:
        """Determine the appropriate query based on the question content."""
        question_lower = question.lower()
        
        if "error" in question_lower or "failed" in question_lower or "exception" in question_lower:
            return "status=error"
        elif "slow" in question_lower and "fastest" not in question_lower:
            # Only apply duration filter if asking for slow traces but NOT fastest
            return "duration>1s"
        elif "fastest" in question_lower or "slowest" in question_lower:
            # For fastest/slowest analysis, get all traces
            return "service.name=*"
        elif "performance" in question_lower or "latency" in question_lower:
            return "duration>1s"
        elif "service" in question_lower and ("list" in question_lower or "show" in question_lower):
            return "service.name=*"
        elif any(keyword in question_lower for keyword in ["show me", "what traces", "available traces", "all traces"]):
            # For general trace queries, don't apply duration filter
            return "service.name=*"
        else:
            return "service.name=*"

    @staticmethod
    def extract_time_range_from_question(question: str) -> str:
        """Extract time range from user question for trace analysis."""
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
