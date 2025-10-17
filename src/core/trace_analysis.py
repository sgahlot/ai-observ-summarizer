"""
Centralized trace analysis logic for observability data.

This module provides reusable trace analysis patterns that can be used
across different observability tools for consistent analysis.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

from common.pylogger import get_python_logger
from core.config import SLOW_TRACE_THRESHOLD_MS
from core.time_utils import calculate_duration_ms
from core.question_classification import TraceErrorDetector

logger = get_python_logger()


@dataclass
class TraceAnalysisResult:
    """Result of trace analysis containing all analyzed data."""
    services: Dict[str, int]
    error_traces: List[Dict[str, Any]]
    slow_traces: List[Dict[str, Any]]
    all_traces_with_duration: List[Dict[str, Any]]


class TraceAnalyzer:
    """Centralized trace analysis functionality."""

    @staticmethod
    def analyze_traces(traces: List[Dict[str, Any]]) -> TraceAnalysisResult:
        """
        Analyze traces for patterns, performance, and errors.
        
        Args:
            traces: List of trace dictionaries
            
        Returns:
            TraceAnalysisResult with analysis data
        """
        services = {}
        error_traces = []
        slow_traces = []
        all_traces_with_duration = []

        for trace in traces:
            service_name = trace.get("rootServiceName", "unknown")
            
            # Calculate duration using centralized function
            duration = calculate_duration_ms(trace)
            
            # Count services
            services[service_name] = services.get(service_name, 0) + 1
            
            # Store all traces with duration for analysis
            trace_with_duration = trace.copy()
            trace_with_duration["durationMs"] = duration
            all_traces_with_duration.append(trace_with_duration)
            
            # Identify slow traces
            if duration > SLOW_TRACE_THRESHOLD_MS:
                slow_traces.append(trace_with_duration)
            
            # Check for error traces
            if TraceErrorDetector.is_error_trace(trace):
                error_traces.append(trace_with_duration)

        return TraceAnalysisResult(
            services=services,
            error_traces=error_traces,
            slow_traces=slow_traces,
            all_traces_with_duration=all_traces_with_duration
        )

    @staticmethod
    def generate_service_activity_summary(services: Dict[str, int]) -> str:
        """Generate a markdown summary of service activity."""
        content = ""
        if services:
            content += "**Services Activity**:\n"
            for service, count in sorted(services.items(), key=lambda x: x[1], reverse=True)[:5]:
                content += f"- {service}: {count} traces\n"
            content += "\n"
        return content

    @staticmethod
    def generate_slow_traces_summary(slow_traces: List[Dict[str, Any]]) -> str:
        """Generate a markdown summary of slow traces."""
        content = ""
        if slow_traces:
            content += f"**⚠️ Performance Issues**: {len(slow_traces)} slow traces found (>1000ms)\n"
            content += "Slowest traces:\n"
            top_slow_traces = sorted(slow_traces, key=lambda x: x.get("durationMs", 0), reverse=True)[:3]
            for i, trace in enumerate(top_slow_traces, 1):
                trace_id = trace.get("traceID", "unknown")
                service = trace.get("rootServiceName", "unknown")
                duration = trace.get("durationMs", 0)
                content += f"{i}. **{service}**: {trace_id} ({duration:.2f}ms)\n"
            content += "\n"
        return content

    @staticmethod
    def generate_error_traces_summary(error_traces: List[Dict[str, Any]]) -> str:
        """Generate a markdown summary of error traces."""
        content = ""
        if error_traces:
            content += f"**🚨 Error Traces**: {len(error_traces)} error traces found\n"
            content += "Recent error traces:\n"
            for trace in error_traces[:3]:
                trace_id = trace.get("traceID", "unknown")
                service = trace.get("rootServiceName", "unknown")
                content += f"- {service}: {trace_id}\n"
            content += "\n"
        return content