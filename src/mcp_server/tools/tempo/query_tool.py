"""Tempo query tool for interacting with Tempo trace data."""

from typing import Dict, Any, List

from common.pylogger import get_python_logger
from core.tempo_service import TempoQueryService

logger = get_python_logger()


class TempoQueryTool:
    """Tool for querying Tempo traces with async support."""

    # Configuration constants
    SLOW_TRACE_THRESHOLD_MS = 1000  # Traces slower than this are considered "slow"
    MAX_PER_SERVICE_LIMIT = 50  # Maximum traces to fetch per service in wildcard queries
    DEFAULT_CHAT_QUERY_LIMIT = 50  # Default limit for chat tool queries
    DEFAULT_QUERY_LIMIT = 20  # Default limit for regular queries
    REQUEST_TIMEOUT_SECONDS = 30.0  # HTTP request timeout

    def __init__(self):
        """Initialize the Tempo query tool with centralized service."""
        self.service = TempoQueryService()

    async def get_available_services(self) -> List[str]:
        """Get list of available services from Tempo/Jaeger."""
        return await self.service.get_available_services()


    async def query_traces(
        self,
        query: str,
        start_time: str,
        end_time: str,
        limit: int = DEFAULT_QUERY_LIMIT
    ) -> Dict[str, Any]:
        """
        Query traces from Tempo using TraceQL syntax.

        Args:
            query (str): TraceQL query string. Supports:
                - Service filtering: "service.name=my-service"
                - Wildcard queries: "service.name=*"
                - Duration filtering: "duration>100ms"
                - Error filtering: "status=error"
                - Complex queries: "service.name=ui && duration>500ms"
            start_time (str): Start time in ISO 8601 format with timezone.
                Examples: "2024-01-01T10:00:00Z", "2024-01-01T10:00:00+00:00"
                The method automatically handles 'Z' suffix conversion to '+00:00'
            end_time (str): End time in ISO 8601 format with timezone.
                Examples: "2024-01-01T11:00:00Z", "2024-01-01T11:00:00+00:00"
                The method automatically handles 'Z' suffix conversion to '+00:00'
            limit (int, optional): Maximum number of traces to return. Defaults to DEFAULT_QUERY_LIMIT (20).

        Returns:
            Dict[str, Any]: Query result containing:
                - success (bool): Whether the query was successful
                - traces (List[Dict]): List of trace data if successful
                - query (str): The original query string
                - time_range (str): Formatted time range
                - error (str): Error message if unsuccessful
        """
        return await self.service.query_traces(query, start_time, end_time, limit)

    async def get_trace_details(self, trace_id: str) -> Dict[str, Any]:
        """Get detailed trace information."""
        return await self.service.get_trace_details(trace_id)
