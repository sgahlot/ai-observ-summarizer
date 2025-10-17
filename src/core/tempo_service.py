"""
Centralized Tempo query service for trace data operations.

This module contains the core business logic for querying Tempo traces,
moving the query logic from the tempo tools to a centralized service.
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .http_client import TempoClient
from .config import (
    TEMPO_URL, TEMPO_TENANT_ID, 
    MAX_PER_SERVICE_LIMIT, DEFAULT_QUERY_LIMIT, REQUEST_TIMEOUT_SECONDS
)
from .models import QueryResponse, TraceDetailsResponse
from .error_handling import TempoErrorClassifier
from .time_utils import convert_duration_to_milliseconds

import logging
logger = logging.getLogger(__name__)


class TempoQueryService:
    """Centralized service for Tempo trace queries."""
    
    # Configuration constants (imported from core.config)
    MAX_PER_SERVICE_LIMIT = MAX_PER_SERVICE_LIMIT
    DEFAULT_QUERY_LIMIT = DEFAULT_QUERY_LIMIT
    REQUEST_TIMEOUT_SECONDS = REQUEST_TIMEOUT_SECONDS
    
    def __init__(self):
        """Initialize the Tempo query service."""
        self.client = TempoClient(TEMPO_URL, TEMPO_TENANT_ID, self.REQUEST_TIMEOUT_SECONDS)
    
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
        return await self.client.get_services()
    
    async def _query_single_service(self, params: Dict[str, Any], query: str, 
                                  start_time: str, end_time: str, duration_filter: Optional[int]) -> Dict[str, Any]:
        """Query traces from a single service."""
        try:
            response = await self.client.query_traces(params)
            
            # Convert Jaeger format to our expected format
            traces = []
            if "data" in response and response["data"]:
                for trace in response["data"]:
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
            
            return QueryResponse(
                success=True,
                query=query,
                traces=traces,
                total=len(traces),
                time_range=f"{start_time} to {end_time}",
                api_endpoint=f"{self.client.base_url}/api/traces/v1/{self.client.tenant_id}/api/traces",
                service_queried=params.get("service", "unknown"),
                duration_filter_ms=duration_filter
            ).to_dict()
            
        except Exception as e:
            logger.error(f"Error querying single service: {e}")
            return QueryResponse(
                success=False,
                query=query,
                error=f"Error querying service: {str(e)}"
            ).to_dict()
    
    async def _query_all_services(self, params: Dict[str, Any], query: str,
                                 start_time: str, end_time: str, duration_filter: Optional[int], 
                                 limit: int) -> Dict[str, Any]:
        """Query traces from all available services."""
        available_services = await self.get_available_services()
        if not available_services:
            return QueryResponse(
                success=False,
                query=query,
                error="No services available or could not retrieve service list"
            ).to_dict()

        logger.info(f"Querying all {len(available_services)} services for wildcard query")

        all_traces = []
        successful_services = []
        failed_services = []

        # Query each service
        for service in available_services:
            service_params = params.copy()
            service_params["service"] = service
            service_params["limit"] = min(limit, self.MAX_PER_SERVICE_LIMIT)

            result = await self._query_single_service(service_params, query, start_time, end_time, duration_filter)

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

        return QueryResponse(
            success=True,
            query=query,
            traces=all_traces,
            total=len(all_traces),
            time_range=f"{start_time} to {end_time}",
            api_endpoint=f"{self.client.base_url}/api/traces/v1/{self.client.tenant_id}/api/traces",
            service_queried=f"all services ({len(successful_services)}/{len(available_services)})",
            duration_filter_ms=duration_filter,
            services_queried=successful_services,
            failed_services=failed_services
        ).to_dict()
    
    def _parse_traceql_query(self, query: str) -> tuple[Optional[str], Optional[int]]:
        """
        Parse TraceQL query to extract service name and duration filter.
        
        Args:
            query: TraceQL query string
            
        Returns:
            Tuple of (service_name, duration_filter_ms)
        """
        service_name = None
        duration_filter = None

        # Parse service name
        if "service.name=" in query:
            parts = query.split("service.name=")
            if len(parts) > 1:
                extracted_name = parts[1].split()[0].strip('"\'')
                if extracted_name != "*" and extracted_name:
                    service_name = extracted_name
        elif "service=" in query:
            parts = query.split("service=")
            if len(parts) > 1:
                extracted_name = parts[1].split()[0].strip('"\'')
                if extracted_name != "*" and extracted_name:
                    service_name = extracted_name

        # Parse duration filter
        if "duration>" in query:
            duration_match = re.search(r'duration>(\d+)([smh]?)', query)
            if duration_match:
                duration_value = int(duration_match.group(1))
                duration_unit = duration_match.group(2) or 's'

                # Convert to milliseconds
                duration_filter = convert_duration_to_milliseconds(duration_value, duration_unit)

        return service_name, duration_filter
    
    async def query_traces(self, query: str, start_time: str, end_time: str, 
                          limit: int = DEFAULT_QUERY_LIMIT) -> Dict[str, Any]:
        """
        Query traces from Tempo using TraceQL syntax.
        
        Args:
            query: TraceQL query string
            start_time: Start time in ISO 8601 format
            end_time: End time in ISO 8601 format
            limit: Maximum number of traces to return
            
        Returns:
            Query result dictionary
        """
        try:
            # Convert times to Unix timestamps
            start_ts = int(datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp())
            end_ts = int(datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp())

            # Parse TraceQL query
            service_name, duration_filter = self._parse_traceql_query(query)

            # Build Jaeger API parameters
            params = {
                "start": start_ts * 1000000,  # Jaeger expects microseconds
                "end": end_ts * 1000000,
                "limit": limit
            }

            if service_name:
                params["service"] = service_name
                # Query single service
                return await self._query_single_service(params, query, start_time, end_time, duration_filter)
            else:
                # For wildcard queries, query all available services
                return await self._query_all_services(params, query, start_time, end_time, duration_filter, limit)

        except Exception as e:
            logger.error(f"Tempo query error: {e}")
            error_msg = str(e)

            # Use robust error classification
            error_type = TempoErrorClassifier.classify_error(error_msg)
            user_friendly_msg = TempoErrorClassifier.get_user_friendly_message(error_type, self.client.base_url)

            return QueryResponse(
                success=False,
                query=query,
                error=user_friendly_msg,
                tempo_url=self.client.base_url,
                error_type=error_type.value
            ).to_dict()
    
    async def get_trace_details(self, trace_id: str) -> Dict[str, Any]:
        """Get detailed trace information."""
        try:
            response = await self.client.get_trace_details(trace_id)
            return TraceDetailsResponse(
                success=True,
                trace=response
            ).to_dict()
        except Exception as e:
            logger.error(f"Trace details error: {e}")
            return TraceDetailsResponse(
                success=False,
                error=str(e)
            ).to_dict()
