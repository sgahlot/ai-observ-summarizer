"""Tempo query functionality for searching and retrieving traces.

This module handles the core querying logic for Tempo traces.
"""

import httpx
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

from .tempo_base import TempoQueryTool

logger = logging.getLogger(__name__)


class TempoQueryService(TempoQueryTool):
    """Service for querying Tempo traces with enhanced functionality."""

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
            # Convert times to Unix timestamps in microseconds (Tempo expects microseconds)
            start_ts = int(datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp() * 1000000)
            end_ts = int(datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp() * 1000000)

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
