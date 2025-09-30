"""Base Tempo client for querying traces from Tempo instance.

This module provides the core functionality for interacting with Tempo traces:
- TempoQueryTool: Core client for API interactions
- Trace data processing and conversion utilities
"""

import httpx
from typing import Dict, Any, List
import logging
from core.config import VERIFY_SSL

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

            async with httpx.AsyncClient(timeout=30.0, verify=VERIFY_SSL) as client:
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
            
            async with httpx.AsyncClient(timeout=30.0, verify=VERIFY_SSL) as client:
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
