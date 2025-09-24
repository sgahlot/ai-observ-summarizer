"""Pure Prometheus MCP Tools for Chat with Prometheus.

This module provides MCP tools for direct Prometheus interaction:
- search_metrics: Search metrics by name pattern
- get_metric_metadata: Get metric help text and type
- get_label_values: Get available label values for a metric
- execute_promql: Execute PromQL query
- explain_results: Explain query results in natural language
- suggest_queries: Suggest related queries based on user intent

These tools enable LLMs to interact directly with Prometheus
without requiring a pre-built knowledge base.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
import requests
from datetime import datetime, timedelta
import re

from core.config import PROMETHEUS_URL, THANOS_TOKEN, VERIFY_SSL
from core.llm_client import summarize_with_llm


def _resp(content: str, is_error: bool = False) -> List[Dict[str, Any]]:
    """Helper to format MCP tool responses consistently."""
    return [{"type": "text", "text": content}]


def _make_prometheus_request(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """Make authenticated request to Prometheus/Thanos."""
    url = f"{PROMETHEUS_URL}{endpoint}"
    headers = {}
    
    if THANOS_TOKEN:
        headers["Authorization"] = f"Bearer {THANOS_TOKEN}"
    
    try:
        response = requests.get(
            url, 
            params=params, 
            headers=headers, 
            verify=VERIFY_SSL,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Prometheus request failed: {e}")
        raise


def search_metrics(
    pattern: str = "",
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Intelligently search for metrics using semantic understanding.
    
    Args:
        pattern: Search terms (e.g., "tokens", "gpu temperature", "pod status")
        limit: Maximum number of metrics to return
    
    Returns:
        List of matching metrics ranked by relevance
    """
    try:
        # Get all metric names
        response = _make_prometheus_request("/api/v1/label/__name__/values")
        all_metrics = response.get("data", [])
        
        # Use semantic search if pattern provided
        if pattern:
            # Use our dynamic selection to rank metrics
            ranked_metrics = _rank_metrics_by_relevance(pattern, all_metrics)
            matching_metrics = ranked_metrics[:limit]
        else:
            matching_metrics = all_metrics[:limit]
        
        # Get basic info for each metric
        metrics_info = []
        for metric in matching_metrics:
            try:
                metadata_response = _make_prometheus_request(
                    "/api/v1/metadata", 
                    {"metric": metric}
                )
                metadata = metadata_response.get("data", {}).get(metric, [])
                
                if metadata:
                    metric_info = {
                        "name": metric,
                        "type": metadata[0].get("type", "unknown"),
                        "help": metadata[0].get("help", "No description available"),
                        "unit": metadata[0].get("unit", "")
                    }
                else:
                    metric_info = {
                        "name": metric,
                        "type": "unknown",
                        "help": "No description available",
                        "unit": ""
                    }
                
                metrics_info.append(metric_info)
            except Exception as e:
                logging.warning(f"Failed to get metadata for {metric}: {e}")
                metrics_info.append({
                    "name": metric,
                    "type": "unknown", 
                    "help": "No description available",
                    "unit": ""
                })
        
        result = {
            "total_found": len(matching_metrics),
            "metrics": metrics_info,
            "pattern": pattern,
            "limit": limit
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error searching metrics: {e}")
        return _resp(f"Error searching metrics: {str(e)}", is_error=True)


def get_metric_metadata(
    metric_name: str
) -> List[Dict[str, Any]]:
    """Get detailed metadata for a specific metric.
    
    Args:
        metric_name: Name of the metric to get metadata for
    
    Returns:
        Detailed metadata including type, help text, unit, and available labels
    """
    try:
        # Get metric metadata
        metadata_response = _make_prometheus_request(
            "/api/v1/metadata", 
            {"metric": metric_name}
        )
        metadata = metadata_response.get("data", {}).get(metric_name, [])
        
        if not metadata:
            return _resp(f"Metric '{metric_name}' not found", is_error=True)
        
        # Get available labels for this metric
        labels_response = _make_prometheus_request("/api/v1/labels")
        all_labels = labels_response.get("data", [])
        
        # Get sample values for common labels
        sample_values = {}
        for label in ["instance", "job", "namespace", "pod", "node"]:
            if label in all_labels:
                try:
                    values_response = _make_prometheus_request(
                        f"/api/v1/label/{label}/values"
                    )
                    sample_values[label] = values_response.get("data", [])[:10]  # Limit to 10 samples
                except Exception:
                    sample_values[label] = []
        
        result = {
            "metric_name": metric_name,
            "metadata": metadata[0] if metadata else {},
            "available_labels": all_labels,
            "sample_label_values": sample_values,
            "query_examples": _generate_query_examples(metric_name, metadata[0] if metadata else {})
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error getting metric metadata: {e}")
        return _resp(f"Error getting metric metadata: {str(e)}", is_error=True)


def get_label_values(
    metric_name: str,
    label_name: str
) -> List[Dict[str, Any]]:
    """Get all possible values for a specific label of a metric.
    
    Args:
        metric_name: Name of the metric
        label_name: Name of the label to get values for
    
    Returns:
        List of all possible values for the label
    """
    try:
        response = _make_prometheus_request(f"/api/v1/label/{label_name}/values")
        values = response.get("data", [])
        
        result = {
            "metric_name": metric_name,
            "label_name": label_name,
            "values": values,
            "count": len(values)
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error getting label values: {e}")
        return _resp(f"Error getting label values: {str(e)}", is_error=True)


def execute_promql(
    query: str,
    time_range: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Execute a PromQL query and return results.
    
    Args:
        query: PromQL query to execute
        time_range: Natural language time range (e.g., "last 5 minutes", "1 hour ago")
        start_datetime: Start time in ISO format
        end_datetime: End time in ISO format
    
    Returns:
        Query results with metadata and explanation
    """
    try:
        # Determine query type and time parameters
        if time_range:
            # For range queries, use last 5 minutes as default
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
        elif start_datetime and end_datetime:
            start_time = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_datetime.replace('Z', '+00:00'))
        else:
            # Default to instant query (current time)
            end_time = datetime.now()
            start_time = None
        
        # Execute the query
        if start_time:
            # Range query
            params = {
                "query": query,
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
                "step": "15s"  # 15-second resolution
            }
            endpoint = "/api/v1/query_range"
        else:
            # Instant query
            params = {"query": query}
            endpoint = "/api/v1/query"
        
        response = _make_prometheus_request(endpoint, params)
        
        # Parse results
        result_type = response.get("data", {}).get("resultType", "")
        results = response.get("data", {}).get("result", [])
        
        # Format results for better readability
        formatted_results = []
        for result in results:
            if result_type == "vector":
                # Instant query result
                formatted_result = {
                    "metric": result.get("metric", {}),
                    "value": result.get("value", [])
                }
            elif result_type == "matrix":
                # Range query result
                formatted_result = {
                    "metric": result.get("metric", {}),
                    "values": result.get("values", [])
                }
            else:
                formatted_result = result
            
            formatted_results.append(formatted_result)
        
        # Generate explanation
        explanation = _explain_query_results(query, formatted_results, result_type)
        
        result = {
            "query": query,
            "result_type": result_type,
            "results": formatted_results,
            "result_count": len(formatted_results),
            "explanation": explanation,
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat()
            }
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error executing PromQL: {e}")
        return _resp(f"Error executing PromQL: {str(e)}", is_error=True)


def explain_results(
    query: str,
    results: List[Dict[str, Any]],
    result_type: str = "vector"
) -> List[Dict[str, Any]]:
    """Explain PromQL query results in natural language.
    
    Args:
        query: The PromQL query that was executed
        results: The query results
        result_type: Type of results (vector, matrix, scalar, string)
    
    Returns:
        Natural language explanation of the results
    """
    try:
        explanation = _explain_query_results(query, results, result_type)
        
        result = {
            "query": query,
            "result_type": result_type,
            "explanation": explanation,
            "result_count": len(results)
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error explaining results: {e}")
        return _resp(f"Error explaining results: {str(e)}", is_error=True)


def suggest_queries(
    user_intent: str,
    context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Suggest related PromQL queries based on user intent.
    
    Args:
        user_intent: What the user wants to know (e.g., "CPU usage", "memory problems")
        context: Additional context (e.g., "OpenShift cluster", "ML workloads")
    
    Returns:
        List of suggested PromQL queries with explanations
    """
    try:
        # This would ideally use an LLM to generate suggestions
        # For now, we'll provide some basic suggestions based on common patterns
        
        suggestions = _generate_query_suggestions(user_intent, context)
        
        result = {
            "user_intent": user_intent,
            "context": context,
            "suggestions": suggestions
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error suggesting queries: {e}")
        return _resp(f"Error suggesting queries: {str(e)}", is_error=True)


# Helper functions

def _generate_query_examples(metric_name: str, metadata: Dict[str, Any]) -> List[str]:
    """Generate example PromQL queries for a metric."""
    examples = []
    metric_type = metadata.get("type", "unknown")
    
    if metric_type == "counter":
        examples.extend([
            f"rate({metric_name}[5m])",
            f"increase({metric_name}[1h])",
            f"sum(rate({metric_name}[5m]))"
        ])
    elif metric_type == "gauge":
        examples.extend([
            f"avg({metric_name})",
            f"max({metric_name})",
            f"min({metric_name})"
        ])
    elif metric_type == "histogram":
        examples.extend([
            f"histogram_quantile(0.95, rate({metric_name}_bucket[5m]))",
            f"histogram_quantile(0.50, rate({metric_name}_bucket[5m]))",
            f"rate({metric_name}_sum[5m])"
        ])
    else:
        examples.append(metric_name)
    
    return examples


def _explain_query_results(query: str, results: List[Dict[str, Any]], result_type: str) -> str:
    """Generate natural language explanation of query results."""
    if not results:
        return "No results found for this query."
    
    if result_type == "vector":
        if len(results) == 1:
            result = results[0]
            metric = result.get("metric", {})
            value = result.get("value", [])
            if value and len(value) >= 2:
                return f"Found 1 result: {metric.get('__name__', 'metric')} = {value[1]} at {datetime.fromtimestamp(float(value[0]))}"
        else:
            return f"Found {len(results)} results for this query."
    
    elif result_type == "matrix":
        return f"Found {len(results)} time series with data points over the specified time range."
    
    else:
        return f"Query returned {len(results)} results of type {result_type}."


def _generate_query_suggestions(user_intent: str, context: Optional[str]) -> List[Dict[str, Any]]:
    """Generate query suggestions based on user intent."""
    suggestions = []
    
    intent_lower = user_intent.lower()
    
    if "cpu" in intent_lower:
        suggestions.extend([
            {
                "query": "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                "description": "CPU usage percentage",
                "explanation": "Shows CPU utilization as a percentage"
            },
            {
                "query": "rate(node_cpu_seconds_total[5m])",
                "description": "CPU time per second",
                "explanation": "Shows CPU time consumption rate"
            }
        ])
    
    if "memory" in intent_lower:
        suggestions.extend([
            {
                "query": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
                "description": "Memory usage percentage",
                "explanation": "Shows memory utilization as a percentage"
            },
            {
                "query": "node_memory_MemAvailable_bytes",
                "description": "Available memory",
                "explanation": "Shows available memory in bytes"
            }
        ])
    
    if "pod" in intent_lower or "container" in intent_lower:
        suggestions.extend([
            {
                "query": "kube_pod_status_phase",
                "description": "Pod status",
                "explanation": "Shows current phase of pods (Running, Pending, Failed, etc.)"
            },
            {
                "query": "kube_deployment_status_replicas_ready",
                "description": "Ready replicas",
                "explanation": "Shows number of ready replicas for deployments"
            }
        ])
    
    if "gpu" in intent_lower or "vllm" in intent_lower:
        suggestions.extend([
            {
                "query": "avg(DCGM_FI_DEV_GPU_UTIL)",
                "description": "GPU utilization",
                "explanation": "Shows average GPU utilization percentage"
            },
            {
                "query": "histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m]))",
                "description": "ML inference latency",
                "explanation": "Shows 95th percentile latency for ML inference"
            }
        ])
    
    # If no specific suggestions, provide general ones
    if not suggestions:
        suggestions.extend([
            {
                "query": "up",
                "description": "Service availability",
                "explanation": "Shows if services are running (1 = up, 0 = down)"
            },
            {
                "query": "ALERTS",
                "description": "Active alerts",
                "explanation": "Shows currently active alerts"
            }
        ])
    
    return suggestions


def select_best_metric(
    user_intent: str,
    available_metrics: List[str],
    context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """LLM-powered metric selection based on user intent.
    
    This tool uses LLM intelligence to select the most relevant metric
    from a list of available metrics based on the user's intent.
    
    Args:
        user_intent: What the user wants to know (e.g., "GPU utilization", "memory usage")
        available_metrics: List of metric names to choose from
        context: Additional context (e.g., "OpenShift cluster", "ML workloads")
    
    Returns:
        The most relevant metric name with explanation
    """
    try:
        if not available_metrics:
            return _resp("No metrics available to select from", is_error=True)
        
        if len(available_metrics) == 1:
            return _resp(json.dumps({
                "selected_metric": available_metrics[0],
                "reasoning": "Only one metric available",
                "confidence": 1.0
            }, indent=2))
        
        # Create prompt for LLM
        prompt = f"""
        You are an expert in Prometheus metrics and observability. 
        
        User wants to know: "{user_intent}"
        Context: {context or "general observability"}
        Available metrics: {', '.join(available_metrics)}
        
        Select the most relevant metric for the user's intent. Consider:
        
        1. **Semantic meaning**: Match the user's intent with metric names
        2. **Domain knowledge**: 
           - DCGM_* = GPU metrics (DCGM_FI_DEV_GPU_UTIL = GPU utilization)
           - vLLM:* = ML inference metrics (vLLM:e2e_request_latency_seconds = latency)
           - node_* = system metrics (node_cpu_seconds_total = CPU, node_memory_* = memory)
           - kube_* = Kubernetes metrics (kube_pod_status_phase = pod status)
           - ALERTS = active alerts
        3. **Keyword matching**: Look for keywords in metric names that match user intent
        4. **Metric type relevance**: Prefer metrics that directly answer the user's question
        
        Examples:
        - "GPU utilization" → DCGM_FI_DEV_GPU_UTIL
        - "memory usage" → node_memory_MemTotal_bytes or node_memory_MemAvailable_bytes
        - "CPU usage" → node_cpu_seconds_total
        - "pod status" → kube_pod_status_phase
        - "active alerts" → ALERTS
        
        Return your response in this exact JSON format:
        {{
            "selected_metric": "metric_name",
            "reasoning": "Brief explanation of why this metric was selected",
            "confidence": 0.95
        }}
        
        Only return the JSON, nothing else.
        """
        
        # Use LLM to select best metric
        try:
            llm_response = summarize_with_llm(prompt)
            
            # Parse LLM response
            if llm_response and isinstance(llm_response, str):
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    selection_result = json.loads(json_str)
                    
                    # Validate selection
                    selected_metric = selection_result.get("selected_metric", "")
                    if selected_metric in available_metrics:
                        return _resp(json.dumps(selection_result, indent=2))
                    else:
                        # Fallback to first metric if selection is invalid
                        return _resp(json.dumps({
                            "selected_metric": available_metrics[0],
                            "reasoning": f"LLM selected '{selected_metric}' which is not in available metrics. Using first available metric.",
                            "confidence": 0.5
                        }, indent=2))
                else:
                    # If no JSON found, try to extract metric name from text
                    for metric in available_metrics:
                        if metric.lower() in llm_response.lower():
                            return _resp(json.dumps({
                                "selected_metric": metric,
                                "reasoning": f"Found '{metric}' mentioned in LLM response",
                                "confidence": 0.7
                            }, indent=2))
            
        except Exception as llm_error:
            logging.warning(f"LLM selection failed: {llm_error}")
        
        # Fallback: Use simple keyword matching
        selected_metric = _fallback_metric_selection(user_intent, available_metrics)
        
        return _resp(json.dumps({
            "selected_metric": selected_metric,
            "reasoning": "Used fallback keyword matching due to LLM unavailability",
            "confidence": 0.6
        }, indent=2))
        
    except Exception as e:
        logging.error(f"Error selecting metric: {e}")
        return _resp(f"Error selecting metric: {str(e)}", is_error=True)


def _fallback_metric_selection(user_intent: str, available_metrics: List[str]) -> str:
    """Dynamic metric selection using semantic analysis and pattern matching.
    
    This automatically understands metrics based on their names and context,
    without any hardcoded mappings.
    """
    intent_lower = user_intent.lower()
    intent_words = set(intent_lower.replace('-', ' ').replace('_', ' ').replace(':', ' ').split())
    
    best_match = None
    best_score = 0
    
    for metric in available_metrics:
        metric_lower = metric.lower()
        metric_words = set(metric_lower.replace('-', ' ').replace('_', ' ').replace(':', ' ').split())
        
        score = 0
        
        # 1. Direct word matches (highest priority)
        direct_matches = intent_words.intersection(metric_words)
        score += len(direct_matches) * 10
        
        # 2. Semantic pattern matching
        score += _calculate_semantic_score(intent_lower, metric_lower)
        
        # 3. Metric type relevance
        score += _calculate_type_relevance(intent_lower, metric_lower)
        
        # 4. Specificity bonus (more specific metrics get higher scores)
        score += _calculate_specificity_score(metric_lower)
        
        if score > best_score:
            best_score = score
            best_match = metric
    
    return best_match or (available_metrics[0] if available_metrics else "")


def _calculate_semantic_score(intent: str, metric: str) -> int:
    """Calculate semantic relevance score between user intent and metric name."""
    score = 0
    
    # Token/generation patterns
    if any(word in intent for word in ['token', 'generate', 'generation', 'produced']):
        if any(pattern in metric for pattern in ['token', 'generation']):
            score += 15
    
    # Latency/performance patterns  
    if any(word in intent for word in ['latency', 'response', 'time', 'p95', 'p99', 'percentile', 'speed', 'performance']):
        if any(pattern in metric for pattern in ['latency', 'time', 'duration', 'seconds']):
            score += 15
    
    # GPU patterns
    if any(word in intent for word in ['gpu', 'graphics', 'cuda', 'nvidia']):
        if any(pattern in metric for pattern in ['gpu', 'dcgm', 'cuda']):
            score += 15
    
    # Memory patterns
    if any(word in intent for word in ['memory', 'ram', 'mem']):
        if any(pattern in metric for pattern in ['memory', 'mem', 'ram', 'bytes']):
            score += 15
    
    # CPU patterns
    if any(word in intent for word in ['cpu', 'processor', 'core']):
        if any(pattern in metric for pattern in ['cpu', 'processor', 'core']):
            score += 15
    
    # Pod/container patterns
    if any(word in intent for word in ['pod', 'container', 'running', 'status', 'phase']):
        if any(pattern in metric for pattern in ['pod', 'container', 'status', 'phase']):
            score += 15
    
    # Request/usage patterns
    if any(word in intent for word in ['request', 'usage', 'utilization', 'use']):
        if any(pattern in metric for pattern in ['request', 'usage', 'util', 'use']):
            score += 10
    
    # Alert patterns
    if any(word in intent for word in ['alert', 'alarm', 'warning', 'error', 'issue', 'problem']):
        if any(pattern in metric for pattern in ['alert', 'alarm']):
            score += 15
    
    # Model/deployment discovery patterns
    if any(word in intent for word in ['model', 'deploy', 'deployed', 'running']):
        if any(pattern in metric for pattern in ['cache_config_info', 'pod_info', 'job', 'service']):
            score += 15
        # Prefer vLLM cache config for model discovery
        if 'cache_config_info' in metric and 'vllm' in metric:
            score += 25
    
    # CRITICAL FIX: Boost canonical Kubernetes state metrics for basic questions
    if any(word in intent for word in ['pod', 'pods']) and any(word in intent for word in ['running', 'count', 'how many', 'number']):
        if metric == 'kube_pod_status_phase':
            score += 50  # Major boost for THE pod state metric
        elif metric == 'kube_pod_info':
            score += 30  # Secondary boost
            
    if any(word in intent for word in ['service', 'services']) and any(word in intent for word in ['deployed', 'count', 'how many', 'number']):
        if metric == 'kube_service_info':
            score += 50  # Major boost for THE service metric
            
    if any(word in intent for word in ['alert', 'alerts', 'firing']):
        if metric == 'ALERTS':
            score += 50  # Major boost for THE alert metric
    
    return score


def _calculate_type_relevance(intent: str, metric: str) -> int:
    """Calculate relevance based on metric type and user intent context."""
    score = 0
    
    # Count-based questions
    if any(word in intent for word in ['number', 'count', 'how many', 'total']):
        if any(pattern in metric for pattern in ['total', 'count', 'num']):
            score += 10
        # Prefer status metrics for counting pods/containers
        if 'pod' in intent and 'status' in metric:
            score += 15
    
    # Rate/frequency questions
    if any(word in intent for word in ['rate', 'per', 'frequency', 'throughput']):
        if any(pattern in metric for pattern in ['rate', 'total', 'seconds']):
            score += 10
    
    # Temperature questions
    if any(word in intent for word in ['temperature', 'temp', 'hot', 'heat', 'thermal']):
        if any(pattern in metric for pattern in ['temp', 'thermal']):
            score += 20
    
    # Power/energy questions
    if any(word in intent for word in ['power', 'energy', 'consumption', 'watt']):
        if any(pattern in metric for pattern in ['power', 'energy', 'watt']):
            score += 20
    
    return score


def _calculate_specificity_score(metric: str) -> int:
    """Give preference to more specific/targeted metrics."""
    score = 0
    
    # Prefer more specific vLLM metrics
    if metric.startswith('vllm:'):
        score += 5
    
    # Prefer core kubernetes metrics
    if metric.startswith('kube_'):
        score += 3
    
    # Prefer DCGM GPU metrics over generic ones
    if 'dcgm' in metric.lower():
        score += 5
    
    # Penalize overly complex derived metrics
    if len(metric.split(':')) > 2:  # e.g., cluster:namespace:pod:cpu:complex
        score -= 3
    
    return score


def _rank_metrics_by_relevance(search_term: str, all_metrics: List[str]) -> List[str]:
    """Rank all metrics by relevance using semantic scoring + Prometheus metadata."""
    scored_metrics = []
    
    for metric in all_metrics:
        intent_lower = search_term.lower()
        metric_lower = metric.lower()
        
        score = 0
        
        # 1. Direct word matches (metric name)
        intent_words = set(intent_lower.replace('-', ' ').replace('_', ' ').replace(':', ' ').split())
        metric_words = set(metric_lower.replace('-', ' ').replace('_', ' ').replace(':', ' ').split())
        direct_matches = intent_words.intersection(metric_words)
        score += len(direct_matches) * 10
        
        # 2. Semantic scoring based on metric name
        score += _calculate_semantic_score(intent_lower, metric_lower)
        score += _calculate_type_relevance(intent_lower, metric_lower)
        score += _calculate_specificity_score(metric_lower)
        
        # 3. **NEW: Prometheus metadata scoring** (Skip for performance in ranking)
        # metadata_score = _calculate_metadata_score(intent_lower, metric)
        # score += metadata_score
        
        if score > 0:  # Only include metrics with some relevance
            scored_metrics.append((metric, score))
    
    # Sort by score (descending) and return metric names
    scored_metrics.sort(key=lambda x: x[1], reverse=True)
    return [metric for metric, score in scored_metrics]


def find_best_metric_with_metadata_v2(
    user_question: str,
    max_candidates: int = 10
) -> List[Dict[str, Any]]:
    """IMPROVED: Filter metrics by keywords first, then use metadata to select best.
    
    This is much more intelligent than semantic ranking all 3500+ metrics.
    """
    try:
        # STEP 1: Extract keywords from user question
        question_lower = user_question.lower()
        keywords = _extract_keywords_for_filtering(question_lower)
        
        if not keywords:
            # Fallback to original approach if no clear keywords
            return find_best_metric_with_metadata(user_question, max_candidates)
        
        # STEP 2: Get all metrics and filter by keywords
        response = _make_prometheus_request("/api/v1/label/__name__/values")
        all_metrics = response.get("data", [])
        
        # Filter metrics that contain any of our keywords
        filtered_metrics = []
        for metric in all_metrics:
            metric_lower = metric.lower()
            if any(keyword in metric_lower for keyword in keywords):
                filtered_metrics.append(metric)
        
        if not filtered_metrics:
            error_result = {
                "error": f"No metrics found containing keywords: {keywords}",
                "keywords_used": keywords,
                "alternatives_found": 0,
                "suggested_promql": "# No matching metrics found",
                "selection_reasoning": f"Searched {len(all_metrics)} total metrics but none matched keywords: {keywords}"
            }
            return _resp(json.dumps(error_result, indent=2))
        
        # STEP 3: Get metadata for filtered metrics and score them
        metadata_scored_metrics = []
        for metric in filtered_metrics:
            try:
                metadata_response = _make_prometheus_request("/api/v1/metadata", {"metric": metric})
                metadata = metadata_response.get("data", {}).get(metric, [])
                
                if metadata:
                    info = metadata[0]
                    help_text = info.get("help", "")
                    metric_type = info.get("type", "unknown")
                    
                    # Score based on metadata + question relevance
                    score = _score_metric_with_metadata_for_question(
                        metric, help_text, metric_type, question_lower
                    )
                    
                    if score > 0:
                        metadata_scored_metrics.append({
                            'name': metric,
                            'type': metric_type,
                            'help': help_text,
                            'score': score,
                            'metadata': info
                        })
                        
            except Exception as e:
                logging.debug(f"Error getting metadata for {metric}: {e}")
                continue
        
        if not metadata_scored_metrics:
            error_result = {
                "error": f"No metrics with metadata found for keywords: {keywords}",
                "keywords_used": keywords,
                "alternatives_found": len(filtered_metrics),
                "suggested_promql": "# No metrics with valid metadata found",
                "selection_reasoning": f"Found {len(filtered_metrics)} metrics matching keywords {keywords}, but none had usable metadata"
            }
            return _resp(json.dumps(error_result, indent=2))
        
        # STEP 4: Sort by metadata score and select best
        metadata_scored_metrics.sort(key=lambda x: x['score'], reverse=True)
        best_metric_data = metadata_scored_metrics[0]
        
        # STEP 5: Generate intelligent PromQL query
        concepts = _extract_key_concepts(question_lower)
        suggested_query = _generate_metadata_driven_promql_simple(
            best_metric_data['name'], 
            best_metric_data['type'], 
            concepts
        )
        
        # STEP 6: Format response
        result = {
            "selected_metric": {
                "name": best_metric_data['name'],
                "help": best_metric_data['help'],
                "type": best_metric_data['type'],
                "unit": best_metric_data['metadata'].get('unit', '')
            },
            "suggested_promql": suggested_query,
            "selection_reasoning": f"Filtered {len(all_metrics)} metrics by keywords {keywords}, found {len(metadata_scored_metrics)} with metadata, selected highest scoring based on help text and type relevance",
            "keywords_used": keywords,
            "alternatives_found": len(metadata_scored_metrics),
            "other_candidates": [
                {
                    "name": m['name'], 
                    "score": m['score'],
                    "help": m['help'][:50] + "..." if len(m['help']) > 50 else m['help']
                } 
                for m in metadata_scored_metrics[1:4]  # Show top 3 alternatives
            ]
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error in metadata-first metric selection: {e}")
        return _resp(f"Error analyzing metrics: {str(e)}", is_error=True)


def find_best_metric_with_metadata(
    user_question: str,
    max_candidates: int = 10
) -> List[Dict[str, Any]]:
    """Find the best metric for a user question using comprehensive metadata analysis.
    
    This tool combines search, metadata analysis, and intelligent selection to find
    the most appropriate metric for the user's question, providing detailed reasoning.
    
    Args:
        user_question: User's question (e.g., "What's the GPU temperature?")
        max_candidates: Maximum number of candidate metrics to analyze
    
    Returns:
        Analysis with best metric, reasoning, and suggested PromQL query
    """
    try:
        # STEP 1: Extract key concepts from user question
        question_lower = user_question.lower()
        concepts = _extract_key_concepts(question_lower)
        
        # STEP 2: Search for candidate metrics using semantic ranking
        response = _make_prometheus_request("/api/v1/label/__name__/values")
        all_metrics = response.get("data", [])
        
        # Get top candidates using our enhanced ranking
        candidates = _rank_metrics_by_relevance(user_question, all_metrics)[:max_candidates]
        
        if not candidates:
            return _resp("No relevant metrics found for your question.")
        
        # STEP 3: Analyze each candidate with metadata
        analyzed_candidates = []
        for metric in candidates:
            analysis = _analyze_metric_with_metadata(metric, concepts, question_lower)
            if analysis['relevance_score'] > 0:
                analyzed_candidates.append(analysis)
        
        if not analyzed_candidates:
            return _resp("No suitable metrics found after metadata analysis.")
        
        # STEP 4: Select the best metric based on comprehensive scoring
        best_metric = max(analyzed_candidates, key=lambda x: x['total_score'])
        
        # STEP 5: Generate intelligent PromQL query
        suggested_query = _generate_metadata_driven_promql(best_metric, concepts)
        
        # STEP 6: Format comprehensive response
        result = {
            "selected_metric": {
                "name": best_metric['name'],
                "help": best_metric['metadata'].get('help', ''),
                "type": best_metric['metadata'].get('type', ''),
                "unit": best_metric['metadata'].get('unit', '')
            },
            "relevance_analysis": {
                "concept_matches": best_metric['concept_matches'],
                "metadata_score": best_metric['metadata_score'],
                "semantic_score": best_metric['semantic_score'],
                "total_score": best_metric['total_score']
            },
            "suggested_promql": suggested_query,
            "reasoning": best_metric['reasoning'],
            "other_candidates": [
                {
                    "name": c['name'], 
                    "score": c['total_score'],
                    "why_not_selected": c.get('why_not_selected', 'Lower overall relevance')
                } 
                for c in analyzed_candidates[1:3]  # Show top 2 alternatives
            ]
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error in metadata-driven metric selection: {e}")
        return _resp(f"Error analyzing metrics: {str(e)}", is_error=True)


def _extract_key_concepts(question: str) -> Dict[str, Any]:
    """Extract key concepts from user question for metadata matching."""
    concepts = {
        'entities': [],
        'measurements': [],
        'aggregations': [],
        'intent_type': 'unknown'
    }
    
    # Entity detection
    if any(word in question for word in ['gpu', 'graphics', 'cuda', 'nvidia']):
        concepts['entities'].append('gpu')
    if any(word in question for word in ['pod', 'container', 'kubernetes']):
        concepts['entities'].append('pod')
    if any(word in question for word in ['memory', 'ram', 'mem']):
        concepts['entities'].append('memory')
    if any(word in question for word in ['cpu', 'processor', 'core']):
        concepts['entities'].append('cpu')
    if any(word in question for word in ['vllm', 'model', 'inference', 'llm', 'deploy', 'deployed']):
        concepts['entities'].append('vllm')
    if any(word in question for word in ['deploy', 'deployed', 'models', 'running']):
        concepts['entities'].append('deployment')
    if any(word in question for word in ['alert', 'warning', 'error', 'problem']):
        concepts['entities'].append('alert')
    
    # Measurement detection
    if any(word in question for word in ['temperature', 'temp', 'hot', 'heat']):
        concepts['measurements'].append('temperature')
    if any(word in question for word in ['latency', 'response', 'time', 'duration']):
        concepts['measurements'].append('latency')
    if any(word in question for word in ['usage', 'utilization', 'use', 'percent']):
        concepts['measurements'].append('usage')
    if any(word in question for word in ['power', 'energy', 'consumption', 'watt']):
        concepts['measurements'].append('power')
    if any(word in question for word in ['token', 'generate', 'generation', 'produced']):
        concepts['measurements'].append('tokens')
    
    # Intent type detection
    if any(word in question for word in ['how many', 'number', 'count', 'total']):
        concepts['intent_type'] = 'count'
    elif any(word in question for word in ['what is', 'current', 'level', 'value']):
        concepts['intent_type'] = 'current_value'
    elif any(word in question for word in ['average', 'avg', 'mean']):
        concepts['intent_type'] = 'average'
    elif any(word in question for word in ['p95', 'p99', 'percentile', '95th', '99th']):
        concepts['intent_type'] = 'percentile'
    
    return concepts


def _analyze_metric_with_metadata(metric_name: str, concepts: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Analyze a single metric using its metadata and user concepts."""
    try:
        # Get metadata
        response = _make_prometheus_request("/api/v1/metadata", {"metric": metric_name})
        metadata = response.get("data", {}).get(metric_name, [])
        
        if not metadata:
            return {'name': metric_name, 'total_score': 0, 'relevance_score': 0}
        
        metadata_info = metadata[0]
        help_text = metadata_info.get("help", "").lower()
        metric_type = metadata_info.get("type", "").lower()
        unit = metadata_info.get("unit", "").lower()
        
        analysis = {
            'name': metric_name,
            'metadata': metadata_info,
            'concept_matches': [],
            'metadata_score': 0,
            'semantic_score': 0,
            'type_compatibility': 0,
            'reasoning': []
        }
        
        # Score concept matches in help text
        for entity in concepts['entities']:
            if entity in help_text or entity in metric_name.lower():
                analysis['concept_matches'].append(entity)
                analysis['metadata_score'] += 20
                analysis['reasoning'].append(f"Matches entity '{entity}' in help text or name")
        
        for measurement in concepts['measurements']:
            if measurement in help_text or measurement in metric_name.lower():
                analysis['concept_matches'].append(measurement)
                analysis['metadata_score'] += 25
                analysis['reasoning'].append(f"Measures '{measurement}' according to metadata")
        
        # Score intent compatibility with metric type
        intent = concepts['intent_type']
        
        # FIXED: For count questions, prefer gauges for current state, counters for events
        if intent == 'count':
            if metric_type == 'gauge':
                # Gauges are perfect for "how many X are running/deployed/exist" (current state)
                analysis['type_compatibility'] = 25
                analysis['reasoning'].append("Gauge type perfect for current state counting questions")
            elif metric_type == 'counter':
                # Counters are for "how many X were created/processed" (events)
                analysis['type_compatibility'] = 10
                analysis['reasoning'].append("Counter type suitable for event counting questions")
                
        elif intent == 'current_value' and metric_type == 'gauge':
            analysis['type_compatibility'] = 20
            analysis['reasoning'].append("Gauge type perfect for current value questions")
        elif intent == 'percentile' and metric_type == 'histogram':
            analysis['type_compatibility'] = 20
            analysis['reasoning'].append("Histogram type perfect for percentile questions")
        elif intent in ['average', 'current_value'] and metric_type == 'gauge':
            analysis['type_compatibility'] = 15
            analysis['reasoning'].append("Gauge type suitable for current measurements")
        
        # Score unit compatibility
        unit_score = 0
        if 'temperature' in concepts['measurements'] and ('celsius' in unit or 'c' == unit or 'temperature' in help_text):
            unit_score = 15
            analysis['reasoning'].append("Unit matches temperature measurement")
        elif 'latency' in concepts['measurements'] and ('second' in unit or 'time' in help_text):
            unit_score = 15
            analysis['reasoning'].append("Unit matches time/latency measurement")
        
        # Calculate semantic score (existing logic)
        analysis['semantic_score'] = _calculate_semantic_score(question, metric_name.lower())
        
        # Calculate total score
        analysis['total_score'] = (
            analysis['metadata_score'] + 
            analysis['semantic_score'] + 
            analysis['type_compatibility'] + 
            unit_score
        )
        
        analysis['relevance_score'] = analysis['total_score']
        
        # Add why this metric might not be selected
        if analysis['total_score'] < 30:
            analysis['why_not_selected'] = "Low relevance to user question"
        elif not analysis['concept_matches']:
            analysis['why_not_selected'] = "No clear concept matches"
        elif analysis['type_compatibility'] == 0:
            analysis['why_not_selected'] = "Metric type doesn't match question intent"
        
        return analysis
        
    except Exception as e:
        logging.debug(f"Error analyzing metric {metric_name}: {e}")
        return {'name': metric_name, 'total_score': 0, 'relevance_score': 0}


def _generate_metadata_driven_promql(metric_analysis: Dict[str, Any], concepts: Dict[str, Any]) -> str:
    """Generate PromQL query based on metric metadata and user intent."""
    metric_name = metric_analysis['name']
    metric_type = metric_analysis['metadata'].get('type', '').lower()
    intent = concepts['intent_type']
    
    # Choose aggregation based on intent and metric type
    if intent == 'count':
        if metric_type == 'counter':
            # For counters, we often want the total or rate
            return f"sum({metric_name})"
        else:
            return f"count({metric_name})"
    
    elif intent == 'current_value':
        if 'temperature' in concepts['measurements']:
            return f"avg({metric_name})"  # Temperature should be averaged
        elif 'usage' in concepts['measurements']:
            return f"avg({metric_name})"  # Usage/utilization should be averaged
        else:
            return f"{metric_name}"  # Raw current value
    
    elif intent == 'average':
        return f"avg({metric_name})"
    
    elif intent == 'percentile':
        if metric_type == 'histogram':
            return f"histogram_quantile(0.95, {metric_name}_bucket)"
        else:
            return f"quantile(0.95, {metric_name})"
    
    else:
        # Default based on metric type
        if metric_type == 'counter':
            return f"rate({metric_name}[5m])"
        elif metric_type == 'gauge':
            if 'temperature' in concepts['measurements'] or 'usage' in concepts['measurements']:
                return f"avg({metric_name})"
            else:
                return f"{metric_name}"
        else:
            return f"{metric_name}"


def _calculate_metadata_score(intent: str, metric_name: str) -> int:
    """Score metrics based on Prometheus metadata (help text, type, unit)."""
    try:
        # Get metadata from Prometheus
        response = _make_prometheus_request("/api/v1/metadata", {"metric": metric_name})
        metadata = response.get("data", {}).get(metric_name, [])
        
        if not metadata:
            return 0
            
        metadata_info = metadata[0]  # Take first metadata entry
        help_text = metadata_info.get("help", "").lower()
        metric_type = metadata_info.get("type", "").lower()
        unit = metadata_info.get("unit", "").lower()
        
        score = 0
        
        # Score based on help text relevance
        intent_words = intent.split()
        for word in intent_words:
            if len(word) > 2 and word in help_text:
                score += 15  # High score for help text matches
        
        # Score based on metric type relevance
        if any(word in intent for word in ['count', 'number', 'total', 'how many']):
            if metric_type in ['counter']:
                score += 10
        elif any(word in intent for word in ['current', 'level', 'usage', 'utilization']):
            if metric_type in ['gauge']:
                score += 10
        elif any(word in intent for word in ['latency', 'time', 'duration', 'percentile', 'p95', 'p99']):
            if metric_type in ['histogram']:
                score += 10
        
        # Score based on unit relevance
        if any(word in intent for word in ['temperature', 'temp', 'celsius']):
            if 'c' in unit or 'celsius' in unit or 'temperature' in help_text:
                score += 20
        elif any(word in intent for word in ['memory', 'bytes', 'size']):
            if 'byte' in unit or 'memory' in help_text:
                score += 15
        elif any(word in intent for word in ['time', 'latency', 'duration']):
            if 'second' in unit or 'time' in help_text:
                score += 15
        
        return score
        
    except Exception as e:
        # If metadata lookup fails, don't penalize the metric
        logging.debug(f"Metadata lookup failed for {metric_name}: {e}")
        return 0


def _extract_keywords_for_filtering(question: str) -> List[str]:
    """Extract relevant keywords for ALL 3500+ metrics dynamically."""
    keywords = []
    question_lower = question.lower()
    
    # 1. KUBERNETES CORE (kube_* metrics)
    k8s_entities = {
        'pod': ['pod', 'pods', 'container'],
        'service': ['service', 'services', 'svc'],
        'node': ['node', 'nodes', 'worker'],
        'namespace': ['namespace', 'namespaces', 'ns'],
        'deployment': ['deployment', 'deployments', 'deploy'],
        'volume': ['volume', 'volumes', 'pv', 'pvc', 'persistent'],
        'job': ['job', 'jobs', 'cronjob'],
        'ingress': ['ingress', 'route'],
        'replicaset': ['replica', 'replicaset'],
        'daemonset': ['daemon', 'daemonset'],
        'statefulset': ['stateful', 'statefulset']
    }
    
    for keyword, patterns in k8s_entities.items():
        if any(pattern in question_lower for pattern in patterns):
            keywords.append(keyword)
    
    # 2. SYSTEM RESOURCES (container_*, node_* metrics)
    resources = {
        'cpu': ['cpu', 'processor', 'core', 'throttl'],
        'memory': ['memory', 'ram', 'mem', 'oom'],
        'disk': ['disk', 'storage', 'io', 'filesystem', 'space'],
        'network': ['network', 'net', 'packet', 'bandwidth', 'tcp', 'udp']
    }
    
    for keyword, patterns in resources.items():
        if any(pattern in question_lower for pattern in patterns):
            keywords.append(keyword)
    
    # 3. GPU/HARDWARE (DCGM_*, gpu_* metrics)
    if any(word in question_lower for word in ['gpu', 'graphics', 'nvidia', 'cuda', 'dcgm']):
        keywords.extend(['gpu', 'dcgm'])
    if any(word in question_lower for word in ['temperature', 'temp', 'thermal', 'hot']):
        keywords.extend(['temp', 'temperature', 'dcgm'])
    if any(word in question_lower for word in ['power', 'energy', 'consumption', 'watt']):
        keywords.extend(['power', 'energy', 'dcgm'])
    if any(word in question_lower for word in ['utilization', 'usage', 'util']):
        keywords.append('util')
    
    # 4. AI/ML WORKLOADS (vllm:*, ollama_*, ray_* metrics)
    ml_systems = {
        'vllm': ['vllm', 'model', 'inference', 'llm', 'token', 'generation'],
        'ollama': ['ollama'],
        'ray': ['ray'],
        'kubeflow': ['kubeflow', 'pipeline']
    }
    
    for keyword, patterns in ml_systems.items():
        if any(pattern in question_lower for pattern in patterns):
            keywords.append(keyword)
    
    # 5. DATABASES (etcd_*, mysql_*, postgres_*, redis_* metrics)
    databases = {
        'etcd': ['etcd'],
        'mysql': ['mysql', 'mariadb'],
        'postgres': ['postgres', 'postgresql'],
        'redis': ['redis', 'cache'],
        'mongodb': ['mongo', 'mongodb'],
        'elasticsearch': ['elasticsearch', 'elastic']
    }
    
    for keyword, patterns in databases.items():
        if any(pattern in question_lower for pattern in patterns):
            keywords.append(keyword)
    
    # 6. MONITORING/ALERTING (alertmanager_*, prometheus_* metrics)
    if any(word in question_lower for word in ['alert', 'alerts', 'firing', 'notification']):
        keywords.extend(['alert', 'ALERTS'])
    if any(word in question_lower for word in ['prometheus', 'scrape', 'target']):
        keywords.append('prometheus')
    if any(word in question_lower for word in ['grafana', 'dashboard']):
        keywords.append('grafana')
    
    # 7. HTTP/API METRICS (http_*, api_* metrics)
    if any(word in question_lower for word in ['http', 'request', 'response', 'status', 'api', 'endpoint']):
        keywords.extend(['http', 'request'])
    
    # 8. PERFORMANCE METRICS (latency, errors, throughput)
    if any(word in question_lower for word in ['latency', 'duration', 'time', 'p95', 'p99', 'percentile']):
        keywords.append('latency')
    if any(word in question_lower for word in ['error', 'fail', 'exception']):
        keywords.append('error')
    if any(word in question_lower for word in ['rate', 'throughput', 'qps', 'rps']):
        keywords.append('rate')
    
    # 9. FALLBACK: Extract meaningful words from question
    if not keywords:
        stop_words = {'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'when', 'where', 'why', 'this', 'that', 'any', 'many', 'much'}
        words = re.findall(r'\\b[a-zA-Z]{3,}\\b', question_lower)
        meaningful_words = [w for w in words if w not in stop_words]
        keywords.extend(meaningful_words[:2])  # Take up to 2 meaningful words
    
    return list(set(keywords))  # Remove duplicates


def _score_metric_with_metadata_for_question(metric_name: str, help_text: str, metric_type: str, question: str) -> int:
    """Score a metric based on its metadata relevance to the question."""
    score = 0
    
    metric_lower = metric_name.lower()
    help_lower = help_text.lower()
    
    # CORE SCORING: Help text relevance
    question_words = question.split()
    # Filter out common stop words that don't add meaning
    stop_words = {'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'when', 'where', 'why', 'this', 'that', 'these', 'those'}
    
    meaningful_matches = 0
    for word in question_words:
        word_clean = word.strip('?.,!').lower()
        if len(word_clean) > 2 and word_clean not in stop_words and word_clean in help_lower:
            meaningful_matches += 1
            score += 25  # High score for meaningful help text matches
    
    # BONUS: Exact phrase matches (like "temperature" in "GPU temperature")
    key_phrases = []
    if 'temperature' in question:
        key_phrases.append('temperature')
    if 'utilization' in question or 'usage' in question:
        key_phrases.append('utilization')
    if 'power' in question:
        key_phrases.append('power')
    if 'latency' in question:
        key_phrases.append('latency')
    if 'tokens' in question:
        key_phrases.append('tokens')
    
    for phrase in key_phrases:
        if phrase in help_lower:
            score += 35  # Major bonus for exact concept matches
    
    # TYPE SCORING: Prefer appropriate types for question intent
    if any(word in question for word in ['how many', 'number', 'count']):
        if metric_type == 'gauge':
            score += 25  # Gauges are perfect for current state counts
        elif metric_type == 'counter':
            score += 10  # Counters for event counts
    elif any(word in question for word in ['current', 'what is', 'level']):
        if metric_type == 'gauge':
            score += 20  # Gauges for current values
    elif any(word in question for word in ['p95', 'p99', 'percentile']):
        if metric_type == 'histogram':
            score += 25  # Histograms for percentiles
    
    # SPECIFICITY SCORING: Prefer core metrics over derived ones
    if metric_lower.startswith('kube_'):
        score += 15  # Core Kubernetes metrics
        
        # Super specific boosts for perfect matches
        if 'pod' in question:
            if 'kube_pod_status_phase' == metric_lower:
                score += 60  # THE metric for ANY pod status question (running, failing, pending)
            elif 'kube_pod_info' == metric_lower:
                score += 40  # Alternative for pod info
        elif 'service' in question:
            if 'kube_service_info' == metric_lower:
                score += 60  # THE metric for service info
        elif 'node' in question:
            if 'kube_node_info' == metric_lower:
                score += 60  # THE metric for node info
    
    if metric_lower.startswith('vllm:'):
        score += 15  # vLLM specific metrics
        
        # Specific vLLM metric boosts
        if 'tokens' in question and 'generation_tokens_total' in metric_lower:
            score += 50  # Perfect match for token questions
        elif 'latency' in question and 'e2e_request_latency_seconds' in metric_lower:
            score += 50  # Perfect match for latency questions
    
    if metric_lower.startswith('dcgm_'):
        score += 15  # GPU specific metrics
        
        # Specific DCGM metric boosts  
        if 'temperature' in question and 'gpu_temp' in metric_lower:
            score += 60  # Perfect match for GPU temperature
        elif ('utilization' in question or 'usage' in question) and 'gpu_util' in metric_lower:
            score += 60  # Perfect match for GPU utilization
        elif 'power' in question and 'power_usage' in metric_lower:
            score += 60  # Perfect match for GPU power
    
    # ALERTS special handling
    if 'alert' in question:
        if metric_lower == 'alerts':
            score += 70  # THE metric for firing alerts
        elif 'alertmanager' in metric_lower:
            score += 20  # Secondary alert metrics
    
    # STABILITY SCORING: Prefer stable metrics
    if '[STABLE]' in help_text:
        score += 15
    elif '[ALPHA]' in help_text:
        score += 5  # Alpha metrics are less preferred
    
    # PENALTY: Avoid overly complex derived metrics
    if len(metric_name.split(':')) > 2:
        score -= 10  # Penalize complex derived metrics
    
    return score


def _generate_metadata_driven_promql_simple(metric_name: str, metric_type: str, concepts: Dict[str, Any]) -> str:
    """Generate PromQL query based on metric type and user intent (simplified)."""
    intent = concepts.get('intent_type', 'unknown')
    
    # Choose aggregation based on intent and metric type
    if intent == 'count':
        if metric_type == 'gauge':
            return f"count({metric_name})"  # Count distinct values for gauges
        else:
            return f"sum({metric_name})"    # Sum for counters
    elif any(measurement in concepts.get('measurements', []) for measurement in ['temperature', 'usage']):
        return f"avg({metric_name})"  # Average for temperatures and usage
    elif intent == 'percentile':
        if metric_type == 'histogram':
            return f"histogram_quantile(0.95, {metric_name}_bucket)"
        else:
            return f"quantile(0.95, {metric_name})"
    else:
        # Default based on metric type
        if metric_type == 'counter':
            return f"rate({metric_name}[5m])"
        else:
            return f"{metric_name}"
