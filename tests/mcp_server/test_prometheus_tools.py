"""Simple, working tests for Prometheus tools."""

import pytest
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestPrometheusToolsBasic:
    """Basic tests for Prometheus tools that actually work."""
    
    def test_all_tools_importable(self):
        """Test that all Prometheus tools can be imported."""
        from mcp_server.tools.prometheus_tools import (
            search_metrics,
            get_metric_metadata,
            get_label_values,
            execute_promql,
            explain_results,
            suggest_queries,
            select_best_metric,
            find_best_metric_with_metadata_v2,
            find_best_metric_with_metadata,
        )
        
        # All tools should be callable
        tools = [
            search_metrics, get_metric_metadata, get_label_values,
            execute_promql, explain_results, suggest_queries,
            select_best_metric, find_best_metric_with_metadata_v2,
            find_best_metric_with_metadata
        ]
        
        for tool in tools:
            assert callable(tool), f"{tool.__name__} should be callable"
    
    def test_keyword_extraction_basic(self):
        """Test basic keyword extraction."""
        from mcp_server.tools.prometheus_tools import _extract_keywords_for_filtering
        
        # Test basic keyword extraction
        keywords = _extract_keywords_for_filtering("How many pods are running?")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "pod" in keywords
    
    def test_metric_scoring_basic(self):
        """Test basic metric scoring."""
        from mcp_server.tools.prometheus_tools import _score_metric_with_metadata_for_question
        
        score = _score_metric_with_metadata_for_question(
            "kube_pod_status_phase",
            "The phase of a pod",
            "gauge",
            "How many pods are running?"
        )
        assert isinstance(score, int)
        assert score >= 0
    
    def test_semantic_scoring_basic(self):
        """Test basic semantic scoring."""
        from mcp_server.tools.prometheus_tools import _calculate_semantic_score
        
        score = _calculate_semantic_score("pod", "kube_pod_status_phase")
        assert isinstance(score, int)
        assert score >= 0
    
    def test_explain_results_with_correct_signature(self):
        """Test explain_results with correct parameters."""
        from mcp_server.tools.prometheus_tools import explain_results
        
        # Use correct signature: (query, results, result_type)
        test_results = [{"metric": {"phase": "Running"}, "value": [1694123456, "7"]}]
        
        result = explain_results(
            query="sum(kube_pod_status_phase{phase='Running'})",
            results=test_results,
            result_type="vector"
        )
        
        assert len(result) > 0
        assert isinstance(result[0]["text"], str)
    
    def test_suggest_queries_basic(self):
        """Test suggest_queries basic functionality."""
        from mcp_server.tools.prometheus_tools import suggest_queries
        
        result = suggest_queries("pod analysis")
        
        assert len(result) > 0
        result_data = json.loads(result[0]["text"])
        assert "suggestions" in result_data
        assert isinstance(result_data["suggestions"], list)


class TestPrometheusToolsErrorHandling:
    """Test error handling in Prometheus tools."""
    
    def test_tools_handle_empty_input(self):
        """Test that tools handle empty input gracefully."""
        from mcp_server.tools.prometheus_tools import (
            _extract_keywords_for_filtering,
            _score_metric_with_metadata_for_question
        )
        
        # Empty question should return empty or default keywords
        keywords = _extract_keywords_for_filtering("")
        assert isinstance(keywords, list)
        
        # Empty parameters should not crash
        score = _score_metric_with_metadata_for_question("", "", "gauge", "")
        assert isinstance(score, int)
        assert score >= 0
    
    def test_tools_handle_invalid_input(self):
        """Test that tools handle invalid input gracefully."""
        from mcp_server.tools.prometheus_tools import suggest_queries
        
        # Should not crash with None or invalid input
        result = suggest_queries(None)
        assert len(result) > 0
        
        result = suggest_queries("")
        assert len(result) > 0


class TestPrometheusToolsIntegration:
    """Integration tests that don't require mocking."""
    
    def test_keyword_extraction_comprehensive(self):
        """Test keyword extraction for various question types."""
        from mcp_server.tools.prometheus_tools import _extract_keywords_for_filtering
        
        test_cases = [
            ("How many pods are running?", "pod"),
            ("What's the GPU temperature?", ["gpu", "temp", "temperature"]),
            ("Show service status", "service"),
            ("Check node health", "node"),
            ("vLLM token generation", "vllm"),
            ("Memory usage trends", "memory"),
            ("CPU utilization", "cpu"),
            ("Alert status", "alert")
        ]
        
        for question, expected in test_cases:
            keywords = _extract_keywords_for_filtering(question)
            
            if isinstance(expected, str):
                assert expected in keywords, f"Expected '{expected}' in keywords for '{question}'"
            else:
                # List of possible keywords - at least one should match
                assert any(exp in keywords for exp in expected), \
                    f"Expected at least one of {expected} in keywords for '{question}'"
    
    def test_metric_scoring_patterns(self):
        """Test metric scoring patterns make sense."""
        from mcp_server.tools.prometheus_tools import _score_metric_with_metadata_for_question
        
        # Pod question should score higher for pod metrics
        pod_score = _score_metric_with_metadata_for_question(
            "kube_pod_status_phase",
            "The phase of a pod",
            "gauge",
            "How many pods are running?"
        )
        
        service_score = _score_metric_with_metadata_for_question(
            "kube_service_info",
            "Information about services",
            "gauge", 
            "How many pods are running?"
        )
        
        # Pod metric should score higher for pod question
        assert pod_score >= service_score, "Pod metric should score higher for pod question"
    
    def test_ranking_returns_valid_list(self):
        """Test that ranking returns a valid list."""
        from mcp_server.tools.prometheus_tools import _rank_metrics_by_relevance
        
        test_metrics = ["kube_pod_status_phase", "kube_service_info", "random_metric"]
        ranked = _rank_metrics_by_relevance("pod", test_metrics)
        
        assert isinstance(ranked, list)
        assert len(ranked) <= len(test_metrics)  # May filter out irrelevant metrics
        assert all(metric in test_metrics for metric in ranked), "All returned metrics should be from input"




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
