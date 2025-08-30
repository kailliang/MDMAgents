#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Stage 6: Integration & Optimization
Tests the complete LangGraph-based MDMAgents system end-to-end.

This test suite validates:
1. Full system integration and workflow
2. Result synthesizer functionality
3. Output formatter compatibility
4. Error recovery and resilience
5. Performance monitoring
6. Production entry point functionality
"""

import pytest
import json
import asyncio
import time
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_result_synthesizer_basic():
    """Test ResultSynthesizer with basic processing results"""
    from langgraph_integration import ResultSynthesizer
    
    synthesizer = ResultSynthesizer()
    
    # Mock basic processing state
    mock_state = {
        "question": "What is hypertension?",
        "difficulty": "basic",
        "final_decision": {
            "analysis": "Hypertension is high blood pressure",
            "final_answer": "A) High blood pressure"
        },
        "token_usage": {"input": 100, "output": 50},
        "processing_stage": "basic_complete"
    }
    
    result = synthesizer.synthesize_result(mock_state, processing_time=2.5)
    
    # Verify result structure
    assert result.question == "What is hypertension?"
    assert result.difficulty == "basic"
    assert result.processing_mode == "basic"
    assert result.processing_time == 2.5
    assert result.success == True
    assert result.token_usage == {"input": 100, "output": 50}
    assert "analysis" in result.final_decision

def test_result_synthesizer_error_handling():
    """Test ResultSynthesizer error handling with malformed state"""
    from langgraph_integration import ResultSynthesizer
    from unittest.mock import patch
    
    synthesizer = ResultSynthesizer()
    
    # Mock state that will cause an exception in _extract_processing_mode
    malformed_state = {
        "question": "Test question",
        "processing_stage": None  # This will cause an error in string operations
    }
    
    # Patch _extract_processing_mode to raise an exception
    with patch.object(synthesizer, '_extract_processing_mode', side_effect=ValueError("Test error")):
        result = synthesizer.synthesize_result(malformed_state, processing_time=1.0)
    
    # Verify error handling
    assert result.success == False
    assert result.error_info is not None
    assert "error" in result.error_info
    assert result.processing_mode == "error"

def test_system_metrics_calculation():
    """Test system metrics calculation with multiple results"""
    from langgraph_integration import ResultSynthesizer
    
    synthesizer = ResultSynthesizer()
    
    # Add multiple results
    states = [
        {
            "question": "Q1", "difficulty": "basic", "final_decision": {"answer": "A"},
            "token_usage": {"input": 50, "output": 25}, "processing_stage": "basic_complete"
        },
        {
            "question": "Q2", "difficulty": "intermediate", "final_decision": {"answer": "B"},
            "token_usage": {"input": 75, "output": 40}, "processing_stage": "intermediate_complete"
        },
        {
            "question": "Q3", "difficulty": "advanced", "final_decision": {},
            "token_usage": {"input": 100, "output": 0}, "processing_stage": "error"
        }
    ]
    
    for state in states:
        synthesizer.synthesize_result(state, processing_time=1.5)
    
    metrics = synthesizer.get_system_metrics()
    
    # Verify metrics
    assert metrics.total_processing_time == 4.5  # 3 * 1.5
    assert metrics.total_token_usage["input"] == 225  # 50 + 75 + 100
    assert metrics.total_token_usage["output"] == 65   # 25 + 40 + 0
    assert metrics.success_rate == 2/3  # 2 successful out of 3
    assert metrics.error_count == 1
    assert "basic" in metrics.processing_modes_used
    assert "intermediate" in metrics.processing_modes_used

def test_output_formatter_evaluation_compatibility():
    """Test OutputFormatter produces evaluation-compatible results"""
    from langgraph_integration import OutputFormatter, ProcessingResult
    
    # Create test result
    result = ProcessingResult(
        question="What causes diabetes?",
        difficulty="basic",
        final_decision={"final_answer": "B) Insulin deficiency", "analysis": "Detailed analysis"},
        token_usage={"input": 80, "output": 40},
        processing_time=1.8,
        processing_mode="basic",
        success=True
    )
    
    options = ["A) Too much sugar", "B) Insulin deficiency", "C) Lack of exercise"]
    ground_truth = "B"
    
    formatted = OutputFormatter.format_for_evaluation(result, options, ground_truth)
    
    # Verify evaluation compatibility
    assert formatted["question"] == "What causes diabetes?"
    assert formatted["options"] == options
    assert formatted["ground_truth"] == ground_truth
    assert formatted["model_response"] == "B) Insulin deficiency"
    assert formatted["difficulty"] == "basic"
    assert formatted["processing_mode"] == "basic"
    assert formatted["token_usage"] == {"input": 80, "output": 40}
    assert formatted["success"] == True
    assert "timestamp" in formatted

def test_output_formatter_answer_extraction():
    """Test answer extraction from various decision formats"""
    from langgraph_integration import OutputFormatter
    
    # Test different decision formats
    test_cases = [
        ({"final_answer": "A) Test answer"}, "A) Test answer"),
        ({"answer": "B) Another answer"}, "B) Another answer"),
        ({"decision": "C) Decision format"}, "C) Decision format"),
        ({"majority_vote": "D) Majority vote"}, "D) Majority vote"),
        ({"final_answer": {"answer": "E) Nested format"}}, "E) Nested format"),
        ({}, "{}"),  # Empty fallback
    ]
    
    for decision, expected in test_cases:
        result = OutputFormatter._extract_answer(decision)
        assert result == expected

def test_batch_output_formatting():
    """Test batch result formatting with metadata"""
    from langgraph_integration import OutputFormatter, ProcessingResult
    
    # Create multiple results
    results = [
        ProcessingResult(
            question=f"Question {i}",
            difficulty="basic",
            final_decision={"final_answer": f"Answer {i}"},
            token_usage={"input": 50 + i, "output": 25 + i},
            processing_time=1.0 + i * 0.1,
            processing_mode="basic" if i % 2 == 0 else "intermediate",
            success=True
        )
        for i in range(3)
    ]
    
    batch_output = OutputFormatter.format_batch_results(results)
    
    # Verify batch structure
    assert "results" in batch_output
    assert "metadata" in batch_output
    assert len(batch_output["results"]) == 3
    
    metadata = batch_output["metadata"]
    assert metadata["total_questions"] == 3
    assert metadata["successful_questions"] == 3
    assert "basic" in metadata["processing_modes"]
    assert "intermediate" in metadata["processing_modes"]
    assert metadata["total_tokens"]["input"] > 0
    assert metadata["total_tokens"]["output"] > 0

def test_error_recovery_retry_logic():
    """Test error recovery retry logic with exponential backoff"""
    from langgraph_integration import ErrorRecoverySystem
    
    recovery = ErrorRecoverySystem(max_retries=2, base_delay=0.1)
    
    # Mock function that fails twice then succeeds
    call_count = 0
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError(f"Attempt {call_count} failed")
        return f"Success on attempt {call_count}"
    
    # Apply retry decorator
    @recovery.with_retry("test_operation")
    def wrapped_function():
        return failing_function()
    
    start_time = time.time()
    result = wrapped_function()
    elapsed = time.time() - start_time
    
    # Verify retry behavior
    assert result == "Success on attempt 3"
    assert call_count == 3
    assert elapsed > 0.1  # Should have delay from retries
    assert recovery.recovery_stats["recovered_errors"] == 1
    assert recovery.recovery_stats["total_errors"] >= 2

def test_error_recovery_fallback_result():
    """Test error recovery fallback result generation"""
    from langgraph_integration import ErrorRecoverySystem
    
    recovery = ErrorRecoverySystem()
    test_error = ValueError("Test processing failure")
    
    fallback = recovery.get_fallback_result("Test question", test_error)
    
    # Verify fallback structure
    assert fallback.question == "Test question"
    assert fallback.difficulty == "unknown"
    assert fallback.processing_mode == "fallback"
    assert fallback.success == False
    assert fallback.error_info is not None
    assert "fallback_used" in fallback.error_info
    assert fallback.error_info["fallback_used"] == True

def test_performance_monitor_operations():
    """Test performance monitoring of operations"""
    from langgraph_integration import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    # Test operation timing
    start_time = monitor.start_operation("test_operation")
    time.sleep(0.1)  # Simulate work
    monitor.end_operation("test_operation", start_time, {"input": 100, "output": 50})
    
    # Add another operation
    start_time2 = monitor.start_operation("test_operation")
    time.sleep(0.05)
    monitor.end_operation("test_operation", start_time2, {"input": 75, "output": 25})
    
    report = monitor.get_performance_report()
    
    # Verify performance tracking
    assert "test_operation" in report["operation_times"]
    operation_stats = report["operation_times"]["test_operation"]
    assert operation_stats["count"] == 2
    assert operation_stats["average"] > 0.05
    assert operation_stats["min"] > 0
    assert operation_stats["max"] > operation_stats["min"]
    
    # Verify token tracking
    assert "test_operation" in report["token_usage_by_mode"]
    token_stats = report["token_usage_by_mode"]["test_operation"]
    assert token_stats["input"] == 175  # 100 + 75
    assert token_stats["output"] == 75  # 50 + 25

def test_performance_monitor_health_checks():
    """Test performance monitor health check functionality"""
    from langgraph_integration import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    # Record health checks
    monitor.record_health_check("api_connection", True, "Connection successful")
    monitor.record_health_check("model_loading", True, "Model loaded")
    monitor.record_health_check("processing", False, "Processing failed")
    
    report = monitor.get_performance_report()
    
    # Verify health check tracking
    health_summary = report["health_summary"]
    assert health_summary["total_checks"] == 3
    assert health_summary["passed_checks"] == 2
    assert health_summary["failed_checks"] == 1
    
    # Verify recent checks
    assert len(report["health_checks"]) <= 10  # Max 10 recent checks

def test_integrated_system_initialization():
    """Test IntegratedMDMSystem initialization and basic functionality"""
    from langgraph_integration import IntegratedMDMSystem
    
    # Test system initialization
    system = IntegratedMDMSystem(model_info="gemini-2.5-flash")
    
    assert system.model_info == "gemini-2.5-flash"
    assert system.synthesizer is not None
    assert system.formatter is not None
    assert system.error_recovery is not None
    assert system.monitor is not None
    assert system.compiled_graph is not None

def test_integrated_system_question_processing():
    """Test integrated system question processing with mocking"""
    from langgraph_integration import IntegratedMDMSystem
    import asyncio
    
    system = IntegratedMDMSystem(model_info="gemini-2.5-flash")
    
    # Mock the compiled graph invoke method
    mock_result_state = {
        "question": "What is the treatment for pneumonia?",
        "difficulty": "basic",
        "final_decision": {
            "analysis": "Pneumonia requires antibiotic treatment",
            "final_answer": "A) Antibiotics"
        },
        "token_usage": {"input": 120, "output": 60},
        "processing_stage": "basic_complete"
    }
    
    with patch.object(system.compiled_graph, 'invoke', return_value=mock_result_state):
        result = asyncio.run(system.process_question(
            question="What is the treatment for pneumonia?",
            options=["A) Antibiotics", "B) Surgery", "C) Physical therapy"],
            ground_truth="A"
        ))
    
    # Verify result structure
    assert result["question"] == "What is the treatment for pneumonia?"
    assert result["model_response"] == "A) Antibiotics"
    assert result["difficulty"] == "basic"
    assert result["success"] == True
    assert result["token_usage"]["input"] == 120
    assert result["token_usage"]["output"] == 60

def test_integrated_system_status():
    """Test integrated system status reporting"""
    from langgraph_integration import IntegratedMDMSystem
    
    system = IntegratedMDMSystem(model_info="gpt-4.1-mini")
    
    status = system.get_system_status()
    
    # Verify status structure
    assert status["model_info"] == "gpt-4.1-mini"
    assert "system_metrics" in status
    assert "performance_report" in status
    assert "recovery_stats" in status
    assert status["graph_status"] == "compiled"

def test_dataset_loader():
    """Test dataset loading functionality"""
    from main import DatasetLoader
    import tempfile
    import json
    
    # Create temporary test dataset
    test_data = [
        {
            "question": "What is diabetes?",
            "options": {"A": "High blood sugar", "B": "Low blood sugar"},
            "answer_idx": "A"
        },
        {
            "question": "What causes hypertension?",
            "options": {"A": "Stress", "B": "Diet", "C": "Both"},
            "answer_idx": "C"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        temp_file = f.name
    
    try:
        # Test loading
        loaded_data = DatasetLoader.load_jsonl(temp_file)
        assert len(loaded_data) == 2
        assert loaded_data[0]["question"] == "What is diabetes?"
        assert loaded_data[1]["answer_idx"] == "C"
        
        # Test question formatting
        question_text, option_list, answer_key = DatasetLoader.create_question_text(test_data[0])
        
        assert "What is diabetes?" in question_text
        assert "Options:" in question_text
        assert len(option_list) == 2
        assert "A) High blood sugar" in option_list
        assert answer_key == "A"
        
    finally:
        os.unlink(temp_file)

def test_output_manager():
    """Test output file management"""
    from main import OutputManager
    import tempfile
    import shutil
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        output_manager = OutputManager(temp_dir)
        
        # Test filename generation
        filename = output_manager.generate_filename(
            "gemini-2.5-flash", "medqa", "adaptive", 100
        )
        
        assert "gemini_2_5_flash" in filename
        assert "medqa" in filename
        assert "adaptive" in filename
        assert "100samples" in filename
        assert filename.endswith(".json")
        
        # Test result saving
        test_results = [
            {
                "question": "Test question",
                "model_response": "Test answer",
                "difficulty": "basic",
                "token_usage": {"input": 50, "output": 25},
                "success": True
            }
        ]
        
        test_metadata = {"test": "metadata"}
        success = output_manager.save_results(test_results, filename, test_metadata)
        
        assert success == True
        
        # Verify file was created and has correct structure
        with open(filename, 'r') as f:
            saved_data = json.load(f)
        
        assert "metadata" in saved_data
        assert "results" in saved_data
        assert saved_data["metadata"]["test"] == "metadata"
        assert len(saved_data["results"]) == 1
        assert saved_data["results"][0]["question"] == "Test question"
        
    finally:
        shutil.rmtree(temp_dir)

def test_end_to_end_processing():
    """Test complete end-to-end processing workflow"""
    from langgraph_integration import IntegratedMDMSystem
    from main import DatasetLoader, OutputManager
    import tempfile
    import json
    import asyncio
    
    # Create test dataset
    test_dataset = [
        {
            "question": "What is the first-line treatment for hypertension?",
            "options": {
                "A": "ACE inhibitors",
                "B": "Beta blockers",
                "C": "Diuretics",
                "D": "Calcium channel blockers"
            },
            "answer_idx": "A"
        }
    ]
    
    # Setup system
    system = IntegratedMDMSystem(model_info="gemini-2.5-flash")
    
    # Mock the processing pipeline
    mock_result_state = {
        "question": "What is the first-line treatment for hypertension?\n\nOptions:\nA) ACE inhibitors\nB) Beta blockers\nC) Diuretics\nD) Calcium channel blockers",
        "difficulty": "basic",
        "final_decision": {
            "analysis": "ACE inhibitors are first-line treatment",
            "final_answer": "A) ACE inhibitors"
        },
        "token_usage": {"input": 150, "output": 75},
        "processing_stage": "basic_complete"
    }
    
    with patch.object(system.compiled_graph, 'invoke', return_value=mock_result_state):
        # Process question
        question_text, options, ground_truth = DatasetLoader.create_question_text(test_dataset[0])
        result = asyncio.run(system.process_question(question_text, options, ground_truth))
        
        # Verify result
        assert result["success"] == True
        assert result["model_response"] == "A) ACE inhibitors"
        assert result["difficulty"] == "basic"
        assert "hypertension" in result["question"].lower()
    
    # Test output saving
    temp_dir = tempfile.mkdtemp()
    try:
        output_manager = OutputManager(temp_dir)
        filename = output_manager.generate_filename("test_model", "test_dataset", "basic", 1)
        
        success = output_manager.save_results([result], filename)
        assert success == True
        
        # Verify file structure
        with open(filename, 'r') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 1
        assert saved_data[0]["success"] == True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)

def test_error_resilience_integration():
    """Test system resilience to various error conditions"""
    from langgraph_integration import IntegratedMDMSystem
    
    system = IntegratedMDMSystem(model_info="gemini-2.5-flash")
    
    # Test with empty question
    with patch.object(system.compiled_graph, 'invoke', side_effect=ValueError("Processing failed")):
        # This should not raise an exception but return a fallback result
        result = asyncio.run(system.process_question(""))
        
        assert result["success"] == False
        assert "error_type" in result.get("error_info", {})

def test_token_usage_accumulation():
    """Test token usage accumulation across system components"""
    from langgraph_integration import ResultSynthesizer
    
    synthesizer = ResultSynthesizer()
    
    # Simulate processing multiple questions with different token usages
    test_states = [
        {
            "question": "Q1", "difficulty": "basic", "final_decision": {"answer": "A"},
            "token_usage": {"input": 100, "output": 50}, "processing_stage": "basic_complete"
        },
        {
            "question": "Q2", "difficulty": "intermediate", "final_decision": {"answer": "B"},
            "token_usage": {"input": 200, "output": 100}, "processing_stage": "intermediate_complete"
        },
        {
            "question": "Q3", "difficulty": "advanced", "final_decision": {"answer": "C"},
            "token_usage": {"input": 300, "output": 150}, "processing_stage": "advanced_complete"
        }
    ]
    
    for state in test_states:
        synthesizer.synthesize_result(state, processing_time=2.0)
    
    metrics = synthesizer.get_system_metrics()
    
    # Verify accumulation
    assert metrics.total_token_usage["input"] == 600  # 100 + 200 + 300
    assert metrics.total_token_usage["output"] == 300  # 50 + 100 + 150
    assert len(metrics.processing_modes_used) == 3
    assert metrics.processing_modes_used["basic"] == 1
    assert metrics.processing_modes_used["intermediate"] == 1
    assert metrics.processing_modes_used["advanced"] == 1

def test_multiple_model_compatibility():
    """Test system works with different model configurations"""
    from langgraph_integration import IntegratedMDMSystem
    
    models_to_test = ["gemini-2.5-flash", "gpt-4.1-mini"]
    
    for model in models_to_test:
        system = IntegratedMDMSystem(model_info=model)
        
        # Verify system initializes correctly
        assert system.model_info == model
        assert system.compiled_graph is not None
        
        # Verify status reporting works
        status = system.get_system_status()
        assert status["model_info"] == model
        assert status["graph_status"] == "compiled"

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])