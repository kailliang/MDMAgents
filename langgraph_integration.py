#!/usr/bin/env python3
"""
LangGraph Integration and Optimization Module
Stage 6: Integration & Optimization implementation.

This module provides:
1. Result Synthesizer - Unified output format across all processing modes
2. Output Formatter - JSON standardization for evaluation compatibility  
3. Error Recovery System - Graceful degradation and retry logic
4. Performance Monitor - Token usage and system health tracking
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Standardized result structure from any processing mode"""
    question: str
    difficulty: str
    final_decision: Dict[str, Any]
    token_usage: Dict[str, int]
    processing_time: float
    processing_mode: str
    success: bool
    error_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "question": self.question,
            "difficulty": self.difficulty,
            "final_decision": self.final_decision,
            "token_usage": self.token_usage,
            "processing_time": self.processing_time,
            "processing_mode": self.processing_mode,
            "success": self.success,
            "error_info": self.error_info
        }


@dataclass 
class SystemMetrics:
    """System performance and health metrics"""
    total_processing_time: float
    total_token_usage: Dict[str, int]
    success_rate: float
    error_count: int
    processing_modes_used: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_processing_time": self.total_processing_time,
            "total_token_usage": self.total_token_usage,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "processing_modes_used": self.processing_modes_used
        }


class ResultSynthesizer:
    """
    Synthesizes results from different processing subgraphs into unified format.
    Ensures compatibility with existing evaluation infrastructure.
    """
    
    def __init__(self):
        self.processed_results: List[ProcessingResult] = []
        self.start_time = time.time()
    
    def synthesize_result(self, state: Dict[str, Any], processing_time: float) -> ProcessingResult:
        """Convert LangGraph state to standardized result format"""
        try:
            # Extract core information
            question = state.get("question", "")
            difficulty = state.get("difficulty", "unknown")
            final_decision = state.get("final_decision", {})
            token_usage = state.get("token_usage", {"input": 0, "output": 0})
            processing_stage = state.get("processing_stage", "unknown")
            
            # Determine processing mode from stage
            processing_mode = self._extract_processing_mode(processing_stage)
            
            # Create standardized result
            result = ProcessingResult(
                question=question,
                difficulty=difficulty,
                final_decision=final_decision,
                token_usage=token_usage,
                processing_time=processing_time,
                processing_mode=processing_mode,
                success=bool(final_decision),
                error_info=None
            )
            
            self.processed_results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error synthesizing result: {e}")
            # Return error result
            error_result = ProcessingResult(
                question=state.get("question", ""),
                difficulty="unknown",
                final_decision={},
                token_usage={"input": 0, "output": 0},
                processing_time=processing_time,
                processing_mode="error",
                success=False,
                error_info={"error": str(e), "type": type(e).__name__}
            )
            self.processed_results.append(error_result)
            return error_result
    
    def _extract_processing_mode(self, processing_stage: str) -> str:
        """Extract processing mode from stage information"""
        if "basic" in processing_stage.lower():
            return "basic"
        elif "intermediate" in processing_stage.lower():
            return "intermediate"
        elif "advanced" in processing_stage.lower():
            return "advanced"
        else:
            return "unknown"
    
    def get_system_metrics(self) -> SystemMetrics:
        """Calculate overall system performance metrics"""
        if not self.processed_results:
            return SystemMetrics(0.0, {"input": 0, "output": 0}, 0.0, 0, {})
        
        # Calculate totals
        total_time = sum(r.processing_time for r in self.processed_results)
        total_tokens = {
            "input": sum(r.token_usage.get("input", 0) for r in self.processed_results),
            "output": sum(r.token_usage.get("output", 0) for r in self.processed_results)
        }
        
        # Calculate success rate
        successful = sum(1 for r in self.processed_results if r.success)
        success_rate = successful / len(self.processed_results) if self.processed_results else 0.0
        
        # Count errors
        error_count = sum(1 for r in self.processed_results if not r.success)
        
        # Count processing modes used
        mode_counts = {}
        for result in self.processed_results:
            mode_counts[result.processing_mode] = mode_counts.get(result.processing_mode, 0) + 1
        
        return SystemMetrics(
            total_processing_time=total_time,
            total_token_usage=total_tokens,
            success_rate=success_rate,
            error_count=error_count,
            processing_modes_used=mode_counts
        )


class OutputFormatter:
    """
    Formats results for compatibility with existing evaluation scripts.
    Maintains the JSON structure expected by evaluate_text_output.py.
    """
    
    @staticmethod
    def format_for_evaluation(result: ProcessingResult, options: List[str] = None, 
                            ground_truth: str = None) -> Dict[str, Any]:
        """Format result for evaluation script compatibility"""
        
        # Extract final answer from decision
        final_answer = OutputFormatter._extract_answer(result.final_decision)
        
        # Create evaluation-compatible format
        formatted_result = {
            "question": result.question,
            "options": options or [],
            "ground_truth": ground_truth or "",
            "model_response": final_answer,
            "full_response": result.final_decision,
            "difficulty": result.difficulty,
            "processing_mode": result.processing_mode,
            "token_usage": result.token_usage,
            "processing_time": result.processing_time,
            "success": result.success,
            "timestamp": time.time()
        }
        
        # Add error information if present
        if result.error_info:
            formatted_result["error_info"] = result.error_info
        
        return formatted_result
    
    @staticmethod
    def _extract_answer(final_decision: Dict[str, Any]) -> str:
        """Extract final answer from decision structure"""
        if not final_decision:
            return "{}"
        
        # Try different answer extraction patterns
        answer_keys = ["final_answer", "answer", "decision", "conclusion", "majority_vote"]
        
        for key in answer_keys:
            if key in final_decision:
                answer = final_decision[key]
                if isinstance(answer, str):
                    return answer
                elif isinstance(answer, dict) and "answer" in answer:
                    return str(answer["answer"])
        
        # Fallback to string representation
        return str(final_decision)
    
    @staticmethod
    def format_batch_results(results: List[ProcessingResult], 
                           dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format batch of results for output file"""
        
        formatted_results = []
        for result in results:
            formatted_result = OutputFormatter.format_for_evaluation(result)
            formatted_results.append(formatted_result)
        
        # Create batch output structure
        batch_output = {
            "results": formatted_results,
            "metadata": {
                "total_questions": len(results),
                "successful_questions": sum(1 for r in results if r.success),
                "processing_modes": {},
                "total_tokens": {"input": 0, "output": 0},
                "total_processing_time": sum(r.processing_time for r in results),
                "timestamp": time.time()
            }
        }
        
        # Calculate processing mode distribution
        for result in results:
            mode = result.processing_mode
            batch_output["metadata"]["processing_modes"][mode] = \
                batch_output["metadata"]["processing_modes"].get(mode, 0) + 1
        
        # Calculate total token usage
        for result in results:
            batch_output["metadata"]["total_tokens"]["input"] += result.token_usage.get("input", 0)
            batch_output["metadata"]["total_tokens"]["output"] += result.token_usage.get("output", 0)
        
        # Add dataset information if provided
        if dataset_info:
            batch_output["metadata"]["dataset_info"] = dataset_info
        
        return batch_output


class ErrorRecoverySystem:
    """
    Handles errors, API failures, and provides graceful degradation.
    Implements retry logic with exponential backoff.
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_counts = {}
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0
        }
    
    def with_retry(self, operation_name: str = "operation"):
        """Decorator for adding retry logic to operations"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        
                        if attempt > 0:
                            logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                            self.recovery_stats["recovered_errors"] += 1
                        
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
                        self.recovery_stats["total_errors"] += 1
                        
                        if attempt < self.max_retries:
                            delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"{operation_name} failed on attempt {attempt + 1}: {e}. "
                                         f"Retrying in {delay} seconds...")
                            time.sleep(delay)
                        else:
                            logger.error(f"{operation_name} failed after {self.max_retries + 1} attempts: {e}")
                            self.recovery_stats["failed_recoveries"] += 1
                
                # All retries failed
                raise last_exception
            
            return wrapper
        return decorator
    
    def get_fallback_result(self, question: str, error: Exception) -> ProcessingResult:
        """Generate fallback result when all processing fails"""
        logger.warning(f"Generating fallback result for question due to error: {error}")
        
        return ProcessingResult(
            question=question,
            difficulty="unknown",
            final_decision={
                "error": "Processing failed",
                "fallback_answer": "Unable to process question due to system error",
                "error_details": str(error)
            },
            token_usage={"input": 0, "output": 0},
            processing_time=0.0,
            processing_mode="fallback",
            success=False,
            error_info={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "fallback_used": True
            }
        )
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get error recovery statistics"""
        return {
            "error_counts_by_operation": self.error_counts.copy(),
            "recovery_stats": self.recovery_stats.copy(),
            "success_rate": (self.recovery_stats["recovered_errors"] / 
                           max(1, self.recovery_stats["total_errors"]))
        }


class PerformanceMonitor:
    """
    Monitors system performance, token usage, and health metrics.
    Provides insights for optimization and debugging.
    """
    
    def __init__(self):
        self.session_start = time.time()
        self.operation_times = {}
        self.token_usage_by_mode = {}
        self.health_checks = []
    
    def start_operation(self, operation_name: str) -> float:
        """Start timing an operation"""
        start_time = time.time()
        return start_time
    
    def end_operation(self, operation_name: str, start_time: float, 
                     token_usage: Dict[str, int] = None):
        """End timing an operation and record metrics"""
        end_time = time.time()
        duration = end_time - start_time
        
        # Record timing
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        self.operation_times[operation_name].append(duration)
        
        # Record token usage by operation
        if token_usage:
            if operation_name not in self.token_usage_by_mode:
                self.token_usage_by_mode[operation_name] = {"input": 0, "output": 0}
            self.token_usage_by_mode[operation_name]["input"] += token_usage.get("input", 0)
            self.token_usage_by_mode[operation_name]["output"] += token_usage.get("output", 0)
    
    def record_health_check(self, check_name: str, status: bool, details: str = ""):
        """Record a health check result"""
        self.health_checks.append({
            "check_name": check_name,
            "status": status,
            "details": details,
            "timestamp": time.time()
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        total_session_time = time.time() - self.session_start
        
        # Calculate average operation times
        avg_operation_times = {}
        for op_name, times in self.operation_times.items():
            avg_operation_times[op_name] = {
                "average": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(times)
            }
        
        # Calculate total token usage
        total_tokens = {"input": 0, "output": 0}
        for usage in self.token_usage_by_mode.values():
            total_tokens["input"] += usage["input"]
            total_tokens["output"] += usage["output"]
        
        # Health check summary
        health_summary = {
            "total_checks": len(self.health_checks),
            "passed_checks": sum(1 for check in self.health_checks if check["status"]),
            "failed_checks": sum(1 for check in self.health_checks if not check["status"])
        }
        
        return {
            "session_duration": total_session_time,
            "operation_times": avg_operation_times,
            "health_summary": health_summary,
            "timestamp": time.time()
        }


class IntegratedMDMSystem:
    """
    Main integration class that orchestrates all Stage 6 components.
    Provides unified interface for production use.
    """
    
    def __init__(self, model_info: str = "gemini-2.5-flash"):
        self.model_info = model_info
        self.synthesizer = ResultSynthesizer()
        self.formatter = OutputFormatter()
        self.error_recovery = ErrorRecoverySystem()
        self.monitor = PerformanceMonitor()
        
        # Import main graph creation function
        from langgraph_mdm import create_mdm_graph
        
        # Create and compile the main graph
        self.graph = create_mdm_graph(model_info=model_info)
        self.compiled_graph = self.graph.compile()
        
        logger.info(f"IntegratedMDMSystem initialized with model: {model_info}")
    
    async def process_question(self, question: str, options: List[str] = None,
                             ground_truth: str = None, forced_difficulty: str = None) -> Dict[str, Any]:
        """Process a single question through the integrated system"""
        
        logger.debug(f"=== PROCESSING QUESTION ===")
        logger.debug(f"Question: {question[:100]}...")
        logger.debug(f"Model: {self.model_info}")
        
        operation_start = self.monitor.start_operation("process_question")
        
        try:
            # Prepare initial state
            initial_state = {
                "messages": [],
                "question": question,
                "answer_options": options,
                "agents": [],
                "token_usage": {"input": 0, "output": 0},
                "processing_stage": "start",
                "final_decision": None
            }
            
            # Handle forced difficulty routing
            if forced_difficulty and forced_difficulty != "adaptive":
                logger.info(f"Forcing {forced_difficulty} processing mode (skipping difficulty assessment)")
                initial_state["difficulty"] = forced_difficulty
                initial_state["confidence"] = 1.0  # High confidence since user specified
                # We'll handle the routing by directly invoking the specific subgraph
            
            logger.debug("Starting LangGraph processing pipeline...")
            
            # Process through graph with error recovery
            @self.error_recovery.with_retry("graph_processing")
            def process_with_graph():
                if forced_difficulty and forced_difficulty != "adaptive":
                    # Direct processing without full graph
                    logger.debug(f"Invoking {forced_difficulty} subgraph directly...")
                    
                    # Import specific subgraph
                    if forced_difficulty == "basic":
                        from langgraph_basic import create_basic_processing_subgraph
                        subgraph = create_basic_processing_subgraph(self.model_info).compile()
                    elif forced_difficulty == "intermediate":
                        from langgraph_intermediate import create_intermediate_processing_subgraph
                        subgraph = create_intermediate_processing_subgraph(self.model_info).compile()
                    elif forced_difficulty == "advanced":
                        from langgraph_advanced import create_advanced_processing_subgraph
                        subgraph = create_advanced_processing_subgraph(self.model_info).compile()
                    else:
                        raise ValueError(f"Unknown forced difficulty: {forced_difficulty}")
                    
                    # Execute the specific subgraph
                    result = subgraph.invoke(initial_state)
                    result["processing_stage"] = f"{forced_difficulty}_complete"
                    logger.debug(f"Forced {forced_difficulty} processing completed")
                    return result
                else:
                    # Normal adaptive processing through full graph
                    logger.debug("Invoking compiled LangGraph...")
                    result = self.compiled_graph.invoke(initial_state)
                    logger.debug(f"LangGraph processing completed. Final stage: {result.get('processing_stage', 'unknown')}")
                    return result
            
            # Execute processing
            processing_start = time.time()
            result_state = process_with_graph()
            processing_time = time.time() - processing_start
            
            logger.debug(f"Total processing time: {processing_time:.3f} seconds")
            logger.debug(f"Final difficulty: {result_state.get('difficulty', 'unknown')}")
            logger.debug(f"Token usage: {result_state.get('token_usage', {})}")
            
            # Synthesize result
            synthesized_result = self.synthesizer.synthesize_result(result_state, processing_time)
            
            # Format for evaluation
            formatted_result = self.formatter.format_for_evaluation(
                synthesized_result, options, ground_truth
            )
            
            # Record successful processing
            self.monitor.end_operation("process_question", operation_start, 
                                     synthesized_result.token_usage)
            self.monitor.record_health_check("question_processing", True, 
                                           f"Successfully processed: {synthesized_result.processing_mode}")
            
            return formatted_result
            
        except Exception as e:
            # Handle complete failure with fallback
            logger.error(f"Complete processing failure for question: {e}")
            
            fallback_result = self.error_recovery.get_fallback_result(question, e)
            formatted_result = self.formatter.format_for_evaluation(
                fallback_result, options, ground_truth
            )
            
            # Record failure
            self.monitor.end_operation("process_question", operation_start, {"input": 0, "output": 0})
            self.monitor.record_health_check("question_processing", False, f"Failed: {str(e)}")
            
            return formatted_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and metrics"""
        return {
            "model_info": self.model_info,
            "system_metrics": self.synthesizer.get_system_metrics(),
            "performance_report": self.monitor.get_performance_report(),
            "recovery_stats": self.error_recovery.get_recovery_stats(),
            "graph_status": "compiled" if self.compiled_graph else "not_compiled"
        }


# Export main classes for use in other modules
__all__ = [
    "ProcessingResult",
    "SystemMetrics", 
    "ResultSynthesizer",
    "OutputFormatter",
    "ErrorRecoverySystem",
    "PerformanceMonitor",
    "IntegratedMDMSystem"
]


if __name__ == "__main__":
    # Basic smoke test
    import asyncio
    
    async def test_integration():
        system = IntegratedMDMSystem("gemini-2.5-flash")
        
        result = await system.process_question(
            "What is the first-line treatment for hypertension?",
            ["A) ACE inhibitors", "B) Beta blockers", "C) Diuretics", "D) Calcium channel blockers"]
        )
        
        print("Integration test result:")
        print(json.dumps(result, indent=2))
        
        status = system.get_system_status()
        print("\nSystem status:")
        print(json.dumps(status, indent=2, default=str))
    
    # Run test if this file is executed directly
    asyncio.run(test_integration())