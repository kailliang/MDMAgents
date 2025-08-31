#!/usr/bin/env python3
"""
LangGraph-based MDMAgents Production Entry Point
Stage 6: Production-ready entry point for the LangGraph implementation.

This module provides a complete replacement for main.py functionality using
the LangGraph-based system while maintaining full compatibility with existing
command-line arguments, output formats, and evaluation infrastructure.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the integrated system
from langgraph_integration import IntegratedMDMSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads datasets in the same format as the original main.py.
    Maintains compatibility with existing data structures.
    """
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Load JSONL file with medical questions"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {e}")
            return []
    
    @staticmethod
    def create_question_text(question_data: Dict[str, Any]) -> tuple:
        """Create formatted question text and extract metadata"""
        question_text = question_data.get("question", "")
        options = question_data.get("options", {})
        answer_key = question_data.get("answer_idx", "")
        
        # Format options as list
        option_list = []
        if isinstance(options, dict):
            for key in sorted(options.keys()):
                option_list.append(f"{key}) {options[key]}")
        elif isinstance(options, list):
            option_list = options
        
        # Return question text without options (options passed separately)
        # This prevents duplication when options are added in prompts
        return question_text, option_list, answer_key


class OutputManager:
    """
    Manages output file generation compatible with existing evaluation scripts.
    Maintains the JSON format expected by evaluate_text_output.py.
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_filename(self, model_info: str, dataset: str, difficulty: str, 
                         num_samples: int) -> str:
        """Generate output filename matching original main.py pattern"""
        # Clean model name for filename
        model_clean = model_info.replace("-", "_").replace(".", "_")[11:]
        
        if difficulty == "adaptive":
            filename = f"{model_clean}_{dataset[:3]}_{difficulty[:3]}_{num_samples}_smpl.json"
        else:
            filename = f"{model_clean}_{dataset[:3]}_{difficulty[:3]}_{num_samples}_smpl.json"
        
        return str(self.output_dir / filename)
    
    def save_results(self, results: List[Dict[str, Any]], output_filename: str, 
                    metadata: Dict[str, Any] = None):
        """Save results in format compatible with evaluation scripts"""
        try:
            # Prepare output structure
            output_data = []
            
            for result in results:
                # Extract core fields for compatibility
                output_entry = {
                    "question": result.get("question", ""),
                    "options": result.get("options", []),
                    "ground_truth": result.get("ground_truth", ""),
                    "model_response": result.get("model_response", ""),
                    "full_response": result.get("full_response", {}),
                    "difficulty": result.get("difficulty", "unknown"),
                    "token_usage": result.get("token_usage", {"input": 0, "output": 0}),
                    "processing_time": result.get("processing_time", 0.0),
                    "success": result.get("success", False)
                }
                
                # Add processing metadata
                if "processing_mode" in result:
                    output_entry["processing_mode"] = result["processing_mode"]
                if "error_info" in result:
                    output_entry["error_info"] = result["error_info"]
                
                output_data.append(output_entry)
            
            # Add metadata if provided
            if metadata:
                final_output = {
                    "metadata": metadata,
                    "results": output_data
                }
            else:
                final_output = output_data
            
            # Save to file
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results to {output_filename}: {e}")
            return False


def extract_answer_letter(text: str) -> str:
    """
    Extract final answer letter (A, B, C, D, E) from model response.
    Uses similar logic to evaluate_text_output.py for consistency.
    """
    if not text:
        return 'X'  # Parse error marker
    
    # Convert to string if needed
    text = str(text)
    
    # Pattern 1: "Answer: C" or "Final answer: C"
    answer_match = re.search(r'\b(?:answer|final[_\s]answer|decision|conclusion):\s*([A-E])\b', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Pattern 2: "A) Some option text" or "B) Option text"
    option_match = re.search(r'\b([A-E])\)\s*[A-Z]', text)
    if option_match:
        return option_match.group(1).upper()
    
    # Pattern 3: "(A) Some text" format
    paren_match = re.search(r'\(([A-E])\)', text)
    if paren_match:
        return paren_match.group(1).upper()
    
    # Pattern 4: Just the letter "A", "B", etc. (with word boundaries)
    letter_match = re.search(r'\b([A-E])\b', text)
    if letter_match:
        return letter_match.group(1).upper()
    
    return 'X'  # Could not parse answer


def check_answer_correctness(model_response: str, ground_truth: str) -> bool:
    """Check if model response matches ground truth answer"""
    extracted_answer = extract_answer_letter(model_response)
    ground_truth_letter = ground_truth.upper() if ground_truth else ''
    
    # Handle parse error case
    if extracted_answer == 'X':
        return False
        
    return extracted_answer == ground_truth_letter


async def process_dataset_async(system: IntegratedMDMSystem, dataset: List[Dict[str, Any]], 
                               num_samples: Optional[int] = None, difficulty_filter: str = "adaptive", verbose: bool = False) -> List[Dict[str, Any]]:
    """Process dataset questions asynchronously through the integrated system"""
    
    if num_samples:
        dataset = dataset[:num_samples]
    
    logger.info(f"Processing {len(dataset)} questions with difficulty filter: {difficulty_filter}")
    
    results = []
    total_questions = len(dataset)
    
    # Initialize accuracy tracking
    correct_count = 0
    total_processed = 0
    
    for i, question_data in enumerate(dataset):
        # Clear separator for each sample
        logger.info("=" * 60)
        logger.info(f"PROCESSING QUESTION {i+1}/{total_questions}")
        logger.info("=" * 60)
        
        try:
            # Create question and extract metadata
            question_text, options, ground_truth = DatasetLoader.create_question_text(question_data)
            
            # Process through integrated system
            # Pass forced difficulty if not adaptive
            forced_diff = None if difficulty_filter == "adaptive" else difficulty_filter
            result = await system.process_question(
                question=question_text,
                options=options,
                ground_truth=ground_truth,
                forced_difficulty=forced_diff
            )
            
            results.append(result)
            
            # Log detailed per-question information
            token_info = result.get("token_usage", {})
            input_tokens = token_info.get("input", 0)
            output_tokens = token_info.get("output", 0)
            total_tokens = input_tokens + output_tokens
            
            # Log token usage
            logger.info(f"Question {i+1}/{total_questions} | Tokens: {input_tokens} in, {output_tokens} out | Total: {total_tokens}")
            
            # Check answer correctness and update tracking
            model_answer = result.get("model_response", "")
            is_correct = check_answer_correctness(model_answer, ground_truth)
            
            if is_correct:
                correct_count += 1
            total_processed += 1
            
            # Calculate and log running accuracy
            current_accuracy = (correct_count / total_processed) * 100
            
            # Log answer correctness with truncated response for readability
            answer_preview = model_answer[:50] + "..." if len(model_answer) > 50 else model_answer
            status_icon = "✓ CORRECT" if is_correct else "✗ WRONG"
            
            logger.info(f"Question {i+1}/{total_questions} | Answer: {answer_preview} | Expected: {ground_truth} | {status_icon}")
            logger.info(f"Question {i+1}/{total_questions} | Running Accuracy: {current_accuracy:.1f}% ({correct_count}/{total_processed})")
            
            # Remove redundant progress logging - handled above
                
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
            # Create error result
            error_result = {
                "question": question_data.get("question", ""),
                "options": question_data.get("options", []),
                "ground_truth": question_data.get("answer_idx", ""),
                "model_response": "ERROR",
                "full_response": {"error": str(e)},
                "difficulty": "unknown",
                "token_usage": {"input": 0, "output": 0},
                "processing_time": 0.0,
                "success": False,
                "error_info": {"error": str(e), "question_index": i}
            }
            results.append(error_result)
    
    return results


def validate_environment():
    """Validate required environment variables and API keys"""
    required_env_vars = {
        "genai_api_key": "Gemini API key",
        "openai_api_key": "OpenAI API key"
    }
    
    missing_vars = []
    for var_name, description in required_env_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"{var_name} ({description})")
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some models may not be available without proper API keys")
        return False
    
    return True


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command-line argument parser compatible with original main.py"""
    parser = argparse.ArgumentParser(
        description="LangGraph-based MDMAgents: Multi-agent medical decision making system"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="medqa",
        help="Dataset name (default: medqa)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemini-2.5-flash",
        choices=[
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-lite-preview-06-17",
            "gpt-4o-mini", 
            "gpt-4.1-mini"
        ],
        help="Model to use for processing (default: gemini-2.5-flash)"
    )
    
    parser.add_argument(
        "--difficulty", 
        type=str, 
        default="adaptive",
        choices=["basic", "intermediate", "advanced", "adaptive"],
        help="Processing difficulty mode (default: adaptive)"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        help="Number of samples to process (default: all)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Data directory (default: data)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Minimal output (errors only)"
    )
    
    parser.add_argument(
        "--test_mode", 
        action="store_true",
        help="Run in test mode with minimal samples"
    )
    
    return parser


async def main():
    """Main entry point for LangGraph-based MDMAgents system"""
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    logger.info("Starting LangGraph-based MDMAgents system")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate environment
    validate_environment()
    
    # Test mode override
    if args.test_mode:
        args.num_samples = min(args.num_samples or 5, 5)
        logger.info(f"Test mode enabled, processing {args.num_samples} samples")
    
    try:
        # Load dataset
        data_path = Path(args.data_dir) / args.dataset / "test.jsonl"
        if not data_path.exists():
            logger.error(f"Dataset file not found: {data_path}")
            sys.exit(1)
        
        logger.info(f"Loading dataset from: {data_path}")
        dataset = DatasetLoader.load_jsonl(str(data_path))
        
        if not dataset:
            logger.error("No data loaded from dataset file")
            sys.exit(1)
        
        logger.info(f"Loaded {len(dataset)} questions from dataset")
        
        # Initialize integrated system
        logger.info(f"Initializing system with model: {args.model}")
        system = IntegratedMDMSystem(model_info=args.model)
        
        # Process dataset
        start_time = time.time()
        results = await process_dataset_async(
            system=system,
            dataset=dataset,
            num_samples=args.num_samples,
            difficulty_filter=args.difficulty,
            verbose=args.verbose
        )
        processing_time = time.time() - start_time
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Generate output filename
        output_manager = OutputManager(args.output_dir)
        num_processed = len(results)
        output_filename = output_manager.generate_filename(
            args.model, args.dataset, args.difficulty, num_processed
        )
        
        # Prepare metadata
        system_status = system.get_system_status()
        system_metrics = system_status["system_metrics"]
        
        metadata = {
            "model": args.model,
            "dataset": args.dataset,
            "difficulty_mode": args.difficulty,
            "total_questions": num_processed,
            "system_metrics": system_metrics.to_dict() if hasattr(system_metrics, 'to_dict') else system_metrics,
            "performance_report": system_status["performance_report"],
            "timestamp": time.time(),
            "langgraph_version": "Stage 6 Complete"
        }
        
        # Save results
        success = output_manager.save_results(results, output_filename, metadata)
        
        if success:
            logger.info("=" * 50)
            logger.info("PROCESSING COMPLETE")
            logger.info("=" * 50)
            logger.info(f"Model: {args.model}")
            logger.info(f"Dataset: {args.dataset}")
            logger.info(f"Questions processed: {num_processed}")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Output file: {output_filename}")
            
            # Display system metrics
            metrics = system_status["system_metrics"]
            if hasattr(metrics, '__dict__'):
                logger.info(f"Success rate: {metrics.success_rate:.1%}")
                logger.info(f"Total tokens: {metrics.total_token_usage['input'] + metrics.total_token_usage['output']:,}")
                logger.info(f"Processing modes: {dict(metrics.processing_modes_used)}")
            
            logger.info("=" * 50)
        else:
            logger.error("Failed to save results")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def sync_main():
    """Synchronous wrapper for main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    sync_main()