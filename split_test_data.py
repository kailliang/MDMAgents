#!/usr/bin/env python3
"""
Script to split the original test dataset into three difficulty-based subsets.
Reads from data/medqa/test.jsonl and creates three separate files based on difficulty assessment.
"""

import os
import json
from tqdm import tqdm
from termcolor import cprint
from utils import determine_difficulty, create_question

def main():
    # Input and output paths
    input_file = 'data/medqa/134test.jsonl'
    output_dir = 'data/medqa'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        cprint(f"Error: Input file {input_file} not found!", "red")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data containers
    basic_data = []
    intermediate_data = []
    advanced_data = []
    
    # Load test data
    cprint(f"Loading data from {input_file}...", "cyan")
    test_samples = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_samples.append(json.loads(line.strip()))
        cprint(f"Loaded {len(test_samples)} test samples", "green")
    except Exception as e:
        cprint(f"Error loading test data: {e}", "red")
        return
    
    # Process each sample and determine difficulty
    cprint("Analyzing difficulty for each sample...", "cyan")
    
    for i, sample in enumerate(tqdm(test_samples, desc="Processing samples")):
        try:
            # Create question format (same as in main.py)
            question, _ = create_question(sample, 'medqa')
            
            # Determine difficulty using adaptive mode with default model
            difficulty, input_tokens, output_tokens = determine_difficulty(question, 'adaptive')
            
            # Add difficulty info to sample
            sample_with_difficulty = sample.copy()
            sample_with_difficulty['difficulty'] = difficulty
            sample_with_difficulty['difficulty_tokens'] = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }
            
            # Sort into appropriate difficulty bucket
            if difficulty == 'basic':
                basic_data.append(sample_with_difficulty)
            elif difficulty == 'intermediate':
                intermediate_data.append(sample_with_difficulty)
            elif difficulty == 'advanced':
                advanced_data.append(sample_with_difficulty)
            else:
                cprint(f"Warning: Unknown difficulty '{difficulty}' for sample {i}. Adding to intermediate.", "yellow")
                intermediate_data.append(sample_with_difficulty)
                
        except Exception as e:
            cprint(f"Error processing sample {i}: {e}", "red")
            # Add to intermediate as fallback
            sample_with_difficulty = sample.copy()
            sample_with_difficulty['difficulty'] = 'intermediate'
            sample_with_difficulty['difficulty_tokens'] = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            intermediate_data.append(sample_with_difficulty)
    
    # Print statistics
    cprint("\n" + "="*50, "magenta")
    cprint("DIFFICULTY DISTRIBUTION", "magenta")
    cprint("="*50, "magenta")
    cprint(f"Basic: {len(basic_data)} samples ({len(basic_data)/len(test_samples)*100:.1f}%)", "green")
    cprint(f"Intermediate: {len(intermediate_data)} samples ({len(intermediate_data)/len(test_samples)*100:.1f}%)", "yellow")
    cprint(f"Advanced: {len(advanced_data)} samples ({len(advanced_data)/len(test_samples)*100:.1f}%)", "red")
    cprint(f"Total: {len(basic_data) + len(intermediate_data) + len(advanced_data)} samples", "cyan")
    cprint("="*50, "magenta")
    
    # Save the split datasets
    output_files = {
        'basic': os.path.join(output_dir, 'test_basic.jsonl'),
        'intermediate': os.path.join(output_dir, 'test_intermediate.jsonl'),
        'advanced': os.path.join(output_dir, 'test_advanced.jsonl')
    }
    
    datasets = {
        'basic': basic_data,
        'intermediate': intermediate_data,
        'advanced': advanced_data
    }
    
    cprint("\nSaving split datasets...", "cyan")
    for difficulty, data in datasets.items():
        output_file = output_files[difficulty]
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            cprint(f"Saved {len(data)} samples to {output_file}", "green")
        except Exception as e:
            cprint(f"Error saving {difficulty} data: {e}", "red")
    
    # Calculate and display token usage statistics
    total_input_tokens = sum(
        sample.get('difficulty_tokens', {}).get('input_tokens', 0) 
        for dataset in datasets.values() 
        for sample in dataset
    )
    total_output_tokens = sum(
        sample.get('difficulty_tokens', {}).get('output_tokens', 0) 
        for dataset in datasets.values() 
        for sample in dataset
    )
    
    cprint(f"\nToken Usage for Difficulty Assessment:", "cyan")
    cprint(f"Total Input Tokens: {total_input_tokens:,}", "yellow")
    cprint(f"Total Output Tokens: {total_output_tokens:,}", "yellow")
    cprint(f"Total Tokens: {total_input_tokens + total_output_tokens:,}", "yellow")
    
    cprint("\nDataset splitting completed successfully!", "green")

if __name__ == "__main__":
    main()