#!/usr/bin/env python3
"""
Step 1: Select Symbolic Language
Analyzes each logic problem and selects the most appropriate symbolic language (LP/FOL/SAT)
"""

import argparse
import json
import os
import random
from tqdm import tqdm
from typing import Dict, List

from utils.llm_helper import LLMHelper
from utils.dataset_detector import detect_dataset


def load_data(input_file: str) -> List[Dict]:
    """Load input data from JSON file"""
    with open(input_file, 'r') as f:
        return json.load(f)


def save_data(output_file: str, data: List[Dict]):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def substitute_prompt_placeholders(prompt_template: str, item: Dict) -> str:
    """
    Substitute placeholders in prompt template
    
    Args:
        prompt_template: Template with placeholders
        item: Problem item
    
    Returns:
        Filled prompt
    """
    context = item.get('context', '')
    question = item.get('question', '')
    
    prompt = prompt_template.replace('${context}', context)
    prompt = prompt.replace('${question}', question)
    
    # Detect dataset type for this specific item
    dataset = detect_dataset(item)
    
    if dataset == 'LogicalDeduction':
        options = item.get('options', [])
        options_text = 'Options:\n' + '\n'.join(options) if isinstance(options, list) else str(options)
        prompt = prompt.replace('${options}', options_text)
    else:
        prompt = prompt.replace('${options}', '')
    
    return prompt


def select_sl_for_problem(llm_helper: LLMHelper, filled_prompt: str) -> Dict:
    """
    Select symbolic language for a single problem
    
    Args:
        llm_helper: LLM helper instance
        filled_prompt: Prompt with placeholders filled
    
    Returns:
        Dict with selected_sl and reasoning
    """
    try:
        # Use a simple system prompt since the full instructions are in the filled prompt
        system_prompt = "You are an expert in symbolic logic and reasoning systems."
        result = llm_helper.generate(
            system_prompt=system_prompt,
            user_prompt=filled_prompt,
            return_json=True
        )
        
        # Validate result
        if isinstance(result, dict) and 'selected_sl' in result:
            print(result['selected_sl'])
            if result['selected_sl'] not in ['LP', 'FOL', 'SAT']:
                print(f"Warning: Invalid SL '{result['selected_sl']}', defaulting to LP")
                result['selected_sl'] = 'LP'
                result['reasoning'] = f"Defaulted to LP due to invalid selection"
            return result
        else:
            return {
                'selected_sl': 'LP',
                'reasoning': 'Defaulted to LP due to parsing error'
            }
    except Exception as e:
        print(f"Error in SL selection: {e}")
        return {
            'selected_sl': 'LP',
            'reasoning': f'Defaulted to LP due to error: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(description='Step 1: Select Symbolic Language for logic problems')
    parser.add_argument('--input_file', type=str, default='data/Mixed/dev.json',
                       help='Input JSON file path')
    parser.add_argument('--output_file', type=str, default='results/deepseek/LogicalDeduction/random/select_sl/result.json',
                       help='Output JSON file path')
    parser.add_argument('--sl_selection', type=str, default='random',
                       choices=['adaptive', 'random', 'LP', 'FOL', 'SAT'],
                       help='SL selection mode: adaptive (use LLM), random (randomly assign), or force specific SL for ablation')
    parser.add_argument('--openai_api_key', type=str, default="1785146889328074779",
                       help='OpenAI API key')
    parser.add_argument('--openai_base_url', type=str, default="https://aigc.sankuai.com/v1/openai/native",
                       help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default='gpt-4-0613', 
                       help='Model name (default: gpt-4-0613)') 
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for generation (default: 0.0)')
    parser.add_argument('--max_tokens', type=int, default=1000,
                       help='Maximum tokens for generation (default: 1000)')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducibility when using random SL selection')
    
    args = parser.parse_args()
    
    # Print mode information
    print(f"SL Selection Mode: {args.sl_selection}")
    if args.sl_selection == 'random':
        print(f"Randomly assigning LP/FOL/SAT to each problem")
        if args.random_seed is not None:
            random.seed(args.random_seed)
            print(f"Using random seed: {args.random_seed}")
    elif args.sl_selection not in ['adaptive', 'random']:
        print(f"Forcing all problems to use {args.sl_selection} for ablation study")
    
    # Only load prompt template and initialize LLM if in adaptive mode
    prompt_template = None
    llm_helper = None
    
    if args.sl_selection == 'adaptive':
        prompt_file = os.path.join(os.path.dirname(__file__), 'prompts', 'sl_selection_prompt.txt')
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
        
        llm_helper = LLMHelper(
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} problems")
    
    results = []
    sl_counts = {'LP': 0, 'FOL': 0, 'SAT': 0}
    
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    
    for idx, item in enumerate(tqdm(data, desc="Selecting SL for problems")):
        result_item = item.copy()
        
        if args.sl_selection == 'adaptive':
            filled_prompt = substitute_prompt_placeholders(prompt_template, item)
            
            # Select SL
            sl_result = select_sl_for_problem(llm_helper, filled_prompt)
            
            result_item['SL'] = sl_result['selected_sl']
            result_item['SL_reasoning'] = sl_result['reasoning']
        elif args.sl_selection == 'random':
            result_item['SL'] = random.choice(['LP', 'FOL', 'SAT'])
            result_item['SL_reasoning'] = f"Randomly selected {result_item['SL']}"
        else:
            result_item['SL'] = args.sl_selection
            result_item['SL_reasoning'] = f"Forced to {args.sl_selection} for ablation study"
        
        results.append(result_item)
        sl_counts[result_item['SL']] += 1
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinal save to {args.output_file}...")
    save_data(args.output_file, results)
    
    print("\n=== SL Selection Statistics ===")
    print(f"Total problems: {len(data)}")
    for sl, count in sl_counts.items():
        percentage = (count / len(data)) * 100 if len(data) > 0 else 0
        print(f"{sl}: {count} ({percentage:.1f}%)")
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()