#!/usr/bin/env python3
"""
Step 2: Solve using Symbolic Solvers
Solves each problem using the appropriate symbolic solver based on selected SL
"""

import argparse
import json
import os
import sys
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add solver_engine to path
solver_engine_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    'solver_engine', 'src'
)
sys.path.insert(0, solver_engine_path)
 
from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program

# Import dataset detector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset_detector import detect_dataset


SOLVER_CLASSES = {
    'LP': Pyke_Program,
    'FOL': FOL_Prover9_Program,
    'SAT': LSAT_Z3_Program,
}


def load_data(input_file: str) -> List[Dict]:
    """Load input data from JSON file"""
    with open(input_file, 'r') as f:
        return json.load(f)


def save_data(output_file: str, data: List[Dict]):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def execute_solver(sl: str, translation: str, item: Dict) -> Tuple[str, str, str, str]:
    """
    Execute symbolic solver for a problem
    
    Args:
        sl: Symbolic language (LP/FOL/SAT)
        translation: Translation in symbolic language
        item: Problem item to detect dataset from
    
    Returns:
        Tuple of (answer, status_code, error_message, reasoning)
    """
    try:
        # Detect dataset from item
        dataset_name = detect_dataset(item)
        
        solver_class = SOLVER_CLASSES[sl]
        program = solver_class(translation, dataset_name)
        
        # Check if parsing succeeded
        if not getattr(program, 'flag', True):
            return 'A', 'parsing error', 'Failed to parse symbolic program', ''
        
        # Execute the program
        try:
            answer, err, reasoning = program.execute_program()
        except Exception as e:
            return 'A', 'execution error', str(e), ''
        
        if answer is None:
            err_str = str(err) if err is not None else 'Unknown error'
            return 'A', 'execution error', err_str, ''
        
        mapped = program.answer_mapping(answer)
        
        status_code = 'success'
        error_message = ''
        if reasoning == '' and sl in ['LP', 'FOL']:
            status_code = 'execution error'
            error_message = 'Empty reasoning indicates execution failure'
        
        return mapped, status_code, error_message, reasoning
        
    except Exception as e:
        return 'A', 'execution error', str(e), ''


def get_gold_answer(item: Dict) -> str:
    """Extract gold answer from item"""
    answer = item.get('answer', '')
    if answer in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return answer
    elif answer in ['True', 'False']:
        return 'A' if answer == 'True' else 'B'
    else:
        if isinstance(answer, str) and len(answer) > 0:
            if len(answer) >= 2 and answer[1] == ')':
                return answer[0].upper()
        return answer


def main():
    parser = argparse.ArgumentParser(description='Step 2: Solve problems using symbolic solvers')
    parser.add_argument('--input_file', type=str, default='results/deepseek/translation/result.json',
                       help='Input JSON file path (from step 1)')
    parser.add_argument('--output_file', type=str, default='results/deepseek/solve/result.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} problems")
    
    results = []
    stats = {
        'LP': {'success': 0, 'parsing error': 0, 'execution error': 0, 'total': 0},
        'FOL': {'success': 0, 'parsing error': 0, 'execution error': 0, 'total': 0},
        'SAT': {'success': 0, 'parsing error': 0, 'execution error': 0, 'total': 0},
    }
    
    for item in tqdm(data, desc="Solving problems"):
        sl = item.get('SL', 'LP')
        translation_dict = item.get('translation', {})
        
        if isinstance(translation_dict, dict):
            translation = translation_dict.get(sl, '')
        else:
            translation = str(translation_dict)
        
        # Skip if no translation
        if not translation or "Translation failed" in translation:
            result_item = item.copy()
            result_item['final_answer'] = 'A'
            result_item['gold_answer'] = get_gold_answer(item)
            result_item['solver_status'] = 'no translation'
            result_item['solver_error'] = 'No valid translation available'
            result_item['reasoning'] = ''
            results.append(result_item)
            continue
        
        # Execute solver
        answer, status_code, error_message, reasoning = execute_solver(
            sl, translation, item
        )
        
        stats[sl][status_code] += 1
        stats[sl]['total'] += 1
        
        result_item = item.copy()
        result_item['final_answer'] = answer
        result_item['gold_answer'] = get_gold_answer(item)
        result_item['solver_status'] = status_code
        result_item['solver_error'] = error_message
        result_item['reasoning'] = reasoning
        
        results.append(result_item)
    
    print(f"\nSaving results to {args.output_file}...")
    save_data(args.output_file, results)
    
    print("\n=== Solver Statistics ===")
    print(f"{'SL':<5} {'Success':<10} {'Parsing Error':<15} {'Execution Error':<17} {'Total':<8}")
    print("-" * 60)
    
    for sl in ['LP', 'FOL', 'SAT']:
        if stats[sl]['total'] > 0:
            success = stats[sl]['success']
            parsing_error = stats[sl]['parsing error']
            execution_error = stats[sl]['execution error']
            total = stats[sl]['total']
            print(f"{sl:<5} {success:<10} {parsing_error:<15} {execution_error:<17} {total:<8}")
    
    print("-" * 60)
    
    # Overall statistics
    total_problems = len(results)
    total_success = sum(stats[sl]['success'] for sl in ['LP', 'FOL', 'SAT'])
    print(f"\nTotal problems: {total_problems}")
    print(f"Total successful solves: {total_success}")
    if total_problems > 0:
        print(f"Success rate: {(total_success / total_problems) * 100:.1f}%")
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()