#!/usr/bin/env python3
"""
Step 4: Evaluate Results
Evaluates the performance of the adaptive solver system
"""

import argparse
import json
import os
from typing import Dict, List


def load_data(input_file: str) -> List[Dict]:
    """Load input data from JSON file"""
    with open(input_file, 'r') as f:
        return json.load(f)


def save_evaluation(output_file: str, evaluation: str):
    """Save evaluation results to text file"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(evaluation)


def normalize_answer(answer: str) -> str:
    """Normalize answer format for comparison"""
    if not answer:
        return ''
    
    answer = str(answer).strip().upper()
    
    # Handle True/False to A/B conversion
    if answer == 'TRUE':
        return 'A'
    elif answer == 'FALSE':
        return 'B'
    
    # Extract letter from formats like "A)" or "(A)"
    if len(answer) >= 2:
        if answer[0] in 'ABCDEFG':
            return answer[0]
        elif answer[0] == '(' and answer[-1] == ')' and len(answer) == 3:
            return answer[1]
    
    return answer


def evaluate_results(data: List[Dict]) -> Dict:
    """
    Evaluate results and compute accuracy
    
    Args:
        data: List of result items with gold_answer and final_answer
    
    Returns:
        Dictionary with evaluation metrics
    """
    total = len(data)
    correct = 0
    
    # Track performance by SL
    sl_performance = {
        'LP': {'correct': 0, 'total': 0},
        'FOL': {'correct': 0, 'total': 0},
        'SAT': {'correct': 0, 'total': 0},
    }
    
    # Track solver status
    status_counts = {
        'success': 0,
        'parsing error': 0,
        'execution error': 0,
        'no translation': 0,
    }
    
    # Evaluate each item
    for item in data:
        gold = normalize_answer(item.get('gold_answer', ''))
        predicted = normalize_answer(item.get('final_answer', ''))
        sl = item.get('SL', 'LP')
        status = item.get('solver_status', 'unknown')
        
        # Update status counts
        if status in status_counts:
            status_counts[status] += 1
        
        # Check correctness
        is_correct = (gold == predicted) and gold != ''
        if is_correct:
            correct += 1
        
        # Update SL performance
        if sl in sl_performance:
            sl_performance[sl]['total'] += 1
            if is_correct:
                sl_performance[sl]['correct'] += 1
        
        # Store evaluation in item (for debugging)
        item['is_correct'] = is_correct
        item['normalized_gold'] = gold
        item['normalized_predicted'] = predicted
    
    # Calculate overall accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Calculate SL-specific accuracies
    sl_accuracies = {}
    for sl, perf in sl_performance.items():
        if perf['total'] > 0:
            sl_accuracies[sl] = perf['correct'] / perf['total'] * 100
        else:
            sl_accuracies[sl] = 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'sl_performance': sl_performance,
        'sl_accuracies': sl_accuracies,
        'status_counts': status_counts,
    }


def format_evaluation_report(evaluation: Dict, dataset: str) -> str:
    """Format evaluation results as a readable report"""
    report = []
    report.append("=" * 60)
    report.append(f"EVALUATION REPORT - {dataset}")
    report.append("=" * 60)
    report.append("")
    
    # Overall performance
    report.append("OVERALL PERFORMANCE:")
    report.append("-" * 30)
    report.append(f"Total problems: {evaluation['total']}")
    report.append(f"Correct predictions: {evaluation['correct']}")
    report.append(f"Overall accuracy: {evaluation['accuracy']:.2f}%")
    report.append("")
    
    # Performance by SL
    report.append("PERFORMANCE BY SYMBOLIC LANGUAGE:")
    report.append("-" * 30)
    for sl in ['LP', 'FOL', 'SAT']:
        perf = evaluation['sl_performance'][sl]
        acc = evaluation['sl_accuracies'][sl]
        if perf['total'] > 0:
            report.append(f"{sl}:")
            report.append(f"  Problems: {perf['total']}")
            report.append(f"  Correct: {perf['correct']}")
            report.append(f"  Accuracy: {acc:.2f}%")
    report.append("")
    
    # Solver status distribution
    report.append("SOLVER STATUS DISTRIBUTION:")
    report.append("-" * 30)
    for status, count in evaluation['status_counts'].items():
        percentage = (count / evaluation['total'] * 100) if evaluation['total'] > 0 else 0
        report.append(f"{status}: {count} ({percentage:.1f}%)")
    report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Step 4: Evaluate adaptive solver results')
    parser.add_argument('--input_file', type=str, default='results/deepseek/ProofWriter/solve/result.json', # ProofWriter/LogicalDeduction/ProntoQA
                       help='Input JSON file path (from step 3)')
    parser.add_argument('--output_file', type=str, default='results/deepseek/ProofWriter/final/result.txt',
                       help='Output evaluation report file path')
    parser.add_argument('--dataset', type=str, default='ProofWriter',
                       help='Dataset name for report header')
    parser.add_argument('--save_detailed',default=True,
                       help='Save detailed results with correctness flags')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading results from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} results")
    
    # Evaluate
    print("Evaluating results...")
    evaluation = evaluate_results(data)
    
    # Format report
    report = format_evaluation_report(evaluation, args.dataset)
    
    # Print report to console
    print("\n" + report)
    
    # Save report
    print(f"\nSaving evaluation report to {args.output_file}...")
    save_evaluation(args.output_file, report)
    
    # Optionally save detailed results
    if args.save_detailed:
        detailed_file = args.output_file.replace('.txt', '_detailed.json')
        print(f"Saving detailed results to {detailed_file}...")
        with open(detailed_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()