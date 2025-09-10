#!/usr/bin/env python3
"""
Dataset mixer script that randomly samples from ProntoQA, ProofWriter, and LogicalDeduction datasets.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sample_from_dataset(data: List[Dict[str, Any]], n_samples: int, dataset_name: str) -> List[Dict[str, Any]]:
    """
    Randomly sample n_samples from the dataset.
    Adds source dataset information to each sample.
    """
    if n_samples > len(data):
        print(f"Warning: Requested {n_samples} samples from {dataset_name}, but only {len(data)} available. Using all.")
        n_samples = len(data)
    
    sampled = random.sample(data, n_samples)
    
    for sample in sampled:
        sample['source_dataset'] = dataset_name
    
    return sampled


def mix_datasets(
    prontoqa_path: Path,
    proofwriter_path: Path,
    logicaldeduction_path: Path,
    n_prontoqa: int,
    n_proofwriter: int,
    n_logicaldeduction: int,
    output_path: Path = None,
    seed: int = None
) -> List[Dict[str, Any]]:
    """
    Mix samples from three datasets.
    
    Args:
        prontoqa_path: Path to ProntoQA dataset
        proofwriter_path: Path to ProofWriter dataset
        logicaldeduction_path: Path to LogicalDeduction dataset
        n_prontoqa: Number of samples from ProntoQA
        n_proofwriter: Number of samples from ProofWriter
        n_logicaldeduction: Number of samples from LogicalDeduction
        output_path: Optional path to save mixed dataset
        seed: Random seed for reproducibility
    
    Returns:
        List of mixed samples
    """
    if seed is not None:
        random.seed(seed)
    
    # Load datasets
    print(f"Loading ProntoQA from {prontoqa_path}")
    prontoqa_data = load_json_data(prontoqa_path)
    
    print(f"Loading ProofWriter from {proofwriter_path}")
    proofwriter_data = load_json_data(proofwriter_path)
    
    print(f"Loading LogicalDeduction from {logicaldeduction_path}")
    logicaldeduction_data = load_json_data(logicaldeduction_path)
    
    # Sample from each dataset
    mixed_data = []
    
    if n_prontoqa > 0:
        prontoqa_samples = sample_from_dataset(prontoqa_data, n_prontoqa, 'ProntoQA')
        mixed_data.extend(prontoqa_samples)
        print(f"Sampled {len(prontoqa_samples)} from ProntoQA")
    
    if n_proofwriter > 0:
        proofwriter_samples = sample_from_dataset(proofwriter_data, n_proofwriter, 'ProofWriter')
        mixed_data.extend(proofwriter_samples)
        print(f"Sampled {len(proofwriter_samples)} from ProofWriter")
    
    if n_logicaldeduction > 0:
        logicaldeduction_samples = sample_from_dataset(logicaldeduction_data, n_logicaldeduction, 'LogicalDeduction')
        mixed_data.extend(logicaldeduction_samples)
        print(f"Sampled {len(logicaldeduction_samples)} from LogicalDeduction")
    
    # Shuffle the mixed dataset
    random.shuffle(mixed_data)
    
    print(f"\nTotal samples in mixed dataset: {len(mixed_data)}")
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mixed_data, f, indent=2, ensure_ascii=False)
        print(f"Mixed dataset saved to {output_path}")
    
    return mixed_data


def main():
    parser = argparse.ArgumentParser(
        description='Mix samples from ProntoQA, ProofWriter, and LogicalDeduction datasets'
    )
    
    # Number of samples arguments
    parser.add_argument(
        '--n_prontoqa', 
        type=int, 
        default=100,
        help='Number of samples from ProntoQA dataset'
    )
    parser.add_argument(
        '--n_proofwriter', 
        type=int, 
        default=100,
        help='Number of samples from ProofWriter dataset'
    )
    parser.add_argument(
        '--n_logicaldeduction', 
        type=int, 
        default=100,
        help='Number of samples from LogicalDeduction dataset'
    )
    
    # Dataset path arguments
    parser.add_argument(
        '--prontoqa_path',
        type=str,
        default='data/ProntoQA/dev.json',
        help='Path to ProntoQA dataset'
    )
    parser.add_argument(
        '--proofwriter_path',
        type=str,
        default='data/ProofWriter/dev.json',
        help='Path to ProofWriter dataset'
    )
    parser.add_argument(
        '--logicaldeduction_path',
        type=str,
        default='data/LogicalDeduction/dev.json',
        help='Path to LogicalDeduction dataset'
    )
    
    # Output and seed arguments
    parser.add_argument(
        '--output',
        type=str,
        default='data/Mixed/dev.json',
        help='Output path for mixed dataset (optional)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (optional)'
    )
    
    args = parser.parse_args()
    
    prontoqa_path = Path(args.prontoqa_path)
    proofwriter_path = Path(args.proofwriter_path)
    logicaldeduction_path = Path(args.logicaldeduction_path)
    output_path = Path(args.output) if args.output else None
    
    # Check if input files exist
    for path, name in [
        (prontoqa_path, 'ProntoQA'),
        (proofwriter_path, 'ProofWriter'),
        (logicaldeduction_path, 'LogicalDeduction')
    ]:
        if not path.exists():
            print(f"Error: {name} dataset not found at {path}")
            return 1
    
    mixed_data = mix_datasets(
        prontoqa_path=prontoqa_path,
        proofwriter_path=proofwriter_path,
        logicaldeduction_path=logicaldeduction_path,
        n_prontoqa=args.n_prontoqa,
        n_proofwriter=args.n_proofwriter,
        n_logicaldeduction=args.n_logicaldeduction,
        output_path=output_path,
        seed=args.seed
    )
    
    # summary statistics
    dataset_counts = {}
    for sample in mixed_data:
        source = sample.get('source_dataset', 'Unknown')
        dataset_counts[source] = dataset_counts.get(source, 0) + 1
    
    print("\nDataset distribution in mixed dataset:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} samples")
    
    return 0


if __name__ == '__main__':
    exit(main())