"""
ProofWriter Dataset Download Script

Example usage:
    # Download all depth-5 OWA dev samples
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


from datasets import load_dataset
from tqdm import tqdm



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download ProofWriter dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/ProofWriter',
        help='Output directory for downloaded data (default: data/ProofWriter)'
    )

    parser.add_argument(
        '--depth',
        type=str,
        default='5',
        choices=['0', '1', '2', '3', '5', 'all'],
        help='Difficulty level / reasoning depth (default: all)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='dev',
        choices=['train', 'dev', 'test'],
        help='Dataset split to download (default: dev).'
    )

    parser.add_argument(
        '--world_assumption',
        type=str,
        default='OWA',
        choices=['OWA', 'CWA'],
        help='World assumption: OWA (Open World) or CWA (Closed World) (default: OWA)'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=300,
        help='Number of samples to download (default: None = all samples)'
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    parser.add_argument(
        '--reasoning_type',
        type=str,
        default='all',
        choices=['AttNeg', 'AttNoneg', 'RelNoneg', 'RelNeg', 'all'],
        help='Reasoning type configuration (default: all)'
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        default='tasksource/proofwriter',
        help='HuggingFace dataset name (default: tasksource/proofwriter)'
    )

    return parser.parse_args()


def extract_depth_from_id(sample_id: str) -> Optional[int]:
    """
    Extract depth level from ProofWriter sample ID.

    Example IDs:
        - "AttNeg-OWA-D0-4611"
        - "RelNoneg-OWA-D5-1234"

    Args:
        sample_id: Sample identifier string

    Returns:
        Depth level (0-5) or None if not found
    """
    match = re.search(r'-D(\d+)-', sample_id)
    if match:
        return int(match.group(1))
    return None


def extract_reasoning_type_from_id(sample_id: str) -> Optional[str]:
    """
    Extract reasoning type from ProofWriter sample ID.

    Args:
        sample_id: Sample identifier string

    Returns:
        Reasoning type (AttNeg, AttNoneg, RelNoneg, RelNeg) or None
    """
    for rtype in ['AttNeg', 'AttNoneg', 'RelNoneg', 'RelNeg']:
        if sample_id.startswith(rtype):
            return rtype
    return None


def filter_samples(
    dataset,
    depth: str,
    world_assumption: str,
    reasoning_type: str,
    num_samples: Optional[int],
    random_seed: int
) -> List[Dict]:
    """
    Filter dataset samples based on criteria and optionally sample randomly.

    Args:
        dataset: HuggingFace dataset object
        depth: Target depth level ('0'-'5' or 'all')
        world_assumption: 'OWA' or 'CWA'
        reasoning_type: Reasoning configuration type or 'all'
        num_samples: Number of samples to randomly select (None = all)
        random_seed: Random seed for reproducibility

    Returns:
        List of filtered sample dictionaries
    """
    print(f"\n[*] Filtering dataset...")
    print(f"   Depth: {depth}")
    print(f"   World Assumption: {world_assumption}")
    print(f"   Reasoning Type: {reasoning_type}")

    filtered = []
    depth_counts = defaultdict(int)
    reasoning_counts = defaultdict(int)

    for sample in tqdm(dataset, desc="Processing samples"):
        sample_id = sample.get('id', '')

        if world_assumption not in sample_id:
            continue

        sample_reasoning = extract_reasoning_type_from_id(sample_id)
        if reasoning_type != 'all' and sample_reasoning != reasoning_type:
            continue

        sample_depth = extract_depth_from_id(sample_id)
        if sample_depth is None:
            continue

        if depth != 'all' and sample_depth != int(depth):
            continue

        filtered.append(sample)
        depth_counts[sample_depth] += 1
        if sample_reasoning:
            reasoning_counts[sample_reasoning] += 1

    print(f"\n[*] Filtering Statistics:")
    print(f"   Total samples: {len(filtered)}")
    print(f"   Depth distribution: {dict(depth_counts)}")
    print(f"   Reasoning type distribution: {dict(reasoning_counts)}")

    if num_samples is not None and num_samples < len(filtered):
        print(f"\n[*] Randomly sampling {num_samples} samples (seed={random_seed})...")
        random.seed(random_seed)
        filtered = random.sample(filtered, num_samples)

    return filtered


def convert_to_standard_format(samples: List[Dict]) -> List[Dict]:
    """
    Convert HuggingFace dataset format to project's standard format.

    Expected output format:
    {
        "id": "ProofWriter_RelNoneg-OWA-D5-861_Q14",
        "context": "...",
        "question": "...",
        "options": ["A) True", "B) False", "C) Unknown"],
        "answer": "C",
        "source_dataset": "ProofWriter"
    }

    Args:
        samples: List of samples in HuggingFace format

    Returns:
        List of samples in standard format
    """
    print(f"\n[*] Converting to standard format...")

    converted = []
    for sample in tqdm(samples, desc="Converting samples"):
        standard_sample = {
            "id": f"ProofWriter_{sample.get('id', 'unknown')}",
            "context": sample.get('theory', '') or sample.get('context', ''),
            "question": sample.get('question', ''),
            "options": [],
            "answer": "",
            "source_dataset": "ProofWriter"
        }

        question_text = standard_sample["question"]
        if not question_text.endswith('?'):
            question_text = f"Based on the above information, is the following statement true, false, or unknown? {question_text}"
        standard_sample["question"] = question_text

        answer = sample.get('answer', '').strip()

        if answer.lower() in ['true', 't', 'yes']:
            standard_sample["answer"] = "A"
        elif answer.lower() in ['false', 'f', 'no']:
            standard_sample["answer"] = "B"
        elif answer.lower() in ['unknown', 'u', 'unk']:
            standard_sample["answer"] = "C"
        else:
            standard_sample["answer"] = answer

        standard_sample["options"] = [
            "A) True",
            "B) False",
            "C) Unknown"
        ]

        converted.append(standard_sample)

    return converted


def save_dataset(samples: List[Dict], output_path: str):
    """
    Save dataset to JSON file.

    Args:
        samples: List of sample dictionaries
        output_path: Path to output JSON file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Saved {len(samples)} samples to: {output_path}")


def main():
    """Main execution function."""
    args = parse_args()

    split = 'validation' if args.split == 'dev' else args.split

    print(f"\n[*] Configuration:")
    print(f"   Dataset: {args.dataset_name}")
    print(f"   Split: {args.split} ({split})")
    print(f"   Depth: {args.depth}")
    print(f"   World Assumption: {args.world_assumption}")
    print(f"   Reasoning Type: {args.reasoning_type}")
    print(f"   Sample Size: {args.num_samples or 'All'}")
    print(f"   Random Seed: {args.random_seed}")
    print(f"   Output Directory: {args.output_dir}")

    print(f"\n[*] Loading dataset from HuggingFace...")
    try:
        dataset = load_dataset(args.dataset_name, split=split)
        print(f"   Loaded {len(dataset)} samples from {split} split")
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        print(f"   Please check if '{args.dataset_name}' is a valid HuggingFace dataset")
        return

    filtered_samples = filter_samples(
        dataset=dataset,
        depth=args.depth,
        world_assumption=args.world_assumption,
        reasoning_type=args.reasoning_type,
        num_samples=args.num_samples,
        random_seed=args.random_seed
    )

    if not filtered_samples:
        print("\n[!] No samples matched the filtering criteria!")
        print("   Try adjusting your filters (depth, world_assumption, reasoning_type)")
        return

    standard_samples = convert_to_standard_format(filtered_samples)

    depth_str = f"depth{args.depth}" if args.depth != 'all' else "all_depths"
    reasoning_str = args.reasoning_type if args.reasoning_type != 'all' else "all_types"
    num_str = f"_{args.num_samples}samples" if args.num_samples else ""

    filename = f"{args.split}_{depth_str}_{args.world_assumption}_{reasoning_str}{num_str}.json"
    output_path = os.path.join(args.output_dir, filename)

    save_dataset(standard_samples, output_path)



if __name__ == "__main__":
    main()
