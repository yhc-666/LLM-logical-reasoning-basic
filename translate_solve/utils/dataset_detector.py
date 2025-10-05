#!/usr/bin/env python3
"""
Dataset detection utility for automatically identifying dataset type from sample data
"""

from typing import Dict, Optional


def detect_dataset(item: Dict) -> str:
    """
    Detect which dataset a sample belongs to based on its characteristics.
    
    Detection priority:
    1. Check 'source_dataset' field (if present from mix_dataset.py)
    2. Check ID prefix patterns
    3. Fallback to structural patterns (options count, question format)
    """
    if 'source_dataset' in item:
        return item['source_dataset']
    
    if 'id' in item:
        item_id = str(item['id'])
        
        if item_id.startswith('ProntoQA'):
            return 'ProntoQA'
        elif item_id.startswith('ProofWriter'):
            return 'ProofWriter'
        elif item_id.startswith('logical_deduction'):
            return 'LogicalDeduction'
    
    question = item.get('question', '').lower()
    options = item.get('options', [])
    context = item.get('context', '')
    
    if len(options) == 5 and 'which of the following is true?' in question:
        return 'LogicalDeduction'
    
    if len(options) == 3 and ('true, false, or unknown' in question or 
                              'unknown' in str(options).lower()):
        return 'ProofWriter'
    
    if len(options) == 2 and ('true or false' in question or 
                              'is the following statement true or false' in question):
        return 'ProntoQA'
    
    if 'five objects arranged in a fixed order' in context:
        return 'LogicalDeduction'
    
    if any(word in context.lower() for word in ['jompuses', 'yumpuses', 'dumpuses', 
                                                  'tumpuses', 'rompuses', 'vumpuses',
                                                  'bompuses', 'numpuses', 'wumpuses']):
        return 'ProntoQA'
    
    print(f"Warning: Could not definitively detect dataset for item {item.get('id', 'unknown')}. Defaulting to ProofWriter.")
    return 'ProofWriter'


def get_dataset_statistics(items: list) -> Dict[str, int]:
    stats = {'ProntoQA': 0, 'ProofWriter': 0, 'LogicalDeduction': 0, 'Unknown': 0}
    
    for item in items:
        dataset = detect_dataset(item)
        if dataset in stats:
            stats[dataset] += 1
        else:
            stats['Unknown'] += 1
    
    return stats