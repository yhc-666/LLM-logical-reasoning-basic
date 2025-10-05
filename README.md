# Official implementation of **"Adaptive Translation for LLM Logical Reasoning"**.

## Overview

This framework enhances logical reasoning capabilities by adaptively selecting the optimal symbolic language (Logic Programming, First-Order Logic, or SAT) for each problem and leveraging specialized symbolic solvers. 

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/adaptive-trans.git
cd adaptive-trans

# Install dependencies
pip install -r requirements.txt

# Set permissions for Prover9 binaries
chmod +x solver_engine/src/symbolic_solvers/fol_solver/../Prover9/bin/*
```

## Quick Start

### Complete Pipeline

```bash
# 1. Select optimal symbolic language for each problem
python adaptive_solver/step1_select_sl.py \
    --dataset ProofWriter \
    --input_file data/ProofWriter/dev.json \
    --output_file results/ProofWriter/select_sl/result.json \
    --model gpt-4

# 2. Translate to selected symbolic language
python adaptive_solver/step2_translate.py \
    --dataset ProofWriter \
    --input_file results/ProofWriter/select_sl/result.json \
    --output_file results/ProofWriter/translation/result.json \
    --model gpt-4

# 3. Execute symbolic solvers
python adaptive_solver/step3_solve.py \
    --dataset ProofWriter \
    --input_file results/ProofWriter/translation/result.json \
    --output_file results/ProofWriter/solve/result.json

# 4. Evaluate results
python adaptive_solver/step4_evaluate.py \
    --input_file results/ProofWriter/solve/result.json \
    --output_file results/ProofWriter/evaluation.txt
```



## Key Components

### 1. Adaptive SL Selection (`step1_select_sl.py`)
- Analyzes problem structure (predicates, quantifiers, constraints)
- Provides selection reasoning for interpretability
- Supports batch processing with progress tracking

### 2. Translation Engine (`step2_translate.py`)
- Template-based translation with placeholder substitution
- Dataset-specific prompt engineering
- Handles complex logical constructs (implications, quantifiers, predicates)

### 3. Solver Integration (`step3_solve.py`)
- Unified interface to multiple symbolic solvers
- Automatic error handling and fallback mechanisms
- Detailed execution statistics and reasoning traces

### 4. Evaluation Framework (`step4_evaluate.py`)
- Per-dataset and per-SL accuracy metrics
- Error analysis (parsing, execution, translation)
- Comparative performance reports

## Datasets

| Dataset | Size |
|---------|------|
| **ProofWriter** | 600 |
| **ProntoQA** | 500 |
| **LogicalDeduction** | 300 |

### ProofWriter Download with Depth 0-5
```bash
python data/dataset_download.py
```


## Acknowledgments
We thank the developers of LogicLM for their excellent symbolic reasoning tools.

