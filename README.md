# Symbolic Language Translation for LLM Logical Reasoning

## Overview

This framework enhances logical reasoning capabilities by translating natural language problems into symbolic languages (Logic Programming, First-Order Logic, or SAT) and leveraging specialized symbolic solvers to achieve accurate reasoning.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set permissions for Prover9 binaries
chmod +x solver_engine/src/symbolic_solvers/fol_solver/../Prover9/bin/*
```

## Project Structure

```
adaptive-trans/
├── translate_solve/           # Main pipeline scripts
│   ├── step1_translate.py    # Translate to symbolic language
│   ├── step2_solve.py        # Execute symbolic solvers
│   ├── step3_evaluate.py     # Evaluate results
│   └── utils/                
│       ├── llm_helper.py     # LLM API wrapper
│       └── dataset_detector.py 
│
├── solver_engine/            # Symbolic solver implementations
│   └── src/
│       └── symbolic_solvers/
│           ├── pyke_solver/    
│           ├── fol_solver/   
│           └── z3_solver/  
│
├── data/                     # Datasets
│   ├── ProofWriter/         
│   ├── ProntoQA/           
│   ├── LogicalDeduction/   
│   └── dataset_download.py # Dataset download script
│
├── results/                 # Output directory
├── requirements.txt         
└── README.md               
```

## Quick Start

### Complete Pipeline

Run the 3-step pipeline to solve logical reasoning problems:

```bash
# Step 1: Translate natural language to symbolic language (LP/FOL/SAT)
python translate_solve/step1_translate.py

# Step 2: Execute symbolic solvers
python translate_solve/step2_solve.py

# Step 3: Evaluate results
python translate_solve/step3_evaluate.py
```


## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| **ProofWriter** | 600 | Multi-step deductive reasoning |
| **ProntoQA** | 500 | Ontological reasoning |
| **LogicalDeduction** | 300 | Constraint-based logical puzzles |

### Download ProofWriter with Different Depths
```bash
python data/dataset_download.py
```

## Acknowledgments

- Thanks to LogicLM for symbolic reasoning tools
