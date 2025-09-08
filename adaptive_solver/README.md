# Adaptive Symbolic Language Selection System

This system automatically selects the optimal symbolic language (LP/FOL/SAT) for each logic problem and solves it using the appropriate solver.

## Components

1. **step1_select_sl.py** - Analyzes each problem and selects the best symbolic language
2. **step2_translate.py** - Translates problems to selected SL using complete prompt templates
3. **step3_solve.py** - Routes problems to appropriate symbolic solvers (Pyke/Prover9/Z3)
4. **step4_evaluate.py** - Evaluates overall accuracy per dataset

## Usage

Run the pipeline sequentially:

```bash
# Step 1: Select optimal SL for each problem
python adaptive_solver/step1_select_sl.py \
    --dataset ProofWriter \
    --input_file data/ProofWriter/dev.json \
    --output_file outputs/adaptive/ProofWriter/step1_selected.json \
    --openai_api_key "YOUR_API_KEY" \
    --openai_base_url "YOUR_API_BASE_URL" \
    --model gpt-4 \
    --temperature 0.0

# Step 2: Translate to selected SL
python adaptive_solver/step2_translate.py \
    --dataset ProofWriter \
    --input_file outputs/adaptive/ProofWriter/step1_selected.json \
    --output_file outputs/adaptive/ProofWriter/step2_translated.json \
    --openai_api_key "YOUR_API_KEY" \
    --openai_base_url "YOUR_API_BASE_URL" \
    --model gpt-4 \
    --temperature 0.0

# Step 3: Solve using symbolic solvers
python adaptive_solver/step3_solve.py \
    --dataset ProofWriter \
    --input_file outputs/adaptive/ProofWriter/step2_translated.json \
    --output_file outputs/adaptive/ProofWriter/step3_solved.json

# Step 4: Evaluate results
python adaptive_solver/step4_evaluate.py \
    --input_file outputs/adaptive/ProofWriter/step3_solved.json \
    --output_file outputs/adaptive/ProofWriter/evaluation.txt \
    --dataset ProofWriter \
    --save_detailed
```

## Supported Datasets

- **ProofWriter**: Logic problems with context and question
- **ProntoQA**: Logic problems with context and question  
- **LogicalDeduction**: Logic problems with context, question, and multiple-choice options

## Features

- Automatic SL selection based on problem characteristics
- Complete prompt templates with ${} placeholder substitution
- Dataset-specific role descriptions for each SL
- Integration with existing solver infrastructure
- Comprehensive evaluation metrics

## Requirements

- Python 3.8+
- OpenAI API access
- Existing solver infrastructure (Pyke, Prover9, Z3)
- Required packages: see requirements.txt in parent directory