#!/bin/bash
# Ablation Study Script
# This script runs the adaptive solver with different SL selection modes for comparison

DATASET="ProofWriter"  # Change to LogicalDeduction or ProntoQA as needed
INPUT_FILE="data/ProofWriter/dev.json"
API_KEY="sk-c54acf253a834d25a8cddec234dd6ccc"
API_URL="https://api.deepseek.com/v1"
MODEL="deepseek-chat"

echo "Running Ablation Study for $DATASET"
echo "====================================="

# 1. Adaptive selection (baseline)
echo "1. Running adaptive SL selection..."
python adaptive_solver/step1_select_sl.py \
    --dataset $DATASET \
    --input_file $INPUT_FILE \
    --output_file results/ablation/$DATASET/adaptive/step1_selected.json \
    --sl_selection adaptive \
    --openai_api_key $API_KEY \
    --openai_base_url $API_URL \
    --model $MODEL

# 2. Force LP
echo "2. Forcing LP for all problems..."
python adaptive_solver/step1_select_sl.py \
    --dataset $DATASET \
    --input_file $INPUT_FILE \
    --output_file results/ablation/$DATASET/LP/step1_selected.json \
    --sl_selection LP \
    --openai_api_key $API_KEY \
    --openai_base_url $API_URL \
    --model $MODEL

# 3. Force FOL
echo "3. Forcing FOL for all problems..."
python adaptive_solver/step1_select_sl.py \
    --dataset $DATASET \
    --input_file $INPUT_FILE \
    --output_file results/ablation/$DATASET/FOL/step1_selected.json \
    --sl_selection FOL \
    --openai_api_key $API_KEY \
    --openai_base_url $API_URL \
    --model $MODEL

# 4. Force SAT
echo "4. Forcing SAT for all problems..."
python adaptive_solver/step1_select_sl.py \
    --dataset $DATASET \
    --input_file $INPUT_FILE \
    --output_file results/ablation/$DATASET/SAT/step1_selected.json \
    --sl_selection SAT \
    --openai_api_key $API_KEY \
    --openai_base_url $API_URL \
    --model $MODEL

echo "====================================="
echo "Ablation study SL selection complete!"
echo "Next steps:"
echo "1. Run step2_translate.py for each output"
echo "2. Run step3_solve.py for each translation"
echo "3. Run step4_evaluate.py to compare results"