#!/bin/bash

# Wait for Q1 to complete
echo "Waiting for Q1 training to complete..."
while pgrep -f 'python3 main.py --train --eval' > /dev/null; do
    sleep 60
done

echo "Q1 Complete! Checking results..."
tail -20 training_q1.log

# Q2: Evaluate on transformed test set
echo ""
echo "======================================"
echo "Starting Q2: Evaluating on transformed test set"
echo "======================================"
source ~/miniconda3/bin/activate hw4-part-1-nlp
python3 main.py --eval_transformed > training_q2.log 2>&1
echo "Q2 Complete!"
tail -10 training_q2.log

# Q3: Train with augmented data and evaluate
echo ""
echo "======================================"
echo "Starting Q3: Training with augmented data"
echo "======================================"
python3 main.py --train_augmented --eval_transformed > training_q3.log 2>&1
echo "Q3 training Complete!"
tail -10 training_q3.log

# Q3: Evaluate augmented model on original test set
echo ""
echo "======================================"
echo "Q3: Evaluating augmented model on original test set"
echo "======================================"
python3 main.py --eval --model_dir out_augmented > training_q3_original.log 2>&1
echo "Q3 original eval Complete!"
tail -10 training_q3_original.log

# Q3: Evaluate augmented model on transformed test set (already done in training)
echo ""
echo "======================================"
echo "All training complete!"
echo "======================================"
echo "Output files generated:"
ls -lh out*.txt

echo ""
echo "Final results summary:"
echo "Q1 (original):"
tail -5 training_q1.log | grep -E 'accuracy|Saving'
echo ""
echo "Q2 (transformed):"
tail -5 training_q2.log | grep 'accuracy'
echo ""
echo "Q3 (augmented on original):"
tail -5 training_q3_original.log | grep 'accuracy'
echo ""
echo "Q3 (augmented on transformed):"
tail -5 training_q3.log | grep 'accuracy'

