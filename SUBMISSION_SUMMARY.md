# HW4 Submission Summary

**Student:** Xibo Xu
**GitHub Repository:** https://github.com/xiborxu/NLP-HW4
**Google Cloud Project:** silent-vent-478518-e8
**VM Instance:** hw4-gpu-vm (zone: us-west4-a, GPU: NVIDIA L4)

---

## Part 1: BERT Fine-tuning for Sentiment Classification ✅ COMPLETE

### Results

#### Q1: Fine-tune BERT on IMDB Dataset
- **Accuracy on Original Test Set:** 92.952%
- **Requirement:** ≥91% ✅
- **Output File:** `part-1-code/out_original.txt`

#### Q2: Out-of-Distribution Evaluation
- **Accuracy on Transformed Test Set:** 88.332%
- **Accuracy Drop:** 4.62 percentage points
- **Requirement:** >4 point drop for full credit ✅
- **Output File:** `part-1-code/out_transformed.txt`

**Transformation Strategy:**
- 40% probability of synonym replacement using WordNet
- 5% probability of character-level typos (swap/delete)
- 3% probability of stopword deletion
- Combined approach creates substantial semantic/syntactic variation

#### Q3: Data Augmentation
- **Augmented Model on Original Test:** 92.776%
- **Augmented Model on Transformed Test:** 92.008%
- **Output Files:**
  - `part-1-code/out_augmented_original.txt`
  - `part-1-code/out_augmented_transformed.txt`

**Augmentation Strategy:**
- Applied same transformation to training data
- Improved robustness to distribution shift

### Implementation Details
- **Model:** bert-base-uncased
- **Training:** Adam optimizer, 3 epochs
- **Data:** IMDB 50k reviews (25k train, 25k test)
- **Key Code:** `part-1-code/utils.py` (custom_transform function)

---

## Part 2: T5 Fine-tuning for Text-to-SQL ⚠️ IN PROGRESS

### Implementation Status ✅ COMPLETE

All required components have been fully implemented:

1. **T5Dataset Class** (`part-2-code/load_data.py`)
   - Data loading and tokenization using T5TokenizerFast
   - Task prefix: "translate English to SQL:"
   - Decoder BOS token: `<extra_id_0>`
   - Different behavior for train/dev vs test sets

2. **Collation Functions** (`part-2-code/load_data.py`)
   - `normal_collate_fn`: Dynamic padding for train/dev with targets
   - `test_collate_fn`: Dynamic padding for test inference
   - Proper decoder input shifting (BOS prepending)

3. **Model Utilities** (`part-2-code/t5_utils.py`)
   - `initialize_model()`: T5-small finetuning or from-scratch training
   - `save_model()`: Checkpoint saving (best and last)
   - `load_model_from_checkpoint()`: Model restoration
   - `setup_wandb()`: Experiment tracking support

4. **Evaluation Functions** (`part-2-code/train_t5.py`)
   - `eval_epoch()`: Loss computation + SQL generation + metrics
   - `test_inference()`: Test set SQL generation
   - Beam search generation (num_beams=4)
   - Integration with provided `compute_metrics()` and `save_queries_and_records()`

### Training Status

**Current Progress:** Epoch 7/15

| Metric | Epoch 0 | Epoch 7 | Target |
|--------|---------|---------|--------|
| Train Loss | 2.89 | 0.107 | N/A |
| Dev Loss | 0.571 | 0.069 | N/A |
| Record F1 | 11.8% | 24.9% | **≥65%** |
| Record EM | 11.8% | 21.2% | N/A |
| SQL EM | 0.0% | 1.1% | N/A |
| SQL Error Rate | 100% | 100% | 0% |

**Training Configuration:**
- Model: google-t5/t5-small (finetuning)
- Learning Rate: 0.0001
- Weight Decay: 0.01
- Batch Size: 16
- Scheduler: Cosine with 1 epoch warmup
- Early Stopping: 3 epochs patience
- Max Epochs: 15

**Observations:**
- Loss is decreasing steadily (train: 2.89→0.107, dev: 0.571→0.069)
- F1 improving but slowly (11.8%→24.9% over 7 epochs)
- 100% SQL error rate indicates generated queries are malformed
- Model is learning but not fast enough to reach 65% F1 target

**Potential Issues:**
1. Learning rate may be too low (0.0001)
2. Model may need longer training or different hyperparameters
3. Generation parameters may need tuning
4. Task may benefit from different prefix or prompt engineering

### Files Generated

**Dev Set Results:**
- `part-2-code/results/t5_ft_ft_experiment_dev.sql` (466 generated queries)
- `part-2-code/records/t5_ft_ft_experiment_dev.pkl` (database records)

**Test Set Results:**
- Will be generated after training completes or reaches acceptable F1
- Target files: `t5_ft_test.sql`, `t5_ft_test.pkl`

**Model Checkpoints:**
- `checkpoints/ft_experiments/ft_lr1e4_wd001/best_model.pt` (242 MB)
- `checkpoints/ft_experiments/ft_lr1e4_wd001/last_model.pt` (242 MB)

---

## Repository Structure

```
NLP-HW4/
├── .gitignore
├── part-1-code/
│   ├── main.py                          # BERT training script
│   ├── utils.py                         # Transformations (custom_transform)
│   ├── requirements.txt
│   ├── run_all_questions.sh             # Automated Q1→Q2→Q3 execution
│   ├── out/                             # Q1 model checkpoints
│   ├── out_augmented/                   # Q3 model checkpoints
│   ├── out_original.txt                 # Q1 submission ✅
│   ├── out_transformed.txt              # Q2 submission ✅
│   ├── out_augmented_original.txt       # Q3 submission ✅
│   └── out_augmented_transformed.txt    # Q3 submission ✅
└── part-2-code/
    ├── train_t5.py                      # T5 training loop (implemented)
    ├── load_data.py                     # Dataset & collation (implemented)
    ├── t5_utils.py                      # Model utils (implemented)
    ├── utils.py                         # Metrics (provided)
    ├── requirements.txt
    ├── data/                            # Flight database
    │   ├── train.nl, train.sql (4225 examples)
    │   ├── dev.nl, dev.sql (466 examples)
    │   ├── test.nl (431 examples)
    │   └── flight_database.db
    ├── checkpoints/
    │   └── ft_experiments/ft_lr1e4_wd001/
    │       ├── best_model.pt            # Best checkpoint (242MB)
    │       └── last_model.pt            # Latest checkpoint (242MB)
    ├── results/
    │   └── t5_ft_ft_experiment_dev.sql  # Dev predictions
    └── records/
        ├── ground_truth_dev.pkl
        └── t5_ft_ft_experiment_dev.pkl  # Dev database records
```

---

## Submission Checklist

### Part 1 (Complete ✅)
- [✅] `out_original.txt` - Q1 results (92.952%)
- [✅] `out_transformed.txt` - Q2 results (88.332%, 4.62 drop)
- [✅] `out_augmented_original.txt` - Q3 on original test
- [✅] `out_augmented_transformed.txt` - Q3 on transformed test
- [✅] Written explanation for Q2 transformation strategy
- [✅] Written explanation for Q3 augmentation strategy

### Part 2 (In Progress ⚠️)
- [✅] All code implementations complete
- [✅] T5Dataset and collation functions
- [✅] Model initialization and checkpointing
- [✅] Evaluation and test inference functions
- [⚠️] Training achieving ≥65% F1 (currently 24.9%)
- [⚠️] `t5_ft_test.sql` - Test predictions (pending)
- [⚠️] `t5_ft_test.pkl` - Test database records (pending)

### Repository & Documentation
- [✅] GitHub repository: https://github.com/xiborxu/NLP-HW4
- [✅] All code pushed and synced
- [✅] README/summary documentation
- [✅] Clean code structure with .gitignore

---

## How to Reproduce

### Part 1 (on GCP VM)
```bash
# Setup environment
conda create -n hw4-part-1-nlp python=3.9 -y
conda activate hw4-part-1-nlp
cd ~/NLP-HW4/part-1-code
pip install -r requirements.txt

# Run all questions sequentially
./run_all_questions.sh
```

### Part 2 (on GCP VM with GPU)
```bash
# Training (currently running)
conda activate hw4-part-1-nlp
cd ~/NLP-HW4/part-2-code
python3 train_t5.py --finetune \
    --learning_rate 0.0001 \
    --weight_decay 0.01 \
    --max_n_epochs 15 \
    --patience_epochs 3 \
    --num_warmup_epochs 1 \
    --batch_size 16 \
    --test_batch_size 16 \
    --experiment_name ft_lr1e4_wd001

# Monitor training
tail -f training_t5.log
```

---

## Known Issues & Future Work

### Part 2 Training Challenges
1. **Low F1 Score:** Current 24.9% vs target 65%
   - Possible solutions: Increase learning rate, longer training, different architecture

2. **100% SQL Error Rate:** All generated queries have syntax errors
   - Indicates model hasn't learned proper SQL grammar yet
   - May need different training approach or prompt engineering

3. **Slow Progress:** F1 improving slowly over epochs
   - May benefit from higher learning rate (try 5e-4 or 1e-3)
   - Or different optimizer/scheduler configuration

### Recommendations for Improvement
- Try higher learning rates (5e-4, 1e-3)
- Experiment with different batch sizes (8, 32)
- Adjust generation parameters (beam size, length penalty)
- Consider using T5-base instead of T5-small
- Add SQL grammar constraints during generation

---

## Contact & Resources

- **GitHub:** https://github.com/xiborxu/NLP-HW4
- **GCP Project:** silent-vent-478518-e8
- **Training Logs:** Available in `part-2-code/training_t5.log` on VM

**Last Updated:** November 18, 2025
