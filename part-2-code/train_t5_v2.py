import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop - improved version')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    parser.add_argument('--model_name', type=str, default='google-t5/t5-base',
                        help="T5 model name (e.g., google-t5/t5-base)")
    parser.add_argument('--max_len', type=int, default=256,
                        help="Maximum sequence length for tokenization")
    parser.add_argument('--num_beams', type=int, default=8,
                        help="Number of beams for beam search during generation")
    parser.add_argument('--gen_max_len', type=int, default=256,
                        help="Maximum length for generated sequences")

    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=50,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=15,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args


def postprocess_sql(sql_string: str) -> str:
    """
    Light postprocessing heuristics to make model output closer to expected SQL format.

    Improvements applied:
    - Remove extra whitespace
    - Remove spaces before commas
    - Ensure query ends with semicolon
    - Normalize whitespace throughout
    """
    sql_string = sql_string.strip()

    # Remove spaces before commas
    sql_string = sql_string.replace(" ,", ",")

    # Ensure semicolon at end
    if not sql_string.endswith(";"):
        sql_string = sql_string + ";"

    # Normalize whitespace (collapse multiple spaces)
    sql_string = " ".join(sql_string.split())

    return sql_string


def unpack_batch(batch, device):
    """
    Helper to unpack batch tensors and move to device.
    Handles both train/dev batches (5 elements) and test batches (3 elements).
    """
    if len(batch) == 5:
        encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder = batch
        return (
            encoder_input.to(device),
            encoder_mask.to(device),
            decoder_input.to(device),
            decoder_targets.to(device),
            initial_decoder.to(device)
        )
    elif len(batch) == 3:
        encoder_input, encoder_mask, initial_decoder = batch
        return (
            encoder_input.to(device),
            encoder_mask.to(device),
            initial_decoder.to(device)
        )
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} elements")


def train(args, model, train_loader, dev_loader, test_loader, optimizer, scheduler):
    best_f1 = -1
    best_epoch = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Paths for dev evaluation
    gt_sql_path = os.path.join('data/dev.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f'{args.experiment_name}_dev.sql')
    model_record_path = os.path.join('records', f'{args.experiment_name}_dev.pkl')

    # Ensure output directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss:.6f}")

        # Clear GPU cache before evaluation to free memory
        torch.cuda.empty_cache()

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss:.6f}, Record F1: {record_f1:.6f}, Record EM: {record_em:.6f}, SQL EM: {sql_em:.6f}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss': tr_loss,
                'dev/loss': eval_loss,
                'dev/record_f1': record_f1,
                'dev/record_em': record_em,
                'dev/sql_em': sql_em,
                'dev/error_rate': error_rate,
            }
            wandb.log(result_dict, step=epoch)

        # Check for improvement
        improved = record_f1 > best_f1

        if improved:
            best_f1 = record_f1
            best_epoch = epoch
            epochs_since_improvement = 0

            # Save best model
            save_model(checkpoint_dir, model, best=True)
            print(f"New best dev Record F1 = {best_f1:.6f} at epoch {epoch} — saved to {checkpoint_dir}")

            # IMPROVEMENT: Automatically generate test predictions when we get a new best model
            # This ensures we always have test predictions from our best checkpoint
            test_sql_path = os.path.join('results', f'{args.experiment_name}_best_test.sql')
            test_record_path = os.path.join('records', f'{args.experiment_name}_best_test.pkl')
            print(f"Generating test predictions for best model...")
            test_inference(args, model, test_loader, test_sql_path, test_record_path)
            print(f"Test predictions saved to:")
            print(f"  SQL   : {test_sql_path}")
            print(f"  PICKLE: {test_record_path}")
        else:
            epochs_since_improvement += 1

        # Save latest checkpoint regardless
        save_model(checkpoint_dir, model, best=False)

        # Early stopping
        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping triggered after {args.patience_epochs} epochs without improvement.")
            break

    print(f"Training complete. Best dev Record F1 = {best_f1:.6f} at epoch {best_epoch}")


def train_epoch(args, model, train_loader, optimizer, scheduler):
    """
    Single training epoch with improved loss calculation.
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        encoder_input, encoder_mask, decoder_input, decoder_targets, _ = unpack_batch(batch, DEVICE)

        # IMPROVEMENT: Use model's built-in loss calculation with labels parameter
        # This is simpler and handles label smoothing if needed
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=decoder_targets,
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluation loop for development set.

    Computes:
    - Cross-entropy loss on dev set
    - Generates SQL queries using the model
    - Computes metrics (F1, EM, error rate)

    Returns:
        eval_loss: Average cross-entropy loss
        record_f1: F1 score for database records
        record_em: Exact match for database records
        sql_em: Exact match for SQL queries
        error_rate: Fraction of queries with SQL execution errors
    '''
    model.eval()
    tokenizer = dev_loader.dataset.tokenizer

    total_loss = 0
    num_batches = 0
    all_generated_queries = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            encoder_input, encoder_mask, decoder_input, decoder_targets, _ = unpack_batch(batch, DEVICE)

            # Compute loss using built-in loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=decoder_targets,
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            # Generate SQL queries with beam search
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.gen_max_len,
                num_beams=args.num_beams,
                early_stopping=True,
            )

            # Decode and postprocess
            batch_queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # IMPROVEMENT: Apply SQL postprocessing to clean up generated queries
            batch_queries = [postprocess_sql(q) for q in batch_queries]

            all_generated_queries.extend(batch_queries)

    # Calculate average loss
    eval_loss = total_loss / num_batches if num_batches > 0 else 0

    # Save generated queries and compute metrics
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )

    # Calculate error rate
    error_rate = len([msg for msg in error_msgs if msg]) / len(error_msgs) if error_msgs else 0

    return eval_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Inference on test set.

    Generates SQL queries for test examples and saves them.
    '''
    model.eval()
    tokenizer = test_loader.dataset.tokenizer

    all_generated_queries = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test inference"):
            encoder_input, encoder_mask, _ = unpack_batch(batch, DEVICE)

            # Generate SQL queries
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.gen_max_len,
                num_beams=args.num_beams,
                early_stopping=True,
            )

            # Decode and postprocess
            batch_queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_queries = [postprocess_sql(q) for q in batch_queries]

            all_generated_queries.extend(batch_queries)

    # Save generated queries and records
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)
    print(f"Generated {len(all_generated_queries)} test queries")
    print(f"Saved to {model_sql_path} and {model_record_path}")


def main():
    # Get arguments
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Load data
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size,
        model_name=args.model_name, max_len=args.max_len
    )

    # Initialize model
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train - test_loader is now passed so we can generate test predictions automatically
    train(args, model, train_loader, dev_loader, test_loader, optimizer, scheduler)

    # Final evaluation with best checkpoint
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    # Evaluate on dev set
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join('data/dev.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f'{args.experiment_name}_final_dev.sql')
    model_record_path = os.path.join('records', f'{args.experiment_name}_final_dev.pkl')

    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    print(f"\n=== Final Dev Set Results ===")
    print(f"Loss: {dev_loss:.6f}")
    print(f"Record F1: {dev_record_f1:.6f}")
    print(f"Record EM: {dev_record_em:.6f}")
    print(f"SQL EM: {dev_sql_em:.6f}")
    print(f"Error rate: {dev_error_rate*100:.2f}%")

    # Generate final test predictions
    model_sql_path = os.path.join('results', f'{args.experiment_name}_final_test.sql')
    model_record_path = os.path.join('records', f'{args.experiment_name}_final_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
