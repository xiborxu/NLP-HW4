import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    if args.use_wandb:
        wandb.init(
            project="hw4-part2-t5",
            name=args.experiment_name,
            config=vars(args)
        )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model or training a T5 model from scratch.
    '''
    # Use t5-base as default for better performance
    model_name = getattr(args, 'model_name', 'google-t5/t5-base')

    if args.finetune:
        # Load the pre-trained model for fine-tuning
        print(f"Loading pre-trained model for fine-tuning: {model_name}")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        # Load the configuration and initialize a new model from scratch
        print(f"Initializing model from scratch with config: {model_name}")
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)

    # Move the model to the correct device
    model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    '''
    Save model checkpoint using HuggingFace's save_pretrained method.
    '''
    # Determine the save path based on whether it's the 'best' model
    if best:
        save_path = os.path.join(checkpoint_dir, 'best_model')
    else:
        # Save as a generic 'latest_checkpoint'
        save_path = os.path.join(checkpoint_dir, 'latest_checkpoint')

    # Create the directory if it doesn't exist
    mkdir(save_path)

    # Use Hugging Face's save_pretrained to save model and config
    print(f"Saving model checkpoint to {save_path}")
    model.save_pretrained(save_path)

def load_model_from_checkpoint(args, best):
    '''
    Load model from a checkpoint using HuggingFace's from_pretrained method.
    '''
    # Determine the load path
    if best:
        load_path = os.path.join(args.checkpoint_dir, 'best_model')
    else:
        load_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint')

    # Check if the path exists
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint directory not found at: {load_path}")

    # Load the model using Hugging Face's from_pretrained
    print(f"Loading model checkpoint from {load_path}")
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(load_path)

    # Move the model to the correct device
    model.to(DEVICE)
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer

def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
