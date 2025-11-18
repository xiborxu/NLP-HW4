import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Initialize T5Dataset for text-to-SQL task.

        Args:
            data_folder: Path to the data directory
            split: One of 'train', 'dev', or 'test'
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

        # Load and process data
        self.encoder_inputs, self.decoder_targets, self.nl_queries = self.process_data(
            data_folder, split, self.tokenizer
        )

        # Decoder BOS token - use extra_id_0 as suggested
        self.decoder_bos_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')

    def process_data(self, data_folder, split, tokenizer):
        '''
        Load and tokenize the data.

        Returns:
            encoder_inputs: List of tokenized natural language queries
            decoder_targets: List of tokenized SQL queries (None for test set)
            nl_queries: List of original natural language queries
        '''
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]

        # Tokenize encoder inputs (natural language)
        encoder_inputs = []
        for query in nl_queries:
            # T5 expects prefix for task - using "translate English to SQL: " prefix
            prefixed_query = f"translate English to SQL: {query}"
            tokenized = tokenizer(prefixed_query, add_special_tokens=True, return_tensors='pt')
            encoder_inputs.append(tokenized['input_ids'].squeeze(0))

        # Load and tokenize SQL queries (if not test set)
        decoder_targets = None
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            with open(sql_path, 'r') as f:
                sql_queries = [line.strip() for line in f.readlines()]

            decoder_targets = []
            for sql in sql_queries:
                tokenized = tokenizer(sql, add_special_tokens=True, return_tensors='pt')
                decoder_targets.append(tokenized['input_ids'].squeeze(0))

        return encoder_inputs, decoder_targets, nl_queries

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        '''
        Get a single example.

        Returns:
            For train/dev: (encoder_input, decoder_target, nl_query)
            For test: (encoder_input, nl_query)
        '''
        encoder_input = self.encoder_inputs[idx]

        if self.split == 'test':
            return encoder_input, self.nl_queries[idx]
        else:
            decoder_target = self.decoder_targets[idx]
            return encoder_input, decoder_target, self.nl_queries[idx]

def normal_collate_fn(batch):
    '''
    Collation function for training and dev sets with dynamic padding.

    Args:
        batch: List of tuples (encoder_input, decoder_target, nl_query)

    Returns:
        encoder_ids: Padded encoder input ids [B, T]
        encoder_mask: Attention mask for encoder [B, T]
        decoder_inputs: Padded decoder input ids [B, T']
        decoder_targets: Padded decoder target ids [B, T']
        initial_decoder_inputs: Initial decoder token for generation [B, 1]
    '''
    encoder_inputs = []
    decoder_targets_list = []

    # Unpack batch
    for item in batch:
        encoder_inputs.append(item[0])
        decoder_targets_list.append(item[1])

    # Pad encoder inputs
    encoder_ids = torch.nn.utils.rnn.pad_sequence(
        encoder_inputs, batch_first=True, padding_value=PAD_IDX
    )

    # Create encoder attention mask (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Pad decoder targets
    decoder_targets = torch.nn.utils.rnn.pad_sequence(
        decoder_targets_list, batch_first=True, padding_value=PAD_IDX
    )

    # Create decoder inputs by shifting targets right and prepending BOS token
    # Use extra_id_0 as BOS token
    bos_token_id = 32099  # <extra_id_0> token id
    batch_size = decoder_targets.size(0)

    # Decoder inputs = [BOS, target[:-1]]
    bos_tokens = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)
    decoder_inputs = torch.cat([bos_tokens, decoder_targets[:, :-1]], dim=1)

    # Initial decoder inputs for generation (just the BOS token)
    initial_decoder_inputs = bos_tokens

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function for test set with dynamic padding.

    Args:
        batch: List of tuples (encoder_input, nl_query)

    Returns:
        encoder_ids: Padded encoder input ids [B, T]
        encoder_mask: Attention mask for encoder [B, T]
        initial_decoder_inputs: Initial decoder token for generation [B, 1]
    '''
    encoder_inputs = []

    # Unpack batch
    for item in batch:
        encoder_inputs.append(item[0])

    # Pad encoder inputs
    encoder_ids = torch.nn.utils.rnn.pad_sequence(
        encoder_inputs, batch_first=True, padding_value=PAD_IDX
    )

    # Create encoder attention mask
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Initial decoder inputs (BOS token)
    bos_token_id = 32099  # <extra_id_0> token id
    batch_size = encoder_ids.size(0)
    initial_decoder_inputs = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x
