import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Combined transformation with:
    # 1. Aggressive synonym replacement (40% probability)
    # 2. Character-level typos (5% probability)  
    # 3. Word deletion for non-essential words (3% probability)
    # This simulates real user input with varied vocabulary, typos, and casual writing
    
    import string
    
    text = example["text"]
    words = word_tokenize(text)
    new_words = []
    
    # Define non-essential words that can be deleted
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been'}
    
    for word in words:
        # Skip if word is deleted (3% chance for stopwords)
        if word.lower() in stopwords and random.random() < 0.03:
            continue
            
        current_word = word
        
        # Apply synonym replacement (40% probability)
        if random.random() < 0.40:
            synsets = wordnet.synsets(word)
            if synsets:
                lemmas = synsets[0].lemmas()
                if len(lemmas) > 1:
                    synonyms = [lemma.name() for lemma in lemmas 
                              if lemma.name().lower() != word.lower()]
                    if synonyms:
                        current_word = random.choice(synonyms).replace('_', ' ')
        
        # Apply character-level typos (5% probability) 
        if len(current_word) > 3 and random.random() < 0.05:
            word_list = list(current_word)
            typo_type = random.choice(['swap', 'delete'])
            
            if typo_type == 'swap' and len(word_list) > 1:
                # Swap two adjacent characters
                pos = random.randint(0, len(word_list) - 2)
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
            elif typo_type == 'delete':
                # Delete a random character
                pos = random.randint(0, len(word_list) - 1)
                del word_list[pos]
            
            current_word = ''.join(word_list)
        
        new_words.append(current_word)
    
    # Detokenize back to text
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
