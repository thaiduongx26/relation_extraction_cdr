from transformers import ElectraTokenizer
from models import src_path
import os
import numpy as np
from utils.utils import load_pickle, save_pickle

TOKENIZER_PATH = src_path+ '/models_saved/electra_small_tokenizer.pkl'

def gen_tokenizer(save_path, pretrained_model='google/electra-small-discriminator'):
    tokenizer = ElectraTokenizer.from_pretrained(pretrained_model)
    tokenizer.add_tokens(['[E]', '[/E]'])
    save_pickle(save_path, data=tokenizer)

def get_tokenizer():
    if not os.path.isfile(TOKENIZER_PATH):
        gen_tokenizer(TOKENIZER_PATH)
    tokenizer = load_pickle(TOKENIZER_PATH)
    return tokenizer

# def get_tokenizer_with_new_vocab(path_vocab):
    