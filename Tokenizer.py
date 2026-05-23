import tempfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer as BPETokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class TokenizerModule:
    def __init__(self):
        self.tokenizer = BPETokenizer(BPE(unk_token="[UNK]"))

        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
    
    def train_tokenizer(self, text_data=None):
        if text_data is None:
            text_data = """Humpty Dumpty sat on a wall,
               Humpty Dumpty had a great fall;
               All the king's horses and all
               the king's men
              Couldn't put Humpty together
              again."""
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            encoding='utf-8'
        )
        temp_file.write(text_data)
        temp_file.close()
        
        self.tokenizer.train([temp_file.name], self.trainer)
        os.remove(temp_file.name)
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()


def create_embeddings(token_ids, embedding_dim):
    """Create token embeddings from token IDs"""
    vocab_size = max(token_ids.max().item() + 1, 256)
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    token_embeddings = embedding_layer(token_ids)
    return token_embeddings, vocab_size


def create_positional_encoding(seq_length, embedding_dim):
    """Create positional encoding for transformer"""
    positional_encoding = torch.zeros(seq_length, embedding_dim)
    positions = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float()
                        * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
    
    positional_encoding[:, 0::2] = torch.sin(positions * div_term)
    positional_encoding[:, 1::2] = torch.cos(positions * div_term)
    
    return positional_encoding


def get_final_embeddings(token_ids, embedding_dim):
    """Get final embeddings with positional encoding"""
    token_embeddings, vocab_size = create_embeddings(token_ids, embedding_dim)
    positional_encoding = create_positional_encoding(token_ids.size(1), embedding_dim)
    final_input_embeddings = token_embeddings + positional_encoding
    return final_input_embeddings, vocab_size