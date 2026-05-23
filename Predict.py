import torch
import torch.nn as nn

from Tokenizer import TokenizerModule, create_positional_encoding, get_final_embeddings
from Encoder import TransformerBlock
from Decoder import DecoderBlock


if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer_module = TokenizerModule()
    tokenizer_module.train_tokenizer()
    
    # Encode text
    output = tokenizer_module.encode("Hello, y'all! How are you 😁 ?")
    token_ids = torch.tensor([output.ids])
    vocab_size = tokenizer_module.get_vocab_size()
    
    print("Token IDs:", token_ids)
    print("Vocab Size:", vocab_size)
    
    # Configuration
    embedding_dim = 8
    forward_expansion = 4
    dropout = 0.1
    num_heads = 2
    
    # Get final embeddings with positional encoding
    final_input_embeddings, _ = get_final_embeddings(token_ids, embedding_dim)
    
    print("Final Input Embeddings Shape:", final_input_embeddings.shape)
    
    # Create encoder and decoder blocks
    encoder_block = TransformerBlock(embedding_dim, num_heads, forward_expansion, dropout)
    decoder_block = DecoderBlock(embedding_dim, num_heads, forward_expansion, dropout)
    
    # Forward pass through encoder
    enc_out = encoder_block(final_input_embeddings, final_input_embeddings, final_input_embeddings)
    
    # Forward pass through decoder
    dec_out = decoder_block(final_input_embeddings, enc_out, enc_out, src_mask=None, trg_mask=None)
    
    # Generate predictions
    fc_out = nn.Linear(embedding_dim, vocab_size)
    logits = fc_out(dec_out)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_token = torch.argmax(probabilities, dim=-1)
    
    print("Predicted tokens from full architecture:", predicted_token)