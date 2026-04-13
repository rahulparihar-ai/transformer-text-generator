import os
import json
import torch
import pickle
import torch.nn.functional as F

from data_pipeline import WordTokenizer
from transformer_model import TransformerModel

def load_environment():
    """
    Safely integrates loading architecture dependencies dynamically tracking configuration.
    """
    if not os.path.exists("tokenizer.pkl"):
        raise FileNotFoundError("Error: 'tokenizer.pkl' context missing. Generate prior mappings fully.")
        
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    # ---------------------------------------------------------------------
    # REQUIREMENT: Automatically detect correct vocab size directly tracking mappings
    # ---------------------------------------------------------------------
    vocab_size = len(tokenizer.word2idx) if hasattr(tokenizer, 'word2idx') else len(tokenizer.stoi)
    print(f"Detected Dynamic Vocab Boundaries initialized at mapping sizes: {vocab_size}")

    # Force initialize parameters mapping natively exactly to token dimension limits 
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_length=32,
        dropout=0.1
    )
    
    if os.path.exists("best_model.pt"):
        # Safely extract explicitly boundaries formats tracking sequences backwards accurately
        state_dict = torch.load("best_model.pt", map_location=torch.device("cpu"), weights_only=True)
        
        # Failsafe logic structurally resizing parameters backwards explicitly matching old trained models natively
        checkpoint_vocab_dim = state_dict['token_embedding.weight'].shape[0]
        if checkpoint_vocab_dim != vocab_size:
            print(f"Alert bounds mismatch tracking sizes: Map checkpoint limits ({checkpoint_vocab_dim}). Overriding outwards constraints natively.")
            vocab_size = checkpoint_vocab_dim
            model = TransformerModel(
                vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2, 
                d_ff=512, max_seq_length=32, dropout=0.1
            )
            
        model.load_state_dict(state_dict)
    else:
        print("Warning: 'best_model.pt' not found locally! Expect random initialized mapping probabilities.")
        
    model.eval()
    return tokenizer, model

def preprocess_text(text: str, tokenizer: WordTokenizer, max_seq_length: int = 32):
    """
    Cleans logic directly fulfilling token index constraints, managing dynamic padding requirements natively.
    """
    indices = tokenizer.encode(text)
    
    if len(indices) > max_seq_length:
        indices = indices[-max_seq_length:]
    elif len(indices) < max_seq_length:
        pad_id = tokenizer.stoi[tokenizer.pad_token] if hasattr(tokenizer, 'stoi') else tokenizer.word2idx.get('<PAD>', 0)
        indices = ([pad_id] * (max_seq_length - len(indices))) + indices
        
    return torch.tensor([indices], dtype=torch.long)

def predict_next_word(model, tokenizer, text: str, top_k=5):
    """
    Outputs cleanly bounded probability indexes mapped backwards intelligently.
    """
    inputs = preprocess_text(text, tokenizer)
    
    with torch.no_grad():
        output = model(inputs)
        
    last_token_logits = output[0, -1, :]
    probabilities = F.softmax(last_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    print(f"\nInput: \"{text}\"")
    print("Output:")
    
    for rank in range(top_k):
        prob_val = top_probs[rank].item()
        mapped_token_id = top_indices[rank].item()
        
        if hasattr(tokenizer, 'itos'):
            predicted_word = tokenizer.itos.get(mapped_token_id, tokenizer.unk_token)
        else:
            idx2word = {v:k for k,v in tokenizer.word2idx.items()}
            predicted_word = idx2word.get(mapped_token_id, '<UNK>')
            
        print(f"{rank + 1}. {predicted_word} - {prob_val:.4f}")

if __name__ == "__main__":
    print("Starting Inference Model Pipeline logic...")
    try:
        tokenizer, model = load_environment()
        sample_sentence = "to be or not to"
        predict_next_word(model, tokenizer, text=sample_sentence)
    except Exception as e:
        print(f"Execution Error properly mapped: {e}")
