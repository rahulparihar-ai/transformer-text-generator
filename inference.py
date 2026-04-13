import os
import torch
import pickle
import torch.nn.functional as F

from data_pipeline import WordTokenizer
from model import TransformerModel
from config import CONFIG

def load_environment():
    if not os.path.exists("tokenizer.pkl"):
        raise FileNotFoundError("Error: 'tokenizer.pkl' not found.")
        
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    model = TransformerModel()

    if os.path.exists("best_model.pt"):
        state_dict = torch.load("best_model.pt", map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("Warning: 'best_model.pt' not found locally!")
        
    model.eval()
    return tokenizer, model

def generate_text(model, tokenizer, text: str, max_new_tokens=20, temperature=0.8, top_k=5):
    indices = tokenizer.encode(text)
    inputs = torch.tensor([indices], dtype=torch.long)
    
    for _ in range(max_new_tokens):
        if inputs.size(1) > CONFIG["max_seq_len"]:
            model_inputs = inputs[:, -CONFIG["max_seq_len"]:]
        else:
            model_inputs = inputs
            
        with torch.no_grad():
            output = model(model_inputs)
            
        logits = output[:, -1, :] / temperature
        
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, top_k)
        
        top_probs = top_probs.squeeze()
        top_indices = top_indices.squeeze()
        
        top_probs = top_probs / top_probs.sum()
        
        sampled_index = torch.multinomial(top_probs, 1)
        
        next_token = top_indices[sampled_index].view(1, 1)
        
        inputs = torch.cat((inputs, next_token), dim=1)
        
    generated_indices = inputs[0].tolist()
    
    if hasattr(tokenizer, 'itos'):
        idx2word = tokenizer.itos
        unk_token = getattr(tokenizer, 'unk_token', '<UNK>')
    else:
        idx2word = {v:k for k,v in tokenizer.word2idx.items()}
        unk_token = '<UNK>'
        
    generated_words = [idx2word.get(idx, unk_token) for idx in generated_indices]
    final_sentence = " ".join(generated_words)
    
    print(final_sentence)
    
    return final_sentence

if __name__ == "__main__":
    tokenizer, model = load_environment()
    sample_sentence = "to be or not to"
    generate_text(model, tokenizer, text=sample_sentence)
