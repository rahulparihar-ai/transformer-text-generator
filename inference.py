import os
import torch
import torch.nn.functional as F
import pickle

from data_pipeline import WordTokenizer # Ensure pickle can load the tokenizer
from model.transformer_model import TransformerModel
from config import CONFIG
import traceback

def load_environment():
    try:
        if not os.path.exists("data/tokenizer.pkl"):
            return None, None
            
        with open("data/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
            
        model = TransformerModel()

        if os.path.exists("model/best_model.pt"):
            state_dict = torch.load("model/best_model.pt", map_location=torch.device("cpu"), weights_only=True)
            model.load_state_dict(state_dict)
            
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading environment: {e}")
        return None, None

def generate_text(model, tokenizer, text: str, max_new_tokens=20, temperature=0.8, top_k=5, repetition_penalty=1.2):
    try:
        if model is None or tokenizer is None:
            return "Error: Model or tokenizer not loaded."
            
        text = str(text).strip()
        if not text:
            return "Error: Empty input provided."

        vocab_size = getattr(model, 'fc_out', None)
        if vocab_size is not None:
             vocab_size = vocab_size.out_features
        else:
             vocab_size = CONFIG.get("vocab_size", 5000)

        # Tokenize safely
        try:
            if hasattr(tokenizer, 'encode'):
                indices = tokenizer.encode(text)
            else:
                return "Error: Tokenizer missing 'encode' method."
        except Exception:
            return "Error during tokenization."

        if not indices:
            return "Error: Tokenization resulted in empty sequence."

        device = torch.device("cpu")
        inputs = torch.tensor([indices], dtype=torch.long, device=device)
        max_seq_len = CONFIG.get("max_seq_len", 64)
        
        for _ in range(max_new_tokens):
            try:
                # Truncate sequence safely
                if inputs.size(1) > max_seq_len:
                    model_inputs = inputs[:, -max_seq_len:]
                else:
                    model_inputs = inputs
                    
                with torch.no_grad():
                    output = model(model_inputs)
                    
                # Shape checking
                if output.dim() != 3 or output.size(1) == 0 or output.size(2) == 0:
                    break
                    
                logits = output[:, -1, :] / max(temperature, 1e-5)
                
                # Apply repetition penalty securely
                try:
                    generated_list = set(inputs[0].tolist())
                    for token_idx in generated_list:
                        if 0 <= token_idx < logits.size(-1):
                            if logits[0, token_idx] < 0:
                                logits[0, token_idx] *= repetition_penalty
                            else:
                                logits[0, token_idx] /= repetition_penalty
                except Exception:
                    pass # Ignore penalty failures
                    
                probs = torch.softmax(logits, dim=-1)
                
                # Secure NaNs/Infs
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)
                    
                probs = probs.squeeze()
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)
                    
                # Fallback if probability shape is broken
                if probs.shape[0] <= 1:
                    next_token = torch.tensor([[0]], dtype=torch.long, device=device)
                else:
                    actual_k = min(top_k, probs.shape[0])
                    if actual_k > 0:
                        top_probs, top_indices = torch.topk(probs, actual_k)
                        
                        # Normalize safely
                        prob_sum = top_probs.sum()
                        if prob_sum > 0 and not torch.isnan(prob_sum) and not torch.isinf(prob_sum):
                            top_probs = top_probs / prob_sum
                        else:
                            top_probs = torch.ones_like(top_probs) / actual_k
                            
                        try:
                            sampled_index = torch.multinomial(top_probs, 1)
                            next_token = top_indices[sampled_index].view(1, 1)
                        except Exception:
                            # Fallback if sampling fails
                            next_token = torch.tensor([[0]], dtype=torch.long, device=device)
                    else:
                        next_token = torch.tensor([[0]], dtype=torch.long, device=device)
                        
                # Clamp to vocab limits globally to prevent index OOB
                next_token = next_token.clamp(0, vocab_size - 1)
                
                # Concat the new token
                inputs = torch.cat((inputs, next_token), dim=1)
                
            except Exception as inner_e:
                print(f"Warning: iteration failed, stopping early. {inner_e}")
                break
                
        # Secure decoding
        generated_indices = inputs[0].tolist()
        
        try:
            if hasattr(tokenizer, 'decode'):
                final_sentence = tokenizer.decode(generated_indices)
            else:
                if hasattr(tokenizer, 'itos'):
                    idx2word = tokenizer.itos
                    unk_token = getattr(tokenizer, 'unk_token', '<UNK>')
                elif hasattr(tokenizer, 'word2idx'):
                    idx2word = {v:k for k,v in tokenizer.word2idx.items()}
                    unk_token = getattr(tokenizer, 'unk_token', '<UNK>')
                else:
                    return "Error: Tokenizer missing decoding data structures."
                    
                generated_words = [idx2word.get(idx, unk_token) for idx in generated_indices]
                final_sentence = " ".join(generated_words)
        except Exception:
            return "Error decoding generated sequence."
            
        return final_sentence
    except Exception as e:
        # Final catch-all for absolute safety
        return f"Safely recovered from exception. Default text. (Error: {e})"

if __name__ == "__main__":
    tokenizer, model = load_environment()
    sample_sentence = "to be or not to"
    result = generate_text(model, tokenizer, text=sample_sentence)
    print(result)
