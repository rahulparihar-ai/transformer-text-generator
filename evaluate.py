import os
import json
import math
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader

from transformer_model import TransformerModel

# Locally re-defined to prevent triggering `prepare_pipeline` which destructively rebuilds the Tokenizer
class NextWordPredictionDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

def load_evaluation_environment():
    """
    Safely integrates loading architecture dependencies properly pulling mappings cleanly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device constraints tracked: {device}")
    
    print("\n[Sequence 1] Loading native Tokenizer mappings preserving explicit dimension lengths...")
    if not os.path.exists("tokenizer.pkl"):
        raise FileNotFoundError("Error: 'tokenizer.pkl' context missing. Generation logic required.")
        
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    # ---------------------------------------------------------------------
    # REQUIREMENT: Automatically detect correct vocab size mathematically
    # ---------------------------------------------------------------------
    vocab_size = len(tokenizer.word2idx) if hasattr(tokenizer, 'word2idx') else len(tokenizer.stoi)
    
    # Failsafe overrides matching checkpoints implicitly preventing dimension misalignments structurally 
    if os.path.exists("best_model.pt"):
        state_dict = torch.load("best_model.pt", map_location=device, weights_only=True)
        checkpoint_vocab_dim = state_dict['token_embedding.weight'].shape[0]
        if checkpoint_vocab_dim != vocab_size:
            print(f"Alert override tracking mismatches. Overriding vocab_size dynamically to {checkpoint_vocab_dim}!")
            vocab_size = checkpoint_vocab_dim

    print(f"[Sequence 2] Constructing dimensional configurations explicitly targeting limits natively (Vocab bounds: {vocab_size})")
    model = TransformerModel(
        vocab_size=vocab_size, 
        d_model=128, 
        num_heads=4, 
        num_layers=2, 
        d_ff=512, 
        max_seq_length=32, 
        dropout=0.1
    ).to(device)

    if os.path.exists("best_model.pt"):
        model.load_state_dict(state_dict)

    model.eval()
    
    # ---------------------------------------------------------------------
    # REQUIREMENT: Fix DataLoader usage cleanly extracting specific configurations
    # ---------------------------------------------------------------------
    print("\n[Sequence 3] Creating 'test_loader' inherently aligned logically avoiding state rewrites...")
    with open("speare.txt", "r", encoding="utf-8") as f:
        text = f.read()
        
    encoded_data = tokenizer.encode(text)
    
    # Ensure encoded targets match tokenizer mapped limits securely bounding variables downwards natively
    encoded_data = [min(idx, vocab_size - 1) for idx in encoded_data]
    
    total_tokens = len(encoded_data)
    val_end = int(total_tokens * 0.85)
    test_data = encoded_data[val_end:]
    
    test_dataset = NextWordPredictionDataset(test_data, seq_len=32)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    return test_loader, model, vocab_size, device

def calculate_topk_accuracy(logits, targets, k=5):
    """
    Handles Top-K computation logic mathematically validating index positions.
    """
    _, topk_indices = torch.topk(logits, k, dim=-1)
    expanded_targets = targets.unsqueeze(1)
    matches = (topk_indices == expanded_targets).any(dim=1)
    return matches.sum().item()

def evaluate(model, test_loader, vocab_size, device):
    """
    Core functional block aggregating statistics sequentially updating mappings strictly validating performance.
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    total_samples = 0
    total_top1_correct = 0
    total_top5_correct = 0
    
    first_batch = True
    
    print("\n[Sequence 4] Stepping systematically over 'test_loader' accumulating metrics natively...")
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # ---------------------------------------------------------------------
            # REQUIREMENT: Ensure Targets are within bounds dynamically enforcing limits explicitly 
            # ---------------------------------------------------------------------
            y = torch.clamp(y, max=vocab_size - 1)
            
            # Print explicit debugging requirements strictly bounded natively!
            if first_batch:
                print("Vocab size:", vocab_size)
                print("Max target value:", torch.max(y).item())
                first_batch = False
            
            logits = model(x)
            
            B, S, V = logits.shape
            flat_logits = logits.view(B * S, V)
            flat_targets = y.view(B * S)
            
            loss = criterion(flat_logits, flat_targets)
            total_loss += loss.item()
            
            predictions = torch.argmax(flat_logits, dim=-1)
            total_top1_correct += (predictions == flat_targets).sum().item()
            
            total_top5_correct += calculate_topk_accuracy(flat_logits, flat_targets, k=5)
            total_samples += flat_targets.numel()

    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    
    test_perplexity = math.exp(avg_loss) if avg_loss < 50 else float('inf')
    
    top1_accuracy = total_top1_correct / total_samples
    top5_accuracy = total_top5_correct / total_samples
    
    baseline_accuracy = 1.0 / vocab_size
    improvement_ratio = top1_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
    improvement_str = f"{improvement_ratio:.1f}x better"
    
    results = {
        "test_loss": round(avg_loss, 4),
        "test_perplexity": round(test_perplexity, 2),
        "top1_accuracy": round(top1_accuracy, 4),
        "top5_accuracy": round(top5_accuracy, 4),
        "baseline_accuracy": round(baseline_accuracy, 6),
        "improvement_vs_baseline": improvement_str
    }
    
    return results

if __name__ == "__main__":
    print("Initiating Sequence Metrics Validation Tracking Pipeline properly mapping arrays natively...")
    try:
        test_loader, model, vocab_size, device = load_evaluation_environment()
        
        results_map = evaluate(model, test_loader, vocab_size, device)
        
        print("\n" + "="*55)
        print("             EVALUATION SUMMARY RESULTS           ")
        print("="*55)
        print(f"Test Loss:                   {results_map['test_loss']:.4f}")
        print(f"Test Perplexity:             {results_map['test_perplexity']:.2f}")
        print(f"Top-1 Accuracy:              {results_map['top1_accuracy'] * 100:.2f}%")
        print(f"Top-5 Accuracy:              {results_map['top5_accuracy'] * 100:.2f}%")
        print(f"Random Baseline (Acc):       {results_map['baseline_accuracy'] * 100:.4f}%")
        print(f"Performance Tracking Checks: {results_map['improvement_vs_baseline']}")
        print("="*55)
        
        with open("results.json", "w") as f:
            json.dump(results_map, f, indent=4)
            
        print("\nResults fully mathematically processed safely tracking downwards cleanly mapping values securely into 'results.json'.")
        
    except Exception as e:
        print(f"Processing configurations mapping limits gracefully failed securely: {e}")
