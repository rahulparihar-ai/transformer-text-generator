import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import os

from data_pipeline import prepare_pipeline
from model.transformer_model import TransformerModel
from config import CONFIG

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def accuracy_fn(logits, targets):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).sum().item()
    return correct / targets.numel()

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, S, V = logits.shape
            loss = criterion(logits.view(B * S, V), y.view(B * S))
            total_loss += loss.item()
            total_acc += accuracy_fn(logits, y)
    return total_loss / len(val_loader), total_acc / len(val_loader)

def train():
    print("=== MODEL CONFIGURATION ===")
    for k, v in CONFIG.items():
        print(f"{k}: {v}")
    print("===========================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, val_loader, test_loader, tokenizer, stats = prepare_pipeline(
        text_path="data/speare.txt",
        seq_len=CONFIG["max_seq_len"],
        batch_size=32
    )

    model = TransformerModel().to(device)
    
    print(f"Embedding shape: {model.token_embedding.weight.shape}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=CONFIG.get("learning_rate", 3e-4), weight_decay=0.001)
    
    EPOCHS = 10
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = min(500, int(total_steps * 0.1) if total_steps > 0 else 1)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    val_interval, best_val_loss, early_stop_patience, epochs_no_improve = 50, float('inf'), 3, 0
    train_losses, val_losses, val_steps, current_step = [], [], [], 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            
            B, S, V = logits.shape
            loss = criterion(logits.view(B * S, V), y.view(B * S))
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.get("clip_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            current_step += 1
            
            if batch_idx % val_interval == 0 or batch_idx == len(train_loader):
                acc = accuracy_fn(logits, y)
                ppl = math.exp(loss.item()) if loss.item() < 50 else float('inf')
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Perplexity: {ppl:.2f} | Acc: {acc:.4f}")
                
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_ppl = math.exp(val_loss) if val_loss < 50 else float('inf')
        val_losses.append(val_loss)
        val_steps.append(current_step)
        
        print(f"[Epoch {epoch} Summary] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "model/best_model.pt")
            print("  --> Model checkpoint saved as 'model/best_model.pt'.")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered!")
            break

if __name__ == "__main__":
    train()
