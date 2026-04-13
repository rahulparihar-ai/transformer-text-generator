import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

# Import our custom components exactly as requested
from data_pipeline import prepare_pipeline
from transformer_model import TransformerModel

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a learning rate scheduler with linear warmup followed by cosine decay.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear Warmup Phase
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine Decay Phase
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

def accuracy_fn(logits, targets):
    """
    Calculates batch-level accuracy by matching highest-probability tokens integers directly against labels.
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total

def validate(model, val_loader, criterion, device):
    """
    Runs an evaluation loop over the validation dataloader tracking metrics.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            # Reformat to 2D for CrossEntropy mapping (B*S, V) -> (B*S)
            B, S, V = logits.shape
            loss = criterion(logits.view(B * S, V), y.view(B * S))
            
            total_loss += loss.item()
            total_acc += accuracy_fn(logits, y)
            
    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    return avg_loss, avg_acc

def train():
    """
    Core executor that fulfills the comprehensive parameters, logic, metrics, and loop parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device connection acquired: {device}")
    
    print("\n[Sequence 1] Initializing Core Data Pipeline...")
    train_loader, val_loader, test_loader, tokenizer, stats = prepare_pipeline(
        text_path="speare.txt",
        seq_len=64,
        batch_size=32
    )
    
    VOCAB_SIZE = stats["vocab_size"]
    print(f"\n[Sequence 2] Building Custom Transformer Model (Dynamic Vocab Bounds: {VOCAB_SIZE})...")
    
    # -------------------------------------------------------------
    # Updated Transformer configuration maximizing logical accuracy safely downwards!
    # -------------------------------------------------------------
    model = TransformerModel(
        vocab_size=VOCAB_SIZE, 
        d_model=96, 
        num_heads=4, 
        num_layers=2, 
        d_ff=512, 
        max_seq_length=64, 
        dropout=0.3
    ).to(device)
    
    print("[Sequence 3] Establishing Loss Function and Adam Optimizer...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # -------------------------------------------------------------
    # Updated to 0.0007 structurally mapping optimizations implicitly
    # -------------------------------------------------------------
    optimizer = Adam(model.parameters(), lr=0.0003, weight_decay=0.001)
    
    # -------------------------------------------------------------
    # Set to 15 explicitly mapped
    # -------------------------------------------------------------
    EPOCHS = 10
    total_steps = EPOCHS * len(train_loader)
    
    warmup_steps = min(500, int(total_steps * 0.1) if total_steps > 0 else 1)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    val_interval = 50
    best_val_loss = float('inf')
    early_stop_patience = 3
    epochs_no_improve = 0
    
    # ----------------------------------------------------
    # Establish tracking lists for metric charting continuously matching metrics
    # ----------------------------------------------------
    train_losses = []
    val_losses = []
    val_steps = []
    current_step = 0
    
    print(f"Configured limits => Total Steps: {total_steps} | Warmup Steps: {warmup_steps}")
    print("\n================== FULL TRAINING INITIATED ==================")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            B, S, V = logits.shape
            loss = criterion(logits.view(B * S, V), y.view(B * S))
            
            loss.backward()
            
            # --- Inject gradient noise for regularization robustness ---
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * 0.001
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # --- Track metrics mapping downwards continually ---
            train_losses.append(loss.item())
            current_step += 1
            
            if batch_idx % val_interval == 0 or batch_idx == len(train_loader):
                acc = accuracy_fn(logits, y)
                ppl = math.exp(loss.item()) if loss.item() < 50 else float('inf')
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Perplexity: {ppl:.2f} | Acc: {acc:.4f}")
                
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_ppl = math.exp(val_loss) if val_loss < 50 else float('inf')
        
        # --- Store validation metrics efficiently mapping bounds ---
        val_losses.append(val_loss)
        val_steps.append(current_step)
        
        print("-" * 65)
        print(f"[Epoch {epoch} Summary] Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), "best_model.pt")
            print("  --> Validation loss successfully minimized! Model safely checkpointed locally as 'best_model.pt'.")
            
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement made... Iterating early stopping patience limits: {epochs_no_improve}/{early_stop_patience}")
            
        print("=" * 65)
            
        if epochs_no_improve >= early_stop_patience:
            print("Early stopping constraints triggered consecutively! Halt flag executed gracefully.")
            break

    # ----------------------------------------------------
    # Generate graphs natively mapping variables visually capturing loss mappings accurately
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color="blue", alpha=0.6)
    plt.plot(val_steps, val_losses, label="Validation Loss", color="red", marker="o", linewidth=2)
    
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("loss_curve.png")
    plt.show()
    plt.close()
    
    print("loss_curve.png saved")

if __name__ == "__main__":
    train()
