import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements standard sinusoidal positional encoding to inject chronological
    order logic into the sequences using sine / cosine frequencies.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer so it's a non-trainable state of the module
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Extract sequence length from input and add the pe to embeddings
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class MultiHeadAttention(nn.Module):
    """
    Executes Multi-Head Attention by separating Q, K, V into explicit heads, 
    computing Scaled Dot-Product Attention natively, mapping back and concatenating.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be fully divisible by num_heads."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key, Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Linear layer for concatenating the heads sequentially
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 1. Project through Linear dimensions (B, S, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Reshape to split into num_heads (B, num_heads, S, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention Phase
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Overwrite future context interactions with -infinity
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply the attention weights identically to V matrices
        context = torch.matmul(attention_weights, V)
        
        # 4. Reconfigure shapes by concatenating heads dynamically
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 5. Final linear projection block 
        output = self.W_o(context)
        return output

class FeedForwardNetwork(nn.Module):
    """
    Standard Feed-Forward Neural Network: Linear -> ReLU -> Linear.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """
    Self-contained Transformer Decoder block using the Pre-LayerNorm configuration.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # MHA Pathway
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        # FeedForward Pathway
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Pre-Norm for Stability + Attention + Residual
        norm1_out = self.norm1(x)
        attn_out = self.mha(norm1_out, mask)
        x = x + self.dropout1(attn_out)

        # 2. Pre-Norm for Stability + Feed-Forward + Residual
        norm2_out = self.norm2(x)
        ffn_out = self.ffn(norm2_out)
        x = x + self.dropout2(ffn_out)
        
        return x

class TransformerModel(nn.Module):
    """
    Top-Level Module tying the components to create a Next-Word Predictor.
    """
    def __init__(self, vocab_size=5000, d_model=96, num_heads=4, num_layers=2, d_ff=512, max_seq_length=64, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # Standard Token Embedding structure
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Sequence positional handling
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        # Stacked N Transformer Decoder Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers (final Pre-Norm is best practice logic)
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, seq_len, device):
        # Lower triangular layout (predicts conditionally only up to t steps backward)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # Expand out mapping to (batch=1, heads=1, S, S)
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 1. Embed tokens (Scaling trick assists magnitude alignment)
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # 2. Apply positional embeddings
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 3. Create active sequence mask logic
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        
        # 4. Filter iteratively directly across transformer blocks
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
            
        # 5. Output the logic up into final dimensional vocab mappings
        x = self.final_norm(x)
        logits = self.fc_out(x)
        
        return logits

def count_parameters(model):
    """Utility to track trainable parameter capacity."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Define strictly configured constraints
    VOCAB_SIZE = 5000
    D_MODEL = 96
    NUM_HEADS = 4
    NUM_LAYERS = 2
    D_FF = 512
    MAX_SEQ_LENGTH = 64
    DROPOUT = 0.3

    print("Initializing Custom Transformer Architecture...")
    model = TransformerModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT
    )
    
    print("-" * 50)
    print(f"Total model parameters: {count_parameters(model):,}")
    print("-" * 50)
    
    # Validation testing
    print("Executing Testing Evaluation pass...")
    batch_size = 32
    seq_len = 64
    
    # 32x64 logic of mapped generic integers
    batch = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    print(f"Input batch shape: {tuple(batch.shape)}")
    
    # Run fully stacked forward pass 
    output = model(batch)
    print(f"Output shape: {tuple(output.shape)}")
    
    if output.shape == (batch_size, seq_len, VOCAB_SIZE):
         print("SUCCESS: Input correctly mapped strictly inside parameters mapping limits!")
