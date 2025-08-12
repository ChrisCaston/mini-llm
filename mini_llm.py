import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from typing import Optional, Tuple
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V matrices
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attention_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MiniLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 512,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal mask to prevent attention to future tokens."""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Create causal mask
        mask = self.create_causal_mask(seq_length, device)
        
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
            
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the current sequence
                logits = self(input_ids)
                
                # Get logits for the last token
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Truncate if sequence gets too long
                if input_ids.size(1) > self.max_seq_length:
                    input_ids = input_ids[:, -self.max_seq_length:]
        
        return input_ids

# Simple tokenizer class for demonstration
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        """Build vocabulary from a list of texts."""
        # Get all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text.lower())
        
        # Create vocab mapping
        chars = sorted(list(all_chars))
        self.vocab = {char: idx for idx, char in enumerate(chars)}
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Add special tokens
        self.vocab['<PAD>'] = self.vocab_size
        self.inverse_vocab[self.vocab_size] = '<PAD>'
        self.vocab_size += 1
        
    def encode(self, text):
        """Convert text to token IDs."""
        return [self.vocab.get(char, self.vocab.get('<PAD>', 0)) for char in text.lower()]
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return ''.join([self.inverse_vocab.get(idx, '') for idx in token_ids])
    
    def save(self, filepath):
        """Save tokenizer to file."""
        with open(filepath, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'inverse_vocab': {str(k): v for k, v in self.inverse_vocab.items()},
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, filepath):
        """Load tokenizer from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}
            self.vocab_size = data['vocab_size']

# Training utilities
def train_step(model, batch, optimizer, criterion):
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    
    # Shift inputs and targets for causal LM
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    # Forward pass
    logits = model(inputs)
    
    # Calculate loss
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example usage
if __name__ == "__main__":
    # Example configuration
    vocab_size = 1000
    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=256,
        max_seq_length=256
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example forward pass
    batch_size = 2
    seq_length = 32
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_length, vocab_size)
    
    # Example generation
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generated sequence length: {generated.shape[1]}")
