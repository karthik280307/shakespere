"""
Model architecture and tokenizer for the Shakespeare Transformer.
Fulfills the requirements called by train.py.
"""

import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F


class CharacterTokenizer:
    """Character-level tokenizer for encoding and decoding text."""
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l: list):
        return "".join([self.itos[i] for i in l if i in self.itos])


class ModelConfig:
    """Configuration class for ShakespeareTransformer hyperparameters."""
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 32,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout


class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size: int, config: ModelConfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads: int, head_size: int, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(   
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config.n_head, head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ShakespeareTransformer(nn.Module):
    """Full character-level GPT Transformer model."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def save_checkpoint(model: nn.Module, tokenizer: CharacterTokenizer, filepath: str, extra_metadata: dict = None):
    """Saves model weights, configuration, vocabulary mappings, and metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "tokenizer_stoi": tokenizer.stoi,
        "tokenizer_itos": tokenizer.itos,
        "metadata": extra_metadata or {},
    }
    with open(filepath, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(filepath: str):
    """Loads a saved checkpoint file and returns the model, tokenizer, and metadata."""
    with open(filepath, "rb") as f:
        checkpoint = pickle.load(f)

    config = checkpoint["config"]
    model = ShakespeareTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = CharacterTokenizer("")
    tokenizer.stoi = checkpoint["tokenizer_stoi"]
    tokenizer.itos = checkpoint["tokenizer_itos"]
    tokenizer.vocab_size = len(tokenizer.stoi)

    return model, tokenizer, checkpoint.get("metadata", {})