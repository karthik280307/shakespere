"""
Training script for the Shakespeare Transformer model.
Trains a character-level Transformer on tiny Shakespeare text and saves the checkpoint to demo_model.pkl.
"""

import os
import sys
import time
from pathlib import Path

import torch
from model import CharacterTokenizer, ModelConfig, ShakespeareTransformer, save_checkpoint


def train_shakespeare_model(
    data_path: str = "input.txt",
    output_path: str = "demo_model.pkl",
    max_iters: int = 3000,
    batch_size: int = 32,
    block_size: int = 32,
    n_embd: int = 64,
    n_head: int = 4,
    n_layer: int = 4,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    eval_interval: int = 300,
    eval_iters: int = 100,
    seed: int = 1337,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    if not os.path.exists(data_path):
        print(f"Downloading Tiny Shakespeare dataset to {data_path}...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharacterTokenizer(text)
    vocab_size = tokenizer.vocab_size
    print(f"Dataset length: {len(text):,} characters")
    print(f"Vocabulary size: {vocab_size} unique characters")

    # Encode data into torch.Tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split: str):
        split_data = train_data if split == "train" else val_data
        ix = torch.randint(len(split_data) - block_size, (batch_size,))
        x = torch.stack([split_data[i : i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # Initialize model
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )
    model = ShakespeareTransformer(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {param_count:,} parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    print(f"Starting training for {max_iters} iterations...")
    start_time = time.time()
    loss_history = []

    for step in range(max_iters + 1):
        if step % eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            print(
                f"Step {step:4d}/{max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Elapsed: {elapsed:.1f}s"
            )
            loss_history.append({"step": step, "train": losses["train"], "val": losses["val"]})

        if step == max_iters:
            break

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # Save model and metadata
    metadata = {
        "dataset": "Tiny Shakespeare",
        "dataset_characters": len(text),
        "total_parameters": param_count,
        "training_iterations": max_iters,
        "final_train_loss": loss_history[-1]["train"] if loss_history else None,
        "final_val_loss": loss_history[-1]["val"] if loss_history else None,
        "loss_history": loss_history,
        "training_time_seconds": round(total_time, 2),
    }

    save_checkpoint(model.cpu(), tokenizer, output_path, extra_metadata=metadata)
    print(f"Model saved successfully to {output_path}")

    # Generate quick demo text
    model.eval()
    context = torch.tensor([tokenizer.encode("To be, or not to be,")], dtype=torch.long)
    generated = model.generate(context, max_new_tokens=150, temperature=0.8)
    sample_text = tokenizer.decode(generated[0].tolist())
    print("\n--- Sample Generation ---")
    print(sample_text)
    print("-------------------------\n")


if __name__ == "__main__":
    train_shakespeare_model()
