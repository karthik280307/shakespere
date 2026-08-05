# Shakespeare Mini Language Model

This project is a small, educational implementation of a Shakespeare-inspired mini language model built in Python with PyTorch. The notebook in [myown.ipynb](myown.ipynb) trains a character-level transformer on a text corpus so it can learn patterns in Shakespeare-style language and generate new text.

## What the project does

- Loads a text file and analyzes its vocabulary
- Converts text into integer token IDs
- Builds training and validation datasets
- Implements a compact transformer architecture with:
  - token embeddings
  - positional embeddings
  - multi-head self-attention
  - feed-forward layers
  - a language modeling head
- Trains the model and generates new text from learned patterns

## Technologies used

- Python
- PyTorch
- NumPy

## Setup

Install the required dependencies:

```bash
pip install torch numpy
```

## Data

The notebook expects a text file to be available at the path specified in the notebook:

```python
C:\Users\Hp\OneDrive\Desktop\myprog\ai\dataset\input.txt
```

If your dataset is stored elsewhere, update the file path in the notebook accordingly.

## How to run

1. Open [myown.ipynb](myown.ipynb)
2. Run the cells in order
3. The training loop will print loss values and generate sample text

## Notes

This is a learning-focused project that demonstrates how a tiny GPT-style model can be built from scratch in a compact, beginner-friendly way. It is intentionally small and inspired by Shakespeare, making it a fun introduction to transformer concepts and character-level text generation.
