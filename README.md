# 📜 Shakespeare Transformer

A character-level GPT Transformer built from scratch using **PyTorch** and deployed as an interactive web application with **Streamlit**.

The model is trained on the Tiny Shakespeare dataset (~1.1 million characters) and learns to generate stylized Elizabethan prose and dialogue conditioned on custom user prompts.

---

## ✨ Features

- **Character-Level Transformer Architecture**: Self-attention, multi-head attention, residual connections, layer normalization, and feed-forward networks based on Andrej Karpathy's nanoGPT design.
- **Interactive Streamlit Web UI**: Elegant, Shakespearean-themed interface with custom styling and responsive typography.
- **Dynamic Sampling Controls**:
  - Configurable generation length (characters/tokens)
  - Temperature control (for creative vs deterministic output)
  - Optional Top-k probability filtering
  - Random seed control for deterministic reproducibility
- **Pre-set Famous Seed Prompts**: One-click selection of iconic quotes from *Hamlet*, *Romeo and Juliet*, *Julius Caesar*, *Macbeth*, and more.
- **Performance & Real-Time Stats**: Displays generation time, output length, and throughput (characters per second).
- **Export Options**: One-click download of generated text.
- **Cached Model Inference**: Uses `st.cache_resource` for zero-overhead multi-turn text generation.

---

## 🏛️ Model Architecture

The Transformer is a decoder-only autoregressive language model designed with the following specifications:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Vocabulary Size** | 65 | Character-level unique characters |
| **Context Window (Block Size)** | 32 | Maximum sequence length for attention |
| **Embedding Dimension ($n_{embd}$)** | 64 | Latent token & positional embedding size |
| **Attention Heads ($n_{head}$)** | 4 | Number of parallel self-attention heads |
| **Transformer Layers ($n_{layer}$)** | 4 | Stack of Transformer blocks |
| **Total Parameters** | ~209,729 | Learnable weights |
| **Device Support** | CPU / CUDA | Runs seamlessly on standard CPU or GPU |

---

## 📁 Project Structure

```
shakesphere/
├── app.py                  # Main Streamlit web application
├── model.py                # Transformer model architecture & tokenizer
├── train.py                # Training script for training checkpoint
├── utils.py                # UI styling, prompts, and performance helpers
├── demo_model.pkl          # Trained Transformer model checkpoint
├── input.txt               # Tiny Shakespeare dataset
├── build_gpt_andrej.ipynb  # Educational notebook (step-by-step GPT build)
├── myown.ipynb             # Model prototyping and experiment notebook
├── index.html              # Standalone web showcase
├── pyproject.toml          # Project configuration & dependencies
├── requirements.txt        # Standard pip requirements
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python `>= 3.10` (Python 3.12 recommended)
- `uv` (recommended) or standard `python3` / `pip`

### 2. Environment Setup & Dependency Installation

#### Option A: Using `uv` (Fast & Recommended)

```bash
# Clone or navigate to the repository
cd shakesphere

# Create virtual environment and sync dependencies
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

#### Option B: Using standard `venv` and `pip`

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 🏋️ Model Training (Optional)

A trained checkpoint is saved in `demo_model.pkl`. If you wish to retrain or experiment with custom hyperparameters:

```bash
python train.py
```

Training runs for 3,000 iterations and takes under ~30 seconds on a standard CPU.

---

## 🖥️ Running the Streamlit Application

Start the web application locally:

```bash
streamlit run app.py
```

Once started, open your web browser at:
`http://localhost:8501`

---

## 💡 Example Usage

1. **Enter a Prompt**: Type `To be, or not to be,` or choose one of the quick preset buttons.
2. **Adjust Controls**:
   - Set **Temperature** to `0.8` for balanced creativity.
   - Set **Output Length** to `300` characters.
   - (Optional) Enable **Top-k Sampling** (`k=15`) for tighter coherence.
3. **Generate**: Click **✨ Generate**.
4. **Save**: Click **📥 Download Text** to export your generated passage.

---

## 📜 License & Acknowledgments

- Built using principles from Andrej Karpathy's *nanoGPT* and *Neural Networks: Zero to Hero* lecture series.
- Dataset: *Tiny Shakespeare* corpus.
