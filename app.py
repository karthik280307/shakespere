"""
Streamlit web application for interactive Shakespeare text generation.
"""

import time
import torch
import streamlit as st
from model import load_checkpoint
from utils import EXAMPLE_PROMPTS, format_stats, get_custom_css

st.set_page_config(
    page_title="Shakespearean Transformer Studio",
    page_icon="📜",
    layout="wide",
)

st.markdown(get_custom_css(), unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    try:
        model, tokenizer, metadata = load_checkpoint("demo_model.pkl")
        return model, tokenizer, metadata
    except FileNotFoundError:
        return None, None, None

model, tokenizer, metadata = load_model_and_tokenizer()

# Main header
st.markdown("""
<div class="main-header">
    <div class="main-title">Shakespearean Transformer</div>
    <div class="main-subtitle">A Character-Level Generative Language Model Trained on Tiny Shakespeare</div>
    <div class="badge-bar">
        <span class="badge-pill">PyTorch GPT</span>
        <span class="badge-pill">Character-Level</span>
        <span class="badge-pill">Transformer Architecture</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Checkpoint file `demo_model.pkl` not found! Please run `python train.py` first to train and save the model.")
    st.stop()

# Sidebar controls
st.sidebar.markdown('<div class="sidebar-section-title">Generation Parameters</div>', unsafe_allow_html=True)
max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=50, max_value=500, value=200, step=25)
temperature = st.sidebar.slider("Temperature (Creativity)", min_value=0.2, max_value=1.5, value=0.8, step=0.1)

st.sidebar.markdown('<div class="sidebar-section-title">Model Metadata</div>', unsafe_allow_html=True)
st.sidebar.json(metadata)

# Preset Prompts Selection using EXAMPLE_PROMPTS from utils.py
st.markdown("### Choose a Classic Seed Prompt")
cols = st.columns(3)
selected_prompt = "To be, or not to be, that is the question:"

for i, p in enumerate(EXAMPLE_PROMPTS):
    with cols[i % 3]:
        if st.button(f"**{p['title']}**\n\n*{p['play']}*", use_container_width=True):
            selected_prompt = p["prompt"]

prompt_input = st.text_area("Or enter your own Shakespearean prompt:", value=selected_prompt, height=100)

if st.button("Generate Verse ✨", type="primary", use_container_width=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    context_tokens = tokenizer.encode(prompt_input)
    if not context_tokens:
        context_tokens = [0]
        
    context = torch.tensor([context_tokens], dtype=torch.long, device=device)

    start_time = time.time()
    with torch.no_grad():
        generated_idx = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
    duration = time.time() - start_time

    generated_text = tokenizer.decode(generated_idx[0].tolist())
    stats = format_stats(generated_text, duration)

    st.markdown(f"""
    <div class="output-card">
        <div class="output-header">
            <span class="output-title">Generated Performance</span>
            <div style="display: flex; gap: 0.5rem;">
                <span class="stat-pill">{stats['character_count']} chars</span>
                <span class="stat-pill">{stats['word_count']} words</span>
                <span class="stat-pill">{stats['duration_seconds']}s</span>
                <span class="stat-pill">{stats['chars_per_second']} chars/sec</span>
            </div>
        </div>
        <div class="output-text">{generated_text}</div>
    </div>
    """, unsafe_allow_html=True)