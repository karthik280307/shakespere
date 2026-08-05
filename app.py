import pickle
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Shakespeare Mini Language Model", page_icon="🎭", layout="centered")


@st.cache_data(show_spinner=False)
def load_demo_artifact():
    artifact_path = Path(__file__).with_name("demo_model.pkl")
    with artifact_path.open("rb") as handle:
        return pickle.load(handle)


def generate_text(artifact: dict, prompt: str, length: int) -> str:
    words = artifact["sample_text"].split()
    seed_words = prompt.strip().split()
    result_words = seed_words[:]

    for _ in range(max(10, length)):
        next_word = words[len(result_words) % len(words)]
        result_words.append(next_word)
        if len(result_words) >= length:
            break

    return " ".join(result_words)


artifact = load_demo_artifact()

st.title("🎭 Shakespeare Mini Language Model")
st.write(artifact["description"])

with st.container():
    st.markdown("### Demo controls")
    prompt = st.text_input("Start with a phrase", value="To be")
    length = st.slider("How many words to generate", 20, 120, 60, 10)

    if st.button("Generate text"):
        result = generate_text(artifact, prompt, length)
        st.text_area("Generated output", result, height=220)

    st.markdown("### Pickle-backed artifact")
    st.json({
        "title": artifact["title"],
        "style": artifact["style"],
        "sample_size": len(artifact["sample_text"].split())
    })
