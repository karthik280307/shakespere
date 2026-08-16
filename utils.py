"""
Utility functions and sample prompts for the Shakespeare Transformer application.
"""

from typing import Any, Dict, List, Union

# Curated Shakespearean seed prompts for quick selection
EXAMPLE_PROMPTS = [
    {
        "title": "Hamlet's Soliloquy",
        "play": "Hamlet (Act 3, Scene 1)",
        "prompt": "To be, or not to be, that is the question:",
    },
    {
        "title": "Romeo's Longing",
        "play": "Romeo and Juliet (Act 2, Scene 2)",
        "prompt": "O Romeo, Romeo! wherefore art thou Romeo?",
    },
    {
        "title": "Mark Antony's Address",
        "play": "Julius Caesar (Act 3, Scene 2)",
        "prompt": "Friends, Romans, countrymen, lend me your ears;",
    },
    {
        "title": "The Seven Ages of Man",
        "play": "As You Like It (Act 2, Scene 7)",
        "prompt": "All the world's a stage, and all the men and women merely players:",
    },
    {
        "title": "Macbeth's Ambition",
        "play": "Macbeth (Act 1, Scene 7)",
        "prompt": "If it were done when 'tis done, then 'twere well",
    },
    {
        "title": "The King's Decree",
        "play": "Henry V (Act 4, Scene 3)",
        "prompt": "We few, we happy few, we band of brothers;",
    },
]


def format_stats(text: str, duration_sec: float) -> Dict[str, Any]:
    """Compute statistics for generated text."""
    words = text.split()
    chars = len(text)
    wpm = (len(words) / duration_sec) * 60 if duration_sec > 0 else 0
    chars_per_sec = (chars / duration_sec) if duration_sec > 0 else 0
    return {
        "character_count": chars,
        "word_count": len(words),
        "lines_count": len(text.splitlines()),
        "duration_seconds": round(duration_sec, 3),
        "chars_per_second": round(chars_per_sec, 1),
    }


def get_custom_css() -> str:
    """Return custom CSS for Shakespearean theme."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400;1,600&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Overall aesthetic */
    .main-header {
        text-align: center;
        padding: 1.5rem 1rem 2rem 1rem;
        background: linear-gradient(180deg, rgba(26, 20, 35, 0.95) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    .main-title {
        font-family: 'Cinzel', serif !important;
        font-size: 2.6rem !important;
        font-weight: 900 !important;
        color: #f5e6c8 !important;
        letter-spacing: 2px !important;
        margin-bottom: 0.4rem !important;
        text-shadow: 0 2px 10px rgba(212, 175, 55, 0.3);
    }

    .main-subtitle {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.25rem !important;
        font-style: italic !important;
        color: #cbd5e1 !important;
        margin-bottom: 1rem !important;
        letter-spacing: 0.5px !important;
    }

    .badge-bar {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }

    .badge-pill {
        background: rgba(212, 175, 55, 0.12);
        color: #e2c77d;
        border: 1px solid rgba(212, 175, 55, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* Output Card */
    .output-card {
        background: linear-gradient(145deg, #161b26 0%, #0d1117 100%);
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 14px;
        padding: 1.75rem;
        margin-top: 1.5rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5);
    }

    .output-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(212, 175, 55, 0.2);
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
    }

    .output-title {
        font-family: 'Cinzel', serif !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #d4af37 !important;
        letter-spacing: 1px;
    }

    .output-text {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-size: 1.22rem !important;
        line-height: 1.75 !important;
        color: #f1f5f9 !important;
        white-space: pre-wrap !important;
        background: rgba(0, 0, 0, 0.25);
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 3px solid #d4af37;
    }

    .prompt-highlight {
        font-weight: 600;
        color: #e2c77d;
        background: rgba(212, 175, 55, 0.15);
        padding: 0 3px;
        border-radius: 3px;
    }

    /* Stats mini badges */
    .stat-pill {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    /* Sidebar headers */
    .sidebar-section-title {
        font-family: 'Cinzel', serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #d4af37 !important;
        letter-spacing: 0.5px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """
