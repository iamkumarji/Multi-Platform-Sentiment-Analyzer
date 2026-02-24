import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st


def generate_wordcloud(texts: list, title: str = "Word Cloud",
                        max_words: int = 100, colormap: str = "viridis") -> plt.Figure:
    """Generate and return a matplotlib figure with a word cloud."""
    combined_text = " ".join(str(t) for t in texts if isinstance(t, str))

    if not combined_text.strip():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No text data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    wc = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color="white",
        colormap=colormap,
        collocations=False,
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def display_sentiment_wordclouds(df, label_col="final_label"):
    """Display positive, negative, and neutral word clouds side by side."""
    col1, col2, col3 = st.columns(3)

    for col, sentiment, title, cmap in [
        (col1, "positive", "Positive Words", "Greens"),
        (col2, "negative", "Negative Words", "Reds"),
        (col3, "neutral", "Neutral Words", "Blues"),
    ]:
        with col:
            texts = df[df[label_col] == sentiment]["clean_text"].dropna().tolist()
            if not texts and "text" in df.columns:
                texts = df[df[label_col] == sentiment]["text"].dropna().tolist()
            fig = generate_wordcloud(texts, title, colormap=cmap)
            st.pyplot(fig)
            plt.close(fig)
