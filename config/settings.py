import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ENV_FILE = BASE_DIR / ".env"

# Reddit API (optional — JSON API works without these)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "SentimentAnalyzer/2.0")

# Twitter / X.com (loaded from .env, can be overridden at runtime)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# Sentiment model
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Platforms
PLATFORMS = ["twitter", "reddit", "amazon"]


def get_twitter_bearer_token() -> str:
    """Get the Twitter bearer token. Checks Streamlit session state first,
    then falls back to environment variable."""
    try:
        import streamlit as st
        token = st.session_state.get("twitter_bearer_token", "")
        if token:
            return token
    except Exception:
        pass
    return TWITTER_BEARER_TOKEN


def save_env_key(key: str, value: str):
    """Save or update a key in the .env file."""
    lines = []
    found = False

    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key}={value}\n")

    with open(ENV_FILE, "w") as f:
        f.writelines(new_lines)

    # Also update the environment variable for the current process
    os.environ[key] = value
