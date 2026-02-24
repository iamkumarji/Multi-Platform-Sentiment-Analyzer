import re


class TextPreprocessor:
    """Platform-aware text cleaning for sentiment analysis."""

    # Patterns compiled once
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _MENTION_RE = re.compile(r"@\w+")
    _HASHTAG_RE = re.compile(r"#(\w+)")
    _HTML_ENTITY_RE = re.compile(r"&\w+;")
    _WHITESPACE_RE = re.compile(r"\s+")
    _MARKDOWN_RE = re.compile(r"[*_~`>{}\[\]]+")
    _RT_RE = re.compile(r"^RT\s+", re.IGNORECASE)

    def preprocess(self, text: str, platform: str = "general") -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._HTML_ENTITY_RE.sub(" ", text)
        text = self._URL_RE.sub("", text)

        if platform == "twitter":
            text = self._RT_RE.sub("", text)
            text = self._MENTION_RE.sub("@user", text)
            text = self._HASHTAG_RE.sub(r"\1", text)  # keep hashtag word
        elif platform == "reddit":
            text = self._MARKDOWN_RE.sub("", text)
        elif platform == "amazon":
            text = re.sub(r"Verified Purchase", "", text, flags=re.IGNORECASE)

        text = self._WHITESPACE_RE.sub(" ", text).strip()
        return text

    def preprocess_for_roberta(self, text: str) -> str:
        """RoBERTa-specific: replace usernames and URLs per model card."""
        tokens = []
        for token in text.split():
            if token.startswith("@") and len(token) > 1:
                token = "@user"
            elif token.startswith("http"):
                token = "http"
            tokens.append(token)
        return " ".join(tokens)
