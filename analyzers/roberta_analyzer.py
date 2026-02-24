import numpy as np
import logging
from scipy.special import softmax

from config.settings import ROBERTA_MODEL

logger = logging.getLogger(__name__)


class RobertaAnalyzer:
    """Deep sentiment analysis using a fine-tuned RoBERTa transformer model."""

    def __init__(self, model_name: str = ROBERTA_MODEL):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

        logger.info(f"Loading RoBERTa model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model.eval()
        logger.info("RoBERTa model loaded successfully")

    def analyze(self, text: str) -> dict:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with __import__("torch").no_grad():
            output = self.model(**encoded)
        scores = softmax(output.logits[0].numpy())
        label_idx = int(np.argmax(scores))
        return {
            "roberta_label": self.config.id2label[label_idx],
            "roberta_score": float(scores[label_idx]),
            "roberta_positive": float(scores[2]),
            "roberta_negative": float(scores[0]),
            "roberta_neutral": float(scores[1]),
        }

    def analyze_batch(self, texts: list, batch_size: int = 32) -> list:
        import torch

        results = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch, return_tensors="pt", truncation=True, padding=True, max_length=512
                )
                output = self.model(**encoded)
                batch_scores = softmax(output.logits.numpy(), axis=1)
                for scores in batch_scores:
                    label_idx = int(np.argmax(scores))
                    results.append({
                        "roberta_label": self.config.id2label[label_idx],
                        "roberta_score": float(scores[label_idx]),
                        "roberta_positive": float(scores[2]),
                        "roberta_negative": float(scores[0]),
                        "roberta_neutral": float(scores[1]),
                    })
        return results
