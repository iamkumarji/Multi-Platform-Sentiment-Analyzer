import pandas as pd
import logging
from tqdm import tqdm

from analyzers.preprocessor import TextPreprocessor
from analyzers.vader_analyzer import VaderAnalyzer
from analyzers.roberta_analyzer import RobertaAnalyzer

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """Orchestrates preprocessing + VADER + RoBERTa into a single analysis flow."""

    def __init__(self, use_roberta: bool = True):
        self.preprocessor = TextPreprocessor()
        self.vader = VaderAnalyzer()
        self.roberta = None
        if use_roberta:
            try:
                self.roberta = RobertaAnalyzer()
            except Exception as e:
                logger.warning(f"RoBERTa unavailable, using VADER only: {e}")

    def analyze_dataframe(self, df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """Runs the full sentiment pipeline on a DataFrame with 'text' and 'platform' columns."""
        df = df.copy()

        # Step 1: Preprocess
        logger.info("Step 1/3: Preprocessing text...")
        df["clean_text"] = df.apply(
            lambda r: self.preprocessor.preprocess(str(r.get("text", "")), str(r.get("platform", ""))),
            axis=1,
        )

        # Step 2: VADER (fast)
        logger.info("Step 2/3: Running VADER analysis...")
        iterator = tqdm(df["clean_text"], desc="VADER") if show_progress else df["clean_text"]
        vader_results = [self.vader.analyze(text) for text in iterator]
        vader_df = pd.DataFrame(vader_results)
        for col in vader_df.columns:
            df[col] = vader_df[col].values

        # Step 3: RoBERTa (slow, batched)
        if self.roberta:
            logger.info("Step 3/3: Running RoBERTa analysis...")
            roberta_texts = df["clean_text"].apply(self.preprocessor.preprocess_for_roberta).tolist()
            roberta_results = self.roberta.analyze_batch(roberta_texts)
            roberta_df = pd.DataFrame(roberta_results)
            for col in roberta_df.columns:
                df[col] = roberta_df[col].values
        else:
            logger.info("Step 3/3: Skipping RoBERTa (not available)")

        # Step 4: Ensemble
        df["final_label"] = df.apply(self._ensemble_label, axis=1)
        df["final_score"] = df.apply(self._ensemble_score, axis=1)

        return df

    def _ensemble_label(self, row) -> str:
        if self.roberta and "roberta_label" in row and pd.notna(row.get("roberta_label")):
            if row["vader_label"] == row["roberta_label"]:
                return row["vader_label"]
            return row["roberta_label"]  # RoBERTa wins on disagreement
        return row["vader_label"]

    def _ensemble_score(self, row) -> float:
        vader_norm = (row["vader_compound"] + 1) / 2  # map -1..1 → 0..1
        if self.roberta and "roberta_positive" in row and pd.notna(row.get("roberta_positive")):
            roberta_score = row["roberta_positive"] - row["roberta_negative"]
            roberta_norm = (roberta_score + 1) / 2
            return 0.3 * vader_norm + 0.7 * roberta_norm
        return vader_norm
