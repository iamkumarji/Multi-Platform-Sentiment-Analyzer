from abc import ABC, abstractmethod
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for all platform collectors.

    All collectors must return a DataFrame with this uniform schema:
        id       : str       - Unique identifier
        text     : str       - Raw text content
        date     : datetime  - Publication timestamp
        author   : str       - Username / source name
        platform : str       - "twitter" | "reddit" | "news" | "amazon"
        metadata : dict      - Platform-specific extras
    """

    SCHEMA_COLUMNS = ["id", "text", "date", "author", "platform", "metadata"]

    @abstractmethod
    def collect(self, query: str, limit: int = 500, **kwargs) -> pd.DataFrame:
        """Collect data and return a DataFrame with the uniform schema."""

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df[self.SCHEMA_COLUMNS]

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.SCHEMA_COLUMNS)
