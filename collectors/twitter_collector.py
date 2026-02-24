import pandas as pd
import logging
import os
import requests
from datetime import datetime

from collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


class TwitterCollector(BaseCollector):
    """Collects tweets from X.com via the Twitter API v2 (Bearer Token)."""

    def collect(self, query: str, limit: int = 100, **kwargs):
        # Get bearer token
        try:
            import streamlit as st
            bearer_token = st.session_state.get("twitter_bearer_token", "")
        except Exception:
            bearer_token = ""

        if not bearer_token:
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")

        if not bearer_token:
            logger.warning("X.com requires a Bearer Token. Add it in the sidebar.")
            return self._empty_df()

        try:
            return self._fetch_tweets(query, limit, bearer_token)
        except Exception as e:
            logger.error(f"X.com collection failed: {e}")
            return self._empty_df()

    def _fetch_tweets(self, query: str, limit: int, bearer_token: str) -> pd.DataFrame:
        headers = {"Authorization": f"Bearer {bearer_token}"}

        params = {
            "query": f"{query} -is:retweet lang:en",
            "max_results": min(limit, 100),
            "tweet.fields": "created_at,public_metrics,author_id",
            "user.fields": "username",
            "expansions": "author_id",
        }

        rows = []
        next_token = None

        while len(rows) < limit:
            if next_token:
                params["next_token"] = next_token

            resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=15)

            if resp.status_code == 401:
                logger.error("X.com API: Invalid Bearer Token.")
                return self._empty_df()
            if resp.status_code == 429:
                logger.warning("X.com API: Rate limit reached.")
                break
            resp.raise_for_status()

            data = resp.json()
            tweets = data.get("data", [])
            if not tweets:
                break

            # Build author lookup from includes
            users = {u["id"]: u["username"] for u in data.get("includes", {}).get("users", [])}

            for tweet in tweets:
                text = tweet.get("text", "")
                if not text.strip():
                    continue

                author_id = tweet.get("author_id", "")
                author_name = users.get(author_id, "unknown")

                created = tweet.get("created_at", "")
                try:
                    date = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError):
                    date = datetime.now()

                metrics = tweet.get("public_metrics", {})

                rows.append({
                    "id": tweet.get("id", ""),
                    "text": text,
                    "date": date,
                    "author": f"@{author_name}",
                    "platform": "twitter",
                    "metadata": {
                        "likes": metrics.get("like_count", 0),
                        "retweets": metrics.get("retweet_count", 0),
                        "replies": metrics.get("reply_count", 0),
                    },
                })

                if len(rows) >= limit:
                    break

            next_token = data.get("meta", {}).get("next_token")
            if not next_token:
                break

        df = pd.DataFrame(rows)
        logger.info(f"X.com: collected {len(df)} tweets")
        return self._validate(df) if not df.empty else self._empty_df()
