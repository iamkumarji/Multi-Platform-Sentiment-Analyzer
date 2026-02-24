import pandas as pd
import requests
import logging
import time
from datetime import datetime

from collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)


class RedditCollector(BaseCollector):
    """Collects Reddit posts by searching via Reddit's public JSON API (no auth needed)."""

    BASE_URL = "https://www.reddit.com/search.json"
    HEADERS = {
        "User-Agent": "SentimentAnalyzer/2.0 (keyword search tool)",
    }

    def collect(self, query: str, limit: int = 100,
                sort: str = "relevance", time_filter: str = "month", **kwargs) -> pd.DataFrame:
        rows = []
        after = None

        while len(rows) < limit:
            params = {
                "q": query,
                "limit": min(100, limit - len(rows)),
                "sort": sort,
                "t": time_filter,
                "type": "link",
            }
            if after:
                params["after"] = after

            try:
                resp = requests.get(
                    self.BASE_URL, headers=self.HEADERS,
                    params=params, timeout=15,
                )

                if resp.status_code == 429:
                    logger.warning("Reddit rate limited, waiting 10s...")
                    time.sleep(10)
                    continue

                if resp.status_code != 200:
                    logger.error(f"Reddit returned status {resp.status_code}")
                    break

                data = resp.json().get("data", {})
                children = data.get("children", [])

                if not children:
                    break

                for child in children:
                    post = child.get("data", {})
                    title = post.get("title", "")
                    selftext = post.get("selftext", "")
                    text = f"{title}. {selftext}".strip() if selftext else title

                    created = post.get("created_utc", 0)
                    try:
                        date = datetime.utcfromtimestamp(created)
                    except (ValueError, OSError):
                        date = datetime.now()

                    rows.append({
                        "id": post.get("id", ""),
                        "text": text,
                        "date": date,
                        "author": post.get("author", "[deleted]"),
                        "platform": "reddit",
                        "metadata": {
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                            "subreddit": post.get("subreddit", ""),
                            "url": f"https://reddit.com{post.get('permalink', '')}",
                        },
                    })

                after = data.get("after")
                if not after:
                    break

                time.sleep(1.5)

            except requests.RequestException as e:
                logger.error(f"Reddit request failed: {e}")
                break

        df = pd.DataFrame(rows[:limit])
        logger.info(f"Reddit: collected {len(df)} posts for '{query}'")
        return self._validate(df)
