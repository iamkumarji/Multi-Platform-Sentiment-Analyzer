import pandas as pd
import cloudscraper
import logging
import time
import hashlib
from datetime import datetime
from urllib.parse import quote_plus

from collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)


class AmazonCollector(BaseCollector):
    """Collects Amazon.in product reviews by searching for a keyword.

    Flow: search Amazon.in for keyword -> get product ASINs -> scrape reviews
    from each product page's FocalReviews widget.
    """

    def _get_scraper(self):
        return cloudscraper.create_scraper()

    def collect(self, query: str, limit: int = 100, **kwargs) -> pd.DataFrame:
        from bs4 import BeautifulSoup

        scraper = self._get_scraper()

        # Step 1: Search Amazon.in for the keyword
        asins = self._search_products(query, scraper, BeautifulSoup)
        if not asins:
            logger.warning(f"No Amazon products found for '{query}'")
            return self._empty_df()

        logger.info(f"Found {len(asins)} products, scraping reviews...")

        # Step 2: Scrape reviews from each product page
        all_rows = []
        for asin in asins[:8]:
            rows = self._scrape_product_reviews(asin, scraper, BeautifulSoup)
            all_rows.extend(rows)
            if len(all_rows) >= limit:
                break
            time.sleep(2)

        df = pd.DataFrame(all_rows[:limit])
        logger.info(f"Amazon: collected {len(df)} reviews for '{query}'")
        return self._validate(df)

    def _search_products(self, query: str, scraper, BeautifulSoup) -> list:
        """Search Amazon.in and return a list of ASINs."""
        url = f"https://www.amazon.in/s?k={quote_plus(query)}"

        try:
            resp = scraper.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Amazon search returned {resp.status_code}")
                return []

            soup = BeautifulSoup(resp.text, "lxml")

            asins = []
            for item in soup.select('[data-asin]'):
                asin = item.get("data-asin", "").strip()
                if asin and len(asin) == 10:
                    asins.append(asin)

            return list(dict.fromkeys(asins))[:10]

        except Exception as e:
            logger.error(f"Amazon search failed: {e}")
            return []

    def _scrape_product_reviews(self, asin: str, scraper, BeautifulSoup) -> list:
        """Scrape reviews from a product page (/dp/ASIN)."""
        url = f"https://www.amazon.in/dp/{asin}"
        rows = []

        try:
            resp = scraper.get(url, timeout=15)
            if resp.status_code != 200:
                return rows

            soup = BeautifulSoup(resp.text, "lxml")

            # Get product title
            title_el = soup.select_one("#productTitle")
            product_title = title_el.get_text(strip=True) if title_el else ""

            # Reviews are in .cr-widget-FocalReviews as .celwidget divs
            focal = soup.select_one(".cr-widget-FocalReviews")
            if not focal:
                return rows

            review_divs = focal.select(".a-section.celwidget")

            for review in review_divs:
                # Body: span[data-hook="review-body"] contains the review text
                body_el = review.select_one('span[data-hook="review-body"]')
                if not body_el:
                    continue
                body_text = body_el.get_text(strip=True)
                body_text = body_text.replace("Read more", "").replace("Read less", "").strip()

                if not body_text or len(body_text) < 5:
                    continue

                # Title: a[data-hook="review-title"] > last span (skip rating spans)
                review_title = ""
                title_link = review.select_one('a[data-hook="review-title"]')
                if title_link:
                    spans = title_link.find_all("span", recursive=False)
                    for span in spans:
                        t = span.get_text(strip=True)
                        if t and "out of" not in t and t.strip():
                            review_title = t

                text = f"{review_title}. {body_text}" if review_title else body_text

                # Rating
                rating = ""
                rating_el = review.select_one(
                    'i[data-hook="review-star-rating"] .a-icon-alt, '
                    'i.review-rating .a-icon-alt'
                )
                if rating_el:
                    rating = rating_el.get_text(strip=True).split()[0]

                # Author
                author_el = review.select_one(".a-profile-content .a-profile-name")
                author = author_el.get_text(strip=True) if author_el else "Amazon Customer"

                # Date
                date_el = review.select_one('span[data-hook="review-date"]')
                date_text = date_el.get_text(strip=True) if date_el else ""
                try:
                    date_clean = date_text.split(" on ")[-1] if " on " in date_text else date_text
                    date = pd.to_datetime(date_clean, dayfirst=True, errors="coerce")
                    if pd.isna(date):
                        date = datetime.now()
                except Exception:
                    date = datetime.now()

                rows.append({
                    "id": hashlib.md5(text[:100].encode()).hexdigest(),
                    "text": text,
                    "date": date,
                    "author": author,
                    "platform": "amazon",
                    "metadata": {
                        "rating": rating,
                        "asin": asin,
                        "product": product_title[:100],
                    },
                })

        except Exception as e:
            logger.debug(f"Review scraping failed for {asin}: {e}")

        return rows
