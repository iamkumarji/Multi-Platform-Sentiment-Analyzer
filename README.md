# Multi-Platform Sentiment Analyzer

A real-time sentiment analysis dashboard built with Streamlit. Scrape live data from **Reddit**, **X.com (Twitter)**, and **Amazon.in**, then analyze it using **VADER** and **RoBERTa** NLP models.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Multi-platform data collection** — Reddit (no API key), X.com (Bearer Token), Amazon.in (web scraping)
- **File upload support** — Analyze your own CSV or TXT files with VADER and/or RoBERTa
- **Dual-model sentiment analysis** — VADER (fast, rule-based) + RoBERTa (transformer-based, more accurate)
- **5-tab interactive dashboard**
  - **Overview** — Distribution bar chart, pie chart, platform radar comparison
  - **Deep Analysis** — Score histogram, confidence box plot, VADER vs RoBERTa scatter, text length violin
  - **Trends & Time** — Sentiment timeline, post volume over time, heatmap (with daily/weekly/monthly toggle)
  - **Text Insights** — Word clouds (positive/negative/neutral), notable texts, engagement analysis, top authors
  - **Data & Export** — Filtered data table with sentiment/platform/score filters, CSV & JSON download
- **Auto-generated summary** with dominant sentiment, platform comparison, and model agreement stats

## Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **Git**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iamkumarji/Multi-Platform-Sentiment-Analyzer.git
cd Multi-Platform-Sentiment-Analyzer
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the RoBERTa model (~500 MB). This is a one-time download.

### 4. Set up environment variables (optional)

```bash
cp .env.example .env
```

Edit `.env` if you want to pre-configure API keys:

```env
# X.com (Twitter) — Bearer Token from https://developer.x.com/en/portal/dashboard
TWITTER_BEARER_TOKEN=your_bearer_token_here

# Reddit — Not required (public JSON API works without credentials)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
```

> You can also set the X.com Bearer Token directly in the app sidebar at runtime.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

### Search Online

1. Select **Search Online** in the sidebar
2. Enter a keyword or topic (e.g., "iPhone", "ChatGPT")
3. Select one or more platforms — Reddit, X.com, Amazon.in
4. Adjust the results limit (10–200 per platform)
5. Toggle **RoBERTa model** for higher accuracy (slower)
6. Click **Search & Analyze**
7. Explore results across the 5 dashboard tabs

### File Upload

1. Select **Upload File** in the sidebar
2. Upload a `.csv` or `.txt` file
3. For CSV files, specify the **text column name** (default: `text`)
4. Toggle **RoBERTa model** if desired
5. Click **Analyze File**

**Supported formats:**

| Format | Requirements |
|--------|-------------|
| **CSV** | Must contain a column with the text to analyze (default name: `text`). Optional columns: `date` (enhances timeline charts), `author` (shown in Top Authors) |
| **TXT** | One text entry per line — each line is treated as a separate text to analyze |

**Notes:**
- Optional `date` and `author` columns in CSV files enhance the timeline and author visualizations
- All 5 dashboard tabs work with uploaded data, same as with online search results

## Platform Setup

| Platform | Setup Required | Notes |
|----------|---------------|-------|
| **Reddit** | None | Works out of the box via public JSON API |
| **X.com (Twitter)** | Bearer Token | Get one from the [Twitter Developer Portal](https://developer.x.com/en/portal/dashboard). Enter it in the sidebar or `.env` file |
| **Amazon.in** | None | Web scraping — may be rate-limited on heavy use |

## Project Structure

```
Multi-Platform-Sentiment-Analyzer/
├── app.py                     # Main Streamlit application
├── analyzers/
│   ├── preprocessor.py        # Text cleaning & preprocessing
│   ├── vader_analyzer.py      # VADER sentiment analysis
│   ├── roberta_analyzer.py    # RoBERTa transformer model
│   └── sentiment_pipeline.py  # Ensemble pipeline (VADER + RoBERTa)
├── collectors/
│   ├── base_collector.py      # Abstract base collector
│   ├── reddit_collector.py    # Reddit data collector
│   ├── twitter_collector.py   # X.com Twitter API v2 collector
│   └── amazon_collector.py    # Amazon.in review scraper
├── utils/
│   ├── chart_helpers.py       # Plotly chart functions
│   └── wordcloud_helper.py    # Word cloud generation
├── config/
│   └── settings.py            # App configuration & env loading
├── .env.example               # Example environment variables
├── .streamlit/
│   └── config.toml            # Streamlit UI configuration
└── requirements.txt           # Python dependencies
```

## Tech Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly, Matplotlib, WordCloud
- **NLP Models:** VADER (vaderSentiment), RoBERTa (HuggingFace Transformers)
- **Data Collection:** Requests, BeautifulSoup, CloudScraper
- **Data Processing:** Pandas, NumPy

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Make sure the virtual environment is activated and run `pip install -r requirements.txt` |
| RoBERTa model download fails | Check your internet connection. The model downloads from HuggingFace on first run |
| X.com returns no results | Verify your Bearer Token is valid and has search access |
| Amazon.in returns no results | May be rate-limited — try again after a few minutes |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
