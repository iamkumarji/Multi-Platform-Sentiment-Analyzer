import streamlit as st
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit deploy button & hamburger menu
st.markdown(
    """<style>
    .stDeployButton, [data-testid="stToolbar"] {display: none !important;}
    </style>""",
    unsafe_allow_html=True,
)

# ── Platform mapping ──────────────────────────────────────────────
PLATFORM_OPTIONS = {
    "X.com (Twitter)": "twitter",
    "Reddit": "reddit",
    "Amazon.in": "amazon",
}

PLATFORM_ICONS = {
    "twitter": "🐦",
    "reddit": "💬",
    "amazon": "🛒",
}


def get_collector(platform_key: str):
    if platform_key == "twitter":
        from collectors.twitter_collector import TwitterCollector
        return TwitterCollector()
    elif platform_key == "reddit":
        from collectors.reddit_collector import RedditCollector
        return RedditCollector()
    elif platform_key == "amazon":
        from collectors.amazon_collector import AmazonCollector
        return AmazonCollector()


@st.cache_resource
def load_pipeline(use_roberta: bool):
    from analyzers.sentiment_pipeline import SentimentPipeline
    return SentimentPipeline(use_roberta=use_roberta)


# ── Helper: load saved X.com API key ─────────────────────────────
from config.settings import save_env_key, BASE_DIR
import os

if "twitter_bearer_token" not in st.session_state:
    st.session_state["twitter_bearer_token"] = os.getenv("TWITTER_BEARER_TOKEN", "")


def _has_twitter_creds() -> bool:
    return bool(st.session_state.get("twitter_bearer_token"))


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Sentiment Analyzer")
    st.caption("Search across platforms & analyze sentiment in real-time")
    st.markdown("---")

    keyword = st.text_input(
        "🔍 Keyword / Topic",
        placeholder="e.g. iPhone, ChatGPT, Modi",
    )

    platforms = st.multiselect(
        "🌐 Select Platforms",
        list(PLATFORM_OPTIONS.keys()),
        default=["Reddit"],
    )

    limit = st.slider("Results per platform", min_value=10, max_value=200, value=50, step=10)

    use_roberta = st.toggle("Use RoBERTa model (slower, more accurate)", value=False)

    st.markdown("---")
    analyze_btn = st.button(
        "🚀 Search & Analyze",
        type="primary",
        use_container_width=True,
    )

    # ── X.com API Settings ──
    st.markdown("---")
    with st.expander(
        "🔗 X.com (Twitter) API",
        expanded=("X.com (Twitter)" in platforms and not _has_twitter_creds()),
    ):
        st.caption("X.com tweets are fetched via the Twitter API v2. Get a Bearer Token from the [Twitter Developer Portal](https://developer.x.com/en/portal/dashboard). Stored locally in `.env`.")

        tw_token = st.text_input(
            "Bearer Token",
            value=st.session_state.get("twitter_bearer_token", ""),
            type="password",
            placeholder="AAAA...",
        )

        if st.button("💾 Save API Key", use_container_width=True):
            t = tw_token.strip()
            st.session_state["twitter_bearer_token"] = t
            save_env_key("TWITTER_BEARER_TOKEN", t)
            if t:
                st.success("Bearer Token saved!")
            else:
                st.info("Bearer Token cleared.")

        if _has_twitter_creds():
            st.markdown("✅ **API key set**")
        else:
            st.markdown("⚠️ **Not set** — add your Bearer Token above")

    # Info box
    with st.expander("ℹ️ Platform Notes"):
        st.markdown("""
**Reddit** — Works out of the box (no API key needed).

**X.com (Twitter)** — Requires a Bearer Token from the
[Twitter Developer Portal](https://developer.x.com/en/portal/dashboard).
Stored locally in `.env` and never shared.

**Amazon.in** — Works out of the box (web scraping).
May be rate-limited on heavy use.
        """)


# ── Main Area ─────────────────────────────────────────────────────
st.header("📊 Multi-Platform Sentiment Analyzer")

if not analyze_btn and "results" not in st.session_state:
    st.markdown("""
    **How it works:**
    1. Enter a keyword in the sidebar
    2. Select one or more platforms (X.com, Reddit, Amazon.in)
    3. Click **Search & Analyze**
    4. The app scrapes live data from the selected platforms and runs sentiment analysis

    Results include sentiment scores, charts, word clouds, and a downloadable data table.
    """)
    st.stop()


# ── Search & Analyze ──────────────────────────────────────────────
if analyze_btn:
    if not keyword or not keyword.strip():
        st.error("Please enter a keyword to search.")
        st.stop()
    if not platforms:
        st.error("Please select at least one platform.")
        st.stop()

    all_dfs = []
    progress_container = st.container()

    with progress_container:
        # Collection phase
        st.subheader("📡 Collecting Data...")
        collection_progress = st.progress(0, text="Starting...")

        for i, platform_display in enumerate(platforms):
            platform_key = PLATFORM_OPTIONS[platform_display]
            icon = PLATFORM_ICONS[platform_key]
            collection_progress.progress(
                (i) / len(platforms),
                text=f"{icon} Scraping {platform_display} for **{keyword}**...",
            )

            collector = get_collector(platform_key)
            df = collector.collect(query=keyword.strip(), limit=limit)

            if df.empty:
                st.warning(f"{icon} No results from {platform_display}. It may be blocked or rate-limited.")
            else:
                st.success(f"{icon} {platform_display}: collected **{len(df)}** items")
                all_dfs.append(df)

        collection_progress.progress(1.0, text="Collection complete!")

        if not all_dfs:
            st.error("No data collected from any platform. Try a different keyword or check the platform notes.")
            st.stop()

        combined = pd.concat(all_dfs, ignore_index=True)

        # Analysis phase
        st.subheader("🧠 Running Sentiment Analysis...")
        analysis_bar = st.progress(0, text="Loading sentiment models...")

        pipeline = load_pipeline(use_roberta)
        analysis_bar.progress(0.3, text="Analyzing text...")

        result = pipeline.analyze_dataframe(combined, show_progress=False)
        analysis_bar.progress(1.0, text="Analysis complete!")

    # Store in session state
    st.session_state["results"] = result
    st.session_state["keyword"] = keyword.strip()
    st.session_state["platforms_used"] = [PLATFORM_OPTIONS[p] for p in platforms]


# ── Display Results ───────────────────────────────────────────────
if "results" in st.session_state:
    df = st.session_state["results"]
    kw = st.session_state["keyword"]
    used_platforms = st.session_state.get("platforms_used", [])

    label_col = "final_label" if "final_label" in df.columns else "vader_label"
    score_col = "final_score" if "final_score" in df.columns else "vader_compound"

    from utils.chart_helpers import (
        sentiment_distribution_bar, sentiment_pie, sentiment_histogram,
        sentiment_timeline, vader_vs_roberta_scatter, platform_comparison_radar,
        sentiment_heatmap, confidence_distribution, text_length_vs_sentiment,
        volume_over_time, engagement_by_sentiment, top_texts_table,
        SENTIMENT_COLORS, PLATFORM_COLORS,
    )
    from utils.wordcloud_helper import display_sentiment_wordclouds
    import plotly.express as px
    import json

    st.markdown("---")
    st.header(f"📊 Results for \"{kw}\"")

    # ── Enhanced KPI Row (6 metrics) ──
    counts = df[label_col].value_counts(normalize=True) * 100 if label_col in df.columns else pd.Series()
    avg_score = df[score_col].mean() if score_col in df.columns else 0
    num_platforms = df["platform"].nunique()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("📝 Total Texts", f"{len(df):,}")
    k2.metric("✅ Positive", f"{counts.get('positive', 0):.1f}%")
    k3.metric("❌ Negative", f"{counts.get('negative', 0):.1f}%")
    k4.metric("➖ Neutral", f"{counts.get('neutral', 0):.1f}%")
    k5.metric("🎯 Avg Score", f"{avg_score:.3f}")
    k6.metric("🌐 Platforms", f"{num_platforms}")

    # ── Auto-generated Summary ──
    dominant = counts.idxmax() if not counts.empty else "N/A"
    dominant_pct = counts.max() if not counts.empty else 0
    summary_parts = [f"**Dominant sentiment: {dominant}** ({dominant_pct:.1f}% of texts)."]

    if num_platforms > 1:
        platform_dominant = df.groupby("platform")[label_col].agg(lambda x: x.value_counts().idxmax())
        comparisons = [f"{p}: {s}" for p, s in platform_dominant.items()]
        summary_parts.append(f"Platform breakdown — {', '.join(comparisons)}.")

    if "roberta_label" in df.columns and "vader_label" in df.columns:
        agreement = (df["vader_label"] == df["roberta_label"]).mean() * 100
        summary_parts.append(f"VADER & RoBERTa agree on {agreement:.0f}% of texts.")

    st.info("💡 " + " ".join(summary_parts))

    st.markdown("---")

    # ── Tabbed Layout ──
    tab_overview, tab_deep, tab_trends, tab_text, tab_data = st.tabs([
        "📊 Overview",
        "🔍 Deep Analysis",
        "📈 Trends & Time",
        "💬 Text Insights",
        "💾 Data & Export",
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — Overview
    # ════════════════════════════════════════════════════════════
    with tab_overview:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(
                sentiment_distribution_bar(df, label_col=label_col),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                sentiment_pie(df, label_col=label_col),
                use_container_width=True,
            )

        if num_platforms > 1:
            st.subheader("🛰️ Platform Comparison")
            st.plotly_chart(
                platform_comparison_radar(df, label_col=label_col),
                use_container_width=True,
            )

    # ════════════════════════════════════════════════════════════
    # TAB 2 — Deep Analysis
    # ════════════════════════════════════════════════════════════
    with tab_deep:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                sentiment_histogram(df, score_col=score_col),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                confidence_distribution(df, score_col=score_col, label_col=label_col),
                use_container_width=True,
            )

        # VADER vs RoBERTa scatter (only when both models were used)
        if "roberta_score" in df.columns and "vader_compound" in df.columns:
            st.subheader("🔄 VADER vs RoBERTa Comparison")
            col_c, col_d = st.columns([3, 1])
            with col_c:
                st.plotly_chart(
                    vader_vs_roberta_scatter(df),
                    use_container_width=True,
                )
            with col_d:
                agreement = (df["vader_label"] == df["roberta_label"]).mean() * 100
                st.metric("🤝 Agreement", f"{agreement:.1f}%")
                vader_pos = (df["vader_label"] == "positive").mean() * 100
                roberta_pos = (df["roberta_label"] == "positive").mean() * 100
                st.metric("VADER Positive %", f"{vader_pos:.1f}%")
                st.metric("RoBERTa Positive %", f"{roberta_pos:.1f}%")

        st.subheader("📏 Text Length Analysis")
        st.plotly_chart(
            text_length_vs_sentiment(df, label_col=label_col),
            use_container_width=True,
        )

    # ════════════════════════════════════════════════════════════
    # TAB 3 — Trends & Time
    # ════════════════════════════════════════════════════════════
    with tab_trends:
        has_dates = "date" in df.columns and df["date"].notna().any()

        if has_dates:
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
            freq_label = st.radio(
                "📅 Time Frequency", list(freq_map.keys()),
                horizontal=True, index=1,
            )
            freq = freq_map[freq_label]

            st.subheader("📉 Sentiment Timeline")
            st.plotly_chart(
                sentiment_timeline(df, score_col=score_col, freq=freq),
                use_container_width=True,
            )

            st.subheader("📊 Post Volume Over Time")
            st.plotly_chart(
                volume_over_time(df, label_col=label_col, freq=freq),
                use_container_width=True,
            )

            if num_platforms > 1:
                st.subheader("🗺️ Sentiment Heatmap")
                st.plotly_chart(
                    sentiment_heatmap(df, label_col=label_col),
                    use_container_width=True,
                )
        else:
            st.info("⏳ No date information available for time-based analysis.")

    # ════════════════════════════════════════════════════════════
    # TAB 4 — Text Insights
    # ════════════════════════════════════════════════════════════
    with tab_text:
        st.subheader("☁️ Word Clouds")
        display_sentiment_wordclouds(df, label_col=label_col)

        st.markdown("---")

        st.subheader("📝 Notable Texts")
        sub_pos, sub_neg = st.tabs(["✅ Most Positive", "❌ Most Negative"])
        with sub_pos:
            pos_df = top_texts_table(df, "positive", label_col=label_col, n=10)
            if not pos_df.empty:
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
            else:
                st.info("No positive texts found.")
        with sub_neg:
            neg_df = top_texts_table(df, "negative", label_col=label_col, n=10)
            if not neg_df.empty:
                st.dataframe(neg_df, use_container_width=True, hide_index=True)
            else:
                st.info("No negative texts found.")

        # Engagement analysis
        if "metadata" in df.columns:
            st.markdown("---")
            st.subheader("📈 Engagement Analysis")
            st.plotly_chart(
                engagement_by_sentiment(df, label_col=label_col),
                use_container_width=True,
            )

        # Top authors
        if "author" in df.columns:
            st.markdown("---")
            st.subheader("👤 Top Authors")
            author_counts = df["author"].value_counts().head(10).reset_index()
            author_counts.columns = ["Author", "Posts"]
            st.dataframe(author_counts, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════
    # TAB 5 — Data & Export
    # ════════════════════════════════════════════════════════════
    with tab_data:
        st.subheader("🔎 Filtered Data Table")

        # Filters
        filter_cols = st.columns(3)
        with filter_cols[0]:
            sent_filter = st.multiselect(
                "Sentiment", ["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"],
            )
        with filter_cols[1]:
            plat_options = df["platform"].unique().tolist()
            plat_filter = st.multiselect("Platform", plat_options, default=plat_options)
        with filter_cols[2]:
            if score_col in df.columns:
                score_min, score_max = float(df[score_col].min()), float(df[score_col].max())
                score_range = st.slider(
                    "Score Range", score_min, score_max, (score_min, score_max),
                )
            else:
                score_range = None

        # Apply filters
        filtered = df[
            (df[label_col].isin(sent_filter)) &
            (df["platform"].isin(plat_filter))
        ]
        if score_range and score_col in df.columns:
            filtered = filtered[
                (filtered[score_col] >= score_range[0]) &
                (filtered[score_col] <= score_range[1])
            ]

        st.caption(f"Showing {len(filtered):,} of {len(df):,} texts")

        display_cols = ["text", "platform", "author", "date", label_col]
        if score_col in df.columns:
            display_cols.append(score_col)
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

        # Downloads
        st.markdown("---")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📥 Download CSV",
                filtered.to_csv(index=False),
                file_name=f"sentiment_{kw.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            json_cols = [c for c in available_cols if c in filtered.columns]
            st.download_button(
                "📥 Download JSON",
                filtered[json_cols].to_json(orient="records", indent=2, default_handler=str),
                file_name=f"sentiment_{kw.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )
