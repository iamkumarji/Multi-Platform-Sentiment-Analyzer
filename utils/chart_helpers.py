import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Consistent color palette
SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
}

PLATFORM_COLORS = {
    "twitter": "#1DA1F2",
    "reddit": "#FF4500",
    "amazon": "#FF9900",
}


def sentiment_distribution_bar(df: pd.DataFrame, label_col: str = "final_label",
                                group_by: str = "platform") -> go.Figure:
    """Grouped bar chart of sentiment distribution per group."""
    if df.empty:
        return _empty_figure("No data available")

    counts = df.groupby([group_by, label_col]).size().reset_index(name="count")
    fig = px.bar(
        counts, x=group_by, y="count", color=label_col,
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution by Platform",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Count", legend_title="Sentiment")
    return fig


def sentiment_pie(df: pd.DataFrame, label_col: str = "final_label") -> go.Figure:
    """Donut chart of overall sentiment distribution."""
    if df.empty:
        return _empty_figure("No data available")

    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts, values="count", names="sentiment",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Overall Sentiment",
        hole=0.4,
    )
    return fig


def sentiment_timeline(df: pd.DataFrame, score_col: str = "final_score",
                        freq: str = "W", color_by: str = "platform") -> go.Figure:
    """Time-series line chart of average sentiment over time."""
    if df.empty or "date" not in df.columns:
        return _empty_figure("No data available")

    df_ts = df.dropna(subset=["date"]).copy()
    df_ts["period"] = df_ts["date"].dt.to_period(freq).dt.to_timestamp()

    agg = df_ts.groupby(["period", color_by])[score_col].mean().reset_index()
    fig = px.line(
        agg, x="period", y=score_col, color=color_by,
        color_discrete_map=PLATFORM_COLORS,
        title="Sentiment Trend Over Time",
        markers=True,
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Avg Sentiment Score", legend_title="")
    return fig


def sentiment_histogram(df: pd.DataFrame, score_col: str = "vader_compound") -> go.Figure:
    """Histogram of sentiment scores."""
    if df.empty or score_col not in df.columns:
        return _empty_figure("No data available")

    fig = px.histogram(
        df, x=score_col, nbins=50,
        color_discrete_sequence=["#3498db"],
        title="Sentiment Score Distribution",
    )
    fig.update_layout(xaxis_title="Sentiment Score", yaxis_title="Count")
    return fig


def vader_vs_roberta_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot comparing VADER and RoBERTa scores."""
    if df.empty or "vader_compound" not in df.columns or "roberta_score" not in df.columns:
        return _empty_figure("No data available")

    fig = px.scatter(
        df, x="vader_compound", y="roberta_score",
        color="final_label",
        color_discrete_map=SENTIMENT_COLORS,
        title="VADER vs RoBERTa Scores",
        opacity=0.5,
    )
    fig.update_layout(xaxis_title="VADER Compound", yaxis_title="RoBERTa Confidence")
    return fig


def platform_comparison_radar(df: pd.DataFrame, label_col: str = "final_label") -> go.Figure:
    """Radar chart comparing sentiment ratios across platforms."""
    if df.empty:
        return _empty_figure("No data available")

    platforms = df["platform"].unique()
    fig = go.Figure()

    for sentiment in ["positive", "negative", "neutral"]:
        values = []
        for platform in platforms:
            platform_df = df[df["platform"] == platform]
            if len(platform_df) > 0:
                ratio = (platform_df[label_col] == sentiment).sum() / len(platform_df)
                values.append(ratio * 100)
            else:
                values.append(0)
        values.append(values[0])  # close the radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(platforms) + [platforms[0]],
            fill="toself",
            name=sentiment,
            line_color=SENTIMENT_COLORS[sentiment],
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Platform Sentiment Comparison (%)",
    )
    return fig


def sentiment_heatmap(df: pd.DataFrame, label_col: str = "final_label") -> go.Figure:
    """Heatmap of positive sentiment % by platform and month."""
    if df.empty or "date" not in df.columns:
        return _empty_figure("No data available")

    df_h = df.dropna(subset=["date"]).copy()
    df_h["month"] = df_h["date"].dt.to_period("M").astype(str)
    df_h["is_positive"] = (df_h[label_col] == "positive").astype(int)

    pivot = df_h.groupby(["platform", "month"])["is_positive"].mean().reset_index()
    pivot_table = pivot.pivot(index="platform", columns="month", values="is_positive")

    fig = px.imshow(
        pivot_table.values * 100,
        labels=dict(x="Month", y="Platform", color="Positive %"),
        x=list(pivot_table.columns),
        y=list(pivot_table.index),
        color_continuous_scale="RdYlGn",
        title="Positive Sentiment % by Platform & Month",
    )
    return fig


def confidence_distribution(df: pd.DataFrame, score_col: str = "final_score",
                             label_col: str = "final_label") -> go.Figure:
    """Box plot of sentiment scores per sentiment category."""
    if df.empty or score_col not in df.columns or label_col not in df.columns:
        return _empty_figure("No data available")

    fig = px.box(
        df, x=label_col, y=score_col, color=label_col,
        color_discrete_map=SENTIMENT_COLORS,
        title="Confidence Distribution by Sentiment",
        points="outliers",
    )
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Score", showlegend=False)
    return fig


def text_length_vs_sentiment(df: pd.DataFrame, label_col: str = "final_label") -> go.Figure:
    """Violin plot of text length by sentiment category."""
    if df.empty or label_col not in df.columns:
        return _empty_figure("No data available")

    df_plot = df.copy()
    text_col = "clean_text" if "clean_text" in df_plot.columns else "text"
    df_plot["text_length"] = df_plot[text_col].astype(str).str.len()

    fig = px.violin(
        df_plot, x=label_col, y="text_length", color=label_col,
        color_discrete_map=SENTIMENT_COLORS,
        title="Text Length by Sentiment",
        box=True, points=False,
    )
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Character Count", showlegend=False)
    return fig


def volume_over_time(df: pd.DataFrame, label_col: str = "final_label",
                     freq: str = "D") -> go.Figure:
    """Stacked area chart of post count over time by sentiment."""
    if df.empty or "date" not in df.columns:
        return _empty_figure("No date data available")

    df_ts = df.dropna(subset=["date"]).copy()
    if df_ts.empty:
        return _empty_figure("No date data available")

    df_ts["period"] = df_ts["date"].dt.to_period(freq).dt.to_timestamp()
    counts = df_ts.groupby(["period", label_col]).size().reset_index(name="count")

    fig = px.area(
        counts, x="period", y="count", color=label_col,
        color_discrete_map=SENTIMENT_COLORS,
        title="Post Volume Over Time",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Posts", legend_title="Sentiment")
    return fig


def engagement_by_sentiment(df: pd.DataFrame, label_col: str = "final_label") -> go.Figure:
    """Bar chart of average engagement metrics per sentiment."""
    if df.empty or "metadata" not in df.columns:
        return _empty_figure("No engagement data available")

    df_eng = df.copy()
    # Extract engagement from metadata dict
    def _get_engagement(meta):
        if not isinstance(meta, dict):
            return 0
        return meta.get("score", 0) or meta.get("likes", 0) or meta.get("rating", 0) or 0

    df_eng["engagement"] = df_eng["metadata"].apply(_get_engagement)

    if df_eng["engagement"].sum() == 0:
        return _empty_figure("No engagement data available")

    agg = df_eng.groupby(label_col)["engagement"].mean().reset_index()
    agg.columns = ["sentiment", "avg_engagement"]

    fig = px.bar(
        agg, x="sentiment", y="avg_engagement", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Avg Engagement by Sentiment",
    )
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Avg Engagement", showlegend=False)
    return fig


def top_texts_table(df: pd.DataFrame, sentiment: str = "positive",
                     label_col: str = "final_label", n: int = 10) -> pd.DataFrame:
    """Return top N texts of a given sentiment."""
    if df.empty:
        return pd.DataFrame()
    filtered = df[df[label_col] == sentiment].copy()
    cols = ["text", "platform", "date", label_col]
    if "vader_compound" in filtered.columns:
        cols.append("vader_compound")
    available = [c for c in cols if c in filtered.columns]
    return filtered[available].head(n)


def _empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font_size=16)
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    return fig
