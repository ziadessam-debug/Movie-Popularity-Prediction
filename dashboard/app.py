"""
CineMetrics — Movie Popularity Prediction dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Reads only artefacts produced by the notebooks:
    data/processed/movies_processed.parquet   (notebook 01, PySpark)
    data/processed/genre_stats.parquet        (notebook 01, PySpark)
    data/processed/yearly_stats.parquet       (notebook 01, PySpark)
    models/best_model.pkl + model_metadata.json + model_comparison.csv
    models/test_predictions.parquet + feature_importance.csv   (notebook 03)
The model is never retrained here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

APP_TITLE = "Movie Popularity Prediction"
APP_TAGLINE = "Big Data & Machine Learning analytics on 45,000 movies"

st.set_page_config(page_title=APP_TITLE, page_icon=":material/movie:", layout="wide",
                   initial_sidebar_state="expanded")

# Design tokens (kept in one place; used by CSS and Plotly)
C = {
    "bg": "#0A0E17", "surface": "#111827", "card": "#151D2E", "border": "#232F48",
    "text": "#E7EBF3", "muted": "#8A96AE", "accent": "#F2B544", "accent_soft": "rgba(242,181,68,0.14)",
    "blue": "#5B8DEF", "teal": "#3CC6B4", "red": "#E5636B",
}
SEQ = ["#1B2540", "#27406E", "#3A62A8", "#5B8DEF", "#9DBBF5", "#F2B544"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LANG_NAMES = {
    "en": "English", "fr": "French", "it": "Italian", "ja": "Japanese", "de": "German", "es": "Spanish",
    "ru": "Russian", "hi": "Hindi", "ko": "Korean", "zh": "Chinese", "sv": "Swedish", "pt": "Portuguese",
    "cn": "Cantonese", "fi": "Finnish", "nl": "Dutch", "da": "Danish", "pl": "Polish", "tr": "Turkish",
    "ta": "Tamil", "cs": "Czech", "el": "Greek", "no": "Norwegian", "fa": "Persian", "hu": "Hungarian",
    "th": "Thai", "te": "Telugu", "ml": "Malayalam", "ar": "Arabic", "he": "Hebrew", "xx": "No language",
}


def lang_label(code: str) -> str:
    return f"{LANG_NAMES.get(code, code.upper())} ({code})"


# ════════════════════════════════════════════════════════════════════════════
# Styling
# ════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, label, input, textarea, button {{
    font-family: 'Inter', system-ui, sans-serif;
}}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(91,141,239,0.10), transparent 60%),
                     radial-gradient(900px 500px at -10% 110%, rgba(242,181,68,0.06), transparent 60%),
                     {C['bg']}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1380px; }}
h1, h2, h3, h4 {{ font-family: 'Sora', 'Inter', sans-serif; letter-spacing: -0.01em; color: {C['text']}; }}
[data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* hero */
.hero {{ padding: 4px 0 18px 0; border-bottom: 1px solid {C['border']}; margin-bottom: 22px; }}
.hero .eyebrow {{ color: {C['accent']}; font-size: .78rem; font-weight: 600; letter-spacing: .14em;
                  text-transform: uppercase; }}
.hero h1 {{ font-size: 2.05rem; margin: 6px 0 6px 0; font-weight: 700; }}
.hero p {{ color: {C['muted']}; font-size: .98rem; margin: 0; max-width: 900px; }}

/* KPI cards */
.kpi {{ background: linear-gradient(180deg, {C['card']}, {C['surface']}); border: 1px solid {C['border']};
        border-radius: 14px; padding: 16px 18px; height: 100%; }}
.kpi .label {{ color: {C['muted']}; font-size: .74rem; font-weight: 600; letter-spacing: .08em;
               text-transform: uppercase; }}
.kpi .value {{ color: {C['text']}; font-family: 'Sora', sans-serif; font-size: 1.65rem; font-weight: 700;
               margin-top: 6px; line-height: 1.15; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.kpi .sub {{ color: {C['muted']}; font-size: .8rem; margin-top: 4px; }}
.kpi.accent {{ border-color: rgba(242,181,68,.45); box-shadow: 0 0 0 1px rgba(242,181,68,.10) inset; }}
.kpi.accent .value {{ color: {C['accent']}; }}

/* section header + insight */
.section {{ margin: 26px 0 10px 0; }}
.section h3 {{ font-size: 1.12rem; margin: 0; }}
.section p {{ color: {C['muted']}; font-size: .86rem; margin: 2px 0 0 0; }}
.insight {{ background: {C['accent_soft']}; border-left: 3px solid {C['accent']}; border-radius: 8px;
            padding: 10px 14px; color: {C['text']}; font-size: .9rem; margin: 6px 0; }}
.insight b {{ color: {C['accent']}; }}
.note {{ color: {C['muted']}; font-size: .8rem; }}

/* movie card */
.movie {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 14px; padding: 18px 20px; }}
.movie h4 {{ margin: 0 0 4px 0; font-size: 1.2rem; }}
.movie .meta {{ color: {C['muted']}; font-size: .85rem; margin-bottom: 10px; }}
.chip {{ display: inline-block; background: rgba(91,141,239,.14); color: #BBD0FA; border-radius: 999px;
         padding: 2px 10px; font-size: .75rem; margin: 0 6px 6px 0; }}

/* prediction result */
.result {{ background: linear-gradient(135deg, rgba(242,181,68,.16), rgba(91,141,239,.10));
           border: 1px solid rgba(242,181,68,.4); border-radius: 16px; padding: 22px 24px; }}
.result .label {{ color: {C['muted']}; font-size: .78rem; letter-spacing: .1em; text-transform: uppercase; font-weight: 600; }}
.result .big {{ font-family: 'Sora', sans-serif; font-size: 3rem; font-weight: 700; color: {C['accent']}; line-height: 1.1; }}
.result .range {{ color: {C['text']}; font-size: .92rem; }}

div[data-testid="stTabs"] button p {{ font-weight: 600; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button {{
    background: {C['accent']}; color: #1A1203; border: none; font-weight: 700; border-radius: 10px; }}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {{ background: #F6C566; color: #1A1203; }}
.sidebar-brand {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.15rem; color: {C['text']}; }}
.sidebar-brand span {{ color: {C['accent']}; }}
</style>""", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, legend: bool = False) -> go.Figure:
    """Apply the shared dark cinematic Plotly look."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C["text"], size=12),
        title_font=dict(family="Sora, sans-serif", size=14, color=C["text"]),
        showlegend=legend, legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
        hoverlabel=dict(bgcolor=C["card"], bordercolor=C["border"], font_color=C["text"]),
        colorway=[C["blue"], C["accent"], C["teal"], C["red"], "#A78BFA", "#9DBBF5"],
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"], linecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"], linecolor=C["border"])
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Data layer (cached; separated from UI)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading processed movie data…")
def load_movies() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "movies_processed.parquet")
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["budget_musd"] = df["budget"] / 1e6
    df["revenue_musd"] = df["revenue"] / 1e6
    df["genres_list"] = df["genres_str"].fillna("").str.split("|").apply(lambda g: [x for x in g if x])
    return df


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / name)


@st.cache_resource(show_spinner="Loading trained model…")
def load_model():
    import joblib
    return joblib.load(MODEL_DIR / "best_model.pkl")


@st.cache_data
def load_model_artifacts() -> dict:
    return {
        "meta": json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8")),
        "comparison": pd.read_csv(MODEL_DIR / "model_comparison.csv"),
        "importance": pd.read_csv(MODEL_DIR / "feature_importance.csv"),
        "predictions": pd.read_parquet(MODEL_DIR / "test_predictions.parquet"),
    }


def model_status() -> tuple[bool, str]:
    needed = ["best_model.pkl", "model_metadata.json", "model_comparison.csv",
              "feature_importance.csv", "test_predictions.parquet"]
    missing = [f for f in needed if not (MODEL_DIR / f).exists()]
    if missing:
        return False, "Missing model files: " + ", ".join(missing) + ". Run notebooks/03_model_training.ipynb."
    return True, ""


def build_feature_row(meta: dict, *, budget_usd: float | None, runtime: float, year: int, month: int,
                      language: str, genres: list[str], has_collection: bool, num_companies: int,
                      num_countries: int, is_us_production: bool) -> pd.DataFrame:
    """Recreate the exact feature definitions of notebook 01 for one movie, in training column order."""
    known = budget_usd is not None and budget_usd >= 1000
    row = {
        "log_budget": float(np.log1p(budget_usd)) if known else 0.0,
        "budget_known": int(known),
        "runtime": float(runtime), "release_year": int(year), "release_month": int(month),
        "num_genres": len(genres), "has_collection": int(has_collection),
        "num_companies": int(num_companies), "num_countries": int(num_countries),
        "is_us_production": int(is_us_production), "original_language": language,
    }
    for g, col in zip(meta["genres"], meta["genre_columns"]):
        row[col] = int(g in genres)
    return pd.DataFrame([row])[meta["feature_order"]]


# ════════════════════════════════════════════════════════════════════════════
# Reusable UI components
# ════════════════════════════════════════════════════════════════════════════
def hero(eyebrow: str, title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p></div>',
                unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "", accent: bool = False) -> None:
    size = ' style="font-size:1.2rem;white-space:normal"' if len(value) > 12 else ""
    col.markdown(f'<div class="kpi{" accent" if accent else ""}"><div class="label">{label}</div>'
                 f'<div class="value" title="{value}"{size}>{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def section(title: str, text: str = "") -> None:
    st.markdown(f'<div class="section"><h3>{title}</h3>{f"<p>{text}</p>" if text else ""}</div>',
                unsafe_allow_html=True)


def insight(html: str) -> None:
    st.markdown(f'<div class="insight">{html}</div>', unsafe_allow_html=True)


def fmt_money(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}"


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    return df.explode("genres_list").rename(columns={"genres_list": "genre"}).dropna(subset=["genre"])


def year_genre_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Sidebar filters shared by the analytical pages."""
    with st.sidebar:
        st.markdown("#### Filters")
        y0, y1 = int(df.release_year.min()), int(df.release_year.max())
        years = st.slider("Release year", y0, y1, (1970, y1), key=f"{key}_years")
        all_genres = sorted({g for gs in df.genres_list for g in gs})
        genres = st.multiselect("Genres (any of)", all_genres, key=f"{key}_genres", placeholder="All genres")
    out = df[df.release_year.between(*years)]
    if genres:
        out = out[out.genres_list.apply(lambda gs: any(g in gs for g in genres))]
    return out


# ════════════════════════════════════════════════════════════════════════════
# Page 1 — Overview
# ════════════════════════════════════════════════════════════════════════════
def page_overview() -> None:
    df_all = load_movies()
    hero("Overview", APP_TITLE,
         "Movie metadata from TMDB (The Movies Dataset, Kaggle) is cleaned and transformed with Apache Spark, "
         "explored, and used to train regression models that estimate a movie's TMDB popularity score from "
         "information available before release.")
    df = year_genre_filters(df_all, "ov")
    if df.empty:
        st.warning("No movies match the current filters. Widen the year range or clear the genre filter.")
        return

    g = explode_genres(df)
    top_genre = g.genre.value_counts()
    best = df.loc[df.popularity.idxmax()]
    cols = st.columns(5)
    kpi(cols[0], "Total movies", f"{len(df):,}", f"of {len(df_all):,} in dataset")
    kpi(cols[1], "Average popularity", f"{df.popularity.mean():.2f}", f"median {df.popularity.median():.2f}")
    kpi(cols[2], "Highest popularity", f"{best.popularity:.1f}", f"{best.title} ({best.release_year})", accent=True)
    kpi(cols[3], "Average budget", fmt_money(df.budget.mean()), f"{df.budget_known.mean()*100:.0f}% report a budget")
    kpi(cols[4], "Most common genre", top_genre.index[0], f"{top_genre.iloc[0]:,} movies")

    left, right = st.columns([1.15, 1])
    with left:
        section("Release trend", "Number of movies released per year and median popularity")
        yr = df.groupby("release_year").agg(movies=("id", "size"), median_pop=("popularity", "median")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=yr.release_year, y=yr.movies, name="Movies released", marker_color=C["blue"], opacity=.85)
        fig.add_scatter(x=yr.release_year, y=yr.median_pop, name="Median popularity", yaxis="y2",
                        line=dict(color=C["accent"], width=2))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="median popularity"),
                          yaxis_title="movies")
        st.plotly_chart(style_fig(fig, 360, legend=True), width="stretch")
    with right:
        section("Popularity distribution", "log scale — most movies have a low score, a few are blockbusters")
        fig = px.histogram(df, x="log_popularity", nbins=70, color_discrete_sequence=[C["blue"]])
        fig.update_layout(xaxis_title="log(1 + popularity)", yaxis_title="movies", bargap=0.03)
        st.plotly_chart(style_fig(fig, 360), width="stretch")

    section("Popularity by genre", "Median TMDB popularity and number of movies per genre")
    gg = g.groupby("genre").agg(movies=("id", "size"), median_pop=("popularity", "median")).reset_index()
    gg = gg[gg.movies >= 20].sort_values("median_pop")
    fig = px.bar(gg, x="median_pop", y="genre", orientation="h", color="median_pop",
                 color_continuous_scale=SEQ, hover_data={"movies": ":,"},
                 labels={"median_pop": "median popularity", "genre": ""})
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style_fig(fig, 470), width="stretch")

    section("Highlights")
    a, b = st.columns(2)
    with a:
        if len(gg):
            insight(f"<b>{gg.iloc[-1].genre}</b> has the highest median popularity "
                    f"({gg.iloc[-1].median_pop:.2f}) while <b>{gg.iloc[0].genre}</b> has the lowest "
                    f"({gg.iloc[0].median_pop:.2f}).")
        kb = df[df.budget_known == 1]
        if len(kb) > 30:
            rho = kb[["budget", "popularity"]].corr(method="spearman").iloc[0, 1]
            insight(f"Budget and popularity have a Spearman correlation of <b>{rho:.2f}</b> among "
                    f"{len(kb):,} movies with a known budget.")
    with b:
        mc = df.groupby("has_collection").popularity.median()
        if len(mc) == 2:
            insight(f"Franchise / collection movies have a median popularity of <b>{mc[1]:.2f}</b> "
                    f"vs <b>{mc[0]:.2f}</b> for standalone films.")
        insight(f"<b>{(df.original_language == 'en').mean()*100:.0f}%</b> of the selected movies were "
                f"originally produced in English.")


# ════════════════════════════════════════════════════════════════════════════
# Page 2 — Movie Explorer
# ════════════════════════════════════════════════════════════════════════════
def page_explorer() -> None:
    df = load_movies()
    hero("Movie Explorer", "Find and compare movies",
         "Search by title and combine filters. The table is sortable; pick a movie below it to see its details.")

    c1, c2, c3 = st.columns([2, 1.4, 1.4])
    query = c1.text_input("Search title", placeholder="e.g. Inception, Toy Story…")
    all_genres = sorted({g for gs in df.genres_list for g in gs})
    genres = c2.multiselect("Genre", all_genres, placeholder="Any genre")
    lang_counts = df.original_language.value_counts()
    langs = c3.multiselect("Language", lang_counts.index.tolist(), format_func=lang_label, placeholder="Any language")

    c4, c5, c6 = st.columns([2, 2, 1.4])
    years = c4.slider("Release year", int(df.release_year.min()), int(df.release_year.max()),
                      (int(df.release_year.min()), int(df.release_year.max())))
    max_b = float(np.ceil(df.budget_musd.max()))
    budget = c5.slider("Budget (million USD)", 0.0, max_b, (0.0, max_b), step=1.0)
    sort_by = c6.selectbox("Sort by", ["Popularity (high → low)", "Popularity (low → high)",
                                       "Release year (newest)", "Budget (high → low)", "Title (A → Z)"])
    include_unknown = st.checkbox("Include movies with unknown budget", value=budget == (0.0, max_b),
                                  help="About three quarters of movies do not report a budget.")

    f = df[df.release_year.between(*years)]
    if query.strip():
        f = f[f.title.str.contains(query.strip(), case=False, na=False, regex=False)]
    if genres:
        f = f[f.genres_list.apply(lambda gs: any(g in gs for g in genres))]
    if langs:
        f = f[f.original_language.isin(langs)]
    in_range = f.budget_musd.between(*budget)
    f = f[in_range | (f.budget.isna() & include_unknown)]

    order = {"Popularity (high → low)": ("popularity", False), "Popularity (low → high)": ("popularity", True),
             "Release year (newest)": ("release_date", False), "Budget (high → low)": ("budget", False),
             "Title (A → Z)": ("title", True)}[sort_by]
    f = f.sort_values(order[0], ascending=order[1], na_position="last")

    cols = st.columns(4)
    kpi(cols[0], "Matching movies", f"{len(f):,}")
    kpi(cols[1], "Median popularity", f"{f.popularity.median():.2f}" if len(f) else "—")
    kpi(cols[2], "Median budget", fmt_money(f.budget.median()) if f.budget.notna().any() else "—")
    kpi(cols[3], "Year span", f"{f.release_year.min()}–{f.release_year.max()}" if len(f) else "—")

    if f.empty:
        st.info("No movies match these filters. Try a shorter search term or remove a filter.")
        return

    st.write("")
    show = f[["title", "release_year", "genres_str", "original_language", "budget_musd", "runtime",
              "popularity", "vote_average"]].head(1000)
    st.dataframe(
        show, width="stretch", hide_index=True, height=420,
        column_config={
            "title": st.column_config.TextColumn("Title", width="large"),
            "release_year": st.column_config.NumberColumn("Year", format="%d"),
            "genres_str": st.column_config.TextColumn("Genres"),
            "original_language": st.column_config.TextColumn("Lang"),
            "budget_musd": st.column_config.NumberColumn("Budget ($M)", format="%.1f"),
            "runtime": st.column_config.NumberColumn("Runtime", format="%d min"),
            "popularity": st.column_config.ProgressColumn("Popularity", format="%.2f", min_value=0,
                                                          max_value=float(df.popularity.quantile(0.999))),
            "vote_average": st.column_config.NumberColumn("TMDB rating", format="%.1f"),
        })
    if len(f) > 1000:
        st.markdown(f'<span class="note">Showing the first 1,000 of {len(f):,} movies.</span>',
                    unsafe_allow_html=True)

    section("Movie details")
    options = f.head(300)
    pick = st.selectbox("Select a movie", options.index,
                        format_func=lambda i: f"{options.at[i, 'title']} ({options.at[i, 'release_year']})")
    m = df.loc[pick]
    pct = (df.popularity < m.popularity).mean() * 100
    chips = "".join(f'<span class="chip">{g}</span>' for g in m.genres_list)
    overview = (m.overview or "No overview available.") if isinstance(m.overview, str) else "No overview available."
    st.markdown(f"""<div class="movie"><h4>{m.title}</h4>
      <div class="meta">{m.release_date:%d %b %Y} · {lang_label(m.original_language)} · {int(m.runtime)} min ·
      Budget {fmt_money(m.budget)} · Revenue {fmt_money(m.revenue)}</div>{chips}
      <p style="color:{C['text']};font-size:.92rem;margin-top:8px">{overview}</p>
      <div class="meta">Popularity <b style="color:{C['accent']}">{m.popularity:.2f}</b> — higher than
      {pct:.1f}% of movies in the dataset · TMDB rating {m.vote_average if pd.notna(m.vote_average) else '—'}
      ({int(m.vote_count) if pd.notna(m.vote_count) else 0:,} votes)</div></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Page 3 — Popularity Prediction
# ════════════════════════════════════════════════════════════════════════════
def page_predict() -> None:
    hero("Popularity Prediction", "Estimate a movie's popularity",
         "Describe a movie using information available before its release. The trained model returns the "
         "expected TMDB popularity score and how it compares with the movies in the dataset.")
    ok, msg = model_status()
    if not ok:
        st.error(msg)
        return
    try:
        model, art = load_model(), load_model_artifacts()
    except Exception as exc:  # corrupted file / version mismatch
        st.error(f"The model could not be loaded ({type(exc).__name__}: {exc}). Re-run notebook 03 with the "
                 f"package versions in requirements.txt.")
        return
    meta, df = art["meta"], load_movies()

    form_col, out_col = st.columns([1, 1.15], gap="large")
    with form_col:
        with st.form("predict"):
            st.markdown("##### Movie profile")
            genres = st.multiselect("Genres", meta["genres"], default=["Action", "Adventure"],
                                    help="Choose one to five genres.")
            a, b = st.columns(2)
            budget_unknown = a.checkbox("Budget unknown", value=False)
            budget_m = b.number_input("Budget (million USD)", min_value=0.0, max_value=500.0, value=60.0,
                                      step=5.0, disabled=budget_unknown)
            runtime = st.slider("Runtime (minutes)", 40, 240, 115)
            c, d = st.columns(2)
            year = c.number_input("Release year", min_value=1920, max_value=2030, value=2016, step=1)
            month = d.selectbox("Release month", range(1, 13), index=6, format_func=lambda m: MONTHS[m - 1])
            langs = meta["languages"]
            language = st.selectbox("Original language", langs, index=langs.index("en") if "en" in langs else 0,
                                    format_func=lang_label)
            e, f_ = st.columns(2)
            companies = e.number_input("Production companies", 0, 20, 2)
            countries = f_.number_input("Production countries", 0, 10, 1)
            g, h = st.columns(2)
            collection = g.toggle("Part of a franchise", value=False)
            us = h.toggle("US production", value=True)
            submitted = st.form_submit_button("Predict popularity", width="stretch")

    with out_col:
        if not submitted:
            st.markdown('<div class="movie"><h4>How it works</h4><div class="meta">The form values are turned '
                        'into exactly the same features used in training (log budget, budget-known flag, genre '
                        'flags, release year/month, language…) and passed to the saved scikit-learn pipeline '
                        f'(<b>{meta["best_model"]}</b>). Fill in the form and press <b>Predict popularity</b>.'
                        '</div></div>', unsafe_allow_html=True)
            return
        errors = []
        if not genres:
            errors.append("Select at least one genre.")
        if len(genres) > 5:
            errors.append("Select at most five genres.")
        if not budget_unknown and budget_m <= 0:
            errors.append("Enter a budget above 0 or tick 'Budget unknown'.")
        if errors:
            for e_ in errors:
                st.error(e_)
            return

        x = build_feature_row(meta, budget_usd=None if budget_unknown else budget_m * 1e6, runtime=runtime,
                              year=year, month=month, language=language, genres=genres,
                              has_collection=collection, num_companies=companies, num_countries=countries,
                              is_us_production=us)
        pred_log = float(model.predict(x)[0])
        pred = float(np.expm1(pred_log))
        rmse_log = meta["test_metrics"]["RMSE (log)"]
        lo, hi = float(np.expm1(max(pred_log - rmse_log, 0))), float(np.expm1(pred_log + rmse_log))
        pct = (df.popularity < pred).mean() * 100

        st.markdown(f"""<div class="result"><div class="label">Predicted popularity score</div>
            <div class="big">{pred:.2f}</div>
            <div class="range">Typical range given the model's test error: <b>{lo:.2f} – {hi:.2f}</b></div>
            <div class="range" style="margin-top:6px">Higher than <b>{pct:.0f}%</b> of the {len(df):,} movies
            in the dataset.</div></div>""", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pct, number=dict(suffix=" pct", font=dict(color=C["text"], size=30)),
            title=dict(text="Percentile within dataset", font=dict(size=13, color=C["muted"])),
            gauge=dict(axis=dict(range=[0, 100], tickcolor=C["muted"]), bar=dict(color=C["accent"]),
                       bgcolor=C["card"], bordercolor=C["border"],
                       steps=[dict(range=[0, 50], color="#151D2E"), dict(range=[50, 90], color="#1B2540"),
                              dict(range=[90, 100], color="#27406E")])))
        st.plotly_chart(style_fig(fig, 240), width="stretch")

        fig = px.histogram(df, x="log_popularity", nbins=70, color_discrete_sequence=[C["blue"]])
        fig.add_vline(x=pred_log, line_color=C["accent"], line_width=3,
                      annotation_text="your movie", annotation_font_color=C["accent"])
        fig.update_layout(xaxis_title="log(1 + popularity)", yaxis_title="movies", bargap=0.03,
                          title="Where the prediction falls in the dataset")
        st.plotly_chart(style_fig(fig, 280), width="stretch")

        if year > 2017:
            st.warning("The training data ends in 2017 — predictions for later years are extrapolations.")
        insight(f"Interpretation: the model expects this movie to be "
                f"{'more' if pct >= 50 else 'less'} popular than the typical movie in the dataset. "
                f"TMDB popularity is a 2017 activity snapshot and the model explains only part of its variance "
                f"(test R² on log scale = <b>{meta['test_metrics']['R² (log)']:.2f}</b>), so treat the score as "
                f"a relative estimate, not a guarantee.")

        similar = df[(df.primary_genre == genres[0])]
        if not budget_unknown:
            similar = similar[similar.budget.between(budget_m * 0.6e6, budget_m * 1.4e6)]
        else:
            similar = similar[similar.budget_known == 0]
        if len(similar):
            section("Comparable movies", f"Same primary genre ({genres[0]}) and similar budget")
            st.dataframe(similar.sort_values("popularity", ascending=False)
                         [["title", "release_year", "budget_musd", "popularity"]].head(8),
                         hide_index=True, width="stretch",
                         column_config={"title": st.column_config.TextColumn("Title"),
                                        "budget_musd": st.column_config.NumberColumn("Budget ($M)", format="%.1f"),
                                        "release_year": st.column_config.NumberColumn("Year", format="%d"),
                                        "popularity": st.column_config.NumberColumn("Popularity", format="%.2f")})


# ════════════════════════════════════════════════════════════════════════════
# Page 4 — Data Insights
# ════════════════════════════════════════════════════════════════════════════
def page_insights() -> None:
    df_all = load_movies()
    hero("Data Insights", "What is associated with popularity?",
         "Interactive exploratory analysis of the processed dataset. Use the sidebar filters to focus on a "
         "period or a set of genres; every chart and finding updates.")
    df = year_genre_filters(df_all, "in")
    if len(df) < 30:
        st.warning("Too few movies match the filters for a meaningful analysis.")
        return
    st.markdown(f'<span class="note">{len(df):,} movies selected</span>', unsafe_allow_html=True)
    tabs = st.tabs(["Genres", "Budget", "Release trends", "Runtime", "Language", "Correlation"])

    with tabs[0]:
        g = explode_genres(df)
        gg = g.groupby("genre").agg(movies=("id", "size"), median_pop=("popularity", "median"),
                                    avg_budget=("budget_musd", "mean")).reset_index()
        gg = gg[gg.movies >= 20]
        metric = st.radio("Metric", ["Median popularity", "Number of movies", "Average budget ($M)"],
                          horizontal=True, key="genre_metric")
        col = {"Median popularity": "median_pop", "Number of movies": "movies",
               "Average budget ($M)": "avg_budget"}[metric]
        fig = px.bar(gg.sort_values(col), x=col, y="genre", orientation="h", color=col,
                     color_continuous_scale=SEQ, labels={col: metric, "genre": ""})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig, 480), width="stretch")
        top = gg.sort_values("median_pop").iloc[-1]
        insight(f"<b>{top.genre}</b> leads with a median popularity of {top.median_pop:.2f} across "
                f"{int(top.movies):,} movies in the selection.")
        order = gg.sort_values("median_pop", ascending=False).genre.tolist()
        fig = px.box(g[g.genre.isin(order)], x="genre", y="log_popularity", category_orders={"genre": order},
                     points=False, color_discrete_sequence=[C["blue"]], labels={"genre": ""},
                     title="Spread of log popularity per genre")
        st.plotly_chart(style_fig(fig, 400), width="stretch")

    with tabs[1]:
        kb = df[df.budget_known == 1]
        a, b = st.columns(2)
        with a:
            fig = px.scatter(kb, x="budget_musd", y="popularity", log_x=True, log_y=True, opacity=.45,
                             hover_name="title", hover_data={"release_year": True},
                             color_discrete_sequence=[C["blue"]], title="Budget vs popularity (log-log)",
                             labels={"budget_musd": "budget ($M)"})
            st.plotly_chart(style_fig(fig, 420), width="stretch")
        with b:
            tiers = pd.cut(df.budget_musd, [0, 1, 10, 50, 100, 1e5], labels=["<1M", "1-10M", "10-50M", "50-100M", "≥100M"])
            tdf = df.assign(tier=tiers.cat.add_categories("unknown").fillna("unknown"))
            fig = px.box(tdf, x="tier", y="log_popularity", points=False, title="log popularity by budget tier",
                         category_orders={"tier": ["unknown", "<1M", "1-10M", "10-50M", "50-100M", "≥100M"]},
                         color_discrete_sequence=[C["accent"]], labels={"tier": "budget tier"})
            st.plotly_chart(style_fig(fig, 420), width="stretch")
        if len(kb) > 30:
            rho = kb[["budget", "popularity"]].corr(method="spearman").iloc[0, 1]
            insight(f"Spearman correlation between budget and popularity: <b>{rho:.2f}</b>. Only "
                    f"{len(kb)/len(df)*100:.0f}% of the selected movies report a budget; those that do have a median "
                    f"popularity of {kb.popularity.median():.2f} vs {df[df.budget_known == 0].popularity.median():.2f} "
                    f"for unknown budgets.")

    with tabs[2]:
        yr = df.groupby("release_year").agg(movies=("id", "size"), median_pop=("popularity", "median"),
                                            avg_budget=("budget_musd", "mean")).reset_index()
        fig = px.area(yr, x="release_year", y="movies", title="Movies released per year",
                      color_discrete_sequence=[C["blue"]], labels={"release_year": ""})
        st.plotly_chart(style_fig(fig, 320), width="stretch")
        a, b = st.columns(2)
        with a:
            mo = df.groupby("release_month").popularity.median().reindex(range(1, 13)).reset_index()
            mo["month"] = [MONTHS[i - 1] for i in mo.release_month]
            fig = px.bar(mo, x="month", y="popularity", title="Median popularity by release month",
                         color_discrete_sequence=[C["accent"]], labels={"popularity": "median popularity", "month": ""})
            st.plotly_chart(style_fig(fig, 340), width="stretch")
        with b:
            dec = df.groupby("release_decade").popularity.median().reset_index()
            fig = px.line(dec, x="release_decade", y="popularity", markers=True,
                          title="Median popularity by decade", color_discrete_sequence=[C["teal"]],
                          labels={"popularity": "median popularity", "release_decade": "decade"})
            st.plotly_chart(style_fig(fig, 340), width="stretch")
        best_m = mo.loc[mo.popularity.idxmax()]
        insight(f"<b>{best_m.month}</b> releases have the highest median popularity "
                f"({best_m.popularity:.2f}) in this selection.")

    with tabs[3]:
        a, b = st.columns(2)
        with a:
            fig = px.histogram(df, x="runtime", nbins=80, range_x=[30, 240], title="Runtime distribution",
                               color_discrete_sequence=[C["blue"]], labels={"runtime": "runtime (min)"})
            st.plotly_chart(style_fig(fig, 360), width="stretch")
        with b:
            bins = pd.cut(df.runtime, [0, 60, 80, 90, 100, 110, 120, 140, 160, 400])
            rt = df.groupby(bins, observed=True).popularity.median().reset_index()
            rt["runtime"] = rt.runtime.astype(str)
            fig = px.bar(rt, x="runtime", y="popularity", title="Median popularity by runtime bucket",
                         color_discrete_sequence=[C["accent"]], labels={"popularity": "median popularity"})
            st.plotly_chart(style_fig(fig, 360), width="stretch")
        rho = df[["runtime", "popularity"]].corr(method="spearman").iloc[0, 1]
        insight(f"Median runtime is <b>{df.runtime.median():.0f} min</b>; Spearman correlation with popularity "
                f"is <b>{rho:.2f}</b>. Very short films (often shorts and TV specials) are the least popular.")

    with tabs[4]:
        lg = df.groupby("original_language").agg(movies=("id", "size"), median_pop=("popularity", "median")).reset_index()
        lg = lg[lg.movies >= 50].sort_values("movies", ascending=False).head(15)
        lg["language"] = lg.original_language.map(lang_label)
        fig = px.scatter(lg, x="movies", y="median_pop", size="movies", text="original_language", log_x=True,
                         hover_name="language", color_discrete_sequence=[C["accent"]],
                         title="Languages: volume vs median popularity (≥50 movies)",
                         labels={"median_pop": "median popularity", "movies": "movies (log)"})
        fig.update_traces(textposition="top center", marker=dict(line=dict(width=0)))
        st.plotly_chart(style_fig(fig, 440), width="stretch")

    with tabs[5]:
        cols = ["log_popularity", "log_budget", "budget_known", "runtime", "release_year", "release_month",
                "num_genres", "has_collection", "num_companies", "num_countries", "is_us_production",
                "vote_count", "vote_average"]
        corr = df[cols].corr(method="spearman").round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale=["#5B8DEF", "#151D2E", "#F2B544"],
                        zmin=-1, zmax=1, aspect="auto", title="Spearman correlation matrix")
        st.plotly_chart(style_fig(fig, 560), width="stretch")
        c = corr["log_popularity"].drop(["log_popularity", "vote_count", "vote_average"]).sort_values(ascending=False)
        insight(f"Strongest pre-release correlates: " + ", ".join(f"<b>{k}</b> ({v:.2f})" for k, v in c.head(3).items())
                + f". <b>vote_count</b> (post-release, {corr.loc['log_popularity', 'vote_count']:.2f}) is excluded "
                  f"from the model to avoid data leakage.")


# ════════════════════════════════════════════════════════════════════════════
# Page 5 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
def page_performance() -> None:
    hero("Model Performance", "How well does the model predict popularity?",
         "Five regression approaches were tuned with 3-fold cross-validation on the training set and evaluated "
         "once on a held-out 20% test set. The best model is selected by cross-validated RMSE.")
    ok, msg = model_status()
    if not ok:
        st.error(msg)
        return
    art = load_model_artifacts()
    meta, comp, imp, preds = art["meta"], art["comparison"], art["importance"], art["predictions"]
    tm = meta["test_metrics"]

    cols = st.columns(4)
    kpi(cols[0], "Best model", meta["best_model"], f"{meta['n_train']:,} train / {meta['n_test']:,} test", accent=True)
    kpi(cols[1], "R² (log scale)", f"{tm['R² (log)']:.3f}", "share of variance explained")
    kpi(cols[2], "MAE (log scale)", f"{tm['MAE (log)']:.3f}", f"≈ ×{np.exp(tm['MAE (log)']):.2f} typical factor")
    kpi(cols[3], "RMSE (log scale)", f"{tm['RMSE (log)']:.3f}", f"original scale RMSE {tm['RMSE']:.2f}")

    section("Model comparison", "Test-set metrics. Log scale = what the models optimise; original scale = raw popularity.")
    show = comp.drop(columns=["Selected"]).copy()
    show.insert(0, " ", np.where(comp.Model == meta["best_model"], "BEST", ""))
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in show.columns
                                if c not in (" ", "Model", "Train time (s)")})

    metric = st.radio("Compare models on", ["R² (log)", "MAE (log)", "RMSE (log)", "MAE", "RMSE", "R²"],
                      horizontal=True)
    cdf = comp[comp.Model != "Baseline (mean)"].copy() if metric.startswith("R²") else comp.copy()
    cdf["is_best"] = np.where(cdf.Model == meta["best_model"], "best", "other")
    fig = px.bar(cdf.sort_values(metric), x=metric, y="Model", orientation="h", color="is_best",
                 color_discrete_map={"best": C["accent"], "other": C["blue"]}, text_auto=".3f", labels={"Model": ""})
    st.plotly_chart(style_fig(fig, 300), width="stretch")

    a, b = st.columns(2)
    with a:
        section("Actual vs predicted", "Held-out test movies, log(1 + popularity)")
        models = [c.replace("pred_log__", "") for c in preds.columns if c.startswith("pred_log__")
                  and "Baseline" not in c]
        chosen = st.selectbox("Model", models, index=models.index(meta["best_model"]) if meta["best_model"] in models else 0)
        p = preds.assign(predicted=preds[f"pred_log__{chosen}"])
        fig = px.scatter(p, x="actual_log", y="predicted", opacity=.35, hover_name="title",
                         color_discrete_sequence=[C["blue"]],
                         labels={"actual_log": "actual", "predicted": "predicted"})
        m = float(max(p.actual_log.max(), p.predicted.max()))
        fig.add_shape(type="line", x0=0, y0=0, x1=m, y1=m, line=dict(color=C["accent"], dash="dash"))
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    with b:
        section("Residuals", "actual − predicted (log scale); centred on 0 = unbiased")
        fig = px.histogram(p.assign(residual=p.actual_log - p.predicted), x="residual", nbins=70,
                           color_discrete_sequence=[C["teal"]])
        fig.add_vline(x=0, line_color=C["accent"], line_dash="dash")
        st.plotly_chart(style_fig(fig, 420 + 62), width="stretch")

    section("Feature importance", f"Permutation importance of the {meta['best_model']}: increase in RMSE when a "
                                  f"feature is randomly shuffled")
    top = imp.head(15).sort_values("importance")
    fig = px.bar(top, x="importance", y="feature", orientation="h", error_x="std", color="importance",
                 color_continuous_scale=SEQ, labels={"feature": "", "importance": "RMSE increase"})
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style_fig(fig, 480), width="stretch")
    t3 = imp.head(3).feature.tolist()
    insight(f"The most influential inputs are <b>{t3[0]}</b>, <b>{t3[1]}</b> and <b>{t3[2]}</b>. "
            f"Best hyper-parameters: <code>{meta['best_params'] or 'defaults'}</code>.")
    st.markdown('<span class="note">Popularity is a time-dependent TMDB activity score; metadata alone cannot explain '
                'marketing, cast, reviews or word of mouth, which limits achievable accuracy.</span>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# App shell
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()
    if not (DATA_DIR / "movies_processed.parquet").exists():
        st.error("Processed data not found at data/processed/movies_processed.parquet. "
                 "Run notebooks/01_data_processing_and_cleaning.ipynb first (see README).")
        st.stop()

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Cine<span>Metrics</span></div>'
                    '<div class="note">Movie popularity analytics</div>', unsafe_allow_html=True)
    pages = [
        st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
        st.Page(page_explorer, title="Movie Explorer", icon=":material/search:", url_path="explorer"),
        st.Page(page_predict, title="Popularity Prediction", icon=":material/insights:", url_path="predict"),
        st.Page(page_insights, title="Data Insights", icon=":material/query_stats:", url_path="insights"),
        st.Page(page_performance, title="Model Performance", icon=":material/model_training:", url_path="model"),
    ]
    nav = st.navigation(pages)
    nav.run()
    with st.sidebar:
        st.markdown("---")
        st.markdown('<div class="note">Data: The Movies Dataset (TMDB / MovieLens), Kaggle — processed with '
                    'Apache Spark. Model: scikit-learn.</div>', unsafe_allow_html=True)


main()
