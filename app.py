"""
CineMatch - Movie Recommendation App

Uses the model saved by movie_recommendation.ipynb:
    model/recommender.pkl -> {"movie_df", "scaler", "kmeans", "feature_columns", "cosine_sim"}

Folder layout:
    app.py
    model/recommender.pkl

Run:
    pip install streamlit pandas numpy scikit-learn joblib
    streamlit run app.py
"""

from pathlib import Path

import base64
import mimetypes
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "recommender.pkl"
POSTER_DIR = Path(__file__).parent / "posters"   # put cover images here

# (gradient start, gradient end, emoji) per genre - used for the poster tiles
GENRE_STYLE = {
    "Action":    ("#FF6B35", "#7A1F0B", "💥"),
    "Adventure": ("#F7B32B", "#8A4B08", "🧭"),
    "Animation": ("#4CC9F0", "#3A0CA3", "🎨"),
    "Comedy":    ("#FFD166", "#C9184A", "😂"),
    "Crime":     ("#8D99AE", "#1B2432", "🕵️"),
    "Drama":     ("#C77DFF", "#3C096C", "🎭"),
    "Horror":    ("#9D0208", "#10002B", "👻"),
    "Romance":   ("#FF758F", "#590D22", "💕"),
    "Sci-Fi":    ("#2EC4B6", "#0B2545", "🚀"),
    "Thriller":  ("#5C7AEA", "#0D1B3E", "🔪"),
    "Unknown":   ("#6C757D", "#212529", "🎬"),
}

st.set_page_config(page_title="CineMatch - Find your next movie", page_icon="🎬",
                   layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap');
    :root {
        --bg:#0B1020; --panel:#141B30; --panel-2:#1D2640; --line:#263050;
        --text:#F2F4FA; --muted:#98A2C3; --gold:#FFB627; --teal:#2EC4B6;
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:radial-gradient(1200px 500px at 15% -10%, #1B2550 0%, var(--bg) 60%); color:var(--text); }
    header[data-testid="stHeader"] { background:transparent; }
    #MainMenu, footer { visibility:hidden; }
    .block-container { padding-top:2rem; max-width:1250px; }
    section[data-testid="stSidebar"] { background:var(--panel); }

    /* Hero */
    .hero { padding:1.2rem 0 1.4rem 0; }
    .brand { font-family:'Sora',sans-serif; font-weight:800; font-size:1.1rem; color:var(--gold); }
    .hero h1 { font-family:'Sora',sans-serif; font-weight:800; font-size:clamp(2rem,4.6vw,3.3rem);
               line-height:1.1; margin:.5rem 0 0 0; color:var(--text); }
    .hero p { color:var(--muted); font-size:1.08rem; margin-top:.7rem; max-width:50ch; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { font-weight:600; color:var(--muted); padding:.6rem 1rem; }
    .stTabs [aria-selected="true"] { color:var(--gold) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background:var(--gold); }

    /* Poster tile + card */
    .card { background:var(--panel); border:1px solid var(--line); border-radius:16px; overflow:hidden;
            transition:transform .18s ease, border-color .18s ease; }
    .card:hover { transform:translateY(-4px); border-color:var(--gold); }
    .poster { position:relative; aspect-ratio:4/5; display:flex; flex-direction:column;
              justify-content:flex-end; padding:1rem; color:#fff; overflow:hidden; }
    .poster .emoji { position:absolute; top:12%; left:50%; transform:translateX(-50%);
                     font-size:3.6rem; opacity:.95; filter:drop-shadow(0 6px 14px rgba(0,0,0,.35)); }
    .poster .title { font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem; line-height:1.2;
                     text-shadow:0 2px 10px rgba(0,0,0,.5); }
    .poster .genre { font-size:.82rem; opacity:.9; margin-top:.2rem; }
    .rating { position:absolute; top:.7rem; right:.7rem; background:rgba(11,16,32,.75); color:var(--gold);
              font-weight:700; font-size:.85rem; padding:.2rem .55rem; border-radius:99px; backdrop-filter:blur(4px); }
    .info { padding:.7rem .9rem .8rem .9rem; color:var(--muted); font-size:.86rem;
            display:flex; justify-content:space-between; gap:.5rem; }

    /* Featured pick */
    .feature { display:grid; grid-template-columns:minmax(200px,300px) 1fr; gap:1.6rem; align-items:center;
               background:linear-gradient(135deg,var(--panel-2),var(--panel)); border:1px solid var(--line);
               border-radius:20px; padding:1.2rem; margin-bottom:1.4rem; }
    .feature .poster { border-radius:14px; }
    .feature h2 { font-family:'Sora',sans-serif; font-size:clamp(1.5rem,3vw,2.2rem); margin:.2rem 0 .5rem 0; color:var(--text); }
    .badge { display:inline-block; background:var(--gold); color:#1A1200; font-weight:700; font-size:.82rem;
             padding:.25rem .7rem; border-radius:99px; }
    .chips { margin-top:.8rem; }
    .chip { display:inline-block; background:var(--panel); border:1px solid var(--line); color:var(--text);
            font-size:.85rem; padding:.3rem .75rem; border-radius:99px; margin:0 .4rem .4rem 0; }
    .lead { color:var(--muted); font-size:1rem; margin:.3rem 0 0 0; }
    @media (max-width:760px){ .feature { grid-template-columns:1fr; } }

    /* Inputs & buttons */
    .stButton > button { border-radius:12px; font-weight:600; border:1px solid var(--line);
                         background:var(--panel-2); color:var(--text); padding:.5rem 1rem; }
    .stButton > button:hover { border-color:var(--gold); color:var(--gold); }
    .stButton > button[kind="primary"] { background:var(--gold); border-color:var(--gold); color:#1A1200; }
    .stButton > button[kind="primary"]:hover { filter:brightness(1.08); color:#1A1200; }
    div[data-baseweb="select"] > div, .stTextInput input { background:var(--panel-2) !important; border-color:var(--line) !important; }
    .stSlider [data-baseweb="slider"] > div > div > div { background:var(--gold); }

    .empty { background:var(--panel); border:1px dashed var(--line); border-radius:16px; padding:2.2rem;
             text-align:center; color:var(--muted); }
    .section-title { font-family:'Sora',sans-serif; font-weight:700; font-size:1.3rem; margin:.4rem 0 .8rem 0; }
    @media (prefers-reduced-motion: reduce) { .card { transition:none; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Getting things ready...")
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


model = load_model()
if model is None:
    st.error(f"Could not find `{MODEL_PATH}`. Place `recommender.pkl` inside a `model/` folder next to app.py.")
    st.stop()

movie_df: pd.DataFrame = model["movie_df"].reset_index(drop=True)
cosine_sim = np.asarray(model["cosine_sim"])
titles = movie_df["Movie_Title"].tolist()
genres = sorted(movie_df["Genre"].unique())
languages = sorted(movie_df["Language"].unique())


def recommend(movie_title):
    """Same logic as recommend_movies() in the notebook:
    movies in the same cluster come first, then higher cosine similarity.
    Returns movie indices, best first (the caller takes the top N)."""
    movie_index = titles.index(movie_title)
    selected_cluster = movie_df.loc[movie_index, "Cluster"]
    scores = cosine_sim[movie_index]
    candidates = [
        (idx, movie_df.loc[idx, "Cluster"], scores[idx])
        for idx in movie_df.index if idx != movie_index
    ]
    candidates.sort(key=lambda c: (c[1] != selected_cluster, -c[2]))
    return [c[0] for c in candidates]


# ----------------------------------------------------------------------------
# STATE + HELPERS
# ----------------------------------------------------------------------------
for k, v in {"watchlist": [], "results": None, "label": ""}.items():
    st.session_state.setdefault(k, v)


def toggle_watchlist(title):
    wl = st.session_state.watchlist
    if title in wl:
        wl.remove(title)
        st.toast(f"Removed {title}")
    else:
        wl.append(title)
        st.toast(f"Saved {title} to your watchlist", icon="✅")


@st.cache_data(show_spinner=False)
def poster_uri(title):
    """Look for posters/<title>.(jpg|jpeg|png|webp). Accepts 'Future World.jpg' or 'future_world.jpg'."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    for name in (title, slug):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = POSTER_DIR / f"{name}{ext}"
            if p.exists():
                mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
                return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return None


def poster_html(m, extra=""):
    c1, c2, emoji = GENRE_STYLE.get(m["Genre"], GENRE_STYLE["Unknown"])
    uri = poster_uri(m["Movie_Title"])
    if uri:  # real cover image, with a dark fade at the bottom so the title stays readable
        bg = f"linear-gradient(to top,rgba(11,16,32,.92) 0%,rgba(11,16,32,0) 55%),url('{uri}') center/cover"
        emoji_html = ""
    else:    # fallback: genre-coloured tile
        bg = f"linear-gradient(160deg,{c1} 0%,{c2} 100%)"
        emoji_html = f'<div class="emoji">{emoji}</div>'
    return (
        f'<div class="poster" style="background:{bg}">'
        f'<div class="rating">★ {m["Rating"]:.1f}</div>{emoji_html}'
        f'<div class="title">{m["Movie_Title"]}</div><div class="genre">{m["Genre"]}</div></div>{extra}'
    )


def save_button(idx, key):
    title = movie_df.loc[idx, "Movie_Title"]
    saved = title in st.session_state.watchlist
    st.button("✓ Saved" if saved else "＋ Save for later", key=f"{key}_{idx}",
              on_click=toggle_watchlist, args=(title,), use_container_width=True)


def card(idx, key):
    m = movie_df.loc[idx]
    info = (f'<div class="info"><span>{int(round(m["Release_Year"]))} · {int(round(m["Runtime_Minutes"]))} min</span>'
            f'<span>{m["Language"]}</span></div>')
    st.markdown(f'<div class="card">{poster_html(m, info)}</div>', unsafe_allow_html=True)
    save_button(idx, key)


def grid(idxs, key, per_row):
    for s in range(0, len(idxs), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, i in zip(cols, idxs[s:s + per_row]):
            with col:
                card(i, key)
        st.write("")


def featured(idx, key, headline):
    m = movie_df.loc[idx]
    left, right = st.columns([1, 2], gap="large", vertical_alignment="center")
    with left:
        st.markdown(f'<div class="card">{poster_html(m)}</div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            f"""<span class="badge">{headline}</span>
            <h2>{m["Movie_Title"]}</h2>
            <p class="lead">A {m["Genre"].lower()} movie in {m["Language"]}, best suited for the {m["Age_Group"]} age group.</p>
            <div class="chips">
              <span class="chip">★ {m["Rating"]:.1f} rating</span>
              <span class="chip">{int(round(m["Release_Year"]))}</span>
              <span class="chip">{int(round(m["Runtime_Minutes"]))} minutes</span>
              <span class="chip">{m["Genre"]}</span>
            </div>""",
            unsafe_allow_html=True,
        )
        save_button(idx, key)
    st.write("")


def empty(msg):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)


def show_results(idxs, key, per_row):
    if not idxs:
        empty("No movies match your filters. Try removing one in <b>Filters</b> above.")
        return
    featured(idxs[0], f"{key}_top", "Top pick for you")
    if len(idxs) > 1:
        st.markdown('<div class="section-title">More you might enjoy</div>', unsafe_allow_html=True)
        grid(idxs[1:], key, per_row)


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """<div class="hero"><div class="brand">🎬 CineMatch</div>
    <h1>Find something great<br>to watch tonight.</h1>
    <p>Pick a movie you love and we'll suggest what to watch next.</p></div>""",
    unsafe_allow_html=True,
)

# Filters (kept in an expander so the page stays clean)
with st.expander("Filters"):
    f1, f2, f3, f4 = st.columns(4)
    n_recs = f1.slider("Number of suggestions", 3, min(12, len(movie_df) - 1), 5)
    genre_f = f2.multiselect("Genre", genres)
    lang_f = f3.multiselect("Language", languages)
    min_rating = f4.slider("Minimum rating", 0.0, 10.0, 0.0, 0.1)
    per_row = 3


def finalize(idxs):
    out = [i for i in idxs
           if (not genre_f or movie_df.loc[i, "Genre"] in genre_f)
           and (not lang_f or movie_df.loc[i, "Language"] in lang_f)
           and movie_df.loc[i, "Rating"] >= min_rating]
    return out[:n_recs]


tab_find, tab_browse, tab_watch = st.tabs(
    ["Find movies", "Browse all", f"My watchlist ({len(st.session_state.watchlist)})"]
)

# ---- Find movies ----------------------------------------------------------------
with tab_find:
    c1, c2, c3 = st.columns([6, 2, 2], vertical_alignment="bottom")
    liked = c1.selectbox("A movie you love", titles, index=None, placeholder="Search for a movie")
    go = c2.button("Show me movies", type="primary", use_container_width=True, disabled=liked is None)
    surprise = c3.button("Surprise me", use_container_width=True)

    if surprise:
        pick = titles[np.random.randint(len(titles))]
        st.session_state.results, st.session_state.label = recommend(pick), pick
    elif go and liked:
        st.session_state.results, st.session_state.label = recommend(liked), liked

    st.write("")
    if st.session_state.results is not None:
        st.markdown(f'<div class="section-title">Because you like {st.session_state.label}</div>', unsafe_allow_html=True)
        show_results(finalize(st.session_state.results), "find", per_row)
    else:
        empty("Choose a movie you enjoyed above, or press <b>Surprise me</b>.")

# ---- Browse all -------------------------------------------------------------------
with tab_browse:
    b1, b2 = st.columns([3, 2])
    q = b1.text_input("Search by title", placeholder="e.g. Future")
    sort = b2.selectbox("Sort by", ["Title (A-Z)", "Highest rated", "Newest"])
    pool = movie_df
    if q:
        pool = pool[pool["Movie_Title"].str.contains(q, case=False, na=False)]
    if genre_f:
        pool = pool[pool["Genre"].isin(genre_f)]
    if lang_f:
        pool = pool[pool["Language"].isin(lang_f)]
    pool = pool[pool["Rating"] >= min_rating]
    pool = {"Title (A-Z)": pool.sort_values("Movie_Title"),
            "Highest rated": pool.sort_values("Rating", ascending=False),
            "Newest": pool.sort_values("Release_Year", ascending=False)}[sort]
    st.caption(f"{len(pool)} movies")
    if pool.empty:
        empty("Nothing found. Try a different search or clear your filters.")
    else:
        grid(pool.index.tolist(), "browse", 4)

# ---- Watchlist --------------------------------------------------------------------
with tab_watch:
    wl = st.session_state.watchlist
    if not wl:
        empty("Nothing saved yet. Tap <b>Save for later</b> on any movie to keep it here.")
    else:
        grid([titles.index(t) for t in wl], "wl", 4)
        st.download_button("Download my list", pd.DataFrame({"Movie": wl}).to_csv(index=False),
                           file_name="my_watchlist.csv", mime="text/csv")
