"""
app.py — Streamlit UI for Dubizzle Smart Ad Classifier
"""

import streamlit as st
import joblib
import os
from utils import predict_with_confidence, clean_text

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dubizzle Smart Ad Classifier",
    page_icon="🏷️",
    layout="centered",
)

# ── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .stButton>button {
        background-color: #E8232A;
        color: white;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-size: 1rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #c01c22; color: white; }
    .category-badge {
        display: inline-block;
        padding: 0.4em 1.2em;
        border-radius: 20px;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 0.5em;
    }
    .confidence-bar { margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Category Config ──────────────────────────────────────────────────────────

CATEGORY_META = {
    "cars":        {"emoji": "🚗", "color": "#2196F3", "desc": "Vehicles & Motors"},
    "property":    {"emoji": "🏠", "color": "#4CAF50", "desc": "Real Estate & Rentals"},
    "electronics": {"emoji": "📱", "color": "#9C27B0", "desc": "Electronics & Gadgets"},
    "jobs":        {"emoji": "💼", "color": "#FF9800", "desc": "Jobs & Employment"},
}

# ── Load Model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists("model.joblib"):
        # Auto-train if model not found
        st.info("⚙️ Training model for the first time…")
        import subprocess
        subprocess.run(["python", "model.py"], check=True)
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer


# ── Header ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("## 🏷️")
with col2:
    st.markdown("## Dubizzle Smart Ad Classifier")

st.markdown(
    "Paste your classified ad below and instantly get the predicted category "
    "— powered by NLP + Logistic Regression."
)
st.divider()

# ── Input ────────────────────────────────────────────────────────────────────

st.markdown("#### ✍️ Enter your ad text")
ad_text = st.text_area(
    label="ad_input",
    label_visibility="collapsed",
    placeholder=(
        "e.g. Toyota Camry 2020, low mileage, GCC specs, single owner, "
        "accident free, price negotiable…"
    ),
    height=150,
)

# ── Example Buttons ──────────────────────────────────────────────────────────

st.markdown("**Try an example:**")
ex_cols = st.columns(4)
examples = {
    "🚗 Car":         "Toyota Camry 2020 low mileage GCC specs single owner accident free negotiable price",
    "🏠 Property":    "Spacious 2 bedroom apartment in Dubai Marina fully furnished sea view gym pool access",
    "📱 Electronics": "iPhone 15 Pro Max 256GB natural titanium 2 months old excellent condition full box accessories",
    "💼 Job":         "Hiring software engineer 3 years experience Python React Dubai startup competitive salary visa",
}
for col, (label, text) in zip(ex_cols, examples.items()):
    if col.button(label, use_container_width=True):
        st.session_state["example_text"] = text
        st.rerun()

if "example_text" in st.session_state:
    ad_text = st.session_state.pop("example_text")
    st.session_state["last_text"] = ad_text
    # re-populate the text area via query params workaround
    st.markdown(f"*Loaded: {ad_text[:60]}…*")

# ── Predict ──────────────────────────────────────────────────────────────────

predict_btn = st.button("🔍 Predict Category")

if predict_btn:
    if not ad_text.strip():
        st.warning("⚠️ Please enter some ad text first.")
    else:
        with st.spinner("Analysing ad…"):
            try:
                model, vectorizer = load_model()
                result = predict_with_confidence(ad_text, model, vectorizer)
                category = result["prediction"]
                scores = result["confidence"]
                meta = CATEGORY_META[category]

                st.divider()
                st.markdown("#### 🎯 Predicted Category")

                st.markdown(
                    f"<div class='category-badge' style='background:{meta['color']}22;"
                    f"color:{meta['color']};border:2px solid {meta['color']};'>"
                    f"{meta['emoji']} {category.upper()} &nbsp;·&nbsp; {meta['desc']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("")
                st.markdown("#### 📊 Confidence Scores")

                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for cat, pct in sorted_scores:
                    m = CATEGORY_META[cat]
                    label_html = f"{m['emoji']} **{cat}**"
                    st.markdown(f"{label_html} — {pct}%")
                    st.progress(int(pct))

                st.divider()
                with st.expander("🔬 Cleaned text sent to model"):
                    st.code(clean_text(ad_text), language=None)

            except FileNotFoundError as e:
                st.error(str(e))

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("")
st.markdown(
    "<small style='color:gray;'>Built with Python · scikit-learn · TF-IDF · "
    "Logistic Regression · Streamlit — portfolio project inspired by Dubizzle</small>",
    unsafe_allow_html=True,
)
