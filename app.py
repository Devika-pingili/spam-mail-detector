# -*- coding: utf-8 -*-
"""
Spam Mail Detector - Streamlit Web Application
Classifies emails, SMS, and text as Spam or Ham using ML + NLP.
"""

import io
import json
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess_text

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
STYLES_PATH = BASE_DIR / "assets" / "styles.css"
DEFAULT_MODEL_ACCURACY = 0.98
METRICS_PATH = BASE_DIR / "metrics.json"

# Known spam / phishing indicators
SUSPICIOUS_KEYWORDS = [
    "free", "win", "winner", "urgent", "prize", "click here", "click",
    "congratulations", "limited offer", "claim", "cash", "bonus",
    "lottery", "selected", "verify", "account", "password", "bank",
    "offer", "deal", "discount", "act now", "expires", "risk-free",
    "guaranteed", "million", "credit", "loan", "viagra", "weight loss",
]

URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|bit\.ly/[^\s]+|tinyurl\.[^\s]+|t\.co/[^\s]+",
    re.IGNORECASE,
)

SUSPICIOUS_DOMAIN_HINTS = [
    "bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd",
    "rb.gy", "cutt.ly", "short.link",
]

PHISHING_KEYWORDS = [
    "verify", "suspend", "locked", "unusual activity", "confirm identity",
    "update payment", "click below", "security alert", "unauthorized",
]


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load trained model and TF-IDF vectorizer from pickle files."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None, (
            "Model files not found. Run `python train_model.py` first to train and save the model."
        )
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, None
    except Exception as exc:
        return None, None, f"Failed to load model: {exc}"


def load_custom_css():
    """Inject custom stylesheet for modern dark UI."""
    if STYLES_PATH.exists():
        css = STYLES_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def find_suspicious_keywords(text: str) -> list[str]:
    """Return spam-indicator phrases found in raw text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for kw in sorted(SUSPICIOUS_KEYWORDS, key=len, reverse=True):
        if kw in text_lower and kw not in found:
            # Avoid marking substrings of longer matches twice
            if not any(kw in longer and kw != longer for longer in found):
                found.append(kw)
    return found


def extract_urls(text: str) -> list[str]:
    """Extract URLs from message text."""
    return URL_PATTERN.findall(text)


def detect_phishing(text: str) -> dict:
    """
    Detect phishing-like URLs and risky patterns.
    Returns dict with flags, urls, and human-readable warnings.
    """
    urls = extract_urls(text)
    warnings = []
    text_lower = text.lower()

    for url in urls:
        url_lower = url.lower()
        if any(hint in url_lower for hint in SUSPICIOUS_DOMAIN_HINTS):
            warnings.append(f"Shortened/suspicious link detected: {url[:80]}")
        if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
            warnings.append(f"IP-based URL (high risk): {url[:80]}")

    if len(urls) >= 3:
        warnings.append(f"Multiple URLs detected ({len(urls)}) — common in spam/phishing.")

    for pk in PHISHING_KEYWORDS:
        if pk in text_lower:
            warnings.append(f"Phishing-style phrase: \"{pk}\"")

    return {
        "is_phishing_risk": len(warnings) > 0,
        "urls": urls,
        "warnings": warnings,
    }


def highlight_suspicious_words_html(text: str, keywords: list[str]) -> str:
    """Wrap suspicious keywords in HTML spans for display."""
    if not keywords:
        return text.replace("\n", "<br>")

    result = text
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(re.escape(kw), re.IGNORECASE)

        def replacer(match):
            return (
                f'<span class="suspicious-word">{match.group(0)}</span>'
            )

        result = pattern.sub(replacer, result)
    return result.replace("\n", "<br>")


def get_word_importance(
    model: MultinomialNB,
    vectorizer: TfidfVectorizer,
    clean_text: str,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Top tokens contributing to spam prediction using NB log-probabilities
    weighted by TF-IDF values for this message.
    """
    if not clean_text.strip():
        return []

    vec = vectorizer.transform([clean_text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    row = vec.toarray().ravel()
    indices = row.nonzero()[0]
    if len(indices) == 0:
        return []

    scores = row[indices] * coef[indices]
    order = np.argsort(scores)[::-1][:top_n]
    top_indices = indices[order]
    return [(feature_names[i], float(scores[order[j]])) for j, i in enumerate(top_indices)]


def generate_explanation(
    is_spam: bool,
    confidence: float,
    suspicious_kw: list[str],
    phishing: dict,
    word_importance: list[tuple[str, float]],
) -> str:
    """Build a human-readable explanation for the prediction."""
    lines = []

    if is_spam:
        lines.append(
            f"The model classifies this message as **SPAM** with **{confidence:.1f}%** confidence."
        )
        lines.append(
            "Spam messages often use urgency, prizes, or calls-to-action to trick recipients."
        )
    else:
        lines.append(
            f"The model classifies this message as **NOT SPAM (Ham)** with **{confidence:.1f}%** confidence."
        )
        lines.append("The language patterns resemble legitimate personal or business communication.")

    if suspicious_kw:
        lines.append(
            f"**Suspicious keywords found:** {', '.join(suspicious_kw[:8])}"
            + ("..." if len(suspicious_kw) > 8 else "")
        )

    if phishing["is_phishing_risk"]:
        lines.append("**Phishing indicators:** " + "; ".join(phishing["warnings"][:3]))

    if word_importance and is_spam:
        top_words = ", ".join(w for w, _ in word_importance[:5])
        lines.append(f"**Strongest ML signals (stemmed tokens):** {top_words}")

    if not suspicious_kw and not phishing["is_phishing_risk"] and is_spam:
        lines.append(
            "No obvious keyword triggers — classification is driven by overall text patterns learned from training data."
        )

    return "\n\n".join(lines)


def predict_spam(
    model: MultinomialNB,
    vectorizer: TfidfVectorizer,
    raw_text: str,
) -> dict:
    """Run full prediction pipeline on raw user text."""
    clean = preprocess_text(raw_text)
    if not clean:
        return {
            "error": "Message is empty or could not be processed after cleaning.",
        }

    vec = vectorizer.transform([clean])
    proba = model.predict_proba(vec)[0]
    pred = int(model.predict(vec)[0])
    spam_prob = float(proba[1])
    ham_prob = float(proba[0])
    confidence = max(spam_prob, ham_prob) * 100
    is_spam = pred == 1

    suspicious = find_suspicious_keywords(raw_text)
    phishing = detect_phishing(raw_text)
    importance = get_word_importance(model, vectorizer, clean)

    return {
        "is_spam": is_spam,
        "label": "Spam" if is_spam else "Not Spam (Ham)",
        "spam_score": spam_prob * 100,
        "ham_score": ham_prob * 100,
        "confidence": confidence,
        "suspicious_keywords": suspicious,
        "phishing": phishing,
        "word_importance": importance,
        "clean_text": clean,
        "explanation": generate_explanation(
            is_spam, confidence, suspicious, phishing, importance
        ),
    }


def detect_language_hint(text: str) -> str:
    """Simple multilingual hint using character patterns (no extra dependency)."""
    if re.search(r"[\u0900-\u097F]", text):
        return "Detected script: Devanagari (Hindi/Marathi) — model trained on English SMS."
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Detected script: CJK — model trained on English SMS."
    if re.search(r"[\u0600-\u06FF]", text):
        return "Detected script: Arabic — model trained on English SMS."
    return "Language: English (primary training language)"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "history": [],
        "total_predictions": 0,
        "spam_count": 0,
        "ham_count": 0,
        "last_result": None,
        "input_text": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_to_history(text_preview: str, result: dict):
    """Append prediction to sidebar history (max 20 items)."""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "preview": text_preview[:60] + ("..." if len(text_preview) > 60 else ""),
        "label": result["label"],
        "confidence": result["confidence"],
        "is_spam": result["is_spam"],
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:20]
    st.session_state.total_predictions += 1
    if result["is_spam"]:
        st.session_state.spam_count += 1
    else:
        st.session_state.ham_count += 1
    st.session_state.last_result = result


def build_report_csv() -> bytes:
    """Export prediction history as CSV bytes."""
    if not st.session_state.history:
        return b""
    df = pd.DataFrame(st.session_state.history)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ AI Spam Mail Detector</h1>
            <p>Machine Learning · NLP · Phishing Awareness · Real-time Classification</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats_row():
    total = st.session_state.total_predictions
    spam = st.session_state.spam_count
    ham = st.session_state.ham_count
    spam_rate = (spam / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (str(total), "Total Scans"),
        (str(spam), "Spam Detected"),
        (str(ham), "Safe Messages"),
        (f"{spam_rate:.1f}%", "Spam Rate"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{val}</div>
                    <div class="stat-label">{lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sidebar(model_accuracy: float):
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["🔍 Detect Spam", "📊 Analytics Dashboard"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 📜 Prediction History")
        if st.session_state.history:
            for item in st.session_state.history[:10]:
                badge_class = "badge-spam" if item["is_spam"] else "badge-ham"
                st.markdown(
                    f'<div class="history-item">'
                    f'<span class="{badge_class}">{item["label"]}</span> '
                    f'({item["confidence"]:.0f}%)<br>'
                    f'<small>{item["time"]} — {item["preview"]}</small></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No predictions yet. Analyze a message to see history.")

        st.markdown("---")
        st.markdown("### ℹ️ App Info")
        st.info(
            f"**Model:** TF-IDF + Multinomial Naive Bayes  \n"
            f"**Accuracy:** ~{model_accuracy * 100:.1f}%  \n"
            f"**Dataset:** SMS Spam Collection (Kaggle/UCI)"
        )

        st.markdown("### 📖 About")
        st.markdown(
            "Classify **emails**, **SMS**, or any text as **Spam** or **Ham**. "
            "Uses NLP preprocessing and ML with phishing URL checks."
        )

        if st.session_state.history:
            st.download_button(
                label="⬇️ Download Report",
                data=build_report_csv(),
                file_name=f"spam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_predictions = 0
            st.session_state.spam_count = 0
            st.session_state.ham_count = 0
            st.session_state.last_result = None
            st.rerun()

    return page


def render_detect_page(model, vectorizer):
    st.markdown('<p class="section-title">Message Input</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a .txt file",
        type=["txt"],
        help="Paste email or SMS content from a text file",
    )

    file_text = ""
    if uploaded is not None:
        try:
            file_text = uploaded.read().decode("utf-8", errors="replace")
            st.success(f"Loaded file: **{uploaded.name}** ({len(file_text)} characters)")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    default = file_text if file_text else st.session_state.get("input_text", "")
    user_text = st.text_area(
        "Enter or paste email / SMS / message text",
        value=default,
        height=180,
        placeholder="Example: Congratulations! You won a $1000 prize. Click here to claim...",
    )

    lang_hint = detect_language_hint(user_text) if user_text.strip() else ""
    if lang_hint:
        st.caption(f"🌐 {lang_hint}")

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        detect_btn = st.button("🔎 Detect Spam", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🔄 Clear", use_container_width=True)

    if clear_btn:
        st.session_state.input_text = ""
        st.session_state.last_result = None
        st.rerun()

    if detect_btn:
        if not user_text.strip():
            st.warning("Please enter or upload some text to analyze.")
        else:
            st.session_state.input_text = user_text
            with st.spinner("Analyzing message with AI model..."):
                time.sleep(0.6)  # Brief animation for UX
                result = predict_spam(model, vectorizer, user_text)

            if "error" in result:
                st.error(result["error"])
            else:
                add_to_history(user_text, result)
                st.session_state.last_result = result
                st.rerun()

    if st.session_state.last_result:
        render_results(user_text or st.session_state.input_text, st.session_state.last_result)


def render_results(raw_text: str, result: dict):
    is_spam = result["is_spam"]
    card_class = "spam" if is_spam else "ham"
    label_class = "spam" if is_spam else "ham"
    icon = "🚨" if is_spam else "✅"

    st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-label {label_class}">{icon} {result['label']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Confidence", f"{result['confidence']:.1f}%")
    with c2:
        st.metric("Spam Score", f"{result['spam_score']:.1f}%")

    st.markdown("**Spam probability meter**")
    st.progress(min(result["spam_score"] / 100.0, 1.0))

    # Phishing warnings
    if result["phishing"]["is_phishing_risk"]:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Phishing / URL Risk Detected")
        for w in result["phishing"]["warnings"]:
            st.markdown(f"- {w}")
        if result["phishing"]["urls"]:
            st.markdown("**URLs found:** " + ", ".join(result["phishing"]["urls"][:5]))
        st.markdown("</div>", unsafe_allow_html=True)
    elif result["phishing"]["urls"]:
        st.markdown(
            f'<div class="safe-box">🔗 URLs found ({len(result["phishing"]["urls"])}): '
            f'Review links before clicking.</div>',
            unsafe_allow_html=True,
        )

    # Suspicious keywords + highlighted text
    if result["suspicious_keywords"]:
        st.markdown("### 🔶 Suspicious Keywords")
        st.markdown(
            " ".join(
                f'<span class="suspicious-word">{kw}</span>'
                for kw in result["suspicious_keywords"]
            ),
            unsafe_allow_html=True,
        )

    if raw_text.strip():
        st.markdown("### 📝 Highlighted Message")
        highlighted = highlight_suspicious_words_html(
            raw_text, result["suspicious_keywords"]
        )
        st.markdown(
            f'<div class="explanation-box">{highlighted}</div>',
            unsafe_allow_html=True,
        )

    # Word importance chart
    if result["word_importance"]:
        st.markdown("### 📈 Word Importance (ML Signals)")
        imp_df = pd.DataFrame(
            result["word_importance"], columns=["Token", "Importance"]
        )
        fig = px.bar(
            imp_df,
            x="Importance",
            y="Token",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#00d4ff", "#ef4444"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
            height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 AI Explanation")
    st.markdown(
        f'<div class="explanation-box">{result["explanation"].replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )


def render_analytics_page():
    st.markdown('<p class="section-title">Spam Analytics Dashboard</p>', unsafe_allow_html=True)

    total = st.session_state.total_predictions
    spam = st.session_state.spam_count
    ham = st.session_state.ham_count

    if total == 0:
        st.info("Run some predictions on the **Detect Spam** page to populate analytics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            names=["Spam", "Ham"],
            values=[spam, ham],
            title="Spam vs Ham Distribution",
            color_discrete_sequence=["#ef4444", "#10b981"],
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        hist_df = pd.DataFrame(st.session_state.history)
        if not hist_df.empty:
            hist_df["confidence"] = pd.to_numeric(hist_df["confidence"], errors="coerce")
            fig_bar = px.bar(
                hist_df.head(10),
                x="time",
                y="confidence",
                color="label",
                title="Recent Prediction Confidence",
                color_discrete_map={"Spam": "#ef4444", "Not Spam (Ham)": "#10b981"},
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f1f5f9",
                xaxis_title="Time",
                yaxis_title="Confidence %",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 📋 Session Summary")
    summary = pd.DataFrame(
        {
            "Metric": ["Total Predictions", "Spam Count", "Ham Count", "Spam Rate %"],
            "Value": [
                total,
                spam,
                ham,
                round(spam / total * 100, 2) if total else 0,
            ],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Spam Mail Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    load_custom_css()

    model, vectorizer, load_error = load_artifacts()
    model_accuracy = DEFAULT_MODEL_ACCURACY

    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            model_accuracy = float(saved.get("accuracy", model_accuracy))
        except Exception:
            pass

    render_header()

    if load_error:
        st.error(load_error)
        st.code("python train_model.py", language="bash")
        st.stop()

    page = render_sidebar(model_accuracy)
    render_stats_row()

    st.markdown("---")

    if page == "🔍 Detect Spam":
        render_detect_page(model, vectorizer)
    else:
        render_analytics_page()

    st.markdown("---")
    st.caption(
        "Built with Streamlit · scikit-learn · NLTK · For educational and production demo use."
    )


if __name__ == "__main__":
    main()
