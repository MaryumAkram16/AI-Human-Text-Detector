import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Authenticity — AI vs Human Text Detector",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

STYLE_FEATURES = ["word_count", "sentence_len_std", "lexical_diversity", "punct_density"]


@st.cache_resource
def load_models():
    nb_model = joblib.load("nb_model.pkl")
    lr_model = joblib.load("lr_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    style_scaler = joblib.load("style_scaler.pkl")
    return nb_model, lr_model, rf_model, tfidf_vectorizer, style_scaler


nb_model, lr_model, rf_model, tfidf_vectorizer, style_scaler = load_models()


# ============ FEATURE ENGINEERING (matches training notebook exactly) ============
def word_count(text):
    return len(text.split())


def sentence_length_std(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return float(np.std(lengths))


def lexical_diversity(text):
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def punctuation_density(text):
    words = text.split()
    if not words:
        return 0.0
    punct_count = len(re.findall(r"[.,;:!?\-\"'()]", text))
    return punct_count / len(words)


def classify_text(text):
    style_vals = pd.DataFrame([{
        "word_count": word_count(text),
        "sentence_len_std": sentence_length_std(text),
        "lexical_diversity": lexical_diversity(text),
        "punct_density": punctuation_density(text),
    }])[STYLE_FEATURES]

    tfidf_vec = tfidf_vectorizer.transform([text])
    style_scaled = style_scaler.transform(style_vals)
    combined = hstack([tfidf_vec, csr_matrix(style_scaled)])

    nb_pred = nb_model.predict(tfidf_vec)[0]
    nb_proba = nb_model.predict_proba(tfidf_vec)[0]
    lr_pred = lr_model.predict(combined)[0]
    lr_proba = lr_model.predict_proba(combined)[0]
    rf_pred = rf_model.predict(combined)[0]
    rf_proba = rf_model.predict_proba(combined)[0]

    label_map = {0: "Human", 1: "AI"}
    classes = list(lr_model.classes_)
    ai_idx = classes.index(1)

    def fmt(pred, proba):
        return {"label": label_map[pred], "ai_confidence": float(proba[ai_idx])}

    return {
        "Naive Bayes": fmt(nb_pred, nb_proba),
        "Logistic Regression": fmt(lr_pred, lr_proba),
        "Random Forest": fmt(rf_pred, rf_proba),
        "style_features": style_vals.iloc[0].to_dict(),
    }


# ============ GLOBAL DARK THEME STYLING ============
st.markdown("""
<style>
:root {
    --bg-main: #0B0F1A;
    --bg-card: #12182B;
    --bg-card-alt: #171F35;
    --border: #262D45;
    --human: #E8B75F;
    --human-dark: #B8853A;
    --ai: #4FD1C5;
    --text-main: #EDEFF5;
    --text-muted: #8890A6;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

section[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4, h5, p, span, div, label {
    color: var(--text-main);
}

/* Eyebrow label */
.eyebrow {
    display: inline-block;
    color: var(--ai);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    font-weight: 600;
    text-transform: uppercase;
    background: rgba(79, 209, 197, 0.1);
    border: 1px solid rgba(79, 209, 197, 0.3);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* Hero section */
.hero { padding: 1rem 0 2rem 0; }
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 1rem;
    color: #FFFFFF;
}
.hero h1 .h { color: var(--human); }
.hero h1 .a { color: var(--ai); }
.hero p {
    color: var(--text-muted);
    font-size: 1.05rem;
    max-width: 640px;
    line-height: 1.6;
}

/* Card */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Badge pills */
.badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    margin-right: 0.4rem;
}
.badge-human { background: rgba(232, 183, 95, 0.15); color: #FCD34D; border: 1px solid rgba(232, 183, 95, 0.3); }
.badge-ai { background: rgba(79, 209, 197, 0.15); color: #5EEAD4; border: 1px solid rgba(79, 209, 197, 0.3); }
.badge-warn { background: rgba(245, 158, 11, 0.15); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.3); }

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(232, 183, 95, 0.12) 0%, rgba(79, 209, 197, 0.08) 100%);
    border: 1px solid rgba(79, 209, 197, 0.35);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
}
.result-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.result-value { font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0; }

/* Confidence bar */
.conf-row { margin-bottom: 0.9rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.3rem; }
.conf-track { background: #1E2438; border-radius: 8px; height: 10px; overflow: hidden; }
.conf-fill { background: linear-gradient(90deg, var(--human), var(--ai)); height: 100%; border-radius: 8px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--ai), #2FA79C);
    color: #04342C;
    border: none;
    font-weight: 700;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6FE3D8, var(--ai));
    color: #04342C;
}

/* Text areas / inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-card-alt) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Metric override */
[data-testid="stMetricValue"] { color: #FFFFFF; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ============ SIDEBAR NAVIGATION ============
with st.sidebar:
    st.markdown("## 🧾 Authenticity")
    st.caption("AI vs human text detection")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍  Try It", "📊  Model Performance", "🧠  Model & Method"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Student project · AI vs Human Text dataset\nNot a plagiarism or authorship ruling")

# ============ TRY IT ============
if page == "🔍  Try It":
    st.markdown('<div class="eyebrow">● TRAINED ON 487,235 ESSAYS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
        <h1>Tell <span class="h">human</span> writing from <span class="a">AI</span> output.</h1>
        <p>Paste an essay, article, or paragraph — this model scores it on writing style,
        not just word choice, and predicts whether it was written by a person or generated
        by AI, using a pipeline trained on 487,235 real essays.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        text_input = st.text_area(
            "Text to classify",
            height=180,
            placeholder="Paste an essay, article, or paragraph here...",
            label_visibility="collapsed"
        )
        predict_clicked = st.button("🔎  Classify This Text", type="primary")

    if predict_clicked:
        if not text_input or not text_input.strip():
            st.warning("Paste some text first — then hit classify.")
        else:
            result = classify_text(text_input)
            lr_result = result["Logistic Regression"]
            badge_class = "badge-ai" if lr_result["label"] == "AI" else "badge-human"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prediction</div>
                <div class="result-value">{lr_result['label']}</div>
                <span class="{badge_class} badge">AI confidence: {lr_result['ai_confidence']*100:.1f}%</span>
                <span class="badge badge-ai">Logistic Regression</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Model breakdown")
            bars_html = '<div class="card">'
            for name in ["Logistic Regression", "Random Forest", "Naive Bayes"]:
                r = result[name]
                pct = r["ai_confidence"] * 100
                bars_html += (
                    '<div class="conf-row">'
                    f'<div class="conf-label"><span>{name} → {r["label"]}</span><span>{pct:.1f}% AI</span></div>'
                    f'<div class="conf-track"><div class="conf-fill" style="width:{pct}%;"></div></div>'
                    '</div>'
                )
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown("#### Stylometric features (this text)")
            sf = result["style_features"]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Word count", f"{sf['word_count']:.0f}")
            with c2:
                st.metric("Sentence variance", f"{sf['sentence_len_std']:.2f}")
            with c3:
                st.metric("Lexical diversity", f"{sf['lexical_diversity']:.2f}")
            with c4:
                st.metric("Punctuation density", f"{sf['punct_density']:.2f}")

            st.caption(
                "Prediction uses the real trained pipeline (TF-IDF + 4 stylometric features → "
                "Logistic Regression / Random Forest; TF-IDF only → Naive Bayes). "
                "Best model accuracy is 99.36% on held-out test data — not a hiring or "
                "publishing decision, a model estimate."
            )

# ============ MODEL PERFORMANCE ============
elif page == "📊  Model Performance":
    st.markdown('<div class="eyebrow">● EVALUATED ON 97,447 HELD-OUT ESSAYS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Test set performance</h1></div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Best model accuracy", "99.36%")
    with m2:
        st.metric("Best model F1", "99.36%")
    with m3:
        st.metric("Test errors", "625 / 97,447")

    st.markdown("### Exploratory analysis")
    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            st.markdown("#### Class balance")
            st.image("chart_class_balance.png", width='stretch')
            st.caption("62.8% human, 37.2% AI.")
    with e2:
        with st.container(border=True):
            st.markdown("#### Word count by class")
            st.image("chart_length_distribution.png", width='stretch')
            st.caption("Human text has a longer tail.")

    e3, e4 = st.columns(2)
    with e3:
        with st.container(border=True):
            st.markdown("#### Sentence length variance")
            st.image("chart_sentence_stats.png", width='stretch')
            st.caption("AI text clusters at low variance — a tighter, more uniform rhythm.")
    with e4:
        with st.container(border=True):
            st.markdown("#### Lexical diversity")
            st.image("chart_lexical_diversity.png", width='stretch')
            st.caption("Weak but real difference between classes.")

    st.markdown("### Model comparison")
    p1, p2 = st.columns(2)
    with p1:
        with st.container(border=True):
            st.markdown("#### Accuracy / F1 by model")
            st.image("chart_model_comparison.png", width='stretch')
            st.caption("Logistic Regression leads on every metric.")
    with p2:
        with st.container(border=True):
            st.markdown("#### Confusion matrix")
            st.image("chart_confusion_matrix.png", width='stretch')
            st.caption("625 errors out of 97,447 test rows, roughly balanced in both directions.")

    with st.container(border=True):
        st.markdown("#### Stylometric feature importance")
        fi_col = st.columns([1, 2, 1])[1]
        with fi_col:
            st.image("chart_feature_importance.png", width='stretch')
        st.caption(
            "sentence_len_std ranks highest, confirming the EDA finding. word_count ranks "
            "second — the honest caveat is that part of this is a dataset-specific shortcut "
            "(this dataset's AI outputs run shorter), not pure style detection."
        )

# ============ MODEL & METHOD ============
else:
    st.markdown('<div class="eyebrow">● HOW IT WORKS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
        <h1>Project overview</h1>
        <p>A classic machine learning pipeline that scores text on writing style, not just
        word choice — trained on the AI vs Human Text dataset and compared across three
        genuinely different algorithm families, as required by the assignment.</p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Essays after cleaning", "487,235")
    with m2:
        st.metric("Algorithms compared", "3")
    with m3:
        st.metric("Best model F1", "99.36%")
    with m4:
        st.metric("Stylometric features", "4")

    st.markdown("### Three genuinely different algorithms")
    st.markdown("""
    <div class="card">
    <table style="width:100%; border-collapse: collapse;">
        <thead>
            <tr style="border-bottom: 1px solid var(--border); text-align:left; font-size:0.8rem; text-transform:uppercase; color:var(--text-muted);">
                <th style="padding:8px 0;">Model</th><th>Family</th><th>Features</th><th>Accuracy</th><th>F1</th>
            </tr>
        </thead>
        <tbody style="font-size:0.9rem;">
            <tr style="border-bottom: 1px solid var(--border); background: rgba(79,209,197,0.08);">
                <td style="padding:10px 0;"><b>Logistic Regression</b> <span class="badge badge-ai">winner</span></td>
                <td>Linear</td><td>TF-IDF + stylometric</td><td><b>99.36%</b></td><td><b>99.36%</b></td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding:10px 0;">Random Forest</td>
                <td>Tree ensemble</td><td>TF-IDF + stylometric</td><td>99.15%</td><td>99.15%</td>
            </tr>
            <tr>
                <td style="padding:10px 0;">Naive Bayes</td>
                <td>Probabilistic</td><td>TF-IDF only</td><td>95.47%</td><td>95.45%</td>
            </tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### The pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    steps = [
        ("1. Clean", "Dropped duplicate and null rows. 487,235 essays remain from the raw file."),
        ("2. EDA", "Checked class balance, length, and sentence structure before building anything."),
        ("3. Engineer", "4 stylometric features kept from EDA. TF-IDF (5,000 features) built alongside."),
        ("4. Combine", "Stylometric features scaled and stacked onto the TF-IDF matrix for 2 of the 3 models."),
        ("5. Compare", "Same train/test split across all three, scored on accuracy, precision, recall, F1."),
    ]
    for col, (title, desc) in zip([p1, p2, p3, p4, p5], steps):
        with col:
            st.markdown(f'<div class="card"><b>{title}</b><br><span style="color:#8890A6;font-size:0.85rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("### Key engineering decisions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="card">
        <b>Why these four stylometric features</b><br><br>
        EDA tested six candidates. Average sentence length barely differed between classes
        and was dropped. Sentence length variance, word count, lexical diversity, and
        punctuation density all showed real separation and made the final feature set.
        </div>
        <div class="card">
        <b>Why Naive Bayes only gets TF-IDF</b><br><br>
        Multinomial Naive Bayes assumes non-negative, count-like input. The scaled, centered
        stylometric features (some negative after standardization) don't fit that assumption,
        so Naive Bayes trains on word content alone.
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="card">
        <b>Why class_weight is balanced</b><br><br>
        The dataset is 62.8% human, 37.2% AI — not severe, but enough to bias a model toward
        the majority class without correction. All three models weight classes inversely to
        frequency during training.
        </div>
        <div class="card">
        <b>Why TF-IDF is capped at 5,000 features</b><br><br>
        A larger vocabulary would help Naive Bayes, but Random Forest gets slow and
        memory-heavy on very high-dimensional sparse input. 5,000 keeps every model
        trainable on the same feature set within a normal Colab session.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### What synthetic negative examples fixed")
    st.markdown("""
    <div class="card">
    The suitability of stylometric features alone wasn't the full story — Logistic Regression
    beat Random Forest and Naive Bayes on every metric (MAE-equivalent: 99.36% vs 99.15% vs
    95.47% accuracy), and sentence-length variance turned out to be a stronger, more genuine
    style signal than raw word count, which is more of a dataset-specific shortcut.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Honest limitations")
    l1, l2, l3 = st.columns(3)
    limitations = [
        ("Shortcut risk", "word_count is the second-strongest stylometric feature. Part of what the model learned is genuinely about writing style — part of it is this dataset's AI outputs tending to run shorter."),
        ("Single dataset", "Trained and tested on one AI vs Human Text dataset. A different AI model family, prompt style, or writing domain could shift these numbers in either direction."),
        ("Simple sentence splitting", "Sentence boundaries use a basic regex on . ! ? — not a proper NLP tokenizer — abbreviations and edge cases can slightly skew the sentence-length features."),
    ]
    for col, (title, desc) in zip([l1, l2, l3], limitations):
        with col:
            st.markdown(f'<div class="card"><span class="badge badge-warn">{title}</span><br><br><span style="color:#8890A6;font-size:0.9rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Models: MultinomialNB, LogisticRegression, RandomForestClassifier · random_state=42 · trained on Google Colab (free tier)")
