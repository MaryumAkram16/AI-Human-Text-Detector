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
    initial_sidebar_state="collapsed"
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
        return {"label": label_map[pred], "ai_confidence": float(proba[ai_idx]) * 100}

    return {
        "Naive Bayes": fmt(nb_pred, nb_proba),
        "Logistic Regression": fmt(lr_pred, lr_proba),
        "Random Forest": fmt(rf_pred, rf_proba),
        "style_features": style_vals.iloc[0].to_dict(),
    }


st.markdown("""
<style>
:root {
    --ink: #0B0F1A;
    --surface: #12182B;
    --surface-alt: #171F35;
    --border: rgba(237, 239, 245, 0.09);
    --border-strong: rgba(237, 239, 245, 0.16);
    --text-primary: #EDEFF5;
    --text-muted: #8890A6;
    --human: #E8B75F;
    --human-dim: rgba(232, 183, 95, 0.12);
    --ai: #4FD1C5;
    --ai-dim: rgba(79, 209, 197, 0.12);
}

.stApp { background-color: var(--ink); color: var(--text-primary); }
h1, h2, h3, h4 { color: var(--text-primary) !important; }

.brand-row { display: flex; align-items: center; gap: 10px; padding-top: 6px; }
.brand-icon {
    width: 26px; height: 26px; border-radius: 7px;
    background: linear-gradient(135deg, var(--human) 0%, var(--human) 49%, var(--ai) 51%, var(--ai) 100%);
}
.brand-name { font-size: 17px; font-weight: 700; color: #FFFFFF; }

.eyebrow {
    display: inline-block; font-size: 0.75rem; letter-spacing: 0.12em;
    color: var(--ai); text-transform: uppercase; font-weight: 600;
    background: var(--ai-dim); border: 1px solid rgba(79,209,197,0.3);
    padding: 0.3rem 0.8rem; border-radius: 20px; margin-bottom: 1rem;
}

.hero h1 {
    font-size: 2.4rem; font-weight: 700; line-height: 1.15; margin-bottom: 0.8rem; color: #FFFFFF;
}
.hero h1 .h { color: var(--human); }
.hero h1 .a { color: var(--ai); }
.hero p { color: var(--text-muted); font-size: 1.02rem; max-width: 640px; line-height: 1.6; }

.demo {
    border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
    display: grid; grid-template-columns: 1fr 1fr; background: var(--surface); margin: 1.2rem 0;
}
.demo-pane { padding: 22px 26px; }
.demo-pane.human { background: linear-gradient(120deg, var(--human-dim), transparent 60%); border-right: 1px solid var(--ai); }
.demo-pane.ai { background: linear-gradient(240deg, var(--ai-dim), transparent 60%); }
.demo-label {
    font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}
.demo-label.human { color: var(--human); }
.demo-label.ai { color: var(--ai); }
.demo-label .pct { margin-left: auto; font-weight: 700; }
.demo-text { font-size: 13.5px; line-height: 1.75; }
.demo-text.human { color: #E8E2D0; }
.demo-text.ai { color: #CFEFEB; }
.demo-text mark { padding: 1px 3px; border-radius: 3px; }
.demo-text.human mark { background: rgba(232,183,95,0.22); color: var(--human); }
.demo-text.ai mark { background: rgba(79,209,197,0.18); color: var(--ai); }
.demo-foot {
    grid-column: 1/-1; border-top: 1px solid var(--border); padding: 10px 26px;
    display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted);
}

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.result-card {
    background: linear-gradient(135deg, var(--human-dim) 0%, var(--ai-dim) 100%);
    border: 1px solid rgba(79,209,197,0.35); border-radius: 14px; padding: 1.4rem 1.6rem; margin-top: 1rem;
}
.result-label { font-size: 0.82rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.result-value { font-size: 1.7rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0; }

.badge { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.22rem 0.65rem; border-radius: 20px; margin-right: 0.35rem; }
.badge-human { background: rgba(232,183,95,0.15); color: #FCD34D; border: 1px solid rgba(232,183,95,0.3); }
.badge-ai { background: rgba(79,209,197,0.15); color: #5EEAD4; border: 1px solid rgba(79,209,197,0.3); }

.conf-row { margin-bottom: 0.8rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.3rem; }
.conf-track { background: #1E2438; border-radius: 8px; height: 9px; overflow: hidden; }
.conf-fill { background: linear-gradient(90deg, var(--human), var(--ai)); height: 100%; border-radius: 8px; }

.try-label { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; }
.try-foot-label { font-size: 12px; color: var(--text-muted); padding-top: 8px; }

.stButton > button {
    background: #FFFFFF; color: var(--ink); font-weight: 700; border: none; border-radius: 8px; padding: 0.55rem 1.4rem;
}
.stButton > button:hover { background: var(--ai); color: var(--ink); }

.stTextArea textarea {
    background: var(--surface-alt) !important; color: #FFFFFF !important;
    border: 1px solid var(--border-strong) !important; caret-color: var(--ai) !important;
}
.stTextArea textarea::placeholder { color: var(--text-muted) !important; opacity: 1 !important; }

.stTabs [role="tablist"] { gap: 4px; background: var(--surface); border-radius: 10px; padding: 4px; border: 1px solid var(--border); width: fit-content; }
.stTabs [role="tab"], .stTabs [data-testid="stTab"] { color: var(--text-muted) !important; border-radius: 7px !important; }
.stTabs [role="tab"] *, .stTabs [data-testid="stTab"] * { color: inherit !important; }
.stTabs [aria-selected="true"] { background: var(--ai) !important; color: #04342C !important; }
.stTabs [aria-selected="true"] * { color: #04342C !important; }

[data-testid="stMetricValue"] { color: #FFFFFF !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="brand-row"><div class="brand-icon"></div><div class="brand-name">Authenticity</div></div>', unsafe_allow_html=True)
tab_try, tab_perf, tab_method = st.tabs(["Try it", "Model performance", "Model & method"])

with tab_try:
    st.markdown('<div class="eyebrow">● 487,235 essays classified</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
        <h1>Tell <span class="h">human</span> writing from <span class="a">AI</span> output.</h1>
        <p>A classic machine learning pipeline that scores text on writing style, not just
        word choice. Paste something below and run it through all three trained models.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="demo">
        <div class="demo-pane human">
            <div class="demo-label human">Human-written <span class="pct">51.8%</span></div>
            <div class="demo-text human">i really love going to the park on weekends!! its so much fun, <mark>esp when the weather is nice</mark>. my friends and i usually just hang out for hours, we dont really plan anything</div>
        </div>
        <div class="demo-pane ai">
            <div class="demo-label ai">AI-generated <span class="pct">97.3%</span></div>
            <div class="demo-text ai">Recreational activities provide numerous benefits for social and physical well-being. <mark>Engaging in outdoor pursuits fosters community connections</mark> and contributes to overall health outcomes.</div>
        </div>
        <div class="demo-foot">
            <span>sentence_len_std · word_count · punct_density · lexical_diversity</span>
            <span>logistic regression · 99.36% F1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="try-label">Paste text to classify</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            "Text to classify", height=140,
            placeholder="Paste an essay, article, or paragraph here...",
            label_visibility="collapsed"
        )
        foot_col1, foot_col2 = st.columns([3, 1])
        with foot_col1:
            st.markdown('<div class="try-foot-label">Naive Bayes · Logistic Regression · Random Forest</div>', unsafe_allow_html=True)
        with foot_col2:
            predict_clicked = st.button("Classify →", type="primary")

    if predict_clicked:
        if not text_input or not text_input.strip():
            st.warning("Paste some text first.")
        else:
            result = classify_text(text_input)
            lr_result = result["Logistic Regression"]
            badge_class = "badge-ai" if lr_result["label"] == "AI" else "badge-human"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prediction (Logistic Regression)</div>
                <div class="result-value">{lr_result['label']}</div>
                <span class="badge {badge_class}">AI confidence: {lr_result['ai_confidence']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Model breakdown")
            bars_html = '<div class="card">'
            for name in ["Logistic Regression", "Random Forest", "Naive Bayes"]:
                r = result[name]
                pct = r["ai_confidence"]
                bars_html += (
                    '<div class="conf-row">'
                    f'<div class="conf-label"><span>{name} → {r["label"]}</span><span>{pct:.1f}% AI</span></div>'
                    f'<div class="conf-track"><div class="conf-fill" style="width:{pct}%;"></div></div>'
                    '</div>'
                )
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)

            sf = result["style_features"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Word count", f"{sf['word_count']:.0f}")
            c2.metric("Sentence variance", f"{sf['sentence_len_std']:.2f}")
            c3.metric("Lexical diversity", f"{sf['lexical_diversity']:.2f}")
            c4.metric("Punctuation density", f"{sf['punct_density']:.2f}")

with tab_perf:
    m1, m2, m3 = st.columns(3)
    m1.metric("Best model accuracy", "99.36%")
    m2.metric("Best model F1", "99.36%")
    m3.metric("Test errors", "625 / 97,447")

    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            st.markdown("#### Class balance")
            st.image("chart_class_balance.png", width='stretch')
    with e2:
        with st.container(border=True):
            st.markdown("#### Word count by class")
            st.image("chart_length_distribution.png", width='stretch')

    e3, e4 = st.columns(2)
    with e3:
        with st.container(border=True):
            st.markdown("#### Sentence variance")
            st.image("chart_sentence_stats.png", width='stretch')
    with e4:
        with st.container(border=True):
            st.markdown("#### Lexical diversity")
            st.image("chart_lexical_diversity.png", width='stretch')

    p1, p2 = st.columns(2)
    with p1:
        with st.container(border=True):
            st.markdown("#### Model comparison")
            st.image("chart_model_comparison.png", width='stretch')
    with p2:
        with st.container(border=True):
            st.markdown("#### Confusion matrix")
            st.image("chart_confusion_matrix.png", width='stretch')

    with st.container(border=True):
        st.markdown("#### Feature importance")
        fi_col = st.columns([1, 2, 1])[1]
        with fi_col:
            st.image("chart_feature_importance.png", width='stretch')

with tab_method:
    st.markdown('<div class="eyebrow">THREE ALGORITHMS</div>', unsafe_allow_html=True)
    st.markdown("### Genuinely different model families")
    st.markdown('<p style="color:var(--text-muted);">Not three variations on the same idea — a probabilistic model, a linear model, and a tree ensemble, each given the inputs suited to how it learns.</p>', unsafe_allow_html=True)

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

    st.markdown('<div class="eyebrow" style="margin-top:2rem;">PIPELINE</div>', unsafe_allow_html=True)
    st.markdown("### How the pipeline works")
    st.markdown('<p style="color:var(--text-muted);">From raw CSV to three compared models.</p>', unsafe_allow_html=True)

    steps = [
        ("1", "Clean", "Drop duplicate and null rows. 487,235 essays remain from the raw file."),
        ("2", "EDA", "Check class balance, length, and sentence structure before building anything."),
        ("3", "Engineer", "Four stylometric features kept from EDA. TF-IDF (5,000 features) built alongside."),
        ("4", "Combine", "Stylometric features scaled and stacked onto the TF-IDF matrix for two of the three models."),
        ("5", "Compare", "Same train/test split across all three, scored on accuracy, precision, recall, F1."),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="card"><span class="badge badge-ai">{num}</span><br><br><b>{title}</b><br><span style="color:#8890A6;font-size:0.85rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:2rem;">REASONING</div>', unsafe_allow_html=True)
    st.markdown("### Key engineering decisions")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="card">
        <b>● Why these four stylometric features</b><br><br>
        EDA tested six candidates. <b>Average sentence length</b> barely differed between classes and was dropped. <b>Sentence length variance</b>, <b>word count</b>, <b>lexical diversity</b>, and <b>punctuation density</b> all showed real separation and made the final feature set.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <b>● Why class_weight is balanced</b><br><br>
        The dataset is 62.8% human, 37.2% AI — not severe, but enough to bias a model toward the majority class without correction. All three models weight classes inversely to frequency during training.
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="card">
        <b>● Why Naive Bayes only gets TF-IDF</b><br><br>
        Multinomial Naive Bayes assumes non-negative, count-like input. The scaled, centered stylometric features (some negative after standardization) don't fit that assumption, so Naive Bayes trains on word content alone while the other two get the combined set.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <b>● Why TF-IDF is capped at 5,000 features</b><br><br>
        A larger vocabulary would help Naive Bayes, but Random Forest gets slow and memory-heavy on very high-dimensional sparse input. 5,000 keeps every model trainable on the same feature set within a normal Colab session.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:2rem;">HONEST LIMITATIONS</div>', unsafe_allow_html=True)
    st.markdown("### Where this can be wrong")

    l1, l2, l3 = st.columns(3)
    limitations = [
        ("Shortcut risk", "word_count is the second-strongest stylometric feature. Part of what the model learned is genuinely about writing style — part of it is this dataset's AI outputs tending to run shorter, which may not hold for other AI writing tools."),
        ("Single dataset", "Trained and tested on one AI vs Human Text dataset. A different AI model family, prompt style, or writing domain could shift these numbers in either direction."),
        ("Simple sentence splitting", "Sentence boundaries are detected with a basic regex on . ! ? — not a proper NLP sentence tokenizer."),
    ]
    for col, (title, desc) in zip([l1, l2, l3], limitations):
        with col:
            st.markdown(f'<div class="card"><span class="badge badge-human">{title}</span><br><br><span style="color:#8890A6;font-size:0.9rem;">{desc}</span></div>', unsafe_allow_html=True)

st.caption("Trained on Google Colab · AI vs Human Text dataset")
