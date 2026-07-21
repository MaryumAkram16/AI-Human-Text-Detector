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


st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Special+Elite&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

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

.stApp { background-color: var(--ink); color: var(--text-primary); font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; }
code { font-family: 'JetBrains Mono', monospace; color: var(--ai); }

.eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 0.12em;
    color: var(--ai); text-transform: uppercase; display: inline-flex; align-items: center; gap: 8px;
    border: 1px solid rgba(79,209,197,0.3); background: var(--ai-dim); padding: 6px 14px; border-radius: 20px;
    margin-bottom: 20px;
}
.eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ai); }

h1.title {
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600; letter-spacing: -0.02em;
    font-size: 46px; line-height: 1.1; max-width: 780px; color: #FFFFFF !important;
    position: static !important; display: block !important; margin: 0 0 18px 0 !important;
}
h1.title .h { color: var(--human) !important; }
h1.title .a { color: var(--ai) !important; }
p.subtitle {
    font-size: 16.5px; color: var(--text-muted); max-width: 620px;
    position: static !important; display: block !important; margin: 0 0 28px 0 !important;
}

.demo {
    border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
    display: grid; grid-template-columns: 1fr 1fr; background: var(--surface);
    position: static !important; margin: 20px 0 !important;
}
.demo-pane { padding: 26px 30px; }
.demo-pane.human { background: linear-gradient(120deg, var(--human-dim), transparent 60%); border-right: 1px solid var(--ai); }
.demo-pane.ai { background: linear-gradient(240deg, var(--ai-dim), transparent 60%); }
.demo-label {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.demo-label.human { color: var(--human); }
.demo-label.ai { color: var(--ai); }
.demo-label .pct { margin-left: auto; font-weight: 600; }
.demo-text.human { font-family: 'Special Elite', cursive; font-size: 14.5px; line-height: 1.85; color: #E8E2D0; }
.demo-text.ai { font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.85; color: #CFEFEB; }
.demo-text mark { padding: 1px 3px; border-radius: 3px; }
.demo-text.human mark { background: rgba(232,183,95,0.22); color: var(--human); }
.demo-text.ai mark { background: rgba(79,209,197,0.18); color: var(--ai); }
.demo-foot {
    grid-column: 1 / -1; border-top: 1px solid var(--border); padding: 12px 30px;
    display: flex; justify-content: space-between; align-items: center;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted);
}

[data-testid="stMetricValue"] { color: #FFFFFF !important; font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

.sec-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;
}
.sec-title { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 600; color: #FFFFFF; margin-bottom: 8px; }
.sec-desc { color: var(--text-muted); font-size: 14.5px; margin-bottom: 20px; }

.chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; padding-bottom: 4px; }
.chart-card-label { padding: 10px 16px; font-size: 12.5px; color: var(--text-muted); }
.chart-card-label b { color: var(--text-primary); font-weight: 500; }

/* Streamlit's native bordered container (st.container(border=True)) - used
   instead of raw <div> wrapping since Streamlit widgets can't actually be
   nested inside HTML tags split across separate st.markdown() calls. */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

.callout {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px 24px; display: flex; gap: 14px; align-items: flex-start; margin-top: 14px;
}
.callout .mark { font-family: 'JetBrains Mono', monospace; color: var(--human); font-size: 13px; flex-shrink: 0; margin-top: 2px; }
.callout p { font-size: 13.5px; color: var(--text-muted); margin: 0; }
.callout p b { color: var(--text-primary); font-weight: 500; }

.model-table { width: 100%; border-collapse: collapse; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 16px; }
.model-table thead th {
    background: var(--surface-alt); text-align: left; padding: 12px 18px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border);
}
.model-table tbody td { padding: 14px 18px; border-bottom: 1px solid var(--border); font-size: 13.5px; }
.model-table tbody tr:last-child td { border-bottom: none; }
.model-table tbody tr.winner { background: var(--ai-dim); }
.model-table .name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: var(--text-primary); }
.model-table .tag {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-muted);
    border: 1px solid var(--border-strong); padding: 2px 8px; border-radius: 12px; display: inline-block; margin-top: 4px;
}
.model-table .score { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14px; color: var(--text-primary); }
.model-table tr.winner .score { color: var(--ai); }

.pipeline-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px; height: 100%; }
.step-num {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ai);
    border: 1px solid rgba(79,209,197,0.3); background: var(--ai-dim);
    width: 22px; height: 22px; border-radius: 6px; display: flex; align-items: center; justify-content: center;
    margin-bottom: 12px;
}
.pipeline-card .title { font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 600; margin-bottom: 6px; color: var(--text-primary); }
.pipeline-card .desc { font-size: 12px; color: var(--text-muted); line-height: 1.6; }

.decision-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; height: 100%; }
.decision-card .title {
    font-family: 'Space Grotesk', sans-serif; font-size: 14.5px; font-weight: 600;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px; color: var(--text-primary);
}
.decision-card .title::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--human); flex-shrink: 0; }
.decision-card .desc { font-size: 13px; color: var(--text-muted); line-height: 1.7; }
.decision-card .desc b { color: var(--text-primary); font-weight: 500; }

.limit-card { background: var(--surface); border: 1px solid rgba(232,183,95,0.25); border-radius: 12px; padding: 18px; height: 100%; }
.limit-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--human); background: var(--human-dim);
    border: 1px solid rgba(232,183,95,0.3); padding: 3px 10px; border-radius: 20px; display: inline-block; margin-bottom: 12px;
}
.limit-card .desc { font-size: 12.5px; color: var(--text-muted); line-height: 1.7; }


.stButton > button {
    background: var(--text-primary) !important; color: var(--ink) !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important; padding: 0.6rem 1.6rem !important;
}
.stButton > button:hover { background: var(--ai) !important; color: var(--ink) !important; }

.stTextArea textarea {
    background: var(--surface-alt) !important; color: #FFFFFF !important;
    border: 1px solid var(--border-strong) !important; font-family: 'JetBrains Mono', monospace !important;
    caret-color: var(--ai) !important;
}
.stTextArea textarea::placeholder { color: var(--text-muted) !important; opacity: 1 !important; }

.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {
    gap: 4px; background: var(--surface); border-radius: 10px; padding: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"], .stTabs button[role="tab"] {
    color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;
    border-radius: 7px !important;
}
.stTabs [aria-selected="true"] { background: var(--ai) !important; color: #04342C !important; }
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

.badge {
    display: inline-block; font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; margin-right: 6px; font-family: 'JetBrains Mono', monospace;
}
.badge-ai { background: rgba(79,209,197,0.15); color: #5EEAD4; border: 1px solid rgba(79,209,197,0.3); }
.badge-human { background: rgba(232,183,95,0.15); color: #FCD34D; border: 1px solid rgba(232,183,95,0.3); }

.footer-stack { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
.stack-pill {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted);
    border: 1px solid var(--border); padding: 5px 12px; border-radius: 20px;
}
hr { border-color: var(--border) !important; }
</style>
""")

st.markdown(
    '<div class="eyebrow"><span class="dot"></span>487,235 essays classified</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<h1 class="title">Tell <span class="h">human</span> writing from <span class="a">AI</span> output.</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">A classic machine learning pipeline that scores text on writing style, '
    'not just word choice — trained on the AI vs Human Text dataset, compared across three '
    'algorithm families.</p>',
    unsafe_allow_html=True
)
st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

tab_try, tab_perf, tab_method = st.tabs(["🔍  Try it", "📊  Model performance", "🧠  Model & method"])

with tab_try:
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

    st.write("")
    with st.container(border=True):
        text_input = st.text_area(
            "Paste text to classify",
            height=180,
            placeholder="Paste an essay, article, or paragraph here...",
            label_visibility="visible"
        )
        classify_clicked = st.button("🔎  Classify", type="primary")

    if classify_clicked:
        if not text_input or not text_input.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner("Running all three models..."):
                result = classify_text(text_input)

            lr_result = result["Logistic Regression"]
            badge_class = "badge-ai" if lr_result["label"] == "AI" else "badge-human"

            st.markdown(f"""
            <div class="demo" style="grid-template-columns: 1fr; margin-top: 20px;">
                <div class="demo-pane" style="grid-column: 1/-1;">
                    <div class="demo-label" style="color:var(--text-muted);">Prediction (Logistic Regression)</div>
                    <h2 style="color:#FFFFFF; margin: 4px 0 10px 0;">{lr_result['label']}</h2>
                    <span class="badge {badge_class}">AI confidence: {lr_result['ai_confidence']:.1f}%</span>
                </div>
                <div class="demo-foot" style="grid-column:1/-1;">
                    <span>All 3 models run live on your text</span>
                    <span>logistic regression · 99.36% F1</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            rows = ""
            for name in ["Logistic Regression", "Random Forest", "Naive Bayes"]:
                r = result[name]
                winner_class = "winner" if name == "Logistic Regression" else ""
                badge = "badge-ai" if r["label"] == "AI" else "badge-human"
                rows += f"""
                <tr class="{winner_class}">
                    <td class="name">{name}</td>
                    <td><span class="badge {badge}">{r['label']}</span></td>
                    <td class="score">{r['ai_confidence']:.1f}%</td>
                </tr>
                """
            st.markdown(f"""
            <table class="model-table">
                <thead><tr><th>Model</th><th>Prediction</th><th>AI confidence</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

            sf = result["style_features"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Word count", f"{sf['word_count']:.0f}")
            c2.metric("Sentence variance", f"{sf['sentence_len_std']:.2f}")
            c3.metric("Lexical diversity", f"{sf['lexical_diversity']:.2f}")
            c4.metric("Punctuation density", f"{sf['punct_density']:.2f}")

            st.caption(
                "Prediction uses the real trained pipeline (TF-IDF + 4 stylometric features → "
                "Logistic Regression / Random Forest; TF-IDF only → Naive Bayes). Not a hiring or "
                "publishing decision — a model estimate."
            )

with tab_perf:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best model F1", "99.36%")
    m2.metric("Essays after cleaning", "487,235")
    m3.metric("Errors on test rows", "625")
    m4.metric("Stylometric features", "4")

    st.markdown('<div class="sec-eyebrow">Exploratory analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">What the data showed</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Sentence-length variance turned out to be the strongest stylometric signal — AI text keeps a tighter, more uniform rhythm than human writing.</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            st.image("chart_class_balance.png", use_container_width=True)
            st.markdown('<div class="chart-card-label"><b>Class balance</b> — 62.8% human, 37.2% AI</div>', unsafe_allow_html=True)
    with e2:
        with st.container(border=True):
            st.image("chart_length_distribution.png", use_container_width=True)
            st.markdown('<div class="chart-card-label"><b>Word count by class</b> — human text has a longer tail</div>', unsafe_allow_html=True)

    e3, e4 = st.columns(2)
    with e3:
        with st.container(border=True):
            st.image("chart_sentence_stats.png", use_container_width=True)
            st.markdown('<div class="chart-card-label"><b>Sentence variance</b> — AI clusters at low variance</div>', unsafe_allow_html=True)
    with e4:
        with st.container(border=True):
            st.image("chart_lexical_diversity.png", use_container_width=True)
            st.markdown('<div class="chart-card-label"><b>Lexical diversity</b> — weak but real difference</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-eyebrow">Model evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Held-out test set performance</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        with st.container(border=True):
            st.image("chart_model_comparison.png", use_container_width=True)
            st.markdown('<div class="chart-card-label"><b>Model comparison</b> — Logistic Regression leads on every metric</div>', unsafe_allow_html=True)
    with p2:
        with st.container(border=True):
            st.image("chart_confusion_matrix.png", use_container_width=True)
            st.markdown('<div class="chart-card-label"><b>Confusion matrix</b> — 625 errors out of 97,447 test rows</div>', unsafe_allow_html=True)

    with st.container(border=True):
        fi_col = st.columns([1, 2, 1])[1]
        with fi_col:
            st.image("chart_feature_importance.png", use_container_width=True)
        st.markdown('<div class="chart-card-label"><b>Stylometric feature importance</b> — sentence_len_std ranks highest, confirming the EDA</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <span class="mark">note</span>
        <p><b>Word count is the honest caveat.</b> It's the second-strongest stylometric feature, which means part of what the model learned is genuinely about sentence rhythm — and part of it is about output length, a dataset-specific shortcut rather than pure style detection.</p>
    </div>
    """, unsafe_allow_html=True)

with tab_method:
    st.markdown('<div class="sec-eyebrow">Three algorithms</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Genuinely different model families</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">Not three variations on the same idea — a probabilistic model, a linear model, and a tree ensemble, each given the inputs suited to how it learns.</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="model-table">
        <thead><tr><th>Model</th><th>Family</th><th>Features</th><th>Accuracy</th><th>F1</th></tr></thead>
        <tbody>
            <tr class="winner">
                <td><div class="name">Logistic Regression</div><span class="tag">winner</span></td>
                <td>Linear</td><td>TF-IDF + stylometric</td>
                <td class="score">99.36%</td><td class="score">99.36%</td>
            </tr>
            <tr>
                <td><div class="name">Random Forest</div></td>
                <td>Tree ensemble</td><td>TF-IDF + stylometric</td>
                <td class="score">99.15%</td><td class="score">99.15%</td>
            </tr>
            <tr>
                <td><div class="name">Naive Bayes</div></td>
                <td>Probabilistic</td><td>TF-IDF only</td>
                <td class="score">95.47%</td><td class="score">95.45%</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-eyebrow">Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">How the pipeline works</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-desc">From raw CSV to three compared models.</div>', unsafe_allow_html=True)

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
            st.markdown(f"""
            <div class="pipeline-card">
                <div class="step-num">{num}</div>
                <div class="title">{title}</div>
                <div class="desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-eyebrow">Reasoning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Key engineering decisions</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="decision-card">
        <div class="title">Why these four stylometric features</div>
        <div class="desc">EDA tested six candidates. <b>Average sentence length</b> barely differed between classes and was dropped. <b>Sentence length variance</b>, <b>word count</b>, <b>lexical diversity</b>, and <b>punctuation density</b> all showed real separation and made the final feature set.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="decision-card">
        <div class="title">Why Naive Bayes only gets TF-IDF</div>
        <div class="desc">Multinomial Naive Bayes assumes non-negative, count-like input. The scaled, centered stylometric features (some negative after standardization) don't fit that assumption, so Naive Bayes trains on word content alone while the other two get the combined set.</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="decision-card">
        <div class="title">Why class_weight is balanced</div>
        <div class="desc">The dataset is 62.8% human, 37.2% AI — not severe, but enough to bias a model toward the majority class without correction. All three models weight classes inversely to frequency during training.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="decision-card">
        <div class="title">Why TF-IDF is capped at 5,000 features</div>
        <div class="desc">A larger vocabulary would help Naive Bayes, but Random Forest gets slow and memory-heavy on very high-dimensional sparse input. 5,000 keeps every model trainable on the same feature set within a normal Colab session.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-eyebrow">Honest limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Where this can be wrong</div>', unsafe_allow_html=True)

    l1, l2, l3 = st.columns(3)
    limits = [
        ("Shortcut risk", "word_count is the second-strongest stylometric feature. Part of what the model learned is genuinely about writing style — part of it is this dataset's AI outputs tending to run shorter, which may not hold for other AI writing tools."),
        ("Single dataset", "Trained and tested on one AI vs Human Text dataset. A different AI model family, prompt style, or writing domain could shift these numbers in either direction."),
        ("Simple sentence splitting", "Sentence boundaries are detected with a basic regex on . ! ? — not a proper NLP sentence tokenizer — abbreviations and edge cases can slightly skew the sentence-length features."),
    ]
    for col, (title, desc) in zip([l1, l2, l3], limits):
        with col:
            st.markdown(f"""
            <div class="limit-card">
                <span class="limit-tag">{title}</span>
                <div class="desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-stack">
    <span class="stack-pill">scikit-learn</span>
    <span class="stack-pill">polars</span>
    <span class="stack-pill">TF-IDF</span>
    <span class="stack-pill">streamlit</span>
</div>
""", unsafe_allow_html=True)
st.caption("Trained on Google Colab · AI vs Human Text dataset")
