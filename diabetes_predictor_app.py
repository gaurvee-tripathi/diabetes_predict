import streamlit as st
import joblib
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌼 Diabetes Predictor",
    page_icon="🌼",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS (Chamkile / Sparkling Colors) ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Nunito:wght@400;600&display=swap');

/* ── Root Variables ── */
:root {
    --gold:       #FFD700;
    --amber:      #FFAA00;
    --coral:      #FF6B6B;
    --teal:       #00D4AA;
    --purple:     #A855F7;
    --pink:       #FF4ECD;
    --dark:       #0F0F1A;
    --card-bg:    #1A1A2E;
    --card2-bg:   #16213E;
    --text:       #F0F0FF;
    --muted:      #9999BB;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: var(--dark);
    color: var(--text);
}
.stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #1A0A2E 50%, #0A1A2E 100%);
    min-height: 100vh;
}

/* ── Animated background sparkles ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(circle at 20% 20%, rgba(255,215,0,0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(168,85,247,0.08) 0%, transparent 50%),
        radial-gradient(circle at 60% 20%, rgba(0,212,170,0.06) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hero Title ── */
.hero-title {
    text-align: center;
    padding: 2rem 0 0.5rem 0;
    font-family: 'Poppins', sans-serif;
}
.hero-title h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FFD700, #FF6B6B, #A855F7, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
    animation: shimmer 3s ease-in-out infinite;
    background-size: 300% 100%;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-title p {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.3rem;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Poppins', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--gold);
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--gold), transparent);
}

/* ── Sliders & Inputs ── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--teal), var(--purple)) !important;
}
.stSlider [data-baseweb="slider"] > div:first-child {
    background: rgba(255,255,255,0.1) !important;
}
.stNumberInput input {
    background: var(--card-bg) !important;
    border: 1px solid rgba(255,215,0,0.3) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
}
.stNumberInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 12px rgba(255,215,0,0.3) !important;
}

/* ── Predict Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #FFD700, #FF6B6B, #A855F7) !important;
    background-size: 200% 200% !important;
    color: #0F0F1A !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.9rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 30px rgba(255, 215, 0, 0.35) !important;
    animation: btn-glow 2s ease-in-out infinite !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 40px rgba(255, 107, 107, 0.5) !important;
}
@keyframes btn-glow {
    0%, 100% { box-shadow: 0 4px 30px rgba(255,215,0,0.35); }
    50%       { box-shadow: 0 4px 40px rgba(168,85,247,0.5); }
}

/* ── Result Cards ── */
.result-positive {
    background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(168,85,247,0.1));
    border: 1px solid rgba(255,107,107,0.6);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 40px rgba(255,107,107,0.2);
    animation: pulse-red 2s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 30px rgba(255,107,107,0.2); }
    50%       { box-shadow: 0 0 50px rgba(255,107,107,0.4); }
}
.result-negative {
    background: linear-gradient(135deg, rgba(0,212,170,0.15), rgba(255,215,0,0.08));
    border: 1px solid rgba(0,212,170,0.6);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 40px rgba(0,212,170,0.2);
    animation: pulse-teal 2s ease-in-out infinite;
}
@keyframes pulse-teal {
    0%, 100% { box-shadow: 0 0 30px rgba(0,212,170,0.2); }
    50%       { box-shadow: 0 0 50px rgba(0,212,170,0.4); }
}
.result-icon  { font-size: 4rem; }
.result-title { font-family:'Poppins',sans-serif; font-size:2rem; font-weight:800; margin: 0.5rem 0; }
.result-sub   { color: var(--muted); font-size: 0.95rem; }

/* ── Probability Bar ── */
.prob-bar-container {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    height: 14px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.prob-bar-fill-pos {
    height: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #FF6B6B, #FF4ECD);
    transition: width 0.8s ease;
}
.prob-bar-fill-neg {
    height: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #00D4AA, #FFD700);
    transition: width 0.8s ease;
}
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Info Cards ── */
.info-card {
    background: var(--card-bg);
    border: 1px solid rgba(255,215,0,0.15);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.6;
}
.info-card strong { color: var(--gold); }

/* ── Column gaps ── */
[data-testid="column"] { padding: 0 6px !important; }

/* ── Metric chips ── */
.metric-chip {
    background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(168,85,247,0.1));
    border: 1px solid rgba(255,215,0,0.25);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-chip .val  { font-family:'Poppins',sans-serif; font-size:1.4rem; font-weight:700; color:var(--gold); }
.metric-chip .lbl  { font-size:0.72rem; letter-spacing:1px; color:var(--muted); text-transform:uppercase; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

/* ── Disclaimer ── */
.disclaimer {
    text-align: center;
    font-size: 0.75rem;
    color: rgba(153,153,187,0.6);
    margin-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("naive_bayes_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ─── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    <h1>🌼 DiabeteSense</h1>
    <p>AI-powered Diabetes Risk Predictor · Gaussian Naïve Bayes</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model: {model_error}")
    st.stop()

st.markdown("---")

# ─── Input Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Patient Health Metrics</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "🤰 Pregnancies", min_value=0, max_value=20, value=1, step=1,
        help="Number of times pregnant"
    )
    glucose = st.number_input(
        "🍬 Glucose (mg/dL)", min_value=0, max_value=300, value=110, step=1,
        help="Plasma glucose concentration (2-hour oral glucose tolerance test)"
    )
    blood_pressure = st.number_input(
        "💉 Blood Pressure (mmHg)", min_value=0, max_value=180, value=72, step=1,
        help="Diastolic blood pressure"
    )
    skin_thickness = st.number_input(
        "📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1,
        help="Triceps skin fold thickness"
    )

with col2:
    insulin = st.number_input(
        "💊 Insulin (µU/mL)", min_value=0, max_value=1000, value=80, step=1,
        help="2-Hour serum insulin"
    )
    bmi = st.number_input(
        "⚖️ BMI (kg/m²)", min_value=0.0, max_value=80.0, value=25.0, step=0.1,
        help="Body Mass Index (weight in kg / height in m²)"
    )
    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35, step=0.01,
        help="Likelihood of diabetes based on family history"
    )
    age = st.number_input(
        "🎂 Age (years)", min_value=1, max_value=120, value=30, step=1,
        help="Age in years"
    )

st.markdown("---")

# ─── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("✨ Predict Diabetes Risk", use_container_width=True)

if predict_clicked:
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    prob_positive = proba[1]
    prob_negative = proba[0]

    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-icon">⚠️</div>
            <div class="result-title" style="color:#FF6B6B;">High Risk Detected</div>
            <div class="result-sub">The model predicts a <strong style="color:#FF6B6B;">Positive</strong> likelihood of Diabetes</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-icon">✅</div>
            <div class="result-title" style="color:#00D4AA;">Low Risk</div>
            <div class="result-sub">The model predicts a <strong style="color:#00D4AA;">Negative</strong> likelihood of Diabetes</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability Bars ──
    st.markdown('<div class="section-header">📊 Prediction Confidence</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <p style="color:#FF6B6B; font-weight:600; margin-bottom:4px;">🔴 Diabetic Risk: {prob_positive*100:.1f}%</p>
    <div class="prob-bar-container">
        <div class="prob-bar-fill-pos" style="width:{prob_positive*100:.1f}%;"></div>
    </div>
    <div class="prob-label"><span>0%</span><span>{prob_positive*100:.1f}%</span><span>100%</span></div>

    <p style="color:#00D4AA; font-weight:600; margin: 1rem 0 4px 0;">🟢 Non-Diabetic: {prob_negative*100:.1f}%</p>
    <div class="prob-bar-container">
        <div class="prob-bar-fill-neg" style="width:{prob_negative*100:.1f}%;"></div>
    </div>
    <div class="prob-label"><span>0%</span><span>{prob_negative*100:.1f}%</span><span>100%</span></div>
    """, unsafe_allow_html=True)

    # ── Metric Summary Chips ──
    st.markdown('<div class="section-header">📋 Input Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    chips = [
        (c1, glucose, "Glucose"),
        (c2, bmi, "BMI"),
        (c3, age, "Age"),
        (c4, blood_pressure, "BP (mmHg)"),
    ]
    for col, val, label in chips:
        with col:
            st.markdown(f"""
            <div class="metric-chip">
                <div class="val">{val}</div>
                <div class="lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Info Section ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this Model & Features"):
    st.markdown("""
    <div class="info-card">
        <strong>Model:</strong> Gaussian Naïve Bayes (GNB) — a probabilistic classifier that assumes features are normally distributed within each class.<br><br>
        <strong>Dataset:</strong> Based on the Pima Indians Diabetes Dataset (NIDDK).<br><br>
        <strong>Classes:</strong> 0 = Non-Diabetic &nbsp;|&nbsp; 1 = Diabetic<br><br>
        <strong>Features Used:</strong> Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
    </div>
    <div class="info-card">
        <strong>⚠️ Disclaimer:</strong> This tool is for <em>educational and demonstration purposes only</em>.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    🌼 Built with Streamlit · Scikit-learn Gaussian Naïve Bayes · For educational use only
</div>
""", unsafe_allow_html=True)
