"""
ANN Model Explorer — Interactive Dark-Themed Streamlit App
------------------------------------------------------------
Loads a Keras/TensorFlow ANN (ANN_model.h5) and lets the user
play with predictions through an animated, dark-themed UI.

Run with:
    streamlit run app.py
"""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ANN Model Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CONFIG — edit these if your model differs from the Iris-style default
# --------------------------------------------------------------------------
MODEL_PATH = "ANN_model.h5"

FEATURE_NAMES = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]
FEATURE_RANGES = [(0.0, 10.0, 5.1), (0.0, 6.0, 3.5), (0.0, 8.0, 1.4), (0.0, 3.0, 0.2)]  # (min, max, default)
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]
CLASS_EMOJIS = ["🌸", "🌺", "🌷"]

# --------------------------------------------------------------------------
# CUSTOM CSS — dark theme, glow, gradients, keyframe animations
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=JetBrains+Mono&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #0B0F19, #131A2A, #1A1033, #0B0F19);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Floating particles */
    .particle {
        position: fixed;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(124,58,237,0.55), transparent 70%);
        pointer-events: none;
        animation: floatUp linear infinite;
        z-index: 0;
    }
    @keyframes floatUp {
        0%   { transform: translateY(0) translateX(0) scale(0.8); opacity: 0; }
        10%  { opacity: 0.8; }
        100% { transform: translateY(-110vh) translateX(30px) scale(1.2); opacity: 0; }
    }

    /* Glowing gradient title */
    .glow-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #A78BFA, #60A5FA, #34D399, #A78BFA);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 6s linear infinite, fadeInDown 1s ease;
        text-shadow: 0 0 30px rgba(124,58,237,0.35);
        margin-bottom: 0;
    }
    @keyframes shine {
        to { background-position: 300% center; }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-25px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .subtitle {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.05rem;
        animation: fadeIn 1.6s ease;
        margin-top: -6px;
    }
    @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        animation: fadeInUp 0.8s ease;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(124,58,237,0.25);
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7C3AED, #4F46E5);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        border: none;
        border-radius: 14px;
        box-shadow: 0 0 20px rgba(124,58,237,0.45);
        transition: all 0.25s ease;
        animation: pulse 2.5s infinite;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 35px rgba(124,58,237,0.75);
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 15px rgba(124,58,237,0.35); }
        50%  { box-shadow: 0 0 30px rgba(124,58,237,0.65); }
        100% { box-shadow: 0 0 15px rgba(124,58,237,0.35); }
    }

    /* Predicted class banner */
    .result-banner {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        padding: 1.1rem;
        border-radius: 16px;
        background: linear-gradient(90deg, rgba(124,58,237,0.25), rgba(52,211,153,0.18));
        border: 1px solid rgba(255,255,255,0.1);
        animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        margin-bottom: 1rem;
    }
    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.7); }
        70%  { transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Custom animated probability bars */
    .prob-row { margin-bottom: 14px; }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        color: #E5E7EB;
        margin-bottom: 4px;
    }
    .prob-track {
        width: 100%;
        height: 14px;
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #7C3AED, #34D399);
        width: 0%;
        animation: growBar 1.2s ease forwards;
        box-shadow: 0 0 12px rgba(124,58,237,0.6);
    }
    @keyframes growBar { to { width: var(--target-width); } }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>

    <div class="particle" style="left:5%;  width:14px; height:14px; animation-duration:14s; animation-delay:0s;"></div>
    <div class="particle" style="left:20%; width:8px;  height:8px;  animation-duration:11s; animation-delay:2s;"></div>
    <div class="particle" style="left:38%; width:20px; height:20px; animation-duration:17s; animation-delay:1s;"></div>
    <div class="particle" style="left:55%; width:10px; height:10px; animation-duration:13s; animation-delay:3s;"></div>
    <div class="particle" style="left:70%; width:16px; height:16px; animation-duration:15s; animation-delay:0.5s;"></div>
    <div class="particle" style="left:85%; width:9px;  height:9px;  animation-duration:12s; animation-delay:4s;"></div>
    <div class="particle" style="left:95%; width:13px; height:13px; animation-duration:16s; animation-delay:2.5s;"></div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown('<div class="glow-title">🧠 ANN Model Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive dark-themed inference dashboard for your trained neural network</div>',
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return tf.keras.models.load_model(path)


with st.spinner("Loading neural network..."):
    try:
        model = load_model(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        model_loaded = False
        load_error = str(e)

if not model_loaded:
    st.error(
        f"Couldn't load `{MODEL_PATH}`. Make sure it's in the same folder as app.py.\n\n"
        f"Details: {load_error}"
    )
    st.stop()

n_features = model.input_shape[-1]
n_classes = model.output_shape[-1]

# --------------------------------------------------------------------------
# SIDEBAR — model info + about
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model Info")
    st.markdown(
        f"""
        <div class="glass-card">
        <b>Architecture</b><br>
        {" → ".join(
            [f"Input({n_features})"] +
            [f"Dense({l.units}, {l.activation.__name__})" for l in model.layers if hasattr(l, "units")]
        )}
        <br><br>
        <b>Total params:</b> {model.count_params():,}<br>
        <b>Inputs:</b> {n_features}<br>
        <b>Output classes:</b> {n_classes}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("### 🎨 Options")
    show_history = st.checkbox("Track prediction history", value=True)
    show_raw = st.checkbox("Show raw output vector", value=False)
    st.write("")
    st.markdown(
        '<div class="subtitle" style="font-size:0.8rem;">Built with Streamlit + TensorFlow</div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# INPUT PANEL
# --------------------------------------------------------------------------
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🎛️ Input Features")

    values = []
    for i in range(n_features):
        if i < len(FEATURE_NAMES):
            name = FEATURE_NAMES[i]
            lo, hi, default = FEATURE_RANGES[i]
        else:
            name = f"Feature {i + 1}"
            lo, hi, default = 0.0, 10.0, 5.0
        v = st.slider(name, min_value=float(lo), max_value=float(hi), value=float(default), step=0.1)
        values.append(v)

    st.write("")
    predict_clicked = st.button("🚀 Run Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# PREDICTION + RESULTS
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Prediction")

    if predict_clicked:
        progress = st.progress(0, text="Feeding data through the network...")
        for pct in range(0, 101, 20):
            time.sleep(0.05)
            progress.progress(pct, text="Feeding data through the network...")
        progress.empty()

        x = np.array([values], dtype=np.float32)
        preds = model.predict(x, verbose=0)[0]

        top_idx = int(np.argmax(preds))
        top_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class {top_idx}"
        top_emoji = CLASS_EMOJIS[top_idx] if top_idx < len(CLASS_EMOJIS) else "🔹"
        confidence = float(preds[top_idx]) * 100

        st.markdown(
            f'<div class="result-banner">{top_emoji} Predicted: {top_label} '
            f'<span style="color:#34D399;">({confidence:.1f}% confidence)</span></div>',
            unsafe_allow_html=True,
        )

        # animated custom bars
        for i, p in enumerate(preds):
            label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
            pct = p * 100
            st.markdown(
                f"""
                <div class="prob-row">
                  <div class="prob-label"><span>{label}</span><span>{pct:.1f}%</span></div>
                  <div class="prob-track">
                    <div class="prob-fill" style="--target-width:{pct}%;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # plotly polar/bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}" for i in range(n_classes)],
                    y=preds * 100,
                    marker=dict(
                        color=preds * 100,
                        colorscale=[[0, "#7C3AED"], [1, "#34D399"]],
                        line=dict(color="rgba(255,255,255,0.15)", width=1),
                    ),
                    text=[f"{p*100:.1f}%" for p in preds],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E5E7EB"),
            yaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.08)", range=[0, 105]),
            xaxis=dict(title=""),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            transition=dict(duration=500, easing="cubic-in-out"),
        )
        st.plotly_chart(fig, use_container_width=True)

        if confidence > 90:
            st.balloons()

        if show_raw:
            st.markdown("**Raw output vector:**")
            st.code(np.array2string(preds, precision=4, separator=", "), language="text")

        if show_history:
            st.session_state.history.insert(
                0, {"inputs": values.copy(), "prediction": top_label, "confidence": f"{confidence:.1f}%"}
            )
    else:
        st.info("Set your input features on the left, then hit **Run Prediction** ✨")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# HISTORY TABLE
# --------------------------------------------------------------------------
if show_history and st.session_state.history:
    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🕓 Prediction History")
    st.dataframe(
        [
            {
                **{FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"F{i+1}": round(row["inputs"][i], 2)
                   for i in range(len(row["inputs"]))},
                "Prediction": row["prediction"],
                "Confidence": row["confidence"],
            }
            for row in st.session_state.history
        ],
        use_container_width=True,
        hide_index=True,
    )
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
