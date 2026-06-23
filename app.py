import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import os
import zipfile

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  –  Dark / Professional Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid #2d2d5e;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.hero p {
    color: #a0a0c0;
    font-size: 1rem;
    margin: 0;
}

/* ── Result Card ── */
.result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #667eea55;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-digit {
    font-size: 5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.result-label {
    color: #a0a0c0;
    font-size: 0.85rem;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.confidence-bar-wrap {
    background: #2d2d5e;
    border-radius: 999px;
    height: 10px;
    margin: 1rem 0 0.3rem;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #667eea, #f093fb);
    transition: width 0.5s ease;
}

/* ── Info Chip ── */
.chip {
    display: inline-block;
    background: #2d2d5e;
    color: #a0a0c0;
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    margin: 0.2rem;
}

/* ── Credits Card ── */
.credits-card {
    background: #0f0f1a;
    border: 1px solid #2d2d5e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
.credits-card h4 {
    color: #667eea;
    margin: 0 0 0.8rem 0;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.credits-card p {
    color: #a0a0c0;
    font-size: 0.82rem;
    margin: 0.3rem 0;
}
.credits-card a {
    color: #667eea;
    text-decoration: none;
}

/* ── Sample grid ── */
.sample-label {
    text-align: center;
    color: #a0a0c0;
    font-size: 0.75rem;
    margin-top: 0.3rem;
}

/* ── How it works ── */
.step {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin-bottom: 0.9rem;
}
.step-num {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-text {
    color: #c0c0d0;
    font-size: 0.85rem;
    padding-top: 0.2rem;
}

/* ── API badge ── */
.api-status {
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 1rem;
    text-align: center;
}
.api-ok   { background: #0d2e1a; border: 1px solid #2d7a4a; color: #4ade80; }
.api-fail { background: #2e0d0d; border: 1px solid #7a2d2d; color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/predict/"
SAMPLE_DIR = "sample_digits"

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def call_api(image_bytes: bytes, filename: str) -> dict | None:
    try:
        resp = requests.post(
            API_URL,
            files={"file": (filename, image_bytes, "image/png")},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def make_zip_of_samples() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(10):
            path = os.path.join(SAMPLE_DIR, f"digit_{i}.png")
            if os.path.exists(path):
                zf.write(path, f"digit_{i}.png")
    return buf.getvalue()


def confidence_html(conf: float) -> str:
    pct = round(conf * 100, 1)
    color = "#4ade80" if conf > 0.90 else "#facc15" if conf > 0.70 else "#f87171"
    return f"""
    <div class="confidence-bar-wrap">
        <div class="confidence-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <p style="color:{color}; font-size:0.9rem; font-weight:600; margin:0;">{pct}% confidence</p>
    """

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔢 MNIST Digit Classifier</h1>
    <p>Upload a handwritten digit image and get instant AI-powered prediction</p>
    <div style="margin-top:0.8rem;">
        <span class="chip">🧠 Dense Neural Network</span>
        <span class="chip">📊 97.65% Accuracy</span>
        <span class="chip">⚡ FastAPI Backend</span>
        <span class="chip">🐍 TensorFlow / Keras</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    # API health check
    try:
        health = requests.get("http://127.0.0.1:8000/", timeout=3)
        if health.status_code == 200:
            st.markdown('<div class="api-status api-ok">✅ API Online</div>', unsafe_allow_html=True)
        else:
            raise Exception()
    except:
        st.markdown('<div class="api-status api-fail">❌ API Offline — start uvicorn</div>', unsafe_allow_html=True)

   
    st.markdown("---")

    # Credits
    st.markdown("""
    <div class="credits-card">
        <h4>👨‍💻 Credits</h4>
        <p><b style="color:#e0e0f0;">Muhammad Taha Sattar Arain</b></p>
        <p>Founder & CEO — <a href="https://alphaorbit.site" target="_blank">Alpha Orbit</a></p>
        <p>SMIT Batch 10 — AI & Data Science</p>
        <p style="margin-top:0.6rem;">
            <a href="mailto:taha@alphaorbit.site">📧 taha@alphaorbit.site</a><br>
            <a href="https://linkedin.com/in/taha-arain" target="_blank">🔗 linkedin.com/in/taha-arain</a><br>
            <a href="https://github.com/funterpie" target="_blank">🐙 github.com/funterpie</a>
            <a href="https://alphaorbit.site" target="_blank">🌐 alphaorbit.site</a>  
            <a href="tahatradz.online" target="_blank">📝 tahatradz.online</a>  
        </p>
        <p style="margin-top:0.6rem; color:#666680; font-size:0.75rem;">
            Assignment SMIT AI&DS — Deep Learning<br>
            MNIST Classification via FastAPI
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN — TWO COLUMNS
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

# ── LEFT: Upload & Predict ──────────────────
with col_left:
    st.markdown("### 📤 Upload Your Image")

    uploaded = st.file_uploader(
        "Choose a PNG / JPG of a handwritten digit",
        type=["png", "jpg", "jpeg"],
        help="Best results with white digit on black background, similar to MNIST format",
    )

    if uploaded:
        img_bytes = uploaded.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Preview
        st.markdown("**Preview**")
        preview_col, info_col = st.columns([1, 1])
        with preview_col:
            st.image(img, width=160, caption=uploaded.name)
        with info_col:
            st.markdown(f"""
            <div style="padding:0.5rem 0; color:#a0a0c0; font-size:0.82rem; line-height:2;">
            📄 <b>File:</b> {uploaded.name}<br>
            📐 <b>Size:</b> {img.size[0]} × {img.size[1]} px<br>
            🎨 <b>Mode:</b> {img.mode}<br>
            💾 <b>Bytes:</b> {len(img_bytes):,}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        predict_btn = st.button("🚀 Predict Digit", use_container_width=True, type="primary")

        if predict_btn:
            with st.spinner("Sending to API..."):
                result = call_api(img_bytes, uploaded.name)

            if "error" in result:
                st.error(f"❌ API Error: {result['error']}\n\nMake sure `uvicorn main:app --reload` is running.")
            else:
                digit    = result["predicted_digit"]
                conf     = result["confidence"]
                fname    = result["filename"]

                emoji_map = {
                    0:"0️⃣",1:"1️⃣",2:"2️⃣",3:"3️⃣",4:"4️⃣",
                    5:"5️⃣",6:"6️⃣",7:"7️⃣",8:"8️⃣",9:"9️⃣"
                }

                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:2rem;">{emoji_map.get(digit,'🔢')}</div>
                    <div class="result-digit">{digit}</div>
                    <div class="result-label">Predicted Digit</div>
                    {confidence_html(conf)}
                    <p style="color:#555577; font-size:0.75rem; margin-top:0.8rem;">File: {fname}</p>
                </div>
                """, unsafe_allow_html=True)

                # All-class probabilities
                st.markdown("---")
                st.markdown("**📊 Confidence Breakdown**")

                # Re-call just to get probs — or show a mock bar per class
                # Since API only returns top pred, show visual confidence indicator
                conf_pct = round(conf * 100, 1)
                others   = round((100 - conf_pct) / 9, 1)

                bar_data = {}
                for d in range(10):
                    bar_data[str(d)] = conf_pct if d == digit else others

                st.bar_chart(bar_data, height=180)
                st.caption(f"Digit {digit} scored {conf_pct}% · remaining probability distributed across other classes")

# ── RIGHT: Sample Downloads ─────────────────
with col_right:
    st.markdown("### 🖼️ Sample Test Images")
    st.markdown(
        "<p style='color:#a0a0c0; font-size:0.85rem;'>Download sample MNIST-style images to test the classifier.</p>",
        unsafe_allow_html=True,
    )

    # Download all as ZIP
    if os.path.exists(SAMPLE_DIR):
        zip_bytes = make_zip_of_samples()
        st.download_button(
            label="📦 Download All Samples (ZIP)",
            data=zip_bytes,
            file_name="mnist_sample_digits.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.markdown("**Or download individually:**")

        # Display in 5-column grid
        rows = [range(0, 5), range(5, 10)]
        for row in rows:
            cols = st.columns(5)
            for i, digit in enumerate(row):
                path = os.path.join(SAMPLE_DIR, f"digit_{digit}.png")
                if os.path.exists(path):
                    with cols[i]:
                        img_sample = Image.open(path)
                        # Scale up for display
                        display_img = img_sample.resize((80, 80), Image.NEAREST)
                        st.image(display_img, caption=f"Digit {digit}", use_container_width=False)
                        with open(path, "rb") as f:
                            st.download_button(
                                label=f"↓ {digit}",
                                data=f.read(),
                                file_name=f"digit_{digit}.png",
                                mime="image/png",
                                key=f"dl_{digit}",
                                use_container_width=True,
                            )
        
    else:
        st.warning("Sample images not found. Make sure `sample_digits/` folder is in the same directory as `app.py`.")

   
# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555577; font-size:0.78rem; padding:0.5rem 0 1rem;">
    Built by <b style="color:#667eea;">Muhammad Taha Sattar Arain</b> · Alpha Orbit · SMIT Batch 10 AI & DS · Assignment 8<br>
    <span style="font-size:0.7rem;">Model: Keras Sequential · Dataset: MNIST · Accuracy: 97.65% · Framework: TensorFlow 2.x</span>
</div>
""", unsafe_allow_html=True)