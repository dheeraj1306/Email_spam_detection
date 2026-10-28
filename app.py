import os
import pickle
import html

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Email Shield | Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #eef2ff; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
.hero { padding: 1.6rem 1.8rem; border: 1px solid #283453; border-radius: 22px;
        background: linear-gradient(120deg, #151f3a 0%, #10172b 60%, #172b38 100%); margin-bottom: 1.3rem; }
.eyebrow { color: #73e0c1; text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.45rem; line-height: 1.12; margin: .45rem 0 .55rem; color: #f4f7ff; }
.hero p { color: #aab7d3; font-size: 1rem; max-width: 700px; margin-bottom: 0; }
.panel { background: #111a30; border: 1px solid #263451; border-radius: 18px; padding: 1.25rem; }
.kicker { color: #91a3c8; font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
.metric { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; color: #f4f7ff; font-weight: 700; margin-top: .25rem; }
.muted { color: #9aa9c7; font-size: .9rem; }
.result { border-radius: 18px; padding: 1.4rem; border: 1px solid; margin-top: .6rem; }
.spam { background: #321c2a; border-color: #a94c6b; }
.ham { background: #122f30; border-color: #2b9b86; }
.result-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.65rem; font-weight: 700; margin: .35rem 0; }
div.stButton > button { border-radius: 10px; min-height: 2.8rem; font-weight: 700; }
div[data-testid="stTextArea"] textarea { background: #0c1428; color: #f4f7ff; border: 1px solid #2a3858; border-radius: 12px; }
div[data-testid="stTextInput"] input { background: #0c1428; color: #f4f7ff; }
section[data-testid="stSidebar"] { background: #0d1426; border-right: 1px solid #202d49; }
hr { border-color: #263451; }
.small-note { color: #7f90b2; font-size: .78rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(base, "gru_model.keras"), compile=False)
    with open(os.path.join(base, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(base, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(base, "label_mapping.pkl"), "rb") as f:
        labels = pickle.load(f)
    return model, tokenizer, config, labels

st.sidebar.markdown("## 🛡️ Email Shield")
st.sidebar.caption("Neural spam detection")
st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.markdown(
    "1. Paste an email message\\n"
    "2. The saved tokenizer converts words to sequences\\n"
    "3. Sequences are padded to the model's expected length\\n"
    "4. The trained GRU model predicts Spam or Ham"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model**  \\nGRU · Binary classifier")
st.sidebar.markdown("**Threshold**  \\n0.50")
st.sidebar.markdown("---")
st.sidebar.caption("Predictions are model estimates. Check suspicious links and requests manually.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered inbox security</div>
  <h1>Keep the noise out.<br>Spot spam before it lands.</h1>
  <p>Analyze an email message with your trained GRU model and get a clear Spam / Ham prediction with a confidence score.</p>
</div>
""", unsafe_allow_html=True)

try:
    model, tokenizer, config, labels = load_artifacts()
except Exception as e:
    st.error("The model files could not be loaded. Make sure all four model artifacts are in the same folder as app.py.")
    st.code(str(e))
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="panel"><div class="kicker">Model architecture</div><div class="metric">GRU Network</div><div class="muted">Sequence-based text classifier</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="panel"><div class="kicker">Vocabulary limit</div><div class="metric">{int(config.get("max_features", 5000)):,}</div><div class="muted">Tokenizer word-index limit</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="panel"><div class="kicker">Input sequence</div><div class="metric">{int(config.get("max_length", 500))} tokens</div><div class="muted">Post-padded for inference</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.markdown("### ✉️ Analyze an email")
    st.markdown('<div class="muted">Paste the subject and body, or try one of the sample messages.</div>', unsafe_allow_html=True)
    samples = {
        "Choose a sample…": "",
        "Likely spam — prize offer": "CONGRATULATIONS! You have won a free cash prize. Claim your reward now by clicking the link below. Limited time offer!",
        "Likely legitimate — meeting": "Hi team, just a reminder that our project meeting is scheduled for tomorrow at 10 AM. Please review the attached agenda before joining. Thanks.",
        "Likely spam — urgent account": "URGENT! Your account has been selected for a special bonus. Verify your details immediately to avoid losing access. Click here now!",
        "Likely legitimate — follow-up": "Hello, thank you for your message. I have reviewed the document and will send my feedback by Friday afternoon. Best regards."
    }
    chosen = st.selectbox("Load an example", list(samples.keys()))
    default_text = samples[chosen]
    email_text = st.text_area(
        "Email content",
        value=default_text,
        height=240,
        placeholder="Paste the email subject and body here…",
        help="The model was trained on text sequences. Include the message content for a more useful prediction."
    )
    predict = st.button("🔍  Check this email", type="primary", use_container_width=True)

with right:
    st.markdown("### 📊 Prediction")
    if predict:
        if not email_text.strip():
            st.warning("Enter an email message first.")
        else:
            sequence = tokenizer.texts_to_sequences([email_text])
            padded = pad_sequences(
                sequence,
                maxlen=int(config.get("max_length", 500)),
                padding="post",
                truncating="post"
            )
            probability = float(np.asarray(model.predict(padded, verbose=0)).reshape(-1)[0])
            predicted_class = 1 if probability >= 0.5 else 0
            label = labels.get(predicted_class, "Spam" if predicted_class else "Ham")
            is_spam = predicted_class == 1
            confidence = probability if is_spam else 1 - probability
            css_class = "spam" if is_spam else "ham"
            icon = "🚨" if is_spam else "✅"
            st.markdown(
                f'<div class="result {css_class}"><div class="kicker">Model decision</div>'
                f'<div class="result-title">{icon} {html.escape(str(label))}</div>'
                f'<div class="muted">Confidence in predicted class</div>'
                f'<div class="metric">{confidence:.1%}</div></div>',
                unsafe_allow_html=True
            )
            st.write("")
            st.progress(max(0.0, min(1.0, confidence)))
            st.caption(f"Raw spam score: {probability:.1%} · Decision threshold: 50%")
            if is_spam:
                st.warning("Treat this message cautiously. Avoid clicking links or sharing sensitive information until verified.")
            else:
                st.success("The model classified this message as likely legitimate. This is not a guarantee of safety.")
    else:
        st.markdown("""
        <div class="panel">
          <div style="font-size:2rem">🧭</div>
          <div class="result-title" style="font-size:1.25rem">Waiting for an email</div>
          <p class="muted">Your prediction, confidence score, and safety guidance will appear here after analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="panel"><div class="kicker">Prediction guide</div><p class="muted">Spam score ≥ 50% → Spam<br>Spam score &lt; 50% → Ham</p><div class="small-note">Confidence is based on the model score, not a calibrated probability of safety.</div></div>', unsafe_allow_html=True)

st.write("")
with st.expander("About this model"):
    st.write(
        "This interface loads the supplied GRU model, tokenizer, configuration, and label mapping. "
        "The email is tokenized using the saved training tokenizer and padded to the configured sequence length. "
        "The model outputs a sigmoid score; a threshold of 0.50 maps the score to Spam (1) or Ham (0)."
    )
st.markdown('<div class="small-note">Email Shield · Local inference · Keep your model artifacts private and do not use predictions as the only security control.</div>', unsafe_allow_html=True)
