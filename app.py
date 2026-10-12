"""Streamlit app. Run: streamlit run app.py"""
import json
import joblib
import streamlit as st
from predict import classify

st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Email Spam Classifier")


@st.cache_resource
def load():
    return joblib.load("models/spam_model.joblib"), json.load(open("models/metadata.json"))


model, meta = load()
st.caption(f"Model: {meta['best_model']} | Test accuracy: {meta['test_accuracy']:.1%} "
           f"| ROC-AUC: {meta['roc_auc']:.3f}")

EXAMPLES = {
    "Spam example": ("Congratulations!!! You have been selected",
                     "Claim your FREE prize now. Click http://claim-now.example and enter your bank details to receive $5000."),
    "Ham example": ("Meeting moved to Thursday",
                    "Hi team, the project review is now Thursday at 2pm. Please send me your status updates beforehand."),
}
choice = st.selectbox("Load an example", ["(write your own)"] + list(EXAMPLES))
subj_default, body_default = EXAMPLES.get(choice, ("", ""))

subject = st.text_input("Subject", value=subj_default, key=f"s_{choice}")
body = st.text_area("Body", value=body_default, height=200, key=f"b_{choice}")
uploaded = st.file_uploader("...or upload a .eml file", type=["eml"])

if st.button("Classify", type="primary"):
    if uploaded is not None:
        from email import policy
        from email.parser import BytesParser
        msg = BytesParser(policy=policy.default).parsebytes(uploaded.getvalue())
        part = msg.get_body(preferencelist=("plain", "html"))
        subject, body = str(msg["subject"] or ""), (part.get_content() if part else "")
    if not (subject or body).strip():
        st.warning("Enter some text first.")
    else:
        label, score = classify(model, subject, body)
        st.metric("Spam score", f"{score:.1%}")
        st.progress(min(max(score, 0.0), 1.0))
        (st.error if label == "SPAM" else st.success)(f"This looks like {label}")
