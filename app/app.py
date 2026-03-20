import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="News Topic Classifier",
    layout="centered"
)

@st.cache_resource
def load_model():
    MODEL_PATH = "D:/news_classifier/model/news_classifier_model"

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model folder not found at: {MODEL_PATH}")
        st.stop()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    return tokenizer, model

tokenizer, model = load_model()

labels = ["World", "Sports", "Business", "Sci/Tech"]

st.title("News Topic Classifier")

headline = st.text_area("News Headline")

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs).item()

    return pred, probs[0]

if st.button("Classify"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        pred, probs = predict(headline)

        st.success(f"Predicted Topic: {labels[pred]}")

        st.subheader("Confidence Scores")

        for i, label in enumerate(labels):
            st.progress(float(probs[i]))
            st.write(f"{label}: {probs[i].item():.3f}")
