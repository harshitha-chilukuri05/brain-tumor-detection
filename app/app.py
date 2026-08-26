"""
app.py
------
Streamlit demo app for the Brain Tumor MRI Classifier.

Run with:
    streamlit run app/app.py

Lets a user upload an MRI scan, see the predicted class + confidence,
and view a Grad-CAM heatmap showing which regions of the scan influenced
the model's decision.
"""

import os
import sys

import streamlit as st
from PIL import Image
import torch

# Allow importing from src/ regardless of where streamlit is launched from
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model import load_checkpoint  # noqa: E402
from predict import predict_with_gradcam  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models", "best_model.pt")

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown(
    """
    Upload a brain MRI scan and this model will classify it into one of four
    categories: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**.

    The heatmap shows the regions the model focused on when making its
    prediction (Grad-CAM), so you can sanity-check *why* it made that call —
    not just trust a black-box number.
    """
)

st.warning(
    "⚠️ This is a portfolio/research project, not a medical device. "
    "It is not validated for clinical use and should never be used for "
    "actual diagnosis.",
    icon="⚠️",
)


@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(MODEL_PATH, device=device)
    return model, class_names, device


if not os.path.exists(MODEL_PATH):
    st.error(
        f"No trained model found at `{MODEL_PATH}`. "
        "Train one first with `python src/train.py` (see README.md)."
    )
    st.stop()

model, class_names, device = get_model()

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_upload.jpg"
    Image.open(uploaded_file).convert("RGB").save(temp_path)

    col1, col2 = st.columns(2)

    with st.spinner("Running model..."):
        pred_class, confidence, overlay = predict_with_gradcam(
            temp_path, model, class_names, device=device
        )

    with col1:
        st.subheader("Uploaded Scan")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, use_container_width=True)

    st.markdown("---")
    label_display = pred_class.replace("_", " ").title()
    if pred_class.lower() == "notumor":
        st.success(f"### Prediction: {label_display}")
    else:
        st.error(f"### Prediction: {label_display}")
    st.metric("Confidence", f"{confidence:.1%}")

    os.remove(temp_path)
else:
    st.info("Upload a JPG or PNG MRI scan to get started.")
