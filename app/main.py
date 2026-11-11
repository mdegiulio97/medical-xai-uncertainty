"""Chest X-ray: explainable classifier with uncertainty (Streamlit app).

Run with:  uv run streamlit run app/main.py
Requires a trained checkpoint: uv run python scripts/train.py

The heavy inference (data download + model) starts only when the button is clicked,
so the first render stays light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results"
CKPT = RESULTS / "model.pt"

st.set_page_config(
    page_title="Explainable, uncertainty-aware medical image classification", layout="wide"
)
st.title("Explainable, uncertainty-aware medical image classification")
st.caption(
    "Chest X-ray, pneumonia vs normal on PneumoniaMNIST. Shows calibrated confidence, "
    "MC-Dropout uncertainty and Grad-CAM. Discriminative model (not generative)."
)

with st.sidebar:
    st.header("Parameters")
    idx = st.number_input("Test image index", min_value=0, value=0, step=1)
    n_mc = st.slider("MC-Dropout samples", 5, 50, 20, step=5)
    unc_thr = st.slider("'High uncertainty' threshold", 0.0, 0.3, 0.1, step=0.01)
    run = st.button("Analyse", type="primary")

if not CKPT.exists():
    st.info(
        "No trained model found. Run first:\n\n"
        "```\nuv run python scripts/download_data.py\n"
        "uv run python scripts/train.py --epochs 3\n```"
    )
    st.stop()


@st.cache_resource
def get_model_and_data():
    import torch

    from cxr_xai.data import load_split, loader_to_tensors
    from cxr_xai.model import build_model, get_device

    device = get_device()
    model = build_model(num_classes=2, dropout=0.3, pretrained=False)
    model.load_state_dict(torch.load(CKPT, map_location=device)["state_dict"])
    model.to(device)
    x_test, y_test = loader_to_tensors(load_split("test", shuffle=False))
    return model, x_test, y_test, device


if run:
    from cxr_xai.data import CLASS_NAMES, denormalize
    from cxr_xai.gradcam import GradCAM, resnet_target_layer
    from cxr_xai.uncertainty import mc_dropout_predict

    model, x_test, y_test, device = get_model_and_data()
    i = int(min(idx, len(x_test) - 1))
    x = x_test[i : i + 1]

    mean_prob, epistemic = mc_dropout_predict(model, x, n_samples=n_mc, device=device)
    p_pneumonia = float(mean_prob[0, 1])
    unc = float(epistemic[0])
    pred = CLASS_NAMES[int(p_pneumonia >= 0.5)]
    truth = CLASS_NAMES[int(y_test[i])]

    cam_fn = GradCAM(model, resnet_target_layer(model))
    cam = cam_fn(x.to(device))
    img = denormalize(x_test[i])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Image")
        st.image(img, clamp=True, use_container_width=True, caption=f"true: {truth}")
    with c2:
        st.subheader("Grad-CAM")
        overlay = np.stack([img] * 3, axis=-1)
        heat = np.zeros_like(overlay)
        heat[..., 0] = cam  # red channel = attention
        st.image(np.clip(0.6 * overlay + 0.4 * heat, 0, 1), use_container_width=True)
    with c3:
        st.subheader("Result")
        st.metric("Prediction", pred, f"p(pneumonia)={p_pneumonia:.2f}")
        st.metric("Epistemic uncertainty", f"{unc:.3f}")
        if unc >= unc_thr:
            st.warning("⚠️ High uncertainty: the prediction is not reliable.")
        else:
            st.success("Low uncertainty.")
        st.caption("Confidence calibrated with temperature scaling during evaluation.")
else:
    st.info("Set the index in the sidebar and press **Analyse**.")
