import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import os
from datetime import datetime

st.set_page_config(page_title="AI DRIVEN CROP DISEASE DETECTION MANAGEMENT SYSTEM", layout = 'wide')

SAMPLE_IMAGE = "/mnt/data/A_digital_photograph_showcases_a_single_green_leaf.png"


st.markdown("""
<style>
.header { background: linear-gradient(90deg, #2ecc71, #16a085); padding:18px; border-radius:12px; color: white; text-align:center; }
.card { background: white; border-radius:10px; padding:14px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
.small { font-size:0.9rem; color: #555; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'><h1>🌾AI DRIVEN CROP DISEASE DETECTION MANAGEMENT SYSTEM</h1><div class='small'>Visual diagnosis + mask + heatmap + report</div></div>", unsafe_allow_html=True)
st.write("---")

# ---------- Helpers ----------
def detect_disease_percent_and_masks(np_img, dark_thresh_v=75):
    # Preprocess: equalize or enhance green contrast
    lab = cv2.cvtColor(np_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, dark_thresh_v])
    mask_dark = cv2.inRange(hsv, lower, upper)

    # small morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_clean = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    diseased = np.count_nonzero(mask_clean)
    total = np_img.shape[0] * np_img.shape[1]
    percent = (diseased / total) * 100

    # heatmap overlay (normalized)
    heat = cv2.GaussianBlur(mask_clean, (25,25), 0)
    heat_norm = cv2.normalize(heat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heat_norm.astype('uint8'), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(np_img, 0.7, heatmap_color, 0.3, 0)

    return percent, mask_clean, overlay

def pil_to_cv2(pil_img):
    arr = np.array(pil_img)
    return arr[:, :, ::-1]  # RGB->BGR

def cv2_to_pil(cv_img):
    return Image.fromarray(cv_img[:, :, ::-1])

def get_download_link_bytes(bytes_data, filename, label):
    b64 = base64.b64encode(bytes_data).decode()
    return f"data:application/octet-stream;base64,{b64}"

# ---------- Sidebar (controls) ----------
st.sidebar.header("Settings")
use_sample = st.sidebar.checkbox("Use sample image", value=True)
uploaded = st.sidebar.file_uploader("Upload leaf image", type=["jpg","jpeg","png"])
threshold = st.sidebar.slider("Diseased threshold (%)", 1, 25, 10)
dark_thresh_v = st.sidebar.slider("Mask darkness V threshold", 20, 120, 75)
show_heatmap = st.sidebar.checkbox("Show heatmap overlay", value=True)
show_mask = st.sidebar.checkbox("Show mask", value=True)
download_report = st.sidebar.checkbox("Enable downloadable report (CSV & simple TXT)", value=True)

# ---------- Load image ----------
if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")
    source_label = f"Uploaded: {getattr(uploaded, 'name', 'file')}"
elif use_sample and os.path.exists(SAMPLE_IMAGE):
    img_pil = Image.open(SAMPLE_IMAGE).convert("RGB")
    source_label = f"Sample: {SAMPLE_IMAGE}"
else:
    st.info("Please upload a leaf image or enable sample image in the sidebar.")
    st.stop()

# ---------- Main layout ----------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Input Image")
    st.image(img_pil, use_column_width=True)
    st.markdown(f"**Image source:** `{source_label}`")
    st.markdown(f"**Analysis time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

np_img = pil_to_cv2(img_pil)
percent, mask_clean, overlay = detect_disease_percent_and_masks(np_img, dark_thresh_v=dark_thresh_v)

with right_col:
    st.subheader("Analysis Results")
    st.metric("Diseased area (%)", f"{percent:.2f}%")
    if percent < threshold:
        st.success("🌿 Leaf is Healthy - Farmer can use this leaf.")
        st.info("No major disease pattern found based on color/spot detection.")
    else:
        st.error("Leaf is Diseased - Farmer should NOT use this leaf.")
        st.warning("Recommended: Spray Mancozeb or Copper-Oxychloride. Consult local expert for dosage.")

    # show visualizations
    st.markdown("### Visualizations")
    viz_cols = st.columns(2)
    with viz_cols[0]:
        st.markdown("**Bar: Healthy vs Diseased**")
        df = pd.DataFrame({"Category":["Healthy","Diseased"], "Percent":[100-percent, percent]})
        st.bar_chart(df.set_index("Category"))
    with viz_cols[1]:
        st.markdown("**Spot stats**")
        st.write(df)

st.markdown("---")
st.subheader("Detailed Visual Output")
vis1, vis2, vis3 = st.columns([1,1,1])
with vis1:
    st.markdown("**Mask (white = detected)**")
    if show_mask:
        st.image(Image.fromarray(mask_clean.astype('uint8')), use_column_width=True)
with vis2:
    st.markdown("**Heatmap overlay**")
    if show_heatmap:
        st.image(cv2_to_pil(overlay), use_column_width=True)
with vis3:
    st.markdown("**Masked area zoom**")
    # create small zoom of masked region
    ys, xs = np.where(mask_clean > 0)
    if ys.size > 0:
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        pad = 10
        y1, y2 = max(0, y1-pad), min(mask_clean.shape[0], y2+pad)
        x1, x2 = max(0, x1-pad), min(mask_clean.shape[1], x2+pad)
        zoom_img = np_img[y1:y2, x1:x2]
        st.image(Image.fromarray(zoom_img[:, :, ::-1]), use_column_width=True)
    else:
        st.info("No marked spots to zoom.")

# ---------- Report & Download ----------
if download_report:
    st.markdown("---")
    st.subheader("Downloadable Report")
    # CSV of basic stats
    df_report = pd.DataFrame([{
        "image_source": source_label,
        "analyzed_at": datetime.now().isoformat(),
        "diseased_percent": float(percent),
        "threshold_used": threshold
    }])
    csv_bytes = df_report.to_csv(index=False).encode()
    st.download_button("📥 Download CSV report", data=csv_bytes, file_name="leaf_report.csv", mime="text/csv")

    # Save mask PNG to bytes
    buf = io.BytesIO()
    Image.fromarray(mask_clean.astype('uint8')).save(buf, format="PNG")
    mask_bytes = buf.getvalue()
    st.download_button("📥 Download detected mask (PNG)", data=mask_bytes, file_name="mask.png", mime="image/png")

    # Simple TXT "farmer readable" report
    txt = f"""Leaf Health Report
Source: {source_label}
Analyzed: {datetime.now().isoformat()}
Diseased area (%): {percent:.2f}
Verdict: {"HEALTHY" if percent < threshold else "DISEASED"}
Advice: {"No treatment needed." if percent < threshold else "Spray Mancozeb or Copper-Oxychloride - consult expert for dosage."}
"""
    st.download_button("📥 Download Farmer Report (TXT)", data=txt.encode(), file_name="farmer_report.txt", mime="text/plain")

st.markdown("---")
st.markdown("### Notes & Next steps")
st.write("""
- This system is color/spot based — for final clinical accuracy, train a multi-class model with labelled diseased images.
- You can increase robustness by: background removal, color normalization, and adding morphological checks.
- I can add a Kaggle dataset + training notebook and final model integration if you want a full AI model pipeline.
""")

