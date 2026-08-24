import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import torch
import hashlib
import colorsys
import io
import time

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO COCO Visual Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#0D9488"

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .block-container {{padding-top: 2rem; max-width: 1200px;}}
    div[data-testid="stMetric"] {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stFileUploaderDropzone"] {{
        border: 2px dashed #CBD5E1;
        border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
    div.stButton > button, div[data-testid="stDownloadButton"] > button {{
        background-color: {ACCENT};
        color: white;
        border: none;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model + inference — the expensive work is cached so UI interactions stay cheap
# ----------------------------------------------------------------------------
MODEL_OPTIONS = {
    "YOLOv8 Nano — fastest": "yolov8n.pt",
    "YOLOv8 Small — balanced": "yolov8s.pt",
    "YOLOv8 Medium — most accurate": "yolov8m.pt",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_file: str) -> YOLO:
    return YOLO(weights_file)


@st.cache_resource(show_spinner=False)
def load_font():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15
        )
    except Exception:
        return ImageFont.load_default()


@st.cache_data(show_spinner=False, max_entries=20)
def detect(image_bytes: bytes, weights_file: str, image_size: int, iou: float):
    """
    Runs once per unique (image, model, resolution, IoU) combo, then is cached.
    Confidence is deliberately NOT part of inference — boxes are captured at a
    low floor (0.05) so the UI can filter by confidence afterwards for free.
    IoU *does* need a re-run when changed: it drives NMS inside the model, so
    it determines which boxes exist at all, not just which ones are shown.
    """
    model = load_model(weights_file)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(
        image, conf=0.05, iou=iou, imgsz=image_size, verbose=False
    )[0]
    boxes = [
        {
            "class": model.names[int(b.cls)],
            "confidence": float(b.conf),
            "xyxy": [round(v, 1) for v in b.xyxy[0].tolist()],
        }
        for b in results.boxes
    ]
    return image, boxes


def class_color(name: str) -> tuple:
    """Deterministic, distinct color per class."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360, 0.6, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


def render_boxes(image: Image.Image, boxes: list, threshold: float, allowed: list):
    """Cheap, local redraw with no model call — sliders/filters update instantly."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = load_font()
    visible = [
        b
        for b in boxes
        if b["confidence"] >= threshold and (not allowed or b["class"] in allowed)
    ]
    for b in visible:
        x1, y1, x2, y2 = b["xyxy"]
        color = class_color(b["class"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f'{b["class"]} {b["confidence"]:.0%}'
        tb = draw.textbbox((x1, y1), label, font=font)
        tag_top = max(0, y1 - (tb[3] - tb[1]) - 8)
        draw.rectangle([x1, tag_top, tb[2] + 8, y1], fill=color)
        draw.text((x1 + 4, tag_top + 2), label, fill="white", font=font)
    return canvas, visible


# ----------------------------------------------------------------------------
# Sidebar — every control lives here so the main area is just results
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Detection settings")

    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    weights_file = MODEL_OPTIONS[model_label]
    active_model = load_model(weights_file)
    known_classes = sorted(active_model.names.values())

    image_size = st.select_slider(
        "Inference resolution",
        options=[320, 480, 640, 960],
        value=640,
        help="Lower = faster, higher = better on small objects",
    )
    conf_threshold = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="Filters instantly — no re-detection needed",
    )
    iou_threshold = st.slider(
        "IoU threshold", 0.10, 0.90, 0.45, 0.05,
        help="Controls duplicate-box suppression. Changing this re-runs detection.",
    )
    class_filter = st.multiselect(
        "Limit to classes (optional)", known_classes,
        help="Leave empty to show every class the model knows",
    )

    st.divider()
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.caption(f"Running on: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
st.title("🔍 YOLO COCO Visual Search")
st.caption("Object detection powered by YOLOv8, trained on the COCO dataset.")

if uploaded_file is None:
    st.info("👈 Upload an image from the sidebar to get started.")
    st.stop()

image_bytes = uploaded_file.getvalue()

try:
    with st.spinner("Running detection..."):
        t0 = time.perf_counter()
        original, all_boxes = detect(image_bytes, weights_file, image_size, iou_threshold)
        elapsed = time.perf_counter() - t0
except Exception as e:
    st.error(f"Couldn't process that image: {e}")
    st.stop()

annotated, visible_boxes = render_boxes(original, all_boxes, conf_threshold, class_filter)

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.markdown("**Original**")
    st.image(original, width="stretch")
with col2:
    st.markdown("**Detected**")
    st.image(annotated, width="stretch")

st.divider()

if not visible_boxes:
    st.info(
        "No objects match the current filters — try lowering the confidence "
        "threshold or clearing the class filter in the sidebar."
    )
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objects detected", len(visible_boxes))
    m2.metric("Unique classes", len({b["class"] for b in visible_boxes}))
    m3.metric(
        "Avg. confidence", f'{np.mean([b["confidence"] for b in visible_boxes]):.0%}'
    )
    m4.metric("Detection time", f"{elapsed*1000:.0f} ms")

    tab_details, tab_chart = st.tabs(["📋 Details", "📊 Class counts"])

    with tab_details:
        df = (
            pd.DataFrame(visible_boxes)[["class", "confidence"]]
            .rename(columns={"class": "Object", "confidence": "Confidence"})
            .sort_values("Confidence", ascending=False)
        )
        df["Confidence"] = df["Confidence"].map(lambda x: f"{x:.1%}")
        st.dataframe(df, width="stretch", hide_index=True)

    with tab_chart:
        counts = pd.Series([b["class"] for b in visible_boxes]).value_counts()
        st.bar_chart(counts)

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    st.download_button(
        "⬇ Download annotated image",
        buf.getvalue(),
        "detected.png",
        "image/png",
    )