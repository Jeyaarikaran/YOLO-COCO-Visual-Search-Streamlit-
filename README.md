# YOLO COCO Visual Search

## 1. Project Title
**YOLO COCO Visual Search**: An Interactive Object Detection Web Application using YOLOv8 and Streamlit.

**Name:** Jeyaarikaran P  
**Register Number:** 21222424006

## 2. Abstract / Introduction
This project implements a real-time object detection web application using the state-of-the-art YOLOv8 (You Only Look Once) model. Built with Streamlit, the application provides an intuitive user interface that allows users to upload images and instantly detect objects within them. The tool is designed to be highly responsive, caching the expensive inference steps and allowing users to dynamically adjust confidence and IoU (Intersection over Union) thresholds to visualize results instantaneously without re-running the model. It serves as a practical demonstration of deploying deep learning models for computer vision tasks in a user-friendly manner.

## 3. Dataset & YOLO Model Details (COCO)
- **Model**: YOLOv8 (Nano, Small, and Medium variants are supported for a balance between speed and accuracy). YOLOv8 is a cutting-edge, anchor-free object detection model developed by Ultralytics.
- **Dataset**: The model is pre-trained on the **COCO (Common Objects in Context)** dataset, which contains over 330k images and 1.5 million object instances spanning 80 distinct object categories (such as person, car, dog, etc.). 
- **Capabilities**: It provides bounding box coordinates, class labels, and confidence scores for each detected object.

## 4. Environment Setup
To run this project, you need to set up a Python environment. We recommend using **Conda** for package and environment management. Ensure you have Anaconda or Miniconda installed on your system.

**Prerequisites:**
- Python 3.9+
- Conda installed
- Visual Studio Code (VS Code)

## 5. CPU / GPU Installation Steps

**Step 1: Create a Conda Environment**
```bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
```

**Step 2: Install PyTorch**
*For CPU Users:*
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
*For GPU (CUDA) Users (adjust the CUDA version based on your system, e.g., 11.8):*
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 3: Install Remaining Dependencies**
```bash
pip install -r requirements.txt
```

*(Note: The requirements.txt includes `streamlit`, `ultralytics`, `pillow`, `numpy`, and `pandas`.)*

## 6. How to Run in VS Code using Conda
1. Open **Visual Studio Code (VS Code)**.
2. Open the folder containing the project files (e.g., `DL WORKSHOP`).
3. Open a new terminal in VS Code (`Terminal` -> `New Terminal`).
4. Activate your conda environment in the terminal:
   ```bash
   conda activate yolo_env
   ```
5. Ensure your terminal prompt reflects the activated environment (e.g., `(yolo_env)`).

## 7. How to Deploy using Streamlit
Once the environment is active and dependencies are installed, deploy the web application locally by running:
```bash
streamlit run app.py
```
This command will start the Streamlit server and automatically open the application in your default web browser (typically at `http://localhost:8501`).

## 8. Enhancements / Innovations Added
- **Dynamic Threshold Filtering**: Implemented a custom rendering function using `PIL` that redraws bounding boxes instantly when confidence or class filters are adjusted, avoiding the need to re-run the heavy YOLO model inference.
- **Color-Coded Classes**: Built a deterministic hashing function to assign unique, distinct colors (via HSV color space) to different object classes for better visual distinction.
- **Analytics Dashboard**: Integrated an interactive data tab showing real-time metrics (object counts, average confidence, detection time) and a bar chart of class distributions using `pandas` and Streamlit's native charting.
- **Performance Optimization**: Heavily utilized Streamlit's `@st.cache_resource` and `@st.cache_data` to cache model loading and inference, ensuring a buttery-smooth user experience.



## 9. Appendix: `app.py` Source Code

```python
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
    \"\"\"
    Runs once per unique (image, model, resolution, IoU) combo, then is cached.
    Confidence is deliberately NOT part of inference — boxes are captured at a
    low floor (0.05) so the UI can filter by confidence afterwards for free.
    IoU *does* need a re-run when changed: it drives NMS inside the model, so
    it determines which boxes exist at all, not just which ones are shown.
    \"\"\"
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
    \"\"\"Deterministic, distinct color per class.\"\"\"
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360, 0.6, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


def render_boxes(image: Image.Image, boxes: list, threshold: float, allowed: list):
    \"\"\"Cheap, local redraw with no model call — sliders/filters update instantly.\"\"\"
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
```

## 10. Output Screenshots

- **Conda Environment Activation in VS Code:**

  <img width="1102" height="496" alt="image" src="https://github.com/user-attachments/assets/ef3ad4b1-506c-43c7-9b00-93fcd21b710e" />


- **Running Streamlit in Terminal:**

  <img width="1077" height="202" alt="image" src="https://github.com/user-attachments/assets/a564718e-2a5e-4cc9-b2d6-e3386718c9e9" />


- **Streamlit Web UI in Browser:**

  <img width="1917" height="1107" alt="image" src="https://github.com/user-attachments/assets/67d9412f-3806-4714-8f1c-277c8dfd4ccc" />


- **Object Detection Result Screen:**

  ### JET :

  ##### Original Image :

  
  <img width="618" height="495" alt="jj" src="https://github.com/user-attachments/assets/4cc6423c-3cad-4083-abde-5b6687b7914e" />


  ##### Detected Image :
  
  
  <img width="618" height="495" alt="detected jet" src="https://github.com/user-attachments/assets/92c9c13e-78de-4428-bc72-e728a011c09c" />


  ### DOG :

  ##### Original Image :
  

  <img width="423" height="472" alt="tamilnadi rajapalayam" src="https://github.com/user-attachments/assets/6965fe6e-249b-4d94-bb91-348df0adc657" />

 ##### Detected Image :


 <img width="423" height="472" alt="detected (1)" src="https://github.com/user-attachments/assets/20567c27-4c76-4840-be64-60c1608f082b" />


 ## 11. Results & Conclusion
The YOLO COCO Visual Search application successfully demonstrates the integration of a powerful computer vision model (YOLOv8) with a modern web framework (Streamlit). The application runs efficiently, providing high-accuracy detections on various images with minimal latency. By separating the heavy model inference from the UI rendering, the tool offers a highly responsive and interactive experience. This project illustrates how deep learning can be made accessible and interactive for end-users without requiring deep technical knowledge.

