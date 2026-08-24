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

## 8. Output Screenshots
*(Please add your screenshots to the `Screenshots` folder in this repository and update the paths below if necessary.)*

- **Conda Environment Activation in VS Code:**
  ![Conda Activation](Screenshots/conda_activation.png)

- **Running Streamlit in Terminal:**
  ![Streamlit Run](Screenshots/streamlit_run.png)

- **Streamlit Web UI in Browser:**
  ![Web UI](Screenshots/web_ui.png)

- **Object Detection Result Screen:**
  ![Detection Result](Screenshots/detection_result.png)

## 9. Enhancements / Innovations Added
- **Dynamic Threshold Filtering**: Implemented a custom rendering function using `PIL` that redraws bounding boxes instantly when confidence or class filters are adjusted, avoiding the need to re-run the heavy YOLO model inference.
- **Color-Coded Classes**: Built a deterministic hashing function to assign unique, distinct colors (via HSV color space) to different object classes for better visual distinction.
- **Analytics Dashboard**: Integrated an interactive data tab showing real-time metrics (object counts, average confidence, detection time) and a bar chart of class distributions using `pandas` and Streamlit's native charting.
- **Performance Optimization**: Heavily utilized Streamlit's `@st.cache_resource` and `@st.cache_data` to cache model loading and inference, ensuring a buttery-smooth user experience.

## 10. Results & Conclusion
The YOLO COCO Visual Search application successfully demonstrates the integration of a powerful computer vision model (YOLOv8) with a modern web framework (Streamlit). The application runs efficiently, providing high-accuracy detections on various images with minimal latency. By separating the heavy model inference from the UI rendering, the tool offers a highly responsive and interactive experience. This project illustrates how deep learning can be made accessible and interactive for end-users without requiring deep technical knowledge.
