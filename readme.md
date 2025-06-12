# Real-time and Offline Object Detection with YOLOv10

This project offers versatile object detection capabilities using the advanced YOLOv10 model. It allows you to perform real-time detection via your webcam, infer objects in static images, and process entire video files, displaying bounding boxes and confidence scores for detected objects.

https://github.com/YourUsername/YOLOv10-Object-Detection/assets/your-video-file.mp4

## Features

* **Real-time Webcam Detection:** Detects objects in a live video stream from your webcam.
* **Image Inference:** Processes static images to identify and label objects.
* **Video Inference:** Analyzes video files frame by frame, performing object detection and saving the output to a new video file.
* **YOLOv10 Integration:** Utilizes the highly efficient and accurate YOLOv10 model for robust object recognition.
* **Bounding Boxes & Labels:** Draws bounding boxes around detected objects and displays their class name and confidence score.
* **Configurable Model:** Easily switch between different YOLOv10 variants (e.g., `yolov10n.pt`, `yolov10s.pt`, `yolov10m.pt`).
* **Adjustable Confidence:** Allows setting a minimum confidence threshold for detections.

---

## Installation

Before running the project, you need to set up your environment and install the required libraries.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YOLOv10-Object-Detection.git](https://github.com/YourUsername/YOLOv10-Object-Detection.git)
    cd YOLOv10-Object-Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python ultralytics
    ```

3.  **Download YOLOv10 Model Weights:**
    The scripts require the YOLOv10 model weights. You can download the `yolov10n.pt` (nano version) or other variants from the Ultralytics GitHub repository or their official website. Place the downloaded `.pt` file in the same directory as your Python scripts.

    For example, for `yolov10n.pt`:
    [Download yolov10n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt) (Link may need to be updated to the latest release assets)

---

## How to Use

Ensure the YOLOv10 model weights (`yolov10n.pt` or your chosen model) are in the same directory as the Python scripts.

### 1. Real-time Webcam Detection

To perform object detection on your live webcam feed:

```bash
python webcam_inference.py