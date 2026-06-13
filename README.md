# AegisVision – AI-Powered PPE Compliance & Workplace Safety Monitoring

## Overview

AegisVision is a computer vision–based workplace safety monitoring system that uses YOLOv8 object detection to identify Personal Protective Equipment (PPE) compliance in construction and industrial environments.

The system analyzes images, videos, or live webcam feeds to detect workers, safety helmets, masks, safety vests, machinery, and workplace hazards. It helps improve workplace safety by automatically identifying PPE violations and generating alerts.

---

## Features

* Real-time PPE detection using YOLOv8
* Helmet and safety vest compliance monitoring
* Webcam, image, and video input support
* Workplace occupancy monitoring
* Automated violation detection framework
* Safety event logging
* Extensible rule-based alert system
* Custom training pipeline for PPE datasets

---

## Dataset

This project uses the **Construction Site Safety Image Dataset (Roboflow)** available on Kaggle.

### Classes

| ID | Class          |
| -- | -------------- |
| 0  | Hardhat        |
| 1  | Mask           |
| 2  | NO-Hardhat     |
| 3  | NO-Mask        |
| 4  | NO-Safety Vest |
| 5  | Person         |
| 6  | Safety Cone    |
| 7  | Safety Vest    |
| 8  | Machinery      |
| 9  | Vehicle        |

---

## Tech Stack

* Python
* OpenCV
* Ultralytics YOLOv8
* PyTorch
* NumPy
* Pandas

---

## Project Structure

```text
AegisVision/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── outputs/
│
├── ppe_rules.py
├── preprocess.py
├── test_detections.py
├── train_yolo.py
├── trainwithhp.py
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/koyya-suchitra/AegisVision.git
cd AegisVision
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Update the dataset configuration in:

```python
train_yolo.py
```

and start training:

```bash
python train_yolo.py
```

Trained model weights will be saved inside:

```text
runs/detect/
```

---

## Running Detection

Launch PPE detection using:

```bash
python test_detections.py
```

The system supports:

* Live webcam detection
* Video file analysis
* Image-based detection

---

## Future Enhancements

* Email and SMS alerts
* Dashboard for safety analytics
* Multi-camera monitoring
* Occupancy heatmaps
* PPE compliance reports
* Cloud deployment

---

## Author

**Suchitra Koyya**

B.Tech – Computer Science & Engineering (Data Science)

Passionate about Artificial Intelligence, Computer Vision, and Workplace Safety Automation.
