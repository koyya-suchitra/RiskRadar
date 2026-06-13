import cv2
from ultralytics import YOLO
from ppe_rules import check_violations
from pathlib import Path
import os
import smtplib
from email.mime.text import MIMEText

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).parent
print("BASE_DIR =", BASE_DIR)
MODEL_PATH = BASE_DIR / "yolov8s.pt"
IMAGE_PATH = BASE_DIR / "test_images" / "image4.jpg" #BASE_DIR / r"C:\Users\gaikw\OneDrive\Desktop\SafetyEye\processed\safetyeye_v1\test\images\-4-_png_jpg.rf.3ab1c17e03f82608a2af7b6ff6d57596.jpg"  # Use for image testing
VIDEO_PATH = None  # Set to file path for video, None for webcam
OUTPUT_DIR = BASE_DIR / "outputs"
CONF_THRESH = 0.25  # Confidence threshold

# Alert settings
ENABLE_EMAIL = False  # Set True to send email alerts
ALERT_EMAIL_FROM = "youremail@gmail.com"
ALERT_EMAIL_TO = "receiver@gmail.com"
EMAIL_PASSWORD = "your-app-password"  # Gmail App Password

# -----------------------------
# Setup output folder
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
model = YOLO("weights/best.pt")
print("Model Classes:", model.names)

# -----------------------------
# Email alert function
# -----------------------------
def send_email_alert(violations):
    if not ENABLE_EMAIL:
        return
    msg = MIMEText(f"Safety Violation Detected: {', '.join(violations)}")
    msg["Subject"] = "PPE Violation Alert"
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(ALERT_EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        print("📧 Email alert sent!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# -----------------------------
# Helper function to visualize detections
# -----------------------------
def visualize_results(frame, results):
    if frame is None:
        return frame, []

    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    violations = check_violations(detected_classes)

    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if violations and cls_name in violations:
            color = (0, 0, 255)  # Red for violation
            label = cls_name + " (Violation)"
        else:
            color = (0, 255, 0)  # Green for safe
            label = cls_name

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Overlay alert if violations exist
    if violations:
        cv2.putText(frame, "ALERT: " + ", ".join(violations),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
    return frame, violations

# -----------------------------
# Process Image
# -----------------------------
if IMAGE_PATH and IMAGE_PATH.is_file():
    frame = cv2.imread(str(IMAGE_PATH))
    if frame is not None:
        results = model(str(IMAGE_PATH), conf=CONF_THRESH)
        frame, violations = visualize_results(frame, results)

        print("Detected Classes:", [model.names[int(box.cls)] for box in results[0].boxes])
        print("Violations:", violations)

        if violations:
            send_email_alert(violations)

        output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_ppe{IMAGE_PATH.suffix}"
        cv2.imwrite(str(output_path), frame)
        print(f"✅ Output image saved at: {output_path}")

        cv2.imshow("PPE Detection - Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

# -----------------------------
# Process Video / Webcam
# -----------------------------
cap = None
if VIDEO_PATH and VIDEO_PATH.is_file():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
else:
    print("🎥 Trying to open Webcam...")
    # Try multiple indexes if default fails
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Webcam opened successfully (index {i})")
            break
    else:
        print("❌ Could not open any webcam. Exiting...")
        exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
# type: ignore
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
output_path = OUTPUT_DIR / "output_ppe.mp4"
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ No frame captured, skipping...")
        continue

    # Show raw webcam feed (for debugging)
    cv2.imshow("Raw Webcam Feed", frame)

    results = model(frame, conf=CONF_THRESH)
    frame, violations = visualize_results(frame, results)

    if violations:
        print("⚠️ Violations Detected:", violations)
        send_email_alert(violations)

    out.write(frame)
    cv2.imshow("PPE Detection - Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Output video saved at: {output_path}")
