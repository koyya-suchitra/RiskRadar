from ultralytics import YOLO  

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")  # small model, better for 3050
    
    model.train(
        data="C:/Users/gaikw/OneDrive/Desktop/SafetyEye/processed/safetyeye_v1/data.yaml",
        epochs=20,              # 30k images → 10 is too low, use 50–100
        imgsz=640,              # keep 640 (standard), can try 512 for speed
        batch=16,               # 3050 usually handles 16 at 640px (try 8 if OOM)
        workers=4,              # for faster dataloading
        device=0,               # ensure GPU is used
        name="ppe_yolov8s",
        patience=20,            # early stopping if no improvement
        optimizer="SGD",        # default, but you can try "AdamW" for faster convergence
        val=True
    )
