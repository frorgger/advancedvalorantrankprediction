import os
from roboflow import Roboflow
from ultralytics import YOLO

# 1. Initialize Roboflow
# rf = Roboflow(api_key="aR3uuLsVptSKviCUppil")
# project = rf.workspace("forgs-workspace").project("valorant-9ufcp-fdhan")

def start_training():
    # 2. Download the Dataset (Bundles your 10k + new images)
    # Ensure you use the correct version number from your Roboflow dashboard
    # dataset = project.version(1).download("yolov11")

    # 3. Load YOLOv11-Nano (Optimized for speed/reaction analysis)
    model = YOLO("runs/detect/valorant_reaction_v217/weights/best.pt")

    # 4. Start Local Training on your 3070 Ti
    print("Training started on your 3070 Ti...")
    model.train(
        # data=f"{dataset.location}/data.yaml",
        data="P:/clips/manual_training/data.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,      # Uses your 3070 Ti
        workers=4,     # Adjust based on your CPU
        name="valorant_reaction_v2"
    )

# THE FIX: This 'if' statement is mandatory on Windows for multi-GPU/process training
if __name__ == '__main__':
    start_training()


# aR3uuLsVptSKviCUppil