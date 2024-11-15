from ultralytics import YOLO

def load_model(model_path: str):
    return YOLO(model_path)