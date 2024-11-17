from ultralytics import YOLO, solutions

def load_model(model_path: str):
    return YOLO(model_path)

def load_model_as_object_counter(model_path: str):
    region_points = [(0, 0), (1280, 0), (1280, 720), (0, 720)]  # 1280 * 720 jpg
    return solutions.ObjectCounter(
        show=True,  # Display the output
        region=region_points,  # Pass region points
        model=model_path,  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
        classes=[0],  # If you want to count specific classes i.e person and car with COCO pretrained model.
        show_in=False,  # Display in counts
        show_out=False,  # Display out counts
        # line_width=2,  # Adjust the line width for bounding boxes and text display
    )