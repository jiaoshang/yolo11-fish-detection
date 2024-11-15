import cv2
from ultralytics import YOLO
from ultralytics import solutions

def detect_objects(image_source: str, output_image_filename: str):
    # Load a pretrained YOLO11n model
    model = YOLO("./runs/detect/train/weights/last.pt")

    # Run inference on the source
    results = model(image_source)

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename=output_image_filename)

def count_objects(image_source: str, output_image_filename: str):
    region_points = [(0, 0), (1280, 0), (1280, 720), (0, 720)] # 1280 * 720 jpg
    counter = solutions.ObjectCounter(
        show=True,  # Display the output
        region=region_points,  # Pass region points
        model="./runs/detect/train/weights/last.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
        # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
        # show_in=True,  # Display in counts
        # show_out=True,  # Display out counts
        # line_width=2,  # Adjust the line width for bounding boxes and text display
    )
    frame = cv2.imread(image_source)
    processed_frame = counter.count(frame)
    cv2.imwrite(output_image_filename, processed_frame)


if __name__ == '__main__':
    train_set_example_source = '../data/demo/train_set_example.jpg'
    validation_set_example_source = '../data/demo/validation_set_example.jpg'
    internet_example_source = '../data/demo/internet_example.jpg'
    example_source = '../data/demo/example.jpg'
    # detect_objects(example_source, 'result.jpg')
    count_objects(example_source, 'count_result.jpg')