import cv2
import streamlit as st

from model import load_model_as_object_counter, load_model


def _save_uploaded_video(video_bytes) -> str:
    file_path = "streamlit/data/uploaded_video/upload." + video_bytes.name.split('.')[-1]
    with open(file_path, 'wb') as out:
        out.write(video_bytes.read())
    return file_path

def _display_detected_frame(file_path: str):
    capture = cv2.VideoCapture(file_path)  # TODO whether it's possible for capture from cache
    st_frame = st.empty()
    while capture.isOpened():
        success, frame = capture.read()
        # Resize the image to a standard size
        # frame = cv2.resize(frame, (720, int(720 * (9 / 16))))
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        processed_frame = model.predict(frame)
        count = len(processed_frame[0])
        annotated_frame = processed_frame[0].plot()  # Visualize the results on the frame
        cv2.putText(annotated_frame, f"fish: {count}", (0, 1024), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (255, 255, 255), 3)
        st_frame.image(annotated_frame,
                       caption='Detected Video',
                       channels="BGR",
                       use_container_width=True
                       )
    capture.release()

with st.sidebar:
    uploaded_video_bytes = st.file_uploader("Choose an video", type=("mp4", "mov"))

try:
    # model = load_model_as_object_counter("streamlit/model.pt") # TODO: think about the relative path later
    model = load_model("streamlit/model.pt")
except Exception as ex:
    st.error("Can't load YOLO model")
    st.error(ex)

if uploaded_video_bytes is not None:
    video_file_path = _save_uploaded_video(uploaded_video_bytes)
    _display_detected_frame(video_file_path)

# TODO check whether it's possible to pause the detection on video