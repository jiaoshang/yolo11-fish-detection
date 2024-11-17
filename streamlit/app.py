import streamlit as st

st.title("Fish Detection with YOLO V11")


pages = {
    "Fish detector": [
        st.Page("image_detector.py", title="Fish detector on image", icon="📷"),
        st.Page("video_detector.py", title="Fish detector on video", icon="🎥"),
    ]
}

pg = st.navigation(pages)
pg.run()