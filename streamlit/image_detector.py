import streamlit as st
from PIL import Image

from model import load_model

with st.sidebar:
    source_img = st.file_uploader("Choose an image", type=("jpg", "jpeg", "png"))


try:
    model = load_model("streamlit/model.pt") # TODO: think about the relative path later
except Exception as ex:
    st.error("Can't load YOLO model")
    st.error(ex)

# TODO setup default image to deal with NameError: name 'uploaded_image' is not defined
col1, col2 = st.columns(2)

with col1:
    try:
        uploaded_image = Image.open(source_img)
        st.image(source_img, caption="Uploaded Image", use_container_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

    with col2:
        res = model.predict(uploaded_image) #, conf=confidence) # TODO: add more parameters
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image', use_container_width=True)
        st.info(f'Number of fishes: {len(res[0])}', icon="ℹ️")