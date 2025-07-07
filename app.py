# your app.py code here
from pathlib import Path
import PIL
import streamlit as st
import supervision as sv
import settings
import helper

st.set_page_config(
    page_title="Advanced Car Detection App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Advanced Car Detection using YOLOv8 & Supervision")

st.sidebar.header("Model Configuration")
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

try:
    model = helper.load_model(settings.DETECTION_MODEL)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {settings.DETECTION_MODEL}")
    st.error(ex)

# Feature toggles
st.sidebar.header("🛠️ Feature Toggles")
show_heatmap = st.sidebar.checkbox("🔴 Show Heatmap", value=True)
show_speed = st.sidebar.checkbox("⚡ Show Speed Estimation", value=True)
show_annotations = st.sidebar.checkbox("🛆 Show Annotations", value=True)
show_graphs = st.sidebar.checkbox("📈 Show Speed Graphs", value=True)

st.sidebar.header("Image/Video Upload")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

# UI placeholders
col1, col2 = st.columns(2)
in_count_placeholder = col1.empty()
out_count_placeholder = col2.empty()
placeholders = {'in': in_count_placeholder, 'out': out_count_placeholder}

if source_radio == settings.VIDEO:
    st.subheader("Video Analytics")
    helper.play_stored_video(
        conf=confidence,
        model=model,
        placeholders=placeholders,
        show_heatmap=show_heatmap,
        show_speed=show_speed,
        show_annotations=show_annotations,
        show_graphs=show_graphs
    )
elif source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png"))
    col1, col2 = st.columns(2)
    with col1:
        if source_img:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image", use_container_width=True)
    with col2:
        if source_img:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_container_width=True)
                with st.expander("Detection Results"):
                    names = model.names
                    for box in boxes:
                        class_name = names[int(box.cls[0])]
                        confidence_score = box.conf[0]
                        st.write(f"**Object:** {class_name}, **Confidence:** {confidence_score:.2f}")
else:
    st.error("Please select a valid source type!")
