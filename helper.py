# your helper.py code here
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from ultralytics import YOLO
import supervision as sv

def load_model(model_path):
    return YOLO(model_path)

def play_stored_video(conf, model, placeholders, show_heatmap=True, show_speed=True, show_annotations=True, show_graphs=True):
    source_vid = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if source_vid:
        st.video(source_vid)
        if st.sidebar.button("Detect Video Objects"):
            try:
                # Setup objects
                LINE_START = sv.Point(360, 0)
                LINE_END = sv.Point(360, 720)
                tracker = sv.ByteTrack()
                line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

                # Optional annotators
                box_annotator = sv.BoxAnnotator(thickness=2)
                label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
                line_annotator = sv.LineZoneAnnotator(thickness=2)

                # Save video to temp file
                with open("temp_video.mp4", "wb") as f:
                    f.write(source_vid.getbuffer())
                cap = cv2.VideoCapture("temp_video.mp4")

                st_frame = st.empty()
                chart_placeholder = st.empty() if show_graphs else None

                speeds = {}
                last_positions = {}
                speed_data = []

                # Heatmap
                ret, frame_sample = cap.read()
                h, w = frame_sample.shape[:2]
                heatmap = np.zeros((h, w), dtype=np.float32) if show_heatmap else None
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_id = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_id += 1

                    frame = cv2.resize(frame, (720, int(720 * (frame.shape[0] / frame.shape[1]))))
                    results = model.predict(frame, conf=conf)
                    detections = sv.Detections.from_ultralytics(results[0])
                    tracked = tracker.update_with_detections(detections)
                    line_zone.trigger(detections=tracked)

                    labels = []
                    colors = []

                    for i, box in enumerate(tracked.xyxy):
                        track_id = tracked.tracker_id[i]
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        pos = np.array([x_center, y_center])

                        # Speed estimation
                        if show_speed and track_id in last_positions:
                            dist = np.linalg.norm(pos - last_positions[track_id])
                            px_per_sec = dist * fps
                            px_to_m = 1 / 20.0  # Adjust this factor per scale
                            speed_kmh = (px_per_sec * px_to_m) * 3.6
                            speeds[track_id] = speed_kmh
                            speed_data.append({'ID': int(track_id), 'Speed': speed_kmh, 'Time': frame_id / fps})

                        last_positions[track_id] = pos

                        # Heatmap update
                        if show_heatmap:
                            heatmap[int(y_center), int(x_center)] += 1

                        # Label prep
                        if show_annotations:
                            speed = speeds.get(track_id, 0) if show_speed else 0
                            labels.append(f"ID {track_id} | {speed:.1f} km/h" if show_speed else f"ID {track_id}")
                            colors.append((255, 0, 0) if speed > 40 else (0, 255, 0))

                    annotated = frame.copy()
                    if show_annotations:
                        for i, box in enumerate(tracked.xyxy):
                            xyxy = box.astype(int)
                            cv2.rectangle(annotated, tuple(xyxy[:2]), tuple(xyxy[2:]), colors[i], 2)
                            cv2.putText(annotated, labels[i], (xyxy[0], xyxy[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

                    # Line count annotation
                    annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)

                    # Apply heatmap if enabled
                    if show_heatmap:
                        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                        heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
                        heatmap_colored = cv2.resize(heatmap_colored, (annotated.shape[1], annotated.shape[0]))
                        annotated = cv2.addWeighted(annotated, 0.7, heatmap_colored, 0.3, 0)

                    # Update UI
                    placeholders['in'].metric("Cars Entered", value=line_zone.in_count)
                    placeholders['out'].metric("Cars Exited", value=line_zone.out_count)
                    st_frame.image(annotated, channels="BGR", use_container_width=True)

                cap.release()

                if show_graphs and len(speed_data) > 10:
                    df = pd.DataFrame(speed_data)
                    chart_placeholder.line_chart(
                        df.pivot(index="Time", columns="ID", values="Speed").fillna(method="ffill")
                    )

            except Exception as e:
                st.sidebar.error(f"Error during processing: {str(e)}")
