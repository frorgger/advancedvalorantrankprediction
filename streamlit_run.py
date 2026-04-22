import streamlit as st
import cv2
import time
import tempfile
import math
from ultralytics import YOLO


# Page setup
st.set_page_config(page_title="Valorant Rank Predictor", layout="wide")
st.title("Advanced Valorant Rank Prediction")


# Session state
if "running" not in st.session_state:
    st.session_state.running = False

if "tracked_objects" not in st.session_state:
    st.session_state.tracked_objects = {}

if "reaction_times" not in st.session_state:
    st.session_state.reaction_times = []


# Sidebar controls
st.sidebar.header("Controls")

source_type = st.sidebar.radio("Input Source", ["Webcam", "Upload Video"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
disappear_buffer = st.sidebar.slider("Disappear Buffer (sec)", 0.1, 2.0, 0.5, 0.1)
center_radius = 20

start = st.sidebar.button("Start")
stop = st.sidebar.button("Stop / Reset")

if start:
    st.session_state.running = True
    st.session_state.tracked_objects = {}
    st.session_state.reaction_times = []

if stop:
    st.session_state.running = False
    st.session_state.tracked_objects = {}
    st.session_state.reaction_times = []


# Load Model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/valorant_reaction_v220/weights/best.pt")


model = load_model()

# Input source setup
cap = None
video_ready = False

if source_type == "Webcam":
    camera_index = st.sidebar.number_input(
        "Camera Index",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )
    cap = cv2.VideoCapture(camera_index)
    video_ready = cap.isOpened()

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Video",
        type=["mp4", "mov", "avi", "mkv"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        video_ready = cap.isOpened()


# Layout
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.subheader("Running Average")
    avg_placeholder = st.empty()

with col3:
    st.subheader("Predicted Rank")
    rank_placeholder = st.empty()


# Rank Estimation (reaction time averaging)
def predict_rank(avg_reaction_time):
    if avg_reaction_time <= 0.180:
        return "Immortal - Radiant"
    elif avg_reaction_time <= 0.200:
        return "Diamond - Ascendant"
    elif avg_reaction_time <= 0.220:
        return "Gold - Platinum"
    else:
        return "Iron - Silver"


# Main processing loop
if st.session_state.running:
    if not video_ready:
        st.error("No valid video source available. Choose webcam or upload a video.")
    else:
        prev_time = time.time()

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.session_state.running = False
                break

            current_time = time.time()
            frame_h, frame_w = frame.shape[:2]
            screen_center = (frame_w // 2, frame_h // 2)

            # Run tracking
            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                verbose=False
            )

            active_ids = set()

            annotated_frame = frame.copy()

            cv2.circle(annotated_frame, screen_center, center_radius, (255, 0, 0), 2)
            cv2.circle(annotated_frame, screen_center, 4, (255, 0, 0), -1)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes

                for box, track_id in zip(boxes.xyxy, boxes.id):
                    track_id = int(track_id.item())
                    active_ids.add(track_id)

                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    distance_to_center = math.sqrt(
                        (cx - screen_center[0]) ** 2 +
                        (cy - screen_center[1]) ** 2
                    )

                    if track_id not in st.session_state.tracked_objects:
                        st.session_state.tracked_objects[track_id] = {
                            "first_seen": current_time,
                            "last_seen": current_time,
                            "centered": False,
                        }
                    else:
                        st.session_state.tracked_objects[track_id]["last_seen"] = current_time

                    obj = st.session_state.tracked_objects[track_id]
                    visible_duration = current_time - obj["first_seen"]

                    # Draw bounding box
                    box_color = (0, 255, 0) if not obj["centered"] else (0, 0, 255)
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"ID {track_id} | {visible_duration:.2f}s",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

                    cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)

                    # Log reaction when enemy first enters center zone
                    if (not obj["centered"]) and (distance_to_center <= center_radius):
                        obj["centered"] = True
                        reaction_time = current_time - obj["first_seen"]

                        if reaction_time >= 0.05:
                            st.session_state.reaction_times.append(reaction_time)

            lost_ids = []
            for track_id, obj in st.session_state.tracked_objects.items():
                if track_id not in active_ids:
                    time_since_seen = current_time - obj["last_seen"]
                    if time_since_seen > disappear_buffer:
                        lost_ids.append(track_id)

            for track_id in lost_ids:
                del st.session_state.tracked_objects[track_id]

            frame_placeholder.image(annotated_frame, channels="BGR")

            # Running average
            if st.session_state.reaction_times:
                running_avg = (
                        sum(st.session_state.reaction_times)
                        / len(st.session_state.reaction_times)
                )
                avg_placeholder.metric("Average Reaction Time", f"{running_avg:.3f}s")
                rank_placeholder.metric("Estimated Rank", predict_rank(running_avg))
            else:
                avg_placeholder.metric("Average Reaction Time", "N/A")
                rank_placeholder.metric("Estimated Rank", "N/A")

        cap.release()


# Show average outside loop too
if st.session_state.reaction_times:
    running_avg = (
            sum(st.session_state.reaction_times)
            / len(st.session_state.reaction_times)
    )
    st.metric("Current Running Average", f"{running_avg:.3f}s")
else:
    st.write("Run a session to calculate reaction time.")
