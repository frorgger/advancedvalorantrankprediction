from inference import InferencePipeline, get_model
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
from ultralytics import YOLO, solutions

API_KEY_FORG = "aR3uuLsVptSKviCUppil"
frame_count = 0

def te():
    # 1. Load your model (local weights)

     model = YOLO("runs/detect/valorant_reaction_v217/weights/best.pt")
     def tracking_test_sink(predictions: dict, video_frame: VideoFrame):
         # A. Get the frame and detections
         detections = predictions["predictions"]
         frame = video_frame.image.copy()

         # B. Visualize every enemy detection with a box
         # If the model is 'tracking' well, these boxes will be steady on the enemy
         for d in detections:
             if d['class'] == 'enemy':
                 x, y, w, h = int(d['x']), int(d['y']), int(d['width']), int(d['height'])
                 # Draw a bright green box for enemies
                 cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                 cv2.putText(frame, "ENEMY", (x - w // 2, y - h // 2 - 5),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

         # C. Display the results
         cv2.imshow("Enemy Tracking Test", frame)
         if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()
    #
     # 2. Point this to your specific Valorant clip (.mp4, .mov, etc.)
     pipeline = InferencePipeline.init(
         model_id="valorant-9ufcp-fdhan/1",
         video_reference="clip_001.mp4",  # <-- SWAP THIS
         on_prediction=tracking_test_sink,
         api_key=API_KEY_FORG,
         confidence=.20,
     )

     print("Starting Tracking Test... Press 'q' to stop.")
     pipeline.start()
     pipeline.join()

def streamlit():
    solutions.inference(model="runs/detect/valorant_reaction_v220/weights/best.pt")

if __name__ == "__main__":
    # model1 = YOLO("valorant-9ufcp-fdhan")
    # run_pipeline()
    # te()
    streamlit()