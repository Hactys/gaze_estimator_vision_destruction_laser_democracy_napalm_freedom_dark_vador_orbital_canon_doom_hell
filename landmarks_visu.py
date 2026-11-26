import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading face landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def draw_landmarks(image, face_landmarks):
    """Draw only the eyes, irises, eyelids and eyebrows."""
    height, width = image.shape[:2]

    visible_landmarks = {
        33, 133, 160, 158, 153, 144,
        362, 263, 387, 385, 373, 380,
        70, 63, 105, 66, 107,
        336, 296, 334, 293, 300,
    }
    iris_landmarks = set(range(468, 478))

    for index, landmark in enumerate(face_landmarks):
        # The last ten landmarks are the five points of each iris.
        if index not in (visible_landmarks | iris_landmarks):
            continue
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        is_iris = index in iris_landmarks
        cv2.circle(
            image,
            (x, y),
            2 if is_iris else 1,
            (255, 0, 255) if is_iris else (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(index),
            (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )



def main():
    ensure_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    timestamp = 0
    with FaceLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp)
                timestamp += 1

                for face in result.face_landmarks:
                    draw_landmarks(frame, face)
                cv2.imshow("MediaPipe - landmarks tête et iris", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

