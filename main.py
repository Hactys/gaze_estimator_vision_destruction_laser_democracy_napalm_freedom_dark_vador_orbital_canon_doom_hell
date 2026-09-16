"""
Calibration-free, head-fixed eye-gaze selection on an on-screen 3x3 grid.

Approach: MediaPipe's FaceLandmarker gives 478 face landmarks per frame,
including the iris centers. For each eye we measure where the iris sits
inside its own eye-socket box (corner-to-corner, lid-to-lid), expressed
as a ratio in [0, 1]. Because the disabled user is assumed not to move 
their head, that eye-socket-relative position is a direct, per-frame proxy
for gaze direction -- there is no per-user calibration step. The two eyes' 
ratios are averaged for robustness, smoothed over time, and thresholded 
into a 3x3 zone. Staying on the same zone for DWELL_SECONDS selects it.
"""

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

# TODO : use outerer ones to achieve more precise tracking
LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": 468}
RIGHT_EYE = {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473}

# Thresholds splitting the smoothed [0, 1] gaze ratio into 3 zones.
H_SPLITS = (0.42, 0.58)
V_SPLITS = (0.42, 0.58)

SMOOTHING = 0.35          # exponential-moving-average weight for each new sample
DWELL_SECONDS = 0.8       # how long gaze must stay on a cell to select it
FLASH_SECONDS = 0.4       # how long a selected cell stays highlighted

GRID_SIZE = 480
CELL = GRID_SIZE // 3


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading face landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def eye_ratio(landmarks, eye):
    """Where the iris sits inside this eye's own socket box, in [0, 1]x[0, 1]."""
    ox, oy = landmarks[eye["outer"]].x, landmarks[eye["outer"]].y
    ix, iy = landmarks[eye["inner"]].x, landmarks[eye["inner"]].y
    tx, ty = landmarks[eye["top"]].x, landmarks[eye["top"]].y
    bx, by = landmarks[eye["bottom"]].x, landmarks[eye["bottom"]].y
    px, py = landmarks[eye["iris"]].x, landmarks[eye["iris"]].y

    x_lo, x_hi = min(ox, ix), max(ox, ix)
    y_lo, y_hi = min(ty, by), max(ty, by)

    rx = (px - x_lo) / (x_hi - x_lo) if x_hi > x_lo else 0.5
    ry = (py - y_lo) / (y_hi - y_lo) if y_hi > y_lo else 0.5
    return rx, ry


def cell_for_ratio(rx, ry):
    col = 0 if rx < H_SPLITS[0] else (1 if rx < H_SPLITS[1] else 2)
    row = 0 if ry < V_SPLITS[0] else (1 if ry < V_SPLITS[1] else 2)
    return row, col


def draw_grid(current_cell, selected_cell, selected_until):
    canvas = np.full((GRID_SIZE, GRID_SIZE, 3), 30, dtype=np.uint8)
    now = time.time()
    for r in range(3):
        for c in range(3):
            x0, y0 = c * CELL, r * CELL
            x1, y1 = x0 + CELL, y0 + CELL
            if selected_cell == (r, c) and now < selected_until:
                cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4), (0, 200, 0), -1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 3)
            elif current_cell == (r, c):
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 255), 4)
            else:
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 60, 60), 2)
    return canvas


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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")

    smoothed_rx, smoothed_ry = 0.5, 0.5
    current_cell = None
    dwell_start = None
    dwell_selected = False
    selected_cell = None
    selected_until = 0.0

    start = time.time()
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            now = time.time()

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                lrx, lry = eye_ratio(lm, LEFT_EYE)
                rrx, rry = eye_ratio(lm, RIGHT_EYE)
                rx, ry = (lrx + rrx) / 2, (lry + rry) / 2

                smoothed_rx += SMOOTHING * (rx - smoothed_rx)
                smoothed_ry += SMOOTHING * (ry - smoothed_ry)

                cell = cell_for_ratio(smoothed_rx, smoothed_ry)
                if cell != current_cell:
                    current_cell = cell
                    dwell_start = now
                    dwell_selected = False
                elif not dwell_selected and now - dwell_start >= DWELL_SECONDS:
                    selected_cell = cell
                    selected_until = now + FLASH_SECONDS
                    dwell_selected = True
                    print(f"Selected cell {selected_cell} (index {selected_cell[0] * 3 + selected_cell[1]})")

                for eye in (LEFT_EYE, RIGHT_EYE):
                    for key in ("outer", "inner", "top", "bottom", "iris"):
                        p = lm[eye[key]]
                        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 0), -1)

                cv2.putText(frame, f"gaze ratio: ({smoothed_rx:.2f}, {smoothed_ry:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                current_cell = None
                dwell_start = None
                dwell_selected = False
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Camera", frame)
            cv2.imshow("Gaze grid", draw_grid(current_cell, selected_cell, selected_until))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
