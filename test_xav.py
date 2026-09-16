import os
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

# Indices des points de repère pour l'iris gauche et droit (API Tasks, refine_landmarks inclus par défaut)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Coins et centre d'iris pour chaque œil, utilisés pour calculer la position du regard
LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": 468}
RIGHT_EYE = {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473}

SMOOTHING = 0.35          # lissage exponentiel du point de regard
CANVAS_SIZE = (720, 1280)  # hauteur, largeur de la fenêtre "curseur"
CURSOR_RADIUS = 12


def eye_ratio(landmarks, eye):
    """Position de l'iris dans son propre socket, en [0, 1]x[0, 1]."""
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


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Téléchargement du modèle vers {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Terminé.")


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
    start = time.time()

    smoothed_rx, smoothed_ry = 0.5, 0.5

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Inverser l'image pour un effet miroir et convertir en RGB (requis par MediaPipe)
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            timestamp_ms = int((time.time() - start) * 1000)

            # Traitement de l'image
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Si un visage est détecté
            if result.face_landmarks:
                mesh_points = result.face_landmarks[0]
                h, w = image.shape[:2]

                # Récupérer les coordonnées de l'iris gauche
                left_iris_points = [(int(mesh_points[p].x * w), int(mesh_points[p].y * h)) for p in LEFT_IRIS]

                # Dessiner un cercle sur l'iris gauche
                for point in left_iris_points:
                    cv2.circle(image, point, 2, (0, 255, 0), -1)

                # Récupérer les coordonnées de l'iris droit
                right_iris_points = [(int(mesh_points[p].x * w), int(mesh_points[p].y * h)) for p in RIGHT_IRIS]

                # Dessiner un cercle sur l'iris droit
                for point in right_iris_points:
                    cv2.circle(image, point, 2, (0, 255, 0), -1)

                # Calcul de la position de regard moyenne (les deux yeux), lissée dans le temps
                lrx, lry = eye_ratio(mesh_points, LEFT_EYE)
                rrx, rry = eye_ratio(mesh_points, RIGHT_EYE)
                rx, ry = (lrx + rrx) / 2, (lry + rry) / 2

                smoothed_rx += SMOOTHING * (rx - smoothed_rx)
                smoothed_ry += SMOOTHING * (ry - smoothed_ry)

            # Fenêtre "curseur" : un point qui suit le regard sur un canvas dédié
            canvas_h, canvas_w = CANVAS_SIZE
            canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)
            cursor_x = int(smoothed_rx * canvas_w)
            cursor_y = int(smoothed_ry * canvas_h)
            cv2.circle(canvas, (cursor_x, cursor_y), CURSOR_RADIUS, (0, 255, 0), -1)
            cv2.putText(canvas, f"({smoothed_rx:.2f}, {smoothed_ry:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Afficher les résultats
            cv2.imshow('Eye Tracker', image)
            cv2.imshow('Curseur', canvas)

            # Quitter avec la touche 'Echap'
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()