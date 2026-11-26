"""Briques réutilisables de l'eye tracker (reprises de test_xav.py).

Usage typique (tracker dans un thread de fond, l'interface lit tracker.gaze quand elle veut) :

    with GazeTracker().start() as tracker:
        while tracker.alive:
            x, y = tracker.gaze          # position normalisée [0, 1] dans la fenêtre
            ...

Le modèle de regard (calibration 16 points + ridge, compensation de la tête, One Euro,
gel pendant les clignements) est identique à test_xav.py.
"""
import math
import os
import queue
import random
import threading
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "face_landmarker.task")

# Points de repère par œil (indices du modèle 478 points). Les coins sont triés par x dans le code,
# donc peu importe quel œil du sujet c'est : A = œil à gauche de l'image affichée, B = à droite.
EYE_A = {"outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": 468, "contour": [469, 470, 471, 472]}
EYE_B = {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473, "contour": [474, 475, 476, 477]}
NOSE_TIP = 1

# --- Réglages ---------------------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30   # 640x480 : bien moins coûteux sur Raspberry Pi

CAL_COLS, CAL_ROWS = 4, 4        # grille de calibration (16 points, ~25 s)
CAL_MARGIN = 0.08                # marge des cibles par rapport au bord de la fenêtre
CAL_SETTLE_S = 0.7               # temps laissé à l'œil pour atteindre la cible (échantillons jetés)
CAL_COLLECT_S = 1.1              # durée de collecte par cible
CAL_MIN_SAMPLES = 8              # échantillons minimum par cible pour la valider
CAL_MIN_POINTS = 8               # cibles validées minimum pour accepter la calibration
RIDGE_LAMBDA = 1.0               # régularisation (sur features standardisées)

ONE_EURO_MIN_CUTOFF = 0.6        # Hz : plus bas = plus stable en fixation, plus de lag
ONE_EURO_BETA = 3.0              # plus haut = moins de lag pendant les mouvements rapides
ONE_EURO_D_CUTOFF = 1.0

BLINK_THRESHOLD = 0.6            # score moyen eyeBlinkLeft/Right au-dessus duquel on gèle
BLINK_HOLD_S = 0.15              # gel supplémentaire après le clignement (paupière qui remonte)

FALLBACK_GAIN_X = 3.0            # mapping approximatif utilisé tant qu'on n'est pas calibré
FALLBACK_GAIN_Y = 6.0

FEATURE_NAMES = ["fx", "fy", "bv", "ap", "yaw", "pitch", "hx", "hy", "hs"]
# Plancher d'écart-type par colonne du design (9 features + fx², fy², fx*fy) : évite d'amplifier
# du bruit quand une feature (ex : la tête) a très peu varié pendant la calibration.
SD_FLOORS = np.array([0.01, 0.005, 0.05, 0.02, 2.0, 2.0, 0.01, 0.01, 0.005, 0.002, 0.0005, 0.0005])


# --- Utilitaires de calcul --------------------------------------------------------------------
class OneEuroFilter:
    """Filtre One Euro (Casiez et al., 2012) pour un signal scalaire."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t_prev = None
        self.x_prev = 0.0
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, 0.0
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


def eye_features(lm, eye, w, h):
    """Position de l'iris dans le repère de l'œil, en unités de largeur d'œil.

    Retourne (fx, fy, ap) : fx > 0 = iris vers la droite de l'image, fy > 0 = iris vers le bas,
    ap = ouverture verticale de l'œil / largeur.
    """
    def p(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    a, b = p(eye["outer"]), p(eye["inner"])
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    width = float(np.linalg.norm(right - left))
    if width < 1e-6:
        return None
    u = (right - left) / width
    v = np.array([-u[1], u[0]])  # perpendiculaire à u, orientée vers le bas de l'image
    center = (left + right) / 2.0
    d = p(eye["iris"]) - center
    ap = float(np.linalg.norm(p(eye["top"]) - p(eye["bottom"])) / width)
    return float(d @ u) / width, float(d @ v) / width, ap


def head_pose(result):
    """(yaw, pitch, roll) en degrés depuis la matrice de transformation du visage."""
    mats = result.facial_transformation_matrixes
    if mats is None or len(mats) == 0:
        return 0.0, 0.0, 0.0
    rot = np.array(mats[0], dtype=float)[:3, :3]
    norms = np.linalg.norm(rot, axis=0)
    if np.any(norms < 1e-9):
        return 0.0, 0.0, 0.0
    rot = rot / norms
    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(-rot[2, 0], math.hypot(rot[2, 1], rot[2, 2])))
    roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return yaw, pitch, roll


def extract_features(result, w, h):
    """Vecteur de features (voir FEATURE_NAMES) + score de clignement. None si inexploitable."""
    lm = result.face_landmarks[0]
    ea, eb = eye_features(lm, EYE_A, w, h), eye_features(lm, EYE_B, w, h)
    if ea is None or eb is None:
        return None
    fx, fy, ap = (ea[0] + eb[0]) / 2, (ea[1] + eb[1]) / 2, (ea[2] + eb[2]) / 2

    bs = {}
    if result.face_blendshapes:
        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
    if bs:
        blink = (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2
    else:
        blink = 1.0 if ap < 0.08 else 0.0
    down = (bs.get("eyeLookDownLeft", 0.0) + bs.get("eyeLookDownRight", 0.0)) / 2
    up = (bs.get("eyeLookUpLeft", 0.0) + bs.get("eyeLookUpRight", 0.0)) / 2
    bv = down - up

    yaw, pitch, _ = head_pose(result)
    hx, hy = lm[NOSE_TIP].x, lm[NOSE_TIP].y
    hs = math.hypot((lm[33].x - lm[263].x) * w, (lm[33].y - lm[263].y) * h) / w  # ~ 1 / distance caméra
    return np.array([fx, fy, bv, ap, yaw, pitch, hx, hy, hs]), blink


def design_matrix(feats):
    fx, fy = feats[:, 0], feats[:, 1]
    return np.hstack([feats, (fx * fx)[:, None], (fy * fy)[:, None], (fx * fy)[:, None]])


class GazeModel:
    """Régression ridge : features -> point (x, y) normalisé dans la fenêtre."""

    def __init__(self, lam):
        self.lam = lam

    def fit(self, feats, targets):
        x = design_matrix(feats)
        self.mu = x.mean(axis=0)
        self.sd = np.maximum(x.std(axis=0), SD_FLOORS)
        xs = (x - self.mu) / self.sd
        self.ym = targets.mean(axis=0)
        a = xs.T @ xs + self.lam * np.eye(xs.shape[1])
        self.w = np.linalg.solve(a, xs.T @ (targets - self.ym))
        return self

    def predict(self, feats):
        x = design_matrix(np.atleast_2d(feats))
        return ((x - self.mu) / self.sd) @ self.w + self.ym


def lopo_error(feats, targets, groups, lam):
    """Erreur absolue moyenne (x, y) en laissant à chaque fois une cible entière de côté."""
    errs = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if tr.sum() < 20:
            continue
        model = GazeModel(lam).fit(feats[tr], targets[tr])
        errs.append(np.abs(model.predict(feats[te]) - targets[te]))
    return np.vstack(errs).mean(axis=0) if errs else np.full(2, np.nan)


def reject_outliers(samples):
    """Retire les échantillons aberrants (clignement partiel, saccade) d'une cible, sur fx/fy."""
    med = np.median(samples[:, :2], axis=0)
    mad = np.maximum(np.median(np.abs(samples[:, :2] - med), axis=0) * 1.4826, 0.003)
    keep = np.all(np.abs(samples[:, :2] - med) <= 3.5 * mad, axis=1)
    return samples[keep]


def fallback_gaze(feats):
    """Mapping approximatif avant calibration (le curseur bouge, mais sans précision)."""
    return np.array([0.5 + FALLBACK_GAIN_X * feats[0], 0.5 + FALLBACK_GAIN_Y * feats[1]])


class Calibration:
    """Machine à états : affiche une cible à la fois, collecte les features, rejette les aberrants."""

    def __init__(self):
        self.targets = [
            (CAL_MARGIN + (1 - 2 * CAL_MARGIN) * c / (CAL_COLS - 1),
             CAL_MARGIN + (1 - 2 * CAL_MARGIN) * r / (CAL_ROWS - 1))
            for r in range(CAL_ROWS) for c in range(CAL_COLS)
        ]
        random.shuffle(self.targets)  # ordre aléatoire : évite que l'œil anticipe la cible suivante
        self.idx = 0
        self.t0 = None
        self.retries = 0
        self.skipped = 0
        self.cur = []
        self.feats, self.targs, self.groups = [], [], []

    @property
    def done(self):
        return self.idx >= len(self.targets)

    @property
    def target(self):
        return self.targets[min(self.idx, len(self.targets) - 1)]

    def phase_time(self, now):
        return 0.0 if self.t0 is None else now - self.t0

    def update(self, now, feats):
        """feats = vecteur valide (visage, pas de clignement) ou None."""
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        if CAL_SETTLE_S <= t < CAL_SETTLE_S + CAL_COLLECT_S:
            if feats is not None:
                self.cur.append(feats)
        elif t >= CAL_SETTLE_S + CAL_COLLECT_S:
            self._finish_point(now)

    def _finish_point(self, now):
        cur = np.array(self.cur) if self.cur else np.empty((0, len(FEATURE_NAMES)))
        self.cur = []
        if len(cur) >= CAL_MIN_SAMPLES:
            cur = reject_outliers(cur)
        if len(cur) >= CAL_MIN_SAMPLES:
            tx, ty = self.target
            self.feats.append(cur)
            self.targs.append(np.tile([tx, ty], (len(cur), 1)))
            self.groups.append(np.full(len(cur), self.idx))
            self.idx += 1
            self.retries = 0
        elif self.retries < 2:
            self.retries += 1  # on refait la même cible (visage perdu, clignements...)
        else:
            self.skipped += 1
            self.idx += 1
            self.retries = 0
        self.t0 = now

    def fit(self):
        """Retourne (modèle, erreur train x/y, erreur LOPO x/y) ou None si trop peu de cibles."""
        if len(self.feats) < CAL_MIN_POINTS:
            return None
        feats, targs, groups = np.vstack(self.feats), np.vstack(self.targs), np.concatenate(self.groups)
        model = GazeModel(RIDGE_LAMBDA).fit(feats, targs)
        train = np.abs(model.predict(feats) - targs).mean(axis=0)
        cv = lopo_error(feats, targs, groups, RIDGE_LAMBDA)
        return model, train, cv


# --- Caméra / modèle --------------------------------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Téléchargement du modèle vers {MODEL_PATH} ...")
    tmp = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)
    print("Terminé.")


def draw_camera_overlay(image, lm, w, h, lines):
    for eye in (EYE_A, EYE_B):
        for i in eye["contour"]:
            cv2.circle(image, (int(lm[i].x * w), int(lm[i].y * h)), 2, (0, 255, 0), -1)
        cv2.circle(image, (int(lm[eye["iris"]].x * w), int(lm[eye["iris"]].y * h)), 3, (0, 0, 255), -1)
        for k in ("outer", "inner"):
            cv2.circle(image, (int(lm[eye[k]].x * w), int(lm[eye[k]].y * h)), 2, (255, 128, 0), -1)
    for n, text in enumerate(lines):
        cv2.putText(image, text, (10, 25 + 22 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


class GazeTracker:
    """Webcam + MediaPipe + modèle de regard.

    Deux façons de l'utiliser :
      * synchrone : appeler step() à chaque tour de boucle ;
      * en tâche de fond (recommandé, ex. Raspberry Pi) : start() lance un thread qui appelle step()
        en continu ; l'interface lit simplement les attributs et peut tourner à sa propre cadence.

    Attributs utiles :
      gaze     : np.array([x, y]) filtré, normalisé [0, 1] (peut sortir légèrement de l'intervalle)
      frame    : image caméra (BGR), annotée si annotate=True
      frame_id : incrémenté à chaque nouvelle image
      usable   : True si la dernière image a donné une mesure (visage vu, pas de clignement)
      status   : texte d'état (calibré ou non, erreur de validation...)
      cal      : Calibration en cours ou None
      alive    : False si la caméra a cessé de répondre

    Les commandes (start_calibration, recenter, reset...) sont mises en file et exécutées par le
    thread du tracker au début du step() suivant : on peut les appeler depuis n'importe quel thread.
    """

    def __init__(self, cam_index=0, resolution=(CAM_WIDTH, CAM_HEIGHT), blink_threshold=BLINK_THRESHOLD,
                 min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA, annotate=False):
        ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,  # le lissage temporel interne de MediaPipe n'est actif qu'avec 1 seul visage
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise SystemExit(f"Impossible d'ouvrir la webcam (index {cam_index}).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.landmarker = FaceLandmarker.create_from_options(options)

        self.blink_threshold = blink_threshold
        self.annotate = annotate
        self.filters = [OneEuroFilter(min_cutoff, beta, ONE_EURO_D_CUTOFF) for _ in range(2)]
        self.raw_hist = deque(maxlen=20)
        self.offset = np.zeros(2)
        self.gaze = np.array([0.5, 0.5])
        self.model, self.cal = None, None
        self.status = "NON CALIBRE - appuie sur C"
        self.blink_until, self.last_ts, self.fps, self.last_frame = 0.0, -1, 0.0, None
        self.last_valid = None   # instant de la dernière mesure de regard exploitable
        self.frame, self.frame_id, self.now, self.usable = None, 0, time.monotonic(), False
        self.alive = True
        self.start_time = time.monotonic()
        self._cmds = queue.Queue()
        self._thread, self._stop = None, threading.Event()

    # --- cycle de vie ---
    def start(self):
        """Lance la boucle de suivi dans un thread de fond."""
        self._thread = threading.Thread(target=self._run, name="gaze-tracker", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            if not self.step():
                self.alive = False
                break

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.landmarker.close()
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- état ---
    @property
    def calibrating(self):
        return self.cal is not None

    @property
    def calibrated(self):
        return self.model is not None

    def gaze_fresh(self, max_age):
        """True si la dernière mesure de regard date de moins de max_age secondes."""
        last = self.last_valid
        return last is not None and time.monotonic() - last <= max_age

    # --- boucle ---
    def step(self):
        """Lit une image et met à jour le regard. Retourne False si la caméra ne répond plus."""
        while not self._cmds.empty():
            self._cmds.get_nowait()()

        success, image = self.cap.read()
        if not success:
            return False

        image = cv2.flip(image, 1)  # miroir ; les coins sont triés par x donc rien à inverser
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now = time.monotonic()
        ts = int((now - self.start_time) * 1000)
        if ts <= self.last_ts:  # MediaPipe exige des timestamps strictement croissants
            ts = self.last_ts + 1
        self.last_ts = ts
        if self.last_frame is not None:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_frame, 1e-3)
        self.last_frame = now

        result = self.landmarker.detect_for_video(mp_image, ts)

        feats, blink = None, 0.0
        if result.face_landmarks:
            out = extract_features(result, w, h)
            if out is not None:
                feats, blink = out
                if self.annotate:
                    lines = [f"FPS {self.fps:.0f}", f"blink {blink:.2f} (seuil {self.blink_threshold})",
                             f"yaw {feats[4]:+.0f} pitch {feats[5]:+.0f}",
                             f"fx {feats[0]:+.3f} fy {feats[1]:+.3f} bv {feats[2]:+.2f}"]
                    draw_camera_overlay(image, result.face_landmarks[0], w, h, lines)
        if feats is None and self.annotate:
            cv2.putText(image, "Pas de visage", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        is_blink = blink > self.blink_threshold
        if is_blink:
            self.blink_until = now + BLINK_HOLD_S
        usable = feats is not None and not is_blink and now >= self.blink_until

        if self.cal is not None:
            self.cal.update(now, feats if usable else None)
            if self.cal.done:
                self._finish_calibration()
        elif usable:
            raw = self.model.predict(feats)[0] if self.model is not None else fallback_gaze(feats)
            self.raw_hist.append(raw)
            t = now - self.start_time
            self.gaze = np.array([self.filters[0](raw[0] + self.offset[0], t),
                                  self.filters[1](raw[1] + self.offset[1], t)])
            self.last_valid = now

        self.frame, self.now, self.usable = image, now, usable
        self.frame_id += 1
        return True

    def _finish_calibration(self):
        fitted = self.cal.fit()
        if fitted is None:
            self.status = "Calibration echouee (trop peu de points) - recommence avec C"
            print(self.status)
        else:
            self.model, train, cv = fitted
            self.offset = np.zeros(2)
            self.raw_hist.clear()
            self.status = (f"CALIBRE - erreur validee: x {cv[0] * 100:.1f}% y {cv[1] * 100:.1f}% "
                           f"(train x {train[0] * 100:.1f}% y {train[1] * 100:.1f}%)")
            print(self.status + f" | cibles ignorees: {self.cal.skipped}")
        self._reset_filters()
        self.last_valid = None
        self.cal = None

    def _reset_filters(self):
        for f in self.filters:
            f.reset()

    # --- commandes (thread-safe : exécutées par le thread du tracker) ---
    def start_calibration(self):
        def cmd():
            self.cal = Calibration()
            print("Calibration : garde la tete naturelle et suis chaque point des yeux.")
        self._cmds.put(cmd)

    def cancel_calibration(self):
        def cmd():
            self.cal = None
            self.status = "Calibration annulee"
        self._cmds.put(cmd)

    def recenter(self):
        """Décale le regard pour que la position actuelle corresponde au centre de la fenêtre."""
        def cmd():
            if self.cal is None and self.raw_hist:
                self.offset = 0.5 - np.mean(self.raw_hist, axis=0)
                self._reset_filters()
                print("Recentre sur le milieu de la fenetre.")
        self._cmds.put(cmd)

    def reset(self):
        def cmd():
            self.model, self.cal, self.offset = None, None, np.zeros(2)
            self.raw_hist.clear()
            self.status = "NON CALIBRE - appuie sur C"
            self._reset_filters()
        self._cmds.put(cmd)

    def calibration_state(self):
        """(cible (x, y) normalisée, progression 0..1, en collecte ?, index, total) ou None."""
        cal = self.cal
        if cal is None:
            return None
        t = cal.phase_time(time.monotonic())
        frac = min(t / (CAL_SETTLE_S + CAL_COLLECT_S), 1.0)
        return cal.target, frac, t >= CAL_SETTLE_S, min(cal.idx + 1, len(cal.targets)), len(cal.targets)
