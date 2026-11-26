"""Eye tracker v2 : modèle géométrique 3D tête + yeux, robuste aux mouvements de tête.

Même API que utils.GazeTracker (remplacer `from utils import GazeTracker` par
`from utils_v2 import GazeTracker`), avec en plus :
  * calibration_hint       : consigne à afficher pendant la calibration (ou None) ;
  * head_pose              : (yaw, pitch, roll en degrés, distance en cm) ou None ;
  * save_calibration(p) / load_calibration(p) : garder la calibration entre deux lancements.

Pourquoi la v1 perd la calibration quand la tête bouge
-------------------------------------------------------
La v1 régresse le point regardé sur la position 2D de l'iris dans l'œil + quelques features de
tête. Pendant la calibration la tête ne bouge presque pas : la régression n'apprend pas comment la
pose de la tête doit corriger le point, et le ridge écrase justement ces coefficients.

Ce que fait la v2
-----------------
1. Modèle 3D de la tête propre à l'utilisateur, construit à partir des points 3D MediaPipe
   (x, y, z) de toutes les images de la calibration (Procrustes généralisé sur ~60 points rigides :
   front, nez, pommettes, coins des yeux ; ni bouche ni sourcils). Échelle métrique via la
   distance inter-pupillaire moyenne (6.3 cm).
2. À chaque image : rétroprojection perspective exacte des landmarks (MediaPipe donne u, v et
   une profondeur relative z), puis pose de la tête (rotation R + position en cm) par Procrustes
   pondéré robuste.
3. Chaque globe oculaire est une sphère fixe dans le repère de la tête. Le rayon caméra qui passe
   par le pixel de l'iris est intersecté avec cette sphère -> position 3D de l'iris -> axe optique
   de l'œil, exprimé DANS LE REPÈRE DE LA TÊTE (angles θ, φ œil/tête, indépendants de la pose).
4. Regard = rayon 3D partant du milieu des deux yeux, de direction R · g(θ', φ') où θ', φ' sont
   les angles corrigés (offset kappa, gain, termes quadratiques, paupière). Il est intersecté
   avec le plan de l'écran (placé près de la caméra, inclinaison ajustée) puis passé dans une
   transformation affine plan -> fenêtre.
   Tourner ou déplacer la tête modifie R et l'origine du rayon : la géométrie le compense, il
   n'y a rien à « apprendre ».
5. Calibration : la grille de 16 points (tête immobile) + quelques points où l'on garde les yeux sur
   la cible en bougeant la tête. Ces échantillons fixent les paramètres que la tête immobile ne peut
   pas déterminer (gain réel des angles d'œil, distance). Ajustement par moindres carrés non
   linéaires robustes (Levenberg-Marquardt, perte soft-L1), avec des a priori physiques faibles.
"""
import math
import queue
import random
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from utils import (BLINK_HOLD_S, BLINK_THRESHOLD, CAM_FPS, CAM_HEIGHT, CAM_WIDTH, CAL_COLLECT_S,
                   CAL_COLS, CAL_MARGIN, CAL_MIN_POINTS, CAL_MIN_SAMPLES, CAL_ROWS, CAL_SETTLE_S,
                   EYE_A, EYE_B, MODEL_PATH, ONE_EURO_BETA, ONE_EURO_D_CUTOFF, ONE_EURO_MIN_CUTOFF,
                   OneEuroFilter, draw_camera_overlay, ensure_model, extract_features, fallback_gaze)

# --- Réglages géométriques ----------------------------------------------------------------------
CAM_HFOV_DEG = 60.0     # champ horizontal supposé de la webcam (ajusté en partie par la calibration)
IPD_CM = 6.3            # distance moyenne entre les centres des deux globes oculaires
EYE_RADIUS_CM = 1.15    # distance centre du globe -> centre de l'iris (pupille apparente)
EYE_DEPTH_CM = 1.0      # centre du globe derrière le milieu des coins de l'œil
MAX_HEAD_ANGLE_DEG = 50  # au-delà, la mesure est jugée inexploitable

# Phase « tête en mouvement » : on garde les yeux sur le point et on bouge doucement la tête.
CAL_HEAD_TARGETS = [(0.5, 0.5), (0.25, 0.3), (0.75, 0.7)]
CAL_HEAD_SETTLE_S = 1.0
CAL_HEAD_COLLECT_S = 4.5
CAL_HEAD_MIN_SAMPLES = 30
HEAD_GROUP_WEIGHT = 3.0  # poids d'une cible « tête en mouvement » relativement à une cible fixe

FIT_SIGMA = 0.03        # bruit typique par échantillon (fraction de fenêtre), échelle de la perte robuste
MODEL_MAX_FRAMES = 150  # images utilisées pour construire le modèle de tête

# Points du maillage MediaPipe à peu près rigides (pas de bouche, mâchoire, sourcils ni paupières).
RIGID_IDX = sorted({
    1, 4, 5, 6, 168, 197, 195, 45, 275, 48, 278, 98, 327, 220, 440,           # nez
    10, 151, 9, 8, 109, 338, 67, 297, 103, 332, 54, 284, 108, 337,            # front
    33, 133, 263, 362, 130, 359, 243, 463, 226, 446, 244, 464,                # coins / orbites
    116, 345, 117, 346, 118, 347, 123, 352, 50, 280, 101, 330, 36, 266,       # pommettes
    127, 356, 234, 454, 93, 323, 21, 251, 162, 389, 139, 368, 71, 301,        # tempes / côtés
})
_RI = {lm: i for i, lm in enumerate(RIGID_IDX)}
IRIS_A, IRIS_B = EYE_A["iris"], EYE_B["iris"]


# --- Géométrie de base ----------------------------------------------------------------------------
def landmarks_array(lm, w, h):
    """Landmarks MediaPipe -> tableau (N, 3) en pixels (z à la même échelle que x)."""
    return np.array([(p.x, p.y, p.z) for p in lm], dtype=float) * np.array([w, h, w])


def umeyama(x, y, wts=None):
    """Similarité qui minimise sum w |y - (s R x + t)|². Retourne (s, R, t)."""
    wts = np.ones(len(x)) if wts is None else wts
    ws = wts / wts.sum()
    mx, my = ws @ x, ws @ y
    xc, yc = x - mx, y - my
    u, d, vt = np.linalg.svd((yc * ws[:, None]).T @ xc)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1.0
    rot = (u * sign) @ vt
    s = float((d * sign).sum() / (ws @ (xc ** 2).sum(axis=1)))
    return s, rot, my - s * rot @ mx


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def euler_deg(rot):
    """(yaw, pitch, roll) de la tête en degrés : yaw > 0 = visage tourné vers la droite de l'image,
    pitch > 0 = tête baissée."""
    fwd = -rot[:, 2]  # direction du visage (l'axe z de la tête pointe vers l'arrière)
    yaw = math.degrees(math.atan2(fwd[0], -fwd[2]))
    pitch = math.degrees(math.atan2(fwd[1], math.hypot(fwd[0], fwd[2])))
    roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return yaw, pitch, roll


class Camera:
    """Caméra sténopé, centre optique au milieu de l'image."""

    def __init__(self, w, h, f=None):
        self.w, self.h = w, h
        self.f = f if f is not None else (w / 2) / math.tan(math.radians(CAM_HFOV_DEG) / 2)
        self.c = np.array([w / 2, h / 2])

    def ray(self, px):
        return np.array([(px[0] - self.c[0]) / self.f, (px[1] - self.c[1]) / self.f, 1.0])

    def backproject(self, lmk):
        """Points rigides en 3D, à un facteur d'échelle Z0 près (profondeur inconnue).

        MediaPipe donne (u, v) = projection perspective exacte et z = écart de profondeur exprimé en
        pixels (dz * f / Z0). Un point est donc à la profondeur Z = Z0 (1 + z / f) et vaut
        Z0 (1 + z / f) K^-1 [u, v, 1] : tout est proportionnel à Z0, que la similarité de Procrustes
        retrouve comme facteur d'échelle. Pas d'approximation orthographique, la perspective (tête
        décalée sur le côté, proche de la caméra) est prise en compte exactement.
        """
        q = lmk[RIGID_IDX]
        rays = np.column_stack([(q[:, :2] - self.c) / self.f, np.ones(len(q))])
        return rays * (1.0 + q[:, 2:3] / self.f)


# --- Modèle de la tête ----------------------------------------------------------------------------
class HeadModel:
    """Forme 3D rigide de la tête (cm, repère tête : x droite, y bas, z vers l'arrière) + yeux."""

    def __init__(self, pts, eyes, cam):
        self.pts = pts      # (K, 3) points RIGID_IDX, centrés
        self.eyes = eyes    # (2, 3) centres des globes (A puis B)
        self.cam = cam

    @classmethod
    def build(cls, lmks, cam):
        """Procrustes généralisé sur une liste de tableaux de landmarks (N, 3) en pixels."""
        if len(lmks) > MODEL_MAX_FRAMES:
            lmks = [lmks[i] for i in np.linspace(0, len(lmks) - 1, MODEL_MAX_FRAMES).astype(int)]
        clouds = [cam.backproject(l) for l in lmks]
        ref = clouds[0] - clouds[0].mean(axis=0)
        ref /= np.sqrt((ref ** 2).sum(axis=1).mean())
        for _ in range(4):
            aligned = []
            for q in clouds:
                s, r, t = umeyama(q, ref)
                aligned.append(s * q @ r.T + t)
            # moyenne robuste : médiane par point (rejette les images aberrantes)
            ref = np.median(np.array(aligned), axis=0)
            ref -= ref.mean(axis=0)
            ref /= np.sqrt((ref ** 2).sum(axis=1).mean())

        mid_a = (ref[_RI[EYE_A["outer"]]] + ref[_RI[EYE_A["inner"]]]) / 2
        mid_b = (ref[_RI[EYE_B["outer"]]] + ref[_RI[EYE_B["inner"]]]) / 2
        k = IPD_CM / np.linalg.norm(mid_b - mid_a)
        pts, mid_a, mid_b = ref * k, mid_a * k, mid_b * k

        # repère du visage : z = normale (direction de moindre variance), orientée vers l'arrière
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0))
        z = vt[2] if vt[2][2] > 0 else -vt[2]
        x = mid_b - mid_a
        x = x - (x @ z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        basis = np.stack([x, y, z], axis=1)
        pts, mid_a, mid_b = pts @ basis, mid_a @ basis, mid_b @ basis
        back = np.array([0.0, 0.0, EYE_DEPTH_CM])
        return cls(pts, np.stack([mid_a + back, mid_b + back]), cam)

    def pose(self, lmk):
        """(R tête->caméra, position caméra de l'origine du modèle en cm, 1/Z0)."""
        q = self.cam.backproject(lmk)
        s, r, t = umeyama(self.pts, q)
        for _ in range(2):  # repondération robuste (landmark mal placé, occlusion partielle)
            res = np.linalg.norm(q - (s * self.pts @ r.T + t), axis=1)
            sig = 1.4826 * np.median(res) + 1e-9
            s, r, t = umeyama(self.pts, q, 1.0 / (1.0 + (res / (2.5 * sig)) ** 2))
        return r, t / s, s

    def observe(self, lmk):
        """Mesure géométrique d'une image : dict (O, R, ang, ...) ou None."""
        r, center, s = self.pose(lmk)
        if s <= 0 or not np.all(np.isfinite(r)):
            return None
        eyes = center + self.eyes @ r.T                        # centres des globes, repère caméra
        cam = self.cam
        proj = eyes[:, :2] / eyes[:, 2:] * cam.f + cam.c
        irises = [lmk[IRIS_A, :2], lmk[IRIS_B, :2]]
        if np.linalg.norm(irises[0] - proj[0]) > np.linalg.norm(irises[0] - proj[1]):
            irises.reverse()  # sécurité : chaque iris va avec l'œil le plus proche
        dirs = []
        for e, px in zip(eyes, irises):
            ray = cam.ray(px)
            a, b, c = ray @ ray, ray @ e, e @ e - EYE_RADIUS_CM ** 2
            disc = b * b - a * c
            lam = (b - math.sqrt(disc)) / a if disc > 0 else b / a  # face avant de la sphère
            v = lam * ray - e
            dirs.append(r.T @ (v / np.linalg.norm(v)))          # axe optique dans le repère tête
        v = dirs[0] + dirs[1]
        v /= np.linalg.norm(v)
        theta = math.atan2(v[0], -v[2])
        phi = math.atan2(v[1], math.hypot(v[0], v[2]))
        return {"O": eyes.mean(axis=0), "R": r, "ang": np.array([theta, phi]), "dirs": dirs}


# --- Modèle de regard géométrique -----------------------------------------------------------------
def _basis(ang, bv):
    th, ph = ang[:, 0], ang[:, 1]
    return np.stack([np.ones_like(th), th, ph, th * th, ph * ph, th * ph, bv], axis=1)


N_BASIS = 7
# Paramètres : 0-5 affine plan->fenêtre | 6-12 correction θ | 13-19 correction φ |
#              20-21 inclinaison de l'écran (autour de x, y) | 22 log(facteur de focale)
P_SCREEN, P_TH, P_PH, P_TILT, P_KF, N_PARAMS = 0, 6, 13, 20, 22, 23
PRIOR_SIGMA = np.full(N_PARAMS, np.inf)
PRIOR_SIGMA[P_TH:P_TH + N_BASIS] = [0.15, 0.5, 0.2, 1.0, 1.0, 1.0, 0.2]
PRIOR_SIGMA[P_PH:P_PH + N_BASIS] = [0.15, 0.2, 0.6, 1.0, 1.0, 1.0, 0.4]
PRIOR_SIGMA[P_TILT:P_TILT + 2] = 0.35
PRIOR_SIGMA[P_KF] = 0.3


def levenberg_marquardt(fun, p, iters=60, tol=1e-8):
    r = fun(p)
    cost, mu = r @ r, 1e-3
    for _ in range(iters):
        jac = np.empty((len(r), len(p)))
        for j in range(len(p)):
            hstep = 1e-6 * max(1.0, abs(p[j]))
            pj = p.copy()
            pj[j] += hstep
            jac[:, j] = (fun(pj) - r) / hstep
        a, g = jac.T @ jac, jac.T @ r
        dg = np.diag(a) + 1e-9
        while True:
            pn = p + np.linalg.solve(a + mu * np.diag(dg), -g)
            rn = fun(pn)
            cn = rn @ rn
            if np.isfinite(cn) and cn < cost:
                break
            mu *= 4.0
            if mu > 1e10:
                return p
        done = cost - cn < tol * cost
        p, r, cost, mu = pn, rn, cn, max(mu / 3.0, 1e-9)
        if done:
            break
    return p


class GeoGazeModel:
    """Rayon de regard 3D (origine entre les yeux, direction R·g(θ',φ')) ∩ plan de l'écran."""

    def __init__(self, params=None):
        self.p = params

    @staticmethod
    def predict_with(p, obs):
        f = _basis(obs["ang"], obs["bv"])
        th = obs["ang"][:, 0] + f @ p[P_TH:P_TH + N_BASIS]
        ph = obs["ang"][:, 1] + f @ p[P_PH:P_PH + N_BASIS]
        gh = np.stack([np.cos(ph) * np.sin(th), np.sin(ph), -np.cos(ph) * np.cos(th)], axis=1)
        g = np.einsum("nij,nj->ni", obs["R"], gh)
        o = obs["O"] * np.array([1.0, 1.0, math.exp(p[P_KF])])
        rs = rot_x(p[P_TILT]) @ rot_y(p[P_TILT + 1])
        n = rs[:, 2]
        den = g @ n
        den = np.where(den > -1e-2, -1e-2, den)  # regard qui s'éloigne de l'écran : on borne
        pt = o + (-(o @ n) / den)[:, None] * g
        a, b = pt @ rs[:, 0], pt @ rs[:, 1]
        s = p[P_SCREEN:P_SCREEN + 6]
        return np.stack([s[0] * a + s[1] * b + s[2], s[3] * a + s[4] * b + s[5]], axis=1)

    def predict(self, obs):
        return self.predict_with(self.p, obs)

    def fit(self, obs, targets, weights, p0=None, iters=60):
        if p0 is None:
            p0 = np.zeros(N_PARAMS)
            p0[[0, 4]] = 1.0  # identité pour l'affine, puis vrai départ par moindres carrés linéaires
            ab = self.predict_with(p0, obs)
            design = np.column_stack([ab, np.ones(len(ab))])
            sw = np.sqrt(weights)[:, None]
            coef = np.linalg.lstsq(design * sw, targets * sw, rcond=None)[0]
            p0[0:3], p0[3:6] = coef[:, 0], coef[:, 1]
        sw = np.sqrt(weights)
        finite = np.isfinite(PRIOR_SIGMA)

        def residuals(p):
            r = ((self.predict_with(p, obs) - targets) / FIT_SIGMA).ravel()
            r = np.sign(r) * np.sqrt(2.0 * (np.sqrt(1.0 + r * r) - 1.0))  # perte soft-L1 (robuste)
            return np.concatenate([r * np.repeat(sw, 2), p[finite] / PRIOR_SIGMA[finite]])

        self.p = levenberg_marquardt(residuals, p0, iters=iters)
        return self


def stack_obs(obs_list, bvs):
    return {"O": np.array([o["O"] for o in obs_list]), "R": np.array([o["R"] for o in obs_list]),
            "ang": np.array([o["ang"] for o in obs_list]), "bv": np.asarray(bvs, dtype=float)}


def subset(obs, mask):
    return {k: v[mask] for k, v in obs.items()}


def group_weights(groups, kinds):
    w = np.zeros(len(groups))
    for g in np.unique(groups):
        m = groups == g
        w[m] = (HEAD_GROUP_WEIGHT if kinds[m][0] == "head" else 1.0) / m.sum()
    return w / w.mean()


# --- Calibration ----------------------------------------------------------------------------------
class Calibration3D:
    """Grille de cibles fixes (tête immobile) puis cibles « tête en mouvement ».

    Les images brutes (landmarks) sont gardées : le modèle de tête est construit à la fin à partir de
    toutes les images, puis les mesures géométriques sont recalculées avec ce modèle.
    """

    def __init__(self, head_phase=True):
        grid = [(CAL_MARGIN + (1 - 2 * CAL_MARGIN) * c / (CAL_COLS - 1),
                 CAL_MARGIN + (1 - 2 * CAL_MARGIN) * r / (CAL_ROWS - 1))
                for r in range(CAL_ROWS) for c in range(CAL_COLS)]
        random.shuffle(grid)
        self.steps = [(t, "fix") for t in grid]
        if head_phase:
            self.steps += [(t, "head") for t in CAL_HEAD_TARGETS]
        self.idx, self.t0, self.retries, self.skipped = 0, None, 0, 0
        self.cur = []
        self.samples = []  # (landmarks, bv, cible, groupe, type)

    @property
    def done(self):
        return self.idx >= len(self.steps)

    @property
    def target(self):
        return self.steps[min(self.idx, len(self.steps) - 1)][0]

    @property
    def kind(self):
        return self.steps[min(self.idx, len(self.steps) - 1)][1]

    def durations(self):
        return (CAL_HEAD_SETTLE_S, CAL_HEAD_COLLECT_S) if self.kind == "head" else (CAL_SETTLE_S, CAL_COLLECT_S)

    def phase_time(self, now):
        return 0.0 if self.t0 is None else now - self.t0

    def update(self, now, sample):
        """sample = (landmarks (N, 3), bv) si exploitable, sinon None."""
        if self.t0 is None:
            self.t0 = now
            self._announce()
        settle, collect = self.durations()
        t = now - self.t0
        if settle <= t < settle + collect:
            if sample is not None:
                self.cur.append(sample)
        elif t >= settle + collect:
            self._finish_point(now)

    def _announce(self):
        if self.kind == "head" and (self.idx == 0 or self.steps[self.idx - 1][1] != "head"):
            print("Calibration : garde les yeux sur le point et bouge doucement la tête "
                  "(gauche-droite, haut-bas, avant-arrière).")

    def _finish_point(self, now):
        cur, self.cur = self.cur, []
        need = CAL_HEAD_MIN_SAMPLES if self.kind == "head" else CAL_MIN_SAMPLES
        if len(cur) >= need:
            for lmk, bv in cur:
                self.samples.append((lmk, bv, self.target, self.idx, self.kind))
            self.idx += 1
            self.retries = 0
        elif self.retries < 2:
            self.retries += 1
        else:
            self.skipped += 1
            self.idx += 1
            self.retries = 0
        self.t0 = now
        if not self.done:
            self._announce()

    def fit(self, cam):
        """(HeadModel, GeoGazeModel, erreur train, erreur LOPO, erreur tête en mouvement) ou None."""
        fix_groups = {s[3] for s in self.samples if s[4] == "fix"}
        if len(fix_groups) < CAL_MIN_POINTS:
            return None
        head = HeadModel.build([s[0] for s in self.samples], cam)

        obs_list, keep = [], []
        for i, s in enumerate(self.samples):
            o = head.observe(s[0])
            if o is not None:
                obs_list.append(o)
                keep.append(i)
        smp = [self.samples[i] for i in keep]
        obs = stack_obs(obs_list, [s[1] for s in smp])
        targets = np.array([s[2] for s in smp], dtype=float)
        groups = np.array([s[3] for s in smp])
        kinds = np.array([s[4] for s in smp])

        # aberrants (clignement partiel, saccade) : seulement pour les cibles fixes, sur les angles œil/tête
        mask = np.ones(len(smp), dtype=bool)
        for g in np.unique(groups[kinds == "fix"]):
            m = groups == g
            a = obs["ang"][m]
            med = np.median(a, axis=0)
            mad = np.maximum(np.median(np.abs(a - med), axis=0) * 1.4826, 0.003)
            mask[np.where(m)[0][~np.all(np.abs(a - med) <= 3.5 * mad, axis=1)]] = False
        obs, targets, groups, kinds = subset(obs, mask), targets[mask], groups[mask], kinds[mask]

        weights = group_weights(groups, kinds)
        model = GeoGazeModel().fit(obs, targets, weights)
        err = np.abs(model.predict(obs) - targets)
        fix = kinds == "fix"
        train = err[fix].mean(axis=0)
        head_err = err[~fix].mean(axis=0) if (~fix).any() else None

        cv = []
        for g in np.unique(groups[fix]):
            w = weights.copy()
            w[groups == g] = 0.0
            m = GeoGazeModel().fit(obs, targets, w, p0=model.p.copy(), iters=3)
            te = groups == g
            cv.append(np.abs(m.predict(subset(obs, te)) - targets[te]))
        cv = np.vstack(cv).mean(axis=0)
        return head, model, train, cv, head_err


# --- Tracker --------------------------------------------------------------------------------------
class GazeTracker:
    """Webcam + MediaPipe + modèle géométrique 3D. Même interface que utils.GazeTracker.

    Attributs : gaze, frame, frame_id, usable, status, cal, alive (voir utils.GazeTracker), plus
    head_pose (yaw, pitch, roll, distance cm) et calibration_hint.
    """

    def __init__(self, cam_index=0, resolution=(CAM_WIDTH, CAM_HEIGHT), blink_threshold=BLINK_THRESHOLD,
                 min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA, annotate=False, head_phase=True):
        ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
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
        self.head_phase = head_phase
        self.filters = [OneEuroFilter(min_cutoff, beta, ONE_EURO_D_CUTOFF) for _ in range(2)]
        self.raw_hist = deque(maxlen=20)
        self.offset = np.zeros(2)
        self.gaze = np.array([0.5, 0.5])
        self.cam = None
        self.head, self.model, self.cal = None, None, None
        self.head_pose = None
        self.status = "NON CALIBRE - appuie sur C"
        self.blink_until, self.last_ts, self.fps, self.last_frame = 0.0, -1, 0.0, None
        self.last_valid = None
        self.frame, self.frame_id, self.now, self.usable = None, 0, time.monotonic(), False
        self.alive = True
        self.start_time = time.monotonic()
        self._cmds = queue.Queue()
        self._thread, self._stop = None, threading.Event()

    # --- cycle de vie ---
    def start(self):
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
        last = self.last_valid
        return last is not None and time.monotonic() - last <= max_age

    @property
    def calibration_hint(self):
        cal = self.cal
        if cal is None:
            return None
        if cal.kind == "head":
            return "Garde les yeux sur le point et bouge doucement la tête"
        return "Fixe le point, tête immobile"

    # --- boucle ---
    def step(self):
        while not self._cmds.empty():
            self._cmds.get_nowait()()

        success, image = self.cap.read()
        if not success:
            return False

        image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        if self.cam is None or (self.cam.w, self.cam.h) != (w, h):
            self.cam = Camera(w, h, self.cam.f * w / self.cam.w if self.cam is not None else None)
            if self.head is not None:
                self.head.cam = self.cam
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now = time.monotonic()
        ts = int((now - self.start_time) * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        if self.last_frame is not None:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_frame, 1e-3)
        self.last_frame = now

        result = self.landmarker.detect_for_video(mp_image, ts)

        feats, blink, lmk, obs = None, 0.0, None, None
        if result.face_landmarks:
            out = extract_features(result, w, h)
            if out is not None:
                feats, blink = out
                lmk = landmarks_array(result.face_landmarks[0], w, h)
                if self.head is not None:
                    try:
                        obs = self.head.observe(lmk)
                    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                        obs = None
        self.head_pose = None
        if obs is not None:
            yaw, pitch, roll = euler_deg(obs["R"])
            self.head_pose = (yaw, pitch, roll, float(obs["O"][2]))
            if max(abs(yaw), abs(pitch)) > MAX_HEAD_ANGLE_DEG:
                obs = None

        if self.annotate:
            if feats is not None:
                lines = [f"FPS {self.fps:.0f}", f"blink {blink:.2f} (seuil {self.blink_threshold})"]
                if self.head_pose is not None:
                    yaw, pitch, roll, dist = self.head_pose
                    lines.append(f"tete yaw {yaw:+.0f} pitch {pitch:+.0f} roll {roll:+.0f} {dist:.0f}cm")
                if obs is not None:
                    th, ph = np.degrees(obs["ang"])
                    lines.append(f"oeil/tete theta {th:+.1f} phi {ph:+.1f}")
                else:
                    lines.append(f"fx {feats[0]:+.3f} fy {feats[1]:+.3f} bv {feats[2]:+.2f}")
                draw_camera_overlay(image, result.face_landmarks[0], w, h, lines)
            else:
                cv2.putText(image, "Pas de visage", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        is_blink = blink > self.blink_threshold
        if is_blink:
            self.blink_until = now + BLINK_HOLD_S
        usable = feats is not None and not is_blink and now >= self.blink_until

        if self.cal is not None:
            self.cal.update(now, (lmk, float(feats[2])) if usable else None)
            if self.cal.done:
                self._finish_calibration()
        elif usable and (self.model is None or obs is not None):
            if self.model is not None:
                o = stack_obs([obs], [feats[2]])
                raw = self.model.predict(o)[0]
            else:
                raw = fallback_gaze(feats)
            self.raw_hist.append(raw)
            t = now - self.start_time
            self.gaze = np.array([self.filters[0](raw[0] + self.offset[0], t),
                                  self.filters[1](raw[1] + self.offset[1], t)])
            self.last_valid = now
        else:
            usable = False

        self.frame, self.now, self.usable = image, now, usable
        self.frame_id += 1
        return True

    def _finish_calibration(self):
        t0 = time.monotonic()
        try:
            fitted = self.cal.fit(self.cam)
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Calibration : erreur d'ajustement ({e})")
            fitted = None
        if fitted is None:
            self.status = "Calibration echouee (trop peu de points) - recommence avec C"
            print(self.status)
        else:
            self.head, self.model, train, cv, head_err = fitted
            self.offset = np.zeros(2)
            self.raw_hist.clear()
            self.status = (f"CALIBRE 3D - erreur validee: x {cv[0] * 100:.1f}% y {cv[1] * 100:.1f}% "
                           f"(train x {train[0] * 100:.1f}% y {train[1] * 100:.1f}%)")
            if head_err is not None:
                self.status += f" | tete en mouvement: x {head_err[0] * 100:.1f}% y {head_err[1] * 100:.1f}%"
            p = self.model.p
            print(self.status + f" | cibles ignorees: {self.cal.skipped} | ajustement "
                  f"{time.monotonic() - t0:.1f}s | gain oeil x {1 + p[P_TH + 1]:.2f} y {1 + p[P_PH + 2]:.2f} "
                  f"| focale x{math.exp(p[P_KF]):.2f} | inclinaison ecran {math.degrees(p[P_TILT]):+.0f} deg")
        self._reset_filters()
        self.last_valid = None
        self.cal = None

    def _reset_filters(self):
        for f in self.filters:
            f.reset()

    # --- commandes (thread-safe : exécutées par le thread du tracker) ---
    def start_calibration(self):
        def cmd():
            self.cal = Calibration3D(head_phase=self.head_phase)
            print("Calibration : suis chaque point des yeux, tête naturelle et immobile.")
        self._cmds.put(cmd)

    def cancel_calibration(self):
        def cmd():
            self.cal = None
            self.status = "Calibration annulee"
        self._cmds.put(cmd)

    def recenter(self):
        def cmd():
            if self.cal is None and self.raw_hist:
                self.offset = 0.5 - np.mean(self.raw_hist, axis=0)
                self._reset_filters()
                print("Recentre sur le milieu de la fenetre.")
        self._cmds.put(cmd)

    def reset(self):
        def cmd():
            self.head, self.model, self.cal, self.offset = None, None, None, np.zeros(2)
            self.raw_hist.clear()
            self.status = "NON CALIBRE - appuie sur C"
            self._reset_filters()
        self._cmds.put(cmd)

    def save_calibration(self, path):
        """Sauvegarde le modèle de tête + regard (fichier .npz). Utile entre deux lancements."""
        def cmd():
            if self.model is None:
                print("Rien a sauvegarder : pas de calibration.")
                return
            np.savez(path, pts=self.head.pts, eyes=self.head.eyes, f_ratio=self.cam.f / self.cam.w,
                     params=self.model.p)
            print(f"Calibration sauvegardee dans {path}")
        self._cmds.put(cmd)

    def load_calibration(self, path):
        def cmd():
            try:
                d = np.load(path)
            except OSError as e:
                print(f"Impossible de charger la calibration : {e}")
                return
            if self.cam is None:  # pas encore d'image : résolution demandée
                self.cam = Camera(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.cam.f = float(d["f_ratio"]) * self.cam.w
            self.head = HeadModel(d["pts"], d["eyes"], self.cam)
            self.model = GeoGazeModel(d["params"])
            self.offset = np.zeros(2)
            self.raw_hist.clear()
            self._reset_filters()
            self.status = "CALIBRE 3D (chargee)"
            print(f"Calibration chargee depuis {path}")
        self._cmds.put(cmd)

    def calibration_state(self):
        """(cible (x, y) normalisée, progression 0..1, en collecte ?, index, total) ou None."""
        cal = self.cal
        if cal is None:
            return None
        t = cal.phase_time(time.monotonic())
        settle, collect = cal.durations()
        frac = min(t / (settle + collect), 1.0)
        return cal.target, frac, t >= settle, min(cal.idx + 1, len(cal.steps)), len(cal.steps)
