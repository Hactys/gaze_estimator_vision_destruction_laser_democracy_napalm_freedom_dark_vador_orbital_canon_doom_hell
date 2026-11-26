"""Interface de sélection par le regard, décrite par un fichier YAML, TOML ou JSON.

Lancement :
    python main.py                  # utilise config.yaml (ou .yml / .toml / .json) à côté de ce fichier
    python main.py mon_layout.toml  # autre fichier de configuration (format déduit de l'extension)
    python main.py --souris         # le regard est remplacé par la souris (tester un layout sans caméra)

Touches : C = calibrer | Espace = recentrer (regarde le centre) | R = reset calibration |
          F = plein écran | D = image caméra (debug) | Échap = quitter (ou annuler la calibration)
"""
import math
import os
import sys
import time

import pygame

import actions
import config
from interface import DROITE, GAUCHE, MORTE, Rendu


class Navigation:
    """Pile de menus ouverts ; chaque menu est découpé en pages fixes (la dernière complétée par du vide)."""

    def __init__(self, menus, taille_page):
        self.menus, self.taille_page = menus, taille_page
        self.pile = [["principal", 0]]

    @property
    def menu(self):
        return self.pile[-1][0]

    @property
    def page(self):
        return self.pile[-1][1]

    @property
    def npages(self):
        return max(1, math.ceil(len(self.menus[self.menu]) / self.taille_page))

    def boutons(self):
        b, debut = self.menus[self.menu], self.page * self.taille_page
        return [b[i] if i < len(b) else None for i in range(debut, debut + self.taille_page)]

    def defiler(self, sens):
        self.pile[-1][1] = (self.page + sens) % self.npages

    def ouvrir(self, menu):
        self.pile.append([menu, 0])

    def retour(self):
        if len(self.pile) > 1:
            self.pile.pop()

    def accueil(self):
        del self.pile[1:]

    def fil(self):
        return "  ›  ".join("Accueil" if m == "principal" else m for m, _ in self.pile)


class Selecteur:
    """Déclenche une cible quand le regard y reste `duree` secondes sans interruption.
    Après un déclenchement il faut quitter la cible, sauf si `repeter` (bandes de défilement)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cible, self.depuis, self.bloque = None, 0.0, False

    def bloquer(self):
        """Empêche la cible actuelle de se déclencher tant qu'on ne l'a pas quittée."""
        self.bloque = True

    def update(self, cible, now, duree, repeter=False):
        if cible != self.cible:
            self.cible, self.depuis, self.bloque = cible, now, False
        if duree is None or self.bloque:
            return False
        if now - self.depuis >= duree:
            if repeter:
                self.depuis = now
            else:
                self.bloque = True
            return True
        return False

    def progression(self, now, duree):
        if duree is None or self.bloque or self.cible is None:
            return 0.0
        return min((now - self.depuis) / duree, 1.0)


class TrackerSouris:
    """Remplace le GazeTracker par la souris (option --souris)."""
    calibrating, calibrated, alive, status, frame, frame_id = False, True, True, "", None, 0

    def __init__(self):
        self.gaze = (0.5, 0.5)

    def maj(self, w, h):
        x, y = pygame.mouse.get_pos()
        self.gaze = (x / w, y / h)

    def gaze_fresh(self, _):
        return pygame.mouse.get_focused()

    def calibration_state(self):
        return None

    def start(self):
        return self

    def close(self):
        pass

    start_calibration = cancel_calibration = recenter = reset = lambda self: None


def creer_fenetre(cfg, plein_ecran):
    if plein_ecran:
        return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    return pygame.display.set_mode(tuple(int(v) for v in cfg.ecran.taille_fenetre), pygame.RESIZABLE)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    souris = "--souris" in sys.argv
    chemin = args[0] if args else config.chercher(os.path.dirname(os.path.abspath(__file__)))
    try:
        cfg = config.charger(chemin)
    except config.ErreurConfig as e:
        print(e)
        sys.exit(1)

    pygame.init()
    pygame.display.set_caption("Eye Tracker")
    plein_ecran = cfg.ecran.plein_ecran
    ecran = creer_fenetre(cfg, plein_ecran)
    rendu = Rendu(cfg)
    rendu.construire(ecran)

    if souris:
        tracker = TrackerSouris()
    else:
        from utils import GazeTracker  # import tardif : MediaPipe est long à charger
        av = cfg.avance
        tracker = GazeTracker(cam_index=av.camera, resolution=tuple(int(v) for v in av.resolution_camera),
                              blink_threshold=av.seuil_clignement, min_cutoff=av.filtre_stabilite,
                              beta=av.filtre_reactivite, annotate=av.afficher_camera)
    tracker.start()

    lignes, colonnes = cfg.ecran.grille
    nav = Navigation(cfg.menus, lignes * colonnes)
    selecteur = Selecteur()
    executeur = actions.Executeur()
    horloge = pygame.time.Clock()
    afficher_camera = cfg.avance.afficher_camera
    curseur, pop, etait_en_calibration = None, None, False
    print(__doc__)

    try:
        while tracker.alive:
            dt = horloge.tick(cfg.avance.images_par_seconde) / 1000
            now = time.monotonic()

            # --- évènements clavier / fenêtre ---
            quitter = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    quitter = True
                elif ev.type == pygame.VIDEORESIZE and not plein_ecran:
                    rendu.construire(pygame.display.get_surface())
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_ESCAPE:
                        if tracker.calibrating:
                            tracker.cancel_calibration()
                        else:
                            quitter = True
                    elif k == pygame.K_c:
                        tracker.start_calibration()
                    elif k == pygame.K_SPACE:
                        tracker.recenter()
                    elif k == pygame.K_r:
                        tracker.reset()
                    elif k == pygame.K_d:
                        afficher_camera = not afficher_camera
                        tracker.annotate = afficher_camera
                        rendu.tout = True
                    elif k == pygame.K_f:
                        plein_ecran = not plein_ecran
                        ecran = creer_fenetre(cfg, plein_ecran)
                        rendu.construire(ecran)
            if quitter:
                break

            # --- commandes venant des actions (navigation, messages) ---
            while not executeur.vers_interface.empty():
                cmd = executeur.vers_interface.get_nowait()
                if cmd[0] == "message":
                    rendu.message(cmd[1])
                    continue
                if cmd[0] == "calibrer":
                    tracker.start_calibration()
                    continue
                if cmd[0] == "ouvrir":
                    nav.ouvrir(cmd[1])
                elif cmd[0] == "retour":
                    nav.retour()
                elif cmd[0] == "accueil":
                    nav.accueil()
                selecteur.bloquer()  # le bouton sous le regard a changé : il faut d'abord le quitter

            # --- calibration ---
            if tracker.calibrating:
                rendu.dessiner_calibration(tracker.calibration_state())
                selecteur.reset()
                etait_en_calibration = True
                continue
            if etait_en_calibration:
                etait_en_calibration = False
                rendu.message("Calibration terminée" if tracker.calibrated else "Calibration annulée")

            # --- regard -> cible ---
            w, h = ecran.get_size()
            if souris:
                tracker.maj(w, h)
            regard_ok = tracker.gaze_fresh(cfg.avance.delai_perte_regard)
            gx = min(max(tracker.gaze[0], 0.0), 1.0) * (w - 1)
            gy = min(max(tracker.gaze[1], 0.0), 1.0) * (h - 1)
            # curseur affiché lissé (l'écran tourne plus vite que la caméra) ; la détection utilise le regard brut
            if curseur is None:
                curseur = [gx, gy]
            k = 1 - math.exp(-dt / 0.05)
            curseur[0] += (gx - curseur[0]) * k
            curseur[1] += (gy - curseur[1]) * k

            boutons = nav.boutons()
            cible, duree, couleur = None, None, None
            if regard_ok:
                cible = rendu.d.zone_at(gx, gy)
                if cible in (GAUCHE, DROITE):
                    duree = cfg.reglages.duree_clic if nav.npages > 1 else None
                elif cible != MORTE and boutons[cible] is not None:
                    duree = boutons[cible].duree_clic or cfg.reglages.duree_clic
                    couleur = boutons[cible].couleur

            if selecteur.update(cible, now, duree, repeter=cible in (GAUCHE, DROITE)):
                pop = ((gx, gy), now)
                if cible in (GAUCHE, DROITE):
                    sens = -1 if cible == GAUCHE else 1
                    nav.defiler(sens)
                    rendu.glisser(sens)
                else:
                    bouton = boutons[cible]
                    executeur.lancer(bouton.etapes)
                    if not bouton.navigue:
                        rendu.message(bouton.texte or "Action")
                        if cfg.reglages.retour_apres_action and len(nav.pile) > 1:
                            executeur.lancer([("accueil", [])])  # après la fin de la séquence
                boutons = nav.boutons()

            statut = None if tracker.calibrated else "Non calibré · touche C"
            rendu.dessiner(nav.menu, nav.page, nav.npages, boutons, nav.fil(), statut,
                           survol=cible if duree is not None else None,
                           progression=selecteur.progression(now, duree),
                           curseur=curseur, regard_ok=regard_ok, pop=pop, couleur_anneau=couleur,
                           camera=(tracker.frame, tracker.frame_id) if afficher_camera and not souris else None)
    finally:
        tracker.close()
        pygame.quit()


if __name__ == "__main__":
    main()
