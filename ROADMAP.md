# ROADMAP

Interface de sélection par le regard (webcam + MediaPipe) pour piloter un ordinateur / un
Raspberry Pi sans les mains : grille de boutons configurable en YAML/TOML/JSON, clic par
fixation, actions clavier, GPIO et HTTP.

## État actuel

- [x] Prototypes de sélection par le regard (`main.py`, `test_xav.py`, `landmarks_visu.py`)
- [x] Tracker v1 (`victor/utils.py`) : calibration 16 points, ridge, compensation de tête, filtre One Euro, gel au clignement
- [x] Tracker v2 (`victor/utils_v2.py`) : modèle 3D tête + globes oculaires, robuste aux mouvements de tête, phase de calibration « tête en mouvement »
- [x] Interface pygame (`victor/interface.py`) : grille 2x2 / 2x3, bandes latérales de défilement, zone morte, thèmes, animations
- [x] Menus et actions décrits dans `config.yaml` (validation avec messages d'erreur lisibles)
- [x] Actions : `touches`, `ecrire`, `lancer`, `attendre`, `afficher`, navigation, `calibrer`, GPIO (gpiozero, simulé hors Pi), HTTP
- [x] Mode `--souris` pour tester un layout sans caméra
- [x] Recentrer à chaque clic en milieu de la cellule (recalibration continue au clic, `recalibration_*` dans la config)

## Priorité 1 — Le « beau backend » : rendre les actions réelles

Aujourd'hui `envoyer_combo` et `envoyer_texte` ([victor/actions.py](victor/actions.py)) ne font
qu'un `print` : les boutons `touches` et `ecrire` n'ont aucun effet.

- [ ] Choisir et brancher un backend clavier selon l'environnement :
  - [ ] Wayland : `wtype` (ou `ydotool`)
  - [ ] X11 : `xdotool`
  - [ ] Bas niveau / console : `uinput` via `python-evdev` (marche partout, demande les droits sur `/dev/uinput`)
- [ ] Détection automatique du backend au démarrage + réglage `avance.backend_clavier` pour forcer
- [ ] Gérer accents et caractères non ASCII dans `ecrire` (disposition AZERTY)
- [ ] Message clair à l'écran si aucun backend n'est disponible (au lieu d'un échec silencieux)
- [ ] Éventuellement : actions souris (`clic`, `deplacer`) pour piloter des applis existantes

## Priorité 2 — Fiabilité et confort d'usage

- [ ] Sauvegarder / recharger automatiquement la calibration au lancement
      (`save_calibration` / `load_calibration` existent dans `utils_v2` mais ne sont pas appelés par `main_v2.py`)
- [ ] Afficher l'erreur de calibration (validation leave-one-out) et proposer de recommencer si elle est mauvaise
- [ ] Indicateur de qualité du suivi (visage perdu, tête trop tournée, mauvaise lumière)
- [ ] Mesurer la précision réelle (erreur moyenne en px / degrés) sur un protocole fixe, v1 vs v2
- [ ] Régler les valeurs par défaut (`duree_clic`, filtres, zone morte) avec de vrais tests utilisateurs
- [ ] Mode "pause" activable au regard (ex. fixer la zone morte longtemps / fermer les yeux) pour éviter les clics involontaires
- [ ] Mode "Safe" lorsque l'utilisateur ne regarde pas l'écran

## Priorité 3 — Nettoyage du code

- [ ] Fusionner `main.py` et `main_v2.py` (quasi identiques) : un seul point d'entrée avec `--tracker v1|v2`
- [ ] Sortir `Navigation`, `Selecteur`, `TrackerSouris` dans leur propre module
- [ ] Décider du sort des prototypes à la racine (`main.py`, `test_xav.py`, `landmarks_visu.py`) : archiver dans `prototypes/` ou supprimer
- [ ] Ajouter `requirements.txt` / `pyproject.toml` (mediapipe, opencv, pygame, numpy, pyyaml, gpiozero optionnel)
- [ ] Ajouter `__pycache__/` au `.gitignore`
- [ ] Tests unitaires sur ce qui se teste sans caméra : lecture de config, analyse des actions, géométrie (`umeyama`, pose de tête)

## Priorité 4 — Déploiement Raspberry Pi

- [ ] Vérifier les performances sur Pi (images/s, latence) et ajuster `resolution_camera`
- [ ] Lancement automatique au démarrage (service systemd ou autostart de la session)
- [ ] Script d'installation (dépendances, droits `uinput` / GPIO, modèle MediaPipe)

## Documentation

- [ ] README : présentation, installation, lancement, touches clavier, écrire sa config
- [ ] Exemples de configurations prêtes à l'emploi (communication oui/non, domotique, navigateur)
