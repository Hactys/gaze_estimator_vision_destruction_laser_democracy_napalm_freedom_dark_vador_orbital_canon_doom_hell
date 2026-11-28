"""Chargement et validation du fichier de configuration (YAML, TOML ou JSON).

Le format est déduit de l'extension (.yaml / .yml, .toml, .json) ; la structure est la même
dans les trois formats. Tous les réglages sont optionnels : ce qui manque prend la valeur de
DEFAUTS. Les erreurs sont toutes collectées puis affichées d'un coup, avec le chemin de la clé
fautive et une suggestion.
"""
import difflib
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from types import SimpleNamespace

import yaml

import actions

DEFAUTS = {
    "ecran": {
        "plein_ecran": False,
        "taille_fenetre": [1280, 720],
        "grille": "2x2",
        "largeur_bandes": 7,
        "zone_morte": {"largeur": 22, "hauteur": 34, "aplatissement_haut": 0.5},
        "theme": "sombre",
        "couleur_accent": "#4F8EF7",
        "taille_texte": 100,
        "police": "",
        "animations": True,
    },
    "reglages": {
        "duree_clic": 2.0,
        "retour_apres_action": False,
        "texte_retour": "← Retour",
        "duree_message": 1.5,
    },
    "avance": {
        "camera": 0,
        "resolution_camera": [640, 480],
        "afficher_camera": False,
        "images_par_seconde": 30,
        "seuil_clignement": 0.6,
        "filtre_stabilite": 0.6,
        "filtre_reactivite": 3.0,
        "delai_perte_regard": 0.5,
        "recalibration_clic": True,
        "recalibration_part_centre": 0.5,
        "recalibration_poids": 1.0,
        "recalibration_oubli": 0.8,
        "recalibration_points_max": 20,
        "recalibration_fenetre": 0.5,
    },
}

# (min, max) pour les valeurs numériques
BORNES = {
    "ecran.largeur_bandes": (0, 30),
    "ecran.zone_morte.largeur": (0, 80),
    "ecran.zone_morte.hauteur": (0, 80),
    "ecran.zone_morte.aplatissement_haut": (0, 1),
    "ecran.taille_texte": (30, 300),
    "reglages.duree_clic": (0.2, 30),
    "reglages.duree_message": (0, 30),
    "avance.images_par_seconde": (5, 120),
    "avance.seuil_clignement": (0, 1),
    "avance.filtre_stabilite": (0.01, 10),
    "avance.filtre_reactivite": (0, 100),
    "avance.delai_perte_regard": (0, 10),
    "avance.recalibration_part_centre": (0, 1),
    "avance.recalibration_poids": (0, 20),
    "avance.recalibration_oubli": (0, 1),
    "avance.recalibration_points_max": (1, 200),
    "avance.recalibration_fenetre": (0.05, 5),
}
CHOIX = {"ecran.theme": ("sombre", "clair")}

COULEURS = {
    "rouge": "#E5484D", "vert": "#30A46C", "bleu": "#4F8EF7", "orange": "#F76B15",
    "jaune": "#F5D90A", "violet": "#8E4EC6", "rose": "#D6409F", "gris": "#8B8D98",
    "turquoise": "#12A594", "marron": "#A07653",
}
CLES_BOUTON = ("texte", "action", "icone", "duree_clic", "couleur")


class ErreurConfig(Exception):
    pass


@dataclass
class Bouton:
    texte: str
    etapes: list                      # [(nom_action, [args])]
    icone: str = None                 # chemin absolu
    duree_clic: float = None          # None = réglage global
    couleur: tuple = None             # (r, g, b) ou None
    est_retour: bool = False
    navigue: bool = field(init=False)

    def __post_init__(self):
        self.navigue = actions.est_navigation(self.etapes)


def lire_couleur(valeur):
    """'rouge' ou '#RRGGBB' -> (r, g, b). Lève ValueError."""
    s = str(valeur).strip().lower()
    s = COULEURS.get(s, s).lower()
    if re.fullmatch(r"#[0-9a-f]{6}", s):
        return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
    raise ValueError(f'couleur "{valeur}" invalide (ex : rouge, bleu, "#4F8EF7" ; '
                     f'noms connus : {", ".join(COULEURS)})')


def _oui_non(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("oui", "vrai", "o", "true"):
        return True
    if s in ("non", "faux", "n", "false"):
        return False
    raise ValueError("attendu : oui ou non (ou true / false)")


def _fusionner(defaut, donne, chemin, erreurs):
    """Fusionne récursivement en vérifiant clés et types. Retourne le dict complet."""
    if donne is None:
        donne = {}
    if not isinstance(donne, dict):
        erreurs.append(f"{chemin or 'racine'} : attendu une liste de réglages « clé: valeur »")
        return defaut
    out = {}
    for cle in donne:
        if cle not in defaut:
            proches = difflib.get_close_matches(str(cle), defaut, n=1)
            aide = f', voulais-tu dire "{proches[0]}" ?' if proches else f" (connus : {', '.join(defaut)})"
            erreurs.append(f'{chemin + "." if chemin else ""}{cle} : réglage inconnu{aide}')
    for cle, dv in defaut.items():
        c = f"{chemin}.{cle}" if chemin else cle
        if cle not in donne:
            out[cle] = dv
            continue
        v = donne[cle]
        try:
            if isinstance(dv, dict):
                out[cle] = _fusionner(dv, v, c, erreurs)
                continue
            if isinstance(dv, bool):
                v = _oui_non(v)
            elif isinstance(dv, (int, float)):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"attendu un nombre, reçu « {v} »")
                lo, hi = BORNES.get(c, (None, None))
                if lo is not None and not lo <= v <= hi:
                    raise ValueError(f"doit être entre {lo} et {hi} (reçu {v})")
            elif isinstance(dv, list):
                if (not isinstance(v, list) or len(v) != len(dv)
                        or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
                    raise ValueError(f"attendu une liste de {len(dv)} nombres, ex : {dv}")
            else:
                v = "" if v is None else str(v)
                if c in CHOIX and v not in CHOIX[c]:
                    raise ValueError(f"choix possibles : {', '.join(CHOIX[c])} (reçu « {v} »)")
            out[cle] = v
        except ValueError as e:
            erreurs.append(f"{c} : {e}")
            out[cle] = dv
    return out


def _lire_grille(texte):
    m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", str(texte))
    if not m:
        raise ValueError(f'« {texte} » invalide, attendu lignes x colonnes, ex : 2x2 ou 2x3')
    lignes, colonnes = int(m.group(1)), int(m.group(2))
    if not (1 <= lignes <= 4 and 1 <= colonnes <= 4):
        raise ValueError("lignes et colonnes doivent être entre 1 et 4")
    return lignes, colonnes


def _lire_menus(brut, dossier, reglages, erreurs):
    if brut is None:
        brut = {}
    if not isinstance(brut, dict):
        erreurs.append("menus : attendu une liste de menus « NomDuMenu: [boutons] »")
        brut = {}
    if "principal" not in brut:
        if brut:
            erreurs.append('menus : il manque le menu "principal" (celui affiché au démarrage)')
        brut = {"principal": [], **brut}
    noms = list(brut)
    menus = {}
    for nom, boutons in brut.items():
        chemin = f"menus.{nom}"
        if boutons is None:
            boutons = []
        if not isinstance(boutons, list):
            erreurs.append(f"{chemin} : attendu une liste de boutons (chaque bouton commence par « - »)")
            boutons = []
        liste = []
        if nom != "principal":
            liste.append(Bouton(reglages["texte_retour"], [("retour", [])], est_retour=True))
        for i, b in enumerate(boutons, 1):
            cb = f"{chemin}, bouton {i}"
            if not isinstance(b, dict):
                erreurs.append(f"{cb} : attendu « texte: ... » et « action: ... »")
                continue
            if "texte" in b and b["texte"] is not None:
                cb = f'{chemin}, bouton {i} « {b["texte"]} »'
            for cle in b:
                if cle not in CLES_BOUTON:
                    proches = difflib.get_close_matches(str(cle), CLES_BOUTON, n=1)
                    aide = f', voulais-tu dire "{proches[0]}" ?' if proches else f" (connus : {', '.join(CLES_BOUTON)})"
                    erreurs.append(f'{cb} : réglage "{cle}" inconnu{aide}')
            texte = "" if b.get("texte") is None else str(b["texte"])
            brut_actions = b.get("action")
            if brut_actions is None:
                erreurs.append(f"{cb} : il manque « action: ... »")
                brut_actions = []
            if not isinstance(brut_actions, list):
                brut_actions = [brut_actions]
            etapes = []
            for a in brut_actions:
                try:
                    etapes.append(actions.analyser(a, noms))
                except ValueError as e:
                    erreurs.append(f"{cb} : {e}")
            icone = None
            if b.get("icone"):
                icone = os.path.join(dossier, str(b["icone"]))
                if not os.path.isfile(icone):
                    erreurs.append(f'{cb} : icône introuvable « {icone} »')
                    icone = None
            duree = b.get("duree_clic")
            if duree is not None and (isinstance(duree, bool) or not isinstance(duree, (int, float))
                                      or not 0.2 <= duree <= 30):
                erreurs.append(f"{cb} : duree_clic doit être un nombre entre 0.2 et 30")
                duree = None
            couleur = None
            if b.get("couleur") is not None:
                try:
                    couleur = lire_couleur(b["couleur"])
                except ValueError as e:
                    erreurs.append(f"{cb} : {e}")
            if not texte and icone is None:
                erreurs.append(f"{cb} : il faut au moins un texte ou une icône")
            liste.append(Bouton(texte, etapes, icone, duree, couleur))
        menus[nom] = liste
    return menus


def _en_objet(d):
    return SimpleNamespace(**{k: _en_objet(v) if isinstance(v, dict) else v for k, v in d.items()})


FORMATS = {".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".json": "json"}
NOMS_PAR_DEFAUT = ("config.yaml", "config.yml", "config.toml", "config.json")
CONSEILS = {
    "yaml": "Vérifie l'indentation (espaces, pas de tabulations) et les « : ».",
    "toml": "Vérifie que les textes sont entre guillemets, les « = » et les titres de section [...].",
    "json": "Vérifie les guillemets doubles, les virgules (aucune après le dernier élément) et les accolades.",
}


def chercher(dossier):
    """Premier fichier config.yaml / .yml / .toml / .json trouvé dans le dossier (ou config.yaml)."""
    for nom in NOMS_PAR_DEFAUT:
        chemin = os.path.join(dossier, nom)
        if os.path.exists(chemin):
            return chemin
    return os.path.join(dossier, NOMS_PAR_DEFAUT[0])


def _lire_fichier(chemin):
    ext = os.path.splitext(chemin)[1].lower()
    fmt = FORMATS.get(ext)
    if fmt is None:
        raise ErreurConfig(f"{chemin} : extension « {ext} » non reconnue "
                           f"(formats acceptés : {', '.join(FORMATS)}).")
    pos = ""
    try:
        if fmt == "toml":
            with open(chemin, "rb") as f:
                return tomllib.load(f)
        with open(chemin, encoding="utf-8-sig") as f:  # -sig : tolère le BOM ajouté par certains éditeurs
            if fmt == "json":
                return json.load(f)
            return yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        raise ErreurConfig(f"{chemin} : le fichier doit être enregistré en UTF-8.")
    except yaml.YAMLError as e:
        if getattr(e, "problem_mark", None) is not None:
            pos = f" (ligne {e.problem_mark.line + 1}, colonne {e.problem_mark.column + 1})"
    except json.JSONDecodeError as e:
        pos = f" (ligne {e.lineno}, colonne {e.colno} : {e.msg})"
    except tomllib.TOMLDecodeError as e:
        pos = f" ({e})"
    raise ErreurConfig(f"{chemin} : erreur de syntaxe{pos}.\n{CONSEILS[fmt]}")


def charger(chemin):
    """Charge le fichier (s'il existe) et retourne la configuration complète. Lève ErreurConfig."""
    brut = {}
    if os.path.exists(chemin):
        brut = _lire_fichier(chemin)
    else:
        print(f"Pas de fichier {chemin} : réglages par défaut.")
    if not isinstance(brut, dict):
        raise ErreurConfig(f"{chemin} : le fichier doit contenir des réglages « clé: valeur ».")

    erreurs = []
    sections = {k: v for k, v in brut.items() if k != "menus"}
    cfg = _fusionner(DEFAUTS, sections, "", erreurs)
    try:
        cfg["ecran"]["grille"] = _lire_grille(cfg["ecran"]["grille"])
    except ValueError as e:
        erreurs.append(f"ecran.grille : {e}")
        cfg["ecran"]["grille"] = (2, 2)
    try:
        cfg["ecran"]["couleur_accent"] = lire_couleur(cfg["ecran"]["couleur_accent"])
    except ValueError as e:
        erreurs.append(f"ecran.couleur_accent : {e}")
        cfg["ecran"]["couleur_accent"] = lire_couleur(DEFAUTS["ecran"]["couleur_accent"])
    dossier = os.path.dirname(os.path.abspath(chemin))
    menus = _lire_menus(brut.get("menus"), dossier, cfg["reglages"], erreurs)

    if erreurs:
        raise ErreurConfig(f"{len(erreurs)} erreur(s) dans {chemin} :\n  - " + "\n  - ".join(erreurs))
    obj = _en_objet(cfg)
    obj.menus = menus
    return obj
