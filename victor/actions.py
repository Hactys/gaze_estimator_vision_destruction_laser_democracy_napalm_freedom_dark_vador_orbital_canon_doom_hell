"""Actions utilisables dans le fichier de configuration.

Dans le YAML, une action s'écrit « verbe arguments », par exemple :
    touches CTRL+C
    ecrire "Bonjour !"
    ouvrir Raccourcis

Pour ajouter une action, il suffit de décorer une fonction avec @action("nom") : elle reçoit le
contexte d'exécution puis les arguments (des chaînes de caractères). Le nombre d'arguments est
vérifié au chargement de la configuration grâce à la signature de la fonction ; on peut en plus
fournir un `verifier(args, menus)` qui renvoie un message d'erreur ou None.
"""
import difflib
import inspect
import json
import queue
import re
import shlex
import subprocess
import threading
import time
import warnings
import urllib.error
import urllib.parse
import urllib.request

ACTIONS = {}


class ErreurAction(Exception):
    """Erreur prévue (réseau injoignable, broche occupée...) : son message est affiché à l'écran."""


def action(nom, aide="", verifier=None):
    def deco(f):
        f.nom, f.aide, f.verifier = nom, aide, verifier
        ACTIONS[nom] = f
        return f
    return deco


# --- Analyse ----------------------------------------------------------------------------------
def analyser(texte, menus):
    """'touches CTRL+C' -> ('touches', ['CTRL+C']). Lève ValueError avec un message lisible."""
    try:
        morceaux = shlex.split(str(texte))
    except ValueError as e:
        raise ValueError(f'"{texte}" : guillemets mal fermés ({e})')
    if not morceaux:
        raise ValueError("action vide")
    nom, args = morceaux[0], morceaux[1:]
    f = ACTIONS.get(nom)
    if f is None:
        proches = difflib.get_close_matches(nom, ACTIONS, n=1)
        aide = f', voulais-tu dire "{proches[0]}" ?' if proches else ""
        raise ValueError(f'action "{nom}" inconnue{aide} (disponibles : {", ".join(sorted(ACTIONS))})')
    try:
        inspect.signature(f).bind(None, *args)
    except TypeError:
        raise ValueError(f'"{texte}" : mauvais nombre d\'arguments. Usage : {f.aide}')
    if f.verifier is not None:
        erreur = f.verifier(args, menus)
        if erreur:
            raise ValueError(f'"{texte}" : {erreur}')
    return nom, args


def est_navigation(etapes):
    return any(nom in ("ouvrir", "retour", "accueil") for nom, _ in etapes)


# --- Exécution --------------------------------------------------------------------------------
class Executeur:
    """Exécute les séquences d'actions une par une dans un thread dédié (l'interface ne gèle pas
    pendant un `attendre`). Les actions qui touchent à l'interface (navigation, messages) postent
    une commande dans `self.vers_interface`, que la boucle principale dépile à chaque image."""

    def __init__(self):
        self.vers_interface = queue.Queue()
        self._travaux = queue.Queue()
        threading.Thread(target=self._boucle, name="actions", daemon=True).start()

    def lancer(self, etapes):
        self._travaux.put(etapes)

    def _boucle(self):
        while True:
            for nom, args in self._travaux.get():
                try:
                    ACTIONS[nom](self, *args)
                except ErreurAction as e:
                    print(f"[ERREUR] action {nom} {args} : {e}")
                    self.vers_interface.put(("message", str(e)))
                    break  # on n'enchaîne pas la suite de la séquence après un échec
                except Exception as e:  # une action qui plante ne doit pas tuer l'interface
                    print(f"[ERREUR] action {nom} {args} : {e}")
                    self.vers_interface.put(("message", f"Erreur : {nom}"))


# --- Clavier ----------------------------------------------------------------------------------
MODIFICATEURS = {"CTRL", "ALT", "MAJ", "SHIFT", "SUPER", "ALTGR"}
TOUCHES_NOMMEES = {"ENTREE", "ECHAP", "TAB", "ESPACE", "RETOUR_ARRIERE", "SUPPR", "INSER",
                   "HAUT", "BAS", "GAUCHE", "DROITE", "DEBUT", "FIN", "PAGE_HAUT", "PAGE_BAS",
                   *(f"F{i}" for i in range(1, 13))}


def _verifier_combo(combo):
    for t in combo.upper().split("+"):
        if not (t in MODIFICATEURS or t in TOUCHES_NOMMEES or len(t) == 1):
            return f'touche "{t}" inconnue (ex. valides : CTRL, ALT, MAJ, A, 5, ENTREE, F5, HAUT)'
    return None


def _verifier_touches(args, menus):
    if not args:
        return "il faut au moins un raccourci, ex : touches CTRL+C"
    for combo in args:
        erreur = _verifier_combo(combo)
        if erreur:
            return erreur
    return None


def envoyer_combo(touches):
    """Point d'accroche du clavier virtuel. PLACEHOLDER : à brancher sur le backend retenu
    (wtype / ydotool / uinput selon l'environnement graphique du Raspberry Pi)."""
    print(f"[CLAVIER] combo {'+'.join(touches)}")


def envoyer_texte(texte):
    """PLACEHOLDER : voir envoyer_combo."""
    print(f"[CLAVIER] texte {texte!r}")


# --- Actions disponibles ----------------------------------------------------------------------
@action("touches", "touches CTRL+C   (plusieurs raccourcis à la suite : touches CTRL+A CTRL+C)",
        verifier=_verifier_touches)
def touches(ctx, *combos):
    for combo in combos:
        envoyer_combo(combo.upper().split("+"))


@action("ecrire", 'ecrire "texte à taper"')
def ecrire(ctx, texte):
    envoyer_texte(texte)


@action("lancer", 'lancer firefox   ou   lancer "firefox --new-window"')
def lancer(ctx, commande, *args):
    argv = shlex.split(commande) + list(args)
    subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def _verifier_attendre(args, menus):
    try:
        ok = 0 <= float(args[0]) <= 60
    except ValueError:
        ok = False
    return None if ok else "la durée doit être un nombre de secondes entre 0 et 60"


@action("attendre", "attendre 1.5   (en secondes)", verifier=_verifier_attendre)
def attendre(ctx, secondes):
    time.sleep(float(secondes))


@action("afficher", 'afficher "message à l\'écran"')
def afficher(ctx, texte):
    ctx.vers_interface.put(("message", texte))


def _verifier_ouvrir(args, menus):
    if args[0] not in menus:
        proches = difflib.get_close_matches(args[0], menus, n=1)
        aide = f', voulais-tu dire "{proches[0]}" ?' if proches else ""
        return f'le menu "{args[0]}" n\'existe pas{aide}'
    return None


@action("ouvrir", "ouvrir NomDuMenu", verifier=_verifier_ouvrir)
def ouvrir(ctx, menu):
    ctx.vers_interface.put(("ouvrir", menu))


@action("retour", "retour   (menu précédent)")
def retour(ctx):
    ctx.vers_interface.put(("retour",))


@action("accueil", "accueil   (menu principal)")
def accueil(ctx):
    ctx.vers_interface.put(("accueil",))


@action("calibrer", "calibrer   (relance la calibration du regard)")
def calibrer(ctx):
    ctx.vers_interface.put(("calibrer",))


# --- GPIO (Raspberry Pi) ----------------------------------------------------------------------
# Numérotation BCM (celle écrite « GPIO17 » sur les schémas de brochage), pas le numéro physique.
# gpiozero fonctionne sur Pi 4 et Pi 5 (backend lgpio). Sans gpiozero ou sans GPIO (ex : sur un PC),
# les actions sont simulées : elles s'affichent dans la console au lieu de planter.
BROCHES_MAX = 27
_gpio_verrou = threading.Lock()
_gpio_module = None          # module gpiozero, False si indisponible, None si pas encore essayé
_gpio_appareils = {}         # broche -> (type, objet gpiozero)
_gpio_simule = {}            # broche -> valeur (mode simulation)


def _broche(texte):
    """'17' ou 'GPIO17' -> 17. Lève ValueError."""
    m = re.fullmatch(r"(?:GPIO)?(\d{1,2})", str(texte).strip(), re.IGNORECASE)
    if not m or not 0 <= int(m.group(1)) <= BROCHES_MAX:
        raise ValueError(f'broche « {texte} » invalide : numéro BCM entre 0 et {BROCHES_MAX} (ex : 17 ou GPIO17)')
    return int(m.group(1))


def _nombre(texte, nom, lo, hi):
    try:
        v = float(texte)
    except ValueError:
        v = None
    if v is None or not lo <= v <= hi:
        raise ValueError(f"{nom} doit être un nombre entre {lo:g} et {hi:g} (reçu « {texte} »)")
    return v


def _verifier_gpio(*bornes):
    """Vérifie la broche (1er argument) puis les nombres suivants : bornes = [(nom, min, max), ...]."""
    def verifier(args, menus):
        try:
            _broche(args[0])
            for texte, (nom, lo, hi) in zip(args[1:], bornes):
                _nombre(texte, nom, lo, hi)
        except ValueError as e:
            return str(e)
        return None
    return verifier


def _appareil(broche, type_):
    """Objet gpiozero pour cette broche (créé à la demande, recréé si on change d'usage), ou None
    en mode simulation."""
    global _gpio_module
    with _gpio_verrou:
        if _gpio_module is None:
            try:
                import gpiozero
                _gpio_module = gpiozero
            except ImportError:
                print("[GPIO] gpiozero non installé : actions GPIO simulées.")
                _gpio_module = False
        if _gpio_module is False:
            return None
        deja = _gpio_appareils.get(broche)
        if deja is not None and deja[0] == type_:
            return deja[1]
        if deja is not None:
            deja[1].close()
            del _gpio_appareils[broche]
        classe = {"sortie": _gpio_module.DigitalOutputDevice, "pwm": _gpio_module.PWMOutputDevice,
                  "entree": _gpio_module.DigitalInputDevice}[type_]
        try:
            with warnings.catch_warnings():  # gpiozero avertit à chaque backend essayé sans succès
                warnings.simplefilter("ignore")
                app = classe(broche)
        except _gpio_module.BadPinFactory:
            print("[GPIO] pas de GPIO sur cette machine : actions GPIO simulées.")
            _gpio_module = False
            return None
        except _gpio_module.GPIOZeroError as e:
            raise ErreurAction(f"GPIO{broche} indisponible ({e or type(e).__name__})")
        _gpio_appareils[broche] = (type_, app)
        return app


def _simuler(broche, valeur, texte):
    _gpio_simule[broche] = valeur
    print(f"[GPIO simulé] GPIO{broche} {texte}")


@action("gpio_allumer", "gpio_allumer 17   (met la broche à l'état haut)", verifier=_verifier_gpio())
def gpio_allumer(ctx, broche):
    b = _broche(broche)
    app = _appareil(b, "sortie")
    app.on() if app else _simuler(b, 1, "allumée")


@action("gpio_eteindre", "gpio_eteindre 17   (met la broche à l'état bas)", verifier=_verifier_gpio())
def gpio_eteindre(ctx, broche):
    b = _broche(broche)
    app = _appareil(b, "sortie")
    app.off() if app else _simuler(b, 0, "éteinte")


@action("gpio_basculer", "gpio_basculer 17   (allumée <-> éteinte)", verifier=_verifier_gpio())
def gpio_basculer(ctx, broche):
    b = _broche(broche)
    app = _appareil(b, "sortie")
    if app:
        app.toggle()
    else:
        v = 0 if _gpio_simule.get(b) else 1
        _simuler(b, v, "allumée" if v else "éteinte")


@action("gpio_impulsion", "gpio_impulsion 17 0.5   (allume pendant 0.5 s puis éteint)",
        verifier=_verifier_gpio(("la durée", 0.01, 60)))
def gpio_impulsion(ctx, broche, secondes):
    b, d = _broche(broche), float(secondes)
    app = _appareil(b, "sortie")
    if app:
        app.blink(on_time=d, off_time=0, n=1, background=True)  # ne bloque pas les actions suivantes
    else:
        _simuler(b, 0, f"impulsion de {d:g} s")


@action("gpio_intensite", "gpio_intensite 18 40   (PWM : 0 à 100 %, ex : variateur de LED)",
        verifier=_verifier_gpio(("l'intensité", 0, 100)))
def gpio_intensite(ctx, broche, pourcent):
    b, p = _broche(broche), float(pourcent)
    app = _appareil(b, "pwm")
    if app:
        app.value = p / 100
    else:
        _simuler(b, p / 100, f"intensité {p:g} %")


@action("gpio_lire", 'gpio_lire 4   ou   gpio_lire 4 "Porte"   (affiche l\'état d\'une entrée)',
        verifier=_verifier_gpio())
def gpio_lire(ctx, broche, nom=None):
    b = _broche(broche)
    app = _appareil(b, "entree")
    valeur = app.value if app else _gpio_simule.get(b, 0)
    ctx.vers_interface.put(("message", f"{nom or f'GPIO{b}'} : {'haut' if valeur else 'bas'}"))


# --- HTTP (réseau local) ----------------------------------------------------------------------
# Pensé pour piloter des appareils du réseau local (prises connectées, ESP32, Home Assistant...) :
# délai court, pas de proxy système, « http:// » ajouté si absent (ex : 192.168.1.20/relay/0?turn=on).
HTTP_DELAI_S = 3.0
HTTP_TAILLE_MAX = 64 * 1024
_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _url(texte):
    return texte if re.match(r"https?://", texte, re.IGNORECASE) else "http://" + texte


def _corps(args):
    """Arguments après l'URL -> (octets, type). Accepte :
       '{"etat": "on"}'      -> JSON tel quel
       etat=on niveau=3      -> JSON {"etat": "on", "niveau": 3}
       "un texte"            -> texte brut"""
    if not args:
        return None, None
    if len(args) == 1 and args[0].lstrip()[:1] in "{[":
        json.loads(args[0])  # lève ValueError si invalide
        return args[0].encode(), "application/json"
    if all("=" in a for a in args):
        donnees = {}
        for a in args:
            cle, val = a.split("=", 1)
            try:
                val = json.loads(val)  # 3 -> nombre, true -> booléen, sinon texte
            except ValueError:
                pass
            donnees[cle] = val
        return json.dumps(donnees).encode(), "application/json"
    return " ".join(args).encode(), "text/plain; charset=utf-8"


def _verifier_http(args, menus):
    parts = urllib.parse.urlsplit(_url(args[0]))
    if not parts.hostname or " " in args[0]:
        return f"adresse « {args[0]} » invalide (ex : 192.168.1.20/on ou http://lampe.local/on)"
    try:
        _corps(args[1:])
    except ValueError:
        return "le contenu JSON est invalide (vérifie les guillemets doubles et les accolades)"
    return None


def _requete(methode, url, args=()):
    url = _url(url)
    hote = urllib.parse.urlsplit(url).hostname
    donnees, type_ = _corps(args)
    req = urllib.request.Request(url, data=donnees, method=methode, headers={"User-Agent": "eye-tracker"})
    if type_:
        req.add_header("Content-Type", type_)
    try:
        with _http.open(req, timeout=HTTP_DELAI_S) as rep:
            return rep.read(HTTP_TAILLE_MAX).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise ErreurAction(f"{hote} a répondu erreur {e.code}")
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ErreurAction(f"{hote} injoignable")


@action("http_get", "http_get 192.168.1.20/relay/0?turn=on", verifier=_verifier_http)
def http_get(ctx, url):
    _requete("GET", url)


@action("http_post", "http_post 192.168.1.20/api etat=on   (ou un JSON entre apostrophes, ou rien)",
        verifier=_verifier_http)
def http_post(ctx, url, *contenu):
    _requete("POST", url, contenu)


@action("http_put", "http_put 192.168.1.20/api etat=off", verifier=_verifier_http)
def http_put(ctx, url, *contenu):
    _requete("PUT", url, contenu)


@action("http_lire", 'http_lire 192.168.1.30/temperature   ou   http_lire ADRESSE "Salon"   (affiche la réponse)',
        verifier=_verifier_http)
def http_lire(ctx, url, nom=None):
    texte = " ".join(_requete("GET", url).split())  # une seule ligne
    if len(texte) > 60:
        texte = texte[:57] + "..."
    ctx.vers_interface.put(("message", f"{nom} : {texte}" if nom else texte or "(réponse vide)"))
