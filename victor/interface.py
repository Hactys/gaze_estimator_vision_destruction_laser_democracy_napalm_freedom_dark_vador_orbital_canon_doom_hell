"""Rendu pygame de l'interface, pensé pour un Raspberry Pi.

Optimisations :
  * tout ce qui est statique (cartes, bandes, zone morte, en-tête) est pré-rendu dans une surface
    « fond », reconstruite seulement quand la page, le menu ou la taille de fenêtre change ;
  * à chaque image on ne redessine que les rectangles qui bougent (curseur, anneau, carte survolée,
    message) et on ne pousse à l'écran que ces rectangles (display.update(rects)) ;
  * les textes, polices et icônes sont mis en cache.
"""
import math
import time

import cv2
import pygame
import pygame.gfxdraw

GAUCHE, DROITE, MORTE = "gauche", "droite", "morte"
DUREE_GLISSEMENT = 0.22
DUREE_POP = 0.35

THEMES = {
    "sombre": dict(fond=(17, 19, 24), carte=(34, 38, 47), carte_retour=(27, 30, 38), contour=(52, 57, 70),
                   vide=(30, 33, 41), morte=(19, 21, 27), texte=(236, 238, 243), texte_doux=(140, 146, 160),
                   piste=(70, 76, 92), alerte=(247, 160, 60)),
    "clair": dict(fond=(236, 238, 242), carte=(255, 255, 255), carte_retour=(246, 247, 250), contour=(208, 212, 222),
                  vide=(226, 229, 235), morte=(232, 234, 239), texte=(24, 27, 34), texte_doux=(105, 110, 122),
                  piste=(200, 204, 214), alerte=(200, 110, 0)),
}


def melanger(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def ease_out(t):
    return 1 - (1 - t) ** 3


# --- Géométrie --------------------------------------------------------------------------------
class Disposition:
    """Découpage de l'écran (en pixels). Sert à la fois au dessin et à la détection du regard."""

    def __init__(self, w, h, ecran):
        self.w, self.h = w, h
        self.lignes, self.colonnes = ecran.grille
        self.bande = round(ecran.largeur_bandes / 100 * w)
        zm = ecran.zone_morte
        self.cx, self.cy = w / 2, h / 2
        self.a, self.b = zm.largeur / 200 * w, zm.hauteur / 200 * h
        self.n = 2 + 8 * zm.aplatissement_haut   # exposant de la super-ellipse du haut
        self.marge = max(4, round(0.012 * min(w, h)))
        self.entete = max(28, round(0.06 * h))
        self.cw = (w - 2 * self.bande) / self.colonnes
        self.ch = h / self.lignes
        m = self.marge

        self.cartes = []
        for r in range(self.lignes):
            for c in range(self.colonnes):
                x0, x1 = round(self.bande + c * self.cw), round(self.bande + (c + 1) * self.cw)
                y0, y1 = round(r * self.ch), round((r + 1) * self.ch)
                left = x0 + (m if c == 0 and self.bande == 0 else m // 2)
                right = x1 - (m if c == self.colonnes - 1 and self.bande == 0 else m // 2)
                top = self.entete if r == 0 else y0 + m // 2
                bottom = y1 - (m if r == self.lignes - 1 else m // 2)
                self.cartes.append(pygame.Rect(left, top, right - left, bottom - top))
        self.segments = [self._segment_visible(rc) for rc in self.cartes]

        self.zone_centre = pygame.Rect(self.bande, self.entete, w - 2 * self.bande, h - self.entete)
        self.bandes = {}
        if self.bande > 0:
            hb = h - self.entete - m
            self.bandes[GAUCHE] = pygame.Rect(m, self.entete, self.bande - m - m // 2, hb)
            self.bandes[DROITE] = pygame.Rect(w - self.bande + m // 2, self.entete, self.bande - m - m // 2, hb)

    def dans_zone_morte(self, x, y, marge=0.0):
        a, b = self.a + marge, self.b + marge
        if self.a <= 0 or self.b <= 0:
            return False
        dx, dy = (x - self.cx) / a, (y - self.cy) / b
        if dy >= 0:
            return dx * dx + dy * dy <= 1
        return abs(dx) ** self.n + abs(dy) ** self.n <= 1

    def zone_at(self, x, y):
        """MORTE, GAUCHE, DROITE ou l'index de la case (ordre de lecture)."""
        if self.dans_zone_morte(x, y):
            return MORTE
        if x < self.bande:
            return GAUCHE
        if x >= self.w - self.bande:
            return DROITE
        c = min(max(int((x - self.bande) / self.cw), 0), self.colonnes - 1)
        r = min(max(int(y / self.ch), 0), self.lignes - 1)
        return r * self.colonnes + c

    def contour_zone_morte(self, marge=0.0, pas=72):
        a, b = self.a + marge, self.b + marge
        pts = []
        for k in range(pas):
            t = 2 * math.pi * k / pas
            c, s = math.cos(t), math.sin(t)
            if s >= 0:   # moitié basse : ellipse
                pts.append((self.cx + a * c, self.cy + b * s))
            else:        # moitié haute : super-ellipse (aplatie)
                e = 2 / self.n
                pts.append((self.cx + a * math.copysign(abs(c) ** e, c), self.cy - b * abs(s) ** e))
        return pts

    def _segment_visible(self, rc):
        """Plus grand intervalle vertical (y0, y1) de la carte non masqué par la zone morte,
        sur la verticale du centre de la carte : c'est là qu'on centre le contenu."""
        y0, y1 = rc.top, rc.bottom
        if self.a <= 0 or self.b <= 0:
            return y0, y1
        a, b = self.a + self.marge, self.b + self.marge
        dx = abs(rc.centerx - self.cx) / a
        if dx >= 1:
            return y0, y1
        haut = self.cy - b * (1 - dx ** self.n) ** (1 / self.n)
        bas = self.cy + b * math.sqrt(1 - dx * dx)
        morceaux = [(y0, min(y1, haut)), (max(y0, bas), y1)]
        return max(morceaux, key=lambda s: s[1] - s[0])


# --- Rendu ------------------------------------------------------------------------------------
class Rendu:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t = THEMES[cfg.ecran.theme]
        self.accent = cfg.ecran.couleur_accent
        self.chemin_police = cfg.ecran.police or pygame.font.match_font(
            "dejavusans,notosans,liberationsans,freesans,arial")
        self._polices, self._textes, self._icones = {}, {}, {}
        self.animations = cfg.ecran.animations

    # --- caches ---
    def police(self, taille):
        taille = max(8, int(taille))
        if taille not in self._polices:
            self._polices[taille] = pygame.font.Font(self.chemin_police, taille)
        return self._polices[taille]

    def texte(self, s, taille, couleur):
        cle = (s, int(taille), couleur)
        if cle not in self._textes:
            if len(self._textes) > 400:
                self._textes.clear()
            self._textes[cle] = self.police(taille).render(s, True, couleur)
        return self._textes[cle]

    def icone(self, chemin, taille):
        cle = (chemin, int(taille))
        if cle not in self._icones:
            img = pygame.image.load(chemin).convert_alpha()
            k = taille / max(img.get_width(), img.get_height())
            self._icones[cle] = pygame.transform.smoothscale(
                img, (max(1, round(img.get_width() * k)), max(1, round(img.get_height() * k))))
        return self._icones[cle]

    # --- construction (à chaque changement de taille de fenêtre) ---
    def construire(self, ecran):
        self.ecran = ecran
        w, h = ecran.get_size()
        self.d = Disposition(w, h, self.cfg.ecran)
        self.rayon = max(6, round(0.022 * min(w, h)))
        self.r_anneau = max(16, round(0.032 * h))
        self.ep_anneau = max(3, round(0.006 * h))
        self.taille_texte = 0.055 * h * self.cfg.ecran.taille_texte / 100
        self.petit = max(12, round(self.d.entete * 0.42))
        self._pages, self._survols = {}, {}
        self.sprite_morte, self.pos_morte = self._sprite_zone_morte()
        self.bandes = {k: {etat: self._sprite_bande(k, etat) for etat in ("normal", "survol", "inactif")}
                       for k in self.d.bandes}
        self.sprite_curseur = self._sprite_curseur(False)
        self.sprite_curseur_perdu = self._sprite_curseur(True)
        self.fond, self.cle_fond = pygame.Surface((w, h)).convert(), None
        self.sales, self.tout = [], True
        self.anim, self.page_affichee = None, None
        self.toast, self.cam = None, (None, -1)

    def _sprite_zone_morte(self):
        d = self.d
        if d.a <= 0 or d.b <= 0:
            return None, (0, 0)
        ext = d.contour_zone_morte(d.marge)
        xs, ys = [p[0] for p in ext], [p[1] for p in ext]
        x0, y0 = math.floor(min(xs)) - 2, math.floor(min(ys)) - 2
        surf = pygame.Surface((math.ceil(max(xs)) - x0 + 3, math.ceil(max(ys)) - y0 + 3), pygame.SRCALPHA)
        ext = [(x - x0, y - y0) for x, y in ext]
        inte = [(x - x0, y - y0) for x, y in d.contour_zone_morte(0)]
        pygame.gfxdraw.filled_polygon(surf, ext, self.t["fond"])
        pygame.gfxdraw.aapolygon(surf, ext, self.t["fond"])
        pygame.gfxdraw.filled_polygon(surf, inte, self.t["morte"])
        pygame.gfxdraw.aapolygon(surf, inte, self.t["piste"])
        return surf, (x0, y0)

    def _sprite_bande(self, cote, etat):
        r = self.d.bandes[cote]
        surf = pygame.Surface(r.size).convert()
        surf.fill(self.t["fond"])
        loc = surf.get_rect()
        if etat == "inactif":
            pygame.draw.rect(surf, self.t["vide"], loc, 1, border_radius=self.rayon)
            couleur = self.t["contour"]
        else:
            fond = self.t["carte"] if etat == "normal" else melanger(self.t["carte"], self.accent, 0.18)
            pygame.draw.rect(surf, fond, loc, border_radius=self.rayon)
            if etat == "survol":
                pygame.draw.rect(surf, self.accent, loc, max(2, self.ep_anneau // 2 + 1), border_radius=self.rayon)
            couleur = self.t["texte"] if etat == "survol" else self.t["texte_doux"]
        # chevron épais anti-aliasé
        s = min(loc.w * 0.22, 26)
        e = max(3, s * 0.4)
        cx, cy = loc.centerx, loc.centery
        if cote == GAUCHE:
            pts = [(cx + s / 2, cy - s), (cx - s / 2, cy), (cx + s / 2, cy + s),
                   (cx + s / 2 + e, cy + s), (cx - s / 2 + e, cy), (cx + s / 2 + e, cy - s)]
        else:
            pts = [(cx - s / 2, cy - s), (cx + s / 2, cy), (cx - s / 2, cy + s),
                   (cx - s / 2 - e, cy + s), (cx + s / 2 - e, cy), (cx - s / 2 - e, cy - s)]
        pts = [(x - e / 2 * (1 if cote == GAUCHE else -1), y) for x, y in pts]
        pygame.gfxdraw.filled_polygon(surf, pts, couleur)
        pygame.gfxdraw.aapolygon(surf, pts, couleur)
        return surf

    def _sprite_curseur(self, perdu):
        r = max(5, round(0.009 * self.d.h))
        halo = r * 2 + 4
        surf = pygame.Surface((halo * 2 + 2, halo * 2 + 2), pygame.SRCALPHA)
        c = halo + 1
        coeur = self.t["texte_doux"] if perdu else (255, 255, 255)
        if not perdu:
            pygame.gfxdraw.filled_circle(surf, c, c, halo, (*self.accent, 55))
        pygame.gfxdraw.filled_circle(surf, c, c, r + 2, (0, 0, 0, 160))
        pygame.gfxdraw.filled_circle(surf, c, c, r, coeur)
        pygame.gfxdraw.aacircle(surf, c, c, r, coeur)
        return surf

    # --- cartes ---
    def _contenu(self, surf, rect, seg, bouton, couleur_texte):
        pad = max(6, round(rect.w * 0.06))
        larg = rect.w - 2 * pad
        haut = (seg[1] - seg[0]) * 0.86
        taille = self.taille_texte
        for _ in range(6):
            pol = self.police(taille)
            lignes = self._couper(pol, bouton.texte, larg) if bouton.texte else []
            h_ligne = pol.get_linesize()
            h_icone = min(haut * 0.55, larg * 0.6) if bouton.icone else 0
            ecart = h_ligne * 0.3 if (h_icone and lignes) else 0
            total = h_icone + ecart + h_ligne * len(lignes)
            if total <= haut and all(pol.size(li)[0] <= larg for li in lignes):
                break
            taille *= 0.85
        y = (seg[0] + seg[1]) / 2 - total / 2
        if bouton.icone and h_icone >= 8:
            img = self.icone(bouton.icone, h_icone)
            surf.blit(img, img.get_rect(midtop=(rect.centerx, round(y))))
            y += h_icone + ecart
        for li in lignes:
            img = self.texte(li, taille, couleur_texte)
            surf.blit(img, img.get_rect(midtop=(rect.centerx, round(y))))
            y += h_ligne

    @staticmethod
    def _couper(pol, texte, larg):
        lignes = []
        for para in texte.split("\n"):
            cur = ""
            for mot in para.split():
                essai = f"{cur} {mot}".strip()
                if pol.size(essai)[0] <= larg or not cur:
                    cur = essai
                else:
                    lignes.append(cur)
                    cur = mot
            lignes.append(cur)
        return lignes

    def _dessiner_carte(self, surf, rect, seg, bouton, survol):
        if bouton is None:
            pygame.draw.rect(surf, self.t["vide"], rect, 1, border_radius=self.rayon)
            return
        fond = self.t["carte_retour"] if bouton.est_retour else self.t["carte"]
        if bouton.couleur:
            fond = melanger(fond, bouton.couleur, 0.3)
        vif = bouton.couleur or self.accent
        if survol:
            fond = melanger(fond, vif, 0.18)
        pygame.draw.rect(surf, fond, rect, border_radius=self.rayon)
        if survol:
            pygame.draw.rect(surf, vif, rect, max(2, self.ep_anneau // 2 + 1), border_radius=self.rayon)
        elif not bouton.est_retour:
            pygame.draw.rect(surf, self.t["contour"], rect, 1, border_radius=self.rayon)
        couleur = self.t["texte_doux"] if bouton.est_retour and not survol else self.t["texte"]
        self._contenu(surf, rect, seg, bouton, couleur)

    def _page(self, cle, boutons):
        if cle not in self._pages:
            if len(self._pages) > 24:
                self._pages.clear()
            surf = pygame.Surface((self.d.w, self.d.h)).convert()
            surf.fill(self.t["fond"])
            for rect, seg, b in zip(self.d.cartes, self.d.segments, boutons):
                self._dessiner_carte(surf, rect, seg, b, False)
            self._pages[cle] = surf
        return self._pages[cle]

    def _survol(self, cle, i, bouton):
        k = (cle, i)
        if k not in self._survols:
            if len(self._survols) > 48:
                self._survols.clear()
            rect = self.d.cartes[i]
            surf = pygame.Surface(rect.size).convert()
            surf.fill(self.t["fond"])
            seg = (self.d.segments[i][0] - rect.top, self.d.segments[i][1] - rect.top)
            self._dessiner_carte(surf, surf.get_rect(), seg, bouton, True)
            self._survols[k] = surf
        return self._survols[k]

    # --- fond statique ---
    def _construire_fond(self, surf_page, page, npages, fil, statut):
        f = self.fond
        f.blit(surf_page, (0, 0))
        f.fill(self.t["fond"], (0, 0, self.d.w, self.d.entete))
        for cote, r in self.d.bandes.items():
            f.blit(self.bandes[cote]["normal" if npages > 1 else "inactif"], r)
        if self.sprite_morte is not None:
            f.blit(self.sprite_morte, self.pos_morte)
        cy = self.d.entete // 2
        img = self.texte(fil, self.petit, self.t["texte_doux"])
        f.blit(img, img.get_rect(midleft=(max(self.d.bande, self.d.marge) + self.d.marge, cy)))
        if npages > 1:
            r = max(3, self.petit // 4)
            pas = r * 4
            x0 = self.d.w / 2 - pas * (npages - 1) / 2
            for i in range(npages):
                x = round(x0 + i * pas)
                if i == page:
                    pygame.gfxdraw.filled_circle(f, x, cy, r, self.accent)
                    pygame.gfxdraw.aacircle(f, x, cy, r, self.accent)
                else:
                    pygame.gfxdraw.aacircle(f, x, cy, r, self.t["texte_doux"])
        if statut:
            img = self.texte(statut, self.petit, self.t["alerte"])
            f.blit(img, img.get_rect(midright=(self.d.w - max(self.d.bande, self.d.marge) - self.d.marge, cy)))

    # --- API ---
    def glisser(self, direction):
        """Lance l'animation de changement de page (+1 = page suivante, venant de la droite)."""
        if self.animations and self.page_affichee is not None:
            self.anim = (self.page_affichee, direction, time.monotonic())

    def message(self, texte):
        img = self.texte(texte, self.petit * 1.25, self.t["texte"])
        pad = img.get_height() // 2
        surf = pygame.Surface((img.get_width() + 4 * pad, img.get_height() + 2 * pad), pygame.SRCALPHA)
        pygame.draw.rect(surf, (*self.t["carte"], 245), surf.get_rect(), border_radius=surf.get_height() // 2)
        pygame.draw.rect(surf, self.accent, surf.get_rect(), 2, border_radius=surf.get_height() // 2)
        surf.blit(img, (2 * pad, pad))
        rect = surf.get_rect(midbottom=(self.d.w // 2, self.d.h - self.d.marge * 3))
        self.toast = (surf, rect, time.monotonic())

    def dessiner(self, menu, page, npages, boutons, fil, statut, survol, progression, curseur, regard_ok,
                 pop=None, couleur_anneau=None, camera=None):
        """Dessine une image. survol = GAUCHE/DROITE/index de case/None ; curseur = (x, y) en pixels."""
        now = time.monotonic()
        ecran, d = self.ecran, self.d
        self.cle_page = (menu, page)
        surf_page = self._page(self.cle_page, boutons)
        cle = (menu, page, npages, fil, statut)
        if cle != self.cle_fond:
            self._construire_fond(surf_page, page, npages, fil, statut)
            self.cle_fond, self.tout = cle, True

        if self.anim is not None:
            ancienne, direction, t0 = self.anim
            t = (now - t0) / DUREE_GLISSEMENT
            if t >= 1:
                self.anim = None
                self.tout = True
            else:
                ecran.blit(self.fond, (0, 0))
                dec = round(ease_out(t) * d.zone_centre.w) * direction
                ecran.set_clip(d.zone_centre)
                ecran.blit(ancienne, (-dec, 0))
                ecran.blit(surf_page, (direction * d.zone_centre.w - dec, 0))
                ecran.set_clip(None)
                if self.sprite_morte is not None:
                    ecran.blit(self.sprite_morte, self.pos_morte)
                self.tout = True
                survol = None
        if self.anim is None:
            if self.tout:
                ecran.blit(self.fond, (0, 0))
            else:
                for r in self.sales:
                    ecran.blit(self.fond, r, r)
        self.page_affichee = surf_page

        rects = []
        if survol in d.bandes and npages > 1:
            rects.append(ecran.blit(self.bandes[survol]["survol"], d.bandes[survol]))
        elif isinstance(survol, int) and boutons[survol] is not None:
            rects.append(ecran.blit(self._survol(self.cle_page, survol, boutons[survol]), d.cartes[survol]))
            if self.sprite_morte is not None:
                r = self.sprite_morte.get_rect(topleft=self.pos_morte)
                if r.colliderect(rects[-1]):
                    ecran.set_clip(rects[-1])
                    ecran.blit(self.sprite_morte, self.pos_morte)
                    ecran.set_clip(None)

        if camera is not None:
            rects.append(self._camera(*camera))

        if self.toast is not None:
            surf, rect, t0 = self.toast
            age = now - t0
            if age > self.cfg.reglages.duree_message:
                self.toast = None
            else:
                reste = self.cfg.reglages.duree_message - age
                surf.set_alpha(int(255 * min(1.0, reste / 0.3, age / 0.12)))
                rects.append(ecran.blit(surf, rect))

        if curseur is not None:
            x, y = round(curseur[0]), round(curseur[1])
            R, ep = self.r_anneau, self.ep_anneau
            if progression > 0:
                rects.append(self._anneau(x, y, R, ep, progression, couleur_anneau or self.accent))
            if pop is not None:
                (px, py), t0 = pop
                t = (now - t0) / DUREE_POP
                if t < 1:
                    rr = round(R * (1 + 0.9 * ease_out(t)))
                    col = (*(couleur_anneau or self.accent), int(255 * (1 - t)))
                    for k in range(ep):
                        pygame.gfxdraw.aacircle(ecran, round(px), round(py), rr - k, col)
                    rects.append(pygame.Rect(round(px) - rr - 1, round(py) - rr - 1, 2 * rr + 3, 2 * rr + 3))
            sprite = self.sprite_curseur if regard_ok else self.sprite_curseur_perdu
            rects.append(ecran.blit(sprite, sprite.get_rect(center=(x, y))))

        rects = [r.clip(ecran.get_rect()) for r in rects]
        if self.tout:
            pygame.display.flip()
        else:
            pygame.display.update(self.sales + rects)
        self.sales, self.tout = rects, False

    def _anneau(self, x, y, R, ep, p, couleur):
        ecran = self.ecran
        pygame.draw.circle(ecran, self.t["piste"], (x, y), R, ep)
        a0 = -math.pi / 2
        n = max(2, int(64 * p))
        ext = [(x + R * math.cos(a0 + 2 * math.pi * p * k / n), y + R * math.sin(a0 + 2 * math.pi * p * k / n))
               for k in range(n + 1)]
        ri = R - ep
        inte = [(x + ri * math.cos(a0 + 2 * math.pi * p * k / n), y + ri * math.sin(a0 + 2 * math.pi * p * k / n))
                for k in range(n, -1, -1)]
        pts = ext + inte
        pygame.gfxdraw.filled_polygon(ecran, pts, couleur)
        pygame.gfxdraw.aapolygon(ecran, pts, couleur)
        return pygame.Rect(x - R - 2, y - R - 2, 2 * R + 5, 2 * R + 5)

    def _camera(self, frame, frame_id):
        surf, fid = self.cam
        if frame is not None and frame_id != fid:
            larg = max(160, self.d.w // 5)
            haut = round(frame.shape[0] * larg / frame.shape[1])
            petit = cv2.cvtColor(cv2.resize(frame, (larg, haut), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(petit.tobytes(), (larg, haut), "RGB").convert()
            self.cam = (surf, frame_id)
        if surf is None:
            return pygame.Rect(0, 0, 0, 0)
        return self.ecran.blit(surf, surf.get_rect(bottomleft=(self.d.marge, self.d.h - self.d.marge)))

    def dessiner_calibration(self, etat):
        """etat = tracker.calibration_state() : (cible, fraction, en_collecte, index, total)."""
        ecran = self.ecran
        ecran.fill(self.t["fond"])
        if etat is not None:
            (tx, ty), frac, collecte, idx, total = etat
            x, y = round(tx * self.d.w), round(ty * self.d.h)
            R = self.r_anneau + 8
            pygame.gfxdraw.aacircle(ecran, x, y, R, self.t["piste"])
            col = self.accent if collecte else self.t["texte_doux"]
            r = max(4, round(R * 0.85 * (1 - frac)))
            pygame.gfxdraw.filled_circle(ecran, x, y, r, col)
            pygame.gfxdraw.aacircle(ecran, x, y, r, col)
            pygame.gfxdraw.filled_circle(ecran, x, y, 3, self.t["texte"])
            img = self.texte(f"Calibration {idx}/{total} — fixe le point   (Échap = annuler)",
                             self.petit, self.t["texte_doux"])
            ecran.blit(img, img.get_rect(midtop=(self.d.w // 2, self.d.marge)))
        pygame.display.flip()
        self.tout, self.anim = True, None

