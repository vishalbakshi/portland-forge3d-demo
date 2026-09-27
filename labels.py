"""Place names drawn onto rendered frames.

Names are projected with the same orbit camera forge3d uses (eye on a sphere around the
target, Y up, vertical FOV), then drawn with PIL so they get a real font, a halo, zoom
bands and simple decluttering. Visibility eases in and out across frames so labels
don't flicker as the camera moves.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont
from shapely.geometry import shape
from shapely.ops import polylabel, unary_union

AVENIR = "/System/Library/Fonts/Avenir Next.ttc"

# (kind, min camera radius, max camera radius) in metres: outside the band a label fades out.
BANDS = {
    "quadrant": (0.0, 1e9),   # shown only when the shot script turns quadrants on
    "portland": (14000.0, 1e9),
    "city": (11000.0, 1e9),
    "hood": (0.0, 30000.0),
}
FADE = 1.35  # band edges fade over this factor of radius


@dataclass
class Place:
    name: str
    kind: str
    x: float
    y: float
    area: float


def load_places(hoods_path: Path, cities_path: Path, bounds, portland_xy, quadrant_xy: dict) -> list[Place]:
    places = [Place("PORTLAND", "portland", portland_xy[0], portland_xy[1], 1e12)]
    places += [Place(name, "quadrant", x, y, 1e13) for name, (x, y) in quadrant_xy.items()]
    xmin, ymin, xmax, ymax = bounds
    parts: dict[str, list] = {}
    for f in json.loads(hoods_path.read_text())["features"]:
        p = f["properties"]
        if f["geometry"] and p.get("SHARED") != "Y":
            parts.setdefault(p["MAPLABEL"], []).append(shape(f["geometry"]).buffer(0))
    for name, geoms in parts.items():
        geom = unary_union(geoms)
        main = max(getattr(geom, "geoms", [geom]), key=lambda g: g.area)
        pt = polylabel(main, tolerance=20.0)
        places.append(Place(name.upper(), "hood", pt.x, pt.y, geom.area))
    for f in json.loads(cities_path.read_text())["features"]:
        name = f["properties"]["CITYNAME"]
        if name == "Portland" or not f["geometry"]:
            continue
        geom = shape(f["geometry"]).buffer(0)
        if geom.area < 2.5e6:  # skip tiny towns (Maywood Park, Rivergrove, ...)
            continue
        main = max(getattr(geom, "geoms", [geom]), key=lambda g: g.area)
        pt = polylabel(main, tolerance=50.0)
        if xmin < pt.x < xmax and ymin < pt.y < ymax:
            places.append(Place(name.upper(), "city", pt.x, pt.y, geom.area))
    return places


class OrbitCamera:
    """Matches forge3d's terrain orbit camera: eye = target + r*(sinθcosφ, cosθ, sinθsinφ)."""

    def __init__(self, target_world, radius, phi_deg, theta_deg, fov_deg, width, height):
        phi, theta = math.radians(phi_deg), math.radians(theta_deg)
        self.target = np.asarray(target_world, float)
        self.eye = self.target + radius * np.array([math.sin(theta) * math.cos(phi), math.cos(theta),
                                                    math.sin(theta) * math.sin(phi)])
        f = self.target - self.eye
        self.f = f / np.linalg.norm(f)
        s = np.cross(self.f, [0.0, 1.0, 0.0])
        self.s = s / np.linalg.norm(s)
        self.u = np.cross(self.s, self.f)
        self.t = math.tan(math.radians(fov_deg) / 2)
        self.w, self.h = width, height

    def project(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """N x 3 world points -> (N x 2 pixel coords, N depths along the view axis)."""
        d = world - self.eye
        x, y, z = d @ self.s, d @ self.u, d @ self.f
        z_safe = np.where(z > 1e-3, z, np.nan)
        ndc_x = x / (z_safe * self.t * self.w / self.h)
        ndc_y = y / (z_safe * self.t)
        return np.column_stack([(ndc_x + 1) * 0.5 * self.w, (1 - ndc_y) * 0.5 * self.h]), z


def band_alpha(kind: str, radius: float) -> float:
    lo, hi = BANDS[kind]
    a = 1.0
    if lo > 0:
        a = min(a, max(0.0, min(1.0, math.log(radius / (lo / FADE)) / math.log(FADE))))
    a = min(a, max(0.0, min(1.0, math.log((hi * FADE) / radius) / math.log(FADE))))
    return a


class LabelLayer:
    def __init__(self, places: list[Place], world_y, style: dict):
        self.places = places
        self.world = np.array([[p.x, world_y(p.x, p.y), -p.y] for p in places])
        self.style = style
        self.alpha = np.zeros(len(places))
        self.fonts: dict[tuple, ImageFont.FreeTypeFont] = {}

    def font(self, size: int, bold: bool):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(AVENIR, size, index=2 if bold else 5)
        return self.fonts[key]

    def draw(self, img, cam: OrbitCamera, radius: float, keepout: list[tuple], ease: float = 1.0,
             kind_alpha: dict[str, float] | None = None) -> None:
        """Draw visible names. `kind_alpha` scales whole kinds (e.g. quadrants on, neighborhoods off)."""
        kind_alpha = kind_alpha or {}
        s = img.height / 1080
        pts, depth = cam.project(self.world)
        draw = ImageDraw.Draw(img, "RGBA")
        order = sorted(range(len(self.places)),
                       key=lambda i: (-int(self.alpha[i] > 0.05),
                                      -{"quadrant": 4, "portland": 3, "city": 2, "hood": 1}[self.places[i].kind],
                                      -self.places[i].area))
        placed: list[tuple] = list(keepout)
        target = np.zeros(len(self.places))
        layout = {}
        for i in order:
            p = self.places[i]
            a = band_alpha(p.kind, radius) * kind_alpha.get(p.kind, 1.0)
            if a <= 0.01 or not np.isfinite(pts[i]).all():
                continue
            # Far-away labels near the horizon crowd together; fade them with distance.
            a *= max(0.0, min(1.0, (3.2 * radius - depth[i]) / (1.2 * radius)))
            if a <= 0.01:
                continue
            st = self.style[p.kind]
            font = self.font(max(8, int(st["size"] * s)), st["bold"])
            text = p.name
            spacing = st["tracking"] * s
            w = sum(draw.textlength(ch, font=font) for ch in text) + spacing * (len(text) - 1)
            h = font.size
            x0, y0 = pts[i][0] - w / 2, pts[i][1] - h / 2
            box = (x0 - 6 * s, y0 - 4 * s, x0 + w + 6 * s, y0 + h + 4 * s)
            if box[0] < 0 or box[1] < 0 or box[2] > img.width or box[3] > img.height:
                continue
            if any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in placed):
                continue
            placed.append(box)
            target[i] = a
            layout[i] = (x0, y0, font, spacing)
        self.alpha += (target - self.alpha) * ease
        for i, (x0, y0, font, spacing) in layout.items():
            a = self.alpha[i]
            if a <= 0.02:
                continue
            st = self.style[self.places[i].kind]
            fill = st["color"] + (int(255 * a),)
            halo = st["halo"] + (int(st["halo_alpha"] * a),)
            x = x0
            for ch in self.places[i].name:
                draw.text((x, y0), ch, font=font, fill=fill, stroke_width=max(1, int(st["halo_width"] * s)),
                          stroke_fill=halo)
                x += draw.textlength(ch, font=font) + spacing
