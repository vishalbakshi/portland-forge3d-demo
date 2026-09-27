"""Render the TriMet bus day animation with forge3d.

Every frame: pick the clock time and camera pose from the shot script, draw each bus
on the road as a disc (vector overlay triangles draped on the terrain), take a viewer
snapshot, stamp place names and the clock/count HUD on it, and pipe it into ffmpeg.

Run with the venv activated so forge3d can find its `interactive_viewer` binary:
    source ../forge3d/bin/activate && python render.py --preview
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import linemerge, unary_union

import forge3d as f3d
from buses import BusDay
from labels import LabelLayer, OrbitCamera, load_places

DEM = Path("data/portland_dem10_utm10n.tif")
TRIPS = Path("data/trips_2026-09-29.npz")
HOODS = Path("data/neighborhoods_utm10n.geojson")
CITIES = Path("data/cities_utm10n.geojson")
SEXTANTS = Path("data/sextants_utm10n.geojson")
Z_SCALE = 1.5
FOV = 38.0
SUN_ELEVATION = 40.0
VIEWER_WINDOW = (640, 360)  # on-screen window; snapshots render offscreen at full size

_TO_UTM = Transformer.from_crs(4326, 32610, always_xy=True)


def utm(lon: float, lat: float) -> tuple[float, float]:
    x, y = _TO_UTM.transform(lon, lat)
    return (float(x), float(y))


DOWNTOWN = utm(-122.6793, 45.5189)        # Pioneer Courthouse Square
METRO = (522000.0, 5036500.0)             # middle of the bus network
PORTLAND_LABEL = (527500.0, 5038200.0)    # where the city name sits on wide shots (inner SE)
QUADRANTS_VIEW = (526000.0, 5041800.0)    # centre of the four quadrants, near the Burnside Bridge
# Hand-placed so the four names sit mid-quadrant and clear of each other in the overhead shot.
QUADRANT_LABELS = {"NORTHWEST": (517500.0, 5045000.0), "NORTHEAST": (533000.0, 5045500.0),
                   "SOUTHEAST": (533500.0, 5036500.0), "SOUTHWEST": (520500.0, 5036000.0)}
DISC_SEGMENTS = 12

# Featured trips (indices into the trip file for Tuesday 2026-09-29).
LINE_14 = 1021   # Hawthorne/Foster westbound, crosses the Hawthorne Bridge at 08:40:30
LINE_35 = 2129   # Macadam/Greeley 10:11-11:33, Oregon City TC to University of Portland via the Steel Bridge
LINE_FX2 = 35    # Division 12:00-13:07, downtown over Tilikum Crossing to Gresham
LINE_20 = 1456   # Burnside/Stark 13:28-15:14, Gresham to Beaverton over the Burnside Bridge
LINE_18 = 1356   # Hillside 15:50-16:13, one full loop from Goose Hollow

# Crisp lines for close shots. The draped basemap is resampled to the DEM grid (10 m),
# so its thin lines go blocky up close; these vector ribbons fade in instead.
RIBBON_CLIP_M = 12000.0            # only build ribbons this far from downtown
# forge3d bug workaround: a layer's opacity is ignored once another layer is added (which
# happens every frame), but its visibility flag sticks. So each ribbon layer is uploaded at
# a few alpha steps and the camera distance picks which copy is visible. Few copies matter:
# every per-frame add/remove gets slower with the total vertex count already loaded.
RIBBON_STEPS = ((4500.0, 1.0), (11000.0, 0.45))  # (visible below this camera radius, alpha)

# Vector overlays are lit as ambient + sun: with ambient 1.0 flat ground gets this gain,
# so exact sRGB colours are converted to linear and divided by it.
LIGHT_GAIN = 1.0 + math.sin(math.radians(SUN_ELEVATION))


def srgb(r: int, g: int, b: int, a: float = 1.0) -> tuple:
    def lin(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (min(lin(r) / LIGHT_GAIN, 1.0), min(lin(g) / LIGHT_GAIN, 1.0), min(lin(b) / LIGHT_GAIN, 1.0), a)


# Bus layers: (name, radius in screen px, centre rgba, rim rgba, drape offset m, z order).
STYLES = {
    "dark": {
        "basemap": Path("data/basemap_dark.png"),
        "background": [0.003, 0.003, 0.005], "exposure": 1.6, "ibl": 0.1,
        "bus": [("glow", 9.0, (1.00, 0.62, 0.22, 0.55), (1.00, 0.62, 0.22, 0.0), 12.0, 1),
                ("core", 2.3, (1.00, 0.86, 0.55, 1.0), (1.00, 0.86, 0.55, 1.0), 14.0, 2)],
        "hero": [("hero_glow", 26.0, (0.20, 0.85, 1.00, 0.65), (0.20, 0.85, 1.00, 0.0), 16.0, 3),
                 ("hero_core", 5.0, (0.75, 1.00, 1.00, 1.0), (0.75, 1.00, 1.00, 1.0), 18.0, 4)],
        "featured_route": (0.20, 0.85, 1.00, 0.45),
        "dividers": (0.85, 0.88, 0.95, 0.8),
        "ribbons": [("routes", 5.0, (0.07, 0.12, 0.19, 0.85))],
        "hud": {"title": (240, 236, 228), "sub": (170, 170, 170), "clock": (255, 214, 150),
                "caption": (140, 230, 255), "halo": None},
        "labels": {
            "quadrant": {"size": 36, "bold": True, "tracking": 7, "color": (240, 240, 246), "halo": (0, 0, 0),
                         "halo_alpha": 220, "halo_width": 4},
            "portland": {"size": 30, "bold": True, "tracking": 7, "color": (236, 236, 242), "halo": (0, 0, 0),
                         "halo_alpha": 200, "halo_width": 3},
            "city": {"size": 19, "bold": True, "tracking": 2.5, "color": (200, 206, 216), "halo": (0, 0, 0),
                     "halo_alpha": 200, "halo_width": 3},
            "hood": {"size": 15, "bold": False, "tracking": 1.6, "color": (170, 182, 198), "halo": (0, 0, 0),
                     "halo_alpha": 190, "halo_width": 2},
        },
    },
    "light": {
        "basemap": Path("data/basemap_light.png"),
        "background": [0.80, 0.79, 0.76], "exposure": 1.2, "ibl": 1.0,
        "bus": [("shadow", 10.0, srgb(20, 20, 24, 0.22), srgb(20, 20, 24, 0.0), 12.0, 1),
                ("ring", 5.0, srgb(255, 255, 255), srgb(255, 255, 255), 13.0, 2),
                ("core", 3.8, srgb(20, 20, 24), srgb(20, 20, 24), 14.0, 3)],
        "hero": [("hero_halo", 28.0, srgb(24, 90, 200, 0.55), srgb(24, 90, 200, 0.0), 16.0, 4),
                 ("hero_ring", 7.8, srgb(255, 255, 255), srgb(255, 255, 255), 17.0, 5),
                 ("hero_core", 6.2, srgb(24, 90, 200), srgb(24, 90, 200), 18.0, 6)],
        "featured_route": srgb(24, 90, 200, 0.70),
        "dividers": srgb(34, 34, 40, 0.85),
        # White neighborhood borders under crisp grey streets, covering the blockier raster versions.
        "ribbons": [("hoods", 16.0, srgb(255, 255, 255, 0.95)), ("routes", 13.0, srgb(172, 172, 182, 0.95))],
        "hud": {"title": (24, 24, 28), "sub": (92, 92, 100), "clock": (20, 20, 24),
                "caption": (24, 90, 200), "halo": (255, 255, 255)},
        "labels": {
            "quadrant": {"size": 36, "bold": True, "tracking": 7, "color": (28, 28, 34), "halo": (255, 255, 255),
                         "halo_alpha": 235, "halo_width": 4},
            "portland": {"size": 30, "bold": True, "tracking": 7, "color": (52, 52, 60), "halo": (255, 255, 255),
                         "halo_alpha": 220, "halo_width": 3},
            "city": {"size": 19, "bold": True, "tracking": 2.5, "color": (78, 76, 84), "halo": (255, 255, 255),
                     "halo_alpha": 220, "halo_width": 3},
            "hood": {"size": 15, "bold": False, "tracking": 1.6, "color": (74, 68, 62), "halo": (255, 255, 255),
                     "halo_alpha": 230, "halo_width": 2},
        },
    },
}


@dataclass
class Key:
    """Shot-script keyframe.

    `target` is a UTM (x, y) or ("trip", index) to follow that bus. `feature`, `caption`
    apply from this key until the next one; `quads` (0..1) eases between keys.
    """
    t: float          # video seconds
    clock: float      # service-day hours
    target: object
    radius: float     # metres from target
    phi: float        # compass direction of the camera, 90 = south of target looking north
    theta: float      # tilt from straight down
    feature: int | None = None
    caption: str | None = None
    quads: float = 0.0


C14 = "Line 14 Hawthorne  ·  westbound over the Hawthorne Bridge"
C_QUADS = "Burnside splits north from south  ·  the Willamette splits east from west"
C35 = "Southwest  ·  Line 35 Macadam/Greeley  ·  Oregon City up the Willamette to University of Portland"
CFX2 = "Southeast  ·  FX2 Division  ·  downtown over Tilikum Crossing to Gresham"
C20 = "Northeast  ·  Line 20 Burnside/Stark  ·  Gresham to Beaverton over the Burnside Bridge"
C18 = "Northwest  ·  Line 18 Hillside  ·  one full loop through the West Hills"

# Camera moves (zoom-ins especially) are given room to breathe; follows keep their pace.
SHOTS = [
    Key(0.0, 5.00, METRO, 62000, 100, 38),
    Key(4.0, 7.75, METRO, 52000, 110, 45),                                     # the city wakes up
    Key(10.0, 8.55, DOWNTOWN, 9000, 125, 52, LINE_14, C14),                    # slow push in to downtown
    Key(13.5, 8.62, ("trip", LINE_14), 3200, 140, 52, LINE_14, C14),           # find the Line 14 bus
    Key(17.5, 8.74, ("trip", LINE_14), 2600, 170, 55),                         # over the Hawthorne Bridge
    Key(22.0, 9.67, QUADRANTS_VIEW, 36000, 90, 8, None, C_QUADS, 1.0),         # rise to the four quadrants
    Key(24.5, 10.08, QUADRANTS_VIEW, 34000, 92, 9, LINE_35, C35, 1.0),         # hold, then head southwest
    Key(28.5, 10.19, ("trip", LINE_35), 7000, 110, 45, LINE_35, C35),
    Key(36.5, 11.55, ("trip", LINE_35), 6000, 150, 48, LINE_FX2, CFX2),        # ...to University of Portland
    Key(40.0, 12.00, ("trip", LINE_FX2), 6000, 150, 48, LINE_FX2, CFX2),       # southeast
    Key(47.5, 13.12, ("trip", LINE_FX2), 6500, 110, 48, LINE_20, C20),         # ...to Gresham
    Key(50.0, 13.47, ("trip", LINE_20), 6500, 100, 48, LINE_20, C20),          # northeast, back west
    Key(58.5, 15.24, ("trip", LINE_20), 6500, 150, 48, LINE_18, C18),          # ...to Beaverton
    Key(62.5, 15.83, ("trip", LINE_18), 2800, 150, 45, LINE_18, C18),          # northwest: the Hillside loop
    Key(69.5, 16.22, ("trip", LINE_18), 2800, 250, 45),                        # slow, full loop
    Key(74.0, 17.75, METRO, 55000, 100, 40),                                   # evening rush, whole metro
    Key(78.0, 24.75, METRO, 64000, 90, 36),                                    # night falls
]


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Terrain:
    def __init__(self, path: Path):
        with rasterio.open(path) as src:
            self.z = src.read(1).astype(np.float32)
            self.inv = ~src.transform
            self.bounds = tuple(src.bounds)
        self.zmin = float(self.z.min())

    def world_y(self, x: float, y: float) -> float:
        col, row = self.inv * (x, y)
        r = int(np.clip(row, 0, self.z.shape[0] - 1))
        c = int(np.clip(col, 0, self.z.shape[1] - 1))
        return (float(self.z[r, c]) - self.zmin) * Z_SCALE


@dataclass
class Frame:
    clock: float
    target: tuple[float, float]
    radius: float
    phi: float
    theta: float
    feature: int | None
    caption: str | None
    quads: float


def follow_xy(day: BusDay, trip: int, clock: float, window_s: float = 120.0) -> tuple[float, float]:
    """Bus position averaged over +-window so the camera glides through stops and turns."""
    pts = np.array([day.trip_xy(trip, s) for s in clock + np.linspace(-window_s, window_s, 9)])
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def frame_at(t: float, day: BusDay) -> Frame:
    i = max(k for k in range(len(SHOTS) - 1) if SHOTS[k].t <= t) if t < SHOTS[-1].t else len(SHOTS) - 2
    a, b = SHOTS[i], SHOTS[i + 1]
    u = (t - a.t) / (b.t - a.t)
    clock = (a.clock + (b.clock - a.clock) * min(max(u, 0.0), 1.0)) * 3600
    e = smooth(u)

    def resolve(target):
        return follow_xy(day, target[1], clock) if isinstance(target[0], str) else target

    ta, tb = resolve(a.target), resolve(b.target)
    target = (ta[0] + (tb[0] - ta[0]) * e, ta[1] + (tb[1] - ta[1]) * e)
    radius = math.exp(math.log(a.radius) + (math.log(b.radius) - math.log(a.radius)) * e)
    return Frame(clock, target, radius, a.phi + (b.phi - a.phi) * e, a.theta + (b.theta - a.theta) * e,
                 a.feature, a.caption, a.quads + (b.quads - a.quads) * e)


def discs(xy: np.ndarray, radius_m: float, core: tuple, edge: tuple, first_id: int = 0):
    """Triangle-fan discs (centre colour `core`, rim colour `edge`) as overlay vertices/indices."""
    n = len(xy)
    if n == 0:
        return [], []
    ang = np.linspace(0, 2 * math.pi, DISC_SEGMENTS, endpoint=False)
    rim = xy[:, None, :] + radius_m * np.column_stack([np.cos(ang), np.sin(ang)])[None]
    pts = np.concatenate([xy[:, None, :], rim], axis=1).reshape(-1, 2)
    colors = np.tile(np.array([core] + [edge] * DISC_SEGMENTS, dtype=float), (n, 1))
    ids = np.repeat(np.arange(first_id, first_id + n), DISC_SEGMENTS + 1)
    verts = np.column_stack([pts[:, 0], np.zeros(len(pts)), -pts[:, 1], colors, ids]).tolist()
    j = np.arange(DISC_SEGMENTS)
    fan = np.stack([np.zeros(DISC_SEGMENTS, int), 1 + j, 1 + (j + 1) % DISC_SEGMENTS], axis=1)
    idx = ((np.arange(n) * (DISC_SEGMENTS + 1))[:, None, None] + fan[None]).reshape(-1).tolist()
    return verts, idx


def ribbon_lines(kind: str, day: BusDay):
    """Deduplicated line network for a ribbon layer: bus routes or neighborhood borders."""
    if kind == "routes":
        return unary_union([LineString(day.shape_xy(int(s))).simplify(1.0) for s in np.unique(day.trip_shape)])
    features = json.loads(HOODS.read_text())["features"]
    return unary_union([shape(f["geometry"]).buffer(0).boundary for f in features if f["geometry"]])


def quadrant_dividers():
    """The Willamette (west|east sextants) and Burnside (north|south sextants) as lines."""
    sx = {f["properties"]["PREFIX"]: shape(f["geometry"]).buffer(0)
          for f in json.loads(SEXTANTS.read_text())["features"]}
    west, east = unary_union([sx["NW"], sx["SW"], sx["S"]]), unary_union([sx["NE"], sx["N"], sx["SE"]])
    lines = [west.boundary.intersection(east.buffer(5.0)),
             sx["NW"].boundary.intersection(unary_union([sx["SW"], sx["S"]]).buffer(5.0)),
             sx["SE"].boundary.intersection(unary_union([sx["NE"], sx["N"]]).buffer(5.0))]
    return unary_union(lines)


def ribbons(network, width_m: float, rgba: tuple, center=None, clip_m: float | None = None):
    """Lines as flat quads, densified so draping follows the ground."""
    if center is not None:
        network = network.intersection(Point(center).buffer(clip_m))
    network = linemerge(network) if network.geom_type == "MultiLineString" else network
    starts, ends = [], []
    for part in getattr(network, "geoms", [network]):
        if part.geom_type != "LineString" or part.length < 1.0:
            continue
        c = np.asarray(part.simplify(3.0).segmentize(60.0).coords)[:, :2]
        starts.append(c[:-1])
        ends.append(c[1:])
    if not starts:
        return [], []
    a, b = np.concatenate(starts), np.concatenate(ends)
    d = b - a
    n = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(np.linalg.norm(d, axis=1), 1e-6)[:, None] * (width_m / 2)
    quad = np.stack([a + n, a - n, b + n, b - n], axis=1).reshape(-1, 2)
    ids = np.repeat(np.arange(len(a)), 4)
    verts = np.column_stack([quad[:, 0], np.zeros(len(quad)), -quad[:, 1],
                             np.tile(rgba, (len(quad), 1)), ids]).tolist()
    idx = ((np.arange(len(a)) * 4)[:, None] + np.array([0, 1, 2, 2, 1, 3])[None]).reshape(-1).tolist()
    return verts, idx


def ribbon_step(radius: float) -> int | None:
    """Index into RIBBON_STEPS for this camera distance, or None when ribbons are hidden."""
    return next((i for i, (max_radius, _) in enumerate(RIBBON_STEPS) if radius < max_radius), None)


def merge_layers(parts: list[tuple[list, list]]) -> tuple[list, list]:
    """Concatenate overlay meshes into one; later parts draw on top of earlier ones."""
    verts, idx = [], []
    for v_, i_ in parts:
        base = len(verts)
        verts += v_
        idx += [base + k for k in i_]
    return verts, idx


_FONTS: dict[tuple, ImageFont.FreeTypeFont] = {}


def load_font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONTS:
        avenir = "/System/Library/Fonts/Avenir Next.ttc"
        _FONTS[key] = (ImageFont.truetype(avenir, size, index=2 if bold else 5)  # Demi Bold / Medium
                       if os.path.exists(avenir) else ImageFont.load_default(size))
    return _FONTS[key]


def clock_text(seconds: float) -> str:
    h = int(seconds // 3600) % 24
    m = int(seconds % 3600 // 60)
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def hud_keepout(width: int, height: int, caption: str | None) -> list[tuple]:
    """Screen boxes the HUD occupies, so place names stay clear of them."""
    s = height / 1080
    boxes = [(0, 0, 760 * s, 150 * s), (width - 520 * s, 0, width, 150 * s)]
    if caption:
        w = load_font(int(30 * s), True).getlength(caption)
        boxes.append((0, height - 110 * s, 48 * s + w + 20 * s, height))
    return boxes


def hud(img: Image.Image, seconds: float, count: int, caption: str | None, date_label: str, colors: dict) -> Image.Image:
    s = img.height / 1080
    draw = ImageDraw.Draw(img, "RGBA")
    pad = int(48 * s)
    halo = {} if colors["halo"] is None else {"stroke_width": max(1, int(3 * s)), "stroke_fill": colors["halo"] + (200,)}

    def text(xy, t, font, color):
        draw.text(xy, t, font=font, fill=color + (255,), **halo)

    text((pad, pad), "Every TriMet bus, one day", load_font(int(40 * s), True), colors["title"])
    text((pad, pad + int(52 * s)), date_label, load_font(int(26 * s)), colors["sub"])
    big = load_font(int(72 * s), True)
    clock = clock_text(seconds)
    text((img.width - pad - draw.textlength(clock, font=big), pad - int(8 * s)), clock, big, colors["clock"])
    sub = f"{count} buses on the road"
    small = load_font(int(26 * s))
    text((img.width - pad - draw.textlength(sub, font=small), pad + int(76 * s)), sub, small, colors["sub"])
    if caption:
        text((pad, img.height - pad - int(30 * s)), caption, load_font(int(30 * s), True), colors["caption"])
    return img


def style_viewer(v, style: dict) -> None:
    """Basemap draped over the terrain, flat background, no sky.

    Raster overlays only draw in PBR mode. Vector overlays (the buses) are lit with the
    legacy `ambient` term, so ambient=1.0 keeps them at full brightness, while the PBR
    terrain's own ambient is scaled by ibl_intensity.
    """
    v.send_ipc({"cmd": "set_terrain", "background": style["background"], "zscale": Z_SCALE,
                "sun_azimuth": 315.0, "sun_elevation": SUN_ELEVATION, "sun_intensity": 1.0,
                "ambient": 1.0, "shadow": 0.6})
    v.send_ipc({"cmd": "set_terrain_pbr", "enabled": True, "exposure": style["exposure"],
                "ibl_intensity": style["ibl"], "normal_strength": 1.0, "sky": {"enabled": False}})
    v.load_overlay("basemap", style["basemap"].resolve(), opacity=1.0, z_order=0, preserve_colors=True)
    v.send_ipc({"cmd": "set_overlay_solid", "solid": True})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=sorted(STYLES), default="light")
    parser.add_argument("--preview", action="store_true", help="1280x720 instead of 1920x1080.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start", type=float, default=0.0, help="Video second to start at.")
    parser.add_argument("--end", type=float, default=None, help="Video second to stop at.")
    parser.add_argument("--stills", type=str, default=None, help="Comma-separated video seconds; save PNGs only.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    style = STYLES[args.style]
    width, height = (1280, 720) if args.preview else (1920, 1080)
    day = BusDay.load(TRIPS)
    terrain = Terrain(DEM)
    places = load_places(HOODS, CITIES, terrain.bounds, PORTLAND_LABEL, QUADRANT_LABELS)
    labels = LabelLayer(places, terrain.world_y, style["labels"])
    date_label = "Tuesday, September 29, 2026  ·  scheduled service (GTFS)"
    end = args.end if args.end is not None else SHOTS[-1].t
    if args.stills:
        times = [float(s) for s in args.stills.split(",")]
    else:
        times = list(np.arange(args.start, end, 1.0 / args.fps))
    tag = f"{args.style}{'_preview' if args.preview else ''}"
    out = args.out or Path(f"out/portland_buses_{tag}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = None
    if not args.stills:
        ffmpeg = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE,
        )

    tmp = Path(tempfile.mkdtemp(prefix="portland_frames_"))
    started = time.perf_counter()
    phase = defaultdict(float)
    with f3d.open_viewer_async(terrain_path=DEM.resolve(), width=VIEWER_WINDOW[0], height=VIEWER_WINDOW[1]) as v:
        style_viewer(v, style)
        ribbon_ids = []  # per kind: one overlay id per RIBBON_STEPS entry
        for z, (kind, width_m, rgba) in enumerate(style["ribbons"]):
            network = ribbon_lines(kind, day)
            ids = []
            for step, (_, alpha) in enumerate(RIBBON_STEPS):
                rv, ri = ribbons(network, width_m, rgba[:3] + (rgba[3] * alpha,), DOWNTOWN, RIBBON_CLIP_M)
                oid = v.add_vector_overlay(f"{kind}_{step}", rv, ri, drape=True, drape_offset=3.0 + z,
                                           z_order=z - 10)
                v.send_ipc({"cmd": "set_vector_overlay_visible", "id": oid, "visible": False})
                ids.append(oid)
            ribbon_ids.append(ids)
        shown_step = None
        dividers = quadrant_dividers()
        print(f"{len(places)} place names, ribbons uploaded", flush=True)

        overlay_ids: list[int] = []
        for n, t in enumerate(times):
            t0 = time.perf_counter()
            f = frame_at(t, day)
            target_world = (f.target[0], terrain.world_y(*f.target), -f.target[1])
            v.set_orbit_camera(phi_deg=f.phi, theta_deg=f.theta, radius=f.radius, fov_deg=FOV, target=target_world)
            step = ribbon_step(f.radius)
            if step != shown_step:
                for ids in ribbon_ids:
                    for old_or_new, visible in ((shown_step, False), (step, True)):
                        if old_or_new is not None:
                            v.send_ipc({"cmd": "set_vector_overlay_visible", "id": ids[old_or_new], "visible": visible})
                shown_step = step

            trips, xy = day.positions(f.clock)
            hero = trips == f.feature if f.feature is not None else np.zeros(len(trips), bool)
            m_per_px = f.radius * 2 * math.tan(math.radians(FOV / 2)) / height
            # Everything that changes per frame goes into ONE overlay (drawn in list order),
            # since each add/remove costs a scan of every loaded overlay.
            parts = []
            if f.quads > 0.01:
                rgba = style["dividers"]
                parts.append(ribbons(dividers, 3.5 * m_per_px, rgba[:3] + (rgba[3] * f.quads,)))
            if f.feature is not None:
                route = LineString(day.shape_xy(day.trip_shape[f.feature]))
                parts.append(ribbons(route, 5.0 * m_per_px, style["featured_route"]))
            parts += [discs(xy[~hero], px * m_per_px, core, rim) for _, px, core, rim, _, _ in style["bus"]]
            if hero.any():
                parts += [discs(xy[hero], px * m_per_px, core, rim) for _, px, core, rim, _, _ in style["hero"]]
            verts, idx = merge_layers(parts)
            for oid in overlay_ids:
                v.send_ipc({"cmd": "remove_vector_overlay", "id": oid})
            overlay_ids = [v.add_vector_overlay("frame", verts, idx, drape=True, drape_offset=12.0, z_order=1)]
            t1 = time.perf_counter()

            frame_png = tmp / "frame.png"
            v.snapshot(frame_png, width=width, height=height)
            t2 = time.perf_counter()
            img = Image.open(frame_png).convert("RGB")
            cam = OrbitCamera(target_world, f.radius, f.phi, f.theta, FOV, width, height)
            labels.draw(img, cam, f.radius, hud_keepout(width, height, f.caption), ease=1.0 if args.stills or n == 0 else 0.2,
                        kind_alpha={"quadrant": f.quads, "hood": 1.0 - f.quads, "portland": 1.0 - f.quads})
            img = hud(img, f.clock, len(trips), f.caption, date_label, style["hud"])
            t3 = time.perf_counter()
            if ffmpeg:
                ffmpeg.stdin.write(img.tobytes())
            else:
                still = out.with_name(f"still_{args.style}_{t:05.1f}s.png")
                img.save(still)
                print(f"wrote {still}")
            t4 = time.perf_counter()
            for k, dt in (("scene", t1 - t0), ("snapshot", t2 - t1), ("compose", t3 - t2), ("encode", t4 - t3)):
                phase[k] += dt
            if n % 30 == 0:
                rate = (time.perf_counter() - started) / (n + 1)
                split = " ".join(f"{k} {v_ / (n + 1):.2f}" for k, v_ in phase.items())
                print(f"frame {n + 1}/{len(times)}  t={t:5.2f}s  clock={clock_text(f.clock)}  buses={len(trips)}  "
                      f"{rate:.2f}s/frame [{split}]  eta {rate * (len(times) - n - 1) / 60:.1f} min", flush=True)

    if ffmpeg:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        print(f"Wrote {out} ({len(times)} frames in {(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
