"""Build the basemap image that gets draped over the terrain.

Two styles:
  dark  - dark hillshade, rivers, faint glowing route lines (the night look)
  light - pale hillshade, Portland neighborhoods as pastel fills with white borders,
          surrounding cities tinted with grey city limits, soft blue rivers, grey routes

Rivers are cells below WATER_MAX_M in the 20 m DEM, which picks out the Columbia,
Willamette and Multnomah Channel cleanly without extra data. (The 10 m product puts the
Willamette surface at ~4.1 m, where a threshold also floods low ground on Sauvie Island,
so the 20 m mask is resampled onto the 10 m grid instead.)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFilter
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape
from shapely.ops import unary_union

from buses import BusDay

WATER_MAX_M = 4.0

STYLES = {
    "dark": {
        "land": (34, 37, 44), "land_light": 110.0, "water": (26, 58, 102),
        "route": (96, 150, 205, 150), "route_glow": True,
    },
    "light": {
        "land": (226, 223, 215), "land_light": 0.0, "land_shade": 0.32, "water": (164, 200, 226),
        "route": (150, 150, 160, 120), "route_glow": False,
        "city_fill": (214, 210, 200, 150), "city_edge": (150, 146, 138, 255),
        "hood_edge": (255, 255, 255, 235), "shared_fill": (236, 233, 226, 200),
        # Pastels; adjacent neighborhoods never share one (greedy graph colouring).
        "hood_palette": [(242, 205, 178), (205, 226, 184), (190, 214, 236), (226, 204, 236),
                         (243, 228, 170), (190, 228, 214), (238, 196, 204), (210, 204, 238)],
        "hood_alpha": 205,
    },
}


def hillshade(z: np.ndarray, res: float, azimuth=315.0, altitude=35.0) -> np.ndarray:
    dy, dx = np.gradient(z, res)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(hs, 0.0, 1.0)


def water_mask(water_dem: Path, shape: tuple[int, int], transform, crs) -> np.ndarray:
    with rasterio.open(water_dem) as src:
        low = (src.read(1) < WATER_MAX_M).astype(np.float32)
        out = np.zeros(shape, np.float32)
        reproject(low, out, src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear)
    return out


def load_features(path: Path) -> list[tuple[dict, object]]:
    fc = json.loads(path.read_text())
    return [(f["properties"], shape(f["geometry"]).buffer(0)) for f in fc["features"] if f["geometry"]]


def neighborhoods(path: Path):
    """(name -> dissolved polygon) for regular neighborhoods, plus the shared/overlap areas."""
    regular: dict[str, list] = {}
    shared = []
    for props, geom in load_features(path):
        if props.get("SHARED") == "Y":
            shared.append(geom)
        else:
            regular.setdefault(props["MAPLABEL"], []).append(geom)
    return {name: unary_union(parts) for name, parts in regular.items()}, shared


def color_graph(polys: dict[str, object], n_colors: int) -> dict[str, int]:
    """Greedy colouring so touching neighborhoods get different palette entries."""
    names = sorted(polys, key=lambda k: -polys[k].area)
    grown = {k: polys[k].buffer(15.0) for k in names}
    colors: dict[str, int] = {}
    counts = [0] * n_colors
    for i, name in enumerate(names):
        used = {colors[o] for o in colors if grown[name].intersects(grown[o])}
        free = [c for c in range(n_colors) if c not in used] or [i % n_colors]
        colors[name] = min(free, key=lambda c: counts[c])  # spread use across the palette
        counts[colors[name]] += 1
    return colors


class Canvas:
    """Draws UTM geometries into basemap pixel space."""

    def __init__(self, transform, scale: int, size: tuple[int, int]):
        self.inv, self.scale, self.size = ~transform, scale, size

    def px(self, coords) -> list[tuple[float, float]]:
        xy = np.asarray(coords)[:, :2]
        cols, rows = self.inv * (xy[:, 0], xy[:, 1])
        return list(zip((cols * self.scale).tolist(), (rows * self.scale).tolist()))

    def polygons(self, geom):
        return getattr(geom, "geoms", [geom])

    def fill(self, draw: ImageDraw.ImageDraw, geom, rgba) -> None:
        for poly in self.polygons(geom):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            draw.polygon(self.px(poly.exterior.coords), fill=rgba)

    def holes(self, draw: ImageDraw.ImageDraw, geom, rgba) -> None:
        for poly in self.polygons(geom):
            for ring in getattr(poly, "interiors", []):
                draw.polygon(self.px(ring.coords), fill=rgba)

    def outline(self, draw: ImageDraw.ImageDraw, geom, rgba, width: int) -> None:
        for poly in self.polygons(geom):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            for ring in [poly.exterior, *poly.interiors]:
                draw.line(self.px(ring.coords), fill=rgba, width=width, joint="curve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=sorted(STYLES), default="dark")
    parser.add_argument("--dem", type=Path, default=Path("data/portland_dem10_utm10n.tif"))
    parser.add_argument("--water-dem", type=Path, default=Path("data/portland_dem_utm10n.tif"),
                        help="DEM whose low cells define the rivers (the 20 m product).")
    parser.add_argument("--trips", type=Path, default=Path("data/trips_2026-09-29.npz"))
    parser.add_argument("--hoods", type=Path, default=Path("data/neighborhoods_utm10n.geojson"))
    parser.add_argument("--cities", type=Path, default=Path("data/cities_utm10n.geojson"))
    parser.add_argument("--scale", type=int, default=1, help="Basemap pixels per DEM pixel.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    style = STYLES[args.style]
    out = args.out or Path(f"data/basemap_{args.style}.png")

    with rasterio.open(args.dem) as src:
        z = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        res = src.res[0]

    hs = hillshade(z, res)
    land = np.array(style["land"], np.float32)
    if args.style == "dark":
        rgb = land[None, None, :] + (hs[..., None] - 0.5).clip(0, None) * style["land_light"] * 2
    else:
        # Multiply-style shading: flat ground keeps the land colour, shaded slopes darken.
        flat = np.sin(np.radians(35.0))
        shade = 1.0 + (hs - flat).clip(None, 0.25) * style["land_shade"]
        rgb = land[None, None, :] * shade[..., None]
    water = water_mask(args.water_dem, z.shape, transform, crs)
    w, h = z.shape[1] * args.scale, z.shape[0] * args.scale
    canvas = Canvas(transform, args.scale, (w, h))
    base = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB").resize((w, h), Image.Resampling.BICUBIC)
    base = base.convert("RGBA")

    if args.style == "light":
        # Neighbouring cities, then Portland neighborhoods on top.
        cities = [(p, g) for p, g in load_features(args.cities) if p["CITYNAME"] != "Portland"]
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for _, geom in cities:
            canvas.fill(draw, geom, style["city_fill"])
        base = Image.alpha_composite(base, layer)

        hoods, shared = neighborhoods(args.hoods)
        colors = color_graph(hoods, len(style["hood_palette"]))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for name, geom in hoods.items():
            canvas.fill(draw, geom, style["hood_palette"][colors[name]] + (style["hood_alpha"],))
        for name, geom in hoods.items():
            canvas.holes(draw, geom, (0, 0, 0, 0))
        for geom in shared:
            canvas.fill(draw, geom, style["shared_fill"])
        base = Image.alpha_composite(base, layer)
        print(f"neighborhoods: {len(hoods)} named, {len(shared)} shared areas, {len(cities)} other cities")

    # Water goes over the fills so rivers read clearly through the neighborhoods.
    water_img = Image.fromarray((water * 255).clip(0, 255).astype(np.uint8), "L").resize((w, h), Image.Resampling.BILINEAR)
    water_img = water_img.point(lambda v: 255 if v >= 128 else 0).filter(ImageFilter.GaussianBlur(0.8))
    base = Image.composite(Image.new("RGBA", (w, h), style["water"] + (255,)), base, water_img)

    lines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(lines)
    if args.style == "light":
        for _, geom in cities:
            canvas.outline(draw, geom, style["city_edge"], max(1, args.scale))
        for geom in hoods.values():
            canvas.outline(draw, geom, style["hood_edge"], args.scale)
    day = BusDay.load(args.trips)
    for s in np.unique(day.trip_shape):
        draw.line(canvas.px(day.shape_xy(int(s))), fill=style["route"], width=max(1, args.scale), joint="curve")
    if style["route_glow"]:
        base = Image.alpha_composite(base, lines.filter(ImageFilter.GaussianBlur(3 * args.scale)))
    out_img = Image.alpha_composite(base, lines).convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out, optimize=False)
    print(f"Wrote {out} {out_img.width}x{out_img.height} (water {(water > 0.5).mean() * 100:.1f}% of cells)")


if __name__ == "__main__":
    main()
