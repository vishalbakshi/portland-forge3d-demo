"""Download a USGS 3DEP DEM covering the TriMet service area, in UTM 10N metres.

forge3d's viewer needs a projected DEM: with a lat/lon DEM the horizontal span is
in degrees while heights are in metres, and the terrain renders as a thin needle.
"""
import argparse
import json
import math
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.merge import merge

EXPORT_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
CRS = 32610  # WGS 84 / UTM zone 10N
TILE_PX = 2400

# TriMet stop extent (lon/lat) plus a margin so buses never run off the terrain edge.
LON_MIN, LON_MAX = -123.16, -122.29
LAT_MIN, LAT_MAX = 45.25, 45.67


def fetch_tile(bbox: tuple, size: tuple[int, int], path: Path) -> None:
    params = {
        "bbox": ",".join(str(v) for v in bbox),
        "bboxSR": CRS,
        "imageSR": CRS,
        "size": f"{size[0]},{size[1]}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        # f=image returns HTTP 500 for this service; f=json returns a link to the TIFF.
        "f": "json",
    }
    url = EXPORT_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                href = json.load(resp)["href"]
            with urllib.request.urlopen(href, timeout=300) as resp:
                path.write_bytes(resp.read())
            return
        except urllib.error.HTTPError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=float, default=10.0, help="Pixel size in metres.")
    parser.add_argument("--out", type=Path, default=Path("data/portland_dem10_utm10n.tif"))
    args = parser.parse_args()

    to_utm = Transformer.from_crs(4326, CRS, always_xy=True)
    xs, ys = to_utm.transform([LON_MIN, LON_MAX, LON_MIN, LON_MAX], [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = int(round((xmax - xmin) / args.res))
    height = int(round((ymax - ymin) / args.res))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Requesting {width}x{height} DEM at {args.res} m ...", flush=True)

    # Large single requests time out (HTTP 504), so fetch tiles of at most TILE_PX and merge.
    nx, ny = math.ceil(width / TILE_PX), math.ceil(height / TILE_PX)
    with tempfile.TemporaryDirectory() as tmp:
        tiles = []
        for j in range(ny):
            for i in range(nx):
                c0, c1 = i * width // nx, (i + 1) * width // nx
                r0, r1 = j * height // ny, (j + 1) * height // ny
                bbox = (xmin + c0 * args.res, ymax - r1 * args.res, xmin + c1 * args.res, ymax - r0 * args.res)
                path = Path(tmp) / f"tile_{j}_{i}.tif"
                fetch_tile(bbox, (c1 - c0, r1 - r0), path)
                tiles.append(path)
                print(f"  tile {len(tiles)}/{nx * ny}", flush=True)
        sources = [rasterio.open(p) for p in tiles]
        mosaic, transform = merge(sources, bounds=(xmin, ymin, xmin + width * args.res, ymax), res=args.res)
        profile = sources[0].profile.copy()
        for src in sources:
            src.close()
    profile.update(width=mosaic.shape[2], height=mosaic.shape[1], transform=transform, compress="deflate")
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(mosaic)

    with rasterio.open(args.out) as src:
        band = src.read(1, masked=True)
        print(f"Wrote {args.out}: {src.width}x{src.height} crs={src.crs} res={src.res} "
              f"elev {float(band.min()):.1f}..{float(band.max()):.1f} m")


if __name__ == "__main__":
    main()
