"""Download Portland neighborhood and metro-area city boundaries (City of Portland open data), in UTM 10N."""
import json
import urllib.parse
import urllib.request
from pathlib import Path

SERVICE = "https://www.portlandmaps.com/od/rest/services/COP_OpenData_Boundary/MapServer"
LAYERS = {
    "neighborhoods": 3,   # Neighborhood Boundaries: NAME, MAPLABEL, COALIT, SHARED
    "cities": 10,         # City Boundaries: CITYNAME (57 cities in the region)
    "sextants": 233,      # Portland Administrative Sextants: N, NE, NW, S, SE, SW
}
CRS = 32610


def fetch(layer: int) -> dict:
    features, offset = [], 0
    while True:
        params = {"where": "1=1", "outFields": "*", "outSR": CRS, "f": "geojson",
                  "resultOffset": offset, "resultRecordCount": 500}
        url = f"{SERVICE}/{layer}/query?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as resp:
            page = json.load(resp)
        features += page["features"]
        if not page.get("properties", {}).get("exceededTransferLimit") and len(page["features"]) < 500:
            return {"type": "FeatureCollection", "features": features}
        offset += len(page["features"])


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    for name, layer in LAYERS.items():
        fc = fetch(layer)
        path = out_dir / f"{name}_utm10n.geojson"
        path.write_text(json.dumps(fc))
        print(f"Wrote {path}: {len(fc['features'])} features")


if __name__ == "__main__":
    main()
