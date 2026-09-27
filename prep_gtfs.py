"""Turn the TriMet GTFS feed into a compact per-trip trajectory file for one service day.

Each trip gets time -> distance keyframes (arrival and departure at every stop, so
buses dwell at stops) and each shape gets a UTM 10N polyline with distance along it.
At render time a bus position is two np.interp calls: time -> distance -> x/y.
"""
import argparse
import datetime as dt
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

CRS = 32610  # WGS 84 / UTM zone 10N, same as the DEM


def active_service_ids(z: zipfile.ZipFile, day: dt.date) -> set[str]:
    key = int(day.strftime("%Y%m%d"))
    weekday = day.strftime("%A").lower()
    cal = pd.read_csv(z.open("calendar.txt"), dtype={"service_id": str})
    active = set(cal.loc[(cal.start_date <= key) & (cal.end_date >= key) & (cal[weekday] == 1), "service_id"])
    dates = pd.read_csv(z.open("calendar_dates.txt"), dtype={"service_id": str})
    dates = dates[dates.date == key]
    active |= set(dates.loc[dates.exception_type == 1, "service_id"])
    active -= set(dates.loc[dates.exception_type == 2, "service_id"])
    return active


def gtfs_seconds(col: pd.Series) -> np.ndarray:
    hms = col.str.split(":", expand=True).astype(np.int32)
    return (hms[0] * 3600 + hms[1] * 60 + hms[2]).to_numpy(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", type=Path, default=Path("gtfs.zip"))
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.date(2026, 9, 29))
    parser.add_argument("--route-types", default="3", help="Comma-separated GTFS route_type values (3 = bus).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    route_types = {int(v) for v in args.route_types.split(",")}
    out = args.out or Path(f"data/trips_{args.date.isoformat()}.npz")
    t0 = time.perf_counter()

    z = zipfile.ZipFile(args.gtfs)
    services = active_service_ids(z, args.date)
    routes = pd.read_csv(z.open("routes.txt"), dtype=str)
    routes = routes[routes.route_type.astype(int).isin(route_types)].reset_index(drop=True)
    trips = pd.read_csv(z.open("trips.txt"), dtype=str)
    trips = trips[trips.service_id.isin(services) & trips.route_id.isin(routes.route_id)].reset_index(drop=True)
    print(f"{args.date} ({args.date:%A}): {len(services)} services, {len(routes)} routes, {len(trips)} trips", flush=True)

    st = pd.read_csv(
        z.open("stop_times.txt"),
        usecols=["trip_id", "arrival_time", "departure_time", "stop_sequence", "shape_dist_traveled"],
        dtype={"trip_id": str, "arrival_time": str, "departure_time": str},
    )
    st = st[st.trip_id.isin(trips.trip_id)].copy()
    st["arr"] = gtfs_seconds(st.arrival_time)
    st["dep"] = gtfs_seconds(st.departure_time)
    st = st.sort_values(["trip_id", "stop_sequence"])
    print(f"stop_times rows for the day: {len(st):,}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    shape_ids = trips.shape_id.unique()
    sh = pd.read_csv(z.open("shapes.txt"), dtype={"shape_id": str})
    sh = sh[sh.shape_id.isin(shape_ids)].sort_values(["shape_id", "shape_pt_sequence"])
    to_utm = Transformer.from_crs(4326, CRS, always_xy=True)
    sx, sy = to_utm.transform(sh.shape_pt_lon.to_numpy(), sh.shape_pt_lat.to_numpy())
    sh["x"], sh["y"] = sx, sy

    shape_index: dict[str, int] = {}
    shape_offsets = [0]
    shape_x, shape_y, shape_d = [], [], []
    for sid, g in sh.groupby("shape_id", sort=False):
        d = g.shape_dist_traveled.to_numpy(np.float64)
        keep = np.concatenate([[True], np.diff(d) > 0])  # np.interp needs increasing distance
        shape_index[sid] = len(shape_index)
        shape_x.append(g.x.to_numpy()[keep])
        shape_y.append(g.y.to_numpy()[keep])
        shape_d.append(d[keep])
        shape_offsets.append(shape_offsets[-1] + int(keep.sum()))

    trip_meta = trips.set_index("trip_id")
    route_index = {rid: i for i, rid in enumerate(routes.route_id)}
    kf_offsets = [0]
    kf_t, kf_d = [], []
    t_start, t_end, t_route, t_shape, t_block = [], [], [], [], []
    for trip_id, g in st.groupby("trip_id", sort=False):
        meta = trip_meta.loc[trip_id]
        if meta.shape_id not in shape_index or len(g) < 2:
            continue
        arr, dep, dist = g.arr.to_numpy(), g.dep.to_numpy(), g.shape_dist_traveled.to_numpy(np.float64)
        # Interleave arrival/departure so the bus holds still while dwelling at a stop.
        times = np.column_stack([arr, dep]).ravel()
        dists = np.repeat(dist, 2)
        keep = np.concatenate([[True], np.diff(times) > 0])
        times, dists = times[keep], dists[keep]
        if len(times) < 2 or np.any(np.isnan(dists)):
            continue
        kf_t.append(times.astype(np.int32))
        kf_d.append(dists)
        kf_offsets.append(kf_offsets[-1] + len(times))
        t_start.append(times[0])
        t_end.append(times[-1])
        t_route.append(route_index[meta.route_id])
        t_shape.append(shape_index[meta.shape_id])
        t_block.append(int(meta.block_id) if str(meta.block_id).isdigit() else -1)

    t_start = np.asarray(t_start, np.int32)
    t_end = np.asarray(t_end, np.int32)
    minutes = np.arange(0, 28 * 3600, 60)
    active = ((t_start[None, :] <= minutes[:, None]) & (t_end[None, :] >= minutes[:, None])).sum(1)
    peak = int(active.argmax())

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        service_date=args.date.isoformat(),
        trip_start=t_start,
        trip_end=t_end,
        trip_route=np.asarray(t_route, np.int32),
        trip_shape=np.asarray(t_shape, np.int32),
        trip_block=np.asarray(t_block, np.int64),
        kf_offsets=np.asarray(kf_offsets, np.int64),
        kf_t=np.concatenate(kf_t),
        kf_d=np.concatenate(kf_d),
        shape_offsets=np.asarray(shape_offsets, np.int64),
        shape_x=np.concatenate(shape_x),
        shape_y=np.concatenate(shape_y),
        shape_d=np.concatenate(shape_d),
        route_id=routes.route_id.to_numpy(str),
        route_short_name=routes.route_short_name.fillna("").to_numpy(str),
        route_long_name=routes.route_long_name.fillna("").to_numpy(str),
        route_color=routes.route_color.fillna("FFFFFF").to_numpy(str),
        active_per_minute=active.astype(np.int32),
    )
    print(f"Wrote {out}: {len(t_start):,} trips, {len(shape_index)} shapes, "
          f"service {t_start.min() / 3600:.2f}h..{t_end.max() / 3600:.2f}h, "
          f"peak {active.max()} buses at {peak // 60:02d}:{peak % 60:02d}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
