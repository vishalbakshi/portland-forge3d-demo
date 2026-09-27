"""Bus positions at any clock time, from the trajectories written by prep_gtfs.py."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BusDay:
    service_date: str
    trip_start: np.ndarray
    trip_end: np.ndarray
    trip_route: np.ndarray
    trip_shape: np.ndarray
    trip_block: np.ndarray
    kf_offsets: np.ndarray
    kf_t: np.ndarray
    kf_d: np.ndarray
    shape_offsets: np.ndarray
    shape_x: np.ndarray
    shape_y: np.ndarray
    shape_d: np.ndarray
    route_short_name: np.ndarray
    route_long_name: np.ndarray
    route_color: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "BusDay":
        z = np.load(path)
        return cls(**{name: z[name] for name in cls.__dataclass_fields__})

    def active(self, seconds: float) -> np.ndarray:
        """Indices of trips on the road at `seconds` after service-day midnight."""
        return np.nonzero((self.trip_start <= seconds) & (self.trip_end >= seconds))[0]

    def trip_xy(self, trip: int, seconds: float) -> tuple[float, float]:
        a, b = self.kf_offsets[trip], self.kf_offsets[trip + 1]
        d = np.interp(seconds, self.kf_t[a:b], self.kf_d[a:b])
        s = self.trip_shape[trip]
        p, q = self.shape_offsets[s], self.shape_offsets[s + 1]
        return (float(np.interp(d, self.shape_d[p:q], self.shape_x[p:q])),
                float(np.interp(d, self.shape_d[p:q], self.shape_y[p:q])))

    def positions(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """(trip indices, N x 2 UTM 10N easting/northing) for every bus on the road."""
        trips = self.active(seconds)
        xy = np.array([self.trip_xy(i, seconds) for i in trips], dtype=np.float64).reshape(-1, 2)
        return trips, xy

    def shape_xy(self, shape: int) -> np.ndarray:
        p, q = self.shape_offsets[shape], self.shape_offsets[shape + 1]
        return np.column_stack([self.shape_x[p:q], self.shape_y[p:q]])
