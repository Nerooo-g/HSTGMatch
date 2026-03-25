"""
Preprocessing utilities for HSTGMatch.

Responsibilities:
  - GridMapper: maps (lon, lat) to grid cell IDs using a uniform 100×100m grid.
  - Preprocessor: Z-score normalises coordinates, computes distance/time
    intervals, and builds the ATA-Graph from trajectory statistics.
"""

import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0  # metres


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance between two (lon, lat) points in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def meters_per_degree_lat(lat: float) -> float:
    """Approximate metres per degree latitude at given latitude."""
    return EARTH_RADIUS_M * math.pi / 180.0


def meters_per_degree_lon(lat: float) -> float:
    """Approximate metres per degree longitude at given latitude."""
    return EARTH_RADIUS_M * math.cos(math.radians(lat)) * math.pi / 180.0


# ---------------------------------------------------------------------------
# Grid Mapper
# ---------------------------------------------------------------------------

class GridMapper:
    """
    Maps (lon, lat) coordinates to integer grid cell IDs.

    The map area is divided into cells of approximately `cell_size_m` metres
    in both directions.  Grid IDs are assigned in row-major order:
        grid_id = row * n_cols + col
    where row increases northward and col increases eastward.

    Args:
        min_lon, max_lon: bounding-box longitude extent.
        min_lat, max_lat: bounding-box latitude extent.
        cell_size_m: target cell size in metres (default 100 m).
    """

    def __init__(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        cell_size_m: float = 100.0,
    ) -> None:
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.cell_size_m = cell_size_m

        # Metres per degree at centre of bounding box
        centre_lat = (min_lat + max_lat) / 2.0
        self.m_per_deg_lat = meters_per_degree_lat(centre_lat)
        self.m_per_deg_lon = meters_per_degree_lon(centre_lat)

        self.deg_per_cell_lat = cell_size_m / self.m_per_deg_lat
        self.deg_per_cell_lon = cell_size_m / self.m_per_deg_lon

        lat_extent = max_lat - min_lat
        lon_extent = max_lon - min_lon

        self.n_rows = max(1, math.ceil(lat_extent / self.deg_per_cell_lat))
        self.n_cols = max(1, math.ceil(lon_extent / self.deg_per_cell_lon))
        self.n_grids = self.n_rows * self.n_cols

    def lonlat_to_grid(self, lon: float, lat: float) -> int:
        """Return the grid cell ID for the given (lon, lat) point."""
        col = int((lon - self.min_lon) / self.deg_per_cell_lon)
        row = int((lat - self.min_lat) / self.deg_per_cell_lat)
        col = max(0, min(self.n_cols - 1, col))
        row = max(0, min(self.n_rows - 1, row))
        return row * self.n_cols + col

    def grid_to_centre(self, grid_id: int) -> Tuple[float, float]:
        """Return the (lon, lat) centre of the given grid cell."""
        row = grid_id // self.n_cols
        col = grid_id % self.n_cols
        lon = self.min_lon + (col + 0.5) * self.deg_per_cell_lon
        lat = self.min_lat + (row + 0.5) * self.deg_per_cell_lat
        return lon, lat

    def all_grid_centres(self) -> np.ndarray:
        """Return (n_grids, 2) array of (lon, lat) grid centres."""
        centres = np.array(
            [self.grid_to_centre(gid) for gid in range(self.n_grids)], dtype=np.float32
        )
        return centres

    def to_dict(self) -> Dict:
        return {
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "cell_size_m": self.cell_size_m,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_grids": self.n_grids,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GridMapper":
        mapper = cls(
            min_lon=d["min_lon"],
            max_lon=d["max_lon"],
            min_lat=d["min_lat"],
            max_lat=d["max_lat"],
            cell_size_m=d["cell_size_m"],
        )
        return mapper


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Full preprocessing pipeline for HSTGMatch.

    Usage:
        preprocessor = Preprocessor.from_files(trajectories_path, road_network_path)
        preprocessor.fit()  # compute grid mapper + normalization stats
        processed = preprocessor.process_trajectories()
    """

    def __init__(
        self,
        trajectories: List[Dict],
        road_network: Dict,
        cell_size_m: float = 100.0,
        distance_threshold_m: float = 200.0,
    ) -> None:
        self.trajectories = trajectories
        self.road_network = road_network
        self.cell_size_m = cell_size_m
        self.distance_threshold_m = distance_threshold_m

        self.grid_mapper: Optional[GridMapper] = None

        # Z-score stats (computed over all trajectory points)
        self.lon_mean: float = 0.0
        self.lon_std: float = 1.0
        self.lat_mean: float = 0.0
        self.lat_std: float = 1.0

        # ATA-Graph statistics
        self.grid_point_counts: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Static loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        trajectories_path: str,
        road_network_path: str,
        cell_size_m: float = 100.0,
        distance_threshold_m: float = 200.0,
    ) -> "Preprocessor":
        with open(trajectories_path, "r") as f:
            trajectories = json.load(f)
        with open(road_network_path, "r") as f:
            road_network = json.load(f)
        return cls(trajectories, road_network, cell_size_m, distance_threshold_m)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> "Preprocessor":
        """Compute bounding box, grid mapper, normalization stats, and grid counts."""
        all_lons, all_lats = [], []
        for traj in self.trajectories:
            for pt in traj["points"]:
                all_lons.append(pt["lon"])
                all_lats.append(pt["lat"])

        # Add road segment nodes to bounding box
        for seg in self.road_network.get("segments", []):
            for pt in seg.get("geometry", []):
                all_lons.append(pt["lon"])
                all_lats.append(pt["lat"])

        lons = np.array(all_lons, dtype=np.float64)
        lats = np.array(all_lats, dtype=np.float64)

        # Bounding box with small padding
        padding = 0.001  # ~100m in degrees
        self.grid_mapper = GridMapper(
            min_lon=float(lons.min()) - padding,
            max_lon=float(lons.max()) + padding,
            min_lat=float(lats.min()) - padding,
            max_lat=float(lats.max()) + padding,
            cell_size_m=self.cell_size_m,
        )

        # Z-score statistics
        self.lon_mean = float(lons.mean())
        self.lon_std = float(lons.std()) or 1.0
        self.lat_mean = float(lats.mean())
        self.lat_std = float(lats.std()) or 1.0

        # Grid point counts for ATA-Graph edge weights
        counts = np.zeros(self.grid_mapper.n_grids, dtype=np.int64)
        for traj in self.trajectories:
            for pt in traj["points"]:
                gid = self.grid_mapper.lonlat_to_grid(pt["lon"], pt["lat"])
                counts[gid] += 1
        self.grid_point_counts = counts

        return self

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def normalize_lon(self, lon: float) -> float:
        return (lon - self.lon_mean) / self.lon_std

    def normalize_lat(self, lat: float) -> float:
        return (lat - self.lat_mean) / self.lat_std

    # ------------------------------------------------------------------
    # Process a single trajectory
    # ------------------------------------------------------------------

    def process_trajectory(self, traj: Dict) -> Dict:
        """
        Process one trajectory into the format expected by the dataset.

        Returns a dict with:
            traj_id:   int
            grid_ids:  List[int]       — grid cell IDs
            coords:    List[Tuple[float,float]]  — normalised (lon, lat)
            distances: List[float]     — Haversine distance from p_0 (metres)
            times:     List[float]     — time delta from p_0 (seconds)
            positions: List[int]       — 1-based position indices
            raw_lons:  List[float]     — original longitudes
            raw_lats:  List[float]     — original latitudes
            timestamps: List[int]      — original timestamps
        """
        points = traj["points"]
        n = len(points)
        p0 = points[0]

        grid_ids = []
        coords = []
        distances = []
        times = []
        positions = []
        raw_lons = []
        raw_lats = []
        timestamps = []

        for i, pt in enumerate(points):
            lon, lat, ts = pt["lon"], pt["lat"], pt["timestamp"]

            gid = self.grid_mapper.lonlat_to_grid(lon, lat)
            grid_ids.append(gid)
            coords.append((self.normalize_lon(lon), self.normalize_lat(lat)))

            dist = haversine(p0["lon"], p0["lat"], lon, lat)
            distances.append(dist)
            times.append(float(abs(ts - p0["timestamp"])))
            positions.append(i + 1)  # 1-based

            raw_lons.append(lon)
            raw_lats.append(lat)
            timestamps.append(ts)

        return {
            "traj_id": traj["traj_id"],
            "grid_ids": grid_ids,
            "coords": coords,
            "distances": distances,
            "times": times,
            "positions": positions,
            "raw_lons": raw_lons,
            "raw_lats": raw_lats,
            "timestamps": timestamps,
        }

    def process_trajectories(self) -> List[Dict]:
        """Process all trajectories."""
        return [self.process_trajectory(t) for t in self.trajectories]

    # ------------------------------------------------------------------
    # Road network helpers
    # ------------------------------------------------------------------

    def get_segment_midpoints(self) -> np.ndarray:
        """
        Return (N_seg, 2) array of (lon, lat) midpoints for each segment,
        indexed by seg_id.
        """
        segments = self.road_network["segments"]
        n = len(segments)
        midpoints = np.zeros((n, 2), dtype=np.float32)
        for seg in segments:
            sid = seg["seg_id"]
            g = seg["geometry"]
            mid_idx = len(g) // 2
            midpoints[sid, 0] = g[mid_idx]["lon"]
            midpoints[sid, 1] = g[mid_idx]["lat"]
        return midpoints

    def get_n_segments(self) -> int:
        return len(self.road_network["segments"])

    # ------------------------------------------------------------------
    # Save / load state
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "grid_mapper": self.grid_mapper.to_dict(),
                    "lon_mean": self.lon_mean,
                    "lon_std": self.lon_std,
                    "lat_mean": self.lat_mean,
                    "lat_std": self.lat_std,
                    "grid_point_counts": self.grid_point_counts,
                    "cell_size_m": self.cell_size_m,
                    "distance_threshold_m": self.distance_threshold_m,
                },
                f,
            )

    @classmethod
    def load_state(cls, path: str, trajectories: List[Dict], road_network: Dict) -> "Preprocessor":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            trajectories=trajectories,
            road_network=road_network,
            cell_size_m=state["cell_size_m"],
            distance_threshold_m=state["distance_threshold_m"],
        )
        obj.grid_mapper = GridMapper.from_dict(state["grid_mapper"])
        obj.lon_mean = state["lon_mean"]
        obj.lon_std = state["lon_std"]
        obj.lat_mean = state["lat_mean"]
        obj.lat_std = state["lat_std"]
        obj.grid_point_counts = state["grid_point_counts"]
        return obj
