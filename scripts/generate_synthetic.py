"""
Synthetic data generator for HSTGMatch.

Generates a small Beijing-like map (2km × 2km) with:
  - A regular grid road network (~20×20 blocks, ~200 segments)
  - GPS trajectories following roads with Gaussian noise
  - Ground-truth label routes

Output: data/synthetic/{trajectories.json, road_network.json, labels.json}
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Beijing city centre approx coordinates
CENTRE_LON = 116.3900
CENTRE_LAT = 39.9100
EARTH_RADIUS_M = 6_371_000.0


def metres_to_deg_lon(m: float, lat: float) -> float:
    return m / (EARTH_RADIUS_M * math.cos(math.radians(lat)) * math.pi / 180.0)


def metres_to_deg_lat(m: float) -> float:
    return m / (EARTH_RADIUS_M * math.pi / 180.0)


def haversine(lon1, lat1, lon2, lat2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(max(0.0, min(1.0, a))))


# ---------------------------------------------------------------------------
# Road network generation
# ---------------------------------------------------------------------------

def build_road_network(
    centre_lon: float,
    centre_lat: float,
    area_m: float = 2000.0,
    block_size_m: float = 200.0,
    seed: int = 42,
) -> Tuple[Dict, List[Tuple[float, float]]]:
    """
    Build a regular grid road network.

    Returns:
        road_network: dict with "segments" list.
        nodes: list of (lon, lat) node positions.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    half = area_m / 2.0
    n_blocks = int(area_m / block_size_m)  # e.g. 10

    # Generate grid node positions
    deg_per_block_lon = metres_to_deg_lon(block_size_m, centre_lat)
    deg_per_block_lat = metres_to_deg_lat(block_size_m)

    min_lon = centre_lon - metres_to_deg_lon(half, centre_lat)
    min_lat = centre_lat - metres_to_deg_lat(half)

    n_nodes_per_side = n_blocks + 1  # e.g. 11

    # Node positions: (row, col) → (lon, lat)
    def node_pos(row: int, col: int) -> Tuple[float, float]:
        lon = min_lon + col * deg_per_block_lon
        lat = min_lat + row * deg_per_block_lat
        return lon, lat

    def node_id(row: int, col: int) -> int:
        return row * n_nodes_per_side + col

    nodes = []
    for r in range(n_nodes_per_side):
        for c in range(n_nodes_per_side):
            nodes.append(node_pos(r, c))

    segments = []
    seg_id = 0

    # Horizontal segments (west to east)
    for r in range(n_nodes_per_side):
        for c in range(n_blocks):
            lon_s, lat_s = node_pos(r, c)
            lon_e, lat_e = node_pos(r, c + 1)
            # Interpolate geometry with 3 shape points
            geometry = [
                {"lon": lon_s, "lat": lat_s},
                {"lon": (lon_s + lon_e) / 2, "lat": (lat_s + lat_e) / 2},
                {"lon": lon_e, "lat": lat_e},
            ]
            segments.append({
                "seg_id": seg_id,
                "start_node": {"lon": lon_s, "lat": lat_s},
                "end_node": {"lon": lon_e, "lat": lat_e},
                "geometry": geometry,
            })
            seg_id += 1

    # Vertical segments (south to north)
    for c in range(n_nodes_per_side):
        for r in range(n_blocks):
            lon_s, lat_s = node_pos(r, c)
            lon_e, lat_e = node_pos(r + 1, c)
            geometry = [
                {"lon": lon_s, "lat": lat_s},
                {"lon": (lon_s + lon_e) / 2, "lat": (lat_s + lat_e) / 2},
                {"lon": lon_e, "lat": lat_e},
            ]
            segments.append({
                "seg_id": seg_id,
                "start_node": {"lon": lon_s, "lat": lat_s},
                "end_node": {"lon": lon_e, "lat": lat_e},
                "geometry": geometry,
            })
            seg_id += 1

    road_network = {"segments": segments}
    return road_network, nodes, n_nodes_per_side, n_blocks, min_lon, min_lat, deg_per_block_lon, deg_per_block_lat


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def sample_trajectory(
    road_network: Dict,
    n_nodes_per_side: int,
    n_blocks: int,
    min_lon: float,
    min_lat: float,
    deg_per_block_lon: float,
    deg_per_block_lat: float,
    rng: random.Random,
    np_rng: np.random.RandomState,
    min_steps: int = 5,
    max_steps: int = 20,
    gps_noise_m: float = 5.0,
    speed_mps: float = 8.0,  # ~30 km/h
    start_timestamp: int = 1609459200,
) -> Optional[Tuple[List[Dict], List[int]]]:
    """
    Sample a random walk on the grid network.

    Returns:
        (gps_points, seg_ids) or None if the walk is too short.
    """
    segments = road_network["segments"]
    seg_id_counter = len(segments)

    # Build adjacency: node -> list of (neighbour_node, seg_id)
    # Nodes are (row * n_nodes_per_side + col)
    adjacency: Dict[int, List[Tuple[int, int]]] = {}
    for seg in segments:
        # Find start and end node index from lon/lat
        lon_s = seg["start_node"]["lon"]
        lat_s = seg["start_node"]["lat"]
        lon_e = seg["end_node"]["lon"]
        lat_e = seg["end_node"]["lat"]

        def lonlat_to_node(lon, lat):
            col = round((lon - min_lon) / deg_per_block_lon)
            row = round((lat - min_lat) / deg_per_block_lat)
            col = max(0, min(n_nodes_per_side - 1, col))
            row = max(0, min(n_nodes_per_side - 1, row))
            return row * n_nodes_per_side + col

        ns = lonlat_to_node(lon_s, lat_s)
        ne = lonlat_to_node(lon_e, lat_e)
        adjacency.setdefault(ns, []).append((ne, seg["seg_id"]))
        # Add reverse direction too (undirected)
        adjacency.setdefault(ne, []).append((ns, seg["seg_id"]))

    # Pick random start node
    start_node = rng.randint(0, n_nodes_per_side * n_nodes_per_side - 1)
    current_node = start_node
    n_steps = rng.randint(min_steps, max_steps)

    gps_points = []
    seg_route = []
    timestamp = start_timestamp

    # Convert node ID to (lon, lat)
    def node_to_lonlat(node_id: int) -> Tuple[float, float]:
        row = node_id // n_nodes_per_side
        col = node_id % n_nodes_per_side
        lon = min_lon + col * deg_per_block_lon
        lat = min_lat + row * deg_per_block_lat
        return lon, lat

    lon0, lat0 = node_to_lonlat(current_node)

    # Noise in degrees
    noise_lon_deg = gps_noise_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0)) * math.pi / 180.0)
    noise_lat_deg = gps_noise_m / (EARTH_RADIUS_M * math.pi / 180.0)

    for step in range(n_steps + 1):
        lon_true, lat_true = node_to_lonlat(current_node)
        # Add Gaussian noise
        lon_obs = lon_true + np_rng.normal(0, noise_lon_deg)
        lat_obs = lat_true + np_rng.normal(0, noise_lat_deg)

        gps_points.append({"lon": round(lon_obs, 7), "lat": round(lat_obs, 7), "timestamp": int(timestamp)})

        if step < n_steps:
            neighbours = adjacency.get(current_node, [])
            if not neighbours:
                break
            next_node, seg_id = rng.choice(neighbours)

            # Travel time along segment
            d = haversine(lon_true, lat_true, *node_to_lonlat(next_node))
            dt = max(5, d / speed_mps)
            timestamp += dt

            if not seg_route or seg_route[-1] != seg_id:
                seg_route.append(seg_id)
            current_node = next_node

    if len(gps_points) < 2:
        return None

    return gps_points, seg_route


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(
    output_dir: str,
    n_trajs: int = 500,
    area_m: float = 2000.0,
    block_size_m: float = 200.0,
    gps_noise_m: float = 5.0,
    seed: int = 42,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    print(f"Building road network ({area_m}m × {area_m}m, block={block_size_m}m)...")
    (road_network,
     nodes,
     n_nodes_per_side,
     n_blocks,
     min_lon,
     min_lat,
     deg_per_block_lon,
     deg_per_block_lat) = build_road_network(
        CENTRE_LON, CENTRE_LAT,
        area_m=area_m,
        block_size_m=block_size_m,
        seed=seed,
    )
    print(f"  Segments: {len(road_network['segments'])}")
    print(f"  Nodes per side: {n_nodes_per_side}")

    # Save road network
    road_path = os.path.join(output_dir, "road_network.json")
    with open(road_path, "w") as f:
        json.dump(road_network, f)
    print(f"  Saved: {road_path}")

    # Generate trajectories
    print(f"Generating {n_trajs} trajectories...")
    trajectories = []
    labels = []
    traj_id = 0
    attempts = 0

    start_ts = 1609459200  # 2021-01-01 00:00:00 UTC
    while traj_id < n_trajs and attempts < n_trajs * 10:
        attempts += 1
        result = sample_trajectory(
            road_network=road_network,
            n_nodes_per_side=n_nodes_per_side,
            n_blocks=n_blocks,
            min_lon=min_lon,
            min_lat=min_lat,
            deg_per_block_lon=deg_per_block_lon,
            deg_per_block_lat=deg_per_block_lat,
            rng=rng,
            np_rng=np_rng,
            min_steps=5,
            max_steps=20,
            gps_noise_m=gps_noise_m,
            start_timestamp=start_ts + rng.randint(0, 86400),
        )
        if result is None:
            continue

        gps_points, seg_route = result
        if len(seg_route) == 0:
            continue

        trajectories.append({"traj_id": traj_id, "points": gps_points})
        labels.append({"traj_id": traj_id, "route": seg_route})
        traj_id += 1

        if traj_id % 100 == 0:
            print(f"  Generated {traj_id}/{n_trajs} trajectories...")

    print(f"  Generated {len(trajectories)} trajectories ({attempts} attempts).")

    # Save trajectories
    traj_path = os.path.join(output_dir, "trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(trajectories, f)
    print(f"  Saved: {traj_path}")

    # Save labels
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f)
    print(f"  Saved: {labels_path}")

    # Print summary statistics
    seg_lengths = [len(l["route"]) for l in labels]
    traj_lengths = [len(t["points"]) for t in trajectories]
    print("\nSummary:")
    print(f"  Road segments:      {len(road_network['segments'])}")
    print(f"  Trajectories:       {len(trajectories)}")
    print(f"  GPS pts per traj:   min={min(traj_lengths)}  max={max(traj_lengths)}  avg={sum(traj_lengths)/len(traj_lengths):.1f}")
    print(f"  Route segs per traj: min={min(seg_lengths)}  max={max(seg_lengths)}  avg={sum(seg_lengths)/len(seg_lengths):.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic HSTGMatch data")
    parser.add_argument("--output_dir", type=str, default="data/synthetic",
                        help="Directory to write output files")
    parser.add_argument("--n_trajs", type=int, default=500,
                        help="Number of trajectories to generate")
    parser.add_argument("--area_m", type=float, default=2000.0,
                        help="Side length of the map area in metres")
    parser.add_argument("--block_size_m", type=float, default=200.0,
                        help="Road block size in metres")
    parser.add_argument("--gps_noise_m", type=float, default=5.0,
                        help="Standard deviation of GPS noise in metres")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    generate(
        output_dir=args.output_dir,
        n_trajs=args.n_trajs,
        area_m=args.area_m,
        block_size_m=args.block_size_m,
        gps_noise_m=args.gps_noise_m,
        seed=args.seed,
    )
