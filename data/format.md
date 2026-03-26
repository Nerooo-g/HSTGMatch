# HSTGMatch Data Format Documentation

## Overview

HSTGMatch requires three input files placed in a data directory (e.g., `data/synthetic/`):

1. `trajectories.json` — GPS trajectory sequences
2. `road_network.json` — Road segment definitions
3. `labels.json` — Ground truth map-matching results

---

## 1. trajectories.json

A JSON array of trajectory objects. Each trajectory is a time-ordered sequence of GPS observations.

```json
[
  {
    "traj_id": 0,
    "points": [
      {"lon": 116.3912, "lat": 39.9073, "timestamp": 1609459200},
      {"lon": 116.3925, "lat": 39.9081, "timestamp": 1609459215},
      {"lon": 116.3940, "lat": 39.9090, "timestamp": 1609459232}
    ]
  },
  {
    "traj_id": 1,
    "points": [
      ...
    ]
  }
]
```

### Fields
| Field | Type | Description |
|-------|------|-------------|
| `traj_id` | int | Unique trajectory identifier |
| `points` | list | Ordered list of GPS observations |
| `points[].lon` | float | Longitude (decimal degrees, WGS-84) |
| `points[].lat` | float | Latitude (decimal degrees, WGS-84) |
| `points[].timestamp` | int | Unix timestamp (seconds since epoch) |

### Constraints
- Points must be ordered by ascending timestamp
- Minimum trajectory length: 2 points
- Longitude range: any valid WGS-84 longitude
- Latitude range: any valid WGS-84 latitude

---

## 2. road_network.json

A JSON object containing a list of road segments.

```json
{
  "segments": [
    {
      "seg_id": 0,
      "start_node": {"lon": 116.391, "lat": 39.907},
      "end_node":   {"lon": 116.393, "lat": 39.908},
      "geometry": [
        {"lon": 116.391, "lat": 39.907},
        {"lon": 116.392, "lat": 39.9075},
        {"lon": 116.393, "lat": 39.908}
      ]
    },
    {
      "seg_id": 1,
      ...
    }
  ]
}
```

### Fields
| Field | Type | Description |
|-------|------|-------------|
| `segments` | list | All road segments in the network |
| `segments[].seg_id` | int | Unique segment identifier (0-indexed, contiguous) |
| `segments[].start_node` | object | Segment start coordinate |
| `segments[].end_node` | object | Segment end coordinate |
| `segments[].geometry` | list | Ordered list of shape points along the segment |

### Constraints
- `seg_id` values must be 0-indexed and contiguous (0, 1, 2, ..., N-1)
- `geometry` must include at least the start and end nodes
- Segments are treated as directed edges

---

## 3. labels.json

A JSON array of ground truth route assignments, one per trajectory.

```json
[
  {"traj_id": 0, "route": [42, 43, 78, 79, 80]},
  {"traj_id": 1, "route": [12, 13, 14]},
  ...
]
```

### Fields
| Field | Type | Description |
|-------|------|-------------|
| `traj_id` | int | Must match a `traj_id` in `trajectories.json` |
| `route` | list[int] | Ordered sequence of `seg_id` values from `road_network.json` |

### Constraints
- Every `traj_id` in `labels.json` must exist in `trajectories.json`
- Every `seg_id` in `route` must exist in `road_network.json`
- Routes may be shorter or longer than the GPS trajectory (they represent road segments traversed)

---

## Internal Representations

After preprocessing (see `src/data/preprocess.py`), the following derived representations are used:

### Grid System
- The bounding box of all GPS points is divided into a uniform grid of 100×100 m cells
- Each cell receives a unique integer `grid_id` (row-major order: `grid_id = row * n_cols + col`)
- GPS point `(lon, lat)` → grid cell `(row, col)` → `grid_id`

### Normalized Coordinates
- Longitude and latitude are Z-score normalized using the global mean and standard deviation computed over all trajectory points
- Stored as `(lon_norm, lat_norm)` floating-point pairs

### Spatial-Temporal Intervals
- **Distance interval**: Haversine distance from `p_0` (first point of trajectory) to `p_i`, in meters
- **Time interval**: Absolute difference `|timestamp_i - timestamp_0|`, in seconds
