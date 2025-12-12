#!/usr/bin/env python3
import argparse
import json
import math
from typing import List, Dict


def meters_per_deg(lat_deg: float) -> (float, float):
    # Approximation suitable for small areas
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2*lat_rad) + 1.175 * math.cos(4*lat_rad) - 0.0023 * math.cos(6*lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3*lat_rad) + 0.118 * math.cos(5*lat_rad)
    return m_per_deg_lat, m_per_deg_lon


def parse_points_txt(path: str) -> List[Dict[str, float]]:
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return pts
    header = lines[0].lower().split()
    for ln in lines[1:]:
        # Expected formats like:
        # name lat lon alt
        # or: name lat, lon alt
        parts = ln.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        # Merge potential comma in lat/lon
        # e.g., "35.08295," "138.85703" "0"
        # or "35.08295," "138.85703" "0.0"
        # or "35.08295," "138.85703" "7.5"
        # If second token contains a comma, split on comma
        # But in the given file, it's "lat," and then lon as next token.
        lat_token = parts[1]
        if lat_token.endswith(','):
            lat_str = lat_token[:-1]
            lon_str = parts[2]
            alt_str = parts[3] if len(parts) > 3 else '0'
        else:
            # Could be lat lon alt without comma
            lat_str = parts[1]
            lon_str = parts[2] if len(parts) > 2 else '0'
            alt_str = parts[3] if len(parts) > 3 else '0'
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            alt = float(alt_str)
        except ValueError:
            # Try removing trailing commas in lon as well
            lon_str2 = lon_str.rstrip(',')
            try:
                lat = float(lat_str)
                lon = float(lon_str2)
                alt = float(alt_str)
            except Exception:
                continue
        pts.append({"name": name, "lat": lat, "lon": lon, "alt": alt})
    return pts


def convert_to_world(points: List[Dict[str, float]]) -> Dict:
    # Find camera as origin for x,y; z=alt with sea surface at z=0.
    cam = None
    for p in points:
        if p["name"].lower().startswith("camera"):
            cam = p
            break
    if cam is None:
        raise RuntimeError("No camera point found (name starting with 'camera').")
    lat0, lon0, z_cam = cam["lat"], cam["lon"], cam["alt"]
    m_per_deg_lat, m_per_deg_lon = meters_per_deg(lat0)

    out_pts = []
    for p in points:
        if p["name"].lower().startswith("camera"):
            # camera used only to define origin; omit from output
            continue
        dlat = p["lat"] - lat0
        dlon = p["lon"] - lon0
        x = dlon * m_per_deg_lon  # east as +x
        y = dlat * m_per_deg_lat  # north as +y
        z = p["alt"]  # assume given alt is relative to sea surface (z=0)
        out_pts.append({
            "id": p["name"],
            "x": x,
            "y": y,
            "z": z,
            "name": p["name"]
        })

    return {
        "coordinate_system": {
            "description": "World: x east, y north, z up; z=0 sea; x=y=0 at camera"
        },
        "points": out_pts
    }


def main():
    ap = argparse.ArgumentParser(description="Convert lat/lon/alt points to world_points.json schema.")
    ap.add_argument("--input", required=True, help="Text file with lines: name lat, lon alt (camera row required)")
    ap.add_argument("--output", required=True, help="Output JSON path (world_points schema)")
    args = ap.parse_args()

    pts = parse_points_txt(args.input)
    world = convert_to_world(pts)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(world, f, indent=2)
    print(f"Wrote {args.output} with {len(world['points'])} points")


if __name__ == "__main__":
    main()
