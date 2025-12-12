#!/usr/bin/env python3
import argparse
import json


def parse_ship_points(path: str):
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # Expect header: name x y z
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        pts.append({"id": name, "x": x, "y": y, "z": z, "name": name})
    return pts


def main():
    ap = argparse.ArgumentParser(description="Convert ship_points.txt to ship_points.json schema")
    ap.add_argument("--input", default="ship_points.txt", help="Input text file: name x y z")
    ap.add_argument("--output", default="ship_points.json", help="Output JSON path")
    args = ap.parse_args()

    pts = parse_ship_points(args.input)
    data = {
        "coordinate_system": {"description": "Ship: x starboard, y bow, z up; origin at sea surface center"},
        "points": pts
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.output} with {len(pts)} points")


if __name__ == "__main__":
    main()

