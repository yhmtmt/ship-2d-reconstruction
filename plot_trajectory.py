#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_trajectory(recon: Dict[str, Any]):
    xs, ys, ts, speeds = [], [], [], []
    for f in recon.get("frames", []):
        pose = f.get("pose")
        t = f.get("timestamp")
        spd = f.get("speed_mps")
        if pose is not None:
            xs.append(pose.get("x"))
            ys.append(pose.get("y"))
            ts.append(t)
            speeds.append(spd)
    return xs, ys, ts, speeds


def extract_world_points(world: Optional[Dict[str, Any]]):
    if not world:
        return [], []
    xs, ys, labels = [], [], []
    for p in world.get("points", []):
        xs.append(p.get("x"))
        ys.append(p.get("y"))
        labels.append(str(p.get("id", "")))
    return (xs, ys, labels)


def main():
    ap = argparse.ArgumentParser(description="Plot 2D trajectory and speed from reconstruction JSON")
    ap.add_argument("--reconstruction", required=True, help="reconstruction.json from reconstruct_2d.py")
    ap.add_argument("--world-points", default=None, help="Optional world_points JSON for static markers")
    ap.add_argument("--title", default="Trajectory and Speed (2D)")
    ap.add_argument("--save", default=None, help="Optional path to save figure (png)")
    args = ap.parse_args()

    recon = load_json(args.reconstruction)
    world = load_json(args.world_points) if args.world_points else None

    xs, ys, ts, speeds = extract_trajectory(recon)
    wxs, wys, wlabels = extract_world_points(world)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectory plot
    ax1.plot(xs, ys, '-', linewidth=2.0, label='trajectory')
    if wxs:
        ax1.scatter(wxs, wys, c='red', marker='x', label='static points')
        for x, y, lab in zip(wxs, wys, wlabels):
            ax1.text(x, y, f" {lab}", color='red')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('2D Sea-Surface Trajectory')
    ax1.grid(True)
    ax1.legend()

    # Speed plot
    ax2.plot(ts, speeds, '-b')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('speed (m/s)')
    ax2.set_title('Speed vs Time')
    ax2.grid(True)

    fig.suptitle(args.title)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
