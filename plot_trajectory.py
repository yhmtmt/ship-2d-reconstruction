#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    # Prefer Savitzky-Golay if SciPy is available (listed in requirements.txt)
    from scipy.signal import savgol_filter  # type: ignore
    _HAS_SAVITZKY_GOLAY = True
except Exception:
    _HAS_SAVITZKY_GOLAY = False


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_trajectory(recon: Dict[str, Any]) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    ts: List[float] = []
    speeds: List[float] = []
    for f in recon.get("frames", []):
        pose = f.get("pose")
        t = f.get("timestamp")
        spd = f.get("speed_mps")
        # Only append when data is complete to avoid None in plots/smoothing
        if pose is not None and t is not None and spd is not None:
            xs.append(float(pose.get("x")))
            ys.append(float(pose.get("y")))
            ts.append(float(t))
            speeds.append(float(spd))
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
    # Optional vertical time markers on the speed plot
    ap.add_argument("--time-lines", default=None, help="Path to time-lines file (JSON list, or CSV-like)")
    ap.add_argument("--time-line-color", default="crimson", help="Color for time marker lines")
    ap.add_argument("--time-line-style", default="--", help="Line style for time markers (e.g., --, -., :)")
    ap.add_argument("--time-line-alpha", type=float, default=0.8, help="Alpha for time marker lines")
    # Smoothing options for speed plot
    ap.add_argument("--smooth-window", type=int, default=11, help="Smoothing window length (odd integer)")
    ap.add_argument(
        "--smooth-method",
        choices=["savgol", "moving"],
        default="savgol",
        help="Smoothing method: Savitzky-Golay (if available) or moving average",
    )
    ap.add_argument("--smooth-polyorder", type=int, default=2, help="Polyorder for Savitzky-Golay filter")
    ap.add_argument("--hide-raw-speed", action="store_true", help="Hide raw speed line, show smoothed only")
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

    # Speed plot (raw + smoothed)
    if speeds and ts:
        if not args.hide_raw_speed:
            ax2.plot(ts, speeds, color='gray', alpha=0.6, linewidth=1.2, label='raw speed')

        # Determine window (must be odd and <= len(speeds))
        n = len(speeds)
        window = max(3, min(args.smooth_window if args.smooth_window > 0 else 11, n - (1 - n % 2)))
        if window % 2 == 0:
            window = max(3, window - 1)

        smoothed = None
        if window >= 3:
            if args.smooth_method == "savgol" and _HAS_SAVITZKY_GOLAY and window > args.smooth_polyorder:
                try:
                    smoothed = savgol_filter(np.asarray(speeds, dtype=float), window_length=window, polyorder=args.smooth_polyorder)
                except Exception:
                    smoothed = None
            if smoothed is None:
                # Fallback: simple moving average using convolution, centered
                k = np.ones(window, dtype=float) / window
                y = np.asarray(speeds, dtype=float)
                # Pad at both ends to maintain length and reduce edge effects
                pad = window // 2
                ypad = np.pad(y, (pad, pad), mode='edge')
                smoothed = np.convolve(ypad, k, mode='valid')

    if smoothed is not None and len(smoothed) == len(speeds):
            ax2.plot(ts, smoothed, '-b', linewidth=2.0, label=f'smoothed (w={window})')
    else:
        ax2.text(0.5, 0.5, 'No speed data', transform=ax2.transAxes, ha='center', va='center')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('speed (m/s)')
    ax2.set_title('Speed vs Time')
    ax2.grid(True)

    # Optional vertical labeled time markers
    def _load_time_lines(path: str) -> List[Tuple[float, str]]:
        lines: List[Tuple[float, str]] = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = f.read().strip()
            # Try JSON first: expecting list of items
            parsed: Any
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        t_val: Optional[float] = None
                        label_val: str = ""
                        if isinstance(item, dict):
                            # Accept a few common time keys, including 'tim' per request
                            for k in ("time", "t", "timestamp", "sec", "s", "tim"):
                                if k in item and item[k] is not None:
                                    try:
                                        t_val = float(item[k])
                                        break
                                    except Exception:
                                        pass
                            for lk in ("label", "name", "tag", "desc", "title"):
                                if lk in item and item[lk] is not None:
                                    label_val = str(item[lk])
                                    break
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            a, b = item[0], item[1]
                            # Heuristic: if first is str, treat as (label, time)
                            if isinstance(a, str):
                                label_val = a
                                try:
                                    t_val = float(b)
                                except Exception:
                                    t_val = None
                            else:
                                # assume (time, label)
                                try:
                                    t_val = float(a)
                                except Exception:
                                    t_val = None
                                label_val = str(b)
                        elif isinstance(item, str):
                            # Try to split on comma or whitespace
                            token = item.strip()
                            if "," in token:
                                parts = [p.strip() for p in token.split(",", 1)]
                            else:
                                parts = token.split()
                            if len(parts) >= 2:
                                # Choose mapping by type of first token
                                try:
                                    # time first
                                    t_val = float(parts[0])
                                    label_val = parts[1]
                                except Exception:
                                    # label first
                                    label_val = parts[0]
                                    try:
                                        t_val = float(parts[1])
                                    except Exception:
                                        t_val = None
                        if t_val is not None:
                            lines.append((t_val, label_val))
                else:
                    # Not a list JSON; fall back to line parsing
                    raise ValueError("JSON root not a list")
            except Exception:
                # Not JSON; parse as line-based CSV-ish: either "time,label" or "label,time"
                for ln in raw.splitlines():
                    s = ln.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "," in s:
                        p1, p2 = [x.strip() for x in s.split(",", 1)]
                        try:
                            t_val = float(p1)
                            label_val = p2
                        except Exception:
                            # maybe label first
                            label_val = p1
                            try:
                                t_val = float(p2)
                            except Exception:
                                continue
                        lines.append((t_val, label_val))
                    else:
                        parts = s.split()
                        if len(parts) >= 2:
                            try:
                                t_val = float(parts[0])
                                label_val = parts[1]
                            except Exception:
                                label_val = parts[0]
                                try:
                                    t_val = float(parts[1])
                                except Exception:
                                    continue
                            lines.append((t_val, label_val))
        except Exception:
            lines = []
        return lines

    if args.time_lines:
        markers = _load_time_lines(args.time_lines)
        if markers:
            for t_mark, lab in markers:
                ax2.axvline(x=t_mark, color=args.time_line_color, linestyle=args.time_line_style, alpha=args.time_line_alpha, linewidth=1.2)
                # Place label near the top of the axes, aligned to the line
                ax2.text(
                    t_mark,
                    0.98,
                    f" {lab}",
                    rotation=90,
                    va='top',
                    ha='left',
                    transform=ax2.get_xaxis_transform(),
                    color=args.time_line_color,
                )

    ax2.legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
