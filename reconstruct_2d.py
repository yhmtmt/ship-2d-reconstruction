#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def rodrigues_to_R(rvec: List[float]) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    return R


def undistort_norm(u: float, v: float, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pts = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist)
    x, y = und[0, 0]
    return np.array([x, y, 1.0], dtype=np.float64)


def backproject_to_plane(u: float, v: float, plane_z: float, K: np.ndarray, dist: np.ndarray,
                         R: np.ndarray, t: np.ndarray) -> Tuple[float, float, float]:
    # X_cam = R X_world + t
    # Ray in camera coords
    d_cam = undistort_norm(u, v, K, dist)  # direction proportional to (x, y, 1)
    # Normalize direction vector for numeric stability
    d_cam = d_cam / np.linalg.norm(d_cam)
    # Camera center in world
    Rt = R.T
    C = -Rt @ t.reshape(3)
    # Direction in world
    d_world = Rt @ d_cam
    if abs(d_world[2]) < 1e-9:
        raise ZeroDivisionError("Ray nearly parallel to plane; cannot intersect.")
    tau = (plane_z - C[2]) / d_world[2]
    Xw = C + tau * d_world
    return float(Xw[0]), float(Xw[1]), float(Xw[2])


def solve_rigid_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    # Procrustes rigid (no scale). src->dst
    mu_s = src_xy.mean(axis=0)
    mu_d = dst_xy.mean(axis=0)
    X = src_xy - mu_s
    Y = dst_xy - mu_d
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1
        R2 = Vt.T @ U.T
    t = mu_d - (R2 @ mu_s)
    yaw = math.degrees(math.atan2(R2[1, 0], R2[0, 0]))
    return R2, t, yaw


def interactive_collect_matches(video_path: str, ship_points: List[Dict[str, Any]],
                                start_frame: int, frame_step: int, max_frames: Optional[int]) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ids = [str(p.get("id", i)) for i, p in enumerate(ship_points)]

    frames: List[Dict[str, Any]] = []
    window = "Reconstruction: click each point (u=undo, s=save, space=skip, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def annotate_frame(frame_idx: int) -> Tuple[int, int, List[Dict[str, int]], str]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        h, w = frame.shape[:2]
        collected: List[Tuple[int, int]] = []
        k = 0

        def on_mouse(event, x, y, flags, param):
            nonlocal collected, k
            if event == cv2.EVENT_LBUTTONDOWN and k < len(ids):
                collected.append((x, y))
                k += 1

        cv2.setMouseCallback(window, on_mouse)
        while True:
            disp = frame.copy()
            cv2.putText(disp, f"Frame {frame_idx}/{total_frames-1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(disp, "Point order:", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            y0 = 70
            for i, pid in enumerate(ids):
                color = (0, 255, 0) if i < k else (128, 128, 128)
                cv2.putText(disp, f"{i+1}:{pid}", (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            for (x, y), pid in zip(collected, ids[:len(collected)]):
                cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(disp, pid, (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow(window, disp)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('u') and collected:
                collected.pop(); k -= 1
            elif key == ord('s') and len(collected) == len(ids):
                obs = [{"point_id": pid, "u": int(x), "v": int(y)} for (x, y), pid in zip(collected, ids)]
                return w, h, obs, "save"
            elif key == ord(' '):
                return w, h, [], "skip"
            elif key == ord('q'):
                return w, h, [], "quit"

    frame_indices: List[int] = []
    fi = max(0, start_frame)
    taken = 0
    while fi < total_frames and (max_frames is None or taken < max_frames):
        w, h, obs, status = annotate_frame(fi)
        if status == "quit":
            break
        frames.append({
            "frame_index": fi,
            "image_size": {"width": w, "height": h},
            "observations": obs
        })
        frame_indices.append(fi)
        taken += 1
        fi += max(1, frame_step)

    cv2.destroyWindow(window)
    cap.release()

    return {
        "video": os.path.basename(video_path),
        "fps": fps,
        "frame_step": frame_step,
        "frames": frames
    }


def reconstruct_trajectory(matches: Dict[str, Any], cam_params: Dict[str, Any], ship_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    K = np.array([[cam_params["intrinsics"]["fx"], cam_params["intrinsics"]["skew"], cam_params["intrinsics"]["cx"]],
                  [0.0, cam_params["intrinsics"]["fy"], cam_params["intrinsics"]["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([cam_params.get("distortion", {}).get("k1", 0.0),
                     cam_params.get("distortion", {}).get("k2", 0.0),
                     cam_params.get("distortion", {}).get("p1", 0.0),
                     cam_params.get("distortion", {}).get("p2", 0.0),
                     cam_params.get("distortion", {}).get("k3", 0.0)], dtype=np.float64)
    R = rodrigues_to_R(cam_params["extrinsics"]["rvec"])  # world->cam
    t = np.array(cam_params["extrinsics"]["tvec"], dtype=np.float64).reshape(3, 1)

    id_to_ship = {str(p.get("id")): p for p in ship_points}
    fps = matches.get("fps", 0.0) or 0.0

    out_frames: List[Dict[str, Any]] = []

    for f in matches.get("frames", []):
        obs = f.get("observations", [])
        world_xy = []
        ship_xy = []
        obs_out = []
        for m in obs:
            pid = str(m["point_id"])
            sp = id_to_ship.get(pid)
            if sp is None:
                continue
            z_world = float(sp.get("z", 0.0))  # ship z relative to sea surface
            u = float(m["u"]); v = float(m["v"])
            try:
                xw, yw, zw = backproject_to_plane(u, v, z_world, K, dist, R, t)
            except ZeroDivisionError:
                continue
            world_xy.append([xw, yw])
            ship_xy.append([float(sp.get("x", 0.0)), float(sp.get("y", 0.0))])
            obs_out.append({
                "point_id": pid,
                "pixel": {"u": int(u), "v": int(v)},
                "world": {"x": xw, "y": yw, "z": zw}
            })

        pose = None
        if len(world_xy) >= 2:
            W = np.array(world_xy, dtype=np.float64)
            S = np.array(ship_xy, dtype=np.float64)
            R2, t2, yaw = solve_rigid_2d(S, W)
            pose = {
                "x": float(t2[0]),
                "y": float(t2[1]),
                "yaw_deg": float(yaw)
            }

        out_frames.append({
            "frame_index": int(f.get("frame_index", 0)),
            "timestamp": float(f.get("frame_index", 0)) / fps if fps > 0 else None,
            "pose": pose,
            "observations": obs_out
        })

    # Compute speed from positions
    speeds = []
    for i in range(1, len(out_frames)):
        p0 = out_frames[i-1]["pose"]
        p1 = out_frames[i]["pose"]
        t0 = out_frames[i-1]["timestamp"]
        t1 = out_frames[i]["timestamp"]
        if p0 is None or p1 is None or t0 is None or t1 is None:
            speeds.append(None)
            continue
        dt = max(1e-6, t1 - t0)
        dx = p1["x"] - p0["x"]
        dy = p1["y"] - p0["y"]
        speed = math.hypot(dx, dy) / dt
        speeds.append(speed)

    # Attach speeds back (speed for frame i uses previous pose)
    for i in range(len(out_frames)):
        if i == 0 or speeds[i-1] is None:
            out_frames[i]["speed_mps"] = None
        else:
            out_frames[i]["speed_mps"] = float(speeds[i-1])

    return {
        "frames": out_frames,
        "metadata": {
            "fps": fps
        }
    }


def main():
    ap = argparse.ArgumentParser(description="2D sea-surface reconstruction from single calibrated camera and ship point tracks.")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--camera-params", required=True, help="Camera parameters JSON (from calibrate_camera.py)")
    ap.add_argument("--ship-points", required=True, help="Ship points JSON with ship-local coordinates")
    ap.add_argument("--matches-in", default=None, help="Existing ship matches JSON to load")
    ap.add_argument("--matches-out", default="ship_matches.json", help="Path to save ship matches JSON")
    ap.add_argument("--output", default="reconstruction.json", help="Path to save reconstruction JSON")
    ap.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    ap.add_argument("--frame-step", type=int, default=5, help="Frame step for annotation")
    ap.add_argument("--max-frames", type=int, default=None, help="Max frames to annotate")

    args = ap.parse_args()

    cam_params = load_json(args.camera_params)
    ship_points = load_json(args.ship_points)["points"]

    if args.matches_in and os.path.exists(args.matches_in):
        matches = load_json(args.matches_in)
    else:
        matches = interactive_collect_matches(
            args.video, ship_points, start_frame=args.start_frame, frame_step=args.frame_step, max_frames=args.max_frames
        )
        save_json(args.matches_out, matches)
        print(f"Saved matches to {args.matches_out}")

    recon = reconstruct_trajectory(matches, cam_params, ship_points)
    save_json(args.output, recon)
    print(f"Saved reconstruction to {args.output}")


if __name__ == "__main__":
    main()
