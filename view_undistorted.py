#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Tuple, List, Dict, Optional

import cv2
import numpy as np


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_K_and_dist(cam_params: dict) -> Tuple[np.ndarray, np.ndarray]:
    intr = cam_params["intrinsics"]
    K = np.array([[float(intr.get("fx", 0.0)), float(intr.get("skew", 0.0)), float(intr.get("cx", 0.0))],
                  [0.0, float(intr.get("fy", 0.0)), float(intr.get("cy", 0.0))],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_p = cam_params.get("distortion", {})
    dist = np.array([
        float(dist_p.get("k1", 0.0)),
        float(dist_p.get("k2", 0.0)),
        float(dist_p.get("p1", 0.0)),
        float(dist_p.get("p2", 0.0)),
        float(dist_p.get("k3", 0.0))
    ], dtype=np.float64)
    return K, dist


def make_maps(K: np.ndarray, dist: np.ndarray, size: Tuple[int, int], alpha: float, mode: str):
    w, h = size
    if mode == "originalK":
        newK = K.copy()
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
        # Compute a reference ROI for display using optimal matrix with alpha=1
        _, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1.0, (w, h))
        return newK, map1, map2, roi
    else:
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)
        return newK, map1, map2, roi


def rodrigues_to_Rt(cam_params: dict) -> Tuple[np.ndarray, np.ndarray]:
    extr = cam_params.get("extrinsics", {})
    rvec = np.array(extr.get("rvec", [0, 0, 0]), dtype=np.float64).reshape(3, 1)
    tvec = np.array(extr.get("tvec", [0, 0, 0]), dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


def project_points(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    img, _ = cv2.projectPoints(points.astype(np.float64), rvec.astype(np.float64), tvec.astype(np.float64), K.astype(np.float64), dist.astype(np.float64))
    return img.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(description="View video undistorted using camera_params.json")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--camera-params", required=True, help="Camera parameters JSON (from calibrate_camera.py)")
    ap.add_argument("--world-points", default=None, help="Optional: world_points JSON to overlay reprojected static points")
    ap.add_argument("--alpha", type=float, default=1.0, help="Free scaling parameter [0..1] for optimal matrix; default 1 keeps full FOV")
    ap.add_argument("--mode", choices=["originalK", "optimal"], default="originalK", help="Use original K (no crop, black borders) or optimal new camera matrix")
    ap.add_argument("--resize-width", type=int, default=None, help="Resize display width in pixels")
    ap.add_argument("--write-out", default=None, help="Optional path to save undistorted video")
    args = ap.parse_args()

    cam_params = load_json(args.camera_params)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first frame")
    h, w = frame.shape[:2]

    K, dist = build_K_and_dist(cam_params)
    R, t = rodrigues_to_Rt(cam_params)
    rvec, _ = cv2.Rodrigues(R)
    alpha = max(0.0, min(1.0, float(args.alpha)))
    mode = args.mode
    newK, map1, map2, roi = make_maps(K, dist, (w, h), alpha, mode)

    world_pts: Optional[np.ndarray] = None
    world_ids: List[str] = []
    if args.world_points:
        try:
            wp = load_json(args.world_points)
            pts = wp.get("points", [])
            arr = []
            for p in pts:
                arr.append([float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))])
                world_ids.append(str(p.get("id", len(world_ids))))
            if arr:
                world_pts = np.array(arr, dtype=np.float64)
        except Exception as e:
            print(f"Warning: failed to load world points: {e}")

    writer = None
    if args.write_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.write_out, fourcc, fps, (w, h))

    paused = False
    show_undistorted = True
    show_proj = True if world_pts is not None else False
    window = "Undistorted Viewer (u toggle, m mode, +/- alpha, space pause, s save, q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def maybe_resize(img):
        if args.resize_width and img is not None and img.shape[1] != args.resize_width:
            scale = args.resize_width / img.shape[1]
            nh = int(round(img.shape[0] * scale))
            return cv2.resize(img, (args.resize_width, nh), interpolation=cv2.INTER_AREA)
        return img

    index = 0
    while True:
        if not paused and index > 0:
            ok, frame = cap.read()
            if not ok:
                break
        index += 1

        if show_undistorted:
            und = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            disp = und
        else:
            disp = frame

        overlay = disp.copy()
        label = (f"undistorted[{mode}] alpha={alpha:.2f}" if show_undistorted else "original")
        cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        # Draw ROI rectangle (valid region) if undistorting
        if show_undistorted and roi is not None:
            x, y, rw, rh = roi
            cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (0, 255, 0), 1)

        # Overlay reprojected static world points
        if show_proj and world_pts is not None:
            if show_undistorted:
                # Undistorted display uses newK with zero distortion
                proj = project_points(world_pts, rvec, t, newK, np.zeros(5))
            else:
                # Original display uses original K & distortion
                proj = project_points(world_pts, rvec, t, K, dist)
            for (u, v), pid in zip(proj, world_ids):
                u_i, v_i = int(round(u)), int(round(v))
                if 0 <= u_i < w and 0 <= v_i < h:
                    cv2.circle(overlay, (u_i, v_i), 4, (0, 0, 255), -1)
                    cv2.putText(overlay, pid, (u_i + 6, v_i - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        disp = overlay

        if writer and show_undistorted:
            writer.write(und)

        cv2.imshow(window, maybe_resize(disp))
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('u'):
            show_undistorted = not show_undistorted
        elif key in (ord('+'), ord('=')):
            alpha = min(1.0, alpha + 0.05)
            newK, map1, map2, roi = make_maps(K, dist, (w, h), alpha, mode)
        elif key in (ord('-'), ord('_')):
            alpha = max(0.0, alpha - 0.05)
            newK, map1, map2, roi = make_maps(K, dist, (w, h), alpha, mode)
        elif key == ord('m'):
            mode = "optimal" if mode == "originalK" else "originalK"
            newK, map1, map2, roi = make_maps(K, dist, (w, h), alpha, mode)
        elif key == ord('p'):
            show_proj = not show_proj
        elif key == ord('s'):
            out_name = f"undistorted_frame_{index:06d}.png"
            cv2.imwrite(out_name, und if show_undistorted else frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
