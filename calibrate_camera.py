#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np


@dataclass
class ImageSize:
    width: int
    height: int


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_initial_intrinsics(initial_cam: Dict[str, Any], image_size: ImageSize,
                             fixed_cx: Optional[float] = None, fixed_cy: Optional[float] = None) -> np.ndarray:
    w, h = image_size.width, image_size.height
    cx = fixed_cx if fixed_cx is not None else (w / 2.0)
    cy = fixed_cy if fixed_cy is not None else (h / 2.0)

    fx = None
    # Option 1: horizontal FOV
    fov = initial_cam.get("fov_deg", {})
    if isinstance(fov, dict) and "horizontal" in fov:
        hfov_rad = math.radians(float(fov["horizontal"]))
        fx = (w / 2.0) / math.tan(hfov_rad / 2.0)
    # Option 2: focal length & sensor size
    if fx is None and "focal_length_mm" in initial_cam:
        f_mm = float(initial_cam["focal_length_mm"]) 
        sensor_w = float(initial_cam.get("sensor_width_mm", 4.8))  # default 1/3" approx
        fx = f_mm / sensor_w * w

    if fx is None:
        # conservative default if nothing given: 60 deg HFOV
        fx = (w / 2.0) / math.tan(math.radians(60.0) / 2.0)

    # Enforce fx == fy in the initial guess
    K = np.array([[fx, 0.0, cx],
                  [0.0, fx, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def draw_points(img: np.ndarray, points: List[Tuple[int, int]], labels: List[str]) -> None:
    for (x, y), label in zip(points, labels):
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        # Show only compact numeric labels to avoid occlusion
        cv2.putText(img, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def interactive_collect_matches(video_path: str, world_points: List[Dict[str, Any]],
                                frame_index: int = 0) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0 or frame_index >= max(1, total_frames):
        frame_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read frame for matching.")

    h, w = frame.shape[:2]
    labels = [str(p.get("id", i)) for i, p in enumerate(world_points)]

    # Display scale to enlarge window and reduce label overlap
    # Keep clicks mapped back to original pixel coordinates
    disp_scale = 1.5 if max(w, h) < 1600 else 1.0

    collected: List[Tuple[int, int]] = []
    idx = 0

    window = "Calib: click points in listed order (u=undo, s=save, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if disp_scale != 1.0:
        cv2.resizeWindow(window, int(w * disp_scale), int(h * disp_scale))

    def on_mouse(event, x, y, flags, param):
        nonlocal collected, idx
        if event == cv2.EVENT_LBUTTONDOWN and idx < len(labels):
            # Map display coords back to original image coords
            if disp_scale != 1.0:
                x_img = int(round(x / disp_scale))
                y_img = int(round(y / disp_scale))
            else:
                x_img, y_img = int(x), int(y)
            # Clamp to image bounds
            x_img = max(0, min(w - 1, x_img))
            y_img = max(0, min(h - 1, y_img))
            collected.append((x_img, y_img))
            idx += 1

    cv2.setMouseCallback(window, on_mouse)

    while True:
        disp = frame.copy()
        # instructions
        cv2.putText(disp, f"Frame {frame_index}/{total_frames-1}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(disp, "Point order:", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        y0 = 70
        for i, lab in enumerate(labels):
            color = (0, 255, 0) if i < idx else (128, 128, 128)
            # Show only point number in the order list
            cv2.putText(disp, f"{i+1}", (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        # Draw points with numeric labels near them
        numeric_labels = [str(i+1) for i in range(len(collected))]
        draw_points(disp, collected, numeric_labels)
        # Scale for display if requested
        if disp_scale != 1.0:
            disp_show = cv2.resize(disp, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_LINEAR)
        else:
            disp_show = disp
        cv2.imshow(window, disp_show)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('u') and collected:
            collected.pop()
            idx -= 1
        elif key == ord('s') and len(collected) == len(labels):
            break
        elif key == ord('q'):
            break

    cv2.destroyWindow(window)
    cap.release()

    matches = []
    for lab, (x, y) in zip(labels, collected):
        matches.append({"point_id": lab, "u": int(x), "v": int(y)})

    return {
        "video": os.path.basename(video_path),
        "frame_index": frame_index,
        "image_size": {"width": w, "height": h},
        "matches": matches,
    }


def calibrate_from_matches(world_points: List[Dict[str, Any]],
                           matches: Dict[str, Any],
                           initial_cam: Dict[str, Any],
                           fixed_cx: Optional[float] = None,
                           fixed_cy: Optional[float] = None,
                           no_distortion: bool = True,
                           fix_aspect_equal: bool = True) -> Dict[str, Any]:
    w = int(matches["image_size"]["width"])  # type: ignore
    h = int(matches["image_size"]["height"])  # type: ignore
    image_size = ImageSize(w, h)

    # Reorder world points to match matches order by id
    id_to_wp = {str(p.get("id")): p for p in world_points}
    object_pts = []
    image_pts = []
    for m in matches["matches"]:
        pid = str(m["point_id"])  # type: ignore
        wp = id_to_wp.get(pid)
        if wp is None:
            raise ValueError(f"Match refers to unknown point id: {pid}")
        object_pts.append([float(wp["x"]), float(wp["y"]), float(wp["z"])])
        image_pts.append([float(m["u"]), float(m["v"])])

    # OpenCV 4.5.x expects float32 Point3f/Point2f in lists of arrays
    object_pts = np.asarray(object_pts, dtype=np.float32).reshape(-1, 1, 3)
    image_pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 1, 2)

    # Prepare data for calibrateCamera (list per view, here one view)
    obj = [object_pts]
    img = [image_pts]

    K0 = build_initial_intrinsics(initial_cam, image_size, fixed_cx=fixed_cx, fixed_cy=fixed_cy)
    # Ensure principal point is exactly at requested location in the guess
    if fixed_cx is not None:
        K0[0, 2] = fixed_cx
    if fixed_cy is not None:
        K0[1, 2] = fixed_cy

    dist0 = np.zeros((5, 1), dtype=np.float64)  # k1,k2,p1,p2,k3

    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    # Fix principal point if requested
    if fixed_cx is not None and fixed_cy is not None:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    # No tangential distortion
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # Disable radial distortion estimation completely
    if no_distortion:
        flags |= (
            cv2.CALIB_FIX_K1 |
            cv2.CALIB_FIX_K2 |
            cv2.CALIB_FIX_K3 |
            cv2.CALIB_FIX_K4 |
            cv2.CALIB_FIX_K5 |
            cv2.CALIB_FIX_K6
        )
    # Enforce fx == fy by fixing the aspect ratio to the initial (1.0)
    if fix_aspect_equal:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=obj,
        imagePoints=img,
        imageSize=(w, h),
        cameraMatrix=K0,
        distCoeffs=dist0,
        flags=flags
    )

    rvec = rvecs[0].reshape(-1).tolist()
    tvec = tvecs[0].reshape(-1).tolist()
    dist = dist.reshape(-1)

    # Compute per-point reprojection errors
    rvec_np = np.array(rvecs[0], dtype=np.float64)
    tvec_np = np.array(tvecs[0], dtype=np.float64)
    # Use float64 for projection inputs
    obj_pts64 = object_pts.reshape(-1, 3).astype(np.float64)
    img_pts64 = image_pts.reshape(-1, 2).astype(np.float64)
    proj, _ = cv2.projectPoints(obj_pts64, rvec_np, tvec_np, K, dist)
    proj = proj.reshape((-1, 2))
    errs = np.linalg.norm(proj - img_pts64, axis=1)
    err_stats = {
        "count": int(errs.size),
        "mean_px": float(errs.mean()) if errs.size else None,
        "std_px": float(errs.std()) if errs.size else None,
        "max_px": float(errs.max()) if errs.size else None
    }

    result = {
        "image_size": {"width": w, "height": h},
        "intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "skew": float(K[0, 1])
        },
        "distortion": {
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0
        },
        "extrinsics": {
            "rvec": rvec,
            "tvec": tvec
        },
        "reprojection_error_rms": float(rms),
        "reprojection_error_per_point_px": [float(e) for e in errs.tolist()],
        "reprojection_error_stats_px": err_stats,
        "metadata": {
            "matches": matches,
            "initial_camera": initial_cam,
            "flags": int(flags)
        }
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="Camera calibrator from a single video frame and known 3D points.")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--world-points", required=True, help="JSON file of surrounding static object points (x,y,z,id)")
    ap.add_argument("--initial-camera", required=True, help="JSON file for initial camera spec (FOV or lens+sensor)")
    ap.add_argument("--matches-out", default="calib_matches.json", help="Path to save pixel-3D matches JSON")
    ap.add_argument("--params-out", default="camera_params.json", help="Path to save estimated camera parameters JSON")
    ap.add_argument("--matches-in", default=None, help="Optional existing matches JSON to reuse")
    ap.add_argument("--frame-index", type=int, default=0, help="Frame index to use for calibration")
    ap.add_argument("--fix-cx", type=float, default=160.0, help="Fixed principal point cx (default 160.0)")
    ap.add_argument("--fix-cy", type=float, default=120.0, help="Fixed principal point cy (default 120.0)")

    args = ap.parse_args()

    world_points = load_json(args.world_points)["points"]
    initial_cam = load_json(args.initial_camera)

    if args.matches_in and os.path.exists(args.matches_in):
        matches = load_json(args.matches_in)
    else:
        matches = interactive_collect_matches(args.video, world_points, frame_index=args.frame_index)
        save_json(args.matches_out, matches)
        print(f"Saved matches to {args.matches_out}")

    params = calibrate_from_matches(
        world_points, matches, initial_cam,
        fixed_cx=args.fix_cx, fixed_cy=args.fix_cy,
        no_distortion=True, fix_aspect_equal=True
    )
    save_json(args.params_out, params)
    print(f"Saved camera parameters to {args.params_out}")


if __name__ == "__main__":
    main()
