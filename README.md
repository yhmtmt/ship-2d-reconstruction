
# Single-Camera 2D Trajectory Reconstruction Toolkit

This repository provides Python tools to calibrate a single surveillance camera from static world points, reconstruct a vessel's 2D sea-surface trajectory and speed from tracked ship points, and visualize/verify results.

- Calibrate camera intrinsics/extrinsics from one annotated frame
- View undistorted video and reprojected static points
- Reconstruct 2D trajectory and speed per video, independently
- Plot trajectory with static markers and speed vs time
- JSON schemas and example inputs included

These tools implement a technique I have used for many years to reconstruct ship trajectories and velocities from fixed surveillance cameras, originally developed for requests from the Japan Coast Guard. The approach exploits the fact that ships operate on the sea surface, enabling monocular reconstruction without multi‑camera setups. Recent neural methods (e.g., SAM3D) deliver impressive automated results, but when exact error bounds are required, this traditional, model‑based method remains essential.

## Install

- Python 3.9+
- `pip install -r requirements.txt`

## Coordinate Systems

- World: x,y in sea-surface plane (z=0), z up; origin at camera. Horizontal axes may be any convenient metric basis (e.g., ENU from camera).
- Ship: x starboard, y bow, z up; origin at sea-surface center.

## Tools

- `calibrate_camera.py`: annotate static world points in one frame; estimate camera params.
- `view_undistorted.py`: view undistorted/original frames; overlay projected world points.
- `reconstruct_2d.py`: annotate ship points across frames; output 2D poses and speed.
- `plot_trajectory.py`: plot trajectory + static markers and speed vs time.
- `convert_geo_to_world.py`: convert `name lat lon alt` to world_points JSON (origin at camera).
- `convert_ship_points.py`: convert `name x y z` to ship_points JSON.

## JSON Schemas

See `schemas/*.schema.json`.

## Typical Workflow

1. Prepare world points (metric x,y,z) or convert from lat/lon/alt:
   - `python convert_geo_to_world.py --input world_points.txt --output world_points.json`
2. Prepare initial camera guess (FOV or focal length + sensor width):
   - `examples/initial_camera_fov.example.json` or `examples/initial_camera_lens.example.json`
3. Calibrate:
   - `python calibrate_camera.py --video YOUR_VIDEO.mp4 --world-points world_points.json --initial-camera examples/initial_camera_fov.example.json --frame-index 0 --matches-out calib_matches.json --params-out camera_params.json --fix-cx <width/2> --fix-cy <height/2>`
   - Keys: click listed points; `u` undo; `s` save; `q` quit.
   - Constraints: no distortion (all k,p=0), fx=fy, fixed principal point.
4. Verify undistortion and projection overlay:
   - `python view_undistorted.py --video YOUR_VIDEO.mp4 --camera-params camera_params.json --world-points world_points.json --mode originalK`
   - Keys: `u` toggle, `m` mode, `+/-` alpha, `p` points overlay, space/s/q.
5. Prepare ship points (ship-local x,y,z) or convert from txt:
   - `python convert_ship_points.py --input ship_points.txt --output ship_points.json`
6. Reconstruct trajectory from annotated frames:
   - `python reconstruct_2d.py --video YOUR_VIDEO.mp4 --camera-params camera_params.json --ship-points ship_points.json --frame-step 5 --matches-out ship_matches.json --output reconstruction.json`
7. Plot trajectory and speed:
   - `python plot_trajectory.py --reconstruction reconstruction.json --world-points world_points.json --title "Sample Run" --save trajectory_speed.png`

## Theory (Brief)

- Camera: `u ~ K [R|t] X` with fx=fy, fixed center, zero distortion to avoid overfitting sparse points. OpenCV `calibrateCamera` with flags fixes principal point, aspect ratio, and all distortion terms.
- Reconstruction: back-project pixel rays, intersect with plane `z = z_s` for each ship point to get world `(x,y)` per frame; fit 2D rigid transform from ship-local `(x,y)` to those world `(x,y)` to obtain `(x,y,yaw)` per frame; speed via finite differences.
- Undistortion: remap using either the original K (full frame, borders) or optimal new K from `getOptimalNewCameraMatrix` (alpha controls crop/FOV).
