# Motion Lab

A standalone GitHub Pages laboratory for exploring MediaPipe Pose estimated motion. One Pose Landmarker pipeline at a time, with all 33 landmarks including wrist, index, pinky, thumb, ankle, heel and foot index. No separate hand model, backend, API key or video recording.

## Deploy and use

Create a GitHub repository, upload `index.html` and `README.md` to its root, then enable Pages under Settings > Pages > Deploy from a branch > main > /(root). Open the published HTTPS URL and allow camera access. Choose Lite first for speed; Full and Heavy are alternatives. GPU is tried by default, with CPU fallback if initialization fails. Stop the camera to change model or processor.

## Explore and record

Choose a landmark and coordinate space; examine position, velocity, acceleration, visibility, or elbow/knee angle (choose the elbow or knee landmark). Compare raw and exponentially filtered signals. Alpha 1 disables extra filtering. Smaller alpha adds lag. Pause freezes the graph, not acquisition. Select recording landmarks individually or with presets, choose duration and label, and press Record. Mark events with M or the button; stop early or let the timer expire. Review a saved trial and scrub its time. Export CSV (long format, one row per landmark per sample), JSON (all selected channels, settings and markers), or marker CSV. Recordings remain in tab memory only; export before reloading. Long all-landmark recordings can consume substantial memory, so start with short trials. Until stopped has a 10-minute cap.

## Coordinate and measurement conventions

- Image: unmirrored normalized x/y plus model-estimated relative z; x right, y down. Mirroring only affects the preview.
- Pixel: x and y scaled by actual camera dimensions; z scaled by width for a display-scale proxy, not measured depth.
- World: model-estimated meters relative to hip midpoint; not calibrated absolute camera or court coordinates.
- Shoulder relative: subtract shoulder midpoint and divide by projected shoulder width; useful for controlling for body scale but unstable when shoulders are foreshortened.
- Velocity and acceleration: finite differences using inference timestamps in seconds. Image/pixel/relative/world units per second or per second squared. These are not ground truth; differentiation amplifies noise. Missing or below-cutoff observations are blank, and derivatives reset across gaps. Raw means model output before app filtering, not raw sensor truth. Visibility is a model estimate, not a guarantee of accuracy.
- Joint angles are 2D image-pixel elbow/knee angles, in degrees, independent of the coordinate-space selector. JSON includes angle traces. CSV focuses on selected landmark channels. `rv/fv` mean raw/filtered velocity and `ra/fa` mean raw/filtered acceleration.

## Suggested experiments

1. Hold your wrist still for 10 seconds at alpha 1, then repeat at alpha 0.25. Compare jitter.
2. Record five deliberate left, center and right flicks, labeled separately. Compare wrist x velocity, elbow angle and hand-point position. Use event markers to identify motion phases.
3. Repeat quick wrist flicks across model choices; compare inference time and pose FPS, and inspect the peak motion and lag.
4. Record ankle, heel and foot index during a step or jump. Compare position and velocity, and observe occlusion or tracking loss.
5. Move nearer/farther from the camera and compare image coordinates against shoulder-relative and world estimates.

## Privacy and troubleshooting

Camera frames are processed on the device and not uploaded by this app. Runtime and model assets are downloaded from public hosting. Use GitHub Pages HTTPS, grant camera permission, and ensure cdn.jsdelivr.net and storage.googleapis.com are accessible. If setup fails, try CPU or Lite. Inference is synchronous and may reduce render rate on slower devices. Fast movements can be undersampled. No separate hand skeleton is provided; pose hand and foot landmarks are coarse estimates. JavaScript syntax checked; live camera behavior requires browser testing.
