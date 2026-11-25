# Motion Lab

## [Launch Motion Lab ↗](https://aaron-ragsdale.github.io/MotionLab/)

[![Open Motion Lab in your browser](https://img.shields.io/badge/Launch-Motion%20Lab-2563eb?style=for-the-badge)](https://aaron-ragsdale.github.io/MotionLab/)

**Direct website:** <https://aaron-ragsdale.github.io/MotionLab/>

No installation, downloads, GitHub account, or API key required.

### Quick start

1. Open the website using the link above.
2. Start the camera and allow camera access when prompted.
3. Choose a landmark and explore your movement in real time.

A standalone GitHub Pages laboratory for exploring MediaPipe Pose estimated motion. One Pose Landmarker pipeline at a time, with all 33 landmarks including wrist, index, pinky, thumb, ankle, heel and foot index. No separate hand model, backend, API key or video recording.

## Using Motion Lab

Open the live website linked above. Choose Lite first for speed; Full and Heavy are alternatives. GPU is tried by default, with CPU fallback if initialization fails. Stop the camera to change model or processor.

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

## Host your own copy (optional)

Fork this repository and enable GitHub Pages under Settings > Pages > Deploy from a branch > main > /(root). Open your published HTTPS URL. This is only necessary if you want to host or modify your own version.


## Train a custom controller model

Open the [live MotionLab](https://aaron-ragsdale.github.io/MotionLab/), choose the detector in section 01 (Pose, Hands, or Both), select a pose variant if applicable, then start the camera. Choose static recognition (one frame) or a temporal window (100–2000 ms). Enter your own class names, including a neutral/background class.

Select a class and record a take. Static takes collect examples over four seconds; dynamic takes capture one movement after a countdown. Record at least three separate takes per class, preferably many more with varied positions and speeds. Choose random forest, K-nearest neighbors, or nearest centroid in section 07 and inspect held-out-take accuracy. Test live predictions, adjust vote threshold and prediction stride, then download the model or training dataset. Import a saved model to test it again using matching camera settings.

### Portable runtime

`controller-runtime.js` exports `frameFeatures(poseResult, handResult, mode)`, `windowFeatures(buffer, timestamp, input)`, `predict(model, features)`, and `validateModel(model)`. The JSON format is `modellab-controller-v1`. It contains class labels, algorithm-specific parameters (trees, saved feature vectors, or centroids), and input configuration (detector, pose variant, 30 Hz resampling, number of frames and feature schema). Frame features use camera-normalized x/y/z plus a presence bit for each landmark, in pose then left-hand then right-hand order, as applicable. Missing landmarks are zero-filled. Use a timestamped buffer of objects `{t, x}` and the same runtime for inference. No raw camera video is exported. Models are camera-relative and may need varied training examples for reliable use.

The vote score is not a calibrated probability. Held-out takes are a preliminary evaluation, not evidence of generalization to new users. Window size and preprocessing should match training; changing them generally requires retraining. The Chrome extension and cross-site input dispatch are not included yet.


The controller studio inherits the camera detector settings. Changing detectors requires a compatible dataset or clearing the old one. The Learn more guide explains windowing, stride, scores, and exporting. KNN exports training feature vectors, not just learned parameters; share these files with that in mind. Scores are algorithm-specific and not calibrated probabilities. The same recording takes can be used to retrain each algorithm.


## Camera startup and dependencies

The camera panel displays setup progress for the selected detector(s), including camera access, JavaScript runtime, WASM selection, model downloads and detector initialization. Model transfers show received bytes and a percentage bar when the server exposes a usable content length. A completed checklist means the selected detectors initialized successfully for this session, not that their files are permanently cached or available offline. Later starts may be faster through normal browser caching. If startup fails, check the failed stage, camera permission, network access and CPU/Lite settings.


## Learning guide

[Open the MotionLab Learning Guide](https://aaron-ragsdale.github.io/MotionLab/tutorial.html) for twelve short chapters on landmarks, coordinate systems, movement signals, smoothing, data collection, temporal windowing, machine learning algorithms, evaluation and controller deployment. Includes interactive smoothing and sliding-window demonstrations and hands-on experiments. The guide opens in a separate tab from the app header to preserve the current working session.


### Unified detector modes

Pose, Hands and Both now share the signal explorer, current landmark table, selected-point trial recording, review and movement classifier studio. Stable IDs are 0–32 for pose, 33–53 for left hand and 54–74 for right hand. Hand-containing selected-point classifier datasets use normalized image coordinates; pose-only datasets retain their original coordinate options. Hand presence is not a calibrated visibility score. The portable controller studio continues to use its separate standardized export schema.
