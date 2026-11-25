# MotionLab

[Open MotionLab](https://aaron-ragsdale.github.io/MotionLab/) · [Learning Guide](https://aaron-ragsdale.github.io/MotionLab/tutorial.html)

A browser-based movement analysis and machine learning laboratory for OpenSportSense. Choose Pose, Hands or Both; explore live positions, velocity, acceleration and filtering; record selected landmarks; review and export trials; and train portable classifiers locally.

## Machine Learning Studio (current version)

The former movement classifier and portable controller studios are merged into section 06. Select all available landmarks or the recording selection, static or dynamic recognition, and coordinate space. Dynamic windows range from 100 to 3000 ms at a target resampling rate of 30 Hz. Include a neutral class and record at least three independent takes per class; 10–20 are recommended. Add compatible saved trials using their last window.

Choose any subset of random forest, KNN, nearest centroid, linear SVM and RBF SVM. Models are evaluated on the same deterministic held-out takes and refit on all examples. Compare held-out accuracy, confusion matrices, classifier-only inference time and approximate export size. Switch active models for live testing or export them independently. Changing the dataset invalidates trained models. Configuration changes require restoring compatible settings or clearing the dataset. SVM uses a local one-vs-rest soft-margin SMO implementation with standardization and a bounded iteration budget, not an external library; it is limited to 1200 examples. Training runs on the browser main thread and large models may temporarily stall the UI.

## Portable formats

New classifiers use `motionlab-classifier-v2` and datasets use `motionlab-dataset-v2`. Earlier classifier export formats are not directly importable into the new studio. `controller-runtime.js` contains the selected-landmark feature extractor, resampling, training, prediction and model validation. `machine-learning-studio.js` handles dataset collection and model comparison. Stable IDs are 0–32 pose, 33–53 left hand and 54–74 right hand. Input configuration includes detector mode, pose variant, IDs, coordinate space, sample rate, frames and visibility threshold. Each selected point contributes x/y/z and a presence flag; missing points are zero-filled with presence zero. SVM exports scaling parameters; KNN and RBF SVM may include training vectors. Scores are not calibrated probabilities. The controller configuration section maps each class to a command, tap or hold behavior, confirmation count and cooldown. The local playground tests these mappings from live predictions; uncertain or missing predictions release held actions. Export `motionlab-controller-v2` to bundle the classifier and mappings. A future compatible extension must reproduce this input pipeline and command behavior; the Chrome extension itself is not included.

## Camera and signal tools

Select the detector at the top. Pose variants Lite, Full and Heavy are available for body tracking. The setup panel reports model downloads and initialization. Model assets may be browser-cached; offline availability is not guaranteed. The shared signal explorer supports normalized and pixel coordinates for hands and pose, plus pose-specific world and shoulder-relative coordinates, visibility and joint angles. Record selected landmarks, mark events, review trials, and export CSV/JSON. Hand presence is not a per-point confidence score. Hand-containing classifier datasets use normalized coordinates.

Camera processing and training run locally; runtime and pretrained assets download from public CDNs. No raw video is exported. Landmark data may be sensitive. Export datasets before refreshing.

## Learning guide

The separate tutorial covers twelve chapters with interactive smoothing and sliding-window demos, definitions, practical tradeoffs and experiments. It opens without requesting camera access.

## Legacy files

`classifier.js` and `training-lab.js` are retained for reference but no longer loaded by the main application.

The training capture button changes from Record example to Cancel countdown / Cancel recording, then returns to Record example after completion.
