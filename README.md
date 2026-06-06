# MediaPipe Learning — Hand & Gesture Utilities

Professional collection of scripts and utilities demonstrating MediaPipe hand landmarking, gesture detection, and lightweight classification helpers.

## Overview

This repository contains example projects and utilities for working with MediaPipe hand landmarks, real-time webcam demos, and a set of small classification/utility scripts. It is intended as a development and learning workspace rather than a production product.

## Contents

- **Entry point:** [main.py](main.py)
- **Webcam / demo scripts:** [webcam_test.py](webcam_test.py), [hand_tracking.py](hand_tracking.py), [hand_test.py](hand_test.py)
- **Utilities:** [count_fingers.py](count_fingers.py), [define_handedness.py](define_handedness.py)
- **Classifier helpers:** scripts prefixed with `is_` such as `is_person.py`, `is_gun.py`, `is_anime.py`, etc.
- **Models:** [models/hand_landmarker.task](models/hand_landmarker.task)

## Requirements

- Python 3.8 or later
- Core Python packages used in the workspace include: `mediapipe`, `opencv-python`, `numpy`, `pillow`, `matplotlib`, and `sounddevice`.

Note: The repository contains virtual environment folders (`mp_env`, `mp_env312`) used locally. It's recommended to create an isolated environment for your work.

## Installation (recommended)

1. Create and activate a virtual environment (example using `venv` on Windows PowerShell):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
```

2. Install core dependencies manually (example):

```powershell
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python numpy pillow matplotlib sounddevice
```

If you prefer to reproduce the exact environment in this workspace and an existing `mp_env` is available, activate it instead:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\mp_env\Scripts\Activate.ps1
```

## Usage

- Run the main demo:

```powershell
python main.py
```

- Run the webcam demo:

```powershell
python webcam_test.py
```

- Run the hand tracking example:

```powershell
python hand_tracking.py
```

- Use `count_fingers.py` to inspect landmark-based finger counting logic.

## Scripts and purpose

- **main.py** — Top-level script that can be used as a starting point for experiments.
- **webcam_test.py** — Simple webcam-based demo for visualizing landmarks.
- **hand_tracking.py** — Focused hand tracking example with frame processing.
- **count_fingers.py** — Example implementation for counting extended fingers.
- **define_handedness.py** — Tools to evaluate and normalize handedness outputs.
- **is_*.py** — Lightweight classifier/heuristic scripts for quick checks (examples: `is_person.py`, `is_gun.py`, `is_anime.py`).

## Models

Model tasks used by the demos are stored under the `models/` folder. The primary hand landmarker task file is:

- [models/hand_landmarker.task](models/hand_landmarker.task)

If you replace or update models, ensure filenames match the references used in the demo scripts.

## Development

- Follow the installation steps to prepare your environment.
- Run or modify the scripts to iterate on hand-tracking workflows.
- Keep virtual environments out of commits; add them to `.gitignore` if you plan to version-control this repository.

## Contributing

Contributions are welcome. Open issues for feature requests or bug reports and submit pull requests with focused changes and explanatory descriptions.

## License

No license is specified for this repository. Add a `LICENSE` file to define terms for reuse and distribution.
