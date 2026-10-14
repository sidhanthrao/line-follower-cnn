# Line Follower Robot (Raspberry Pi + Camera + CNN)

A camera-based line follower robot built on a Raspberry Pi 4B. An OpenCV
pipeline steers the robot for now, and a small CNN (trained in PyTorch, run with ONNX Runtime)
is being developed as an alternative.

![Robot photo](docs/images/robot-top.jpg)

## Status
- [x] Hardware built (RPi 4B, 2x BTS7960, RPi Camera, 3S 18650 pack)
- [x] Classical vision pipeline (colour threshold -> contour -> centroid -> 3-zone steering)
- [ ] CNN dataset collection (in progress)
- [ ] CNN training and on-robot testing (code written, untested)

## Hardware
| Part | Details |
|---|---|
| Computer | Raspberry Pi 4B, Raspberry Pi OS 64-bit |
| Camera | Raspberry Pi Camera (picamera2) |
| Motor drivers | 2x BTS7960, one per motor |
| Power | 3S 18650 pack -> LM2596 buck (7.4 V, motors) and LM2596 buck (5.1 V, Pi) |

## Wiring (BCM pin numbers)
| Motor | BTS7960 pin | RPi GPIO (BCM) |
|---|---|---|
| Left | RPWM | 18 |
| Left | LPWM | 19 |
| Left | R_EN | 20 |
| Left | L_EN | 21 |
| Right | RPWM | 12 |
| Right | LPWM | 13 |
| Right | R_EN | 5 |
| Right | L_EN | 6 |

BTS7960 logic VCC is tied to the Pi 3.3 V pin. **Pi and motor supply must share a common ground.**

![Wiring photo](docs/images/wiring.jpg)

## How it works
1. Camera frame is captured with picamera2.
2. **Classical mode:** BGR threshold -> largest contour -> centroid x -> left / straight / right.
3. **CNN mode:** frame resized to 64x64, normalised, classified by `LFRNet` (`model.py`),
   exported to ONNX and run on the Pi (`infer_pi.py`).

## Repo layout
```
model.py          CNN definition
train.py          training + ONNX export (needs dataset/{left,right,straight}/)
infer_pi.py       on-robot inference + motor control
requirements.txt
docs/images/      photos and diagrams
```

## Running
```bash
pip install -r requirements.txt
python train.py        # on a laptop, once you have a dataset
python infer_pi.py     # on the Pi, with lfr.onnx present
```

## Next steps
- Collect a labelled dataset, using the classical pipeline as an auto-labeller
- Compare CNN vs classical pipeline (lap time, failures, FPS)
- Quantise (INT8) if inference is too slow on the Pi

## License
MIT
