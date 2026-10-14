"""Runs on the Raspberry Pi: picamera2 -> ONNX Runtime -> BTS7960 motors."""
import time
import cv2
import numpy as np
import onnxruntime as ort
import RPi.GPIO as GPIO
from picamera2 import Picamera2

from model import IMG_SIZE, CLASSES

BASE_SPEED = 40   # % duty
TURN_SPEED = 25
CONF_MIN = 0.6    

# BCM pins (BTS7960, one driver per motor)
LEFT = dict(RPWM=18, LPWM=19, R_EN=20, L_EN=21)
RIGHT = dict(RPWM=12, LPWM=13, R_EN=5, L_EN=6)

GPIO.setmode(GPIO.BCM)
pwms = {}
for side in (LEFT, RIGHT):
    for p in side.values():
        GPIO.setup(p, GPIO.OUT)
    GPIO.output(side["R_EN"], GPIO.HIGH)
    GPIO.output(side["L_EN"], GPIO.HIGH)
    pwms[side["RPWM"]] = GPIO.PWM(side["RPWM"], 1000)
    pwms[side["LPWM"]] = GPIO.PWM(side["LPWM"], 1000)
    pwms[side["RPWM"]].start(0)
    pwms[side["LPWM"]].start(0)


def drive(left, right):
    """Duty cycles 0-100, forward only (swap RPWM/LPWM roles if your motors spin backwards)."""
    pwms[LEFT["RPWM"]].ChangeDutyCycle(left)
    pwms[RIGHT["RPWM"]].ChangeDutyCycle(right)


ACTIONS = {
    "straight": (BASE_SPEED, BASE_SPEED),
    "left": (TURN_SPEED, BASE_SPEED),
    "right": (BASE_SPEED, TURN_SPEED),
}


def preprocess(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5                      # same Normalize as training
    x = np.transpose(x, (2, 0, 1))           # HWC -> CHW
    return np.expand_dims(x, 0)              # add batch dim


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


sess = ort.InferenceSession("lfr.onnx", providers=["CPUExecutionProvider"])
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (320, 240), "format": "RGB888"}))
cam.start()
time.sleep(1)

try:
    while True:
        frame = cam.capture_array()  # picamera2 "RGB888" is BGR byte order, hence preprocess() converts
        logits = sess.run(None, {"input": preprocess(frame)})[0][0]
        probs = softmax(logits)
        k = int(probs.argmax())
        if probs[k] < CONF_MIN:
            drive(0, 0)
        else:
            drive(*ACTIONS[CLASSES[k]])
except KeyboardInterrupt:
    pass
finally:
    drive(0, 0)
    for p in pwms.values():
        p.stop()
    cam.stop()
    GPIO.cleanup()
