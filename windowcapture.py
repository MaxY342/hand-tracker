
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import mss
import numpy as np
import pygetwindow as gw

# Find scrcpy window
windows = gw.getWindowsWithTitle("HandTracking")

if not windows:
    raise RuntimeError("scrcpy window not found")

window = windows[0]

with mss.mss() as sct:
    while True:
        # Get current window position
        region = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height
        }

        # Capture desktop region
        screenshot = sct.grab(region)

        # Convert BGRA screenshot to OpenCV BGR
        frame = cv2.cvtColor(
            np.asarray(screenshot),
            cv2.COLOR_BGRA2BGR
        )

        frame = cv2.flip(frame, 1)

        # Frame is now usable with OpenCV
        cv2.imshow("OpenCV Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
