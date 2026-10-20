
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import mss
import numpy as np
import pygetwindow as gw
import threading
import time

# Find scrcpy window
windows = gw.getWindowsWithTitle("HandTracking")

if not windows:
    raise RuntimeError("scrcpy window not found")

window = windows[0]

# Mediapipe initialization
model_path = r"ai_models\hand_landmarker.task"
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode
latest_result = None
latest_timestamp = -1
result_lock = threading.Lock()
start_time = time.monotonic()
last_timestamp = -1
last_used_result = -1

# Mediapipe callback function
def on_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, latest_timestamp

    with result_lock:
        latest_result = result
        latest_timestamp = timestamp_ms

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result)

HAND_CONNECTIONS = [
    # Wrist to thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle joints connection
    (5, 9), (9, 13), (13, 17)
]

with mss.mss() as sct, HandLandmarker.create_from_options(options) as landmarker:
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
        h, w, _ = frame.shape

        # Convert frame to RGB usable by mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )
        timestamp_ms = max(timestamp_ms, last_timestamp + 1)
        last_timestamp = timestamp_ms
        landmarker.detect_async(mp_image, timestamp_ms)

        with result_lock:
            result = latest_result
            result_ts = latest_timestamp

        if result is not None and result_ts > last_used_result:
            last_used_result = result_ts

            if result.hand_landmarks:
                 for hand_landmarks in result.hand_landmarks:
                    # Draw the 21 landmark points
                    for landmark in hand_landmarks:
                        # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
                        pixel_x = int(landmark.x * w)
                        pixel_y = int(landmark.y * h)
                        cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), -1)
                        
                    # Draw the lines connecting the landmarks
                    for connection in HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        start_point = hand_landmarks[start_idx]
                        end_point = hand_landmarks[end_idx]
                        
                        pt1 = (int(start_point.x * w), int(start_point.y * h))
                        pt2 = (int(end_point.x * w), int(end_point.y * h))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Frame is now usable with OpenCV
        cv2.imshow("OpenCV Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        

cv2.destroyAllWindows()
