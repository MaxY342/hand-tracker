import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pyautogui
import time
import os
import numpy as np
import math

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# cv2 initialization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Mediapipe initialization
model_path = r"ai_models\hand_landmarker.task"
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode
latest_result = None
latest_timestamp = -1
start_time = time.monotonic()
last_timestamp = -1

# Mediapipe callback function
def on_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, latest_timestamp

    latest_result = result
    latest_timestamp = timestamp_ms

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

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

# Virtual box calibration initialization
calibration_points = []
homography_matrix = None
SCREEN_W, SCREEN_H = pyautogui.size()
SMOOTHING = 0.25

def get_calibration_point(finger_landmark):
    global homography_matrix

    calibration_points.append([finger_landmark.x, finger_landmark.y])

    if len(calibration_points) == 4:
        src = np.array(calibration_points, dtype=np.float32)

        # Calibration order: top-left, top-right, bottom-right, bottom-left (-1 for zero indexing)
        dst = np.array([
            [0, 0],
            [SCREEN_W-1, 0],
            [SCREEN_W-1, SCREEN_H-1],
            [0, SCREEN_H-1]
        ], dtype=np.float32)

        # Mapping oriented capture to flat 2-D screen, expects (4, 2) ndarray format, returns (3, 3)
        homography_matrix = cv2.getPerspectiveTransform(src, dst)

def get_cursor_position(finger_landmark):
    point = np.array(
        [[[finger_landmark.x, finger_landmark.y]]],
        dtype=np.float32
    )

    # Apply point to homography matrix, expects (N, 1, 2) ndarray point and (3, 3) ndarray for m, returns (N, 1, 2)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)

    screen_x, screen_y = transformed_point[0][0]

    # Keep within screen bounds
    screen_x = np.clip(screen_x, 0, SCREEN_W - 1)
    screen_y = np.clip(screen_y, 0, SCREEN_H - 1)

    return screen_x, screen_y

def main():
    global last_timestamp
    smooth_x, smooth_y = pyautogui.position()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            attempt = 0
            success, frame = cap.read()

            while not success and attempt < 5:
                time.sleep(0.2)
                success, frame = cap.read()
                attempt += 1
            if not success:
                print("error finding camera capture")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
    
            # Convert BGR frame to RGB usable by mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )
            timestamp_ms = max(timestamp_ms, last_timestamp + 1)
            last_timestamp = timestamp_ms
            landmarker.detect_async(mp_image, timestamp_ms)
    
            if latest_result is not None:
    
                if latest_result.hand_landmarks and homography_matrix is not None:
                    index_tip = latest_result.hand_landmarks[0][8]
                    thumb_tip = latest_result.hand_landmarks[0][4]
                    middle_tip = latest_result.hand_landmarks[0][12]

                    # Cursor movement
                    target_x, target_y = get_cursor_position(index_tip)

                    smooth_x += (target_x - smooth_x) * SMOOTHING
                    smooth_y += (target_y - smooth_y) * SMOOTHING
                    
                    pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0)

                    # Left-click
                    distance = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
                    touch_threshold = 0.05
                    if distance < touch_threshold:
                        pyautogui.click()

                # Landmark visualization
                
                for hand_landmarks in latest_result.hand_landmarks:
                    # Draw the landmark points
                    for landmark in hand_landmarks:
                        # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
                        pixel_x = int(landmark.x * w)
                        pixel_y = int(landmark.y * h)
                        cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), -1)
                        
                    # Draw landmark lines
                    for connection in HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        start_point = hand_landmarks[start_idx]
                        end_point = hand_landmarks[end_idx]
                        
                        pt1 = (int(start_point.x * w), int(start_point.y * h))
                        pt2 = (int(end_point.x * w), int(end_point.y * h))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                
                    
            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return

            elif key == 32 and homography_matrix is None:
                if latest_result and latest_result.hand_landmarks:
                    get_calibration_point(latest_result.hand_landmarks[0][8])
                    print("Got calibration point")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()