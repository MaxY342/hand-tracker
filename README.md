# hand-tracker
mouse control functionalities with hand movements
# initialization
pip install required libraries through: pip install -r requirements.txt
# video-capture
change cv2.VideoCapture(#) to wanted video capture device index (top to bottom, zero-indexed)
# window-capture
scrcpy must be installed and used
initial scrcpy command: scrcpy --max-size=1280 --max-fps=60 --video-bit-rate=16M --no-audio --window-title="HandTracking" (used before running python file)
# test.py and temp.py
scrcpy v5.0.1 must be installed
scrcpy folder must be added to path for adb and scrcpy commands through subprocess