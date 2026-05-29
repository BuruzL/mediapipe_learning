import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

timestamp_ms = 0

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape

            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            wrist = hand_landmarks[0]
            index_tip = hand_landmarks[8]

            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            cv2.line(frame, (wrist_x, wrist_y), (index_x, index_y), (255, 0, 0), 2)

    cv2.imshow("MediaPipe Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()