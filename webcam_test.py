import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Try changing VideoCapture(0) to VideoCapture(1).")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()