import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="botsort.yaml",
        verbose=False,
    )[0]

    annotated = result.plot()
    cv2.imshow("BoT-SORT ReID", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()