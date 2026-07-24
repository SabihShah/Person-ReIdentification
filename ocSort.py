import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import OcSort

model = YOLO("yolov8n.pt")

tracker = OcSort(
    det_thresh=0.5,     # min detection confidence to consider
    max_age=30,         # frames a lost track survives on Kalman-only prediction
    min_hits=3,         # frames needed before a track is confirmed
    iou_threshold=0.3,  # IoU gating for association
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], verbose=False)[0]

    # boxmot expects an (N, 6) array: x1, y1, x2, y2, conf, cls
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = float(box.cls[0])
        dets.append([x1, y1, x2, y2, conf, cls])
    dets = np.array(dets) if dets else np.empty((0, 6))

    tracks = tracker.update(dets, frame)  # returns (M, 7): x1,y1,x2,y2,id,conf,cls

    for t in tracks:
        x1, y1, x2, y2, tid = t[:5]
        x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("OC-SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()