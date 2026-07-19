import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8m.pt")

tracker = DeepSort(
    max_iou_distance=0.5,        # IoU gating threshold (1 - IoU)
    max_age=60,                  # frames to keep a lost track alive via Kalman-only prediction
    n_init=3,                    # frames needed before a track is "confirmed"
    max_cosine_distance=0.5,     # ReID similarity threshold for matching
    nn_budget=100,               # max embedding stored per track id (gallery size)
    embedder="torchreid",        
    embedder_model_name="osnet_x1_0",
    embedder_wts="/home/sabih-shah/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth",
    half=True,
    embedder_gpu=True
    )

cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    results = model(frame, classes=[0], verbose=False, imgsz=(640, 480))[0]
 
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, "person"))
 
    tracks = tracker.update_tracks(detections, frame=frame)
 
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
    cv2.imshow("DeepSORT ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()