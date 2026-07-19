import torch
from ultralytics import YOLO
import cv2
import numpy as np
from torchreid.utils import FeatureExtractor

SIM_THRESHOLD = 0.7
MAX_GALLERY_AGE = 300
EMA_ALPHA = 0.9

extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path="/home/sabih-shah/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth",
    image_size=(640, 480),
    device="cuda" if torch.cuda.is_available() else "cpu"
)

model = YOLO("yolov8m.pt")
# model2 = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(1)

gallery = {}
track_to_global = {}
next_global_id = 0
frame_idx = 0

def cosine_sim(a, b):
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-6))

def match_or_create(embedding):
    global next_global_id
    best_id, best_sim = None, 0.0
    for gid, entry in gallery.items():
        if entry["active"]:
            continue
        sim = cosine_sim(embedding, entry["embedding"])
        if sim > best_sim:
            best_sim, best_id = sim, gid
 
    if best_id is not None and best_sim >= SIM_THRESHOLD:
        return best_id
 
    gid = next_global_id
    next_global_id += 1
    return gid
 
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    result = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)[0]
    active_track_ids = set()
 
    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
 
        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
 
            emb = extractor([crop]).cpu().numpy()[0]
            active_track_ids.add(tid)
 
            if tid in track_to_global:
                gid = track_to_global[tid]
                gallery[gid]["embedding"] = EMA_ALPHA * gallery[gid]["embedding"] + (1 - EMA_ALPHA) * emb
            else:
                gid = match_or_create(emb)
                track_to_global[tid] = gid
                gallery[gid] = {"embedding": emb, "last_seen": frame_idx, "active": True}
 
            gallery[gid]["active"] = True
            gallery[gid]["last_seen"] = frame_idx
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {gid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
    for tid in list(track_to_global.keys()):
        if tid not in active_track_ids:
            gallery[track_to_global[tid]]["active"] = False
            del track_to_global[tid]
 
    for gid in list(gallery.keys()):
        if not gallery[gid]["active"] and frame_idx - gallery[gid]["last_seen"] > MAX_GALLERY_AGE:
            del gallery[gid]
 
    cv2.imshow("Single Camera ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    frame_idx += 1
 
cap.release()
cv2.destroyAllWindows()