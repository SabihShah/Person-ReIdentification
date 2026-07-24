import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from deep_sort_realtime.deepsort_tracker import DeepSort
# from boxmot import OCSORT

from trackers import track_ultralytics, track_deepsort, track_ocsort, ByteTrackReID, DEFAULT_YAMLS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracker",
        choices=["bytetrack", "bytetrack_reid", "botsort", "deepsort", "ocSort"],
        required=True,
        help="Choose tracker, bytetrack is just tracking, no reidentification"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cap  = cv2.VideoCapture(0)

    detector = None
    tracker = None
    model = None
    tracker_yaml = None

    if args.tracker in DEFAULT_YAMLS:
        model = YOLO("yolov8m.pt")
    elif args.tracker == "bytetrack_reid":
        yolo_model = YOLO("yolov8m.pt")
        extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="/home/hassan/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth",
            device="cuda"
        )
        tracker = ByteTrackReID(yolo_model, extractor, 
                                sim_threshold = 0.5,         # HIGHER = STRICTER
                                max_gallery_age = 300,
                                ema_alpha = 0.7)
    elif args.tracker == "deepsort":
        detector = YOLO("yolov8m.pt")
        tracker = DeepSort(
            max_iou_distance=0.5,        # IoU gating threshold (1 - IoU)
            max_age=200,                 # frames to keep a lost track alive via Kalman-only prediction
            n_init=3,                    # frames needed before a track is "confirmed"
            max_cosine_distance=0.5,     # ReID similarity threshold for matching
            nn_budget=100,               # max embedding stored per track id (gallery size)
            embedder="torchreid",        
            embedder_model_name="osnet_x1_0",
            embedder_wts="/home/hassan/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth",
            # embedder_wts="osnet_ain_ms_d_c.pth",
            half=True,
            embedder_gpu=True
        )
    # elif args.tracker == "ocsort":
    #     detector = YOLO("yolov8m.pt")
    #     tracker = OCSORT(
    #         det_thresh=0.5,     # min detection confidence to consider
    #         max_age=30,         # frames a lost track survives on Kalman-only prediction
    #         min_hits=3,         # frames needed before a track is confirmed
    #         iou_threshold=0.3,  # IoU gating for association
    #     )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.tracker in DEFAULT_YAMLS:
            boxes = track_ultralytics(model, frame, args.tracker, "botsort-reid.yaml")
        elif args.tracker == "bytetrack_reid":
            boxes = tracker.update(frame)
        elif args.tracker == "deepsort":
            boxes = track_deepsort(detector, tracker, frame)
        else:
            boxes = track_ocsort(detector, tracker, frame)

        for x1, y1, x2, y2, tid in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(args.tracker, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()