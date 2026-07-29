import argparse
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from boxmot import OCSORT
from torchreid.utils import FeatureExtractor

from trackers import (
    track_ultralytics, track_deepsort, track_ocsort,
    ByteTrackReID, CrossCameraGallery, DEFAULT_YAMLS,
)

MODEL_PATH = "weights/yolov8m.pt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracker",
        choices=["bytetrack", "bytetrack_reid", "botsort", "deepsort", "ocsort"],
        required=True,
        help="Tracker to run identically on every camera/video",
    )
    return parser.parse_args()


def is_touching_boundary(x1, y1, x2, y2, frame_shape, margin=0):
    h, w = frame_shape[:2]
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


def resolve_source(src):
    """Webcam index strings ('0','1') become ints; everything else
    (file paths, rtsp://, http:// links) is passed through as-is."""
    return int(src) if src.isdigit() else src


def prompt_sources():
    n = int(input("How many cameras/videos do you want to run: ").strip())
    sources = []
    for i in range(n):
        src = input(f"Source {i + 1} (camera index / rtsp link / video path): ").strip()
        sources.append(resolve_source(src))
    return sources


def build_tracker(tracker_name, extractor):
    """Returns (model_or_detector, tracker_or_None) - one fresh instance per camera."""
    if tracker_name in DEFAULT_YAMLS:
        return YOLO(MODEL_PATH), None
    if tracker_name == "bytetrack_reid":
        model = YOLO(MODEL_PATH)
        return model, ByteTrackReID(model, extractor)
    if tracker_name == "deepsort":
        detector = YOLO(MODEL_PATH)
        tracker = DeepSort(
            max_age=30, n_init=5, nms_max_overlap=0.7,
            max_cosine_distance=0.2, nn_budget=100,
            embedder="torchreid", embedder_model_name="osnet_x1_0",
            embedder_wts="weights/osnet_x1_0_imagenet.pth", half=True, embedder_gpu=True,
        )
        return detector, tracker
    if tracker_name == "ocsort":
        detector = YOLO(MODEL_PATH)
        tracker = OCSORT(det_thresh=0.7, max_age=30, min_hits=3, asso_threshold=0.5)
        return detector, tracker
    raise ValueError(tracker_name)


def get_boxes(tracker_name, model_or_detector, tracker, frame):
    if tracker_name in DEFAULT_YAMLS:
        return track_ultralytics(model_or_detector, frame, tracker_name)
    if tracker_name == "bytetrack_reid":
        return tracker.update(frame)
    if tracker_name == "deepsort":
        return track_deepsort(model_or_detector, tracker, frame)
    return track_ocsort(model_or_detector, tracker, frame)


def main():
    args = parse_args()
    sources = prompt_sources()
    num_cams = len(sources)
    cam_ids = [f"cam{i + 1}" for i in range(num_cams)]

    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path="weights/osnet_x1_0_imagenet.pth",
        device="cuda",
    )
    gallery = CrossCameraGallery()

    caps = []
    for src in sources:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source={src!r}")
        caps.append(cap)

    trackers = [build_tracker(args.tracker, extractor) for _ in range(num_cams)]

    def resolve(camera_id, frame, boxes, active_keys):
        resolved = []
        for x1, y1, x2, y2, local_id in boxes:
            key = (camera_id, local_id)
            # brand-new track (no global id yet) still touching the edge - likely a
            # partial-body (foot/shoulder) detection, defer until it's fully inside
            if key not in gallery.local_to_global and is_touching_boundary(x1, y1, x2, y2, frame.shape):
                continue
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            emb = extractor([crop]).cpu().numpy()[0]
            gid = gallery.get_global_id(camera_id, local_id, emb)
            active_keys.add((camera_id, local_id))
            resolved.append((x1, y1, x2, y2, gid))
        return resolved

    while True:
        frames = []
        ok = True
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                ok = False
                break
            frames.append(cv2.resize(frame, (640, 480)))
        if not ok:
            break

        raw_boxes = [
            get_boxes(args.tracker, trackers[i][0], trackers[i][1], frames[i])
            for i in range(num_cams)
        ]

        # build this frame's raw active-key set BEFORE any matching, then prune first -
        # this is what lets a same-frame cam-exit/cam-entry crossing still match
        active_keys = set()
        for cam_id, boxes in zip(cam_ids, raw_boxes):
            for (*_, tid) in boxes:
                active_keys.add((cam_id, tid))
        gallery.prune_inactive(active_keys)

        resolved_boxes = [
            resolve(cam_ids[i], frames[i], raw_boxes[i], active_keys)
            for i in range(num_cams)
        ]
        gallery.purge_stale()

        for frame, boxes in zip(frames, resolved_boxes):
            for x1, y1, x2, y2, gid in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(f"{gid}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cv2.putText(frame, f"{gid}", (cx - tw//2, cy - th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined = cv2.hconcat(frames)
        cv2.namedWindow(f"{args.tracker} - {num_cams} sources", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{args.tracker} - {num_cams} sources", 640, 480)
        cv2.imshow(f"{args.tracker} - {num_cams} sources", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()