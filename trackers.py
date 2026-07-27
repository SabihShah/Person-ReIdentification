import numpy as np


class CrossCameraGallery:
    """Tracker-agnostic global gallery. Key local tracks by (camera_id, local_id)
    since local IDs reset independently per camera/tracker instance."""

    def __init__(self, sim_threshold=0.5, max_gallery_age=300, ema_alpha=0.9):
        self.gallery = {}          # global_id -> {"embedding", "last_seen", "active"}
        self.local_to_global = {}  # (camera_id, local_id) -> global_id
        self.next_global_id = 0
        self.frame_idx = 0
        self.sim_threshold = sim_threshold
        self.max_gallery_age = max_gallery_age
        self.ema_alpha = ema_alpha

    @staticmethod
    def _cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def _match_or_create(self, embedding):
        best_id, best_sim = None, 0.0
        for gid, entry in self.gallery.items():
            if entry["active"]:
                continue
            sim = self._cosine_sim(embedding, entry["embedding"])
            if sim > best_sim:
                best_sim, best_id = sim, gid
        if best_id is not None and best_sim >= self.sim_threshold:
            return best_id
        if best_id is not None:
            print(f"[cross-cam gallery] best candidate sim={best_sim:.3f} (threshold={self.sim_threshold}) - new ID assigned instead of matching")
        gid = self.next_global_id
        self.next_global_id += 1
        return gid

    def get_global_id(self, camera_id, local_id, embedding):
        key = (camera_id, local_id)
        if key in self.local_to_global:
            gid = self.local_to_global[key]
            self.gallery[gid]["embedding"] = (
                self.ema_alpha * self.gallery[gid]["embedding"] + (1 - self.ema_alpha) * embedding
            )
        else:
            gid = self._match_or_create(embedding)
            self.local_to_global[key] = gid
            self.gallery[gid] = {"embedding": embedding, "last_seen": self.frame_idx, "active": True}
        self.gallery[gid]["active"] = True
        self.gallery[gid]["last_seen"] = self.frame_idx
        return gid

    def prune_inactive(self, active_keys_this_frame):
        """Call BEFORE resolving this frame's detections, using the raw set of
        (camera_id, local_id) keys seen this frame. Flips anything missing to
        inactive first, so a same-frame disappear-cam1/appear-cam2 crossing
        can still match against it."""
        for key in list(self.local_to_global.keys()):
            if key not in active_keys_this_frame:
                self.gallery[self.local_to_global[key]]["active"] = False
                del self.local_to_global[key]

    def purge_stale(self):
        """Call once per frame after resolving, to drop very old inactive entries."""
        for gid in list(self.gallery.keys()):
            if not self.gallery[gid]["active"] and self.frame_idx - self.gallery[gid]["last_seen"] > self.max_gallery_age:
                del self.gallery[gid]
        self.frame_idx += 1


class ByteTrackReID:
    """ByteTrack + a manual appearance gallery on top, so identities survive
    full occlusion/exit-reentry instead of just ByteTrack's short track_buffer."""

    def __init__(self, model, extractor, sim_threshold=0.7, max_gallery_age=300, ema_alpha=0.9):
        self.model = model
        self.extractor = extractor
        self.sim_threshold = sim_threshold
        self.max_gallery_age = max_gallery_age
        self.ema_alpha = ema_alpha

        self.gallery = {}          # global_id -> {"embedding", "last_seen", "active"}
        self.track_to_global = {}  # bytetrack_id -> global_id
        self.next_global_id = 0
        self.frame_idx = 0

    @staticmethod
    def _cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def _match_or_create(self, embedding):
        best_id, best_sim = None, 0.0
        for gid, entry in self.gallery.items():
            if entry["active"]:
                continue
            sim = self._cosine_sim(embedding, entry["embedding"])
            if sim > best_sim:
                best_sim, best_id = sim, gid

        if best_id is not None and best_sim >= self.sim_threshold:
            return best_id

        gid = self.next_global_id
        self.next_global_id += 1
        return gid

    def update(self, frame):
        result = self.model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)[0]
        active_track_ids = set()
        boxes = []

        if result.boxes.id is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid in zip(xyxy, track_ids):
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                crop = frame[max(0, y1i):y2i, max(0, x1i):x2i]
                if crop.size == 0:
                    continue

                emb = self.extractor([crop]).cpu().numpy()[0]
                active_track_ids.add(tid)

                if tid in self.track_to_global:
                    gid = self.track_to_global[tid]
                    self.gallery[gid]["embedding"] = (
                        self.ema_alpha * self.gallery[gid]["embedding"] + (1 - self.ema_alpha) * emb
                    )
                else:
                    gid = self._match_or_create(emb)
                    self.track_to_global[tid] = gid
                    self.gallery[gid] = {"embedding": emb, "last_seen": self.frame_idx, "active": True}

                self.gallery[gid]["active"] = True
                self.gallery[gid]["last_seen"] = self.frame_idx
                boxes.append((x1i, y1i, x2i, y2i, gid))

        for tid in list(self.track_to_global.keys()):
            if tid not in active_track_ids:
                self.gallery[self.track_to_global[tid]]["active"] = False
                del self.track_to_global[tid]

        for gid in list(self.gallery.keys()):
            if not self.gallery[gid]["active"] and self.frame_idx - self.gallery[gid]["last_seen"] > self.max_gallery_age:
                del self.gallery[gid]

        self.frame_idx += 1
        return boxes


DEFAULT_YAMLS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort_reid.yaml",
}


def track_ultralytics(model, frame, tracker_name, custom_yaml=None):
    """Shared by ByteTrack and BoT-SORT. Pass custom_yaml to override the default config."""
    tracker_yaml = custom_yaml or DEFAULT_YAMLS[tracker_name]
    result = model.track(frame, persist=True, classes=[0], tracker=tracker_yaml, verbose=False)[0]
    boxes = []
    if result.boxes.id is not None:
        xyxy = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            boxes.append((int(x1), int(y1), int(x2), int(y2), int(tid)))
    return boxes


def track_deepsort(detector, tracker, frame, conf_thresh=0.5):
    results = detector(frame, classes=[0], conf=conf_thresh, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    boxes = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        l, t, r, b = map(int, track.to_ltrb())
        boxes.append((l, t, r, b, int(track.track_id)))
    return boxes


def track_ocsort(detector, tracker, frame):
    results = detector(frame, classes=[0], verbose=False)[0]
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = float(box.cls[0])
        dets.append([x1, y1, x2, y2, conf, cls])
    dets = np.array(dets) if dets else np.empty((0, 6))

    tracks = tracker.update(dets, frame)
    boxes = []
    for t in tracks:
        x1, y1, x2, y2, tid = t[:5]
        boxes.append((int(x1), int(y1), int(x2), int(y2), int(tid)))
    return boxes