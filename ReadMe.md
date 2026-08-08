# Multi-Camera Person Tracking & Re-Identification

A learning project exploring multi-object tracking algorithms and cross-camera
person re-identification — recognizing the same person as they move from one
camera's field of view into another's.

![Cross-camera re-identification](asset/reid_diagram.png)

## Why This Matters

Any real-world camera network — a mall, a campus, a warehouse floor — has
non-overlapping or partially-overlapping fields of view. A tracker alone only
keeps identity consistent _within_ one camera's continuous footage; the
moment someone leaves that view, their identity is lost unless something
re-establishes it elsewhere. That "something" is Re-ID, and it's what turns a
set of independent camera feeds into a single coherent system that can
reason about a person across space, not just across frames.

**Applications:**

- **Security & surveillance** — following a person of interest across a
  building's camera network without manual review of every feed.
- **Retail analytics** — customer path/dwell-time tracking across a store's
  camera zones for layout and staffing decisions.
- **Smart city / traffic** — pedestrian flow analysis across intersections
  or transit hubs.
- **Sports analytics** — maintaining player identity across broadcast camera
  cuts and angle changes.
- **Loss prevention** — linking a flagged individual's movement across
  multiple store cameras in real time.

## 1. Single-Camera Tracking

Five tracker variants were implemented and compared to understand the
tradeoffs between motion-only, appearance-integrated, and hybrid approaches
to identity association — particularly how each handles occlusion and a
person leaving/re-entering the frame.

| Tracker                                | Approach                                                                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **ByteTrack**                          | Motion-only, two-stage (high/low confidence) association                                                                                         |
| **ByteTrack + ReID** (`bytetrack_reid`) | ByteTrack + a manual OSNet-based gallery layered on top, so identity survives full occlusion/re-entry, not just ByteTrack's short `track_buffer` |
| **DeepSORT**                           | Motion (Kalman/IoU) + appearance embedding cascade matching                                                                                      |
| **BoT-SORT**                           | ByteTrack + camera motion compensation + optional ReID branch                                                                                    |
| **OC-SORT**                            | Robust motion-only Kalman formulation, no appearance model                                                                                       |

**Run:**

```bash
python main.py --tracker <bytetrack|bytetrack_reid|botsort|deepsort|ocsort>
```

## 2. Cross-Camera Re-Identification

**Goal:** assign the same global ID to a person as they cross from one
camera's view into another's — since local track IDs reset independently
per camera, this needs a separate appearance-matching layer on top of
whichever tracker is running.

**Method:**

- Each camera runs its own tracker instance, producing local track IDs.
- A shared **OSNet** (Market1501, multi-source domain generalization
  weights) extracts an appearance embedding for every detected box.
- A `CrossCameraGallery` matches new tracks against currently out-of-view
  ("inactive") global identities via cosine similarity, assigning a shared
  global ID on a match or creating a new one otherwise.
- **Boundary-touch filtering**: a brand-new detection still touching the
  frame edge (e.g. a foot or shoulder entering frame) is not embedded or
  assigned an ID until fully inside the frame — this prevents a partial-body
  embedding from polluting the gallery and causing false non-matches.

AGW is also implemented using fastreid. Use the below command to run the program. Replace osnet with 
agw to use AGW feature extractor. 

fast-reid will have to be installed for using AGW extractor. Run below commands to install fastreid

```
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
pip install -r docs/requirements.txt
```

```
export PYTHONPATH=$PYTHONPATH:</absolute/path/to/fast-reid>
```

**Note:** fastreid is not installed as a package in python so the official repo have to be used for its implementation. export the PYTHONPATH in ubuntu to use it outside the fastreid repo.

Also if you are using python>3.10, collections library is not imported directly from fastreid, it is changed to fastreid.abc so a mapping has been added in the code to fix that. If you have python<=3.10, that part can be removed from main.py

**Run:**
```bash
pip install -r requirements.txt
python main.py --tracker bytetrack_reid --reid osnet
```

You'll be prompted for the number of camera/video sources and a path (webcam
index, video file, or RTSP URL) for each.

## Current Limitation

OSNet's appearance similarity degrades across different camera
angles/lighting conditions, which limits cross-camera matching accuracy.
This is the open problem going into the next phase — testing stronger
ReID backbones (AGW, TransReID, CLIP-ReID) that generalize better to
viewpoint and domain shift.


**Additonal checks that can be added to improve Re-Identification (Future Works):**
- Geometric/position consistency check to replace that lost constraint for overlapping views — either a homography between the two camera views (mark corresponding ground-plane points once, then check whether two simultaneous detections map to a plausible shared location) or, for non-overlapping setups, defined exit/entry zones per camera with a minimum plausible transit time between them.
- Color Histogram Similarity, a simple HSV color histogram comparison (cv2.compareHist) is extremely sensitive to exact clothing color
- Multi-frame confirmation — don't commit to a match on a single frame's embedding. Require the same candidate to win the match for 2-3 consecutive frames before finalizing the global ID assignment

## Requirements

See `requirements.txt`.
