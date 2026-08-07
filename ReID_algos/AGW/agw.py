import sys
import torch
import cv2
import torch

FASTREID_ROOT = "ReID_algos/fast-reid"
sys.path.append(FASTREID_ROOT)

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.utils.file_io import PathManager

class AGWExtrcator:
    def __init__(self, config_file, weights_path, device="cuda"):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)
        self.input_size = cfg.INPUT.SIZE_TEST

    def _preprocess_one(self, crop_bgr):
        img = crop_bgr[:, :, ::-1]  # BGR -> RGB
        img = cv2.resize(img, tuple(self.input_size[::-1]), interpolation=cv2.INTER_CUBIC)
        return torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

    def __call__(self, crops):
        batch = torch.stack([self._preprocess_one(c) for c in crops])
        return self.predictor(batch)