"""Nvidia/TensorRT inference backend via ultralytics."""

import numpy as np

from detector import COCO_CLASSES, COCO_NAMES


class NvidiaBackend:
    """Nvidia GPU inference backend using ultralytics YOLO (context manager).

    Parameters
    ----------
    model_path : str
        Path to the .engine (TensorRT) or .pt model file.
    conf_thresh : float
        Minimum confidence score for detections.
    """

    def __init__(self, model_path: str, conf_thresh: float):
        self._model_path = model_path
        self._conf_thresh = conf_thresh
        self._model = None
        self._gpu_poller = None
        self.input_height: int = 640
        self.input_width: int = 640
        self.max_tiles: int | None = None  # no cap; pass all tiles in one predict call

    def __enter__(self):
        from ultralytics import YOLO
        print(f"Loading model: {self._model_path}")
        self._model = YOLO(self._model_path)

        imgsz = self._model.overrides.get("imgsz", 640)
        if isinstance(imgsz, (list, tuple)):
            self.input_height, self.input_width = int(imgsz[0]), int(imgsz[1])
        else:
            self.input_height = self.input_width = int(imgsz)
        print(f"Model input: {self.input_width}x{self.input_height}")

        from monitor import GpuStatsPoller
        self._gpu_poller = GpuStatsPoller()
        return self

    def __exit__(self, *exc):
        self._model = None

    def infer(self, images: list) -> list:
        """Run inference on a list of images.

        ultralytics handles letterboxing internally and returns coords in each
        image's own pixel space.

        Parameters
        ----------
        images : list of np.ndarray

        Returns
        -------
        list of list
            One detection list per image; each detection is [x1, y1, x2, y2, conf, class_id].
        """
        if not images:
            return []

        results = self._model.predict(images, verbose=False, conf=self._conf_thresh)

        all_dets = []
        for r in results:
            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                data = r.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, class_id]
                for row in data:
                    x1, y1, x2, y2, conf, class_id = row
                    class_id = int(class_id)
                    class_name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}"
                    if COCO_CLASSES.get(class_name, 0) == 0:
                        continue
                    dets.append([float(x1), float(y1), float(x2), float(y2), float(conf), class_id])
            all_dets.append(dets)
        return all_dets

    def get_hw_stats(self) -> dict:
        if self._gpu_poller is None:
            return {}
        return self._gpu_poller.get()
