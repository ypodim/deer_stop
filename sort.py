"""Pure-Python SORT tracker.

Based on: Bewley et al., "Simple Online and Realtime Tracking" (2016)
Requires: filterpy, scipy
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def _bbox_to_z(bbox):
    """Convert [x1, y1, x2, y2] to centre/scale/ratio state vector [cx, cy, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([
        bbox[0] + w / 2.0,  # cx
        bbox[1] + h / 2.0,  # cy
        w * h,               # scale (area)
        w / float(h),        # aspect ratio
    ]).reshape((4, 1))


def _z_to_bbox(x):
    """Convert state vector [cx, cy, s, r, ...] back to [x1, y1, x2, y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]).reshape((1, 4))


def _iou_matrix(a, b):
    """Compute IoU between every pair in a (N×4) and b (M×4); returns N×M matrix."""
    n, m = len(a), len(b)
    iou = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            ix1 = max(a[i, 0], b[j, 0])
            iy1 = max(a[i, 1], b[j, 1])
            ix2 = min(a[i, 2], b[j, 2])
            iy2 = min(a[i, 3], b[j, 3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter == 0.0:
                continue
            union = ((a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1]) +
                     (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1]) - inter)
            iou[i, j] = inter / union
    return iou


class _KalmanBoxTracker:
    """Tracks a single bounding box with a constant-velocity Kalman filter.

    State: [cx, cy, s, r, vcx, vcy, vs]  (s=area, r=aspect ratio)
    Measurement: [cx, cy, s, r]
    """

    _count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition: position += velocity each step
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = 1.0
        self.kf.F[1, 5] = 1.0
        self.kf.F[2, 6] = 1.0

        # Measurement picks out [cx, cy, s, r] from state
        self.kf.H = np.eye(4, 7)

        # Measurement noise: higher uncertainty on scale/ratio
        self.kf.R[2:, 2:] *= 10.0

        # Initial covariance: high uncertainty on velocity components
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process noise: low velocity noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = _bbox_to_z(bbox)

        self.id = _KalmanBoxTracker._count + 1
        _KalmanBoxTracker._count += 1

        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        # Prevent negative scale
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_bbox(self.kf.x)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(_bbox_to_z(bbox))

    def get_state(self):
        return _z_to_bbox(self.kf.x)


class Sort:
    """SORT multi-object tracker.

    Args:
        max_age:       frames a track survives without a detection match
        min_hits:      detections required before a track is reported
        iou_threshold: minimum IoU to consider a detection-track match
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: list[_KalmanBoxTracker] = []
        self._frame = 0

    def update(self, dets: np.ndarray = np.empty((0, 5))) -> np.ndarray:
        """Feed detections for one frame; returns confirmed tracks.

        Args:
            dets: N×5 array of [x1, y1, x2, y2, score]. Pass empty array
                  when there are no detections.

        Returns:
            M×5 array of [x1, y1, x2, y2, track_id] for active tracks.
        """
        self._frame += 1

        # Predict next position for every existing tracker
        predicted = []
        dead = []
        for i, trk in enumerate(self._trackers):
            pos = trk.predict()[0]
            if np.any(np.isnan(pos)):
                dead.append(i)
            else:
                predicted.append(pos)
        for i in reversed(dead):
            self._trackers.pop(i)

        pred_arr = np.array(predicted) if predicted else np.empty((0, 4))

        # Match detections to predictions
        matched, new_dets, lost_trks = self._associate(dets, pred_arr)

        # Update matched trackers
        for di, ti in matched:
            self._trackers[ti].update(dets[di])

        # Start new trackers for unmatched detections
        for di in new_dets:
            self._trackers.append(_KalmanBoxTracker(dets[di]))

        # Collect output and prune stale trackers
        out = []
        for i in reversed(range(len(self._trackers))):
            trk = self._trackers[i]
            if trk.time_since_update > self.max_age:
                self._trackers.pop(i)
                continue
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self._frame <= self.min_hits
            ):
                box = trk.get_state()[0]
                out.append(np.append(box, trk.id))

        return np.array(out) if out else np.empty((0, 5))

    def _associate(self, dets, trks):
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))

        iou = _iou_matrix(dets[:, :4], trks[:, :4])
        rows, cols = linear_sum_assignment(1.0 - iou)

        matched, unmatched_dets, unmatched_trks = [], [], []
        matched_rows, matched_cols = set(), set()

        for r, c in zip(rows, cols):
            if iou[r, c] >= self.iou_threshold:
                matched.append((r, c))
                matched_rows.add(r)
                matched_cols.add(c)

        unmatched_dets = [i for i in range(len(dets)) if i not in matched_rows]
        unmatched_trks = [i for i in range(len(trks)) if i not in matched_cols]

        return matched, unmatched_dets, unmatched_trks
