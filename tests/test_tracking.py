"""Tests for Track lifecycle and MultiTracker matching."""

import numpy as np
from cat_tracker.tracker import Track
from cat_tracker.multi_tracker import MultiTracker


class TestTrack:
    def setup_method(self):
        # Reset global ID counter before each test
        Track._next_id = 1

    def test_create(self):
        t = Track(np.array([100, 100, 50, 50]), 0.9)
        assert t.id == 1
        assert t.hits == 1
        assert t.missed_frames == 0
        assert t.name == "Unknown"

    def test_predict_returns_bbox(self):
        t = Track(np.array([100, 100, 50, 50]), 0.9)
        pred = t.predict()
        assert pred.shape == (4,)

    def test_update_increments_hits(self):
        t = Track(np.array([100, 100, 50, 50]), 0.9)
        t.predict()
        t.update(np.array([102, 101, 50, 50]), 0.85)
        assert t.hits == 2
        assert t.missed_frames == 0

    def test_confirm_requires_min_hits(self):
        t = Track(np.array([100, 100, 50, 50]), 0.9)
        assert not t.is_confirmed()  # only 1 hit
        for _ in range(2):
            t.predict()
            t.update(np.array([100, 100, 50, 50]), 0.9)
        assert t.is_confirmed()  # 3 hits

    def test_mark_missed(self):
        t = Track(np.array([100, 100, 50, 50]), 0.9)
        t.predict()
        t.mark_missed()
        assert t.missed_frames == 1

    def test_should_delete_after_max_missed(self):
        t = Track(np.array([100, 100, 50, 50]), 0.9)
        for _ in range(12):
            t.predict()
            t.mark_missed()
        assert t.should_delete(max_missed=10)


class TestMultiTracker:
    def setup_method(self):
        Track._next_id = 1

    def test_new_detection_creates_track(self):
        mt = MultiTracker(max_missed=10, min_hits=1, iou_threshold=0.1)
        dets = [{'box': np.array([100, 100, 50, 50]), 'confidence': 0.9}]
        mt.update(dets)
        assert len(mt.tracks) == 1

    def test_matching_detection_updates_existing(self):
        mt = MultiTracker(max_missed=10, min_hits=1, iou_threshold=0.1)
        det = [{'box': np.array([100, 100, 50, 50]), 'confidence': 0.9}]
        mt.update(det)
        # Same position → should match, not create new
        mt.update(det)
        assert len(mt.tracks) == 1
        assert mt.tracks[0].hits == 2

    def test_distant_detection_creates_new_track(self):
        mt = MultiTracker(max_missed=10, min_hits=1, iou_threshold=0.1)
        mt.update([{'box': np.array([100, 100, 50, 50]), 'confidence': 0.9}])
        mt.update([
            {'box': np.array([100, 100, 50, 50]), 'confidence': 0.9},
            {'box': np.array([500, 400, 50, 50]), 'confidence': 0.8}
        ])
        assert len(mt.tracks) == 2

    def test_empty_detections(self):
        mt = MultiTracker()
        confirmed = mt.update([])
        assert confirmed == []

    def test_confirmed_tracks_returned(self):
        mt = MultiTracker(max_missed=10, min_hits=3, iou_threshold=0.1)
        det = [{'box': np.array([100, 100, 50, 50]), 'confidence': 0.9}]
        # Need min_hits=3 matches
        for _ in range(3):
            confirmed = mt.update(det)
        assert len(confirmed) == 1

    def test_gated_detection_creates_new_track(self):
        # Detection beyond max_match_dist (model_w * 0.5 = 160) must not match
        # the existing track even if iou_threshold would allow it.
        mt = MultiTracker(max_missed=10, min_hits=1, iou_threshold=0.0, model_w=320, model_h=320)
        mt.update([{'box': np.array([100.0, 160.0, 40.0, 40.0]), 'confidence': 0.9}])
        # Center distance = 165 > 160 → gated
        far_det = [{'box': np.array([265.0, 160.0, 40.0, 40.0]), 'confidence': 0.9}]
        mt.update(far_det)
        assert len(mt.tracks) == 2

    def test_close_detection_not_gated(self):
        # Detection within max_match_dist should still match normally.
        mt = MultiTracker(max_missed=10, min_hits=1, iou_threshold=0.1, model_w=320, model_h=320)
        det = [{'box': np.array([100.0, 100.0, 50.0, 50.0]), 'confidence': 0.9}]
        mt.update(det)
        mt.update(det)
        assert len(mt.tracks) == 1
        assert mt.tracks[0].hits == 2

    def test_hybrid_cost_prefers_closer_candidate(self):
        # Two tracks, two detections. The detection closer in center distance
        # to each track should win even when IoU scores are similar.
        mt = MultiTracker(max_missed=10, min_hits=1, iou_threshold=0.3, model_w=320, model_h=320)
        mt.update([
            {'box': np.array([80.0,  160.0, 60.0, 60.0]), 'confidence': 0.9},
            {'box': np.array([240.0, 160.0, 60.0, 60.0]), 'confidence': 0.9},
        ])
        # Detections stay close to their respective tracks
        confirmed = mt.update([
            {'box': np.array([82.0,  160.0, 60.0, 60.0]), 'confidence': 0.9},
            {'box': np.array([238.0, 160.0, 60.0, 60.0]), 'confidence': 0.9},
        ])
        # Both tracks updated (hits=2), not swapped or duplicated
        assert len(mt.tracks) == 2
        assert all(t.hits == 2 for t in mt.tracks)

    def test_compensate_camera_motion_inflates_velocity_covariance(self):
        mt = MultiTracker(min_hits=1, model_w=320, model_h=320)
        mt.update([{'box': np.array([160.0, 160.0, 50.0, 50.0]), 'confidence': 0.9}])
        track = mt.tracks[0]
        p44_before = track.kf.kf.P[4, 4]
        p55_before = track.kf.kf.P[5, 5]
        mt.compensate_camera_motion(dx=5.0, dy=0.0)
        assert track.kf.kf.P[4, 4] > p44_before
        assert track.kf.kf.P[5, 5] > p55_before

    def test_compensate_camera_motion_zeroes_velocity(self):
        mt = MultiTracker(min_hits=1, model_w=320, model_h=320)
        det = {'box': np.array([160.0, 160.0, 50.0, 50.0]), 'confidence': 0.9}
        # Two updates to build up some velocity
        mt.update([det])
        mt.update([{'box': np.array([170.0, 160.0, 50.0, 50.0]), 'confidence': 0.9}])
        mt.compensate_camera_motion(dx=10.0, dy=0.0)
        track = mt.tracks[0]
        assert track.kf.kf.x[4] == 0.0
        assert track.kf.kf.x[5] == 0.0
