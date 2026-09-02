"""Parity checks: C++ tracking core vs the reference Python implementation.

Skipped unless the extension has been built:
    cmake -B cpp/build -S cpp && cmake --build cpp/build -j
"""

import pathlib
import sys

import numpy as np
import pytest

_BUILD = pathlib.Path(__file__).resolve().parent.parent / "cpp" / "build"
if _BUILD.is_dir():
    sys.path.insert(0, str(_BUILD))

cattrack_cpp = pytest.importorskip(
    "cattrack_cpp", reason="build the C++ module: cmake --build cpp/build"
)

from scipy.optimize import linear_sum_assignment as scipy_lsa

from cat_tracker.kalman_filter import BBoxKalmanFilter as PyKF
from cat_tracker.multi_tracker import MultiTracker as PyMT
from cat_tracker.tracker import Track as PyTrack
from cat_tracker import utils as py_utils
from cat_tracker import detection as py_detection
from cat_tracker.config import load_config as py_load_config

TOL = 1e-6


def _rng():
    return np.random.default_rng(20260831)


def test_kalman_predict_update_sequence():
    rng = _rng()
    for _ in range(50):
        start = rng.uniform([20, 20, 15, 15], [300, 300, 90, 90])
        py = PyKF(start.copy())
        cpp = cattrack_cpp.BBoxKalmanFilter(start.copy())

        pos = start.copy()
        for step in range(40):
            py_pred = py.predict()
            cpp_pred = cpp.predict()
            np.testing.assert_allclose(cpp_pred, py_pred, rtol=0, atol=TOL)

            if step % 5 != 3:  # occasionally miss a detection
                pos = pos + rng.normal(0, 4, 4)
                pos[2:] = np.clip(pos[2:], 10, None)
                py.update(pos.copy())
                cpp.update(pos.copy())
            else:
                py.kf.x[4] *= 0.5
                py.kf.x[5] *= 0.5
                py.kf.x[6] = 0.0
                py.kf.x[7] = 0.0
                cpp.on_missed()

            np.testing.assert_allclose(
                cpp.get_state(), py.get_state(), rtol=0, atol=TOL
            )
            np.testing.assert_allclose(
                cpp.get_velocity(), py.get_velocity(), rtol=0, atol=TOL
            )


def test_kalman_camera_motion_compensation():
    rng = _rng()
    start = np.array([150.0, 150.0, 40.0, 30.0])
    py = PyKF(start.copy())
    cpp = cattrack_cpp.BBoxKalmanFilter(start.copy())

    for _ in range(10):
        py.predict()
        cpp.predict()
        meas = start + rng.normal(0, 3, 4)
        meas[2:] = np.clip(meas[2:], 10, None)
        py.update(meas.copy())
        cpp.update(meas.copy())

    dx, dy = 12.5, -7.0
    py.kf.x[0] -= dx
    py.kf.x[1] += dy
    py.kf.x[4] = 0.0
    py.kf.x[5] = 0.0
    py.kf.P[0, 0] += 500.0
    py.kf.P[1, 1] += 500.0
    py.kf.P[4, 4] += 200.0
    py.kf.P[5, 5] += 200.0
    cpp.compensate_camera_motion(dx, dy)

    np.testing.assert_allclose(cpp.get_state(), py.get_state(), rtol=0, atol=TOL)
    for i in range(8):
        assert abs(cpp._x(i) - py.kf.x[i].item()) < TOL
    for i in range(8):
        for j in range(8):
            assert abs(cpp._P(i, j) - py.kf.P[i, j].item()) < 1e-4


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (2, 5), (6, 3), (4, 4), (8, 8), (1, 7)])
def test_assignment_matches_scipy(shape):
    rng = _rng()
    rows, cols = shape
    for _ in range(30):
        cost = rng.uniform(0, 100, size=(rows, cols))
        r_cpp, c_cpp = cattrack_cpp.linear_sum_assignment(cost)
        r_sp, c_sp = scipy_lsa(cost)

        # Same total cost => both optimal (individual pairs can differ on ties).
        assert cost[r_cpp, c_cpp].sum() == pytest.approx(cost[r_sp, c_sp].sum())
        assert sorted(r_cpp) == sorted(r_sp)
        assert len(set(c_cpp)) == len(c_cpp)
        assert list(r_cpp) == sorted(r_cpp)


def test_assignment_with_gate_costs():
    """The real _match cost matrix has 1e6 sentinels for gated-out pairs."""
    cost = np.array(
        [
            [0.10, 1e6, 1e6],
            [1e6, 0.20, 1e6],
            [1e6, 1e6, 0.05],
        ]
    )
    r_cpp, c_cpp = cattrack_cpp.linear_sum_assignment(cost)
    r_sp, c_sp = scipy_lsa(cost)
    assert cost[r_cpp, c_cpp].sum() == pytest.approx(cost[r_sp, c_sp].sum())
    assert list(zip(r_cpp, c_cpp)) == [(0, 0), (1, 1), (2, 2)]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("a,b", [
    ([100, 100, 50, 50], [100, 100, 50, 50]),   # identical
    ([100, 100, 50, 50], [130, 110, 40, 60]),   # partial overlap
    ([100, 100, 50, 50], [400, 400, 30, 30]),   # disjoint
    ([100, 100, 50, 50], [100, 100, 10, 10]),   # nested
])
def test_geom_matches_python(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    assert cattrack_cpp.iou(a, b) == pytest.approx(py_utils.iou(a, b))
    assert cattrack_cpp.euclidean_distance(a, b) == pytest.approx(
        py_utils.euclidean_distance(a, b)
    )


# --------------------------------------------------------------------------- #
# Track lifecycle
# --------------------------------------------------------------------------- #

def _track_pair(bbox, conf=0.9, min_hits=3):
    b = np.array(bbox, dtype=float)
    return PyTrack(b.copy(), conf, min_hits=min_hits), cattrack_cpp.Track(
        b.copy(), conf, min_hits=min_hits
    )


def _assert_track_agrees(py, cpp):
    assert py.id == cpp.id
    assert py.hits == cpp.hits
    assert py.missed_frames == cpp.missed_frames
    assert py.age == cpp.age
    assert py.is_confirmed() == cpp.is_confirmed()
    np.testing.assert_allclose(cpp.bbox, py.bbox, rtol=0, atol=TOL)
    np.testing.assert_allclose(cpp.velocity, py.velocity, rtol=0, atol=TOL)


def test_track_lifecycle_parity():
    rng = _rng()
    py, cpp = _track_pair([100, 100, 50, 50])
    pos = np.array([100.0, 100.0, 50.0, 50.0])

    for step in range(30):
        py.predict()
        cpp.predict()
        if step % 7 == 4:
            py.mark_missed()
            cpp.mark_missed()
        else:
            pos = pos + np.concatenate([rng.normal(0, 3, 2), rng.normal(0, 1, 2)])
            pos[2:] = np.clip(pos[2:], 10, None)
            py.update(pos.copy(), 0.8)
            cpp.update(pos.copy(), 0.8)
        _assert_track_agrees(py, cpp)

    assert py.should_delete(max_missed=10) == cpp.should_delete(max_missed=10)


# --------------------------------------------------------------------------- #
# MultiTracker cycle
# --------------------------------------------------------------------------- #

def _det(box, conf=0.9):
    return {"box": np.array(box, dtype=float), "confidence": conf}


def _run(mt, frames):
    for dets in frames:
        mt.update(dets)


def _assert_trackers_agree(py, cpp):
    pt, ct = list(py.tracks), list(cpp.tracks)
    assert [t.id for t in pt] == [t.id for t in ct]
    for p, c in zip(pt, ct):
        assert (p.hits, p.missed_frames, p.age) == (c.hits, c.missed_frames, c.age)
        assert p.is_confirmed() == c.is_confirmed()
        np.testing.assert_allclose(c.bbox, p.bbox, rtol=0, atol=1e-5)


def test_multitracker_two_cats_walking():
    """Two well-separated cats moving on smooth tracks: association is
    unambiguous, so the full pipelines must stay in lock-step."""
    rng = _rng()
    mt_py = PyMT(max_missed=10, min_hits=3, iou_threshold=0.3)
    mt_cpp = cattrack_cpp.MultiTracker(max_missed=10, min_hits=3, iou_threshold=0.3)

    a = np.array([80.0, 80.0, 40.0, 40.0])
    b = np.array([240.0, 220.0, 50.0, 45.0])
    frames = []
    for _ in range(25):
        a = a + np.concatenate([rng.normal(1.0, 0.5, 2), np.zeros(2)])
        b = b + np.concatenate([rng.normal(-0.8, 0.5, 2), np.zeros(2)])
        frames.append([_det(a), _det(b)])

    for dets in frames:
        mt_py.update(dets)
        mt_cpp.update(dets)
        _assert_trackers_agree(mt_py, mt_cpp)


def test_multitracker_miss_spawn_prune():
    mt_py = PyMT(max_missed=3, min_hits=2, iou_threshold=0.2)
    mt_cpp = cattrack_cpp.MultiTracker(max_missed=3, min_hits=2, iou_threshold=0.2)

    c = np.array([160.0, 160.0, 50.0, 50.0])
    frames = [
        [_det(c)],
        [_det(c + [4, 0, 0, 0])],
        [_det(c + [8, 0, 0, 0])],
        [],                                   # miss
        [],                                   # miss
        [_det(np.array([60.0, 260.0, 30.0, 30.0]))],   # far: new track
        [],
        [],
        [],                                   # first cat pruned
    ]
    for dets in frames:
        mt_py.update(dets)
        mt_cpp.update(dets)
        _assert_trackers_agree(mt_py, mt_cpp)


def test_multitracker_camera_compensation():
    mt_py = PyMT(min_hits=1, model_w=320, model_h=320)
    mt_cpp = cattrack_cpp.MultiTracker(min_hits=1, model_w=320, model_h=320)

    d = _det(np.array([160.0, 160.0, 50.0, 50.0]))
    for _ in range(4):
        mt_py.update([d])
        mt_cpp.update([d])

    mt_py.compensate_camera_motion(15.0, -6.0)
    mt_cpp.compensate_camera_motion(15.0, -6.0)

    tp, tc = mt_py.tracks[0], mt_cpp.tracks[0]
    for i in range(8):
        assert abs(tc.kf._x(i) - tp.kf.kf.x[i].item()) < 1e-6
        for j in range(8):
            assert abs(tc.kf._P(i, j) - tp.kf.kf.P[i, j].item()) < 1e-4


# --------------------------------------------------------------------------- #
# YOLO output parsing + NMS
# --------------------------------------------------------------------------- #

def _synthetic_yolo_output(rng, n_boxes=800, n_classes=80):
    """(1, 4 + n_classes, n_boxes) with a handful of planted cat boxes,
    some overlapping (NMS must collapse them), plus non-cat and low-conf noise."""
    out = rng.uniform(0.0, 0.05, size=(1, 4 + n_classes, n_boxes)).astype(np.float32)
    out[0, :4, :] = rng.uniform(20, 300, size=(4, n_boxes)).astype(np.float32)

    planted = [
        (60, (100.0, 100.0, 50.0, 40.0), 0.90),
        (61, (102.0, 101.0, 52.0, 41.0), 0.82),   # overlaps 60 -> suppressed
        (62, (240.0, 180.0, 45.0, 45.0), 0.75),
        (63, (150.0, 260.0, 30.0, 30.0), 0.10),   # below threshold
    ]
    for col, (cx, cy, w, h), score in planted:
        out[0, 0, col] = cx
        out[0, 1, col] = cy
        out[0, 2, col] = w
        out[0, 3, col] = h
        out[0, 4:, col] = 0.01
        out[0, 4 + py_detection.CAT_CLASS_ID, col] = score  # class 15 = cat

    # a high-confidence dog (class 16) that must be ignored
    out[0, 0:4, 70] = [80.0, 80.0, 40.0, 40.0]
    out[0, 4:, 70] = 0.01
    out[0, 4 + 16, 70] = 0.95
    return out


def _sort_dets(dets):
    return sorted(
        ([round(float(x), 4) for x in d["box"]], round(float(d["confidence"]), 4))
        for d in dets
    )


def test_parse_yolo_output_parity():
    rng = _rng()
    for _ in range(20):
        out = _synthetic_yolo_output(rng)
        py = py_detection.parse_yolo_output(out, conf_threshold=0.15, iou_threshold=0.4)
        cpp = cattrack_cpp.parse_yolo_output(out, conf_threshold=0.15, iou_threshold=0.4)
        assert _sort_dets(py) == _sort_dets(cpp)


def test_parse_yolo_output_empty():
    out = np.zeros((1, 84, 500), dtype=np.float32)
    assert cattrack_cpp.parse_yolo_output(out) == []


# --------------------------------------------------------------------------- #
# frame preprocessing (bilinear resize -> CHW -> /255)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("src_w,src_h", [(640, 480), (1280, 720), (321, 241)])
def test_preprocess_frame_matches_cv2(src_w, src_h):
    rng = _rng()
    frame = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)

    py = py_detection.preprocess_frame(frame, 320, 320)
    cpp = cattrack_cpp.preprocess_frame(frame, 320, 320)

    assert cpp.shape == py.shape == (1, 3, 320, 320)
    # cv2.resize uses fixed-point weights; allow tiny per-pixel deviation but
    # require the frames to agree closely on average.
    assert np.abs(cpp - py).mean() < 5e-3
    assert np.abs(cpp - py).max() < 3e-2


# --------------------------------------------------------------------------- #
# config.yaml reader
# --------------------------------------------------------------------------- #

_CONFIG_PATH = str(pathlib.Path(__file__).resolve().parent.parent / "config.yaml")


def test_config_matches_pyyaml():
    py = py_load_config(_CONFIG_PATH)
    cpp = cattrack_cpp.Config.load(_CONFIG_PATH)

    assert cpp.get_int("camera", "width") == py["camera"]["width"]
    assert cpp.get_double("camera", "fps") == pytest.approx(py["camera"]["fps"])
    assert cpp.get_bool("servo", "enabled") == py["servo"]["enabled"]
    assert cpp.get_double("servo", "patrol_step") == pytest.approx(py["servo"]["patrol_step"])
    assert cpp.get_string("detection", "model_path") == py["detection"]["model_path"]
    assert cpp.get_double("detection", "confidence_threshold") == pytest.approx(
        py["detection"]["confidence_threshold"]
    )
    assert cpp.get_int("tracking", "inference_every") == py["tracking"]["inference_every"]
    assert cpp.get_double("tracking", "iou_threshold") == pytest.approx(
        py["tracking"]["iou_threshold"]
    )
    assert cpp.get_doubles("identification", "hsv_weights") == pytest.approx(
        py["identification"]["hsv_weights"]
    )
    assert cpp.get_int("identification", "min_saturation") == py["identification"]["min_saturation"]
    assert cpp.get_string("logging", "position_log_path") == py["logging"]["position_log_path"]
