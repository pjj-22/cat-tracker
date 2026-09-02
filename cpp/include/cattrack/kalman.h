// 8-state constant-velocity Kalman filter for a bounding box: [x, y, w, h]
// and their rates, center format, model-space pixels.
//
// The Python side (cat_tracker/kalman_filter.py) is a wrapper around
// filterpy.kalman.KalmanFilter (v1.4.5): it sets F/H/R/Q/P and lets filterpy
// run predict/update. This class collapses both: the constructor mirrors that
// F/H/R/Q/P setup, and predict()/update() implement the same equations filterpy
// runs (see its KalmanFilter.predict / .update). tests/test_cpp_parity.py runs
// this against the real filterpy on random sequences and checks it agrees to
// ~1e-6, so filterpy is the reference, not something being translated.
#pragma once

#include <array>

#include "cattrack/linalg.h"

namespace cattrack {

class BBoxKalmanFilter {
public:
    // bbox: {x_center, y_center, width, height}
    explicit BBoxKalmanFilter(const std::array<double, 4>& bbox);

    // Advance one step; returns the predicted {x, y, w, h}.
    std::array<double, 4> predict();

    // Fold in a measurement {x, y, w, h}.
    void update(const std::array<double, 4>& bbox);

    std::array<double, 4> state() const;      // {x, y, w, h}
    std::array<double, 2> velocity() const;   // {vx, vy}

    // Mirrors Track.mark_missed(): damp translational velocity, zero size rate.
    void on_missed();

    // Mirrors MultiTracker.compensate_camera_motion() for a single track.
    void compensate_camera_motion(double dx, double dy);

    double x(std::size_t i) const { return x_(i, 0); }
    double P(std::size_t i, std::size_t j) const { return P_(i, j); }

private:
    void clamp_dimensions();

    Mat<8, 1> x_{};
    Mat<8, 8> P_{};
    Mat<8, 8> F_{};
    Mat<4, 8> H_{};
    Mat<8, 8> Q_{};
    Mat<4, 4> R_{};
};

}  // namespace cattrack
