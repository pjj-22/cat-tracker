#include "cattrack/kalman.h"

#include <algorithm>

namespace cattrack {

BBoxKalmanFilter::BBoxKalmanFilter(const std::array<double, 4>& bbox) {
    // State transition: constant velocity, dt = 1 (one YOLO interval).
    F_ = Mat<8, 8>::identity();
    F_(0, 4) = 1.0;  // x  += vx
    F_(1, 5) = 1.0;  // y  += vy
    F_(2, 6) = 1.0;  // w  += vw
    F_(3, 7) = 1.0;  // h  += vh

    // Measurement matrix: we observe position/size, not velocity.
    H_ = Mat<4, 8>::zeros();
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;
    H_(3, 3) = 1.0;

    // Measurement noise: filterpy default eye(4), then *= 2.0.
    R_ = Mat<4, 4>::identity();
    for (std::size_t i = 0; i < 4; ++i) R_(i, i) *= 2.0;

    // Process noise: filterpy default eye(8); translational velocity loosened,
    // size velocity clamped hard (cats change place fast, not size).
    Q_ = Mat<8, 8>::identity();
    Q_(4, 4) *= 10.0;
    Q_(5, 5) *= 10.0;
    Q_(6, 6) *= 0.01;
    Q_(7, 7) *= 0.01;

    // Initial covariance: filterpy default eye(8); huge prior uncertainty on
    // velocity since we start from a single frame.
    P_ = Mat<8, 8>::identity();
    P_(4, 4) *= 1000.0;
    P_(5, 5) *= 1000.0;
    P_(6, 6) *= 100.0;
    P_(7, 7) *= 100.0;

    x_ = Mat<8, 1>::zeros();
    for (std::size_t i = 0; i < 4; ++i) x_(i, 0) = bbox[i];
}

void BBoxKalmanFilter::clamp_dimensions() {
    x_(2, 0) = std::max(10.0, x_(2, 0));
    x_(3, 0) = std::max(10.0, x_(3, 0));
}

std::array<double, 4> BBoxKalmanFilter::predict() {
    // filterpy KalmanFilter.predict(): x = F x, then P = alpha_sq * F P F^T + Q.
    // alpha_sq (the fading-memory factor) is 1, so it drops out.
    x_ = F_ * x_;
    P_ = (F_ * P_ * transpose(F_)) + Q_;
    clamp_dimensions();
    return state();
}

void BBoxKalmanFilter::update(const std::array<double, 4>& bbox) {
    Mat<4, 1> z{};
    for (std::size_t i = 0; i < 4; ++i) z(i, 0) = bbox[i];

    // Same steps filterpy's KalmanFilter.update() runs
    //   y    = z - H x
    //   PHT  = P H^T
    //   S    = H PHT + R
    //   K    = PHT S^-1
    //   x    = x + K y
    //   I_KH = I - K H
    //   P    = I_KH P I_KH^T + K R K^T
    // The P line is the Joseph form. filterpy's own source comment says it uses
    // this over the shorter P = (I - KH) P because it "is more numerically
    // stable and works for non-optimal K". The short form can leave P
    // non-symmetric / non-PSD under round-off in K; the Joseph form adds two
    // PSD terms so P stays a valid covariance.
    // more info: anuncommonlab.com/articles/how-kalman-filters-work/part2.html
    const Mat<4, 1> y = z - (H_ * x_);
    const Mat<8, 4> PHT = P_ * transpose(H_);
    const Mat<4, 4> S = (H_ * PHT) + R_;
    const Mat<8, 4> K = PHT * inverse<4>(S);

    x_ = x_ + (K * y);

    const Mat<8, 8> I_KH = Mat<8, 8>::identity() - (K * H_);
    P_ = (I_KH * P_ * transpose(I_KH)) + (K * R_ * transpose(K));

    clamp_dimensions();
}

std::array<double, 4> BBoxKalmanFilter::state() const {
    return {x_(0, 0), x_(1, 0), x_(2, 0), x_(3, 0)};
}

std::array<double, 2> BBoxKalmanFilter::velocity() const {
    return {x_(4, 0), x_(5, 0)};
}

void BBoxKalmanFilter::on_missed() {
    x_(4, 0) *= 0.5;
    x_(5, 0) *= 0.5;
    x_(6, 0) = 0.0;
    x_(7, 0) = 0.0;
}

void BBoxKalmanFilter::compensate_camera_motion(double dx, double dy) {
    x_(0, 0) -= dx;
    x_(1, 0) += dy;
    x_(4, 0) = 0.0;
    x_(5, 0) = 0.0;
    P_(0, 0) += 500.0;
    P_(1, 1) += 500.0;
    P_(4, 4) += 200.0;
    P_(5, 5) += 200.0;
}

}  // namespace cattrack
