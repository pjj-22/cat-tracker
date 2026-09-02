// One tracked cat: a Kalman filter plus a tentative -> confirmed -> deleted
// lifecycle. Ports cat_tracker/tracker.py. Coat identification stays in
// Python and writes back the name / candidate fields.
#pragma once

#include <string>

#include "cattrack/geom.h"
#include "cattrack/kalman.h"

namespace cattrack {

class Track {
public:
    Track(const BBox& bbox, double confidence, int track_id = 1, int min_hits = 3);

    BBox predict();                                  // Kalman forward step, bumps age
    void update(const BBox& bbox, double confidence);  // fold in a matched detection
    void mark_missed();                              // no match this frame

    bool is_confirmed() const { return hits >= min_hits_; }
    bool should_delete(int max_missed = 10) const { return missed_frames > max_missed; }
    std::array<double, 2> velocity() const { return kf_.velocity(); }

    BBoxKalmanFilter& kf() { return kf_; }
    const BBoxKalmanFilter& kf() const { return kf_; }

    int id;
    BBox bbox;              // posterior after update, else the prediction
    BBox predicted_bbox;    // last prediction, used for data association
    double confidence;
    int hits = 1;
    int missed_frames = 0;
    int age = 0;

    // driven from Python
    std::string name = "Unknown";
    double name_confidence = 0.0;
    std::string candidate_name = "Unknown";
    int candidate_streak = 0;

private:
    int min_hits_;
    BBoxKalmanFilter kf_;
};

}  // namespace cattrack
