// Owns every Track and runs the per-frame cycle:
//   predict -> Hungarian match -> update / miss / spawn -> prune -> dedup
// Ports cat_tracker/multi_tracker.py.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "cattrack/geom.h"
#include "cattrack/track.h"

namespace cattrack {

struct Detection {
    BBox box;
    double confidence;
};

class MultiTracker {
public:
    MultiTracker(int max_missed = 10, int min_hits = 3, double iou_threshold = 0.3,
                 int model_w = 320, int model_h = 320);

    std::vector<Track*> update(const std::vector<Detection>& detections);
    std::vector<Track*> predict_only();  // skipped-inference frame
    void compensate_camera_motion(double dx, double dy);

    std::vector<Track*> tracks();
    std::vector<Track*> confirmed();

private:
    struct MatchResult {
        std::vector<std::pair<int, int>> matches;  // (track_idx, det_idx)
        std::vector<int> unmatched_dets;
        std::vector<int> unmatched_tracks;
    };

    MatchResult match(const std::vector<Detection>& detections) const;
    void deduplicate();
    int alloc_id() { return next_id_++; }

    std::vector<std::unique_ptr<Track>> tracks_;
    int max_missed_;
    int min_hits_;
    double iou_threshold_;
    int next_id_ = 1;
    double img_diagonal_;
    double max_match_dist_;
};

}  // namespace cattrack
