#include "cattrack/multi_tracker.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "cattrack/hungarian.h"

namespace cattrack {

namespace {
constexpr double kGate = 1e6;  // cost sentinel for distance-gated pairs
}

MultiTracker::MultiTracker(int max_missed, int min_hits, double iou_threshold,
                           int model_w, int model_h)
    : max_missed_(max_missed),
      min_hits_(min_hits),
      iou_threshold_(iou_threshold),
      img_diagonal_(std::sqrt(static_cast<double>(model_w) * model_w +
                              static_cast<double>(model_h) * model_h)),
      max_match_dist_(model_w * 0.5) {}

std::vector<Track*> MultiTracker::update(const std::vector<Detection>& detections) {
    for (auto& t : tracks_) t->predict();

    MatchResult mr;
    if (!detections.empty() && !tracks_.empty()) {
        mr = match(detections);
    } else {
        for (std::size_t i = 0; i < detections.size(); ++i)
            mr.unmatched_dets.push_back(static_cast<int>(i));
        for (std::size_t i = 0; i < tracks_.size(); ++i)
            mr.unmatched_tracks.push_back(static_cast<int>(i));
    }

    for (const auto& [track_idx, det_idx] : mr.matches) {
        tracks_[track_idx]->update(detections[det_idx].box,
                                   detections[det_idx].confidence);
    }

    for (int track_idx : mr.unmatched_tracks) tracks_[track_idx]->mark_missed();

    for (int det_idx : mr.unmatched_dets) {
        tracks_.push_back(std::make_unique<Track>(detections[det_idx].box,
                                                  detections[det_idx].confidence,
                                                  alloc_id(), min_hits_));
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [&](const std::unique_ptr<Track>& t) {
                           return t->should_delete(max_missed_);
                       }),
        tracks_.end());

    deduplicate();

    return confirmed();
}

std::vector<Track*> MultiTracker::predict_only() {
    for (auto& t : tracks_) t->predict();
    return confirmed();
}

void MultiTracker::compensate_camera_motion(double dx, double dy) {
    for (auto& t : tracks_) {
        if (!t->is_confirmed()) continue;
        t->kf().compensate_camera_motion(dx, dy);
    }
}

std::vector<Track*> MultiTracker::tracks() {
    std::vector<Track*> out;
    out.reserve(tracks_.size());
    for (auto& t : tracks_) out.push_back(t.get());
    return out;
}

std::vector<Track*> MultiTracker::confirmed() {
    std::vector<Track*> out;
    for (auto& t : tracks_)
        if (t->is_confirmed()) out.push_back(t.get());
    return out;
}

MultiTracker::MatchResult MultiTracker::match(
    const std::vector<Detection>& detections) const {
    const std::size_t n_tracks = tracks_.size();
    const std::size_t n_dets = detections.size();

    std::vector<double> cost(n_tracks * n_dets, 0.0);
    std::vector<double> iou_m(n_tracks * n_dets, 0.0);
    std::vector<double> dist_m(n_tracks * n_dets, 0.0);

    for (std::size_t i = 0; i < n_tracks; ++i) {
        for (std::size_t j = 0; j < n_dets; ++j) {
            const double iou_score = iou(tracks_[i]->predicted_bbox, detections[j].box);
            const double center_dist =
                euclidean_distance(tracks_[i]->predicted_bbox, detections[j].box);
            const std::size_t k = i * n_dets + j;
            iou_m[k] = iou_score;
            dist_m[k] = center_dist;

            if (center_dist > max_match_dist_) {
                cost[k] = kGate;
            } else {
                const double normalized_dist = center_dist / img_diagonal_;
                cost[k] = 0.7 * (1.0 - iou_score) + 0.3 * normalized_dist;
            }
        }
    }

    const Assignment a = linear_sum_assignment(cost, n_tracks, n_dets);

    MatchResult mr;
    for (std::size_t i = 0; i < n_dets; ++i) mr.unmatched_dets.push_back(static_cast<int>(i));
    for (std::size_t i = 0; i < n_tracks; ++i) mr.unmatched_tracks.push_back(static_cast<int>(i));

    for (std::size_t p = 0; p < a.row_ind.size(); ++p) {
        const int track_idx = a.row_ind[p];
        const int det_idx = a.col_ind[p];
        const std::size_t k = static_cast<std::size_t>(track_idx) * n_dets + det_idx;

        if (cost[k] >= kGate) continue;
        const bool close_enough = dist_m[k] < max_match_dist_ * 0.5;
        if (iou_m[k] >= iou_threshold_ || close_enough) {
            mr.matches.emplace_back(track_idx, det_idx);
            mr.unmatched_dets.erase(std::remove(mr.unmatched_dets.begin(),
                                                mr.unmatched_dets.end(), det_idx),
                                    mr.unmatched_dets.end());
            mr.unmatched_tracks.erase(std::remove(mr.unmatched_tracks.begin(),
                                                  mr.unmatched_tracks.end(), track_idx),
                                      mr.unmatched_tracks.end());
        }
    }

    return mr;
}

void MultiTracker::deduplicate() {
    // Same coat name on two confirmed tracks => one is a ghost; drop whichever
    // isn't currently detected (or has fewer hits). Unidentified tracks: skip.
    std::vector<Track*> conf = confirmed();
    std::set<int> to_delete;

    for (std::size_t i = 0; i < conf.size(); ++i) {
        for (std::size_t j = i + 1; j < conf.size(); ++j) {
            Track* a = conf[i];
            Track* b = conf[j];
            if (a->name == "Unknown" || a->name != b->name) continue;
            if (a->missed_frames == 0 && b->missed_frames == 0) continue;

            if (a->missed_frames == 0 && b->missed_frames > 0) {
                to_delete.insert(b->id);
            } else if (b->missed_frames == 0 && a->missed_frames > 0) {
                to_delete.insert(a->id);
            } else {
                to_delete.insert(a->hits >= b->hits ? b->id : a->id);
            }
        }
    }

    if (!to_delete.empty()) {
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [&](const std::unique_ptr<Track>& t) {
                                         return to_delete.count(t->id) > 0;
                                     }),
                      tracks_.end());
    }
}

}  // namespace cattrack
