#include "cattrack/track.h"

namespace cattrack {

Track::Track(const BBox& bbox, double confidence, int track_id, int min_hits)
    : id(track_id),
      bbox(bbox),
      predicted_bbox(bbox),
      confidence(confidence),
      min_hits_(min_hits),
      kf_(bbox) {}

BBox Track::predict() {
    predicted_bbox = kf_.predict();
    bbox = predicted_bbox;
    ++age;
    return predicted_bbox;
}

void Track::update(const BBox& measurement, double conf) {
    kf_.update(measurement);
    bbox = kf_.state();  // smoothed posterior, not the raw detection
    confidence = conf;
    ++hits;
    missed_frames = 0;
}

void Track::mark_missed() {
    ++missed_frames;
    kf_.on_missed();
    bbox = predicted_bbox;
}

}  // namespace cattrack
