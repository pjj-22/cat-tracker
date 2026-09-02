#include "cattrack/detection.h"

#include <algorithm>
#include <numeric>

namespace cattrack {

namespace {

struct Candidate {
    BBox box;      // center format {cx, cy, w, h}
    double score;
};

// IoU on top-left rects {x, y, w, h}, matching cv2.dnn.NMSBoxes.
double rect_iou(const BBox& a, const BBox& b) {
    const double ax2 = a[0] + a[2], ay2 = a[1] + a[3];
    const double bx2 = b[0] + b[2], by2 = b[1] + b[3];
    const double ix = std::max(0.0, std::min(ax2, bx2) - std::max(a[0], b[0]));
    const double iy = std::max(0.0, std::min(ay2, by2) - std::max(a[1], b[1]));
    const double inter = ix * iy;
    const double uni = a[2] * a[3] + b[2] * b[3] - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

// Greedy NMS, score-descending, same suppression rule as OpenCV (drop a box
// once its IoU with a kept box exceeds the threshold).
std::vector<std::size_t> nms(const std::vector<Candidate>& cands, double iou_threshold) {
    std::vector<std::size_t> order(cands.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t i, std::size_t j) { return cands[i].score > cands[j].score; });

    std::vector<std::size_t> kept;
    for (std::size_t idx : order) {
        BBox tl = {cands[idx].box[0] - cands[idx].box[2] / 2.0,
                   cands[idx].box[1] - cands[idx].box[3] / 2.0,
                   cands[idx].box[2], cands[idx].box[3]};
        bool keep = true;
        for (std::size_t k : kept) {
            BBox ktl = {cands[k].box[0] - cands[k].box[2] / 2.0,
                        cands[k].box[1] - cands[k].box[3] / 2.0,
                        cands[k].box[2], cands[k].box[3]};
            if (rect_iou(tl, ktl) > iou_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) kept.push_back(idx);
    }
    return kept;
}

}  // namespace

std::vector<Detection> parse_yolo_output(const float* raw, std::size_t n_attrs,
                                         std::size_t n_boxes, double conf_threshold,
                                         double iou_threshold) {
    const std::size_t n_classes = n_attrs - 4;
    std::vector<Candidate> cands;

    for (std::size_t b = 0; b < n_boxes; ++b) {
        std::size_t best = 0;
        float best_score = raw[4 * n_boxes + b];
        for (std::size_t c = 1; c < n_classes; ++c) {
            const float s = raw[(4 + c) * n_boxes + b];
            if (s > best_score) {
                best_score = s;
                best = c;
            }
        }

        if (static_cast<int>(best) != kCatClassId || best_score <= conf_threshold) continue;

        cands.push_back({{raw[0 * n_boxes + b], raw[1 * n_boxes + b],
                          raw[2 * n_boxes + b], raw[3 * n_boxes + b]},
                         static_cast<double>(best_score)});
    }

    std::vector<Detection> out;
    for (std::size_t i : nms(cands, iou_threshold)) {
        out.push_back({cands[i].box, cands[i].score});
    }
    return out;
}

}  // namespace cattrack
