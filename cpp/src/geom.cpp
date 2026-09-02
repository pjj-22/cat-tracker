#include "cattrack/geom.h"

#include <algorithm>
#include <cmath>

namespace cattrack {

namespace {
std::array<double, 4> to_xyxy(const BBox& b) {
    return {b[0] - b[2] / 2.0, b[1] - b[3] / 2.0,
            b[0] + b[2] / 2.0, b[1] + b[3] / 2.0};
}
}  // namespace

double iou(const BBox& a, const BBox& b) {
    const auto box1 = to_xyxy(a);
    const auto box2 = to_xyxy(b);

    const double x1 = std::max(box1[0], box2[0]);
    const double y1 = std::max(box1[1], box2[1]);
    const double x2 = std::min(box1[2], box2[2]);
    const double y2 = std::min(box1[3], box2[3]);

    const double intersection =
        std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);

    const double area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const double area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    const double uni = area1 + area2 - intersection;
    if (uni == 0.0) return 0.0;
    return intersection / uni;
}

double euclidean_distance(const BBox& a, const BBox& b) {
    return std::hypot(a[0] - b[0], a[1] - b[1]);
}

}  // namespace cattrack
