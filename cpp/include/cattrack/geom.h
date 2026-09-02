// Bounding-box geometry (ports utils.iou / utils.euclidean_distance).
// Boxes are center format {x_center, y_center, w, h} in model-space pixels.
#pragma once

#include <array>

namespace cattrack {

using BBox = std::array<double, 4>;

double iou(const BBox& a, const BBox& b);
double euclidean_distance(const BBox& a, const BBox& b);

}  // namespace cattrack
