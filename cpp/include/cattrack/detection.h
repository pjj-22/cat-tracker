// YOLO11 output parsing + NMS (ports detection.parse_yolo_output).
#pragma once

#include <cstddef>
#include <vector>

#include "cattrack/multi_tracker.h"  // Detection

namespace cattrack {

inline constexpr int kCatClassId = 15;  // COCO "cat"

// raw: model output (n_attrs, n_boxes) row-major, so attribute a of box b is
// raw[a * n_boxes + b]. YOLO11 imgsz 320 -> (84, 8400).
std::vector<Detection> parse_yolo_output(const float* raw, std::size_t n_attrs,
                                         std::size_t n_boxes,
                                         double conf_threshold = 0.15,
                                         double iou_threshold = 0.4);

}  // namespace cattrack
