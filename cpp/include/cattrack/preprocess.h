// Frame preprocessing for YOLO (ports detection.preprocess_frame).
#pragma once

#include <cstdint>
#include <vector>

namespace cattrack {

// src: interleaved RGB, src_h * src_w * 3 bytes. Returns a CHW (1,3,model_h,
// model_w) float tensor: bilinear resize, /255.
std::vector<float> preprocess_frame(const std::uint8_t* src, int src_w, int src_h,
                                    int model_w, int model_h);

}  // namespace cattrack
