// Rectangle outlines on an interleaved RGB24 buffer, for the frame stream.
#pragma once

#include <cstdint>

namespace cattrack {

void draw_rect(std::uint8_t* rgb, int w, int h, int x0, int y0, int x1, int y1,
               std::uint8_t r, std::uint8_t g, std::uint8_t b, int thickness = 2);

}  // namespace cattrack
