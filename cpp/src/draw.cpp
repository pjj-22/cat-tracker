#include "cattrack/draw.h"

#include <algorithm>

namespace cattrack {

namespace {
void put(std::uint8_t* rgb, int w, int h, int x, int y, std::uint8_t r, std::uint8_t g,
         std::uint8_t b) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    std::uint8_t* p = rgb + (static_cast<std::size_t>(y) * w + x) * 3;
    p[0] = r;
    p[1] = g;
    p[2] = b;
}
}  // namespace

void draw_rect(std::uint8_t* rgb, int w, int h, int x0, int y0, int x1, int y1,
               std::uint8_t r, std::uint8_t g, std::uint8_t b, int thickness) {
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);

    for (int t = 0; t < thickness; ++t) {
        for (int x = x0; x <= x1; ++x) {
            put(rgb, w, h, x, y0 + t, r, g, b);
            put(rgb, w, h, x, y1 - t, r, g, b);
        }
        for (int y = y0; y <= y1; ++y) {
            put(rgb, w, h, x0 + t, y, r, g, b);
            put(rgb, w, h, x1 - t, y, r, g, b);
        }
    }
}

}  // namespace cattrack
