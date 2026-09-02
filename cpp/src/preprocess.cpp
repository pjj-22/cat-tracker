#include "cattrack/preprocess.h"

#include <algorithm>
#include <cmath>

namespace cattrack {

std::vector<float> preprocess_frame(const std::uint8_t* src, int src_w, int src_h,
                                    int model_w, int model_h) {
    std::vector<float> out(static_cast<std::size_t>(3) * model_w * model_h);

    const double sx = static_cast<double>(src_w) / model_w;
    const double sy = static_cast<double>(src_h) / model_h;
    const std::size_t plane = static_cast<std::size_t>(model_w) * model_h;

    for (int dy = 0; dy < model_h; ++dy) {
        // sample at the destination pixel center, like cv2 INTER_LINEAR
        double fy = (dy + 0.5) * sy - 0.5;
        fy = std::clamp(fy, 0.0, static_cast<double>(src_h - 1));
        const int y0 = static_cast<int>(fy);
        const int y1 = std::min(y0 + 1, src_h - 1);
        const double wy = fy - y0;

        for (int dx = 0; dx < model_w; ++dx) {
            double fx = (dx + 0.5) * sx - 0.5;
            fx = std::clamp(fx, 0.0, static_cast<double>(src_w - 1));
            const int x0 = static_cast<int>(fx);
            const int x1 = std::min(x0 + 1, src_w - 1);
            const double wx = fx - x0;

            const std::size_t p00 = (static_cast<std::size_t>(y0) * src_w + x0) * 3;
            const std::size_t p01 = (static_cast<std::size_t>(y0) * src_w + x1) * 3;
            const std::size_t p10 = (static_cast<std::size_t>(y1) * src_w + x0) * 3;
            const std::size_t p11 = (static_cast<std::size_t>(y1) * src_w + x1) * 3;
            const std::size_t dst = static_cast<std::size_t>(dy) * model_w + dx;

            for (int c = 0; c < 3; ++c) {
                const double top = src[p00 + c] * (1.0 - wx) + src[p01 + c] * wx;
                const double bot = src[p10 + c] * (1.0 - wx) + src[p11 + c] * wx;
                out[c * plane + dst] = static_cast<float>((top * (1.0 - wy) + bot * wy) / 255.0);
            }
        }
    }
    return out;
}

}  // namespace cattrack
