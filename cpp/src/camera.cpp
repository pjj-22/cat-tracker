#include "cattrack/camera.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cattrack {

namespace {
std::string default_command(int w, int h, double fps) {
    return "rpicam-vid -t 0 --nopreview --inline --codec yuv420"
           " --width " + std::to_string(w) +
           " --height " + std::to_string(h) +
           " --framerate " + std::to_string(static_cast<int>(std::lround(fps))) +
           " -o -";
}

bool looks_like_path(const std::string& s) {
    return !s.empty() && s.find(' ') == std::string::npos &&
           (s.front() == '/' || s.front() == '.');
}
}  // namespace

FrameSource::FrameSource(int width, int height, double fps, PixelFormat fmt,
                         const std::string& command)
    : width_(width), height_(height), fmt_(fmt) {
    const std::size_t px = static_cast<std::size_t>(width) * height;
    raw_.resize(fmt == PixelFormat::YUV420 ? px * 3 / 2 : px * 3);

    if (looks_like_path(command)) {
        stream_ = std::fopen(command.c_str(), "rb");
        is_pipe_ = false;
    } else {
        const std::string cmd = command.empty() ? default_command(width, height, fps) : command;
        stream_ = ::popen(cmd.c_str(), "r");
        is_pipe_ = true;
    }
    if (!stream_) throw std::runtime_error("cattrack::FrameSource: cannot open frame stream");
}

FrameSource::~FrameSource() {
    if (!stream_) return;
    if (is_pipe_) ::pclose(stream_);
    else std::fclose(stream_);
}

bool FrameSource::read(std::vector<std::uint8_t>& rgb) {
    if (std::fread(raw_.data(), 1, raw_.size(), stream_) != raw_.size()) return false;

    rgb.resize(static_cast<std::size_t>(width_) * height_ * 3);
    if (fmt_ == PixelFormat::YUV420) {
        yuv420_to_rgb(raw_.data(), width_, height_, rgb.data());
    } else {
        std::memcpy(rgb.data(), raw_.data(), rgb.size());
    }
    return true;
}

void yuv420_to_rgb(const std::uint8_t* yuv, int w, int h, std::uint8_t* rgb) {
    const std::uint8_t* Y = yuv;
    const std::uint8_t* U = Y + static_cast<std::size_t>(w) * h;
    const std::uint8_t* V = U + static_cast<std::size_t>(w) * h / 4;
    const int cw = w / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double yy = 1.164 * (Y[y * w + x] - 16);
            const double u = U[(y / 2) * cw + x / 2] - 128;
            const double v = V[(y / 2) * cw + x / 2] - 128;
            const std::size_t o = (static_cast<std::size_t>(y) * w + x) * 3;
            rgb[o + 0] = static_cast<std::uint8_t>(std::clamp(yy + 1.596 * v, 0.0, 255.0));
            rgb[o + 1] =
                static_cast<std::uint8_t>(std::clamp(yy - 0.392 * u - 0.813 * v, 0.0, 255.0));
            rgb[o + 2] = static_cast<std::uint8_t>(std::clamp(yy + 2.017 * u, 0.0, 255.0));
        }
    }
}

}  // namespace cattrack
