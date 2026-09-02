// Frame source: raw frames from a subprocess (rpicam-vid) or a file.
// No libcamera link.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace cattrack {

enum class PixelFormat { RGB24, YUV420 };

class FrameSource {
public:
    // command: shell command streaming raw frames to stdout, or a file path.
    // Empty -> a default rpicam-vid line for the given geometry.
    FrameSource(int width, int height, double fps, PixelFormat fmt = PixelFormat::YUV420,
                const std::string& command = "");
    ~FrameSource();

    FrameSource(const FrameSource&) = delete;
    FrameSource& operator=(const FrameSource&) = delete;

    // Fills `rgb` with width*height*3 bytes. Returns false at end of stream.
    bool read(std::vector<std::uint8_t>& rgb);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    std::FILE* stream_ = nullptr;
    bool is_pipe_ = false;
    int width_;
    int height_;
    PixelFormat fmt_;
    std::vector<std::uint8_t> raw_;
};

// Planar I420 -> interleaved RGB24, BT.601 limited range.
void yuv420_to_rgb(const std::uint8_t* yuv, int w, int h, std::uint8_t* rgb);

}  // namespace cattrack
