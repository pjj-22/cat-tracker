// ONNX Runtime wrapper for the YOLO11 session (ORT hidden behind a pimpl).
#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace cattrack {

class YoloSession {
public:
    explicit YoloSession(const std::string& model_path, int intra_threads = 4);
    ~YoloSession();

    YoloSession(const YoloSession&) = delete;
    YoloSession& operator=(const YoloSession&) = delete;

    struct Output {
        const float* data;      // valid until the next run()
        std::size_t n_attrs;    // 84 for YOLO11
        std::size_t n_boxes;    // 8400 at imgsz 320
    };

    Output run(const float* chw, int model_h, int model_w);

    int model_w() const;
    int model_h() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cattrack
