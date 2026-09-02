// Standalone C++ tracker: camera -> YOLO -> parse -> MultiTracker -> stdout.
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cattrack/camera.h"
#include "cattrack/config.h"
#include "cattrack/detection.h"
#include "cattrack/draw.h"
#include "cattrack/inference.h"
#include "cattrack/multi_tracker.h"
#include "cattrack/preprocess.h"

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

const char* arg_value(int argc, char** argv, const char* flag, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

bool arg_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

constexpr std::uint8_t kPalette[][3] = {
    {0, 255, 0},   {0, 0, 255},   {255, 0, 0},   {0, 255, 255},
    {255, 0, 255}, {255, 128, 0}, {128, 0, 255}, {0, 128, 255},
};

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);  // stream reader closing shouldn't abort us

    const std::string config_path = arg_value(argc, argv, "--config", "config.yaml");
    const cattrack::Config cfg = cattrack::Config::load(config_path);

    const int cam_w = cfg.get_int("camera", "width", 640);
    const int cam_h = cfg.get_int("camera", "height", 480);
    const double cam_fps = cfg.get_double("camera", "fps", 15.0);

    const std::string model_path =
        arg_value(argc, argv, "--model",
                  cfg.get_string("detection", "model_path", "yolo11s.onnx").c_str());
    const double conf_thr = cfg.get_double("detection", "confidence_threshold", 0.15);
    const double det_iou = cfg.get_double("detection", "iou_threshold", 0.4);

    const int max_missed = cfg.get_int("tracking", "max_missed", 45);
    const int min_hits = cfg.get_int("tracking", "min_hits", 3);
    const double trk_iou = cfg.get_double("tracking", "iou_threshold", 0.3);
    const int inference_every =
        std::atoi(arg_value(argc, argv, "--inference-every",
                            std::to_string(cfg.get_int("tracking", "inference_every", 3)).c_str()));

    const std::string source = arg_value(argc, argv, "--source", "");
    const bool emit_frames = arg_flag(argc, argv, "--emit-frames");

    cattrack::YoloSession yolo(model_path, 4);
    const int mw = yolo.model_w();
    const int mh = yolo.model_h();

    // track lifetimes stay consistent regardless of inference cadence
    const int effective_max_missed = std::max(1, max_missed / std::max(1, inference_every));
    cattrack::MultiTracker tracker(effective_max_missed, min_hits, trk_iou, mw, mh);

    cattrack::FrameSource cam(cam_w, cam_h, cam_fps, cattrack::PixelFormat::YUV420, source);
    std::vector<std::uint8_t> frame;

    std::fprintf(stderr,
                 "[cattrack] model=%s in=%dx%d cam=%dx%d every=%d %s\n",
                 model_path.c_str(), mw, mh, cam_w, cam_h, inference_every,
                 emit_frames ? "(emitting RGB frames on stdout)" : "");
    if (!emit_frames) std::printf("frame,id,cx,cy,w,h\n");

    const double scale_x = static_cast<double>(cam_w) / mw;
    const double scale_y = static_cast<double>(cam_h) / mh;

    long frame_no = 0;
    auto t0 = std::chrono::steady_clock::now();
    long fps_frames = 0;

    while (!g_stop && cam.read(frame)) {
        std::vector<cattrack::Track*> confirmed;

        if (frame_no % inference_every == 0) {
            const std::vector<float> chw =
                cattrack::preprocess_frame(frame.data(), cam_w, cam_h, mw, mh);
            const auto out = yolo.run(chw.data(), mh, mw);
            const auto dets = cattrack::parse_yolo_output(out.data, out.n_attrs, out.n_boxes,
                                                          conf_thr, det_iou);
            confirmed = tracker.update(dets);
        } else {
            confirmed = tracker.predict_only();
        }

        if (emit_frames) {
            // header line: "<frame> <n> [<id>,<cx>,<cy>,<w>,<h> ...]" then raw RGB
            char line[512];
            int off = std::snprintf(line, sizeof line, "%ld %zu", frame_no, confirmed.size());
            for (const cattrack::Track* t : confirmed) {
                const int cx = static_cast<int>(t->bbox[0] * scale_x);
                const int cy = static_cast<int>(t->bbox[1] * scale_y);
                const int bw = static_cast<int>(t->bbox[2] * scale_x);
                const int bh = static_cast<int>(t->bbox[3] * scale_y);
                if (off > static_cast<int>(sizeof line) - 40) break;
                off += std::snprintf(line + off, sizeof line - off, " %d,%d,%d,%d,%d", t->id,
                                     cx, cy, bw, bh);

                const auto& c = kPalette[t->id % (sizeof(kPalette) / 3)];
                cattrack::draw_rect(frame.data(), cam_w, cam_h, cx - bw / 2, cy - bh / 2,
                                    cx + bw / 2, cy + bh / 2, c[0], c[1], c[2]);
            }
            line[off++] = '\n';
            std::fwrite(line, 1, off, stdout);
            if (std::fwrite(frame.data(), 1, frame.size(), stdout) != frame.size()) break;
        } else {
            for (const cattrack::Track* t : confirmed) {
                std::printf("%ld,%d,%.1f,%.1f,%.1f,%.1f\n", frame_no, t->id, t->bbox[0],
                            t->bbox[1], t->bbox[2], t->bbox[3]);
            }
        }
        std::fflush(stdout);

        ++frame_no;
        ++fps_frames;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t0).count();
        if (elapsed >= 5.0) {
            std::fprintf(stderr, "[cattrack] %.1f fps  (%zu tracks)\n", fps_frames / elapsed,
                         tracker.tracks().size());
            t0 = now;
            fps_frames = 0;
        }
    }

    std::fprintf(stderr, "[cattrack] stopped after %ld frames\n", frame_no);
    return 0;
}
