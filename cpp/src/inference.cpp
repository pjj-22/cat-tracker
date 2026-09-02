#include "cattrack/inference.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <stdexcept>
#include <vector>

namespace cattrack {

struct YoloSession::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "cattrack"};
    Ort::SessionOptions opts;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    std::string input_name;
    std::string output_name;
    int in_w = 0;
    int in_h = 0;

    Ort::Value output{nullptr};

    Impl(const std::string& model_path, int intra_threads) {
        opts.SetIntraOpNumThreads(intra_threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, model_path.c_str(), opts);

        input_name = session.GetInputNameAllocated(0, alloc).get();
        output_name = session.GetOutputNameAllocated(0, alloc).get();

        const auto shape =  // (1, 3, H, W)
            session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        in_h = static_cast<int>(shape[2]);
        in_w = static_cast<int>(shape[3]);
    }
};

YoloSession::YoloSession(const std::string& model_path, int intra_threads)
    : impl_(std::make_unique<Impl>(model_path, intra_threads)) {}

YoloSession::~YoloSession() = default;

int YoloSession::model_w() const { return impl_->in_w; }
int YoloSession::model_h() const { return impl_->in_h; }

YoloSession::Output YoloSession::run(const float* chw, int model_h, int model_w) {
    const std::array<int64_t, 4> shape{1, 3, model_h, model_w};
    const std::size_t count = static_cast<std::size_t>(3) * model_h * model_w;

    Ort::Value input = Ort::Value::CreateTensor<float>(
        impl_->mem_info, const_cast<float*>(chw), count, shape.data(), shape.size());

    const char* in_names[] = {impl_->input_name.c_str()};
    const char* out_names[] = {impl_->output_name.c_str()};

    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                                      out_names, 1);
    impl_->output = std::move(outputs.front());

    const auto out_shape =
        impl_->output.GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3)
        throw std::runtime_error("cattrack::YoloSession: unexpected output rank");

    return {impl_->output.GetTensorData<float>(),
            static_cast<std::size_t>(out_shape[1]),
            static_cast<std::size_t>(out_shape[2])};
}

}  // namespace cattrack
