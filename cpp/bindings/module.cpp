// pybind11 surface for the C++ tracking core: Kalman filter, assignment
// solver, and the Track / MultiTracker cycle, matching the Python API.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cattrack/config.h"
#include "cattrack/detection.h"
#include "cattrack/geom.h"
#include "cattrack/preprocess.h"
#include "cattrack/hungarian.h"
#include "cattrack/kalman.h"
#include "cattrack/multi_tracker.h"
#include "cattrack/track.h"

namespace py = pybind11;

namespace {

std::array<double, 4> to_bbox(const py::array_t<double>& a) {
    auto buf = a.unchecked<1>();
    if (buf.shape(0) != 4) throw std::runtime_error("bbox must have length 4");
    return {buf(0), buf(1), buf(2), buf(3)};
}

py::array_t<double> vec4(const std::array<double, 4>& v) {
    py::array_t<double> out(4);
    auto w = out.mutable_unchecked<1>();
    for (int i = 0; i < 4; ++i) w(i) = v[i];
    return out;
}

py::array_t<double> vec2(const std::array<double, 2>& v) {
    py::array_t<double> out(2);
    auto w = out.mutable_unchecked<1>();
    w(0) = v[0];
    w(1) = v[1];
    return out;
}

py::tuple assignment(const py::array_t<double, py::array::c_style | py::array::forcecast>& cost) {
    if (cost.ndim() != 2) throw std::runtime_error("cost matrix must be 2D");
    const std::size_t rows = static_cast<std::size_t>(cost.shape(0));
    const std::size_t cols = static_cast<std::size_t>(cost.shape(1));
    std::vector<double> flat(cost.data(), cost.data() + rows * cols);

    auto res = cattrack::linear_sum_assignment(flat, rows, cols);

    py::array_t<long> ri(res.row_ind.size());
    py::array_t<long> ci(res.col_ind.size());
    auto rw = ri.mutable_unchecked<1>();
    auto cw = ci.mutable_unchecked<1>();
    for (std::size_t k = 0; k < res.row_ind.size(); ++k) {
        rw(k) = res.row_ind[k];
        cw(k) = res.col_ind[k];
    }
    return py::make_tuple(ri, ci);
}

py::list detections_to_py(const std::vector<cattrack::Detection>& dets) {
    py::list out;
    for (const auto& d : dets) {
        py::dict e;
        e["box"] = vec4(d.box);
        e["confidence"] = d.confidence;
        out.append(e);
    }
    return out;
}

// detections: [{'box': np.ndarray(4), 'confidence': float}, ...]
std::vector<cattrack::Detection> to_detections(const py::iterable& items) {
    std::vector<cattrack::Detection> out;
    for (py::handle item : items) {
        py::dict d = py::reinterpret_borrow<py::dict>(item);
        cattrack::Detection det;
        det.box = to_bbox(d["box"].cast<py::array_t<double>>());
        det.confidence = d["confidence"].cast<double>();
        out.push_back(det);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(cattrack_cpp, m) {
    m.doc() = "C++ tracking core for cat-tracker";

    py::class_<cattrack::BBoxKalmanFilter>(m, "BBoxKalmanFilter")
        .def(py::init([](const py::array_t<double>& bbox) {
            return cattrack::BBoxKalmanFilter(to_bbox(bbox));
        }))
        .def("predict", [](cattrack::BBoxKalmanFilter& self) { return vec4(self.predict()); })
        .def("update", [](cattrack::BBoxKalmanFilter& self, const py::array_t<double>& bbox) {
            self.update(to_bbox(bbox));
        })
        .def("get_state", [](const cattrack::BBoxKalmanFilter& self) { return vec4(self.state()); })
        .def("get_velocity",
             [](const cattrack::BBoxKalmanFilter& self) { return vec2(self.velocity()); })
        .def("on_missed", &cattrack::BBoxKalmanFilter::on_missed)
        .def("compensate_camera_motion", &cattrack::BBoxKalmanFilter::compensate_camera_motion,
             py::arg("dx"), py::arg("dy"))
        .def("_x", &cattrack::BBoxKalmanFilter::x, py::arg("i"))
        .def("_P", &cattrack::BBoxKalmanFilter::P, py::arg("i"), py::arg("j"));

    py::class_<cattrack::Track>(m, "Track")
        .def(py::init([](const py::array_t<double>& bbox, double confidence,
                         int track_id, int min_hits) {
                 return cattrack::Track(to_bbox(bbox), confidence, track_id, min_hits);
             }),
             py::arg("bbox"), py::arg("confidence"), py::arg("track_id") = 1,
             py::arg("min_hits") = 3)
        .def("predict", [](cattrack::Track& self) { return vec4(self.predict()); })
        .def("update", [](cattrack::Track& self, const py::array_t<double>& bbox, double conf) {
            self.update(to_bbox(bbox), conf);
        })
        .def("mark_missed", &cattrack::Track::mark_missed)
        .def("is_confirmed", &cattrack::Track::is_confirmed)
        .def("should_delete", &cattrack::Track::should_delete, py::arg("max_missed") = 10)
        .def_readonly("id", &cattrack::Track::id)
        .def_readwrite("confidence", &cattrack::Track::confidence)
        .def_readwrite("hits", &cattrack::Track::hits)
        .def_readwrite("missed_frames", &cattrack::Track::missed_frames)
        .def_readwrite("age", &cattrack::Track::age)
        .def_readwrite("name", &cattrack::Track::name)
        .def_readwrite("name_confidence", &cattrack::Track::name_confidence)
        .def_readwrite("_candidate_name", &cattrack::Track::candidate_name)
        .def_readwrite("_candidate_streak", &cattrack::Track::candidate_streak)
        .def_property_readonly("bbox", [](const cattrack::Track& self) { return vec4(self.bbox); })
        .def_property_readonly("predicted_bbox",
                               [](const cattrack::Track& self) { return vec4(self.predicted_bbox); })
        .def_property_readonly("velocity",
                               [](const cattrack::Track& self) { return vec2(self.velocity()); })
        .def_property_readonly("kf", [](cattrack::Track& self) -> cattrack::BBoxKalmanFilter& {
            return self.kf();
        }, py::return_value_policy::reference_internal);

    py::class_<cattrack::MultiTracker>(m, "MultiTracker")
        .def(py::init<int, int, double, int, int>(), py::arg("max_missed") = 10,
             py::arg("min_hits") = 3, py::arg("iou_threshold") = 0.3,
             py::arg("model_w") = 320, py::arg("model_h") = 320)
        .def("update", [](cattrack::MultiTracker& self, const py::iterable& detections) {
            return self.update(to_detections(detections));
        }, py::return_value_policy::reference_internal)
        .def("predict_only", &cattrack::MultiTracker::predict_only,
             py::return_value_policy::reference_internal)
        .def("compensate_camera_motion", &cattrack::MultiTracker::compensate_camera_motion,
             py::arg("dx"), py::arg("dy"))
        .def_property_readonly("tracks", &cattrack::MultiTracker::tracks,
                               py::return_value_policy::reference_internal);

    m.def("parse_yolo_output",
          [](const py::array_t<float, py::array::c_style | py::array::forcecast>& out,
             double conf_threshold, double iou_threshold) {
              if (out.ndim() != 3)
                  throw std::runtime_error("output must be (1, n_attrs, n_boxes)");
              const auto n_attrs = static_cast<std::size_t>(out.shape(1));
              const auto n_boxes = static_cast<std::size_t>(out.shape(2));
              return detections_to_py(cattrack::parse_yolo_output(
                  out.data(), n_attrs, n_boxes, conf_threshold, iou_threshold));
          },
          py::arg("output"), py::arg("conf_threshold") = 0.15,
          py::arg("iou_threshold") = 0.4);

    m.def("preprocess_frame",
          [](const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& rgb,
             int model_w, int model_h) {
              if (rgb.ndim() != 3 || rgb.shape(2) != 3)
                  throw std::runtime_error("frame must be (H, W, 3) uint8");
              const int src_h = static_cast<int>(rgb.shape(0));
              const int src_w = static_cast<int>(rgb.shape(1));
              auto chw = cattrack::preprocess_frame(rgb.data(), src_w, src_h, model_w, model_h);

              py::array_t<float> out({1, 3, model_h, model_w});
              std::memcpy(out.mutable_data(), chw.data(), chw.size() * sizeof(float));
              return out;
          },
          py::arg("frame"), py::arg("model_w"), py::arg("model_h"));

    py::class_<cattrack::Config>(m, "Config")
        .def_static("load", &cattrack::Config::load, py::arg("path"))
        .def("has", &cattrack::Config::has, py::arg("section"), py::arg("key"))
        .def("get_string", &cattrack::Config::get_string, py::arg("section"),
             py::arg("key"), py::arg("def") = "")
        .def("get_int", &cattrack::Config::get_int, py::arg("section"), py::arg("key"),
             py::arg("def") = 0)
        .def("get_double", &cattrack::Config::get_double, py::arg("section"),
             py::arg("key"), py::arg("def") = 0.0)
        .def("get_bool", &cattrack::Config::get_bool, py::arg("section"), py::arg("key"),
             py::arg("def") = false)
        .def("get_doubles", &cattrack::Config::get_doubles, py::arg("section"),
             py::arg("key"));

    m.def("iou", [](const py::array_t<double>& a, const py::array_t<double>& b) {
        return cattrack::iou(to_bbox(a), to_bbox(b));
    });
    m.def("euclidean_distance", [](const py::array_t<double>& a, const py::array_t<double>& b) {
        return cattrack::euclidean_distance(to_bbox(a), to_bbox(b));
    });

    m.def("linear_sum_assignment", &assignment,
          "Minimum cost rectangular assignment (scipy-compatible).");
}
