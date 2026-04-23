#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#define private public
#define protected public
#include "System.h"
#undef protected
#undef private

namespace py = pybind11;

namespace {

bool IsFiniteMatrix4(const Eigen::Matrix4f& m) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(m(r, c))) {
                return false;
            }
        }
    }
    return true;
}

cv::Mat NumpyRgbToMat(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image
) {
    const py::buffer_info info = image.request();
    if (info.ndim != 3 || info.shape[2] != 3) {
        throw std::runtime_error("process_mono_rgb expects HxWx3 uint8 image");
    }

    cv::Mat mat(
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[1]),
        CV_8UC3,
        info.ptr
    );
    return mat.clone();
}

cv::Mat NumpyGrayToMat(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image
) {
    const py::buffer_info info = image.request();
    if (info.ndim != 2) {
        throw std::runtime_error("process_mono expects HxW uint8 image");
    }

    cv::Mat mat(
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[1]),
        CV_8UC1,
        info.ptr
    );
    return mat.clone();
}

cv::Mat NumpyDepthToMat(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& depth
) {
    const py::buffer_info info = depth.request();
    if (info.ndim != 2) {
        throw std::runtime_error("process_rgbd expects HxW float32 depth image");
    }

    cv::Mat mat(
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[1]),
        CV_32F,
        info.ptr
    );
    return mat.clone();
}

py::array_t<float> Matrix4ToNumpy(const Eigen::Matrix4f& m) {
    py::array_t<float> out(std::vector<py::ssize_t>{4, 4});
    auto dst = out.mutable_unchecked<2>();
    for (py::ssize_t r = 0; r < 4; ++r) {
        for (py::ssize_t c = 0; c < 4; ++c) {
            dst(r, c) = m(static_cast<int>(r), static_cast<int>(c));
        }
    }
    return out;
}

py::array_t<float> EmptyPointArray() {
    return py::array_t<float>(std::vector<py::ssize_t>{0, 3});
}

std::vector<ORB_SLAM3::Map*> SortedValidMaps(ORB_SLAM3::Atlas* atlas) {
    if (!atlas) {
        return {};
    }

    std::vector<ORB_SLAM3::Map*> maps = atlas->GetAllMaps();
    maps.erase(
        std::remove_if(
            maps.begin(),
            maps.end(),
            [](ORB_SLAM3::Map* map) { return map == nullptr || map->IsBad(); }
        ),
        maps.end()
    );
    std::sort(
        maps.begin(),
        maps.end(),
        [](ORB_SLAM3::Map* a, ORB_SLAM3::Map* b) { return a->GetId() < b->GetId(); }
    );
    return maps;
}

py::array_t<float> PointsToNumpy(const std::vector<float>& xyz) {
    if (xyz.empty()) {
        return EmptyPointArray();
    }

    const py::ssize_t n = static_cast<py::ssize_t>(xyz.size() / 3);
    py::array_t<float> out(std::vector<py::ssize_t>{n, 3});
    std::memcpy(out.mutable_data(), xyz.data(), xyz.size() * sizeof(float));
    return out;
}

std::vector<float> CollectMapPointCoords(const std::vector<ORB_SLAM3::MapPoint*>& points) {
    std::vector<float> out;
    out.reserve(points.size() * 3);

    std::unordered_set<unsigned long> seen_ids;
    seen_ids.reserve(points.size() * 2 + 1);

    for (ORB_SLAM3::MapPoint* p : points) {
        if (!p || p->isBad()) {
            continue;
        }
        if (!seen_ids.insert(p->mnId).second) {
            continue;
        }

        const Eigen::Vector3f w = p->GetWorldPos();
        const float x = w(0);
        const float y = w(1);
        const float z = w(2);
        if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z))) {
            continue;
        }

        out.push_back(x);
        out.push_back(y);
        out.push_back(z);
    }
    return out;
}

}  // namespace


class PyOrbslamSystem {
public:
    PyOrbslamSystem(
        const std::string& vocab_path,
        const std::string& settings_path,
        ORB_SLAM3::System::eSensor sensor,
        bool use_viewer
    ) {
        slam_ = std::make_unique<ORB_SLAM3::System>(
            vocab_path,
            settings_path,
            sensor,
            use_viewer
        );
    }

    ~PyOrbslamSystem() {
        try {
            shutdown();
        } catch (...) {
        }
    }

    bool process_mono_rgb(
        const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image,
        double timestamp
    ) {
        cv::Mat rgb = NumpyRgbToMat(image);
        return TrackMonocular(rgb, timestamp);
    }

    bool process_mono(
        const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image,
        double timestamp
    ) {
        cv::Mat gray = NumpyGrayToMat(image);
        return TrackMonocular(gray, timestamp);
    }

    bool process_rgbd(
        const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& depth,
        double timestamp
    ) {
        cv::Mat rgb = NumpyRgbToMat(image);
        cv::Mat depth_mat = NumpyDepthToMat(depth);
        return TrackRGBD(rgb, depth_mat, timestamp);
    }

    bool process_mono_depth_guided(
        const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image,
        const py::array_t<float, py::array::c_style | py::array::forcecast>& depth,
        double timestamp,
        const ORB_SLAM3::DepthGuidanceConfig& guidance,
        int random_seed
    ) {
        cv::Mat gray = NumpyGrayToMat(image);
        cv::Mat depth_mat = NumpyDepthToMat(depth);
        return TrackMonocularDepthGuided(gray, depth_mat, timestamp, guidance, random_seed);
    }

    py::object get_pose() const {
        if (!has_pose_) {
            return py::none();
        }
        Eigen::Matrix4f Twc = last_tcw_.inverse();
        return Matrix4ToNumpy(Twc);
    }

    int get_tracking_state() const {
        if (!slam_) {
            return 0;
        }
        return slam_->GetTrackingState();
    }

    py::dict get_last_depth_guidance_stats() const {
        py::dict out;
        if (!slam_ || !slam_->mpTracker) {
            out["raw_features"] = 0;
            out["valid_depth_features"] = 0;
            out["enhanced_features"] = 0;
            out["far_points_removed"] = 0;
            out["final_features"] = 0;
            out["invalid_depth_ratio"] = 0.0;
            out["filter_applied"] = false;
            out["fallback_to_raw"] = false;
            return out;
        }

        out["raw_features"] = slam_->mpTracker->GetLastDepthGuidanceRawFeatures();
        out["valid_depth_features"] = slam_->mpTracker->GetLastDepthGuidanceValidDepthFeatures();
        out["enhanced_features"] = slam_->mpTracker->GetLastDepthGuidanceEnhancedFeatures();
        out["far_points_removed"] = slam_->mpTracker->GetLastDepthGuidanceFarPointsRemoved();
        out["final_features"] = slam_->mpTracker->GetLastDepthGuidanceFinalFeatures();
        out["invalid_depth_ratio"] = slam_->mpTracker->GetLastDepthGuidanceInvalidDepthRatio();
        out["filter_applied"] = slam_->mpTracker->WasLastDepthGuidanceFilterApplied();
        out["fallback_to_raw"] = slam_->mpTracker->DidLastDepthGuidanceFallbackToRaw();
        return out;
    }

    py::array_t<float> get_tracked_map_points() const {
        if (!slam_) {
            return EmptyPointArray();
        }

        std::vector<ORB_SLAM3::MapPoint*> tracked;
        {
            py::gil_scoped_release release;
            tracked = slam_->GetTrackedMapPoints();
        }

        return PointsToNumpy(CollectMapPointCoords(tracked));
    }

    py::array_t<float> get_map_points() const {
        return get_current_map_points();
    }

    py::array_t<float> get_current_map_points() const {
        if (!slam_ || !slam_->mpAtlas) {
            return EmptyPointArray();
        }

        std::vector<ORB_SLAM3::MapPoint*> points;
        {
            py::gil_scoped_release release;
            // Для визуализации в Python по умолчанию используем только текущую
            // карту, чтобы не склеивать облака от старых карт Atlas.
            ORB_SLAM3::Map* current = slam_->mpAtlas->GetCurrentMap();
            if (current && !current->IsBad()) {
                const std::vector<ORB_SLAM3::MapPoint*> v = current->GetAllMapPoints();
                points.insert(points.end(), v.begin(), v.end());
            }

            // Fallback: если текущая карта пуста, возвращаем объединение всех карт.
            if (points.empty()) {
                const std::vector<ORB_SLAM3::Map*> maps = slam_->mpAtlas->GetAllMaps();
                for (ORB_SLAM3::Map* map : maps) {
                    if (!map || map->IsBad()) {
                        continue;
                    }
                    const std::vector<ORB_SLAM3::MapPoint*> v = map->GetAllMapPoints();
                    points.insert(points.end(), v.begin(), v.end());
                }
            }
        }

        return PointsToNumpy(CollectMapPointCoords(points));
    }

    py::array_t<float> get_all_map_points() const {
        if (!slam_ || !slam_->mpAtlas) {
            return EmptyPointArray();
        }

        std::vector<ORB_SLAM3::MapPoint*> points;
        {
            py::gil_scoped_release release;
            const std::vector<ORB_SLAM3::Map*> maps = SortedValidMaps(slam_->mpAtlas);
            for (ORB_SLAM3::Map* map : maps) {
                const std::vector<ORB_SLAM3::MapPoint*> v = map->GetAllMapPoints();
                points.insert(points.end(), v.begin(), v.end());
            }
        }

        return PointsToNumpy(CollectMapPointCoords(points));
    }

    py::list get_atlas_map_point_groups() const {
        py::list out;
        if (!slam_ || !slam_->mpAtlas) {
            return out;
        }

        std::vector<ORB_SLAM3::Map*> maps;
        unsigned long current_map_id = 0;
        bool has_current_map = false;
        {
            py::gil_scoped_release release;
            maps = SortedValidMaps(slam_->mpAtlas);
            ORB_SLAM3::Map* current = slam_->mpAtlas->GetCurrentMap();
            if (current && !current->IsBad()) {
                current_map_id = current->GetId();
                has_current_map = true;
            }
        }

        for (ORB_SLAM3::Map* map : maps) {
            const std::vector<float> xyz = CollectMapPointCoords(map->GetAllMapPoints());
            py::dict group;
            group["map_id"] = py::int_(map->GetId());
            group["is_current"] = py::bool_(
                has_current_map && map->GetId() == current_map_id
            );
            group["points"] = PointsToNumpy(xyz);
            out.append(group);
        }

        return out;
    }

    py::array_t<float> get_current_map_keyframe_trajectory() const {
        if (!slam_ || !slam_->mpAtlas) {
            return EmptyPointArray();
        }

        std::vector<ORB_SLAM3::KeyFrame*> keyframes;
        {
            py::gil_scoped_release release;
            ORB_SLAM3::Map* current = slam_->mpAtlas->GetCurrentMap();
            if (current && !current->IsBad()) {
                keyframes = current->GetAllKeyFrames();
            }
        }

        if (keyframes.empty()) {
            return EmptyPointArray();
        }

        std::sort(
            keyframes.begin(),
            keyframes.end(),
            [](const ORB_SLAM3::KeyFrame* a, const ORB_SLAM3::KeyFrame* b) {
                if (a == nullptr) {
                    return false;
                }
                if (b == nullptr) {
                    return true;
                }
                return a->mnId < b->mnId;
            }
        );

        std::vector<float> out;
        out.reserve(keyframes.size() * 3);
        for (ORB_SLAM3::KeyFrame* kf : keyframes) {
            if (!kf || kf->isBad()) {
                continue;
            }
            const Sophus::SE3f Twc = kf->GetPoseInverse();
            const Eigen::Vector3f t = Twc.translation();
            const float x = t(0);
            const float y = t(1);
            const float z = t(2);
            if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z))) {
                continue;
            }
            out.push_back(x);
            out.push_back(y);
            out.push_back(z);
        }

        return PointsToNumpy(out);
    }

    void save_trajectory_tum(const std::string& path) {
        if (!slam_) {
            return;
        }
        py::gil_scoped_release release;
        slam_->SaveTrajectoryTUM(path);
    }

    void save_keyframe_trajectory_tum(const std::string& path) {
        if (!slam_) {
            return;
        }
        py::gil_scoped_release release;
        slam_->SaveKeyFrameTrajectoryTUM(path);
    }

    void shutdown() {
        if (!slam_ || is_shutdown_) {
            return;
        }
        py::gil_scoped_release release;
        slam_->Shutdown();
        is_shutdown_ = true;
    }

private:
    bool TrackMonocular(const cv::Mat& image, double timestamp) {
        if (!slam_ || is_shutdown_) {
            return false;
        }

        Sophus::SE3f Tcw;
        {
            py::gil_scoped_release release;
            Tcw = slam_->TrackMonocular(image, timestamp);
        }

        const Eigen::Matrix4f m = Tcw.matrix();
        if (IsFiniteMatrix4(m)) {
            last_tcw_ = m;
            has_pose_ = true;
        }

        const int state = slam_->GetTrackingState();
        return state != 0;
    }

    bool TrackMonocularDepthGuided(
        const cv::Mat& image,
        const cv::Mat& depth,
        double timestamp,
        const ORB_SLAM3::DepthGuidanceConfig& guidance,
        int random_seed
    ) {
        if (!slam_ || is_shutdown_) {
            return false;
        }

        Sophus::SE3f Tcw;
        {
            py::gil_scoped_release release;
            Tcw = slam_->TrackMonocularDepthGuided(image, depth, timestamp, guidance, random_seed);
        }

        const Eigen::Matrix4f m = Tcw.matrix();
        if (IsFiniteMatrix4(m)) {
            last_tcw_ = m;
            has_pose_ = true;
        }

        const int state = slam_->GetTrackingState();
        return state != 0;
    }

    bool TrackRGBD(const cv::Mat& image, const cv::Mat& depth, double timestamp) {
        if (!slam_ || is_shutdown_) {
            return false;
        }

        Sophus::SE3f Tcw;
        {
            py::gil_scoped_release release;
            Tcw = slam_->TrackRGBD(image, depth, timestamp);
        }

        const Eigen::Matrix4f m = Tcw.matrix();
        if (IsFiniteMatrix4(m)) {
            last_tcw_ = m;
            has_pose_ = true;
        }

        const int state = slam_->GetTrackingState();
        return state != 0;
    }

    std::unique_ptr<ORB_SLAM3::System> slam_;
    Eigen::Matrix4f last_tcw_ = Eigen::Matrix4f::Identity();
    bool has_pose_ = false;
    bool is_shutdown_ = false;
};


PYBIND11_MODULE(pyorbslam3_custom, m) {
    m.doc() = "Custom ORB-SLAM3 pybind11 wrapper";

    py::class_<ORB_SLAM3::DepthGuidanceConfig>(m, "DepthGuidanceConfig")
        .def(py::init<>())
        .def_readwrite("depth_threshold_m", &ORB_SLAM3::DepthGuidanceConfig::depth_threshold_m)
        .def_readwrite(
            "far_point_drop_probability",
            &ORB_SLAM3::DepthGuidanceConfig::far_point_drop_probability
        )
        .def_readwrite(
            "recursive_keep_ratio",
            &ORB_SLAM3::DepthGuidanceConfig::recursive_keep_ratio
        )
        .def_readwrite(
            "max_recursion_depth",
            &ORB_SLAM3::DepthGuidanceConfig::max_recursion_depth
        )
        .def_readwrite("min_block_points", &ORB_SLAM3::DepthGuidanceConfig::min_block_points)
        .def_readwrite(
            "min_feature_retention_ratio",
            &ORB_SLAM3::DepthGuidanceConfig::min_feature_retention_ratio
        )
        .def_readwrite(
            "min_feature_retention_count",
            &ORB_SLAM3::DepthGuidanceConfig::min_feature_retention_count
        )
        .def_readwrite(
            "max_invalid_depth_ratio",
            &ORB_SLAM3::DepthGuidanceConfig::max_invalid_depth_ratio
        );

    py::enum_<ORB_SLAM3::System::eSensor>(m, "Sensor")
        .value("MONOCULAR", ORB_SLAM3::System::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::STEREO)
        .value("RGBD", ORB_SLAM3::System::RGBD)
        .value("IMU_MONOCULAR", ORB_SLAM3::System::IMU_MONOCULAR)
        .value("IMU_STEREO", ORB_SLAM3::System::IMU_STEREO)
        .value("IMU_RGBD", ORB_SLAM3::System::IMU_RGBD);

    py::class_<PyOrbslamSystem>(m, "System")
        .def(
            py::init<
                const std::string&,
                const std::string&,
                ORB_SLAM3::System::eSensor,
                bool
            >(),
            py::arg("vocab_path"),
            py::arg("settings_path"),
            py::arg("sensor"),
            py::arg("use_viewer") = false
        )
        .def("process_mono_rgb", &PyOrbslamSystem::process_mono_rgb)
        .def("process_mono", &PyOrbslamSystem::process_mono)
        .def("process_rgbd", &PyOrbslamSystem::process_rgbd)
        .def("process_mono_depth_guided", &PyOrbslamSystem::process_mono_depth_guided)
        .def("get_pose", &PyOrbslamSystem::get_pose)
        .def("get_tracking_state", &PyOrbslamSystem::get_tracking_state)
        .def("get_last_depth_guidance_stats", &PyOrbslamSystem::get_last_depth_guidance_stats)
        .def("get_map_points", &PyOrbslamSystem::get_map_points)
        .def("get_current_map_points", &PyOrbslamSystem::get_current_map_points)
        .def("get_all_map_points", &PyOrbslamSystem::get_all_map_points)
        .def("get_atlas_map_point_groups", &PyOrbslamSystem::get_atlas_map_point_groups)
        .def("get_current_map_keyframe_trajectory", &PyOrbslamSystem::get_current_map_keyframe_trajectory)
        .def("get_tracked_map_points", &PyOrbslamSystem::get_tracked_map_points)
        .def("save_trajectory_tum", &PyOrbslamSystem::save_trajectory_tum)
        .def("save_keyframe_trajectory_tum", &PyOrbslamSystem::save_keyframe_trajectory_tum)
        .def("shutdown", &PyOrbslamSystem::shutdown);
}
