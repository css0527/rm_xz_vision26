cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/home/c/opt/intel/openvino_ubuntu22_2025/runtime/cmake")

find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT
    rune_detector.cpp
    rune_solver.cpp
    power_rune.cpp
)

target_link_libraries(auto_buff openvino::runtime ${CERES_LIBRARIES} yaml-cpp)