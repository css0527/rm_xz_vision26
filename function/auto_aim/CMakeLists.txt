# cmake_minimum_required(VERSION 3.16)
# set(OpenVINO_DIR "/home/c/opt/intel/openvino_ubuntu22_2025/runtime/cmake")

# find_package(OpenVINO REQUIRED)
# find_package(Ceres REQUIRED)

# add_subdirectory(ecu)

# add_library(auto_aim OBJECT
#     detector.cpp
#     armor.cpp
#     classifier.cpp

# )

# target_link_libraries(auto_aim openvino::runtime ${CERES_LIBRARIES} yaml-cpp ecu)

cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/home/c/opt/intel/openvino_ubuntu22_2025/runtime/cmake")

# 移除重复的find_package
# find_package(OpenVINO REQUIRED)
# find_package(Ceres REQUIRED)

# 移除add_subdirectory(ecu)

add_library(auto_aim STATIC
    detector.cpp
    armor.cpp
    classifier.cpp
)

# 注意：这里不链接ecu，避免循环依赖
# 可执行文件应该同时链接ecu和auto_aim
# target_link_libraries(auto_aim ecu openvino::runtime ${CERES_LIBRARIES} yaml-cpp)