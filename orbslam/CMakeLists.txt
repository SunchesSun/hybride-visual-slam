cmake_minimum_required(VERSION 3.10)
project(pyorbslam3_custom)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

set(ORB_SLAM3_DIR "/home/sunches/vkr/ORB_SLAM3" CACHE PATH "Path to ORB_SLAM3 source root")
if(NOT EXISTS "${ORB_SLAM3_DIR}/include/System.h")
    message(FATAL_ERROR "Invalid ORB_SLAM3_DIR: ${ORB_SLAM3_DIR}. Expected to find include/System.h")
endif()

pybind11_add_module(pyorbslam3_custom wrapper.cpp)

target_include_directories(pyorbslam3_custom PRIVATE
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/DBoW2
    ${ORB_SLAM3_DIR}/Thirdparty/g2o/g2o
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

set(ORB_SLAM3_LIB ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so)

target_link_libraries(pyorbslam3_custom PRIVATE
    ${ORB_SLAM3_LIB}
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
)
target_include_directories(pyorbslam3_custom PRIVATE ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
target_link_libraries(pyorbslam3_custom PRIVATE ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
target_compile_definitions(pyorbslam3_custom PRIVATE GL_GLEXT_PROTOTYPES)
set_target_properties(pyorbslam3_custom PROPERTIES
    CXX_STANDARD 14
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/..
)
