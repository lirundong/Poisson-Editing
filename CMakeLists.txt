cmake_minimum_required(VERSION 3.5)

project(poisson)
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

if (${EIGEN3_VERSION_MINOR} GREATER 2)  # Eigen 3.3
  set(link_libs ${OpenCV_LIBS} Eigen3::Eigen)
else ()
  include_directories(${EIGEN3_INCLUDE_DIR})
  set(link_libs ${OpenCV_LIBS})
endif ()

include_directories(${PROJECT_SOURCE_DIR}/include)
set(lib_src
        ${PROJECT_SOURCE_DIR}/src/common.cpp
        ${PROJECT_SOURCE_DIR}/src/poisson_clone.cpp
        ${PROJECT_SOURCE_DIR}/src/poisson_matting.cpp)
set(src
        ${PROJECT_SOURCE_DIR}/tools/visualize.cpp)

add_library(${PROJECT_NAME} SHARED ${lib_src})
target_link_libraries(${PROJECT_NAME} ${link_libs})

set(vis_exe vis_${PROJECT_NAME})
add_executable(${vis_exe} ${src})
target_link_libraries(${vis_exe} ${PROJECT_NAME})
