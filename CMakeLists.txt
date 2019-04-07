cmake_minimum_required(VERSION 3.5)

project(poisson_editing)
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(src ${PROJECT_SOURCE_DIR}/tools/main.cpp
        ${PROJECT_SOURCE_DIR}/src/utils.cpp
        ${PROJECT_SOURCE_DIR}/src/poisson_clone.cpp)

add_executable(${PROJECT_NAME} ${src})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Eigen3::Eigen)
