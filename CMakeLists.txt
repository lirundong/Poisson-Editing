cmake_minimum_required(VERSION 3.14)

project(poisson)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(INTEL_ROOT /opt/intel CACHE STRING "Root path of Intel performace libraries")
set(public_libs "")
set(private_libs "")
set(include_dirs "")
set(link_dirs "")

# ---[ setup Eigen and OpenCV
list(APPEND public_libs ${OpenCV_LIBS})
if (${EIGEN3_VERSION_MINOR} GREATER 2)  # Eigen 3.3
    list(APPEND public_libs Eigen3::Eigen)
else ()
    list(APPEND include_dirs ${EIGEN3_INCLUDE_DIR})
endif ()

# ---[ setup MKL
# TODO: wrap into FindMKL.cmake
set(MKL_ROOT ${INTEL_ROOT}/mkl)
list(APPEND include_dirs ${MKL_ROOT}/include)
list(APPEND private_libs mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
if (APPLE)
    list(APPEND link_dirs ${INTEL_ROOT}/lib ${MKL_ROOT}/lib)
else ()
    list(APPEND link_dirs ${INTEL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64)
endif ()

# ---[ project source files
list(APPEND include_dirs ${PROJECT_SOURCE_DIR}/include)
set(lib_src
        ${PROJECT_SOURCE_DIR}/src/common.cpp
        ${PROJECT_SOURCE_DIR}/src/poisson_clone.cpp
        ${PROJECT_SOURCE_DIR}/src/poisson_matting.cpp)

add_library(${PROJECT_NAME} ${lib_src})
target_include_directories(${PROJECT_NAME} PUBLIC ${include_dirs})
target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_USE_MKL_ALL)
target_compile_options(${PROJECT_NAME} PRIVATE -m64)
target_link_directories(${PROJECT_NAME} PUBLIC ${link_dirs})
target_link_libraries(${PROJECT_NAME} PRIVATE ${private_libs} PUBLIC ${public_libs})

# ---[ project executables
add_executable(visualize ${PROJECT_SOURCE_DIR}/tools/visualize.cpp)
target_link_libraries(visualize PRIVATE ${PROJECT_NAME} gflags)

add_executable(debug_utils ${PROJECT_SOURCE_DIR}/tools/debug_utils.cpp)
target_link_libraries(debug_utils PRIVATE ${PROJECT_NAME})
