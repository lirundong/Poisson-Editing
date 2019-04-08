//
// Created by lirundong on 2019-04-05.
//

#ifndef POISSON_EDITING_INCLUDE_POISSON_CLONE_HPP_
#define POISSON_EDITING_INCLUDE_POISSON_CLONE_HPP_

#include <iostream>

#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>

#define WITHIN(i, N) (0 <= (i) && (i) < (N))
#define CLAMP(x, lb, ub) (max((lb), min((x), (ub))))
#define TO_PIXEL(x) (static_cast<uint8_t>(CLAMP(x, 0., 255.)))
#define CHECK_EIGEN(exp, solver) do { \
  exp; \
  if (solver.info() != Eigen::Success) { \
    std::cerr << #exp " failed" << std::endl; \
  } \
} while(0)

namespace poisson {

using std::max;
using std::min;
using cv::Mat;
using cv::Size;
using cv::Rect;
using cv::Vec3b;
using cv::resize;

typedef struct CloneCfg {
  int x1, y1;
  int obj_size;
} CloneCfg;

Mat seamless_clone(const Mat &back, const Mat &forge,
                   const Mat &mask, const CloneCfg &cfg);

}

#endif //POISSON_EDITING_INCLUDE_POISSON_CLONE_HPP_
