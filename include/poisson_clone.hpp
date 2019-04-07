//
// Created by lirundong on 2019-04-05.
//

#ifndef POISSON_EDITING_INCLUDE_POISSON_CLONE_HPP_
#define POISSON_EDITING_INCLUDE_POISSON_CLONE_HPP_

#include <opencv2/opencv.hpp>

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
