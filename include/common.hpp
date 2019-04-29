//
// Created by lirundong on 2019-04-28.
//

#ifndef POISSON_EDITING_INCLUDE_COMMON_HPP_
#define POISSON_EDITING_INCLUDE_COMMON_HPP_

#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

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
using std::vector;

using cv::Mat;
using cv::Size;
using cv::Rect;
using cv::Vec3b;
using cv::Vec3i;
using cv::Point2i;
using cv::resize;
using Eigen::VectorXd;

using triplet = Eigen::Triplet<double>;
using triplets = std::vector<triplet>;
using spMat = Eigen::SparseMatrix<double>;
using vecMap = Eigen::Map<Eigen::VectorXd>;

template<typename T>
inline uint8_t real_to_pixel(const T value) {
  CV_Assert(0.0 <= value && value <= 1.0);
  return TO_PIXEL(value * 255.);
}

template<typename PointT>
inline typename PointT::value_type l2_dist(const PointT &p1, const PointT &p2) {
  auto diff = p2 - p1;
  return diff.x * diff.x + diff.y * diff.y;
}

template<typename PointT>
inline PointT find_nearest(const PointT &p, const vector<PointT> &border) {
  return std::min_element(border.cbegin(), border.cend(),
                          [&](PointT &&p1, PointT &&p2) -> bool {
                            return l2_dist(p1, p) < l2_dist(p2, p);
                          });
}

template<typename MaskT>
Mat fill_nearest(const Mat &img, const Mat &mask, const MaskT fore_val);

}

#endif //POISSON_EDITING_INCLUDE_COMMON_HPP_
