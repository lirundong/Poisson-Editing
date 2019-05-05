//
// Created by lirundong on 2019-04-28.
//

#ifndef POISSON_EDITING_INCLUDE_COMMON_HPP_
#define POISSON_EDITING_INCLUDE_COMMON_HPP_

#include <algorithm>
#include <vector>
#include <map>
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
  auto solver_info = solver.info(); \
  if (solver_info != Eigen::Success) { \
    auto err_info = EIGEN_COMPUTATION_ERROR.find(solver_info); \
    std::cerr << "`" #exp "` failed," << std::endl \
              << err_info->second << std::endl; \
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
using cv::Vec3d;
using cv::Point2i;
using cv::resize;
using Eigen::VectorXd;

using triplet = Eigen::Triplet<double>;
using triplets = std::vector<triplet>;
using spMat = Eigen::SparseMatrix<double>;
using vecMap = Eigen::Map<Eigen::VectorXd>;

static const std::map<Eigen::ComputationInfo, std::string>
    EIGEN_COMPUTATION_ERROR{
    {Eigen::NumericalIssue, "The provided data did not satisfy the prerequisites"},
    {Eigen::NoConvergence, "Iterative procedure did not converge"},
    {Eigen::InvalidInput, "The inputs are invalid, or the algorithm has been "
                          "improperly called"},
};

template<typename T>
inline uint8_t real_to_pixel(const T value) {
  CV_Assert(0.0 <= value && value <= 1.0);
  return TO_PIXEL(value * 255.);
}

template<typename PointT>
inline decltype(auto) l2_dist(PointT &&p1, PointT &&p2) {
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

inline std::tuple<int, int> idx2yx(const int idx, const int W) {
  const int x = idx % W, y = idx / W;
  return {y, x};
}

inline int xy2idx(const int x, const int y, const int W) {
  return y * W + x;
}

inline int yx2idx(const int y, const int x, const int W) {
  return y * W + x;
}

inline Vec3d operator/(const Vec3d &lhs, const Vec3d &rhs) {
  return {lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2]};
}

template<typename MaskT>
Mat fill_nearest(const Mat &img, const Mat &mask, const MaskT fore_val);

}

#include "common_impl.hpp"

#endif //POISSON_EDITING_INCLUDE_COMMON_HPP_
