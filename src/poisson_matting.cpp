#include <mutex>
#include <unordered_map>
#include <Eigen/SparseCholesky>

#include "poisson_matting.hpp"

namespace poisson {

const double ALPHA_FORE = 1.0;
const double ALPHA_BACK = 0.0;
const double ALPHA_OMEGA = -1.0;

Mat global_matting(const Mat &img, const Mat &trimap, const MateCfg &cfg) {
  CV_Assert(img.size() == trimap.size());
  CV_Assert(trimap.channels() == 1);

  const int H = img.rows, W = img.cols;
  std::mutex m;
  vector<int> omega2spatial;
  std::unordered_map<int, int> spatial2omega;
  Mat alpha{cv::Mat::zeros(H, W, CV_64FC1)},
      alpha_map{trimap.clone()},
      f_b_diff{cv::Mat::zeros(H, W, CV_64FC3)},
      f_b_diff_smooth;

  alpha.forEach<double>([&](double &alpha_val, const int *position) -> void {
    const Point2i p(position[1], position[0]);  // TODO: x-y order?
    const auto trimap_val = alpha_map.at<uint8_t>(p);
    if (cfg.fore_val == trimap_val) {
      alpha_val = ALPHA_FORE;
    } else if (cfg.back_val == trimap_val) {
      alpha_val = ALPHA_BACK;
    } else if (cfg.omega_val == trimap_val) {
      alpha_val = ALPHA_OMEGA;
      int spatial_idx = position[0] * W + position[1];
      std::lock_guard<std::mutex> g(m);
      omega2spatial.push_back(spatial_idx);
    }
  });

  for (auto i{omega2spatial.cbegin()}; i != omega2spatial.cend(); ++i) {
    spatial2omega.insert({*i, i - omega2spatial.cbegin()});
  }

  auto fore_filled = fill_nearest(img, alpha_map, cfg.fore_val);
  auto back_filled = fill_nearest(img, alpha_map, cfg.back_val);
  f_b_diff += fore_filled;
  f_b_diff -= back_filled;
  // TODO: check gaussian parameters here
  cv::GaussianBlur(f_b_diff, f_b_diff_smooth, Size(5, 5), 1.0, 1.0);

  // build laplacian
  const auto omega_size = omega2spatial.size();
  triplets A_trip;
  VectorXd bB(omega_size), bG(omega_size), bR(omega_size);
  A_trip.reserve(omega_size * 5);

  for (int i{0}; i < omega_size; ++i) {
    auto[y, x] = idx2yx(omega2spatial[i], W);
    int n4_count{0};
    Vec3d b{0, 0, 0}, I_p{img.at<Vec3b>(y, x)};

    auto add_laplacian = [&](int y, int x) {
      if (WITHIN(y, H) && WITHIN(x, W)) {
        n4_count++;
        int j = yx2idx(y, x, W);
        Vec3d I_pn{img.at<Vec3b>(y, x)};
        b += (I_p - I_pn) / f_b_diff_smooth.at<Vec3d>(y, x);
        auto alpha_pn = alpha_map.at<uint8_t>(y, x);
        if (cfg.omega_val == alpha_pn) {  // in Omega
          A_trip.emplace_back(i, spatial2omega[j], -1.);
        } else if (cfg.fore_val == alpha_pn) {  // on foreground border
          b += Vec3d{1, 1, 1};
        } else if (cfg.back_val == alpha_pn) {  // on background border
          b += Vec3d{-1, -1, -1};
        }
      }
    };

    add_laplacian(y + 1, x);
    add_laplacian(y - 1, x);
    add_laplacian(y, x + 1);
    add_laplacian(y, x - 1);

    A_trip.emplace_back(i, i, static_cast<double>(n4_count));
    bB(i) = b[0];
    bG(i) = b[1];
    bR(i) = b[2];
  }

  // solve Poisson by Eigen
  spMat A(omega_size, omega_size);
  A.setFromTriplets(A_trip.begin(), A_trip.end());
  VectorXd xB, xG, xR;

  Eigen::SimplicialLDLT<spMat> solver(A);
  CHECK_EIGEN(xB = solver.solve(bB), solver);
  CHECK_EIGEN(xG = solver.solve(bG), solver);
  CHECK_EIGEN(xR = solver.solve(bR), solver);

#ifndef NDEBUG
  std::cout << "xB sum: " << xB.sum() << std::endl;
  std::cout << "xG sum: " << xG.sum() << std::endl;
  std::cout << "xR sum: " << xR.sum() << std::endl;
#endif

  for (int i{0}; i < omega_size; ++i) {
    // TODO: fill the results to alpha
    auto[y, x] = idx2yx(omega2spatial[i], W);
    alpha.at<double>(y, x) = xB(i);
  }

  return alpha;
}

}
