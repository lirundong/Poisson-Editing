#include <unordered_map>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

#include "poisson_matting.hpp"

namespace poisson {

const double ALPHA_FORE = 1.0;
const double ALPHA_BACK = 0.0;
const double ALPHA_OMEGA = -1.0;

Mat global_matting(const Mat &img, const Mat &trimap, const MateCfg &cfg) {
  CV_Assert(img.size() == trimap.size());
  CV_Assert(trimap.channels() == 1);

  const int H = img.rows, W = img.cols;
  int omega_idx = 0, spatial_idx = 0;
  vector<int> omega2spatial;
  std::unordered_map<int, int> spatial2omega;

  for (auto b = trimap.begin<uint8_t>(); b != trimap.end<uint8_t>();
       ++b, ++spatial_idx) {
    if (*b == cfg.omega_val) {
      omega2spatial.push_back(spatial_idx);
      spatial2omega.insert({spatial_idx, omega_idx});
      omega_idx++;
    }
  }

  Mat img_gray, img_fore, img_back, fb_diff;
  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

  img_fore = fill_nearest(img_gray, trimap, cfg.fore_val);
  img_back = fill_nearest(img_gray, trimap, cfg.back_val);
  img_fore.convertTo(img_fore, CV_64FC1);
  img_back.convertTo(img_back, CV_64FC1);
  fb_diff = Mat::zeros(H, W, CV_64FC1);
  fb_diff += img_fore;
  fb_diff -= img_back;
  // TODO: check gaussian parameters here
  cv::GaussianBlur(fb_diff, fb_diff, Size(5, 5), 1.0, 1.0);

  // build laplacian
  int omega_size = omega2spatial.size();
  triplets A_trip;
  VectorXd b(omega_size);
  A_trip.reserve(omega_size * 5);

  for (int i{0}; i < omega_size; ++i) {
    auto[y, x] = idx2yx(omega2spatial[i], W);
    int n4_count{0};
    double I = img_gray.at<uint8_t>(y, x), b_ = 0, diff = fb_diff.at<double>(y, x);

    auto add_laplacian = [&](int y, int x) {
      if (WITHIN(y, H) && WITHIN(x, W)) {
        n4_count++;
        int j = yx2idx(y, x, W);
        double J = img_gray.at<uint8_t>(y, x);
        b_ += (I - J) / diff;
        auto t = trimap.at<uint8_t>(y, x);
        if (cfg.omega_val == t) {  // in Omega
          A_trip.emplace_back(i, spatial2omega[j], -1.);
        } else if (cfg.fore_val == t) {  // on foreground border
          b_ += 1;
        } else if (cfg.back_val == t) {  // on background border
          b_ -= 1;
        }
      }
    };

    add_laplacian(y + 1, x);
    add_laplacian(y - 1, x);
    add_laplacian(y, x + 1);
    add_laplacian(y, x - 1);

    A_trip.emplace_back(i, i, static_cast<double>(n4_count));
    b(i) = b_;
  }

  // solve Poisson by Eigen
  spMat A(omega_size, omega_size);
  A.setFromTriplets(A_trip.begin(), A_trip.end());
  VectorXd x_alpha;

  Eigen::SimplicialLDLT<spMat> solver(A);
  // Eigen::ConjugateGradient<spMat> solver(A);
  CHECK_EIGEN(x_alpha = solver.solve(b), solver);

#ifndef NDEBUG
  std::cout << "x_alpha sum: " << x_alpha.sum() << std::endl;
  std::cout << "x_alpha min: " << x_alpha.minCoeff() << std::endl;
  std::cout << "x_alpha max: " << x_alpha.maxCoeff() << std::endl;
#endif

  Mat alpha = trimap.clone();
  int f_pos = 0, n_neg = 0, nan = 0;
  for (int i{0}; i < omega_size; ++i) {
    auto[y, x] = idx2yx(omega2spatial[i], W);
    double a = x_alpha(i);
    if (std::isnan(a)) {
      nan++;
    } else if (0.95 <= a) {
      f_pos++;
    } else if (a <= 0.05) {
      n_neg++;
    }
    alpha.at<uint8_t>(y, x) = static_cast<uint8_t>(CLAMP(a, 0., 1.) * 255.);
  }

#ifndef NDEBUG
  std::cout << "#postive pixels:\t" << f_pos << std::endl;
  std::cout << "#negative pixels:\t" << n_neg << std::endl;
  std::cout << "#nan pixels:\t" << nan << std::endl;
#endif

  return alpha;
}

}
