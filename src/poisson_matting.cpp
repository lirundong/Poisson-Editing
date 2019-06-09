#include <unordered_map>

#include "poisson_matting.hpp"

namespace poisson {

Mat global_matting(const Mat &img, const Mat &trimap, const MateCfg &cfg) {
  CV_Assert(img.size() == trimap.size());
  CV_Assert(trimap.channels() == 1);

  const int H = img.rows, W = img.cols;
  Mat alpha {trimap.clone()}, trimap_step {trimap.clone()};
  Mat img_gray;
  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

  for (int step = 0; step < cfg.max_iter; ++step) {

    int omega_idx = 0, spatial_idx = 0;
    vector<int> omega2spatial;
    std::unordered_map<int, int> spatial2omega;
    for (auto b = trimap_step.begin<uint8_t>(); b != trimap_step.end<uint8_t>();
         ++b, ++spatial_idx) {
      if (*b == cfg.omega_val) {
        omega2spatial.push_back(spatial_idx);
        spatial2omega.insert({spatial_idx, omega_idx});
        omega_idx++;
      }
    }

    Mat img_fore, img_back, fb_diff {Mat::zeros(H, W, CV_64FC1)};
    img_fore = fill_nearest(img_gray, trimap_step, cfg.fore_val);
    img_back = fill_nearest(img_gray, trimap_step, cfg.back_val);
    img_fore.convertTo(img_fore, CV_64FC1);
    img_back.convertTo(img_back, CV_64FC1);
    fb_diff += img_fore;
    fb_diff -= img_back;

    // TODO: check gaussian parameters here
    cv::GaussianBlur(fb_diff, fb_diff, Size(0, 0), 0.75, 0);

    // build laplacian
    int omega_size = omega2spatial.size();
    triplets A_trip;
    VectorXd b(omega_size);
    A_trip.reserve(omega_size * 5);

    for (int i = 0; i < omega_size; ++i) {
      auto[y, x] = idx2yx(omega2spatial[i], W);
      int n4_count = 0;
      double I = img_gray.at<uint8_t>(y, x), b_I = 0, diff_I = fb_diff.at<double>(y, x);

      auto add_laplacian = [&](int y, int x) {
        if (WITHIN(y, H) && WITHIN(x, W)) {
          n4_count++;
          double J = img_gray.at<uint8_t>(y, x);
          b_I += (I - J) / (diff_I + EPS);
          auto t = trimap_step.at<uint8_t>(y, x);
          if (cfg.omega_val == t) {  // in Omega
            int j = yx2idx(y, x, W);
            A_trip.emplace_back(i, spatial2omega[j], -1.);
          } else if (cfg.fore_val == t) {  // on foreground border
            b_I += 1;
          } else if (cfg.back_val == t) {  // on background border
            // trimap_background_border = 0, do nothing
          }
        }
      };

      add_laplacian(y + 1, x);
      add_laplacian(y - 1, x);
      add_laplacian(y, x + 1);
      add_laplacian(y, x - 1);

      A_trip.emplace_back(i, i, static_cast<double>(n4_count));
      b(i) = b_I;
    }

    // solve Poisson by Eigen
    spMat A(omega_size, omega_size);
    A.setFromTriplets(A_trip.begin(), A_trip.end());
    VectorXd x_alpha;

    EIGEN_SP_SOLVER<spMat> solver(A);
    EIGEN_CHECK(x_alpha = solver.solve(b), solver);

    int f_pos = 0, n_neg = 0, outbound = 0;
    for (int i{0}; i < omega_size; ++i) {
      auto[y, x] = idx2yx(omega2spatial[i], W);
      double a = x_alpha(i);
      if (std::isnan(a) || a < 0.0 || 1.0 < a) {
        outbound++;
        trimap_step.at<uint8_t>(y, x) = cfg.omega_val;
      } else if (0.95 <= a && a <= 1.0) {
        f_pos++;
        trimap_step.at<uint8_t>(y, x) = cfg.fore_val;
      } else if (0.0 <= a && a <= 0.05) {
        n_neg++;
        trimap_step.at<uint8_t>(y, x) = cfg.back_val;
      } else {
        trimap_step.at<uint8_t>(y, x) = cfg.omega_val;
      }
      alpha.at<uint8_t>(y, x) = static_cast<uint8_t>(CLAMP(a, 0., 1.) * 255.);
    }

#ifndef NDEBUG
    std::cout << "[Step " << step << "]:" << std::endl;
    std::cout << "\tx_alpha min: " << x_alpha.minCoeff() << std::endl;
    std::cout << "\tx_alpha max: " << x_alpha.maxCoeff() << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "\t#postive:  " << f_pos << std::endl;
    std::cout << "\t#negative: " << n_neg << std::endl;
    std::cout << "\t#outbound: " << outbound << std::endl;
    std::cout << "================================" << std::endl;
#endif
  }

  return alpha;
}

}
