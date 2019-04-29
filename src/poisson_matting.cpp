#include "poisson_matting.hpp"

namespace poisson {

Mat global_matting(const Mat &img, const Mat &trimap, const MateCfg &cfg) {
  CV_Assert(img.size() == trimap.size());
  CV_Assert(trimap.channels() == 1);

  const int H = img.rows, W = img.cols;
  int omega_idx{0}, spatial_idx{0};
  vector<int> omega2spatial(H * W, -1), spatial2omega(H * W, -1);
  vector<Point2i> border_f, border_b;  // interior border
  Mat alpha{cv::Mat::zeros(H, W, CV_64FC1)},
      f_b_diff{cv::Mat::zeros(H, W, CV_64FC3)},
      f_b_diff_smooth;

  auto fore_filled = fill_nearest(img, trimap, cfg.fore_val);
  auto back_filled = fill_nearest(img, trimap, cfg.back_val);
  f_b_diff += fore_filled;
  f_b_diff -= back_filled;
  // TODO: check gaussian parameters here
  cv::GaussianBlur(f_b_diff, f_b_diff_smooth, Size(5, 5), 1.0, 1.0);


  return alpha;
}

}
