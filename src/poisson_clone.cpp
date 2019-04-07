#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "poisson_clone.hpp"
#include "utils.hpp"

namespace poisson {

using std::vector;
using Eigen::Triplet;
using Eigen::VectorXd;
using triplet = Triplet<double>;
using triplets = vector<triplet>;

Mat seamless_clone(const Mat &back, const Mat &forge,
                   const Mat &mask, const CloneCfg &cfg) {
  CV_Assert(forge.size() == mask.size());
  CV_Assert(cfg.y1 < back.rows && cfg.x1 < back.cols);

  Mat ret = back.clone();

  // long edge of object resize to cfg.obj_size
  int obj_size_max = max(mask.rows, mask.cols);
  double scale = double(cfg.obj_size) / double(obj_size_max);
  int obj_w = mask.cols * scale, obj_h = mask.rows * scale;
  CV_Assert(cfg.y1 + obj_h < back.rows && cfg.x1 + obj_w < back.cols);
  Mat obj, obj_mask;
  resize(forge, obj, Size(obj_w, obj_h));
  resize(mask, obj_mask, Size(obj_w, obj_h), 0, 0, cv::INTER_NEAREST);

  // select background image on region (cfg.x1, cfg.y1) with size (obj_w, obj_h)
  Rect roi(cfg.x1, cfg.y1, obj_w, obj_h);
  Mat back_roi = ret(roi);

  // clone obj to back_roi by Poisson filling
  triplets A_trip;
  vector<double> b_B, b_G, b_R;

  for (int y = 0; y < obj_h; ++y) {
    for (int x = 0; x < obj_w; ++x) {
      // TODO: go though channels? same A, different b
      if (obj_mask.at<uint8_t>(y, x)) {
        // TODO: boundary conditions for N4(y, x)
        // i: row-major index of output pixel
        // j: row-major index of current (inner) pixel
        int n4_count = 0, i = y * obj_h + x, j = 0, v_i = int(obj_mask.at<uint8_t>(y, x));
        int bB, bG, bR;
        if (y - 1) {  // N_top
          n4_count++;
          Vec3b &I_f
          if (obj_mask.at<uint8_t>(y - 1, x)) {  // in Omega
            j = (y - 1) * obj_h + x;
            A_trip.push_back(triplet(i, j, -1.));
          } else {  // on border

          }
        }
      }
    }
  }

  return ret;
}

}
