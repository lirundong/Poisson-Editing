#include <iostream>

#include "poisson_clone.hpp"

namespace poisson {

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
  CV_Assert(back_roi.size() == obj.size());

  // clone obj to back_roi by Poisson filling
  triplets A_trip;
  vector<int> spatial2omega(obj_h * obj_w, -1), omega2spatial(obj_h * obj_w, -1);
  int spatial_idx{0}, omega_idx{0};

  // build index mapping: plain spatial index <-> index in Omega
  for (auto b = obj_mask.begin<uint8_t>(), e = obj_mask.end<uint8_t>();
       b != e; ++b, ++spatial_idx) {
    if (*b) {
      spatial2omega[spatial_idx] = omega_idx;
      omega2spatial[omega_idx] = spatial_idx;
      omega_idx++;
    }
  }

  vector<double> b_B, b_G, b_R;
  A_trip.reserve(omega_idx * 5);  // full Laplacian
  b_B.reserve(omega_idx);
  b_G.reserve(omega_idx);
  b_R.reserve(omega_idx);
  for (int i{0}, ii, j, y, x, n4_count; i < omega_idx; ++i) {
    // i: index of (y, x)
    // j: index of N(i)
    n4_count = 0;
    ii = omega2spatial[i];
    y = ii / obj_w;
    x = ii % obj_w;
    Vec3i b{0, 0, 0}, I_f{obj.at<Vec3b>(y, x)}, I_b{back_roi.at<Vec3b>(y, x)};

    // update rhs Laplacian
    auto add_laplacian = [&](int y, int x) {
      if (WITHIN(y, obj_h) && WITHIN(x, obj_w)) {
        n4_count++;
        j = y * obj_w + x;
        Vec3i I_ft{obj.at<Vec3b>(y, x)};
        b += I_f - I_ft;
        if (obj_mask.at<uint8_t>(y, x)) {  // in Omega
          CV_Assert(spatial2omega[j] >= 0);
          A_trip.emplace_back(i, spatial2omega[j], -1.);
        } else {  // on border
          Vec3i I_bt{back_roi.at<Vec3b>(y, x)};
          b += I_bt;
        }
      }
    };

    add_laplacian(y - 1, x);
    add_laplacian(y + 1, x);
    add_laplacian(y, x - 1);
    add_laplacian(y, x + 1);

    b_B.push_back(double(b[0]));
    b_G.push_back(double(b[1]));
    b_R.push_back(double(b[2]));

    // update lhs Laplacian
    A_trip.emplace_back(i, i, double(n4_count));
  }

  // solve Poisson by Eigen
  spMat A(omega_idx, omega_idx);
  A.setFromTriplets(A_trip.begin(), A_trip.end());
  VectorXd bB{vecMap(b_B.data(), b_B.size())},
      bG{vecMap(b_G.data(), b_G.size())},
      bR{vecMap(b_R.data(), b_R.size())}, xB, xG, xR;

  Eigen::SimplicialLDLT<spMat> solver(A);
  EIGEN_CHECK(xB = solver.solve(bB), solver);
  EIGEN_CHECK(xG = solver.solve(bG), solver);
  EIGEN_CHECK(xR = solver.solve(bR), solver);

#ifndef NDEBUG
  std::cout << "xB sum: " << xB.sum() << std::endl;
  std::cout << "xG sum: " << xG.sum() << std::endl;
  std::cout << "xR sum: " << xR.sum() << std::endl;
#endif

  // fill resolved data to image
  for (int i{0}, ii, y, x; i < omega_idx; ++i) {
    ii = omega2spatial[i];
    y = ii / obj_w;
    x = ii % obj_w;
    back_roi.at<Vec3b>(y, x) = {TO_PIXEL(xB(i)), TO_PIXEL(xG(i)), TO_PIXEL(xR(i))};
  }

  return ret;
}

}
