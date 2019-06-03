#include <array>
#include <mutex>
#include <limits>

#include "common.hpp"

namespace poisson {

const int FORE_VAL = 0;
const int BACK_VAL = 1;

template<typename T>
Mat _label_to_image(const Mat &src, const Mat &fb_mask,
                    const Mat &label, const size_t max_label) {
  // 1st pass: map label to foreground pixel value
  vector<T> label2val(max_label);
  label.forEach<int>([&](int &v, const int *p) -> void {
    Point2i xy(p[1], p[0]);
    if (fb_mask.at<uint8_t>(xy) == FORE_VAL) {
      label2val[v] = src.at<T>(xy);
    }
  });

  // 2nd pass: build return image from label map
  Mat ret = src.clone();
  label.forEach<int>([&](int &v, const int *p) -> void {
    Point2i xy(p[1], p[0]);
    ret.at<T>(xy) = label2val[v];
  });

  return ret;
}

Mat fill_nearest(const Mat &img, const Mat &trimap, const int trimap_fore) {
  CV_Assert(img.size() == trimap.size());
  CV_Assert(trimap.channels() == 1);

  Mat fb_mask = trimap.clone();
  fb_mask.forEach<uint8_t>([&](uint8_t &v, const int *p) -> void {
    if (v == trimap_fore) {
      v = FORE_VAL;
    } else {
      v = BACK_VAL;
    }
  });

  Mat dist, label;
  cv::distanceTransform(fb_mask, dist, label, cv::DIST_L2,
                        cv::DIST_MASK_PRECISE, cv::DIST_LABEL_PIXEL);

  double min_label, max_label;
  cv::minMaxIdx(label, &min_label, &max_label, nullptr, nullptr);

  if (img.channels() == 3) {
    return _label_to_image<Vec3b>(img, fb_mask, label, max_label);
  } else if (img.channels() == 1) {
    return _label_to_image<uint8_t>(img, fb_mask, label, max_label);
  } else {
    CV_Error(cv::Error::StsBadArg, "`img` should with either 1 or 3 channels.");
  }

}

}
