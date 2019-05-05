#include <mutex>
#include <limits>

namespace poisson {

template<typename MaskT>
Mat fill_nearest(const Mat &img, const Mat &mask, const MaskT fore_val) {
  CV_Assert(img.size() == mask.size());
  CV_Assert(img.channels() == 3);
  CV_Assert(mask.channels() == 1);

  Mat ret = img.clone();
  ret.forEach<Vec3b>([&](Vec3b &pixel, const int *position) -> void {
    const Point2i src_p(position[0], position[1]);
    if (fore_val != mask.at<MaskT>(src_p)) {
      std::mutex dist_mutex;  // TODO: is this necessary?
      int min_dist{std::numeric_limits<int>::max()};
      Vec3b min_dist_pixel{img.at<Vec3b>(src_p)};
      // find nearest pixel
      mask.forEach<MaskT>([&](MaskT &mask_val, const int *mask_p) -> void {
        if (fore_val == mask_val) {
          const Point2i dst_p(mask_p[0], mask_p[1]);
          auto dist = l2_dist(src_p, dst_p);
          std::lock_guard<std::mutex> g(dist_mutex);
          if (dist < min_dist) {
            min_dist_pixel = img.at<Vec3b>(dst_p);
            min_dist = dist;
          }
        }
      });
      pixel = min_dist_pixel;
    }
  });

  return ret;
}

}
