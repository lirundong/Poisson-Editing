#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <boost/algorithm/string.hpp>

#include "poisson"

DEFINE_string(task, "clone", "task to perform: {clone, matting}");
DEFINE_string(cfg, "", "argument of specified task, separated by commas");
DEFINE_string(src, "", "path to source image");
DEFINE_string(src_fore, "", "path to source foreground image");
DEFINE_string(src_back, "", "path to source background image");
DEFINE_string(mask, "", "path to fore/back/unknown map image");
DEFINE_string(dst, "", "path to output image");

int main(int argc, char *argv[]) {
  using std::string;
  using std::cout;
  using std::vector;
  using cv::imread;
  using cv::imwrite;

  gflags::SetUsageMessage("C++ implementation of Poisson matting and clone.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  vector<string> cfg_tokens;
  vector<int> cfg;
  boost::split(cfg_tokens, FLAGS_cfg, boost::is_any_of(" ,"));
  for (const auto &v : cfg_tokens) {
    cfg.push_back(std::stoi(v));
  }

  if (FLAGS_task == "clone") {
    cout << "performing Poisson editing..." << std::endl;

    poisson::CloneCfg clone_cfg{cfg[0], cfg[1], cfg[2]};

    auto fore_img = imread(FLAGS_src_fore, cv::IMREAD_COLOR);
    auto back_img = imread(FLAGS_src_back, cv::IMREAD_COLOR);
    auto mask = imread(FLAGS_mask, cv::IMREAD_GRAYSCALE);
    auto fused_img = poisson::seamless_clone(back_img, fore_img, mask, clone_cfg);
    imwrite(FLAGS_dst, fused_img);

  } else if (FLAGS_task == "matting") {
    cout << "performing Poisson matting..." << std::endl;

    poisson::MateCfg mate_cfg{cfg[0], cfg[1], cfg[2], cfg[3]};

    auto mate_src_img = imread(FLAGS_src, cv::IMREAD_COLOR);
    auto mate_trimap = imread(FLAGS_mask, cv::IMREAD_GRAYSCALE);
    auto mate_result = poisson::global_matting(mate_src_img, mate_trimap, mate_cfg);
    imwrite(FLAGS_dst, mate_result);

  } else {
    std::cerr << "invalid task: " << FLAGS_task << std::endl;
    return -1;

  }

  return 0;
}
