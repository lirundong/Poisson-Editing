#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "poisson"

int main(int argc, char **argv) {
  using std::string;
  using cv::imread;
  using cv::imwrite;

  poisson::CloneCfg cfg {2700, 500, 800};
  const string fore_path{"../data/seamless_clone/Japan.airlines.b777-300.ja733j.arp.jpg"};
  const string back_path{"../data/seamless_clone/Big_Tree_with_Red_Sky_in_the_Winter_Night.jpg"};
  const string mask_path{"../data/seamless_clone/mask.png"};
  const string output_path{"../data/seamless_clone/result.png"};

  auto fore_img = imread(fore_path, cv::IMREAD_COLOR);
  auto back_img = imread(back_path, cv::IMREAD_COLOR);
  auto mask = imread(mask_path, cv::IMREAD_GRAYSCALE);
  auto fused_img = poisson::seamless_clone(back_img, fore_img, mask, cfg);

  imwrite(output_path, fused_img);
  return 0;
}
