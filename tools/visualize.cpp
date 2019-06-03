#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "poisson"

int main(int argc, char **argv) {
  using std::string;
  using std::cout;
  using cv::imread;
  using cv::imwrite;

//  cout << "performing Poisson editing..." << std::endl;
//
//  poisson::CloneCfg clone_cfg{2700, 500, 800};
//  const string fore_path{"../data/seamless_clone/Japan.airlines.b777-300.ja733j.arp.jpg"};
//  const string back_path{"../data/seamless_clone/Big_Tree_with_Red_Sky_in_the_Winter_Night.jpg"};
//  const string mask_path{"../data/seamless_clone/mask.png"};
//  const string edit_output_path{"../data/seamless_clone/result.png"};
//
//  auto fore_img = imread(fore_path, cv::IMREAD_COLOR);
//  auto back_img = imread(back_path, cv::IMREAD_COLOR);
//  auto mask = imread(mask_path, cv::IMREAD_GRAYSCALE);
//  auto fused_img = poisson::seamless_clone(back_img, fore_img, mask, clone_cfg);
//  imwrite(edit_output_path, fused_img);

  cout << "performing Poisson matting..." << std::endl;

  poisson::MateCfg mate_cfg{255, 0, 128, 10};
  const string mate_src_path{"../data/matting/src.png"};
  const string mate_trimap_path{"../data/matting/trimap.png"};
  const string mate_output_path{"../data/matting/result.png"};

  auto mate_src_img = imread(mate_src_path, cv::IMREAD_COLOR);
  auto mate_trimap = imread(mate_trimap_path, cv::IMREAD_GRAYSCALE);
  auto mate_result = poisson::global_matting(mate_src_img, mate_trimap, mate_cfg);
  imwrite(mate_output_path, mate_result);

  return 0;
}
