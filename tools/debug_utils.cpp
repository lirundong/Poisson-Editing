//
// Created by lirundong on 2019-06-02.
//

#include <string>

#include "poisson"

int main(int argc, char **argv) {
  using std::string;

  const string src_path{"../data/matting/src.png"};
  const string trimap_path{"../data/matting/trimap.png"};
  const string output_path{"./result.png"};

  auto src_img = imread(src_path, cv::IMREAD_COLOR);
  auto trimap = imread(trimap_path, cv::IMREAD_GRAYSCALE);
  auto foreground_img = poisson::fill_nearest(src_img, trimap, 255);
  imwrite(output_path, foreground_img);

  return 0;
}
