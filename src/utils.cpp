//
// Created by lirundong on 2019-04-05.
//

#include <array>
#include "utils.hpp"

namespace poisson {

Mask crop_mask(const Mask &mask) {
  std::array<int, 4> idx_border {-1, -1, -1, -1}; // [y1, y2], [x1, x2]
  auto valid_row = mask.rowwise().any();
  auto valid_col = mask.colwise().any();

  return Mask(4, 4);
}

}
