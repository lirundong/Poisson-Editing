//
// Created by lirundong on 2019-04-28.
//

#ifndef POISSON_INCLUDE_POISSON_MATTING_HPP_
#define POISSON_INCLUDE_POISSON_MATTING_HPP_

#include "common.hpp"

namespace poisson {

struct MateCfg {
  uint8_t fore_val, back_val, omega_val;
  int max_iter;

  MateCfg(const int f, const int b, const int o, const int m) :
    fore_val(f), back_val(b), omega_val(o), max_iter(m) { }
};

Mat global_matting(const Mat &img, const Mat &trimap, const MateCfg &cfg);

}

#endif //POISSON_INCLUDE_POISSON_MATTING_HPP_
