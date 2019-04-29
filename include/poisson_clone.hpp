//
// Created by lirundong on 2019-04-05.
//

#ifndef POISSON_EDITING_INCLUDE_POISSON_CLONE_HPP_
#define POISSON_EDITING_INCLUDE_POISSON_CLONE_HPP_

#include "common.hpp"

namespace poisson {

struct CloneCfg {
  int x1, y1;
  int obj_size;
};

Mat seamless_clone(const Mat &back, const Mat &forge,
                   const Mat &mask, const CloneCfg &cfg);

}

#endif //POISSON_EDITING_INCLUDE_POISSON_CLONE_HPP_
