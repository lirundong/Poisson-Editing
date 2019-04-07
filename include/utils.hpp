//
// Created by lirundong on 2019-04-05.
//

#ifndef POISSON_EDITING_INCLUDE_UTILS_HPP_
#define POISSON_EDITING_INCLUDE_UTILS_HPP_

#include <Eigen/Dense>


namespace poisson {

using Mask = Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

Mask crop_mask(const Mask &mask);

}

#endif //POISSON_EDITING_INCLUDE_UTILS_HPP_
