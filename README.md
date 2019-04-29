# Poisson Image Editing

> Author: Rundong Li<br/>
> Mail: lird@shanghaitech.edu.cn

![seamless_clone](data/seamless_clone/result.png)

This repository contains an effective C++ implementation of 
[Pérez et, al. Poisson image editing](https://dl.acm.org/citation.cfm?id=882269).
We briefly describe the guidance for reproduction, and our implementation details.

## Get Started

This implementation is depend on [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
and [OpenCV](https://opencv.org/), (of course we did not use the OpenCV build-in
[`seamlessClone`](https://docs.opencv.org/master/df/da0/group__photo__clone.html#ga2bf426e4c93a6b1f21705513dfeca49d)
function). We assume that you are using a GNU/UNIX system:

1. Install dependence:
   ```bash
   brew install eigen opencv cmake
   ```
2. Setup [git-lfs](https://git-lfs.github.com/) before cloning this repository.
   Large source / result images in this repository are hold by `git-lfs`;
3. Compile the executable `poisson_editing`:
   ```bash
   mkdir build && cd build && cmake .. && make
   ./poisson_editing
   ```
   Then you should find the output image in `data/seamless_clone.png`.

## Visual Results

Background | Object | Mask
-----------|--------|------
![back](data/seamless_clone/Big_Tree_with_Red_Sky_in_the_Winter_Night.jpg) | ![plain](data/seamless_clone/Japan.airlines.b777-300.ja733j.arp.jpg) | ![mask](data/seamless_clone/mask.png)

Seamless cloning results: (see the title image).

## Implementation Details

The major idea of this implementation is building a (sparse) linear system based
on (4-neighbor) Laplacian matching, then effectively solving this system by Eigen.

1. Build Laplacian of unknown values `f` within mask (the Omega in paper):
   
   The general case here is: at position `i` of Omega, and relevant spatial 
   location `(y, x)` on source image, the Laplacian `L_f` of `f` is:
   ```python
   L_f[i] = 4 * f[i] - f[y + 1, x] - f[y - 1, x] - f[y, x + 1] - f[y, x - 1]
   ```

   There are three points that we should take into consideration:
   1. Build the mapping between plain index `i` and spatial index `(y, x)`. This
      is simply done by scanning though the input binary mask, then recording
      the mappings to two vector: 
      ```C++
      vector<int> spatial2omega(obj_h * obj_w, -1), omega2spatial(obj_h * obj_w, -1);
      ```
   2. Boundary conditions when `i` is on RoI edge, such that number of its 
      neighborhoods are less than 4. This is handled in `add_laplacian`:
      ```C++
      #define WITHIN(i, N) (0 <= (i) && (i) < (N)

      auto add_laplacian = [&](int y, int x) {
        if (WITHIN(y, obj_h) && WITHIN(x, obj_w)) {
          n4_count++;
          // ...
        }
      };
      ```
   3. When `i` is on edge of Omega, the `L_f` should computed from both unknown
      neighborhoods and background pixels `b` on edge. Say, `(y + 1, x)` and 
      `(y, x + 1)` are on edge:
      ```python
      L_f[i] = 4 * f[i] - b[y + 1, x] - f[y - 1, x] - b[y, x + 1] - f[y, x - 1]
      ```
2. Build Laplacian of guidance values `g` within mask (the Omega in paper):

   Basically the `g` is drawn from forge-ground image, and no boundary conditions
   should be considered:
   ```python
   L_g[i] = 4 * g[i] - g[y + 1, x] - g[y - 1, x] - g[y, x + 1] - g[y, x - 1]
   ```
3. Build a linear system `A.dot(f) = b`, `A` and `b` are inferred from the 
   Laplacian matching `L_f == F_g`. In our example the `A` is a `103183 * 103183`
   big sparse matrix. To speedup the building process we use the `setFromTriplets`
   methods provided by Eigen parse matrix.

   To solve this sparse, positive-defined linear system, we use the
   [`SimplicialLDLT`](https://eigen.tuxfamily.org/dox/classEigen_1_1SimplicialLDLT.html)
   solver, as suggested by Eigen official document.

## Performance

Our implementation is able to fuse `800 * 600` sized RoI in about 2 seconds.
We profiled our implementation: from the perspective of `main`, major time consumptions
reside in I/O:
![main prof](data/profiler/main.png)
From the perspective of major fusing function `seamless_clone`, major consumptions
resides in Eigen API `Eigen::SimplicialLDLT::solve` and `Eigen::SparseMatrix::setFromTriplets`:
![seamless prof](data/profiler/seamless_clone.png)
Thus our implementation is effective enough.

## Acknowledgment

I referred, but did not copy or rewrite from these tutorials during implementing:
1. [Seamless Cloning using OpenCV (Python , C++)](https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp/)
2. [Project: Poisson Image Editing](https://cs.brown.edu/courses/cs129/asgn/proj3_poisson/index.html)
