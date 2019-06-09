# Poisson Image Editing

This repository is an effective C++ implementation of 
[Pérez et al. Poisson image editing](https://dl.acm.org/citation.cfm?id=882269)
and [Sun et al. Poisson matting](https://dl.acm.org/citation.cfm?id=1015721).
We briefly describe our design of reproduction and implementation details.

## Get Started

Dependencies of this project include:
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [OpenCV](https://opencv.org/)
- [gflags](https://github.com/gflags/gflags)
- [Boost](https://www.boost.org/)
- (*optional*) [Intel MKL](https://software.intel.com/en-us/mkl) for accelerating
  sparse system solving.

1. Install dependence, on macOS it is recommended to using `homebrew`:
   ```bash
   brew install cmake eigen opencv gflags boost
   ```
   
   For intel MKL, go ahead to their [official website](https://software.intel.com/en-us/mkl)
   to apply a community license;
2. Setup [git-lfs](https://git-lfs.github.com/) before cloning this repository.
   Images in this repository are quite large thus all hold by `git-lfs`;
3. Compile the executable `visualize`:
   ```bash
   mkdir build && cd build
   cmake -DINTEL_ROOT=<path/to/intel_performance_library> ..
   make
   ./visualize <args>
   ```

Arguments of `visualize` executable include:
```
visualize: C++ implementation of Poisson matting and clone.

  Flags from /Users/lirundong/Projects/poisson_editing/tools/visualize.cpp:
    -cfg (argument of specified task, separated by commas) type: string
      default: ""
    -dst (path to output image) type: string default: ""
    -mask (path to fore/back/unknown map image) type: string default: ""
    -src (path to source image) type: string default: ""
    -src_back (path to source background image) type: string default: ""
    -src_fore (path to source foreground image) type: string default: ""
    -task (task to perform: {clone, matting}) type: string default: "clone"
```
where configurations (the `-cfg` argument) are passed by a string of numbers, 
separated by commas or spaces. For each of the tasks, the configurations are:
- For Poisson cloning: `-cfg="<object_x1>, <object_y1>, <object_long_edge_size>"`;
- For Poisson matting: `-cfg="<trimap_value_for_foreground>, <for_background>,
  <for_unknown_region>, <max_number_of_solving_iterations>"`;

## Visual Results

### Seamless Poisson Cloning

Background | Object | Mask
-----------|--------|------
![back](data/seamless_clone/Big_Tree_with_Red_Sky_in_the_Winter_Night.jpg) | ![plain](data/seamless_clone/Japan.airlines.b777-300.ja733j.arp.jpg) | ![mask](data/seamless_clone/mask.png)

Result:
![seamless_clone](data/seamless_clone/result.png)

### Poisson Matting

Source Image | Trimap
-------------|--------
![src](data/matting/src.png) | ![trimap](data/matting/trimap.png)

Result after 10 iterations:
![alpha](data/matting/result.png)

## Implementation Details

The core of this implementation is building a (sparse) linear system from
(4-neighbor) Laplacian matching, and specifying border conditions by specific
tasks. 

For Poisson cloning, the border condition is *border of unknown region is
consistent with background image*; for Poisson matting, it's *exterior border
between unknown region and absolute foreground region have alpha value of 1, for
background side the alpha value is 0*.

1. To build Laplacian of unknown values `f` within mask (the Omega in paper):
   
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
2. To build Laplacian of guidance values `g` within mask (the Omega in paper):

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

This implementation is able to fuse `800 * 600` sized RoI in about 2 seconds.
We profiled this implementation: the major time consumptions reside in I/O:
![main prof](data/profiler/main.png)
From the perspective of fusing function `seamless_clone`, major consumptions
resides in Eigen API `Eigen::SimplicialLDLT::solve` and `Eigen::SparseMatrix::setFromTriplets`:
![seamless prof](data/profiler/seamless_clone.png)

Thus (the core logic of) this implementation is effective enough. Linking against
Intel MKL will further accelerate the solving process.

## License

This work is licensed under Apache License 2.0. See [LICENSE](LICENSE) for details.

## References

1. [Seamless Cloning using OpenCV (Python , C++)](https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp/)
2. [Project: Poisson Image Editing](https://cs.brown.edu/courses/cs129/asgn/proj3_poisson/index.html)
3. [MarcoForte/poisson-matting](https://github.com/MarcoForte/poisson-matting)
