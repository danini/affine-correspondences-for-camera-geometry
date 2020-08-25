# Making Affine Correspondences Work in Camera Geometry Computation

The framework proposed in paper: `Barath, Daniel, et al. Making Affine Correspondences Work in Camera Geometry Computation. ECCV 2020`.
It is available at https://arxiv.org/pdf/1906.02290

# Installation C++

To build and install only the C++ implementation of the framework, clone or download this repository and then build the project by CMAKE. 
```shell
$ git clone --recursive https://github.com/danini/affine-correspondences-for-camera-geometry.git
$ cd build
$ cmake ..
$ make
```

# Install Python package and compile C++

The framework contains a Python binding and Jupyter Notebook examples. To compile with Python the following should be done.

```bash
python3 ./setup.py install
```

or

```bash
pip3 install -e .
```

# Example project

To build the sample project showing examples of fundamental matrix, homography and essential matrix fitting, set variable `CREATE_SAMPLE_PROJECT = ON` when creating the project in CMAKE. 
Then 
```shell
$ cd build
$ ./SampleProject
```

# Jupyter Notebook example

The example for homography fitting is available at: [link][example1].
 
The example for fundamental matrix fitting is available at: [link][example1].

The example for essential matrix fitting is available at: [link][example1].

[example1]: <https://github.com/danini/affine-correspondences-for-camera-geometry/blob/master/examples/example_essential_matrix.ipynb>
[example2]: <https://github.com/danini/affine-correspondences-for-camera-geometry/blob/master/examples/example_fundamental_matrix.ipynb>
[example3]: <https://github.com/danini/affine-correspondences-for-camera-geometry/blob/master/examples/example_homography_matrix.ipynb>

# Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- A modern compiler with C++17 support


# Acknowledgements

When using the algorithm, please cite `Barath, Daniel, et al. "Making Affine Correspondences Work in Camera Geometry Computation". Proceedings of the IEEE European Conference on Computer Vision. 2020`.

If you use it with Graph-Cut RANSAC, please cite `Barath, Daniel, and Matas, Jiří. "Graph-cut RANSAC." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018`.
