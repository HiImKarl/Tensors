
# Tensor

C++11 Tensor Template header-only library, with compile time rank. This library makes heavy use of template expressions and meta-programming to optimize Tensor mathematics and provide simple syntax. 

## Getting Started


### Prerequisites
C++11 is the minimum C++ standard required to compile the library. Below is a list of compilers that has successfully ran the unit tests:

The following compilers have succesfully built and ran the tests:
* gcc 6.3+


### Installation

The entire library is contained in a single header file: include/tensor.hh. The only requirement is to clone the repo (or download from the browser): 
```
git clone git@github.com:HiImKarl/Tensors.git
```
And then copy include/tensor.hh into your project. 

To build the tests and benchmarks, [Google's benchmark framework](https://github.com/google/benchmark) is required. You can follow the install instructions in the Github link. 

If you have all of the prerequisites, from the from the root directory:
```
mkdir build && cd build
cmake -G <Prefered Generator> ..
make
```

### Running the tests

Tests are developed with the [catch](https://github.com/catchorg/Catch2) framework.
The test executable is named tests, located in ./test/, relative to the build directory.

### Running the benchmarks

The benchmark executable is named benchmarks, located in ./benchmark/, relative to the build directory.
