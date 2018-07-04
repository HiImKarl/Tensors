
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

### Running the tests

To build the tests:
```
mkdir build && cd build
cmake -G <Prefered Generator> .. 
make tests
```

To build the benchmarks, [Google's benchmark framework](https://github.com/google/benchmark) is required. You can follow the install instructions on the Github repository. Then, from the from the root directory:
```
mkdir build && cd build
cmake -G <Prefered Generator> .. 
make benchmarks
```

If you want to run a specific test/benchmark, a single benchmark or test case can be built:
```
make test_<unit>
make benchmark_<unit>
```

Each benchmark/test unit has its own compilation unit in the benchmark/test folders, so take a look there for a full lists of benchmarks/tests. Do not run make without targets because EVERYTHING will be built.

Tests are written with the [catch](https://github.com/catchorg/Catch2) framework.
The test executable is located in ./test/, relative to the build directory. After building, run the tests with
```
./test/tests # or 
./test/test_<unit>
```

Benchmarks are written with [Google benchmark](https://github.com/google/benchmark)
The benchmark executable is located in ./benchmark/, relative to the build directory. After building, run the benchmarks with
```
./benchmark/benchmarks # or 
./benchmark/benchmark_<unit>
```

## Contributing

Apart from the conventions described in the wiki, follow [Google's C++ Style Guide](https://google.github.io/styleguide/cppguide.html). 
Run all of the tests and note benchmark differences before opening pull requests.

## Libraries Used

* [Doxygen](https://github.com/doxygen/doxygen)
* [Catch](https://github.com/catchorg/Catch2)
* [Google Benchmark](https://github.com/catchorg/Catch2)
