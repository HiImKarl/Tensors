
# Tensor

## Description
This is a multi-dimensional array library, where array rank must be known at compile time. The entire library is contained in a single header file. This library makes heavy use of template expressions and meta-programming to optimize Tensor mathematics and provide simple syntax. 

## Table of Contents
   * [Tensor](#tensor)
      * [Getting Started](#getting-started)
         * [Prerequisites](#prerequisites)
         * [Installation](#installation)
         * [Running the tests](#running-the-tests)
      * [Contributing](#contributing)
      * [Libraries Used](#libraries-used)

## Getting Started

### Prerequisites
C++11 is the minimum C++ standard required to compile the library. Below is a list of compilers that has successfully ran the unit tests:

The following compilers have succesfully built and ran the tests:
* gcc 6.3+
* clang 3.5+

### Installation

The entire library is contained in a single header file: include/tensor.hh. The only requirement is to clone the repo (or download from the browser): 
```
git clone git@github.com:HiImKarl/Tensors.git
```
And then copy include/tensor.hh into your project. 

### Usage

#### Creating Tensors

Creating a Tensor is simple:
```C++
// This will create a 5x3x5 array of integers. The elements are **not** zero initialized.
Tensor<int, 3> t0({5, 3, 5});

// This will create a 5x3x5 array of integers, initialized to the value -1.
Tensor<int, 3> t1({5, 3, 5}, -1);

// This will create a 5x3x5 array of integers, with each value initialized by int rand() (from C stdlib)
// Note: this will call rand() 65 times.
Tensor<int, 3> t2(5, 3, 5}, &rand);

// Tensors can be filled with the elements of other containers
vector<int> ones(65, 1); // 65 element vector of ones
Tensor<int, 3> t3({5, 3, 5});
Fill(t3, ones.begin(), ones.end()); // t3 now contains ones

// Copy constructor allocates and copies over elements
Tensor<int, 3> t4 = t3; // t4 is a 5x3x5 tensor of ones

// Move constructor takes ownership of the underlying data
Tensor<int, 3> t5 = move(t4); // t5 is 5x3x5 tensor of ones, t4 is destroyed

// "Reference constructor" (not a typical C++ idiom) shares 
// ownership of the data
Tensor<int, 3> t6 = t5.ref() // t6 shares the same 5x3x5 data array as t4

// Any changes made to t6 will also be made to t5
Fill(t6, 0); // t5 is now also filled with zeros
```

#### Indexing Tensors

Tensors can be accessed with `operator()` or the `at` method. It is not necessary to
access the tensor to the individual element; partial access will return a tensor with
the slice of the original tensor with the same underlying data (effectively a reference
to that tensor slice).
**Tensors are indexed beginning at 1!** This is true for all tensor methods.
```C++
// Create a 2x3x4x5 or ensor of boolean false
Tensor<bool, 4> t0({2, 3, 4, 5}, false); 
// t1, a 4x5 tensor of booleans (false) is the (2, 2) slice of t0. 
Tensor<bool, 2> t1 = t0(2, 2); 
// t1 is also the (2, 2) slice of t0, and is effectively the same as t1.
Tensor<bool, 2> t2 = t0.at(2, 2);
```

The only difference between `operator()` and `at` is that if a Scalar value is returned,
`operator()` returns a reference to the value itself, while `at` returns a 0-rank tensor,
whose only element is the value. Since references are maintained by Tensors, using `at` 
is less efficient that `operator()` (if you don't the reference).

```C++
// returns a bool&, the (2, 2, 2, 2) slice of t0
t0(2, 2, 2, 2) 
// returns a 0-rank tenor, whose underlying value is the (2, 2, 2, 2) slice of t0
t0.at(2, 2, 2, 2) 

```


#### Tensor mathematics

``` C++

```

### Documentation

Building the documentation requires [Doxygen](https://github.com/doxygen/doxygen) installed. Simplying running Doxygen will create the html/markdowm documentation (which will be dumped in a folder called `Documentation`):

```
# in the root folder of the project
doxygen
```

### Running the tests/benchmarks

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
The benchmark executable is located in ./benchmark/, relative to the build directory. 
To get an accurate benchmark, you may need to disable certain cpu power/performance features.
You can run the benchmarks with:
```
sudo cpupower frequency-set --governor performance # disable power save for linux systems
./benchmark/benchmarks # or 
./benchmark/benchmark_<unit>
sudo cpupower frequency-set --governor powersave   # back to default mode
```

## Contributing

Apart from the conventions described in the wiki, follow [Google's C++ Style Guide](https://google.github.io/styleguide/cppguide.html). 
Run all of the tests and note benchmark differences before opening pull requests.

## Libraries Used

* [Doxygen](https://github.com/doxygen/doxygen)
* [Catch](https://github.com/catchorg/Catch2)
* [Google Benchmark](https://github.com/catchorg/Catch2)
