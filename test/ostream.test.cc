#include <tensor.hh>
#include <catch.hh>
#include <iostream>
#include <sstream>

using namespace tensor;
using namespace std;

#define TENSOR_STRINGSTREAM_CORRECT_0 \
  "[[[[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]]]"

#define TEST_CASES(CONTAINER) \
  TEST_CASE(#CONTAINER " Tensor") { \
    Tensor<int, 4, CONTAINER<int>> tensor({1, 3, 4, 2}, -1); \
    SECTION("stringstream") { \
      stringstream sstream {}; \
      sstream << tensor; \
      REQUIRE(sstream.str() == TENSOR_STRINGSTREAM_CORRECT_0); \
    } \
  } \
 \
  TEST_CASE(#CONTAINER " Scalar") { \
    Scalar<int, CONTAINER<int>> scalar {-1}; \
    SECTION("stringstream") { \
      stringstream sstream {}; \
      sstream << scalar; \
      REQUIRE(sstream.str() == "-1"); \
    } \
  }

// Instantiate Tests
TEST_CASES(data::Array);
TEST_CASES(data::HashMap);
