#include <tensor.hh>
#include <catch.hh>
#include <iostream>
#include <sstream>

using namespace tensor;
using namespace std;

#define TENSOR_STRINGSTREAM_CORRECT_0 \
  "[[[[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]]]"

TEST_CASE("Tensor") {
  Tensor<int, 4> tensor({1, 3, 4, 2}, -1);
  SECTION("stringstream") {
    stringstream sstream {};
    sstream << tensor;
    REQUIRE(sstream.str() == TENSOR_STRINGSTREAM_CORRECT_0);
  }
}

TEST_CASE("Scalar") {
  Scalar<int> scalar {-1};
  SECTION("stringstream") {
    stringstream sstream {};
    sstream << scalar;
    REQUIRE(sstream.str() == "-1");
  }
}
