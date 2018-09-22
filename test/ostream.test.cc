#include <tensor.hh>
#include <catch.hh>
#include <iostream>
#include <sstream>
#include "test.hh"

using namespace tensor;
using namespace std;

#define TENSOR_STRINGSTREAM_CORRECT_0 \
  "[[[[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]]]"

template <template <class> class Container>
void TensorTests() {
  Tensor<int, 4, Container> tensor({1, 3, 4, 2}, -1); 
  SECTION("stringstream") { 
    stringstream sstream {}; 
    sstream << tensor; 
    REQUIRE(sstream.str() == TENSOR_STRINGSTREAM_CORRECT_0); 
  } 
} 

template <template <class> class Container>
void ScalarTests() {
  Scalar<int, Container> scalar {-1}; 
  SECTION("stringstream") { 
    stringstream sstream {}; 
    sstream << scalar; 
    REQUIRE(sstream.str() == "-1"); 
  } 
}

// Instantiate Tests

TEST_CASE("Tensor") { 
  TensorTests<data::Array>();
  TensorTests<data::HashMap>();
}

TEST_CASE("Scalar") { 
  ScalarTests<data::Array>();
  ScalarTests<data::HashMap>();
}
