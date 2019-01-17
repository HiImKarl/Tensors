#include <regex>
#include <sstream>
#include "test.hh"

using namespace tensor;
using namespace std;

#define PATTERN_0 \
  R"([[[[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]]])"

#define PATTERN_1 \
  R"(-1)"

#define PATTERN_2 \
  R"(BinaryAddExpr\([^)]*\))"

#define PATTERN_3 \
  R"(BinarySubExpr\([^)]*\))"

#define PATTERN_4 \
  R"(BinaryMulExpr<\d, \d>\([^)]*\))"

template <template <class> class Container>
void TensorTest() {
  auto tensor = Tensor<int, 4, Container>({1, 3, 4, 2}, -1); 
  auto sstream = stringstream{}; 
  sstream << tensor; 
  REQUIRE(sstream.str() == PATTERN_0); 
} 

template <template <class> class Container>
void ScalarTest() {
  auto scalar = Scalar<int, Container>{-1}; 
  auto sstream = stringstream{}; 
  sstream << scalar;
  REQUIRE(sstream.str() == PATTERN_1); 
}

template <template <class> class Container> 
void BinaryAddExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  auto sstream = stringstream{}; 
  sstream << tensor_1 + tensor_2;
  REQUIRE(regex_match(sstream.str(), regex(PATTERN_2)));
}

template <template <class> class Container> 
void BinarySubExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  auto sstream = stringstream{}; 
  sstream << tensor_1 - tensor_2;
  REQUIRE(regex_match(sstream.str(), regex(PATTERN_3)));
}

template <template <class> class Container> 
void BinaryMulExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  auto sstream = stringstream{}; 
  sstream << tensor_1 * tensor_2;
  REQUIRE(regex_match(sstream.str(), regex(PATTERN_4)));
}

// Instantiate Tests

TEST_CASE("Tensor") { 
  TensorTest<data::Array>();
  TensorTest<data::HashMap>();
}

TEST_CASE("Scalar") { 
  ScalarTest<data::Array>();
  ScalarTest<data::HashMap>();
}

TEST_CASE("BinaryAddExpr") { 
  BinaryAddExprTest<data::Array>();
  BinaryAddExprTest<data::HashMap>();
}

TEST_CASE("BinarySubExpr") { 
  BinarySubExprTest<data::Array>();
  BinarySubExprTest<data::HashMap>();
}

TEST_CASE("BinaryMulExpr") { 
  BinaryMulExprTest<data::Array>();
  BinaryMulExprTest<data::HashMap>();
}
