#include <regex>
#include "test.hh"

using namespace tensor;
using namespace std;

#define PATTERN_0 \
  R"([[[[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]]])"

#define PATTERN_1 \
  R"(-1)"

#define PATTERN_2 \
  R"(\(\+ [^)]*\))"

#define PATTERN_3 \
  R"(\(- [^)]*\))"

#define PATTERN_4 \
  R"(<\d, \d>\(\* [^)]*\))"

#define PATTERN_5 \
  R"(\(% [^)]*\))"

#define PATTERN_6 \
  R"(\(<map> [^)]*\))"

#define PATTERN_7 \
  R"(\(<reduce> [^)]*\))"

#define PATTERN_8 \
  R"(\(- [^)]*\))"

template <template <class> class Container>
void TensorTest() {
  auto tensor = Tensor<int32_t, 4, Container>({1, 3, 4, 2}, -1);
  REQUIRE(tensor.str() == PATTERN_0);
}

template <template <class> class Container>
void ScalarTest() {
  auto scalar = Scalar<int, Container>{-1}; 
  REQUIRE(scalar.str() == PATTERN_1); 
}

template <template <class> class Container> 
void BinaryAddExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  REQUIRE(regex_match((tensor_1 + tensor_2).str(), regex(PATTERN_2)));
}

template <template <class> class Container> 
void BinarySubExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  REQUIRE(regex_match((tensor_1 - tensor_2).str(), regex(PATTERN_3)));
}

template <template <class> class Container> 
void BinaryMulExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  REQUIRE(regex_match((tensor_1 * tensor_2).str(), regex(PATTERN_4)));
}

template <template <class> class Container> 
void BinaryHadExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  REQUIRE(regex_match((tensor_1 % tensor_2).str(), regex(PATTERN_5)));
}

template <template <class> class Container> 
void MapExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  auto tensor_3 = Tensor<unsigned, 3, Container>({3, 3, 3}, 0);
  auto transformer = [](float x, int y) { return x += y; };
  REQUIRE(regex_match(_map_(transformer, tensor_1, tensor_2).str(), regex(PATTERN_6)));
}

template <template <class> class Container> 
void ReduceExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  auto tensor_2 = Tensor<int, 3, Container>({3, 3, 3}, -1);
  auto tensor_3 = Tensor<unsigned, 3, Container>({3, 3, 3}, 0);
  auto accumulator = [](float x, int y) { return x += y; };
  REQUIRE(regex_match(_reduce_(accumulator, tensor_1, tensor_2).str(), regex(PATTERN_7)));
}

template <template <class> class Container> 
void UnaryNegExprTest() {
  auto tensor_1 = Tensor<float, 3, Container>({3, 3, 3}, 1);
  REQUIRE(regex_match((-tensor_1).str(), regex(PATTERN_8)));
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

TEST_CASE("BinaryHadExpr") { 
  BinaryHadExprTest<data::Array>();
  BinaryHadExprTest<data::HashMap>();
}

TEST_CASE("MapExpr") { 
  MapExprTest<data::Array>();
  MapExprTest<data::HashMap>();
}

TEST_CASE("ReduceExpr") { 
  ReduceExprTest<data::Array>();
  ReduceExprTest<data::HashMap>();
}

TEST_CASE("UnaryNegExpr") { 
  UnaryNegExprTest<data::Array>();
  UnaryNegExprTest<data::HashMap>();
}
