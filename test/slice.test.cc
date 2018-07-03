#include "tensor.hh"
#include "catch.hh"

using namespace tensor;

TEST_CASE("Slicing Tensors", "[pod]") {
  Tensor<int32_t, 3> tensor_1({2, 4, 6});
  for (size_t i = 1; i <= tensor_1.dimension(1); ++i) 
    for (size_t j = 1; j <= tensor_1.dimension(2); ++j) 
      for (size_t k = 1; k <= tensor_1.dimension(3); ++k) 
        tensor_1(i, j, k)  = i * 100 + j * 10 + k;

  SECTION("Const correctness") {
    auto const &const_tensor = tensor_1;
    REQUIRE(!std::is_const<decltype(tensor_1.slice<1, 1>(1))>::value);
    REQUIRE(std::is_const<decltype(const_tensor.slice<1, 1>(1))>::value);
  }

  SECTION("Rank and Dimensions") {
    REQUIRE(tensor_1.slice<1>(3, 5).rank() == 1);
    REQUIRE(tensor_1.slice<1>(3, 5).dimension(1) == 2);
    REQUIRE(tensor_1.slice<1, 3>(4).dimension(1) == 2);
    REQUIRE(tensor_1.slice<1, 3>(4).dimension(2) == 6);
    REQUIRE(tensor_1.slice<1, 2, 3>().rank() == 3);
    REQUIRE(tensor_1.slice<1, 2, 3>().dimension(1) == 2);
    REQUIRE(tensor_1.slice<1, 2, 3>().dimension(2) == 4);
    REQUIRE(tensor_1.slice<1, 2, 3>().dimension(3) == 6);
  }

  SECTION("Values") {
    for (size_t j = 1; j <= tensor_1.dimension(2); ++j) 
        REQUIRE(tensor_1.slice<2>(2, 5)(j) == (int)(205 + 10 * j));
    for (size_t i = 1; i <= tensor_1.dimension(1); ++i) 
      for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
        REQUIRE(tensor_1.slice<1, 3>(4)(i, k) == (int)(i * 100 + 40 + k));
    for (size_t j = 1; j <= tensor_1.dimension(2); ++j) 
      for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
        REQUIRE(tensor_1.slice<2, 3>(1)(j, k) == (int)(100 + 10 * j + k));
    for (size_t i = 1; i <= tensor_1.dimension(1); ++i) 
      for (size_t j = 1; j <= tensor_1.dimension(2); ++j) 
        for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
          REQUIRE(tensor_1.slice<1, 2, 3>()(i, j, k) == (int)(100 * i + 10 * j + k));
  }

  SECTION("Const Reference Cast") {
    auto const &const_tensor = tensor_1;
    for (size_t j = 1; j <= const_tensor.dimension(2); ++j) 
        REQUIRE(const_tensor.slice<2>(2, 5)(j) == (int)(205 + 10 * j));
    for (size_t i = 1; i <= const_tensor.dimension(1); ++i) 
      for (size_t k = 1; k <= const_tensor.dimension(3); ++k)
        REQUIRE(const_tensor.slice<1, 3>(4)(i, k) == (int)(i * 100 + 40 + k));
    for (size_t j = 1; j <= const_tensor.dimension(2); ++j) 
      for (size_t k = 1; k <= const_tensor.dimension(3); ++k)
        REQUIRE(const_tensor.slice<2, 3>(1)(j, k) == (int)(100 + 10 * j + k));
    for (size_t i = 1; i <= const_tensor.dimension(1); ++i) 
      for (size_t j = 1; j <= const_tensor.dimension(2); ++j) 
        for (size_t k = 1; k <= const_tensor.dimension(3); ++k)
          REQUIRE(const_tensor.slice<1, 2, 3>()(i, j, k) == (int)(100 * i + 10 * j + k));
  }
}

