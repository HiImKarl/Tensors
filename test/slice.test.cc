#include "tensor.hh"
#include "catch.hh"

using namespace tensor;

#define TEST_CASES(CONTAINER) \
  TEST_CASE(#CONTAINER " Slicing Tensors", "[pod]") { \
    Tensor<int32_t, 3, CONTAINER<int32_t>> tensor_1({2, 4, 6}); \
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) \
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) \
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) \
          tensor_1(i, j, k)  = i * 100 + j * 10 + k; \
 \
    SECTION("Const correctness") { \
      auto const &const_tensor = tensor_1; \
      REQUIRE(!std::is_const<decltype(tensor_1.slice<0, 0>(0))>::value); \
      REQUIRE(std::is_const<decltype(const_tensor.slice<0, 0>(0))>::value); \
    } \
 \
    SECTION("Rank and Dimensions") { \
      REQUIRE(tensor_1.slice<0>(2, 4).rank() == 1); \
      REQUIRE(tensor_1.slice<0>(2, 4).dimension(0) == 2); \
      REQUIRE(tensor_1.slice<0, 2>(3).dimension(0) == 2); \
      REQUIRE(tensor_1.slice<0, 2>(3).dimension(1) == 6); \
      REQUIRE(tensor_1.slice<0, 1, 2>().rank() == 3); \
      REQUIRE(tensor_1.slice<0, 1, 2>().dimension(0) == 2); \
      REQUIRE(tensor_1.slice<0, 1, 2>().dimension(1) == 4); \
      REQUIRE(tensor_1.slice<0, 1, 2>().dimension(2) == 6); \
    } \
 \
    SECTION("Values") { \
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) \
          REQUIRE(tensor_1.slice<1>(1, 4)(j) == (int)(104 + 10 * j)); \
      for (size_t i = 0; i < tensor_1.dimension(0); ++i) \
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) \
          REQUIRE(tensor_1.slice<0, 2>(3)(i, k) == (int)(i * 100 + 30 + k)); \
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) \
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) \
          REQUIRE(tensor_1.slice<1, 2>(1)(j, k) == (int)(100 + 10 * j + k)); \
      for (size_t i = 0; i < tensor_1.dimension(0); ++i) \
        for (size_t j = 0; j < tensor_1.dimension(1); ++j) \
          for (size_t k = 0; k < tensor_1.dimension(2); ++k) \
            REQUIRE(tensor_1.slice<0, 1, 2>()(i, j, k) == (int)(100 * i + 10 * j + k)); \
    } \
 \
    SECTION("Const Reference Cast") { \
      auto const &const_tensor = tensor_1; \
      for (size_t j = 0; j < const_tensor.dimension(1); ++j) \
          REQUIRE(const_tensor.slice<1>(1, 4)(j) == (int)(104 + 10 * j)); \
      for (size_t i = 0; i < const_tensor.dimension(0); ++i) \
        for (size_t k = 0; k < const_tensor.dimension(2); ++k) \
          REQUIRE(const_tensor.slice<0, 2>(3)(i, k) == (int)(i * 100 + 30 + k)); \
      for (size_t j = 0; j < const_tensor.dimension(1); ++j) \
        for (size_t k = 0; k < const_tensor.dimension(2); ++k) \
          REQUIRE(const_tensor.slice<1, 2>(0)(j, k) == (int)(10 * j + k)); \
      for (size_t i = 0; i < const_tensor.dimension(0); ++i) \
        for (size_t j = 0; j < const_tensor.dimension(1); ++j) \
          for (size_t k = 0; k < const_tensor.dimension(2); ++k) \
            REQUIRE(const_tensor.slice<0, 1, 2>()(i, j, k) == (int)(100 * i + 10 * j + k)); \
    } \
  } 

// Instantiate Tests
TEST_CASES(data::Array);
TEST_CASES(data::HashMap);


