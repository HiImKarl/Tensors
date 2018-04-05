#include <tensor.hh>
#include <catch.hh>

using namespace tensor;

TEST_CASE("Basic Tensor Arithmetic") {
  auto tensor_1 = Tensor<uint32_t, 3>({2, 3, 4});
  auto tensor_2 = Tensor<uint32_t, 3>({2, 3, 4});
  for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i) 
    for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j) 
      for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k) 
        tensor_1(i, j, k) = 100000 * i + 10000 * j + 1000 * k;

  for (uint32_t i = 1; i <= tensor_2.dimension(1); ++i) 
    for (uint32_t j = 1; j <= tensor_2.dimension(2); ++j) 
      for (uint32_t k = 1; k <= tensor_2.dimension(3); ++k) 
        tensor_2(i, j, k) = 100 * i + 10 * j + 1 * k;

  auto tensor_3 = tensor_1 + tensor_2; 
  auto tensor_4 = tensor_3 - tensor_1;

  SECTION("Two Term Addition") {
    for (uint32_t i = 1; i <= tensor_3.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_3.dimension(2); ++j) 
        for (uint32_t k = 1; k <= tensor_3.dimension(3); ++k) 
          REQUIRE(tensor_3(i, j, k) == 100100 * i + 10010 * j + 1001 * k);
  }

  SECTION("Two Term Subtraction") {
    for (uint32_t i = 1; i <= tensor_4.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_4.dimension(2); ++j) 
        for (uint32_t k = 1; k <= tensor_4.dimension(3); ++k) 
          tensor_4(i, j, k) = 100 * i + 10 * j + 1 * k;
  }
}
