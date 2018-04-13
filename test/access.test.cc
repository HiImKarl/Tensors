#include <tensor.hh>
#include <catch.hh>
#include <array>

using namespace tensor;

TEST_CASE("Tensor Access", "[int]") {

  /*     -- Tensor Initialization --   */
  auto tensor_1 = Tensor<int32_t, 4>({1, 2, 3, 4});

  // Initialize values
  for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i)
    for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j)
      for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k)
        for (uint32_t l = 1; l <= tensor_1.dimension(4); ++l) 
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;

  /*     --------------------------   */

  SECTION("Using Indices") {
    Indices<4> indices({1, 1, 1, 1});
    REQUIRE(tensor_1[indices] == 1111);
  }
}
