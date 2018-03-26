#include "tensor.hh"
#include "catch.hh"

using namespace tensor;
using namespace std;

TEST_CASE("Intializing Tensors", "[int]") {

  /*     -- Tensor Initialization --   */
  auto tensor_1 = Tensor<int32_t, 4>({1, 2, 3, 4});

  // Initialize values
  for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i)
    for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j)
      for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k)
        for (uint32_t l = 1; l <= tensor_1.dimension(4); ++l)
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;

  /*     --------------------------   */

  SECTION("Rank and Dimensions") {
    REQUIRE(tensor_1.rank() == 4);
    REQUIRE(tensor_1(1).rank() == 3);
    REQUIRE(tensor_1(1, 1).rank() == 2);
    REQUIRE(tensor_1(1, 1, 1).rank() == 1);
    REQUIRE(tensor_1(1, 1, 1, 1).rank() == 0);
    for (int i = 1; i <= 4; ++i) REQUIRE(tensor_1.dimension(i) == i);
  }

  SECTION("Initializing Values") {
    for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i)
      for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j)
        for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k)
          for (uint32_t l = 1; l <= tensor_1.dimension(4); ++l)
            REQUIRE(tensor_1(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
  }

  SECTION("Const Reference Cast") {
    auto const& c_tensor = tensor_1;
    for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i)
      for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j)
        for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k)
          for (uint32_t l = 1; l <= tensor_1.dimension(4); ++l)
            REQUIRE(c_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
  }

  SECTION("Copy Constructor") {
    auto copy_tensor = tensor_1;
    for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i)
      for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j)
        for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k)
          for (uint32_t l = 1; l <= tensor_1.dimension(4); ++l)
            REQUIRE(copy_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
  }

  SECTION("Move Constructor") {
    auto move_tensor = std::move(tensor_1);
    for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i)
      for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j)
        for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k)
          for (uint32_t l = 1; l <= tensor_1.dimension(4); ++l)
            REQUIRE(move_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
    REQUIRE(!tensor_1.is_owner());
  }

}

TEST_CASE("Initializing Scalars") {

  /*     -- Tensor Initialization --   */

  Tensor<int32_t> scalar_1{};       // Initialize a scalar tensor;
  Tensor<int32_t> scalar_2(0);      // Initialize and assign in the same expression

  /*     --------------------------   */

  SECTION("Rank and Dimensions") {
    REQUIRE(scalar_1.rank() == 0);
    REQUIRE(scalar_2.rank() == 0);
  }

  SECTION("Initializing Values") {
    scalar_1 = 0;
    REQUIRE(scalar_1 == 0);
    REQUIRE(scalar_2 == 0);
  }

  SECTION("Const Reference Cast") {
    scalar_1 = 0;
    auto const& c_scalar_1 = scalar_1;
    auto const& c_scalar_2 = scalar_2;
    REQUIRE(c_scalar_1 == 0);
    REQUIRE(c_scalar_2 == 0);
  }

}

