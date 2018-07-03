#include <tensor.hh>
#include "catch.hh"

using namespace tensor;

TEST_CASE("Shapes") {
  
  //     -- Shape Initialization --     
  auto shape = Shape<8>({1, 2, 3, 4, 5, 6, 7, 8});
  auto scalar_shape = Shape<0>();
  
  SECTION("Rank and Dimensions") {
    REQUIRE(shape.rank() == 8);
    for (size_t i = 1; i <= 8; ++i)
      REQUIRE(shape[i] == i);
  }

  SECTION("Tensor Constructor") {
    Tensor<int, 8> tensor(shape);
    REQUIRE(tensor.rank() == 8);
    for (size_t i = 1; i <= 8; ++i)
      REQUIRE(tensor.dimension(i) == i);
  }

  SECTION("Scalar Constructor") {
    auto tensor = Tensor<int>(scalar_shape);
    REQUIRE(tensor.rank() == 0);
    REQUIRE(tensor() == 0);
  }

}
