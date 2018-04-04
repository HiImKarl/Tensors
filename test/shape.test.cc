#include <tensor.hh>
#include "catch.hh"

using namespace tensor;

TEST_CASE("Shapes") {
  
  /*     -- Shape Initialization --     */
  auto shape = Shape<8>({1, 2, 3, 4, 5, 6, 7, 8});
  
  SECTION("Rank and Dimensions") {
    REQUIRE(shape.rank() == 8);
    for (uint32_t i = 1; i <= 8; ++i)
      REQUIRE(shape.dimension(i) == i);
  }

  SECTION("Tensor Constructor") {
    Tensor<int, 8> tensor(shape);
    REQUIRE(tensor.rank() == 8);
    for (uint32_t i = 1; i <= 8; ++i)
      REQUIRE(tensor.dimension(i) == i);
  }

}
