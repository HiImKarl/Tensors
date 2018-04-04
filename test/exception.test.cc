#include <tensor.hh>
#include <catch.hh>

using namespace tensor;

TEST_CASE("Logic Errors") {
  Tensor<int, 5> tensor({1, 2, 3, 4, 5});
  SECTION("Requesting Out of Bounds Dimension") {
    try { tensor.dimension(0); } 
    catch (const std::logic_error &e) { REQUIRE(1); }

    try { tensor.dimension(6); } 
    catch (const std::logic_error &e) { REQUIRE(1); }
  }

  SECTION("Accessing Out of Bounds Dimension") {
    try { tensor(2); }
    catch (const std::logic_error &e) { REQUIRE(1); }

    try { tensor(1, 2, 3, 4, 0); }
    catch (const std::logic_error &e) { REQUIRE(1); }
  }
}
