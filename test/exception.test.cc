#include <tensor.hh>
#include <catch.hh>

using namespace tensor;

TEST_CASE("Logic Errors") {
  Tensor<int, 5> tensor({1, 2, 3, 4, 5});
  SECTION("Requesting Out of Bounds Dimension") {
    try { 
      tensor.dimension(0); 
      REQUIRE(0);
    } catch (const std::logic_error &e) { REQUIRE(1); }

    try { 
      tensor.dimension(6); 
      REQUIRE(0);
    } catch (const std::logic_error &e) { REQUIRE(1); }
  }

  SECTION("Accessing Out of Bounds Dimension") {
    try {
      auto tmp = tensor(6);
      REQUIRE(0);
    } catch (const std::logic_error &e) { REQUIRE(1); }

    try { 
      tensor(2, 2, 3, 4, 5); 
      REQUIRE(0);
    } catch (const std::logic_error &e) { REQUIRE(1); }
  }

  SECTION("Addition Subtraction Shape Mismatch") {
    Tensor<int, 5> tensor_2({1, 2, 3, 4, 6});
    try {
      tensor_2 = tensor + tensor_2;
      REQUIRE(0);
    } catch (const std::logic_error &e) { REQUIRE(1); }
  }

  SECTION("Multiplication Shape Mismatch") {
    Tensor<int32_t, 3> tensor_1({1, 2, 3});
    Tensor<int32_t, 2> tensor_2({4, 2});
    try {
      Tensor<int32_t, 3> tensor_3 = tensor_1 * tensor_2;
      REQUIRE(0);
    } catch (const std::logic_error &e) { REQUIRE(1); }
  }
}
