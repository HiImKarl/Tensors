#include <tensor.hh>
#include <catch.hh>

using namespace tensor;

TEST_CASE("Logic Errors") {
  Tensor<int, 5> tensor({1, 2, 3, 4, 5});
  SECTION("Requesting Out of Bounds Dimension") {
    REQUIRE_THROWS_AS(tensor.dimension(0), std::logic_error);
    REQUIRE_THROWS_AS(tensor.dimension(6), std::logic_error);
  }

  SECTION("Accessing Out of Bounds Dimension") {
    REQUIRE_THROWS_AS(tensor(6), std::logic_error);
    REQUIRE_THROWS_AS(tensor(2, 2, 3, 4, 5), std::logic_error);
  }

  SECTION("Addition Subtraction Shape Mismatch") {
    Tensor<int, 5> tensor_2({1, 2, 3, 4, 6});
    REQUIRE_THROWS_AS(tensor_2 = tensor + tensor_2, std::logic_error);
  }

  SECTION("Multiplication Shape Mismatch") {
    Tensor<int32_t, 3> tensor_1({1, 2, 3});
    Tensor<int32_t, 2> tensor_2({4, 2});
    Tensor<int32_t, 3> tensor_3({1, 2, 3});
    REQUIRE_THROWS_AS(tensor_3 = tensor_1 * tensor_2, std::logic_error);
  }

  SECTION("Fill") {
    Tensor<int32_t, 3> tensor_1({1, 2, 3});
    std::vector<int32_t> vec_incorrect(7, 0);
    std::vector<int32_t> vec_correct(6, 0);
    REQUIRE_NOTHROW(Fill(tensor_1, vec_correct));
    REQUIRE_THROWS_AS(Fill(tensor_1, vec_incorrect), std::logic_error);
  }

  SECTION("Iterator") {
    REQUIRE_THROWS_AS(tensor.begin(0), std::logic_error);
    REQUIRE_THROWS_AS(tensor.begin(6), std::logic_error);
    REQUIRE_NOTHROW(tensor.begin(5));
  }
}
