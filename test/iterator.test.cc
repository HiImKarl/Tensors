#include <tensor.hh>
#include <catch.hh>

using namespace tensor;

TEST_CASE("Forwards Iterator") {
  Tensor<int32_t, 3> tensor({2, 3, 4});
  for (uint32_t i = 1; i <= tensor.dimension(1); ++i)
    for (uint32_t j = 1; j <= tensor.dimension(2); ++j)
      for (uint32_t k = 1; k <= tensor.dimension(3); ++k)
        tensor(i, j, k) = 100 * i + 10 * j + k;

  SECTION("Tensor") {
    auto begin = tensor.begin(1);
    auto end = tensor.end(1);
    REQUIRE((*begin).rank() == 2);
    REQUIRE(begin->rank() == 2);
    REQUIRE(begin->shape()[1] == 3);
    REQUIRE(begin->shape()[2] == 4);
    REQUIRE(end->rank() == 2);
    REQUIRE(end->shape()[1] == 3);
    REQUIRE(end->shape()[2] == 4);

    int32_t i = 0;
    for (auto it = begin; it != end; ++it) {
      ++i;
      REQUIRE((*it)(1, 1) == 100 * i + 11);
    }
    REQUIRE(i == 2);
  }

  SECTION("Scalar") {
    auto begin = tensor(1, 3).begin();
    auto end = tensor(1, 3).end();
    REQUIRE(begin->rank() == 0);
    REQUIRE(end->rank() == 0);
    int32_t i = 0;
    for (auto it = begin; it != end; ++it) {
      ++i;
      REQUIRE((*it)() == i + 130);
    }
    REQUIRE(i == 4);
  }
}
