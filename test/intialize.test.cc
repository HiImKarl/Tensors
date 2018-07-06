#include <catch.hh>
#include <tensor.hh>
#include <deque>

using namespace tensor;

TEST_CASE("Intializing Tensors", "[int]") {

  /*     -- Tensor Initialization --   */
  auto tensor_1 = Tensor<int32_t, 4>({1, 2, 3, 4});

  // Initialize values
  for (size_t i = 1; i <= tensor_1.dimension(1); ++i)
    for (size_t j = 1; j <= tensor_1.dimension(2); ++j)
      for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
        for (size_t l = 1; l <= tensor_1.dimension(4); ++l)
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;

  /*     --------------------------   */

  SECTION("Const correctness") {
    const auto& const_tensor = tensor_1;
    REQUIRE(!std::is_const<decltype(tensor_1(1, 2, 3))>::value);
    REQUIRE(std::is_const<decltype(const_tensor(1, 2, 3))>::value);
  }

  SECTION("Rank and Dimensions") {
    REQUIRE(tensor_1.rank() == 4);
    REQUIRE(tensor_1.at(1).rank() == 3);
    REQUIRE(tensor_1.at(1, 1).rank() == 2);
    REQUIRE(tensor_1.at(1, 1, 1).rank() == 1);
    REQUIRE(tensor_1.at(1, 1, 1, 1).rank() == 0);
    for (int i = 1; i <= 4; ++i) REQUIRE(tensor_1.dimension(i) == i);
    for (int i = 1; i <= 3; ++i) REQUIRE(tensor_1(1).dimension(i) == i + 1);
  }

  SECTION("Initializing Values") {
    for (size_t i = 1; i <= tensor_1.dimension(1); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(2); ++j)
        for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
          for (size_t l = 1; l <= tensor_1.dimension(4); ++l)
            REQUIRE(tensor_1(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
  }

  SECTION("Const Reference Cast") {
    auto const& c_tensor = tensor_1;
    for (size_t i = 1; i <= tensor_1.dimension(1); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(2); ++j)
        for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
          for (size_t l = 1; l <= tensor_1.dimension(4); ++l)
            REQUIRE(c_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
  }

  SECTION("Copy Constructor") {
    auto copy_tensor = tensor_1;
    for (size_t i = 1; i <= tensor_1.dimension(1); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(2); ++j)
        for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
          for (size_t l = 1; l <= tensor_1.dimension(4); ++l)
            REQUIRE(copy_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
  }

  SECTION("Move Constructor") {
    auto move_tensor = std::move(tensor_1);
    for (size_t i = 1; i <= tensor_1.dimension(1); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(2); ++j)
        for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
          for (size_t l = 1; l <= tensor_1.dimension(4); ++l)
            REQUIRE(move_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
 }

 SECTION("Value Constructor") {
    auto ones = Tensor<int32_t, 3>({2, 3, 4}, 1);
    for (size_t i = 1; i <= ones.dimension(1); ++i)
      for (size_t j = 1; j <= ones.dimension(2); ++j)
        for (size_t k = 1; k <= ones.dimension(3); ++k)
          REQUIRE(ones(i, j, k) == 1);
    auto twos = Tensor<int32_t, 3>({2, 3, 4}, 2.3f);
    for (size_t i = 1; i <= twos.dimension(1); ++i)
      for (size_t j = 1; j <= twos.dimension(2); ++j)
        for (size_t k = 1; k <= twos.dimension(3); ++k)
          REQUIRE(twos(i, j, k) == 2);
 }

 SECTION("Fill Method") {
    auto naturals = Tensor<int32_t, 3>({2, 3, 4});
    std::deque<int32_t> container{};
    for (int i = 0; i < 24; ++i) container.push_back(i);
    Fill(naturals, container.begin(), container.end());
    for (size_t i = 1; i <= naturals.dimension(1); ++i)
      for (size_t j = 1; j <= naturals.dimension(2); ++j)
        for (size_t k = 1; k <= naturals.dimension(3); ++k)
          REQUIRE((size_t)naturals(i, j, k) == (i - 1) * 12 + (j - 1) * 4 + k - 1);

    Fill(naturals, 1);
    for (size_t i = 1; i <= naturals.dimension(1); ++i)
      for (size_t j = 1; j <= naturals.dimension(2); ++j)
        for (size_t k = 1; k <= naturals.dimension(3); ++k)
          REQUIRE((size_t)naturals(i, j, k) == 1);
 }

 SECTION("Factory Method") {
    int count = 0;
    std::function<int(int)> factory = [&count](int x) -> int {
      count += x;
      return count - x;
    };

    auto naturals = Tensor<int32_t, 3>(Shape<3>({2, 3, 4}), factory, 1);
    for (size_t i = 1; i <= naturals.dimension(1); ++i)
      for (size_t j = 1; j <= naturals.dimension(2); ++j)
        for (size_t k = 1; k <= naturals.dimension(3); ++k)
          REQUIRE((size_t)naturals(i, j, k) == (i - 1) * 12 + (j - 1) * 4 + k - 1);
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

