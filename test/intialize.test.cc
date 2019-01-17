#include <deque>
#include "test.hh"

using namespace tensor;

template <template <class> class Container>
void InitializingTensorTests() {
  auto tensor_1 = Tensor<int32_t, 4, Container>{1, 2, 3, 4}; 
 
  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l; 
 
  SECTION("Const correctness") { 
    const auto& const_tensor = tensor_1; 
    REQUIRE(!std::is_const<decltype(tensor_1(0, 1, 2))>::value); 
    REQUIRE(std::is_const<decltype(const_tensor(0, 1, 2))>::value); 
  } 
 
  SECTION("Rank and Dimensions") { 
    REQUIRE(tensor_1.rank() == 4); 
    REQUIRE(tensor_1.at(0).rank() == 3); 
    REQUIRE(tensor_1.at(0, 0).rank() == 2); 
    REQUIRE(tensor_1.at(0, 0, 0).rank() == 1); 
    REQUIRE(tensor_1.at(0, 0, 0, 0).rank() == 0); 
    for (size_t i = 0; i < 4; ++i) REQUIRE(tensor_1.dimension(i) == i + 1); 
    for (size_t i = 0; i < 3; ++i) REQUIRE(tensor_1(0).dimension(i) == i + 2); 
  } 
 
  SECTION("Initializing Values") { 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(tensor_1(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l)); 
  } 
 
  SECTION("Const Reference Cast") { 
    auto const& c_tensor = tensor_1; 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(c_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l)); 
  } 
 
  SECTION("Copy Constructor") { 
    auto copy_tensor = tensor_1; 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(copy_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l)); 
  } 
 
  SECTION("Move Constructor") { 
    auto move_tensor = std::move(tensor_1); 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(move_tensor(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l)); 
 } 
 
 SECTION("Value Constructor") { 
    auto ones = Tensor<int32_t, 3, Container>({2, 3, 4}, 1); 
    for (size_t i = 0; i < ones.dimension(0); ++i) 
      for (size_t j = 0; j < ones.dimension(1); ++j) 
        for (size_t k = 0; k < ones.dimension(2); ++k) 
          REQUIRE(ones(i, j, k) == 1); 
    auto twos = Tensor<int32_t, 3, Container>({2, 3, 4}, 2); 
    for (size_t i = 0; i < twos.dimension(0); ++i) 
      for (size_t j = 0; j < twos.dimension(1); ++j) 
        for (size_t k = 0; k < twos.dimension(2); ++k) 
          REQUIRE(twos(i, j, k) == 2); 
 } 
 
 SECTION("C Multi-dimensional arrays") { 
    Tensor<int32_t, 3, Container> naturals = _C<int[1][6][2]>({{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}}}); 
    REQUIRE(naturals.rank() == 3); 
    REQUIRE(naturals.dimension(0) == 1); 
    REQUIRE(naturals.dimension(1) == 6); 
    REQUIRE(naturals.dimension(2) == 2); 
    int counter = 0; 
    for (size_t i = 0; i < naturals.dimension(0); ++i) 
      for (size_t j = 0; j < naturals.dimension(1); ++j) 
        for (size_t k = 0; k < naturals.dimension(2); ++k) 
          REQUIRE(naturals(i, j, k) == ++counter); 
 } 
 
 SECTION("Fill Method") { 
    auto naturals = Tensor<int32_t, 3, Container>{2, 3, 4}; 
    std::deque<int32_t> container{}; 
    for (int i = 0; i < 24; ++i) container.push_back(i); 
    Fill(naturals, container.begin(), container.end()); 
    for (size_t i = 0; i < naturals.dimension(0); ++i) 
      for (size_t j = 0; j < naturals.dimension(1); ++j) 
        for (size_t k = 0; k < naturals.dimension(2); ++k) 
          REQUIRE((size_t)naturals(i, j, k) == i * 12 + j * 4 + k); 
 
    Fill(naturals, 1); 
    for (size_t i = 0; i < naturals.dimension(0); ++i) 
      for (size_t j = 0; j < naturals.dimension(1); ++j) 
        for (size_t k = 0; k < naturals.dimension(2); ++k) 
          REQUIRE((size_t)naturals(i, j, k) == 1); 
 } 
 
 SECTION("Factory Method") { 
    int count = 0; 
    std::function<int(int)> factory = [&count](int x) -> int { 
      count += x; 
      return count - x; 
    }; 
 
    auto naturals = Tensor<int32_t, 3, Container>(Shape<3>{2, 3, 4}, factory, 1); 
    for (size_t i = 0; i < naturals.dimension(0); ++i) 
      for (size_t j = 0; j < naturals.dimension(1); ++j) 
        for (size_t k = 0; k < naturals.dimension(2); ++k) 
          REQUIRE((size_t)naturals(i, j, k) == i * 12 + j * 4 + k); 
 } 
} 
 
template <template <class> class Container>
void InitializingScalarTests() {
  Scalar<int32_t, Container> scalar_1{}; 
  Scalar<int32_t, Container> scalar_2(0); 
 
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

TEST_CASE("Intializing Tensors") { 
  InitializingTensorTests<data::Array>();
  InitializingTensorTests<data::HashMap>();
}

TEST_CASE("Initializing Scalars") { 
  InitializingScalarTests<data::Array>();
  InitializingScalarTests<data::HashMap>();
}
