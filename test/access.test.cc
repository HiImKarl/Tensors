#include <tensor.hh>
#include <catch.hh>
#include <array>

using namespace tensor;

TEST_CASE("Tensor Access", "[int]") {

  //     -- Tensor Initialization --   
  auto tensor_1 = Tensor<int32_t, 4>{1, 2, 3, 4};

  // Initialize values
  for (size_t i = 0; i < tensor_1.dimension(0); ++i)
    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
      for (size_t k = 0; k < tensor_1.dimension(2); ++k)
        for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;

  SECTION("Dimensions and Rank") {
    auto tensor_2 = tensor_1(0);
    REQUIRE(tensor_2.rank() == 3);
    for (size_t i = 0; i < tensor_2.rank(); ++i)
      REQUIRE(tensor_2.dimension(i) == i + 2);
    auto tensor_3 = tensor_2(1);
    REQUIRE(tensor_3.rank() == 2);
    for (size_t i = 0; i < tensor_3.rank(); ++i)
      REQUIRE(tensor_3.dimension(i) == i + 3);
  }

  SECTION("Using Indices") {
    Indices<4> indices({0, 0, 0, 3}, tensor_1.shape());
    REQUIRE(tensor_1[indices] == 3);
    Increment(indices);
    REQUIRE(tensor_1[indices] == 10);
    Decrement(indices);
    REQUIRE(tensor_1[indices] == 3);
  }

  SECTION("at vs operator()") {
    REQUIRE(typeid(tensor_1(0)) == typeid(tensor_1.at(0)));
    REQUIRE(typeid(tensor_1(0, 0)) == typeid(tensor_1.at(0, 0)));
    REQUIRE(typeid(tensor_1(0, 0, 0)) == typeid(tensor_1.at(0, 0, 0)));
    REQUIRE(typeid(tensor_1(0, 0, 0, 0)) != typeid(tensor_1.at(0, 0, 0, 0)));
    REQUIRE(typeid(int) == typeid(tensor_1(0,0,0,0)));
    REQUIRE(typeid(int) != typeid(tensor_1.at(0,0,0,0)));
  }
}
