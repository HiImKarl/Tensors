#include <array>
#include "test.hh"

using namespace tensor;
using namespace std;

template <template <class> class Container>
void TensorAccessTests() {
  auto tensor_1 = Tensor<int32_t, 4, Container>{1, 2, 3, 4}; 

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l; 

  SECTION("operator()(...)") { 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(tensor_1(i, j, k, l) == (int32_t)(1000 * i + 100 * j + 10 * k + l)); 
  } 

  SECTION("operator()(...) const") { 
    auto const &const_tensor_1 = tensor_1; 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(const_tensor_1(i, j, k, l) == (int32_t)(1000 * i + 100 * j + 10 * k + l)); 
  } 

  SECTION("operator[](Indices)") { 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(tensor_1[Indices<4>{i, j, k, l}] == (int32_t)(1000 * i + 100 * j + 10 * k + l)); 
  } 

  SECTION("operator[](Indices) const") { 
    auto const &const_tensor_1 = tensor_1; 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(const_tensor_1[Indices<4>{i, j, k, l}] == (int32_t)(1000 * i + 100 * j + 10 * k + l)); 
  } 

  SECTION("operator[](Indices), decrement") { 
    Indices<4> indices{0, 0, 1, 0}; 
    REQUIRE(tensor_1[indices] == 10); 
    REQUIRE(indices.decrement(tensor_1.shape()) == true);   /* 0, 0, 0, 3 */ 
    REQUIRE(tensor_1[indices] == 3); 
    REQUIRE(indices.decrement(tensor_1.shape()) == true);   /* 0, 0, 0, 2 */ 
    REQUIRE(tensor_1[indices] == 2); 
    REQUIRE(indices.decrement(tensor_1.shape()) == true);   /* 0, 0, 0, 1 */ 
    REQUIRE(tensor_1[indices] == 1); 
    REQUIRE(indices.decrement(tensor_1.shape()) == true);   /* 0, 0, 0, 0 */ 
    REQUIRE(tensor_1[indices] == 0); 
    REQUIRE(indices.decrement(tensor_1.shape()) == false);  /* 0, 1, 2, 3 */ 
    REQUIRE(tensor_1[indices] == 123); 
  } 

  SECTION("operator[](Indices), increment") { 
    Indices<4> indices{0, 1, 1, 3}; 
    REQUIRE(tensor_1[indices] == 113); 
    REQUIRE(indices.increment(tensor_1.shape()) == true);   /* 0, 1, 2, 0 */ 
    REQUIRE(tensor_1[indices] == 120); 
    REQUIRE(indices.increment(tensor_1.shape()) == true);   /* 0, 1, 2, 1 */ 
    REQUIRE(tensor_1[indices] == 121); 
    REQUIRE(indices.increment(tensor_1.shape()) == true);   /* 0, 0, 0, 2 */ 
    REQUIRE(tensor_1[indices] == 122); 
    REQUIRE(indices.increment(tensor_1.shape()) == true);   /* 0, 1, 2, 3 */ 
    REQUIRE(tensor_1[indices] == 123); 
    REQUIRE(indices.increment(tensor_1.shape()) == false);  /* 0, 0, 0, 0 */ 
    REQUIRE(tensor_1[indices] == 0); 
  } 

  SECTION("at vs operator()") { 
    REQUIRE(typeid(tensor_1(0)) == typeid(tensor_1.at(0))); 
    REQUIRE(typeid(tensor_1(0, 0)) == typeid(tensor_1.at(0, 0))); 
    REQUIRE(typeid(tensor_1(0, 0, 0)) == typeid(tensor_1.at(0, 0, 0))); 
    REQUIRE(typeid(tensor_1(0, 0, 0, 0)) != typeid(tensor_1.at(0, 0, 0, 0))); 
    REQUIRE(typeid(int) == typeid(tensor_1(0,0,0,0))); 
    REQUIRE(typeid(int) != typeid(tensor_1.at(0,0,0,0))); 
  } 

  SECTION("Sub-Tensor") { 
    auto const tensor_2 = tensor_1(0); 
    REQUIRE(tensor_2.rank() == 3); 
    for (size_t i = 0; i < tensor_2.rank(); ++i) 
      REQUIRE(tensor_2.dimension(i) == i + 2); 
    for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
          REQUIRE(tensor_2(i, j, k) == (int32_t)(100 * i + 10 * j + k)); 

    auto const tensor_3 = tensor_2(1); 
    REQUIRE(tensor_3.rank() == 2); 
    for (size_t i = 0; i < tensor_3.rank(); ++i) 
      REQUIRE(tensor_3.dimension(i) == i + 3); 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
    REQUIRE(tensor_3(i, j) == (int32_t)(100 + 10 * i + j)); 

    auto const tensor_4 = tensor_1[Indices<1>{0}]; 
    REQUIRE(tensor_4.rank() == 3); 
    for (size_t i = 0; i < tensor_4.rank(); ++i) 
      REQUIRE(tensor_4.dimension(i) == i + 2); 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
    REQUIRE(tensor_4[Indices<3>{i, j, k}] == (int32_t)(100 * i + 10 * j + k)); 

    auto const tensor_5 = tensor_4[Indices<1>{1}]; 
    REQUIRE(tensor_5.rank() == 2); 
    for (size_t i = 0; i < tensor_5.rank(); ++i) 
      REQUIRE(tensor_5.dimension(i) == i + 3); 
    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        REQUIRE(tensor_5[Indices<2>{i, j}] == (int32_t)(100 + 10 * i + j)); 
  }
}

// Instantiate test cases

TEST_CASE("Tensor Access") { 
  TensorAccessTests<data::Array>();
  TensorAccessTests<data::HashMap>();
}
