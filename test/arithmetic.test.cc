#include <tensor.hh>
#include <catch.hh>
#include "test.hh"

using namespace tensor;

template <template <class> class C>
void AdditionSubtractionTests() {
  Tensor<size_t , 4, C> tensor_1({2, 4, 4, 2});
  Tensor<size_t , 4, C> tensor_2({2, 4, 4, 2});

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;

  for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
        for (size_t l = 0; l < tensor_2.dimension(3); ++l) 
          tensor_2(i, j, k, l) = 2000 * i + 200 * j + 20 * k + 2 * l;

  SECTION("Addition") {

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(add(tensor_1, tensor_1)(i, j, k, l) 
                == 2000 * i + 200 * j + 20 * k + 2 * l);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(add(add(tensor_1,  tensor_2), tensor_1)(i, j, k, l)
              == 4000 * i + 400 * j + 40 * k + 4 *l);

  }

  SECTION("Subtraction") {

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(sub(tensor_1, tensor_1)(i, j, k, l) == 0);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(sub(tensor_2, sub(tensor_2, sub(tensor_1, tensor_1)))
                (i, j, k, l) == 0);

  }
}
 

template <template <class> class C>
void MiscTests() {
  auto tensor_1 = Tensor<int32_t, 3, C>{2, 3, 4}; 
  auto tensor_2 = Tensor<int32_t, 3, C>{2, 3, 4}; 
  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        tensor_1(i, j, k) = 100000 * i + 10000 * j + 1000 * k; 

  SECTION("Negation") { 
    auto tensor_3 = tensor_1.neg(); 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(-100000 * i + -10000 * j + -1000 * k)); 
  } 
}

TEST_CASE(BeginTest("Addition/Subtraction", "Array")) {
  AdditionSubtractionTests<data::Array>();
}

TEST_CASE(BeginTest("Addition/Subtraction", "HashMap")) {
  AdditionSubtractionTests<data::HashMap>();
}

TEST_CASE(BeginTest("Misc", "Array")) {
  MiscTests<data::Array>();
}

TEST_CASE(BeginTest("Misc", "HashMap")) {
  MiscTests<data::HashMap>();
}
