#include <tensor.hh>
#include <catch.hh>
#include "test.hh"

using namespace tensor;

template <template <class> class C>
void AddSubtractTests() {
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

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(add(tensor_1,  tensor_2, tensor_1, tensor_2)(i, j, k, l)
              == 6000 * i + 600 * j + 60 * k + 6 *l);

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

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(sub(tensor_2, sub(tensor_2, sub(tensor_1, tensor_1)))
                (i, j, k, l) == 0);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(sub(tensor_2, sub(tensor_2, tensor_1, tensor_1), tensor_1)
                (i, j, k, l) == 1000 * i + 100 * j + 10 * k + l);

  }
}

template <template <class> class C>
void HadarmardTests() {
  Tensor<size_t , 4, C> tensor_1({2, 4, 4, 2});
  Tensor<size_t , 4, C> tensor_2({2, 4, 4, 2});

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
          tensor_1(i, j, k, l) = 2;

  for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
        for (size_t l = 0; l < tensor_2.dimension(3); ++l) 
          tensor_2(i, j, k, l) = -2;

  SECTION("Hadarmard") {
    
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(hadamard(tensor_1, tensor_1)(i, j, k, l) == 4);
             

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(hadamard(tensor_2, hadamard(tensor_2, hadamard(tensor_1, tensor_1)))
                (i, j, k, l) == 16);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(hadamard(tensor_2, tensor_2, hadamard(tensor_1, tensor_1))
                (i, j, k, l) == 16);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_1.dimension(3); ++l) 
            REQUIRE(hadamard(tensor_2, hadamard(tensor_2, tensor_1, tensor_1), tensor_1)
                (i, j, k, l) == 32);
  }
}
 
template <template <class> class C>
void MultiplicationTests() {
  Matrix<int, C> tensor_1({4, 4});
  Matrix<int, C> tensor_2({4, 4});
  
  for (size_t i = 0; i < tensor_1.dimension(0); ++i)
    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
      tensor_1(i, j) = (int)(i + j);

  for (size_t i = 0; i < tensor_2.dimension(0); ++i)
    for (size_t j = 0; j < tensor_2.dimension(1); ++j)
      tensor_2(i, j) = -(int)(i + j);

  Tensor<int, 4, C> tensor_3({4, 4, 2, 2}, 1);
  Tensor<int, 4, C> tensor_4({2, 2, 4, 4}, 1);

  Tensor<int, 3, C> tensor_5({4, 3, 2}, 1);
  Tensor<int, 3, C> tensor_6({4, 3, 2}, 1);

  SECTION("Multiplication -- Default Facing Indices") {
    REQUIRE(mul(tensor_1, tensor_2)(0, 0) == -14);
    REQUIRE(mul(tensor_1, tensor_2)(0, 1) == -20);
    REQUIRE(mul(tensor_1, tensor_2)(0, 2) == -26);
    REQUIRE(mul(tensor_1, tensor_2)(0, 3) == -32);
    REQUIRE(mul(tensor_1, tensor_2)(1, 0) == -20);
    REQUIRE(mul(tensor_1, tensor_2)(1, 1) == -30);
    REQUIRE(mul(tensor_1, tensor_2)(1, 2) == -40);
    REQUIRE(mul(tensor_1, tensor_2)(1, 3) == -50);
    REQUIRE(mul(tensor_1, tensor_2)(2, 0) == -26);
    REQUIRE(mul(tensor_1, tensor_2)(2, 1) == -40);
    REQUIRE(mul(tensor_1, tensor_2)(2, 2) == -54);
    REQUIRE(mul(tensor_1, tensor_2)(2, 3) == -68);
    REQUIRE(mul(tensor_1, tensor_2)(3, 0) == -32);
    REQUIRE(mul(tensor_1, tensor_2)(3, 1) == -50);
    REQUIRE(mul(tensor_1, tensor_2)(3, 2) == -68);
    REQUIRE(mul(tensor_1, tensor_2)(3, 3) == -86);

    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(0, 0) == 2296);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(0, 1) == 3520);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(0, 2) == 4744);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(0, 3) == 5968);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(1, 0) == 3520);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(1, 1) == 5400);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(1, 2) == 7280);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(1, 3) == 9160);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(2, 0) == 4744);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(2, 1) == 7280);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(2, 2) == 9816);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(2, 3) == 12352);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(3, 0) == 5968);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(3, 1) == 9160);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(3, 2) == 12352);
    REQUIRE(mul(tensor_1, tensor_2, tensor_1, tensor_2)(3, 3) == 15544);

    for (size_t i = 0; i < tensor_3.dimension(0); ++i)
      for (size_t j = 0; j < tensor_3.dimension(1); ++j)
        for (size_t k = 0; k < tensor_3.dimension(2); ++k)
          for (size_t l = 0; l < tensor_4.dimension(0); ++l)
            for (size_t m = 0; m < tensor_4.dimension(1); ++m)
              for (size_t n = 0; n < tensor_4.dimension(2); ++n)
             REQUIRE(mul(tensor_3, tensor_4)(i, j, k, l, m, n) == 2);

    for (size_t i = 0; i < tensor_3.dimension(0); ++i)
      for (size_t j = 0; j < tensor_4.dimension(3); ++j)
        REQUIRE(mul(tensor_3.template slice<0, 3>(3, 1), tensor_4.template slice<1, 3>(0, 3))(i, j) == 2);
  }

  SECTION("Multiplication -- Specific Facing Indices") {
    REQUIRE(mul<0, 0>(tensor_5, tensor_6).rank() == 4);
    REQUIRE(mul<0, 0>(tensor_5, tensor_6).dimension(0) == 3);
    REQUIRE(mul<0, 0>(tensor_5, tensor_6).dimension(1) == 2);
    REQUIRE(mul<0, 0>(tensor_5, tensor_6).dimension(2) == 3);
    REQUIRE(mul<0, 0>(tensor_5, tensor_6).dimension(3) == 2);

    REQUIRE(mul<1, 1>(tensor_5, tensor_6).rank() == 4);
    REQUIRE(mul<1, 1>(tensor_5, tensor_6).dimension(0) == 4);
    REQUIRE(mul<1, 1>(tensor_5, tensor_6).dimension(1) == 2);
    REQUIRE(mul<1, 1>(tensor_5, tensor_6).dimension(2) == 4);
    REQUIRE(mul<1, 1>(tensor_5, tensor_6).dimension(3) == 2);

    REQUIRE(mul<2, 2>(tensor_5, tensor_6).rank() == 4);
    REQUIRE(mul<2, 2>(tensor_5, tensor_6).dimension(0) == 4);
    REQUIRE(mul<2, 2>(tensor_5, tensor_6).dimension(1) == 3);
    REQUIRE(mul<2, 2>(tensor_5, tensor_6).dimension(2) == 4);
    REQUIRE(mul<2, 2>(tensor_5, tensor_6).dimension(3) == 3);

    for (size_t i = 0; i < tensor_5.dimension(1); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(2); ++j) 
        for (size_t k = 0; k < tensor_6.dimension(1); ++k) 
          for (size_t l = 0; l < tensor_6.dimension(2); ++l) 
            REQUIRE(mul<0, 0>(tensor_5, tensor_6)(i, j, k, l) == 4);

    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(2); ++j) 
        for (size_t k = 0; k < tensor_6.dimension(0); ++k) 
          for (size_t l = 0; l < tensor_6.dimension(2); ++l) 
            REQUIRE(mul<1, 1>(tensor_5, tensor_6)(i, j, k, l) == 3);

    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_6.dimension(0); ++k) 
          for (size_t l = 0; l < tensor_6.dimension(1); ++l) 
            REQUIRE(mul<2, 2>(tensor_5, tensor_6)(i, j, k, l) == 2);
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

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        tensor_2(i, j, k) = (int)(-100000 * i + -10000 * j + -1000 * k); 

  SECTION("Negation") { 
    auto tensor_3 = tensor_1.neg(); 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(-100000 * i + -10000 * j + -1000 * k)); 
  } 

  SECTION("Map") {
    auto tensor_3 = map<int, data::Array>([](int y, int z) { return y - z; }, tensor_1, tensor_2);
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(200000 * i + 20000 * j + 2000 * k));
  }

  SECTION("Reduction") {
    auto scalar = reduce(10, [](int &x, int y, int z) { return x + y + z + 1; }, tensor_1, tensor_2);
    REQUIRE(scalar == 10 + 2 * 3 * 4);
  }
}

template <template <class> class C>
void ExpressionTests() {
  auto tensor_1 = Tensor<int32_t, 3, C>{2, 3, 4}; 
  auto tensor_2 = Tensor<int32_t, 3, C>{2, 3, 4}; 

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        tensor_1(i, j, k) = 100000 * i + 10000 * j + 1000 * k; 

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        tensor_2(i, j, k) = (int)(-100000 * i + -10000 * j + -1000 * k); 

  // FIXME
  /*
  SECTION("Map") {
    auto tensor_3 = map<int, data::Array>([](int y, int z) { return y; }, tensor_1 - tensor_2);
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(100000 * i + 20000 * j + 2000 * k));
  }

  SECTION("Reduction") {
    auto scalar = reduce(10, [](int &x, int y, int z) { return x + y + 1; }, tensor_1 + tensor_2);
    REQUIRE(scalar == 10 + 2 * 3 * 4);
  }
  */
}

TEST_CASE("Add/Subtract" " | "  "Array") {
  AddSubtractTests<data::Array>();
}

TEST_CASE("Add/Subtract" " | "  "HashMap") {
  AddSubtractTests<data::HashMap>();
}

TEST_CASE("Hadarmard" " | "  "Array") {
  HadarmardTests<data::Array>();
}

TEST_CASE("Hadarmard" " | "  "HashMap") {
  HadarmardTests<data::HashMap>();
}

TEST_CASE("Multiplication" " | "  "Array") {
  MultiplicationTests<data::Array>(); 
}

TEST_CASE("Multiplication" " | "  "HashMap") {
  MultiplicationTests<data::HashMap>(); 
}

TEST_CASE("Misc" " | "  "Array") {
  MiscTests<data::Array>();
}

TEST_CASE("Misc" " | "  "HashMap") {
  MiscTests<data::HashMap>();
}

TEST_CASE("Expression" " | "  "Array") {
  ExpressionTests<data::Array>();
}

TEST_CASE("Expression" " | "  "HashMap") {
  ExpressionTests<data::HashMap>();
}
