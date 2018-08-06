#include <tensor.hh>
#include <catch.hh>
#include "test.hh"

using namespace tensor;

template <template <class> class Container>
void RawExpressionTests() {
  auto tensor_1 = Tensor<int32_t, 3, Container>{6, 4, 6}; 
  auto tensor_2 = Tensor<int32_t, 3, Container>{6, 4, 6}; 
  REQUIRE((tensor_1 - tensor_2 + tensor_1).rank() == 3);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).dimension(1) == 4);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).dimension(2) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1)(2, 2).rank() == 1);
  REQUIRE((tensor_1 - tensor_2 + tensor_1)(2, 2).dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1)[Indices<2>{2, 2}].rank() == 1);
  REQUIRE((tensor_1 - tensor_2 + tensor_1)[Indices<2>{2, 2}].dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).template slice<0, 1>(2).rank() == 2);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).template slice<0, 1>(2).dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).template slice<0, 1>(2).dimension(1) == 4);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).template slice<0, 2>(Indices<1>{2}).rank() == 2);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).template slice<0, 2>(Indices<1>{2}).dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).template slice<0, 2>(Indices<1>{2}).dimension(1) == 6);

  REQUIRE((tensor_1 * tensor_2 * tensor_1).rank() == 5);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).dimension(0) == 6);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).dimension(1) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).dimension(2) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).dimension(3) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).dimension(4) == 6);
  REQUIRE((tensor_1 * tensor_2 * tensor_1)(2, 2).rank() == 3);
  REQUIRE((tensor_1 * tensor_2 * tensor_1)(2, 2).dimension(0) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1)(2, 2).dimension(1) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1)(2, 2).dimension(2) == 6);
  REQUIRE((tensor_1 * tensor_2 * tensor_1)[Indices<2>{2, 2}].rank() == 3);
  REQUIRE((tensor_1 * tensor_2 * tensor_1)[Indices<2>{2, 2}].dimension(0) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1)[Indices<2>{2, 2}].dimension(1) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1)[Indices<2>{2, 2}].dimension(2) == 6);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).rank() == 4);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).dimension(0) == 6);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).dimension(1) == 4);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).dimension(2) == 4);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).dimension(3) == 6);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<1, 2>(Indices<2>{2, 2}).rank() == 3);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<1, 2>(Indices<2>{2, 2}).dimension(0) == 4);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<1, 2>(Indices<2>{2, 2}).dimension(1) == 4);
  //REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<1, 2>(Indices<2>{2, 2}).dimension(2) == 6);
}

template <template <class> class Container> 
void TensorArithmeticTests() {
  auto tensor_1 = Tensor<int32_t, 3, Container>{2, 3, 4}; 
  auto tensor_2 = Tensor<int32_t, 3, Container>{2, 3, 4}; 
  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        tensor_1(i, j, k) = 100000 * i + 10000 * j + 1000 * k; 
 
  for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
        tensor_2(i, j, k) = 100 * i + 10 * j + 1 * k; 
 
  SECTION("Two Term Addition/Subtract") { 
    Tensor<int32_t, 3, Container> tensor_3 = tensor_1 + tensor_2; 
    Tensor<int32_t, 3, Container> tensor_4 = tensor_3 - tensor_1; 
 
    REQUIRE(typeid(tensor_3) != typeid(tensor_1 + tensor_2)); 
    REQUIRE(typeid(tensor_4) != typeid(tensor_3 - tensor_1)); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(100100 * i + 10010 * j + 1001 * k)); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE((tensor_1 + tensor_2)(i, j, k) == (int)(100100 * i + 10010 * j + 1001 * k)); 
 
    Shape<3> shape_1 = (tensor_1 + tensor_2).shape(); 
    Indices<3> indices{}; 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) { 
          REQUIRE((tensor_1 + tensor_2)[indices] == (int)(100100 * i + 10010 * j + 1001 * k)); 
          indices.increment(shape_1); 
        } 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE(tensor_4(i, j, k) == (int)(100 * i + 10 * j + 1 * k)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_1)(i, j, k) == (int)(100 * i + 10 * j + 1 * k)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_1)[Indices<3>{i, j, k}] == (int)(100 * i + 10 * j + 1 * k)); 
  } 
 
  SECTION("Multi-Term Addition/Subtract") { 
    Tensor<int32_t, 3, Container> tensor_3 = tensor_2 + tensor_1 + tensor_2; 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(100200 * i + 10020 * j + 1002 * k)); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE((tensor_2 + tensor_1 + tensor_2)(i, j, k) == (int)(100200 * i + 10020 * j + 1002 * k)); 
 
    Shape<3> shape_1 = (tensor_2 + tensor_1 + tensor_2).shape(); 
    Indices<3> indices{}; 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) { 
          REQUIRE((tensor_2 + tensor_1 + tensor_2)[indices] == (int)(100200 * i + 10020 * j + 1002 * k)); 
          indices.increment(shape_1); 
        } 
 
    Tensor<int32_t, 3, Container> tensor_4 = tensor_3 - tensor_2 + tensor_1; 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE(tensor_4(i, j, k) == (int)(200100 * i + 20010 * j + 2001 * k)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_2 + tensor_1)(i, j, k) == (int)(200100 * i + 20010 * j + 2001 * k)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_2 + tensor_1)[Indices<3>{i, j, k}] == (int)(200100 * i + 20010 * j + 2001 * k)); 
  } 
 
  SECTION("Negation") { 
    auto tensor_3 = -tensor_1; 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(-100000 * i + -10000 * j + -1000 * k)); 
  } 
 
  SECTION("Assignment Arithmetic") { 
    tensor_1 = tensor_1 + tensor_2; 
 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          REQUIRE(tensor_1(i, j, k) == (int)(100100 * i + 10010 * j + 1001 * k)); 
 
    tensor_1 = tensor_1 - tensor_2; 
 
    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          REQUIRE(tensor_1(i, j, k) == (int)(100000 * i + 10000 * j + 1000 * k)); 
  } 
} 
 
template <template <class> class Container>
void ScalarArithmeticTests() {
  auto tensor_1 = Scalar<int32_t, Container>(10); 
  auto tensor_2 = Scalar<int32_t, Container>(-10); 
 
  SECTION("Binary Addition/Subtract") { 
    REQUIRE(tensor_1() + tensor_2() == 0); 
    REQUIRE((tensor_1 + tensor_2)() == 0); 
    REQUIRE(tensor_1() - tensor_2() == 20); 
    REQUIRE((tensor_1 - tensor_2)() == 20); 
    REQUIRE(tensor_1 + 10 == 20); 
    REQUIRE(10 + tensor_1 == 20); 
    REQUIRE(tensor_1 - 10 == 0); 
 
    Scalar<int32_t, Container> tensor_3 = tensor_1 + tensor_2; 
    REQUIRE(tensor_3 == 0); 
 
    Scalar<int32_t, Container> tensor_4 = tensor_1 - tensor_2; 
    REQUIRE(tensor_4 == 20); 
  } 
 
  SECTION("Multi-Term Addition/Subtract") { 
    REQUIRE(tensor_1 + 10 + tensor_2 + 5 == 15); 
    REQUIRE(tensor_1 - 10 - tensor_2 - 5 == 5); 
    REQUIRE(tensor_1 - 10 + tensor_2 - 5 == -15); 
    REQUIRE(tensor_1 + 10 - tensor_2 + 5 == 35); 
  } 
 
  SECTION("Multiplication") { 
    REQUIRE(tensor_1() * tensor_2() == -100); 
    REQUIRE((tensor_1 * tensor_2)() == -100); 
    REQUIRE(tensor_1 * 4 == 40); 
 
    Scalar<int32_t, Container> tensor_3 = tensor_1 * tensor_2; 
    REQUIRE(tensor_3 == -100); 
  } 
 
  SECTION("Assignment arithmetic") { 
    tensor_1 += 10; 
    REQUIRE(tensor_1() == 20); 
    tensor_1 -= 10; 
    REQUIRE(tensor_1() == 10); 
    tensor_1 *= 10; 
    REQUIRE(tensor_1() == 100); 
    tensor_1 += tensor_2; 
    REQUIRE(tensor_1() == 90); 
    tensor_1 -= tensor_2; 
    REQUIRE(tensor_1() == 100); 
    tensor_1 /= tensor_2; 
    REQUIRE(tensor_1() == -10); 
    tensor_1 += tensor_1 + tensor_2; 
    REQUIRE(tensor_1() == -30); 
    tensor_1 -= tensor_1 + tensor_2; 
    REQUIRE(tensor_1() == 10); 
    tensor_1 *= tensor_1 - tensor_2; 
    REQUIRE(tensor_1() == 200); 
    tensor_1 /= tensor_1 + 10 * tensor_2; 
    REQUIRE(tensor_1() == 2); 
  } 
 
  SECTION("Negation") { 
    REQUIRE(-tensor_1 == -10); 
    REQUIRE(-tensor_1 - 10 + -tensor_2 - 5 == -15); 
  } 
} 
 
template <template <class> class Container>
void ElementwiseArithmeticTests() {
  auto tensor_1 = Tensor<int32_t, 3, Container>({2, 3, 4}, 1); 
 
  SECTION("Scalar Tensor") { 
    Tensor<int32_t, 3, Container> tensor_2 = elem_wise(tensor_1, 4, 
        [](int x, int y) -> int { return x + y; }); 

    for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
          REQUIRE(tensor_2(i, j, k) == 5); 
 
    tensor_2 = elem_wise(tensor_1, 6, 
        [](int x, int y) -> int { return y - x; }); 
    for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
          REQUIRE(tensor_2(i, j, k) == 5); 
 
    tensor_2 = elem_wise(tensor_1, 100, 
        [](int x, int y) -> int { return x * (-y); }); 
    for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
          REQUIRE(tensor_2(i, j, k) == -100); 
 
    tensor_2 = elem_wise(tensor_1, tensor_1, 
        [](int x, int y) -> int { return (x + 9)/(x + y); }); 
    for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
          REQUIRE(tensor_2(i, j, k) == 5); 
  } 
} 
 
template <template <class> class Container>
void TensorMultiplicationTests() {
  auto tensor_1 = Tensor<int32_t, 3, Container>{2, 3, 4}; 
  auto tensor_2 = Tensor<int32_t, 3, Container>{4, 3, 2}; 
 
  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        tensor_1(i, j, k) = 1; 
 
  for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
        tensor_2(i, j, k) = 1; 
 
  SECTION("Binary Multiplication") { 
 
    Tensor<int32_t, 4, Container> tensor_3 = tensor_1 * tensor_2; 
 
    REQUIRE(typeid(tensor_3) != typeid(tensor_1 * tensor_2)); 
    REQUIRE((tensor_1 * tensor_2).rank() == 4); 
    REQUIRE((tensor_1 * tensor_2).dimension(0) == 2); 
    REQUIRE((tensor_1 * tensor_2).dimension(1) == 3); 
    REQUIRE((tensor_1 * tensor_2).dimension(2) == 3); 
    REQUIRE((tensor_1 * tensor_2).dimension(3) == 2); 
    REQUIRE(tensor_3.rank() == 4); 
    REQUIRE(tensor_3.dimension(0) == 2); 
    REQUIRE(tensor_3.dimension(1) == 3); 
    REQUIRE(tensor_3.dimension(2) == 3); 
    REQUIRE(tensor_3.dimension(3) == 2); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_3.dimension(3); ++l) 
            REQUIRE(tensor_3(i, j, k, l) == 4); 
 
    Shape<4> shape_1 = (tensor_1 * tensor_2).shape(); 
    Indices<4> indices_1{}; 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_3.dimension(3); ++l) { 
            REQUIRE((tensor_1 * tensor_2)[indices_1] == 4); 
            indices_1.increment(shape_1); 
          } 
 
    Tensor<int32_t, 2, Container> tensor_4 = tensor_1.template slice<0, 1>(1) * tensor_2(1); 
 
    REQUIRE(typeid(tensor_4) != typeid((tensor_1.template slice<0, 1>(1) * tensor_2(1)))); 
    REQUIRE((tensor_1.template slice<0, 1>(1) * tensor_2(1)).rank() == 2); 
    REQUIRE((tensor_1.template slice<0, 1>(1) * tensor_2(1)).dimension(0) == 2); 
    REQUIRE((tensor_1.template slice<0, 1>(1) * tensor_2(1)).dimension(1) == 2); 
    REQUIRE(tensor_4.rank() == 2); 
    REQUIRE(tensor_4.dimension(0) == 2); 
    REQUIRE(tensor_4.dimension(1) == 2); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        REQUIRE(tensor_4(i, j) == 3); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        REQUIRE((tensor_1.template slice<0, 1>(1) * tensor_2(1))(i, j) == 3); 
 
    Shape<2> shape_2 = (tensor_1.template slice<0, 1>(1) * tensor_2(1)).shape(); 
    Indices<2> indices_2{}; 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) { 
        REQUIRE((tensor_1.template slice<0, 1>(1) * tensor_2(1))[indices_2] == 3); 
        indices_2.increment(shape_2); 
      } 
 
    Tensor<int32_t, 4, Container> tensor_5 = tensor_4 * tensor_3 * tensor_4; 
 
    REQUIRE(tensor_5.rank() == 4); 
    REQUIRE(tensor_5.dimension(0) == 2); 
    REQUIRE(tensor_5.dimension(1) == 3); 
    REQUIRE(tensor_5.dimension(2) == 3); 
    REQUIRE(tensor_5.dimension(3) == 2); 
    REQUIRE((tensor_4 * tensor_3 * tensor_4).rank() == 4); 
    REQUIRE((tensor_4 * tensor_3 * tensor_4).dimension(0) == 2); 
    REQUIRE((tensor_4 * tensor_3 * tensor_4).dimension(1) == 3); 
    REQUIRE((tensor_4 * tensor_3 * tensor_4).dimension(2) == 3); 
    REQUIRE((tensor_4 * tensor_3 * tensor_4).dimension(3) == 2); 
 
    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_5.dimension(3); ++l) 
            REQUIRE(tensor_5(i, j, k, l) == 144); 
 
    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_5.dimension(3); ++l) 
            REQUIRE((tensor_4 * tensor_3 * tensor_4)(i, j, k, l) == 144); 
 
    Shape<4> shape_3 = (tensor_4 * tensor_3 * tensor_4).shape(); 
    Indices<4> indices_3 {}; 
    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_5.dimension(3); ++l) { 
            REQUIRE((tensor_4 * tensor_3 * tensor_4)[indices_3] == 144); 
            indices_3.increment(shape_3); 
          } 
 
 
    Tensor<int32_t, 2, Container> tensor_6{2, 2}; 
 
    tensor_6(0, 0) = 2; 
    tensor_6(0, 1) = 1; 
    tensor_6(1, 0) = 3; 
    tensor_6(1, 1) = 2; 
 
    Tensor<int32_t, 2, Container> tensor_7 = tensor_6 * tensor_6; 
 
    REQUIRE((tensor_6 * tensor_6).rank() == 2); 
    REQUIRE((tensor_6 * tensor_6).dimension(0) == 2); 
    REQUIRE((tensor_6 * tensor_6).dimension(1) == 2); 
    REQUIRE(tensor_7.rank() == 2); 
    REQUIRE(tensor_7.dimension(0) == 2); 
    REQUIRE(tensor_7.dimension(1) == 2); 
 
    REQUIRE(tensor_7(0, 0) == 7); 
    REQUIRE(tensor_7(0, 1) == 4); 
    REQUIRE(tensor_7(1, 0) == 12); 
    REQUIRE(tensor_7(1, 1) == 7); 
    REQUIRE((tensor_6 * tensor_6)(0, 0) == 7); 
    REQUIRE((tensor_6 * tensor_6)(0, 1) == 4); 
    REQUIRE((tensor_6 * tensor_6)(1, 0) == 12); 
    REQUIRE((tensor_6 * tensor_6)(1, 1) == 7); 
    REQUIRE((tensor_6 * tensor_6)[Indices<2>{0, 0}] == 7); 
    REQUIRE((tensor_6 * tensor_6)[Indices<2>{0, 1}] == 4); 
    REQUIRE((tensor_6 * tensor_6)[Indices<2>{1, 0}] == 12); 
    REQUIRE((tensor_6 * tensor_6)[Indices<2>{1, 1}] == 7); 
  } 
 
  SECTION("Combined Multiplication and Addition/Subtraction") { 
 
    Tensor<int32_t, 4, Container> tensor_3 = (tensor_1 * tensor_2 + tensor_1 * tensor_2); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_3.dimension(3); ++l) 
            REQUIRE(tensor_3(i, j, k, l) == 8); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_3.dimension(3); ++l) 
            REQUIRE((tensor_1 * tensor_2 + tensor_1 * tensor_2)(i, j, k, l) == 8); 
 
    Tensor<int32_t, 2, Container> tensor_4 = 
      (tensor_1.template slice<0, 1>(1) - tensor_1.template slice<0, 1>(1)) * tensor_2(1); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        REQUIRE(tensor_4(i, j) == 0); 
  } 
 
  SECTION("Assignment Arithmetic") { 
    Tensor<int32_t, 2, Container> tensor_3 = _A<int[2][2]>({{2, 3}, {4, 5}}); 
    Tensor<int32_t, 2, Container> tensor_4 = _A<int[2][2]>({{6, 7}, {8, 9}}); 
    tensor_3 = tensor_3 * tensor_4 + tensor_4; 
    int correct_vals[2][2] = {{42, 48}, {72, 82}}; 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        REQUIRE(tensor_3(i, j) == correct_vals[i][j]); 
  } 
}

TEST_CASE(BeginTest("Raw Expressions", "Array")) { 
  RawExpressionTests<data::Array>();
}

TEST_CASE(BeginTest("Raw Expressions", "HashMap")) { 
  RawExpressionTests<data::HashMap>();
}
   
TEST_CASE(BeginTest("Basic Tensor Arithmetic", "Array")) { 
  TensorArithmeticTests<data::Array>();
}

TEST_CASE(BeginTest("Basic Tensor Arithmetic", "HashMap")) { 
  TensorArithmeticTests<data::HashMap>();
}

TEST_CASE(BeginTest("Scalar Arithmetic", "Array")) { 
  ScalarArithmeticTests<data::Array>();
}

TEST_CASE(BeginTest("Scalar Arithmetic", "HashMap")) { 
  ScalarArithmeticTests<data::HashMap>();
}

TEST_CASE(BeginTest("Elementwise Arithmetic", "Array")) { 
  ElementwiseArithmeticTests<data::Array>();
}

TEST_CASE(BeginTest("Elementwise Arithmetic", "HashMap")) { 
  ElementwiseArithmeticTests<data::HashMap>();
}

TEST_CASE(BeginTest("Tensor Multplication", "Array")) { 
  TensorMultiplicationTests<data::Array>();
}

TEST_CASE(BeginTest("Tensor Multplication", "HashMap")) { 
  TensorMultiplicationTests<data::HashMap>();
}

