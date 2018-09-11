#include <tensor.hh>
#include <catch.hh>
#include "test.hh"

using namespace tensor;

template <template <class> class C>
void RawExpressionTests() {
  auto tensor_1 = Tensor<int32_t, 3, C>{6, 4, 6}; 
  auto tensor_2 = Tensor<int32_t, 3, C>{6, 4, 6}; 
  auto tensor_3 = Tensor<int32_t, 2, C>{9, 9};
  auto tensor_4 = Tensor<int32_t, 2, C>{9, 9};
  REQUIRE((tensor_1 - tensor_2 + tensor_1).rank() == 3);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).dimension(1) == 4);
  REQUIRE((tensor_1 - tensor_2 + tensor_1).dimension(2) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1)(2, 2).rank() == 1);
  REQUIRE((tensor_1 - tensor_2 + tensor_1)(2, 2).dimension(0) == 6);

  REQUIRE((tensor_1 - tensor_2 + tensor_1 - tensor_2)[Indices<2>{2, 2}].rank() == 1);
  REQUIRE((tensor_1 - tensor_2 + tensor_1 - tensor_2)[Indices<2>{2, 2}].dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1 - tensor_2).template slice<0, 1>(2).rank() == 2);
  REQUIRE((tensor_1 - tensor_2 + tensor_1 - tensor_2).template slice<0, 1>(2).dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1 - tensor_2).template slice<0, 1>(2).dimension(1) == 4);
  REQUIRE((tensor_1 - tensor_2 + tensor_1 - tensor_2).template slice<0, 2>(Indices<1>{2}).rank() == 2);
  REQUIRE((tensor_1 - tensor_2 + tensor_1 - tensor_2).template slice<0, 2>(Indices<1>{2}).dimension(0) == 6);
  REQUIRE((tensor_1 - tensor_2 + tensor_1 - tensor_2).template slice<0, 2>(Indices<1>{2}).dimension(1) == 6);

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

  REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).rank() == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).dimension(0) == 6);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).dimension(1) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).dimension(2) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<0, 1, 3>(2).dimension(3) == 6);
  REQUIRE((tensor_1 * tensor_1 * tensor_1).template slice<1, 2>(Indices<2>{2, 2}).rank() == 3);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<1, 2>(Indices<2>{2, 2}).dimension(0) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<1, 2>(Indices<2>{2, 2}).dimension(1) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1).template slice<1, 2>(Indices<2>{2, 2}).dimension(2) == 6);

  REQUIRE((tensor_1 * tensor_2 * tensor_1 * tensor_2).template slice<0, 3, 5>(2, 2, 3).rank() == 3);
  REQUIRE((tensor_1 * tensor_2 * tensor_1 * tensor_2).template slice<0, 3, 5>(2, 2, 3).dimension(0) == 6);
  REQUIRE((tensor_1 * tensor_2 * tensor_1 * tensor_2).template slice<0, 3, 5>(2, 2, 3).dimension(1) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1 * tensor_2).template slice<0, 3, 5>(2, 2, 3).dimension(2) == 6);
  REQUIRE((tensor_1 * tensor_1 * tensor_1 * tensor_2).template slice<1, 2, 3, 4>(Indices<2>{2, 2}).rank() == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1 * tensor_2).template slice<1, 2, 3, 4>(Indices<2>{2, 2}).dimension(0) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1 * tensor_2).template slice<1, 2, 3, 4>(Indices<2>{2, 2}).dimension(1) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1 * tensor_2).template slice<1, 2, 3, 4>(Indices<2>{2, 2}).dimension(2) == 4);
  REQUIRE((tensor_1 * tensor_2 * tensor_1 * tensor_2).template slice<1, 2, 3, 4>(Indices<2>{2, 2}).dimension(3) == 4);

  REQUIRE((tensor_4 * tensor_3 * tensor_4 * tensor_3 * tensor_4).template slice<0>(2).rank() == 1);
  REQUIRE((tensor_4 * tensor_3 * tensor_4 * tensor_3 * tensor_4).template slice<0>(2).dimension(0) == 9);
  REQUIRE((tensor_4 * tensor_3 * tensor_4 * tensor_3 * tensor_4).template slice<0>(Indices<0>{}).rank() == 2);
  REQUIRE((tensor_4 * tensor_3 * tensor_4 * tensor_3 * tensor_4).template slice<0>(Indices<0>{}).dimension(0) == 9);
  REQUIRE((tensor_4 * tensor_3 * tensor_4 * tensor_3 * tensor_4).template slice<0>(Indices<0>{}).dimension(1) == 9);
}

template <template <class> class C> 
void TensorArithmeticTests() {
  auto tensor_1 = Tensor<int32_t, 3, C>{2, 3, 4}; 
  auto tensor_2 = Tensor<int32_t, 3, C>{2, 3, 4}; 
  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        tensor_1(i, j, k) = 100000 * i + 10000 * j + 1000 * k; 
 
  for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
        tensor_2(i, j, k) = 100 * i + 10 * j + 1 * k; 
 
  SECTION("Two Term Addition/Subtract") { 
    Tensor<int32_t, 3, C> tensor_3 = tensor_1 + tensor_2; 
    Tensor<int32_t, 3, C> tensor_4 = tensor_3 - tensor_1; 
    Tensor<int32_t, 3, C> tensor_5 = tensor_2 % tensor_2; 
 
    REQUIRE(typeid(tensor_3) != typeid(tensor_1 + tensor_2)); 
    REQUIRE(typeid(tensor_4) != typeid(tensor_3 - tensor_1)); 
    REQUIRE(typeid(tensor_5) != typeid(tensor_2 % tensor_2)); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(100100 * i + 10010 * j + 1001 * k)); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE((tensor_1 + tensor_2)(i, j, k) == (int)(100100 * i + 10010 * j + 1001 * k)); 

    for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
        REQUIRE((tensor_1 + tensor_2).template slice<1, 2>(0)(j, k) == (int)(10010 * j + 1001 * k)); 
 
    Shape<3> shape_1 = (tensor_1 + tensor_2).shape(); 
    Indices<3> indices_1{}; 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i)
      for (size_t j = 0; j < tensor_3.dimension(1); ++j)
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) { 
          REQUIRE((tensor_1 + tensor_2)[indices_1] == (int)(100100 * i + 10010 * j + 1001 * k)); 
          indices_1.increment(shape_1); 
        }
 
    Shape<1> shape_2 = (tensor_1 + tensor_2).template slice<2>(1, 1).shape(); 
    Indices<1> indices_2{}; 
    for (size_t k = 0; k < tensor_3.dimension(2); ++k) {
          REQUIRE((tensor_1 + tensor_2).template slice<2>(1, 1)[indices_2] == (int)(100100 + 10010 + 1001 * k)); 
          indices_2.increment(shape_2); 
    }
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE(tensor_4(i, j, k) == (int)(100 * i + 10 * j + k)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_1)(i, j, k) == (int)(100 * i + 10 * j + 1 * k)); 

    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_1).template slice<>()(i, j, k) == (int)(100 * i + 10 * j + 1 * k)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_1)[Indices<3>{i, j, k}] == (int)(100 * i + 10 * j + 1 * k)); 

    for (size_t i = 0; i < tensor_4.dimension(0); ++i)
      for (size_t k = 0; k < tensor_4.dimension(2); ++k)
        REQUIRE((tensor_3 - tensor_1).template slice<0, 2>(2)[Indices<2>{i, k}] == (int)(100 * i + 20 + 1 * k)); 

    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
          REQUIRE(tensor_5(i, j, k) == (int)((100 * i + 10 * j + k) * (100 * i + 10 * j + k)));

    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
          REQUIRE((tensor_2 % tensor_2)(i, j, k) == 
              (int)((100 * i + 10 * j + 1 * k) * (100 * i + 10 * j + 1 * k))); 

    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
          REQUIRE((tensor_2 % tensor_2).template slice<>()(i, j, k) == 
              (int)((100 * i + 10 * j + 1 * k) * (100 * i + 10 * j + 1 * k))); 
 
    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
          REQUIRE((tensor_2 % tensor_2)[Indices<3>{i, j, k}] == 
              (int)((100 * i + 10 * j + 1 * k) * (100 * i + 10 * j + 1 * k))); 

    for (size_t i = 0; i < tensor_5.dimension(0); ++i)
      for (size_t k = 0; k < tensor_5.dimension(2); ++k)
        REQUIRE((tensor_2 % tensor_2).template slice<0, 2>(2)[Indices<2>{i, k}] == 
            (int)((100 * i + 20 + 1 * k) * (100 * i + 20 + 1 * k))); 
  } 

  SECTION("Negation") {
    Tensor<int32_t, 3, C> tensor_3 = -tensor_1;

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        REQUIRE(tensor_3(i, j, k) == -100000 * (int)i + -10000 * (int)j + -1000 * (int)k); 

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        REQUIRE(-tensor_2(i, j, k) == -100 * (int)i + -10 * (int)j + -1 * (int)k); 

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        REQUIRE(-tensor_2[Indices<3>{i, j, k}] == -100 * (int)i + -10 * (int)j + -1 * (int)k); 

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
      REQUIRE(-tensor_2.template slice<0, 1, 2>()(i, j, k) == -100 * (int)i + -10 * (int)j + -1 * (int)k); 

  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      REQUIRE(-tensor_2.template slice<0, 1>(2)[Indices<2>{i, j}] == -100 * (int)i + -10 * (int)j + -2); 

  }
 
  SECTION("Multi-Term Arithmetic") { 
    Tensor<int32_t, 3, C> tensor_3 = tensor_2 + tensor_1 + tensor_2; 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(100200 * i + 10020 * j + 1002 * k)); 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE((tensor_2 + tensor_1 + tensor_2)(i, j, k) 
              == (int)(100200 * i + 10020 * j + 1002 * k)); 
 
    Shape<3> shape_1 = (tensor_2 + tensor_1 + tensor_2).shape(); 
    Indices<3> indices{}; 
 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) { 
          REQUIRE((tensor_2 + tensor_1 + tensor_2)[indices] 
              == (int)(100200 * i + 10020 * j + 1002 * k)); 
          indices.increment(shape_1); 
        } 
 
    Tensor<int32_t, 3, C> tensor_4 = tensor_3 - tensor_2 + tensor_1; 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE(tensor_4(i, j, k) == (int)(200100 * i + 20010 * j + 2001 * k)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_2 + tensor_1)(i, j, k) 
              == (int)(200100 * i + 20010 * j + 2001 * k)); 

    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_2 + tensor_1).template slice<>(i, j, k) 
              == (int)(200100 * i + 20010 * j + 2001 * k)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_2 + tensor_1)[Indices<3>{i, j, k}] 
              == (int)(200100 * i + 20010 * j + 2001 * k)); 

    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_2 + tensor_1).template slice<0, 2>(2)[Indices<2>{i, k}] 
              == (int)(200100 * i + 40020 + 2001 * k)); 

    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_2 + (-tensor_1))(i, j, k) 
              == (int)(100 * i + 10 * j + k)); 

    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      REQUIRE((tensor_3 - tensor_2 + (-tensor_1)).template slice<0>(1, 1)(i)
              == (int)(100 * i + 10 + 1)); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(2); ++k) 
          REQUIRE((tensor_3 - tensor_2 + (-tensor_1))[Indices<3>{i, j, k}] 
              == (int)(100 * i + 10 * j + k)); 

    REQUIRE((tensor_3 - tensor_2 + (-tensor_1)).template slice<0, 1, 2>()[Indices<3>{1, 1, 1}] 
              == (int)(100 + 10 + 1)); 
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

    tensor_1 = _NA(tensor_1 + tensor_2);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          REQUIRE(tensor_1(i, j, k) == (int)(100100 * i + 10010 * j + 1001 * k)); 

    tensor_1 = _NA(tensor_1 - tensor_2);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
          REQUIRE(tensor_1(i, j, k) == (int)(100000 * i + 10000 * j + 1000 * k)); 
  } 
} 
 
template <template <class> class C>
void ScalarArithmeticTests() {
  auto scalar_1 = Scalar<int32_t, C>(10); 
  auto scalar_2 = Scalar<int32_t, C>(-10); 
 
  SECTION("Binary Addition/Subtract") { 
    REQUIRE(scalar_1() + scalar_2() == 0); 
    REQUIRE((scalar_1 + scalar_2)() == 0); 
    REQUIRE(scalar_1() - scalar_2() == 20); 
    REQUIRE((scalar_1 - scalar_2)() == 20); 
    REQUIRE(scalar_1 + 10 == 20); 
    REQUIRE(10 + scalar_1 == 20); 
    REQUIRE(scalar_1 - 10 == 0); 
 
    Scalar<int32_t, C> scalar_3 = scalar_1 + scalar_2; 
    REQUIRE(scalar_3 == 0); 
 
    Scalar<int32_t, C> scalar_4 = scalar_1 - scalar_2; 
    REQUIRE(scalar_4 == 20); 
  } 
 
  SECTION("Multi-Term Addition/Subtract") { 
    REQUIRE(scalar_1 + 10 + scalar_2 + 5 == 15); 
    REQUIRE(scalar_1 - 10 - scalar_2 - 5 == 5); 
    REQUIRE(scalar_1 - 10 + scalar_2 - 5 == -15); 
    REQUIRE(scalar_1 + 10 - scalar_2 + 5 == 35); 
  } 
 
  SECTION("Multiplication") { 
    REQUIRE(scalar_1() * scalar_2() == -100); 
    REQUIRE((scalar_1 * scalar_2)() == -100); 
    REQUIRE(scalar_1 * 4 == 40); 
 
    Scalar<int32_t, C> scalar_3 = scalar_1 * scalar_2; 
    REQUIRE(scalar_3 == -100); 
  } 
 
  SECTION("Assignment arithmetic") { 
    scalar_1 += 10; 
    REQUIRE(scalar_1() == 20); 
    scalar_1 -= 10; 
    REQUIRE(scalar_1() == 10); 
    scalar_1 *= 10; 
    REQUIRE(scalar_1() == 100); 
    scalar_1 += scalar_2; 
    REQUIRE(scalar_1() == 90); 
    scalar_1 -= scalar_2; 
    REQUIRE(scalar_1() == 100); 
    scalar_1 /= scalar_2; 
    REQUIRE(scalar_1() == -10); 
    scalar_1 += scalar_1 + scalar_2; 
    REQUIRE(scalar_1() == -30); 
    scalar_1 -= scalar_1 + scalar_2; 
    REQUIRE(scalar_1() == 10); 
    scalar_1 *= scalar_1 - scalar_2; 
    REQUIRE(scalar_1() == 200); 
    scalar_1 /= scalar_1 + 10 * scalar_2; 
    REQUIRE(scalar_1() == 2); 
  } 
 
  SECTION("Negation") { 
    REQUIRE(scalar_1.neg() == -10); 
  } 
} 
 
template <template <class> class C>
void TensorMultiplicationTests() {
  auto tensor_1 = Tensor<int32_t, 3, C>{2, 3, 4}; 
  auto tensor_2 = Tensor<int32_t, 3, C>{4, 3, 2}; 
 
  for (size_t i = 0; i < tensor_1.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_1.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_1.dimension(2); ++k) 
        tensor_1(i, j, k) = 1; 
 
  for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
    for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
      for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
        tensor_2(i, j, k) = 1; 

  auto tensor_3 = Tensor<int, 3, C>({4, 3, 2}, 1);
  auto tensor_4 = Tensor<int, 3, C>({4, 3, 2}, 1);
 
  SECTION("Binary Multiplication") { 
 
    Tensor<int32_t, 4, C> tensor_3 = tensor_1 * tensor_2; 
 
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
 
    Tensor<int32_t, 2, C> tensor_4 = tensor_1.template slice<0, 1>(1) * tensor_2(1); 
 
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
 
    Tensor<int32_t, 4, C> tensor_5 = tensor_4 * tensor_3 * tensor_4; 
 
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

    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
        for (size_t l = 0; l < tensor_5.dimension(3); ++l) 
          REQUIRE((tensor_4 * tensor_3 * tensor_4).template slice<0, 2>(2)(i, k, l) == 144); 
 
    Shape<4> shape_3 = (tensor_4 * tensor_3 * tensor_4).shape(); 
    Indices<4> indices_3 {}; 
    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_5.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_5.dimension(2); ++k) 
          for (size_t l = 0; l < tensor_5.dimension(3); ++l) { 
            REQUIRE((tensor_4 * tensor_3 * tensor_4)[indices_3] == 144); 
            indices_3.increment(shape_3); 
          } 

    Shape<2> shape_4 = (tensor_4 * tensor_3 * tensor_4).template slice<0>(1, 1).shape(); 
    Indices<2> indices_4 {}; 
    for (size_t i = 0; i < tensor_5.dimension(0); ++i) 
        for (size_t l = 0; l < tensor_5.dimension(3); ++l) {
          REQUIRE((tensor_4 * tensor_3 * tensor_4).template slice<0>(1, 1)[indices_4] == 144); 
          indices_4.increment(shape_4);
        }
 
    Tensor<int32_t, 2, C> tensor_6{2, 2}; 
 
    tensor_6(0, 0) = 2; 
    tensor_6(0, 1) = 1; 
    tensor_6(1, 0) = 3; 
    tensor_6(1, 1) = 2; 
 
    Tensor<int32_t, 2, C> tensor_7 = tensor_6 * tensor_6; 
 
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

    REQUIRE((tensor_6 * tensor_6).template slice<1>(0)(0) == 7); 
    REQUIRE((tensor_6 * tensor_6).template slice<1>(0)(1) == 4); 
    REQUIRE((tensor_6 * tensor_6).template slice<0>(0)(1) == 12); 
    REQUIRE((tensor_6 * tensor_6).template slice<>()(1, 1) == 7); 

    REQUIRE((tensor_6 * tensor_6)[Indices<2>{0, 0}] == 7); 
    REQUIRE((tensor_6 * tensor_6)[Indices<2>{0, 1}] == 4); 
    REQUIRE((tensor_6 * tensor_6)[Indices<2>{1, 0}] == 12); 
    REQUIRE((tensor_6 * tensor_6)[Indices<2>{1, 1}] == 7); 

    REQUIRE((tensor_6 * tensor_6).template slice<0>(0)[Indices<1>{0}] == 7); 
    REQUIRE((tensor_6 * tensor_6).template slice<0>(1)[Indices<1>{0}] == 4); 
    REQUIRE((tensor_6 * tensor_6).template slice<1>(1)[Indices<1>{0}] == 12); 
    REQUIRE((tensor_6 * tensor_6).template slice<0, 1>()[Indices<2>{1, 1}] == 7); 
  } 

  SECTION("Multiplication -- Specific Facing Indices") {
    REQUIRE(_mul<0, 0>(tensor_3, tensor_4).rank() == 4);
    REQUIRE(_mul<0, 0>(tensor_3, tensor_4).dimension(0) == 3);
    REQUIRE(_mul<0, 0>(tensor_3, tensor_4).dimension(1) == 2);
    REQUIRE(_mul<0, 0>(tensor_3, tensor_4).dimension(2) == 3);
    REQUIRE(_mul<0, 0>(tensor_3, tensor_4).dimension(3) == 2);

    REQUIRE(_mul<1, 1>(tensor_3, tensor_4).rank() == 4);
    REQUIRE(_mul<1, 1>(tensor_3, tensor_4).dimension(0) == 4);
    REQUIRE(_mul<1, 1>(tensor_3, tensor_4).dimension(1) == 2);
    REQUIRE(_mul<1, 1>(tensor_3, tensor_4).dimension(2) == 4);
    REQUIRE(_mul<1, 1>(tensor_3, tensor_4).dimension(3) == 2);

    REQUIRE(_mul<2, 2>(tensor_3, tensor_4).rank() == 4);
    REQUIRE(_mul<2, 2>(tensor_3, tensor_4).dimension(0) == 4);
    REQUIRE(_mul<2, 2>(tensor_3, tensor_4).dimension(1) == 3);
    REQUIRE(_mul<2, 2>(tensor_3, tensor_4).dimension(2) == 4);
    REQUIRE(_mul<2, 2>(tensor_3, tensor_4).dimension(3) == 3);

    for (size_t i = 0; i < tensor_3.dimension(1); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(2); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(1); ++k) 
          for (size_t l = 0; l < tensor_4.dimension(2); ++l) 
            REQUIRE(_mul<0, 0>(tensor_3, tensor_4)(i, j, k, l) == 4);

    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(2); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(0); ++k) 
          for (size_t l = 0; l < tensor_4.dimension(2); ++l) 
            REQUIRE(_mul<1, 1>(tensor_3, tensor_4)(i, j, k, l) == 3);

    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_4.dimension(0); ++k) 
          for (size_t l = 0; l < tensor_4.dimension(1); ++l) 
            REQUIRE(_mul<2, 2>(tensor_3, tensor_4)(i, j, k, l) == 2);
  }
 
  SECTION("Combined Multiplication and Addition/Subtraction") { 
 
    Tensor<int32_t, 4, C> tensor_3 = (tensor_1 * tensor_2 + tensor_1 * tensor_2); 
 
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
 
    Tensor<int32_t, 2, C> tensor_4 = 
      (tensor_1.template slice<0, 1>(1) - tensor_1.template slice<0, 1>(1)) * tensor_2(1); 
 
    for (size_t i = 0; i < tensor_4.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_4.dimension(1); ++j) 
        REQUIRE(tensor_4(i, j) == 0); 
  } 
 
  SECTION("Assignment Arithmetic") { 
    Tensor<int32_t, 2, C> tensor_3 = _C<int[2][2]>({{2, 3}, {4, 5}}); 
    Tensor<int32_t, 2, C> tensor_4 = _C<int[2][2]>({{6, 7}, {8, 9}}); 
    tensor_3 = tensor_3 * tensor_4 + tensor_4; 
    int correct_vals[2][2] = {{42, 48}, {72, 82}}; 
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        REQUIRE(tensor_3(i, j) == correct_vals[i][j]); 
  } 
}

template <template <class> class C>
void TensorManipulationTests() 
{
  Tensor<int, 3, C> tensor_1{2, 2, 2};
  Tensor<int, 3, C> tensor_2{2, 2, 2};
  Tensor<int, 3, C> tensor_3{2, 2, 2};
  
  for (size_t i = 0; i < tensor_1.dimension(0); ++i)
    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
      for (size_t k = 0; k < tensor_1.dimension(2); ++k)
        tensor_1(i, j, k) = -(int)(i + j + k);

  for (size_t i = 0; i < tensor_1.dimension(0); ++i)
    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
      for (size_t k = 0; k < tensor_1.dimension(2); ++k)
        tensor_2(i, j, k) = (int)(i + j + k);

  for (size_t i = 0; i < tensor_1.dimension(0); ++i)
    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
      for (size_t k = 0; k < tensor_1.dimension(2); ++k)
        tensor_3(i, j, k) = (int)(100 * i + 10 * j + k);

  SECTION("Single Tensor Map") {
    Tensor<int, 3, C> tensor_ = _map([](int x) { return 2 * x; }, tensor_1);
    for (size_t i = 0; i < tensor_1.dimension(0); ++i)
      for (size_t j = 0; j < tensor_1.dimension(1); ++j)
        for (size_t k = 0; k < tensor_1.dimension(2); ++k)
          REQUIRE(tensor_(i, j, k) == -(int)(i + j + k) * 2);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i)
      for (size_t j = 0; j < tensor_1.dimension(1); ++j)
        for (size_t k = 0; k < tensor_1.dimension(2); ++k)
          REQUIRE(_map([](int x) { return 2 * x; }, tensor_1)(i, j, k) 
              == -(int)(i + j + k) * 2);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i)
      for (size_t j = 0; j < tensor_1.dimension(1); ++j)
        for (size_t k = 0; k < tensor_1.dimension(2); ++k)
          REQUIRE(_map([](int x) { return 2 * x; }, tensor_1)[Indices<3>{i, j, k}]
              == -(int)(i + j + k) * 2);

    for (size_t i = 0; i < tensor_1.dimension(0); ++i)
      for (size_t k = 0; k < tensor_1.dimension(2); ++k)
        REQUIRE(_map([](int x) { return 2 * x; }, tensor_1).template slice<0, 2>(1)(i, k) == -(int)(i + 1 + k) * 2);

    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
      REQUIRE(_map([](int x) { return 2 * x; }, tensor_1).template 
          slice<1>(1, 1)[Indices<1>{j}] == -(int)(1 + j + 1) * 2);
  }

  SECTION("Multi Tensor Map") {
    auto add_neg = [](int x, int y, int z) { return -(x + y + z); };
    Tensor<int, 3, C> tensor_ = _map(add_neg, tensor_1, tensor_2, tensor_3);
    for (size_t i = 0; i < tensor_1.dimension(0); ++i)
      for (size_t j = 0; j < tensor_1.dimension(1); ++j)
        for (size_t k = 0; k < tensor_1.dimension(2); ++k)
          REQUIRE(tensor_(i, j, k) == -(int)(100 * i + 10 * j + k));

    for (size_t i = 0; i < tensor_1.dimension(0); ++i)
      for (size_t j = 0; j < tensor_1.dimension(1); ++j)
        for (size_t k = 0; k < tensor_1.dimension(2); ++k)
          REQUIRE(_map(add_neg, tensor_1, tensor_2, tensor_3)(i, j, k) 
              == -(int)(100 * i + 10 * j + k));

    for (size_t i = 0; i < tensor_1.dimension(0); ++i)
      for (size_t j = 0; j < tensor_1.dimension(1); ++j)
        for (size_t k = 0; k < tensor_1.dimension(2); ++k)
          REQUIRE(_map(add_neg, tensor_1, tensor_2, tensor_3)[Indices<3>{i, j, k}]
              == -(int)(100 * i + 10 * j + k));

    for (size_t i = 0; i < tensor_1.dimension(0); ++i)
      for (size_t k = 0; k < tensor_1.dimension(2); ++k)
        REQUIRE(_map(add_neg, tensor_1, tensor_2, tensor_3).template 
            slice<0, 2>(1)(i, k) == -(int)(100 * i + 10 + k));

    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
        REQUIRE(_map(add_neg, tensor_1, tensor_2, tensor_3).template 
          slice<1>(1, 1)[Indices<1>{j}] == -(int)(100 + 10 * j + 1));

    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
        REQUIRE(_map(add_neg, tensor_1 + tensor_2, tensor_2 + tensor_1, tensor_3).template 
          slice<1>(1, 1)[Indices<1>{j}] == -(int)(100 + 10 * j + 1));

    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
      for (size_t k = 0; k < tensor_1.dimension(2); ++k)
        REQUIRE(_map(add_neg, tensor_1 * tensor_2, tensor_1 * tensor_1, (tensor_1 + tensor_2) * tensor_3).template 
          slice<1, 2>(1, 1)[Indices<2>{j, k}] == 0);
  }

  SECTION("Single Tensor Reduce") {
    Scalar<int> x = _reduce(0, [](int x, int y) { return x + y; }, tensor_1);
    REQUIRE(x == -12);
    REQUIRE(_reduce(0, [](int x, int y) { return x + y; }, tensor_2)() == 12);
    REQUIRE(_reduce(0, [](int x, int y) { return x + y; }, tensor_2)[Indices<0>{}] == 12);
    REQUIRE(_reduce(0, [](int x, int y) { return x + y; }, tensor_2).template slice<>() == 12);
    REQUIRE(_reduce(0, [](int x, int y) { return x + y; }, tensor_2).template slice<>(Indices<0>{}) == 12);

    REQUIRE(_reduce(0, [](int x, int y) { return x + y; }, tensor_1 - tensor_2)() == -24);
    REQUIRE(_reduce(0, [](int x, int y) { return x + y; }, 
          tensor_1 + tensor_1 - tensor_1 + tensor_2)[Indices<0>{}] == 0);
    REQUIRE(_reduce(0, [](int x, int y) { return x + y; }, tensor_2 - tensor_1).template slice<>() == 24);
    REQUIRE(_reduce(0, [](int x, int y) { return x + y; }, tensor_1 - tensor_1 + tensor_1).template slice<>(Indices<0>{}) == -12);

    x = _reduce(0, [](int x, int y) { return x + y; }, tensor_1 * tensor_2 * (tensor_2 - tensor_1));
    REQUIRE(x == -1056);
  }

  SECTION("Multi Tensor Reduce") {
    auto add = [](int x, int y1, int y2, int y3) { return x + y1 - y2 - y3; };
    Scalar<int> x = _reduce(0, add, tensor_1, tensor_2, tensor_3);
    REQUIRE(x == -468);
    REQUIRE(_reduce(0, add, tensor_1, tensor_1, tensor_3)() == -444);
    REQUIRE(_reduce(0, add, tensor_1, tensor_1, tensor_3)[Indices<0>{}] == -444);
    REQUIRE(_reduce(-111, add, tensor_1, tensor_1, tensor_3).template slice<>() == -555);
    REQUIRE(_reduce(111, add, tensor_1, tensor_1, tensor_3).template slice<>(Indices<0>{}) == -333);
  }

  SECTION("Combined Map/Reduce/Arithmetic") {
    auto fn1 = [](int x, int y) { return x * 2 - y * 2; };
    auto fn2 = [](int &x, int y1, int y2) { return x + y1 * 2 + y2 * 2; }; 
    // FIXME -- Manually compute the return value
    Scalar<int> x = _reduce(0, fn2, tensor_2 * tensor_1, _map(fn1, tensor_1 * tensor_1, tensor_2 * tensor_1)
          - tensor_2 * tensor_2);

    auto fn3 = [](int y1, int y2) { return y1 * 2 + y2 * 2; };
    auto fn4 = [](int &x, int y1, int y2, int y3) { return x + y1 + y2 + y3; };
    
    REQUIRE(_reduce(0, fn4, tensor_1, tensor_2, 
          _map(fn3, tensor_1, tensor_1) + tensor_2)() == -36);
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

TEST_CASE(BeginTest("Tensor Multplication", "Array")) { 
  TensorMultiplicationTests<data::Array>();
}

TEST_CASE(BeginTest("Tensor Multplication", "HashMap")) { 
  TensorMultiplicationTests<data::HashMap>();
}

TEST_CASE(BeginTest("Tensor Manipulation", "Array")) {
  TensorManipulationTests<data::Array>();
}

TEST_CASE(BeginTest("Tensor Manipulation", "HashMap")) {
  TensorManipulationTests<data::HashMap>();
}

