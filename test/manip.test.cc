#include <tensor.hh>
#include <catch.hh>
#include "test.hh"

using namespace tensor;

template <template <class> class Container>
void MethodTests() {
  auto tensor = Tensor<int32_t, 4, Container>{2, 4, 6, 8}; 
  auto tensor_2 = Tensor<int32_t, 3, Container>({2, 3, 4}, 1); 

  for (size_t i = 0; i < tensor.dimension(0); ++i) 
    for (size_t j = 0; j < tensor.dimension(1); ++j) 
      for (size_t k = 0; k < tensor.dimension(2); ++k) 
        for (size_t l = 0; l < tensor.dimension(3); ++l) 
          tensor(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l; 

  SECTION("Tranpose") { 
    Matrix<int32_t, Container> mat = tensor.template slice<1, 3>(1, 1); 
    auto mat_t = transpose(mat); 
    REQUIRE(mat_t.rank() == 2); 
    REQUIRE(mat_t.dimension(0) == 8); 
    REQUIRE(mat_t.dimension(1) == 4); 
    for (size_t i = 0; i < mat.dimension(0); ++i) 
      for (size_t j = 0; j < mat.dimension(1); ++j) 
          REQUIRE(mat(i, j) == mat_t(j, i)); 

    Tensor<int32_t, 1, Container> vec = mat.template slice<1>(3); 
    Tensor<int32_t, 2, Container> vec_t = transpose(vec); 
    REQUIRE(vec_t.rank() == 2); 
    REQUIRE(vec_t.dimension(0) == 1); 
    REQUIRE(vec_t.dimension(1) == 8); 
    for (size_t i = 0; i < vec.dimension(0); ++i) 
      REQUIRE(vec(i) == vec_t(0, i)); 
  } 

  SECTION("Map") { 
    auto fn = [](int &x) { x = -x; }; 
    auto tensor_1 = tensor; 
    Map(fn, tensor_1); 
    for (size_t i = 0; i < tensor.dimension(0); ++i) 
      for (size_t j = 0; j < tensor.dimension(1); ++j) 
        for (size_t k = 0; k < tensor.dimension(2); ++k) 
          for (size_t l = 0; l < tensor.dimension(3); ++l) 
            REQUIRE(tensor_1(i, j, k, l) == 
                -1000 * (int)i + -100 * (int)j + -10 * (int)k - (int)l); 

    auto fn2 = [](int &x, int y) { x = y + x; }; 
    Map(fn2, tensor_1, tensor); 
    for (size_t i = 0; i < tensor.dimension(0); ++i) 
      for (size_t j = 0; j < tensor.dimension(1); ++j) 
        for (size_t k = 0; k < tensor.dimension(2); ++k) 
          for (size_t l = 0; l < tensor.dimension(3); ++l) 
            REQUIRE(tensor_1(i, j, k, l) == 0); 

    auto fn3 = [](int &x, int y, int z) { x = 3 * y - z; }; 
    Map(fn3, tensor_1, tensor, tensor); 
    for (size_t i = 0; i < tensor.dimension(0); ++i) 
      for (size_t j = 0; j < tensor.dimension(1); ++j) 
        for (size_t k = 0; k < tensor.dimension(2); ++k) 
          for (size_t l = 0; l < tensor.dimension(3); ++l) 
            REQUIRE(tensor_1(i, j, k, l) == 
                2000 * (int)i + 200 * (int)j + 20 * (int)k + 2 * (int)l); 

    Tensor<int32_t, 3, Container> tensor_3 (tensor_2.shape());
    Map([](int &x, int y) { x = 4 + y; }, 
        tensor_3, tensor_2); 

    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == 5); 
 
    Map([](int &x, int y) { x = 6 - y; },
        tensor_3, tensor_2);
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == 5); 
 
    Map([](int &x, int y) { x = y * -100; },
        tensor_3, tensor_2);
    for (size_t i = 0; i < tensor_3.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_3.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_3.dimension(2); ++k) 
          REQUIRE(tensor_3(i, j, k) == -100); 
 
    Map([](int &x, int y, int z, int a) { x = a * (y + 9)/(y + z); },
        tensor_2, tensor_2, tensor_2, tensor_3);
    for (size_t i = 0; i < tensor_2.dimension(0); ++i) 
      for (size_t j = 0; j < tensor_2.dimension(1); ++j) 
        for (size_t k = 0; k < tensor_2.dimension(2); ++k) 
          REQUIRE(tensor_2(i, j, k) == -500); 
  } 
} 

template <template <class> class Container>
void ConstMethodTests() {
  auto tensor = Tensor<int32_t, 4, Container>{2, 4, 6, 8}; 
  for (size_t i = 0; i < tensor.dimension(0); ++i) 
    for (size_t j = 0; j < tensor.dimension(1); ++j) 
      for (size_t k = 0; k < tensor.dimension(2); ++k) 
        for (size_t l = 0; l < tensor.dimension(3); ++l) 
          tensor(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l; 

  SECTION("resize") { 
    Matrix<int32_t, Container> mat = tensor.template slice<1, 3>(1, 2); 
    Vector<int32_t, Container> vec = mat.template resize<1>(Shape<1>({32})); 
    REQUIRE(vec.rank() == 1); 
    REQUIRE(vec.dimension(0) == 32); 
    int32_t correct_number = 1020; 
    int32_t countdown = 8; 
    for (size_t i = 0; i < vec.dimension(0); ++i) { 
      REQUIRE(correct_number == vec(i)); 
      ++correct_number; 
      --countdown; 
      if (!countdown) { 
        correct_number += 92; 
        countdown = 8; 
      }
    }
  }

  SECTION("reduce") { 
    auto fn = [](size_t accum, size_t x) { 
      return accum + x; 
    }; 
    int sum  = 0; 
    for (size_t i = 0; i < tensor.dimension(0); ++i) 
      for (size_t j = 0; j < tensor.dimension(1); ++j) 
        for (size_t k = 0; k < tensor.dimension(2); ++k) 
          for (size_t l = 0; l < tensor.dimension(3); ++l) 
            sum += 1000 * i + 100 * j + 10 * k + l; 
    REQUIRE(tensor.reduce(0, fn) == sum); 
    REQUIRE(reduce(0, fn, tensor) == sum); 

    auto tensor_2 = tensor; 
    sum *= 2; 
    auto fn2 = [](size_t accum, size_t x, size_t y) { 
      return accum + x + y; 
    }; 
    REQUIRE(reduce(0, fn2, tensor, tensor_2) == sum); 
  } 
}
 
template <template <class> class Container>
void ElementwiseArithmeticTests() {
  auto tensor_1 = Tensor<int32_t, 3, Container>({2, 4, 6}, 5); 
  auto tensor_2 = Tensor<int32_t, 3, Container>({2, 4, 6}, -5); 

  SECTION("Eager Evaluation") {
    auto tensor = elemwise([](int x, int y) { return x + y; },
        tensor_1, tensor_2);
    for (size_t i = 0; i < tensor.dimension(0); ++i)
      for (size_t j = 0; j < tensor.dimension(1); ++j) 
        for (size_t k = 0; k < tensor.dimension(2); ++k)
          REQUIRE(tensor(i, j, k) == 0);
  }
} 

TEST_CASE(BeginTest("Methods", "Array")) { 
  MethodTests<data::Array>();
}

TEST_CASE(BeginTest("Methods", "HashMap")) { 
  MethodTests<data::HashMap>();
}

TEST_CASE(BeginTest("Const Methods", "Array")) { 
  ConstMethodTests<data::Array>(); 
}

TEST_CASE(BeginTest("Const Methods", "HashMap")) { 
  ConstMethodTests<data::HashMap>(); 
}

TEST_CASE(BeginTest("Elementwise Arithmetic", "Array")) { 
  ElementwiseArithmeticTests<data::Array>();
}

TEST_CASE(BeginTest("Elementwise Arithmetic", "HashMap")) { 
  ElementwiseArithmeticTests<data::HashMap>();
}

