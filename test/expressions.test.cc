#include <tensor.hh>
#include <catch.hh>

using namespace tensor;

TEST_CASE("Basic Tensor Arithmetic, Non-scalar") {
  auto tensor_1 = Tensor<int32_t, 3>({2, 3, 4});
  auto tensor_2 = Tensor<int32_t, 3>({2, 3, 4});
  for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i) 
    for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j) 
      for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k) 
        tensor_1(i, j, k) = 100000 * i + 10000 * j + 1000 * k;

  for (uint32_t i = 1; i <= tensor_2.dimension(1); ++i) 
    for (uint32_t j = 1; j <= tensor_2.dimension(2); ++j) 
      for (uint32_t k = 1; k <= tensor_2.dimension(3); ++k) 
        tensor_2(i, j, k) = 100 * i + 10 * j + 1 * k;


  SECTION("Two Term Addition/Subtract") {
    Tensor<int32_t, 3> tensor_3 = tensor_1 + tensor_2; 
    Tensor<int32_t, 3> tensor_4 = tensor_3 - tensor_1;

    for (uint32_t i = 1; i <= tensor_3.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_3.dimension(2); ++j) 
        for (uint32_t k = 1; k <= tensor_3.dimension(3); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(100100 * i + 10010 * j + 1001 * k));

    for (uint32_t i = 1; i <= tensor_4.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_4.dimension(2); ++j) 
        for (uint32_t k = 1; k <= tensor_4.dimension(3); ++k) 
          REQUIRE(tensor_4(i, j, k) == (int)(100 * i + 10 * j + 1 * k));
  }

  SECTION("Multi-Term Addition/Subtract") {
    Tensor<int32_t, 3> tensor_3 = tensor_2 + tensor_1 + tensor_2;

    for (uint32_t i = 1; i <= tensor_3.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_3.dimension(2); ++j) 
        for (uint32_t k = 1; k <= tensor_3.dimension(3); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(100200 * i + 10020 * j + 1002 * k));

    Tensor<int32_t, 3> tensor_4 = tensor_3 - tensor_2 + tensor_1;

    for (uint32_t i = 1; i <= tensor_4.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_4.dimension(2); ++j) 
        for (uint32_t k = 1; k <= tensor_4.dimension(3); ++k) 
          REQUIRE(tensor_4(i, j, k) == (int)(200100 * i + 20010 * j + 2001 * k));
  }

  SECTION("Negation") {
    auto tensor_3 = -tensor_1;
    for (uint32_t i = 1; i <= tensor_3.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_3.dimension(2); ++j) 
        for (uint32_t k = 1; k <= tensor_3.dimension(3); ++k) 
          REQUIRE(tensor_3(i, j, k) == (int)(-100000 * i + -10000 * j + -1000 * k));
  }
}


TEST_CASE("Basic Tensor Arithmetic, Scalar") {
  auto tensor_1 = Tensor<int32_t>(10);
  auto tensor_2 = Tensor<int32_t>(-10); 

  SECTION("Binary Addition/Subtract") {
    REQUIRE(tensor_1() + tensor_2() == 0);
    REQUIRE((tensor_1 + tensor_2)() == 0);
    REQUIRE(tensor_1() - tensor_2() == 20);
    REQUIRE((tensor_1 - tensor_2)() == 20);
    REQUIRE(tensor_1 + 10 == 20);
    REQUIRE(10 + tensor_1 == 20);
    REQUIRE(tensor_1 - 10 == 0);
    REQUIRE(10 - tensor_1 == 0);
  }

  SECTION("Multi-Term Addition/Subtract") {
    REQUIRE(tensor_1 + 10 + tensor_2 + 5 == 15);
    REQUIRE(tensor_1 - 10 - tensor_2 - 5 == 5);
    REQUIRE(tensor_1 - 10 + tensor_2 - 5 == -15);
    REQUIRE(tensor_1 + 10 - tensor_2 + 5 == 35);
  }

  SECTION("Negation") {
    REQUIRE(-tensor_1 == -10);
    REQUIRE(-tensor_1 - 10 + -tensor_2 - 5 == -15);
  }
}

TEST_CASE("Tensor Multplication, Scalar") {
  auto tensor_1 = Tensor<int32_t, 3>({2, 3, 4});
  auto tensor_2 = Tensor<int32_t, 3>({4, 3, 2});
  for (uint32_t i = 1; i <= tensor_1.dimension(1); ++i) 
    for (uint32_t j = 1; j <= tensor_1.dimension(2); ++j) 
      for (uint32_t k = 1; k <= tensor_1.dimension(3); ++k) 
        tensor_1(i, j, k) = 1;

  for (uint32_t i = 1; i <= tensor_2.dimension(1); ++i) 
    for (uint32_t j = 1; j <= tensor_2.dimension(2); ++j) 
      for (uint32_t k = 1; k <= tensor_2.dimension(3); ++k) 
        tensor_2(i, j, k) = 1;

  SECTION("Binary Multiplication") {
    Tensor<int32_t, 4> tensor_3 = tensor_1 * tensor_2;
    REQUIRE(tensor_3.rank() == 4);
    REQUIRE(tensor_3.dimension(1) == 2);
    REQUIRE(tensor_3.dimension(2) == 3);
    REQUIRE(tensor_3.dimension(3) == 3);
    REQUIRE(tensor_3.dimension(4) == 2);
    for (uint32_t i = 1; i <= tensor_3.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_3.dimension(2); ++j) 
        for (uint32_t k = 1; k <= tensor_3.dimension(3); ++k) 
          for (uint32_t l = 1; l <= tensor_3.dimension(4); ++l) 
            REQUIRE(tensor_3(i, j, k, l) == 4);

    Tensor<int32_t, 2> tensor_4 = tensor_1.slice<1, 2>(1) * tensor_2(1);
    REQUIRE(tensor_4.rank() == 2);
    REQUIRE(tensor_4.dimension(1) == 2);
    REQUIRE(tensor_4.dimension(2) == 2);
    for (uint32_t i = 1; i <= tensor_4.dimension(1); ++i) 
      for (uint32_t j = 1; j <= tensor_4.dimension(2); ++j) 
        REQUIRE(tensor_4(i, j) == 3);

    Tensor<int32_t, 4> tensor_5 = tensor_4 * tensor_3 * tensor_4;
    REQUIRE(tensor_5.rank() == 4);
    REQUIRE(tensor_5.dimension(1) == 2);
    REQUIRE(tensor_5.dimension(2) == 3);
    REQUIRE(tensor_5.dimension(3) == 3);
    REQUIRE(tensor_5.dimension(4) == 2);
  }
}
