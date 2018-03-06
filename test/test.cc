#include "tensor.hh"

#define CATCH_CONFIG_MAIN
#include "catch.hh"

using namespace tensor;
using namespace std;

// Initialize a tensor 
auto t1 = Tensor<int32_t, 4>({1, 2, 3, 4});

// Initialize a scalar tensor;
Tensor<int32_t> st1{};


TEST_CASE("Intializing Tensors ") {
  // Initialize values
  for (uint32_t i = 1; i <= t1.dimension(1); ++i)
    for (uint32_t j = 1; j <= t1.dimension(2); ++j)
      for (uint32_t k = 1; k <= t1.dimension(3); ++k)
        for (uint32_t l = 1; l <= t1.dimension(4); ++l)
          t1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;

  st1 = 0;

  SECTION("Rank and Dimensions") {
    REQUIRE(t1.rank() == 4);
    for (int i = 1; i <= 4; ++i)
      REQUIRE(t1.dimension(i) == i);
    REQUIRE(st1.rank() == 0);
  }

  SECTION("Initializing Values") {
    for (uint32_t i = 1; i <= t1.dimension(1); ++i)
      for (uint32_t j = 1; j <= t1.dimension(2); ++j)
        for (uint32_t k = 1; k <= t1.dimension(3); ++k)
          for (uint32_t l = 1; l <= t1.dimension(4); ++l)
            REQUIRE(t1(i, j, k, l) == (int)(1000 * i + 100 * j + 10 * k + l));
    REQUIRE(st1 == 0);
  }

  Tensor<int32_t> st2(0);
  SECTION("Initialize and assign a scalar tensor") {
    REQUIRE(st2 == 0);
  }
}


TEST_CASE("Tensor Assignment") {
  SECTION("Assigning Tensors to Tensors") {
    Tensor<int32_t, 2> t2({3, 4});
    for (uint32_t i = 1; i <= t2.dimension(1); ++i)
      for (uint32_t j = 1; j <= t2.dimension(2); ++j)
        t2(i, j) = t1(1, 1, i, j) - 1000;

    t1(1, 1) = t2;

    for (uint32_t i = 1; i <= t1.dimension(3); ++i)
      for (uint32_t j = 1; j <= t1.dimension(4); ++j)
        REQUIRE(t1(1, 1, i, j) == (int)(100 + 10 * i + j));


    for (uint32_t i = 1; i <= t1.dimension(3); ++i)
      for (uint32_t j = 1; j <= t1.dimension(4); ++j)
        REQUIRE(t1(1, 2, i, j) == (int)(1000 + 200 + 10 * i + j));

    Tensor<int32_t, 4> t3({1, 2, 3, 4});
    for (uint32_t i = 1; i <= t3.dimension(1); ++i)
      for (uint32_t j = 1; j <= t3.dimension(2); ++j)
        for (uint32_t k = 1; k <= t3.dimension(3); ++k)
          for (uint32_t l = 1; l <= t3.dimension(4); ++l)
            t3(i, j, k, l) = -1 * (int)(i + 10 * j + 100 * k + 1000 * l);
    
    t1 = t3;
    for (uint32_t i = 1; i <= t3.dimension(1); ++i)
      for (uint32_t j = 1; j <= t3.dimension(2); ++j)
        for (uint32_t k = 1; k <= t3.dimension(3); ++k)
          for (uint32_t l = 1; l <= t3.dimension(4); ++l)
            REQUIRE(t1(i, j, k, l) == -1 * (int)(i + 10 * j + 100 * k + 1000 * l));
  }

  SECTION("Assigning Scalar Values to Tensors") {
    int32_t VALUE = -1200;
    t1(1, 1, 1, 1) = VALUE;
    REQUIRE(t1(1, 1, 1, 1) == VALUE);
    VALUE = -1500;
    t1(t1.dimension(1), t1.dimension(2), t1.dimension(3), t1.dimension(4)) = VALUE;
    REQUIRE(
        t1(t1.dimension(1), t1.dimension(2), t1.dimension(3), t1.dimension(4)) == VALUE
    );
  }

  SECTION("Assigning Scalar Tensors to Tensors") {
    t1(1, 1, 1, 1) = st1;
    REQUIRE(t1(1, 1, 1, 1) == 0);
    REQUIRE(t1(1, 1, 1, 1) == st1);
  }

  SECTION("Assigning Scalar Values to Scalar Tensors") {
    int32_t VALUE = -1200;
    st1 = VALUE;
    REQUIRE(st1 == VALUE);
  }

  // Restore tensor values
  for (uint32_t i = 1; i <= t1.dimension(1); ++i)
    for (uint32_t j = 1; j <= t1.dimension(2); ++j)
      for (uint32_t k = 1; k <= t1.dimension(3); ++k)
        for (uint32_t l = 1; l <= t1.dimension(4); ++l)
          t1(i, j, k, l) = (int)(1000 * i + 100 * j + 10 * k + l);
  st1 = 0;
}

