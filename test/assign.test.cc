#include <tensor.hh>
#include "catch.hh"

using namespace tensor;

TEST_CASE("Tensor Assignment", "[int]") {

  /*     -- Tensor Initialization --   */
  auto tensor_1 = Tensor<int32_t, 4>({1, 2, 3, 4});

  // Initialize values
  for (size_t i = 1; i <= tensor_1.dimension(1); ++i)
    for (size_t j = 1; j <= tensor_1.dimension(2); ++j)
      for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
        for (size_t l = 1; l <= tensor_1.dimension(4); ++l) 
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;

  /*     --------------------------   */

  SECTION("Assigning Tensors to Tensors") {

    Tensor<int32_t, 2> t2({3, 4});
    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j) 
        t2(i, j) = tensor_1(1, 1, i, j) - 1000;

    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j) 
        REQUIRE(t2(i, j) == (int)(100 + 10 * i + j));

    tensor_1(1, 1) = t2;

    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 1, i, j) == (int)(100 + 10 * i + j));

    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 2, i, j) == (int)(1000 + 200 + 10 * i + j));

    Tensor<double, 4> t3({1, 2, 3, 4});
    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            t3(i, j, k, l) = -1 * (int)(i + 10 * j + 100 * k + 1000 * l);

    tensor_1 = t3;

    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            REQUIRE(tensor_1(i, j, k, l) == -1 * (int)(i + 10 * j + 100 * k + 1000 * l));
  }

  SECTION("Assigning Integral Different-Typed Tensors to Tensors") {
    Tensor<long double, 2> t2({3, 4});
    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j)
        t2(i, j) = tensor_1(1, 1, i, j) - 1000;

    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j)
        REQUIRE(t2(i, j) == tensor_1(1, 1, i, j) - 1000);

    tensor_1(1, 1) = t2;

    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 1, i, j) == (int)(100 + 10 * i + j));


    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 2, i, j) == (int)(1000 + 200 + 10 * i + j));

    Tensor<int64_t, 4> t3({1, 2, 3, 4});
    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            t3(i, j, k, l) = -1 * (int)(i + 10 * j + 100 * k + 1000 * l);

    tensor_1 = t3;
    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            REQUIRE(tensor_1(i, j, k, l) == -1 * (int)(i + 10 * j + 100 * k + 1000 * l));
  }

  SECTION("Assigning Tensors to Reference Tensors") {

    Tensor<int32_t, 4> tensor_1_ref = tensor_1.ref();
    Tensor<int32_t, 2> t2({3, 4});
    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j) 
        t2(i, j) = tensor_1(1, 1, i, j) - 1000;

    tensor_1_ref(1, 1) = t2;

    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 1, i, j) == (int)(100 + 10 * i + j));

    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 2, i, j) == (int)(1000 + 200 + 10 * i + j));

    Tensor<double, 4> t3({1, 2, 3, 4});
    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            t3(i, j, k, l) = -1 * (int)(i + 10 * j + 100 * k + 1000 * l);

    tensor_1_ref  = t3;

    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            REQUIRE(tensor_1(i, j, k, l) == -1 * (int)(i + 10 * j + 100 * k + 1000 * l));
  }

  SECTION("Assigning Scalar Values to Tensors") {
    int32_t VALUE = -1200;
    tensor_1(1, 1, 1, 1) = VALUE;
    REQUIRE(tensor_1(1, 1, 1, 1) == -1200);
    VALUE = -1500;
    tensor_1(tensor_1.dimension(1), tensor_1.dimension(2), tensor_1.dimension(3), tensor_1.dimension(4)) = VALUE;
    REQUIRE(
        tensor_1(tensor_1.dimension(1), tensor_1.dimension(2), tensor_1.dimension(3), tensor_1.dimension(4)) == VALUE
    );
  }

  SECTION("Assigning Scalar Integral Different-Type Values to Tensors") {
    long double VALUE = -1200;
    tensor_1(1, 1, 1, 1) = VALUE;
    REQUIRE(tensor_1(1, 1, 1, 1) == -1200);
    VALUE = -1500;
    tensor_1(tensor_1.dimension(1), tensor_1.dimension(2), tensor_1.dimension(3), tensor_1.dimension(4)) = VALUE;
    REQUIRE(
        tensor_1(tensor_1.dimension(1), tensor_1.dimension(2), tensor_1.dimension(3), tensor_1.dimension(4)) == VALUE
    );
  }

  SECTION("Assigning Scalar Tensors to Tensors") {
    /* same type */
    Tensor<int32_t> scalar_1(0);
    tensor_1(1, 1, 1, 1) = scalar_1;
    REQUIRE(tensor_1(1, 1, 1, 1) == 0);
    REQUIRE(tensor_1(1, 1, 1, 1) == scalar_1);

    /* different type */
    Tensor<double> scalar_2(-12);
    tensor_1(1, 1, 1, 1) = scalar_2;
    REQUIRE(tensor_1(1, 1, 1, 1) == -12);
    REQUIRE(tensor_1(1, 1, 1, 1) == scalar_2);
  }

  SECTION("Assigning Tensors to Lvalues") {
    /* same type */
    int32_t VALUE = -tensor_1(1, 1, 1, 1);
    REQUIRE(VALUE == -1111);

    /* different_type */
    float fVALUE = -tensor_1(1,1,1,1);
    REQUIRE(fVALUE == -1111);
  }
}

TEST_CASE("Scalar Assignment", "[int]") {

  /*     -- Tensor Initialization --   */
  auto tensor_1 = Tensor<int32_t, 4>({1, 2, 3, 4});

  // Initialize a scalar tensor;
  Tensor<int32_t> stensor_1{};

  // Initialize values
  for (size_t i = 1; i <= tensor_1.dimension(1); ++i)
    for (size_t j = 1; j <= tensor_1.dimension(2); ++j)
      for (size_t k = 1; k <= tensor_1.dimension(3); ++k)
        for (size_t l = 1; l <= tensor_1.dimension(4); ++l) {
          tensor_1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;
        }

  stensor_1 = 0;

  /*     --------------------------   */

  SECTION("Assigning Tensors to Tensors") {
    Tensor<int32_t, 2> t2({3, 4});
    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j) 
        t2(i, j) = tensor_1(1, 1, i, j) - 1000;

    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j) 
        REQUIRE(t2(i, j) == (int)(100 + 10 * i + j));

    tensor_1(1, 1) = t2;

    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 1, i, j) == (int)(100 + 10 * i + j));


    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 2, i, j) == (int)(1000 + 200 + 10 * i + j));

    Tensor<int32_t, 4> t3({1, 2, 3, 4});
    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            t3(i, j, k, l) = -1 * (int)(i + 10 * j + 100 * k + 1000 * l);

    tensor_1 = t3;
    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            REQUIRE(tensor_1(i, j, k, l) == -1 * (int)(i + 10 * j + 100 * k + 1000 * l));
  }

  SECTION("Assigning Integral Different-Typed Tensors to Tensors") {
    Tensor<long double, 2> t2({3, 4});
    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j)
        t2(i, j) = tensor_1(1, 1, i, j) - 1000;

    for (size_t i = 1; i <= t2.dimension(1); ++i)
      for (size_t j = 1; j <= t2.dimension(2); ++j)
        REQUIRE(t2(i, j) == tensor_1(1, 1, i, j) - 1000);

    tensor_1(1, 1) = t2;

    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 1, i, j) == (int)(100 + 10 * i + j));


    for (size_t i = 1; i <= tensor_1.dimension(3); ++i)
      for (size_t j = 1; j <= tensor_1.dimension(4); ++j)
        REQUIRE(tensor_1(1, 2, i, j) == (int)(1000 + 200 + 10 * i + j));

    Tensor<int64_t, 4> t3({1, 2, 3, 4});
    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            t3(i, j, k, l) = -1 * (int)(i + 10 * j + 100 * k + 1000 * l);

    tensor_1 = t3;
    for (size_t i = 1; i <= t3.dimension(1); ++i)
      for (size_t j = 1; j <= t3.dimension(2); ++j)
        for (size_t k = 1; k <= t3.dimension(3); ++k)
          for (size_t l = 1; l <= t3.dimension(4); ++l)
            REQUIRE(tensor_1(i, j, k, l) == -1 * (int)(i + 10 * j + 100 * k + 1000 * l));
  }

  SECTION("Assigning Scalar Values to Tensors") {
    int32_t VALUE = -1200;
    tensor_1(1, 1, 1, 1) = VALUE;
    REQUIRE(tensor_1(1, 1, 1, 1) == -1200);
    VALUE = -1500;
    tensor_1(tensor_1.dimension(1), tensor_1.dimension(2), tensor_1.dimension(3), tensor_1.dimension(4)) = VALUE;
    REQUIRE(
        tensor_1(tensor_1.dimension(1), tensor_1.dimension(2), tensor_1.dimension(3), tensor_1.dimension(4)) == VALUE
    );
  }

  SECTION("Assigning Scalar Integral Different-Type Values to Tensors") {
    long double VALUE = -1200;
    tensor_1(1, 1, 1, 1) = VALUE;
    REQUIRE(tensor_1(1, 1, 1, 1) == -1200);
    VALUE = -1500;
    tensor_1(tensor_1.dimension(1), tensor_1.dimension(2), tensor_1.dimension(3), tensor_1.dimension(4)) = VALUE;
    REQUIRE(
        tensor_1(tensor_1.dimension(1), tensor_1.dimension(2), tensor_1.dimension(3), tensor_1.dimension(4)) == VALUE
    );
  }

  SECTION("Assigning Scalar Tensors to Tensors") {
    tensor_1(1, 1, 1, 1) = stensor_1;
    REQUIRE(tensor_1(1, 1, 1, 1) == 0);
    REQUIRE(tensor_1(1, 1, 1, 1) == stensor_1);
  }

  SECTION("Assigning Scalar Values to Scalar Tensors") {
    int32_t VALUE = -1200;
    stensor_1 = VALUE;
    REQUIRE(stensor_1 == VALUE);
  }

  SECTION("Assigning Tensors to Lvalues") {
    int32_t VALUE = -tensor_1(1, 1, 1, 1);
    REQUIRE(VALUE == -1111);
  }

  SECTION("Assigning Scalar Tensors to Lvalues") {
    int32_t VALUE = stensor_1;
    REQUIRE(VALUE == 0);
  }
}

