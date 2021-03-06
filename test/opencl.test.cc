#ifdef _ENABLE_OPENCL
#include <CL/cl2.hpp>
#include "test.hh"

using namespace tensor;

template <template <class> class Container>
void SimpleExpressionTests() {
  Matrix<float, Container> mat1({1111, 100}, 1); 
  Matrix<float, Container> mat2({1111, 100}, -10); 

  Matrix<float, Container> mat3({3, 5}, 2);
  Matrix<float, Container> mat4({5, 3}, 2);

  SECTION("Basic Arithmetic") {
    Matrix<float, Container> result = (mat1 + mat2 + mat2).opencl();
    for (size_t i = 0; i < result.dimension(0); ++i)
      for (size_t j = 0; j < result.dimension(1); ++j)
        REQUIRE(result(i, j) == -19);

    result = (mat1 - mat2 - mat2).opencl();
    for (size_t i = 0; i < result.dimension(0); ++i)
      for (size_t j = 0; j < result.dimension(1); ++j)
        REQUIRE(result(i, j) == 21);

    result = (mat2 % mat2 - mat1).opencl();
    for (size_t i = 0; i < result.dimension(0); ++i)
      for (size_t j = 0; j < result.dimension(1); ++j)
        REQUIRE(result(i, j) == 99);
  }

  SECTION("Map") {
    Matrix<float, Container> result = _map_(math::mul{}, -mat1, mat2, mat2).opencl();
    for (size_t i = 0; i < result.dimension(0); ++i)
      for (size_t j = 0; j < result.dimension(1); ++j)
        REQUIRE(result(i, j) == -100);
  }

  SECTION("Reduce") {
    Scalar<float, Container> result = _reduce_(0.0f, math::add{}, -mat1).opencl();
    REQUIRE(result() == -111100);
    result = _reduce_(0.0f, math::add{}, mat2).opencl();
    REQUIRE(result() == -1111000);
  }

  SECTION("Multiplication") {
    Matrix<float, Container> result1 = (mat3 * mat4).opencl();
    Matrix<float, Container> result2 = _mul_<0, 0>(mat1, mat2).opencl();
    Matrix<float, Container> result3 = _mul_<1, 1>(mat2, mat1).opencl();

    for (size_t i = 0; i < result1.dimension(0); ++i)
      for (size_t j = 0; j < result1.dimension(1); ++j)
        REQUIRE(result1(i, j) == 20);

    for (size_t i = 0; i < result2.dimension(0); ++i)
      for (size_t j = 0; j < result2.dimension(1); ++j)
        REQUIRE(result2(i, j) == 1111 * -10);

    for (size_t i = 0; i < result3.dimension(0); ++i)
      for (size_t j = 0; j < result3.dimension(1); ++j)
        REQUIRE(result3(i, j) == 100 * -10);
  }
}

template <template <class> class Container>
void CombinationExpressionTests() {
  Matrix<float, Container> mat1({1000, 1000}, -1); 
  Matrix<float, Container> mat2({1000, 1000}, 10); 

  SECTION("Map") {
    Matrix<float, Container> result = _map_(math::div{}, mat1 + mat1, mat2 % mat1).opencl();
    for (size_t i = 0; i < result.dimension(0); ++i)
      for (size_t j = 0; j < result.dimension(1); ++j)
        REQUIRE(result(i, j) == 0.2f);
  }

  SECTION("Reduce") {
    Scalar<float, Container> result = _reduce_(1e6f, math::sub{}, mat1 + mat1 % mat2).opencl();
    REQUIRE(result() == 12e6f);
  }

  SECTION("Multipliation") {
    Matrix<float, Container> result = _mul_<0, 1>(-mat1 - mat1, -(mat1 + mat2)).opencl();
    for (size_t i = 0; i < result.dimension(0); ++i)
      for (size_t j = 0; j < result.dimension(1); ++j)
        REQUIRE(result(i, j) == -18000); 
  }
}

// Instantiate test cases
TEST_CASE("Simple Expressions") { 
  SimpleExpressionTests<data::Array>();
  SimpleExpressionTests<data::HashMap>();
}

TEST_CASE("Combination Expressions") { 
  CombinationExpressionTests<data::Array>();
  CombinationExpressionTests<data::HashMap>();
}

#endif // ifdef _ENABLE_OPENCL
