#include <tensor.hh>
#include <catch.hh>

using namespace tensor;

#define TEST_CASES(CONTAINER) \
  TEST_CASE(#CONTAINER ": methods") { \
    auto tensor = Tensor<int32_t, 4, CONTAINER>{2, 4, 6, 8}; \
    for (size_t i = 0; i < tensor.dimension(0); ++i) \
      for (size_t j = 0; j < tensor.dimension(1); ++j) \
        for (size_t k = 0; k < tensor.dimension(2); ++k) \
          for (size_t l = 0; l < tensor.dimension(3); ++l) \
            tensor(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l; \
 \
    SECTION("Tranpose") { \
      Matrix<int32_t, CONTAINER> mat = tensor.slice<1, 3>(1, 1); \
      auto mat_t = transpose(mat); \
      REQUIRE(mat_t.rank() == 2); \
      REQUIRE(mat_t.dimension(0) == 8); \
      REQUIRE(mat_t.dimension(1) == 4); \
      for (size_t i = 0; i < mat.dimension(0); ++i) \
        for (size_t j = 0; j < mat.dimension(1); ++j) \
            REQUIRE(mat(i, j) == mat_t(j, i)); \
 \
      Tensor<int32_t, 1, CONTAINER> vec = mat.slice<1>(3); \
      Tensor<int32_t, 2, CONTAINER> vec_t = transpose(vec); \
      REQUIRE(vec_t.rank() == 2); \
      REQUIRE(vec_t.dimension(0) == 1); \
      REQUIRE(vec_t.dimension(1) == 8); \
      for (size_t i = 0; i < vec.dimension(0); ++i) \
        REQUIRE(vec(i) == vec_t(0, i)); \
    } \
 \
    SECTION("Map") { \
      auto fn = [](int &x) { x = -x; }; \
      auto tensor_1 = tensor; \
      Map(tensor_1, fn); \
      for (size_t i = 0; i < tensor.dimension(0); ++i) \
        for (size_t j = 0; j < tensor.dimension(1); ++j) \
          for (size_t k = 0; k < tensor.dimension(2); ++k) \
            for (size_t l = 0; l < tensor.dimension(3); ++l) \
              REQUIRE(tensor_1(i, j, k, l) == \
                  -1000 * (int)i + -100 * (int)j + -10 * (int)k - (int)l); \
 \
      auto fn2 = [](int &x, int y) { x = y + x; }; \
      Map(tensor_1, tensor, fn2); \
      for (size_t i = 0; i < tensor.dimension(0); ++i) \
        for (size_t j = 0; j < tensor.dimension(1); ++j) \
          for (size_t k = 0; k < tensor.dimension(2); ++k) \
            for (size_t l = 0; l < tensor.dimension(3); ++l) \
              REQUIRE(tensor_1(i, j, k, l) == 0); \
 \
      auto fn3 = [](int &x, int y, int z) { x = 3 * y - z; }; \
      Map(tensor_1, tensor, tensor, fn3); \
      for (size_t i = 0; i < tensor.dimension(0); ++i) \
        for (size_t j = 0; j < tensor.dimension(1); ++j) \
          for (size_t k = 0; k < tensor.dimension(2); ++k) \
            for (size_t l = 0; l < tensor.dimension(3); ++l) \
              REQUIRE(tensor_1(i, j, k, l) == \
                  2000 * (int)i + 200 * (int)j + 20 * (int)k + 2 * (int)l); \
    } \
  } \
 \
  TEST_CASE(#CONTAINER ": const methods") { \
    auto tensor = Tensor<int32_t, 4, CONTAINER>{2, 4, 6, 8}; \
    for (size_t i = 0; i < tensor.dimension(0); ++i) \
      for (size_t j = 0; j < tensor.dimension(1); ++j) \
        for (size_t k = 0; k < tensor.dimension(2); ++k) \
          for (size_t l = 0; l < tensor.dimension(3); ++l) \
            tensor(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l; \
 \
    SECTION("resize") { \
      Matrix<int32_t, CONTAINER> mat = tensor.slice<1, 3>(1, 2); \
      Vector<int32_t, CONTAINER> vec = mat.resize(Shape<1>({32})); \
      REQUIRE(vec.rank() == 1); \
      REQUIRE(vec.dimension(0) == 32); \
      int32_t correct_number = 1020; \
      int32_t countdown = 8; \
      for (size_t i = 0; i < vec.dimension(0); ++i) { \
        REQUIRE(correct_number == vec(i)); \
        ++correct_number; \
        --countdown; \
        if (!countdown) { \
          correct_number += 92; \
          countdown = 8; \
        } \
      } \
    } \
 \
    SECTION("reduce") { \
      auto fn = [](size_t accum, size_t x) { \
        return accum + x; \
      }; \
      size_t sum  = 0; \
      for (size_t i = 0; i < tensor.dimension(0); ++i) \
        for (size_t j = 0; j < tensor.dimension(1); ++j) \
          for (size_t k = 0; k < tensor.dimension(2); ++k) \
            for (size_t l = 0; l < tensor.dimension(3); ++l) \
              sum += 1000 * i + 100 * j + 10 * k + l; \
      REQUIRE(tensor.reduce(fn, 0) == sum); \
 \
      auto tensor_2 = tensor; \
      sum *= 2; \
      auto fn2 = [](size_t accum, size_t x, size_t y) { \
        return accum + x + y; \
      }; \
      REQUIRE(tensor.reduce(tensor_2, fn2, 0) == sum); \
    } \
  }

// Instantiate Tests
TEST_CASES(data::Array);
TEST_CASES(data::HashMap);

