#include <tensor.hh>
#include <catch.hh>

using namespace tensor;

#define CONTAINER data::Array

TEST_CASE(#CONTAINER " Miscillaneous") { 
  auto tensor = Tensor<int32_t, 4, CONTAINER<int32_t>>{2, 4, 6, 8}; 
  for (size_t i = 0; i < tensor.dimension(0); ++i) 
    for (size_t j = 0; j < tensor.dimension(1); ++j) 
      for (size_t k = 0; k < tensor.dimension(2); ++k) 
        for (size_t l = 0; l < tensor.dimension(3); ++l) 
          tensor(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l; 
 
  SECTION("Tranpose") { 
    Tensor<int32_t, 2, CONTAINER<int32_t>> mat = tensor.slice<1, 3>(1, 1); 
    auto mat_t = transpose(mat); 
    REQUIRE(mat_t.rank() == 2); 
    REQUIRE(mat_t.dimension(0) == 8); 
    REQUIRE(mat_t.dimension(1) == 4); 
    for (size_t i = 0; i < mat.dimension(0); ++i) 
      for (size_t j = 0; j < mat.dimension(1); ++j) 
          REQUIRE(mat(i, j) == mat_t(j, i)); 
 
    Tensor<int32_t, 1, CONTAINER<int32_t>> vec = mat.slice<1>(3); 
    Tensor<int32_t, 2, CONTAINER<int32_t>> vec_t = transpose(vec); 
    REQUIRE(vec_t.rank() == 2); 
    REQUIRE(vec_t.dimension(0) == 1); 
    REQUIRE(vec_t.dimension(1) == 8); 
    for (size_t i = 0; i < vec.dimension(0); ++i) 
      REQUIRE(vec(i) == vec_t(0, i)); 
  } 
 
  SECTION("Resize") { 
    Tensor<int32_t, 2, CONTAINER<int32_t>> mat = tensor.slice<1, 3>(1, 2); 
    Tensor<int32_t, 1, CONTAINER<int32_t>> vec = mat.resize(Shape<1>({32})); 
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
}

