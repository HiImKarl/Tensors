#include "test.hh"

using namespace tensor;
using namespace std;

template <template <class> class C1, template <class> class C2>
void ConversionTests() {
  auto matrix_1 = Tensor<uint32_t, 2, C1>{{3, 3}};
  for (size_t i = 0; i < matrix_1.dimension(0); ++i) 
    for (size_t j = 0; j < matrix_1.dimension(1); ++j) 
      matrix_1(i, j) = 10 * i + j;

  auto matrix_2 = Tensor<uint32_t, 2, C2>{matrix_1};
  REQUIRE(matrix_2.dimension(0) == 3);
  REQUIRE(matrix_2.dimension(1) == 3);
  for (size_t i = 0; i < matrix_2.dimension(0); ++i) 
    for (size_t j = 0; j < matrix_2.dimension(1); ++j) 
      REQUIRE(matrix_2(i, j) == 10 * i + j);

  auto signed_matrix_2 = Tensor<int32_t, 2, C2>{matrix_1};
  REQUIRE(signed_matrix_2.dimension(0) == 3);
  REQUIRE(signed_matrix_2.dimension(1) == 3);
  for (size_t i = 0; i < signed_matrix_2.dimension(0); ++i) 
    for (size_t j = 0; j < signed_matrix_2.dimension(1); ++j) 
      REQUIRE(signed_matrix_2(i, j) == (int)(10 * i + j));
}

TEST_CASE("Conversions") {
  SECTION("Matrix to Sparse Matrix") {
    ConversionTests<data::Array, data::HashMap>();
  }
  SECTION("Sparse Matrix to Matrix") {
    ConversionTests<data::Array, data::HashMap>();
  }
}
