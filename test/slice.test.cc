#include "tensor.hh"
#include "catch.hh"

using namespace tensor;

TEST_CASE("Slicing Tensors", "[pod]") {
  Tensor<int32_t, 3> tensor_1({2, 4, 6});
  for (size_t i = 1; i <= tensor_1.dimension(1); ++i) 
    for (size_t j = 1; j <= tensor_1.dimension(2); ++j) 
      for (size_t k = 1; k <= tensor_1.dimension(3); ++k) 
        tensor_1(i, j, k)  = i * 100 + j * 10 + k;
}
