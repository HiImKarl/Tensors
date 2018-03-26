#include "tensor.hh"
#include "catch.hh"

using namespace tensor;

TEST_CASE("Slicing Tensors", "[pod]") {
  Tensor<int32_t, 3> tensor_1({2, 4, 6});
  tensor_1.slice<1, 2, 3>();
}
