#include <tensor.hh>
#include <catch.hh>
#include "test.hh"

using namespace tensor;

void ShapeTests() {
  auto shape = Shape<8>({1, 2, 3, 4, 5, 6, 7, 8});
  auto scalar_shape = Shape<0>();
  
  SECTION("Rank and Dimensions") {
    REQUIRE(shape.rank() == 8);
    for (size_t i = 0; i < 8; ++i)
      REQUIRE(shape[i] == i + 1);
  }

  SECTION("Tensor Constructor") {
    Tensor<int, 8> tensor(shape);
    REQUIRE(tensor.rank() == 8);
    for (size_t i = 0; i < 8; ++i)
      REQUIRE(tensor.dimension(i) == i + 1);
  }

  SECTION("Scalar Constructor") {
    auto tensor = Scalar<int>(scalar_shape);
    REQUIRE(tensor.rank() == 0);
    REQUIRE(tensor() == 0);
  }
}

void TensorShapeTests() {
  Tensor<int, 3> tensor_1{2, 3, 4};
  Tensor<int, 2> tensor_2{4, 2};
  Tensor<int, 3> tensor_3{2, 3, 4};
  Scalar<int> scalar{};

  SECTION("Tensors") {
    Shape<3> shape_1 = tensor_1.shape();
    Shape<2> shape_2 = tensor_2.shape();
    REQUIRE(shape_1.rank() == 3);
    REQUIRE(shape_1[0] == 2);
    REQUIRE(shape_1[1] == 3);
    REQUIRE(shape_1[2] == 4);
    REQUIRE(shape_2.rank() == 2);
    REQUIRE(shape_2[0] == 4);
    REQUIRE(shape_2[1] == 2);
  }

  SECTION("Scalars") {
    Shape<0> shape = scalar.shape();
    REQUIRE(shape.rank() == 0);
  }

  SECTION("Tensor Addition/Subtraction Expression") {
    Shape<3> shape = 
      (tensor_1 + tensor_2 - tensor_2 - tensor_1 + tensor_2).shape();
    REQUIRE(shape.rank() == 3);
    REQUIRE(shape[0] == 2);
    REQUIRE(shape[1] == 3);
    REQUIRE(shape[2] == 4);
  }

  SECTION("Tensor Multiplication Expression") {
    Shape<4> shape = (tensor_1 * tensor_2 * tensor_3).shape();
    REQUIRE(shape.rank() == 4);
    REQUIRE(shape[0] == 2);
    REQUIRE(shape[1] == 3);
    REQUIRE(shape[2] == 3);
    REQUIRE(shape[3] == 4);
  }
}

TEST_CASE("Shapes") {
  ShapeTests();
}

TEST_CASE("Tensor Shapes") {
  TensorShapeTests();
}
