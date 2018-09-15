#include <tensor.hh>
#include <catch.hh>
#include "test.hh"

using namespace tensor;
using namespace std;

// FIXME -- add more tests

template <template <class> class C1, template <class> class C2>
void CrossContainerOperations()
{
  Tensor<int, 3, C1>      tensor_1{2, 4, 2};
  Tensor<double, 3, C2>   tensor_2{2, 4, 2};

  for (size_t i = 0; i < tensor_1.dimension(0); ++i)
    for (size_t j = 0; j < tensor_1.dimension(1); ++j)
      for (size_t k = 0; k < tensor_1.dimension(2); ++k)
        tensor_1(i, j, k) = (int)(100 * i + 10 * j + k); 

  for (size_t i = 0; i < tensor_2.dimension(0); ++i)
    for (size_t j = 0; j < tensor_2.dimension(1); ++j)
      for (size_t k = 0; k < tensor_2.dimension(2); ++k)
        tensor_2(i, j, k) = (double)(i + 10 * j + 100 * k); 

  SECTION("Addition/Subtraction") {
    Tensor<int, 3, C1> tensor_3 = tensor_1 + tensor_2 + tensor_1;

    REQUIRE(tensor_3.rank() == 3);
    REQUIRE(tensor_3.dimension(0) == 2);
    REQUIRE(tensor_3.dimension(1) == 4);
    REQUIRE(tensor_3.dimension(2) == 2);

    for (size_t i = 0; i < tensor_3.dimension(0); ++i)
      for (size_t j = 0; j < tensor_3.dimension(1); ++j)
        for (size_t k = 0; k < tensor_3.dimension(2); ++k)
          REQUIRE(tensor_3(i, j, k) == (int)(201 * i + 30 * j + 102 * k)); 
  }
}

TEST_CASE("CrossContainer") {
  CrossContainerOperations<data::Array, data::HashMap>();
  CrossContainerOperations<data::HashMap, data::Array>();
}
