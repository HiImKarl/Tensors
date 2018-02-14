#include "tensor.hh"

#define CATCH_CONFIG_MAIN
#include "catch.hh"

using namespace tensor;
using namespace std;

TEST_CASE("Intializing Tensors ") {
  
  Tensor<uint32_t> t1{ 1, 2, 3, 4};

  SECTION("Rank and Dimensions") {
    REQUIRE(t1.rank() == 4);
    for (int i = 1; i <= 4; ++i)
      REQUIRE(t1.dimension(i) == i);
  }

  SECTION("Initializing Values") {
    for (uint32_t i = 1; i <= t1.dimension(1); ++i)
      for (uint32_t j = 1; j <= t1.dimension(2); ++j)
        for (uint32_t k = 1; k <= t1.dimension(3); ++k)
          for (uint32_t l = 1; l <= t1.dimension(4); ++l)
            t1(i, j, k, l) = 1000 * i + 100 * j + 10 * k + l;
  }

  SECTION("Tensor Assignment") {
    Tensor<uint32_t> t2{ 3, 4};
    t1(1, 1) = t2;
    cout << t1 << '\n';
  }
}



