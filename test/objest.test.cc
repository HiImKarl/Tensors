#include "tensor.hh"
#include "catch.hh"

template <int *ConstructorCounter, int *DestructorCounter>
struct TestStruct {
  TestStruct() { *ConstructorCounter += 1; }
  ~TestStruct() { *DestructorCounter += 2; }
};

using namespace tensor;
static int constructor_counter;
static int destructor_counter;

TEST_CASE("Object") {
  constructor_counter = 0;
  destructor_counter = 0;
  
  SECTION("Constructor") {
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor({2, 2, 2, 2});
    REQUIRE(constructor_counter == 16);
  }

  SECTION("Desctructor") {
    // anonymous scope to call desctructor
    { Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor({2, 2, 2, 2}); }
    REQUIRE(destructor_counter == 32);
  }
}
