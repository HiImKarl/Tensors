#include "tensor.hh"
#include "catch.hh"

template <int *ConstructorCounter, int *DestructorCounter>
struct TestStruct {
  TestStruct() { *ConstructorCounter += 1; }
  ~TestStruct() { *DestructorCounter += 2; }
};

template <int *i1, int *i2>
std::ostream& operator<<(std::ostream &os, TestStruct<i1, i2> const &T)
{
  os << "##\n";
  return os;
}

using namespace tensor;
static int constructor_counter;
static int destructor_counter;

static Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> test_func()
{
  Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor({2, 2, 2, 2});
  return tensor;
}

TEST_CASE("Single Tensor") {
  constructor_counter = 0;
  destructor_counter = 0;
  
  SECTION("Single Tensor Constructor") {
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor({2, 2, 2, 2});
    REQUIRE(constructor_counter == 16);
  }

  SECTION("Single Tensor Desctructor") {
    { Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor({2, 2, 2, 2}); }
    REQUIRE(destructor_counter == 32);
  }

  SECTION("Single Scalar Constructor") {
    Tensor<TestStruct<&constructor_counter, &destructor_counter>> scalar{};
    REQUIRE(constructor_counter == 1);
  }
}

TEST_CASE("Multiple Tensors") {
  constructor_counter = 0;
  destructor_counter = 0;
  
  SECTION("Copy Constructor") {
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor_1({2, 2, 2, 2});
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor_2 = tensor_1;
    REQUIRE(constructor_counter == 32);
  }

  SECTION("Move Constructor") {
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor_1({2, 2, 2, 2});
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor_2 = std::move(tensor_1);
    REQUIRE(constructor_counter == 16);
  }

  SECTION("Copy Destructor") {
    {
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor_1({2, 2, 2, 2});
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor_2 = tensor_1;
    }
    REQUIRE(destructor_counter == 64);
  }

  SECTION("Move Construction") {
    {
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor_1({2, 2, 2, 2});
    Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4> tensor_2 = std::move(tensor_1);
    }
    REQUIRE(destructor_counter == 32);
  }

  SECTION("Functional Constructor") {
    test_func();
    REQUIRE(constructor_counter == 16);
    auto tensor = test_func();
    REQUIRE(constructor_counter == 32);
  }

  SECTION("Functional Destructor") {
    { test_func(); }
    REQUIRE(destructor_counter == 32);
    { auto tensor = test_func(); }
    REQUIRE(destructor_counter == 64);
  }
}
