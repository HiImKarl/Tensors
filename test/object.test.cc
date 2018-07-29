#include <tensor.hh>
#include <catch.hh>

template <int *ConstructorCounter, int *DestructorCounter>
struct TestStruct {
  TestStruct() { *ConstructorCounter += 1; }
  ~TestStruct() { *DestructorCounter += 1; }
};

#define TEST_STRUCT \
  TestStruct<&constructor_counter, &destructor_counter>
#define TEST_STRUCT_4 \
  Tensor<TestStruct<&constructor_counter, &destructor_counter>, 4>
#define TEST_SCALAR \
  Scalar<TestStruct<&constructor_counter, &destructor_counter>>

template <int *i1, int *i2>
std::ostream& operator<<(std::ostream &os, TestStruct<i1, i2> const &)
{
  os << "##\n";
  return os;
}

using namespace tensor;
int constructor_counter;
int destructor_counter;

TEST_STRUCT operator+(TEST_STRUCT const&, TEST_STRUCT const&) {
  --constructor_counter;
  --destructor_counter; 
  return TEST_STRUCT();
}

TEST_STRUCT operator-(TEST_STRUCT const&, TEST_STRUCT const&) {
  --constructor_counter;
  --destructor_counter;
  return TEST_STRUCT();
}

static Tensor<TEST_STRUCT, 4> test_func()
{
  TEST_STRUCT_4 tensor({2, 2, 2, 2});
  return tensor;
}

/**
 *  Notice that the copy constructor and move constructor should behave in the same way
 *  the move constructor is not explicitly defined
 */

TEST_CASE("Single Tensor") {
  constructor_counter = 0;
  destructor_counter = 0;

  SECTION("Single Tensor Constructor") {
    Tensor<TEST_STRUCT, 4> tensor({2, 2, 2, 2});
    REQUIRE(constructor_counter == 16);
  }

  SECTION("Single Tensor Desctructor") {
    { TEST_STRUCT_4 tensor({2, 2, 2, 2}); }
    REQUIRE(destructor_counter == 16);
  }

  SECTION("Single Scalar Constructor") {
    TEST_SCALAR scalar{};
    REQUIRE(constructor_counter == 1);
  }
}

TEST_CASE("Multiple Tensors") {
  constructor_counter = 0;
  destructor_counter = 0;

  SECTION("Copy Constructor") {
    TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
    TEST_STRUCT_4 tensor_2 = tensor_1;
    REQUIRE(constructor_counter == 32);
  }

  SECTION("Move Constructor") {
    TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
    TEST_STRUCT_4 tensor_2 = std::move(tensor_1);
    REQUIRE(constructor_counter == 16);
  }

  SECTION("Copy Destructor") {
    {
      TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
      TEST_STRUCT_4 tensor_2 = tensor_1;
    }
    REQUIRE(destructor_counter == 32);
  }

  SECTION("Move Destruction") {
    {
      TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
      TEST_STRUCT_4 tensor_2 = std::move(tensor_1);
    }
    REQUIRE(destructor_counter == 16);
  }

  SECTION("Ref Constructor") {
    TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
    TEST_STRUCT_4 tensor_2 = tensor_1.ref();
    REQUIRE(constructor_counter == 16);
  }

  SECTION("Move Destruction") {
    {
      TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
      TEST_STRUCT_4 tensor_2 = tensor_1.ref();
    }
    REQUIRE(destructor_counter == 16);
  }

  SECTION("Functional Constructor") {
    test_func();
    REQUIRE(constructor_counter == 16);
    auto tensor = test_func();
    REQUIRE(constructor_counter == 32);
  }

  SECTION("Functional Destructor") {
    { test_func(); }
    REQUIRE(destructor_counter == 16);
    { auto tensor = test_func(); }
    REQUIRE(destructor_counter == 32);
  }
}

TEST_CASE("Explicit Copy Method") {
  constructor_counter = 0;
  destructor_counter = 0;

  SECTION("Constructor") {
    TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
    TEST_STRUCT_4 tensor_2 = tensor_1.copy();
    REQUIRE(constructor_counter == 32);
  }

  SECTION("Destructor") {
    {
      TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
      TEST_STRUCT_4 tensor_2 = tensor_1.copy();
    }
    REQUIRE(destructor_counter == 32);
  }
}

TEST_CASE("Arithmetic") {
  constructor_counter = 0;
  destructor_counter = 0;

  SECTION("Template Expressions constructor") {
    TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
    TEST_STRUCT_4 tensor_2 = tensor_1;
    constructor_counter = 0;                        // reset alloc counter
    TEST_STRUCT_4 tensor_3 = 
      tensor_1 - tensor_1 + tensor_1
      + tensor_2 - tensor_1 + tensor_1 + tensor_2;  // *Only one* alloc -> 16
    REQUIRE(constructor_counter == 16);
  }

  SECTION("Template Expressions desstructor") {
    {
      TEST_STRUCT_4 tensor_1({2, 2, 2, 2}); // One alloc -> 16
      TEST_STRUCT_4 tensor_2 = tensor_1;    // Second alloc -> 32 
      destructor_counter -= 32;                  // reset alloc counter
      TEST_STRUCT_4 tensor_3 = 
        tensor_1 - tensor_2
        + tensor_2 - tensor_1 + tensor_1;        // *Only one* alloc -> 16 
    }
    REQUIRE(destructor_counter == 16);
  }
}

TEST_CASE("Tensor Constructions/Destructions") {
  eDebugConstructorCounter = 0; 

  SECTION("Template Expressions Constructor") {
    TEST_STRUCT_4 tensor_1({2, 2, 2, 2});
    TEST_STRUCT_4 tensor_2 = tensor_1;
    eDebugConstructorCounter = 0;                   // reset alloc counter
    TEST_STRUCT_4 tensor_3 = tensor_1 - tensor_1 + tensor_1
      + tensor_2 - tensor_1 + tensor_1 + tensor_2;  // *Only one* alloc -> 1
    REQUIRE(eDebugConstructorCounter == 1);
  }

  SECTION("Template Expressions Assignment") {
    TEST_STRUCT_4 tensor_1({2, 2, 2, 2}); 
    TEST_STRUCT_4 tensor_2 = tensor_1;   
    eDebugConstructorCounter = 0;                    // reset alloc counter
    tensor_2 = tensor_1 - tensor_2 - tensor_2 
      + tensor_1 + tensor_2 - tensor_1 + tensor_1;   // *Only one* alloc -> 1
    REQUIRE(eDebugConstructorCounter == 1);
  }
}
