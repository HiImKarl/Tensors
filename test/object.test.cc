#include <tensor.hh>
#include <catch.hh>

template <int *ConstructorCounter, int *DestructorCounter>
struct TestStruct {
  TestStruct() { *ConstructorCounter += 1; }
  ~TestStruct() { *DestructorCounter += 1; }
};

#define TEST_STRUCT \
  TestStruct<&constructor_counter, &destructor_counter>

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

// This will force SparseTensor to not fill elements 
bool operator==(TEST_STRUCT const&, TEST_STRUCT const&) {
  return true;
}

/** Notice that the copy constructor and move constructor should behave in the same way
 *  the move constructor is not explicitly defined
 */

template <template <typename> class Container>
Tensor<TEST_STRUCT, 4, Container<TEST_STRUCT>> test_func()
{ 
  Tensor<TEST_STRUCT, 4, Container<TEST_STRUCT>> tensor({2, 2, 2, 2}); \
  return tensor; 
} 

// compile time exponent 

#define TEST_CASES(CONTAINER, ALLOC_SIZE) \
  TEST_CASE(#CONTAINER ": Single Tensor") { \
    constructor_counter = 0; \
    destructor_counter = 0; \
 \
    SECTION("Single Tensor Constructor") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor({2, 2, 2, 2}); \
      REQUIRE(constructor_counter == ALLOC_SIZE); \
    } \
 \
    SECTION("Single Tensor Assignment") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2({2, 2, 2, 2}); \
      tensor_1 = tensor_2; \
      REQUIRE(constructor_counter == 2 * ALLOC_SIZE); \
    } \
 \
    SECTION("Single Tensor Desctructor") { \
      { Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor({2, 2, 2, 2}); } \
      REQUIRE(destructor_counter == ALLOC_SIZE); \
    } \
 \
    SECTION("Single Scalar Constructor") { \
      Scalar<TEST_STRUCT, CONTAINER<TEST_STRUCT>> scalar{}; \
      REQUIRE(constructor_counter == 1); \
    } \
  } \
 \
  TEST_CASE(#CONTAINER ": Multiple Tensors") { \
    constructor_counter = 0; \
    destructor_counter = 0; \
 \
    SECTION("Copy Constructor") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1; \
      REQUIRE(constructor_counter == 2 * ALLOC_SIZE); \
    } \
 \
    SECTION("Copy Assignment") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2({2, 2, 2, 2}); \
      tensor_2 = tensor_1; \
      REQUIRE(constructor_counter == 2 * ALLOC_SIZE); \
    } \
 \
    SECTION("Move Constructor") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = std::move(tensor_1); \
      REQUIRE(constructor_counter == ALLOC_SIZE); \
    } \
 \
    SECTION("Copy Destructor") { \
      { \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1; \
      } \
      REQUIRE(destructor_counter == 2 * ALLOC_SIZE); \
    } \
 \
    SECTION("Move Destruction") { \
      { \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = std::move(tensor_1); \
      } \
      REQUIRE(destructor_counter == ALLOC_SIZE); \
    } \
 \
    SECTION("Ref Constructor") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1.ref(); \
      REQUIRE(constructor_counter == ALLOC_SIZE); \
    } \
 \
    SECTION("Move Destruction") { \
      { \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1.ref(); \
      } \
      REQUIRE(destructor_counter == ALLOC_SIZE); \
    } \
 \
    SECTION("Functional Constructor") { \
      test_func<CONTAINER>(); \
      REQUIRE(constructor_counter == ALLOC_SIZE); \
      auto tensor = test_func<CONTAINER>(); \
      REQUIRE(constructor_counter == 2 * ALLOC_SIZE); \
    } \
 \
    SECTION("Functional Destructor") { \
      { test_func<CONTAINER>(); } \
      REQUIRE(destructor_counter == ALLOC_SIZE); \
      { auto tensor = test_func<CONTAINER>(); } \
      REQUIRE(destructor_counter == 2 * ALLOC_SIZE); \
    } \
  } \
 \
  TEST_CASE(#CONTAINER ": Explicit Copy Method") { \
    constructor_counter = 0; \
    destructor_counter = 0; \
 \
    SECTION("Constructor") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1.copy(); \
      REQUIRE(constructor_counter == 2 * ALLOC_SIZE); \
    } \
 \
    SECTION("Destructor") { \
      { \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1.copy(); \
      } \
      REQUIRE(destructor_counter == 2 * ALLOC_SIZE); \
    } \
  } \
 \
  TEST_CASE(#CONTAINER ": Arithmetic") { \
    constructor_counter = 0; \
    destructor_counter = 0; \
 \
    SECTION("Template Expressions constructor") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1; \
      constructor_counter = 0; \
      /* *Only one* alloc */ \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_3 = \
        tensor_1 - tensor_1 + tensor_1 \
        + tensor_2 - tensor_1 + tensor_1 + tensor_2; \
      REQUIRE(constructor_counter == ALLOC_SIZE); \
    } \
 \
    SECTION("Template Expressions Assignment") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1; \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_3({2, 2, 2, 2}); \
      constructor_counter = 0; \
      /* *Only one* alloc */ \
      tensor_3 = tensor_1 - tensor_1 + tensor_1 \
        + tensor_2 - tensor_1 + tensor_1 + tensor_2; \
      REQUIRE(constructor_counter == ALLOC_SIZE); \
    } \
 \
    SECTION("Template Expressions desstructor") { \
      { \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1; \
        destructor_counter -= 2 * ALLOC_SIZE; \
        /* *Only one* alloc */ \
        Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_3 = \
          tensor_1 - tensor_2 \
          + tensor_2 - tensor_1 + tensor_1; \
      } \
      REQUIRE(destructor_counter == ALLOC_SIZE); \
    } \
  } \
 \
  TEST_CASE(#CONTAINER ": Tensor Constructions/Destructions") { \
    eDebugConstructorCounter = 0; \
 \
    SECTION("Template Expressions Constructor") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1; \
      eDebugConstructorCounter = 0; \
      /* *Only one* alloc */ \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_3 = tensor_1 - tensor_1 + tensor_1 \
        + tensor_2 - tensor_1 + tensor_1 + tensor_2; \
      REQUIRE(eDebugConstructorCounter == 1); \
    } \
 \
    SECTION("Template Expressions Assignment") { \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_1({2, 2, 2, 2}); \
      Tensor<TEST_STRUCT, 4, CONTAINER<TEST_STRUCT>> tensor_2 = tensor_1; \
      eDebugConstructorCounter = 0; \
      /* *Only one* alloc */ \
      tensor_2 = tensor_1 - tensor_2 - tensor_2 \
        + tensor_1 + tensor_2 - tensor_1 + tensor_1; \
      REQUIRE(eDebugConstructorCounter == 1); \
    } \
  }

// instantiate test cases
TEST_CASES(data::Array, 16);
TEST_CASES(data::HashMap, 1);
