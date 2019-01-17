#include "test.hh"

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
Tensor<TEST_STRUCT, 4, Container> test_func()
{ 
  Tensor<TEST_STRUCT, 4, Container> tensor({2, 2, 2, 2}); \
  return tensor; 
} 

// compile time exponent 

template <template <class> class Container>
void SingleTensorTest(int AllocSize) {
  constructor_counter = 0; 
  destructor_counter = 0; 

  SECTION("Single Tensor Constructor") { 
    Tensor<TEST_STRUCT, 4, Container> tensor({2, 2, 2, 2}); 
    REQUIRE(constructor_counter == AllocSize); 
  } 

  SECTION("Single Tensor Assignment") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2({2, 2, 2, 2}); 
    tensor_1 = tensor_2; 
    REQUIRE(constructor_counter == 2 * AllocSize); 
  } 

  SECTION("Single Tensor Desctructor") { 
    { Tensor<TEST_STRUCT, 4, Container> tensor({2, 2, 2, 2}); } 
    REQUIRE(destructor_counter == AllocSize); 
  } 

  SECTION("Single Scalar Constructor") { 
    Scalar<TEST_STRUCT, Container> scalar{}; 
    REQUIRE(constructor_counter == 1); 
  } 
} 

template <template <class> class Container>
void MultipleTensorTest(int AllocSize) {
  constructor_counter = 0; 
  destructor_counter = 0; 

  SECTION("Copy Constructor") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1; 
    REQUIRE(constructor_counter == 2 * AllocSize); 
  } 

  SECTION("Copy Assignment") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2({2, 2, 2, 2}); 
    tensor_2 = tensor_1; 
    REQUIRE(constructor_counter == 2 * AllocSize); 
  } 

  SECTION("Move Constructor") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2 = std::move(tensor_1); 
    REQUIRE(constructor_counter == AllocSize); 
  } 

  SECTION("Copy Destructor") { 
    { 
      Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
      Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1; 
    } 
    REQUIRE(destructor_counter == 2 * AllocSize); 
  } 

  SECTION("Move Destruction") { 
    { 
      Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
      Tensor<TEST_STRUCT, 4, Container> tensor_2 = std::move(tensor_1); 
    } 
    REQUIRE(destructor_counter == AllocSize); 
  } 

  SECTION("Ref Constructor") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1.ref(); 
    REQUIRE(constructor_counter == AllocSize); 
  } 

  SECTION("Move Destruction") { 
    { 
      Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
      Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1.ref(); 
    } 
    REQUIRE(destructor_counter == AllocSize); 
  } 

  SECTION("Functional Constructor") { 
    test_func<Container>(); 
    REQUIRE(constructor_counter == AllocSize); 
    auto tensor = test_func<Container>(); 
    REQUIRE(constructor_counter == 2 * AllocSize); 
  } 

  SECTION("Functional Destructor") { 
    { test_func<Container>(); } 
    REQUIRE(destructor_counter == AllocSize); 
    { auto tensor = test_func<Container>(); } 
    REQUIRE(destructor_counter == 2 * AllocSize); 
  } 
} 

template <template <class> class Container>
void ExplicitCopyTest(int AllocSize) {
  constructor_counter = 0; 
  destructor_counter = 0; 

  SECTION("Constructor") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1.copy(); 
    REQUIRE(constructor_counter == 2 * AllocSize); 
  } 

  SECTION("Destructor") { 
    { 
      Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
      Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1.copy(); 
    } 
    REQUIRE(destructor_counter == 2 * AllocSize); 
  } 
} 

template <template <class> class Container>
void ArithmeticTest(int AllocSize) {
  constructor_counter = 0; 
  destructor_counter = 0; 

  SECTION("Template Expressions constructor") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1; 
    constructor_counter = 0; 
    /* *Only one* alloc */ 
    Tensor<TEST_STRUCT, 4, Container> tensor_3 = 
      tensor_1 - tensor_1 + tensor_1 
      + tensor_2 - tensor_1 + tensor_1 + tensor_2; 
    REQUIRE(constructor_counter == AllocSize); 
  } 

  SECTION("Template Expressions Assignment") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1; 
    Tensor<TEST_STRUCT, 4, Container> tensor_3({2, 2, 2, 2}); 
    constructor_counter = 0; 
    /* *Only one* alloc */ 
    tensor_3 = tensor_1 - tensor_1 + tensor_1 
      + tensor_2 - tensor_1 + tensor_1 + tensor_2; 
    REQUIRE(constructor_counter == AllocSize); 
  } 

  SECTION("Template Expressions desstructor") { 
    { 
      Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
      Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1; 
      destructor_counter -= 2 * AllocSize; 
      /* *Only one* alloc */ 
      Tensor<TEST_STRUCT, 4, Container> tensor_3 = 
        tensor_1 - tensor_2 
        + tensor_2 - tensor_1 + tensor_1; 
    } 
    REQUIRE(destructor_counter == AllocSize); 
  } 
} 

template <template <class> class Container>
void TensorConstructionDestructionTests() {
  eDebugConstructorCounter = 0; 

  SECTION("Template Expressions Constructor") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1; 
    eDebugConstructorCounter = 0; 
    /* *Only one* alloc */ 
    Tensor<TEST_STRUCT, 4, Container> tensor_3 = tensor_1 - tensor_1 + tensor_1 
      + tensor_2 - tensor_1 + tensor_1 + tensor_2; 
    REQUIRE(eDebugConstructorCounter == 1); 
  } 

  SECTION("Template Expressions Assignment") { 
    Tensor<TEST_STRUCT, 4, Container> tensor_1({2, 2, 2, 2}); 
    Tensor<TEST_STRUCT, 4, Container> tensor_2 = tensor_1; 
    eDebugConstructorCounter = 0; 
    /* *Only one* alloc */ 
    tensor_2 = tensor_1 - tensor_2 - tensor_2 
      + tensor_1 + tensor_2 - tensor_1 + tensor_1; 
    REQUIRE(eDebugConstructorCounter == 1); 
  } 
}

TEST_CASE("Single Tensor") { 
  SingleTensorTest<data::Array>(16);
  SingleTensorTest<data::HashMap>(1);
}

TEST_CASE("Multiple Tensors") { 
  MultipleTensorTest<data::Array>(16);
  MultipleTensorTest<data::HashMap>(1);
}

TEST_CASE("Explicit Copy Method") { 
  ExplicitCopyTest<data::Array>(16);
  ExplicitCopyTest<data::HashMap>(1);
}

TEST_CASE("Arithmetic") { 
  ArithmeticTest<data::Array>(16); 
  ArithmeticTest<data::HashMap>(1);
}

TEST_CASE("Tensor Constructions/Destructions") { 
  TensorConstructionDestructionTests<data::Array>();
  TensorConstructionDestructionTests<data::HashMap>();
}
