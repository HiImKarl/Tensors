#include "test.hh"

using namespace tensor;

template <template <class> class Container>
void IteratorTests() {
  Tensor<int32_t, 3, Container> tensor({2, 3, 4}); 
  for (size_t i = 0; i < tensor.dimension(0); ++i) 
    for (size_t j = 0; j < tensor.dimension(1); ++j) 
      for (size_t k = 0; k < tensor.dimension(2); ++k) 
        tensor(i, j, k) = 100 * i + 10 * j + k; 
 
  SECTION("Tensor") { 
    auto begin = tensor.begin(0); 
    auto end = tensor.end(0); 
    REQUIRE(begin == tensor.begin(0)); 
    REQUIRE(begin != tensor.begin(1)); 
    REQUIRE(begin != tensor.begin(2)); 
    REQUIRE(begin != ++tensor.begin(0)); 
 
    REQUIRE(end == tensor.end(0)); 
    REQUIRE(end != --tensor.end(0)); 
    REQUIRE(end != tensor.end(1)); 
    REQUIRE(end != tensor.end(2)); 
 
    REQUIRE((*begin).rank() == 2); 
    REQUIRE(begin->rank() == 2); 
    REQUIRE(begin->shape()[0] == 3); 
    REQUIRE(begin->shape()[1] == 4); 
 
    REQUIRE((*end).rank() == 2); 
    REQUIRE(end->rank() == 2); 
    REQUIRE(end->shape()[0] == 3); 
    REQUIRE(end->shape()[1] == 4); 
 
    int32_t i = 0; 
    for (auto it = begin; it != end; ++it) 
      REQUIRE((*it)(1, 1) == 100 * (i++) + 11); 
    REQUIRE(i == 2); 
 
    i = 0; 
    for (auto it = begin; it != end; it++) 
      REQUIRE((*it)(1, 1) == 100 * (i++) + 11); 
    REQUIRE(i == 2); 
 
    REQUIRE((*++begin)(1, 1) == 111); 
    REQUIRE((*--begin)(1, 1) == 11); 
    REQUIRE((*begin++)(1, 1) == 11);
    REQUIRE((*begin)(2, 2) == 122); 
    REQUIRE((*begin--)(2, 2) == 122);
    REQUIRE((*begin)(0, 0) == 0); 
  } 
 
  SECTION("Scalar") { 
    auto begin = tensor(1, 2).begin(); 
    auto end = tensor(1, 2).end(); 
    int32_t i = 0; 
    for (auto it = begin; it != end; it++) 
      REQUIRE((*it) == (i++) + 120); 
    REQUIRE(i == 4); 
 
    i = 0; 
    for (auto it = begin; it != end; ++it) 
      REQUIRE((*it) == (i++) + 120); 
    REQUIRE(i == 4); 
  } 
 
  SECTION("Assignment") { 
    for (auto it1 = tensor.begin(0); it1 != tensor.end(0); ++it1) 
      for (auto it2 = it1->begin(0); it2 != it1->end(0); ++it2) 
        for (auto it3 = it2->begin(0); it3 != it2->end(0); ++it3) 
          (*it3) = 0; 
 
    for (size_t i = 0; i < tensor.dimension(0); ++i) 
      for (size_t j = 0; j < tensor.dimension(1); ++j) 
        for (size_t k = 0; k < tensor.dimension(2); ++k) 
          REQUIRE(tensor(i, j, k) == 0); 
  } 
 
  SECTION("Range based") { 
    int32_t i = 0; 
    for (auto&& _tensor : tensor(1, 1)) 
      REQUIRE(_tensor == 110 + (i++)); 
    REQUIRE(i == 4); 
  } 

  SECTION("Random Access Iterator Operations") {
    tensor(0, 0, 0) = 2;
    tensor(0, 0, 1) = 1;
    tensor(0, 0, 2) = 7;
    tensor(0, 0, 3) = -5;

    std::sort(tensor->begin(0)->begin(0)->begin(), tensor->begin(0)->begin(0)->end());

    REQUIRE(tensor(0, 0, 0) == -5);
    REQUIRE(tensor(0, 0, 1) == 1);
    REQUIRE(tensor(0, 0, 2) == 2);
    REQUIRE(tensor(0, 0, 3) == 7);
  }
} 
 
template <template <class> class Container>
void ConstIteratorTests() {
  Tensor<int32_t, 3, Container> tensor({2, 3, 4}); 
  for (size_t i = 0; i < tensor.dimension(0); ++i) 
    for (size_t j = 0; j < tensor.dimension(1); ++j) 
      for (size_t k = 0; k < tensor.dimension(2); ++k) 
        tensor(i, j, k) = 100 * i + 10 * j + k; 
 
  SECTION("Constness") { 
    auto begin = tensor.crbegin(0); 
    auto end = tensor.crend(0); 
    REQUIRE(std::is_const<decltype(*begin)>::value); 
    REQUIRE(std::is_const<decltype(*end)>::value); 
  } 
 
  SECTION("Tensor") { 
    auto begin = tensor.cbegin(0); 
    auto end = tensor.cend(0); 
    REQUIRE((*begin).rank() == 2); 
    REQUIRE(begin->rank() == 2); 
    REQUIRE(begin->shape()[0] == 3); 
    REQUIRE(begin->shape()[1] == 4); 
    REQUIRE(end->rank() == 2); 
    REQUIRE(end->shape()[0] == 3); 
    REQUIRE(end->shape()[1] == 4); 

    int32_t i = 0; 
    for (auto it = begin; it != end; ++it) 
      REQUIRE((*it)(1, 1) == 100 * (i++) + 11); 
    REQUIRE(i == 2); 
 
    i = 0; 
    for (auto it = begin; it != end; it++) 
      REQUIRE((*it)(1, 1) == 100 * (i++) + 11); 
    REQUIRE(i == 2); 
 
    REQUIRE((*++begin)(2, 2) == 122); 
    REQUIRE((*--begin)(1, 1) == 11); 
    REQUIRE((*begin++)(1, 1) == 11);
    REQUIRE((*begin)(2, 2) == 122); 
    REQUIRE((*begin--)(2, 2) == 122);
    REQUIRE((*begin)(1, 1) == 11); 
  } 
 
  SECTION("Scalar") { 
    auto begin = tensor(1, 2).cbegin(); 
    auto end = tensor(1, 2).cend(); 
    int32_t i = 0; 
    for (auto it = begin; it != end; it++) 
      REQUIRE(*it == i++ + 120); 
    REQUIRE(i == 4); 
 
    i = 0; 
    for (auto it = begin; it != end; ++it) 
      REQUIRE(*it == i++ + 120); 
    REQUIRE(i == 4); 
  } 
} 
 
template <template <class> class Container>
void ReverseIteratorTests() {
  Tensor<int32_t, 3, Container> tensor({2, 3, 4}); 
  for (size_t i = 0; i < tensor.dimension(0); ++i) 
    for (size_t j = 0; j < tensor.dimension(1); ++j) 
      for (size_t k = 0; k < tensor.dimension(2); ++k) 
        tensor(i, j, k) = 100 * i + 10 * j + k; 
 
  SECTION("Tensor") { 
    auto begin = tensor.rbegin(0); 
    auto end = tensor.rend(0); 
 
    REQUIRE((*begin).rank() == 2); 
    REQUIRE(begin->rank() == 2); 
    REQUIRE(begin->shape()[0] == 3); 
    REQUIRE(begin->shape()[1] == 4); 
    REQUIRE(end->rank() == 2); 
    REQUIRE(end->shape()[0] == 3); 
    REQUIRE(end->shape()[1] == 4); 
 
    int32_t i = 0; 
    for (auto it = begin; it != end; ++it) 
      REQUIRE((*it)(1, 1) == 100 * ((int)tensor.dimension(0) - (i++) - 1) + 11); 
    REQUIRE(i == 2); 
 
    i = 0; 
    for (auto it = begin; it != end; it++) 
      REQUIRE((*it)(1, 1) == 100 * ((int)tensor.dimension(0) - (i++) - 1) + 11); 
    REQUIRE(i == 2); 
 
    REQUIRE((*++begin)(2, 1) == 21); 
    REQUIRE((*--begin)(1, 2) == 112); 
    REQUIRE((*begin++)(1, 0) == 110); 
    REQUIRE((*begin)(0, 2) == 2); 
    REQUIRE((*begin--)(2, 2) == 22); 
    REQUIRE((*begin)(1, 1) == 111); 
  } 
 
  SECTION("Scalar") { 
    auto begin = tensor(1, 2).rbegin(); 
    auto end = tensor(1, 2).rend(); 
    int32_t i = 0; 

    for (auto it = begin; it != end; it++) 
      REQUIRE(*it == ((int)tensor.dimension(2) - (i++) - 1) + 120); 
    REQUIRE(i == 4); 
 
    i = 0; 
    for (auto it = begin; it != end; ++it) 
      REQUIRE(*it == ((int)tensor.dimension(2) - (i++) - 1) + 120); 
    REQUIRE(i == 4); 
  } 
 
  SECTION("Assignment") { 
    for (auto it1 = tensor.rbegin(0); it1 != tensor.rend(0); ++it1) 
      for (auto it2 = it1->rbegin(0); it2 != it1->rend(0); ++it2) 
        for (auto it3 = it2->rbegin(0); it3 != it2->rend(0); ++it3) 
          (*it3) = 0; 
 
    for (size_t i = 0; i < tensor.dimension(0); ++i) 
      for (size_t j = 0; j < tensor.dimension(1); ++j) 
        for (size_t k = 0; k < tensor.dimension(2); ++k) 
          REQUIRE(tensor(i, j, k) == 0); 
  } 

  SECTION("Random Access Iterator Operations") {
    tensor(0, 0, 0) = 2;
    tensor(0, 0, 1) = 1;
    tensor(0, 0, 2) = 7;
    tensor(0, 0, 3) = -5;

    std::sort(tensor->begin(0)->begin(0)->rbegin(), tensor->begin(0)->begin(0)->rend());

    REQUIRE(tensor(0, 0, 0) == 7);
    REQUIRE(tensor(0, 0, 1) == 2);
    REQUIRE(tensor(0, 0, 2) == 1);
    REQUIRE(tensor(0, 0, 3) == -5);
  }
} 

template <template <class> class Container>
void ReverseConstIteratorTests() {
  Tensor<int32_t, 3, Container> tensor({2, 3, 4}); 
  for (size_t i = 0; i < tensor.dimension(0); ++i) 
    for (size_t j = 0; j < tensor.dimension(1); ++j) 
      for (size_t k = 0; k < tensor.dimension(2); ++k) 
        tensor(i, j, k) = 100 * i + 10 * j + k; 
 
  SECTION("Constness") { 
    auto begin = tensor.crbegin(0); 
    auto end = tensor.crend(0); 
    REQUIRE(std::is_const<decltype(*begin)>::value); 
    REQUIRE(std::is_const<decltype(*end)>::value); 
  } 
 
  SECTION("Tensor") { 
    auto begin = tensor.crbegin(0); 
    auto end = tensor.crend(0); 
 
    REQUIRE((*begin).rank() == 2); 
    REQUIRE(begin->rank() == 2); 
    REQUIRE(begin->shape()[0] == 3); 
    REQUIRE(begin->shape()[1] == 4); 
    REQUIRE(end->rank() == 2); 
    REQUIRE(end->shape()[0] == 3); 
    REQUIRE(end->shape()[1] == 4); 
 
    int32_t i = 0; 
    for (auto it = begin; it != end; ++it) 
      REQUIRE((*it)(1, 1) == 100 * ((int)tensor.dimension(0) - (i++) - 1) + 11); 
    REQUIRE(i == 2); 
 
    i = 0; 
    for (auto it = begin; it != end; it++) 
      REQUIRE((*it)(1, 1) == 100 * ((int)tensor.dimension(0) - (i++) - 1) + 11); 
    REQUIRE(i == 2); 
 
    REQUIRE((*++begin)(2, 2) == 22); 
    REQUIRE((*--begin)(1, 1) == 111); 
    REQUIRE((*begin++)(1, 1) == 111); 
    REQUIRE((*begin)(2, 2) == 22); 
    REQUIRE((*begin--)(2, 2) == 22); 
    REQUIRE((*begin)(1, 1) == 111); 
  } 
 
  SECTION("Scalar") { 
    auto begin = tensor(1, 2).crbegin(); 
    auto end = tensor(1, 2).crend(); 
 
    int32_t i = 0; 
    for (auto it = begin; it != end; it++) 
      REQUIRE(*it == ((int)tensor.dimension(2) - (i++) - 1) + 120); 
    REQUIRE(i == 4); 
 
    i = 0; 
    for (auto it = begin; it != end; ++it) 
      REQUIRE(*it == ((int)tensor.dimension(2) - (i++) - 1) + 120); 
    REQUIRE(i == 4); 
  } 
}

// Instantiate test cases

TEST_CASE("Iterator" " | "  "Array") { 
  IteratorTests<data::Array>();
  IteratorTests<data::HashMap>();
}

TEST_CASE("Const Iterator" " | "  "Array") { 
  ConstIteratorTests<data::Array>();
  ConstIteratorTests<data::HashMap>();
}

TEST_CASE("Reverse Iterator" " | "  "Array") { 
  ReverseIteratorTests<data::Array>();
  ReverseIteratorTests<data::HashMap>();
}

TEST_CASE("Reverse Const Iterator" " | "  "Array") { 
  ReverseConstIteratorTests<data::Array>();
  ReverseConstIteratorTests<data::HashMap>();
}
