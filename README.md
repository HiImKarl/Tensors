
Tensor Template header-only library, where the rank of the tensor must be a compile time constant. As so far, operations on the tensors have not been implemented except for addition, subtraction, and ostream.

## Definition

Tensors are defined as follows, where T is the tensor data type (which must be a non-reference type) and N is the rank of the tensor.

```c++
template <typename T, uint32_t N = 0>
class Tensor;
```

The rank and dimension of tensor values can be retrieved with the following methods

```c++
constexpr uint32_t rank() const noexcept;
uint32_t dimension() const;
```

## Instaniation and Access

You can access tensor values with operator(), which is vardiacally expanded and checked at compile time to ensure the number of arguments don't exceed the rank of the tensor

 ```c++
 // The following instantiates a 3rd degree tensor with dimensions 3x4x5
 // The circle brackets are required, an std::initializer_list constructor is not implemented 
 Tensor<string, 3> my_tensor({3, 4, 5});
 
 // The following instantiates a scalar
 Tensor<double> my_scalar();
 
 // Tensors are indexed beginning at 1
 // You can access a 4x5 tensor in my_tensor with
 my_tensor(1);
 
 // You can access the scalar with 
 my_scalar();
 // or just
 my_scalar;
 
 // example
 string str = my_tensor(2, 1, 4);
 double x = my_scalar;
```


## Assignment
```c++
// Values are undefined/default-constructed until user assignment 
Tensor<double, 3> my_tensor({2, 4, 3});
for (int i = 1; i <= my_tensor.dimension(1); ++i) 
  for (int j = 1; j <= my_tensor.dimension(2); ++j) 
    for (int k = 1; k <= my_tensor.dimension(3); ++k) 
      my_tensor(i, j, k) = i + 10 * j + 100 * k;  

// scalars can be given its template parameter
Tensor<double> my_scalar {};
my_scalar = 3.1415;

// multi-dimensional tensors can also be assigned to each other as long as the indices match
Tensor<double, 4> tensor_1 {3, 7, 14, 5};
Tensor<double, 3> tensor_2 {7, 14, 5};
//... instantiate the values in tensor_1 and tensor_2
tensor_1(2) = tensor_2;
tensor_2(1, 3) = tensor_2(7, 12);
```

## Addition and Subtraction
```c++

// A naive implementation for addition and subtraction have beed defined as an example
Tensor<int, 4> tensor_1 {1, 2, 3, 4};
Tensor<long, 2> tensor_2 {3, 4};
//... instantiate values
tensor_1(1, 1) += tensor_2;
```

## Printing the Tensor
```c++
// An implementation for ostream << has also been provided
Tensor<int, 4> tensor_1 {1, 2, 3, 4};
//... instantiate values
std::cout << tensor_1 << std::endl;
```

## Error Handling

If the rank of the tensor does not match what you are trying to do, compilation will fail. If you try to access an element that is out of bounds, an std::logic_error exception will be thrown.

## Running the Unit Tests

The tests are in /test/test.cc. (For now) you will need a port of gcc if you are on Windows.
Run the following in the root directory:
```
make test
```

