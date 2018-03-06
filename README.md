
Tensor Template header-only library, where the rank of the tensor must be a compile time constant. As so far, operations on the tensors have not been implemented except for addition, subtraction, and ostream.

## Running the Test Cases

The test cases are in /test/test.cc. (For now) you will need a port of gcc if you are on Windows.
Run the following in the root directory:
```
make test
```

## Decleration:
```c++
// The following instantiates a 3rd degree tensor with dimensions 3x4x5
Tensor<double, 3> my_tensor {3, 4, 5};

// The following instantiates a scalar
Tensor<double> my_scalar {};
```

## Assignment
```c++
// Values are undefined/default-constructed until user assignment 
// tensors index beginning at 1
Tensor<double, 3> my_tensor {2, 4, 3};
for (int i = 1; i <= my_tensor.dimension(1); ++i) {
  for (int j = 1; j <= my_tensor.dimension(2); ++j) {
    for (int k = 1; k <= my_tensor.dimension(3); ++k) {
      my_tensor(i, j, k) = i + 0.1 * j + 0.01 * k;  
    }
  } 
}

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

// addition and subtraction have beed defined
// tensors must have the same dimensions
Tensor<int, 4> tensor_1 {1, 2, 3, 4};
Tensor<int, 2> tensor_2 {3, 4};
//... instantiate values
tensor_1(1, 1) += tensor_2;
```

## Printing the Tensor
```c++
// Just send to ostream
Tensor<int, 4> tensor_1 {1, 2, 3, 4};
//... instantiate values
std::cout << tensor_1 << std::endl;
```
