
Tensor Template header-only library. Everything except tensor addition and subtraction has to be user implemenented.

Tensors are multi-dimensional arrays. This library consists only of one headerfile, just include "tensors.hh" and everything is good to go.

## Decleration:
```c++
// The following instantiates a 3rd degree tensor with dimensions 3x4x5
Tensor<double> my_tensor {3, 4, 5};

// The followinf instantiates a scalar
Tensor<double> my_scalar {};
```

## Assignment
```c++
// Values are undefined/default-constructed until user assignment 
// tensors index beginning at 1
Tensor<double> my_tensor {2, 4, 3};
for (int i = 1; i <= my_tensor.dimension(1); ++i) {
  for (int j = 1; j <= my_tensor.dimension(2); ++j) {
    for (int k = 1; k <= my_tensor.dimension(3); ++k) {
      my_tensor(i, j) = i + 0.1 * j + 0.01 * k;  
    }
  } 
}

// scalars can be given its template parameter
Tensor<double> my_scalar {};
my_scalar = 3.1415;

// multi-dimensional tensors can also be assigned to each other as long as the indices match
Tensor<double> tensor_1 {3, 7, 14, 5};
Tensor<double> tensor_2 {7, 14, 5};
//... instantiate the values in tensor_2
tensor_1(2) = tensor_2;
tensor_2(1, 3) = tensor_2(5);
```

## Addition and Subtraction
```c++

// addition and subtraction are predefined
// tensors must have the same dimensions
Tensor<int> tensor_1 {1, 2, 3, 4};
Tensor<int> tensor_2 {3, 4};
//... instantiate values
tensor_1(1, 1) += tensor_2;
```
