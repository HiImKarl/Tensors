#include "tensor.hh"
using namespace tensor;
using namespace std;
int main()
{
  Tensor<double> tensor{3, 3, 4, 7, 4, 3};
  for (int i = 1; i < 4; ++i) 
    for (int j = 1; j < 4; ++j) 
      for (int k = 1; k < 5; ++k) 
        for (int l = 1; l < 8; ++l) 
          for (int m = 1; m < 5; ++m) 
            for (int n = 1; n < 4; ++n)
              tensor(i, j, k, l, m, n) = i + j + k + l + m + n;

  Tensor<double> tensor_2{7, 4, 3};
  for (int i = 1; i < 8; ++i) 
    for (int j = 1; j < 5; ++j) 
      for (int k = 1; k < 4; k++)
        tensor_2(i, j, k) = 100 * i + 10 * j + k;
	

	Tensor<double> tensor_3{12, 4};
	for (int i = 1; i <= 12; ++i)
		for (int j = 1; j <= 4; ++j)
			tensor_3(i, j) = 23.34 * i + -324.3 * j;

	// Subtensors can be assigned if dimensions match 
  tensor(1, 2, 2) = tensor_2;
  cout << tensor << '\n';
	cout << "<------------------------------------------------------->\n";

	// scalars are declared without intitalizer arguments
  Tensor<double> scalar{};
  scalar = 109000;
  tensor_2(1, 1, 1) += scalar;
  cout << tensor_2 << '\n';
	cout << "<------------------------------------------------------->\n";
	
	// uh oh dimensions don't match
	tensor_2(3) = tensor_3;

	// uh oh dimension out of bounds
	tensor_2(8) = tensor_3;
	tensor_3(4, 0) = 23;
	cout << "<------------------------------------------------------->\n";

	cout << tensor_3;

  return 0;
}
