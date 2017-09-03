#include "tensor.hh"
using namespace tensor;
using namespace std;
int main() {
  Tensor<double> tensor{3, 3, 4, 7, 4, 3};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 4; ++m) {
            for (int n = 0; n < 3; ++n)
              tensor(i, j, k, l, m, n) = i + j + k + l + m + n;
          }
        }
      }
    }
  }
  Tensor<double> tensor_2{7, 4, 3};
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 3; k++)
        tensor_2(i, j, k) = 100 * i + 10 * j + k;
    }
  }
  tensor(1, 2, 2) = tensor_2;
  cout << tensor_2;
}