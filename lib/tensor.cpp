#include "tensor.hh"
using namespace tensor;
using namespace std;
int main() {
   Tensor<double> tensor {3, 3, 4, 7, 4};
   for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
              for (int l = 0; l < 7; ++l) {
                for (int m = 0; m < 4; ++m) {
                  tensor(i, j, k, l, m) = i + j + k + l + m;
                }
              }
            }
        }
   }
   Tensor<double> tensor_2 {4};
   for (int i = 0; i < 4; ++i) {
        tensor_2(i) = 10 + i;
   } 
   tensor(1, 2, 2, 1) = tensor_2;
   Tensor<double> tensor_3 = tensor_2;
   cout << tensor;
}