#include <tensor.hh>
#include "test.hh"

using namespace tensor;
using namespace std;

int main(int argc, char **argv)
{
  Matrix<float> mat1({10000, 1000}, 10); 
  Matrix<float> mat2({100, 100}, -10); 
  //cout << _reduce(0, [](int &x, int y) { x += y; }, mat).opencl_model().kernel() << endl;
  //Matrix<float> mat_ = (mat1 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2).opencl();
  Matrix<float> mat_ = _map(math::cos{}, mat1).opencl();
  //Matrix<float> mat_ = _map(math::min{}, mat1, mat2).opencl();
  //cout << mat_ << endl;
}
