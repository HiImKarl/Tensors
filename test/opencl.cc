#include <tensor.hh>
#include <CL/cl2.hpp>
#include "test.hh"

using namespace tensor;
using namespace std;

int main(int argc, char **argv)
{
  // cout << opencl::Info::get_platforms().size() << endl;
  // cout << opencl::Info::v().platform().getInfo<CL_PLATFORM_NAME>() << endl;
  // cout << opencl::Info::v().device().getInfo<CL_DEVICE_NAME>() << endl;
  Matrix<float> mat1({10, 10}, 1); 
  Matrix<float> mat2({10, 10}, -10); 
  //Matrix<int> mat_ =  _reduce(0, [](int &x, int y) { x += y; }, mat).opencl();
  //Matrix<float> mat_ = (mat1 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2 + mat1 - mat2).opencl();
  //Matrix<float> mat_ = _map(math::tan{}, _map(math::sin{}, (_map(math::cos{}, mat1)))).opencl();
  //Matrix<float> mat_ = _map(math::min{}, mat1, -mat2).opencl();
  //Matrix<float> mat_ = (mat1 + mat2).opencl();
  /* Scalar<float> scalar = */ _reduce(0.0f, math::plus{}, mat1, mat2, mat1).opencl();
  //cout << mat_ << endl;
}
