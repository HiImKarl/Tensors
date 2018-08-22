#include <tensor.hh>
#include "test.hh"

using namespace tensor;
using namespace std;

int main(int argc, char **argv)
{
  Matrix<int> mat({2, 2}, 10); 
  //cout << _reduce(0, [](int &x, int y) { x += y; }, mat).opencl_model().kernel() << endl;
  (mat + mat).opencl_model();
}
