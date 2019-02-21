#include <CImg.h>
#include <tensor.hh>

namespace tensor {
template <>
struct StringFormat<unsigned char> {
  std::string operator()(unsigned char x) const { return std::to_string(x); }
};
}

int main()
{
  auto image = cimg_library::CImg<unsigned char>(_PROJECT_SOURCE_DIR "/example/image/koala.jpg");
  auto height = image.height();
  auto width = image.width();
  auto t = tensor::Tensor<unsigned char, 2>({height, width}); 
  t.Fill(image.data(), image.data() + width * height);

  auto rotated_image = cimg_library::CImg<unsigned char>(height, width);
  for (int i = 0; i < height; ++i)
    for (int j = 0; j < width; ++j)
      *rotated_image.data(i, j) = t(height - i - 1, width - j - 1);

  auto display = cimg_library::CImgDisplay(rotated_image, "Rotated Image");
  while (1) {
    if (display.is_closed()) break;
  }
  return 0;
}
