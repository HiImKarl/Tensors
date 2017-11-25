#pragma once
#ifndef TENSOR_H_
#define TENSOR_H_
#include <initializer_list>
#include <iostream>
#include <stdint.h>
#include <string>

namespace tensor {

namespace util {

// Product of unsigned int array
inline uint32_t ArrayProduct(const uint16_t *xs, uint16_t size)
{
  uint32_t product = 1;
  for (uint16_t i = 0; i < size; ++i) {
    product *= xs[i];
  }
  return product;
}

// Copies an array with size specified by size and returns the ptr
template <typename T> inline T *ArrayCopy(const T *xs, uint32_t size)
{
  T *copy = new T[size];
  for (uint32_t i = 0; i < size; ++i) {
    copy[i] = xs[i];
  }
  return copy;
}

// Compares Arrays of equivalent sizes
inline bool ArrayCompare(uint16_t const *xs, uint16_t const *ys, uint16_t size)
{
	for (uint16_t i = 0; i < size; ++i)
		if (xs[i] != ys[i]) return false;
	return true;
}

// Repeats a string n time
std::string RepeatString(const std::string &to_repeat, uint32_t n)
{
  std::string out{};
  // string::size returns byte size
  out.reserve(n * to_repeat.size());
  for (uint32_t i = 0; i < n; ++i)
    out += to_repeat;
  return out;
}

void DebugLog(const std::string &msg) 
{
#ifndef _NDEBUG
	std::cout << "Debug :: " << msg << '\n';
#endif
}
} // namespace util

template <typename T> class Tensor {
  public:
  // Constructors
  Tensor();
  explicit Tensor(std::initializer_list<uint16_t> indices);
  Tensor(const Tensor<T> &tensor);
  Tensor(Tensor<T> &&tensor);

  // Assignment
  Tensor<T> &operator=(const Tensor<T> &tensor);
  Tensor<T> &operator=(Tensor<T> &&tensor);

  // Destructor
  ~Tensor();

  // Getters
  size_t degree() const { return degree_; }
  const uint16_t *dimensions() const { return dimensions_; }
  uint16_t dimension(uint16_t index) const;

  // Access to elements_
  template <typename... Args> Tensor<T> operator()(Args... args);

  // Setters
  Tensor<T> &operator=(T elem);

  // print
  template <typename X>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X> &tensor);

  // Operations
  template <typename X>
  friend Tensor<X> operator+(const Tensor<X> &tensor_1,
                             const Tensor<X> &tensor_2);
  template <typename X>
  friend Tensor<X> operator-(const Tensor<X> &tensor_1,
                             const Tensor<X> &tensor_2);
  void operator-=(const Tensor<T> &tensor);
  void operator+=(const Tensor<T> &tensor);
  template <typename X>
  friend Tensor<X> operator*(const Tensor<X> &, const Tensor<X> &);

  private:
  uint16_t degree_;
  uint16_t *dimensions_;
  T *elements_;

  // This flag denotes that this tensor object is the principle owner of memory
  // Smart ptr is unecessary
  bool is_owner_;

  // hidden constructor for access expansion
  // the first argument is moved
  Tensor(uint16_t *, unsigned);

  // Access Expansion
  Tensor<T> pAccessExpansion(uint16_t *, unsigned);
  template <typename... Args>
  Tensor<T> pAccessExpansion(uint16_t *, unsigned, uint16_t, Args...);
  uint32_t pCumulativeIndex(uint16_t *, unsigned);

}; // Tensor

// Tensor Methods

// Constructors, Destructor, Assignment
template <typename T>
Tensor<T>::Tensor() : dimensions_(nullptr), degree_(0), is_owner_(true)
{
  elements_ = new T;
}

template <typename T> Tensor<T>::Tensor(std::initializer_list<uint16_t> indices)
{
  degree_ = indices.size();
  dimensions_ = util::ArrayCopy(indices.begin(), degree_);
  uint32_t num_elements = util::ArrayProduct(dimensions_, degree_);
  elements_ = new T[num_elements];
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T> &tensor)
    : degree_(tensor.degree_), is_owner_(true)
{
  dimensions_ = util::ArrayCopy(tensor.dimensions_, degree);
  elements_ =
      util::ArrayCopy(tensor.elements_, pVectorProduct(dimensions_, degree));
}
template <typename T>
Tensor<T>::Tensor(Tensor<T> &&tensor) : degree_(tensor.degree_), is_owner_(true)
{
  // If tensor owns its memory, move; ow. copy
  dimensions_ = util::ArrayCopy(tensor.dimensions_, degree_);
  if (tensor.is_owner_) {
    elements_ = tensor.elements_;
    tensor.elements_ = nullptr;
    tensor.is_owner_ = false;
  } else {
    elements_ = util::ArrayCopy(tensor.elements_,
                                util::ArrayProduct(dimensions_, degree_));
  }
}

template <typename T> Tensor<T>::~Tensor()
{
  if (is_owner_) {
    delete[] elements_;
  }
  delete[] dimensions_;
}
template <typename T> Tensor<T> &Tensor<T>::operator=(const Tensor<T> &tensor)
{
  if (degree_ != tensor.degree_) {
    util::DebugLog("Tensor assignment failed, tensors do not have the same degree");
    return *this;
  }
  for (uint16_t i = 0; i < degree_; ++i) {
    if (dimensions_[i] != tensor.dimensions_[i]) {
      util::DebugLog("Tensor assignment failed, tensor dimension mismatch");
      return *this;
    }
  }
  uint32_t total_elems = util::ArrayProduct(tensor.dimensions_, degree_);
  for (uint32_t i = 0; i < total_elems; ++i) {
    elements_[i] = tensor.elements_[i];
  }

  return *this;
}

template <typename T> Tensor<T> &Tensor<T>::operator=(Tensor<T> &&tensor)
{
  if (degree_ != tensor.degree) {
			util::DebugLog("Tensor assignment failed, tensors do not have the same degree");
    return *this;
  }
  for (uint16_t i = 0; i < degree_; ++i) {
    if (dimensions_[i] != tensor.dimensions_[i]) {
      util::DebugLog("Tensor assignment failed, tensor dimension mismatch");
      return *this;
    }
  }
  // DO NOT MOVE DATA unless boths tensors are principle owners
  if (tensor.is_owner_ && is_owner_) {
    elements_ = tensor.elements_;
    tensor.elements_ = nullptr;
    tensor.is_owner_ = false;
  } else {
    uint32_t total_elems = util::ArrayProduct(dimensions_, degree_);
    for (uint32_t i = 0; i < total_elems; ++i) {
      elements_[i] = tensor.elements_[i];
    }
  }
  return *this;
}

// Getters
template <typename T> uint16_t Tensor<T>::dimension(uint16_t index) const
{
  if (degree_ < index || index == 0) {
    util::DebugLog("Attempt to access invalid dimension");
    return 0;
  }
  return dimensions_[index - 1];
}

// Setters
template <typename T> Tensor<T> &Tensor<T>::operator=(T elem)
{
  if (degree_ != 0) {
    util::DebugLog("Cannot assign scalar to multi dimensional tensor");
    return *this;
  }
  *elements_ = elem;
  return *this;
}

// Access to elements_
template <typename T>
Tensor<T>::Tensor(uint16_t *dimensions, unsigned degree)
    : dimensions_(dimensions), degree_(degree)
{
  uint32_t num_elements = util::ArrayProduct(dimensions_, degree_);
  elements_ = new T[num_elements];
}

template <typename T>
template <typename... Args>
Tensor<T> Tensor<T>::operator()(Args... args)
{
  uint16_t *indices = new uint16_t[degree_];
  unsigned curr_dim = 0;
  return pAccessExpansion(indices, curr_dim, args...);
}

template <typename T>
Tensor<T> Tensor<T>::pAccessExpansion(uint16_t *indices, unsigned curr_dim)
{
  uint32_t indices_product = pCumulativeIndex(indices, curr_dim);
  uint16_t *new_indices = new uint16_t[degree_ - curr_dim];
  for (unsigned i = curr_dim; i < degree_; ++i) {
    new_indices[i - curr_dim] = dimensions_[i];
  }
  // ptr for new_indices is moved so delete[] is unecessary
  Tensor<T> new_tensor(new_indices, degree_ - curr_dim);
  new_tensor.elements_ = elements_ + indices_product;
  new_tensor.is_owner_ = false;
  return new_tensor;
}

template <typename T>
template <typename... Args>
Tensor<T> Tensor<T>::pAccessExpansion(uint16_t *indices, unsigned curr_index,
                                      uint16_t next_index, Args... rest)
{
  if (curr_index == degree_) {
    util::DebugLog("Dimension " + std::to_string(curr_index) + " out of bounds");
    return Tensor<T>();
  }
  if (next_index > dimensions_[curr_index] || next_index == 0) {
    util::DebugLog("Index " + std::to_string(curr_index) + " (" + std::to_string(next_index) + ") Out of Bounds");
    return Tensor<T>();
  }
  // adjust for 1 index array access
  indices[curr_index] = --next_index;
  return pAccessExpansion(indices, ++curr_index, rest...);
}

template <typename T>
uint32_t Tensor<T>::pCumulativeIndex(uint16_t *xs, unsigned size)
{
  uint32_t total_elems = util::ArrayProduct(dimensions_, degree_);
  uint32_t cumul = 0;
  for (size_t i = 0; i < size; ++i) {
    total_elems /= dimensions_[i];
    cumul += xs[i] * total_elems;
  }
  return cumul;
}

// Overloaded operators
template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor)
{
  // if its a scalar just return;
  if (!tensor.degree()) {
    os << tensor.elements_[0];
    return os;
  }
  uint32_t indices_product =
      util::ArrayProduct(tensor.dimensions(), tensor.degree());
  uint32_t bracket_mod_arr[tensor.degree()];
  uint32_t prod = 1;
  for (size_t i = 0; i < tensor.degree(); ++i) {
    prod *= tensor.dimensions()[tensor.degree() - i - 1];
    bracket_mod_arr[i] = prod;
  }
  os << util::RepeatString("[", tensor.degree());
  os << tensor.elements_[0] << ", ";
  for (uint32_t i = 1; i < indices_product - 1; ++i) {
    os << tensor.elements_[i];
    uint16_t num_brackets = 0;
    for (; num_brackets < tensor.degree(); ++num_brackets) {
      if (!((i + 1) % bracket_mod_arr[num_brackets]) == 0) break;
    }
    os << util::RepeatString("]", num_brackets);
    os << ", ";
    os << util::RepeatString("[", num_brackets);
  }
  if (indices_product - 1) os << tensor.elements_[indices_product - 1];
  os << util::RepeatString("]", tensor.degree());
  return os;
}

template <typename T> void Tensor<T>::operator+=(const Tensor<T> &tensor)
{
  if (degree_ != tensor.degree_ || !util::ArrayCompare(dimensions_, tensor.dimensions_, degree_)) {
    util::DebugLog("operator+=(Tensor) Failed :: Incompatible Dimensions");
		return;
  }
  uint32_t total_dims = util::ArrayProduct(dimensions_, degree_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    elements_[i] += tensor.elements_[i];
  }
}

template <typename T> void Tensor<T>::operator-=(const Tensor<T> &tensor)
{
  if (degree_ != tensor.degree_ || !util::ArrayCompare(dimensions_, tensor.dimensions_, degree_)) {
    util::DebugLog("operator-=(Tensor) Failed :: Incompatible Dimensions");
    return Tensor<T>();
  }
  uint32_t total_dims = util::ArrayProduct(dimensions_, degree_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    elements_[i] -= tensor.elements_[i];
  }
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &tensor_1, const Tensor<T> &tensor_2)
{
  if (tensor_1.degree_ != tensor_2.degree_ || !util::ArrayCompare(tensor_1.dimensions_, tensor_2.dimensions_, tensor_1.degree_)) {
    util::DebugLog("operator+(Tensor, Tensor) Failed :: Incompatible Dimensions");
    return Tensor<T>();
  }
  Tensor<T> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    new_tensor.elements_[i] = tensor_1.elements_[i] + tensor_2.elements_[i];
  }
  return new_tensor;
}

template <typename T>
Tensor<T> operator-(const Tensor<T> &tensor_1, const Tensor<T> &tensor_2)
{
  if (tensor_1.degree_ != tensor_2.degree_ || !util::ArrayCompare(tensor_1.dimensions_, tensor_2.dimensions_, tensor_1.degree_)) {
    util::DebugLog("operator+(Tensor, Tensor) Failed :: Incompatible Dimensions");
    return Tensor<T>();
  }
  Tensor<T> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    new_tensor.elements_[i] = tensor_1.elements_[i] - tensor_2.elements_[i];
  }
  return new_tensor;
}



} // tensor

#endif // TENSORS_H_
