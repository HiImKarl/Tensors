#pragma once
#ifndef TENSOR_H_
#define TENSOR_H_
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <exception>
#include <utility>
#include <type_traits>

namespace tensor {
namespace util {

// Product of unsigned int array
inline uint32_t ArrayProduct(const uint32_t *xs, uint32_t size)
{
  uint32_t product = 1;
  for (uint32_t i = 0; i < size; ++i) {
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

// Compares uint32_t arrays of equivalent sizes
inline bool ArrayCompare(uint32_t const *xs, uint32_t const *ys, uint32_t size)
{
  for (uint32_t i = 0; i < size; ++i) if (xs[i] != ys[i]) return false;
  return true;
}

} // namespace util

// TENSORS TYPE CANNOT BE A REFERENCE
template <typename T>
class Tensor {
public:
  // typedefs
  typedef T value_type;

  // Constructors
  Tensor();
  explicit Tensor(value_type &&val);
  explicit Tensor(std::initializer_list<uint32_t> indices);
  Tensor(const Tensor<T> &tensor);
  Tensor(Tensor<T> &&tensor);

  // Assignment
  Tensor<T> &operator=(const Tensor<T> &tensor);
  Tensor<T> &operator=(Tensor<T> &&tensor);

  // Destructor
  ~Tensor();

  // Getters
  uint32_t rank() const { return degree_; }
  uint32_t dimension(uint32_t index) const;

  // Access to elements_
  template <typename... Args> Tensor<T> operator()(Args... args);

  // Setters
  template <typename X, 
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type, 
            typename std::remove_reference<X>::type>::value>::type>
  Tensor<T> &operator=(X&& elem);

  // print
  template <typename X>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X> &tensor);

  // equivalence
  bool operator==(Tensor const& tensor) const;
  bool operator!=(Tensor const& tensor) const { return !(*this == tensor); }

  // single val equivalence
  template <typename X>
  bool operator==(X val) const;
  template <typename X>
  bool operator!=(X val) const { return !(*this == val); }


  // Example operations that can be implemented
  Tensor<T> operator-() const;
  template <typename X>
  friend Tensor<X> operator+(const Tensor<X> &tensor_1, const Tensor<X> &tensor_2);
  template <typename X>
  friend Tensor<X> operator-(const Tensor<X> &tensor_1, const Tensor<X> &tensor_2);
  template <typename X>
  Tensor<T> operator+(X &&val) const;
  template <typename X>
  Tensor<T> operator-(X &&val) const;
  void operator-=(const Tensor<T> &tensor);
  void operator+=(const Tensor<T> &tensor);
  template <typename X>
  friend Tensor<X> operator*(const Tensor<X> &tensor_1, const Tensor<X> &tensor_2);

  // Do the tensors have the same rank and dimensions?
  template <typename X>
  friend bool DimensionsMatch(Tensor<X> const &tensor_1, const Tensor<X> &tensor_2);

  private:
  uint32_t degree_;
  uint32_t *dimensions_;
  T *elements_;

  // This flag denotes that this tensor object is the principle owner of memory
  // Smart ptr is unecessary
  bool is_owner_;

  // Hidden constructor for access expansion
  // The first argument is moved
  Tensor(uint32_t *dimensions, uint32_t degree);

  // Access Expansion
  Tensor<T> pAccessExpansion(uint32_t *indices, uint32_t curr_dim);
  template <typename... Args>
  Tensor<T> pAccessExpansion(uint32_t *dimensions, uint32_t degree, uint32_t, 
      Args...);
  uint32_t pCumulativeIndex(uint32_t *xs, uint32_t size);
}; // Tensor

// Tensor Methods

// Constructors, Destructor, Assignment
template <typename T>
Tensor<T>::Tensor() : degree_(0), dimensions_(nullptr), is_owner_(true)
{
  elements_ = new T;
}

template <typename T>
Tensor<T>::Tensor(T &&val) : degree_(0), dimensions_(nullptr), is_owner_(true)
{
  elements_ = new T(std::forward<T>(val));
}

template <typename T> Tensor<T>::Tensor(std::initializer_list<uint32_t> indices): is_owner_(true)
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
  dimensions_ = util::ArrayCopy(tensor.dimensions_, degree_);
  elements_ =
      util::ArrayCopy(tensor.elements_, util::ArrayProduct(dimensions_, degree_));
}

template <typename T>
Tensor<T>::Tensor(Tensor<T> &&tensor) : degree_(tensor.degree_), is_owner_(tensor.is_owner_)
{
  dimensions_ = util::ArrayCopy(tensor.dimensions_, degree_);
  elements_ = tensor.elements_;
  tensor.is_owner_ = false;
}

template <typename T> Tensor<T>::~Tensor()
{
  if (is_owner_)
    delete[] elements_;
  delete[] dimensions_;
}

template <typename T> Tensor<T> &Tensor<T>::operator=(const Tensor<T> &tensor)
{
  if (degree_ != tensor.degree_) 
    throw std::logic_error("Tensor::operator=(Tensor const&) Failed -- Tensors do not have equivalent rank");

  for (uint32_t i = 0; i < degree_; ++i) {
    if (dimensions_[i] != tensor.dimensions_[i]) 
      throw std::logic_error("Tensor::operator=(Tensor const&) Failed -- Tensor dimension mismatch");
  }

  uint32_t total_elems = util::ArrayProduct(tensor.dimensions_, degree_);
  for (uint32_t i = 0; i < total_elems; ++i) {
    elements_[i] = tensor.elements_[i];
  }

  return *this;
}

template <typename T> Tensor<T> &Tensor<T>::operator=(Tensor<T> &&tensor)
{
  if (degree_ != tensor.degree_) 
      throw std::logic_error("Tensor assignment failed, tensors do not have the same degree");

  for (uint32_t i = 0; i < degree_; ++i) 
    if (dimensions_[i] != tensor.dimensions_[i]) 
      throw std::logic_error("Tensor assignment failed, tensor dimension mismatch");

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
template <typename T> uint32_t Tensor<T>::dimension(uint32_t index) const
{
  if (degree_ < index || index == 0) 
    throw std::logic_error("Attempt to access invalid dimension");

  // indexing begins at 1
  return dimensions_[index - 1];
}

// Setters
template <typename T> 
template <typename X, typename>
Tensor<T> &Tensor<T>::operator=(X&& elem)
{
  if (degree_ != 0) throw std::logic_error("Tensor::operator=(T&&) Failure -- Non-zero rank");
  *elements_ = std::forward<X>(elem);
  return *this;
}

// Access to elements_
template <typename T>
Tensor<T>::Tensor(uint32_t *dimensions, uint32_t degree)
    : degree_(degree), dimensions_(dimensions) {}

template <typename T>
template <typename... Args>
Tensor<T> Tensor<T>::operator()(Args... args)
{
  uint32_t *indices = new uint32_t[degree_];
  uint32_t curr_dim = 0;
  return pAccessExpansion(indices, curr_dim, args...);
}

template <typename T>
Tensor<T> Tensor<T>::pAccessExpansion(uint32_t *indices, uint32_t curr_index)
{
  uint32_t indices_product = pCumulativeIndex(indices, curr_index);
  uint32_t *new_indices = new uint32_t[degree_ - curr_index];
  for (uint32_t i = curr_index; i < degree_; ++i) 
    new_indices[i - curr_index] = dimensions_[i];

  // ptr for new_indices is moved so delete[] is unecessary
  Tensor<T> new_tensor(new_indices, degree_ - curr_index);
  new_tensor.elements_ = elements_ + indices_product;
  new_tensor.is_owner_ = false;
  return new_tensor;
}

template <typename T>
template <typename... Args>
Tensor<T> Tensor<T>::pAccessExpansion(uint32_t *indices, uint32_t curr_index,
                                      uint32_t next_index, Args... rest)
{
  if (curr_index == degree_) throw std::logic_error("Tensor::operator(Args...) failed -- Dimension Out of Bounds");
  if (next_index > dimensions_[curr_index] || next_index == 0) throw std::logic_error("Tensor::operator(Args...) failed -- Index Out of Bounds");

  // adjust for 1 index array access
  indices[curr_index] = --next_index;
  return pAccessExpansion(indices, ++curr_index, rest...);
}

template <typename T>
uint32_t Tensor<T>::pCumulativeIndex(uint32_t *xs, uint32_t size)
{
  uint32_t total_elems = util::ArrayProduct(dimensions_, degree_);
  uint32_t cumul = 0;
  for (uint32_t i = 0; i < size; ++i) {
    total_elems /= dimensions_[i];
    cumul += xs[i] * total_elems;
  }
  return cumul;
}

template <typename T>
bool Tensor<T>::operator==(Tensor<T> const& tensor) const
{
  if (!DimensionsMatch(*this, tensor)) throw std::logic_error("Tensor::operator==(Tensor const&) Failure, rank/dimension mismatch");
  uint32_t indices_product = util::ArrayProduct(tensor.dimensions_, tensor.degree_);
  for (uint32_t i = 0; i < indices_product; ++i)
    if (elements_[i] != tensor.elements_[i]) return false;
  return true;
}

template <typename T>
template <typename X>
bool Tensor<T>::operator==(X val) const
{
  if (degree_) throw std::logic_error("Tensor::operator==(value_type) Failure -- Non-zero rank");
  return *elements_ == val;
}

template <typename T>
template <typename X>
Tensor<T> Tensor<T>::operator+(X &&val) const
{
  if (degree_) throw std::logic_error("Tensor::operator+(X) Failure -- Non-zero rank");
  return new Tensor(val + *elements_);
}

template <typename T>
template <typename X>
Tensor<T> Tensor<T>::operator-(X &&val) const
{
  if (degree_) throw std::logic_error("Tensor::operator+(X) Failure -- Non-zero rank");
  return Tensor(*elements_ - val);
}

// iostream overload implemented for debugging
template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) 
{
  // if its a scalar just return;
  if (!tensor.degree_) {
    os << tensor.elements_[0];
    return os;
  }
  uint32_t indices_product = util::ArrayProduct(tensor.dimensions_, tensor.degree_);
  uint32_t bracket_mod_arr[tensor.degree_];
  uint32_t prod = 1;
  for (uint32_t i = 0; i < tensor.degree_; ++i) {
    prod *= tensor.dimensions_[tensor.degree_ - i - 1];
    bracket_mod_arr[i] = prod;
  }
  for (size_t i = 0; i < tensor.degree_; ++i) os << '[';
  os << tensor.elements_[0] << ", ";
  for (uint32_t i = 1; i < indices_product - 1; ++i) {
    os << tensor.elements_[i];
    uint32_t num_brackets = 0;
    for (; num_brackets < tensor.degree_; ++num_brackets) {
      if (!((i + 1) % bracket_mod_arr[num_brackets]) == 0) break;
    }
    for (size_t i = 0; i < num_brackets; ++i) os << ']';
    os << ", ";
    for (size_t i = 0; i < num_brackets; ++i) os << '[';
  }
  if (indices_product - 1) os << tensor.elements_[indices_product - 1];
  for (size_t i = 0; i < tensor.degree_; ++i) os << ']';
  return os;
}

/*  EXAMPLE OPERATORS THAT CAN BE IMPLEMENTED   */

template <typename T> Tensor<T> Tensor<T>::operator-() const
{
  Tensor<T> neg_tensor = *this;
  uint32_t total_dims = util::ArrayProduct(dimensions_, degree_);
  for (uint32_t i = 0; i < total_dims; ++i) 
    neg_tensor.elements_[i] *= -1;
  return neg_tensor;
}

template <typename T> void Tensor<T>::operator+=(const Tensor<T> &tensor)
{
  if (degree_ != tensor.degree_ || !util::ArrayCompare(dimensions_, tensor.dimensions_, degree_)) 
    throw std::logic_error("Tensor::operator+=(Tensor const&) Failed :: Incompatible Dimensions");

  uint32_t total_dims = util::ArrayProduct(dimensions_, degree_);
  for (uint32_t i = 0; i < total_dims; ++i) 
    elements_[i] += tensor.elements_[i];
}

template <typename T> void Tensor<T>::operator-=(const Tensor<T> &tensor)
{
  if (degree_ != tensor.degree_ || !util::ArrayCompare(dimensions_, tensor.dimensions_, degree_)) 
    throw std::logic_error("Tensor::operator-=(Tensor const&) Failed :: Incompatible Dimensions");

  uint32_t total_dims = util::ArrayProduct(dimensions_, degree_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    elements_[i] -= tensor.elements_[i];
  }
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &tensor_1, const Tensor<T> &tensor_2)
{
  if (!DimensionsMatch(tensor_1, tensor_2))
    throw std::logic_error("operator+(Tensor, Tensor) Failed :: Incompatible Dimensions");

  Tensor<T> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) 
    new_tensor.elements_[i] = tensor_1.elements_[i] + tensor_2.elements_[i];
  return new_tensor;
}

template <typename T>
Tensor<T> operator-(const Tensor<T> &tensor_1, const Tensor<T> &tensor_2)
{
  if (!DimensionsMatch(tensor_1, tensor_2))
    throw std::logic_error("operator-(Tensor, Tensor) Failed :: Incompatible Dimensions");

  Tensor<T> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    new_tensor.elements_[i] = tensor_1.elements_[i] - tensor_2.elements_[i];
  }
  return new_tensor;
}

// Do the tensors have the same rank and dimensions?
template <typename T>
bool DimensionsMatch(Tensor<T> const &tensor_1, const Tensor<T> &tensor_2)
{
  return !(tensor_1.degree_ != tensor_2.degree_ || !util::ArrayCompare(tensor_1.dimensions_, tensor_2.dimensions_, tensor_1.degree_));
}

} // tensor

#endif // TENSORS_H_
