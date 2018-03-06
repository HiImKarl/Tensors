#pragma once
#ifndef TENSOR_H_
#define TENSOR_H_
#include <cstdint>
#include <cstring>
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

template <typename T, uint32_t N = 0>
class Tensor {
public:
  // typedefs
  typedef T value_type;
  template <typename X, uint32_t M> friend class Tensor;

  // Constructors
  Tensor();
  explicit Tensor(value_type &&val);
  explicit Tensor(uint32_t const (&indices)[N]);
  Tensor(const Tensor<T, N> &tensor);
  Tensor(Tensor<T, N> &&tensor);

  // Assignment
  Tensor<T, N> &operator=(Tensor<T, N> const &tensor);
  Tensor<T, N> &operator=(Tensor<T, N> &&tensor);

  // Destructor
  ~Tensor();

  // Getters
  constexpr uint32_t rank() const noexcept { return N; }
  uint32_t dimension(uint32_t index) const;

  // Access to elements_
  template <typename... Args> auto operator()(Args... args);

  // Setters
  template <typename X, 
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type, 
            typename std::remove_reference<X>::type>::value>::type>
  Tensor<T, N> &operator=(X&& elem);

  // print
  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

  // equivalence
  bool operator==(Tensor const& tensor) const;
  bool operator!=(Tensor const& tensor) const { return !(*this == tensor); }

  // single val equivalence
  template <typename X>
  bool operator==(X val) const;
  template <typename X>
  bool operator!=(X val) const { return !(*this == val); }

  // Example operations that can be implemented
  Tensor<T, N> operator-() const;
  template <typename X, uint32_t M>
  friend Tensor<X, M> operator+(Tensor<X, M> const &tensor_1, Tensor<X, M> const &tensor_2);
  template <typename X, uint32_t M>
  friend Tensor<X, M> operator-(Tensor<X, M> const &tensor_1, Tensor<X, M> const &tensor_2);
  template <typename X>
  Tensor<T, N> operator+(X &&val) const;
  template <typename X>
  Tensor<T, N> operator-(X &&val) const;
  void operator-=(const Tensor<T, N> &tensor);
  void operator+=(const Tensor<T, N> &tensor);
  template <typename X, uint32_t M>
  friend Tensor<X, M> operator*(const Tensor<X, M> &tensor_1, const Tensor<X, M> &tensor_2);

  // Do the tensors have the same dimensions?
  template <typename X, typename Y>
  friend bool DimensionsMatch(Tensor<X, N> const &tensor_1, const Tensor<Y, N> &tensor_2)
  {
    return util::ArrayCompare(tensor_1.dimensions_, tensor_2.dimensions_, N);
  }

  private:
  uint32_t dimensions_[N];
  T *elements_;

  // This flag denotes that this tensor object is the principle owner of memory
  // Smart ptr is unecessary
  bool is_owner_;

  // declare all fields of the constructor at once
  Tensor(uint32_t const (&dimensions)[N], T *elements, bool is_owner_);

  // Access Expansion for operator()
  template <uint32_t M>
  Tensor<T, N - M> pAccessExpansion(uint32_t *indices);
  template <uint32_t M, typename... Args>
  Tensor<T, N - M> pAccessExpansion(uint32_t *indices, uint32_t next_index, Args...);
  uint32_t pCumulativeIndex(uint32_t *xs, uint32_t size);
}; // Tensor

// Tensor Methods
// Constructors, Destructor, Assignment
template <typename T, uint32_t N>
Tensor<T, N>::Tensor() : is_owner_(true)
{
  static_assert(N == 0);
  elements_ = new T;
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(T &&val) : is_owner_(true)
{
  static_assert(N == 0);
  elements_ = new T(std::forward<T>(val));
}

template <typename T, uint32_t N> 
Tensor<T, N>::Tensor(uint32_t const (&indices)[N])
: is_owner_(true)
{
  // ill defined if N is zero, 
  // default constructor should be used instead
  static_assert(N != 0);

  memcpy(dimensions_, indices, sizeof(dimensions_));
  uint32_t num_elements = util::ArrayProduct(dimensions_, N);
  elements_ = new T[num_elements];
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(const Tensor<T, N> &tensor): is_owner_(true)
{
  // memcpy is safe here
  memcpy(dimensions_, tensor.dimensions_, sizeof(dimensions_));
  elements_ = util::ArrayCopy(tensor.elements_, util::ArrayProduct(dimensions_, N));
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Tensor<T, N> &&tensor): is_owner_(tensor.is_owner_)
{
  // memcpy is safe here
  memcpy(dimensions_, tensor.dimensions_, sizeof(dimensions_));
  elements_ = tensor.elements_;
  tensor.is_owner_ = false;
}

// private constructor
template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const (&dimensions)[N], T *elements, bool is_owner)
  : elements_(elements), is_owner_(is_owner)
{
  memcpy(dimensions_, dimensions, sizeof(dimensions));
}

template <typename T, uint32_t N> Tensor<T, N>::~Tensor()
{
  if (is_owner_) delete[] elements_;
}

template <typename T, uint32_t N> 
Tensor<T, N> &Tensor<T, N>::operator=(const Tensor<T, N> &tensor)
{
  for (uint32_t i = 0; i < N; ++i) {
    if (dimensions_[i] != tensor.dimensions_[i]) 
      throw std::logic_error("Tensor::operator=(Tensor const&) Failed -- Tensor dimension mismatch");
  } 
  uint32_t total_elems = util::ArrayProduct(tensor.dimensions_, N);
  for (uint32_t i = 0; i < total_elems; ++i)
    elements_[i] = tensor.elements_[i];

  return *this;
}

template <typename T, uint32_t N> Tensor<T, N> &Tensor<T, N>::operator=(Tensor<T, N> &&tensor)
{
  for (uint32_t i = 0; i < N; ++i) 
    if (dimensions_[i] != tensor.dimensions_[i]) 
      throw std::logic_error("Tensor assignment failed, tensor dimension mismatch");

  // DO NOT MOVE DATA unless boths tensors are principle owners
  if (tensor.is_owner_ && is_owner_) {
    elements_ = tensor.elements_;
    tensor.elements_ = nullptr;
    tensor.is_owner_ = false;
  } else {
    uint32_t total_elems = util::ArrayProduct(dimensions_, N);
    for (uint32_t i = 0; i < total_elems; ++i)
      elements_[i] = tensor.elements_[i];
  }
  return *this;
}

// Getters
template <typename T, uint32_t N> 
uint32_t Tensor<T, N>::dimension(uint32_t index) const 
{
  if (N < index || index == 0) 
    throw std::logic_error("Attempt to access invalid dimension");

  // indexing begins at 1
  return dimensions_[index - 1];
}

// Scalar setter
template <typename T, uint32_t N> 
template <typename X, typename>
Tensor<T, N> &Tensor<T, N>::operator=(X&& elem)
{
  static_assert(N == 0);
  *elements_ = std::forward<X>(elem);
  return *this;
}

template <typename T, uint32_t N>
template <typename... Args>
auto Tensor<T, N>::operator()(Args... args) 
{
  static_assert(N >= sizeof...(args), "Tensor::operator(Args...) -- Rank Requested Out of Bounds");
  uint32_t *indices = new uint32_t[sizeof...(args)];
  return pAccessExpansion<sizeof...(args)>(indices, args...);
}

template <typename T, uint32_t N>
template <uint32_t M>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(uint32_t *indices)
{
  uint32_t indices_product = pCumulativeIndex(indices, N);
  uint32_t new_indices[N -M];
  for (uint32_t i = M; i < N; ++i) new_indices[i - M] = dimensions_[i];
  // ptr for new_indices is moved so delete[] is unecessary
  Tensor<T, N - M> new_tensor(new_indices, elements_ + indices_product, false);
  return new_tensor;
}

template <typename T, uint32_t N>
template <uint32_t M, typename... Args>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(uint32_t *indices, uint32_t next_index, Args... rest)
{
  static_assert(N > sizeof...(rest));
  if (next_index > dimensions_[N - sizeof...(rest) - 1] || next_index == 0) throw std::logic_error("Tensor::operator(Args...) failed -- Index Out of Bounds");

  // adjust for 1 index array access
  indices[N - sizeof...(rest) - 1] = --next_index;
  return pAccessExpansion<M>(indices, rest...);
}

template <typename T, uint32_t N>
uint32_t Tensor<T, N>::pCumulativeIndex(uint32_t *xs, uint32_t size)
{
  uint32_t total_elems = util::ArrayProduct(dimensions_, N);
  uint32_t cumul = 0;
  for (uint32_t i = 0; i < size; ++i) {
    total_elems /= dimensions_[i];
    cumul += xs[i] * total_elems;
  }
  return cumul;
}

template <typename T, uint32_t N>
bool Tensor<T, N>::operator==(Tensor<T, N> const& tensor) const
{
  if (!DimensionsMatch(*this, tensor)) throw std::logic_error("Tensor::operator==(Tensor const&) Failure, rank/dimension mismatch");
  uint32_t indices_product = util::ArrayProduct(dimensions_, N);
  for (uint32_t i = 0; i < indices_product; ++i)
    if (elements_[i] != tensor.elements_[i]) return false;
  return true;
}

template <typename T, uint32_t N>
template <typename X>
bool Tensor<T, N>::operator==(X val) const
{
  static_assert(N == 0, "Tensor::operator==(value_type) -- Non-Zero Rank");
  return *elements_ == val;
}

template <typename T, uint32_t N>
template <typename X>
Tensor<T, N> Tensor<T, N>::operator+(X &&val) const
{
  static_assert(N == 0, "Tensor::operator+(X) -- Non-zero rank");
  return Tensor(val + *elements_);
}

template <typename T, uint32_t N>
template <typename X>
Tensor<T, N> Tensor<T, N>::operator-(X &&val) const
{
  static_assert(N == 0, "Tensor::operator+(X) -- Non-zero rank");
  return Tensor(*elements_ - val);
}

/*  EXAMPLE OPERATORS THAT CAN BE IMPLEMENTED   */

// possible iostream overload implemention 
template <typename T, uint32_t N>
std::ostream &operator<<(std::ostream &os, const Tensor<T, N> &tensor) 
{
  uint32_t indices_product = util::ArrayProduct(tensor.dimensions_, N);
  uint32_t bracket_mod_arr[N];
  uint32_t prod = 1;
  for (uint32_t i = 0; i < N; ++i) {
    prod *= tensor.dimensions_[N - i - 1];
    bracket_mod_arr[i] = prod;
  }
  for (size_t i = 0; i < N; ++i) os << '[';
  os << tensor.elements_[0] << ", ";
  for (uint32_t i = 1; i < indices_product - 1; ++i) {
    os << tensor.elements_[i];
    uint32_t num_brackets = 0;
    for (; num_brackets < N; ++num_brackets) {
      if (!((i + 1) % bracket_mod_arr[num_brackets]) == 0) break;
    }
    for (size_t i = 0; i < num_brackets; ++i) os << ']';
    os << ", ";
    for (size_t i = 0; i < num_brackets; ++i) os << '[';
  }
  if (indices_product - 1) os << tensor.elements_[indices_product - 1];
  for (size_t i = 0; i < N; ++i) os << ']';
  return os;
}

template <typename T, uint32_t N> Tensor<T, N> Tensor<T, N>::operator-() const
{
  Tensor<T, N> neg_tensor = *this;
  uint32_t total_dims = util::ArrayProduct(dimensions_, N);
  for (uint32_t i = 0; i < total_dims; ++i) 
    neg_tensor.elements_[i] *= -1;
  return neg_tensor;
}

template <typename T, uint32_t N> 
void Tensor<T, N>::operator+=(const Tensor<T, N> &tensor)
{
  if (!util::ArrayCompare(dimensions_, tensor.dimensions_, N)) 
    throw std::logic_error("Tensor::operator+=(Tensor const&) Failed :: Incompatible Dimensions");

  uint32_t total_dims = util::ArrayProduct(dimensions_, N);
  for (uint32_t i = 0; i < total_dims; ++i) 
    elements_[i] += tensor.elements_[i];
}

template <typename T, uint32_t N> void Tensor<T, N>::operator-=(const Tensor<T, N> &tensor)
{
  if (!util::ArrayCompare(dimensions_, tensor.dimensions_, N)) 
    throw std::logic_error("Tensor::operator-=(Tensor const&) Failed :: Incompatible Dimensions");

  uint32_t total_dims = util::ArrayProduct(dimensions_, N);
  for (uint32_t i = 0; i < total_dims; ++i) {
    elements_[i] -= tensor.elements_[i];
  }
}

template <typename T, uint32_t N>
Tensor<T, N> operator+(const Tensor<T, N> &tensor_1, const Tensor<T, N> &tensor_2)
{
  if (!DimensionsMatch(tensor_1, tensor_2))
    throw std::logic_error("operator+(Tensor, Tensor) Failed :: Incompatible Dimensions");
  Tensor<T, N> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) 
    new_tensor.elements_[i] = tensor_1.elements_[i] + tensor_2.elements_[i];
  return new_tensor;
}

template <typename T, uint32_t N>
Tensor<T, N> operator-(const Tensor<T, N> &tensor_1, const Tensor<T, N> &tensor_2)
{
  if (!DimensionsMatch(tensor_1, tensor_2))
    throw std::logic_error("operator-(Tensor, Tensor) Failed :: Incompatible Dimensions");
  Tensor<T, N> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) 
    new_tensor.elements_[i] = tensor_1.elements_[i] - tensor_2.elements_[i];
  return new_tensor;
}

} // tensor

#endif // TENSORS_H_
