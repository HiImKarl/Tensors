#pragma once
#ifndef TENSOR_H_
#define TENSOR_H_
#include <iostream>
#include <algorithm>
#include <exception>
#include <utility>
#include <type_traits>
#include <numeric>
#include <functional>

// Error messages
#define NTENSOR_0CONSTRUCTOR \
  "Invalid Instantiation of N-Tensor -- Use a N-Constructor"
#define NCONSTRUCTOR_0TENSOR \
  "Invalid Instantiation of 0-Tensor -- Use a 0-Constructor"

namespace tensor {
template <typename T, uint32_t N = 0>
class Tensor {
public:
  // typedefs
  typedef T                 value_type;
  typedef T&                reference;
  typedef T const&          const_reference;
  typedef size_t            size_type;
  typedef ptrdiff_t         difference_type;

  // friend classes
  template <typename X, uint32_t M> friend class Tensor;

  // Constructors
  Tensor();
  explicit Tensor(value_type &&val);
  explicit Tensor(uint32_t const (&indices)[N]);
  Tensor(const Tensor<T, N> &tensor);
  Tensor(Tensor<T, N> &&tensor);

  // Assignment
  Tensor<T, N> &operator=(Tensor<T, N> const &tensor);
  template <typename X> Tensor<T, N> &operator=(Tensor<X, N> const &tensor);
  Tensor<T, N> &operator=(Tensor<T, N> &&tensor);

  // Destructor
  ~Tensor();

  // Getters
  constexpr uint32_t rank() const noexcept { return N; }
  uint32_t dimension(uint32_t index) const;

  // Access to data_
  template <typename... Args> decltype(auto) operator()(Args... args);

  // Setters
  template <typename X,
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type>
  Tensor<T, N> &operator=(X&& elem);

  // Print
  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

  // Equivalence
  bool operator==(Tensor<T, N> const& tensor) const;
  bool operator!=(Tensor<T, N> const& tensor) const { return !(*this == tensor); }
  template <typename X>
  bool operator==(Tensor<X, N> const& tensor) const;
  template <typename X>
  bool operator!=(Tensor<X, N> const& tensor) const { return !(*this == tensor); }

  // Single val equivalence
  template <typename X,
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type>
  bool operator==(X val) const;
  template <typename X,
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type>
  bool operator!=(X val) const { return !(*this == val); }

  // Type conversion for single element values
  operator T &();
  operator T const&() const;

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

  private:
  // Dimensions and access step
  uint32_t dimensions_[N];
  uint32_t steps_[N];

  // Data
  T *data_;

  // This flag denotes the tensor object's ownership of memory
  bool is_owner_;

  // Declare all fields of the constructor at once
  Tensor(uint32_t const *dimensions, T *elements);

  // Access expansion for operator()
  template <uint32_t M>
  Tensor<T, N - M> pAccessExpansion(uint32_t, uint32_t cumul_index);
  template <uint32_t M, typename... Args>
  Tensor<T, N - M> pAccessExpansion(uint32_t weight, uint32_t cumul_index,
    uint32_t next_index, Args...);

}; // Tensor

// Tensor Methods
// Constructors, Destructor, Assignment
template <typename T, uint32_t N>
Tensor<T, N>::Tensor() : is_owner_(true)
{
  static_assert(N == 0, NTENSOR_0CONSTRUCTOR);
  data_ = new T;
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(T &&val) : is_owner_(true)
{
  static_assert(N == 0, NTENSOR_0CONSTRUCTOR);
  data_ = new T(std::forward<T>(val));
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const (&indices)[N])
: is_owner_(true)
{
  // ill defined if N is zero
  static_assert(N != 0, NCONSTRUCTOR_0TENSOR);

  std::copy_n(indices, N, dimensions_);
  std::fill_n(steps_, N, 1);
  uint32_t num_elements =
    std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());

  data_ = new T[num_elements];
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(const Tensor<T, N> &tensor): is_owner_(true)
{
  std::copy_n(tensor.dimensions_, N, dimensions_);
  std::copy_n(tensor.steps_, N, steps_);
  size_t count =
    std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
  data_ = new T[count];
  std::copy_n(tensor.data_, count, data_);
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Tensor<T, N> &&tensor): is_owner_(tensor.is_owner_)
{
  std::copy_n(tensor.dimensions_, N, dimensions_);
  data_ = tensor.data_;
  tensor.is_owner_ = false;
}

// private constructor
template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const *dimensions, T *elements)
  : data_(elements), is_owner_(false)
{
  std::copy_n(dimensions, N, dimensions_);
}

template <typename T, uint32_t N> Tensor<T, N>::~Tensor()
{
  if (is_owner_) delete[] data_;
}

template <typename T, uint32_t N>
Tensor<T, N> &Tensor<T, N>::operator=(const Tensor<T, N> &tensor)
{
  if (!std::equal(dimensions_, dimensions_ + N, tensor.dimensions_))
      throw std::logic_error("Tensor::operator=(Tensor const&) Failed -- Tensor dimension mismatch");

  uint32_t total_elems =
      std::accumulate(tensor.dimensions_, tensor.dimensions_ + N, 1, std::multiplies<size_t>());

  for (uint32_t i = 0; i < total_elems; ++i)
    data_[i] = tensor.data_[i];

  return *this;
}

template <typename T, uint32_t N>
Tensor<T, N> &Tensor<T, N>::operator=(Tensor<T, N> &&tensor)
{
  if (!std::equal(dimensions_, dimensions_ + N, tensor.dimensions_))
    throw std::logic_error("Tensor assignment failed, tensor dimension mismatch");

  // DO NOT MOVE DATA unless boths tensors are principle owners
  if (tensor.is_owner_ && is_owner_) {
    data_ = tensor.data_;
    tensor.is_owner_ = false;
  } else {
    uint32_t total_elems =
      std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
    for (uint32_t i = 0; i < total_elems; ++i)
      data_[i] = tensor.data_[i];
  }
  return *this;
}

template <typename T, uint32_t N>
template <typename X>
Tensor<T, N> &Tensor<T, N>::operator=(Tensor<X, N> const &tensor)
{
  if (!std::equal(dimensions_, dimensions_ + N, tensor.dimensions_))
    throw std::logic_error("Tensor::operator=(Tensor const&) Failed -- Tensor dimension mismatch");

  uint32_t total_elems =
      std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
  for (uint32_t i = 0; i < total_elems; ++i)
    data_[i] = tensor.data_[i];
  return *this;
}

template <typename T, uint32_t N>
template <typename X, typename>
Tensor<T, N> &Tensor<T, N>::operator=(X&& elem)
{
  static_assert(N == 0);
  *data_ = std::forward<X>(elem);
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

template <typename T, uint32_t N>
template <typename... Args>
decltype(auto) Tensor<T, N>::operator()(Args... args)
{
  static_assert(N >= sizeof...(args),
    "Tensor::operator(Args...) -- Rank Requested Out of Bounds");
  uint32_t weight =
    std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
  return pAccessExpansion<sizeof...(args)>(weight, 0, args...);
}

template <typename T, uint32_t N>
bool Tensor<T, N>::operator==(Tensor<T, N> const& tensor) const
{
  if (!std::equal(dimensions_, dimensions_ + N, tensor.dimensions_))
    throw std::logic_error("Tensor::operator==(Tensor const&) Failure, rank/dimension mismatch");

  uint32_t indices_product =
    std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
  for (uint32_t i = 0; i < indices_product; ++i)
    if (data_[i] != tensor.data_[i]) return false;
  return true;
}

template <typename T, uint32_t N>
template <typename X>
bool Tensor<T, N>::operator==(Tensor<X, N> const& tensor) const
{
  if (!std::equal(dimensions_, dimensions_ + N, tensor.dimensions_))
    throw std::logic_error("Tensor::operator==(Tensor const&) Failure, rank/dimension mismatch");

  uint32_t indices_product =
    std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
  for (uint32_t i = 0; i < indices_product; ++i)
    if (data_[i] != tensor.data_[i]) return false;
  return true;
}

template <typename T, uint32_t N>
template <typename X, typename>
bool Tensor<T, N>::operator==(X val) const
{
  static_assert(N == 0, "Tensor::operator==(value_type) -- Non-Zero Rank");
  return *data_ == val;
}

template <typename T, uint32_t N>
template <typename X>
Tensor<T, N> Tensor<T, N>::operator+(X &&val) const
{
  static_assert(N == 0, "Tensor::operator+(X) -- Non-zero rank");
  return Tensor(val + *data_);
}

template <typename T, uint32_t N>
template <typename X>
Tensor<T, N> Tensor<T, N>::operator-(X &&val) const
{
  static_assert(N == 0, "Tensor::operator+(X) -- Non-zero rank");
  return Tensor(*data_ - val);
}

template <typename T, uint32_t N>
Tensor<T, N>::operator T &()
{
  static_assert(N == 0, "Tensor Implicit Cast to ValueType failed -- Tensor must be of rank zero to cast to a scalar");
  return *data_;
}

template <typename T, uint32_t N>
Tensor<T, N>::operator T const&() const
{
  static_assert(N == 0, "Tensor Implicit Cast to ValueType failed -- Tensor must be of rank zero to cast to a scalar");
  return *data_;
}

/*  EXAMPLE OPERATORS THAT CAN BE IMPLEMENTED   */

// possible iostream overload implemention
template <typename T, uint32_t N>
std::ostream &operator<<(std::ostream &os, const Tensor<T, N> &tensor)
{
  uint32_t indices_product =
    std::accumulate(tensor.dimensions_, tensor.dimensions_ + N, 1, std::multiplies<size_t>());
  uint32_t bracket_mod_arr[N];
  uint32_t prod = 1;
  for (uint32_t i = 0; i < N; ++i) {
    prod *= tensor.dimensions_[N - i - 1];
    bracket_mod_arr[i] = prod;
  }
  for (size_t i = 0; i < N; ++i) os << '[';
  os << tensor.data_[0] << ", ";
  for (uint32_t i = 1; i < indices_product - 1; ++i) {
    os << tensor.data_[i];
    uint32_t num_brackets = 0;
    for (; num_brackets < N; ++num_brackets) {
      if (!((i + 1) % bracket_mod_arr[num_brackets]) == 0) break;
    }
    for (size_t i = 0; i < num_brackets; ++i) os << ']';
    os << ", ";
    for (size_t i = 0; i < num_brackets; ++i) os << '[';
  }
  if (indices_product - 1) os << tensor.data_[indices_product - 1];
  for (size_t i = 0; i < N; ++i) os << ']';
  return os;
}

template <typename T, uint32_t N> Tensor<T, N> Tensor<T, N>::operator-() const
{
  Tensor<T, N> neg_tensor = *this;
  uint32_t total_dims =
    std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
  for (uint32_t i = 0; i < total_dims; ++i)
    neg_tensor.data_[i] *= -1;
  return neg_tensor;
}

template <typename T, uint32_t N>
void Tensor<T, N>::operator+=(const Tensor<T, N> &tensor)
{
  if (!std::equal(dimensions_, dimensions_ + N, tensor.dimensions_))
    throw std::logic_error("Tensor::operator+=(Tensor const&) Failed :: Incompatible Dimensions");

  uint32_t total_dims =
    std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
  for (uint32_t i = 0; i < total_dims; ++i)
    data_[i] += tensor.data_[i];
}

template <typename T, uint32_t N> void Tensor<T, N>::operator-=(const Tensor<T, N> &tensor)
{
  if (!std::equal(dimensions_, dimensions_ + N, tensor.dimensions_))
    throw std::logic_error("Tensor::operator-=(Tensor const&) Failed :: Incompatible Dimensions");

  uint32_t total_dims =
    std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
  for (uint32_t i = 0; i < total_dims; ++i) {
    data_[i] -= tensor.data_[i];
  }
}

template <typename T, uint32_t N>
Tensor<T, N> operator+(const Tensor<T, N> &tensor_1, const Tensor<T, N> &tensor_2)
{
  if (!std::equal(tensor_1.dimensions_, tensor_1.dimensions_ + N, tensor_2.dimensions_))
    throw std::logic_error("operator+(Tensor, Tensor) Failed :: Incompatible Dimensions");
  Tensor<T, N> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i)
    new_tensor.data_[i] = tensor_1.data_[i] + tensor_2.data_[i];
  return new_tensor;
}

template <typename T, uint32_t N>
Tensor<T, N> operator-(const Tensor<T, N> &tensor_1, const Tensor<T, N> &tensor_2)
{
  if (!std::equal(tensor_1.dimensions_, tensor_1.dimensions_ + N, tensor_2.dimensions_))
    throw std::logic_error("operator-(Tensor, Tensor) Failed :: Incompatible Dimensions");
  Tensor<T, N> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i)
    new_tensor.data_[i] = tensor_1.data_[i] - tensor_2.data_[i];
  return new_tensor;
}

// private methods
template <typename T, uint32_t N>
template <uint32_t M>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(uint32_t, uint32_t cumul_index)
{
  return Tensor<T, N - M>(dimensions_ + M, data_ + cumul_index);
}

template <typename T, uint32_t N>
template <uint32_t M, typename... Args>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(
 uint32_t weight, uint32_t cumul_index, uint32_t next_index, Args... rest)
{
  if (next_index > dimensions_[N - sizeof...(rest) - 1] || next_index == 0)
    throw std::logic_error("Tensor::operator(Args...) failed -- Index Out of Bounds");

  weight /= dimensions_[N - sizeof...(rest) - 1];
  // adjust for 1 index array access
  cumul_index += weight * (next_index - 1);
  return pAccessExpansion<M>(weight, cumul_index, rest...);
}
} // tensor

#undef NTENSOR_0CONSTRUCTOR
#undef NCONSTRUCTOR_0TENSOR

#endif // TENSORS_H_
