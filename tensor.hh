#pragma once
#ifndef TENSOR_H_
#define TENSOR_H_
#include <cstddef>
#include <iostream>
#include <algorithm>
#include <exception>
#include <utility>
#include <type_traits>
#include <numeric>
#include <functional>
#include <memory>
#include <cassert>
#include <string>

/* FIXME -- Remove debug macros */
#ifndef _NDEBUG

#define GET_MACRO(_1,_2,_3,NAME,...) NAME

#define ARRAY_SIZE(x) sizeof(x)/sizeof(x[0])
#define PRINT(x) std::cout << x << '\n';
#define PRINTV(x) std::cout << #x << ": " << x << '\n';
#define PRINT_ARRAY1(x) \
  std::cout << #x << ": "; \
  for (size_t i = 0; i < ARRAY_SIZE(x); ++i) std::cout << x[i] << ' '; \
  std::cout << '\n';

#define PRINT_ARRAY2(x, n) \
  std::cout << #x << ": "; \
  for (size_t i = 0; i < n; ++i) std::cout << x[i] << ' '; \
  std::cout << '\n';

#define PRINT_ARRAY(...) \
  GET_MACRO(__VA_ARGS__, PRINT_ARRAY1, PRINT_ARRAY2)(__VA_ARGS__)

#endif

/* ---------------- Error Messages ---------------- */

// Constructors
#define NTENSOR_0CONSTRUCTOR \
  "Invalid Instantiation of N-Tensor -- Use a N-Constructor"
#define NCONSTRUCTOR_0TENSOR \
  "Invalid Instantiation of 0-Tensor -- Use a 0-Constructor"
#define NELEMENTS \
  "Incorrect number of elements provided -- "\

// Out of bounds
#define DIMENSION_INVALID(METHOD) \
  METHOD " Failed -- Attempt To Access Invalid Dimension"
#define RANK_OUT_OF_BOUNDS(METHOD) \
  METHOD " Failed -- Rank requested out of bounds"
#define INDEX_OUT_OF_BOUNDS(METHOD) \
  METHOD " Failed -- Index requested out of bounds"
#define ZERO_INDEX(METHOD) \
  METHOD " Failed -- Tensors are indexed beginning at 1"

// Slicing
#define SLICES_EMPTY \
  "Tensor::slice(Indices...) Failed -- At least one dimension must be sliced"
#define SLICES_OUT_OF_BOUNDS \
  "Tensor::slice(Indices...) Failed -- Slices out of bounds"
#define SLICE_INDICES_REPEATED \
  "Tensor::slice(Indices...) Failed -- Repeated slice indices"
#define SLICE_INDICES_DESCENDING \
  "Tensor::slice(Indices...) Failed -- Slice indices must be listed in ascending order"

// Arithmetic Operations
#define RANK_MISMATCH(METHOD) \
  METHOD "Failed -- Tensors have different ranks"
#define DIMENSION_MISMATCH(METHOD) \
  METHOD " Failed -- Tensor have different dimensions"
#define INNER_DIMENSION_MISMATCH(METHOD) \
  METHOD " Failed -- Tensors have different inner dimensions"
#define SCALAR_TENSOR_MULT(METHOD) \
  METHOD " Failed -- Cannot multiple tensors with scalars"

/* ---------------- Debug Messages --------------- */

#define DEBUG \
  "This static assertion should never fire"
#define OVERLOAD_RESOLUTION(METHOD) \
  METHOD " :: Overload resolution :: " DEBUG

/* ------------------ Lambdas -------------------- */

#define _ARRAY_DELETER(type) \
  ([](type *ptr) { delete[] ptr; })

/* ----------------------------------------------- */

namespace tensor {

/* --------------- Forward Declerations --------------- */
template <uint32_t N> class Shape;
template <typename T, uint32_t N = 0> class Tensor;
template <typename LHS, typename RHS> class BinaryAdd;
template <typename LHS, typename RHS> class BinarySub;
template <typename LHS, typename RHS> class BinaryMul;

/* ----------------- Template Meta-Patterns ----------------- */

// does this really not exist in the standard library?
template <bool B1, bool B2>
struct LogicalAnd { static bool const value = B1 && B2; };

template <typename T>
struct IsTensor { static bool const value = false; };

template <>
template <typename T, uint32_t N>
struct IsTensor<Tensor<T, N>> { static bool const value = true; };

template <typename T>
struct IsScalar { static bool const value = true; };

template <>
template <typename T>
struct IsScalar<Tensor<T, 0>> { static bool const value = true; };

template <>
template <typename T, uint32_t N>
struct IsScalar<Tensor<T, N>> { static bool const value = false; };

template <typename T>
struct ValueToTensor {
  ValueToTensor(T &&val): value(std::forward<T>(val)) {}
  Tensor<T, 0> value;
};

template <>
template <typename T, uint32_t N>
struct ValueToTensor<Tensor<T, N>> {
  ValueToTensor(Tensor<T, N> const &val): value(val) {}
  Tensor<T, N> const &value;
};

template <>
template <typename LHS, typename RHS>
struct ValueToTensor<BinaryAdd<LHS, RHS>> {
  ValueToTensor(BinaryAdd<LHS, RHS> const &val): value(val) {}
  BinaryAdd<LHS, RHS> const &value;
};

template <>
template <typename LHS, typename RHS>
struct ValueToTensor<BinarySub<LHS, RHS>> {
  ValueToTensor(BinarySub<LHS, RHS> const &val): value(val) {}
  BinarySub<LHS, RHS> const &value;
};

template <>
template <typename LHS, typename RHS>
struct ValueToTensor<BinaryMul<LHS, RHS>> {
  ValueToTensor(BinaryMul<LHS, RHS> const &val): value(val) {}
  BinaryMul<LHS, RHS> const &value;
};

/* ---------------------------------------------------------- */

template <typename NodeType>
struct Expression {
  inline NodeType &self() { return *static_cast<NodeType *>(this); }
  inline NodeType const &self() const { return *static_cast<NodeType const*>(this); }
};

// Tensor Shape
template <uint32_t N> /*@Shape<N>*/
class Shape {
public:
  /* -------------------- typedefs -------------------- */
  typedef size_t                    size_type;
  typedef ptrdiff_t                 difference_type;
  typedef Shape<N>                  self_type;

  /* ----------------- friend classes ----------------- */

  template <typename X, uint32_t M> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ------------------ Constructors ------------------ */

  explicit Shape(uint32_t const (&dimensions)[N]);
  Shape(Shape<N> const &shape);

  /* ------------------ Assignment -------------------- */

  Shape<N> &operator=(Shape<N> const &shape);

  /* -------------------- Getters --------------------- */

  constexpr static uint32_t rank() { return N; }
  uint32_t &operator[](uint32_t index);
  uint32_t operator[](uint32_t index) const;

  /* -------------------- Equality -------------------- */

  // Equality only compares against dimensions, not step size
  bool operator==(Shape<N> const& shape) const noexcept;
  bool operator!=(Shape<N> const& shape) const noexcept { return !(*this == shape); }
  template <uint32_t M>
  bool operator==(Shape<M> const& shape) const noexcept;
  template <uint32_t M>
  bool operator!=(Shape<M> const& shape) const noexcept { return !(*this == shape); }

  /* ----------------- Expressions ------------------ */

  template <typename X, typename Y, uint32_t M1, uint32_t M2>
  friend Tensor<X, M1 + M2 - 2> Multiply(Tensor<X, M1> const& tensor_1, Tensor<Y, M2> const& tensor_2);

  /* ------------------- Utility -------------------- */

  uint32_t IndexProduct() const noexcept;

  /* -------------------- Print --------------------- */

  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, Tensor<X, M> const&tensor);
  template <uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, Shape<M> const &shape);

private:
  /* ------------------ Data ------------------ */

  uint32_t dimensions_[N];

  /* --------------- Constructor --------------- */

  // FIXME :: dumb hack to avoid ambiguous overload
  Shape(uint32_t const *dimensions, int);
  Shape() = default;
};

template <uint32_t N>
Shape<N>::Shape(uint32_t const (&dimensions)[N])
{
  std::copy_n(dimensions, N, dimensions_);
}

template <uint32_t N>
Shape<N>::Shape(Shape const &shape)
{
  std::copy_n(shape.dimensions_, N, dimensions_);
}

template <uint32_t N>
Shape<N>::Shape(uint32_t const *dimensions, int)
{
  std::copy_n(dimensions, N, dimensions_);
}

template <uint32_t N>
Shape<N> &Shape<N>::operator=(Shape<N> const &shape)
{
  std::copy_n(shape.dimensions_, N, dimensions_);
  return *this;
}

template <uint32_t N>
uint32_t &Shape<N>::operator[](uint32_t index)
{
  if (N < index || index == 0)
    throw std::logic_error(DIMENSION_INVALID("Tensor::dimension(uint32_t)"));

  // indexing begins at 1
  return dimensions_[index - 1];
}

template <uint32_t N>
uint32_t Shape<N>::operator[](uint32_t index) const
{
  if (N < index || index == 0)
    throw std::logic_error(DIMENSION_INVALID("Tensor::dimension(uint32_t)"));

  // indexing begins at 1
  return dimensions_[index - 1];
}

template <uint32_t N>
bool Shape<N>::operator==(Shape<N> const& shape) const noexcept
{
  return std::equal(dimensions_, dimensions_ + N, shape.dimensions_);
}

template <uint32_t N>
template <uint32_t M>
bool Shape<N>::operator==(Shape<M> const& shape) const noexcept
{
  if (N != M) return false;
  return std::equal(dimensions_, dimensions_ + N, shape.dimensions_);
}

template <uint32_t N>
uint32_t Shape<N>::IndexProduct() const noexcept
{
  return std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
}

template <uint32_t N>
std::ostream &operator<<(std::ostream &os, const Shape<N> &shape)
{
  os << "S{";
  for (size_t i = 0; i < N - 1; ++i) os << shape.dimensions_[i] << ", ";
  os << shape.dimensions_[N - 1] << "}";
  return os;
}

// Is this really necessary?
template <uint32_t N>
class Indices {
public:
  explicit Indices(uint32_t const (&indices)[N]);
  uint32_t operator[](size_t index) const;
private:
  uint32_t indices_[N];
};

template <uint32_t N>
Indices<N>::Indices(uint32_t const (&indices)[N])
{
  std::copy_n(indices, N, indices_);
}

template <uint32_t N>
uint32_t Indices<N>::operator[](size_t index) const
{
  if (index > N) throw std::logic_error(DIMENSION_INVALID("Indices::operator[]"));
  return indices_[index];
}

/**
 *  Tensor Object with compile time rank information
 */
template <typename T, uint32_t N>
class Tensor: public Expression<Tensor<T, N>> { /*@Tensor<T, N>*/
public:

  /* ------------------ Type Definitions --------------- */
  typedef T                                     value_type;
  typedef T&                                    reference;
  typedef T const&                              const_reference;
  typedef size_t                                size_type;
  typedef ptrdiff_t                             difference_type;
  typedef Tensor<T, N>                          self_type;

  /* ----------------- Friend Classes ----------------- */

  template <typename X, uint32_t M> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ------------------ Constructors ------------------ */

  explicit Tensor(uint32_t const (&indices)[N]);
  Tensor(uint32_t const (&dimensions)[N], T const &value);
  explicit Tensor(Shape<N> shape);
  Tensor(Shape<N> shape, T const &value);
  Tensor(Tensor<T, N> const &tensor);
  template <typename NodeType>
  Tensor(Expression<NodeType> const& expression);

  /* ------------------- Assignment ------------------- */

  Tensor<T, N> &operator=(Tensor<T, N> const &tensor);
  template <typename NodeType>
  Tensor<T, N> &operator=(Expression<NodeType> const &rhs);

  /* ----------------- Getters ----------------- */

  constexpr static uint32_t rank() { return N; }
  uint32_t dimension(uint32_t index) const { return shape_[index]; }
  Shape<N> shape() const noexcept { return shape_; }

  /* ------------------ Access To Data ----------------- */

  template <typename... Indices>
  Tensor<T, N - sizeof...(Indices)> operator()(Indices... args);
  template <typename... Indices>
  Tensor<T, N - sizeof...(Indices)> const operator()(Indices... args) const;

  // Access with containers
  // The container must have compile time fixed size
  // accessible through a size() method
  template <uint32_t M>
  Tensor<T, N - M> operator[](Indices<M> const &indices);

  // slicing
  template <uint32_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices)> slice(Indices... indices);
  template <uint32_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices)> const slice(Indices... indices) const;

  /* -------------------- Expressions ------------------- */

  template <typename X, typename Y, uint32_t M>
  friend Tensor<X, M> Add(Tensor<X, M> const& tensor_1, Tensor<Y, M> const& tensor_2);
  template <typename X, typename Y, uint32_t M>
  friend Tensor<X, M> Subtract(Tensor<X, M> const& tensor_1, Tensor<Y, M> const& tensor_2);
  template <typename X, typename Y, uint32_t M1, uint32_t M2>
  friend Tensor<X, M1 + M2 - 2> Multiply(Tensor<X, M1> const& tensor_1, Tensor<Y, M2> const& tensor_2);
  Tensor<T, N> operator-() const;

  /* ------------------ Print to ostream --------------- */

  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

  /* -------------------- Equivalence ------------------ */

  bool operator==(Tensor<T, N> const& tensor) const;
  bool operator!=(Tensor<T, N> const& tensor) const { return !(*this == tensor); }
  template <typename X>
  bool operator==(Tensor<X, N> const& tensor) const;
  template <typename X>
  bool operator!=(Tensor<X, N> const& tensor) const { return !(*this == tensor); }

  /* ----------------- Useful Functions ---------------- */

  Tensor<T, N> copy() const;
  template <typename U, uint32_t M, typename Container>
  friend void Fill(Tensor<U, M> &tensor, Container const &container);

  /* -------------------- Iterators --------------------- */

  class Iterator { /*@Iterator<T, N>*/
  public:
    Tensor<T, N> operator*();
    Tensor<T, N> const operator*() const;
    Iterator operator++(int);
    Iterator &operator++();
    Iterator operator--(int);
    Iterator &operator--();
    bool operator==(Iterator const &it) const { return (it.data_ == this->data_); }
    bool operator!=(Iterator const &it) const { return !(it == *this); }
  private:
    Iterator(Tensor<T, N + 1> const &tensor, uint32_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    uint32_t strides_[N];
    value_type *data_;
    std::shared_ptr<T> ref_;

    /**
     * Step size of the underlying data pointer per increment
     */
    uint32_t stride_;
  };

  typename Tensor<T, N - 1>::Iterator begin(uint32_t index);
  typename Tensor<T, N - 1>::Iterator end(uint32_t index);

private:

  /* ----------------- Data ---------------- */

  Shape<N> shape_;
  uint32_t strides_[N];
  value_type *data_;
  std::shared_ptr<T> ref_;

  /* --------------- Getters --------------- */

  uint32_t const *strides() const noexcept { return strides_; }

  /* ------------- Expansion for operator()() ------------- */

  // Expansion
  template <uint32_t M>
  Tensor<T, N - M> pAccessExpansion(uint32_t cumul_index);
  template <uint32_t M, typename... Indices>
  Tensor<T, N - M> pAccessExpansion(uint32_t cumul_index, uint32_t next_index, Indices...);

  /* ------------- Expansion for slice() ------------- */

  // Expansion
  template <uint32_t M, typename... Indices>
  Tensor<T, N - M> pSliceExpansion(uint32_t * placed_indices, uint32_t array_index, uint32_t next_index, Indices... indices);
  template <uint32_t M>
  Tensor<T, N - M> pSliceExpansion(uint32_t * placed_indices, uint32_t);

  // Index checking and placement
  template <uint32_t I1, uint32_t I2, uint32_t... Indices>
  static void pSliceIndex(uint32_t *placed_indices);
  template <uint32_t I1>
  static void pSliceIndex(uint32_t *placed_indices);

  /* ----------------- Utility -------------------- */

  // Data mapping
  template <uint32_t M>
  void pUpdateQuotas(uint32_t (&dim_quotas)[M], uint32_t quota_offset = 0,
      uint32_t offset = 0) const;
  template <uint32_t M>
  uint32_t pEvaluateIndex(uint32_t const (&dim_quotas)[M], uint32_t offset = 0) const;
  void pMap(std::function<void(T *lhs)> const &fn);
  template <typename X>
  void pUnaryMap(Tensor<X, N> const &tensor, std::function<void(T *lhs, X *rhs)> const &fn);
  template <typename X, typename Y>
  void pBinaryMap(Tensor<X, N> const &tensor_1, Tensor<Y, N> const &tensor_2,
      std::function<void(T *lhs, X *rhs1, Y *rhs2)> const& fn);

  // allocate new space and copy data
  value_type * pDuplicateData() const;

  // Initialize strides :: DIMENSIONS MUST BE INITIALIZED FIRST
  void pInitializeSteps();

  // Declare all fields in jthe constructor
  Tensor(uint32_t const *dimensions, uint32_t const *strides, T *data, std::shared_ptr<T> &&_ref);

}; // Tensor

/* ----------------------------- Constructors ------------------------- */

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const (&dimensions)[N])
  : shape_(Shape<N>(dimensions)), data_(new T[shape_.IndexProduct()]),
  ref_(data_, _ARRAY_DELETER(T))
{ pInitializeSteps(); }

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const (&dimensions)[N], T const& value)
  : shape_(Shape<N>(dimensions)), data_(new T[shape_.IndexProduct()]),
  ref_(data_, _ARRAY_DELETER(T))
{
  pInitializeSteps();
  std::function<void(T *)> allocate = [&value](T *x) -> void { *x = value; };
  pMap(allocate);
}

template <typename T, uint32_t N, typename Container>
void Fill(Tensor<T, N> &tensor, Container const &container)
{
  uint32_t cumul_sum = tensor.shape_.IndexProduct();
  if (container.size() != cumul_sum)
    throw std::logic_error(NELEMENTS);
  auto it = container.begin();
  std::function<void(T *)> allocate = [&it](T *x) -> void
  {
    *x = *it;
    ++it;
  };
  tensor.pMap(allocate);
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Shape<N> shape)
  : shape_(Shape<N>(shape)), data_(new T[shape.IndexProduct()]), ref_(data_, _ARRAY_DELETER(T))
{ pInitializeSteps(); }

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Shape<N> shape, T const &value)
  : shape_(Shape<N>(shape)), data_(new T[shape.IndexProduct()]), ref_(data_, _ARRAY_DELETER(T))
{
  pInitializeSteps();
  std::function<void(T *)> allocate = [&value](T *x) -> void { *x = value; };
  pMap(allocate);
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Tensor<T, N> const &tensor)
  : shape_(tensor.shape_), data_(tensor.data_), ref_(tensor.ref_)
{
  std::copy_n(tensor.strides_, N, strides_);
}

template <typename T, uint32_t N>
template <typename NodeType>
Tensor<T, N>::Tensor(Expression<NodeType> const& expression)
{
  auto result = expression.self()();
  std::copy_n(result.strides_, N, strides_);
  shape_ = result.shape_;
  data_ = result.data_;
  ref_ = std::move(result.ref_);
}

template <typename T, uint32_t N>
Tensor<T, N> &Tensor<T, N>::operator=(const Tensor<T, N> &tensor)
{
  if (shape_ != tensor.shape_)
      throw std::logic_error(DIMENSION_MISMATCH("Tensor::operator=(Tensor const&)"));
  std::function<void(T*, T*)> fn = [](T *x, T *y) -> void { *x = *y; };
  pUnaryMap(tensor, fn);
  return *this;
}

template <typename T, uint32_t N>
template <typename NodeType>
Tensor<T, N> &Tensor<T, N>::operator=(Expression<NodeType> const &rhs)
{
  auto tensor = rhs.self()();
  if (shape_ != tensor.shape_)
    throw std::logic_error(DIMENSION_MISMATCH("Tensor::operator=(Tensor const&)"));
  std::function<void(T *, typename NodeType::value_type*)> fn =
    [](T *x, typename NodeType::value_type *y) -> void { *x = *y; };
  pUnaryMap(tensor, fn);
  return *this;
}

template <typename T, uint32_t N>
template <typename... Indices>
Tensor<T, N - sizeof...(Indices)> Tensor<T, N>::operator()(Indices... args)
{
  static_assert(N >= sizeof...(args), RANK_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));
  return pAccessExpansion<sizeof...(args)>(0, args...);
}

template <typename T, uint32_t N>
template <typename... Indices>
Tensor<T, N - sizeof...(Indices)> const Tensor<T, N>::operator()(Indices... args) const
{
  return (*const_cast<self_type*>(this))(args...);
}

template <typename T, uint32_t N>
template <uint32_t M>
Tensor<T, N - M> Tensor<T, N>::operator[](Indices<M> const &indices)
{
  uint32_t cumul_index = 0;
  for (size_t i = 0; i < M; ++i)
    cumul_index += strides_[N - i - 1] * (indices[i] - 1);
  return Tensor<T, N - M>(shape_.dimensions_ + M, strides_ + M,  data_ + cumul_index, std::shared_ptr<T>(ref_));
}

template <typename T, uint32_t N>
template <uint32_t... Slices, typename... Indices>
Tensor<T, sizeof...(Slices)> Tensor<T, N>::slice(Indices... indices)
{
  static_assert(sizeof...(Slices));
  static_assert(N == sizeof...(Slices) + sizeof...(indices), SLICES_OUT_OF_BOUNDS);
  uint32_t placed_indices[N];
  // Initially fill the array with 1s
  // place 0s where the indices are sliced
  std::fill_n(placed_indices, N, 1);
  this->pSliceIndex<Slices...>(placed_indices);
  uint32_t index = 0;
  for (; index < N && !placed_indices[index]; ++index);
  return pSliceExpansion<sizeof...(indices)>(placed_indices, index, indices...);
}

template <typename T, uint32_t N>
template <uint32_t... Slices, typename... Indices>
Tensor<T, sizeof...(Slices)> const Tensor<T, N>::slice(Indices... indices) const
{
  return const_cast<self_type*>(this)->slice<Slices...>(indices...);
}

template <typename T, uint32_t N>
bool Tensor<T, N>::operator==(Tensor<T, N> const& tensor) const
{
  if (shape_ != tensor.shape_)
    throw std::logic_error(DIMENSION_MISMATCH("Tensor::operator==(Tensor const&)"));

  uint32_t indices_product = shape_.IndexProduct();
  for (uint32_t i = 0; i < indices_product; ++i)
    if (data_[i] != tensor.data_[i]) return false;
  return true;
}

template <typename T, uint32_t N>
template <typename X>
bool Tensor<T, N>::operator==(Tensor<X, N> const& tensor) const
{
  if (shape_ != tensor.shape_)
    throw std::logic_error(DIMENSION_MISMATCH("Tensor::operator==(Tensor const&)"));

  uint32_t indices_product = shape_.IndexProduct();
  for (uint32_t i = 0; i < indices_product; ++i)
    if (data_[i] != tensor.data_[i]) return false;
  return true;
}

// possible iostream overload implemention
template <typename T, uint32_t N>
std::ostream &operator<<(std::ostream &os, const Tensor<T, N> &tensor)
{
  auto add_brackets = [&os](uint32_t n, bool left)
  {
    for (uint32_t i = 0; i < n; ++i) os << (left ?'[' : ']');
  };
  uint32_t cumul_index = tensor.shape_.IndexProduct();
  uint32_t dim_quotas[N];
  std::copy_n(tensor.shape_.dimensions_, N, dim_quotas);

  add_brackets(N, true); // opening brackets
  os << tensor.data_[0]; // first element
  for (size_t i = 0; i < cumul_index - 1; ++i) {
    uint32_t dim_index = N, bracket_count = 0;
    bool propogate = true;
    // find the correct index to "step" to
    while (dim_index && propogate) {
      --dim_quotas[dim_index - 1];
      ++bracket_count;
      if (!dim_quotas[dim_index - 1])
        dim_quotas[dim_index - 1] = tensor.shape_.dimensions_[dim_index - 1];
      else
        propogate = false;
      --dim_index;
    }
    uint32_t index = tensor.pEvaluateIndex(dim_quotas);
    add_brackets(bracket_count - 1, false);
    os << ", ";
    add_brackets(bracket_count - 1, true);
    os << tensor.data_[index];
  }
  add_brackets(N, false); // closing brackets
  return os;
}

// private methods
template <typename T, uint32_t N>
template <uint32_t M>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(uint32_t cumul_index)
{
  return Tensor<T, N - M>(shape_.dimensions_ + M, strides_ + M, data_ + cumul_index, std::shared_ptr<T>(ref_));
}

template <typename T, uint32_t N>
template <uint32_t M, typename... Indices>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(
 uint32_t cumul_index, uint32_t next_index, Indices... rest)
{
  if (next_index > shape_.dimensions_[M - sizeof...(rest) - 1] || next_index == 0)
    throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));

  // adjust for 1 index array access
  cumul_index += strides_[M - sizeof...(rest) - 1] * (next_index - 1);
  return pAccessExpansion<M>(cumul_index, rest...);
}

/* ------------- Slice Expansion ------------- */

template <typename T, uint32_t N>
template <uint32_t I1, uint32_t I2, uint32_t... Indices>
void Tensor<T, N>::pSliceIndex(uint32_t *placed_indices)
{
  static_assert(N >= I1, INDEX_OUT_OF_BOUNDS("Tensor::Slice(Indices...)"));
  static_assert(I1 != I2, SLICE_INDICES_REPEATED);
  static_assert(I1 < I2, SLICE_INDICES_DESCENDING);
  static_assert(I1, ZERO_INDEX("Tensor::Slice(Indices...)"));
  placed_indices[I1 - 1] = 0;
  pSliceIndex<I2, Indices...>(placed_indices);
}

template <typename T, uint32_t N>
template <uint32_t Index>
void Tensor<T, N>::pSliceIndex(uint32_t *placed_indices)
{
  static_assert(N >= Index, INDEX_OUT_OF_BOUNDS("Tensor::Slice(Indices...)"));
  static_assert(Index, ZERO_INDEX("Tensor::Slice(Indices...)"));
  placed_indices[Index - 1] = 0;
}

template <typename T, uint32_t N>
template <uint32_t M, typename... Indices>
Tensor<T, N - M> Tensor<T, N>::pSliceExpansion(uint32_t * placed_indices, uint32_t array_index, uint32_t next_index, Indices... indices)
{
  if (shape_.dimensions_[array_index] < next_index)
    throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor::Slice(Indices...)"));
  if (!next_index)
    throw std::logic_error(ZERO_INDEX("Tensor::Slice(Indices...)"));
  placed_indices[array_index] = next_index;
  ++array_index;
  for (; array_index < N && !placed_indices[array_index]; ++array_index);
  return pSliceExpansion<M>(placed_indices, array_index, indices...);
}

template <typename T, uint32_t N>
template <uint32_t M>
Tensor<T, N - M> Tensor<T, N>::pSliceExpansion(uint32_t *placed_indices, uint32_t)
{
  uint32_t offset = 0;
  uint32_t dimensions[N - M];
  uint32_t strides[N - M];
  uint32_t array_index = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (placed_indices[i]) {
      offset += (placed_indices[i] - 1) * strides_[i]; // adjust for 1-based indexing
    } else {
      strides[array_index] = strides_[i];
      dimensions[array_index] = shape_.dimensions_[i];
      ++array_index;
    }
  }
  return Tensor<T, N - M>(dimensions, strides, data_ + offset, std::shared_ptr<T>(ref_));
}

/* ------------ Utility Methods ------------ */

template <typename T, uint32_t N>
T * Tensor<T, N>::pDuplicateData() const
{
  uint32_t count = shape_.IndexProduct();
  T * data = new T[count];
  std::copy_n(this->data_, count, data);
  return data;
}

// Update the quotas after one iterator increment
template <typename T, uint32_t N>
template <uint32_t M>
void Tensor<T, N>::pUpdateQuotas(uint32_t (&dim_quotas)[M], uint32_t quota_offset,
    uint32_t offset) const
{
  uint32_t dim_index = M;
  bool propogate = true;
  while (dim_index && propogate) {
    --dim_quotas[dim_index - 1];
    --dim_index;
    if (!dim_quotas[dim_index - 1 + quota_offset])
      dim_quotas[dim_index - 1 + quota_offset] = shape_.dimensions_[dim_index - 1 + quota_offset + offset];
    else
      propogate = false;
  }
}

// Obtain absolute data offset from dim quotas
template <typename T, uint32_t N>
template <uint32_t M>
uint32_t Tensor<T, N>::pEvaluateIndex(uint32_t const (&dim_quotas)[M], uint32_t offset) const
{
  uint32_t index = 0;
  for (size_t i = 0; i < M; ++i)
      index += strides_[i + offset] * (shape_.dimensions_[i + offset] - dim_quotas[i]);
  return index;
}

template <typename T, uint32_t N>
void Tensor<T, N>::pMap(std::function<void(T *lhs)> const &fn)
{
  // this is the index upper bound for iteration
  uint32_t cumul_index = shape_.IndexProduct();
  uint32_t dim_quotas[N];
  std::copy_n(shape_.dimensions_, N, dim_quotas);
  for (size_t i = 0; i < cumul_index; ++i) {
    uint32_t index = pEvaluateIndex(dim_quotas);
    fn(&(data_[index]));
    pUpdateQuotas(dim_quotas);
  }
}

template <typename T, uint32_t N>
template <typename X>
void Tensor<T, N>::pUnaryMap(Tensor<X, N> const &tensor,
    std::function<void(T *lhs, X *rhs)> const &fn)
{
  // this is the index upper bound for iteration
  uint32_t cumul_index = shape_.IndexProduct();

  uint32_t dim_quotas[N];
  std::copy_n(shape_.dimensions_, N, dim_quotas);
  for (size_t i = 0; i < cumul_index; ++i) {
    uint32_t index = pEvaluateIndex(dim_quotas);
    uint32_t t_index = tensor.pEvaluateIndex(dim_quotas);
    fn(&(data_[index]), &(tensor.data_[t_index]));
    pUpdateQuotas(dim_quotas);
  }
}

template <typename T, uint32_t N>
template <typename X, typename Y>
void Tensor<T, N>::pBinaryMap(Tensor<X, N> const &tensor_1, Tensor<Y, N> const &tensor_2, std::function<void(T *lhs, X *rhs1, Y *rhs2)> const &fn)
{
  uint32_t cumul_index = shape_.IndexProduct();
  uint32_t dim_quotas[N];
  std::copy_n(shape_.dimensions_, N, dim_quotas);
  for (size_t i = 0; i < cumul_index; ++i) {
    uint32_t index = pEvaluateIndex(dim_quotas);
    uint32_t t1_index = tensor_1.pEvaluateIndex(dim_quotas);
    uint32_t t2_index = tensor_2.pEvaluateIndex(dim_quotas);
    fn(&data_[index], &tensor_1.data_[t1_index], &tensor_2.data_[t2_index]);
    pUpdateQuotas(dim_quotas);
  }
}

template <typename T, uint32_t N>
void Tensor<T, N>::pInitializeSteps()
{
  size_t accumulator = 1;
  for (size_t i = 0; i < N; ++i) {
    strides_[N - i - 1] = accumulator;
    accumulator *= shape_.dimensions_[N - i - 1];
  }
}

// private constructor
template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const *dimensions, uint32_t const *strides, T *data, std::shared_ptr<T> &&ref)
  : shape_(Shape<N>(dimensions, 0)), data_(data), ref_(std::move(ref))
{
  std::copy_n(strides, N, strides_);
}

/* -------------------------- Expressions -------------------------- */

template <typename X, typename Y, uint32_t M>
Tensor<X, M> Add(Tensor<X, M> const& tensor_1, Tensor<Y, M> const& tensor_2)
{
  if (tensor_1.shape_ != tensor_2.shape_) throw std::logic_error(DIMENSION_MISMATCH("Add(Tensor const&, Tensor const&)"));
  Tensor<X, M> sum_tensor(tensor_1.shape_);
  std::function<void(X *, X*, Y*)> add = [](X *x, X *y, Y *z) -> void
  {
    *x = *y + *z;
  };
  sum_tensor.pBinaryMap(tensor_1, tensor_2, add);
  return sum_tensor;
}

template <typename X, typename Y, uint32_t M>
Tensor<X, M> Subtract(Tensor<X, M> const& tensor_1, Tensor<Y, M> const& tensor_2)
{
  if (tensor_1.shape_ != tensor_2.shape_) throw std::logic_error(DIMENSION_MISMATCH("Subtract(Tensor const&, Tensor const&)"));
  Tensor<X, M> diff_tensor(tensor_1.shape_);
  std::function<void(X *, X*, Y*)> sub = [](X *x, X *y, Y *z) -> void
  {
    *x = *y - *z;
  };
  diff_tensor.pBinaryMap(tensor_1, tensor_2, sub);
  return diff_tensor;
}

template <typename X, typename Y, uint32_t M1, uint32_t M2>
Tensor<X, M1 + M2 - 2> Multiply(Tensor<X, M1> const& tensor_1, Tensor<Y, M2> const& tensor_2)
{
  static_assert(M1 || M2, OVERLOAD_RESOLUTION("Multiply(Tensor const&, Tensor const&)"));
  static_assert(M1, SCALAR_TENSOR_MULT("Multiply(Tensor const&, Tensor const&)"));
  static_assert(M2, SCALAR_TENSOR_MULT("Multiply(Tensor const&, Tensor const&)"));
  if (tensor_1.shape_.dimensions_[0] != tensor_2.shape_.dimensions_[M2 - 1])
    throw std::logic_error(INNER_DIMENSION_MISMATCH("Multiply(Tensor const&, Tensor const&)"));
  auto shape = Shape<M1 + M2 - 2>();

  std::copy_n(tensor_1.shape_.dimensions_, M1 - 1, shape.dimensions_);
  std::copy_n(tensor_2.shape_.dimensions_ + 1, M2 - 1, shape.dimensions_ + M1 - 1);
  Tensor<X, M1 + M2 - 2> prod_tensor(shape);
  uint32_t cumul_index_1 = tensor_1.shape_.IndexProduct() / tensor_1.shape_.dimensions_[M1 - 1];
  uint32_t cumul_index_2 = tensor_2.shape_.IndexProduct() / tensor_2.shape_.dimensions_[0];
  uint32_t dim_quotas_1[M1 - 1], dim_quotas_2[M2 - 1];
  std::copy_n(tensor_1.shape_.dimensions_, M1 - 1, dim_quotas_1);
  std::copy_n(tensor_2.shape_.dimensions_ + 1, M2 - 1, dim_quotas_2);
  uint32_t index = 0;
  for (size_t i1 = 0; i1 < cumul_index_1; ++i1) {
    for (size_t i2 = 0; i2 < cumul_index_2; ++i2) {
      uint32_t t1_index = tensor_1.pEvaluateIndex(dim_quotas_1);
      uint32_t t2_index = tensor_2.pEvaluateIndex(dim_quotas_2, 1);
      X value {};
      for (size_t x = 0; x < tensor_1.shape_.dimensions_[M1 - 1]; ++x)
          value += *(tensor_1.data_ + t1_index + tensor_1.strides_[M1 - 1] * x) *
            *(tensor_2.data_ + t2_index + tensor_2.strides_[0] * x);
      prod_tensor.data_[index] = value;
      tensor_2.pUpdateQuotas(dim_quotas_2, 1, 1);
      ++index;
    }
    tensor_1.pUpdateQuotas(dim_quotas_1, 1);
  }
  return prod_tensor;
}

template <typename T, uint32_t N>
Tensor<T, N> Tensor<T, N>::operator-() const
{
  Tensor<T, N> neg_tensor(shape_);
  std::function<void(T *, T *)> neg = [](T *x, T *y) -> void
  {
    *x = -(*y);
  };
  neg_tensor.pUnaryMap(*this, neg);
  return neg_tensor;
}

/* --------------------------- Useful Functions ------------------------- */

// creates a tensor with a different shape but the same number of elements
// template <typename T, uint32_t N1, uint32_t N2>
// Tensor<T, N1> Resize(Tensor<T, N2> const &tensor) {}

template <typename T, uint32_t N>
Tensor<T, N> Tensor<T, N>::copy() const
{
  T *data = pDuplicateData();
  return Tensor<T, N>(shape_.dimensions_, strides_, data,
      std::shared_ptr<T>(data, _ARRAY_DELETER(T)));
}

/* ------------------------------- Iterator ----------------------------- */

template <typename T, uint32_t N>
Tensor<T, N>::Iterator::Iterator(Tensor<T, N + 1> const &tensor, uint32_t index)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N && "This should throw earlier");
  std::copy_n(tensor.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.dimensions_ + index + 1, 
      N - index - 1, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index - 1, strides_ + index);
}

template <typename T, uint32_t N>
Tensor<T, N> Tensor<T, N>::Iterator::operator*()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, uint32_t N>
Tensor<T, N> const Tensor<T, N>::Iterator::operator*() const
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, uint32_t N>
typename Tensor<T, N>::Iterator Tensor<T, N>::Iterator::operator++(int)
{
  Tensor<T, N>::Iterator it {*this};
  ++(*this);
  return it;
}

template <typename T, uint32_t N>
typename Tensor<T, N>::Iterator &Tensor<T, N>::Iterator::operator++()
{
  data_ += stride_;
  return *this;
}

template <typename T, uint32_t N>
typename Tensor<T, N>::Iterator Tensor<T, N>::Iterator::operator--(int)
{
  Tensor<T, N>::Iterator it {*this};
  --(*this);
  return it;
}

template <typename T, uint32_t N>
typename Tensor<T, N>::Iterator &Tensor<T, N>::Iterator::operator--()
{
  data_ -= stride_;
  return *this;
}

template <typename T, uint32_t N>
typename Tensor<T, N - 1>::Iterator Tensor<T, N>::begin(uint32_t index)
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::Iterator::Iterator()"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::Iterator::Iterator()"));
  --index;
  return Tensor<T, N - 1>::Iterator(*this, index);
}

template <typename T, uint32_t N>
typename Tensor<T, N - 1>::Iterator Tensor<T, N>::end(uint32_t index)
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::Iterator::Iterator()"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::Iterator::Iterator()"));
  --index;
  typename Tensor<T, N - 1>::Iterator it{*this, index};
  it.data_ += strides_[index] * shape_.dimensions_[index];
  return it;
}

/* ------------------------ Scalar Specializations ---------------------- */

template <>
class Shape<0> { /*@Shape<0>*/
public:

  /* -------------------- typedefs -------------------- */
  typedef size_t                    size_type;
  typedef ptrdiff_t                 difference_type;
  typedef Shape<0>                  self_type;

  /* ----------------- friend classes ----------------- */

  template <typename X, uint32_t M> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ------------------ Constructors ------------------ */

  explicit Shape() {}
  Shape(Shape<0> const&) {}

  /* -------------------- Getters --------------------- */

  constexpr static uint32_t rank() { return 0; }

  /* -------------------- Equality -------------------- */

  // Equality only compares against dimensions, not step size
  bool operator==(Shape<0> const&) const noexcept { return true; }
  bool operator!=(Shape<0> const&) const noexcept { return false; }
  template <uint32_t M>
  bool operator==(Shape<M> const&) const noexcept { return false; }
  template <uint32_t M>
  bool operator!=(Shape<M> const&) const noexcept { return true; }

  /* ---------------- Print ----------------- */

  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

private:
};

// Scalar specialization
template <typename T>
class Tensor<T, 0>: public Expression<Tensor<T, 0>> { /*@Tensor<T, 0>*/
public:
  typedef T                 value_type;
  typedef T&                reference_type;
  typedef T const&          const_reference_type;
  typedef Tensor<T, 0>      self_type;

  /* ------------- Friend Classes ------------- */

  template <typename X, uint32_t M> friend class Tensor;

  /* -------------- Constructors -------------- */

  Tensor();
  explicit Tensor(value_type &&val);
  explicit Tensor(Shape<0>);
  Tensor(Tensor<T, 0> const &tensor);
  template <typename NodeType>
  Tensor(Expression<NodeType> const& expression);

  /* ------------- Assignment ------------- */

  Tensor<T, 0> &operator=(Tensor<T, 0> const &tensor);
  template <typename X> Tensor<T, 0> &operator=(Tensor<X, 0> const &tensor);

  /* -------------- Getters -------------- */

  constexpr static uint32_t rank() { return 0; }
  Shape<0> shape() const noexcept { return shape_; }
  value_type &operator()() { return *data_; }
  value_type const &operator()() const { return *data_; }

  /* -------------- Setters -------------- */

  template <typename X,
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type>
  Tensor<T, 0> &operator=(X&& elem);

  /* --------------- Print --------------- */

  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);
  template <typename X>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, 0> &tensor);

  /* ------------ Equivalence ------------ */

  bool operator==(Tensor<T, 0> const& tensor) const { return *data_ == *(tensor.data_); }
  bool operator!=(Tensor<T, 0> const& tensor) const { return !(*this == tensor); }
  template <typename X>
  bool operator==(Tensor<X, 0> const& tensor) const { return *data_ == *(tensor.data_); }
  template <typename X>
  bool operator!=(Tensor<X, 0> const& tensor) const { return !(*this == tensor); }

  template <typename X,
            typename = typename std::enable_if
            <std::is_convertible
            <typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type
  > bool operator==(X val) const;
  template <typename X,
            typename = typename std::enable_if
            <std::is_convertible
            <typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type
  > bool operator!=(X val) const { return !(*this == val); }

  /* ------------ Expressions ------------- */

  // Addition
  template <typename X, typename Y>
  friend Tensor<X, 0> Add(Tensor<X, 0> const &tensor_1, Tensor<Y, 0> const &tensor_2);

  // Subtraction
  template <typename X, typename Y>
  friend Tensor<X, 0> Subtract(Tensor<X, 0> const &tensor_1, Tensor<Y, 0> const &tensor_2);

  // Multiplication
  template <typename X, typename Y>
  friend Tensor<X, 0> Multiply(Tensor<X, 0> const &tensor_1, Tensor<Y, 0> const &tensor_2);

  // Negation
  Tensor<T, 0> operator-() const;

  /* ---------- Type Conversion ----------- */

  operator T &() { return *data_; }
  operator T const&() const { return *data_; }

  /* ------------- Useful Functions ------------ */

  Tensor<T, 0> copy() const;

  /* ---------------- Iterator -------------- */

  class Iterator { /*@Iterator<T, 0>*/
  public:
    Tensor<T, 0> operator*();
    Tensor<T, 0> const operator*() const;
    Iterator operator++(int);
    Iterator &operator++();
    Iterator operator--(int);
    Iterator &operator--();
  private:
    Iterator(Tensor<T, 1> const &tensor, uint32_t);

    /**
     * Data describing the underlying tensor
     */
    Shape<0> shape_;
    value_type *data_;
    std::shared_ptr<T> ref_;

  };

private:

  /* ------------------- Data ------------------- */

  Shape<0> shape_;
  value_type * data_;
  std::shared_ptr<T> ref_;

  /* ------------------ Utility ----------------- */

  Tensor(uint32_t const *, uint32_t const *, T *data, std::shared_ptr<T> &&ref_);

};

/* ------------------- Constructors ----------------- */

template <typename T>
Tensor<T, 0>::Tensor() : shape_(Shape<0>()), data_(new T[1]()), ref_(data_, _ARRAY_DELETER(T))
{}

template <typename T>
Tensor<T, 0>::Tensor(Shape<0>) : shape_(Shape<0>()), data_(new T[1]()), ref_(data_, _ARRAY_DELETER(T))
{}

template <typename T>
Tensor<T, 0>::Tensor(T &&val) : shape_(Shape<0>()), data_(new T[1])
{
  *data_ = std::forward<T>(val);
  ref_ = std::shared_ptr<T>(data_);
}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0> const &tensor): shape_(Shape<0>()), data_(tensor.data_), ref_(tensor.ref_)
{}

template <typename T>
template <typename NodeType>
Tensor<T, 0>::Tensor(Expression<NodeType> const& expression)
{
  Tensor<T, 0> result = expression.self()();
  data_ = result.data_;
  ref_ = std::move(result.ref_);
}

/* ---------------------- Assignment ---------------------- */

template <typename T>
Tensor<T, 0> &Tensor<T, 0>::operator=(Tensor<T, 0> const &tensor)
{
  *data_ = *(tensor.data_);
  return *this;
}

template <typename T>
template <typename X>
Tensor<T, 0> &Tensor<T, 0>::operator=(Tensor<X, 0> const &tensor)
{
  *data_ = *(tensor.data_);
  return *this;
}

template <typename T>
template <typename X, typename>
Tensor<T, 0> &Tensor<T, 0>::operator=(X&& elem)
{
  *data_ = std::forward<X>(elem);
  return *this;
}

/* ------------------------ Equality ----------------------- */

template <typename T>
template <typename X, typename>
bool Tensor<T, 0>::operator==(X val) const
{
  return *data_ == val;
}

/* ----------------------- Expressions ----------------------- */

template <typename X, typename Y>
Tensor<X, 0> Add(Tensor<X, 0> const &tensor_1, Tensor<Y, 0> const &tensor_2)
{
  return Tensor<X, 0>(tensor_1() + tensor_2());
}

template <typename X, typename Y, typename = typename std::enable_if<
          LogicalAnd<!IsTensor<X>::value, !IsTensor<Y>::value>::value>>
inline Tensor<X, 0> Add(X const& x, Y const & y) { return Tensor<X, 0>(x + y); }

template <typename X, typename Y>
Tensor<X, 0> Subtract(Tensor<X, 0> const &tensor_1, Tensor<Y, 0> const &tensor_2)
{
  return Tensor<X, 0>(tensor_1() - tensor_2());
}

template <typename X, typename Y, typename = typename std::enable_if<
          LogicalAnd<!IsTensor<X>::value, !IsTensor<Y>::value>::value>>
inline Tensor<X, 0> Subtract(X const& x, Y const & y) { return Tensor<X, 0>(x - y); }

// Directly overload operator*
template <typename X, typename Y>
inline Tensor<X, 0> operator*(Tensor<X, 0> const &tensor_1, Tensor<Y, 0> const &tensor_2)
{
  return Tensor<X, 0>(tensor_1() * tensor_2());
}

template <typename T>
Tensor<T, 0> Tensor<T, 0>::operator-() const
{
  return Tensor<T, 0>(-(*data_));
}

/* ------------------------ Utility --------------------------- */

template <typename T>
Tensor<T, 0>::Tensor(uint32_t const *, uint32_t const *, T *data, std::shared_ptr<T> &&ref)
  : data_(data), ref_(std::move(ref))
{}

/* ------------------------ Overloads ------------------------ */

template <typename X>
std::ostream &operator<<(std::ostream &os, const Tensor<X, 0> &tensor)
{
  os << *(tensor.data_);
  return os;
}

/* ------------------- Useful Functions ---------------------- */

template <typename T>
Tensor<T, 0> Tensor<T, 0>::copy() const
{
  T *data = new T[1]();
  *data = *data_;
  // decrement count because it gets incremented in the constructor
  return Tensor<T, 0>(nullptr, nullptr, data,
      std::shared_ptr<T>(data, _ARRAY_DELETER(T)));
}

/* ---------------------- Iterators ------------------------- */

template <typename T>
Tensor<T, 0> Tensor<T, 0>::Iterator::operator*()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
Tensor<T, 0> const Tensor<T, 0>::Iterator::operator*() const
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

/* ---------------------- Expressions ------------------------ */

template <typename LHSType, typename RHSType>
class BinaryAdd: public Expression<BinaryAdd<LHSType, RHSType>> {
public:

  /* ---------------- typedefs --------------- */

  typedef typename LHSType::value_type value_type;
  typedef BinaryAdd                    self_type;

  /* -------------- Constructors -------------- */

  BinaryAdd(LHSType const &lhs, RHSType const &rhs);

  /* ---------------- Getters ----------------- */

  constexpr static uint32_t rank() { return LHSType::rank(); }
  uint32_t dimension(uint32_t index) const { return lhs_.dimension(index); }
  template <typename... Indices>
  Tensor<value_type, LHSType::rank() - sizeof...(Indices)> operator()(Indices... indices) const;

private:
  LHSType const &lhs_;
  RHSType const &rhs_;
};

template <typename LHSType, typename RHSType>
BinaryAdd<LHSType, RHSType>::BinaryAdd(LHSType const &lhs, RHSType const &rhs)
  : lhs_(lhs), rhs_(rhs) {}

template <typename LHSType, typename RHSType>
template <typename... Indices>
Tensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)> BinaryAdd<LHSType, RHSType>::operator()(Indices... indices) const
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Addition"));
  return Add(ValueToTensor<LHSType>(lhs_).value(indices...),
             ValueToTensor<RHSType>(rhs_).value(indices...));
}

template <typename LHSType, typename RHSType>
BinaryAdd<LHSType, RHSType> operator+(Expression<LHSType> const &lhs, Expression<RHSType> const &rhs)
{
  return BinaryAdd<LHSType, RHSType>(lhs.self(), rhs.self());
}

/* ----------------- Print ----------------- */

template <typename LHSType, typename RHSType>
std::ostream &operator<<(std::ostream &os, BinaryAdd<LHSType, RHSType> const &binary_add)
{
  os << binary_add();
  return os;
}

/* ----------------------------------------- */

template <typename LHSType, typename RHSType>
class BinarySub: public Expression<BinarySub<LHSType, RHSType>> {
public:
  /* ---------------- typedefs --------------- */

  typedef typename LHSType::value_type value_type;
  typedef BinarySub                    self_type;

  /* -------------- Constructors -------------- */

  BinarySub(LHSType const &lhs, RHSType const &rhs);

  /* ---------------- Getters ----------------- */

  constexpr static uint32_t rank() { return LHSType::rank(); }
  uint32_t dimension(uint32_t index) const { return lhs_.dimension(index); }
  template <typename... Indices>
  Tensor<value_type, LHSType::rank() - sizeof...(Indices)> operator()(Indices... indices) const;

  /* ------------------------------------------ */

private:
  LHSType const &lhs_;
  RHSType const &rhs_;
};

template <typename LHSType, typename RHSType>
BinarySub<LHSType, RHSType>::BinarySub(LHSType const &lhs, RHSType const &rhs)
  : lhs_(lhs), rhs_(rhs) {}

template <typename LHSType, typename RHSType>
template <typename... Indices>
Tensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)> BinarySub<LHSType, RHSType>::operator()(Indices... indices) const
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Subtraction"));
  return Subtract(
      ValueToTensor<LHSType>(lhs_).value(indices...),
      ValueToTensor<RHSType>(rhs_).value(indices...));
}

template <typename LHSType, typename RHSType>
BinarySub<LHSType, RHSType> operator-(Expression<LHSType> const &lhs, Expression<RHSType> const &rhs)
{
  return BinarySub<LHSType, RHSType>(lhs.self(), rhs.self());
}

/* ----------------- Print ----------------- */

template <typename LHSType, typename RHSType>
std::ostream &operator<<(std::ostream &os, BinarySub<LHSType, RHSType> const &binary_sub)
{
  os << binary_sub();
  return os;
}

/**
 * Multiplies the inner dimensions
 * i.e. 3x4x5 * 5x4x3 produces a 3x4x4x3 tensor
 */

template <typename LHSType, typename RHSType>
class BinaryMul: public Expression<BinaryMul<LHSType, RHSType>> {
public:

  /* ---------------- typedefs --------------- */

  typedef typename LHSType::value_type value_type;
  typedef BinaryMul                    self_type;

  /* -------------- Constructors -------------- */

  BinaryMul(LHSType const &lhs, RHSType const &rhs);

  /* ---------------- Getters ----------------- */

  constexpr static uint32_t rank() { return LHSType::rank() + RHSType::rank() - 2; }
  uint32_t dimension(uint32_t index) const;
  template <typename... Indices>
  Tensor<value_type, LHSType::rank() + RHSType::rank() - sizeof...(Indices) - 2> operator()(Indices... indices) const;

private:
  LHSType const &lhs_;
  RHSType const &rhs_;
};

template <typename LHSType, typename RHSType>
BinaryMul<LHSType, RHSType>::BinaryMul(LHSType const &lhs, RHSType const &rhs)
  : lhs_(lhs), rhs_(rhs) {}

template <typename LHSType, typename RHSType>
template <typename... Indices>
Tensor<typename LHSType::value_type, LHSType::rank() + RHSType::rank() - sizeof...(Indices) - 2> BinaryMul<LHSType, RHSType>::operator()(Indices... indices) const
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Multiplication"));
  return Multiply(ValueToTensor<LHSType>(lhs_).value(),
                  ValueToTensor<RHSType>(rhs_).value())(indices...);
}

template <typename LHSType, typename RHSType>
BinaryMul<LHSType, RHSType> operator*(Expression<LHSType> const &lhs, Expression<RHSType> const &rhs)
{
  return BinaryMul<LHSType, RHSType>(lhs.self(), rhs.self());
}

} // tensor

#undef NTENSOR_0CONSTRUCTOR
#undef NCONSTRUCTOR_0TENSOR
#undef NELEMENTS
#undef DIMENSION_INVALID
#undef RANK_OUT_OF_BOUNDS
#undef INDEX_OUT_OF_BOUNDS
#undef ZERO_INDEX
#undef SLICES_EMPTY
#undef SLICES_OUT_OF_BOUNDS
#undef SLICE_INDICES_REPEATED
#undef SLICE_INDICES_DESCENDING
#undef RANK_MISMATCH
#undef DIMENSION_MISMATCH
#undef INNER_DIMENSION_MISMATCH
#undef SCALAR_TENSOR_MULT
#undef DEBUG
#undef OVERLOAD_RESOLUTION
#undef _ARRAY_DELETER

/* ----------------------------------------------- */


#endif // TENSORS_H_
