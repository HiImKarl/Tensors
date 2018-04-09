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
/* ----------------------------------------------- */

namespace tensor {

/* --------------- Forward Declerations --------------- */
template <uint32_t N> class Shape;
template <typename T, uint32_t N = 0> class Tensor;
template <typename LHS, typename RHS> class BinaryAdd;
template <typename LHS, typename RHS> class BinarySub;

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
  ValueToTensor(T&& val): value(std::forward<T>(val)) {}
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

/* ---------------------------------------------------------- */


template <typename NodeType>
struct Expression { 
  inline NodeType &self() { return *static_cast<NodeType *>(this); }
  inline NodeType const &self() const { return *static_cast<NodeType const*>(this); }
};

// Tensor Shape
template <uint32_t N>
class Shape {
public:
  /* -------------------- typedefs -------------------- */
  typedef size_t                    size_type;
  typedef ptrdiff_t                 difference_type;
  typedef Shape<N>                  self_type;

  /* ----------------- friend classes ----------------- */

  template <typename X, uint32_t M> friend class Tensor;

  /* ------------------ Constructors ------------------ */

  explicit Shape(uint32_t const (&dimensions)[N]);
  Shape(Shape<N> const &shape);

  /* -------------------- Getters --------------------- */

  constexpr static uint32_t rank() { return N; }
  uint32_t dimension(uint32_t index) const ;

  /* -------------------- Equality -------------------- */

  // Equality only compares against dimensions, not step size
  bool operator==(Shape<N> const& shape) const noexcept;
  bool operator!=(Shape<N> const& shape) const noexcept { return !(*this == shape); }
  template <uint32_t M>
  bool operator==(Shape<M> const& shape) const noexcept;
  template <uint32_t M>
  bool operator!=(Shape<M> const& shape) const noexcept { return !(*this == shape); }

  /* ------------------- Utility -------------------- */

  uint32_t IndexProduct() const noexcept;

  /* ---------------- Print ----------------- */
  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

private:
  uint32_t dimensions_[N];
  uint32_t steps_[N];

  explicit Shape(uint32_t const *dimensions, uint32_t const *steps);
};

template <uint32_t N>
Shape<N>::Shape(uint32_t const (&dimensions)[N])
{
  std::copy_n(dimensions, N, dimensions_);
  size_t accumulator = 1;
  for (size_t i = 0; i < N; ++i) {
    steps_[N - i - 1] = accumulator;
    accumulator *= dimensions_[N - i - 1];
  }
}

template <uint32_t N>
Shape<N>::Shape(Shape const &shape)
{
  std::copy_n(shape.dimensions_, N, dimensions_);
  size_t accumulator = 1;
  for (size_t i = 0; i < N; ++i) {
    steps_[N - i - 1] = accumulator;
    accumulator *= dimensions_[N - i - 1];
  }
}

template <uint32_t N>
Shape<N>::Shape(uint32_t const *dimensions, uint32_t const *steps)
{
  std::copy_n(dimensions, N, dimensions_);
  std::copy_n(steps, N, steps_);
}

template <uint32_t N>
uint32_t Shape<N>::dimension(uint32_t index) const 
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

template <typename T, uint32_t N>
class Tensor: public Expression<Tensor<T, N>> {
public:
  /* ------------------ Type Definitions --------------- */

  typedef T                 value_type;
  typedef T&                reference;
  typedef T const&          const_reference;
  typedef size_t            size_type;
  typedef ptrdiff_t         difference_type;
  typedef Tensor<T, N>      self_type;

  /* ----------------- Friend Classes ----------------- */

  template <typename X, uint32_t M> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;

  /* ----------------- Constructors ----------------- */

  explicit Tensor(uint32_t const (&indices)[N]);
  explicit Tensor(Shape<N> shape);
  Tensor(Tensor<T, N> const &tensor);
  Tensor(Tensor<T, N> &&tensor);
  template <typename NodeType>
  Tensor(Expression<NodeType> const& expression);

  /* ----------------- Assignment ----------------- */

  Tensor<T, N> &operator=(Tensor<T, N> const &tensor);
  template <typename NodeType>
  Tensor<T, N> &operator=(Expression<NodeType> const &rhs);

  /* ----------------- Destructor ----------------- */

  ~Tensor();

  /* ----------------- Getters ----------------- */
  
  constexpr static uint32_t rank() { return N; }
  uint32_t dimension(uint32_t index) const { return shape_.dimension(index); }
  Shape<N> const& shape() const noexcept { return shape_; }

  /* ------------------ Access To Data ----------------- */

  template <typename... Indices> 
  Tensor<T, N - sizeof...(Indices)> operator()(Indices... args);
  template <typename... Indices> 
  Tensor<T, N - sizeof...(Indices)> const operator()(Indices... args) const;
  template <uint32_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices)> slice(Indices... indices);
  template <uint32_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices)> const slice(Indices... indices) const;

  /* -------------------- Expressions ------------------- */

  template <typename X, typename Y, uint32_t M>
  friend Tensor<X, M> Add(Tensor<X, M> const& tensor_1, Tensor<Y, M> const& tensor_2);
  template <typename X, typename Y, uint32_t M>
  friend Tensor<X, M> Subtract(Tensor<X, M> const& tensor_1, Tensor<Y, M> const& tensor_2);
  Tensor<T, N> negative() const;

  /* ------------------ Print to ostream --------------- */

  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

  /* ----------------- Equivalennce ------------------ */

  bool operator==(Tensor<T, N> const& tensor) const;
  bool operator!=(Tensor<T, N> const& tensor) const { return !(*this == tensor); }
  template <typename X>
  bool operator==(Tensor<X, N> const& tensor) const;
  template <typename X>
  bool operator!=(Tensor<X, N> const& tensor) const { return !(*this == tensor); }

  /* -------------- Useful Functions ------------------- */

  template <typename U, uint32_t M>
  friend Tensor<U, M> Zeros(uint32_t const (&dimensions)[M]);
  template <typename U, uint32_t M>
  friend Tensor<U, M> Ones(uint32_t const (&dimensions)[M]);


/* --------------------------- Debug Information --------------------------- */
#ifndef _NDBEUG
  bool owner_flag() const noexcept { return owner_flag_; }
  uint32_t const *dimensions() const noexcept { return shape_.dimensions_; }
#endif
/* ------------------------------------------------------------------------- */

private:

  /* ----------------- Data ---------------- */

  Shape<N> shape_;
  value_type *const data_;
  bool owner_flag_;           // ownership of memory

  /* --------------- Getters --------------- */

  uint32_t const *steps() const noexcept { return shape_.steps_; }

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
  void pMap(std::function<void(T *lhs)> const &fn);
  template <typename X>
  void pUnaryMap(Tensor<X, N> const &tensor, std::function<void(T *lhs, X *rhs)> const &fn); 
  template <typename X, typename Y>
  void pBinaryMap(Tensor<X, N> const &tensor_1, Tensor<Y, N> const &tensor_2, 
      std::function<void(T *lhs, X *rhs1, Y *rhs2)> const& fn);

  // allocate new space and copy data
  value_type * pDuplicateData() const;
  // move data pointer
  value_type * pMoveData();

  // Declare all fields of the constructor at once
  Tensor(uint32_t const *dimensions, uint32_t const *steps, T *data);


}; // Tensor

/* ----------------------------- Constructors ------------------------- */

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const (&dimensions)[N])
  : shape_(Shape<N>(dimensions)), data_(new T[shape_.IndexProduct()]), owner_flag_(true) {}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Shape<N> shape)
  : shape_(Shape<N>(shape)), data_(new T[shape.IndexProduct()]), owner_flag_(true) {} 

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Tensor<T, N> const &tensor)
  : shape_(tensor.shape_), data_(tensor.pDuplicateData()), owner_flag_(true) {}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Tensor<T, N> &&tensor): shape_(tensor.shape_), data_(tensor.data_), owner_flag_(tensor.owner_flag_)
{
  tensor.owner_flag_ = false;
}

template <typename T, uint32_t N>
template <typename NodeType>
Tensor<T, N>::Tensor(Expression<NodeType> const& expression) 
  : shape_(expression.self().shape()), data_(expression.self()().pMoveData()), owner_flag_(true)
{}

// private constructor
template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const *dimensions, uint32_t const *steps, T *data)
  : shape_(Shape<N>(dimensions, steps)), data_(data), owner_flag_(false) {}

template <typename T, uint32_t N> Tensor<T, N>::~Tensor()
{
  if (owner_flag_) delete[] data_;
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
  NodeType const& tensor = rhs.self();
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
  uint32_t dim_trackers[N];
  std::copy_n(tensor.shape_.dimensions_, N, dim_trackers);

  add_brackets(N, true); // opening brackets
  os << tensor.data_[0]; // first element
  for (size_t i = 0; i < cumul_index - 1; ++i) {
    uint32_t index = 0, dim_index = N;
    bool propogate = true;
    uint32_t bracket_count = 0;
    // find the correct index to "step" to
    while (dim_index && propogate) {
      --dim_trackers[dim_index - 1];
      ++bracket_count;
      if (!dim_trackers[dim_index - 1]) 
        dim_trackers[dim_index - 1] = tensor.shape_.dimensions_[dim_index - 1];
      else 
        propogate = false;
      --dim_index;
    }
    for (size_t j = 0; j < N; ++j)
      index += tensor.shape_.steps_[j] * (tensor.shape_.dimensions_[j] - dim_trackers[j]);
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
  return Tensor<T, N - M>(shape_.dimensions_ + M, shape_.steps_ + M, data_ + cumul_index);
}

template <typename T, uint32_t N>
template <uint32_t M, typename... Indices>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(
 uint32_t cumul_index, uint32_t next_index, Indices... rest)
{
  if (next_index > shape_.dimensions_[N - sizeof...(rest) - 1] || next_index == 0)
    throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));

  // adjust for 1 index array access
  cumul_index += shape_.steps_[N - sizeof...(rest) - 1] * (next_index - 1);
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
  uint32_t steps[N - M];
  uint32_t array_index = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (placed_indices[i]) {
      offset += (placed_indices[i] - 1) * shape_.steps_[i]; // adjust for 1-based indexing
    } else {
      steps[array_index] = shape_.steps_[i];
      dimensions[array_index] = shape_.dimensions_[i];
      ++array_index;
    }
  }
  return Tensor<T, N - M>(dimensions, steps, data_ + offset);
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

template <typename T, uint32_t N>
T * Tensor<T, N>::pMoveData() 
{
  owner_flag_ = false;
  return data_;
}

template <typename T, uint32_t N>
void Tensor<T, N>::pMap(std::function<void(T *lhs)> const &fn)
{
  // this is the index upper bound for iteration
  uint32_t cumul_index = shape_.IndexProduct();

  uint32_t dim_trackers[N];
  std::copy_n(shape_.dimensions_, N, dim_trackers);
  for (size_t i = 0; i < cumul_index; ++i) {
    uint32_t index = 0;
    for (size_t j = 0; j < N; ++j) 
      index += shape_.steps_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
    uint32_t dim_index = N;
    bool propogate = true;
    fn(&(data_[index]));
    // find the correct index to "step" to
    while (dim_index && propogate) {
      --dim_trackers[dim_index - 1];
      --dim_index;
      if (!dim_trackers[dim_index - 1]) 
        dim_trackers[dim_index - 1] = shape_.dimensions_[dim_index - 1];
      else 
        propogate = false;
    }
  }
}

template <typename T, uint32_t N>
template <typename X>
void Tensor<T, N>::pUnaryMap(Tensor<X, N> const &tensor, 
    std::function<void(T *lhs, X *rhs)> const &fn)
{
  // this is the index upper bound for iteration
  uint32_t cumul_index = shape_.IndexProduct();

  uint32_t dim_trackers[N];
  std::copy_n(shape_.dimensions_, N, dim_trackers);
  for (size_t i = 0; i < cumul_index; ++i) {
    uint32_t index = 0, t_index = 0;
    for (size_t j = 0; j < N; ++j) {
      index += shape_.steps_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
      t_index += tensor.shape_.steps_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
    }
    uint32_t dim_index = N;
    bool propogate = true;
    fn(&(data_[index]), &(tensor.data_[t_index]));
    // find the correct index to "step" to
    while (dim_index && propogate) {
      --dim_trackers[dim_index - 1];
      --dim_index;
      if (!dim_trackers[dim_index - 1]) 
        dim_trackers[dim_index - 1] = shape_.dimensions_[dim_index - 1];
      else 
        propogate = false;
    }
  }
}

template <typename T, uint32_t N>
template <typename X, typename Y>
void Tensor<T, N>::pBinaryMap(Tensor<X, N> const &tensor_1, Tensor<Y, N> const &tensor_2, 
    std::function<void(T *lhs, X *rhs1, Y *rhs2)> const &fn)
{
  // this is the index upper bound for iteration
  uint32_t cumul_index = shape_.IndexProduct();

  uint32_t dim_trackers[N];
  std::copy_n(shape_.dimensions_, N, dim_trackers);
  for (size_t i = 0; i < cumul_index; ++i) {
    uint32_t index = 0, t1_index = 0, t2_index = 0;
    for (size_t j = 0; j < N; ++j) {
      index += shape_.steps_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
      t1_index += tensor_1.shape_.steps_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
      t2_index += tensor_2.shape_.steps_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
    }
    uint32_t dim_index = N;
    bool propogate = true;
    fn(&data_[index], &tensor_1.data_[t1_index], &tensor_2.data_[t2_index]); 
    // find the correct index to "step" to
    while (dim_index && propogate) {
      --dim_trackers[dim_index - 1];
      --dim_index;
      if (!dim_trackers[dim_index - 1]) dim_trackers[dim_index - 1] = shape_.dimensions_[dim_index - 1];
      else propogate = false;
    }
  }
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

template <typename T, uint32_t N>
Tensor<T, N> Tensor<T, N>::negative() const
{
  Tensor<T, N> neg_tensor(shape_);
  std::function<void(T *, T *)> neg = [](T *x, T *y) -> void 
  {
    *x = -(*y);
  };
  neg_tensor.pUnaryMap(*this, neg);
  return neg_tensor;
}

/* ------------------------ Useful Functions ------------------- */

template <typename U, uint32_t M>
Tensor<U, M> Zeros(uint32_t const (&dimensions)[M]) 
{
  Tensor<U, M> zero_tensor(dimensions);
  std::function<void(U *)> zero = [](U *x) -> void 
  {
    *x = 0;
  };
  zero_tensor.pMap(zero);
  return zero_tensor;
}

template <typename U, uint32_t M>
Tensor<U, M> Ones(uint32_t const (&dimensions)[M]) 
{
  Tensor<U, M> one_tensor(dimensions);
  std::function<void(U *)> one = [](U *x) -> void 
  {
    *x = 1;
  };
  one_tensor.pMap(one);
  return one_tensor;
}

/* ------------------------ Scalar Specialization ----------------------- */

template <>
class Shape<0> {
public:
  /* -------------------- typedefs -------------------- */
  typedef size_t                    size_type;
  typedef ptrdiff_t                 difference_type;
  typedef Shape<0>                  self_type;

  /* ----------------- friend classes ----------------- */

  template <typename X, uint32_t M> friend class Tensor;

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
class Tensor<T, 0>: public Expression<Tensor<T, 0>> {
public:
  typedef T                 value_type;
  typedef T&                reference;
  typedef T const&          const_reference;
  typedef Tensor<T, 0>      self_type;

  /* ------------- Friend Classes ------------- */

  template <typename X, uint32_t M> friend class Tensor;

  /* ------------- Constructors ------------- */

  Tensor();
  explicit Tensor(value_type &&val);
  Tensor(Tensor<T, 0> const &tensor);
  Tensor(Tensor<T, 0> &&tensor);
  ~Tensor();

  /* ------------- Assignment ------------- */

  Tensor<T, 0> &operator=(Tensor<T, 0> const &tensor);
  template <typename X> Tensor<T, 0> &operator=(Tensor<X, 0> const &tensor);

  /* -------------- Getters -------------- */

  constexpr static uint32_t rank() { return 0; }
  Shape<0> const &shape() const noexcept { return shape_; }
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
  template <typename X, typename Y>
  friend Tensor<X, 0> Add(Tensor<X, 0> const &tensor, Y const& value);
  template <typename X, typename Y>
  friend Tensor<X, 0> Add(X const& value, Tensor<Y, 0> const &tensor); 

  // Subtraction
  template <typename X, typename Y>
  friend Tensor<X, 0> Subtract(Tensor<X, 0> const &tensor_1, Tensor<Y, 0> const &tensor_2); 
  template <typename X, typename Y>
  friend Tensor<X, 0> Subtract(Tensor<X, 0> const &tensor, Y const& value);
  template <typename X, typename Y>
  friend Tensor<X, 0> Subtract(X const& value, Tensor<Y, 0> const &tensor); 

  // negation
  Tensor<T, 0> negative() const;

  /* ---------- Type Conversion ----------- */

  operator T &() { return *data_; }
  operator T const&() const { return *data_; }

private:

  /* ------------------- Data ------------------- */
  
  Shape<0> shape_;
  value_type * const data_;
  bool owner_flag_;

  /* ------------------ Utility ----------------- */

  Tensor(uint32_t const *, uint32_t const *, T *data)
    : data_(data), owner_flag_(false) {}

};

/* ------------------- Constructors ----------------- */

template <typename T>
Tensor<T, 0>::Tensor() : shape_(Shape<0>()), data_(new T), owner_flag_(true) {}

template <typename T>
Tensor<T, 0>::Tensor(T &&val) : shape_(Shape<0>()), data_(new T(std::forward<T>(val))), owner_flag_(true) {}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0> const &tensor): shape_(Shape<0>()), data_(new T(*tensor.data_)), owner_flag_(true) {}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0> &&tensor): shape_(Shape<0>()), data_(tensor.data_), owner_flag_(tensor.owner_flag_)
{
  tensor.owner_flag_ = false;
}

template <typename T>
Tensor<T, 0>::~Tensor()
{
  if (owner_flag_) delete data_;
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

template <typename X, typename Y>
Tensor<X, 0> Add(Tensor<X, 0> const &tensor, Y const& value)
{
  return Tensor<X, 0>(tensor() + value);
}

template <typename X, typename Y>
Tensor<X, 0> Add(X const& value, Tensor<Y, 0> const &tensor)
{
  return Tensor<X, 0>(tensor() + value);
}

template <typename X, typename Y, typename = typename std::enable_if<
          LogicalAnd<!IsTensor<X>::value, !IsTensor<Y>::value>::value>>
inline Tensor<X, 0> Add(X const& x, Y const & y) { return Tensor<X, 0>(x + y); }

template <typename X, typename Y>
Tensor<X, 0> Subtract(Tensor<X, 0> const &tensor_1, Tensor<Y, 0> const &tensor_2)
{
  return Tensor<X, 0>(tensor_1() - tensor_2());
}

template <typename X, typename Y>
Tensor<X, 0> Subtract(Tensor<X, 0> const &tensor, Y const& value)
{
  return Tensor<X, 0>(tensor() - value);
}

template <typename X, typename Y>
Tensor<X, 0> Subtract(X const& value, Tensor<Y, 0> const &tensor)
{
  return Tensor<X, 0>(value - tensor());
}

template <typename X, typename Y, typename = typename std::enable_if<
          LogicalAnd<!IsTensor<X>::value, !IsTensor<Y>::value>::value>>
inline Tensor<X, 0> Subtract(X const& x, Y const & y) { return Tensor<X, 0>(x - y); }

template <typename T>
Tensor<T, 0> Tensor<T, 0>::negative() const
{
  return Tensor<T, 0>(-(*data_));
}

/* ------------------------ Overloads ------------------------ */

template <typename X>
std::ostream &operator<<(std::ostream &os, const Tensor<X, 0> &tensor)
{
  os << *(tensor.data_);
  return os;
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
  Shape<LHSType::rank()> const &shape() const noexcept { return lhs_.shape(); }

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
  Shape<LHSType::rank()> const &shape() const { return lhs_.shape(); }
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


/*
 * The following methods are built on top of the core of the libarary
 */


} // tensor

#undef NTENSOR_0CONSTRUCTOR
#undef NCONSTRUCTOR_0TENSOR
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

#endif // TENSORS_H_
