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
template <typename T, uint32_t N = 0> class Tensor;

template <typename NodeType>
struct Expression { inline NodeType const &self() const { return *static_cast<NodeType const*>(this); }};

template <typename LHSType, typename RHSType>
class BinaryAdd: public Expression<BinaryAdd<LHSType, RHSType>> {
public:
  /* ---------------- typedefs --------------- */
  typedef typename LHSType::value_type value_type;
  typedef BinaryAdd                    self_type;         

  /* -------------- constructors -------------- */
  BinaryAdd(LHSType const &lhs, RHSType const &rhs);

  static uint32_t rank() { return LHSType::rank(); }
  template <typename... Indices>
  Tensor<value_type, LHSType::rank() - sizeof...(Indices)> operator()(Indices... indices);
  template <typename... Indices>
  Tensor<value_type, LHSType::rank() - sizeof...(Indices)> const operator()(Indices... indices) const;

private:
  LHSType const &lhs_;
  RHSType const &rhs_;
};

template <typename LHSType, typename RHSType>
BinaryAdd<LHSType, RHSType>::BinaryAdd(LHSType const &lhs, RHSType const &rhs)
  : lhs_(lhs), rhs_(rhs)
{
  static_assert(lhs.rank() == rhs.rank(), RANK_MISMATCH("Binary Addition"));
  for (uint32_t i = 1; i <= lhs.rank(); ++i)
    if (lhs.dimension(i) != rhs.dimension(i))
      throw std::logic_error(DIMENSION_MISMATCH("Binary Addition"));
}

template <typename LHSType, typename RHSType>
template <typename... Indices>
Tensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)> BinaryAdd<LHSType, RHSType>::operator()(Indices... indices)
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Addition"));
  return lhs_(indices...) + rhs_(indices...);
}

template <typename LHSType, typename RHSType>
template <typename... Indices>
Tensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)> const BinaryAdd<LHSType, RHSType>::operator()(Indices... indices) const
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Addition"));
  return lhs_(indices...) + rhs_(indices...);
}

template <typename LHSType, typename RHSType>
BinaryAdd<LHSType, RHSType> operator+(Expression<LHSType> const &lhs, Expression<RHSType> const &rhs)
{
  return BinaryAdd<LHSType, RHSType>(lhs, rhs);
}

template <typename LHSType, typename RHSType>
class BinarySub: public Expression<BinarySub<LHSType, RHSType>> {
public:
  /* ---------------- typedefs --------------- */
  typedef typename LHSType::value_type value_type;
  typedef BinarySub                    self_type;         

  /* -------------- constructors -------------- */
  BinarySub(LHSType const &lhs, RHSType const &rhs);

  static uint32_t rank() { return LHSType::rank(); }
  template <typename... Indices>
  Tensor<value_type, LHSType::rank() - sizeof...(Indices)> operator()(Indices... indices);
  template <typename... Indices>
  Tensor<value_type, LHSType::rank() - sizeof...(Indices)> const operator()(Indices... indices) const;

private:
  LHSType const &lhs_;
  RHSType const &rhs_;
};

template <typename LHSType, typename RHSType>
BinarySub<LHSType, RHSType>::BinarySub(LHSType const &lhs, RHSType const &rhs)
  : lhs_(lhs), rhs_(rhs)
{
  static_assert(lhs.rank() == rhs.rank(), RANK_MISMATCH("Binary Addition"));
  for (uint32_t i = 1; i <= lhs.rank(); ++i)
    if (lhs.dimension(i) != rhs.dimension(i))
      throw std::logic_error(DIMENSION_MISMATCH("Binary Addition"));
}

template <typename LHSType, typename RHSType>
template <typename... Indices>
Tensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)> BinarySub<LHSType, RHSType>::operator()(Indices... indices)
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Addition"));
  return lhs_(indices...) - rhs_(indices...);
}

template <typename LHSType, typename RHSType>
template <typename... Indices>
Tensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)> const BinarySub<LHSType, RHSType>::operator()(Indices... indices) const
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Addition"));
  return lhs_(indices...) - rhs_(indices...);
}

template <typename LHSType, typename RHSType>
BinarySub<LHSType, RHSType> operator-(Expression<LHSType> const &lhs, Expression<RHSType> const &rhs)
{
  return BinarySub<LHSType, RHSType>(lhs, rhs);
}

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

  static uint32_t rank() { return N; }
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
  std::copy_n(shape.steps_, N, steps_);
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
  Tensor(Tensor<T, N> const &tensor);
  Tensor(Tensor<T, N> &&tensor);

  /* ----------------- Assignment ----------------- */

  Tensor<T, N> &operator=(Tensor<T, N> const &tensor);
  template <typename NodeType>
  Tensor<T, N> &operator=(Expression<NodeType> const &rhs);

  /* ----------------- Destructor ----------------- */

  ~Tensor();

  /* ----------------- Getters ----------------- */
  
  static uint32_t rank() { return N; }
  uint32_t dimension(uint32_t index) const noexcept { return shape_.dimension(index); }

  /* ------------------ Access To Data --------------- */

  template <typename... Indices> 
  Tensor<T, N - sizeof...(Indices)> operator()(Indices... args);
  template <typename... Indices> 
  Tensor<T, N - sizeof...(Indices)> const operator()(Indices... args) const;
  template <uint32_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices)> slice(Indices... indices);
  template <uint32_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices)> const slice(Indices... indices) const;

  /* ------------------ print to ostream --------------- */

  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

  /* ----------------- Equivalnce ------------------ */

  bool operator==(Tensor<T, N> const& tensor) const;
  bool operator!=(Tensor<T, N> const& tensor) const { return !(*this == tensor); }
  template <typename X>
  bool operator==(Tensor<X, N> const& tensor) const;
  template <typename X>
  bool operator!=(Tensor<X, N> const& tensor) const { return !(*this == tensor); }

  /* ---------------- Arithmetic Operations ----------------- */

  template <typename X, typename Y>
  friend Tensor<X, N> operator+(Tensor<X, N> tensor_1, Tensor<Y, N> tensor_2);
  template <typename X, typename Y>
  friend Tensor<X, N> operator-(Tensor<X, N> tensor_1, Tensor<Y, N> tensor_2);

/* --------------------------- Debug Information --------------------------- */
#ifndef _NDBEUG
  bool is_owner() const noexcept { return is_owner_; }
  uint32_t const *dimensions() const noexcept { return shape_.dimensions_; }
#endif
/* ------------------------------------------------------------------------- */

private:

  /* ----------------- Data ---------------- */

  // Shape
  Shape<N> shape_;

  // Data
  value_type *const data_;

  // This flag denotes the tensor object's ownership of memory
  bool is_owner_;

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

  /* ----------------- Utility -------------------------- */

  // Data mapping
  template <typename X>
  void pUnaryMap(Tensor<X, N> const &tensor, std::function<void(T *lhs, X *rhs)> const &fn); 
  template <typename X, typename Y>
  void pBinaryMap(Tensor<X, N> const &tensor_1, Tensor<Y, N> const &tensor_2, 
      std::function<void(T *lhs, X *rhs1, Y *rhs2)> const& fn);

  // allocate new space and copy data
  value_type * pDuplicateData() const;

  // Declare all fields of the constructor at once
  Tensor(uint32_t const *dimensions, uint32_t const *steps, T *data);


}; // Tensor

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const (&dimensions)[N])
  : shape_(Shape<N>(dimensions)), data_(new T[std::accumulate(dimensions, dimensions + N, 1, std::multiplies<uint32_t>())]), is_owner_(true) {}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Tensor<T, N> const &tensor)
  : shape_(tensor.shape_), data_(tensor.pDuplicateData()), is_owner_(true) {}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Tensor<T, N> &&tensor): shape_(tensor.shape_), data_(tensor.data_), is_owner_(tensor.is_owner_)
{
  tensor.is_owner_ = false;
}

// private constructor
template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const *dimensions, uint32_t const *steps, T *data)
  : shape_(Shape<N>(dimensions, steps)), data_(data), is_owner_(false) {}

template <typename T, uint32_t N> Tensor<T, N>::~Tensor()
{
  if (is_owner_) delete[] data_;
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
      index += tensor.steps_[j] * (tensor.shape_.dimensions_[j] - dim_trackers[j]);
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
      if (!dim_trackers[dim_index - 1]) 
        dim_trackers[dim_index - 1] = shape_.dimensions_[dim_index - 1];
      else 
        propogate = false;
      --dim_index;
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
      t1_index += tensor_1.steps_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
      t2_index += tensor_2.steps_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
    }
    uint32_t dim_index = N;
    bool propogate = true;
    fn(&data_[index], &tensor_1.data_[t1_index], &tensor_2.data_[t2_index]); 
    // find the correct index to "step" to
    while (dim_index && propogate) {
      --dim_trackers[dim_index - 1];
      if (!dim_trackers[dim_index - 1]) dim_trackers[dim_index - 1] = shape_.dimensions_[dim_index - 1];
      else propogate = false;
      --dim_index;
    }
  }
}

template <typename X, typename Y, uint32_t N>
Tensor<X, N> operator+(Tensor<X, N> tensor_1, Tensor<Y, N> tensor_2)
{
  // FIXME :: IMPLEMENT 
  return tensor_1;
}

template <typename X, typename Y, uint32_t N>
Tensor<X, N> operator-(Tensor<X, N> tensor_1, Tensor<Y, N> tensor_2)
{
  // FIXME :: IMPLEMENT 
  return tensor_1;
}

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

  /* ------------- Assignment ------------- */
  Tensor<T, 0> &operator=(Tensor<T, 0> const &tensor);
  template <typename X> Tensor<T, 0> &operator=(Tensor<X, 0> const &tensor);

  // Destructor
  ~Tensor();

  // Getters
  static uint32_t rank() { return 0; }
  uint32_t dimension() const noexcept 
  {
    return 0; 
  }

  // Access to data_
  value_type &operator()() { return *data_; }

  // Print
  template <typename X, uint32_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

  // Setters
  template <typename X,
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type>
  Tensor<T, 0> &operator=(X&& elem);

  // Equivalence
  bool operator==(Tensor<T, 0> const& tensor) const { return *data_ == *(tensor.data_); }
  bool operator!=(Tensor<T, 0> const& tensor) const { return !(*this == tensor); }
  template <typename X>
  bool operator==(Tensor<X, 0> const& tensor) const { return *data_ == *(tensor.data_); }
  template <typename X>
  bool operator!=(Tensor<X, 0> const& tensor) const { return !(*this == tensor); }

  // Value type equivalence
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

  template <typename X>
  Tensor<T, 0> operator+(X &&val) const;

  template <typename X>
  Tensor<T, 0> operator-(X &&val) const;

  // Type conversion for single element values
  operator T &() { return *data_; }
  operator T const&() const { return *data_; }

  // Print
  template <typename X>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X> &tensor);

private:
  value_type * const data_;
  bool is_owner_;

  /* --------------- Getters --------------- */

  uint32_t const *steps() const noexcept 
  {
    return nullptr; 
  }

  /* --------------- Utility --------------- */
  Tensor(uint32_t const *, uint32_t const *, T *data)
    : data_(data), is_owner_(false) {}

};

// Tensor Methods
// Constructors, Destructor, Assignment
template <typename T>
Tensor<T, 0>::Tensor() : data_(new T), is_owner_(true) {}

template <typename T>
Tensor<T, 0>::Tensor(T &&val) : data_(new T(std::forward<T>(val))), is_owner_(true) {}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0> const &tensor): data_(new T(*tensor.data_)), is_owner_(true) {}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0> &&tensor): data_(tensor.data_), is_owner_(tensor.is_owner_)
{
  tensor.is_owner_ = false;
}

template <typename T>
Tensor<T, 0>::~Tensor()
{
  if (is_owner_) delete data_;
} 

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

template <typename T>
template <typename X, typename>
bool Tensor<T, 0>::operator==(X val) const
{
  return *data_ == val;
}

template <typename T>
template <typename X>
Tensor<T, 0> Tensor<T, 0>::operator+(X &&val) const
{
  return Tensor(val + *data_);
}

template <typename T>
template <typename X>
Tensor<T, 0> Tensor<T, 0>::operator-(X &&val) const
{
  return Tensor(*data_ - val);
}

/* ------------------------ Overloads ------------------------ */

template <typename X>
std::ostream &operator<<(std::ostream &os, const Tensor<X, 0> &tensor)
{
  os << *(tensor.data_);
  return os;
}

/* ----------------------------------------------------------- */

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
