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
#define INNER_DIMENSION_MISMATCH(METHOD) \
  METHOD " Failed -- Tensors have different inner dimensions"
#define SCALAR_TENSOR_MULT(METHOD) \
  METHOD " Failed -- Cannot multiple tensors with scalars"

/* ----------------------------------------------- */

/* ---------------- Debug Messages --------------- */

#define DEBUG \
  "This static assertion should never fire"
#define OVERLOAD_RESOLUTION(METHOD) \
  METHOD " :: Overload resolution :: " DEBUG

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
template <uint32_t N>
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
  uint32_t dimension(uint32_t index) const ;

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
  Shape() {}
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

/* ------------ Reference Counting ------------ */

template <typename T>
struct ReferenceFrame {
  ReferenceFrame(T *data_) : data(data_), count(1) {}
  T *data;
  size_t count;
};

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
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ----------------- Constructors ----------------- */

  explicit Tensor(uint32_t const (&indices)[N]);
  Tensor(uint32_t const (&dimensions)[N], T const &value);
  explicit Tensor(Shape<N> shape);
  Tensor(Shape<N> shape, T const &value);
  Tensor(Tensor<T, N> const &tensor);
  template <typename NodeType>
  Tensor(Expression<NodeType> const& expression);
  ~Tensor();

  /* ----------------- Assignment ----------------- */

  Tensor<T, N> &operator=(Tensor<T, N> const &tensor);
  template <typename NodeType>
  Tensor<T, N> &operator=(Expression<NodeType> const &rhs);

  /* ----------------- Getters ----------------- */
  
  constexpr static uint32_t rank() { return N; }
  uint32_t dimension(uint32_t index) const { return shape_.dimension(index); }
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

  /* ----------------- Equivalennce ------------------ */

  bool operator==(Tensor<T, N> const& tensor) const;
  bool operator!=(Tensor<T, N> const& tensor) const { return !(*this == tensor); }
  template <typename X>
  bool operator==(Tensor<X, N> const& tensor) const;
  template <typename X>
  bool operator!=(Tensor<X, N> const& tensor) const { return !(*this == tensor); }

  /* -------------- Useful Functions ------------------- */
  template <typename U, uint32_t M>
  Tensor<U, M> CopyTensor(Tensor<U, M> const &tensor);


/* --------------------------- Debug Information --------------------------- */
#ifndef _NDBEUG
  uint32_t const *dimensions() const noexcept { return shape_.dimensions_; }
#endif
/* ------------------------------------------------------------------------- */

private:

  /* ----------------- Data ---------------- */

  Shape<N> shape_;
  uint32_t strides_[N];
  value_type *data_;
  ReferenceFrame<T> *frame_;

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
  void pUpdateTrackers(uint32_t (&dim_trackers)[N]) const;
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
  Tensor(uint32_t const *dimensions, uint32_t const *strides, T *data, ReferenceFrame<T> *frame);


}; // Tensor

/* ----------------------------- Constructors ------------------------- */

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const (&dimensions)[N])
  : shape_(Shape<N>(dimensions)), data_(new T[shape_.IndexProduct()])
{
  pInitializeSteps();
  // create a new reference 
  frame_ = new ReferenceFrame<T>(data_);
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(uint32_t const (&dimensions)[N], T const& value) 
  : shape_(Shape<N>(dimensions)), data_(new T[shape_.IndexProduct()])
{
  pInitializeSteps();
  std::function<void(T *)> allocate = [&value](T *x) -> void { *x = value; };
  pMap(allocate);
  frame_ = new ReferenceFrame<T>(data_);
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Shape<N> shape)
  : shape_(Shape<N>(shape)), data_(new T[shape.IndexProduct()])
{
  pInitializeSteps();
  // create a new reference
  frame_ = new ReferenceFrame<T>(data_);
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Shape<N> shape, T const &value)
  : shape_(Shape<N>(shape)), data_(new T[shape.IndexProduct()])
{
  pInitializeSteps();
  std::function<void(T *)> allocate = [&value](T *x) -> void { *x = value; };
  pMap(allocate);
  frame_ = new ReferenceFrame<T>(data_);
}

template <typename T, uint32_t N>
Tensor<T, N>::Tensor(Tensor<T, N> const &tensor)
  : shape_(tensor.shape_), data_(tensor.data_), frame_(tensor.frame_)
{
  std::copy_n(tensor.strides_, N, strides_);
  ++frame_->count;
}

template <typename T, uint32_t N>
template <typename NodeType>
Tensor<T, N>::Tensor(Expression<NodeType> const& expression) 
{
  auto result = expression.self()();
  std::copy_n(result.strides_, N, strides_);
  shape_ = result.shape_;
  data_ = result.data_;
  frame_ = result.frame_;
  ++frame_->count;
}

template <typename T, uint32_t N> Tensor<T, N>::~Tensor()
{
  --frame_->count;
  if (!frame_->count) {
    delete[] frame_->data;
    delete frame_;
  }
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
  return Tensor<T, N - M>(shape_.dimensions_ + M, strides_ + M,  data_ + cumul_index, frame_);
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
      index += tensor.strides_[j] * (tensor.shape_.dimensions_[j] - dim_trackers[j]);
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
  return Tensor<T, N - M>(shape_.dimensions_ + M, strides_ + M, data_ + cumul_index, frame_);
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
  return Tensor<T, N - M>(dimensions, strides, data_ + offset, frame_);
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

// find the correct index to "step" to
template <typename T, uint32_t N>
void Tensor<T, N>::pUpdateTrackers(uint32_t (&dim_trackers)[N]) const
{
  uint32_t dim_index = N;
  bool propogate = true;
  while (dim_index && propogate) {
    --dim_trackers[dim_index - 1];
    --dim_index;
    if (!dim_trackers[dim_index - 1]) 
      dim_trackers[dim_index - 1] = shape_.dimensions_[dim_index - 1];
    else 
      propogate = false;
  }
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
      index += strides_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
    fn(&(data_[index]));
    pUpdateTrackers(dim_trackers);
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
      index += strides_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
      t_index += tensor.strides_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
    }
    fn(&(data_[index]), &(tensor.data_[t_index]));
    pUpdateTrackers(dim_trackers);
  }
}

template <typename T, uint32_t N>
template <typename X, typename Y>
void Tensor<T, N>::pBinaryMap(Tensor<X, N> const &tensor_1, Tensor<Y, N> const &tensor_2, std::function<void(T *lhs, X *rhs1, Y *rhs2)> const &fn)
{
  uint32_t cumul_index = shape_.IndexProduct();

  uint32_t dim_trackers[N];
  std::copy_n(shape_.dimensions_, N, dim_trackers);
  for (size_t i = 0; i < cumul_index; ++i) {
    uint32_t index = 0, t1_index = 0, t2_index = 0;
    for (size_t j = 0; j < N; ++j) {
      index += strides_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
      t1_index += tensor_1.strides_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
      t2_index += tensor_2.strides_[j] * (shape_.dimensions_[j] - dim_trackers[j]);
    }
    fn(&data_[index], &tensor_1.data_[t1_index], &tensor_2.data_[t2_index]); 
    pUpdateTrackers(dim_trackers);
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
Tensor<T, N>::Tensor(uint32_t const *dimensions, uint32_t const *strides, T *data, ReferenceFrame<T> *frame)
  : shape_(Shape<N>(dimensions, 0)), data_(data), frame_(frame)
{
  std::copy_n(strides, N, strides_);
  ++frame_->count;
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

  // FIXME :: Very messy, is there any way to implement an internal iterator?
  std::copy_n(tensor_1.shape_.dimensions_, M1 - 1, shape.dimensions_);
  std::copy_n(tensor_2.shape_.dimensions_ + 1, M2 - 1, shape.dimensions_ + M1 - 1);
  Tensor<X, M1 + M2 - 2> prod_tensor(shape);
  uint32_t cumul_index_1 = tensor_1.shape_.IndexProduct() / tensor_1.shape_.dimensions_[M1 - 1];
  uint32_t cumul_index_2 = tensor_2.shape_.IndexProduct() / tensor_2.shape_.dimensions_[0];
  uint32_t dim_trackers_1[M1 - 1], dim_trackers_2[M2 - 1];
  std::copy_n(tensor_1.shape_.dimensions_, M1 - 1, dim_trackers_1);
  std::copy_n(tensor_2.shape_.dimensions_ + 1, M2 - 1, dim_trackers_2);
  uint32_t index = 0;
  for (size_t i1 = 0; i1 < cumul_index_1; ++i1) {
    for (size_t i2 = 0; i2 < cumul_index_2; ++i2) {
      uint32_t t1_index = 0, t2_index = 0;
      for (size_t j = 0; j < M1 - 1; ++j) 
        t1_index += tensor_1.strides_[j] * (tensor_1.shape_.dimensions_[j] - dim_trackers_1[j]);
      for (size_t j = 0; j < M2 - 1; ++j) 
        t2_index += tensor_2.strides_[j + 1] * (tensor_2.shape_.dimensions_[j + 1] - dim_trackers_2[j]);
      X value {};
      for (size_t x = 0; x < tensor_1.shape_.dimensions_[M1 - 1]; ++x) 
          value += *(tensor_1.data_ + t1_index + tensor_1.strides_[M1 - 1] * x) *
            *(tensor_2.data_ + t2_index + tensor_2.strides_[0] * x);
      prod_tensor.data_[index] = value;
      uint32_t dim_index = M2 - 1;
      bool propogate = true;
      while (dim_index && propogate) {
        --dim_trackers_2[dim_index - 1];
        if (!dim_trackers_2[dim_index - 1]) dim_trackers_2[dim_index - 1] = tensor_2.shape_.dimensions_[dim_index];
        else propogate = false;
        --dim_index;
      }
      ++index;
    }
    uint32_t dim_index = M1 - 1;
    bool propogate = true;
    // find the correct index to "step" to
    while (dim_index && propogate) {
      --dim_trackers_1[dim_index - 1];
      if (!dim_trackers_1[dim_index - 1]) dim_trackers_1[dim_index - 1] = tensor_1.shape_.dimensions_[dim_index - 1];
      else propogate = false;
      --dim_index;
    }
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
Tensor<T, N> CopyTensor(Tensor<T, N> const &tensor)
{
  T *data = tensor.pDuplicateData();
  ReferenceFrame<T> *frame = new ReferenceFrame<T>(data);
  // decrement reference because its incremented in the constructor
  --frame->count;
  return Tensor<T, N>(tensor.shape_.dimensions_, tensor.strides_, data, frame);
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
class Tensor<T, 0>: public Expression<Tensor<T, 0>> {
public:
  typedef T                 value_type;
  typedef T&                reference_type;
  typedef T const&          const_reference_type;
  typedef Tensor<T, 0>      self_type;

  /* ------------- Friend Classes ------------- */

  template <typename X, uint32_t M> friend class Tensor;

  /* ------------- Constructors ------------- */

  Tensor();
  explicit Tensor(value_type &&val);
  explicit Tensor(Shape<0>);
  Tensor(Tensor<T, 0> const &tensor);
  template <typename NodeType>
  Tensor(Expression<NodeType> const& expression);
  ~Tensor();

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
  template <typename U>
  Tensor<U, 0> CopyTensor(Tensor<U, 0> const &tensor);

private:
  
  /* ------------------- Data ------------------- */
  
  Shape<0> shape_;
  value_type * data_;
  ReferenceFrame<T> *frame_;

  /* ------------------ Utility ----------------- */

  Tensor(uint32_t const *, uint32_t const *, T *data, ReferenceFrame<T> *frame);

};

/* ------------------- Constructors ----------------- */

template <typename T>
Tensor<T, 0>::Tensor() : shape_(Shape<0>()), data_(new T[1]())
{
  frame_ = new ReferenceFrame<T>(data_);
}

template <typename T>
Tensor<T, 0>::Tensor(Shape<0>) : shape_(Shape<0>()), data_(new T[1]())
{
  frame_ = new ReferenceFrame<T>(data_);
}

template <typename T>
Tensor<T, 0>::Tensor(T &&val) : shape_(Shape<0>()), data_(new T[1])
{
  *data_ = std::forward<T>(val);
  frame_ = new ReferenceFrame<T>(data_);
}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0> const &tensor): shape_(Shape<0>()), data_(tensor.data_), frame_(tensor.frame_)
{
  ++frame_->count;
}

template <typename T>
template <typename NodeType>
Tensor<T, 0>::Tensor(Expression<NodeType> const& expression) 
{
  Tensor<T, 0> result = expression.self()();
  data_ = result.data_;
  frame_ = result.frame_;
  ++frame_->count;
}

template <typename T>
Tensor<T, 0>::~Tensor()
{
  --frame_->count;
  if (!frame_->count) {
    delete[] frame_->data;
    delete frame_;
  }
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
Tensor<T, 0>::Tensor(uint32_t const *, uint32_t const *, T *data, ReferenceFrame<T> *frame)
  : data_(data), frame_(frame) 
{
  ++frame_->count;
}

/* ------------------------ Overloads ------------------------ */

template <typename X>
std::ostream &operator<<(std::ostream &os, const Tensor<X, 0> &tensor)
{
  os << *(tensor.data_);
  return os;
}

/* ------------------- Useful Functions ---------------------- */

template <typename T>
Tensor<T, 0> CopyTensor(Tensor<T, 0> const &tensor)
{ 
  T *data = new T(*tensor.data_);
  ReferenceFrame<T> *frame = new ReferenceFrame<T>(data);
  // decrement count because it gets incremented in the constructor
  --frame->count;
  return Tensor<T, 0>(nullptr, nullptr, data, frame);
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
