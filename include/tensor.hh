#pragma once
#ifndef TENSOR_H_
#define TENSOR_H_
#include <cstddef>
#include <iostream>
#include <algorithm>
#include <exception>
#include <utility>
#include <type_traits>
#include <initializer_list>
#include <numeric>
#include <functional>
#include <memory>
#include <cassert>
#include <string>
#include <unordered_map>

extern bool eDebugFlag;

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
  "Incorrect number of elements provided -- "
#define ZERO_ELEMENT \
  "Cannot be constructed with a zero dimension"
#define EXPECTING_C_ARRAY \
  "Expecting argument to be a C-array"

// Out of bounds
#define DIMENSION_INVALID \
  "Attempt To Access Invalid Dimension"
#define RANK_OUT_OF_BOUNDS \
  "Rank out of bounds"
#define INDEX_OUT_OF_BOUNDS \
  "Index out of bounds"

// Slicing
#define SLICES_EMPTY \
  "Tensor::slice(Indices...) Failed -- At least one dimension must be sliced"
#define SLICES_OUT_OF_BOUNDS \
  "Tensor::slice(Indices...) Failed -- Slices out of bounds"
#define SLICE_INDICES_REPEATED \
  "Tensor::slice(Indices...) Failed -- Repeated slice indices"
#define SLICE_INDICES_DESCENDING \
  "Tensor::slice(Indices...) Failed -- Slice indices must be listed in ascending order"

// Iteration
#define BEGIN_ON_NON_VECTOR \
  "Tensor::begin() should only be called on rank 1 tensors, " \
  "use Tensor::begin(size_t) instead"

// Arithmetic Operations
#define RANK_MISMATCH \
  "Expecting same rank"
#define EXPECTED_SCALAR \
  "Expecting Scalar"
#define DIMENSION_MISMATCH \
  "Shapes have different dimensions"
#define INNER_DIMENSION_MISMATCH \
  "Shapes have different inner dimensions"
#define ELEMENT_COUNT_MISMATCH \
  "Shapes have different total number of elements" 
#define SCALAR_TENSOR_MULT \
  "Failed -- Cannot multiple tensors with scalars"

/* ---------------- Debug Messages --------------- */

#define DEBUG \
  "This static assertion should never fire"

/* ------------------ Lambdas -------------------- */

/* ----------------------------------------------- */

namespace tensor {

/* --------------- Forward Declerations --------------- */

template <typename T> class Array;
template <size_t N> class Shape;
template <typename T, size_t N, typename ContainerType = Array<T>> class Tensor;
template <typename LHS, typename RHS> class BinaryAdd;
template <typename LHS, typename RHS> class BinarySub;
template <typename LHS, typename RHS> class BinaryMul;

/* -------------------- Type Definitions ----------------- */

template <typename T, typename ContainerType = Array<T>> 
using Scalar = Tensor<T, 0, ContainerType>;
template <typename T, typename ContainerType = Array<T>> 
using Vector = Tensor<T, 1, ContainerType>;
template <typename T, typename ContainerType = Array<T>> 
using Matrix = Tensor<T, 2, ContainerType>;

/* ----------------- Template Meta-Patterns ----------------- */

/** Meta-template logical && */
template <bool B1, bool B2>
struct LogicalAnd { static bool const value = B1 && B2; };

/** Boolean member `value` is true if T is an any-rank Tensor 
 * object, false o.w.  
 */
template <typename T>
struct IsTensor { static bool const value = false; };

/** Tensor specialization of IsTensor, Boolean member 
 * `value` is true
 */
template <typename T, size_t N, typename ContainerType>
struct IsTensor<Tensor<T, N, ContainerType>> { static bool const value = true; };

/** Boolean member `value` is true if T is a 0-rank 
 * Tensor object, false o.w.
 */
template <typename T>
struct IsScalar { static bool const value = true; };

/** Scalar specialization of IsScalar, Boolean member 
 * `value` is true
 */
template <typename T, typename ContainerType>
struct IsScalar<Tensor<T, 0, ContainerType>> { static bool const value = true; };

/** Tensor specialization of IsScalar, Boolean member 
 * `value` is false 
 */
template <typename T, size_t N, typename ContainerType>
struct IsScalar<Tensor<T, N, ContainerType>> { static bool const value = false; };

/** Provides `value` equal to the rank of `type`, 0 
 *  if not a tensor::Tensor type (C mutli-dimensional arrays 
 *  are considered to be rank 0 in this context)
 */
template <typename T>
struct Rank { enum: size_t { value = 0 }; };

/** Provides `value` equal to the rank of `type`, 0 
 *  if not a tensor::Tensor type (C mutli-dimensional arrays 
 *  are considered to be rank 0 in this context)
 */
template <typename T, size_t N, typename ContainerType>
struct Rank<Tensor<T, N, ContainerType>> { enum: size_t { value = N }; };

/** Tensor member `value` is a wrapper for input `val` if val is
 *  a Tensor, o.w. `value` is a reference to Tensor `val`
 */
template <typename T>
struct ValueAsTensor {
  ValueAsTensor(T &&val): value(std::forward<T>(val)) {}
  T value;
  T &operator()() { return value; }
};

/** Tensor specialization of ValueAsTensor, `value` is a 
 *  const reference to the provided Tensor
 */
template <typename T, size_t N, typename ContainerType>
struct ValueAsTensor<Tensor<T, N, ContainerType>> {
  ValueAsTensor(Tensor<T, N, ContainerType> const &val): value(val) {}
  Tensor<T, N, ContainerType> const &value;

  template <typename... Indices,
            typename = typename std::enable_if<N != sizeof...(Indices)>::type>
  Tensor<T, N - sizeof...(Indices), ContainerType> operator()(Indices... indices) { return value(indices...); }

  template <typename... Indices,
            typename = typename std::enable_if<N == sizeof...(Indices)>::type>
  T const &operator()(Indices... indices) { return value(indices...); }
};

/** BinaryAdd specialization of ValueAsTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <typename LHS, typename RHS>
struct ValueAsTensor<BinaryAdd<LHS, RHS>> {
  ValueAsTensor(BinaryAdd<LHS, RHS> const &val): value(val) {}
  BinaryAdd<LHS, RHS> const &value;
  typedef typename LHS::value_type          value_type;
  typedef typename LHS::container_type      container_type;
  constexpr static size_t N =               LHS::rank();

  template <typename... Indices,
            typename = typename std::enable_if<N != sizeof...(Indices)>::type>
  Tensor<value_type, N - sizeof...(Indices), container_type> operator()(Indices... indices) { return value(indices...); }

  template <typename... Indices,
            typename = typename std::enable_if<N == sizeof...(Indices)>::type>
  value_type const &operator()(Indices... indices) { return value(indices...); }
};

/** BinarySub specialization of ValueAsTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <typename LHS, typename RHS>
struct ValueAsTensor<BinarySub<LHS, RHS>> {
  ValueAsTensor(BinarySub<LHS, RHS> const &val): value(val) {}
  BinarySub<LHS, RHS> const &value;

  typedef typename LHS::value_type          value_type;
  typedef typename LHS::container_type      container_type;

  constexpr static size_t N =      LHS::rank();
  template <typename... Indices,
            typename = typename std::enable_if<N != sizeof...(Indices)>::type>
  Tensor<value_type, N - sizeof...(Indices), container_type> operator()(Indices... indices) { return value(indices...); }

  template <typename... Indices,
            typename = typename std::enable_if<N == sizeof...(Indices)>::type>
  value_type const &operator()(Indices... indices) { return value(indices...); }
};

/** BinaryMul specialization of ValueAsTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <typename LHS, typename RHS>
struct ValueAsTensor<BinaryMul<LHS, RHS>> {
  ValueAsTensor(BinaryMul<LHS, RHS> const &val): value(val) {}
  BinaryMul<LHS, RHS> const &value;

  typedef typename LHS::value_type          value_type;
  typedef typename LHS::container_type      container_type;

  constexpr static size_t N =      LHS::rank() + RHS::rank() - 2;
  template <typename... Indices,
            typename = typename std::enable_if<N != sizeof...(Indices)>::type>
  Tensor<value_type, N - sizeof...(Indices), container_type> operator()(Indices... indices) { return value(indices...); }

  template <typename... Indices,
            typename = typename std::enable_if<N == sizeof...(Indices)>::type>
  value_type const &operator()(Indices... indices) { return value(indices...); }
};

/* -------------------- Data Containers --------------------- */

template <typename T>
class Array { /*@Array<T>*/
public:
  explicit Array(size_t capacity);
  Array(size_t capacity, T const &value);
  template <typename It>
  Array(size_t capacity, It const &first, It const &end);
  ~Array();
  T &operator[](size_t index) { return data_[index]; };
  T const &operator[](size_t index) const { return data_[index]; }
private:
  T *data_;
};

template <typename T>
Array<T>::Array(size_t capacity)
  : data_(new T[capacity]()) {
  if (eDebugFlag) PRINT("!!!!");
  }

template <typename T>
Array<T>::Array(size_t capacity, T const &value)
  : data_(new T[capacity]())
{
  if (eDebugFlag) PRINT("!!!!");
  std::fill(data_, data_ + capacity, value);
}

template <typename T>
template <typename It>
Array<T>::Array(size_t capacity, It const &first, It const &end)
  : data_(new T[capacity]())
{
  if (eDebugFlag) PRINT("!!!!");
  std::copy(first, end, data_);
}

template <typename T>
Array<T>::~Array() 
{
  delete[] data_;
}


/* ---------------------------------------------------------- */

/** CRTP base for Tensor expressions */
template <typename NodeType>
struct Expression {
  inline NodeType &self() { return *static_cast<NodeType *>(this); }
  inline NodeType const &self() const { return *static_cast<NodeType const*>(this); }
};

template <size_t N> /*@Shape<N>*/
class Shape {
/** Tensor shape object, where size_t template `N` represents the Tensor rank.
 *  Implemented as a wrapper around size_t[N].
 */
public:
  /* -------------------- typedefs -------------------- */

  typedef size_t                    size_type;
  typedef ptrdiff_t                 difference_type;
  typedef Shape<N>                  self_type;

  /* ----------------- friend classes ----------------- */

  template <typename X, size_t M, typename CType_> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ------------------ Constructors ------------------ */

  explicit Shape(std::initializer_list<size_t> dimensions); /**< initializer_list constructor */
  Shape(Shape<N> const &shape);                  /**< Copy constructor */

  /* ------------------ Assignment -------------------- */

  Shape<N> &operator=(Shape<N> const &shape);   /**< Copy assignment */

  /* -------------------- Getters --------------------- */

  constexpr static size_t rank() { return N; } /**< Get `N` */

  /** Get dimension reference at `index`. Throws a std::logic_error 
   * exception if index is out of bounds. Note: 1-based indexing.
   */
  size_t &operator[](size_t index);            

  /** Get dimension at `index`. Throws a std::logic_error 
   * exception if index is out of bounds. Note: 1-based indexing.
   */
  size_t operator[](size_t index) const;

  /* -------------------- Equality -------------------- */

  /** `true` iff every dimension is identical */
  bool operator==(Shape<N> const& shape) const noexcept; 

  /** `true` iff any dimension is different */
  bool operator!=(Shape<N> const& shape) const noexcept { return !(*this == shape); }

  /** `true` iff ranks are identical and every dimension is identical */
  template <size_t M>
  bool operator==(Shape<M> const& shape) const noexcept; 

  /** `true` iff ranks are not identical or any dimension is different */
  template <size_t M>
  bool operator!=(Shape<M> const& shape) const noexcept { return !(*this == shape); }

  /* ----------------- Expressions ------------------ */

  template <typename X, typename Y, size_t M1, size_t M2, typename CType_>
  friend Tensor<X, M1 + M2 - 2, CType_> multiply(Tensor<X, M1, CType_> const& tensor_1, Tensor<Y, M2, CType_> const& tensor_2);

  template <typename U, typename CType_> friend Tensor<U, 2, CType_> transpose(Tensor<U, 2, CType_> &mat);
  template <typename U, typename CType_> friend Tensor<U, 2, CType_> transpose(Tensor<U, 1, CType_> &vec);

  /* ------------------- Utility -------------------- */

  /** Returns the product of all of the indices */
  size_t index_product() const noexcept;

  /* -------------------- Print --------------------- */

  template <typename U, size_t M, typename CType_>
  friend std::ostream &operator<<(std::ostream &os, Tensor<U, M, CType_> const&tensor);
  template <size_t M>
  friend std::ostream &operator<<(std::ostream &os, Shape<M> const &shape);

private:
  /* ------------------ Data ------------------ */

  size_t dimensions_[N]; /**< Underlying data */

  /* --------------- Constructor --------------- */

  Shape(size_t const *dimensions);
  Shape() = default;                      
};

template <size_t N>
Shape<N>::Shape(std::initializer_list<size_t> dimensions)
{
  assert(dimensions.size() == N && RANK_MISMATCH);
  size_t *shape_ptr = dimensions_;
  for (size_t dim : dimensions) {
    assert(dim && ZERO_ELEMENT);
    *(shape_ptr++) = dim;
  }
}

template <size_t N>
Shape<N>::Shape(Shape const &shape)
{
  for (size_t i = 0; i < N; ++i) 
    assert((shape.dimensions_[i]) && ZERO_ELEMENT);
  std::copy_n(shape.dimensions_, N, dimensions_);
}

template <size_t N>
Shape<N>::Shape(size_t const *dimensions)
{
  std::copy_n(dimensions, N, dimensions_);
}

template <size_t N>
Shape<N> &Shape<N>::operator=(Shape<N> const &shape)
{
  std::copy_n(shape.dimensions_, N, dimensions_);
  return *this;
}

template <size_t N>
size_t &Shape<N>::operator[](size_t index)
{
  assert((N > index) && DIMENSION_INVALID);
  return dimensions_[index];
}

template <size_t N>
size_t Shape<N>::operator[](size_t index) const
{
  assert((N > index) && DIMENSION_INVALID);
  return dimensions_[index];
}

template <size_t N>
bool Shape<N>::operator==(Shape<N> const& shape) const noexcept
{
  return std::equal(dimensions_, dimensions_ + N, shape.dimensions_);
}

template <size_t N>
template <size_t M>
bool Shape<N>::operator==(Shape<M> const& shape) const noexcept
{
  if (N != M) return false;
  return std::equal(dimensions_, dimensions_ + N, shape.dimensions_);
}

template <size_t N>
size_t Shape<N>::index_product() const noexcept
{
  return std::accumulate(dimensions_, dimensions_ + N, 1, std::multiplies<size_t>());
}

template <size_t N>
std::ostream &operator<<(std::ostream &os, const Shape<N> &shape)
{
  os << "S{";
  for (size_t i = 0; i < N - 1; ++i) os << shape.dimensions_[i] << ", ";
  os << shape.dimensions_[N - 1] << "}";
  return os;
}

template <size_t N>
class Indices { /*@Indices*/
  /** Wrapper around size_t[N] and Shape<N> to provide a specialized array
   *  for accessing Tensors and convenient looping.
   */
public:
  template <typename U, size_t M, typename C_> friend class Tensor;
  Indices(Shape<N> const &shape);
  Indices(size_t const (&indices)[N], Shape<N> const &shape);
  Shape<N> const &shape() const { return shape_; }
  size_t const &operator[](size_t index) const;
  size_t &operator[](size_t index);

  template <size_t M> friend void Increment(Indices<M> &indices);

  template <size_t M> friend void Decrement(Indices<M> &indices);
private:
  size_t indices_[N];
  Shape<N> shape_;
};

/** Increments `indices` so it effectively refers to the
 *  next element in an N-rank Tensor. The dimensions are
 *  propogated accordingly, I.e. (1, 3) indices with a 2x3 
 *  shape will become (2, 0) after increment.
 */
template <size_t M>
void Increment(Indices<M> &indices)
{
  int dim_index = M - 1;
  bool propogate = true;
  while (propogate) {
    ++indices[dim_index];
    if (indices[dim_index] == indices.shape_[dim_index])
      indices[dim_index] = 0;
    else 
      propogate = false;
    --dim_index;
  }
}

/** Decrements `indices` so it effectively refers to the
 *  next element in an N-rank Tensor. The dimensions are
 *  propogated accordingly, I.e. (2, 0) indices with a 2x3 
 *  shape will become (1, 3) after decrement.
 */
template <size_t M>
void Decrement(Indices<M> &indices)
{
  int dim_index = M - 1;
  bool propogate = true;
  while (propogate) {
    if (indices[dim_index] == 0) {
      indices[dim_index] = indices.shape_[dim_index] - 1;
    } else  {
      --indices[dim_index];
      propogate = false;
    }
    --dim_index;
  }
}

template <size_t N>
Indices<N>::Indices(size_t const (&indices)[N], Shape<N> const &shape): shape_(shape)
{
  for (size_t i = 0; i < N; ++i)
    assert((indices[i] < shape[i]) && INDEX_OUT_OF_BOUNDS);
  std::copy_n(indices, N, indices_);
}

template <size_t N>
Indices<N>::Indices(Shape<N> const &shape) : shape_(shape)
{
  std::fill_n(indices_, N, 0);
}

template <size_t N>
size_t const &Indices<N>::operator[](size_t index) const
{
  assert(index < N && DIMENSION_INVALID);
  return indices_[index];
}

template <size_t N>
size_t &Indices<N>::operator[](size_t index)
{
  assert(index < N && DIMENSION_INVALID);
  return indices_[index];
}

/** Proxy object used to construct 
 *
 */
template <typename Array>
struct _A {
  _A(Array const &_value);
  Array const &value;
};

template <typename Array>
_A<Array>::_A(Array const &_value): value(_value)
{
  static_assert(std::is_array<Array>::value, EXPECTING_C_ARRAY);
}

template <typename T, size_t N, typename ContainerType>
class Tensor: public Expression<Tensor<T, N, ContainerType>> { /*@Tensor<T, N, ContainerType>*/
/** Any-rank array of type `T`, where rank `N` is a size_t template.
 *  The underlying data is implemented as a dynamically allocated contiguous
 *  array.
 */  
public:

  /* ------------------ Type Definitions --------------- */
  typedef T                                     value_type;
  typedef ContainerType                         container_type;
  typedef T&                                    reference;
  typedef T const&                              const_reference;
  typedef size_t                                size_type;
  typedef ptrdiff_t                             difference_type;
  typedef Tensor<T, N, ContainerType>           self_type;

  /* ----------------- Friend Classes ----------------- */

  template <typename X, size_t M, typename CType_> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ------------------ Proxy Objects ----------------- */

  class Proxy { /*@Proxy<T,N>*/
  /**
   * Proxy Tensor Object used for building tensors from reference.
   * This is used to differentiate proxy construction 
   * for move and copy construction only.
   */
  public:
    template <typename U, size_t M, typename CType_> friend class Tensor;
    Proxy() = delete;
  private:
    Proxy(Tensor<T, N, ContainerType> const &tensor): tensor_(tensor) {}
    Proxy(Proxy const &proxy): tensor_(proxy.tensor_) {}
    Tensor<T, N, ContainerType> const &tensor_;
  }; 

  /* ------------------ Constructors ------------------ */

  /** Creates a Tensor with dimensions described by `dimensions`.
   *  Elements are zero initialized. Note: dimensions index from 1.
   */
  explicit Tensor(std::initializer_list<size_t> dimensions);

  /** Creates a Tensor with dimensions described by `dimensions`.
   *  Elements are copy initialized to value. Note: dimensions index from 1.
   */
  Tensor(size_t const (&dimensions)[N], T const &value);

  /** Creates a Tensor with dimensions described by `dimensions`.
   *  Elements are copy initialized to the values returned by `factory(args...)`
   *  Note: dimensions index from 1.
   */
  template <typename FunctionType, typename... Arguments>
  Tensor(size_t const (&dimensions)[N], std::function<FunctionType> &f, Arguments&&... args);

  /** Use a C multi-dimensional array to initialize the tensor. The
   *  multi-dimensional array must be enclosed by the _A struct, and
   *  be equal to the tensor's declared rank. 
   */
  template <typename Array>
  Tensor(_A<Array> &&md_array);

  /** Creates a Tensor with dimensions described by `shape`.
   *  Elements are zero initialized. Note: dimensions index from 1.
   */
  explicit Tensor(Shape<N> const &shape);

  /** Creates a Tensor with dimensions described by `shape`.
   *  Elements are copy initialized to value. Note: dimensions index from 1.
   */
  Tensor(Shape<N> const &shape, T const &value): Tensor(shape.dimensions_, value) {}

  /** Creates a Tensor with dimensions described by `shape`.
   *  Elements are copy initialized to the values returned by `factory`
   *  Note: dimensions index from 1.
   */
  template <typename FunctionType, typename... Arguments>
  Tensor(Shape<N> const &shape, std::function<FunctionType> &f, Arguments&&... args)
    : Tensor(shape.dimensions_, f, std::forward<Arguments>(args)...) {}

  /** Copy construction, allocates memory and copies from `tensor` */
  Tensor(Tensor<T, N, ContainerType> const &tensor); 

  /** Move construction, takes ownership of underlying data, `tensor` is destroyed */
  Tensor(Tensor<T, N, ContainerType> &&tensor); 

  /** Constructs a reference to the `proxy` tensor. The tensors share 
   *  the same underyling data, so changes will affect both tensors.
   */

  Tensor(typename Tensor<T, N, ContainerType>::Proxy const &proxy); 

  /** Constructs the tensor produced by the expression */
  template <typename NodeType,
            typename = typename std::enable_if<NodeType::rank() == N>::type>
  Tensor(Expression<NodeType> const& expression);

  /* ------------------- Assignment ------------------- */

  /** Copy Constructs from `tensor`. Destroys itself first */
  Tensor<T, N, ContainerType> &operator=(Tensor<T, N, ContainerType> const &tensor);

  /** Evaluates `rhs` and move constructs. Destroys itself first */
  template <typename NodeType>
  Tensor<T, N, ContainerType> &operator=(Expression<NodeType> const &rhs);

  /* ----------------- Getters ----------------- */

  constexpr static size_t rank() { return N; } /**< Get `N` */

  /** Get the dimension at index. Throws std::logic_error if index
   *  is out of bounds. Note: indexing starts at 1.
   */
  size_t dimension(size_t index) const { return shape_[index]; }

  Shape<N> const &shape() const noexcept { return shape_; } /**< Get the tensor shape */
  // FIXME :: Is there a way to hide these and keep -> functional?
  Tensor *operator->() { return this; } /**> used to implement iterator-> */
  Tensor const *operator->() const { return this; } /**> used to implement const_iterator-> */

  /* ------------------ Access To Data ----------------- */

  template <typename... Indices>
  Tensor<T, N - sizeof...(Indices), ContainerType> at(Indices... args);

  /** Returns the resulting tensor by applying left to right index expansion of
   *  the provided arguments. I.e. calling `tensor(1, 2)` on a rank 4 tensor is
   *  equivalent to `tensor(1, 2, :, :)`. Throws std::logic_error if any of the 
   *  indices are out bounds. Note: indexing starts at 1.
   */
  template <typename... Indices,
            typename = typename std::enable_if<N != sizeof...(Indices)>::type>
  Tensor<T, N - sizeof...(Indices), ContainerType> operator()(Indices... args);

  template <typename... Indices,
            typename = typename std::enable_if<N == sizeof...(Indices)>::type>
  T &operator()(Indices... args);

  template <typename... Indices>
  Tensor<T, N - sizeof...(Indices), ContainerType> const at(Indices... args) const;

  /** See operator() */
  template <typename... Indices,
            typename = typename std::enable_if<N != sizeof...(Indices)>::type>
  Tensor<T, N - sizeof...(Indices), ContainerType> const operator()(Indices... args) const;

  /** See operator() */
  template <typename... Indices,
            typename = typename std::enable_if<N == sizeof...(Indices)>::type>
  T const &operator()(Indices... args) const;

  /** See operator() */
  template <size_t M>//, typename = typename std::enable_if<N != M>::type>
  Tensor<T, N - M, ContainerType> operator[](Indices<M> const &indices);

  /** Slices denotate the dimensions which are left free, while indices
   *  fix the remaining dimensions at the specified index. I.e. calling
   *  `tensor.slice<1, 3, 5>(1, 2)` on a rank 5 tensor is equivalent to
   *  `tensor(:, 1, :, 2, :)` and will produce a rank 3 tensor. Throws
   *   std::logic_error if any of the indices are out of bounds. Note:
   *   indexing begins at 1.
   */
  template <size_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices), ContainerType> slice(Indices... indices);
  template <size_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices), ContainerType> const slice(Indices... indices) const;

  /* -------------------- Expressions ------------------- */

  template <typename X, typename Y, size_t M, typename CType_, typename FunctionType>
  friend Tensor<X, M, CType_> elem_wise(Tensor<X, M, CType_> const &tensor, Y const &scalar,
      FunctionType &&fn);

  template <typename X, typename Y, size_t M, typename CType_, typename FunctionType>
  friend Tensor<X, M, CType_> elem_wise(Tensor<X, M, CType_> const &tensor1, Tensor<Y, M, CType_> const &tensor_2, FunctionType &&fn);

  template <typename X, typename Y, size_t M, typename C1, typename C2>
  friend Tensor<X, M, C1> add(
      Tensor<X, M, C1> const& tensor_1, 
      Tensor<Y, M, C2> const& tensor_2);

  template <typename RHS>
  Tensor<T, N, ContainerType> &operator+=(Expression<RHS> const &rhs);

  template <typename X, typename Y, size_t M, typename CType_>
  friend Tensor<X, M, CType_> subtract(Tensor<X, M, CType_> const& tensor_1, Tensor<Y, M, CType_> const& tensor_2);

  template <typename RHS>
  Tensor<T, N, ContainerType> &operator-=(Expression<RHS> const &rhs);

  template <typename X, typename Y, size_t M1, size_t M2, typename CType_>
  friend Tensor<X, M1 + M2 - 2, CType_> multiply(Tensor<X, M1, CType_> const& tensor_1, Tensor<Y, M2, CType_> const& tensor_2);

  template <typename RHS>
  Tensor<T, N, ContainerType> &operator*=(Expression<RHS> const &rhs);

  /** Allocates a Tensor with shape equivalent to *this, and whose
   *  elements are equivalent to *this with operator-() applied.
   */
  Tensor<T, N, ContainerType> operator-() const;

  /* ------------------ Print to ostream --------------- */

  template <typename X, size_t M, typename CType_>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M, CType_> &tensor);

  /* -------------------- Equivalence ------------------ */

  /** Returns true iff the tensor's dimensions and data are equivalent */
  bool operator==(Tensor<T, N, ContainerType> const& tensor) const; 

  /** Returns true iff the tensor's dimensions or data are not equivalent */
  bool operator!=(Tensor<T, N, ContainerType> const& tensor) const { return !(*this == tensor); }

  /** Returns true iff the tensor's dimensions are equal and every element satisfies e1 == e2 */
  template <typename X>
  bool operator==(Tensor<X, N, ContainerType> const& tensor) const;

  /** Returns true iff the tensor's dimensions are different or any element satisfies e1 != e2 */
  template <typename X>
  bool operator!=(Tensor<X, N, ContainerType> const& tensor) const { return !(*this == tensor); }

  /* -------------------- Iterators --------------------- */

  class Iterator { /*@Iterator<T, N>*/
  public:
    /** Iterator with freedom across one dimension of a Tensor.
     *  Allows access to the underlying tensor data.
     */

    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, typename CType_> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    Iterator(Iterator const &it);  

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    Iterator(Iterator &&it);       

    Tensor<T, N, ContainerType> operator*();   /**< Create a reference to the underlying Tensor */
    Tensor<T, N, ContainerType> operator->();  /**< Syntatic sugar for (*it). */
    Iterator operator++(int);   /**< Increment (postfix). Returns a temporary before increment */
    Iterator &operator++();     /**< Increment (prefix). Returns *this */
    Iterator operator--(int);   /**< Decrement (postfix). Returns a temporary before decrement */
    Iterator &operator--();     /**< Decrement (prefix). Returns *this */

    /** Returns true iff the shapes and underlying pointers are identical */
    bool operator==(Iterator const &it) const;

    /** Returns true iff the shapes or underlying pointers are not identical */
    bool operator!=(Iterator const &it) const { return !(it == *this); }

  private:

    // Direct construction
    Iterator(Tensor<T, N + 1, ContainerType> const &tensor, size_t index);
    Shape<N> shape_; // Data describing the underlying tensor 
    size_t strides_[N];
    size_t offset_;
    std::shared_ptr<ContainerType> ref_;
    size_t stride_; // Step size of the underlying data pointer per increment
  };

  class ConstIterator { /*@ConstIterator<T, N>*/
  public:
    /** Constant iterator with freedom across one dimension of a Tensor.
     *  Does not allow write access to the underlying tensor data.
     */

    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, typename CType_> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    ConstIterator(ConstIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ConstIterator(ConstIterator &&it);

    Tensor<T, N, ContainerType> const operator*();  /**< Create a reference to the underlying Tensor */
    Tensor<T, N, ContainerType> const operator->(); /**< Syntatic sugar for (*it). */                                
    ConstIterator operator++(int);   /**< Increment (postfix). Returns a temporary before increment */
    ConstIterator &operator++();     /**< Increment (prefix). Returns *this */
    ConstIterator operator--(int);   /**< Decrement (postfix). Returns a temporary before decrement */
    ConstIterator &operator--();     /**< Decrement (prefix). Returns *this */
    bool operator==(ConstIterator const &it) const;
    bool operator!=(ConstIterator const &it) const { return !(it == *this); }
  private:
    ConstIterator(Tensor<T, N + 1, ContainerType> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    size_t offset_;
    std::shared_ptr<ContainerType> ref_;

    /**
     * Step size of the underlying data pointer per increment
     */
    size_t stride_;
  };

  class ReverseIterator { /*@ReverseIterator<T, N>*/
  public:
    /** Reverse iterator with freedom across one dimension of a Tensor.
     *  Does not allow write access to the underlying tensor data.
     */

    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, typename CType_> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    ReverseIterator(ReverseIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ReverseIterator(ReverseIterator &&it);

    Tensor<T, N, ContainerType> operator*();        /**< Create a reference to the underlying Tensor */
    Tensor<T, N, ContainerType> operator->();       /**< Syntatic sugar for (*it). */                                
    ReverseIterator operator++(int); /**< Increment (postfix). Returns a temporary before increment */
    ReverseIterator &operator++();   /**< Increment (prefix). Returns *this */
    ReverseIterator operator--(int); /**< Decrement (postfix). Returns a temporary before decrement */
    ReverseIterator &operator--();   /**< Decrement (prefix). Returns *this */
    bool operator==(ReverseIterator const &it) const;
    bool operator!=(ReverseIterator const &it) const { return !(it == *this); }
  private:
    ReverseIterator (Tensor<T, N + 1, ContainerType> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    size_t offset_;
    std::shared_ptr<ContainerType> ref_;

    /**
     * Step size of the underlying data pointer per increment
     */
    size_t stride_;
  };

  class ConstReverseIterator { /*@ConstReverseIterator<T, N>*/
  public:
    /** Constant reverse iterator with freedom across one dimension of a Tensor.
     *  Does not allow write access to the underlying tensor data.
     */

    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, typename CType_> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy constructs an iterator to the same underlying Tensor */
    ConstReverseIterator(ConstReverseIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ConstReverseIterator(ConstReverseIterator &&it);
    Tensor<T, N, ContainerType> const operator*();       /**< Create a reference to the underlying Tensor */
    Tensor<T, N, ContainerType> const operator->();      /**< Syntatic sugar for (*it). */                                
    ConstReverseIterator operator++(int); /**< Increment (postfix). Returns a temporary before increment */
    ConstReverseIterator &operator++();   /**< Increment (prefix). Returns *this */
    ConstReverseIterator operator--(int); /**< Decrement (postfix). Returns a temporary before decrement */
    ConstReverseIterator &operator--();   /**< Decrement (prefix). Returns *this */
    bool operator==(ConstReverseIterator const &it) const;
    bool operator!=(ConstReverseIterator const &it) const { return !(it == *this); }
  private:
    ConstReverseIterator (Tensor<T, N + 1, ContainerType> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    size_t offset_;
    std::shared_ptr<ContainerType> ref_;

    /**
     * Step size of the underlying data pointer per increment
     */
    size_t stride_;
  };

  /** Returns an iterator for a Tensor, equivalent to *this dimension
   *  fixed at index (the iteration index). Note: indexing begins at 
   *  1. std::logic_error will be thrown if `index` is out of bounds.
   */
  typename Tensor<T, N - 1, ContainerType>::Iterator begin(size_t index);

  /** Returns a just-past-the-end iterator for a Tensor, equivalent 
   * to *this dimension fixed at index (the iteration index). 
   * Note: indexing begins at 1. std::logic_error will be thrown 
   * if `index` is out of bounds.
   */
  typename Tensor<T, N - 1, ContainerType>::Iterator end(size_t index);

  /** Equivalent to Tensor<T, N, ContainerType>::begin(0) */
  typename Tensor<T, N - 1, ContainerType>::Iterator begin();
  /** Equivalent to Tensor<T, N, ContainerType>::end(0) */
  typename Tensor<T, N - 1, ContainerType>::Iterator end();

  /** See Tensor<T, N, ContainerType>::begin(size_t), except returns a const iterator */
  typename Tensor<T, N - 1, ContainerType>::ConstIterator cbegin(size_t index) const;
  /** See Tensor<T, N, ContainerType>::end(size_t), except returns a const iterator */
  typename Tensor<T, N - 1, ContainerType>::ConstIterator cend(size_t index) const;
  /** See Tensor<T, N, ContainerType>::begin(), except returns a const iterator */
  typename Tensor<T, N - 1, ContainerType>::ConstIterator cbegin() const;
  /** See Tensor<T, N, ContainerType>::end(), except returns a const iterator */
  typename Tensor<T, N - 1, ContainerType>::ConstIterator cend() const;

  /** See Tensor<T, N, ContainerType>::begin(size_t), except returns a reverse iterator */
  typename Tensor<T, N - 1, ContainerType>::ReverseIterator rbegin(size_t index);
  /** See Tensor<T, N, ContainerType>::end(size_t), except returns a reverse iterator */
  typename Tensor<T, N - 1, ContainerType>::ReverseIterator rend(size_t index);
  /** See Tensor<T, N, ContainerType>::begin(), except returns a reverse iterator */
  typename Tensor<T, N - 1, ContainerType>::ReverseIterator rbegin();
  /** See Tensor<T, N, ContainerType>::end(), except returns a reverse iterator */
  typename Tensor<T, N - 1, ContainerType>::ReverseIterator rend();

  /** See Tensor<T, N, ContainerType>::begin(size_t), except returns a const reverse iterator */
  typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator crbegin(size_t index) const;
  /** See Tensor<T, N, ContainerType>::end(size_t), except returns a const reverse iterator */
  typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator crend(size_t index) const;
  /** See Tensor<T, N, ContainerType>::begin(), except returns a const reverse iterator */
  typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator crbegin() const;
  /** See Tensor<T, N, ContainerType>::end(), except returns a const reverse iterator */
  typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator crend() const;

  /* ----------------- Utility Functions ---------------- */
  /** Returns a deep copy of this tensor, equivalent to calling copy constructor */
  Tensor<T, N, ContainerType> copy() const; 

  /** Returns a reference: only used to invoke reference constructor */
  typename Tensor<T, N, ContainerType>::Proxy ref();

  template <typename U, size_t M, typename CType_, typename RAIt>
  friend void Fill(Tensor<U, M, CType_> &tensor, RAIt const &begin, RAIt const &end);

  template <typename U, size_t M, typename CType_, typename X>
  friend void Fill(Tensor<U, M, CType_> &tensor, X const &value);

  /** Allocates a Tensor with shape `shape`, whose total number of elements 
   *  must be equivalent to *this (or std::logic_error is thrown). The 
   *  resulting Tensor is filled by iterating through *this and copying
   *  over the values.
   */
  template <size_t M>
  Tensor<T, M, ContainerType> resize(Shape<M> const &shape) const;

  template <typename U, typename CType_> friend Tensor<U, 2, CType_> transpose(Tensor<U, 2, CType_> &mat);
  template <typename U, typename CType_> friend Tensor<U, 2, CType_> transpose(Tensor<U, 1, CType_> &vec);

private:
  /* ----------------- data ---------------- */

  Shape<N> shape_;
  size_t strides_[N];
  size_t offset_;
  std::shared_ptr<ContainerType> ref_;

  /* --------------- Getters --------------- */

  size_t const *strides() const noexcept { return strides_; }

  /* ------------- Expansion for operator()() ------------- */

  template <size_t M, typename... Indices>
  size_t pIndicesExpansion(Indices... args);

  /* ------------- Expansion for slice() ------------- */

  // Expansion
  template <size_t M, typename... Indices>
  Tensor<T, N - M, ContainerType> pSliceExpansion(size_t * placed_indices, size_t array_index, size_t next_index, Indices... indices);
  template <size_t M>
  Tensor<T, N - M, ContainerType> pSliceExpansion(size_t * placed_indices, size_t);

  // Index checking and placement
  template <size_t I1, size_t I2, size_t... Indices>
  static void pSliceIndex(size_t *placed_indices);
  template <size_t I1>
  static void pSliceIndex(size_t *placed_indices);

  /* ----------------- Utility -------------------- */

  // Copy the dimensions of a C multi-dimensional array 
  template <typename Array, size_t Index, size_t Limit>
  struct SetDimensions {
    void operator()(size_t (&dimensions)[N]) {
      dimensions[Index] = std::extent<Array, Index>::value;
      SetDimensions<Array, Index + 1, Limit>{}(dimensions);
    }
  };
  
  // Base condition
  template <typename Array, size_t Limit>
  struct SetDimensions<Array, Limit, Limit> {
    void operator()(size_t (&)[N]) {}
  };

  // Used to wrap the index with a Tensor reference when calling pUpdateIndices
  struct IndexReference {
    template <typename U, size_t M, typename CType_> friend class Tensor;
    IndexReference(Tensor<T, N, ContainerType> const &_tensor)
      : index(0), tensor(_tensor) {}
    int index;
    Tensor<T, N, ContainerType> const &tensor;
  };

  // Data mapping for pMap
  template <size_t M>
  static void pUpdateIndices(size_t (&indices)[M], IndexReference &index, size_t quota_offset = 0);

  // Data mapping for pUnaryMap -- ASSUMES EQUAL SHAPES
  template <typename U, size_t M, typename CType_>
  static void pUpdateIndices(size_t (&indices)[M],
    typename Tensor<T, N, ContainerType>::IndexReference &index1,
    typename Tensor<U, N, CType_>::IndexReference &index2);

  // Data mapping for pBinaryMap -- ASSUMES EQUAL SHAPES
  template <typename U, typename V, size_t M, typename CType1, typename CType2>
  static void pUpdateIndices(size_t (&indices)[M], 
    typename Tensor<T, N, ContainerType>::IndexReference &index1,
    typename Tensor<U, N, CType1>::IndexReference &index2, 
    typename Tensor<V, N, CType2>::IndexReference &index3);

  void pMap(std::function<void(T *lhs)> const &fn);
  void pMap(std::function<void(T const &lhs)> const &fn) const;

  template <typename U, typename CType_>
  void pUnaryMap(Tensor<U, N, CType_> const &tensor, std::function<void(T *lhs, U *rhs)> const &fn);

  template <typename U, typename V, typename C1, typename C2>
  void pBinaryMap(Tensor<U, N, C1> const &tensor_1, Tensor<V, N, C2> const &tensor_2,
      std::function<void(T *lhs, U *rhs1, V *rhs2)> const& fn);

  // Initialize strides :: DIMENSIONS MUST BE INITIALIZED FIRST
  void pInitializeStrides();

  // Declare all fields in the constructor, but initialize strides
  // assuming no gaps
  Tensor(size_t const *dimensions, size_t offset, std::shared_ptr<ContainerType> &&_ref);

  // Declare all fields in the constructor
  Tensor(size_t const *dimensions, size_t const *strides, size_t offset, std::shared_ptr<ContainerType> &&_ref);

}; // Tensor

/* ----------------------------- Constructors ------------------------- */

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Tensor(std::initializer_list<size_t> dimensions)
  : shape_(Shape<N>(dimensions)), offset_(0),
  ref_(std::make_shared<ContainerType>(shape_.index_product()))
{ 
  pInitializeStrides(); 
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Tensor(size_t const (&dimensions)[N], T const& value)
  : shape_(Shape<N>(dimensions)), offset_(0)
{
  for (size_t i = 0; i < N; ++i)
    assert(dimensions[i] && ZERO_ELEMENT);
  pInitializeStrides();
  size_t cumul = shape_.index_product();
  ref_ = std::make_shared<ContainerType>(cumul, value);
}

template <typename T, size_t N, typename ContainerType>
template <typename FunctionType, typename... Arguments>
Tensor<T, N, ContainerType>::Tensor(size_t const (&dimensions)[N], std::function<FunctionType> &f, Arguments&&... args)
  : shape_(Shape<N>(dimensions)), offset_(0)
{
  for (size_t i = 0; i < N; ++i)
    assert(dimensions[i] && ZERO_ELEMENT);
  pInitializeStrides();
  std::function<void(T*)> value_setter = 
    [&f, &args...](T *lhs) -> void { *lhs = f(args...); };
  ref_ = std::make_shared<ContainerType>(shape_.index_product());
  pMap(value_setter);
}

template <typename T, size_t N, typename ContainerType>
template <typename Array>
Tensor<T, N, ContainerType>::Tensor(_A<Array> &&md_array): offset_(0)
{
  static_assert(std::rank<Array>::value == N, RANK_MISMATCH);
  using ArrayType = typename std::remove_all_extents<Array>::type;
  SetDimensions<Array, 0, N>{}(shape_.dimensions_);
  pInitializeStrides();
  // Make use of the fact C multi-dimensional arrays are 
  // allocated in memory contiguously
  ArrayType *ptr = (ArrayType *)md_array.value;
  std::function<void(T*)> value_setter = 
    [&ptr](T *lhs) -> void { *lhs = *(ptr++); };
  ref_ = std::make_shared<ContainerType>(shape_.index_product());
  pMap(value_setter);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Tensor(Shape<N> const &shape)
  : shape_(shape), offset_(0), 
    ref_(std::make_shared<ContainerType>(shape_.index_product()))
{
  pInitializeStrides();
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Tensor(Tensor<T, N, ContainerType> const &tensor)
  : shape_(tensor.shape_), offset_(0)
{
  pInitializeStrides();
  size_t cumul = shape_.index_product();
  std::function<void(T*, T*)> fn = [](T *x, T *y) -> void { *x = *y; };
  ref_ = std::make_shared<ContainerType>(cumul);
  pUnaryMap(tensor, fn);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Tensor(Tensor<T, N, ContainerType> &&tensor)
  : shape_(tensor.shape_), offset_(tensor.offset_), ref_(std::move(tensor.ref_))
{
  std::copy_n(tensor.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Tensor(typename Tensor<T, N, ContainerType>::Proxy const &proxy)
  : shape_(proxy.tensor_.shape_), offset_(proxy.tensor_.offset_), ref_(proxy.tensor_.ref_)
{
  std::copy_n(proxy.tensor_.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
template <typename NodeType, typename>
Tensor<T, N, ContainerType>::Tensor(Expression<NodeType> const& expression)
{
  auto const &expression_tree = expression.self()();
  /*
  shape_ = result.shape();
  std::function<void(T*)> value_setter = 
    [&expression_tree](T *lhs) -> void { *lhs = f(args...); };
  */
  shape_ = expression_tree.shape();
  pInitializeStrides();
  ref_ = std::move(expression_tree.ref_);
  offset_ = 0;
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> &Tensor<T, N, ContainerType>::operator=(const Tensor<T, N, ContainerType> &tensor)
{
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
  std::function<void(T*, T*)> fn = [](T *x, T *y) -> void { *x = *y; };
  pUnaryMap(tensor, fn);
  return *this;
}

template <typename T, size_t N, typename ContainerType>
template <typename NodeType>
Tensor<T, N, ContainerType> &Tensor<T, N, ContainerType>::operator=(Expression<NodeType> const &rhs)
{
  auto tensor = rhs.self()();
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
  std::function<void(T *, typename NodeType::value_type*)> fn =
    [](T *x, typename NodeType::value_type *y) -> void { *x = *y; };
  pUnaryMap<typename NodeType::value_type>(tensor, fn);
  return *this;
}

template <typename T, size_t N, typename ContainerType>
template <typename... Indices>
Tensor<T, N - sizeof...(Indices), ContainerType> Tensor<T, N, ContainerType>::at(Indices... args)
{
  constexpr size_t M = sizeof...(args); 
  size_t cumul_index = pIndicesExpansion<M>(args...);
  return Tensor<T, N - M, ContainerType>(shape_.dimensions_ + M, strides_ + M, offset_ + cumul_index, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
template <typename... Indices, typename>
Tensor<T, N - sizeof...(Indices), ContainerType> Tensor<T, N, ContainerType>::operator()(Indices... args)
{
  constexpr size_t M = sizeof...(args); 
  size_t cumul_index = pIndicesExpansion<M>(args...);
  return Tensor<T, N - M, ContainerType>(shape_.dimensions_ + M, strides_ + M, offset_ + cumul_index, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
template <typename... Indices, typename>
T &Tensor<T, N, ContainerType>::operator()(Indices... args)
{
  constexpr size_t M = sizeof...(args); 
  size_t cumul_index = pIndicesExpansion<M>(args...);
  return (*ref_)[cumul_index + offset_];
}

template <typename T, size_t N, typename ContainerType>
template <typename... Indices>
Tensor<T, N - sizeof...(Indices), ContainerType> const Tensor<T, N, ContainerType>::at(Indices... args) const
{
 return (*const_cast<self_type*>(this))(args...);
}

template <typename T, size_t N, typename ContainerType>
template <typename... Indices, typename>
Tensor<T, N - sizeof...(Indices), ContainerType> const Tensor<T, N, ContainerType>::operator()(Indices... args) const
{
  return (*const_cast<self_type*>(this))(args...);
}

template <typename T, size_t N, typename ContainerType>
template <typename... Indices, typename>
T const &Tensor<T, N, ContainerType>::operator()(Indices... args) const
{
  return (*const_cast<self_type*>(this))(args...);
}

template <typename T, size_t N, typename ContainerType>
template <size_t M>//, typename>
Tensor<T, N - M, ContainerType> Tensor<T, N, ContainerType>::operator[](Indices<M> const &indices)
{
  size_t cumul_index = 0;
  for (size_t i = 0; i < M; ++i)
    cumul_index += strides_[N - i - 1] * indices.indices_[M - i - 1];
  return Tensor<T, N - M, ContainerType>(shape_.dimensions_ + M, strides_ + M,  offset_ + cumul_index, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
template <size_t... Slices, typename... Indices>
Tensor<T, sizeof...(Slices), ContainerType> Tensor<T, N, ContainerType>::slice(Indices... indices)
{
  static_assert(sizeof...(Slices), SLICES_OUT_OF_BOUNDS);
  static_assert(N == sizeof...(Slices) + sizeof...(indices), SLICES_OUT_OF_BOUNDS);
  size_t placed_indices[N];
  // Initially fill the array with 1s
  // place 0s where the indices are sliced
  std::fill_n(placed_indices, N, 1);
  this->pSliceIndex<Slices...>(placed_indices);
  size_t index = 0;
  for (; index < N && !placed_indices[index]; ++index);
  return pSliceExpansion<sizeof...(indices)>(placed_indices, index, indices...);
}

template <typename T, size_t N, typename ContainerType>
template <size_t... Slices, typename... Indices>
Tensor<T, sizeof...(Slices), ContainerType> const Tensor<T, N, ContainerType>::slice(Indices... indices) const
{
  return const_cast<self_type*>(this)->slice<Slices...>(indices...);
}

template <typename T, size_t N, typename ContainerType>
bool Tensor<T, N, ContainerType>::operator==(Tensor<T, N, ContainerType> const& tensor) const
{
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);

  size_t indices_product = shape_.index_product();
  for (size_t i = 0; i < indices_product; ++i)
    if ((*ref_)[i + offset_] != (*tensor.ref_)[i + tensor.offset_]) return false;
  return true;
}

template <typename T, size_t N, typename ContainerType>
template <typename X>
bool Tensor<T, N, ContainerType>::operator==(Tensor<X, N, ContainerType> const& tensor) const
{
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);

  size_t indices_product = shape_.index_product();
  for (size_t i = 0; i < indices_product; ++i)
    if ((*ref_)[i + offset_] != (*tensor.ref_)[i + tensor.offset_]) return false;
  return true;
}

/** Prints `tensor` to an ostream, using square braces "[]" to denotate
 *  dimensions. I.e. a 1x1x1 Tensor with element x will appear as [[[x]]]
 */
template <typename T, size_t N, typename CType_>
std::ostream &operator<<(std::ostream &os, const Tensor<T, N, CType_> &tensor)
{
  auto add_brackets = [&os](size_t n, bool left) -> void {
    for (size_t i = 0; i < n; ++i) os << (left ?'[' : ']');
  };
  size_t cumul_index = tensor.shape_.index_product();
  size_t dim_quotas[N];
  std::copy_n(tensor.shape_.dimensions_, N, dim_quotas);
  size_t index = 0;

  add_brackets(N, true);
  os << (*tensor.ref_)[tensor.offset_]; 
  for (size_t i = 0; i < cumul_index - 1; ++i) {
    size_t bracket_count = 0;
    bool propogate = true;
    size_t dim_index = N - 1;
    // find the correct index to "step" to
    while (propogate) {
      --dim_quotas[dim_index];
      ++bracket_count;
      index += tensor.strides_[dim_index];
      if (!dim_quotas[dim_index]) {
        dim_quotas[dim_index] = tensor.shape_.dimensions_[dim_index];
        index -= dim_quotas[dim_index] * tensor.strides_[dim_index];
      } else {
        propogate = false;
      }
      --dim_index;
    }
    add_brackets(bracket_count - 1, false);
    os << ", ";
    add_brackets(bracket_count - 1, true);
    os << (*tensor.ref_)[index + tensor.offset_];
  }
  add_brackets(N, false); // closing brackets
  return os;
}

/* ----------- Expansion for operator()() ----------- */

template <typename T, size_t N, typename ContainerType>
template <size_t M, typename... Indices>
size_t Tensor<T, N, ContainerType>::pIndicesExpansion(Indices... args) 
{
  static_assert(N >= M, RANK_OUT_OF_BOUNDS);
  size_t index = M - 1;
  auto convert_index = [&](size_t dim) -> size_t {
    assert((dim < shape_.dimensions_[M - index - 1]) && INDEX_OUT_OF_BOUNDS);
    return strides_[M - (index--) - 1] * dim;
  }; 
  size_t cumul_index = 0;
#ifdef _MSC_VER
  // MSVC parses {} as an initializer_list
  (void)std::initializer_list<int>{(cumul_index += convert_index(args), 0)...};
#else
  (void)(int[]){(cumul_index += convert_index(args), 0)...};
#endif
  // surpress compiler unused lval warnings
  (void)(convert_index);
  return cumul_index;
}

/* ------------- Slice Expansion ------------- */

template <typename T, size_t N, typename ContainerType>
template <size_t I1, size_t I2, size_t... Indices>
void Tensor<T, N, ContainerType>::pSliceIndex(size_t *placed_indices)
{
  static_assert(N > I1, INDEX_OUT_OF_BOUNDS);
  static_assert(I1 != I2, SLICE_INDICES_REPEATED);
  static_assert(I1 < I2, SLICE_INDICES_DESCENDING);
  placed_indices[I1] = 0;
  pSliceIndex<I2, Indices...>(placed_indices);
}

template <typename T, size_t N, typename ContainerType>
template <size_t Index>
void Tensor<T, N, ContainerType>::pSliceIndex(size_t *placed_indices)
{
  static_assert(N > Index, INDEX_OUT_OF_BOUNDS);
  placed_indices[Index] = 0;
}

template <typename T, size_t N, typename ContainerType>
template <size_t M, typename... Indices>
Tensor<T, N - M, ContainerType> Tensor<T, N, ContainerType>::pSliceExpansion(size_t * placed_indices, size_t array_index, size_t next_index, Indices... indices)
{
  assert((shape_.dimensions_[array_index] > next_index) && INDEX_OUT_OF_BOUNDS);
  // 0 is reserved for the "sliced" indices, so add one to all
  // of the accessed indices to differentiate
  placed_indices[array_index] = next_index + 1;
  ++array_index;
  for (; array_index < N && !placed_indices[array_index]; ++array_index);
  return pSliceExpansion<M>(placed_indices, array_index, indices...);
}

template <typename T, size_t N, typename ContainerType>
template <size_t M>
Tensor<T, N - M, ContainerType> Tensor<T, N, ContainerType>::pSliceExpansion(size_t *placed_indices, size_t)
{
  size_t offset = 0;
  size_t dimensions[N - M];
  size_t strides[N - M];
  size_t array_index = 0;
  for (size_t i = 0; i < N; ++i) {
    if (placed_indices[i]) {
      // accessed indices were offset by +1 to differentiate
      // between sliced indices, so convert back
      offset += (placed_indices[i] - 1) * strides_[i];
    } else {
      strides[array_index] = strides_[i];
      dimensions[array_index] = shape_.dimensions_[i];
      ++array_index;
    }
  }
  return Tensor<T, N - M, ContainerType>(dimensions, strides, offset_ + offset, std::shared_ptr<ContainerType>(ref_));
}

/* ------------ Utility Methods ------------ */

// Update the quotas after one iterator increment
template <typename T, size_t N, typename ContainerType>
template <size_t M>
void Tensor<T, N, ContainerType>::pUpdateIndices(
    size_t (&indices)[M], IndexReference &index, size_t quota_offset)
{
  int dim_index = M - 1;
  bool propogate = true;
  while (dim_index >= 0 && propogate) {
    ++indices[dim_index];
    index.index += index.tensor.strides_[dim_index + quota_offset];
    if (indices[dim_index] == index.tensor.shape_.dimensions_[dim_index + quota_offset]) {
      indices[dim_index] = 0;
      index.index -= 
        index.tensor.shape_.dimensions_[dim_index + quota_offset] * index.tensor.strides_[dim_index + quota_offset];
    } else {
      propogate = false;
    }
    --dim_index;
  }
}

template <typename T, size_t N, typename ContainerType>
template <typename U, size_t M, typename CType_>
void Tensor<T, N, ContainerType>::pUpdateIndices(
    size_t (&indices)[M],
    typename Tensor<T, N, ContainerType>::IndexReference &index1, 
    typename Tensor<U, N, CType_>::IndexReference &index2)
{
  int dim_index = M - 1;
  bool propogate = true;
  while (dim_index >= 0 && propogate) {
    ++indices[dim_index];
    index1.index += index1.tensor.strides_[dim_index];
    index2.index += index2.tensor.strides_[dim_index];
    if (indices[dim_index] == index1.tensor.shape_.dimensions_[dim_index]) {
      indices[dim_index] = 0;
      index1.index -= index1.tensor.shape_.dimensions_[dim_index] 
        * index1.tensor.strides_[dim_index];
      index2.index -= index2.tensor.shape_.dimensions_[dim_index] 
        * index2.tensor.strides_[dim_index];
    } else {
      propogate = false;
    }
    --dim_index;
  }
}

template <typename T, size_t N, typename ContainerType>
template <typename U, typename V, size_t M, typename CType1, typename CType2>
void Tensor<T, N, ContainerType>::pUpdateIndices(
    size_t (&indices)[M],
    IndexReference &index1,
    typename Tensor<U, N, CType1>::IndexReference &index2, 
    typename Tensor<V, N, CType2>::IndexReference &index3)
{
  int dim_index = M - 1;
  bool propogate = true;
  while (dim_index >= 0 && propogate) {
    ++indices[dim_index];
    index1.index += index1.tensor.strides_[dim_index];
    index2.index += index2.tensor.strides_[dim_index];
    index3.index += index3.tensor.strides_[dim_index];
    if (indices[dim_index] == index1.tensor.shape_.dimensions_[dim_index]) {
      indices[dim_index] = 0;
      index1.index -= index1.tensor.shape_.dimensions_[dim_index] 
        * index1.tensor.strides_[dim_index];
      index2.index -= index2.tensor.shape_.dimensions_[dim_index] 
        * index2.tensor.strides_[dim_index];
      index3.index -= index3.tensor.shape_.dimensions_[dim_index] 
        * index3.tensor.strides_[dim_index];
    } else {
      propogate = false;
    }
    --dim_index;
  }
}

template <typename T, size_t N, typename ContainerType>
void Tensor<T, N, ContainerType>::pMap(std::function<void(T *lhs)> const &fn)
{
  // this is the index upper bound for iteration
  size_t cumul_index = shape_.index_product();
  size_t indices[N];
  std::fill_n(indices, N, 0);
  typename Tensor<T, N, ContainerType>::IndexReference index{*this};
  for (size_t i = 0; i < cumul_index; ++i) {
    fn(&(*ref_)[offset_ + index.index]);
    pUpdateIndices(indices, index);
  }
}

template <typename T, size_t N, typename ContainerType>
void Tensor<T, N, ContainerType>::pMap(std::function<void(T const &lhs)> const &fn) const
{
  // this is the index upper bound for iteration
  size_t cumul_index = shape_.index_product();
  size_t indices[N];
  std::fill_n(indices, N, 0);
  typename Tensor<T, N, ContainerType>::IndexReference index{*this};
  for (size_t i = 0; i < cumul_index; ++i) {
    fn((*ref_)[offset_ + index.index]);
    pUpdateIndices(indices, index);
  }
}

template <typename T, size_t N, typename ContainerType>
template <typename U, typename CType_>
void Tensor<T, N, ContainerType>::pUnaryMap(Tensor<U, N, CType_> const &tensor,
    std::function<void(T *lhs, U *rhs)> const &fn)
{
  // this is the index upper bound for iteration
  size_t cumul_index = shape_.index_product();
  size_t indices[N];
  std::fill_n(indices, N, 0); 
  typename Tensor<T, N, ContainerType>::IndexReference index{*this};
  typename Tensor<U, N, CType_>::IndexReference t_index{tensor};
  for (size_t i = 0; i < cumul_index; ++i) {
    fn(&((*ref_)[offset_ + index.index]), &((*tensor.ref_)[tensor.offset_ + t_index.index]));
    pUpdateIndices<U, N, CType_>(indices, index, t_index);
  }
}

template <typename T, size_t N, typename ContainerType>
template <typename U, typename V, typename C1, typename C2>
void Tensor<T, N, ContainerType>::pBinaryMap(Tensor<U, N, C1> const &tensor_1, 
    Tensor<V, N, C2> const &tensor_2, 
    std::function<void(T *lhs, U *rhs1, V *rhs2)> const &fn)
{
  size_t cumul_index = shape_.index_product();
  size_t indices[N];
  std::fill_n(indices, N, 0); 
  typename Tensor<T, N, ContainerType>::IndexReference index{*this};
  typename Tensor<U, N, C1>::IndexReference t1_index{tensor_1};
  typename Tensor<V, N, C2>::IndexReference t2_index{tensor_2};
  for (size_t i = 0; i < cumul_index; ++i) {
    fn(&(*ref_)[offset_ + index.index], 
        &(*tensor_1.ref_)[tensor_1.offset_ + t1_index.index], 
        &(*tensor_2.ref_)[tensor_2.offset_ + t2_index.index]);
    pUpdateIndices<U, V, N, C1, C2>(indices, index, t1_index, t2_index);
  }
}

template <typename T, size_t N, typename ContainerType>
void Tensor<T, N, ContainerType>::pInitializeStrides()
{
  size_t accumulator = 1;
  for (size_t i = 0; i < N; ++i) {
    strides_[N - i - 1] = accumulator;
    accumulator *= shape_.dimensions_[N - i - 1];
  }
}

// private constructors
template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Tensor(size_t const *dimensions, size_t offset, std::shared_ptr<ContainerType> &&ref)
  : shape_(Shape<N>(dimensions)), offset_(offset), ref_(std::move(ref))
{
  pInitializeStrides();
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Tensor(size_t const *dimensions, size_t const *strides, size_t offset, 
    std::shared_ptr<ContainerType> &&ref)
  : shape_(Shape<N>(dimensions)), offset_(offset), ref_(std::move(ref))
{
  std::copy_n(strides, N, strides_);
}


/* -------------------------- Expressions -------------------------- */

/** Elementwise Scalar-Tensor operation. Returns a Tensor with shape 
 *  `tensor`, where each element of the new Tensor is the result of `fn` 
 *  applied with the corresponding `tensor` elements and `scalar`.
 */
template <typename X, typename Y, size_t M, typename CType_, typename FunctionType>
Tensor<X, M, CType_> elem_wise(Tensor<X, M, CType_> const &tensor, Y const &scalar,
      FunctionType &&fn)
{
  Tensor<X, M, CType_> result {tensor.shape()};
  std::function<void(X*, X*)> set_vals = [&scalar, &fn](X *lhs, X *rhs) -> void {
    *lhs = fn(*rhs, scalar);
  };
  result.pUnaryMap(tensor, set_vals);
  return result;
}

template <typename X, typename Y, size_t M, typename CType_, typename FunctionType>
Tensor<X, M, CType_> elem_wise(Tensor<X, M, CType_> const &tensor1, Tensor<Y, M, CType_> const &tensor2,
      FunctionType &&fn)
{
  Tensor<X, M, CType_> result {tensor1.shape()};
  std::function<void(X*, X*, Y*)> set_vals = [&fn](X *lhs, X *rhs1, Y *rhs2) -> void {
    *lhs = fn(*rhs1, *rhs2);
  };
  result.pBinaryMap(tensor1, tensor2, set_vals);
  return result;
}

/** Creates a Tensor whose elements are the elementwise sum of `tensor1` 
 *  and `tensor2`. `tensor1` and `tensor2` must have equivalent shape, or
 *  a std::logic_error is thrown. 
 */
template <typename X, typename Y, size_t M, typename C1, typename C2>
Tensor<X, M, C1> add(
    Tensor<X, M, C1> const& tensor_1, 
    Tensor<Y, M, C2> const& tensor_2)
{
  assert((tensor_1.shape_ == tensor_2.shape_)  && DIMENSION_MISMATCH);
  Tensor<X, M, C1> sum_tensor(tensor_1.shape_);
  std::function<void(X *, X*, Y*)> add = [](X *x, X *y, Y *z) -> void
  {
    *x = *y + *z;
  };
  sum_tensor.pBinaryMap(tensor_1, tensor_2, add);
  return sum_tensor;
}

template <typename T, size_t N, typename ContainerType>
template <typename RHS>
Tensor<T, N, ContainerType> &Tensor<T, N, ContainerType>::operator+=(Expression<RHS> const &rhs)
{
  auto tensor = rhs.self()();
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
  *this = *this + tensor;
  return *this;
}

/** Creates a Tensor whose elements are the elementwise difference of `tensor1` 
 *  and `tensor2`. `tensor1` and `tensor2` must have equivalent shape, or
 *  a std::logic_error is thrown. 
 */
template <typename X, typename Y, size_t M, typename CType_>
Tensor<X, M, CType_> subtract(Tensor<X, M, CType_> const& tensor_1, Tensor<Y, M, CType_> const& tensor_2)
{
  assert((tensor_1.shape_ == tensor_2.shape_) && DIMENSION_MISMATCH);
  Tensor<X, M, CType_> diff_tensor(tensor_1.shape_);
  std::function<void(X *, X*, Y*)> sub = [](X *x, X *y, Y *z) -> void
  {
    *x = *y - *z;
  };
  diff_tensor.pBinaryMap(tensor_1, tensor_2, sub);
  return diff_tensor;
}

template <typename T, size_t N, typename ContainerType>
template <typename RHS>
Tensor<T, N, ContainerType> &Tensor<T, N, ContainerType>::operator-=(Expression<RHS> const &rhs)
{
  auto tensor = rhs.self()();
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
  *this = *this - tensor;
  return *this;
}

/** Produces a Tensor which is the Tensor product of `tensor_1` and 
 *  `tensor_2`. Tensor multiplication is equivalent to matrix multiplication
 *  scaled to higher dimensions, i.e. shapes 2x3x4 * 4x3x2 -> 2x3x3x2
 *  The inner dimensions of `tensor_1` and `tensor_2` must match, or 
 *  std::logic error is thrown. Note: VERY EXPENSIVE, the time complexity
 *  to produce a N rank Tensor with all dimensions m is O(m^(N+1)).
 */
template <typename X, typename Y, size_t M1, size_t M2, typename CType_>
Tensor<X, M1 + M2 - 2, CType_> multiply(Tensor<X, M1, CType_> const& tensor_1, Tensor<Y, M2, CType_> const& tensor_2)
{
  static_assert(M1, SCALAR_TENSOR_MULT);
  static_assert(M2, SCALAR_TENSOR_MULT);
  assert((tensor_1.shape_.dimensions_[0] == tensor_2.shape_.dimensions_[M2 - 1]) 
      && INNER_DIMENSION_MISMATCH);

  auto shape = Shape<M1 + M2 - 2>();
  std::copy_n(tensor_1.shape_.dimensions_, M1 - 1, shape.dimensions_);
  std::copy_n(tensor_2.shape_.dimensions_ + 1, M2 - 1, shape.dimensions_ + M1 - 1);
  Tensor<X, M1 + M2 - 2, CType_> prod_tensor(shape);
  size_t cumul_index_1 = tensor_1.shape_.index_product() / tensor_1.shape_.dimensions_[M1 - 1];
  size_t cumul_index_2 = tensor_2.shape_.index_product() / tensor_2.shape_.dimensions_[0];
  size_t indices_1[M1 - 1], indices_2[M2 - 1];
  std::fill_n(indices_1, M1 - 1, 0); 
  std::fill_n(indices_2, M2 - 1, 0); 
  size_t index = 0;
  typename Tensor<X, M1, CType_>::IndexReference t1_index{tensor_1};
  for (size_t i1 = 0; i1 < cumul_index_1; ++i1) {
    typename Tensor<Y, M2, CType_>::IndexReference t2_index{tensor_2};
    for (size_t i2 = 0; i2 < cumul_index_2; ++i2) {
      X value {};
      for (size_t x = 0; x < tensor_1.shape_.dimensions_[M1 - 1]; ++x)
          value += (*tensor_1.ref_)[tensor_1.offset_ + t1_index.index + tensor_1.strides_[M1 - 1] * x] *
            (*tensor_2.ref_)[tensor_2.offset_ + t2_index.index + tensor_2.strides_[0] * x];
      (*prod_tensor.ref_)[index] = value;
      Tensor<Y, M2, CType_>::pUpdateIndices(indices_2, t2_index, 1);
      ++index;
    }
    Tensor<X, M1, CType_>::pUpdateIndices(indices_1, t1_index);
  }
  return prod_tensor;
}

template <typename T, size_t N, typename ContainerType>
template <typename RHS>
Tensor<T, N, ContainerType> &Tensor<T, N, ContainerType>::operator*=(Expression<RHS> const &rhs)
{
  auto tensor = rhs.self()();
  *this = *this * tensor;
  return *this;
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> Tensor<T, N, ContainerType>::operator-() const
{
  Tensor<T, N, ContainerType> neg_tensor(shape_);
  std::function<void(T *, T *)> neg = [](T *x, T *y) -> void
  {
    *x = -(*y);
  };
  neg_tensor.pUnaryMap(*this, neg);
  return neg_tensor;
}

/* --------------------------- Useful Functions ------------------------- */

template <typename T, size_t N, typename ContainerType>
template <size_t M>
Tensor<T, M, ContainerType> Tensor<T, N, ContainerType>::resize(Shape<M> const &shape) const
{
  size_t num_elems = shape_.index_product();
  assert(num_elems == shape.index_product() && ELEMENT_COUNT_MISMATCH);

  // some compilers cannot capture stack allocated arrays by reference
  T cbuff[num_elems];
  T *buff = cbuff; 
  size_t index = 0;
  std::function<void(T const &lhs)> fill_buff = [&buff, &index](T const &lhs) ->void {
    buff[index++] = lhs;
  };
  pMap(fill_buff);
  auto tensor = Tensor<T, M, ContainerType>(shape.dimensions_, 0, 
      std::make_shared<ContainerType>(num_elems, (T *)buff, buff + num_elems));
  return tensor;
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> Tensor<T, N, ContainerType>::copy() const
{
  return Tensor<T, N, ContainerType>(*this);
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::Proxy Tensor<T, N, ContainerType>::ref() 
{
  return Proxy(*this);
}

/** Fills the elements of `tensor` with the elements between.
 *  `begin` and `end`, which must be random access iterators. The number 
 *  elements between `begin` and `end` must be equivalent to the capacity of
 *  `tensor`, otherwise std::logic_error is thrown.
 */
template <typename U, size_t M, typename CType_, typename RAIt>
void Fill(Tensor<U, M, CType_> &tensor, RAIt const &begin, RAIt const &end)
{
  size_t cumul_sum = tensor.shape_.index_product();
  auto dist_sum = std::distance(begin, end);
  assert((dist_sum > 0 && cumul_sum == (size_t)dist_sum) && NELEMENTS);
  RAIt it = begin;
  std::function<void(U *)> allocate = [&it](U *x) -> void
  {
    *x = *it;
    ++it;
  };
  tensor.pMap(allocate);
}

/** Assigns to each element in Tensor the value `value`
 */
template <typename U, size_t M, typename CType_, typename X>
void Fill(Tensor<U, M, CType_> &tensor, X const &value)
{
  std::function<void(U *)> allocate = [&value](U *x) -> void
  {
    *x = value;
  };
  tensor.pMap(allocate);
}

/** Returns a transposed Matrix, sharing the same underlying data as `mat`. If
 *  the dimension of `mat` is [n, m], the dimensions of the resulting matrix will be
 *  [m, n]. Note: This is only applicable to matrices and vectors
 */
template <typename U, typename CType_> 
Tensor<U, 2, CType_> transpose(Tensor<U, 2, CType_> &mat)
{
  size_t transposed_dimensions[2];
  transposed_dimensions[0] = mat.shape_.dimensions_[1];
  transposed_dimensions[1] = mat.shape_.dimensions_[0];
  size_t transposed_strides[2];
  transposed_strides[0] = mat.strides_[1];
  transposed_strides[1] = mat.strides_[0];
  return Tensor<U, 2, CType_>(transposed_dimensions, transposed_strides,
      mat.offset_, std::shared_ptr<CType_>(mat.ref_));
}

/** See Tensor<T, N, ContainerType> transpose(Tensor<T, N, ContainerType> &)
 */
template <typename U, typename CType_>
Tensor<U, 2, CType_> const transpose(Tensor<U, 2, CType_> const &mat) 
{
  return (*const_cast<typename std::decay<decltype(mat)>::type*>(mat)).tranpose();
}

/** Returns a transposed Matrix, sharing the same underlying data as vec. If
 *  the dimension of `vec` is [n], the dimensions of the resulting matrix will be
 *  [1, n]. Note: This is only applicable to matrices and vectors
 */
template <typename T, typename ContainerType>
Tensor<T, 2, ContainerType> transpose(Tensor<T, 1, ContainerType> &vec) 
{
  size_t transposed_dimensions[2];
  transposed_dimensions[0] = 1;
  transposed_dimensions[1] = vec.shape_.dimensions_[0];
  size_t transposed_strides[2];
  transposed_strides[0] = 1;
  transposed_strides[1] = vec.strides_[0];
  return Tensor<T, 2, ContainerType>(transposed_dimensions, transposed_strides,
      vec.offset_, std::shared_ptr<ContainerType>(vec.ref_));
}

/* ------------------------------- Iterator ----------------------------- */

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Iterator::Iterator(Tensor<T, N + 1, ContainerType> const &tensor, size_t index)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Iterator::Iterator(Iterator const &it)
  : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::Iterator::Iterator(Iterator &&it)
  : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> Tensor<T, N, ContainerType>::Iterator::operator*()
{
  return Tensor<T, N, ContainerType>(shape_.dimensions_, strides_, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> Tensor<T, N, ContainerType>::Iterator::operator->()
{
  return Tensor<T, N, ContainerType>(shape_.dimensions_, strides_, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::Iterator Tensor<T, N, ContainerType>::Iterator::operator++(int)
{
  Tensor<T, N, ContainerType>::Iterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::Iterator &Tensor<T, N, ContainerType>::Iterator::operator++()
{
  offset_ += stride_;
  return *this;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::Iterator Tensor<T, N, ContainerType>::Iterator::operator--(int)
{
  Tensor<T, N, ContainerType>::Iterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::Iterator &Tensor<T, N, ContainerType>::Iterator::operator--()
{
  offset_ -= stride_;
  return *this;
} 

template <typename T, size_t N, typename ContainerType>
bool Tensor<T, N, ContainerType>::Iterator::operator==(Tensor<T, N, ContainerType>::Iterator const &it) const
{
  if (shape_ != it.shape_) return false;
  if (stride_ != it.stride_) return false;
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ----------------------------- ConstIterator --------------------------- */

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ConstIterator::ConstIterator(Tensor<T, N + 1, ContainerType> const &tensor, size_t index)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ConstIterator::ConstIterator(ConstIterator const &it)
  : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ConstIterator::ConstIterator(ConstIterator &&it)
  : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> const Tensor<T, N, ContainerType>::ConstIterator::operator*()
{
  return Tensor<T, N, ContainerType>(shape_.dimensions_, strides_, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> const Tensor<T, N, ContainerType>::ConstIterator::operator->()
{
  return Tensor<T, N, ContainerType>(shape_.dimensions_, strides_, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ConstIterator Tensor<T, N, ContainerType>::ConstIterator::operator++(int)
{
  Tensor<T, N, ContainerType>::ConstIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ConstIterator &Tensor<T, N, ContainerType>::ConstIterator::operator++()
{
  offset_ += stride_;
  return *this;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ConstIterator Tensor<T, N, ContainerType>::ConstIterator::operator--(int)
{
  Tensor<T, N, ContainerType>::ConstIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ConstIterator &Tensor<T, N, ContainerType>::ConstIterator::operator--()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, size_t N, typename ContainerType>
bool Tensor<T, N, ContainerType>::ConstIterator::operator==(Tensor<T, N, ContainerType>::ConstIterator const &it) const
{
  if (shape_ != it.shape_) return false;
  if (stride_ != it.stride_) return false;
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ---------------------------- ReverseIterator -------------------------- */

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ReverseIterator::ReverseIterator(Tensor<T, N + 1, ContainerType> const &tensor, size_t index)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
  offset_ += stride_ * (tensor.shape_.dimensions_[index] - 1);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ReverseIterator::ReverseIterator(ReverseIterator const &it)
  : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ReverseIterator::ReverseIterator(ReverseIterator &&it)
  : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> Tensor<T, N, ContainerType>::ReverseIterator::operator*()
{
  return Tensor<T, N, ContainerType>(shape_.dimensions_, strides_, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> Tensor<T, N, ContainerType>::ReverseIterator::operator->()
{
  return Tensor<T, N, ContainerType>(shape_.dimensions_, strides_, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ReverseIterator Tensor<T, N, ContainerType>::ReverseIterator::operator++(int)
{
  Tensor<T, N, ContainerType>::ReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ReverseIterator &Tensor<T, N, ContainerType>::ReverseIterator::operator++()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ReverseIterator Tensor<T, N, ContainerType>::ReverseIterator::operator--(int)
{
  Tensor<T, N, ContainerType>::ReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ReverseIterator &Tensor<T, N, ContainerType>::ReverseIterator::operator--()
{
  offset_ += stride_;
  return *this;
}

template <typename T, size_t N, typename ContainerType>
bool Tensor<T, N, ContainerType>::ReverseIterator::operator==(Tensor<T, N, ContainerType>::ReverseIterator const &it) const
{
  if (shape_ != it.shape_) return false;
  if (stride_ != it.stride_) return false;
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ------------------------- ConstReverseIterator ----------------------- */

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ConstReverseIterator::ConstReverseIterator(Tensor<T, N + 1, ContainerType> const &tensor, size_t index)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
  offset_ += stride_ * (tensor.shape_.dimensions_[index] - 1);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator const &it)
  : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator &&it)
  : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> const Tensor<T, N, ContainerType>::ConstReverseIterator::operator*()
{
  return Tensor<T, N, ContainerType>(shape_.dimensions_, strides_, offset_, 
      std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
Tensor<T, N, ContainerType> const Tensor<T, N, ContainerType>::ConstReverseIterator::operator->()
{
  return Tensor<T, N, ContainerType>(shape_.dimensions_, strides_, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ConstReverseIterator Tensor<T, N, ContainerType>::ConstReverseIterator::operator++(int)
{
  Tensor<T, N, ContainerType>::ConstReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ConstReverseIterator &Tensor<T, N, ContainerType>::ConstReverseIterator::operator++()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ConstReverseIterator Tensor<T, N, ContainerType>::ConstReverseIterator::operator--(int)
{
  Tensor<T, N, ContainerType>::ConstReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N, ContainerType>::ConstReverseIterator &Tensor<T, N, ContainerType>::ConstReverseIterator::operator--()
{
  offset_ += stride_;
  return *this;
}

template <typename T, size_t N, typename ContainerType>
bool Tensor<T, N, ContainerType>::ConstReverseIterator::operator==(
    Tensor<T, N, ContainerType>::ConstReverseIterator const &it) const
{
  if (shape_ != it.shape_) return false;
  if (stride_ != it.stride_) return false;
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ----------------------- Iterator Construction ----------------------- */

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::Iterator Tensor<T, N, ContainerType>::begin(size_t index)
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  return typename Tensor<T, N - 1, ContainerType>::Iterator(*this, index);
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::Iterator Tensor<T, N, ContainerType>::end(size_t index)
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  typename Tensor<T, N - 1, ContainerType>::Iterator it{*this, index};
  it.offset_ += strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::Iterator Tensor<T, N, ContainerType>::begin() 
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->begin(0); 
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::Iterator Tensor<T, N, ContainerType>::end() 
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->end(0); 
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ConstIterator Tensor<T, N, ContainerType>::cbegin(size_t index) const
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  return typename Tensor<T, N - 1, ContainerType>::ConstIterator(*this, index);
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ConstIterator Tensor<T, N, ContainerType>::cend(size_t index) const
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  typename Tensor<T, N - 1, ContainerType>::ConstIterator it{*this, index};
  it.offset_ += strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ConstIterator Tensor<T, N, ContainerType>::cbegin() const
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->cbegin(0); 
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ConstIterator Tensor<T, N, ContainerType>::cend() const
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->cend(0); 
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ReverseIterator Tensor<T, N, ContainerType>::rbegin(size_t index)
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  return typename Tensor<T, N - 1, ContainerType>::ReverseIterator(*this, index);
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ReverseIterator Tensor<T, N, ContainerType>::rend(size_t index)
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  typename Tensor<T, N - 1, ContainerType>::ReverseIterator it{*this, index};
  it.offset_ -= strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ReverseIterator Tensor<T, N, ContainerType>::rbegin() 
{
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->rbegin(0); 
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ReverseIterator Tensor<T, N, ContainerType>::rend() 
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->rend(0); 
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator Tensor<T, N, ContainerType>::crbegin(size_t index) const
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  return typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator(*this, index);
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator Tensor<T, N, ContainerType>::crend(size_t index) const
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator it{*this, index};
  it.offset_ -= strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator Tensor<T, N, ContainerType>::crbegin() const
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->crbegin(0); 
}

template <typename T, size_t N, typename ContainerType>
typename Tensor<T, N - 1, ContainerType>::ConstReverseIterator Tensor<T, N, ContainerType>::crend() const
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->crend(0); 
}

/* ------------------------ Scalar Specializations ---------------------- */

template <>
class Shape<0> { /*@Shape<0>*/
public:
  /** Scalar specialization of Shape. This is an empty structure 
   *  (sizeof(Shape<0>) will return 1 byte). This is specialized
   *  solely to provide convience for scalar specializing Tensor
   */

  /* -------------------- typedefs -------------------- */
  typedef size_t                    size_type;
  typedef ptrdiff_t                 difference_type;
  typedef Shape<0>                  self_type;

  /* ----------------- friend classes ----------------- */

  template <typename X, size_t M, typename ContainerType> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ------------------ Constructors ------------------ */

  explicit Shape() {}
  Shape(Shape<0> const&) {}

  /* -------------------- Getters --------------------- */

  constexpr static size_t rank() { return 0; }

  /* -------------------- Equality -------------------- */

  /** This will always return true */
  bool operator==(Shape<0> const&) const noexcept { return true; }

  /** This will always return false */
  bool operator!=(Shape<0> const&) const noexcept { return false; }

  /** This will always return false */
  template <size_t M>
  bool operator==(Shape<M> const&) const noexcept { return false; }
  
  /** This will always return true */
  template <size_t M>
  bool operator!=(Shape<M> const&) const noexcept { return true; }

  /* ---------------- Print ----------------- */

  template <typename X, size_t M, typename CType_>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M, CType_> &tensor);

private:
};

// Scalar specialization
template <typename T, typename ContainerType>
class Tensor<T, 0, ContainerType>: public Expression<Tensor<T, 0, ContainerType>> { /*@Tensor<T, 0, ContainerType>*/
public:
  /** Scalar specialization of Tensor object. The major motivation is
   *  the ability to implicitly convert to the underlying data type,
   *  allowing the Scalar to effectively be used as an ordinary value.
   */
  typedef T                                value_type;
  typedef ContainerType                    container_type;
  typedef T&                               reference_type;
  typedef T const&                         const_reference_type;
  typedef Tensor<T, 0, ContainerType>      self_type;

  /* ------------- Friend Classes ------------- */

  template <typename X, size_t M, typename CType_> friend class Tensor;

  /* ----------- Proxy Objects ------------ */

  class Proxy { /*@Proxy<T,0>*/
  /**
   * Proxy Tensor Object used for building tensors from reference
   * This is used only to differentiate proxy tensor Construction
   */
  public:
    template <typename U, size_t N, typename CType_> friend class Tensor;
    Proxy() = delete;
  private:
    Proxy(Tensor<T, 0, ContainerType> const &tensor): tensor_(tensor) {}
    Proxy(Proxy const &proxy): tensor_(proxy.tensor_) {} 
    Tensor<T, 0, ContainerType> const &tensor_;
  }; 

  /* -------------- Constructors -------------- */

  Tensor(); /**< Constructs a new Scalar whose value is zero-initialized */
  /**< Constructs a new Scalar whose value is forwarded `val` */
  explicit Tensor(value_type &&val); 
  explicit Tensor(Shape<0>); /**< Constructs a new Scalar whose value is zero-initialized */
  /**< Constructs a new Scalar and whose value copies `tensor`'s */
  Tensor(Tensor<T, 0, ContainerType> const &tensor); 
  /**< Moves data from `tensor`. `tensor` is destroyed. */
  Tensor(Tensor<T, 0, ContainerType> &&tensor);      
  /**< Constructs a Scalar who shares underlying data with proxy's underyling Scalar. */
  Tensor(typename Tensor<T, 0, ContainerType>::Proxy const &proxy); 
  /**< Evaluates `expression` and move constructs from the resulting scalar */
  template <typename NodeType,
            typename = typename std::enable_if<NodeType::rank() == 0>::type>
  Tensor(Expression<NodeType> const& expression);

  /* ------------- Assignment ------------- */

  /** Copy Constructs from `tensor`. Destroys itself first */
  Tensor<T, 0, ContainerType> &operator=(Tensor<T, 0, ContainerType> const &tensor);

  /** Evaluates `rhs` and move constructs. Destroys itself first */
  template <typename X> Tensor<T, 0, ContainerType> &operator=(Tensor<X, 0, ContainerType> const &tensor);

  /* -------------- Getters -------------- */

  constexpr static size_t rank() { return 0; } /**< Returns 0 */
  Shape<0> shape() const noexcept { return shape_; } /**< Returns a scalar shape */
  value_type &operator()() { return (*ref_)[offset_]; } /**< Returns the data as a reference */
  /**< Returns the data as a const reference */
  value_type const &operator()() const { return (*ref_)[offset_]; }
  /**< Used to implement iterator->, should not be used explicitly */
  Tensor *operator->() { return this; } 
  /**< Used to implement const_iterator->, should not be used explicitly */
  Tensor const *operator->() const { return this; }

  /* -------------- Setters -------------- */

  template <typename X,
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type>
  Tensor<T, 0, ContainerType> &operator=(X&& elem); /**< Assigns `elem` to the underlying data */

  /* --------------- Print --------------- */
 
  template <typename X, size_t M, typename CType_>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M, CType_> &tensor);

  template <typename X, typename CType_>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, 0, CType_> &tensor);

  /* ------------ Equivalence ------------ */

  /** Equivalence between underlying data */
  bool operator==(Tensor<T, 0, ContainerType> const& tensor) const; 

  /** Non-equivalence between underlying data */
  bool operator!=(Tensor<T, 0, ContainerType> const& tensor) const { return !(*this == tensor); }

  /** Equivalence between underlying data */
  template <typename X>
  bool operator==(Tensor<X, 0, ContainerType> const& tensor) const;

  /** Non-equivalence between underlying data */
  template <typename X>
  bool operator!=(Tensor<X, 0, ContainerType> const& tensor) const { return !(*this == tensor); }

  /** Equivalence Scalar with an element of a type default-convertible 
   *  to the underlying data type.
   */
  template <typename X,
            typename = typename std::enable_if
            <std::is_convertible
            <typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type
  > bool operator==(X val) const;

  /** Non-equivalence Scalar with an element of a type default-convertible 
   *  to the underlying data type.
   */
  template <typename X,
            typename = typename std::enable_if
            <std::is_convertible
            <typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type
  > bool operator!=(X val) const { return !(*this == val); }

  /* ------------ Expressions ------------- */

  template <typename X, typename Y, typename C1, typename C2>
  friend Tensor<X, 0, C1> add(
      Tensor<X, 0, C1> const &tensor_1, 
      Tensor<Y, 0, C2> const &tensor_2);

  template <typename RHS>
  Tensor<T, 0, ContainerType> &operator+=(Expression<RHS> const &rhs);

  Tensor<T, 0, ContainerType> &operator+=(T const &scalar);

  template <typename X, typename Y>
  friend Tensor<X, 0, ContainerType> subtract(Tensor<X, 0, ContainerType> const &tensor_1, Tensor<Y, 0, ContainerType> const &tensor_2);

  template <typename RHS>
  Tensor<T, 0, ContainerType> &operator-=(Expression<RHS> const &rhs);

  Tensor<T, 0, ContainerType> &operator-=(T const &scalar);

  template <typename X, typename Y>
  friend Tensor<X, 0, ContainerType> multiply(Tensor<X, 0, ContainerType> const &tensor_1, Tensor<Y, 0, ContainerType> const &tensor_2);

  template <typename RHS>
  Tensor<T, 0, ContainerType> &operator*=(Expression<RHS> const &rhs);

  Tensor<T, 0, ContainerType> &operator*=(T const &scalar);

  template <typename RHS>
  Tensor<T, 0, ContainerType> &operator/=(Expression<RHS> const &rhs);

  Tensor<T, 0, ContainerType> &operator/=(T const &scalar);

  Tensor<T, 0, ContainerType> operator-() const;

  /* ---------------- Iterator -------------- */

  class Iterator { /*@Iterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, typename CType_> friend class Tensor;

    /* --------------- Constructors --------------- */

    Iterator(Iterator const &it);
    Iterator(Iterator &&it);
    Tensor<T, 0, ContainerType> operator*();
    Tensor<T, 0, ContainerType> const operator*() const;
    Tensor<T, 0, ContainerType> operator->();
    Tensor<T, 0, ContainerType> const operator->() const;
    Iterator operator++(int);
    Iterator &operator++();
    Iterator operator--(int);
    Iterator &operator--();
    bool operator==(Iterator const &it) const;
    bool operator!=(Iterator const &it) const { return !(it == *this); }
  private:
    Iterator(Tensor<T, 1, ContainerType> const &tensor, size_t);

    size_t offset_;
    std::shared_ptr<ContainerType> ref_;
    size_t stride_;
  };

  /* -------------- ConstIterator ------------ */

  class ConstIterator { /*@ConstIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, typename CType_> friend class Tensor;

    /* --------------- Constructors --------------- */

    ConstIterator(ConstIterator const &it);
    ConstIterator(ConstIterator &&it);
    Tensor<T, 0, ContainerType> const operator*();
    Tensor<T, 0, ContainerType> const operator->();
    ConstIterator operator++(int);
    ConstIterator &operator++();
    ConstIterator operator--(int);
    ConstIterator &operator--();
    bool operator==(ConstIterator const &it) const;
    bool operator!=(ConstIterator const &it) const { return !(it == *this); }
  private:
    ConstIterator(Tensor<T, 1, ContainerType> const &tensor, size_t);

    size_t offset_;
    std::shared_ptr<ContainerType> ref_;
    size_t stride_;
  };

  /* ------------- ReverseIterator ----------- */

  class ReverseIterator { /*@ReverseIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, typename CType_> friend class Tensor;

    /* --------------- Constructors --------------- */

    ReverseIterator(ReverseIterator const &it);
    ReverseIterator(ReverseIterator &&it);
    Tensor<T, 0, ContainerType> operator*();
    Tensor<T, 0, ContainerType> operator->();
    ReverseIterator operator++(int);
    ReverseIterator &operator++();
    ReverseIterator operator--(int);
    ReverseIterator &operator--();
    bool operator==(ReverseIterator const &it) const;
    bool operator!=(ReverseIterator const &it) const { return !(it == *this); }
  private:
    ReverseIterator(Tensor<T, 1, ContainerType> const &tensor, size_t);

    size_t offset_;
    std::shared_ptr<ContainerType> ref_;
    size_t stride_;
  };

  /* ----------- ConstReverseIterator --------- */

  class ConstReverseIterator { /*@ReverseIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, typename CType_> friend class Tensor;

    /* --------------- Constructors --------------- */

    ConstReverseIterator(ConstReverseIterator const &it);
    ConstReverseIterator(ConstReverseIterator &&it);
    Tensor<T, 0, ContainerType> const operator*();
    Tensor<T, 0, ContainerType> const operator->();
    ConstReverseIterator operator++(int);
    ConstReverseIterator &operator++();
    ConstReverseIterator operator--(int);
    ConstReverseIterator &operator--();
    bool operator==(ConstReverseIterator const &it) const;
    bool operator!=(ConstReverseIterator const &it) const { return !(it == *this); }
  private:
    ConstReverseIterator(Tensor<T, 1, ContainerType> const &tensor, size_t);

    size_t offset_;
    std::shared_ptr<ContainerType> ref_;
    size_t stride_;
  };

  /* ------------- Utility Functions ------------ */

  /** Returns an identical Tensor<T, 0, ContainerType> (copy constructed) of `*this` */
  Tensor<T, 0, ContainerType> copy() const;

  /** Returns a proxy object of `this`, used only for Tensor<T, 0, ContainerType>::Tensor(Tensor<T, 0, ContainerType>::Proxy const&) */
  typename Tensor<T, 0, ContainerType>::Proxy ref();

private:

  /* ------------------- Data ------------------- */

  Shape<0> shape_;
  size_t offset_;
  std::shared_ptr<ContainerType> ref_;

  /* ------------------ Utility ----------------- */

  Tensor(size_t const *, size_t const *, size_t, std::shared_ptr<ContainerType> &&ref);

};

/* ------------------- Constructors ----------------- */

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Tensor() : shape_(Shape<0>()), offset_(0), ref_(std::make_shared<ContainerType>(1))
{}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Tensor(Shape<0>) : shape_(Shape<0>()), offset_(0), ref_(std::make_shared<ContainerType>(1))
{}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Tensor(T &&val) : shape_(Shape<0>()), offset_(0)
{
  ref_ = std::make_shared<ContainerType>(1, val);
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Tensor(Tensor<T, 0, ContainerType> const &tensor): shape_(Shape<0>()), offset_(0),
  ref_(std::make_shared<ContainerType>(1), (*tensor.ref_)[0])
{}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Tensor(Tensor<T, 0, ContainerType> &&tensor): shape_(Shape<0>()), offset_(tensor.offset_),
  ref_(std::make_shared<ContainerType>(1))
{}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Tensor(typename Tensor<T, 0, ContainerType>::Proxy const &proxy)
  : shape_(Shape<0>()), offset_(proxy.tensor_.offset_), ref_(proxy.tensor_.ref_)
{}

template <typename T, typename ContainerType>
template <typename NodeType, typename>
Tensor<T, 0, ContainerType>::Tensor(Expression<NodeType> const& expression)
  : offset_(0)
{
  T result = expression.self()();
  ref_ = std::make_shared<ContainerType>(1, result);
}

/* ---------------------- Assignment ---------------------- */

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator=(Tensor<T, 0, ContainerType> const &tensor)
{
  (*ref_)[offset_] = (*tensor.ref_)[tensor.offset_];
  return *this;
}

template <typename T, typename ContainerType>
template <typename X>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator=(Tensor<X, 0, ContainerType> const &tensor)
{
  (*ref_)[offset_] = (*tensor.ref_)[tensor.offset_];
  return *this;
}

template <typename T, typename ContainerType>
template <typename X, typename>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator=(X&& elem)
{
  (*ref_)[offset_] = std::forward<X>(elem);
  return *this;
}

/* ------------------------ Equivalence ----------------------- */

template <typename T, typename ContainerType>
bool Tensor<T, 0, ContainerType>::operator==(Tensor<T, 0, ContainerType> const &tensor) const
{
  return (*ref_)[offset_] == (*tensor.ref_)[tensor.offset_];
}

template <typename T, typename ContainerType>
template <typename X>
bool Tensor<T, 0, ContainerType>::operator==(Tensor<X, 0, ContainerType> const &tensor) const
{
  return (*ref_)[offset_] == (*tensor.ref_)[tensor.offset_];
}

template <typename T, typename ContainerType>
template <typename X, typename>
bool Tensor<T, 0, ContainerType>::operator==(X val) const
{
  return (*ref_)[offset_] == val;
}

/* ----------------------- Expressions ----------------------- */

template <typename X, typename Y, typename C1, typename C2>
Tensor<X, 0, C1> add(
    Tensor<X, 0, C1> const &tensor_1, 
    Tensor<Y, 0, C2> const &tensor_2)
{
  return Tensor<X, 0, C1>(tensor_1() + tensor_2());
}

template <typename X, typename Y,
         typename = typename std::enable_if
         <LogicalAnd<!IsTensor<X>::value, !IsTensor<Y>::value>::value>>
inline X add(X const& x, Y const & y) { return x + y; }

template <typename X, typename CType_>
Tensor<X, 0, CType_> operator+(Tensor<X, 0, CType_> const &tensor, X const &scalar) 
{
  return Tensor<X, 0, CType_>(tensor() + scalar);
}

template <typename X, typename CType_>
Tensor<X, 0, CType_> operator+(X const &scalar, Tensor<X, 0, CType_> const &tensor) 
{
  return Tensor<X, 0, CType_>(tensor() + scalar);
}

template <typename T, typename ContainerType>
template <typename RHS>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator+=(Expression<RHS> const &rhs)
{
  auto scalar = rhs.self()();
  static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
  *this = *this + scalar;
  return *this;
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator+=(T const &scalar)
{
  (*ref_)[offset_] += scalar;
  return *this;
}

template <typename X, typename Y, typename CType_>
Tensor<X, 0, CType_> subtract(Tensor<X, 0, CType_> const &tensor_1, Tensor<Y, 0, CType_> const &tensor_2)
{
  return Tensor<X, 0, CType_>(tensor_1() - tensor_2());
}

template <typename X, typename Y,
         typename = typename std::enable_if<
          LogicalAnd<!IsTensor<X>::value, !IsTensor<Y>::value>::value>>
inline X subtract(X const& x, Y const & y) { return x - y; }

template <typename X, typename CType_>
Tensor<X, 0, CType_> operator-(Tensor<X, 0, CType_> const &tensor, X const &scalar) 
{
  return Tensor<X, 0, CType_>(tensor() - scalar);
}

template <typename X, typename CType_>
Tensor<X, 0, CType_> operator-(X const &scalar, Tensor<X, 0, CType_> const &tensor) 
{
  return Tensor<X, 0, CType_>(scalar - tensor());
}

template <typename T, typename ContainerType>
template <typename RHS>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator-=(Expression<RHS> const &rhs)
{
  auto scalar = rhs.self()();
  static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
  *this = *this - scalar;
  return *this;
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator-=(T const &scalar)
{
  (*ref_)[offset_] -= scalar;
  return *this;
}

// Directly overload operator*
template <typename X, typename Y, typename CType_>
inline Tensor<X, 0, CType_> operator*(Tensor<X, 0, CType_> const &tensor_1, Tensor<Y, 0, CType_> const &tensor_2)
{
  return Tensor<X, 0, CType_>(tensor_1() * tensor_2());
}

template <typename X, typename CType_>
inline Tensor<X, 0, CType_> operator*(Tensor<X, 0, CType_> const &tensor, X const &scalar) 
{
  return Tensor<X, 0, CType_>(tensor() * scalar);
}

template <typename X, typename CType_>
inline Tensor<X, 0, CType_> operator*(X const &scalar, Tensor<X, 0, CType_> const &tensor) 
{
  return Tensor<X, 0, CType_>(tensor() * scalar);
}

template <typename T, typename ContainerType>
template <typename RHS>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator*=(Expression<RHS> const &rhs)
{
  auto scalar = rhs.self()();
  static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
  *this = *this * scalar;
  return *this;
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator*=(T const &scalar)
{
  (*ref_)[offset_] *= scalar;
  return *this;
}

// Directly overload operator/
template <typename X, typename Y, typename CType_>
inline Tensor<X, 0, CType_> operator/(Tensor<X, 0, CType_> const &tensor_1, Tensor<Y, 0, CType_> const &tensor_2)
{
  return Tensor<X, 0, CType_>(tensor_1() / tensor_2());
}

template <typename X, typename CType_>
Tensor<X, 0, CType_> operator/(Tensor<X, 0, CType_> const &tensor, X const &scalar) 
{
  return Tensor<X, 0, CType_>(tensor() / scalar);
}

template <typename X, typename CType_>
Tensor<X, 0, CType_> operator/(X const &scalar, Tensor<X, 0, CType_> const &tensor) 
{
  return Tensor<X, 0, CType_>(tensor() / scalar);
}


template <typename T, typename ContainerType>
template <typename RHS>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator/=(Expression<RHS> const &rhs)
{
  auto scalar = rhs.self()();
  static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
  *this = *this / scalar;
  return *this;
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> &Tensor<T, 0, ContainerType>::operator/=(T const &scalar)
{
  (*ref_)[offset_] /= scalar;
  return *this;
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> Tensor<T, 0, ContainerType>::operator-() const
{
  return Tensor<T, 0, ContainerType>(-(*ref_)[offset_]);
}

/* ------------------------ Utility --------------------------- */

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Tensor(size_t const *, size_t const *, size_t offset, std::shared_ptr<ContainerType> &&ref)
  : offset_(offset), ref_(std::move(ref))
{}

/* ------------------------ Overloads ------------------------ */

template <typename X, typename CType_>
std::ostream &operator<<(std::ostream &os, const Tensor<X, 0, CType_> &tensor)
{
  os << (*tensor.ref_)[tensor.offset_];
  return os;
}

/* ------------------- Useful Functions ---------------------- */

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> Tensor<T, 0, ContainerType>::copy() const
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, 0, 
      std::make_shared<ContainerType>(1, (*ref_)[offset_]));
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::Proxy Tensor<T, 0, ContainerType>::ref() 
{
  return Proxy(*this);
}

/* ---------------------- Iterators ------------------------- */

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Iterator::Iterator(Tensor<T, 1, ContainerType> const &tensor, size_t)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Iterator::Iterator(Iterator const &it)
  : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::Iterator::Iterator(Iterator &&it)
  : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> Tensor<T, 0, ContainerType>::Iterator::operator*()
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> Tensor<T, 0, ContainerType>::Iterator::operator->()
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::Iterator Tensor<T, 0, ContainerType>::Iterator::operator++(int)
{
  Tensor<T, 0, ContainerType>::Iterator it {*this};
  ++(*this);
  return it;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::Iterator &Tensor<T, 0, ContainerType>::Iterator::operator++()
{
  offset_ += stride_;
  return *this;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::Iterator Tensor<T, 0, ContainerType>::Iterator::operator--(int)
{
  Tensor<T, 0, ContainerType>::Iterator it {*this};
  --(*this);
  return it;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::Iterator &Tensor<T, 0, ContainerType>::Iterator::operator--()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, typename ContainerType>
bool Tensor<T, 0, ContainerType>::Iterator::operator==(Tensor<T, 0, ContainerType>::Iterator const &it) const
{
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ------------------- ConstIterators ---------------------- */

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ConstIterator::ConstIterator(Tensor<T, 1, ContainerType> const &tensor, size_t)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ConstIterator::ConstIterator(ConstIterator const &it)
  : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ConstIterator::ConstIterator(ConstIterator &&it)
  : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> const Tensor<T, 0, ContainerType>::ConstIterator::operator*()
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> const Tensor<T, 0, ContainerType>::ConstIterator::operator->()
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ConstIterator Tensor<T, 0, ContainerType>::ConstIterator::operator++(int)
{
  Tensor<T, 0, ContainerType>::ConstIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ConstIterator &Tensor<T, 0, ContainerType>::ConstIterator::operator++()
{
  offset_ += stride_;
  return *this;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ConstIterator Tensor<T, 0, ContainerType>::ConstIterator::operator--(int)
{
  Tensor<T, 0, ContainerType>::ConstIterator it {*this};
  --(*this);
  return it;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ConstIterator &Tensor<T, 0, ContainerType>::ConstIterator::operator--()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, typename ContainerType>
bool Tensor<T, 0, ContainerType>::ConstIterator::operator==(Tensor<T, 0, ContainerType>::ConstIterator const &it) const
{
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* -------------------- ReverseIterator ---------------------- */

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ReverseIterator::ReverseIterator(Tensor<T, 1, ContainerType> const &tensor, size_t)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{
  offset_ += stride_ * (tensor.shape_.dimensions_[0] - 1);
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ReverseIterator::ReverseIterator(ReverseIterator const &it)
  : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ReverseIterator::ReverseIterator(ReverseIterator &&it)
  : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> Tensor<T, 0, ContainerType>::ReverseIterator::operator*()
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> Tensor<T, 0, ContainerType>::ReverseIterator::operator->()
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ReverseIterator Tensor<T, 0, ContainerType>::ReverseIterator::operator++(int)
{
  Tensor<T, 0, ContainerType>::ReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ReverseIterator &Tensor<T, 0, ContainerType>::ReverseIterator::operator++()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ReverseIterator Tensor<T, 0, ContainerType>::ReverseIterator::operator--(int)
{
  Tensor<T, 0, ContainerType>::ReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ReverseIterator &Tensor<T, 0, ContainerType>::ReverseIterator::operator--()
{
  offset_ += stride_;
  return *this;
}

template <typename T, typename ContainerType>
bool Tensor<T, 0, ContainerType>::ReverseIterator::operator==(Tensor<T, 0, ContainerType>::ReverseIterator const &it) const
{
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ----------------- ConstReverseIterator ------------------- */

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ConstReverseIterator::ConstReverseIterator(Tensor<T, 1, ContainerType> const &tensor, size_t)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{
  offset_ += stride_ * (tensor.shape_.dimensions_[0] - 1);
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator const &it)
  : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator &&it)
  : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> const Tensor<T, 0, ContainerType>::ConstReverseIterator::operator*()
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, typename ContainerType>
Tensor<T, 0, ContainerType> const Tensor<T, 0, ContainerType>::ConstReverseIterator::operator->()
{
  return Tensor<T, 0, ContainerType>(nullptr, nullptr, offset_, std::shared_ptr<ContainerType>(ref_));
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ConstReverseIterator Tensor<T, 0, ContainerType>::ConstReverseIterator::operator++(int)
{
  Tensor<T, 0, ContainerType>::ConstReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ConstReverseIterator &Tensor<T, 0, ContainerType>::ConstReverseIterator::operator++()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ConstReverseIterator Tensor<T, 0, ContainerType>::ConstReverseIterator::operator--(int)
{
  Tensor<T, 0, ContainerType>::ConstReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, typename ContainerType>
typename Tensor<T, 0, ContainerType>::ConstReverseIterator &Tensor<T, 0, ContainerType>::ConstReverseIterator::operator--()
{
  offset_ += stride_;
  return *this;
}

template <typename T, typename ContainerType>
bool Tensor<T, 0, ContainerType>::ConstReverseIterator::operator==(Tensor<T, 0, ContainerType>::ConstReverseIterator const &it) const
{
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ---------------------- Expressions ------------------------ */

template <typename LHSType, typename RHSType>
class BinaryAdd: public Expression<BinaryAdd<LHSType, RHSType>> {
public:

  /* ---------------- typedefs --------------- */

  typedef typename LHSType::value_type        value_type;
  typedef typename LHSType::container_type    container_type;
  typedef BinaryAdd                           self_type;

  /* -------------- Constructors -------------- */

  BinaryAdd(LHSType const &lhs, RHSType const &rhs);

  /* ---------------- Getters ----------------- */

  constexpr static size_t rank() { return LHSType::rank(); }
  size_t dimension(size_t index) const { return lhs_.dimension(index); }
  Shape<rank()> shape() { return lhs_.shape(); }

  template <typename... Indices>
  auto operator()(Indices... indices) const
    -> typename std::remove_reference<decltype(std::declval<LHSType>()(indices...))>::type;

private:
  LHSType const &lhs_;
  RHSType const &rhs_;
};

template <typename LHSType, typename RHSType>
BinaryAdd<LHSType, RHSType>::BinaryAdd(LHSType const &lhs, RHSType const &rhs)
  : lhs_(lhs), rhs_(rhs) {}

template <typename LHSType, typename RHSType>
template <typename... Indices>
auto BinaryAdd<LHSType, RHSType>::operator()(Indices... indices) const
  -> typename std::remove_reference<decltype(std::declval<LHSType>()(indices...))>::type
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS);
  return add(ValueAsTensor<LHSType>(lhs_)(indices...),
             ValueAsTensor<RHSType>(rhs_)(indices...));
}

template <typename LHSType, typename RHSType>
BinaryAdd<LHSType, RHSType> operator+(Expression<LHSType> const &lhs, Expression<RHSType> const &rhs)
{
  return BinaryAdd<LHSType, RHSType>(lhs.self(), rhs.self());
}

template <typename NodeType>
typename NodeType::value_type operator+(
    Expression<NodeType> const &expression, 
    typename NodeType::value_type const& scalar)
{
  return expression.self()() + scalar;
}

template <typename NodeType>
typename NodeType::value_type operator+(
    typename NodeType::value_type const& scalar,
    Expression<NodeType> const &expression)
{
  return expression.self()() + scalar;
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
class BinarySub: public Expression<BinarySub<LHSType, RHSType>> { /*@BinarySub*/

public:
  /* ---------------- typedefs --------------- */

  typedef typename LHSType::value_type value_type;
  typedef typename LHSType::container_type    container_type;
  typedef BinarySub                    self_type;

  /* -------------- Constructors -------------- */

  BinarySub(LHSType const &lhs, RHSType const &rhs);

  /* ---------------- Getters ----------------- */

  constexpr static size_t rank() { return LHSType::rank(); }
  size_t dimension(size_t index) const { return lhs_.dimension(index); }
  Shape<rank()> shape() { return lhs_.shape(); }
  template <typename... Indices>
  auto operator()(Indices... indices) const
  -> typename std::remove_reference<decltype(std::declval<LHSType>()(indices...))>::type;
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
auto BinarySub<LHSType, RHSType>::operator()(Indices... indices) const
  -> typename std::remove_reference<decltype(std::declval<LHSType>()(indices...))>::type
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS);
  return subtract(
      ValueAsTensor<LHSType>(lhs_)(indices...),
      ValueAsTensor<RHSType>(rhs_)(indices...));
}

template <typename LHSType, typename RHSType>
BinarySub<LHSType, RHSType> operator-(Expression<LHSType> const &lhs, Expression<RHSType> const &rhs)
{
  return BinarySub<LHSType, RHSType>(lhs.self(), rhs.self());
}

template <typename NodeType>
typename NodeType::value_type operator-(
    Expression<NodeType> const &expression, 
    typename NodeType::value_type const& scalar)
{
  return expression.self()() - scalar;
}

template <typename NodeType>
typename NodeType::value_type operator-(
    typename NodeType::value_type const& scalar, 
    Expression<NodeType> const &expression)
{
  return scalar - expression.self()();
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
class BinaryMul: public Expression<BinaryMul<LHSType, RHSType>> { /*@BinaryMul*/
public:

  /* ---------------- typedefs --------------- */

  typedef typename LHSType::value_type        value_type;
  typedef typename LHSType::container_type    container_type;
  typedef BinaryMul                           self_type;

  /* -------------- Constructors -------------- */

  BinaryMul(LHSType const &lhs, RHSType const &rhs);

  /* ---------------- Getters ----------------- */

  constexpr static size_t rank() { return LHSType::rank() + RHSType::rank() - 2; }
  size_t dimension(size_t index) const;
  template <typename... Indices>
  Tensor<value_type, LHSType::rank() + RHSType::rank() - sizeof...(Indices) - 2, container_type> 
    operator()(Indices... indices) const;

private:
  LHSType const &lhs_;
  RHSType const &rhs_;
};

template <typename LHSType, typename RHSType>
BinaryMul<LHSType, RHSType>::BinaryMul(LHSType const &lhs, RHSType const &rhs)
  : lhs_(lhs), rhs_(rhs) {}

template <typename LHSType, typename RHSType>
template <typename... Indices>
Tensor<typename LHSType::value_type, 
  LHSType::rank() + RHSType::rank() - sizeof...(Indices) - 2, 
  typename LHSType::container_type> 
BinaryMul<LHSType, RHSType>::operator()(Indices... indices) const
{
  static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS);
  return multiply(ValueAsTensor<LHSType>(lhs_)(),
                  ValueAsTensor<RHSType>(rhs_)())(indices...);
}

template <typename LHSType, typename RHSType>
BinaryMul<LHSType, RHSType> operator*(Expression<LHSType> const &lhs, Expression<RHSType> const &rhs)
{
  return BinaryMul<LHSType, RHSType>(lhs.self(), rhs.self());
}

template <typename NodeType>
Tensor<typename NodeType::value_type, NodeType::rank(), typename NodeType::container_type>
operator*(Expression<NodeType> const &expression, typename NodeType::value_type const& scalar)
{
  return expression.self()() * scalar;
}

template <typename NodeType>
Tensor<typename NodeType::value_type, NodeType::rank(), typename NodeType::container_type>
operator*(typename NodeType::value_type const& scalar, Expression<NodeType> const &expression)
{
  return scalar * expression.self()();
}

} // tensor

#undef NTENSOR_0CONSTRUCTOR
#undef NCONSTRUCTOR_0TENSOR
#undef NELEMENTS
#undef ZERO_ELEMENT
#undef EXPECTING_C_ARRAY
#undef DIMENSION_INVALID
#undef RANK_OUT_OF_BOUNDS
#undef INDEX_OUT_OF_BOUNDS
#undef ZERO_INDEX
#undef SLICES_EMPTY
#undef SLICES_OUT_OF_BOUNDS
#undef SLICE_INDICES_REPEATED
#undef SLICE_INDICES_DESCENDING
#undef RANK_MISMATCH
#undef EXPECTED_SCALAR
#undef DIMENSION_MISMATCH
#undef INNER_DIMENSION_MISMATCH
#undef ELEMENT_COUNT_MISMATCH
#undef SCALAR_TENSOR_MULT
#undef DEBUG

/* ----------------------------------------------- */


#endif // TENSORS_H_
