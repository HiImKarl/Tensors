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

#ifdef _TEST

extern int eDebugConstructorCounter; 

#endif

/* FIXME -- Remove debug macros */
#ifndef _NDEBUG

#define PRINT(x) std::cout << x << '\n';
#define PRINTV(x) std::cout << #x << ": " << x << '\n';

#define ARRAY_SIZE(x) sizeof(x)/sizeof(x[0])
#define GET_MACRO(_1,_2,_3,NAME,...) NAME
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
  "Tensor::slice(Args...) Failed -- At least one dimension must be sliced"
#define SLICES_OUT_OF_BOUNDS \
  "Tensor::slice(Args...) Failed -- Slices out of bounds"
#define SLICE_INDICES_REPEATED \
  "Tensor::slice(Args...) Failed -- Repeated slice indices"
#define SLICE_INDICES_DESCENDING \
  "Tensor::slice(Args...) Failed -- Slice indices must be listed in ascending order"

// Iteration
#define BEGIN_ON_NON_VECTOR \
  "Tensor::begin() should only be called on rank() 1 tensors, " \
  "use Tensor::begin(size_t) instead"

// Arithmetic Operations
#define RANK_MISMATCH \
  "Expecting same rank()"
#define SHAPE_MISMATCH \
  "Expecting same rank()"
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
#define NO_TENSORS_PROVIDED \
  "This method requires at least one tensor as an argument"

/* ---------------- Debug Messages --------------- */

#define PANIC_ASSERTION \
  "This assertion should never fire -> the developer messed up"

/* -------------------- Macros ------------------- */

// MSVC parses {} as an initializer_list
#ifdef _MSC_VER
#define VARDIAC_MAP(EXPR) \
  (void)std::initializer_list<int>{(EXPR, 0)...};
#else
#define VARDIAC_MAP(EXPR) \
  (void)(int[]){(EXPR, 0)...};
#endif

namespace tensor {

/* --------------- Forward Declerations --------------- */

namespace data {

template <typename T> class Array;
template <typename T> class HashMap;

} // namespace data


template <size_t N> class Shape;
template <> class Shape<0>;
template <size_t N> class Indices;
template <> class Indices<0>;
template <typename T, size_t N, template <class> class C = data::Array> class Tensor;
template <typename LHS, typename RHS> class BinaryAdd;
template <typename LHS, typename RHS> class BinarySub;
template <typename LHS, typename RHS> class BinaryMul;

/* ----------------- Type Definitions ---------------- */

template <typename T, template <class> class C = data::Array> 
using Scalar = Tensor<T, 0, C>;
template <typename T, template <class> class C = data::Array> 
using Vector = Tensor<T, 1, C>;
template <typename T, template <class> class C = data::Array> 
using Matrix = Tensor<T, 2, C>;

template <typename T, size_t N>
using SparseTensor = Tensor<T, N, data::HashMap>;
template <typename T>
using SparseMatrix = Tensor<T, 2, data::HashMap>;
template <typename T>
using SparseVector = Tensor<T, 1, data::HashMap>;
template <typename T>
using SparseScalar = Tensor<T, 0, data::HashMap>;

/* ------------- Template Meta-Patterns -------------- */

namespace meta {

/** returns x - y if x > y, 0 o.w. */
constexpr size_t NonZeroDifference(size_t x, size_t y)
{
  return x > y ? x - y : 0;
}

/** returns x if x < y, y o.w. */
constexpr size_t Min(size_t x, size_t y)
{
  return x < y ? x : y;
}

/** Returns 1 if x < y, 0 o.w. */
constexpr bool LessThan(size_t x, size_t y)
{
  return x < y;
}

/** template && */
template <bool B1, bool B2>
struct LogicalAnd { static constexpr bool value = B1 && B2; };

template <size_t Index, size_t Middle, size_t End>
struct FillSecond {
  template <typename... Args>
  FillSecond(size_t (&array1)[Middle], size_t (&array2)[End - Middle], size_t next, Args... args)
  {
    array2[Index - Middle] = next;
    FillSecond<Index + 1, Middle, End>(array1, array2, args...);
  }
};

template <size_t Middle, size_t End>
struct FillSecond<End, Middle, End> {
  template <typename... Args>
  FillSecond(size_t (&)[Middle], size_t (&)[End - Middle]) {}
};

template <size_t Index, size_t Middle, size_t End>
struct FillFirst {
  template <typename... Args>
  FillFirst(size_t (&array1)[Middle], size_t (&array2)[End - Middle], size_t next, Args... args)
  {
    array1[Index] = next;
    FillFirst<Index + 1, Middle, End>(array1, array2, args...);
  }
}; 
template <size_t Middle, size_t End>
struct FillFirst<Middle, Middle, End> {
  template <typename... Args>
  FillFirst(size_t (&array1)[Middle], size_t (&array2)[End - Middle], size_t next, Args... args)
  {
    array2[0] = next;
    FillSecond<Middle + 1, Middle, End>(array1, array2, args...);
  }
};

template <>
struct FillFirst<0, 0, 0> {
  template <typename... Args>
  FillFirst(size_t (&)[0], size_t (&)[0]) {}
};

template <size_t Middle, size_t End>
struct FillArgs {
  size_t array1[Middle];
  size_t array2[meta::NonZeroDifference(End, Middle)];

  template <typename... Args>
  FillArgs(Args... args)
  {
    static_assert(End == sizeof...(Args), PANIC_ASSERTION);
    FillFirst<0, Middle, Middle + meta::NonZeroDifference(End, Middle)>(
        array1, array2, args...);
  }
};

/** Member enum `value` is `true` iff `I...` is a 
 *  strictly increasing sequence of unsigned integers.
 */
template <size_t...>
struct IsIncreasingSequence;

/** Member enum `value` is `true` iff `I...` is a 
 *  strictly increasing sequence of unsigned integers.
 *  Recursive case.
 */
template <size_t I1, size_t I2, size_t... I>
struct IsIncreasingSequence<I1, I2, I...> {
  enum: bool { value = ((I1 < I2) && IsIncreasingSequence<I2, I...>::value) };
};

/** Member enum `value` is `true` iff `I...` is a 
 *  strictly increasing sequence of unsigned integers.
 *  Single value base case.
 */
template <size_t Index>
struct IsIncreasingSequence<Index> {
  enum: bool { value = true };
};

/** Member enum `value` is `true` iff `I...` is a 
 *  strictly increasing sequence of unsigned integers.
 *  Empty sequence base case.
 */
template <>
struct IsIncreasingSequence<> {
  enum: bool { value = true };
};

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`.
 */
template <size_t, size_t...>
struct CountLTMax;

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`. Recursive case.
 */
template <size_t Max, size_t Index, size_t... I>
struct CountLTMax<Max, Index, I...> {
  enum: size_t { value = LessThan(Index, Max) + CountLTMax<Max, I...>::value };
};

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`. Base case.
 */
template <size_t Max>
struct CountLTMax<Max> {
  enum: size_t { value = 0 }; 
};

/** Wrapper around a vardiac size_t pack */
template <size_t...> 
struct Sequence {};

/** Extends `Sequence<I...>` by placing `Index` in front */
template <size_t Index, typename>
struct Append;

/** Transform Sequence */
template <typename, template <size_t...> class, size_t...>
struct SequenceTransformer;

/** provides typedef `sequence` which is a sequence with each 
 *  element offset by `offset`.
 */
template <size_t...>
struct SequenceOffset;

/** Creates a `Sequence<I...>` where `I...` is the integers [`Index`, `N`)  */
template <size_t Index, size_t N>
struct MakeIndexSequence {
  using sequence = typename Append<Index, 
                   typename MakeIndexSequence<Index + 1, N>::sequence>::sequence;
};

/** Creates a `Sequence<I...>` where `I...` is the natural numbers up to `N`  */
template <size_t N>
struct MakeIndexSequence<N, N> {
  using sequence = Sequence<>;
};

/** provides typedef `sequence` which is a sequence with each 
 *  element offset by `offset`. Recursive case.
 */
template <size_t Offset, size_t Index, size_t... I>
struct SequenceOffset<Offset, Index, I...> {
    using sequence = typename Append<Index - Offset, 
                     typename SequenceOffset<Offset, I...>::sequence>::sequence;
};

/** provides typedef `sequence` which is a sequence with each 
 *  element offset by `offset`. Base case.
 */
template <size_t Offset>
struct SequenceOffset<Offset> {
    using sequence = Sequence<>;
};

/** Provides typedef `sequence` which is 
 *  `sequence` transformed with `transformer` 
 */ 
template <size_t... I, template <size_t...> class Transformer, size_t... Initial>
struct SequenceTransformer<Sequence<I...>, Transformer, Initial...> {
    using sequence = typename Transformer<Initial..., I...>::sequence;
};

/** Extends `Sequence<I...>` by placing `Index` in front */
template <size_t Index, size_t... I> 
struct Append<Index, Sequence<I...>> {
  using sequence = Sequence<Index, I...>;
};

/** Typedef `sequence` is a `Sequence<X...>` where 
 *  `X...` are the first `ThreshHold` elements of `I...` 
 */
template <size_t ThreshHold, size_t... I>
struct MakeSequence1;

/** Typedef `sequence` is a `Sequence<X...>` where 
 *  `X...` are the first `ThreshHold` elements of `I...` 
 *  Recursive case.
 */
template <size_t ThreshHold, size_t Index, size_t... I>
struct MakeSequence1<ThreshHold, Index, I...> {
  using sequence = typename Append<Index, typename MakeSequence1<ThreshHold - 1, I...>::sequence>::sequence;
};

/** Typedef `sequence` is a `Sequence<X...>` where 
 *  `X...` are the first `ThreshHold` elements of `I...` 
 *  Base case.
 */
template <size_t Index, size_t... I>
struct MakeSequence1<0, Index, I...> {
  using sequence = Sequence<>;
};

/** Typedef `sequence` is a `Sequence<X...>` where 
 *  `X...` are the first `ThreshHold` elements of `I...` 
 *  Base case.
 */
template <>
struct MakeSequence1<0> {
  using sequence = Sequence<>;
};

/** Typedef `sequence` is a `Sequence<X...>` where 
 *  `X...` are the  elements of `I...` after the first 
 *  `ThreshHold` elements. 
 */
template <size_t ThreshHold, size_t... I>
struct MakeSequence2;

/** Typedef `sequence` is a `Sequence<X...>` where 
 *  `X...` are the  elements of `I...` after the first 
 *  `ThreshHold` elements. Recursive Case.
 */
template <size_t ThreshHold, size_t Index, size_t... I>
struct MakeSequence2<ThreshHold, Index, I...> {
    using sequence = typename MakeSequence2<ThreshHold - 1, I...>::sequence;
};

/** Typedef `sequence` is a `Sequence<X...>` where 
 *  `X...` are the  elements of `I...` after the first 
 *  `ThreshHold` elements. Recursive Case.
 */
template <size_t Index, size_t... I>
struct MakeSequence2<0, Index, I...> {
    using sequence = typename Append<Index, typename MakeSequence2<0, I...>::sequence>::sequence;
};

/** Typedef `sequence` is a `Sequence<X...>` where 
 *  `X...` are the  elements of `I...` after the first 
 *  `ThreshHold` elements. Base Case.
 */
template <>
struct MakeSequence2<0> {
    using sequence = Sequence<>;
};

template <size_t Size, size_t... I>
struct InsertZeros;

template <size_t Size, size_t I1, size_t I2, size_t... Indices>
struct InsertZeros<Size, I1, I2, Indices...> {
  InsertZeros(size_t *indices)
  {
    static_assert(I1 != I2, SLICE_INDICES_REPEATED);
    static_assert(I1 < I2, SLICE_INDICES_DESCENDING);
    indices[I1] = 0;
    InsertZeros<Size, I2, Indices...>{indices};
  }
};

template <size_t Size, size_t Index>
struct InsertZeros<Size, Index> {
  InsertZeros(size_t *indices) 
  {
    static_assert(Index < Size, INDEX_OUT_OF_BOUNDS);
    indices[Index] = 0;
  }
};

template <size_t Size>
struct InsertZeros<Size> {
  InsertZeros(size_t *) {}
};

} // namespace meta

namespace details {

template <size_t N, size_t Count>
void UpdateIndices(Indices<0> &, Shape<N> const &, size_t (&)[Count], 
    size_t const * const (&)[Count], size_t = 0) {}

template <size_t M, size_t N, size_t Count>
void UpdateIndices(Indices<M> &reference_indices, Shape<N> const &shape, size_t (&indices)[Count], 
    size_t const * const (&strides)[Count], size_t quota_offset = 0)
{
  static_assert(M, PANIC_ASSERTION);
  int dim_index = M - 1;
  bool propogate = true;
  while (dim_index >= 0 && propogate) {
    ++reference_indices[dim_index];
    for (size_t i = 0; i < Count; ++i)
      indices[i] += strides[i][dim_index + quota_offset];
    if (reference_indices[dim_index] == shape[dim_index + quota_offset]) {
      reference_indices[dim_index] = 0;
      for (size_t i = 0; i < Count; ++i)
        indices[i] -= shape[dim_index + quota_offset] * strides[i][dim_index + quota_offset];
    } else {
      propogate = false;
    }
    --dim_index;
  }
}

} // namespace details

/* -------------- Tensor Meta-Patterns --------------- */

/** Boolean member `value` is true if T is an any-rank() Tensor 
 * object, false o.w.  
 */
template <typename T>
struct IsTensor { static bool const value = false; };

/** Tensor specialization of IsTensor, Boolean member 
 * `value` is true
 */
template <typename T, size_t N, template <class> class C>
struct IsTensor<Tensor<T, N, C>> { static bool const value = true; };

/** Boolean member `value` is true if T is a 0-rank() 
 * Tensor object, false o.w.
 */
template <typename T>
struct IsScalar { static bool const value = true; };

/** Scalar specialization of IsScalar, Boolean member 
 * `value` is true
 */
template <typename T, template <class> class C>
struct IsScalar<Tensor<T, 0, C>> { static bool const value = true; };

/** Tensor specialization of IsScalar, Boolean member 
 * `value` is false 
 */
template <typename T, size_t N, template <class> class C>
struct IsScalar<Tensor<T, N, C>> { static bool const value = false; };

/** Provides `value` equal to the rank() of `type`, 0 
 *  if not a tensor::Tensor type (C mutli-dimensional arrays 
 *  are considered to be rank() 0 in this context)
 */
template <typename T>
struct Rank { enum: size_t { value = 0 }; };

/** Provides `value` equal to the rank() of `type`, 0 
 *  if not a tensor::Tensor type (C mutli-dimensional arrays 
 *  are considered to be rank() 0 in this context)
 */
template <typename T, size_t N, template <class> class C>
struct Rank<Tensor<T, N, C>> { enum: size_t { value = N }; };

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
template <typename T, size_t N, template <class> class C>
struct ValueAsTensor<Tensor<T, N, C>> {
  ValueAsTensor(Tensor<T, N, C> const &val): value(val) {}
  Tensor<T, N, C> const &value;

  template <typename... Args>
  auto operator()(Args... args) const
    -> decltype(std::declval<Tensor<T, N, C> const>()(args...))
    { return value(args...); }

  template <typename... Args>
  auto at(Args... args) const
    -> decltype(std::declval<Tensor<T, N, C> const>().at(args...))
    { return value.at(args...); }

  template <size_t M> 
  auto operator[](Indices<M> const &indices) const
    -> decltype(std::declval<Tensor<T, N, C> const>()[indices])
    { return value[indices]; }

  template <size_t... Slices, typename... Args>
  auto slice(Args... args) const
    -> decltype(std::declval<Tensor<T, N, C> const>().slice(args...))
    { return value.template slice<Slices...>(args...); }

  template <size_t... Slices, size_t M>
  auto slice(Indices<M> const &indices) const
    -> decltype(std::declval<Tensor<T, N, C> const>().slice(indices))
    { return value.template slice<Slices...>(indices); }

};

/** BinaryAdd specialization of ValueAsTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <typename LHS, typename RHS>
struct ValueAsTensor<BinaryAdd<LHS, RHS>> {
  ValueAsTensor(BinaryAdd<LHS, RHS> const &val): value(val) {}
  BinaryAdd<LHS, RHS> const &value;
  typedef typename LHS::value_type          value_type;
  template <typename X>
  using container_type = typename           LHS::template container_type<X>;
  constexpr static size_t rank()            { return LHS::rank(); }
  typedef typename LHS::return_type         return_type;                       

  template <typename... Args>
  auto operator()(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>()(args...))>::type
    { return value(args...); }

  template <typename... Args>
  auto at(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().at(args...))>::type
    { return value.at(args...); }

  template <size_t M> 
  auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>()[indices])>::type
    { return value[indices]; }

  template <size_t... Slices, typename... Args>
  auto slice(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().slice(args...))>::type
    { return value.template slice<Slices...>(args...); }

  template <size_t... Slices, size_t M>
  auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().slice(indices))>::type
    { return value.template slice<Slices...>(indices); }
};

/** BinarySub specialization of ValueAsTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <typename LHS, typename RHS>
struct ValueAsTensor<BinarySub<LHS, RHS>> {
  ValueAsTensor(BinarySub<LHS, RHS> const &val): value(val) {}
  BinarySub<LHS, RHS> const &value;
  typedef typename LHS::value_type          value_type;
  template <typename X>
  using container_type = typename           LHS::template container_type<X>;
  constexpr static size_t rank()            { return LHS::rank(); }
  typedef typename LHS::return_type                  return_type;                       

  template <typename... Args>
  auto operator()(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>()(args...))>::type
    { return value(args...); }

  template <typename... Args>
  auto at(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().at(args...))>::type
    { return value.at(args...); }

  template <size_t M> 
  auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>()[indices])>::type
    { return value[indices]; }

  template <size_t... Slices, typename... Args>
  auto slice(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().slice(args...))>::type
    { return value.template slice<Slices...>(args...); }

  template <size_t... Slices, size_t M>
  auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().slice(indices))>::type
    { return value.template slice<Slices...>(indices); }
};

/** BinaryMul specialization of ValueAsTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <typename LHS, typename RHS>
struct ValueAsTensor<BinaryMul<LHS, RHS>> {
  ValueAsTensor(BinaryMul<LHS, RHS> const &val): value(val) {}
  BinaryMul<LHS, RHS> const &value;
  typedef typename LHS::value_type                   value_type;
  template <typename X>
  using container_type = typename                    LHS::template container_type<X>;
  constexpr static size_t rank()                     { return LHS::rank() + RHS::rank() - 2; }
  typedef Tensor<value_type, rank(), container_type> return_type;

  template <typename... Args>
  auto operator()(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>()(args...))>::type
    { return value(args...); }

  template <typename... Args>
  auto at(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().at(args...))>::type
    { return value.at(args...); }

  template <size_t M> 
  auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>()[indices])>::type
    { return value[indices]; }

  template <size_t... Slices, typename... Args>
  auto slice(Args... args) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().slice(args...))>::type
    { return value.template slice<Slices...>(args...); }

  template <size_t... Slices, size_t M>
  auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<
       decltype(std::declval<return_type const>().slice(indices))>::type
    { return value.template slice<Slices...>(indices); }
};

/* -------------------- Data Containers --------------------- */

namespace data {

template <typename T>
class Array { /*@Array<T>*/
/** Data container that allocates data in a single contiguous
 *  array. Thus indexing and iterating is very fast, but resizing
 *  any dimension will require reallocating and copying the 
 *  the entire array.
 */
public:

  /** Allocates an array of size `capacity`, with `operator new[]`. 
   *  Initializes every element to `T{}`, 
   */
  explicit Array(size_t capacity);

  /** Allocates an array of size `capacity`, with `operator new[]`. 
   *  Initialize every element to `value`, 
   */
  Array(size_t capacity, T const &value);

  /** Alloates an array of size `capacity` with `operator new[]`.
   *  Fills the array with  the elements between [`first`, `end`). 
   *  `std::distance(first, end)` must be equivalent to `capacity`.
   */
  template <typename It>
  Array(size_t capacity, It const &first, It const &end);

  /** Constructs an element at `index` with `value`. Must perform 
   *  correct forwarding. `index` must be less than capacity.
   */
  template <typename U>
  void assign(size_t index, U&& value)
    { data_[index] = std::forward<U>(value); }

  /** Calls `operator delete[]` on the underlying data */
  ~Array();

  /** Reference to the element at `index` offset of the 
   *  underlying array. `index` must be less than `capacity`.
   */
  T &operator[](size_t index) { return data_[index]; };

  /** Const-reference to the element at `index` offset of the 
   *  underlying array. `index` must be less than `capacity`.
   */
  T const &operator[](size_t index) const { return data_[index]; }

private:
  T *data_; /**< Contiguously allocated array of `T` */
};

template <typename T>
Array<T>::Array(size_t capacity)
  : data_(new T[capacity]()) {}

template <typename T>
Array<T>::Array(size_t capacity, T const &value)
  : data_(new T[capacity]())
{
  std::fill(data_, data_ + capacity, value);
}

template <typename T>
template <typename It>
Array<T>::Array(size_t capacity, It const &first, It const &end)
  : data_(new T[capacity]())
{
  std::copy(first, end, data_);
}

template <typename T>
Array<T>::~Array() 
{
  delete[] data_;
}

template <typename T>
class HashMap { /*@HashMap<T>*/
/** Data container that allocates data in a hash map 
 *  (STL unordered_map). The map only stores non-zero 
 *  entries (configurable zero-entry). Memory efficiency 
 *  for sparse tensors is good, but quickly degrades 
 *  as density increases. Index and iterating is fast,
 *  O(1), but resizing will require destroying and 
 *  rehashing larger dimensions.
 */
public:

  /** Creates a HashMap with initial capacity `capacity`,
   *  and sets the zero-element `T{}`.
   */
  explicit HashMap(size_t capacity);

  /** Creates a HashMap with initial capacity `capacity`,
   *  and sets the zero-element `value`.
   */
  HashMap(size_t capacity, T const &value);

  /** Creates a HashMap with initial capacity `capacity`. 
   *  Fills the array with  the elements between [`first`, `end`). 
   *  `std::distance(first, end)` must be equivalent to `capacity`.
   *  The zero-element is set to `T{}`.
   */
  template <typename It>
  HashMap(size_t capacity, It const &first, It const &end);

  /** Invokes STL unordered_map destructor */
  ~HashMap() = default;

  /** Constructs an element at `index` with `value`. Must perform 
   *  correct forwarding. `index` must be less than capacity.
   */
  template <typename U>
  void assign(size_t index, U&& value);

  /** Reference to the element at `index` offset of the 
   *  underlying array. `index` must be less than `capacity`.
   */
  T &operator[](size_t index);

  /** Const-reference to the element at `index` offset of the 
   *  underlying array. `index` must be less than `capacity`.
   */
  T const &operator[](size_t index) const;

private:
  std::unordered_map<size_t, T> data_; /**< underlying STL hash map */
  T zero_elem_;
};

template <typename T>
HashMap<T>::HashMap(size_t)
  : data_({}), zero_elem_(T{})
{}

template <typename T>
HashMap<T>::HashMap(size_t, T const &value)
  : data_({}), zero_elem_(value)
{}

template <typename T>
template <typename It>
HashMap<T>::HashMap(size_t capacity, It const &first, It const &end)
  : data_({}), zero_elem_(T{})
{
  assert((long)capacity == std::distance(first, end) && PANIC_ASSERTION);
  (void)capacity;
  size_t index = 0;
  for (auto it = first; it != end; ++it) {
    if (*it != zero_elem_) data_.insert({index, *it});
    ++index;
  }
} 

template <typename T>
template <typename U>
void HashMap<T>::assign(size_t index, U&& value)
{
  if (value == zero_elem_) data_.erase(index);
  else data_[index] = std::forward<U>(value);
}

template <typename T>
T &HashMap<T>::operator[](size_t index)
{
  auto it = data_.find(index);
  if (it != data_.end()) return it->second;
  return (data_[index] = zero_elem_);
}

template <typename T>
T const &HashMap<T>::operator[](size_t index) const
{
  auto it = data_.find(index);
  if (it != data_.end()) return it->second;
  return zero_elem_;
}

} // namespace data

/* ----------------------- Core Data Structures ----------------------- */

/** CRTP base for Tensor expressions */
template <typename NodeType>
struct Expression {
  inline NodeType &self() { return *static_cast<NodeType *>(this); }
  inline NodeType const &self() const { return *static_cast<NodeType const*>(this); }
};

template <size_t N> /*@Shape<N>*/
class Shape {
/** Tensor shape object, where size_t template `N` represents the Tensor rank().
 *  Implemented as a wrapper around size_t[N].
 */
public:
  /* -------------------- typedefs -------------------- */

  typedef size_t                    size_type;
  typedef ptrdiff_t                 difference_type;
  typedef Shape<N>                  self_type;

  /* ----------------- friend classes ----------------- */

  template <typename X, size_t M, template <class> class C_> friend class Tensor;
  template <typename LHS, typename RHS> friend class BinaryAdd;
  template <typename LHS, typename RHS> friend class BinarySub;
  template <typename LHS, typename RHS> friend class BinaryMul;

  /* ------------------ Constructors ------------------ */

  explicit Shape(std::initializer_list<size_t> dimensions); /**< initializer_list constructor */
  Shape(Shape<N> const &shape);                  /**< Copy constructor */

  /* ------------------ Assignment -------------------- */

  Shape<N> &operator=(Shape<N> const &shape);   /**< Copy assignment */

  /* -------------------- Getters --------------------- */

  constexpr static size_t rank() { return N; } /**< `N` */

  /** Get dimension reference at `index`. If debugging, fails an assertion 
   * if index is out of bounds. Note: 0-based indexing.
   */
  size_t &operator[](size_t index);            

  /** Get dimension at `index`. If debugging, fails an assertion
   * if index is out of bounds. Note: 0-based indexing.
   */
  size_t operator[](size_t index) const;

  /* -------------------- Equality -------------------- */

  /** `true` iff every dimension is identical */
  bool operator==(Shape<N> const& shape) const noexcept; 

  /** `true` iff any dimension is different */
  bool operator!=(Shape<N> const& shape) const noexcept { return !(*this == shape); }

  /** `true` iff rank()s are identical and every dimension is identical */
  template <size_t M>
  bool operator==(Shape<M> const& shape) const noexcept; 

  /** `true` iff rank()s are not identical or any dimension is different */
  template <size_t M>
  bool operator!=(Shape<M> const& shape) const noexcept { return !(*this == shape); }

  /* ----------------- Expressions ------------------ */

  template <typename X, typename Y, size_t M1, size_t M2, template <class> class C1, template <class> class C2>
  friend Tensor<X, M1 + M2 - 2, C1> multiply(Tensor<X, M1, C1> const& tensor_1, Tensor<Y, M2, C2> const& tensor_2);

  template <typename U, template <class> class C_> friend Tensor<U, 2, C_> transpose(Tensor<U, 2, C_> &mat);
  template <typename U, template <class> class C_> friend Tensor<U, 2, C_> transpose(Tensor<U, 1, C_> &vec);

  /* ------------------- Utility -------------------- */

  /** Returns the product of all of the indices */
  size_t index_product() const noexcept;

  /* -------------------- Print --------------------- */

  template <typename U, size_t M, template <class> class C_>
  friend std::ostream &operator<<(std::ostream &os, Tensor<U, M, C_> const&tensor);
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
class Indices { /*@Indices<N>*/
  /** Wrapper around size_t[N] and Shape<N> to provide a specialized array
   *  for accessing Tensors and convenient looping.
   */
public:

  /* --------------- Friend Classes ------------- */

  template <typename U, size_t M, template <class> class C_> friend class Tensor;
  template <typename LHS, typename RHS> friend class BinaryAdd;
  template <typename LHS, typename RHS> friend class BinarySub;
  template <typename LHS, typename RHS> friend class BinaryMul;

  /* ---------------- Constructors -------------- */

  Indices();
  Indices(std::initializer_list<size_t> indices);
  size_t const &operator[](size_t index) const;
  size_t &operator[](size_t index);

  /** Increments `this` so it effectively refers to the
   *  next element in an N-rank() Tensor. The dimensions are
   *  propogated accordingly, I.e. (1, 3) indices with a 2x3 
   *  shape will become (2, 0) after increment. indices will
   *  overflow if incremented beyond their maximum value, returns
   *  true if overflow, false o.w.
   */
  bool increment(Shape<N> const &shape);
  
  /** Decrements `indices` so it effectively refers to the
   *  next element in an N-rank() Tensor. The dimensions are
   *  propogated accordingly, I.e. (2, 0) indices with a 2x3 
   *  shape will become (1, 3) after decrement. Indices will
   *  underflow if decremented at 0, returns true if 
   *  underflow, false o.w.
   */
  bool decrement(Shape<N> const &shape);
  
private:
  size_t indices_[N];

  // private constructor
  Indices(size_t const (&indices)[N]);
};

template <size_t N>
Indices<N>::Indices(std::initializer_list<size_t> indices)
{
  assert(indices.size() == N && RANK_MISMATCH);
  size_t *ptr = indices_;
  for (size_t index : indices)
    *(ptr++) = index;
}

template <size_t N>
Indices<N>::Indices()
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

template <size_t N>
bool Indices<N>::increment(Shape<N> const &shape)
{
  int dim_index = N - 1;
  bool propogate = true;
  while (dim_index >= 0 && propogate) {
    ++indices_[dim_index];
    if (indices_[dim_index] == shape[dim_index]) {
      indices_[dim_index] = 0;
      --dim_index;
    } else {
      propogate = false;
    }
  }
  return dim_index < 0;
}

template <size_t N>
bool Indices<N>::decrement(Shape<N> const &shape)
{
  int dim_index = N - 1;
  bool propogate = true;
  while (dim_index >= 0 && propogate) {
    if (indices_[dim_index] == 0) {
      indices_[dim_index] = shape[dim_index] - 1;
      --dim_index;
    } else  {
      --indices_[dim_index];
      propogate = false;
    }
  }
  return dim_index < 0;
}

template <size_t N>
Indices<N>::Indices(size_t const (&indices)[N])
{
  std::copy_n(indices, N, indices_);
}

/** Proxy object used to construct */
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

template <typename T, size_t N, template <class> class C>
class Tensor: public Expression<Tensor<T, N, C>> { 
//  @Tensor<T, N, C>
/** Any-rank() array of type `T`, where rank() `N` is a size_t template.
 *  The underlying data is implemented as a dynamically allocated contiguous
 *  array.
 */  
public:

  /* ------------------ Type Definitions --------------- */
  typedef T                                     value_type;
  template <typename X> using container_type =  C<X>;
  typedef T&                                    reference;
  typedef T const&                              const_reference;
  typedef size_t                                size_type;
  typedef ptrdiff_t                             difference_type;
  typedef Tensor<T, N, C>           self_type;
  typedef Tensor<T, N, C>           return_type;

  /* ----------------- Friend Classes ----------------- */

  template <typename X, size_t M, template <class> class C_> friend class Tensor;
  template <typename LHS, typename RHS> friend class BinaryAdd;
  template <typename LHS, typename RHS> friend class BinarySub;
  template <typename LHS, typename RHS> friend class BinaryMul;
  template <typename X> friend struct ValueAsTensor;

  /* ------------------ Proxy Objects ----------------- */

  class Proxy { /*@Proxy<T,N>*/
  /**
   * Proxy Tensor Object used for building tensors from reference.
   * This is used to differentiate proxy construction 
   * for move and copy construction only.
   */
  public:
    template <typename U, size_t M, template <class> class C_> friend class Tensor;
    Proxy() = delete;
  private:
    Proxy(Tensor<T, N, C> const &tensor): tensor_(tensor) {}
    Proxy(Proxy const &proxy): tensor_(proxy.tensor_) {}
    Tensor<T, N, C> const &tensor_;
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
   *  be equal to the tensor's declared rank(). 
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
  Tensor(Tensor<T, N, C> const &tensor); 

  /** Move construction, takes ownership of underlying data, `tensor` is destroyed */
  Tensor(Tensor<T, N, C> &&tensor); 

  /** Constructs a reference to the `proxy` tensor. The tensors share 
   *  the same underyling data, so changes will affect both tensors.
   */

  Tensor(typename Tensor<T, N, C>::Proxy const &proxy); 

  /** Constructs the tensor produced by the expression */
  template <typename NodeType,
            typename = typename std::enable_if<NodeType::rank() == N>::type>
  Tensor(Expression<NodeType> const& expression);

  /* ------------------- Destructor ------------------- */

  ~Tensor() = default;

  /* ------------------- Assignment ------------------- */

  /** Assign to every element of `this` the corresponding element in
   *  `tensor`. The shapes must match.
   */
  template <typename U, template <class> class C_>
  Tensor<T, N, C> &operator=(Tensor<U, N, C_> const &tensor);

  /** Assign to every element of `this` the corresponding element in
   *  `tensor`. The shapes must match.
   */
  Tensor<T, N, C> &operator=(Tensor<T, N, C> const &tensor);

  /** Assign to every element of `this` the corresponding element in
   *  `rhs` after expression evaluation. The shapes must match.
   */
  template <typename NodeType>
  Tensor<T, N, C> &operator=(Expression<NodeType> const &rhs);

  /* --------------------- Getters --------------------- */

  constexpr static size_t rank() { return N; } /**< Get `N` */

  /** Get the dimension at index. If debugging, fails an assertion
   *  if is out of bounds. Note: indexing starts at 0.
   */
  size_t dimension(size_t index) const { return shape_[index]; }

  /** Get a reference to the tensor shape */
  Shape<N> const &shape() const noexcept { return shape_; } 

  // FIXME :: Is there a way to hide these and keep it-> functional?
  /** used to implement iterator-> */
  Tensor *operator->() { return this; } 

  /** used to implement const_iterator-> */
  Tensor const *operator->() const { return this; } 

  /* ------------------ Access To Data ----------------- */

  template <typename... Args>
  Tensor<T, N - sizeof...(Args), C> at(Args... args);

  /** Returns the resulting tensor by applying left to right index expansion of
   *  the provided arguments. I.e. calling `tensor(1, 2)` on a rank() 4 tensor is
   *  equivalent to `tensor(1, 2, :, :)`. If debugging, fails an assertion if
   *  any of the indices are out bounds. Note: indexing starts at 0.
   */
  template <typename... Args,
            typename = typename std::enable_if<N != sizeof...(Args)>::type>
  Tensor<T, N - sizeof...(Args), C> operator()(Args... args);

  template <typename... Args,
            typename = typename std::enable_if<N == sizeof...(Args)>::type>
  T &operator()(Args... args);

  template <typename... Args>
  Tensor<T, N - sizeof...(Args), C> const at(Args... args) const;

  /** See operator() */
  template <typename... Args,
            typename = typename std::enable_if<N != sizeof...(Args)>::type>
  Tensor<T, N - sizeof...(Args), C> const operator()(Args... args) const;

  /** See operator() */
  template <typename... Args,
            typename = typename std::enable_if<N == sizeof...(Args)>::type>
  T const &operator()(Args... args) const;

  /** See operator() */
  template <size_t M, 
      typename = typename std::enable_if<N != M>::type>
  Tensor<T, N - M, C> operator[](Indices<M> const &indices);

  template <size_t M, 
      typename = typename std::enable_if<N == M>::type>
  T &operator[](Indices<M> const &indices);

  /** See operator() const */
  template <size_t M,
      typename = typename std::enable_if<N != M>::type>
  Tensor<T, N - M, C> const operator[](Indices<M> const &indices) const;

  template <size_t M,
      typename = typename std::enable_if<N == M>::type>
  T const &operator[](Indices<M> const &indices) const;


  /** Slices denotate the dimensions which are left free, while indices
   *  fix the remaining dimensions at the specified index. I.e. calling
   *  `tensor.slice<1, 3, 5>(1, 2)` on a rank() 5 tensor is equivalent to
   *  `tensor(:, 1, :, 2, :)` and produces a rank() 3 tensor. If debugging,
   *   fails an assertion if any of the indices are out of bounds. Note:
   *   indexing begins at 0.
   */
  template <size_t... Slices, typename... Args,
            typename = typename std::enable_if<N != sizeof...(Args)>::type>
  Tensor<T, N - sizeof...(Args), C> slice(Args... args);

  /** Const version of slice<Slices...>(Args.. args) */
  template <size_t... Slices, typename... Args,
            typename = typename std::enable_if<N != sizeof...(Args)>::type>
  Tensor<T, N - sizeof...(Args), C> const slice(Args... args) const;

  /** Indentical to operator()(Args...) && `sizeof...(Args) == N` 
   *  with a static check to verify `sizeof...(Slices) == 0`.
   */
  template <size_t... Slices, typename... Args,
            typename = typename std::enable_if<N == sizeof...(Args)>::type>
  T &slice(Args... args);

  /** Const version of slice<>(Args...) && `sizeof...(Args) == N` */
  template <size_t... Slices, typename... Args,
            typename = typename std::enable_if<N == sizeof...(Args)>::type>
  T const &slice(Args... args) const;

  /** See slice<Slices...>(Args..); */
  template <size_t... Slices, size_t M,
            typename = typename std::enable_if<N != M>::type>
  Tensor<T, N - M, C> slice(Indices<M> const &indices);
  
  /** See slice<Slices...>(Args..); */
  template <size_t... Slices, size_t M,
            typename = typename std::enable_if<N != M>::type>
  Tensor<T, N - M, C> const slice(Indices<M> const &indices) const;

  /** Indentical to operator[](Indices<M> const&) && `M == N` 
   *  with a static check to verify `sizeof...(Slices) == 0`.
   */
  template <size_t... Slices, size_t M,
            typename = typename std::enable_if<N == M>::type>
  T &slice(Indices<M> const&);

  /** Const version of slice<>(Indices<M> const&) && `M == N` */
  template <size_t... Slices, size_t M,
            typename = typename std::enable_if<N == M>::type>
  T const &slice(Indices<M> const&) const;

  /* -------------------- Expressions ------------------- */

  template <typename X, typename Y, size_t M, template <class> class C_, typename FunctionType>
  friend Tensor<X, M, C_> elem_wise(Tensor<X, M, C_> const &tensor, Y const &scalar,
      FunctionType &&fn);

  template <typename X, typename Y, size_t M, template <class> class C_, typename FunctionType>
  friend Tensor<X, M, C_> elem_wise(Tensor<X, M, C_> const &tensor1, Tensor<Y, M, C_> const &tensor_2, FunctionType &&fn);

  template <typename X, typename Y, size_t M, template <class> class C1, template <class> class C2>
  friend Tensor<X, M, C1> add(
      Tensor<X, M, C1> const& tensor_1, 
      Tensor<Y, M, C2> const& tensor_2);

  template <typename RHS>
  Tensor<T, N, C> &operator+=(Expression<RHS> const &rhs);

  template <typename X, typename Y, size_t M, template <class> class C_>
  friend Tensor<X, M, C_> subtract(Tensor<X, M, C_> const& tensor_1, Tensor<Y, M, C_> const& tensor_2);

  template <typename RHS>
  Tensor<T, N, C> &operator-=(Expression<RHS> const &rhs);

  template <typename X, typename Y, size_t M1, size_t M2, template <class> class C1, template <class> class C2>
  friend Tensor<X, M1 + M2 - 2, C1> multiply(Tensor<X, M1, C1> const& tensor_1, Tensor<Y, M2, C2> const& tensor_2);

  template <typename RHS>
  Tensor<T, N, C> &operator*=(Expression<RHS> const &rhs);

  /** Allocates a Tensor with shape equivalent to *this, and whose
   *  elements are equivalent to *this with operator-() applied.
   */
  Tensor<T, N, C> operator-() const;

  /* ------------------ Print to ostream --------------- */

  template <typename X, size_t M, template <class> class C_>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M, C_> &tensor);

  /* -------------------- Equivalence ------------------ */

  /** Returns true iff the tensor's dimensions and data are equivalent */
  bool operator==(Tensor<T, N, C> const& tensor) const; 

  /** Returns true iff the tensor's dimensions or data are not equivalent */
  bool operator!=(Tensor<T, N, C> const& tensor) const { return !(*this == tensor); }

  /** Returns true iff the tensor's dimensions are equal and every element satisfies e1 == e2 */
  template <typename X>
  bool operator==(Tensor<X, N, C> const& tensor) const;

  /** Returns true iff the tensor's dimensions are different or any element satisfies e1 != e2 */
  template <typename X>
  bool operator!=(Tensor<X, N, C> const& tensor) const { return !(*this == tensor); }

  /* -------------------- Iterators --------------------- */

  class Iterator { /*@Iterator<T, N>*/
  public:
    /** Iterator with freedom across one dimension of a Tensor.
     *  Allows access to the underlying tensor data.
     */

    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, template <class> class C_> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    Iterator(Iterator const &it);  

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    Iterator(Iterator &&it);       

    Tensor<T, N, C> operator*();   /**< Create a reference to the underlying Tensor */
    Tensor<T, N, C> operator->();  /**< Syntatic sugar for (*it). */
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
    Iterator(Tensor<T, N + 1, C> const &tensor, size_t index);
    Shape<N> shape_; // Data describing the underlying tensor 
    size_t strides_[N];
    size_t offset_;
    std::shared_ptr<C<T>> ref_;
    size_t stride_; // Step size of the underlying data pointer per increment
  };

  class ConstIterator { /*@ConstIterator<T, N>*/
  public:
    /** Constant iterator with freedom across one dimension of a Tensor.
     *  Does not allow write access to the underlying tensor data.
     */

    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, template <class> class C_> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    ConstIterator(ConstIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ConstIterator(ConstIterator &&it);

    Tensor<T, N, C> const operator*();  /**< Create a reference to the underlying Tensor */
    Tensor<T, N, C> const operator->(); /**< Syntatic sugar for (*it). */                                
    ConstIterator operator++(int);   /**< Increment (postfix). Returns a temporary before increment */
    ConstIterator &operator++();     /**< Increment (prefix). Returns *this */
    ConstIterator operator--(int);   /**< Decrement (postfix). Returns a temporary before decrement */
    ConstIterator &operator--();     /**< Decrement (prefix). Returns *this */
    bool operator==(ConstIterator const &it) const;
    bool operator!=(ConstIterator const &it) const { return !(it == *this); }
  private:
    ConstIterator(Tensor<T, N + 1, C> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    size_t offset_;
    std::shared_ptr<C<T>> ref_;

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

    template <typename U, size_t M, template <class> class C_> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    ReverseIterator(ReverseIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ReverseIterator(ReverseIterator &&it);

    Tensor<T, N, C> operator*();        /**< Create a reference to the underlying Tensor */
    Tensor<T, N, C> operator->();       /**< Syntatic sugar for (*it). */                                
    ReverseIterator operator++(int); /**< Increment (postfix). Returns a temporary before increment */
    ReverseIterator &operator++();   /**< Increment (prefix). Returns *this */
    ReverseIterator operator--(int); /**< Decrement (postfix). Returns a temporary before decrement */
    ReverseIterator &operator--();   /**< Decrement (prefix). Returns *this */
    bool operator==(ReverseIterator const &it) const;
    bool operator!=(ReverseIterator const &it) const { return !(it == *this); }
  private:
    ReverseIterator (Tensor<T, N + 1, C> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    size_t offset_;
    std::shared_ptr<C<T>> ref_;

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

    template <typename U, size_t M, template <class> class C_> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy constructs an iterator to the same underlying Tensor */
    ConstReverseIterator(ConstReverseIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ConstReverseIterator(ConstReverseIterator &&it);
    Tensor<T, N, C> const operator*();       /**< Create a reference to the underlying Tensor */
    Tensor<T, N, C> const operator->();      /**< Syntatic sugar for (*it). */                                
    ConstReverseIterator operator++(int); /**< Increment (postfix). Returns a temporary before increment */
    ConstReverseIterator &operator++();   /**< Increment (prefix). Returns *this */
    ConstReverseIterator operator--(int); /**< Decrement (postfix). Returns a temporary before decrement */
    ConstReverseIterator &operator--();   /**< Decrement (prefix). Returns *this */
    bool operator==(ConstReverseIterator const &it) const;
    bool operator!=(ConstReverseIterator const &it) const { return !(it == *this); }
  private:
    ConstReverseIterator (Tensor<T, N + 1, C> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    size_t offset_;
    std::shared_ptr<C<T>> ref_;

    /**
     * Step size of the underlying data pointer per increment
     */
    size_t stride_;
  };

  /** Returns an iterator for a Tensor, equivalent to *this dimension
   *  fixed at index (the iteration index). If debugging, an assertion
   *  will fail if index is out of bounds. Note: indexing begins at 0. 
   */
  typename Tensor<T, N - 1, C>::Iterator begin(size_t index);

  /** Returns a just-past-the-end iterator for a Tensor, equivalent 
   *  to *this dimension fixed at index (the iteration index). 
   *  If debugging, and assertion will fail if `index` is out of 
   *  bounds. Note: indexing begins at 0.
   */
  typename Tensor<T, N - 1, C>::Iterator end(size_t index);

  /** Equivalent to Tensor<T, N, C>::begin(0) */
  typename Tensor<T, N - 1, C>::Iterator begin();
  /** Equivalent to Tensor<T, N, C>::end(0) */
  typename Tensor<T, N - 1, C>::Iterator end();

  /** See Tensor<T, N, C>::begin(size_t), except returns a const iterator */
  typename Tensor<T, N - 1, C>::ConstIterator cbegin(size_t index) const;
  /** See Tensor<T, N, C>::end(size_t), except returns a const iterator */
  typename Tensor<T, N - 1, C>::ConstIterator cend(size_t index) const;
  /** See Tensor<T, N, C>::begin(), except returns a const iterator */
  typename Tensor<T, N - 1, C>::ConstIterator cbegin() const;
  /** See Tensor<T, N, C>::end(), except returns a const iterator */
  typename Tensor<T, N - 1, C>::ConstIterator cend() const;

  /** See Tensor<T, N, C>::begin(size_t), except returns a reverse iterator */
  typename Tensor<T, N - 1, C>::ReverseIterator rbegin(size_t index);
  /** See Tensor<T, N, C>::end(size_t), except returns a reverse iterator */
  typename Tensor<T, N - 1, C>::ReverseIterator rend(size_t index);
  /** See Tensor<T, N, C>::begin(), except returns a reverse iterator */
  typename Tensor<T, N - 1, C>::ReverseIterator rbegin();
  /** See Tensor<T, N, C>::end(), except returns a reverse iterator */
  typename Tensor<T, N - 1, C>::ReverseIterator rend();

  /** See Tensor<T, N, C>::begin(size_t), except returns a const reverse iterator */
  typename Tensor<T, N - 1, C>::ConstReverseIterator crbegin(size_t index) const;
  /** See Tensor<T, N, C>::end(size_t), except returns a const reverse iterator */
  typename Tensor<T, N - 1, C>::ConstReverseIterator crend(size_t index) const;
  /** See Tensor<T, N, C>::begin(), except returns a const reverse iterator */
  typename Tensor<T, N - 1, C>::ConstReverseIterator crbegin() const;
  /** See Tensor<T, N, C>::end(), except returns a const reverse iterator */
  typename Tensor<T, N - 1, C>::ConstReverseIterator crend() const;

  /* ----------------- Utility Functions ---------------- */
  
  /** Allocates a Tensor with shape `shape`, whose total number of elements 
   *  must be equivalent to *this (or an assertion will fail during debug).
   *  The resulting Tensor is filled by iterating through *this and copying
   *  over the values.
   */
  template <size_t M, template <class> class C_ = C>
  Tensor<T, M, C_> resize(Shape<M> const &shape) const;

  /** Returns a deep copy of this tensor, equivalent to calling copy constructor */
  Tensor<T, N, C> copy() const; 

  /** Returns a reference: only used to invoke reference constructor */
  typename Tensor<T, N, C>::Proxy ref();

  template <typename U, size_t M, template <class> class C_, typename RAIt>
  friend void Fill(Tensor<U, M, C_> &tensor, RAIt const &begin, RAIt const &end);

  template <typename U, size_t M, template <class> class C_, typename X>
  friend void Fill(Tensor<U, M, C_> &tensor, X const &value);

  template <typename U, template <class> class C_> 
  friend Tensor<U, 2, C_> transpose(Tensor<U, 2, C_> &mat);

  template <typename U, template <class> class C_> 
  friend Tensor<U, 2, C_> transpose(Tensor<U, 1, C_> &vec);

  template <typename U, typename FunctionType> 
  U reduce( U&& initial_value, FunctionType&& fun) const;

  template <typename FunctionType, typename... Tensors>
  friend void Map(FunctionType &&fn, Tensors&&... tensors);

  template <typename U, typename FunctionType, typename... Tensors>
  friend U Reduce(U&& initial_value, FunctionType &&fn, Tensors const&... tensors);

private:
  /* ----------------- data ---------------- */

  Shape<N> shape_;
  size_t strides_[N];
  size_t offset_;
  std::shared_ptr<C<T>> ref_;

  /* --------------- Getters --------------- */

  size_t const *strides() const noexcept { return strides_; }

  /* ----------- Expansion for operator()(...) ----------- */

  template <typename... Args>
  size_t pIndicesExpansion(Args... args) const;

  /* ------------- Expansion for slice() ------------- */

  // Expansion
  template <size_t M>
  Tensor<T, N - M, C> pSliceExpansion(size_t (&placed_indices)[N], Indices<M> const &indices);

  /* --------- Expansion for Map() & Reduce() --------- */

  inline T const &pGet(size_t *indices, size_t index) const;
  inline T &pGet(size_t *indices, size_t index);

  template <typename U, size_t M, template <class> class C_, typename... Tensors>
  friend inline Shape<M> const &pGetShape(Tensor<U, M, C_> tensor, Tensors&&... tensors);

  template <typename FunctionType, size_t... I, typename... Tensors>
  friend inline void pMapForwardSequence(
      FunctionType &&fn, size_t *indices, meta::Sequence<I...>, Tensors&&... tensors);

  template <typename U, typename FunctionType, size_t... I, typename... Tensors>
  friend inline void pReduceForwardSequence(U &ret_val,
      FunctionType &&fn, size_t *indices, meta::Sequence<I...>, Tensors const&... tensors);

  /* 0----------------- Utility --------------------- */

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

  // Initialize strides :: DIMENSIONS MUST BE INITIALIZED FIRST
  void pInitializeStrides();

  // Declare all fields in the constructor, but initialize strides assuming no
  // gaps
  Tensor(size_t const *dimensions, size_t offset,
      std::shared_ptr<C<T>> &&_ref);

  // Declare all fields in the constructor
  Tensor(size_t const *dimensions, size_t const *strides, size_t offset,
      std::shared_ptr<C<T>> &&_ref);

}; // Tensor

/* ----------------------------- Constructors ------------------------- */

template <typename T, size_t N, template <class> class C> 
Tensor<T, N, C>::Tensor(std::initializer_list<size_t> dimensions) 
  : shape_(Shape<N>(dimensions)), offset_(0), 
  ref_(std::make_shared<C<T>>(shape_.index_product())) 
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  pInitializeStrides(); 
}

template <typename T, size_t N, template <class> class C> 
Tensor<T, N, C>::Tensor(size_t const (&dimensions)[N], T const& value)
  : shape_(Shape<N>(dimensions)), offset_(0) 
{ 
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  for (size_t i = 0; i < N; ++i)
    assert(dimensions[i] && ZERO_ELEMENT); 
  pInitializeStrides(); 
  size_t cumul = shape_.index_product(); 
  ref_ = std::make_shared<C<T>>(cumul, value); 
}

template <typename T, size_t N, template <class> class C> 
template <typename FunctionType, typename... Arguments> 
Tensor<T, N, C>::Tensor(size_t const (&dimensions)[N], 
    std::function<FunctionType> &f, Arguments&&...  args) 
  : shape_(Shape<N>(dimensions)), offset_(0) 
{ 
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  for (size_t i = 0; i < N; ++i) 
    assert(dimensions[i] && ZERO_ELEMENT); 
  pInitializeStrides();
  auto value_setter = 
    [&f, &args...](T &lhs) -> void { lhs = f(args...); }; 
  ref_ = std::make_shared<C<T>>(shape_.index_product());
  Map(value_setter, *this); 
}

template <typename T, size_t N, template <class> class C> 
template <typename Array> 
Tensor<T, N, C>::Tensor(_A<Array> &&md_array): offset_(0) 
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  static_assert(std::rank<Array>::value == N, RANK_MISMATCH); 
  using ArrayType = typename std::remove_all_extents<Array>::type; 
  SetDimensions<Array, 0, N>{}(shape_.dimensions_); 
  pInitializeStrides();
  // Make use of the fact C arrays are contiguously allocated
  ArrayType *ptr = (ArrayType *)md_array.value; 
  size_t cumul = shape_.index_product();
  ref_ = std::make_shared<C<T>>(shape_.index_product(), ptr, ptr + cumul);
}

template <typename T, size_t N, template <class> class C> 
Tensor<T, N, C>::Tensor(Shape<N> const &shape)
   : shape_(shape), offset_(0), 
   ref_(std::make_shared<C<T>>(shape_.index_product())) 
{ 
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  pInitializeStrides(); 
}

template <typename T, size_t N, template <class> class C> 
Tensor<T, N, C>::Tensor(Tensor<T, N, C> const &tensor)
   : shape_(tensor.shape_), offset_(0) 
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  pInitializeStrides(); 
  size_t cumul = shape_.index_product();
  ref_ = std::make_shared<C<T>>(cumul); 
  Indices<N> reference_indices{};
  size_t indices[2] = {};
  size_t const * const strides[] = {this->strides_, tensor.strides_};
  for (size_t i = 0; i < cumul; ++i) {
    ref_->assign(indices[0],
                (static_cast<C<T> const&>(*tensor.ref_))[tensor.offset_ + indices[1]]);
    details::UpdateIndices(reference_indices, this->shape_, indices, strides);
  }
}

template <typename T, size_t N, template <class> class C> 
Tensor<T, N, C>::Tensor(Tensor<T, N, C> &&tensor) 
  : shape_(tensor.shape_), offset_(tensor.offset_),
    ref_(std::move(tensor.ref_)) 
{ 
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  std::copy_n(tensor.strides_, N, strides_); 
}

template <typename T, size_t N, template <class> class C> 
Tensor<T, N, C>::Tensor(
    typename Tensor<T, N, C>::Proxy const &proxy)
  : shape_(proxy.tensor_.shape_), offset_(proxy.tensor_.offset_), 
  ref_(proxy.tensor_.ref_) 
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  std::copy_n(proxy.tensor_.strides_, N, strides_); 
}

template <typename T, size_t N, template <class> class C> template <typename
NodeType, typename> Tensor<T, N, C>::Tensor(Expression<NodeType> const& rhs) 
  : offset_(0)
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  auto const &expression = rhs.self(); 
  shape_ = expression.shape(); 
  pInitializeStrides(); 
  size_t cumul = shape_.index_product();
  ref_ = std::make_shared<C<T>>(cumul);
  Indices<N> reference_indices{};
  size_t indices[1] = {};
  size_t const * const strides[] = { this->strides_ };
  for (size_t i = 0; i < cumul; ++i) {
    ref_->assign(indices[0], expression[reference_indices]);
    details::UpdateIndices(reference_indices, this->shape_, indices, strides);
  }
}

template <typename T, size_t N, template <class> class C>
template <typename U, template <class> class C_>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(Tensor<U, N, C_> const &tensor)
{
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
  size_t cumul = shape_.index_product();
  Indices<N> reference_indices{};
  size_t indices[2] = {};
  size_t const * const strides[2] = { this->strides_, tensor.strides_ };
  for (size_t i = 0; i < cumul; ++i) {
    ref_->assign(indices[0],
        (static_cast<C_<U> const&>(*tensor.ref_))[tensor.offset_ + indices[1]]);
    details::UpdateIndices(reference_indices, this->shape_, indices, strides);
  }
  return *this;
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(Tensor<T, N, C> const &tensor)
{
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
  size_t cumul = shape_.index_product();
  Indices<N> reference_indices{};
  size_t indices[2] = {};
  size_t const * const strides[] = {this->strides_, tensor.strides_};
  for (size_t i = 0; i < cumul; ++i) {
    ref_->assign(indices[0],
                (static_cast<C<T> const&>(*tensor.ref_))[tensor.offset_ + indices[1]]);
    details::UpdateIndices(reference_indices, this->shape_, indices, strides);
  }
  return *this;
}

template <typename T, size_t N, template <class> class C>
template <typename NodeType>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(Expression<NodeType> const &rhs)
{
  auto const &expression = rhs.self();
  assert((shape_ == expression.shape()) && DIMENSION_MISMATCH);
  Tensor<T, N, C> tensor(this->shape());
  Indices<N> reference_indices{};
  size_t indices[1] = {};
  size_t const * const strides[1] = { tensor.strides_ };
  for (size_t i = 0; i < shape_.index_product(); ++i) {
    tensor.ref_->assign(indices[0], expression[reference_indices]);
    details::UpdateIndices(reference_indices, this->shape_, indices, strides);
  }
  *this = tensor;
  return *this;
}

template <typename T, size_t N, template <class> class C>
template <typename... Args>
Tensor<T, N - sizeof...(Args), C> Tensor<T, N, C>::at(Args... args)
{
  constexpr size_t M = sizeof...(args); 
  size_t cumul_index = pIndicesExpansion(args...);
  return Tensor<T, N - M, C>(shape_.dimensions_ + M, strides_ + M, offset_ + cumul_index, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
template <typename... Args, typename>
Tensor<T, N - sizeof...(Args), C> Tensor<T, N, C>::operator()(Args... args)
{
  constexpr size_t M = sizeof...(args); 
  size_t cumul_index = pIndicesExpansion(args...);
  return Tensor<T, N - M, C>(shape_.dimensions_ + M, strides_ + M, offset_ + cumul_index, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
template <typename... Args, typename>
T &Tensor<T, N, C>::operator()(Args... args)
{
  size_t cumul_index = pIndicesExpansion(args...);
  return (*ref_)[cumul_index + offset_];
}

template <typename T, size_t N, template <class> class C>
template <typename... Args>
Tensor<T, N - sizeof...(Args), C> const Tensor<T, N, C>::at(Args... args) const
{
 return (*const_cast<self_type*>(this))(args...);
}

template <typename T, size_t N, template <class> class C>
template <typename... Args, typename>
Tensor<T, N - sizeof...(Args), C> const Tensor<T, N, C>::operator()(Args... args) const
{
  return (*const_cast<self_type*>(this))(args...);
}

template <typename T, size_t N, template <class> class C>
template <typename... Args, typename>
T const &Tensor<T, N, C>::operator()(Args... args) const
{
  size_t cumul_index = pIndicesExpansion(args...);
  return (*static_cast<C<T> const *>(ref_.get()))[cumul_index + offset_];
}

template <typename T, size_t N, template <class> class C>
template <size_t M, typename>
Tensor<T, N - M, C> Tensor<T, N, C>::operator[](Indices<M> const &indices)
{
  size_t cumul_index = 0;
  for (size_t i = 0; i < M; ++i)
    cumul_index += strides_[M - i - 1] * indices[M - i - 1];
  return Tensor<T, N - M, C>(
      shape_.dimensions_ + M, strides_ + M,  offset_ + cumul_index, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
template <size_t M, typename>
T &Tensor<T, N, C>::operator[](Indices<M> const &indices)
{
  size_t cumul_index = 0;
  for (size_t i = 0; i < N; ++i)
    cumul_index += strides_[N - i - 1] * indices[N - i - 1];
  return (*ref_)[cumul_index + offset_];
}

template <typename T, size_t N, template <class> class C>
template <size_t M, typename>
Tensor<T, N - M, C> const Tensor<T, N, C>::operator[](Indices<M> const &indices) const
{
  return (*const_cast<self_type*>(this))[indices];
}

template <typename T, size_t N, template <class> class C>
template <size_t M, typename>
T const &Tensor<T, N, C>::operator[](Indices<M> const &indices) const
{
  size_t cumul_index = 0;
  for (size_t i = 0; i < N; ++i)
    cumul_index += strides_[N - i - 1] * indices[N - i - 1];
  return (*static_cast<C<T> const*>(ref_.get()))[cumul_index + offset_];
}

template <typename T, size_t N, template <class> class C>
template <size_t... Slices, typename... Args, typename>
Tensor<T, N - sizeof...(Args), C> Tensor<T, N, C>::slice(Args... args)
{
  static_assert(N >= sizeof...(Slices) + sizeof...(args), SLICES_OUT_OF_BOUNDS);
  size_t placed_indices[N];
  // Initially fill the array with 1s
  // place 0s where the indices are sliced
  std::fill_n(placed_indices, N, 1);
  meta::InsertZeros<N, Slices...>{placed_indices};
  // slice dimensions (aka set 0) not explicitly sliced nor 
  // filled by `indices`
  size_t unfilled_dimensions = N - sizeof...(Args) - sizeof...(Slices);
  for (size_t i = 0; i < N; ++i) {
    if (!unfilled_dimensions) break;
    if (placed_indices[N - i - 1]) {
      placed_indices[N - i - 1] = 0;
      --unfilled_dimensions;
    }
  }
  Indices<sizeof...(args)> indices {};
  size_t index = 0;
  auto contract_indices = [&](size_t arg) -> void {
    indices[index++] = arg;
  };
  VARDIAC_MAP(contract_indices(args));
  (void)(contract_indices);
  return pSliceExpansion<sizeof...(args)>(placed_indices, indices);
}

template <typename T, size_t N, template <class> class C>
template <size_t... Slices, typename... Args, typename>
Tensor<T, N - sizeof...(Args), C> const Tensor<T, N, C>::slice(Args... indices) const
{
  return const_cast<self_type*>(this)->slice<Slices...>(indices...);
}

template <typename T, size_t N, template <class> class C>
template <size_t... Slices, typename... Args, typename>
T &Tensor<T, N, C>::slice(Args... args)
{
  static_assert(sizeof...(Args) == N, PANIC_ASSERTION);
  static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
  return (*this)(args...);
}

template <typename T, size_t N, template <class> class C>
template <size_t... Slices, typename... Args, typename> 
T const &Tensor<T, N, C>::slice(Args... args) const
{
  static_assert(sizeof...(Args) == N, PANIC_ASSERTION);
  static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
  return (*this)(args...);
}

template <typename T, size_t N, template <class> class C>
template <size_t... Slices, size_t M, typename>
Tensor<T, N - M, C>
    Tensor<T, N, C>::slice(Indices<M> const &indices)
{
  static_assert(N >= M + sizeof...(Slices), SLICES_OUT_OF_BOUNDS);
  size_t placed_indices[N];
  // Initially fill the array with 1s
  // place 0s where the indices are sliced
  std::fill_n(placed_indices, N, 1);
  meta::InsertZeros<N, Slices...>{placed_indices};
  // slice dimensions (aka set 0) not explicitly sliced nor 
  // filled by `indices`
  size_t unfilled_dimensions = N - M - sizeof...(Slices);
  for (size_t i = 0; i < N; ++i) {
    if (!unfilled_dimensions) break;
    if (placed_indices[N - i - 1]) {
      placed_indices[N - i - 1] = 0;
      --unfilled_dimensions;
    }
  }
  return pSliceExpansion<M>(placed_indices, indices);
}

template <typename T, size_t N, template <class> class C>
template <size_t... Slices, size_t M, typename>
Tensor<T, N - M, C> const 
    Tensor<T, N, C>::slice(Indices<M> const &indices) const
{
  return const_cast<self_type*>(this)->slice<Slices...>(indices);
}

template <typename T, size_t N, template <class> class C>
template <size_t... Slices, size_t M, typename>
T &Tensor<T, N, C>::slice(Indices<M> const& indices)
{
  static_assert(M == N, PANIC_ASSERTION);
  static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
  return (*this)[indices];
}

template <typename T, size_t N, template <class> class C>
template <size_t... Slices, size_t M, typename>
T const &Tensor<T, N, C>::slice(Indices<M> const& indices) const
{
  static_assert(M == N, PANIC_ASSERTION);
  static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
  return (*this)[indices];
}

template <typename T, size_t N, template <class> class C>
bool Tensor<T, N, C>::operator==(Tensor<T, N, C> const& tensor) const
{
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);

  size_t indices_product = shape_.index_product();
  for (size_t i = 0; i < indices_product; ++i)
    if ((*ref_)[i + offset_] != (*tensor.ref_)[i + tensor.offset_]) return false;
  return true;
}

template <typename T, size_t N, template <class> class C>
template <typename X>
bool Tensor<T, N, C>::operator==(Tensor<X, N, C> const& tensor) const
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
template <typename T, size_t N, template <class> class C_>
std::ostream &operator<<(std::ostream &os, const Tensor<T, N, C_> &tensor)
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
    int dim_index = N - 1;
    // find the correct index to "step" to
    while (dim_index >= 0 && propogate) {
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

template <typename T, size_t N, template <class> class C>
template <typename... Args>
size_t Tensor<T, N, C>::pIndicesExpansion(Args... args) const
{
  constexpr size_t M = sizeof...(Args);
  static_assert(N >= M, RANK_OUT_OF_BOUNDS);
  size_t index = M - 1;
  auto convert_index = [&](size_t dim) -> size_t {
    assert((dim < shape_.dimensions_[M - index - 1]) && INDEX_OUT_OF_BOUNDS);
    return strides_[M - (index--) - 1] * dim;
  }; 
  size_t cumul_index = 0;
  VARDIAC_MAP(cumul_index += convert_index(args));
  // surpress compiler unused lval warnings
  (void)(convert_index);
  return cumul_index;
}

/* ------------ Expansion for Slice() ------------ */

template <typename T, size_t N, template <class> class C>
template <size_t M> 
Tensor<T, N - M, C> Tensor<T, N, C>::pSliceExpansion(size_t (&placed_indices)[N], Indices<M> const &indices)
{
  // FIXME -- This function can be much more concise
  size_t array_index = 0;
  for (size_t i = 0; i < N; ++i) {
    if (!placed_indices[i]) continue;
    assert((shape_.dimensions_[i] > indices[array_index]) && INDEX_OUT_OF_BOUNDS);
    // 0 is reserved for the "sliced" indices, so add one to all
    // of the accessed indices to differentiate
    placed_indices[i] = indices[array_index++] + 1;
  }
  size_t offset = 0;
  size_t dimensions[N - M];
  size_t strides[N - M];
  array_index = 0;
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
  return Tensor<T, N - M, C>(dimensions, strides, offset_ + offset, std::shared_ptr<C<T>>(ref_));
}

/* ------------ Expansion for Map() ------------ */

template <typename T, size_t N, template <class> class C>
T const &Tensor<T, N, C>::pGet(size_t *indices, size_t index) const
{
  return (static_cast<C<T> const&>(*ref_))[offset_ + indices[index]];
}

template <typename T, size_t N, template <class> class C>
T &Tensor<T, N, C>::pGet(size_t *indices, size_t index)
{
  return (*ref_)[offset_ + indices[index]];
}

template <typename U, size_t M, template <class> class C_, typename... Tensors>
Shape<M> const &pGetShape(Tensor<U, M, C_> tensor, Tensors&&...)
{
  return tensor.shape();
}

template <typename FunctionType, size_t... I, typename... Tensors>
void pMapForwardSequence(FunctionType &&fn, size_t *indices, meta::Sequence<I...>, Tensors&&... tensors)
{
  fn(tensors.pGet(indices, I)...);
}

template <typename U, typename FunctionType, size_t... I, typename... Tensors>
void pReduceForwardSequence(U &ret_val, FunctionType &&fn, size_t *indices, 
    meta::Sequence<I...>, Tensors const&... tensors)
{
  ret_val = fn(ret_val, tensors.pGet(indices, I)...);
}

/* -------------- Utility Methods --------------- */
template <typename T, size_t N, template <class> class C>
void Tensor<T, N, C>::pInitializeStrides()
{
  size_t accumulator = 1;
  for (size_t i = 0; i < N; ++i) {
    strides_[N - i - 1] = accumulator;
    accumulator *= shape_.dimensions_[N - i - 1];
  }
}

// private constructors
template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::Tensor(size_t const *dimensions, size_t offset, std::shared_ptr<C<T>> &&ref)
  : shape_(Shape<N>(dimensions)), offset_(offset), ref_(std::move(ref))
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  pInitializeStrides();
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::Tensor(size_t const *dimensions, size_t const *strides, size_t offset, 
    std::shared_ptr<C<T>> &&ref)
  : shape_(Shape<N>(dimensions)), offset_(offset), ref_(std::move(ref))
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  std::copy_n(strides, N, strides_);
}


/* -------------------------- Expressions -------------------------- */

/** Elementwise Scalar-Tensor operation. Returns a Tensor with shape 
 *  `tensor`, where each element of the new Tensor is the result of `fn` 
 *  applied with the corresponding `tensor` elements and `scalar`.
 */
template <typename X, typename Y, size_t M, template <class> class C_, typename FunctionType>
Tensor<X, M, C_> elem_wise(Tensor<X, M, C_> const &tensor, Y const &scalar,
      FunctionType &&fn)
{
  Tensor<X, M, C_> result {tensor.shape()};
  auto set_vals = [&scalar, &fn](X &lhs, X const &rhs) -> void {
    lhs = fn(rhs, scalar);
  };
  Map(set_vals, result, tensor);
  return result;
}

template <typename X, typename Y, size_t M, template <class> class C_, typename FunctionType>
Tensor<X, M, C_> elem_wise(Tensor<X, M, C_> const &tensor1, Tensor<Y, M, C_> const &tensor2,
      FunctionType &&fn)
{
  Tensor<X, M, C_> result {tensor1.shape()};
  auto set_vals = [&fn](X &lhs, X const &rhs1, Y const &rhs2) -> void {
    lhs = fn(rhs1, rhs2);
  };
  Map(set_vals, result, tensor1, tensor2);
  return result;
}

/** Creates a Tensor whose elements are the elementwise sum of `tensor1` 
 *  and `tensor2`. `tensor1` and `tensor2` must have equivalent shape, or
 *  an assertion will fail during debug.
 */
template <typename X, typename Y, size_t M, template <class> class C1, template <class> class C2>
Tensor<X, M, C1> add(
    Tensor<X, M, C1> const& tensor_1, 
    Tensor<Y, M, C2> const& tensor_2)
{
  assert((tensor_1.shape_ == tensor_2.shape_)  && DIMENSION_MISMATCH);
  Tensor<X, M, C1> sum_tensor(tensor_1.shape_);
  auto add = [](X &x, X const &y, Y const &z) -> void
  {
    x = y + z;
  };
  Map(add, sum_tensor, tensor_1, tensor_2);
  return sum_tensor;
}

template <typename T, size_t N, template <class> class C>
template <typename RHS>
Tensor<T, N, C> &Tensor<T, N, C>::operator+=(Expression<RHS> const &rhs)
{
  auto tensor = rhs.self()();
  assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
  *this = *this + tensor;
  return *this;
}

/** Creates a Tensor whose elements are the elementwise difference of `tensor1` 
 *  and `tensor2`. `tensor1` and `tensor2` must have equivalent shape, or
 *  an assertion will fail during debug.
 */
template <typename X, typename Y, size_t M, template <class> class C_>
Tensor<X, M, C_> subtract(Tensor<X, M, C_> const& tensor_1, Tensor<Y, M, C_> const& tensor_2)
{
  assert((tensor_1.shape_ == tensor_2.shape_) && DIMENSION_MISMATCH);
  Tensor<X, M, C_> diff_tensor(tensor_1.shape_);
  auto sub = [](X &x, X const &y, Y const &z) -> void
  {
    x = y - z;
  };
  Map(sub, diff_tensor, tensor_1, tensor_2);
  return diff_tensor;
}

template <typename T, size_t N, template <class> class C>
template <typename RHS>
Tensor<T, N, C> &Tensor<T, N, C>::operator-=(Expression<RHS> const &rhs)
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
 *  to produce a N rank() Tensor with all dimensions m is O(m^(N+1)).
 */
template <typename X, typename Y, size_t M1, size_t M2, template <class> class C1, template <class> class C2>
Tensor<X, M1 + M2 - 2, C1> multiply(Tensor<X, M1, C1> const& tensor_1, Tensor<Y, M2, C2> const& tensor_2)
{
  static_assert(M1, SCALAR_TENSOR_MULT);
  static_assert(M2, SCALAR_TENSOR_MULT);
  assert((tensor_1.shape_.dimensions_[M1 - 1] == tensor_2.shape_.dimensions_[0]) 
      && INNER_DIMENSION_MISMATCH);

  auto shape = Shape<M1 + M2 - 2>();
  std::copy_n(tensor_1.shape_.dimensions_, M1 - 1, shape.dimensions_);
  std::copy_n(tensor_2.shape_.dimensions_ + 1, M2 - 1, shape.dimensions_ + M1 - 1);
  Tensor<X, M1 + M2 - 2, C1> prod_tensor(shape);
  size_t cumul_index_1 = tensor_1.shape_.index_product() / tensor_1.shape_.dimensions_[M1 - 1];
  size_t cumul_index_2 = tensor_2.shape_.index_product() / tensor_2.shape_.dimensions_[0];
  Indices<M1 - 1> reference_indices_1{};
  Indices<M2 - 1> reference_indices_2{};
  size_t index = 0;
  size_t t1_indices[1] = {};
  size_t const * const t1_strides[] = { tensor_1.strides_ };
  for (size_t i1 = 0; i1 < cumul_index_1; ++i1) {
    size_t t2_indices[1] = {};
    size_t const * const t2_strides[] = { tensor_2.strides_ };
    for (size_t i2 = 0; i2 < cumul_index_2; ++i2) {
      X value {};
      for (size_t x = 0; x < tensor_1.shape_.dimensions_[M1 - 1]; ++x)
          value += (*tensor_1.ref_)[tensor_1.offset_ + t1_indices[0] + tensor_1.strides_[M1 - 1] * x] *
            (*tensor_2.ref_)[tensor_2.offset_ + t2_indices[0] + tensor_2.strides_[0] * x];
      (*prod_tensor.ref_)[index] = value;
      details::UpdateIndices(reference_indices_2, tensor_2.shape(), t2_indices, t2_strides, 1);
      ++index;
    }
    details::UpdateIndices(reference_indices_1, tensor_1.shape(), t1_indices, t1_strides);
  }
  return prod_tensor;
}

/** FIXME -- Vector-Vector multiplication is usually left undefined by literature:
 *  for now define it as the hadamard product.
 */ 
template <typename X, typename Y, template <class> class C1, template <class> class C2>
X multiply(Tensor<X, 1, C1> const& tensor_1, Tensor<Y, 1, C2> const& tensor_2)
{
  assert(tensor_1.shape() == tensor_2.shape() && SHAPE_MISMATCH);
  auto mul_vals = [](X &accum, X const &x, Y const &y) {
    return accum + x * y; 
  };
  //return tensor_1.reduce(tensor_2, mul_vals, X{});
  return Reduce(X{}, mul_vals, tensor_1, tensor_2);
}

template <typename T, size_t N, template <class> class C>
template <typename RHS>
Tensor<T, N, C> &Tensor<T, N, C>::operator*=(Expression<RHS> const &rhs)
{
  auto tensor = rhs.self()();
  *this = *this * tensor;
  return *this;
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> Tensor<T, N, C>::operator-() const
{
  Tensor<T, N, C> neg_tensor(shape_);
  auto neg = [](T &x, T const &y) -> void
  {
    x = -y;
  };
  Map(neg, neg_tensor, *this);
  return neg_tensor;
}

/* --------------------------- Useful Functions ------------------------- */

template <typename T, size_t N, template <class> class C>
template <size_t M, template <class> class C_>
Tensor<T, M, C_> Tensor<T, N, C>::resize(Shape<M> const &shape) const
{
  assert(shape_.index_product() == shape.index_product() && ELEMENT_COUNT_MISMATCH);
  Tensor<T, M, C_> resized_tensor(shape);
  Indices<N> indices = Indices<N>();
  Indices<M> indices_resized = Indices<M>();
  do {
    resized_tensor[indices_resized] = (*this)[indices];
    indices_resized.increment(shape);
  } while (!indices.increment(shape_));
  return resized_tensor;
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> Tensor<T, N, C>::copy() const
{
  return Tensor<T, N, C>(*this);
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::Proxy Tensor<T, N, C>::ref() 
{
  return Proxy(*this);
}

/** Fills the elements of `tensor` with the elements between.
 *  `begin` and `end`, which must be random access iterators. The number 
 *  elements between `begin` and `end` must be equivalent to the capacity of
 *  `tensor`, otherwise an assertion will fail during debug.
 */
template <typename U, size_t M, template <class> class C_, typename RAIt>
void Fill(Tensor<U, M, C_> &tensor, RAIt const &begin, RAIt const &end)
{
  size_t cumul_sum = tensor.shape_.index_product();
  auto dist_sum = std::distance(begin, end);
  assert((dist_sum > 0 && cumul_sum == (size_t)dist_sum) && NELEMENTS);
  RAIt it = begin;
  auto allocate = [&it](U &x) -> void
  {
    x = *it;
    ++it;
  };
  Map(allocate, tensor);
}

/** Assigns to each element in Tensor the value `value`
 */
template <typename U, size_t M, template <class> class C_, typename X>
void Fill(Tensor<U, M, C_> &tensor, X const &value)
{
  auto allocate = [&value](U &x) -> void
  {
    x = value;
  };
  Map(allocate, tensor);
}

/** Returns a transposed Matrix, sharing the same underlying data as `mat`. If
 *  the dimension of `mat` is [n, m], the dimensions of the resulting matrix will be
 *  [m, n]. Note: This is only applicable to matrices and vectors
 */
template <typename U, template <class> class C_> 
Tensor<U, 2, C_> transpose(Tensor<U, 2, C_> &mat)
{
  size_t transposed_dimensions[2];
  transposed_dimensions[0] = mat.shape_.dimensions_[1];
  transposed_dimensions[1] = mat.shape_.dimensions_[0];
  size_t transposed_strides[2];
  transposed_strides[0] = mat.strides_[1];
  transposed_strides[1] = mat.strides_[0];
  return Tensor<U, 2, C_>(transposed_dimensions, transposed_strides,
      mat.offset_, std::shared_ptr<C_<U>>(mat.ref_));
}

/** See Tensor<T, N, C> transpose(Tensor<T, N, C> &)
 */
template <typename U, template <class> class C_>
Tensor<U, 2, C_> const transpose(Tensor<U, 2, C_> const &mat) 
{
  return (*const_cast<typename std::decay<decltype(mat)>::type*>(mat)).tranpose();
}

/** Returns a transposed Matrix, sharing the same underlying data as vec. If
 *  the dimension of `vec` is [n], the dimensions of the resulting matrix will be
 *  [1, n]. Note: This is only applicable to matrices and vectors
 */
template <typename T, template <class> class C>
Tensor<T, 2, C> transpose(Tensor<T, 1, C> &vec) 
{
  size_t transposed_dimensions[2];
  transposed_dimensions[0] = 1;
  transposed_dimensions[1] = vec.shape_.dimensions_[0];
  size_t transposed_strides[2];
  transposed_strides[0] = 1;
  transposed_strides[1] = vec.strides_[0];
  return Tensor<T, 2, C>(transposed_dimensions, transposed_strides,
      vec.offset_, std::shared_ptr<C<T>>(vec.ref_));
}

template <typename T, size_t N, template <class> class C>
template <typename U, typename FunctionType> 
U Tensor<T, N, C>::reduce(U&& initial_value, FunctionType&& fun) const
{
  U ret_val = std::forward<U>(initial_value);
  auto accum = [&](T const &x) {
    ret_val = fun(ret_val, x);
  };
  Map(accum, *this);
  return ret_val;
}

template <typename FunctionType, typename... Tensors>
void Map(FunctionType &&fn, Tensors&&... tensors)
{
  static_assert(sizeof...(tensors), NO_TENSORS_PROVIDED);
  auto shape = pGetShape(tensors...);
  constexpr size_t M = shape.rank();
  VARDIAC_MAP(assert(shape == tensors.shape() && SHAPE_MISMATCH));
  size_t cumul_index = shape.index_product();
  Indices<M> reference_indices {};
  size_t indices[sizeof...(Tensors)] = {};
  size_t const * const strides[sizeof...(Tensors)] = { tensors.strides_... };
  auto sequence = typename meta::MakeIndexSequence<0, sizeof...(Tensors)>::sequence{};
  for (size_t i = 0; i < cumul_index; ++i) {
    pMapForwardSequence(fn, indices, sequence, tensors...);
    details::UpdateIndices(reference_indices, shape, indices, strides);
  }
}

template <typename U, typename FunctionType, typename... Tensors>
U Reduce(U&& initial_value, FunctionType &&fn, Tensors const&... tensors)
{
  static_assert(sizeof...(tensors), NO_TENSORS_PROVIDED);
  auto shape = pGetShape(tensors...);
  VARDIAC_MAP(assert(shape == tensors.shape() && SHAPE_MISMATCH));
  U ret_val = std::forward<U>(initial_value);
  constexpr size_t M = shape.rank();
  size_t cumul_index = shape.index_product();
  Indices<M> reference_indices {};
  size_t indices[sizeof...(Tensors)] = {};
  size_t const * const strides[sizeof...(Tensors)] = { tensors.strides_... };
  auto sequence = typename meta::MakeIndexSequence<0, sizeof...(Tensors)>::sequence{};
  for (size_t i = 0; i < cumul_index; ++i) {
    pReduceForwardSequence(ret_val, fn, indices, sequence, tensors...);
    details::UpdateIndices(reference_indices, shape, indices, strides);
  }
  return ret_val;
}

/* ------------------------------- Iterator ----------------------------- */

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::Iterator::Iterator(Tensor<T, N + 1, C> const &tensor, size_t index)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::Iterator::Iterator(Iterator const &it)
  : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::Iterator::Iterator(Iterator &&it)
  : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> Tensor<T, N, C>::Iterator::operator*()
{
  return Tensor<T, N, C>(shape_.dimensions_, strides_, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> Tensor<T, N, C>::Iterator::operator->()
{
  return Tensor<T, N, C>(shape_.dimensions_, strides_, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::Iterator Tensor<T, N, C>::Iterator::operator++(int)
{
  Tensor<T, N, C>::Iterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::Iterator &Tensor<T, N, C>::Iterator::operator++()
{
  offset_ += stride_;
  return *this;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::Iterator Tensor<T, N, C>::Iterator::operator--(int)
{
  Tensor<T, N, C>::Iterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::Iterator &Tensor<T, N, C>::Iterator::operator--()
{
  offset_ -= stride_;
  return *this;
} 

template <typename T, size_t N, template <class> class C>
bool Tensor<T, N, C>::Iterator::operator==(
  typename Tensor<T, N, C>::Iterator const &it) const
{
  if (shape_ != it.shape_) return false;
  if (stride_ != it.stride_) return false;
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ----------------------------- ConstIterator --------------------------- */

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ConstIterator::ConstIterator(Tensor<T, N + 1, C> const &tensor, size_t index)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ConstIterator::ConstIterator(ConstIterator const &it)
  : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ConstIterator::ConstIterator(ConstIterator &&it)
  : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> const Tensor<T, N, C>::ConstIterator::operator*()
{
  return Tensor<T, N, C>(shape_.dimensions_, strides_, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> const Tensor<T, N, C>::ConstIterator::operator->()
{
  return Tensor<T, N, C>(shape_.dimensions_, strides_, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ConstIterator Tensor<T, N, C>::ConstIterator::operator++(int)
{
  Tensor<T, N, C>::ConstIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ConstIterator &Tensor<T, N, C>::ConstIterator::operator++()
{
  offset_ += stride_;
  return *this;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ConstIterator Tensor<T, N, C>::ConstIterator::operator--(int)
{
  Tensor<T, N, C>::ConstIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ConstIterator &Tensor<T, N, C>::ConstIterator::operator--()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, size_t N, template <class> class C>
bool Tensor<T, N, C>::ConstIterator::operator==(
  typename Tensor<T, N, C>::ConstIterator const &it) const
{
  if (shape_ != it.shape_) return false;
  if (stride_ != it.stride_) return false;
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ---------------------------- ReverseIterator -------------------------- */

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ReverseIterator::ReverseIterator(Tensor<T, N + 1, C> const &tensor, size_t index)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
  offset_ += stride_ * (tensor.shape_.dimensions_[index] - 1);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ReverseIterator::ReverseIterator(ReverseIterator const &it)
  : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ReverseIterator::ReverseIterator(ReverseIterator &&it)
  : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> Tensor<T, N, C>::ReverseIterator::operator*()
{
  return Tensor<T, N, C>(shape_.dimensions_, strides_, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> Tensor<T, N, C>::ReverseIterator::operator->()
{
  return Tensor<T, N, C>(shape_.dimensions_, strides_, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ReverseIterator Tensor<T, N, C>::ReverseIterator::operator++(int)
{
  Tensor<T, N, C>::ReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ReverseIterator &Tensor<T, N, C>::ReverseIterator::operator++()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ReverseIterator Tensor<T, N, C>::ReverseIterator::operator--(int)
{
  Tensor<T, N, C>::ReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ReverseIterator &Tensor<T, N, C>::ReverseIterator::operator--()
{
  offset_ += stride_;
  return *this;
}

template <typename T, size_t N, template <class> class C>
bool Tensor<T, N, C>::ReverseIterator::operator==(
  typename Tensor<T, N, C>::ReverseIterator const &it) const
{
  if (shape_ != it.shape_) return false;
  if (stride_ != it.stride_) return false;
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ------------------------- ConstReverseIterator ----------------------- */

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ConstReverseIterator::ConstReverseIterator(Tensor<T, N + 1, C> const &tensor, size_t index)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
  offset_ += stride_ * (tensor.shape_.dimensions_[index] - 1);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator const &it)
  : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator &&it)
  : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> const Tensor<T, N, C>::ConstReverseIterator::operator*()
{
  return Tensor<T, N, C>(shape_.dimensions_, strides_, offset_, 
      std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
Tensor<T, N, C> const Tensor<T, N, C>::ConstReverseIterator::operator->()
{
  return Tensor<T, N, C>(shape_.dimensions_, strides_, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ConstReverseIterator Tensor<T, N, C>::ConstReverseIterator::operator++(int)
{
  Tensor<T, N, C>::ConstReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ConstReverseIterator &Tensor<T, N, C>::ConstReverseIterator::operator++()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ConstReverseIterator Tensor<T, N, C>::ConstReverseIterator::operator--(int)
{
  Tensor<T, N, C>::ConstReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N, C>::ConstReverseIterator &Tensor<T, N, C>::ConstReverseIterator::operator--()
{
  offset_ += stride_;
  return *this;
}

template <typename T, size_t N, template <class> class C>
bool Tensor<T, N, C>::ConstReverseIterator::operator==(
    typename Tensor<T, N, C>::ConstReverseIterator const &it) const
{
  if (shape_ != it.shape_) return false;
  if (stride_ != it.stride_) return false;
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ----------------------- Iterator Construction ----------------------- */

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::Iterator Tensor<T, N, C>::begin(size_t index)
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  return typename Tensor<T, N - 1, C>::Iterator(*this, index);
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::Iterator Tensor<T, N, C>::end(size_t index)
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  typename Tensor<T, N - 1, C>::Iterator it{*this, index};
  it.offset_ += strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::Iterator Tensor<T, N, C>::begin() 
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->begin(0); 
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::Iterator Tensor<T, N, C>::end() 
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->end(0); 
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ConstIterator Tensor<T, N, C>::cbegin(size_t index) const
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  return typename Tensor<T, N - 1, C>::ConstIterator(*this, index);
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ConstIterator Tensor<T, N, C>::cend(size_t index) const
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  typename Tensor<T, N - 1, C>::ConstIterator it{*this, index};
  it.offset_ += strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ConstIterator Tensor<T, N, C>::cbegin() const
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->cbegin(0); 
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ConstIterator Tensor<T, N, C>::cend() const
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->cend(0); 
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ReverseIterator Tensor<T, N, C>::rbegin(size_t index)
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  return typename Tensor<T, N - 1, C>::ReverseIterator(*this, index);
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ReverseIterator Tensor<T, N, C>::rend(size_t index)
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  typename Tensor<T, N - 1, C>::ReverseIterator it{*this, index};
  it.offset_ -= strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ReverseIterator Tensor<T, N, C>::rbegin() 
{
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->rbegin(0); 
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ReverseIterator Tensor<T, N, C>::rend() 
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->rend(0); 
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ConstReverseIterator Tensor<T, N, C>::crbegin(size_t index) const
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  return typename Tensor<T, N - 1, C>::ConstReverseIterator(*this, index);
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ConstReverseIterator Tensor<T, N, C>::crend(size_t index) const
{
  assert((index < N) && INDEX_OUT_OF_BOUNDS);
  typename Tensor<T, N - 1, C>::ConstReverseIterator it{*this, index};
  it.offset_ -= strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ConstReverseIterator Tensor<T, N, C>::crbegin() const
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->crbegin(0); 
}

template <typename T, size_t N, template <class> class C>
typename Tensor<T, N - 1, C>::ConstReverseIterator Tensor<T, N, C>::crend() const
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
   *  to resolve compiler issues with handling 0-length arrays.
   */

  /* -------------------- typedefs -------------------- */
  typedef size_t                    size_type;
  typedef ptrdiff_t                 difference_type;
  typedef Shape<0>                  self_type;

  /* ----------------- friend classes ----------------- */

  template <typename X, size_t M, template <class> class C> friend class Tensor;
  template <typename LHS, typename RHS> friend class BinaryAdd;
  template <typename LHS, typename RHS> friend class BinarySub;
  template <typename LHS, typename RHS> friend class BinaryMul;

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

  template <typename X, size_t M, template <class> class C_>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M, C_> &tensor);

private:
};

template <>
class Indices<0> { /*@Indices<0>*/
  /** Scalar specialized 0-length Indices. This is an empty object
   *  (sizeof(Indices<0>) will return 1) to resolve compiler issues with
   *  0-length arrays.
   */
public:
  template <typename U, size_t M, template <class> class C_> friend class Tensor;
  Indices() {}
  Indices(size_t const (&)[0]) {}

  /** This method is only defined because MSVC++ requires a definition
   *  to be present, even if the method is never actually called.
   *  An assertion is immediately called if this method is invoked.
   */
  size_t &operator[](size_t) 
    { assert(0 && PANIC_ASSERTION); return *(size_t *)this;  }

  /** See operator[](size_t) */
  size_t const &operator[](size_t) const 
    { assert(0 && PANIC_ASSERTION); return *(size_t *)this; }

  /** This method is only defined because MSVC++ requires a definition
   *  to be present, even if the method is never actually called.
   *  An assertion is immediately called if this method is invoked.
   */
  bool increment(Shape<0> const&) 
    { assert(0 && PANIC_ASSERTION); return true;  }

  /** See increment(Shape<0> const&) */
  bool decrement(Shape<0> const&) 
    { assert(0 && PANIC_ASSERTION); return true;  }
};

// Scalar specialization
template <typename T, template <class> class C>
class Tensor<T, 0, C>: public Expression<Tensor<T, 0, C>> { 
  /*@Tensor<T, 0, C>*/
  /** Scalar specialization of Tensor object. The major motivation is
   *  the ability to implicitly convert to the underlying data type,
   *  allowing the Scalar to effectively be used as an ordinary value.
   */
public:
  typedef T                                       value_type;
  template <typename X> using container_type =    C<X>;
  typedef T&                                      reference_type;
  typedef T const&                                const_reference_type;
  typedef Tensor<T, 0, C>                         self_type;
  typedef Tensor<T, 0, C>                         return_type;

  /* ------------- Friend Classes ------------- */

  template <typename X, size_t M, template <class> class C_> friend class Tensor;

  /* ----------- Proxy Objects ------------ */

  class Proxy { /*@Proxy<T,0>*/
  /**
   * Proxy Tensor Object used for building tensors from reference
   * This is used only to differentiate proxy tensor Construction
   */
  public:
    template <typename U, size_t N, template <class> class C_> friend class Tensor;
    Proxy() = delete;
  private:
    Proxy(Tensor<T, 0, C> const &tensor): tensor_(tensor) {}
    Proxy(Proxy const &proxy): tensor_(proxy.tensor_) {} 
    Tensor<T, 0, C> const &tensor_;
  }; 

  /* -------------- Constructors -------------- */

  Tensor(); /**< Constructs a new Scalar whose value is zero-initialized */
  /**< Constructs a new Scalar whose value is forwarded `val` */
  explicit Tensor(value_type &&val); 
  explicit Tensor(Shape<0>); /**< Constructs a new Scalar whose value is zero-initialized */
  /**< Constructs a new Scalar and whose value copies `tensor`'s */
  Tensor(Tensor<T, 0, C> const &tensor); 
  /**< Moves data from `tensor`. `tensor` is destroyed. */
  Tensor(Tensor<T, 0, C> &&tensor);      
  /**< Constructs a Scalar who shares underlying data with proxy's underyling Scalar. */
  Tensor(typename Tensor<T, 0, C>::Proxy const &proxy); 
  /**< Evaluates `expression` and move constructs from the resulting scalar */
  template <typename NodeType,
            typename = typename std::enable_if<NodeType::rank() == 0>::type>
  Tensor(Expression<NodeType> const& expression);

  /* -------------- Destructor ------------- */

  ~Tensor() = default;

  /* ------------- Assignment ------------- */

  /** Assigns the value from `tensor`, to its element. */
  Tensor<T, 0, C> &operator=(Tensor<T, 0, C> const &tensor);

  /** Assigns the value from `tensor`, to its element. */
  template <typename X> Tensor<T, 0, C> &operator=(Tensor<X, 0, C> const &tensor);

  /* -------------- Getters -------------- */

  constexpr static size_t rank() { return 0; }  
  Shape<0> shape() const noexcept { return shape_; } 
  /** Returns the data as a reference */
  value_type &operator()() { return (*ref_)[offset_]; } 
  /** Returns the data as a const reference */
  value_type const &operator()() const { return (*ref_)[offset_]; }
  /** Creates a Scalar with the same underlying data */
  Tensor<T, 0, C> at() { return Tensor<T, 0, C>(this->ref()); }
  /** Creates a const Scalar with the same underlying data */
  Tensor<T, 0, C> const at() const { return Tensor<T, 0, C>(this->ref()); }
  /** Creates a Scalar with the same underlying data */
  Tensor<T, 0, C> slice() { return Tensor<T, 0, C>(this->ref()); }
  /** Creates a const Scalar with the same underlying data */
  Tensor<T, 0, C> const slice() const { return Tensor<T, 0, C>(this->ref()); }

  /** Used to implement iterator->, should not be used explicitly */
  Tensor *operator->() { return this; } 
  /** Used to implement const_iterator->, should not be used explicitly */
  Tensor const *operator->() const { return this; }

  /* -------------- Setters -------------- */

  template <typename X,
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type>
  Tensor<T, 0, C> &operator=(X&& elem); /**< Assigns `elem` to the underlying data */

  /* --------------- Print --------------- */
 
  template <typename X, size_t M, template <class> class C_>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M, C_> &tensor);

  template <typename X, template <class> class C_>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, 0, C_> &tensor);

  /* ------------ Equivalence ------------ */

  /** Equivalence between underlying data */
  bool operator==(Tensor<T, 0, C> const& tensor) const; 

  /** Non-equivalence between underlying data */
  bool operator!=(Tensor<T, 0, C> const& tensor) const { return !(*this == tensor); }

  /** Equivalence between underlying data */
  template <typename X>
  bool operator==(Tensor<X, 0, C> const& tensor) const;

  /** Non-equivalence between underlying data */
  template <typename X>
  bool operator!=(Tensor<X, 0, C> const& tensor) const { return !(*this == tensor); }

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

  template <typename X, typename Y, template <class> class C1, template <class> class C2>
  friend Tensor<X, 0, C1> add(
      Tensor<X, 0, C1> const &tensor_1, 
      Tensor<Y, 0, C2> const &tensor_2);

  template <typename RHS>
  Tensor<T, 0, C> &operator+=(Expression<RHS> const &rhs);

  Tensor<T, 0, C> &operator+=(T const &scalar);

  template <typename X, typename Y>
  friend Tensor<X, 0, C> subtract(Tensor<X, 0, C> const &tensor_1, Tensor<Y, 0, C> const &tensor_2);

  template <typename RHS>
  Tensor<T, 0, C> &operator-=(Expression<RHS> const &rhs);

  Tensor<T, 0, C> &operator-=(T const &scalar);

  template <typename X, typename Y>
  friend Tensor<X, 0, C> multiply(Tensor<X, 0, C> const &tensor_1, Tensor<Y, 0, C> const &tensor_2);

  template <typename RHS>
  Tensor<T, 0, C> &operator*=(Expression<RHS> const &rhs);

  Tensor<T, 0, C> &operator*=(T const &scalar);

  template <typename RHS>
  Tensor<T, 0, C> &operator/=(Expression<RHS> const &rhs);

  Tensor<T, 0, C> &operator/=(T const &scalar);

  Tensor<T, 0, C> operator-() const;

  /* ---------------- Iterator -------------- */

  class Iterator { /*@Iterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, template <class> class C_> friend class Tensor;

    /* --------------- Constructors --------------- */

    Iterator(Iterator const &it);
    Iterator(Iterator &&it);
    Tensor<T, 0, C> operator*();
    Tensor<T, 0, C> const operator*() const;
    Tensor<T, 0, C> operator->();
    Tensor<T, 0, C> const operator->() const;
    Iterator operator++(int);
    Iterator &operator++();
    Iterator operator--(int);
    Iterator &operator--();
    bool operator==(Iterator const &it) const;
    bool operator!=(Iterator const &it) const { return !(it == *this); }
  private:
    Iterator(Tensor<T, 1, C> const &tensor, size_t);

    size_t offset_;
    std::shared_ptr<C<T>> ref_;
    size_t stride_;
  };

  /* -------------- ConstIterator ------------ */

  class ConstIterator { /*@ConstIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, template <class> class C_> friend class Tensor;

    /* --------------- Constructors --------------- */

    ConstIterator(ConstIterator const &it);
    ConstIterator(ConstIterator &&it);
    Tensor<T, 0, C> const operator*();
    Tensor<T, 0, C> const operator->();
    ConstIterator operator++(int);
    ConstIterator &operator++();
    ConstIterator operator--(int);
    ConstIterator &operator--();
    bool operator==(ConstIterator const &it) const;
    bool operator!=(ConstIterator const &it) const { return !(it == *this); }
  private:
    ConstIterator(Tensor<T, 1, C> const &tensor, size_t);

    size_t offset_;
    std::shared_ptr<C<T>> ref_;
    size_t stride_;
  };

  /* ------------- ReverseIterator ----------- */

  class ReverseIterator { /*@ReverseIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, template <class> class C_> friend class Tensor;

    /* --------------- Constructors --------------- */

    ReverseIterator(ReverseIterator const &it);
    ReverseIterator(ReverseIterator &&it);
    Tensor<T, 0, C> operator*();
    Tensor<T, 0, C> operator->();
    ReverseIterator operator++(int);
    ReverseIterator &operator++();
    ReverseIterator operator--(int);
    ReverseIterator &operator--();
    bool operator==(ReverseIterator const &it) const;
    bool operator!=(ReverseIterator const &it) const { return !(it == *this); }
  private:
    ReverseIterator(Tensor<T, 1, C> const &tensor, size_t);

    size_t offset_;
    std::shared_ptr<C<T>> ref_;
    size_t stride_;
  };

  /* ----------- ConstReverseIterator --------- */

  class ConstReverseIterator { /*@ReverseIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M, template <class> class C_> friend class Tensor;

    /* --------------- Constructors --------------- */

    ConstReverseIterator(ConstReverseIterator const &it);
    ConstReverseIterator(ConstReverseIterator &&it);
    Tensor<T, 0, C> const operator*();
    Tensor<T, 0, C> const operator->();
    ConstReverseIterator operator++(int);
    ConstReverseIterator &operator++();
    ConstReverseIterator operator--(int);
    ConstReverseIterator &operator--();
    bool operator==(ConstReverseIterator const &it) const;
    bool operator!=(ConstReverseIterator const &it) const { return !(it == *this); }
  private:
    ConstReverseIterator(Tensor<T, 1, C> const &tensor, size_t);

    size_t offset_;
    std::shared_ptr<C<T>> ref_;
    size_t stride_;
  };

  /* ------------- Utility Functions ------------ */

  /** Returns an identical Tensor<T, 0, C> (copy constructed) of `*this` */
  Tensor<T, 0, C> copy() const;

  /** Returns a proxy object of `this`, used only for Tensor<T, 0, C>::Tensor(Tensor<T, 0, C>::Proxy const&) */
  typename Tensor<T, 0, C>::Proxy ref();

private:

  /* ------------------- Data ------------------- */

  Shape<0> shape_;
  size_t offset_;
  std::shared_ptr<C<T>> ref_;

  /* ------------------ Utility ----------------- */

  Tensor(size_t const *, size_t const *, size_t, std::shared_ptr<C<T>> &&ref);

};

/* ------------------- Constructors ----------------- */

template <typename T, template <class> class C>
Tensor<T, 0, C>::Tensor() : shape_(Shape<0>()), offset_(0), ref_(std::make_shared<C<T>>(1))
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
}

template <typename T, template <class> class C>
Tensor<T, 0, C>::Tensor(Shape<0>) : shape_(Shape<0>()), offset_(0), ref_(std::make_shared<C<T>>(1))
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
}

template <typename T, template <class> class C>
Tensor<T, 0, C>::Tensor(T &&val) : shape_(Shape<0>()), offset_(0)
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  ref_ = std::make_shared<C<T>>(1, val);
}

template <typename T, template <class> class C>
Tensor<T, 0, C>::Tensor(Tensor<T, 0, C> const &tensor): shape_(Shape<0>()), offset_(0),
  ref_(std::make_shared<C<T>>(1), (*tensor.ref_)[0])
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
}

template <typename T, template <class> class C>
Tensor<T, 0, C>::Tensor(Tensor<T, 0, C> &&tensor): shape_(Shape<0>()), offset_(tensor.offset_),
  ref_(std::make_shared<C<T>>(1))
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
}

template <typename T, template <class> class C>
Tensor<T, 0, C>::Tensor(typename Tensor<T, 0, C>::Proxy const &proxy)
  : shape_(Shape<0>()), offset_(proxy.tensor_.offset_), ref_(proxy.tensor_.ref_)
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
}

template <typename T, template <class> class C>
template <typename NodeType, typename>
Tensor<T, 0, C>::Tensor(Expression<NodeType> const& expression)
  : offset_(0)
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
  T result = expression.self()();
  ref_ = std::make_shared<C<T>>(1, result);
}

/* ---------------------- Assignment ---------------------- */

template <typename T, template <class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator=(Tensor<T, 0, C> const &tensor)
{
  (*ref_)[offset_] = (*tensor.ref_)[tensor.offset_];
  return *this;
}

template <typename T, template <class> class C>
template <typename X>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator=(Tensor<X, 0, C> const &tensor)
{
  (*ref_)[offset_] = (*tensor.ref_)[tensor.offset_];
  return *this;
}

template <typename T, template <class> class C>
template <typename X, typename>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator=(X&& elem)
{
  (*ref_)[offset_] = std::forward<X>(elem);
  return *this;
}

/* ------------------------ Equivalence ----------------------- */

template <typename T, template <class> class C>
bool Tensor<T, 0, C>::operator==(Tensor<T, 0, C> const &tensor) const
{
  return (*ref_)[offset_] == (*tensor.ref_)[tensor.offset_];
}

template <typename T, template <class> class C>
template <typename X>
bool Tensor<T, 0, C>::operator==(Tensor<X, 0, C> const &tensor) const
{
  return (*ref_)[offset_] == (*tensor.ref_)[tensor.offset_];
}

template <typename T, template <class> class C>
template <typename X, typename>
bool Tensor<T, 0, C>::operator==(X val) const
{
  return (*ref_)[offset_] == val;
}

/* ----------------------- Expressions ----------------------- */

template <typename X, typename Y, template <class> class C1, template <class> class C2>
Tensor<X, 0, C1> add(
    Tensor<X, 0, C1> const &tensor_1, 
    Tensor<Y, 0, C2> const &tensor_2)
{
  return Tensor<X, 0, C1>(tensor_1() + tensor_2());
}

template <typename X, typename Y,
         typename = typename std::enable_if
         <meta::LogicalAnd<!IsTensor<X>::value, !IsTensor<Y>::value>::value>>
inline X add(X const& x, Y const & y) { return x + y; }

template <typename X, template <class> class C_>
Tensor<X, 0, C_> operator+(Tensor<X, 0, C_> const &tensor, X const &scalar) 
{
  return Tensor<X, 0, C_>(tensor() + scalar);
}

template <typename X, template <class> class C_>
Tensor<X, 0, C_> operator+(X const &scalar, Tensor<X, 0, C_> const &tensor) 
{
  return Tensor<X, 0, C_>(tensor() + scalar);
}

template <typename T, template <class> class C>
template <typename RHS>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator+=(Expression<RHS> const &rhs)
{
  auto scalar = rhs.self()();
  static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
  *this = *this + scalar;
  return *this;
}

template <typename T, template <class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator+=(T const &scalar)
{
  (*ref_)[offset_] += scalar;
  return *this;
}

template <typename X, typename Y, template <class> class C_>
Tensor<X, 0, C_> subtract(Tensor<X, 0, C_> const &tensor_1, Tensor<Y, 0, C_> const &tensor_2)
{
  return Tensor<X, 0, C_>(tensor_1() - tensor_2());
}

template <typename X, typename Y,
         typename = typename std::enable_if<
          meta::LogicalAnd<!IsTensor<X>::value, !IsTensor<Y>::value>::value>>
inline X subtract(X const& x, Y const & y) { return x - y; }

template <typename X, template <class> class C_>
Tensor<X, 0, C_> operator-(Tensor<X, 0, C_> const &tensor, X const &scalar) 
{
  return Tensor<X, 0, C_>(tensor() - scalar);
}

template <typename X, template <class> class C_>
Tensor<X, 0, C_> operator-(X const &scalar, Tensor<X, 0, C_> const &tensor) 
{
  return Tensor<X, 0, C_>(scalar - tensor());
}

template <typename T, template <class> class C>
template <typename RHS>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator-=(Expression<RHS> const &rhs)
{
  auto scalar = rhs.self()();
  static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
  *this = *this - scalar;
  return *this;
}

template <typename T, template <class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator-=(T const &scalar)
{
  (*ref_)[offset_] -= scalar;
  return *this;
}

/** Directly overload operator* for scalar multiplication so tensor multiplication expressions
 *  don't have to deal with it. 
 */
template <typename X, typename Y, template <class> class C_>
inline Tensor<X, 0, C_> operator*(Tensor<X, 0, C_> const &tensor_1, Tensor<Y, 0, C_> const &tensor_2)
{
  return Tensor<X, 0, C_>(tensor_1() * tensor_2());
}

template <typename X, template <class> class C_>
inline Tensor<X, 0, C_> operator*(Tensor<X, 0, C_> const &tensor, X const &scalar) 
{
  return Tensor<X, 0, C_>(tensor() * scalar);
}

template <typename X, template <class> class C_>
inline Tensor<X, 0, C_> operator*(X const &scalar, Tensor<X, 0, C_> const &tensor) 
{
  return Tensor<X, 0, C_>(tensor() * scalar);
}

template <typename T, template <class> class C>
template <typename RHS>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator*=(Expression<RHS> const &rhs)
{
  auto scalar = rhs.self()();
  static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
  *this = *this * scalar;
  return *this;
}

template <typename T, template <class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator*=(T const &scalar)
{
  (*ref_)[offset_] *= scalar;
  return *this;
}

// Directly overload operator/
template <typename X, typename Y, template <class> class C_>
inline Tensor<X, 0, C_> operator/(Tensor<X, 0, C_> const &tensor_1, Tensor<Y, 0, C_> const &tensor_2)
{
  return Tensor<X, 0, C_>(tensor_1() / tensor_2());
}

template <typename X, template <class> class C_>
Tensor<X, 0, C_> operator/(Tensor<X, 0, C_> const &tensor, X const &scalar) 
{
  return Tensor<X, 0, C_>(tensor() / scalar);
}

template <typename X, template <class> class C_>
Tensor<X, 0, C_> operator/(X const &scalar, Tensor<X, 0, C_> const &tensor) 
{
  return Tensor<X, 0, C_>(tensor() / scalar);
}


template <typename T, template <class> class C>
template <typename RHS>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator/=(Expression<RHS> const &rhs)
{
  auto scalar = rhs.self()();
  static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
  *this = *this / scalar;
  return *this;
}

template <typename T, template <class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator/=(T const &scalar)
{
  (*ref_)[offset_] /= scalar;
  return *this;
}

template <typename T, template <class> class C>
Tensor<T, 0, C> Tensor<T, 0, C>::operator-() const
{
  return Tensor<T, 0, C>(-(*ref_)[offset_]);
}

/* ------------------------ Utility --------------------------- */

template <typename T, template <class> class C>
Tensor<T, 0, C>::Tensor(size_t const *, size_t const *, size_t offset, std::shared_ptr<C<T>> &&ref)
  : offset_(offset), ref_(std::move(ref))
{
#ifdef _TEST
 ++eDebugConstructorCounter;
#endif
}

/* ------------------------ Overloads ------------------------ */

template <typename X, template <class> class C_>
std::ostream &operator<<(std::ostream &os, const Tensor<X, 0, C_> &tensor)
{
  os << (*tensor.ref_)[tensor.offset_];
  return os;
}

/* ------------------- Useful Functions ---------------------- */

template <typename T, template <class> class C>
Tensor<T, 0, C> Tensor<T, 0, C>::copy() const
{
  return Tensor<T, 0, C>(nullptr, nullptr, 0, 
      std::make_shared<C<T>>(1, (*ref_)[offset_]));
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::Proxy Tensor<T, 0, C>::ref() 
{
  return Proxy(*this);
}

/* ---------------------- Iterators ------------------------- */

template <typename T, template <class> class C>
Tensor<T, 0, C>::Iterator::Iterator(Tensor<T, 1, C> const &tensor, size_t)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{}

template <typename T, template <class> class C>
Tensor<T, 0, C>::Iterator::Iterator(Iterator const &it)
  : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T, template <class> class C>
Tensor<T, 0, C>::Iterator::Iterator(Iterator &&it)
  : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T, template <class> class C>
Tensor<T, 0, C> Tensor<T, 0, C>::Iterator::operator*()
{
  return Tensor<T, 0, C>(nullptr, nullptr, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, template <class> class C>
Tensor<T, 0, C> Tensor<T, 0, C>::Iterator::operator->()
{
  return Tensor<T, 0, C>(nullptr, nullptr, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::Iterator Tensor<T, 0, C>::Iterator::operator++(int)
{
  Tensor<T, 0, C>::Iterator it {*this};
  ++(*this);
  return it;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::Iterator &Tensor<T, 0, C>::Iterator::operator++()
{
  offset_ += stride_;
  return *this;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::Iterator Tensor<T, 0, C>::Iterator::operator--(int)
{
  Tensor<T, 0, C>::Iterator it {*this};
  --(*this);
  return it;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::Iterator &Tensor<T, 0, C>::Iterator::operator--()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, template <class> class C>
bool Tensor<T, 0, C>::Iterator::operator==(
  typename Tensor<T, 0, C>::Iterator const &it) const
{
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ------------------- ConstIterators ---------------------- */

template <typename T, template <class> class C>
Tensor<T, 0, C>::ConstIterator::ConstIterator(Tensor<T, 1, C> const &tensor, size_t)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{}

template <typename T, template <class> class C>
Tensor<T, 0, C>::ConstIterator::ConstIterator(ConstIterator const &it)
  : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T, template <class> class C>
Tensor<T, 0, C>::ConstIterator::ConstIterator(ConstIterator &&it)
  : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T, template <class> class C>
Tensor<T, 0, C> const Tensor<T, 0, C>::ConstIterator::operator*()
{
  return Tensor<T, 0, C>(nullptr, nullptr, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, template <class> class C>
Tensor<T, 0, C> const Tensor<T, 0, C>::ConstIterator::operator->()
{
  return Tensor<T, 0, C>(nullptr, nullptr, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ConstIterator Tensor<T, 0, C>::ConstIterator::operator++(int)
{
  Tensor<T, 0, C>::ConstIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ConstIterator &Tensor<T, 0, C>::ConstIterator::operator++()
{
  offset_ += stride_;
  return *this;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ConstIterator Tensor<T, 0, C>::ConstIterator::operator--(int)
{
  Tensor<T, 0, C>::ConstIterator it {*this};
  --(*this);
  return it;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ConstIterator &Tensor<T, 0, C>::ConstIterator::operator--()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, template <class> class C>
bool Tensor<T, 0, C>::ConstIterator::operator==(
  typename Tensor<T, 0, C>::ConstIterator const &it) const
{
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* -------------------- ReverseIterator ---------------------- */

template <typename T, template <class> class C>
Tensor<T, 0, C>::ReverseIterator::ReverseIterator(Tensor<T, 1, C> const &tensor, size_t)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{
  offset_ += stride_ * (tensor.shape_.dimensions_[0] - 1);
}

template <typename T, template <class> class C>
Tensor<T, 0, C>::ReverseIterator::ReverseIterator(ReverseIterator const &it)
  : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T, template <class> class C>
Tensor<T, 0, C>::ReverseIterator::ReverseIterator(ReverseIterator &&it)
  : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T, template <class> class C>
Tensor<T, 0, C> Tensor<T, 0, C>::ReverseIterator::operator*()
{
  return Tensor<T, 0, C>(nullptr, nullptr, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, template <class> class C>
Tensor<T, 0, C> Tensor<T, 0, C>::ReverseIterator::operator->()
{
  return Tensor<T, 0, C>(nullptr, nullptr, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ReverseIterator Tensor<T, 0, C>::ReverseIterator::operator++(int)
{
  Tensor<T, 0, C>::ReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ReverseIterator &Tensor<T, 0, C>::ReverseIterator::operator++()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ReverseIterator Tensor<T, 0, C>::ReverseIterator::operator--(int)
{
  Tensor<T, 0, C>::ReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ReverseIterator &Tensor<T, 0, C>::ReverseIterator::operator--()
{
  offset_ += stride_;
  return *this;
}

template <typename T, template <class> class C>
bool Tensor<T, 0, C>::ReverseIterator::operator==(
  typename Tensor<T, 0, C>::ReverseIterator const &it) const
{
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ----------------- ConstReverseIterator ------------------- */

template <typename T, template <class> class C>
Tensor<T, 0, C>::ConstReverseIterator::ConstReverseIterator(Tensor<T, 1, C> const &tensor, size_t)
  : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{
  offset_ += stride_ * (tensor.shape_.dimensions_[0] - 1);
}

template <typename T, template <class> class C>
Tensor<T, 0, C>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator const &it)
  : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T, template <class> class C>
Tensor<T, 0, C>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator &&it)
  : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T, template <class> class C>
Tensor<T, 0, C> const Tensor<T, 0, C>::ConstReverseIterator::operator*()
{
  return Tensor<T, 0, C>(nullptr, nullptr, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, template <class> class C>
Tensor<T, 0, C> const Tensor<T, 0, C>::ConstReverseIterator::operator->()
{
  return Tensor<T, 0, C>(nullptr, nullptr, offset_, std::shared_ptr<C<T>>(ref_));
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ConstReverseIterator Tensor<T, 0, C>::ConstReverseIterator::operator++(int)
{
  Tensor<T, 0, C>::ConstReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ConstReverseIterator &Tensor<T, 0, C>::ConstReverseIterator::operator++()
{
  offset_ -= stride_;
  return *this;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ConstReverseIterator Tensor<T, 0, C>::ConstReverseIterator::operator--(int)
{
  Tensor<T, 0, C>::ConstReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, template <class> class C>
typename Tensor<T, 0, C>::ConstReverseIterator &Tensor<T, 0, C>::ConstReverseIterator::operator--()
{
  offset_ += stride_;
  return *this;
}

template <typename T, template <class> class C>
bool Tensor<T, 0, C>::ConstReverseIterator::operator==(
  typename Tensor<T, 0, C>::ConstReverseIterator const &it) const
{
  if (ref_.get() != it.ref_.get()) return false;
  return offset_ == it.offset_;
}

/* ---------------------- Expressions ------------------------ */

template <typename LHS, typename RHS>
class BinaryAdd: public Expression<BinaryAdd<LHS, RHS>> { 
/*@BinaryAdd<LHS, RHS>*/
public:

  /* ---------------- typedefs --------------- */

  typedef typename LHS::value_type        value_type;
  template <typename X>
  using container_type = typename         LHS::template container_type<X>;
  constexpr static size_t rank()          { return LHS::rank(); } 
  typedef BinaryAdd                       self_type;
  typedef typename LHS::return_type       return_type;

  /* ---------------- Friends ---------------- */
  
  template <typename LHS_, typename RHS_>
  friend BinaryAdd<LHS_, RHS_> 
    operator+(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

  /* ---------------- Getters ----------------- */

  size_t dimension(size_t index) const { return lhs_.dimension(index); }
  Shape<LHS::rank()> const &shape() const { return lhs_.shape(); }

  template <typename... Args>
  auto operator()(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type;

  template <typename... Args>
  auto at(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type;

  template <size_t M>
  auto operator[](Indices<M> const &indices) const 
    -> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type;

  template <size_t... Slices, typename... Args>
  auto slice(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(args...))>::type;

  template <size_t... Slices, size_t M>
  auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(indices))>::type;

private:

  /* -------------- Constructors -------------- */

  BinaryAdd(LHS const &lhs, RHS const &rhs);
  BinaryAdd(BinaryAdd<LHS, RHS> const&) = default;

  /* ------------------ Data ------------------ */

  LHS const &lhs_;
  RHS const &rhs_;
};

template <typename LHS, typename RHS>
BinaryAdd<LHS, RHS>::BinaryAdd(LHS const &lhs, RHS const &rhs)
  : lhs_(lhs), rhs_(rhs) {}

template <typename LHS, typename RHS>
template <typename... Args>
auto BinaryAdd<LHS, RHS>::operator()(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type
{
  static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
  return add(ValueAsTensor<LHS>(lhs_)(args...),
             ValueAsTensor<RHS>(rhs_)(args...));
}

template <typename LHS, typename RHS>
template <typename... Args>
auto BinaryAdd<LHS, RHS>::at(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type
{
  static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
  return Tensor<value_type, rank() - sizeof...(args), container_type>(
            add(ValueAsTensor<LHS>(lhs_)(args...),
                ValueAsTensor<RHS>(rhs_)(args...)));
}

template <typename LHS, typename RHS>
template <size_t M>
auto BinaryAdd<LHS, RHS>::operator[](Indices<M> const &indices) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type
{
  static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
  return add(ValueAsTensor<LHS>(lhs_)[indices],
             ValueAsTensor<RHS>(rhs_)[indices]);
}

template <typename LHS, typename RHS>
template <size_t... Slices, typename... Args>
auto BinaryAdd<LHS, RHS>::slice(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(args...))>::type
{
  static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
  return add(ValueAsTensor<LHS>(lhs_).template slice<Slices...>(args...),
             ValueAsTensor<RHS>(rhs_).template slice<Slices...>(args...));
}

template <typename LHS, typename RHS>
template <size_t... Slices, size_t M>
auto BinaryAdd<LHS, RHS>::slice(Indices<M> const &indices) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(indices))>::type
{
  static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
  return add(ValueAsTensor<LHS>(lhs_).template slice<Slices...>(indices),
             ValueAsTensor<RHS>(rhs_).template slice<Slices...>(indices));
}

template <typename LHS, typename RHS>
BinaryAdd<LHS, RHS> operator+(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
  return BinaryAdd<LHS, RHS>(lhs.self(), rhs.self());
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

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os, BinaryAdd<LHS, RHS> const &binary_add)
{
  os << binary_add();
  return os;
}

/* ----------------------------------------- */

template <typename LHS, typename RHS>
class BinarySub: public Expression<BinarySub<LHS, RHS>> { 
/*@BinarySub<LHS, RHS>*/
public:
  /* ---------------- typedefs --------------- */

  typedef typename LHS::value_type        value_type;
  template <typename X>
  using container_type = typename         LHS::template container_type<X>;
  typedef BinarySub                       self_type;
  constexpr static size_t rank()          { return LHS::rank(); }
  typedef typename LHS::return_type       return_type;

  /* ---------------- Friend ----------------- */

  template <typename LHS_, typename RHS_>
  friend BinarySub<LHS_, RHS_> 
    operator-(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

  /* ---------------- Getters ----------------- */

  size_t dimension(size_t index) const { return lhs_.dimension(index); }
  Shape<LHS::rank()> const &shape() const { return lhs_.shape(); }

  template <typename... Args>
  auto operator()(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type;

  template <typename... Args>
  auto at(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type;

  template <size_t M>
  auto operator[](Indices<M> const &indices) const 
    -> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type;

  template <size_t... Slices, typename... Args>
  auto slice(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(args...))>::type;

  template <size_t... Slices, size_t M>
  auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(indices))>::type;

private:

  /* -------------- Constructors -------------- */

  BinarySub(LHS const &lhs, RHS const &rhs);
  BinarySub(BinarySub<LHS, RHS> const&) = default;

  /* ------------------ Data ------------------ */

  LHS const &lhs_;
  RHS const &rhs_;
};

template <typename LHS, typename RHS>
BinarySub<LHS, RHS>::BinarySub(LHS const &lhs, RHS const &rhs)
  : lhs_(lhs), rhs_(rhs) {}

template <typename LHS, typename RHS>
template <typename... Args>
auto BinarySub<LHS, RHS>::operator()(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type
{
  static_assert(rank() >= sizeof...(Args), RANK_OUT_OF_BOUNDS);
  return subtract(
      ValueAsTensor<LHS>(lhs_)(args...),
      ValueAsTensor<RHS>(rhs_)(args...));
}

template <typename LHS, typename RHS>
template <size_t M>
auto BinarySub<LHS, RHS>::operator[](Indices<M> const &indices) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type
{
  static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
  return subtract(ValueAsTensor<LHS>(lhs_)[indices],
                  ValueAsTensor<RHS>(rhs_)[indices]);
}

template <typename LHS, typename RHS>
template <typename... Args>
auto BinarySub<LHS, RHS>::at(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type
{
  static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
  return Tensor<value_type, rank() - sizeof...(args), container_type>(
            subtract(ValueAsTensor<LHS>(lhs_)(args...),
                     ValueAsTensor<RHS>(rhs_)(args...)));
}

template <typename LHS, typename RHS>
template <size_t... Slices, typename... Args>
auto BinarySub<LHS, RHS>::slice(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(args...))>::type
{
  static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
  return subtract(ValueAsTensor<LHS>(lhs_).template slice<Slices...>(args...),
                  ValueAsTensor<RHS>(rhs_).template slice<Slices...>(args...));
}

template <typename LHS, typename RHS>
template <size_t... Slices, size_t M>
auto BinarySub<LHS, RHS>::slice(Indices<M> const &indices) const
  -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(indices))>::type
{
  static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
  return subtract(ValueAsTensor<LHS>(lhs_).template slice<Slices...>(indices),
                   ValueAsTensor<RHS>(rhs_).template slice<Slices...>(indices));
}

template <typename LHS, typename RHS>
BinarySub<LHS, RHS> operator-(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
  return BinarySub<LHS, RHS>(lhs.self(), rhs.self());
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

template <typename LHS, typename RHS>
std::ostream &operator<<(std::ostream &os, BinarySub<LHS, RHS> const &binary_sub)
{
  os << binary_sub();
  return os;
}

/**
 * Multiplies the inner dimensions
 * i.e. 3x4x5 * 5x4x3 produces a 3x4x4x3 tensor
 */

template <typename LHS, typename RHS>
class BinaryMul: public Expression<BinaryMul<LHS, RHS>> { 
//   @BinaryMul
public:

  /* ---------------- typedefs --------------- */

  typedef typename LHS::value_type                      value_type;
  template <typename X>
  using container_type = typename                       LHS::template container_type<X>;
  typedef BinaryMul                                     self_type;
  constexpr static size_t rank()                        { return LHS::rank() + RHS::rank() - 2; }
  typedef 
  Tensor<value_type, self_type::rank(), container_type> return_type;

  /* ---------------- Friend ----------------- */
  
  template <typename LHS_, typename RHS_>
  friend BinaryMul<LHS_, RHS_> 
    operator*(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

  /* ---------------- Getters ----------------- */

  Shape<self_type::rank()> const &shape() const { return shape_; }
  size_t dimension(size_t index) const; 

  template <typename... Args>
  auto operator()(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<return_type const>()(args...))>::type;

  template <typename... Args>
  auto at(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<return_type const>().at(args...))>::type;
  
  template <size_t M>
  auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<return_type const>()[indices])>::type;

  template <size_t... Slices, typename... Args>
  auto slice(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<return_type const>().slice(args...))>::type;

  template <size_t... Slices, size_t M>
  auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<return_type const>().slice(indices))>::type;

private:

  /* -------------- Constructors -------------- */

  BinaryMul(LHS const &lhs, RHS const &rhs);
  BinaryMul(BinaryMul<LHS, RHS> const&) = default;

  /* ---------------- Utility ---------------- */

  // Accepts the slice sequences and forwards to `multiply()`. Ideally, this
  // would be a lambda in `slice()` but C++11 has no support for templated lambdas 
  template <size_t... Slices1, size_t... Slices2, typename... Args>
  auto pSliceSequences(meta::Sequence<Slices1...>, meta::Sequence<Slices2...>, Args... args) const 
  -> typename std::remove_reference<decltype(std::declval<return_type const>().slice(args...))>::type;

  template <size_t... Slices1, size_t... Slices2, size_t M>
  auto pSliceSequences(meta::Sequence<Slices1...>, meta::Sequence<Slices2...>, Indices<M> const &indices) const 
  -> typename std::remove_reference<decltype(std::declval<return_type const>().slice(indices))>::type;

  /* ------------------ Data ------------------ */

  // cache so that `shape()` can return a const reference
  Shape<self_type::rank()> shape_;
  LHS const &lhs_;
  RHS const &rhs_;

};

template <typename LHS, typename RHS>
BinaryMul<LHS, RHS>::BinaryMul(LHS const &lhs, RHS const &rhs)
  : lhs_(lhs), rhs_(rhs)
{
  static_assert(LHS::rank(), PANIC_ASSERTION);
  static_assert(RHS::rank(), PANIC_ASSERTION);
  constexpr size_t M1 = LHS::rank();
  constexpr size_t M2 = RHS::rank();
  // shape <- lhs.shape[:-1] :: rhs.shape[1:]
  for (size_t i = 0; i < M1 - 1; ++i)
    shape_[i] = lhs.dimension(i);
  for (size_t i = 0; i < M2 - 1; ++i)
    shape_[M1 - 1 + i] = rhs.dimension(i + 1);
}

template <typename LHS, typename RHS>
size_t BinaryMul<LHS, RHS>::dimension(size_t index) const
{
  assert(index < self_type::rank() && INDEX_OUT_OF_BOUNDS);
  return shape_[index];
}

template <typename LHS, typename RHS>
template <typename... Args>
auto BinaryMul<LHS, RHS>::operator()(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<return_type const>()(args...))>::type
{
  static_assert(rank() >= sizeof...(Args), RANK_OUT_OF_BOUNDS);
  constexpr size_t left = meta::Min(LHS::rank() - 1, sizeof...(args));
  constexpr size_t right = meta::NonZeroDifference(sizeof...(args), LHS::rank() - 1);
  meta::FillArgs<left, right + left> seperate_args(args...);
  return multiply(ValueAsTensor<LHS>(lhs_)[Indices<left>(seperate_args.array1)],
                  ValueAsTensor<RHS>(rhs_).template slice<0>(Indices<right>(seperate_args.array2)));
}

template <typename LHS, typename RHS>
template <typename... Args>
auto BinaryMul<LHS, RHS>::at(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<return_type const>().at(args...))>::type
{
  static_assert(rank() >= sizeof...(Args), RANK_OUT_OF_BOUNDS);
  constexpr size_t left = meta::Min(LHS::rank() - 1, sizeof...(args));
  constexpr size_t right = meta::NonZeroDifference(sizeof...(args), LHS::rank() - 1);
  meta::FillArgs<left, right + left> seperate(args...);
  return Tensor<value_type, self_type::rank() - sizeof...(Args), container_type>(
      multiply(ValueAsTensor<LHS>(lhs_)[Indices<left>(seperate.array1)],
               ValueAsTensor<RHS>(rhs_).template slice<0>(Indices<right>(seperate.array2))));
}

template <typename LHS, typename RHS>
template <size_t M>
auto BinaryMul<LHS, RHS>::operator[](Indices<M> const &indices) const
  -> typename std::remove_reference<decltype(std::declval<return_type const>()[indices])>::type
{
  static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
  constexpr size_t left = meta::Min(LHS::rank() - 1, M);
  constexpr size_t right = meta::NonZeroDifference(M, LHS::rank() - 1);
  size_t array1[left], array2[right];
  for (size_t i = 0; i < left; ++i) array1[i] = indices[i];
  for (size_t i = 0; i < right; ++i) array2[i]= indices[i + left];
  return multiply(ValueAsTensor<LHS>(lhs_)[Indices<left>(array1)],
                  ValueAsTensor<RHS>(rhs_).template slice<0>(Indices<right>(array2)));
}

template <typename LHS, typename RHS>
template <size_t... Slices, typename... Args>
auto BinaryMul<LHS, RHS>::slice(Args... args) const
  -> typename std::remove_reference<decltype(std::declval<return_type const>().slice(args...))>::type
{
  using namespace meta; // WARNING -- using namespace
  static_assert(IsIncreasingSequence<Slices...>::value, SLICE_INDICES_DESCENDING);
  constexpr size_t thresh_hold = CountLTMax<LHS::rank(), Slices...>::value;
  auto sequence1 = typename MakeSequence1<thresh_hold, Slices...>::sequence{};
  // FIXME -- Very difficult to read
  auto sequence2 = typename Append<0, typename SequenceTransformer<
        typename MakeSequence2<thresh_hold, Slices...>::sequence,
        SequenceOffset, LHS::rank() - 1>::sequence>::sequence{};
  return pSliceSequences(sequence1, sequence2, args...);
}

template <typename LHS, typename RHS>
template <size_t... Slices, size_t M>
auto BinaryMul<LHS, RHS>::slice(Indices<M> const &indices) const
  -> typename std::remove_reference<decltype(std::declval<return_type const>().slice(indices))>::type
{
  using namespace meta; // WARNING -- using namespace
  static_assert(IsIncreasingSequence<Slices...>::value, SLICE_INDICES_DESCENDING);
  constexpr size_t thresh_hold = CountLTMax<LHS::rank(), Slices...>::value;
  auto sequence1 = typename MakeSequence1<thresh_hold, Slices...>::sequence{};
  // FIXME -- Very difficult to read
  auto sequence2 = typename Append<0, typename SequenceTransformer<
        typename MakeSequence2<thresh_hold, Slices...>::sequence,
        SequenceOffset, LHS::rank() - 1>::sequence>::sequence{};
  return pSliceSequences(sequence1, sequence2, indices);
}

template <typename LHS, typename RHS>
BinaryMul<LHS, RHS> operator*(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
  return BinaryMul<LHS, RHS>(lhs.self(), rhs.self());
}

/* ------------------------------ Utility ------------------------------ */

template <typename LHS, typename RHS>
template <size_t... Slices1, size_t... Slices2, typename... Args>
auto BinaryMul<LHS, RHS>::pSliceSequences(meta::Sequence<Slices1...>, 
    meta::Sequence<Slices2...>, Args... args) const 
  -> typename std::remove_reference<decltype(std::declval<return_type const>().slice(args...))>::type
{
  constexpr size_t left = meta::Min(LHS::rank() - 1, sizeof...(args));
  constexpr size_t right = meta::NonZeroDifference(sizeof...(args), LHS::rank() - 1);
  meta::FillArgs<left, right + left> seperate(args...);
  return multiply(ValueAsTensor<LHS>(lhs_).template slice<Slices1...>(Indices<left>(seperate.array1)),
                  ValueAsTensor<RHS>(rhs_).template slice<Slices2...>(Indices<right>(seperate.array2)));
}

template <typename LHS, typename RHS>
template <size_t... Slices1, size_t... Slices2, size_t M>
auto BinaryMul<LHS, RHS>::pSliceSequences(meta::Sequence<Slices1...>, 
    meta::Sequence<Slices2...>, Indices<M> const &indices) const 
  -> typename std::remove_reference<decltype(std::declval<return_type const>().slice(indices))>::type
{
  constexpr size_t left = meta::Min(LHS::rank() - 1, M);
  constexpr size_t right = meta::NonZeroDifference(M, LHS::rank() - 1);
  size_t array1[left], array2[right];
  for (size_t i = 0; i < left; ++i) array1[i] = indices[i];
  for (size_t i = 0; i < right; ++i) array2[i]= indices[i + left];
  return multiply(ValueAsTensor<LHS>(lhs_).template slice<Slices1...>(Indices<left>(array1)),
                  ValueAsTensor<RHS>(rhs_).template slice<Slices2...>(Indices<right>(array2)));
}

} // namespace tensor

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
#undef SHAPE_MISMATCH
#undef EXPECTED_SCALAR
#undef DIMENSION_MISMATCH
#undef INNER_DIMENSION_MISMATCH
#undef ELEMENT_COUNT_MISMATCH
#undef SCALAR_TENSOR_MULT
#undef PANIC_ASSERTION

/* ----------------------------------------------- */


#endif // TENSORS_H_
