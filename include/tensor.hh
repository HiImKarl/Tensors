#ifndef TENSOR_H_
#define TENSOR_H_

#include <cstddef>
#include <algorithm>
#include <utility>
#include <cmath>
#include <type_traits>
#include <tuple>
#include <initializer_list>
#include <numeric>
#include <functional>
#include <memory>
#include <cassert>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <iterator>

#ifdef _ENABLE_OPENCL
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif // defined __GNUC__

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/cl2.hpp>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif // defined __GNUC__

#endif // defined _ENABLE_OPENCL

#ifdef _TEST
extern int eDebugConstructorCounter;
#endif // defined _TEST

#ifndef NDEBUG

#include <iostream>

#define PRINT(x) std::cout << x << '\n';
#define PRINTV(x) std::cout << #x << ": " << x << '\n';
#define ARRAY_SIZE(x) sizeof(x)/sizeof(x[0])
#define PRINT_ARRAY1(x) \
  std::cout << #x << ": "; \
  for (size_t i = 0; i < ARRAY_SIZE(x); ++i) std::cout << x[i] << ' '; \
  std::cout << '\n';
#define PRINT_ARRAY2(x, n) \
  std::cout << #x << ": "; \
  for (size_t i = 0; i < n; ++i) std::cout << x[i] << ' '; \
  std::cout << '\n';
#define GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define PRINT_ARRAY(...) \
  GET_MACRO(__VA_ARGS__, PRINT_ARRAY1, PRINT_ARRAY2)(__VA_ARGS__)

#endif // not defined NDEBUG

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
#define EXPECTING_TENSOR \
  "Expecting argument to be an any-rank Tensor"
#define EXPECTING_SCALAR \
  "Expecting argument to be a Scalar"
#define EXPECTING_EXPRESSION \
  "Expecting argument to be an Expression"

// Out of bounds
#define DIMENSION_INVALID \
  "Attempt To Access Invalid Dimension"
#define RANK_OUT_OF_BOUNDS \
  "Rank Out Of Bounds"
#define INDEX_OUT_OF_BOUNDS \
  "Index Out Of Bounds"
#define TOO_MANY_INDICES \
  "Too Many Indices Provided"
#define FLATTEN_INDICES_ERROR \
  "Attempt to flatten failed -- Error with Indices Provided"

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
#define INVALID_ITERATOR_COMPARISON \
  "Cannot Perform Iterator Comparison -- Underlying Data Does Not Match"

// Arithmetic Operations
#define RANK_MISMATCH \
  "Expecting same rank()"
#define SHAPE_MISMATCH \
  "Expecting same rank()"
#define EXPECTED_SCALAR \
  "Expecting Scalar"
#define EXPECTED_TENSOR \
  "Expecting Tensor"
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
#define NO_EXPRESSIONS_PROVIDED \
  "This method requires at least one tensor as an argument"
#define SLICING_MULTIPLICATION_AXIS \
  "Attempt to slice a multiplication expression along one of its axes"

// Broadcasting
#define TENSOR_RANK_MISMATCH \
  "Attempt to broadcast failed -- mismatched tensor rank"

// OpenCL
#define OPENCL_NO_PLATFORMS \
  "No OpenCL platforms found -- consult your device vendor for up-to-date OpenCL drivers"
#define OPENCL_NO_DEVICES \
  "No OpenCL devices found -- consult your device vendor for up-to-date OpenCL drivers"
#define OPENCL_BUFFER_ERROR \
  "OpenCL -- Buffer Error"
#define OPENCL_KERNEL_ERROR \
  "OpenCL -- Kernel Error"
#define OPENCL_ARITY_ERROR \
  "OpenCL -- Number of arguments provided to OpenCL function is incorrect"
#define OPENCL_REDUCTION_SIZE_ERROR \
  "OpenCL -- OpenCL reduction can only take one argument"

/* ---------------- Debug Messages --------------- */

#define PANIC_ASSERTION \
  "This assertion should never fire... the developer messed up"

/* -------------------- Macros ------------------- */

// MSVC prioritizes parsing {} as an initializer_list
#ifdef _MSC_VER
#define VARDIAC_MAP(EXPR) \
  (void)std::initializer_list<int>{(EXPR, 0)...};
#else
#define VARDIAC_MAP(EXPR) \
  (void)(int[]){(EXPR, 0)...};
#endif // defined _MSC_VER

namespace tensor {

/* --------------- Forward Declerations --------------- */

namespace data {

template<typename T>
class Array;

template<typename T>
class HashMap;

} // namespace data

template<typename NodeType>
struct Expression;

template<size_t N>
class Shape;

template<>
class Shape<0>;

template<size_t N>
class Indices;

template<>
class Indices<0>;

template<typename T, size_t N, template<class> class C = data::Array>
class Tensor;

template<typename LHS, typename RHS>
class BinaryAddExpr;

template<typename LHS, typename RHS>
class BinarySubExpr;

template<size_t I1, size_t I2, typename LHS, typename RHS>
class BinaryMulExpr;

template<typename LHS, typename RHS>
class BinaryHadExpr;

template<typename Function, typename... Exprs>
class MapExpr;

template<typename T, typename Function, typename... Exprs>
class ReduceExpr;

template<typename RHS>
class UnaryNegExpr;

namespace meta {

template<typename T>
struct RemoveCVRef;

} // namespace meta

namespace opencl {

template<typename NodeType>
class Model;

} // namespace opencl

/* ----------------- Type Definitions ---------------- */

template<typename T, template<class> class C = data::Array>
using Scalar = Tensor<T, 0, C>;
template<typename T, template<class> class C = data::Array>
using Vector = Tensor<T, 1, C>;
template<typename T, template<class> class C = data::Array>
using Matrix = Tensor<T, 2, C>;

template<typename T, size_t N>
using SparseTensor = Tensor<T, N, data::HashMap>;
template<typename T>
using SparseMatrix = Tensor<T, 2, data::HashMap>;
template<typename T>
using SparseVector = Tensor<T, 1, data::HashMap>;
template<typename T>
using SparseScalar = Tensor<T, 0, data::HashMap>;

/* -------------- Tensor Meta-Patterns --------------- */

/** Boolean member `value` is true if T is a non-cv qualified, non-ref
 *  Tensor of any rank, false o.w.
 */
template<typename T>
struct IsNonCVRefTensor : std::false_type {
};

/** Tensor specialization of IsNonCVRefTensor, boolean enum `value` is true */
template<typename T, size_t N, template<class> class C>
struct IsNonCVRefTensor<Tensor<T, N, C>> : std::true_type {
};

/** Boolean member `value` is true if T is a Tensor of any rank,
 *  ignoring cv, ref qualification, false o.w.
 */
template<typename T>
struct IsTensor {
    enum : bool {
        value = IsNonCVRefTensor<typename meta::RemoveCVRef<T>::type>::value
    };
};

/** Boolean member `value` is true if T is a non-cv qualified, non-ref
 *  0-rank Tensor, false o.w.
 */
template<typename T>
struct IsNonCVRefScalarTensor {
    static bool const value = true;
};

/** Scalar specialization of IsScalar, Boolean member `value` is true */
template<typename T, template<class> class C>
struct IsNonCVRefScalarTensor<Tensor<T, 0, C>> {
    static bool const value = true;
};

/** Enum `value` is true if T is a 0-rank Tensor, ingoring
 *  CV, ref qualification, false o.w.
 */
template<typename T>
struct IsScalar {
    enum : bool {
        value = IsNonCVRefScalarTensor<typename meta::RemoveCVRef<T>::type>::value
    };
};

/** Boolean member `value` is true if T is a non-cv qualified, non-ref
 *  Expression of any NodeType, false o.w.
 */
template<typename T>
struct IsNonCVRefExpression : std::false_type {
};

template<typename T, size_t N, template<class> class C>
struct IsNonCVRefExpression<Tensor<T, N, C>> : std::true_type {
};

template<typename LHS, typename RHS>
struct IsNonCVRefExpression<BinaryAddExpr<LHS, RHS>> : std::true_type {
};

template<typename LHS, typename RHS>
struct IsNonCVRefExpression<BinarySubExpr<LHS, RHS>> : std::true_type {
};

template<size_t I1, size_t I2, typename LHS, typename RHS>
struct IsNonCVRefExpression<BinaryMulExpr<I1, I2, LHS, RHS>> : std::true_type {
};

template<typename LHS, typename RHS>
struct IsNonCVRefExpression<BinaryHadExpr<LHS, RHS>> : std::true_type {
};

template<typename Function, typename... Exprs>
struct IsNonCVRefExpression<MapExpr<Function, Exprs...>> : std::true_type {
};

template<typename T, typename Function, typename... Exprs>
struct IsNonCVRefExpression<ReduceExpr<T, Function, Exprs...>> : std::true_type {
};

template<typename RHS>
struct IsNonCVRefExpression<UnaryNegExpr<RHS>> : std::true_type {
};

/** Boolean member `value` is true if T is an Expression,
 *  regardless of cv/ref qualification, false o.w.
 */
template<typename T>
struct IsExpression {
    enum : bool {
        value = IsNonCVRefExpression<
                typename std::remove_reference<typename std::remove_cv<T>::type>::type>::value
    };
};

/** Provides `value` equal to the rank() of `type`, 0
 *  if not a non-cv qualified, Tensor type (
 *  are considered to be rank() 0 in this context)
 */
template<typename T>
struct Rank {
    enum : size_t {
        value = 0
    };
};

/** Provides `value` equal to the rank() of `type`, 0
 *  if not a tensor::Tensor type (C mutli-dimensional arrays
 *  are considered to be rank() 0 in this context)
 */
template<typename T, size_t N, template<class> class C>
struct Rank<Tensor<T, N, C>> {
    enum : size_t {
        value = N
    };
};

/** Provides typedef 'type' as the type of the first Tensor
 *  in `Tensors...`. Definition Statement.
 */
template<typename... Tensors>
struct FirstTensor;

/** Provides typedef 'type' as the type of the first Tensor in `Tensors...` */
template<typename Tensor, typename... Tensors>
struct FirstTensor<Tensor, Tensors...> {
    using type = typename std::remove_reference<Tensor>::type;
    static_assert(IsTensor<type>::value, EXPECTED_TENSOR);
};

/** Provides typedef 'type' as the type of the first Tensor in `Tensors...` */
template<typename... Exprs>
struct FirstExpression;

/** Provides typedef 'type' as the type of the first Tensor in `Tensors...` */
template<typename Expr, typename... Exprs>
struct FirstExpression<Expr, Exprs...> {
    using type = typename std::remove_reference<Expr>::type;
};

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

/** returns x if x > y, y o.w. */
constexpr size_t Max(size_t x, size_t y)
{
    return x > y ? x : y;
}

/** Returns 1 if x < y, 0 o.w. */
constexpr bool LessThan(size_t x, size_t y)
{
    return x < y;
}

/** Returns 1 if a < x < b, 0 o.w. */
constexpr bool InBetween(size_t x, size_t a, size_t b)
{
    return (x >= a) && (x < b);
}

/** Combine std::remove_reference and std::remove_cv */
template<typename T>
struct RemoveCVRef {
    using type = typename std::remove_cv<
            typename std::remove_reference<T>::type>::type;
};

/** template && */
template<bool B1, bool B2>
struct LogicalAnd {
    static constexpr bool value = B1 && B2;
};

/** Fills `array2` with the values of `args...` between [Middle, End) */
template<size_t Index, size_t Middle, size_t End>
struct FillSecond {
    template<typename... Args>
    FillSecond(
            std::array<size_t, Middle> &array1, std::array<size_t, End - Middle> &array2,
            size_t next, Args &&... args
    )
    {
        array2[Index - Middle] = next;
        FillSecond<Index + 1, Middle, End>(array1, array2, args...);
    }
};

template<size_t Middle, size_t End>
struct FillSecond<End, Middle, End> {
    template<typename... Args>
    FillSecond(std::array<size_t, Middle> &, std::array<size_t, End - Middle> &)
    {}
};

template<size_t End>
struct FillSecond<End, 0, End> {
    template<typename... Args>
    FillSecond(std::array<size_t, 0> &, std::array<size_t, End> &)
    {}
};

/** Fills `array1` with the values of `args...` between [0, Middle) */
template<size_t, size_t, size_t>
struct FillFirst;

template<size_t Index, size_t Middle, size_t End>
struct FillFirst {
    template<typename... Args>
    FillFirst(
            std::array<size_t, Middle> &array1, std::array<size_t, End - Middle> &array2,
            size_t next, Args... args
    )
    {
        array1[Index] = next;
        FillFirst<Index + 1, Middle, End>(array1, array2, args...);
    }
};

template<size_t Index, size_t Middle>
struct FillFirst<Index, Middle, Middle> {
    template<typename... Args>
    FillFirst(
            std::array<size_t, Middle> &array1, std::array<size_t, 0> &array2,
            size_t next, Args... args
    )
    {
        array1[Index] = next;
        FillFirst<Index + 1, Middle, Middle>(array1, array2, args...);
    }
};

template<size_t Middle, size_t End>
struct FillFirst<Middle, Middle, End> {
    template<typename... Args>
    FillFirst(
            std::array<size_t, Middle> &array1, std::array<size_t, End - Middle> &array2,
            size_t next, Args... args
    )
    {
        array2[0] = next;
        FillSecond<Middle + 1, Middle, End>(array1, array2, args...);
    }
};

template<size_t End>
struct FillFirst<0, 0, End> {
    template<typename... Args>
    FillFirst(
            std::array<size_t, 0> &array1, std::array<size_t, End> &array2,
            size_t next, Args... args
    )
    {
        array2[0] = next;
        FillSecond<1, 0, End>(array1, array2, args...);
    }
};

template<size_t End>
struct FillFirst<End, End, End> {
    template<typename... Args>
    FillFirst(std::array<size_t, End> &, std::array<size_t, 0> &)
    {}
};

template<>
struct FillFirst<0, 0, 0> {
    template<typename... Args>
    FillFirst(std::array<size_t, 0> &, std::array<size_t, 0> &)
    {}
};

/** Member STL arrays `array1` and `array2` hold the values of `args...` between
 *  [0, Middle) and [Middle, End) respectively. `Args...` must be of type size_t
 */
template<size_t Middle, size_t End>
struct FillArgs {
    std::array<size_t, Middle> array1;
    std::array<size_t, meta::NonZeroDifference(End, Middle)> array2;

    template<typename... Args>
    explicit FillArgs(Args... args)
    {
        static_assert(End == sizeof...(Args), PANIC_ASSERTION);
        FillFirst<0, Middle, Middle + meta::NonZeroDifference(End, Middle)>(
                array1,
                array2,
                args...
        );
    }
};

/** Member enum `value` is `true` iff `I...` is a
 *  strictly increasing sequence of unsigned integers.
 */
template<size_t...>
struct IsIncreasing;

/** Member enum 'value' is 'true' iff 'I...' contains 'Index` */
template<size_t...>
struct ContainsIndex;

/** Member enum `value` is `true` iff `I...` is a
 *  strictly increasing sequence of unsigned integers.
 */
template<size_t I1, size_t I2, size_t... I>
struct IsIncreasing<I1, I2, I...> {
    enum : bool {
        value = ((I1 < I2) && IsIncreasing<I2, I...>::value)
    };
};

/** Member enum `value` is `true` iff `I...` is a
 *  strictly increasing sequence of unsigned integers.
 */
template<size_t Index>
struct IsIncreasing<Index> : std::true_type {
};

/** Member enum `value` is `true` iff `I...` is a
 *  strictly increasing sequence of unsigned integers.
 */
template<>
struct IsIncreasing<> : std::true_type {
};

/** Member enum `value` is `true` iff `I...` is a
 *  consectutive sequence of unsigned integers.
 */
template<size_t...>
struct IsConsecutive;

template<size_t I1, size_t I2, size_t... I>
struct IsConsecutive<I1, I2, I...> {
    enum : bool {
        value = ((I1 == I2 - 1) && IsIncreasing<I2, I...>::value)
    };
};

template<size_t I>
struct IsConsecutive<I> : std::true_type {
};

template<>
struct IsConsecutive<> : std::true_type {
};

/** Member enum `value` is the last value in `I...` */
template<size_t...>
struct FirstIndex;

template<size_t I1, size_t I2, size_t... I>
struct FirstIndex<I1, I2, I...> {
    enum : bool {
        value = I1
    };
};

/** Member enum `value` is the last value in `I...` */
template<size_t...>
struct LastIndex;

template<size_t I1, size_t I2, size_t... I>
struct LastIndex<I1, I2, I...> {
    enum : bool {
        value = LastIndex<I2, I...>::value
    };
};

template<size_t I>
struct LastIndex<I> {
    enum : bool {
        value = I
    };
};

/** Member enum 'value' is 'true' iff 'I...' contains 'Index` */
template<size_t Index, size_t Next, size_t... I>
struct ContainsIndex<Index, Next, I...> {
    enum : bool {
        value = (Index == Next) || ContainsIndex<Index, I...>::value
    };
};

/** Member enum 'value' is 'true' iff 'I...' contains 'Index` */
template<size_t Index>
struct ContainsIndex<Index> : std::false_type {
};

/** Wrapper around a vardiac size_t pack */
template<size_t... I>
struct Sequence {
    enum : size_t {
        size = sizeof...(I)
    };
};

/** Extends `Sequence<I...>` by placing `Index` in front */
template<size_t Index, typename>
struct Append;

/** Extends `Sequence<I...>` by placing `Index` at the End */
template<size_t Index, typename>
struct AppendEnd;

/** Concatenates `Sequence1` and `Sequence2` together */
template<typename, typename>
struct Concatenate;

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`.
 */
template<size_t, size_t...>
struct CountLTMax;

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`.
 */
template<size_t, typename>
struct CountLTMaxSequence;

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`.
 */
template<size_t Max, size_t Index, size_t... I>
struct CountLTMax<Max, Index, I...> {
    enum : size_t {
        value = LessThan(Index, Max) + CountLTMax<Max, I...>::value
    };
};

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`.
 */
template<size_t Max>
struct CountLTMax<Max> {
    enum : size_t {
        value = 0
    };
};

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`.
 */
template<size_t Max, size_t... I>
struct CountLTMaxSequence<Max, Sequence<I...>> {
    enum : size_t {
        value = CountLTMax<Max, I...>::value
    };
};

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`, and greater than or equal to `Min`.
 */
template<size_t, size_t, size_t...>
struct CountInBetween;

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`, and greater than or equal to `Min`.
 */
template<size_t, size_t, typename>
struct CountInBetweenSequence;

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`, and greater than or equal to `Min`.
 */
template<size_t Min, size_t Max, size_t Index, size_t... I>
struct CountInBetween<Min, Max, Index, I...> {
    enum : size_t {
        value = InBetween(Index, Min, Max) +
                CountInBetween<Min, Max, I...>::value
    };
};

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`, and greater than or equal to `Min`.
 */
template<size_t Min, size_t Max>
struct CountInBetween<Min, Max> {
    enum : size_t {
        value = 0
    };
};

/** Member enum `value` contains the number of elements
 *  in `I...` strictly less than `Max`, and greater than or equal to `Min`.
 */
template<size_t Min, size_t Max, size_t... I>
struct CountInBetweenSequence<Min, Max, Sequence<I...>> {
    enum : size_t {
        value = CountInBetween<Min, Max, I...>::value
    };
};

/** typedef `sequence` is `Index` inserted into ordered sequence `I...` */
template<size_t...>
struct InsertIntoOrderedSequence;

/** provides typedef `sequence` which is a sequence with each
 *  element subtracted by `offset`.
 */
template<size_t...>
struct SequenceNegativeOffset;

/** provides typedef `sequence` which is a sequence with each
 *  element subtracted by `offset`.
 */
template<size_t...>
struct SequencePositiveOffset;

/** Transform Sequence */
template<typename, template<size_t...> class, size_t...>
struct SequenceTransformer;

/** provides typedef `sequence` which is a sequence with each
 *  element subtracted by `offset`. Recursive case.
 */
template<size_t Offset, size_t Index, size_t... I>
struct SequenceNegativeOffset<Offset, Index, I...> {
    using sequence = typename Append<Index - Offset,
            typename SequenceNegativeOffset<Offset, I...>::sequence>::sequence;
};

/** provides typedef `sequence` which is a sequence with each
 *  element subtracted by `offset`. Base case.
 */
template<size_t Offset>
struct SequenceNegativeOffset<Offset> {
    using sequence = Sequence<>;
};

/** provides typedef `sequence` which is a sequence with each
 *  element added by `offset`. Recursive case.
 */
template<size_t Offset, size_t Index, size_t... I>
struct SequencePositiveOffset<Offset, Index, I...> {
    using sequence = typename Append<Index + Offset,
            typename SequenceNegativeOffset<Offset, I...>::sequence>::sequence;
};

/** provides typedef `sequence` which is a sequence with each
 *  element added by `offset`. Base case.
 */
template<size_t Offset>
struct SequencePositiveOffset<Offset> {
    using sequence = Sequence<>;
};

/** Provides typedef `sequence` which is
 *  `sequence` transformed with `transformer`
 */
template<size_t... I, template<size_t...> class Transformer, size_t... Initial>
struct SequenceTransformer<Sequence<I...>, Transformer, Initial...> {
    using sequence = typename Transformer<Initial..., I...>::sequence;
};

/** Extends `Sequence<I...>` by placing `Index` in front */
template<size_t Index, size_t... I>
struct Append<Index, Sequence<I...>> {
    using sequence = Sequence<Index, I...>;
};

/** Extends `Sequence<I...>` by placing `Index` at the end */
template<size_t Index, size_t... I>
struct AppendEnd<Index, Sequence<I...>> {
    using sequence = Sequence<I..., Index>;
};

/** Concatenates `Sequence1` and `Sequence2` together */
template<size_t... I1, size_t... I2>
struct Concatenate<Sequence<I1...>, Sequence<I2...>> {
    using sequence = Sequence<I1..., I2...>;
};

/** Creates a `Sequence<I...>` where `I...` is the integers [`Index`, `N`)  */
template<size_t Index, size_t N>
struct MakeIndexSequence {
    using sequence = typename Append<Index,
            typename MakeIndexSequence<Index + 1, N>::sequence>::sequence;
};

/** Creates a `Sequence<I...>` where `I...` is the natural numbers up to `N`  */
template<size_t N>
struct MakeIndexSequence<N, N> {
    using sequence = Sequence<>;
};

/** Typedef `sequence` is a `Sequence<X...>` where
 *  `X...` are the first `ThreshHold` elements of `I...`
 */
template<size_t ThreshHold, size_t... I>
struct MakeLeftSequence;

/** Typedef `sequence` is a `Sequence<X...>` where
 *  `X...` are the first `ThreshHold` elements of `I...`
 *  Recursive case.
 */
template<size_t ThreshHold, size_t Index, size_t... I>
struct MakeLeftSequence<ThreshHold, Index, I...> {
    using sequence = typename Append<
            Index, typename MakeLeftSequence<ThreshHold - 1, I...>::sequence>::sequence;
};

/** Typedef `sequence` is a `Sequence<X...>` where
 *  `X...` are the first `ThreshHold` elements of `I...`
 *  Base case.
 */
template<size_t Index, size_t... I>
struct MakeLeftSequence<0, Index, I...> {
    using sequence = Sequence<>;
};

/** Typedef `sequence` is a `Sequence<X...>` where
 *  `X...` are the first `ThreshHold` elements of `I...`
 *  Base case.
 */
template<>
struct MakeLeftSequence<0> {
    using sequence = Sequence<>;
};

/** Typedef `sequence` is a `Sequence<X...>` where
 *  `X...` are the  elements of `I...` after the first
 *  `ThreshHold` elements.
 */
template<size_t ThreshHold, size_t... I>
struct MakeRightSequence;

/** Typedef `sequence` is a `Sequence<X...>` where the
 * elements of `I...` after the first `ThreshHold` elements.
 */
template<size_t, typename>
struct RightSequence;

/** Typedef `sequence` is a `Sequence<X...>` where `X...` are
 *  the first `ThreshHold` elements of `I...`
 */
template<size_t, typename>
struct LeftSequence;

/** Typedef `sequence` is a `Sequence<X...>` where
 *  `X...` are the  elements of `I...` after the first
 *  `ThreshHold` elements. Recursive Case.
 */
template<size_t ThreshHold, size_t Index, size_t... I>
struct MakeRightSequence<ThreshHold, Index, I...> {
    using sequence = typename MakeRightSequence<ThreshHold - 1, I...>::sequence;
};

/** Typedef `sequence` is a `Sequence<X...>` where
 *  `X...` are the  elements of `I...` after the first
 *  `ThreshHold` elements. Recursive Case.
 */
template<size_t Index, size_t... I>
struct MakeRightSequence<0, Index, I...> {
    using sequence = typename Append<Index,
            typename MakeRightSequence<0, I...>::sequence>::sequence;
};

/** Typedef `sequence` is a `Sequence<X...>` where
 *  `X...` are the  elements of `I...` after the first
 *  `ThreshHold` elements.
 */
template<>
struct MakeRightSequence<0> {
    using sequence = Sequence<>;
};

/** Typedef `sequence` is a `Sequence<X...>` where `X...` are
 *  the first `ThreshHold` elements of `I...`
 */
template<size_t ThreshHold, size_t... I>
struct LeftSequence<ThreshHold, Sequence<I...>> {
    using sequence = typename MakeLeftSequence<ThreshHold, I...>::sequence;
};

/** Typedef `sequence` is a `Sequence<X...>` where `X...` are the
 *  elements of `I...` after the first `ThreshHold` elements.
 */
template<size_t ThreshHold, size_t... I>
struct RightSequence<ThreshHold, Sequence<I...>> {
    using sequence = typename MakeRightSequence<ThreshHold, I...>::sequence;
};

/** Sets `a` at every index in `I...` to zero */
template<typename T, size_t Size, size_t... I>
struct InsertValueIntoArray;

template<typename T, size_t Size, size_t I1, size_t I2, size_t... Indices>
struct InsertValueIntoArray<T, Size, I1, I2, Indices...> {
    InsertValueIntoArray(std::array<T, Size> &a, T const &value)
    {
        static_assert(I1 != I2, SLICE_INDICES_REPEATED);
        static_assert(I1 < I2, SLICE_INDICES_DESCENDING);
        a[I1] = value;
        InsertValueIntoArray<T, Size, I2, Indices...>{a, value};
    }
};

template<typename T, size_t Size, size_t Index>
struct InsertValueIntoArray<T, Size, Index> {
    InsertValueIntoArray(std::array<T, Size> &a, T const &value)
    {
        static_assert(Index < Size, INDEX_OUT_OF_BOUNDS);
        a[Index] = value;
    }
};

template<typename T, size_t Size>
struct InsertValueIntoArray<T, Size> {
    InsertValueIntoArray(std::array<T, Size> &, T const &)
    {}
};

/** Enum 'value' is true if all arguments are Tensors, false o.w. */
template<typename... Args>
struct AreTensors;

template<typename Arg, typename... Args>
struct AreTensors<Arg, Args...> {
    enum : bool {
        value = IsTensor<Arg>::value && AreTensors<Args...>::value
    };
};

template<>
struct AreTensors<> : std::true_type {
};

/** Enum 'value' is true if all arguments are Expressions, false o.w. */
template<typename... Args>
struct AreExpressions;

template<typename Arg, typename... Args>
struct AreExpressions<Arg, Args...> {
    enum : bool {
        value = (IsExpression<Arg>::value || IsTensor<Arg>::value)
                && AreExpressions<Args...>::value
    };
};

template<>
struct AreExpressions<> : std::true_type {
};

/** Member `value` is the sum of the ranks of every Expression in `Exprs...` */
template<typename... Exprs>
struct RankSum;

template<typename Expr, typename... Exprs>
struct RankSum<Expr, Exprs...> {
    enum : size_t {
        value = Expr::rank() + RankSum<Exprs...>::value
    };
};

template<>
struct RankSum<> {
    enum : size_t {
        value = 0
    };
};

} // namespace meta

namespace details {

// FIXME Give this a better name + documentation
template<size_t N, size_t NStrides>
void UpdateIndices(Indices<N> &relative_indices, Shape<N> const &shape,
        std::array<size_t, NStrides> &absolute_indices,
        size_t const *const (&strides_array)[NStrides])
{
    static_assert(N, PANIC_ASSERTION);
    int dim_index = N - 1;
    bool propogate = true;
    while (dim_index >= 0 && propogate) {
        ++relative_indices[dim_index];
        for (size_t i = 0; i < NStrides; ++i)
            absolute_indices[i] += strides_array[i][dim_index];
        if (relative_indices[dim_index] == shape[dim_index]) {
            relative_indices[dim_index] = 0;
            for (size_t i = 0; i < NStrides; ++i)
                absolute_indices[i] -= shape[dim_index] * strides_array[i][dim_index];
        } else {
            propogate = false;
        }
        --dim_index;
    }
}

// scalar override provided to match APIs
// nothing needs to be done if the Tensor is scalar
template<size_t NStrides>
inline void UpdateIndices(Indices<0> &, Shape<0> const &, std::array<size_t, NStrides> &,
        size_t const *const (&)[NStrides], size_t = 0) 
{}

/**
 * Fill the first uncovered N array elements.
 *
 * @tparam T array type
 * @tparam Size array size
 * @param value to fill the array
 * @param a array to fill
 * @param cover skips the elements where cover is true
 * @param n number of elements to fill
 */
template<typename T, size_t Size>
void FillFirstUncoveredN(std::array<T, Size> &a, std::array<bool, Size> const &cover, 
    size_t n, T const &value)
{
    for (size_t i = 0; i < Size; ++i) {
        if (!n) break;
        if (!cover[Size - i - 1]) {
            a[Size - i - 1] = value;
            --n;
        }
    }
}

// fills `result` with the elements of `input`, skipping the element at position `Index`
template<size_t Index, size_t N>
void FillExceptForIndex(size_t const *input, size_t *result)
{
    static_assert(N > Index, INDEX_OUT_OF_BOUNDS);
    std::copy_n(input, Index, result);
    std::copy_n(input + Index + 1, N - Index - 1, result + Index);
}

// bit twiddle to get the nearest power of 2 larger than `x`
inline size_t NextPowerOfTwo(size_t x)
{
    static_assert(sizeof(x) == 4 || sizeof(x) == 8, PANIC_ASSERTION);
    x--;
    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    // most compilers will elide this
    if (sizeof(size_t) == 8)
        x |= x >> 32u;
    return x + 1;
}

/* ------- Expansion for methods which take Tensors...  --------- */

// Given a pack of Expressions, return the shape of the first expression
template<typename NodeType, typename... Expressions>
inline Shape<NodeType::rank()> const &GetShape(Expression<NodeType> const &expr,
        Expressions const &...)
{
    return expr.self().shape();
}

// Forward tuple of expressions `tensors` to non-void `fn`
template<typename U, typename FunctionType, typename Tuple, size_t... I>
inline U map_forward_sequence(FunctionType &&fn, size_t *indices, Tuple &&tensors,
        meta::Sequence<I...>)
{
    return std::forward<FunctionType>(fn)(
        std::get<I>(std::forward<Tuple>(tensors)).pGet(indices, I)...);
}

// Forward tuple of expressions `tensors` to void `fn`
template<typename FunctionType, typename Tuple, size_t... I>
inline void MapForwardSequence(FunctionType &&fn, size_t *indices,
        Tuple &&tensors, meta::Sequence<I...>)
{
    std::forward<FunctionType>(fn)(
            std::get<I>(std::forward<Tuple>(tensors)).pGet(indices, I)...);
}

template<typename U, typename FunctionType, typename Tuple, size_t... I>
inline void ReduceForwardSequence(
        U &&ret_val, FunctionType &&fn, size_t *indices,
        Tuple &&tensors, meta::Sequence<I...>
)
{
    ret_val = std::forward<FunctionType>(fn)(std::forward<U>(ret_val),
            std::get<I>(std::forward<Tuple>(tensors)).pGet(indices, I)...);
}

template<typename U, size_t M, template<class> class C_, typename FunctionType, 
  typename Tuple, size_t... I>
inline void ElemWiseForwardSequence(
        Tensor<U, M, C_> &tensor, size_t index,
        FunctionType &&fn, size_t *indices, Tuple &&tensors,
        meta::Sequence<I...>
)
{
    tensor.pSet(index, std::forward<FunctionType>(fn)(
          std::get<I>(std::forward<Tuple>(tensors)).pGet(indices, I)...));
}

/** Map receiver after all Expressions have been converted to Tensors */
template<typename FunctionType, typename... Tensors>
void Map(FunctionType &&fn, Tensors &&... tensors)
{
    static_assert(sizeof...(tensors), PANIC_ASSERTION);

    auto const &shape = details::GetShape(tensors...);
    VARDIAC_MAP(assert(shape == tensors.shape() && SHAPE_MISMATCH));

    constexpr size_t M = std::remove_reference<decltype(shape)>::type::rank();

    size_t cumulative_index = shape.index_product();
    auto reference_indices = Indices<M>{};
    auto indices = std::array<size_t, sizeof...(Tensors)>{};
    size_t const *const strides[sizeof...(Tensors)] = {tensors.strides().data()...};
    auto sequence = typename meta::MakeIndexSequence<0, sizeof...(Tensors)>::sequence{};

    for (size_t i = 0; i < cumulative_index; ++i) {
        details::MapForwardSequence(
                std::forward<FunctionType>(fn),
                indices.data(),
                std::forward_as_tuple(tensors...),
                sequence);
        details::UpdateIndices(reference_indices, shape, indices, strides);
    }
}

/** map receiver after all Expressions have been converted to Tensors */
template<typename U, template<class> class C_, typename FunctionType, typename... Tensors>
Tensor<U, FirstExpression<Tensors...>::type::rank(), C_>
map(FunctionType &&fn, Tensors const &... tensors)
{
    static_assert(sizeof...(Tensors), NO_TENSORS_PROVIDED);
    static_assert(meta::AreExpressions<Tensors...>::value, EXPECTING_EXPRESSION);
    auto const &shape = details::GetShape(tensors...);
    VARDIAC_MAP(assert(shape == tensors.shape() && SHAPE_MISMATCH));
    constexpr size_t M = FirstExpression<Tensors...>::type::rank();
    auto tensor = Tensor<U, M, C_>(shape);
    size_t cumulative_index = shape.index_product();
    auto reference_indices = Indices<M>{};
    auto indices = std::array<size_t, sizeof...(Tensors)>{};
    size_t const *const strides[sizeof...(Tensors)] = {tensors.strides().data()...};
    auto sequence = typename meta::MakeIndexSequence<0, sizeof...(Tensors)>::sequence{};
    // Forward the values produced by `fn` to `tensor`
    for (size_t i = 0; i < cumulative_index; ++i) {
        // `tensor`'s strides will be contiguous
        tensor.pSet(i,
                details::map_forward_sequence<U>(std::forward<FunctionType>(fn), indices.data(),
                        std::forward_as_tuple(tensors...), sequence));
        details::UpdateIndices(reference_indices, shape, indices, strides);
    }
    return tensor;
}

/** reduce receiver after all Expressions have been converted to Tensors */
template<typename U, typename FunctionType, typename... Tensors>
U reduce(U &&initial_value, FunctionType &&fn, Tensors const &... tensors)
{
    static_assert(sizeof...(tensors), NO_TENSORS_PROVIDED);
    auto const &shape = details::GetShape(tensors...);
    VARDIAC_MAP(assert(shape == tensors.shape() && SHAPE_MISMATCH));
    constexpr size_t M = std::remove_reference<decltype(shape)>::type::rank();

    auto ret_val = std::forward<U>(initial_value);
    size_t cumulative_index = shape.index_product();
    auto reference_indices = Indices<M>{};
    auto indices = std::array<size_t, sizeof...(Tensors)>{};
    size_t const *const strides[sizeof...(Tensors)] = { tensors.strides().data()... };
    auto sequence = typename meta::MakeIndexSequence<0, sizeof...(Tensors)>::sequence{};

    for (size_t i = 0; i < cumulative_index; ++i) {
        // `tensor`'s strides are contiguous
        details::ReduceForwardSequence(ret_val, std::forward<FunctionType>(fn), 
            indices.data(), std::forward_as_tuple(tensors...), sequence);
        details::UpdateIndices(reference_indices, shape, indices, strides);
    }
    return ret_val;
}

// reduce receiver after Expression has been converted to a Tensor
template<typename FunctionType, typename TensorType, size_t... ReducedIndices>
Tensor<typename TensorType::value_type, TensorType::rank() - sizeof...(ReducedIndices),
        TensorType::template container_type>
reduce_dimensions(FunctionType &&fn, TensorType const &tensor)
{
    using T = typename TensorType::value_type;
    constexpr size_t N = TensorType::rank();
    constexpr size_t M = sizeof...(ReducedIndices);

    // place `true` where dimensions are not being reduced
    auto reduced_indices = std::array<bool, TensorType::rank()>{};
    std::fill(reduced_indices.begin(), reduced_indices.end(), true);
    meta::InsertValueIntoArray<bool, TensorType::rank(), ReducedIndices...>{reduced_indices, false};
    auto result_dimensions = std::array<size_t, N - M>{};
    
    // FIXME refactor this into helper method?
    auto tensor_i = size_t{0};
    for (size_t i = 0; i < N; ++i) {
        if (reduced_indices[i]) result_dimensions[tensor_i++] = tensor.dimension(i);
    }
    auto result = Tensor<T, N - M, TensorType::template container_type>(
        Shape<N - M>(result_dimensions));
    auto result_indices = Indices<N - M>{};
    do {
      // FIXME use a tensor view instead, should be thread safe
      result[result_indices] = tensor.template slice<ReducedIndices...>(result_indices)
        .reduce(T{}, fn);
    } while (result_indices.increment(result.shape()));
    return result;
}

/** Multiply receiver after Expressions have been converted to tensors */
template<typename X, template<class> class C_, size_t I1, size_t I2, typename Y,
        typename Z, size_t M1, size_t M2, template<class> class C1, template<class> class C2>
Tensor<X, M1 + M2 - 2, C_> mul(Tensor<Y, M1, C1> const &tensor_1, 
    Tensor<Z, M2, C2> const &tensor_2)
{
    static_assert(M1, SCALAR_TENSOR_MULT);
    static_assert(M2, SCALAR_TENSOR_MULT);
    assert((tensor_1.shape_[I1] == tensor_2.shape_[I2]) && PANIC_ASSERTION);
    auto shape = Shape<M1 + M2 - 2>();
    details::FillExceptForIndex<I1, M1>(tensor_1.shape_.data(), shape.data());
    details::FillExceptForIndex<I2, M2>(tensor_2.shape_.data(), shape.data() + M1 - 1);
    auto prod_tensor = Tensor<X, M1 + M2 - 2, C_>(shape);

    size_t cumulative_index_1 = tensor_1.shape_.index_product() / tensor_1.shape_[I1];
    size_t cumulative_index_2 = tensor_2.shape_.index_product() / tensor_2.shape_[I2];
    auto reference_indices_1 = Indices<M1 - 1>{};
    auto reference_indices_2 = Indices<M2 - 1>{};
    size_t index = 0;

    // Set up strides and indices for tensor_1
    auto t1_strides = std::array<size_t, M1 - 1>{};
    auto t1_shape = Shape<M1 - 1>{};
    details::FillExceptForIndex<I1, M1>(tensor_1.strides_.data(), t1_strides.data());
    details::FillExceptForIndex<I1, M1>(tensor_1.shape_.data(), t1_shape.data());
    size_t const * const t1_strides_array[] = {t1_strides.data()};
    auto t1_indices = std::array<size_t, 1>{};

    // Set up strides and indices for tensor_2
    auto t2_strides = std::array<size_t, M2 - 1>{};
    auto t2_shape = Shape<M2 - 1>{};
    details::FillExceptForIndex<I2, M2>(tensor_2.strides_.data(), t2_strides.data());
    details::FillExceptForIndex<I2, M2>(tensor_2.shape_.data(), t2_shape.data());
    size_t const * const t2_strides_array[] = {t2_strides.data()};
    for (size_t _i1 = 0; _i1 < cumulative_index_1; ++_i1) {
        auto t2_indices = std::array<size_t, 1>{};
        for (size_t _i2 = 0; _i2 < cumulative_index_2; ++_i2) {
            auto value = X{};
            for (size_t x = 0; x < tensor_1.shape_[I1]; ++x) {
                value += tensor_1.ref_[tensor_1.offset_ + t1_indices[0] +
                                       tensor_1.strides_[I1] * x] *
                         tensor_2.ref_[tensor_2.offset_ + t2_indices[0] +
                                       tensor_2.strides_[I2] * x];
            }
            prod_tensor.ref_[index] = value;
            details::UpdateIndices(reference_indices_2, t2_shape, t2_indices, t2_strides_array);
            ++index;
        }
        details::UpdateIndices(reference_indices_1, t1_shape, t1_indices, t1_strides_array);
    }
    return prod_tensor;
}

} // namespace details

/* ---------------------------- String Formatting ---------------------------- */

/** Overload this template to define how custom data types should be printed.
 *  Overloads are provided for primitive integral types.
 *  Provide `std::string operator()(T const &) const;`
 */
template<typename T>
struct StringFormat;

template<>
struct StringFormat<int> {
    std::string operator()(int x) const { return std::to_string(x); }
};

template<>
struct StringFormat<unsigned> {
    std::string operator()(unsigned x) const { return std::to_string(x); }
};

template<>
struct StringFormat<long> {
    std::string operator()(long x) const { return std::to_string(x); }
};

template<>
struct StringFormat<long long> {
    std::string operator()(long long x) const { return std::to_string(x); }
};

template<>
struct StringFormat<unsigned long long> {
    std::string operator()(unsigned long long x) const { return std::to_string(x); }
};

template<>
struct StringFormat<float> {
    std::string operator()(float x) const { return std::to_string(x); }
};

template<>
struct StringFormat<double> {
    std::string operator()(double x) const { return std::to_string(x); }
};

template<>
struct StringFormat<long double> {
    std::string operator()(long double x) const { return std::to_string(x); }
};

/* --------------------------- OpenCL Meta-Patterns --------------------------- */

namespace opencl {
namespace meta {

// FIXME

} // namespace meta
} // namespace opencl

/* ------------------------------ OpenCL ------------------------------ */

#ifdef _ENABLE_OPENCL
namespace opencl {

class Info {
/** Singleton collection of opencl context information */
public:
    /** Reference to Singleton */
    static Info &v();

    /** Returns a vector of platforms avaliable to the platform */
    static std::vector<cl::Platform> get_platforms();

    /** Sets the target that OpenCL will compile kernels. This can be changed dynamically.
     *  This will also set the target device to the default platform device.
     */
    void set_platform(cl::Platform const &platform);

    /** Reference to the current target platform */
    cl::Platform const &platform() const { return platform_; }

    /** Returns a list of devices avaliable to the specified platform */
    static std::vector<cl::Device> get_devices(cl::Platform const &platform);

    /** Returns a list of devices avaliable to the current platform */
    std::vector<cl::Device> get_devices() const noexcept;

    /** Sets the device OpenCL will run kernels on */
    void set_device(cl::Device const &device);

    /** Reference to the current device */
    cl::Device const &device() const noexcept { return device_; }

    /** Reference to the device context */
    cl::Context const &context() const noexcept { return context_; }

    /** Set platform and device with one function call. If the specified device is not
     *  supported by the current platform, OpenCL will return an error
     */
    void Set(cl::Platform const &platform, cl::Device const &device);

private:
    Info();

    cl::Platform platform_;
    cl::Device device_;
    cl::Context context_;
};

inline Info &Info::v()
{
    static auto instance = Info{};
    return instance;
}

inline void Info::set_platform(cl::Platform const &platform)
{
    platform_ = platform;
    auto devices = std::vector<cl::Device>{get_devices(platform)};
    assert(!devices.empty() && OPENCL_NO_DEVICES);
    device_ = devices[0];
    context_ = cl::Context({device_});
}

inline std::vector<cl::Platform> Info::get_platforms()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(!platforms.empty() && OPENCL_NO_PLATFORMS);
    return platforms;
}

inline void Info::set_device(cl::Device const &device)
{
    device_ = device;
    context_ = cl::Context({device_});
}

inline std::vector<cl::Device> Info::get_devices(cl::Platform const &platform)
{
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    assert(!devices.empty() && OPENCL_NO_DEVICES);
    return devices;
}

inline std::vector<cl::Device> Info::get_devices() const noexcept
{
    return get_devices(platform_);
}

inline void Info::Set(cl::Platform const &platform, cl::Device const &device)
{
    platform_ = platform;
    device_ = device;
    context_ = cl::Context({device_});
}

inline Info::Info()
{
    // Defaults to the first platform, first device given by the OpenCL API
    platform_ = get_platforms()[0];
    device_ = get_devices(platform_)[0];
    context_ = cl::Context({device_});
}

namespace details {

// Keywords of OpenCL identifiers
constexpr char cKernelIdentifier[] = "kernel";
constexpr char cGlobalIdentifier[] = "global";
constexpr char cLocalIdentifier[] = "local";
constexpr char cConstIdentifier[] = "const";
constexpr char cPointerIdentifier[] = "*";

// Keywords of OpenCL data types
constexpr char cBoolType[] = "bool";
constexpr char cCharType[] = "char";
constexpr char cUCharType[] = "uchar";
constexpr char cShortType[] = "short";
constexpr char cUShortType[] = "ushort";
constexpr char cIntType[] = "int";
constexpr char cUIntType[] = "uint";
constexpr char cSizeTType[] = "size_t";
constexpr char cLongType[] = "long";
constexpr char cULongType[] = "ulong";
constexpr char cFloatType[] = "float";
constexpr char cVoidType[] = "void";

// Naming prefixes
constexpr char cVariablePrefix[] = "v";
constexpr char cFunctionPrefix[] = "f";
constexpr char cKernelPrefix[] = "k";
constexpr char cLocalPrefix[] = "l";
constexpr char cForPrefix[] = "i";

// Variable names
// Group Ids
constexpr char cGlobalIdName[] = "global_id";
constexpr char cLocalIdName[] = "local_id";
constexpr char cGroupIdName[] = "group_id";
constexpr char cLocalSizeName[] = "local_size";
constexpr char cOutputName[] = "o";

// Dimensions
constexpr char cReductionSize[] = "N";
constexpr char cWorkGroupSize[] = "WGS";
constexpr char cOffsetName[] = "offset";
constexpr char cWGSizeName[] = "wg_size";
constexpr char cInitialValueName[] = "ival";

// Built in functions
constexpr char cLocalMemFence[] = "CLK_LOCAL_MEM_FENCE";

// Multiplication specific
constexpr char cLHSCumulDimsName[] = "lhs_cdims";
constexpr char cRHSCumulDimsName[] = "rhs_cdims";
constexpr char cLHSStridesName[] = "lhs_strides";
constexpr char cRHSStridesName[] = "rhs_strides";
constexpr char cLHSOffsetName[] = "lhs_offset";
constexpr char cRHSOffsetName[] = "rhs_offset";

// Thread ID functions
constexpr char cGlobalIdFunction[] = "get_global_id";
constexpr char cLocalIdFunction[] = "get_local_id";
constexpr char cGroupIdFunction[] = "get_group_id";
constexpr char cLocalSizeFunction[] = "get_local_size";

// Synchronization functions
constexpr char cBarrierFunction[] = "barrier";

template<typename T>
struct OpenCLType;

template<>
struct OpenCLType<bool> {
    constexpr static char const *value = cBoolType;
};

template<>
struct OpenCLType<char> {
    constexpr static char const *value = cCharType;
};

template<>
struct OpenCLType<unsigned char> {
    constexpr static char const *value = cUCharType;
};

template<>
struct OpenCLType<short> {
    constexpr static char const *value = cShortType;
};

template<>
struct OpenCLType<unsigned short> {
    constexpr static char const *value = cUShortType;
};

template<>
struct OpenCLType<int> {
    constexpr static char const *value = cIntType;
};

template<>
struct OpenCLType<unsigned> {
    constexpr static char const *value = cUIntType;
};

template<>
struct OpenCLType<long> {
    constexpr static char const *value = cLongType;
};

template<>
struct OpenCLType<unsigned long> {
    constexpr static char const *value = cULongType;
};

template<>
struct OpenCLType<float> {
    constexpr static char const *value = cFloatType;
};

/** Creates a kernel accepting `buffers.size()` arguments and writes to a new
 *  buffer with size `num_elems`. The values of the new buffer are obtained by
 *  evaluating `expr`, an OpenCL expression.
 */
template<typename T>
cl::Buffer evaluate_basic_kernel(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string const &arg_list,
        std::string const &expr,
        size_t num_elems
)
{
    auto kernel_code = std::string("kernel void k(") + arg_list + "global "
                       + OpenCLType<T>::value + " *output) {";
    kernel_code += "\n\tsize_t global_id = get_global_id(0);";
    kernel_code += std::string("\noutput[global_id] = ") + expr + ";\n}\n";

    cl_int err = 0;
    cl::Buffer output_buffer(
            opencl::Info::v().context(),
            CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            num_elems * sizeof(T),
            nullptr,
            &err
    );
    assert((err == CL_SUCCESS) && OPENCL_KERNEL_ERROR);
    cl::Program::Sources sources({kernel_code});
    cl::Program program(Info::v().context(), sources);
    err = program.build({Info::v().device()});
    assert((err == CL_SUCCESS) && OPENCL_KERNEL_ERROR);

    auto kernel = cl::Kernel(program, cKernelPrefix);
    for (size_t i = 0; i < buffers.size(); ++i)
        kernel.setArg(i, buffers[i]);
    kernel.setArg(buffers.size(), output_buffer);
    err = cqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num_elems));
    assert((err == CL_SUCCESS) && OPENCL_KERNEL_ERROR);
    return output_buffer;
}

/** Given a Tensor Expression node, create its output buffer */
template<typename NodeType>
cl::Buffer create_buffer(NodeType const &node)
{
    auto cqueue = cl::CommandQueue(Info::v().context(), Info::v().device());
    auto buffers = std::vector<cl::Buffer>{};
    auto cl_arg_list = std::string{};
    auto cl_expr = node.opencl_kernel_code(cqueue, buffers, cl_arg_list);
    return node.opencl_evaluate_kernel(cqueue, buffers, cl_arg_list, cl_expr);
}

/** Returns the OpenCL code, as an std::string, for a reduction kernel with one input  */
template<typename ReturnType, typename Function>
std::string create_reduction_kernel_code()
{
    // The kernel's 1st argument is the input buffer, and the 2nd argument a local buffer,
    // 3rd argument is an output buffer which stores the result of the reduction,
    // 4th argument is the size of half of the reduction group, the fifth argument is
    // the offset, the sixth argument is the workgroup_size

    // Kernel decleration & argument list
    std::string kernel_code =
            std::string("kernel void k(global ") + OpenCLType<ReturnType>::value
            + " const *v, local " + OpenCLType<ReturnType>::value + " *l, global "
            + OpenCLType<ReturnType>::value + " *output, size_t const N, size_t const offset,"
            + " size_t const wg_size) {";

    kernel_code += R"(
  size_t global_id = get_global_id(0);
  size_t group_id = get_group_id(0);
  size_t local_id = get_local_id(0);
  size_t local_size = get_local_size(0);)";

    kernel_code += R"(
  l[local_id] = v[global_id + offset * wg_size];)";

    kernel_code += R"(
  barrier(CLK_LOCAL_MEM_FENCE);
  for (size_t i = N; i > 0; i >>= 1) {
    if (local_id < i && i + local_id < local_size)
      )";

    kernel_code += Function::opencl_reduce(
            std::string("l[local_id]"),
            std::string("l[local_id + i]")) + ";";

    kernel_code += R"(
    barrier(CLK_LOCAL_MEM_FENCE);
  };
  if (local_id == 0) output[offset] = l[0];
})";

    return kernel_code;
}

/** Returns the OpenCL code, as an std::string, for an offset kernel */
template<typename ReturnType, typename Function>
std::string create_offset_kernel_code()
{
    // The kernel's 1st argument is the input buffer, containing one element,
    // and the second argument is the initial value

    std::string kernel_code = R"(kernel void k(global )";
    kernel_code += std::string(OpenCLType<ReturnType>::value) + " *output, "
                   + OpenCLType<ReturnType>::value + " const ival) {" + "\n\toutput[0] = "
                   + Function::opencl_map(std::string("ival"), std::string("output[0]")) + ";\n}";

    return kernel_code;
}

template<typename ReturnType, typename Function>
cl::Buffer create_reduction_kernel(
        cl::CommandQueue &cqueue,
        cl::Buffer const &buffer,
        size_t cumul,
        ReturnType const &initial_value
)
{
    // Create the code for the reduction kernel
    std::string kernel_code = create_reduction_kernel_code<ReturnType, Function>();

    // Join output buffer, which is used as the reduction output buffer as well
    cl_int err = 0;
    auto output_buffer = cl::Buffer(
            Info::v().context(),
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            cumul * sizeof(ReturnType),
            nullptr,
            &err
    );
    assert((err == CL_SUCCESS) && OPENCL_KERNEL_ERROR);

    // Set up the program and kernel
    auto program = cl::Program(Info::v().context(), {kernel_code});
    err = program.build({Info::v().device()});
    assert((err == CL_SUCCESS) && OPENCL_KERNEL_ERROR);
    auto kernel = cl::Kernel(program, cKernelPrefix);
    assert((err == CL_SUCCESS) && OPENCL_KERNEL_ERROR);

    size_t const kernel_work_group_size =
            kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(Info::v().device(), &err);
    kernel.setArg(0, buffer);
    kernel.setArg(2, output_buffer);
    kernel.setArg(5, kernel_work_group_size);

    // the number of work groups of size `KERNEL_WORK_GROUP_SIZE` that fit the number of elements
    size_t num_work_groups;
    // the modulus of the number of elements and `KERNEL_WORK_GROUP_SIZE`, i.e. the remainder
    size_t last_group_size;
    // the total number of work groups, including the remainder if it exists
    size_t total_num_work_groups = cumul;

    // Perform reduction by reducing each data vector in segments of `KERNEL_WORK_GROUP_SIZE`
    // After each reduction, the size of the remaining data will be the ceiling of the remaining
    // number of elements divided by the KERNEL_WORK_GROUP_SIZE
    do {
        // Compute parameters for kernel
        num_work_groups = total_num_work_groups / kernel_work_group_size;
        last_group_size = total_num_work_groups % kernel_work_group_size;
        total_num_work_groups = num_work_groups + (last_group_size ? 1 : 0);

        // Reduction of individual work groups that fit the kernel workgroup size
        for (size_t i = 0; i < num_work_groups; ++i) {
            kernel.setArg(1, sizeof(ReturnType) * kernel_work_group_size, nullptr);
            kernel.setArg(3, kernel_work_group_size >> 1u);
            kernel.setArg(4, i);
            cqueue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(kernel_work_group_size),
                    cl::NDRange(kernel_work_group_size)
            );
            assert((err == CL_SUCCESS) && PANIC_ASSERTION);
        }

        // Reduction of the work group that does not fit the kernel workgroup size, if it exists
        if (last_group_size) {
            kernel.setArg(1, sizeof(ReturnType) * last_group_size, nullptr);
            kernel.setArg(3, tensor::details::NextPowerOfTwo(last_group_size) >> 1u);
            kernel.setArg(4, num_work_groups);
            cqueue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(last_group_size),
                    cl::NDRange(last_group_size)
            );
            assert((err == CL_SUCCESS) && PANIC_ASSERTION);
        }

        // perform the reduction on the same buffer
        kernel.setArg(0, output_buffer);
    } while (total_num_work_groups != 1);

    // Transform, depending on the function, the reduced value with the specified initial value
    kernel_code = create_offset_kernel_code<ReturnType, Function>();

    // Recompile the program
    program = cl::Program(Info::v().context(), {kernel_code});
    err = program.build({Info::v().device()});
    assert((err == CL_SUCCESS) && OPENCL_KERNEL_ERROR);
    kernel = cl::Kernel(program, cKernelPrefix);
    assert((err == CL_SUCCESS) && OPENCL_KERNEL_ERROR);

    // The inputs to the kernel are the reduced buffer and the initial value
    kernel.setArg(0, output_buffer);
    kernel.setArg(1, initial_value);

    // Run a single thread
    cqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1));

    return output_buffer;
}

template<size_t I1, size_t I2, typename LHSType, typename RHSType, size_t M1, size_t M2>
std::string create_multiplication_kernel_code(
        Shape<M1> const &lhs_shape,
        Shape<M2> const &rhs_shape
)
{
    // initialize contiguous strides
    auto lhs_strides = std::array<size_t, M1>{};
    auto rhs_strides = std::array<size_t, M2>{};
    lhs_shape.pInitializeStrides(lhs_strides);
    rhs_shape.pInitializeStrides(rhs_strides);

    // Initialize cumulative dimensions, excluding the axes that being folded over
    auto lhs_cumul_dim = std::array<size_t, M1 - 1>{};
    auto rhs_cumul_dim = std::array<size_t, M2 - 1>{};
    tensor::details::FillExceptForIndex<I1, M1>(
            lhs_shape.data(),
            lhs_cumul_dim.data()
    );
    tensor::details::FillExceptForIndex<I2, M2>(rhs_shape.data(), rhs_cumul_dim.data());

    // Accumulate lhs
    size_t accumulator = 1;
    for (size_t i = 0; i < M1 - 1; ++i) {
        size_t tmp = accumulator;
        accumulator *= lhs_cumul_dim[M1 - 1 - (i + 1)];
        lhs_cumul_dim[M1 - 1 - (i + 1)] = tmp;
    }

    // Accumulate rhs
    accumulator = 1;
    for (size_t i = 0; i < M2 - 1; ++i) {
        size_t tmp = accumulator;
        accumulator *= rhs_cumul_dim[M2 - 1 - (i + 1)];
        rhs_cumul_dim[M2 - 1 - (i + 1)] = tmp;
    }

    // The kernel's 1st and 2nd arguments are the first and second input buffers. The
    // third argument is the output buffer.

    // Matrix dimension definitions:
    // K: dimensionality of axis being folded over
    // S1: lhs stride of axis being folded over
    // S2: rhs stride of axis being folded over
    // N1: rank of LHS shape - 1
    // N2: rank of RHS shape - 1
    std::string kernel_code =
            std::string("#define K ") + std::to_string(lhs_shape[I1]) + "\n"
            + "#define S1 " + std::to_string(lhs_strides[I1]) + "\n"
            + "#define S2 " + std::to_string(rhs_strides[I2]) + "\n"
            + "#define N1 " + std::to_string(M1 - 1) + "\n"
            + "#define N2 " + std::to_string(M2 - 1) + "\n";

    // lhs cumulative dimensions and strides
    kernel_code += "size_t constant lhs_cdims[N1] = {";
    for (size_t i = 0; i < M1 - 1; ++i)
        kernel_code += std::to_string(lhs_cumul_dim[i]) + ", ";
    kernel_code[kernel_code.size() - 2] = '}';
    kernel_code[kernel_code.size() - 1] = ';';
    kernel_code += "\nsize_t constant lhs_strides[N1] = {";
    for (size_t i = 0; i < M1; ++i) {
        // skip the dimension of the axis being foleded over
        if (i != I1) kernel_code += std::to_string(lhs_strides[i]) + ", ";
    }
    kernel_code[kernel_code.size() - 2] = '}';
    kernel_code[kernel_code.size() - 1] = ';';

    // rhs cumulative dimensions and strides
    kernel_code += "\nsize_t constant rhs_cdims[N2] = {";
    for (size_t i = 0; i < M2 - 1; ++i)
        kernel_code += std::to_string(rhs_cumul_dim[i]) + ", ";

    // Replace the trailing comma
    kernel_code[kernel_code.size() - 2] = '}';
    kernel_code[kernel_code.size() - 1] = ';';

    kernel_code += "\nsize_t constant rhs_strides[N2] = {";
    // skip the dimension of the axis being foleded over
    for (size_t i = 0; i < M2; ++i)
        if (i != I2)
            kernel_code += std::to_string(rhs_strides[i]) + ", ";

    // Replace the trailing comma
    kernel_code[kernel_code.size() - 2] = '}';
    kernel_code[kernel_code.size() - 1] = ';';

    // Kernel decleration & argument list
    kernel_code += std::string("\nkernel void k(global ")
                   + OpenCLType<LHSType>::value + " const *lhs, global "
                   + OpenCLType<LHSType>::value + " const *rhs, global "
                   + OpenCLType<LHSType>::value + " *output) {\n";

    // Global size and id declerations
    kernel_code += R"(
  size_t lhs_id = get_global_id(0);
  size_t rhs_id = get_global_id(1);)";

    // Compute the lhs offset
    kernel_code += R"(
  size_t lhs_offset = 0;
  for (size_t i = 0; i < N1; ++i) {
    size_t tmp = lhs_id / lhs_cdims[i];
    lhs_id -= tmp * lhs_cdims[i];
    lhs_offset += tmp * lhs_strides[i];
  })";

    // Compute the rhs offset
    kernel_code += R"(
  size_t rhs_offset = 0;
  for (size_t i = 0; i < N2; ++i) {
    size_t tmp = rhs_id / rhs_cdims[i];
    rhs_id -= tmp * rhs_cdims[i];
    rhs_offset += tmp * rhs_strides[i];
  })";

    // Accumulate and store the result
    kernel_code += std::string("\n  ") + OpenCLType<LHSType>::value + " accum = 0;";
    kernel_code += R"(
  for (size_t i = 0; i < K; ++i)
    accum += lhs[lhs_offset + i * S1] * rhs[rhs_offset + i * S2];
  output[get_global_id(0) * get_global_size(1) + get_global_id(1)] = accum;
})";

    return kernel_code;
}

template<size_t I1, size_t I2, typename LHSType, typename RHSType, size_t M1, size_t M2>
cl::Buffer create_multiplication_kernel(
        cl::CommandQueue &cqueue,
        cl::Buffer &lhs_buffer,
        cl::Buffer &rhs_buffer,
        Shape<M1> const &lhs_shape,
        Shape<M2> const &rhs_shape
)
{
    // total number of elements
    size_t lhs_cumul = lhs_shape.index_product();
    size_t rhs_cumul = rhs_shape.index_product();
    size_t output_cumul = lhs_cumul / lhs_shape[I1] * rhs_cumul / rhs_shape[I2];

    std::string kernel_code = create_multiplication_kernel_code<I1, I2, LHSType, RHSType>(
            lhs_shape, rhs_shape
    );

    cl_int status = 0;

    // buffer to store multiplication result
    cl::Buffer output_buffer(
            Info::v().context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            output_cumul * sizeof(LHSType), nullptr, &status
    );
    assert((status == CL_SUCCESS) && OPENCL_BUFFER_ERROR);

    // Create the program and kernel
    cl::Program::Sources sources({kernel_code});
    cl::Program program(Info::v().context(), sources);
    status = program.build({Info::v().device()});
    assert((status == CL_SUCCESS) && OPENCL_KERNEL_ERROR);
    cl::Kernel kernel(program, cKernelPrefix);

    kernel.setArg(0, lhs_buffer);
    kernel.setArg(1, rhs_buffer);
    kernel.setArg(2, output_buffer);

    // Run the multiplication kernel
    status = cqueue.enqueueNDRangeKernel(
            kernel, cl::NullRange,
            cl::NDRange(
                    lhs_cumul / lhs_shape[I1],
                    rhs_cumul / rhs_shape[I2]
            ));
    assert((status == CL_SUCCESS) && OPENCL_KERNEL_ERROR);

    return output_buffer;
}

} // namespace details
} // namespace opencl

#endif

/* -------------------- Data Containers --------------------- */

namespace data {

template<typename T>
class Array { /*@data::Array<T>*/
/** Shared data container (using std::shared_ptr) that allocates data in a single contiguous
 *  array. Indexing and iterating is very fast, but resizing any dimension will
 *  require reallocating and copying the the entire array.
 *  Access to data is *not* atomic.
 */
public:

    /** Sets the underlying shared_ptr to `nullptr` */
    Array() = default;

    /** Allocates an array of size `capacity`, with `operator new[]`.
   *  Initializes every element to `T{}`,
   */
    explicit Array(size_t capacity);

    /** Copy constructor -- copies the underlying shared_ptr */
    Array(Array const &a) = default;

    Array &operator=(Array const &a) = default;

    /** Move constructor -- moves the underlying std::shared_ptr */
    Array(Array &&a) noexcept = default;

    Array &operator=(Array &&a) noexcept = default;

    /** Allocates an array of size `capacity`, with `operator new[]`.
   *  Initialize every element to `value`,
   */
    Array(size_t capacity, T const &value);

    /** Alloates an array of size `capacity` with `operator new[]`.
   *  Fills the array with  the elements between [`first`, `end`).
   *  `std::distance(first, end)` must be equivalent to `capacity`.
   */
    template<typename It>
    Array(size_t capacity, It const &first, It const &end);

    /** Takes direct ownership of `data` */
    Array(size_t, T *data) : data_(data, [](T *p) { delete[] p; })
    {}

    /** Constructs an element at `index` with `value`. Must perform
   *  correct forwarding. `index` must be less than capacity.
   */
    template<typename U>
    void Set(size_t index, U &&value)
    { data_.get()[index] = std::forward<U>(value); }

    /** Reference to the element at `index` offset of the
   *  underlying array. `index` must be less than `capacity`.
   */
    T &operator[](size_t index)
    { return data_.get()[index]; };

    /** Const-reference to the element at `index` offset of the
   *  underlying array. `index` must be less than `capacity`.
   */
    T const &operator[](size_t index) const
    { return data_.get()[index]; }

    /** Compares the underlying pointers */
    template<typename T_>
    friend bool operator==(Array<T_> const &x, Array<T_> const &y);

    /** Compares the underlying pointers */
    template<typename T_>
    friend bool operator!=(Array<T_> const &x, Array<T_> const &y);

private:
    std::shared_ptr<T> data_; /**< Contiguously allocated array of `T` */
};

template<typename T>
Array<T>::Array(size_t capacity)
        : data_(new T[capacity](), [](T *p) { delete[] p; })
{}

template<typename T>
Array<T>::Array(size_t capacity, T const &value)
        : data_(new T[capacity](), [](T *p) { delete[] p; })
{
    std::fill(data_.get(), data_.get() + capacity, value);
}

template<typename T>
template<typename It>
Array<T>::Array(size_t capacity, It const &first, It const &end)
        : data_(new T[capacity](), [](T *p) { delete[] p; })
{
    std::copy(first, end, data_.get());
}

/** Compares the underlying pointers */
template<typename T_>
bool operator==(Array<T_> const &x, Array<T_> const &y)
{
    return x.data_ == y.data_;
}

/** Compares the underlying pointers */
template<typename T_>
bool operator!=(Array<T_> const &x, Array<T_> const &y)
{
    return x.data_ != y.data_;
}

template<typename T>
class HashMap { /*@data::HashMap<T>*/
/** Shared data container (std::shared_ptr) that allocates data in a hash map
 *  (STL unordered_map). The map only stores non-zero entries (configurable zero-entry).
 *  Memory efficiency for sparse tensors is good, but quickly degrades as density increases.
 *  Index and iterating is fast, O(1), but resizing will require destroying and
 *  rehashing larger dimensions. Access to data is *not* atomic.
 */
public:

    typedef std::unordered_map<size_t, T> container_type;

    HashMap() = default;

    /** Copy constructor -- copies the STL shared_ptr */
    HashMap(HashMap const &hash_map) = default;

    HashMap &operator=(HashMap const &hash_map) = default;

    /** Move constructor -- move the STL shared_ptr */
    HashMap(HashMap &&hash_map) noexcept = default;

    HashMap &operator=(HashMap &&hash_map) noexcept = default;

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
    template<typename It>
    HashMap(size_t capacity, It const &first, It const &end);

    /** Equivalent to HashMap(size_t, It,  It), except the memory owned by
     *  `data` is also deleted (with delete[])
     */
    HashMap(size_t capacity, T *data);

    /** Invokes STL unordered_map destructor */
    ~HashMap() = default;

    /** Constructs an element at `index` with `value`. Must perform
     *  correct forwarding. `index` must be less than capacity.
     */
    template<typename U>
    void Set(size_t index, U &&value);

    /** Reference to the element at `index` offset of the
     *  underlying array. `index` must be less than `capacity`.
     */
    T &operator[](size_t index);

    /** Const-reference to the element at `index` offset of the
     *  underlying array. `index` must be less than `capacity`.
     */
    T const &operator[](size_t index) const;

    template<typename T_>
    friend bool operator==(HashMap<T_> const &x, HashMap<T_> const &y);

    template<typename T_>
    friend bool operator!=(HashMap<T_> const &x, HashMap<T_> const &y);

private:
    /** underlying STL hash map */
    std::shared_ptr<container_type> data_;
    /** zero element that isn't allocated in the map */
    T zero_elem_;
};

template<typename T>
HashMap<T>::HashMap(size_t): data_(std::make_shared<container_type>()), zero_elem_(T{}) {}

template<typename T>
HashMap<T>::HashMap(size_t, T const &value)
  : data_(std::make_shared<container_type>()), zero_elem_(value) {}

template<typename T>
template<typename It>
HashMap<T>::HashMap(size_t capacity, It const &first, It const &end)
        : data_(std::make_shared<container_type>()), zero_elem_(T{})
{
    assert((long) capacity == std::distance(first, end) && PANIC_ASSERTION);
    size_t index = 0;
    for (auto it = first; it != end; ++it) {
        if (*it != zero_elem_) data_->insert({index, *it});
        ++index;
    }
}

template<typename T>
HashMap<T>::HashMap(size_t capacity, T *data)
{
    HashMap<T *>(capacity, data, data + capacity);
    delete[] data;
}

template<typename T>
template<typename U>
void HashMap<T>::Set(size_t index, U &&value)
{
    if (value == (U)zero_elem_) data_->erase(index);
    else (*data_)[index] = std::forward<U>(value);
}

template<typename T>
T &HashMap<T>::operator[](size_t index)
{
    auto it = data_->find(index);
    if (it != data_->end()) return it->second;
    return ((*data_)[index] = zero_elem_);
}

template<typename T>
T const &HashMap<T>::operator[](size_t index) const
{
    auto it = data_->find(index);
    if (it != data_->end()) return it->second;
    return zero_elem_;
}

/** Compares the underlying pointers */
template<typename T_>
bool operator==(HashMap<T_> const &x, HashMap<T_> const &y)
{
    return x.data_ == y.data_;
}

/** Compares the underlying jointers */
template<typename T_>
bool operator!=(HashMap<T_> const &x, HashMap<T_> const &y)
{
    return x.data_ != y.data_;
}

} // namespace data

/* ---------------------------------- Proxy Objects ---------------------------------- */

/** Proxy object used to construct Tensors with C-arrays */
template<typename Array>
struct CArrayProxy {
    /** Capture C-array by reference */
    explicit CArrayProxy(Array const &_value);

    Array const &value;
};

template<typename Array>
CArrayProxy<Array>::CArrayProxy(Array const &_value): value(_value)
{
    static_assert(std::is_array<Array>::value, EXPECTING_C_ARRAY);
}

/** Shorthand proxxy wrapper function */
template<typename Array>
CArrayProxy<Array> _C(Array const &value)
{
    return CArrayProxy<Array>(value);
}

/** Proxy object used to avoid alias restrictions for tensor assignment */
template<typename NodeType>
struct NoAliasProxy {
    /** Capture Expression by reference */
    explicit NoAliasProxy(Expression<NodeType> const &_expr);

    Expression<NodeType> const &expr;
};

template<typename NodeType>
NoAliasProxy<NodeType>::NoAliasProxy(Expression<NodeType> const &_expr): expr(_expr)
{}

/** Shorthand proxxy wrapper function */
template<typename NodeType>
NoAliasProxy<NodeType> _NA(Expression<NodeType> const &expr)
{
    return NoAliasProxy<NodeType>(expr);
}

/* ----------------------- Core Data Structures ----------------------- */

/** CRTP base for Tensor expressions */
template<typename NodeType>
struct Expression {
    inline NodeType &self()
    { return *static_cast<NodeType *>(this); }

    inline NodeType const &self() const
    { return *static_cast<NodeType const *>(this); }
};

template<size_t N> /*@Shape<N>*/
class Shape {
/** Tensor shape object, where size_t template `N` represents the Tensor rank().
 *  Implemented as a wrapper around size_t[N].
 */
public:
    /* -------------------- typedefs -------------------- */

    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef Shape<N> self_type;

    /* -------------------- friends -------------------- */

    template<size_t M> friend
    class Shape;

    template<typename X, size_t M, template<class> class C_> friend
    class Tensor;

    template<typename LHS, typename RHS> friend
    class BinaryAddExpr;

    template<typename LHS, typename RHS> friend
    class BinarySubExpr;

    template<size_t I1, size_t I2, typename LHS, typename RHS> friend
    class BinaryMulExpr;

    template<typename LHS, typename RHS> friend
    class BinaryHadExpr;

    template<typename Function, typename... Exprs> friend
    class MapExpr;

    template<typename U, typename Function, typename... Exprs> friend
    class ReduceExpr;

    template<typename RHS> friend
    class UnaryNegExpr;

#if _ENABLE_OPENCL

    template<typename Expr> friend
    class opencl::Model;

#endif

#ifdef _ENABLE_OPENCL

    template<size_t I1, size_t I2, typename LHSType, typename RHSType, size_t M1, size_t M2>
    friend std::string opencl::details::create_multiplication_kernel_code(
            Shape<M1> const &lhs_shape,
            Shape<M2> const &rhs_shape
    );

#endif

    /* ------------------ Constructors ------------------ */

    /** Initializer_list constructor */
    Shape(std::initializer_list<size_t> dimensions);

    /** Array constructor */
    explicit Shape(std::array<size_t, N> const &data);

    /** Iterator constructor */
    template<typename RAIt>
    Shape(RAIt begin, RAIt const &end);

    /** Copy constructor */
    Shape(Shape<N> const &shape);

    /* ------------------ Assignment -------------------- */

    Shape<N> &operator=(Shape<N> const &shape);   /**< Copy assignment */

    /* -------------------- Getters --------------------- */

    constexpr static size_t rank()
    { return N; } /**< `N` */

    /** Get dimension reference at `index`. If debugging, fails an assertion
   *  if index is out of bounds. Note: 0-based indexing.
   */
    size_t &operator[](size_t index);

    /** Get dimension at `index`. If debugging, fails an assertion
   *  if index is out of bounds. Note: 0-based indexing.
   */
    size_t operator[](size_t index) const;

    /** Reference to the underlying size_t array */
    std::array<size_t, N> &dimensions()
    { return dimensions_; }

    /** Const Reference to the underlying size_t array */
    std::array<size_t, N> const &dimensions() const
    { return dimensions_; }

    /** Pointer to the underlying memory */
    size_t *data()
    { return dimensions_.data(); }

    /** Const pointer to the underlying memory */
    size_t const *data() const
    { return dimensions_.data(); }

    /* -------------------- Equality -------------------- */

    /** `true` iff every dimension is identical */
    bool operator==(Shape<N> const &shape) const noexcept;

    /** `true` iff any dimension is different */
    bool operator!=(Shape<N> const &shape) const noexcept
    { return !(*this == shape); }

    /** `true` iff rank()s are identical and every dimension is identical */
    template<size_t M>
    bool operator==(Shape<M> const &shape) const noexcept;

    /** `true` iff rank()s are not identical or any dimension is different */
    template<size_t M>
    bool operator!=(Shape<M> const &shape) const noexcept
    { return !(*this == shape); }

    /* ----------------- Expressions ------------------ */

    template<
            typename X,
            template<class> class C_,
            size_t I1,
            size_t I2,
            typename Y,
            typename Z,
            size_t M1,
            size_t M2,
            template<class> class C1,
            template<class> class C2
    >
    friend Tensor<X, M1 + M2 - 2, C_>
    details::mul(Tensor<Y, M1, C1> const &tensor_1, Tensor<Z, M2, C2> const &tensor_2);

    template<typename U, template<class> class C_>
    friend Tensor<U, 2, C_> transpose(Tensor<U, 2, C_> &mat);

    template<typename U, template<class> class C_, typename X,
            template<class> class C__>
    friend Tensor<X, 2, C__> transpose(Tensor<U, 2, C_> const &mat);

    template<typename U, template<class> class C_>
    friend Tensor<U, 2, C_> transpose(Tensor<U, 1, C_> &vec);

    template<typename U, template<class> class C_, typename X,
            template<class> class C__>
    friend Tensor<X, 2, C__> transpose(Tensor<U, 1, C_> const &vec);

    /* ------------------- Utility -------------------- */

    /** Returns the product of all of the indices */
    size_t index_product() const noexcept;

    /* -------------------- Print --------------------- */

    /** Produces "S{<dim1>, <dim2>, ..., <dimN>}" */
    std::string str() const;

private:

    /* ------------------ Data ------------------ */

    std::array<size_t, N> dimensions_; /**< Underlying data */

    /* ----------------- Utility ----------------- */

    /** Fills `strides` by accumulating over `dimensions_` */
    void pInitializeStrides(std::array<size_t, N> &strides) const noexcept;

    /* --------------- Constructor --------------- */

    explicit Shape(size_t const *data);

    Shape() = default;
};

template<size_t N>
Shape<N>::Shape(std::initializer_list<size_t> dimensions)
{
    assert(dimensions.size() == N && RANK_MISMATCH);
    size_t *shape_ptr = dimensions_.data();
    for (size_t dim : dimensions) {
        assert(dim && ZERO_ELEMENT);
        *(shape_ptr++) = dim;
    }
}

template<size_t N>
Shape<N>::Shape(Shape const &shape)
{
    for (size_t i = 0; i < N; ++i)
        assert((shape.dimensions_[i]) && ZERO_ELEMENT);
    dimensions_ = shape.dimensions_;
}

template<size_t N>
Shape<N>::Shape(std::array<size_t, N> const &data): dimensions_(data)
{}

template<size_t N>
template<typename RAIt>
Shape<N>::Shape(RAIt begin, RAIt const &end)
{
    std::copy(begin, end, dimensions_.begin());
}

template<size_t N>
Shape<N>::Shape(size_t const *data)
{
    std::copy_n(data, N, dimensions_.data());
}

template<size_t N>
Shape<N> &Shape<N>::operator=(Shape<N> const &shape)
{
    dimensions_ = shape.dimensions_;
    return *this;
}

template<size_t N>
size_t &Shape<N>::operator[](size_t index)
{
    assert((N > index) && DIMENSION_INVALID);
    return dimensions_[index];
}

template<size_t N>
size_t Shape<N>::operator[](size_t index) const
{
    assert((N > index) && DIMENSION_INVALID);
    return dimensions_[index];
}

template<size_t N>
bool Shape<N>::operator==(Shape<N> const &shape) const noexcept
{
    return dimensions_ == shape.dimensions_;
}

template<size_t N>
template<size_t M>
bool Shape<N>::operator==(Shape<M> const &shape) const noexcept
{
    return dimensions_ == shape.dimensions_;
}

template<size_t N>
size_t Shape<N>::index_product() const noexcept
{
    return std::accumulate(
            dimensions_.begin(),
            dimensions_.end(),
            1,
            std::multiplies<size_t>()
    );
}

template<size_t N>
std::string Shape<N>::str() const
{
    auto s = std::string("S{");
    for (size_t i = 0; i < N - 1; ++i)
        s += std::to_string(dimensions_[i]) + ", ";
    s += std::to_string(dimensions_[N - 1]) + "}";
    return s;
}

template<size_t N>
void Shape<N>::pInitializeStrides(std::array<size_t, N> &strides) const noexcept
{
    static_assert(N, PANIC_ASSERTION);
    size_t accumulator = 1;
    for (size_t i = 0; i < N; ++i) {
        strides[N - i - 1] = accumulator;
        accumulator *= dimensions_[N - i - 1];
    }
}

template<size_t N>
class Indices { /*@Indices<N>*/
/** Wrapper around size_t[N] to provide a specialized array
 *  for accessing Tensors and looping.
 */
public:

    /* ------------------ Typedefs ---------------- */

    typedef Indices self_type;
    typedef std::size_t size_type;

    /* --------------- Friend Classes ------------- */

    template<typename U, size_t M, template<class> class C_> friend
    class Tensor;

    template<typename LHS, typename RHS> friend
    class BinaryAddExpr;

    template<typename LHS, typename RHS> friend
    class BinarySubExpr;

    template<size_t I1, size_t I2, typename LHS, typename RHS> friend
    class BinaryMulExpr;

    template<typename LHS, typename RHS> friend
    class BinaryHadExpr;

    template<typename Function, typename... Exprs> friend
    class MapExpr;

    template<typename U, typename Function, typename... Exprs> friend
    class ReduceExpr;

    template<typename RHS> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

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
   *  false if overflow, true o.w.
   */
    bool increment(Shape<N> const &shape);

    /** Decrements `indices` so it effectively refers to the
   *  next element in an N-rank() Tensor. The dimensions are
   *  propogated accordingly, I.e. (2, 0) indices with a 2x3
   *  shape will become (1, 3) after decrement. Indices will
   *  underflow if decremented at 0, returns false if
   *  underflow, true o.w.
   */
    bool decrement(Shape<N> const &shape);

    /* ----------------- String Format ----------------- */

    /** Produces "I{<dim1>, <dim2>, ..., <dimN>}" */
    std::string str() const;

private:
    size_t indices_[N];

    // private constructors
    explicit Indices(size_t const *indices);
};

template<size_t N>
Indices<N>::Indices(std::initializer_list<size_t> indices)
{
    assert(indices.size() == N && RANK_MISMATCH);
    size_t *ptr = indices_;
    for (size_t index : indices)
        *(ptr++) = index;
}

template<size_t N>
Indices<N>::Indices()
{
    std::fill_n(indices_, N, 0);
}

template<size_t N>
size_t const &Indices<N>::operator[](size_t index) const
{
    assert(index < N && DIMENSION_INVALID);
    return indices_[index];
}

template<size_t N>
size_t &Indices<N>::operator[](size_t index)
{
    assert(index < N && DIMENSION_INVALID);
    return indices_[index];
}

template<size_t N>
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
    return dim_index >= 0;
}

template<size_t N>
bool Indices<N>::decrement(Shape<N> const &shape)
{
    int dim_index = N - 1;
    bool propogate = true;
    while (dim_index >= 0 && propogate) {
        if (indices_[dim_index] == 0) {
            indices_[dim_index] = shape[dim_index] - 1;
            --dim_index;
        } else {
            --indices_[dim_index];
            propogate = false;
        }
    }
    return dim_index >= 0;
}

template<size_t N>
Indices<N>::Indices(size_t const *indices)
{
    for (size_t i = 0; i < N; ++i)
        indices_[i] = indices[i];
}

template<size_t N>
std::string Indices<N>::str() const
{
    static_assert(N, PANIC_ASSERTION);
    auto s = std::string("I{");
    for (size_t i = 0; i < N - 1; ++i)
        s += std::to_string(indices_[i]) + ", ";
    s += std::to_string(indices_[N - 1]) + "}";
    return s;
}

template<typename T, size_t N, template<class> class C>
class Tensor : public Expression<Tensor<T, N, C>> {
/*@Tensor<T, N, C>*/
/** Any-rank() array of type `T`, with compile time rank `N`.
 *  The underlying data is implemented as a dynamically allocated contiguous array.
 */
public:

    /* ------------------ Type Definitions --------------- */
    typedef T value_type;
    template<typename X>
    using container_type =                        C<X>;
    typedef T &reference;
    typedef T const &const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef Tensor<T, N, C> self_type;
    typedef Tensor<T, N, C> return_type;

    /* ----------------- Friend Classes ----------------- */

    template<typename X, size_t M, template<class> class C_> friend
    class Tensor;

    template<typename LHS, typename RHS> friend
    class BinaryAddExpr;

    template<typename LHS, typename RHS> friend
    class BinarySubExpr;

    template<size_t I1, size_t I2, typename LHS, typename RHS> friend
    class BinaryMulExpr;

    template<typename LHS, typename RHS> friend
    class BinaryHadExpr;

    template<typename Function, typename... Exprs> friend
    class MapExpr;

    template<typename U, typename Function, typename... Exprs> friend
    class ReduceExpr;

    template<typename RHS> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ------------------ Proxy Objects ----------------- */

    class Proxy { /*@Proxy<T,N>*/
        /**
   * Proxy Tensor Object used for building tensors from reference.
   * This is used to differentiate proxy construction
   * for move and copy construction only.
   */
    public:
        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        Proxy() = delete;

    private:
        explicit Proxy(Tensor<T, N, C> const &tensor) : tensor_(tensor) {}

        Proxy(Proxy const &proxy) : tensor_(proxy.tensor_) {}

        Tensor<T, N, C> const &tensor_;
    };

    /* ------------------ Constructors ------------------ */

    /** Creates a Tensor with dimensions described by `dimensions`.
     *  Elements are zero initialized. Note: dimensions index from 0.
     */
    Tensor(std::initializer_list<size_t> dimensions);

    /** Creates a Tensor with dimensions described by `dimensions`.
     *  Values returned by `factory(args...)` are forwarded to the elements in the Tensor
     *  Note: dimensions index from 0.
     */
    template<typename FunctionType, typename... Arguments>
    Tensor(
            std::array<size_t, N> const &dimensions,
            std::function<FunctionType> &f,
            Arguments &&... args
    );

    /** Use a C multi-dimensional array to initialize the tensor. The
     *  multi-dimensional array is enclosed by the _C struct, and
     *  must be equal to the tensor's declared rank.
     */
    template<typename Array>
    Tensor(CArrayProxy<Array> &&md_array);

    /** Creates a Tensor with dimensions described by `shape`.
     *  Elements are zero initialized.
     */
    explicit Tensor(Shape<N> const &shape);

    /** Creates a Tensor with dimensions described by `shape`.
     *  Elements are copy initialized to value. Note: dimensions index from 0.
     */
    Tensor(Shape<N> const &shape, T const &value);

    /** Creates a Tensor with dimensions described by `shape`.
     *  Elements are copy initialized to the values returned by `factory`
     *  Note: dimensions index from 0.
     */
    template<typename FunctionType, typename... Arguments>
    Tensor(Shape<N> const &shape, std::function<FunctionType> &f, Arguments &&... args);

    /** Copy construction, allocates memory and copies from `tensor` */
    Tensor(Tensor<T, N, C> const &tensor);

    /** Copy construction, allocates memory and copies from `tensor` */
    template<typename U, template<class> class C_>
    Tensor(Tensor<U, N, C_> const &tensor);

    /** Move construction, takes ownership of underlying data, `tensor` is destroyed */
    Tensor(Tensor<T, N, C> &&tensor) noexcept;

    /** Constructs a reference to the `proxy` tensor. The tensors share
     *  the same underyling data, so changes will affect both tensors.
     */
    Tensor(typename Tensor<T, N, C>::Proxy const &proxy);

    /** Constructs the tensor produced by the expression */
    template<typename NodeType,
            typename = typename std::enable_if<NodeType::rank() == N>::type>
    Tensor(Expression<NodeType> const &expression);

#ifdef _ENABLE_OPENCL

    /** Constructs the tensor using the result from an OpenCL computation */
    template<typename NodeType>
    Tensor(opencl::Model<NodeType> const &model);

#endif

    /* ------------------- Destructor ------------------- */

    ~Tensor() noexcept = default;

    /* ------------------- Assignment ------------------- */

    /** Assign to every element of `this` the corresponding element in
     *  `tensor`. The shapes must match.
     */
    template<typename U, template<class> class C_>
    Tensor<T, N, C> &operator=(Tensor<U, N, C_> const &tensor);

    /** Assign to every element of `this` the corresponding element in
     *  `tensor`. The shapes must match.
     */
    Tensor<T, N, C> &operator=(Tensor<T, N, C> const &tensor);

    /** Move to every element of `this` the corresponding element in
     *  `tensor`. The shapes must match.
     */
    Tensor<T, N, C> &operator=(Tensor<T, N, C> &&tensor) noexcept;

    /** Move to every element of `this` the corresponding element in
     *  `tensor`. The shapes must match.
     */
    template<typename U, template<class> class C_>
    Tensor<T, N, C> &operator=(Tensor<U, N, C_> &&tensor);

    /** Assigns to the tensor the elements of a C multi-dimensional array,
     *  enclosed by the _C struct, and have rank() and dimensions
     *  equivalent to *this.
     */
    template<typename Array>
    Tensor<T, N, C> &operator=(CArrayProxy<Array> &&md_array);

    /** Assign to every element of `this` the corresponding element in
     *  `rhs` after expression evaluation. The shapes must match.
     */
    template<typename NodeType>
    Tensor<T, N, C> &operator=(Expression<NodeType> const &rhs);

    /** Expression evaluation under the assumption that no aliasing
     *  of the LHS occurs on the RHS, thus no temporary Tensor is created.
     */
    template<typename NodeType>
    Tensor<T, N, C> &operator=(NoAliasProxy<NodeType> const &rhs);

    /* --------------------- Getters --------------------- */

    /** Returns a reference to itself */
    Tensor &eval() noexcept
    { return *this; }

    /** Returns a const reference to itself */
    Tensor const &eval() const noexcept
    { return *this; }

    /** Get `N` */
    constexpr static size_t rank() { return N; } 

    /** Get the dimension at index. If debugging, fails an assertion
     *  if is out of bounds. Note: indexing starts at 0.
     */
    size_t dimension(size_t index) const { return shape_[index]; }

    /** Get a reference to the tensor shape */
    Shape<N> const &shape() const noexcept
    { return shape_; }

    /** Used to implement iterator-> */
    Tensor *operator->()
    { return this; }

    /** Used to implement const_iterator-> */
    Tensor const *operator->() const
    { return this; }

    /** Returns a reference to the element at position `indices` */
    T const &get(Indices<N> const &indices) const noexcept;

    /* -------------------- Setters -------------------- */

    /** Elements of `*this` are copy assigned by dereferencing from [`begin`, `end`)
     *  The number of elements between [`begin`, `end`) must be equivalent
     *  to the number of elements in the Tensor, or an assertion will fail.
     */
    template<typename RAIt>
    void Fill(RAIt begin, RAIt const &end);

    /** Forwards valye to the element at position `indices` */
    template<typename U>
    void Set(Indices<N> const &indices, U &&value);

    /* ------------------ Access To Data ----------------- */

    template<typename... Args>
    Tensor<T, N - sizeof...(Args), C> at(Args... args);

    /** Returns the resulting tensor by applying left to right index expansion of
     *  the provided arguments. I.e. calling `tensor(1, 2)` on a rank() 4 tensor is
     *  equivalent to `tensor(1, 2, :, :)`. If debugging, fails an assertion if
     *  any of the indices are out bounds. Note: indexing starts at 0.
     */
    template<typename... Args, typename = typename std::enable_if<N != sizeof...(Args)>::type>
    Tensor<T, N - sizeof...(Args), C> operator()(Args... args);

    template<typename... Args, typename = typename std::enable_if<N == sizeof...(Args)>::type>
    T &operator()(Args... args);

    template<typename... Args>
    Tensor<T, N - sizeof...(Args), C> const at(Args... args) const;

    /** See operator() */
    template<typename... Args, typename = typename std::enable_if<N != sizeof...(Args)>::type>
    Tensor<T, N - sizeof...(Args), C> const operator()(Args... args) const;

    /** See operator() */
    template<typename... Args, typename = typename std::enable_if<N == sizeof...(Args)>::type>
    T const &operator()(Args... args) const;

    /** See operator() */
    template<size_t M, typename = typename std::enable_if<N != M>::type>
    Tensor<T, N - M, C> operator[](Indices<M> const &indices);

    template<size_t M, typename = typename std::enable_if<N == M>::type>
    T &operator[](Indices<M> const &indices);

    /** See operator() const */
    template<size_t M,
            typename = typename std::enable_if<N != M>::type>
    Tensor<T, N - M, C> const operator[](Indices<M> const &indices) const;

    /** See operator() const */
    template<size_t M, typename = typename std::enable_if<N == M>::type>
    T const &operator[](Indices<M> const &indices) const;

    /** Slices denotate the dimensions which are left free, while indices
     *  fix the remaining dimensions at the specified index. I.e. calling
     *  `tensor.slice<1, 3, 5>(1, 2)` on a rank() 5 tensor is equivalent to
     *  `tensor(:, 1, :, 2, :)` and produces a rank() 3 tensor. If debugging,
     *   fails an assertion if any of the indices are out of bounds. Note:
     *   indexing begins at 0.
     */
    template<size_t... Slices, typename... Args,
            typename = typename std::enable_if<N != sizeof...(Args)>::type>
    Tensor<T, N - sizeof...(Args), C> slice(Args... args);

    /** Const slice<Slices...>(Args.. args) */
    template<size_t... Slices, typename... Args,
            typename = typename std::enable_if<N != sizeof...(Args)>::type>
    Tensor<T, N - sizeof...(Args), C> const slice(Args... indices) const;

    /** Indentical to operator()(Args...) && `sizeof...(Args) == N`
     *  with a static check to verify `sizeof...(Slices) == 0`.
     */
    template<size_t... Slices, typename... Args,
            typename = typename std::enable_if<N == sizeof...(Args)>::type>
    T &slice(Args... args);

    /** Const slice<>(Args...) && `sizeof...(Args) == N` */
    template<size_t... Slices, typename... Args,
            typename = typename std::enable_if<N == sizeof...(Args)>::type>
    T const &slice(Args... args) const;

    /** See slice<Slices...>(Args..); */
    template<
            size_t... Slices,
            size_t M,
            typename = typename std::enable_if<N != M>::type
    >
    Tensor<T, N - M, C> slice(Indices<M> const &indices);

    /** See slice<Slices...>(Args..); */
    template<
            size_t... Slices,
            size_t M,
            typename = typename std::enable_if<N != M>::type
    >
    Tensor<T, N - M, C> const slice(Indices<M> const &indices) const;

    /** Indentical to operator[](Indices<M> const&) && `M == N`
     *  with a static check to verify `sizeof...(Slices) == 0`.
     */
    template<size_t... Slices, size_t M,
            typename = typename std::enable_if<N == M>::type>
    T &slice(Indices<M> const &);

    /** Const version of slice<>(Indices<M> const&) && `M == N` */
    template<size_t... Slices, size_t M,
            typename = typename std::enable_if<N == M>::type>
    T const &slice(Indices<M> const &) const;

    /* -------------------- Expressions ------------------- */

    template<typename FunctionType, typename... Tensors>
    friend auto elemwise(FunctionType &&fn, Tensors &&... tensors)
    -> typename FirstTensor<Tensors...>::type;

    template<typename RHS>
    Tensor<T, N, C> &operator+=(Expression<RHS> const &rhs);

    template<typename RHS>
    Tensor<T, N, C> &operator-=(Expression<RHS> const &rhs);

    template<typename X, template<class> class C_, size_t I1, size_t I2, typename Y,
            typename Z, size_t M1, size_t M2, template<class> class C1,
            template<class> class C2>
    friend Tensor<X, M1 + M2 - 2, C_> details::mul(Tensor<Y, M1, C1> const &tensor_1, 
        Tensor<Z, M2, C2> const &tensor_2);

    /** Tensor multiplication between `*this` and the expression `rhs`
     *  The product is assigned to `*this`, and thus must have the same shape.
     */
    template<typename RHS>
    Tensor<T, N, C> &operator*=(Expression<RHS> const &rhs);

    /** Allocates a Tensor with shape equivalent to *this, and whose
     *  elements are equivalent to *this with operator-() applied.
     */
    Tensor<T, N, C> neg() const;

    /** Flattens the tensor, compressing the indices [I1, I2].
     *  For example, S{2, 3, 4} with indices 1, 2 flattened becomes S{2, 12}
     */
    template<size_t I1, size_t I2>
    Tensor<T, N - (I2 - I1), C> flatten();

    /* ---------------------- OpenCL ---------------------- */

#ifdef _ENABLE_OPENCL

    /** Creates an opencl buffer containing the tensor's data
     *  and adds it to `buffers`, adds its value to `arg_list`,
     *  and returns a string that represents a single element
     */
    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers, std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

    /* ------------------ Convert to string --------------- */

    /** Converts tensor into a string using square brace "[]" form
     *  (identical to python's MD arrays) I.e. a 1x1x1 Tensor
     *  with element x will appear as [[[x]]]
     */
    std::string str() const;

    /* -------------------- Equivalence ------------------ */

    /** Returns true iff the tensor's dimensions and data are equivalent */
    bool operator==(Tensor<T, N, C> const &tensor) const;

    /** Returns true iff the tensor's dimensions or data are not equivalent */
    bool operator!=(Tensor<T, N, C> const &tensor) const
    { return !(*this == tensor); }

    /** Returns true iff the tensor's dimensions are equal and every element satisfies e1 == e2 */
    template<typename X>
    bool operator==(Tensor<X, N, C> const &tensor) const;

    /** Returns true iff the tensor's dimensions are different or any element satisfies e1 != e2 */
    template<typename X>
    bool operator!=(Tensor<X, N, C> const &tensor) const
    { return !(*this == tensor); }

    /* -------------------- Iterators --------------------- */

    class Iterator { /*@Iterator<T, N, C>*/
    public:
        /** Iterator with freedom across one dimension of a Tensor.
         *  Allows access to the underlying tensor data.
         */

        /* ------------- Iterator Traits -------------- */

        typedef Tensor<T, N, C> value_type;
        typedef Tensor<T, N, C> &reference;
        typedef Tensor<T, N, C> *pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

        /* -------------- Friend Classes -------------- */

        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        /* --------------- Constructors --------------- */

        /** Copy construct an iterator to the same underlying Tensor */
        Iterator(Iterator const &it);

        Iterator &operator=(Iterator const &it);

        /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
        Iterator(Iterator &&it) noexcept;

        Iterator &operator=(Iterator &&it) noexcept;

        /** Create a reference to the underlying Tensor */
        Tensor<T, N, C> operator*() const;

        /** Syntatic sugar for (*it). */
        Tensor<T, N, C> operator->() const;

        /** Increment (postfix). Returns a temporary before increment */
        Iterator const operator++(int);

        /** Increment (prefix). Returns *this */
        Iterator &operator++();

        /** Decrement (postfix). Returns a temporary before decrement */
        Iterator const operator--(int);

        /** Decrement (prefix). Returns *this */
        Iterator &operator--();

        /** Returns true iff the shapes and underlying pointers are identical */
        bool operator==(Iterator const &it) const;

        /** Returns true iff the shapes or underlying pointers are not identical */
        bool operator!=(Iterator const &it) const
        { return !(it == *this); }

        /** Returns true iff the offset difference is positive */
        bool operator>(Iterator const &it) const noexcept;

        /** Returns true iff the offset difference is non-negative */
        bool operator>=(Iterator const &it) const noexcept;

        /** Returns true iff the offset difference is negative */
        bool operator<(Iterator const &it) const noexcept
        { return !(*this >= it); }

        /** Returns true iff the offset difference is non-positive */
        bool operator<=(Iterator const &it) const noexcept
        { return !(*this > it); }

        /** Returns the difference in offsets */
        std::ptrdiff_t operator-(Iterator const &it) const noexcept;

        /** Increments the iterator by `incr` elements */
        Iterator operator+(std::ptrdiff_t incr) const noexcept;

        /** Decrements the iterator by `decr` elements */
        Iterator operator-(std::ptrdiff_t decr) const noexcept;

    private:
        // Direct construction
        Iterator(Tensor<T, N + 1, C> const &tensor, size_t index);

        // Data describing the underlying tensor
        Shape<N> shape_;
        std::array<size_t, N> strides_;
        size_t offset_;
        C<T> ref_;
        // Step size of the underlying data pointer per increment
        size_t stride_;
    };

    class ConstIterator { /*@ConstIterator<T, N>*/
    public:
        /** Constant iterator with freedom across one dimension of a Tensor.
         *  Does not allow write access to the underlying tensor data.
         */

        /* ------------- Iterator Traits -------------- */

        typedef Tensor<T, N, C> const value_type;
        typedef Tensor<T, N, C> const &reference;
        typedef Tensor<T, N, C> const *pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

        /* -------------- Friend Classes -------------- */

        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        /* --------------- Constructors --------------- */

        /** Copy construct an iterator to the same underlying Tensor */
        ConstIterator(ConstIterator const &it);

        ConstIterator &operator=(ConstIterator const &it);

        /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
        ConstIterator(ConstIterator &&it) noexcept;

        ConstIterator &operator=(ConstIterator &&it) noexcept;

        /** Create a reference to the underlying Tensor */
        Tensor<T, N, C> const operator*() const;

        /** Syntatic sugar for (*it). */
        Tensor<T, N, C> const operator->() const;

        /** Increment (postfix). Returns a temporary before increment */
        ConstIterator const operator++(int);

        /** Increment (prefix). Returns *this */
        ConstIterator &operator++();

        /** Decrement (postfix). Returns a temporary before decrement */
        ConstIterator const operator--(int);

        /** Decrement (prefix). Returns *this */
        ConstIterator &operator--();

        bool operator==(ConstIterator const &it) const;

        bool operator!=(ConstIterator const &it) const
        { return !(it == *this); }

        /** Returns true iff the offset difference is positive */
        bool operator>(ConstIterator const &it) const noexcept;

        /** Returns true iff the offset difference is non-negative */
        bool operator>=(ConstIterator const &it) const noexcept;

        /** Returns true iff the offset difference is negative */
        bool operator<(ConstIterator const &it) const noexcept
        { return !(*this >= it); }

        /** Returns true iff the offset difference is non-positive */
        bool operator<=(ConstIterator const &it) const noexcept
        { return !(*this > it); }

        /** Returns the difference in offsets */
        std::ptrdiff_t operator-(ConstIterator const &it) const noexcept;

        /** Increments the iterator by `incr` elements */
        ConstIterator operator+(std::ptrdiff_t incr) const noexcept;

        /** Decrements the iterator by `decr` elements */
        ConstIterator operator-(std::ptrdiff_t decr) const noexcept;

    private:
        ConstIterator(Tensor<T, N + 1, C> const &tensor, size_t index);

        // Data describing the underlying tensor
        Shape<N> shape_;
        std::array<size_t, N> strides_;
        size_t offset_;
        C<T> ref_;

        // Step size of the underlying data pointer per increment size_t stride_;
        size_t stride_;
    };

    class ReverseIterator { /*@ReverseIterator<T, N>*/
    public:
        /** Reverse iterator with freedom across one dimension of a Tensor.
         *  Does not allow write access to the underlying tensor data.
         */

        /* ------------- Iterator Traits -------------- */

        typedef Tensor<T, N, C> value_type;
        typedef Tensor<T, N, C> &reference;
        typedef Tensor<T, N, C> *pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

        /* -------------- Friend Classes -------------- */

        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        /* --------------- Constructors --------------- */

        /** Copy construct an iterator to the same underlying Tensor */
        ReverseIterator(ReverseIterator const &it);

        ReverseIterator &operator=(ReverseIterator const &it);

        /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
        ReverseIterator(ReverseIterator &&it) noexcept;

        ReverseIterator &operator=(ReverseIterator &&it) noexcept;

        /** Create a reference to the underlying Tensor */
        Tensor<T, N, C> operator*() const;

        /** Syntatic sugar for (*it). */
        Tensor<T, N, C> operator->() const;

        /** Increment (postfix). Returns a temporary before increment */
        ReverseIterator const operator++(int);

        /** Increment (prefix). Returns *this */
        ReverseIterator &operator++();

        /** Decrement (postfix). Returns a temporary before decrement */
        ReverseIterator const operator--(int);

        /** Decrement (prefix). Returns *this */
        ReverseIterator &operator--();

        bool operator==(ReverseIterator const &it) const;

        bool operator!=(ReverseIterator const &it) const
        { return !(it == *this); }

        /** Returns true iff the offset difference is positive */
        bool operator>(ReverseIterator const &it) const noexcept;

        /** Returns true iff the offset difference is non-negative */
        bool operator>=(ReverseIterator const &it) const noexcept;

        /** Returns true iff the offset difference is negative */
        bool operator<(ReverseIterator const &it) const noexcept
        { return !(*this >= it); }

        /** Returns true iff the offset difference is non-positive */
        bool operator<=(ReverseIterator const &it) const noexcept
        { return !(*this > it); }

        /** Returns the difference in offsets */
        std::ptrdiff_t operator-(ReverseIterator const &it) const noexcept;

        /** Increments the iterator by `incr` elements */
        ReverseIterator operator+(std::ptrdiff_t incr) const noexcept;

        /** Decrements the iterator by `decr` elements */
        ReverseIterator operator-(std::ptrdiff_t decr) const noexcept;

    private:
        ReverseIterator(Tensor<T, N + 1, C> const &tensor, size_t index);

        // Data describing the underlying tensor
        Shape<N> shape_;
        std::array<size_t, N> strides_;
        size_t offset_;
        C<T> ref_;

        // Step size of the underlying data pointer per increment
        size_t stride_;
    };

    class ConstReverseIterator { /*@ConstReverseIterator<T, N>*/
    public:
        /** Constant reverse iterator with freedom across one dimension of a Tensor.
         *  Does not allow write access to the underlying tensor data.
         */

        /* ------------- Iterator Traits -------------- */

        typedef Tensor<T, N, C> const value_type;
        typedef Tensor<T, N, C> const &reference;
        typedef Tensor<T, N, C> const *pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

        /* -------------- Friend Classes -------------- */

        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        /* --------------- Constructors --------------- */

        /** Copy constructs an iterator to the same underlying Tensor */
        ConstReverseIterator(ConstReverseIterator const &it);

        ConstReverseIterator &operator=(ConstReverseIterator const &it);

        /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
        ConstReverseIterator(ConstReverseIterator &&it) noexcept;

        ConstReverseIterator &operator=(ConstReverseIterator &&it) noexcept;

        /** Create a reference to the underlying Tensor */
        Tensor<T, N, C> const operator*() const;

        /** Syntatic sugar for (*it). */
        Tensor<T, N, C> const operator->() const;

        /** Increment (postfix). Returns a temporary before increment */
        ConstReverseIterator const operator++(int);

        /** Increment (prefix). Returns *this */
        ConstReverseIterator &operator++();

        /** Decrement (postfix). Returns a temporary before decrement */
        ConstReverseIterator const operator--(int);

        /** Decrement (prefix). Returns *this */
        ConstReverseIterator &operator--();

        bool operator==(ConstReverseIterator const &it) const;

        bool operator!=(ConstReverseIterator const &it) const
        { return !(it == *this); }

        /** Returns true iff the offset difference is positive */
        bool operator>(ConstReverseIterator const &it) const noexcept;

        /** Returns true iff the offset difference is non-negative */
        bool operator>=(ConstReverseIterator const &it) const noexcept;

        /** Returns true iff the offset difference is negative */
        bool operator<(ConstReverseIterator const &it) const noexcept
        { return !(*this >= it); }

        /** Returns true iff the offset difference is non-positive */
        bool operator<=(ConstReverseIterator const &it) const noexcept
        { return !(*this > it); }

        /** Returns the difference in offsets */
        std::ptrdiff_t operator-(ConstReverseIterator const &it) const noexcept;

        /** Increments the iterator by `incr` elements */
        ConstReverseIterator operator+(std::ptrdiff_t incr) const noexcept;

        /** Decrements the iterator by `decr` elements */
        ConstReverseIterator operator-(std::ptrdiff_t decr) const noexcept;

    private:
        ConstReverseIterator(Tensor<T, N + 1, C> const &tensor, size_t index);

        // Data describing the underlying tensor
        Shape<N> shape_;
        std::array<size_t, N> strides_;
        size_t offset_;
        C<T> ref_;

        // Step size of the underlying data pointer per increment
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

    template<typename U, size_t M, template<class> class C_>
    friend void Set(Tensor<U, M, C_> &tensor, Indices<M> const &indices, U &&value);

    /** Allocates a Tensor with shape `shape`, whose total number of elements
     *  must be equivalent to *this (or an assertion will fail during debug).
     *  The resulting Tensor is filled by iterating through *this and copying
     *  over the values.
     */
    template<size_t M, template<class> class C_ = self_type::container_type>
    Tensor<T, M, C_> reshape(Shape<M> const &shape) const noexcept;

    /** Returns a deep copy of this tensor, equivalent to calling copy constructor */
    Tensor<T, N, C> copy() const;

    /** Returns a reference: only used to invoke reference constructor */
    typename Tensor<T, N, C>::Proxy ref();

    template<typename U, typename FunctionType>
    U reduce(U &&initial_value, FunctionType &&fn) const;

    template<typename FunctionType, size_t... ReducedIndices>
    Tensor<T, N - sizeof...(ReducedIndices), C> reduce_dimensions(FunctionType &&fn) const;

    template<typename U, size_t M, template<class> class C_, typename RAIt>
    friend void Fill(Tensor<U, M, C_> &tensor, RAIt begin, RAIt const &end);

    template<typename U, size_t M, template<class> class C_, typename X>
    friend void Fill(Tensor<U, M, C_> &tensor, X const &value);

    template<typename U, template<class> class C_>
    friend Tensor<U, 2, C_> transpose(Tensor<U, 2, C_> &mat);

    template<typename U, template<class> class C_, typename X,
            template<class> class C__>
    friend Tensor<X, 2, C__> transpose(Tensor<U, 2, C_> const &mat);

    template<typename U, template<class> class C_>
    friend Tensor<U, 2, C_> transpose(Tensor<U, 1, C_> &vec);

    template<typename U, template<class> class C_, typename X,
            template<class> class C__>
    friend Tensor<X, 2, C__> transpose(Tensor<U, 1, C_> const &vec);

    template<typename U, template<class> class C_, typename FunctionType, 
             typename... Expressions>
    friend Tensor<U, FirstExpression<Expressions...>::type::rank(), C_>
    details::map(FunctionType &&fn, Expressions const &... exprs);

    template<typename FunctionType, typename... Tensors>
    friend void details::Map(FunctionType &&fn, Tensors &&... tensors);

    template<typename U, typename FunctionType, typename... Tensors>
    friend U details::reduce(U &&initial_value, FunctionType &&fn, Tensors const &... tensors);

private:
    /* ---------------------- Data ---------------------- */

    Shape<N> shape_;
    std::array<size_t, N> strides_;
    size_t offset_;
    C<T> ref_;

    /* -------------------- Getters -------------------- */

    std::array<size_t, N> const &strides() const noexcept { return strides_; }

    /* ----------- Expansion for operator()(...) ----------- */

    template<size_t... I, typename... Args>
    size_t pIndicesExpansion(meta::Sequence<I...>, Args... args) const;

    /* --------------- Expansion for slice() --------------- */

    template<size_t M>
    Tensor<T, N - M, C> pSliceExpansion(
            std::array<bool, N> &sliced_indices, Indices<M> const &indices);

    /* ------- Expansion for methods which take Tensors --------- */

    inline T const &pGet(size_t *indices, size_t index) const noexcept
    {
        return (static_cast<C<T> const &>(ref_))[offset_ + indices[index]];
    }

    inline T &pGet(size_t *indices, size_t index) { return ref_[offset_ + indices[index]]; }

    template<typename U>
    inline void pSet(size_t index, U &&value) { ref_.Set(index, std::forward<U>(value)); }

    template<typename U, typename FunctionType, typename Tuple, size_t... I>
    friend inline U details::map_forward_sequence(FunctionType &&fn, size_t *indices,
        Tuple &&tensors, meta::Sequence<I...>);

    template<typename FunctionType, typename Tuple, size_t... I>
    friend inline void details::MapForwardSequence(FunctionType &&fn, size_t *indices,
        Tuple &&tensors, meta::Sequence<I...>);

    template<typename U, typename FunctionType, typename Tuple, size_t... I> 
    friend inline void details::ReduceForwardSequence(U &&ret_val, FunctionType &&fn, 
        size_t *indices, Tuple &&tensors, meta::Sequence<I...>);

    template<typename U, size_t M, template<class> class C_, typename FunctionType,
            typename Tuple, size_t... I>
    friend inline void details::ElemWiseForwardSequence( 
        Tensor<U, M, C_> &tensor, size_t index, FunctionType &&fn, size_t *indices, 
        Tuple &&tensors, meta::Sequence<I...>);

    /* ------------------ Utility --------------------- */

    // Copy the dimensions of a C-array `Array` into `dimensions`
    template<typename Array, size_t Index, size_t Limit>
    struct SetCArrayDimensions {
        void operator()(std::array<size_t, N> &dimensions)
        {
            dimensions[Index] = std::extent<Array, Index>::value;
            SetCArrayDimensions<Array, Index + 1, Limit>{}(dimensions);
        }
    };

    // Base condition
    template<typename Array, size_t Limit>
    struct SetCArrayDimensions<Array, Limit, Limit> {
        void operator()(std::array<size_t, N> &)
        {}
    };

    // Asserts that the dimensions of C-array `Array`
    // are equivalent to `dimensions`
    template<typename Array, size_t Index, size_t Limit>
    struct AssertCArrayDimensions {
        void operator()(std::array<size_t, N> &dimensions)
        {
            assert((dimensions[Index] == std::extent<Array, Index>::value && DIMENSION_MISMATCH));
            AssertCArrayDimensions<Array, Index + 1, Limit>{}(dimensions);
        }
    };

    // Base condition
    template<typename Array, size_t Limit>
    struct AssertCArrayDimensions<Array, Limit, Limit> {
        void operator()(std::array<size_t, N> &) {}
    };

    // Declare all fields in the constructor, but initialize strides assuming no gaps
    Tensor(typename std::array<size_t, N>::const_iterator dimensions_begin, 
        typename std::array<size_t, N>::const_iterator dimensions_end, 
        size_t offset, C<T> const &_ref);

    Tensor(typename std::array<size_t, N>::const_iterator dimensions_begin, 
        typename std::array<size_t, N>::const_iterator dimensions_end, 
        size_t offset, C<T> &&_ref);

    // Declare all fields in the constructor
    Tensor(typename std::array<size_t, N>::const_iterator dimensions_begin,
        typename std::array<size_t, N>::const_iterator dimensions_end, 
        typename std::array<size_t, N>::const_iterator strides_begin, 
        typename std::array<size_t, N>::const_iterator strides_end, 
        size_t offset, C<T> &&_ref);

    Tensor(typename std::array<size_t, N>::const_iterator dimensions_begin, 
        typename std::array<size_t, N>::const_iterator dimensions_end, 
        typename std::array<size_t, N>::const_iterator strides_begin, 
        typename std::array<size_t, N>::const_iterator strides_end, 
        size_t offset, C<T> const &_ref);

}; // Tensor

/* ----------------------------- Constructors ------------------------- */

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(std::initializer_list<size_t> dimensions)
        : shape_(Shape<N>(dimensions)),
          offset_(0),
          ref_(shape_.index_product())
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    shape_.pInitializeStrides(strides_);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(Shape<N> const &shape)
        : shape_(shape),
          offset_(0),
          ref_(shape_.index_product())
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    for (size_t i = 0; i < N; ++i)
        assert(shape[i] && ZERO_ELEMENT);
    shape_.pInitializeStrides(strides_);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(Shape<N> const &shape, T const &value)
        : shape_(shape), offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    for (size_t i = 0; i < N; ++i)
        assert(shape[i] && ZERO_ELEMENT);
    shape_.pInitializeStrides(strides_);
    size_t cumul = shape_.index_product();
    ref_ = container_type<T>(cumul, value);
}

template<typename T, size_t N, template<class> class C>
template<typename FunctionType, typename... Arguments>
Tensor<T, N, C>::Tensor(
        Shape<N> const &shape,
        std::function<FunctionType> &f,
        Arguments &&...  args
) : shape_(shape), offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    for (size_t i = 0; i < N; ++i)
        assert(shape[i] && ZERO_ELEMENT);
    shape_.pInitializeStrides(strides_);
    auto value_setter =
            [&f, &args...](T &lhs) -> void { lhs = f(args...); };
    ref_ = container_type<T>(shape_.index_product());
    Map(value_setter, *this);
}

template<typename T, size_t N, template<class> class C>
template<typename Array>
Tensor<T, N, C>::Tensor(CArrayProxy<Array> &&md_array): offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    static_assert(std::rank<Array>::value == N, RANK_MISMATCH);
    using ArrayType = typename std::remove_all_extents<Array>::type;
    SetCArrayDimensions < Array, 0, N > {}(shape_.dimensions_);
    shape_.pInitializeStrides(strides_);
    // Make use of the fact C arrays are contiguously allocated
    auto *ptr = (ArrayType *) md_array.value;
    size_t cumul = shape_.index_product();
    ref_ = container_type<T>(shape_.index_product(), ptr, ptr + cumul);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(Tensor<T, N, C> const &tensor)
        : shape_(tensor.shape_), offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    shape_.pInitializeStrides(strides_);
    size_t cumul = shape_.index_product();
    ref_ = container_type<T>(cumul);
    size_t index = 0;
    auto indices = Indices<N>();
    do {
        pSet(index++, tensor[indices]);
    } while (indices.increment(tensor.shape()));
}

template<typename T, size_t N, template<class> class C>
template<typename U, template<class> class C_>
Tensor<T, N, C>::Tensor(Tensor<U, N, C_> const &tensor)
        : shape_(tensor.shape_), offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    shape_.pInitializeStrides(strides_);
    size_t cumul = shape_.index_product();
    ref_ = container_type<T>(cumul);
    size_t index = 0;
    auto indices = Indices<N>();
    do {
        pSet(index++, tensor[indices]);
    } while (indices.increment(tensor.shape()));
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(Tensor<T, N, C> &&tensor) noexcept
        : shape_(tensor.shape_), offset_(tensor.offset_), ref_(std::move(tensor.ref_))
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    strides_ = tensor.strides_;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(typename Tensor<T, N, C>::Proxy const &proxy)
        : shape_(proxy.tensor_.shape_), offset_(proxy.tensor_.offset_), ref_(proxy.tensor_.ref_)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    strides_ = proxy.tensor_.strides_;
}

template<typename T, size_t N, template<class> class C>
template<typename NodeType, typename>
Tensor<T, N, C>::Tensor(Expression<NodeType> const &rhs)
        : offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    auto const &expression = rhs.self();
    shape_ = expression.shape();
    shape_.pInitializeStrides(strides_);
    size_t cumul = shape_.index_product();
    ref_ = container_type<T>(cumul);
    size_t index = 0;
    auto indices = Indices<N>();
    do {
        pSet(index++, expression[indices]);
    } while (indices.increment(expression.shape()));
}

#ifdef _ENABLE_OPENCL

template<typename T, size_t N, template<class> class C>
template<typename NodeType>
Tensor<T, N, C>::Tensor(opencl::Model<NodeType> const &model)
        : offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    static_assert(N == NodeType::rank(), RANK_MISMATCH);
    shape_ = model.shape();
    shape_.pInitializeStrides(strides_);
    size_t cumul = shape_.index_product();
    auto *data = new T[cumul];
    model.template pFill<T>(data, cumul);
    ref_ = container_type<T>(cumul, data[0]);
}

#endif

/* ------------------------- Assignment ------------------------- */

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(Tensor<T, N, C> const &tensor)
{
    assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
    size_t cumul = shape_.index_product();
    auto reference_indices = Indices<N>{};
    auto indices = std::array<size_t, 2>{};
    size_t const *const strides[] = {this->strides_.data(), tensor.strides_.data()};
    for (size_t i = 0; i < cumul; ++i) {
        ref_.Set(indices[0] + offset_, tensor.pGet(indices.data(), 1));
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(Tensor<T, N, C> &&tensor) noexcept
{
    assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
    size_t cumul = shape_.index_product();
    auto reference_indices = Indices<N>{};
    auto indices = std::array<size_t, 2>{};
    size_t const *const strides[] = {this->strides_.data(), tensor.strides_.data()};
    for (size_t i = 0; i < cumul; ++i) {
        ref_.Set(indices[0] + offset_, std::move(tensor.pGet(indices.data(), 1)));
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }
    return *this;
}

template<typename T, size_t N, template<class> class C>
template<typename U, template<class> class C_>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(Tensor<U, N, C_> const &tensor)
{
    assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);

    size_t cumul = shape_.index_product();
    auto reference_indices = Indices<N>{};
    auto indices = std::array<size_t, 2>{};
    size_t const *const strides[2] = {this->strides_.data(), tensor.strides_.data()};
    for (size_t i = 0; i < cumul; ++i) {
        ref_.Set(indices[0] + offset_, tensor.pGet(indices.data(), 1));
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }

    return *this;
}

template<typename T, size_t N, template<class> class C>
template<typename U, template<class> class C_>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(Tensor<U, N, C_> &&tensor)
{
    assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);

    size_t cumul = shape_.index_product();
    auto reference_indices = Indices<N>{};
    auto indices = std::array<size_t, 2>{};
    size_t const *const strides[2] = {this->strides_.data(), tensor.strides_.data()};
    for (size_t i = 0; i < cumul; ++i) {
        ref_.Set(indices[0] + offset_, std::move(tensor.pGet(indices.data(), 1)));
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }

    return *this;
}

template<typename T, size_t N, template<class> class C>
template<typename Array>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(CArrayProxy<Array> &&md_array)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    static_assert(std::rank<Array>::value == N, RANK_MISMATCH);
    AssertCArrayDimensions < Array, 0, N > {}(shape_.dimensions_);

    // Make use of the fact C arrays are contiguously allocated
    using ArrayType = typename std::remove_all_extents<Array>::type;
    auto *ptr = (ArrayType *) md_array.value;
    auto reference_indices = Indices<N>{};
    auto indices = std::array<size_t, 1>{};
    size_t const *const strides[1] = {strides_.data()};
    for (size_t i = 0; i < shape_.index_product(); ++i) {
        ref_.Set(indices[0] + offset_, *(ptr++));
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }

    return *this;
}

template<typename T, size_t N, template<class> class C>
template<typename NodeType>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(Expression<NodeType> const &rhs)
{
    auto const &expression = rhs.self();
    assert((shape_ == expression.shape()) && DIMENSION_MISMATCH);

    // Allocate and assign into a new tensor to prevent aliasing errors
    auto tensor = Tensor<T, N, C>(this->shape());
    auto reference_indices = Indices<N>{};
    auto indices = std::array<size_t, 1>{};
    size_t const *const strides[1] = {tensor.strides_.data()};

    // evaluate every element and assign it in place
    for (size_t i = 0; i < shape_.index_product(); ++i) {
        tensor.ref_.Set(indices[0] + offset_, expression[reference_indices]);
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }

    *this = tensor;
    return *this;
}

template<typename T, size_t N, template<class> class C>
template<typename NodeType>
Tensor<T, N, C> &Tensor<T, N, C>::operator=(NoAliasProxy<NodeType> const &rhs)
{
    auto const &expression = rhs.expr.self();
    assert((shape_ == expression.shape()) && DIMENSION_MISMATCH);

    auto reference_indices = Indices<N>{};
    auto indices = std::array<size_t, 1>{};
    size_t const *const strides[1] = {strides_.data()};
    for (size_t i = 0; i < shape_.index_product(); ++i) {
        ref_.Set(indices[0] + offset_, expression[reference_indices]);
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }

    return *this;
}

/* ------------------------------- Getters ------------------------------- */

template<typename T, size_t N, template<class> class C>
T const &Tensor<T, N, C>::get(Indices<N> const &indices) const noexcept
{
    size_t cumulative_index = 0;
    for (size_t i = 0; i < N; ++i)
        cumulative_index += strides_[N - i - 1] * indices[N - i - 1];
    return (static_cast<C<T> const &>(ref_))[cumulative_index + offset_];
}

/* ------------------------------- Setters ------------------------------- */

template<typename T, size_t N, template<class> class C>
template<typename RAIt>
void Tensor<T, N, C>::Fill(RAIt begin, RAIt const &end)
{
    size_t diff = std::distance(begin, end);
    size_t cumul = shape_.index_product();
    assert((diff > 0 && cumul == diff) && ELEMENT_COUNT_MISMATCH);
    Indices<N> reference_indices{};
    auto indices = std::array<size_t, 1>{};
    size_t const *const strides[1] = {strides_.data()};
    for (size_t i = 0; i < shape_.index_product(); ++i) {
        ref_.Set(indices[0] + offset_, *(begin++));
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }
}

template<typename T, size_t N, template<class> class C>
template<typename U>
void Tensor<T, N, C>::Set(Indices<N> const &indices, U &&value)
{
    size_t cumulative_index = 0;
    for (size_t i = 0; i < N; ++i)
        cumulative_index += strides_[N - i - 1] * indices[N - i - 1];
    ref_.Set(cumulative_index + offset_, std::forward<U>(value));
}

/* ------------------------------- Access ------------------------------- */

template<typename T, size_t N, template<class> class C>
template<typename... Args>
Tensor<T, N - sizeof...(Args), C> Tensor<T, N, C>::at(Args... args)
{
    static_assert(N >= sizeof...(Args), TOO_MANY_INDICES);
    constexpr size_t M = sizeof...(args);
    size_t cumulative_index = pIndicesExpansion(
            typename meta::MakeIndexSequence<0, sizeof...(Args)>::sequence{},
            args...
    );

    return Tensor<T, N - M, C>(
            shape_.data() + M,
            strides_.data() + M,
            offset_ + cumulative_index,
            ref_
    );
}

template<typename T, size_t N, template<class> class C>
template<typename... Args, typename>
Tensor<T, N - sizeof...(Args), C> Tensor<T, N, C>::operator()(Args... args)
{
    static_assert(N > sizeof...(Args), TOO_MANY_INDICES);
    constexpr size_t M = sizeof...(args);
    size_t cumulative_index = pIndicesExpansion(
            typename meta::MakeIndexSequence<0, sizeof...(Args)>::sequence{},
            args...
    );

    return Tensor<T, N - M, C>(
            shape_.data() + M,
            strides_.data() + M,
            offset_ + cumulative_index,
            ref_
    );
}

template<typename T, size_t N, template<class> class C>
template<typename... Args, typename>
T &Tensor<T, N, C>::operator()(Args... args)
{
    static_assert(N == sizeof...(Args), PANIC_ASSERTION);
    size_t cumulative_index = pIndicesExpansion(
            typename meta::MakeIndexSequence<0, sizeof...(Args)>::sequence{},
            args...
    );

    return ref_[cumulative_index + offset_];
}

template<typename T, size_t N, template<class> class C>
template<typename... Args>
Tensor<T, N - sizeof...(Args), C> const Tensor<T, N, C>::at(Args... args) const
{
    return (*const_cast<self_type *>(this))(args...);
}

template<typename T, size_t N, template<class> class C>
template<typename... Args, typename>
Tensor<T, N - sizeof...(Args), C> const Tensor<T, N, C>::operator()(Args... args) const
{
    return (*const_cast<self_type *>(this))(args...);
}

template<typename T, size_t N, template<class> class C>
template<typename... Args, typename>
T const &Tensor<T, N, C>::operator()(Args... args) const
{
    static_assert(N == sizeof...(Args), PANIC_ASSERTION);
    size_t cumulative_index = pIndicesExpansion(
            typename meta::MakeIndexSequence<0, sizeof...(Args)>::sequence{},
            args...
    );
    return (static_cast<C<T> const &>(ref_))[cumulative_index + offset_];
}

template<typename T, size_t N, template<class> class C>
template<size_t M, typename>
Tensor<T, N - M, C> Tensor<T, N, C>::operator[](Indices<M> const &indices)
{
    static_assert(N > M, TOO_MANY_INDICES);
    size_t cumulative_index = 0;
    for (size_t i = 0; i < M; ++i)
        cumulative_index += strides_[M - i - 1] * indices[M - i - 1];

    return Tensor<T, N - M, C>(
            shape_.data() + M,
            strides_.data() + M,
            offset_ + cumulative_index,
            ref_
    );
}

template<typename T, size_t N, template<class> class C>
template<size_t M, typename>
T &Tensor<T, N, C>::operator[](Indices<M> const &indices)
{
    static_assert(N == M, PANIC_ASSERTION);
    size_t cumulative_index = 0;
    for (size_t i = 0; i < N; ++i)
        cumulative_index += strides_[N - i - 1] * indices[N - i - 1];
    return ref_[cumulative_index + offset_];
}

template<typename T, size_t N, template<class> class C>
template<size_t M, typename>
Tensor<T, N - M, C> const Tensor<T, N, C>::operator[](Indices<M> const &indices) const
{
    return (*const_cast<self_type *>(this))[indices];
}

template<typename T, size_t N, template<class> class C>
template<size_t M, typename>
T const &Tensor<T, N, C>::operator[](Indices<M> const &indices) const
{
    static_assert(N == M, PANIC_ASSERTION);
    size_t cumulative_index = 0;
    for (size_t i = 0; i < N; ++i)
        cumulative_index += strides_[N - i - 1] * indices[N - i - 1];

    return (static_cast<C<T> const &>(ref_))[cumulative_index + offset_];
}

template<typename U, size_t M, template<class> class C_>
void Set(Tensor<U, M, C_> &tensor, Indices<M> const &indices, U &&value)
{
    static_assert(M, PANIC_ASSERTION);
    size_t cumulative_index = tensor.offset_;
    for (size_t i = 0; i < M; ++i)
        cumulative_index += tensor.strides_[M - i - 1] * indices[M - i - 1];

    tensor.ref_.Set(cumulative_index, std::forward<U>(value));
}

template<typename T, size_t N, template<class> class C>
template<size_t... Slices, typename... Args, typename>
Tensor<T, N - sizeof...(Args), C> Tensor<T, N, C>::slice(Args... args)
{
    static_assert(N >= sizeof...(Slices) + sizeof...(args), SLICES_OUT_OF_BOUNDS);
    auto sliced_indices = std::array<bool, N>{};

    // place `true` where the indices are sliced, `false` o.w.
    std::fill(sliced_indices.begin(), sliced_indices.end(), false);
    meta::InsertValueIntoArray<bool, N, Slices...>{sliced_indices, true};

    // slice dimensions (aka set to `true`) not explicitly sliced nor filled
    details::FillFirstUncoveredN<bool, N>(sliced_indices, sliced_indices,
            N - sizeof...(Args) - sizeof...(Slices), true);

    // expand `args` into Indices and forward both
    auto indices = Indices<sizeof...(args)>{(size_t) args...};
    return pSliceExpansion<sizeof...(args)>(sliced_indices, indices);
}

template<typename T, size_t N, template<class> class C>
template<size_t... Slices, typename... Args, typename>
Tensor<T, N - sizeof...(Args), C> const Tensor<T, N, C>::slice(Args... indices) const
{
    return const_cast<self_type *>(this)->slice<Slices...>(indices...);
}

template<typename T, size_t N, template<class> class C>
template<size_t... Slices, typename... Args, typename>
T &Tensor<T, N, C>::slice(Args... args)
{
    static_assert(sizeof...(Args) == N, PANIC_ASSERTION);
    static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
    return (*this)(args...);
}

template<typename T, size_t N, template<class> class C>
template<size_t... Slices, typename... Args, typename>
T const &Tensor<T, N, C>::slice(Args... args) const
{
    static_assert(sizeof...(Args) == N, PANIC_ASSERTION);
    static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
    return (*this)(args...);
}

template<typename T, size_t N, template<class> class C>
template<size_t... Slices, size_t M, typename>
Tensor<T, N - M, C> Tensor<T, N, C>::slice(Indices<M> const &indices)
{
    static_assert(N >= M + sizeof...(Slices), SLICES_OUT_OF_BOUNDS);
    auto sliced_indices = std::array<bool, N>{};

    // place `true` where the indices are sliced, `false` o.w.
    std::fill(sliced_indices.begin(), sliced_indices.end(), false);
    meta::InsertValueIntoArray<bool, N, Slices...>{sliced_indices, true};

    // slice dimensions (aka set `true`) not explicitly sliced or filled
    details::FillFirstUncoveredN<bool, N>(sliced_indices, sliced_indices,
            N - M - sizeof...(Slices), true);
    return pSliceExpansion<M>(sliced_indices, indices);
}

template<typename T, size_t N, template<class> class C>
template<size_t... Slices, size_t M, typename>
Tensor<T, N - M, C> const
Tensor<T, N, C>::slice(Indices<M> const &indices) const
{
    return const_cast<self_type *>(this)->slice<Slices...>(indices);
}

template<typename T, size_t N, template<class> class C>
template<size_t... Slices, size_t M, typename>
T &Tensor<T, N, C>::slice(Indices<M> const &indices)
{
    static_assert(M == N, PANIC_ASSERTION);
    static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
    return (*this)[indices];
}

template<typename T, size_t N, template<class> class C>
template<size_t... Slices, size_t M, typename>
T const &Tensor<T, N, C>::slice(Indices<M> const &indices) const
{
    static_assert(M == N, PANIC_ASSERTION);
    static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
    return (*this)[indices];
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::operator==(Tensor<T, N, C> const &tensor) const
{
    assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
    size_t indices_product = shape_.index_product();
    for (size_t i = 0; i < indices_product; ++i)
        if (ref_[i + offset_] != tensor.ref_[i + tensor.offset_])
            return false;
    return true;
}

template<typename T, size_t N, template<class> class C>
template<typename X>
bool Tensor<T, N, C>::operator==(Tensor<X, N, C> const &tensor) const
{
    assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
    size_t index_product = shape_.index_product();
    for (size_t i = 0; i < index_product; ++i)
        if (ref_[i + offset_] != tensor.ref_[i + tensor.offset_])
            return false;
    return true;
}

template<typename T, size_t N, template<class> class C>
std::string Tensor<T, N, C>::str() const
{
    auto s = std::string{};
    auto add_brackets = [&s](size_t n, bool is_left) -> void {
        for (size_t i = 0; i < n; ++i) { 
            s += (is_left ? '[' : ']');
        }
    };

    size_t cumulative_index = shape_.index_product();
    size_t index = 0;
    size_t dim_capacities[N] = {};
    std::copy_n(shape_.data(), N, dim_capacities);

    add_brackets(N, true);
    s += StringFormat<value_type>{}(ref_[offset_]);

    for (size_t i = 0; i < cumulative_index - 1; ++i) {
        size_t bracket_count = 0;
        bool propogate = true;
        int dim_index = N - 1;

        // find the correct index to "step" to
        while (dim_index >= 0 && propogate) {
            --dim_capacities[dim_index];
            ++bracket_count;
            index += strides_[dim_index];
            if (!dim_capacities[dim_index]) {
                dim_capacities[dim_index] = shape_[dim_index];
                index -= dim_capacities[dim_index] * strides_[dim_index];
            } else {
                propogate = false;
            }
            --dim_index;
        }

        // add closing brackets
        add_brackets(bracket_count - 1, false);
        s += ", ";
        add_brackets(bracket_count - 1, true);
        s += StringFormat<value_type>{}(ref_[index + offset_]);
    }

    // last set of closing brackets
    add_brackets(N, false);
    return s;
}

/* ----------- Expansion for operator()() ----------- */

template<typename T, size_t N, template<class> class C>
template<size_t... I, typename... Args>
size_t Tensor<T, N, C>::pIndicesExpansion(meta::Sequence<I...>, Args... args) const
{
    constexpr size_t M = sizeof...(Args);
    static_assert(N >= M, RANK_OUT_OF_BOUNDS);

    // scale indices with corresponding stride
    auto convert_index = [&](size_t dim, size_t index) -> size_t {
        assert((dim < shape_[index]) && INDEX_OUT_OF_BOUNDS);
        return strides_[index] * dim;
    };

    // sum reduce args after conversion
    size_t cumulative_index = 0;
    VARDIAC_MAP(cumulative_index += convert_index(args, I));
    (void) (convert_index);
    return cumulative_index;
}

/* ------------ Expansion for Slice() ------------ */

template<typename T, size_t N, template<class> class C>
template<size_t M>
Tensor<T, N - M, C>
Tensor<T, N, C>::pSliceExpansion(std::array<bool, N> &sliced_indices, Indices<M> const &indices)
{
    auto dimensions = std::array<size_t, N - M>{};
    auto strides = std::array<size_t, N - M>{};
    size_t offset = 0;
    size_t indices_i = 0;
    size_t tensor_i = 0;

    for (size_t i = 0; i < N; ++i) {
        if (!sliced_indices[i]) {
            assert((shape_[i] > indices[indices_i]) && INDEX_OUT_OF_BOUNDS);
            offset += indices[indices_i++] * strides_[i];
        } else {
            strides[tensor_i] = strides_[i];
            dimensions[tensor_i] = shape_[i];
            ++tensor_i;
        }
    }

    return Tensor<T, N - M, C>(
            dimensions.begin(), dimensions.end(), strides.begin(), strides.end(),
            offset_ + offset, ref_
    );
}

/* -------------- Utility Methods --------------- */

// private constructors
template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(
        typename std::array<size_t, N>::const_iterator dimensions_begin,
        typename std::array<size_t, N>::const_iterator dimensions_end,
        size_t offset,
        C<T> &&_ref
) : shape_(Shape<N>(dimensions_begin, dimensions_end)), offset_(offset), ref_(std::move(_ref))
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    shape_.pInitializeStrides(strides_);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(
        typename std::array<size_t, N>::const_iterator dimensions_begin,
        typename std::array<size_t, N>::const_iterator dimensions_end,
        size_t offset,
        C<T> const &_ref
) : shape_(Shape<N>(dimensions_begin, dimensions_end)), offset_(offset), ref_(_ref)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    shape_.pInitializeStrides(strides_);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(
        typename std::array<size_t, N>::const_iterator dimensions_begin,
        typename std::array<size_t, N>::const_iterator dimensions_end,
        typename std::array<size_t, N>::const_iterator strides_begin,
        typename std::array<size_t, N>::const_iterator strides_end,
        size_t offset,
        C<T> const &_ref
) : shape_(Shape<N>(dimensions_begin, dimensions_end)), offset_(offset), ref_(_ref)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    std::copy(strides_begin, strides_end, strides_.begin());
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Tensor(
        typename std::array<size_t, N>::const_iterator dimensions_begin,
        typename std::array<size_t, N>::const_iterator dimensions_end,
        typename std::array<size_t, N>::const_iterator strides_begin,
        typename std::array<size_t, N>::const_iterator strides_end,
        size_t offset,
        C<T> &&_ref
)  : shape_(Shape<N>(dimensions_begin, dimensions_end)), offset_(offset), ref_(std::move(_ref))
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    std::copy(strides_begin, strides_end, strides_.begin());
}

/* ------------------- OpenCL --------------------- */

#ifdef _ENABLE_OPENCL

template<typename T, size_t N, template<class> class C>
std::string Tensor<T, N, C>::opencl_kernel_code(
        cl::CommandQueue &,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    // Allocate all of the elements in a single contiguous buffer
    // multiple writes to the device aren't necessary
    size_t cumul = shape().index_product();
    auto reference_indices = Indices<N>{};
    auto indices = std::array<size_t, 1>{};
    size_t const *const strides[1] = {this->strides_.data()};
    auto *data = new T[cumul];

    for (size_t i = 0; i < cumul; ++i) {
        data[i] = pGet(indices.data(), 0);
        details::UpdateIndices(reference_indices, this->shape_, indices, strides);
    }

    // Copy the preallocated memory into a single OpenCL buffer
    cl_int err = 0;
    buffers.emplace_back(opencl::Info::v().context(),
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
            cumul * sizeof(T), data, &err);
    assert((err == CL_SUCCESS) && OPENCL_BUFFER_ERROR);
    delete[] data;

    // WARNING -- using namespace
    using namespace opencl::details;

    // add the new tensor to the argument
    size_t arg_index = buffers.size() - 1;
    arg_list += std::string(cGlobalIdentifier) + " " + (OpenCLType<T>::value) + " "
                + cConstIdentifier + " " + cPointerIdentifier + cVariablePrefix
                + std::to_string(arg_index) + ", ";

    // return the name of the element
    return cVariablePrefix + std::to_string(arg_index) + "[" + cGlobalIdName + "]";
}

template<typename T, size_t N, template<class> class C>
cl::Buffer Tensor<T, N, C>::opencl_evaluate_kernel(cl::CommandQueue &, 
    std::vector<cl::Buffer> &buffers, std::string &, std::string &) const noexcept
{
    return buffers.back();
}

#endif

/* -------------------------- Expressions -------------------------- */

/** Elementwise Scalar-Tensor operation. Returns a Tensor where each element
 *  of the new Tensor is the result of `fn` applied to the corresponding elements
 *  in each provided Tensor. Asserts that all of the provided Tensors have
 *  the same shape.
 */
template<typename FunctionType, typename... Tensors>
auto elemwise(FunctionType &&fn, Tensors &&... tensors)
-> typename FirstTensor<Tensors...>::type
{
    using return_type = typename FirstTensor<Tensors...>::type;
    static_assert(sizeof...(tensors), NO_TENSORS_PROVIDED);
    auto const &shape = details::GetShape(tensors...);
    constexpr size_t M = std::remove_reference<decltype(shape)>::type::rank();
    VARDIAC_MAP(assert(shape == tensors.shape() && SHAPE_MISMATCH));

    auto result = return_type(shape);
    size_t cumulative_index = shape.index_product();
    auto reference_indices = Indices<M>{};
    auto indices = std::array<size_t, sizeof...(Tensors) + 1>{};
    size_t const *const strides[sizeof...(Tensors) + 1] = {
            result.strides_.data(),
            tensors.strides_.data()...
    };

    auto sequence = typename meta::MakeIndexSequence<0, sizeof...(Tensors)>::sequence{};

    for (size_t i = 0; i < cumulative_index; ++i) {
        details::ElemWiseForwardSequence(
                result, indices[0], fn, indices.data() + 1,
                std::forward_as_tuple(tensors...), sequence
        );
        details::UpdateIndices(reference_indices, shape, indices, strides);
    }
    return result;
}

/** Creates a Tensor whose elements are the elementwise sum of `lhs`
 *  and `rhs`. `lhs` and `rhs` must have equivalent shape, or
 *  an assertion will fail during debug. The value type and the container
 *  type of the resulting tensor can be supplied as template arguments,
 *  and will default to copying the types of `LHS`.
 */
template<typename LHS, typename RHS, typename X = typename LHS::value_type,
        template<class> class C_ = LHS::template container_type>
Tensor<X, LHS::rank(), C_> add(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    assert((lhs.self().shape() == rhs.self().shape()) && DIMENSION_MISMATCH);
    Tensor<X, LHS::rank(), C_> result(lhs.self().shape());

    using Y = typename LHS::value_type;
    using Z = typename RHS::value_type;

    auto fn = [](X &x, Y const &y, Z const &z) -> void { x = y + z; };
    Map(fn, result, lhs.self(), rhs.self());

    return result;
}

/** Creates a Tensor whose elements are the elementwise sum of `lhs`, and every
 *  In debug builds, expression in `exprs...`, which must all have equivalent shape,
 *  or an assertion will fail.
 */
template<typename LHS, typename... Expressions,
        typename = typename std::enable_if<sizeof...(Expressions) >= 2>::type>
Tensor<typename LHS::value_type, LHS::rank(), LHS::template container_type>
add(Expression<LHS> const &lhs, Expressions const &... exprs)
{
    static_assert(meta::AreExpressions<Expressions...>::value, EXPECTING_EXPRESSION);

    auto result =
            Tensor<typename LHS::value_type, LHS::rank(), LHS::template container_type>(lhs);

    VARDIAC_MAP(assert(result.shape() == exprs.shape() && SHAPE_MISMATCH));
    VARDIAC_MAP(result = add(result, exprs));

    return result;
}

/** In place addition. The elements of `tensor` are assigned the elementwise sum of `rhs`
 *  and `tensor`. In debug builds, `tensor` and `rhs` must have equivalent shape, or
 *  an assertion will fail.
 */
template<typename X, size_t M, template<class> class C_, typename RHS>
void Add(Tensor<X, M, C_> &tensor, Expression<RHS> const &rhs)
{
    assert((tensor.shape() == rhs.shape()) && DIMENSION_MISMATCH);
    using Y = typename RHS::value_type;
    auto elemwise_add = [](X &x, Y const &y) -> void { x += y; };
    Map(elemwise_add, tensor, rhs.self());
}

/** In place addition with a variable amount of arguments */
template<typename X, size_t M, template<class> class C_, typename... Expressions,
        typename = typename std::enable_if<sizeof...(Expressions) >= 2>::type>
void Add(Tensor<X, M, C_> &tensor, Expressions const &... exprs)
{
    static_assert(meta::AreExpressions<Expressions...>::value, EXPECTING_EXPRESSION);
    auto const &shape = tensor.shape();
    VARDIAC_MAP(assert(shape = exprs.shape() && SHAPE_MISMATCH));
    VARDIAC_MAP(Add(tensor, exprs));
}

template<typename T, size_t N, template<class> class C>
template<typename RHS>
Tensor<T, N, C> &Tensor<T, N, C>::operator+=(Expression<RHS> const &rhs)
{
    auto tensor = rhs.self()();
    assert((shape_ == tensor.shape_) && DIMENSION_MISMATCH);
    *this = *this + tensor;
    return *this;
}

/** Creates a Tensor whose elements are the elementwise difference of `lhs`
 *  and `rhs`. `lhs` and `rhs` must have equivalent shape, or
 *  an assertion will fail during debug. The value type and the container
 *  type of the resulting tensor can be supplied as template arguments,
 *  and will default to copying the types of `LHS`.
 */
template<typename LHS, typename RHS, typename X = typename LHS::value_type,
        template<class> class C_ = LHS::template container_type>
Tensor<X, LHS::rank(), C_> sub(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    assert((lhs.self().shape() == rhs.self().shape()) && DIMENSION_MISMATCH);
    Tensor<X, LHS::rank(), C_> result(lhs.self().shape());

    using Y = typename LHS::value_type;
    using Z = typename RHS::value_type;

    auto fn = [](X &x, Y const &y, Z const &z)
            -> void { x = y - z; };
    Map(fn, result, lhs.self(), rhs.self());
    return result;
}

/** Creates a Tensor whose elements are the elementwise difference
 * of `lhs`, and every expression in `exprs...`, which must all have
 * equivalent shape, or an assertion will fail during debug.
 */
template<typename LHS, typename... Expressions,
         typename = typename std::enable_if<sizeof...(Expressions) >= 2>::type>
Tensor<typename LHS::value_type, LHS::rank(), LHS::template container_type>
sub(Expression<LHS> const &lhs, Expressions const &... exprs)
{
    static_assert(meta::AreExpressions<Expressions...>::value, EXPECTING_EXPRESSION);

    auto result = 
      Tensor<typename LHS::value_type, LHS::rank(), LHS::template container_type>(lhs.self());

    VARDIAC_MAP(assert(result.shape() == exprs.shape() && SHAPE_MISMATCH));
    VARDIAC_MAP(result = sub(result, exprs));

    return result;
}

/** The elements of `tensor` are assigned the elementwise difference of
 *  `tensor` and `rhs`. `tensor` and `rhs` must have equivalent shape, or
 *  an assertion will fail during debug.
 */
template<typename X, size_t M, template<class> class C_, typename RHS>
void Sub(Tensor<X, M, C_> &tensor, Expression<RHS> const &rhs)
{
    assert((tensor.shape() == rhs.shape()) && DIMENSION_MISMATCH);

    using Y = typename RHS::value_type;
    auto fn = [](X &x, Y const &y)
            -> void { x -= y; };
    Map(fn, tensor, rhs.self());
}

/** In place subtraction with a variable amount of arguments */
template<typename X, size_t M, template<class> class C_, typename... Expressions,
        typename = typename std::enable_if<sizeof...(Expressions) >= 2>::type>
void Sub(Tensor<X, M, C_> &tensor, Expressions const &... exprs)
{
    auto const &shape = tensor.shape();
    VARDIAC_MAP(assert(shape = exprs.shape() && SHAPE_MISMATCH));
    VARDIAC_MAP(Sub(tensor, exprs));
}

template<typename T, size_t N, template<class> class C>
template<typename RHS>
Tensor<T, N, C> &Tensor<T, N, C>::operator-=(Expression<RHS> const &rhs)
{
    auto tensor = rhs.self()();
    assert((shape_ == tensor.shape_) && SHAPE_MISMATCH);
    *this = *this - tensor;
    return *this;
}

/** Creates a Tensor whose elements are the elementwise product of `lhs`
 *  and `rhs`. `lhs` and `rhs` must have equivalent shape, or
 *  an assertion will fail during debug. The value type and the container
 *  type of the resulting tensor can be supplied as template arguments,
 *  and will default to copying the types of `LHS`.
 */
template<typename LHS, typename RHS, typename X = typename LHS::value_type,
        template<class> class C_ = LHS::template container_type>
Tensor<X, LHS::rank(), C_> hadamard(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    assert((lhs.self().shape() == rhs.self().shape()) && DIMENSION_MISMATCH);
    using Y = typename LHS::value_type;
    using Z = typename RHS::value_type;

    auto result = Tensor<X, LHS::rank(), C_>(lhs.self().shape());
    auto fn = [](X &x, Y const &y, Z const &z) -> void { x = y * z; };
    Map(fn, result, lhs.self(), rhs.self());

    return result;
}

/** Creates a Tensor whose elements are the elementwise product of `lhs`, and every
 *  expression in `exprs...`, which must all have equivalent shape,
 *  or an assertion will fail in debug builds.
 */
template<typename LHS, typename... Expressions,
         typename = typename std::enable_if<sizeof...(Expressions) >= 2>::type>
Tensor<typename LHS::value_type, LHS::rank(), LHS::template container_type>
hadamard(Expression<LHS> const &lhs, Expressions const &... exprs)
{
    auto result = Tensor<typename LHS::value_type, LHS::rank(), LHS::template container_type>(lhs);

    VARDIAC_MAP(assert(result.shape() == exprs.shape() && SHAPE_MISMATCH));
    VARDIAC_MAP(result = hadamard(result, exprs));

    return result;
}

/** The elements of `tensor` are assigned the elementwise product of
 *  `tensor` and `rhs`. `tensor` and `rhs` must have equivalent shape, or
 *  an assertion will fail during debug.
 */
template<typename X, size_t M, template<class> class C_, typename RHS>
void Hadamard(Tensor<X, M, C_> &tensor, Expression<RHS> const &rhs)
{
    assert((tensor.shape() == rhs.self().shape()) && DIMENSION_MISMATCH);

    using Y = typename RHS::value_type;

    auto fn = [](X &x, Y const &y) -> void { x *= y; };
    Map(fn, tensor, rhs.self());
}

/** Inplace hadamard with a variable amount of arguments */
template<typename X, size_t M, template<class> class C_, typename... Expressions,
         typename = typename std::enable_if<sizeof...(Expressions) >= 2>::type>
void Hadamard(Tensor<X, M, C_> &tensor, Expressions const &... exprs)
{
    auto const &shape = tensor.shape();
    VARDIAC_MAP(assert(shape = exprs.shape() && SHAPE_MISMATCH));
    VARDIAC_MAP(Hadamard(tensor, exprs));
}

/** Produces a Tensor which is the Tensor product of `tensor_1` and
 *  `tensor_2`. The inner dimensions of `tensor_1` and `tensor_2` must match, or
 *  std::logic error is thrown. The axis facing the multiplication can
 *  be given as template arguments: `I1` is the facing axis of the LHS,
 *  and `I2` of the right hand side. If not specified, I1 is `M1 - 1` and I2 is `0`.
 *  Tensor multiplication is equivalent to matrix multiplication scaled to higher
 *  dimensions, i.e. with I1 = 2, I2 = 0: shapes 2x3x4 * 4x3x2 -> 2x3x3x2
 *  Note: VERY EXPENSIVE, the time complexity to produce a N rank() Tensor
 *  with all dimensions equal to `m` is O(m^(N+1)).
 */
template<size_t I1, size_t I2, typename LHS, typename RHS,
        typename X = typename LHS::value_type,
        template<class> class C_ = LHS::template container_type>
Tensor<X, LHS::rank() + RHS::rank() - 2, C_>
mul(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    assert(lhs.self().dimension(I1) == lhs.self().dimension(I1) && SHAPE_MISMATCH);
    return details::mul<X, C_, I1, I2>(lhs.self().eval(), rhs.self().eval());
}

/** Tensor multiplication with default axis, (defaults to M1 - 1 facing 0) */
template<typename LHS, typename RHS, typename X = typename LHS::value_type,
        template<class> class C_ = LHS::template container_type>
inline Tensor<X, LHS::rank() + RHS::rank() - 2, C_> mul(Expression<LHS> const &lhs, 
    Expression<RHS> const &rhs)
{
    return mul<LHS::rank() - 1, 0>(lhs.self(), rhs.self());
}

/** Tensor multiplication with variable number of tensor arguments */
template<typename LHS, typename RHS, typename... Expressions,
        typename = typename std::enable_if<(sizeof...(Expressions) >= 1)>::type>
auto mul(Expression<LHS> const &lhs, Expression<RHS> const &rhs, Expressions const &... exprs)
-> Tensor<typename LHS::value_type,
        LHS::rank() + RHS::rank() + meta::RankSum<Expressions...>::value
        - 2 * (sizeof...(Expressions) + 1), LHS::template container_type>
{
    static_assert(meta::AreExpressions<Expressions...>::value, EXPECTING_EXPRESSION);
    auto prod_tensor = mul(lhs, rhs);
    return mul(prod_tensor, exprs...);
}

template<typename T, size_t N, template<class> class C>
template<typename RHS>
Tensor<T, N, C> &Tensor<T, N, C>::operator*=(Expression<RHS> const &rhs)
{
    auto tensor = rhs.self()();
    *this = *this * tensor;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> Tensor<T, N, C>::neg() const
{
    Tensor<T, N, C> neg_tensor(shape_);
    auto neg = [](T &x, T const &y) -> void { x = -y; };
    Map(neg, neg_tensor, *this);
    return neg_tensor;
}

/** Allocates a Tensor with shape equivalent to `tensor`, and whose
 *  elements are equivalent to `tensor` with operator-() applied.
 */
template<typename U, size_t M, template<class> class C_>
Tensor<U, M, C_> neg(Tensor<U, M, C_> const &tensor)
{
    auto neg_tensor = Tensor<U, M, C_>(tensor.shape_);
    auto neg = [](U &x, U const &y) -> void { x = -y; };
    Map(neg, neg_tensor, tensor);
    return neg_tensor;
}

template<typename T, size_t N, template<class> class C>
template<size_t I1, size_t I2>
Tensor<T, N - (I2 - I1), C> Tensor<T, N, C>::flatten()
{
    static_assert(I1 < I2, FLATTEN_INDICES_ERROR);
    static_assert(I2 < N, FLATTEN_INDICES_ERROR);
    auto dimensions = std::array<size_t, N - (I2 - I1)>{};
    auto strides = std::array<size_t, N - (I2 - I1)>{};
    std::copy_n(shape_.data(), I1, dimensions.data());
    std::copy_n(shape_.data() + I2 + 1, N - I2 - 1, dimensions.data() + I1 + 1);
    std::copy_n(strides_.data(), I1, strides.data());
    std::copy_n(strides_.data() + I2 + 1, N - I2 - 1, strides.data() + I1 + 1);

    size_t compressed_dimension = std::accumulate(shape_.data() + I1, shape_.data() + I2 + 1,
            1, [](size_t accum, size_t n) { return accum * n; });

    dimensions[I1] = compressed_dimension;
    strides[I1] = strides_[I2];

    return Tensor<T, N - (I2 - I1), C>(dimensions.data(), strides.data(), offset_, ref_);
}

/* --------------------------- Useful Functions ------------------------- */

template<typename T, size_t N, template<class> class C>
template<size_t M, template<class> class C_>
Tensor<T, M, C_> Tensor<T, N, C>::reshape(Shape<M> const &shape) const noexcept
{
    assert(shape_.index_product() == shape.index_product() && ELEMENT_COUNT_MISMATCH);
    auto resized_tensor = Tensor<T, M, C_>(shape);
    auto indices = Indices<N>();
    size_t index = 0;

    // the new tensor is contiguous
    do {
        resized_tensor.pSet(index++, (*this)[indices]);
    } while (indices.increment(shape_));

    return resized_tensor;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> Tensor<T, N, C>::copy() const
{
    return Tensor<T, N, C>(*this);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::Proxy Tensor<T, N, C>::ref()
{
    return Proxy(*this);
}

/** Fills the elements of `tensor` with the elements between.
 *  `begin` and `end`, which must be random access iterators. The number
 *  elements between `begin` and `end` must be equivalent to the capacity of `tensor`.
 */
template<typename U, size_t M, template<class> class C_, typename RAIt>
void Fill(Tensor<U, M, C_> &tensor, RAIt begin, RAIt const &end)
{
    tensor.Fill(begin, end);
}

/** Assigns to each element in `Tensor` `value` */
template<typename U, size_t M, template<class> class C_, typename X>
void Fill(Tensor<U, M, C_> &tensor, X const &value)
{
    auto alloc = [&value](U &x) -> void { x = value; };
    Map(alloc, tensor);
}

/** Returns a transposed Matrix, sharing the same underlying data as `mat`. If
 *  the dimension of `mat` is [n, m], the dimensions of the resulting matrix will be
 *  [m, n]. Note: This is only applicable to 2-rank tensors.
 */
template<typename T, template<class> class C>
Tensor<T, 2, C> transpose(Tensor<T, 2, C> &mat)
{
    auto transposed_dimensions = std::array<size_t, 2>{mat.shape_[1], mat.shape_[0]};
    auto transposed_strides = std::array<size_t, 2>{mat.strides_[1], mat.strides_[0]};
    return Tensor<T, 2, C>(transposed_dimensions.begin(), transposed_dimensions.end(),
            transposed_strides.begin(), transposed_strides.end(), mat.offset_, mat.ref_);
}

/** See Tensor<T, 2, C> transpose(Tensor<T, 2, C> &) */
template<typename T, template<class> class C, typename U = T, template<class> class C_ = C>
Tensor<U, 2, C_> transpose(Tensor<T, 2, C> const &mat)
{
    auto mat_t = Tensor<U, 2, C_>(mat);
    std::swap(mat_t.shape_[0], mat_t.shape_[1]);
    std::swap(mat_t.strides_[0], mat_t.strides_[1]);
    return mat_t;
}

/** Returns a transposed Matrix, sharing the same underlying data as vec. If
 *  the dimension of `vec` is [n], the dimensions of the resulting matrix will be
 *  [1, n]. This is only applicable to 1-rank tensors.
 */
template<typename T, template<class> class C>
Tensor<T, 2, C> transpose(Tensor<T, 1, C> &vec)
{
    auto transposed_dimensions = std::array<size_t, 2>{1, vec.shape_[0]};
    auto transposed_strides = std::array<size_t, 2>{1, vec.strides_[0]};
    return Tensor<T, 2, C>(transposed_dimensions.data(), transposed_strides.data(),
            vec.offset_, vec.ref_);
}

/** See Tensor<T, 2, C> transpose(Tensor<T, 1, C> &) */
template<typename T, template<class> class C, typename U = T, template<class> class C_ = C>
Tensor<U, 2, C_> transpose(Tensor<T, 1, C> const &vec)
{
    auto vec_t = Tensor<U, 2, C_>({1, vec.dimension(0)});
    vec_t.Fill(vec.cbegin(), vec.cend());
    return vec_t;
}

/** Given accumulator `initial_value`, function `fn` *  apply `fn` to each element of `*this`
 *  `fn` must have arity equivalent `1` and each argument to `fn` must have be const ref or
 *  value qualified, and return value must be trivially convertible to `T`.
 *  Returns the accumulator.
 */
template<typename T, size_t N, template<class> class C>
template<typename U, typename FunctionType>
U Tensor<T, N, C>::reduce(U &&initial_value, FunctionType &&fun) const
{
    auto ret_val = std::forward<U>(initial_value);
    auto accum = [&](T const &x) { ret_val = fun(ret_val, x); };
    Map(accum, *this);
    return ret_val;
}

/** Given an expression, reduces the expression along the specified dimensions `Indices`.
 *  The provided function `fn` must have arity equivalent to `sizeof...(Indices)`.
 */
template<typename T, size_t N, template<class> class C>
template<typename FunctionType, size_t... ReducedIndices>
Tensor<T, N - sizeof...(ReducedIndices), C> 
Tensor<T, N, C>::reduce_dimensions(FunctionType &&fn) const
{
  return details::reduce_dimensions<FunctionType, Tensor<T, N, C>, ReducedIndices...>(
      std::forward<FunctionType>(fn), *this);
}

/** Given a function `fn` and a variable sequence of Expressions, `exprs`,
 *  apply the function to the elements of `exprs` and return
 *  a new tensor of type, with shape equivalent to `Expressions`
 *  `fn` must have arity equivalent to `sizeof...(exprs)` and each argument
 *  to `fn` must have be const ref or value qualified. The return value
 *  must be trivially convertible to `T`.
 */
template<
        typename U, template<class> class C_, typename FunctionType,
        typename... Expressions
>
Tensor<U, FirstExpression<Expressions...>::type::rank(), C_>
map(FunctionType &&fn, Expressions const &... exprs)
{
    // verify arguments are expressions
    static_assert(sizeof...(Expressions), NO_EXPRESSIONS_PROVIDED);

    // if expression, evaluate and forward as rvalue reference,
    // if tensor, forward as lvalue reference
    return details::map<U, C_>(std::forward<FunctionType>(fn), exprs.eval()...);
}

/** Given a function `fn` and a variable sequence of Expressions, `exprs`,
 *  apply the function to the elements of `exprs` and return
 *  a new tensor of type, with shape equivalent to `Expressions`
 *  `fn` must have arity equivalent to `sizeof...(exprs)` and each argument
 *  to `fn` must have be const ref or value qualified. The return value
 *  must be trivially convertible to `T`.
 */
template<
        typename U, typename FunctionType, typename... Expressions
>
Tensor<
        U, FirstExpression<Expressions...>::type::rank(),
        FirstExpression<Expressions...>::type::template container_type
>
map(FunctionType &&fn, Expressions const &... exprs)
{
    // verify arguments are expressions
    static_assert(sizeof...(Expressions), NO_EXPRESSIONS_PROVIDED);

    // if expression, evaluate and forward as rvalue reference,
    // if tensor, forward as lvalue reference
    return details::map<U, FirstExpression<Expressions...>::type::container_type>(
            std::forward<FunctionType>(fn), exprs.eval()...
    );
}

/** Given a function `fn` and a variable sequence of Expressions, `exprs`,
 *  apply the function to the elements of `exprs` and return
 *  a new tensor of type, with shape equivalent to `Expressions`
 *  `fn` must have arity equivalent to `sizeof...(exprs)` and each argument
 *  to `fn` must have be const ref or value qualified. The return value
 *  must be trivially convertible to `T`.
 */
template<typename FunctionType, typename... Expressions>
Tensor<
        typename FirstExpression<Expressions...>::type::value_type,
        FirstExpression<Expressions...>::type::rank(),
        FirstExpression<Expressions...>::type::template container_type
>
map(FunctionType &&fn, Expressions const &... exprs)
{
    // verify arguments are expressions
    static_assert(sizeof...(Expressions), NO_EXPRESSIONS_PROVIDED);

    // if expression, evaluate and forward as rvalue reference,
    // if tensor, forward as lvalue reference
    return details::map<
            typename FirstExpression<Expressions...>::type::value_type,
            FirstExpression<Expressions...>::type::template container_type
    >(
            std::forward<FunctionType>(fn), exprs.eval()...
    );
}

/** Given a function `fn` and a variable sequence of Expressions, `Expressions`,
 *  each element of `Expressions` is forwarded as arguments to `fn`.
 *  `fn` must have arity equivalent to `sizeof...(Expressions)` and each argument
 *  to `fn` must have the same cv-qualified ref-unqualified type as the
 *  corresponding Expression. Mutate `Tensors` in place; if `fn` has a
 *  return value it is discarded.
 */
template<typename FunctionType, typename... Expressions>
void Map(FunctionType &&fn, Expressions &&... exprs)
{
    // verify arguments are expressions
    static_assert(sizeof...(Expressions), NO_EXPRESSIONS_PROVIDED);

    // if expression, evaluate and forward as rvalue reference,
    // if tensor, forward as lvalue reference
    details::Map(std::forward<FunctionType>(fn), exprs.eval()...);
}

/** Given an accumulator `initial_value`,  function `fn` and a variable sequence of
 *  Expressions `Expressions`, each element of `Expressions` is forwarded as arguments to
 *  `fn` sequentially, and the result accumulated. `fn` must have arity equivalent to
 *  `sizeof...(Expressions) + 1` and each argument to `fn` must have the same cv-qualified
 *  ref-unqualified type as the corresponding Expression. The accumulator is returned.
 */
template<typename U, typename FunctionType, typename... Expressions>
U reduce(U &&initial_value, FunctionType &&fn, Expressions const &... exprs)
{
    // verify arguments are expressions
    static_assert(sizeof...(Expressions), NO_EXPRESSIONS_PROVIDED);
    static_assert(meta::AreExpressions<Expressions...>::value, EXPECTING_EXPRESSION);

    // if expression, evaluate and forward as rvalue reference,
    // if tensor, forward as lvalue reference
    return details::reduce(
            std::forward<U>(initial_value),
            std::forward<FunctionType>(fn),
            exprs.eval()...
    );
}

/** Given an expression, reduces the expression along the specified dimensions `Indices`.
 *  The provided function `fn` must have arity equivalent to `sizeof...(Indices)`.
 */
template<typename FunctionType, typename Expression, size_t... ReducedIndices>
Tensor<
        typename Expression::value_type, Expression::rank() - sizeof...(ReducedIndices),
        Expression::template container_type
>
reduce_dimensions(FunctionType &&fn, Expression const &expression)
{
    // verify arguments are expressions
    static_assert(meta::AreExpressions<Expression>::value, EXPECTING_EXPRESSION);

    // if expression, evaluate and forward as rvalue reference,
    // if tensor, forward as lvalue reference
    auto tensor = expression.eval();

    return details::reduce_dimensions<FunctionType, decltype(tensor), ReducedIndices...>(
        std::forward<FunctionType>(fn), tensor);
}

/** Creates a Tensor whose elements are the elementwise std::sin evaluation of `rhs`.
 *  The value type and the container type of the resulting tensor can be supplied as
 *  template arguments, and will default to copying the types of `rhs`.
 */
template<
        typename LHS,
        typename X = typename LHS::value_type,
        template<class> class C_ = LHS::template container_type
>
Tensor<X, LHS::rank(), C_> sin(Expression<LHS> const &rhs)
{
    return map<X, C_>(
            [](X const &x) { return std::sin(x); },
            rhs.self()
    );
}

/** Creates a Tensor whose elements are the elementwise std::cos evaluation of `rhs`.
 *  The value type and the container type of the resulting tensor can be supplied as
 *  template arguments, and will default to copying the types of `rhs`.
 */
template<
        typename LHS,
        typename X = typename LHS::value_type,
        template<class> class C_ = LHS::template container_type
>
Tensor<X, LHS::rank(), C_> cos(Expression<LHS> const &rhs)
{
    return map<X, C_>(
            [](X const &x) { return std::cos(x); },
            rhs.self()
    );
}

/** Creates a Tensor whose elements are the elementwise std::tan evaluation of `rhs`.
 *  The value type and the container type of the resulting tensor can be supplied as
 *  template arguments, and will default to copying the types of `rhs`.
 */
template<
        typename LHS,
        typename X = typename LHS::value_type,
        template<class> class C_ = LHS::template container_type
>
Tensor<X, LHS::rank(), C_> tan(Expression<LHS> const &rhs)
{
    return map<X, C_>(
            [](X const &x) { return std::tan(x); },
            rhs.self()
    );
}

/* ------------------------------- Iterator ----------------------------- */

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Iterator::Iterator(Tensor<T, N + 1, C> const &tensor, size_t index)
        : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
    assert(index < N + 1 && PANIC_ASSERTION);
    std::copy_n(tensor.shape_.data(), index, shape_.data());
    std::copy_n(tensor.shape_.data() + index + 1, N - index, shape_.data() + index);
    std::copy_n(tensor.strides_.data(), index, strides_.data());
    std::copy_n(tensor.strides_.data() + index + 1, N - index, strides_.data() + index);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Iterator::Iterator(Iterator const &it)
        : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
    strides_ = it.strides_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::Iterator::operator=(Iterator const &it) -> Iterator &
{
    shape_ = it.shape_;
    offset_ = it.offset_;
    ref_ = it.ref_;
    strides_ = it.strides_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::Iterator::Iterator(Iterator &&it) noexcept
        : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
    strides_ = it.strides_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::Iterator::operator=(Iterator &&it) noexcept -> Iterator &
{
    shape_ = it.shape_;
    offset_ = it.offset_;
    ref_ = std::move(it.ref_);
    strides_ = it.strides_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> Tensor<T, N, C>::Iterator::operator*() const
{
    return Tensor<T, N, C>(
            shape_.dimensions().begin(), shape_.dimensions().end(),
            strides_.begin(), strides_.end(), offset_, ref_
    );
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> Tensor<T, N, C>::Iterator::operator->() const
{
    return Tensor<T, N, C>(
            shape_.dimensions().begin(), shape_.dimensions().end(),
            strides_.begin(), strides_.end(), offset_, ref_
    );
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::Iterator const Tensor<T, N, C>::Iterator::operator++(int)
{
    auto it = Tensor<T, N, C>::Iterator{*this};
    ++(*this);
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::Iterator &Tensor<T, N, C>::Iterator::operator++()
{
    offset_ += stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::Iterator const Tensor<T, N, C>::Iterator::operator--(int)
{
    auto it = Tensor<T, N, C>::Iterator{*this};
    --(*this);
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::Iterator &Tensor<T, N, C>::Iterator::operator--()
{
    offset_ -= stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::Iterator::operator==(
        typename Tensor<T, N, C>::Iterator const &it
) const
{
    if (shape_ != it.shape_) return false;
    if (strides_ != it.strides_) return false;
    if (stride_ != it.stride_) return false;
    if (ref_ != it.ref_) return false;
    return offset_ == it.offset_;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::Iterator::operator>(
        typename Tensor<T, N, C>::Iterator const &it
) const noexcept
{
    assert(shape_ == it.shape_);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ > it.offset_;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::Iterator::operator>=(
        typename Tensor<T, N, C>::Iterator const &it
) const noexcept
{
    assert(shape_ == it.shape_);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ >= it.offset_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::Iterator::operator+(std::ptrdiff_t incr) const noexcept
-> Tensor<T, N, C>::Iterator
{
    auto incr_it = Tensor<T, N, C>::Iterator(*this);
    incr_it.offset_ += incr * incr_it.stride_;
    return incr_it;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::Iterator::operator-(std::ptrdiff_t decr) const noexcept
-> Tensor<T, N, C>::Iterator
{
    auto incr_it = Tensor<T, N, C>::Iterator(*this);
    incr_it.offset_ -= decr * incr_it.stride_;
    return incr_it;
}

template<typename U, size_t M, template<class> class C_>
typename Tensor<U, M, C_>::Iterator
operator+(std::ptrdiff_t incr, typename Tensor<U, M, C_>::Iterator const &it)
{
    return it + incr;
}

template<typename U, size_t M, template<class> class C_>
typename Tensor<U, M, C_>::Iterator
operator-(std::ptrdiff_t decr, typename Tensor<U, M, C_>::Iterator const &it)
{
    return it - decr;
}

template<typename T, size_t N, template<class> class C>
std::ptrdiff_t
Tensor<T, N, C>::Iterator::operator-(Tensor<T, N, C>::Iterator const &it) const noexcept
{
    assert(ref_ == it->ref_ && INVALID_ITERATOR_COMPARISON);
    assert((offset_ - it->offset_) % strides_[N - 1] == 0 && PANIC_ASSERTION);
    assert(stride_ == stride_);
    return ((std::ptrdiff_t) offset_ - (std::ptrdiff_t) it.offset_) / stride_;
}

/* ----------------------------- ConstIterator --------------------------- */

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ConstIterator::ConstIterator(Tensor<T, N + 1, C> const &tensor, size_t index)
        : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
    assert(index < N + 1 && "This should throw earlier");
    std::copy_n(tensor.shape_.data(), index, shape_.data());
    std::copy_n(tensor.shape_.data() + index + 1, N - index, shape_.data() + index);
    std::copy_n(tensor.strides_.data(), index, strides_.data());
    std::copy_n(tensor.strides_.data() + index + 1, N - index, strides_.data() + index);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ConstIterator::ConstIterator(ConstIterator const &it)
        : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
    strides_ = it.strides_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ConstIterator::operator=(ConstIterator const &it)
-> ConstIterator &
{
    shape_ = it.shape_;
    offset_ = it.offset_;
    ref_ = it.ref_;
    strides_ = it.strides_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ConstIterator::ConstIterator(ConstIterator &&it) noexcept
        : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
    strides_ = it.strides_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ConstIterator::operator=(ConstIterator &&it) noexcept
-> ConstIterator &
{
    shape_ = it.shape_;
    offset_ = it.offset_;
    ref_ = std::move(it.ref_);
    strides_ = it.strides_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> const Tensor<T, N, C>::ConstIterator::operator*() const
{
    return Tensor<T, N, C>(
            shape_.dimensions().begin(), shape_.dimensions().end(),
            strides_.begin(), strides_.end(), offset_, ref_
    );
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> const Tensor<T, N, C>::ConstIterator::operator->() const
{
    return Tensor<T, N, C>(
            shape_.dimensions().begin(), shape_.dimensions().end(),
            strides_.begin(), strides_.end(), offset_, ref_
    );
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ConstIterator const Tensor<T, N, C>::ConstIterator::operator++(int)
{
    auto it = Tensor<T, N, C>::ConstIterator{*this};
    ++(*this);
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ConstIterator &Tensor<T, N, C>::ConstIterator::operator++()
{
    offset_ += stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ConstIterator const Tensor<T, N, C>::ConstIterator::operator--(int)
{
    Tensor<T, N, C>::ConstIterator it{*this};
    --(*this);
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ConstIterator &Tensor<T, N, C>::ConstIterator::operator--()
{
    offset_ -= stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ConstIterator::operator==(
        typename Tensor<T, N, C>::ConstIterator const &it
) const
{
    if (shape_ != it.shape_) return false;
    if (strides_ != it.strides_) return false;
    if (stride_ != it.stride_) return false;
    if (ref_ != it.ref_) return false;
    return offset_ == it.offset_;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ConstIterator::operator>(ConstIterator const &it) const noexcept
{
    assert(shape_ == it.shape_);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ > it.offset_;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ConstIterator::operator>=(ConstIterator const &it) const noexcept
{
    assert(shape_ == it.shape_);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ >= it.offset_;
}

template<typename T, size_t N, template<class> class C>
std::ptrdiff_t
Tensor<T, N, C>::ConstIterator::operator-(ConstIterator const &it) const noexcept
{
    assert(ref_ == it->ref_ && INVALID_ITERATOR_COMPARISON);
    assert((offset_ - it->offset_) % strides_[N - 1] == 0 && PANIC_ASSERTION);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    return ((std::ptrdiff_t) offset_ - (std::ptrdiff_t) it.offset_) / stride_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ConstIterator::operator+(std::ptrdiff_t incr) const noexcept
-> ConstIterator
{
    auto incr_it = Tensor<T, N, C>::Iterator(*this);
    incr_it.offset_ += incr * incr_it.stride_;
    return incr_it;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ConstIterator::operator-(std::ptrdiff_t decr) const noexcept
-> ConstIterator
{
    auto incr_it = Tensor<T, N, C>::Iterator(*this);
    incr_it.offset_ -= decr * incr_it.stride_;
    return incr_it;
}

template<typename U, size_t M, template<class> class C_>
typename Tensor<U, M, C_>::ConstIterator
operator+(std::ptrdiff_t incr, typename Tensor<U, M, C_>::ConstIterator const &it)
{
    return it + incr;
}

template<typename U, size_t M, template<class> class C_>
typename Tensor<U, M, C_>::ConstIterator
operator-(std::ptrdiff_t decr, typename Tensor<U, M, C_>::ConstIterator const &it)
{
    return it - decr;
}

/* ---------------------------- ReverseIterator -------------------------- */

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ReverseIterator::ReverseIterator(
        Tensor<T, N + 1, C> const &tensor,
        size_t index
) : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
    assert(index < N + 1 && "This should throw earlier");
    std::copy_n(tensor.shape_.data(), index, shape_.data());
    std::copy_n(tensor.shape_.data() + index + 1, N - index, shape_.data() + index);
    std::copy_n(tensor.strides_.data(), index, strides_.data());
    std::copy_n(tensor.strides_.data() + index + 1, N - index, strides_.data() + index);
    offset_ += stride_ * (tensor.shape_[index] - 1);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ReverseIterator::ReverseIterator(ReverseIterator const &it)
        : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
    strides_ = it.strides_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ReverseIterator::operator=(ReverseIterator const &it)
-> ReverseIterator &
{
    shape_ = it.shape_;
    offset_ = it.offset_;
    ref_ = it.ref_;
    strides_ = it.strides_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ReverseIterator::ReverseIterator(ReverseIterator &&it) noexcept
        : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
    strides_ = it.strides_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ReverseIterator::operator=(ReverseIterator &&it) noexcept
-> ReverseIterator &
{
    shape_ = it.shape_;
    offset_ = it.offset_;
    ref_ = std::move(it.ref_);
    strides_ = it.strides_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> Tensor<T, N, C>::ReverseIterator::operator*() const
{
    return Tensor<T, N, C>(
            shape_.dimensions().begin(), shape_.dimensions().end(),
            strides_.begin(), strides_.end(), offset_, ref_
    );
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> Tensor<T, N, C>::ReverseIterator::operator->() const
{
    return Tensor<T, N, C>(
            shape_.dimensions().begin(), shape_.dimensions().end(),
            strides_.begin(), strides_.end(), offset_, ref_
    );
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ReverseIterator const Tensor<T, N, C>::ReverseIterator::operator++(int)
{
    auto it = Tensor<T, N, C>::ReverseIterator{*this};
    ++(*this);
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ReverseIterator &Tensor<T, N, C>::ReverseIterator::operator++()
{
    offset_ -= stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ReverseIterator const Tensor<T, N, C>::ReverseIterator::operator--(int)
{
    Tensor<T, N, C>::ReverseIterator it{*this};
    --(*this);
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ReverseIterator &Tensor<T, N, C>::ReverseIterator::operator--()
{
    offset_ += stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ReverseIterator::operator==(
        typename Tensor<T, N, C>::ReverseIterator const &it
) const
{
    if (shape_ != it.shape_) return false;
    if (stride_ != it.stride_) return false;
    if (ref_ != it.ref_) return false;
    return offset_ == it.offset_;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ReverseIterator::operator>(ReverseIterator const &it) const noexcept
{
    assert(shape_ == it.shape_);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ > it.offset_;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ReverseIterator::operator>=(ReverseIterator const &it) const noexcept
{
    assert(shape_ == it.shape_);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ <= it.offset_;
}

template<typename T, size_t N, template<class> class C>
std::ptrdiff_t
Tensor<T, N, C>::ReverseIterator::operator-(ReverseIterator const &it) const noexcept
{
    assert(ref_ == it->ref_ && INVALID_ITERATOR_COMPARISON);
    assert((offset_ - it->offset_) % strides_[N - 1] == 0 && PANIC_ASSERTION);
    assert(strides_ == it->strides_);
    assert(stride_ == it->stride_);
    return ((std::ptrdiff_t) it.offset_ - (std::ptrdiff_t) offset_) / stride_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ReverseIterator::operator+(std::ptrdiff_t incr) const noexcept
-> ReverseIterator
{
    auto incr_it = Tensor<T, N, C>::Iterator(*this);
    incr_it.offset_ -= incr * incr_it.stride_;
    return incr_it;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ReverseIterator::operator-(std::ptrdiff_t decr) const noexcept
-> ReverseIterator
{
    auto incr_it = Tensor<T, N, C>::Iterator(*this);
    incr_it.offset_ += decr * incr_it.stride_;
    return incr_it;
}

template<typename U, size_t M, template<class> class C_>
typename Tensor<U, M, C_>::ReverseIterator
operator+(std::ptrdiff_t incr, typename Tensor<U, M, C_>::ReverseIterator const &it)
{
    return it + incr;
}

template<typename U, size_t M, template<class> class C_>
typename Tensor<U, M, C_>::ReverseIterator
operator-(std::ptrdiff_t decr, typename Tensor<U, M, C_>::ReverseIterator const &it)
{
    return it - decr;
}

/* ------------------------- ConstReverseIterator ----------------------- */

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ConstReverseIterator::ConstReverseIterator(
        Tensor<T, N + 1, C> const &tensor,
        size_t index
)
        : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
    assert(index < N + 1 && "This should throw earlier");
    std::copy_n(tensor.shape_.data(), index, shape_.data());
    std::copy_n(tensor.shape_.data() + index + 1, N - index, shape_.data() + index);
    std::copy_n(tensor.strides_.data(), index, strides_.data());
    std::copy_n(tensor.strides_.data() + index + 1, N - index, strides_.data() + index);
    offset_ += stride_ * (tensor.shape_[index] - 1);
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator const &it)
        : shape_(it.shape_), offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{
    strides_ = it.strides_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ConstReverseIterator::operator=(ConstReverseIterator const &it)
-> ConstReverseIterator &
{
    shape_ = it.shape_;
    offset_ = it.offset_;
    ref_ = it.ref_;
    strides_ = it.strides_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator &&it) noexcept
        : shape_(it.shape_), offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
    strides_ = it.strides_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ConstReverseIterator::operator=(ConstReverseIterator &&it) noexcept
-> ConstReverseIterator &
{
    shape_ = it.shape_;
    offset_ = it.offset_;
    ref_ = std::move(it.ref_);
    strides_ = it.strides_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> const Tensor<T, N, C>::ConstReverseIterator::operator*() const
{
    return Tensor<T, N, C>(
            shape_.dimensions().begin(), shape_.dimensions().end(),
            strides_.begin(), strides_.end(), offset_, ref_
    );
}

template<typename T, size_t N, template<class> class C>
Tensor<T, N, C> const Tensor<T, N, C>::ConstReverseIterator::operator->() const
{
    return Tensor<T, N, C>(
            shape_.dimensions().begin(), shape_.dimensions().end(),
            strides_.begin(), strides_.end(), offset_, ref_
    );
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ConstReverseIterator const
Tensor<T, N, C>::ConstReverseIterator::operator++(int)
{
    Tensor<T, N, C>::ConstReverseIterator it{*this};
    ++(*this);
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ConstReverseIterator &
Tensor<T, N, C>::ConstReverseIterator::operator++()
{
    offset_ -= stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ConstReverseIterator const
Tensor<T, N, C>::ConstReverseIterator::operator--(int)
{
    Tensor<T, N, C>::ConstReverseIterator it{*this};
    --(*this);
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N, C>::ConstReverseIterator &
Tensor<T, N, C>::ConstReverseIterator::operator--()
{
    offset_ += stride_;
    return *this;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ConstReverseIterator::operator==(
        typename Tensor<T, N, C>::ConstReverseIterator const &it
) const
{
    if (shape_ != it.shape_) return false;
    if (stride_ != it.stride_) return false;
    if (ref_ != it.ref_) return false;
    return offset_ == it.offset_;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ConstReverseIterator::operator>(
        ConstReverseIterator const &it
) const noexcept
{
    assert(shape_ == it.shape_);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ > it.offset_;
}

template<typename T, size_t N, template<class> class C>
bool Tensor<T, N, C>::ConstReverseIterator::operator>=(
        ConstReverseIterator const &it
) const noexcept
{
    assert(shape_ == it.shape_);
    assert(strides_ == it.strides_);
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ <= it.offset_;
}

template<typename T, size_t N, template<class> class C>
std::ptrdiff_t
Tensor<T, N, C>::ConstReverseIterator::operator-(
        ConstReverseIterator const &it
) const noexcept
{
    assert(ref_ == it->ref_ && INVALID_ITERATOR_COMPARISON);
    assert((offset_ - it->offset_) % strides_[N - 1] == 0 && PANIC_ASSERTION);
    assert(stride_ == it.stride_);
    assert(strides_ == it.strides_);
    return ((std::ptrdiff_t) it.offset_ - (std::ptrdiff_t) offset_) / stride_;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ConstReverseIterator::operator+(std::ptrdiff_t incr) const noexcept
-> ConstReverseIterator
{
    auto incr_it = Tensor<T, N, C>::Iterator(*this);
    incr_it.offset_ -= incr * incr_it.stride_;
    return incr_it;
}

template<typename T, size_t N, template<class> class C>
auto Tensor<T, N, C>::ConstReverseIterator::operator-(std::ptrdiff_t decr) const noexcept
-> ConstReverseIterator
{
    auto incr_it = Tensor<T, N, C>::Iterator(*this);
    incr_it.offset_ += decr * incr_it.stride_;
    return incr_it;
}

template<typename U, size_t M, template<class> class C_>
typename Tensor<U, M, C_>::ConstReverseIterator
operator+(std::ptrdiff_t incr, typename Tensor<U, M, C_>::ConstReverseIterator const &it)
{
    return it + incr;
}

template<typename U, size_t M, template<class> class C_>
typename Tensor<U, M, C_>::ConstReverseIterator
operator-(std::ptrdiff_t decr, typename Tensor<U, M, C_>::ConstReverseIterator const &it)
{
    return it - decr;
}

/* ----------------------- Iterator Construction ----------------------- */

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::Iterator Tensor<T, N, C>::begin(size_t index)
{
    assert((index < N) && INDEX_OUT_OF_BOUNDS);
    return typename Tensor<T, N - 1, C>::Iterator(*this, index);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::Iterator Tensor<T, N, C>::end(size_t index)
{
    assert((index < N) && INDEX_OUT_OF_BOUNDS);
    typename Tensor<T, N - 1, C>::Iterator it{*this, index};
    it.offset_ += strides_[index] * shape_[index];
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::Iterator Tensor<T, N, C>::begin()
{
    // Probably a mistake if using this on a non-vector
    static_assert(N == 1, BEGIN_ON_NON_VECTOR);
    return this->begin(0);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::Iterator Tensor<T, N, C>::end()
{
    static_assert(N == 1, BEGIN_ON_NON_VECTOR);
    return this->end(0);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ConstIterator Tensor<T, N, C>::cbegin(size_t index) const
{
    assert((index < N) && INDEX_OUT_OF_BOUNDS);
    return typename Tensor<T, N - 1, C>::ConstIterator(*this, index);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ConstIterator Tensor<T, N, C>::cend(size_t index) const
{
    assert((index < N) && INDEX_OUT_OF_BOUNDS);
    typename Tensor<T, N - 1, C>::ConstIterator it{*this, index};
    it.offset_ += strides_[index] * shape_[index];
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ConstIterator Tensor<T, N, C>::cbegin() const
{
    // Probably a mistake if using this on a non-vector
    static_assert(N == 1, BEGIN_ON_NON_VECTOR);
    return this->cbegin(0);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ConstIterator Tensor<T, N, C>::cend() const
{
    static_assert(N == 1, BEGIN_ON_NON_VECTOR);
    return this->cend(0);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ReverseIterator Tensor<T, N, C>::rbegin(size_t index)
{
    assert((index < N) && INDEX_OUT_OF_BOUNDS);
    return typename Tensor<T, N - 1, C>::ReverseIterator(*this, index);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ReverseIterator Tensor<T, N, C>::rend(size_t index)
{
    assert((index < N) && INDEX_OUT_OF_BOUNDS);
    typename Tensor<T, N - 1, C>::ReverseIterator it{*this, index};
    it.offset_ -= strides_[index] * shape_[index];
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ReverseIterator Tensor<T, N, C>::rbegin()
{
    static_assert(N == 1, BEGIN_ON_NON_VECTOR);
    return this->rbegin(0);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ReverseIterator Tensor<T, N, C>::rend()
{
    static_assert(N == 1, BEGIN_ON_NON_VECTOR);
    return this->rend(0);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ConstReverseIterator
Tensor<T, N, C>::crbegin(size_t index) const
{
    assert((index < N) && INDEX_OUT_OF_BOUNDS);
    return typename Tensor<T, N - 1, C>::ConstReverseIterator(*this, index);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ConstReverseIterator Tensor<T, N, C>::crend(size_t index) const
{
    assert((index < N) && INDEX_OUT_OF_BOUNDS);
    typename Tensor<T, N - 1, C>::ConstReverseIterator it{*this, index};
    it.offset_ -= strides_[index] * shape_[index];
    return it;
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ConstReverseIterator Tensor<T, N, C>::crbegin() const
{
    // Probably a mistake if using this on a non-vector
    static_assert(N == 1, BEGIN_ON_NON_VECTOR);
    return this->crbegin(0);
}

template<typename T, size_t N, template<class> class C>
typename Tensor<T, N - 1, C>::ConstReverseIterator Tensor<T, N, C>::crend() const
{
    static_assert(N == 1, BEGIN_ON_NON_VECTOR);
    return this->crend(0);
}

/* ------------------------ Scalar Specializations ---------------------- */

template<>
class Shape<0> { /*@Shape<0>*/
public:
    /** Scalar specialization of Shape. This is an empty structure
   *  (sizeof(Shape<0>) will return 1 byte). This is specialized
   *  to resolve compiler issues with handling 0-length arrays.
   */

    /* -------------------- typedefs -------------------- */
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef Shape<0> self_type;

    /* ----------------- friend classes ----------------- */

    template<typename X, size_t M, template<class> class C> friend
    class Tensor;

    template<typename LHS, typename RHS> friend
    class BinaryAddExpr;

    template<typename LHS, typename RHS> friend
    class BinarySubExpr;

    template<size_t I1, size_t I2, typename LHS, typename RHS> friend
    class BinaryMulExpr;

    template<typename LHS, typename RHS> friend
    class BinaryHadExpr;

    template<typename Function, typename... Exprs> friend
    class MapExpr;

    template<typename U, typename Function, typename... Exprs> friend
    class ReduceExpr;

    template<typename RHS> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ------------------ Constructors ------------------ */

    explicit Shape() = default;

    // to match @Shape<N> constructor
    explicit Shape(std::array<size_t, 0> const &): Shape() {};

    Shape(Shape<0> const &) = default;

    /* -------------------- Getters --------------------- */

    constexpr static size_t rank()
    { return 0; }

    constexpr static size_t index_product()
    { return 1; }

    /* -------------------- Equality -------------------- */

    /** This will always return true */
    bool operator==(Shape<0> const &) const noexcept
    { return true; }

    /** This will always return false */
    bool operator!=(Shape<0> const &) const noexcept
    { return false; }

    /** This will always return false */
    template<size_t M>
    bool operator==(Shape<M> const &) const noexcept
    { return false; }

    /** This will always return true */
    template<size_t M>
    bool operator!=(Shape<M> const &) const noexcept
    { return true; }

};

template<>
class Indices<0> { /*@Indices<0>*/
/** Scalar specialized 0-length Indices. This is an empty object
 *  (sizeof(Indices<0>) will return 1) to resolve compiler issues with
 *  0-length arrays.
 */
public:

    /* ------------------ typedefs ---------------- */

    typedef Indices self_type;
    typedef std::size_t size_t;

    /* --------------- Friend Classes ------------- */

    template<typename U, size_t M, template<class> class C_> friend
    class Tensor;

    template<typename LHS, typename RHS> friend
    class BinaryAddExpr;

    template<typename LHS, typename RHS> friend
    class BinarySubExpr;

    template<size_t I1, size_t I2, typename LHS, typename RHS> friend
    class BinaryMulExpr;

    template<typename LHS, typename RHS> friend
    class BinaryHadExpr;

    template<typename Function, typename... Exprs> friend
    class MapExpr;

    template<typename U, typename Function, typename... Exprs> friend
    class ReduceExpr;

    template<typename RHS> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* --------------- Constructors ------------- */

    template<typename U, size_t M, template<class> class C_> friend
    class Tensor;

    Indices() = default;

    /* ----------------- Getters --------------- */

    /** This method is only defined because MSVC++ requires a definition
     *  to be present, even if the method is never actually called.
     *  An assertion is immediately called if this method is invoked.
     */
    size_t &operator[](size_t)
    {
        assert(0 && PANIC_ASSERTION);
        return *(size_t *) this;
    }

    /** See operator[](size_t) */
    size_t const &operator[](size_t) const
    {
        assert(0 && PANIC_ASSERTION);
        return *(size_t *) this;
    }

    /* ----------------- Utility --------------- */

    /** Since the rank is 0, immediately returns done */
    bool increment(Shape<0> const &) { return false; }

    /** Since the rank is 0, immediately returns done */
    bool decrement(Shape<0> const &) { return false; }

    std::string str() const { return "I{0}"; }

private:

    // mirrors Indices<N> API -- doesn't do anything
    explicit Indices(size_t const *)
    {}

};

// Scalar specialization
template<typename T, template<class> class C>
class Tensor<T, 0, C> : public Expression<Tensor<T, 0, C>> {
/*@Tensor<T, 0, C>*/
/** Scalar specialization of Tensor object. The major motivation is
 *  that certain functions can be much more efficient if multi-dimensions
 *  do not have to be considered.
 */
public:
    typedef T value_type;
    template<typename X> using container_type =    C<X>;
    typedef T &reference_type;
    typedef T const &const_reference_type;
    typedef Tensor<T, 0, C> self_type;
    typedef Tensor<T, 0, C> return_type;

    /* ------------- Friend Classes ------------- */

    template<typename X, size_t M, template<class> class C_> friend
    class Tensor;

    /* ----------- Proxy Objects ------------ */

    class Proxy { /*@Proxy<T,0>*/
        /**
   * Proxy Tensor Object used for building tensors from reference
   * This is used only to differentiate proxy tensor Construction
   */
    public:
        template<typename U, size_t N, template<class> class C_> friend
        class Tensor;

        Proxy() = delete;

    private:
        explicit Proxy(Tensor<T, 0, C> const &tensor) : tensor_(tensor)
        {}

        Proxy(Proxy const &proxy) : tensor_(proxy.tensor_)
        {}

        Tensor<T, 0, C> const &tensor_;
    };

    /* -------------- Constructors -------------- */

    /** Constructs a new Scalar whose value is zero-initialized */
    Tensor();

    /** Constructs a new Scalar whose value is forwarded `val` */
    explicit Tensor(value_type &&val);

    /** Constructs a new Scalar whose value is zero-initialized */
    explicit Tensor(Shape<0>);

    /** Constructs a new Scalar and whose value copies `tensor`'s */
    Tensor(Tensor<T, 0, C> const &tensor);

    /** Constructs a new Scalar and whose value copies `tensor`'s */
    template<typename U, template<class> class C_>
    Tensor(Tensor<U, 0, C_> const &tensor);

    /** Moves data from `tensor`. `tensor` is destroyed. */
    Tensor(Tensor<T, 0, C> &&tensor) noexcept;

    /** Constructs a Scalar who shares underlying data with proxy's underyling Scalar. */
    Tensor(typename Tensor<T, 0, C>::Proxy const &proxy);

    /** Evaluates `expression` and move constructs from the resulting scalar */
    template<typename NodeType, typename = typename std::enable_if<NodeType::rank() == 0>::type>
    Tensor(Expression<NodeType> const &expression);

#ifdef _ENABLE_OPENCL

    /** Constructs the tensor using the result from an OpenCL computation */
    template<typename NodeType>
    Tensor(opencl::Model<NodeType> const &model);

#endif

    /* -------------- Destructor ------------- */

    ~Tensor() noexcept = default;

    /* ------------- Assignment ------------- */

    /** Assigns the value from `tensor`, to its element. */
    Tensor<T, 0, C> &operator=(Tensor<T, 0, C> const &tensor);

    /** Moves the value from `tensor`, to its element. */
    Tensor<T, 0, C> &operator=(Tensor<T, 0, C> &&tensor) noexcept;

    /** Assigns the value from `tensor`, to its element. */
    template<typename U, template<class> class C_>
    Tensor<T, 0, C> &operator=(Tensor<U, 0, C_> const &tensor);

    /** Assigns the value from `tensor`, to its element. */
    template<typename U, template<class> class C_>
    Tensor<T, 0, C> &operator=(Tensor<U, 0, C_> &&tensor);

    /* -------------- Getters -------------- */

    /** Returns a reference to itself */
    Tensor &eval() noexcept
    { return *this; }

    /** Returns a const reference to itself */
    Tensor const &eval() const noexcept
    { return *this; }

    /** Get the rank: compile time constant 0 */
    constexpr static size_t rank()
    { return 0; }

    /** Get the shape: 0-size shape (sizeof(Shape<0>) is 1) */
    Shape<0> shape() const noexcept
    { return shape_; }

    /** Returns the data as a reference */
    value_type &operator()()
    { return ref_[offset_]; }

    /** Returns the data as a const reference */
    value_type const &operator()() const
    { return ref_[offset_]; }

    /** Returns the data as a reference */
    value_type &operator[](Indices<0> const &) { return ref_[offset_]; }
 
    /** Returns the data as a const reference */
    value_type const &operator[](Indices<0> const &) const { return ref_[offset_]; }

    /** Creates a Scalar with the same underlying data */
    Tensor<T, 0, C> at()
    { return Tensor<T, 0, C>(this->ref()); }

    /** Creates a const Scalar with the same underlying data */
    Tensor<T, 0, C> const at() const
    { return Tensor<T, 0, C>(this->ref()); }

    /** Creates a Scalar with the same underlying data */
    Tensor<T, 0, C> slice()
    { return Tensor<T, 0, C>(this->ref()); }

    /** Creates a const Scalar with the same underlying data */
    Tensor<T, 0, C> const slice() const
    { return Tensor<T, 0, C>(this->ref()); }

    /** Used to implement iterator->, should not be used explicitly */
    Tensor *operator->()
    { return this; }

    /** Used to implement const_iterator->, should not be used explicitly */
    Tensor const *operator->() const
    { return this; }

    /* ------------- Comparators ------------ */

    bool operator<(Tensor const &scalar) const
    { return (*this)() < scalar(); }

    bool operator<=(Tensor const &scalar) const
    { return (*this)() <= scalar(); }

    bool operator>(Tensor const &scalar) const
    { return (*this)() > scalar(); }

    bool operator>=(Tensor const &scalar) const
    { return (*this)() >= scalar(); }

    /* -------------- Setters -------------- */

    template<
            typename X,
            typename = typename std::enable_if<std::is_convertible<
                    typename std::remove_reference<T>::type,
                    typename std::remove_reference<X>::type>::value>::type
    >
    Tensor<T, 0, C> &operator=(X &&elem); /**< Assigns `elem` to the underlying data */

    template<typename U, template<class> class C_>
    friend void Set(Tensor<U, 0, C_> &tensor, Indices<0> const &, U &&value);

    template<typename U, template<class> class C_>
    friend inline void Set(Tensor<U, 0, C_> &tensor, U &&value);

    /* ---------- Convert To String ------------ */

    std::string str() const;

    /* ------------ Equivalence ------------ */

    /** Equivalence between underlying data */
    bool operator==(Tensor<T, 0, C> const &tensor) const;

    /** Non-equivalence between underlying data */
    bool operator!=(Tensor<T, 0, C> const &tensor) const
    { return !(*this == tensor); }

    /** Equivalence between underlying data */
    template<typename X>
    bool operator==(Tensor<X, 0, C> const &tensor) const;

    /** Non-equivalence between underlying data */
    template<typename X>
    bool operator!=(Tensor<X, 0, C> const &tensor) const
    { return !(*this == tensor); }

    /** Equivalence Scalar with an element of a type default-convertible
   *  to the underlying data type.
   */
    template<
            typename X,
            typename = typename std::enable_if<std::is_convertible<
                    typename std::remove_reference<T>::type,
                    typename std::remove_reference<X>::type>::value>::type
    >
    bool operator==(X val) const;

    /** Non-equivalence Scalar with an element of a type default-convertible
   *  to the underlying data type.
   */
    template<
            typename X,
            typename = typename std::enable_if<std::is_convertible<
                    typename std::remove_reference<T>::type,
                    typename std::remove_reference<X>::type>::value>::type
    >
    bool operator!=(X val) const
    { return !(*this == val); }

    /* ------------ Expressions ------------- */

    /** Equivalent to `*this = *this + rhs` */
    template<typename RHS>
    Tensor<T, 0, C> &operator+=(Expression<RHS> const &rhs);

    /** Equivalent to `*this = *this + scalar` */
    Tensor<T, 0, C> &operator+=(T const &scalar);

    /** Equivalent to `*this = *this - rhs` */
    template<typename RHS>
    Tensor<T, 0, C> &operator-=(Expression<RHS> const &rhs);

    /** Equivalent to `*this = *this - scalar` */
    Tensor<T, 0, C> &operator-=(T const &scalar);

    /** Equivalent to `*this = *this * rhs` */
    template<typename RHS>
    Tensor<T, 0, C> &operator*=(Expression<RHS> const &rhs);

    /** Equivalent to `*this = *this * scalar` */
    Tensor<T, 0, C> &operator*=(T const &scalar);

    /** Equivalent to `*this = *this / rhs` */
    template<typename RHS>
    Tensor<T, 0, C> &operator/=(Expression<RHS> const &rhs);

    /** Equivalent to `*this = *this / scalar` */
    Tensor<T, 0, C> &operator/=(T const &scalar);

    /** Creates a scalar by applying unary - to this scalar's value */
    Tensor<T, 0, C> neg() const;

    /** Creates a scalar by applying unary - to this scalar's value */
    Tensor<T, 0, C> operator-() const
    { return this->neg(); }

    /* ----------------- OpenCL --------------- */

#ifdef _ENABLE_OPENCL

    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

    /* ---------------- Iterator -------------- */

    class Iterator { /*@Iterator<T, 0, C>*/
    public:
        /**
   * Scalar iterator are provided to match the API of Tensor<T, N, C>'s iterator.
   * They can be dereferenced to obtain the underlying value, and incrementing once
   * will move past the underlying data.
   */

        /* ------------- Iterator Traits -------------- */

        typedef T value_type;
        typedef T &reference;
        typedef T *pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

        /* ----------------- Friends ----------------- */

        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        /* --------------- Constructors --------------- */

        Iterator(Iterator const &it);

        Iterator &operator=(Iterator const &it);

        Iterator(Iterator &&it) noexcept;

        Iterator &operator=(Iterator &&it) noexcept;

        constexpr T &operator*() const;

        constexpr T *operator->() const;

        Iterator const operator++(int);

        Iterator &operator++();

        Iterator const operator--(int);

        Iterator &operator--();

        bool operator==(Iterator const &it) const;

        bool operator!=(Iterator const &it) const
        { return !(it == *this); }

        std::ptrdiff_t operator-(Iterator const &it) const noexcept;

        Iterator operator+(std::ptrdiff_t incr) const;

        Iterator operator-(std::ptrdiff_t decr) const;

        /** Returns true iff the offset difference is positive */
        bool operator>(Iterator const &it) const;

        /** Returns true iff the offset difference is non-negative */
        bool operator>=(Iterator const &it) const;

        /** Returns true iff the offset difference is negative */
        bool operator<(Iterator const &it) const
        { return !(*this >= it); }

        /** Returns true iff the offset difference is non-positive */
        bool operator<=(Iterator const &it) const
        { return !(*this > it); }

    private:

        Iterator(Tensor<T, 1, C> const &tensor, size_t);

        size_t offset_;
        C<T> ref_;
        size_t stride_;
    };

    /* -------------- ConstIterator ------------ */

    class ConstIterator { /*@ConstIterator<T, 0, C>*/
        /**
   * Scalar const iterator are provided to match the API of Tensor<T, N, C>'s
   * const iterator.  They can be dereferenced to obtain the underlying value,
   * and incrementing once will move past the underlying data.
   */
    public:

        /* ------------- Iterator Traits -------------- */

        typedef T const value_type;
        typedef T const &reference;
        typedef T const *pointer;
        typedef std::ptrdiff_t difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

        /* -------------- Friend Classes -------------- */

        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        /* --------------- Constructors --------------- */

        ConstIterator(ConstIterator const &it);

        ConstIterator &operator=(ConstIterator const &it);

        ConstIterator(ConstIterator &&it) noexcept;

        ConstIterator &operator=(ConstIterator &&it) noexcept;

        constexpr T const &operator*() const;

        constexpr T const *operator->() const;

        ConstIterator const operator++(int);

        ConstIterator &operator++();

        ConstIterator const operator--(int);

        ConstIterator &operator--();

        bool operator==(ConstIterator const &it) const;

        bool operator!=(ConstIterator const &it) const
        { return !(it == *this); }

        std::ptrdiff_t operator-(ConstIterator const &it) const noexcept;

        ConstIterator operator+(std::ptrdiff_t incr) const;

        ConstIterator operator-(std::ptrdiff_t decr) const;

        /** Returns true iff the offset difference is positive */
        bool operator>(ConstIterator const &it) const;

        /** Returns true iff the offset difference is non-negative */
        bool operator>=(ConstIterator const &it) const;

        /** Returns true iff the offset difference is negative */
        bool operator<(ConstIterator const &it) const
        { return !(*this >= it); }

        /** Returns true iff the offset difference is non-positive */
        bool operator<=(ConstIterator const &it) const
        { return !(*this > it); }

    private:
        ConstIterator(Tensor<T, 1, C> const &tensor, size_t);

        size_t offset_;
        C<T> ref_;
        size_t stride_;
    };

    /* ------------- ReverseIterator ----------- */

    class ReverseIterator { /*@ReverseIterator<T, 0, C>*/
    public:
        /**
   * Scalar reverse iterator are provided to match the API of Tensor<T, N, C>'s
   * reverse iterator. They can be dereferenced to obtain the underlying value,
   * and incrementing once will move past the underlying data.
   */

        /* ------------- Iterator Traits -------------- */

        typedef T value_type;
        typedef T &reference;
        typedef std::ptrdiff_t difference_type;
        typedef T *pointer;
        typedef std::bidirectional_iterator_tag iterator_category;

        /* -------------- Friend Classes -------------- */

        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        /* --------------- Constructors --------------- */

        ReverseIterator(ReverseIterator const &it);

        ReverseIterator &operator=(ReverseIterator const &it);

        ReverseIterator(ReverseIterator &&it) noexcept;

        ReverseIterator &operator=(ReverseIterator &&it) noexcept;

        constexpr T &operator*() const;

        constexpr T *operator->() const;

        ReverseIterator const operator++(int);

        ReverseIterator &operator++();

        ReverseIterator const operator--(int);

        ReverseIterator &operator--();

        bool operator==(ReverseIterator const &it) const;

        bool operator!=(ReverseIterator const &it) const
        { return !(it == *this); }

        std::ptrdiff_t operator-(ReverseIterator const &it) const noexcept;

        ReverseIterator operator+(std::ptrdiff_t incr) const;

        ReverseIterator operator-(std::ptrdiff_t decr) const;

        /** Returns true iff the offset difference is positive */
        bool operator>(ReverseIterator const &it) const;

        /** Returns true iff the offset difference is non-negative */
        bool operator>=(ReverseIterator const &it) const;

        /** Returns true iff the offset difference is negative */
        bool operator<(ReverseIterator const &it) const
        { return !(*this >= it); }

        /** Returns true iff the offset difference is non-positive */
        bool operator<=(ReverseIterator const &it) const
        { return !(*this > it); }

    private:
        ReverseIterator(Tensor<T, 1, C> const &tensor, size_t);

        size_t offset_;
        C<T> ref_;
        size_t stride_;
    };

    /* ----------- ConstReverseIterator --------- */

    class ConstReverseIterator { /*@ConstReverseIterator<T, 0, C>*/
    public:
        /**
   * Scalar const reverse iterator are provided to match the API of Tensor<T, N, C>'s
   * const reverse iterator. They can be dereferenced to obtain the underlying value,
   * and incrementing once will move past the underlying data.
   */

        /* ------------- Iterator Traits -------------- */

        typedef T const value_type;
        typedef T const &reference;
        typedef std::ptrdiff_t difference_type;
        typedef T const *pointer;
        typedef std::bidirectional_iterator_tag iterator_category;

        /* -------------- Friend Classes -------------- */

        template<typename U, size_t M, template<class> class C_> friend
        class Tensor;

        /* --------------- Constructors --------------- */

        ConstReverseIterator(ConstReverseIterator const &it);

        ConstReverseIterator &operator=(ConstReverseIterator const &it);

        ConstReverseIterator(ConstReverseIterator &&it) noexcept;

        ConstReverseIterator &operator=(ConstReverseIterator &&it) noexcept;

        constexpr T const &operator*() const;

        constexpr T const *operator->() const;

        ConstReverseIterator const operator++(int);

        ConstReverseIterator &operator++();

        ConstReverseIterator const operator--(int);

        ConstReverseIterator &operator--();

        bool operator==(ConstReverseIterator const &it) const;

        bool operator!=(ConstReverseIterator const &it) const
        { return !(it == *this); }

        std::ptrdiff_t operator-(ConstReverseIterator const &it) const noexcept;

        ConstReverseIterator operator+(std::ptrdiff_t incr) const;

        ConstReverseIterator operator-(std::ptrdiff_t decr) const;

        /** Returns true iff the offset difference is positive */
        bool operator>(ConstReverseIterator const &it) const;

        /** Returns true iff the offset difference is non-negative */
        bool operator>=(ConstReverseIterator const &it) const;

        /** Returns true iff the offset difference is negative */
        bool operator<(ConstReverseIterator const &it) const
        { return !(*this >= it); }

        /** Returns true iff the offset difference is non-positive */
        bool operator<=(ConstReverseIterator const &it) const
        { return !(*this > it); }

    private:
        ConstReverseIterator(Tensor<T, 1, C> const &tensor, size_t);

        size_t offset_;
        C<T> ref_;
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
    C<T> ref_;

    /* ------------------ Utility ----------------- */

    // match the Tensor<T, N, C> interface
    inline T const &
    pGet(size_t *, size_t) const noexcept
    { return (static_cast<C<T> const &>(ref_))[offset_]; }

    // match the Tensor<T, N, C> interface
    inline T &pGet(size_t *, size_t) noexcept
    { return ref_[offset_]; }

    // match the Tensor<T, N, C> interface
    Tensor(size_t const *, size_t const *, size_t, C<T> &&ref);

    Tensor(size_t const *, size_t const *, size_t, C<T> const &ref);

};

/* ------------------- Constructors ----------------- */

template<typename T, template<class> class C>
Tensor<T, 0, C>::Tensor()
        : shape_(Shape<0>()), offset_(0), ref_(1)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::Tensor(Shape<0>)
        : shape_(Shape<0>()), offset_(0), ref_(1)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::Tensor(T &&val)
        : shape_(Shape<0>()), offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    ref_ = container_type<T>(1, val);
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::Tensor(Tensor<T, 0, C> const &tensor)
        : shape_(Shape<0>()), offset_(0), ref_(1, tensor.ref_[0])
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
}

template<typename T, template<class> class C>
template<typename U, template<class> class C_>
Tensor<T, 0, C>::Tensor(Tensor<U, 0, C_> const &tensor)
        : shape_(Shape<0>()), offset_(0), ref_(1, tensor.ref_[0])
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::Tensor(Tensor<T, 0, C> &&tensor) noexcept
        : shape_(Shape<0>()), offset_(tensor.offset_), ref_(1)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::Tensor(typename Tensor<T, 0, C>::Proxy const &proxy)
        : shape_(Shape<0>()), offset_(proxy.tensor_.offset_), ref_(proxy.tensor_.ref_)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
}

template<typename T, template<class> class C>
template<typename NodeType, typename>
Tensor<T, 0, C>::Tensor(Expression<NodeType> const &expression)
        : offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    T result = expression.self()();
    ref_ = container_type<T>(1, result);
}

#ifdef _ENABLE_OPENCL

template<typename T, template<class> class C>
template<typename NodeType>
Tensor<T, 0, C>::Tensor(opencl::Model<NodeType> const &model)
        : offset_(0)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
    static_assert(!NodeType::rank(), RANK_MISMATCH);
    T data[1];
    model.template pFill<T>(data, 1);
    ref_ = container_type<T>(1, data[0]);
}

#endif // defined _ENABLE_OPENCL

/* ---------------------- Assignment ---------------------- */

template<typename T, template<class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator=(Tensor<T, 0, C> const &tensor)
{
    ref_[offset_] = tensor.ref_[tensor.offset_];
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator=(Tensor<T, 0, C> &&tensor) noexcept
{
    ref_[offset_] = std::move(tensor.ref_[tensor.offset_]);
    return *this;
}

template<typename T, template<class> class C>
template<typename U, template<class> class C_>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator=(Tensor<U, 0, C_> const &tensor)
{
    ref_[offset_] = tensor.ref_[tensor.offset_];
    return *this;
}

template<typename T, template<class> class C>
template<typename U, template<class> class C_>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator=(Tensor<U, 0, C_> &&tensor)
{
    ref_[offset_] = std::move(tensor.ref_[tensor.offset_]);
    return *this;
}

template<typename T, template<class> class C>
template<typename X, typename>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator=(X &&elem)
{
    ref_[offset_] = std::forward<X>(elem);
    return *this;
}

template<typename U, template<class> class C_>
void Set(Tensor<U, 0, C_> &tensor, Indices<0> const &, U &&value)
{
    tensor.ref_.Set(tensor.offset_, std::forward<U>(value));
}

template<typename U, template<class> class C_>
inline void Set(Tensor<U, 0, C_> &tensor, U &&value)
{
    tensor.ref_.Set(tensor.offset_, std::forward<U>(value));
}

/* ------------------------ Equivalence ----------------------- */

template<typename T, template<class> class C>
bool Tensor<T, 0, C>::operator==(Tensor<T, 0, C> const &tensor) const
{
    return ref_[offset_] == tensor.ref_[tensor.offset_];
}

template<typename T, template<class> class C>
template<typename X>
bool Tensor<T, 0, C>::operator==(Tensor<X, 0, C> const &tensor) const
{
    return ref_[offset_] == tensor.ref_[tensor.offset_];
}

template<typename T, template<class> class C>
template<typename X, typename>
bool Tensor<T, 0, C>::operator==(X val) const
{
    return ref_[offset_] == val;
}

/* ----------------------- Expressions ----------------------- */

template<typename X, typename Y, template<class> class C1,
        template<class> class C2>
Tensor<X, 0, C1> add(Tensor<X, 0, C1> const &tensor_1, Tensor<Y, 0, C2> const &tensor_2)
{
    return Tensor<X, 0, C1>(tensor_1() + tensor_2());
}

template<typename X, typename Y, template<class> class C1,
        template<class> class C2>
Tensor<X, 0, C1> sub(Tensor<X, 0, C1> const &tensor_1, Tensor<Y, 0, C2> const &tensor_2)
{
    return Tensor<X, 0, C2>(tensor_1() - tensor_2());
}

template<typename T, template<class> class C>
template<typename RHS>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator+=(Expression<RHS> const &rhs)
{
    auto scalar = rhs.self()();
    static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
    *this = *this + scalar;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator+=(T const &scalar)
{
    ref_[offset_] += scalar;
    return *this;
}

template<typename T, template<class> class C>
template<typename RHS>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator-=(Expression<RHS> const &rhs)
{
    auto scalar = rhs.self()();
    static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
    *this = *this - scalar;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator-=(T const &scalar)
{
    ref_[offset_] -= scalar;
    return *this;
}

// Directly overload operators for scalar multiplication so template expressions
// don't have to deal with them.

/** Creates a Value equal to value of `tensor_1` plus `tensor_2` */
template<typename X, typename Y, template<class> class C_>
X operator+(Tensor<X, 0, C_> const &tensor_1, Tensor<Y, 0, C_> const &tensor_2)
{
    return tensor_1() + tensor_2();
}

/** Creates a Value equal to value of `tensor` plus `scalar` */
template<typename X, template<class> class C_>
X operator+(Tensor<X, 0, C_> const &tensor, X const &scalar)
{
    return tensor() + scalar;
}

/** Creates a Value equal to value of `scalar` plus `tensor` */
template<typename X, template<class> class C_>
X operator+(X const &scalar, Tensor<X, 0, C_> const &tensor)
{
    return tensor() + scalar;
}

/** Creates a Value equal to value of `tensor_1` minus `tensor_2` */
template<typename X, typename Y, template<class> class C_>
X operator-(Tensor<X, 0, C_> const &tensor_1, Tensor<Y, 0, C_> const &tensor_2)
{
    return tensor_1() - tensor_2();
}

/** Creates a Value equal to value of `tensor` minus `scalar` */
template<typename X, template<class> class C_>
X operator-(Tensor<X, 0, C_> const &tensor, X const &scalar)
{
    return tensor() - scalar;
}

/** Creates a Value equal to value of `scalar` minus `tensor` */
template<typename X, template<class> class C_>
X operator-(X const &scalar, Tensor<X, 0, C_> const &tensor)
{
    return scalar - tensor();
}

/** Creates a Value equal to value of `tensor_1` modulus `tensor_2` */
template<typename X, typename Y, template<class> class C1,
        template<class> class C2>
X operator%(Tensor<X, 0, C1> const &tensor_1, Tensor<Y, 0, C2> const &tensor_2)
{
    return tensor_1() % tensor_2();
}

/** Creates a Value equal to value of `tensor` modulus `scalar` */
template<typename X, template<class> class C_>
X operator%(Tensor<X, 0, C_> const &tensor, X const &scalar)
{
    return tensor() % scalar;
}

/** Creates a Value equal to value of `scalar` modulus `tensor` */
template<typename X, template<class> class C_>
X operator%(X const &scalar, Tensor<X, 0, C_> const &tensor)
{
    return tensor() % scalar;
}

/** Creates a Value equal to value of `tensor_1` multplied by `tensor_2` */
template<typename X, typename Y, template<class> class C1,
        template<class> class C2>
X operator*(Tensor<X, 0, C1> const &tensor_1, Tensor<Y, 0, C2> const &tensor_2)
{
    return tensor_1() * tensor_2();
}

/** Creates a Value equal to value of `tensor_1` multplied by `scalar` */
template<typename X, template<class> class C_>
X operator*(Tensor<X, 0, C_> const &tensor, X const &scalar)
{
    return tensor() * scalar;
}

/** Creates a Value euqal to `scalar` multplied by the value of `tensor` */
template<typename X, template<class> class C_>
X operator*(X const &scalar, Tensor<X, 0, C_> const &tensor)
{
    return tensor() * scalar;
}

template<typename T, template<class> class C>
template<typename RHS>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator*=(Expression<RHS> const &rhs)
{
    auto scalar = rhs.self()();
    static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
    *this = *this * scalar;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator*=(T const &scalar)
{
    ref_[offset_] *= scalar;
    return *this;
}

/** Creates a Scalar with the value of `tensor_1` divided by `tensor_2` */
template<typename X, typename Y, template<class> class C1,
        template<class> class C2>
Tensor<X, 0, C1> operator/(
        Tensor<X, 0, C1> const &tensor_1,
        Tensor<Y, 0, C2> const &tensor_2
)
{
    return Tensor<X, 0, C1>(tensor_1() / tensor_2());
}

/** Creates a Scalar with the value of `tensor` divided by scalar */
template<typename X, template<class> class C_>
Tensor<X, 0, C_> operator/(Tensor<X, 0, C_> const &tensor, X const &scalar)
{
    return Tensor<X, 0, C_>(tensor() / scalar);
}

/** Creates a Scalar with `scalar` divided by `tensor` */
template<typename X, template<class> class C_>
Tensor<X, 0, C_> operator/(X const &scalar, Tensor<X, 0, C_> const &tensor)
{
    return Tensor<X, 0, C_>(tensor() / scalar);
}

template<typename T, template<class> class C>
template<typename RHS>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator/=(Expression<RHS> const &rhs)
{
    auto scalar = rhs.self()();
    static_assert(Rank<decltype(scalar)>::value == 0, EXPECTED_SCALAR);
    *this = *this / scalar;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C> &Tensor<T, 0, C>::operator/=(T const &scalar)
{
    ref_[offset_] /= scalar;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C> Tensor<T, 0, C>::neg() const
{
    return Tensor<T, 0, C>(-ref_[offset_]);
}

/* ------------------------ OpenCL --------------------------- */

#ifdef _ENABLE_OPENCL

template<typename T, template<class> class C>
std::string Tensor<T, 0, C>::opencl_kernel_code(
        cl::CommandQueue &,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    cl_int cl_status = 0;
    buffers.push_back(
            cl::Buffer(
                    opencl::Info::v().context(),
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR |
                    CL_MEM_COPY_HOST_PTR, sizeof(T), ref_.get(), &cl_status
            ));
    assert((cl_status == CL_SUCCESS) && OPENCL_BUFFER_ERROR);

    using namespace opencl::details; // WARNING -- using namespace
    // add the new tensor to the argument list and the expression
    size_t arg_index = buffers.size() - 1;
    arg_list += std::string(cGlobalIdentifier) + " " + (OpenCLType<T>::value) + " "
                + cConstIdentifier + " " + cVariablePrefix
                + std::to_string(arg_index) + ", ";

    // return the name of the element
    return cVariablePrefix + std::to_string(arg_index);
}

template<typename T, template<class> class C>
cl::Buffer Tensor<T, 0, C>::opencl_evaluate_kernel(
        cl::CommandQueue &, std::vector<cl::Buffer> &buffers, std::string &, std::string &
) const noexcept
{
    return buffers.back();
}

#endif // defined _ENABLE_OPENCL

/* ------------------------ Utility --------------------------- */

template<typename T, template<class> class C>
Tensor<T, 0, C>::Tensor(size_t const *, size_t const *, size_t offset, C<T> &&ref)
        : offset_(offset), ref_(std::move(ref))
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::Tensor(size_t const *, size_t const *, size_t offset, C<T> const &ref)
        : offset_(offset), ref_(ref)
{
#ifdef _TEST
    ++eDebugConstructorCounter;
#endif
}

/* ------------------------ Overloads ------------------------ */

template<typename T, template<class> class C>
std::string Tensor<T, 0, C>::str() const
{
    return StringFormat<value_type>{}(ref_[offset_]);
}

/* ------------------- Useful Functions ---------------------- */

template<typename T, template<class> class C>
Tensor<T, 0, C> Tensor<T, 0, C>::copy() const
{
    return Tensor<T, 0, C>(nullptr, nullptr, 0, std::make_shared<C<T>>(1, ref_[offset_]));
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::Proxy Tensor<T, 0, C>::ref()
{
    return Proxy(*this);
}

/* ---------------------- Iterators ------------------------- */

template<typename T, template<class> class C>
Tensor<T, 0, C>::Iterator::Iterator(Tensor<T, 1, C> const &tensor, size_t)
        : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0])
{}

template<typename T, template<class> class C>
Tensor<T, 0, C>::Iterator::Iterator(Iterator const &it)
        : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::Iterator::operator=(Iterator const &it) -> Iterator &
{
    offset_ = it.offset_;
    ref_ = it.ref_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::Iterator::Iterator(Iterator &&it) noexcept
        : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::Iterator::operator=(Iterator &&it) noexcept -> Iterator &
{
    offset_ = it.offset_;
    ref_ = std::move(it.ref_);
    stride_ = it.stride_;
    return *this;
}

template<typename T, template<class> class C>
constexpr T &Tensor<T, 0, C>::Iterator::operator*() const
{
    return const_cast<C<T> &>(ref_)[offset_];
}

template<typename T, template<class> class C>
constexpr T *Tensor<T, 0, C>::Iterator::operator->() const
{
    return &const_cast<C<T> &>(ref_)[offset_];
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::Iterator const Tensor<T, 0, C>::Iterator::operator++(int)
{
    Tensor<T, 0, C>::Iterator it{*this};
    ++(*this);
    return it;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::Iterator &Tensor<T, 0, C>::Iterator::operator++()
{
    offset_ += stride_;
    return *this;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::Iterator const Tensor<T, 0, C>::Iterator::operator--(int)
{
    Tensor<T, 0, C>::Iterator it{*this};
    --(*this);
    return it;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::Iterator &Tensor<T, 0, C>::Iterator::operator--()
{
    offset_ -= stride_;
    return *this;
}

template<typename T, template<class> class C>
bool Tensor<T, 0, C>::Iterator::operator==(
        typename Tensor<T, 0, C>::Iterator const &it
) const
{
    if (ref_ != it.ref_) return false;
    if (stride_ != it.stride_) return false;
    return offset_ == it.offset_;
}

template<typename T, template<class> class C>
std::ptrdiff_t Tensor<T, 0, C>::Iterator::operator-(
        typename Tensor<T, 0, C>::Iterator const &it
) const noexcept
{
    assert(ref_ == it.ref_);
    assert(stride_ == it.stride_);
    return ((std::ptrdiff_t) offset_ - (std::ptrdiff_t) it.offset_) / (std::ptrdiff_t) stride_;
}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::Iterator::operator+(std::ptrdiff_t incr) const -> Iterator
{
    auto incr_it = Tensor<T, 0, C>::Iterator(*this);
    incr_it.offset_ += incr * incr_it.stride_;
    return incr_it;
}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::Iterator::operator-(std::ptrdiff_t decr) const -> Iterator
{
    auto incr_it = Tensor<T, 0, C>::Iterator(*this);
    incr_it.offset_ -= decr * incr_it.stride_;
    return incr_it;
}

template<typename U, template<class> class C_>
typename Tensor<U, 0, C_>::Iterator
operator+(std::ptrdiff_t incr, typename Tensor<U, 0, C_>::Iterator const &it)
{
    return it + incr;
}

template<typename U, template<class> class C_>
typename Tensor<U, 0, C_>::Iterator
operator-(std::ptrdiff_t decr, typename Tensor<U, 0, C_>::Iterator const &it)
{
    return it - decr;
}

/** Returns true iff the offset difference is positive */
template<typename U, template<class> class C_>
bool Tensor<U, 0, C_>::Iterator::operator>(Iterator const &it) const
{
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ > it.offset_;
}

/** Returns true iff the offset difference is non-negative */
template<typename U, template<class> class C_>
bool Tensor<U, 0, C_>::Iterator::operator>=(Iterator const &it) const
{
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ >= it.offset_;
}

/* ------------------- ConstIterators ---------------------- */

template<typename T, template<class> class C>
Tensor<T, 0, C>::ConstIterator::ConstIterator(Tensor<T, 1, C> const &tensor, size_t)
        : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0])
{}

template<typename T, template<class> class C>
Tensor<T, 0, C>::ConstIterator::ConstIterator(ConstIterator const &it)
        : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ConstIterator::operator=(ConstIterator const &it)
-> ConstIterator &
{
    offset_ = it.offset_;
    ref_ = it.ref_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::ConstIterator::ConstIterator(ConstIterator &&it) noexcept
        : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ConstIterator::operator=(ConstIterator &&it) noexcept
-> ConstIterator &
{
    offset_ = it.offset_;
    ref_ = std::move(it.ref_);
    stride_ = it.stride_;
    return *this;
}

template<typename T, template<class> class C>
constexpr T const &Tensor<T, 0, C>::ConstIterator::operator*() const
{
    return static_cast<C<T> const &>(ref_)[offset_];
}

template<typename T, template<class> class C>
constexpr T const *Tensor<T, 0, C>::ConstIterator::operator->() const
{
    return &static_cast<C<T> const &>(ref_)[offset_];
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ConstIterator const Tensor<T, 0, C>::ConstIterator::operator++(int)
{
    Tensor<T, 0, C>::ConstIterator it{*this};
    ++(*this);
    return it;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ConstIterator &Tensor<T, 0, C>::ConstIterator::operator++()
{
    offset_ += stride_;
    return *this;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ConstIterator const Tensor<T, 0, C>::ConstIterator::operator--(int)
{
    Tensor<T, 0, C>::ConstIterator it{*this};
    --(*this);
    return it;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ConstIterator &Tensor<T, 0, C>::ConstIterator::operator--()
{
    offset_ -= stride_;
    return *this;
}

template<typename T, template<class> class C>
bool Tensor<T, 0, C>::ConstIterator::operator==(
        typename Tensor<T, 0, C>::ConstIterator const &it
) const
{
    if (ref_ != it.ref_) return false;
    return offset_ == it.offset_;
}

template<typename T, template<class> class C>
std::ptrdiff_t Tensor<T, 0, C>::ConstIterator::operator-(
        typename Tensor<T, 0, C>::ConstIterator const &it
) const noexcept
{
    assert(ref_ == it.ref_);
    assert(stride_ == it.stride_);
    return ((std::ptrdiff_t) offset_ - (std::ptrdiff_t) it.offset_) / (std::ptrdiff_t) stride_;
}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ConstIterator::operator+(std::ptrdiff_t incr) const -> ConstIterator
{
    auto incr_it = Tensor<T, 0, C>::ConstIterator(*this);
    incr_it.offset_ += incr * incr_it.stride_;
    return incr_it;
}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ConstIterator::operator-(std::ptrdiff_t decr) const -> ConstIterator
{
    auto incr_it = Tensor<T, 0, C>::ConstIterator(*this);
    incr_it.offset_ -= decr * incr_it.stride_;
    return incr_it;
}

template<typename U, template<class> class C_>
typename Tensor<U, 0, C_>::ConstIterator
operator+(std::ptrdiff_t incr, typename Tensor<U, 0, C_>::ConstIterator const &it)
{
    return it + incr;
}

template<typename U, template<class> class C_>
typename Tensor<U, 0, C_>::ConstIterator
operator-(std::ptrdiff_t decr, typename Tensor<U, 0, C_>::ConstIterator const &it)
{
    return it - decr;
}

/** Returns true iff the offset difference is positive */
template<typename U, template<class> class C_>
bool Tensor<U, 0, C_>::ConstIterator::operator>(ConstIterator const &it) const
{
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ > it.offset_;
}

/** Returns true iff the offset difference is non-negative */
template<typename U, template<class> class C_>
bool Tensor<U, 0, C_>::ConstIterator::operator>=(ConstIterator const &it) const
{
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ >= it.offset_;
}

/* -------------------- ReverseIterator ---------------------- */

template<typename T, template<class> class C>
Tensor<T, 0, C>::ReverseIterator::ReverseIterator(Tensor<T, 1, C> const &tensor, size_t)
        : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0])
{
    offset_ += stride_ * (tensor.shape_[0] - 1);
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::ReverseIterator::ReverseIterator(ReverseIterator const &it)
        : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ReverseIterator::operator=(ReverseIterator const &it)
-> ReverseIterator &
{
    offset_ = it.offset_;
    ref_ = it.ref_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::ReverseIterator::ReverseIterator(ReverseIterator &&it) noexcept
        : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ReverseIterator::operator=(ReverseIterator &&it) noexcept
-> ReverseIterator &
{
    offset_ = it.offset_;
    ref_ = std::move(it.ref_);
    stride_ = it.stride_;
    return *this;
}

template<typename T, template<class> class C>
constexpr T &Tensor<T, 0, C>::ReverseIterator::operator*() const
{
    return const_cast<C<T> &>(ref_)[offset_];
}

template<typename T, template<class> class C>
constexpr T *Tensor<T, 0, C>::ReverseIterator::operator->() const
{
    return &const_cast<C<T> &>(ref_)[offset_];
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ReverseIterator const Tensor<T, 0, C>::ReverseIterator::operator++(int)
{
    auto it = Tensor<T, 0, C>::ReverseIterator{*this};
    ++(*this);
    return it;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ReverseIterator &Tensor<T, 0, C>::ReverseIterator::operator++()
{
    offset_ -= stride_;
    return *this;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ReverseIterator const Tensor<T, 0, C>::ReverseIterator::operator--(int)
{
    auto it = Tensor<T, 0, C>::ReverseIterator{*this};
    --(*this);
    return it;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ReverseIterator &Tensor<T, 0, C>::ReverseIterator::operator--()
{
    offset_ += stride_;
    return *this;
}

template<typename T, template<class> class C>
bool Tensor<T, 0, C>::ReverseIterator::operator==(
        typename Tensor<T, 0, C>::ReverseIterator const &it
) const
{
    if (ref_ != it.ref_)
        return false;
    return offset_ == it.offset_;
}

template<typename T, template<class> class C>
std::ptrdiff_t Tensor<T, 0, C>::ReverseIterator::operator-(
        typename Tensor<T, 0, C>::ReverseIterator const &it
) const noexcept
{
    assert(ref_ == it.ref_);
    assert(stride_ == it.stride_);
    return ((std::ptrdiff_t) it.offset_ - (std::ptrdiff_t) offset_) / (std::ptrdiff_t) stride_;
}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ReverseIterator::operator+(std::ptrdiff_t incr) const -> ReverseIterator
{
    auto incr_it = Tensor<T, 0, C>::ReverseIterator(*this);
    incr_it.offset_ -= incr * incr_it.stride_;
    return incr_it;
}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ReverseIterator::operator-(std::ptrdiff_t decr) const -> ReverseIterator
{
    auto incr_it = Tensor<T, 0, C>::ReverseIterator(*this);
    incr_it.offset_ += decr * incr_it.stride_;
    return incr_it;
}

template<typename U, template<class> class C_>
typename Tensor<U, 0, C_>::ReverseIterator
operator+(std::ptrdiff_t incr, typename Tensor<U, 0, C_>::ReverseIterator const &it)
{
    return it + incr;
}

template<typename U, template<class> class C_>
typename Tensor<U, 0, C_>::ReverseIterator
operator-(std::ptrdiff_t decr, typename Tensor<U, 0, C_>::ReverseIterator const &it)
{
    return it - decr;
}

/** Returns true iff the offset difference is positive */
template<typename U, template<class> class C_>
bool Tensor<U, 0, C_>::ReverseIterator::operator>(ReverseIterator const &it) const
{
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ < it.offset_;
}

/** Returns true iff the offset difference is non-negative */
template<typename U, template<class> class C_>
bool Tensor<U, 0, C_>::ReverseIterator::operator>=(ReverseIterator const &it) const
{
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ <= it.offset_;
}

/* ----------------- ConstReverseIterator ------------------- */

template<typename T, template<class> class C>
Tensor<T, 0, C>::ConstReverseIterator::ConstReverseIterator(
        Tensor<T, 1, C> const &tensor,
        size_t
) : offset_(tensor.offset_), ref_(tensor.ref_), stride_(tensor.strides_[0])
{
    offset_ += stride_ * (tensor.shape_[0] - 1);
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator const &it)
        : offset_(it.offset_), ref_(it.ref_), stride_(it.stride_)
{}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ConstReverseIterator::operator=(ConstReverseIterator const &it)
-> ConstReverseIterator &
{
    offset_ = it.offset_;
    ref_ = it.ref_;
    stride_ = it.stride_;
    return *this;
}

template<typename T, template<class> class C>
Tensor<T, 0, C>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator &&it) noexcept
        : offset_(it.offset_), ref_(std::move(it.ref_)), stride_(it.stride_)
{}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ConstReverseIterator::operator=(ConstReverseIterator &&it) noexcept
-> ConstReverseIterator &
{
    offset_ = it.offset_;
    ref_ = std::move(it.ref_);
    stride_ = it.stride_;
    return *this;
}

template<typename T, template<class> class C>
constexpr T const &Tensor<T, 0, C>::ConstReverseIterator::operator*() const
{
    return static_cast<C<T> const &>(ref_)[offset_];
}

template<typename T, template<class> class C>
constexpr T const *Tensor<T, 0, C>::ConstReverseIterator::operator->() const
{
    return &static_cast<C<T> const &>(ref_)[offset_];
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ConstReverseIterator const
Tensor<T, 0, C>::ConstReverseIterator::operator++(int)
{
    auto it = Tensor<T, 0, C>::ConstReverseIterator{*this};
    ++(*this);
    return it;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ConstReverseIterator &
Tensor<T, 0, C>::ConstReverseIterator::operator++()
{
    offset_ -= stride_;
    return *this;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ConstReverseIterator const
Tensor<T, 0, C>::ConstReverseIterator::operator--(int)
{
    auto it = Tensor<T, 0, C>::ConstReverseIterator{*this};
    --(*this);
    return it;
}

template<typename T, template<class> class C>
typename Tensor<T, 0, C>::ConstReverseIterator &
Tensor<T, 0, C>::ConstReverseIterator::operator--()
{
    offset_ += stride_;
    return *this;
}

template<typename T, template<class> class C>
bool Tensor<T, 0, C>::ConstReverseIterator::operator==(
        typename Tensor<T, 0, C>::ConstReverseIterator const &it
) const
{
    if (ref_ != it.ref_) return false;
    return offset_ == it.offset_;
}

template<typename T, template<class> class C>
std::ptrdiff_t Tensor<T, 0, C>::ConstReverseIterator::operator-(
        typename Tensor<T, 0, C>::ConstReverseIterator const &it
) const noexcept
{
    assert(ref_ == it.ref_);
    assert(stride_ == it.stride_);
    return ((std::ptrdiff_t) it.offset_ - (std::ptrdiff_t) offset_) / (std::ptrdiff_t) stride_;
}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ConstReverseIterator::operator+(std::ptrdiff_t incr) const
-> ConstReverseIterator
{
    auto incr_it = Tensor<T, 0, C>::ConstReverseIterator(*this);
    incr_it.offset_ -= incr * incr_it.stride_;
    return incr_it;
}

template<typename T, template<class> class C>
auto Tensor<T, 0, C>::ConstReverseIterator::operator-(std::ptrdiff_t decr) const
-> ConstReverseIterator
{
    auto incr_it = Tensor<T, 0, C>::ConstReverseIterator(*this);
    incr_it.offset_ += decr * incr_it.stride_;
    return incr_it;
}

template<typename U, template<class> class C_>
typename Tensor<U, 0, C_>::ConstReverseIterator
operator+(std::ptrdiff_t incr, typename Tensor<U, 0, C_>::ConstReverseIterator const &it)
{
    return it + incr;
}

template<typename U, template<class> class C_>
typename Tensor<U, 0, C_>::ConstReverseIterator
operator-(std::ptrdiff_t decr, typename Tensor<U, 0, C_>::ConstReverseIterator const &it)
{
    return it - decr;
}

/** Returns true iff the offset difference is positive */
template<typename U, template<class> class C_>
bool Tensor<U, 0, C_>::ConstReverseIterator::operator>(ConstReverseIterator const &it) const
{
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ < it.offset_;
}

/** Returns true iff the offset difference is non-negative */
template<typename U, template<class> class C_>
bool Tensor<U, 0, C_>::ConstReverseIterator::operator>=(ConstReverseIterator const &it) const
{
    assert(stride_ == it.stride_);
    assert(ref_ == it.ref_);
    return offset_ <= it.offset_;
}

/* ----------------------------- Math ----------------------------- */

namespace math {

/** Vardiac wrapper around std::plus with OpenCL code emission */
struct add {
    template<typename T, typename... Args>
    inline T operator()(T const &arg1, Args const &... args) const;

#ifdef _ENABLE_OPENCL

    /** Creates the reduce expression for the given accumulator and variable names */
    static std::string opencl_reduce(std::string const &accum, std::string const &v1)
    {
        return accum + " += " + v1;
    }

    static constexpr size_t arity()
    { return 0; }

    template<typename... Args>
    static std::string opencl_map(Args const &... args);

#endif
};

template<typename T, typename... Args>
T add::operator()(T const &arg1, Args const &... args) const
{
    T result = arg1;
    VARDIAC_MAP(result += args);
    return result;
}

#ifdef _ENABLE_OPENCL

template<typename... Args>
std::string add::opencl_map(Args const &... args)
{
    // surround expression in brackets to give priority
    auto expr = std::string{"("};
    VARDIAC_MAP((expr += args + " + "));

    // replace trailing " + " with ")";
    expr.pop_back();
    expr.pop_back();
    expr.back() = ')';

    return expr;
}

#endif

/** Vardiac wrapper around std::minus with OpenCL code emission */
struct sub {
    template<typename T, typename... Args>
    inline T operator()(T const &arg1, Args const &... args) const;

#ifdef _ENABLE_OPENCL

    /** Creates the reduce expression for the given accumulator and variable names */
    static std::string opencl_reduce(std::string const &accum, std::string const &v1)
    {
        return accum + " += " + v1;
    }

    /** Expected number of arguments is a compile time value */
    static constexpr size_t arity()
    { return 0; }

    template<typename... Args>
    static std::string opencl_map(Args const &... args);

#endif
};

template<typename T, typename... Args>
inline T sub::operator()(T const &arg1, Args const &... args) const
{
    auto result = T{arg1};
    VARDIAC_MAP(result -= args);
    return result;
}

#ifdef _ENABLE_OPENCL

template<typename... Args>
std::string sub::opencl_map(Args const &... args)
{
    // surround expression in brackets to give priority
    auto expr = std::string{"("};
    VARDIAC_MAP((expr += args + " - "));

    // replace trailing " - " with ")";
    expr.pop_back();
    expr.pop_back();
    expr.back() = ')';

    return expr;
}

#endif

/** Vardiac wrapper around std::multiplies with OpenCL code emission */
struct mul {
    template<typename T, typename... Args>
    inline T operator()(T const &arg1, Args const &... args) const;

#ifdef _ENABLE_OPENCL

    /** Creates the reduce expression for the given accumulator and variable names */
    inline static std::string opencl_reduce(std::string const &accum, std::string const &v1)
    {
        return accum + " *= " + v1;
    }

    static constexpr size_t arity()
    { return 0; }

    template<typename... Args>
    static std::string opencl_map(Args const &... args);

#endif
};

template<typename T, typename... Args>
inline T mul::operator()(T const &arg1, Args const &... args) const
{
    T result = arg1;
    VARDIAC_MAP(result *= args);
    return result;
}

#ifdef _ENABLE_OPENCL

template<typename... Args>
std::string mul::opencl_map(Args const &... args)
{
    // surround expression in brackets to give priority
    auto expr = std::string{"("};
    VARDIAC_MAP((expr += args + " * "));

    // replace trailing " * " with ")";
    expr.pop_back();
    expr.pop_back();
    expr.back() = ')';

    return expr;
}

#endif

/** Vardiac wrapper around std::divides with OpenCL code emission */
struct div {
    template<typename T, typename... Args>
    inline T operator()(T const &arg1, Args const &... args) const;

#ifdef _ENABLE_OPENCL

    /** Creates the reduce expression for the given accumulator and variable names */
    static std::string opencl_reduce(std::string const &accum, std::string const &v1)
    {
        return accum + " *= " + v1;
    }

    static constexpr size_t arity()
    { return 0; }

    template<typename... Args>
    static std::string opencl_map(Args const &... args);

#endif
};

template<typename T, typename... Args>
inline T div::operator()(T const &arg1, Args const &... args) const
{
    T result = arg1;
    VARDIAC_MAP(result /= args);
    return result;
}

#ifdef _ENABLE_OPENCL

template<typename... Args>
std::string div::opencl_map(Args const &... args)
{
    // surround expression in brackets to give priority
    auto expr = std::string{"("};
    VARDIAC_MAP((expr += args + " / "));

    // replace trailing " / " with ")";
    expr.pop_back();
    expr.pop_back();
    expr.back() = ')';

    return expr;
}

#endif

/** Wrapper around std::sin with OpenCL code emission */
struct sin {
    template<typename T>
    T operator()(T const &x) const
    { return std::sin(x); }

#ifdef _ENABLE_OPENCL

    /** OpenCL built in function, along with a single left paranthesis */
    static std::string opencl_map(std::string const &arg)
    { return "sin(" + arg + ")"; }

#endif

    static constexpr size_t arity()
    { return 1; }
};

/** Wrapper around std::cos with OpenCL code emission */
struct cos {
    template<typename T>
    T operator()(T const &x) const
    { return std::cos(x); }

#ifdef _ENABLE_OPENCL

    /** OpenCL built in function, along with a single left paranthesis */
    static std::string opencl_map(std::string const &arg)
    { return "cos(" + arg + ")"; }

#endif

    static constexpr size_t arity()
    { return 1; }
};

/** Wrapper around std::tan with OpenCL code emission */
struct tan {
    template<typename T>
    T operator()(T const &x) const
    { return std::tan(x); }

#ifdef _ENABLE_OPENCL

    /** OpenCL built in function, along with a single left paranthesis */
    static std::string opencl_map(std::string const &arg)
    { return "tan(" + arg + ")"; }

#endif

    static constexpr size_t arity()
    { return 1; }
};

/** Wrapper around std::max with OpenCL code emission */
struct min {
    template<typename T>
    T const &operator()(T const &x, T const &y) const
    { return std::min(x, y); }

#ifdef _ENABLE_OPENCL

    /** OpenCL built in function, along with a single left paranthesis */
    static std::string opencl_map(std::string const &accum, std::string const &v)
    {
        return accum + " = min(" + accum + ", " + v + ")";
    }

    /** Creates the reduce expression for the given accumulator and variable names */
    static std::string opencl_reduce(std::string const &accum, std::string const &v1);

#endif

    static constexpr size_t arity()
    { return 2; }
};

} // namespace math

/* ---------------------------- OpenCL ---------------------------- */

#ifdef _ENABLE_OPENCL
namespace opencl {

template<typename NodeType>
class Model { /*@opencl::Model*/
public:
// FIXME -- Documentation

    /* -------------- Friends -------------- */

    template<typename X, size_t M, template<class> class C_> friend
    class tensor::Tensor;

    template<typename LHS, typename RHS> friend
    class tensor::BinaryAddExpr;

    template<typename LHS, typename RHS> friend
    class tensor::BinarySubExpr;

    template<size_t I1, size_t I2, typename LHS, typename RHS> friend
    class tensor::BinaryMulExpr;

    template<typename LHS, typename RHS> friend
    class tensor::BinaryHadExpr;

    template<typename Function, typename... Exprs> friend
    class tensor::MapExpr;

    template<typename U, typename Function, typename... Exprs> friend
    class tensor::ReduceExpr;

    template<typename RHS> friend
    class tensor::UnaryNegExpr;

    /* ------------- Getters -------------- */

    Shape<NodeType::rank()> const &shape() const noexcept
    { return node_.shape(); }

    constexpr static size_t rank()
    { return NodeType::rank(); }

private:

    /* --------------- Data --------------- */

    cl::Buffer buffer_;
    cl::CommandQueue cqueue_;
    NodeType const &node_;

    /* ------------- Methods -------------- */

    explicit Model(Expression<NodeType> const &expr);

    template<typename U>
    void pFill(U *ptr, size_t num_elems) const;

}; // Model

template<typename NodeType>
Model<NodeType>::Model(Expression<NodeType> const &expr)
        : cqueue_(Info::v().context(), Info::v().device()), node_(expr.self())
{
    auto buffers = std::vector<cl::Buffer>{};
    auto cl_arg_list = std::string{};
    std::string cl_expr = node_.opencl_kernel_code(cqueue_, buffers, cl_arg_list);
    buffer_ = node_.opencl_evaluate_kernel(cqueue_, buffers, cl_arg_list, cl_expr);
}

template<typename NodeType>
template<typename U>
void Model<NodeType>::pFill(U *ptr, size_t num_elems) const
{
    cl_int err = cqueue_.enqueueReadBuffer(buffer_, CL_TRUE, 0, num_elems * sizeof(U), ptr);
    assert((err == CL_SUCCESS) && OPENCL_BUFFER_ERROR);
}

} // namespace opencl
#endif // if defined _ENABLE_OPENCL

/* ---------------------- Expressions ------------------------ */

template<typename LHS, typename RHS>
class BinaryAddExpr : public Expression<BinaryAddExpr<LHS, RHS>> {
/*@BinaryAddExpr<LHS, RHS>*/
// FIXME -- Documentation
public:

    /* ---------------- typedefs --------------- */

    template<typename X>
    using container_type = typename LHS::template container_type<X>;
    typedef typename LHS::value_type value_type;

    constexpr static size_t rank()
    { return LHS::rank(); }

    typedef BinaryAddExpr self_type;
    typedef typename LHS::return_type return_type;

    /* ---------------- Friends ---------------- */

    template<typename LHS_, typename RHS_>
    friend BinaryAddExpr<LHS_, RHS_>
    operator+(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

    template<typename LHS_, typename RHS_>
    friend BinaryAddExpr<LHS_, RHS_>
    _add_(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

    template<typename LHS_, typename RHS_> friend
    class BinaryAddExpr;

    template<typename LHS_, typename RHS_> friend
    class BinarySubExpr;

    template<size_t I1_, size_t I2_, typename LHS_, typename RHS_> friend
    class BinaryMulExpr;

    template<typename LHS_, typename RHS_> friend
    class BinaryHadExpr;

    template<typename Function_, typename... Exprs_> friend
    class MapExpr;

    template<typename U, typename Function_, typename... Exprs_> friend
    class ReduceExpr;

    template<typename RHS_> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ---------------- Getters ----------------- */

    size_t dimension(size_t index) const
    { return lhs_.dimension(index); }

    Shape<LHS::rank()> const &shape() const
    { return lhs_.shape(); }

    /** Evaluate and return the resulting Tensor */
    return_type eval() const
    { return (*this)(); }

    template<typename... Args>
    auto operator()(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type;

    template<typename... Args>
    auto at(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type;

    template<size_t M>
    auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type;

    template<size_t... Slices, typename... Args>
    auto slice(Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<LHS const>().slice(args...))>::type;

    template<size_t... Slices, size_t M>
    auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(
    std::declval<LHS const>().slice(indices))>::type;

    /** Produces "(+ `lhs`, `rhs`)" */
    std::string str() const;

    /* ---------------- OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

    opencl::Model<self_type> opencl() const
    { return opencl::Model<self_type>(*this); }

    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

private:

    /* -------------- Constructors -------------- */

    BinaryAddExpr(LHS const &lhs, RHS const &rhs);

    BinaryAddExpr(BinaryAddExpr<LHS, RHS> const &) = default;

    /* ------------------ Data ------------------ */

    LHS const &lhs_;
    RHS const &rhs_;
};

/* ---------------- Friends ---------------- */

template<typename LHS, typename RHS>
BinaryAddExpr<LHS, RHS> operator+(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    return BinaryAddExpr<LHS, RHS>(lhs.self(), rhs.self());
}

template<typename LHS, typename RHS>
BinaryAddExpr<LHS, RHS> _add_(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    return BinaryAddExpr<LHS, RHS>(lhs.self(), rhs.self());
}

/* ---------------- Getters ---------------- */

template<typename LHS, typename RHS>
template<typename... Args>
auto BinaryAddExpr<LHS, RHS>::operator()(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return lhs_(args...) + rhs_(args...);
}

template<typename LHS, typename RHS>
template<typename... Args>
auto BinaryAddExpr<LHS, RHS>::at(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return lhs_(args...) + rhs_(args...);
}

template<typename LHS, typename RHS>
template<size_t M>
auto BinaryAddExpr<LHS, RHS>::operator[](Indices<M> const &indices) const
-> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    return lhs_[indices] + rhs_[indices];
}

template<typename LHS, typename RHS>
template<size_t... Slices, typename... Args>
auto BinaryAddExpr<LHS, RHS>::slice(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().slice(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return lhs_.template slice<Slices...>(args...) + rhs_.template slice<Slices...>(args...);
}

template<typename LHS, typename RHS>
template<size_t... Slices, size_t M>
auto BinaryAddExpr<LHS, RHS>::slice(Indices<M> const &indices) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().slice(indices))>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    return lhs_.template slice<Slices...>(indices) + rhs_.template slice<Slices...>(indices);
}

template<typename LHS, typename RHS>
std::string BinaryAddExpr<LHS, RHS>::str() const
{
    return "(+ " + lhs_.str() + " " + rhs_.str() + ")";
}

/* -------------- Constructors -------------- */

template<typename LHS, typename RHS>
BinaryAddExpr<LHS, RHS>::BinaryAddExpr(LHS const &lhs, RHS const &rhs)
        : lhs_(lhs), rhs_(rhs)
{}

/* ----------------- OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

template<typename LHS, typename RHS>
std::string BinaryAddExpr<LHS, RHS>::opencl_kernel_code(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    std::string expr = std::string("(") + lhs_.opencl_kernel_code(cqueue, buffers, arg_list)
                       + " +  " + rhs_.opencl_kernel_code(cqueue, buffers, arg_list) + ")";
    return expr;
}

template<typename LHS, typename RHS>
cl::Buffer BinaryAddExpr<LHS, RHS>::opencl_evaluate_kernel(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list,
        std::string &expr
) const noexcept
{
    return opencl::details::evaluate_basic_kernel<value_type>(
            cqueue,
            buffers,
            arg_list,
            expr,
            lhs_.shape().index_product()
    );
}

#endif

/* ----------------------------------------- */

template<typename LHS, typename RHS>
class BinarySubExpr : public Expression<BinarySubExpr<LHS, RHS>> {
/*@BinarySubExpr<LHS, RHS>*/
public:
    /* ---------------- typedefs --------------- */

    typedef typename LHS::value_type value_type;
    template<typename X>
    using container_type = typename LHS::template container_type<X>;
    typedef BinarySubExpr self_type;

    constexpr static size_t rank()
    { return LHS::rank(); }

    typedef typename LHS::return_type return_type;

    /* ---------------- Friends ---------------- */

    template<typename LHS_, typename RHS_>
    friend BinarySubExpr<LHS_, RHS_>
    operator-(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

    template<typename LHS_, typename RHS_>
    friend BinarySubExpr<LHS_, RHS_>
    _sub_(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

    template<typename LHS_, typename RHS_> friend
    class BinaryAddExpr;

    template<typename LHS_, typename RHS_> friend
    class BinarySubExpr;

    template<size_t I1_, size_t I2_, typename LHS_, typename RHS_> friend
    class BinaryMulExpr;

    template<typename LHS_, typename RHS_> friend
    class BinaryHadExpr;

    template<typename Function_, typename... Exprs_> friend
    class MapExpr;

    template<typename U, typename Function_, typename... Exprs_> friend
    class ReduceExpr;

    template<typename RHS_> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ---------------- Getters ----------------- */

    size_t dimension(size_t index) const
    { return lhs_.dimension(index); }

    Shape<LHS::rank()> const &shape() const
    { return lhs_.shape(); }

    /** Evaluate and return the resulting Tensor */
    return_type eval() const
    { return (*this)(); }

    template<typename... Args>
    auto operator()(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type;

    template<typename... Args>
    auto at(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type;

    template<size_t M>
    auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type;

    template<size_t... Slices, typename... Args>
    auto slice(Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<LHS const>().slice(args...))>::type;

    template<size_t... Slices, size_t M>
    auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(
    std::declval<LHS const>().slice(indices))>::type;

    /** Produces "(- `lhs`, `rhs`)" */
    std::string str() const;

    /* ---------------- OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

    opencl::Model<self_type> opencl() const
    { return opencl::Model<self_type>(*this); }

    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

private:

    /* -------------- Constructors -------------- */

    BinarySubExpr(LHS const &lhs, RHS const &rhs);

    BinarySubExpr(BinarySubExpr<LHS, RHS> const &) = default;

    /* ------------------ Data ------------------ */

    LHS const &lhs_;
    RHS const &rhs_;
};

/* ----------------- Friends ----------------- */

template<typename LHS, typename RHS>
BinarySubExpr<LHS, RHS> operator-(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    return BinarySubExpr<LHS, RHS>(lhs.self(), rhs.self());
}

template<typename LHS, typename RHS>
BinarySubExpr<LHS, RHS> _sub_(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    return BinarySubExpr<LHS, RHS>(lhs.self(), rhs.self());
}

/* ----------------- Getters ----------------- */

template<typename LHS, typename RHS>
template<typename... Args>
auto BinarySubExpr<LHS, RHS>::operator()(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type
{
    static_assert(rank() >= sizeof...(Args), RANK_OUT_OF_BOUNDS);
    return lhs_(args...) - rhs_(args...);
}

template<typename LHS, typename RHS>
template<typename... Args>
auto BinarySubExpr<LHS, RHS>::at(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return lhs_(args...) - rhs_(args...);
}

template<typename LHS, typename RHS>
template<size_t M>
auto BinarySubExpr<LHS, RHS>::operator[](Indices<M> const &indices) const
-> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    return lhs_[indices] - rhs_[indices];
}

template<typename LHS, typename RHS>
template<size_t... Slices, typename... Args>
auto BinarySubExpr<LHS, RHS>::slice(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().slice(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return lhs_.template slice<Slices...>(args...) - rhs_.template slice<Slices...>(args...);
}

template<typename LHS, typename RHS>
template<size_t... Slices, size_t M>
auto BinarySubExpr<LHS, RHS>::slice(Indices<M> const &indices) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().slice(indices))>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    return lhs_.template slice<Slices...>(indices) - rhs_.template slice<Slices...>(indices);
}

template<typename LHS, typename RHS>
std::string BinarySubExpr<LHS, RHS>::str() const
{
    return "(- " + lhs_.str() + " " + rhs_.str() + ")";
}

/* ----------------- OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

template<typename LHS, typename RHS>
std::string BinarySubExpr<LHS, RHS>::opencl_kernel_code(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    return std::string("(") + lhs_.opencl_kernel_code(cqueue, buffers, arg_list)
           + " - " + rhs_.opencl_kernel_code(cqueue, buffers, arg_list) + ")";
}

template<typename LHS, typename RHS>
cl::Buffer BinarySubExpr<LHS, RHS>::opencl_evaluate_kernel(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list,
        std::string &expr
) const noexcept
{
    return opencl::details::evaluate_basic_kernel<value_type>(
            cqueue,
            buffers,
            arg_list,
            expr, lhs_.shape().index_product()
    );
}

#endif

/* -------------- Constructors -------------- */

template<typename LHS, typename RHS>
BinarySubExpr<LHS, RHS>::BinarySubExpr(LHS const &lhs, RHS const &rhs)
        : lhs_(lhs), rhs_(rhs)
{}

/**
 * Multiplies the inner dimensions
 * i.e. 3x4x5 * 5x4x3 produces a 3x4x4x3 tensor
 */
template<size_t I1, size_t I2, typename LHS, typename RHS>
class BinaryMulExpr : public Expression<BinaryMulExpr<I1, I2, LHS, RHS>> {
/*@BinaryMulExpr*/
public:

    /* ---------------- typedefs --------------- */

    constexpr static size_t rank()
    { return LHS::rank() + RHS::rank() - 2; }

    typedef typename LHS::value_type value_type;
    template<typename X>
    using container_type = typename LHS::template container_type<X>;
    typedef BinaryMulExpr self_type;
    typedef
    Tensor<value_type, self_type::rank(), container_type> return_type;

    /* ---------------- Friends ---------------- */

    template<typename LHS_, typename RHS_>
    friend BinaryMulExpr<LHS_::rank() - 1, 0, LHS_, RHS_>
    operator*(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

    template<size_t I1_, size_t I2_, typename LHS_, typename RHS_>
    friend BinaryMulExpr<I1_, I2_, LHS_, RHS_>
    _mul_(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

    template<typename LHS_, typename RHS_> friend
    class BinaryAddExpr;

    template<typename LHS_, typename RHS_> friend
    class BinarySubExpr;

    template<size_t I1_, size_t I2_, typename LHS_, typename RHS_> friend
    class BinaryMulExpr;

    template<typename LHS_, typename RHS_> friend
    class BinaryHadExpr;

    template<typename Function_, typename... Exprs_> friend
    class MapExpr;

    template<typename U, typename Function_, typename... Exprs_> friend
    class ReduceExpr;

    template<typename RHS_> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ---------------- Getters ----------------- */

    Shape<self_type::rank()> const &shape() const noexcept
    { return shape_; }

    size_t dimension(size_t index) const noexcept;

    /** Evaluate and return the resulting Tensor */
    return_type eval() const
    { return (*this)(); }

    template<
            typename... Args,
            typename = typename std::enable_if<self_type::rank() != sizeof...(Args)>::type
    >
    auto operator()(Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>()(args...))>::type;

    template<typename... Args, typename =
    typename std::enable_if<self_type::rank() == sizeof...(Args)>::type>
    value_type operator()(Args... args) const;

    template<typename... Args>
    auto at(Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().at(args...))>::type;

    template<size_t M, typename =
    typename std::enable_if<self_type::rank() != M>::type>
    auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>()[indices])>::type;

    template<size_t M, typename =
    typename std::enable_if<self_type::rank() == M>::type>
    value_type operator[](Indices<M> const &indices) const;

    template<size_t... Slices, typename... Args>
    auto slice(Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().slice(args...))>::type;

    template<size_t... Slices, size_t M>
    auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().slice(indices))>::type;

    /** Produces "<`I1`, `I2`>(* `lhs`, `rhs`)" */
    std::string str() const;

    /* --------------------------- OpenCL ----------------------------- */

#ifdef _ENABLE_OPENCL

    opencl::Model<self_type> opencl() const
    { return opencl::Model<self_type>(*this); }

    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

private:

    /* -------------- Constructors -------------- */

    BinaryMulExpr(LHS const &lhs, RHS const &rhs);

    BinaryMulExpr(BinaryMulExpr<I1, I2, LHS, RHS> const &) = default;

    /* ---------------- Utility ---------------- */

    // Accepts the slice sequences and forwards to `mul()`. Ideally, this
    // would be a lambda in `slice()` but C++11 has no support for templated lambdas
    template<
            size_t... Slices1,
            size_t... Slices2,
            typename... Args,
            typename = typename std::enable_if<self_type::rank() != sizeof...(Args)>::type
    >
    auto pSliceSequences(
            meta::Sequence<Slices1...>,
            meta::Sequence<Slices2...>,
            Args... args
    ) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().slice(args...))>::type;

    template<
            size_t... Slices1,
            size_t... Slices2,
            typename... Args,
            typename = typename std::enable_if<self_type::rank() == sizeof...(Args)>::type
    >
    value_type pSliceSequences(
            meta::Sequence<Slices1...>,
            meta::Sequence<Slices2...>,
            Args... args
    ) const;

    template<
            size_t... Slices1,
            size_t... Slices2,
            size_t M,
            typename = typename std::enable_if<self_type::rank() != M>::type
    >
    auto pSliceSequences(
            meta::Sequence<Slices1...>,
            meta::Sequence<Slices2...>,
            Indices<M> const &indices
    ) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().slice(indices))>::type;

    template<
            size_t... Slices1,
            size_t... Slices2,
            size_t M,
            typename = typename std::enable_if<self_type::rank() == M>::type
    >
    value_type pSliceSequences(
            meta::Sequence<Slices1...>,
            meta::Sequence<Slices2...>,
            Indices<M> const &indices
    ) const;

    /* ------------------ Data ------------------ */

    // cache shape that `shape()` can return a const reference
    Shape<self_type::rank()> shape_;
    LHS const &lhs_;
    RHS const &rhs_;
};

template<size_t I1, size_t I2, typename LHS, typename RHS>
BinaryMulExpr<I1, I2, LHS, RHS>::BinaryMulExpr(LHS const &lhs, RHS const &rhs)
        : lhs_(lhs), rhs_(rhs)
{
    static_assert(LHS::rank(), PANIC_ASSERTION);
    static_assert(RHS::rank(), PANIC_ASSERTION);
    static_assert(I1 < LHS::rank(), RANK_OUT_OF_BOUNDS);
    static_assert(I2 < RHS::rank(), RANK_OUT_OF_BOUNDS);
    assert(lhs.dimension(I1) == rhs.dimension(I2) && DIMENSION_MISMATCH);

    // shape <- lhs.shape[:~I1] ++ rhs.shape[:~I2]
    details::FillExceptForIndex<I1, LHS::rank()>(lhs.shape().data(), shape_.data());
    details::FillExceptForIndex<I2, RHS::rank()>(
            rhs.shape().data(),
            shape_.data() + LHS::rank() - 1
    );
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
size_t BinaryMulExpr<I1, I2, LHS, RHS>::dimension(size_t index) const noexcept
{
    assert(index < self_type::rank() && INDEX_OUT_OF_BOUNDS);
    return shape_[index];
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<typename... Args, typename>
auto BinaryMulExpr<I1, I2, LHS, RHS>::operator()(Args... args) const
-> typename std::remove_reference<decltype(std::declval<return_type const>()(args...))>::type
{
    static_assert(rank() >= sizeof...(Args), RANK_OUT_OF_BOUNDS);
    constexpr size_t left = meta::Min(LHS::rank() - 1, sizeof...(args));
    constexpr size_t right = meta::NonZeroDifference(sizeof...(args), LHS::rank() - 1);
    meta::FillArgs<left, right + left> seperate_args(args...);
    return lhs_.template slice<I1>(Indices<left>(seperate_args.array1.data())) *
           rhs_.template slice<I2>(Indices<right>(seperate_args.array2.data()));
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<typename... Args, typename>
typename BinaryMulExpr<I1, I2, LHS, RHS>::value_type
BinaryMulExpr<I1, I2, LHS, RHS>::operator()(Args... args) const
{
    constexpr size_t left = meta::Min(LHS::rank() - 1, sizeof...(args));
    constexpr size_t right = meta::NonZeroDifference(sizeof...(args), LHS::rank() - 1);
    meta::FillArgs<left, right + left> seperate_args(args...);

    auto lhs = lhs_.template slice<I1>(Indices<left>(seperate_args.array1.data()));
    auto rhs = rhs_.template slice<I2>(Indices<right>(seperate_args.array2.data()));
    auto mul_vals = [](
            value_type &accum,
            value_type const &x,
            typename RHS::value_type const &y
    ) {
        return accum + x * y;
    };

    return reduce(value_type{}, mul_vals, lhs, rhs);
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<typename... Args>
auto BinaryMulExpr<I1, I2, LHS, RHS>::at(Args... args) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().at(args...))>::type
{
    static_assert(rank() >= sizeof...(Args), RANK_OUT_OF_BOUNDS);
    constexpr size_t left = meta::Min(LHS::rank() - 1, sizeof...(args));
    constexpr size_t right = meta::NonZeroDifference(sizeof...(args), LHS::rank() - 1);

    meta::FillArgs<left, right + left> seperate(args...);
    return lhs_.template slice<I1>(Indices<left>(seperate.array1.data())) *
           rhs_.template slice<I2>(Indices<right>(seperate.array2.data()));
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<size_t M, typename>
auto BinaryMulExpr<I1, I2, LHS, RHS>::operator[](Indices<M> const &indices) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>()[indices])>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    constexpr size_t left = meta::Min(LHS::rank() - 1, M);
    constexpr size_t right = meta::NonZeroDifference(M, LHS::rank() - 1);

    auto array1 = std::array<size_t, left>{};
    auto array2 = std::array<size_t, right>{};
    for (size_t i = 0; i < left; ++i) array1[i] = indices[i];
    for (size_t i = 0; i < right; ++i) array2[i] = indices[i + left];

    return lhs_.template slice<I1>(Indices<left>(array1.data())) *
           rhs_.template slice<I2>(Indices<right>(array2.data()));
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<size_t M, typename>
typename BinaryMulExpr<I1, I2, LHS, RHS>::value_type
BinaryMulExpr<I1, I2, LHS, RHS>::operator[](Indices<M> const &indices) const
{
    constexpr size_t left = meta::Min(LHS::rank() - 1, M);
    constexpr size_t right = meta::NonZeroDifference(M, LHS::rank() - 1);

    auto array1 = std::array<size_t, left>{};
    auto array2 = std::array<size_t, right>{};
    for (size_t i = 0; i < left; ++i) array1[i] = indices[i];
    for (size_t i = 0; i < right; ++i) array2[i] = indices[i + left];

    auto lhs = lhs_.template slice<I1>(Indices<left>(array1.data()));
    auto rhs = rhs_.template slice<I2>(Indices<right>(array2.data()));
    auto mul_vals =
            [](
                    value_type &accum,
                    value_type const &x,
                    typename RHS::value_type const &y
            ) {
                return accum + x * y;
            };

    return reduce(value_type{}, mul_vals, lhs, rhs);
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<size_t... Slices, typename... Args>
auto BinaryMulExpr<I1, I2, LHS, RHS>::slice(Args... args) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().slice(args...))>::type
{
    using namespace meta; // WARNING -- using namespace
    static_assert(IsIncreasing<Slices...>::value, SLICE_INDICES_DESCENDING);

    constexpr size_t lhs_left_count = CountLTMax<I1, Slices...>::value;
    constexpr size_t lhs_right_count = CountInBetween<I1, LHS::rank() - 1, Slices...>::value;
    constexpr size_t lhs_count = lhs_left_count + lhs_right_count;

    auto lhs_left_sequence = typename MakeLeftSequence<lhs_left_count, Slices...>::sequence{};

    auto lhs_right_sequence = typename LeftSequence<lhs_right_count,
            typename MakeRightSequence<lhs_left_count, Slices...>::sequence>::sequence{};

    auto lhs_sequence = typename Concatenate<
            typename AppendEnd<I1,
                    decltype(lhs_left_sequence)>::sequence,
            typename SequenceTransformer<decltype(lhs_right_sequence),
                    SequencePositiveOffset, 1>::sequence
    >::sequence{};

    auto rhs_raw_sequence = typename SequenceTransformer<
            typename MakeRightSequence<lhs_count, Slices...>::sequence,
            SequenceNegativeOffset, Max(LHS::rank(), 2) - 2
    >::sequence{};

    constexpr size_t rhs_left_count = CountLTMaxSequence<I2 + 1,
            decltype(rhs_raw_sequence)>::value;

    auto rhs_left_sequence = typename LeftSequence<rhs_left_count,
            decltype(rhs_raw_sequence)>::sequence{};

    auto rhs_right_sequence = typename RightSequence<rhs_left_count,
            decltype(rhs_raw_sequence)>::sequence{};

    auto rhs_sequence = typename Concatenate<typename AppendEnd<I2,
            typename SequenceTransformer<decltype(rhs_left_sequence),
                    SequenceNegativeOffset, 1>::sequence>::sequence, decltype(
    rhs_right_sequence)>::sequence{};

    static_assert(
            decltype(lhs_sequence)::size + decltype(rhs_sequence)::size ==
            sizeof...(Slices) + 2, PANIC_ASSERTION
    );

    return pSliceSequences(lhs_sequence, rhs_sequence, args...);
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<size_t... Slices, size_t M>
auto BinaryMulExpr<I1, I2, LHS, RHS>::slice(Indices<M> const &indices) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().slice(indices))>::type
{
    // WARNING -- using namespace
    using namespace meta;
    static_assert(IsIncreasing<Slices...>::value, SLICE_INDICES_DESCENDING);

    constexpr size_t lhs_left_count = CountLTMax<I1, Slices...>::value;
    constexpr size_t lhs_right_count = CountInBetween<I1, LHS::rank() - 1, Slices...>::value;
    constexpr size_t lhs_count = lhs_left_count + lhs_right_count;

    auto lhs_left_sequence = typename MakeLeftSequence<lhs_left_count, Slices...>::sequence{};
    auto lhs_right_sequence = typename LeftSequence<
            lhs_right_count, typename MakeRightSequence<
                    lhs_left_count, Slices...>::sequence>::sequence{};

    auto lhs_sequence = typename Concatenate<typename AppendEnd<I1, decltype(
    lhs_left_sequence)>::sequence, typename SequenceTransformer<decltype(
    lhs_right_sequence), SequencePositiveOffset, 1>::sequence>::sequence{};

    auto rhs_raw_sequence = typename SequenceTransformer<typename MakeRightSequence<
            lhs_count, Slices...>::sequence, SequenceNegativeOffset, Max(
            LHS::rank(), 2
    ) - 2>::sequence{};

    constexpr size_t rhs_left_count = CountLTMaxSequence<I2, decltype(rhs_raw_sequence)>::value;

    auto rhs_left_sequence = typename LeftSequence<rhs_left_count, decltype(
    rhs_raw_sequence)>::sequence{};

    auto rhs_right_sequence = typename RightSequence<rhs_left_count, decltype(
    rhs_raw_sequence)>::sequence{};

    auto rhs_sequence = typename Concatenate<typename AppendEnd<I2,
            typename SequenceTransformer<decltype(rhs_left_sequence),
                    SequenceNegativeOffset, 1>::sequence>::sequence, decltype(
    rhs_right_sequence)>::sequence{};

    static_assert(
            decltype(lhs_sequence)::size + decltype(rhs_sequence)::size
            == sizeof...(Slices) + 2, PANIC_ASSERTION
    );

    return pSliceSequences(lhs_sequence, rhs_sequence, indices);
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
std::string BinaryMulExpr<I1, I2, LHS, RHS>::str() const
{
    return "<" + std::to_string(I1) + ", " + std::to_string(I2)
           + ">(* " + lhs_.str() + " " + rhs_.str() + ")";
}

template<typename LHS, typename RHS>
BinaryMulExpr<LHS::rank() - 1, 0, LHS, RHS> operator*(
        Expression<LHS> const &lhs,
        Expression<RHS> const &rhs
)
{
    return BinaryMulExpr<LHS::rank() - 1, 0, LHS, RHS>(lhs.self(), rhs.self());
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
BinaryMulExpr<I1, I2, LHS, RHS> _mul_(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    return BinaryMulExpr<I1, I2, LHS, RHS>(lhs.self(), rhs.self());
}

/* ------------------------------ Utility ------------------------------ */

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<size_t... Slices1, size_t... Slices2, typename... Args, typename>
auto BinaryMulExpr<I1, I2, LHS, RHS>::pSliceSequences(
        meta::Sequence<Slices1...>,
        meta::Sequence<Slices2...>,
        Args... args
) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().slice(args...))>::type
{
    constexpr size_t left = meta::Min(LHS::rank() - sizeof...(Slices1), sizeof...(args));
    constexpr size_t right = meta::NonZeroDifference(
            sizeof...(args),
            LHS::rank() - sizeof...(Slices1)
    );
    meta::FillArgs<left, right + left> seperate(args...);

    return lhs_.template slice<Slices1...>(Indices<left>(seperate.array1.data())) *
           rhs_.template slice<Slices2...>(Indices<right>(seperate.array2.data()));
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<size_t... Slices1, size_t... Slices2, typename... Args, typename>
typename BinaryMulExpr<I1, I2, LHS, RHS>::value_type
BinaryMulExpr<I1, I2, LHS, RHS>::pSliceSequences(
        meta::Sequence<Slices1...>,
        meta::Sequence<Slices2...>, Args... args
) const
{
    constexpr size_t left = meta::Min(LHS::rank() - sizeof...(Slices1), sizeof...(args));
    constexpr size_t right = meta::NonZeroDifference(
            sizeof...(args), LHS::rank() - sizeof...(Slices1));

    meta::FillArgs<left, right + left> seperate(args...);

    auto lhs = lhs_.template slice<Slices1...>(Indices<left>(seperate.array1.data()));
    auto rhs = rhs_.template slice<Slices2...>(Indices<right>(seperate.array2.data()));

    auto mul_vals =
            [](
                    value_type &accum,
                    value_type const &x,
                    typename RHS::value_type const &y
            ) {
                return accum + x * y;
            };

    return reduce(value_type{}, mul_vals, lhs, rhs);
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<size_t... Slices1, size_t... Slices2, size_t M, typename>
auto BinaryMulExpr<I1, I2, LHS, RHS>::pSliceSequences(
        meta::Sequence<Slices1...>,
        meta::Sequence<Slices2...>,
        Indices<M> const &indices
) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().slice(indices))>::type
{
    constexpr size_t left = meta::Min(LHS::rank() - sizeof...(Slices1), M);
    constexpr size_t right = meta::NonZeroDifference(M, LHS::rank() - sizeof...(Slices1));

    auto array1 = std::array<size_t, left>{};
    auto array2 = std::array<size_t, right>{};

    for (size_t i = 0; i < left; ++i)
        array1[i] = indices[i];
    for (size_t i = 0; i < right; ++i)
        array2[i] = indices[i + left];

    return lhs_.template slice<Slices1...>(
            Indices<left>(array1.data())) * rhs_.template slice<Slices2...>(
            Indices<right>(array2.data()));
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
template<size_t... Slices1, size_t... Slices2, size_t M, typename>
typename BinaryMulExpr<I1, I2, LHS, RHS>::value_type
BinaryMulExpr<I1, I2, LHS, RHS>::pSliceSequences(
        meta::Sequence<Slices1...>,
        meta::Sequence<Slices2...>,
        Indices<M> const &indices
) const
{
    constexpr size_t left = meta::Min(LHS::rank() - sizeof...(Slices1), M);
    constexpr size_t right = meta::NonZeroDifference(M, LHS::rank() - sizeof...(Slices1));
    auto array1 = std::array<size_t, left>{};
    auto array2 = std::array<size_t, right>{};
    for (size_t i = 0; i < left; ++i) array1[i] = indices[i];
    for (size_t i = 0; i < right; ++i) array2[i] = indices[i + left];

    auto lhs = lhs_.template slice<Slices1...>(Indices<left>(array1.data()));
    auto rhs = rhs_.template slice<Slices2...>(Indices<right>(array2).data());
    auto mul_vals =
            [](
                    value_type &accum,
                    value_type const &x,
                    typename RHS::value_type const &y
            ) {
                return accum + x * y;
            };

    return reduce(value_type{}, mul_vals, lhs, rhs);
}

/* ----------------------------- OpenCL ----------------------------- */

#ifdef _ENABLE_OPENCL

template<size_t I1, size_t I2, typename LHS, typename RHS>
std::string BinaryMulExpr<I1, I2, LHS, RHS>::opencl_kernel_code(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    // WARNING using namespace
    using namespace opencl::details;
    using lhs_t = typename LHS::value_type;
    using rhs_t = typename RHS::value_type;

    // Create the multiplication kernel lhs and rhs buffers
    cl::Buffer lhs_buffer = opencl::details::create_buffer(lhs_);
    cl::Buffer rhs_buffer = opencl::details::create_buffer(rhs_);

    buffers.push_back(
            create_multiplication_kernel<I1, I2, lhs_t, rhs_t>(
                    cqueue, lhs_buffer, rhs_buffer, lhs_.shape(), rhs_.shape()));

    // add the new scalar to the argument list and the expression
    size_t arg_index = buffers.size() - 1;

    // Add pointer to the the buffer to the element list
    arg_list += std::string(cGlobalIdentifier) + " " + (OpenCLType<value_type>::value) + " "
                + cConstIdentifier + " " + cPointerIdentifier + cVariablePrefix
                + std::to_string(arg_index) + ", ";

    // return the name of the element
    return cVariablePrefix + std::to_string(arg_index) + "[" + cGlobalIdName + "]";
}

template<size_t I1, size_t I2, typename LHS, typename RHS>
cl::Buffer BinaryMulExpr<I1, I2, LHS, RHS>::opencl_evaluate_kernel(
        cl::CommandQueue &,
        std::vector<cl::Buffer> &buffers,
        std::string &,
        std::string &
) const noexcept
{
    return buffers.back();
}

#endif // _ENABLE_OPENCL

/* --------------------------------------------------------------------- */

template<typename LHS, typename RHS>
class BinaryHadExpr : public Expression<BinaryHadExpr<LHS, RHS>> {
/*@BinaryHadExpr*/
public:

    /* ---------------- typedefs --------------- */

    constexpr static size_t rank()
    { return LHS::rank(); }

    typedef typename LHS::value_type value_type;
    template<typename X> using container_type = typename LHS::template container_type<X>;
    typedef BinaryHadExpr self_type;
    typedef
    Tensor<value_type, self_type::rank(), container_type> return_type;

    /* ---------------- Friends ---------------- */

    template<typename LHS_, typename RHS_>
    friend BinaryHadExpr<LHS_, RHS_>
    operator%(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

    template<typename LHS_, typename RHS_>
    friend BinaryHadExpr<LHS_, RHS_>
    _had_(Expression<LHS_> const &lhs, Expression<RHS_> const &rhs);

    /** Produces "(% `lhs_` `rhs_`)" */
    std::string str() const;

    template<typename LHS_, typename RHS_> friend
    class BinaryAddExpr;

    template<typename LHS_, typename RHS_> friend
    class BinarySubExpr;

    template<size_t I1_, size_t I2_, typename LHS_, typename RHS_> friend
    class BinaryMulExpr;

    template<typename LHS_, typename RHS_> friend
    class BinaryHadExpr;

    template<typename Function_, typename... Exprs_> friend
    class MapExpr;

    template<typename U, typename Function_, typename... Exprs_> friend
    class ReduceExpr;

    template<typename RHS_> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ---------------- Getters ----------------- */

    size_t dimension(size_t index) const
    { return lhs_.dimension(index); }

    Shape<LHS::rank()> const &shape() const
    { return lhs_.shape(); }

    /** Evaluate and return the resulting Tensor */
    return_type eval() const noexcept
    { return (*this)(); }

    template<
            typename... Args,
            typename = typename std::enable_if<sizeof...(Args) != self_type::rank()>::type
    >
    auto operator()(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type;

    template<
            typename... Args, typename =
            typename std::enable_if<sizeof...(Args) == self_type::rank()>::type
    >
    value_type operator()(Args... args) const;

    template<
            typename... Args,
            typename = typename std::enable_if<sizeof...(Args) != self_type::rank()>::type
    >
    auto at(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type;

    template<
            typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == self_type::rank()>::type
    >
    Tensor<value_type, 0, container_type> at(Args... args) const;

    template<
            size_t M,
            typename = typename std::enable_if<M != self_type::rank()>::type
    >
    auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type;

    template<
            size_t M,
            typename = typename std::enable_if<M == self_type::rank()>::type
    >
    value_type operator[](Indices<M> const &indices) const;

    template<
            size_t... Slices,
            typename... Args,
            typename = typename std::enable_if<sizeof...(Args) != self_type::rank()>::type
    >
    auto slice(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(args...))>::type;

    template<
            size_t... Slices,
            typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == self_type::rank()>::type
    >
    value_type slice(Args... args) const;

    template<
            size_t... Slices,
            size_t M,
            typename = typename std::enable_if<M != self_type::rank()>::type
    >
    auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<LHS const>().slice(indices))>::type;

    template<
            size_t... Slices,
            size_t M,
            typename = typename std::enable_if<M == self_type::rank()>::type
    >
    value_type slice(Indices<M> const &indices) const;

    /* ---------------- OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

    opencl::Model<self_type> opencl() const
    { return opencl::Model<self_type>(*this); }

    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

private:

    /* -------------- Constructors -------------- */

    BinaryHadExpr(LHS const &lhs, RHS const &rhs);

    BinaryHadExpr(BinaryHadExpr<LHS, RHS> const &) = default;

    /* ------------------ Data ------------------ */

    LHS const &lhs_;
    RHS const &rhs_;

private:

};

/* ---------------- Friends ---------------- */

template<typename LHS, typename RHS>
BinaryHadExpr<LHS, RHS> operator%(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    return BinaryHadExpr<LHS, RHS>(lhs.self(), rhs.self());
}

template<typename LHS, typename RHS>
BinaryHadExpr<LHS, RHS> _had_(Expression<LHS> const &lhs, Expression<RHS> const &rhs)
{
    return BinaryHadExpr<LHS, RHS>(lhs.self(), rhs.self());
}

/* ---------------- Getters ---------------- */

template<typename LHS, typename RHS>
template<typename... Args, typename>
auto BinaryHadExpr<LHS, RHS>::operator()(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>()(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return lhs_(args...) % rhs_(args...);
}

template<typename LHS, typename RHS>
template<typename... Args, typename>
typename BinaryHadExpr<LHS, RHS>::value_type
BinaryHadExpr<LHS, RHS>::operator()(Args... args) const
{
    return lhs_(args...) * rhs_(args...);
}

template<typename LHS, typename RHS>
template<typename... Args, typename>
auto BinaryHadExpr<LHS, RHS>::at(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().at(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return lhs_(args...) % rhs_(args...);
}

template<typename LHS, typename RHS>
template<typename... Args, typename>
Tensor<
        typename BinaryHadExpr<LHS, RHS>::value_type,
        0,
        BinaryHadExpr<LHS, RHS>::template container_type
>
BinaryHadExpr<LHS, RHS>::at(Args... args) const
{
    return lhs_(args...) * rhs_(args...);
}

template<typename LHS, typename RHS>
template<size_t M, typename>
auto BinaryHadExpr<LHS, RHS>::operator[](Indices<M> const &indices) const
-> typename std::remove_reference<decltype(std::declval<LHS const>()[indices])>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    return lhs_[indices] % rhs_[indices];
}

template<typename LHS, typename RHS>
template<size_t M, typename>
typename BinaryHadExpr<LHS, RHS>::value_type
BinaryHadExpr<LHS, RHS>::operator[](Indices<M> const &indices) const
{
    return lhs_[indices] * rhs_[indices];
}

template<typename LHS, typename RHS>
template<size_t... Slices, typename... Args, typename>
auto BinaryHadExpr<LHS, RHS>::slice(Args... args) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().slice(args...))>::type
{
    static_assert(rank() >= sizeof...(Slices) + sizeof...(args), RANK_OUT_OF_BOUNDS);
    return lhs_.template slice<Slices...>(args...) % rhs_.template slice<Slices...>(args...);
}

template<typename LHS, typename RHS>
template<size_t... Slices, typename... Args, typename>
typename BinaryHadExpr<LHS, RHS>::value_type
BinaryHadExpr<LHS, RHS>::slice(Args... args) const
{
    static_assert(!sizeof...(Slices), RANK_OUT_OF_BOUNDS);
    return lhs(args...) * rhs(args...);
}

template<typename LHS, typename RHS>
template<size_t... Slices, size_t M, typename>
auto BinaryHadExpr<LHS, RHS>::slice(Indices<M> const &indices) const
-> typename std::remove_reference<decltype(std::declval<LHS const>().slice(indices))>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    return lhs_.template slice<Slices...>(indices) % rhs_.template slice<Slices...>(indices);
}

template<typename LHS, typename RHS>
template<size_t... Slices, size_t M, typename>
typename BinaryHadExpr<LHS, RHS>::value_type
BinaryHadExpr<LHS, RHS>::slice(Indices<M> const &indices) const
{
    static_assert(!sizeof...(Slices), RANK_OUT_OF_BOUNDS);
    return lhs_[indices] * rhs_[indices];
}

/* -------------- Constructors -------------- */

template<typename LHS, typename RHS>
BinaryHadExpr<LHS, RHS>::BinaryHadExpr(LHS const &lhs, RHS const &rhs)
        : lhs_(lhs), rhs_(rhs)
{}

/* ----------------- Print ----------------- */

template<typename LHS, typename RHS>
std::string BinaryHadExpr<LHS, RHS>::str() const
{
    return "(% " + lhs_.str() + " " + rhs_.str() + ")";
}

/* ----------------- OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

template<typename LHS, typename RHS>
std::string BinaryHadExpr<LHS, RHS>::opencl_kernel_code(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    std::string expr = std::string("(") + lhs_.opencl_kernel_code(cqueue, buffers, arg_list)
                       + " * " + rhs_.opencl_kernel_code(cqueue, buffers, arg_list) + ")";
    return expr;
}

template<typename LHS, typename RHS>
cl::Buffer BinaryHadExpr<LHS, RHS>::opencl_evaluate_kernel(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list,
        std::string &expr
) const noexcept
{
    return opencl::details::evaluate_basic_kernel<value_type>(
            cqueue,
            buffers,
            arg_list,
            expr,
            lhs_.shape().index_product()
    );
}

#endif // _ENABLE_OPENCL

template<typename Function, typename... Exprs>
class MapExpr : public Expression<MapExpr<Function, Exprs...>> {
/*@MapExpr<Function, Exprs...>*/
public:

    /* ----------------------------- typedefs ------------------------------ */

    constexpr static size_t rank()
    { return first_t::rank(); }

    typedef typename meta::RemoveCVRef<
            typename std::result_of<Function(
                    typename Exprs::value_type...
            )>::type>::type value_type;
    typedef typename FirstExpression<Exprs...>::type first_t;
    template<typename X>
    using container_type = typename first_t::template container_type<X>;
    typedef MapExpr self_type;
    typedef
    Tensor<value_type, self_type::rank(), container_type> return_type;

    /* ------------------------------ Friend ------------------------------ */

    template<typename Function_, typename... Exprs_>
    friend MapExpr<Function_, Exprs_...> _map_(
            Function_ &&fn,
            Expression<Exprs_> const &... exprs
    );

    /** Produces "(<map> `expr_...`)" */
    std::string str() const;

    template<typename LHS_, typename RHS_> friend
    class BinaryAddExpr;

    template<typename LHS_, typename RHS_> friend
    class BinarySubExpr;

    template<size_t I1_, size_t I2_, typename LHS_, typename RHS_> friend
    class BinaryMulExpr;

    template<typename LHS_, typename RHS_> friend
    class BinaryHadExpr;

    template<typename U, typename Function_, typename... Exprs_> friend
    class ReduceExpr;

    template<typename Function_, typename... Exprs_> friend
    class MapExpr;

    template<typename RHS_> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ----------------------------- Getters ------------------------------ */

    size_t dimension(size_t index) const
    { return std::get<0>(exprs_).dimension(index); }

    Shape<self_type::rank()> const &shape() const
    { return std::get<0>(exprs_).shape(); }

    /** Evaluate and return the resulting Tensor */
    return_type eval() const
    { return (*this)(); }

    template<typename... Args>
    auto operator()(Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>()(args...))>::type;

    template<typename... Args>
    auto at(Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().at(args...))>::type;

    template<size_t M>
    auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>()[indices])>::type;

    template<size_t... Slices, typename... Args>
    auto slice(Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().slice(args...))>::type;

    template<size_t... Slices, size_t M>
    auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().slice(indices))>::type;

    /* ---------------- OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

    opencl::Model<self_type> opencl() const
    { return opencl::Model<self_type>(*this); }

    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

private:

    /* ------------------------- Constructors ------------------------- */

    explicit MapExpr(Function &&fn, Exprs const &... exprs);

    MapExpr(MapExpr<Function, Exprs...> const &) = default;

    /* --------------------------- Utility ------------------------------ */

    template<
            size_t... TupleIndices, typename... Args,
            typename = typename std::enable_if<self_type::rank() != sizeof...(Args)>::type>
    auto pMapExpansion(meta::Sequence<TupleIndices...>, Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>()(args...))>::type;

    template<
            size_t... TupleIndices, typename... Args,
            typename = typename std::enable_if<self_type::rank() == sizeof...(Args)>::type>
    value_type pMapExpansion(meta::Sequence<TupleIndices...>, Args... args) const;

    template<
            size_t... TupleIndices, size_t M,
            typename = typename std::enable_if<self_type::rank() != M>::type>
    auto pMapExpansion(meta::Sequence<TupleIndices...>, Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>()[indices])>::type;

    template<
            size_t... TupleIndices, size_t M,
            typename = typename std::enable_if<self_type::rank() == M>::type>
    value_type pMapExpansion(meta::Sequence<TupleIndices...>, Indices<M> const &indices) const;

    template<
            size_t... Slices, size_t... TupleIndices, typename... Args,
            typename = typename std::enable_if<self_type::rank() != sizeof...(Args)>::type>
    auto pMapSliceExpansion(meta::Sequence<TupleIndices...>, Args... args) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().slice(args...))>::type;

    template<
            size_t... Slices, size_t... TupleIndices, typename... Args,
            typename = typename std::enable_if<self_type::rank() == sizeof...(Args)>::type>
    value_type pMapSliceExpansion(meta::Sequence<TupleIndices...>, Args... args) const;

    template<
            size_t... Slices,
            size_t... TupleIndices,
            size_t M,
            typename = typename std::enable_if<self_type::rank() != M>::type
    >
    auto pMapSliceExpansion(meta::Sequence<TupleIndices...>, Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(
    std::declval<return_type const>().slice(indices))>::type;

    template<
            size_t... Slices,
            size_t... TupleIndices,
            size_t M,
            typename = typename std::enable_if<self_type::rank() == M>::type
    >
    value_type
    pMapSliceExpansion(meta::Sequence<TupleIndices...>, Indices<M> const &indices) const;

    template<size_t... TupleIndices>
    std::string pStringExpansion(meta::Sequence<TupleIndices...>) const;

    /* ---------------------------- OpenCL ---------------------------- */

#ifdef _ENABLE_OPENCL

    template<size_t... TupleIndices>
    std::string popencl_kernel_codeTupleExpansion(
            meta::Sequence<TupleIndices...>,
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

#endif

    /* ---------------------------- Data ------------------------------ */

    std::tuple<Exprs const &...> exprs_;
    Function &fn_;
};

template<typename Function, typename... Exprs>
template<typename... Args>
auto MapExpr<Function, Exprs...>::operator()(Args... args) const
-> typename std::remove_reference<decltype(std::declval<return_type const>()(args...))>::type
{
    static_assert(self_type::rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return pMapExpansion(
            typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{}, args...
    );
}

template<typename Function, typename... Exprs>
template<typename... Args>
auto MapExpr<Function, Exprs...>::at(Args... args) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().at(args...))>::type
{
    static_assert(self_type::rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return pMapExpansion(
            typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{}, args...
    );
}

template<typename Function, typename... Exprs>
template<size_t M>
auto MapExpr<Function, Exprs...>::operator[](Indices<M> const &indices) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>()[indices])>::type
{
    static_assert(self_type::rank() >= M, RANK_OUT_OF_BOUNDS);
    return pMapExpansion(
            typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{}, indices
    );
}

template<typename Function, typename... Exprs>
template<size_t... Slices, typename... Args>
auto MapExpr<Function, Exprs...>::slice(Args... args) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().slice(args...))>::type
{
    static_assert(
            self_type::rank() >= sizeof...(Slices) + sizeof...(Args),
            SLICES_OUT_OF_BOUNDS
    );

    return pMapSliceExpansion<Slices...>(
            typename meta::MakeIndexSequence<0,
                    sizeof...(Exprs)>::sequence{}, args...
    );
}

template<typename Function, typename... Exprs>
template<size_t... Slices, size_t M>
auto MapExpr<Function, Exprs...>::slice(Indices<M> const &indices) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().slice(indices))>::type
{
    static_assert(self_type::rank() >= M + sizeof...(Slices), SLICES_OUT_OF_BOUNDS);
    return pMapSliceExpansion<Slices...>(
            typename meta::MakeIndexSequence<0,
                    sizeof...(Exprs)>::sequence{}, indices
    );
}

template<typename Function, typename... Exprs>
MapExpr<Function, Exprs...> _map_(Function &&fn, Expression<Exprs> const &... exprs)
{
    return MapExpr<Function, Exprs...>(std::forward<Function>(fn), exprs.self()...);
}

template<typename Function, typename... Exprs>
MapExpr<Function, Exprs...>::MapExpr(Function &&fn, Exprs const &... exprs)
        : exprs_(std::forward_as_tuple(exprs...)), fn_(fn)
{}

template<typename Function, typename... Exprs>
std::string MapExpr<Function, Exprs...>::str() const
{
    return pStringExpansion(typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{});
}

/* --------------------------- Utility ------------------------------ */

template<typename Function, typename... Exprs>
template<size_t... TupleIndices, typename... Args, typename>
auto MapExpr<Function, Exprs...>::pMapExpansion(
        meta::Sequence<TupleIndices...>,
        Args... args
) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>()(args...))>::type
{
    static_assert(self_type::rank() != sizeof...(args), PANIC_ASSERTION);
    return _map_(fn_, std::get<TupleIndices>(exprs_)(args...)...);
}

template<typename Function, typename... Exprs>
template<size_t... TupleIndices, typename... Args, typename>
auto MapExpr<Function, Exprs...>::pMapExpansion(
        meta::Sequence<TupleIndices...>, Args... args
) const -> value_type
{
    static_assert(self_type::rank() == sizeof...(args), PANIC_ASSERTION);
    return fn_(std::get<TupleIndices>(exprs_)(args...)...);
}

template<typename Function, typename... Exprs>
template<size_t... TupleIndices, size_t M, typename>
auto MapExpr<Function, Exprs...>::pMapExpansion(
        meta::Sequence<TupleIndices...>,
        Indices<M> const &indices
) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>()[indices])>::type
{
    static_assert(self_type::rank() != M, PANIC_ASSERTION);
    return _map_(fn_, std::get<TupleIndices>(exprs_)[indices]...);
}

template<typename Function, typename... Exprs>
template<size_t... TupleIndices, size_t M, typename>
typename MapExpr<Function, Exprs...>::value_type
MapExpr<Function, Exprs...>::pMapExpansion(
        meta::Sequence<TupleIndices...>,
        Indices<M> const &indices
) const
{
    static_assert(self_type::rank() == M, PANIC_ASSERTION);
    return fn_(std::get<TupleIndices>(exprs_)[indices]...);
}

template<typename Function, typename... Exprs>
template<size_t... Slices, size_t... TupleIndices, typename... Args, typename>
auto MapExpr<Function, Exprs...>::pMapSliceExpansion(
        meta::Sequence<TupleIndices...>,
        Args... args
) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().slice(args...))>::type
{
    static_assert(self_type::rank() >= sizeof...(Args) + sizeof...(Slices), PANIC_ASSERTION);
    static_assert(self_type::rank() != sizeof...(Args), PANIC_ASSERTION);
    return _map_(fn_, std::get<TupleIndices>(exprs_).template slice<Slices...>(args...)...);
}

template<typename Function, typename... Exprs>
template<size_t... Slices, size_t... TupleIndices, typename... Args, typename>
typename MapExpr<Function, Exprs...>::value_type
MapExpr<Function, Exprs...>::pMapSliceExpansion(
        meta::Sequence<TupleIndices...>, Args... args
) const
{
    static_assert(self_type::rank() == sizeof...(Args), PANIC_ASSERTION);
    static_assert(sizeof...(Slices) == 0, PANIC_ASSERTION);
    return fn_(std::get<TupleIndices>(exprs_)(args...)...);
}

template<typename Function, typename... Exprs>
template<size_t... Slices, size_t... TupleIndices, size_t M, typename>
auto MapExpr<Function, Exprs...>::pMapSliceExpansion(
        meta::Sequence<TupleIndices...>,
        Indices<M> const &indices
) const
-> typename std::remove_reference<decltype(
std::declval<return_type const>().slice(indices))>::type
{
    static_assert(self_type::rank() != M, PANIC_ASSERTION);
    static_assert(self_type::rank() >= M + sizeof...(Slices), PANIC_ASSERTION);
    return _map_(fn_, std::get<TupleIndices>(exprs_).template slice<Slices...>(indices)...);
}

template<typename Function, typename... Exprs>
template<size_t... Slices, size_t... TupleIndices, size_t M, typename>
typename MapExpr<Function, Exprs...>::value_type
MapExpr<Function, Exprs...>::pMapSliceExpansion(
        meta::Sequence<TupleIndices...>,
        Indices<M> const &indices
) const
{
    static_assert(self_type::rank() == M, PANIC_ASSERTION);
    static_assert(sizeof...(Slices) == 0, PANIC_ASSERTION);
    return fn_(std::get<TupleIndices>(exprs_)[indices]...);
}

template<typename Function, typename... Exprs>
template<size_t... TupleIndices>
std::string
MapExpr<Function, Exprs...>::pStringExpansion(meta::Sequence<TupleIndices...>) const
{
    auto s = std::string("(<map>");
    VARDIAC_MAP(s += " " + std::get<TupleIndices>(exprs_).str());
    s += ")";
    return s;
}

/* ---------------------------- OpenCL ---------------------------- */

#ifdef _ENABLE_OPENCL

template<typename Function, typename... Exprs>
std::string MapExpr<Function, Exprs...>::opencl_kernel_code(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    static_assert(!Function::arity() || sizeof...(Exprs) == Function::arity(), OPENCL_ARITY_ERROR);
    return popencl_kernel_codeTupleExpansion(
            typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{},
            cqueue,
            buffers,
            arg_list
    );
}

template<typename Function, typename... Exprs>
template<size_t... TupleIndices>
std::string MapExpr<Function, Exprs...>::popencl_kernel_codeTupleExpansion(
        meta::Sequence<TupleIndices...>,
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    return Function::opencl_map(
            std::get<TupleIndices>(exprs_).opencl_kernel_code(cqueue, buffers, arg_list)...
    );
}

template<typename Function, typename... Exprs>
cl::Buffer MapExpr<Function, Exprs...>::opencl_evaluate_kernel(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list,
        std::string &expr
) const noexcept
{
    return opencl::details::evaluate_basic_kernel<value_type>(
            cqueue,
            buffers,
            arg_list,
            expr,
            shape().index_product()
    );
}

#endif

template<typename T, typename Function, typename... Exprs>
class ReduceExpr : public Expression<ReduceExpr<T, Function, Exprs...>> {
/*@ReduceExpr<T, Function, Exprs...>>*/
public:

    /* ----------------------------- typedefs ------------------------------ */

    typedef typename FirstExpression<Exprs...>::type first_t;
    typedef T value_type;
    template<typename X>
    using container_type = typename first_t::template container_type<X>;

    constexpr static size_t rank()
    { return 0; }

    typedef ReduceExpr self_type;
    typedef Tensor<T, 0, container_type> return_type;

    /* ------------------------------ Friend ------------------------------ */

    template<typename U, typename Function_, typename... Exprs_>
    friend ReduceExpr<U, Function_, Exprs_...>
    _reduce_(U &&value, Function_ &&fn, Expression<Exprs_> const &... exprs);

    template<typename LHS_, typename RHS_> friend
    class BinaryAddExpr;

    template<typename LHS_, typename RHS_> friend
    class BinarySubExpr;

    template<size_t I1_, size_t I2_, typename LHS_, typename RHS_> friend
    class BinaryMulExpr;

    template<typename LHS_, typename RHS_> friend
    class BinaryHadExpr;

    template<typename U, typename Function_, typename... Exprs_> friend
    class ReduceExpr;

    template<typename Function_, typename... Exprs_> friend
    class MapExpr;

    template<typename RHS_> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ----------------------------- Getters ------------------------------ */

    Shape<0> shape() const
    { return Shape<0>(); }

    /** Evaluate and return the resulting Tensor */
    return_type eval() const noexcept
    { return (*this)(); }

    T operator()() const;

    auto at() const -> Tensor<T, 0, container_type>;

    T operator[](Indices<0> const &indices) const;

    template<size_t... Slices>
    T slice() const;

    template<size_t... Slices>
    T slice(Indices<0> const &indices) const;

    std::string str() const;

    /* --------------------------- OpenCL ----------------------------- */

#ifdef _ENABLE_OPENCL

    opencl::Model<self_type> opencl() const
    { return opencl::Model<self_type>(*this); }

    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

private:

    /* ------------------------- Constructors ------------------------- */

    ReduceExpr(T &&value, Function &&fn, Exprs const &... exprs);

    ReduceExpr(ReduceExpr<T, Function, Exprs...> const &) = default;

    /* --------------------------- Utility ------------------------------ */

    template<size_t... TupleIndices>
    T pReduceExpansion(meta::Sequence<TupleIndices...>) const;

    template<size_t... TupleIndices>
    std::string pStringExpansion(meta::Sequence<TupleIndices...>) const;

    /* ---------------------------- Data ------------------------------ */

    std::tuple<Exprs const &...> exprs_;
    Function &fn_;
    T return_value_;
};

template<typename U, typename Function_, typename... Exprs_>
ReduceExpr<U, Function_, Exprs_...>
_reduce_(U &&value, Function_ &&fn, Expression<Exprs_> const &... exprs)
{
    auto const &shape = details::GetShape(exprs...);
    VARDIAC_MAP(assert(shape == exprs.self().shape() && SHAPE_MISMATCH));
    static_assert(meta::AreExpressions<Exprs_...>::value, EXPECTING_EXPRESSION);
    return ReduceExpr<U, Function_, Exprs_...>(
            std::forward<U>(value), std::forward<Function_>(fn), exprs.self()...
    );
}

/* ----------------------------- Getters ------------------------------ */

template<typename T, typename Function, typename... Exprs>
T ReduceExpr<T, Function, Exprs...>::operator()() const
{
    return pReduceExpansion(typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{});
}

template<typename T, typename Function, typename... Exprs>
auto ReduceExpr<T, Function, Exprs...>::at() const
-> Tensor<T, 0, container_type>
{
    return pReduceExpansion(typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{});
}

template<typename T, typename Function, typename... Exprs>
T ReduceExpr<T, Function, Exprs...>::operator[](Indices<0> const &) const
{
    return pReduceExpansion(typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{});
}

template<typename T, typename Function, typename... Exprs>
template<size_t... Slices>
T ReduceExpr<T, Function, Exprs...>::slice() const
{
    static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
    return pReduceExpansion(typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{});
}

template<typename T, typename Function, typename... Exprs>
template<size_t... Slices>
T ReduceExpr<T, Function, Exprs...>::slice(Indices<0> const &) const
{
    static_assert(sizeof...(Slices) == 0, SLICES_OUT_OF_BOUNDS);
    return pReduceExpansion(typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{});
}

template<typename T, typename Function, typename... Exprs>
std::string ReduceExpr<T, Function, Exprs...>::str() const
{
    return pStringExpansion(typename meta::MakeIndexSequence<0, sizeof...(Exprs)>::sequence{});
}

/* ----------------------- Constructors --------------------- */

template<typename T, typename Function, typename... Exprs>
ReduceExpr<T, Function, Exprs...>::ReduceExpr(T &&value, Function &&fn, Exprs const &... exprs)
        : exprs_(std::forward_as_tuple(exprs...)), fn_(fn), return_value_(std::forward<T>(value))
{
    static_assert(sizeof...(Exprs), NO_TENSORS_PROVIDED);
}

/* ------------------------- Utility ----------------------- */

template<typename T, typename Function, typename... Exprs>
template<size_t... TupleIndices>
T ReduceExpr<T, Function, Exprs...>::pReduceExpansion(meta::Sequence<TupleIndices...>) const
{
    auto const &shape = details::GetShape<Exprs...>(std::get<TupleIndices>(exprs_)...);
    constexpr size_t M = std::remove_reference<decltype(shape)>::type::rank();
    auto indices = Indices<M>{};
    T return_value = return_value_;
    do {
        return_value = fn_(return_value, std::get<TupleIndices>(exprs_)[indices]...);
    } while (indices.increment(shape));
    return return_value;
}

template<typename T, typename Function, typename... Exprs>
template<size_t... TupleIndices>
std::string ReduceExpr<T, Function, Exprs...>::pStringExpansion(
        meta::Sequence<TupleIndices...>
) const
{
    auto s = std::string("(<reduce>");
    VARDIAC_MAP(s += " " + std::get<TupleIndices>(exprs_).str());
    s += ")";
    return s;
}

/* --------------------- OpenCL --------------------- */

#ifdef _ENABLE_OPENCL

template<typename T, typename Function, typename... Exprs>
std::string ReduceExpr<T, Function, Exprs...>::opencl_kernel_code(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    using namespace opencl::details; // WARNING -- using namespace

    // Create the reduction kernel, evaluate, and store the resulting buffer
    static_assert(sizeof...(Exprs) == 1, OPENCL_REDUCTION_SIZE_ERROR);
    auto const &shape = std::get<0>(exprs_).shape();
    cl::Buffer buffer = opencl::details::create_buffer(std::get<0>(exprs_));
    buffers.push_back(
            create_reduction_kernel<T, Function>(
                    cqueue, buffer, shape.index_product(), return_value_
            ));

    // add the new scalar to the argument list and the expression
    size_t arg_index = buffers.size() - 1;

    // There a pointer to the single element in the buffer to the element list
    arg_list += std::string(cGlobalIdentifier) + " " + (OpenCLType<T>::value) + " "
                + cConstIdentifier + " " + cPointerIdentifier + cVariablePrefix
                + std::to_string(arg_index) + ", ";

    // There is only one element in a reduction buffer
    return cVariablePrefix + std::to_string(arg_index) + "[0]";
}

template<typename T, typename Function, typename... Exprs>
cl::Buffer ReduceExpr<T, Function, Exprs...>::opencl_evaluate_kernel(
        cl::CommandQueue &,
        std::vector<cl::Buffer> &buffers,
        std::string &,
        std::string &
) const noexcept
{
    return buffers.back();
}

#endif

/* -------------------------------------------------- */

template<typename RHS>
class UnaryNegExpr : public Expression<UnaryNegExpr<RHS>> {
/*@UnaryNegExpr*/
public:

    /* ---------------- typedefs --------------- */

    constexpr static size_t rank()
    { return RHS::rank(); }

    typedef typename RHS::value_type value_type;
    template<typename X> using container_type = typename RHS::template container_type<X>;
    typedef UnaryNegExpr self_type;
    typedef typename RHS::return_type return_type;

    /* ---------------- Friends ---------------- */

    template<typename RHS_>
    friend UnaryNegExpr<RHS_> operator-(Expression<RHS_> const &rhs);

    template<typename RHS_>
    friend UnaryNegExpr<RHS_> _neg_(Expression<RHS_> const &rhs);

    template<typename LHS_, typename RHS_> friend
    class BinaryAddExpr;

    template<typename LHS_, typename RHS_> friend
    class BinarySubExpr;

    template<size_t I1_, size_t I2_, typename LHS_, typename RHS_> friend
    class BinaryMulExpr;

    template<typename LHS_, typename RHS_> friend
    class BinaryHadExpr;

    template<typename Function_, typename... Exprs_> friend
    class MapExpr;

    template<typename U, typename Function_, typename... Exprs_> friend
    class ReduceExpr;

    template<typename RHS_> friend
    class UnaryNegExpr;

    template<typename Expr> friend
    class opencl::Model;

    /* ---------------- Getters ----------------- */

    size_t dimension(size_t index) const
    { return rhs_.dimension(index); }

    Shape<RHS::rank()> const &shape() const
    { return rhs_.shape(); }

    /** Evaluate and return the resulting Tensor */
    return_type eval() const noexcept
    { return (*this)(); }

    template<typename... Args>
    auto operator()(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<RHS const>()(args...))>::type;

    template<typename... Args>
    auto at(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<RHS const>().at(args...))>::type;

    template<size_t M>
    auto operator[](Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<RHS const>()[indices])>::type;

    template<size_t... Slices, typename... Args>
    auto slice(Args... args) const
    -> typename std::remove_reference<decltype(std::declval<RHS const>().slice(args...))>::type;

    template<size_t... Slices, size_t M>
    auto slice(Indices<M> const &indices) const
    -> typename std::remove_reference<decltype(std::declval<RHS const>().slice(indices))>::type;

    std::string str() const;


    /* ---------------- OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

    opencl::Model<self_type> opencl() const
    { return opencl::Model<self_type>(*this); }

    std::string opencl_kernel_code(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list
    ) const noexcept;

    cl::Buffer opencl_evaluate_kernel(
            cl::CommandQueue &cqueue,
            std::vector<cl::Buffer> &buffers,
            std::string &arg_list,
            std::string &expr
    ) const noexcept;

#endif

private:

    /* -------------- Constructors -------------- */

    explicit UnaryNegExpr(RHS const &rhs);

    UnaryNegExpr(UnaryNegExpr<RHS> const &) = default;

    /* ------------------ Data ------------------ */

    RHS const &rhs_;
};

template<typename RHS>
UnaryNegExpr<RHS> operator-(Expression<RHS> const &rhs)
{
    return UnaryNegExpr<RHS>(rhs.self());
}

template<typename RHS>
UnaryNegExpr<RHS> _neg_(Expression<RHS> const &rhs)
{
    return UnaryNegExpr<RHS>(rhs.self());
}

/* ------------------ Getters ------------------ */

template<typename RHS>
template<typename... Args>
auto UnaryNegExpr<RHS>::operator()(Args... args) const
-> typename std::remove_reference<decltype(std::declval<RHS const>()(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return -rhs_(args...);
}

template<typename RHS>
template<typename... Args>
auto UnaryNegExpr<RHS>::at(Args... args) const
-> typename std::remove_reference<decltype(std::declval<RHS const>().at(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    return -rhs_(args...);
}

template<typename RHS>
template<size_t M>
auto UnaryNegExpr<RHS>::operator[](Indices<M> const &indices) const
-> typename std::remove_reference<decltype(std::declval<RHS const>()[indices])>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    return -rhs_[indices];
}

template<typename RHS>
template<size_t... Slices, typename... Args>
auto UnaryNegExpr<RHS>::slice(Args... args) const
-> typename std::remove_reference<decltype(std::declval<RHS const>().slice(args...))>::type
{
    static_assert(rank() >= sizeof...(args), RANK_OUT_OF_BOUNDS);
    static_assert(rank() >= sizeof...(Slices) + sizeof...(Args), SLICES_OUT_OF_BOUNDS);
    return -rhs_.template slice<Slices...>(args...);
}

template<typename RHS>
template<size_t... Slices, size_t M>
auto UnaryNegExpr<RHS>::slice(Indices<M> const &indices) const
-> typename std::remove_reference<decltype(std::declval<RHS const>().slice(indices))>::type
{
    static_assert(rank() >= M, RANK_OUT_OF_BOUNDS);
    static_assert(rank() >= sizeof...(Slices) + M, SLICES_OUT_OF_BOUNDS);
    return -rhs_.template slice<Slices...>(indices);
}

/* ----------------- Print ----------------- */

template<typename RHS>
std::string UnaryNegExpr<RHS>::str() const
{
    return "(- " + rhs_.str() + ")";
}

/* ------------------ OpenCL ----------------- */

#ifdef _ENABLE_OPENCL

template<typename RHS>
std::string UnaryNegExpr<RHS>::opencl_kernel_code(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list
) const noexcept
{
    return "-" + rhs_.opencl_kernel_code(cqueue, buffers, arg_list);
}

template<typename RHS>
cl::Buffer UnaryNegExpr<RHS>::opencl_evaluate_kernel(
        cl::CommandQueue &cqueue,
        std::vector<cl::Buffer> &buffers,
        std::string &arg_list,
        std::string &expr
) const noexcept
{
    return opencl::details::evaluate_basic_kernel<value_type>(
            cqueue,
            buffers,
            arg_list,
            expr,
            shape().index_product()
    );
}

#endif

/* ------------------ Constructors ------------------ */

template<typename RHS>
UnaryNegExpr<RHS>::UnaryNegExpr(RHS const &rhs)
        : rhs_(rhs)
{}

} // namespace tensor

#undef NTENSOR_0CONSTRUCTOR
#undef NCONSTRUCTOR_0TENSOR
#undef NELEMENTS
#undef ZERO_ELEMENT
#undef EXPECTING_C_ARRAY
#undef EXPECTING_TENSOR
#undef EXPECTING_SCALAR
#undef EXPECTING_EXPRESSION
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
#undef NO_TENSORS_PROVIDED
#undef SLICING_MULTIPLICATION_AXIS
#undef PANIC_ASSERTION
#undef OPENCL_NO_PLATFORMS
#undef OPENCL_NO_DEVICES
#undef OPENCL_ARITY_ERROR
#undef OPENCL_REDUCTION_SIZE_ERROR

#ifndef NDEBUG

#undef PRINT
#undef PRINTV
#undef ARRAY_SIZE
#undef PRINT_ARRAY1
#undef PRINT_ARRAY2
#undef GET_MACRO
#undef PRINT_ARRAY

#endif // not defined NDEBUG

#endif // TENSORS_H_
