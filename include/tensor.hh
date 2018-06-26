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
  "Incorrect number of elements provided -- "
#define ZERO_ELEMENT(CLASS) \
  CLASS " Cannot be constructed with a zero dimension"

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

// Iteration
#define BEGIN_ON_NON_VECTOR \
  "Tensor::begin() should only be called on rank 1 tensors, " \
  "use Tensor::begin(size_t) instead"

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

template <size_t N> class Shape;
template <typename T, size_t N = 0> class Tensor;
template <typename LHS, typename RHS> class BinaryAdd;
template <typename LHS, typename RHS> class BinarySub;
template <typename LHS, typename RHS> class BinaryMul;

/* ----------------- Template Meta-Patterns ----------------- */

/** Meta-template logical &&
 */
template <bool B1, bool B2>
struct LogicalAnd { static bool const value = B1 && B2; };

/** Boolean member `value` is true if T is an any-rank Tensor object, false o.w.
 */
template <typename T>
struct IsTensor { static bool const value = false; };

/** Tensor specialization of IsTensor, Boolean member `value` is true
 */
template <>
template <typename T, size_t N>
struct IsTensor<Tensor<T, N>> { static bool const value = true; };

/** Boolean member `value` is true if T is a 0-rank Tensor object, false o.w.
 */
template <typename T>
struct IsScalar { static bool const value = true; };

/** Scalar specialization of IsScalar, Boolean member `value` is true
 */
template <>
template <typename T>
struct IsScalar<Tensor<T, 0>> { static bool const value = true; };

/** Tensor specialization of IsScalar, Boolean member `value` is false 
 */
template <>
template <typename T, size_t N>
struct IsScalar<Tensor<T, N>> { static bool const value = false; };

/** Tensor member `value` is a wrapper for input `val` if val is
 *  a Tensor, o.w. `value` is a reference to Tensor `val`
 */
template <typename T>
struct ValueToTensor {
  ValueToTensor(T &&val): value(std::forward<T>(val)) {}
  Tensor<T, 0> value;
};

/** Tensor specialization of ValueToTensor, `value` is a 
 *  const reference to the provided Tensor
 */
template <>
template <typename T, size_t N>
struct ValueToTensor<Tensor<T, N>> {
  ValueToTensor(Tensor<T, N> const &val): value(val) {}
  Tensor<T, N> const &value;
};

/** BinaryAdd specialization of ValueToTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <>
template <typename LHS, typename RHS>
struct ValueToTensor<BinaryAdd<LHS, RHS>> {
  ValueToTensor(BinaryAdd<LHS, RHS> const &val): value(val) {}
  BinaryAdd<LHS, RHS> const &value;
};

/** BinarySub specialization of ValueToTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <>
template <typename LHS, typename RHS>
struct ValueToTensor<BinarySub<LHS, RHS>> {
  ValueToTensor(BinarySub<LHS, RHS> const &val): value(val) {}
  BinarySub<LHS, RHS> const &value;
};

/** BinaryMul specialization of ValueToTensor, `value` is a 
 *  const reference to the provided binary expression
 */
template <>
template <typename LHS, typename RHS>
struct ValueToTensor<BinaryMul<LHS, RHS>> {
  ValueToTensor(BinaryMul<LHS, RHS> const &val): value(val) {}
  BinaryMul<LHS, RHS> const &value;
};

/* ---------------------------------------------------------- */

/** CRTP base for Tensor expressions
 */
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

  template <typename X, size_t M> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ------------------ Constructors ------------------ */

  explicit Shape(size_t const (&dimensions)[N]); /**< size_t[] constructor */
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

  template <typename X, typename Y, size_t M1, size_t M2>
  friend Tensor<X, M1 + M2 - 2> Multiply(Tensor<X, M1> const& tensor_1, Tensor<Y, M2> const& tensor_2);

  /* ------------------- Utility -------------------- */

  /** Returns the product of all of the indices */
  size_t index_product() const noexcept;

  /* -------------------- Print --------------------- */

  template <typename X, size_t M>
  friend std::ostream &operator<<(std::ostream &os, Tensor<X, M> const&tensor);
  template <size_t M>
  friend std::ostream &operator<<(std::ostream &os, Shape<M> const &shape);

private:
  /* ------------------ Data ------------------ */

  size_t dimensions_[N]; /**< Underlying data */

  /* --------------- Constructor --------------- */

  // FIXME :: dumb hack to avoid ambiguous overload
  Shape(size_t const *dimensions, int);  
  Shape() = default;                      
};

template <size_t N>
Shape<N>::Shape(size_t const (&dimensions)[N])
{
  for (size_t i = 0; i < N; ++i) 
    if (!dimensions[i]) throw std::logic_error(ZERO_ELEMENT("Shape"));
  std::copy_n(dimensions, N, dimensions_);
}

template <size_t N>
Shape<N>::Shape(Shape const &shape)
{
  std::copy_n(shape.dimensions_, N, dimensions_);
}

template <size_t N>
Shape<N>::Shape(size_t const *dimensions, int)
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
  if (N < index || index == 0)
    throw std::logic_error(DIMENSION_INVALID("Tensor::dimension(size_t)"));

  // indexing begins at 1
  return dimensions_[index - 1];
}

template <size_t N>
size_t Shape<N>::operator[](size_t index) const
{
  if (N < index || index == 0)
    throw std::logic_error(DIMENSION_INVALID("Tensor::dimension(size_t)"));

  // indexing begins at 1
  return dimensions_[index - 1];
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
class Indices {
  /** Wrapper around size_t[N] to provide a specialized static array
   *  for accessing Tensors. Largely unnecessory.
   */
public:
  explicit Indices(size_t const (&indices)[N]);
  size_t operator[](size_t index) const;
private:
  size_t indices_[N];
};

template <size_t N>
Indices<N>::Indices(size_t const (&indices)[N])
{
  std::copy_n(indices, N, indices_);
}

template <size_t N>
size_t Indices<N>::operator[](size_t index) const
{
  if (index > N) throw std::logic_error(DIMENSION_INVALID("Indices::operator[]"));
  return indices_[index];
}

template <typename T, size_t N>
class Tensor: public Expression<Tensor<T, N>> { /*@Tensor<T, N>*/
/** Any-rank array of type `T`, where rank `N` is a size_t template.
 *  The underlying data is implemented as a dynamically allocated contiguous
 *  array.
 */  
public:

  /* ------------------ Type Definitions --------------- */
  typedef T                                     value_type;
  typedef T&                                    reference;
  typedef T const&                              const_reference;
  typedef size_t                                size_type;
  typedef ptrdiff_t                             difference_type;
  typedef Tensor<T, N>                          self_type;

  /* ----------------- Friend Classes ----------------- */

  template <typename X, size_t M> friend class Tensor;
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
    template <typename U, size_t M> friend class Tensor;
    Proxy() = delete;
  private:
    Proxy(Tensor<T, N> const &tensor): tensor_(tensor) {}
    Proxy(Proxy const &proxy): tensor_(proxy.tensor_) {}
    Tensor<T, N> const &tensor_;
  }; 

  /* ------------------ Constructors ------------------ */

  /** Creates a Tensor with dimensions described by `dimensions`.
   *  Elements are zero initialized. Note: dimensions index from 1.
   */
  explicit Tensor(size_t const (&dimensions)[N]);

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

  /** Creates a Tensor with dimensions described by `shape`.
   *  Elements are zero initialized. Note: dimensions index from 1.
   */
  explicit Tensor(Shape<N> const &shape): Tensor(shape.dimensions_) {}

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
  Tensor(Tensor<T, N> const &tensor); 

  /** Move construction, takes ownership of underlying data, `tensor` is destroyed */
  Tensor(Tensor<T, N> &&tensor); 

  /** Constructs a reference to the `proxy` tensor. The tensors share 
   *  the same underyling data, so changes will affect both tensors.
   */
  Tensor(Tensor<T, N>::Proxy const &proxy); 

  /** Constructs the tensor produced by the expression */
  template <typename NodeType>
  Tensor(Expression<NodeType> const& expression);

  /* ------------------- Assignment ------------------- */

  /** Copy Constructs from `tensor`. Destroys itself first */
  Tensor<T, N> &operator=(Tensor<T, N> const &tensor);

  /** Proxy constructs from `tensor`. Destroys itself first */
  template <typename NodeType>
  Tensor<T, N> &operator=(Expression<NodeType> const &rhs);

  /* ----------------- Getters ----------------- */

  constexpr static size_t rank() { return N; } /**< Get `N` */

  /** Get the dimension at index. Throws std::logic_error if index
   *  is out of bounds. Note: indexing starts at 1.
   */
  size_t dimension(size_t index) const { return shape_[index]; }

  Shape<N> shape() const noexcept { return shape_; } /**< Get the tensor shape */
  // FIXME :: Is there a way to hide these and keep -> functional?
  Tensor *operator->() { return this; } /**> used to implement iterator-> */
  Tensor const *operator->() const { return this; } /**> used to implement const_iterator-> */

  /* ------------------ Access To Data ----------------- */

  /** Returns the resulting tensor by applying left to right index expansion of
   *  the provided arguments. I.e. calling `tensor(1, 2)` on a rank 4 tensor is
   *  equivalent to `tensor(1, 2, :, :)`. Throws std::logic_error if any of the 
   *  indices are out bounds. Note: indexing starts at 1.
   */
  template <typename... Indices>
  Tensor<T, N - sizeof...(Indices)> operator()(Indices... args);

  /** See operator()
   */
  template <typename... Indices>
  Tensor<T, N - sizeof...(Indices)> const operator()(Indices... args) const;

  /** See operator()
   */
  template <size_t M>
  Tensor<T, N - M> operator[](Indices<M> const &indices);

  /** Slices denotate the dimensions which are left free, while indices
   *  fix the remaining dimensions at the specified index. I.e. calling
   *  `tensor.slice<1, 3, 5>(1, 2)` on a rank 5 tensor is equivalent to
   *  `tensor(:, 1, :, 2, :)` and will produce a rank 3 tensor. Throws
   *   std::logic_error if any of the indices are out of bounds. Note:
   *   indexing begins at 1.
   */
  template <size_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices)> slice(Indices... indices);
  template <size_t... Slices, typename... Indices>
  Tensor<T, sizeof...(Slices)> const slice(Indices... indices) const;

  /* -------------------- Expressions ------------------- */

  template <typename X, typename Y, size_t M>
  friend Tensor<X, M> Add(Tensor<X, M> const& tensor_1, Tensor<Y, M> const& tensor_2);

  template <typename X, size_t M, typename>
  friend Tensor<X, M> operator+(Tensor<X, M> const &tensor, X const &scalar);

  template <typename X, typename Y, size_t M>
  friend Tensor<X, M> Subtract(Tensor<X, M> const& tensor_1, Tensor<Y, M> const& tensor_2);

  template <typename X, size_t M, typename>
  friend Tensor<X, M> operator-(Tensor<X, M> const &tensor, X const &scalar);

  template <typename X, size_t M, typename>
  friend Tensor<X, M> operator-(X const &scalar, Tensor<X, M> const &tensor);

  template <typename X, typename Y, size_t M1, size_t M2>
  friend Tensor<X, M1 + M2 - 2> Multiply(Tensor<X, M1> const& tensor_1, Tensor<Y, M2> const& tensor_2);
  Tensor<T, N> operator-() const;

  template <typename X, size_t M, typename>
  friend Tensor<X, M> operator*(Tensor<X, M> const &tensor, X const &scalar);

  /* ------------------ Print to ostream --------------- */

  template <typename X, size_t M>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X, M> &tensor);

  /* -------------------- Equivalence ------------------ */

  /** Returns true iff the tensor's dimensions and data are equivalent */
  bool operator==(Tensor<T, N> const& tensor) const; 

  /** Returns true iff the tensor's dimensions or data are not equivalent */
  bool operator!=(Tensor<T, N> const& tensor) const { return !(*this == tensor); }

  /** Returns true iff the tensor's dimensions are equal and every element satisfies e1 == e2 */
  template <typename X>
  bool operator==(Tensor<X, N> const& tensor) const;

  /** Returns true iff the tensor's dimensions are different or any element satisfies e1 != e2 */
  template <typename X>
  bool operator!=(Tensor<X, N> const& tensor) const { return !(*this == tensor); }

  /* -------------------- Iterators --------------------- */

  class Iterator { /*@Iterator<T, N>*/
  public:
    /** Iterator with freedom across one dimension of a Tensor.
     *  Allows access to the underlying tensor data.
     */

    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    Iterator(Iterator const &it);  

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    Iterator(Iterator &&it);       

    Tensor<T, N> operator*();   /**< Create a reference to the underlying Tensor */
    Tensor<T, N> operator->();  /**< Syntatic sugar for (*it). */
    Iterator operator++(int);   /**< Increment (postfix). Returns a temporary before increment */
    Iterator &operator++();     /**< Increment (prefix). Returns *this */
    Iterator operator--(int);   /**< Decrement (postfix). Returns a temporary before decrement */
    Iterator &operator--();     /**< Decrement (prefix). Returns *this */

    /** Returns true iff the underlying pointers are identical */
    bool operator==(Iterator const &it) const { return (it.data_ == this->data_); }

    /** Returns true iff the underlying pointers are not identical */
    bool operator!=(Iterator const &it) const { return !(it == *this); }

  private:

    // Direct construction
    Iterator(Tensor<T, N + 1> const &tensor, size_t index);
    Shape<N> shape_; // Data describing the underlying tensor 
    size_t strides_[N];
    value_type *data_;
    std::shared_ptr<T> ref_;
    size_t stride_; // Step size of the underlying data pointer per increment
  };

  class ConstIterator { /*@ConstIterator<T, N>*/
  public:
    /** Constant iterator with freedom across one dimension of a Tensor.
     *  Does not allow write access to the underlying tensor data.
     */

    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    ConstIterator(ConstIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ConstIterator(ConstIterator &&it);

    Tensor<T, N> const operator*();  /**< Create a reference to the underlying Tensor */
    Tensor<T, N> const operator->(); /**< Syntatic sugar for (*it). */                                
    ConstIterator operator++(int);   /**< Increment (postfix). Returns a temporary before increment */
    ConstIterator &operator++();     /**< Increment (prefix). Returns *this */
    ConstIterator operator--(int);   /**< Decrement (postfix). Returns a temporary before decrement */
    ConstIterator &operator--();     /**< Decrement (prefix). Returns *this */
    bool operator==(ConstIterator const &it) const { return (it.data_ == this->data_); }
    bool operator!=(ConstIterator const &it) const { return !(it == *this); }
  private:
    ConstIterator(Tensor<T, N + 1> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    value_type *data_;
    std::shared_ptr<T> ref_;

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

    template <typename U, size_t M> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy construct an iterator to the same underlying Tensor */
    ReverseIterator(ReverseIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ReverseIterator(ReverseIterator &&it);

    Tensor<T, N> operator*();        /**< Create a reference to the underlying Tensor */
    Tensor<T, N> operator->();       /**< Syntatic sugar for (*it). */                                
    ReverseIterator operator++(int); /**< Increment (postfix). Returns a temporary before increment */
    ReverseIterator &operator++();   /**< Increment (prefix). Returns *this */
    ReverseIterator operator--(int); /**< Decrement (postfix). Returns a temporary before decrement */
    ReverseIterator &operator--();   /**< Decrement (prefix). Returns *this */
    bool operator==(ReverseIterator const &it) const { return (it.data_ == this->data_); }
    bool operator!=(ReverseIterator const &it) const { return !(it == *this); }
  private:
    ReverseIterator (Tensor<T, N + 1> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    value_type *data_;
    std::shared_ptr<T> ref_;

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

    template <typename U, size_t M> friend class Tensor;

    /* --------------- Constructors --------------- */

    /** Copy constructs an iterator to the same underlying Tensor */
    ConstReverseIterator(ConstReverseIterator const &it);

    /** Move construct an iterator to the same underlying Tensor. Destroys `it`. */
    ConstReverseIterator(ConstReverseIterator &&it);
    Tensor<T, N> const operator*();       /**< Create a reference to the underlying Tensor */
    Tensor<T, N> const operator->();      /**< Syntatic sugar for (*it). */                                
    ConstReverseIterator operator++(int); /**< Increment (postfix). Returns a temporary before increment */
    ConstReverseIterator &operator++();   /**< Increment (prefix). Returns *this */
    ConstReverseIterator operator--(int); /**< Decrement (postfix). Returns a temporary before decrement */
    ConstReverseIterator &operator--();   /**< Decrement (prefix). Returns *this */
    bool operator==(ConstReverseIterator const &it) const { return (it.data_ == this->data_); }
    bool operator!=(ConstReverseIterator const &it) const { return !(it == *this); }
  private:
    ConstReverseIterator (Tensor<T, N + 1> const &tensor, size_t index);

    /**
     * Data describing the underlying tensor
     */
    Shape<N> shape_;
    size_t strides_[N];
    value_type *data_;
    std::shared_ptr<T> ref_;

    /**
     * Step size of the underlying data pointer per increment
     */
    size_t stride_;
  };

  typename Tensor<T, N - 1>::Iterator begin(size_t index);
  typename Tensor<T, N - 1>::Iterator end(size_t index);
  typename Tensor<T, N - 1>::Iterator begin();
  typename Tensor<T, N - 1>::Iterator end();

  typename Tensor<T, N - 1>::ConstIterator cbegin(size_t index) const;
  typename Tensor<T, N - 1>::ConstIterator cend(size_t index) const;
  typename Tensor<T, N - 1>::ConstIterator cbegin() const;
  typename Tensor<T, N - 1>::ConstIterator cend() const;

  typename Tensor<T, N - 1>::ReverseIterator rbegin(size_t index);
  typename Tensor<T, N - 1>::ReverseIterator rend(size_t index);
  typename Tensor<T, N - 1>::ReverseIterator rbegin();
  typename Tensor<T, N - 1>::ReverseIterator rend();

  typename Tensor<T, N - 1>::ConstReverseIterator crbegin(size_t index) const;
  typename Tensor<T, N - 1>::ConstReverseIterator crend(size_t index) const;
  typename Tensor<T, N - 1>::ConstReverseIterator crbegin() const;
  typename Tensor<T, N - 1>::ConstReverseIterator crend() const;

  /* ----------------- Utility Functions ---------------- */

  Tensor<T, N> copy() const;
  Tensor<T, N>::Proxy ref();
  template <typename U, size_t M, typename Container>
  friend void Fill(Tensor<U, M> &tensor, Container const &container);
  

private:

  /* ----------------- Data ---------------- */

  Shape<N> shape_;
  size_t strides_[N];
  value_type *data_;
  std::shared_ptr<T> ref_;

  /* --------------- Getters --------------- */

  size_t const *strides() const noexcept { return strides_; }

  /* ------------- Expansion for operator()() ------------- */

  // Expansion
  template <size_t M>
  Tensor<T, N - M> pAccessExpansion(size_t cumul_index);
  template <size_t M, typename... Indices>
  Tensor<T, N - M> pAccessExpansion(size_t cumul_index, size_t next_index, Indices...);

  /* ------------- Expansion for slice() ------------- */

  // Expansion
  template <size_t M, typename... Indices>
  Tensor<T, N - M> pSliceExpansion(size_t * placed_indices, size_t array_index, size_t next_index, Indices... indices);
  template <size_t M>
  Tensor<T, N - M> pSliceExpansion(size_t * placed_indices, size_t);

  // Index checking and placement
  template <size_t I1, size_t I2, size_t... Indices>
  static void pSliceIndex(size_t *placed_indices);
  template <size_t I1>
  static void pSliceIndex(size_t *placed_indices);

  /* ----------------- Utility -------------------- */

  // Data mapping
  template <size_t M>
  void pUpdateQuotas(size_t (&dim_quotas)[M], size_t quota_offset = 0,
      size_t offset = 0) const;
  template <size_t M>
  size_t pEvaluateIndex(size_t const (&dim_quotas)[M], size_t offset = 0) const;
  void pMap(std::function<void(T *lhs)> const &fn);
  template <typename X>
  void pUnaryMap(Tensor<X, N> const &tensor, std::function<void(T *lhs, X *rhs)> const &fn);
  template <typename X, typename Y>
  void pBinaryMap(Tensor<X, N> const &tensor_1, Tensor<Y, N> const &tensor_2,
      std::function<void(T *lhs, X *rhs1, Y *rhs2)> const& fn);

  // allocate new space and copy data
  value_type * pDuplicateData() const;

  // Initialize strides :: DIMENSIONS MUST BE INITIALIZED FIRST
  void pInitializeStrides();

  // Declare all fields in jthe constructor
  Tensor(size_t const *dimensions, size_t const *strides, T *data, std::shared_ptr<T> &&_ref);

}; // Tensor

/* ----------------------------- Constructors ------------------------- */

template <typename T, size_t N>
Tensor<T, N>::Tensor(size_t const (&dimensions)[N])
  : shape_(Shape<N>(dimensions)), data_(new T[shape_.index_product()]),
  ref_(data_, _ARRAY_DELETER(T))
{ 
  for (size_t i = 0; i < N; ++i) 
    if (!dimensions[i]) throw std::logic_error(ZERO_ELEMENT("Tensor"));
  pInitializeStrides(); 
}

template <typename T, size_t N>
Tensor<T, N>::Tensor(size_t const (&dimensions)[N], T const& value)
  : shape_(Shape<N>(dimensions))
{
  for (size_t i = 0; i < N; ++i) 
    if (!dimensions[i]) throw std::logic_error(ZERO_ELEMENT("Tensor"));
  pInitializeStrides();
  size_t cumul = shape_.index_product();
  data_ = new T[cumul];
  std::fill(data_, data_ + cumul, value);
  ref_ = std::shared_ptr<T>(data_, _ARRAY_DELETER(T));
}

template <typename T, size_t N>
template <typename FunctionType, typename... Arguments>
Tensor<T, N>::Tensor(size_t const (&dimensions)[N], std::function<FunctionType> &f, Arguments&&... args)
  : shape_(Shape<N>(dimensions))
{
  for (size_t i = 0; i < N; ++i) 
    if (!dimensions[i]) throw std::logic_error(ZERO_ELEMENT("Tensor"));
  pInitializeStrides();
  data_ = new T[shape_.index_product()];
  std::function<void(T*)> value_setter = [&f, &args...](T *lhs) -> void {
    *lhs = f(args...);
  };
  pMap(value_setter);
  ref_ = std::shared_ptr<T>(data_, _ARRAY_DELETER(T));
}

template <typename T, size_t N, typename Container>
void Fill(Tensor<T, N> &tensor, Container const &container)
{
  size_t cumul_sum = tensor.shape_.index_product();
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

template <typename T, size_t N>
Tensor<T, N>::Tensor(Tensor<T, N> const &tensor)
  : shape_(tensor.shape_)
{
  pInitializeStrides();
  size_t cumul = shape_.index_product();
  this->data_ = new T[cumul];
  size_t dim_quotas[N];
  std::copy_n(shape_.dimensions_, N, dim_quotas);
  for (size_t i = 0; i < cumul; ++i) {
    size_t index = tensor.pEvaluateIndex(dim_quotas);
    this->data_[i] = tensor.data_[index];
    pUpdateQuotas(dim_quotas);
  }
  ref_ = std::shared_ptr<T>(data_, _ARRAY_DELETER(T));
}

template <typename T, size_t N>
Tensor<T, N>::Tensor(Tensor<T, N> &&tensor)
  : shape_(tensor.shape_), data_(tensor.data_), ref_(std::move(tensor.ref_))
{
  std::copy_n(tensor.strides_, N, strides_);
  tensor.data_ = nullptr;
}

template <typename T, size_t N>
Tensor<T, N>::Tensor(Tensor<T, N>::Proxy const &proxy)
  : shape_(proxy.tensor_.shape_), data_(proxy.tensor_.data_), ref_(proxy.tensor_.ref_)
{
  std::copy_n(proxy.tensor_.strides_, N, strides_);
}

template <typename T, size_t N>
template <typename NodeType>
Tensor<T, N>::Tensor(Expression<NodeType> const& expression)
{
  auto result = expression.self()();
  std::copy_n(result.strides_, N, strides_);
  shape_ = result.shape_;
  data_ = result.data_;
  ref_ = std::move(result.ref_);
}

template <typename T, size_t N>
Tensor<T, N> &Tensor<T, N>::operator=(const Tensor<T, N> &tensor)
{
  if (shape_ != tensor.shape_)
      throw std::logic_error(DIMENSION_MISMATCH("Tensor::operator=(Tensor const&)"));
  std::function<void(T*, T*)> fn = [](T *x, T *y) -> void { *x = *y; };
  pUnaryMap(tensor, fn);
  return *this;
}

template <typename T, size_t N>
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

template <typename T, size_t N>
template <typename... Indices>
Tensor<T, N - sizeof...(Indices)> Tensor<T, N>::operator()(Indices... args)
{
  static_assert(N >= sizeof...(args), RANK_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));
  return pAccessExpansion<sizeof...(args)>(0, args...);
}

template <typename T, size_t N>
template <typename... Indices>
Tensor<T, N - sizeof...(Indices)> const Tensor<T, N>::operator()(Indices... args) const
{
  return (*const_cast<self_type*>(this))(args...);
}

template <typename T, size_t N>
template <size_t M>
Tensor<T, N - M> Tensor<T, N>::operator[](Indices<M> const &indices)
{
  size_t cumul_index = 0;
  for (size_t i = 0; i < M; ++i)
    cumul_index += strides_[N - i - 1] * (indices[i] - 1);
  return Tensor<T, N - M>(shape_.dimensions_ + M, strides_ + M,  data_ + cumul_index, std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
template <size_t... Slices, typename... Indices>
Tensor<T, sizeof...(Slices)> Tensor<T, N>::slice(Indices... indices)
{
  static_assert(sizeof...(Slices));
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

template <typename T, size_t N>
template <size_t... Slices, typename... Indices>
Tensor<T, sizeof...(Slices)> const Tensor<T, N>::slice(Indices... indices) const
{
  return const_cast<self_type*>(this)->slice<Slices...>(indices...);
}

template <typename T, size_t N>
bool Tensor<T, N>::operator==(Tensor<T, N> const& tensor) const
{
  if (shape_ != tensor.shape_)
    throw std::logic_error(DIMENSION_MISMATCH("Tensor::operator==(Tensor const&)"));

  size_t indices_product = shape_.index_product();
  for (size_t i = 0; i < indices_product; ++i)
    if (data_[i] != tensor.data_[i]) return false;
  return true;
}

template <typename T, size_t N>
template <typename X>
bool Tensor<T, N>::operator==(Tensor<X, N> const& tensor) const
{
  if (shape_ != tensor.shape_)
    throw std::logic_error(DIMENSION_MISMATCH("Tensor::operator==(Tensor const&)"));

  size_t indices_product = shape_.index_product();
  for (size_t i = 0; i < indices_product; ++i)
    if (data_[i] != tensor.data_[i]) return false;
  return true;
}

// possible iostream overload implemention
template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const Tensor<T, N> &tensor)
{
  auto add_brackets = [&os](size_t n, bool left)
  {
    for (size_t i = 0; i < n; ++i) os << (left ?'[' : ']');
  };
  size_t cumul_index = tensor.shape_.index_product();
  size_t dim_quotas[N];
  std::copy_n(tensor.shape_.dimensions_, N, dim_quotas);

  add_brackets(N, true); // opening brackets
  os << tensor.data_[0]; // first element
  for (size_t i = 0; i < cumul_index - 1; ++i) {
    size_t dim_index = N, bracket_count = 0;
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
    size_t index = tensor.pEvaluateIndex(dim_quotas);
    add_brackets(bracket_count - 1, false);
    os << ", ";
    add_brackets(bracket_count - 1, true);
    os << tensor.data_[index];
  }
  add_brackets(N, false); // closing brackets
  return os;
}

// private methods
template <typename T, size_t N>
template <size_t M>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(size_t cumul_index)
{
  return Tensor<T, N - M>(shape_.dimensions_ + M, strides_ + M, data_ + cumul_index, std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
template <size_t M, typename... Indices>
Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(
 size_t cumul_index, size_t next_index, Indices... rest)
{
  if (next_index > shape_.dimensions_[M - sizeof...(rest) - 1] || next_index == 0)
    throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));

  // adjust for 1 index array access
  cumul_index += strides_[M - sizeof...(rest) - 1] * (next_index - 1);
  return pAccessExpansion<M>(cumul_index, rest...);
}

/* ------------- Slice Expansion ------------- */

template <typename T, size_t N>
template <size_t I1, size_t I2, size_t... Indices>
void Tensor<T, N>::pSliceIndex(size_t *placed_indices)
{
  static_assert(N >= I1, INDEX_OUT_OF_BOUNDS("Tensor::Slice(Indices...)"));
  static_assert(I1 != I2, SLICE_INDICES_REPEATED);
  static_assert(I1 < I2, SLICE_INDICES_DESCENDING);
  static_assert(I1, ZERO_INDEX("Tensor::Slice(Indices...)"));
  placed_indices[I1 - 1] = 0;
  pSliceIndex<I2, Indices...>(placed_indices);
}

template <typename T, size_t N>
template <size_t Index>
void Tensor<T, N>::pSliceIndex(size_t *placed_indices)
{
  static_assert(N >= Index, INDEX_OUT_OF_BOUNDS("Tensor::Slice(Indices...)"));
  static_assert(Index, ZERO_INDEX("Tensor::Slice(Indices...)"));
  placed_indices[Index - 1] = 0;
}

template <typename T, size_t N>
template <size_t M, typename... Indices>
Tensor<T, N - M> Tensor<T, N>::pSliceExpansion(size_t * placed_indices, size_t array_index, size_t next_index, Indices... indices)
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

template <typename T, size_t N>
template <size_t M>
Tensor<T, N - M> Tensor<T, N>::pSliceExpansion(size_t *placed_indices, size_t)
{
  size_t offset = 0;
  size_t dimensions[N - M];
  size_t strides[N - M];
  size_t array_index = 0;
  for (size_t i = 0; i < N; ++i) {
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

template <typename T, size_t N>
T * Tensor<T, N>::pDuplicateData() const
{
  size_t count = shape_.index_product();
  T * data = new T[count];
  std::copy_n(this->data_, count, data);
  return data;
}

// Update the quotas after one iterator increment
template <typename T, size_t N>
template <size_t M>
void Tensor<T, N>::pUpdateQuotas(size_t (&dim_quotas)[M], size_t quota_offset,
    size_t offset) const
{
  size_t dim_index = M;
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
template <typename T, size_t N>
template <size_t M>
size_t Tensor<T, N>::pEvaluateIndex(size_t const (&dim_quotas)[M], size_t offset) const
{
  size_t index = 0;
  for (size_t i = 0; i < M; ++i)
      index += strides_[i + offset] * (shape_.dimensions_[i + offset] - dim_quotas[i]);
  return index;
}

template <typename T, size_t N>
void Tensor<T, N>::pMap(std::function<void(T *lhs)> const &fn)
{
  // this is the index upper bound for iteration
  size_t cumul_index = shape_.index_product();
  size_t dim_quotas[N];
  std::copy_n(shape_.dimensions_, N, dim_quotas);
  for (size_t i = 0; i < cumul_index; ++i) {
    size_t index = pEvaluateIndex(dim_quotas);
    fn(&(data_[index]));
    pUpdateQuotas(dim_quotas);
  }
}

template <typename T, size_t N>
template <typename X>
void Tensor<T, N>::pUnaryMap(Tensor<X, N> const &tensor,
    std::function<void(T *lhs, X *rhs)> const &fn)
{
  // this is the index upper bound for iteration
  size_t cumul_index = shape_.index_product();

  size_t dim_quotas[N];
  std::copy_n(shape_.dimensions_, N, dim_quotas);
  for (size_t i = 0; i < cumul_index; ++i) {
    size_t index = pEvaluateIndex(dim_quotas);
    size_t t_index = tensor.pEvaluateIndex(dim_quotas);
    fn(&(data_[index]), &(tensor.data_[t_index]));
    pUpdateQuotas(dim_quotas);
  }
}

template <typename T, size_t N>
template <typename X, typename Y>
void Tensor<T, N>::pBinaryMap(Tensor<X, N> const &tensor_1, Tensor<Y, N> const &tensor_2, std::function<void(T *lhs, X *rhs1, Y *rhs2)> const &fn)
{
  size_t cumul_index = shape_.index_product();
  size_t dim_quotas[N];
  std::copy_n(shape_.dimensions_, N, dim_quotas);
  for (size_t i = 0; i < cumul_index; ++i) {
    size_t index = pEvaluateIndex(dim_quotas);
    size_t t1_index = tensor_1.pEvaluateIndex(dim_quotas);
    size_t t2_index = tensor_2.pEvaluateIndex(dim_quotas);
    fn(&data_[index], &tensor_1.data_[t1_index], &tensor_2.data_[t2_index]);
    pUpdateQuotas(dim_quotas);
  }
}

template <typename T, size_t N>
void Tensor<T, N>::pInitializeStrides()
{
  size_t accumulator = 1;
  for (size_t i = 0; i < N; ++i) {
    strides_[N - i - 1] = accumulator;
    accumulator *= shape_.dimensions_[N - i - 1];
  }
}

// private constructor
template <typename T, size_t N>
Tensor<T, N>::Tensor(size_t const *dimensions, size_t const *strides, T *data, std::shared_ptr<T> &&ref)
  : shape_(Shape<N>(dimensions, 0)), data_(data), ref_(std::move(ref))
{
  std::copy_n(strides, N, strides_);
}

/* -------------------------- Expressions -------------------------- */

template <typename X, typename Y, size_t M>
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

/** Elementwise Scalar-Tensor addition. Returns a Tensor with shape 
 *  `tensor`, where each element of `tensor` is incremeneted with `scalar`.
 */
template <typename X, size_t N,
          typename = typename std::enable_if<N != 0>::type>
Tensor<X, N> operator+(Tensor<X, N> const &tensor, X const &scalar)
{
  Tensor<X, N> result {tensor.shape()};
  std::function<void(X*, X*)> set_vals = [&scalar](X *lhs, X *rhs) -> void {
    *lhs = scalar + *rhs;
  };
  result.pUnaryMap(tensor, set_vals);
  return result;
}

/** See operator+(Tensor<X, N> const &tensor, X const &scalar) */
template <typename X, size_t N,
          typename = typename std::enable_if<N != 0>::type>
Tensor<X, N> operator+(X const &scalar, Tensor<X, N> const &tensor)
{
  return tensor + scalar;
}

template <typename X, typename Y, size_t M>
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

/** Elementwise Scalar-Tensor subtraction. Returns a Tensor with shape 
 *  `tensor`, where each element of `tensor` is subtracted by `scalar`.
 */
template <typename X, size_t M,
          typename = typename std::enable_if<M != 0>::type>
Tensor<X, M> operator-(Tensor<X, M> const &tensor, X const &scalar)
{
  Tensor<X, M> result {tensor.shape()};
  std::function<void(X*, X*)> set_vals = [&scalar](X *lhs, X *rhs) -> void {
    *lhs = *rhs - scalar;
  };
  result.pUnaryMap(tensor, set_vals);
  return result;
}

/** See operator-(Tensor<X, N> const &tensor, X const &scalar) */
template <typename X, size_t M,
          typename = typename std::enable_if<M != 0>::type>
Tensor<X, M> operator-(X const &scalar, Tensor<X, M> const &tensor)
{
  Tensor<X, M> result {tensor.shape()};
  std::function<void(X*, X*)> set_vals = [&scalar](X *lhs, X *rhs) -> void {
    *lhs = scalar - *rhs;
  };
  result.pUnaryMap(tensor, set_vals);
  return result;
}

template <typename X, typename Y, size_t M1, size_t M2>
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
  size_t cumul_index_1 = tensor_1.shape_.index_product() / tensor_1.shape_.dimensions_[M1 - 1];
  size_t cumul_index_2 = tensor_2.shape_.index_product() / tensor_2.shape_.dimensions_[0];
  size_t dim_quotas_1[M1 - 1], dim_quotas_2[M2 - 1];
  std::copy_n(tensor_1.shape_.dimensions_, M1 - 1, dim_quotas_1);
  std::copy_n(tensor_2.shape_.dimensions_ + 1, M2 - 1, dim_quotas_2);
  size_t index = 0;
  for (size_t i1 = 0; i1 < cumul_index_1; ++i1) {
    for (size_t i2 = 0; i2 < cumul_index_2; ++i2) {
      size_t t1_index = tensor_1.pEvaluateIndex(dim_quotas_1);
      size_t t2_index = tensor_2.pEvaluateIndex(dim_quotas_2, 1);
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

/** Elementwise Scalar-Tensor multiplication. Returns a Tensor with shape 
 *  `tensor`, where each element of `tensor` is multiplied by `scalar`.
 */
template <typename X, size_t M, 
          typename = typename std::enable_if<M != 0>::type>
Tensor<X, M> operator*(Tensor<X, M> const &tensor, X const &scalar)
{
  Tensor<X, M> result {tensor.shape()};
  std::function<void(X*, X*)> set_vals = [&scalar](X *lhs, X *rhs) -> void {
    *lhs = scalar * *rhs;
  };
  result.pUnaryMap(tensor, set_vals);
  return result;
}

/** See operator*(Tensor<X, N> const &tensor, X const &scalar) */
template <typename X, size_t N,
          typename = typename std::enable_if<N != 0>::type>
Tensor<X, N> operator*(X const &scalar, Tensor<X, N> const &tensor)
{
  return tensor * scalar;
}

template <typename T, size_t N>
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
// template <typename T, size_t N1, size_t N2>
// Tensor<T, N1> Resize(Tensor<T, N2> const &tensor) {}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::copy() const
{
  return Tensor<T, N>(*this);
}

template <typename T, size_t N>
typename Tensor<T, N>::Proxy Tensor<T, N>::ref() 
{
  return Proxy(*this);
}

/* ------------------------------- Iterator ----------------------------- */

template <typename T, size_t N>
Tensor<T, N>::Iterator::Iterator(Tensor<T, N + 1> const &tensor, size_t index)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
}

template <typename T, size_t N>
Tensor<T, N>::Iterator::Iterator(Iterator const &it)
  : shape_(it.shape_), data_(it.data_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N>
Tensor<T, N>::Iterator::Iterator(Iterator &&it)
  : shape_(it.shape_), data_(it.data_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::Iterator::operator*()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::Iterator::operator->()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
typename Tensor<T, N>::Iterator Tensor<T, N>::Iterator::operator++(int)
{
  Tensor<T, N>::Iterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N>::Iterator &Tensor<T, N>::Iterator::operator++()
{
  data_ += stride_;
  return *this;
}

template <typename T, size_t N>
typename Tensor<T, N>::Iterator Tensor<T, N>::Iterator::operator--(int)
{
  Tensor<T, N>::Iterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N>::Iterator &Tensor<T, N>::Iterator::operator--()
{
  data_ -= stride_;
  return *this;
}

/* ----------------------------- ConstIterator --------------------------- */

template <typename T, size_t N>
Tensor<T, N>::ConstIterator::ConstIterator(Tensor<T, N + 1> const &tensor, size_t index)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
}

template <typename T, size_t N>
Tensor<T, N>::ConstIterator::ConstIterator(ConstIterator const &it)
  : shape_(it.shape_), data_(it.data_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N>
Tensor<T, N>::ConstIterator::ConstIterator(ConstIterator &&it)
  : shape_(it.shape_), data_(it.data_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N>
Tensor<T, N> const Tensor<T, N>::ConstIterator::operator*()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
Tensor<T, N> const Tensor<T, N>::ConstIterator::operator->()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
typename Tensor<T, N>::ConstIterator Tensor<T, N>::ConstIterator::operator++(int)
{
  Tensor<T, N>::ConstIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N>::ConstIterator &Tensor<T, N>::ConstIterator::operator++()
{
  data_ += stride_;
  return *this;
}

template <typename T, size_t N>
typename Tensor<T, N>::ConstIterator Tensor<T, N>::ConstIterator::operator--(int)
{
  Tensor<T, N>::ConstIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N>::ConstIterator &Tensor<T, N>::ConstIterator::operator--()
{
  data_ -= stride_;
  return *this;
}

/* ---------------------------- ReverseIterator -------------------------- */

template <typename T, size_t N>
Tensor<T, N>::ReverseIterator::ReverseIterator(Tensor<T, N + 1> const &tensor, size_t index)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
  data_ += stride_ * (tensor.shape_.dimensions_[index] - 1);
}

template <typename T, size_t N>
Tensor<T, N>::ReverseIterator::ReverseIterator(ReverseIterator const &it)
  : shape_(it.shape_), data_(it.data_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N>
Tensor<T, N>::ReverseIterator::ReverseIterator(ReverseIterator &&it)
  : shape_(it.shape_), data_(it.data_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::ReverseIterator::operator*()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::ReverseIterator::operator->()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
typename Tensor<T, N>::ReverseIterator Tensor<T, N>::ReverseIterator::operator++(int)
{
  Tensor<T, N>::ReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N>::ReverseIterator &Tensor<T, N>::ReverseIterator::operator++()
{
  data_ -= stride_;
  return *this;
}

template <typename T, size_t N>
typename Tensor<T, N>::ReverseIterator Tensor<T, N>::ReverseIterator::operator--(int)
{
  Tensor<T, N>::ReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N>::ReverseIterator &Tensor<T, N>::ReverseIterator::operator--()
{
  data_ += stride_;
  return *this;
}

/* ------------------------- ConstReverseIterator ----------------------- */

template <typename T, size_t N>
Tensor<T, N>::ConstReverseIterator::ConstReverseIterator(Tensor<T, N + 1> const &tensor, size_t index)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[index])
{
  assert(index < N + 1 && "This should throw earlier");
  std::copy_n(tensor.shape_.dimensions_, index, shape_.dimensions_);
  std::copy_n(tensor.shape_.dimensions_ + index + 1, N - index, shape_.dimensions_ + index);
  std::copy_n(tensor.strides_, index, strides_);
  std::copy_n(tensor.strides_+ index + 1, N - index, strides_ + index);
  data_ += stride_ * (tensor.shape_.dimensions_[index] - 1);
}

template <typename T, size_t N>
Tensor<T, N>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator const &it)
  : shape_(it.shape_), data_(it.data_), ref_(it.ref_), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N>
Tensor<T, N>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator &&it)
  : shape_(it.shape_), data_(it.data_), ref_(std::move(it.ref_)), stride_(it.stride_)
{
  std::copy_n(it.strides_, N, strides_);
}

template <typename T, size_t N>
Tensor<T, N> const Tensor<T, N>::ConstReverseIterator::operator*()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
Tensor<T, N> const Tensor<T, N>::ConstReverseIterator::operator->()
{
  return Tensor<T, N>(shape_.dimensions_, strides_, data_, 
      std::shared_ptr<T>(ref_));
}

template <typename T, size_t N>
typename Tensor<T, N>::ConstReverseIterator Tensor<T, N>::ConstReverseIterator::operator++(int)
{
  Tensor<T, N>::ConstReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N>::ConstReverseIterator &Tensor<T, N>::ConstReverseIterator::operator++()
{
  data_ -= stride_;
  return *this;
}

template <typename T, size_t N>
typename Tensor<T, N>::ConstReverseIterator Tensor<T, N>::ConstReverseIterator::operator--(int)
{
  Tensor<T, N>::ConstReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N>::ConstReverseIterator &Tensor<T, N>::ConstReverseIterator::operator--()
{
  data_ += stride_;
  return *this;
}

/* ----------------------- Iterator Construction ----------------------- */

template <typename T, size_t N>
typename Tensor<T, N - 1>::Iterator Tensor<T, N>::begin(size_t index)
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::begin(size_t)"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::begin(size_t)"));
  --index;
  return typename Tensor<T, N - 1>::Iterator(*this, index);
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::Iterator Tensor<T, N>::end(size_t index)
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::end(size_t)"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::end(size_t)"));
  --index;
  typename Tensor<T, N - 1>::Iterator it{*this, index};
  it.data_ += strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::Iterator Tensor<T, N>::begin() 
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->begin(1); 
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::Iterator Tensor<T, N>::end() 
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->end(1); 
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ConstIterator Tensor<T, N>::cbegin(size_t index) const
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::begin(size_t)"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::begin(size_t)"));
  --index;
  return typename Tensor<T, N - 1>::ConstIterator(*this, index);
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ConstIterator Tensor<T, N>::cend(size_t index) const
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::end(size_t)"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::end(size_t)"));
  --index;
  typename Tensor<T, N - 1>::ConstIterator it{*this, index};
  it.data_ += strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ConstIterator Tensor<T, N>::cbegin() const
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->cbegin(1); 
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ConstIterator Tensor<T, N>::cend() const
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->cend(1); 
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ReverseIterator Tensor<T, N>::rbegin(size_t index)
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::rbegin(size_t)"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::rbegin(size_t)"));
  --index;
  return typename Tensor<T, N - 1>::ReverseIterator(*this, index);
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ReverseIterator Tensor<T, N>::rend(size_t index)
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::rend(size_t)"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::rend(size_t)"));
  --index;
  typename Tensor<T, N - 1>::ReverseIterator it{*this, index};
  it.data_ -= strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ReverseIterator Tensor<T, N>::rbegin() 
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->rbegin(1); 
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ReverseIterator Tensor<T, N>::rend() 
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->rend(1); 
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ConstReverseIterator Tensor<T, N>::crbegin(size_t index) const
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::rbegin(size_t)"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::rbegin(size_t)"));
  --index;
  return typename Tensor<T, N - 1>::ConstReverseIterator(*this, index);
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ConstReverseIterator Tensor<T, N>::crend(size_t index) const
{
  if (!index) throw std::logic_error(ZERO_INDEX("Tensor<T, N>::rend(size_t)"));
  if (index > N) throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor<T, N>::rend(size_t)"));
  --index;
  typename Tensor<T, N - 1>::ConstReverseIterator it{*this, index};
  it.data_ -= strides_[index] * shape_.dimensions_[index];
  return it;
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ConstReverseIterator Tensor<T, N>::crbegin() const
{
  // Probably a mistake if using this on a non-vector 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->crbegin(1); 
}

template <typename T, size_t N>
typename Tensor<T, N - 1>::ConstReverseIterator Tensor<T, N>::crend() const
{ 
  static_assert(N == 1, BEGIN_ON_NON_VECTOR);
  return this->crend(1); 
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

  template <typename X, size_t M> friend class Tensor;
  template <typename LHSType, typename RHSType> friend class BinaryAdd;
  template <typename LHSType, typename RHSType> friend class BinarySub;
  template <typename LHSType, typename RHSType> friend class BinaryMul;

  /* ------------------ Constructors ------------------ */

  explicit Shape() {}
  Shape(Shape<0> const&) {}

  /* -------------------- Getters --------------------- */

  constexpr static size_t rank() { return 0; }

  /* -------------------- Equality -------------------- */

  // Equality only compares against dimensions, not step size
  bool operator==(Shape<0> const&) const noexcept { return true; }
  bool operator!=(Shape<0> const&) const noexcept { return false; }
  template <size_t M>
  bool operator==(Shape<M> const&) const noexcept { return false; }
  template <size_t M>
  bool operator!=(Shape<M> const&) const noexcept { return true; }

  /* ---------------- Print ----------------- */

  template <typename X, size_t M>
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

  template <typename X, size_t M> friend class Tensor;

  /* ----------- Proxy Objects ------------ */

  /**
   * Proxy Tensor Object used for building tensors from reference
   * This is used only to differentiate proxy tensor Construction
   */

  class Proxy { /*@Proxy<T,0>*/
  public:
    template <typename U, size_t N> friend class Tensor;
    Proxy() = delete;
  private:
    Proxy(Tensor<T, 0> const &tensor): tensor_(tensor) {}
    Proxy(Proxy const &proxy): tensor_(proxy.tensor_) {} 
    Tensor<T, 0> const &tensor_;
  }; 

  /* -------------- Constructors -------------- */

  Tensor();
  explicit Tensor(value_type &&val);
  explicit Tensor(Shape<0>);
  Tensor(Tensor<T, 0> const &tensor);
  Tensor(Tensor<T, 0> &&tensor);
  Tensor(Tensor<T, 0>::Proxy const &proxy);
  template <typename NodeType>
  Tensor(Expression<NodeType> const& expression);

  /* ------------- Assignment ------------- */

  Tensor<T, 0> &operator=(Tensor<T, 0> const &tensor);
  template <typename X> Tensor<T, 0> &operator=(Tensor<X, 0> const &tensor);

  /* -------------- Getters -------------- */

  constexpr static size_t rank() { return 0; }
  Shape<0> shape() const noexcept { return shape_; }
  value_type &operator()() { return *data_; }
  value_type const &operator()() const { return *data_; }
  Tensor *operator->() { return this; }
  Tensor const *operator->() const { return this; }

  /* -------------- Setters -------------- */

  template <typename X,
            typename = typename std::enable_if<std::is_convertible<
            typename std::remove_reference<T>::type,
            typename std::remove_reference<X>::type>::value>::type>
  Tensor<T, 0> &operator=(X&& elem);

  /* --------------- Print --------------- */

  template <typename X, size_t M>
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

  /* ---------------- Iterator -------------- */

  class Iterator { /*@Iterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M> friend class Tensor;

    /* --------------- Constructors --------------- */

    Iterator(Iterator const &it);
    Iterator(Iterator &&it);
    Tensor<T, 0> operator*();
    Tensor<T, 0> const operator*() const;
    Tensor<T, 0> operator->();
    Tensor<T, 0> const operator->() const;
    Iterator operator++(int);
    Iterator &operator++();
    Iterator operator--(int);
    Iterator &operator--();
    bool operator==(Iterator const &it) const { return (it.data_ == this->data_); }
    bool operator!=(Iterator const &it) const { return !(it == *this); }
  private:
    Iterator(Tensor<T, 1> const &tensor, size_t);

    value_type *data_;
    std::shared_ptr<T> ref_;
    size_t stride_;
  };

  /* -------------- ConstIterator ------------ */

  class ConstIterator { /*@ConstIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M> friend class Tensor;

    /* --------------- Constructors --------------- */

    ConstIterator(ConstIterator const &it);
    ConstIterator(ConstIterator &&it);
    Tensor<T, 0> const operator*();
    Tensor<T, 0> const operator->();
    ConstIterator operator++(int);
    ConstIterator &operator++();
    ConstIterator operator--(int);
    ConstIterator &operator--();
    bool operator==(ConstIterator const &it) const { return (it.data_ == this->data_); }
    bool operator!=(ConstIterator const &it) const { return !(it == *this); }
  private:
    ConstIterator(Tensor<T, 1> const &tensor, size_t);

    value_type *data_;
    std::shared_ptr<T> ref_;
    size_t stride_;
  };

  /* ------------- ReverseIterator ----------- */

  class ReverseIterator { /*@ReverseIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M> friend class Tensor;

    /* --------------- Constructors --------------- */

    ReverseIterator(ReverseIterator const &it);
    ReverseIterator(ReverseIterator &&it);
    Tensor<T, 0> operator*();
    Tensor<T, 0> operator->();
    ReverseIterator operator++(int);
    ReverseIterator &operator++();
    ReverseIterator operator--(int);
    ReverseIterator &operator--();
    bool operator==(ReverseIterator const &it) const { return (it.data_ == this->data_); }
    bool operator!=(ReverseIterator const &it) const { return !(it == *this); }
  private:
    ReverseIterator(Tensor<T, 1> const &tensor, size_t);

    value_type *data_;
    std::shared_ptr<T> ref_;
    size_t stride_;
  };

  /* ----------- ConstReverseIterator --------- */

  class ConstReverseIterator { /*@ReverseIterator<T, 0>*/
  public:
    /* -------------- Friend Classes -------------- */

    template <typename U, size_t M> friend class Tensor;

    /* --------------- Constructors --------------- */

    ConstReverseIterator(ConstReverseIterator const &it);
    ConstReverseIterator(ConstReverseIterator &&it);
    Tensor<T, 0> const operator*();
    Tensor<T, 0> const operator->();
    ConstReverseIterator operator++(int);
    ConstReverseIterator &operator++();
    ConstReverseIterator operator--(int);
    ConstReverseIterator &operator--();
    bool operator==(ConstReverseIterator const &it) const { return (it.data_ == this->data_); }
    bool operator!=(ConstReverseIterator const &it) const { return !(it == *this); }
  private:
    ConstReverseIterator(Tensor<T, 1> const &tensor, size_t);

    value_type *data_;
    std::shared_ptr<T> ref_;
    size_t stride_;
  };

  /* ------------- Utility Functions ------------ */

  Tensor<T, 0> copy() const;
  Tensor<T, 0>::Proxy ref();

private:

  /* ------------------- Data ------------------- */

  Shape<0> shape_;
  value_type * data_;
  std::shared_ptr<T> ref_;

  /* ------------------ Utility ----------------- */

  Tensor(size_t const *, size_t const *, T *data, std::shared_ptr<T> &&ref_);

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
  ref_ = std::shared_ptr<T>(data_, _ARRAY_DELETER(T));
}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0> const &tensor): shape_(Shape<0>()), data_(new T[1]()), 
  ref_(data_, _ARRAY_DELETER(T))
{
  *data_ = *tensor.data_;
}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0> &&tensor): shape_(Shape<0>()), data_(tensor.data_),
  ref_(std::move(tensor.ref_))
{
  tensor.data_ = nullptr;
}

template <typename T>
Tensor<T, 0>::Tensor(Tensor<T, 0>::Proxy const &proxy)
  : shape_(Shape<0>()), data_(proxy.tensor_.data_), ref_(proxy.tensor_.ref_)
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
Tensor<T, 0>::Tensor(size_t const *, size_t const *, T *data, std::shared_ptr<T> &&ref)
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

template <typename T>
typename Tensor<T, 0>::Proxy Tensor<T, 0>::ref() 
{
  return Proxy(*this);
}

/* ---------------------- Iterators ------------------------- */

template <typename T>
Tensor<T, 0>::Iterator::Iterator(Tensor<T, 1> const &tensor, size_t)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{}

template <typename T>
Tensor<T, 0>::Iterator::Iterator(Iterator const &it)
  : data_(it.data_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T>
Tensor<T, 0>::Iterator::Iterator(Iterator &&it)
  : data_(it.data_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T>
Tensor<T, 0> Tensor<T, 0>::Iterator::operator*()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
Tensor<T, 0> Tensor<T, 0>::Iterator::operator->()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
typename Tensor<T, 0>::Iterator Tensor<T, 0>::Iterator::operator++(int)
{
  Tensor<T, 0>::Iterator it {*this};
  ++(*this);
  return it;
}

template <typename T>
typename Tensor<T, 0>::Iterator &Tensor<T, 0>::Iterator::operator++()
{
  data_ += stride_;
  return *this;
}

template <typename T>
typename Tensor<T, 0>::Iterator Tensor<T, 0>::Iterator::operator--(int)
{
  Tensor<T, 0>::Iterator it {*this};
  --(*this);
  return it;
}

template <typename T>
typename Tensor<T, 0>::Iterator &Tensor<T, 0>::Iterator::operator--()
{
  data_ -= stride_;
  return *this;
}

/* ------------------- ConstIterators ---------------------- */

template <typename T>
Tensor<T, 0>::ConstIterator::ConstIterator(Tensor<T, 1> const &tensor, size_t)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{}

template <typename T>
Tensor<T, 0>::ConstIterator::ConstIterator(ConstIterator const &it)
  : data_(it.data_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T>
Tensor<T, 0>::ConstIterator::ConstIterator(ConstIterator &&it)
  : data_(it.data_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T>
Tensor<T, 0> const Tensor<T, 0>::ConstIterator::operator*()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
Tensor<T, 0> const Tensor<T, 0>::ConstIterator::operator->()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
typename Tensor<T, 0>::ConstIterator Tensor<T, 0>::ConstIterator::operator++(int)
{
  Tensor<T, 0>::ConstIterator it {*this};
  ++(*this);
  return it;
}

template <typename T>
typename Tensor<T, 0>::ConstIterator &Tensor<T, 0>::ConstIterator::operator++()
{
  data_ += stride_;
  return *this;
}

template <typename T>
typename Tensor<T, 0>::ConstIterator Tensor<T, 0>::ConstIterator::operator--(int)
{
  Tensor<T, 0>::ConstIterator it {*this};
  --(*this);
  return it;
}

template <typename T>
typename Tensor<T, 0>::ConstIterator &Tensor<T, 0>::ConstIterator::operator--()
{
  data_ -= stride_;
  return *this;
}

/* -------------------- ReverseIterator ---------------------- */

template <typename T>
Tensor<T, 0>::ReverseIterator::ReverseIterator(Tensor<T, 1> const &tensor, size_t)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{
  data_ += stride_ * (tensor.shape_.dimensions_[0] - 1);
}

template <typename T>
Tensor<T, 0>::ReverseIterator::ReverseIterator(ReverseIterator const &it)
  : data_(it.data_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T>
Tensor<T, 0>::ReverseIterator::ReverseIterator(ReverseIterator &&it)
  : data_(it.data_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T>
Tensor<T, 0> Tensor<T, 0>::ReverseIterator::operator*()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
Tensor<T, 0> Tensor<T, 0>::ReverseIterator::operator->()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
typename Tensor<T, 0>::ReverseIterator Tensor<T, 0>::ReverseIterator::operator++(int)
{
  Tensor<T, 0>::ReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T>
typename Tensor<T, 0>::ReverseIterator &Tensor<T, 0>::ReverseIterator::operator++()
{
  data_ -= stride_;
  return *this;
}

template <typename T>
typename Tensor<T, 0>::ReverseIterator Tensor<T, 0>::ReverseIterator::operator--(int)
{
  Tensor<T, 0>::ReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T>
typename Tensor<T, 0>::ReverseIterator &Tensor<T, 0>::ReverseIterator::operator--()
{
  data_ += stride_;
  return *this;
}

/* ----------------- ConstReverseIterator ------------------- */

template <typename T>
Tensor<T, 0>::ConstReverseIterator::ConstReverseIterator(Tensor<T, 1> const &tensor, size_t)
  : data_(tensor.data_), ref_(tensor.ref_), stride_(tensor.strides_[0]) 
{
  data_ += stride_ * (tensor.shape_.dimensions_[0] - 1);
}

template <typename T>
Tensor<T, 0>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator const &it)
  : data_(it.data_), ref_(it.ref_), stride_(it.stride_) {}

template <typename T>
Tensor<T, 0>::ConstReverseIterator::ConstReverseIterator(ConstReverseIterator &&it)
  : data_(it.data_), ref_(std::move(it.ref_)), stride_(it.stride_) {}

template <typename T>
Tensor<T, 0> const Tensor<T, 0>::ConstReverseIterator::operator*()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
Tensor<T, 0> const Tensor<T, 0>::ConstReverseIterator::operator->()
{
  return Tensor<T, 0>(nullptr, nullptr, data_, std::shared_ptr<T>(ref_));
}

template <typename T>
typename Tensor<T, 0>::ConstReverseIterator Tensor<T, 0>::ConstReverseIterator::operator++(int)
{
  Tensor<T, 0>::ConstReverseIterator it {*this};
  ++(*this);
  return it;
}

template <typename T>
typename Tensor<T, 0>::ConstReverseIterator &Tensor<T, 0>::ConstReverseIterator::operator++()
{
  data_ -= stride_;
  return *this;
}

template <typename T>
typename Tensor<T, 0>::ConstReverseIterator Tensor<T, 0>::ConstReverseIterator::operator--(int)
{
  Tensor<T, 0>::ConstReverseIterator it {*this};
  --(*this);
  return it;
}

template <typename T>
typename Tensor<T, 0>::ConstReverseIterator &Tensor<T, 0>::ConstReverseIterator::operator--()
{
  data_ += stride_;
  return *this;
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

  constexpr static size_t rank() { return LHSType::rank(); }
  size_t dimension(size_t index) const { return lhs_.dimension(index); }
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

  constexpr static size_t rank() { return LHSType::rank(); }
  size_t dimension(size_t index) const { return lhs_.dimension(index); }
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
class BinaryMul: public Expression<BinaryMul<LHSType, RHSType>> { /*@BinaryMul*/
public:

  /* ---------------- typedefs --------------- */

  typedef typename LHSType::value_type value_type;
  typedef BinaryMul                    self_type;

  /* -------------- Constructors -------------- */

  BinaryMul(LHSType const &lhs, RHSType const &rhs);

  /* ---------------- Getters ----------------- */

  constexpr static size_t rank() { return LHSType::rank() + RHSType::rank() - 2; }
  size_t dimension(size_t index) const;
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
#undef ZERO_ELEMENT
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
