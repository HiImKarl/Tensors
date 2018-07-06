[1mdiff --git a/README.md b/README.md[m
[1mindex 5c63719..3d0c28e 100644[m
[1m--- a/README.md[m
[1m+++ b/README.md[m
[36m@@ -1,7 +1,17 @@[m
 [m
 # Tensor[m
 [m
[31m-C++11 Tensor Template header-only library, with compile time rank. This library makes heavy use of template expressions and meta-programming to optimize Tensor mathematics and provide simple syntax. [m
[32m+[m[32m## Description[m
[32m+[m[32mThis is a multi-dimensional array library, where array rank must be known at compile time. The entire library is contained in a single header file. This library makes heavy use of template expressions and meta-programming to optimize Tensor mathematics and provide simple syntax.[m[41m [m
[32m+[m
[32m+[m[32m## Table of Contents[m
[32m+[m[32m   * [Tensor](#tensor)[m
[32m+[m[32m      * [Getting Started](#getting-started)[m
[32m+[m[32m         * [Prerequisites](#prerequisites)[m
[32m+[m[32m         * [Installation](#installation)[m
[32m+[m[32m         * [Running the tests](#running-the-tests)[m
[32m+[m[32m      * [Contributing](#contributing)[m
[32m+[m[32m      * [Libraries Used](#libraries-used)[m
 [m
 ## Getting Started[m
 [m
[36m@@ -19,7 +29,61 @@[m [mgit clone git@github.com:HiImKarl/Tensors.git[m
 ```[m
 And then copy include/tensor.hh into your project. [m
 [m
[31m-### Running the tests[m
[32m+[m[32m### Usage[m
[32m+[m
[32m+[m[32m#### Creating Tensors[m
[32m+[m
[32m+[m[32mCreating a Tensor is simple:[m
[32m+[m[32m```C++[m
[32m+[m[32m// This will create a 5x3x5 array of integers. The elements are **not** zero initialized.[m
[32m+[m[32mTensor<int, 3> t0({5, 3, 5});[m
[32m+[m
[32m+[m[32m// This will create a 5x3x5 array of integers, initialized to the value -1.[m
[32m+[m[32mTensor<int, 3> t1({5, 3, 5}, -1);[m
[32m+[m
[32m+[m[32m// This will create a 5x3x5 array of integers, with each value initialized by int rand() (from C stdlib)[m
[32m+[m[32m// Note: this will call rand() 65 times.[m
[32m+[m[32mTensor<int, 3> t2(5, 3, 5}, &rand);[m
[32m+[m
[32m+[m[32m// Tensors can be filled with the elements of other containers[m
[32m+[m[32mvector<int> ones(65, 1); // 65 element vector of ones[m
[32m+[m[32mTensor<int, 3> t3({5, 3, 5});[m
[32m+[m[32mFill(t3, ones.begin(), ones.end()); // t3 now contains ones[m
[32m+[m
[32m+[m[32m// Copy constructor allocates and copies over elements[m
[32m+[m[32mTensor<int, 3> t4 = t3; // t4 is a 5x3x5 tensor of ones[m
[32m+[m
[32m+[m[32m// Move constructor takes ownership of the underlying data[m
[32m+[m[32mTensor<int, 3> t5 = move(t4); // t5 is 5x3x5 tensor of ones, t4 is destroyed[m
[32m+[m
[32m+[m[32m// "Reference constructor" (not a typical C++ idiom) shares[m[41m [m
[32m+[m[32m// ownership of the data[m
[32m+[m[32mTensor<int, 3> t6 = t5.ref() // t6 shares the same 5x3x5 data array as t4[m
[32m+[m
[32m+[m[32m// Any changes made to t6 will also be made to t5[m
[32m+[m[32mFill(t6, 0); // t5 is now also filled with zeros[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m#### Indexing Tensors[m
[32m+[m
[32m+[m
[32m+[m
[32m+[m[32m#### Tensor mathematics[m
[32m+[m
[32m+[m[32m``` C++[m
[32m+[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Documentation[m
[32m+[m
[32m+[m[32mBuilding the documentation requires [Doxygen](https://github.com/doxygen/doxygen) installed. Simplying running Doxygen will create the html/markdowm documentation (which will be dumped in a folder called `Documentation`):[m
[32m+[m
[32m+[m[32m```[m
[32m+[m[32m# in the root folder of the project[m
[32m+[m[32mdoxygen[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Running the tests/benchmarks[m
 [m
 To build the tests:[m
 ```[m
[1mdiff --git a/include/tensor.hh b/include/tensor.hh[m
[1mindex 379a523..61f058c 100644[m
[1m--- a/include/tensor.hh[m
[1m+++ b/include/tensor.hh[m
[36m@@ -158,49 +158,86 @@[m [mstruct IsScalar<Tensor<T, N>> { static bool const value = false; };[m
  *  a Tensor, o.w. `value` is a reference to Tensor `val`[m
  */[m
 template <typename T>[m
[31m-struct ValueToTensor {[m
[31m-  ValueToTensor(T &&val): value(std::forward<T>(val)) {}[m
[31m-  Tensor<T, 0> value;[m
[32m+[m[32mstruct ValueAsTensor {[m
[32m+[m[32m  ValueAsTensor(T &&val): value(std::forward<T>(val)) {}[m
[32m+[m[32m  T value;[m
[32m+[m[32m  T &operator()() { return value; }[m
 };[m
 [m
[31m-/** Tensor specialization of ValueToTensor, `value` is a [m
[32m+[m[32m/** Tensor specialization of ValueAsTensor, `value` is a[m[41m [m
  *  const reference to the provided Tensor[m
  */[m
 template <>[m
 template <typename T, size_t N>[m
[31m-struct ValueToTensor<Tensor<T, N>> {[m
[31m-  ValueToTensor(Tensor<T, N> const &val): value(val) {}[m
[32m+[m[32mstruct ValueAsTensor<Tensor<T, N>> {[m
[32m+[m[32m  ValueAsTensor(Tensor<T, N> const &val): value(val) {}[m
   Tensor<T, N> const &value;[m
[32m+[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N != sizeof...(Indices)>::type>[m
[32m+[m[32m  Tensor<T, N - sizeof...(Indices)> operator()(Indices... indices) { return value(indices...); }[m
[32m+[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N == sizeof...(Indices)>::type>[m
[32m+[m[32m  T const &operator()(Indices... indices) { return value(indices...); }[m
 };[m
 [m
[31m-/** BinaryAdd specialization of ValueToTensor, `value` is a [m
[32m+[m[32m/** BinaryAdd specialization of ValueAsTensor, `value` is a[m[41m [m
  *  const reference to the provided binary expression[m
  */[m
 template <>[m
 template <typename LHS, typename RHS>[m
[31m-struct ValueToTensor<BinaryAdd<LHS, RHS>> {[m
[31m-  ValueToTensor(BinaryAdd<LHS, RHS> const &val): value(val) {}[m
[32m+[m[32mstruct ValueAsTensor<BinaryAdd<LHS, RHS>> {[m
[32m+[m[32m  ValueAsTensor(BinaryAdd<LHS, RHS> const &val): value(val) {}[m
   BinaryAdd<LHS, RHS> const &value;[m
[32m+[m[32m  typedef typename LHS::value_type   value_type;[m
[32m+[m[32m  constexpr static size_t N =      LHS::rank();[m
[32m+[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N != sizeof...(Indices)>::type>[m
[32m+[m[32m  Tensor<value_type, N - sizeof...(Indices)> operator()(Indices... indices) { return value(indices...); }[m
[32m+[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N == sizeof...(Indices)>::type>[m
[32m+[m[32m  value_type const &operator()(Indices... indices) { return value(indices...); }[m
 };[m
 [m
[31m-/** BinarySub specialization of ValueToTensor, `value` is a [m
[32m+[m[32m/** BinarySub specialization of ValueAsTensor, `value` is a[m[41m [m
  *  const reference to the provided binary expression[m
  */[m
 template <>[m
 template <typename LHS, typename RHS>[m
[31m-struct ValueToTensor<BinarySub<LHS, RHS>> {[m
[31m-  ValueToTensor(BinarySub<LHS, RHS> const &val): value(val) {}[m
[32m+[m[32mstruct ValueAsTensor<BinarySub<LHS, RHS>> {[m
[32m+[m[32m  ValueAsTensor(BinarySub<LHS, RHS> const &val): value(val) {}[m
   BinarySub<LHS, RHS> const &value;[m
[32m+[m[32m  typedef typename LHS::value_type   value_type;[m
[32m+[m[32m  constexpr static size_t N =      LHS::rank();[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N != sizeof...(Indices)>::type>[m
[32m+[m[32m  Tensor<value_type, N - sizeof...(Indices)> operator()(Indices... indices) { return value(indices...); }[m
[32m+[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N == sizeof...(Indices)>::type>[m
[32m+[m[32m  value_type const &operator()(Indices... indices) { return value(indices...); }[m
 };[m
 [m
[31m-/** BinaryMul specialization of ValueToTensor, `value` is a [m
[32m+[m[32m/** BinaryMul specialization of ValueAsTensor, `value` is a[m[41m [m
  *  const reference to the provided binary expression[m
  */[m
 template <>[m
 template <typename LHS, typename RHS>[m
[31m-struct ValueToTensor<BinaryMul<LHS, RHS>> {[m
[31m-  ValueToTensor(BinaryMul<LHS, RHS> const &val): value(val) {}[m
[32m+[m[32mstruct ValueAsTensor<BinaryMul<LHS, RHS>> {[m
[32m+[m[32m  ValueAsTensor(BinaryMul<LHS, RHS> const &val): value(val) {}[m
   BinaryMul<LHS, RHS> const &value;[m
[32m+[m[32m  typedef typename LHS::value_type   value_type;[m
[32m+[m[32m  constexpr static size_t N =      LHS::rank() + RHS::rank() - 2;[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N != sizeof...(Indices)>::type>[m
[32m+[m[32m  Tensor<value_type, N - sizeof...(Indices)> operator()(Indices... indices) { return value(indices...); }[m
[32m+[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N == sizeof...(Indices)>::type>[m
[32m+[m[32m  value_type const &operator()(Indices... indices) { return value(indices...); }[m
 };[m
 [m
 /* ---------------------------------------------------------- */[m
[36m@@ -521,22 +558,37 @@[m [mpublic:[m
 [m
   /* ------------------ Access To Data ----------------- */[m
 [m
[32m+[m[32m  template <typename... Indices>[m
[32m+[m[32m  Tensor<T, N - sizeof...(Indices)> at(Indices... args);[m
[32m+[m
   /** Returns the resulting tensor by applying left to right index expansion of[m
    *  the provided arguments. I.e. calling `tensor(1, 2)` on a rank 4 tensor is[m
    *  equivalent to `tensor(1, 2, :, :)`. Throws std::logic_error if any of the [m
    *  indices are out bounds. Note: indexing starts at 1.[m
    */[m
[31m-  template <typename... Indices>[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N != sizeof...(Indices)>::type>[m
   Tensor<T, N - sizeof...(Indices)> operator()(Indices... args);[m
 [m
[31m-  /** See operator()[m
[31m-   */[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N == sizeof...(Indices)>::type>[m
[32m+[m[32m  T &operator()(Indices... args);[m
[32m+[m
   template <typename... Indices>[m
[32m+[m[32m  Tensor<T, N - sizeof...(Indices)> const at(Indices... args) const;[m
[32m+[m
[32m+[m[32m  /** See operator() */[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N != sizeof...(Indices)>::type>[m
   Tensor<T, N - sizeof...(Indices)> const operator()(Indices... args) const;[m
 [m
[31m-  /** See operator()[m
[31m-   */[m
[31m-  template <size_t M>[m
[32m+[m[32m  /** See operator() */[m
[32m+[m[32m  template <typename... Indices,[m
[32m+[m[32m            typename = typename std::enable_if<N == sizeof...(Indices)>::type>[m
[32m+[m[32m  T const &operator()(Indices... args) const;[m
[32m+[m
[32m+[m[32m  /** See operator() */[m
[32m+[m[32m  template <size_t M>//, typename = typename std::enable_if<N != M>::type>[m
   Tensor<T, N - M> operator[](Indices<M> const &indices);[m
 [m
   /** Slices denotate the dimensions which are left free, while indices[m
[36m@@ -824,8 +876,11 @@[m [mpublic:[m
   /** Returns a reference: only used to invoke reference constructor */[m
   Tensor<T, N>::Proxy ref();[m
 [m
[31m-  template <typename U, size_t M, typename Container>[m
[31m-  friend void Fill(Tensor<U, M> &tensor, Container const &container);[m
[32m+[m[32m  template <typename U, size_t M, typename RAIt>[m
[32m+[m[32m  friend void Fill(Tensor<U, M> &tensor, RAIt const &begin, RAIt const &end);[m
[32m+[m
[32m+[m[32m  template <typename U, size_t M, typename X>[m
[32m+[m[32m  friend void Fill(Tensor<U, M> &tensor, X const &value);[m
 [m
   /** Allocates a Tensor with shape `shape`, whose total number of elements [m
    *  must be equivalent to *this (or std::logic_error is thrown). The [m
[36m@@ -853,11 +908,17 @@[m [mprivate:[m
 [m
   /* ------------- Expansion for operator()() ------------- */[m
 [m
[31m-  // Expansion[m
[32m+[m[32m  // Expansion which returns a Tensor[m
   template <size_t M>[m
[31m-  Tensor<T, N - M> pAccessExpansion(size_t cumul_index);[m
[32m+[m[32m  Tensor<T, N - M> pTensorExpansion(size_t cumul_index);[m
   template <size_t M, typename... Indices>[m
[31m-  Tensor<T, N - M> pAccessExpansion(size_t cumul_index, size_t next_index, Indices...);[m
[32m+[m[32m  Tensor<T, N - M> pTensorExpansion(size_t cumul_index, size_t next_index, Indices...);[m
[32m+[m
[32m+[m[32m  // Expansion which returns a scalar[m
[32m+[m[32m  template <typename... Indices>[m
[32m+[m[32m  T &pElementExpansion(size_t cumul_index) { return data_[cumul_index]; }[m
[32m+[m[32m  template <typename... Indices>[m
[32m+[m[32m  T &pElementExpansion(size_t cumul_index, size_t next_index, Indices...);[m
 [m
   /* ------------- Expansion for slice() ------------- */[m
 [m
[36m@@ -965,19 +1026,20 @@[m [mTensor<T, N>::Tensor(size_t const (&dimensions)[N], std::function<FunctionType>[m
   ref_ = std::shared_ptr<T>(data_, _ARRAY_DELETER(T));[m
 }[m
 [m
[31m-/** Fills the elements of `tensor` with the elements in container.[m
[31m- *  container must implement a forwards iterator. The number of [m
[31m- *  elements in container must be equivalent to the capacity of[m
[31m- *  `tensor`, otherwise a std::logic_error is thrown.[m
[32m+[m[32m/** Fills the elements of `tensor` with the elements between.[m
[32m+[m[32m *  `begin` and `end`, which must be random access iterators. The number[m[41m [m
[32m+[m[32m *  elements between `begin` and `end` must be equivalent to the capacity of[m
[32m+[m[32m *  `tensor`, otherwise std::logic_error is thrown.[m
  */[m
[31m-template <typename T, size_t N, typename Container>[m
[31m-void Fill(Tensor<T, N> &tensor, Container const &container)[m
[32m+[m[32mtemplate <typename U, size_t M, typename RAIt>[m
[32m+[m[32mvoid Fill(Tensor<U, M> &tensor, RAIt const &begin, RAIt const &end)[m
 {[m
   size_t cumul_sum = tensor.shape_.index_product();[m
[31m-  if (container.size() != cumul_sum)[m
[32m+[m[32m  auto dist_sum = std::distance(begin, end);[m
[32m+[m[32m  if (dist_sum > 0 && cumul_sum != (size_t)dist_sum)[m
     throw std::logic_error(NELEMENTS);[m
[31m-  auto it = container.begin();[m
[31m-  std::function<void(T *)> allocate = [&it](T *x) -> void[m
[32m+[m[32m  RAIt it = begin;[m
[32m+[m[32m  std::function<void(U *)> allocate = [&it](U *x) -> void[m
   {[m
     *x = *it;[m
     ++it;[m
[36m@@ -985,6 +1047,16 @@[m [mvoid Fill(Tensor<T, N> &tensor, Container const &container)[m
   tensor.pMap(allocate);[m
 }[m
 [m
[32m+[m[32mtemplate <typename U, size_t M, typename X>[m
[32m+[m[32mvoid Fill(Tensor<U, M> &tensor, X const &value)[m
[32m+[m[32m{[m
[32m+[m[32m  std::function<void(U *)> allocate = [&value](U *x) -> void[m
[32m+[m[32m  {[m
[32m+[m[32m    *x = value;[m
[32m+[m[32m  };[m
[32m+[m[32m  tensor.pMap(allocate);[m
[32m+[m[32m}[m
[32m+[m
 template <typename T, size_t N>[m
 Tensor<T, N>::Tensor(Tensor<T, N> const &tensor)[m
   : shape_(tensor.shape_)[m
[36m@@ -1053,21 +1125,51 @@[m [mTensor<T, N> &Tensor<T, N>::operator=(Expression<NodeType> const &rhs)[m
 [m
 template <typename T, size_t N>[m
 template <typename... Indices>[m
[32m+[m[32mTensor<T, N - sizeof...(Indices)> Tensor<T, N>::at(Indices... args)[m
[32m+[m[32m{[m
[32m+[m[32m  static_assert(N >= sizeof...(args), RANK_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));[m
[32m+[m[32m  return pTensorExpansion<sizeof...(args)>(0, args...);[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32mtemplate <typename T, size_t N>[m
[32m+[m[32mtemplate <typename... Indices, typename>[m
 Tensor<T, N - sizeof...(Indices)> Tensor<T, N>::operator()(Indices... args)[m
 {[m
   static_assert(N >= sizeof...(args), RANK_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));[m
[31m-  return pAccessExpansion<sizeof...(args)>(0, args...);[m
[32m+[m[32m  return pTensorExpansion<sizeof...(args)>(0, args...);[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32mtemplate <typename T, size_t N>[m
[32m+[m[32mtemplate <typename... Indices, typename>[m
[32m+[m[32mT &Tensor<T, N>::operator()(Indices... args)[m
[32m+[m[32m{[m
[32m+[m[32m  static_assert(N >= sizeof...(args), RANK_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));[m
[32m+[m[32m  return pElementExpansion(0, args...);[m
 }[m
 [m
 template <typename T, size_t N>[m
 template <typename... Indices>[m
[32m+[m[32mTensor<T, N - sizeof...(Indices)> const Tensor<T, N>::at(Indices... args) const[m
[32m+[m[32m{[m
[32m+[m[32m return (*const_cast<self_type*>(this))(args...);[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32mtemplate <typename T, size_t N>[m
[32m+[m[32mtemplate <typename... Indices, typename>[m
 Tensor<T, N - sizeof...(Indices)> const Tensor<T, N>::operator()(Indices... args) const[m
 {[m
   return (*const_cast<self_type*>(this))(args...);[m
 }[m
 [m
 template <typename T, size_t N>[m
[31m-template <size_t M>[m
[32m+[m[32mtemplate <typename... Indices, typename>[m
[32m+[m[32mT const &Tensor<T, N>::operator()(Indices... args) const[m
[32m+[m[32m{[m
[32m+[m[32m  return (*const_cast<self_type*>(this))(args...);[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32mtemplate <typename T, size_t N>[m
[32m+[m[32mtemplate <size_t M>//, typename>[m
 Tensor<T, N - M> Tensor<T, N>::operator[](Indices<M> const &indices)[m
 {[m
   size_t cumul_index = 0;[m
[36m@@ -1166,14 +1268,14 @@[m [mstd::ostream &operator<<(std::ostream &os, const Tensor<T, N> &tensor)[m
 // private methods[m
 template <typename T, size_t N>[m
 template <size_t M>[m
[31m-Tensor<T, N - M> Tensor<T, N>::pAccessExpansion(size_t cumul_index)[m
[32m+[m[32mTensor<T, N - M> Tensor<T, N>::pTensorExpansion(size_t cumul_index)[m
 {[m
   return Tensor<T, N - M>(shape_.dimensions_ + M, strides_ + M, data_ + cumul_index, std::shared_ptr<T>(ref_));[m
 }[m
 [m
 template <typename T, size_t N>[m
 template <size_t M, typename... Indices>[m
[31m-Tensor<T, N - M> Tensor<T, N>::pAccessExpansion([m
[32m+[m[32mTensor<T, N - M> Tensor<T, N>::pTensorExpansion([m
  size_t cumul_index, size_t next_index, Indices... rest)[m
 {[m
   if (next_index > shape_.dimensions_[M - sizeof...(rest) - 1] || next_index == 0)[m
[36m@@ -1181,7 +1283,19 @@[m [mTensor<T, N - M> Tensor<T, N>::pAccessExpansion([m
 [m
   // adjust for 1 index array access[m
   cumul_index += strides_[M - sizeof...(rest) - 1] * (next_index - 1);[m
[31m-  return pAccessExpansion<M>(cumul_index, rest...);[m
[32m+[m[32m  return pTensorExpansion<M>(cumul_index, rest...);[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32mtemplate <typename T, size_t N>[m
[32m+[m[32mtemplate <typename... Indices>[m
[32m+[m[32mT &Tensor<T, N>::pElementExpansion(size_t cumul_index, size_t next_index, Indices... rest)[m[41m  [m
[32m+[m[32m{[m
[32m+[m[32m  if (next_index > shape_.dimensions_[N - sizeof...(rest) - 1] || next_index == 0)[m
[32m+[m[32m    throw std::logic_error(INDEX_OUT_OF_BOUNDS("Tensor::operator(Indices...)"));[m
[32m+[m
[32m+[m[32m  // adjust for 1 index array access[m
[32m+[m[32m  cumul_index += strides_[N - sizeof...(rest) - 1] * (next_index - 1);[m
[32m+[m[32m  return pElementExpansion(cumul_index, rest...);[m
 }[m
 [m
 /* ------------- Slice Expansion ------------- */[m
[36m@@ -2816,11 +2930,12 @@[m [mBinaryAdd<LHSType, RHSType>::BinaryAdd(LHSType const &lhs, RHSType const &rhs)[m
 [m
 template <typename LHSType, typename RHSType>[m
 template <typename... Indices>[m
[31m-Tensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)> BinaryAdd<LHSType, RHSType>::operator()(Indices... indices) const[m
[32m+[m[32mTensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)>[m[41m [m
[32m+[m[32m  BinaryAdd<LHSType, RHSType>::operator()(Indices... indices) const[m
 {[m
   static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Addition"));[m
[31m-  return add(ValueToTensor<LHSType>(lhs_).value(indices...),[m
[31m-             ValueToTensor<RHSType>(rhs_).value(indices...));[m
[32m+[m[32m  return add(ValueAsTensor<LHSType>(lhs_)(indices...),[m
[32m+[m[32m             ValueAsTensor<RHSType>(rhs_)(indices...));[m
 }[m
 [m
 template <typename LHSType, typename RHSType>[m
[36m@@ -2876,8 +2991,8 @@[m [mTensor<typename LHSType::value_type, LHSType::rank() - sizeof...(Indices)> Binar[m
 {[m
   static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Subtraction"));[m
   return subtract([m
[31m-      ValueToTensor<LHSType>(lhs_).value(indices...),[m
[31m-      ValueToTensor<RHSType>(rhs_).value(indices...));[m
[32m+[m[32m      ValueAsTensor<LHSType>(lhs_)(indices...),[m
[32m+[m[32m      ValueAsTensor<RHSType>(rhs_)(indices...));[m
 }[m
 [m
 template <typename LHSType, typename RHSType>[m
[36m@@ -2934,8 +3049,8 @@[m [mtemplate <typename... Indices>[m
 Tensor<typename LHSType::value_type, LHSType::rank() + RHSType::rank() - sizeof...(Indices) - 2> BinaryMul<LHSType, RHSType>::operator()(Indices... indices) const[m
 {[m
   static_assert(rank() >= sizeof...(Indices), RANK_OUT_OF_BOUNDS("Binary Multiplication"));[m
[31m-  return multiply(ValueToTensor<LHSType>(lhs_).value(),[m
[31m-                  ValueToTensor<RHSType>(rhs_).value())(indices...);[m
[32m+[m[32m  return multiply(ValueAsTensor<LHSType>(lhs_)(),[m
[32m+[m[32m                  ValueAsTensor<RHSType>(rhs_)())(indices...);[m
 }[m
 [m
 template <typename LHSType, typename RHSType>[m
[1mdiff --git a/test/access.test.cc b/test/access.test.cc[m
[1mindex 6fe5ca5..23394a5 100644[m
[1m--- a/test/access.test.cc[m
[1m+++ b/test/access.test.cc[m
[36m@@ -31,4 +31,12 @@[m [mTEST_CASE("Tensor Access", "[int]") {[m
     Indices<4> indices({1, 1, 1, 1});[m
     REQUIRE(tensor_1[indices] == 1111);[m
   }[m
[32m+[m
[32m+[m[32m  SECTION("at vs operator()") {[m
[32m+[m[32m    REQUIRE(typeid(tensor_1(1)) == typeid(tensor_1.at(1)));[m
[32m+[m[32m    REQUIRE(typeid(tensor_1(1, 1)) == typeid(tensor_1.at(1, 1)));[m
[32m+[m[32m    REQUIRE(typeid(tensor_1(1, 1, 1)) == typeid(tensor_1.at(1, 1, 1)));[m
[32m+[m[32m    REQUIRE(typeid(tensor_1(1, 1, 1, 1)) != typeid(tensor_1.at(1, 1, 1, 1)));[m
[32m+[m[32m    REQUIRE(typeid(int) == typeid(tensor_1(1,1,1,1)));[m
[32m+[m[32m  }[m
 }[m
[1mdiff --git a/test/exception.test.cc b/test/exception.test.cc[m
[1mindex cae0e72..1f34f54 100644[m
[1m--- a/test/exception.test.cc[m
[1m+++ b/test/exception.test.cc[m
[36m@@ -43,8 +43,8 @@[m [mTEST_CASE("Logic Errors") {[m
     Tensor<int32_t, 3> tensor_1({1, 2, 3});[m
     std::vector<int32_t> vec_incorrect(7, 0);[m
     std::vector<int32_t> vec_correct(6, 0);[m
[31m-    REQUIRE_NOTHROW(Fill(tensor_1, vec_correct));[m
[31m-    REQUIRE_THROWS_AS(Fill(tensor_1, vec_incorrect), std::logic_error);[m
[32m+[m[32m    REQUIRE_NOTHROW(Fill(tensor_1, vec_correct.begin(), vec_correct.end()));[m
[32m+[m[32m    REQUIRE_THROWS_AS(Fill(tensor_1, vec_incorrect.begin(), vec_incorrect.end()), std::logic_error);[m
   }[m
 [m
   SECTION("Iterator") {[m
[1mdiff --git a/test/intialize.test.cc b/test/intialize.test.cc[m
[1mindex 75e99c7..6d04daa 100644[m
[1m--- a/test/intialize.test.cc[m
[1m+++ b/test/intialize.test.cc[m
[36m@@ -26,10 +26,10 @@[m [mTEST_CASE("Intializing Tensors", "[int]") {[m
 [m
   SECTION("Rank and Dimensions") {[m
     REQUIRE(tensor_1.rank() == 4);[m
[31m-    REQUIRE(tensor_1(1).rank() == 3);[m
[31m-    REQUIRE(tensor_1(1, 1).rank() == 2);[m
[31m-    REQUIRE(tensor_1(1, 1, 1).rank() == 1);[m
[31m-    REQUIRE(tensor_1(1, 1, 1, 1).rank() == 0);[m
[32m+[m[32m    REQUIRE(tensor_1.at(1).rank() == 3);[m
[32m+[m[32m    REQUIRE(tensor_1.at(1, 1).rank() == 2);[m
[32m+[m[32m    REQUIRE(tensor_1.at(1, 1, 1).rank() == 1);[m
[32m+[m[32m    REQUIRE(tensor_1.at(1, 1, 1, 1).rank() == 0);[m
     for (int i = 1; i <= 4; ++i) REQUIRE(tensor_1.dimension(i) == i);[m
     for (int i = 1; i <= 3; ++i) REQUIRE(tensor_1(1).dimension(i) == i + 1);[m
   }[m
[36m@@ -86,11 +86,17 @@[m [mTEST_CASE("Intializing Tensors", "[int]") {[m
     auto naturals = Tensor<int32_t, 3>({2, 3, 4});[m
     std::deque<int32_t> container{};[m
     for (int i = 0; i < 24; ++i) container.push_back(i);[m
[31m-    Fill(naturals, container);[m
[32m+[m[32m    Fill(naturals, container.begin(), container.end());[m
     for (size_t i = 1; i <= naturals.dimension(1); ++i)[m
       for (size_t j = 1; j <= naturals.dimension(2); ++j)[m
         for (size_t k = 1; k <= naturals.dimension(3); ++k)[m
           REQUIRE((size_t)naturals(i, j, k) == (i - 1) * 12 + (j - 1) * 4 + k - 1);[m
[32m+[m
[32m+[m[32m    Fill(naturals, 1);[m
[32m+[m[32m    for (size_t i = 1; i <= naturals.dimension(1); ++i)[m
[32m+[m[32m      for (size_t j = 1; j <= naturals.dimension(2); ++j)[m
[32m+[m[32m        for (size_t k = 1; k <= naturals.dimension(3); ++k)[m
[32m+[m[32m          REQUIRE((size_t)naturals(i, j, k) == 1);[m
  }[m
 [m
  SECTION("Factory Method") {[m
