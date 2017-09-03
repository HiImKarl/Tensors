#ifndef TENSOR_H_
#define TENSOR_H_
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <stdint.h>
#include <string>
#include <vector>

namespace tensor {

template <typename T> class Tensor {
  template <typename X> using vector = std::vector<X>;
  using string = std::string;

public:
  // Constructors
  explicit Tensor()
      : dimensions_(vector<uint16_t>()), elements_(nullptr), is_owner_(false) {}
  explicit Tensor(const vector<uint16_t> &indices)
      : dimensions_(indices), is_owner_(true) {
    elements_ = new T[pVectorProduct(indices)];
  }
  explicit Tensor(vector<uint16_t> &&indices)
      : dimensions_(std::move(indices)), is_owner_(true) {
    elements_ = new T[pVectorProduct(dimensions_)];
  }
  explicit Tensor(std::initializer_list<uint16_t> indices) : is_owner_(true) {
    dimensions_ = vector<uint16_t>(indices.begin(), indices.end());
    elements_ = new T[pVectorProduct(dimensions_)];
  }
  Tensor(const Tensor<T> &tensor)
      : dimensions_(tensor.dimensions_), is_owner_(true) {
    elements_ = pArrayCopy(tensor.elements_, pVectorProduct(dimensions_));
  }
  Tensor(Tensor<T> &&tensor)
      : dimensions_(tensor.dimensions_), is_owner_(true) {
    // If tensor owns its memory, move; ow. copy
    if (tensor.is_owner_) {
      elements_ = tensor.elements_;
      tensor.elements_ = nullptr;
    } else {
      elements_ = pArrayCopy(tensor.elements_, pVectorProduct(dimensions_));
    }
  }

  // Assignment
  Tensor<T> &operator=(const Tensor<T> &tensor);
  Tensor<T> &operator=(Tensor<T> &&tensor);

  // Destructor
  ~Tensor() {
    if (is_owner_) {
      delete[] elements_;
    }
    dimensions_.clear();
  }

  // Getters
  size_t degree() const { return dimensions_.size(); }
  const vector<uint16_t> &dimensions() const { return dimensions_; }

  // Access to elements_
  template <typename... Args> Tensor<T> operator()(Args... args) {
    vector<uint16_t> indices{};
    indices.reserve(4 * sizeof(dimensions_));
    return pAccessExpansion(indices, args...);
  }

  // Setters
  Tensor<T> &operator=(T elem) {
    if (dimensions_.size() != 0) {
      std::cout << "Cannot assign scalar to multi dimensional vector\n";
      return *this;
    }
    *elements_ = elem;
    return *this;
  }

  // print
  template <typename X>
  friend std::ostream &operator<<(std::ostream &os, const Tensor<X> &tensor);

  // Operations
  template <typename X>
  friend Tensor<X> operator+(const Tensor<X> &tensor_1,
                             const Tensor<X> &tensor_2);
  template <typename X>
  friend Tensor<X> operator-(const Tensor<X> &tensor_1,
                             const Tensor<X> &tensor_2);
  void operator-=(const Tensor<T> &tensor);
  void operator+=(const Tensor<T> &tensor);
  template <typename X>
  friend Tensor<X> operator*(const Tensor<X> &, const Tensor<X> &);

private:
  vector<uint16_t> dimensions_;
  T *elements_;

  // This flag denotes that this tensor object is the principle owner of memory
  // Using a shared pointer would dramatically slow down
  // operations that require the generation of large numbers of tensors
  bool is_owner_;

  // Access Expansion
  Tensor<T> pAccessExpansion(const vector<uint16_t> &);
  template <typename... Args>
  Tensor<T> pAccessExpansion(vector<uint16_t> &, uint16_t, Args...);
  uint32_t pCumulativeIndex(const vector<uint16_t> &xs);

  // Utility functions
  static uint32_t pVectorProduct(const vector<uint16_t> &xs);
  // Copies an array with size specified by size and returns the ptr
  static T *pArrayCopy(T *xs, uint32_t size);
  static string pRepeatString(string to_repeat, uint32_t n);

}; // Tensor

// Private static utility functions
template <typename T>
uint32_t Tensor<T>::pVectorProduct(const vector<uint16_t> &xs) {
  uint32_t product = 1;
  for (uint16_t i : xs) {
    product *= i;
  }
  return product;
}
// Copies an array with size specified by size and returns the ptr
template <typename T> T *Tensor<T>::pArrayCopy(T *xs, uint32_t size) {
  T *copy = new T[size];
  for (uint32_t i = 0; i < size; ++i) {
    copy[i] = xs[i];
  }
  return copy;
}

template <typename T>
std::string Tensor<T>::pRepeatString(string to_repeat, uint32_t n) {
  string out{};
  // string::size returns byte size
  out.reserve(n * to_repeat.size());
  for (uint32_t i = 0; i < n; ++i)
    out += to_repeat;
  return out;
}

// Tensor Methods
template <typename T> Tensor<T> &Tensor<T>::operator=(const Tensor<T> &tensor) {
  if (dimensions_ != tensor.dimensions_) {
    std::cout << "Tensor Assignment Failed :: tensors are not the same size\n";
    return *this;
  } else {
    uint32_t total_elems = pVectorProduct(dimensions_);
    for (uint32_t i = 0; i < total_elems; ++i) {
      elements_[i] = tensor.elements_[i];
    }
    return *this;
  }
}

template <typename T> Tensor<T> &Tensor<T>::operator=(Tensor<T> &&tensor) {
  if (dimensions_ != tensor.dimensions_) {
    std::cout << "Tensor Assignment Failed :: tensors are not the same size\n";
    return *this;
  }
  // DO NOT MOVE DATA unless boths tensors are principle owners
  if (tensor.is_owner_ && is_owner_) {
    elements_ = tensor.elements_;
    tensor.elements_ = nullptr;
    return *this;
  }
  if (dimensions_ != tensor.dimensions_) {
    std::cout << "Tensor Assignment Failed :: tensors are not the same size\n";
    return *this;
  } else {
    uint32_t total_elems = pVectorProduct(dimensions_);
    for (uint32_t i = 0; i < total_elems; ++i) {
      elements_[i] = tensor.elements_[i];
    }
    return *this;
  }
}

template <typename T>
Tensor<T> Tensor<T>::pAccessExpansion(const vector<uint16_t> &indices) {
  uint32_t indices_product = pCumulativeIndex(indices);
  vector<uint16_t> new_indices{};
  new_indices.reserve(4 * (dimensions_.size() - indices.size()));
  for (size_t i = indices.size(); i < dimensions_.size(); ++i) {
    new_indices.push_back(dimensions_.at(i));
  }
  Tensor<T> new_tensor(new_indices);
  new_tensor.elements_ = elements_ + indices_product;
  new_tensor.is_owner_ = false;
  return new_tensor;
}

template <typename T>
template <typename... Args>
Tensor<T> Tensor<T>::pAccessExpansion(vector<uint16_t> &indices,
                                      uint16_t next_index, Args... rest) {
  if (next_index >= dimensions_.at(indices.size())) {
    std::cout << "Index " << indices.size() << " (" << next_index
              << ") Out of Bounds\n";
    return Tensor<T>();
  }
  indices.push_back(next_index);
  return pAccessExpansion(indices, rest...);
}

template <typename T>
uint32_t Tensor<T>::pCumulativeIndex(const vector<uint16_t> &xs) {
  uint32_t total_elems = pVectorProduct(dimensions_);
  uint32_t product = 0;
  for (size_t i = 0; i < xs.size(); ++i) {
    total_elems /= dimensions_.at(i);
    product += xs.at(i) * total_elems;
  }
  return product;
}

// Overloaded operators
template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
  uint32_t indices_product = Tensor<T>::pVectorProduct(tensor.dimensions());
  uint32_t bracket_mod_arr[tensor.degree()];
  uint32_t prod = 1;
  for (size_t i = 0; i < tensor.degree(); ++i) {
    prod *= tensor.dimensions().at(tensor.degree() - i - 1);
    bracket_mod_arr[i] = prod;
  }
  os << tensor.pRepeatString("[", tensor.degree());
  os << tensor.elements_[0] << ", ";
  for (uint32_t i = 1; i < indices_product - 1; ++i) {
    os << tensor.elements_[i];
    uint16_t num_brackets = 0;
    for (; num_brackets < tensor.degree(); ++num_brackets) {
      if (!((i + 1) % bracket_mod_arr[num_brackets]) == 0)
        break;
    }
    os << tensor.pRepeatString("]", num_brackets);
    os << ", ";
    os << tensor.pRepeatString("[", num_brackets);
  }
  os << tensor.elements_[indices_product - 1];
  os << tensor.pRepeatString("]", tensor.degree());
  return os;
}

template <typename T> void Tensor<T>::operator+=(const Tensor<T> &tensor) {
  if (dimensions_ != tensor.dimensions_) {
    std::cout << "operator+=(Tensor) Failed :: Incompatible Dimensions\n";
    return Tensor<T>();
  }
  uint32_t total_dims = pVectorProduct(dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    elements_[i] += tensor.elements_[i];
  }
}

template <typename T> void Tensor<T>::operator-=(const Tensor<T> &tensor) {
  if (dimensions_ != tensor.dimensions_) {
    std::cout << "operator-=(Tensor) Failed :: Incompatible Dimensions\n";
    return Tensor<T>();
  }
  uint32_t total_dims = pVectorProduct(dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    elements_[i] -= tensor.elements_[i];
  }
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &tensor_1, const Tensor<T> &tensor_2) {
  if (tensor_1.dimensions_ != tensor_2.dimensions_) {
    std::cout
        << "operator+(Tensor, Tensor) Failed :: Incompatible Dimensions\n";
    return Tensor<T>();
  }
  Tensor<T> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    new_tensor.elements_[i] = tensor_1.elements_[i] + tensor_2.elements_[i];
  }
  return new_tensor;
}

template <typename T>
Tensor<T> operator-(const Tensor<T> &tensor_1, const Tensor<T> &tensor_2) {
  if (tensor_1.dimensions_ != tensor_2.dimensions_) {
    std::cout
        << "operator+(Tensor, Tensor) Failed :: Incompatible Dimensions\n";
    return Tensor<T>();
  }
  Tensor<T> new_tensor{tensor_1.dimensions_};
  uint32_t total_dims = pVectorProduct(tensor_1.dimensions_);
  for (uint32_t i = 0; i < total_dims; ++i) {
    new_tensor.elements_[i] = tensor_1.elements_[i] - tensor_2.elements_[i];
  }
  return new_tensor;
}
} // tensor
#endif // TENSORS_H_
