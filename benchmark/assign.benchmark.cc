#include <tensor.hh>
#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

using namespace tensor;

static void BM_TensorTensor_Assignment(benchmark::State &state)
{
  size_t dim = state.range(0);
  Tensor<int32_t, 5> t1({dim, dim, dim, dim, dim});
  Tensor<int32_t, 5> t2({dim, dim, dim, dim, dim});
  for (auto _ : state) t1 = t2;
}

BENCHMARK(BM_TensorTensor_Assignment)->RangeMultiplier(2)->Range(2,  4 * sizeof(size_t));

static void BM_ScalarScalar_Assignment(benchmark::State &state)
{
  Tensor<int32_t> t1{};
  Tensor<int32_t> t2{};
  for (auto _ : state) t1 = t2;
}

BENCHMARK(BM_ScalarScalar_Assignment);

static void BM_VectorTensor_Assignment(benchmark::State &state)
{
  size_t dim = state.range(0);
  std::vector<int32_t> vec ((size_t)pow(dim, 5), -1);
  Tensor<int32_t, 5> t1({dim, dim, dim, dim, dim});
  for (auto _ : state) Fill(t1, vec);
}

BENCHMARK(BM_VectorTensor_Assignment)->RangeMultiplier(2)->Range(2, 4 * sizeof(size_t)); 
