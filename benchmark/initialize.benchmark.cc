#include <tensor.hh>
#include <benchmark/benchmark.h>
#include <cmath>

using namespace tensor;

static void BM_TensorInitialization(benchmark::State &state) 
{
  size_t dim = state.range(0);
  for (auto _ : state)
    Tensor<int32_t, 5>({dim, dim, dim, dim, dim});
}

BENCHMARK(BM_TensorInitialization)->RangeMultiplier(2)->Range(2, 4 * sizeof(size_t));

static void BM_TensorValueInitialization(benchmark::State &state) 
{
  size_t dim = state.range(0);
  for (auto _ : state)
    Tensor<int32_t, 5>({dim, dim, dim, dim, dim}, -1);
}

BENCHMARK(BM_TensorValueInitialization)->RangeMultiplier(2)->Range(2, 4 * sizeof(size_t));

static void BM_ScalarInitialization(benchmark::State &state) 
{
  for (auto _ : state) Tensor<int32_t>();
}

BENCHMARK(BM_ScalarInitialization);

static void BM_ScalarValueInitialization(benchmark::State &state) 
{
  for (auto _ : state) Tensor<int32_t>(-1);
}

BENCHMARK(BM_ScalarValueInitialization);
