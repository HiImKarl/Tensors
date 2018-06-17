#include <tensor.hh>
#include <benchmark/benchmark.h>
#include <cmath>

using namespace tensor;

static void BM_Degree5_TensorInitialization(benchmark::State &state) 
{
  uint32_t dim = state.range(0);
  for (auto _ : state) {
    Tensor<int32_t, 5>({dim, dim, dim, dim, dim});
  }
}

BENCHMARK(BM_Degree5_TensorInitialization)->RangeMultiplier(2)->Range(2, 64);

static void BM_Degree5_TensorValueInitialization(benchmark::State &state) 
{
  uint32_t dim = state.range(0);
  for (auto _ : state) {
    Tensor<int32_t, 5>({dim, dim, dim, dim, dim}, -1);
  }
}

BENCHMARK(BM_Degree5_TensorValueInitialization)->RangeMultiplier(2)->Range(2, 64);

static void BM_Degree0_TensorInitialization(benchmark::State &state) 
{
  for (auto _ : state) {
    Tensor<int32_t>();
  }
}

BENCHMARK(BM_Degree0_TensorInitialization);

static void BM_Degree0_TensorValueInitialization(benchmark::State &state) 
{
  for (auto _ : state) {
    Tensor<int32_t>(-1);
  }
}

BENCHMARK(BM_Degree0_TensorValueInitialization);
