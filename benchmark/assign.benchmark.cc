#include <tensor.hh>
#include <benchmark/benchmark.h>

using namespace tensor;

static void BM_TensorTensor_Assignment(benchmark::State &state)
{
  uint32_t dim = state.range(0);
  Tensor<int32_t, 5> t1({dim, dim, dim, dim, dim});
  Tensor<int32_t, 5> t2({dim, dim, dim, dim, dim});
  for (auto _ : state) t1 = t2;
}

BENCHMARK(BM_TensorTensor_Assignment)->RangeMultiplier(2)->Range(2, 64);

static void BM_ScalarScalar_Assignment(benchmark::State &state)
{
  Tensor<int32_t> t1{};
  Tensor<int32_t> t2{};
  for (auto _ : state) t1 = t2;
}

BENCHMARK(BM_ScalarScalar_Assignment);
