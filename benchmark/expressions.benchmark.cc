#include <tensor.hh>
#include <benchmark/benchmark.h>

using namespace tensor;

static void BM_TensorTensor_Addition(benchmark::State &state)
{
  size_t dim = state.range(0);
  auto t1 = Tensor<int32_t, 5>({dim, dim, dim, dim, dim});
  auto t2 = Tensor<int32_t, 5>({dim, dim, dim, dim, dim});
  for (auto _ : state) Tensor<int32_t, 5>(t1 + t2);
}

BENCHMARK(BM_TensorTensor_Addition)->RangeMultiplier(2)->Range(2,  4 * sizeof(size_t));

static void BM_ScalarScalar_Addition(benchmark::State &state)
{
  for (auto _ : state) Tensor<int32_t>(Tensor<int32_t>() + Tensor<int32_t>());
}

BENCHMARK(BM_ScalarScalar_Addition);

static void BM_TensorTensor_Multiplication(benchmark::State &state)
{
  size_t dim = state.range(0);
  auto t1 = Tensor<int32_t, 3>({dim, dim, dim});
  auto t2 = Tensor<int32_t, 3>({dim, dim, dim});
  for (auto _ : state) Tensor<int32_t, 4>(t1 * t2);
}

BENCHMARK(BM_TensorTensor_Multiplication)->RangeMultiplier(2)->Range(2,  4 * sizeof(size_t));

static void BM_ScalarScalar_Multiplication(benchmark::State &state)
{
  for (auto _ : state) Tensor<int32_t>(Tensor<int32_t>() * Tensor<int32_t>());
}

BENCHMARK(BM_ScalarScalar_Multiplication);

static void BM_TensorExpression(benchmark::State &state)
{
  size_t dim = state.range(0);
  auto t1 = Tensor<int32_t, 2>({dim, dim}); 
  auto t2 = Tensor<int32_t, 2>({dim, dim});
  auto t3 = Tensor<int32_t, 2>({dim, dim});
  auto t4 = Tensor<int32_t, 2>({dim, dim});
  for (auto _ : state) Tensor<int32_t, 2>(t1 * t2 + t3 * (t2 - t1));
}

BENCHMARK(BM_TensorExpression)->RangeMultiplier(2)->Range(2,  4 * sizeof(size_t));
