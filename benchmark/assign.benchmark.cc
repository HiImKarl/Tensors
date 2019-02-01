#include "benchmark.hh"

using namespace tensor;

static void BM_TensorTensor_Assignment(benchmark::State &state)
{
  size_t dim = state.range(0);
  auto t1 = Tensor<int32_t, 5>({dim, dim, dim, dim, dim});
  auto t2 = Tensor<int32_t, 5>({dim, dim, dim, dim, dim});
  for (auto _ : state) t1 = t2;
}

BENCHMARK(BM_TensorTensor_Assignment)->RangeMultiplier(2)->Range(2,  128);

static void BM_ScalarScalar_Assignment(benchmark::State &state)
{
  auto t1 = Scalar<int32_t>{};
  auto t2 = Scalar<int32_t>{};
  for (auto _ : state) t1 = t2;
}

BENCHMARK(BM_ScalarScalar_Assignment);

static void BM_VectorTensor_Assignment(benchmark::State &state)
{
  size_t dim = state.range(0);
  auto vec = std::vector<int32_t>((size_t)pow(dim, 5), -1);
  auto t1 = Tensor<int32_t, 5>({dim, dim, dim, dim, dim});
  for (auto _ : state) Fill(t1, vec.begin(), vec.end());
}

BENCHMARK(BM_VectorTensor_Assignment)->RangeMultiplier(2)->Range(2, 128); 

static void BM_TensorFill_Assignment(benchmark::State &state) 
{
  size_t dim = state.range(0);
  auto vec = std::vector<int32_t>(std::pow(dim, 5), -1);
  auto t1 = Tensor<int32_t, 5>({dim, dim, dim, dim, dim}, -1);
  for (auto _ : state)
    t1.Fill(vec.begin(), vec.end());
}

BENCHMARK(BM_TensorFill_Assignment)->RangeMultiplier(2)->Range(2, 128);


