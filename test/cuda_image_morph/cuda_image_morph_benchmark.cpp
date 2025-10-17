#include <cuda_image_morph/cuda_image_morph.hpp>

#include <benchmark/benchmark.h>

void BM_NaiveDilate(benchmark::State &state) {
    int N = state.range(0);
    for (auto _ : state) {
        state.PauseTiming();
        auto input = std::vector<unsigned char>(N * N, 0);
        auto output = std::vector<unsigned char>(N * N, 0);
        CudaImageMorph morpher(N, N, input.data());
        state.ResumeTiming();
        morpher.naive_dilate();
    }
    state.SetComplexityN(N);
}

BENCHMARK(BM_NaiveDilate)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);

void BM_SharedDilate(benchmark::State &state) {
    int N = state.range(0);
    for (auto _ : state) {
        state.PauseTiming();
        auto input = std::vector<unsigned char>(N * N, 0);
        auto output = std::vector<unsigned char>(N * N, 0);
        CudaImageMorph morpher(N, N, input.data());
        state.ResumeTiming();
        morpher.shared_dilate();
    }
    state.SetComplexityN(N);
}

BENCHMARK(BM_SharedDilate)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();