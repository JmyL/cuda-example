#include <cuda_image_morph/cuda_image_morph.hpp>

#include <benchmark/benchmark.h>

void BM_IntervalMapAssign(benchmark::State &state) {
    int N = state.range(0);
    for (auto _ : state) {
        state.PauseTiming();
        auto input = std::vector<unsigned char>(N * N, 0);
        auto output = std::vector<unsigned char>(N * N, 0);
        CudaImageMorph morpher(N, N, input.data());
        state.ResumeTiming();
        morpher.dilate();
    }
    state.SetComplexityN(N);
}

BENCHMARK(BM_IntervalMapAssign)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();