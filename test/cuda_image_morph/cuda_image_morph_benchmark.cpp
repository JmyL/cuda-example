#include <benchmark/benchmark.h>

void BM_IntervalMapAssign(benchmark::State &state) {
    int N = state.range(0);
    for (auto _ : state) {
        state.PauseTiming();
        state.ResumeTiming();
    }
    state.SetComplexityN(N);
}

BENCHMARK(BM_IntervalMapAssign)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 20)
    ->Complexity();

BENCHMARK_MAIN();