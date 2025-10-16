#include "cuda_sum/cuda_sum.hpp"
#include "cuda_sum/cuda_util.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>

using ::testing::ElementsAre;

TEST(ACudaUtil, PrintsDeviceInfo) { util::print_device_info(); }

TEST(ACudaUtil, ceilsNumberToPowerOfTwo) {
    EXPECT_EQ(32, util::ceilPowerOfTwo(17));
    EXPECT_EQ(32, util::ceilPowerOfTwo(32));
}

TEST(ACudaSum, HandlesRandomNumberOfBlocks) {
    // generate random number
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<> int_dist{1, 1024 * 1024};

    for (auto i = 100; i > 0; --i) {
        auto N = int_dist(gen);
        std::vector<int> data(N, 1); // Fill with 1s, sum should be 1024

        CudaSum sum(data.size(), data.data());
        EXPECT_EQ(N, sum.compute_sum());
    }
}