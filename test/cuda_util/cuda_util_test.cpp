#include <cuda_util/cuda_util.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>

using ::testing::ElementsAre;

TEST(ACudaUtil, PrintsDeviceInfo) { util::print_device_info(); }

TEST(ACudaUtil, ceilsNumberToPowerOfTwo) {
    EXPECT_EQ(32, util::ceilPowerOfTwo(17));
    EXPECT_EQ(32, util::ceilPowerOfTwo(32));
}
