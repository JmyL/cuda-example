#include <cuda_image_morph/cuda_image_morph.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

void print_image(const unsigned char *img, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%d ", img[y * width + x]);
        }
        printf("\n");
    }
}

TEST(CudaImageMorph, Dilates5x5) {
    // 5x5 image: center pixel is 1, rest are 0
    unsigned char input[25] = {
        // clang-format off
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
        // clang-format on
    };
    unsigned char expected[25] = {
        // clang-format off
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0
        // clang-format on
    };
    unsigned char output[25] = {0};

    CudaImageMorph morpher(5, 5, input);
    morpher.dilate(output);

    EXPECT_THAT(output, ElementsAreArray(expected));

    if (::testing::Test::HasFailure()) {
        printf("\nExpected:\n");
        print_image(expected, 5, 5);
        printf("\nActual output:\n");
        print_image(output, 5, 5);
    }
}

TEST(CudaImageMorph, Erodes5x5) {
    // 5x5 image: center pixel is 1, rest are 0
    unsigned char input[25] = {
        // clang-format off
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
        // clang-format on
    };
    unsigned char expected[25] = {
        // clang-format off
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
        // clang-format on
    };
    unsigned char output[25] = {0};

    CudaImageMorph morpher(5, 5, input);
    morpher.erode(output);

    EXPECT_THAT(output, ElementsAreArray(expected));

    if (::testing::Test::HasFailure()) {
        printf("\nExpected:\n");
        print_image(expected, 5, 5);
        printf("\nActual output:\n");
        print_image(output, 5, 5);
    }
}