#include <cuda_image_morph/cuda_image_morph.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

#include <sstream>

std::string print_image(const unsigned char *img, int width, int height) {
    std::stringstream ss;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            ss << static_cast<int>(img[y * width + x]) << " ";
        }
        ss << "\n";
    }
    ss << "\n";
    return ss.str();
}

std::string print_image(const unsigned char *img, int width, int height,
                        int center_x, int center_y, int radius) {
    std::stringstream ss;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (center_y + dy < 0 || center_y + dy >= height ||
                center_x + dx < 0 || center_x + dx >= width) {
                continue;
            }
            ss << static_cast<int>(
                      img[(center_y + dy) * width + (center_x + dx)])
               << " ";
        }
        ss << "\n";
    }
    ss << "\n";
    return ss.str();
}

TEST(CudaImageMorph, NaiveDilates5x5) {
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
    morpher.naive_dilate(output);

    EXPECT_THAT(output, ElementsAreArray(expected))
        << "Expected:\n"               //
        << print_image(expected, 5, 5) //
        << "Actual output:\n"          //
        << print_image(output, 5, 5);
}

TEST(CudaImageMorph, NaiveErodes5x5Pattern2) {
    // 5x5 image: center pixel is 1, rest are 0
    unsigned char input[25] = {
        // clang-format off
        1, 1, 1, 0, 0,
        1, 1, 1, 0, 0,
        1, 1, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
        // clang-format on
    };
    unsigned char expected[25] = {
        // clang-format off
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
        // clang-format on
    };
    unsigned char output[25] = {0};

    CudaImageMorph morpher(5, 5, input);
    morpher.naive_erode(output);

    EXPECT_THAT(output, ElementsAreArray(expected))
        << "Expected:\n"               //
        << print_image(expected, 5, 5) //
        << "Actual output:\n"          //
        << print_image(output, 5, 5);
}

TEST(CudaImageMorph, SharedDilates5x5) {
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
    morpher.shared_dilate(output);

    EXPECT_THAT(output, ElementsAreArray(expected))
        << "Expected:\n"               //
        << print_image(expected, 5, 5) //
        << "Actual output:\n"          //
        << print_image(output, 5, 5);
}

TEST(CudaImageMorph, SharedErodes5x5) {
    // 5x5 image: center pixel is 1, rest are 0
    unsigned char input[25] = {
        // clang-format off
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0
        // clang-format on
    };
    unsigned char expected[25] = {
        // clang-format off
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
        // clang-format on
    };
    unsigned char output[25] = {0};

    CudaImageMorph morpher(5, 5, input);
    morpher.shared_erode(output);

    EXPECT_THAT(output, ElementsAreArray(expected))
        << "Expected:\n"               //
        << print_image(expected, 5, 5) //
        << "Actual output:\n"          //
        << print_image(output, 5, 5);
}

TEST(CudaImageMorph, NaiveDialate256x256) {
    // 5x5 image: center pixel is 1, rest are 0
    constexpr auto N = 256;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, N - 2);
    for (auto i = 0; i < 500; ++i) {
        auto x = dis(gen);
        auto y = dis(gen);
        unsigned char input[N * N] = {
            0,
        };
        unsigned char expected[N * N] = {
            0,
        };
        input[y * N + x] = 1;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                expected[(y + dy) * N + (x + dx)] = 1;
            }
        }
        unsigned char output[N * N] = {0};

        CudaImageMorph morpher(N, N, input);
        morpher.naive_dilate(output);

        ASSERT_THAT(output, ElementsAreArray(expected))
            << "Centered at (" << x << ", " << y << ")\n"
            << "Expected:\n"                            //
            << print_image(expected, 256, 256, x, y, 1) //
            << "Actual output:\n"                       //
            << print_image(output, 256, 256, x, y, 1);
    }
}

TEST(CudaImageMorph, NaiveErode256x256) {
    // 5x5 image: center pixel is 1, rest are 0
    constexpr auto N = 256;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, N - 2);
    for (auto i = 0; i < 500; ++i) {
        auto x = dis(gen);
        auto y = dis(gen);
        unsigned char input[N * N] = {
            0,
        };
        unsigned char expected[N * N] = {
            0,
        };
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                input[(y + dy) * N + (x + dx)] = 1;
                if (((y + dy == 0 || y + dy == N - 1) && dx == 0) ||
                    ((x + dx == 0 || x + dx == N - 1) && dy == 0)) {
                    expected[(y + dy) * N + (x + dx)] = 1;
                }
            }
        }
        expected[y * N + x] = 1;
        unsigned char output[N * N] = {0};

        CudaImageMorph morpher(N, N, input);
        morpher.naive_erode(output);

        ASSERT_THAT(output, ElementsAreArray(expected))
            << "Centered at (" << x << ", " << y << ")\n"
            << "Expected:\n"                            //
            << print_image(expected, 256, 256, x, y, 1) //
            << "Actual output:\n"                       //
            << print_image(output, 256, 256, x, y, 1);
    }
}

TEST(CudaImageMorph, SharedDialate256x256) {
    // 5x5 image: center pixel is 1, rest are 0
    constexpr auto N = 256;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, N - 2);
    for (auto i = 0; i < 500; ++i) {
        auto x = dis(gen);
        auto y = dis(gen);
        unsigned char input[N * N] = {
            0,
        };
        unsigned char expected[N * N] = {
            0,
        };
        input[y * N + x] = 1;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                expected[(y + dy) * N + (x + dx)] = 1;
            }
        }
        unsigned char output[N * N] = {0};

        CudaImageMorph morpher(N, N, input);
        morpher.shared_dilate(output);

        ASSERT_THAT(output, ElementsAreArray(expected))
            << "Centered at (" << x << ", " << y << ")\n"
            << "Expected:\n"                            //
            << print_image(expected, 256, 256, x, y, 1) //
            << "Actual output:\n"                       //
            << print_image(output, 256, 256, x, y, 1);
    }
}

TEST(CudaImageMorph, SharedErode256x256) {
    // 5x5 image: center pixel is 1, rest are 0
    constexpr auto N = 256;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, N - 2);
    for (auto i = 0; i < 500; ++i) {
        auto x = dis(gen);
        auto y = dis(gen);
        unsigned char input[N * N] = {
            0,
        };
        unsigned char expected[N * N] = {
            0,
        };
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                input[(y + dy) * N + (x + dx)] = 1;
                if (((y + dy == 0 || y + dy == N - 1) && dx == 0) ||
                    ((x + dx == 0 || x + dx == N - 1) && dy == 0)) {
                    expected[(y + dy) * N + (x + dx)] = 1;
                }
            }
        }
        expected[y * N + x] = 1;
        unsigned char output[N * N] = {0};

        CudaImageMorph morpher(N, N, input);
        morpher.shared_erode(output);

        ASSERT_THAT(output, ElementsAreArray(expected))
            << "Centered at (" << x << ", " << y << ")\n"
            << "Expected:\n"                            //
            << print_image(expected, 256, 256, x, y, 1) //
            << "Actual output:\n"                       //
            << print_image(output, 256, 256, x, y, 1);
    }
}