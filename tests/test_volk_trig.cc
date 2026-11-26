/* -*- c++ -*- */
/*
 * This file is part of VOLK.
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_test.h"

#include <gtest/gtest.h>
#include <volk/volk.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>

namespace {

struct TrigKernel {
    decltype(&volk_32f_sin_32f_manual) run;
    volk_func_desc_t descriptor;
    float (*reference)(float);
};

TEST(VolkTrig, HalfWaveRangeReductionAndFallback)
{
    constexpr float infinity = std::numeric_limits<float>::infinity();
    constexpr float pi = std::numbers::pi_v<float>;
    const std::array kernels{
        TrigKernel{ volk_32f_sin_32f_manual, volk_32f_sin_32f_get_func_desc(), std::sin },
        TrigKernel{ volk_32f_cos_32f_manual, volk_32f_cos_32f_get_func_desc(), std::cos },
    };

    bool tested = false;
    for (const auto& kernel : kernels) {
        for (const auto& impl : get_kernel_implementation_name_list(kernel.descriptor)) {
            if (impl != "neon" && impl != "neonv8" && !impl.ends_with("_avx2_fma") &&
                !impl.ends_with("_avx512f")) {
                continue;
            }
            tested = true;
            const float cutoff = impl == "neon" ? 0x1p12f : 0x1p20f;
            const std::array fast_angles{ 0.0f,          -0.0f,  pi,      -pi / 2,
                                          0x1.921f8p+0f, cutoff, -cutoff, -0.75f };
            const std::array fallback_angles{
                std::nextafter(cutoff, infinity),
                std::nextafter(-cutoff, -infinity),
                std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max(),
                infinity,
                -infinity,
                std::numeric_limits<float>::quiet_NaN(),
                1.0f,
            };
            const unsigned int offset = is_aligned_implementation_name(impl) ? 0 : 1;
            constexpr unsigned int chunk = 16;
            constexpr unsigned int count = 3 * chunk + 3;
            alignas(64) std::array<float, count + 1> input{};
            alignas(64) std::array<float, count + 1> output{};
            for (unsigned int i = 0; i < count; ++i) {
                input[offset + i] = i / chunk == 1
                                        ? fallback_angles[i % fallback_angles.size()]
                                        : fast_angles[i % fast_angles.size()];
            }

            kernel.run(
                output.data() + offset, input.data() + offset, count, impl.c_str());
            for (unsigned int i = 0; i < count; ++i) {
                const float angle = input[offset + i];
                const float expected = kernel.reference(angle);
                SCOPED_TRACE(testing::Message() << impl << " angle " << angle);
                if (std::isnan(expected)) {
                    EXPECT_TRUE(std::isnan(output[offset + i]));
                } else {
                    EXPECT_NEAR(output[offset + i], expected, 2.4e-7f);
                    EXPECT_LE(std::abs(output[offset + i]), 1.0f);
                }
            }
        }
    }
    if (!tested) {
        GTEST_SKIP() << "No supported half-wave SIMD implementations";
    }
}

TEST(VolkTrig, NeonSincosQuadrants)
{
    constexpr float pi = std::numbers::pi_v<float>;
    const std::array input{
        0.0f,  pi / 2,      pi,    3 * pi / 2,     2 * pi,     -pi / 2,
        -pi,   -3 * pi / 2, 0.25f, pi / 2 + 0.25f, pi + 0.25f, 3 * pi / 2 + 0.25f,
        -2.0f,
    };
    bool tested = false;
    for (const auto& impl :
         get_kernel_implementation_name_list(volk_32f_sincos_32f_x2_get_func_desc())) {
        if (impl != "neon" && impl != "neonv8") {
            continue;
        }
        tested = true;
        std::array<float, input.size()> sines{};
        std::array<float, input.size()> cosines{};
        volk_32f_sincos_32f_x2_manual(
            sines.data(), cosines.data(), input.data(), input.size(), impl.c_str());
        for (size_t i = 0; i < input.size(); ++i) {
            SCOPED_TRACE(testing::Message() << impl << " angle " << input[i]);
            EXPECT_NEAR(sines[i], std::sin(input[i]), 2.4e-7f);
            EXPECT_NEAR(cosines[i], std::cos(input[i]), 2.4e-7f);
        }
    }
    if (!tested) {
        GTEST_SKIP() << "No NEON implementations";
    }
}

} // namespace
