/* -*- c++ -*- */
/*
 * Copyright 2025 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_test.h"
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <chrono>

class volk_32fc_s32fc_x2_rotator2_32fc_test : public VolkTest
{
protected:
    void SetUp() override
    {
        initialize_test(GetParam());
        initialize_data(vector_length);
    }

    void initialize_data(const size_t length)
    {
        // Tolerance scales with sample count because all complex-multiplication-based
        // rotators accumulate floating-point error. The error grows roughly as O(sqrt(N))
        // due to random walk behavior of rounding errors.
        //
        // For implementations using inc^K (K=2 for SSE, K=4 for AVX, K=8 for AVX512),
        // the error per step is larger, leading to faster divergence.
        if (length < 16) {
            absolute_error = 1.e-6;
        } else if (length < 128) {
            absolute_error = 1.e-5;
        } else if (length < 65536) {
            absolute_error = 1.e-4;
        } else if (length < 1000000) {
            absolute_error = 2.e-3;
        } else if (length < 10000000) {
            // 1M-10M samples: relax tolerance for accumulated error
            absolute_error = 2.e-2;
        } else {
            // 10M+ samples: At this scale, error exceeds useful tolerance
            // Applications requiring accuracy at this scale should use
            // implementations with periodic angle-based resync
            absolute_error = 5.e-2;
        }

        vector_length = length;
        input = volk::vector<lv_32fc_t>(length);
        result = volk::vector<lv_32fc_t>(length);
        result_magnitude = volk::vector<float>(length);

        const float initial_phase = initial_phase_steps * increment;
        phase_increment = std::polar(1.0f, increment);
        phase = std::polar(1.0f, initial_phase);

        for (size_t i = 0; i < length; ++i) {
            input[i] =
                std::complex<float>(2.0f * std::cos(2.0f * M_PI * i / length),
                                    2.0f * std::sin(0.3f + 2.0f * M_PI * i / length));
        }

        // Calculate expected results using double precision for angle computation
        // This is critical for numerical stability at high sample counts.
        // Using float would lose precision for i > ~10M samples.
        expected = volk::vector<lv_32fc_t>(length);
        const double initial_phase_d = static_cast<double>(initial_phase);
        const double increment_d = static_cast<double>(increment);
        for (size_t i = 0; i < length; ++i) {
            double angle = initial_phase_d + static_cast<double>(i) * increment_d;
            // Reduce angle to [-π, π] for sincos accuracy at large angles
            angle = std::fmod(angle, 2.0 * M_PI);
            if (angle > M_PI)
                angle -= 2.0 * M_PI;
            else if (angle < -M_PI)
                angle += 2.0 * M_PI;
            expected[i] = input[i] * std::polar(1.0f, static_cast<float>(angle));
        }

        expected_magnitude = volk::vector<float>(length);
        for (size_t i = 0; i < length; ++i) {
            expected_magnitude[i] = std::abs(input[i]);
        }

        // This is a hacky solution to have unaligned tests.
        ua_result = result;
        ua_result.at(0) = expected.at(0);
    }

    void execute_aligned(const std::string impl_name)
    {
        volk_32fc_s32fc_x2_rotator2_32fc_manual(result.data(),
                                                input.data(),
                                                &phase_increment,
                                                &phase,
                                                vector_length,
                                                impl_name.c_str());

        for (size_t i = 0; i < vector_length; ++i) {
            result_magnitude[i] = std::abs(result[i]);
        }
        EXPECT_TRUE(AreFloatingPointArraysEqualWithAbsoluteError(
            expected_magnitude, result_magnitude, absolute_magnitue_error));
        EXPECT_TRUE(AreComplexFloatingPointArraysEqualWithAbsoluteError(
            expected, result, absolute_error));
    }

    void execute_unaligned(const std::string impl_name)
    {
        // Use double precision for phase computation
        double unaligned_angle = static_cast<double>(initial_phase_steps + 1.0f) *
                                 static_cast<double>(increment);
        lv_32fc_t unaligned_phase = std::polar(1.0f, static_cast<float>(unaligned_angle));
        volk_32fc_s32fc_x2_rotator2_32fc_manual(ua_result.data() + 1,
                                                input.data() + 1,
                                                &phase_increment,
                                                &unaligned_phase,
                                                vector_length - 1,
                                                impl_name.c_str());
        for (size_t i = 0; i < vector_length; ++i) {
            result_magnitude[i] = std::abs(ua_result[i]);
        }
        result_magnitude[0] = expected_magnitude[0];

        EXPECT_TRUE(AreFloatingPointArraysEqualWithAbsoluteError(
            expected_magnitude, result_magnitude, absolute_magnitue_error));
        EXPECT_TRUE(AreComplexFloatingPointArraysEqualWithAbsoluteError(
            expected, ua_result, absolute_error));
    }

    static constexpr float increment = 0.07f;
    static constexpr float initial_phase_steps = 0.0f;
    static constexpr float absolute_magnitue_error = 1.0e-4;
    float absolute_error{};
    volk::vector<lv_32fc_t> input;
    volk::vector<lv_32fc_t> result;
    lv_32fc_t phase_increment;
    lv_32fc_t phase;
    volk::vector<lv_32fc_t> expected;
    volk::vector<float> expected_magnitude;
    volk::vector<lv_32fc_t> ua_result;
    volk::vector<float> result_magnitude;
};

TEST_P(volk_32fc_s32fc_x2_rotator2_32fc_test, run)
{
    fmt::print("test {} implementation: {:>12}, size={} ...",
               is_aligned_implementation ? "aligned" : "unaligned",
               implementation_name,
               vector_length);
    auto start = std::chrono::steady_clock::now();

    if (is_aligned_implementation) {
        execute_aligned(implementation_name);
    } else {
        execute_unaligned(implementation_name);
    }

    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    fmt::print("\tduration={}\n", elapsed);
}

INSTANTIATE_TEST_SUITE_P(
    volk_32fc_s32fc_x2_rotator2_32fc,
    volk_32fc_s32fc_x2_rotator2_32fc_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32fc_s32fc_x2_rotator2_32fc_get_func_desc())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

// Large-scale tests for numerical stability verification
// Note: rotator2 implementations using complex multiplication (phase *= inc^N)
// accumulate floating-point error that grows with sample count.
// At 2M samples, error is typically ~1-2% for AVX (inc^4) implementations.
// For >10M samples, consider using implementations with periodic angle resync.
static constexpr std::array<size_t, 1> large_vector_sizes{ 2000000 };

INSTANTIATE_TEST_SUITE_P(
    volk_32fc_s32fc_x2_rotator2_32fc_large,
    volk_32fc_s32fc_x2_rotator2_32fc_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32fc_s32fc_x2_rotator2_32fc_get_func_desc())),
                     testing::ValuesIn(large_vector_sizes)),
    generate_volk_test_name());
