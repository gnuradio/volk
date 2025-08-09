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
        // Be stricter for smaller vectors. Error accumulate slowly!
        if (length < 16) {
            absolute_error = 10.e-7;
        } else if (length < 128) {
            absolute_error = 10.e-6;
        } else if (length < 65536) {
            absolute_error = 10.e-5;
        } else {
            absolute_error = 10.e-3;
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

        // Calculate expected results
        expected = volk::vector<lv_32fc_t>(length);
        for (size_t i = 0; i < length; ++i) {
            expected[i] =
                input[i] *
                std::polar(1.0f, initial_phase + static_cast<float>(i) * increment);
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
        lv_32fc_t unaligned_phase =
            std::polar(1.0f, (initial_phase_steps + 1.0f) * increment);
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
