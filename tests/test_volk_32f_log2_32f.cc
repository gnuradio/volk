/* -*- c++ -*- */
/*
 * Copyright 2026 Johannes Demel
 *
 * This file is part of VOLK.
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_test.h"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>

#include <string_view>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <span>

namespace {
constexpr float ABSOLUTE_LOG_TEST_ERROR = 1e-5f;
constexpr std::array LOG2_EDGE_CASE_VALUES{
    -1.0f,
    0.0f,
    -0.0f,
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::quiet_NaN(),
    65536.0f,
    0.125f,
    0x1.fffffep-4f, // std::nextafter(0.125f, 0.0f)
    0x1.000002p-3f, // std::nextafter(0.125f, 1.0f)
    0.124962f,
    std::numeric_limits<float>::min(),
    1.0f,
    2.0f,
};

void log2_reference(std::span<float> output, std::span<const float> input)
{
    std::transform(input.begin(), input.end(), output.begin(), [](const float value) {
        const double result = std::log2(static_cast<double>(value));
        if (std::isnan(result)) {
            return static_cast<float>(result);
        }
        return std::isinf(result) ? std::copysign(127.0f, static_cast<float>(result))
                                  : static_cast<float>(result);
    });
}

} // namespace

class volk_32f_log2_32f_test : public VolkTest
{
protected:
    void SetUp() override
    {
        initialize_test(GetParam());
        const size_t offset = is_aligned_implementation ? 0 : 1;

        input_storage.resize(vector_length + offset);
        expected_storage.resize(vector_length + offset);
        result_storage.resize(vector_length + offset);

        input = std::span(input_storage).subspan(offset, vector_length);
        expected = std::span(expected_storage).subspan(offset, vector_length);
        result = std::span(result_storage).subspan(offset, vector_length);

        std::mt19937 generator(0xDEADBEEF);
        std::uniform_int_distribution<int> exponent_distribution(-126, 126);
        std::uniform_real_distribution<float> significand_distribution(1.0f, 2.0f);
        std::generate(input.begin(), input.end(), [&] {
            return std::ldexp(significand_distribution(generator),
                              exponent_distribution(generator));
        });

        if (input.size() >= 1000) {
            std::copy(LOG2_EDGE_CASE_VALUES.begin(),
                      LOG2_EDGE_CASE_VALUES.end(),
                      input.begin() + 100);
        }

        log2_reference(expected, input);
    }

    void run(std::span<float> output,
             std::span<const float> values,
             const std::string_view implementation) const
    {
        volk_32f_log2_32f_manual(output.data(),
                                 values.data(),
                                 static_cast<unsigned int>(values.size()),
                                 implementation.data());
    }

    volk::vector<float> input_storage;
    volk::vector<float> expected_storage;
    volk::vector<float> result_storage;
    std::span<float> input;
    std::span<float> expected;
    std::span<float> result;
};

TEST_P(volk_32f_log2_32f_test, run)
{
    run(result, input, implementation_name);
    EXPECT_TRUE(AreFloatingPointArraysEqualWithAbsoluteError(
        expected, result, ABSOLUTE_LOG_TEST_ERROR));
}

INSTANTIATE_TEST_SUITE_P(
    volk_32f_log2_32f,
    volk_32f_log2_32f_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         volk_32f_log2_32f_get_func_desc())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());
