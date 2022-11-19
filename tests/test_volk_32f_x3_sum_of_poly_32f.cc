/* -*- c++ -*- */
/*
 * Copyright 2022 Johannes Demel
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


class volk_32f_x3_sum_of_poly_32f_test : public VolkTest
{
protected:
    void SetUp() override
    {
        initialize_implementation_names(volk_32f_x3_sum_of_poly_32f_get_func_desc());
        initialize_data(GetParam());
    }

    void initialize_data(const size_t length)
    {
        vector_length = length;

        vec0 = volk::vector<float>(length);
        for (size_t i = 0; i < length; ++i) {
            vec0[i] = float(2.8f + i * 0.14f);
        }

        ua_vec0 = volk::vector<float>({ 0.0f });
        for (auto v : vec0) {
            ua_vec0.push_back(v);
        }

        center_points = volk::vector<float>({ 4.4, 2.1, 0.3, 0.05, 4.1 });
        ua_center_points = volk::vector<float>({
            0.0,
        });
        for (auto v : center_points) {
            ua_center_points.push_back(v);
        }

        cutoff = volk::vector<float>({ -1.5 });
        ua_cutoff = cutoff;
        ua_cutoff.push_back(cutoff.at(0));

        expected = 0.0f;
        for (auto value : vec0) {
            value = std::max(value, cutoff.at(0));
            auto sq = value * value;
            auto cube = sq * value;
            auto quartic = value * cube;
            expected += value * center_points[0] + sq * center_points[1] +
                        cube * center_points[2] + quartic * center_points[3];
        }
        expected += center_points[4] * float(length);

        result = volk::vector<float>(1, 0.0);
        ua_result.push_back(result.at(0));
        ua_result.push_back(result.at(0));
    }

    void execute_aligned(const std::string impl_name)
    {
        volk_32f_x3_sum_of_poly_32f_manual(result.data(),
                                           vec0.data(),
                                           center_points.data(),
                                           cutoff.data(),
                                           vector_length,
                                           impl_name.c_str());
    }

    void execute_unaligned(const std::string impl_name)
    {
        volk_32f_x3_sum_of_poly_32f_manual(ua_result.data() + 1,
                                           ua_vec0.data() + 1,
                                           ua_center_points.data() + 1,
                                           ua_cutoff.data() + 1,
                                           vector_length,
                                           impl_name.c_str());
    }

    // void TearDown() override {}
    size_t vector_length;
    volk::vector<float> vec0;
    volk::vector<float> ua_vec0;
    volk::vector<float> center_points;
    volk::vector<float> ua_center_points;
    volk::vector<float> cutoff;
    volk::vector<float> ua_cutoff;
    volk::vector<float> result;
    volk::vector<float> ua_result;
    float expected = 0.0f;
};


TEST_P(volk_32f_x3_sum_of_poly_32f_test, aligned)
{
    for (auto name : implementation_names) {
        auto tol = std::max(expected * 1e-5, 1e-5);
        fmt::print(
            "test aligned implementation: {:>12}, size={} ...", name, vector_length);
        auto start = std::chrono::steady_clock::now();

        execute_aligned(name);

        std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        fmt::print("\tduration={}\n", elapsed);
        EXPECT_NEAR(result.at(0), expected, tol);
    }
}

TEST_P(volk_32f_x3_sum_of_poly_32f_test, unaligned)
{
    for (auto name : unaligned_impl_names) {
        auto tol = std::max(expected * 1e-5, 1e-5);
        fmt::print(
            "test unaligned implementation: {:>12}, size={} ...", name, vector_length);
        auto start = std::chrono::steady_clock::now();

        execute_unaligned(name);

        std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        fmt::print("\tduration={}\n", elapsed);
        EXPECT_NEAR(ua_result.at(1), expected, tol);
    }
}


INSTANTIATE_TEST_SUITE_P(volk_32f_x3_sum_of_poly_32f,
                         volk_32f_x3_sum_of_poly_32f_test,
                         //  testing::Values(8, 32)
                         testing::Values(7, 32, 128, 1023, 65535, 131071));
