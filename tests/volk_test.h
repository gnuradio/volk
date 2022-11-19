/* -*- c++ -*- */
/*
 * Copyright 2022 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <volk/volk.h>
#include <tuple>


std::vector<std::string> get_kernel_implementation_name_list(volk_func_desc_t desc);

std::tuple<std::vector<std::string>, std::vector<std::string>>
separate_implementations_by_alignment(std::vector<std::string> names);

class VolkTest : public ::testing::TestWithParam<int>
{
protected:
    void initialize_implementation_names(volk_func_desc_t desc)
    {
        implementation_names = get_kernel_implementation_name_list(desc);
        std::tie(aligned_impl_names, unaligned_impl_names) =
            separate_implementations_by_alignment(implementation_names);
    }

    std::vector<std::string> implementation_names;
    std::vector<std::string> aligned_impl_names;
    std::vector<std::string> unaligned_impl_names;
};


template <class T>
::testing::AssertionResult AreComplexFloatingPointArraysAlmostEqual(const T& expected,
                                                                    const T& actual)
{
    ::testing::AssertionResult result = ::testing::AssertionFailure();
    if (expected.size() != actual.size()) {
        return result << "expected result size=" << expected.size()
                      << " differs from actual size=" << actual.size();
    }
    const unsigned long length = expected.size();

    int errorsFound = 0;
    const char* separator = " ";
    for (unsigned long index = 0; index < length; index++) {
        auto expected_real = ::testing::internal::FloatingPoint(expected[index].real());
        auto expected_imag = ::testing::internal::FloatingPoint(expected[index].imag());
        auto actual_real = ::testing::internal::FloatingPoint(actual[index].real());
        auto actual_imag = ::testing::internal::FloatingPoint(actual[index].imag());
        if (not expected_real.AlmostEquals(actual_real) or
            not expected_imag.AlmostEquals(actual_imag))

        {
            if (errorsFound == 0) {
                result << "Differences found:";
            }
            if (errorsFound < 3) {
                result << separator << expected[index] << " != " << actual[index] << " @ "
                       << index;
                separator = ",\n";
            }
            errorsFound++;
        }
    }
    if (errorsFound > 0) {
        result << separator << errorsFound << " differences in total";
        return result;
    }
    return ::testing::AssertionSuccess();
}
