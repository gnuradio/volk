/* -*- c++ -*- */
/*
 * Copyright 2022 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <volk/volk.h>
#include <algorithm>
#include <array>
#include <complex>
#include <optional>
#include <span>
#include <tuple>

static constexpr std::array<size_t, 5> default_vector_sizes{ 7, 32, 128, 1023, 131071 };
std::vector<std::string> get_kernel_implementation_name_list(const volk_func_desc_t desc);

bool is_aligned_implementation_name(const std::string& name);

std::tuple<std::vector<std::string>, std::vector<std::string>>
separate_implementations_by_alignment(const std::vector<std::string>& names);

std::vector<std::string>
get_aligned_kernel_implementation_names(const volk_func_desc_t desc);
std::vector<std::string>
get_unaligned_kernel_implementation_names(const volk_func_desc_t desc);

struct generate_volk_test_name {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
    {
        return fmt::format("{}_{}", std::get<0>(info.param), std::get<1>(info.param));
    }
};

template <typename Parameter = std::tuple<std::string, size_t>>
class VolkTestImpl : public ::testing::TestWithParam<Parameter>
{
protected:
    void initialize_test(const Parameter& param)
    {
        implementation_name = std::get<0>(param);
        vector_length = std::get<1>(param);
        is_aligned_implementation = is_aligned_implementation_name(implementation_name);
    }

    std::string implementation_name;
    bool is_aligned_implementation;
    size_t vector_length;
};

using VolkTest = VolkTestImpl<>;

template <typename T>
struct is_complex : std::false_type {
};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {
};

template <typename T>
::testing::AssertionResult AreIntegerArraysEqual(std::span<const T> expected,
                                                 std::span<const T> actual)
{
    ::testing::AssertionResult result = ::testing::AssertionFailure();
    if (expected.size() != actual.size()) {
        return result << "expected result size=" << expected.size()
                      << " differs from actual size=" << actual.size();
    }

    int errors_found = 0;
    const char* separator = " ";
    for (size_t index = 0; index < expected.size(); ++index) {
        if (expected[index] != actual[index]) {
            if (errors_found == 0) {
                result << "Differences found:";
            }
            if (errors_found < 3) {
                result << separator << expected[index] << " != " << actual[index] << " @ "
                       << index;
                separator = ",\n";
            }
            ++errors_found;
        }
    }
    if (errors_found > 0) {
        result << separator << errors_found << " differences in total";
        return result;
    }
    return ::testing::AssertionSuccess();
}


template <typename T>
::testing::AssertionResult
AreComplexFloatingPointArraysAlmostEqual(std::span<const T> expected,
                                         std::span<const T> actual)
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
            not expected_imag.AlmostEquals(actual_imag)) {
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

template <typename T>
::testing::AssertionResult
AreComplexFloatingPointArraysEqualWithAbsoluteError(std::span<const T> expected,
                                                    std::span<const T> actual,
                                                    const float absolute_error = 1.0e-7)
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
        if (expected_real.is_nan() or actual_real.is_nan() or expected_imag.is_nan() or
            actual_imag.is_nan() or
            std::abs(expected[index].real() - actual[index].real()) > absolute_error or
            std::abs(expected[index].imag() - actual[index].imag()) > absolute_error) {
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

template <typename T>
::testing::AssertionResult
AreFloatingPointArraysEqualWithAbsoluteError(std::span<const T> expected,
                                             std::span<const T> actual,
                                             const float absolute_error = 1.0e-7)
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
        auto expected_value = ::testing::internal::FloatingPoint(expected[index]);
        auto actual_value = ::testing::internal::FloatingPoint(actual[index]);
        if (expected_value.is_nan() or actual_value.is_nan() or
            std::abs(expected[index] - actual[index]) > absolute_error) {
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
