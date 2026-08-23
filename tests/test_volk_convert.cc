/* -*- c++ -*- */
/*
 * Copyright 2026 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "volk_test.h"
#include <gtest/gtest.h>
#include <string_view>
#include <type_traits>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>

namespace {

template <typename T>
T random_scalar(std::mt19937& generator)
{
    if constexpr (std::is_integral_v<T>) {
        return std::uniform_int_distribution<T>(std::numeric_limits<T>::lowest(),
                                                std::numeric_limits<T>::max())(generator);
    } else {
        return static_cast<T>(std::uniform_real_distribution<long double>(
            static_cast<long double>(std::numeric_limits<T>::lowest()),
            static_cast<long double>(std::numeric_limits<T>::max()))(generator));
    }
}

template <typename T>
T random_value(std::mt19937& generator)
{
    if constexpr (is_complex<T>::value) {
        return { random_scalar<typename T::value_type>(generator),
                 random_scalar<typename T::value_type>(generator) };
    } else {
        return random_scalar<T>(generator);
    }
}

template <typename Output>
auto random_narrowing_float(std::mt19937& generator)
{
    if constexpr (is_complex<Output>::value) {
        const long double standard_deviation = static_cast<long double>(
            std::numeric_limits<typename Output::value_type>::max());
        auto distribution =
            std::normal_distribution<long double>(0.0L, standard_deviation);
        return lv_32fc_t{ static_cast<float>(distribution(generator)),
                          static_cast<float>(distribution(generator)) };
    } else {
        const long double standard_deviation =
            static_cast<long double>(std::numeric_limits<Output>::max());
        return static_cast<float>(
            std::normal_distribution<long double>(0.0L, standard_deviation)(generator));
    }
}

template <typename T>
T rounded_saturated(float value)
{
    const auto rounded = static_cast<double>(std::rint(value));
    return static_cast<T>(
        std::clamp(rounded,
                   static_cast<double>(std::numeric_limits<T>::lowest()),
                   static_cast<double>(std::numeric_limits<T>::max())));
}

template <typename Input, typename Output>
struct KernelSpec {
    using input_type = Input;
    using output_type = Output;

    static input_type random_input(std::mt19937& generator)
    {
        return random_value<input_type>(generator);
    }
};

static constexpr std::array<float, 2> scaling_values{ 0.5f, 2.0f };
static constexpr std::array<float, 2> bias_values{ 0.0f, 128.0f };

struct generate_volk_test_scaling_name : generate_volk_test_name {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
    {
        static_assert(std::tuple_size_v<ParamType> >= 3);
        std::string scale_name = fmt::format("{:.2f}", std::get<2>(info.param));
        std::replace(scale_name.begin(), scale_name.end(), '.', '_');
        return fmt::format(
            "{}_scale_{}", generate_volk_test_name::operator()(info), scale_name);
    }
};

struct generate_volk_test_scaling_bias_name : generate_volk_test_scaling_name {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
    {
        static_assert(std::tuple_size_v<ParamType> >= 4);
        std::string bias_name = fmt::format("{:.2f}", std::get<3>(info.param));
        std::replace(bias_name.begin(), bias_name.end(), '.', '_');
        return fmt::format(
            "{}_bias_{}", generate_volk_test_scaling_name::operator()(info), bias_name);
    }
};

struct Volk8iConvert16i : KernelSpec<int8_t, int16_t> {
    static constexpr int KERNEL_SCALING_VALUE = 256;
    static output_type expected(const input_type value)
    {
        return static_cast<int16_t>(value) * KERNEL_SCALING_VALUE;
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl)
    {
        volk_8i_convert_16i_manual(out.data(), in.data(), in.size(), impl.data());
    }
    static volk_func_desc_t descriptor() { return volk_8i_convert_16i_get_func_desc(); }
};

struct Volk16iConvert8i : KernelSpec<int16_t, int8_t> {
    static output_type expected(const input_type value)
    {
        return static_cast<int8_t>(value >> 8);
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl)
    {
        volk_16i_convert_8i_manual(out.data(), in.data(), in.size(), impl.data());
    }
    static volk_func_desc_t descriptor() { return volk_16i_convert_8i_get_func_desc(); }
};

struct Volk16icConvert32fc : KernelSpec<lv_16sc_t, lv_32fc_t> {
    static output_type expected(const input_type value)
    {
        return { static_cast<float>(value.real()), static_cast<float>(value.imag()) };
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl)
    {
        volk_16ic_convert_32fc_manual(out.data(), in.data(), in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_16ic_convert_32fc_get_func_desc();
    }
};

struct Volk32fConvert64f : KernelSpec<float, double> {
    static output_type expected(const input_type value) { return value; }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl)
    {
        volk_32f_convert_64f_manual(out.data(), in.data(), in.size(), impl.data());
    }
    static volk_func_desc_t descriptor() { return volk_32f_convert_64f_get_func_desc(); }
};

struct Volk64fConvert32f : KernelSpec<double, float> {
    static output_type expected(const input_type value)
    {
        return static_cast<float>(value);
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl)
    {
        volk_64f_convert_32f_manual(out.data(), in.data(), in.size(), impl.data());
    }
    static volk_func_desc_t descriptor() { return volk_64f_convert_32f_get_func_desc(); }
};

struct Volk32fS32fConvert16i : KernelSpec<float, int16_t> {
    static input_type random_input(std::mt19937& generator)
    {
        return random_narrowing_float<output_type>(generator);
    }

    static output_type expected(const input_type value, const float scale)
    {
        return rounded_saturated<int16_t>(value * scale);
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl,
                    const float scale)
    {
        volk_32f_s32f_convert_16i_manual(
            out.data(), in.data(), scale, in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_32f_s32f_convert_16i_get_func_desc();
    }
};

struct Volk32fS32fConvert32i : KernelSpec<float, int32_t> {
    static input_type random_input(std::mt19937& generator)
    {
        return random_narrowing_float<output_type>(generator);
    }

    static output_type expected(const input_type value, const float scale)
    {
        return rounded_saturated<int32_t>(value * scale);
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl,
                    const float scale)
    {
        volk_32f_s32f_convert_32i_manual(
            out.data(), in.data(), scale, in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_32f_s32f_convert_32i_get_func_desc();
    }
};

struct Volk32fS32fConvert8i : KernelSpec<float, int8_t> {
    static input_type random_input(std::mt19937& generator)
    {
        return random_narrowing_float<output_type>(generator);
    }

    static output_type expected(const input_type value, const float scale)
    {
        return rounded_saturated<int8_t>(value * scale);
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl,
                    const float scale)
    {
        volk_32f_s32f_convert_8i_manual(
            out.data(), in.data(), scale, in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_32f_s32f_convert_8i_get_func_desc();
    }
};

struct Volk32iS32fConvert32f : KernelSpec<int32_t, float> {
    static output_type expected(const input_type value, const float scale)
    {
        return static_cast<float>(value) / scale;
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl,
                    const float scale)
    {
        volk_32i_s32f_convert_32f_manual(
            out.data(), in.data(), scale, in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_32i_s32f_convert_32f_get_func_desc();
    }
};

struct Volk8iS32fConvert32f : KernelSpec<int8_t, float> {
    static output_type expected(const input_type value, const float scale)
    {
        return static_cast<float>(value) / scale;
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl,
                    const float scale)
    {
        volk_8i_s32f_convert_32f_manual(
            out.data(), in.data(), scale, in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_8i_s32f_convert_32f_get_func_desc();
    }
};

struct Volk32fcConvert16ic : KernelSpec<lv_32fc_t, lv_16sc_t> {
    static input_type random_input(std::mt19937& generator)
    {
        return random_narrowing_float<output_type>(generator);
    }

    static output_type expected(const input_type value)
    {
        return { rounded_saturated<int16_t>(value.real()),
                 rounded_saturated<int16_t>(value.imag()) };
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl)
    {
        volk_32fc_convert_16ic_manual(out.data(), in.data(), in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_32fc_convert_16ic_get_func_desc();
    }
};

struct Volk16iS32fConvert32f : KernelSpec<int16_t, float> {
    static output_type expected(const input_type value, const float scale)
    {
        return static_cast<float>(value) / scale;
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl,
                    const float scale)
    {
        volk_16i_s32f_convert_32f_manual(
            out.data(), in.data(), scale, in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_16i_s32f_convert_32f_get_func_desc();
    }
};

struct Volk32fS32fX2Convert8u : KernelSpec<float, uint8_t> {
    static input_type random_input(std::mt19937& generator)
    {
        return random_narrowing_float<output_type>(generator);
    }

    static output_type
    expected(const input_type value, const float scale, const float bias)
    {
        return rounded_saturated<uint8_t>(value * scale + bias);
    }
    static void run(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view impl,
                    const float scale,
                    const float bias)
    {
        volk_32f_s32f_x2_convert_8u_manual(
            out.data(), in.data(), scale, bias, in.size(), impl.data());
    }
    static volk_func_desc_t descriptor()
    {
        return volk_32f_s32f_x2_convert_8u_get_func_desc();
    }
};

template <size_t ParameterCount, typename Kernel, typename Parameter>
struct ConvertTestFunctions;

template <typename Kernel, typename Parameter>
struct ConvertTestFunctions<2, Kernel, Parameter> {
    static typename Kernel::output_type expected(const typename Kernel::input_type value,
                                                 const Parameter&)
    {
        return Kernel::expected(value);
    }

    static void run(std::span<typename Kernel::output_type> out,
                    std::span<const typename Kernel::input_type> in,
                    const std::string_view implementation,
                    const Parameter&)
    {
        Kernel::run(out, in, implementation);
    }
};

template <typename Kernel, typename Parameter>
struct ConvertTestFunctions<3, Kernel, Parameter> {
    static typename Kernel::output_type expected(const typename Kernel::input_type value,
                                                 const Parameter& parameter)
    {
        return Kernel::expected(value, std::get<2>(parameter));
    }

    static void run(std::span<typename Kernel::output_type> out,
                    std::span<const typename Kernel::input_type> in,
                    const std::string_view implementation,
                    const Parameter& parameter)
    {
        Kernel::run(out, in, implementation, std::get<2>(parameter));
    }
};

template <typename Kernel, typename Parameter>
struct ConvertTestFunctions<4, Kernel, Parameter> {
    static typename Kernel::output_type expected(const typename Kernel::input_type value,
                                                 const Parameter& parameter)
    {
        return Kernel::expected(value, std::get<2>(parameter), std::get<3>(parameter));
    }

    static void run(std::span<typename Kernel::output_type> out,
                    std::span<const typename Kernel::input_type> in,
                    const std::string_view implementation,
                    const Parameter& parameter)
    {
        Kernel::run(
            out, in, implementation, std::get<2>(parameter), std::get<3>(parameter));
    }
};

template <typename Kernel, typename Parameter = std::tuple<std::string, size_t>>
class ConvertTest : public VolkTestImpl<Parameter>
{
protected:
    using parameter_type = Parameter;
    using input_type = typename Kernel::input_type;
    using output_type = typename Kernel::output_type;

    void SetUp() override
    {
        this->initialize_test(this->GetParam());
        input.resize(this->vector_length);
        expected.resize(this->vector_length);
        result.resize(this->vector_length);
        unaligned_input.resize(this->vector_length + 1);
        unaligned_result.resize(this->vector_length + 1);
        std::mt19937 generator{ 0x424242u };
        std::generate(
            input.begin(), input.end(), [&] { return Kernel::random_input(generator); });
        std::transform(input.begin(),
                       input.end(),
                       expected.begin(),
                       [&](const input_type value) { return expected_value(value); });
        std::copy(input.begin(), input.end(), unaligned_input.begin() + 1);
    }

    void run(const std::string_view implementation)
    {
        const std::span<const input_type> in =
            this->is_aligned_implementation ? std::span{ input }
                                            : std::span{ unaligned_input }.subspan(1);
        const std::span<output_type> out = this->is_aligned_implementation
                                               ? std::span{ result }
                                               : std::span{ unaligned_result }.subspan(1);
        run_kernel(out, in, implementation);

        if constexpr (std::is_floating_point_v<output_type>) {
            EXPECT_TRUE(
                AreFloatingPointArraysEqualWithAbsoluteError<output_type>(expected, out));
        } else if constexpr (requires { expected.front().real(); }) {
            if constexpr (std::is_floating_point_v<decltype(expected.front().real())>) {
                EXPECT_TRUE(
                    AreComplexFloatingPointArraysAlmostEqual<output_type>(expected, out));
            } else {
                EXPECT_TRUE(AreIntegerArraysEqual<output_type>(expected, out));
            }
        } else {
            EXPECT_TRUE(AreIntegerArraysEqual<output_type>(expected, out));
        }
    }

    volk::vector<input_type> input, unaligned_input;
    volk::vector<output_type> expected, result, unaligned_result;

    virtual output_type expected_value(const input_type value)
    {
        static_assert(std::tuple_size_v<parameter_type> >= 2);
        return ConvertTestFunctions<std::tuple_size_v<parameter_type>,
                                    Kernel,
                                    parameter_type>::expected(value, this->GetParam());
    }

    virtual void run_kernel(std::span<output_type> out,
                            std::span<const input_type> in,
                            const std::string_view implementation)
    {
        static_assert(std::tuple_size_v<parameter_type> >= 2);
        ConvertTestFunctions<std::tuple_size_v<parameter_type>, Kernel, parameter_type>::
            run(out, in, implementation, this->GetParam());
    }
};

template <typename Kernel, typename Parameter = std::tuple<std::string, size_t, float>>
class ConvertTestWithScaling : public ConvertTest<Kernel, Parameter>
{
protected:
    using Base = ConvertTest<Kernel, Parameter>;
    using typename Base::input_type;
    using typename Base::output_type;
    using typename Base::parameter_type;

    output_type expected_value(const input_type value) override
    {
        static_assert(std::tuple_size_v<parameter_type> >= 3);
        return ConvertTestFunctions<std::tuple_size_v<parameter_type>,
                                    Kernel,
                                    parameter_type>::expected(value, this->GetParam());
    }

    void run_kernel(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view implementation) override
    {
        static_assert(std::tuple_size_v<parameter_type> >= 3);
        ConvertTestFunctions<std::tuple_size_v<parameter_type>, Kernel, parameter_type>::
            run(out, in, implementation, this->GetParam());
    }
};

template <typename Kernel,
          typename Parameter = std::tuple<std::string, size_t, float, float>>
class ConvertTestWithBias : public ConvertTestWithScaling<Kernel, Parameter>
{
protected:
    using Base = ConvertTestWithScaling<Kernel, Parameter>;
    using typename Base::input_type;
    using typename Base::output_type;
    using typename Base::parameter_type;

    output_type expected_value(const input_type value) override
    {
        static_assert(std::tuple_size_v<parameter_type> >= 4);
        return ConvertTestFunctions<std::tuple_size_v<parameter_type>,
                                    Kernel,
                                    parameter_type>::expected(value, this->GetParam());
    }

    void run_kernel(std::span<output_type> out,
                    std::span<const input_type> in,
                    const std::string_view implementation) override
    {
        static_assert(std::tuple_size_v<parameter_type> >= 4);
        ConvertTestFunctions<std::tuple_size_v<parameter_type>, Kernel, parameter_type>::
            run(out, in, implementation, this->GetParam());
    }
};

using volk_8i_convert_16i_test = ConvertTest<Volk8iConvert16i>;
TEST_P(volk_8i_convert_16i_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_8i_convert_16i,
    volk_8i_convert_16i_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk8iConvert16i::descriptor())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

using volk_16i_convert_8i_test = ConvertTest<Volk16iConvert8i>;
TEST_P(volk_16i_convert_8i_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_16i_convert_8i,
    volk_16i_convert_8i_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk16iConvert8i::descriptor())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

using volk_16ic_convert_32fc_test = ConvertTest<Volk16icConvert32fc>;
TEST_P(volk_16ic_convert_32fc_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_16ic_convert_32fc,
    volk_16ic_convert_32fc_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk16icConvert32fc::descriptor())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

using volk_32f_convert_64f_test = ConvertTest<Volk32fConvert64f>;
TEST_P(volk_32f_convert_64f_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_32f_convert_64f,
    volk_32f_convert_64f_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk32fConvert64f::descriptor())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

using volk_64f_convert_32f_test = ConvertTest<Volk64fConvert32f>;
TEST_P(volk_64f_convert_32f_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_64f_convert_32f,
    volk_64f_convert_32f_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk64fConvert32f::descriptor())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

using volk_32f_s32f_convert_16i_test = ConvertTestWithScaling<Volk32fS32fConvert16i>;
TEST_P(volk_32f_s32f_convert_16i_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_32f_s32f_convert_16i,
    volk_32f_s32f_convert_16i_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk32fS32fConvert16i::descriptor())),
                     testing::ValuesIn(default_vector_sizes),
                     testing::ValuesIn(scaling_values)),
    generate_volk_test_scaling_name());

using volk_32f_s32f_convert_32i_test = ConvertTestWithScaling<Volk32fS32fConvert32i>;
TEST_P(volk_32f_s32f_convert_32i_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_32f_s32f_convert_32i,
    volk_32f_s32f_convert_32i_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk32fS32fConvert32i::descriptor())),
                     testing::ValuesIn(default_vector_sizes),
                     testing::ValuesIn(scaling_values)),
    generate_volk_test_scaling_name());

using volk_32f_s32f_convert_8i_test = ConvertTestWithScaling<Volk32fS32fConvert8i>;
TEST_P(volk_32f_s32f_convert_8i_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_32f_s32f_convert_8i,
    volk_32f_s32f_convert_8i_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk32fS32fConvert8i::descriptor())),
                     testing::ValuesIn(default_vector_sizes),
                     testing::ValuesIn(scaling_values)),
    generate_volk_test_scaling_name());

using volk_32i_s32f_convert_32f_test = ConvertTestWithScaling<Volk32iS32fConvert32f>;
TEST_P(volk_32i_s32f_convert_32f_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_32i_s32f_convert_32f,
    volk_32i_s32f_convert_32f_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk32iS32fConvert32f::descriptor())),
                     testing::ValuesIn(default_vector_sizes),
                     testing::ValuesIn(scaling_values)),
    generate_volk_test_scaling_name());

using volk_8i_s32f_convert_32f_test = ConvertTestWithScaling<Volk8iS32fConvert32f>;
TEST_P(volk_8i_s32f_convert_32f_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_8i_s32f_convert_32f,
    volk_8i_s32f_convert_32f_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk8iS32fConvert32f::descriptor())),
                     testing::ValuesIn(default_vector_sizes),
                     testing::ValuesIn(scaling_values)),
    generate_volk_test_scaling_name());

using volk_32fc_convert_16ic_test = ConvertTest<Volk32fcConvert16ic>;
TEST_P(volk_32fc_convert_16ic_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_32fc_convert_16ic,
    volk_32fc_convert_16ic_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk32fcConvert16ic::descriptor())),
                     testing::ValuesIn(default_vector_sizes)),
    generate_volk_test_name());

using volk_16i_s32f_convert_32f_test = ConvertTestWithScaling<Volk16iS32fConvert32f>;
TEST_P(volk_16i_s32f_convert_32f_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_16i_s32f_convert_32f,
    volk_16i_s32f_convert_32f_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk16iS32fConvert32f::descriptor())),
                     testing::ValuesIn(default_vector_sizes),
                     testing::ValuesIn(scaling_values)),
    generate_volk_test_scaling_name());

using volk_32f_s32f_x2_convert_8u_test = ConvertTestWithBias<Volk32fS32fX2Convert8u>;
TEST_P(volk_32f_s32f_x2_convert_8u_test, run) { run(implementation_name); }
INSTANTIATE_TEST_SUITE_P(
    volk_32f_s32f_x2_convert_8u,
    volk_32f_s32f_x2_convert_8u_test,
    testing::Combine(testing::ValuesIn(get_kernel_implementation_name_list(
                         Volk32fS32fX2Convert8u::descriptor())),
                     testing::ValuesIn(default_vector_sizes),
                     testing::ValuesIn(scaling_values),
                     testing::ValuesIn(bias_values)),
    generate_volk_test_scaling_bias_name());

} // namespace
