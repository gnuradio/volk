#include <gtest/gtest.h>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>


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


TEST(Multiply, AVX)
{
    const size_t vector_length = 32;
    auto vec0 = volk::vector<lv_32fc_t>(vector_length);
    auto vec1 = volk::vector<lv_32fc_t>(vector_length);
    auto result = volk::vector<lv_32fc_t>(vector_length);
    for (size_t i = 0; i < vector_length; ++i) {
        vec0[i] = std::complex<float>(i * 3.14, i * 0.45);
        vec1[i] = std::complex<float>(i * -2.78, i * 5.44);
    }

    auto expected = volk::vector<lv_32fc_t>(vector_length);
    for (size_t i = 0; i < vector_length; ++i) {
        expected[i] = vec0[i] * vec1[i];
    }

    volk_32fc_x2_multiply_32fc_manual(result.data(), vec0.data(), vec1.data(), vector_length);
    // EXPECT_ITERABLE_COMPLEX_FLOAT_EQ(volk::vector<lv_32fc_t>, expected, result);
    EXPECT_TRUE(AreComplexFloatingPointArraysAlmostEqual(expected, result));
}
