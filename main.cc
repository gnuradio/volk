#include <fmt/core.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

/*
 * These type definitions are in line with our C definitions.
 *
 * Alternativele, we could go with the NumPy scheme:
 * np.complex64 aka std::complex<float>
 * np.complex128 aka std::complex<double>
 * The underlying types are probably defined like Ctypes.
 * This is about the idea.
 */
typedef std::complex<int8_t> ic8;
typedef std::complex<int16_t> ic16;
typedef std::complex<int32_t> ic32;
typedef std::complex<int64_t> ic64;
typedef std::complex<float> fc32;
typedef std::complex<double> fc64;

#include <volk/volk.h>
#include <volk/volk_alloc.hh>

/* C++ Interface requirements
 *
 * 1. Make C++ STL types usable `std::vector`, `std::complex`.
 * 2. Make aligned vectors aka `volk::vector` usable.
 * 3. Allow call-by-pointer for GR buffer interface usage etc.
 *
 * These requirements result in at least 3 functions.
 * We might want to think about fancy new C++ features e.g. concepts to consolidate these.
 */

namespace volk {

/*
 * Start of wrapper for volk_32fc_s32fc_multiply_32fc
 */
void cppscalarmultiply_pointers(fc32* result,
                                const fc32* input0,
                                const fc32 scalar,
                                const unsigned int num_points)
{
    volk_32fc_s32fc_multiply_32fc(reinterpret_cast<lv_32fc_t*>(result),
                                  reinterpret_cast<const lv_32fc_t*>(input0),
                                  lv_32fc_t{ scalar.real(), scalar.imag() },
                                  num_points);
}

void cppscalarmultiply_stl_vector(std::vector<fc32>& result,
                                  const std::vector<fc32>& input0,
                                  const fc32 scalar)
{
    unsigned int num_points = std::min({ result.size(), input0.size() });
    cppscalarmultiply_pointers(result.data(), input0.data(), scalar, num_points);
}

void cppscalarmultiply_aligned_vector(volk::vector<fc32>& result,
                                      const volk::vector<fc32>& input0,
                                      const fc32 scalar)
{
    unsigned int num_points = std::min({ result.size(), input0.size() });
    cppscalarmultiply_pointers(result.data(), input0.data(), scalar, num_points);
}

/*
 * Start of wrapper for volk_32fc_x2_multiply_32fc
 */
void cppmultiply_pointers(fc32* result,
                          const fc32* input0,
                          const fc32* input1,
                          const unsigned int num_points)
{
    volk_32fc_x2_multiply_32fc(reinterpret_cast<lv_32fc_t*>(result),
                               reinterpret_cast<const lv_32fc_t*>(input0),
                               reinterpret_cast<const lv_32fc_t*>(input1),
                               num_points);
}

void cppmultiply_stl_vector(std::vector<fc32>& result,
                            const std::vector<fc32>& input0,
                            const std::vector<fc32>& input1)
{
    unsigned int num_points = std::min({ result.size(), input0.size(), input1.size() });
    cppmultiply_pointers(result.data(), input0.data(), input1.data(), num_points);
}

void cppmultiply_aligned_vector(volk::vector<fc32>& result,
                                const volk::vector<fc32>& input0,
                                const volk::vector<fc32>& input1)
{
    unsigned int num_points = std::min({ result.size(), input0.size(), input1.size() });
    cppmultiply_pointers(result.data(), input0.data(), input1.data(), num_points);
}

} // namespace volk


std::vector<fc32> fill_vector(int num_points, float step_value)
{
    std::vector<fc32> vec(num_points);

    for (unsigned int ii = 0; ii < num_points; ++ii) {
        float real_1 = std::cos(step_value * (float)ii);
        float imag_1 = std::sin(step_value * (float)ii);
        vec[ii] = fc32(real_1, imag_1);
    }
    return vec;
}

void function_test_vectors(int num_points)
{
    std::vector<fc32> uin0(fill_vector(num_points, 0.3f));
    volk::vector<fc32> in0(uin0.begin(), uin0.end());
    std::vector<fc32> uin1(fill_vector(num_points, 0.1f));
    volk::vector<fc32> in1(uin1.begin(), uin1.end());
    std::vector<fc32> uout(num_points);
    volk::vector<fc32> out(num_points);

    volk::cppmultiply_aligned_vector(out, in0, in1);

    volk::cppmultiply_stl_vector(uout, uin0, uin1);
    volk::cppmultiply_pointers(uout.data(), in0.data(), in1.data(), num_points);

    for (int ii = 0; ii < num_points; ++ii) {
        fc32 v0 = in0[ii];
        fc32 v1 = in1[ii];
        fc32 o = out[ii];

        fmt::print(
            "in0=({:+.1f}{:+.1f}j), in1=({:+.1f}{:+.1f}j), out=({:+.1f}{:+.1f}j)\n",
            std::real(v0),
            std::imag(v0),
            std::real(v1),
            std::imag(v1),
            std::real(o),
            std::imag(o));
    }
}

void function_test_with_scalar(int num_points)
{
    std::vector<fc32> uin0(fill_vector(num_points, 0.3f));
    volk::vector<fc32> in0(uin0.begin(), uin0.end());
    fc32 scalar{ 0.5f, 4.3f };
    std::vector<fc32> uout(num_points);
    volk::vector<fc32> out(num_points);

    volk::cppscalarmultiply_aligned_vector(out, in0, scalar);

    volk::cppscalarmultiply_stl_vector(uout, uin0, scalar);
    volk::cppscalarmultiply_pointers(uout.data(), in0.data(), scalar, num_points);

    fmt::print("scalar=({:+.1f}{:+.1f}j)\n", std::real(scalar), std::imag(scalar));
    for (int ii = 0; ii < num_points; ++ii) {
        fc32 v0 = in0[ii];
        fc32 o = out[ii];

        fmt::print("in0=({:+.1f}{:+.1f}j), out=({:+.1f}{:+.1f}j)\n",
                   std::real(v0),
                   std::imag(v0),
                   std::real(o),
                   std::imag(o));
    }
}

int main(int argc, char* argv[])
{
    fmt::print("Vector function test\n");
    function_test_vectors(16);

    fmt::print("Scalar function test\n");
    function_test_with_scalar(16);

    lv_32fc_t fc_cpl[4];
    fmt::print("float={}, complex float={}, complex float array[4]={}\n",
               sizeof(float),
               sizeof(lv_32fc_t),
               sizeof(fc_cpl));


    std::vector<lv_32fc_t> vec(4);
    for (int i = 0; i < 4; i++) {
        auto foo = std::complex<float>((i + 3), (i + 8));
        fmt::print("std::complex: ({:+.1f}{:+.1f}j)\n", std::real(foo), std::imag(foo));
        lv_32fc_t bar = lv_32fc_t{ 5, 6 };
        vec.at(i) = bar;
    }

    for (auto& val : vec) {
        float r = __real__ val;
        float i = __imag__ val;
        fmt::print("sizeof(val)={}, {:+.1f}{:+.1f}j\n", sizeof(val), r, i);
    }
}