#include <fmt/core.h>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

typedef std::complex<float> cmplxf;

#include <volk/volk.h>
#include <volk/volk_alloc.hh>


void cppmultiply(volk::vector<cmplxf>& result,
                 volk::vector<cmplxf>& input0,
                 volk::vector<cmplxf>& input1)
{
    volk_32fc_x2_multiply_32fc(reinterpret_cast<lv_32fc_t*>(result.data()),
                               reinterpret_cast<lv_32fc_t*>(input0.data()),
                               reinterpret_cast<lv_32fc_t*>(input1.data()),
                               input0.size());
}

void function_test(int num_points)
{
    volk::vector<cmplxf> in0(num_points);
    volk::vector<cmplxf> in1(num_points);
    volk::vector<cmplxf> out(num_points);

    for (unsigned int ii = 0; ii < num_points; ++ii) {
        // Generate two tones
        float real_1 = std::cos(0.3f * (float)ii);
        float imag_1 = std::sin(0.3f * (float)ii);
        in0[ii] = cmplxf(real_1, imag_1);
        float real_2 = std::cos(0.1f * (float)ii);
        float imag_2 = std::sin(0.1f * (float)ii);
        in1[ii] = cmplxf(real_2, imag_2);
    }

    cppmultiply(out, in0, in1);

    for (int ii = 0; ii < num_points; ++ii) {
        cmplxf v0 = in0[ii];
        cmplxf v1 = in1[ii];
        cmplxf o = out[ii];

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


int main(int argc, char* argv[])
{
    function_test(32);
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