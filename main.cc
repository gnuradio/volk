#include <fmt/core.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <complex>


#include <volk/volk_complex.h>


int main(int argc, char* argv[])
{
    lv_32fc_t fc_cpl[4];
    fmt::print("float={}, complex float={}, complex float array[4]={}\n",
               sizeof(float),
               sizeof(lv_32fc_t),
               sizeof(fc_cpl));


    std::vector<lv_32fc_t> vec(4);
    for (int i = 0; i < 4; i++) {
        auto foo = std::complex<float>( (i + 3), (i + 8) );
        fmt::print("std::complex: ({:+.1f}{:+.1f}j)\n", std::real(foo), std::imag(foo));
        lv_32fc_t bar = lv_32fc_t{5, 6};
        vec.at(i) = bar;
        
    }

    for(auto &val : vec){
        float r = __real__ val;
        float i = __imag__ val;
        fmt::print("sizeof(val)={}, {:+.1f}{:+.1f}j\n", sizeof(val), r, i);
    }
}