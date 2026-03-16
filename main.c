
#include <math.h>
#include <stdio.h>
#include <volk/volk.h>

void function_test(int num_points)
{
    unsigned int alignment = volk_get_alignment();
    lv_32fc_t* in0 = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * num_points, alignment);
    lv_32fc_t* in1 = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * num_points, alignment);
    lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * num_points, alignment);

    for (unsigned int ii = 0; ii < num_points; ++ii) {
        // Generate two tones
        float real_1 = cosf(0.3f * (float)ii);
        float imag_1 = sinf(0.3f * (float)ii);
        in0[ii] = lv_cmake(real_1, imag_1);
        float real_2 = cosf(0.1f * (float)ii);
        float imag_2 = sinf(0.1f * (float)ii);
        in1[ii] = lv_cmake(real_2, imag_2);
    }

    volk_32fc_x2_multiply_32fc(out, in0, in1, num_points);

    for (unsigned int ii = 0; ii < num_points; ++ii) {
        lv_32fc_t v0 = in0[ii];
        lv_32fc_t v1 = in1[ii];
        lv_32fc_t o = out[ii];
        printf("in0=(%+.1f%+.1fj), in1=(%+.1f%+.1fj), out=(%+.1f%+.1fj)\n",
               creal(v0),
               cimag(v0),
               creal(v1),
               cimag(v1),
               creal(o),
               cimag(o));
    }

    volk_free(in0);
    volk_free(in1);
    volk_free(out);
}

int main(int argc, char* argv[])
{
    function_test(32);

    lv_32fc_t fc_cpl[4];
    printf("float=%lu, complex float=%lu, complex float array[4]=%lu\n",
           sizeof(float),
           sizeof(lv_32fc_t),
           sizeof(fc_cpl));

    for (int i = 0; i < 4; i++) {
        fc_cpl[i] = (i + 3) + I * (i + 8);

        fc_cpl[i] = lv_cmake(i + 3, i + 8);
    }
    for (int i = 0; i < 4; i++) {
        lv_32fc_t val = fc_cpl[i];
        lv_32fc_t cval = conj(val);
        lv_32fc_t gval = ~val;
        lv_32fc_t mult = val * val;
        printf("val      = %+.1f%+.1fj\n", creal(val), cimag(val));
        printf("conj(val)= %+.1f%+.1fj\n", creal(cval), cimag(cval));
        printf("gcc: ~val= %+.1f%+.1fj\n", creal(gval), cimag(gval));
        printf("val*val  = %+.1f%+.1fj\n", creal(mult), cimag(mult));
    }

    lv_8sc_t sc_cpl[4];
    printf("\n\nchar=%lu, complex char=%lu, complex char array[4]=%lu\n",
           sizeof(char),
           sizeof(lv_8sc_t),
           sizeof(sc_cpl));

    for (int i = 0; i < 4; i++) {
        // lv_8sc_t value = (i + 3) + I * (i + 8);
        // printf("value=%+hhi%+hhij\n", creal(value), cimag(value));
        // sc_cpl[i] = (i + 3) + I * (i + 8);
        sc_cpl[i] = lv_cmake(i + 3, i + 8);
        // printf("%i + j %i\n", creal(sc_cpl[i]), cimag(sc_cpl[i]));
    }
    for (int i = 0; i < 4; i++) {
        lv_8sc_t val = sc_cpl[i];
        lv_8sc_t cval = conj(val);
        // lv_8sc_t cval = lv_cmake(creal(val), -cimag(val));
        lv_8sc_t gval = ~val;
        lv_8sc_t mult = val * val;
        printf("val      = %+hhi%+hhij\n", __real__ val, __imag__ val);
        printf("conj(val)= %+hhi%+hhij\n", __real__ cval, __imag__ cval);
        printf("gcc: ~val= %+hhi%+hhij\n", __real__ gval, __imag__ gval);
        printf("val*val  = %+hhi%+hhij\n", __real__ mult, __imag__ mult);
    }

    //     char* values = (char*) sc_cpl;
    //   for (int i = 0; i < 8; i++) {
    //     printf("%hhi\n", values[i]);
    //   }
}
