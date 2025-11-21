/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_convert_32fc
 *
 * \b Overview
 *
 * Converts a complex vector of 16-bits integer each component
 * into a complex vector of 32-bits float each component.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_convert_32fc(lv_32fc_t* outputVector, const lv_16sc_t* inputVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector:  The complex 16-bit integer input data buffer.
 * \li num_points:   The number of data values to be converted.
 *
 * \b Outputs
 * \li outputVector: pointer to a vector holding the converted vector.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * unsigned int alignment = volk_get_alignment();
 * lv_16sc_t* input  = (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t)*N, alignment);
 * lv_32fc_t* output  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 * volk_16ic_convert_32f(output, input, N);
 *
 * volk_free(input);
 * volk_free(output);
 * \endcode
 */


#ifndef INCLUDED_volk_16ic_convert_32fc_a_H
#define INCLUDED_volk_16ic_convert_32fc_a_H

#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_a_avx2(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 4;
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)inputVector;
    float* outputVectorPtr = (float*)outputVector;
    __m256 outVal;
    __m256i outValInt;
    __m128i cplxValue;

    for (number = 0; number < avx_iters; number++) {
        cplxValue = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        outValInt = _mm256_cvtepi16_epi32(cplxValue);
        outVal = _mm256_cvtepi32_ps(outValInt);
        _mm256_store_ps((float*)outputVectorPtr, outVal);

        outputVectorPtr += 8;
    }

    number = avx_iters * 8;
    for (; number < num_points * 2; number++) {
        *outputVectorPtr++ = (float)*complexVectorPtr++;
    }
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_a_avx512(lv_32fc_t* outputVector,
                                                    const lv_16sc_t* inputVector,
                                                    unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 8;
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)inputVector;
    float* outputVectorPtr = (float*)outputVector;
    __m512 outVal;
    __m512i outValInt;
    __m256i cplxValue;

    for (number = 0; number < avx512_iters; number++) {
        // Load 16 int16 values (8 complex = 16 floats)
        cplxValue = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;

        // Convert int16 → int32 → float
        outValInt = _mm512_cvtepi16_epi32(cplxValue);
        outVal = _mm512_cvtepi32_ps(outValInt);
        _mm512_store_ps((float*)outputVectorPtr, outVal);

        outputVectorPtr += 16;
    }

    number = avx512_iters * 16;
    for (; number < num_points * 2; number++) {
        *outputVectorPtr++ = (float)*complexVectorPtr++;
    }
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_convert_32fc_generic(lv_32fc_t* outputVector,
                                                  const lv_16sc_t* inputVector,
                                                  unsigned int num_points)
{
    unsigned int i;
    for (i = 0; i < num_points; i++) {
        outputVector[i] =
            lv_cmake((float)lv_creal(inputVector[i]), (float)lv_cimag(inputVector[i]));
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_convert_32fc_a_sse2(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 2;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;
    __m128 a;
    unsigned int number;

    for (number = 0; number < sse_iters; number++) {
        a = _mm_set_ps(
            (float)(lv_cimag(_in[1])),
            (float)(lv_creal(_in[1])),
            (float)(lv_cimag(_in[0])),
            (float)(lv_creal(
                _in[0]))); // //load (2 byte imag, 2 byte real) x 2 into 128 bits reg
        _mm_store_ps((float*)_out, a);
        _in += 2;
        _out += 2;
    }
    if (num_points & 1) {
        *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
        _in++;
    }
}

#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_a_avx(lv_32fc_t* outputVector,
                                                const lv_16sc_t* inputVector,
                                                unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;
    __m256 a;
    unsigned int i, number;

    for (number = 0; number < sse_iters; number++) {
        a = _mm256_set_ps(
            (float)(lv_cimag(_in[3])),
            (float)(lv_creal(_in[3])),
            (float)(lv_cimag(_in[2])),
            (float)(lv_creal(_in[2])),
            (float)(lv_cimag(_in[1])),
            (float)(lv_creal(_in[1])),
            (float)(lv_cimag(_in[0])),
            (float)(lv_creal(
                _in[0]))); // //load (2 byte imag, 2 byte real) x 2 into 128 bits reg
        _mm256_store_ps((float*)_out, a);
        _in += 4;
        _out += 4;
    }

    for (i = 0; i < (num_points % 4); ++i) {
        *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
        _in++;
    }
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_convert_32fc_neon(lv_32fc_t* outputVector,
                                               const lv_16sc_t* inputVector,
                                               unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 2;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;

    int16x4_t a16x4;
    int32x4_t a32x4;
    float32x4_t f32x4;
    unsigned int i, number;

    for (number = 0; number < sse_iters; number++) {
        a16x4 = vld1_s16((const int16_t*)_in);
        __VOLK_PREFETCH(_in + 4);
        a32x4 = vmovl_s16(a16x4);
        f32x4 = vcvtq_f32_s32(a32x4);
        vst1q_f32((float32_t*)_out, f32x4);
        _in += 2;
        _out += 2;
    }
    for (i = 0; i < (num_points % 2); ++i) {
        *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
        _in++;
    }
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_32fc_convert_16ic_a_H */

#ifndef INCLUDED_volk_16ic_convert_32fc_u_H
#define INCLUDED_volk_16ic_convert_32fc_u_H

#include <volk/volk_complex.h>


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_u_avx2(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 4;
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)inputVector;
    float* outputVectorPtr = (float*)outputVector;
    __m256 outVal;
    __m256i outValInt;
    __m128i cplxValue;

    for (number = 0; number < avx_iters; number++) {
        cplxValue = _mm_loadu_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        outValInt = _mm256_cvtepi16_epi32(cplxValue);
        outVal = _mm256_cvtepi32_ps(outValInt);
        _mm256_storeu_ps((float*)outputVectorPtr, outVal);

        outputVectorPtr += 8;
    }

    number = avx_iters * 8;
    for (; number < num_points * 2; number++) {
        *outputVectorPtr++ = (float)*complexVectorPtr++;
    }
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_u_avx512(lv_32fc_t* outputVector,
                                                    const lv_16sc_t* inputVector,
                                                    unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 8;
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)inputVector;
    float* outputVectorPtr = (float*)outputVector;
    __m512 outVal;
    __m512i outValInt;
    __m256i cplxValue;

    for (number = 0; number < avx512_iters; number++) {
        // Load 16 int16 values (8 complex = 16 floats) - unaligned
        cplxValue = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;

        // Convert int16 → int32 → float
        outValInt = _mm512_cvtepi16_epi32(cplxValue);
        outVal = _mm512_cvtepi32_ps(outValInt);
        _mm512_storeu_ps((float*)outputVectorPtr, outVal);

        outputVectorPtr += 16;
    }

    number = avx512_iters * 16;
    for (; number < num_points * 2; number++) {
        *outputVectorPtr++ = (float)*complexVectorPtr++;
    }
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_convert_32fc_u_sse2(lv_32fc_t* outputVector,
                                                 const lv_16sc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 2;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;
    __m128 a;
    unsigned int number;

    for (number = 0; number < sse_iters; number++) {
        a = _mm_set_ps(
            (float)(lv_cimag(_in[1])),
            (float)(lv_creal(_in[1])),
            (float)(lv_cimag(_in[0])),
            (float)(lv_creal(
                _in[0]))); // //load (2 byte imag, 2 byte real) x 2 into 128 bits reg
        _mm_storeu_ps((float*)_out, a);
        _in += 2;
        _out += 2;
    }
    if (num_points & 1) {
        *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
        _in++;
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_u_avx(lv_32fc_t* outputVector,
                                                const lv_16sc_t* inputVector,
                                                unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;
    __m256 a;
    unsigned int i, number;

    for (number = 0; number < sse_iters; number++) {
        a = _mm256_set_ps(
            (float)(lv_cimag(_in[3])),
            (float)(lv_creal(_in[3])),
            (float)(lv_cimag(_in[2])),
            (float)(lv_creal(_in[2])),
            (float)(lv_cimag(_in[1])),
            (float)(lv_creal(_in[1])),
            (float)(lv_cimag(_in[0])),
            (float)(lv_creal(
                _in[0]))); // //load (2 byte imag, 2 byte real) x 2 into 128 bits reg
        _mm256_storeu_ps((float*)_out, a);
        _in += 4;
        _out += 4;
    }

    for (i = 0; i < (num_points % 4); ++i) {
        *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
        _in++;
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_convert_32fc_rvv(lv_32fc_t* outputVector,
                                              const lv_16sc_t* inputVector,
                                              unsigned int num_points)
{
    const int16_t* in = (const int16_t*)inputVector;
    float* out = (float*)outputVector;
    size_t n = num_points * 2;
    for (size_t vl; n > 0; n -= vl, in += vl, out += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4_t v = __riscv_vle16_v_i16m4(in, vl);
        __riscv_vse32(out, __riscv_vfwcvt_f(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_convert_16ic_u_H */
