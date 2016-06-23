/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
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
 * void volk_16ic_convert_32fc(lv_32fc_t* outputVector, const lv_16sc_t* inputVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector:  The complex 16-bit integer input data buffer.
 * \li num_points:   The number of data values to be converted.
 *
 * \b Outputs
 * \li outputVector: pointer to a vector holding the converted vector.
 *
 */


#ifndef INCLUDED_volk_16ic_convert_32fc_H
#define INCLUDED_volk_16ic_convert_32fc_H

#include <volk/volk_complex.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_convert_32fc_generic(lv_32fc_t* outputVector, const lv_16sc_t* inputVector, unsigned int num_points)
{
    unsigned int i;
    for(i = 0; i < num_points; i++)
        {
            outputVector[i] = lv_cmake((float)lv_creal(inputVector[i]), (float)lv_cimag(inputVector[i]));
        }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_convert_32fc_a_sse2(lv_32fc_t* outputVector, const lv_16sc_t* inputVector, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 2;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;
    __m128 a;
    unsigned int i, number;

    for(number = 0; number < sse_iters; number++)
        {
            a = _mm_set_ps((float)(lv_cimag(_in[1])), (float)(lv_creal(_in[1])), (float)(lv_cimag(_in[0])), (float)(lv_creal(_in[0]))); // //load (2 byte imag, 2 byte real) x 2 into 128 bits reg
            _mm_store_ps((float*)_out, a);
            _in += 2;
            _out += 2;
        }
    for (i = 0; i < (num_points % 2); ++i)
        {
            *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
            _in++;
        }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_convert_32fc_u_sse2(lv_32fc_t* outputVector, const lv_16sc_t* inputVector, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 2;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;
    __m128 a;
    unsigned int i, number;

    for(number = 0; number < sse_iters; number++)
        {
            a = _mm_set_ps((float)(lv_cimag(_in[1])), (float)(lv_creal(_in[1])), (float)(lv_cimag(_in[0])), (float)(lv_creal(_in[0]))); // //load (2 byte imag, 2 byte real) x 2 into 128 bits reg
            _mm_storeu_ps((float*)_out, a);
            _in += 2;
            _out += 2;
        }
    for (i = 0; i < (num_points % 2); ++i)
        {
            *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
            _in++;
        }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_u_axv(lv_32fc_t* outputVector, const lv_16sc_t* inputVector, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;
    __m256 a;
    unsigned int i, number;

    for(number = 0; number < sse_iters; number++)
        {
            a = _mm256_set_ps((float)(lv_cimag(_in[3])), (float)(lv_creal(_in[3])), (float)(lv_cimag(_in[2])), (float)(lv_creal(_in[2])), (float)(lv_cimag(_in[1])), (float)(lv_creal(_in[1])), (float)(lv_cimag(_in[0])), (float)(lv_creal(_in[0]))); // //load (2 byte imag, 2 byte real) x 2 into 128 bits reg
            _mm256_storeu_ps((float*)_out, a);
            _in += 4;
            _out += 4;
        }
    _mm256_zeroupper();
    for (i = 0; i < (num_points % 4); ++i)
        {
            *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
            _in++;
        }
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16ic_convert_32fc_a_axv(lv_32fc_t* outputVector, const lv_16sc_t* inputVector, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;
    __m256 a;
    unsigned int i, number;

    for(number = 0; number < sse_iters; number++)
        {
            a = _mm256_set_ps((float)(lv_cimag(_in[3])), (float)(lv_creal(_in[3])), (float)(lv_cimag(_in[2])), (float)(lv_creal(_in[2])), (float)(lv_cimag(_in[1])), (float)(lv_creal(_in[1])), (float)(lv_cimag(_in[0])), (float)(lv_creal(_in[0]))); // //load (2 byte imag, 2 byte real) x 2 into 128 bits reg
            _mm256_store_ps((float*)_out, a);
            _in += 4;
            _out += 4;
        }
    _mm256_zeroupper();
    for (i = 0; i < (num_points % 4); ++i)
        {
            *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
            _in++;
        }
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_convert_32fc_neon(lv_32fc_t* outputVector, const lv_16sc_t* inputVector, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 2;

    const lv_16sc_t* _in = inputVector;
    lv_32fc_t* _out = outputVector;

    int16x4_t a16x4;
    int32x4_t a32x4;
    float32x4_t f32x4;
    unsigned int i, number;

    for(number = 0; number < sse_iters; number++)
        {
            a16x4 = vld1_s16((const int16_t*)_in);
            __builtin_prefetch(_in + 4);
            a32x4 = vmovl_s16(a16x4);
            f32x4 = vcvtq_f32_s32(a32x4);
            vst1q_f32((float32_t*)_out, f32x4);
            _in += 2;
            _out += 2;
        }
    for (i = 0; i < (num_points % 2); ++i)
        {
            *_out++ = lv_cmake((float)lv_creal(*_in), (float)lv_cimag(*_in));
            _in++;
        }
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_32fc_convert_16ic_H */