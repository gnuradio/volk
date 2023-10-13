/* -*- c++ -*- */
/*
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_reciprocal_32f
 *
 * \b Overview
 *
 * Computes the reciprocal of the input vector and stores the results
 * in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_reciprocal_32f(float* out, const float* in, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li in: A pointer to the input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: A pointer to the output vector of floats.
 *
 * \b Example
 * \code
    int N = 10;
    unsigned int alignment = volk_get_alignment();
    float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
    float* out = (float*)volk_malloc(sizeof(float)*N, alignment);

    for(unsigned int ii = 1; ii < N; ++ii){
        in[ii] = (float)(ii*ii);
    }

    volk_32f_reciprocal_32f(out, in, N);

    for(unsigned int ii = 0; ii < N; ++ii){
        printf("out(%i) = %f\n", ii, out[ii]);
    }

    volk_free(in);
    volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_reciprocal_32f_a_H
#define INCLUDED_volk_32f_reciprocal_32f_a_H

#ifdef LV_HAVE_GENERIC
static inline void
volk_32f_reciprocal_32f_generic(float* out, const float* in, unsigned int num_points)
{
    unsigned int i = 0;
    for (; i < num_points; i++) {
        out[i] = 1.f / in[i];
    }
}
#endif /* LV_HAVE_GENERIC */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_avx2_fma(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    for (; number < eighth_points; number++) {
        __m256 x = _mm256_load_ps(in);
        in += 8;

        __m256 x_inv = _mm256_rcp1_avx2_fma_ps(x);

        _mm256_store_ps(out, x_inv);
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *out++ = 1.f / (*in++);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_a_avx_nr(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_load_ps(in);
        in += 8;

        __m256 x_inv = _mm256_rcp1_avx_ps(x);

        _mm256_store_ps(out, x_inv);
        out += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *out++ = 1.f / (*in++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
static inline void
volk_32f_reciprocal_32f_a_avx_div(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    const __m256 ONE = _mm256_set1_ps(1.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_load_ps(in);
        in += 8;

        __m256 x_inv = _mm256_div_ps(ONE, x);

        _mm256_store_ps(out, x_inv);
        out += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *out++ = 1.f / (*in++);
    }
}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_reciprocal_32f_a_H */

#ifndef INCLUDED_volk_32f_reciprocal_32f_u_H
#define INCLUDED_volk_32f_reciprocal_32f_u_H

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_avx2_fma(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    for (; number < eighth_points; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;

        __m256 x_inv = _mm256_rcp1_avx2_fma_ps(x);

        _mm256_storeu_ps(out, x_inv);
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *out++ = 1.f / (*in++);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
static inline void
volk_32f_reciprocal_32f_u_avx_nr(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;

        __m256 x_inv = _mm256_rcp1_avx_ps(x);

        _mm256_storeu_ps(out, x_inv);
        out += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *out++ = 1.f / (*in++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
static inline void
volk_32f_reciprocal_32f_u_avx_div(float* out, const float* in, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    const __m256 ONE = _mm256_set1_ps(1.0f);

    for (; number < eighthPoints; number++) {
        __m256 x = _mm256_loadu_ps(in);
        in += 8;

        __m256 x_inv = _mm256_div_ps(ONE, x);

        _mm256_storeu_ps(out, x_inv);
        out += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *out++ = 1.f / (*in++);
    }
}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_reciprocal_32f_u_H */
