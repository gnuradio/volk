/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32fc_multiply_32fc
 *
 * \b Deprecation
 *
 * This kernel is deprecated, because passing in \p lv_32fc_t by value results in
 * undefined behavior, causing a segmentation fault on some architectures.
 * Use \ref volk_32fc_s32fc_multiply2_32fc instead.
 *
 * \b Overview
 *
 * Multiplies each element of a complex floating-point vector by a complex scalar.
 * For each element, computes cVector[i] = aVector[i] * scalar using standard complex
 * multiplication.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_multiply_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector,
 * const lv_32fc_t scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: Complex input vector of length \p num_points (lv_32fc_t).
 * \li scalar: The complex scalar multiplier (lv_32fc_t, passed by value).
 * \li num_points: The number of complex elements in \p aVector.
 *
 * \b Outputs
 * \li cVector: Complex output vector of length \p num_points (lv_32fc_t).
 *
 * \b Example
 * Generate points around the unit circle and shift the phase by pi/3 radians.
 * \code
 * #include <volk/volk.h>
 * #include <stdio.h>
 * #include <math.h>
 * #include <complex>
 *
 * int main() {
 *     unsigned int N = 10;
 *     unsigned int alignment = volk_get_alignment();
 *     lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *     lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 *     // Scalar rotates phase by pi/3
 *     lv_32fc_t scalar = lv_cmake((float)cos(M_PI / 3.0), (float)sin(M_PI / 3.0));
 *
 *     // Generate points around the unit circle
 *     float delta = 2.0f * (float)M_PI / (float)N;
 *     for (unsigned int i = 0; i < N / 2; ++i) {
 *         float real = cosf(delta * (float)i);
 *         float imag = sinf(delta * (float)i);
 *         in[i]         = lv_cmake(real, imag);
 *         in[i + N / 2] = lv_cmake(-real, -imag);
 *     }
 *
 *     volk_32fc_s32fc_multiply_32fc(out, in, scalar, N);
 *
 *     printf("  in mag   in phase  |  out mag  out phase\n");
 *     for (unsigned int i = 0; i < N; ++i) {
 *         printf("  %+1.2f    %+1.2f    |   %+1.2f    %+1.2f\n",
 *                std::abs(in[i]),  std::arg(in[i]),
 *                std::abs(out[i]), std::arg(out[i]));
 *     }
 *
 *     volk_free(in);
 *     volk_free(out);
 *     return 0;
 * }
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32fc_multiply_32fc_u_H
#define INCLUDED_volk_32fc_s32fc_multiply_32fc_u_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_32fc_s32fc_multiply2_32fc.h>
#include <volk/volk_complex.h>

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void volk_32fc_s32fc_multiply_32fc_u_avx_fma(lv_32fc_t* cVector,
                                                           const lv_32fc_t* aVector,
                                                           const lv_32fc_t scalar,
                                                           unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_u_avx_fma(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX

static inline void volk_32fc_s32fc_multiply_32fc_u_avx(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t scalar,
                                                       unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_u_avx(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3

static inline void volk_32fc_s32fc_multiply_32fc_u_sse3(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t scalar,
                                                        unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_u_sse3(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32fc_multiply_32fc_generic(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t scalar,
                                                         unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_generic(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_x2_multiply_32fc_u_H */
#ifndef INCLUDED_volk_32fc_s32fc_multiply_32fc_a_H
#define INCLUDED_volk_32fc_s32fc_multiply_32fc_a_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void volk_32fc_s32fc_multiply_32fc_a_avx_fma(lv_32fc_t* cVector,
                                                           const lv_32fc_t* aVector,
                                                           const lv_32fc_t scalar,
                                                           unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_a_avx_fma(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX

static inline void volk_32fc_s32fc_multiply_32fc_a_avx(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t scalar,
                                                       unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_a_avx(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3

static inline void volk_32fc_s32fc_multiply_32fc_a_sse3(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t scalar,
                                                        unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_a_sse3(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_NEON

static inline void volk_32fc_s32fc_multiply_32fc_neon(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_neon(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8

static inline void volk_32fc_s32fc_multiply_32fc_neonv8(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t scalar,
                                                        unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_neonv8(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32fc_x2_multiply_32fc_a_H */
