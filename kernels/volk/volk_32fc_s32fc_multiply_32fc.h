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
 * This kernel is deprecated, because passing in `lv_32fc_t` by value results in
 * Undefined Behaviour, causing a segmentation fault on some architectures.
 * Use `volk_32fc_s32fc_multiply2_32fc` instead.
 *
 * \b Overview
 *
 * Multiplies the input complex vector by a complex scalar and returns
 * the results.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_multiply_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const
 * lv_32fc_t scalar, unsigned int num_points); \endcode
 *
 * \b Inputs
 * \li aVector: The input vector to be multiplied.
 * \li scalar The complex scalar to multiply against aVector.
 * \li num_points: The number of complex values in aVector.
 *
 * \b Outputs
 * \li cVector: The vector where the results will be stored.
 *
 * \b Example
 * Generate points around the unit circle and shift the phase pi/3 rad.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t scalar = lv_cmake((float)std::cos(M_PI/3.f), (float)std::sin(M_PI/3.f));
 *
 *   float delta = 2.f*M_PI / (float)N;
 *   for(unsigned int ii = 0; ii < N/2; ++ii){
 *       // Generate points around the unit circle
 *       float real = std::cos(delta * (float)ii);
 *       float imag = std::sin(delta * (float)ii);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii+N/2] = lv_cmake(-real, -imag);
 *   }
 *
 *   volk_32fc_s32fc_multiply_32fc(out, in, scalar, N);
 *
 *   printf(" mag   phase  |   mag   phase\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("%+1.2f  %+1.2f  |  %+1.2f   %+1.2f\n",
 *           std::abs(in[ii]), std::arg(in[ii]),
 *           std::abs(out[ii]), std::arg(out[ii]));
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
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
