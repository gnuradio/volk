/* -*- c++ -*- */
/*
 * Copyright 2012, 2013, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32fc_x2_rotator_32fc
 *
 * \b Deprecation
 *
 * This kernel is deprecated, because passing in `lv_32fc_t` by value results in
 * Undefined Behaviour, causing a segmentation fault on some architectures.
 * Use `volk_32fc_s32fc_x2_rotator2_32fc` instead.
 *
 * \b Overview
 *
 * Rotate input vector at fixed rate per sample from initial phase
 * offset.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_x2_rotator_32fc(lv_32fc_t* outVector, const lv_32fc_t* inVector,
 * const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inVector: Vector to be rotated.
 * \li phase_inc: rotational velocity.
 * \li phase: initial phase offset.
 * \li num_points: The number of values in inVector to be rotated and stored into
 * outVector.
 *
 * \b Outputs
 * \li outVector: The vector where the results will be stored.
 *
 * \b Example
 * Generate a tone at f=0.3 (normalized frequency) and use the rotator with
 * f=0.1 to shift the tone to f=0.4. Change this example to start with a DC
 * tone (initialize in with lv_cmake(1, 0)) to observe rotator signal generation.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       // Generate a tone at f=0.3
 *       float real = std::cos(0.3f * (float)ii);
 *       float imag = std::sin(0.3f * (float)ii);
 *       in[ii] = lv_cmake(real, imag);
 *   }
 *   // The oscillator rotates at f=0.1
 *   float frequency = 0.1f;
 *   lv_32fc_t phase_increment = lv_cmake(std::cos(frequency), std::sin(frequency));
 *   lv_32fc_t phase= lv_cmake(1.f, 0.0f); // start at 1 (0 rad phase)
 *
 *   // rotate so the output is a tone at f=0.4
 *   volk_32fc_s32fc_x2_rotator_32fc(out, in, phase_increment, &phase, N);
 *
 *   // print results for inspection
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %+1.2f %+1.2fj\n",
 *           ii, lv_creal(out[ii]), lv_cimag(out[ii]));
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H
#define INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <volk/volk_32fc_s32fc_x2_rotator_32fc.h>
#include <volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32fc_x2_rotator_32fc_generic(lv_32fc_t* outVector,
                                                           const lv_32fc_t* inVector,
                                                           const lv_32fc_t phase_inc,
                                                           lv_32fc_t* phase,
                                                           unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_generic(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON

static inline void volk_32fc_s32fc_x2_rotator_32fc_neon(lv_32fc_t* outVector,
                                                        const lv_32fc_t* inVector,
                                                        const lv_32fc_t phase_inc,
                                                        lv_32fc_t* phase,
                                                        unsigned int num_points)

{
    volk_32fc_s32fc_x2_rotator2_32fc_neon(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8

static inline void volk_32fc_s32fc_x2_rotator_32fc_neonv8(lv_32fc_t* outVector,
                                                          const lv_32fc_t* inVector,
                                                          const lv_32fc_t phase_inc,
                                                          lv_32fc_t* phase,
                                                          unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_neonv8(
        outVector, inVector, &phase_inc, phase, num_points);
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_SSE4_1

static inline void volk_32fc_s32fc_x2_rotator_32fc_a_sse4_1(lv_32fc_t* outVector,
                                                            const lv_32fc_t* inVector,
                                                            const lv_32fc_t phase_inc,
                                                            lv_32fc_t* phase,
                                                            unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_a_sse4_1(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_SSE4_1 for aligned */


#ifdef LV_HAVE_SSE4_1

static inline void volk_32fc_s32fc_x2_rotator_32fc_u_sse4_1(lv_32fc_t* outVector,
                                                            const lv_32fc_t* inVector,
                                                            const lv_32fc_t phase_inc,
                                                            lv_32fc_t* phase,
                                                            unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_u_sse4_1(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX

static inline void volk_32fc_s32fc_x2_rotator_32fc_a_avx(lv_32fc_t* outVector,
                                                         const lv_32fc_t* inVector,
                                                         const lv_32fc_t phase_inc,
                                                         lv_32fc_t* phase,
                                                         unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_a_avx(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_AVX for aligned */


#ifdef LV_HAVE_AVX

static inline void volk_32fc_s32fc_x2_rotator_32fc_u_avx(lv_32fc_t* outVector,
                                                         const lv_32fc_t* inVector,
                                                         const lv_32fc_t phase_inc,
                                                         lv_32fc_t* phase,
                                                         unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_u_avx(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void volk_32fc_s32fc_x2_rotator_32fc_a_avx_fma(lv_32fc_t* outVector,
                                                             const lv_32fc_t* inVector,
                                                             const lv_32fc_t phase_inc,
                                                             lv_32fc_t* phase,
                                                             unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_a_avx_fma(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA for aligned*/

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void volk_32fc_s32fc_x2_rotator_32fc_u_avx_fma(lv_32fc_t* outVector,
                                                             const lv_32fc_t* inVector,
                                                             const lv_32fc_t phase_inc,
                                                             lv_32fc_t* phase,
                                                             unsigned int num_points)
{
    volk_32fc_s32fc_x2_rotator2_32fc_u_avx_fma(
        outVector, inVector, &phase_inc, phase, num_points);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA*/

#endif /* INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H */
