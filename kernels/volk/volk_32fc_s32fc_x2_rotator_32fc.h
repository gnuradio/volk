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
 * This kernel is deprecated, because passing in \p lv_32fc_t by value results in
 * undefined behavior, causing a segmentation fault on some architectures.
 * Use \ref volk_32fc_s32fc_x2_rotator2_32fc instead.
 *
 * \b Overview
 *
 * Applies a complex phase rotation to each element of a complex floating-point input
 * vector. The rotation advances by a fixed phase increment per sample, starting from
 * a caller-supplied initial phase. On return, the phase accumulator is updated to
 * reflect the final phase, allowing back-to-back calls for continuous rotation across
 * blocks.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_x2_rotator_32fc(lv_32fc_t* outVector, const lv_32fc_t* inVector,
 * const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inVector: Complex input vector of length \p num_points (lv_32fc_t).
 * \li phase_inc: Per-sample phase increment as a unit-magnitude complex number
 *     (lv_32fc_t, passed by value).
 * \li phase: Pointer to the complex phase accumulator; provides the initial phase and
 *     is updated on return (lv_32fc_t*).
 * \li num_points: The number of complex elements to process.
 *
 * \b Outputs
 * \li outVector: Complex output vector of length \p num_points (lv_32fc_t).
 * \li phase: Updated to the phase value after the last sample.
 *
 * \b Example
 * Generate a tone at normalized frequency f = 0.3 and use the rotator with
 * f = 0.1 to shift the tone to f = 0.4.
 * \code
 * #include <volk/volk.h>
 * #include <math.h>
 * #include <stdio.h>
 *
 * int main() {
 *     unsigned int N = 10;
 *     unsigned int alignment = volk_get_alignment();
 *     lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *     lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 *     // Generate a tone at f = 0.3
 *     for (unsigned int i = 0; i < N; ++i) {
 *         float angle = 0.3f * (float)i;
 *         in[i] = lv_cmake(cosf(angle), sinf(angle));
 *     }
 *
 *     // Phase increment corresponding to f = 0.1
 *     float freq = 0.1f;
 *     lv_32fc_t phase_inc = lv_cmake(cosf(freq), sinf(freq));
 *     lv_32fc_t phase     = lv_cmake(1.0f, 0.0f); // start at 0 rad
 *
 *     // Rotate so the output is a tone at f = 0.4
 *     volk_32fc_s32fc_x2_rotator_32fc(out, in, phase_inc, &phase, N);
 *
 *     for (unsigned int i = 0; i < N; ++i) {
 *         printf("out[%u] = %+1.4f %+1.4fj\n",
 *                i, lv_creal(out[i]), lv_cimag(out[i]));
 *     }
 *
 *     volk_free(in);
 *     volk_free(out);
 *     return 0;
 * }
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

#endif /* INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H */
