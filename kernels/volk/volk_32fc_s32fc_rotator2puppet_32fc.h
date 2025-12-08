/* -*- c++ -*- */
/*
 * Copyright 2012, 2013, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#ifndef INCLUDED_volk_32fc_s32fc_rotator2puppet_32fc_a_H
#define INCLUDED_volk_32fc_s32fc_rotator2puppet_32fc_a_H


#include <stdio.h>
#include <volk/volk_32fc_s32fc_x2_rotator2_32fc.h>
#include <volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32fc_rotator2puppet_32fc_generic(lv_32fc_t* outVector,
                                                               const lv_32fc_t* inVector,
                                                               const lv_32fc_t* phase_inc,
                                                               unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, 0.95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_generic(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void volk_32fc_s32fc_rotator2puppet_32fc_neon(lv_32fc_t* outVector,
                                                            const lv_32fc_t* inVector,
                                                            const lv_32fc_t* phase_inc,
                                                            unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, 0.95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_neon(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_s32fc_rotator2puppet_32fc_neonv8(lv_32fc_t* outVector,
                                                              const lv_32fc_t* inVector,
                                                              const lv_32fc_t* phase_inc,
                                                              unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, 0.95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_neonv8(
        outVector, inVector, &phase_inc_n, phase, num_points);
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32fc_s32fc_rotator2puppet_32fc_a_sse4_1(lv_32fc_t* outVector,
                                             const lv_32fc_t* inVector,
                                             const lv_32fc_t* phase_inc,
                                             unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_a_sse4_1(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
static inline void
volk_32fc_s32fc_rotator2puppet_32fc_u_sse4_1(lv_32fc_t* outVector,
                                             const lv_32fc_t* inVector,
                                             const lv_32fc_t* phase_inc,
                                             unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_u_sse4_1(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_s32fc_rotator2puppet_32fc_a_avx(lv_32fc_t* outVector,
                                                             const lv_32fc_t* inVector,
                                                             const lv_32fc_t* phase_inc,
                                                             unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_a_avx(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_s32fc_rotator2puppet_32fc_u_avx(lv_32fc_t* outVector,
                                                             const lv_32fc_t* inVector,
                                                             const lv_32fc_t* phase_inc,
                                                             unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_u_avx(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32fc_s32fc_rotator2puppet_32fc_a_avx512f(lv_32fc_t* outVector,
                                              const lv_32fc_t* inVector,
                                              const lv_32fc_t* phase_inc,
                                              unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_a_avx512f(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32fc_s32fc_rotator2puppet_32fc_u_avx512f(lv_32fc_t* outVector,
                                              const lv_32fc_t* inVector,
                                              const lv_32fc_t* phase_inc,
                                              unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_u_avx512f(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_AVX512F */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32fc_s32fc_rotator2puppet_32fc_a_avx_fma(lv_32fc_t* outVector,
                                              const lv_32fc_t* inVector,
                                              const lv_32fc_t* phase_inc,
                                              unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_a_avx_fma(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA*/


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32fc_s32fc_rotator2puppet_32fc_u_avx_fma(lv_32fc_t* outVector,
                                              const lv_32fc_t* inVector,
                                              const lv_32fc_t* phase_inc,
                                              unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_u_avx_fma(
        outVector, inVector, &phase_inc_n, phase, num_points);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA*/

#ifdef LV_HAVE_RVV
static inline void volk_32fc_s32fc_rotator2puppet_32fc_rvv(lv_32fc_t* outVector,
                                                           const lv_32fc_t* inVector,
                                                           const lv_32fc_t* phase_inc,
                                                           unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_rvv(
        outVector, inVector, &phase_inc_n, phase, num_points);
}
#endif /*LV_HAVE_RVV*/


#ifdef LV_HAVE_RVVSEG
static inline void volk_32fc_s32fc_rotator2puppet_32fc_rvvseg(lv_32fc_t* outVector,
                                                              const lv_32fc_t* inVector,
                                                              const lv_32fc_t* phase_inc,
                                                              unsigned int num_points)
{
    lv_32fc_t phase[1] = { lv_cmake(.3f, .95393f) };
    (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    const lv_32fc_t phase_inc_n =
        *phase_inc / hypotf(lv_creal(*phase_inc), lv_cimag(*phase_inc));
    volk_32fc_s32fc_x2_rotator2_32fc_rvv(
        outVector, inVector, &phase_inc_n, phase, num_points);
}
#endif /*LV_HAVE_RVVSEG*/
#endif /* INCLUDED_volk_32fc_s32fc_rotator2puppet_32fc_a_H */
