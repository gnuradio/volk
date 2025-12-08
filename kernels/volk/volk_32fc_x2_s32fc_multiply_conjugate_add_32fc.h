/* -*- c++ -*- */
/*
 * Copyright 2019 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_s32fc_multiply_conjugate_add_32fc
 *
 * \b Deprecation
 *
 * This kernel is deprecated, because passing in `lv_32fc_t` by value results in
 * Undefined Behaviour, causing a segmentation fault on some architectures.
 * Use `volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc` instead.
 *
 * \b Overview
 *
 * Conjugate the input complex vector, multiply them by a complex scalar,
 * add the another input complex vector and returns the results.
 *
 * c[i] = a[i] + conj(b[i]) * scalar
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_s32fc_multiply_conjugate_add_32fc(lv_32fc_t* cVector, const
 * lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t scalar, unsigned int
 * num_points); \endcode
 *
 * \b Inputs
 * \li aVector: The input vector to be added.
 * \li bVector: The input vector to be conjugate and multiplied.
 * \li scalar: The complex scalar to multiply against conjugated bVector.
 * \li num_points: The number of complex values in aVector and bVector to be conjugate,
 * multiplied and stored into cVector.
 *
 * \b Outputs
 * \li cVector: The vector where the results will be stored.
 *
 * \b Example
 * Calculate coefficients.
 *
 * \code
 * int n_filter = 2 * N + 1;
 * unsigned int alignment = volk_get_alignment();
 *
 * // state is a queue of input IQ data.
 * lv_32fc_t* state = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * n_filter, alignment);
 * // weight and next one.
 * lv_32fc_t* weight = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * n_filter, alignment);
 * lv_32fc_t* next = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * n_filter, alignment);
 * ...
 * // push back input IQ data into state.
 * foo_push_back_queue(state, input);
 *
 * // get filtered output.
 * lv_32fc_t output = lv_cmake(0.f,0.f);
 * for (int i = 0; i < n_filter; i++) {
 *   output += state[i] * weight[i];
 * }
 *
 * // update weight using output.
 * float real = lv_creal(output) * (1.0 - std::norm(output)) * MU;
 * lv_32fc_t factor = lv_cmake(real, 0.f);
 * volk_32fc_x2_s32fc_multiply_conjugate_add_32fc(next, weight, state, factor, n_filter);
 * lv_32fc_t *tmp = next;
 * next = weight;
 * weight = tmp;
 * weight[N + 1] = lv_cmake(lv_creal(weight[N + 1]), 0.f);
 * ...
 * volk_free(state);
 * volk_free(weight);
 * volk_free(next);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_H
#define INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc.h>
#include <volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_generic(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* bVector,
                                                       const lv_32fc_t scalar,
                                                       unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_generic(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_avx(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     const lv_32fc_t scalar,
                                                     unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_avx(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_sse3(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_sse3(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_avx(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     const lv_32fc_t scalar,
                                                     unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_avx(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_sse3(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_sse3(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_neon(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    const lv_32fc_t scalar,
                                                    unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_neon(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_neonv8(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_neonv8(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_H */
