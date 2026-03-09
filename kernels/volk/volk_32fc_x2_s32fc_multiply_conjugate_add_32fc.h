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
 * This kernel is deprecated, because passing in \p lv_32fc_t by value results in
 * undefined behavior, causing a segmentation fault on some architectures.
 * Use \ref volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc instead.
 *
 * \b Overview
 *
 * Computes the conjugate of each element in a complex input vector, multiplies
 * by a complex scalar, and adds the result to a second complex input vector.
 * For each element the operation is:
 *
 * \f[
 * c[i] = a[i] + \operatorname{conj}(b[i]) \times \text{scalar}
 * \f]
 *
 * This pattern appears in adaptive filtering algorithms such as LMS (Least Mean
 * Squares), where filter weights are updated using the conjugate of the input
 * signal.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_s32fc_multiply_conjugate_add_32fc(lv_32fc_t* cVector,
 *     const lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t scalar,
 *     unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: Complex input vector to be added, of length \p num_points (lv_32fc_t).
 * \li bVector: Complex input vector to be conjugated and multiplied, of length
 *     \p num_points (lv_32fc_t).
 * \li scalar: Complex scalar multiplied against the conjugated elements of \p bVector
 *     (lv_32fc_t, passed by value).
 * \li num_points: The number of complex elements to process.
 *
 * \b Outputs
 * \li cVector: Complex output vector of length \p num_points (lv_32fc_t).
 *
 * \b Example
 * Conjugate-multiply a vector by a scalar and add to another vector.
 * \code
 * #include <volk/volk.h>
 * #include <stdio.h>
 *
 * int main() {
 *     unsigned int N = 4;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     lv_32fc_t* aVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *     lv_32fc_t* bVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *     lv_32fc_t* cVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 *     // Initialize inputs
 *     for (unsigned int i = 0; i < N; ++i) {
 *         aVector[i] = lv_cmake(1.0f, 0.5f);
 *         bVector[i] = lv_cmake(2.0f, -1.0f);
 *     }
 *     lv_32fc_t scalar = lv_cmake(0.5f, 0.0f);
 *
 *     // c[i] = a[i] + conj(b[i]) * scalar
 *     volk_32fc_x2_s32fc_multiply_conjugate_add_32fc(cVector, aVector, bVector,
 *         scalar, N);
 *
 *     for (unsigned int i = 0; i < N; ++i) {
 *         printf("c[%u] = %+1.4f %+1.4fj\n",
 *             i, lv_creal(cVector[i]), lv_cimag(cVector[i]));
 *     }
 *
 *     volk_free(aVector);
 *     volk_free(bVector);
 *     volk_free(cVector);
 *     return 0;
 * }
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
