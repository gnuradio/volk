/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_null_32f
 *
 * \b Overview
 *
 * Copies 32-bit floating-point values from the input vector to the output
 * vector. This is a null (pass-through) kernel useful as a baseline for
 * benchmarking memory bandwidth.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_null_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li num_points: The number of float values to copy.
 *
 * \b Outputs
 * \li bVector: The output vector where values are copied to.
 *
 * \b Example
 * Copy a small vector of floats through the null kernel.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main() {
 *     unsigned int N = 5;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     // Allocate aligned input and output buffers
 *     float* input = (float*)volk_malloc(N * sizeof(float), alignment);
 *     float* output = (float*)volk_malloc(N * sizeof(float), alignment);
 *
 *     // Fill input with sample values
 *     input[0] = 1.5f;
 *     input[1] = -3.14f;
 *     input[2] = 0.0f;
 *     input[3] = 42.0f;
 *     input[4] = 7.77f;
 *
 *     // Copy input to output
 *     volk_32f_null_32f(output, input, N);
 *
 *     // Verify the copy
 *     for (unsigned int i = 0; i < N; i++) {
 *       printf("output[%u] = %f\n", i, output[i]);
 *     }
 *
 *     volk_free(input);
 *     volk_free(output);
 *     return 0;
 *   }
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_null_32f_a_H
#define INCLUDED_volk_32f_null_32f_a_H

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_null_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = *aPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_null_32f_u_H */
