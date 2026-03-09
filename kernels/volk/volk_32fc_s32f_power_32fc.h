/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_power_32fc
 *
 * \b Overview
 *
 * Raises each complex input value to the specified real-valued power and stores
 * the results in the output vector. The computation is performed in polar form.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_power_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const
 * float power, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: The complex float input vector.
 * \li power: The real-valued exponent applied to each element.
 * \li num_points: The number of complex values to process.
 *
 * \b Outputs
 * \li cVector: The complex float output vector.
 *
 * \b Example
 * Raise a vector of complex values to the power of 2.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main(){
 *     unsigned int N = 4;
 *     unsigned int alignment = volk_get_alignment();
 *     float power = 2.0f;
 *
 *     lv_32fc_t* input  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *     lv_32fc_t* output = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 *     // Initialize with some complex values
 *     input[0] = lv_cmake(1.0f, 0.0f);
 *     input[1] = lv_cmake(0.0f, 1.0f);
 *     input[2] = lv_cmake(1.0f, 1.0f);
 *     input[3] = lv_cmake(2.0f, -1.0f);
 *
 *     volk_32fc_s32f_power_32fc(output, input, power, N);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *       printf("(%1.2f, %1.2f)^%1.1f = (%1.4f, %1.4f)\n",
 *              lv_creal(input[i]), lv_cimag(input[i]), power,
 *              lv_creal(output[i]), lv_cimag(output[i]));
 *     }
 *
 *     volk_free(input);
 *     volk_free(output);
 *     return 0;
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_power_32fc_a_H
#define INCLUDED_volk_32fc_s32f_power_32fc_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

//! raise a complex float to a real float power
static inline lv_32fc_t __volk_s32fc_s32f_power_s32fc_a(const lv_32fc_t exp,
                                                        const float power)
{
    const float arg = power * atan2f(lv_creal(exp), lv_cimag(exp));
    const float mag =
        powf(lv_creal(exp) * lv_creal(exp) + lv_cimag(exp) * lv_cimag(exp), power / 2);
    return mag * lv_cmake(-cosf(arg), sinf(arg));
}

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32f_power_32fc_generic(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const float power,
                                                     unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = __volk_s32fc_s32f_power_s32fc_a((*aPtr++), power);
    }
}

#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_s32f_power_32fc_a_H */
