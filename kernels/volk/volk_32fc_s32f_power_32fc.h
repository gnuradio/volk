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
 * Takes each the input complex vector value to the specified power
 * and stores the results in the return vector. The output is scaled
 * and converted to 16-bit shorts.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_power_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const
 * float power, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: The complex input vector.
 * \li power: The power value to be applied to each data point.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li cVector: The output value as 16-bit shorts.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_32fc_s32f_power_32fc();
 *
 * volk_free(x);
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
