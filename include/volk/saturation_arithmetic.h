/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#ifndef INCLUDED_volk_saturation_arithmetic_H_
#define INCLUDED_volk_saturation_arithmetic_H_

#include <limits.h>

static inline int16_t sat_adds16i(int16_t x, int16_t y)
{
    int32_t res = (int32_t)x + (int32_t)y;

    if (res < SHRT_MIN)
        res = SHRT_MIN;
    if (res > SHRT_MAX)
        res = SHRT_MAX;

    return res;
}

static inline int16_t sat_muls16i(int16_t x, int16_t y)
{
    int32_t res = (int32_t)x * (int32_t)y;

    if (res < SHRT_MIN)
        res = SHRT_MIN;
    if (res > SHRT_MAX)
        res = SHRT_MAX;

    return res;
}

#endif /* INCLUDED_volk_saturation_arithmetic_H_ */
