/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

/*!
 * \page volk_32u_popcnt
 *
 * \b Overview
 *
 * Computes the population count (popcnt), or Hamming distance of a
 * binary string. This kernel takes in a single unsigned 32-bit value
 * and returns the count of 1's that the value contains.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32u_popcnt(uint32_t* ret, const uint32_t value)
 * \endcode
 *
 * \b Inputs
 * \li value: The input value.
 *
 * \b Outputs
 * \li ret: The return value containing the popcnt.
 *
 * \b Example
 * \code
    int N = 10;
    unsigned int alignment = volk_get_alignment();

    uint32_t bitstring = 0x55555555;
    uint32_t hamming_distance = 0;

    volk_32u_popcnt(&hamming_distance, bitstring);
    printf("hamming distance of %x = %i\n", bitstring, hamming_distance);
 * \endcode
 */

#ifndef INCLUDED_VOLK_32u_POPCNT_A16_H
#define INCLUDED_VOLK_32u_POPCNT_A16_H

#include <stdio.h>
#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32u_popcnt_generic(uint32_t* ret, const uint32_t value)
{
  // This is faster than a lookup table
  uint32_t retVal = value;

  retVal = (retVal & 0x55555555) + (retVal >> 1 & 0x55555555);
  retVal = (retVal & 0x33333333) + (retVal >> 2 & 0x33333333);
  retVal = (retVal + (retVal >> 4)) & 0x0F0F0F0F;
  retVal = (retVal + (retVal >> 8));
  retVal = (retVal + (retVal >> 16)) & 0x0000003F;

  *ret = retVal;
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE4_2

#include <nmmintrin.h>

static inline void
volk_32u_popcnt_a_sse4_2(uint32_t* ret, const uint32_t value)
{
  *ret = _mm_popcnt_u32(value);
}

#endif /*LV_HAVE_SSE4_2*/

#endif /*INCLUDED_VOLK_32u_POPCNT_A16_H*/
