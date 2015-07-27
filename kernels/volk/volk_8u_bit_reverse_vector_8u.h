/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
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
 * \page volk_8u_bit_reverse_vector_8u
 *
 * \b Overview
 *
 * reverses unpacked bits according to bit reversed positions.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8u_bit_reverse_vector_8u(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li out_buf: target buffer for reversed bits
 * \li in_buf: source buffer in natural bit order
 * \li num_points: power of 2. Number of bits to be reversed.
 *
 * \b Outputs
 * \li out_but: bits in bit-reversed order.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_8u_bit_reverse_vector_8u(out_buf, in_buf, num_points);
 *
 * volk_free(x);
 * \endcode
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_BIT_REVERSE_VECTOR_8U_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_BIT_REVERSE_VECTOR_8U_H_

//static inline void shuffle_vector16(unsigned char* outputPtr, const unsigned char* inputPtr){
//  const unsigned int bit_reversed_positions16 [16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
//  unsigned int point;
//  for(point = 0; point < 16; ++point){
//    *outputPtr++ = inputPtr[bit_reversed_positions16[point]];
//  }
//}
//
//static inline unsigned int next_lower_power_of_two(unsigned int num){
//  unsigned int ret = 0x01;
//  while(num){
//    num = num >> 1;
//    ret = ret << 1;
//  }
//  return ret >> 1;
//}
//
#ifdef LV_HAVE_GENERIC
#include <stdio.h>

static inline void volk_8u_bit_reverse_vector_8u_generic(unsigned char* out_buf, const unsigned char* in_buf, unsigned int num_points){
//
//  unsigned int block;
//  const unsigned char* inputPtr = in_buf;
//  unsigned char* outputPtr = out_buf;
//
//  const unsigned int block_size = 16;
//  const unsigned int num_block = next_lower_power_of_two(num_points / block_size);
//
//  for(block = 0; block < num_block; ++block){
//    shuffle_vector16(outputPtr, inputPtr);
//    outputPtr += block_size;
//    inputPtr += block_size;
//  }

}
#endif /* LV_HAVE_GENERIC */



#endif /* VOLK_KERNELS_VOLK_VOLK_8U_BIT_REVERSE_VECTOR_8U_H_ */
