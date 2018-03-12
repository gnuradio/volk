/* -*- c++ -*- */
/*
  Copyright (C) 2018 Free Software Foundation, Inc.

  This file is pat of libVOLK

  All rights reserved.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License version 2.1, as
  published by the Free Software Foundation.  This program is
  distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
  License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with this program; if not, see <http://www.gnu.org/licenses/>.
*/

/*!
 * \page volk_32u_reverse_32u
 *
 * \b bit reversal of the input 32 bit word

 * <b>Dispatcher Prototype</b>
 * \code volk_32u_reverse_32u(uint32_t *outputVector, uint32_t *inputVector; unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector
 * \li num_points The number of data points.
 *
 * \b Outputs
 * \li outputVector: The vector where the results will be stored, which is the bit-reversed input
 *
 * \endcode
 */
#ifndef INCLUDED_VOLK_32u_REVERSE_32u_U_H
struct dword_split {
  int b00: 1;
  int b01: 1;
  int b02: 1;
  int b03: 1;
  int b04: 1;
  int b05: 1;
  int b06: 1;
  int b07: 1;
  int b08: 1;
  int b09: 1;
  int b10: 1;
  int b11: 1;
  int b12: 1;
  int b13: 1;
  int b14: 1;
  int b15: 1;
  int b16: 1;
  int b17: 1;
  int b18: 1;
  int b19: 1;
  int b20: 1;
  int b21: 1;
  int b22: 1;
  int b23: 1;
  int b24: 1;
  int b25: 1;
  int b26: 1;
  int b27: 1;
  int b28: 1;
  int b29: 1;
  int b30: 1;
  int b31: 1;
};
struct char_split {
  uint8_t b00: 1;
  uint8_t b01: 1;
  uint8_t b02: 1;
  uint8_t b03: 1;
  uint8_t b04: 1;
  uint8_t b05: 1;
  uint8_t b06: 1;
  uint8_t b07: 1;
};

//Idea from "Bit Twiddling Hacks", which dedicates this method to public domain
//http://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable
static const unsigned char BitReverseTable256[] = {
  0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30,
  0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98,
  0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64,
  0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC,
  0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 0x02,
  0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2,
  0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A,
  0xDA, 0x3A, 0xBA, 0x7A, 0xFA, 0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6,
  0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E, 0x4E, 0xCE, 0x2E,
  0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE, 0x01, 0x81,
  0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71,
  0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9,
  0x39, 0xB9, 0x79, 0xF9, 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15,
  0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD,
  0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD, 0x03, 0x83, 0x43,
  0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
  0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B,
  0xBB, 0x7B, 0xFB, 0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97,
  0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F,
  0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};
#ifdef LV_HAVE_GENERIC
static inline void volk_32u_reverse_32u_dword_shuffle(uint32_t* out, const uint32_t* in,
                           unsigned int num_points)
{
  const struct dword_split *in_ptr = (const struct dword_split*)in;
  struct dword_split * out_ptr = (struct dword_split*)out;
  unsigned int number = 0;
  for(; number < num_points; ++number){
    out_ptr->b00 = in_ptr->b31;
    out_ptr->b01 = in_ptr->b30;
    out_ptr->b02 = in_ptr->b29;
    out_ptr->b03 = in_ptr->b28;
    out_ptr->b04 = in_ptr->b27;
    out_ptr->b05 = in_ptr->b26;
    out_ptr->b06 = in_ptr->b25;
    out_ptr->b07 = in_ptr->b24;
    out_ptr->b08 = in_ptr->b23;
    out_ptr->b09 = in_ptr->b22;
    out_ptr->b10 = in_ptr->b21;
    out_ptr->b11 = in_ptr->b20;
    out_ptr->b12 = in_ptr->b19;
    out_ptr->b13 = in_ptr->b18;
    out_ptr->b14 = in_ptr->b17;
    out_ptr->b15 = in_ptr->b16;
    out_ptr->b16 = in_ptr->b15;
    out_ptr->b17 = in_ptr->b14;
    out_ptr->b18 = in_ptr->b13;
    out_ptr->b19 = in_ptr->b12;
    out_ptr->b20 = in_ptr->b11;
    out_ptr->b21 = in_ptr->b10;
    out_ptr->b22 = in_ptr->b09;
    out_ptr->b23 = in_ptr->b08;
    out_ptr->b24 = in_ptr->b07;
    out_ptr->b25 = in_ptr->b06;
    out_ptr->b26 = in_ptr->b05;
    out_ptr->b27 = in_ptr->b04;
    out_ptr->b28 = in_ptr->b03;
    out_ptr->b29 = in_ptr->b02;
    out_ptr->b30 = in_ptr->b01;
    out_ptr->b31 = in_ptr->b00;
    ++in_ptr;
    ++out_ptr;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC
static inline void volk_32u_reverse_32u_byte_shuffle(uint32_t* out, const uint32_t* in,
                           unsigned int num_points)
{
  const uint32_t *in_ptr = in;
  uint32_t *out_ptr = out;
  unsigned int number = 0;
  for(; number < num_points; ++number){
    const struct char_split *in8 = (const struct char_split*)in_ptr;
    struct char_split *out8 = (struct char_split*)out_ptr;

    out8[3].b00 = in8[0].b07;
    out8[3].b01 = in8[0].b06;
    out8[3].b02 = in8[0].b05;
    out8[3].b03 = in8[0].b04;
    out8[3].b04 = in8[0].b03;
    out8[3].b05 = in8[0].b02;
    out8[3].b06 = in8[0].b01;
    out8[3].b07 = in8[0].b00;

    out8[2].b00 = in8[1].b07;
    out8[2].b01 = in8[1].b06;
    out8[2].b02 = in8[1].b05;
    out8[2].b03 = in8[1].b04;
    out8[2].b04 = in8[1].b03;
    out8[2].b05 = in8[1].b02;
    out8[2].b06 = in8[1].b01;
    out8[2].b07 = in8[1].b00;

    out8[1].b00 = in8[2].b07;
    out8[1].b01 = in8[2].b06;
    out8[1].b02 = in8[2].b05;
    out8[1].b03 = in8[2].b04;
    out8[1].b04 = in8[2].b03;
    out8[1].b05 = in8[2].b02;
    out8[1].b06 = in8[2].b01;
    out8[1].b07 = in8[2].b00;

    out8[0].b00 = in8[3].b07;
    out8[0].b01 = in8[3].b06;
    out8[0].b02 = in8[3].b05;
    out8[0].b03 = in8[3].b04;
    out8[0].b04 = in8[3].b03;
    out8[0].b05 = in8[3].b02;
    out8[0].b06 = in8[3].b01;
    out8[0].b07 = in8[3].b00;
    ++in_ptr;
    ++out_ptr;
  }
}
#endif /* LV_HAVE_GENERIC */

//Idea from "Bit Twiddling Hacks", which dedicates this method to public domain
//http://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable
#ifdef LV_HAVE_GENERIC
static inline void volk_32u_reverse_32u_lut(uint32_t* out, const uint32_t* in,
                           unsigned int num_points)
{
  const uint32_t *in_ptr = in;
  uint32_t *out_ptr = out;
  unsigned int number = 0;
  for(; number < num_points; ++number){
    *out_ptr =
      (BitReverseTable256[*in_ptr & 0xff]         << 24) |
      (BitReverseTable256[(*in_ptr >>  8) & 0xff] << 16) |
      (BitReverseTable256[(*in_ptr >> 16) & 0xff] <<  8) |
      (BitReverseTable256[(*in_ptr >> 24) & 0xff]);
    ++in_ptr;
    ++out_ptr;
  }
}
#endif /* LV_HAVE_GENERIC */

//Single-Byte code from "Bit Twiddling Hacks", which dedicates this method to public domain
//http://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64Bits
#ifdef LV_HAVE_GENERIC
static inline void volk_32u_reverse_32u_2001magic(uint32_t* out, const uint32_t* in,
                                           unsigned int num_points)
{
  const uint32_t *in_ptr = in;
  uint32_t *out_ptr = out;
  const uint8_t *in8;
  uint8_t *out8;
  unsigned int number = 0;
  for(; number < num_points; ++number){
    in8 = (const uint8_t*)in_ptr;
    out8 = (uint8_t*)out_ptr;
    out8[3] = ((in8[0] * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
    out8[2] = ((in8[1] * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
    out8[1] = ((in8[2] * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
    out8[0] = ((in8[3] * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
    ++in_ptr;
    ++out_ptr;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC
// Current gr-pager implementation
static inline void volk_32u_reverse_32u_1972magic(uint32_t* out, const uint32_t* in,
                                                 unsigned int num_points)
{
  const uint32_t *in_ptr = in;
  uint32_t *out_ptr = out;
  const uint8_t *in8;
  uint8_t *out8;
  unsigned int number = 0;
  for(; number < num_points; ++number){
    in8 = (const uint8_t*)in_ptr;
    out8 = (uint8_t*)out_ptr;
    out8[3] =  (in8[0] * 0x0202020202ULL & 0x010884422010ULL) % 1023;
    out8[2] =  (in8[1] * 0x0202020202ULL & 0x010884422010ULL) % 1023;
    out8[1] =  (in8[2] * 0x0202020202ULL & 0x010884422010ULL) % 1023;
    out8[0] =  (in8[3] * 0x0202020202ULL & 0x010884422010ULL) % 1023;
    ++in_ptr;
    ++out_ptr;
  }
}
#endif /* LV_HAVE_GENERIC */

//After lengthy thought and quite a bit of whiteboarding:
#ifdef LV_HAVE_GENERIC
static inline void volk_32u_reverse_32u_bintree_permute_top_down(uint32_t* out, const uint32_t* in,
                                                 unsigned int num_points)
{
  const uint32_t *in_ptr = in;
  uint32_t *out_ptr = out;
  unsigned int number = 0;
  for(; number < num_points; ++number){
    uint32_t tmp = *in_ptr;
    /* permute uint16:
       The idea is to simply shift the lower 16 bit up, and the upper 16 bit down.
     */
    tmp = ( tmp << 16 ) | ( tmp >> 16 );
    /* permute bytes:
       shift up by 1 B first, then only consider even bytes, and OR with the unshifted even bytes
     */
    tmp = ((tmp & (0xFF | 0xFF << 16)) << 8) | ((tmp >> 8) & (0xFF | 0xFF << 16));
    /* permute 4bit tuples:
       Same idea, but the "consideration" mask expression becomes unwieldy
     */
    tmp = ((tmp & (0xF | 0xF << 8 | 0xF << 16 | 0xF << 24)) << 4) | ((tmp >> 4) & (0xF | 0xF << 8 | 0xF << 16 | 0xF << 24));
    /* permute 2bit tuples:
       Here, we collapsed the "consideration" mask to a simple hexmask: 0b0011 =
       3; we need those every 4b, which coincides with a hex digit!
    */
    tmp = ((tmp & (0x33333333)) << 2) | ((tmp >> 2) & (0x33333333));
    /* permute odd/even:
       0x01 = 0x1;  we need these every 2b, which works out: 0x01 | (0x01 << 2) = 0x05!
     */
    tmp = ((tmp & (0x55555555)) << 1) | ((tmp >> 1) & (0x55555555));

    *out_ptr = tmp;
    ++in_ptr;
    ++out_ptr;
  }
}
#endif /* LV_HAVE_GENERIC */
#ifdef LV_HAVE_GENERIC
static inline void volk_32u_reverse_32u_bintree_permute_bottom_up(uint32_t* out, const uint32_t* in,
                                                 unsigned int num_points)
{
  //same stuff as top_down, inverted order (permutation matrices don't care, you know!)
  const uint32_t *in_ptr = in;
  uint32_t *out_ptr = out;
  unsigned int number = 0;
  for(; number < num_points; ++number){
    uint32_t tmp = *in_ptr;
    tmp = ((tmp & (0x55555555)) << 1) | ((tmp >> 1) & (0x55555555));
    tmp = ((tmp & (0x33333333)) << 2) | ((tmp >> 2) & (0x33333333));
    tmp = ((tmp & (0xF | 0xF << 8 | 0xF << 16 | 0xF << 24)) << 4) | ((tmp >> 4) & (0xF | 0xF << 8 | 0xF << 16 | 0xF << 24));
    tmp = ((tmp & (0xFF | 0xFF << 16)) << 8) | ((tmp >> 8) & (0xFF | 0xFF << 16));
    tmp = ( tmp << 16 ) | ( tmp >> 16 );

    *out_ptr = tmp;
    ++in_ptr;
    ++out_ptr;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

#define DO_RBIT                                          \
    asm("rbit %1,%0" : "=r" (*out_ptr) : "r" (*in_ptr)); \
    in_ptr++;                                            \
    out_ptr++;

static inline void volk_32u_reverse_32u_arm(uint32_t* out, const uint32_t* in,
                                            unsigned int num_points)
{

    const uint32_t *in_ptr = in;
    uint32_t *out_ptr = out;
    const unsigned int eighthPoints = num_points/8;
    unsigned int number = 0;
    for(; number < eighthPoints; ++number){
        __VOLK_PREFETCH(in_ptr+8);
        DO_RBIT; DO_RBIT; DO_RBIT; DO_RBIT;
        DO_RBIT; DO_RBIT; DO_RBIT; DO_RBIT;
    }
    number = eighthPoints*8;
    for(; number < num_points; ++number){
        DO_RBIT;
    }
}
#undef DO_RBIT
#endif /* LV_HAVE_NEON */


#endif /* INCLUDED_volk_32u_reverse_32u_u_H */

