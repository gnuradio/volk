/* -*- c++ -*- */
/*
 * Copyright 2015, 2026 Free Software Foundation, Inc.
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#ifndef INCLUDED_VOLK_POLAR_HELPERS_H
#define INCLUDED_VOLK_POLAR_HELPERS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static inline uint32_t log2_of_power_of_2(const uint32_t value)
{
    // algorithm from: https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    static const uint32_t b[] = { UINT32_C(0xAAAAAAAA),
                                  UINT32_C(0xCCCCCCCC),
                                  UINT32_C(0xF0F0F0F0),
                                  UINT32_C(0xFF00FF00),
                                  UINT32_C(0xFFFF0000) };

    uint32_t res = (value & b[0]) != 0;
    res |= (uint32_t)((value & b[4]) != 0) << 4;
    res |= (uint32_t)((value & b[3]) != 0) << 3;
    res |= (uint32_t)((value & b[2]) != 0) << 2;
    res |= (uint32_t)((value & b[1]) != 0) << 1;
    return res;
}

static inline void encodepolar_single_stage(unsigned char* frame_ptr,
                                            const unsigned char* temp_ptr,
                                            const uint32_t num_branches,
                                            const uint32_t frame_half)
{
    for (uint32_t branch = 0; branch < num_branches; ++branch) {
        for (uint32_t bit = 0; bit < frame_half; ++bit) {
            *frame_ptr = *temp_ptr ^ *(temp_ptr + 1);
            *(frame_ptr + frame_half) = *(temp_ptr + 1);
            ++frame_ptr;
            temp_ptr += 2;
        }
        frame_ptr += frame_half;
    }
}

static inline uint32_t next_lower_power_of_two(const uint32_t val)
{
    // algorithm found and adopted from:
    // https://acius2.blogspot.com/2007/11/calculating-next-power-of-2.html
    uint32_t res = val;
    res = (res >> 1) | res;
    res = (res >> 2) | res;
    res = (res >> 4) | res;
    res = (res >> 8) | res;
    res = (res >> 16) | res;
    return res - (res >> 1);
}

static inline void adjust_frozen_mask(unsigned char* mask, size_t frame_size)
{
    // just like the rest of the puppet this function exists for test purposes only.
    for (size_t i = 0; i < frame_size; ++i) {
        *mask = (*mask & 0x80) ? 0xFF : 0x00;
        mask++;
    }
}

static inline float llr_odd(const float la, const float lb)
{
    const float ala = fabsf(la);
    const float alb = fabsf(lb);
    return copysignf(1.0f, la) * copysignf(1.0f, lb) * (ala > alb ? alb : ala);
}

static inline void llr_odd_stages(float* llrs,
                                  const int min_stage,
                                  const int depth,
                                  const int frame_size,
                                  const int row)
{
    for (int loop_stage = depth - 1; loop_stage >= min_stage; --loop_stage) {
        float* dst_llr_ptr = llrs + loop_stage * frame_size + row;
        const float* src_llr_ptr = dst_llr_ptr + frame_size;
        const int stage_size = 1 << loop_stage;
        for (int el = 0; el < stage_size; ++el) {
            *dst_llr_ptr++ = llr_odd(*src_llr_ptr, *(src_llr_ptr + 1));
            src_llr_ptr += 2;
        }
    }
}

static inline float llr_even(const float la, const float lb, const unsigned char f)
{
    return f == 0 ? lb + la : lb - la;
}

static inline void
even_u_values(unsigned char* u_even, const unsigned char* u, const int u_num)
{
    ++u;
    for (int i = 1; i < u_num; i += 2) {
        *u_even++ = *u;
        u += 2;
    }
}

static inline void
odd_xor_even_values(unsigned char* u_xor, const unsigned char* u, const int u_num)
{
    for (int i = 1; i < u_num; i += 2) {
        *u_xor++ = *u ^ *(u + 1);
        u += 2;
    }
}

static inline int calculate_max_stage_depth_for_row(const int frame_exp, const int row)
{
    int max_stage_depth = 0;
    int half_stage_size = 0x01;
    int stage_size = half_stage_size << 1;
    while (max_stage_depth < (frame_exp - 1)) { // last stage holds received values.
        if (!(row % stage_size < half_stage_size)) {
            break;
        }
        half_stage_size <<= 1;
        stage_size <<= 1;
        ++max_stage_depth;
    }
    return max_stage_depth;
}

static inline void interleave_frozen_and_info_bits(unsigned char* target,
                                                   const unsigned char* frozen_bit_mask,
                                                   const unsigned char* frozen_bits,
                                                   const unsigned char* info_bits,
                                                   const uint32_t frame_size)
{
    for (uint32_t bit = 0; bit < frame_size; ++bit) {
        *target++ = *frozen_bit_mask++ ? *frozen_bits++ : *info_bits++;
    }
}

static inline void sanitize_bytes(unsigned char* u, const size_t elements)
{
    for (size_t i = 0; i < elements; ++i) {
        u[i] &= 0x01;
    }
}

static inline void clean_up_intermediate_values(float* llrs,
                                                unsigned char* u,
                                                const size_t frame_size,
                                                const size_t elements)
{
    memset(u + frame_size, 0, sizeof(unsigned char) * (elements - frame_size));
    memset(llrs + frame_size, 0, sizeof(float) * (elements - frame_size));
}

static inline void generate_error_free_input_vector(float* llrs,
                                                    unsigned char* u,
                                                    const uint32_t frame_size,
                                                    void (*encode)(unsigned char*,
                                                                   unsigned char*,
                                                                   unsigned int))
{
    memset(u, 0, frame_size);
    unsigned char* target = u + frame_size;
    encode(target, u + 2 * frame_size, frame_size);
    float* ft = llrs;
    for (uint32_t i = 0; i < frame_size; ++i) {
        *ft = (-2 * ((float)*target++)) + 1.0f;
        ft++;
    }
}

static inline void
print_llr_tree(const float* llrs, const size_t frame_size, const size_t frame_exp)
{
    for (size_t s = 0; s < frame_size; s++) {
        for (size_t e = 0; e < frame_exp + 1; e++) {
            printf("%+4.2f ", llrs[e * frame_size + s]);
        }
        printf("\n");
        if ((s + 1) % 8 == 0) {
            printf("\n");
        }
    }
}

static inline uint32_t maximum_frame_size(const uint32_t elements)
{
    const uint32_t frame_size = next_lower_power_of_two(elements);
    const uint32_t frame_exp = log2_of_power_of_2(frame_size);
    return next_lower_power_of_two(frame_size / frame_exp);
}

#endif /* INCLUDED_VOLK_POLAR_HELPERS_H */
