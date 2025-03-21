/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * for documentation see 'volk_8u_x3_encodepolar_8u_x2.h'
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_U_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_U_H_
#include <string.h>

static inline unsigned int log2_of_power_of_2(unsigned int val)
{
    // algorithm from: https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    static const unsigned int b[] = {
        0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000
    };

    unsigned int res = (val & b[0]) != 0;
    res |= ((val & b[4]) != 0) << 4;
    res |= ((val & b[3]) != 0) << 3;
    res |= ((val & b[2]) != 0) << 2;
    res |= ((val & b[1]) != 0) << 1;
    return res;
}

static inline void encodepolar_single_stage(unsigned char* frame_ptr,
                                            const unsigned char* temp_ptr,
                                            const unsigned int num_branches,
                                            const unsigned int frame_half)
{
    unsigned int branch, bit;
    for (branch = 0; branch < num_branches; ++branch) {
        for (bit = 0; bit < frame_half; ++bit) {
            *frame_ptr = *temp_ptr ^ *(temp_ptr + 1);
            *(frame_ptr + frame_half) = *(temp_ptr + 1);
            ++frame_ptr;
            temp_ptr += 2;
        }
        frame_ptr += frame_half;
    }
}

#ifdef LV_HAVE_GENERIC

static inline void volk_8u_x2_encodeframepolar_8u_generic(unsigned char* frame,
                                                          unsigned char* temp,
                                                          unsigned int frame_size)
{
    unsigned int stage = log2_of_power_of_2(frame_size);
    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;

    while (stage) {
        // encode stage
        encodepolar_single_stage(frame, temp, num_branches, frame_half);
        memcpy(temp, frame, sizeof(unsigned char) * frame_size);

        // update all the parameters.
        num_branches = num_branches << 1;
        frame_half = frame_half >> 1;
        --stage;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_u_ssse3(unsigned char* frame,
                                                          unsigned char* temp,
                                                          unsigned int frame_size)
{
    if (frame_size < 16) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    // prepare constants
    const __m128i mask_stage1 = _mm_set_epi8(0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF);

    // get some SIMD registers to play with.
    __m128i r_frame0, r_temp0, shifted;

    {
        __m128i r_frame1, r_temp1;
        const __m128i shuffle_separate =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            // for stage = 5 a branch has 32 elements. So upper stages are even bigger.
            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 16) {
                    r_temp0 = _mm_loadu_si128((__m128i*)temp_ptr);
                    temp_ptr += 16;
                    r_temp1 = _mm_loadu_si128((__m128i*)temp_ptr);
                    temp_ptr += 16;

                    shifted = _mm_srli_si128(r_temp0, 1);
                    shifted = _mm_and_si128(shifted, mask_stage1);
                    r_temp0 = _mm_xor_si128(shifted, r_temp0);
                    r_temp0 = _mm_shuffle_epi8(r_temp0, shuffle_separate);

                    shifted = _mm_srli_si128(r_temp1, 1);
                    shifted = _mm_and_si128(shifted, mask_stage1);
                    r_temp1 = _mm_xor_si128(shifted, r_temp1);
                    r_temp1 = _mm_shuffle_epi8(r_temp1, shuffle_separate);

                    r_frame0 = _mm_unpacklo_epi64(r_temp0, r_temp1);
                    _mm_storeu_si128((__m128i*)frame_ptr, r_frame0);

                    r_frame1 = _mm_unpackhi_epi64(r_temp0, r_temp1);
                    _mm_storeu_si128((__m128i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 16;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    // This last part requires at least 16-bit frames.
    // Smaller frames are useless for SIMD optimization anyways. Just choose GENERIC!

    // reset pointers to correct positions.
    frame_ptr = frame;
    temp_ptr = temp;

    // prefetch first chunk
    __VOLK_PREFETCH(temp_ptr);

    const __m128i shuffle_stage4 =
        _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    const __m128i mask_stage4 = _mm_set_epi8(0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF);
    const __m128i mask_stage3 = _mm_set_epi8(0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF);
    const __m128i mask_stage2 = _mm_set_epi8(0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF);

    for (branch = 0; branch < num_branches; ++branch) {
        r_temp0 = _mm_loadu_si128((__m128i*)temp_ptr);

        // prefetch next chunk
        temp_ptr += 16;
        __VOLK_PREFETCH(temp_ptr);

        // shuffle once for bit-reversal.
        r_temp0 = _mm_shuffle_epi8(r_temp0, shuffle_stage4);

        shifted = _mm_srli_si128(r_temp0, 8);
        shifted = _mm_and_si128(shifted, mask_stage4);
        r_frame0 = _mm_xor_si128(shifted, r_temp0);

        shifted = _mm_srli_si128(r_frame0, 4);
        shifted = _mm_and_si128(shifted, mask_stage3);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        shifted = _mm_srli_si128(r_frame0, 2);
        shifted = _mm_and_si128(shifted, mask_stage2);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        shifted = _mm_srli_si128(r_frame0, 1);
        shifted = _mm_and_si128(shifted, mask_stage1);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        // store result of chunk.
        _mm_storeu_si128((__m128i*)frame_ptr, r_frame0);
        frame_ptr += 16;
    }
}

#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_u_avx2(unsigned char* frame,
                                                         unsigned char* temp,
                                                         unsigned int frame_size)
{
    if (frame_size < 32) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    // prepare constants
    const __m256i mask_stage1 = _mm256_set_epi8(0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF);

    const __m128i mask_stage0 = _mm_set_epi8(0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF);
    // get some SIMD registers to play with.
    __m256i r_frame0, r_temp0, shifted;
    __m128i r_temp2, r_frame2, shifted2;
    {
        __m256i r_frame1, r_temp1;
        __m128i r_frame3, r_temp3;
        const __m256i shuffle_separate = _mm256_setr_epi8(0,
                                                          2,
                                                          4,
                                                          6,
                                                          8,
                                                          10,
                                                          12,
                                                          14,
                                                          1,
                                                          3,
                                                          5,
                                                          7,
                                                          9,
                                                          11,
                                                          13,
                                                          15,
                                                          0,
                                                          2,
                                                          4,
                                                          6,
                                                          8,
                                                          10,
                                                          12,
                                                          14,
                                                          1,
                                                          3,
                                                          5,
                                                          7,
                                                          9,
                                                          11,
                                                          13,
                                                          15);
        const __m128i shuffle_separate128 =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            // for stage = 5 a branch has 32 elements. So upper stages are even bigger.
            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 32) {
                    if ((frame_half - bit) <
                        32) // if only 16 bits remaining in frame, not 32
                    {
                        r_temp2 = _mm_loadu_si128((__m128i*)temp_ptr);
                        temp_ptr += 16;
                        r_temp3 = _mm_loadu_si128((__m128i*)temp_ptr);
                        temp_ptr += 16;

                        shifted2 = _mm_srli_si128(r_temp2, 1);
                        shifted2 = _mm_and_si128(shifted2, mask_stage0);
                        r_temp2 = _mm_xor_si128(shifted2, r_temp2);
                        r_temp2 = _mm_shuffle_epi8(r_temp2, shuffle_separate128);

                        shifted2 = _mm_srli_si128(r_temp3, 1);
                        shifted2 = _mm_and_si128(shifted2, mask_stage0);
                        r_temp3 = _mm_xor_si128(shifted2, r_temp3);
                        r_temp3 = _mm_shuffle_epi8(r_temp3, shuffle_separate128);

                        r_frame2 = _mm_unpacklo_epi64(r_temp2, r_temp3);
                        _mm_storeu_si128((__m128i*)frame_ptr, r_frame2);

                        r_frame3 = _mm_unpackhi_epi64(r_temp2, r_temp3);
                        _mm_storeu_si128((__m128i*)(frame_ptr + frame_half), r_frame3);
                        frame_ptr += 16;
                        break;
                    }
                    r_temp0 = _mm256_loadu_si256((__m256i*)temp_ptr);
                    temp_ptr += 32;
                    r_temp1 = _mm256_loadu_si256((__m256i*)temp_ptr);
                    temp_ptr += 32;

                    shifted = _mm256_srli_si256(r_temp0, 1); // operate on 128 bit lanes
                    shifted = _mm256_and_si256(shifted, mask_stage1);
                    r_temp0 = _mm256_xor_si256(shifted, r_temp0);
                    r_temp0 = _mm256_shuffle_epi8(r_temp0, shuffle_separate);

                    shifted = _mm256_srli_si256(r_temp1, 1);
                    shifted = _mm256_and_si256(shifted, mask_stage1);
                    r_temp1 = _mm256_xor_si256(shifted, r_temp1);
                    r_temp1 = _mm256_shuffle_epi8(r_temp1, shuffle_separate);

                    r_frame0 = _mm256_unpacklo_epi64(r_temp0, r_temp1);
                    r_temp1 = _mm256_unpackhi_epi64(r_temp0, r_temp1);
                    r_frame0 = _mm256_permute4x64_epi64(r_frame0, 0xd8);
                    r_frame1 = _mm256_permute4x64_epi64(r_temp1, 0xd8);

                    _mm256_storeu_si256((__m256i*)frame_ptr, r_frame0);

                    _mm256_storeu_si256((__m256i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 32;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    // This last part requires at least 32-bit frames.
    // Smaller frames are useless for SIMD optimization anyways. Just choose GENERIC!

    // reset pointers to correct positions.
    frame_ptr = frame;
    temp_ptr = temp;

    // prefetch first chunk
    __VOLK_PREFETCH(temp_ptr);

    const __m256i shuffle_stage4 = _mm256_setr_epi8(0,
                                                    8,
                                                    4,
                                                    12,
                                                    2,
                                                    10,
                                                    6,
                                                    14,
                                                    1,
                                                    9,
                                                    5,
                                                    13,
                                                    3,
                                                    11,
                                                    7,
                                                    15,
                                                    0,
                                                    8,
                                                    4,
                                                    12,
                                                    2,
                                                    10,
                                                    6,
                                                    14,
                                                    1,
                                                    9,
                                                    5,
                                                    13,
                                                    3,
                                                    11,
                                                    7,
                                                    15);
    const __m256i mask_stage4 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF);
    const __m256i mask_stage3 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF);
    const __m256i mask_stage2 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF);

    for (branch = 0; branch < num_branches / 2; ++branch) {
        r_temp0 = _mm256_loadu_si256((__m256i*)temp_ptr);

        // prefetch next chunk
        temp_ptr += 32;
        __VOLK_PREFETCH(temp_ptr);

        // shuffle once for bit-reversal.
        r_temp0 = _mm256_shuffle_epi8(r_temp0, shuffle_stage4);

        shifted = _mm256_srli_si256(r_temp0, 8); // 128 bit lanes
        shifted = _mm256_and_si256(shifted, mask_stage4);
        r_frame0 = _mm256_xor_si256(shifted, r_temp0);


        shifted = _mm256_srli_si256(r_frame0, 4);
        shifted = _mm256_and_si256(shifted, mask_stage3);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        shifted = _mm256_srli_si256(r_frame0, 2);
        shifted = _mm256_and_si256(shifted, mask_stage2);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        shifted = _mm256_srli_si256(r_frame0, 1);
        shifted = _mm256_and_si256(shifted, mask_stage1);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        // store result of chunk.
        _mm256_storeu_si256((__m256i*)frame_ptr, r_frame0);
        frame_ptr += 32;
    }
}
#endif /* LV_HAVE_AVX2 */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_U_H_ */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_A_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_A_H_

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_a_ssse3(unsigned char* frame,
                                                          unsigned char* temp,
                                                          unsigned int frame_size)
{
    if (frame_size < 16) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    // prepare constants
    const __m128i mask_stage1 = _mm_set_epi8(0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF);

    // get some SIMD registers to play with.
    __m128i r_frame0, r_temp0, shifted;

    {
        __m128i r_frame1, r_temp1;
        const __m128i shuffle_separate =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            // for stage = 5 a branch has 32 elements. So upper stages are even bigger.
            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 16) {
                    r_temp0 = _mm_load_si128((__m128i*)temp_ptr);
                    temp_ptr += 16;
                    r_temp1 = _mm_load_si128((__m128i*)temp_ptr);
                    temp_ptr += 16;

                    shifted = _mm_srli_si128(r_temp0, 1);
                    shifted = _mm_and_si128(shifted, mask_stage1);
                    r_temp0 = _mm_xor_si128(shifted, r_temp0);
                    r_temp0 = _mm_shuffle_epi8(r_temp0, shuffle_separate);

                    shifted = _mm_srli_si128(r_temp1, 1);
                    shifted = _mm_and_si128(shifted, mask_stage1);
                    r_temp1 = _mm_xor_si128(shifted, r_temp1);
                    r_temp1 = _mm_shuffle_epi8(r_temp1, shuffle_separate);

                    r_frame0 = _mm_unpacklo_epi64(r_temp0, r_temp1);
                    _mm_store_si128((__m128i*)frame_ptr, r_frame0);

                    r_frame1 = _mm_unpackhi_epi64(r_temp0, r_temp1);
                    _mm_store_si128((__m128i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 16;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    // This last part requires at least 16-bit frames.
    // Smaller frames are useless for SIMD optimization anyways. Just choose GENERIC!

    // reset pointers to correct positions.
    frame_ptr = frame;
    temp_ptr = temp;

    // prefetch first chunk
    __VOLK_PREFETCH(temp_ptr);

    const __m128i shuffle_stage4 =
        _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    const __m128i mask_stage4 = _mm_set_epi8(0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF);
    const __m128i mask_stage3 = _mm_set_epi8(0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0xFF,
                                             0xFF);
    const __m128i mask_stage2 = _mm_set_epi8(0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF,
                                             0x0,
                                             0x0,
                                             0xFF,
                                             0xFF);

    for (branch = 0; branch < num_branches; ++branch) {
        r_temp0 = _mm_load_si128((__m128i*)temp_ptr);

        // prefetch next chunk
        temp_ptr += 16;
        __VOLK_PREFETCH(temp_ptr);

        // shuffle once for bit-reversal.
        r_temp0 = _mm_shuffle_epi8(r_temp0, shuffle_stage4);

        shifted = _mm_srli_si128(r_temp0, 8);
        shifted = _mm_and_si128(shifted, mask_stage4);
        r_frame0 = _mm_xor_si128(shifted, r_temp0);

        shifted = _mm_srli_si128(r_frame0, 4);
        shifted = _mm_and_si128(shifted, mask_stage3);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        shifted = _mm_srli_si128(r_frame0, 2);
        shifted = _mm_and_si128(shifted, mask_stage2);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        shifted = _mm_srli_si128(r_frame0, 1);
        shifted = _mm_and_si128(shifted, mask_stage1);
        r_frame0 = _mm_xor_si128(shifted, r_frame0);

        // store result of chunk.
        _mm_store_si128((__m128i*)frame_ptr, r_frame0);
        frame_ptr += 16;
    }
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8u_x2_encodeframepolar_8u_a_avx2(unsigned char* frame,
                                                         unsigned char* temp,
                                                         unsigned int frame_size)
{
    if (frame_size < 32) {
        volk_8u_x2_encodeframepolar_8u_generic(frame, temp, frame_size);
        return;
    }

    const unsigned int po2 = log2_of_power_of_2(frame_size);

    unsigned int stage = po2;
    unsigned char* frame_ptr = frame;
    unsigned char* temp_ptr = temp;

    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;
    unsigned int branch;
    unsigned int bit;

    // prepare constants
    const __m256i mask_stage1 = _mm256_set_epi8(0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF,
                                                0x0,
                                                0xFF);

    const __m128i mask_stage0 = _mm_set_epi8(0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF,
                                             0x0,
                                             0xFF);
    // get some SIMD registers to play with.
    __m256i r_frame0, r_temp0, shifted;
    __m128i r_temp2, r_frame2, shifted2;
    {
        __m256i r_frame1, r_temp1;
        __m128i r_frame3, r_temp3;
        const __m256i shuffle_separate = _mm256_setr_epi8(0,
                                                          2,
                                                          4,
                                                          6,
                                                          8,
                                                          10,
                                                          12,
                                                          14,
                                                          1,
                                                          3,
                                                          5,
                                                          7,
                                                          9,
                                                          11,
                                                          13,
                                                          15,
                                                          0,
                                                          2,
                                                          4,
                                                          6,
                                                          8,
                                                          10,
                                                          12,
                                                          14,
                                                          1,
                                                          3,
                                                          5,
                                                          7,
                                                          9,
                                                          11,
                                                          13,
                                                          15);
        const __m128i shuffle_separate128 =
            _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

        while (stage > 4) {
            frame_ptr = frame;
            temp_ptr = temp;

            // for stage = 5 a branch has 32 elements. So upper stages are even bigger.
            for (branch = 0; branch < num_branches; ++branch) {
                for (bit = 0; bit < frame_half; bit += 32) {
                    if ((frame_half - bit) <
                        32) // if only 16 bits remaining in frame, not 32
                    {
                        r_temp2 = _mm_load_si128((__m128i*)temp_ptr);
                        temp_ptr += 16;
                        r_temp3 = _mm_load_si128((__m128i*)temp_ptr);
                        temp_ptr += 16;

                        shifted2 = _mm_srli_si128(r_temp2, 1);
                        shifted2 = _mm_and_si128(shifted2, mask_stage0);
                        r_temp2 = _mm_xor_si128(shifted2, r_temp2);
                        r_temp2 = _mm_shuffle_epi8(r_temp2, shuffle_separate128);

                        shifted2 = _mm_srli_si128(r_temp3, 1);
                        shifted2 = _mm_and_si128(shifted2, mask_stage0);
                        r_temp3 = _mm_xor_si128(shifted2, r_temp3);
                        r_temp3 = _mm_shuffle_epi8(r_temp3, shuffle_separate128);

                        r_frame2 = _mm_unpacklo_epi64(r_temp2, r_temp3);
                        _mm_store_si128((__m128i*)frame_ptr, r_frame2);

                        r_frame3 = _mm_unpackhi_epi64(r_temp2, r_temp3);
                        _mm_store_si128((__m128i*)(frame_ptr + frame_half), r_frame3);
                        frame_ptr += 16;
                        break;
                    }
                    r_temp0 = _mm256_load_si256((__m256i*)temp_ptr);
                    temp_ptr += 32;
                    r_temp1 = _mm256_load_si256((__m256i*)temp_ptr);
                    temp_ptr += 32;

                    shifted = _mm256_srli_si256(r_temp0, 1); // operate on 128 bit lanes
                    shifted = _mm256_and_si256(shifted, mask_stage1);
                    r_temp0 = _mm256_xor_si256(shifted, r_temp0);
                    r_temp0 = _mm256_shuffle_epi8(r_temp0, shuffle_separate);

                    shifted = _mm256_srli_si256(r_temp1, 1);
                    shifted = _mm256_and_si256(shifted, mask_stage1);
                    r_temp1 = _mm256_xor_si256(shifted, r_temp1);
                    r_temp1 = _mm256_shuffle_epi8(r_temp1, shuffle_separate);

                    r_frame0 = _mm256_unpacklo_epi64(r_temp0, r_temp1);
                    r_temp1 = _mm256_unpackhi_epi64(r_temp0, r_temp1);
                    r_frame0 = _mm256_permute4x64_epi64(r_frame0, 0xd8);
                    r_frame1 = _mm256_permute4x64_epi64(r_temp1, 0xd8);

                    _mm256_store_si256((__m256i*)frame_ptr, r_frame0);

                    _mm256_store_si256((__m256i*)(frame_ptr + frame_half), r_frame1);
                    frame_ptr += 32;
                }

                frame_ptr += frame_half;
            }
            memcpy(temp, frame, sizeof(unsigned char) * frame_size);

            num_branches = num_branches << 1;
            frame_half = frame_half >> 1;
            stage--;
        }
    }

    // This last part requires at least 32-bit frames.
    // Smaller frames are useless for SIMD optimization anyways. Just choose GENERIC!

    // reset pointers to correct positions.
    frame_ptr = frame;
    temp_ptr = temp;

    // prefetch first chunk.
    __VOLK_PREFETCH(temp_ptr);

    const __m256i shuffle_stage4 = _mm256_setr_epi8(0,
                                                    8,
                                                    4,
                                                    12,
                                                    2,
                                                    10,
                                                    6,
                                                    14,
                                                    1,
                                                    9,
                                                    5,
                                                    13,
                                                    3,
                                                    11,
                                                    7,
                                                    15,
                                                    0,
                                                    8,
                                                    4,
                                                    12,
                                                    2,
                                                    10,
                                                    6,
                                                    14,
                                                    1,
                                                    9,
                                                    5,
                                                    13,
                                                    3,
                                                    11,
                                                    7,
                                                    15);
    const __m256i mask_stage4 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF);
    const __m256i mask_stage3 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0xFF,
                                                0xFF);
    const __m256i mask_stage2 = _mm256_set_epi8(0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF,
                                                0x0,
                                                0x0,
                                                0xFF,
                                                0xFF);

    for (branch = 0; branch < num_branches / 2; ++branch) {
        r_temp0 = _mm256_load_si256((__m256i*)temp_ptr);

        // prefetch next chunk
        temp_ptr += 32;
        __VOLK_PREFETCH(temp_ptr);

        // shuffle once for bit-reversal.
        r_temp0 = _mm256_shuffle_epi8(r_temp0, shuffle_stage4);

        shifted = _mm256_srli_si256(r_temp0, 8); // 128 bit lanes
        shifted = _mm256_and_si256(shifted, mask_stage4);
        r_frame0 = _mm256_xor_si256(shifted, r_temp0);

        shifted = _mm256_srli_si256(r_frame0, 4);
        shifted = _mm256_and_si256(shifted, mask_stage3);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        shifted = _mm256_srli_si256(r_frame0, 2);
        shifted = _mm256_and_si256(shifted, mask_stage2);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        shifted = _mm256_srli_si256(r_frame0, 1);
        shifted = _mm256_and_si256(shifted, mask_stage1);
        r_frame0 = _mm256_xor_si256(shifted, r_frame0);

        // store result of chunk.
        _mm256_store_si256((__m256i*)frame_ptr, r_frame0);
        frame_ptr += 32;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8u_x2_encodeframepolar_8u_rvv(unsigned char* frame,
                                                      unsigned char* temp,
                                                      unsigned int frame_size)
{
    unsigned int stage = log2_of_power_of_2(frame_size);
    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;

    while (stage) {
        // encode stage
        if (frame_half < 8) {
            encodepolar_single_stage(frame, temp, num_branches, frame_half);
        } else {
            unsigned char *in = temp, *out = frame;
            for (size_t branch = 0; branch < num_branches; ++branch) {
                size_t n = frame_half;
                for (size_t vl; n > 0; n -= vl, in += vl * 2, out += vl) {
                    vl = __riscv_vsetvl_e8m1(n);
                    vuint16m2_t vc = __riscv_vle16_v_u16m2((uint16_t*)in, vl);
                    vuint8m1_t v1 = __riscv_vnsrl(vc, 0, vl);
                    vuint8m1_t v2 = __riscv_vnsrl(vc, 8, vl);
                    __riscv_vse8(out, __riscv_vxor(v1, v2, vl), vl);
                    __riscv_vse8(out + frame_half, v2, vl);
                }
                out += frame_half;
            }
        }
        memcpy(temp, frame, sizeof(unsigned char) * frame_size);

        // update all the parameters.
        num_branches = num_branches << 1;
        frame_half = frame_half >> 1;
        --stage;
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_8u_x2_encodeframepolar_8u_rvvseg(unsigned char* frame,
                                                         unsigned char* temp,
                                                         unsigned int frame_size)
{
    unsigned int stage = log2_of_power_of_2(frame_size);
    unsigned int frame_half = frame_size >> 1;
    unsigned int num_branches = 1;

    while (stage) {
        // encode stage
        if (frame_half < 8) {
            encodepolar_single_stage(frame, temp, num_branches, frame_half);
        } else {
            unsigned char *in = temp, *out = frame;
            for (size_t branch = 0; branch < num_branches; ++branch) {
                size_t n = frame_half;
                for (size_t vl; n > 0; n -= vl, in += vl * 2, out += vl) {
                    vl = __riscv_vsetvl_e8m1(n);
                    vuint8m1x2_t vc = __riscv_vlseg2e8_v_u8m1x2(in, vl);
                    vuint8m1_t v1 = __riscv_vget_u8m1(vc, 0);
                    vuint8m1_t v2 = __riscv_vget_u8m1(vc, 1);
                    __riscv_vse8(out, __riscv_vxor(v1, v2, vl), vl);
                    __riscv_vse8(out + frame_half, v2, vl);
                }
                out += frame_half;
            }
        }
        memcpy(temp, frame, sizeof(unsigned char) * frame_size);

        // update all the parameters.
        num_branches = num_branches << 1;
        frame_half = frame_half >> 1;
        --stage;
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X2_ENCODEFRAMEPOLAR_8U_A_H_ */
