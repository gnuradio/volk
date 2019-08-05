/* -*- c++ -*- */
/*
 * Copyright 2012, 2013, 2014 Free Software Foundation, Inc.
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
 * \page volk_32fc_s32fc_x2_rotator_32fc
 *
 * \b Overview
 *
 * Rotate input vector at fixed rate per sample from initial phase
 * offset.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_x2_rotator_32fc(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inVector: Vector to be rotated.
 * \li phase_inc: rotational velocity.
 * \li phase: initial phase offset.
 * \li num_points: The number of values in inVector to be rotated and stored into outVector.
 *
 * \b Outputs
 * \li outVector: The vector where the results will be stored.
 *
 * \b Example
 * Generate a tone at f=0.3 (normalized frequency) and use the rotator with
 * f=0.1 to shift the tone to f=0.4. Change this example to start with a DC
 * tone (initialize in with lv_cmake(1, 0)) to observe rotator signal generation.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       // Generate a tone at f=0.3
 *       float real = std::cos(0.3f * (float)ii);
 *       float imag = std::sin(0.3f * (float)ii);
 *       in[ii] = lv_cmake(real, imag);
 *   }
 *   // The oscillator rotates at f=0.1
 *   float frequency = 0.1f;
 *   lv_32fc_t phase_increment = lv_cmake(std::cos(frequency), std::sin(frequency));
 *   lv_32fc_t phase= lv_cmake(1.f, 0.0f); // start at 1 (0 rad phase)
 *
 *   // rotate so the output is a tone at f=0.4
 *   volk_32fc_s32fc_x2_rotator_32fc(out, in, phase_increment, &phase, N);
 *
 *   // print results for inspection
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %+1.2f %+1.2fj\n",
 *           ii, lv_creal(out[ii]), lv_cimag(out[ii]));
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H
#define INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H


#include <volk/volk_complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define ROTATOR_RELOAD 512


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32fc_x2_rotator_32fc_generic(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points){
    unsigned int i = 0;
    int j = 0;
    for(i = 0; i < (unsigned int)(num_points/ROTATOR_RELOAD); ++i) {
        for(j = 0; j < ROTATOR_RELOAD; ++j) {
            *outVector++ = *inVector++ * (*phase);
            (*phase) *= phase_inc;
        }
#ifdef __cplusplus
        (*phase) /= std::abs((*phase));
#else
        //(*phase) /= cabsf((*phase));
        (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
#endif
    }
    for(i = 0; i < num_points%ROTATOR_RELOAD; ++i) {
        *outVector++ = *inVector++ * (*phase);
        (*phase) *= phase_inc;
    }

}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void volk_32fc_s32fc_x2_rotator_32fc_neon(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points)

{
    lv_32fc_t* outputVectorPtr = outVector;
    const lv_32fc_t* inputVectorPtr = inVector;
    lv_32fc_t incr = 1;
    lv_32fc_t phasePtr[4] = {(*phase), (*phase), (*phase), (*phase)};
    float32x4x2_t input_vec;
    float32x4x2_t output_vec;
    
    unsigned int i = 0, j = 0;
    const unsigned int quarter_points = num_points / 4;
    
    for(i = 0; i < 4; ++i) {
        phasePtr[i] *= incr;
        incr *= (phase_inc);
    }
    
    // Notice that incr has be incremented in the previous loop
    const lv_32fc_t incrPtr[4] = {incr, incr, incr, incr};
    const float32x4x2_t incr_vec = vld2q_f32((float*) incrPtr);
    float32x4x2_t phase_vec = vld2q_f32((float*) phasePtr);
    
    for(i = 0; i < (unsigned int)(quarter_points/ROTATOR_RELOAD); i++) {
        for(j = 0; j < ROTATOR_RELOAD; j++) {
            input_vec = vld2q_f32((float*) inputVectorPtr);
            // Prefetch next one, speeds things up
            __VOLK_PREFETCH(inputVectorPtr+4);
            // Rotate
            output_vec = _vmultiply_complexq_f32(input_vec, phase_vec);
            // Increase phase
            phase_vec = _vmultiply_complexq_f32(phase_vec, incr_vec);
            // Store output
            vst2q_f32((float*)outputVectorPtr, output_vec);
            
            outputVectorPtr+=4;
            inputVectorPtr+=4;
        }
        // normalize phase so magnitude doesn't grow because of
        // floating point rounding error
        const float32x4_t mag_squared = _vmagnitudesquaredq_f32(phase_vec);
        const float32x4_t inv_mag = _vinvsqrtq_f32(mag_squared);
        // Multiply complex with real
        phase_vec.val[0] = vmulq_f32(phase_vec.val[0], inv_mag);
        phase_vec.val[1] = vmulq_f32(phase_vec.val[1], inv_mag);
    }
    
    for(i = 0; i < quarter_points % ROTATOR_RELOAD; i++) {
        input_vec = vld2q_f32((float*) inputVectorPtr);
        // Prefetch next one, speeds things up
        __VOLK_PREFETCH(inputVectorPtr+4);
        // Rotate
        output_vec = _vmultiply_complexq_f32(input_vec, phase_vec);
        // Increase phase
        phase_vec = _vmultiply_complexq_f32(phase_vec, incr_vec);
        // Store output
        vst2q_f32((float*)outputVectorPtr, output_vec);
        
        outputVectorPtr+=4;
        inputVectorPtr+=4;
    }
    // if(i) == true means we looped above
    if (i) {
        // normalize phase so magnitude doesn't grow because of
        // floating point rounding error
        const float32x4_t mag_squared = _vmagnitudesquaredq_f32(phase_vec);
        const float32x4_t inv_mag = _vinvsqrtq_f32(mag_squared);
        // Multiply complex with real
        phase_vec.val[0] = vmulq_f32(phase_vec.val[0], inv_mag);
        phase_vec.val[1] = vmulq_f32(phase_vec.val[1], inv_mag);
    }
    // Store current phase
    vst2q_f32((float*)phasePtr, phase_vec);
    
    // Deal with the rest
    for(i = 0; i < num_points % 4; i++) {
        *outputVectorPtr++ = *inputVectorPtr++ * phasePtr[0];
        phasePtr[0] *= (phase_inc);
    }
    
    // For continious phase next time we need to call this function
    (*phase) = phasePtr[0];
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_32fc_s32fc_x2_rotator_32fc_a_sse4_1(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points){
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;
    lv_32fc_t incr = 1;
    lv_32fc_t phase_Ptr[2] = {(*phase), (*phase)};

    unsigned int i, j = 0;

    for(i = 0; i < 2; ++i) {
        phase_Ptr[i] *= incr;
        incr *= (phase_inc);
    }

    /*printf("%f, %f\n", lv_creal(phase_Ptr[0]), lv_cimag(phase_Ptr[0]));
    printf("%f, %f\n", lv_creal(phase_Ptr[1]), lv_cimag(phase_Ptr[1]));
    printf("incr: %f, %f\n", lv_creal(incr), lv_cimag(incr));*/
    __m128 aVal, phase_Val, inc_Val, yl, yh, tmp1, tmp2, z, ylp, yhp, tmp1p, tmp2p;

    phase_Val = _mm_loadu_ps((float*)phase_Ptr);
    inc_Val = _mm_set_ps(lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr));

    const unsigned int halfPoints = num_points / 2;


    for(i = 0; i < (unsigned int)(halfPoints/ROTATOR_RELOAD); i++) {
        for(j = 0; j < ROTATOR_RELOAD; ++j) {

            aVal = _mm_load_ps((float*)aPtr);

            yl = _mm_moveldup_ps(phase_Val);
            yh = _mm_movehdup_ps(phase_Val);
            ylp = _mm_moveldup_ps(inc_Val);
            yhp = _mm_movehdup_ps(inc_Val);

            tmp1 = _mm_mul_ps(aVal, yl);
            tmp1p = _mm_mul_ps(phase_Val, ylp);

            aVal = _mm_shuffle_ps(aVal, aVal, 0xB1);
            phase_Val = _mm_shuffle_ps(phase_Val, phase_Val, 0xB1);
            tmp2 = _mm_mul_ps(aVal, yh);
            tmp2p = _mm_mul_ps(phase_Val, yhp);

            z = _mm_addsub_ps(tmp1, tmp2);
            phase_Val = _mm_addsub_ps(tmp1p, tmp2p);

            _mm_store_ps((float*)cPtr, z);

            aPtr += 2;
            cPtr += 2;
        }
        tmp1 = _mm_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm_hadd_ps(tmp1, tmp1);
        tmp1 = _mm_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm_sqrt_ps(tmp1);
        phase_Val = _mm_div_ps(phase_Val, tmp2);
    }
    for(i = 0; i < halfPoints%ROTATOR_RELOAD; ++i) {
        aVal = _mm_load_ps((float*)aPtr);

        yl = _mm_moveldup_ps(phase_Val);
        yh = _mm_movehdup_ps(phase_Val);
        ylp = _mm_moveldup_ps(inc_Val);
        yhp = _mm_movehdup_ps(inc_Val);

        tmp1 = _mm_mul_ps(aVal, yl);

        tmp1p = _mm_mul_ps(phase_Val, ylp);

        aVal = _mm_shuffle_ps(aVal, aVal, 0xB1);
        phase_Val = _mm_shuffle_ps(phase_Val, phase_Val, 0xB1);
        tmp2 = _mm_mul_ps(aVal, yh);
        tmp2p = _mm_mul_ps(phase_Val, yhp);

        z = _mm_addsub_ps(tmp1, tmp2);
        phase_Val = _mm_addsub_ps(tmp1p, tmp2p);

        _mm_store_ps((float*)cPtr, z);

        aPtr += 2;
        cPtr += 2;
    }
    if (i) {
        tmp1 = _mm_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm_hadd_ps(tmp1, tmp1);
        tmp1 = _mm_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm_sqrt_ps(tmp1);
        phase_Val = _mm_div_ps(phase_Val, tmp2);
    }

    _mm_storeu_ps((float*)phase_Ptr, phase_Val);
    for(i = 0; i < num_points%2; ++i) {
        *cPtr++ = *aPtr++ * phase_Ptr[0];
        phase_Ptr[0] *= (phase_inc);
    }

    (*phase) = phase_Ptr[0];

}

#endif /* LV_HAVE_SSE4_1 for aligned */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_32fc_s32fc_x2_rotator_32fc_u_sse4_1(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points){
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;
    lv_32fc_t incr = 1;
    lv_32fc_t phase_Ptr[2] = {(*phase), (*phase)};

    unsigned int i, j = 0;

    for(i = 0; i < 2; ++i) {
        phase_Ptr[i] *= incr;
        incr *= (phase_inc);
    }

    /*printf("%f, %f\n", lv_creal(phase_Ptr[0]), lv_cimag(phase_Ptr[0]));
    printf("%f, %f\n", lv_creal(phase_Ptr[1]), lv_cimag(phase_Ptr[1]));
    printf("incr: %f, %f\n", lv_creal(incr), lv_cimag(incr));*/
    __m128 aVal, phase_Val, inc_Val, yl, yh, tmp1, tmp2, z, ylp, yhp, tmp1p, tmp2p;

    phase_Val = _mm_loadu_ps((float*)phase_Ptr);
    inc_Val = _mm_set_ps(lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr));

    const unsigned int halfPoints = num_points / 2;


    for(i = 0; i < (unsigned int)(halfPoints/ROTATOR_RELOAD); i++) {
        for(j = 0; j < ROTATOR_RELOAD; ++j) {

            aVal = _mm_loadu_ps((float*)aPtr);

            yl = _mm_moveldup_ps(phase_Val);
            yh = _mm_movehdup_ps(phase_Val);
            ylp = _mm_moveldup_ps(inc_Val);
            yhp = _mm_movehdup_ps(inc_Val);

            tmp1 = _mm_mul_ps(aVal, yl);
            tmp1p = _mm_mul_ps(phase_Val, ylp);

            aVal = _mm_shuffle_ps(aVal, aVal, 0xB1);
            phase_Val = _mm_shuffle_ps(phase_Val, phase_Val, 0xB1);
            tmp2 = _mm_mul_ps(aVal, yh);
            tmp2p = _mm_mul_ps(phase_Val, yhp);

            z = _mm_addsub_ps(tmp1, tmp2);
            phase_Val = _mm_addsub_ps(tmp1p, tmp2p);

            _mm_storeu_ps((float*)cPtr, z);

            aPtr += 2;
            cPtr += 2;
        }
        tmp1 = _mm_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm_hadd_ps(tmp1, tmp1);
        tmp1 = _mm_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm_sqrt_ps(tmp1);
        phase_Val = _mm_div_ps(phase_Val, tmp2);
    }
    for(i = 0; i < halfPoints%ROTATOR_RELOAD; ++i) {
        aVal = _mm_loadu_ps((float*)aPtr);

        yl = _mm_moveldup_ps(phase_Val);
        yh = _mm_movehdup_ps(phase_Val);
        ylp = _mm_moveldup_ps(inc_Val);
        yhp = _mm_movehdup_ps(inc_Val);

        tmp1 = _mm_mul_ps(aVal, yl);

        tmp1p = _mm_mul_ps(phase_Val, ylp);

        aVal = _mm_shuffle_ps(aVal, aVal, 0xB1);
        phase_Val = _mm_shuffle_ps(phase_Val, phase_Val, 0xB1);
        tmp2 = _mm_mul_ps(aVal, yh);
        tmp2p = _mm_mul_ps(phase_Val, yhp);

        z = _mm_addsub_ps(tmp1, tmp2);
        phase_Val = _mm_addsub_ps(tmp1p, tmp2p);

        _mm_storeu_ps((float*)cPtr, z);

        aPtr += 2;
        cPtr += 2;
    }
    if (i) {
        tmp1 = _mm_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm_hadd_ps(tmp1, tmp1);
        tmp1 = _mm_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm_sqrt_ps(tmp1);
        phase_Val = _mm_div_ps(phase_Val, tmp2);
    }

    _mm_storeu_ps((float*)phase_Ptr, phase_Val);
    for(i = 0; i < num_points%2; ++i) {
        *cPtr++ = *aPtr++ * phase_Ptr[0];
        phase_Ptr[0] *= (phase_inc);
    }

    (*phase) = phase_Ptr[0];

}

#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_s32fc_x2_rotator_32fc_a_avx(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points){
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;
    lv_32fc_t incr = 1;
    lv_32fc_t phase_Ptr[4] = {(*phase), (*phase), (*phase), (*phase)};

    unsigned int i, j = 0;

    for(i = 0; i < 4; ++i) {
        phase_Ptr[i] *= incr;
        incr *= (phase_inc);
    }

    /*printf("%f, %f\n", lv_creal(phase_Ptr[0]), lv_cimag(phase_Ptr[0]));
    printf("%f, %f\n", lv_creal(phase_Ptr[1]), lv_cimag(phase_Ptr[1]));
    printf("%f, %f\n", lv_creal(phase_Ptr[2]), lv_cimag(phase_Ptr[2]));
    printf("%f, %f\n", lv_creal(phase_Ptr[3]), lv_cimag(phase_Ptr[3]));
    printf("incr: %f, %f\n", lv_creal(incr), lv_cimag(incr));*/
    __m256 aVal, phase_Val, inc_Val, yl, yh, tmp1, tmp2, z, ylp, yhp, tmp1p, tmp2p;

    phase_Val = _mm256_loadu_ps((float*)phase_Ptr);
    inc_Val = _mm256_set_ps(lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr));
    const unsigned int fourthPoints = num_points / 4;


    for(i = 0; i < (unsigned int)(fourthPoints/ROTATOR_RELOAD); i++) {
        for(j = 0; j < ROTATOR_RELOAD; ++j) {

            aVal = _mm256_load_ps((float*)aPtr);

            yl = _mm256_moveldup_ps(phase_Val);
            yh = _mm256_movehdup_ps(phase_Val);
            ylp = _mm256_moveldup_ps(inc_Val);
            yhp = _mm256_movehdup_ps(inc_Val);

            tmp1 = _mm256_mul_ps(aVal, yl);
            tmp1p = _mm256_mul_ps(phase_Val, ylp);

            aVal = _mm256_shuffle_ps(aVal, aVal, 0xB1);
            phase_Val = _mm256_shuffle_ps(phase_Val, phase_Val, 0xB1);
            tmp2 = _mm256_mul_ps(aVal, yh);
            tmp2p = _mm256_mul_ps(phase_Val, yhp);

            z = _mm256_addsub_ps(tmp1, tmp2);
            phase_Val = _mm256_addsub_ps(tmp1p, tmp2p);

            _mm256_store_ps((float*)cPtr, z);

            aPtr += 4;
            cPtr += 4;
        }
        tmp1 = _mm256_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        tmp1 = _mm256_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm256_sqrt_ps(tmp1);
        phase_Val = _mm256_div_ps(phase_Val, tmp2);
    }
    for(i = 0; i < fourthPoints%ROTATOR_RELOAD; ++i) {
        aVal = _mm256_load_ps((float*)aPtr);

        yl = _mm256_moveldup_ps(phase_Val);
        yh = _mm256_movehdup_ps(phase_Val);
        ylp = _mm256_moveldup_ps(inc_Val);
        yhp = _mm256_movehdup_ps(inc_Val);

        tmp1 = _mm256_mul_ps(aVal, yl);

        tmp1p = _mm256_mul_ps(phase_Val, ylp);

        aVal = _mm256_shuffle_ps(aVal, aVal, 0xB1);
        phase_Val = _mm256_shuffle_ps(phase_Val, phase_Val, 0xB1);
        tmp2 = _mm256_mul_ps(aVal, yh);
        tmp2p = _mm256_mul_ps(phase_Val, yhp);

        z = _mm256_addsub_ps(tmp1, tmp2);
        phase_Val = _mm256_addsub_ps(tmp1p, tmp2p);

        _mm256_store_ps((float*)cPtr, z);

        aPtr += 4;
        cPtr += 4;
    }
    if (i) {
        tmp1 = _mm256_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        tmp1 = _mm256_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm256_sqrt_ps(tmp1);
        phase_Val = _mm256_div_ps(phase_Val, tmp2);
    }

    _mm256_storeu_ps((float*)phase_Ptr, phase_Val);
    for(i = 0; i < num_points%4; ++i) {
        *cPtr++ = *aPtr++ * phase_Ptr[0];
        phase_Ptr[0] *= (phase_inc);
    }

    (*phase) = phase_Ptr[0];

}

#endif /* LV_HAVE_AVX for aligned */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_s32fc_x2_rotator_32fc_u_avx(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points){
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;
    lv_32fc_t incr = 1;
    lv_32fc_t phase_Ptr[4] = {(*phase), (*phase), (*phase), (*phase)};

    unsigned int i, j = 0;

    for(i = 0; i < 4; ++i) {
        phase_Ptr[i] *= incr;
        incr *= (phase_inc);
    }

    /*printf("%f, %f\n", lv_creal(phase_Ptr[0]), lv_cimag(phase_Ptr[0]));
    printf("%f, %f\n", lv_creal(phase_Ptr[1]), lv_cimag(phase_Ptr[1]));
    printf("%f, %f\n", lv_creal(phase_Ptr[2]), lv_cimag(phase_Ptr[2]));
    printf("%f, %f\n", lv_creal(phase_Ptr[3]), lv_cimag(phase_Ptr[3]));
    printf("incr: %f, %f\n", lv_creal(incr), lv_cimag(incr));*/
    __m256 aVal, phase_Val, inc_Val, yl, yh, tmp1, tmp2, z, ylp, yhp, tmp1p, tmp2p;

    phase_Val = _mm256_loadu_ps((float*)phase_Ptr);
    inc_Val = _mm256_set_ps(lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr));
    const unsigned int fourthPoints = num_points / 4;


    for(i = 0; i < (unsigned int)(fourthPoints/ROTATOR_RELOAD); i++) {
        for(j = 0; j < ROTATOR_RELOAD; ++j) {

            aVal = _mm256_loadu_ps((float*)aPtr);

            yl = _mm256_moveldup_ps(phase_Val);
            yh = _mm256_movehdup_ps(phase_Val);
            ylp = _mm256_moveldup_ps(inc_Val);
            yhp = _mm256_movehdup_ps(inc_Val);

            tmp1 = _mm256_mul_ps(aVal, yl);
            tmp1p = _mm256_mul_ps(phase_Val, ylp);

            aVal = _mm256_shuffle_ps(aVal, aVal, 0xB1);
            phase_Val = _mm256_shuffle_ps(phase_Val, phase_Val, 0xB1);
            tmp2 = _mm256_mul_ps(aVal, yh);
            tmp2p = _mm256_mul_ps(phase_Val, yhp);

            z = _mm256_addsub_ps(tmp1, tmp2);
            phase_Val = _mm256_addsub_ps(tmp1p, tmp2p);

            _mm256_storeu_ps((float*)cPtr, z);

            aPtr += 4;
            cPtr += 4;
        }
        tmp1 = _mm256_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        tmp1 = _mm256_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm256_sqrt_ps(tmp1);
        phase_Val = _mm256_div_ps(phase_Val, tmp2);
    }
    for(i = 0; i < fourthPoints%ROTATOR_RELOAD; ++i) {
        aVal = _mm256_loadu_ps((float*)aPtr);

        yl = _mm256_moveldup_ps(phase_Val);
        yh = _mm256_movehdup_ps(phase_Val);
        ylp = _mm256_moveldup_ps(inc_Val);
        yhp = _mm256_movehdup_ps(inc_Val);

        tmp1 = _mm256_mul_ps(aVal, yl);

        tmp1p = _mm256_mul_ps(phase_Val, ylp);

        aVal = _mm256_shuffle_ps(aVal, aVal, 0xB1);
        phase_Val = _mm256_shuffle_ps(phase_Val, phase_Val, 0xB1);
        tmp2 = _mm256_mul_ps(aVal, yh);
        tmp2p = _mm256_mul_ps(phase_Val, yhp);

        z = _mm256_addsub_ps(tmp1, tmp2);
        phase_Val = _mm256_addsub_ps(tmp1p, tmp2p);

        _mm256_storeu_ps((float*)cPtr, z);

        aPtr += 4;
        cPtr += 4;
    }
    if (i) {
        tmp1 = _mm256_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        tmp1 = _mm256_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm256_sqrt_ps(tmp1);
        phase_Val = _mm256_div_ps(phase_Val, tmp2);
    }

    _mm256_storeu_ps((float*)phase_Ptr, phase_Val);
    for(i = 0; i < num_points%4; ++i) {
        *cPtr++ = *aPtr++ * phase_Ptr[0];
        phase_Ptr[0] *= (phase_inc);
    }

    (*phase) = phase_Ptr[0];

}

#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_s32fc_x2_rotator_32fc_a_avx_fma(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points){
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;
    lv_32fc_t incr = 1;
    __VOLK_ATTR_ALIGNED(32) lv_32fc_t phase_Ptr[4] = {(*phase), (*phase), (*phase), (*phase)};

    unsigned int i, j = 0;

    for(i = 0; i < 4; ++i) {
        phase_Ptr[i] *= incr;
        incr *= (phase_inc);
    }

    __m256 aVal, phase_Val, inc_Val, yl, yh, tmp1, tmp2, z, ylp, yhp, tmp1p, tmp2p;

    phase_Val = _mm256_load_ps((float*)phase_Ptr);
    inc_Val = _mm256_set_ps(lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr));
    const unsigned int fourthPoints = num_points / 4;

    for(i = 0; i < (unsigned int)(fourthPoints/ROTATOR_RELOAD); i++) {
        for(j = 0; j < ROTATOR_RELOAD; ++j) {

            aVal = _mm256_load_ps((float*)aPtr);

            yl = _mm256_moveldup_ps(phase_Val);
            yh = _mm256_movehdup_ps(phase_Val);
            ylp = _mm256_moveldup_ps(inc_Val);
            yhp = _mm256_movehdup_ps(inc_Val);

            tmp1 = aVal;
            tmp1p = phase_Val;

            aVal = _mm256_shuffle_ps(aVal, aVal, 0xB1);
            phase_Val = _mm256_shuffle_ps(phase_Val, phase_Val, 0xB1);
            tmp2 = _mm256_mul_ps(aVal, yh);
            tmp2p = _mm256_mul_ps(phase_Val, yhp);

            z = _mm256_fmaddsub_ps(tmp1, yl, tmp2);
            phase_Val = _mm256_fmaddsub_ps(tmp1p, ylp, tmp2p);

            _mm256_store_ps((float*)cPtr, z);

            aPtr += 4;
            cPtr += 4;
        }
        tmp1 = _mm256_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        tmp1 = _mm256_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm256_sqrt_ps(tmp1);
        phase_Val = _mm256_div_ps(phase_Val, tmp2);
    }
    for(i = 0; i < fourthPoints%ROTATOR_RELOAD; ++i) {
        aVal = _mm256_load_ps((float*)aPtr);

        yl = _mm256_moveldup_ps(phase_Val);
        yh = _mm256_movehdup_ps(phase_Val);
        ylp = _mm256_moveldup_ps(inc_Val);
        yhp = _mm256_movehdup_ps(inc_Val);

        tmp1 = aVal;
        tmp1p = ylp;

        aVal = _mm256_shuffle_ps(aVal, aVal, 0xB1);
        phase_Val = _mm256_shuffle_ps(phase_Val, phase_Val, 0xB1);
        tmp2 = _mm256_mul_ps(aVal, yh);
        tmp2p = _mm256_mul_ps(phase_Val, yhp);

        z = _mm256_fmaddsub_ps(tmp1, yl, tmp2);
        phase_Val = _mm256_fmaddsub_ps(tmp1p, ylp, tmp2p);

        _mm256_store_ps((float*)cPtr, z);

        aPtr += 4;
        cPtr += 4;
    }
    if (i) {
        tmp1 = _mm256_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        tmp1 = _mm256_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm256_sqrt_ps(tmp1);
        phase_Val = _mm256_div_ps(phase_Val, tmp2);
    }

    _mm256_store_ps((float*)phase_Ptr, phase_Val);
    for(i = 0; i < num_points%4; ++i) {
        *cPtr++ = *aPtr++ * phase_Ptr[0];
        phase_Ptr[0] *= (phase_inc);
    }

    (*phase) = phase_Ptr[0];

}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA for aligned*/

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_s32fc_x2_rotator_32fc_u_avx_fma(lv_32fc_t* outVector, const lv_32fc_t* inVector, const lv_32fc_t phase_inc, lv_32fc_t* phase, unsigned int num_points){
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;
    lv_32fc_t incr = 1;
    lv_32fc_t phase_Ptr[4] = {(*phase), (*phase), (*phase), (*phase)};

    unsigned int i, j = 0;

    for(i = 0; i < 4; ++i) {
        phase_Ptr[i] *= incr;
        incr *= (phase_inc);
    }

    __m256 aVal, phase_Val, inc_Val, yl, yh, tmp1, tmp2, z, ylp, yhp, tmp1p, tmp2p;

    phase_Val = _mm256_loadu_ps((float*)phase_Ptr);
    inc_Val = _mm256_set_ps(lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr),lv_cimag(incr), lv_creal(incr));
    const unsigned int fourthPoints = num_points / 4;

    for(i = 0; i < (unsigned int)(fourthPoints/ROTATOR_RELOAD); i++) {
        for(j = 0; j < ROTATOR_RELOAD; ++j) {

            aVal = _mm256_loadu_ps((float*)aPtr);

            yl = _mm256_moveldup_ps(phase_Val);
            yh = _mm256_movehdup_ps(phase_Val);
            ylp = _mm256_moveldup_ps(inc_Val);
            yhp = _mm256_movehdup_ps(inc_Val);

            tmp1 = aVal;
            tmp1p = phase_Val;

            aVal = _mm256_shuffle_ps(aVal, aVal, 0xB1);
            phase_Val = _mm256_shuffle_ps(phase_Val, phase_Val, 0xB1);
            tmp2 = _mm256_mul_ps(aVal, yh);
            tmp2p = _mm256_mul_ps(phase_Val, yhp);

            z = _mm256_fmaddsub_ps(tmp1, yl, tmp2);
            phase_Val = _mm256_fmaddsub_ps(tmp1p, ylp, tmp2p);

            _mm256_storeu_ps((float*)cPtr, z);

            aPtr += 4;
            cPtr += 4;
        }
        tmp1 = _mm256_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        tmp1 = _mm256_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm256_sqrt_ps(tmp1);
        phase_Val = _mm256_div_ps(phase_Val, tmp2);
    }
    for(i = 0; i < fourthPoints%ROTATOR_RELOAD; ++i) {
        aVal = _mm256_loadu_ps((float*)aPtr);

        yl = _mm256_moveldup_ps(phase_Val);
        yh = _mm256_movehdup_ps(phase_Val);
        ylp = _mm256_moveldup_ps(inc_Val);
        yhp = _mm256_movehdup_ps(inc_Val);

        tmp1 = aVal;
        tmp1p = ylp;

        aVal = _mm256_shuffle_ps(aVal, aVal, 0xB1);
        phase_Val = _mm256_shuffle_ps(phase_Val, phase_Val, 0xB1);
        tmp2 = _mm256_mul_ps(aVal, yh);
        tmp2p = _mm256_mul_ps(phase_Val, yhp);

        z = _mm256_fmaddsub_ps(tmp1, yl, tmp2);
        phase_Val = _mm256_fmaddsub_ps(tmp1p, ylp, tmp2p);

        _mm256_storeu_ps((float*)cPtr, z);

        aPtr += 4;
        cPtr += 4;
    }
    if (i) {
        tmp1 = _mm256_mul_ps(phase_Val, phase_Val);
        tmp2 = _mm256_hadd_ps(tmp1, tmp1);
        tmp1 = _mm256_shuffle_ps(tmp2, tmp2, 0xD8);
        tmp2 = _mm256_sqrt_ps(tmp1);
        phase_Val = _mm256_div_ps(phase_Val, tmp2);
    }

    _mm256_storeu_ps((float*)phase_Ptr, phase_Val);
    for(i = 0; i < num_points%4; ++i) {
        *cPtr++ = *aPtr++ * phase_Ptr[0];
        phase_Ptr[0] *= (phase_inc);
    }

    (*phase) = phase_Ptr[0];

}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA*/

#endif /* INCLUDED_volk_32fc_s32fc_rotator_32fc_a_H */
