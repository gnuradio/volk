/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_calc_spectral_noise_floor_32f
 *
 * \b Overview
 *
 * Computes the spectral noise floor of an input power spectrum.
 *
 * Calculates the spectral noise floor of an input power spectrum by
 * determining the mean of the input power spectrum, then
 * recalculating the mean excluding any power spectrum values that
 * exceed the mean by the spectralExclusionValue (in dB).  Provides a
 * rough estimation of the signal noise floor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_calc_spectral_noise_floor_32f(float* noiseFloorAmplitude, const
 * float* realDataPoints, const float spectralExclusionValue, const unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li realDataPoints: The input power spectrum.
 * \li spectralExclusionValue: The number of dB above the noise floor that a data point
 * must be to be excluded from the noise floor calculation - default value is 20. \li
 * num_points: The number of data points.
 *
 * \b Outputs
 * \li noiseFloorAmplitude: The noise floor of the input spectrum, in dB.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_32f_s32f_calc_spectral_noise_floor_32f
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_calc_spectral_noise_floor_32f_a_H
#define INCLUDED_volk_32f_s32f_calc_spectral_noise_floor_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#include <volk/volk_32f_accumulator_s32f.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_s32f_calc_spectral_noise_floor_32f_a_avx(float* noiseFloorAmplitude,
                                                  const float* realDataPoints,
                                                  const float spectralExclusionValue,
                                                  const unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* dataPointsPtr = realDataPoints;
    __VOLK_ATTR_ALIGNED(32) float avgPointsVector[8];

    __m256 dataPointsVal;
    __m256 avgPointsVal = _mm256_setzero_ps();
    // Calculate the sum (for mean) for all points
    for (; number < eighthPoints; number++) {

        dataPointsVal = _mm256_load_ps(dataPointsPtr);

        dataPointsPtr += 8;

        avgPointsVal = _mm256_add_ps(avgPointsVal, dataPointsVal);
    }

    _mm256_store_ps(avgPointsVector, avgPointsVal);

    float sumMean = 0.0;
    sumMean += avgPointsVector[0];
    sumMean += avgPointsVector[1];
    sumMean += avgPointsVector[2];
    sumMean += avgPointsVector[3];
    sumMean += avgPointsVector[4];
    sumMean += avgPointsVector[5];
    sumMean += avgPointsVector[6];
    sumMean += avgPointsVector[7];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        sumMean += realDataPoints[number];
    }

    // calculate the spectral mean
    // +20 because for the comparison below we only want to throw out bins
    // that are significantly higher (and would, thus, affect the mean more
    const float meanAmplitude = (sumMean / ((float)num_points)) + spectralExclusionValue;

    dataPointsPtr = realDataPoints; // Reset the dataPointsPtr
    __m256 vMeanAmplitudeVector = _mm256_set1_ps(meanAmplitude);
    __m256 vOnesVector = _mm256_set1_ps(1.0);
    __m256 vValidBinCount = _mm256_setzero_ps();
    avgPointsVal = _mm256_setzero_ps();
    __m256 compareMask;
    number = 0;
    // Calculate the sum (for mean) for any points which do NOT exceed the mean amplitude
    for (; number < eighthPoints; number++) {

        dataPointsVal = _mm256_load_ps(dataPointsPtr);

        dataPointsPtr += 8;

        // Identify which items do not exceed the mean amplitude
        compareMask = _mm256_cmp_ps(dataPointsVal, vMeanAmplitudeVector, _CMP_LE_OQ);

        // Mask off the items that exceed the mean amplitude and add the avg Points that
        // do not exceed the mean amplitude
        avgPointsVal =
            _mm256_add_ps(avgPointsVal, _mm256_and_ps(compareMask, dataPointsVal));

        // Count the number of bins which do not exceed the mean amplitude
        vValidBinCount =
            _mm256_add_ps(vValidBinCount, _mm256_and_ps(compareMask, vOnesVector));
    }

    // Calculate the mean from the remaining data points
    _mm256_store_ps(avgPointsVector, avgPointsVal);

    sumMean = 0.0;
    sumMean += avgPointsVector[0];
    sumMean += avgPointsVector[1];
    sumMean += avgPointsVector[2];
    sumMean += avgPointsVector[3];
    sumMean += avgPointsVector[4];
    sumMean += avgPointsVector[5];
    sumMean += avgPointsVector[6];
    sumMean += avgPointsVector[7];

    // Calculate the number of valid bins from the remaining count
    __VOLK_ATTR_ALIGNED(32) float validBinCountVector[8];
    _mm256_store_ps(validBinCountVector, vValidBinCount);

    float validBinCount = 0;
    validBinCount += validBinCountVector[0];
    validBinCount += validBinCountVector[1];
    validBinCount += validBinCountVector[2];
    validBinCount += validBinCountVector[3];
    validBinCount += validBinCountVector[4];
    validBinCount += validBinCountVector[5];
    validBinCount += validBinCountVector[6];
    validBinCount += validBinCountVector[7];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        if (realDataPoints[number] <= meanAmplitude) {
            sumMean += realDataPoints[number];
            validBinCount += 1.0;
        }
    }

    float localNoiseFloorAmplitude = 0;
    if (validBinCount > 0.0) {
        localNoiseFloorAmplitude = sumMean / validBinCount;
    } else {
        localNoiseFloorAmplitude =
            meanAmplitude; // For the odd case that all the amplitudes are equal...
    }

    *noiseFloorAmplitude = localNoiseFloorAmplitude;
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_s32f_calc_spectral_noise_floor_32f_a_sse(float* noiseFloorAmplitude,
                                                  const float* realDataPoints,
                                                  const float spectralExclusionValue,
                                                  const unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* dataPointsPtr = realDataPoints;
    __VOLK_ATTR_ALIGNED(16) float avgPointsVector[4];

    __m128 dataPointsVal;
    __m128 avgPointsVal = _mm_setzero_ps();
    // Calculate the sum (for mean) for all points
    for (; number < quarterPoints; number++) {

        dataPointsVal = _mm_load_ps(dataPointsPtr);

        dataPointsPtr += 4;

        avgPointsVal = _mm_add_ps(avgPointsVal, dataPointsVal);
    }

    _mm_store_ps(avgPointsVector, avgPointsVal);

    float sumMean = 0.0;
    sumMean += avgPointsVector[0];
    sumMean += avgPointsVector[1];
    sumMean += avgPointsVector[2];
    sumMean += avgPointsVector[3];

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        sumMean += realDataPoints[number];
    }

    // calculate the spectral mean
    // +20 because for the comparison below we only want to throw out bins
    // that are significantly higher (and would, thus, affect the mean more
    const float meanAmplitude = (sumMean / ((float)num_points)) + spectralExclusionValue;

    dataPointsPtr = realDataPoints; // Reset the dataPointsPtr
    __m128 vMeanAmplitudeVector = _mm_set_ps1(meanAmplitude);
    __m128 vOnesVector = _mm_set_ps1(1.0);
    __m128 vValidBinCount = _mm_setzero_ps();
    avgPointsVal = _mm_setzero_ps();
    __m128 compareMask;
    number = 0;
    // Calculate the sum (for mean) for any points which do NOT exceed the mean amplitude
    for (; number < quarterPoints; number++) {

        dataPointsVal = _mm_load_ps(dataPointsPtr);

        dataPointsPtr += 4;

        // Identify which items do not exceed the mean amplitude
        compareMask = _mm_cmple_ps(dataPointsVal, vMeanAmplitudeVector);

        // Mask off the items that exceed the mean amplitude and add the avg Points that
        // do not exceed the mean amplitude
        avgPointsVal = _mm_add_ps(avgPointsVal, _mm_and_ps(compareMask, dataPointsVal));

        // Count the number of bins which do not exceed the mean amplitude
        vValidBinCount = _mm_add_ps(vValidBinCount, _mm_and_ps(compareMask, vOnesVector));
    }

    // Calculate the mean from the remaining data points
    _mm_store_ps(avgPointsVector, avgPointsVal);

    sumMean = 0.0;
    sumMean += avgPointsVector[0];
    sumMean += avgPointsVector[1];
    sumMean += avgPointsVector[2];
    sumMean += avgPointsVector[3];

    // Calculate the number of valid bins from the remaining count
    __VOLK_ATTR_ALIGNED(16) float validBinCountVector[4];
    _mm_store_ps(validBinCountVector, vValidBinCount);

    float validBinCount = 0;
    validBinCount += validBinCountVector[0];
    validBinCount += validBinCountVector[1];
    validBinCount += validBinCountVector[2];
    validBinCount += validBinCountVector[3];

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        if (realDataPoints[number] <= meanAmplitude) {
            sumMean += realDataPoints[number];
            validBinCount += 1.0;
        }
    }

    float localNoiseFloorAmplitude = 0;
    if (validBinCount > 0.0) {
        localNoiseFloorAmplitude = sumMean / validBinCount;
    } else {
        localNoiseFloorAmplitude =
            meanAmplitude; // For the odd case that all the amplitudes are equal...
    }

    *noiseFloorAmplitude = localNoiseFloorAmplitude;
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_s32f_calc_spectral_noise_floor_32f_generic(float* noiseFloorAmplitude,
                                                    const float* realDataPoints,
                                                    const float spectralExclusionValue,
                                                    const unsigned int num_points)
{
    float sumMean = 0.0;
    unsigned int number;
    // find the sum (for mean), etc
    for (number = 0; number < num_points; number++) {
        // sum (for mean)
        sumMean += realDataPoints[number];
    }

    // calculate the spectral mean
    // +20 because for the comparison below we only want to throw out bins
    // that are significantly higher (and would, thus, affect the mean more)
    const float meanAmplitude = (sumMean / num_points) + spectralExclusionValue;

    // now throw out any bins higher than the mean
    sumMean = 0.0;
    unsigned int newNumDataPoints = num_points;
    for (number = 0; number < num_points; number++) {
        if (realDataPoints[number] <= meanAmplitude)
            sumMean += realDataPoints[number];
        else
            newNumDataPoints--;
    }

    float localNoiseFloorAmplitude = 0.0;
    if (newNumDataPoints == 0)                    // in the odd case that all
        localNoiseFloorAmplitude = meanAmplitude; // amplitudes are equal!
    else
        localNoiseFloorAmplitude = sumMean / ((float)newNumDataPoints);

    *noiseFloorAmplitude = localNoiseFloorAmplitude;
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_s32f_calc_spectral_noise_floor_32f_neon(float* noiseFloorAmplitude,
                                                 const float* realDataPoints,
                                                 const float spectralExclusionValue,
                                                 const unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* dataPointsPtr = realDataPoints;
    float32x4_t avgPointsVal = vdupq_n_f32(0.0f);

    // Calculate the sum (for mean) for all points
    for (; number < quarterPoints; number++) {
        float32x4_t dataPointsVal = vld1q_f32(dataPointsPtr);
        dataPointsPtr += 4;
        avgPointsVal = vaddq_f32(avgPointsVal, dataPointsVal);
    }

    // Horizontal sum
    float32x2_t sum2 = vadd_f32(vget_low_f32(avgPointsVal), vget_high_f32(avgPointsVal));
    float sumMean = vget_lane_f32(vpadd_f32(sum2, sum2), 0);

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        sumMean += realDataPoints[number];
    }

    // calculate the spectral mean
    const float meanAmplitude = (sumMean / ((float)num_points)) + spectralExclusionValue;

    dataPointsPtr = realDataPoints;
    float32x4_t vMeanAmplitudeVector = vdupq_n_f32(meanAmplitude);
    float32x4_t vOnesVector = vdupq_n_f32(1.0f);
    float32x4_t vValidBinCount = vdupq_n_f32(0.0f);
    avgPointsVal = vdupq_n_f32(0.0f);
    number = 0;

    // Calculate the sum (for mean) for any points which do NOT exceed the mean amplitude
    for (; number < quarterPoints; number++) {
        float32x4_t dataPointsVal = vld1q_f32(dataPointsPtr);
        dataPointsPtr += 4;

        // Identify which items do not exceed the mean amplitude
        uint32x4_t compareMask = vcleq_f32(dataPointsVal, vMeanAmplitudeVector);

        // Mask off the items that exceed the mean amplitude
        float32x4_t maskedData = vbslq_f32(compareMask, dataPointsVal, vdupq_n_f32(0.0f));
        avgPointsVal = vaddq_f32(avgPointsVal, maskedData);

        // Count the number of bins which do not exceed the mean amplitude
        float32x4_t maskedOnes = vbslq_f32(compareMask, vOnesVector, vdupq_n_f32(0.0f));
        vValidBinCount = vaddq_f32(vValidBinCount, maskedOnes);
    }

    // Horizontal sums
    sum2 = vadd_f32(vget_low_f32(avgPointsVal), vget_high_f32(avgPointsVal));
    sumMean = vget_lane_f32(vpadd_f32(sum2, sum2), 0);

    float32x2_t cnt2 =
        vadd_f32(vget_low_f32(vValidBinCount), vget_high_f32(vValidBinCount));
    float validBinCount = vget_lane_f32(vpadd_f32(cnt2, cnt2), 0);

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        if (realDataPoints[number] <= meanAmplitude) {
            sumMean += realDataPoints[number];
            validBinCount += 1.0f;
        }
    }

    float localNoiseFloorAmplitude = 0;
    if (validBinCount > 0.0f) {
        localNoiseFloorAmplitude = sumMean / validBinCount;
    } else {
        localNoiseFloorAmplitude = meanAmplitude;
    }

    *noiseFloorAmplitude = localNoiseFloorAmplitude;
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_s32f_calc_spectral_noise_floor_32f_neonv8(float* noiseFloorAmplitude,
                                                   const float* realDataPoints,
                                                   const float spectralExclusionValue,
                                                   const unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* dataPointsPtr = realDataPoints;
    float32x4_t avgPointsVal0 = vdupq_n_f32(0.0f);
    float32x4_t avgPointsVal1 = vdupq_n_f32(0.0f);

    // Calculate the sum (for mean) for all points
    for (; number < eighthPoints; number++) {
        __VOLK_PREFETCH(dataPointsPtr + 16);
        float32x4_t dataPointsVal0 = vld1q_f32(dataPointsPtr);
        float32x4_t dataPointsVal1 = vld1q_f32(dataPointsPtr + 4);
        dataPointsPtr += 8;
        avgPointsVal0 = vaddq_f32(avgPointsVal0, dataPointsVal0);
        avgPointsVal1 = vaddq_f32(avgPointsVal1, dataPointsVal1);
    }

    // Combine and horizontal sum using vaddvq_f32
    float32x4_t avgPointsVal = vaddq_f32(avgPointsVal0, avgPointsVal1);
    float sumMean = vaddvq_f32(avgPointsVal);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        sumMean += realDataPoints[number];
    }

    // calculate the spectral mean
    const float meanAmplitude = (sumMean / ((float)num_points)) + spectralExclusionValue;

    dataPointsPtr = realDataPoints;
    float32x4_t vMeanAmplitudeVector = vdupq_n_f32(meanAmplitude);
    float32x4_t vOnesVector = vdupq_n_f32(1.0f);
    float32x4_t vValidBinCount0 = vdupq_n_f32(0.0f);
    float32x4_t vValidBinCount1 = vdupq_n_f32(0.0f);
    avgPointsVal0 = vdupq_n_f32(0.0f);
    avgPointsVal1 = vdupq_n_f32(0.0f);
    number = 0;

    // Calculate the sum (for mean) for any points which do NOT exceed the mean amplitude
    for (; number < eighthPoints; number++) {
        __VOLK_PREFETCH(dataPointsPtr + 16);
        float32x4_t dataPointsVal0 = vld1q_f32(dataPointsPtr);
        float32x4_t dataPointsVal1 = vld1q_f32(dataPointsPtr + 4);
        dataPointsPtr += 8;

        // Identify which items do not exceed the mean amplitude
        uint32x4_t compareMask0 = vcleq_f32(dataPointsVal0, vMeanAmplitudeVector);
        uint32x4_t compareMask1 = vcleq_f32(dataPointsVal1, vMeanAmplitudeVector);

        // Mask off the items that exceed the mean amplitude
        float32x4_t maskedData0 =
            vbslq_f32(compareMask0, dataPointsVal0, vdupq_n_f32(0.0f));
        float32x4_t maskedData1 =
            vbslq_f32(compareMask1, dataPointsVal1, vdupq_n_f32(0.0f));
        avgPointsVal0 = vaddq_f32(avgPointsVal0, maskedData0);
        avgPointsVal1 = vaddq_f32(avgPointsVal1, maskedData1);

        // Count the number of bins which do not exceed the mean amplitude
        float32x4_t maskedOnes0 = vbslq_f32(compareMask0, vOnesVector, vdupq_n_f32(0.0f));
        float32x4_t maskedOnes1 = vbslq_f32(compareMask1, vOnesVector, vdupq_n_f32(0.0f));
        vValidBinCount0 = vaddq_f32(vValidBinCount0, maskedOnes0);
        vValidBinCount1 = vaddq_f32(vValidBinCount1, maskedOnes1);
    }

    // Combine and horizontal sums
    avgPointsVal = vaddq_f32(avgPointsVal0, avgPointsVal1);
    sumMean = vaddvq_f32(avgPointsVal);

    float32x4_t vValidBinCount = vaddq_f32(vValidBinCount0, vValidBinCount1);
    float validBinCount = vaddvq_f32(vValidBinCount);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        if (realDataPoints[number] <= meanAmplitude) {
            sumMean += realDataPoints[number];
            validBinCount += 1.0f;
        }
    }

    float localNoiseFloorAmplitude = 0;
    if (validBinCount > 0.0f) {
        localNoiseFloorAmplitude = sumMean / validBinCount;
    } else {
        localNoiseFloorAmplitude = meanAmplitude;
    }

    *noiseFloorAmplitude = localNoiseFloorAmplitude;
}
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32f_s32f_calc_spectral_noise_floor_32f_a_H */

#ifndef INCLUDED_volk_32f_s32f_calc_spectral_noise_floor_32f_u_H
#define INCLUDED_volk_32f_s32f_calc_spectral_noise_floor_32f_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_s32f_calc_spectral_noise_floor_32f_u_avx(float* noiseFloorAmplitude,
                                                  const float* realDataPoints,
                                                  const float spectralExclusionValue,
                                                  const unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* dataPointsPtr = realDataPoints;
    __VOLK_ATTR_ALIGNED(16) float avgPointsVector[8];

    __m256 dataPointsVal;
    __m256 avgPointsVal = _mm256_setzero_ps();
    // Calculate the sum (for mean) for all points
    for (; number < eighthPoints; number++) {

        dataPointsVal = _mm256_loadu_ps(dataPointsPtr);

        dataPointsPtr += 8;

        avgPointsVal = _mm256_add_ps(avgPointsVal, dataPointsVal);
    }

    _mm256_storeu_ps(avgPointsVector, avgPointsVal);

    float sumMean = 0.0;
    sumMean += avgPointsVector[0];
    sumMean += avgPointsVector[1];
    sumMean += avgPointsVector[2];
    sumMean += avgPointsVector[3];
    sumMean += avgPointsVector[4];
    sumMean += avgPointsVector[5];
    sumMean += avgPointsVector[6];
    sumMean += avgPointsVector[7];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        sumMean += realDataPoints[number];
    }

    // calculate the spectral mean
    // +20 because for the comparison below we only want to throw out bins
    // that are significantly higher (and would, thus, affect the mean more
    const float meanAmplitude = (sumMean / ((float)num_points)) + spectralExclusionValue;

    dataPointsPtr = realDataPoints; // Reset the dataPointsPtr
    __m256 vMeanAmplitudeVector = _mm256_set1_ps(meanAmplitude);
    __m256 vOnesVector = _mm256_set1_ps(1.0);
    __m256 vValidBinCount = _mm256_setzero_ps();
    avgPointsVal = _mm256_setzero_ps();
    __m256 compareMask;
    number = 0;
    // Calculate the sum (for mean) for any points which do NOT exceed the mean amplitude
    for (; number < eighthPoints; number++) {

        dataPointsVal = _mm256_loadu_ps(dataPointsPtr);

        dataPointsPtr += 8;

        // Identify which items do not exceed the mean amplitude
        compareMask = _mm256_cmp_ps(dataPointsVal, vMeanAmplitudeVector, _CMP_LE_OQ);

        // Mask off the items that exceed the mean amplitude and add the avg Points that
        // do not exceed the mean amplitude
        avgPointsVal =
            _mm256_add_ps(avgPointsVal, _mm256_and_ps(compareMask, dataPointsVal));

        // Count the number of bins which do not exceed the mean amplitude
        vValidBinCount =
            _mm256_add_ps(vValidBinCount, _mm256_and_ps(compareMask, vOnesVector));
    }

    // Calculate the mean from the remaining data points
    _mm256_storeu_ps(avgPointsVector, avgPointsVal);

    sumMean = 0.0;
    sumMean += avgPointsVector[0];
    sumMean += avgPointsVector[1];
    sumMean += avgPointsVector[2];
    sumMean += avgPointsVector[3];
    sumMean += avgPointsVector[4];
    sumMean += avgPointsVector[5];
    sumMean += avgPointsVector[6];
    sumMean += avgPointsVector[7];

    // Calculate the number of valid bins from the remaining count
    __VOLK_ATTR_ALIGNED(16) float validBinCountVector[8];
    _mm256_storeu_ps(validBinCountVector, vValidBinCount);

    float validBinCount = 0;
    validBinCount += validBinCountVector[0];
    validBinCount += validBinCountVector[1];
    validBinCount += validBinCountVector[2];
    validBinCount += validBinCountVector[3];
    validBinCount += validBinCountVector[4];
    validBinCount += validBinCountVector[5];
    validBinCount += validBinCountVector[6];
    validBinCount += validBinCountVector[7];

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        if (realDataPoints[number] <= meanAmplitude) {
            sumMean += realDataPoints[number];
            validBinCount += 1.0;
        }
    }

    float localNoiseFloorAmplitude = 0;
    if (validBinCount > 0.0) {
        localNoiseFloorAmplitude = sumMean / validBinCount;
    } else {
        localNoiseFloorAmplitude =
            meanAmplitude; // For the odd case that all the amplitudes are equal...
    }

    *noiseFloorAmplitude = localNoiseFloorAmplitude;
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_s32f_calc_spectral_noise_floor_32f_rvv(float* noiseFloorAmplitude,
                                                const float* realDataPoints,
                                                const float spectralExclusionValue,
                                                const unsigned int num_points)
{
    float sum;
    volk_32f_accumulator_s32f_rvv(&sum, realDataPoints, num_points);
    float meanAmplitude = sum / num_points + spectralExclusionValue;

    vfloat32m8_t vbin = __riscv_vfmv_v_f_f32m8(meanAmplitude, __riscv_vsetvlmax_e32m8());
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
    size_t n = num_points, binCount = 0;
    for (size_t vl; n > 0; n -= vl, realDataPoints += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(realDataPoints, vl);
        vbool4_t m = __riscv_vmfle(v, vbin, vl);
        binCount += __riscv_vcpop(m, vl);
        vsum = __riscv_vfadd_tumu(m, vsum, vsum, v, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t v = RISCV_SHRINK8(vfadd, f, 32, vsum);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    sum = __riscv_vfmv_f(__riscv_vfredusum(v, z, vl));

    *noiseFloorAmplitude = binCount == 0 ? meanAmplitude : sum / binCount;
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_calc_spectral_noise_floor_32f_u_H */
