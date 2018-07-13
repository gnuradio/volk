#ifndef INCLUDED_VOLK_32F_S32F_MOD_RANGEPUPPET_32F_H
#define INCLUDED_VOLK_32F_S32F_MOD_RANGEPUPPET_32F_H

#include <volk/volk_32f_s32f_s32f_mod_range_32f.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_32f_s32f_mod_rangepuppet_32f_generic(float *output, const float *input, float bound, unsigned int num_points){
  volk_32f_s32f_s32f_mod_range_32f_generic(output, input, bound-3.141f, bound, num_points);
}
#endif


#ifdef LV_HAVE_SSE
static inline void volk_32f_s32f_mod_rangepuppet_32f_u_sse(float *output, const float *input, float bound, unsigned int num_points){
  volk_32f_s32f_s32f_mod_range_32f_u_sse(output, input, bound-3.141f, bound, num_points);
}
#endif
#ifdef LV_HAVE_SSE
static inline void volk_32f_s32f_mod_rangepuppet_32f_a_sse(float *output, const float *input, float bound, unsigned int num_points){
  volk_32f_s32f_s32f_mod_range_32f_a_sse(output, input, bound-3.141f, bound, num_points);
}
#endif

#ifdef LV_HAVE_SSE2
static inline void volk_32f_s32f_mod_rangepuppet_32f_u_sse2(float *output, const float *input, float bound, unsigned int num_points){
  volk_32f_s32f_s32f_mod_range_32f_u_sse2(output, input, bound-3.141f, bound, num_points);
}
#endif
#ifdef LV_HAVE_SSE2
static inline void volk_32f_s32f_mod_rangepuppet_32f_a_sse2(float *output, const float *input, float bound, unsigned int num_points){
  volk_32f_s32f_s32f_mod_range_32f_a_sse2(output, input, bound-3.141f, bound, num_points);
}
#endif

#ifdef LV_HAVE_AVX
static inline void volk_32f_s32f_mod_rangepuppet_32f_u_avx(float *output, const float *input, float bound, unsigned int num_points){
  volk_32f_s32f_s32f_mod_range_32f_u_avx(output, input, bound-3.141f, bound, num_points);
}
#endif
#ifdef LV_HAVE_AVX
static inline void volk_32f_s32f_mod_rangepuppet_32f_a_avx(float *output, const float *input, float bound, unsigned int num_points){
  volk_32f_s32f_s32f_mod_range_32f_a_avx(output, input, bound-3.141f, bound, num_points);
}
#endif
#endif
