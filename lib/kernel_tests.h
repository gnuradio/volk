#include "qa_utils.h"

#include <volk/volk.h>

#include <boost/assign/list_of.hpp>
#include <vector>

// macros for initializing volk_test_case_t. Maccros are needed to generate
// function names of the pattern kernel_name_*

// for puppets we need to get all the func_variants for the puppet and just
// keep track of the actual function name to write to results
#define VOLK_INIT_PUPP(func, puppet_master_func, test_params)\
    volk_test_case_t(func##_get_func_desc(), (void(*)())func##_manual, std::string(#func),\
    std::string(#puppet_master_func), test_params)

#define VOLK_INIT_TEST(func, test_params)\
    volk_test_case_t(func##_get_func_desc(), (void(*)())func##_manual, std::string(#func),\
    test_params)

std::vector<volk_test_case_t> init_test_list(volk_test_params_t test_params)
{

    // Some kernels need a lower tolerance
    volk_test_params_t test_params_inacc = volk_test_params_t(1e-2, test_params.scalar(),
            test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex());
    volk_test_params_t test_params_inacc_tenth = volk_test_params_t(1e-1, test_params.scalar(),
            test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex());
    volk_test_params_t test_params_int1 = volk_test_params_t(1, test_params.scalar(),
            test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex());

    std::vector<volk_test_case_t> test_cases = boost::assign::list_of
        (VOLK_INIT_PUPP(volk_64u_popcntpuppet_64u, volk_64u_popcnt,     test_params))

        (VOLK_INIT_PUPP(volk_16u_byteswappuppet_16u, volk_16u_byteswap, test_params))
        (VOLK_INIT_PUPP(volk_32u_byteswappuppet_32u, volk_32u_byteswap, test_params))
        (VOLK_INIT_PUPP(volk_32u_popcntpuppet_32u, volk_32u_popcnt_32u,  test_params))
        (VOLK_INIT_PUPP(volk_64u_byteswappuppet_64u, volk_64u_byteswap, test_params))
        (VOLK_INIT_PUPP(volk_32fc_s32fc_rotatorpuppet_32fc, volk_32fc_s32fc_x2_rotator_32fc, test_params))
        (VOLK_INIT_PUPP(volk_8u_conv_k7_r2puppet_8u, volk_8u_x4_conv_k7_r2_8u, volk_test_params_t(0, test_params.scalar(), test_params.vlen(), test_params.iter()/10, test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_PUPP(volk_32f_x2_fm_detectpuppet_32f, volk_32f_s32f_32f_fm_detect_32f, test_params))
        (VOLK_INIT_TEST(volk_16ic_s32f_deinterleave_real_32f,           test_params))
        (VOLK_INIT_TEST(volk_16ic_deinterleave_real_8i,                 test_params))
        (VOLK_INIT_TEST(volk_16ic_deinterleave_16i_x2,                  test_params))
        (VOLK_INIT_TEST(volk_16ic_s32f_deinterleave_32f_x2,             test_params))
        (VOLK_INIT_TEST(volk_16ic_deinterleave_real_16i,                test_params))
        (VOLK_INIT_TEST(volk_16ic_magnitude_16i,                        test_params_int1))
        (VOLK_INIT_TEST(volk_16ic_s32f_magnitude_32f,                   test_params))
        (VOLK_INIT_TEST(volk_16ic_convert_32fc,                         test_params))
        (VOLK_INIT_TEST(volk_16ic_x2_multiply_16ic,                     test_params))
        (VOLK_INIT_TEST(volk_16ic_x2_dot_prod_16ic,                     test_params))
        (VOLK_INIT_TEST(volk_16i_s32f_convert_32f,                      test_params))
        (VOLK_INIT_TEST(volk_16i_convert_8i,                            test_params))
        (VOLK_INIT_TEST(volk_16i_32fc_dot_prod_32fc,                    test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_accumulator_s32f,                      test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_x2_add_32f,                            test_params))
        (VOLK_INIT_TEST(volk_32f_index_max_16u,                         test_params))
        (VOLK_INIT_TEST(volk_32f_index_max_32u,                         test_params))
        (VOLK_INIT_TEST(volk_32fc_32f_multiply_32fc,                    test_params))
        (VOLK_INIT_TEST(volk_32f_log2_32f,           volk_test_params_t(3, test_params.scalar(), test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_TEST(volk_32f_expfast_32f,        volk_test_params_t(1e-1, test_params.scalar(), test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_TEST(volk_32f_x2_pow_32f,         volk_test_params_t(1e-2, test_params.scalar(), test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_TEST(volk_32f_sin_32f,                               test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_cos_32f,                               test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_tan_32f,                               test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_atan_32f,                              test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_asin_32f,                              test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_acos_32f,                              test_params_inacc))
        (VOLK_INIT_TEST(volk_32fc_s32f_power_32fc,                      test_params))
        (VOLK_INIT_TEST(volk_32f_s32f_calc_spectral_noise_floor_32f,    test_params_inacc))
        (VOLK_INIT_TEST(volk_32fc_s32f_atan2_32f,                       test_params))
        (VOLK_INIT_TEST(volk_32fc_x2_conjugate_dot_prod_32fc,           test_params_inacc))
        (VOLK_INIT_TEST(volk_32fc_deinterleave_32f_x2,                  test_params))
        (VOLK_INIT_TEST(volk_32fc_deinterleave_64f_x2,                  test_params))
        (VOLK_INIT_TEST(volk_32fc_s32f_deinterleave_real_16i,           test_params))
        (VOLK_INIT_TEST(volk_32fc_deinterleave_imag_32f,                test_params))
        (VOLK_INIT_TEST(volk_32fc_deinterleave_real_32f,                test_params))
        (VOLK_INIT_TEST(volk_32fc_deinterleave_real_64f,                test_params))
        (VOLK_INIT_TEST(volk_32fc_x2_dot_prod_32fc,                     test_params_inacc))
        (VOLK_INIT_TEST(volk_32fc_32f_dot_prod_32fc,                    test_params_inacc))
        (VOLK_INIT_TEST(volk_32fc_index_max_16u,      volk_test_params_t(3, test_params.scalar(), test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_TEST(volk_32fc_index_max_32u,      volk_test_params_t(3, test_params.scalar(), test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_TEST(volk_32fc_s32f_magnitude_16i,                   test_params_int1))
        (VOLK_INIT_TEST(volk_32fc_magnitude_32f,                        test_params_inacc_tenth))
        (VOLK_INIT_TEST(volk_32fc_magnitude_squared_32f,                test_params))
        (VOLK_INIT_TEST(volk_32fc_x2_multiply_32fc,                     test_params))
        (VOLK_INIT_TEST(volk_32fc_x2_multiply_conjugate_32fc,           test_params))
        (VOLK_INIT_TEST(volk_32fc_x2_divide_32fc,                       test_params))
        (VOLK_INIT_TEST(volk_32fc_conjugate_32fc,                       test_params))
        (VOLK_INIT_TEST(volk_32f_s32f_convert_16i,                      test_params))
        (VOLK_INIT_TEST(volk_32f_s32f_convert_32i,    volk_test_params_t(1, test_params.scalar(), test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_TEST(volk_32f_convert_64f,                           test_params))
        (VOLK_INIT_TEST(volk_32f_s32f_convert_8i,     volk_test_params_t(1, test_params.scalar(), test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_TEST(volk_32fc_convert_16ic,                         test_params))
        (VOLK_INIT_TEST(volk_32fc_s32f_power_spectrum_32f,              test_params))
        (VOLK_INIT_TEST(volk_32fc_x2_square_dist_32f,                   test_params))
        (VOLK_INIT_TEST(volk_32fc_x2_s32f_square_dist_scalar_mult_32f,  test_params))
        (VOLK_INIT_TEST(volk_32f_x2_divide_32f,                         test_params))
        (VOLK_INIT_TEST(volk_32f_x2_dot_prod_32f,                       test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_x2_s32f_interleave_16ic, volk_test_params_t(1, test_params.scalar(), test_params.vlen(), test_params.iter(), test_params.benchmark_mode(), test_params.kernel_regex())))
        (VOLK_INIT_TEST(volk_32f_x2_interleave_32fc,                    test_params))
        (VOLK_INIT_TEST(volk_32f_x2_max_32f,                            test_params))
        (VOLK_INIT_TEST(volk_32f_x2_min_32f,                            test_params))
        (VOLK_INIT_TEST(volk_32f_x2_multiply_32f,                       test_params))
        (VOLK_INIT_TEST(volk_32f_s32f_normalize,                        test_params))
        (VOLK_INIT_TEST(volk_32f_s32f_power_32f,                        test_params))
        (VOLK_INIT_TEST(volk_32f_sqrt_32f,                              test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_s32f_stddev_32f,                       test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_stddev_and_mean_32f_x2,                test_params_inacc))
        (VOLK_INIT_TEST(volk_32f_x2_subtract_32f,                       test_params))
        (VOLK_INIT_TEST(volk_32f_x3_sum_of_poly_32f,                    test_params_inacc))
        (VOLK_INIT_TEST(volk_32i_x2_and_32i,                            test_params))
        (VOLK_INIT_TEST(volk_32i_s32f_convert_32f,                      test_params))
        (VOLK_INIT_TEST(volk_32i_x2_or_32i,                             test_params))
        (VOLK_INIT_TEST(volk_32f_x2_dot_prod_16i,                       test_params))
        (VOLK_INIT_TEST(volk_64f_convert_32f,                           test_params))
        (VOLK_INIT_TEST(volk_64f_x2_max_64f,                            test_params))
        (VOLK_INIT_TEST(volk_64f_x2_min_64f,                            test_params))
        (VOLK_INIT_TEST(volk_8ic_deinterleave_16i_x2,                   test_params))
        (VOLK_INIT_TEST(volk_8ic_s32f_deinterleave_32f_x2,              test_params))
        (VOLK_INIT_TEST(volk_8ic_deinterleave_real_16i,                 test_params))
        (VOLK_INIT_TEST(volk_8ic_s32f_deinterleave_real_32f,            test_params))
        (VOLK_INIT_TEST(volk_8ic_deinterleave_real_8i,                  test_params))
        (VOLK_INIT_TEST(volk_8ic_x2_multiply_conjugate_16ic,            test_params))
        (VOLK_INIT_TEST(volk_8ic_x2_s32f_multiply_conjugate_32fc,       test_params))
        (VOLK_INIT_TEST(volk_8i_convert_16i,                            test_params))
        (VOLK_INIT_TEST(volk_8i_s32f_convert_32f,                       test_params))
        (VOLK_INIT_TEST(volk_32fc_s32fc_multiply_32fc,                  test_params))
        (VOLK_INIT_TEST(volk_32f_s32f_multiply_32f,                     test_params))
        (VOLK_INIT_TEST(volk_32f_binary_slicer_32i,                     test_params))
        (VOLK_INIT_TEST(volk_32f_binary_slicer_8i,                      test_params))
        (VOLK_INIT_TEST(volk_32f_tanh_32f,                              test_params_inacc))
        (VOLK_INIT_PUPP(volk_8u_x3_encodepolarpuppet_8u, volk_8u_x3_encodepolar_8u_x2, test_params))
        (VOLK_INIT_PUPP(volk_32f_8u_polarbutterflypuppet_32f, volk_32f_8u_polarbutterfly_32f, test_params))
        // no one uses these, so don't test them
        //VOLK_PROFILE(volk_16i_x5_add_quad_16i_x4, 1e-4, 2046, 10000, &results, benchmark_mode, kernel_regex);
        //VOLK_PROFILE(volk_16i_branch_4_state_8, 1e-4, 2046, 10000, &results, benchmark_mode, kernel_regex);
        //VOLK_PROFILE(volk_16i_max_star_16i, 0, 0, 204602, 10000, &results, benchmark_mode, kernel_regex);
        //VOLK_PROFILE(volk_16i_max_star_horizontal_16i, 0, 0, 204602, 10000, &results, benchmark_mode, kernel_regex);
        //VOLK_PROFILE(volk_16i_permute_and_scalar_add, 1e-4, 0, 2046, 10000, &results, benchmark_mode, kernel_regex);
        //VOLK_PROFILE(volk_16i_x4_quad_max_star_16i, 1e-4, 0, 2046, 10000, &results, benchmark_mode, kernel_regex);
        // we need a puppet for this one
        //(VOLK_INIT_TEST(volk_32fc_s32f_x2_power_spectral_density_32f,   test_params))

        ;


    return test_cases;
}
