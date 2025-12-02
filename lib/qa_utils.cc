/* -*- c++ -*- */
/*
 * Copyright 2011 - 2020, 2022 Free Software Foundation, Inc.
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "qa_utils.h"
#include <volk/volk.h>

#include <volk/volk.h>        // for volk_func_desc_t
#include <volk/volk_malloc.h> // for volk_free, volk_m...

#include <assert.h>    // for assert
#include <stdint.h>    // for uint16_t, uint64_t
#include <sys/time.h>  // for CLOCKS_PER_SEC
#include <sys/types.h> // for int16_t, int32_t
#include <chrono>
#include <cmath>    // for sqrt, fabs, abs
#include <cstring>  // for memcpy, memset
#include <ctime>    // for clock
#include <iomanip>  // for setw, left
#include <iostream> // for cout, cerr
#include <limits>   // for numeric_limits
#include <map>      // for map, map<>::mappe...
#include <random>
#include <sstream> // for ostringstream
#include <vector>  // for vector, _Bit_refe...

// Warmup time for CPU frequency scaling (ms)
static double g_warmup_ms = 2000.0;
static bool g_warmup_done = false;

double volk_test_get_warmup_ms() { return g_warmup_ms; }
void volk_test_set_warmup_ms(double ms) { g_warmup_ms = ms; }
void volk_test_reset_warmup() { g_warmup_done = false; }

template <typename T>
void random_floats(void* buf, unsigned int n, std::default_random_engine& rnd_engine)
{
    T* array = static_cast<T*>(buf);
    std::uniform_real_distribution<T> uniform_dist(T(-1), T(1));
    for (unsigned int i = 0; i < n; i++) {
        array[i] = uniform_dist(rnd_engine);
    }
}

void load_random_data(void* data,
                      volk_type_t type,
                      unsigned int n,
                      const std::vector<float>& float_edge_cases,
                      const std::vector<lv_32fc_t>& complex_edge_cases)
{
    std::random_device rnd_device;
    std::default_random_engine rnd_engine(rnd_device());

    unsigned int edge_case_count = 0;

    // Inject complex edge cases for complex float types
    if (type.is_float && type.is_complex && !complex_edge_cases.empty()) {
        edge_case_count = std::min((unsigned int)complex_edge_cases.size(), n);
        if (type.size == 8) {
            lv_64fc_t* array = static_cast<lv_64fc_t*>(data);
            for (unsigned int i = 0; i < edge_case_count; i++) {
                array[i] = lv_cmake((double)lv_creal(complex_edge_cases[i]),
                                    (double)lv_cimag(complex_edge_cases[i]));
            }
        } else {
            lv_32fc_t* array = static_cast<lv_32fc_t*>(data);
            for (unsigned int i = 0; i < edge_case_count; i++) {
                array[i] = complex_edge_cases[i];
            }
        }
    }
    // Inject float edge cases for non-complex float types
    else if (type.is_float && !type.is_complex && !float_edge_cases.empty()) {
        edge_case_count = std::min((unsigned int)float_edge_cases.size(), n);
        if (type.size == 8) {
            double* array = static_cast<double*>(data);
            for (unsigned int i = 0; i < edge_case_count; i++) {
                array[i] = static_cast<double>(float_edge_cases[i]);
            }
        } else {
            float* array = static_cast<float*>(data);
            for (unsigned int i = 0; i < edge_case_count; i++) {
                array[i] = float_edge_cases[i];
            }
        }
    }

    unsigned int remaining_n = n - edge_case_count;
    if (type.is_complex)
        remaining_n *= 2;

    if (type.is_float) {
        if (type.size == 8) {
            double* array = static_cast<double*>(data);
            random_floats<double>(array + edge_case_count * (type.is_complex ? 2 : 1),
                                  remaining_n,
                                  rnd_engine);
        } else {
            float* array = static_cast<float*>(data);
            random_floats<float>(array + edge_case_count * (type.is_complex ? 2 : 1),
                                 remaining_n,
                                 rnd_engine);
        }
    } else {
        if (type.is_complex)
            n *= 2;
        switch (type.size) {
        case 8:
            if (type.is_signed) {
                std::uniform_int_distribution<int64_t> uniform_dist(
                    std::numeric_limits<int64_t>::min(),
                    std::numeric_limits<int64_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((int64_t*)data)[i] = uniform_dist(rnd_engine);
            } else {
                std::uniform_int_distribution<uint64_t> uniform_dist(
                    std::numeric_limits<uint64_t>::min(),
                    std::numeric_limits<uint64_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((uint64_t*)data)[i] = uniform_dist(rnd_engine);
            }
            break;
        case 4:
            if (type.is_signed) {
                std::uniform_int_distribution<int32_t> uniform_dist(
                    std::numeric_limits<int32_t>::min(),
                    std::numeric_limits<int32_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((int32_t*)data)[i] = uniform_dist(rnd_engine);
            } else {
                std::uniform_int_distribution<uint32_t> uniform_dist(
                    std::numeric_limits<uint32_t>::min(),
                    std::numeric_limits<uint32_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((uint32_t*)data)[i] = uniform_dist(rnd_engine);
            }
            break;
        case 2:
            if (type.is_signed) {
                std::uniform_int_distribution<int16_t> uniform_dist(-6, 6);
                for (unsigned int i = 0; i < n; i++)
                    ((int16_t*)data)[i] = uniform_dist(rnd_engine);
            } else {
                std::uniform_int_distribution<uint16_t> uniform_dist(
                    std::numeric_limits<uint16_t>::min(),
                    std::numeric_limits<uint16_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((uint16_t*)data)[i] = uniform_dist(rnd_engine);
            }
            break;
        case 1:
            if (type.is_signed) {
                std::uniform_int_distribution<int16_t> uniform_dist(
                    std::numeric_limits<int8_t>::min(),
                    std::numeric_limits<int8_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((int8_t*)data)[i] = uniform_dist(rnd_engine);
            } else {
                std::uniform_int_distribution<uint16_t> uniform_dist(
                    std::numeric_limits<uint8_t>::min(),
                    std::numeric_limits<uint8_t>::max());
                for (unsigned int i = 0; i < n; i++)
                    ((uint8_t*)data)[i] = uniform_dist(rnd_engine);
            }
            break;
        default:
            throw "load_random_data: no support for data size > 8 or < 1"; // no
                                                                           // shenanigans
                                                                           // here
        }
    }
}

static std::vector<std::string> get_arch_list(volk_func_desc_t desc)
{
    std::vector<std::string> archlist;

    for (size_t i = 0; i < desc.n_impls; i++) {
        archlist.push_back(std::string(desc.impl_names[i]));
    }

    return archlist;
}

template <typename T>
T volk_lexical_cast(const std::string& str)
{
    for (unsigned int c_index = 0; c_index < str.size(); ++c_index) {
        if (str.at(c_index) < '0' || str.at(c_index) > '9') {
            throw "not all numbers!";
        }
    }
    T var;
    std::istringstream iss;
    iss.str(str);
    iss >> var;
    // deal with any error bits that may have been set on the stream
    return var;
}

volk_type_t volk_type_from_string(std::string name)
{
    volk_type_t type;
    type.is_float = false;
    type.is_scalar = false;
    type.is_complex = false;
    type.is_signed = false;
    type.size = 0;
    type.str = name;

    if (name.size() < 2) {
        throw std::string("name too short to be a datatype");
    }

    // is it a scalar?
    if (name[0] == 's') {
        type.is_scalar = true;
        name = name.substr(1, name.size() - 1);
    }

    // get the data size
    size_t last_size_pos = name.find_last_of("0123456789");
    if (last_size_pos == std::string::npos) {
        throw std::string("no size spec in type ").append(name);
    }
    // will throw if malformed
    int size = volk_lexical_cast<int>(name.substr(0, last_size_pos + 1));

    assert(((size % 8) == 0) && (size <= 64) && (size != 0));
    type.size = size / 8; // in bytes

    for (size_t i = last_size_pos + 1; i < name.size(); i++) {
        switch (name[i]) {
        case 'f':
            type.is_float = true;
            break;
        case 'i':
            type.is_signed = true;
            break;
        case 'c':
            type.is_complex = true;
            break;
        case 'u':
            type.is_signed = false;
            break;
        default:
            throw std::string("Error: no such type: '") + name[i] + "'";
        }
    }

    return type;
}

std::vector<std::string> split_signature(const std::string& protokernel_signature)
{
    std::vector<std::string> signature_tokens;
    std::string token;
    for (unsigned int loc = 0; loc < protokernel_signature.size(); ++loc) {
        if (protokernel_signature.at(loc) == '_') {
            // this is a break
            signature_tokens.push_back(token);
            token = "";
        } else {
            token.push_back(protokernel_signature.at(loc));
        }
    }
    // Get the last one to the end of the string
    signature_tokens.push_back(token);
    return signature_tokens;
}

static void get_signatures_from_name(std::vector<volk_type_t>& inputsig,
                                     std::vector<volk_type_t>& outputsig,
                                     std::string name)
{

    std::vector<std::string> toked = split_signature(name);

    assert(toked[0] == "volk");
    toked.erase(toked.begin());

    // ok. we're assuming a string in the form
    //(sig)_(multiplier-opt)_..._(name)_(sig)_(multiplier-opt)_..._(alignment)

    enum { SIDE_INPUT, SIDE_NAME, SIDE_OUTPUT } side = SIDE_INPUT;
    std::string fn_name;
    volk_type_t type;
    for (unsigned int token_index = 0; token_index < toked.size(); ++token_index) {
        std::string token = toked[token_index];
        try {
            type = volk_type_from_string(token);
            if (side == SIDE_NAME)
                side = SIDE_OUTPUT; // if this is the first one after the name...

            if (side == SIDE_INPUT)
                inputsig.push_back(type);
            else
                outputsig.push_back(type);
        } catch (...) {
            if (token[0] == 'x' && (token.size() > 1) &&
                (token[1] > '0' && token[1] < '9')) { // it's a multiplier
                if (side == SIDE_INPUT)
                    assert(inputsig.size() > 0);
                else
                    assert(outputsig.size() > 0);
                int multiplier = volk_lexical_cast<int>(
                    token.substr(1, token.size() - 1)); // will throw if invalid
                for (int i = 1; i < multiplier; i++) {
                    if (side == SIDE_INPUT)
                        inputsig.push_back(inputsig.back());
                    else
                        outputsig.push_back(outputsig.back());
                }
            } else if (side ==
                       SIDE_INPUT) { // it's the function name, at least it better be
                side = SIDE_NAME;
                fn_name.append("_");
                fn_name.append(token);
            } else if (side == SIDE_OUTPUT) {
                if (token != toked.back())
                    throw; // the last token in the name is the alignment
            }
        }
    }
    // we don't need an output signature (some fn's operate on the input data, "in
    // place"), but we do need at least one input!
    assert(inputsig.size() != 0);
}

inline void run_cast_test1(volk_fn_1arg func,
                           std::vector<void*>& buffs,
                           unsigned int vlen,
                           unsigned int iter,
                           std::string arch)
{
    while (iter--)
        func(buffs[0], vlen, arch.c_str());
}

inline void run_cast_test2(volk_fn_2arg func,
                           std::vector<void*>& buffs,
                           unsigned int vlen,
                           unsigned int iter,
                           std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], vlen, arch.c_str());
}

inline void run_cast_test3(volk_fn_3arg func,
                           std::vector<void*>& buffs,
                           unsigned int vlen,
                           unsigned int iter,
                           std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], buffs[2], vlen, arch.c_str());
}

inline void run_cast_test4(volk_fn_4arg func,
                           std::vector<void*>& buffs,
                           unsigned int vlen,
                           unsigned int iter,
                           std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], buffs[2], buffs[3], vlen, arch.c_str());
}

inline void run_cast_test1_s32f(volk_fn_1arg_s32f func,
                                std::vector<void*>& buffs,
                                float scalar,
                                unsigned int vlen,
                                unsigned int iter,
                                std::string arch)
{
    while (iter--)
        func(buffs[0], scalar, vlen, arch.c_str());
}

inline void run_cast_test2_s32f(volk_fn_2arg_s32f func,
                                std::vector<void*>& buffs,
                                float scalar,
                                unsigned int vlen,
                                unsigned int iter,
                                std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], scalar, vlen, arch.c_str());
}

inline void run_cast_test3_s32f(volk_fn_3arg_s32f func,
                                std::vector<void*>& buffs,
                                float scalar,
                                unsigned int vlen,
                                unsigned int iter,
                                std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], buffs[2], scalar, vlen, arch.c_str());
}

inline void run_cast_test1_s32fc(volk_fn_1arg_s32fc func,
                                 std::vector<void*>& buffs,
                                 lv_32fc_t scalar,
                                 unsigned int vlen,
                                 unsigned int iter,
                                 std::string arch)
{
    while (iter--)
        func(buffs[0], &scalar, vlen, arch.c_str());
}

inline void run_cast_test2_s32fc(volk_fn_2arg_s32fc func,
                                 std::vector<void*>& buffs,
                                 lv_32fc_t scalar,
                                 unsigned int vlen,
                                 unsigned int iter,
                                 std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], &scalar, vlen, arch.c_str());
}

inline void run_cast_test3_s32fc(volk_fn_3arg_s32fc func,
                                 std::vector<void*>& buffs,
                                 lv_32fc_t scalar,
                                 unsigned int vlen,
                                 unsigned int iter,
                                 std::string arch)
{
    while (iter--)
        func(buffs[0], buffs[1], buffs[2], &scalar, vlen, arch.c_str());
}

template <class t>
bool fcompare(t* in1, t* in2, unsigned int vlen, float tol, bool absolute_mode)
{
    bool fail = false;
    int print_max_errs = 10;
    for (unsigned int i = 0; i < vlen; i++) {
        // Check for special values (NaN, inf)
        bool in1_special = std::isnan(((t*)(in1))[i]) || std::isinf(((t*)(in1))[i]);
        bool in2_special = std::isnan(((t*)(in2))[i]) || std::isinf(((t*)(in2))[i]);

        if (in1_special || in2_special) {
            // For NaN: both must be NaN (NaN != NaN, so use isnan)
            // For inf: both must be same signed infinity
            bool values_match =
                (std::isnan(((t*)(in1))[i]) && std::isnan(((t*)(in2))[i])) ||
                (((t*)(in1))[i] == ((t*)(in2))[i]);
            if (!values_match) {
                fail = true;
                if (print_max_errs-- > 0) {
                    std::cout << "offset " << i << " in1: " << t(((t*)(in1))[i])
                              << " in2: " << t(((t*)(in2))[i]);
                    std::cout << " tolerance was: " << tol << std::endl;
                }
            }
            continue; // Skip normal comparison for special values
        }

        if (absolute_mode) {
            if (fabs(((t*)(in1))[i] - ((t*)(in2))[i]) > tol) {
                fail = true;
                if (print_max_errs-- > 0) {
                    std::cout << "offset " << i << " in1: " << t(((t*)(in1))[i])
                              << " in2: " << t(((t*)(in2))[i]);
                    std::cout << " tolerance was: " << tol << std::endl;
                }
            }
        } else {
            // for very small numbers we'll see round off errors due to limited
            // precision. So a special test case...
            if (fabs(((t*)(in1))[i]) < 1e-30) {
                if (fabs(((t*)(in2))[i]) > tol) {
                    fail = true;
                    if (print_max_errs-- > 0) {
                        std::cout << "offset " << i << " in1: " << t(((t*)(in1))[i])
                                  << " in2: " << t(((t*)(in2))[i]);
                        std::cout << " tolerance was: " << tol << std::endl;
                    }
                }
            }
            // the primary test is the percent different greater than given tol
            else if (fabs(((t*)(in1))[i] - ((t*)(in2))[i]) / fabs(((t*)in1)[i]) > tol) {
                fail = true;
                if (print_max_errs-- > 0) {
                    std::cout << "offset " << i << " in1: " << t(((t*)(in1))[i])
                              << " in2: " << t(((t*)(in2))[i]);
                    std::cout << " tolerance was: " << tol << std::endl;
                }
            }
        }
    }

    return fail;
}

template <class t>
bool ccompare(t* in1, t* in2, unsigned int vlen, float tol, bool absolute_mode)
{
    bool fail = false;
    int print_max_errs = 10;
    for (unsigned int i = 0; i < 2 * vlen; i += 2) {
        // Check for special values (NaN, inf) and verify they match
        bool in1_has_special = std::isnan(in1[i]) || std::isnan(in1[i + 1]) ||
                               std::isinf(in1[i]) || std::isinf(in1[i + 1]);
        bool in2_has_special = std::isnan(in2[i]) || std::isnan(in2[i + 1]) ||
                               std::isinf(in2[i]) || std::isinf(in2[i + 1]);

        if (in1_has_special || in2_has_special) {
            // For NaN: both must be NaN (NaN != NaN, so use isnan)
            // For inf: both must be same signed infinity
            bool real_match =
                (std::isnan(in1[i]) && std::isnan(in2[i])) || (in1[i] == in2[i]);
            bool imag_match = (std::isnan(in1[i + 1]) && std::isnan(in2[i + 1])) ||
                              (in1[i + 1] == in2[i + 1]);

            if (!real_match || !imag_match) {
                fail = true;
                if (print_max_errs-- > 0) {
                    std::cout << "offset " << i / 2 << " in1: " << in1[i] << " + "
                              << in1[i + 1] << "j  in2: " << in2[i] << " + " << in2[i + 1]
                              << "j";
                    std::cout << " tolerance was: " << tol << std::endl;
                }
            }
            continue; // Skip normal comparison for special values
        }
        t diff[2] = { in1[i] - in2[i], in1[i + 1] - in2[i + 1] };
        t err = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1]);
        t norm = std::sqrt(in1[i] * in1[i] + in1[i + 1] * in1[i + 1]);

        if (absolute_mode) {
            if (err > tol) {
                fail = true;
                if (print_max_errs-- > 0) {
                    std::cout << "offset " << i / 2 << " in1: " << in1[i] << " + "
                              << in1[i + 1] << "j  in2: " << in2[i] << " + " << in2[i + 1]
                              << "j";
                    std::cout << " tolerance was: " << tol << std::endl;
                }
            }
        } else {
            // for very small numbers we'll see round off errors due to limited
            // precision. So a special test case...
            if (norm < 1e-30) {
                if (err > tol) {
                    fail = true;
                    if (print_max_errs-- > 0) {
                        std::cout << "offset " << i / 2 << " in1: " << in1[i] << " + "
                                  << in1[i + 1] << "j  in2: " << in2[i] << " + "
                                  << in2[i + 1] << "j";
                        std::cout << " tolerance was: " << tol << std::endl;
                    }
                }
            }
            // the primary test is the percent different greater than given tol
            else if ((err / norm) > tol) {
                fail = true;
                if (print_max_errs-- > 0) {
                    std::cout << "offset " << i / 2 << " in1: " << in1[i] << " + "
                              << in1[i + 1] << "j  in2: " << in2[i] << " + " << in2[i + 1]
                              << "j";
                    std::cout << " tolerance was: " << tol << std::endl;
                }
            }
        }
    }

    return fail;
}

template <class t>
bool icompare(t* in1, t* in2, unsigned int vlen, unsigned int tol)
{
    bool fail = false;
    int print_max_errs = 10;
    for (unsigned int i = 0; i < vlen; i++) {
        if (((uint64_t)abs(int64_t(((t*)(in1))[i]) - int64_t(((t*)(in2))[i]))) > tol) {
            fail = true;
            if (print_max_errs-- > 0) {
                std::cout << "offset " << i
                          << " in1: " << static_cast<int64_t>(t(((t*)(in1))[i]))
                          << " in2: " << static_cast<int64_t>(t(((t*)(in2))[i]));
                std::cout << " tolerance was: " << tol << std::endl;
            }
        }
    }

    return fail;
}

class volk_qa_aligned_mem_pool
{
public:
    void* get_new(size_t size)
    {
        size_t alignment = volk_get_alignment();
        void* ptr = volk_malloc(size, alignment);
        memset(ptr, 0x00, size);
        _mems.push_back(ptr);
        return ptr;
    }
    ~volk_qa_aligned_mem_pool()
    {
        for (unsigned int ii = 0; ii < _mems.size(); ++ii) {
            volk_free(_mems[ii]);
        }
    }

private:
    std::vector<void*> _mems;
};

bool run_volk_tests(volk_func_desc_t desc,
                    void (*manual_func)(),
                    std::string name,
                    volk_test_params_t test_params,
                    std::vector<volk_test_results_t>* results,
                    std::string puppet_master_name)
{
    return run_volk_tests(desc,
                          manual_func,
                          name,
                          test_params.tol(),
                          test_params.scalar(),
                          test_params.vlen(),
                          test_params.iter(),
                          results,
                          puppet_master_name,
                          test_params.absolute_mode(),
                          test_params.benchmark_mode(),
                          test_params.float_edge_cases(),
                          test_params.complex_edge_cases());
}

bool run_volk_tests(volk_func_desc_t desc,
                    void (*manual_func)(),
                    std::string name,
                    float tol,
                    lv_32fc_t scalar,
                    unsigned int vlen,
                    unsigned int iter,
                    std::vector<volk_test_results_t>* results,
                    std::string puppet_master_name,
                    bool absolute_mode,
                    bool benchmark_mode,
                    const std::vector<float>& float_edge_cases,
                    const std::vector<lv_32fc_t>& complex_edge_cases)
{
    // Initialize this entry in results vector
    results->push_back(volk_test_results_t());
    results->back().name = name;
    results->back().vlen = vlen;
    results->back().iter = iter;
    std::cout << std::endl; // Blank line for separation
    std::cout << "RUN_VOLK_TESTS: " << name << "(" << vlen << "," << iter << ")"
              << std::endl;

    // vlen_twiddle will increase vlen for malloc and data generation
    // but kernels will still be called with the user provided vlen.
    // This is useful for causing errors in kernels that do bad reads
    const unsigned int vlen_twiddle = 5;
    vlen = vlen + vlen_twiddle;

    const float tol_f = tol;
    const unsigned int tol_i = static_cast<unsigned int>(tol);

    // first let's get a list of available architectures for the test
    std::vector<std::string> arch_list = get_arch_list(desc);

    // Reorder arch_list to put generic implementations first for consistent output
    // Priority: "generic" first, then other generic_* variants, then everything else
    std::vector<std::string> plain_generic;
    std::vector<std::string> other_generic_impls;
    std::vector<std::string> other_impls;
    for (const auto& arch : arch_list) {
        if (arch == "generic") {
            plain_generic.push_back(arch);
        } else if (arch.find("generic") == 0) { // starts with "generic"
            other_generic_impls.push_back(arch);
        } else {
            other_impls.push_back(arch);
        }
    }
    arch_list.clear();
    arch_list.insert(arch_list.end(), plain_generic.begin(), plain_generic.end());
    arch_list.insert(
        arch_list.end(), other_generic_impls.begin(), other_generic_impls.end());
    arch_list.insert(arch_list.end(), other_impls.begin(), other_impls.end());

    if ((!benchmark_mode) && (arch_list.size() < 2)) {
        std::cout << "no architectures to test" << std::endl;
        return false;
    }

    // something that can hang onto memory and cleanup when this function exits
    volk_qa_aligned_mem_pool mem_pool;

    // now we have to get a function signature by parsing the name
    std::vector<volk_type_t> inputsig, outputsig;
    try {
        get_signatures_from_name(inputsig, outputsig, name);
    } catch (std::exception& error) {
        std::cerr << "Error: unable to get function signature from kernel name"
                  << std::endl;
        std::cerr << "  - " << name << std::endl;
        return false;
    }

    // pull the input scalars into their own vector
    std::vector<volk_type_t> inputsc;
    for (size_t i = 0; i < inputsig.size(); i++) {
        if (inputsig[i].is_scalar) {
            inputsc.push_back(inputsig[i]);
            inputsig.erase(inputsig.begin() + i);
            i -= 1;
        }
    }
    std::vector<void*> inbuffs;
    for (unsigned int inputsig_index = 0; inputsig_index < inputsig.size();
         ++inputsig_index) {
        volk_type_t sig = inputsig[inputsig_index];
        if (!sig.is_scalar) // we don't make buffers for scalars
            inbuffs.push_back(
                mem_pool.get_new(vlen * sig.size * (sig.is_complex ? 2 : 1)));
    }
    for (size_t i = 0; i < inbuffs.size(); i++) {
        load_random_data(
            inbuffs[i], inputsig[i], vlen, float_edge_cases, complex_edge_cases);
    }

    // ok let's make a vector of vector of void buffers, which holds the input/output
    // vectors for each arch
    std::vector<std::vector<void*>> test_data;
    for (size_t i = 0; i < arch_list.size(); i++) {
        std::vector<void*> arch_buffs;
        for (size_t j = 0; j < outputsig.size(); j++) {
            arch_buffs.push_back(mem_pool.get_new(vlen * outputsig[j].size *
                                                  (outputsig[j].is_complex ? 2 : 1)));
        }
        for (size_t j = 0; j < inputsig.size(); j++) {
            void* arch_inbuff = mem_pool.get_new(vlen * inputsig[j].size *
                                                 (inputsig[j].is_complex ? 2 : 1));
            memcpy(arch_inbuff,
                   inbuffs[j],
                   vlen * inputsig[j].size * (inputsig[j].is_complex ? 2 : 1));
            arch_buffs.push_back(arch_inbuff);
        }
        test_data.push_back(arch_buffs);
    }

    std::vector<volk_type_t> both_sigs;
    both_sigs.insert(both_sigs.end(), outputsig.begin(), outputsig.end());
    both_sigs.insert(both_sigs.end(), inputsig.begin(), inputsig.end());

    // now run the test
    vlen = vlen - vlen_twiddle;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::vector<double> profile_times;

    // Warmup to let CPU reach full turbo frequency (only for first kernel)
    const double warmup_target_ms = g_warmup_done ? 0.0 : volk_test_get_warmup_ms();
    {
        // Run a quick test to estimate time per iteration
        start = std::chrono::system_clock::now();
        switch (both_sigs.size()) {
        case 1:
            if (inputsc.size() == 0) {
                run_cast_test1(
                    (volk_fn_1arg)(manual_func), test_data[0], vlen, iter, "generic");
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test1_s32fc((volk_fn_1arg_s32fc)(manual_func),
                                         test_data[0],
                                         scalar,
                                         vlen,
                                         iter,
                                         "generic");
                } else {
                    run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                        test_data[0],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        "generic");
                }
            }
            break;
        case 2:
            if (inputsc.size() == 0) {
                run_cast_test2(
                    (volk_fn_2arg)(manual_func), test_data[0], vlen, iter, "generic");
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test2_s32fc((volk_fn_2arg_s32fc)(manual_func),
                                         test_data[0],
                                         scalar,
                                         vlen,
                                         iter,
                                         "generic");
                } else {
                    run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                        test_data[0],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        "generic");
                }
            }
            break;
        case 3:
            if (inputsc.size() == 0) {
                run_cast_test3(
                    (volk_fn_3arg)(manual_func), test_data[0], vlen, iter, "generic");
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test3_s32fc((volk_fn_3arg_s32fc)(manual_func),
                                         test_data[0],
                                         scalar,
                                         vlen,
                                         iter,
                                         "generic");
                } else {
                    run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                        test_data[0],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        "generic");
                }
            }
            break;
        case 4:
            run_cast_test4(
                (volk_fn_4arg)(manual_func), test_data[0], vlen, iter, "generic");
            break;
        default:
            break;
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double test_time_ms = 1000.0 * elapsed.count();

        // If we haven't reached 500ms yet, calculate how many more iterations we need
        if (test_time_ms < warmup_target_ms) {
            double remaining_ms = warmup_target_ms - test_time_ms;
            unsigned int warmup_iterations =
                (unsigned int)((remaining_ms / test_time_ms) * iter);
            if (warmup_iterations > 0) {
                // Run additional warmup iterations
                switch (both_sigs.size()) {
                case 1:
                    if (inputsc.size() == 0) {
                        run_cast_test1((volk_fn_1arg)(manual_func),
                                       test_data[0],
                                       vlen,
                                       warmup_iterations,
                                       "generic");
                    } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                        if (inputsc[0].is_complex) {
                            run_cast_test1_s32fc((volk_fn_1arg_s32fc)(manual_func),
                                                 test_data[0],
                                                 scalar,
                                                 vlen,
                                                 warmup_iterations,
                                                 "generic");
                        } else {
                            run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                                test_data[0],
                                                scalar.real(),
                                                vlen,
                                                warmup_iterations,
                                                "generic");
                        }
                    }
                    break;
                case 2:
                    if (inputsc.size() == 0) {
                        run_cast_test2((volk_fn_2arg)(manual_func),
                                       test_data[0],
                                       vlen,
                                       warmup_iterations,
                                       "generic");
                    } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                        if (inputsc[0].is_complex) {
                            run_cast_test2_s32fc((volk_fn_2arg_s32fc)(manual_func),
                                                 test_data[0],
                                                 scalar,
                                                 vlen,
                                                 warmup_iterations,
                                                 "generic");
                        } else {
                            run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                                test_data[0],
                                                scalar.real(),
                                                vlen,
                                                warmup_iterations,
                                                "generic");
                        }
                    }
                    break;
                case 3:
                    if (inputsc.size() == 0) {
                        run_cast_test3((volk_fn_3arg)(manual_func),
                                       test_data[0],
                                       vlen,
                                       warmup_iterations,
                                       "generic");
                    } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                        if (inputsc[0].is_complex) {
                            run_cast_test3_s32fc((volk_fn_3arg_s32fc)(manual_func),
                                                 test_data[0],
                                                 scalar,
                                                 vlen,
                                                 warmup_iterations,
                                                 "generic");
                        } else {
                            run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                                test_data[0],
                                                scalar.real(),
                                                vlen,
                                                warmup_iterations,
                                                "generic");
                        }
                    }
                    break;
                case 4:
                    run_cast_test4((volk_fn_4arg)(manual_func),
                                   test_data[0],
                                   vlen,
                                   warmup_iterations,
                                   "generic");
                    break;
                default:
                    break;
                }
            }
        }
        g_warmup_done = true;
    }

    // Reset all test buffers after warmup
    for (size_t i = 0; i < arch_list.size(); i++) {
        for (size_t j = 0; j < outputsig.size(); j++) {
            memset(test_data[i][j],
                   0,
                   vlen * outputsig[j].size * (outputsig[j].is_complex ? 2 : 1));
        }
        // Reload input buffers from original data
        for (size_t j = 0; j < inputsig.size(); j++) {
            memcpy(test_data[i][outputsig.size() + j],
                   inbuffs[j],
                   vlen * inputsig[j].size * (inputsig[j].is_complex ? 2 : 1));
        }
    }

    for (size_t i = 0; i < arch_list.size(); i++) {
        start = std::chrono::system_clock::now();

        switch (both_sigs.size()) {
        case 1:
            if (inputsc.size() == 0) {
                run_cast_test1(
                    (volk_fn_1arg)(manual_func), test_data[i], vlen, iter, arch_list[i]);
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test1_s32fc((volk_fn_1arg_s32fc)(manual_func),
                                         test_data[i],
                                         scalar,
                                         vlen,
                                         iter,
                                         arch_list[i]);
                } else {
                    run_cast_test1_s32f((volk_fn_1arg_s32f)(manual_func),
                                        test_data[i],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        arch_list[i]);
                }
            } else
                throw "unsupported 1 arg function >1 scalars";
            break;
        case 2:
            if (inputsc.size() == 0) {
                run_cast_test2(
                    (volk_fn_2arg)(manual_func), test_data[i], vlen, iter, arch_list[i]);
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test2_s32fc((volk_fn_2arg_s32fc)(manual_func),
                                         test_data[i],
                                         scalar,
                                         vlen,
                                         iter,
                                         arch_list[i]);
                } else {
                    run_cast_test2_s32f((volk_fn_2arg_s32f)(manual_func),
                                        test_data[i],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        arch_list[i]);
                }
            } else
                throw "unsupported 2 arg function >1 scalars";
            break;
        case 3:
            if (inputsc.size() == 0) {
                run_cast_test3(
                    (volk_fn_3arg)(manual_func), test_data[i], vlen, iter, arch_list[i]);
            } else if (inputsc.size() == 1 && inputsc[0].is_float) {
                if (inputsc[0].is_complex) {
                    run_cast_test3_s32fc((volk_fn_3arg_s32fc)(manual_func),
                                         test_data[i],
                                         scalar,
                                         vlen,
                                         iter,
                                         arch_list[i]);
                } else {
                    run_cast_test3_s32f((volk_fn_3arg_s32f)(manual_func),
                                        test_data[i],
                                        scalar.real(),
                                        vlen,
                                        iter,
                                        arch_list[i]);
                }
            } else
                throw "unsupported 3 arg function >1 scalars";
            break;
        case 4:
            run_cast_test4(
                (volk_fn_4arg)(manual_func), test_data[i], vlen, iter, arch_list[i]);
            break;
        default:
            throw "no function handler for this signature";
            break;
        }

        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        double arch_time = 1000.0 * elapsed_seconds.count();

        volk_test_time_t result;
        result.name = arch_list[i];
        result.time = arch_time;
        result.units = "ms";
        result.pass = true;
        results->back().results[result.name] = result;

        profile_times.push_back(arch_time);
    }

    // and now compare each output to the generic output
    // first we have to know which output is the generic one, they aren't in order...
    size_t generic_offset = 0;
    for (size_t i = 0; i < arch_list.size(); i++) {
        if (arch_list[i] == "generic") {
            generic_offset = i;
        }
    }

    // Just in case a kernel wrote to OOB memory, use the twiddled vlen
    vlen = vlen + vlen_twiddle;
    bool fail;
    bool fail_global = false;
    std::vector<bool> arch_results;
    for (size_t i = 0; i < arch_list.size(); i++) {
        fail = false;
        if (i != generic_offset) {
            for (size_t j = 0; j < both_sigs.size(); j++) {
                if (both_sigs[j].is_float) {
                    if (both_sigs[j].size == 8) {
                        if (both_sigs[j].is_complex) {
                            fail = ccompare((double*)test_data[generic_offset][j],
                                            (double*)test_data[i][j],
                                            vlen,
                                            tol_f,
                                            absolute_mode);
                        } else {
                            fail = fcompare((double*)test_data[generic_offset][j],
                                            (double*)test_data[i][j],
                                            vlen,
                                            tol_f,
                                            absolute_mode);
                        }
                    } else {
                        if (both_sigs[j].is_complex) {
                            fail = ccompare((float*)test_data[generic_offset][j],
                                            (float*)test_data[i][j],
                                            vlen,
                                            tol_f,
                                            absolute_mode);
                        } else {
                            fail = fcompare((float*)test_data[generic_offset][j],
                                            (float*)test_data[i][j],
                                            vlen,
                                            tol_f,
                                            absolute_mode);
                        }
                    }
                } else {
                    // i could replace this whole switch statement with a memcmp if i
                    // wasn't interested in printing the outputs where they differ
                    switch (both_sigs[j].size) {
                    case 8:
                        if (both_sigs[j].is_signed) {
                            fail = icompare((int64_t*)test_data[generic_offset][j],
                                            (int64_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i);
                        } else {
                            fail = icompare((uint64_t*)test_data[generic_offset][j],
                                            (uint64_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i);
                        }
                        break;
                    case 4:
                        if (both_sigs[j].is_complex) {
                            if (both_sigs[j].is_signed) {
                                fail = icompare((int16_t*)test_data[generic_offset][j],
                                                (int16_t*)test_data[i][j],
                                                vlen * (both_sigs[j].is_complex ? 2 : 1),
                                                tol_i);
                            } else {
                                fail = icompare((uint16_t*)test_data[generic_offset][j],
                                                (uint16_t*)test_data[i][j],
                                                vlen * (both_sigs[j].is_complex ? 2 : 1),
                                                tol_i);
                            }
                        } else {
                            if (both_sigs[j].is_signed) {
                                fail = icompare((int32_t*)test_data[generic_offset][j],
                                                (int32_t*)test_data[i][j],
                                                vlen * (both_sigs[j].is_complex ? 2 : 1),
                                                tol_i);
                            } else {
                                fail = icompare((uint32_t*)test_data[generic_offset][j],
                                                (uint32_t*)test_data[i][j],
                                                vlen * (both_sigs[j].is_complex ? 2 : 1),
                                                tol_i);
                            }
                        }
                        break;
                    case 2:
                        if (both_sigs[j].is_signed) {
                            fail = icompare((int16_t*)test_data[generic_offset][j],
                                            (int16_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i);
                        } else {
                            fail = icompare((uint16_t*)test_data[generic_offset][j],
                                            (uint16_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i);
                        }
                        break;
                    case 1:
                        if (both_sigs[j].is_signed) {
                            fail = icompare((int8_t*)test_data[generic_offset][j],
                                            (int8_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i);
                        } else {
                            fail = icompare((uint8_t*)test_data[generic_offset][j],
                                            (uint8_t*)test_data[i][j],
                                            vlen * (both_sigs[j].is_complex ? 2 : 1),
                                            tol_i);
                        }
                        break;
                    default:
                        fail = 1;
                    }
                }
                if (fail) {
                    volk_test_time_t* result = &results->back().results[arch_list[i]];
                    result->pass = false;
                    fail_global = true;
                    std::cout << name << ": fail on arch " << arch_list[i] << std::endl;
                }
            }
        }
        arch_results.push_back(!fail);
    }

    double best_time_a = std::numeric_limits<double>::max();
    double best_time_u = std::numeric_limits<double>::max();
    std::string best_arch_a = "generic";
    std::string best_arch_u = "generic";
    for (size_t i = 0; i < arch_list.size(); i++) {
        if ((profile_times[i] < best_time_u) && arch_results[i] &&
            desc.impl_alignment[i] == 0) {
            best_time_u = profile_times[i];
            best_arch_u = arch_list[i];
        }
        if ((profile_times[i] < best_time_a) && arch_results[i]) {
            best_time_a = profile_times[i];
            best_arch_a = arch_list[i];
        }
    }

    // Calculate total data transferred (bytes read + written) for throughput display
    size_t bytes_per_call = 0;
    for (size_t j = 0; j < outputsig.size(); j++) {
        bytes_per_call += outputsig[j].size * (outputsig[j].is_complex ? 2 : 1) * vlen;
    }
    for (size_t j = 0; j < inputsig.size(); j++) {
        bytes_per_call += inputsig[j].size * (inputsig[j].is_complex ? 2 : 1) * vlen;
    }
    double total_mb = (bytes_per_call * iter) / 1e6; // Total megabytes transferred

    // Build formatted output strings with proper alignment
    std::vector<std::string> output_lines;
    const int total_width = 60;
    int ms_end_position = 0;  // Track where " ms" ends (arch name alignment)
    int mbs_end_position = 0; // Track where "MB/s)" ends (speedup alignment)

    for (size_t i = 0; i < arch_list.size(); i++) {
        // Calculate throughput in MB/s
        double time_seconds = profile_times[i] / 1000.0;
        double throughput_mbps = total_mb / time_seconds;

        // Build the timing/throughput string
        std::ostringstream timing_str;
        timing_str << std::fixed << std::setprecision(4) << profile_times[i] << " ms"
                   << " (" << std::setw(8) << std::setprecision(1) << throughput_mbps
                   << " MB/s)";

        // Calculate padding needed (without star)
        int padding = total_width - arch_list[i].length() - timing_str.str().length();
        if (padding < 1)
            padding = 1;

        // Build the full line with left-adjusted name and right-adjusted timing
        std::string line = arch_list[i] + std::string(padding, ' ') + timing_str.str();

        // Add star if this is a best arch (after padding calculation)
        if (arch_list[i] == best_arch_a || arch_list[i] == best_arch_u) {
            line += " *";
        }

        // Track alignment positions
        if (i == 0) {
            size_t ms_pos = line.find(" ms");
            if (ms_pos != std::string::npos) {
                ms_end_position = ms_pos + 3; // Position after " ms"
            }
            size_t mbs_pos = line.find("MB/s)");
            if (mbs_pos != std::string::npos) {
                mbs_end_position = mbs_pos + 5; // Position after "MB/s)"
            }
        }

        output_lines.push_back(line);
    }

    // Print all lines
    for (const auto& line : output_lines) {
        std::cout << line << std::endl;
    }

    // Get generic timing for speedup calculation
    double generic_time = 0.0;
    for (size_t i = 0; i < arch_list.size(); i++) {
        if (arch_list[i] == "generic") {
            generic_time = profile_times[i];
            break;
        }
    }

    // Print best arch lines: arch name aligns to "ms", speedup ) aligns to MB/s )
    auto print_best_line = [&](const char* label, const std::string& arch, double time) {
        std::ostringstream speedup_str;
        if (arch != "generic" && generic_time > 0) {
            double speedup = generic_time / time;
            speedup_str << "(" << std::fixed << std::setprecision(2) << speedup << "x)";
        }

        // Arch name right-aligned to ms_end_position
        std::string line = label;
        int arch_padding = ms_end_position - line.length() - arch.length();
        if (arch_padding < 1)
            arch_padding = 1;
        line += std::string(arch_padding, ' ') + arch;

        // Speedup right-aligned to mbs_end_position
        if (speedup_str.str().length() > 0) {
            int speedup_padding =
                mbs_end_position - line.length() - speedup_str.str().length();
            if (speedup_padding < 1)
                speedup_padding = 1;
            line += std::string(speedup_padding, ' ') + speedup_str.str();
        }
        std::cout << line << std::endl;
    };

    print_best_line("Best aligned arch:", best_arch_a, best_time_a);
    print_best_line("Best unaligned arch:", best_arch_u, best_time_u);

    std::cout << std::string(80, '-') << std::endl;

    if (puppet_master_name == "NULL") {
        results->back().config_name = name;
    } else {
        results->back().config_name = puppet_master_name;
    }
    results->back().best_arch_a = best_arch_a;
    results->back().best_arch_u = best_arch_u;

    return fail_global;
}
