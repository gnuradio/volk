/* -*- c++ -*- */
/*
 * Copyright 2012-2014 Free Software Foundation, Inc.
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

#include "qa_utils.h"
#include "kernel_tests.h"

#include <volk/volk.h>

#include <iostream>


int main()
{
    bool qa_ret_val = 0;

    float def_tol = 1e-6;
    lv_32fc_t def_scalar = 327.0;
    int def_iter = 1;
    int def_vlen = 131071;
    bool def_benchmark_mode = true;
    std::string def_kernel_regex = "";

    volk_test_params_t test_params(def_tol, def_scalar, def_vlen, def_iter,
        def_benchmark_mode, def_kernel_regex);
    std::vector<volk_test_case_t> test_cases = init_test_list(test_params);

    std::vector<std::string> qa_failures;
    // Test every kernel reporting failures when they occur
    for(unsigned int ii = 0; ii < test_cases.size(); ++ii) {
        bool qa_result = false;
        volk_test_case_t test_case = test_cases[ii];
        try {
            qa_result = run_volk_tests(test_case.desc(), test_case.kernel_ptr(), test_case.name(),
                test_case.test_parameters(), 0, test_case.puppet_master_name());
        }
        catch(...) {
            // TODO: what exceptions might we need to catch and how do we handle them?
            std::cerr << "Exception found on kernel: " << test_case.name() << std::endl;
            qa_result = false;
        }

        if(qa_result) {
            std::cerr << "Failure on " << test_case.name() << std::endl;
            qa_failures.push_back(test_case.name());
        }
    }

    // Summarize QA results
    std::cerr << "Kernel QA finished: " << qa_failures.size() << " failures out of "
        << test_cases.size() << " tests." << std::endl;
    if(qa_failures.size() > 0) {
        std::cerr << "The following kernels failed QA:" << std::endl;
        for(unsigned int ii = 0; ii < qa_failures.size(); ++ii) {
            std::cerr << "    " << qa_failures[ii] << std::endl;
        }
        qa_ret_val = 1;
    }

    return qa_ret_val;
}
