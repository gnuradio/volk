/* -*- c++ -*- */
/*
 * Copyright 2012-2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <stdbool.h> // for bool
#include <iosfwd>    // for ofstream
#include <string>    // for string
#include <vector>    // for vector

class volk_test_results_t;

void read_results(std::vector<volk_test_results_t>* results);
void read_results(std::vector<volk_test_results_t>* results, std::string path);
void write_results(const std::vector<volk_test_results_t>* results, bool update_result);
void write_results(const std::vector<volk_test_results_t>* results,
                   bool update_result,
                   const std::string path);
void write_json(std::ofstream& json_file, std::vector<volk_test_results_t> results);
