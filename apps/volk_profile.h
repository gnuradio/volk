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
