/* -*- c++ -*- */
/*
 * Copyright 2018-2020 Free Software Foundation, Inc.
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

#ifndef VOLK_VOLK_OPTION_HELPERS_H
#define VOLK_VOLK_OPTION_HELPERS_H

#include <limits.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>

typedef enum {
    VOID_CALLBACK,
    INT_CALLBACK,
    BOOL_CALLBACK,
    STRING_CALLBACK,
    FLOAT_CALLBACK,
    STRING,
} VOLK_OPTYPE;

class option_t
{
public:
    option_t(std::string t_longform,
             std::string t_shortform,
             std::string t_msg,
             void (*t_callback)());
    option_t(std::string t_longform,
             std::string t_shortform,
             std::string t_msg,
             void (*t_callback)(int));
    option_t(std::string t_longform,
             std::string t_shortform,
             std::string t_msg,
             void (*t_callback)(float));
    option_t(std::string t_longform,
             std::string t_shortform,
             std::string t_msg,
             void (*t_callback)(bool));
    option_t(std::string t_longform,
             std::string t_shortform,
             std::string t_msg,
             void (*t_callback)(std::string));
    option_t(std::string t_longform,
             std::string t_shortform,
             std::string t_msg,
             std::string t_printval);

    std::string longform;
    std::string shortform;
    std::string msg;
    VOLK_OPTYPE option_type;
    std::string printval;
    void (*callback)();
};

class option_list
{
public:
    option_list(std::string program_name);
    bool present(std::string option_name);

    void add(option_t opt);

    void parse(int argc, char** argv);

    void help();

private:
    std::string d_program_name;
    std::vector<option_t> d_internal_list;
    std::map<std::string, int> d_present_options;
};


#endif // VOLK_VOLK_OPTION_HELPERS_H
