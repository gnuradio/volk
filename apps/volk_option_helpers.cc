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


#include "volk_option_helpers.h"

#include <limits.h>  // IWYU pragma: keep
#include <cstdlib>   // IWYU pragma: keep
#include <cstring>   // IWYU pragma: keep
#include <exception> // for exception
#include <iostream>  // for operator<<, endl, basic_ostream, cout, ostream
#include <utility>   // for pair

/*
 * Option type
 */
option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)())
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback(t_callback)
{
    option_type = VOID_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)(int))
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback((void (*)())t_callback)
{
    option_type = INT_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)(float))
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback((void (*)())t_callback)
{
    option_type = FLOAT_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)(bool))
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback((void (*)())t_callback)
{
    option_type = BOOL_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)(std::string))
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback((void (*)())t_callback)
{
    option_type = STRING_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   std::string t_printval)
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      printval(t_printval)
{
    option_type = STRING;
}


/*
 * Option List
 */

option_list::option_list(std::string program_name) : d_program_name(program_name)
{
    d_internal_list = std::vector<option_t>();
}


void option_list::add(option_t opt) { d_internal_list.push_back(opt); }

void option_list::parse(int argc, char** argv)
{
    for (int arg_number = 0; arg_number < argc; ++arg_number) {
        for (std::vector<option_t>::iterator this_option = d_internal_list.begin();
             this_option != d_internal_list.end();
             this_option++) {
            int int_val = INT_MIN;
            if (this_option->longform == std::string(argv[arg_number]) ||
                this_option->shortform == std::string(argv[arg_number])) {

                if (d_present_options.count(this_option->longform) == 0) {
                    d_present_options.insert(
                        std::pair<std::string, int>(this_option->longform, 1));
                } else {
                    d_present_options[this_option->longform] += 1;
                }
                switch (this_option->option_type) {
                case VOID_CALLBACK:
                    this_option->callback();
                    break;
                case INT_CALLBACK:
                    try {
                        int_val = atoi(argv[++arg_number]);
                        ((void (*)(int))this_option->callback)(int_val);
                    } catch (std::exception& exc) {
                        std::cout << "An int option can only receive a number"
                                  << std::endl;
                        throw std::exception();
                    };
                    break;
                case FLOAT_CALLBACK:
                    try {
                        double double_val = atof(argv[++arg_number]);
                        ((void (*)(float))this_option->callback)(double_val);
                    } catch (std::exception& exc) {
                        std::cout << "A float option can only receive a number"
                                  << std::endl;
                        throw std::exception();
                    };
                    break;
                case BOOL_CALLBACK:
                    try {
                        if (arg_number == (argc - 1)) { // this is the last arg
                            int_val = 1;
                        } else { // sneak a look at the next arg since it's present
                            char* next_arg = argv[arg_number + 1];
                            if ((strncmp(next_arg, "-", 1) == 0) ||
                                (strncmp(next_arg, "--", 2) == 0)) {
                                // the next arg is actually an arg, the bool is just
                                // present, set to true
                                int_val = 1;
                            } else if (strncmp(next_arg, "true", 4) == 0) {
                                int_val = 1;
                            } else if (strncmp(next_arg, "false", 5) == 0) {
                                int_val = 0;
                            } else {
                                // we got a number or a string.
                                // convert it to a number and depend on the catch to
                                // report an error condition
                                int_val = (bool)atoi(argv[++arg_number]);
                            }
                        }
                    } catch (std::exception& e) {
                        int_val = INT_MIN;
                    };
                    if (int_val == INT_MIN) {
                        std::cout
                            << "option: '" << argv[arg_number - 1]
                            << "' -> received an unknown value. Boolean "
                               "options should receive one of '0', '1', 'true', 'false'."
                            << std::endl;
                        throw std::exception();
                    } else if (int_val) {
                        ((void (*)(bool))this_option->callback)(int_val);
                    }
                    break;
                case STRING_CALLBACK:
                    try {
                        ((void (*)(std::string))this_option->callback)(
                            argv[++arg_number]);
                    } catch (std::exception& exc) {
                        throw std::exception();
                    };
                case STRING:
                    std::cout << this_option->printval << std::endl;
                    break;
                }
            }
        }
        if (std::string("--help") == std::string(argv[arg_number]) ||
            std::string("-h") == std::string(argv[arg_number])) {
            d_present_options.insert(std::pair<std::string, int>("--help", 1));
            help();
        }
    }
}

bool option_list::present(std::string option_name)
{
    if (d_present_options.count("--" + option_name)) {
        return true;
    } else {
        return false;
    }
}

void option_list::help()
{
    std::cout << d_program_name << std::endl;
    std::cout << "  -h [ --help ] \t\tdisplay this help message" << std::endl;
    for (std::vector<option_t>::iterator this_option = d_internal_list.begin();
         this_option != d_internal_list.end();
         this_option++) {
        std::string help_line("  ");
        if (this_option->shortform == "-") {
            help_line += this_option->longform + " ";
        } else {
            help_line += this_option->shortform + " [ " + this_option->longform + " ]";
        }

        switch (help_line.size() / 8) {
        case 0:
            help_line += "\t";
        case 1:
            help_line += "\t";
        case 2:
            help_line += "\t";
        case 3:
            help_line += "\t";
        }
        help_line += this_option->msg;
        std::cout << help_line << std::endl;
    }
}
