/* -*- c++ -*- */
/*
 * Copyright 2018-2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
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
                        if (arg_number + 1 >= argc) {
                            std::cerr << "Warning: option '" << argv[arg_number]
                                      << "' expects a numeric value" << std::endl;
                            break;
                        }
                        {
                            char* next_arg = argv[arg_number + 1];
                            bool is_number =
                                (next_arg[0] >= '0' && next_arg[0] <= '9') ||
                                (next_arg[0] == '-' && next_arg[1] >= '0' &&
                                 next_arg[1] <= '9');
                            if (!is_number) {
                                std::cerr << "Warning: option '" << argv[arg_number]
                                          << "' expects a numeric value" << std::endl;
                                break;
                            }
                        }
                        int_val = atoi(argv[++arg_number]);
                        ((void (*)(int))this_option->callback)(int_val);
                    } catch (std::exception& exc) {
                        std::cerr << "An int option can only receive a number"
                                  << std::endl;
                        throw std::exception();
                    };
                    break;
                case FLOAT_CALLBACK:
                    try {
                        if (arg_number + 1 >= argc) {
                            std::cerr << "Warning: option '" << argv[arg_number]
                                      << "' expects a numeric value" << std::endl;
                            break;
                        }
                        {
                            char* next_arg = argv[arg_number + 1];
                            bool is_number =
                                (next_arg[0] >= '0' && next_arg[0] <= '9') ||
                                (next_arg[0] == '-' && next_arg[1] >= '0' &&
                                 next_arg[1] <= '9') ||
                                (next_arg[0] == '.');
                            if (!is_number) {
                                std::cerr << "Warning: option '" << argv[arg_number]
                                          << "' expects a numeric value" << std::endl;
                                break;
                            }
                        }
                        double double_val = atof(argv[++arg_number]);
                        ((void (*)(float))this_option->callback)(double_val);
                    } catch (std::exception& exc) {
                        std::cerr << "A float option can only receive a number"
                                  << std::endl;
                        throw std::exception();
                    };
                    break;
                case BOOL_CALLBACK:
                    if (arg_number == (argc - 1)) { // this is the last arg
                        int_val = 1;
                    } else { // sneak a look at the next arg since it's present
                        char* next_arg = argv[arg_number + 1];
                        if (strncmp(next_arg, "-", 1) == 0) {
                            // the next arg is actually a flag; the bool is just
                            // present, set to true
                            int_val = 1;
                        } else if (strcmp(next_arg, "true") == 0) {
                            int_val = 1;
                        } else if (strcmp(next_arg, "false") == 0) {
                            int_val = 0;
                        } else if (next_arg[0] >= '0' && next_arg[0] <= '9') {
                            // consume an explicit numeric bool value (0 or 1)
                            int_val = (bool)atoi(argv[++arg_number]);
                        } else {
                            // unrecognized token: treat flag as present=true,
                            // do not consume the next argument
                            int_val = 1;
                        }
                    }
                    if (int_val) {
                        ((void (*)(bool))this_option->callback)(int_val);
                    }
                    break;
                case STRING_CALLBACK:
                    try {
                        if (arg_number + 1 >= argc) {
                            std::cerr << "Warning: option '" << argv[arg_number]
                                      << "' expects a value" << std::endl;
                            break;
                        }
                        ((void (*)(std::string))this_option->callback)(
                            argv[++arg_number]);
                    } catch (std::exception& exc) {
                        throw std::exception();
                    };
                    break;
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

        while (help_line.size() < 32) {
            help_line += " ";
        }
        help_line += this_option->msg;
        std::cout << help_line << std::endl;
    }
}
