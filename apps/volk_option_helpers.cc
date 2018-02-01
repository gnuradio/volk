//
// Created by nathan on 2/1/18.
//

#include "volk_option_helpers.h"

#include <iostream>

/*
 * Option type
 */
option_t::option_t(std::string longform, std::string shortform, std::string msg, void (*callback)())
    : longform("--" + longform),
      shortform("-" + shortform),
      msg(msg),
      callback(callback)
{ option_type = CALLBACK; }

option_t::option_t(std::string longform, std::string shortform, std::string msg, std::string printval)
    : longform("--" + longform),
      shortform("-" + shortform),
      msg(msg),
      printval(printval)
{ option_type = STRING; }


/*
 * Option List
 */

option_list::option_list(std::string program_name) :
program_name(program_name)
{
  { internal_list = std::vector<option_t>(); }
}

void option_list::add(option_t opt)
{ internal_list.push_back(opt); }

void option_list::parse(int argc, char **argv)
{
  for (int arg_number = 0; arg_number < argc; ++arg_number) {
    for (std::vector<option_t>::iterator this_option = internal_list.begin();
         this_option != internal_list.end();
         this_option++) {
      if (this_option->longform == std::string(argv[arg_number]) ||
          this_option->shortform == std::string(argv[arg_number])) {
        switch (this_option->option_type) {
          case CALLBACK:
            this_option->callback();
            break;
          case STRING:
            std::cout << this_option->printval << std::endl;
            break;
        }
      }

    }
    if (std::string("--help") == std::string(argv[arg_number]) ||
        std::string("-h") == std::string(argv[arg_number])) {
      help();
    }
  }
}

void option_list::help()
{
  std::cout << program_name << std::endl;
  std::cout << "  -h [ --help ] \t\tdisplay this help message" << std::endl;
  for (std::vector<option_t>::iterator this_option = internal_list.begin();
       this_option != internal_list.end();
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
