//
// Created by nathan on 2/1/18.
//

#ifndef VOLK_VOLK_OPTION_HELPERS_H
#define VOLK_VOLK_OPTION_HELPERS_H

#include <string>
#include <cstring>
#include <limits.h>
#include <vector>
#include <map>

typedef enum
{
  VOID_CALLBACK,
    INT_CALLBACK,
    BOOL_CALLBACK,
    STRING_CALLBACK,
    FLOAT_CALLBACK,
  STRING,
} VOLK_OPTYPE;

class option_t {
  public:
  option_t(std::string longform, std::string shortform, std::string msg, void (*callback)());
    option_t(std::string longform, std::string shortform, std::string msg, void (*callback)(int));
    option_t(std::string longform, std::string shortform, std::string msg, void (*callback)(float));
    option_t(std::string longform, std::string shortform, std::string msg, void (*callback)(bool));
    option_t(std::string longform, std::string shortform, std::string msg, void (*callback)(std::string));
  option_t(std::string longform, std::string shortform, std::string msg, std::string printval);

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

  void parse(int argc, char **argv);

  void help();
  private:
  std::string program_name;
  std::vector<option_t> internal_list;
  std::map<std::string, int> present_options;
};


#endif //VOLK_VOLK_OPTION_HELPERS_H
