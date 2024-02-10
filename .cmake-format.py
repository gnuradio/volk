# Copyright 2021 Marcus Müller, 2024 Johannes Demel
# SPDX-License-Identifier: LGPL-3.0-or-later

class _clang_format_options:
    def __init__(self, clangfile=None):
        if not clangfile:
            clangfile = ".clang-format"
        self.lines = []
        with open(clangfile, encoding="utf-8") as opened:
            for line in opened:
                if line.strip().startswith("#"):
                    continue
                self.lines.append(line.rstrip().split(":"))

    def __getitem__(self, string):
        path = string.split(".")
        value = None
        for crumble in path:
            for line in self.lines:
                if line[0].strip() == crumble:
                    if len(line) > 1:
                        value = line[1].strip().rstrip()
                    break
        return value


_clang_format = _clang_format_options()

# ----------------------------------
# Options affecting listfile parsing
# ----------------------------------
with section("parse"):
    additional_commands = {
        'gr_python_install': {
            'flags': [],
            'kwargs': {
                "PROGRAMS": "*",
                "FILES": "*",
                "DESTINATION": "*"
            }
        },
    }

with section("markup"):
    first_comment_is_literal = True
    enable_markup = False

with section("format"):
    # Disable formatting entirely, making cmake-format a no-op
    disable = False

    # How wide to allow formatted cmake files
    line_width = int(_clang_format["ColumnLimit"])

    # How many spaces to tab for indent
    tab_size = int(_clang_format["IndentWidth"])

    # If true, lines are indented using tab characters (utf-8 0x09) instead of
    # <tab_size> space characters (utf-8 0x20). In cases where the layout would
    # require a fractional tab character, the behavior of the  fractional
    # indentation is governed by <fractional_tab_policy>
    use_tabchars = _clang_format["UseTab"] in ("ForIndentation",
                                               "ForContinuationAndIndentation",
                                               "Always")

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = False

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = False

    # If the statement spelling length (including space and parenthesis) is
    # smaller than this amount, then force reject nested layouts.
    min_prefix_chars = tab_size

    # If the statement spelling length (including space and parenthesis) is larger
    # than the tab width by more than this amount, then force reject un-nested
    # layouts.
    max_prefix_chars = 3 * tab_size

    # What style line endings to use in the output.
    line_ending = "unix"

    # Format command names consistently as 'lower' or 'upper' case
    command_case = "canonical"

    # Format keywords consistently as 'lower' or 'upper' case
    keyword_case = "upper"


with section("lint"):
    max_arguments = 6
    max_localvars = 20
    max_statements = 75
