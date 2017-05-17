# Define the encoding, for Python 2 compatibility:
# This Python file uses the following encoding: utf-8

# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see http://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



"""
This is the .pyx preprocessor script.
It can be run with the following three sets of arguments,
all arguments being filenames:
- module.py commons.py
  Creates module.pyx, a version of module.py with cython-legal and
  optimized syntax.
- .types.pyx commons.py .types.pyx module0.pyx module1.pyx ...
  Creates .types.pyx, a file containing import for all extension classes
  from module0.pyx, module1.pyx, ..., together with globally defined
  types.
- module.pyx commons.py .types.pyx
  Created module.pxd, the cython header for module.pyx.

In the first case where a .pyx file is created from a .py file,
the following changes happens to the source code (in the .pyx file):
- Insert the line 'cimport cython' at the very top,
  though below any __future__ imports.
- Transform statements written over multiple lines into single lines.
  The exception is decorator statements, which remain multilined.
- Removes pure Python commands between 'if not cython.compiled:' and
  'else:', including these lines themselves. Also removes the triple
  quotes around the Cython statements in the else body. The 'else'
  clause is optional.
- Insert the line 'from commons cimport *'
  just below 'from commons import *'.
- Transform the 'cimport()' function calls into proper cimports.
- Replace 'ℝ[expression]' with a double variable and 'ℤ[expression]'
  with a Py_ssize_t variable which is equal to 'expression' and defined
  on a suitable line.
- Replaces the cython.header and cython.pheader decorators with
  all of the Cython decorators which improves performance. The
  difference between the two is that cython.header turns into
  cython.cfunc and cython.inline (among others), while cython.pheader
  turns into cython.ccall (among others).
- Integer powers will be replaced by products.
- Calls to build_struct will be replaced with specialized C structs
  which are declared dynamically from the call. Type declarations
  of this struct, its fields and its corresponding dict are inserted.
- Unicode non-ASCII letters will be replaced with ASCII-strings.
- __init__ methods in cclasses are renamed to __cinit__.
- Replace (with '0') or remove ':' and '...' intelligently, when taking
  the address of arrays.
- Replace alloc, realloc and free with the corresponding PyMem_
  functions and take care of the casting from the void* to the
  appropriate pointer type.
- Replaced the cast() function with actual Cython syntax, e.g. 
  <double[::1]>.
- Loop unswitching is performed on if statements under an unswitch
  context manager, which are indented under one or more loops.
- A comment will be added to the end of the file, listing all the
  implemented extension types within the file.

This script is not written very elegantly, and do not leave
the modified code in a very clean state either. Sorry...
"""



# For Python 2.x compatibility
from __future__ import nested_scopes, generators, division
from __future__ import absolute_import, with_statement, print_function, unicode_literals
import sys
if sys.version_info.major < 3:
    from codecs import open
def non_nested_exec(s):
    exec(s)
# General imports
from copy import deepcopy
import collections, imp, itertools, os, re, shutil, unicodedata
# For math
import numpy as np



def cimport_cython(lines):
    for i, line in enumerate(lines):
        if (line.strip()
            and not line.lstrip().startswith('#')
            and not '__future__' in line
            ):
            lines = lines[:i] + ['cimport cython\n'] + lines[i:]
            break
    return lines



def oneline(lines):
    in_quotes = [False, False]
    in_triple_quotes = [False, False]
    paren_counts = {'paren': 0, 'brack': 0, 'curly': 0}
    def count_parens(line):
        if line.lstrip().startswith('#'):
            return line
        L = len(line)
        for j, ch in enumerate(line):
            # Inside quotations?
            if ch in ("'", '"'):
                if ch == "'" and not in_quotes[1]:
                    in_quotes[0] = not in_quotes[0]
                elif ch == '"' and not in_quotes[0]:
                    in_quotes[1] = not in_quotes[1]
                if j >= 2 and line[(j-2):(j + 1)] == "'''":
                    in_quotes[0] = not in_quotes[0]
                    in_triple_quotes[0] = not in_triple_quotes[0]
                    if not in_triple_quotes[0]:
                        in_quotes[0] = in_quotes[1] = False
                elif j >= 2 and line[(j-2):(j + 1)] == '"""':
                    in_quotes[1] = not in_quotes[1]
                    in_triple_quotes[1] = not in_triple_quotes[1]
                    if not in_triple_quotes[1]:
                        in_quotes[0] = in_quotes[1] = False
            # Break at # and remove inline comment
            if ch == '#' and not in_quotes[0] and not in_quotes[1]:
                line = line[:j].rstrip() + '\n'
                break
            # Count parentheses outside quotes
            if (    not in_quotes[0]
                and not in_quotes[1]
                and not in_triple_quotes[0]
                and not in_triple_quotes[1]
                ):
                if ch == '(':
                    paren_counts['paren'] += 1
                elif ch == ')':
                    paren_counts['paren'] -= 1
                elif ch == '[':
                    paren_counts['brack'] += 1
                elif ch == ']':
                    paren_counts['brack'] -= 1
                elif ch == '{':
                    paren_counts['curly'] += 1
                elif ch == '}':
                    paren_counts['curly'] -= 1
        return line
    new_lines = []
    multiline_statement = []
    multiline = False
    multiline_decorator = False
    for i, line in enumerate(lines):
        line = count_parens(line)
        if (paren_counts['paren'] > 0 or
            paren_counts['brack'] > 0 or
            paren_counts['curly'] > 0) and not multiline:
            # Multiline statement begins
            multiline = True
            if line.lstrip().startswith('@'):
                multiline_decorator = True
                new_lines.append(line)
                continue
            if line.lstrip().startswith('#'):
                # Delete full-line comment
                # within multiline statement.
                line = ''
            if line:
                multiline_statement.append(line.rstrip())
        elif (paren_counts['paren'] > 0 or
              paren_counts['brack'] > 0 or
              paren_counts['curly'] > 0) and multiline:
            # Multiline statement continues
            if multiline_decorator:
                new_lines.append(line)
                continue
            if line.lstrip().startswith('#'):
                # Delete full-line comment
                # within multiline statement.
                line = ''
            if line:
                multiline_statement.append(' ' + line.strip())
        elif multiline:
            # Multiline statement ends
            multiline = False
            if multiline_decorator:
                multiline_decorator = False
                new_lines.append(line)
                continue
            multiline_statement.append(' ' + line.lstrip())
            new_lines.append(''.join(multiline_statement))
            multiline_statement = []
        else:
            new_lines.append(line)
    return new_lines



def cythonstring2code(lines):
    new_lines = []
    in_purePythonsection = False
    unindent = False
    purePythonsection_start = 0
    indentation = 0
    for i, line in enumerate(lines):
        if (unindent and line.rstrip() != ''
                     and (len(line) > indentation
                     and line[indentation] != ' ')):
            unindent = False
        if line.lstrip().startswith('if not cython.compiled:'):
            indentation = len(line) - len(line.lstrip())
            in_purePythonsection = True
            purePythonsection_start = i
        if not in_purePythonsection:
            if unindent:
                line_without_triple_quotes = line
                if (   line.startswith(' '*(indentation + 4) + '"""') 
                    or line.startswith(' '*(indentation + 4) + "'''")):
                    line_without_triple_quotes = line.replace('"""', '').replace("'''", '')
                if len(line_without_triple_quotes) > 4:
                    new_lines.append(line_without_triple_quotes[4:])
            else:
                new_lines.append(line)
        if (i != purePythonsection_start and in_purePythonsection
                                         and len(line) >= indentation
                                         and line[indentation] != ' '
                                         and line.strip()):
            in_purePythonsection = False
            if 'else:' in line:
                unindent = True
            else:
                new_lines.append(line)
                unindent = False
    return new_lines



def cython_structs(lines):
    # Function which returns a copy of the build_struct function
    # from commons.py:
    def get_build_struct():
        build_struct_code = []
        with open('{}.py'.format(commons_name), 'r', encoding='utf-8') as commonsfile:
            indentation = -1
            for line in commonsfile:
                if line.lstrip().startswith('def build_struct('):
                    # Start of build_struct found
                    indentation = len(line) - len(line.lstrip())
                    build_struct_code.append(line[indentation:])
                    continue
                if indentation != -1:
                    if (    not line.lstrip().startswith('#')
                        and len(line) - len(line.lstrip()) <= indentation
                        and line.strip()):
                        # End of build_struct found
                        break
                    if len(line) - len(line.lstrip()) > indentation:
                        build_struct_code.append(line[indentation:])
        for line in reversed(build_struct_code):
            if not line.strip() or line.lstrip().startswith('#'):
                build_struct_code.pop()
            else:
                break
        return build_struct_code
    # Search the file for calls to build_struct
    build_struct_code = []
    new_lines = []
    struct_kinds = []
    for line in lines:
        if (    'build_struct(' in line
            and '=build_struct(' in line.replace(' ', '')
            and not line.lstrip().startswith('#')
            ):
            # Call found.
            # Get assigned names.
            varnames = line[:line.index('=')].replace(' ', '').split(',')
            # Get field names, types and values. These are stored
            # as triples og strings in struct_content.
            struct_content = line[(line.index('(') + 1):line.rindex(')')].split('=')
            for i, part in enumerate(struct_content[:-1]):
                # The field name
                if i == 0:
                    name = part
                else:
                    name = part[(part.rindex(',') + 1):].strip()
                decl = struct_content[i + 1][:struct_content[i + 1].rindex(',')]
                if re.search('\(.*,', decl.replace(' ', '')):
                    # Both type and value given.
                    # Find type.
                    ctype_start = len(decl)
                    if "'" in decl:
                        ctype_start = decl.index("'")
                    if '"' in decl and decl.index('"') < ctype_start:
                        ctype_start = decl.index('"')
                    quote_type = decl[ctype_start]
                    ctype = ''
                    for j, c in enumerate(decl[(ctype_start + 1):]):
                        if c == quote_type:
                            break
                        ctype += c
                    # Find value
                    value = decl[(ctype_start + 1 + j + 1):].strip()
                    if value[0] == ',':
                        value = value[1:]
                    if value[-1] == ')':
                        value = value[:-1]
                    value = value
                else:
                    # Only type given. Initialize to zero.
                    if decl.count('"') == 2:
                        ctype = re.search('(".*")', decl).group(1)
                    if decl.count("'") == 2:
                        ctype = re.search("('.*')", decl).group(1)
                    ctype = ctype.replace('"', '').replace("'", '').strip()
                    value = '0'
                struct_content[i] = (name.strip(), ctype.strip(), value.strip())
            struct_content.pop()
            # The name of the struct type is eg. struct_double_double_int
            struct_kind = '_'.join([t[1] for t in struct_content])
            # Insert modified version of the build_struct function,
            # initializing all values to zero.
            if not build_struct_code:
                build_struct_code = get_build_struct()
            for build_struct_line in build_struct_code:
                build_struct_line = build_struct_line.replace('build_struct(',
                                                            'build_struct_{}('.format(struct_kind))
                build_struct_line = build_struct_line.replace('...',
                                                              'struct_{}({})'.format(struct_kind,
                                                             ', '.join(['0']*len(struct_content))))
                
                new_lines.append(build_struct_line)
            # Insert declaration of struct
            indentation = len(line) - len(line.lstrip())
            new_lines.append(' '*indentation + "cython.declare({}='struct_{}')\n"
                                               .format(varnames[0], struct_kind))
            # Insert declaration of dict
            if len(varnames) == 2:
                new_lines.append(' '*indentation + "cython.declare({}='dict')\n"
                                                   .format(varnames[1]))
            # Insert modified build_struct call
            new_lines.append(line.replace('build_struct(',
                                          'build_struct_{}('.format(struct_kind)))
            # Set values
            for name, ctype, value in struct_content:
                if value != '0':
                    new_lines.append("{}{}.{} = {}['{}']\n".format(' '*indentation,
                                                                   varnames[0], 
                                                                   name,
                                                                   varnames[1],
                                                                   name)
                                     )
            # Insert pxd declaration of the struct
            if struct_kind not in struct_kinds:
                struct_kinds.append(struct_kind)
                new_lines.append(' '*indentation + 'pxd = """\n')
                new_lines.append('{}ctypedef struct struct_{}:\n'.format(' '*indentation,
                                                                         struct_kind))
                for name, ctype, val in struct_content:
                    new_lines.append('{}    {} {}\n'.format(' '*indentation,
                                                           ctype.replace('"', '').replace("'", ''),
                                                            name))
                new_lines.append(' '*indentation + '"""\n')
        else:
            # No call found in this line
            new_lines.append(line)
    return new_lines



def cimport_commons(lines):
    for i, line in enumerate(lines):
        if line.startswith('from {} import *'.format(commons_name)):
            lines = (  lines[:(i + 1)]
                     + ['from {} cimport *\n'.format(commons_name)]
                     + lines[(i + 1):])
            break
    return lines



def cimport_function(lines):
    new_lines = []
    for i, line in enumerate(lines):
        if line.replace(' ', '').startswith('cimport('):
            line = re.sub('cimport.*\((.*?)\)',
                          lambda match: eval(match.group(1)).replace('import ', 'cimport '),
                          line).rstrip()
            if line.endswith(','):
                line = line[:-1]
            line = '{}\n'.format(line)
            if line.lstrip().startswith('from ') and ' as ' not in line:
                # Add normal imports enclosed in try/except
                indentation = len(line) - len(line.lstrip())
                words = line.strip(' \n').split(' ')
                module = words[1]
                functions = [function.strip(' ,') for function in words[3:]
                             if function.strip(' ,')]
                for function in functions:
                    new_lines.append(' '*indentation + 'try:\n')
                    new_lines.append(' '*(indentation + 4)
                                     + 'from {} import {}\n'.format(module, function))
                    new_lines.append(' '*indentation + 'except:\n')
                    new_lines.append(' '*(indentation + 4) + '...\n')
            # Add cimport import
            new_lines.append(line)
        else:
            new_lines.append(line)
    return new_lines



def loop_unswitching(lines):
    # Run through the lines and replace any unswitch context manager
    # with the unswitched loop(s).
    new_lines = []
    while True:
        skip = 0
        for i, line in enumerate(lines):
            # Should this line be skipped?
            if skip > 0:
                skip -= 1
                continue
            # Search for the following constructs:
            # with unswitch:
            #    ...
            # with unswitch():  # Same as above
            #    ...
            # with unswitch(n):  # With n an integer literal or expression
            #    ...
            match = re.search('with +unswitch *(\(.*\))? *:', line.strip())
            if match and not line.lstrip().startswith('#'):
                # Loop unswitching found.
                # Determine the number of indentation levels to do
                # loop unswitching on. This is the n in unswitch(n).
                # If not specified, this is set to infinity.
                n = match.group(1)
                if n:
                    n = n[1:-1]
                    if n:
                        n = eval(n)
                    else:
                        n = float('inf')
                else:
                    n = float('inf')
                # If n is zero or negative,
                # no loop unswitching shold be performed.
                if n < 1:
                    continue
                # Search n nested loops upwards.
                # Allowed constructs to pass are
                # - for loops
                # - while loops
                # - if/elif/else statements
                # - with statements
                # - try/except/finally statements
                unswitch_lvl = (len(line) - len(line.lstrip()))//4
                n_nested_loops = 0
                smallest_lvl = unswitch_lvl
                for j, line in enumerate(reversed(lines[:i])):
                    # Skip empty lines and comment lines
                    line_stripped = line.strip()
                    if not line_stripped or line_stripped.startswith('#'):
                        continue
                    # The indentation level
                    lvl = (len(line) - len(line.lstrip()))//4
                    # Check for nested loop
                    if (   re.search('for .*:'  , line_stripped)
                        or re.search('while .*:', line_stripped)
                        ) and lvl < smallest_lvl:
                        J = j
                        n_nested_loops += 1
                        if n_nested_loops == n:
                            break
                    # Record smallest indentation level
                    if lvl < smallest_lvl:
                        smallest_lvl = lvl
                    # Break out of the search when encountering a special
                    # block header (such as a def statement).
                    if line_stripped.endswith(':'):
                        if not (   re.search('for .*:'      , line_stripped)
                                or re.search('while .*:'    , line_stripped)
                                or re.search('if .*:'       , line_stripped)
                                or re.search('elif .*:'     , line_stripped)
                                or re.search('else *:'    , line_stripped)
                                or re.search('with .*:'     , line_stripped)
                                or re.search('try ?.*:'     , line_stripped)
                                or re.search('except ?.*:'  , line_stripped)
                                or re.search('finally *:' , line_stripped)
                                ):
                            break
                # The loop lines to copy
                loop_lines = lines[(i - J - 1):i]
                # Now search downwards to find the if statements indented
                # under the unswitch context manager.
                outer_loop_lvl = (len(loop_lines[0]) - len(loop_lines[0].lstrip()))//4
                patterns = ['{}if .*:'  .format('    '*(unswitch_lvl + 1)),
                            '{}elif .*:'.format('    '*(unswitch_lvl + 1)),
                            '{}else *:' .format('    '*(unswitch_lvl + 1)),
                            ]
                if_headers = []
                if_bodies = []
                loop_body = []
                under_unswitch = True
                for j, line in enumerate(lines[(i + 1):]):
                    # Skip empty lines and comment lines
                    line_stripped = line.strip()
                    if not line_stripped or line_stripped.startswith('#'):
                        continue
                    # The indentation level
                    lvl = (len(line) - len(line.lstrip()))//4
                    # Determine whether or not we are out of the
                    # unswitch context manager.
                    if lvl <= unswitch_lvl:
                        under_unswitch = False
                    # Break out when all nested loops have ended
                    if lvl <= outer_loop_lvl:
                        break
                    loop_body_end_nr = i + 1 + j
                    # Record if/elif/else
                    if under_unswitch:
                        line_rstripped = line.rstrip()
                        if lvl == unswitch_lvl + 1 and any(re.search(pattern, line_rstripped)
                                                           for pattern in patterns):
                            if_headers.append(line)
                            if_bodies.append([])
                            if_body = if_bodies[-1]
                            continue
                    # Record code indented under if/elif/else
                    if under_unswitch:
                        if_body.append(line)
                        continue
                    # Record code also indented under the nested loops,
                    # but not part of the unswitching.
                    loop_body.append(line)
                # If no else block is present, insert a dummy
                # else block. This is needed because the unswitched loop
                # should be run even if the if statement within it
                # is false.
                if not if_headers[-1].lstrip().startswith('else'):
                    # Only insert the else block if it is going to
                    # contain some code.
                    empty_else = True
                    for loop_body_line in loop_body:
                        loop_body_line = loop_body_line.lstrip()
                        if loop_body_line and not loop_body_line.startswith('#'):
                            empty_else = False
                            break
                    else:
                        if len(loop_lines) > 1:
                            for loop_line in loop_lines[1:]:
                                loop_line = loop_line.lstrip()
                                if loop_line and not loop_line.startswith('#'):
                                    empty_else = False
                                    break
                    if not empty_else:
                        indentation = '    '*(unswitch_lvl + 1)
                        if_headers.append('{}else:  # This is an autoinserted else block\n'
                                          .format(indentation))
                        if_bodies.append(['{}    # End of autoinserted else block\n'
                                          .format(indentation)])
                # Stitch together the recorded pieces to perform
                # the loop unswitching.
                lines_unswitched = []
                for if_header, if_body in zip(if_headers, if_bodies):
                    # Unswitched if/elif/else statement
                    indentation = ' '*(len(loop_lines[0]) - len(loop_lines[0].lstrip()))
                    lines_unswitched.append(indentation + if_header.lstrip())
                    # Nested loops
                    lines_unswitched += ['    ' + line for line in loop_lines]
                    # Body of unswitched if/elif/else statement
                    lines_unswitched += [line[4:] for line in if_body]
                    # Remaining loop body
                    lines_unswitched += ['    ' + line for line in loop_body]
                # Pop the nested for loops from new_lines, as these should
                # only be included as the loop unswitching.
                new_lines = new_lines[:-len(loop_lines)]
                # Append the unswitched loop lines
                new_lines += lines_unswitched
                # Flag the line loop to skip the following lines which
                # are already included in the loop unswitching lines.
                skip = loop_body_end_nr  - i
            else:
                # No loop unswitching on this line
                new_lines.append(line)
        # Done with all lines. Run them through again, in case of
        # nested use of the unswitching context manager.
        # Break out when no change has happened to any line.
        if len(lines) == len(new_lines):
            break
        lines = new_lines
        new_lines = []
    return lines



def constant_expressions(lines):
    sets = {'ℝ': 'double',
            'ℤ': 'Py_ssize_t',
            }
    def indentation_level(line):
        line_lstrip = line.lstrip()
        if not line_lstrip or line_lstrip.startswith('#'):
            return -1
        return len(line) - len(line_lstrip)
    def variable_changed(var, line):
        line_ori = line
        line = line.replace(' ', '')
        if line.startswith('#'):
            return False
        def multi_assign_in_line(var, line):
            match = re.search(r' {}( *[,=] *[_a-zA-Z][_a-zA-Z0-9]*)+'.format(var),
                              ' {} '.format(line))
            if not match:
                return False
            return ('=' in match.group())
        return (   re.search('for +{} +in '.format(var), line_ori)
                or multi_assign_in_line(var, line_ori)
                or ('=' + var + '=') in line
                or line.startswith(var + '='  )
                or line.startswith(var + '+=' )
                or line.startswith(var + '-=' )
                or line.startswith(var + '*=' )
                or line.startswith(var + '/=' )
                or line.startswith(var + '**=')
                or line.startswith(var + '^=' )
                or line.startswith(var + '&=' )
                or line.startswith(var + '|=' )
                or line.startswith(var + '@=' )
                or (',' + var + '='  ) in line
                or (',' + var + '+=' ) in line
                or (',' + var + '-=' ) in line
                or (',' + var + '*=' ) in line
                or (',' + var + '/=' ) in line
                or (',' + var + '**=') in line
                or (',' + var + '^=' ) in line
                or (',' + var + '&=' ) in line
                or (',' + var + '|=' ) in line
                or (',' + var + '@=' ) in line
                or (';' + var + '='  ) in line
                or (';' + var + '+=' ) in line
                or (';' + var + '-=' ) in line
                or (';' + var + '*=' ) in line
                or (';' + var + '/=' ) in line
                or (';' + var + '**=') in line
                or (';' + var + '^=' ) in line
                or (';' + var + '&=' ) in line
                or (';' + var + '|=' ) in line
                or (';' + var + '@=' ) in line
                )
    for blackboard_bold_symbol, ctype in sets.items():
        # Find constant expressions using the
        # ℝ[expression] or ℤ[expression] syntax.
        expressions = []
        expressions_cython = []
        declaration_linenrs = []
        declaration_placements = []
        operators = collections.OrderedDict([('.',  'DOT' ),
                                             ('+',  'PLS' ),
                                             ('-',  'MIN' ),
                                             ('**', 'POW' ),
                                             ('*',  'TIM' ),
                                             ('/',  'DIV' ),
                                             ('\\', 'BSL' ),
                                             ('^',  'CAR' ),
                                             ('&',  'AND' ),
                                             ('|',  'BAR' ),
                                             ('@',  'AT'  ),
                                             (',',  'COM' ),
                                             ('(',  'OPAR'),
                                             (')',  'CPAR'),
                                             ('[',  'OBRA'),
                                             (']',  'CBRA'),
                                             ('{',  'OCUR'),
                                             ('}',  'CCUR'),
                                             ("'",  'QTE' ),
                                             ('"',  'DQTE'),
                                             (':',  'COL' ),
                                             (';',  'SCOL'),
                                             ('!',  'BAN' ),
                                             ('#',  'SHA' ),
                                             ('$',  'DOL' ),
                                             ('%',  'PER' ),
                                             ('?',  'QUE' ),
                                             ('<',  'LTH' ),
                                             ('>',  'GTH' ),
                                             ('`',  'GRA' ),
                                             ('~',  'TIL' ),
                                             ])
        while True:
            no_blackboard_bold_R = True
            module_scope = True
            function_scope_indentation_level = 0
            for i, line in enumerate(lines):
                line = line.rstrip('\n')
                if line.lstrip().startswith('def '):
                    module_scope = False
                    function_scope_indentation_level = 4 + len(line) - len(line.lstrip())
                elif len(line) > 0 and line[0] not in ' #':
                    module_scope = True
                    function_scope_indentation_level = 0
                search = re.search(blackboard_bold_symbol + '\[.+\]', line)
                if not search or line.replace(' ', '').startswith('#'):
                    continue
                # Blackboard bold symbol found on this line
                R_statement_fullmatch = search.group(0)
                R_statement = R_statement_fullmatch[:2]
                for c in R_statement_fullmatch[2:]:
                    R_statement += c
                    if R_statement.count('[') == R_statement.count(']'):
                        break
                expression = R_statement[2:-1].strip()
                # Integer literals to double literals when dividing
                if ctype == 'double':
                    # Integer in numerator
                    expression = re.sub('[0-9]+\.?[0-9]*/[^/]',
                                        lambda numerator: (
                                                 numerator.group() if '.' in numerator.group()
                                                                   else (numerator.group()[:-2]
                                                                         + '.0/'
                                                                         + numerator.group()[-1])),
                                        expression)
                    # Integer in denominator
                    expression = re.sub('[^/]/[0-9]+\.?[0-9]*',
                                        lambda denominator: (
                                                  denominator.group() if '.' in denominator.group()
                                                                  else denominator.group() + '.0'),
                                        expression)
                no_blackboard_bold_R = False
                expressions.append(expression)
                expression_cython = blackboard_bold_symbol + '_' + expression.replace(' ', '')
                for op, op_name in operators.items():
                    expression_cython = expression_cython.replace(op, '__{}__'.format(op_name))
                expressions_cython.append(expression_cython)
                lines[i] = '{}\n'.format(line.replace(R_statement, expression_cython))
                # Find out where the declaration should be
                variables = expression.replace(' ', '')
                variables = re.sub('(?P<quote>[\'"]).*?(?P=quote)', # Remove string literals using
                                   '', variables)                   # single and double quotes.
                variables = [variables]
                for op in operators.keys():
                    # Split variables on operators, excluding '.'
                    # (attributes are handled afterwards).
                    if op != '.':
                        variables = list(itertools.chain(*[var.split(op) for var in variables]))
                for v, var in enumerate(variables):
                    # Variable attributes are not consideres variables
                    if '.' in var:
                        variables[v] = var[:var.index('.')]
                variables = [var for var in list(set(variables))
                             if var and var[0] not in '.0123456789']
                linenr_where_defined = [-1]*len(variables)
                placements = ['below']*len(variables)
                for w, end in enumerate((i + 1, len(lines))):  # Second time: Check module scope
                    if w == 1 and module_scope:
                        break
                    for v, var in enumerate(variables):
                        if linenr_where_defined[v] != -1:
                            continue
                        for j, line2 in enumerate(reversed(lines[:end])):
                            line2 = line2.rstrip('\n')
                            line2_ori = line2
                            line2 = (' '*(len(line2) - len(line2.lstrip()))
                                     + line2.replace(' ', ''))
                            if (    line2_ori.lstrip().startswith('def ')
                                and re.search('[\(,]{}[,=\)]'.format(var), line2)):
                                # var as function argument
                                linenr_where_defined[v] = end - 1 - j
                                break
                            else:
                                if variable_changed(var, line2_ori):
                                    # var declaration found
                                    linenr_where_defined[v] = end - 1 - j
                                    # Continue searching for var in previous lines
                                    linenr_where_defined_first = -1
                                    for k, line3 in enumerate(reversed(lines[:linenr_where_defined[v]])):
                                        line3 = line3.rstrip('\n')
                                        line3_ori = line3
                                        line3 = ' '*(len(line3) - len(line3.lstrip()))  + line3.replace(' ', '')
                                        if line3_ori.lstrip().startswith('def '):
                                            # Function definition reached
                                            break
                                        if indentation_level(line3_ori) == function_scope_indentation_level:
                                            # Upper level of function reached.
                                            # Definitions above this point does not matter.
                                            break
                                        if variable_changed(var, line3_ori):
                                            # Additional var declaration found
                                            linenr_where_defined_first = linenr_where_defined[v] - 1 - k
                                    # Locate "barriers" between linenr_where_defined and linenr_where_defined_first.
                                    # A "barrier" consists of code with an indentation level smaller
                                    # than that at linenr_where_defined, located somewhere between
                                    # linenr_where_defined and linenr_where_defined_first.
                                    # If such barriers exist, the defintion to be inserted cannot just
                                    # be placed right below linenr_where_defined, but should be
                                    # moved closer down towards where it is used and placed at an
                                    # indentation level equal to that of the barrier with the smallest
                                    # indentation level.
                                    if linenr_where_defined_first != -1:
                                        indentationlvl_where_defined = indentation_level(lines[linenr_where_defined[v]])
                                        indentationlvl_barrier = indentationlvl_where_defined
                                        for line3 in lines[(linenr_where_defined_first + 1):linenr_where_defined[v]]:
                                            indentationlvl = indentation_level(line3)
                                            if -1 < indentationlvl < indentationlvl_barrier:
                                                indentationlvl_barrier = indentationlvl
                                        if indentationlvl_barrier < indentationlvl_where_defined:
                                            # A barrier exists!
                                            # Search downwards from linenr_where_defined[v] to i
                                            # (the line where ℝ[...] is used) until an
                                            # indentation level <= indentationlvl_barrier is found.
                                            # This should be the place where the new declaration should be placed.
                                            for k, line3 in enumerate(lines[(linenr_where_defined[v] + 1):(i + 1)]):
                                                indentationlvl = indentation_level(line3)
                                                if -1 < indentationlvl <= indentationlvl_barrier:
                                                    # Change linenr_where_defined[v] to be this linenr,
                                                    # where the barrier does not matter.
                                                    linenr_where_defined[v] = linenr_where_defined[v] + 1 + k
                                                    # Note that the declaration should now be inserted above
                                                    # this line!
                                                    placements[v] = 'above'
                                                    break
                                    # Break because original var declaration is found
                                    break
                if linenr_where_defined:
                    index_max = np.argmax(linenr_where_defined)
                    declaration_linenrs.append(linenr_where_defined[index_max])
                    declaration_placements.append(placements[index_max])
                else:
                    declaration_linenrs.append(-1)
                    declaration_placements.append('below')
                # If inside a function and declaration_linenrs[-1] == -1,
                # the variable may be declared in the module scope.
                # Remove again if duplicate
                for j in range(len(expressions) - 1):
                    if (expressions[j] == expressions[-1]
                        and declaration_linenrs[j] == declaration_linenrs[-1]):
                        expressions.pop()
                        expressions_cython.pop()
                        declaration_linenrs.pop()
                        declaration_placements.pop()
                        break
            if no_blackboard_bold_R:
                break
        # Find out where the last import statement is. Unrecognized
        # definitions should occur below this line.
        linenr_unrecognized = -1
        for i, line in enumerate(lines):
            if 'import ' in line:
                if '#' in line and line.index('#') < line.index('import '):
                    continue
                if line.index('import ') != 0 and line[line.index('import ') - 1] not in 'c ':
                    continue
                if i + 1 < len(lines) and ('"""' in lines[i + 1] or "'''" in lines[i + 1]):
                    linenr_unrecognized = i + 1
                    continue
                # Go down until indentation level 0 is reached
                for j, line in enumerate(lines[(i + 1):]):
                    if (len(line) > 0
                        and line[0] not in '# '
                        and not line.startswith('"""')
                        and not line.startswith("'''")):
                        linenr_unrecognized = i + j
                        break
        # Insert Cython declarations of constant expressions
        new_lines = []
        fname = None
        declarations_placed = collections.defaultdict(list)
        for i, line in enumerate(lines):
            # Append original line
            new_lines.append(line)
            # Detect whether we are inside a function or not
            if line.lstrip().startswith('def '):
                fname = line[4:line.index('(')].lstrip()
            elif line.strip() and line[0] not in (' ', '#'):
                fname = None
            # Unrecognized definitions
            if i == linenr_unrecognized:
                for e, expression_cython in enumerate(expressions_cython):
                    if declaration_linenrs[e] == -1:
                        if not expression_cython in declarations_placed[fname]:
                            new_lines.append('cython.declare(' + expression_cython + "='{}')\n".format(ctype))
                            if fname:
                                # Remember that this variable has been declared in this function
                                declarations_placed[fname].append(expression_cython)
                        new_lines.append(expression_cython + ' = <{}>('.format(ctype)
                                                           + expressions[e]
                                                           + ')\n'
                                         )
                new_lines.append('\n')
            for e, n in enumerate(declaration_linenrs):
                if i == n:
                    indentation = ' '*(len(line) - len(line.lstrip()))
                    if declaration_placements[e] == 'below' and line.rstrip().endswith(':'):
                        indentation += ' '*4
                    if declaration_placements[e] == 'above':
                        new_lines.pop()
                    if not expressions_cython[e] in declarations_placed[fname]:
                        new_lines.append(indentation + 'cython.declare('
                                                     + expressions_cython[e]
                                                     + "='{}')\n".format(ctype)
                                         )
                        if fname:
                            # Remember that this variable has been declared in this function
                            declarations_placed[fname].append(expressions_cython[e])
                    new_lines.append(indentation + expressions_cython[e]
                                                 + ' = <{}>('.format(ctype)
                                                 + expressions[e]
                                                 + ')\n'
                                     )
                    if declaration_placements[e] == 'above':
                        new_lines.append(line)
        # Exchange the original lines with the modified lines
        lines = new_lines
    return lines



def remove_duplicate_declarations(lines):
    new_lines = []
    in_function = False
    for line in lines:
        if line.startswith('def '):
            in_function = True
            declarations = {}
        elif line and line[0] not in ' #\n':
            in_function = False
        if not in_function:
            new_lines.append(line)
            continue
        if line.lstrip().startswith('cython.declare('):
            indentation = ' '*(len(line) - len(line.lstrip()))
            info = re.search('cython\.declare\((.*)\)', line).group(1).split('=')
            for i, (varname, vartype) in enumerate(zip(info[:-1], info[1:])):
                if i > 0:
                    index = varname.rfind(',')
                    varname = varname[(index + 1):]
                if i < len(info) - 2:
                    index = vartype.rfind(',')
                    vartype = vartype[:index]
                varname = varname.strip(' ,')
                vartype = vartype.strip(' ,')
                vartype_prev = declarations.get(varname)
                if vartype_prev:
                    vartype_bare, vartype_prev_bare = vartype, vartype_prev
                    for char in ' ,"\'':
                        vartype_bare      = vartype_bare     .replace(char, '')
                        vartype_prev_bare = vartype_prev_bare.replace(char, '')
                    if vartype_bare != vartype_prev_bare:
                        print('Warning: {} declared as both {} and {}'
                              .format(varname, vartype_prev, vartype),
                              file=sys.stderr)
                else:
                    new_lines.append('{}cython.declare({}={})\n'
                                     .format(indentation, varname, vartype))
                    declarations[varname] = vartype
        else:
            new_lines.append(line)
    return new_lines



def cython_decorators(lines):
    inside_class = False
    for i, line in enumerate(lines):
        if (   line.startswith('class ')
            or line.startswith('cdef class ')
            or line.startswith('cpdef class ')):
            inside_class = True
        elif inside_class and len(line.rstrip('\n')) > 0 and line[0] not in ' #':
            inside_class = False
        for headertype in ('pheader', 'header'):
            if line.lstrip().startswith('@cython.' + headertype):
                # Search for def statement
                for j, line2 in enumerate(lines[(i + 1):]):
                    if 'def ' in line2:
                        def_line = line2
                        for k, c in enumerate(def_line):
                            if c != ' ':
                                n_spaces = k  # Indentation
                                break
                        break
                headstart = i
                headlen = j + 1
                header = lines[headstart:(headstart + headlen)]
                # Look for returntype
                returntype = ''
                for j, hline in enumerate(header):
                    hline = re.sub('returns *= *', 'returns=', hline)
                    if 'returns=' in hline and not hline.lstrip().startswith('#'):
                        in_brackets = 0
                        for c in hline[(hline.index('returns=') + 8):]:
                            if c == '[':
                                in_brackets += 1
                            elif c == ']':
                                in_brackets -= 1
                            elif c == ')' or (c == ',' and not in_brackets):
                                break
                            returntype += c
                        returntype = returntype.strip()
                        header[j] = hline.replace('returns=' + returntype,
                                                  ' '*len('returns=' + returntype))
                        if not header[j].replace(',', '').strip():
                            del header[j]
                        else:
                            # Looks for lonely comma
                            # due to removal of "returns=".
                            lonely = True
                            for k, c in enumerate(header[j]):
                                if c == ',' and lonely:
                                    header[j] = header[j][:k] + ' ' + header[j][(k + 1):] 
                                if c in (',', '('):
                                    lonely == True
                                elif c != ' ':
                                    lonely = False
                        break
                for j, hline in enumerate(header):
                    if '@cython.' + headertype + '(' in hline:
                        I = header[j].index('@cython.' + headertype + '(') + 15
                        for k, c in enumerate(header[j][I:]):
                            if c == ',':
                                header[j] = header[j][:I] + header[j][(I + k + 1):]
                                break
                            elif c != ' ':
                                break
                # Change @cython.header to @cython.locals,
                # if header contains declarations.
                # Otherwise, remove it.
                if '=' in ''.join(header):
                    header[0] = header[0].replace(headertype, 'locals')
                else:
                    header = []
                # Add in all the other decorators.
                # A @cython.header should transform into @cython.cfunc
                # whiel a @cython.pheader should transform into a
                # @cython.ccall. Additional decorators should be placed
                # below. The @cython.inline decorator should not be
                # placed on top of:
                # - So-called special methods, like __init__
                # - Class methods decorated with @cython.pheader (cpdef)
                pyfuncs = ('__init__', '__cinit__', '__dealloc__')
                decorators = [decorator for decorator in
                              (('ccall' if headertype == 'pheader' else 'cfunc')
                               if all(' ' + pyfunc + '(' not in def_line
                                      for pyfunc in pyfuncs) else '',
                              'inline' if (all(' ' + pyfunc + '(' not in def_line
                                              for pyfunc in pyfuncs)
                                           and (not inside_class or headertype != 'pheader')
                                           ) else '',
                              'boundscheck(False)',
                              'cdivision(True)',
                              'initializedcheck(False)',
                              'wraparound(False)',
                               ) if decorator
                              ]
                header = ([' '*n_spaces + '@cython.' + decorator + '\n' for decorator in decorators]
                          + header)
                if returntype:
                    header += [' '*n_spaces + '@cython.returns(' + returntype + ')\n']
                # Place the new header among the lines
                del lines[headstart:(headstart + headlen)]
                for hline in reversed(header):
                    lines.insert(headstart, hline)
    return lines



def power2product(lines):
    keywords = ('assert',
                'elif',
                'except',
                'from',
                'if',
                'in',
                'is',
                'not',
                'or',
                'raise',
                'return',
                'try',
                'while',
                'with',
                'yield',
                )
    new_lines = []
    for line in lines:
        while True:
            match = re.search('([\w.]+(\[.*\])?|\(.*\))\s*\*\*\s*([0-9.]+)', line)
            if not match:
                break
            match_str   = [None]*4
            start_index = [None]*4
            end_index   = [None]*4
            for i in (1, 3):
                match_str[i] = match.group(i)
                start_index[i], end_index[i] = match.span(i)
            # Only integer exponents should be accepted
            try:
                exponent = int(match_str[3])
                if exponent != float(match_str[3]):
                    break
            except:
                break
            # Nested parentheses can lead to too small start_index
            # for the first group. Fix this issue.
            if ')' in match_str[1]:
                N_parens = 0
                for i, c in enumerate(reversed(match_str[1])):
                    if c == '(':
                        N_parens += 1
                        if N_parens == 0:
                            start_index[1] = end_index[1] - i - 1
                            break
                    elif c == ')':
                        N_parens -= 1
                # If the base is a function call, do nothing.
                # Be careful with keywords prior to the base.
                match_before = re.search('[\w.]+\s*$', line[:start_index[1]])
                if match_before:
                    match_before_str = match_before.group().rstrip()
                    if match_before_str not in keywords:
                        break
            elif ']' in match_str[1]:
                # If more than one pair of [] is in the line,
                # and the base is not in parentheses, every [] left
                # of the base gets in the match. Fix thís issue.
                N_brackets = 0
                break_on_nonword = False
                for i, c in enumerate(reversed(match_str[1])):
                    if break_on_nonword and not re.search('[\w.]', c):
                        start_index[1] = end_index[1] - i
                        break
                    if c == '[':
                        N_brackets += 1
                        if N_brackets == 0:
                            # Leftmost bracket found. Keep searching
                            # to the left until non-word-character
                            # is found.
                            break_on_nonword = True
                    elif c == ']': 
                        N_brackets -= 1
            # Stitch together new line
            mul = '*'.join([line[start_index[1]:end_index[1]]]*exponent)
            line = '{}({}){}'.format(line[:start_index[1]], mul, line[end_index[3]:])
        new_lines.append(line)
    return new_lines



def unicode2ASCII(lines):
    new_lines = [commons.asciify(line) for line in lines]
    return new_lines



def __init__2__cinit__(lines):
    new_lines = []
    in_cclass = False
    for line in lines:
        if len(line) > 13 and line[:14] == '@cython.cclass':
            in_cclass = True
        elif line[0] not in ' \n' and not (len(line) > 4 and line[:5] == 'class'):
            in_cclass = False
        if (in_cclass and len(line) > 16
                      and line[:17] == '    def __init__('):
            line = '    def __cinit__(' + line[17:]
        new_lines.append(line)
    return new_lines



def fix_addresses(lines):
    replacement_str = '__REPLACEADDRESS{}__'
    new_lines = []
    for line in lines:
        # 'address(' to 'cython.address'
        if 'address(' in line:
            line = line.replace('address(', 'cython.address(')
            line = line.replace('cython.cython.', 'cython.')
        # cython.address(a[7, ...]) to cython.address(a[7, 0])
        # cython.address(a[7, :, 1]) to cython.address(a[7, 0, 1])
        # cython.address(a[7, 9:, 1]) to cython.address(a[7, 9, 1])
        replacements = []
        line_ori = ''
        while True:
            line_ori = line
            colons_or_ellipsis = True
            while 'cython.address(' in line and colons_or_ellipsis:
                parens = 0
                address_index = line.find('cython.address(', 2) + 14
                for i, c in enumerate(line[address_index:]):
                    if c == '(':
                        parens += 1
                    elif c == ')':
                        parens -= 1
                    if parens == 0:
                        break
                addressof = line[(address_index + 1):(address_index + i)].replace(' ', '')
                addressof = addressof.replace('...', '0')
                for j, c in enumerate(addressof):
                    if c == ':':
                        if ((j == 0 or addressof[j - 1] in '[,')
                            and (j == (len(addressof) - 1) or addressof[j + 1] in '],')):
                            # The case cython.address(a[7, :, 1])
                            addressof = addressof[:j] + '0' + addressof[(j + 1):]
                        else:
                            # The case cython.address(a[7, 9:, 1])
                            addressof = addressof[:j] + ' ' + addressof[(j + 1):]
                colons_or_ellipsis = ':' in addressof or '...' in addressof               
                line = line[:(address_index + 1)] + addressof + line[(address_index + i):]
            # One address corrected
            if line == line_ori:
                break
            # Replace the first occurence of cython.address(...) with
            # a temporary string.
            pattern = r'cython\.address\(.+?\)'
            replacements.append(re.search(pattern, line).group())
            line = re.sub(pattern,
                          replacement_str.format(len(replacements) - 1),
                          line,
                          count=1,
                          )
        # Reinsert the original strings instead of
        # the replacement strings.
        for j, replacement in enumerate(replacements):
            line = line.replace(replacement_str.format(j), replacement)
        new_lines.append(line)
    return new_lines



def malloc_realloc(lines):
    new_lines = []
    for line in lines:
        found_alloc = False
        for alloc in ('malloc(', 'realloc('):
            if alloc in line and 'sizeof(' in line and not line.lstrip().startswith('#'):
                found_alloc = True
                paren = 1
                dtype = ''
                for i in range(line.find('sizeof(') + 7, len(line)):
                    symbol = line[i]
                    if symbol == '(':
                        paren += 1
                    elif symbol == ')':
                        paren -= 1
                    if paren == 0:
                        break
                    dtype += symbol
                dtype = dtype.replace("'", '').replace('"', '')
                line = (line.replace(alloc, '<' + dtype
                        + '*> PyMem_' + alloc.capitalize()))
                new_lines.append(line)
                # Add exception
                LHS = line[:line.find('=')].strip()
                indentation = (len(line[:line.find('=')])
                               - len(line[:line.find('=')].lstrip()))
                new_lines.append(' '*indentation + 'if not ' + LHS + ':\n')
                new_lines.append(' '*(indentation + 4)
                                 + "raise MemoryError('Could not "
                                 + alloc[:-1] + ' ' + LHS + "')\n")
        if not found_alloc:
            if line.lstrip().startswith('free('):
                # Normal frees
                indent_lvl = len(line) - len(line.lstrip())
                ptr = re.search('free\((.+?)\)', line).group(1)
                new_lines.append(' '*indent_lvl + 'if ' + ptr + ' is not NULL:\n')
                new_lines.append(' '*4 + line.replace('free(', 'PyMem_Free('))
            elif re.search('^gsl_.+?_free\(', line.lstrip()):
                # GSL frees
                indent_lvl = len(line) - len(line.lstrip())
                ptr = re.search('gsl_.+?_free\((.+?)\)', line).group(1)
                new_lines.append(' '*indent_lvl + 'if ' + ptr + ' is not NULL:\n')
                new_lines.append(' '*4 + line)
            else:
                new_lines.append(line)
    return new_lines



def C_casting(lines):
    new_lines = []
    # Transform to Cython syntax
    for line in lines:
        while True:
            match = re.search('[^0-9a-zA-Z_]cast\(', line)
            if not match:
                break
            parens = 1
            brackets = 0
            curlys = 0
            for i in range(match.end(), len(line)):
                if line[i] == '(':
                    parens += 1
                elif line[i] == ')':
                    parens -= 1
                if line[i] == '[':
                    brackets += 1
                elif line[i] == ']':
                    brackets -= 1
                if line[i] == '{':
                    curlys += 1
                elif line[i] == '}':
                    curlys -= 1
                if line[i] == ',' and parens == 1 and brackets == 0 and curlys == 0:
                    comma = i
                if parens == 0:
                    end = i
                    expression = line[match.end():comma].strip()
                    ctype = line[(comma + 1):end].strip(' "\'')
                    break
            line = line[:match.end() - 5] + '(<{}>({}))'.format(ctype, expression) + line[(end + 1):]
        new_lines.append(line)
    return new_lines

    # new_lines = []
    # # Transform to Cython syntax
    # for line in lines:
    #     while re.search('(^| )cast\(', line):
    #         match = re.search('(^| )cast\(', line)
    #         start = match.start()
    #         if line[start] == ' ':
    #             start += 1
    #         paren = 1
    #         in_quotes = [False, False]
    #         for i in range(start + 5, len(line)):
    #             symbol = line[i]
    #             if symbol == "'":
    #                 in_quotes[0] = not in_quotes[0]
    #             if symbol == '"':
    #                 in_quotes[1] = not in_quotes[1]
    #             if symbol == '(':
    #                 paren += 1
    #             elif symbol == ')':
    #                 paren -= 1
    #             if paren == 0:
    #                 break
    #             if symbol == ',' and not in_quotes[0] and not in_quotes[1]:
    #                 comma_index = i
    #         cast_to = ('<' + line[(comma_index + 1):i]
    #                    .replace("'", '').replace('"', '').strip() + '>')
    #         obj_to_cast = ('(' + line[(start + 5):comma_index]
    #                        + ')')
    #         line = (line[:line.find('cast(')] + '(' + cast_to + obj_to_cast + ')'
    #                 + line[(i + 1):])
    #     new_lines.append(line)
    # return new_lines



def find_extension_types(lines):
    # Find extension types
    class_names = []
    for i, line in enumerate(lines):
        if line.startswith('@cython.cclass'):
            # Class found
            class_name = None
            for line in lines[(i + 1):]:
                if len(line) > 6 and line[:6] == 'class ':
                    class_name = line[6:line.index(':')].strip()
                    break
            if class_name is None:
                break
            # Classname found
            class_names.append(class_name)
    # Append all extension types at the end of the .pyx file
    if class_names:
        lines.append('# Extension types implemented by this module:\n')
        lines.append('# {}{}\n'.format(' '*4, ', '.join(class_names)))
    return lines



def make_types(filename):
    # Dictionary mapping custom ctypes to their definiton
    # or an import of their definition.
    custom_types = {# External definitions
                    'fftw_plan':          'cdef extern from "fft.c":\n'
                                          '    ctypedef struct fftw_plan_struct:\n'
                                          '        pass\n'
                                          '    ctypedef fftw_plan_struct *fftw_plan',
                    'fftw_return_struct': 'cdef extern from "fft.c":\n'
                                          '    ctypedef struct fftw_plan_struct:\n'
                                          '        pass\n'
                                          '    ctypedef fftw_plan_struct *fftw_plan\n'
                                          '    struct fftw_return_struct:\n'
                                          '        ptrdiff_t gridsize_local_i\n'
                                          '        ptrdiff_t gridsize_local_j\n'
                                          '        ptrdiff_t gridstart_local_i\n'
                                          '        ptrdiff_t gridstart_local_j\n'
                                          '        double* grid\n'
                                          '        fftw_plan plan_forward\n'
                                          '        fftw_plan plan_backward',
                    # GSL functions (... understood by make_pxd)
                    'gsl_...': 'from cython_gsl cimport *',
                    }
    # Add Cython classes (extension types) to custom_types
    for other_pyxfile in all_pyxfiles:
        module = other_pyxfile[:-4]
        with open(other_pyxfile, 'r', encoding='utf-8') as pyxfile:
            code = pyxfile.read().split('\n')
        for i, line in enumerate(reversed(code)):
            if line == '# __pyxinfo__':
                code = code[-i:]
                break
        for i, line in enumerate(code):
            if line == '# Extension types implemented by this module:':
                line = code[i + 1].lstrip('#').lstrip(' ')
                for extension_type in line.split(','):
                    extension_type = extension_type.replace(' ', '')
                    custom_types[extension_type] = 'from {} cimport {}'.format(module, extension_type)
                break
    # Do not write to the types file
    # if it already has the correct content.
    if os.path.isfile(filename):
        with open(filename, 'r', encoding='utf-8') as types_file:
            existing_custom_types_content = types_file.read()
        try:
            existing_custom_types = eval(existing_custom_types_content)
            if existing_custom_types == custom_types:
                return
        except:
            print('Warning: Could not interpret the content of "{}".'.format(filename))
    # Write the dictionary to the types file:
    with open(filename, 'w', encoding='utf-8') as types_file:
        types_file.write(str(custom_types))



def make_pxd(filename):
    # Read in the custom types from the types file
    with open(filename_types, 'r', encoding='utf-8') as pyxfile:
        custom_types_content = pyxfile.read()
    custom_types = eval(custom_types_content)
    # Begin constructing pxd
    header_lines = []
    pxd_filename = filename[:-3] + 'pxd'
    pxd_lines = []
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        code_str = pyxfile.read()
    code = code_str.split('\n')
    # Find pxd hints of the form 'pxd = """'
    #                             int var1
    #                             double var2
    #                             """'
    pxd_lines.append('# pxd hints\n')
    for i, line in enumerate(code):
        if (   line.replace(' ', '').startswith('pxd="""')
            or line.replace(' ', '').startswith("pxd='''")):
            quote_type = '"""' if line.replace(' ', '').startswith('pxd="""') else "'''"
            for j, line in enumerate(code[(i + 1):]):
                if line.startswith(quote_type):
                    break
                pxd_lines.append(line + '\n')
    # Remove the '# pxd hints' line if no pxd hints were found
    if pxd_lines[-1].startswith('# pxd hints'):
        pxd_lines.pop()
    else:
        pxd_lines.append('\n')
    # Import all types with spaces (e.g. "long int") from the commons module
    types_with_spaces = [(key.replace(' ', ''), key)
                         for key in commons.C2np.keys()
                         if ' ' in key]
    types_with_spaces = sorted(types_with_spaces, key=lambda t: len(t[1]), reverse=True)
    # Function that finds non-indented function definitions in a block
    # of code (list of lines). It appends to header_lines and pxd_lines.
    def find_functions(code, indent=0, only_funcname=None):
        for i, line in enumerate(code):
            if line.startswith('def '):
                # Function definition found.
                # Find out whether cdef (cfunc)
                # or cpdef (ccall) function.
                cpdef = False
                purepy_func = True
                for cp_line in reversed(code[:i]):
                    if cp_line.startswith('def '):
                        break
                    if cp_line.startswith('@cython.ccall'):
                        purepy_func = False
                        cpdef = True
                        break
                    if cp_line.startswith('@cython.cfunc'):
                        purepy_func = False
                        break
                # Do not add declarations of pure Python functions
                if purepy_func:
                    continue
                # Find function name and args
                open_paren = line.index('(')
                function_name = line[3:open_paren].strip()
                N_parens = 0
                for j, c in enumerate(line[open_paren:]):
                    if c == '(':
                        N_parens += 1
                    elif c == ')':
                        N_parens -= 1
                    if N_parens == 0:
                        closed_paren = open_paren + j
                        break
                function_args = line[(open_paren + 1):closed_paren]
                function_args = function_args.strip()
                if len(function_args) > 0 and function_args[-1] == ',':
                    function_args = function_args[:-1]
                    function_args = function_args.strip()
                # Function name and args found.
                # If searching for a specific function name and this is
                # not it, continue.
                if only_funcname and only_funcname != function_name:
                    continue
                # Replace default keyword argument values
                # with an asterisk.
                function_args_bak = ''
                while function_args_bak != function_args:
                    function_args_bak = deepcopy(function_args)
                    inside_quote = {"'": False, '"': False}
                    for j, c in enumerate(function_args):
                        # Inside a quote?
                        for key, val in inside_quote.items():
                            if c == key:
                                inside_quote[key] = not val
                        # Replace spaces inside quotes with a temporary string
                        if True in inside_quote.values() and c == ' ':
                            function_args = function_args[:j] + '__pyxpp_space__' + function_args[(j + 1):]
                            break
                for j, c in enumerate(function_args):
                    if c == '=':
                        if function_args[j + 1] == ' ':
                            function_args = function_args[:(j + 1)] + function_args[(j + 2):]
                        if function_args[j - 1] == ' ':
                            function_args = function_args[:(j - 1)] + function_args[j:]
                from_top = False
                function_args_bak = ''
                while function_args_bak != function_args:
                    function_args_bak = deepcopy(function_args)
                    for j, c in enumerate(function_args):
                        if c == '=' and function_args[j + 1] != '*':
                            for k in range(j + 1, len(function_args)):
                                if function_args[k] in (',', ' '):
                                    function_args = (function_args[:(j + 1)] + '*'
                                                                             + function_args[k:])
                                    break
                                elif k == len(function_args) - 1:
                                    function_args = function_args[:(j + 1)] + '*'
                                    break
                            break
                function_args = function_args.replace('__pyxpp_space__', ' ')
                # Find types for the arguments and write them in
                # front of the arguments in function_args.
                function_args = function_args.split(',')
                return_vals = [None]*len(function_args)
                for j in range(len(function_args)):
                    function_args[j] = function_args[j].strip()
                line_before = deepcopy(code[i - 1])
                for j, arg in enumerate(function_args):
                    if '=*' in arg:
                        arg = arg[:-2]
                    for k, line in enumerate(reversed(code[:i])):
                        break_k = False
                        if len(line) > 0 and line[0] not in ('@', ' ', '#'):
                            # Above function decorators
                            break
                        if line.startswith('@cython.returns(') and return_vals[j] is None:
                            # Return value found.
                            # Assume it is a one-liner.
                            return_val = line[16:].strip()
                            if return_val[-1] == ')':
                                return_val = return_val[:-1].strip()
                            return_val = return_val.replace('"', '')
                            return_val = return_val.replace("'", '')
                            return_vals[j] = return_val
                        if k != 0 and line.startswith('def '):
                            # Previous function reached. The current
                            # function must be a pure Python function.
                            function_args[j] = None
                            break
                        line = line.replace(' ', '')
                        if (arg + '=') in line:
                            for l in range(len(line) - len(arg + '=')):
                                if line[l:(l + len(arg + '='))] == (arg + '='):
                                    if l == 0 or line[l - 1] in (' ', ',', '('):
                                        argtype = deepcopy(line[(l + len(arg + '=')):])
                                        if ',' in argtype:
                                            commas = [m for m, c in enumerate(argtype) if c == ',']
                                            for m in commas:
                                                a = argtype[:m]
                                                if a.count('[') == a.count(']'):
                                                    break
                                            else:
                                                a = argtype
                                            argtype = a
                                        argtype = argtype.strip()
                                        argtype = argtype.strip(',')
                                        argtype = argtype.strip()
                                        argtype = argtype.strip(')')
                                        argtype = argtype.strip()
                                        argtype = argtype.replace('"', '')
                                        argtype = argtype.replace("'", '')
                                        # Add spaces back to multiword argument types
                                        for t in types_with_spaces:
                                            argtype = argtype.replace(t[0], t[1])
                                        function_args[j] = function_args[j].strip()
                                        function_args[j] = function_args[j].strip(',')
                                        function_args[j] = function_args[j].strip()
                                        if function_args[j]:
                                            function_args[j] = argtype + ' ' + function_args[j]
                                        break_k = True
                                        break
                            if break_k:
                                ine_before = deepcopy(line)
                                break
                        line_before = deepcopy(line)
                # Due to a bug in the above, the last argument can
                # sometimes include the closing parenthesis.
                # Remove this parenthesis.
                function_args[-1] = function_args[-1].replace(')', '')
                # None's in function_args means pure Python functions
                if None in function_args:
                    continue
                # Remove quotes from function arguments
                for j in range(len(function_args)):
                    function_args[j] = function_args[j].replace('"', '')
                    function_args[j] = function_args[j].replace("'", '')
                # Add the function definition
                s = '    '
                if cpdef:
                    s += 'cpdef '
                if return_vals[j] is not None:
                    s += return_vals[j] + ' '
                s += function_name + '('
                for arg in function_args:
                    s += arg + ', '
                if len(s) > 1 and s[-2:] == ', ':
                    s = s[:-2]
                s += ')\n'
                pxd_lines.append(' '*indent + s)
    # Find classes
    pxd_lines.append('# Classes\n')
    for i, line in enumerate(code):
        if line.startswith('@cython.cclass'):
            # Class found
            class_name = None
            for j0, line in enumerate(code[(i + 1):]):
                if len(line) > 6 and line[:6] == 'class ':
                    class_name = line[6:line.index(':')].strip()
                    break
            if class_name is None:
                break
            # Classname found. Now find __cinit__
            for j, line in enumerate(code[(j0 + i + 1):]):
                if line.startswith('    def __cinit__('):
                    # __cinit__ found. Locate triple quoted string
                    for k, line in enumerate(code[(j + j0 + i + 2):]):
                        if len(line) > 0 and line[0] not in ' #':
                            # Out of class
                            break
                        if line.startswith(' '*8 + '"""'):
                            pxd_lines.append('cdef class ' + class_name + ':\n')
                            pxd_lines.append('    cdef:\n')
                            pxd_lines.append(' '*8 + '# Data attributes\n')
                            for l, line in enumerate(code[(k + j + j0 + i + 3):]):
                                if line.startswith('        """'):
                                    break
                                pxd_lines.append(line + '\n')
                            break
                    break
            # Now locate methods.
            # Find end of the class.
            for m, line in enumerate(code[(l + k + j + j0 + i + 4):]):
                if len(line) > 0 and line[0] not in ' #':
                    break
            if m == len(code) - 1:
                m += 1
            class_code_unindented = [line if len(line) < 4 else line[4:]
                                     for line in code[(l + k + j + j0 + i + 4)
                                                      :(m + l + k + j + j0 + i + 4 )]]
            pxd_lines.append(' '*8 + '# Methods\n')
            find_functions(class_code_unindented, indent=4)
            # Remove the '# Methods' line if the class has no methods
            if pxd_lines[-1].lstrip().startswith('#'):
                pxd_lines.pop()
    # Remove the '# Classes' line if no class were found
    if pxd_lines[-1].startswith('# Classes'):
        pxd_lines.pop()
    else:
        pxd_lines.append('\n')
    # Find functions
    pxd_lines.append('# Functions\n')
    pxd_lines.append('cdef:\n')
    find_functions(code)
    # Remove the '# Functions' and 'cdef:' lines
    # if no functions were found.
    if pxd_lines[-2].startswith('# Functions'):
        pxd_lines.pop()
        pxd_lines.pop()
    else:
        pxd_lines.append('\n')
    # Find global variables (global cython.declare() statements)
    pxd_lines.append('# Variables\n')
    pxd_lines.append('cdef:\n')
    variable_index = len(pxd_lines)
    globals_phony_funcname = '__pyxpp_phony__'
    globals_code = deepcopy(code)
    while True:
        done = True
        for i, line in enumerate(globals_code):
            if line.startswith('cython.declare('):
                done = False
                declaration = line[14:]
                declaration = re.sub("'.*?'", '', declaration)
                declaration = declaration.replace('=', '')
                globals_code = (globals_code[:i] + ['@cython.cfunc',
                                                   '@cython.locals' + line[14:],
                                                   ('def '
                                                    + globals_phony_funcname + declaration
                                                    + ':'),
                                                   '    ...'] + globals_code[(i + 1):])
                break
        if done:
            break
    find_functions(globals_code, only_funcname=globals_phony_funcname)
    phony_start = '    ' + globals_phony_funcname + '('
    while True:
        done = True
        for i, line in enumerate(pxd_lines[variable_index:]):
            if line.startswith(phony_start):
                line = line[len(phony_start):-2].strip()
                if ',' in line:
                    lines = line.split(',')
                    while True:
                        done2 = True
                        for j in range(len(lines)):
                            if lines[j].count('[') > lines[j].count(']'):
                                done2 = False
                                combined_line = lines[j] + ',' + lines[j + 1]
                                lines = lines[:j] + [combined_line] + lines[(j + 2):]
                                break
                        if done2:
                            break
                    for j in range(len(lines)):
                        lines[j] = '    ' + lines[j].strip() + '\n'
                    pxd_lines = (pxd_lines[:(variable_index + i)]
                                 + lines
                                 + pxd_lines[(variable_index + i + 1):])
                    done = False
                    break
                else:
                    pxd_lines[variable_index + i] = '    ' + line + '\n'
        if done:
            break
    # Remove the '# Variables' and 'cdef:' lines
    # if no variables were found.
    if pxd_lines[-2].startswith('# Variables'):
        pxd_lines.pop()
        pxd_lines.pop()
    else:
        pxd_lines.append('\n')
    # Find declarations of non-builtin types (extension types)
    for vartype, vardeclaration in custom_types.items():
        # Skip declaration if vartype is defined in this module
        if vardeclaration.startswith('from {} cimport '.format(filename[:-4])):
            continue
        # vartype may contain ellipses. These should be replaced
        # with regular expressions which describe a single
        # variable name.
        vartype = vartype.replace('...', r'[_a-zA-Z0-9]*')
        # Search the entire .pyx file for string of the form
        # varname = 'vartype'.
        if re.search((  r"""(^|[,;(\s])[_a-zA-Z][_a-zA-Z0-9]*\s*="""
                      + r"""\s*(?P<quote>['"]){vartype}\*?(?P=quote)"""
                      ).format(vartype=vartype),
                     code_str,
                     re.MULTILINE):
            # Match found.
            # Assume this is a declaration of the extension type.
            header_lines.append(vardeclaration)
            continue
        # For extension type attributes another syntax is used;
        # vartype varname
        # Also search after this.
        if re.search((  r"""\s*{vartype}\s*\*?\s*"""
                      + r"""(^|[,;(\s])[_a-zA-Z][_a-zA-Z0-9]*\s*"""
                      ).format(vartype=vartype),
                     code_str,
                     re.MULTILINE):
            header_lines.append(vardeclaration)
    # Combine header_lines and pxd_lines
    header_lines = list(set(header_lines))
    while len(header_lines) > 0 and len(header_lines[0].strip()) == 0:
        header_lines = header_lines[1:]
    if header_lines:
        header_lines = ['# Non-builtin types'] + header_lines
    for i in range(len(header_lines)):
        header_lines[i] += '\n'
    total_lines = header_lines
    while len(pxd_lines) > 0 and len(pxd_lines[0].strip()) == 0:
        pxd_lines = pxd_lines[1:]
    if total_lines != []:
        total_lines.append('\n')
    total_lines += pxd_lines
    # Add 'cimport cython' as the top line
    total_lines = ['# Get full access to all of Cython\n',
                   'cimport cython\n',
                   '\n',
                   ] + total_lines
    # Remove duplicates
    total_lines_unique = []
    counter = collections.Counter(total_lines)
    for i, line in enumerate(total_lines):
        if (    line.strip()
            and not line.startswith(' ')
            and len(line.strip().split()) > 1
            and counter[line] > 1):
            counter[line] -= 1
            continue
        total_lines_unique.append(line)
    total_lines = total_lines_unique
    # If nothing else, place a comment in the pxd file
    if not total_lines:
        total_lines = ['# This module does not expose any c-level functions or classes '
                       'to the outside world\n']
    # Do not write to pxd if it already exist in the correct state
    if os.path.isfile(pxd_filename):
        with open(pxd_filename, 'r', encoding='utf-8') as pxdfile:
            existing_pxd_lines = pxdfile.readlines()
        if total_lines == existing_pxd_lines:
            return
    # Update/create .pxd
    with open(pxd_filename, 'w', encoding='utf-8') as pxdfile:
        pxdfile.writelines(total_lines)



# Interpret input argument
filename = sys.argv[1]
filename_commons = sys.argv[2]
if len(sys.argv) > 3:
    filename_types = sys.argv[3]
if len(sys.argv) > 4:
    all_pyxfiles = sys.argv[4:]
# Import the non-compiled commons module
commons_name = filename_commons[:-3]
commons = imp.load_source(commons_name, filename_commons)
# Perform operations
if len(sys.argv) > 4:
    # Make the types file, containing the definitions of all custom
    # types implemented in the .pyx files.
    if filename.endswith('.pyx'):
        make_types(filename)  # filename == filename_types
    else:
        raise Exception('Got "{}" which is not a .pyx file as the first argument, '
                        'while receiving more than three arguments'.format(filename))
else:
    if filename.endswith('.py'):
        # A .py-file is passed.
        # Read in the lines of the file.
        with open(filename, 'r', encoding='utf-8') as pyfile:
            lines = pyfile.readlines()
        # Apply transformations on the lines
        lines = cimport_cython(lines)
        lines = oneline(lines)
        lines = cythonstring2code(lines)
        lines = cython_structs(lines)
        lines = cimport_commons(lines)
        lines = cimport_function(lines)
        lines = constant_expressions(lines)
        lines = unicode2ASCII(lines)
        lines = loop_unswitching(lines)
        lines = remove_duplicate_declarations(lines)
        lines = cython_decorators(lines)
        lines = power2product(lines)
        lines = __init__2__cinit__(lines)
        lines = fix_addresses(lines)
        lines = malloc_realloc(lines)
        lines = C_casting(lines)
        lines = find_extension_types(lines)       
        # Write the modified lines to the .pyx-file
        filename_pyx = filename[:-2] + 'pyx'
        with open(filename_pyx, 'w', encoding='utf-8') as pyxfile:
            pyxfile.writelines(lines)
    elif filename.endswith('.pyx'):
        # A .pyx-file is passed.
        # Make the .pxd.
        make_pxd(filename)
    else:
        raise Exception('Got "{}", which is neither a .py nor a .pyx file'.format(filename))
