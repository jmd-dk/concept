# Copyright (C) 2015 Jeppe Mosgard Dakin
#
# This file is part of CONCEPT, the cosmological N-body code in Python
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.



"""
This is the .pyx preprocessor script. Run it with a .pyx file as the first
argument, the parameterfile as the second and the mode ("pyx" or "pxd") as
the third. When mode is "pxd", a .pxd file will be created. When the mode is
"pyx", the preexisting .pyx file will be changed in the following ways:
- Replace 'from commons import *' with the content itself.
- Transform statements written over multiple lines into single lines.
  The exception is decorator statements, which remain multilined.
- Removes pure Python commands between 'if not cython.compiled:' and 'else:',
  including these lines themselves. Also removes the triple quotes around the
  Cython statements in the else body.
- Integer powers will be replaced by products.
- Unicode non-ASCII letters will be replaced with ASCII-strings.
- __init__ methods in cclasses are renamed to __cinit__.
- Replaces : with 0 when taking the address of arrays.
- Replaces alloc, realloc and free with the corresponding PyMem_ functions and
  takes care of the casting from the void* to the appropriate pointer type.
- Replaced the cast() function with actual Cython syntax, e.g. <double[::1]>.

  This script is not written very elegantly, and do not leave
  the modified code in a very clean state. Sorry...
"""

import sys
import re
from os.path import isfile
from copy import deepcopy
import unicodedata


def oneline(filename):
    in_quotes = [False, False]
    in_triple_quotes = [False, False]
    paren = 0
    brack = 0
    curly = 0
    def count_parens(line):
        nonlocal in_quotes, paren, brack, curly
        if line.lstrip().startswith('#'):
            return (paren, brack, curly)
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
                #if in_triple_quotes[0] or in_triple_quotes[1]:
                #    continue

            if ch == '#':
                break
            # Count parentheses outside quotes
            if not in_quotes[0] and not in_quotes[1]:
                if ch == '(':
                    paren += 1
                elif ch == ')':
                    paren -= 1
                elif ch == '[':
                    brack += 1
                elif ch == ']':
                    brack -= 1
                elif ch == '{':
                    curly += 1
                elif ch == '}':
                    curly -= 1
        return (paren, brack, curly)

    new_lines = []
    multiline_statement = []
    multiline = False
    multiline_decorator = False
    with open(filename, 'r') as pyxfile:
        for i, line in enumerate(pyxfile):
            count_parens(line)
            if (paren > 0 or brack > 0 or curly > 0) and not multiline:
                # Multiline statement begins
                multiline = True
                if line.lstrip().startswith('@'):
                    multiline_decorator = True
                    new_lines.append(line)
                    continue
                if '#' in line:
                    line = line[:line.index('#')]
                if line:
                    multiline_statement.append(line.rstrip())
            elif (paren > 0 or brack > 0 or curly > 0) and multiline:
                # Multiline statement continues
                if multiline_decorator:
                    new_lines.append(line)
                    continue
                if '#' in line:
                    line = line[:line.index('#')]
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
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)





def import_params(filename):
    new_lines = []
    import_line = 'from ' + 'commons' + ' import *'
    with open(filename, 'r') as pyxfile:
        for line in pyxfile:
            if line.startswith(import_line):
                with open('commons' + '.py', 'r') as active_params_file:
                    for active_params_line in active_params_file:
                        new_lines.append(active_params_line)
            else:
                new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)


def cythonstring2code(filename):
    new_lines = []
    with open(filename, 'r') as pyxfile:
        in_purePythonsection = False
        unindent = False
        purePythonsection_start = 0
        indentation = 0
        for i, line in enumerate(pyxfile):
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
                    line_without_triple_quotes = line.replace('"""', '').replace("'''", '')
                    if len(line_without_triple_quotes) > 4:
                        new_lines.append(line_without_triple_quotes[4:])
                else:
                    new_lines.append(line)
            if (i != purePythonsection_start and in_purePythonsection
                                             and len(line) >= indentation
                                             and line[indentation] != ' '):
                in_purePythonsection = False
                if 'else:' in line:
                    unindent = True
                else:
                    new_lines.append(line)
                    unindent = False
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)


def power2product(filename):
    pyxpp_power = '__pyxpp_power__'

    def pow2prod(line, firstcall=True):
        if '**' in line or pyxpp_power in line:
            line = line.rstrip('\n')
            line = line.replace(pyxpp_power, '**', 1)
            # Place spaces before and after **
            indices = [starstar.start() for starstar
                       in re.finditer('\*\*', line)]
            for i, index in enumerate(indices):
                line = line[:(index + 2*i)] + ' ** ' + line[(index + 2*i) + 2:]
            # Split line into segments containing ** operators
            indices = [-2] + [starstar.start() for starstar
                              in re.finditer('\*\*', line + '** **')]
            expressions = [line[indices[i-1] + 2:indices[i+1]] for i, index
                           in enumerate(indices[:-2]) if i > 0]
            modified_line = ''
            # Only change nonvariable, int powers. This also excludes **kwargs
            for ex, expression in enumerate(expressions):
                power = None
                expression_power = expression[expression.find('**') + 2:]
                # Exclude **kwargs
                if expression_power.strip().startswith('kwargs'):
                    modified_line = modified_line if modified_line else line
                    continue
                if expression_power.replace(' ', '')[0] == '(':
                    # Power in parentheses
                    parentheses = 0
                    done = False
                    for i, symbol in enumerate(expression_power):
                        if symbol == '(':
                            parentheses += 1
                            done = True
                        elif symbol == ')':
                            parentheses -= 1
                        if parentheses == 0 and done:
                            break
                    power = expression_power[:i + 1]
                    after_power = expression_power[i + 1:]
                else:
                    # Power not in parentheses
                    power_with_sign = (True if
                                       expression_power.replace(' ', '')[0]
                                       in '+-' else False)
                    nr_of_signs = 0
                    for i, symbol in enumerate(expression_power):
                        nr_of_signs += symbol in '+-'
                        if (nr_of_signs > power_with_sign or symbol not in '+- .0123456789'):
                            break
                    # I'm not sure if "expression_power[-1] != ')'" is safe!
                    if i == len(expression_power) - 1 and expression_power[-1] != ')':
                        i += 1
                    power = expression_power[:i]
                    after_power = expression_power[i:]
                integer_power = False
                if power is not None:
                    try:
                        x = eval(power)
                        if x == int(x):
                            integer_power = True
                    except:
                        pass
                expression_base = expression[:expression.find('**')]
                base = ''
                before_base = ''
                nested_power = False
                if integer_power and (expression_base.replace(' ', '') == '' or (expression_base.replace(' ', '')[-1] == ')' and expression_base.count(')') != expression_base.count('('))) and firstcall:
                    # Nested power, as in the last ** in '(a**2 + b**2)**3'
                    nested_power = True
                elif expression_base.replace(' ', '')[-1] == ')':
                    # Base in parentheses
                    done = False
                    parentheses = 0
                    for i, symbol in enumerate(reversed(expression_base)):
                        if symbol == '(':
                            parentheses += 1
                        elif symbol == ')':
                            parentheses -= 1
                            done = True
                        if parentheses == 0 and done:
                            break
                    base = expression_base[::-1][0:i + 1][::-1]
                    before_base = expression_base[::-1][i + 1:][::-1]
                else:
                    # Base not in parentheses
                    brackets_width_content = ''
                    if expression_base.replace(' ', '')[-1] == ']':
                        # Base ends with a bracket
                        done = False
                        brackets = 0
                        for i, symbol in enumerate(reversed(expression_base)):
                            if symbol == '[':
                                brackets += 1
                            elif symbol == ']':
                                brackets -= 1
                                done = True
                            if brackets == 0 and done:
                                break
                        brackets_width_content = expression_base[(len(expression_base) - i - 1):].rstrip()
                        expression_base = expression_base[:(len(expression_base) - i - 1)]
                    for i, symbol in enumerate(reversed(expression_base)):
                        try:
                            base = symbol + base
                            exec(base.replace('.', '') + ' = 0')
                        except:
                            if symbol not in '.0123456789' and base != ' '*len(base):
                                break
                    base = base[1:].replace(' ', '') + brackets_width_content
                    before_base = expression_base[::-1][i:][::-1]
                # Replaces ** with a string of multiplications 
                # for integer powers
                if integer_power:
                    if nested_power and firstcall:
                        operation = expression_base + pyxpp_power + power
                    else:
                        operation = ('(' + ((base + '*')
                                     *int(abs(eval(power))))[:-1] + ')')
                        if eval(power) < 0:
                            operation = '(1/' + operation + ')'
                else:
                        operation = base + '**' + power
                # Stitch together a modified version of the line
                if ex < len(expressions) - 1 and len(expressions) > 1:
                    expressions[ex] = expressions[ex][:-1]
                    expressions[ex + 1] = expressions[ex + 1][len(power) + 0:]
                if ex < len(expressions) - 1:
                    modified_line += before_base + operation
                else:
                    modified_line += before_base + operation + after_power
            return modified_line + '\n'
        return line

    new_lines = []
    with open(filename, 'r') as pyxfile:
        for line in pyxfile:
            line = pow2prod(line)
            while pyxpp_power in line:
                line = pow2prod(line, firstcall=False)
            new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)


def unicode2ASCII(filename):
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        text = [char for char in pyxfile.read()]
    for i, char in enumerate(text):
        if ord(char) > 127:
            text[i] = unicodedata.name(char).replace(' ', '_')
    text = ''.join(text)
    with open(filename, 'w') as pyxfile:
        pyxfile.write(text)
    return

def __init__2__cinit__(filename):
    new_lines = []
    with open(filename, 'r') as pyxfile:
        in_cclass = False
        for line in pyxfile:
            if len(line) > 13 and line[:14] == '@cython.cclass':
                in_cclass = True
            elif (line[0] not in ' \n'
                  and not (len(line) > 4
                  and line[:5] == 'class')):
                in_cclass = False
            if (in_cclass and len(line) > 16
                          and line[:17] == '    def __init__('):
                line = '    def __cinit__(' + line[17:]
            new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)


def colon2zero_in_addresses(filename):
    new_lines = []
    with open(filename, 'r') as pyxfile:
        for line in pyxfile:
            if 'cython.address(' in line:
                for i in range(line.find('cython.address('), len(line)):
                    if line[i] == ')':
                        break
                    elif line[i] == ':':
                        line = line[:i] + '0' + line[i + 1:]
                        break
            new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)


def malloc_realloc(filename):
    new_lines = []
    with open(filename, 'r') as pyxfile:
        for line in pyxfile:
            found_alloc = False
            for alloc in ('malloc(', 'realloc('):
                if alloc in line and 'sizeof(' in line:
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
                line = line.replace(' free(', ' PyMem_Free(')
                new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)


def C_casting(filename):
    new_lines = []
    with open(filename, 'r') as pyxfile:
        # Transform to Cython syntax
        for line in pyxfile:
            if 'cast(' in line:
                paren = 1
                in_quotes = [False, False]
                for i in range(line.find('cast(') + 5, len(line)):
                    symbol = line[i]
                    if symbol == "'":
                        in_quotes[0] = not in_quotes[0]
                    if symbol == '"':
                        in_quotes[1] = not in_quotes[1]
                    if symbol == '(':
                        paren += 1
                    elif symbol == ')':
                        paren -= 1
                    if paren == 0:
                        break
                    if symbol == ',' and not in_quotes[0] and not in_quotes[1]:
                        comma_index = i
                cast_to = ('<' + line[(comma_index + 1):i]
                           .replace("'", '').replace('"', '').strip() + '>')
                obj_to_cast = ('(' + line[(line.find('cast(') + 5):comma_index]
                               + ')')
                line = (line[:line.find('cast(')] + cast_to + obj_to_cast
                        + line[(i + 1):])
            new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)


def cython_decorators(filename):
    with open(filename, 'r') as pyxfile:
        lines = pyxfile.read().split('\n')
    for i, line in enumerate(lines):
        if '@cython.header' in line:
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
                if 'returns=' in hline:
                    in_brackets = 0
                    for c in hline[(hline.index('returns=') + 8):]:
                        if c == '[':
                            in_brackets += 1
                        elif c == ']':
                            in_brackets -= 1
                        elif c == ')' or (c == ',' and not in_brackets):
                            break
                        returntype += c
                    header[j] = header[j].replace('returns=' + returntype, ' '*len('returns=' + returntype))
                    if not header[j].replace(',', '').strip():
                        del header[j]
                    else:
                        # Looks for lonely comma due to removal of "returns="
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
                if '@cython.header(' in hline:
                    I = header[j].index('@cython.header(') + 15
                    for k, c in enumerate(header[j][I:]):
                        if c == ',':
                            header[j] = header[j][:I] + header[j][(I + k + 1):]
                            break
                        elif c != ' ':
                            break
            # Change @cython.header to @cython.locals, if header contains declarations.
            # Otherwise, remove it.
            if '=' in ''.join(header):
                header[0] = header[0].replace('header', 'locals')
            else:
                header = []
            # Add in all the other decorators
            pyfuncs = ('__init__', '__cinit__', '__dealloc__')
            decorators = [decorator for decorator in
                          ('cfunc' if all(' ' + pyfunc + '(' not in def_line
                                          for pyfunc in pyfuncs) else '',
                          'inline' if all(' ' + pyfunc + '(' not in def_line
                                          for pyfunc in pyfuncs) else '',
                          'boundscheck(False)',
                          'cdivision(True)',
                          'initializedcheck(False)',
                          'wraparound(False)',
                           ) if decorator
                          ]
            header = [' '*n_spaces + '@cython.' + decorator for decorator in decorators] + header
            if returntype:
                header += [' '*n_spaces + '@cython.returns(' + returntype + ')']
            # Place the new header among the lines
            del lines[headstart:(headstart + headlen)]
            for hline in reversed(header):
                lines.insert(headstart, hline)
    # Write all lines to file
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines('\n'.join(lines))


def make_pxd(filename):
    commons_functions = ('abs', 'max', 'min', 'mod', 'sum', 'prod', 'sinc', 'masterprint', 'masterwarn', 'progressprint')
    customs = {'Particles':    'from species cimport Particles',
               'func_b_ddd':   'ctypedef bint    (*func_b_ddd_pxd)  (double, double, double)',
               'func_d_dd':    'ctypedef double  (*func_d_dd_pxd)   (double, double)',
               'func_d_ddd':   'ctypedef double  (*func_d_ddd_pxd)  (double, double, double)',
               'func_ddd_ddd': 'ctypedef double* (*func_ddd_ddd_pxd)(double, double, double)',
               }
    header_lines = []
    pxd_filename = filename[:-3] + 'pxd'
    pxd_lines = []
    with open(filename, 'r') as pyxfile:
        code = pyxfile.read().split('\n')
    # Find pxd hints of the form 'pxd = """'
    pxd_lines.append('cdef:\n')
    for i, line in enumerate(code):
        if line.startswith('pxd = """'):
            for j, line in enumerate(code[(i + 1):]):
                if line.startswith('"""'):
                    pxd_lines.append('\n')
                    break
                pxd_lines.append('    ' + line + '\n')
    # Find classes
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
                        if len(line) > 0 and line[0] != ' ':
                            # Out of class
                            break
                        if line.startswith(' '*8 + '"""'):
                            pxd_lines.append('cdef class ' + class_name + ':\n')
                            pxd_lines.append('    cdef:\n')
                            for l, line in enumerate(code[(k + j + j0 + i + 3):]):
                                if line.startswith('        """'):
                                    break
                                pxd_lines.append(line + '\n')
                            pxd_lines.append('\n')
                            break
                    break
    # Find functions
    pxd_lines.append('cdef:\n')
    for i, line in enumerate(code):
        if line.startswith('def '):
            # Function definition found
            open_paren = line.index('(')
            function_name = line[3:open_paren].strip()
            if function_name in commons_functions:
                continue
            try:
                closed_paren = line.index(')')
                function_args = line[(open_paren + 1):closed_paren]
            except ValueError:
                # Closed paren on a later line
                function_args = []
                for line in code[i:]:
                    function_args.append(line)
                    if ')' in line:
                        closed_paren = line.index(')')
                        function_args[0] = function_args[0][(open_paren + 1):]
                        function_args[-1] = function_args[-1][:closed_paren]
                        function_args = re.sub(' +', ' ', ' '.join(function_args))
                        break
            function_args = function_args.strip()
            if len(function_args) > 0 and function_args[-1] == ',':
                function_args = function_args[:-1]
                function_args = function_args.strip()
            # Function name and args found.
            # Replace default keyword argument values with an asterisk.
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
                                function_args = function_args[:(j + 1)] + '*' + function_args[k:]
                                break
                            elif k == len(function_args) - 1:
                                function_args = function_args[:(j + 1)] + '*'
                                break
                        break
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
                        # Return value found. Assume it is a one-liner
                        return_val = line[16:].strip()
                        if return_val[-1] == ')':
                            return_val = return_val[:-1].strip()
                        return_val = return_val.replace('"', '')
                        return_val = return_val.replace("'", '')
                        return_vals[j] = return_val
                    if k != 0 and line.startswith('def '):
                        # Previous function reached. The current function
                        # must be a pure Python function
                        function_args[j] = None
                        break
                    line = line.replace(' ', '')
                    if (arg + '=') in line:
                        for l in range(len(line) - len(arg + '=')):
                            if line[l:(l + len(arg + '='))] == (arg + '='):
                                if l == 0 or line[l - 1] in (' ', ',', '('):
                                    argtype = deepcopy(line[(l + len(arg + '=')):])
                                    if ',' in argtype:
                                        commas = [m for m, c in enumerate(range(len(argtype))) if c == ',']
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
                                    argtype = argtype.replace('"', '')
                                    argtype = argtype.replace("'", '')
                                    # Add suffix _pxd to the "func_" types
                                    if argtype in customs:
                                        header_lines.append(customs[argtype])
                                    if 'func_' in argtype:
                                        argtype += '_pxd'
                                    function_args[j] = function_args[j].strip()
                                    function_args[j] = function_args[j].strip(',')
                                    function_args[j] = function_args[j].strip()
                                    function_args[j] = argtype + ' ' + function_args[j]
                                    break_k = True
                                    break
                        if break_k:
                            ine_before = deepcopy(line)
                            break
                    line_before = deepcopy(line)
            # None's in function_args means pure Python functions
            if None in function_args:
                continue
            # Remove quotes from function arguments
            for j in range(len(function_args)):
                function_args[j] = function_args[j].replace('"', '')
                function_args[j] = function_args[j].replace("'", '')
            # Add the function definition
            s = '    '
            if return_vals[j] is not None:
                s += return_vals[j] + ' '
            s += function_name + '('
            for arg in function_args:
                s += arg + ', '
            if len(s) > 1 and s[-2:] == ', ':
                s = s[:-2]
            s += ')\n'
            pxd_lines.append(s)
    pxd_lines_backup = []
    while pxd_lines_backup != pxd_lines:
        pxd_lines_backup = deepcopy(pxd_lines)
        for i, line in enumerate(pxd_lines):
            OK = False
            if line.startswith('cdef:'):
                if len(pxd_lines) > (i + 1):
                    if len(pxd_lines[i + 1]) > 4:
                        if pxd_lines[i + 1][:4] == '    ' and pxd_lines[i + 1][4] != ' ':
                            OK = True
                if not OK:
                    pxd_lines.pop(i)
                    break
    header_lines = list(set(header_lines))
    for i in range(len(header_lines)):
        header_lines[i] += '\n'
    total_lines = header_lines
    if total_lines != []:
        total_lines.append('\n')
    total_lines += pxd_lines
    # Do not write to .pxd if it already has the correct content.
    # This is important as the .pxd files are used by the makefile.
    if isfile(pxd_filename):
        total_lines_nonewlines = deepcopy(total_lines)
        for i in range(len(total_lines_nonewlines)):
            total_lines_nonewlines[i] = total_lines_nonewlines[i].replace('\n', '')
        with open(pxd_filename, 'r') as pxdfile:
            existing = pxdfile.read().split('\n')
        if len(existing) == 1 + len(total_lines_nonewlines) and existing[-1] == '':
            existing.pop(-1)
        if existing == total_lines_nonewlines:
            return
    # Update/create .pxd
    with open(pxd_filename, 'w') as pxdfile:
        pxdfile.writelines(total_lines)


# Edit the .pyx file
filename = sys.argv[1]
active_params_module = sys.argv[2][:-3]
mode = sys.argv[3]
if filename[-3:] == '.py':
    # For safety reasons
    print(filename + ' is a source code (.py) file.')
    print('You probably do not want pyxpp to edit this. Aborting.')
elif mode == 'pyx':
    import_params(filename)
    oneline(filename)
    cython_decorators(filename)
    cythonstring2code(filename)
    power2product(filename)
    unicode2ASCII(filename)
    __init__2__cinit__(filename)
    colon2zero_in_addresses(filename)
    malloc_realloc(filename)
    C_casting(filename)
elif mode == 'pxd':
    make_pxd(filename)
