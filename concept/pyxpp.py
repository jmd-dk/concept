# This Python file uses the following encoding: utf-8

# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: you can redistribute it and/or modify
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
# The auther of CO𝘕CEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



"""
This is the .pyx preprocessor script. Run it with a .py file or a
.pyx file file as the first argument. If a .pyx file is given, a .pxd
file will be created, containing all Cython declarations of classes
(@cython.cclass), functions (@cython.header, @cython.pheader,
@cython.cfunc, @cython.ccall) and variables (cython.declare()).
When a .py file is given, a .pyx copy of this file will be created,
and then changed in the following ways:
- Transform statements written over multiple lines into single lines.
  The exception is decorator statements, which remain multilined.
- Removes pure Python commands between 'if not cython.compiled:' and
  'else:', including these lines themselves. Also removes the triple
  quotes around the Cython statements in the else body. The 'else'
  clause is optional.
- Insert the lines 'cimport cython' and 'from commons cimport *' just
  below 'from commons import *'.
- Do a cimport of all @cython.cclass classes (not applied to commons.py)
- Transform the 'cimport()' function calls into proper cimports.
- Replace 'ℝ[expression]' with a double variable, which is equal to
  'expression' and defined on a suitable line.
- Replaces the cython.header and cython.pheader decorators with
  all of the Cython decorators which improves performance. The
  difference between the two is that cython.header turns into
  cython.cfunc and cython.inline (among others), while cython.pheader
  turns into cython.ccall (among others).
- Integer powers will be replaced by products.
- Unicode non-ASCII letters will be replaced with ASCII-strings.
- __init__ methods in cclasses are renamed to __cinit__.
- Replace (with '0') or remove ':' and '...' intelligently, when taking
  the address of arrays.
- Replace alloc, realloc and free with the corresponding PyMem_
  functions and take care of the casting from the void* to the
  appropriate pointer type.
- Replaced the cast() function with actual Cython syntax, e.g. 
  <double[::1]>.

  This script is not written very elegantly, and do not leave
  the modified code in a very clean state either. Sorry...
"""



# For Python 2.x compatibility
from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals
import sys
if sys.version_info.major < 3:
    from codecs import open
def non_nested_exec(s):
    exec(s)


from copy import deepcopy
import imp, itertools, os, re, shutil, unicodedata
# For development purposes only
from time import sleep


# Mapping of modules to extension types defined within (Cython classes)
extension_types = {'species':  'Particles',
                   'snapshot': 'StandardSnapshot, GadgetSnapshot',
                   }



def oneline(filename):
    in_quotes = [False, False]
    in_triple_quotes = [False, False]
    paren_counts = {'paren': 0, 'brack': 0, 'curly': 0}
    def count_parens(line):
        if line.lstrip().startswith('#'):
            return
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
            if ch == '#':
                break
            # Count parentheses outside quotes
            if not in_quotes[0] and not in_quotes[1]:
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
        return

    new_lines = []
    multiline_statement = []
    multiline = False
    multiline_decorator = False
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        for i, line in enumerate(pyxfile):
            count_parens(line)
            if (paren_counts['paren'] > 0 or
                paren_counts['brack'] > 0 or
                paren_counts['curly'] > 0) and not multiline:
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
            elif (paren_counts['paren'] > 0 or
                  paren_counts['brack'] > 0 or
                  paren_counts['curly'] > 0) and multiline:
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
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines(new_lines)



def cimport_commons(filename):
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        lines = pyxfile.read().split('\n')
    for i, line in enumerate(lines):
        if line.startswith('from commons import *'):
            lines = lines[:(i + 1)] + ['cimport cython', 'from commons cimport *'] + lines[(i + 1):]
            break
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines('\n'.join(lines))



def cimport_cclasses(filename):
    # Do not import cclasses into the commons module
    if filename == 'commons.pyx':
        return
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        lines = pyxfile.read().split('\n')
    insert_on_line = -1
    for i, line in enumerate(lines):
        if line.startswith('from commons cimport *'):
            insert_on_line = i
            break
    cimports = []
    for key, val in extension_types.items():
        if filename != key + '.pyx':
            cimports.append('from {} cimport {}'.format(key, val))
    lines = lines[:(insert_on_line + 1)] + cimports + lines[(insert_on_line + 1):]
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines('\n'.join(lines))



def cimport_function(filename):
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        lines = pyxfile.read().split('\n')
    for i, line in enumerate(lines):
        if line.lstrip().startswith('cimport'):
            lines[i] = re.sub('cimport.*\((.*?)\)', lambda match: match.group(1).replace('import ', 'cimport ').replace("'", '').replace("'", ''), line)
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines('\n'.join(lines))



def cythonstring2code(filename):
    new_lines = []
    with open(filename, 'r', encoding='utf-8') as pyxfile:
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
                    line_without_triple_quotes = line
                    if line.startswith(' '*(indentation + 4) + '"""') or line.startswith(' '*(indentation + 4) + "'''"):
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
    with open(filename, 'w', encoding='utf-8') as pyxfile:
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
                    parentheses = 0
                    done = False
                    completed = False
                    symbol_before_paren = ''
                    for symbol in reversed(expression_base.replace(' ', '')):
                        if completed:
                            symbol_before_paren = symbol
                            break
                        if symbol == '(':
                            parentheses += 1
                        elif symbol == ')':
                            parentheses -= 1
                            done = True
                        if parentheses == 0 and done:
                            completed = True
                    if symbol_before_paren and symbol_before_paren not in ('+-*/%&|^@()='):
                        # Base is really a function input, like sin(a)**2. Do nothing
                        modified_line = modified_line if modified_line else line
                        continue
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
                    base = expression_base[::-1][0:(i + 1)][::-1]
                    before_base = expression_base[::-1][(i + 1):][::-1]
                    before_base_rstrip = before_base.rstrip()
                    if before_base_rstrip and before_base_rstrip[-1].lower() not in ('+-*/%&|^@()='):
                        # Base is really a function input, like sin(a)**2. Do nothing
                        modified_line = modified_line if modified_line else line
                        continue
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
                            non_nested_exec(base.replace('.', '') + ' = 0')
                        except:
                            if symbol not in '.0123456789' and base != ' '*len(base):
                                break
                    base = base[1:].replace(' ', '') + brackets_width_content
                    before_base = expression_base[::-1][i:][::-1]
                # Replaces ** with a string of multiplications for integer powers
                if integer_power:
                    if nested_power and firstcall:
                        operation = expression_base + pyxpp_power + power
                    else:
                        operation = ('(' + ((base + '*')*int(abs(eval(power))))[:-1] + ')')
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
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        for line in pyxfile:
            # Replace 'a **= b' with 'a = a**b'
            if '**=' in line:
                LHS = line[:line.find('**=')]
                RHS = line[(line.find('**=') + 3):].strip()
                if '#' in RHS:
                    RHS = RHS[:RHS.find('#')].rstrip()
                line = LHS + '= ' + LHS.strip() + '**(' + RHS + ')'
            # Replace ** --> pyxpp_power
            line = line.replace('**', ' ' + pyxpp_power + ' ')
            # Integer power --> products
            line = pow2prod(line)
            while pyxpp_power in line:
                line = pow2prod(line, firstcall=False)
            new_lines.append(line)
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines(new_lines)



def unicode2ASCII(filename):
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        text = [char for char in pyxfile.read()]
    for i, char in enumerate(text):
        if ord(char) > 127:
            text[i] = '__UNICODE__' + unicodedata.name(char)
            text[i] = text[i].replace(' ', '__space__')
            text[i] = text[i].replace('-', '__dash__')
    text = ''.join(text)
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.write(text)
    return



def __init__2__cinit__(filename):
    new_lines = []
    with open(filename, 'r', encoding='utf-8') as pyxfile:
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
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines(new_lines)



def fix_addresses(filename):
    new_lines = []
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        for line in pyxfile:
            # 'address(' to 'cython.address'
            if 'address(' in line:
                line = line.replace('address(', 'cython.address(')
                line = line.replace('cython.cython.', 'cython.')
            # cython.address(a[7, ...]) to cython.address(a[7, 0])
            # cython.address(a[7, :, 1]) to cython.address(a[7, 0, 1])
            # cython.address(a[7, 9:, 1]) to cython.address(a[7, 9, 1])
            colons_or_ellipsis = True
            while 'cython.address(' in line and colons_or_ellipsis:
                parens = 0
                address_index = line.find('cython.address(') + 14
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
                colons_or_ellipsis = (':' in addressof or '...' in addressof)
                line = (line[:(address_index + 1)] + addressof + line[(address_index + i):])
            new_lines.append(line)
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines(new_lines)



def malloc_realloc(filename):
    new_lines = []
    with open(filename, 'r', encoding='utf-8') as pyxfile:
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
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines(new_lines)


def C_casting(filename):
    new_lines = []
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        # Transform to Cython syntax
        for line in pyxfile:
            while 'cast(' in line:
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
                line = (line[:line.find('cast(')] + '(' + cast_to + obj_to_cast + ')'
                        + line[(i + 1):])
            new_lines.append(line)
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines(new_lines)


def cython_decorators(filename):
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        lines = pyxfile.read().split('\n')
    for i, line in enumerate(lines):
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
                    if '@cython.' + headertype + '(' in hline:
                        I = header[j].index('@cython.' + headertype + '(') + 15
                        for k, c in enumerate(header[j][I:]):
                            if c == ',':
                                header[j] = header[j][:I] + header[j][(I + k + 1):]
                                break
                            elif c != ' ':
                                break
                # Change @cython.header to @cython.locals, if header contains declarations.
                # Otherwise, remove it.
                if '=' in ''.join(header):
                    header[0] = header[0].replace(headertype, 'locals')
                else:
                    header = []
                # Add in all the other decorators
                pyfuncs = ('__init__', '__cinit__', '__dealloc__')
                decorators = [decorator for decorator in
                              (('ccall' if headertype == 'pheader' else 'cfunc')
                               if all(' ' + pyfunc + '(' not in def_line
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
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines('\n'.join(lines))



def make_pxd(filename):
    customs = {# Classes
               'Particles':        'from species cimport Particles',
               'StandardSnapshot': 'from snapshot cimport StandardSnapshot',
               'GadgetSnapshot':   'from snapshot cimport GadgetSnapshot',
               # Function pointers
               'func_b_ddd':   'ctypedef bint '    + '(*func_b_ddd_pxd)'   + '(double, double, double)',
               'func_d_dd':    'ctypedef double '  + '(*func_d_dd_pxd)'    + '(double, double)',
               'func_d_ddd':   'ctypedef double '  + '(*func_d_ddd_pxd)'   + '(double, double, double)',
               'func_ddd_ddd': 'ctypedef double* ' + '(*func_ddd_ddd_pxd)' + '(double, double, double)',
               # External definitions
               'fftw_plan':          ('cdef extern from "fft.c":\n'
                                      '    ctypedef struct fftw_plan_struct:\n'
                                      '        pass\n'
                                      '    ctypedef fftw_plan_struct *fftw_plan'),
               'fftw_return_struct': ('cdef extern from "fft.c":\n'
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
                                      '        fftw_plan plan_backward'),
               }
    header_lines = []
    pxd_filename = filename[:-3] + 'pxd'
    pxd_lines = []
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        code = pyxfile.read().split('\n')
    # Find pxd hints of the form 'pxd = """'
    #                             int var1
    #                             double var2
    #                             """'
    pxd_lines.append('# pxd hints\n')
    for i, line in enumerate(code):
        if line.replace(' ', '').startswith('pxd="""') or line.replace(' ', '').startswith("pxd='''"):
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
    # Import all types with spaces (e.g. "long int") from commons.py
    types_with_spaces = [(key.replace(' ', ''), key) for key in imp.load_source('commons', 'commons.py').C2np.keys() if ' ' in key]
    types_with_spaces = sorted(types_with_spaces, key=lambda t: len(t[1]), reverse=True)
    # Function that finds non-indented function definitions in a block
    # of code (list of lines). It appends to header_lines and pxd_lines.
    def find_functions(code, indent=0, only_funcname=None):
        for i, line in enumerate(code):
            if line.startswith('def '):
                # Function definition found.
                # Find out whether cdef (cfunc) of cpdef (ccall) function
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
                # If searching for a specific function name and this is
                # not it, continue.
                if only_funcname and only_funcname != function_name:
                    continue
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
                                        # Add suffix _pxd to the "func_" types
                                        if argtype.replace('*', '') in customs:
                                            header_lines.append(customs[argtype.replace('*', '')])
                                        if 'func_' in argtype:
                                            argtype += '_pxd'
                                            argtype = argtype.replace('*', '') + '*'*argtype.count('*')
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
                # Due to a bug in the above, the last argument can
                # sometimes include the closing parenthesis.
                # Remove this
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
            class_code_unindented = [line if len(line) < 4 else line[4:] for line in code[(l + k + j + j0 + i + 4):(m + l + k + j + j0 + i + 4 )]]
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
                                                   'def ' + globals_phony_funcname + declaration + ':',
                                                   '    pass'] + globals_code[(i + 1):])
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
                    pxd_lines = pxd_lines[:(variable_index + i)] + lines + pxd_lines[(variable_index + i + 1):]
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
    # If nothing else, place a comment in the pxd file
    if not total_lines:
        total_lines = ['# This module does not expose any c-level functions or classes to the outside world\n']
    # Update/create .pxd
    with open(pxd_filename, 'w', encoding='utf-8') as pxdfile:
        pxdfile.writelines(total_lines)



def constant_expressions(filename):
    # Find constant expressions using the ℝ['expression'] syntax
    with open(filename, 'r', encoding='utf-8') as pyxfile:
        lines = pyxfile.read().split('\n')
    expressions = []
    expressions_cython = []
    declaration_linenrs = []
    operations = ('.', '+', '-', '**', '*', '/', '^', '&', '|', '@', ',', '(', ')', '[', ']', '{', '}')
    operations_names = ('dot', 'pls', 'min', 'pow', 'tim', 'div', 'car', 'and', 'bar', 'at', 'com', 'opar', 'cpar', 'obra', 'cbra', 'ocur', 'ccur')
    while True:
        no_blackboard_bold_R = True
        for i, line in enumerate(lines):
            search = re.search('ℝ\[(.*?)\]', line)
            if not search or line.replace(' ', '').startswith('#'):
                continue
            no_blackboard_bold_R = False
            expression = search.group(1)
            expressions.append(expression)
            expression_cython = 'ℝ_' + expression.replace(' ', '')
            for op, op_name in zip(operations, operations_names):
                expression_cython = expression_cython.replace(op, '_{}_'.format(op_name))
            expression_cython = expression_cython.replace('__', '_')
            expressions_cython.append(expression_cython)
            lines[i] = line.replace(search.group(0), expression_cython)
            # Find out where the declaration should be
            variables = [expression.replace(' ', '')]
            for op in operations:
                variables = list(itertools.chain(*[var.split(op) for var in variables]))
            variables = [var for var in list(set(variables)) if var and var[0] not in '.0123456789']
            linenr_where_defined = [-1]*len(variables)
            for v, var in enumerate(variables):
                for j, line2 in enumerate(reversed(lines[:(i + 1)])):
                    line2 = ' '*(len(line2) - len(line2.lstrip()))  + line2.replace(' ', '')
                    for op in operations:
                        line2 = line2.replace(op, '')
                    if ' ' + var + '=' in line2 or ',' + var + '=' in line2 or ';' + var + '=' in line2 or '=' + var + '=' in line2 or line2.startswith(var + '='):
                        linenr_where_defined[v] = i - j
                        break
            if linenr_where_defined:
                declaration_linenrs.append(max(linenr_where_defined))
            else:
                declaration_linenrs.append(-1)
            # Remove again if duplicate
            for j in range(len(expressions) - 1):
                if expressions[j] == expressions[-1] and declaration_linenrs[j] == declaration_linenrs[-1]:
                    expressions.pop()
                    expressions_cython.pop()
                    declaration_linenrs.pop()
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
                if len(line) > 0 and line[0] not in '# ' and not line.startswith('"""') and not line.startswith("'''"):
                    linenr_unrecognized = i + j
                    break
    # Insert Cython declarations of constant expressions
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        # Unrecognized definitions
        if i == linenr_unrecognized:
            for e, expression_cython in enumerate(expressions_cython):
                if declaration_linenrs[e] == -1:
                    new_lines.append('cython.declare(' + expression_cython + "='double')")
                    new_lines.append(expression_cython + ' = ' + expressions[e])
            new_lines.append('')
        for e, n in enumerate(declaration_linenrs):
            if i == n:
                indentation = ' '*(len(lines[i - 1]) - len(lines[i - 1].lstrip()))
                new_lines.append(indentation + 'cython.declare(' + expressions_cython[e] + "='double')")
                new_lines.append(indentation + expressions_cython[e] + ' = ' + expressions[e])
    with open(filename, 'w', encoding='utf-8') as pyxfile:
        pyxfile.writelines('\n'.join(new_lines))



# Interpret the input argument
filename = sys.argv[1]
if filename.endswith('.py'):
    # A .py-file is passed. Copy file to .pyx and work with this copy
    filename_pyx = filename[:-2] + 'pyx'
    shutil.copy(filename, filename_pyx)
    filename = filename_pyx
    # Actions
    oneline(filename)
    cythonstring2code(filename)
    cimport_commons(filename)
    cimport_cclasses(filename)
    cimport_function(filename)
    constant_expressions(filename)
    cython_decorators(filename)
    power2product(filename)
    unicode2ASCII(filename)
    __init__2__cinit__(filename)
    fix_addresses(filename)
    malloc_realloc(filename)
    C_casting(filename)
elif filename.endswith('.pyx'):
    # A .pyx-file is passed. Make the pxd
    make_pxd(filename)
else:
    raise Exception('Got "{}", which is neither a .py nor a .pyx file'.format(filename))
