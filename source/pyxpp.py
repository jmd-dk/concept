"""
This is the .pyx preprocessor script. Run it with a .pyx file as the first
argument and the parameterfile as the second, and the .pyx file will be
changed in the following ways:
- Replace 'from _params_active import *' with the content itself.
- Removes pure Python commands between 'if not cython.compiled:' and 'else:',
  including these lines themselves. Also removes the triple quotes around the
  Cython statements in the else body.
- Integer powers will be replaced by products.
- Greek letters will be replaced with ASCII-strings.
- __init__ methods in cclasses are renamed to __cinit__.
- Removes ctypedef's if it already exists in the .pxd file.
- Fixes the way the % operator works when the left operand is negative
  (so that it works as in Python). Also allows for the %= operator. Negative
  right operand still behave differently than in Python! Parentheses are
  placed around the % operator as tight as possible! This is different than
  in pure Python, where the rules seem to be rather random. One should always
  put parentheses manually, as needed. Also, only one % operator is allowed on
  a single line.
- Replaces : with 0 when taking the address of arrays.
- Replaces alloc, realloc and free with the corresponding PyMem_ functions and
  takes care of the casting from the void* to the appropriate pointer type.
- Replaced the cast() function with actual Cython syntax, e.g. <double[::1]>.

  This script is not written very elegantly, and do not leave
  the modified code in a very clean state.
"""

import sys
import re
from os.path import isfile

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
        for i, line in enumerate(pyxfile):
            if unindent and line.rstrip() != '' and line[0] != ' ':
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
            if i != purePythonsection_start and in_purePythonsection and len(line) >= indentation and line[indentation] != ' ':
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
            indices = [starstar.start() for starstar in re.finditer('\*\*', line)]
            for i, index in enumerate(indices):
                line = line[:(index + 2*i)] + ' ** ' + line[(index + 2*i) + 2:]
            # Split line into segments containing ** operators
            indices = [-2] + [starstar.start() for starstar in re.finditer('\*\*', line + '** **')]
            expressions = [line[indices[i-1] + 2:indices[i+1]] for i, index in enumerate(indices[:-2]) if i > 0]
            modified_line = ''
            # Only change nonvariable, integer powers. This also excludes **kwargs
            for ex, expression in enumerate(expressions):
                power = None
                expression_power = expression[expression.find('**') + 2:]
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
                    power_with_sign = True if expression_power.replace(' ', '')[0] in '+-' else False
                    nr_of_signs = 0
                    for i, symbol in enumerate(expression_power):
                        nr_of_signs += symbol in '+-'
                        if nr_of_signs > power_with_sign or symbol not in '+- .0123456789':
                            break
                    if i == len(expression_power) - 1:
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
                    for i, symbol in enumerate(reversed(expression_base)):
                        try:
                            base = symbol + base
                            exec(base.replace('.', '') + ' = 0')
                        except:
                            if symbol not in '.0123456789' and base != ' '*len(base):
                                break
                    base = base[1:].replace(' ', '')
                    before_base = expression_base[::-1][i:][::-1]
                # Replaces ** with a string of multiplications for integer powers
                if integer_power:
                    if nested_power and firstcall:
                        operation = expression_base + pyxpp_power + power
                    else:
                        operation = '(' + ((base + '*') * int(abs(eval(power))))[:-1] + ')'
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

def greek2ASCII(filename):
    # From http://en.wikipedia.org/wiki/Greek_letters_used_in_mathematics,_science,_and_engineering
    alphabet = {'α': 'alpha',
                'β': 'beta',
                'γ': 'gamma',
                'δ': 'delta',
                'ϵ': 'epsilon',
                'ε': 'varepsilon',
                'ɛ': 'varepsilon',  # This IS different from the above
                'ϝ': 'digamma',
                'ζ': 'zeta',
                'η': 'eta',
                'θ': 'theta',
                'ϑ': 'vartheta',
                'ι': 'iota',
                'κ': 'kappa',
                'ϰ': 'varkappa',
                'λ': 'lambda',
                'μ': 'mu',
                'ν': 'nu',
                'ξ': 'xi',
                'ο': 'omicron',
                'π': 'pi',
                'ϖ': 'varpi',
                'ρ': 'rho',
                'ϱ': 'varrho',
                'σ': 'sigma',
                'ς': 'varsigma',
                'τ': 'tau',
                'υ': 'upsilon',
                'φ': 'phi',
                'ϕ': 'varphi',
                'χ': 'chi',
                'ψ': 'psi',
                'ω': 'omega',
                'Α': 'Alpha',
                'Β': 'Beta',
                'Γ': 'Gamma',
                'Δ': 'Delta',
                'Ε': 'Epsilon',
                'Ϝ': 'Digamma',
                'Ζ': 'Zeta',
                'Η': 'Eta',
                'Θ': 'Theta',
                'Ι': 'Iota',
                'Κ': 'Kappa',
                'Λ': 'Lambda',
                'Μ': 'Mu',
                'Ν': 'Nu',
                'Ξ': 'Xi',
                'Ο': 'Omicron',
                'Π': 'Pi',
                'Ρ': 'Rho',
                'Σ': 'Sigma',
                'Τ': 'Tau',
                'Υ': 'Upsilon',
                'Φ': 'Phi',
                'Χ': 'Chi',
                'Ψ': 'Psi',
                'Ω': 'Omega'}
    new_lines = []
    with open(filename, 'r') as pyxfile:
        for line in pyxfile:
            for key, value in alphabet.items():
                line = line.replace(key, '__pyxpp_greek_' + value)
            new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)

def __init__2__cinit__(filename):
    new_lines = []
    with open(filename, 'r') as pyxfile:
        in_cclass = False
        for line in pyxfile:
            if len(line) > 13 and line[:14] == '@cython.cclass':
                in_cclass = True
            elif line[0] not in ' \n' and not (len(line) > 4 and line[:5] == 'class'):
                in_cclass = False
            if in_cclass and len(line) > 16 and line[:17] == '    def __init__(':
                line = '    def __cinit__(' + line[17:]
            new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)

def del_ctypedef_redeclarations(filename):
    pxdfilename = filename[:-3] + 'pxd'
    if isfile(pxdfilename):
        ctypedefs = []
        with open(pxdfilename, 'r') as pxdfile:
            for line in pxdfile:
                if 'ctypedef' in line:
                    ctypedefs.append(line.rstrip())
        if ctypedefs:
            new_lines = []
            with open(filename, 'r') as pyxfile:
                for line in pyxfile:
                    if line.rstrip() in ctypedefs:
                        new_lines.append('# ctypedef redeclaration. Commented by pyxpp: ' + line)
                    else:
                        new_lines.append(line)
            with open(filename, 'w') as pyxfile:
                pyxfile.writelines(new_lines)

def fix_modulus(filename):
    new_lines = []
    with open(filename, 'r') as pyxfile:
        for line in pyxfile:
            if '%=' in line:
                left_operand = line[:line.find('%=')]
                right_operand = line[line.find('%=') + 2:]
                line = left_operand + '= ' + left_operand.strip() + ' % ' + '(' + right_operand.rstrip() + ')\n'
            if '%' in line:
                right_operand = ''
                parentheses = 0
                right_end = len(line) - 1
                for i, symbol in enumerate(line[line.find('%') + 1:]):
                    if symbol == '(':
                        parentheses += 1
                        right_operand += symbol
                    elif parentheses > 0 and symbol == ')':
                        parentheses -= 1
                        right_operand += symbol
                        if parentheses == 0:
                            right_end = i + line.find('%') + 2
                            break
                    elif parentheses > 0 or symbol not in '+-*/,=)':
                        right_operand += symbol
                    else:
                        right_end = i + line.find('%') + 1
                        break
                right_operand = '(' + right_operand.strip() + ')'
                left_operand = ''
                parentheses = 0
                left_end = 0
                for i, symbol in enumerate(line[:line.find('%')][::-1]):
                    if symbol == ')':
                        parentheses += 1
                        left_operand += symbol
                    elif parentheses > 0 and symbol == '(':
                        parentheses -= 1
                        left_operand += symbol
                        if parentheses == 0:
                            left_end = -i + line.find('%') - 1
                            break
                    elif parentheses > 0 or symbol not in '+-*/=,(':
                        left_operand += symbol
                    else:
                        left_end = -i + line.find('%')
                        break
                left_operand = left_operand[::-1]
                left_operand = '(' + left_operand.strip() + ')'
                modified_line = line[:left_end] + ' (' + left_operand + ' % ' + right_operand + ' + (' + left_operand + ' < 0)*' + right_operand + ') ' + line[right_end:]
                new_lines.append(modified_line)
            else:
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
                    line = line.replace(alloc, '<' + dtype + '*> PyMem_' + alloc.capitalize())
                    new_lines.append(line)
                    # Add exception
                    LHS = line[:line.find('=')].strip()
                    indentation = len(line[:line.find('=')]) - len(line[:line.find('=')].lstrip())
                    new_lines.append(' '*indentation + 'if not ' + LHS + ':\n')
                    new_lines.append(' '*(indentation + 4) + "raise MemoryError('Could not " + alloc[:-1] + ' ' + LHS + "')\n")
            if not found_alloc:
                line = line.replace(' free(', ' PyMem_Free(')
                new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)

def C_casting(filename):
    new_lines = []
    with open(filename, 'r') as pyxfile:
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
                cast_to = '<' + line[(comma_index + 1):i].replace("'", '').replace('"', '').strip() + '>'
                obj_to_cast = '(' + line[(line.find('cast(') + 5):comma_index] + ')'
                line = line[:line.find('cast(')] + cast_to + obj_to_cast + line[(i + 1):]
            new_lines.append(line)
    with open(filename, 'w') as pyxfile:
        pyxfile.writelines(new_lines)

# Edit the .pyx file
filename = sys.argv[1]
active_params_module = sys.argv[2][:-3]
if filename[-3:] == '.py':
    # For safety reasons
    print(filename + ' is a source code (.py) file.')
    print('You probably do not want pyxpp to edit this. Aborting.')
else:
    import_params(filename)
    cythonstring2code(filename)
    power2product(filename)
    greek2ASCII(filename)
    __init__2__cinit__(filename)
    del_ctypedef_redeclarations(filename)
    fix_modulus(filename)
    colon2zero_in_addresses(filename)
    malloc_realloc(filename)
    C_casting(filename)

