# Define the encoding, for Python 2 compatibility:
# This Python file uses the following encoding: utf-8

# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2021 Jeppe Mosgaard Dakin.
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
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



"""
This is the .pyx preprocessor script.
It can be run with the following three sets of arguments,
all arguments being filenames except the optional --no-optimizations:
- module.py commons.py [--no-optimizations]
  Creates module.pyx, a version of module.py with cython-legal and
  optimized syntax.
- .types.pyx commons.py .types.pyx module0.pyx module1.pyx ...
  Creates .types.pyx, a file containing imports for all extension
  classes from module0.pyx, module1.pyx, ..., together with globally
  defined types.
- module.pyx commons.py .types.pyx
  Created module.pxd, the cython header for module.pyx.

In the first case where a .pyx file is created from a .py file,
the following changes happens to the source code (in the .pyx file):
- Insert the line 'cimport cython' at the very top,
  though below any __future__ imports.
- Transform statements written over multiple lines into single lines.
  The exception is decorator statements, which remain multi-lined.
- Removes pure Python commands between 'if not cython.compiled:' and
  'else:', including these lines themselves. Also removes the triple
  quotes around the Cython statements in the else body. The 'else'
  clause is optional.
- Code blocks within copy_on_import context managers will be copied
  over when * importing.
- Calls to build_struct will be replaced with specialized C structs
  which are declared dynamically from the call. Type declarations
  of this struct, its fields and its corresponding dict are inserted.
- Insert the line 'from commons cimport *'
  just below 'from commons import *'.
- Transform the 'cimport()' function calls into proper cimports.
- Replaces calls to iterator functions decorated with @cython.iterator
  with the bare source code in these functions, effectively inlining
  the call.
- Replace 'ℝ[expression]' with a double variable, 'ℤ[expression]' with a
  Py_ssize_t variable and '𝔹[expression]' with a bint variable which is
  equal to 'expression' and defined on a suitable line.
- Unicode non-ASCII letters will be replaced with ASCII-strings.
- Loop unswitching is performed on if statements under an unswitch
  context manager, which are indented under one or more loops.
- Replaces the cython.header and cython.pheader decorators with
  all of the Cython decorators which improves performance. The
  difference between the two is that cython.header turns into
  cython.cfunc (among others), while cython.pheader
  turns into cython.ccall (among others).
- __init__ methods in cclasses are renamed to __cinit__.
- Replace (with '0') or remove ':' and '...' intelligently, when taking
  the address of arrays.
- Replace alloc, realloc and free with the corresponding PyMem_
  functions and take care of the casting from the void* to the
  appropriate pointer type.
- Replaced the cast() function with actual Cython syntax, e.g.
  <double[::1]>.
- A comment will be added to the end of the file, listing all the
  implemented extension types within the file.

This script is not written very elegantly, and do not leave
the modified code in a very clean state either. Sorry...
"""



# General imports
import ast, collections, contextlib, copy, functools, importlib, inspect, itertools
import keyword, os, re, subprocess, sys, unicodedata
# For math
import numpy as np



# Function for importing a *.py module,
# even when a *.so module of the same name is present.
def import_py_module(module_name):
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    with disable_loader('.so'):
        module = importlib.import_module(module_name)
    return module
@contextlib.contextmanager
def disable_loader(ext):
    ext = '.' + ext.lstrip('.')
    # Push any loaders for the ext extension to the back
    edits = collections.defaultdict(list)
    path_importer_cache = list(sys.path_importer_cache.values())
    for i, finder in enumerate(path_importer_cache):
        loaders = getattr(finder, '_loaders', None)
        if loaders is None:
            continue
        for j, loader in enumerate(loaders):
            if j + len(edits[i]) == len(loaders):
                break
            if loader[0] != ext:
                continue
            # Loader for the ext extension found.
            # Push to the back.
            loaders.append(loaders.pop(j))
            edits[i].append(j)
    try:
        # Yield control back to the caller
        yield
    finally:
        # Undo changes to path importer cache
        for i, edit in edits.items():
            loaders = path_importer_cache[i]._loaders
            for j in reversed(edit):
                loaders.insert(j, loaders.pop())



def cimport_cython(lines, no_optimization):
    for i, line in enumerate(lines):
        if (line.strip()
            and not line.lstrip().startswith('#')
            and '__future__' not in line
            ):
            lines = lines[:i] + ['cimport cython\n'] + lines[i:]
            break
    return lines



def remove_functions(lines, no_optimization):
    new_lines = []
    remove = False
    def_encountered = False
    for line in lines:
        if remove:
            if not def_encountered:
                if line.startswith('def '):
                    def_encountered = True
                continue
            if (
                line.strip()
                and not line.startswith(' ')
                and not line.startswith('#')
                and not line.startswith('"')
                and not line.startswith("'")
            ):
                # Outside function to be removed
                remove = False
                new_lines.append(line)
                continue
            # Do not include this line
            # unless a left-aligned comment.
            if line.startswith('#'):
                new_lines.append(line)
            continue
        # Check for @cython.remove
        match = re.search(r'^@cython *\. *remove *(.*)', line)
        if not match:
            new_lines.append(line)
            continue
        remove = True
        def_encountered = False
        arg = match.group(1)
        arg = re.search(r'^\(.*\)', arg)
        if arg:
            arg = arg.group()
            if arg:
                arg = eval(arg)
                if not arg:
                    # Encountered @cython.remove(False)
                    remove = False
                    continue
        # Encountered @cython.remove.
        # Pop previous decorators from new_lines.
        while new_lines[-1].startswith('@'):
            new_lines.pop()
        new_lines.append('\n')
    return new_lines



def walrus(lines, no_optimization):
    """Remove this function once Python 3.8 assignment expressions get
    implemented in Cython. See
      https://github.com/cython/cython/pull/3691
    """
    new_lines = []
    for line in lines:
        if ':=' in line:
            varname = re.search(r'\( *([a-zA-Z_][a-zA-Z0-9_]*) *:=', line).group(1)
            index_bgn = line.index(':=') + 2
            for index_end in range(len(line), index_bgn + 1, -1):
                expression = line[index_bgn:index_end].strip()
                try:
                    parsed = ast.parse(expression)
                except:
                    continue
                if len(parsed.body) == 1:
                    break
            else:
                print(
                    f'Could not perform substitution of walrus operator '
                    f'in the following line:\n"{line}"',
                    file=sys.stderr,
                )
                sys.exit(1)
            line = re.sub(rf'\( *{varname} *:=', '( ', line)
            line = line.replace(varname, f'({expression})')
        new_lines.append(line)
    return new_lines



def copy_on_import(lines, no_optimization):
    workfile = filename.removesuffix('.py').removesuffix('.pyx')
    def get_new_funcname(funcname):
        return funcname + f'__{workfile}__'
    funcnames = []
    new_lines = []
    unindent = False
    for i, line in enumerate(lines):
        # Remove copy_on_import context manager
        # and unindent block.
        if re.search(r'^ *with +copy_on_import *:', line):
            unindent = True
            indentation_lvl = len(line) - len(line.lstrip())
            continue
        elif unindent:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                indentation_lvl_new = len(line) - len(line.lstrip())
                if indentation_lvl_new <= indentation_lvl:
                    # End of context manager
                    unindent = False
        if unindent and line[:4] == ' '*4:
            new_lines.append(line[4:])
            continue
        new_lines.append(line)
        # Handle * (c)imports of modules containing copy_on_import
        match = re.search(r'^ *from +(.+) +c?import +\*', line)
        if not match:
            match = re.search(r'^ *cimport *\(["\']+ *from +(.+) +c?import +\* *["\']+\)', line)
            if not match:
                continue
        module_name = match.group(1).strip()
        try:
            module = import_py_module(module_name)
        except ImportError:
            continue
        lines_imported = inspect.getsourcelines(module)[0]
        import_indentation = len(line) - len(line.lstrip())
        blocks = []
        in_block = False
        for line_imported in lines_imported:
            if re.search(r'^ *with +copy_on_import *:', line_imported):
                in_block = True
                block = []
                blocks.append(block)
                indentation_lvl_imported = len(line_imported) - len(line_imported.lstrip())
                continue
            elif in_block:
                line_imported_stripped = line_imported.strip()
                if line_imported_stripped:
                    indentation_lvl_imported_new = len(line_imported) - len(line_imported.lstrip())
                    if indentation_lvl_imported_new <= indentation_lvl_imported:
                        if line_imported_stripped.startswith('#'):
                            # Leave unindented comment out of block
                            continue
                        else:
                            # End of context manager
                            in_block = False
            if in_block:
                block.append(line_imported)
        for block in blocks:
            if not block:
                continue
            block_new = []
            indentation_lvl_block = len(block[0]) - len(block[0].lstrip())
            funcnames_block = []
            for line_block in block:
                line_block_stripped = line_block.strip()
                if line_block_stripped and not line_block_stripped.startswith('#'):
                    match = re.search(r'( *def +)(.+?)( *\()', line_block)
                    if match:
                        funcname = match.group(2)
                        if funcname not in funcnames_block:
                            funcnames_block.append(funcname)
                        if funcname in funcnames:
                            print(f'Warning: Function {funcname}() appears twice in {workfile} due to copy_on_import', file=sys.stderr)
                        # Substitute name of function in definition
                        line_block = re.sub(
                            r'( *def +)(.+?)( *\()',
                            lambda m: m.group(1) + get_new_funcname(m.group(2)) + m.group(3),
                            line_block,
                        )
                if line_block.startswith(' '*indentation_lvl_block):
                    line_block = line_block[indentation_lvl_block:]
                line_block = ' '*import_indentation + line_block
                block_new.append(line_block)
            funcnames += funcnames_block
            new_lines.append('\n')
            new_lines.append(' '*import_indentation + f'# Copied from the {module_name} module\n')
            block = block_new
            block_new = []
            for j, line_block in enumerate(block):
                if line_block.strip():
                    block_new += block[j:]
                    break
            for line_block in reversed(block_new):
                if line_block.strip():
                    break
                block_new.pop()
            for line_block in block_new:
                new_lines.append(line_block)
            new_lines.append('\n')
    new_lines = commons.onelinerize(new_lines)
    # Substitute name of functions used throughout the module,
    # if defined within the copy block.
    unused_funcs = funcnames.copy()
    patterns = [rf'(^|[^\w.])({funcname})($|\W)' for funcname in funcnames]
    for i, line in enumerate(new_lines):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('#'):
            continue
        for funcname, pattern in zip(funcnames, patterns):
            if not re.search(pattern, line_stripped):
                continue
            line_nostr = line_stripped
            line_nostr = re.subn(r'\'.*?\'', '', line_nostr)[0]
            line_nostr = re.subn(r'".*?"', '', line_nostr)[0]
            if not re.search(pattern, line_nostr):
                continue
            line_modified, n = re.subn(
                pattern,
                lambda m: m.group(1) + get_new_funcname(m.group(2)) + m.group(3),
                line,
            )
            if n > 0:
                new_lines[i] = line_modified
                # Is the function used or merely defined using =?
                if not re.search(rf'(^|[^\w.]){funcname} *=', line):
                    if funcname in unused_funcs:
                        unused_funcs.remove(funcname)
    new_lines = commons.onelinerize(new_lines)
    # Remove copied function definitions if not used
    for funcname in unused_funcs:
        pattern = rf' *def +{get_new_funcname(funcname)} *\('
        lines = new_lines
        new_lines = []
        in_unused_function = False
        decorator_begin = -1
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('@'):
                if decorator_begin == -1:
                    decorator_begin = i
            match = re.search(pattern, line)
            if not match and line_stripped.startswith('def '):
                decorator_begin = -1
            if match:
                in_unused_function = True
                indentation_level = len(line) - len(line.lstrip())
                # Unused function found. Remove decorators above.
                if decorator_begin != -1:
                    for _ in range(i - decorator_begin):
                        new_lines.pop()
                new_lines.append(' '*indentation_level + 'pass\n')
                continue
            elif in_unused_function:
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                indentation_level_new = len(line) - len(line.lstrip())
                if indentation_level_new <= indentation_level:
                    in_unused_function = False
            if not in_unused_function:
                new_lines.append(line)
    return new_lines



def format_pxdhints(lines, no_optimization):
    """Change pxd(...) to pxd = ...
    """
    new_lines = []
    skip = 0
    for i, line in enumerate(lines):
        if skip > 0:
            skip -= 1
            continue
        if re.search(r'^pxd *\(', line):
            if line.count('(') == line.count(')'):
                new_lines.append(re.sub(r'^pxd *\((.*)\)', r'pxd = \g<1>', line))
            else:
                # Multi-line via triple quotes
                new_lines.append(re.sub(r'^pxd *\(', 'pxd = ', line))
                quote_type = '"""' if '"""' in line else "'''"
                quote_ended = False
                for line in lines[i + 1:]:
                    skip += 1
                    if quote_type in line:
                        quote_ended = True
                    if quote_ended and ')' in line:
                        new_lines.append(line.replace(')', ''))
                        break
                    else:
                        new_lines.append(line)
        else:
            new_lines.append(line)
    return new_lines



def cythonstring2code(lines, no_optimization):
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



def cython_structs(lines, no_optimization):
    # Function which returns a copy of the build_struct function
    # from commons.py:
    def get_build_struct():
        build_struct_code = []
        with open(f'{commons_name}.py', mode='r', encoding='utf-8') as commonsfile:
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
            # as triples of strings in struct_content.
            struct_content = [
                part.replace('�', '==') for part in
                line[(line.index('(') + 1):line.rindex(')')].replace('==', '�').split('=')
            ]
            for i, part in enumerate(struct_content[:-1]):
                # The field name
                if i == 0:
                    name = part
                else:
                    name = part[(part.rindex(',') + 1):].strip()
                decl = struct_content[i + 1][:struct_content[i + 1].rindex(',')]
                if re.search(r'\(.*,', decl.replace(' ', '')):
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
                else:
                    # Only type given. Initialise pointer type to None,
                    # non-pointer type to 0.
                    if decl.count('"') == 2:
                        ctype = re.search('(".*")', decl).group(1)
                    if decl.count("'") == 2:
                        ctype = re.search("('.*')", decl).group(1)
                    ctype = ctype.replace('"', '').replace("'", '').strip()
                    value = 'b""' if '*' in ctype else '0'
                struct_content[i] = (name.strip(), ctype.strip(), value.strip())
            struct_content.pop()
            # The name of the struct type is eg. struct_double_double_int
            struct_kind = '_'.join([t[1] for t in struct_content]).replace('*', 'star')
            # Insert modified version of the build_struct function,
            # initialising all pointer values to None
            # and non-pointer values to 0.
            if not build_struct_code:
                build_struct_code = get_build_struct()
            for build_struct_line in build_struct_code:
                build_struct_line = build_struct_line.replace('build_struct(',
                                                            'build_struct_{}('.format(struct_kind))
                build_struct_line = build_struct_line.replace('...', 'struct_{}({})'.format(
                    struct_kind,
                    ', '.join(['b""' if '*' in struct_content_i[1] else '0'
                        for struct_content_i in struct_content])
                ))
                new_lines.append(build_struct_line)
            # Insert declaration of struct
            indentation = len(line) - len(line.lstrip())
            new_lines.append(' '*indentation + 'cython.declare({}=struct_{})\n'
                                               .format(varnames[0], struct_kind))
            # Insert declaration of dict
            if len(varnames) == 2:
                new_lines.append(' '*indentation + "cython.declare({}=dict)\n"
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



def cimport_commons(lines, no_optimization):
    for i, line in enumerate(lines):
        if line.startswith('from {} import *'.format(commons_name)):
            lines = (  lines[:(i + 1)]
                     + ['from {} cimport *\n'.format(commons_name)]
                     + lines[(i + 1):])
            break
    return lines



def cimport_function(lines, no_optimization):
    def construct_cimport(module, function=None):
        # Add normal import enclosed in try/except,
        # followed by the cimport.
        module = module.strip()
        if function is None:
            return [
                f'try:\n',
                f'    import {module}\n',
                f'except:\n',
                f'    pass\n',
                f'cimport {module}\n',
            ]
        else:
            function = function.strip()
            return [
                f'try:\n',
                f'    from {module} import {function}\n',
                f'except:\n',
                f'    pass\n',
                f'from {module} cimport {function}\n',
            ]
    def handle_iterator(line, new_lines):
        match = re.search(r'^ *from +(.+) +cimport +(.+)', line)
        if not match:
            return
        filename = match.group(1).strip()
        try:
            module = import_py_module(filename)
        except ImportError:
            return
        funcnames = [
            funcname.strip(' \n')
            for funcname in match.group(2).strip().split(',')
            if funcname.strip(' \n')
        ]
        inline_iterators_found = collections.defaultdict(set)
        for funcname in funcnames:
            try:
                iterator_lines = inspect.getsourcelines(module.__dict__[funcname])[0]
            except:
                continue
            if not iterator_lines[0].startswith('@'):
                continue
            iterator_lines = commons.onelinerize(iterator_lines)
            for j, iterator_line in enumerate(iterator_lines):
                iterator_line_stripped = iterator_line.strip(' \n')
                if not iterator_line_stripped.startswith('@'):
                    break
                if not iterator_line_stripped.startswith('@cython.iterator'):
                    continue
                # Cython inlinable iterator found
                inline_iterators_found[filename].add(funcname)
                # Remove the "depends" argument
                depends = []
                if '(' in iterator_line_stripped:
                    iterator_arg = iterator_line_stripped[
                        iterator_line_stripped.index('(') + 1:-1
                    ].strip()
                    try:
                        depends = eval(iterator_arg)
                    except:
                        pass
                    if not depends:
                        tmp_dict = {}
                        exec(iterator_arg, tmp_dict)
                        depends = tmp_dict['depends']
                    if isinstance(depends, str):
                        depends = [depends]
                    depends = list(depends)
                    iterator_lines[j] = '@cython.iterator\n'
                # Copy the source code
                new_lines.append('\n')
                new_lines.append(
                    f'# The Cython iterator "{funcname}" below '
                    f'is copied from "{filename}.py"\n'
                )
                new_lines += iterator_lines
                new_lines.append('\n')
                # Insert import for dependencies
                if not depends:
                    break
                new_lines.append('pass\n')  # This ensures that the comment below stays
                new_lines.append(
                    f'# The Cython iterator "{funcname}" depends upon '
                    + ', '.join([f'"{depend}"' for depend in depends])
                    + f', which we import from {filename} below\n'
                )
                for depend in depends:
                    new_lines += construct_cimport(filename, depend)
                new_lines.append('\n')
                break
        return inline_iterators_found
    new_lines = []
    for i, line in enumerate(lines):
        if line.replace(' ', '').startswith('cimport('):
            if ' as ' in line:
                print(
                    'The "as" keyword is not allowed in the argument to cimport()',
                    file=sys.stderr,
                )
                sys.exit(1)
            line = re.sub(
                r'cimport.*\((.*?)\)',
                lambda match: eval(match.group(1)).replace('import ', 'cimport '),
                line
            ).rstrip(' ,\n')
            line = f'{line}\n'
            # Check for cimported Cython inlinable iterator
            # and copy it if found.
            inline_iterators_found = handle_iterator(line, new_lines)
            # Add cimport
            if line.strip().startswith('cimport '):
                # Module import
                modules = re.search(r'^cimport +(.+)', line).group(1).split(',')
                for module in modules:
                    new_lines += construct_cimport(module)
            else:
                # Function (or other object) import from within module
                match = re.search(r'^from +(.+) +cimport +(.+)', line)
                module = match.group(1)
                functions = match.group(2).split(',')
                for function in functions:
                    function = function.strip()
                    if (
                        inline_iterators_found is None
                        or function not in inline_iterators_found[module]
                    ):
                        new_lines += construct_cimport(module, function)
            new_lines.append('\n')
        else:
            new_lines.append(line)
    return new_lines



def float_literals(lines, no_optimization):
    if no_optimization:
        return lines
    # Straightforward literals
    literals = {
        'π'        : commons.π,
        'τ'        : commons.τ,
        'machine_ϵ': commons.machine_ϵ,
    }
    # Add other literals if supported
    legal_literals = check_float_literals()
    pylit_to_clit = {
        'NaN': 'NAN',
        'ထ'  : 'INFINITY',
    }
    for pylit, clit in pylit_to_clit.items():
        if clit in legal_literals:
            literals[pylit] = clit
    # Encapsulate literals in parentheses
    for name, value in literals.copy().items():
        for fun in commons.asciify, commons.unicode:
            literals[fun(name)] = f'({value})'
    # Swap variables for literals
    new_lines = []
    for line in lines:
        if not line.lstrip().startswith('#'):
            for name, value in literals.items():
                while True:
                    for match in re.finditer(name, line):
                        line_modified = line.replace('==', ' +')
                        index_equal = -1
                        if '=' in line_modified:
                            index_equal = len(line_modified) - 1 - line_modified[::-1].index('=')
                        # Check that name is not used
                        # as a subname of an identifier.
                        index_bgn, index_end = match.span()
                        if index_bgn > 0:
                            if line[index_bgn-1:index_end].isidentifier():
                                continue
                        if index_end < len(line):
                            if line[index_bgn:index_end+1].isidentifier():
                                continue
                        # Check that name is not assigned to
                        if index_bgn < index_equal:
                            continue
                        # Check that the name is not used inside a string
                        quote = ''
                        for c in line[:index_bgn]:
                            if c in ('"', "'"):
                                if not quote:
                                    quote = c
                                elif c == quote:
                                    quote = ''
                        if quote:
                            continue
                        # Do replacement
                        line = line[:index_bgn] + value + line[index_end:]
                        break
                    else:
                        break
        new_lines.append(line)
    return new_lines
@functools.lru_cache
def check_float_literals():
    legal_literals = []
    try:
        completed_process = subprocess.run(
            ['make', 'check-float-literals'],
            capture_output=True,
        )
        legal_literals = completed_process.stdout.decode().split('\n')
    except:
        pass
    legal_literals = [
        legal_literal
        for legal_literal in legal_literals
        if legal_literal.strip()
    ]
    return tuple(legal_literals)



def inline_iterators(lines, no_optimization):
    # We need to import the *.py file given by the global
    # "filename" variable, and then investigate its content.
    module = import_py_module(filename)
    # Function for processing the source lines
    # of an inline iterator function.
    def process_inline_iterator_lines(func_name, iterator_lines):
        if func_name in cached_iterators:
            t = cached_iterators[func_name]
            return t[0].copy(), t[1]
        # Remove the function definition lines(s)
        def_encountered = False
        parens = 0
        parens_any = False
        for i, iterator_line in enumerate(iterator_lines):
            if iterator_line.lstrip().startswith('#'):
                continue
            parens += iterator_line.count('(')
            if parens:
                parens_any = True
            parens -= iterator_line.count(')')
            if iterator_line.lstrip().startswith('def '):
                def_encountered = True
            if not def_encountered:
                continue
            if iterator_line.rstrip().endswith(':') and parens == 0 and parens_any:
                break
        definition = ''.join(iterator_lines[:i+1])
        iterator_lines = iterator_lines[i+1:]
        # Extract function signature.
        # Note that this does not work whenever a global variable is
        # referenced as part of the definition of a default value.
        for line in definition.split('\n'):
            if line.startswith('def '):
                break
        def_str = re.sub(r'^def +.+?\(', 'def func(', line).strip()
        def_str = f'{def_str}\n    pass'
        sig_namespace = {}
        exec(def_str, sig_namespace)
        sig = inspect.signature(sig_namespace['func'])
        # Find indentation level
        for iterator_line in iterator_lines:
            iterator_line_stripped = iterator_line.strip()
            if iterator_line_stripped and not iterator_line_stripped.startswith('#'):
                indentation = len(iterator_line) - len(iterator_line.lstrip())
                break
        # Remove indentation
        for i, iterator_line in enumerate(iterator_lines):
            if iterator_line[:indentation] != ' '*indentation:
                continue
            iterator_lines[i] = iterator_line[indentation:]
        # Store the unindented source lines of the iterator
        # together with its function signature.
        cached_iterators[func_name] = (iterator_lines, sig)
        return iterator_lines.copy(), sig
    cached_iterators = {}
    # Replace usages of inline iterators with the content of the
    # iterator functions themselves.
    new_lines = []
    for_indentation = -1
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        indentation = len(line) - len(line.lstrip())
        if indentation <= for_indentation:
            for_indentation = -1
        if for_indentation != -1:
            line = ' '*(inlined_indentation - for_indentation - 4) + line
        if line.lstrip().startswith('#'):
            new_lines.append(line)
            continue
        # Check for usage of inline iterator at this line
        match = re.search(r'^ *for +(.+?) +in +(.+?) *\(', line)
        if not match:
            new_lines.append(line)
            continue
        assignments_in_for = match.group(1).strip()
        func_name = match.group(2)
        stop = False
        for no in r'.,()[]{}':
            if no in func_name:
                stop = True
                break
        if stop:
            new_lines.append(line)
            continue
        if func_name.startswith('"') or func_name.startswith("'"):
            new_lines.append(line)
            continue
        if func_name not in module.__dict__:
            new_lines.append(line)
            continue
        try:
            iterator_lines = inspect.getsourcelines(module.__dict__[func_name])[0]
        except:
            new_lines.append(line)
            continue
        iterator_lines = commons.onelinerize(iterator_lines)
        for iterator_line in iterator_lines:
            iterator_line_lstripped = iterator_line.lstrip()
            if iterator_line_lstripped.startswith('#'):
                continue
            if iterator_line_lstripped.startswith('@cython.iterator'):
                break
        else:
            new_lines.append(line)
            continue
        iterator_lines, sig = process_inline_iterator_lines(
            func_name, iterator_lines,
        )
        # Replace yield statement with assignment
        iterator_lines[-1] = re.sub(' yield ', f' {assignments_in_for} = ', iterator_lines[-1], 1)
        # Separate multi-assignment into many assignments
        names, values = iterator_lines[-1].split('=')
        names  = names .split(',')
        values = values.split(',')
        if len(names) == len(values):
            yield_indentation = ' '*(len(iterator_lines[-1]) - len(iterator_lines[-1].lstrip()))
            iterator_lines[-1] = f'{yield_indentation}pass  # yield statement was here\n'
            for name, value in zip(names, values):
                name  = name .strip()
                value = value.strip()
                if name != value:
                    iterator_lines.append(f'{yield_indentation}{name} = {value}\n')
        # Extract call
        parens = 0
        encountered = set()
        for i, c in enumerate(line[::-1]):
            encountered.add(c)
            parens += (c == '(') - (c == ')')
            if ':' in encountered and '(' in encountered and parens == 0:
                break
        else:
            # Something went wrong
            print(f'Failed to detect generator function call to {func_name}()', file=sys.stderr)
            sys.exit(1)
        call = line[-(i+1):].strip(' (,):\n')
        # Perform binding of function signature from the call
        args = []
        for arg in call.split(','):
            if '=' in arg:
                lhs, rhs = arg.split('=')
                lhs = lhs.strip()
                rhs = rhs.strip()
                if rhs[0] in r'"\'':
                    rhs = rhs.strip(r'"\'')
                    rhs = f'"(str){rhs}"'
                else:
                    rhs = f'"{rhs}"'
                arg = '='.join([lhs, rhs])
            else:
                arg = arg.strip()
                if arg[0] in r'"\'':
                    arg = arg.strip(r'"\'')
                    arg = f'"(str){arg}"'
                else:
                    arg = f'"{arg}"'
            args.append(arg)
        args = ', '.join(args)
        ba = eval(f'sig.bind({args})')
        arguments_call = {}
        for key, val in ba.arguments.items():
            if isinstance(val, str) and val.startswith('(str)'):
                val = f'"{val[5:]}"'
            arguments_call[key] = val
        ba.apply_defaults()
        arguments = {}
        for key, val in ba.arguments.items():
            if key in arguments_call:
                arguments[key] = arguments_call[key]
                continue
            if isinstance(val, str):
                val = f'"{val}"'
            arguments[key] = val
        # Remove loop unswitches within the iterator lines
        # which depend on the keyword-only arguments.
        if not no_optimization:
            # Get values of keyword-only arguments
            kw_onlys = {}
            for key, param in sig.parameters.items():
                if param.kind != param.KEYWORD_ONLY:
                    continue
                val = arguments[key]
                if isinstance(val, str) and val[0] not in r'"\'':
                    try:
                        val = eval(val)
                    except:
                        pass
                kw_onlys[key] = val
            for blackboardbold in blackboardbolds:
                blackboardbold_normalized = unicodedata.normalize('NFKC', blackboardbold)
                kw_onlys[blackboardbold_normalized] = getattr(commons, blackboardbold_normalized)
            pattern = r'^ *with +unswitch *(\(.*\))? *:'
            unswitch_indentation = -1
            iterator_lines_new = []
            for iterator_line in iterator_lines:
                iterator_line_stripped = iterator_line.strip()
                if not iterator_line_stripped or iterator_line_stripped.startswith('#'):
                    iterator_lines_new.append(iterator_line)
                    continue
                if unswitch_indentation == -1:
                    if re.search(pattern, iterator_line):
                        # At unswitch
                        unswitch_indentation = len(iterator_line) - len(iterator_line.lstrip())
                    iterator_lines_new.append(iterator_line)
                    continue
                indentation = len(iterator_line) - len(iterator_line.lstrip())
                if indentation <= unswitch_indentation:
                    if re.search(pattern, iterator_line):
                        # At another unswitch
                        unswitch_indentation = len(iterator_line) - len(iterator_line.lstrip())
                    else:
                        # Outside unswitch
                        unswitch_indentation = -1
                    iterator_lines_new.append(iterator_line)
                    continue
                if indentation == unswitch_indentation + 4:
                    if iterator_line_stripped.startswith('else'):
                        # At else
                        iterator_lines_new.append(iterator_line)
                        continue
                    match = re.search(r'^(if|elif) (.+) *:', iterator_line_stripped)
                    if not match:
                        print(
                            f'Improper indentation beneath unswitch in {func_name}()',
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    # At if/elif
                    if_elif = match.group(1)
                    condition = match.group(2).strip()
                    try:
                        result = eval(condition, kw_onlys)
                    except:
                        # Could not determine value of condition
                        iterator_lines_new.append(iterator_line)
                        continue
                    # Compile time constant value of condition
                    result = bool(result)
                    iterator_lines_new.append(' '*indentation + f'{if_elif} {result}:\n')
                elif indentation > unswitch_indentation:
                    # Inside unswitch
                    iterator_lines_new.append(iterator_line)
                    continue
            iterator_lines = iterator_lines_new
        # Apply correct indentation
        for_indentation = len(line) - len(line.lstrip())
        iterator_lines = [
            ' '*for_indentation + iterator_line for iterator_line in iterator_lines
        ]
        arguments_lines = [
            ' '*for_indentation + f'{arg} = {val}' for arg, val in arguments.items()
        ]
        # Replace iterator call with inlined iterator code lines
        new_lines += [
            '\n',
            ' '*for_indentation + f'# Beginning of inlined iterator "{func_name}"\n',
        ]
        new_lines += [default_argument + '\n' for default_argument in arguments_lines]
        new_lines += iterator_lines
        # Indent the rest of the function according
        # to the inlined iterator.
        for iterator_line in iterator_lines[::-1]:
            if not iterator_line.strip():
                continue
            if iterator_line.lstrip().startswith('#'):
                continue
            inlined_indentation = len(iterator_line) - len(iterator_line.lstrip())
            break
        new_lines += [
            ' '*inlined_indentation + f'# End of inlined iterator "{func_name}"\n',
            '\n',
        ]
    # Replace the inline iterator function definitions themselves with
    # a trivial function definition, to reduce compilation time.
    # We cannot remove these function definitions completely,
    # as the functions must still be importable.
    lines = new_lines
    new_lines = []
    skip = 0
    for j, line in enumerate(lines):
        if skip:
            skip -= 1
            continue
        line_lstripped = line.lstrip()
        if line_lstripped.startswith('#'):
            new_lines.append(line)
            continue
        if line_lstripped.startswith('@cython.iterator'):
            indentation = len(line) - len(line.lstrip())
            def_encountered = False
            parens = 0
            parens_any = False
            for i, iterator_line in enumerate(lines[j+1:], j+1):
                if iterator_line.lstrip().startswith('#'):
                    continue
                parens += iterator_line.count('(')
                if parens:
                    parens_any = True
                parens -= iterator_line.count(')')
                if iterator_line.lstrip().startswith('def '):
                    def_encountered = True
                if not def_encountered:
                    continue
                if iterator_line.rstrip().endswith(':') and parens == 0 and parens_any:
                    break
            new_lines += (lines[j+1:i+1]
                + [
                    ' '*(indentation + 4)
                        + '# This function was originally decorated with @cython.iterator.\n',
                    ' '*(indentation + 4)
                        + '# Its original content has been removed.\n',
                    ' '*(indentation + 4) + 'pass\n',
                    '\n',
                ]
            )
            for i, iterator_line in enumerate(lines[i+1:], i+1):
                if iterator_line.lstrip().startswith('#'):
                    continue
                if not iterator_line.strip():
                    continue
                if len(iterator_line) - len(iterator_line.lstrip()) == indentation:
                    break
            skip = i - j - 1
            continue
        else:
            new_lines.append(line)
    return new_lines



def remove_trvial_branching(lines, no_optimization):
    if no_optimization:
        return lines
    def unindent(line, unincent_length):
        line_lstripped = line.lstrip()
        indentation = len(line) - len(line_lstripped)
        indentation -= unincent_length
        line = ' '*indentation + line_lstripped
        return line
    def remove_branches(lines):
        changed = False
        new_lines = []
        skip = 0
        for i, line in enumerate(lines):
            if skip:
                skip -= 1
                continue
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                new_lines.append(line)
                continue
            match = re.search(r'^(if|elif) +(.+) *:', line_stripped)
            if not match:
                new_lines.append(line)
                continue
            condition = match.group(2)
            try:
                result = eval(condition, {})
            except:
                new_lines.append(line)
                continue
            # Trivial if/elif found
            result = bool(result)
            changed = True
            if_elif = match.group(1)
            indentation = len(line) - len(line.lstrip())
            if result:
                # A truthy if/elif was found
                indentation_comment = indentation
                if if_elif == 'if':
                    unindent_length = 4
                else:
                    unindent_length = 0
                if if_elif == 'if':
                    # Remove above unswitch if present
                    for j, line_prev in enumerate(reversed(new_lines)):
                        line_prev_stripped = line_prev.strip()
                        if not line_prev_stripped or line_prev_stripped.startswith('#'):
                            continue
                        if re.search(r'^ *with +unswitch', line_prev):
                            # Found unswitch to be removed
                            unindent_length += 4
                            indentation_comment -= 4
                            new_lines[len(new_lines) - j - 1] = (
                                ' '*indentation_comment
                                + '# Trivial unswitch removed due removal of below if\n'
                            )
                        else:
                            break
                else:
                    # Replace with else
                    new_lines.append(' '*indentation + 'else:\n')
                # Removing all further elif/else
                in_elif_else = False
                for line_body in lines[i+1:]:
                    line_body_lstripped = line_body.lstrip()
                    indentation_body = len(line_body) - len(line_body_lstripped)
                    if indentation_body > indentation:
                        if not in_elif_else:
                            new_lines.append(unindent(line_body, unindent_length))
                        skip += 1
                    elif indentation_body == indentation:
                        if (
                               re.search(r'^else *:'      , line_body_lstripped)
                            or re.search(r'^elif +(.+) *:', line_body_lstripped)
                        ):
                            # Skip elif/else
                            in_elif_else = True
                            skip += 1
                        else:
                            # Outside if body
                            break
                    else:
                        # Outside if body
                        break
            else:
                # A falsy if/elif was found. Remove it, but insert a
                # trivial truthy if in the case of an otherwise complete
                # removal of the entire if/elif/else.
                for line_body in lines[i+1:]:
                    line_body_lstripped = line_body.lstrip()
                    indentation_body = len(line_body) - len(line_body_lstripped)
                    if indentation_body > indentation:
                        skip += 1
                    elif indentation_body == indentation:
                        at_elif = re.search(r'^elif +(.+) *:', line_body_lstripped)
                        if at_elif:
                            # At elif
                            if if_elif == 'if':
                                # Convert to if
                                new_lines.append(' '*indentation + line_body_lstripped[2:])
                                skip += 1
                            break
                        at_else = re.search(r'^else *:'      , line_body_lstripped)
                        if at_else:
                            # At else
                            if if_elif == 'if':
                                # Convert to trivial if
                                new_lines.append(' '*indentation + 'if True:\n')
                                skip += 1
                            break
                        # Outside if body
                        if if_elif == 'if':
                            # Insert trivial truthy if
                            new_lines.append(' '*indentation + 'if True:\n')
                            new_lines.append(' '*(indentation + 4) + 'pass\n')
                        break
                    else:
                        # Outside if body
                        if if_elif == 'if':
                            # Insert trivial truthy if
                            new_lines.append(' '*indentation + 'if True:\n')
                            new_lines.append(' '*(indentation + 4) + 'pass\n')
                        break
        return new_lines, changed
    changed = True
    while changed:
        lines, changed = remove_branches(lines)
    return lines



def constant_expressions(lines, no_optimization, first_call=True):
    non_c_native = {'𝕆', '𝕊'}
    non_callable = {'𝕆', }
    # Handle nested constant expressions.
    # If first_call is True, this is the original call to this function.
    # Edit all nested occurrences of constant expressions so that only
    # the inner most expression survived, while the outer ones will
    # be assigned a nesting number, e.g.
    # ℝ[2 + ℝ[3*4]] -> ℝ0[2 + ℝ1[3*4]].
    if first_call:
        find_nested = True
        while find_nested:
            find_nested = False
            new_lines = []
            for i, line in enumerate(lines):
                line = line.rstrip('\n')
                search = re.search(r'[{}]\[.+\]'.format(''.join(blackboardbolds.keys())), line)
                if not search or line.replace(' ', '').startswith('#'):
                    new_lines.append(line + '\n')
                    continue
                # Blackboard bold symbol found on this line
                find_nested = True
                R_statement_fullmatch = search.group(0)
                R_statement = R_statement_fullmatch[:2]
                for c in R_statement_fullmatch[2:]:
                    R_statement += c
                    if R_statement.count('[') == R_statement.count(']'):
                        break
                expression = re.sub(' +', ' ', R_statement[2:-1].strip())
                edited_line = []
                for blackboard_bold_symbol_i in blackboardbolds:
                    if blackboard_bold_symbol_i in expression:
                        # Nested blackboard bold expression found
                        lvl_indices = []
                        lvl = -1
                        bracket_types = []
                        c_before = ''
                        for c in line:
                            write_lvl = False
                            if c == '[':
                                if c_before in blackboardbolds:
                                    lvl += 1
                                    bracket_types.append('constant expression')
                                    write_lvl = True
                                else:
                                    bracket_types.append('other')
                            elif c == ']':
                                bracket_type = bracket_types.pop()
                                if bracket_type == 'constant expression':
                                    lvl -= 1
                                elif bracket_type == 'other':
                                    pass
                            edited_line.append(c)
                            if write_lvl:
                                edited_line.pop()
                                lvl_indices.append(len(edited_line))
                                edited_line.append(str(lvl))
                                edited_line.append('[')
                            c_before = c
                        break
                if edited_line:
                    # Invert the lvl's so that the inner expressions have
                    # the largest lvl.
                    lvls = [int(edited_line[lvl_index]) for lvl_index in lvl_indices]
                    if max(lvls) > 0:
                        j = 0
                        for i in range(1, len(lvls)):
                            if lvls[i] == 0:
                                max_lvl = max(lvls[j:i])
                                lvls[j:i] = [max_lvl - lvls[j] for j in range(j, i)]
                                j = i
                    for lvl, lvl_index in zip(lvls, lvl_indices):
                        edited_line[lvl_index] = str(lvl)
                    line = ''.join(edited_line)
                else:
                    # At least the first constant expression on this line
                    # is not nested. Place a nesting number of 0.
                    line = re.sub('(' + '|'.join(blackboardbolds) + r')\[', r'\g<1>0[', line, 1)
                new_lines.append(line + '\n')
            lines = new_lines
    # Remove the nest lvl on constant expressions at lvl 0
    # and decrease all other lvl's by 1.
    all_lvls = set()
    def replace(m):
        s = m.group()
        lvl = int(s[1:-1])
        all_lvls.add(lvl)
        lvl -= 1
        if lvl == -1:
            return s[0] + '['
        else:
            return s[0] + str(lvl) + '['
    new_lines = []
    for i, line in enumerate(lines):
        line = line.rstrip('\n')
        search = re.search(r'[{}]([0-9]+)\[.+\]'.format(''.join(blackboardbolds.keys())), line)
        if search and not line.replace(' ', '').startswith('#'):
            line, _ = re.subn(
                r'[{}][0-9]+\['.format(''.join(blackboardbolds.keys())),
                replace,
                line,
            )
        new_lines.append(line + '\n')
    lines = new_lines
    # Helper functions
    def indentation_level(line):
        line_lstrip = line.lstrip()
        if not line_lstrip or line_lstrip.startswith('#'):
            return -1
        return len(line) - len(line_lstrip)
    def variable_changed(var, line):
        line_ori = line
        line = line.strip()
        if line.startswith('#'):
            return False
        line = line.replace(' ', '')
        if not line:
            return False
        def multi_assign_in_for(var, line):
            line = line.strip()
            line = line.replace(',', ' ').replace('(', ' ').replace(')', ' ')
            if not (line.startswith('for ') and ' {} '.format(var) in line and ' in ' in line):
                return
            if line.index(' {} '.format(var)) < line.index(' in '):
                return True
        def multi_assign_in_line(var, line):
            line = line.replace(' ', '')
            # Remove function calls
            removed_args = True
            while removed_args:
                removed_args = False
                in_func_call = False
                for i, c in enumerate(line):
                    if not in_func_call and c == '(' and i > 0 and re.search(r'\w', line[i - 1]):
                        in_func_call = True
                        i_start = i
                        parens = 1
                        continue
                    if in_func_call:
                        if c == '(':
                            parens += 1
                        elif c == ')':
                            parens -= 1
                            if parens == 0:
                                i_end = i
                                if i_end != i_start + 1:
                                    line = line[:(i_start + 1)] + line[i_end:]
                                    removed_args = True
                                in_func_call = False
                                break
            # Remove parens
            line = line.replace('(', ' ')
            line = line.replace(')', ' ')
            # Now do the searching
            if re.search(rf'^{var},[^=]*=[^=]', line):  # var in front
                return True
            if re.search(rf',{var},[^=]*=[^=]', line):  # var somewhere in the middle
                return True
            if re.search(rf',{var}=[^=]',       line):  # var at the end
                return True
            return False
        def used_as_function_arg(var, line):
            # This function checks whether the variable var is used as
            # a function argument in the line, the idea being that if
            # the variable is mutable, we cannot guarantee that it is
            # not altered by the function. However, most variables used
            # in constant expressions are immutable, and so actually
            # using this function will make the optimizations worse.
            # We therefore ignore whether or not variables are used
            # in a function call. It is thus up to the developer to
            # remember the danger of using constant expressions
            # containing mutable values.
            return False
            # The actual function, should we some day decide to use it
            match = re.search(rf'\w *\(.*{var}[, )]', line)
            if match:
                s = match.group().rstrip(')')
                if s.count('(') - s.count(')') == 1:
                    return True
            return False
        return (   multi_assign_in_for(var, line_ori)
                or multi_assign_in_line(var, line_ori)
                or used_as_function_arg(var, line_ori)
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
    def affectable(lines):
        # Unindent loops
        for i, line in enumerate(lines):
            if line.lstrip().startswith('for ') or line.lstrip().startswith('while '):
                indentation_i = indentation_level(line)
                for j, line in enumerate(lines[(i + 1):]):
                    indentation_j = indentation_level(line)
                    if not line.lstrip().startswith('#') and indentation_j <= indentation_i:
                        break
                    lines[i + 1 + j] = ' '*indentation_i + line.lstrip()
        # Find 'else', 'elif' and 'except' barriers between
        # first and last line.
        if not lines:
            return True
        important_lines = [lines[0]]
        for line in lines:
            if line.lstrip().startswith('if '):
                important_lines.append(line)
            elif line.lstrip().startswith('else:'):
                important_lines.append(line)
            elif line.lstrip().startswith('elif '):
                important_lines.append(line)
            elif line.lstrip().startswith('try:'):
                important_lines.append(line)
            elif line.lstrip().startswith('except:') or line.lstrip().startswith('except '):
                important_lines.append(line)
            elif line.lstrip().startswith('for '):
                important_lines.append(line)
            elif line.lstrip().startswith('while '):
                important_lines.append(line)
        important_lines.append(lines[-1])
        linenr_to_skip = set()
        for i, line in enumerate(important_lines):
            if line.lstrip().startswith('if '):
                linenr_to_skip.add(i)
                indentation_i = indentation_level(line)
                for j, line in enumerate(important_lines[(i + 1):]):
                    indentation_j = indentation_level(line)
                    if (indentation_i == indentation_j
                        and (   line.lstrip().startswith('else:')
                             or line.lstrip().startswith('elif ')
                             )
                        ):
                        linenr_to_skip.add(i + 1 + j)
                    else:
                        break
            elif line.lstrip().startswith('try:'):
                linenr_to_skip.add(i)
                indentation_i = indentation_level(line)
                for j, line in enumerate(important_lines[(i + 1):]):
                    indentation_j = indentation_level(line)
                    if (indentation_i == indentation_j
                        and (   line.lstrip().startswith('except:')
                             or line.lstrip().startswith('except ')
                             or line.lstrip().startswith('else:')
                             )
                        ):
                        linenr_to_skip.add(i + 1 + j)
                    else:
                        break
            elif line.lstrip().startswith('for ') or line.lstrip().startswith('while '):
                linenr_to_skip.add(i)
                indentation_i = indentation_level(line)
                for j, line in enumerate(important_lines[(i + 1):]):
                    indentation_j = indentation_level(line)
                    if (indentation_i == indentation_j
                        and line.lstrip().startswith('else:')
                        ):
                        linenr_to_skip.add(i + 1 + j)
                    else:
                        break
        filtered_lines = []
        for i, line in enumerate(important_lines):
            if i not in linenr_to_skip:
                filtered_lines.append(line)
        indentation_start = indentation_level(lines[ 0])
        indentation_end   = indentation_level(lines[-1])
        for line in filtered_lines[1:-1]:
            indentation = indentation_level(line)
            if (indentation < indentation_start and indentation < indentation_end
                and (   line.lstrip().startswith('except:')
                     or line.lstrip().startswith('except ')
                     or line.lstrip().startswith('else:')
                     or line.lstrip().startswith('elif ')
                     )
                ):
                # Barrier located
                return False
        return True
    # The placement of the definitions of the constant expressions may
    # be erroneous in cases where a variable in the expression is only
    # perhaps set on a given line. For example, variables defined within
    # if statements (under the if, the elif or the else) are dangerous
    # because the constant expression should only be defined after the
    # end of the loop, at least in the case where the constant
    # expression is not used before the loop has ended.
    # We can fix this issue by placing dummy definitions after the end
    # of each if statement, defining every variable defined within the
    # loop.
    found_another_statement = True
    dummy_declaration_value = '__PYXPP_DELETE__'
    dummy_statement_comment = '  # ' + dummy_declaration_value
    while found_another_statement:
        found_another_statement = False
        new_lines = []
        statement = ''
        for line in lines:
            # Empty and comment lines
            line_lstrip = line.lstrip()
            if not line_lstrip.rstrip() or line_lstrip.startswith('#'):
                new_lines.append(line)
                continue
            # Look for beginning of statements
            if not statement:
                # Look for beginning of statement
                if not line.rstrip().endswith(dummy_declaration_value):
                    if line_lstrip.startswith('if '):
                        statement = 'if'
                    elif line_lstrip.startswith('while '):
                        statement = 'while'
                    elif line_lstrip.startswith('for '):
                        statement = 'for'
                    elif line_lstrip.startswith('with '):
                        statement = 'with'
                    elif line_lstrip.startswith('try') or line_lstrip.startswith('try:'):
                        statement = 'try'
                # Done with this line
                if statement:
                    found_another_statement = True
                    statement_lvl = len(line) - len(line_lstrip)
                    varnames = []
                    # Mark the line
                    line = line[:-1] + dummy_statement_comment + '\n'
                new_lines.append(line)
                continue
            # Still inside statement?
            still_inside = True
            lvl = len(line) - len(line.lstrip())
            if lvl < statement_lvl:
                still_inside = False
            elif lvl == statement_lvl:
                if statement == 'if':
                    if not (   line_lstrip.startswith('elif ')
                            or line_lstrip.startswith('else ')
                            or line_lstrip.startswith('else:')
                            ):
                        still_inside = False
                elif statement in ('while', 'for'):
                    if not (   line_lstrip.startswith('else ')
                            or line_lstrip.startswith('else: ')
                            ):
                        still_inside = False
                elif statement == 'with':
                    still_inside = False
                elif statement == 'try':
                    if not (   line_lstrip.startswith('except ')
                            or line_lstrip.startswith('else')
                            or line_lstrip.startswith('else:')
                            or line_lstrip.startswith('finally')
                            or line_lstrip.startswith('finally:')
                            ):
                        still_inside = False
            if still_inside:
                # Inside the statement.
                # Look for defined/updated variables.
                candidates = [s for s in re.findall(r'\w+', line)
                              if s.isidentifier() and not keyword.iskeyword(s)]
                for candidate in candidates:
                    if candidate not in varnames and variable_changed(candidate, line):
                        varnames.append(candidate)
            else:
                # Statement has ended
                statement = ''
                # Add dummy declarations
                for varname in varnames:
                    new_lines.append('{}{} = {}\n'.format(' '*statement_lvl,
                                                          varname,
                                                          dummy_declaration_value))
            # Add the line regardless of whether inside a statement or not
            new_lines.append(line)
        lines = new_lines
    # Now do the actual work
    for blackboard_bold_symbol, ctype in blackboardbolds.items():
        # Find constant expressions using the ℝ[expression] syntax
        expressions = []
        expressions_cython = []
        declaration_linenrs = []
        declaration_placements = []
        operators = {
            '.': 'DOT',
            '+': 'PLS',
            '-': 'MIN',
            '**': 'POW',
            '*': 'TIM',
            '/': 'DIV',
            '\\': 'BSL',
            '^': 'CAR',
            '&': 'AND',
            '|': 'BAR',
            '@': 'AT',
            ',': 'COM',
            '(': 'OPAR',
            ')': 'CPAR',
            '[': 'OBRA',
            ']': 'CBRA',
            '{': 'OCUR',
            '}': 'CCUR',
            "'": 'QTE',
            '"': 'DQTE',
            ':': 'COL',
            ';': 'SCOL',
            '==': 'CMP',
            '!=': 'NCMP',
            '=': 'EQ',
            '!': 'BAN',
            '<': 'LTH',
            '>': 'GTH',
            '#': 'SHA',
            '$': 'DOL',
            '%': 'PER',
            '?': 'QUE',
            '`': 'GRA',
            '~': 'TIL',
        }
        while True:
            no_blackboard_bold_symbol = True
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
                search = re.search(blackboard_bold_symbol + r'\[.+\]', line)
                if not search or line.replace(' ', '').startswith('#'):
                    continue
                # Blackboard bold symbol found on this line
                R_statement_fullmatch = search.group(0)
                R_statement = R_statement_fullmatch[:2]
                for c in R_statement_fullmatch[2:]:
                    R_statement += c
                    if R_statement.count('[') == R_statement.count(']'):
                        break
                expression = re.sub(' +', ' ', R_statement[2:-1].strip())
                # Ensure float division within double expressions
                if ctype == 'double':
                    # Replace all slashes inside quotes
                    quotes = {'"': False, "'": False}
                    expression_chars = []
                    for c in expression:
                        if c == '"':
                            quotes['"'] ^= True
                        elif c == "'":
                            quotes["'"] ^= True
                        if c == '/' and (quotes['"'] or quotes["'"]):
                            expression_chars.append('�')
                        else:
                            expression_chars.append(c)
                    expression = ''.join(expression_chars)
                    # Insert a cast to double at all denominators
                    def replace(m):
                        s = m.group()
                        if s.count('/') == 1:
                            return s[0] + r'/<double>' + s[2]
                        else:
                            return s
                    expression = re.sub('./.', replace, expression)
                    # Re-insert slashes inside quotes
                    expression = expression.replace('�', '/')
                no_blackboard_bold_symbol = False
                # If optimizations are disabled, simply remove the
                # blackboard bold symbol and insert the updated
                # expression with float literals.
                if no_optimization:
                    lines[i] = '{}\n'.format(line.replace(R_statement, '({})'.format(expression)))
                    continue
                # Do the optimization
                expressions.append(expression)
                expression_cython = blackboard_bold_symbol + '_' + expression.replace(' ', '')
                for op, op_name in operators.items():
                    expression_cython = expression_cython.replace(op, f'�{op_name}�')
                expression_cython = expression_cython.replace('��', '�').strip('�')
                expression_cython = expression_cython.replace('�', '_')
                expressions_cython.append(expression_cython)
                lines[i] = '{}\n'.format(line.replace(R_statement, expression_cython))
                # Find out where the declaration should be
                variables = re.sub('(?P<quote>[\'"]).*?(?P=quote)', # Remove string literals using
                                   '', expression)                  # single and double quotes.
                variables = [variables]
                # Split variables on operators, including ' ' but
                # excluding '.'. Attributes are handled afterwards.
                for op in itertools.chain(operators.keys(), ' '):
                    if op != '.':
                        variables = list(itertools.chain(*[var.split(op) for var in variables]))
                # When a variable is really an instance attribute,
                # the attribute as well as the instance itself are
                # considered variables.
                for v, var in enumerate(variables.copy()):
                    if '.' in var:
                        obj = var.partition('.')[0]
                        variables[v] = obj
                        for attr in var.split('.')[1:]:
                            obj += f'.{attr}'
                            variables.append(obj)
                variables = [var for var in list(set(variables))
                             if var and var[0] not in '.0123456789']
                # Remove non-variables
                nonvariables = {
                    'bint', 'int', 'float', 'double', 'Py_ssize_t',
                    'tuple', 'list', 'set', 'dict',
                    'sin', 'cos', 'tan',
                    'arcsin', 'arccos', 'arctan', 'arctan2',
                    'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
                    'exp', 'log', 'log2', 'log10',
                    'sqrt', 'cbrt',
                    'erf', 'erfc',
                    'floor', 'ceil', 'round',
                    'abs', 'mod',
                }
                variables = list(set(variables) - nonvariables)
                linenr_where_defined = [-1]*len(variables)
                placements = ['below']*len(variables)
                defined_in_function = [False]*len(variables)
                for w, end in enumerate((i + 1, len(lines))):  # Second time: Check module scope
                    if w == 1 and module_scope:
                        break
                    for v, var in enumerate(variables):
                        if linenr_where_defined[v] != -1:
                            continue
                        in_docstring = {'"""': False, "'''": False}
                        for j, line2 in enumerate(reversed(lines[:end])):
                            # Skip doc strings
                            line2_lstripped = line2.lstrip()
                            for triple_quote in ('"""', "'''"):
                                if line2_lstripped.startswith(triple_quote):
                                    in_docstring[triple_quote] = not in_docstring[triple_quote]
                                    if line2.rstrip().endswith(triple_quote) and len(line2.strip()) > 5:
                                        in_docstring[triple_quote] = not in_docstring[triple_quote]
                                elif not line2_lstripped.startswith('#') and line2.rstrip().endswith(triple_quote):
                                    in_docstring[triple_quote] = not in_docstring[triple_quote]
                            if in_docstring['"""'] or  in_docstring["'''"]:
                                continue
                            line2 = line2.rstrip('\n')
                            line2_ori = line2
                            line2 = (' '*(len(line2) - len(line2.lstrip()))
                                     + line2.replace(' ', ''))
                            if (    line2_ori.lstrip().startswith('def ')
                                and re.search(r'[\(,]{}[,=\)]'.format(var), line2)):
                                # var as function argument
                                linenr_where_defined[v] = end - 1 - j
                                defined_in_function[v] = (w == 0)
                                break
                            else:
                                if (    variable_changed(var, line2_ori)
                                    and affectable(lines[(end - j - 1):(i + 1)])):
                                    # var declaration found
                                    linenr_where_defined[v] = end - 1 - j
                                    defined_in_function[v] = (w == 0)
                                    # Continue searching for var in previous lines
                                    linenr_where_defined_first = -1
                                    for k, line3 in enumerate(reversed(lines[:linenr_where_defined[v]])):
                                        line3 = line3.rstrip('\n')
                                        line3_ori = line3
                                        line3 = ' '*(len(line3) - len(line3.lstrip()))  + line3.replace(' ', '')
                                        if line3_ori.lstrip().startswith('def '):
                                            # Function definition reached
                                            break
                                        if (indentation_level(line3_ori) == function_scope_indentation_level
                                            and not line3_ori.lstrip().startswith('elif ')
                                            and not line3_ori.lstrip().startswith('else:')
                                            and not line3_ori.lstrip().startswith('except:')
                                            and not line3_ori.lstrip().startswith('except ')
                                            and not line3_ori.lstrip().startswith('finally:')
                                            and not line3_ori.lstrip().startswith('finally ')
                                            ):
                                            # Upper level of function reached.
                                            # Definitions above this point does not matter.
                                            break
                                        if (    variable_changed(var, line3_ori)
                                            and affectable(lines[(linenr_where_defined[v] - k - 1):(i + 1)])):
                                            # Additional var declaration found
                                            linenr_where_defined_first = linenr_where_defined[v] - 1 - k
                                    # Locate "barriers" between linenr_where_defined and linenr_where_defined_first.
                                    # A "barrier" consists of code with an indentation level smaller
                                    # than that at linenr_where_defined, located somewhere between
                                    # linenr_where_defined and linenr_where_defined_first.
                                    # If such barriers exist, the definition to be inserted cannot just
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
                    # If the constant expression is in the module scope,
                    # simply use the last line where the variables
                    # are defined. If the constant expression is inside
                    # a function, use the last line inside the function
                    # where the variables are defined, if any. If not,
                    # use the last line where the variables are defined
                    # in the module scope.
                    if module_scope:
                        index_max = np.argmax(linenr_where_defined)
                    else:
                        for index_max in reversed(np.argsort(linenr_where_defined)):
                            if defined_in_function[index_max]:
                                break
                        else:
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
            if no_blackboard_bold_symbol:
                break
        # Find out where the last import statement is. Unrecognised
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
                # Make sure that we are not inside a function or class
                indentation_level_i = indentation_level(line)
                if indentation_level_i > 0:
                    inside_func_or_class = False
                    for line in reversed(lines[:i]):
                        if line and line[0] not in ' #':
                            # Reached indentation level 0
                            break
                        line_lstripped = line.lstrip()
                        if line_lstripped.startswith('def ') or line_lstripped.startswith('class '):
                            indentation_level_above = indentation_level(line)
                            if indentation_level_above < indentation_level_i:
                                # Import inside function or class
                                inside_func_or_class = True
                                break
                    if inside_func_or_class:
                        continue
                # Go down until indentation level 0 is reached
                for j, line in enumerate(lines[(i + 1):]):
                    if (len(line) > 0
                        and line[0] not in '# '
                        and not line.startswith('"""')
                        and not line.startswith("'''")
                    ):
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
            # Unrecognised definitions
            if i == linenr_unrecognized:
                for e, expression_cython in enumerate(expressions_cython):
                    if declaration_linenrs[e] == -1:
                        if expression_cython not in declarations_placed[fname]:
                            if blackboard_bold_symbol in non_c_native:
                                new_lines.append(
                                    f'cython.declare({expression_cython}={ctype})\n'
                                )
                            else:
                                new_lines.append(
                                    f'cython.declare({expression_cython}=\'{ctype}\')\n'
                                )
                            if fname:
                                # Remember that this variable has been declared in this function
                                declarations_placed[fname].append(expression_cython)
                        if blackboard_bold_symbol in non_c_native:
                            if blackboard_bold_symbol in non_callable:
                                new_lines.append(
                                    f'{expression_cython} = ({expressions[e]})\n'
                                )
                            else:
                                new_lines.append(
                                    f'{expression_cython} = {ctype}({expressions[e]})\n'
                                )
                        else:
                            new_lines.append(
                                f'{expression_cython} = <{ctype}>({expressions[e]})\n'
                            )
                new_lines.append('\n')
            for e, n in enumerate(declaration_linenrs):
                if i == n:
                    indentation = ' '*(len(line) - len(line.lstrip()))
                    if declaration_placements[e] == 'below' and re.search(': *(#.*)?$', line):
                        indentation += ' '*4
                    if declaration_placements[e] == 'above':
                        new_lines.pop()
                    if expressions_cython[e] not in declarations_placed[fname]:
                        if blackboard_bold_symbol in non_c_native:
                            new_lines.append(
                                f'{indentation}cython.declare({expressions_cython[e]}={ctype})\n'
                            )
                        else:
                            new_lines.append(
                                f'{indentation}cython.declare({expressions_cython[e]}=\'{ctype}\')\n'
                            )
                        if fname:
                            # Remember that this variable has been declared in this function
                            declarations_placed[fname].append(expressions_cython[e])
                    if blackboard_bold_symbol in non_c_native:
                        if blackboard_bold_symbol in non_callable:
                            new_lines.append(
                                f'{indentation}{expressions_cython[e]} = ({expressions[e]})\n'
                            )
                        else:
                            new_lines.append(
                                f'{indentation}{expressions_cython[e]} = {ctype}({expressions[e]})\n'
                            )
                    else:
                        new_lines.append(
                            f'{indentation}{expressions_cython[e]} = <{ctype}>({expressions[e]})\n'
                        )
                    if declaration_placements[e] == 'above':
                        new_lines.append(line)
        # Exchange the original lines with the modified lines
        lines = new_lines
    # Remove the inserted dummy declarations and comments
    new_lines = []
    for line in lines:
        line = line.replace(dummy_statement_comment, '')
        if dummy_declaration_value not in line:
            new_lines.append(line)
    # Call recursively until all nested constant expressions has been
    # taken care off.
    if not all_lvls or all_lvls == {0}:
        return new_lines
    return constant_expressions(new_lines, no_optimization, first_call=False)
blackboardbolds = {
    '𝔹': 'bint',
    '𝕆': 'object',
    'ℝ': 'double',
    '𝕊': 'str',
    'ℤ': 'Py_ssize_t',
}



def loop_unswitching(lines, no_optimization):
    """The following constructs will be recognised
    as loop unswitching:
      with unswitch:
          ...
      with unswitch():
          ...
      with unswitch(n):  # With n an integer literal or expression
          ...
    """
    # The strategy of this function is to perform one complete unswitch
    # at a time. To this end, we first go through the lines and replace
    # 'unswitch' with 'unswitch_nr', where 'nr' (0, 1, 2, ...) uniquely
    # identifies the unswitch.
    def extract_n(match):
        if not match:
            return -1
        n = match.group(1)
        if not n:
            return float('inf')
        n = n[1:-1]
        if not n:
            return float('inf')
        return eval(n)
    def loop_over_unswitched_lines(lines):
        pattern = r'with +unswitch *(\(.*\))? *:'
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                continue
            match = re.search(pattern, line_stripped)
            if not match:
                continue
            yield i, line, match
    def build_ordering_key(indentation_lvl, n, index):
        # This function returns its integer arguments packed into
        # a tuple. The lexicographical ordering of such tuples determine
        # the 'nr' in 'unswitch_nr' and thus the order in which the
        # unswitches are carried out. The algorithm of this function is
        # such that we need to perform the outer unswitches before the
        # inner ones, in case of nested unswitches.
        # Thus, indentation_lvl should be the first element of
        # the tuple. For two consecutive unswitches at the same
        # indentation level, the order in which we carry out the
        # unswitching does not matter. If we however do the first
        # encountered unswitch first, the spacing of the comments will
        # remain correct. Thus index is chosen as the second key in the
        # tuple. As this index is unique to the unswitch, it means
        # that the remaining argument, n, does not play a role.
        # We add it to the key anyway.
        return (indentation_lvl, index, n)
    ordering_keys = []
    for i, line, match in loop_over_unswitched_lines(lines):
        indentation_lvl = (len(line) - len(line.lstrip()))//4
        n = extract_n(match)
        index = len(ordering_keys)
        ordering_key = build_ordering_key(indentation_lvl, n, index)
        ordering_keys.append(ordering_key)
    n_unswitches = len(ordering_keys)
    ordering_keys = sorted(ordering_keys)
    ordering_keys2 = []
    for i, line, match in loop_over_unswitched_lines(lines):
        indentation_lvl = (len(line) - len(line.lstrip()))//4
        n = extract_n(match)
        index = len(ordering_keys2)
        ordering_key2 = build_ordering_key(indentation_lvl, n, index)
        ordering_keys2.append(ordering_key2)
        unswitch_nr = ordering_keys.index(ordering_key2)
        lines[i] = lines[i].replace('unswitch', f'unswitch_{unswitch_nr}')
    # Potentially much too many if statements will be
    # inserted, creating code with a structure like this:
    # if condition:      # Keep
    #     if condition:  # Delete
    #         ...        # Keep
    #     else:          # Delete
    #         ....       # Delete
    # where the condition is exactly the same in each if statement.
    # The code is correct, but Cython can take a very long time
    # handling this code, so we reduce it here.
    # As indicated by the comments above, when two nested and equal
    # if statements are found, the body of the if should be kept
    # (and unindented one level), whereas everything else about the
    # if statement should be removed.
    def remove_double_if(lines):
        pattern = r'^ *if +(.+):'
        new_lines = []
        skip = 0
        for i, line in enumerate(lines):
            if skip > 0:
                skip -= 1
                continue
            new_lines.append(line)
            line_no_inline_comment = line
            if '#' in line_no_inline_comment:
                line_no_inline_comment = (
                    line_no_inline_comment[:line_no_inline_comment.index('#')].rstrip() + '\n')
            match = re.search(pattern, line_no_inline_comment)
            if match and not line.lstrip().startswith('#'):
                # If statement found
                condition = match.group(1).strip()
                # Check whether the next line is an if statement
                # with the same condition.
                double_if = False
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    next_line_stripped = next_line.strip()
                    if not next_line_stripped or next_line_stripped.startswith('#'):
                        continue
                    next_line_no_inline_comment = next_line
                    if '#' in next_line_no_inline_comment:
                        next_line_no_inline_comment = (
                            next_line_no_inline_comment[
                                :next_line_no_inline_comment.index('#')].rstrip() + '\n'
                        )
                    match = re.search(pattern, next_line_no_inline_comment)
                    if not match:
                        break
                    next_condition = match.group(1).strip()
                    if condition == next_condition:
                        # Double if found
                        double_if = True
                    elif not (
                           re.search(r'\w *\('      , next_condition)
                        or re.search(r'\w *\['      , next_condition)
                        or re.search(r'\. *[^\W0-9]', next_condition)
                    ):
                        # Double if not found, but some other,
                        # nested if is found, with a condition that
                        # cannot change the state of other variables
                        # by being evaluated. Continue the search for
                        # double if down the nested if's.
                        continue
                    break
                if not double_if:
                    continue
                # Double if statement found.
                # Insert any skipped additional nested if's.
                for k in range(i + 1, j):
                    new_lines.append(lines[k])
                skip += j - i  # Skip the inner if statement(s)
                # Locate the entirety of
                # the inner if block, keeping only the if body.
                inside_inner_if = True
                inner_if_lvl = (len(next_line) - len(next_line.lstrip()))//4
                for line in lines[(j + 1):]:
                    line_stripped = line.strip()
                    if not line_stripped or line_stripped.startswith('#'):
                        # Comment or empty line
                        new_lines.append(line)
                        skip += 1
                        continue
                    lvl = (len(line) - len(line.lstrip()))//4
                    if lvl > inner_if_lvl:
                        if inside_inner_if:
                            # In inner if body
                            new_lines.append(line[4:])
                            skip += 1
                        else:
                            # Inside elif or else clause
                            skip += 1
                    elif lvl == inner_if_lvl:
                        inside_inner_if = False
                        if re.search(r'elif +(.+):', line) or re.search(r'else *:', line):
                            # Beginning of elif or else
                            skip += 1
                        elif line.strip() and not line.lstrip().startswith('#'):
                            # Outside entire inner if block
                            break
                        else:
                            # Comment or empty line
                            new_lines.append(line)
                            skip += 1
                    elif line_stripped and not line_stripped.startswith('#'):
                        # Outside entire inner if block
                        break
                    else:
                        print('From pyxpp.py: How did I end up here?', file=sys.stderr)
                        sys.exit(1)
        return new_lines
    # It is also possible that we have the following
    # impossible construct:
    # if condition:     # Keep
    #     ...           # Keep
    # else:             # Keep
    #    if condition:  # Delete
    #        ...        # Delete
    # where the condition is exactly the same in each if statement.
    # The code is correct, but Cython can take a very long time
    # handling this code, so we reduce it here.
    def remove_impossible_if(lines):
        pattern = r'^ *if +(.+):'
        pattern_else = r'^ *else *:'
        new_lines = []
        skip = 0
        for i, line in enumerate(lines):
            if skip > 0:
                skip -= 1
                continue
            new_lines.append(line)
            line_no_inline_comment = line
            if '#' in line_no_inline_comment:
                line_no_inline_comment = (
                    line_no_inline_comment[:line_no_inline_comment.index('#')].rstrip() + '\n')
            match = re.search(pattern_else, line_no_inline_comment)
            if match and not line.lstrip().startswith('#'):
                # Else clause found. Check whether the first line inside
                # the else block is an if statement.
                next_line = lines[i + 1]
                next_line_no_inline_comment = next_line
                if '#' in next_line_no_inline_comment:
                    next_line_no_inline_comment = (
                        next_line_no_inline_comment[
                            :next_line_no_inline_comment.index('#')].rstrip() + '\n'
                    )
                match = re.search(pattern, next_line_no_inline_comment)
                if not match or line.lstrip().startswith('#'):
                    continue
                # Else clause with if statement inside it found
                condition = match.group(1).strip()
                # Check whether the original if statement accompanying
                # the else clause has the exact same condition as this
                # inner if statement.
                lvl = (len(line) - len(line.lstrip()))//4
                for prev_line in reversed(lines[:i]):
                    prev_lvl = (len(prev_line) - len(prev_line.lstrip()))//4
                    if prev_lvl != lvl:
                        continue
                    prev_line_no_inline_comment = prev_line
                    if '#' in prev_line_no_inline_comment:
                        prev_line_no_inline_comment = (
                            prev_line_no_inline_comment[
                                :prev_line_no_inline_comment.index('#')].rstrip() + '\n'
                        )
                    match = re.search(pattern, prev_line_no_inline_comment)
                    if match and not line.lstrip().startswith('#'):
                        prev_condition = match.group(1).strip()
                        break
                if prev_condition != condition:
                    continue
                # Impossible if statement found within else clause.
                # Replace all of the contents within this impossible if
                # statement with a single pass statement, and change
                # the condition to False. This keeps the code structure,
                # but will be compiled away by Cython
                # and/or the C compiler.
                new_lines.append(
                    '    '*(lvl + 1) + 'if False:  # Auto-inserted dummy if statement\n'
                )
                skip += 1  # Skip the inner if statement
                new_lines.append(
                    '    '*(lvl + 2) + 'pass       # Auto-inserted dummy if statement\n'
                )
                for line in lines[(i + 2):]:
                    if not line.strip() or line.lstrip().startswith('#'):
                        # Comment or empty line
                        skip += 1
                        continue
                    next_lvl = (len(line) - len(line.lstrip()))//4
                    if next_lvl <= lvl + 1:
                        # Line outside of the impossible if body
                        break
                    # Line inside the impossible if body
                    skip += 1
        return new_lines
    # The remove_impossible_if function can produce code like this:
    # if False:  # Delete
    #     ...    # Delete
    # else:      # Delete
    #     ...    # Unindent 4 spaces
    # As stated above, the function below serves to delete the falsy
    # if statement along with its entire body and the else statement,
    # and in addition unindent the entire else body by a single
    # indentation level. The simpler case where no else clause exists
    # is also taken care of. Also, if an elif statement exists,
    # this is converted into an if (effectively taking the place of the
    # original if statement).
    def remove_falsy_if(lines):
        pattern = r'^ *if +(False|0|0\.|\.0|0\.0) *:'
        pattern_else = r'^ *else *:'
        pattern_elif = r'^ *elif +(.+):'
        new_lines = []
        skip = 0
        for i, line in enumerate(lines):
            if skip > 0:
                skip -= 1
                continue
            line_no_inline_comment = line
            if '#' in line_no_inline_comment:
                line_no_inline_comment = (
                    line_no_inline_comment[:line_no_inline_comment.index('#')].rstrip() + '\n')
            match = re.search(pattern, line_no_inline_comment)
            if not match or line.lstrip().startswith('#'):
                new_lines.append(line)
                continue
            else:
                # Falsy if statement found.
                # Find matching else statement.
                lvl = (len(line) - len(line.lstrip()))//4
                for j, next_line in enumerate(lines[i+1:], i+1):
                    if not next_line.strip() or next_line.lstrip().startswith('#'):
                        continue
                    next_lvl = (len(next_line) - len(next_line.lstrip()))//4
                    next_line_no_inline_comment = next_line
                    if '#' in next_line_no_inline_comment:
                        next_line_no_inline_comment = (
                            next_line_no_inline_comment[
                                :next_line_no_inline_comment.index('#')].rstrip() + '\n'
                        )
                    if next_lvl == lvl and re.search(pattern_else, next_line_no_inline_comment):
                        # Else statement found.
                        # Skip the if statement and body.
                        skip = j - i - 1
                        # Find end of else body. Insert unindented else
                        # body lines as we go.
                        for k, next_line in enumerate(lines[j+1:], j+1):
                            if not next_line.strip() or next_line.lstrip().startswith('#'):
                                new_lines.append(next_line)
                                continue
                            next_lvl = (len(next_line) - len(next_line.lstrip()))//4
                            if next_lvl > lvl:
                                # Inside else body
                                next_line_unindented = next_line[4:]
                                new_lines.append(next_line_unindented)
                            else:
                                # Outside of else body
                                break
                        else:
                            # The else block is the last thing
                            # in the file.
                            return new_lines
                        # Skip over else body lines, as these has
                        # already been inserted.
                        skip = k - i - 1
                        break
                    elif next_lvl == lvl and re.search(pattern_elif, next_line):
                        # Elif statement found.
                        # Replace original if statement with this.
                        new_lines.append(' '*4*lvl + 'if' + next_line[4*lvl + 4:])
                        skip = j - i
                        break
                    elif next_lvl <= lvl:
                        # Outside if statement, no else found.
                        # Skip the if statement and body.
                        skip = j - i - 1
                        break
                else:
                    # The if statement is the last thing in the file
                    return new_lines
        return new_lines
    # Function which replaces for loops containing only a single pass
    # statement with a pass statement at the level of the for statement.
    def remove_empty_loop(lines):
        new_lines = []
        skip = 0
        for i, line in enumerate(lines):
            if skip:
                skip -= 1
                continue
            line_stripped = line.strip()
            if not line_stripped:
                new_lines.append(line)
                continue
            if line_stripped.startswith('#'):
                new_lines.append(line)
                continue
            if not line_stripped.startswith('for '):
                new_lines.append(line)
                continue
            # For loop detected
            indentation = len(line) - len(line.lstrip())
            loop_body = []
            for j, loop_line in enumerate(lines[i+1:], i + 1):
                loop_line_stripped = loop_line.strip()
                if not loop_line_stripped:
                    continue
                if loop_line_stripped.startswith('#'):
                    continue
                loop_indentation = len(loop_line) - len(loop_line.lstrip())
                if loop_indentation <= indentation:
                    break
                else:
                    loop_body.append(loop_line)
            if {loop_line.strip() for loop_line in loop_body}.difference({'', 'pass'}):
                # The loop contains a non-empty body
                new_lines.append(line)
                continue
            else:
                # The loop is empty
                skip = j - i - 1
                # new_lines.append(line)
                continue
        return new_lines
    # Run through the lines and replace any unswitch context manager
    # with the unswitched loop(s). Do this one unswitch nr at a time.
    def construct_trivial_statement(line):
        indentation = ' '*(len(line) - len(line.lstrip()))
        statement_lines = []
        statement_lines.append(
            ''.join([
                indentation,
                '# unswitch context manager replaced with trivial if statement\n',
            ])
        )
        statement_lines.append(f'{indentation}if True:\n')
        return statement_lines
    for current_unswitch_nr in range(n_unswitches):
        pattern = rf'^ *with +unswitch_{current_unswitch_nr} *(\(.*\))? *:'
        new_lines = []
        while True:
            # Repeatedly apply the remove_* functions until all
            # occurrences of unnecessary if statements are gone.
            if not no_optimization:
                lines_len_ori = -1
                while len(lines) != lines_len_ori:
                    lines_len_ori = len(lines)
                    lines = remove_double_if    (lines)
                    lines = remove_impossible_if(lines)
                    lines = remove_falsy_if     (lines)
            nounswitching = False
            nounswitching_counter = 0
            skip = 0
            for i, line in enumerate(lines):
                # Should this line be skipped?
                if skip > 0:
                    skip -= 1
                    continue
                # Set the state of nounswitching from
                # cython.nounswitching(bool) decorators.
                # We rely on this decorator not being used
                # on functions containing closures.
                line_lstripped = line.lstrip()
                match = re.search(r'^@ *cython *\. *nounswitching *(\(.*\))?', line_lstripped)
                if match:
                    nounswitching = True
                    nounswitching_arg = match.group(1)
                    if nounswitching_arg:
                        nounswitching_arg = nounswitching_arg[1:-1]
                        if nounswitching_arg:
                            nounswitching = eval(nounswitching_arg)
                    if nounswitching:
                        nounswitching_counter = 2
                elif line_lstripped.startswith('def '):
                    nounswitching_counter -= 1
                    if nounswitching_counter == 0:
                        nounswitching = False
                # Search for unswitch context managers with an unswitch
                # level matching the current level.
                line_no_inline_comment = line
                if '#' in line_no_inline_comment:
                    line_no_inline_comment = (
                        line_no_inline_comment[:line_no_inline_comment.index('#')].rstrip() + '\n')
                match = re.search(pattern, line_no_inline_comment.strip())
                if match and not line_lstripped.startswith('#'):
                    # Loop unswitching found.
                    # If optimizations are disabled or we are in a function with
                    # explicit disabling of unswitching, simply replace the
                    # with statement with "if True:".
                    if no_optimization or nounswitching:
                        new_lines += construct_trivial_statement(line)
                        continue
                    # Determine the number of indentation levels to do
                    # loop unswitching on. This is the n in unswitch(n).
                    # If not specified, this is set to infinity.
                    n = extract_n(match)
                    # If n is zero or negative,
                    # no loop unswitching should be performed.
                    if n < 1:
                        new_lines += construct_trivial_statement(line)
                        continue
                    # Search n nested loops upwards.
                    # Allowed constructs to pass are
                    # - for loops
                    # - while loops
                    # - if/elif/else statements
                    # - with statements
                    # - try/except/finally statements
                    indentation_lvl = (len(line) - len(line.lstrip()))//4
                    n_nested_loops = 0
                    smallest_lvl = indentation_lvl
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
                        # Break out of the search when
                        # encountering a special block header
                        # (such as a def statement).
                        if line_stripped.endswith(':'):
                            if not (   re.search('for .*:'    , line_stripped)
                                    or re.search('while .*:'  , line_stripped)
                                    or re.search('if .*:'     , line_stripped)
                                    or re.search('elif .*:'   , line_stripped)
                                    or re.search('else *:'    , line_stripped)
                                    or re.search('with .*:'   , line_stripped)
                                    or re.search('try ?.*:'   , line_stripped)
                                    or re.search('except ?.*:', line_stripped)
                                    or re.search('finally *:' , line_stripped)
                                    ):
                                break
                    # The loop lines to copy
                    loop_lines = lines[(i - J - 1):i]
                    # Now search downwards to find the if statements
                    # indented under the unswitch context manager.
                    outer_loop_lvl = (len(loop_lines[0]) - len(loop_lines[0].lstrip()))//4
                    patterns = ['{}if .*:'  .format('    '*(indentation_lvl + 1)),
                                '{}elif .*:'.format('    '*(indentation_lvl + 1)),
                                '{}else *:' .format('    '*(indentation_lvl + 1)),
                                ]
                    if_headers = []
                    if_bodies = []
                    loop_body = []
                    after_if = []
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
                        if lvl <= indentation_lvl:
                            under_unswitch = False
                        # Break out when all nested loops have ended
                        if lvl <= outer_loop_lvl:
                            break
                        loop_body_end_nr = i + 1 + j
                        # Record if/elif/else
                        if under_unswitch:
                            line_no_inline_comment = line
                            if '#' in line_no_inline_comment:
                                line_no_inline_comment = (
                                    line_no_inline_comment[
                                        :line_no_inline_comment.index('#')].rstrip() + '\n'
                                )
                            line_no_inline_comment_rstripped = line_no_inline_comment.rstrip()
                            if after_if:
                                after_if.append(line)
                                continue
                            else:
                                if lvl == indentation_lvl + 1:
                                    if any(re.search(pattern, line_no_inline_comment_rstripped)
                                           for pattern in patterns):
                                        if_headers.append(line)
                                        if_bodies.append([])
                                        if_body = if_bodies[-1]
                                    elif not after_if:
                                        after_if.append(line)
                                    continue
                        # Record code indented under if/elif/else
                        if under_unswitch:
                            if_body.append(line)
                            continue
                        # Record code also indented under the
                        # nested loops, but not part of the unswitching.
                        loop_body.append(line)
                    # If no else block is present, insert a dummy
                    # else block. This is needed because the unswitched
                    # loop should be run even if the if statement within
                    # it is false.
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
                            indentation = '    '*(indentation_lvl + 1)
                            if_headers.append(
                                f'{indentation}else:  # This is an auto-inserted else block\n'
                            )
                            if_bodies.append([
                                f'{indentation}    pass\n',  # If no statement is present above
                                f'{indentation}    # End of auto-inserted else block\n'
                            ])
                    # Stitch together the recorded pieces to perform
                    # the loop unswitching.
                    lines_unswitched = []
                    for if_header, if_body in zip(if_headers, if_bodies):
                        # Unswitched if/elif/else statement
                        indentation = ' '*(len(loop_lines[0]) - len(loop_lines[0].lstrip()))
                        if_header_stripped = if_header.strip()
                        lines_unswitched.append(
                              indentation
                            + if_header_stripped
                            + ('  # unswitched' if if_header_stripped.startswith('if ') else '')
                            + '\n'
                        )
                        # Nested loops
                        lines_unswitched += ['    ' + line for line in loop_lines]
                        # Body of unswitched if/elif/else statement
                        lines_unswitched += [line[4:] for line in if_body]
                        # Additional lines under the unswitch context
                        # manager which is not part
                        # of the actual unswitching.
                        lines_unswitched += after_if
                        # Remaining loop body
                        lines_unswitched += ['    ' + line for line in loop_body]
                    # Pop the nested for loops from new_lines,
                    # as these should only be included as the
                    # loop unswitching.
                    new_lines = new_lines[:-len(loop_lines)]
                    # Append the unswitched loop lines
                    new_lines += lines_unswitched
                    # Flag the line loop to skip the following lines
                    # which are already included in the loop
                    # unswitching lines.
                    skip = loop_body_end_nr - i
                else:
                    # No loop unswitching on this line
                    new_lines.append(line)
            # Done with all lines. Run them through again, in case of
            # nested use of the unswitching context manager.
            # Break out when no change has happened to any line.
            if lines == new_lines:
                break
            lines = new_lines
            new_lines = []
    # The lines have now been unswitched.
    # Repeatedly apply the remove_* functions until all
    # occurrences of unnecessary if statements are gone.
    if not no_optimization:
        lines_len_ori = -1
        while len(lines) != lines_len_ori:
            lines_len_ori = len(lines)
            lines = remove_double_if    (lines)
            lines = remove_impossible_if(lines)
            lines = remove_falsy_if     (lines)
            lines = remove_empty_loop   (lines)
    # Remove the cython.nounswitching decorator
    new_lines = []
    for line in lines:
        if line.lstrip().startswith('@cython.nounswitching'):
            continue
        new_lines.append(line)
    return new_lines



def unicode2ASCII(lines, no_optimization):
    new_lines = [commons.asciify(line) for line in lines]
    return new_lines



def remove_duplicate_declarations(lines, no_optimization):
    new_lines = []
    in_function = False
    in_locals = False
    declarations_outer = {}
    declarations_locals = {}
    for j, line in enumerate(lines):
        # Get function variable declarations within cython.locals()
        line_lstripped = line.lstrip()
        if line_lstripped.startswith('@cython.locals('):
            in_locals = True
            locals_begin = j
            declarations_locals = {}
        if in_locals:
            valid = False
            try:
                ast.parse(''.join(lines[locals_begin:j+1]).strip()[1:])
                valid = True
            except:
                pass
            if valid:
                locals_str = ''.join([
                    l.strip() for l in lines[locals_begin:j+1] if not l.lstrip().startswith('#')
                ])
                locals_str = locals_str[locals_str.index('(')+1:].strip(',')
                locals_str = re.sub(r', *]', ']', locals_str)
                locals_str = re.sub(r': *,', ':;', locals_str)
                for s in locals_str.split(','):
                    if '=' not in s:
                        continue
                    varname, vartype = s.replace(':;', ':,').split('=')
                    varname = varname.strip()
                    vartype = vartype.strip()
                    declarations_locals[varname] = vartype
                in_locals = False
        if line.startswith('def ') or (not in_function and line.startswith('    def ')):
            in_function = True
            declarations_inner = declarations_locals
            declarations_locals = {}
            first_linenr_of_function = len(new_lines) + 1
            n_indentation_def = len(line) - len(line_lstripped)
            indentation_def = ' '*n_indentation_def
        elif line and (
            (line[0] not in ' #\n')
            or (in_function and (set(line[:n_indentation_def+1]) - set(' #\n')))
        ):
            in_function = False
        declarations = declarations_inner if in_function else declarations_outer
        if line_lstripped.startswith('cython.declare('):
            indentation = ' '*(len(line) - len(line_lstripped))
            info = re.search(r'cython\.declare\((.*)\)', line).group(1).split('=')
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
                        print(
                            f'Warning: {varname} declared as both {vartype_prev} and {vartype}',
                            file=sys.stderr,
                        )
                else:
                    if in_function and first_linenr_of_function < len(new_lines):
                        # Move declaration to top of function
                        new_line = f'{indentation_def}    cython.declare({varname}={vartype})\n'
                        new_lines.insert(first_linenr_of_function, new_line)
                    else:
                        new_line = f'{indentation}cython.declare({varname}={vartype})\n'
                        new_lines.append(new_line)
                    declarations[varname] = vartype
        else:
            new_lines.append(line)
    return new_lines



def remove_self_assignments(lines, no_optimization):
    """This function removes lines like
        a = a
    Such statements ought to be harmless (and, if the type is known at
    compile time, the C compiler will optimize it away).
    However, when a is a memory view, Cython produces code with
    improper reference counting, which is the real reason for the
    existence of this function.
    This Cython bug is documented here:
      https://github.com/cython/cython/issues/3827
    """
    new_lines = []
    for line in lines:
        match = re.search(r'^(.+)=(.+)$', line)
        if match:
            lhs = match.group(1).strip()
            rhs = match.group(2).strip()
            if lhs == rhs and lhs.isidentifier():
                # Self-assignment found. Skip it.
                continue
        new_lines.append(line)
    return new_lines



def cython_decorators(lines, no_optimization):
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
                # Look for return type
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
                                    lonely = True
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
                # while a @cython.pheader should transform into a
                # @cython.ccall. Additional decorators should be placed
                # below.
                pyfuncs = ('__init__', '__cinit__', '__dealloc__')
                decorators = [decorator for decorator in
                              (('ccall' if headertype == 'pheader' else 'cfunc')
                               if all(' ' + pyfunc + '(' not in def_line
                                      for pyfunc in pyfuncs) else '',
                              'boundscheck({})'.format(no_optimization),
                              'cdivision(True)',
                              'initializedcheck({})'.format(no_optimization),
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



def __init__2__cinit__(lines, no_optimization):
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



def fix_addresses(lines, no_optimization):
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
            # Replace the first occurrence of cython.address(...) with
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



def malloc_realloc(lines, no_optimization):
    # Check and print memory copies by realloc()?
    check_realloc_copy = False
    # Replace the allocations
    counter = 0
    new_lines = []
    for line in lines:
        found_alloc = False
        for alloc in ('malloc', 'realloc'):
            if f'{alloc}(' in line and 'sizeof(' in line and not line.lstrip().startswith('#'):
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
                indentation = ' '*(len(line) - len(line.lstrip()))
                if check_realloc_copy and alloc == 'realloc':
                    ptr_old = re.search(rf'{alloc} *\((.+?),', line).group(1).strip()
                    ptr_old_cp = f'__realloc_copy_check_{counter}__'
                    new_lines.append(f'{indentation}cython.declare({ptr_old_cp}=\'{dtype}*\')\n')
                    new_lines.append(f'{indentation}{ptr_old_cp} = {ptr_old}\n')
                    counter += 1
                line = (line.replace(alloc, f'<{dtype}*> PyMem_' + alloc.capitalize()))
                new_lines.append(line)
                # Add exception
                LHS = line[:line.find('=')].strip()
                new_lines.append(indentation + 'if not ' + LHS + ':\n')
                new_lines.append(indentation + ' '*4
                                 + "raise MemoryError('Could not "
                                 + alloc + ' ' + LHS + "')\n")
                # Add printout on realloc copy
                if check_realloc_copy and alloc == 'realloc':
                    new_lines.append(f'{indentation}if {LHS} != {ptr_old_cp}:\n')
                    line_fmt = line.strip().replace('\'', '\\\'')
                    new_lines.append(f'{indentation}    __line_realloc__ = \'{line_fmt}\'\n')
                    new_lines.append(
                        f'{indentation}    fancyprint(f\'Copy due to realloc() by rank {{rank}} '
                        f'in the following line:\\n{{__line_realloc__}}\', '
                        f'fun=terminal.bold_cyan, wrap=False)\n'
                    )
        if not found_alloc:
            if line.lstrip().startswith('free('):
                # Normal frees
                indent_lvl = len(line) - len(line.lstrip())
                ptr = re.search(r'free\((.+?)\)', line).group(1)
                new_lines.append(' '*indent_lvl + 'if ' + ptr + ' is not NULL:\n')
                new_lines.append(' '*4 + line.replace('free(', 'PyMem_Free('))
            elif re.search(r'^gsl_.+?_free\(', line.lstrip()):
                # GSL frees
                indent_lvl = len(line) - len(line.lstrip())
                ptr = re.search(r'gsl_.+?_free\((.+?)\)', line).group(1)
                new_lines.append(' '*indent_lvl + 'if ' + ptr + ' is not NULL:\n')
                new_lines.append(' '*4 + line)
            else:
                new_lines.append(line)
    return new_lines



def C_casting(lines, no_optimization):
    new_lines = []
    # Transform to Cython syntax
    for line in lines:
        while True:
            match = re.search(r'[^\w.]cast *\(', line)
            if not match:
                break
            if line.lstrip().startswith('#'):
                break
            parens = 1
            brackets = 0
            curlys = 0
            comma = -1
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
                if (    comma == -1
                    and line[i] == ','
                    and parens == 1
                    and brackets == 0
                    and curlys == 0
                    ):
                    comma = i
                if parens == 0:
                    end = i
                    expression = line[match.end():comma].strip()
                    ctype = line[(comma + 1):end].strip(' "\',')
                    break
            line = line[:match.end() - 5] + '(<{}>({}))'.format(ctype, expression) + line[(end + 1):]
        new_lines.append(line)
    return new_lines



def find_extension_types(lines, no_optimization):
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



def make_types(filename, no_optimization):
    # Find Cython classes (extension types) in all .pyx files
    extension_types = {}
    for other_pyxfile in all_pyxfiles:
        module = other_pyxfile[:-4]
        with open(other_pyxfile, mode='r', encoding='utf-8') as pyxfile:
            code = pyxfile.read().split('\n')
        for i, line in enumerate(reversed(code)):
            if line == '# __pyxinfo__':
                code = code[-i:]
                break
        for i, line in enumerate(code):
            if line == '# Extension types implemented by this module:':
                line = code[i + 1].lstrip('#')
                for extension_type in line.split(','):
                    if ':' in extension_type:
                        extension_type, import_str = extension_type.split(':')
                        extension_types[extension_type.strip()] = import_str.strip()
                    else:
                        extension_type = extension_type.strip()
                        extension_types[extension_type] = 'from {} cimport {}'.format(
                            module, extension_type)
    # Do not write to the types file
    # if it already has the correct content.
    if os.path.isfile(filename):
        with open(filename, mode='r', encoding='utf-8') as types_file:
            existing_extension_types_content = types_file.read()
        try:
            existing_extension_types = eval(existing_extension_types_content)
            if existing_extension_types == extension_types:
                return
        except:
            print(f'Warning: Could not interpret the content of "{filename}".', file=sys.stderr)
    # Write the dictionary to the types file:
    with open(filename, mode='w', encoding='utf-8') as types_file:
        types_file.write(str(extension_types))



def make_pxd(filename, no_optimization):
    # Read in the extension types from the types file
    with open(filename_types, mode='r', encoding='utf-8') as pyxfile:
        extension_types_content = pyxfile.read()
    extension_types = eval(extension_types_content)
    pxd_lines = []
    header_lines = []
    # Read in the code
    pxd_filename = filename[:-3] + 'pxd'
    with open(filename, mode='r', encoding='utf-8') as pyxfile:
        code_str = pyxfile.read()
    code = code_str.split('\n')
    # Add the '# pxd hints' line
    pxd_lines.append('# pxd hints\n')
    # Find pxd hints of the form "pxd = '...'
    for line in code:
        m = re.match('^pxd *= *"(.*)"', line)
        if m:
            text = m.group(1).strip()
            if text != '"':
                pxd_lines.append(text + '\n')
        m = re.match("^pxd *= *'(.*)'", line)
        if m:
            text = m.group(1).strip()
            if text != "'":
                pxd_lines.append(text + '\n')
    # Find pxd hints of the form "pxd('...')"
    for line in code:
        m = re.match(r'^pxd *\( *"(.*)" *\)', line)
        if m:
            text = m.group(1).strip()
            if text != '"':
                pxd_lines.append(text + '\n')
        m = re.match(r"^pxd *\( *'(.*)' *\)", line)
        if m:
            text = m.group(1).strip()
            if text != "'":
                pxd_lines.append(text + '\n')
    # Find pxd hints of the form 'pxd = """'
    #                             int var1
    #                             double var2
    #                             """'
    for i, line in enumerate(code):
        if (   line.replace(' ', '').startswith('pxd="""')
            or line.replace(' ', '').startswith("pxd='''")):
            quote_type = '"""' if line.replace(' ', '').startswith('pxd="""') else "'''"
            for j, line in enumerate(code[(i + 1):]):
                if line.startswith(quote_type):
                    break
                pxd_lines.append(line + '\n')
    # Find pxd hints of the form 'pxd("""'
    #                             int var1
    #                             double var2
    #                             """)'
    for i, line in enumerate(code):
        if (   line.replace(' ', '').startswith('pxd("""')
            or line.replace(' ', '').startswith("pxd('''")):
            quote_type = '"""' if line.replace(' ', '').startswith('pxd("""') else "'''"
            for j, line in enumerate(code[(i + 1):]):
                if line.startswith(quote_type):
                    break
                pxd_lines.append(line + '\n')
    # Remove the '# pxd hints' line if no pxd hints were found
    if pxd_lines[-1].startswith('# pxd hints'):
        pxd_lines.pop()
    else:
        pxd_lines.append('\n')
    # Import all types with spaces (e.g. "long int")
    # from the commons module.
    types_with_spaces = [(key.replace(' ', ''), key)
                         for key in commons.C2np.keys()
                         if ' ' in key]
    # Include const and static types, (e.g. "const int")
    types_with_spaces += [
        (modifier + key.replace(' ', ''), f'{modifier} {key}') for key in commons.C2np.keys()
        for modifier in ('const', 'static')
    ]
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
                    function_args_bak = copy.deepcopy(function_args)
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
                    function_args_bak = copy.deepcopy(function_args)
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
                line_before = copy.deepcopy(code[i - 1])
                for j, arg in enumerate(function_args):
                    if '=*' in arg:
                        arg = arg[:-2]
                    N_parens = 0
                    inside_cython_header = False
                    for k, line in enumerate(reversed(code[:i])):
                        break_k = False
                        if len(line) > 0 and line[0] not in ('@', ' ', '#', ')'):
                            if not (line.lstrip().startswith(')') and not inside_cython_header):
                                # Above function decorators
                                break
                        N_oparnes = line.count('(')
                        N_cparens = line.count(')')
                        if N_cparens > 0:
                            inside_cython_header = True
                        N_parens += N_oparnes - N_cparens
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
                                        argtype = copy.deepcopy(line[(l + len(arg + '=')):])
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
                                        # Add spaces back to multi-word argument types
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
                                line_before = copy.deepcopy(line)
                                break
                        line_before = copy.deepcopy(line)
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
    # Remove all triple quotes with no indentation
    code_notriplequotes = []
    inside_quotes = {"'": False, '"': False}
    for line in code:
        for quote in ("'", '"'):
            if inside_quotes[quote]:
                if line.count(quote*3)%2:
                    inside_quotes[quote] = False
                break
        else:
            for quote in ("'", '"'):
                if line.count(quote*3)%2 and line[0] not in ' #':
                    inside_quotes[quote] = True
                    break
            else:
                code_notriplequotes.append(line)
    code = code_notriplequotes
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
    globals_code = copy.deepcopy(code)
    while True:
        done = True
        for i, line in enumerate(globals_code):
            if line.startswith('cython.declare('):
                done = False
                declaration = line[14:]
                declaration = re.sub(r'=.*\)', '', declaration) + ')'
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
    # Find declarations of non-built-in types (extension types)
    for vartype, vardeclaration in extension_types.items():
        # Skip declaration if vartype is defined in this module
        if vardeclaration.startswith('from {} cimport '.format(filename[:-4])):
            continue
        # vartype may contain ellipses. These should be replaced
        # with regular expressions which describe a single
        # variable name.
        vartype = vartype.replace('...', r'\w*')
        # Search the entire .pyx file for string of the form
        # varname = 'vartype'.
        if re.search((  r"""(^|[,;(\s])[^\W0-9]\w*\s*="""
                      + r"""\s*(?P<quote>['"]){vartype}\*?(?P=quote)"""
                      ).format(vartype=vartype),
                     code_str,
                     re.MULTILINE):
            # Match found.
            # Assume this is a declaration of the extension type.
            header_lines.append(vardeclaration)
            continue
        # The extension type may not be quoted
        if re.search((  r"""(^|[,;(\s])[^\W0-9]\w*\s*="""
                      + r"""\s*{vartype}\*?"""
                      ).format(vartype=vartype),
                     code_str,
                     re.MULTILINE):
            header_lines.append(vardeclaration)
            continue
        # For extension type attributes another syntax is used;
        # vartype varname
        # Also search after this.
        if re.search((  r"""\s*{vartype}\s*\*?\s*"""
                      + r"""(^|[,;(\s])[^\W0-9]\w*\s*"""
                      ).format(vartype=vartype),
                     code_str,
                     re.MULTILINE):
            header_lines.append(vardeclaration)
    # Remove duplicates
    header_lines = list(set(header_lines))
    # The header_lines are all independent imports or declarations.
    # To ensure deterministic content of the pxd file,
    # the header_lines are sorted.
    header_lines = sorted(header_lines)
    # Combine header_lines and pxd_lines
    while len(header_lines) > 0 and len(header_lines[0].strip()) == 0:
        header_lines = header_lines[1:]
    if header_lines:
        header_lines = ['# Non-built-in types'] + header_lines
    for i in range(len(header_lines)):
        header_lines[i] += '\n'
    total_lines = header_lines
    while len(pxd_lines) > 0 and len(pxd_lines[0].strip()) == 0:
        pxd_lines = pxd_lines[1:]
    if total_lines != []:
        total_lines.append('\n')
    total_lines += pxd_lines
    # Add import of certain float literals at the top, if supported
    legal_literals = check_float_literals()
    if legal_literals:
        total_lines = [
            '# Mathematical constants\n',
            'from libc.math cimport {}\n'.format(', '.join(legal_literals)),
            '\n',
        ] + total_lines
    # Add 'cimport cython' as the top line
    total_lines = [
        '# Get full access to all of Cython\n',
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
        total_lines = ['# This module does not expose any C-level functions or classes '
                       'to the outside world\n']
    # Do not write to pxd if it already exist in the correct state
    if os.path.isfile(pxd_filename):
        with open(pxd_filename, mode='r', encoding='utf-8') as pxdfile:
            existing_pxd_lines = pxdfile.readlines()
        if total_lines == existing_pxd_lines:
            return
    # Update/create .pxd
    with open(pxd_filename, mode='w', encoding='utf-8') as pxdfile:
        pxdfile.writelines(total_lines)



# Only run this file as a script when invoked directly
if __name__ == '__main__':
    # Interpret input argument
    filename = sys.argv[1]
    if not filename.endswith('.py') and not filename.endswith('.pyx'):
        raise Exception(f'Got "{filename}", which is neither a .py nor a .pyx file')
    filename_commons = sys.argv[2]
    no_optimization = False
    if len(sys.argv) > 3:
        if sys.argv[3] == '--no-optimizations':
            no_optimization = True
        else:
            filename_types = sys.argv[3]
            if not filename_types.endswith('.pyx'):
                raise Exception(
                    f'Got "{filename_types}" as the third argument, '
                    f'which should be either a .pyx file or '
                    f'the "--no-optimizations" flag'
                )
    if len(sys.argv) > 4:
        all_pyxfiles = [f for f in sys.argv[4:] if f.endswith('.pyx')]
        if not filename.endswith('.pyx'):
            raise Exception(
                f'Got "{filename}" which is not a .pyx file as the '
                f'first argument, while receiving more than three arguments'
            )
    # Import the non-compiled commons module
    commons_name = filename_commons[:-3]
    @contextlib.contextmanager
    def suppress_stdout():
        with open(os.devnull, mode='w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
    with suppress_stdout():
        commons = import_py_module('commons')
    # Perform operations
    if len(sys.argv) < 5:
        if filename.endswith('.py'):
            # A .py-file is passed.
            # Read in the lines of the file.
            with open(filename, mode='r', encoding='utf-8') as pyfile:
                lines = pyfile.readlines()
            # Apply transformations on the lines
            lines = cimport_cython               (lines, no_optimization)
            lines = commons.onelinerize          (lines                 )
            lines = remove_functions             (lines, no_optimization)
            lines = walrus                       (lines, no_optimization)
            lines = copy_on_import               (lines, no_optimization)
            lines = format_pxdhints              (lines, no_optimization)
            lines = cythonstring2code            (lines, no_optimization)
            lines = cython_structs               (lines, no_optimization)
            lines = cimport_commons              (lines, no_optimization)
            lines = cimport_function             (lines, no_optimization)
            lines = inline_iterators             (lines, no_optimization)
            lines = remove_trvial_branching      (lines, no_optimization)
            lines = float_literals               (lines, no_optimization)
            lines = constant_expressions         (lines, no_optimization)
            lines = unicode2ASCII                (lines, no_optimization)
            lines = loop_unswitching             (lines, no_optimization)
            lines = cython_decorators            (lines, no_optimization)
            lines = remove_duplicate_declarations(lines, no_optimization)
            lines = remove_self_assignments      (lines, no_optimization)
            lines = __init__2__cinit__           (lines, no_optimization)
            lines = fix_addresses                (lines, no_optimization)
            lines = malloc_realloc               (lines, no_optimization)
            lines = C_casting                    (lines, no_optimization)
            lines = find_extension_types         (lines, no_optimization)
            # Write the modified lines to the .pyx-file
            filename_pyx = filename[:-2] + 'pyx'
            with open(filename_pyx, mode='w', encoding='utf-8') as pyxfile:
                pyxfile.writelines(lines)
        elif filename.endswith('.pyx'):
            # A .pyx-file is passed.
            # Make the .pxd file.
            make_pxd(filename, no_optimization)
    else:
        # All the .pyx files are passed.
        # Make the types file, containing the definitions of all custom
        # types implemented in the .pyx files.
        make_types(filename, no_optimization)  # filename == filename_types
