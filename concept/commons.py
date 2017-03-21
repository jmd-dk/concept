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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# This module contains imports, Cython declarations and values
# of parameters common to all other modules. Each module should have
# 'from commons import *' as its first statement.



############################################
# Imports common to pure Python and Cython #
############################################
from __future__ import division  # Needed for Python3 division in Cython
# Miscellaneous modules
import collections, contextlib, ctypes, cython, imp, inspect, itertools
import os, re, shutil, sys, textwrap, types, unicodedata
# For math
# (note that numpy.array is purposely not imported directly into the
# global namespace, as this does not play well with Cython).
import numpy as np
from numpy import arange, asarray, empty, linspace, ones, zeros
# For plotting
import matplotlib
matplotlib.use('Agg')  # Use a matplotlib backend that does not require a running X-server
import matplotlib.pyplot as plt
# For I/O
from glob import glob
import h5py
# For fancy terminal output
import blessings
# For timing
from time import sleep, time
from datetime import timedelta



#############
# MPI setup #
#############
from mpi4py import MPI
cython.declare(master='bint',
               nprocs='int',
               rank='int',
               )
# MPI functions for communication (only ones used in the code)
comm = MPI.COMM_WORLD
Allgather  = comm.Allgather
Allgatherv = comm.Allgatherv
Barrier    = comm.Barrier
Gather     = comm.Gather
Isend      = comm.Isend
Reduce     = comm.Reduce
Recv       = comm.Recv
Send       = comm.Send
Sendrecv   = comm.Sendrecv
allreduce  = comm.allreduce
bcast      = comm.bcast
sendrecv   = comm.sendrecv
# Number of processes started with mpiexec
nprocs = comm.size
# The unique rank of the running process
rank = comm.rank
# Flag identifying the master/root process (that which have rank 0)
master = not rank



#################################
# Miscellaneous initialisations #
#################################
# The time before the main computation begins
if master:
    start_time = time()
# Initialize the pseudo-random number generator and declare the
# functions random and random_gassian, returning random numbers from
# the uniform distibution between 0 and 1 and a gaussian distribution
# with variable mean and spread, respectively.
# Both the pure Python and the compiled version of the functions use the
# Mersenne Twister algorithm to generate the random numbers.
# Despite of this, their exact implementation differs enough to make
# the generated sequence of random numbers completely different for
# pure Python and compiled runs.
# Use a seed which is different for each MPI rank. Also avoid a seed
# of 0, as this may lead to clashes with the default seed used by GSL.
random_seed = 1 + rank
if not cython.compiled:
    # In pure Python, use NumPy's random module
    np.random.seed(random_seed)
    random = np.random.random
    random_gaussian = np.random.normal
else:
    # Use GSL in compiled mode
    cython.declare(random_number_generator='gsl_rng*')
    random_number_generator = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(random_number_generator, random_seed)
    @cython.header(returns='double')
    def random():
        return gsl_rng_uniform_pos(random_number_generator)
    @cython.header(loc='double',
                   scale='double',
                   returns='double',
                   )
    def random_gaussian(loc=0, scale=1):
        return loc + gsl_ran_gaussian(random_number_generator, scale)
# Initialise a Blessings Terminal object,
# capable of producing fancy terminal formatting.
terminal = blessings.Terminal(force_styling=True)
# Customize matplotlib
matplotlib.rcParams.update({# Use a nice font that ships with matplotlib
                            'text.usetex'       : False,
                            'font.family'       : 'serif',
                            'font.serif'        : 'cmr10',
                            'mathtext.fontset'  : 'cm',
                            'axes.unicode_minus': False,
                            # Use outward pointing ticks
                            'xtick.direction': 'out',
                            'ytick.direction': 'out',
                            })
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
try:
    # Matplotlib ≥ 1.5
    matplotlib.rcParams.update({# Default colors
                                'axes.prop_cycle': matplotlib.cycler('color', color_cycle),
                                })
except:
    # Matplotlib < 1.5
    matplotlib.rcParams.update({# Default colors
                                'axes.color_cycle': color_cycle,
                                })



#############################
# Print and abort functions #
#############################
# Function which takes in a previously saved time as input
# and returns the time which has elapsed since then, nicely formatted.
def time_since(initial_time):
    """The argument should be in the UNIX time stamp format,
    e.g. what is returned by the time function.
    """
    # Time since initial_time, in seconds
    seconds = time() - initial_time
    # Construct the time interval with a sensible amount of
    # significant figures.
    milliseconds = 0
    # More than a millisecond; use whole milliseconds
    if seconds >= 1e-3:
        milliseconds = round(1e+3*seconds)
    interval = timedelta(milliseconds=milliseconds)
    # More than a second; use whole deciseconds
    if interval.total_seconds() >= 1e+0:
        seconds = 1e-1*round(1e-2*milliseconds)
        interval = timedelta(seconds=seconds)    
    # More than 10 seconds; use whole seconds
    if interval.total_seconds() >= 1e+1:
        seconds = round(1e-3*milliseconds)
        interval = timedelta(seconds=seconds)
    # Return a fitting string representation of the interval
    total_seconds = interval.total_seconds()
    if total_seconds == 0:
        return '< 1 ms'
    if total_seconds < 1:
        return '{} ms'.format(int(1e+3*total_seconds))
    if total_seconds < 10:
        return '{} s'.format(total_seconds)
    if total_seconds < 60:
        return '{} s'.format(int(total_seconds))
    if total_seconds < 3600:
        return str(interval)[2:]
    return str(interval)

# Function for printing messages as well as timed progress messages
def masterprint(*args, indent=0, sep=' ', end='\n', fun=None, wrap=True, **kwargs):
    if not master:
        return
    terminal_resolution = 80
    if args[0] == 'done':
        # End of progress message
        text = ' done after {}'.format(time_since(progressprint['time']))
        if len(args) > 1:
            text += sep + sep.join([str(arg) for arg in args[1:]])
        # Convert to proper Unicode characters
        text = unicode(text)
        # The progressprint['maxintervallength'] variable store the
        # length of the longest interval-message so far.
        if len(text) > progressprint['maxintervallength']:
            progressprint['maxintervallength'] = len(text)
        # Prepend the text with whitespace so that all future
        # interval-messages lign up to the right.
        text = ' '*(+ terminal_resolution
                    - progressprint['length']
                    - progressprint['maxintervallength']) + text
        # Apply supplied function to text
        if fun:
            text = fun(text)
        # Print out timing
        print(text, flush=True, end=end, **kwargs)
    else:
        # Stitch text pieces together
        text = sep.join([str(arg) for arg in args])
        # Convert to proper Unicode characters
        text = unicode(text)
        # Convert any paths in text (recognized by surrounding quotes)
        # to sensible paths.
        text = re.sub('"(.+?)"', lambda m: '"{}"'.format(sensible_path(m.group(1))), text)
        # Add indentation, and also wrap long message if wrap == True
        indentation = ' '*indent
        is_progress_message = text.endswith('...')
        if wrap:
            # Wrap text into lines which fit the terminal resolution.
            # Also indent all lines. Do this in a way that preserves
            # any newline characters already present in the text.
            lines = list(itertools.chain(*[textwrap.wrap(hard_line, terminal_resolution,
                                                         initial_indent=indentation,
                                                         subsequent_indent=indentation,
                                                         replace_whitespace=False,
                                                         break_long_words=False,
                                                         break_on_hyphens=False,
                                                         )
                                           for hard_line in text.split('\n')
                                           ])
                         )
            # If the text ends with '...', it is the start of a
            # progress message. In that case, the last line should
            # have some left over space to the right
            # for the upcomming "done in ???".
            if is_progress_message:
                maxlength = terminal_resolution - progressprint['maxintervallength'] - 1
                # Separate the last line from the rest
                last_line = lines.pop().lstrip()
                # The trailing ... should never stand on its own
                if last_line == '...':
                    last_line = lines.pop().lstrip() + ' ...'
                # Replace spaces before ... with underscores
                last_line = re.sub('( +)\.\.\.$', lambda m: '_'*len(m.group(1)) + '...', last_line)
                # Add the wrapped and indented last line
                # back in with the rest.
                lines += textwrap.wrap(last_line, maxlength,
                                       initial_indent=indentation,
                                       subsequent_indent=indentation,
                                       replace_whitespace=False,
                                       break_long_words=False,
                                       break_on_hyphens=False,
                                       )
                # Convert the inserted underscores back into spaces
                lines[-1] = re.sub('(_+)\.\.\.$', lambda m: ' '*len(m.group(1)) + '...', lines[-1])
                progressprint['length'] = len(lines[-1])
            text = '\n'.join(lines)          
        else:
            # Do not wrap the text into multiple lines,
            # regardless of the length of the text.
            # Add indentation.
            text = indentation + text
            # If the text ends with '...', it is the start of a
            # progress message. In that case, the text should
            # have some left over space to the right
            # for the upcomming "done in ???".
            if is_progress_message:
                progressprint['length'] = len(text)
        # General progress message handling
        if is_progress_message:
            progressprint['time'] = time()
            end = ''
        # Apply supplied function to text
        if fun:
            text = fun(text)
        # Print out message
        print(text, flush=True, end=end, **kwargs)
progressprint = {'maxintervallength': len(' done after ??? ms')}

# Function for printing warnings
def masterwarn(*args, skipline=True, prefix='Warning', wrap=True, **kwargs):
    if not master:
        return
    try:
        universals.any_warnings = True
    except:
        ...
    # Add initial newline (if skipline == True) to prefix
    # and append a colon.
    prefix = '{}{}{}'.format('\n' if skipline else '', prefix, ':' if args else '')
    # Print out message
    masterprint(prefix, *args, fun=terminal.bold_red, wrap=wrap, file=sys.stderr, **kwargs)

# Raised exceptions inside cdef functions do not generally propagte
# out to the caller. In places where exceptions are normally raised
# manualy, call this function with a descriptive message instead.
# Also, to ensure correct teardown of the MPI environment before
# exiting, call this function with exit_code=0 to shutdown a
# successfull CO𝘕CEPT run.
def abort(*args, exit_code=1, prefix='Aborting', **kwargs):
    # Print out final messages
    if master:
        if exit_code != 0:
            masterwarn(*args, prefix=prefix, **kwargs)
            sleep(1)
        masterprint('Total execution time: {}'.format(time_since(start_time)), **kwargs)
    # For a proper exit, all processes should reach this point
    if exit_code == 0:
        Barrier()
    # Ensure that every printed message is flushed
    sys.stderr.flush()
    sys.stdout.flush()
    sleep(1)
    # Teardown the MPI environment, either gently or by force
    if exit_code == 0:
        MPI.Finalize()
    else:
        comm.Abort(exit_code)
    # Exit Python
    sys.exit(exit_code)



###########
# C types #
###########
# Import the signed integer type ptrdiff_t
pxd = """
from libc.stddef cimport ptrdiff_t
"""
# C type names to NumPy dtype names
cython.declare(C2np='dict')
C2np = {# Booleans
        'bint': np.bool,
        # Integers
        'char'         : np.byte,
        'short'        : np.short,
        'int'          : np.intc,
        'long int'     : np.long,
        'long long int': np.longlong,
        'ptrdiff_t'    : np.intp,
        'Py_ssize_t'   : np.intp,
        # Unsgined integers
        'unsigned char'         : np.ubyte,
        'unsigned short'        : np.ushort,
        'unsigned int'          : np.uintc,
        'unsigned long int'     : np.uint,
        'unsigned long long int': np.ulonglong,
        'size_t'                : np.uintp,
        # Floating-point numbers
        'float'     : np.single,
        'double'    : np.double,
        'long float': np.longfloat,
        }
# In NumPy, binary operations between some unsigned int types (unsigned
# long int, unsigned long long int, size_t) and signed int types results
# in a double, rather than a signed int.
# Get around this bug by never using these particular unsigned ints.
if not cython.compiled:
    C2np['unsigned long int'] = C2np['long int']
    C2np['unsigned long long int'] = C2np['long long int']
    C2np['size_t'] = C2np['ptrdiff_t']



#####################
# Pure Python stuff #
#####################
# Definitions used by pure Python to understand Cython syntax
if not cython.compiled:
    # No-op decorators for Cython compiler directives
    def dummy_decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # Called as @dummy_decorator. Return function
            return args[0]
        else:
            # Called as @dummy_decorator(*args, **kwargs).
            # Return decorator
            return dummy_decorator
    # Already builtin: cfunc, inline, locals, returns
    for directive in ('boundscheck',
                      'cdivision',
                      'initializedcheck',
                      'wraparound',
                      'header',
                      'pheader',
                      ):
        setattr(cython, directive, dummy_decorator)
    # Address (pointers into arrays)
    def address(a):
        dtype = re.search('ctypeslib\.(.*?)_Array', np.ctypeslib.as_ctypes(a).__repr__()).group(1)
        return a.ctypes.data_as(ctypes.POINTER(eval('ctypes.' + dtype)))
    setattr(cython, 'address', address)
    # C allocation syntax for memory management
    def sizeof(dtype):
        # C dtype names to NumPy dtype names
        if dtype in C2np:
            dtype = C2np[dtype]
        elif dtype[-1] == '*':
            # Allocate pointer array of pointers (eg. int**).
            # Emulate these as lists of arrays.
            return [empty(1, dtype=sizeof(dtype[:-1]).dtype)]
        elif dtype.startswith('func_'):
            dtype='object'
        elif master:
            abort(dtype + ' not implemented as a NumPy dtype in commons.py')
        return np.array([1], dtype=dtype)
    def malloc(a):
        if isinstance(a, list):
            # Pointer to pointer represented as list of arrays
            return a
        return empty(int(a[0]), dtype=a.dtype)
    def realloc(p, a):
        # Reallocation of pointer assumed
        p.resize(int(a[0]), refcheck=False)
        return p
    def free(a):
        # NumPy arrays cannot be manually freed.
        # Resize the array to the minimal size.
        a.resize(0, refcheck=False)
    # Casting
    def cast(a, dtype):
        if not isinstance(dtype, str):
            dtype = dtype.__name__
        match = re.search('(.*)\[', dtype)
        if match:
            # Pointer to array cast assumed
            shape = dtype.replace(':', '')
            shape = shape[(shape.index('[') + 1):]
            shape = shape.rstrip()[:-1]
            if shape[-1] != ',':
                shape += ','
            shape = '(' + shape + ')'
            try:
                shape = eval(shape, inspect.stack()[1][0].f_locals)
            except:
                shape = eval(shape, inspect.stack()[1][0].f_globals)
            a = np.ctypeslib.as_array(a, shape)
            a = np.reshape(a[:np.prod(shape)], shape)
            return a
        elif dtype in C2np:
            # Scalar
            return C2np[dtype](a)
        else:
            # Extension type (Python class in pure Python)
            return a
    # Dummy fused types
    number = number2 = integer = floating = signed_number = signed_number2 = number_mv = []
    # Mathematical functions
    from numpy import (sin, cos, tan,
                       arcsin, arccos, arctan,
                       sinh, cosh, tanh,
                       arcsinh, arccosh, arctanh,
                       exp, log, log2, log10,
                       sqrt,
                       floor, ceil, round,
                       )
    cbrt = lambda x: x**(1/3)
    from math import erf, erfc
    # The closest thing to a Null pointer in pure Python is the
    # None object (not presently used inside pure Python code).
    NULL = None
    # Dummy functions and constants
    def dummy_func(*args, **kwargs):
        ...
    gsl_interp_accel_alloc = dummy_func
    gsl_interp_cspline     = dummy_func
    gsl_spline_alloc       = dummy_func
    # Dummy ℝ and ℤ dict for constant expressions
    class DummyDict(dict):
        def __getitem__(self, key):
            return key
    ℝ = DummyDict()
    ℤ = DummyDict()
    # The cimport function, which in the case of pure Python should
    # simply execute the statements passed to it as a string,
    # within the namespace of the call.
    def cimport(import_statement):
        import_statement = import_statement.strip()
        if import_statement.endswith(','):
            import_statement = import_statement[:-1]
        exec(import_statement, inspect.getmodule(inspect.stack()[1][0]).__dict__)
    # A dummy context manager for use with loop unswitching
    class DummyContextManager:
        def __call__(self, *args):
            return self
        def __enter__(self):
            ...
        def __exit__(self, *exc_info):
            ...
    unswitch = DummyContextManager()
# Function for building "structs" (really simple namespaces).
# In compiled mode, this function body will be copied and
# specialised for each kind of struct created.
# This function returns a struct and a dict over the passed fields.
# Note that these are not linked, meaning that you should not alter
# the values of a (struct, dict)-pair after creation if you are using
# the dict at all (the dict is useful for dynamic evaluations).
# In the current implementation, pointers are not allowed.
def build_struct(**kwargs):
    # Regardless of the input kwargs, bring it to the form {key: val}
    for key, val in kwargs.items():
        if isinstance(val, tuple):
            # Type and value given
            ctype, val = val
        else:
            # Only type given. Initialize value to zero.
            ctype = val
            try:
                val = C2np[ctype]()
            except:
                val = eval(ctype)()
        kwargs[key] = val
    for key, val in kwargs.items():
        if isinstance(val, str):
            # Evaluate value given as string expression
            namespace = {k: v for d in (globals(), kwargs) for k, v in d.items()}
            try:
                kwargs[key] = eval(val, namespace)
            except:
                ...
    if not cython.compiled:
        # In pure Python, emulate a struct by a simple namespace
        struct = types.SimpleNamespace(**kwargs)
    else:
        struct = ...  # To be added by pyxpp
    return struct, kwargs



###################################
# Cython imports and declarations #
###################################
pxd = """
# Get full access to all of Cython
cimport cython
# Import everything from cython_gsl,
# granting access to most of GNU Scientific Library.
from cython_gsl cimport *
# Functions for manual memory management
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# Function type definitions of the form func_returntype_argumenttypes
ctypedef bint    (*func_b_ddd)    (double, double, double)
ctypedef double  (*func_d_d)      (double)
ctypedef double  (*func_d_dd)     (double, double)
ctypedef double  (*func_d_ddd)    (double, double, double)
ctypedef double* (*func_dstar_ddd)(double, double, double)
# Create a fused number type containing all necessary numerical types
ctypedef fused number:
    cython.int
    cython.size_t
    cython.Py_ssize_t
    cython.float
    cython.double
# Create another fused number type, so that function arguments can have
# different specializations.
ctypedef fused number2:
    cython.int
    cython.size_t
    cython.Py_ssize_t
    cython.float
    cython.double
# Create integer and floating fused types
ctypedef fused integer:
    cython.int
    cython.size_t
    cython.Py_ssize_t
ctypedef fused floating:
    cython.float
    cython.double
# Create two identical signed number fused types
ctypedef fused signed_number:
    cython.int
    cython.float
    cython.double
ctypedef fused signed_number2:
    cython.int
    cython.float
    cython.double
# Mathematical functions
from libc.math cimport (sin, cos, tan,
                        asin as arcsin, 
                        acos as arccos, 
                        atan as arctan,
                        sinh, cosh, tanh,
                        asinh as arcsinh, 
                        acosh as arccosh, 
                        atanh as arctanh,
                        exp, log, log2, log10,
                        sqrt, cbrt,
                        erf, erfc,
                        floor, ceil, round,
                        )
"""



#####################
# Unicode functions #
#####################
# The pyxpp script convert all Unicode source code characters into
# ASCII using the below function.
@cython.pheader(# Arguments
                s='str',
                # Locals
                c='str',
                char_list='list',
                in_unicode_char='bint',
                pat='str',
                sub='str',
                unicode_char='str',
                unicode_name='str',
                returns='str',
                )
def asciify(s):
    char_list = []
    in_unicode_char = False
    unicode_char = ''
    for c in s:
        if in_unicode_char or ord(c) > 127:
            # Unicode
            in_unicode_char = True
            unicode_char += c
            try:
                unicode_name = unicodedata.name(unicode_char)
                # unicode_char is a string (of length 1 or more)
                # regarded as a single univode character.
                for pat, sub in unicode_subs.items():
                    unicode_name = unicode_name.replace(pat, sub)
                char_list.append('{}{}{}'.format(unicode_tags['begin'],
                                                 unicode_name,
                                                 unicode_tags['end'],
                                                 )
                                 )
                in_unicode_char = False
                unicode_char = ''
            except:
                ...
        else:
            # ASCII
            char_list.append(c)
    return ''.join(char_list)
cython.declare(unicode_subs='dict', unicode_tags='dict')
unicode_subs = {' ': '__SPACE__',
                '-': '__DASH__',
                }
unicode_tags = {'begin': 'BEGIN_UNICODE__',
                'end':   '__END_UNICODE',
                }

# The function below grants the code access to
# Unicode string literals by undoing the convertion of the
# asciify function above.
@cython.pheader(s='str', returns='str')
def unicode(s):
    return re.sub('{}(.*?){}'.format(unicode_tags['begin'], unicode_tags['end']), unicode_repl, s)
@cython.pheader(# Arguments
                match='object',  # re.match object
                # Locals
                pat='str',
                s='str', 
                sub='str',
                returns='str',
                )
def unicode_repl(match):
    s = match.group(1)
    for pat, sub in unicode_subs.items():
        s = s.replace(sub, pat)
    return unicodedata.lookup(s)

# This function takes in a number (string) and
# returns it written in Unicode subscript.
@cython.header(# Arguments
               s='str',
               # Locals
               c='str',
               returns='str',
               )
def unicode_subscript(s):
    return ''.join([unicode_subscripts[c] for c in s])
cython.declare(unicode_subscripts='dict')
unicode_subscripts = dict(zip('0123456789-+e',
                              [unicode(c) for c in ('₀', '₁', '₂', '₃', '₄',
                                                    '₅', '₆', '₇', '₈', '₉',
                                                    '₋', '₊', 'ₑ')]))

# This function takes in a number (string) and
# returns it written in Unicode superscript.
def unicode_superscript(s):
    return ''.join([unicode_superscripts[c] for c in s])
cython.declare(unicode_supercripts='dict')
unicode_superscripts = dict(zip('0123456789-+e',
                                [unicode(c) for c in ('⁰', '¹', '²', '³', '⁴',
                                                      '⁵', '⁶', '⁷', '⁸', '⁹',
                                                      '⁻', '', '×10')]))

# Function which converts a string containing (possibly) units
# to the corresponding numerical value.
@cython.pheader(# Arguments
                unit_str='str',
                namespace='dict',
                fail_on_error='bint',
                # Locals
                ASCII_char='str',
                after='str',
                before='str',
                c='str',
                i='Py_ssize_t',
                mapping='dict',
                operators='str',
                pat='str',
                rep='str',
                unicode_superscript='str',
                unit='object',  # double or NoneType
                unit_list='list',
                returns='object',  # double or NoneType
                )
def eval_unit(unit_str, namespace=None, fail_on_error=True):
    """This function is roughly equivalent to
    eval(unit_str, units_dict). Here however more stylized versions
    of unit_str are legal, e.g. 'm☉ Mpc Gyr⁻¹'.
    You may specify some other dict than the global units_dict
    as the namespace to perform the evaluation within,
    by passing it as a second argument.
    If you wish to allow for failures of evaluation, set fail_on_error
    to False. A failure will now not raise an exception,
    but merely return None.
    """
    # Ensure unicode
    unit_str = unicode(unit_str)
    # Replace multiple spaces wiht a single space
    unit_str = re.sub(r'\s+', ' ', unit_str).strip()
    # Replace spaces with an asterisk
    # if no operator is to be find on either side.
    operators = '+-*/^&|@%<>='
    unit_list = list(unit_str)
    for i, c in enumerate(unit_list):
        if c != ' ' or (i == 0 or i == len(unit_list) - 1):
            continue
        before = unit_list[i - 1]
        after  = unit_list[i + 1]
        if before not in operators and after not in operators:
            unit_list[i] = '*'
    unit_str = ''.join(unit_list)
    # Various replacements
    mapping = {'×'          : '*',
               '^'          : '**',
               unicode('m☉'): 'm_sun',
               }
    for pat, rep in mapping.items():
        unit_str = unit_str.replace(pat, rep)
    # Convert superscript to power
    unit_str = re.sub(unicode('[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+'), r'**(\g<0>)', unit_str)
    # Convert unicode superscript ASCII text
    for ASCII_char, unicode_superscript in unicode_superscripts.items():
        if unicode_superscript:
            unit_str = unit_str.replace(unicode_superscript, ASCII_char)
    # Insert an asterisk between letters and numbers
    # (though not the other way around).
    unit_str = re.sub(r'([^_a-zA-Z0-9\.][0-9\.\(\)]+)([_a-zA-Z])', r'\g<1>*\g<2>', unit_str)
    # Evaluate the transformed unit string
    if namespace is None:
        namespace = units_dict
    if fail_on_error:
        unit = eval(unit_str, namespace)
    else:
        try:
            unit = eval(unit_str, namespace)
        except:
            unit = None
    return unit



##################################
# Paths to directories and files #
##################################
# The paths are stored in the top_dir/.paths file
cython.declare(paths='dict')
top_dir = os.path.abspath('.')
while True:
    if '.paths' in os.listdir(top_dir):
        break
    elif master and top_dir == '/':
        abort('Cannot find the .paths file!')
    top_dir = os.path.dirname(top_dir)
paths_module = imp.load_source('paths', top_dir + '/.paths')
paths = {key: value for key, value in paths_module.__dict__.items()
         if isinstance(key, str) and not key.startswith('__')}
# Function for converting an absolute path to its "sensible" form.
# That is, this function returns the relative path with respect to the
# concept directory, if it is no more than one directory above the
# concept directory. Otherwise, return the absolute path back again.
@cython.header(# Arguments
               path='str',
               # Locals
               relpath='str',
               returns='str',
               )
def sensible_path(path):
    if not path:
        return path
    relpath = os.path.relpath(path, paths['concept_dir'])
    if relpath.startswith('../../'):
        return path
    return relpath



##########################
# Command line arguments #
##########################
# Handle command line arguments given to the Python interpreter
# (not those explicitly given to the concept script).
# Construct a dict from command line arguments of the form
# "param='value'"
cython.declare(argd='dict',
               globals_dict='dict',
               jobid='unsigned long long int',
               )
argd = {}
for arg in sys.argv:
    with contextlib.suppress(Exception):
        exec(arg, argd)
globals_dict = {}
exec('', globals_dict)
for key in globals_dict.keys():
    argd.pop(key, None)
# Extract command line arguments from the dict.
# If not given, give the arguments some default values.
# The parameter file
paths['params'] = argd.get('params', '')
# The jobid of the current run
jobid = int(argd.get('jobid', 0))



######################
# Read paramter file #
######################
# Read in the content of the parameter file.
# All handling of parameters defined in the parameter file
# will be done later
cython.declare(params_file_content='str')
params_file_content = ''
if os.path.isfile(paths['params']):
    with open(paths['params'], encoding='utf-8') as params_file:
        params_file_content = params_file.read()



###########################
# Dimensionless constants #
###########################
cython.declare(machine_ϵ='double',
               π='double',
               ϱ_vacuum='double',
               ထ='double',
               )
machine_ϵ = np.finfo(C2np['double']).eps
π = np.pi
ϱ_vacuum = 1e+2*machine_ϵ
ထ = cast(np.inf, 'double')



##################
# Physical units #
##################
# Dicts relating all implemented units to the basic
# three units (pc, yr, m_sun). Julian years are used.
unit_length_relations = {'pc' : 1,
                         'kpc': 1e+3,
                         'Mpc': 1e+6,
                         'Gpc': 1e+9,
                         'AU' :                π/(60*60*180),
                         'm'  :                π/(60*60*180)/149597870700,
                         'mm' : 1e-3          *π/(60*60*180)/149597870700,
                         'cm' : 1e-2          *π/(60*60*180)/149597870700,
                         'km' : 1e+3          *π/(60*60*180)/149597870700,
                         'ly' :      299792458*π/(60*60*180)/149597870700/(1/365.25/24/60/60),
                         'kly': 1e+3*299792458*π/(60*60*180)/149597870700/(1/365.25/24/60/60),
                         'Mly': 1e+6*299792458*π/(60*60*180)/149597870700/(1/365.25/24/60/60),
                         'Gly': 1e+9*299792458*π/(60*60*180)/149597870700/(1/365.25/24/60/60),
                         }
unit_time_relations = {'yr' :     1,
                       'kyr':     1e+3,
                       'Myr':     1e+6,
                       'Gyr':     1e+9,
                       'day':     1/365.25,
                       'hr' :     1/365.25/24,
                       'minutes': 1/365.25/24/60,
                       's':       1/365.25/24/60/60,
                       }
unit_mass_relations = {'m_sun' : 1,
                       'km_sun': 1e+3,
                       'Mm_sun': 1e+6,
                       'Gm_sun': 1e+9,
                       'kg'    :      1/1.989e+30,
                       'g'     : 1e-3*1/1.989e+30,
                       }
# Attempt to read in the parameter file. This is really only
# to get any user defined units, as the parameter file will
# be properly read in later. First construct a namespace in which the
# parameters can be read in.
user_params = vars(np).copy()
user_params.update({# The paths dict
                    'paths': paths,
                    # Modules
                    'numpy': np,
                    'np'   : np,
                    'os'   : os,
                    're'   : re,
                    'sys'  : sys,
                    # Functions
                    'rand'  : np.random.random,
                    'random': np.random.random,
                    # MPI variables
                    'master': master,
                    'nprocs': nprocs,
                    'rank'  : rank,
                    # Constants
                    'machine_ϵ' : machine_ϵ,
                    'eps'       : machine_ϵ,
                    unicode('π'): π,
                    unicode('ထ'): ထ,
                    'ထ'         : ထ,
                    })
# Add units to the user_params namespace.
# These do not represent the choice of units; the names should merely
# exist to avoid errors when reading the parameter file.
user_params.update(unit_length_relations)
user_params.update(unit_time_relations)
user_params.update(unit_mass_relations)
# Execute the content of the parameter file in the namespace defined
# by user_params in order to get the user defined units.
with contextlib.suppress(Exception):
    exec(params_file_content, user_params)
# The names of the three fundamental units,
# all with a numerical value of 1. If these are not defined in the
# parameter file, give them some reasonable values.
cython.declare(unit_length='str',
               unit_time='str',
               unit_mass='str',
               )
unit_length = user_params.get('unit_length', 'Mpc')
unit_time   = user_params.get('unit_time',   'Gyr')
unit_mass   = user_params.get('unit_mass',   '1e+10*m_sun')
# Construct a struct containing the values of all units.
# Note that 'min' is not a good name for minutes,
# as this name is already taken by the min function.
units, units_dict = build_struct(# Values of basic units,
                                 # determined from the choice of fundamental units.
                                 pc     = ('double', 1/eval_unit(unit_length, unit_length_relations)),
                                 yr     = ('double', 1/eval_unit(unit_time,   unit_time_relations)),
                                 m_sun  = ('double', 1/eval_unit(unit_mass,   unit_mass_relations)),
                                 # Prefixes of the basic units
                                 kpc    = ('double', 'unit_length_relations["kpc"]*pc'),
                                 Mpc    = ('double', 'unit_length_relations["Mpc"]*pc'),
                                 Gpc    = ('double', 'unit_length_relations["Gpc"]*pc'),
                                 kyr    = ('double', 'unit_time_relations["kyr"]*yr'),
                                 Myr    = ('double', 'unit_time_relations["Myr"]*yr'),
                                 Gyr    = ('double', 'unit_time_relations["Gyr"]*yr'),
                                 km_sun = ('double', 'unit_mass_relations["km_sun"]*m_sun'),
                                 Mm_sun = ('double', 'unit_mass_relations["Mm_sun"]*m_sun'),
                                 Gm_sun = ('double', 'unit_mass_relations["Gm_sun"]*m_sun'),
                                 # Non-basic units
                                 ly      = ('double', 'unit_length_relations["ly"]*pc'),
                                 kly     = ('double', 'unit_length_relations["kly"]*pc'),
                                 Mly     = ('double', 'unit_length_relations["Mly"]*pc'),
                                 Gly     = ('double', 'unit_length_relations["Gly"]*pc'),
                                 AU      = ('double', 'unit_length_relations["AU"]*pc'),
                                 m       = ('double', 'unit_length_relations["m"]*pc'),
                                 mm      = ('double', 'unit_length_relations["mm"]*pc'),
                                 cm      = ('double', 'unit_length_relations["cm"]*pc'),
                                 km      = ('double', 'unit_length_relations["km"]*pc'),
                                 day     = ('double', 'unit_time_relations["day"]*yr'),
                                 hr      = ('double', 'unit_time_relations["hr"]*yr'),
                                 minutes = ('double', 'unit_time_relations["minutes"]*yr'),
                                 s       = ('double', 'unit_time_relations["s"]*yr'),
                                 kg      = ('double', 'unit_mass_relations["kg"]*m_sun'),
                                 g       = ('double', 'unit_mass_relations["g"]*m_sun'),
                                 )



################################################################
# Import all user specified parameters from the parameter file #
################################################################
# Function for converting non-iterables to lists of one element
# (and cast iterables to lists).
def any2iter(val):
    if not hasattr(val, '__iter__') or hasattr(val, '__len__'):
        val = np.ravel(val)
    return list(val)
# Subclass the dict to create a dict-like object which keeps track of
# the number of lookups on each key. This is used to identify unknown
# (and therefore unused) parameters defined by the user.
# Transform all keys to unicode during lookups and assignments.
class DictWithCounter(dict):
    def __init__(self, d):
        self.counter = collections.defaultdict(int)
        super().__init__(d)
    # Lookup methods, which increase the count by 1
    def __getitem__(self, key):
        key = unicode(key)
        self.counter[key] += 1
        return super().__getitem__(key)
    def get(self, key, default=None):
        key = unicode(key)
        self.counter[key] += 1
        return super().get(key, default)
    def __contains__(self, key):
        key = unicode(key)
        self.counter[key] += 1
        return super().__contains__(key)
    # Other methods
    def __setitem__(self, key, value):
        key = unicode(key)
        return super().__setitem__(key, value)
    def use(self, key):
        key = unicode(key)
        self.counter[key] += 1
    def useall(self):
        for key in self.keys():
            self.use(key)
    # def unuse(self, key):
        # self.use(key, times=-1)
    # List of specified but unused parameters, not including parameters
    # starting with an '_'.
    @property
    def unused(self):
        list_of_unused = []
        for key in self.keys():
            key = unicode(key)
            if self.counter[key] < 1 and not key.startswith('_'):
                list_of_unused.append(key)
        return list_of_unused
# Dict-like object constituting the namespace for the statements
# in the user specified parameter file.
# Note that the previously defined user_params is overwritten.
# Everything from NumPy should be available when defining parameters.
user_params = DictWithCounter(vars(np).copy())
# Units from the units struct should be available
# when defining parameters.
user_params.update(units_dict)
# Additional things which should be available when defining parameters
user_params.update({# The paths dict
                    'paths': paths,
                    # Modules
                    'numpy': np,
                    'np'   : np,
                    'os'   : os,
                    're'   : re,
                    'sys'  : sys,
                    # Functions
                    'rand'  : np.random.random,
                    'random': np.random.random,
                    # MPI variables
                    'master': master,
                    'nprocs': nprocs,
                    'rank'  : rank,
                    # Constants
                    'machine_ϵ' : machine_ϵ,
                    'eps'       : machine_ϵ,
                    unicode('π'): π,
                    unicode('ထ'): ထ,
                    'ထ'         : ထ,
                    })
# At this point, user_params does not contain actual parameters.
# Mark all items in user_params as used.
user_params.useall()
# "Import" the parameter file by executing it
# in the namespace defined by the user_params namespace.
exec(params_file_content, user_params)
# Also mark the unit-parameters as used
for u in ('length', 'time', 'mass'):
    user_params.use('unit_{}'.format(u))
# The parameters are now being processed as follows:
# - All parameters are explicitly casted to their appropriate type.
# - Parameters which should be integer are rounded before casted.
# - Parameters not present in the parameter file
#   will be given default values.
# - Spaces are removed from the 'snapshot_type' parameter, and all
#   characters are converted to lowercase.
# - The 'output_times' are sorted and duplicates (for each type of
#   output) are removed.
# - Ellipses used as dictionary values will be replaced by whatever
#   non-ellipsis value also present in the same dictionary. If using
#   Python ≥ 3.6.0, where dictionaries are ordered, ellipses will be
#   replaced by their nearest, previous non-ellipsis value. Below is the
#   function replace_ellipsis which takes care of this replacement.
# - Paths below or just one level above the concept directory are made
#   relative to this directory in order to reduce screen clutter.
# - Colors are transformed to (r, g, b) arrays. Below is the function
#   to_rgb which handles this transformation.
def to_int(value):
    return int(round(float(value)))
def to_rgb(value):
    if isinstance(value, int) or isinstance(value, float):
        value = str(value)
    try:
        rgb = np.array(matplotlib.colors.ColorConverter().to_rgb(value), dtype=C2np['double'])
    except:
        # Could not convert value to color
        return np.array([-1, -1, -1])
    return rgb
def replace_ellipsis(d):
    parameter = ''
    for val in d.values():
        if val is not ... and any(any2iter(val)):
            parameter = val
            break
    for key, val in d.items():
        if val is ...:
            d[key] = parameter
        elif any(any2iter(val)):
            parameter = val
cython.declare(# Input/output
               IC_file='str',
               snapshot_type='str',
               output_dirs='dict',
               output_bases='dict',
               output_times='dict',
               powerspec_select='dict',
               powerspec_plot_select='dict',
               render_select='dict',
               autosave='double',
               # Numerical parameter
               boxsize='double',
               ewald_gridsize='Py_ssize_t',
               φ_gridsize='ptrdiff_t',
               p3m_scale='double',
               p3m_cutoff='double',
               softeningfactors='dict',
               R_tophat='double',
               # Cosmology
               H0='double',
               Ωm='double',
               Ωr='double',
               ΩΛ='double',
               a_begin='double',
               t_begin='double',
               # Graphics
               render_colors='dict',
               bgcolor='double[::1]',
               resolution='int',
               terminal_render_colormap='str',
               terminal_render_resolution='unsigned int',
               # Physics
               forces='dict',
               w_eos='dict',
               # Simlation options
               use_φ='bint',
               use_p3m='bint',
               fftw_rigor='str',
               # Debugging options
               enable_debugging='bint',
               enable_Ewald='bint',
               enable_Hubble='bint',
               # Hidden parameters
               special_params='dict',
               )
# Input/output
IC_file = sensible_path(str(user_params.get('IC_file', '')))
snapshot_type = (str(user_params.get('snapshot_type', 'standard'))
                 .lower().replace(' ', ''))
if master and snapshot_type not in ('standard', 'gadget2'):
    abort('Does not recognize snapshot type "{}"'.format(user_params['snapshot_type']))
output_dirs = dict(user_params.get('output_dirs', {}))
replace_ellipsis(output_dirs)
for kind in ('snapshot', 'powerspec', 'render'):
    output_dirs[kind] = str(output_dirs.get(kind, paths['output_dir']))
    if not output_dirs[kind]:
        output_dirs[kind] = paths['output_dir']
output_dirs = {key: sensible_path(path) for key, path in output_dirs.items()}
output_bases = dict(user_params.get('output_bases', {}))
replace_ellipsis(output_bases)
for kind in ('snapshot', 'powerspec', 'render'):
    output_bases[kind] = str(output_bases.get(kind, kind))
output_times = dict(user_params.get('output_times', {}))
replace_ellipsis(output_times)
# Output times not explicitly written as either of type 'a' or 't'
# is understood as being of type 'a'.
for time_param in ('a', 't'):
    output_times[time_param] = dict(output_times.get(time_param, {}))
    replace_ellipsis(output_times[time_param])
for key, val in output_times.items():
    if key not in ('a', 't'):
        output_times['a'][key] = (  tuple(any2iter(output_times['a'].get(key, ())))
                                  + tuple(any2iter(val)))
for time_param in ('a', 't'):
    output_times[time_param] = dict(output_times.get(time_param, {}))
    for kind in ('snapshot', 'powerspec', 'render', 'terminal render'):
        output_times[time_param][kind] = output_times[time_param].get(kind, ())
output_times = {time_param: {key: tuple(sorted(set([float(eval_unit(nr)
                                                          if isinstance(nr, str) else nr)
                                                    for nr in any2iter(val)
                                                    if nr or nr == 0])))
                             for key, val in output_times[time_param].items()}
                for time_param in ('a', 't')}
powerspec_select = {'all': True}
if 'powerspec_select' in user_params:
    if isinstance(user_params['powerspec_select'], dict):
        powerspec_select = user_params['powerspec_select']
        replace_ellipsis(powerspec_select)
    else:
        powerspec_select = {'all': user_params['powerspec_select']}
powerspec_select = {key.lower(): bool(val) for key, val in powerspec_select.items()}
powerspec_plot_select = {'all': True}
if 'powerspec_plot_select' in user_params:
    if isinstance(user_params['powerspec_plot_select'], dict):
        powerspec_plot_select = user_params['powerspec_plot_select']
        replace_ellipsis(powerspec_plot_select)
    else:
        powerspec_plot_select = {'all': user_params['powerspec_plot_select']}
powerspec_plot_select = {key.lower(): bool(val) for key, val in powerspec_plot_select.items()}
render_select = {'all': True}
if 'render_select' in user_params:
    if isinstance(user_params['render_select'], dict):
        render_select = user_params['render_select']
        replace_ellipsis(render_select)
    else:
        render_select = {'all': user_params['render_select']}
render_select = {key.lower(): bool(val) for key, val in render_select.items()}
autosave = float(0 if not user_params.get('autosave', 0) else user_params.get('autosave', 0))
if autosave < 0:
    autosave = 0
# Numerical parameters
boxsize = float(user_params.get('boxsize', 1))
ewald_gridsize = to_int(user_params.get('ewald_gridsize', 64))
φ_gridsize = to_int(user_params.get('φ_gridsize', 64))
p3m_scale = float(user_params.get('p3m_scale', 1.25))
p3m_cutoff = float(user_params.get('p3m_cutoff', 4.8))
softeningfactors = dict(user_params.get('softeningfactors', {}))
replace_ellipsis(softeningfactors)
for kind in ('dark matter particles', ):
    softeningfactors[kind] = float(softeningfactors.get(kind, 0.03))
R_tophat = float(user_params.get('R_tophat', 8*units.Mpc))
# Cosmology
H0 = float(user_params.get('H0', 70*units.km/(units.s*units.Mpc)))
Ωr = float(user_params.get('Ωr', 0))
Ωm = float(user_params.get('Ωm', 0.3))
ΩΛ = float(user_params.get('ΩΛ', 0.7))
a_begin = float(user_params.get('a_begin', 1))
t_begin = float(user_params.get('t_begin', 0))
# Physics
forces = dict(user_params.get('forces', {}))
default_force_methods = {'gravity': 'pm',
                         }
replace_ellipsis(forces)
for key, val in forces.items():
    if isinstance(val, dict):
        forces[key] = [(key2, val2) for key2, val2 in val.items()]
    elif isinstance(val, str):
        forces[key] = [(val, default_force_methods[val.lower()])]
    elif isinstance(val, tuple) or isinstance(val, list):
        if len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], str):
            if val[1] in default_force_methods:
                forces[key] = [(val[0], default_force_methods[val[0]]),
                               (val[1], default_force_methods[val[1]])
                               ]
            else:
                forces[key] = [tuple(val)]
        else:
            new_val = []
            for el in val:
                if isinstance(el, dict):
                    new_val += [(key2, val2) for key2, val2 in el.items()]
                elif isinstance(el, tuple) or isinstance(el, list):
                    if len(el) == 2:
                        new_val += [(el[0], el[1])]
                elif isinstance(el, str):
                    new_val += [(el, default_force_methods[el.lower()])]
            forces[key] = new_val
    for i, t in enumerate(forces[key]):
        t = (t[0].lower(), t[1].lower())
        t = (t[0].replace(' ', '_'), t[1].replace(' ', '_'))
        t = (t[0].replace('-', '_'), t[1].replace('-', '_'))
        t = (t[0].replace('^', '' ), t[1].replace('^', '' ))
        for n in range(10):
            t = (t[0].replace(unicode_superscript(str(n)), str(n)),
                 t[1].replace(unicode_superscript(str(n)), str(n)))
        forces[key][i] = t
force_methods = {t[1] for l in forces.values() for t in l}
w_eos = dict(user_params.get('w_eos', {}))
replace_ellipsis(w_eos)
# Simulation options
if (   'φ_gridsize' in user_params
    or (set(('pm', 'p3m')) & force_methods)
    or any([output_times[time_param]['powerspec'] for time_param in ('a', 't')])):
    use_φ = bool(user_params.get('use_φ', True))
else:
    use_φ = bool(user_params.get('use_φ', False))
if 'p3m' in force_methods:
    use_p3m = bool(user_params.get('use_p3m', True))
else:
    use_p3m = bool(user_params.get('use_p3m', False))
fftw_rigor = user_params.get('fftw_rigor', 'estimate').lower()
# Graphics
render_colors = {}
if 'render_colors' in user_params:
    if isinstance(user_params['render_colors'], dict):
        render_colors = user_params['render_colors']
        replace_ellipsis(render_colors)
    else:
        render_colors = {'all': user_params['render_colors']}
render_colors = {key.lower(): to_rgb(val) for key, val in render_colors.items()}
bgcolor = to_rgb(user_params.get('bgcolor', 'black'))
resolution = to_int(user_params.get('resolution', 1080))
terminal_render_colormap = str(user_params.get('terminal_render_colormap', 'gnuplot2'))
terminal_render_resolution = to_int(user_params.get('terminal_render_resolution', 80))
# Debugging options
enable_debugging = bool(user_params.get('enable_debugging', False))
enable_Ewald = bool(user_params.get('enable_Ewald',
                                    True if 'pp' in force_methods else False))
enable_Hubble = bool(user_params.get('enable_Hubble', True))
# Extra hidden parameters via the special_params variable
special_params = dict(user_params.get('special_params', {}))



######################
# Global allocations #
######################
# Useful for temporary storage of 3D vector
cython.declare(vector='double*',
               vector_mv='double[::1]',
               )
vector = malloc(3*sizeof('double'))
vector_mv = cast(vector, 'double[:3]')



##############
# Universals #
##############
# Universals are non-constant cross-module level global variables.
# These are stored in the following struct.
# Note that you should never use the universals_dict, as updates to
# universals are not reflected in universals_dict.
universals, universals_dict = build_struct(# Flag specifying whether any warnings have been given
                                           any_warnings=('bint', False),
                                           # Scale factor and cosmic time
                                           a=('double', a_begin),
                                           t=('double', t_begin),
                                           )



############################################
# Derived and internally defined constants #
############################################
cython.declare(light_speed='double',
               G_Newton='double',
               φ_gridsize3='long long int',
               φ_gridsize_half='Py_ssize_t',
               slab_size_padding='ptrdiff_t',
               ewald_file='str',
               powerspec_dir='str',
               powerspec_base='str',
               powerspec_times='dict',
               render_dir='str',
               render_base='str',
               render_times='dict',
               snapshot_dir='str',
               snapshot_base='str',
               snapshot_times='dict',
               terminal_render_times='dict',
               ϱ_crit='double',
               ϱ_mbar='double',
               pm_fac_const='double',
               longrange_exponent_fac='double',
               p3m_cutoff_phys='double',
               p3m_scale_phys='double',
               )
# Extract output variables from output dicts
snapshot_dir          = output_dirs['snapshot']
snapshot_base         = output_bases['snapshot']
snapshot_times        = {time_param: output_times[time_param]['snapshot'] 
                         for time_param in ('a', 't')}
powerspec_dir         = output_dirs['powerspec']
powerspec_base        = output_bases['powerspec']
powerspec_times       = {time_param: output_times[time_param]['powerspec'] 
                         for time_param in ('a', 't')}
render_dir            = output_dirs['render']
render_base           = output_bases['render']
render_times          = {time_param: output_times[time_param]['render'] 
                         for time_param in ('a', 't')}
terminal_render_times = {time_param: output_times[time_param]['terminal render'] 
                         for time_param in ('a', 't')}
# The speed of light in vacuum
light_speed = 299792458*units.m/units.s
# Newton's gravitational constant
G_Newton = 6.6738e-11*units.m**3/(units.kg*units.s**2)
# The average, comoing density (the critical
# comoving density since we only study flat universes).
# We allow ourselves to use the symbol ϱ in ϱ_crit even though
# this refers to ϱ = a**(3*(1 + w))*ρ where ρ is the proper density,
# a is the scale factor and w is the equation of state parameter.
ϱ_crit = 3*H0**2/(8*π*G_Newton)
# The average, comoving matter density
ϱ_mbar = Ωm*ϱ_crit
# The real size of the padded (last) dimension of global slab grid
slab_size_padding = 2*(φ_gridsize//2 + 1)
# Half of φ_gridsize (use of the ℝ-syntax requires doubles)
φ_gridsize_half = φ_gridsize//2
# The cube of φ_gridsize. This is defined here because it is a very
# large integer (long long int) (use of the ℝ-syntax requires doubles)
φ_gridsize3 = cast(φ_gridsize, 'long long int')**3
# Name of file storing the Ewald grid
ewald_file = '.ewald_gridsize=' + str(ewald_gridsize) + '.hdf5'
# All constant factors across the PM scheme is gathered in the
# pm_fac_const variable. Its contributions are:
# Normalization due to forwards and backwards Fourier transforms:
#     1/φ_gridsize**3
# Factor in the Greens function:
#     -4*π*G_Newton/((2*π/((boxsize/φ_gridsize)*φ_gridsize))**2)   
# The acceleration is the negative gradient of the potential:
#     -1
# For converting acceleration to momentum (for particles)
#     particles.mass*Δt
# Everything except the mass and the time are constant, and is condensed
# into the pm_fac_const variable.
pm_fac_const = G_Newton*boxsize**2/(π*φ_gridsize**3) 
# The exponential cutoff for the long-range force looks like
# exp(-k2*rs2). In the code, the wave vector is in grid units in stead
# of radians. The conversion is 2*π/φ_gridsize. The total factor on k2
# in the exponential is then
longrange_exponent_fac = -(2*π/φ_gridsize*p3m_scale)**2
# The short-range/long-range force scale
p3m_scale_phys = p3m_scale*boxsize/φ_gridsize
# Particles within this distance to the surface of the domain should
# interact with particles in the neighboring domain via the shortrange
# force, when the P3M algorithm is used.
p3m_cutoff_phys = p3m_scale_phys*p3m_cutoff



#####################
# Update units_dict #
#####################
# In the code, the units_dict is often used as a name space for
# evaluating a Python expression given as a str which may include units.
# These expressions are some times supplied by the user, and so may
# contain all sorts of "units". Update the units_dict with quantities so
# that quite general evaluations can take place.
# Add physical quantities.
units_dict.setdefault('G_Newton'              , G_Newton              )
units_dict.setdefault('H0'                    , H0                    )
units_dict.setdefault('p3m_cutoff_phys'       , p3m_cutoff_phys       )
units_dict.setdefault('p3m_scale_phys'        , p3m_scale_phys        )
units_dict.setdefault('pm_fac_const'          , pm_fac_const          )
units_dict.setdefault('R_tophat'              , R_tophat              )
units_dict.setdefault('a_begin'               , a_begin               )
units_dict.setdefault('boxsize'               , boxsize               )
units_dict.setdefault('longrange_exponent_fac', longrange_exponent_fac)
units_dict.setdefault('t_begin'               , t_begin               )
units_dict.setdefault(        'Ωm'            , Ωm                    )
units_dict.setdefault(unicode('Ωm')           , Ωm                    )
units_dict.setdefault(        'Ωr'            , Ωr                    )
units_dict.setdefault(unicode('Ωr')           , Ωr                    )
units_dict.setdefault(        'ΩΛ'            , ΩΛ                    )
units_dict.setdefault(unicode('ΩΛ')           , ΩΛ                    )
units_dict.setdefault(        'ϱ_vacuum'      , ϱ_vacuum              )
units_dict.setdefault(unicode('ϱ_vacuum')     , ϱ_vacuum              )
units_dict.setdefault(        'ϱ_crit'        , ϱ_crit                )
units_dict.setdefault(unicode('ϱ_crit')       , ϱ_crit                )
units_dict.setdefault(        'ϱ_mbar'        , ϱ_mbar                )
units_dict.setdefault(unicode('ϱ_mbar')       , ϱ_mbar                )
# Add dimensionless sizes
units_dict.setdefault('p3m_scale'                 , p3m_scale                 )
units_dict.setdefault('p3m_cutoff'                , p3m_cutoff                )
units_dict.setdefault('ewald_gridsize'            , ewald_gridsize            )
units_dict.setdefault('resolution'                , resolution                )
units_dict.setdefault('slab_size_padding'         , slab_size_padding         )
units_dict.setdefault('softeningfactors'          , softeningfactors          )
units_dict.setdefault('terminal_render_resolution', terminal_render_resolution)
units_dict.setdefault(        'φ_gridsize'        , φ_gridsize                )
units_dict.setdefault(unicode('φ_gridsize')       , φ_gridsize                )
units_dict.setdefault(        'φ_gridsize_half'   , φ_gridsize_half           )
units_dict.setdefault(unicode('φ_gridsize_half')  , φ_gridsize_half           )
units_dict.setdefault(        'φ_gridsize3'       , φ_gridsize3               )
units_dict.setdefault(unicode('φ_gridsize3')      , φ_gridsize3               )
# Add numbers
units_dict.setdefault(        'machine_ϵ' , machine_ϵ)
units_dict.setdefault(unicode('machine_ϵ'), machine_ϵ)
units_dict.setdefault(        'π'         , π        )
units_dict.setdefault(unicode('π')        , π        )
units_dict.setdefault(        'ထ'         , ထ        )
units_dict.setdefault(unicode('ထ')        , ထ        )
# Add everything from NumPy
for key, val in vars(np).items():
    units_dict.setdefault(key, val)



############################
# Custom defined functions #
############################
# Absolute function for numbers
if not cython.compiled:
    # Use NumPy's abs function in pure Python
    abs = np.abs
else:
    @cython.header(x=signed_number,
                   returns=signed_number,
                   )
    def abs(x):
        if x < 0:
            return -x
        return x

# Max function for 1D memory views of numbers
if not cython.compiled:
    # Use NumPy's max function in pure Python
    max = np.max
else:
    """
    @cython.header(returns=number)
    def max(number[::1] a):
        cdef:
            Py_ssize_t i
            number m
        m = a[0]
        for i in range(1, a.shape[0]):
            if a[i] > m:
                m = a[i]
        return m
    """

# Max function for 2 numbers
@cython.header(a=number,
               b=number,
               returns=number,
               )
def pairmax(a, b):
    if a > b:
        return a
    return b

# Min function for 1D memory views of numbers
if not cython.compiled:
    # Use NumPy's min function in pure Python
    min = np.min
else:
    """
    @cython.header(returns=number)
    def min(number[::1] a):
        cdef:
            Py_ssize_t i
            number m
        m = a[0]
        for i in range(1, a.shape[0]):
            if a[i] < m:
                m = a[i]
        return m
    """

# Modulo function for numbers
@cython.header(x=signed_number,
               length=signed_number2,
               returns=signed_number,
               )
def mod(x, length):
    """Warning: mod(integer, floating) not possible. Note that
    no error will occur if called with illegal types!
    Note also that -length < x < 2*length must be true for this
    function to compute the modulo properly. A more general
    prescription would be x = (x % length) + (x < 0)*length.
    """
    if not (signed_number in integer and signed_number2 in floating):
        if x < 0:
            x += length
            if x == length:
                x = 0
        elif x >= length:
            x -= length
            if x < 0:
                x = 0
        return x

# Summation function for 1D memory views of numbers
if not cython.compiled:
    # Use NumPy's sum function in pure Python
    sum = np.sum
else:
    """
    @cython.header(returns=number)
    def sum(number[::1] a):
        cdef:
            number Σ
            Py_ssize_t N
            Py_ssize_t i
        N = a.shape[0]
        if N == 0:
            return 0
        Σ = a[0]
        for i in range(1, N):
            Σ += a[i]
        return Σ
    """

# Product function for 1D memory views of numbers
if not cython.compiled:
    # Use NumPy's prod function in pure Python
    prod = np.prod
else:
    """
    @cython.header(returns=number)
    def prod(number[::1] a):
        cdef:
            number Π
            Py_ssize_t N
            Py_ssize_t i
        N = a.shape[0]
        if N == 0:
            return 1
        Π = a[0]
        for i in range(1, N):
            Π *= a[i]
        return Π
    """

# Unnormalized sinc function (faster than gsl_sf_sinc)
@cython.header(x='double',
               y='double',
               returns='double',
               )
def sinc(x):
    y = sin(x)
    if y == x:
        return 1
    else:
        return y/x

# Function that compares two numbers (identical to math.isclose)
@cython.header(# Arguments
               a=number,
               b=number,
               rel_tol='double',
               abs_tol='double',
               # Locals
               size_a='double',
               size_b='double',
               tol='double',
               returns='bint',
               )
def isclose(a, b, rel_tol=1e-9, abs_tol=0):
    size_a, size_b = abs(a), abs(b)
    if size_a > size_b:
        tol = rel_tol*size_a
    else:
        tol = rel_tol*size_b
    if tol < abs_tol:
        tol = abs_tol
    return abs(a - b) <= tol

# Function that checks if a (floating point) number
# is actually an integer.
@cython.header(x='double',
               rel_tol='double',
               abs_tol='double',
               returns='bint',
               )
def isint(x, abs_tol=1e-6):
    return isclose(x, round(x), 0, abs_tol)

# Function which format numbers to have a
# specific number of significant figures.
@cython.pheader(# Arguments
                numbers='object',  # Single number or container of numbers
                nfigs='int',
                fmt='str',
                # Locals
                coefficient='str',
                exponent='str',
                n_missing_zeros='int',
                number='object',  # Single number of any type
                number_str='str',
                return_list='list',
                returns='object',  # String or list of strings
                )
def significant_figures(numbers, nfigs, fmt='', incl_zeros=True, scientific=False):
    """This function formats a floating point number to have nfigs
    significant figures.
    Set fmt to 'TeX' to format to TeX math code
    (e.g. '1.234\times 10^{-5}') or 'Unicode' to format to superscript
    Unicode (e.g. 1.234×10⁻⁵).
    Set incl_zeros to False to avoid zero-padding.
    Set scientific to True to force scientific notation.
    """
    fmt = fmt.lower()
    if fmt not in ('', 'tex', 'unicode'):
        abort('Formatting mode "{}" not understood'.format(fmt))
    return_list = []
    for number in any2iter(numbers):
        # Format the number using nfigs
        number_str = ('{{:.{}{}}}'
                      .format((nfigs - 1) if scientific else nfigs, 'e' if scientific else 'g')
                      .format(number)
                      )
        # Handle the exponent
        if 'e' in number_str:
            e_index = number_str.index('e')
            coefficient = number_str[:e_index]
            exponent = number_str[e_index:]
            # Remove superfluous 0 in exponent
            if exponent.startswith('e+0') or exponent.startswith('e-0'):
                exponent = exponent[:2] + exponent[3:]
            # Remove plus sign in exponent
            if exponent.startswith('e+'):
                exponent = 'e' + exponent[2:]
            # Handle formatting
            if fmt == 'tex':
                exponent = exponent.replace('e', r'\times 10^{') + '}'
            elif fmt == 'unicode':
                exponent = unicode_superscript(exponent)
        else:
            coefficient = number_str
            exponent = ''
        # Remove coefficient if it is just 1 (as in 1×10¹⁰ --> 10¹⁰)
        if coefficient == '1' and exponent and fmt in ('tex', 'unicode') and not scientific:
            coefficient = ''
            exponent = exponent.replace(r'\times ', '').replace(unicode('×'), '')
        # Pad with zeros in case of too few significant digits
        if incl_zeros:
            digits = coefficient.replace('.', '').replace('-', '')
            for i, d in enumerate(digits):
                if d != '0':
                    digits = digits[i:]
                    break
            n_missing_zeros = nfigs - len(digits)
            if n_missing_zeros > 0:
                if not '.' in coefficient:
                    coefficient += '.'
                coefficient += '0'*n_missing_zeros
        # Combine
        number_str = coefficient + exponent
        # The mathtext matplotlib module has a typographical bug;
        # it inserts a space after a decimal point.
        # Prevent this by not letting the decimal point be part
        # of the mathematical expression.
        if fmt == 'tex':
            number_str = number_str.replace('.', '$.$')
        return_list.append(number_str)
    # If only one string should be returned,
    # return as a string directly.
    if len(return_list) == 1:
        return return_list[0]
    else:
        return return_list



####################################################
# Sanity checks and corrections to user parameters #
####################################################
if master and fftw_rigor not in ('estimate', 'measure', 'patient', 'exhaustive'):
    abort('Does not recognize FFTW rigor "{}"'.format(user_params['fftw_rigor']))
# Warn about unused but specified parameters.
if user_params.unused:
    if len(user_params.unused) == 1:
        msg = 'The following unknown parameter was specified:\n'
    else:
        msg = 'The following unknown parameters were specified:\n'
    masterwarn(msg + '\n'.join(user_params.unused))
# Warn about non-flat geometry due to Ωr + Ωm + ΩΛ ≠ 1
if not isclose(Ωr + Ωm + ΩΛ, 1, rel_tol=0, abs_tol=1e-4):
    masterwarn('The density parameters '
               'Ωr = {:.4g}, '
               'Ωm = {:.4g}, '
               'ΩΛ = {:.4g} '
               'add up to {:.4g} ≠ 1, implying a non-flat geometry. '
               'Only flat geometries are properly handled by the code.'
               .format(Ωr, Ωm, ΩΛ, Ωr + Ωm + ΩΛ)
               )
# Output times very close to t_begin or a_begin
# are probably meant to be exactly at t_begin or a_begin
for time_param in ('t', 'a'):
    output_times[time_param] = {key: tuple([a_begin if isclose(float(nr), a_begin) else nr
                                            for nr in val])
                                for key, val in output_times[time_param].items()}
snapshot_times        = {time_param: output_times[time_param]['snapshot'] 
                         for time_param in ('a', 't')}
powerspec_times       = {time_param: output_times[time_param]['powerspec'] 
                         for time_param in ('a', 't')}
render_times          = {time_param: output_times[time_param]['render'] 
                         for time_param in ('a', 't')}
terminal_render_times = {time_param: output_times[time_param]['terminal render'] 
                         for time_param in ('a', 't')}
# Warn about cosmological autosaving time
if autosave > 1*units.yr:
    masterwarn('Autosaving will take place every {} {}. Have you forgotten to '
               'specify the unit of the "autosave" parameter?'.format(autosave, unit_time))



###########################################################
# Functionality for "from commons import *" when compiled #
###########################################################
# Function which floods the current module namespace with variables from
# the uncompiled version of this module. This is effectively equivalent
# to "from commons import *", except that the uncompiled version is
# guaranteed to be used.
def commons_flood():
    if cython.compiled:
        commons_module = imp.load_source('commons_pure_python', '{}/commons.py'.format(paths['concept_dir']))
        inspect.getmodule(inspect.stack()[0][0]).__dict__.update(commons_module.__dict__)
    else:
        # Running in pure Python mode.
        # It is assumed that "from commons import *" has already
        # been run, leaving nothing to do.
        ...
