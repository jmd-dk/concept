# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
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
import collections, contextlib, ctypes, cython, imp, inspect, itertools, matplotlib, numpy as np
import os, re, shutil, sys, types, unicodedata
# For math
# (note that numpy.array is purposely not imported directly into the
# global namespace, as this does not play well with Cython).
from numpy import (arange, asarray, concatenate, cumsum, delete, empty, linspace, loadtxt, ones,
                   unravel_index, zeros)
from numpy.random import random
# Use a matplotlib backend that does not require a running X-server
matplotlib.use('Agg')
# Customize matplotlib
matplotlib.rcParams.update({# Use a nice font that ships with matplotlib
                            'text.usetex'       : False,
                            'font.family'       : 'serif',
                            'font.serif'        : 'cmr10',
                            'axes.unicode_minus': False,
                            })
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
try:
    # Matplotlib >= 1.5
    matplotlib.rcParams.update({# Default colors
                                'axes.prop_cycle': matplotlib.cycler('color', color_cycle),
                                })
except:
    # Matplotlib < 1.5
    matplotlib.rcParams.update({# Default colors
                                'axes.color_cycle': color_cycle,
                                })
    
# For plotting
import matplotlib.pyplot as plt
# For I/O
import h5py
# For fancy terminal output
from blessings import Terminal
terminal = Terminal(force_styling=True)
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
# Functions for communication
comm = MPI.COMM_WORLD
Allgather = comm.Allgather
Allgatherv = comm.Allgatherv
Allreduce = comm.Allreduce
Barrier = comm.Barrier
Bcast = comm.Bcast
Gather = comm.Gather
Gatherv = comm.Gatherv
Isend = comm.Isend
Reduce = comm.Reduce
Recv = comm.Recv
Scatter = comm.Scatter
Send = comm.Send
Sendrecv = comm.Sendrecv
allreduce = comm.allreduce
reduce = comm.reduce
sendrecv = comm.sendrecv
# Number of processes started with mpiexec
nprocs = comm.size
# The unique rank of the running process
rank = comm.rank
# Flag identifying the master/root process (that which have rank 0)
master = not rank



#############################
# Print and abort functions #
#############################
# Function for printing messages as well as timed progress messages
def masterprint(*args, indent=0, sep=' ', end='\n', fun=None, **kwargs):
    global progressprint_time, progressprint_length, progressprint_maxintervallength
    if not master:
        return
    terminal_resolution = 80
    if args[0] == 'done':
        # End of progress message
        seconds = time() - progressprint_time
        # Construct the time interval with a sensible amount of
        # significant figures.
        milliseconds = 0
        # More than a millisecond; use whole milliseconds
        if seconds >= 1e-3:
            milliseconds = round(1e+3*seconds)
        interval = timedelta(milliseconds=milliseconds)
        # More than a second; whose whole deciseconds
        if interval.total_seconds() >= 1e+0:
            seconds = 1e-1*round(1e-2*milliseconds)
            interval = timedelta(seconds=seconds)    
        # More than 10 seconds; use whole seconds
        if interval.total_seconds() >= 1e+1:
            seconds = round(1e-3*milliseconds)
            interval = timedelta(seconds=seconds)
        # Print the time interval
        total_seconds = interval.total_seconds()
        if total_seconds == 0:
            interval = '< 1 ms'
        elif total_seconds < 1:
            interval = '{} ms'.format(int(1e+3*total_seconds))
        elif total_seconds < 10:
            interval = '{} s'.format(total_seconds)
        elif total_seconds < 60:
            interval = '{} s'.format(int(total_seconds))
        elif total_seconds < 60*60:
            interval = interval.__str__()[2:]
        else:
            interval = interval.__str__()
        # Stitch text pieces together
        text = ' done after {}'.format(interval)
        if len(args) > 1:
            text += sep + sep.join([str(arg) for arg in args[1:]])
        # Convert to proper Unicode characters
        text = unicode(text)
        # The progressprint_maxintervallength variable store the length
        # of the longest interval-message so far.
        if len(text) > progressprint_maxintervallength:
            progressprint_maxintervallength = len(text)
        # Prepend the text with whitespace so that all future
        # interval-messages lign up to the right.
        text = ' '*(terminal_resolution - progressprint_length
                                        - progressprint_maxintervallength) + text
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
        # Indent text
        text = ' '*indent + ('\n' + ' '*indent).join(text.split('\n'))
        # If the text ends with '...',
        # it is the start of a progress message.
        if text.endswith('...'):
            progressprint_time = time()
            progressprint_maxlength = terminal_resolution - progressprint_maxintervallength - 1
            if len(text) > progressprint_maxlength:
                # The progress message should span multiple lines.
                # Do the line splitting explicitly, so that there is
                # room left over for the time interval.
                indentation = len(text) - len(text.lstrip())
                words = text.split()
                if words[-1] == '...':
                    words[-2] += ' ...'
                    words.pop()
                partial_text = []
                N_words = len(words)
                for i in range(N_words):
                    word = words[i]
                    partial_text.append(word)
                    if indentation + len(' '.join(partial_text)) > progressprint_maxlength:
                        print(' '*indentation + ' '.join(partial_text[:-1]),
                              flush=True, end='\n', **kwargs)
                        partial_text = [word]
                text = ' '*indentation + ' '.join(partial_text)
            progressprint_length = len(text)
            end = ''
        # Apply supplied function to text
        if fun:
            text = fun(text)
        # Print out message
        print(text, flush=True, end=end, **kwargs)
progressprint_maxintervallength = len(' done after ??? ms')

# Function for printing warnings
def masterwarn(*args, skipline=True, prefix='Warning', **kwargs):
    if not master:
        return
    try:
        universals.any_warnings = True
    except:
        ...
    # Add initial newline (if skipline == True) to prefix
    # and append a colon.
    prefix = ('\n' if skipline else '') + prefix + ':'
    # Print out message
    masterprint(prefix, *args, fun=terminal.bold_red, file=sys.stderr, **kwargs)

# Raised exceptions inside cdef functions do not generally propagte
# out to the caller. In places where exceptions are normally raised
# manualy, call this function with a descriptive message instead.
def abort(*args, **kwargs):
    masterwarn(*args, prefix='Aborting', **kwargs)
    sys.stderr.flush()
    sys.stdout.flush()
    sleep(1)
    comm.Abort(1)



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
                       arcsin,  arccos, arctan,
                       sinh, cosh, tanh,
                       arcsinh, arccosh, arctanh,
                       exp, log, log2, log10,
                       sqrt,
                       ceil, round,
                       )
    from math import erf, erfc
    # Dummy ℝ and ℤ dict for constant expressions
    class DummyDict(dict):
        def __getitem__(self, key):
            return key
    ℝ = DummyDict()
    ℤ = DummyDict()
    # The cimport function, which in the case of pure Python should
    # simply execute the statements parsed to it as a string,
    # within the namespace of the call.
    def cimport(import_statement):
        import_statement = import_statement.strip()
        if import_statement.endswith(','):
            import_statement = import_statement[:-1]
        exec(import_statement, inspect.getmodule(inspect.stack()[1][0]).__dict__)
# Function for building "structs" (really simple namespaces).
# In compiled mode, this function body will be copied and
# specialised for each kind of struct created.
# This function returns a struct and a dict over the parsed fields.
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
# GNU Scientific Library
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
                        sqrt, erf, erfc,
                        ceil, round,
                        )
"""



#####################
# Unicode functions #
#####################
# The pyxpp script convert all Unicode source code characters into
# ASCII using the below function.
@cython.header(# Arguments
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
unicode_tags = {'begin': '__BEGIN_UNICODE__',
                'end':   '__END_UNICODE__',
                }

# The function below grants the code access to
# Unicode string literals by undoing the convertion of the
# ascii function above.
@cython.header(s='str', returns='str')
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
# "params='/path/to/params'"
cython.declare(argd='dict',
               globals_dict='dict',
               scp_password='str',
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
# The scp password
scp_password = argd.get('scp_password', '')



##################
# Pure constants #
##################
cython.declare(machine_ϵ='double',
               inf='double',
               π='double',
               )
machine_ϵ = np.finfo(C2np['double']).eps
inf = cast(np.inf, 'double')
π = np.pi



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
user_params = dict(vars(np))
user_params.update({# The paths dict
                    'paths': paths,
                    # Modules
                    'numpy': np,
                    'np'   : np,
                    'os'   : os,
                    're'   : re,
                    'sys'  : sys,
                    # Constants
                    'machine_ϵ' : machine_ϵ,
                    'eps'       : machine_ϵ,
                    'inf'       : inf,
                    unicode('π'): π,
                    })
# Add units to the user_params namespace.
# These do not represent the choice of units; the names should merely
# exist to avoid errors when reading the parameter file.
user_params.update(unit_length_relations)
user_params.update(unit_time_relations)
user_params.update(unit_mass_relations)
# Now do the actual read of the parameters
# in order to get the user defined units.
if os.path.isfile(paths['params']):
    with open(paths['params'], encoding='utf-8') as params_file:
        with contextlib.suppress(Exception):
            exec(params_file.read(), user_params)
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
# Construct a struct containing the values of all units
units, units_dict = build_struct(# Values of basic units,
                                 # determined from the choice of fundamental units.
                                 pc     = ('double', 1/eval(unit_length, unit_length_relations)),
                                 yr     = ('double', 1/eval(unit_time,   unit_time_relations)),
                                 m_sun  = ('double', 1/eval(unit_mass,   unit_mass_relations)),
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

# Function which converts a string of units to the
# corresponding numerical value.
@cython.header(# Arguments
               unit_str='str',
               # Locals
               c='str',
               i='Py_ssize_t',
               key='str',
               mapping='dict',
               pat='str',
               previous='str',
               rep='str',
               superscript_ASCII='str',
               superscript_unicode='str',
               unit='double',
               unit_list='list',
               returns='double',
               )
def evaluate_unit(unit_str):
    """Example unit_str: 'm☉ Mpc Gyr⁻¹'
    """
    # Mapping of replacement in unit_str
    mapping = {' ' : '*',
               'm☉': 'm_sun',
               }
    # Do replacements
    for pat, rep in mapping.items():
        unit_str = unit_str.replace(pat, rep)
    # Construct list of single unicode characters
    unit_list = list(unicode(unit_str))
    # Place '**' before and parentheses around superscripts.
    # This alters the elements of unit_list.
    for i, c in enumerate(unit_list):
        for superscript_ASCII, superscript_unicode in unicode_superscripts.items():
            if c == superscript_unicode:
                # Convert to ASCII
                unit_list[i] = superscript_ASCII
                # Insert '**(' if first character of exponent 
                previous = unit_list[i - 1]
                if len(previous) > 1:
                    previous = previous[len(previous) - 1]
                if previous not in '+-0123456789':
                    unit_list[i] = '**(' + unit_list[i]
                # Insert ')' if last character of exponent
                if (   i + 1 == len(unit_list)
                    or unit_list[i + 1] not in unicode_superscripts.values()):
                    unit_list[i] += ')'
                break
    # Transformation done
    unit_str = ''.join(unit_list)
    # Evaluate the transformed unit string
    unit = eval(unit_str, units_dict)
    return unit

################################################################
# Import all user specified parameters from the parameter file #
################################################################
# Function for converting non-iterables to interables of length 1
# (and cast iterables to lists).
def any2iter(val):
    return list(val) if hasattr(val, '__iter__') and not hasattr(val, '__len__') else np.ravel(val)
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
user_params = DictWithCounter(dict(vars(np)))
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
                    # Constants
                    'machine_ϵ' : machine_ϵ,
                    'eps'       : machine_ϵ,
                    'inf'       : inf,
                    unicode('π'): π,
                    })
# At this point, user_params does not contain actual parameters.
# Mark all items in user_params as used.
user_params.useall()
# "Import" the parameter file by executing it
# in the namespace defined by the user_params namespace.
if os.path.isfile(paths['params']):
    with open(paths['params'], encoding='utf-8') as params_file:
        exec(params_file.read(), user_params)
# Insert asciify'ed versions of 
# for key in dict(user_params).keys():
#     ascii_key = asciify(key)
#     if ascii_key != key:
#         user_params[ascii_key] = user_params[key]
#         user_params.unuse(key)
# Also mark the unit-parameters as used
for u in ('length', 'time', 'mass'):
    user_params.use('unit_{}'.format(u))
# The parameters are now being processed as follows:
# - Some parameters are explicitly casted.
# - Spaces are removed from the 'snapshot_type' parameter, and all
#   characters are converted to lowercase.
# - The 'output_times' are sorted and duplicates (for each type of
#   output) are removed.
# - Paths below or just one level above the concept directory are made
#   relative to this directory in order to reduce screen clutter.
# - The 'special_params' parameter is set to an empty dictionary if it
#   is not defined in params.py.
# - Colors are transformed to (r, g, b) arrays. Below is the function
#   that handles the color input.
def to_rgb(value):
    if isinstance(value, int) or isinstance(value, float):
        value = str(value)
    try:
        rgb = np.array(matplotlib.colors.ColorConverter().to_rgb(value), dtype=C2np['double'])
    except:
        # Could not convert value to color
        return np.array([-1, -1, -1])
    return rgb
cython.declare(# Input/output
               IC_file='str',
               snapshot_type='str',
               output_dirs='dict',
               output_bases='dict',
               output_times='dict',
               powerspec_select='dict',
               powerspec_plot_select='dict',
               render_select='dict',
               # Numerical parameter
               boxsize='double',
               ewald_gridsize='Py_ssize_t',
               φ_gridsize='ptrdiff_t',
               P3M_scale='double',
               P3M_cutoff='double',
               softeningfactors='dict',
               Δt_factor='double',
               R_tophat='double',
               # Cosmological parameters
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
               liverender='str',
               remote_liverender='str',
               terminal_render_colormap='str',
               terminal_render_resolution='unsigned int',
               # Simlation options
               kick_algorithms='dict',
               use_φ='bint',
               use_P3M='bint',
               fftw_rigor='str',
               # Debugging options
               enable_debugging='bint',
               enable_Ewald='bint',
               enable_gravity='bint',
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
for kind in ('snapshot', 'powerspec', 'render'):
    output_dirs[kind] = str(output_dirs.get(kind, paths['output_dir']))
    if not output_dirs[kind]:
        output_dirs[kind] = paths['output_dir']
output_dirs = {key: sensible_path(path) for key, path in output_dirs.items()}
output_bases = dict(user_params.get('output_bases', {}))
for kind in ('snapshot', 'powerspec', 'render'):
    output_bases[kind] = str(output_bases.get(kind, kind))
output_times = dict(user_params.get('output_times', {}))
# Output times not explicitly written as either of type 'a' or 't'
# is understood as being of type 'a'.
for time_param in ('a', 't'):
    output_times[time_param] = dict(output_times.get(time_param, {}))
for key, val in output_times.items():
    if key not in ('a', 't'):
        output_times['a'][key] = tuple(any2iter(output_times['a'].get(key, ()))) + tuple(any2iter(val))
for time_param in ('a', 't'):
    output_times[time_param] = dict(output_times.get(time_param, {}))
    for kind in ('snapshot', 'powerspec', 'render', 'terminal render'):
        output_times[time_param][kind] = output_times[time_param].get(kind, ())
output_times = {time_param: {key: tuple(sorted(set([float(eval(nr, units_dict) if isinstance(nr, str) else nr)
                                                    for nr in any2iter(val)
                                                    if nr or nr == 0])))
                             for key, val in output_times[time_param].items()}
                for time_param in ('a', 't')}
powerspec_select = {}
if 'powerspec_select' in user_params:
    if isinstance(user_params['powerspec_select'], dict):
        powerspec_select = user_params['powerspec_select']
    else:
        powerspec_select = {'all': bool(user_params['powerspec_select'])}
powerspec_select = {key.lower(): bool(val) for key, val in powerspec_select.items()}
powerspec_plot_select = {}
if 'powerspec_plot_select' in user_params:
    if isinstance(user_params['powerspec_plot_select'], dict):
        powerspec_plot_select = user_params['powerspec_plot_select']
    else:
        powerspec_plot_select = {'all': bool(user_params['powerspec_plot_select'])}
powerspec_plot_select = {key.lower(): bool(val) for key, val in powerspec_plot_select.items()}
render_select = {}
if 'render_select' in user_params:
    if isinstance(user_params['render_select'], dict):
        render_select = user_params['render_select']
    else:
        render_select = {'all': bool(user_params['render_select'])}
render_select = {key.lower(): bool(val) for key, val in render_select.items()}
# Numerical parameters
boxsize = float(user_params.get('boxsize', 1))
ewald_gridsize = int(user_params.get('ewald_gridsize', 64))
φ_gridsize = int(user_params.get('φ_gridsize', 64))
P3M_scale = float(user_params.get('P3M_scale', 1.25))
P3M_cutoff = float(user_params.get('P3M_cutoff', 4.8))
softeningfactors = dict(user_params.get('softeningfactors', {}))
for kind in ('dark matter particles', ):
    softeningfactors[kind] = float(softeningfactors.get(kind, 0.03))
Δt_factor = float(user_params.get('Δt_factor', 0.01))
R_tophat = float(user_params.get('R_tophat', 8*units.Mpc))
# Cosmological parameters
H0 = float(user_params.get('H0', 70*units.km/(units.s*units.Mpc)))
Ωr = float(user_params.get('Ωr', 0))
Ωm = float(user_params.get('Ωm', 0.3))
ΩΛ = float(user_params.get('ΩΛ', 0.7))
a_begin = float(user_params.get('a_begin', 1))
t_begin = float(user_params.get('t_begin', 0))
# Graphics
if isinstance(user_params.get('render_colors', {}), dict):
    render_colors = user_params.get('render_colors', {})
elif to_rgb(user_params['render_colors'])[0] != -1:
    render_colors = {'all': to_rgb(user_params['render_colors'])}
else:
    render_colors = dict(user_params['render_colors'])
render_colors = {key.lower(): to_rgb(val) for key, val in render_colors.items()}
bgcolor = to_rgb(user_params.get('bgcolor', 'black'))
resolution = int(user_params.get('resolution', 1080))
liverender = sensible_path(str(user_params.get('liverender', '')))
if liverender and not liverender.endswith('.png'):
    liverender += '.png'
remote_liverender = str(user_params.get('remote_liverender', ''))
if remote_liverender and not remote_liverender.endswith('.png'):
    remote_liverender += '.png'
terminal_render_colormap = str(user_params.get('terminal_render_colormap', 'gnuplot2'))
terminal_render_resolution = int(user_params.get('terminal_render_resolution', 80))
# Simulation options
kick_algorithms = dict(user_params.get('kick_algorithms', {}))
if (   'φ_gridsize' in user_params
    or (set(('PM', 'P3M')) & set(kick_algorithms.values()))
    or any([output_times[time_param]['powerspec'] for time_param in ('a', 't')])):
    use_φ = bool(user_params.get('use_φ', True))
else:
    use_φ = bool(user_params.get('use_φ', False))
if 'P3M' in kick_algorithms.values():
    use_P3M = bool(user_params.get('use_P3M', True))
else:
    use_P3M = bool(user_params.get('use_P3M', False))
fftw_rigor = user_params.get('fftw_rigor', 'estimate').lower()
# Debugging options
enable_debugging = bool(user_params.get('enable_debugging', False))
enable_Ewald = bool(user_params.get('enable_Ewald',
                                    True if 'PP' in kick_algorithms.values() else False))
enable_gravity = bool(user_params.get('enable_gravity', True))
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
universals, _ = build_struct(# Flag specifying whether or not any warnings have been given
                             any_warnings=('bint', False),
                             # Scale factor and cosmic time
                             a=('double', a_begin),
                             t=('double', t_begin),
                             )



############################################
# Derived and internally defined constants #
############################################
cython.declare(G_Newton='double',
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
               scp_host='str',
               snapshot_dir='str',
               snapshot_base='str',
               snapshot_times='dict',
               terminal_render_times='dict',
               ϱbar='double',
               ϱmbar='double',
               PM_fac_const='double',
               longrange_exponent_fac='double',
               P3M_cutoff_phys='double',
               P3M_scale_phys='double',
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
# Newtons constant
G_Newton = 6.6738e-11*units.m**3/units.kg/units.s**2
# The average, comoing density (the critical
# comoving density since we only study flat universes)
ϱbar = 3*H0**2/(8*π*G_Newton)
# The average, comoving matter density
ϱmbar = Ωm*ϱbar
# The real size of the padded (last) dimension of global slab grid
slab_size_padding = 2*(φ_gridsize//2 + 1)
# Half of φ_gridsize (use of the ℝ-syntax requires doubles)
φ_gridsize_half = φ_gridsize//2
# The cube of φ_gridsize. This is defined here because it is a very
# large integer (long long int) (use of the ℝ-syntax requires doubles)
φ_gridsize3 = cast(φ_gridsize, 'long long int')**3
# Name of file storing the Ewald grid
ewald_file = '.ewald_gridsize=' + str(ewald_gridsize) + '.hdf5'
# All constant factors across the PM scheme is gathered in the PM_fac
# variable. Its contributions are:
# For CIC interpolating particle/fluid element masses to the grid
# points, when really it should be mass densities:
#     1/(boxsize/φ_gridsize)**3
# Normalization due to forwards and backwards Fourier transforms:
#     1/φ_gridsize**3
# Factor in the Greens function:
#     -4*π*G_Newton/((2*π/((boxsize/φ_gridsize)*φ_gridsize))**2)   
# The acceleration is the negative gradient of the potential:
#     -1
# For converting acceleration to momentum
#     particles.mass*Δt
# Everything except the mass and the time are constant, and is condensed
# into the PM_fac_const variable.
PM_fac_const = G_Newton/(π*boxsize)
# The exponential cutoff for the long-range force looks like
# exp(-k2*rs2). In the code, the wave vector is in grid units in stead
# of radians. The conversion is 2*π/φ_gridsize. The total factor on k2
# in the exponential is then
longrange_exponent_fac = -(2*π/φ_gridsize*P3M_scale)**2
# The short-range/long-range force scale
P3M_scale_phys = P3M_scale*boxsize/φ_gridsize
# Particles within this distance to the surface of the domain should
# interact with particles in the neighboring domain via the shortrange
# force, when the P3M algorithm is used.
P3M_cutoff_phys = P3M_scale_phys*P3M_cutoff
# The host name in the 'remote_liverender' parameter
scp_host = re.search('@(.*):', remote_liverender).group(1) if remote_liverender else ''



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
                number='double',
                nfigs='int',
                fmt='str',
                # Locals
                coefficient='str',
                exponent='str',
                n_missing_zeros='int',
                number_str='str',
                returns='str',
                )
def significant_figures(number, nfigs, fmt='', incl_zeros=True, scientific=False):
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
    # Format the number using nfigs
    number_str = ('{:.' + str(nfigs) + ('e' if scientific else 'g') + '}').format(number)
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
    return number_str



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
