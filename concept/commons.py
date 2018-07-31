# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2018 Jeppe Mosgaard Dakin.
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



# This module contains imports, Cython declarations and values
# of parameters common to all other modules. Each module should have
# 'from commons import *' as its first statement.


######################
# __future__ imports #
######################
from __future__ import division  # Needed for Python3 division in Cython



#######################
# The CO𝘕CEPT version #
#######################
__version__ = 'master'



############################################
# Imports common to pure Python and Cython #
############################################
# Miscellaneous modules
import ast, collections, contextlib, ctypes, cython, functools, hashlib, inspect, itertools
import keyword, os, re, shutil, sys, textwrap, types, unicodedata, warnings
# For math
# (note that numpy.array is purposely not imported directly into the
# global namespace, as this does not play well with Cython).
import math
import numpy as np
from numpy import arange, asarray, empty, linspace, logspace, ones, zeros
np.warnings.filterwarnings(  # Suppress warning from NumPy 14.1 caused by H5Py
    'ignore',
    'Conversion of the second argument of issubdtype from',
    FutureWarning,
)
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import scipy.signal
# For plotting
import matplotlib
matplotlib.use('agg')  # Use a matplotlib backend that does not require a running X-server
import matplotlib.mathtext
import matplotlib.pyplot as plt
# For I/O
from glob import glob
import h5py
# CLASS
from classy import Class
# For fancy terminal output
import blessings
# For timing
from time import sleep, time
import datetime
# The pyxpp preprocessor module
import pyxpp



###########
# C types #
###########
# The pxd function is a no-op in compiled mode
# (the pyxpp script simply looks for the word "pxd").
# As it is used below, here we define it as a no-op function.
def pxd(s):
    pass
# Import the signed integer type ptrdiff_t
pxd('from libc.stddef cimport ptrdiff_t')
# C type names to NumPy dtype names
cython.declare(C2np=dict)
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



#############
# MPI setup #
#############
from mpi4py import MPI
cython.declare(master='bint',
               master_node='int',
               master_rank='int',
               nnodes='int',
               node='int',
               node_master='bint',
               node_master_rank='int',
               node_master_ranks='int[::1]',
               node_ranks='int[::1]',
               nodes='int[::1]',
               nprocs='int',
               nprocs_node='int',
               nprocs_nodes='int[::1]',
               rank='int',
               )
# The MPI communicator
comm = MPI.COMM_WORLD
# Number of processes started with mpiexec
nprocs = comm.size
# The unique rank of the running process
rank = comm.rank
# The rank of the master/root process
# and a flag identifying this process.
master_rank = 0
master = (rank == master_rank)
# MPI functions for communication.
# For never versions of NumPy, we have to pass the dtype of the arrays
# explicitly when using uppercase communication methods.
def buf_and_dtype(buf):
    try:
        arr = asarray(buf)
        if arr.shape:
            return (buf, arr.dtype.char)
    except:
        pass
    return buf
Allgather = lambda sendbuf, recvbuf: comm.Allgather(
    buf_and_dtype(sendbuf), recvbuf)
Allgatherv = lambda sendbuf, recvbuf: comm.Allgatherv(
    buf_and_dtype(sendbuf), recvbuf)
Barrier = comm.Barrier
Bcast = lambda buf, root=master_rank: comm.Bcast(buf_and_dtype(buf), root)
Gather = lambda sendbuf, recvbuf, root=master_rank: comm.Gather(
    buf_and_dtype(sendbuf), recvbuf, root)
Gatherv = lambda sendbuf, recvbuf, root=master_rank: comm.Gatherv(
    buf_and_dtype(sendbuf), recvbuf, root)
Isend = lambda buf, dest, tag=0: comm.Isend(buf_and_dtype(buf), dest, tag)
Reduce = lambda sendbuf, recvbuf, op=MPI.SUM, root=master_rank: comm.Reduce(
    buf_and_dtype(sendbuf), recvbuf, op, root)
Recv = lambda buf, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG: comm.Recv(
    buf_and_dtype(buf), source, tag)
Send = lambda buf, dest, tag=0: comm.Send(buf_and_dtype(buf), dest, tag)
Sendrecv = (lambda sendbuf, dest, sendtag=0, recvbuf=None, source=MPI.ANY_SOURCE,
    recvtag=MPI.ANY_TAG, status=None: comm.Sendrecv(buf_and_dtype(sendbuf), dest, sendtag,
        recvbuf, source, recvtag, status)
)
allgather  = comm.allgather
allreduce  = comm.allreduce
bcast      = lambda obj, root=master_rank: comm.bcast (obj, root)
gather     = lambda obj, root=master_rank: comm.gather(obj, root)
iprobe     = comm.iprobe
isend      = comm.isend
recv       = comm.recv
reduce     = lambda obj, op=MPI.SUM, root=master_rank: comm.reduce(obj, op, root)
send       = comm.send
sendrecv   = comm.sendrecv
# Find out on which node the processes are running.
# The nodes will be numbered 0 through nnodes - 1.
master_node = 0
node_name = MPI.Get_processor_name()
node_names = allgather(node_name)
if master:
    nodes = empty(nprocs, dtype=C2np['int'])
    node_names2numbers = {node_name: master_node}
    node_i = -1
    cython.declare(other_rank='int')
    for other_rank, other_node_name in enumerate(node_names):
        if other_node_name not in node_names2numbers:
            node_i += 1
            if node_i == master_node:
                node_i += 1
            node_names2numbers[other_node_name] = node_i
        nodes[other_rank] = node_names2numbers[other_node_name]
    node_numbers2names = {val: key for key, val in node_names2numbers.items()}
nodes = bcast(asarray(nodes) if master else None)
node = nodes[rank]
# The number of nodes
nnodes = len(set(nodes))
# The number of processes in all nodes and in this node
nprocs_nodes = asarray([np.sum(asarray(nodes) == n) for n in range(nnodes)], dtype=C2np['int'])
nprocs_node = nprocs_nodes[node]
# Ranks of processes within the same node
node_ranks = asarray(np.where(asarray(nodes) == node)[0], dtype=C2np['int'])
# Determine if this process is a "node master" (the process with lowest
# rank within its node) or not.
# The rank of the node master process
# and a flag identifying this process.
node_master_rank = node_ranks[0]
node_master = (rank == node_master_rank)
# Ranks of all node masters
node_master_ranks = asarray(np.where(allgather(node_master))[0], dtype=C2np['int'])
# Custom version of the barrier function, where all slaves wait on
# the master. In between the pinging of the master by the slaves,
# they sleep for the designated time, freeing up the CPUs to do other
# work (such as helping with OpenMP work loads).
def sleeping_barrier(sleep_time, mode):
    """When mode == 'single node', all processes (but the global master)
    wait for the global master process.
    When mode == 'MPI', all processes (but the node masters) wait for
    their respective node master.
    """
    mode = mode.lower()
    if mode not in ('single node', 'mpi'):
        abort(
            f'sleeping_barrier called with mode = "{mode}", '
            f'but only "single node" and "MPI" are allowed'
        )
    if (mode == 'single node' and master) or (mode == 'mpi' and node_master):
        # Signal slaves to continue
        requests = []
        if mode == 'single node':
            slave_ranks = range(nprocs)
        elif mode == 'mpi':
            slave_ranks = node_ranks
        for slave_rank in slave_ranks:
            if slave_rank != rank:
                requests.append(isend(True, dest=slave_rank))
        # Remember to wait for the requests to finish
        for request in requests:
            request.wait()
    else:
        # Wait for global or node master
        if mode == 'single node':
            source = master_rank
        elif mode == 'mpi':
            source = node_master_rank
        while not iprobe(source=source):
            sleep(sleep_time)
        # Remember to receive the message
        recv(source=source)
# Function that can call another function that uses OpenMP.
# The (global or node) master process is the only one that actually does
# the call, while slave processes periodically asks whether their
# master is done so that they may continue. This period is controlled by
# sleep_time, given in seconds. While sleeping, the slave processes
# can be utilized to cary out OpenMP work by their master.
def call_openmp_lib(func, *args, sleep_time=0.1, mode='single node', **kwargs):
    """When mode == 'single node', only the master process calls
    the function, while all other processes sleeps. Those processes also
    on the master node will be available for OpenMP work.
    When mode == 'MPI', all node masters call the function, while all
    other processes sleep. All node slaves will be available for OpenMP
    work from their node master.
    """
    mode = mode.lower()
    if (mode == 'single node' and master) or (mode == 'mpi' and node_master):
        func(*args, **kwargs)
    sleeping_barrier(sleep_time, mode)



#################################
# Miscellaneous initialisations #
#################################
# The time before the main computation begins
if master:
    start_time = time()
# Initialise a Blessings Terminal object,
# capable of producing fancy terminal formatting.
terminal = blessings.Terminal(force_styling=True)
# Monkey patch internal NumPy functions handling encoding during I/O,
# replacing the Latin1 encoding by UTF-8. On some systems, this is
# needed for reading and writing unicode characters in headers of text
# files using np.loadtxt and np.savetxt.
asbytes = lambda s: s if isinstance(s, bytes) else str(s).encode('utf-8')
asstr = lambda s: s.decode('utf-8') if isinstance(s, bytes) else str(s)
np.compat.py3k .asbytes   = asbytes
np.compat.py3k .asstr     = asstr
np.compat.py3k .asunicode = asstr
np.lib   .npyio.asbytes   = asbytes
np.lib   .npyio.asstr     = asstr
np.lib   .npyio.asunicode = asstr
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
    interval = datetime.timedelta(milliseconds=milliseconds)
    # More than a second; use whole deciseconds
    if interval.total_seconds() >= 1e+0:
        seconds = 1e-1*round(1e-2*milliseconds)
        interval = datetime.timedelta(seconds=seconds)
    # More than 10 seconds; use whole seconds
    if interval.total_seconds() >= 1e+1:
        seconds = round(1e-3*milliseconds)
        interval = datetime.timedelta(seconds=seconds)
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
def fancyprint(
    *args,
    indent=0,
    sep=' ',
    end='\n',
    fun=None,
    wrap=True,
    ensure_newline_after_ellipsis=True,
    is_warning=False,
    **kwargs,
    ):
    # If called without any arguments, print the empty string
    if not args:
        args = ('', )
    args = tuple([str(arg) for arg in args])
    # Handle indentation due to non-fisnished progress print.
    # If indent == -1, no indentation should be used, not even that
    # due to the non-finished progressprint.
    is_done_message = (args[0] == 'done')
    leftover_indentation = progressprint['indentation']
    if leftover_indentation:
        if indent != -1:
            indent += progressprint['indentation']
        if ensure_newline_after_ellipsis and progressprint['previous'].endswith('...'):
            args = ('\n', ) + args
    if indent == -1:
        indent = 0
    if len(args) > 1 and args[0] == '\n':
        args = ('\n' + args[1], ) + args[2:]
    # If the first character supplied is '\n', this can get lost
    newline_prefix = (isinstance(args[0], str) and args[0].startswith('\n'))
    if is_done_message:
        # End of progress message
        N_args_usual = 2 if leftover_indentation else 1
        indent -= 4
        progressprint['indentation'] -= 4
        text = ' done after {}'.format(time_since(progressprint['time'].pop()))
        if len(args) > N_args_usual:
            text += sep + sep.join([str(arg) for arg in args[N_args_usual:]])
        # Convert to proper Unicode characters
        text = unicode(text)
        # The progressprint['maxintervallength'] variable stores the
        # length of the longest interval-message so far.
        if len(text) > progressprint['maxintervallength']:
            progressprint['maxintervallength'] = len(text)
        # Prepend the text with whitespace so that all future
        # interval-messages lign up to the right.
        text = ' '*(+ terminal_width
                    - progressprint['length']
                    - progressprint['maxintervallength']
                    ) + text
        # Apply supplied function to text
        if fun:
            text = fun(text)
        # Print out timing
        print(text, flush=True, end=end, **kwargs)
        progressprint['length'] = 0
        progressprint['previous'] = text
    else:
        # Stitch text pieces together
        text = sep.join([str(arg) for arg in args])
        # Convert to proper Unicode characters
        text = unicode(text)
        # Convert any paths in text (recognized by surrounding quotes)
        # to sensible paths.
        text = re.sub(r'"(.+?)"', lambda m: '"{}"'.format(sensible_path(m.group(1))), text)
        # Add indentation, and also wrap long message if wrap is True
        indentation = ' '*indent
        is_progress_message = text.endswith('...')
        if wrap:
            # Allow for paths to be wrapped before slashes. We do this
            # by using the break_on_hyphens feature of textwrap.wrap.
            # We thus place a hyphen before slashes in paths. To not
            # get confused with actual hyphens (on which we do not wish
            # to wrap), we replace these with a replacement
            # character '�'. Also, since the break_on_hyphens feature
            # only breaks if the character after the hyphen is a
            # lingual character (and thus not a slash), we replace these
            # slashes with a lingual character. For no particular
            # reason, the 'À' is chosen for this job.
            replacement_characters = (unicode('�'), unicode('À'))
            text = text.replace('-', replacement_characters[0])
            subs = collections.deque(
                [sub[0] for sub in re.findall(
                    r'[^"]/',
                    ''.join([path[1:] for path in re.findall('"(.+?)"', text)]),
                )]
            )
            text = re.sub(
                r'"(.+?)"',
                lambda m: '"{}"'.format(
                    re.sub(r'./', '-' + replacement_characters[1], m.group(1))
                ),
                text,
            )
            # Wrap text into lines which fit the terminal resolution.
            # Also indent all lines. Do this in a way that preserves
            # any newline characters already present in the text.
            lines = list(itertools.chain(*[textwrap.wrap(
                hard_line,
                terminal_width,
                initial_indent=indentation,
                subsequent_indent=indentation,
                replace_whitespace=False,
                break_long_words=False,
                break_on_hyphens=True,
                )
                for hard_line in text.split('\n')
            ]))
            # Replace the inserted hyphens and replacement characters
            # with their original characters.
            text = '\n'.join(lines)
            text = re.sub(r'-', lambda m: subs.popleft(), text)
            text = text.replace(replacement_characters[1], '/')
            text = text.replace(replacement_characters[0], '-')
            lines = text.split('\n')
            # If the text ends with '...', it is the start of a
            # progress message. In that case, the last line should
            # have some left over space to the right
            # for the upcomming "done in ???".
            if is_progress_message:
                maxlength = terminal_width - progressprint['maxintervallength'] - 1
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
            progressprint['time'].append(time())
            progressprint['indentation'] += 4
            progressprint['previous'] = text
            end = ''
        # Apply supplied function to text
        if fun:
            text = fun(text)
        # If a newline prefix got lost, reinsert it
        if newline_prefix and not text.startswith('\n'):
            text = '\n{}'.format(text)
        # Add the ending to the text. We can not just use the end
        # keyword of the print function, as the ending is needed
        # to update progressprint['length'] below.
        text = text + end
        # If we are printing in between the start and finish of a
        # progress print, the amount of spacing to add to the final
        # timing message needs to be altered.
        progressprint['length'] = len(text.split('\n')[-1])
        # Compare text to what is given in suppress_output and
        # suppress the output if any match is found.
        if is_warning and suppress_output['err']:
            text_linearized = re.sub(r'\s+', ' ', text)
            for pattern in suppress_output['err']:
                if re.search(pattern, text_linearized):
                    return
        elif not is_warning and suppress_output['out']:
            text_linearized = re.sub(r'\s+', ' ', text)
            for pattern in suppress_output['out']:
                if re.search(pattern, text_linearized):
                    return
        # Print out message
        print(text, flush=True, end='', **kwargs)
progressprint = {'maxintervallength': len(' done after ??? ms'),
                 'time'             : [],
                 'indentation'      : 0,
                 'previous'         : '',
                 }
# As the suppress_output and terminal_width user parameters are used
# in fancyprint, they need to be defined before they are actually
# read in as parameters.
cython.declare(suppress_output=dict, terminal_width='int')
suppress_output = {'out': set(), 'err': set()}
terminal_width = 80

# Functions for printing warnings
def warn(*args, skipline=True, prefix='Warning', wrap=True, **kwargs):
    try:
        universals.any_warnings = True
    except:
        ...
    # Add initial newline (if skipline is True) to prefix
    # and append a colon.
    prefix = '{}{}{}'.format('\n' if skipline else '', prefix, ':' if prefix else '')
    args = list(args)
    if prefix == '\n':
        if args:
            args[0] = '\n' + str(args[0])
        else:
            args = ['\n']
    if prefix:
        args = [prefix] + args
    # Print out message
    fancyprint(*args, fun=terminal.bold_red, wrap=wrap, file=sys.stderr, is_warning=True, **kwargs)

# Versions of fancyprint and warn which may be called collectively
# but only the master will do any printing.
def masterprint(*args, **kwargs):
    if master:
        fancyprint(*args, **kwargs)
def masterwarn(*args, **kwargs):
    if master:
        warn(*args, **kwargs)

# Raised exceptions inside cdef functions do not generally propagte
# out to the caller. In places where exceptions are normally raised
# manualy, call this function with a descriptive message instead.
# Also, to ensure correct teardown of the MPI environment before
# exiting, call this function with exit_code=0 to shutdown a
# successfull CO𝘕CEPT run.
def abort(*args, exit_code=1, prefix='Aborting', **kwargs):
    # Print out final messages
    if exit_code != 0:
        warn(*args, prefix=prefix, **kwargs)
        sleep(0.1)
    if master:
        masterprint('Total execution time: {}'.format(time_since(start_time)), **kwargs)
    # Ensure that every printed message is flushed
    sys.stderr.flush()
    sys.stdout.flush()
    sleep(0.1)
    # For a proper exit, all processes should reach this point
    if exit_code == 0:
        Barrier()
    # Shut down the Python process unless we are running interactively
    if not sys.flags.interactive:
        # Tear down the MPI environment, either gently or forcefully
        if exit_code == 0:
            MPI.Finalize()
        else:
            comm.Abort(exit_code)
        # Exit Python
        sys.exit(exit_code)



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
            # Allocate pointer array of pointers (e.g. double**).
            # Emulate these as lists of arrays.
            return [empty(1, dtype=sizeof(dtype[:-1]).dtype)]
        else:
            dtype=object
        return np.array([1], dtype=dtype)
    def malloc(a):
        if isinstance(a, list):
            # Pointer to pointer represented as list of arrays
            return a
        return empty(int(a[0]), dtype=a.dtype)
    def realloc(p, a):
        if isinstance(p, list):
            # Reallocation of pointer array (e.g. double**)
            size = len(a)
            p = p[:size] + [None]*(size - len(p))
        else:
            # Reallocation pointer (e.g. double*)
            size = int(a[0])
            p.resize(size, refcheck=False)
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
                       arcsin, arccos, arctan, arctan2,
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
    # The ℝ, ℤ and 𝔹 dicts for constant expressions
    class BlackboardBold(dict):
        def __init__(self, constant_type):
            self.constant_type = constant_type
        def __getitem__(self, key):
            return self.constant_type(key)
    ℝ = BlackboardBold(float)
    ℤ = BlackboardBold(int)
    𝔹 = BlackboardBold(bool)
    # The cimport function, which in the case of pure Python should
    # simply execute the statements passed to it as a string,
    # within the namespace of the call.
    def cimport(import_statement):
        import_statement = import_statement.strip()
        if import_statement.endswith(','):
            import_statement = import_statement[:-1]
        module = inspect.getmodule(inspect.stack()[1][0])
        d = globals() if module is None else module.__dict__
        exec(import_statement, d)
    # A dummy context manager for use with loop unswitching
    class DummyContextManager:
        def __call__(self, *args):
            return self
        def __enter__(self):
            ...
        def __exit__(self, *exc_info):
            ...
    unswitch = DummyContextManager()
    # The pxd function, which in pure Python defines the variables
    # passed in as a string in the global name space.
    class DummyPxd:
        def __getitem__(self, key):
            return self
        def __getattr__(self, name):
            return self
        def __call__(self, *args, **kwargs):
            return self
    dummypxd = DummyPxd()
    def pxd(s):
        # Remove comments
        lines = s.split('\n')
        code_lines = []
        for line in lines:
            if not line.lstrip().startswith('#'):
                code_lines.append(line)
        s = ' '.join(code_lines)
        # Find all words
        words = set(re.findall('[_a-zA-Z][_0-9a-zA-Z]*', s))
        # Remove non-variable words
        for word in (
            '',
            # Python keywords
            *keyword.kwlist,
            # Cython keywords
            'cdef', 'cpdef', 'cimport', 'extern', 'ctypedef', 'struct',
            # Types
            'void', 'bint', 'char', 'short', 'int', 'long', 'ptrdiff_t', 'Py_ssize_t', 'unsigned',
            'size_t', 'float', 'double', 'list', 'tuple', 'str', 'dict', 'set',
            ):
            words.discard(word)
        # Declare variables in the name space of the caller
        module = inspect.getmodule(inspect.stack()[1][0])
        d = globals() if module is None else module.__dict__
        for varname in words:
            try:
                d.setdefault(varname, dummypxd)
            except:
                pass
# Function for building "structs" (really simple namespaces).
# In compiled mode, this function body will be copied and
# specialised for each kind of struct created.
# This function returns a struct and a dict over the passed fields.
# Note that these are not linked, meaning that you should not alter
# the values of a (struct, dict)-pair after creation if you are using
# the dict at all (the dict is useful for dynamic evaluations).
# The only pointer type implemented is char*. As other pointer types
# have no corresponding Python type, it is not straightforward
# to implement these.
def build_struct(**kwargs):
    # Regardless of the input kwargs, bring it to the form {key: val}
    ctypes = {}
    for key, val in kwargs.items():
        if isinstance(val, tuple):
            # Type and value given
            ctype, val = val
        else:
            # Only type given. Initialize value to None
            # for pointer type or 0 for non-pointer type.
            ctype, val = val, '__build_struct_undefined__'
        try:
            ctype = C2np[ctype]
        except:
            try:
                ctype = eval(ctype)
            except:
                pass
        ctypes[key] = ctype
        if val == '__build_struct_undefined__':
            try:
                val = ctype()
            except:
                val = b'' if '*' in ctype else 0
        kwargs[key] = val
    for key, val in kwargs.copy().items():
        if isinstance(val, str):
            # Evaluate value given as string expression
            namespace = {k: v for d in (globals(), kwargs) for k, v in d.items()}
            try:
                val = eval(val, namespace)
            except:
                pass
        try:
            kwargs[key] = ctypes[key](val)
        except:
            pass
    if not cython.compiled:
        # In pure Python, emulate a struct by a simple namespace
        struct = types.SimpleNamespace(**kwargs)
    else:
        struct = ...  # To be added by pyxpp
    return struct, kwargs



###################################
# Cython imports and declarations #
###################################
pxd("""
# Import everything from cython_gsl,
# granting access to most of GNU Scientific Library.
from cython_gsl cimport *
# Functions for manual memory management
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
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
                        asin  as arcsin,
                        acos  as arccos,
                        atan  as arctan,
                        atan2 as arctan2,
                        sinh, cosh, tanh,
                        asinh as arcsinh,
                        acosh as arccosh,
                        atanh as arctanh,
                        exp, log, log2, log10,
                        sqrt, cbrt,
                        erf, erfc,
                        floor, ceil, round,
                        fmod,
                        )
""")
# Custom extension types using @cython.cclass will be found by the
# pyxpp preprocessor. A comment containing such types will be placed in
# the .pyx version of the given .py file. These are then collected
# together into the .types.pyx file. The content of the cython_gsl
# module cimported above also contains some extension types however.
# Here we add this comment manually, directly in the .py file.
# Extension types implemented by this module:
#     gsl_...: from cython_gsl cimport *



#####################
# Unicode functions #
#####################
# The pyxpp script convert all Unicode source code characters into
# ASCII using the below function.
@cython.pheader(# Arguments
                s=str,
                # Locals
                c=str,
                char_list=list,
                in_unicode_char='bint',
                pat=str,
                sub=str,
                unicode_char=str,
                unicode_name=str,
                returns=str,
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
cython.declare(unicode_subs=dict, unicode_tags=dict)
unicode_subs = {' ': '__SPACE__',
                '-': '__DASH__',
                }
unicode_tags = {'begin': 'BEGIN_UNICODE__',
                'end':   '__END_UNICODE',
                }

# The function below grants the code access to
# Unicode string literals by undoing the convertion of the
# asciify function above.
@cython.pheader(s=str, returns=str)
def unicode(s):
    return re.sub('{}(.*?){}'.format(unicode_tags['begin'], unicode_tags['end']), unicode_repl, s)
@cython.pheader(# Arguments
                match=object,  # re.match object
                # Locals
                pat=str,
                s=str,
                sub=str,
                returns=str,
                )
def unicode_repl(match):
    s = match.group(1)
    for pat, sub in unicode_subs.items():
        s = s.replace(sub, pat)
    return unicodedata.lookup(s)

# This function takes in a number (string) and
# returns it written in Unicode subscript.
@cython.header(s=str, returns=str)
def unicode_subscript(s):
    return ''.join([unicode_subscripts.get(c, c) for c in s])
cython.declare(unicode_subscripts=dict)
unicode_subscripts = dict(zip('0123456789-+e.', [unicode(c) for c in
    ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉', '₋', '₊', 'ₑ', '.')]))

# This function takes in a number (string) and
# returns it written in Unicode superscript.
def unicode_superscript(s):
    return ''.join([unicode_superscripts.get(c, c) for c in s])
cython.declare(unicode_supercripts=dict)
unicode_superscripts = dict(zip('0123456789-+e.', [unicode(c) for c in (
    '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁻', '', '×10', '⋅')]))

# Function which takes in a string possibly containing units formatted
# in fancy ways and returns a unformatted version.
@cython.pheader(
    # Arguments
    unit_str=str,
    # Locals
    ASCII_char=str,
    after=str,
    before=str,
    c=str,
    i='Py_ssize_t',
    mapping=dict,
    operators=str,
    pat=str,
    rep=str,
    unicode_superscript=str,
    unit_list=list,
    returns=str,
)
def unformat_unit(unit_str):
    """Example of effect:
    '10¹⁰ m☉' -> '10**(10)*m_sun'
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
    mapping = {'×'           : '*',
               '^'           : '**',
               unicode('m☉') : 'm_sun',
               unicode('m_☉'): 'm_sun',
               }
    for pat, rep in mapping.items():
        unit_str = unit_str.replace(pat, rep)
    # Convert superscript to power
    unit_str = re.sub(unicode('[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+'), r'**(\g<0>)', unit_str)
    # Convert unicode superscript ASCII text
    for ASCII_char, unicode_superscript in unicode_superscripts.items():
        if unicode_superscript:
            unit_str = unit_str.replace(unicode_superscript, ASCII_char)
    # Insert an asterisk between numbers and letters
    # (though not the other way around).
    unit_str = re.sub(r'(([^_a-zA-Z0-9\.]|^)[0-9\.\)]+) ?([_a-zA-Z])', r'\g<1>*\g<3>', unit_str)
    unit_str = re.sub(r'([0-9])\*e([0-9+\-])', r'\g<1>e\g<2>', unit_str)
    return unit_str

# Function which converts a string containing (possibly) units
# to the corresponding numerical value.
@cython.pheader(# Arguments
                unit_str=str,
                namespace=dict,
                fail_on_error='bint',
                # Locals
                unit=object,  # double or NoneType
                returns=object,  # double or NoneType
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
    # Remove any formatting on the unit string
    unit_str = unformat_unit(unit_str)
    # Evaluate the transformed unit string
    if namespace is None:
        namespace = units_dict
    namespace = {
        **vars(np),
        **namespace,
        unicode('π'): π,
        asciify('π'): π,
    }
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
cython.declare(paths=dict)
top_dir = os.path.abspath('.')
while True:
    if '.paths' in os.listdir(top_dir):
        break
    elif master and top_dir == '/':
        abort('Cannot find the .paths file!')
    top_dir = os.path.dirname(top_dir)
paths_module = pyxpp.load_source('paths', top_dir + '/.paths')
paths = {key: value for key, value in paths_module.__dict__.items()
    if isinstance(key, str) and not key.startswith('__')}
# Function for converting an absolute path to its "sensible" form.
# That is, this function returns the relative path with respect to the
# concept directory, if it is no more than one directory above the
# concept directory. Otherwise, return the absolute path back again.
@cython.header(# Arguments
               path=str,
               # Locals
               relpath=str,
               returns=str,
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
cython.declare(argd=dict,
               globals_dict=dict,
               jobid='long long int',
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
paths['params_cp'] = argd.get('params_cp', '')
# The jobid of the current run
jobid = int(argd.get('jobid', -1))



#######################
# Read parameter file #
#######################
# Read in the content of the parameter file
cython.declare(params_file_content=str)
params_file_content = ''
if master and os.path.isfile(paths['params_cp']):
    with open(paths['params_cp'], encoding='utf-8') as params_file:
        params_file_content = params_file.read()
params_file_content = bcast(params_file_content)
# Often, h ≡ H0/(100*km/(s*Mpc)) is used in unit specifications.
# To allow for this, we add lines defining h from H0.
params_file_content += '\n'.join([
    '\n# Added by commons.py',
    'h = (H0 if "H0" in globals() else 1)/(100*km/(s*Mpc))',
    'h = float(f"{h:g}")',
])
# All further handling of parameters defined in the parameter file
# will be done later.



###########################
# Dimensionless constants #
###########################
cython.declare(machine_ϵ='double',
               π='double',
               ρ_vacuum='double',
               ထ='double',
               )
machine_ϵ = np.finfo(C2np['double']).eps
π = np.pi
ρ_vacuum = 1e+2*machine_ϵ
ထ = cast(np.inf, 'double')



##################
# Physical units #
##################
# Dict relating all implemented units to the basic three units
# (pc, yr, m_sun). Note that 'min' is not a good
# name for minutes, as this name is already taken by the min function.
# You may specify units here which shall not be part of the units
# actually available in the code (this will come later).
# The specifications here ar purely for the purpose of defining a unit
# system based on the user input.
unit_relations = {
    'yr':    1,
    'pc':    1,
    'm_sun': 1,
}
# Add other time units
unit_relations['kyr'    ] = 1e+3            *unit_relations['yr'     ]
unit_relations['Myr'    ] = 1e+6            *unit_relations['yr'     ]
unit_relations['Gyr'    ] = 1e+9            *unit_relations['yr'     ]
unit_relations['day'    ] = 1/365.25        *unit_relations['yr'     ]  # Exact Julian year
unit_relations['hr'     ] = 1/24            *unit_relations['day'    ]
unit_relations['minutes'] = 1/60            *unit_relations['hr'     ]
unit_relations['s'      ] = 1/60            *unit_relations['minutes']
# Add other length units
unit_relations['kpc'    ] = 1e+3            *unit_relations['pc'     ]
unit_relations['Mpc'    ] = 1e+6            *unit_relations['pc'     ]
unit_relations['Gpc'    ] = 1e+9            *unit_relations['pc'     ]
unit_relations['AU'     ] = π/(60*60*180)   *unit_relations['pc'     ]  # IAU exact definition, 2015
unit_relations['m'      ] = 1/149597870700  *unit_relations['AU'     ]  # IAU exact definition, 2012
unit_relations['mm'     ] = 1e-3            *unit_relations['m'      ]
unit_relations['cm'     ] = 1e-2            *unit_relations['m'      ]
unit_relations['km'     ] = 1e+3            *unit_relations['m'      ]
unit_relations['ly'     ] = (                                           # CGPM exact definition, 1983
    (299792458*unit_relations['m']/unit_relations['s'])*unit_relations['yr']
)
unit_relations['kly'    ] = 1e+3            *unit_relations['ly'     ]
unit_relations['Mly'    ] = 1e+6            *unit_relations['ly'     ]
unit_relations['Gly'    ] = 1e+9            *unit_relations['ly'     ]
# Add other mass units
unit_relations['km_sun' ] = 1e+3            *unit_relations['m_sun'  ]
unit_relations['Mm_sun' ] = 1e+6            *unit_relations['m_sun'  ]
unit_relations['Gm_sun' ] = 1e+9            *unit_relations['m_sun'  ]
unit_relations['kg'     ] = 1/1.98848e+30   *unit_relations['m_sun'  ]  # Particle data group, 2017
unit_relations['g'      ] = 1e-3            *unit_relations['kg'     ]
# Add energy units
unit_relations['J'      ] = (
    unit_relations['kg']*unit_relations['m']**2*unit_relations['s']**(-2)
)
unit_relations['eV'     ] = 1.6021766208e-19*unit_relations['J'      ]  # Particle data group, 2017
unit_relations['keV'    ] = 1e+3            *unit_relations['eV'     ]
unit_relations['MeV'    ] = 1e+6            *unit_relations['eV'     ]
unit_relations['GeV'    ] = 1e+9            *unit_relations['eV'     ]
# Add additional units
unit_relations['light_speed'] = unit_relations['c'] = unit_relations['ly']/unit_relations['yr']
unit_relations['h_bar'] = unit_relations['hbar'] = unit_relations[unicode('ħ')] = unit_relations[asciify('ħ')] = (
    1.054571800e-34                                                     # Particle data group, 2017
    *unit_relations['kg']*unit_relations['m']**2/unit_relations['s']
)
unit_relations['G_Newton'] = unit_relations['G'] = (                     # Particle data group, 2017
    6.67408e-11*unit_relations['m']**3/(unit_relations['kg']*unit_relations['s']**2)
)



# Function which given a function name present both in NumPy and in the
# Python builtins will produce a function which first calls the NumPy
# version, and then (on failure) calls the builtin version. This is
# useful for e.g. generating a 'min' function which works on both a
# single multi-dimensional array as well as multiple scalar inputs.
def produce_np_and_builtin_function(funcname):
    np_func = getattr(np, funcname)
    try:
        builtin_func = getattr(__builtins__, funcname)
    except:
        builtin_func = __builtins__[funcname]
    def np_and_builtin_function(*args, **kwargs):
        try:
            return np_func(*args, **kwargs)
        except:
            return builtin_func(*args, **kwargs)
    return np_and_builtin_function
# Attempt to read in the parameter file. This is really only
# to get any user defined units, as the parameter file will
# be properly read in later. First construct a namespace in which the
# parameters can be read in.
def construct_user_params_namespace(params_iteration):
    return {# Include all of NumPy
            **vars(np).copy(),
            # Overwrite the NumPy min and max function with NumPy and
            # builtin hybrid min and max functions.
            'min': produce_np_and_builtin_function('min'),
            'max': produce_np_and_builtin_function('max'),
            # The paths dict
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
            # MPI variables and functions
            'master'         : master,
            'nprocs'         : nprocs,
            'rank'           : rank,
            'bcast'          : bcast,
            'call_openmp_lib': call_openmp_lib,
            # Constants
            'machine_ϵ'         : machine_ϵ,
            unicode('machine_ϵ'): machine_ϵ,
            'eps'               : machine_ϵ,
            'ထ'                 : ထ,
            unicode('ထ')        : ထ,
            # Print and abort functions
            'fancyprint' : fancyprint,
            'masterprint': masterprint,
            'warn'       : warn,
            'masterwarn' : masterwarn,
            'abort'      : abort,
            # The number of times this namespace has been constructed
            'params_iteration': params_iteration,
            }
user_params = construct_user_params_namespace('units')
# Add units to the user_params namespace.
# These do not represent the choice of units; the names should merely
# exist to avoid errors when reading the parameter file.
user_params.update(unit_relations)
# Set default values of inferred parameters
inferred_params = {unicode(key): val for key, val in
    {
        'Ων': 0,
    }.items()
}
for key, val in inferred_params.items():
    user_params.setdefault(key, val)
# Function which repeatedly executes the str content in the
# dictionary d, ignoring any execeptions. The executation stops when
# no more exceptions are resolved by definitions further down
# in the content.
def exec_params(content, d, suppress_exceptions=True):
    lines = pyxpp.oneline(content.split('\n'))
    lines_executed = []
    lines_executed_prev = [-1]
    while lines_executed != lines_executed_prev:
        lines_executed_prev = lines_executed
        lines_executed = []
        d_copy = d.copy()
        for n, line in enumerate(lines):
            try:
                exec(line, d)
                lines_executed.append(n)
            except:
                pass
    if not suppress_exceptions:
        exec(content, d)
# Execute the content of the parameter file in the namespace defined
# by user_params in order to get the user defined units.
exec_params(params_file_content, user_params)
# The names of the three fundamental units,
# all with a numerical value of 1. If these are not defined in the
# parameter file, give them some reasonable values.
cython.declare(unit_time=str,
               unit_length=str,
               unit_mass=str,
               )
unit_time   = unformat_unit(user_params.get('unit_time'  , 'Gyr'        ))
unit_length = unformat_unit(user_params.get('unit_length', 'Mpc'        ))
unit_mass   = unformat_unit(user_params.get('unit_mass'  , '1e+10*m_sun'))
# Construct a struct containing the values of all units
units, units_dict = build_struct(# Values of basic units,
                                 # determined from the choice of fundamental units.
                                 yr      = ('double', 1/eval_unit(unit_time  , unit_relations)),
                                 pc      = ('double', 1/eval_unit(unit_length, unit_relations)),
                                 m_sun   = ('double', 1/eval_unit(unit_mass  , unit_relations)),
                                 # Other time units
                                 kyr     = ('double', 'unit_relations["kyr"    ]*yr'      ),
                                 Myr     = ('double', 'unit_relations["Myr"    ]*yr'      ),
                                 Gyr     = ('double', 'unit_relations["Gyr"    ]*yr'      ),
                                 day     = ('double', 'unit_relations["day"    ]*yr'      ),
                                 hr      = ('double', 'unit_relations["hr"     ]*yr'      ),
                                 minutes = ('double', 'unit_relations["minutes"]*yr'      ),
                                 s       = ('double', 'unit_relations["s"      ]*yr'      ),
                                 # Other length units
                                 kpc     = ('double', 'unit_relations["kpc"    ]*pc'      ),
                                 Mpc     = ('double', 'unit_relations["Mpc"    ]*pc'      ),
                                 Gpc     = ('double', 'unit_relations["Gpc"    ]*pc'      ),
                                 AU      = ('double', 'unit_relations["AU"     ]*pc'      ),
                                 m       = ('double', 'unit_relations["m"      ]*pc'      ),
                                 mm      = ('double', 'unit_relations["mm"     ]*pc'      ),
                                 cm      = ('double', 'unit_relations["cm"     ]*pc'      ),
                                 km      = ('double', 'unit_relations["km"     ]*pc'      ),
                                 ly      = ('double', 'unit_relations["ly"     ]*pc'      ),
                                 kly     = ('double', 'unit_relations["kly"    ]*pc'      ),
                                 Mly     = ('double', 'unit_relations["Mly"    ]*pc'      ),
                                 Gly     = ('double', 'unit_relations["Gly"    ]*pc'      ),
                                 # Other mass units
                                 km_sun  = ('double', 'unit_relations["km_sun" ]*m_sun'   ),
                                 Mm_sun  = ('double', 'unit_relations["Mm_sun" ]*m_sun'   ),
                                 Gm_sun  = ('double', 'unit_relations["Gm_sun" ]*m_sun'   ),
                                 kg      = ('double', 'unit_relations["kg"     ]*m_sun'   ),
                                 g       = ('double', 'unit_relations["g"      ]*m_sun'   ),
                                 # Energy units
                                 J       = ('double', ('unit_relations["J"     ]*m_sun   '
                                                       '                        *pc**2   '
                                                       '                        *yr**(-2)'
                                                       )                                  ),
                                 eV      = ('double', ('unit_relations["eV"    ]*m_sun   '
                                                       '                        *pc**2   '
                                                       '                        *yr**(-2)'
                                                       )                                  ),
                                 keV     = ('double', ('unit_relations["keV"   ]*m_sun   '
                                                       '                        *pc**2   '
                                                       '                        *yr**(-2)'
                                                       )                                  ),
                                 MeV     = ('double', ('unit_relations["MeV"   ]*m_sun   '
                                                       '                        *pc**2   '
                                                       '                        *yr**(-2)'
                                                       )                                  ),
                                 GeV     = ('double', ('unit_relations["GeV"   ]*m_sun   '
                                                       '                        *pc**2   '
                                                       '                        *yr**(-2)'
                                                       )                                  ),
                                 )



######################
# Physical constants #
######################
cython.declare(light_speed='double',
               ħ='double',
               G_Newton='double',
               )
# The speed of light in vacuum
light_speed = units.ly/units.yr
# Reduced Planck constant
ħ = (
    unit_relations['ħ']
    /(unit_relations['kg']*unit_relations['m']**2/unit_relations['s'])
    *units.kg*units.m**2/units.s
)
# Newton's gravitational constant
G_Newton = (
    unit_relations['G_Newton']
    /(unit_relations['m']**3/(unit_relations['kg']*unit_relations['s']**2))
    *units.m**3/(units.kg*units.s**2)
)



################################################################
# Import all user specified parameters from the parameter file #
################################################################
# Function for converting a non-iterables to lists of one element and
# iterables to lists. The str and bytes types are treated
# as non-iterables. If a generator is passed it will be consumed.
def any2list(val):
    if isinstance(val, (str, bytes)):
        val = [val]
    try:
        iter(val)
    except TypeError:
        val = [val]
    return list(val)
# Function which replaces ellipses in a dict with previous value
def replace_ellipsis(d):
    """Note that for deterministic behavior,
    this function assumes that the dictionary d is ordered.
    """
    if not isinstance(d, dict):
        return d
    truthy_val = None
    for _ in range(2):
        for key, val in d.items():
            if any(any2list(truthy_val)) and val is ...:
                d[key] = truthy_val
            elif val is not ... and any(any2list(val)):
                truthy_val = val
    falsy_val = truthy_val
    for key, val in d.items():
        if val is ...:
            d[key] = falsy_val
        else:
            falsy_val = val
    return d
# Subclass the dict to create a dict-like object which keeps track of
# the number of lookups on each key. This is used to identify unknown
# (and therefore unused) parameters defined by the user.
# Transform all keys to unicode during lookups and assignments.
class DictWithCounter(dict):
    def __init__(self, d=None):
        self.counter = collections.defaultdict(int)
        if d is None:
            super().__init__()
        else:
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
user_params = DictWithCounter(construct_user_params_namespace('inferrables'))
# Additional things which should be available when defining parameters
user_params.update({# Units from the units struct
                    **units_dict,
                    # Physical constants
                    'light_speed': light_speed,
                    'ħ'          : ħ,
                    'h_bar'      : ħ,
                    'G_Newton'   : G_Newton,
                    # Inferred parameters (values not yet inferred)
                    **inferred_params,
                    })
# At this point, user_params does not contain actual parameters.
# Mark all items in user_params as used.
user_params.useall()
# "Import" the parameter file by executing it
# in the namespace defined by the user_params namespace.
exec_params(params_file_content, user_params, suppress_exceptions=False)
# Also mark the unit-parameters as used
for u in ('length', 'time', 'mass'):
    user_params.use(f'unit_{u}')
# Update the class_params inside of the user_params with default values.
# This is important for the variable inference to come. For the actual
# (module level) class_params, this has to be done again (see the
# CLASS setup section).
if 'class_params' in user_params:
    replace_ellipsis(user_params['class_params'])
    # Specification of general, default CLASS parameters.
    class_params_default = {}
    if 'H0' in user_params:
        class_params_default['H0'] = user_params['H0']/(units.km/(units.s*units.Mpc))
    if 'Ωcdm' in user_params:
        class_params_default['Omega_cdm'] = user_params['Ωcdm']
    if 'Ωb' in user_params:
        class_params_default['Omega_b'] = user_params['Ωb']
    # Add in neutrino CLASS parameters, if neutrinos are present
    if int(user_params['class_params'].get('N_ncdm', 0)) != 0:
        class_params_default.update({# Disable fluid approximation for non-CDM species
                                     'ncdm_fluid_approximation': 3,
                                     # Neutrino options needed for accurate δP/δρ
                                     'Quadrature strategy': 3,
                                     'evolver': 0,
                                     'Number of momentum bins': 25,
                                     'Maximum q': 15.0,
                                     'l_max_ncdm': 50,
                                     })
    # Apply updates to the CLASS parameters
    for param_name, param_value in class_params_default.items():
        user_params['class_params'].setdefault(param_name, param_value)
# Find out which of the inferrable parameters are not explicitly set,
# and so should be inferred.
user_params_changed_inferrables = user_params.copy()
for key, val in inferred_params.items():
    # Make sure to implement every possible type
    # of the inferrable parameters.
    if val is None:
        user_params_changed_inferrables[key] = False
    elif val is ...:
        user_params_changed_inferrables[key] = True
    elif isinstance(val, bool):
        user_params_changed_inferrables[key] = not val
    elif isinstance(val, bytes):
        user_params_changed_inferrables[key] += b' '
    elif isinstance(val, str):
        user_params_changed_inferrables[key] += ' '
    elif isinstance(val, (tuple, list, set)):
        user_params_changed_inferrables[key] = type(val)(list(val) + [0])
    else:
        # Assume numeric type
        user_params_changed_inferrables[key] += 1
user_params_changed_inferrables['params_iteration'] = 'inferrables (changed)'
exec_params(params_file_content, user_params_changed_inferrables, suppress_exceptions=False)
inferred_params_set = DictWithCounter({
    key: user_params[key] == user_params_changed_inferrables[key] for key in inferred_params
})
# Infer inferrable parameters which have not been explicitly set
# by the user. To ensure that every key gets looked up in their proper
# unicode form, we wrap the inferred_params dict in a DictWithCounter.
inferred_params = DictWithCounter(inferred_params)
inferred_params_final = DictWithCounter()
cython.declare(
    Ων='double',
)
Ων = float(inferred_params['Ων'])
if inferred_params_set['Ων']:
    Ων = float(user_params['Ων'])
else:
    if int(user_params.get('class_params', {}).get('N_ncdm', 0)) != 0:
        cosmo = Class()
        cosmo.set(user_params.get('class_params', {}))
        masterprint('Calling CLASS in order to determine Ων ...')
        call_openmp_lib(cosmo.compute)
        masterprint('done')
        background = cosmo.get_background()
        if bcast('(.)rho_ncdm[0]' in background):
            Ων = bcast(background['(.)rho_ncdm[0]'][-1]/background['(.)rho_crit'][-1]
                if master else None)
inferred_params_final['Ων'] = Ων
# Update user_params with the correct values for the inferred params
user_params.update(inferred_params_final)
# Re-"import" the parameter file by executing it
# in the namespace defined by the user_params namespace,
# this time with the inferrable values in place.
user_params['params_iteration'] = 'final'
exec_params(params_file_content, user_params, suppress_exceptions=False)
# Backup of originally supplied user parameter names
user_params_keys_raw = set(user_params.keys())
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
def to_rgbα(value, default_α=1):
    if isinstance(value, str):
        return to_rgb(value), default_α
    value = any2list(value)
    if len(value) == 1:
        return to_rgb(value[0]), default_α
    elif len(value) == 2:
        return to_rgb(value[0]), value[1]
    elif len(value) == 3:
        return to_rgb(value), default_α
    elif len(value) == 4:
        return to_rgb(value[:3]), value[3]
    # Could not convert value to color and α
    return np.array([-1, -1, -1]), default_α
cython.declare(# Input/output
               initial_conditions=object,  # str or container of str's
               snapshot_type=str,
               output_dirs=dict,
               output_bases=dict,
               output_times=dict,
               autosave_interval='double',
               snapshot_select=dict,
               powerspec_select=dict,
               render2D_select=dict,
               render3D_select=dict,
               # Numerical parameter
               boxsize='double',
               ewald_gridsize='Py_ssize_t',
               φ_gridsize='ptrdiff_t',
               p3m_scale='double',
               p3m_cutoff='double',
               R_tophat='double',
               modes_per_decade='double',
               # Cosmology
               H0='double',
               Ωcdm='double',
               Ωb='double',
               a_begin='double',
               t_begin='double',
               class_params=dict,
               # Physics
               select_forces=dict,
               select_class_species=dict,
               select_eos_w=dict,
               select_boltzmann_closure=dict,
               select_realization_options=dict,
               select_approximations=dict,
               select_softening_length=dict,
               # Simlation options
               fftw_wisdom_rigor=str,
               fftw_wisdom_reuse='bint',
               random_seed='unsigned long int',
               fluid_scheme_select=dict,
               fluid_options=dict,
               class_k_max=dict,
               class_reuse='bint',
               class_plot_perturbations='bint',
               class_extra_background=set,
               class_extra_perturbations=set,
               # Graphics
               terminal_width='int',
               suppress_output=dict,
               render2D_options=dict,
               render3D_colors=dict,
               render3D_bgcolor='double[::1]',
               render3D_resolution='int',
               # Debugging options
               enable_Hubble='bint',
               enable_class_background='bint',
               enable_Ewald='bint',
               enable_debugging='bint',
               # Hidden parameters
               special_params=dict,
               output_times_full=dict,
               initial_time_step='Py_ssize_t',
               Δt_begin_autosave='double',
               Δt_autosave='double',
               )
# Input/output
initial_conditions = user_params.get('initial_conditions', '')
user_params['initial_conditions'] = initial_conditions
snapshot_type = (str(user_params.get('snapshot_type', 'standard'))
    .lower().replace(' ', '').replace('-', '')
)
user_params['snapshot_type'] = snapshot_type
output_dirs = dict(user_params.get('output_dirs', {}))
replace_ellipsis(output_dirs)
for kind in ('snapshot', 'powerspec', 'render2D', 'render3D'):
    output_dirs[kind] = str(output_dirs.get(kind, paths['output_dir']))
    if not output_dirs[kind]:
        output_dirs[kind] = paths['output_dir']
output_dirs['autosave'] = str(output_dirs.get('autosave', ''))
if not output_dirs['autosave']:
    output_dirs['autosave'] = paths['ics_dir']
output_dirs = {key: sensible_path(path) for key, path in output_dirs.items()}
user_params['output_dirs'] = output_dirs
output_bases = dict(user_params.get('output_bases', {}))
replace_ellipsis(output_bases)
for kind in ('snapshot', 'powerspec', 'render2D', 'render3D'):
    output_bases[kind] = str(output_bases.get(kind, kind))
user_params['output_bases'] = output_bases
output_times = dict(user_params.get('output_times', {}))
replace_ellipsis(output_times)
user_params['output_times'] = output_times
autosave_interval = float(
    0 if not user_params.get('autosave_interval', 0)
      else   user_params.get('autosave_interval', 0)
)
user_params['autosave_interval'] = autosave_interval
snapshot_select = {'all': True}
if user_params.get('snapshot_select'):
    if isinstance(user_params['snapshot_select'], dict):
        snapshot_select = user_params['snapshot_select']
        replace_ellipsis(snapshot_select)
    else:
        snapshot_select = {'all': user_params['snapshot_select']}
user_params['snapshot_select'] = snapshot_select
powerspec_select = {'all': True, 'all combinations': True}
if user_params.get('powerspec_select'):
    if isinstance(user_params['powerspec_select'], dict):
        powerspec_select = user_params['powerspec_select']
        replace_ellipsis(powerspec_select)
    else:
        powerspec_select = {'all':              bool(user_params['powerspec_select']),
                            'all combinations': bool(user_params['powerspec_select']),
                            }
for key, val in powerspec_select.copy().items():
    if isinstance(val, dict):
        val.setdefault('data', False)
        val.setdefault('plot', False)
    else:
        powerspec_select[key] = {'data': bool(val), 'plot': bool(val)}
user_params['powerspec_select'] = powerspec_select
render2D_select = {'all': True, 'all combinations': True}
if user_params.get('render2D_select'):
    if isinstance(user_params['render2D_select'], dict):
        render2D_select = user_params['render2D_select']
        replace_ellipsis(render2D_select)
    else:
        render2D_select = {
            'all':              bool(user_params['render2D_select']),
            'all combinations': bool(user_params['render2D_select']),
        }
for key, val in render2D_select.copy().items():
    if isinstance(val, dict):
        val.setdefault('data', False)
        val.setdefault('image', False)
        val.setdefault('terminal image', False)
    else:
        render2D_select[key] = {
            'data'          : bool(val),
            'image'         : bool(val),
            'terminal image': bool(val),
        }
user_params['render2D_select'] = render2D_select
render3D_select = {'all': True}
if user_params.get('render3D_select'):
    if isinstance(user_params['render3D_select'], dict):
        render3D_select = user_params['render3D_select']
        replace_ellipsis(render3D_select)
    else:
        render3D_select = {'all': user_params['render3D_select']}
user_params['render3D_select'] = render3D_select
# Numerical parameters
boxsize = float(user_params.get('boxsize', 1))
user_params['boxsize'] = boxsize
ewald_gridsize = to_int(user_params.get('ewald_gridsize', 64))
user_params['ewald_gridsize'] = ewald_gridsize
φ_gridsize = to_int(user_params.get('φ_gridsize', 64))
user_params['φ_gridsize'] = φ_gridsize
p3m_scale = float(user_params.get('p3m_scale', 1.25))
user_params['p3m_scale'] = p3m_scale
p3m_cutoff = float(user_params.get('p3m_cutoff', 4.8))
user_params['p3m_cutoff'] = p3m_cutoff
R_tophat = float(user_params.get('R_tophat', 8*units.Mpc))
user_params['R_tophat'] = R_tophat
modes_per_decade = float(user_params.get('modes_per_decade', 100))
user_params['modes_per_decade'] = modes_per_decade
# Cosmology
H0 = float(user_params.get('H0', 70*units.km/(units.s*units.Mpc)))
user_params['H0'] = H0
Ωcdm = float(user_params.get('Ωcdm', 0.25))
user_params['Ωcdm'] = Ωcdm
Ωb = float(user_params.get('Ωb', 0.05))
user_params['Ωb'] = Ωb
a_begin = float(user_params.get('a_begin', 1))
user_params['a_begin'] = a_begin
t_begin = float(user_params.get('t_begin', 0))
user_params['t_begin'] = t_begin
class_params = dict(user_params.get('class_params', {}))
replace_ellipsis(class_params)
user_params['class_params'] = class_params
# Physics
default_force_method = {
    'gravity': 'pm',
}
select_forces = {}
for key, val in replace_ellipsis(dict(user_params.get('select_forces', {}))).items():
    if isinstance(val, dict):
        select_forces[key] = replace_ellipsis(val)
    elif isinstance(val, str):
        select_forces[key] = {val: default_force_method[val.lower()]}
    elif isinstance(val, tuple) or isinstance(val, list):
        if len(val) == 1:
            select_forces[key] = {val[0]: default_force_method[val[0].lower()]}
        elif len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], str):
            if val[1] in default_force_method:
                select_forces[key] = {
                    val[0]: default_force_method[val[0].lower()],
                    val[1]: default_force_method[val[1].lower()],
                }
            else:
                select_forces[key] = {val[0]: val[1]}
        else:
            new_val = {}
            for el in val:
                if isinstance(el, dict):
                    new_val.update(replace_ellipsis(el))
                elif isinstance(el, tuple) or isinstance(el, list):
                    if len(el) == 1:
                        new_val[el[0]] = default_force_method[el[0].lower()]
                    if len(el) == 2:
                        new_val[el[0]] = el[1]
                elif isinstance(el, str):
                    new_val[el] = default_force_method[el.lower()]
            select_forces[key] = new_val
    subd = {}
    for subd_key, subd_val in select_forces[key].items():
        subd_key, subd_val = subd_key.lower(), subd_val.lower()
        for char in ' -^()':
            subd_key, subd_val = subd_key.replace(char, ''), subd_val.replace(char, '')
        for n in range(10):
            subd_key = subd_key.replace(unicode_superscript(str(n)), str(n))
            subd_val = subd_val.replace(unicode_superscript(str(n)), str(n))
        subd[subd_key] = subd_val
    select_forces[key] = subd
user_params['select_forces'] = select_forces
select_class_species = {}
if user_params.get('select_class_species'):
    if isinstance(user_params['select_class_species'], dict):
        select_class_species = user_params['select_class_species']
        replace_ellipsis(select_class_species)
    else:
        select_class_species = {'all': str(user_params['select_class_species'])}
select_class_species['default'] = 'default'
user_params['select_class_species'] = select_class_species
select_eos_w = {}
if user_params.get('select_eos_w'):
    if isinstance(user_params['select_eos_w'], dict):
        select_eos_w = user_params['select_eos_w']
        replace_ellipsis(select_eos_w)
    else:
        select_eos_w = {'all': user_params['select_eos_w']}
select_eos_w.setdefault('particles', 'default')
select_eos_w['default'] = 'class'
user_params['select_eos_w'] = select_eos_w
select_boltzmann_closure = {}
if user_params.get('select_boltzmann_closure'):
    if isinstance(user_params['select_boltzmann_closure'], dict):
        select_boltzmann_closure = user_params['select_boltzmann_closure']
        replace_ellipsis(select_boltzmann_closure)
    else:
        select_boltzmann_closure = {'all': str(user_params['select_boltzmann_closure'])}
select_boltzmann_closure['default'] = 'class'
user_params['select_boltzmann_closure'] = select_boltzmann_closure
select_realization_options = {}
if user_params.get('select_realization_options'):
    select_realization_options = dict(
        user_params['select_realization_options']
    )
    replace_ellipsis(select_realization_options)
    if select_realization_options:
        for d in select_realization_options.values():
            replace_ellipsis(d)
user_params['select_realization_options'] = select_realization_options
select_approximations = {}
if user_params.get('select_approximations'):
    select_approximations = dict(user_params['select_approximations'])
    replace_ellipsis(select_approximations)
    if select_approximations:
        key = tuple(select_approximations.keys())[0]
        if isinstance(key, str) and '=' in key:
            select_approximations = {'all': select_approximations}
        for d in select_approximations.values():
            replace_ellipsis(d)
select_approximations['default'] = {'P=wρ': False}
user_params['select_approximations'] = select_approximations
select_softening_length = {}
if user_params.get('select_softening_length'):
    if isinstance(user_params['select_softening_length'], dict):
        select_softening_length = user_params['select_softening_length']
        replace_ellipsis(select_softening_length)
    else:
        select_softening_length = {'all': user_params['select_softening_length']}
select_softening_length.setdefault('particles', '0.03*boxsize/cbrt(N)')
select_softening_length.setdefault('fluid', 0)
user_params['select_softening_length'] = select_softening_length
# Simulation options
fftw_wisdom_rigor = user_params.get('fftw_wisdom_rigor', 'estimate').lower()
user_params['fftw_wisdom_rigor'] = fftw_wisdom_rigor
fftw_wisdom_reuse = bool(user_params.get('fftw_wisdom_reuse', True))
user_params['fftw_wisdom_reuse'] = fftw_wisdom_reuse
random_seed = to_int(user_params.get('random_seed', 1))
user_params['random_seed'] = random_seed
fluid_scheme_select = {'all': 'Kurganov-Tadmor'}
if user_params.get('fluid_scheme_select'):
    if isinstance(user_params['fluid_scheme_select'], dict):
        fluid_scheme_select = user_params['fluid_scheme_select']
        replace_ellipsis(fluid_scheme_select)
    else:
        fluid_scheme_select = {'all': user_params['fluid_scheme_select']}
fluid_scheme_select['default'] = 'Kurganov-Tadmor'
fluid_scheme_select = {
    key: str(val).lower().replace(' ', '').replace('-', '')
    for key, val in fluid_scheme_select.items()
}
user_params['fluid_scheme_select'] = fluid_scheme_select
fluid_options_defaults = {
    'kurganovtadmor': {
        'rungekuttaorder': 2,
        'flux_limiter_select': 'minmod',
    },
    'maccormack': {
        'vacuum_corrections_select'    : True,
        'max_vacuum_corrections_select': (1, 'gridsize'),
        'foresight_select'             : 30,
        'smoothing_select'             : 1,
    }
}
fluid_options = dict(user_params.get('fluid_options', {}))
fluid_options = {
    scheme.lower().replace(' ', '').replace('-', ''): {
        key.lower().replace(' ', '').replace('-', ''): val for key, val in d.items()
    }
    for scheme, d in fluid_options.items()
}
for scheme, d in fluid_options_defaults.items():
    fluid_options.setdefault(scheme, d.copy())
    for key, val in d.items():
        fluid_options[scheme].setdefault(key, val)
for scheme, d in fluid_options.items():
    for key, val in d.copy().items():
        if isinstance(val, dict):
            replace_ellipsis(fluid_options[scheme][key])
        else:
            fluid_options[scheme][key] = {
                'all': val,
            }
for scheme, d in fluid_options_defaults.items():
    for key, val in d.items():
        fluid_options[scheme][key]['default'] = val
fluid_options['kurganovtadmor']['flux_limiter_select'] = {
    key: val.lower().replace(' ', '').replace('-', '')
    for key, val in fluid_options['kurganovtadmor']['flux_limiter_select'].items()
}
fluid_options['maccormack']['vacuum_corrections_select'] = {
    key: bool(val)
    for key, val in fluid_options['maccormack']['vacuum_corrections_select'].items()
}
for key, val in fluid_options['maccormack']['max_vacuum_corrections_select'].copy().items():
    val = any2list(val)
    for i, el in enumerate(val):
        if isinstance(el, str):
            el = el.lower()
        else:
            el = int(np.round(el))
        val[i] = el
    if len(val) == 1:
        val *= 2
    fluid_options['maccormack']['max_vacuum_corrections_select'][key] = val
fluid_options['maccormack']['foresight_select'] = {
    key: int(np.round(val))
    for key, val in fluid_options['maccormack']['foresight_select'].items()
}
fluid_options['maccormack']['smoothing_select'] = {
    key: float(val)
    for key, val in fluid_options['maccormack']['smoothing_select'].items()
}
user_params['fluid_options'] = fluid_options
if 'class_k_max' in user_params:
    if isinstance(user_params['class_k_max'], dict):
        class_k_max = replace_ellipsis(user_params['class_k_max'])
    else:
        class_k_max = {'all': user_params['class_k_max']}
else:
    class_k_max = {}
user_params['class_k_max'] = class_k_max
class_reuse = bool(user_params.get('class_reuse', True))
user_params['class_reuse'] = class_reuse
class_plot_perturbations = bool(user_params.get('class_plot_perturbations', False))
user_params['class_plot_perturbations'] = class_plot_perturbations
class_extra_background = set(
    str(el) for el in any2list(user_params.get('class_extra_background', [])) if el
)
user_params['class_extra_background'] = class_extra_background
class_extra_perturbations = set(
    str(el) for el in any2list(user_params.get('class_extra_perturbations', [])) if el
)
user_params['class_extra_perturbations'] = class_extra_perturbations
# Graphics
terminal_width = to_int(user_params.get('terminal_width', 80))
user_params['terminal_width'] = terminal_width
suppress_output = {}
if 'suppress_output' in user_params:
    if isinstance(user_params['suppress_output'], str):
        suppress_output = {'all': {user_params['suppress_output']}}
    elif isinstance(user_params['suppress_output'], dict):
        suppress_output = user_params['suppress_output']
    else:
        suppress_output = {'all': set(user_params['suppress_output'])}
suppress_output.setdefault('out', set())
suppress_output.setdefault('err', set())
suppress_output.setdefault('all', set())
for key, val in suppress_output.copy().items():
    if isinstance(val, str):
        suppress_output[key] = {val}
    else:
        suppress_output[key] = set(val)
    s = set()
    for pattern in suppress_output[key]:
        pattern = str(pattern)
        if pattern:
            s.add(pattern)
    suppress_output[key] = s
suppress_output['out'] |= suppress_output['all']
suppress_output['err'] |= suppress_output['all']
user_params['suppress_output'] = suppress_output
render2D_options_defaults = {
    'axis'               : 'z',
    'extend'             : (0, 0.1*boxsize),
    'terminal resolution': np.min([terminal_width, φ_gridsize]),
    'colormap'           : 'inferno',
    'enhance'            : True,
}
render2D_options = dict(user_params.get('render2D_options', {}))
for key, val in render2D_options_defaults.items():
    render2D_options.setdefault(key, val)
for key, val in render2D_options.copy().items():
    if isinstance(val, dict):
        replace_ellipsis(render2D_options[key])
    else:
        render2D_options[key] = {
            'all': val,
            'all combinations': val,
        }
for key, val in render2D_options['extend'].copy().items():
    if len(any2list(val)) == 1:
        render2D_options['extend'][key] = (0, val)
    else:
        render2D_options['extend'][key] = (np.min(val), np.max(val))
for key, val in render2D_options_defaults.items():
    render2D_options[key]['default'] = val
user_params['render2D_options'] = render2D_options
render3D_colors = {}
if 'render3D_colors' in user_params:
    if isinstance(user_params['render3D_colors'], dict):
        render3D_colors = user_params['render3D_colors']
        replace_ellipsis(render3D_colors)
    else:
        render3D_colors = {'all': user_params['render3D_colors']}
render3D_colors = {key.lower(): to_rgbα(val, 0.2) for key, val in render3D_colors.items()}
user_params['render3D_colors'] = render3D_colors
render3D_bgcolor = to_rgb(user_params.get('render3D_bgcolor', 'black'))
user_params['render3D_bgcolor'] = render3D_bgcolor
render3D_resolution = to_int(user_params.get('render3D_resolution', 1080))
user_params['render3D_resolution'] = render3D_resolution
# Debugging options
enable_Hubble = bool(user_params.get('enable_Hubble', True))
user_params['enable_Hubble'] = enable_Hubble
enable_class_background = bool(user_params.get('enable_class_background', True))
user_params['enable_class_background'] = enable_class_background
enable_Ewald = bool(user_params.get('enable_Ewald',  # !!! Only used by gravity_old.py
                                    True if 'pp' in list(itertools.chain.from_iterable(d.values() for d in select_forces.values())) else False))
user_params['enable_Ewald'] = enable_Ewald
enable_debugging = bool(user_params.get('enable_debugging', False))
user_params['enable_debugging'] = enable_debugging
# Additional, hidden parameters (parameters not supplied by the user)
special_params = dict(user_params.get('special_params', {}))
user_params['special_params'] = special_params
output_times_full = dict(user_params.get('output_times_full', {}))
user_params['output_times_full'] = output_times_full
initial_time_step = to_int(user_params.get('initial_time_step', 0))
user_params['initial_time_step'] = initial_time_step
Δt_begin_autosave = float(user_params.get('Δt_begin_autosave', -1))
user_params['Δt_begin_autosave'] = Δt_begin_autosave
Δt_autosave = float(user_params.get('Δt_autosave', -1))
user_params['Δt_autosave'] = Δt_autosave



#####################################################
# Global (cross-module) definitions and allocations #
#####################################################
# The ANSI ESC character used for ANSI/VT100 control sequences
cython.declare(ANSI_ESC=str)
ANSI_ESC = '\x1b'
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
# For char* fields however, we will use universals_dict
# instead of universals.
universals, universals_dict = build_struct(
    # Flag specifying whether any warnings have been given
    any_warnings=('bint', False),
    # Current scale factor and cosmic time
    a=('double', a_begin),
    t=('double', t_begin),
    # Initial time of simulation
    a_begin=('double', a_begin),
    t_begin=('double', t_begin),
    z_begin=('double', (ထ if a_begin == 0 else 1/a_begin - 1)),
    # Scale factor and cosmic time at next time step
    a_next='double',
    t_next='double',
    # Current time step
    time_step=('Py_ssize_t', 0),
    # '+'-separated strings of CO𝘕CEPT/CLASS species present in the simulation
    species_present='char*',
    class_species_present='char*',
)



############################################
# Derived and internally defined constants #
############################################
cython.declare(snapshot_dir=str,
               snapshot_base=str,
               snapshot_times=dict,
               powerspec_dir=str,
               powerspec_base=str,
               powerspec_times=dict,
               render2D_dir=str,
               render2D_base=str,
               render2D_times=dict,
               render3D_dir=str,
               render3D_base=str,
               render3D_times=dict,
               autosave_dir=str,
               ρ_crit='double',
               Ωm='double',
               ρ_mbar='double',
               slab_size_padding='ptrdiff_t',
               pm_fac_const='double',
               longrange_exponent_fac='double',
               p3m_cutoff_phys='double',
               p3m_scale_phys='double',
               )
# Output times not explicitly written as either of type 'a' or 't'
# is understood as being of type 'a' when Hubble expansion is enabled
# and of type 't' if it is disabled.
for time_param in ('a', 't'):
    output_times[time_param] = dict(output_times.get(time_param, {}))
    replace_ellipsis(output_times[time_param])
default_time_param = 'a' if enable_Hubble else 't'
for key, val in output_times.items():
    if key not in ('a', 't'):
        output_times[default_time_param][key] = tuple(
            any2list(output_times[default_time_param].get(key, [])) + any2list(val)
        )
for time_param in ('a', 't'):
    output_times[time_param] = dict(output_times.get(time_param, {}))
    for kind in ('snapshot', 'powerspec', 'render2D', 'render3D'):
        output_times[time_param][kind] = output_times[time_param].get(kind, ())
output_times = {
    time_param: {
        key: tuple(sorted(set([
            float(eval_unit(nr) if isinstance(nr, str) else nr)
            for nr in any2list(val)
            if nr or nr == 0])))
        for key, val in output_times[time_param].items()
    }
    for time_param in ('a', 't')
}
# Extract output variables from output dicts
snapshot_dir = output_dirs['snapshot']
snapshot_base = output_bases['snapshot']
snapshot_times = {
    time_param: output_times[time_param]['snapshot'] for time_param in ('a', 't')
}
powerspec_dir = output_dirs['powerspec']
powerspec_base = output_bases['powerspec']
powerspec_times = {
    time_param: output_times[time_param]['powerspec'] for time_param in ('a', 't')
}
render2D_dir = output_dirs['render2D']
render2D_base = output_bases['render2D']
render2D_times = {
    time_param: output_times[time_param]['render2D'] for time_param in ('a', 't')
}
render3D_dir = output_dirs['render3D']
render3D_base = output_bases['render3D']
render3D_times = {
    time_param: output_times[time_param]['render3D'] for time_param in ('a', 't')
}
autosave_dir = output_dirs['autosave']
# The average, comoing density (the critical
# comoving density since we only study flat universes).
ρ_crit = 3*H0**2/(8*π*G_Newton)
# The density parameter for all matter
Ωm = Ωcdm + Ωb
# The average, comoving matter density
ρ_mbar = Ωm*ρ_crit
# The real size of the padded (last) dimension of global slab grid
slab_size_padding = 2*(φ_gridsize//2 + 1)
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
pm_fac_const = G_Newton*boxsize**2/(π*φ_gridsize**3)  # ONLY USED BY gravity_old.py !!!
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
units_dict.setdefault(        'Ωcdm'          , Ωcdm                  )
units_dict.setdefault(unicode('Ωcdm')         , Ωcdm                  )
units_dict.setdefault(        'Ωb'            , Ωb                    )
units_dict.setdefault(unicode('Ωb')           , Ωb                    )
units_dict.setdefault(        'Ωm'            , Ωm                    )
units_dict.setdefault(unicode('Ωm')           , Ωm                    )
units_dict.setdefault(        'ρ_vacuum'      , ρ_vacuum              )
units_dict.setdefault(unicode('ρ_vacuum')     , ρ_vacuum              )
units_dict.setdefault(        'ρ_crit'        , ρ_crit                )
units_dict.setdefault(unicode('ρ_crit')       , ρ_crit                )
units_dict.setdefault(        'ρ_mbar'        , ρ_mbar                )
units_dict.setdefault(unicode('ρ_mbar')       , ρ_mbar                )
# Add dimensionless sizes
units_dict.setdefault('p3m_scale'          , p3m_scale          )
units_dict.setdefault('p3m_cutoff'         , p3m_cutoff         )
units_dict.setdefault('ewald_gridsize'     , ewald_gridsize     )
units_dict.setdefault('render3D_resolution', render3D_resolution)
units_dict.setdefault('slab_size_padding'  , slab_size_padding  )
units_dict.setdefault(        'φ_gridsize' , φ_gridsize         )
units_dict.setdefault(unicode('φ_gridsize'), φ_gridsize         )
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
# Overwrite the NumPy min and max function with NumPy and
# builtin hybrid min and max functions.
units_dict['min'] = produce_np_and_builtin_function('min')
units_dict['max'] = produce_np_and_builtin_function('max')
# Add everything from the math module
for key, val in vars(math).items():
    units_dict.setdefault(key, val)
# Add special functions
units_dict.setdefault('cbrt', lambda x: x**(1/3))



###############
# CLASS setup #
###############
# Update class_params with default values. This has already been done
# before for the version of class_params inside of user_params.
# Make sure that the same variables are updated here as there.
# Specification of general, default CLASS parameters
class_params_default = {'H0'       : H0/(units.km/(units.s*units.Mpc)),
                        'Omega_cdm': Ωcdm,
                        'Omega_b'  : Ωb,
                        }
# Add in neutrino CLASS parameters, if neutrinos are present
if int(class_params.get('N_ncdm', 0)) != 0:
    class_params_default.update({# Disable fluid approximation for non-CDM species
                                 'ncdm_fluid_approximation': 3,
                                 # Neutrino options needed for accurate δP/δρ
                                 'Quadrature strategy': 3,
                                 'evolver': 0,
                                 'Number of momentum bins': 25,
                                 'Maximum q': 15.0,
                                 'l_max_ncdm': 50,
                                 })
# Apply updates to the CLASS parameters
for param_name, param_value in class_params_default.items():
    class_params.setdefault(param_name, param_value)
# Function which turns a dict of items into a dict of str's
def stringify_dict(d):
    # Convert keys and values to str's
    with disable_numpy_summarization():
        d = {str(key): ', '.join([str(el) for el in any2list(val)]) for key, val in d.items()}
    # To ensure that the resultant str's are the same in pure Python
    # and compiled mode, all floats should have no more than 12 digits.
    d_modified = {}
    for key, val in d.items():
        try:
            f = ast.literal_eval(key)
            if isinstance(f, float):
                key = f'{f:.12g}'
        except:
            pass
        try:
            f = ast.literal_eval(val)
            if isinstance(f, float):
                val = f'{f:.12g}'
        except:
            pass
        d_modified[key] = val
    return d_modified
# Function that can call out to CLASS,
# correctly taking advantage of OpenMP and MPI.
@cython.pheader(# Arguments
                extra_params=dict,
                sleep_time='double',
                mode=str,
                # Locals
                compute_perturbations='bint',
                k_output_value=str,
                k_output_values=list,
                k_output_values_node=list,
                k_output_values_node_indices='Py_ssize_t[::1]',
                k_output_values_nodes=list,
                k_output_values_nodes_deque=object,  # collections.deque
                k_output_values_proc=list,
                k_output_values_procs=list,
                k_output_values_procs_deque=object,  # collections.deque
                method=str,
                n_modes='Py_ssize_t',
                n_surplus='Py_ssize_t',
                nprocs_node_i='int',
                params_specialized=dict,
                returns=object,  # classy.Class or (classy.Class, Py_ssize_t[::1])
                )
def call_class(extra_params=None, sleep_time=0.1, mode='single node'):
    """If mode == 'MPI' and 'k_output_values' is present in the
    CLASS parameters, these k values will be divided fairly among
    the nodes. Note that this means that each node master will store
    its own chunk of the perturbations.
    """
    if extra_params is None:
        extra_params = {}
    mode = mode.lower()
    if mode not in ('single node', 'mpi'):
        abort(
            f'call_class called with mode = "{mode}", '
            f'but only "single node" and "mpi" are allowed'
        )
    # Set the perturbations_verbose to some negative integer,
    # which is not a standard CLASS value but is used to signal
    # printout for the MPI class implementation.
    extra_params.setdefault('perturbations_verbose', -1)
    # Merge global and extra CLASS parameters
    params_specialized = class_params.copy()
    params_specialized.update(extra_params)
    # If non-cdm perturbations should be computed, the CLASS run
    # may take quite some time to finish. Class has been patched to
    # enable printout of status updates along the way. Flag whether
    # perturbations are to be computed.
    compute_perturbations = (
        'k_output_values' in params_specialized and 'output' in params_specialized
    )
    # Transform all CLASS container parameters to str's of
    # comma-separated values. All other CLASS parameters will also
    # be converted to their str representation.
    params_specialized = stringify_dict(params_specialized)
    # Fairly distribute the k modes among the nodes,
    # taking the number of processes in each node into account.
    if 'k_output_values' in params_specialized:
        k_output_values = params_specialized['k_output_values'].split(',')
        if k_output_values != sorted(k_output_values, key=float):
            masterwarn(
                'Unsorted k_output_values passed to call_class(). '
                'This may lead to unexpected behavior'
            )
    if mode == 'mpi':
        if 'k_output_values' not in params_specialized:
            abort('Cannot call CLASS in node mode when no k_output_values are given')
        n_modes = len(k_output_values)
        # Put the sorted k modes in a deque.
        # If the number of k modes cannot be evenly distributed
        # over the processes, skip the lowest few k modes for now.
        n_surplus = n_modes % nprocs
        k_output_values_procs_deque = collections.deque(k_output_values[n_surplus:])
        # Distribute the k values evenly over the number of processes.
        # To fairly distribute the workload per process, a given process
        # is first assigned a k mode from the large k end, then a k mode
        # from the low k end.
        k_output_values_procs = [[] for _ in range(nprocs)]
        while k_output_values_procs_deque:
            for method in ('pop', 'popleft'):
                for k_output_values_proc in k_output_values_procs:
                    if k_output_values_procs_deque:
                        k_output_values_proc.append(getattr(k_output_values_procs_deque, method)())
        # Include the skipped low k modes
        for k_output_value, k_output_values_proc in zip(k_output_values[:n_surplus],
                                                        reversed(k_output_values_procs),
                                                        ):
            k_output_values_proc.append(k_output_value)
        # Collect the process distributed k modes into node distributed
        # k modes, based on the number of processes in each node.
        # The processes with the larger assigned k modes will be
        # designated the nodes with the most processes.
        k_output_values_nodes_deque = collections.deque(k_output_values_procs)
        k_output_values_nodes = [[] for _ in range(nnodes)]
        while k_output_values_nodes_deque:
            for method in ('pop', 'popleft'):
                for nprocs_node_i, k_output_values_node in zip(sorted(nprocs_nodes, reverse=True), k_output_values_nodes):
                    if len(k_output_values_node) < nprocs_node_i:
                        k_output_values_node.append(getattr(k_output_values_nodes_deque, method)())
        k_output_values_nodes = [
            list(itertools.chain.from_iterable(k_output_values_node))
            for k_output_values_node in k_output_values_nodes
        ]
        # The k_output_values_nodes list now store one list of k modes
        # for each node, but the order is scrambled. This is due to
        # the reversed sorting of nprocs_nodes above. This reversed
        # sorting is undone here.
        k_output_values_nodes = [
            list(map(str, k_output_values_node_arr))
            for k_output_values_node_arr in
            asarray(list(reversed(k_output_values_nodes)))[np.argsort(np.argsort(nprocs_nodes))]
        ]
        # Sort the k modes within each node
        for k_output_values_node in k_output_values_nodes:
            k_output_values_node.sort(key=float)
        # Select the list of k modes designated this node
        # and insert it in the CLASS parameters.
        k_output_values_node = k_output_values_nodes[node]
        params_specialized['k_output_values'] = ','.join(k_output_values_node)
        # Indices of elements of k_output_values
        # which should be computed on this node.
        k_output_values_node_indices = asarray(
            [k_output_values.index(k_output_value) for k_output_value in k_output_values_node],
            dtype=C2np['Py_ssize_t'],
        )
    else:
        if 'k_output_values' in params_specialized:
            k_output_values_node_indices = arange(len(k_output_values), dtype=C2np['Py_ssize_t'])
    # Write out progress message. If perturbations will be computed,
    # the node masters will print out status updates from within the
    # Class C code. Thus we need to skip to the line below the progress
    # message itself in order not to mess up the first line of these
    # status updates.
    masterprint('Calling CLASS ...')
    if compute_perturbations:
        masterprint('\n', end='')
    # Instantiate a classy.Class instance and populate it with the
    # CLASS parameters. Feed the Class instance with information about
    # the local node (number) and current terminal indentation due to
    # progress printing, enabling it to write out nice status updates.
    # No line wrapping will be performed, but usually the progress print
    # indentation level is not deep enough at this point to make the
    # Class output exceed the terminal width.
    cosmo = Class(node=node, indentation=bcast(progressprint['indentation']))
    cosmo.set(params_specialized)
    # Call cosmo.compute in such a way as to allow
    # for OpenMP parallelization.
    Barrier()
    call_openmp_lib(cosmo.compute, sleep_time=sleep_time, mode=mode)
    Barrier()
    masterprint('done')
    # Always return the cosmo object. If perturbations have
    # been computed, also return the indices of the k_output_values
    # that have been computed on this node.
    if 'k_output_values' in params_specialized:
        return cosmo, k_output_values_node_indices
    else:
        return cosmo



#########################
# Pseudo-random numbers #
#########################
# From the random_seed, generate seeds individual to each process.
# The pseudo-random number generators on each proces will be seeded
# using these unique seeds. The master process will have a seed
# equal to the user parameer random_seed.
cython.declare(process_seed='unsigned long int')
process_seed = random_seed + rank
# Initialize the pseudo-random number generator and declare the
# functions random and random_gaussian, returning random numbers from
# the uniform distibution between 0 and 1 and a gaussian distribution
# with variable mean and spread, respectively.
# Both the pure Python and the compiled version of the functions use the
# Mersenne Twister algorithm to generate the random numbers.
# Despite of this, their exact implementations differ enough to make
# the generated sequence of random numbers completely different for
# pure Python and compiled runs.
if not cython.compiled:
    # In pure Python, use NumPy's random module
    def seed_rng(seed=process_seed):
        np.random.seed(seed)
    random = np.random.random
    random_gaussian = np.random.normal
else:
    # Use GSL in compiled mode
    cython.declare(random_number_generator='gsl_rng*')
    random_number_generator = gsl_rng_alloc(gsl_rng_mt19937)
    @cython.header(seed='unsigned long int')
    def seed_rng(seed=process_seed):
        gsl_rng_set(random_number_generator, seed)
    @cython.header(returns='double')
    def random():
        return gsl_rng_uniform_pos(random_number_generator)
    @cython.header(loc='double',
                   scale='double',
                   returns='double',
                   )
    def random_gaussian(loc=0, scale=1):
        return loc + gsl_ran_gaussian(random_number_generator, scale)
# Seed the pseudo-random number generator
seed_rng()



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

# Max function for pairs of numbers
@cython.header(a=number, b=number, returns=number)
def pairmax(a, b):
    if a > b:
        return a
    return b

# Min function for pairs of numbers
@cython.header(a=number, b=number, returns=number)
def pairmin(a, b):
    if a < b:
        return a
    return b

# Modulo function for numbers
if not cython.compiled:
    mod = np.mod
else:
    @cython.header(# Arguments
                   x=signed_number,
                   length=signed_number,
                   remainder_i='Py_ssize_t',
                   returns=signed_number,
                   )
    def mod(x, length):
        """This function computes the proper modulos, which (given a
        positive length) is always positive. Note that this is different
        from x%length in C, which results in the signed remainder.
        Note that mod(floating, integer) is not supported.
        """
        if signed_number in floating:
            remainder_f = fmod(x, length)
            if remainder_f == 0:
                return 0
            elif x < 0:
                return remainder_f + length
            return remainder_f
        else:
            remainder_i = x%length
            if remainder_i == 0:
                return 0
            elif x < 0:
                return remainder_i + length
            return remainder_i

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

# Mean function for 1D memory views of numbers
if not cython.compiled:
    # Use NumPy's mean function in pure Python
    mean = np.mean
else:
    """
    @cython.header(returns=number)
    def mean(number[::1] a):
        return sum(a)/a.shape[0]
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
                numbers=object,  # Single number or container of numbers
                nfigs='int',
                fmt=str,
                # Locals
                coefficient=str,
                exponent=str,
                n_missing_zeros='int',
                number=object,  # Single number of any type
                number_str=str,
                return_list=list,
                returns=object,  # String or list of strings
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
    for number in any2list(numbers):
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

# Function which searches a dictionary for a component
# or a set of components.
# Note that something goes wrong when trying to make this into
# a cdef function, so we leave it as a pure Python function.
def is_selected(component_or_components, d, accumulate=False):
    """This function searches for the given component in the given
    dict d. Both the component instance itself and its name, species
    and representation attributes are used, as well as the str's 'all'
    and 'default'.
    The precedence of these (lower takes precedence) are:
    - 'default'
    - 'all'
    - component.representation
    - component.species
    - component.name
    - component
    If multiple components are given, d is searched for an iterable
    containing all these components (and no more). Both the
    component instances themselves and their names, as well as the
    str's 'all combinations' and 'default' are used.
    The precedence of these (lower takes precedence) are:
    - 'default'
    - 'all combinations'
    - {component0.name, component1.name, ...}
    - {component0, component1, ...}
    All str's are compared case insensitively. If a str key is not found
    in d, a regular expression match of the entire str is attempted.
    If the component is found in d, its value in d is returned.
    Otherwise, None is returned.
    If accumulate is True, the above precedence will not be used.
    Rather, every match will be stored and returned. In this case,
    the returned value will be a list of matched values. If all these
    values happens to be dicts, they will be combined into a single
    dict which is then returned.
    """
    # Determine whether a single or multiple components are passed
    component_or_components = any2list(component_or_components)
    if len(component_or_components) == 1:
        component = component_or_components[0]
        keys = (
            'default',
            'all',
            component.representation.lower(),
            component.species.lower(),
            component.name.lower(),
            component,
            )
    else:
        components = frozenset(component_or_components)
        names = frozenset([component.name.lower() for component in components])
        keys = (
            'default',
            'all combinations',
            names,
            components,
            )
    # Ensure lowercase on all str keys
    # and transform otherwise iterable keys to sets.
    d_transformed = {}
    for key, val in d.items():
        if isinstance(key, str):
            key = key.lower()
        else:
            key = any2list(key)
            if len(key) == 1:
                key = key.pop()
            elif len(key) == 0:
                continue
            else:
                key = frozenset(key)
        d_transformed[key] = val
    d = d_transformed
    # Do the lookup
    selected = []
    for key in keys:
        val = d.get(key)
        if val is None and isinstance(key, str):
            # Maybe the key is a regular expression
            for d_key, d_val in d.items():
                if isinstance(d_key, str) and re.compile(d_key).fullmatch(key):
                    val = d_val
                    break
        if val is not None:
            selected.append(val)
    # Return either the last (highest precedence) selection or a
    # list/dict of all selections, based on the value of accumulate.
    if accumulate:
        if all(isinstance(val, dict) for val in selected):
            selected_dict = {}
            for val in selected:
                selected_dict.update(val)
            return selected_dict
        else:
            return selected
    else:
        if selected:
            return selected[-1]
        else:
            return None

# Context manager which suppresses all output to stdout
@contextlib.contextmanager
def suppress_stdout(f=sys.stdout):
    with open(os.devnull, 'w') as devnull:
        # Point sys.stdout to /dev/null
        sys.stdout = devnull
        try:
            # Yield control back to the caller
            yield
        finally:
            # Cleanup: Reset sys.stdout
            sys.stdout = f

# Context manager which disables summarization of NumPy arrays
# (the ellipses appearing in str representations of large arrays).
@contextlib.contextmanager
def disable_numpy_summarization():
    # Backup the current print threshold
    threshold = np.get_printoptions()['threshold']
    # Set the threshold to infinity so that the str representation
    # of arrays wil not contain any ellipses
    # (full printout rather than summarization).
    np.set_printoptions(threshold=ထ)
    try:
        # Yield control back to the caller
        yield
    finally:
        # Cleanup: Reset print options
        np.set_printoptions(threshold=threshold)

# Function used to set the minor tick formatting in log plots
def fix_minor_tick_labels(ax=None):
    """In matplotlib 2.x, tick labels are automatically placed at minor
    ticks in log plots if the axis range is less than a full decade.
    Here, a '\times' glyph is used, which is not handled correctly
    due to it being inclosed in \mathdefault{}, leading to a
    MathTextWarning and the '\times' being replaced with a dummy symbol.
    The problem is fully described here:
    https://stackoverflow.com/questions/47253462
    /matplotlib-2-mathtext-glyph-errors-in-tick-labels
    This function serves as a workaround. To avoid the warning,
    you must call this function before calling tight_layout().
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    # Force the figure to be drawn,
    # ignoring the warning about \times not being recognized.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=matplotlib.mathtext.MathTextWarning)
        fig.canvas.draw()
    # Remove '\mathdefault' from all minor tick labels
    labels = [label.get_text().replace(r'\mathdefault', '')
              for label in ax.get_xminorticklabels()]
    ax.set_xticklabels(labels, minor=True)
    labels = [label.get_text().replace(r'\mathdefault', '')
              for label in ax.get_yminorticklabels()]
    ax.set_yticklabels(labels, minor=True)

# Function taking in some iterable of integers
# and returning a nice, short str representation.
def get_integerset_strrep(integers):
    intervals = []
    for key, group in itertools.groupby(enumerate(sorted(set(any2list(integers)))),
                                  lambda t: t[0] - t[1]
                                  ):
        interval = [t[1] for t in group]
        if len(interval) < 3:
            intervals += interval
        else:
            intervals.append(f'{interval[0]}–{interval[-1]}')
    str_rep = ', '.join(map(str, intervals))
    return str_rep

# Function which should be used when opening hdf5 files
def open_hdf5(filename, **kwargs):
    """This function is equivalent to just doing
    h5py.File(filename, *args, **kwargs)
    except that it will not throw an exception if the file is
    temporarily unavailable, which happens when multiple processes
    attempts to open the same file in write mode. When this is the case,
    this function waits (possibly indefinitely) for the file to
    become available.
    The function supports both collective and non-collective calls.
    It is an error to call non-collectively from any process but the
    master mode.
    """
    # Minimum and maximum time to wait between checks on the file
    sleep_time_min = 1
    sleep_time_max = 300
    # Determine if this is a collective call or not
    collective = (kwargs.get('driver') == 'mpio')
    if not collective and not master:
        abort(
            f'A non-collective call to open_hdf5() was performed on process {rank}, '
            f'which is not the master'
        )
    # Let the master check if the file is available for opening
    # in the mode given by **kwargs.
    if master:
        # As this check is done by the master only,
        # we must not open it using a collective driver.
        kwargs_noncollective = kwargs.copy()
        if collective:
            for kwarg in ('driver', 'comm'):
                if kwarg in kwargs_noncollective:
                    del kwargs_noncollective[kwarg]
        sleep_time = sleep_time_min
        while True:
            try:
                h5py.File(filename, **kwargs_noncollective).close()
                break
            except OSError as e:
                if 'File exists' not in str(e):
                    raise e
            if sleep_time == sleep_time_min:
                masterprint(
                    f'File "{filename}" is temporarily unavailable, '
                    f'possibly because it is already opened in write mode by another process. '
                    f'Waiting for the file to become available ...'
                )
            sleep(sleep_time)
            sleep_time = np.min([2*sleep_time, sleep_time_max])
        if sleep_time > sleep_time_min:
            masterprint('done')
    # Let the slaves wait here for collective calls
    if collective:
        Barrier()
    # The file was very recently available
    try:
        hdf5_file = h5py.File(filename, **kwargs)
    except OSError:
        # We did not make it. Try again.
        return open_hdf5(filename, **kwargs)
    return hdf5_file



##############################################################
# Sanity checks and corrections/additions to user parameters #
##############################################################
# Abort on unrecognized snapshot_type
if snapshot_type not in ('standard', 'gadget2'):
    abort('Does not recognize snapshot type "{}"'.format(user_params['snapshot_type']))
# Abort on illegal FFTW rigor
if fftw_wisdom_rigor not in ('estimate', 'measure', 'patient', 'exhaustive'):
    abort('Does not recognize FFTW rigor "{}"'.format(user_params['fftw_wisdom_rigor']))
# Warn if random_seed is chosen to be 0, as this may lead to clashes
# with the default seed used by GSL.
if random_seed < 1:
    masterwarn(
        f'A random_seed of {random_seed} was specified. '
        f'This should be > 0 to avoid clashes with the default GSL seed.'
    )
# Warn about unused but specified parameters.
if user_params.unused:
    if len(user_params.unused) == 1:
        msg = 'The following unknown parameter was specified:\n'
    else:
        msg = 'The following unknown parameters were specified:\n'
    masterwarn(msg + '\n'.join(user_params.unused))
# Output times very close to t_begin or a_begin
# are probably meant to be exactly at t_begin or a_begin
for time_param in ('t', 'a'):
    output_times[time_param] = {key: tuple([a_begin if isclose(float(nr), a_begin) else nr
                                            for nr in val])
                                for key, val in output_times[time_param].items()}
# Output times very close to a = 1
# are probably meant to be exactly at a = 1.
output_times['a'] = {key: tuple([1 if isclose(float(nr), 1) else nr
                                 for nr in val])
                     for key, val in output_times['a'].items()}
# Warn about output times being greater than a = 1
for output_kind, times in output_times['a'].items():
    if any([a > 1 for a in times]):
        masterwarn('{} output is requested at a = {}, '
                   'but the simulation will not continue after a = 1.'
                   .format(output_kind.capitalize(), np.max(times)))
# Reassign output times of the individual types
snapshot_times = {
    time_param: output_times[time_param]['snapshot'] for time_param in ('a', 't')
}
powerspec_times = {
    time_param: output_times[time_param]['powerspec'] for time_param in ('a', 't')
}
render2D_times = {
    time_param: output_times[time_param]['render2D'] for time_param in ('a', 't')
}
render3D_times = {
    time_param: output_times[time_param]['render3D'] for time_param in ('a', 't')
}
# If no output_times_full has been given (the normal case),
# set this equal to output_times.
if not output_times_full:
    output_times_full = output_times
# Warn about cosmological autosave interval
if autosave_interval > 1*units.yr:
    masterwarn(f'Autosaving will take place every {autosave_interval} {unit_time}. '
               f'Have you forgotten to specify the unit of the "autosave_interval" parameter?'
               )
if autosave_interval < 0:
    autosave_interval = 0
# Abort on negative a_begin
if a_begin <= 0:
    abort(
        f'Beginning of simulation set to a = {a_begin}, but 0 < a is required'
    )
# Abort on negative t_begin when running a cosmological simulation.
# Even in a non-cosmological context, negative t_begin might
# cause trouble, so we print a warning here.
if t_begin < 0:
    if enable_Hubble:
        abort(
            f'Cannot start the cosmological simulation at t = {t_begin} {unit_time} '
            f'as this is prior to the Big Bang at t = 0. '
            f'To run a non-cosmological simulation, set enable_Hubble to False.'
        )
    else:
        masterwarn(
            f'The simulation start at t = {t_begin} {unit_time} < 0. '
            f'Negative times might lead to unexpected behavior.'
        )
# Allow for easier names in class_extra_background
if any(D1_name in class_extra_background for D1_name in {'D', 'D1'}):
    class_extra_background.add('gr.fac. D')
if any(f1_name in class_extra_background for f1_name in {'f', 'f1'}):
    class_extra_background.add('gr.fac. f')
if any(τ_name in class_extra_background for τ_name in {unicode('τ'), asciify('τ'), 'tau'}):
    class_extra_background.add('conf. time')

###########################################################
# Functionality for "from commons import *" when compiled #
###########################################################
# Function which floods the module namespace of the caller with
# variables from the uncompiled version of this module.
# This is effectively equivalent to "from commons import *",
# except that the uncompiled version is guaranteed to be used.
def commons_flood():
    if cython.compiled:
        with suppress_stdout():
            commons_module = pyxpp.load_source(
                'commons_pure_python',
                '{}/commons.py'.format(paths['concept_dir']),
            )
        stack = inspect.stack()
        if len(stack) == 1:
            frame = stack[0].frame
        else:
            frame = stack[1].frame
        try:
            inspect.getmodule(frame).__dict__.update(commons_module.__dict__)
        except:
            pass
        try:
            frame.f_locals.update(commons_module.__dict__)
        except:
            pass
    else:
        # Running in pure Python mode.
        # It is assumed that "from commons import *" has already
        # been run, leaving nothing to do.
        pass



###############################
# Print out setup information #
###############################
# Only print out setup information if this is an actual simulation run.
# In this case, a jobid different from -1 is set.
if jobid != -1:
    # Print out MPI process mapping
    if master:
        masterprint('MPI layout:')
        node_name_max_length = np.max(
            [len(other_node_name) for other_node_name in node_names2numbers.keys()]
        )
        for other_node in range(nnodes):
            other_node_name = node_numbers2names[other_node]
            spaces = ' '*(node_name_max_length - len(other_node_name))
            other_ranks = np.where(asarray(nodes) == other_node)[0]
            masterprint(
                f'Node {other_node} ({other_node_name}):{spaces}',
                ('Process' if nnodes == 1 else 'Process  ')
                if len(other_ranks) == 1 else 'Processes',
                get_integerset_strrep(other_ranks),
                indent=4,
                )
    # Print out proper inferred variables
    # (those which did not just get their default value).
    inferred_params_units = DictWithCounter(
        {
            'Ων': '',
        }
    )
    inferred_any = False
    for key, val in inferred_params_final.items():
        if not inferred_params_set[key] and val != inferred_params[key]:
            val_str = significant_figures(val, 4, fmt='unicode')
            unit_str = inferred_params_units.get(key, '')
            if unit_str:
                unit_str = ' ' + unit_str
            if not inferred_any:
                masterprint('Inferred:')
                inferred_any = True
            masterprint(f'{key} = {val_str}{unit_str}', indent=4)



# Let all processes catch up, ensuring the correct ordering of the
# printed messages above.
Barrier()
