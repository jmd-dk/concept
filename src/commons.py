# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2024 Jeppe Mosgaard Dakin.
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



# This module contains imports, Cython declarations and values
# of parameters common to all other modules. Each module should have
# 'from commons import *' as its first statement.



#######################
# The CO𝘕CEPT version #
#######################
__version__ = 'master'



############################################
# Imports common to pure Python and Cython #
############################################
# Miscellaneous
import ast, collections, contextlib, ctypes, cython, functools, hashlib
import importlib, inspect, itertools, keyword, logging, operator, os, re
import shutil, struct, sys, textwrap, traceback, types
import unicodedata, warnings
from copy import deepcopy
# Numerics
# (note that numpy.array is purposely not imported directly into the
# global namespace, as this does not play well with Cython).
import decimal
import numpy as np
from numpy import arange, asarray, empty, linspace, logspace, ones, zeros
# MPI
import mpi4py.rc; mpi4py.rc.threads = False  # Do not use threads
from mpi4py import MPI
# I/O
import io
from glob import glob
# For fancy terminal output
import blessings
# For timing
from time import sleep, time
import datetime



##########################
# Command-line arguments #
##########################
# Handle command-line arguments given to the Python interpreter
# (not those explicitly given to the concept script).
cython.declare(
    param=str,
    jobid='Py_ssize_t',
)
# Construct a dict from command-line arguments of the form
# "param='value'"
argd = {}
for arg in sys.argv[1:]:
    if '=' not in arg:
        continue
    with contextlib.suppress(Exception):
        exec(arg, None, argd)
# The original (i.e. not the copy) parameter file
param = argd.get('param', '')
if param == 'None':
    param = ''
# The job ID of the current run
jobid = int(argd.get('jobid', -1))



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
# C data type names to NumPy dtype names
cython.declare(C2np=dict)
C2np = {
    # Booleans
    'bint': np.bool_,
    # Signed integers
    'signed char'  : np.byte,
    'short'        : np.short,
    'int'          : np.intc,
    'long int'     : np.int_,
    'long long int': np.longlong,
    'ptrdiff_t'    : np.intp,
    'Py_ssize_t'   : np.intp,
    # Unsigned integers
    'unsigned char'         : np.ubyte,
    'unsigned short'        : np.ushort,
    'unsigned int'          : np.uintc,
    'unsigned long int'     : np.uint,
    'unsigned long long int': np.ulonglong,
    'size_t'                : np.uintp,
    # Floating-point numbers
    'float'     : np.float32,
    'double'    : np.float64,
    'long float': np.float128,
}
# Format characters to C data type names
cython.declare(fmt2C=dict)
fmt2C = {
    # Booleans
    '?': 'bint',
    # Signed integers
    's': 'signed char',
    'h': 'short',
    'i': 'int',
    'l': 'long int',
    'q': 'long long int',
    # Unsigned integers
    'B': 'unsigned char',
    'H': 'unsigned short',
    'I': 'unsigned int',
    'L': 'unsigned long int',
    'Q': 'unsigned long long int',
    'N': 'size_t',
    # Floating-point numbers
    'f': 'float',
    'd': 'double',
}
# Sizes (in bytes) of C data types
cython.declare(sizesC=dict)
sizesC = {fmt: struct.calcsize(fmt) for fmt in fmt2C}



#############
# MPI setup #
#############
cython.declare(
    master='bint',
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
# For newer versions of NumPy, we have to pass the dtype of the arrays
# explicitly when using upper-case communication methods.
def buf_and_dtype(buf):
    try:
        arr = asarray(buf)
        if arr.shape:
            return (buf, arr.dtype.char)
    except Exception:
        pass
    return buf
Allgather = lambda sendbuf, recvbuf: comm.Allgather(
    buf_and_dtype(sendbuf), recvbuf)
Allgatherv = lambda sendbuf, recvbuf: comm.Allgatherv(
    buf_and_dtype(sendbuf), recvbuf)
Allreduce = lambda sendbuf, recvbuf, op=MPI.SUM: comm.Allreduce(
    buf_and_dtype(sendbuf), recvbuf, op)
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
bcast      = lambda obj=None, root=master_rank: comm.bcast(obj, root)
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
    cython.declare(rank_other='int')
    for rank_other, other_node_name in enumerate(node_names):
        if other_node_name not in node_names2numbers:
            node_i += 1
            if node_i == master_node:
                node_i += 1
            node_names2numbers[other_node_name] = node_i
        nodes[rank_other] = node_names2numbers[other_node_name]
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
        sleep_time = any2list(sleep_time)
        if len(sleep_time) == 1:
            sleep_time = sleep_time_max = sleep_time[0]
        elif len(sleep_time) == 2:
            sleep_time, sleep_time_max = sleep_time
        else:
            abort(f'sleeping_barrier called with sleep_time = {sleep_time}')
        while not iprobe(source=source):
            sleep(sleep_time)
            sleep_time = np.min([2*sleep_time, sleep_time_max])
        # Remember to receive the message
        recv(source=source)
# Function that can call another function that uses OpenMP.
# The (global or node) master process is the only one that actually does
# the call, while slave processes periodically asks whether their
# master is done so that they may continue. This period is controlled
# by sleep_time, given in seconds. While sleeping, the slave processes
# can be utilized to carry out OpenMP work by their master.
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



####################
# Print MPI layout #
####################
# Function for converting a non-iterable to lists of one element and
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

# Function taking in some iterable of integers
# and returning a nice, short str representation.
def get_integerset_strrep(integers):
    intervals = []
    for key, group in itertools.groupby(
        enumerate(sorted(set(any2list(integers)))),
        lambda t: t[0] - t[1]
    ):
        interval = [t[1] for t in group]
        if len(interval) < 3:
            intervals += interval
        else:
            # We wish to use the en dash character '–'.
            # Normally we would just write it as a str literal and wrap
            # it in a call to unicode(), however this function will not
            # be defined before further down.
            en_dash = unicodedata.lookup('EN DASH')
            intervals.append(f'{interval[0]}{en_dash}{interval[-1]}')
    return ', '.join(map(str, intervals))

# Function that aligns a list of str's by inserting spaces.
# The alignment points are specified by the 'alignat' character.
def align_text(lines, alignat='$', indent=0, rstrip=True, handle_numbers=True):
    if not lines:
        return lines
    lines = [line.split(alignat) for line in map(str, any2list(lines))]
    # Compute max length of each part
    n_parts_max = np.max(list(map(len, lines)))
    parts_len_max = [0]*n_parts_max
    for parts in lines:
        for i, part in enumerate(parts):
            size = len(part)
            if parts_len_max[i] < size:
                parts_len_max[i] = size
    # Add spacing to the parts to ensure alignment
    for parts in lines:
        for i, (part, part_len_max) in enumerate(zip(parts, parts_len_max)):
            space = ' '*(part_len_max - len(part))
            part += space
            parts[i] = part
    # Align numbers at the decimal point
    strip = ' ,:%\n'
    if handle_numbers:
        for i in range(n_parts_max):
            # Only do number-alignment
            # if all rows in the column are numbers.
            all_are_nums = False
            for j, parts in enumerate(lines):
                if len(parts) <= i:
                    continue
                part = parts[i]
                num = part.rstrip(strip)
                if '×' in num:
                    num = num[:num.index('×')]
                try:
                    float(num)
                except Exception:
                    all_are_nums = False
                    break
                all_are_nums = True
            if not all_are_nums:
                continue
            # All rows in this column are numbers
            dot_locations = {}
            for j, parts in enumerate(lines):
                if len(parts) <= i:
                    continue
                part = parts[i]
                num = part.rstrip(strip)
                if '×' in num:
                    num = num[:num.index('×')]
                try:
                    float(num)
                except Exception:
                    continue
                # This part is a number
                if '.' in part:
                    dot_location = part.index('.')
                else:
                    dot_location = len(part.rstrip(strip))
                dot_locations[j] = dot_location
            if not dot_locations:
                continue
            dot_location_rightmost = np.max(list(dot_locations.values()))
            for j, dot_location in dot_locations.items():
                parts = lines[j]
                part = parts[i]
                lspacing = dot_location_rightmost - dot_location
                rspacing_max = len(part) - len(part.rstrip()) - 1
                rspacing = np.min([lspacing, rspacing_max])
                part = ' '*lspacing + part[:len(part)-rspacing]
                parts[i] = f'{part}{alignat}'
    # Join aligned parts together into lines
    for i, parts in enumerate(lines):
        line = ''.join(parts).rstrip(alignat)
        lines[i] = line
    # If new alignment characters have been placed by
    # number handling, take care of these by calling
    # this function once more.
    for line in lines:
        if alignat in line:
            lines = align_text(lines, handle_numbers=False)
            break
    # Optional finishing touches
    if rstrip:
        lines = [line.rstrip() for line in lines]
    if indent:
        indentation = ' '*indent
        lines = [f'{indentation}{line}' for line in lines]
    return lines

# Print out MPI process mapping
if master and jobid != -1:
    lines = []
    for other_node in range(nnodes):
        other_node_name = node_numbers2names[other_node]
        other_ranks = np.where(asarray(nodes) == other_node)[0]
        lines.append(''.join([
            f'    Node ${other_node} $({other_node_name}): ',
            '$Process ' if len(other_ranks) == 1 else '$Processes ',
            '$', get_integerset_strrep(other_ranks),
        ]))
    print('MPI layout:')
    print('\n'.join(align_text(lines)), flush=True)



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
# needed for reading and writing Unicode characters in headers of text
# files using np.loadtxt and np.savetxt.
asbytes = lambda s: s if isinstance(s, bytes) else str(s).encode('utf-8')
asstr = lambda s: s.decode('utf-8') if isinstance(s, bytes) else str(s)
np.compat.py3k .asbytes   = asbytes
np.compat.py3k .asstr     = asstr
np.compat.py3k .asunicode = asstr
np.lib   .npyio.asbytes   = asbytes
np.lib   .npyio.asstr     = asstr
np.lib   .npyio.asunicode = asstr



############
# Plotting #
############
# We use Matplotlib for various plotting tasks.
# To avoid importing Matplotlib unnecessarily, we do not simply import
# it here in the global scope. Instead, the below function will do the
# import and return the Matplotlib module object. This function should
# then be called whenever Matplotlib is needed.
# In addition to importing Matplotlib and its pyplot interface,
# the function additionally:
#   - Ensures that we use the agg back-end.
#   - Sets some custom preferences.
#   - Ensure that I/O exceptions trigger abort
#   - Patches bugs.
def get_matplotlib():
    if matplotlib_cache:
        return matplotlib_cache[0]
    # Use a Matplotlib back-end that does not require a running X-server
    import matplotlib
    matplotlib.use('agg')
    # Now import the pyplot interface
    import matplotlib.pyplot as plt
    # Customize Matplotlib
    matplotlib.rcParams.update({
        # Use a nice font that ships with Matplotlib
        'text.usetex'                : False,
        'font.family'                : 'serif',
        'font.serif'                 : 'cmr10',
        'mathtext.fontset'           : 'cm',
        'axes.formatter.use_mathtext': True,
        'axes.unicode_minus'         : False,
        # Use outward pointing ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })
    # Monkey patch I/O functions so that exceptions trigger abort
    wrap_method(
        [plt, matplotlib.figure.Figure],
        ['imread', 'imsave', 'savefig'],
        tryexcept_wrapper,
        '{obj}.{func}() failed',
    )
    # Patch a bug about automatic minor tick labels by monkey patching
    # tight_layout() and savefig().
    # The bug is reported here:
    #   https://github.com/matplotlib/matplotlib/issues/10029
    import matplotlib.mathtext
    def fix_minor_tick_labels(fig=None):
        if fig is None:
            fig = plt.gcf()
        # Force the figure to be drawn, ignoring the warning about
        # \times not being recognised.
        logger = logging.getLogger('matplotlib.mathtext')
        original_level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR)
        fig.canvas.draw()
        logger.setLevel(original_level)
        # Remove \mathdefault from all minor tick labels
        for ax in fig.axes:
            for xy in 'xy':
                labels = [
                    label.get_text().replace(r'\mathdefault', '')
                    for label in getattr(ax, f'get_{xy}minorticklabels')()
                ]
                getattr(ax, f'set_{xy}ticklabels')(labels, minor=True)
    def fix_minor_tick_labels_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = None
            if len(args) > 0 and isinstance(args[0], matplotlib.figure.Figure):
                fig = args[0]
            # Matplotlib might emit several differet UserWarning's,
            # which we do not want to display.
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                fix_minor_tick_labels(fig)
                return func(*args, **kwargs)
        return wrapper
    wrap_method(
        [plt, matplotlib.figure.Figure],
        ['tight_layout', 'savefig'],
        fix_minor_tick_labels_decorator,
    )
    # On some systems the axis bounding box may some times become
    # infinite or NaN (for whatever reason), which ultimately results in
    # a ValueError. Here we detect this prior to calling draw() and
    # replaces the non-finite values with some default values,
    # avoiding the error. Note that these default values will not be
    # optimal in general, and so a warning will be printed.
    def ensure_finite_bbox(fig=None):
        if fig is None:
            fig = plt.gcf()
        warned = False
        defaults = {'x0': 0.0, 'y0': 0.0, 'x1': 1.0, 'y1': 1.0}
        for ax in fig.axes:
            try:
                bbox = ax.transAxes._boxout._bbox
            except Exception:
                continue
            for attr, default in defaults.items():
                val = getattr(bbox, attr, 0)
                if not np.isfinite(val):
                    if not warned:
                        warn('Non-finite bounding box within Matplotlib:', bbox)
                        warned = True
                    setattr(bbox, attr, default)
    def ensure_finite_bbox_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = None
            if len(args) > 0 and isinstance(args[0], matplotlib.figure.Figure):
                fig = args[0]
            ensure_finite_bbox(fig)
            return func(*args, **kwargs)
        return wrapper
    wrap_method(
        [plt, matplotlib.figure.Figure],
        ['draw'],
        ensure_finite_bbox_decorator,
    )
    # Suppress RuntimeWarning's from triangulations
    def suppress_runtime_warning(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(action='ignore', category=RuntimeWarning):
                return func(*args, **kwargs)
        return wrapper
    wrap_method(
        [plt.Axes, matplotlib.tri],
        ['tricontour', 'tricontourf', 'tripcolor', 'triplot'],
        suppress_runtime_warning,
    )
    # Add the matplotlib module to the global store for future lookups
    matplotlib_cache.append(matplotlib)
    # Return the matplotlib module itself
    return matplotlib
# Global store for the matplotlib module
matplotlib_cache = []
# Function wrapper for enabling MPI abort() on error. This is useful as
# Cython often ignores errors (error messages are produced but the
# program continues in an erroneous state).
def tryexcept_wrapper(func, abort_msg=''):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        val = None
        try:
            val = func(*args, **kwargs)
        except BaseException:
            traceback.print_exc()
            abort(abort_msg)
        return val
    return wrapper
# Function for applying wrappers to methods of objects
def wrap_method(objects, func_names, wrapper, *args):
    for obj in any2list(objects):
        for func_name in any2list(func_names):
            func = getattr(obj, func_name, None)
            if func is None:
                continue
            extra_args = [
                arg.format(
                    obj=str(getattr(obj, '__name__', obj)),
                    func=str(getattr(func, '__name__', func)),
                )
                if isinstance(arg, str) else arg
                for arg in args
            ]
            setattr(obj, func_name, wrapper(func, *extra_args))



#############################
# Print and abort functions #
#############################
# ANSI/VT100 escape sequences (when blessings are not enough)
cython.declare(
    esc=str,
    esc_normal=str,
    esc_italic=str,
    esc_no_italic=str,
    esc_grey=str,
    esc_background=str,
    esc_concept=str,
    esc_set_color=str,
)
esc = '\x1b'
esc_normal     = f'{esc}[0m'
esc_italic     = f'{esc}[3m'
esc_no_italic  = f'{esc}[23m'
esc_grey       = f'{esc}[37m'
esc_background = f'{esc}[48;5;{{}}m'
esc_set_color  = f'{esc}]4;{{}};rgb:{{}}/{{}}/{{}}{esc}\\'
esc_concept    = f'CO{esc_italic}N{esc_no_italic}CEPT'
# Function which takes in a previously saved time as input
# and returns the time which has elapsed since then, nicely formatted.
def time_since(initial_time):
    """The argument should be in the UNIX time stamp format,
    e.g. what is returned by the time function.
    """
    # Time since initial_time, in seconds
    seconds = time() - initial_time
    # Delegate the formatting work
    return nice_time_interval(seconds)
# Function for formatting a time in seconds as a nice string
def nice_time_interval(seconds):
    # Construct the time interval with a sensible amount of
    # significant figures.
    milliseconds = 0
    if seconds >= 1e-3:
        # More than a millisecond; use whole milliseconds
        milliseconds = round(1e+3*seconds)
    interval = datetime.timedelta(milliseconds=milliseconds)
    total_seconds = interval.total_seconds()
    if total_seconds >= 60:
        # More than a minute; use whole seconds
        seconds = round(1e-3*milliseconds)
        interval = datetime.timedelta(seconds=seconds)
    elif total_seconds >= 10:
        # More than 10 seconds; use whole deciseconds
        seconds = 1e-1*round(1e-2*milliseconds)
        interval = datetime.timedelta(seconds=seconds)
    elif total_seconds >= 1:
        # More than a second; use whole centiseconds
        seconds = 1e-2*round(1e-1*milliseconds)
        interval = datetime.timedelta(seconds=seconds)
    # Return a fitting string representation of the interval
    total_seconds = interval.total_seconds()
    if total_seconds == 0:
        return '< 1 ms'
    if total_seconds < 1:
        return f'{int(1e+3*total_seconds)} ms'
    if total_seconds < 10:
        return f'{total_seconds:.2f} s'
    if total_seconds < 60:
        return f'{total_seconds} s'
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
    bullet='',
    do_print=True,
    **kwargs,
):
    if bullet:
        bullet = unicode(bullet)
    # If called without any arguments, print the empty string
    if not args:
        args = ('', )
    args = tuple([str(arg) for arg in args])
    # Handle indentation due to non-finished progress print.
    # If indent == -1, no indentation should be used, not even that
    # due to the non-finished progress-print.
    is_done_message = (args[0] == 'done')
    leftover_indentation = progressprint['indentation']
    if leftover_indentation:
        if indent != -1:
            indent += progressprint['indentation']
        if ensure_newline_after_ellipsis and progressprint['previous_print_ends_in_ellipsis']:
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
        text = ' {{}}({}){{}}'.format(time_since(progressprint['time'].pop()))
        text_length = len(text) - 4
        if enable_terminal_formatting:
            text = text.format(esc_grey, esc_normal)
        else:
            text = text.format('', '')
        if len(args) > N_args_usual:
            text += sep + sep.join([str(arg) for arg in args[N_args_usual:]])
        # Convert to proper Unicode characters
        text = unicode(text)
        # The progressprint['maxintervallength'] variable stores the
        # length of the longest interval-message so far.
        if text_length > progressprint['maxintervallength']:
            progressprint['maxintervallength'] = text_length
        # Prepend the text with whitespace so that all future
        # interval-messages line up to the right.
        text = ' '*(
            + terminal_width
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
        progressprint['previous_print_ends_in_ellipsis'] = False
    else:
        # Stitch text pieces together
        text = sep.join([str(arg) for arg in args])
        # Convert to proper Unicode characters
        text = unicode(text)
        # Convert any paths in text (recognised by surrounding quotes)
        # to sensible paths.
        text = re.sub(r'"(.+?)"', lambda m: '"{}"'.format(sensible_path(m.group(1))), text)
        # Add indentation, and also wrap long message if wrap is True
        indentation = ' '*indent
        is_progress_message = text.endswith('...')
        progressprint['previous_print_ends_in_ellipsis'] = is_progress_message
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
            if bullet:
                indent += len(bullet) + 1
            indentation = ' '*indent
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
            if bullet:
                lines[0] = ' '*(indent - (len(bullet) + 1)) + f'{bullet} ' + lines[0].lstrip()
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
            # for the upcoming time delta.
            if is_progress_message:
                maxlength = terminal_width - progressprint['maxintervallength'] - 1
                # Separate the last line from the rest
                last_line = lines.pop().lstrip()
                # The trailing ... should never stand on its own
                if last_line == '...':
                    last_line = lines.pop().lstrip() + ' ...'
                # Replace spaces before ... with underscores
                last_line = re.sub(
                    r'( +)\.\.\.$',
                    lambda m: '_'*len(m.group(1)) + '...',
                    last_line,
                )
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
                lines[-1] = re.sub(
                    r'(_+)\.\.\.$',
                    lambda m: ' '*len(m.group(1)) + '...',
                    lines[-1],
                )
                progressprint['length'] = len(lines[-1])
            text = '\n'.join(lines)
        else:
            # Do not wrap the text into multiple lines,
            # regardless of the length of the text.
            # Add bullet.
            if bullet:
                text = f'{bullet} ' + text
            # Add indentation.
            text = indentation + text
            # If the text ends with '...', it is the start of a
            # progress message. In that case, the text should
            # have some left over space to the right
            # for the upcoming time delta.
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
        if do_print:
            print(text, flush=True, end='', **kwargs)
        else:
            return text
progressprint = {
    'maxintervallength'              : len(' (??? ms)'),
    'time'                           : [],
    'indentation'                    : 0,
    'previous'                       : '',
    'previous_print_ends_in_ellipsis': False,
}
# As the suppress_output, terminal_width and enable_terminal_formatting
# user parameters are used in fancyprint, they need to be defined before
# they are actually read in as parameters.
cython.declare(suppress_output=dict, terminal_width='int', enable_terminal_formatting='bint')
suppress_output = {'out': set(), 'err': set()}
terminal_width = 80
enable_terminal_formatting = True

# Functions for printing warnings
def warn(*args, skipline=True, prefix='Warning', wrap=True, **kwargs):
    try:
        universals.any_warnings = True
    except Exception:
        pass
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

# Raised exceptions inside cdef functions do not generally propagate
# out to the caller. In places where exceptions are normally raised
# manually, call this function with a descriptive message instead.
# Also, to ensure correct tear-down of the MPI environment before
# exiting, call this function with exit_code=0 to shutdown a
# successful CO𝘕CEPT run.
def abort(*args, exit_code=1, prefix='Aborting', **kwargs):
    # Print out message on error
    if exit_code != 0:
        # In an effort to prevent the error message from being printed
        # by multiple processes, all slave processes sleep a little.
        # If the master process is also aborting, this will fire the
        # comm.Abort() before the slaves wake up, prohibiting the same
        # error message from appearing many times.
        if not master:
            sleep(1)
        warn(*args, prefix=prefix, **kwargs)
        sleep(0.1)
    if master:
        masterprint(f'Total execution time: {time_since(start_time)}', **kwargs)
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

# Function for converting lines of Python code to equivalent lines
# but with multi-line expressions rewritten into single lines.
def onelinerize(lines):
    # Maybe replace with
    # def onelinerize(s):
    #    return re.sub('\n+', '\n', re.sub(r'([{\[\(,\.]|(?:\\))\s+', r'\1', s, flags=re.M))
    in_quotes = [False, False]
    in_triple_quotes = [False, False]
    paren_counts = {'paren': 0, 'brack': 0, 'curly': 0}
    def count_parens(line):
        if line.lstrip().startswith('#'):
            return line
        for j, ch in enumerate(line):
            # Inside quotes?
            if ch in ("'", '"'):
                if ch == "'" and not in_quotes[1]:
                    in_quotes[0] = not in_quotes[0]
                elif ch == '"' and not in_quotes[0]:
                    in_quotes[1] = not in_quotes[1]
                if j >= 2 and line[(j-2):(j + 1)] == "'"*3:
                    in_quotes[0] = not in_quotes[0]
                    in_triple_quotes[0] = not in_triple_quotes[0]
                    if not in_triple_quotes[0]:
                        in_quotes[0] = in_quotes[1] = False
                elif j >= 2 and line[(j-2):(j + 1)] == '"'*3:
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
    stay_as_multiline = False
    for i, line in enumerate(lines):
        line = count_parens(line)
        if (
               paren_counts['paren'] > 0
            or paren_counts['brack'] > 0
            or paren_counts['curly'] > 0
        ):
            if not multiline:
                # Multi-line statement begins
                multiline = True
                line_lstripped = line.lstrip()
                if (
                    line_lstripped.startswith('@')
                    and not line_lstripped.startswith('@cython.iterator')
                ):
                    stay_as_multiline = True
                    new_lines.append(line)
                    continue
                if line.count('"'*3) == 1 or line.count("'"*3) == 1:
                    stay_as_multiline = True
                    new_lines.append(line)
                    continue
                if line_lstripped.startswith('#'):
                    # Delete full-line comment
                    # within multi-line statement.
                    line = ''
                if line:
                    multiline_statement.append(line.rstrip())
            else:
                # Multi-line statement continues
                if stay_as_multiline:
                    new_lines.append(line)
                    continue
                if line.lstrip().startswith('#'):
                    # Delete full-line comment
                    # within multi-line statement.
                    line = ''
                if line:
                    multiline_statement.append(' ' + line.strip())
        elif multiline:
            # Multi-line statement ends
            multiline = False
            if stay_as_multiline:
                stay_as_multiline = False
                new_lines.append(line)
                continue
            multiline_statement.append(' ' + line.lstrip())
            new_lines.append(''.join(multiline_statement))
            multiline_statement = []
        else:
            new_lines.append(line)
    return new_lines



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
    for directive in (
        'header',
        'iterator',
        'nounswitching',
        'pheader',
        'remove',
    ):
        setattr(cython, directive, dummy_decorator)
    # Address (pointers into arrays)
    def address(a):
        dtype = np.ctypeslib.as_ctypes(a)._type_.__name__.split('_Array')[0]
        return a.ctypes.data_as(ctypes.POINTER(getattr(ctypes, dtype)))
    setattr(cython, 'address', address)
    # C allocation syntax for memory management
    def sizeof(dtype):
        # Extract number of pointer references and base datatype
        nstars = dtype.count('*')
        dtype = C2np.get(dtype.rstrip('*'), object)
        # Return both results
        return dtype, nstars
    def malloc(a):
        if not a:
            return None
        # Unpack
        dtype, nstars = a[0], a[1]
        size = len(a)//2
        # Emulate direct pointer as NumPy array
        if nstars == 0:
            return empty(size, dtype=dtype)
        # Emulate double (or higher) pointers as lists
        return [None for _ in range(size)]
    def realloc(p, a):
        if p is None:
            return malloc(a)
        size = len(a)//2
        if isinstance(p, list):
            # Reallocation of pointer array (e.g. double**)
            p = p[:size] + [None]*(size - len(p))
        else:
            # Reallocation pointer (e.g. double*)
            p.resize(size, refcheck=False)
        return p
    def free(a):
        if a is None or len(a) == 0:
            return
        # Do nothing in the case of pointer arrays
        if isinstance(a, list):
            return
        # NumPy arrays cannot be manually freed.
        # Resize the array to the minimal size.
        a.resize(0, refcheck=False)
    # Casting
    def cast(a, dtype):
        if not isinstance(dtype, str):
            dtype = dtype.__name__
        match = re.search(r'(.*)\[', dtype)
        if match:
            # Pointer to array cast assumed
            frame = inspect.stack()[1][0]
            shape = dtype.replace(':', '')
            shape = shape[(shape.index('[') + 1):]
            shape = shape.rstrip(' \t\n,')[:-1] + ','
            shape = f'({shape})'
            shape = eval(shape, frame.f_globals, frame.f_locals)
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
    fused_numeric = fused_numeric2 = fused_integral = fused_floating = []
    # Mathematical functions
    from numpy import (
        sin, cos, tan,
        arcsin, arccos, arctan, arctan2,
        sinh, cosh, tanh,
        arcsinh, arccosh, arctanh,
        exp, log, log2, log10,
        sqrt, cbrt,
        floor, ceil, round,
    )
    from scipy.special import erf, erfc
    # The closest thing to a Null pointer in pure Python
    # is the None object.
    NULL = None
    # Dummy functions and constants
    def dummy_func(*args, **kwargs):
        pass
    # The BlackboardBold dicts for constant expressions
    class BlackboardBold(dict):
        def __init__(self, constant_type):
            self.constant_type = constant_type
        def __getitem__(self, key):
            return self.convert(key)
        def convert(self, key):
            if self.constant_type is object:
                return key
            return self.constant_type(key)
    𝔹 = BlackboardBold(bool)
    𝕆 = BlackboardBold(object)
    ℝ = BlackboardBold(float)
    𝕊 = BlackboardBold(str)
    ℤ = BlackboardBold(int)
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
    # and copy_on_import.
    class DummyContextManager:
        def __call__(self, *args):
            return self
        def __enter__(self):
            pass
        def __exit__(self, *exc_info):
            pass
    unswitch = DummyContextManager()
    copy_on_import = DummyContextManager()
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
        lines = onelinerize(s.split('\n'))
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
            except Exception:
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
            # Only type given. Initialise value to None
            # for pointer type or 0 for non-pointer type.
            ctype, val = val, '__build_struct_undefined__'
        try:
            ctype = C2np[ctype]
        except Exception:
            try:
                ctype = eval(ctype)
            except Exception:
                pass
        ctypes[key] = ctype
        if val == '__build_struct_undefined__':
            try:
                val = ctype()
            except Exception:
                val = b'' if '*' in ctype else 0
        kwargs[key] = val
    for key, val in kwargs.copy().items():
        if isinstance(val, str):
            # Evaluate value given as string expression
            namespace = {k: v for d in (globals(), kwargs) for k, v in d.items()}
            try:
                val = eval(val, namespace)
            except Exception:
                pass
        try:
            kwargs[key] = ctypes[key](val)
        except Exception:
            pass
    if not cython.compiled:
        # In pure Python, emulate a struct by a simple namespace
        structure = types.SimpleNamespace(**kwargs)
    else:
        structure = ...  # To be added by pyxpp
    return structure, kwargs



###################################
# Cython imports and declarations #
###################################
pxd("""
# Import everything from cython_gsl,
# granting access to most of GNU Scientific Library.
from cython_gsl cimport *
# Functions for manual memory management
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# Create a fused integral type consisting of all the (signed) integral
# types used by the code.
ctypedef fused fused_integral:
    signed char
    int
    Py_ssize_t
# Create a fused floating type consisting of all the floating
# types used by the code.
ctypedef fused fused_floating:
    float
    double
# Create a fused numeric type consisting of the union of integral
# and floating types above. In fact we make two, useful for separate
# specializations of separate function arguments.
ctypedef fused fused_numeric:
    signed char
    int
    Py_ssize_t
    float
    double
ctypedef fused fused_numeric2:
    signed char
    int
    Py_ssize_t
    float
    double
""")
with copy_on_import:
    pxd("""
    # Mathematical functions
    from libc.math cimport (
        sin, cos, tan,
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
    )
    from libc.math   cimport  fabs
    from libc.stdlib cimport llabs
    """)
# Custom extension types using @cython.cclass will be found by the
# pyxpp preprocessor. A comment containing such types will be placed in
# the .pyx version of the given .py file. These are then collected
# together into the .types.pyx file. The fused types defined above
# as well as the content of the cython_gsl module cimported above
# also contain some extension types however.
# Here we add this comment manually, directly in the .py file.
# Extension types implemented by this module:
#     fused_integral, fused_floating, fused_numeric, fused_numeric2, gsl_...: from cython_gsl cimport *



#####################
# Unicode functions #
#####################
# The pyxpp script convert all Unicode source code characters into
# ASCII using the below function. It is also used during runtime.
@cython.pheader(
    # Arguments
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
    begin = unicode_tags['begin']
    end = unicode_tags['end']
    for c in s:
        if in_unicode_char or ord(c) > 127:
            # Unicode
            in_unicode_char = True
            unicode_char += c
            try:
                unicode_name = unicodedata.name(unicode_char)
                # unicode_char is a string (of length 1 or more)
                # regarded as a single unicode character.
                for pat, sub in unicode_subs.items():
                    unicode_name = unicode_name.replace(pat, sub)
                char_list.append(f'{begin}{unicode_name}{end}')
                in_unicode_char = False
                unicode_char = ''
            except Exception:
                pass
        else:
            # ASCII
            char_list.append(c)
    return ''.join(char_list)
cython.declare(unicode_subs=dict, unicode_tags=dict)
unicode_subs = {
    ' ': '__SPACE__',
    '-': '__DASH__',
}
unicode_tags = {
    'begin': 'UNICODE_',
    'end'  : '_EDOCINU',
}

# The function below grants the code access to
# Unicode string literals by undoing the conversion of the
# asciify function above.
@cython.pheader(s=str, returns=str)
def unicode(s):
    begin = unicode_tags['begin']
    end = unicode_tags['end']
    return re.sub(f'{begin}(.*?){end}', unicode_repl, s)
@cython.pheader(
    # Arguments
    match=object,  # re.Match
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
def unicode_subscript(s):
    return ''.join([unicode_subscripts.get(c, c) for c in s])
cython.declare(unicode_subscripts=dict, unicode_subscripts_inv=dict)
unicode_subscripts = dict(zip('0123456789-+e.', [unicode(c) for c in
    ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉', '₋', '₊', 'ₑ', '.')]))
unicode_subscripts_inv = {value: key for key, value in unicode_subscripts.items()}

# This function takes in a number (string) and
# returns it written in Unicode superscript.
def unicode_superscript(s):
    return ''.join([unicode_superscripts.get(c, c) for c in s])
cython.declare(unicode_superscripts=dict, unicode_superscripts_inv=dict)
unicode_superscripts = dict(zip('0123456789-+e.', [unicode(c) for c in (
    '⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁻', '', '×10', '⋅')]))
unicode_superscripts_inv = {value: key for key, value in unicode_superscripts.items()}

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
    kword=str,
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
    '10¹⁰ m☉' → '10**(10)*m_sun'
    """
    # Ensure Unicode
    unit_str = unicode(unit_str)
    # Replace multiple spaces with a single space
    unit_str = re.sub(r'\s+', ' ', unit_str).strip()
    # Replace spaces with an asterisk
    # if no operator is to be find on either side.
    parse_ok = True
    try:
        ast.parse(unit_str)
    except SyntaxError:
        parse_ok = False
    operators = '+-*/^&|@%<>='
    unit_list = list(unit_str)
    for i, c in enumerate(unit_str):
        if c != ' ' or i == 0 or i == len(unit_list) - 1:
            continue
        before = unit_list[i - 1]
        after  = unit_list[i + 1]
        if before in operators or after in operators:
            continue
        # Also check for keywords
        for kword in keyword.kwlist:
            if i - len(kword) >= 0:
                before = ''.join(unit_list[i-len(kword):i])
                if before == kword:
                    break
            if i + len(kword) + 1 <= len(unit_list):
                after = ''.join(unit_list[i+1:i+len(kword)+1])
                if after == kword:
                    break
        else:
            # Do replacement
            unit_list[i] = '*'
            try:
                ast.parse(''.join(unit_list))
                parse_ok = True
            except SyntaxError:
                if parse_ok:
                    # Undo replacement
                    unit_list[i] = c
    unit_str = ''.join(unit_list)
    # Various replacements
    mapping = {
        '×'           : '*',
        '^'           : '**',
        unicode('m☉') : 'm_sun',
        unicode('m_☉'): 'm_sun',
        asciify('m☉') : 'm_sun',
        asciify('m_☉'): 'm_sun',
    }
    for pat, rep in mapping.items():
        unit_str = unit_str.replace(pat, rep)
    # Convert superscript to power
    unit_str = re.sub(unicode('[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+'), r'**(\g<0>)', unit_str)
    # Convert Unicode superscript ASCII text
    for ASCII_char, unicode_superscript in unicode_superscripts.items():
        if unicode_superscript:
            unit_str = unit_str.replace(unicode_superscript, ASCII_char)
    # Insert an asterisk between numbers and letters
    # (though not the other way around).
    unit_str = re.sub(r'(([^_a-zA-Z0-9\.]|^)[0-9\.\)]+) ?([_a-zA-Z])', r'\g<1>*\g<3>', unit_str)
    unit_str = re.sub(r'([0-9])\*e([0-9+\-])', r'\g<1>e\g<2>', unit_str)
    unit_str = unit_str.replace(',*', ',')
    return unit_str

# Function which converts a string containing (possibly) units
# to the corresponding numerical value.
@cython.pheader(
    # Arguments
    unit_str_in=object,  # str or bytes
    namespace=dict,
    fail_on_error='bint',
    # Locals
    unit=object,  # double or NoneType
    unit_str=str,
    returns=object,  # double or NoneType
)
def eval_unit(unit_str_in, namespace=None, fail_on_error=True):
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
    if isinstance(unit_str_in, bytes):
        unit_str = unit_str_in.decode()
    else:
        unit_str = unit_str_in
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
        unicode('τ'): τ,
        asciify('τ'): τ,
        'tau': τ,
    }
    namespace.pop('min')
    namespace.pop('max')
    if fail_on_error:
        unit = eval(unit_str, namespace)
    else:
        try:
            unit = eval(unit_str, namespace)
        except Exception:
            unit = None
    return unit



##################################
# Paths to directories and files #
##################################
# The paths are stored in the .path file
cython.declare(path=object)  # Path
def load_source(module_name, filename):
    loader = importlib.machinery.SourceFileLoader(module_name, filename)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module
if master:
    def find_concept_dir():
        concept_dir_candidates = [os.path.abspath('.')] + sys.path
        for concept_dir in concept_dir_candidates:
            if not os.path.exists(concept_dir):
                continue
            if not os.path.isdir(concept_dir):
                concept_dir = os.path.dirname(concept_dir)
            while concept_dir != '/':
                if '.path' in os.listdir(concept_dir):
                    return concept_dir
                concept_dir = os.path.dirname(concept_dir)
    concept_dir = find_concept_dir()
    if concept_dir is None:
        abort('Could not find the .path file!')
    path_module = load_source('path', f'{concept_dir}/.path')
    path = {
        key: value
        for key, value in path_module.__dict__.items()
        if isinstance(key, str) and not key.startswith('__')
    }
path = bcast(path if master else None)
# Wrap the path variable in a convenient container class,
# enabling both bracketed and dotted lookup.
class Path(dict):
    def __getattr__(self, name):
       return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        self.pop(name)
path = Path(path)
# Function for converting an absolute path to its "sensible" form.
# That is, this function returns the relative path with respect to the
# concept directory, if it is no more than one directory above the
# concept directory. Otherwise, return the absolute path back again.
@cython.header(
    # Arguments
    abspath=str,
    # Locals
    relpath=str,
    returns=str,
)
def sensible_path(abspath):
    if not abspath:
        return abspath
    relpath = os.path.relpath(abspath, path['concept_dir'])
    if relpath.startswith('../../'):
        return abspath
    return relpath



#######################
# Read parameter file #
#######################
# Read in the content of the copied parameter file
cython.declare(param_file_content=str)
param_file_content = ''
if master and jobid != -1:
    # The path to the copied parameter file
    param_cp = path['job_dir'] + f'/{jobid}/param'
    # If no parameter file is specified we do not attempt to read
    # in the copy of it (unless such a copy mysteriously exists anyway).
    if param or os.path.isfile(param_cp):
        # Allow a little time for the copied parameter file to appear
        sleep_time = 0.1
        sleep_max = 30
        slept = 0
        while slept < sleep_max:
            if os.path.isfile(param_cp):
                break
            sleep(sleep_time)
            slept += sleep_time
        else:
            if not param:
                abort(f'No copy of parameters to be found at "{param_cp}"')
            else:
                abort(
                    f'No copy of parameter file "{param}" '
                    f'to be found at "{param_cp}"'
                )
        with open(param_cp, mode='r', encoding='utf-8') as param_file:
            param_file_content = param_file.read()
param_file_content = bcast(param_file_content)
# Often, h ≡ H0/(100*km/(s*Mpc)) is used in unit specifications.
# To allow for this, we add lines defining h from H0.
param_file_content += '\n'.join([
    '\n# Added by commons.py',
    'try:',
    '    h = H0/(100*km/(s*Mpc))',
    'except NameError:',
    '    h = 1',
    'h = float("{:.15f}".format(h))',
    'h = h',  # To ensure h gets flagged as used
])
# All further handling of parameters defined in the parameter file
# will be done later.



###########################
# Dimensionless constants #
###########################
cython.declare(
    machine_ϵ='double',
    machine_ϵ_32='float',
    π='double',
    τ='double',
    ρ_vacuum='double',
    ထ='double',
    NaN='double',
)
with warnings.catch_warnings(action='ignore', category=UserWarning):
    machine_ϵ = float(np.finfo(C2np['double']).eps)
    machine_ϵ_32 = np.finfo(C2np['float']).eps
π = float(np.pi)
τ = 2*π
ρ_vacuum = float(1e+2*machine_ϵ)
ထ = float('inf')
NaN = float('nan')  # Note: nan conflicts with the nan() function of C's math lib and NAN conflicts with the C macro



##################
# Physical units #
##################
# Dict relating all implemented units to the basic three units
# (pc, yr, m_sun). Note that 'min' is not a good
# name for minutes, as this name is already taken by the min function.
# You may specify units here which shall not be part of the units
# actually available in the code (this will come later).
# The specifications here are purely for the purpose of defining
# a unit system based on the user input.
unit_relations = {
    'yr':    1,
    'pc':    1,
    'm_sun': 1,
}
# Add other time units
unit_relations['kyr'    ] = 1e+3           *unit_relations['yr'     ]
unit_relations['Myr'    ] = 1e+6           *unit_relations['yr'     ]
unit_relations['Gyr'    ] = 1e+9           *unit_relations['yr'     ]
unit_relations['day'    ] = 1/365.25       *unit_relations['yr'     ]  # Exact Julian year
unit_relations['hr'     ] = 1/24           *unit_relations['day'    ]
unit_relations['minutes'] = 1/60           *unit_relations['hr'     ]
unit_relations['s'      ] = 1/60           *unit_relations['minutes']
# Add other length units
unit_relations['kpc'    ] = 1e+3           *unit_relations['pc'     ]
unit_relations['Mpc'    ] = 1e+6           *unit_relations['pc'     ]
unit_relations['Gpc'    ] = 1e+9           *unit_relations['pc'     ]
unit_relations['AU'     ] = τ/(60*60*360)  *unit_relations['pc'     ]  # IAU exact definition, 2015
unit_relations['m'      ] = 1/149597870700 *unit_relations['AU'     ]  # IAU exact definition, 2012
unit_relations['mm'     ] = 1e-3           *unit_relations['m'      ]
unit_relations['cm'     ] = 1e-2           *unit_relations['m'      ]
unit_relations['km'     ] = 1e+3           *unit_relations['m'      ]
unit_relations['ly'     ] = (                                          # CGPM exact definition, 1983
    (299792458*unit_relations['m']/unit_relations['s'])*unit_relations['yr']
)
unit_relations['kly'    ] = 1e+3           *unit_relations['ly'     ]
unit_relations['Mly'    ] = 1e+6           *unit_relations['ly'     ]
unit_relations['Gly'    ] = 1e+9           *unit_relations['ly'     ]
# Add other mass units
unit_relations['km_sun' ] = 1e+3           *unit_relations['m_sun'  ]
unit_relations['Mm_sun' ] = 1e+6           *unit_relations['m_sun'  ]
unit_relations['Gm_sun' ] = 1e+9           *unit_relations['m_sun'  ]
unit_relations['kg'     ] = 1/1.98841e+30  *unit_relations['m_sun'  ]  # Particle data group, 2019
unit_relations['g'      ] = 1e-3           *unit_relations['kg'     ]
# Add energy units
unit_relations['J'      ] = (
    unit_relations['kg']*unit_relations['m']**2*unit_relations['s']**(-2)
)
unit_relations['eV'     ] = 1.602176634e-19*unit_relations['J'      ]  # Particle data group, 2019
unit_relations['meV'    ] = 1e-3           *unit_relations['eV'     ]
unit_relations['keV'    ] = 1e+3           *unit_relations['eV'     ]
unit_relations['MeV'    ] = 1e+6           *unit_relations['eV'     ]
unit_relations['GeV'    ] = 1e+9           *unit_relations['eV'     ]
unit_relations['TeV'    ] = 1e+12          *unit_relations['eV'     ]
# Add additional units/constants
unit_relations['light_speed'] = unit_relations['c'] = unit_relations['ly']/unit_relations['yr']
unit_relations['h_bar'] = unit_relations['hbar'] = unit_relations[unicode('ħ')] = unit_relations[asciify('ħ')] = (
    6.62607015e-34/τ                                                   # Exact definition, 2019
    *unit_relations['kg']*unit_relations['m']**2/unit_relations['s']
)
unit_relations['G_Newton'] = unit_relations['G'] = (                   # Particle data group, 2019
    6.67430e-11*unit_relations['m']**3/(unit_relations['kg']*unit_relations['s']**2)
)



# Function which given a function name present both in NumPy and in the
# Python built-in's will produce a function which first calls the NumPy
# version, and then (on failure) calls the built-in version. This is
# useful for e.g. generating a 'min' function which works on both a
# single multi-dimensional array as well as multiple scalar inputs.
def produce_np_and_builtin_function(funcname):
    np_func = getattr(np, funcname)
    try:
        builtin_func = getattr(__builtins__, funcname)
    except Exception:
        builtin_func = __builtins__[funcname]
    def np_and_builtin_function(*args, **kwargs):
        try:
            return np_func(*args, **kwargs)
        except Exception:
            return builtin_func(*args, **kwargs)
    return np_and_builtin_function
# Convenient class for wrapping the param variable,
# for easy and flexible use within parameter files.
class Param:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(self.path)
        self.dir = os.path.dirname(self.path)
    __str__  = lambda self: self.name
    __add__  = lambda self, s: self.name + s
    __radd__ = lambda self, s: s + self.name
# Attempt to read in the parameter file. This is really only
# to get any user defined units, as the parameter file will
# be properly read in later. First construct a namespace in which the
# parameters can be read in.
def construct_user_params_namespace(params_iteration):
    user_params = {
        # Include the command-line arguments,
        # but wrap param in a useful container.
        'jobid': jobid,
        'param': Param(param),
        # Include all of NumPy
        **vars(np).copy(),
        # Overwrite the NumPy min and max function with NumPy and
        # built-in hybrid min and max functions.
        'min': produce_np_and_builtin_function('min'),
        'max': produce_np_and_builtin_function('max'),
        # The path container
        'path': path,
        # Modules
        'numpy': np,
        'np'   : np,
        'os'   : os,
        're'   : re,
        'sys'  : sys,
        # Functions
        'rand'    : np.random.random,
        'random'  : np.random.random,
        'basename': os.path.basename,
        'dirname' : os.path.dirname,
        'glob'    : glob,
        # MPI variables and functions
        'master'         : master,
        'nprocs'         : nprocs,
        'rank'           : rank,
        'node'           : node,
        'node_master'    : node_master,
        'nprocs_node'    : nprocs_node,
        'bcast'          : bcast,
        'call_openmp_lib': call_openmp_lib,
        # Constants
        asciify('π')        : π,
        unicode('π')        : π,
        asciify('τ')        : τ,
        unicode('τ')        : τ,
        'tau'               : τ,
        asciify('machine_ϵ'): machine_ϵ,
        unicode('machine_ϵ'): machine_ϵ,
        asciify('machine_ε'): machine_ϵ,
        unicode('machine_ε'): machine_ϵ,
        'eps'               : machine_ϵ,
        asciify('ထ')        : ထ,
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
    # Remove "size" (np.size) form the user_params.
    # We do this because of the _size parameter idiom, where with
    # np.size available in user_params it appears as though the
    # underscore is unnecessary.
    user_params.pop('size')
    return user_params
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
# dictionary-like object d, ignoring any exceptions.
# The execution stops when no more exceptions are resolved
# by definitions further down in the content.
def exec_params(content, d_in, suppress_exceptions=True):
    d = dict(d_in)
    # Perform execution
    lines = onelinerize(content.split('\n'))
    lines_executed = set()
    lines_executed_prev = {-1}
    while lines_executed != lines_executed_prev:
        lines_executed_prev = lines_executed
        lines_executed = set()
        lines_failed = []
        for n, line in enumerate(lines):
            try:
                exec(line, d)
                lines_executed.add(n)
                lines_failed = []
                continue
            except Exception:
                pass
            lines_failed.append(n)
            if len(lines_failed) < 2:
                continue
            try:
                exec('\n'.join([lines[m] for m in lines_failed]), d)
            except Exception:
                continue
            for m in lines_failed:
                lines_executed.add(m)
            lines_failed = []
    if lines_failed:
        try:
            exec('\n'.join([lines[m] for m in lines_failed]), d)
        except Exception:
            pass
    if not suppress_exceptions:
        # If exceptions should raise an error, we do an extra exec
        # on the full content.
        exec(content, d)
    # Changes should be reflected in the input container
    d_in.update(d)
# Execute the content of the parameter file in the namespace defined
# by user_params in order to get the user defined units.
exec_params(param_file_content, user_params)
# The names of the three fundamental units,
# all with a numerical value of 1. If these are not defined in the
# parameter file, give them some reasonable values.
cython.declare(
    unit_time=str,
    unit_length=str,
    unit_mass=str,
)
unit_time   = unformat_unit(user_params.get('unit_time'  , 'Gyr'    ))
unit_length = unformat_unit(user_params.get('unit_length', 'Mpc'    ))
unit_mass   = unformat_unit(user_params.get('unit_mass'  , '10¹⁰ m☉'))
# Construct a struct containing the values of all units
units, units_dict = build_struct(
    # Values of basic units,
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
cython.declare(
    light_speed='double',
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
# Function which replaces ellipses in a dict with previous value
def replace_ellipsis(d):
    """Note that for deterministic behaviour,
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
# Context manager which disables summarization of NumPy arrays
# (the ellipses appearing in str representations of large arrays).
@contextlib.contextmanager
def disable_numpy_summarization():
    # Backup the current print threshold
    threshold = np.get_printoptions()['threshold']
    # Set the threshold to infinity so that the str representation
    # of arrays will not contain any ellipses
    # (full printout rather than summarization).
    np.set_printoptions(threshold=ထ)
    try:
        # Yield control back to the caller
        yield
    finally:
        # Cleanup: Reset print options
        np.set_printoptions(threshold=threshold)
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
            if isinstance(f, (float, np.floating)):
                key = f'{f:.12g}'
        except Exception:
            pass
        try:
            f = ast.literal_eval(val)
            if isinstance(f, (float, np.floating)):
                val = f'{f:.12g}'
        except Exception:
            pass
        d_modified[key] = val
    return d_modified
# Function which calls CLASS with given parameters
# and returns the background.
def get_class_background(class_params, class_call_reason=''):
    from classy import Class
    cosmo = Class(concept_class_call=True)
    cosmo.set(stringify_dict(class_params))
    if class_call_reason:
        class_call_reason = class_call_reason.strip() + ' '
    masterprint(f'Calling CLASS {class_call_reason}...')
    cosmo.compute()
    masterprint('done')
    background = cosmo.get_background()
    return background
# This function returns the number of proper massive neutrino species,
# i.e. excluding warm dark matter. User parameters are referenced via
# the user_params dict so that this function may be called prior to
# initializing/finalizing these.
def get_number_of_massive_neutrino_species():
    num_ncdm = int(float(user_params.get('class_params', {}).get('N_ncdm', 0)))
    num_ncdm -= int(float(user_params.get('enable_warm_dark_matter', False)))
    if num_ncdm == -1:
        masterwarn(
            'You are running with warm dark matter '
            'but class_params contains no ncdm species'
        )
    return num_ncdm
# Function that updates given CLASS parameters with default values
# matching the CO𝘕CEPT parameters.
def update_class_params(class_params, namespace=None):
    """This function mutates the passed class_params dict by adding
    (never replacing existing) default parameters. If a namespace is
    supplied, these default parameters are taken from here.
    Otherwise, they are taken from the global variables.
    """
    replace_ellipsis(class_params)
    # Specification of the cosmology,
    # inherited from the CO𝘕CEPT parameters.
    if namespace is None:
        class_params_default = {
            'H0'       : H0/(units.km/(units.s*units.Mpc)),
            'Omega_b'  : Ωb,
            'Omega_cdm': Ωcdm,
        }
    else:
        class_params_default = {}
        if 'H0' in namespace:
            class_params_default['H0'] = namespace['H0']/(units.km/(units.s*units.Mpc))
        if 'Ωb' in namespace:
            class_params_default['Omega_b'] = namespace['Ωb']
        if 'Ωcdm' in namespace:
            class_params_default['Omega_cdm'] = namespace['Ωcdm']
    # Apply updates to the CLASS parameters
    for param_name, param_value in class_params_default.items():
        class_params.setdefault(param_name, param_value)
    # Set the CO𝘕CEPT specific growthfac_contrib_ncdm CLASS parameter
    # if we are running with warm dark matter.
    if float(user_params.get('enable_warm_dark_matter', False)):
        num_ncdm = get_number_of_massive_neutrino_species()
        class_params.setdefault('growthfac_contrib_ncdm', [0.0]*num_ncdm + [1.0])
    # Transform all CLASS container parameters to str's of
    # comma-separated values. All other CLASS parameters will also
    # be converted to their str representation.
    class_params = stringify_dict(class_params)
    return class_params
# Subclass the dict to create a dict-like object which keeps track of
# the number of lookups on each key. This is used to identify unknown
# (and therefore unused) parameters defined by the user.
# Transform all keys to Unicode during lookups and assignments.
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
                val = super().__getitem__(key)
                if not isinstance(val, types.ModuleType):
                    list_of_unused.append(key)
        return list_of_unused
# Dict-like object constituting the namespace for the statements
# in the user specified parameter file.
# Note that the previously defined user_params is overwritten.
user_params = DictWithCounter(construct_user_params_namespace('inferables'))
# Additional things which should be available when defining parameters
user_params.update({
    # Names of the fundamental units
    'unit_time'  : unit_time,
    'unit_length': unit_length,
    'unit_mass'  : unit_mass,
    # Units from the units struct
    **units_dict,
    'units': types.SimpleNamespace(**units_dict),
    # Physical constants
    'light_speed': light_speed,
    'c'          : light_speed,
    asciify('ħ') : ħ,
    unicode('ħ') : ħ,
    'h_bar'      : ħ,
    'G_Newton'   : G_Newton,
    # Constants
    asciify('π')        : π,
    unicode('π')        : π,
    asciify('τ')        : τ,
    unicode('τ')        : τ,
    'tau'               : τ,
    asciify('machine_ϵ'): machine_ϵ,
    unicode('machine_ϵ'): machine_ϵ,
    asciify('machine_ε'): machine_ϵ,
    unicode('machine_ε'): machine_ϵ,
    'eps'               : machine_ϵ,
    asciify('ထ')        : ထ,
    unicode('ထ')        : ထ,
    # Inferred parameters (values not yet inferred)
    **inferred_params,
})
# At this point, user_params does not contain actual parameters.
# Mark all items in user_params as used.
user_params.useall()
# "Import" the parameter file by executing it
# in the namespace defined by the user_params namespace.
exec_params(param_file_content, user_params, suppress_exceptions=False)
# Update the class_params inside of the user_params with default values.
# This is important for the variable inference to come. For the actual
# (module level) class_params, this has to be done again (see the
# CLASS setup section).
if 'class_params' in user_params:
    user_params['class_params'] = update_class_params(user_params['class_params'], user_params)
# Find out which of the inferable parameters are not explicitly set,
# and so should be inferred.
user_params_changed_inferables = user_params.copy()
for key, val in inferred_params.items():
    # Make sure to implement every possible type
    # of the inferable parameters.
    if val is None:
        user_params_changed_inferables[key] = False
    elif val is ...:
        user_params_changed_inferables[key] = True
    elif isinstance(val, bool):
        user_params_changed_inferables[key] = not val
    elif isinstance(val, bytes):
        user_params_changed_inferables[key] += b' '
    elif isinstance(val, str):
        user_params_changed_inferables[key] += ' '
    elif isinstance(val, (tuple, list, set)):
        user_params_changed_inferables[key] = type(val)(list(val) + [0])
    else:
        # Assume numeric type
        user_params_changed_inferables[key] += 1
user_params_changed_inferables['params_iteration'] = 'inferables (changed)'
exec_params(param_file_content, user_params_changed_inferables, suppress_exceptions=False)
inferred_params_set = DictWithCounter({
    key: user_params[key] == user_params_changed_inferables[key] for key in inferred_params
})
# Infer inferable parameters which have not been explicitly set
# by the user. To ensure that every key gets looked up in their proper
# Unicode form, we wrap the inferred_params dict in a DictWithCounter.
inferred_params = DictWithCounter(inferred_params)
inferred_params_final = DictWithCounter()
cython.declare(
    Ων='double',
)
Ων = float(inferred_params['Ων'])
if inferred_params_set['Ων']:
    Ων = float(user_params['Ων'])
else:
    num_ncdm = get_number_of_massive_neutrino_species()
    if num_ncdm > 0:
        if master:
            background = get_class_background(
                user_params.get('class_params', {}),
                class_call_reason=(
                    f'in order to determine Ων = '
                    + ' + '.join(['Ων' + unicode_subscript(str(i)) for i in range(num_ncdm)])
                ),
            )
            Ων = 0
            for i in range(num_ncdm):
                Ων += background[f'(.)rho_ncdm[{i}]'][-1]
            Ων /= background['(.)rho_crit'][-1]
        Ων = bcast(Ων)
inferred_params_final['Ων'] = Ων
# Update user_params with the correct values for the inferred params
user_params.update(inferred_params_final)
# Re-"import" the parameter file by executing it
# in the namespace defined by the user_params namespace,
# this time with the inferable values in place.
user_params['params_iteration'] = 'final'
exec_params(param_file_content, user_params, suppress_exceptions=False)
# Backup of originally supplied user parameter names
user_params_keys_raw = set(user_params.keys())
# Read in all parameters
def to_int(value):
    return int(round(float(value)))
def to_rgb(value, *, collective=False):
    # If called with collective=True, only the master process
    # does the work, the result of which is then broadcasted.
    # This is just to avoid importing matplotlib on all processes.
    if not collective or master:
        matplotlib = get_matplotlib()
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = str(value)
        try:
            rgb = asarray(matplotlib.colors.ColorConverter().to_rgb(value), dtype=C2np['double'])
        except Exception:
            # Could not convert value to colour
            abort(f'Colour not understood: {value}')
    if collective:
        rgb = bcast(rgb if master else None)
    return rgb
def to_rgbα(value, α=1, *, collective=False):
    if isinstance(value, str):
        rgb = value
    else:
        value = any2list(value)
        if len(value) == 1:
            rgb = value[0]
        elif len(value) == 2:
            rgb, α = value
        elif len(value) == 3:
            rgb = value
        elif len(value) == 4:
            rgb, α = value[:3], value[3]
        else:
            # Could not convert value to colour and α
            abort(f'Colour + alpha not understood: {value}')
    return to_rgb(rgb, collective=collective), α
cython.declare(
    # Input/output
    initial_conditions=object,  # str or container of str's
    output_dirs=dict,
    output_bases=dict,
    output_times=dict,
    autosave_interval='double',
    snapshot_select=dict,
    powerspec_select=dict,
    bispec_select=dict,
    render2D_select=dict,
    render3D_select=dict,
    snapshot_type=str,
    gadget_snapshot_params=dict,
    snapshot_wrap='bint',
    life_output_order=tuple,
    select_particle_id=dict,
    class_plot_perturbations='bint',
    class_extra_background=set,
    class_extra_perturbations=set,
    # Numerical parameter
    boxsize='double',
    potential_options=dict,
    ewald_gridsize='Py_ssize_t',
    shortrange_params=dict,
    powerspec_options=dict,
    bispec_options=dict,
    bispec_antialiasing='bint',
    class_dedicated_spectra='bint',
    class_modes_per_decade=dict,
    # Cosmology
    H0='double',
    Ωb='double',
    Ωcdm='double',
    a_begin='double',
    t_begin='double',
    primordial_spectrum=dict,
    class_params=dict,
    enable_warm_dark_matter='bint',
    # Physics
    select_forces=dict,
    select_species=dict,
    select_eos_w=dict,
    select_boltzmann_order=dict,
    select_boltzmann_closure=dict,
    realization_options=dict,
    select_lives=dict,
    select_approximations=dict,
    softening_kernel=str,
    select_softening_length=dict,
    # Simulation options
    Δt_base_background_factor='double',
    Δt_base_nonlinear_factor='double',
    Δt_increase_max_factor='double',
    Δt_rung_factor='double',
    Δa_max_early='double',
    Δa_max_late='double',
    static_timestepping=object,  # str, callable or None
    N_rungs='Py_ssize_t',
    fftw_wisdom_rigor=str,
    fftw_wisdom_reuse='bint',
    fftw_wisdom_share='bint',
    random_generator=str,
    random_seeds=dict,
    primordial_noise_imprinting=str,
    primordial_amplitude_fixed='bint',
    primordial_phase_shift='double',
    cell_centered='bint',
    fourier_structure_caching=dict,
    fluid_scheme_select=dict,
    fluid_options=dict,
    class_k_max=dict,
    class_reuse='bint',
    # Graphics
    terminal_width='int',
    enable_terminal_formatting='bint',
    suppress_output=dict,
    bispec_plot_prefer=str,
    render2D_options=dict,
    render3D_options=dict,
    # Debugging options
    print_load_imbalance=object,
    allow_snapshot_multifile_singleload='bint',
    particle_reordering=object,
    enable_Hubble='bint',
    enable_class_background='bint',
    # Hidden parameters
    special_params=dict,
)
# Input/output
initial_conditions = user_params.get('initial_conditions', '')
user_params['initial_conditions'] = initial_conditions
output_kinds = ('snapshot', 'powerspec', 'bispec', 'render2D', 'render3D')
if isinstance(user_params.get('output_dirs'), str):
    output_dirs = {
        kind: user_params['output_dirs']
        for kind in output_kinds
    }
else:
    output_dirs = dict(user_params.get('output_dirs', {}))
replace_ellipsis(output_dirs)
for kind in output_kinds:
    output_dirs[kind] = str(output_dirs.get(kind, path['output_dir']))
    if not output_dirs[kind]:
        output_dirs[kind] = path['output_dir']
output_dirs['autosave'] = str(output_dirs.get('autosave', ''))
if not output_dirs['autosave']:
    output_dirs['autosave'] = path['ic_dir'] + '/autosave'
output_dirs = {key: sensible_path(path) for key, path in output_dirs.items()}
user_params['output_dirs'] = output_dirs
output_bases = dict(user_params.get('output_bases', {}))
replace_ellipsis(output_bases)
for kind in output_kinds:
    output_bases[kind] = str(output_bases.get(kind, kind))
user_params['output_bases'] = output_bases
output_times = dict(user_params.get('output_times', {}))
replace_ellipsis(output_times)
user_params['output_times'] = output_times
autosave_interval = float(user_params.get('autosave_interval', ထ))
user_params['autosave_interval'] = autosave_interval
snapshot_select = {}
if 'snapshot_select' in user_params:
    if isinstance(user_params['snapshot_select'], dict):
        snapshot_select = user_params['snapshot_select']
        replace_ellipsis(snapshot_select)
        if 'save' not in snapshot_select and 'load' not in snapshot_select:
            snapshot_select = {'save': snapshot_select, 'load': snapshot_select}
    else:
        snapshot_select = {
            'save': {'all': user_params['snapshot_select']},
            'load': {'all': user_params['snapshot_select']},
        }
    for key, val in snapshot_select.items():
        for key2, val2 in val.copy().items():
            if isinstance(val2, dict):
                replace_ellipsis(val2)
            else:
                val[key2] = {'pos': val2, 'mom': val2, 'ϱ': val2, 'J': val2, '𝒫': val2, 'ς': val2}
for key in ('save', 'load'):
    key_exists = (key in snapshot_select)
    snapshot_select.setdefault(key, {})
    snapshot_select[key].setdefault(
        'default',
        {
            'pos': not key_exists, 'mom': not key_exists,
            'ϱ': not key_exists, 'J': not key_exists, '𝒫': not key_exists, 'ς': not key_exists,
        },
    )
    if isinstance(snapshot_select[key], dict):
        replace_ellipsis(snapshot_select[key])
    else:
        snapshot_select[key] = {'all': bool(snapshot_select[key])}
    for d in snapshot_select[key].values():
        for key2, val2 in d.copy().items():
            d[unicode(key2)] = val2
            d[asciify(key2)] = val2
user_params['snapshot_select'] = snapshot_select
if 'powerspec_select' in user_params:
    if isinstance(user_params['powerspec_select'], dict):
        powerspec_select = user_params['powerspec_select']
    else:
        powerspec_select = {'default': user_params['powerspec_select']}
    for key, val in powerspec_select.copy().items():
        if not isinstance(val, dict):
            continue
        powerspec_select[key] = {
            key2.replace(' ', '').replace('-', '').replace('_', ''): val2
            for key2, val2 in val.items()
        }
    powerspec_select.setdefault(
        'default',
        {
            'data': False,
            'corrected': False,
            'linear': False,
            'plot': False,
        },
    )
else:
    powerspec_select = {
        'default': {
            'data': True,
            'corrected': False,
            'linear': True,
            'plot': True,
        },
    }
replace_ellipsis(powerspec_select)
for key, val in powerspec_select.copy().items():
    if isinstance(val, dict):
        val.setdefault('data', False)
        val.setdefault('corrected', False)
        val.setdefault('linear', False)
        val.setdefault('plot', False)
        unknown = ', '.join([
            f'"{do}"'
            for do in set(val.keys()) - {'data', 'corrected', 'linear', 'plot'}
        ])
        if unknown:
            abort(f'Unknown selections in powerspec_select["{key}"]: {unknown}')
    else:
        powerspec_select[key] = {
            'data': bool(val),
            'corrected': bool(val),
            'linear': bool(val),
            'plot': bool(val),
        }
user_params['powerspec_select'] = powerspec_select
if 'bispec_select' in user_params:
    if isinstance(user_params['bispec_select'], dict):
        bispec_select = user_params['bispec_select']
    else:
        bispec_select = {'default': user_params['bispec_select']}
    for key, val in bispec_select.copy().items():
        if not isinstance(val, dict):
            continue
        bispec_select[key] = {
            key2.replace(' ', '').replace('-', '').replace('_', ''): val2
            for key2, val2 in val.items()
        }
    bispec_select.setdefault(
        'default', {
            'data': False,
            'reduced': False,
            'treelevel': False,
            'plot': False,
        },
    )
else:
    bispec_select = {
        'default': {
            'data': True,
            'reduced': True,
            'treelevel': True,
            'plot': True,
        },
    }
replace_ellipsis(bispec_select)
for key, val in bispec_select.copy().items():
    if isinstance(val, dict):
        val.setdefault('data', False)
        val.setdefault('reduced', False)
        val.setdefault('treelevel', False)
        val.setdefault('plot', False)
        unknown = ', '.join([
            f'"{do}"'
            for do in set(val.keys()) - {
                'data', 'reduced', 'treelevel', 'plot',
            }
        ])
        if unknown:
            abort(f'Unknown selections in bispec_select["{key}"]: {unknown}')
    else:
        bispec_select[key] = {
            'data': bool(val),
            'reduced': bool(val),
            'treelevel': bool(val),
            'plot': bool(val),
        }
user_params['bispec_select'] = bispec_select
if 'render2D_select' in user_params:
    if isinstance(user_params['render2D_select'], dict):
        render2D_select = user_params['render2D_select']
    else:
        render2D_select = {'default': user_params['render2D_select']}
    for key, val in render2D_select.copy().items():
        if not isinstance(val, dict):
            continue
        render2D_select[key] = {
            key2.replace(' ', '').replace('-', '').replace('_', ''): val2
            for key2, val2 in val.items()
        }
    render2D_select.setdefault(
        'default',
        {
            'data': False,
            'image': False,
            'terminalimage': False,
        },
    )
else:
    render2D_select = {
        'default': {
            'data': True,
            'image': True,
            'terminalimage': True,
        },
    }
replace_ellipsis(render2D_select)
for key, val in render2D_select.copy().items():
    if isinstance(val, dict):
        val.setdefault('data', False)
        val.setdefault('image', False)
        val.setdefault('terminalimage', False)
        unknown = ', '.join([
            f'"{do}"'
            for do in set(val.keys()) - {'data', 'image', 'terminalimage'}
        ])
        if unknown:
            abort(f'Unknown selections in render2D_select["{key}"]: {unknown}')
    else:
        render2D_select[key] = {'data': bool(val), 'image': bool(val), 'terminalimage': bool(val)}
user_params['render2D_select'] = render2D_select
if 'render3D_select' in user_params:
    if isinstance(user_params['render3D_select'], dict):
        render3D_select = user_params['render3D_select']
    else:
        render3D_select = {'default': user_params['render3D_select']}
    render3D_select.setdefault('default', {'image': False})
else:
    render3D_select = {
        'default': {'image': True},
    }
replace_ellipsis(render3D_select)
for key, val in render3D_select.copy().items():
    if isinstance(val, dict):
        val.setdefault('image', False)
        unknown = ', '.join([f'"{do}"' for do in set(val.keys()) - {'image'}])
        if unknown:
            abort(f'Unknown selections in render3D_select["{key}"]: {unknown}')
    else:
        render3D_select[key] = {'image': bool(val)}
user_params['render3D_select'] = render3D_select
snapshot_type = (str(user_params.get('snapshot_type', 'concept'))
    .replace(unicode('𝘕'), 'N').replace(asciify('𝘕'), 'N')
    .replace(' ', '').replace('-', '')
    .lower()
)
user_params['snapshot_type'] = snapshot_type
gadget_snapshot_params_defaults = {
    'snapformat': 2,
    'dataformat': {
        'POS': 32,
        'VEL': 32,
        'ID' : 'automatic',
    },
    'particles per file': 'automatic',
    'parallel write': True,
    'Nall high word': 'NallHW',
    'header': {},
    'settle': 0,
    'units': {
        'length'  : 'kpc/h',
        'velocity': 'km/s',
        'mass'    : '10¹⁰ m☉/h',
    },
}
gadget_snapshot_params = dict(user_params.get('gadget_snapshot_params', {}))
for key, val in gadget_snapshot_params.copy().items():
    key_transformed = (
        str(key).lower().replace(' ', '').replace('_', '').replace('-', '')
    )
    for key_default in gadget_snapshot_params_defaults.keys():
        key_default_transformed = (
            str(key_default).lower().replace(' ', '').replace('_', '').replace('-', '')
        )
        if key_transformed == key_default_transformed:
            gadget_snapshot_params[key_default] = gadget_snapshot_params.pop(key)
            break
for key, val in gadget_snapshot_params_defaults.items():
    gadget_snapshot_params.setdefault(key, val)
gadget_snapshot_params['snapformat'] = int(gadget_snapshot_params['snapformat'])
if gadget_snapshot_params['snapformat'] not in (1, 2):
    abort(
        f'gadget_snapshot_params["snapformat"] = '
        f'{gadget_snapshot_params["snapformat"]} but must be 1 or 2'
    )
gadget_snapshot_params_dataformat = {}
for key, val in gadget_snapshot_params['dataformat'].items():
    for valid_key in gadget_snapshot_params_defaults['dataformat'].keys():
        if key.upper().startswith(valid_key):
            break
    else:
        abort(
            f'Unknown GADGET snapshot block "{key}" '
            f'listed in gadget_snapshot_params["dataformat"]'
        )
    gadget_snapshot_params_dataformat[valid_key] = val
replace_ellipsis(gadget_snapshot_params_dataformat)
for key, val in gadget_snapshot_params_defaults['dataformat'].items():
    gadget_snapshot_params_dataformat.setdefault(key, val)
for key, val in gadget_snapshot_params_dataformat.items():
    val_transformed = str(val).lower()
    if 'auto' in val_transformed:
        if key.lower() == 'id':
            val = 'automatic'
        else:
            val = 32
    elif any([
        pattern in val_transformed
        for pattern in ['64', '8', 'long', 'double', 'big']
    ]) or val_transformed in {'d', }:
        val = 64
    elif any([
        pattern in val_transformed
        for pattern in ['32', '4', 'int', 'uint', 'single', 'float', 'small', 'short']
    ]) or val_transformed in {'i', 'f'}:
        val = 32
    else:
        abort(
            f'Unknown format "{val}" specified as '
            f'gadget_snapshot_params["dataformat"]["{key}"]'
        )
    gadget_snapshot_params_dataformat[key] = val
gadget_snapshot_params['dataformat'] = gadget_snapshot_params_dataformat
if isinstance(gadget_snapshot_params['particles per file'], str):
    if 'auto' in gadget_snapshot_params['particles per file'].lower():
        gadget_snapshot_params['particles per file'] = -1
    else:
        abort(
            f'Could not understand gadget_snapshot_params["particles per file"] '
            f'= {gadget_snapshot_params["particles per file"]}'
        )
gadget_snapshot_params['particles per file'] = int(round(float(
    gadget_snapshot_params['particles per file']
)))
gadget_snapshot_params['parallel write'] = bool(gadget_snapshot_params['parallel write'])
for key in gadget_snapshot_params.copy():
    key_transformed = (
        key.lower().replace(' ', '').replace('-', '').replace('[', '').replace(']', '')
    )
    if any([pattern in key_transformed for pattern in ['nall', 'nallhw', 'high', 'word', 'hw']]):
        val = gadget_snapshot_params.pop(key)
        val_transformed = (
            val.lower().replace(' ', '').replace('-', '').replace('[', '').replace(']', '')
            .replace('nall', 'Nall').replace('hw', 'HW')
        )
        if val_transformed not in ('NallHW', 'Nall'):
            abort(
                f'Unrecognised gadget_snapshot_params["Nall high word"] = '
                f'"{val}" ∉ {{"NallHW", "Nall"}}'
            )
        gadget_snapshot_params['Nall high word'] = val_transformed
        break
if not gadget_snapshot_params['header']:
    gadget_snapshot_params['header'] = {}
replace_ellipsis(gadget_snapshot_params['header'])
gadget_snapshot_params_units = {}
for key, val in gadget_snapshot_params['units'].items():
    for valid_key in gadget_snapshot_params_defaults['units'].keys():
        if key == valid_key:
            break
    else:
        abort(f'Unknown GADGET unit "{key}"')
    gadget_snapshot_params_units[key] = val
for key, val in gadget_snapshot_params_defaults['units'].items():
    gadget_snapshot_params_units.setdefault(key, val)
gadget_snapshot_params['units'] = gadget_snapshot_params_units
gadget_snapshot_params['settle'] = int(gadget_snapshot_params['settle'])%2
for key in gadget_snapshot_params:
    if key not in gadget_snapshot_params_defaults:
        abort(f'Unknown sub-parameter "{key}" in gadget_snapshot_params')
user_params['gadget_snapshot_params'] = gadget_snapshot_params
snapshot_wrap = bool(user_params.get('snapshot_wrap', False))
life_output_order = tuple(user_params.get('life_output_order', ()))
life_output_order = tuple([act.lower() for act in life_output_order])
life_output_order = tuple([
    'terminate'
    if act.startswith('term') or act.startswith('deact') else act
    for act in life_output_order
])
life_output_order = tuple([
    'activate'
    if act.startswith('act') else act
    for act in life_output_order
])
life_output_order = tuple([
    'dump'
    if act.startswith('dump') or act.startswith('out') else act
    for act in life_output_order
])
acts = ('terminate', 'activate', 'dump')
for act in acts:
    if act not in life_output_order:
        life_output_order += (act,)
if set(life_output_order) != set(acts):
    abort(f'life_output_order = {life_output_order} not understood')
user_params['life_output_order'] = life_output_order
select_particle_id = {}
if user_params.get('select_particle_id'):
    if isinstance(user_params['select_particle_id'], dict):
        select_particle_id = user_params['select_particle_id']
        replace_ellipsis(select_particle_id)
    else:
        select_particle_id = {'all': user_params['select_particle_id']}
select_particle_id.setdefault('default', False)
user_params['select_particle_id'] = select_particle_id
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
# Numerical parameters
boxsize = float(user_params.get('boxsize', 512*units.Mpc))
user_params['boxsize'] = boxsize
if isinstance(user_params.get('potential_options', {}), (int, float, np.integer, np.floating)):
    potential_options = {
        'gridsize': int(round(user_params['potential_options'])),
    }
else:
    potential_options = dict(user_params.get('potential_options', {}))
valid_potential_options = {
    'gridsize',
    'interpolation',
    'deconvolve',
    'interlace',
    'differentiation',
}
for key in potential_options:
    if key not in valid_potential_options:
        abort(f'Option "{key}" in potential_options not understood')
potential_forces_implemented = {
    'gravity': ['pm', 'p3m'],  # Default force
    'lapse': ['pm'],
}
potential_methods_implemented = set(itertools.chain(*potential_forces_implemented.values()))
PotentialGridsizesComponent = collections.namedtuple(
    'PotentialGridsizesComponent',
    ('upstream', 'downstream'),
)
def canonicalize_input_str(method):
    method = re.sub(r'[ _\-^()]', '', method.lower())
    for n in range(10):
        method = method.replace(unicode_superscript(str(n)), str(n))
    return method
if isinstance(potential_options.get('gridsize', {}), dict):
    potential_gridsizes = replace_ellipsis(potential_options.get('gridsize', {}))
    potential_gridsizes.setdefault('global', -1)
else:
    val = potential_options['gridsize']
    potential_gridsizes = {'global': val, 'all': val}
for key in potential_gridsizes.copy():
    if key in potential_forces_implemented.keys():
        val = potential_gridsizes.pop(key)
        for key2 in ('global', 'all'):
            if key2 not in potential_gridsizes or potential_gridsizes[key2] in (-1, (-1, -1)):
                potential_gridsizes[key2] = {}
            potential_gridsizes[key2].setdefault(key, val)
for key, val in potential_gridsizes.copy().items():
    if not isinstance(val, dict):
        for potential_forces in potential_forces_implemented:
            val = {potential_forces: val}
            break
    potential_gridsizes[key] = replace_ellipsis(val)
for key, d in potential_gridsizes.items():
    for key in d.copy():
        key = canonicalize_input_str(key)
        val = d.pop(key)
        if isinstance(val, dict):
            val = {canonicalize_input_str(key2): val2 for key2, val2 in val.items()}
            for potential_method in potential_methods_implemented:
                val.setdefault(potential_method, -1)
        else:
            val = {
                potential_method: val
                for potential_method in potential_methods_implemented
            }
        d[key] = replace_ellipsis(val)
for key, val in potential_gridsizes.copy().items():
    for potential_force in potential_forces_implemented:
        val.setdefault(potential_force, {})
    potential_gridsizes[key] = val
    for key2, val2 in val.copy().items():
        for potential_method in potential_methods_implemented:
            val2.setdefault(potential_method, -1)
        val2 = {
            key3: val3
            for key3, val3 in val2.items()
            if key3 in potential_forces_implemented[key2]
        }
        val[key2] = val2
for key, val in potential_gridsizes.items():
    if key == 'global':
        for key2, val2 in val.items():
            for key3, val3 in val2.copy().items():
                val3 = set(any2list(val3))
                if len(val3) == 1:
                    val2[key3] = int(round(val3.pop()))
                    continue
                abort(
                    f'Only a single grid size may be specified for a global potential, but '
                    f'potential_options["gridsize"]["{key}"]["{key2}"]["{key3}"] = {val2[key3]}'
                )
    else:
        for key2, val2 in val.items():
            for key3, val3 in val2.copy().items():
                val3 = any2list(val3)
                if len(val3) == 1:
                    val3 *= 2
                if len(val3) == 2:
                    val2[key3] = PotentialGridsizesComponent(
                        val3[0] if isinstance(val3[0], str) else int(round(val3[0])),
                        val3[1] if isinstance(val3[1], str) else int(round(val3[1])),
                    )
                    continue
                abort(
                    f'potential_options["gridsize"]["{key}"]["{key2}"]["{key3}"] = {val2[key3]} '
                    f'but should to be a tuple of two integers, corresponding to the upstream '
                    f'and downstream grid size'
                )
potential_options['gridsize'] = potential_gridsizes
interpolation_orders = {'NGP': 1, 'CIC': 2, 'TSC': 3, 'PCS': 4}
force_interpolations = {
    'gravity': {
        'pm' : 'CIC',
        'p3m': 'CIC',
    },
    'lapse': {
        'pm': 'CIC',
    },
}
for key, val in replace_ellipsis(dict(potential_options.get('interpolation', {}))).items():
    key = key.lower()
    if isinstance(val, dict):
        force_interpolations[key].update({
            subd_key.lower(): subd_val for subd_key, subd_val in replace_ellipsis(val).items()
        })
    elif isinstance(val, (tuple, list)):
        force_interpolations[key][val[0].lower()] = val[1]
    elif isinstance(val, str):
        force_interpolations[key] = {'pm': val.lower(), 'p3m': val.lower()}
    else:
        abort('Could not interpret the potential_options["interpolation"] parameter')
for key, val in force_interpolations.copy().items():
    subd = {}
    for subd_key, subd_val in val.items():
        subd_key = re.sub(r'[ _\-^()]', '', subd_key.lower())
        for n in range(10):
            subd_key = subd_key.replace(unicode_superscript(str(n)), str(n))
        subd_val = int(interpolation_orders.get(str(subd_val).upper(), subd_val))
        subd[subd_key] = subd_val
    force_interpolations[key] = subd
potential_options['interpolation'] = force_interpolations


PotentialUpstreamDownstreamPair = collections.namedtuple(
    'PotentialUpstreamDownstreamPair',
    ('upstream', 'downstream'),
)
force_deconvolutions = {
    'gravity': {
        'pm' : PotentialUpstreamDownstreamPair(True, True),
        'p3m': PotentialUpstreamDownstreamPair(True, True),
    },
    'lapse': {
        'pm' : PotentialUpstreamDownstreamPair(True, True),
    },
}
for key, val in replace_ellipsis(dict(potential_options.get('deconvolve', {}))).items():
    key = key.lower()
    if isinstance(val, dict):
        force_deconvolutions[key].update({
            subd_key.lower(): subd_val for subd_key, subd_val in replace_ellipsis(val).items()
        })
    else:
        force_deconvolutions[key] = {'pm': val, 'p3m': val}
    for key2, val2 in force_deconvolutions[key].copy().items():
        val2 = any2list(val2)
        if len(val2) == 1:
            val2 *= 2
        if len(val2) == 2:
            val2 = PotentialUpstreamDownstreamPair(bool(val2[0]), bool(val2[1]))
        else:
            abort(
                f'potential_options["deconvolve"]["{key}"]["{key2}"] = {val2} '
                f'but should to be a tuple of two Booleans, corresponding to the upstream '
                f'and downstream deconvolution'
            )
        force_deconvolutions[key][key2] = val2
for key, val in force_deconvolutions.copy().items():
    subd = {}
    for subd_key, subd_val in val.items():
        subd_key = re.sub(r'[ _\-^()]', '', subd_key.lower())
        for n in range(10):
            subd_key = subd_key.replace(unicode_superscript(str(n)), str(n))
        subd[subd_key] = subd_val
    force_deconvolutions[key] = subd
potential_options['deconvolve'] = force_deconvolutions
force_interlacings = {
    'gravity': {
        'pm' : PotentialUpstreamDownstreamPair('sc', 'sc'),
        'p3m': PotentialUpstreamDownstreamPair('sc', 'sc'),
    },
    'lapse': {
        'pm' : PotentialUpstreamDownstreamPair('sc', 'sc'),
    },
}
def interlace2latticekind(interlace):
    if isinstance(interlace, (tuple, list)):
        return (interlace2latticekind(interlace[0]), interlace2latticekind(interlace[1]))
    if isinstance(interlace, str):
        return interlace
    if interlace:
        return 'bcc'  # body-centered lattice (2 primitive sc lattices, standard interlacing)
    return 'sc'  # simple cubic lattice (no interlacing)
for key, val in replace_ellipsis(dict(potential_options.get('interlace', {}))).items():
    key = key.lower()
    if isinstance(val, dict):
        force_interlacings[key].update({
            subd_key.lower(): subd_val for subd_key, subd_val in replace_ellipsis(val).items()
        })
    else:
        force_interlacings[key] = {
            'pm' : interlace2latticekind(val),
            'p3m': interlace2latticekind(val),
        }
    for key2, val2 in force_interlacings[key].copy().items():
        val2 = any2list(val2)
        if len(val2) == 1:
            val2 *= 2
        if len(val2) == 2:
            val2 = PotentialUpstreamDownstreamPair(*interlace2latticekind(val2))
        else:
            abort(
                f'potential_options["interlace"]["{key}"]["{key2}"] = {val2} '
                f'but should to be a tuple of two Booleans or strs, corresponding to the upstream '
                f'and downstream interlacing'
            )
        force_interlacings[key][key2] = val2
for key, val in force_interlacings.copy().items():
    subd = {}
    for subd_key, subd_val in val.items():
        subd_key = re.sub(r'[ _\-^()]', '', subd_key.lower())
        for n in range(10):
            subd_key = subd_key.replace(unicode_superscript(str(n)), str(n))
        subd[subd_key] = subd_val
    force_interlacings[key] = subd
potential_options['interlace'] = force_interlacings
potential_differentiations_default = {
    'gravity': {
        'pm' : 2,
        'p3m': 4,
    },
    'lapse': {
        'pm': 2,
    },
}
potential_differentiations = {'default': potential_differentiations_default.copy()}
potential_differentiations_input = replace_ellipsis(
    dict(potential_options.get('differentiation', {}))
)
for name, d in potential_differentiations_input.items():
    potential_differentiations[name] = d.copy()
    for key, val in d.items():
        key = key.lower()
        if isinstance(val, dict):
            potential_differentiations[name][key].update({
                subd_key.lower(): subd_val for subd_key, subd_val in replace_ellipsis(val).items()
            })
        elif isinstance(val, (tuple, list)):
            potential_differentiations[name][key][val[0].lower()] = val[1]
        elif isinstance(val, (int, float, np.integer, np.floating)):
            potential_differentiations[name][key] = {'pm': int(round(val)), 'p3m': int(round(val))}
        elif isinstance(val, str):
            potential_differentiations[name][key] = {'pm': val, 'p3m': val}
        else:
            abort('Could not interpret the potential_options["differentiation"] parameter')
    for key, val in potential_differentiations[name].copy().items():
        subd = {}
        for subd_key, subd_val in val.items():
            subd_key = re.sub(r'[ _\-^()]', '', subd_key.lower())
            for n in range(10):
                subd_key = subd_key.replace(unicode_superscript(str(n)), str(n))
            if isinstance(subd_val, (int, float, np.integer, np.floating)):
                subd_val = int(round(subd_val))
            elif isinstance(subd_val, str):
                subd_val = subd_val.lower()
                if subd_val != 'fourier':
                    abort(f'Invalid potential_options["differentiation"] value {subd_val}')
                # Encode 'fourier' differentiation as a value of zero
                subd_val = 0
            subd[subd_key] = subd_val
        potential_differentiations[name][key] = subd
potential_options['differentiation'] = potential_differentiations
user_params['potential_options'] = potential_options
ewald_gridsize = to_int(user_params.get('ewald_gridsize', 64))
user_params['ewald_gridsize'] = ewald_gridsize
shortrange_params = dict(user_params.get('shortrange_params', {}))
user_specification_involves_gridsize = collections.defaultdict(bool)
for force, d in shortrange_params.items():
    gridsize_in_d_vals = False
    if not isinstance(d, dict):
        continue
    for val in d.values():
        if not isinstance(val, str):
            continue
        if 'gridsize' in val:
            user_specification_involves_gridsize[force] = True
            break
if shortrange_params and not isinstance(list(shortrange_params.values())[0], dict):
    shortrange_params = {'gravity': shortrange_params}  # Gravity defined as the primary tiling
shortrange_params_defaults = {
    'trivial': {
        'scale'    : ထ,
        'range'    : ထ,
        'tilesize' : ထ,
        'subtiling': (1, 1, 1),
        'tablesize': -1,
    },
    'gravity': {
        'scale'    : '1.25*boxsize/gridsize',
        'range'    : '4.5*scale',
        'tilesize' : 'range',
        'subtiling': 'automatic',
        'tablesize': 2**12,
    },
}
for force, d in shortrange_params_defaults.items():
    shortrange_params.setdefault(force, d)
subtiling_refinement_period_default = 16
for force, d in shortrange_params.items():
    for key, val in shortrange_params_defaults.get(force, {}).items():
        d.setdefault(key, val)
    scale = d.get('scale')
    if isinstance(scale, str) and 'N' in scale:
        scale = d['scale'] = {'all': scale}
    keys = ['scale', 'range', 'tilesize']
    if isinstance(scale, dict):
        keys.remove('scale')
        forcerange = d.get('range')
        if isinstance(forcerange, str) and 'scale' in forcerange:
            d['range'] = {'all combinations': forcerange}
            keys.remove('range')
    for key in keys:
        val = d.get(key)
        if val is None:
            continue
        if not isinstance(val, str):
            continue
        val = val.replace('boxsize', str(boxsize))
        if 'gridsize' in val:
            gridsize = potential_options['gridsize']['global'][force]['p3m']
            if gridsize == -1:
                if user_specification_involves_gridsize[force]:
                    abort(
                        f'Could not detect grid size needed for '
                        f'shortrange_params["{force}"]["{key}"]. '
                        f'You should specify it by setting the '
                        f'potential_options["gridsize"]["global"]["{force}"]["p3m"] parameter.'
                    )
            val = val.replace('gridsize', str(gridsize))
        if 'scale' in val:
            scale = d.get('scale')
            if not isinstance(scale, (int, float, np.integer, np.floating)):
                abort(
                    f'Could not detect scale needed for '
                    f'shortrange_params["{force}"]["{key}"]. '
                    f'Is the scale specified in shortrange_params["{force}"]?'
                )
            if 'scale_i' in val:
                val = val.replace('scale_i', str(scale))
            if 'scale_j' in val:
                val = val.replace('scale_j', str(scale))
            if 'scale' in val:
                val = val.replace('scale', str(scale))
        if 'range' in val:
            forcerange = d.get('range')
            if not isinstance(forcerange, (int, float, np.integer, np.floating)):
                abort(
                    f'Could not detect range needed for '
                    f'shortrange_params["{force}"]["{key}"]. '
                    f'Is the range specified in shortrange_params["{force}"]?'
                )
            val = val.replace('range', str(forcerange))
        d[key] = eval_unit(val)
    subtiling = d.get('subtiling', (1, 1, 1))
    if isinstance(subtiling, str):
        if subtiling.lower().startswith('auto'):
            d['subtiling'] = ('automatic', subtiling_refinement_period_default)
        else:
            abort(
                f'Could not understand subtiling = "{subtiling}" of shortrange_params["{force}"]'
            )
    else:
        subtiling = tuple(any2list(subtiling))
        if len(subtiling) == 1:
            subtiling = subtiling[0]
            if isinstance(subtiling, str):
                if subtiling.lower().startswith('auto'):
                    subtiling = ('automatic', subtiling_refinement_period_default)
            else:
                subtiling = (int(subtiling),)*3
        elif len(subtiling) == 2:
            if isinstance(subtiling[1], str):
                subtiling = (subtiling[1], int(subtiling[0]))
            if isinstance(subtiling[0], str) and subtiling[0].lower().startswith('auto'):
                subtiling = ('automatic', int(subtiling[1]))
        d['subtiling'] = subtiling
    tablesize = int(round(d.get('tablesize', -1)))
    d['tablesize'] = tablesize
user_params['shortrange_params'] = shortrange_params
powerspec_options_defaults = {
    'upstream gridsize': {
        'default': -1,
    },
    'global gridsize': {
        'default': -1,
    },
    'interpolation': {
        'default': 'PCS',
    },
    'deconvolve': {
        'default': True,
    },
    'interlace': {
        'default': True,
    },
    'realization correction': {
        'default': True,
    },
    'k_max': {
        'default': 'Nyquist',
    },
    'bins per decade': {
        'default': {
            '  4*k_min':  4,
            '100*k_min': 40,
        },
    },
    'tophat': {
        'default': '8*Mpc/h',
    },
    'significant figures': {
        'default': 8,
    },
}
powerspec_options = dict(user_params.get('powerspec_options', {}))
for key, val in powerspec_options.items():
    replace_ellipsis(val)
if 'gridsize' in powerspec_options:
    d = powerspec_options['gridsize']
    if not isinstance(d, dict):
        d = {'default': d}
    powerspec_options.setdefault('upstream gridsize', d.copy())
    powerspec_options.setdefault('global gridsize'  , d.copy())
    powerspec_options.pop('gridsize')
for key, d in powerspec_options.copy().items():
    if not isinstance(d, dict):
        powerspec_options[key] = {'default': d}
for key, d_defaults in powerspec_options_defaults.items():
    powerspec_options.setdefault(key, {})
    d = powerspec_options[key]
    for key, val in d_defaults.items():
        d.setdefault(key, val)
d = powerspec_options['global gridsize']
for key, val in d.copy().items():
    d[key] = int(round(val))
d = powerspec_options['interpolation']
for key, val in d.copy().items():
    d[key] = int(interpolation_orders.get(str(val).upper(), val))
d = powerspec_options['interlace']
for key, val in d.copy().items():
    d[key] = interlace2latticekind(val)
d = powerspec_options['realization correction']
for key, val in d.copy().items():
    if isinstance(val, str):
        d[key] = bool(val)
d = powerspec_options['k_max']
for key, val in d.copy().items():
    if isinstance(val, str):
        d[key] = val.replace('Nyquist', 'nyquist')
d = powerspec_options['bins per decade']
for key, val in d.copy().items():
    if not isinstance(val, dict):
        val = {1: val}
    replace_ellipsis(val)
    if len(val) == 1:
        val.update({
            (f'{key} + 1' if isinstance(key, str) else key + 1): val2
            for key, val2 in val.items()
        })
    d[key] = val
for key in powerspec_options:
    if key not in powerspec_options_defaults:
        abort(f'powerspec_options["{key}"] not implemented')
user_params['powerspec_options'] = powerspec_options
bispec_options_defaults = {
    'configuration': {
        'default': ('equilateral', 20),
    },
    'shellthickness': {
        'default': [
            {
                '1*k_fundamental': '0.25*k_fundamental',
                '4*k_fundamental': 'max(3*k_fundamental, 1/20*log(10)*k)',
            },
        ]*3,
    },
    'upstream gridsize': {
        'default': -1,
    },
    'global gridsize': {
        'default': -1,
    },
    'interpolation': {
        'default': 'PCS',
    },
    'deconvolve': {
        'default': True,
    },
    'interlace': {
        'default': True,
    },
    'significant figures': {
        'default': 8,
    },
}
bispec_options = dict(user_params.get('bispec_options', {}))
for key, val in bispec_options.items():
    replace_ellipsis(val)
if 'gridsize' in bispec_options:
    d = bispec_options['gridsize']
    if not isinstance(d, dict):
        d = {'default': d}
    bispec_options.setdefault('upstream gridsize', d.copy())
    bispec_options.setdefault('global gridsize'  , d.copy())
    bispec_options.pop('gridsize')
for key, d in bispec_options.copy().items():
    if not isinstance(d, dict):
        bispec_options[key] = {'default': d}
for key, d_defaults in bispec_options_defaults.items():
    bispec_options.setdefault(key, {})
    d = bispec_options[key]
    for key, val in d_defaults.items():
        d.setdefault(key, val)
d = bispec_options['global gridsize']
for key, val in d.copy().items():
    d[key] = int(round(val))
d = bispec_options['configuration']
replace_ellipsis(d)
def transform(key, val):
    n = 20
    if isinstance(val, (tuple, list)) and len(val) == 2:
        if not isinstance(val[0], str) and not isinstance(val[1], str):
            abort(
                f'bispec_options["configuration"]["{key}"] contains '
                f'{val} with 2 elements, none of which is a str'
            )
        if isinstance(val[1], str):
            val = (val[1], val[0])
        val, n = val
    if isinstance(val, str):
        # Format: str, together with n
        val = (val, n)
    elif isinstance(val, dict):
        # Format: {'k': ..., 't': ..., 'μ': ...}
        replace_ellipsis(val)
        d = {}
        for key2, val2 in val.copy().items():
            key2 = key2.replace('k_', 'k')
            key2 = key2.replace('1', '')
            key2 = key2.replace(unicode('₁'), '').replace(asciify('₁'), '')
            key2 = key2.replace(unicode('¹'), '').replace(asciify('¹'), '')
            key2 = key2.replace(unicode('μ'), 'μ').replace(asciify('μ'), 'μ').replace('mu', 'μ')
            if key2 not in ('k', 't', 'μ'):
                abort(f'Key "{key2}" in bispec_options["configuration"]["{key}"] not understood')
            d[key2] = val2
        for key2 in ('k', 't', 'μ'):
            if key2 not in d:
                abort(
                    f'Mapping {val} in bispec_options["configuration"]["{key}"] '
                    f'is missing required key "{key2}"'
                )
        val = d
    elif isinstance(val, (tuple, list)):
        if len(val) == 3:
            # Format: (k, t, μ)
            if not all(
                [isinstance(el, (int, float, np.integer, np.floating, str)) for el in val]
            ):
                abort(
                    f'bispec_options["configuration"]["{key}"] contains '
                    f'{val} which has an element which is not a number or a str'
                )
        else:
            abort(
                f'bispec_options["configuration"]["{key}"] contains '
                f'{val} which has {len(val)} ≠ 3 elements'
            )
        val = tuple(val)
    else:
        abort(
            f'Failed to parse bispec_options["configuration"]["{key}"] = '
            f'{val} of type {type(val)}'
        )
    return val
for key, val in d.copy().items():
    if isinstance(val, (tuple, list)):
        if isinstance(val, tuple) and len(val) in {2, 3} and all(
            [isinstance(el, (int, float, np.integer, np.floating, str)) for el in val]
        ):
            val = [transform(key, tuple(val))]
        else:
            val = list(val)
            for i, el in enumerate(val):
                val[i] = transform(key, el)
    else:
        val = [transform(key, val)]
    d[key] = val
d = bispec_options['shellthickness']
replace_ellipsis(d)
for key, val in d.copy().items():
    if isinstance(val, (int, float, np.integer, np.floating)):
        val = {1: val}
    elif isinstance(val, str):
        val = {0: val, ထ: val}  # signals that this should be applied from k_min to k_max
    if isinstance(val, dict):
        val = [val]
    if isinstance(val, (tuple, list)):
        if len(val) == 1:
            val *= 3
        if len(val) != 3:
            abort(
                f'You have specified {len(val)} shell thicknesses in '
                f'bispec_options["shellthickness"]["{key}"]. Please specify 3.'
            )
    val = list(val)
    for i, d2 in enumerate(val):
        if isinstance(d2, (int, float, np.integer, np.floating, str)):
            d2 = {1: d2}  # modified below
        replace_ellipsis(d2)
        if len(d2) == 1:
            key2 = tuple(d2.keys())[0]
            val2 = tuple(d2.values())[0]
            if isinstance(val2, str):
                d2.clear()
                d2 |= {0: val2, ထ: val2}
            else:
                if isinstance(key2, str):
                    d2[f'{key2} + 1'] = val2
                else:
                    d2[key2 + 1] = val2
        val[i] = d2
    d[key] = tuple(val)
d = bispec_options['interpolation']
for key, val in d.copy().items():
    d[key] = int(interpolation_orders.get(str(val).upper(), val))
d = bispec_options['interlace']
for key, val in d.copy().items():
    d[key] = interlace2latticekind(val)
for key in bispec_options:
    if key not in bispec_options_defaults:
        abort(f'bispec_options["{key}"] not implemented')
user_params['bispec_options'] = bispec_options
bispec_antialiasing = bool(user_params.get('bispec_antialiasing', True))
user_params['bispec_antialiasing'] = bispec_antialiasing
class_dedicated_spectra = bool(user_params.get('class_dedicated_spectra', False))
user_params['class_dedicated_spectra'] = class_dedicated_spectra
if isinstance(
    user_params.get('class_modes_per_decade', {}),
    (int, float, np.integer, np.floating),
):
    class_modes_per_decade = {1: user_params['class_modes_per_decade']}
else:
    class_modes_per_decade = dict(user_params.get('class_modes_per_decade', {}))
    replace_ellipsis(class_modes_per_decade)
    if not class_modes_per_decade:
        class_modes_per_decade = {
            3e-3*units.Mpc**(-1): 10,
            3e-2*units.Mpc**(-1): 30,
            3e-1*units.Mpc**(-1): 30,
            1e+0*units.Mpc**(-1): 10,
        }
if len(class_modes_per_decade) == 1:
    class_modes_per_decade.update({(key + 1): val for key, val in class_modes_per_decade.items()})
user_params['class_modes_per_decade'] = class_modes_per_decade
# Cosmology
H0 = float(user_params.get('H0', 67*units.km/(units.s*units.Mpc)))
user_params['H0'] = H0
Ωb = float(user_params.get('Ωb', 0.049))
user_params['Ωb'] = Ωb
Ωcdm = float(user_params.get('Ωcdm', 0.27))
user_params['Ωcdm'] = Ωcdm
a_begin = float(user_params.get('a_begin', 1))
user_params['a_begin'] = a_begin
t_begin = float(user_params.get('t_begin', 0))
user_params['t_begin'] = t_begin
primordial_spectrum = dict(user_params.get('primordial_spectrum', {}))
replace_ellipsis(primordial_spectrum)
for key, val in primordial_spectrum.copy().items():
    key_ori = key
    key = unicode(key).lower()
    for c in '_sk':
        key = key.replace(c, '')
    key = key.replace('alpha', 'α').replace(unicode('α'), 'α').replace(asciify('α'), 'α')
    for shortened_key, full_key in {'a': 'A_s', 'n': 'n_s', 'α': 'α_s', 'pivot': 'pivot'}.items():
        if shortened_key == key:
            primordial_spectrum[full_key] = val
            break
    else:
        masterwarn(f'Could not understand primordial spectrum parameter "{key_ori}"')
primordial_spectrum = {
    key: float(primordial_spectrum.get(key, val))
    for key, val in {'A_s': 2.1e-9, 'n_s': 0.96, 'α_s': 0, 'pivot': 0.05/units.Mpc}.items()
}
user_params['primordial_spectrum'] = primordial_spectrum
class_params = dict(user_params.get('class_params', {}))
replace_ellipsis(class_params)
user_params['class_params'] = class_params
enable_warm_dark_matter = bool(float(user_params.get('enable_warm_dark_matter', False)))
# Physics
default_force_method = {
    'gravity': 'p3m',
    'lapse'  : 'pm',
}
methods_implemented = ('ppnonperiodic', 'pp', 'p3m', 'pm')
select_forces = {}
for key, val in replace_ellipsis(dict(user_params.get('select_forces', {}))).items():
    key = key.lower()
    if isinstance(val, dict):
        select_forces[key] = replace_ellipsis(val)
    elif isinstance(val, str):
        select_forces[key] = {val: default_force_method[val.lower()]}
    subd = {}
    for subd_key, method in select_forces[key].items():
        subd[canonicalize_input_str(subd_key)] = canonicalize_input_str(method)
    select_forces[key] = subd
if not select_forces:
    for key_force, dict_method in potential_options['gridsize']['global'].items():
        methods = {
            key_method for key_method, gridsize in dict_method.items() if gridsize != -1
        }
        if not methods:
            continue
        select_forces.setdefault('particles', {})
        select_forces.setdefault('fluid', {})
        if methods == {'pm'}:
            select_forces['particles'].setdefault(key_force, 'pm')
            select_forces['fluid'    ].setdefault(key_force, 'pm')
        elif methods in ({'p3m'}, {'pm', 'p3m'}):
            select_forces['particles'].setdefault(key_force, 'p3m')
            select_forces['fluid'    ].setdefault(key_force, 'pm')
        else:
            abort(
                f'Force methods "{methods}" from potential_options["gridsize"]["global"]["{key_force}""] '
                f'not understood'
            )
select_forces.setdefault('metric', {'gravity': 'pm'})
select_forces.setdefault('lapse', {'lapse': 'pm'})
user_params['select_forces'] = select_forces
select_species = {}
if user_params.get('select_species'):
    if isinstance(user_params['select_species'], dict):
        select_species = user_params['select_species']
        replace_ellipsis(select_species)
    else:
        select_species = {'all': str(user_params['select_species'])}
select_species.setdefault('default', 'matter')
user_params['select_species'] = select_species
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
select_boltzmann_order = {}
if user_params.get('select_boltzmann_order'):
    if isinstance(user_params['select_boltzmann_order'], dict):
        select_boltzmann_order = user_params['select_boltzmann_order']
        replace_ellipsis(select_boltzmann_order)
    else:
        select_boltzmann_order = {'all': int(user_params['select_boltzmann_order'])}
select_boltzmann_order['default'] = 1
select_boltzmann_order.setdefault('metric', -1)
select_boltzmann_order.setdefault('lapse', -1)
user_params['select_boltzmann_order'] = select_boltzmann_order
select_boltzmann_closure = {}
if user_params.get('select_boltzmann_closure'):
    if isinstance(user_params['select_boltzmann_closure'], dict):
        select_boltzmann_closure = user_params['select_boltzmann_closure']
        replace_ellipsis(select_boltzmann_closure)
    else:
        select_boltzmann_closure = {'all': str(user_params['select_boltzmann_closure'])}
select_boltzmann_closure['default'] = 'class'
user_params['select_boltzmann_closure'] = select_boltzmann_closure
realization_options_defaults = {
    'gauge': {
        'default': 'N-body',
    },
    'backscale': {
        'default': False,
    },
    'lpt': {
        'default': 1,
    },
    'dealias': {
        'default': False,
    },
    'nongaussianity': {
        'default': 0,
    },
    'structure': {
        'default': 'non-linear',
    },
    'compound': {
        'default': 'linear',
    },
}
realization_options = dict(user_params.get('realization_options', {}))
for key, val in realization_options.items():
    replace_ellipsis(val)
for key, d in realization_options.copy().items():
    if not isinstance(d, dict):
        realization_options[key] = {'default': d}
def simplify_str(s):
    if isinstance(s, str):
        return s.lower().replace('-', '').replace('_', '').replace(' ', '')
    return s
for key, d in realization_options.copy().items():
    realization_options.pop(key)
    realization_options[simplify_str(key)] = {
        key2: simplify_str(val2)
        for key2, val2 in d.items()
    }
for key, d_defaults in realization_options_defaults.items():
    realization_options.setdefault(simplify_str(key), {})
    d = realization_options[simplify_str(key)]
    for key, val in d_defaults.items():
        d.setdefault(simplify_str(key), simplify_str(val))
d = realization_options['gauge']
for key, val in d.copy().items():
    val = str(val)
    if val.startswith('nb'):
        val = 'nbody'
    elif val.startswith('syn'):
        val = 'synchronous'
    elif val.startswith('newt'):
        val = 'newtonian'
    d[key] = val
d = realization_options['backscale']
for key, val in d.copy().items():
    d[key] = bool(val)
d = realization_options['lpt']
for key, val in d.copy().items():
    d[key] = int(round(val))
    if d[key] not in {1, 2, 3}:
        abort(f'{d[key]}LPT not implemented')
d = realization_options['dealias']
for key, val in d.copy().items():
    d[key] = bool(val)
for s in ('nongauss', 'nongaussian', 'nongaussianity'):
    if s not in realization_options:
        continue
    d = realization_options.pop(s)
    for key, val in d.copy().items():
        d[key] = float(val)
    realization_options['nongaussianity'] = d
    break
d = realization_options['structure']
for key, val in d.copy().items():
    if val not in {'primordial', 'nonlinear'}:
        abort(
            f'Unrecognised value "{val}" ∉ {{"primordial", "non-linear"}} '
            f'for realization_options["structure"]["{key}"]'
        )
d = realization_options['compound']
for key, val in d.copy().items():
    if val not in {'linear', 'nonlinear'}:
        abort(
            f'Unrecognised value "{val}" ∉ {{"linear", "non-linear"}} '
            f'for realization_options["compound"]["{key}"]'
        )
for key in realization_options:
    if key not in realization_options_defaults:
        abort(f'realization_options["{key}"] not implemented')
user_params['realization_options'] = realization_options
select_lives = {}
if user_params.get('select_lives'):
    if isinstance(user_params['select_lives'], dict):
        select_lives = user_params['select_lives']
        replace_ellipsis(select_lives)
    else:
        select_lives = {'all': user_params['select_lives']}
    replace_ellipsis(select_lives)
else:
    select_lives = {'all': (0, ထ)}
for key, val in select_lives.copy().items():
    val = tuple([float(el) for el in sorted(any2list(val))])
    if len(val) != 2:
        abort(f'select_lives["{key}"] = {select_lives[key]} not understood')
    select_lives[key] = val
select_lives['default'] = (0, ထ)
user_params['select_lives'] = select_lives
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
softening_kernel = user_params.get('softening_kernel', 'spline')
softening_kernel = softening_kernel.lower()
user_params['softening_kernel'] = softening_kernel
select_softening_length = {}
if user_params.get('select_softening_length'):
    if isinstance(user_params['select_softening_length'], dict):
        select_softening_length = user_params['select_softening_length']
        replace_ellipsis(select_softening_length)
    else:
        select_softening_length = {'all': user_params['select_softening_length']}
select_softening_length.setdefault('default', '0.025*boxsize/cbrt(N)')
user_params['select_softening_length'] = select_softening_length
# Simulation options
Δt_base_background_factor = float(user_params.get('Δt_base_background_factor', 1))
user_params['Δt_base_background_factor'] = Δt_base_background_factor
Δt_base_nonlinear_factor = float(user_params.get('Δt_base_nonlinear_factor', 1))
user_params['Δt_base_nonlinear_factor'] = Δt_base_nonlinear_factor
Δt_increase_max_factor = float(user_params.get('Δt_increase_max_factor', ထ))
user_params['Δt_increase_max_factor'] = Δt_increase_max_factor
Δt_rung_factor = float(user_params.get('Δt_rung_factor', 1))
user_params['Δt_rung_factor'] = Δt_rung_factor
Δa_max_early = float(user_params.get('Δa_max_early', 0.00153))
user_params['Δa_max_early'] = Δa_max_early
Δa_max_late = float(user_params.get('Δa_max_late', 0.022))
user_params['Δa_max_late'] = Δa_max_late
static_timestepping = user_params.get('static_timestepping')
if static_timestepping == '':
    static_timestepping = None
user_params['static_timestepping'] = static_timestepping
N_rungs = int(user_params.get('N_rungs', 8))
user_params['N_rungs'] = N_rungs
fftw_wisdom_rigor = user_params.get('fftw_wisdom_rigor', 'measure').lower()
user_params['fftw_wisdom_rigor'] = fftw_wisdom_rigor
fftw_wisdom_reuse = bool(user_params.get('fftw_wisdom_reuse', True))
user_params['fftw_wisdom_reuse'] = fftw_wisdom_reuse
fftw_wisdom_share = bool(user_params.get('fftw_wisdom_share', False))
user_params['fftw_wisdom_share'] = fftw_wisdom_share
random_generator = user_params.get('random_generator', 'PCG64DXSM')
user_params['random_generator'] = random_generator
random_seeds_default = {
    'general'              :     0,
    'primordial amplitudes': 1_000,
    'primordial phases'    : 2_000,
}
random_seeds = user_params.get('random_seeds', {})
for key, val in random_seeds_default.items():
    random_seeds.setdefault(key, val)
    random_seeds[key] = to_int(random_seeds[key])
for key in random_seeds.keys():
    if key not in random_seeds_default:
        abort(f'Key {key} in random_seeds not understood')
primordial_noise_imprinting = str(
    user_params.get('primordial_noise_imprinting', 'distributed')
).lower()
if primordial_noise_imprinting not in {'simple', 'distributed'}:
    abort(
        f'primordial_noise_imprinting = "{primordial_noise_imprinting}" '
        f'∉ {{"simple", "distributed"}}'
    )
user_params['primordial_noise_imprinting'] = primordial_noise_imprinting
primordial_amplitude_fixed = bool(user_params.get('primordial_amplitude_fixed', False))
user_params['primordial_amplitude_fixed'] = primordial_amplitude_fixed
primordial_phase_shift = np.mod(float(user_params.get('primordial_phase_shift', 0)), τ)
user_params['primordial_phase_shift'] = primordial_phase_shift
cell_centered = bool(user_params.get('cell_centered', True))
user_params['cell_centered'] = cell_centered
fourier_structure_caching = {'primordial': True, 'all': True}
if 'fourier_structure_caching' in user_params:
    if isinstance(user_params['fourier_structure_caching'], dict):
        fourier_structure_caching = user_params['fourier_structure_caching']
        replace_ellipsis(fourier_structure_caching)
    else:
        fourier_structure_caching = {
            'primordial': bool(user_params['fourier_structure_caching']),
            'all'       : bool(user_params['fourier_structure_caching']),
        }
fourier_structure_caching['default'] = True
user_params['fourier_structure_caching'] = fourier_structure_caching
fluid_scheme_select = {'all': 'MacCormack'}
if user_params.get('fluid_scheme_select'):
    if isinstance(user_params['fluid_scheme_select'], dict):
        fluid_scheme_select = user_params['fluid_scheme_select']
        replace_ellipsis(fluid_scheme_select)
    else:
        fluid_scheme_select = {'all': user_params['fluid_scheme_select']}
fluid_scheme_select['default'] = 'MacCormack'
fluid_scheme_select = {
    key: str(val).lower().replace(' ', '').replace('-', '')
    for key, val in fluid_scheme_select.items()
}
user_params['fluid_scheme_select'] = fluid_scheme_select
fluid_options_defaults = {
    'maccormack': {
        'vacuum_corrections_select'    : True,
        'max_vacuum_corrections_select': (1, 'gridsize'),
        'foresight_select'             : 25,
        'smoothing_select'             : {
            'default': 1.0,
            # Matter fluids require a lot of smoothing
            'baryon'                   : 2.0,
            'cold dark matter'         : 2.0,
            'decaying cold dark matter': 2.0,
            'matter'                   : 2.0,
        },
    },
    'kurganovtadmor': {
        'rungekuttaorder': 2,
        'flux_limiter_select': {
            'default': 'mc',
            # Matter fluids require a lot
            # of artificial viscosity (smoothing).
            'baryon'                   : 'minmod',
            'cold dark matter'         : 'minmod',
            'decaying cold dark matter': 'minmod',
            'matter'                   : 'minmod',
        },
    },
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
        if not isinstance(val, dict):
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
for key, val in class_k_max.copy().items():
    class_k_max[key] = any2list(val)[0]
user_params['class_k_max'] = class_k_max
class_reuse = bool(user_params.get('class_reuse', True))
user_params['class_reuse'] = class_reuse
# Graphics
terminal_width = to_int(user_params.get('terminal_width', 80))
user_params['terminal_width'] = terminal_width
enable_terminal_formatting = bool(user_params.get('enable_terminal_formatting', True))
user_params['enable_terminal_formatting'] = enable_terminal_formatting
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
bispec_plot_prefer = str(user_params.get('bispec_plot_prefer', 't'))
if bispec_plot_prefer.lower() == 'mu':
    bispec_plot_prefer = 'μ'
bispec_plot_prefer = unicode(bispec_plot_prefer)
if bispec_plot_prefer not in ('t', unicode('μ')):
    abort(f'bispec_plot_prefer = "{bispec_plot_prefer}" ∉ {{"t", "μ"}}')
user_params['bispec_plot_prefer'] = bispec_plot_prefer
render2D_options_defaults = {
    'upstream gridsize': {
        'default': -1,
    },
    'global gridsize': {
        'default': -1,
    },
    'terminal resolution': {
        'default': -1,
    },
    'interpolation': {
        'default': 'PCS',
    },
    'deconvolve': {
        'default': False,
    },
    'interlace': {
        'default': False,
    },
    'axis': {
        'default': 'z',
    },
    'extent': {
        'default': (0, 0.1*boxsize),
    },
    'colormap': {
        'default': 'inferno',
    },
    'enhance': {
        'default': True,
    },
}
render2D_options = dict(user_params.get('render2D_options', {}))
for key, val in render2D_options.items():
    replace_ellipsis(val)
if 'gridsize' in render2D_options:
    d = render2D_options['gridsize']
    if not isinstance(d, dict):
        d = {'default': d}
    render2D_options.setdefault('upstream gridsize', d.copy())
    render2D_options.setdefault('global gridsize'  , d.copy())
    render2D_options.pop('gridsize')
for key, d in render2D_options.copy().items():
    if not isinstance(d, dict):
        render2D_options[key] = {'default': d}
for key, d_defaults in render2D_options_defaults.items():
    render2D_options.setdefault(key, {})
    d = render2D_options[key]
    for key, val in d_defaults.items():
        d.setdefault(key, val)
d = render2D_options['global gridsize']
for key, val in d.copy().items():
    d[key] = int(round(val))
d = render2D_options['terminal resolution']
for key, val in d.copy().items():
    d[key] = int(round(val))
d = render2D_options['interpolation']
for key, val in d.copy().items():
    d[key] = int(interpolation_orders.get(str(val).upper(), val))
d = render2D_options['interlace']
for key, val in d.copy().items():
    d[key] = interlace2latticekind(val)
d = render2D_options['axis']
for key, val in d.copy().items():
    d[key] = val.lower()
d = render2D_options['extent']
for key, val in d.copy().items():
    val = tuple(any2list(val))
    if len(val) == 1:
        val = (0, val[0])
    d[key] = (np.min(val), np.max(val))
for key in render2D_options:
    if key not in render2D_options_defaults:
        abort(f'render2D_options["{key}"] not implemented')
user_params['render2D_options'] = render2D_options
def generate_enhancement_default():
    enhancement = {}
    def _(t, a):  # must not be done with a lambda
        return 0.5
    enhancement['contrast'] = _
    def _(t, a):  # must not be done with a lambda
        return 0.45, 1 - 1e-6*a
    enhancement['clip'] = _
    def _(t, a):  # must not be done with a lambda
        return 1
    enhancement['α'] = _
    def _(t, a):  # must not be done with a lambda
        return 0.35
    enhancement['brightness'] = _
    return enhancement
render3D_options_defaults = {
    'upstream gridsize': {
        'default': -1,
    },
    'global gridsize': {
        'default': -1,
    },
    'interpolation': {
        'default': 'PCS',
    },
    'deconvolve': {
        'default': False,
    },
    'interlace': {
        'default': False,
    },
    'elevation': {
        'default': π/6,
    },
    'azimuth': {
        'default': -π/3,
    },
    'roll': {
        'default': 0,
    },
    'zoom': {
        'default': 1.05,
    },
    'projection': {
        'default': 'perspective',
    },
    'color': {
        'default': None,
    },
    'depthshade': {
        'default': True,
    },
    'enhancement': {
        'default': generate_enhancement_default(),
    },
    'background': {
        'default': 'black',
    },
    'fontsize': {
        'default': '0.022*resolution',
    },
    'resolution': {
        'default': 1080,
    },
}
render3D_options = dict(user_params.get('render3D_options', {}))
for key, val in render3D_options.items():
    replace_ellipsis(val)
if 'gridsize' in render3D_options:
    d = render3D_options['gridsize']
    if not isinstance(d, dict):
        d = {'default': d}
    render3D_options.setdefault('upstream gridsize', d.copy())
    render3D_options.setdefault('global gridsize'  , d.copy())
    render3D_options.pop('gridsize')
for key, d in render3D_options.copy().items():
    if not isinstance(d, dict):
        render3D_options[key] = {'default': d}
for key, d_defaults in render3D_options_defaults.items():
    render3D_options.setdefault(key, {})
    d = render3D_options[key]
    for key, val in d_defaults.items():
        d.setdefault(key, val)
d = render3D_options['global gridsize']
for key, val in d.copy().items():
    d[key] = int(round(val))
d = render3D_options['interpolation']
for key, val in d.copy().items():
    if val is None:
        val = 0
    d[key] = int(interpolation_orders.get(str(val).upper(), val))
d = render3D_options['interlace']
for key, val in d.copy().items():
    d[key] = interlace2latticekind(val)
d = render3D_options['projection']
for key, val in d.copy().items():
    focal_length = 1
    if isinstance(val, str):
        projection = val
    else:
        if isinstance(val[1], str):
            val = (val[1], val[0])
        projection, focal_length = val
    projection = projection.lower()
    if projection.startswith('ortho'):  # orthographic / orthogonal
        d[key] = ('ortho', None)
    elif projection.startswith('persp'):  # perspective
        d[key] = ('persp', focal_length)
    else:
        abort(
            f'Got render3D_options["projection"]["{key}"] = "{projection}" '
            f'∉ {{"orthographic", "perspective"}}'
        )
d = render3D_options['depthshade']
for key, val in d.copy().items():
    val = bool(val)
d = render3D_options['enhancement']
def make_callable(val):
    if callable(val):
        return val
    val = any2list(val)
    if len(val) == 1:
        def _(a, val=val[0]):
            return val
        return _
    return type(val)([make_callable(v) for v in val])
for key, d2 in d.items():
    for key2, val2 in d2.copy().items():
        val2 = make_callable(val2)
        d2[key2] = val2
        if key2 in (unicode('α'), asciify('α'), 'alpha'):
            d2[unicode('α')] = d2[asciify('α')] = val2
        elif key2 not in ('brightness', 'contrast', 'clip'):
            abort(f'Unrecognised render3D_options["enhancement"] key = "{key2}"')
    for key2, val2 in generate_enhancement_default().items():
        d2.setdefault(key2, val2)
d = render3D_options['background']
for key, val in d.copy().items():
    if str(val).lower() in {'none', 'transparent'}:
        d[key] = None
    else:
        d[key] = to_rgb(val, collective=True)
d = render3D_options['resolution']
for key, val in d.copy().items():
    val = int(round(val))
for key in render3D_options:
    if key not in render3D_options_defaults:
        abort(f'render3D_options["{key}"] not implemented')
user_params['render3D_options'] = render3D_options
# Debugging options
print_load_imbalance = user_params.get('print_load_imbalance', True)
if isinstance(print_load_imbalance, str):
    print_load_imbalance = print_load_imbalance.lower()
user_params['print_load_imbalance'] = print_load_imbalance
allow_snapshot_multifile_singleload = user_params.get(
    'allow_snapshot_multifile_singleload', False,
)
user_params['allow_snapshot_multifile_singleload'] = allow_snapshot_multifile_singleload
particle_reordering = user_params.get('particle_reordering', True)
if isinstance(particle_reordering, str):
    particle_reordering = particle_reordering.lower()
user_params['particle_reordering'] = particle_reordering
enable_Hubble = bool(user_params.get('enable_Hubble', True))
user_params['enable_Hubble'] = enable_Hubble
enable_class_background = bool(user_params.get('enable_class_background', enable_Hubble))
user_params['enable_class_background'] = enable_class_background
# Additional, hidden parameters (parameters not supplied by the user)
special_params = dict(user_params.get('special_params', {}))
user_params['special_params'] = special_params



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
    # Current scale factor, cosmic time and time step
    a=('double', a_begin),
    t=('double', t_begin),
    time_step=('Py_ssize_t', 0),
    # Initial time of simulation
    a_begin=('double', a_begin),
    t_begin=('double', t_begin),
    z_begin=('double', (ထ if a_begin == 0 else 1/a_begin - 1)),
    # '+'-separated strings of CO𝘕CEPT/CLASS species
    # present in the simulation.
    species_present='char*',
    class_species_present='char*',
)



############################################
# Derived and internally defined constants #
############################################
cython.declare(
    nghosts='int',
    ρ_crit='double',
    Ωdcdm='double',
    Ωm='double',
    Ωwdm='double',
    ρ_mbar='double',
    matter_class_species=str,
    radiation_class_species=str,
    neutrino_class_species=str,
    massive_neutrino_class_species=str,
    dark_energy_class_species=str,
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
    for kind in output_kinds:
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
# The number of ghost point layers around the domain grids (so that the
# full shape of each grid is (nghosts + shape[0] + nghosts,
# nghosts + shape[1] + nghosts, nghosts + shape[2] + nghosts). This is
# determined by the interpolation orders of potential, power spectrum,
# bispectrum, render2D and render3D interpolations (order 1 (NGP): 0 ghost
# layers, 2 (CIC): 1 ghost layer, order 3 (TSC): 1 ghost layer,
# order 4 (PCS): 2 ghost layers), as well as force differentiations
# (order 1: 1 ghost layer, order 2: 1 ghost layer, order 4: 2 ghost
# layers, order 6: 3 ghost layers, order 8: 4 ghost layers).
# One additional ghost layer is required for odd order interpolations in
# the case of grid interlacing (as the particles are shifted by half a
# grid cell), or if cell-vertex discretisation is used while applying
# interlacing. Finally, second-order differentiation is used to compute
# fluid source terms, and so nghosts should always be at least 1.
nghosts = 0
for options in (powerspec_options, bispec_options, render2D_options, render3D_options):
    interpolation_order_option = np.max(list(options['interpolation'].values()))
    interlace_option = any([val != 'sc' for val in options['interlace'].values()])
    nghosts_option = interpolation_order_option//2
    if interlace_option and (interpolation_order_option%2 != 0 or not cell_centered):
        nghosts_option += 1
    nghosts = np.max([nghosts, nghosts_option])
for force, d in potential_options['interpolation'].items():
    for method, force_interpolation in d.items():
        nghosts = np.max([
            nghosts,
            force_interpolation//2 + (
                potential_options['interlace'][force][method] != ('sc', 'sc')
                and force_interpolation%2 != 0
            )
        ])
for name, d0 in potential_options['differentiation'].items():
    for force, d1 in d0.items():
        nghosts = np.max([nghosts, (np.max(tuple(d1.values())) + 1)//2])
if nghosts < 1:
    nghosts = 1
# The average, comoving density (the critical
# comoving density since we only study flat universes).
ρ_crit = 3*H0**2/(8*π*G_Newton)
# The density parameter for all matter
Ωdcdm = Ωwdm = 0
if any([key in class_params for key in
    {'Omega_dcdmdr', 'omega_dcdmdr', 'Omega_ini_dcdm', 'omega_ini_dcdm'}
]):
    # Decaying dark matter is present. We compute its contribution to
    # the total matter density as though it was stable. Though wrong,
    # the relation ρ_dcdm(a) = Ωdcdm/a**3 will hold at early times.
    # The only critical use of this is in the computation of the
    # dynamical time scale. Late time steps may thus be smaller than
    # necessary, but that is OK.
    if master:
        background = get_class_background(
            class_params | {'Gamma_dcdm': 0},
            class_call_reason='in order to determine Ωdcdm',
        )
        Ωdcdm = background['(.)rho_dcdm'][-1]/background['(.)rho_crit'][-1]
    Ωdcdm = bcast(Ωdcdm)
num_ncdm = get_number_of_massive_neutrino_species()
if enable_warm_dark_matter:
    # We are running with warm dark matter
    if master:
        background = get_class_background(
            class_params,
            class_call_reason='in order to determine Ωwdm',
        )
        Ωwdm = background[f'(.)rho_ncdm[{num_ncdm}]'][-1]/background['(.)rho_crit'][-1]
    Ωwdm = bcast(Ωwdm)
Ωm = Ωb + Ωcdm + Ωdcdm + Ωwdm
# Specification of which CLASS species together constitute "matter"
# in the current simulation. Ignore species with very low Ω.
matter_class_species_Ω = {
    'b': Ωb,
    'cdm': Ωcdm,
    'dcdm': Ωdcdm,
    f'ncdm[{num_ncdm}]': Ωwdm,
}
matter_class_species = '+'.join([
    class_species
    for class_species, Ω in matter_class_species_Ω.items()
    if Ω > 1e-9
])
# The average, comoving matter density
ρ_mbar = Ωm*ρ_crit
# Specification of which CLASS species together constitute "radiation",
# "neutrino" and "massive neutrino" in the current simulation.
radiation_class_species = 'g'  # photons always present in CLASS
neutrino_class_species = ''
massive_neutrino_class_species = ''
if 'N_ur' in class_params:
    if class_params['N_ur'] > 0:
        radiation_class_species += '+ur'
        neutrino_class_species += '+ur'
elif 'N_eff' in class_params:
    if class_params['N_eff'] > 0:
        radiation_class_species += '+ur'
        neutrino_class_species += '+ur'
elif 'Omega_ur' in class_params:
    if class_params['Omega_ur'] > 0:
        radiation_class_species += '+ur'
        neutrino_class_species += '+ur'
elif 'Omega_ur' in class_params:
    if class_params['Omega_ur'] > 0:
        radiation_class_species += '+ur'
        neutrino_class_species += '+ur'
else:
    # Massless neutrinos present in CLASS by default
    radiation_class_species += '+ur'
    neutrino_class_species += '+ur'
num_ncdm = get_number_of_massive_neutrino_species()
if num_ncdm > 0:
    massive_neutrino_class_species += '+'.join([f'ncdm[{i}]' for i in range(num_ncdm)])
    neutrino_class_species += f'+{massive_neutrino_class_species}'
if Ωdcdm > 1e-9 and class_params.get('Gamma_dcdm', 0) > 0:
    radiation_class_species += '+dr'
radiation_class_species = radiation_class_species.strip('+')
neutrino_class_species = neutrino_class_species.strip('+')
massive_neutrino_class_species = massive_neutrino_class_species.strip('+')
# Specification of which CLASS species together constitute
# "dark energy" in the current simulation.
dark_energy_class_species = ''
if 'Omega_Lambda' in class_params:
    if class_params['Omega_Lambda'] == 0:
        dark_energy_class_species += '+fld'
    else:
        dark_energy_class_species += '+lambda'
        dark_energy_class_species += '+fld'
if 'Omega_fld' in class_params:
    if class_params['Omega_fld'] == 0:
        dark_energy_class_species += '+lambda'
    else:
        dark_energy_class_species += '+lambda'
        dark_energy_class_species += '+fld'
if not dark_energy_class_species:
    dark_energy_class_species = 'lambda'
dark_energy_class_species = '+'.join(set(dark_energy_class_species.strip('+').split('+')))



#####################
# Update units_dict #
#####################
# In the code, the units_dict is often used as a name space for
# evaluating a Python expression given as a str which may include units.
# These expressions are some times supplied by the user, and so may
# contain all sorts of "units". Update the units_dict with quantities so
# that quite general evaluations can take place.
# Add physical quantities.
units_dict.setdefault(        'G_Newton' , G_Newton)
units_dict.setdefault(        'H0'       , H0      )
units_dict.setdefault(        'a_begin'  , a_begin )
units_dict.setdefault(        'boxsize'  , boxsize )
units_dict.setdefault(        't_begin'  , t_begin )
units_dict.setdefault(        'Ωb'       , Ωb      )
units_dict.setdefault(unicode('Ωb')      , Ωb      )
units_dict.setdefault(asciify('Ωcdm')    , Ωcdm    )
units_dict.setdefault(unicode('Ωcdm')    , Ωcdm    )
units_dict.setdefault(asciify('Ωm')      , Ωm      )
units_dict.setdefault(unicode('Ωm')      , Ωm      )
units_dict.setdefault(asciify('ρ_vacuum'), ρ_vacuum)
units_dict.setdefault(unicode('ρ_vacuum'), ρ_vacuum)
units_dict.setdefault(asciify('ρ_crit')  , ρ_crit  )
units_dict.setdefault(unicode('ρ_crit')  , ρ_crit  )
units_dict.setdefault(asciify('ρ_mbar')  , ρ_mbar  )
units_dict.setdefault(unicode('ρ_mbar')  , ρ_mbar  )
# Add dimensionless sizes
units_dict.setdefault('ewald_gridsize', ewald_gridsize)
# Add numbers
units_dict.setdefault(asciify('machine_ϵ'), machine_ϵ)
units_dict.setdefault(unicode('machine_ϵ'), machine_ϵ)
units_dict.setdefault(asciify('machine_ε'), machine_ϵ)
units_dict.setdefault(unicode('machine_ε'), machine_ϵ)
units_dict.setdefault('eps'               , machine_ϵ)
units_dict.setdefault(asciify('π')        , π        )
units_dict.setdefault(unicode('π')        , π        )
units_dict.setdefault('pi'                , π        )
units_dict.setdefault(asciify('τ')        , τ        )
units_dict.setdefault(unicode('τ')        , τ        )
units_dict.setdefault('tau'               , τ        )
units_dict.setdefault(asciify('ထ')        , ထ        )
units_dict.setdefault(unicode('ထ')        , ထ        )
# Add everything from NumPy
for key, val in vars(np).items():
    units_dict.setdefault(key, val)
# Overwrite the NumPy min and max function with NumPy and
# built-in hybrid min and max functions.
units_dict['min'] = produce_np_and_builtin_function('min')
units_dict['max'] = produce_np_and_builtin_function('max')
# Add special functions
units_dict.setdefault('cbrt', lambda x: x**(1/3))



###############
# CLASS setup #
###############
# Warn if the Hubble parameter, the baryon or cold dark matter density
# parameter is specified explicitly in class_params.
for class_param in ('H0', 'h', 'theta_s'):
    if class_param in class_params:
        masterwarn(
            f'You have specified the Hubble constant by declaring "{class_param}" in '
            f'class_params. You should instead define H0 as a normal {esc_concept} parameter.'
        )
for class_param in ('Omega_b', 'omega_b'):
    if class_param in class_params:
        masterwarn(
            f'You have specified the baryon density parameter by declaring "{class_param}" in '
            f'class_params. You should instead define Ωb as a normal {esc_concept} parameter.'
        )
for class_param in ('Omega_cdm', 'omega_cdm'):
    if class_param in class_params:
        masterwarn(
            f'You have specified the cold dark matter density parameter by declaring '
            f'"{class_param}" in class_params. You should instead define Ωcdm as a normal '
            f'{esc_concept} parameter.'
        )
# Update class_params with default values. This has already been done
# before for the version of class_params inside of user_params.
class_params = update_class_params(class_params)
# Function that can call out to CLASS,
# correctly taking advantage of OpenMP and MPI.
@cython.pheader(
    # Arguments
    extra_params=dict,
    sleep_time=object,  # float, int or container or floats and ints
    mode=str,
    class_call_reason=str,
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
    param=str,
    params_specialized=dict,
    transformations=dict,
    value=str,
    value_proper=str,
    values_improper=set,
    returns=object,  # classy.Class or (classy.Class, Py_ssize_t[::1])
)
def call_class(extra_params=None, sleep_time=0.1, mode='single node', class_call_reason=''):
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
    # Merge global and extra CLASS parameters
    params_specialized = class_params | extra_params
    # Print warnings when CLASS parameters are given which does not
    # affect the linear computation which is to take place.
    for param in ('A_s', 'n_s', 'alpha_s', 'k_pivot'):
        if param in params_specialized:
            masterwarn(
                f'The CLASS parameter "{param}" was specified. This will not change the CLASS '
                f'computation. To specify "{param}" for a {esc_concept} simulation, '
                f'specify it in the "primordial_spectrum" dict.'
            )
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
    # Transform parameters to correct CLASS syntax
    for param, transformations in {
        'use_ppf': {'yes': {'True', '1', '1.0'}, 'no': {'False', '0', '0.0'}},
    }.items():
        value = params_specialized.get(param)
        if value is None:
            continue
        for value_proper, values_improper in transformations.items():
            if value in values_improper:
                params_specialized[param] = value_proper
                break
    # Get a copy of the original k_output_values
    k_output_values_global_str = params_specialized.get('k_output_values', '')
    # Fairly distribute the k modes among the nodes,
    # taking the number of processes in each node into account.
    if 'k_output_values' in params_specialized:
        k_output_values = params_specialized['k_output_values'].split(',')
        if k_output_values != sorted(k_output_values, key=float):
            masterwarn(
                'Unsorted k_output_values passed to call_class(). '
                'This may lead to unexpected behaviour'
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
                for nprocs_node_i, k_output_values_node in zip(
                    sorted(nprocs_nodes, reverse=True), k_output_values_nodes):
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
            asarray(list(reversed(k_output_values_nodes)), dtype=object)[
                np.argsort(np.argsort(nprocs_nodes))
           ]
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
        # If no k modes were delegated to this node,
        # add a fake, cheap one.
        if len(k_output_values_node_indices) == 0:
            params_specialized['k_output_values'] = '1e-6'
    else:
        if 'k_output_values' in params_specialized:
            k_output_values_node_indices = arange(len(k_output_values), dtype=C2np['Py_ssize_t'])
    # Choose the thread (OpenMP) scheme for the CLASS computation.
    # This should always be 'mpi'; the 'env' option is available for
    # debugging only. With 'mpi', the number of available threads on
    # each node is set to match the number of MPI processes. With 'env',
    # the number of available threads is instead taken from the
    # environment variable OMP_NUM_THREADS. A warning will be emitted
    # if OMP_NUM_THREADS is set while using the 'mpi' thread scheme.
    thread_scheme = ('mpi', 'env')[0]
    OMP_NUM_THREADS = os.environ.get('OMP_NUM_THREADS')
    warn_on_set_OMP_NUM_THREADS = (thread_scheme == 'mpi' and OMP_NUM_THREADS)
    if thread_scheme == 'mpi':
        num_threads = nprocs_node
    elif thread_scheme == 'env':
        try:
            num_threads = int(OMP_NUM_THREADS)
        except Exception:
            num_threads = -1
    # Write out progress message. If perturbations will be computed,
    # the node masters will print out status updates from within the
    # Class C code. Thus we need to skip to the line below the progress
    # message itself in order not to mess up the first line of these
    # status updates.
    if class_call_reason:
        class_call_reason = class_call_reason.strip() + ' '
    masterprint(f'Calling CLASS {class_call_reason}...')
    if compute_perturbations:
        masterprint('\n', end='')
        if warn_on_set_OMP_NUM_THREADS:
            masterwarn(
                f'The environment contains OMP_NUM_THREADS={OMP_NUM_THREADS}. '
                f'This will be ignored.'
            )
    # The Class instance shall be fed with information about
    # the local node (number) and number of threads (processes on the
    # local node), as well as the progress message to write during
    # perturbation computation.
    message = ''
    if k_output_values_global_str:
        significant_figures = 4
        if master and 'e' in k_output_values_global_str:
            significant_figures = len(
                k_output_values_global_str[:k_output_values_global_str.index('e')]
            ) - 1
        significant_figures = bcast(significant_figures)
        if significant_figures < 1:
            significant_figures = 1
        if master:
            modes_max = np.max([
                len(k_output_values_other_node)
                for k_output_values_other_node in k_output_values_nodes
            ])
            inserts = [
                '%{}d'.format(len(str(nnodes - 1))),
                '%{}d'.format(len(str(np.max(nprocs_nodes) - 1))),
                '%.{}e'.format(significant_figures - 1),
                *(2*['%{}d'.format(len(str(modes_max + 1)))]),
            ]
            message = 'Node {}, thread {}: Evolving mode k = {}/Mpc ({}/{})\n'.format(*inserts)
            message = fancyprint(message % ((0, )*len(inserts)), do_print=False)
            for insert in inserts:
                message = re.subn((insert%0).replace(r'+', r'\+'), insert, message, 1)[0]
        message = bcast(message)
    # Depending on the mode, initialize a Class instance
    # on the master or node masters.
    # the other precesses will not run CLASS.
    cosmo = func = None
    if (mode == 'single node' and master) or (mode == 'mpi' and node_master):
        from classy import Class
        cosmo = Class(concept_class_call=True, node=node, num_threads=num_threads, message=message)
        cosmo.set(params_specialized)
        func = cosmo.compute
    # The background should be available to all processes.
    # Broadcast the background and instantiate fake classy.Class
    # instances on slave processes, with the get_background() method
    # in place.
    fake_cosmo = get_fake_cosmo(cosmo)
    if cosmo is None:
        cosmo = fake_cosmo
    # Call cosmo.compute in such a way as to allow
    # for OpenMP parallelization.
    Barrier()
    call_openmp_lib(func, sleep_time=sleep_time, mode=mode)
    Barrier()
    masterprint('done')
    # Always return the cosmo object. If perturbations have
    # been computed, also return the indices of the k_output_values
    # that have been computed on this node.
    if 'k_output_values' in params_specialized:
        return cosmo, k_output_values_node_indices
    else:
        return cosmo
# Helper function for the call_class() function
def get_fake_cosmo(cosmo=None):
    background = bcast(cosmo.get_background() if master else None)
    h = bcast(cosmo.h() if master else None)
    perturbations = {}
    return FakeClass(
        lambda: background,
        lambda: perturbations,
        lambda: h,
        lambda: None,
    )
FakeClass = collections.namedtuple(
    'FakeClass',
    ['get_background', 'get_perturbations', 'h', 'struct_cleanup'],
)



############################
# Custom defined functions #
############################
# Absolute function for numbers
with copy_on_import:
    if not cython.compiled:
        # Use NumPy's abs function in pure Python.
        # If the argument is a Python int, let the result by a
        # Python int as well (instead of a NumPy int).
        def abs(x):
            ispyint = isinstance(x, int)
            x = np.abs(x)
            if ispyint:
                x = int(x)
            return x
    else:
        @cython.inline
        @cython.header(
            x=fused_numeric,
            returns=fused_numeric,
        )
        def abs(x):
            if fused_numeric in fused_floating:
                return fabs(x)
            else:
                return llabs(x)

# Integer square root
with copy_on_import:
    @cython.inline
    @cython.header(
        # Arguments
        x='Py_ssize_t',
        # Locals
        returns='Py_ssize_t',
    )
    def isqrt(x):
        """This function implements the square root for integers x such
        that isqrt(x) = int(sqrt(x)). Though proper integer-only code
        for this purpose (similar to the code for icbrt() below)
        is available, I have found that using the floating-point sqrt()
        function is in fact the fastest option. Note that unlike
        int(cbrt(x)), int(sqrt(x)) always returns the correct answer.
        """
        return int(sqrt(x))

# Check whether integer is quadratic
with copy_on_import:
    @cython.inline
    @cython.header(
        # Arguments
        x='Py_ssize_t',
        # Locals
        returns='bint',
    )
    def isquadratic(x):
        return (isqrt(x)**2 == x)

# Integer cube root
with copy_on_import:
    @cython.inline
    @cython.header(
        # Arguments
        x='Py_ssize_t',
        # Locals
        b='Py_ssize_t',
        bits='Py_ssize_t',
        one='Py_ssize_t',
        s='Py_ssize_t',
        y='Py_ssize_t',
        returns='Py_ssize_t',
    )
    def icbrt(x):
        """This function implements the cube root for integers x such
        that icbrt(x) = int(cbrt(x)), though this function is about
        twice as fast as using the floating-point cbrt(). Unlike the
        floating-point cbrt(), this function is exact
        (e.g. int(cbrt(15**3)) == 14). A defining relation for this
        function more appropriate than icbrt(x) = int(cbrt(x)) is then
        that y = icbrt(x) such that y**3 == x or (y + 1)**3 > x.
        The code below works for integers with at most the first 'bits'
        bits set, where 'bits' must be a multiple of 3. Lowering 'bits'
        results in faster execution.
        """
        bits = 3*14  # largest x: 2**(3*14) - 1 = (2**14)**3 - 1 = 16384**3 - 1
        for s in range(bits - 3, -3, -3):
            if (x >> s) == 0:
                continue
            one = 1  # we need this to be 64-bit
            x -= one << s
            y = 1
            for s in range(s - 3, -3, -3):
                y += y
                b = 1 + 3*y*(y + 1)
                if (x >> s) >= b:
                    x -= b << s
                    y += 1
            return y
        return 0

# Check whether integer is cubic
with copy_on_import:
    @cython.inline
    @cython.header(
        # Arguments
        x='Py_ssize_t',
        # Locals
        returns='bint',
    )
    def iscubic(x):
        return (icbrt(x)**3 == x)

# Function returning a nice str representation of cubic-ish integers
def get_cubenum_strrep(n):
    if n < 8:
        return str(n)
    if iscubic(n):
        return '{}³'.format(icbrt(n))
    for denom in range(2, 5):
        r = n//denom
        if r*denom == n and iscubic(r):
            return '{}×{}³'.format(denom, str(icbrt(r)))
    return str(n)

# Integer binary logarithm
with copy_on_import:
    @cython.inline
    @cython.header(
        # Arguments
        x='Py_ssize_t',
        # Locals
        y='Py_ssize_t',
        returns='Py_ssize_t',
    )
    def ilog2(x):
        """This function implements the binary logarithm (log base 2)
        for integers x such that ilog2(x) = int(log2(x)).
        For negative (illegal) x, we return x. The same is done for
        x = 0, as -∞ cannot be expressed as an integer.
        """
        if x < 1:
            # Illegal input
            return x
        y = 0
        while x > 1:
            x >>= 1
            y += 1
        return y

# Check whether integer is a power of two
with copy_on_import:
    @cython.inline
    @cython.header(
        # Arguments
        x='Py_ssize_t',
        # Locals
        returns='bint',
    )
    def ispowerof2(x):
        return (2**ilog2(x) == x)

# Max function for 1D memory views of numbers
with copy_on_import:
    if not cython.compiled:
        # Use NumPy's max function in pure Python
        max = np.max
    else:
        """
        @cython.inline
        @cython.header(returns=fused_numeric)
        def max(fused_numeric[::1] a):
            cdef:
                Py_ssize_t i
                fused_numeric m
            m = a[0]
            for i in range(1, a.shape[0]):
                if a[i] > m:
                    m = a[i]
            return m
        """

# Min function for 1D memory views of numbers
with copy_on_import:
    if not cython.compiled:
        # Use NumPy's min function in pure Python
        min = np.min
    else:
        """
        @cython.inline
        @cython.header(returns=fused_numeric)
        def min(fused_numeric[::1] a):
            cdef:
                Py_ssize_t i
                fused_numeric m
            m = a[0]
            for i in range(1, a.shape[0]):
                if a[i] < m:
                    m = a[i]
            return m
        """

# Maximum function for pairs of numbers.
# Note that this breaks if either of the numbers are infinite.
with copy_on_import:
    @cython.inline
    @cython.header(a=fused_numeric, b=fused_numeric, returns=fused_numeric)
    def pairmax(a, b):
        return a*(b <= a) + b*(a < b)

# Minimum function for pairs of numbers.
# Note that this breaks if either of the numbers are infinite.
with copy_on_import:
    @cython.inline
    @cython.header(a=fused_numeric, b=fused_numeric, returns=fused_numeric)
    def pairmin(a, b):
        return a*(a <= b) + b*(b < a)

# Proper modulo function mod(x, length) for scalars,
# with x ∈ [0, length) for length > 0.
# Note that this is different from both x%length
# and fmod(x, length) in C.
with copy_on_import:
    if not cython.compiled:
        def mod(x, length):
            x = np.mod(x, length)
            # Ensure x ∈ [0, length)
            if x == length:
                return 0
            return x
    else:
        @cython.inline
        @cython.header(
            # Arguments
            x=fused_numeric,
            length=fused_numeric2,
            # Locals
            returns=fused_numeric,
        )
        def mod(x, length):
            if fused_numeric in fused_integral and fused_numeric2 in fused_floating:
                # Not supported
                return -1
            else:
                while x >= length:
                    x -= length
                while x < 0:
                    x += length
                if fused_numeric in fused_floating:
                    if x == length:
                        return 0
                return x

# Summation function for 1D memory views of numbers
with copy_on_import:
    if not cython.compiled:
        # Use NumPy's sum function in pure Python
        sum = np.sum
    else:
        """
        @cython.inline
        @cython.header(returns=fused_numeric)
        def sum(fused_numeric[::1] a):
            cdef:
                fused_numeric Σ
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
with copy_on_import:
    if not cython.compiled:
        # Use NumPy's prod function in pure Python
        prod = np.prod
    else:
        """
        @cython.inline
        @cython.header(returns=fused_numeric)
        def prod(fused_numeric[::1] a):
            cdef:
                fused_numeric Π
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
with copy_on_import:
    if not cython.compiled:
        # Use NumPy's mean function in pure Python
        mean = np.mean
    else:
        """
        @cython.inline
        @cython.header(returns=fused_numeric)
        def mean(fused_numeric[::1] a):
            return sum(a)/a.shape[0]
        """

# Function that compares two numbers (identical to math.isclose)
with copy_on_import:
    @cython.inline
    @cython.pheader(
        # Arguments
        a=fused_numeric,
        b=fused_numeric,
        rel_tol='double',
        abs_tol='double',
        # Locals
        answer='bint',
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
        answer = abs(a - b) <= tol
        if not answer and (size_a == 0 or size_b == 0) and abs_tol == 0:
            warn(
                'isclose() was called with one argument being 0 with no abs_tol. '
                'This can never result in the arguments being deemed close, '
                'regardless of the smallness of the other (non-zero) argument.'
            )
        return answer

# Function that checks if a (floating-point) number
# is actually an integer.
with copy_on_import:
    @cython.inline
    @cython.header(
        x='double',
        rel_tol='double',
        abs_tol='double',
        returns='bint',
    )
    def isint(x, abs_tol=1e-6):
        return isclose(x, round(x), 0, abs_tol)

# Function which format numbers to have a
# specific number of significant figures.
@cython.pheader(
    # Arguments
    numbers=object,  # Single number or container of numbers
    nfigs='int',
    fmt=str,
    # Locals
    coefficient=str,
    coefficient_int='Py_ssize_t',
    coefficient_float='double',
    exponent=str,
    fmt_str=str,
    fmt_str_scientific=str,
    is_small='bint',
    n_missing_zeros='int',
    number=object,  # Single number of any type
    number_str=str,
    return_list=list,
    returns=object,  # String or list of strings
)
def significant_figures(
    numbers, nfigs, fmt='', incl_zeros=True, force_scientific=False,
    base_call=True,
):
    """This function formats a floating-point number to have nfigs
    significant figures.
    Set fmt to 'TeX' to format to TeX math code
    (e.g. '1.234\times 10^{-5}') or 'Unicode' to format to superscript
    Unicode (e.g. 1.234×10⁻⁵).
    Set incl_zeros to False to avoid zero-padding.
    Set force_scientific to True to force full scientific notation.
    """
    fmt = fmt.lower()
    if fmt not in ('', 'tex', 'unicode'):
        abort('Formatting mode "{}" not understood'.format(fmt))
    return_list = []
    for number in any2list(numbers):
        is_small = (np.abs(number) < 1)
        # Format the number using nfigs
        fmt_str_scientific = f'{{:.{nfigs - 1}e}}'
        if force_scientific:
            fmt_str = fmt_str_scientific
        else:
            fmt_str = f'{{:.{nfigs}g}}'
            if 'e' in fmt_str.format(number) or (number != 0 and 'e' in fmt_str.format(1/number)):
                fmt_str = fmt_str_scientific
        number_str = fmt_str.format(number)
        if is_small and base_call:
            # For consistent formatting for numbers with absolute values
            # below unity, we need to call this function once more,
            # with the rounded number.
            number_str = significant_figures(
                float(number_str), nfigs, fmt, incl_zeros, force_scientific,
                base_call=False,
            )
            return_list.append(number_str)
            continue
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
        coefficient_float = float(coefficient)
        coefficient_int = int(coefficient_float)
        if coefficient_int == coefficient_float:
            coefficient = str(coefficient_int)
        # Remove coefficient if it is just 1 (as in 1×10¹⁰ → 10¹⁰)
        if coefficient == '1' and exponent and fmt in ('tex', 'unicode') and not force_scientific:
            coefficient = ''
            exponent = exponent.replace(r'\times ', '').replace(unicode('×'), '')
        elif incl_zeros:
            # Pad with zeros in case of too few significant digits
            digits = coefficient.replace('.', '').replace('-', '')
            for i, d in enumerate(digits):
                if d != '0':
                    digits = digits[i:]
                    break
            n_missing_zeros = nfigs - len(digits)
            if n_missing_zeros > 0:
                if '.' not in coefficient:
                    coefficient += '.'
                coefficient += '0'*n_missing_zeros
        # Combine
        number_str = coefficient + exponent
        # The mathtext Matplotlib module has a typographical bug;
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

# Function for correcting floating-point numbers
def correct_float(val_raw):
    """Example: correct_float(1.234499999999998) → 1.2345
    """
    isnumber = False
    try:
        val_raw = float(val_raw)
        isnumber = True
    except Exception:
        pass
    if not isnumber:
        # Assume container
        vals = [correct_float(val_raw_i) for val_raw_i in any2list(val_raw)]
        val_type = type(val_raw)
        if val_type is np.ndarray:
            return asarray(vals)
        return val_type(vals)
    val_g = float(f'{val_raw:g}')
    if val_g == val_raw:
        return val_g
    val_str = str(np.abs(val_raw))
    if 'e' in val_str:
        val_str = val_str[:val_str.index('e')]
    val_str = val_str.replace('.', '')
    if len(val_str) < 15:
        return val_raw
    val_new_lower = val_raw*(1 - 10*machine_ϵ)
    val_new_upper = val_raw*(1 + 10*machine_ϵ)
    val_correct = val_new = val_new_lower
    while val_new <= val_new_upper:
        if len(str(val_new)) < len(str(val_correct)):
            val_correct = val_new
        val_new = np.nextafter(val_new, ထ)
    return (val_correct if len(str(val_correct)) < len(str(val_raw)) - 2 else val_raw)

# Function for finding unique elements of a floating-point array,
# using fuzzy comparison.
@cython.pheader(
    # Arguments
    arr='double[::1]',
    rel_tol='double',
    abs_tol='double',
    # Locals
    mask='unsigned char[::1]',  # really Boolean np.ndarray
    a='double',
    b='double',
    i='Py_ssize_t',
    i_accept='Py_ssize_t',
    returns='double[::1]',
)
def unique(arr, rel_tol=1e-9, abs_tol=0):
    # Find truly unique elements. This also sorts the elements.
    arr = np.unique(arr)
    # Build mask
    mask = ones(arr.shape[0], dtype=C2np['bint'])
    i_accept = 0  # Always include first element
    a = arr[i_accept]
    i = -1
    for i in range(i_accept + 1, arr.shape[0]):
        b = arr[i]
        if isclose(a, b, rel_tol, abs_tol):
            mask[i] = False
        else:
            a = b
            i_accept = i
    # Always include the last element
    if i_accept != i != -1:
        mask[i_accept] = False
        mask[i       ] = True
    # Apply mask
    return asarray(arr)[mask]

# Generalized getattr() function accepting dotted (nested) arguments
def getattr_nested(obj, attr):
    if '[:].' in attr:
        attr_0, attr_1 = attr.split('[:].')
        return [getattr(el, attr_1) for el in getattr(obj, attr_0)]
    return eval(f'obj.{attr}')

# Function creating and returning a spline
# from a dictionary of control points.
def get_controlpoint_spline(
    d,
    transform_x=(lambda x: x), transform_x_onlookup=False, kind='linear',
):
    import scipy.interpolate
    x = np.fromiter(d.keys(), dtype=C2np['double'])
    y = np.fromiter(d.values(), dtype=C2np['double'])
    fill_min = d[np.min(x)]
    fill_max = d[np.max(x)]
    if transform_x is np.log10:
        _transform_x = transform_x
        def transform_x_scalar(x):
            return _transform_x(x) if x > 0 else -ထ
        def transform_x(x):
            if isinstance(x, (int, float, np.integer, np.floating)):
                return transform_x_scalar(x)
            return asarray([transform_x_scalar(x_i) for x_i in x])
    x_transformed = transform_x(x)
    if transform_x_onlookup:
        transform_x_lookup = transform_x
    else:
        transform_x_lookup = (lambda x: x)
    interp = lambda x_new, *, f=scipy.interpolate.interp1d(
        x_transformed,
        y,
        kind=kind,
        bounds_error=False,
        fill_value=(fill_min, fill_max),
    ): float(f(transform_x_lookup(x_new)))
    return interp

# Function which searches a dictionary for a component
# or a set of components.
# Note that something goes wrong when trying to make this into
# a cdef function, so we leave it as a pure Python function.
def is_selected(component_or_components, d, accumulate=False, default=None):
    """This function searches for the given component in the given
    dict d. Both the component instance itself and its name, species
    and representation attributes are used, as well as the str's 'all'
    and 'default'.
    The precedence of these (lower takes precedence) are:
    - 'default'
    - 'all'
    - component.representation
    - any single species in component.species.split('+')
    - component.species
    - component.name
    - component
    If multiple components are given, d is searched for an iterable
    containing all these components (and no more). Both the
    component instances themselves, their names, species and
    representations as well as the str's 'all combinations' and
    'default' are used.
    The precedence of these (lower takes precedence) are:
    - 'default'
    - 'all combinations'
    - {component0.representation, component1.representation, ...}
    - {component0.name, component1.name, ...}
    - {component0, component1, ...}
    All str's are compared case insensitively. If a str key is not found
    in d, a regular expression match of the entire str is attempted.
    If the component is found in d, its value in d is returned.
    Otherwise, the value specified by the default argument is returned.
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
            *[single_species.lower() for single_species in component.species.split('+')],
            component.species.lower(),
            component.name.lower(),
            component,
        )
    else:
        components = frozenset(component_or_components)
        names = frozenset([component.name.lower() for component in components])
        representations = frozenset([component.representation.lower() for component in components])
        single_species = frozenset([
            single_species.lower()
            for component in components
            for single_species in component.species.split('+')
        ])
        species = frozenset([component.species.lower() for component in components])
        keys = (
            'default',
            'all combinations',
            representations,
            single_species,
            species,
            names,
            components,
        )
    # Ensure lower-case on all str keys
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
    notfound_token = '__�__'
    selected = []
    for key in keys:
        val = d.get(key, notfound_token)
        if val is notfound_token and isinstance(key, str):
            # Maybe the key is a regular expression
            for d_key, d_val in d.items():
                if isinstance(d_key, str) and re.compile(d_key).fullmatch(key):
                    val = d_val
                    break
        if val is not notfound_token:
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
            return default

# Function for evaluting a function f(), f(t), f(a), f(t, a) or f(a, t)
# at the current cosmic time t and/or scale factor a.
def eval_func_of_t_and_a(f):
    current_values = {'t': universals.t, 'a': universals.a}
    return f(*[
        current_values[param]
        for param in inspect.signature(f).parameters
        if param in current_values
    ])

# Function for generating file names for reusable data
def get_reusable_filename(kind, *objects, extension=''):
    # The file name file name will be generated based on the hash of the
    # passed objects. The hahs will be expressed using all lowercase
    # alphanumeric ASCII characters, of which there are 36. A 50% chance
    # of a single hash collision will then occur after ~ 36**(l/2)
    # hashes, l being the length of the 36-character hash.
    algorithm = 'sha512'
    size = 2**4  # with 2⁴: 50% chance of a single hash collision after ~ 10¹² hashes
    table = '0123456789abcdefghijklmnopqrstuvwxyz'
    # Transform non-ordered containers to sorted lists
    objects = list(objects)
    for i, obj in enumerate(objects):
        if isinstance(obj, set):
            objects[i] = sorted(obj)
        elif isinstance(obj, dict):
            objects[i] = sorted(obj.items())
    # Create str representation. We convert integer-valued floats to
    # integers directly on the str representation.
    objects_str = (str(objects)[1:-1] + ',').replace('.0,', ',')
    # Compute the hash
    hexdigest = getattr(hashlib, algorithm)(objects_str.encode('utf-8')).hexdigest()
    # Shorten hexdigest using the table
    chunk_size = len(hexdigest)//size
    digest = []
    for i in range(size):
        chunk = hexdigest[chunk_size*i:chunk_size*(i + 1)]
        s = np.sum([int(c, base=16) for c in chunk])
        digest.append(table[s%len(table)])
    digest = ''.join(digest)
    if extension:
        extension = '.' + extension.lstrip('.')
    filename = f'{path.reusable_dir}/{kind}/{digest}{extension}'
    return filename

# Function for doing lookup into shortrange_params which depend on the
# component(s) in question, i.e. if the values involves the number of
# particles N.
@cython.header(
    # Arguments
    component_or_components=object,  # Component or list of Components
    interaction_name=str,
    param=str,
    # Locals
    component=object,  # actually Component, but not allowed in the commons module
    components=list,
    key=tuple,
    val=object,
    returns='double',
)
def get_shortrange_param(component_or_components, interaction_name, param):
    components = any2list(component_or_components)
    # Look up in cache
    key = (tuple([component.name for component in components]), interaction_name, param)
    val = get_shortrange_param_cache.get(key)
    if val is not None:
        return val
    # Get value from shortrange_params
    val = shortrange_params[interaction_name][param]
    if isinstance(val, dict):
        # Get the value corresponding to the passed component(s)
        if len(components) == 1:
            val_selected = is_selected(components[0], val)
        else:
            val_selected = is_selected(components, val)
            if val_selected is None:
                for component in components:
                    val_selected = is_selected(component, val)
                    if val_selected is not None:
                        break
        if val_selected is None:
            abort(
                f'Lookup of',
                [component.name for component in any2list(component_or_components)],
                f'in shortrange_params["{interaction_name}"]["{param}"] failed'
            )
        val = val_selected
    # Substitute string expressions
    if isinstance(val, str):
        val = val.replace('boxsize', str(boxsize))
        if 'N' in val:
            component = any2list(component_or_components)[0]
            val = val.replace('N', str(component.N))
        # Replace scale(s)
        scale_strs = (
            unicode('scaleⱼ'), asciify('scaleⱼ'), 'scale_j', 'scalej', 'scale[j]',
            unicode('scaleₛ'), asciify('scaleₛ'), 'scales', 'scale[s]',
            'scale[1]', 'scale_1', 'scale1',
        )
        scale = None
        for scale_str in scale_strs:
            if scale_str in val:
                if scale is None:
                    scale = str(get_shortrange_param(components[1], interaction_name, 'scale'))
                val = val.replace(scale_str, scale)
        scale_strs = (
            unicode('scaleᵢ'), asciify('scaleᵢ'), 'scale_i', 'scalei', 'scale[i]',
            unicode('scaleᵣ'), asciify('scaleᵣ'), 'scaler', 'scale[r]',
            'scale[0]', 'scale_0', 'scale0', 'scale',
        )
        scale = None
        for scale_str in scale_strs:
            if scale_str in val:
                if scale is None:
                    scale = str(get_shortrange_param(components[0], interaction_name, 'scale'))
                val = val.replace(scale_str, scale)
        val = eval_unit(val)
    # Cache and return float result
    get_shortrange_param_cache[key] = val
    val = float(val)
    return val
cython.declare(get_shortrange_param_cache=dict)
get_shortrange_param_cache = {}

# Context manager which suppresses all output to stdout
@contextlib.contextmanager
def suppress_stdout(f=sys.stdout):
    with open(os.devnull, mode='w') as devnull:
        # Point sys.stdout to /dev/null
        sys.stdout = devnull
        try:
            # Yield control back to the caller
            yield
        finally:
            # Cleanup: Reset sys.stdout
            sys.stdout = f

# Function decorator cache equivalent to functools.lru_cache,
# but extended with copying functionality (when called with copy=True)
# so that the returned object is freshly instantiated,
# handy if the returned object is mutable.
def lru_cache(maxsize=128, typed=False, copy=False):
    if not copy:
        return functools.lru_cache(maxsize, typed)
    def decorator(f):
        cached_func = functools.lru_cache(maxsize, typed)(f)
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))
        return wrapper
    return decorator

# The terminal object from blessings is used for formatted printing.
# If this is disabled, we replace the terminal object
# with a dummy object.
if not enable_terminal_formatting:
    class DummyTerminal:
        @staticmethod
        def dummy_func(x):
            return x
        def __getattr__(self, att):
            return self.dummy_func
    terminal = DummyTerminal()

# As already done for some pyplot methods, monkey patch I/O functions
# so that exceptions lead to abort.
open_file = tryexcept_wrapper(open, 'open() failed')  # must not overwrite the standard "open"
wrap_method(os, 'makedirs', tryexcept_wrapper, '{obj}.{func}() failed')
wrap_method(
    np,
    ['load', 'save', 'loadtxt', 'savetxt'],
    tryexcept_wrapper,
    '{obj}.{func}() failed',
)
# NumPy's savetxt has fmt='%.18e' by default. This is wasteful as %.16e
# can encode any double-precision number without loss of precision.
# We change this default accordingly.
np.savetxt = functools.partial(np.savetxt, fmt='%.16e')
# For h5py.File the monkey patch is more involved
def open_hdf5(filename, raise_exception=False, **kwargs):
    """This function is equivalent to just doing
    h5py.File(filename, **kwargs)
    except that it will not throw an exception if the file is
    temporarily unavailable, which happens when multiple processes
    attempts to open the same file in write mode. When this is the case,
    this function waits (possibly indefinitely) for the file to
    become available.
    The function supports both collective and non-collective calls.
    It is an error to call non-collectively from any process but the
    master mode.
    """
    # A warning about np.float128 might be emitted from _get_machar()
    # within numpy/core/getlimits.py upon importing h5py.
    with warnings.catch_warnings(action='ignore', category=UserWarning):
        import h5py
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
                    kwargs_noncollective.pop(kwarg)
        sleep_time = sleep_time_min
        while True:
            try:
                h5py.File(filename, **kwargs_noncollective).close()
                break
            except OSError as e:
                if raise_exception:
                    raise e
                if 'File exists' not in str(e):
                    traceback.print_exc()
                    abort('h5py.File() failed')
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
# Abort on unrecognised snapshot_type
snapshot_types = ['concept', 'gadget', 'tipsy']
if snapshot_type not in snapshot_types:
    abort(f'Unrecognised snapshot type "{snapshot_type}" ∉ {snapshot_types}')
# Abort on unrecognised output kinds
for key in output_dirs:
    if key not in (output_kinds + ('autosave',)):
        abort(f'Unrecognised output type "{key}"')
for key in output_bases:
    if key not in output_kinds:
        abort(f'Unrecognised output type "{key}"')
for d in output_times.values():
    for key in d:
        if key not in output_kinds:
            abort(f'Unrecognised output type "{key}"')
# Warn about odd force differentiation
for name, d0 in potential_options['differentiation'].items():
    for force, d1 in d0.items():
        for method, order in d1.items():
            if order % 2:
                masterwarn(
                    f'As potential_options["differentiation"]["{force}"]["{method}"] = {order} '
                    f'is odd, this will lead to asymmetric differentiation'
                )
# Check format on 'subtiling' values in shortrange_params.
# Among other things, the refinement period used for automatic subtiling
# refinement has to at least as big as subtiling_refinement_period_min,
# as otherwise the implemented subtiling refinement scheme will not
# function properly. To get the right value of
# subtiling_refinement_period_min, consult the anticipation_period and
# judgement_period variables in the interactions module.
cython.declare(subtiling_refinement_period_min='Py_ssize_t')
subtiling_refinement_period_min = 7
for key, d in shortrange_params.items():
    subtiling = d['subtiling']
    if len(subtiling) == 3:
        for el in subtiling:
            if not isinstance(el, (int, np.integer)):
                abort(
                    f'Could not understand shortrange_params["{key}"]["subtiling"] == {subtiling}'
                )
            if el < 1:
                abort(
                    f'shortrange_params["{key}"]["subtiling"] == {subtiling}, '
                    f'but must be at least 1 in every direction.'
                )
    elif len(subtiling) == 2:
        if not (subtiling[0] == 'automatic' and isinstance(subtiling[1], (int, np.integer))):
            abort(
                f'shortrange_params["{key}"]["subtiling"] == {subtiling}. '
                f'When two values are specified, the first should be the str "automatic" '
                f'and the second should be an int specifying the subtiling refinement period.'
            )
        # The subtiling refinement period needs to be at least 7 for the
        # automatic subtiling refinement scheme to function properly.
        if subtiling[1] < subtiling_refinement_period_min:
            masterwarn(
                f'The automatic subtiling refinement period specified in '
                f'shortrange_params["{key}"]["subtiling"] is {subtiling[1]}, '
                f'which is too small. It has been increased to {subtiling_refinement_period_min}.'
            )
            d['subtiling'] = (subtiling[0], subtiling_refinement_period_min)
    else:
        abort(f'Could not understand shortrange_params["{key}"]["subtiling"] == {subtiling}')
# The time step size must be allowed to increase
if Δt_increase_max_factor <= 1:
    abort(f'You must have Δt_increase_max_factor > 1')
# The maximum allowed Δa at late times is supposed to be bigger than the
# maximum allowed value of Δa at early times,
# and they must both be positive.
if Δa_max_early < 0:
    abort('You must have Δa_max_early ≥ 0')
if Δa_max_late <= 0:
    abort('You must have Δa_max_late > 0')
if Δa_max_early > Δa_max_late:
    masterwarn(f'You have Δa_max_early = {Δa_max_early} > Δa_max_late = {Δa_max_late}')
# Static time-stepping (both recording and application) uses scale
# factor values and this requires the Hubble expansion to be enabled.
if static_timestepping is not None and not enable_Hubble:
    abort('You may not specify static_timestepping with the Hubble expansion disabled')
# Abort for non-positive number of rungs. Also, since the rung indices
# are stored as signed chars, the largest rung index that can be
# represented is 127, corresponding to the highest rung for
# N_rungs = 128. In fact, memory errors has been observed already for
# N_rungs >= 32. To be safe, we choose a largest allowed N_rungs of 30.
if N_rungs < 1:
    abort(f'N_rungs = {N_rungs}, but at least one rung must exist')
if N_rungs > 30:
    abort(f'N_rungs = {N_rungs}, but must not be greater than 30')
# Warn about Δt_rung_factor != 1 (the default) when not using rungs,
# as the value of Δt_rung_factor then does not matter.
if N_rungs == 1 and Δt_rung_factor != 1:
    masterwarn(
        f'You are running without rungs (N_rungs = 1), but have set '
        f'Δt_rung_factor = {Δt_rung_factor}. This value does not matter.'
    )
# Abort on illegal FFTW rigour
if fftw_wisdom_rigor not in ('estimate', 'measure', 'patient', 'exhaustive'):
    abort('Does not recognise FFTW rigour "{}"'.format(user_params['fftw_wisdom_rigor']))
# Abort on negative random seed
for random_seed in random_seeds.values():
    if random_seed < 0:
        abort(f'A random seed of {random_seed} < 0 was specified')
# Sanity check on the primordial phase shift
if primordial_phase_shift == 1:
    masterwarn(
        f'You have specified primordial_phase_shift = 1. '
        f'Are you sure that you do not mean primordial_phase_shift = π?'
    )
# Warn about unused but specified parameters
for key in ['h', ] + list(inferred_params_final):
    user_params.use(key)
if user_params.unused:
    if len(user_params.unused) == 1:
        msg = 'The following unknown parameter was specified:\n'
    else:
        msg = 'The following unknown parameters were specified:\n'
    masterwarn(msg + '    ' + '\n    '.join(user_params.unused))
# Output times very close to t_begin or a_begin
# are probably meant to be exactly at t_begin or a_begin
for time_param, time_begin in zip(('t', 'a'), (t_begin, a_begin)):
    output_times[time_param] = {
        key: tuple([
            time_begin if isclose(float(nr), time_begin,
                abs_tol=(0 if (float(nr) and time_begin) else machine_ϵ))
            else correct_float(nr)
            for nr in val])
        for key, val in output_times[time_param].items()
    }
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
# Warn about cosmological autosave interval
if autosave_interval > 1*units.yr and autosave_interval != ထ:
    masterwarn(
        f'Autosaving will take place every {autosave_interval} {unit_time} (real-world time!). '
        f'Have you forgotten to specify the unit of the "autosave_interval" parameter?'
    )
if autosave_interval < 0:
    autosave_interval = 0
    user_params['autosave_interval'] = autosave_interval
# Check keys and values in shortrange_params
for d in shortrange_params.values():
    for key, val in d.items():
        if key not in {'scale', 'range', 'tilesize', 'subtiling', 'tablesize'}:
            masterwarn(f'Unrecognised parameter "{key}" in shortrange_params')
        if key == 'subtiling':
            if isinstance(val, str) and val != 'automatic':
                abort(f'Failed to interpret subtiling "{val}"')
# Replace h in power spectrum top-hat filter
d = powerspec_options['tophat']
for key, val in d.copy().items():
    if isinstance(val, str):
        if enable_Hubble:
            h = H0/(100*units.km/(units.s*units.Mpc))
        else:
            h = 1
        d[key] = eval_unit(val.replace('h', f'({h})'), units_dict)
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
            f'Negative times might lead to unexpected behaviour.'
        )
# Sanity check on the 2D render axis
for key, val in render2D_options['axis'].items():
    if val not in {'x', 'y', 'z'}:
        abort(
            f'render2D_options["extent"][{key}] = "{val}" ∉ {{"x", "y", "z"}}'
        )
# Check that the 2D render extent is within the box and non-zero
for key, val in render2D_options['extent'].items():
    if val[0] < 0 or val[1] > boxsize:
        abort(
            f'render2D_options["extent"]["{key}"] '
            f'= ({val[0]}*{unit_length}, {val[1]}*{unit_length}) '
            f'is out-of-bounds'
        )
    if val[0] == val[1]:
        abort(
            f'Equal limits on render2D_options["extent"]["{key}"] '
            f'= ({val[0]}*{unit_length}, {val[1]}*{unit_length})'
        )
# If the Hubble expansion is deactivated, warn if the CLASS background
# is meant to be used.
if not enable_Hubble and enable_class_background:
    masterwarn('Hubble expansion is deactivated, but enable_class_background is True')
# Allow for easier names in class_extra_background
for val, keys in {
    'gr.fac. D'       : {'D', 'D1'},
    'gr.fac. f'       : {'f', 'f1'},
    'gr.fac. D2'      : {'D2'},
    'gr.fac. f2'      : {'f2'},
    'gr.fac. D3a'     : {'D3a'},
    'gr.fac. f3a'     : {'f3a'},
    'gr.fac. D3b'     : {'D3b'},
    'gr.fac. f3b'     : {'f3b'},
    'gr.fac. D3c'     : {'D3c'},
    'gr.fac. f3c'     : {'f3c'},
    'conf. time [Mpc]': {unicode('τ'), asciify('τ'), 'tau'},
}.items():
    if keys & class_extra_background:
        class_extra_background.difference_update(keys)
        class_extra_background.add(val)



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
            commons_module = load_source(
                'commons_pure_python',
                '{}/commons.py'.format(path['src_dir']),
            )
        stack = inspect.stack()
        if len(stack) == 1:
            frame = stack[0].frame
        else:
            frame = stack[1].frame
        try:
            inspect.getmodule(frame).__dict__.update(commons_module.__dict__)
        except Exception:
            pass
        try:
            frame.f_locals.update(commons_module.__dict__)
        except Exception:
            pass
    else:
        # Running in pure Python mode.
        # It is assumed that "from commons import *" has already
        # been run, leaving nothing to do.
        pass



##########################################
# Print out additional setup information #
##########################################
# Only print out setup information if this is an actual simulation run.
# In this case, a jobid different from -1 is set.
if jobid != -1:
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
