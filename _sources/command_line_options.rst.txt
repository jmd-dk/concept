Command-line options
====================
This page elaborates on the many options to the ``concept`` script, used to
launch CO\ *N*\ CEPT.

If you have yet to play around with the ``concept`` script, you are advised to
take the :doc:`tutorial </tutorial/first_simulations>`.

Most options have a short and a long format --- e.g. ``-h`` and ``--help`` ---
which always does the same thing. Some however only have the long format. When
using the long format, you do not have to spell out the entire option, as long
as the first part uniquely identifies the full option. For example, ``--he``
may be used in place of ``--help``.

Many options require a value, e.g. ``-n 2`` or equivalently
``--nprocs 2``. Instead of a space, the option name and value may be separated
by an equal sign; ``-n=2`` or ``--nprocs=2``.

Every command-line option has a corresponding ``CONCEPT_*`` environment
variable, named according to the long format of the option (with dashes ``-``
replaced by underscores ``_``). As an example, executing

.. code-block:: bash

   export CONCEPT_nprocs=4

will effectively supply ``--nprocs 4`` to future invocations of ``concept`` in
an implicit manner. Manually supplying ``--nprocs``/``-n`` will overrule the
value set by ``CONCEPT_nprocs``.

The command-line options are grouped into four categories as listed below.

.. contents::
   :local:
   :depth: 2



.. _basics:

Basics
------
The following basic command-line options are all you need to know in order to
run CO\ *N*\ CEPT simulations locally.



Help: ``-h``, ``--help``
........................
Displays a short description of each command-line option and exits:

.. code-block:: bash

   ./concept -h

This is helpful if you have forgotten the name of some particular option. For
learning about the usage of a given option, this page is much preferable.



.. _parameter_file:

Parameter file: ``-p``, ``--param``
...................................
Specifies the parameter file to use:

.. code-block:: bash

   ./concept -p </path/to/parameter-file>

The path may be specified in any of these formats:

* Absolutely.
* Relative to your current working directory.
* Relative to the CO\ *N*\ CEPT installation directory.
* Relative to the ``param`` directory.

Typically, parameter files are kept in the ``param`` directory. With the
parameter file ``my_param`` located in this directory, specifying the use
of this parameter file may then look like e.g.

.. code-block:: bash

   ./concept -p param/my_param

The many possible parameters to put inside parameter files are each described
in the sections under :doc:`Parameters </parameters/parameters>`. Parameters
absent from the supplied parameter file will take on default values. Leaving
out the ``-p`` parameter file specification when invoking ``concept``, *all*
parameters take on their default values, which does not result in an actual
simulation as no output is specified by default.



.. _command_line_parameters:

Command-line parameters: ``-c``, ``--command-line-params``
..........................................................
Directly specifies parameters to use, without referring to a parameter file.
E.g.

.. code-block:: bash

   ./concept -c "boxsize = 512*Mpc"

Often this is used in combination with a parameter file, e.g. if a suite of
similar simulations is to be run where only a few parameter values change
between the simulations. E.g.

.. code-block:: bash

   # First simulation
   ./concept \
       -p param/my_param \
       -c "Ωb = 0.049" \
       -c "Ωcdm = 0.270"

   # Second simulation
   ./concept \
       -p param/my_param \
       -c "Ωb = 0.048" \
       -c "Ωcdm = 0.271"

Note that the ``-c`` option may be specified multiple times.

Specifying command-line parameters while also making use of a parameter file
is equivalent to having the command-line parameters defined *at the bottom* of
the parameter file. Despite of this, the command-line parameters will become
available both during the simulation *and* to the other parameters in the
parameter file during read-in.

.. caution::
   Be careful about trying to overwrite parameter values defined in a
   parameter file using ``-c``. As stated above, specifying a command-line
   parameter with ``-c`` is equivalent to defining it at the bottom of the
   parameter file, which makes it take on the command-line value at the top of
   the parameter file and down until the parameter is (re)assigned (see
   :doc:`non-linear parsing of parameter file content </parameters/parameters>`
   for how this works). During the actual simulation, the value given by
   ``-c`` will be used.

   For a completely stand-alone parameter with no other parameters depending
   on its value, there is then no danger in overwriting its value using
   ``-c``. It might not be obvious whether a given parameter is stand-alone
   or not, and so it is generally cleaner to just not have any parameters be
   defined both in the parameter file and on the command-line.



.. _number_of_processes:

Number of processes: ``-n``, ``--nprocs``
.........................................
Specifies the number of MPI processes to use, with each MPI process being
mapped to its own CPU core (assuming enough of these are available). To run
using e.g. 8 processes:

.. code-block:: bash

   ./concept -n 8

If not specified, this defaults to 1.

You may write the number of processes as a (Python 3) expression. All of these
specifies that 8 processes should be used:

.. code-block:: bash

   ./concept -n "2*4"
   ./concept -n "2**3"
   ./concept -n "3 + 5"



.. _specifying_multiple_nodes:

Specifying multiple nodes
~~~~~~~~~~~~~~~~~~~~~~~~~
When running on a cluster with multiple compute nodes, you may also specify
the number of nodes to be used. The following examples all specify 8 MPI
processes distributed over 2 nodes each with 4 CPU cores:

.. code-block:: bash

   ./concept -n 2:4
   ./concept -n "2:2*2"
   ./concept -n "2 2*2"
   ./concept -n "1 + 1 : 2**2"

Note that inhomogeneous layouts are not describable. If you leave out the node
specification (i.e. only supply a single number to ``-n``) and the cluster is
running Slurm, the specified total number of CPU cores may be distributed in
any which way between the available nodes. If the cluster is running
TORQUE/PBS, you should always explicitly specify the number of nodes as well
as the number of CPU cores/node.

.. note::
   Specifying a number of nodes will enable
   :ref:`automatic job submission <submit>` unless explicitly disabled.



.. _utility:

Utility: ``-u``, ``--utility``
..............................
Signals that one of the CO\ *N*\ CEPT utilities is to be run instead of a
simulation. To run e.g. the ``powerspec`` utility, do

.. code-block:: bash

   ./concept -u powerspec </path/to/snapshot>

which will then produce a power spectrum of the snapshot file located at
``</path/to/snapshot>``.

Each utility comes with its own command-line options (for the ``powerspec``
utility, a required path to a snapshot), which you should specify together
with the normal ``concept`` command-line options. In the case of the
``powerspec`` utility, this could look like

.. code-block:: bash

   ./concept -n 4 -u powerspec </path/to/snapshot>
   # or
   ./concept -u powerspec </path/to/snapshot> -n 4

both of which will produce a power spectrum of the snapshot using 4 processes.
Some utilities have elaborate command-line interfaces of their own. Brief
summaries gets printed if you do e.g.

.. code-block:: bash

   ./concept -u powerspec -h

You can also read about the different utilities and their command-line
interfaces under :doc:`Utilities </utilities/utilities>`.

Many utilities further make use of the standard ``concept`` command-line
options, like ``-n`` to set the number of processes as in the example above,
or ``-p`` to specify a parameter file to use.

While some utilities are always run locally, the ones that potentially
involves large computational resources are subject to the same
:ref:`remote submission behaviour <remote_job_submission>` as standard
simulations.



.. _remote_job_submission:

Remote job submission
---------------------
When running CO\ *N*\ CEPT on a cluster with a job scheduler (Slurm and
TORQUE/PBS supported), directly running simulations (on the front-end) is
disabled by default (but can be :ref:`overruled <local>`). Instead, a job
script is generated. To automatically submit the job to the scheduler, supply
the ``--submit`` :ref:`option <submit>`. If running locally, none of these
options have any effect.

On top of the :ref:`basic <basics>` options, the options below are used for
additional resource specification when submitting remote jobs. Note that
additional possibilities arise for the ``-n`` option when running on a cluster
with multiple compute nodes, as documented
:ref:`above <specifying_multiple_nodes>`.



.. _submit:

Submit: ``--submit``
....................
Automatic job submission will be requested by using

.. code-block:: bash

   ./concept --submit
   # or
   ./concept --submit True

Conversely, job submission can be held back by specifying

.. code-block:: bash

   ./concept --submit False

This is almost a Boolean command-line option, with the caveat that its default
value ("unset") is neither ``True`` nor ``False``. When left unset, no
submission occurs unless other options from the
:ref:`remote job submission <remote_job_submission>` category are specified.
See the note in the description of the
``--watch`` :ref:`option <watch_cmdoption>` for details on Boolean
command-line options.



.. _queue:

Queue: ``-q``, ``--queue``
..........................
Specifies the name of the queue (called 'partition' in Slurm) to which the
remote job is to be submitted:

.. code-block:: bash

   ./concept -q <queue>

If using Slurm, you can specify multiple queues:

.. code-block:: bash

   ./concept -q <queue1>,<queue2>,<queue3>



.. _walltime:

Wall time: ``-w``, ``--walltime``
.................................
Specifies the maximum wall time (total computation time) within which the
remote job has to have completed. If the job runs for the entire specified
maximum wall time, it will be killed by the job scheduler.

You have a lot of freedom in how you want to express this time:

.. code-block:: bash

   ./concept -w 2:30:00       #  2 hours and 30 minutes
   ./concept -w 45min         #              45 minutes
   ./concept -w 100m          #  1 hour  and 40 minutes
   ./concept -w "3 days"      # 72 hours
   ./concept -w 3:12:00:00    # 84 hours
   ./concept -w "2d + 7.5hr"  # 55 hours and 30 minutes

If not specified, the system typically sets some default wall time.



.. _memory:

Memory: ``--memory``
....................
Specifies the amount of memory allocated for the remote job. If you assign
insufficient memory to a job, it will be killed (usually with a somewhat
cryptic error message) by the system once its memory needs exceed the
specified amount.

Examples of memory specifications:

.. code-block:: bash

   ./concept --memory 4GB       #   4 gigabytes
   ./concept --memory 2048MB    #   2 gigabytes
   ./concept --memory "8*2GB"   #  16 gigabytes
   ./concept --memory "0.5*TB"  # 512 gigabytes
   ./concept --mem 8G           #   8 gigabytes

Note that the specified memory is the total memory available to the job, to be
shared amongst all MPI processes / CPU cores, even when running on multiple
compute nodes.

If not specified, the system typically sets some default memory limit.



Job name: ``--job-name``
........................
Specify a name for the job, used by the job scheduler:

.. code-block:: bash

   ./concept --job-name "my job"

This name then shows up when listing your current jobs (i.e. via ``squeue`` in
Slurm and via ``qstat`` in TORQUE/PBS). If not specified, the default name
'``concept``' will be used. Having several jobs with the same name is not a
problem, as this name is not a replacement for the job ID.



Job directive: ``-j``, ``--job-directive``
..........................................
The specifications of system resources --- the designated
:ref:`queue(s) <queue>`, the
:ref:`number of nodes and processes <number_of_processes>`, the allowed
:ref:`wall time <walltime>` and the allocated :ref:`memory <memory>` --- gets
saved as 'job directives' within a job script, which is then submitted to the
job scheduler. If you desire further fine tuning of system resources, you may
supply further such directives using this command-line option.

What job directives are available depend on the job scheduler, as well as the
local configuration of the cluster and queue. As an example, consider

.. code-block:: bash

   ./concept -j="--exclusive"

which tells Slurm to give the job exclusive access to the allocated nodes, so
that the nodes will not be shared with other running jobs which could otherwise
make use of still available resources.

The effect of this is simply to place the line

   #SBATCH \-\-exclusive

in the header of the job script --- assuming Slurm --- or

   #PBS \-\-exclusive

in the case of TORQUE/PBS (the specific example of ``--exclusive``
does not mean anything to TORQUE/PBS, though).

.. caution::
   Since the value ``--exclusive`` starts with '``-``', using
   ``-j --exclusive`` or ``-j "--exclusive"`` is not legal as the
   parser registers ``--exclusive`` as a whole new (and non-existent)
   option to the ``concept`` script.

The ``-j`` option may be specified multiple times.



.. _watch_cmdoption:

Watch: ``--watch``
..................
After submitting a remote job, rather than put you back at the system prompt,
CO\ *N*\ CEPT will run the :doc:`watch utility </utilities/watch>` in order
for you to follow the progression of the job. This have no effect on the job
itself, and you may stop watching its printout using ``Ctrl``\ +\ ``C``.

If you have no desire to watch the job progression, you may specify this
as follows:

.. code-block:: bash

   ./concept --watch False

in which case the watch utility will not be run at all.

.. note::
   This is an example of a Boolean command-line option. As a value, you may
   use any of ``False``/``f``/``no``/``n``/``off``/``disable``/``0``/``0.0``
   for signifying ``False`` and any of
   ``True``/``t``/``yes``/``y``/``on``/``enable``/``1``/``1.0`` for signifying
   ``True`` (all case insensitive). In addition, specifying the command-line
   option alone with no value is the same as setting it to ``True``, i.e.
   ``--watch`` is equivalent to ``--watch True``. As ``True`` also happens to
   be the default value, supplying ``--watch`` by itself then does nothing.
   This is not so for Boolean command-line options which default to ``False``,
   e.g. the ``--local`` :ref:`option <local>`.



.. _building_and_running:

Building and running
--------------------
The following options change the way that CO\ *N*\ CEPT is built or run.

While the Python source code for CO\ *N*\ CEPT lives in the ``scr`` directory,
default invocation of the ``concept`` script launches a job that runs off of
a compiled build, placed in the ``build`` directory. If changes to the source
is detected, the code is recompiled, updating the contents of ``build``. With
the compiled code ready, the requested CO\ *N*\ CEPT run is performed.
In addition, when working on a remote server/cluster (through SSH),
rather than starting the run directly, it is submitted as a remote job.



.. _local:

Local: ``--local``
...................
Supply this option to disregard the fact that you are running on a remote
server/cluster. That is, do not submit the CO\ *N*\ CEPT run as a remote job
using the job scheduler, but run it directly as if you were running locally.

.. code-block:: bash

   ./concept --local
   # or
   ./conceot --local True

This is a Boolean command-line option, defaulting to ``False``. See the note
in the description of the ``--watch`` :ref:`option <watch_cmdoption>`
for details.



.. _pure_python:

Pure Python: ``--pure-python``
..............................
When this option is supplied, CO\ *N*\ CEPT is run directly off of the Python
source files in ``src``, disregarding the presence of
any :ref:`build <build>`:

.. code-block:: bash

   ./concept --pure-python
   # or
   ./concept --pure-python True

While handy for development, running actual simulations in pure Python mode
is impractical due to an enormous performance hit.

This is a Boolean command-line option, defaulting to ``False``. See the note
in the description of the ``--watch`` :ref:`option <watch_cmdoption>`
for details.



.. _build:

Build: ``-b``, ``--build``
..........................
Specifies a build directory to use:

.. code-block:: bash

   ./concept -b my_build

If a build already exists in this directory and is up-to-date with the source
in ``src``, it will be used. Otherwise, the code will be (re)build within this
directory. If not specified, the ``build`` directory will be used.

This option is handy if you need to maintain several builds of the code, e.g.
for different queues consisting of nodes with different
hardware architectures.

When working on a cluster, the building of the code will take place as part of
the remote job, i.e. on the compute node and not the front-end. Using a
designated build directory for a given queue / set of nodes, it is then safe
to apply architecture-dependent,
:ref:`native optimizations <native_optimizations>`.

Moreover, Bash variable expansion *at runtime* is supported, making it
possible to use a temporary scratch directory for the build. Say your cluster
creates a designated ``/scratch/<ID>`` for every job and it uses Slurm,
you could then run CO\ *N*\ CEPT as

.. code-block:: bash

   ./concept -b '/scratch/$SLURM_JOB_ID'

making use of the Slurm variable ``SLURM_JOB_ID``, holding the job ID.

.. caution::
   Note the use of single-quotes above. Were we to do

   .. code-block:: bash

      ./concept -b "/scratch/$SLURM_JOB_ID"  # wrong!

   it would not work, as ``$SLURM_JOB_ID`` would be expanded right there on
   the command-line, where it does not hold a value.

.. caution::
   When the code is built as part of the job, the build process will be
   carried out on the 'master node' only. For multi-node jobs it is thus
   important to choose a build directory that can be accessed from all nodes.
   This may not be the case for the temporary scratch directory, as this may
   be local to each node, so that the same path ``/scratch/$SLURM_JOB_ID``
   really corresponds to a separate directory on each node.



.. _rebuild:

Rebuild: ``--rebuild``
......................
Even if the compiled code in the :ref:`build <build>` directory is up-to-date
with the source in ``src``, a rebuild can be triggered by using

.. code-block:: bash

   ./concept --rebuild
   # or
   ./concept --rebuild True

Conversely, an out-of-date build will be used as is, if you specify

.. code-block:: bash

   ./concept --rebuild False

This is almost a Boolean command-line option, with the caveat that its
default value ("unset") is neither ``True`` nor ``False``, as by default the
code is rebuild conditionally, depending on whether the present build is
up-to-date with the source. See the note in the description of the
``--watch`` :ref:`option <watch_cmdoption>` for details on Boolean
command-line options.



.. _optimizations:

Optimizations: ``--optimizations``
..................................
During compilation of CO\ *N*\ CEPT, a lot of optimizations are performed.
These include source code transformations performed by ``pyxpp.py`` as well
as standard C compiler optimizations (including
:ref:`LTO <link_time_optimizations>`), applied in the ``Makefile``. Though
these optimizations should not be disabled under normal circumstances, you may
do so by supplying

.. code-block:: bash

   ./concept --optimizations False

.. note::
   If the :ref:`build <build>` directory already contains compiled code that
   is up-to-date with the source at ``src`` (built with or without
   optimizations), the code will not be rebuild. To rebuild without
   optimizations, you can make use of the :ref:`rebuild <rebuild>` option:

   .. code-block:: bash

      ./concept --optimizations False --rebuild

This is a Boolean command-line option, defaulting to ``True``. See the note
in the description of the ``--watch`` :ref:`option <watch_cmdoption>`
for details.



.. _link_time_optimizations:

Link time optimizations: ``--linktime-optimizations``
.....................................................
Link time optimizations (LTO) are on by default when compiling CO\ *N*\ CEPT,
if supported by the compiler. Though usually preferable, the associated
increase in build time and especially memory may be undesirable. To build
CO\ *N*\ CEPT without link time optimizations, supply

.. code-block:: bash

   ./concept --linktime-optimizations False

If optimizations are :ref:`disabled generally <optimizations>`
(``--optimizations False``), LTO are disabled as well.

.. note::
   If the :ref:`build <build>` directory already contains compiled code that
   is up-to-date with the source at ``src`` (built with or without LTO),
   the code will not be rebuild. To rebuild without LTO, you can make use of
   the :ref:`rebuild <rebuild>` option:

   .. code-block:: bash

      ./concept --linktime-optimizations False --rebuild

This is a Boolean command-line option, defaulting to ``True``. See the note
in the description of the ``--watch`` :ref:`option <watch_cmdoption>`
for details.



.. _native_optimizations:

Native optimizations: ``--native-optimizations``
................................................
The default optimizations performed during compilation are all portable, so
that the compiled code may be run on different hardware (within reason).

Supply this option if you are willing to use non-portable optimizations native
to your particular system, by which it is possible to squeeze a further few
percent performance increase out of the compilation:

.. code-block:: bash

   ./concept --native-optimizations
   # or
   ./concept --native-optimizations True

Specifically, this adds the ``-march=native`` compiler optimization.

Enabling native optimizations is not possible if optimizations are
:ref:`disabled generally <optimizations>` (``--optimizations False``).

If working on a cluster with nodes of different architectures, having separate
build directories for these is recommended if building with native
optimizations. See the :ref:`build <build>` option for details.

.. note::
   If the :ref:`build <build>` directory already contains compiled code that
   is up-to-date with the source at ``src`` (built with or without
   native optimizations), the code will not be rebuild. To rebuild with native
   optimizations, you can make use of the :ref:`rebuild <rebuild>` option:

   .. code-block:: bash

      ./concept --native-optimizations --rebuild

This is a Boolean command-line option, defaulting to ``False``. See the note
in the description of the ``--watch`` :ref:`option <watch_cmdoption>`
for details.



Safe build: ``--safe-build``
............................
By default the compilation process is carried out in a safe manner, meaning
that changes within a file triggers recompilation of all other files which
rely on the specific file in question. If you know that a change is entirely
internal to the given file, you may save yourself some compilation time by
supplying

.. code-block:: bash

   ./concept --safe-build False

in which case all interdependencies between the files will be ignored
during compilation.

.. caution::
   Using ``--safe-build False`` really *is* unsafe and may very well lead to a
   buggy build. To clean up after an unsuccessful build, use the
   :ref:`rebuild <rebuild>` option or remove the ``build`` directory
   entirely using

   .. code-block:: bash

      (source concept && make clean)

This is a Boolean command-line option, defaulting to ``True``. See the note
in the description of the ``--watch`` :ref:`option <watch_cmdoption>`
for details.



.. _specials:

Specials
--------
The following options are mostly useful for development. As demonstrated by
the below examples though, "development" does not have to be restricted to the
CO\ *N*\ CEPT code itself. That is, these special options come in handy should
you wish to hook into the code, or simply check that the code is in a working
state.



.. _test:

Test: ``-t``, ``--tests``
.........................
CO\ *N*\ CEPT comes with an integration test suite, located in the
``test`` directory. Each subdirectory within this directory implements a given
test, with the test name given by the name of the subdirectory. You may use
this option to run these tests, checking that the code works correctly.

To run e.g. the ``concept_vs_gadget_p3m`` test --- which runs a small
CO\ *N*\ CEPT P³M simulation and a (supposedly) equivalent GADGET-2 TreePM
simulation, after which it compares the results --- do e.g.

.. code-block:: bash

   ./concept -t concept_vs_gadget_p3m

The path to the test may be specified in any of these formats:

* Absolutely.
* Relative to your current working directory.
* Relative to the CO\ *N*\ CEPT installation directory.
* Relative to the ``test`` directory.

Tests are always performed locally, i.e. not submitted as remote jobs even
when working on a remote cluster/server.

Once a test is complete, it will report either success or failure. Most
tests also produce some artefacts within their subdirectory, most notably
plots.

Multiple tests may be specified together:

.. code-block:: bash

   ./concept -t concept_vs_gadget_p3m render

The entire test suite may be run using

.. code-block:: bash

   ./concept -t all

which runs each test sequentially. If one of the tests fails, the process
terminates immediately. To clean up after all tests, do

.. code-block:: bash

   (source concept && make clean-test)

.. note::
   When testing with an initially clean ``job`` directory, any warnings
   produced by tests will count as errors, leading to test failure. This
   behaviour is not present if starting from a non-empty ``job`` directory.



.. _main_entry_point:

Main entry point: ``-m``, ``--main``
....................................
The responsibilities of the ``concept`` script are to set up the environment,
build the actual CO\ *N*\ CEPT code (if not running in
:ref:`pure Python mode <pure_python>`) and then launch a job (or submit a
remote job which then is launched from within a generated job script).
Skipping over many details, this final launch step looks something like

.. code-block:: bash

   python -m build/main.so  # compiled mode
   python src/main.py       # pure Python mode

which fires up the ``main`` module, which in turn initiates the simulation.
We can use this option to switch out the default ``main`` module for some
other main entry point.

As an example, consider the ``hubble.py`` script below:

.. code-block:: python3
   :caption: hubble.py
   :name: hubble

   from commons import *  # import everything from the CO𝘕CEPT commons module

   h = H0/(100*units.km/(units.s*units.Mpc))
   print(f'{h = :g}')

We can run this script using

.. code-block:: bash

   ./concept \
       -m hubble.py \
       --pure-python

.. note::
   Here we need :ref:`pure Python mode <pure_python>` since the ``hubble.py``
   script itself is not compiled, and the compiled ``commons`` module does
   not expose ``H0`` as a Python-level object.

To see that this really does hook into the CO\ *N*\ CEPT machinery:

.. code-block:: bash

   ./concept \
       -m hubble.py \
       -c "H0 = 72*km/(s*Mpc)" \
       --pure-python

Using ``-m`` with a script as above is convenient for e.g. making plotting
scripts which need access to some CO\ *N*\ CEPT functionality. For small
printouts, we can specify the Python code directly as the value to ``-m``. To
e.g. get the value of the gravitational constant implemented in CO\ *N*\ CEPT,
we can do

.. code-block:: bash

   ./concept \
       -m "from commons import *; \
           unit = units.m**3/(units.kg*units.s**2); \
           print(f'{G_Newton/unit:g}'); \
       " \
       --pure-python \
   | tail -n 1



.. _interactive:

Interactive: ``-i``, ``--interactive``
......................................
Normally when running ``concept``, you are sent back to the system prompt once
the run has completed. You can use this option to instead end up in a live
Python prompt, with the :ref:`main <main_entry_point>` scope available for
exploration and further execution.

You could run a simulation interactively if you wanted to:

.. code-block:: bash

   ./concept \
       -c "a_begin = 0.1" \
       -c "initial_conditions = {'species': 'matter', 'N': 16**3}" \
       -i \
       --pure-python

.. code-block:: python3

   >>> output_times['a']['render2D'] = (1, )
   >>> timeloop()
   >>> exit()

where ``>>>`` indicates input which should be typed at the interactive Python
prompt. We use :ref:`pure Python mode <pure_python>` as our interactive
commands are interpreted directly by Python.

.. note::
   At least when using the standard
   :ref:`main entry point <main_entry_point>`, running in interactive mode
   using :ref:`more than one process <number_of_processes>` is not a good
   idea, as the interactive prompt is attached to the master process only.

While the above example requires knowledge of the internal code and serves no
real use outside of development, a perhaps more useful usage of interactive
mode is to combine it with the :ref:`main entry point <main_entry_point>`
option when writing auxiliary scripts, or to use it purely exploratory.

Say we did not know the variable name and value of the gravitational constant
implemented in CO\ *N*\ CEPT, an we wanted to find out. We might go exploring,
doing something like

.. code-block:: bash

   ./concept -i --pure-python

.. code-block:: python3

   >>> # 💭 What variables are available?
   >>> dir()

   >>> # 💭 I see one called 'G_Newton'
   >>> G_Newton

   >>> # 💭 Its value looks unrecognisable! It must not be given in SI units.
   >>> # 💭 I see something called 'units'.
   >>> units
   >>> units.m
   >>> units.Mpc

   >>> # 💭 What if I try ...
   >>> G_Newton/(units.m**3/(units.kg*units.s**2))

   >>> # 💭 Success!
   >>> exit()

This is a Boolean command-line option, defaulting to ``False``. See the note
in the description of the ``--watch`` :ref:`option <watch_cmdoption>`
for details.



Version: ``-v``, ``--version``
..............................
Prints out the version of CO\ *N*\ CEPT that is installed:

.. code-block:: bash

   ./concept -v

