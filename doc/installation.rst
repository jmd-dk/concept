Installation
============
This page contains detailed information about how to install CO\ *N*\ CEPT
along with its many dependencies. Unless you have some special need, it is
recommended to just follow the simple,
:ref:`standard installation <standard_installation>` instructions, including
those covering how to obtain
:ref:`optimal network performance <optimal_network_performance_on_clusters>`
if you intend to install CO\ *N*\ CEPT on a cluster.

The :ref:`standard <standard_installation>` way of installing CO\ *N*\ CEPT
utilizes the ``install`` script to install CO\ *N*\ CEPT and all of its
dependencies. Many more details about how to use this script for more
customized installation is available under the
':ref:`install script in-depth <the_install_script_in_depth>`' entry.

Though not recommended, you may choose to not make use of the ``install``
script, in which case you need to install all of the
:ref:`dependencies <dependencies>` yourself. After
`downloading <https://github.com/jmd-dk/concept>`__ the CO\ *N*\ CEPT source
code, you then need to
:ref:`specify environment information <the_path_and_env_files>` by editing
the ``.path`` and ``.env`` files.


Entries on this page:

.. contents::
   :local:
   :depth: 2



.. _supported_platforms:

Supported platforms
-------------------

.. |linux| image:: /static/linux.png
   :height: 35px

.. |windows| image:: /static/windows.png
   :height: 35px

.. |macos| image:: /static/macos.png
   :height: 35px

.. |docker| image:: /static/docker.png
   :height: 35px

.. tabs::

   .. tab:: |linux| :math:`\,\,` Linux

      CO\ *N*\ CEPT should be :ref:`trivial to install <standard_installation>`
      on all major Linux distributions. The system may be a laptop, a
      workstation, a massively parallel computer cluster, a Raspberry Pi,
      a virtual machine, etc.

   .. tab:: |windows| :math:`\,\,` Windows

      Though CO\ *N*\ CEPT does not run natively on Windows, support is
      obtained via the
      `Windows Subsytem for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`__.
      If installing CO\ *N*\ CEPT on Windows this way, make sure to install it
      into a proper Linux directory (e.g. your home directory ``~``) and not a
      Windows directory (e.g. ``/mnt/c/...``).

      The more traditional approach of installing Linux in a virtual machine
      within Windows, or running CO\ *N*\ CEPT through Docker, of course works
      too.

   .. tab:: |macos| :math:`\,\,` macOS

      Currently the only means of running CO\ *N*\ CEPT on a mac are either to
      install it within a virtual Linux machine or run it through Docker.

      Though much of the POSIX infrastructure needed to install and run
      CO\ *N*\ CEPT is available to macOS via `Homebrew <https://brew.sh/>`__,
      numerous incompatibilities between the GNU and BSD tools make the
      porting to macOS non-trivial.

   .. tab:: |docker| :math:`\,\,` Docker

      You may wish to run CO\ *N*\ CEPT through one of the provided
      `Docker images <https://hub.docker.com/r/jmddk/concept/>`__, freeing you
      from the installation process entirely (assuming you have
      `Docker <https://www.docker.com/>`__ already installed).

      To start a Docker container based on the latest CO\ *N*\ CEPT Docker
      image, do e.g.

      .. code-block:: bash

         docker run --rm -it -v "${PWD}:/concept/output" jmddk/concept

      Any files dumped in the CO\ *N*\ CEPT ``output`` directory will then
      persist in your current directory after the Docker container is stopped.

      .. note::
         If running Windows, the above command is valid within
         `PowerShell <https://docs.microsoft.com/powershell/>`__, not
         `CMD <https://docs.microsoft.com/windows-server/administration/windows-commands/windows-commands>`__

      .. note::
         While running CO\ *N*\ CEPT via Docker is great for experimental use,
         :ref:`proper installation <standard_installation>` on a Linux host is
         preferable for running large simulations, ensuring maximum
         performance.

      .. tip::
         All available
         `Docker images <https://hub.docker.com/r/jmddk/concept/>`__ are based
         on `Debian <https://hub.docker.com/_/debian>`__. Newer images will
         have the `nano text editor <https://www.nano-editor.org/>`__
         installed. If you would like access to additional software (e.g. your
         text editor of choice or Git) within a Docker container, first run

         .. code-block:: bash

            sudo apt update
            sudo apt install -y ca-certificates 2>/dev/null || sudo apt install -y ca-certificates
            sudo apt install -y apt-utils
            sudo apt install -y gfortran

         after which you may search for software packages to install using

         .. code-block:: bash

            apt search <keyword>

         and install a package using

         .. code-block:: bash

            sudo apt install <package>

         Note that the default user (``concept``) has root access via ``sudo``
         with no password required.



.. _standard_installation:

Standard installation
---------------------
The easiest way to install CO\ *N*\ CEPT along with all of its dependencies is
to use the CO\ *N*\ CEPT ``install`` Bash script. This script is part of the
CO\ *N*\ CEPT source code, and so if you have CO\ *N*\ CEPT already
`downloaded <https://github.com/jmd-dk/concept>`__, you may locate and run
this script:

.. code-block:: bash

   bash install

However, the simplest way to run the script is to just execute

.. code-block:: bash

   concept_version=v1.0.1
   bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/install)

in a terminal, which will fetch and run the ``install`` script directly from
the CO\ *N*\ CEPT GitHub repository, specifically version 1.0.1. Check out the
CO\ *N*\ CEPT
`releases <https://github.com/jmd-dk/concept/releases>`__ for other available
release versions, or use ``concept_version=master`` for the absolute newest
(and unstable!) version.

.. note::
   The initial ``bash`` in the above commands is required regardless of your
   shell

.. note::
   Your system may fail to verify the certificate of the domain name
   (``raw.githubusercontent.com``), causing ``wget`` to refuse the download.
   If you encounter this issue, you can supply ``--no-check-certificate`` as a
   further option to ``wget``.

.. tip::
   If you are installing CO\ *N*\ CEPT on a cluster, you should make sure to
   obtain
   :ref:`optimal network performance <optimal_network_performance_on_clusters>`.
   If you are installing via the cluster's front-end over SSH, you may want to
   make use of a tool such as `tmux <https://github.com/tmux/tmux/wiki>`__ or
   `Screen <https://www.gnu.org/software/screen/>`__, so that you may close the
   connection without stopping the installation process.

.. note::
   It is recommended to have `GCC <https://gcc.gnu.org/>`__ 6.5 or later
   installed. If you are installing CO\ *N*\ CEPT on a cluster, various
   versions of GCC may be available, e.g. through the
   `module system <https://hpc-wiki.info/hpc/Modules>`__.

.. note::
   For the remainder of this page, we shall make use of the shorter
   ``bash install`` syntax, but here the local file '``install``' may
   always be substituted with the online '``<(wget ...)``' version.

The ``install`` script will prompt you for an installation directory, as well
as for permission to install each
:ref:`system dependency <system_dependencies>`, should any be missing. At the
very beginning of the installation, a table of software to be installed will
be shown.

Should the installation process end prematurely (e.g. if you kill
it yourself), simply rerun the same installation command and it will pick up
from where it was.

CO\ *N*\ CEPT along with all of the libraries it needs will be installed into
the chosen installation directory. To completely remove the CO\ *N*\ CEPT
installation, simply remove this directory.

The installation will take a couple of hours on modern hardware. When
installing onto a cluster, this time is increased as the default policy of the
``install`` script is to then not build in parallel, though this can
:ref:`be changed<parallel_builds>`.

For each program, the ``install`` script will attempt the installation using
various combinations of flags until a successful set is found. For
this reason, installing CO\ *N*\ CEPT on atypical systems (e.g. non-x86) may
take a lot longer.

The fully installed CO\ *N*\ CEPT distribution takes up about 1.6 GB
of storage. This can be reduced to about 0.3 GB if installing in 'slim' mode,
see the ``--slim`` :ref:`command-line option <command_line_options>`.



.. _optimal_network_performance_on_clusters:

Optimal network performance on clusters
.......................................
To install CO\ *N*\ CEPT on a cluster, the standard installation procedure
above may be used. However, the performance of simulations running on multiple
CPUs may be very poor, as the MPI library installed by the ``install`` script
may not be configured optimally regarding the network and accompanying
software present on the cluster.

To test the network performance of MPI, you may simply run a small simulation
using 1, 2 and 4 processes. If the various timings printed out by the code
increases significantly as you increase the number of processes, you are
affected by this issue. Furthermore, running a simulation on 2 nodes each with
2 processes should not be significantly slower than running on a single node
with 4 processes, at least not if your cluster features a high-speed network
such as InfiniBand.

If you are affected by this issue, it is recommended to reinstall
CO\ *N*\ CEPT using an MPI library already present on the cluster, presumably
configured optimally by the system administrator. To do this, simply set the
``mpi_dir`` variable to the directory containing this MPI library when
invoking the ``install`` script, e.g.

.. code-block:: bash

   mpi_dir=/path/to/mpi bash install

It is important that your environment is set up so that the MPI installation
in question is runnable, meaning e.g. that ``/path/to/mpi/bin/mpicc`` works.
On clusters this often requires enabling MPI through something like the
``module load`` command.

Note that you *will* have to reinstall CO\ *N*\ CEPT and its dependencies in
their entirety, if you want to swap out the MPI library.

.. tip::
   If you have ``/path/to/mpi/bin`` already in your ``PATH`` you may use
   simply

   .. code-block:: bash

      mpi_dir="$(which mpicc)" bash install

   Though ``which mpicc`` does not evaluate exactly to ``/path/to/mpi``, it
   is close enough that the ``install`` script understands it.

For the MPI implementation, CO\ *N*\ CEPT officially supports MPICH and
OpenMPI, though it should not matter much (CO\ *N*\ CEPT has been successfully
tested on MVAPICH and Spectrum MPI). What is important is that the MPI library
used conforms to the MPI-3 (or MPI-3.1) standard.



Cloning with Git
----------------
For installing CO\ *N*\ CEPT, cloning the code repository is not needed, as
a copy of the code (without Git history) is downloaded during
:ref:`installation <standard_installation>`.

If however you wish to work with a full clone of the CO\ *N*\ CEPT
`code repository <https://github.com/jmd-dk/concept/>`__, the easiest way to
do so is to first install the master version of CO\ *N*\ CEPT *without* Git,
following the
:ref:`standard installation instructions <standard_installation>`. Once
CO\ *N*\ CEPT is installed, ``git clone`` the online CO\ *N*\ CEPT repository
into a temporary directory and simply drop the ``.git`` subdirectory into the
root of your CO\ *N*\ CEPT installation.

The above can be achieved from within the CO\ *N*\ CEPT installation directory
using

.. code-block:: bash

   git clone https://github.com/jmd-dk/concept.git git
   mv git/.git .
   rm -rf git

If you now do a

.. code-block:: bash

   git status

it should detect changes to the ``.path`` and ``.env``
:ref:`files <the_path_and_env_files>` only. These changes represent
customizations carried out during installation.

.. tip::
   If you do not want Git to show the local changes to the ``.path`` and
   ``.env`` files, you can tell Git to ignore these. The recommended way of
   doing so is as follows:

   .. code-block:: bash

      (source concept \
          && git stash push "${path}" "${env}" -m path-env \
          && git stash apply stash^{/path-env} \
          && git update-index --skip-worktree "${path}" "${env}" \
          && git status \
      )

   Should you need to undo this, run

   .. code-block:: bash

      (source concept && git update-index --no-skip-worktree "${path}" "${env}")

   If you need to explicitly (re)insert the ``.path`` and ``.env`` files
   (e.g. because you changed to a different branch), you can use

   .. code-block:: bash

      git stash apply stash^{/path-env}



.. _the_install_script_in_depth:

The ``install`` script in-depth
---------------------------------
The ``install`` script may be run with additional options or influential
environment variables, which can be used to e.g. request for a specific
version of a dependency to be installed, or to skip the installation of a
particular dependency and instead make use of a pre-installed version of the
same library. Before discussing such features, it is good to know exactly
what dependencies are installed.


Programs installed
..................
The ``install`` script partitions all dependencies into *library* and *system*
dependencies. Unless explicitly specified, the ``install`` script installs all
library dependencies regardless of whether these already exist on the system.
The system dependencies consist of standard tools commonly installed
system-wide. If the ``install`` script detects a missing system dependency, it
will prompt for system-wide (root) installation through the package manager on
the system.

The ``install`` script is able to bootstrap itself up from just Bash, GNU
Core Utilities and a package manager (or just Bash and GNU Core Utilities if
all system dependencies are already present). In addition, it builds all
library dependencies from source, increasing the total number of dependencies
(as many dependencies have other dependencies, and so on). For the absolute
minimum dependency requirements needed to *run* CO\ *N*\ CEPT, see the
:ref:`Dependencies <dependencies>` entry.

The complete set of system dependencies needed/installed by the ``install``
script is:

* **GNU tools**: AWK, grep, sed, Gzip, Tar, Wget.
* **Build tools**: GCC, g++, gfortran, glibc, GNU Make, as and ld (binutils),
  Linux headers.

The complete list of all library dependencies (and their dependency relations)
installed by the ``install`` script is given below:

* **zlib**
* **Perl**
* **BLAS** and **LAPACK**
* **MPI**
* **FFTW 3**
* **FFTW 2**
* **ncurses**
* **libffi**
* **OpenSSL** (depends on Perl)
* **GSL** (depends on BLAS)
* **HDF5** (depends on zlib)
* **GADGET-2** (depends on MPI, GSL, FFTW 2)
* **Python 3** (depends on zlib)

  - **pip** (depends on OpenSSL, libffi), **setuptools** and **wheel**;
    needed to install the Python packages below:

    - **Blessings** (depends on ncurses)
    - **Cython**
    - **CythonGSL** (depends on Cython, GSL)
    - **NumPy** (depends on BLAS, LAPACK)
    - **SciPy** (depends on BLAS, LAPACK)
    - **Matplotlib**
    - **MPI4Py** (depends on MPI, Cython)
    - **H5Py** (depends on HDF5 and MPI)
    - **Sphinx**
    - **Sphinx-copybutton** (depends on Sphinx)
    - **Sphinx-rtd-theme** (depends on Sphinx)
    - **Sphinx-tabs** (depends on Sphinx)

* **CLASS** + **classy** (depends on Cython, NumPy)

Finally, CO\ *N*\ CEPT itself depends directly on MPI, FFTW (3), GADGET-2,
Python, Blessings, Cython, CythonGSL, NumPy, SciPy, Matplotlib, MPI4Py, H5Py,
classy. For testing, CO\ *N*\ CEPT additionally needs GADGET-2, and building
documentation requires Sphinx, sphinx-copybutton, sphinx-rtd-theme,
sphinx-tabs.

Many of the above Python packages have other Python packages as dependencies,
and so the truly complete list of these is in fact much larger. This is mostly
a non-issue, as such Python package dependencies are generally auto-installed
when installing Python packages with `pip <https://pip.pypa.io/>`__.

The ``install`` script installs the
`OpenBLAS <https://github.com/xianyi/OpenBLAS>`__ library (which depends on
Perl) in order to provide both BLAS and LAPACK. For MPI,
`MPICH <https://www.mpich.org/>`__
(:ref:`default <installing_mpich_or_openmpi>`) or
`OpenMPI <https://www.open-mpi.org/>`__ is installed (both depends on Perl). If
tests are to be performed during the installation (see the ``--tests``
:ref:`command-line option <command_line_options>` to ``concept``), the pytest
and Hypothesis Python packages will additionally be installed (needed for
testing e.g. NumPy).



.. _command_line_options:

Command-line options
....................
When invoking the ``install`` script --- whether a local copy or directly off
of GitHub --- you may supply optional command-line arguments, the most useful
of which is probably the installation path. That is,

.. code-block:: bash

   bash install /path/to/concept

will not prompt you for an installation directory but instead use the supplied
``/path/to/concept``.

Other command-line options to the ``install`` script are listed below.

.. raw:: html

   <h6>
     Help:
     <code class="docutils literal notranslate"><span class="pre">
       -h
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --help
     </span></code>
   </h6>

This prints out a short description of how to use the ``install`` script
and then exits. Generally though, this documentation page is preferable.

.. raw:: html

   <h6>
     Tests:
     <code class="docutils literal notranslate"><span class="pre">
       -t
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --tests
     </span></code>
   </h6>

With this option, close to all dependency programs will be tested after their
individual installations. CO\ *N*\ CEPT itself will similarly be tested. On
failure (fatal or non-fatal) of any test, a log file of the test output will
be placed in the installation subdirectory of the given program. Any test
failures will be reported at the end of the entire installation process.

This option is helpful for debugging if it is known that the installation
results in a non-functioning CO\ *N*\ CEPT, but it is unknown which of the
many dependencies does not function correctly. Generally though, this option
is not recommended as it increases the installation time by a couple of hours.

The tests performed on the CO\ *N*\ CEPT code itself are those of

.. code:: bash

   ./concept -t all

and so may also be run at any time after the installation, whether or not the
``--tests`` option was used for the installation. See the ``concept``
:ref:`test <test>` command-line option for further details.

.. raw:: html

   <h6>
     Yes to defaults:
     <code class="docutils literal notranslate"><span class="pre">
       -y
     </span></code>
     ,
     <code class="docutils literal notranslate"><span class="pre">
       --yes-to-defaults
     </span></code>
   </h6>

Assume "yes" as answer to all prompts (accept defaults) and run
non-interactively. These include system-wide installations of system
dependencies (should any be missing), which require root access.
Also, any other already running CO\ *N*\ CEPT installation process will be
terminated on sight.

.. raw:: html

   <h6>
     Slim:
     <code class="docutils literal notranslate"><span class="pre">
       --slim
     </span></code>
   </h6>

This produces a slimmed down installation with uncritical content removed,
both from CO\ *N*\ CEPT itself and its dependency programs. This brings the
size of the installation down from about 1.6 GB to about 0.3 GB. Though
CO\ *N*\ CEPT itself remains fully functional in such a slim installation,
some features of the dependency programs will be broken. Debugging any issues
encountered will also generally be harder.

.. note::

   The provided CO\ *N*\ CEPT :ref:`Docker images <supported_platforms>` are
   built using a slim installation

Below we list the exact changes caused by the ``--slim`` option.

* **Fewer license and acknowledgement files**: During normal installation, all
  recognised license and acknowledgement files for the various dependencies
  (and CO\ *N*\ CEPT itself) are copied over to the respective installation
  directories. This is now skipped.

* **Less documentation**: Most documentation (full-blown documentation,
  README's, changelogs, etc.) for the dependency programs will be removed. The
  documentation for CO\ *N*\ CEPT itself (these pages) will not be built but
  remains available as source.

* **Fewer example and test files**: Example files demonstrating how to use the
  various dependency programs, as well as test suites for demonstrating their
  correctness, will be removed.

* **Fewer dependency programs**: Some dependency programs are only strictly
  needed during installation of the dependency stack, not for running
  CO\ *N*\ CEPT itself. These will be completely removed. The dependencies in
  question are: Perl, zlib.

* **No static libraries of the dependency programs**: All static library files
  belonging to the installed dependencies will be removed, meaning that any
  further linking against these must be done using *shared* libraries
  (if available).

* **Fewer symbols in binaries**: The majority of all binary files will have
  their symbol table stripped.

* **Fewer configuration files**: The majority of configuration files
  describing the installation of the dependencies will be removed.

* **No cached Python files**: All bytecode files compiled and cached by Python
  during installation will be removed.

* **No compilation artefacts from CO**\ :bolditalic:`N`\ **CEPT and CLASS**:
  Only the final shared object files (as well as the bare source) resulting
  from compilation of CO\ *N*\ CEPT and CLASS will be present after
  installation.

.. raw:: html

   <h6>
     Fix SSH:
     <code class="docutils literal notranslate"><span class="pre">
       --fix-ssh
     </span></code>
   </h6>

.. warning::
   Do *not* use this option if you seek to install CO\ *N*\ CEPT

Invoking the ``install`` script with this option will not install
CO\ *N*\ CEPT, but instead attempt to configure the local ``~/.ssh`` directory
of the user for use with remote jobs running on multiple nodes, as described
:ref:`here <problems_when_using_multiple_nodes>`.



.. _influential_environment_variables:

Influential environment variables
.................................
The behaviour of the ``install`` script is governed by a large set of
environment variables. An example is the ``mpi_dir`` variable described in
':ref:`optimal network performance on clusters <optimal_network_performance_on_clusters>`',
through which we can let the ``install`` script make use of a pre-installed
MPI library, rather than letting it install one itself. We can specify
``mpi_dir`` either directly in the invocation of ``install``;

.. code-block:: bash

   mpi_dir=/path/to/mpi bash install

or defining it as an environment variable prior to the invocation;

.. code-block:: bash

   export mpi_dir=/path/to/mpi  # assuming Bash-like shell
   bash install

All other influential environment variables may be set in similar ways.



.. _making_use_of_preinstalled_libraries:

Making use of pre-installed libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To make the ``install`` script make use of a pre-installed library rather than
installing it itself, you must set the corresponding ``*_dir`` variable. The
complete list of such variables is: ``blas_dir``, ``class_dir``,
``concept_dir``, ``fftw_dir``, ``fftw_for_gadget_dir``, ``gadget_dir``,
``gsl_dir``, ``hdf5_dir``, ``libffi_dir``, ``mpi_dir``, ``ncurses_dir``,
``openssl_dir``, ``perl_dir``, ``python_dir``, ``zlib_dir``.

Note that if using a pre-installed Python distribution ---
``python_dir=/path/to/python`` --- OpenSSL, libffi and ncurses are assumed to
be already installed and built into the Python distribution, as these cannot
be tacked on after Python is build. Also, Python should come with at least pip
built-in. The ``install`` script will install any other missing
Python packages.

If e.g. ``mpi_dir`` is set, the value of ``mpi_version`` is not used.



.. _specifying_dependency_versions:

Specifying dependency versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The version of each dependency (and CO\ *N*\ CEPT itself) to install is
specified near the top of the ``install`` script (look for 'Specification
of software versions'). You may direct the ``install`` script to use other
versions through the ``*_version`` variables. As with the ``*_dir``
:ref:`variables <making_use_of_preinstalled_libraries>`, these include
``blas_version``, ``class_version``, ``concept_version``, ``fftw_version``,
``fftw_for_gadget_version``, ``gadget_version``, ``gsl_version``,
``hdf5_version``, ``libffi_version``, ``mpi_version``, ``ncurses_version``,
``openssl_version``, ``perl_version``, ``python_version``, ``zlib_version``.
Furthermore, each Python package also has a version, specified by
``blessings_version``, ``cython_version``, ``cythongsl_version``,
``h5py_version``, ``matplotlib_version``, ``mpi4py_version``,
``numpy_version``, ``pip_version``, ``scipy_version``, ``setuptools_version``,
``sphinx_version``, ``sphinx_copybutton_version``,
``sphinx_rtd_theme_version``, ``sphinx_tabs_version``, ``wheel_version``.



Specifying dependency URLs
~~~~~~~~~~~~~~~~~~~~~~~~~~
When a ``*_dir`` :ref:`variable <making_use_of_preinstalled_libraries>` is
not set for a dependency (or CO\ *N*\ CEPT itself), the ``install`` script
will download the source code of the dependency using a pre-defined URL.
Of course, the exact URL needs to factor in the version of the dependency,
and so this depends on the corresponding ``*_version``
:ref:`variable <specifying_dependency_versions>`. If you wish to install some
non-standard version of a given dependency, you can provide the corresponding
``*_url`` variable yourself. This should point to some downloadable archive
(``.tar.gz``, ``.zip``, etc.). The complete list of ``*_url`` variables is:
``blas_url``, ``class_url``, ``concept_url``, ``fftw_url``,
``fftw_for_gadget_url``, ``gadget_url``, ``gsl_url``, ``hdf5_url``,
``libffi_url``, ``mpi_url``, ``ncurses_url``, ``openssl_url``, ``perl_url``,
``python_url``, ``zlib_url``.



.. _choosing_compiler_precedence:

Choosing compiler precedence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
At the beginning of the installation, the ``install`` script will locate the
various compilers on the system. Its findings are presented under the
'Compiler precedence' heading. If the installation of a given program fails,
the ``install`` script moves on to the next compiler and tries again,
and so on. This is part of the overall philosophy of the ``install`` script to
"try, try again" if a particular choice of compiler/flags/etc. does
not succeed. This is one reason why the installation can be so time consuming.
It does however make the installation process very robust.

If you wish to have a say in the order in which the compilers are tried out,
you may define the ``compiler_precedence`` variable. To prefer e.g. Intel
compilers and then GNU compilers, set ``compiler_precedence="intel gnu"``.
Only compilers found on the system will be taken into account. Also, say the
system further has e.g. the Clang compiler, this will be tried out also, but
after any compilers specified in ``compiler_precedence``.

The supported compilers --- written in order of default precedence --- are:

- ``specified_mpi``: Use the compilers included in the MPI library
  specified by ``mpi_dir``.

- ``default``: Run as is, without altering the environment. If e.g. the ``CC``
  environment variable is set, this will probably be picked up by the
  installation of the program.

- ``gnu``: The GNU compilers (``gcc``, ``g++``, ``gfortran``).

- ``clang``: The Clang compilers (``clang``, ``clang++``).

- ``mpi``: MPI compilers found on the system (``mpicc``, ``mpicxx``,
  ``mpifort``, ...).

- ``intel``:  The Intel compilers (``icc``, ``icpc``, ``ifort``).

- ``cray``: The Cray compilers (``craycc``, ``crayCC``, ``crayftn``).

- ``portland``: The Portland compilers (``pgcc``, ``pgCC``,
  ``pgf77``, ``pgf90``).

- ``generic``: Non-specific compilers found on the system (``cc``,
  ``c++``, ``fortran``).

- ``unset``: Explicitly unset environment variables such as ``CC``, ``CXX``,
  ``FC``, and run without choosing any compiler.

Many of the dependency programs do some compiler discovery of their own, and
so no guarantee of what compiler is actually used can be given.



.. _installing_mpich_or_openmpi:

Installing MPICH or OpenMPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you let the ``install`` script install its own MPI library (i.e. leave
``mpi_dir`` unspecified), you may choose between MPICH and OpenMPI by setting
either ``mpi=mpich`` or ``mpi=openmpi``. If ``mpi`` is left unset, MPICH is
installed. Note that the same variable ``mpi_version`` is used regardless of
which MPI implementation is used.



.. _parallel_builds:

Parallel builds
~~~~~~~~~~~~~~~
Much of the installation process can be sped up if we allow the Make tool
to build in parallel. This is controlled through the ``make_jobs`` variable.
To enforce serial builds, set ``make_jobs="-j 1"``. To enforce parallel builds
using e.g. 2 processors, specify ``make_jobs="-j 2"``. You can also specify an
unlimited amount of available parallel processors using just
``make_jobs="-j"``.

By default, when ``make_jobs`` is not specified, unlimited parallel builds are
used if installing locally, while serial builds are used if working remotely.



Using the ``install`` script to install specific libraries but not CONCEPT itself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``install`` script may be used outside the context of CO\ *N*\ CEPT,
should you be in need of any of the dependencies for some other purpose. Which
programs to install is governed by ``*_install`` variables. By default,
``concept_install=True``, which in turn sets ``*_install=True`` for its
immediate dependencies, which in turn sets ``*_install=True`` for their
dependencies, and so on. If you run the ``install`` script with
``concept_install=False``, nothing will be installed.

To install e.g. just FFTW, use

.. code-block:: bash

   concept_install=False fftw_install=True bash install

possibly adding in a specific version (``fftw_version=...``) and an MPI
library (``mpi_dir=...`` or ``mpi_install=True``) to link against. As FFTW
does not absolutely need MPI, ``mpi_install`` is not set by having
``fftw_install=True``.

To install GADGET-2, use

.. code-block:: bash

   concept_install=False gadget_install=True bash install

This time, MPI, GSL and FFTW (2) will be installed as well, as these are hard
dependencies.



.. _dependencies:

Dependencies
------------
This entry lists the dependency stack of CO\ *N*\ CEPT. Knowledge about this
stack is not necessary if installing CO\ *N*\ CEPT using the ``install``
script (which is *highly* recommended), but it is important if for some reason
you want to build (parts of) this stack yourself.

Typically, the exact version of any given dependency is not crucial. An effort
is made to ensure that CO\ *N*\ CEPT functions with recent stable versions of
each dependency. As many of the CO\ *N*\ CEPT dependencies also depend on
each other, finding a working set of dependency versions may be non-trivial.
You may draw inspiration from the current or older versions of the
`install script <https://github.com/jmd-dk/concept/blob/master/install>`__
(look for 'Specification of software versions').



Python dependencies
...................
The CO\ *N*\ CEPT source depends explicitly on
`Python <https://www.python.org/>`__ version 3.11 or newer, together with the
following Python packages (many of which have heavy dependencies of their
own):

* `Cython <https://cython.org/>`__: Needed for transpilation (*cythonization*)
  of the pure Python source code of CO\ *N*\ CEPT into equivalent C code.

* `NumPy <https://www.numpy.org/>`__: Provides the basic array types used for
  representing all primary data, together with various array/math functions.

* `SciPy <https://www.scipy.org/>`__: Provides various numerical methods and
  additional math functions.

* `Matplotlib <https://matplotlib.org/>`__: Provides plotting functionality for
  2D and 3D figures, as well as colour transformations used for coloured
  terminal output.

* `MPI4Py <https://mpi4py.readthedocs.io/>`__: Provides Python bindings for
  `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`__, used for
  all inter-process communication.

* `H5Py <https://www.h5py.org/>`__: Provides Python bindings for
  `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__, used for various binary
  input/output.

* `CythonGSL <https://github.com/twiecki/CythonGSL>`__: Provides Cython
  bindings for `GSL <https://www.gnu.org/software/gsl/>`__, used for more
  performant replacements of some NumPy/SciPy functionalities when running
  CO\ *N*\ CEPT in compiled mode.

* `Blessings <https://github.com/erikrose/blessings>`__: Provides terminal
  formatting.

In addition, the `Sphinx <http://www.sphinx-doc.org/>`__,
`sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/>`__,
`sphinx-rtd-theme <https://sphinx-rtd-theme.readthedocs.io/>`__ and
`sphinx-tabs <https://github.com/djungelorm/sphinx-tabs/>`__ Python packages
are needed to build the documentation, but may otherwise be left out.



Other primary dependencies
..........................
In addition to Python, the Python packages listed above and their respective
dependencies, CO\ *N*\ CEPT further depends explicitly on
`FFTW <http://www.fftw.org/>`__ 3 for its distributed FFT capabilities.

.. note::
   CO\ *N*\ CEPT does not make use of any standard Python bindings for FFTW
   --- such as `pyFFTW <https://github.com/pyFFTW/pyFFTW>`__ --- as these
   (at least traditionally) do not include the distributed (MPI)
   FFTs required. Instead, CO\ *N*\ CEPT provides its own minimal wrapper,
   ``fft.c``. This is the only C file in the CO\ *N*\ CEPT source code.

If building FFTW yourself, remember to link against an MPI library. The same
goes for building HDF5 and installing MPI4Py and H5Py. Also, the MPI library
has to conform to the MPI-3 (or MPI-3.1) standard.

For testing, CO\ *N*\ CEPT compares itself against
`GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/>`__, specifically
version 2.0.7. When installing using the ``install`` script, GADGET-2 is
patched in order to increase the maximum allowed size of path names and
slightly change the values of various internal physical constants as to
match the values adopted by CO\ *N*\ CEPT. If you do not care about running
the CO\ *N*\ CEPT test suite, you do not have to install GADGET-2 at all.

The last non-standard dependency of CO\ *N*\ CEPT is the
`CLASS <http://class-code.net/>`__ code, along with its Python wrapper
'classy'. When installing using the ``install`` script, CLASS and classy are
patched in order to enable larger and new kinds of output, fix bugs and
improve interoperability with CO\ *N*\ CEPT (which includes (slight) changes
to internal values of physical constants, to match the values adopted by
CO\ *N*\ CEPT). If installing without the use of the ``install`` script, you
will have to obtain the patched CLASS + classy by some other means. A good
solution is to install everything else first, including CO\ *N*\ CEPT itself,
and then make use of the CO\ *N*\ CEPT ``update`` utility to install and patch
CLASS + classy:

.. code-block:: bash

   ./concept -u update --class <version>

Here, ``<version>`` should be replaced with the CO\ *N*\ CEPT version whose
``install`` script should be used to install and patch CLASS + classy. Due to
the patching of CLASS + classy, only the specific CLASS version written in the
``install`` script is likely to function. For CO\ *N*\ CEPT version 1.0,
this is CLASS version 2.7.2.



.. _system_dependencies:

System dependencies
...................
In addition to the many dependencies above, CO\ *N*\ CEPT further uses a lot
of *system dependencies*, by which is meant programs that are usually
installed system-wide. These are
`Bash <https://www.gnu.org/software/bash/>`__ 3.0 or newer and
`GNU Core Utilities <https://www.gnu.org/software/coreutils/>`__. In addition,
the `GNU Find Utilities <https://www.gnu.org/software/findutils/>`__ are
needed for slimming down the installation, building the documentation and for
automated deployment. For just installing and running the code, the GNU Find
Utilities are not essential. All three of these system dependencies comes
pre-installed on the vast majority of Linux distributions. None of them will
be installed by the ``install`` script.

Other system dependencies needed for the core CO\ *N*\ CEPT functionality are
`AWK <https://en.wikipedia.org/wiki/AWK>`__,
`grep <https://www.gnu.org/software/grep/>`__
and `sed <https://www.gnu.org/software/sed/>`__. Also, the ``install`` script
and ``update`` utility further makes use of
`Gzip <https://www.gnu.org/software/gzip/>`__,
`Tar <https://www.gnu.org/software/tar/>`__
and `Wget <https://www.gnu.org/software/wget/>`__. That is, you may run
simulations without these last three components installed. If running the
``install`` script or ``update`` utility without these, you will be prompted
for system-wide (root) installation.

Lastly, CO\ *N*\ CEPT needs standard tools for compiling and linking C (C99)
code. An ``mpicc`` C compiler/linker should be bundled with the MPI library
used. The `GNU Make <https://www.gnu.org/software/make/>`__ utility is
also needed.



.. _the_path_and_env_files:

The ``.path`` and ``.env`` files
---------------------------------
The ``.path`` file and the ``.env`` file are special files storing static
information about the CO\ *N*\ CEPT installation. The ``.path`` file store
absolute paths to various files and directories, while the ``.env`` file store
environment variables as they should be set when running CO\ *N*\ CEPT.

Both of these files are generated by the ``install`` script
during installation.
Should you wish to not use the ``install`` script, you should grab ``.path``
and ``.env`` from the
`online repository <https://github.com/jmd-dk/concept>`__ and edit
them manually.



The ``.path`` file
..................
This is simply a Bash script of variable declarations, each variable storing
the absolute path to some file or directory. To install CO\ *N*\ CEPT without
the use of the ``install`` script, you must manually set these paths.

From the comment above each variable within the file, exactly what each of
them refers to should be apparent. An exception is perhaps the many ``mpi_*``
variables, which are explained below:

- ``mpi_dir`` is the root directory for the MPI library, typically
  containing the ``bin``, ``lib`` and ``include`` subdirectories.

- ``mpi_compilerdir`` is the directory that contains the MPI C compiler,
  ``mpicc``.

- ``mpi_bindir`` is the directory that contains the MPI executable
  ``mpiexec``/``mpirun``.

- ``mpi_libdir`` is the directory that contains MPI library files, e.g.
  ``libmpi.so``.

- ``mpi_includedir`` is the directory that contains MPI header files, e.g.
  ``mpi.h``.

- ``mpi_symlinkdir`` is an optional directory in which to put symbolic links
  to MPI library files not present (or present under non-standard names) in
  ``mpi_libdir``, but needed when linking MPI programs. You do not necessarily
  have to set this.

In CO\ *N*\ CEPT parameter files, all variables defined in the ``.path``
file are available through the ``path`` variable. To e.g. get the absolute
path to the ``output`` directory, you may use

.. code-block:: python3

   path.output_dir

in your parameter file.

You are free to define further paths (or even variables in general) in the
``.path`` file, in which case they two will be available in parameter files
via the ``path`` variable.



The ``.env`` file
.................
This file is meant to set up the required environment variables needed for
building and running CO\ *N*\ CEPT. It is
"`sourced <https://en.wikipedia.org/wiki/Dot_(command)#Source>`__" by the
``concept`` script before building and running the code.

.. tip::
   Should you want the environment of your interactive shell to be populated
   with the environment variables defined in ``.env``, it is recommended to
   "source" the ``concept`` script, rather than the ``.env`` file. This is
   becasue the ``concept`` scritp further sets up the environment in ways that
   are not meant to be user defined. You may need to do this e.g. before
   invoking ``make`` directly.

The ``.env`` file is populated with ``PATH``-like environment variables
present during installation, if using the ``install`` script. On a cluster,
you typically "source" scripts or load modules prior to the installation
itself in order to gain access to compilers and/or libraries. The intend is
for the ``.env`` file to define all necessary environment variables, so that
the same "sourcing" or module loading does not have to be repeated manually
before running CO\ *N*\ CEPT.

If you are installing CO\ *N*\ CEPT without the use of the ``install`` script
or some crucial part of the environment was not picked up during the
installation, you may add it yourself to the ``.env`` file, i.e. place

.. code-block:: bash

   export name="value"

somewhere in ``.env`` to make the variable ``name`` with value ``value`` be
part of the global CO\ *N*\ CEPT environment.



``PATH``-like environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Many environment variables (e.g. ``PATH``, ``LD_LIBRARY_PATH``) are
"``PATH``-like", meaning that their values are colon-separated substrings.
Such environment variables are potentially dangerous to overwrite, so
instead they are merely updated by either prepending or appending new
substrings to their present value, e.g.

.. code-block:: bash

   export PATH="/some/new/path:${PATH}"  # prepending
   export PATH="${PATH}:/some/new/path"  # appending

The above syntax is allowed in ``.env``. Equivalently, the
``pathenv_name_value_pairs_custom`` array variable may be used;

.. code-block:: bash

   pathenv_name_value_pairs_custom=(PATH "/some/new/path")

which either prepends or appends ``/some/new/path`` to ``PATH``, depending on
whether the ``concatenate`` variable in ``.env`` is set to ``prepend`` or
``append``.

When using the ``install`` script, any such ``PATH``-like environment
variables present during install time will be placed in a similar array
structure. Whether these are prepended or appended to the pre-existing values
of the same ``PATH``-like environment variables when building/running
CO\ *N*\ CEPT is similarly determined by ``concatenate``.



.. _eliminating_interference_from_foreign_Python_installations:

Eliminating interference from foreign Python installations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When using the ``install`` script and Python is installed as part of the
CO\ *N*\ CEPT installation (the default), the following lines are placed in
``.env`` (see the
`Python documentation <https://docs.python.org/3/using/cmdline.html#environment-variables>`__
for details):

.. code-block:: bash

   unset PYTHONPATH
   unset PYTHONHOME
   export PYTHONNOUSERSITE="True"

This is to eliminate any chance of interference from other Python
installations on the system. If you have installed CO\ *N*\ CEPT manually and
experience problems with Python (e.g. ``ImportError``), try adding the above
lines to the ``.env`` file yourself.



The ``mpi_executor``
~~~~~~~~~~~~~~~~~~~~
The ``mpi_executor`` variable determines which program is responsible for
launching CO\ *N*\ CEPT as an MPI program, when submitted as a job on a remote
cluster. You may leave this empty or undefined in which case a (hopefully)
suitable value will be determined by the ``concept`` script.

After submitting a remote CO\ *N*\ CEPT job, see the ``job/<ID>/jobscript``
for the chosen value of ``mpi_executor``.

See :ref:`this <chosing_an_mpi_executor>` troubleshooting entry for further
details.

.. note::

   If an environment variable ``mpi_executor`` is set and non-empty, this
   takes precedence over the ``mpi_executor`` set in the ``.env`` file.



The ``make_jobs`` environment variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``make_jobs`` variable may be set in the ``.env`` file in order to
specify whether CO\ *N*\ CEPT should be built in parallel. The default
behaviour is to build in parallel when working locally and build serially
when working remotely. To overrule this, add one of e.g.

.. code-block:: bash

   export make_jobs="-j 1"  # always build serially
   export make_jobs="-j 2"  # always build in parallel, using 2 cores
   export make_jobs="-j"    # always build in parallel, using any number of cores

to ``.env``.

.. note::

   If an environment variable ``make_jobs`` is set and non-empty, this
   takes precedence over the ``make_jobs`` set in the ``.env`` file.

