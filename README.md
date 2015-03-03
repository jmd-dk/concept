### *N*-body code
This is the readme file for the *N*-body code by Jeppe Dakin.
The code is written in Cython and C, but can also be run purely in Python,
without the need of cythonization, C compilation and linking.

#### Dependencies
The code should be able to compile and run on any POSIX system
with GCC and GNU make.
The code has the following non-standard but open source dependencies:
- The **GSL** library.
- An **MPI3** implementation (e.g. Open MPI).
- The **HDF5** library, linked to the installed MPI implementation and
  configured to be parallel.
- The **FFTW** library, version 3.3 or newer, linked to the installed MPI
  implementation and configured to be parallel.
- **(C)Python3** with the following site-packages:
  - **cython**.
  - **cython_gsl** linked to the GSL library.
  - **h5py** version 2.4 or newer, linked to the HDF5
    library and configured to be parallel.
  - **matplotlib**.
  - **mpi4py** version 1.3.1 or newer, linked to the
    installed MPI implementation.
  - **numpy**.
  - **pexpect**.
  - **scipy**.
- The **Gadget2** *N*-body code. This is only needed for running some
  tests and can be omitted. Gadget2 is also dependent on GSL, MPI and
  FFTW. Note that Gadget2 is incompatible with FFTW 3.x, so a seperate
  FFTW 2.x must also be installed.

#### Easy installation
The installation script `installer` automates the process of setting up
all dependencies by downloading and installing everything in one place.
The `installer` script will not take into account preinstalled versions
of any of the above libraries, though no conflicts will occur.

For even more automatization, simply execute this command to download
and run the `installer` script:

    bash <(wget -O- --no-ch tiny.cc/nbody) [installdir]

To keep track of all the components, `installer` writes a bunch of absolute
paths to a file called `.paths`. If you choose to manually install some or
all components, you must edit this file accordingly.

The `installer` script uses the Anaconda Python distribution for CPython3
with the site-packages and Open MPI for the MPI3 implementation.

The `installer` script will prompt for the installation directory when run,
or it can be given as an argument.

As everything is installed in one directory, uninstallation is done simply
by removing this directory.

#### Building and running the code
The code can be run in either compiled or pure Python mode.
For easy building and running of the code in both modes, the `run` script
should be used. Execute

    ./run --help

to learn how to use it.

