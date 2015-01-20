### *N*-body code
This is the readme file for the *N*-body code by Jeppe Dakin.
The code is written in Cython and C, but can also be run purely in Python,
without the need of cythonization, C compilation and linking.

#### Dependencies
The code has the following non-standard but open source dependencies:
- An **MPI3** implementation (e.g. Open MPI).
- The **HDF5** library, linked to the installed MPI implementation and
  configured parallel.
- The **FFTW** library, version 3.3 or newer, linked to the installed MPI
  implementation and configured parallel.
- **(C)Python3** with the following site-packages:
  - **cython**.
  - **numpy**.
  - **scipy**.
  - **matplotlib**.
  - **mpi4py** version 1.3.1 or newer, linked to the
    installed MPI implementation.
  - **h5py** version 2.4 or newer, linked to the HDF5
    library and configured parallel.

#### Easy installation
The installation script `installer` automates the process of setting up
all dependencies by downloading and installing everything in one place.
The `installer` script will not take into account preinstalled versions
of any of the above libraries, though no conflicts will happen.

For even more automatization, simply execute this command to download
and run the `install` script:

    <(wget -O- --no-ch tiny.cc/nbody) [installdir]

By using `installer` to install the code you also avoid having to fiddle
with the `makefile` and `run` script, as the paths to the various files
will be correct as is.

The `installer` script uses the Anaconda Python distribution for CPython3
and the site-packages and Open MPI for the MPI3 implementation.

The `installer` script will prompt for the installation directory when run,
or it can be given as an argument.

As everything is installed in one directory, uninstallation is done simply
by removing this directory.
##### Dependency licenses
By using `installer` it is expected that the license terms for the use of
Anaconda, Open MPI, FFTW and HDF5 are agreed upon. Copies of these licenses
are included with the installation.

#### Building and running the code
The code can be run in either compiled or pure Python mode.
For easy building and running of the code in both modes, the `run` script
should be used. Execute

    ./run --help

to learn how to use it.

