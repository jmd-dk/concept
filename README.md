### *N*-body code
This is the readme file for the *N*-body code by Jeppe Dakin.
The code is written in Cython and C, but can also be run in pure Python.

##### Dependencies
The code has the following non-standard dependencies:
- An **MPI3** implementation (e.g. Open MPI).
- The **HDF5** library, linked to the installed MPI implementation and configured parallel.
- The **FFTW** library, version 3.x, linked to the installed MPI implementation and configured parallel.
- **(C)Python3** with the following site-packages:
  - **cython**.
  - **numpy**.
  - **scipy**.
  - **matplotlib**.
  - **mpi4py** version 1.3.1 or newer, linked to the installed MPI implementation.
  - **h5py** version 2.4 or newer, linked to the HDF5 library and configured parallel.
The installation script
