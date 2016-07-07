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



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
from communication import smart_mpi
cimport('from communication import communicate_domain_boundaries,                    '
                                  'communicate_domain_ghosts,                        '
                                  'rank_neighboring_domain,                          '
                                  'domain_layout_local_indices,                      '
                                  'domain_subdivisions,                              '
                                  'domain_size_x,  domain_size_y,  domain_size_z,    '
                                  'domain_start_x, domain_start_y, domain_start_z,   '
        )



# Seperate but roughly equivalent imports in pure Python and Cython
if not cython.compiled:
    # Emulate FFTW's fftw_execute in pure Python
    def fftw_execute(plan):
        # The pure Python FFT implementation is serial.
        # Every process computes the entire FFT of the temporary
        # varaible φ_global_pure_python.
        φ_global_pure_python = empty((φ_gridsize, φ_gridsize, slab_size_padding))
        Allgatherv(slab, φ_global_pure_python)
        if plan == plan_forward:
            # Delete the padding on last dimension
            for i in range(slab_size_padding - φ_gridsize):
                φ_global_pure_python = delete(φ_global_pure_python, -1, axis=2)
            # Do real transform via NumPy
            φ_global_pure_python = np.fft.rfftn(φ_global_pure_python)
            # FFTW transposes the first two dimensions
            φ_global_pure_python = φ_global_pure_python.transpose([1, 0, 2])
            # FFTW represents the complex array by doubles only
            tmp = empty((φ_gridsize, φ_gridsize, slab_size_padding))
            for i in range(slab_size_padding):
                if i % 2:
                    tmp[:, :, i] = φ_global_pure_python.imag[:, :, i//2]
                else:
                    tmp[:, :, i] = φ_global_pure_python.real[:, :, i//2]
            φ_global_pure_python = tmp
            # As in FFTW, distribute the slabs along the y-dimension
            # (which is the first dimension now, due to transposing).
            slab[...] = φ_global_pure_python[slab_start_j:(slab_start_j + slab_size_j), :, :]
        elif plan == plan_backward:
            # FFTW represents the complex array by doubles only.
            # Go back to using complex entries.
            tmp = zeros((φ_gridsize, φ_gridsize, slab_size_padding/2),
                        dtype='complex128')
            for i in range(slab_size_padding):
                if i % 2:
                    tmp[:, :, i//2] += 1j*φ_global_pure_python[:, :, i]
                else:
                    tmp[:, :, i//2] += φ_global_pure_python[:, :, i]
            φ_global_pure_python = tmp
            # FFTW transposes the first
            # two dimensions back to normal.
            φ_global_pure_python = φ_global_pure_python.transpose([1, 0, 2])
            # Do real inverse transform via NumPy
            φ_global_pure_python = np.fft.irfftn(φ_global_pure_python, s=[φ_gridsize]*3)
            # Remove the autoscaling provided by NumPy
            φ_global_pure_python *= φ_gridsize3
            # Add padding on last dimension, as in FFTW
            padding = empty((φ_gridsize,
                             φ_gridsize,
                             slab_size_padding - φ_gridsize,
                             ))
            φ_global_pure_python = concatenate((φ_global_pure_python, padding), axis=2)
            # As in FFTW, distribute the slabs along the x-dimension
            slab[...] = φ_global_pure_python[slab_start_i:(slab_start_i + slab_size_i), :, :]

else:
    # Lines in triple quotes will be executed in the .pyx file
    """
    # FFT functionality via FFTW from fft.c
    cdef extern from "fft.c":
        # The fftw_plan type
        ctypedef struct fftw_plan_struct:
            pass
        ctypedef fftw_plan_struct *fftw_plan
        # The returned struct of fftw_setup
        struct fftw_return_struct:
            ptrdiff_t gridsize_local_i
            ptrdiff_t gridsize_local_j
            ptrdiff_t gridstart_local_i
            ptrdiff_t gridstart_local_j
            double* grid
            fftw_plan plan_forward
            fftw_plan plan_backward
        # Functions
        fftw_return_struct fftw_setup(ptrdiff_t gridsize_i,
                                      ptrdiff_t gridsize_j,
                                      ptrdiff_t gridsize_k,
                                      char*     rigor)
        void fftw_execute(fftw_plan plan)
        void fftw_clean(double* grid, fftw_plan plan_forward,
                                      fftw_plan plan_backward)
    """



# Function for easy partitioning of multidimensional arrays
@cython.header(# Arguments
               size='Py_ssize_t',
               size_point='Py_ssize_t',
               # Locals
               i='Py_ssize_t',
               rank_upper='int',
               size_local_lower='Py_ssize_t',
               sizes_local='Py_ssize_t[::1]',
               start_local='Py_ssize_t',
               start_local_extra='int',
               returns='Py_ssize_t[::1]',
               )
def partition(size, size_point=1):
    """This function takes in the size (nr. of elements) of an array as
    and returns an array of local sizes, corresponding to a linear, fair
    partition of the array. If the array cannot be partitioned
    completely fair, the higher ranks will recieve one additional
    element. The starting index for a given rank can of course be
    computed using the array of local sizes. For the sake of efficiency
    though, it is stored as the 'nprocs' element of the returned array.
    In each gridpoint can be placed a scalar, a vector etc. This changes
    the sizes and the start index. To indicate the nr. of numbers in
    each gridpoint, set the optional 'size_point' argument.
    """
    # The size of the local part of the array, for lower ranks
    size_local_lower = size//nprocs
    # The lowest rank of the higher processes,
    # which recieve one additional element.
    rank_upper = nprocs*(size_local_lower + 1) - size
    # The sizes of the local parts of the array
    sizes_local = size_local_lower*ones(nprocs + 1, dtype=C2np['Py_ssize_t'])
    for i in range(rank_upper, nprocs):
        sizes_local[i] += 1
    # The local start index in the global array
    start_local = size_local_lower*rank
    start_local_extra = rank - rank_upper
    if start_local_extra > 0:
        start_local += start_local_extra
    # Pack the start index into the sizes_local array
    sizes_local[nprocs] = start_local
    # Correct for non-scalar elements
    if size_point > 1:
        for i in range(nprocs + 1):
            sizes_local[i] *= size_point
    return sizes_local

# Function for initializing and tabulating a cubic grid with
# vector values of a given dimension.
@cython.header(# Arguments
               gridsize='Py_ssize_t',
               N_dim='Py_ssize_t',
               func='func_dstar_ddd',
               factor='double',
               filename='str',
               # Locals
               dim='Py_ssize_t',
               grid='double[:, :, :, ::1]',
               grid_local='double[::1]',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               m='Py_ssize_t',
               n_vectors='Py_ssize_t',
               n_vectors_local='Py_ssize_t',
               shape='Py_ssize_t[::1]',
               size='Py_ssize_t',
               size_edge='Py_ssize_t',
               size_face='Py_ssize_t',
               size_local='Py_ssize_t',
               size_point='Py_ssize_t',
               sizes_local='Py_ssize_t[::1]',
               start_local='Py_ssize_t',
               vector_value='double*',
               ℓ_local='Py_ssize_t',
               ℓ='Py_ssize_t',
               returns='double[:, :, :, ::1]',
               )
def tabulate_vectorfield(gridsize, N_dim, func, factor, filename=''):
    """ This function tabulates a cubic grid of size
    gridsize*gridsize*gridsize with vector values of the given
    dimension N_dim computed by the function func,
    as grid[i, j, k] = func(i*factor, 
                            j*factor,
                            k*factor).
    If filename is set, the tabulated grid is saved to a hdf5 file
    with this name.
    The first point grid[0, 0, 0] corresponds to the origin of the box,
    while the last point grid[gridsize - 1,
                              gridsize - 1,
                              gridsize - 1]
    corresponds to the physical point ((gridsize - 1)/gridsize*boxsize,
                                       (gridsize - 1)/gridsize*boxsize,
                                       (gridsize - 1)/gridsize*boxsize).
    This means that grid[gridsize, gridsize, gridsize] is the corner
    of the box the longest away from the origin. This point is not
    represented in the grid because of the periodicity of the box,
    which means that physically, this point is the same as the origin.
    The mapping from grid indices to physical coordinates is thus
    (i, j, z) --> (i/gridsize*boxsize,
                   j/gridsize*boxsize,
                   k/gridsize*boxsize)
    Therefore, if you wish to tabulate a global field (extending over
    the entire box), the factor argument should be
    proportional to boxsize/gridsize. If you wish to only tabulate say
    one octant of the box, this volume is no more periodic, and so
    the factor should be proportional to 0.5*boxsize/(gridsize - 1).
    """
    # The grid has a shape of gridsize*gridsize*gridsize*dimensions.
    # That is, grid is not really cubic, but rather four-dimensional.
    shape = np.array([gridsize]*3 + [N_dim], dtype=C2np['Py_ssize_t'])
    # The number of elements in each point, edge, face and
    # in the total, when the grid is viewed as a 3D box of vectors.
    size_point = shape[3]
    size_edge = shape[2]*size_point
    size_face = shape[1]*size_edge
    size = shape[0]*size_face
    n_vectors = size//size_point
    # Initialize the grid to be of shape gridsize*gridsize*gridsize*3.
    # That is, grid is not really cubic, but rather four-dimensional.
    grid = empty(shape, dtype=C2np['double'])
    # Partition the grid fairly among the processes. Each part is
    # defined by a linear size and starting index.
    sizes_local = asarray(partition(n_vectors, size_point),
                          dtype=C2np['Py_ssize_t'])
    size_local = sizes_local[rank]
    start_local = sizes_local[nprocs]
    n_vectors_local = size_local//size_point
    # Make a linear array for storing the local part of the grid
    grid_local = empty(size_local, dtype=C2np['double'])
    # Tabulate the local grid
    for m in range(n_vectors_local):
        # The local and global linear index
        ℓ_local = m*size_point
        ℓ = start_local + ℓ_local
        # The global 3D indices
        i = ℓ//size_face
        ℓ -= i*size_face
        j = ℓ//size_edge
        ℓ -= j*size_edge
        k = ℓ//size_point
        # Fill grid_local
        vector_value = func(i*factor, j*factor, k*factor)
        for dim in range(N_dim):
            grid_local[ℓ_local + dim] = vector_value[dim]
    # Gather the local grid parts into a common, global grid
    Allgatherv(sendbuf=grid_local, 
               recvbuf=(grid, sizes_local[:nprocs]))
    # Return now if the grid should not be saved to disk
    if not filename:
        return grid
    # Save grid to disk using parallel HDF5
    with h5py.File(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
        dset = hdf5_file.create_dataset('data', (size, ), dtype=C2np['double'])
        dset[start_local:(start_local + size_local)] = grid_local
    return grid

# Function for doing lookup in a grid with scalar values and
# CIC-interpolating to specified coordinates.
@cython.header(# Argument
               grid='double[:, :, :]',
               x='double',
               y='double',
               z='double',
               # Locals
               Wxl='double',
               Wxu='double',
               Wyl='double',
               Wyu='double',
               Wzl='double',
               Wzu='double',
               x_lower='Py_ssize_t',
               x_upper='Py_ssize_t',
               y_lower='Py_ssize_t',
               y_upper='Py_ssize_t',
               z_lower='Py_ssize_t',
               z_upper='Py_ssize_t',
               returns='double',
               )
def CIC_scalargrid2coordinates(grid, x, y, z):
    """This function looks up tabulated scalars in a grid and
    interpolates to (x, y, z) via the cloud in cell (CIC) method. Input
    arguments must be normalized so that 0 <= x, y, z < 1. If x, y or z
    is exactly equal to 1, they will be corrected to 1 - ϵ. It is
    assumed that the grid is nonperiodic (that is,
    the grid has closed ends).
    """
    # Correct for extreme values in the parsed coordinates.
    # This is to catch inputs which are slighly larger than 1 due to
    # numerical errors.
    if x >= 1:
        x = ℝ[1 - machine_ϵ]
    if y >= 1:
        y = ℝ[1 - machine_ϵ]
    if z >= 1:
        z = ℝ[1 - machine_ϵ]
    # Scale the coordinates so that 0 <= x, y, z < (gridsize - 1)
    x *= grid.shape[0] - 1
    y *= grid.shape[1] - 1
    z *= grid.shape[2] - 1
    # Indices of the 8 vertices (6 faces)
    # of the grid surrounding (x, y, z).
    x_lower = int(x)
    y_lower = int(y)
    z_lower = int(z)
    x_upper = x_lower + 1
    y_upper = y_lower + 1
    z_upper = z_lower + 1
    # The linear weights according to the
    # CIC rule W = 1 - |dist| if |dist| < 1.
    Wxl = x_upper - x  # = 1 - (x - x_lower)
    Wyl = y_upper - y  # = 1 - (y - y_lower)
    Wzl = z_upper - z  # = 1 - (z - z_lower)
    Wxu = x - x_lower  # = 1 - (x_upper - x)
    Wyu = y - y_lower  # = 1 - (y_upper - y)
    Wzu = z - z_lower  # = 1 - (z_upper - z)
    # Return the sum of the weighted grid values
    return (  grid[x_lower, y_lower, z_lower]*ℝ[Wxl*Wyl]*Wzl
            + grid[x_lower, y_lower, z_upper]*ℝ[Wxl*Wyl]*Wzu
            + grid[x_lower, y_upper, z_lower]*ℝ[Wxl*Wyu]*Wzl
            + grid[x_lower, y_upper, z_upper]*ℝ[Wxl*Wyu]*Wzu
            + grid[x_upper, y_lower, z_lower]*ℝ[Wxu*Wyl]*Wzl
            + grid[x_upper, y_lower, z_upper]*ℝ[Wxu*Wyl]*Wzu
            + grid[x_upper, y_upper, z_lower]*ℝ[Wxu*Wyu]*Wzl
            + grid[x_upper, y_upper, z_upper]*ℝ[Wxu*Wyu]*Wzu)

# Function for doing lookup in a grid with vector values and
# CIC-interpolating to specified coordinates
@cython.header(# Argument
               grid='double[:, :, :, :]',
               x='double',
               y='double',
               z='double',
               # Locals
               Wxl='double',
               Wxu='double',
               Wyl='double',
               Wyu='double',
               Wzl='double',
               Wzu='double',
               dim='int',
               x_lower='Py_ssize_t',
               x_upper='Py_ssize_t',
               y_lower='Py_ssize_t',
               y_upper='Py_ssize_t',
               z_lower='Py_ssize_t',
               z_upper='Py_ssize_t',
               returns='double*',
               )
def CIC_vectorgrid2coordinates(grid, x, y, z):
    """This function looks up tabulated vectors in a grid and
    interpolates to (x, y, z) via the cloud in cell (CIC) method.
    Input arguments must be normalized so that 0 <= x, y, z < 1.
    If x, y or z is exactly equal to 1, they will be corrected to 1 - ϵ.
    It is assumed that the grid is nonperiodic (that is, the first and
    the last gridpoint in any dimension are physical distinct and that
    the grid has closed ends).
    """
    # Correct for extreme values in the parsed coordinates.
    # This is to catch inputs which are slighly larger than 1 due to
    # numerical errors.
    if x >= 1:
        x = ℝ[1 - machine_ϵ]
    if y >= 1:
        y = ℝ[1 - machine_ϵ]
    if z >= 1:
        z = ℝ[1 - machine_ϵ]
    # Scale the coordinates so that 0 <= x, y, z < (gridsize - 1)
    x *= grid.shape[0] - 1
    y *= grid.shape[1] - 1
    z *= grid.shape[2] - 1
    # Indices of the 8 vertices (6 faces)
    # of the grid surrounding (x, y, z).
    x_lower = int(x)
    y_lower = int(y)
    z_lower = int(z)
    x_upper = x_lower + 1
    y_upper = y_lower + 1
    z_upper = z_lower + 1
    # The linear weights according to the
    # CIC rule W = 1 - |dist| if |dist| < 1.
    Wxl = x_upper - x  # = 1 - (x - x_lower)
    Wyl = y_upper - y  # = 1 - (y - y_lower)
    Wzl = z_upper - z  # = 1 - (z - z_lower)
    Wxu = x - x_lower  # = 1 - (x_upper - x)
    Wyu = y - y_lower  # = 1 - (y_upper - y)
    Wzu = z - z_lower  # = 1 - (z_upper - z)
    # Assign the weighted grid values to the vector components
    for dim in range(3):
        vector[dim] = (  grid[x_lower, y_lower, z_lower, dim]*ℝ[Wxl*Wyl]*Wzl
                       + grid[x_lower, y_lower, z_upper, dim]*ℝ[Wxl*Wyl]*Wzu
                       + grid[x_lower, y_upper, z_lower, dim]*ℝ[Wxl*Wyu]*Wzl
                       + grid[x_lower, y_upper, z_upper, dim]*ℝ[Wxl*Wyu]*Wzu
                       + grid[x_upper, y_lower, z_lower, dim]*ℝ[Wxu*Wyl]*Wzl
                       + grid[x_upper, y_lower, z_upper, dim]*ℝ[Wxu*Wyl]*Wzu
                       + grid[x_upper, y_upper, z_lower, dim]*ℝ[Wxu*Wyu]*Wzl
                       + grid[x_upper, y_upper, z_upper, dim]*ℝ[Wxu*Wyu]*Wzu)
    return vector

# Function which interpolates one grid onto another grid,
# optionally multiplying the interpolated values by a factor.
@cython.header(# Arguments
               gridA='double[:, :, :]',
               gridB='double[:, :, :]',
               fac='double',
               # Locals
               Wil='double',
               Wjl='double',
               Wkl='double',
               Wiu='double',
               Wju='double',
               Wku='double',
               dim='int',
               i='Py_ssize_t',
               iA='double',
               iA_lower='Py_ssize_t',
               iA_upper='Py_ssize_t',
               iB='Py_ssize_t',
               j='Py_ssize_t',
               jA='double',
               jA_lower='Py_ssize_t',
               jA_upper='Py_ssize_t',
               jB='Py_ssize_t',
               k='Py_ssize_t',
               kA='double',
               kA_lower='Py_ssize_t',
               kA_upper='Py_ssize_t',
               kB='Py_ssize_t',
               scaling_i='double',
               scaling_j='double',
               scaling_k='double',
               shapeA='tuple',
               shapeB='tuple',
               value='double',
               )
def CIC_grid2grid(gridA, gridB, fac=1):
    """This function CIC-interpolates values from one grid (gridB) onto
    another grid (gridA). The physical extend of the parsed grids are
    assumed to be the same. It is assumed that both grids are closed,
    meaning that the upper grid points (for all three directions)
    recide on the physical boundary of the region in which the grid
    is placed. For domain grids, this corresponds to the inclusion
    of pseudo points (but not ghost points) in the grids.
    The interpolated values will be added to gridA. Therefore, if the
    grid should contain the interpolated vales only, the grid must be
    nullified beforehand.
    Before adding the interpolated values to gridA, they are multiplied
    by fac.
    """
    # If the two grids have the same shape, each grid point in gridA is
    # simply updated based on the equivalent grid point in gridB
    if (    gridA.shape[0] == gridB.shape[0]
        and gridA.shape[1] == gridB.shape[1]
        and gridA.shape[2] == gridB.shape[2]):
        # The two grids have equal shapes
        for i in range(gridA.shape[0]):
            for j in range(gridA.shape[1]):
                for k in range(gridA.shape[2]):
                    gridA[i, j, k] += fac*gridB[i, j, k]
        return
    # The two grids have different shapes. Perform CIC-interpolation.
    # Extract the shape of the grids (without the end points).
    shapeA = tuple([gridA.shape[dim] - 1 for dim in range(3)])
    shapeB = tuple([gridB.shape[dim] - 1 for dim in range(3)])
    # Factors which scales grid indices in gridB
    # to (floating point) grid indices in gridA.
    scaling_i = shapeA[0]/shapeB[0]
    scaling_j = shapeA[1]/shapeB[1]
    scaling_k = shapeA[2]/shapeB[2]
    for iB in range(gridB.shape[0]):
        # The i-indices in gridA around the iB-index in gridB
        iA = iB*scaling_i
        if iA >= ℝ[shapeA[0]]:
            # The lower index must not be a boundary index
            iA = ℝ[shapeA[0]*(1 - machine_ϵ)]
        iA_lower = int(iA)
        iA_upper = iA_lower + 1
        for jB in range(gridB.shape[1]):
            # The j-indices in gridA around the jB-index in gridB
            jA = jB*scaling_j
            if jA >= ℝ[shapeA[1]]:
                # The lower index must not be a boundary index
                jA = ℝ[shapeA[1]*(1 - machine_ϵ)]
            jA_lower = int(jA)
            jA_upper = jA_lower + 1
            for kB in range(gridB.shape[2]):
                # The k-indices in gridA around the kB-index in gridB
                kA = kB*scaling_k
                if kA >= ℝ[shapeA[2]]:
                    # The lower index must not be a boundary index
                    kA = ℝ[shapeA[2]*(1 - machine_ϵ)]
                kA_lower = int(kA)
                kA_upper = kA_lower + 1
                # The value which should be interpolated
                value = fac*gridB[iB, jB, kB]
                # The linear weights according to the
                # CIC rule W = 1 - |dist| if |dist| < 1.
                Wil = iA_upper - iA  # = 1 - (iA - iA_lower)
                Wjl = jA_upper - jA  # = 1 - (jA - jA_lower)
                Wkl = kA_upper - kA  # = 1 - (kA - kA_lower)
                Wiu = iA - iA_lower  # = 1 - (iA_upper - iA)
                Wju = jA - jA_lower  # = 1 - (jA_upper - jA)
                Wku = kA - kA_lower  # = 1 - (kA_upper - kA)
                # Assign the weights to the grid points
                gridA[iA_lower, jA_lower, kA_lower] += ℝ[value*Wil*Wjl]*Wkl
                gridA[iA_lower, jA_lower, kA_upper] += ℝ[value*Wil*Wjl]*Wku
                gridA[iA_lower, jA_upper, kA_lower] += ℝ[value*Wil*Wju]*Wkl
                gridA[iA_lower, jA_upper, kA_upper] += ℝ[value*Wil*Wju]*Wku
                gridA[iA_upper, jA_lower, kA_lower] += ℝ[value*Wiu*Wjl]*Wkl
                gridA[iA_upper, jA_lower, kA_upper] += ℝ[value*Wiu*Wjl]*Wku
                gridA[iA_upper, jA_upper, kA_lower] += ℝ[value*Wiu*Wju]*Wkl
                gridA[iA_upper, jA_upper, kA_upper] += ℝ[value*Wiu*Wju]*Wku

# Function for CIC-interpolating particle/fluid element coordinates of
# all components to a doamin grid.
@cython.header(# Argument
               components='list',
               domain_grid='double[:, :, ::1]',
               # Locals
               component='Component',
               domain_grid_noghosts='double[:, :, :]',
               posx='double*',
               posy='double*',
               posz='double*',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               mass='double',
               shape='tuple',
               size='Py_ssize_t',
               x='double',
               y='double',
               z='double',
               x_lower='int',
               y_lower='int',
               z_lower='int',
               x_upper='int',
               y_upper='int',
               z_upper='int',
               Wxl='double',
               Wyl='double',
               Wzl='double',
               Wxu='double',
               Wyu='double',
               Wzu='double',
               δ='double*',
               δ_mv='double[:, :, ::1]',
               δ_noghosts='double[:, :, :]',
               fluidscalar='FluidScalar',
               )
def CIC_components2domain_grid(components, domain_grid):
    """This function CIC-interpolates particle/fluid element coordinates
    to domain_grid storing scalar values. The physical extend of the
    parsed domain_grid should match the domain exactly. The interpolated
    values will be added to the grid. Therefore, if the grid should
    contain the interpolated vales only, the grid must be nullified 
    beforehand.
    Each interpolation will be multiplied by the particle/fluid element
    mass, so that it is the total mass of a component (and not its
    resolution (particle number/gridsize)) that governs the overall
    weight for that component.
    """
    # Memoryview of the domain grid without the ghost layers
    domain_grid_noghosts = domain_grid[2:(domain_grid.shape[0] - 2),
                                       2:(domain_grid.shape[1] - 2),
                                       2:(domain_grid.shape[2] - 2)]
    for component in components:
        # Do the interpolation
        if component.representation == 'particles':
            # Extract variables
            mass = component.mass
            posx = component.posx
            posy = component.posy
            posz = component.posz
            # Extract the shape of the grid
            shape = tuple([domain_grid_noghosts.shape[dim] - 1 for dim in range(3)])
            # Interpolate each particle
            for i in range(component.N_local):
                # Get, translate and scale the coordinates so that
                # 0 <= i < shape[i] - 1 for i in (x, y, z).
                x = (posx[i] - domain_start_x)*ℝ[shape[0]/domain_size_x]
                y = (posy[i] - domain_start_y)*ℝ[shape[1]/domain_size_y]
                z = (posz[i] - domain_start_z)*ℝ[shape[2]/domain_size_z]
                # Correct for coordinates which are
                # exactly at an upper domain boundary.
                if x >= ℝ[shape[0]]:
                    x = ℝ[shape[0]*(1 - machine_ϵ)]
                if y >= ℝ[shape[1]]:
                    y = ℝ[shape[1]*(1 - machine_ϵ)]
                if z >= ℝ[shape[2]]:
                    z = ℝ[shape[2]*(1 - machine_ϵ)]
                # Indices of the 8 vertices (6 faces)
                # of the grid surrounding (x, y, z).
                x_lower = int(x)
                y_lower = int(y)
                z_lower = int(z)
                x_upper = x_lower + 1
                y_upper = y_lower + 1
                z_upper = z_lower + 1
                # The linear weights according to the
                # CIC rule W = 1 - |dist| if |dist| < 1.
                Wxl = x_upper - x  # = 1 - (x - x_lower)
                Wyl = y_upper - y  # = 1 - (y - y_lower)
                Wzl = z_upper - z  # = 1 - (z - z_lower)
                Wxu = x - x_lower  # = 1 - (x_upper - x)
                Wyu = y - y_lower  # = 1 - (y_upper - y)
                Wzu = z - z_lower  # = 1 - (z_upper - z)
                # Assign the weights to the grid points
                domain_grid_noghosts[x_lower, y_lower, z_lower] += ℝ[mass*Wxl*Wyl]*Wzl
                domain_grid_noghosts[x_lower, y_lower, z_upper] += ℝ[mass*Wxl*Wyl]*Wzu
                domain_grid_noghosts[x_lower, y_upper, z_lower] += ℝ[mass*Wxl*Wyu]*Wzl
                domain_grid_noghosts[x_lower, y_upper, z_upper] += ℝ[mass*Wxl*Wyu]*Wzu
                domain_grid_noghosts[x_upper, y_lower, z_lower] += ℝ[mass*Wxu*Wyl]*Wzl
                domain_grid_noghosts[x_upper, y_lower, z_upper] += ℝ[mass*Wxu*Wyl]*Wzu
                domain_grid_noghosts[x_upper, y_upper, z_lower] += ℝ[mass*Wxu*Wyu]*Wzl
                domain_grid_noghosts[x_upper, y_upper, z_upper] += ℝ[mass*Wxu*Wyu]*Wzu
            # Values of local pseudo mesh points contribute to the lower
            # mesh points of domain grid on other processes.
            # Do the needed communication.
            communicate_domain_boundaries(domain_grid, mode=0)
        elif component.representation == 'fluid':
            # Extract δ grid variables
            fluidscalar = component.fluidvars['δ']
            δ          = fluidscalar.grid
            δ_mv       = fluidscalar.grid_mv
            δ_noghosts = fluidscalar.grid_noghosts
            # Add 1 to the δ grid, transforming it to ρ/ρbar.
            # Each grid value will then be proportional to the amount of
            # mass in that grid point, and the sum of all grid values
            # will equal the number of grid points.
            # Here we also loop over pseudo and ghost points,
            # but this is of no importance.
            size = δ_mv.shape[0]*δ_mv.shape[1]*δ_mv.shape[2]
            for i in range(size):
                δ[i] += 1
            # The correct values of the local pseudo points reside as
            # lower grid points on other processes.
            # Do the needed communication.
            communicate_domain_boundaries(δ_mv, mode=1)
            # CIC-interpolate the manipulated δ to the parsed domain
            # grid, using the average fluid element mass as an
            # additional factor in the weights. This has the effect of
            # interpolating the total mass of the component onto the
            # domain grid, accoring to a distribution determined by δ.
            CIC_grid2grid(domain_grid_noghosts, δ_noghosts, component.mass)
            # Now transform δ back to its original form, δ = ρ/ρbar - 1
            for i in range(size):
                δ[i] -= 1    

# Function for CIC-interpolating particle coordinates
# to a cubic grid holding scalar values.
@cython.header(# Argument
               component='Component',
               # Locals
               posx='double*',
               posy='double*',
               posz='double*',
               i='Py_ssize_t',
               x='double',
               y='double',
               z='double',
               x_lower='int',
               y_lower='int',
               z_lower='int',
               x_upper='int',
               y_upper='int',
               z_upper='int',
               Wxl='double',
               Wyl='double',
               Wzl='double',
               Wxu='double',
               Wyu='double',
               Wzu='double',
               var='str',
               δ_noghosts='double[:, :, :]',
               δ='double*',
               ux='double*',
               uy='double*',
               uz='double*',
               ux_noghosts='double[:, :, :]',
               uy_noghosts='double[:, :, :]',
               uz_noghosts='double[:, :, :]',
               momx='double*',
               momy='double*',
               momz='double*',
               momx_i='double',
               momy_i='double',
               momz_i='double',
               δ_fac='double',
               u_fac='double',
               δ_mv='double[:, :, ::1]',
               ux_mv='double[:, :, ::1]',
               uy_mv='double[:, :, ::1]',
               uz_mv='double[:, :, ::1]',
               original_representation='str',
               dim='int',
               shape='tuple',
               size='Py_ssize_t',
               fluidscalar='FluidScalar',
               )
def CIC_particles2fluid(component):
    """This function CIC-interpolates particle positions to fluid grids.
    The parsed component should contain particle data, but not
    necessarily fluid data (any pre-existing fluid data will be
    overwritten). The particle data are then used to create the fluid
    grids (δ, ux, uy, uz). In addition, since the relation between
    momentum (particle data) p and velocity (fluid data) u is
    p = u*m*a,
    where a is the scale factor (m is the particle mass), the current
    value of universals.a is used in order to do the transformation.
    The size of the fluid grids are determined by component.gridsize.
    Note that only the truly local part of the fluid grids will contain
    the correct data after calling this function (both pseudo points
    and ghost layers will contain junk).
    To save memory, the particle data will be freed (resized to a
    minimum size) during the process.
    The behaviour of this function is independent of
    component.representation.
    """
    # Backup of original representation
    original_representation = component.representation
    # Instantiate fluid grids spanning the local domains
    component.representation = 'fluid'
    shape = tuple([component.gridsize//ds for ds in domain_subdivisions])
    if master and any([component.gridsize != domain_subdivisions[dim]*shape[dim]
                       for dim in range(3)]):
            abort('The gridsize of the {} component is {}\n'
                  'which cannot be equally shared among {} processes'
                  .format(component.name, component.gridsize, nprocs))
    component.resize(shape)
    # Extract fluid data variables
    fluidscalar = component.fluidvars['δ']
    δ           = fluidscalar.grid
    δ_mv        = fluidscalar.grid_mv
    δ_noghosts  = fluidscalar.grid_noghosts
    fluidscalar = component.fluidvars['ux']
    ux          = fluidscalar.grid
    ux_mv       = fluidscalar.grid_mv
    ux_noghosts = fluidscalar.grid_noghosts
    fluidscalar = component.fluidvars['uy']
    uy          = fluidscalar.grid
    uy_mv       = fluidscalar.grid_mv
    uy_noghosts = fluidscalar.grid_noghosts
    fluidscalar = component.fluidvars['uz']
    uz          = fluidscalar.grid
    uz_mv       = fluidscalar.grid_mv
    uz_noghosts = fluidscalar.grid_noghosts
    # The number of elements in the fluid grids, including pseudo and
    # and ghost points. Even though the ghost points are never touched
    # by this function, it is easier to include them when performing
    # operations on an entire grid, as otherwise the memory layout is
    # non-contiguous. That is, more elements are operated on that
    # needed, but the logic required to skip these elements is
    # itself skipped.
    size = δ_mv.shape[0]*δ_mv.shape[1]*δ_mv.shape[2]
    # Nullify fluid grids
    for i in range(size):
        δ[i] = 0
        ux[i] = 0
        uy[i] = 0
        uz[i] = 0
    # Extract particle data variables
    posx = component.posx
    posy = component.posy
    posz = component.posz
    momx = component.momx
    momy = component.momy
    momz = component.momz
    # Interpolate each particle to the grids.
    # In the loop, simply assign the raw weight to the δ grid and the
    # unweighted momenta to the ux, uy and uz grids. Convertions to the
    # correct units will be done afterwards.
    for i in range(component.N_local):
        # Get, translate and scale the coordinates so that
        # 0 <= i < shape[i] - 1 for i in (x, y, z).
        x = (posx[i] - domain_start_x)*ℝ[shape[0]/domain_size_x]
        y = (posy[i] - domain_start_y)*ℝ[shape[1]/domain_size_y]
        z = (posz[i] - domain_start_z)*ℝ[shape[2]/domain_size_z]
        # Correct for coordinates which are
        # exactly at an upper domain boundary.
        if x >= ℝ[shape[0]]:
            x = ℝ[shape[0]*(1 - machine_ϵ)]
        if y >= ℝ[shape[1]]:
            y = ℝ[shape[1]*(1 - machine_ϵ)]
        if z >= ℝ[shape[2]]:
            z = ℝ[shape[2]*(1 - machine_ϵ)]
        # Indices of the 8 vertices (6 faces)
        # of the grid surrounding (x, y, z).
        x_lower = int(x)
        y_lower = int(y)
        z_lower = int(z)
        x_upper = x_lower + 1
        y_upper = y_lower + 1
        z_upper = z_lower + 1
        # The linear weights according to the
        # CIC rule W = 1 - |dist| if |dist| < 1.
        Wxl = x_upper - x  # = 1 - (x - x_lower)
        Wyl = y_upper - y  # = 1 - (y - y_lower)
        Wzl = z_upper - z  # = 1 - (z - z_lower)
        Wxu = x - x_lower  # = 1 - (x_upper - x)
        Wyu = y - y_lower  # = 1 - (y_upper - y)
        Wzu = z - z_lower  # = 1 - (z_upper - z)
        # Assign the raw weights to the δ grid
        δ_noghosts[x_lower, y_lower, z_lower] += ℝ[Wxl*Wyl]*Wzl
        δ_noghosts[x_lower, y_lower, z_upper] += ℝ[Wxl*Wyl]*Wzu
        δ_noghosts[x_lower, y_upper, z_lower] += ℝ[Wxl*Wyu]*Wzl
        δ_noghosts[x_lower, y_upper, z_upper] += ℝ[Wxl*Wyu]*Wzu
        δ_noghosts[x_upper, y_lower, z_lower] += ℝ[Wxu*Wyl]*Wzl
        δ_noghosts[x_upper, y_lower, z_upper] += ℝ[Wxu*Wyl]*Wzu
        δ_noghosts[x_upper, y_upper, z_lower] += ℝ[Wxu*Wyu]*Wzl
        δ_noghosts[x_upper, y_upper, z_upper] += ℝ[Wxu*Wyu]*Wzu
        # Extract momentum of the i'th particle
        momx_i = momx[i]
        momy_i = momy[i]
        momz_i = momz[i]
        # Assign the unweighted momentum to the ux grid
        ux_noghosts[x_lower, y_lower, z_lower] += momx_i
        ux_noghosts[x_lower, y_lower, z_upper] += momx_i
        ux_noghosts[x_lower, y_upper, z_lower] += momx_i
        ux_noghosts[x_lower, y_upper, z_upper] += momx_i
        ux_noghosts[x_upper, y_lower, z_lower] += momx_i
        ux_noghosts[x_upper, y_lower, z_upper] += momx_i
        ux_noghosts[x_upper, y_upper, z_lower] += momx_i
        ux_noghosts[x_upper, y_upper, z_upper] += momx_i
        # Assign the unweighted momentum to the uy grid
        uy_noghosts[x_lower, y_lower, z_lower] += momy_i
        uy_noghosts[x_lower, y_lower, z_upper] += momy_i
        uy_noghosts[x_lower, y_upper, z_lower] += momy_i
        uy_noghosts[x_lower, y_upper, z_upper] += momy_i
        uy_noghosts[x_upper, y_lower, z_lower] += momy_i
        uy_noghosts[x_upper, y_lower, z_upper] += momy_i
        uy_noghosts[x_upper, y_upper, z_lower] += momy_i
        uy_noghosts[x_upper, y_upper, z_upper] += momy_i
        # Assign the unweighted momentum to the uz grid
        uz_noghosts[x_lower, y_lower, z_lower] += momz_i
        uz_noghosts[x_lower, y_lower, z_upper] += momz_i
        uz_noghosts[x_lower, y_upper, z_lower] += momz_i
        uz_noghosts[x_lower, y_upper, z_upper] += momz_i
        uz_noghosts[x_upper, y_lower, z_lower] += momz_i
        uz_noghosts[x_upper, y_lower, z_upper] += momz_i
        uz_noghosts[x_upper, y_upper, z_lower] += momz_i
        uz_noghosts[x_upper, y_upper, z_upper] += momz_i
    # The particle data is no longer needed. Free it to save memory
    component.representation = 'particles'
    component.resize(1)
    # Values of local pseudo mesh points contribute to the lower
    # mesh points of the domain (fluid) grids on other processes.
    # Do the needed communications.
    communicate_domain_boundaries(δ_mv, mode=0)
    communicate_domain_boundaries(ux_mv, mode=0)
    communicate_domain_boundaries(uy_mv, mode=0)
    communicate_domain_boundaries(uz_mv, mode=0)
    # The formula for getting the δ grid from the particles is
    # δ = CIC_grid/μ - 1,
    # where CIC_grid is the interpolated grid (using just raw weights,
    # meaning that each particle contribute a total of 1 to the grid)
    # and μ is the mean of the interpolated grid:
    # μ = sum(CIC_grid)/prod(CIC_grid.shape)
    #   = N/gridsize**3,
    # As of now, the values in δ correspond to those of CIC_grid.
    #
    # The formula for getting the ux, uy and uz grid from the
    # particles is
    # u = mom/(mass*a),
    # where a is the current scale factor.
    # As of now, the values in ux, uy and uz correspond to those of mom.
    #
    # Apply the above transformations.
    δ_fac = float(component.gridsize)**3/component.N
    u_fac = 1/(component.mass*universals.a)
    tot_δ = 0
    for i in range(size):
        δ[i] = δ[i]*δ_fac - 1
        tot_δ += δ[i]
        ux[i] *= u_fac
        uy[i] *= u_fac
        uz[i] *= u_fac
    # Re-insert the original representation string
    component.representation = original_representation

# Function for CIC interpolating the component coordinates of all
# components to the slabs. Each particle/fluid element will be weighted
# by their mass.
@cython.header(# Arguments
               components='list',
               )
def CIC_components2slabs(components):
    # Nullify the slab and φ grid
    slab[...] = 0
    φ[...] = 0
    # Interpolate component coordinates weighted by their masses to φ
    CIC_components2domain_grid(components, φ)
    # Communicate the interpolated data in φ into the slabs
    φ2slabs()

# Function for transfering the interpolated data in φ to the slabs
@cython.header(# Locals
               ℓ='int',
               request='object',  # mpi4py.MPI.Request object
               )
def φ2slabs():
    # Communicate the interpolated φ to the slabs
    for ℓ in range(N_φ2slabs_communications):
        # Send part of the local domain
        # grid to the corresponding process.
        if ℓ < slabs2φ_sendrecv_ranks.shape[0]:
            # A non-blocking send is used, because the communication
            # is not pairwise.
            # Since the slabs extend throughout the entire yz-plane,
            # we should send the entire yz-part of φ
            # (excluding ghost and pseudo points).
            request = smart_mpi(φ_noghosts[φ_sendrecv_i_start[ℓ]:φ_sendrecv_i_end[ℓ],
                                           :φ_size_j,
                                           :φ_size_k],
                                dest=slabs2φ_sendrecv_ranks[ℓ],
                                mpifun='Isend')
        # The lower ranks storing the slabs receives the message.
        # In the x-dimension, the slabs are always thinner than (or at
        # least as thin as) the domain.
        if ℓ < φ2slabs_recvsend_ranks.shape[0]:
            smart_mpi(block_recv=slab[:,
                                      slab_sendrecv_j_start[ℓ]:slab_sendrecv_j_end[ℓ],
                                      slab_sendrecv_k_start[ℓ]:slab_sendrecv_k_end[ℓ]],
                      source=φ2slabs_recvsend_ranks[ℓ],
                      mpifun='Recv')
        # Wait for the non-blockind send to be complete before
        # continuing. Otherwise, data in the send buffer - which is
        # still in use by the non-blocking send - might get overwritten
        # by the next (non-blocking) send.
        request.wait()

# Function for transfering the data in the slabs to φ
@cython.header(# Locals
               ℓ='int',
               request='object',  # mpi4py.MPI.Request object
               )
def slabs2φ():
    # Communicate the slabs to φ
    for ℓ in range(N_φ2slabs_communications):
        # The lower ranks storing the slabs sends part of their slab
        if ℓ < φ2slabs_recvsend_ranks.shape[0]:
            # A non-blocking send is used, because the communication
            # is not pairwise.
            # In the x-dimension, the slabs are always thinner than (or
            # at least as thin as) the domain.
            request = smart_mpi(block_send=slab[:,
                                                slab_sendrecv_j_start[ℓ]:slab_sendrecv_j_end[ℓ],
                                                slab_sendrecv_k_start[ℓ]:slab_sendrecv_k_end[ℓ]],
                                dest=φ2slabs_recvsend_ranks[ℓ],
                                mpifun='Isend')
        # The corresponding process receives the message.
        # Since the slabs extend throughout the entire yz-plane,
        # we receive into the entire yz-part of φ
        # (excluding ghost and pseudo points).
        if ℓ < slabs2φ_sendrecv_ranks.shape[0]:
            smart_mpi(block_recv=φ_noghosts[φ_sendrecv_i_start[ℓ]:φ_sendrecv_i_end[ℓ],
                                            :φ_size_j,
                                            :φ_size_k],
                      source=slabs2φ_sendrecv_ranks[ℓ],
                      mpifun='Recv')
        # Wait for the non-blockind send to be complete before
        # continuing. Otherwise, data in the send buffer - which is
        # still in use by the non-blocking send - might get overwritten
        # by the next (non-blocking) send.
        request.wait()
    # The right/forward/upper boundaries (the layer of pseudo points,
    # not the ghost layer) of φ should be a copy of the
    # left/backward/lower boundaries of the neighboring
    # right/forward/upper domain. Do the needed communication.
    communicate_domain_boundaries(φ, mode=1)
    # Communicate the ghost layers of φ
    communicate_domain_ghosts(φ)

# Function performing a forward Fourier transformation of the slabs
@cython.header()
def slabs_FFT():
    # Fourier transform the slabs forwards from real to Fourier space
    fftw_execute(plan_forward)

# Function performing a backward Fourier transformation of the slabs
@cython.header()
def slabs_IFFT():
    # Fourier transform the slabs backwards from Fourier to real
    # space. Note that this is an unnormalized transform, as defined by
    # FFTW. To do the normalization, divide all elements of the slab
    # by φ_gridsize**3.
    fftw_execute(plan_backward)

# This function differentiates a given grid along the dim dimension.
# The parsed grid must include psuedo and ghost points. The pseudo
# points will be differentiated along with the actual grid points.
# To achieve proper units, the physical grid spacing may be specified
# as h. If not given, grid units (h == 1) are used.
# If the buffer argument is not given, the meshbuf is used to store the
# result. A view of this buffer will be returned.
# If a buffer is supplied, the result of the differentiations will be
# added to this buffer instead of being stored in meshbuf. Note that
# this buffer has to be contiguous (this criterion could be removed
# if needed).
# Note that a grid cannot be differentiated in-place by parsing the
# grid as both the first and third argument, as the differentiation
# of each grid point requires information from the original
# (non-differentiated) grid.
# The optional order argument specifies the order of accuracy of the
# differentiation (the number of neighboring grid points used to
# approximate the derivative).
@cython.pheader(# Arguments
                grid='double[:, :, ::1]',
                dim='int',
                h='double',
                buffer='double[:, :, ::1]',
                order='int',
                # Locals
                buffer_i='Py_ssize_t',
                buffer_j='Py_ssize_t',
                buffer_k='Py_ssize_t',
                grid_im1='Py_ssize_t',
                grid_im2='Py_ssize_t',
                grid_ip1='Py_ssize_t',
                grid_ip2='Py_ssize_t',
                grid_jm1='Py_ssize_t',
                grid_jm2='Py_ssize_t',
                grid_jp1='Py_ssize_t',
                grid_jp2='Py_ssize_t',
                grid_km1='Py_ssize_t',
                grid_km2='Py_ssize_t',
                grid_kp1='Py_ssize_t',
                grid_kp2='Py_ssize_t',
                i='Py_ssize_t',
                j='Py_ssize_t',
                k='Py_ssize_t',
                meshbuf_mv='double[:, :, ::1]',
                shape='tuple',
                returns='double[:, :, ::1]',
                )
def diff(grid, dim, h=1, buffer=None, order=4):
    if buffer is None:
        # Get a buffer for storing the data
        shape = tuple([grid.shape[dim] - 2*2 for dim in range(3)])
        meshbuf_mv = get_meshbuf(shape)
        # Now do the differentiation and save the results in the buffer
        if dim == 0:
            if order == 2:
                abort('not implemented yet')
            elif order == 4:
                # Differentiate along the x-direction via the four point rule
                for i in range(2, grid.shape[0] - 2):
                    buffer_i = i - 2
                    grid_ip1, grid_im1, grid_ip2, grid_im2 = i + 1, i - 1, i + 2, i - 2
                    for j in range(2, grid.shape[1] - 2):
                        buffer_j = j - 2
                        for k in range(2, grid.shape[2] - 2):
                            meshbuf_mv[buffer_i,
                                       buffer_j, k - 2] = (+ ℝ[2/(3*h)] *(+ grid[grid_ip1, j, k]
                                                                          - grid[grid_im1, j, k])
                                                           - ℝ[1/(12*h)]*(+ grid[grid_ip2, j, k]
                                                                          - grid[grid_im2, j, k]))
            elif master:
                abort('Differentiation of order {} is not implemented'.format(order))
        elif dim == 1:
            if order == 2:
                abort('not implemented yet')
            elif order == 4:
                # Differentiate along the y-direction via the four point rule
                for j in range(2, grid.shape[1] - 2):
                    buffer_j = j - 2
                    grid_jp1, grid_jm1, grid_jp2, grid_jm2 = j + 1, j - 1, j + 2, j - 2
                    for k in range(2, grid.shape[2] - 2):
                        buffer_k = k - 2
                        for i in range(2, grid.shape[0] - 2):
                            meshbuf_mv[i - 2, buffer_j,
                                              buffer_k] = (+ ℝ[2/(3*h)] *(+ grid[i, grid_jp1, k]
                                                                          - grid[i, grid_jm1, k])
                                                           - ℝ[1/(12*h)]*(+ grid[i, grid_jp2, k]
                                                                          - grid[i, grid_jm2, k]))
            elif master:
                abort('Differentiation of order {} is not implemented'.format(order))
        elif dim == 2:
            if order == 2:
                abort('not implemented yet')
            elif order == 4:
                # Differentiate along the z-direction via the four point rule
                for k in range(2, grid.shape[2] - 2):
                    buffer_k = k - 2
                    grid_kp1, grid_km1, grid_kp2, grid_km2 = k + 1, k - 1, k + 2, k - 2
                    for i in range(2, grid.shape[0] - 2):
                        buffer_i = i - 2
                        for j in range(2, grid.shape[1] - 2):
                            meshbuf_mv[buffer_i, j - 2,
                                       buffer_k] = (+ ℝ[2/(3*h)] *(+ grid[i, j, grid_kp1]
                                                                   - grid[i, j, grid_km1])
                                                    - ℝ[1/(12*h)]*(+ grid[i, j, grid_kp2]
                                                                   - grid[i, j, grid_km2]))
            elif master:
                abort('Differentiation of order {} is not implemented'.format(order))
        elif master:
            abort(unicode('The dim argument should be ∈ {0, 1, 2}'))
        return meshbuf_mv
    else:
        # Do the differentiation
        # and add the results to the supplied buffer.
        if dim == 0:
            if order == 1:
                # Differentiate along the x-direction via forward difference
                for i in range(2, grid.shape[0] - 2):
                    buffer_i = i - 2
                    grid_ip1 = i + 1
                    for j in range(2, grid.shape[1] - 2):
                        buffer_j = j - 2
                        for k in range(2, grid.shape[2] - 2):
                            buffer[buffer_i,
                                   buffer_j, k - 2] += ℝ[1/h]*(+ grid[grid_ip1, j, k]
                                                               - grid[i,        j, k])
            elif order == 2:
                # Differentiate along the x-direction via the two point rule
                for i in range(2, grid.shape[0] - 2):
                    buffer_i = i - 2
                    grid_ip1, grid_im1 = i + 1, i - 1
                    for j in range(2, grid.shape[1] - 2):
                        buffer_j = j - 2
                        for k in range(2, grid.shape[2] - 2):
                            buffer[buffer_i,
                                   buffer_j, k - 2] += (ℝ[1/(2*h)]*(+ grid[grid_ip1, j, k]
                                                                    - grid[grid_im1, j, k])
                                                        )
            elif order == 4:
                # Differentiate along the x-direction via the four point rule
                for i in range(2, grid.shape[0] - 2):
                    buffer_i = i - 2
                    grid_ip1, grid_im1, grid_ip2, grid_im2 = i + 1, i - 1, i + 2, i - 2
                    for j in range(2, grid.shape[1] - 2):
                        buffer_j = j - 2
                        for k in range(2, grid.shape[2] - 2):
                            buffer[buffer_i,
                                   buffer_j, k - 2] += (+ ℝ[2/(3*h)] *(+ grid[grid_ip1, j, k]
                                                                       - grid[grid_im1, j, k])
                                                        - ℝ[1/(12*h)]*(+ grid[grid_ip2, j, k]
                                                                       - grid[grid_im2, j, k]))
            elif master:
                abort('Differentiation of order {} is not implemented'.format(order))
        elif dim == 1:
            if order == 1:
                # Differentiate along the y-direction via forward difference
                for j in range(2, grid.shape[1] - 2):
                    buffer_j = j - 2
                    grid_jp1 = j + 1
                    for k in range(2, grid.shape[2] - 2):
                        buffer_k = k - 2
                        for i in range(2, grid.shape[0] - 2):
                            buffer[i - 2, buffer_j,
                                          buffer_k] += ℝ[1/h]*(+ grid[i, grid_jp1, k]
                                                               - grid[i, j,        k])
            elif order == 2:
                # Differentiate along the y-direction via the two point rule
                for j in range(2, grid.shape[1] - 2):
                    buffer_j = j - 2
                    grid_jp1, grid_jm1 = j + 1, j - 1
                    for k in range(2, grid.shape[2] - 2):
                        buffer_k = k - 2
                        for i in range(2, grid.shape[0] - 2):
                            buffer[i - 2, buffer_j,
                                          buffer_k] += (ℝ[1/(2*h)]*(+ grid[i, grid_jp1, k]
                                                                    - grid[i, grid_jm1, k])
                                                        )
            elif order == 4:
                # Differentiate along the y-direction via the four point rule
                for j in range(2, grid.shape[1] - 2):
                    buffer_j = j - 2
                    grid_jp1, grid_jm1, grid_jp2, grid_jm2 = j + 1, j - 1, j + 2, j - 2
                    for k in range(2, grid.shape[2] - 2):
                        buffer_k = k - 2
                        for i in range(2, grid.shape[0] - 2):
                            buffer[i - 2, buffer_j,
                                          buffer_k] += (+ ℝ[2/(3*h)] *(+ grid[i, grid_jp1, k]
                                                                       - grid[i, grid_jm1, k])
                                                        - ℝ[1/(12*h)]*(+ grid[i, grid_jp2, k]
                                                                       - grid[i, grid_jm2, k]))
            elif master:
                abort('Differentiation of order {} is not implemented'.format(order))
        elif dim == 2:
            if order == 1:
                # Differentiate along the z-direction via forward difference
                for k in range(2, grid.shape[2] - 2):
                    buffer_k = k - 2
                    grid_kp1 = k + 1
                    for i in range(2, grid.shape[0] - 2):
                        buffer_i = i - 2
                        for j in range(2, grid.shape[1] - 2):
                            buffer[buffer_i, j - 2,
                                   buffer_k] += ℝ[1/h]*(+ grid[i, j, grid_kp1]
                                                        - grid[i, j, k])
            elif order == 2:
                # Differentiate along the z-direction via the two point rule
                for k in range(2, grid.shape[2] - 2):
                    buffer_k = k - 2
                    grid_kp1, grid_km1 = k + 1, k - 1
                    for i in range(2, grid.shape[0] - 2):
                        buffer_i = i - 2
                        for j in range(2, grid.shape[1] - 2):
                            buffer[buffer_i, j - 2,
                                   buffer_k] += (ℝ[1/(2*h)]*(+ grid[i, j, grid_kp1]
                                                             - grid[i, j, grid_km1])
                                                 )
            elif order == 4:
                # Differentiate along the z-direction via the four point rule
                for k in range(2, grid.shape[2] - 2):
                    buffer_k = k - 2
                    grid_kp1, grid_km1, grid_kp2, grid_km2 = k + 1, k - 1, k + 2, k - 2
                    for i in range(2, grid.shape[0] - 2):
                        buffer_i = i - 2
                        for j in range(2, grid.shape[1] - 2):
                            buffer[buffer_i, j - 2,
                                   buffer_k] += (+ ℝ[2/(3*h)] *(+ grid[i, j, grid_kp1]
                                                                - grid[i, j, grid_km1])
                                                 - ℝ[1/(12*h)]*(+ grid[i, j, grid_kp2]
                                                                - grid[i, j, grid_km2]))
            elif master:
                abort('Differentiation of order {} is not implemented'.format(order))
        elif master:
            abort(unicode('The dim argument should be ∈ {0, 1, 2}'))
        return buffer

# Function which wraps the meshbuf buffer in a memory view
@cython.header(# Arguments
               shape='tuple',
               nullify='bint',
               # Locals
               i='Py_ssize_t',
               meshbuf_mv='double[:, :, ::1]',
               size='Py_ssize_t',
               returns='double[:, :, ::1]',
               )
def get_meshbuf(shape, nullify=False):
    global meshbuf, meshbuf_size
    # The buffer size needed
    size = np.prod(shape)
    # Enlarge meshbuf if needed
    if size > meshbuf_size:
        meshbuf_size = size
        meshbuf = realloc(meshbuf, meshbuf_size*sizeof('double'))
    # Wrap meshbuf in a memory view of the given shape
    meshbuf_mv = cast(meshbuf, 'double[:shape[0], :shape[1], :shape[2]]')
    # If requested, nullify the buffer
    if nullify:
        for i in range(size):
            meshbuf[i] = 0
    # Return the memory view 
    return meshbuf_mv



# Initializes φ and related stuff (e.g. the slabs) at import time,
# if φ should be used.
cython.declare(# The slab grid
               fftw_struct='fftw_return_struct',
               slab_size_i='ptrdiff_t',
               slab_size_j='ptrdiff_t',
               slab_start_i='ptrdiff_t',
               slab_start_j='ptrdiff_t',
               slab='double[:, :, ::1]',
               plan_forward='fftw_plan',
               plan_backward='fftw_plan',
               # The φ grid
               φ='double[:, :, ::1]',
               φ_noghosts='double[:, :, :]',
               φ_size_i='Py_ssize_t',
               φ_size_j='Py_ssize_t',
               φ_size_k='Py_ssize_t',
               # For communication between PM and φ
               N_φ2slabs_communications='int',
               φ_sendrecv_i_end='int[::1]',
               φ_sendrecv_i_start='int[::1]',
               slabs2φ_sendrecv_ranks='int[::1]',
               slab_sendrecv_j_start='int[::1]',
               slab_sendrecv_k_start='int[::1]',
               slab_sendrecv_j_end='int[::1]',
               slab_sendrecv_k_end='int[::1]',
               φ2slabs_recvsend_ranks='int[::1]',
               )
if use_φ:
    # Initialize the slab grid, distributed along the x-dimension
    # when in real space and along the y dimension when in
    # Fourier space.
    if not cython.compiled:
        # Initialization of the slabs in pure Python
        slab_size_i = slab_size_j = int(φ_gridsize/nprocs)
        if master and slab_size_i != φ_gridsize/nprocs:
            # If φ_gridsize is not divisible by nprocs, the code cannot
            # figure out exactly how FFTW distribute the grid among the
            # processes. In stead of guessing, do not even try to
            # emulate the behaviour of FFTW.
            msg = ('The PM method in pure Python mode only works '
                   'when\n{}_gridsize is divisible by the number '
                   'of processes!'.format(unicode('φ'))
                   )
            abort(msg)
        slab_start_i = slab_start_j = slab_size_i*rank
        slab = empty((slab_size_i, φ_gridsize, slab_size_padding), dtype=C2np['double'])
        # The output of the following function is formatted just
        # like that of the MPI implementation of FFTW.
        plan_backward = 'plan_backward'
        plan_forward = 'plan_forward'
    else:
        # Sanity check on user defined fftw_rigor
        fftw_rigors = ('exhaustive', 'patient', 'measure', 'estimate')
        if fftw_rigor not in fftw_rigors:
            masterwarn(('Did not recognize fftw_rigor "{}". '
                        + 'Falling back to "estimate".').format(fftw_rigor))
            fftw_rigor = 'estimate'
        # Use a better rigor if wisdom already exist
        for fftw_rigor in fftw_rigors[:(fftw_rigors.index(fftw_rigor) + 1)]:
            wisdom_filename = ('.fftw_wisdom_gridsize={}_nprocs={}_rigor={}'
                               .format(φ_gridsize, nprocs, fftw_rigor))
            if os.path.isfile(wisdom_filename):
                break
        # Initialize fftw_mpi, allocate the grid, initialize the
        # local grid sizes and start indices and do FFTW planning.
        if not os.path.isfile(wisdom_filename):
            msg = ('Acquiring FFTW wisdom ({}) for grid of linear size {} on {} {} ...'
                   ).format(fftw_rigor,
                            φ_gridsize,
                            nprocs,
                            'processes' if nprocs > 1 else 'process')
            masterprint(msg)
            fftw_struct = fftw_setup(φ_gridsize,
                                     φ_gridsize,
                                     φ_gridsize,
                                     bytes(fftw_rigor, encoding='ascii'))
            masterprint('done')
        else:
            fftw_struct = fftw_setup(φ_gridsize,
                                     φ_gridsize,
                                     φ_gridsize,
                                     bytes(fftw_rigor, encoding='ascii'))
        # If less rigouros wisdom exists for the same problem,
        # delete it.
        for rigor in fftw_rigors[(fftw_rigors.index(fftw_rigor) + 1):]:
            wisdom_filename = ('.fftw_wisdom_gridsize={}_nprocs={}_rigor={}'
                               .format(φ_gridsize, nprocs, rigor))
            if master and os.path.isfile(wisdom_filename):
                os.remove(wisdom_filename)
        # Unpack every variable from fftw_struct except for the grid
        slab_size_i = fftw_struct.gridsize_local_i
        slab_size_j = fftw_struct.gridsize_local_j
        slab_start_i = fftw_struct.gridstart_local_i
        slab_start_j = fftw_struct.gridstart_local_j
        plan_forward  = fftw_struct.plan_forward
        plan_backward = fftw_struct.plan_backward
        # Now unpack the grid from fftw_struct, but wrap it in a
        # memoryview. Looping over this memoryview should be done as 
        # noted in fft.c, but use slab[i, j, k] when in real space
        # and slab[j, i, k] when in Fourier space.
        if slab_size_i > 0:
            slab = cast(fftw_struct.grid,
                        'double[:slab_size_i, :φ_gridsize, :slab_size_padding]')
        else:
            # The process do not participate in the FFT computations
            slab = empty((0, φ_gridsize, slab_size_padding))
    # Initialize the φ grid, distributed according to the
    # domain decomposition of the box.
    # The φ grid stores the same data as the slab grid,
    # but instead of being distributed in slabs, it is distributed
    # accoring to the domain of the local process.
    # It is given an additional layer of points of thickness 1 in
    # the right/forward/upward ends. These are the pseudo points,
    # having the same physical coordinates as the first points in the
    # next domain, for the three directions.
    # Additionally, around the whole cube, a layer of points of
    # thickness 2 is added, called the ghost layer. The data here
    # are simply copied over from neighboring domains.
    φ = empty([φ_gridsize//domain_subdivisions[dim] + 1 + 2*2 for dim in range(3)],
              dtype=C2np['double'])
    # Memoryview of the φ grid without the ghost layers
    φ_noghosts = φ[2:(φ.shape[0] - 2), 2:(φ.shape[1] - 2), 2:(φ.shape[2] - 2)]
    # Test if the grid has been constructed correctly.
    # If not it is because nprocs and φ_gridsize are incompatible.
    if master:
        if any(φ_gridsize != domain_subdivisions[dim]*(φ_noghosts.shape[dim] - 1)
               for dim in range(3)):
            msg = ('A {}_gridsize of {} cannot be equally shared among {} processes'
                   .format(unicode('φ'), φ_gridsize, nprocs))
            abort(msg)
        if any((φ_noghosts.shape[dim] - 1) < 1 for dim in range(3)):
            msg = ('A {}_gridsize of {} is too small for {} processes'
                   .format(unicode('φ'), φ_gridsize, nprocs))
            abort(msg)
    # The size (number of grid points) of the truly local part of the φ,
    # excluding both ghost layers and pseudo points, for each dimension.
    φ_size_i = φ_noghosts.shape[0] - 1
    φ_size_j = φ_noghosts.shape[1] - 1
    φ_size_k = φ_noghosts.shape[2] - 1
    # Check if the slab is large enough for P3M to work,
    # if the P3M algorithm is to be used.
    if master and use_P3M:
        if (   φ_size_i < P3M_scale*P3M_cutoff
            or φ_size_j < P3M_scale*P3M_cutoff
            or φ_size_k < P3M_scale*P3M_cutoff):
            msg = ('A {}_gridsize of {} and {} processes results in the following domain '
                   'partitioning: {}.\n The smallest domain width is {} grid cells, while the '
                   'choice of P3M_scale ({}) and P3M_cutoff ({})\nmeans that the domains must '
                   'be at least {} grid cells for the P3M algorithm to work.'
                   ).format(unicode('φ'),
                            φ_gridsize,
                            nprocs,
                            list(domain_subdivisions),
                            np.min([φ_size_i, φ_size_j, φ_size_k]),
                            P3M_scale,
                            P3M_cutoff,
                            int(np.ceil(P3M_scale*P3M_cutoff)),
                            )
            abort(msg)
        if ((   φ_size_i < 2*P3M_scale*P3M_cutoff
             or φ_size_j < 2*P3M_scale*P3M_cutoff
             or φ_size_k < 2*P3M_scale*P3M_cutoff) and np.min(domain_subdivisions) < 3):
            # If the above is True, the left and the right (say) process
            # is the same and the boundaries will be send to it twice,
            # and these will overlap with each other in the left/right
            # domain, eventually leading to gravity being applied twice.
            msg = ('A {}_gridsize of {} and {} processes results in the following domain '
                   'partitioning: {}.\nThe smallest domain width is {} grid cells, while the '
                   'choice of P3M_scale ({}) and P3M_cutoff ({})\nmeans that the domains must '
                   'be at least {} grid cells for the P3M algorithm to work.'
                   ).format(unicode('φ'),
                            φ_gridsize,
                            nprocs,
                            list(domain_subdivisions),
                            np.min([φ_size_i, φ_size_j, φ_size_k]),
                            P3M_scale,
                            P3M_cutoff,
                            int(np.ceil(2*P3M_scale*P3M_cutoff)),
                            )
            abort(msg)
    # Additional information about φ and the slabs,
    # used in the φ2slabs function.
    # The global start and end indices of the local domain in the total
    # φ grid.
    φ_start_i = domain_layout_local_indices[0]*φ_size_i
    φ_start_j = domain_layout_local_indices[1]*φ_size_j
    φ_start_k = domain_layout_local_indices[2]*φ_size_k
    φ_end_i = φ_start_i + φ_size_i
    φ_end_j = φ_start_j + φ_size_j
    φ_end_k = φ_start_k + φ_size_k
    # The value of slab_size_i is determined by FFTW. It is equal to
    # φ_gridsize//nprocs (though not less than 1) for
    # ranks < φ_gridsize. If nprocs > φ_gridsize so
    # that slab_size_i == 1, the higher ranked processes cannot
    # take part in the FFT computation, and so their local version
    # of slab_size_i is set to 0.
    # The global version - slab_size_i_global - defined below,
    # is equal to the nonzero value on all processes.
    if rank < φ_gridsize and slab_size_i == 0:
        slab_size_i = 1 
    slab_size_i_global = slab_size_i
    if slab_size_i_global == 0:
        slab_size_i_global = 1
    # Find local i-indices to send and to which process by
    # shifting a piece of the number line in order to match
    # the communication pattern used.
    φ_sendrecv_i_start = np.roll(asarray([ℓ - φ_start_i
                                          for ℓ in range(φ_start_i,
                                                          φ_end_i,
                                                          slab_size_i_global)],
                                         dtype=C2np['int']),
                                 -rank)
    φ_sendrecv_i_end = np.roll(asarray([ℓ - φ_start_i + slab_size_i_global
                                        for ℓ in range(φ_start_i,
                                                        φ_end_i,
                                                        slab_size_i_global)],
                                        dtype=C2np['int']),
                               -rank)
    slabs2φ_sendrecv_ranks = np.roll(asarray([ℓ//slab_size_i_global
                                              for ℓ in range(φ_start_i, 
                                                              φ_end_i,
                                                              slab_size_i_global)],
                                             dtype=C2np['int']),
                                     -rank)
    # FIXME: THIS IS NOT SUFFICIENT! IF nprocs > φ_gridsize THE PROGRAM WILL HALT AT φ2slabs and slabs2φ !!!!!
    # Communicate the start and end (j, k)-indices of the slab,
    # where parts of the local domains should be received into.
    slab_sendrecv_j_start  = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_k_start  = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_j_end    = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_k_end    = empty(nprocs, dtype=C2np['int'])
    φ2slabs_recvsend_ranks = empty(nprocs, dtype=C2np['int'])
    cython.declare(index='Py_ssize_t')  # Just to remove Cython warning
    index = 0
    for ℓ in range(nprocs):
        # Process ranks to send/recieve to/from
        rank_send = np.mod(rank + ℓ, nprocs)
        rank_recv = np.mod(rank - ℓ, nprocs)
        # Send the global y and z start and end indices of the domain
        # to be send, if anything should be send to process rank_send.
        # Otherwise send None.
        sendtuple = ((φ_start_j, φ_start_k, φ_end_j, φ_end_k)
                     if rank_send in asarray(slabs2φ_sendrecv_ranks) else None)
        recvtuple = sendrecv(sendtuple, dest=rank_send, source=rank_recv)
        if recvtuple is not None:
            slab_sendrecv_j_start[index] = recvtuple[0]
            slab_sendrecv_k_start[index] = recvtuple[1]
            slab_sendrecv_j_end[index]   = recvtuple[2]
            slab_sendrecv_k_end[index]   = recvtuple[3]
            φ2slabs_recvsend_ranks[index]    = rank_recv
            index += 1
    # Cut off the tails
    slab_sendrecv_j_start = slab_sendrecv_j_start[:index]
    slab_sendrecv_k_start = slab_sendrecv_k_start[:index]
    slab_sendrecv_j_end = slab_sendrecv_j_end[:index]
    slab_sendrecv_k_end = slab_sendrecv_k_end[:index]
    φ2slabs_recvsend_ranks = φ2slabs_recvsend_ranks[:index]
    # The maximum number of communications it takes to communicate
    # φ to the slabs (or vice versa).
    N_φ2slabs_communications = np.max([slabs2φ_sendrecv_ranks.shape[0],
                                       φ2slabs_recvsend_ranks.shape[0]])
    # Initialize meshbuf, a buffer used for temporary (scalar) mesh
    # data. It is e.g. used in the diff function, which is e.g. used to
    # differentiate the potential. Therefore, it should at least have
    # a size equal to φ (ghost points not needed). The exact shape of
    # meshbuf should be decided dynamically by wrapping it in a
    # memoryview as needed. For this, use the get_meshbuf function,
    # which will also enlarge meshbuf if needed.
    cython.declare(meshbuf_size='Py_ssize_t',
                   meshbuf='double*',
                   )
    meshbuf_size = φ_noghosts.shape[0]*φ_noghosts.shape[1]*φ_noghosts.shape[2]
    meshbuf = malloc(meshbuf_size*sizeof('double'))
else:
    # As these should be importable,
    # they need to be assigned even if not used.
    slab = φ = φ_noghosts = empty((1, 1, 1), dtype=C2np['double'])
    slab_size_i = 1
    slab_size_j = 1
    slab_start_i = 0
    slab_start_j = 0

