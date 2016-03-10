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
cimport('from communication import cutout_domains, neighboring_ranks')

# Seperate but roughly equivalent imports in pure Python and Cython
if not cython.compiled:
    # FFT functionality via NumPy
    from numpy.fft import rfftn, irfftn
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
               grid='double[:, :, ::1]',
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
               gridsize_x_minus_1='int',
               gridsize_y_minus_1='int',
               gridsize_z_minus_1='int',
               x_lower='Py_ssize_t',
               x_upper='Py_ssize_t',
               y_lower='Py_ssize_t',
               y_upper='Py_ssize_t',
               z_lower='Py_ssize_t',
               z_upper='Py_ssize_t',
               returns='double',
               )
def CIC_grid2coordinates_scalar(grid, x, y, z):
    """This function look up tabulated scalars in a grid and
    interpolates to (x, y, z) via the cloud in cell (CIC) method. Input
    arguments must be normalized so that 0 <= x, y, z < 1. If x, y or z
    is exactly equal to 1, they will be corrected to 1 - ϵ. It is
    assumed that the grid is nonperiodic (that is,
    the grid has closed ends).
    """
    # Extract the size of the regular grid
    gridsize_x_minus_1 = grid.shape[0] - 1
    gridsize_y_minus_1 = grid.shape[1] - 1
    gridsize_z_minus_1 = grid.shape[2] - 1
    # Correct for extreme values in the passed coordinates.
    # This is to catch inputs which are slighly larger than 1 due to
    # numerical errors.
    if x >= 1:
        x = 1 - ℝ[2*machine_ϵ]
    if y >= 1:
        y = 1 - ℝ[2*machine_ϵ]
    if z >= 1:
        z = 1 - ℝ[2*machine_ϵ]
    # Scale the coordinates so that 0 <= x, y, z < (gridsize - 1)
    x *= gridsize_x_minus_1
    y *= gridsize_y_minus_1
    z *= gridsize_z_minus_1
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
    return (  grid[x_lower, y_lower, z_lower]*Wxl*Wyl*Wzl
            + grid[x_lower, y_lower, z_upper]*Wxl*Wyl*Wzu
            + grid[x_lower, y_upper, z_lower]*Wxl*Wyu*Wzl
            + grid[x_lower, y_upper, z_upper]*Wxl*Wyu*Wzu
            + grid[x_upper, y_lower, z_lower]*Wxu*Wyl*Wzl
            + grid[x_upper, y_lower, z_upper]*Wxu*Wyl*Wzu
            + grid[x_upper, y_upper, z_lower]*Wxu*Wyu*Wzl
            + grid[x_upper, y_upper, z_upper]*Wxu*Wyu*Wzu)

# Function for doing lookup in a grid with vector values and
# CIC-interpolating to specified coordinates
@cython.header(# Argument
               grid='double[:, :, :, ::1]',
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
               dim='Py_ssize_t',
               gridsize_x_minus_1='int',
               gridsize_y_minus_1='int',
               gridsize_z_minus_1='int',
               x_lower='Py_ssize_t',
               x_upper='Py_ssize_t',
               y_lower='Py_ssize_t',
               y_upper='Py_ssize_t',
               z_lower='Py_ssize_t',
               z_upper='Py_ssize_t',
               returns='double*',
               )
def CIC_grid2coordinates_vector(grid, x, y, z):
    """This function look up tabulated vectors in a grid and
    interpolates to (x, y, z) via the cloud in cell (CIC) method.
    Input arguments must be normalized so that 0 <= x, y, z < 1.
    If x, y or z is exactly equal to 1, they will be corrected to 1 - ϵ.
    It is assumed that the grid is nonperiodic (that is, the first and
    the last gridpoint in any dimension are physical distinct and that
    the grid has closed ends).
    """
    # Extract the size of the regular grid
    gridsize_x_minus_1 = grid.shape[0] - 1
    gridsize_y_minus_1 = grid.shape[1] - 1
    gridsize_z_minus_1 = grid.shape[2] - 1
    # Correct for extreme values in the passed coordinates.
    # This is to catch inputs which are slighly larger than 1 due to
    # numerical errors.
    if x >= 1:
        x = 1 - ℝ[2*machine_ϵ]
    if y >= 1:
        y = 1 - ℝ[2*machine_ϵ]
    if z >= 1:
        z = 1 - ℝ[2*machine_ϵ]
    # Scale the coordinates so that 0 <= x, y, z < (gridsize - 1)
    x *= gridsize_x_minus_1
    y *= gridsize_y_minus_1
    z *= gridsize_z_minus_1
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
        vector[dim] = ( grid[x_lower, y_lower, z_lower, dim]*Wxl*Wyl*Wzl
                      + grid[x_lower, y_lower, z_upper, dim]*Wxl*Wyl*Wzu
                      + grid[x_lower, y_upper, z_lower, dim]*Wxl*Wyu*Wzl
                      + grid[x_lower, y_upper, z_upper, dim]*Wxl*Wyu*Wzu
                      + grid[x_upper, y_lower, z_lower, dim]*Wxu*Wyl*Wzl
                      + grid[x_upper, y_lower, z_upper, dim]*Wxu*Wyl*Wzu
                      + grid[x_upper, y_upper, z_lower, dim]*Wxu*Wyu*Wzl
                      + grid[x_upper, y_upper, z_upper, dim]*Wxu*Wyu*Wzu)
    return vector

# Function for CIC-interpolating particle coordinates
# to a cubic grid with scalar values.
@cython.header(# Argument
               particles='Particles',
               grid='double[:, :, ::1]',
               # Locals
               posx='double*',
               posy='double*',
               posz='double*',
               gridsize_i_minus_1='int',
               gridsize_j_minus_1='int',
               gridsize_k_minus_1='int',
               gridsize_i_minus_1_over_domain_size_x='double',
               gridsize_j_minus_1_over_domain_size_y='double',
               gridsize_k_minus_1_over_domain_size_z='double',
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
               )
def CIC_particles2grid(particles, grid):
    """This function CIC-interpolates particle coordinates
    to grid storing scalar values. The passed grid should be
    nullified beforehand.
    """
    # Extract variables
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    # Extract the size of the regular grid
    gridsize_i_minus_1 = grid.shape[0] - 1
    gridsize_j_minus_1 = grid.shape[1] - 1
    gridsize_k_minus_1 = grid.shape[2] - 1
    # The conversion factors between comoving length and grid units
    gridsize_i_minus_1_over_domain_size_x = gridsize_i_minus_1/domain_size_x
    gridsize_j_minus_1_over_domain_size_y = gridsize_j_minus_1/domain_size_y
    gridsize_k_minus_1_over_domain_size_z = gridsize_k_minus_1/domain_size_z
    # Interpolate each particle
    for i in range(particles.N_local):
        # Get, translate and scale the coordinates so that
        # 0 <= i < gridsize_i - 1 for i in (x, y, z).
        x = (posx[i] - domain_start_x)*gridsize_i_minus_1_over_domain_size_x
        y = (posy[i] - domain_start_y)*gridsize_j_minus_1_over_domain_size_y
        z = (posz[i] - domain_start_z)*gridsize_k_minus_1_over_domain_size_z
        # Correct for coordinates which are
        # exactly at an upper domain boundary.
        if x == gridsize_i_minus_1:
            x -= ℝ[2*machine_ϵ]
        if y == gridsize_j_minus_1:
            y -= ℝ[2*machine_ϵ]
        if z == gridsize_k_minus_1:
            z -= ℝ[2*machine_ϵ]
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
        grid[x_lower, y_lower, z_lower] += Wxl*Wyl*Wzl
        grid[x_lower, y_lower, z_upper] += Wxl*Wyl*Wzu
        grid[x_lower, y_upper, z_lower] += Wxl*Wyu*Wzl
        grid[x_lower, y_upper, z_upper] += Wxl*Wyu*Wzu
        grid[x_upper, y_lower, z_lower] += Wxu*Wyl*Wzl
        grid[x_upper, y_lower, z_upper] += Wxu*Wyl*Wzu
        grid[x_upper, y_upper, z_lower] += Wxu*Wyu*Wzl
        grid[x_upper, y_upper, z_upper] += Wxu*Wyu*Wzu


# Function for communicating boundary values of the
# domain grid between processes.
@cython.header(# Arguments
               mode='int',
               # Locals
               i='int',
               j='int',
               k='int',
               grid_slice_backward='double[:, ::1]',
               grid_slice_backwarddown='double[:]',
               grid_slice_down='double[:, :]',
               grid_slice_forward='double[:, ::1]',
               grid_slice_forwardup='double[:]',
               grid_slice_left='double[:, ::1]',
               grid_slice_leftbackward='double[::1]',
               grid_slice_leftdown='double[:]',
               grid_slice_right='double[:, ::1]',
               grid_slice_rightup='double[:]',
               grid_slice_rightforward='double[::1]',
               grid_slice_up='double[:, :]',
               )
def communicate_domain_boundaries(mode=0):
    """The domain_grid_noghosts is here referred to as "the grid".
    This function can operate in either mode 0 or mode 1.
    Mode 0: The upper three faces (right, forward, up) of the grid as
    well as the upper three edges (right forward, right upward, forward
    upward) and the right, forward, upward point is communicated to the
    processes where these correspond to the lower faces/edges/point.
    The received values are added to the existing lower
    faces/edges/point. Mode 1: The lower three faces (left, backward,
    down) of the grid as well as the lower three edges (left backward,
    left downward, backward downward) and the left, backward, downward
    point is communicated to the processes where these correspond to the
    upper faces/edges/point. The received values replace the existing
    upper faces/edges/point.
    """
    global domain_grid_noghosts
    global sendbuf_faceij, sendbuf_faceik, sendbuf_facejk
    global recvbuf_faceij, recvbuf_faceik, recvbuf_facejk
    global sendbuf_edge, recvbuf_edge
    # 2D slices (contiguous and noncontiguous) of the domain grid
    grid_slice_right = domain_grid_noghosts[domain_size_i, :, :]
    grid_slice_left = domain_grid_noghosts[0, :, :]
    grid_slice_forward = domain_grid_noghosts[:, domain_size_j, :]
    grid_slice_backward = domain_grid_noghosts[:, 0, :]
    grid_slice_up = domain_grid_noghosts[:, :, domain_size_k]
    grid_slice_down = domain_grid_noghosts[:, :, 0]
    # 1D slices (contiguous and noncontiguous) of the domain grid
    grid_slice_rightforward = domain_grid_noghosts[domain_size_i, domain_size_j, :]
    grid_slice_leftbackward = domain_grid_noghosts[0, 0, :]
    grid_slice_rightup = domain_grid_noghosts[domain_size_i, :, domain_size_k]
    grid_slice_leftdown = domain_grid_noghosts[0, :, 0]
    grid_slice_forwardup = domain_grid_noghosts[:, domain_size_j, domain_size_k]
    grid_slice_backwarddown = domain_grid_noghosts[:, 0, 0]
    # If mode == 0, communicate the upper faces/edges/point to the
    # corresponding processes. Add the received data to the existing
    # lower values.
    if mode == 0:
        # Cummunicate the right face
        for j in range(domain_size_j):
            for k in range(domain_size_k):
                sendbuf_facejk[j, k] = grid_slice_right[j, k]
        Sendrecv(sendbuf_facejk, dest=rank_right, recvbuf=recvbuf_facejk,
                 source=rank_left)
        # Add the received contribution to the left face
        for j in range(domain_size_j):
            for k in range(domain_size_k):
                grid_slice_left[j, k] += recvbuf_facejk[j, k]
        # Cummunicate the forward face
        for i in range(domain_size_i):
            for k in range(domain_size_k):
                sendbuf_faceik[i, k] = grid_slice_forward[i, k]
        Sendrecv(sendbuf_faceik, dest=rank_forward, recvbuf=recvbuf_faceik,
                 source=rank_backward)
        # Add the received contribution to the backward face
        for i in range(domain_size_i):
            for k in range(domain_size_k):
                grid_slice_backward[i, k] += recvbuf_faceik[i, k]
        # Cummunicate the upward face
        for i in range(domain_size_i):
            for j in range(domain_size_j):
                sendbuf_faceij[i, j] = grid_slice_up[i, j]
        Sendrecv(sendbuf_faceij, dest=rank_up, recvbuf=recvbuf_faceij,
                 source=rank_down)
        # Add the received contribution to the lower face
        for i in range(domain_size_i):
            for j in range(domain_size_j):
                grid_slice_down[i, j] += recvbuf_faceij[i, j]
        # Communicate the right, forward edge
        for k in range(domain_size_k):
            sendbuf_edge[k] = grid_slice_rightforward[k]
        Sendrecv(sendbuf_edge[:domain_size_k], dest=rank_rightforward,
                 recvbuf=recvbuf_edge, source=rank_leftbackward)
        # Add the received contribution to the left, backward edge
        for k in range(domain_size_k):
            grid_slice_leftbackward[k] += recvbuf_edge[k]
        # Communicate the right, upward edge
        for j in range(domain_size_j):
            sendbuf_edge[j] = grid_slice_rightup[j]
        Sendrecv(sendbuf_edge[:domain_size_j], dest=rank_rightup,
                 recvbuf=recvbuf_edge, source=rank_leftdown)
        # Add the received contribution to the left, downward edge
        for j in range(domain_size_j):
            grid_slice_leftdown[j] += recvbuf_edge[j]
        # Communicate the forward, upward edge
        for i in range(domain_size_i):
            sendbuf_edge[i] = grid_slice_forwardup[i]
        Sendrecv(sendbuf_edge[:domain_size_i], dest=rank_forwardup,
                 recvbuf=recvbuf_edge, source=rank_backwarddown)
        # Add the received contribution to the backward, downward edge
        for i in range(domain_size_i):
            grid_slice_backwarddown[i] += recvbuf_edge[i]
        # Communicate the right, forward, upward point
        domain_grid_noghosts[0, 0, 0] += sendrecv(domain_grid_noghosts[domain_size_i,
                                                                       domain_size_j,
                                                                       domain_size_k],
                                                  dest=rank_rightforwardup,
                                                  source=rank_leftbackwarddown)
    # If mode == 1, communicate the lower faces/edges/point to the
    # corresponding processes. Replace the existing upper values with
    # the received data.
    elif mode == 1:
        # Cummunicate the left face
        for j in range(domain_size_j):
            for k in range(domain_size_k):
                sendbuf_facejk[j, k] = grid_slice_left[j, k]
        Sendrecv(sendbuf_facejk, dest=rank_left, recvbuf=recvbuf_facejk,
                 source=rank_right)
        # Copy the received contribution to the right face
        for j in range(domain_size_j):
            for k in range(domain_size_k):
                grid_slice_right[j, k] = recvbuf_facejk[j, k]
        # Cummunicate the backward face
        for i in range(domain_size_i):
            for k in range(domain_size_k):
                sendbuf_faceik[i, k] = grid_slice_backward[i, k]
        Sendrecv(sendbuf_faceik, dest=rank_backward, recvbuf=recvbuf_faceik,
                 source=rank_forward)
        # Copy the received contribution to the forward face
        for i in range(domain_size_i):
            for k in range(domain_size_k):
                grid_slice_forward[i, k] = recvbuf_faceik[i, k]
        # Cummunicate the downward face
        for i in range(domain_size_i):
            for j in range(domain_size_j):
                sendbuf_faceij[i, j] = grid_slice_down[i, j]
        Sendrecv(sendbuf_faceij, dest=rank_down, recvbuf=recvbuf_faceij,
                 source=rank_up)
        # Copy the received contribution to the upper face
        for i in range(domain_size_i):
            for j in range(domain_size_j):
                grid_slice_up[i, j] = recvbuf_faceij[i, j]
        # Communicate the left, backward edge
        for k in range(domain_size_k):
            sendbuf_edge[k] = grid_slice_leftbackward[k]
        Sendrecv(sendbuf_edge[:domain_size_k], dest=rank_leftbackward,
                 recvbuf=recvbuf_edge, source=rank_rightforward)
        # Copy the received contribution to the right, forward edge
        for k in range(domain_size_k):
            grid_slice_rightforward[k] = recvbuf_edge[k]
        # Communicate the left, downward edge
        for j in range(domain_size_j):
            sendbuf_edge[j] = grid_slice_leftdown[j]
        Sendrecv(sendbuf_edge[:domain_size_j], dest=rank_leftdown,
                 recvbuf=recvbuf_edge, source=rank_rightup)
        # Copy the received contribution to the right, upward edge
        for j in range(domain_size_j):
            grid_slice_rightup[j] = recvbuf_edge[j]
        # Communicate the backward, downward edge
        for i in range(domain_size_i):
            sendbuf_edge[i] = grid_slice_backwarddown[i]
        Sendrecv(sendbuf_edge[:domain_size_i], dest=rank_backwarddown,
                 recvbuf=recvbuf_edge, source=rank_forwardup)
        # Copy the received contribution to the forward, upward edge
        for i in range(domain_size_i):
            grid_slice_forwardup[i] = recvbuf_edge[i]
        # Communicate the left, backward, downward point
        domain_grid_noghosts[domain_size_i,
                             domain_size_j,
                             domain_size_k] = sendrecv(domain_grid_noghosts[0, 0, 0],
                                                       dest=rank_leftbackwarddown,
                                                       source=rank_rightforwardup)


# Function for communicating ghost layers of the
# domain grid between processes.
@cython.header(# Locals
               ghost_backward='double[:, :, ::1]',
               ghost_down='double[:, :, ::1]',
               ghost_forward='double[:, :, ::1]',
               ghost_left='double[:, :, ::1]',
               ghost_right='double[:, :, ::1]',
               ghost_up='double[:, :, ::1]',
               layer_backward='double[:, :, ::1]',
               layer_down='double[:, :, ::1]',
               layer_forward='double[:, :, ::1]',
               layer_left='double[:, :, ::1]',
               layer_right='double[:, :, ::1]',
               layer_up='double[:, :, ::1]',
               )
def communicate_domain_ghosts():
    """The ghost layers are two gridpoints in thickness.
    """
    global domain_grid, domain_grid_noghosts
    global sendbuf_ghostij, sendbuf_ghostik, sendbuf_ghostjk
    global recvbuf_ghostij, recvbuf_ghostik, recvbuf_ghostjk

    # The boundary layers (faces of thickness 2) which should
    # be send to other processes and used as ghost layers.
    layer_right = domain_grid_noghosts[(domain_grid_noghosts.shape[0] - 3):
                                       (domain_grid_noghosts.shape[0] - 1),
                                       :,
                                       :]
    layer_left = domain_grid_noghosts[1:3, :, :]
    layer_forward = domain_grid_noghosts[:, (domain_grid_noghosts.shape[1] - 3):
                                            (domain_grid_noghosts.shape[1] - 1),
                                            :]
    layer_backward = domain_grid_noghosts[:, 1:3, :]
    layer_up = domain_grid_noghosts[:, :, (domain_grid_noghosts.shape[2] - 3):
                                          (domain_grid_noghosts.shape[2] - 1)]
    layer_down = domain_grid_noghosts[:, :, 1:3]
    # Ghost layers of the local domain grid
    ghost_right = domain_grid[(domain_grid.shape[0] - 2):,
                              2:(domain_grid.shape[1] - 2),
                              2:(domain_grid.shape[2] - 2)]
    ghost_left = domain_grid[:2,
                             2:(domain_grid.shape[1] - 2),
                             2:(domain_grid.shape[2] - 2)]
    ghost_forward = domain_grid[2:(domain_grid.shape[0] - 2),
                                (domain_grid.shape[1] - 2):,
                                2:(domain_grid.shape[2] - 2)]
    ghost_backward = domain_grid[2:(domain_grid.shape[0] - 2),
                                 :2,
                                 2:(domain_grid.shape[2] - 2)]
    ghost_up = domain_grid[2:(domain_grid.shape[0] - 2),
                           2:(domain_grid.shape[1] - 2),
                           (domain_grid.shape[2] - 2):]
    ghost_down = domain_grid[2:(domain_grid.shape[0] - 2),
                             2:(domain_grid.shape[1] - 2),
                             :2]
    # Cummunicate the right boundary layer
    sendbuf_ghostjk[...] = layer_right[...]
    Sendrecv(sendbuf_ghostjk, dest=rank_right, recvbuf=recvbuf_ghostjk, source=rank_left)
    ghost_left[...] = recvbuf_ghostjk[...]
    # Cummunicate the left boundary layer
    sendbuf_ghostjk[...] = layer_left[...]
    Sendrecv(sendbuf_ghostjk, dest=rank_left, recvbuf=recvbuf_ghostjk, source=rank_right)
    ghost_right[...] = recvbuf_ghostjk[...]
    # Cummunicate the forward boundary layer
    sendbuf_ghostik[...] = layer_forward[...]
    Sendrecv(sendbuf_ghostik, dest=rank_forward, recvbuf=recvbuf_ghostik, source=rank_backward)
    ghost_backward[...] = recvbuf_ghostik[...]
    # Cummunicate the backward boundary layer
    sendbuf_ghostik[...] = layer_backward[...]
    Sendrecv(sendbuf_ghostik, dest=rank_backward, recvbuf=recvbuf_ghostik, source=rank_forward)
    ghost_forward[...] = recvbuf_ghostik[...]
    # Cummunicate the upward boundary layer
    sendbuf_ghostij[...] = layer_up[...]
    Sendrecv(sendbuf_ghostij, dest=rank_up, recvbuf=recvbuf_ghostij, source=rank_down)
    ghost_down[...] = recvbuf_ghostij[...]
    # Cummunicate the downward boundary layer
    sendbuf_ghostij[...] = layer_down[...]
    Sendrecv(sendbuf_ghostij, dest=rank_down, recvbuf=recvbuf_ghostij, source=rank_up)
    ghost_up[...] = recvbuf_ghostij[...]

# Function for transfering the interpolated data
# in the domain grid to the PM grid.
@cython.header(# Locals
               ID_send='int',
               ID_recv='int',
               i='int',
               j='int',
               k='int',
               ℓ='int',
               )
def domain2PM():
    global PM_grid, domainPM_sendbuf, domainPM_recvbuf
    # Communicate the interpolated domain grid to the PM grid
    for ℓ in range(ℓmax):
        # Send part of the local domain
        # grid to the corresponding process.
        if ℓ < PM_send_rank.shape[0]:
            ID_send = PM_send_rank[ℓ]
            for i in range(PM_send_i_start[ℓ], PM_send_i_end[ℓ]):
                for j in range(domain_size_j):
                    for k in range(domain_size_k):
                        domainPM_sendbuf[i - PM_send_i_start[ℓ],
                                         j,
                                         k] = domain_grid_noghosts[i, j, k]
            # A non-blocking send is used. Otherwise the
            # program will hang on large messages.
            Isend(domainPM_sendbuf, dest=ID_send)
        # The lower ranks storing the PM mesh receives the message
        if ℓ < PM_recv_rank.shape[0]:
            ID_recv = PM_recv_rank[ℓ]
            Recv(domainPM_recvbuf, source=ID_recv)
            for i in range(PM_recv_i_start[ℓ], PM_recv_i_end[ℓ]):
                for j in range(PM_recv_j_start[ℓ], PM_recv_j_end[ℓ]):
                    for k in range(PM_recv_k_start[ℓ], PM_recv_k_end[ℓ]):
                        PM_grid[i, j, k] = domainPM_recvbuf[i,
                                                            j - PM_recv_j_start[ℓ],
                                                            k - PM_recv_k_start[ℓ]]
        # Catch-up point for the processes. This ensures
        # that the communication is complete, and hence that
        # the non-blocking send is done.
        Barrier()


# Function for transfering the data in the PM grid to the domain grid
@cython.header(# Locals
               ID_send='int',
               ID_recv='int',
               i='int',
               j='int',
               k='int',
               ℓ='int',
               )
def PM2domain():
    global domain_grid_noghosts, domainPM_sendbuf, domainPM_recvbuf
    # Communicate the interpolated domain grid to the PM grid
    for ℓ in range(ℓmax):
        # The lower ranks storing the PM mesh sends part of their slab
        if ℓ < PM_recv_rank.shape[0]:
            ID_send = PM_recv_rank[ℓ]
            for i in range(PM_recv_i_start[ℓ], PM_recv_i_end[ℓ]):
                for j in range(PM_recv_j_start[ℓ], PM_recv_j_end[ℓ]):
                    for k in range(PM_recv_k_start[ℓ], PM_recv_k_end[ℓ]):
                        domainPM_recvbuf[i,
                                         j - PM_recv_j_start[ℓ],
                                         k - PM_recv_k_start[ℓ],
                                         ] = PM_grid[i, j, k]
            # A non-blocking send is used. Otherwise the program will
            # hang on large messages.
            Isend(domainPM_recvbuf, dest=ID_send)
        # The corresponding process receives the message
        if ℓ < PM_send_rank.shape[0]:
            ID_recv = PM_send_rank[ℓ]
            Recv(domainPM_sendbuf, source=ID_recv)
            for i in range(PM_send_i_start[ℓ], PM_send_i_end[ℓ]):
                for j in range(domain_size_j):
                    for k in range(domain_size_k):
                        domain_grid_noghosts[i, j, k] = domainPM_sendbuf[i - PM_send_i_start[ℓ],
                                                                         j,
                                                                         k]
        # Catch-up point for the processes. This ensures that the communication
        # is complete, and hence that the non-blocking send is done.
        Barrier()
    # The upper boundaries (not the ghost layers) of the domain grid
    # should be a copy of the lower boundaries of the next domain.
    # Do the needed communication.
    communicate_domain_boundaries(mode=1)
    # Communicate the ghost layers of the domain grid
    communicate_domain_ghosts()

# Function for CIC interpolating particles to the PM mesh
@cython.header(# Arguments
               particles='Particles',
               )
def PM_CIC(particles):
    global PM_grid, domain_grid, domain_grid_noghosts
    # Nullify the PM mesh and the domain grid
    PM_grid[...] = 0
    domain_grid[...] = 0
    # Interpolate particle coordinates to the domain grid
    # (without the ghost layers).
    CIC_particles2grid(particles, domain_grid_noghosts)
    # Values of local pseudo mesh points contribute to the lower mesh
    # points of domain_grid on other processes.
    # Do the needed communication.
    communicate_domain_boundaries(mode=0)
    # Communicate the interpolated data
    # in the domain grid into the PM grid.
    domain2PM()

# Function performing a forward Fourier transformation of the PM mesh
@cython.header()
def PM_FFT():
    # Fourier transform the PM grid forwards from real to Fourier space
    fftw_execute(plan_forward)

# Function performing a backward Fourier transformation of the PM mesh
@cython.header()
def PM_IFFT():
    # Fourier transform the PM grid backwards from Fourier to real
    # space. Note that this is an unnormalized transform, as defined by
    # FFTW. To do the normalization, devide all elements of PM_grid
    # by PM_gridsize**3.
    fftw_execute(plan_backward)



# Initializes PM_grid and related stuff at import time,
# if the PM_grid should be used.
cython.declare(fftw_struct='fftw_return_struct',
               fftw_struct_grid='double*',
               PM_gridsize_local_i='ptrdiff_t',
               PM_gridsize_local_j='ptrdiff_t',
               PM_gridstart_local_i='ptrdiff_t',
               PM_gridstart_local_j='ptrdiff_t',
               PM_grid='double[:, :, ::1]',
               plan_forward='fftw_plan',
               plan_backward='fftw_plan',
               )
if use_PM:
    # The PM mesh and functions on it
    if not cython.compiled:
        # Initialization of the PM mesh in pure Python.
        PM_gridsize_local_i = PM_gridsize_local_j = int(PM_gridsize/nprocs)
        if master and PM_gridsize_local_i != PM_gridsize/nprocs:
            # If PM_gridsize is not divisible by nprocs, the code cannot
            # figure out exactly how FFTW distribute the grid among the
            # processes. In stead of guessing, do not even try to
            # emulate the behaviour of FFTW.
            msg = ('The PM method in pure Python mode only works '
                   + 'when\nPM_gridsize is divisible by the number '
                   + 'of processes!')
            abort(msg)
        PM_gridstart_local_i = PM_gridstart_local_j = PM_gridsize_local_i*rank
        PM_grid = empty((PM_gridsize_local_i, PM_gridsize,
                         PM_gridsize_padding), dtype=C2np['double'])
        # The output of the following function is formatted just
        # like that of the MPI implementation of FFTW.
        plan_backward = 'plan_backward'
        plan_forward = 'plan_forward'
        def fftw_execute(plan):
            global PM_grid
            # The pure Python FFT implementation is serial.
            # Every process computes the entire FFT of the temporary
            # varaible PM_grid_global.
            PM_grid_global = empty((PM_gridsize, PM_gridsize,
                                    PM_gridsize_padding))
            Allgatherv(PM_grid, PM_grid_global)
            if plan == plan_forward:
                # Delete the padding on last dimension
                for i in range(PM_gridsize_padding - PM_gridsize):
                    PM_grid_global = delete(PM_grid_global, -1, axis=2)
                # Do real transform
                PM_grid_global = rfftn(PM_grid_global)
                # FFTW transposes the first two dimensions
                PM_grid_global = PM_grid_global.transpose([1, 0, 2])
                # FFTW represents the complex array by doubles only
                tmp = empty((PM_gridsize, PM_gridsize, PM_gridsize_padding))
                for i in range(PM_gridsize_padding):
                    if i % 2:
                        tmp[:, :, i] = PM_grid_global.imag[:, :, i//2]
                    else:
                        tmp[:, :, i] = PM_grid_global.real[:, :, i//2]
                PM_grid_global = tmp
                # As in FFTW, distribute the slabs along the y-dimension
                # (which is the first dimension now, due to transposing)
                PM_grid[...] = PM_grid_global[PM_gridstart_local_j:(PM_gridstart_local_j
                                                                    + PM_gridsize_local_j),
                                              :,
                                              :]
            elif plan == plan_backward:
                # FFTW represents the complex array by doubles only.
                # Go back to using complex entries
                tmp = zeros((PM_gridsize, PM_gridsize, PM_gridsize_padding/2),
                            dtype='complex128')
                for i in range(PM_gridsize_padding):
                    if i % 2:
                        tmp[:, :, i//2] += 1j*PM_grid_global[:, :, i]
                    else:
                        tmp[:, :, i//2] += PM_grid_global[:, :, i]
                PM_grid_global = tmp
                # FFTW transposes the first
                # two dimensions back to normal.
                PM_grid_global = PM_grid_global.transpose([1, 0, 2])
                # Do real inverse transform
                PM_grid_global = irfftn(PM_grid_global, s=[PM_gridsize]*3)
                # Remove the autoscaling provided by Numpy
                PM_grid_global[...] *= PM_gridsize3
                # Add padding on last dimension, as in FFTW
                padding = empty((PM_gridsize,
                                 PM_gridsize,
                                 PM_gridsize_padding - PM_gridsize,
                                 ))
                PM_grid_global = concatenate((PM_grid_global, padding), axis=2)
                # As in FFTW, distribute the slabs along the x-dimension
                PM_grid[...] = PM_grid_global[PM_gridstart_local_i:(PM_gridstart_local_i
                                                                    + PM_gridsize_local_i),
                                              :,
                                              :]
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
                               .format(PM_gridsize, nprocs, fftw_rigor))
            if os.path.isfile(wisdom_filename):
                break
        # Initialize fftw_mpi, allocate the grid, initialize the
        # local grid sizes and start indices and do FFTW planning.
        if not os.path.isfile(wisdom_filename):
            msg = ('Acquiring FFTW wisdom ({}) for grid of linear size {} on '
                   + '{} {} ...').format(fftw_rigor,
                                         PM_gridsize,
                                         nprocs,
                                         'processes' if nprocs > 1
                                                     else 'process')
            masterprint(msg)
            fftw_struct = fftw_setup(PM_gridsize,
                                     PM_gridsize,
                                     PM_gridsize,
                                     bytes(fftw_rigor, encoding='ascii'))
            masterprint('done')
        else:
            fftw_struct = fftw_setup(PM_gridsize,
                                     PM_gridsize,
                                     PM_gridsize,
                                     bytes(fftw_rigor, encoding='ascii'))
        # If less rigouros wisdom exist for the same problem, delete it
        for rigor in fftw_rigors[(fftw_rigors.index(fftw_rigor) + 1):]:
            wisdom_filename = ('.fftw_wisdom_gridsize={}_nprocs={}_rigor={}'
                               .format(PM_gridsize, nprocs, rigor))
            if master and os.path.isfile(wisdom_filename):
                os.remove(wisdom_filename)
        # Unpack fftw_struct
        PM_gridsize_local_i = fftw_struct.gridsize_local_i
        PM_gridsize_local_j = fftw_struct.gridsize_local_j
        PM_gridstart_local_i = fftw_struct.gridstart_local_i
        PM_gridstart_local_j = fftw_struct.gridstart_local_j
        # Wrap a memoryview around the grid. Loop as noted in fft.c, but
        # use PM_grid[i, j, k] when in real space and PM_grid[j, i, k]
        # when in Fourier space
        fftw_struct_grid = fftw_struct.grid
        if PM_gridsize_local_i > 0:
            PM_grid = cast(fftw_struct.grid,
                           'double[:PM_gridsize_local_i, :PM_gridsize, :PM_gridsize_padding]')
        else:
            # The process do not participate in the FFT computations
            PM_grid = empty((0, PM_gridsize, PM_gridsize_padding))
        plan_forward  = fftw_struct.plan_forward
        plan_backward = fftw_struct.plan_backward
else:
    # As these should be importable,
    # they need to be assigned even if not used.
    PM_grid = empty((1, 1, 1), dtype=C2np['double'])
    PM_gridsize_local_i = 1
    PM_gridsize_local_j = 1
    PM_gridstart_local_i = 0
    PM_gridstart_local_j = 0

# Information about the domain grid used in the
# communicate_domain_grid and domain2PM functions.
# Declarations for the communicate_domain_grid function.
cython.declare(domain_cuts='int[::1]',
               domain_layout='int[:, :, ::1]',
               domain_local='int[::1]',
               domain_size_x='double',
               domain_size_y='double',
               domain_size_z='double',
               domain_start_x='double',
               domain_start_y='double',
               domain_start_z='double',
               domain_end_x='double',
               domain_end_y='double',
               domain_end_z='double',
               domain_size_i='int',
               domain_size_j='int',
               domain_size_k='int',
               rank_right='int',
               rank_left='int',
               rank_forward='int',
               rank_backward='int',
               rank_up='int',
               rank_down='int',
               rank_rightforward='int',
               rank_leftbackward='int',
               rank_rightup='int',
               rank_leftdown='int',
               rank_forwardup='int',
               rank_backwarddown='int',
               rank_rightforwardup='int',
               rank_leftbackwarddown='int',
               recvbuf_edge='double[::1]',
               recvbuf_faceij='double[:, ::1]',
               recvbuf_faceik='double[:, ::1]',
               recvbuf_facejk='double[:, ::1]',
               sendbuf_edge='double[::1]',
               sendbuf_faceij='double[:, ::1]',
               sendbuf_faceik='double[:, ::1]',
               sendbuf_facejk='double[:, ::1]',
               )
# Declarations for the domain2PM function
cython.declare(ID_recv='int',
               ID_send='int',
               PM_gridsize_global_i='int',
               PM_send_i_end='int[::1]',
               PM_send_i_end_list='list',
               PM_send_i_start='int[::1]',
               PM_send_i_start_list='list',
               PM_send_rank='int[::1]',
               PM_send_rank_list='list',
               PM_recv_i_start='int[::1]',
               PM_recv_i_start_list='list',
               PM_recv_j_start='int[::1]',
               PM_recv_j_start_list='list',
               PM_recv_k_start='int[::1]',
               PM_recv_k_start_list='list',
               PM_recv_i_end='int[::1]',
               PM_recv_i_end_list='list',
               PM_recv_j_end='int[::1]',
               PM_recv_j_end_list='list',
               PM_recv_k_end='int[::1]',
               PM_recv_k_end_list='list',
               PM_recv_rank='int[::1]',
               PM_recv_rank_list='list',
               domain_start_i='int',
               domain_start_j='int',
               domain_start_k='int',
               domain_end_i='int',
               domain_end_j='int',
               domain_end_k='int',
               domainPM_sendbuf='double[:, :, ::1]',
               domainPM_recvbuf='double[:, :, ::1]',
               ℓ='int',
               ℓmax='int',
               )
if use_PM:
    # Number of domains in all three dimensions
    domain_cuts = np.array(cutout_domains(nprocs), dtype=C2np['int'])
    # The 3D layout of the division of the box
    domain_layout = arange(nprocs, dtype=C2np['int']).reshape(domain_cuts)
    # The indices in domain_layout of the local domain
    domain_local = np.array(np.unravel_index(rank, domain_cuts), dtype=C2np['int'])
    # The linear size of the domains, which are the same for all of them
    domain_size_x = boxsize/domain_cuts[0]
    domain_size_y = boxsize/domain_cuts[1]
    domain_size_z = boxsize/domain_cuts[2]
    # The start and end positions of the local domain
    domain_start_x = domain_local[0]*domain_size_x
    domain_start_y = domain_local[1]*domain_size_y
    domain_start_z = domain_local[2]*domain_size_z
    domain_end_x = domain_start_x + domain_size_x
    domain_end_y = domain_start_x + domain_size_x
    domain_end_z = domain_start_x + domain_size_x
    # Get the ranks of the 6 neighboring processes
    neighbors = neighboring_ranks()
    rank_right = neighbors['right']
    rank_left = neighbors['left']
    rank_forward = neighbors['forward']
    rank_backward = neighbors['backward']
    rank_up = neighbors['up']
    rank_down = neighbors['down']
    # Now get the ranks of the 6 diagonal neighboring processes
    rank_rightforward = neighbors['rightforward']
    rank_leftbackward = neighbors['leftbackward']
    rank_rightup = neighbors['rightup']
    rank_leftdown = neighbors['leftdown']
    rank_forwardup = neighbors['forwardup']
    rank_backwarddown = neighbors['backwarddown']
    # Finally get the ranks of the two 3D-diagonal neighboring processes
    rank_rightforwardup = neighbors['rightforwardup']
    rank_leftbackwarddown = neighbors['leftbackwarddown']
    # The actual size of the domain grid. This is 1 less than the
    # allocated size in each dimension, as the last element is actually
    # the first element of the domain on some other process.
    domain_size_i = PM_gridsize//domain_cuts[0]
    domain_size_j = PM_gridsize//domain_cuts[1]
    domain_size_k = PM_gridsize//domain_cuts[2]
    # Send/recieve buffers.
    # Separate buffers for each face is needed to ensure contiguousity. 
    sendbuf_faceij = empty((domain_size_i, domain_size_j), dtype=C2np['double'])
    recvbuf_faceij = empty((domain_size_i, domain_size_j), dtype=C2np['double'])
    sendbuf_faceik = empty((domain_size_i, domain_size_k), dtype=C2np['double'])
    recvbuf_faceik = empty((domain_size_i, domain_size_k), dtype=C2np['double'])
    sendbuf_facejk = empty((domain_size_j, domain_size_k), dtype=C2np['double'])
    recvbuf_facejk = empty((domain_size_j, domain_size_k), dtype=C2np['double'])
    sendbuf_edge = empty(np.max((domain_size_i, domain_size_j, domain_size_k)),
                         dtype=C2np['double'])
    recvbuf_edge = empty(np.max((domain_size_i, domain_size_j, domain_size_k)),
                         dtype=C2np['double'])

    # Additional information about the domain grid and the PM mesh,
    # used in the domain2PM function.
    # The global start and end indices of
    # the local domain in the total PM_grid.
    domain_start_i = domain_local[0]*domain_size_i
    domain_start_j = domain_local[1]*domain_size_j
    domain_start_k = domain_local[2]*domain_size_k
    domain_end_i = domain_start_i + domain_size_i
    domain_end_j = domain_start_j + domain_size_j
    domain_end_k = domain_start_k + domain_size_k
    # PM_gridsize_local_i is the same for all processes participating
    # in the PM algorithm and 0 otherwise. The global version is equal
    # to the nonzero value on all processes.
    PM_gridsize_local_i = PM_gridsize//nprocs
    if rank < PM_gridsize and PM_gridsize_local_i == 0:
        PM_gridsize_local_i = 1 
    PM_gridsize_global_i = PM_gridsize_local_i
    if PM_gridsize_global_i == 0:
        PM_gridsize_global_i = 1
    # Find local i-indices to send and to which process
    PM_send_i_start_list = []
    PM_send_i_end_list = []
    PM_send_rank_list = []
    for ℓ in range(domain_start_i, domain_end_i, PM_gridsize_global_i):
        PM_send_i_start_list.append(ℓ - domain_start_i)
        PM_send_i_end_list.append(ℓ - domain_start_i + PM_gridsize_global_i)
        PM_send_rank_list.append(ℓ//PM_gridsize_global_i)
    # Shift the elements so that they
    # match the communication pattern used.
    PM_send_i_start_list = list(np.roll(PM_send_i_start_list, -rank))
    PM_send_i_end_list = list(np.roll(PM_send_i_end_list, -rank))
    PM_send_rank_list = list(np.roll(PM_send_rank_list, -rank))
    #
    # FIXME: THIS IS NOT SUFFICIENT! IF nprocs > PM_grid THE PROGRAM WILL HALT AT domain2PM and PM2domain !!!!!!!!!!!!!!!!!!
    #
    # Communicate the start and end (j, k)-indices of the PM grid,
    # where future parts of the local domains should be received into.
    PM_recv_i_start_list = []
    PM_recv_j_start_list = []
    PM_recv_k_start_list = []
    PM_recv_i_end_list = []
    PM_recv_j_end_list = []
    PM_recv_k_end_list = []
    PM_recv_rank_list = []
    for ℓ in range(nprocs):
        # Process ranks to send/recieve to/from
        ID_send = mod(rank + ℓ, nprocs)
        ID_recv = mod(rank - ℓ, nprocs)
        # Send the global y and z start and end indices of the region
        # to be send, if anything should be send to process ID_send.
        # Otherwize send None.
        sendbuf = (domain_start_j,
                   domain_start_k,
                   domain_end_j,
                   domain_end_k) if ID_send in PM_send_rank_list else None
        recvbuf = sendrecv(sendbuf, dest=ID_send, source=ID_recv)
        if recvbuf is not None:
            PM_recv_i_start_list.append(0)
            PM_recv_i_end_list.append(PM_gridsize_local_i)
            PM_recv_j_start_list.append(recvbuf[0])
            PM_recv_k_start_list.append(recvbuf[1])
            PM_recv_j_end_list.append(recvbuf[2])
            PM_recv_k_end_list.append(recvbuf[3])
            PM_recv_rank_list.append(ID_recv)
    # Memoryview versions of the lists
    PM_send_i_start = np.array(PM_send_i_start_list, dtype=C2np['int'])
    PM_send_i_end = np.array(PM_send_i_end_list, dtype=C2np['int'])
    PM_send_rank = np.array(PM_send_rank_list, dtype=C2np['int'])
    PM_recv_i_start = np.array(PM_recv_i_start_list, dtype=C2np['int'])
    PM_recv_j_start = np.array(PM_recv_j_start_list, dtype=C2np['int'])
    PM_recv_k_start = np.array(PM_recv_k_start_list, dtype=C2np['int'])
    PM_recv_i_end = np.array(PM_recv_i_end_list, dtype=C2np['int'])
    PM_recv_j_end = np.array(PM_recv_j_end_list, dtype=C2np['int'])
    PM_recv_k_end = np.array(PM_recv_k_end_list, dtype=C2np['int'])
    PM_recv_rank = np.array(PM_recv_rank_list, dtype=C2np['int'])
    # Buffers
    domainPM_sendbuf = empty((PM_gridsize_global_i, domain_size_j, domain_size_k),
                             dtype=C2np['double'])
    if PM_recv_rank_list != []:
        domainPM_recvbuf = empty((PM_gridsize_global_i, domain_size_j, domain_size_k),
                                 dtype=C2np['double'])
    # ℓ will be the communication loop index.
    # It runs from 0 t0 ℓmax - 1.
    ℓmax = np.max([PM_send_rank.shape[0], PM_recv_rank.shape[0]])
    # Send/recieve buffers used in the communicate_domain_ghosts
    # function. Separate buffers for each face is needed to
    # ensure contiguousity.
    cython.declare(sendbuf_ghostij='double[:, :, ::1]',
                   recvbuf_ghostij='double[:, :, ::1]',
                   sendbuf_ghostik='double[:, :, ::1]',
                   recvbuf_ghostik='double[:, :, ::1]',
                   sendbuf_ghostjk='double[:, :, ::1]',
                   recvbuf_ghostjk='double[:, :, ::1]',
                   )
    sendbuf_ghostij = empty((domain_size_i + 1, domain_size_j + 1, 2), dtype=C2np['double'])
    recvbuf_ghostij = empty((domain_size_i + 1, domain_size_j + 1, 2), dtype=C2np['double'])
    sendbuf_ghostik = empty((domain_size_i + 1, 2, domain_size_k + 1), dtype=C2np['double'])
    recvbuf_ghostik = empty((domain_size_i + 1, 2, domain_size_k + 1), dtype=C2np['double'])
    sendbuf_ghostjk = empty((2, domain_size_j + 1, domain_size_k + 1), dtype=C2np['double'])
    recvbuf_ghostjk = empty((2, domain_size_j + 1, domain_size_k + 1), dtype=C2np['double'])
else:
    # As these should be importable,
    # they need to be assigned even if not used.
    domain_end_i = 1
    domain_end_j = 1
    domain_end_k = 1
    domain_end_x = 1
    domain_end_y = 1
    domain_end_z = 1
    domain_size_i = 1
    domain_size_j = 1
    domain_size_k = 1
    domain_size_x = 1
    domain_size_y = 1
    domain_size_z = 1
    domain_start_i = 0
    domain_start_k = 0
    domain_start_k = 0
    domain_start_x = 0
    domain_start_y = 0
    domain_start_z = 0

# Check if PM_grid is large enough for P3M to work, if the P3M
# algorithm is to be used.
if master and 'P3M' in kick_algorithms.values():
    if (   domain_size_i < P3M_scale*P3M_cutoff
        or domain_size_j < P3M_scale*P3M_cutoff
        or domain_size_k < P3M_scale*P3M_cutoff):
        msg = ('A PM_gridsize of ' + str(PM_gridsize) + ' and ' + str(nprocs) + ' processes '
               + 'results in following domain partition: ' + str(list(domain_cuts)) + '.\n'
               + 'The smallest domain width is ' + str(np.min([domain_size_i,
                                                               domain_size_j,
                                                               domain_size_k]))
               + ' grid cells, while the choice of P3M_scale (' + str(P3M_scale) + ') and '
               + 'P3M_cutoff (' + str(P3M_cutoff) + ')\nmeans that the domains must be at least '
               + str(int(np.ceil(P3M_scale*P3M_cutoff)))
               + ' grid cells for the P3M algorithm to work.'
               )
        abort(msg)
    if ((   domain_size_i < 2*P3M_scale*P3M_cutoff
         or domain_size_j < 2*P3M_scale*P3M_cutoff
         or domain_size_k < 2*P3M_scale*P3M_cutoff) and np.min(domain_cuts) < 3):
        # This is only allowed if domain_cuts are at least 3 in each
        # direction. Otherwise the left and the right (say) process
        # is the same, and the boundaries will be send to it twize,
        # and these will overlap with each other in the left/right
        # domain and gravity will be applied twize.
        msg = ('A PM_gridsize of ' + str(PM_gridsize) + ' and '
               + str(nprocs) + ' processes results in following domain'
               + ' partition: ' + str(list(domain_cuts))
               + '.\nThe smallest domain width is '
               + str(np.min([domain_size_i, domain_size_j,
                             domain_size_k]))
               + ' grid cells, while the choice of P3M_scale ('
               + str(P3M_scale) + ') and P3M_cutoff ('
               + str(P3M_cutoff) + ')\nmeans that the domains must be '
               + 'at least '
               + str(int(np.ceil(2*P3M_scale*P3M_cutoff))) + ' grid cells for the '
               + 'P3M algorithm to work.'
            )
        abort(msg)

# Initialize the domain grid if the PM method should be used
cython.declare(i='Py_ssize_t',
               domain_grid='double[:, :, ::1]',
               domain_grid_noghosts='double[:, :, ::1]',
               )
if use_PM:
    # A grid over the local domain. An additional layer of thickness 1
    # is given to the domain grid, so that these outer points
    # corresponds to the same physical coordinates as the first points
    # in the next domain. Also, an additional layer of thickness 2 is
    # given on top of the previous layer. This shall be used as a ghost
    # layer for finite differencing.
    domain_grid = zeros([PM_gridsize//domain_cuts[i] + 1 + 2*2 for i in range(3)],
                        dtype=C2np['double'])
    # Memoryview of the domain grid without the ghost layers
    domain_grid_noghosts = domain_grid[2:(domain_grid.shape[0] - 2),
                                       2:(domain_grid.shape[1] - 2),
                                       2:(domain_grid.shape[2] - 2)]
    # Test if the grid has been constructed correctly.
    # If not it is because nprocs and PM_gridsize are incompatible.
    for i in range(3):
        if not master:
            break
        domain_gridsize = domain_grid_noghosts.shape[i] - 1
        if PM_gridsize != domain_cuts[i]*domain_gridsize:
            msg = ('A PM_gridsize of {} cannot be equally shared among {} processes'
                   .format(PM_gridsize, nprocs))
            masterprint('domain_cuts[0] =', domain_cuts[0],
                'domain_grid_noghosts.shape[0] =', domain_grid_noghosts.shape[0])
            abort(msg)
        if domain_gridsize < 2:
            msg = ('A PM_gridsize of {} is too small for {} processes'
                   .format(PM_gridsize, nprocs))
            abort(msg)
else:
    # As these should be importable,
    # they need to be assigned even if not used.
    domain_grid = empty([5, 5, 5], dtype=C2np['double'])
    domain_grid_noghosts = domain_grid[2:(domain_grid.shape[0] - 2),
                                       2:(domain_grid.shape[1] - 2),
                                       2:(domain_grid.shape[2] - 2)]
