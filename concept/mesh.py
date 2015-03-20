# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """


# Function for tabulating a cubic grid with vector values
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               gridsize='int',
               func='func_ddd_ddd',
               factor='double',
               filename='str',
               # Locals
               dim='size_t',
               grid='double[:, :, :, ::1]',
               grid_local='double[:, :, :, ::1]',
               i='size_t',
               i_end='size_t',
               i_start='size_t',
               j='size_t',
               j_end='size_t',
               j_start='size_t',
               k='size_t',
               k_end='size_t',
               k_start='size_t',
               shape='tuple',
               )
@cython.returns('double[:, :, :, ::1]')
def tabulate_vectorfield(gridsize, func, factor, filename):
    """ This function tabulates a cubic grid of size
    gridsize*gridsize*gridsize with vector values computed by the
    function func, as grid[i, j, k] = func(i*factor, j*factor, k*factor).
    The tabulated grid is saved to a hdf5 file named filename.
    """

    # Initialize the grid to be of shape gridsize*gridsize*gridsize*3.
    # That is, grid is not really cubic, but rather four-dimensional.
    shape = (gridsize, )*3 + (3, )
    grid = empty(shape)
    # Each process tabulate its part of the grid
    (i_start, j_start, k_start), (i_end, j_end, k_end) = partition(shape[:3])
    grid_local = empty([i_end - i_start,
                        j_end - j_start,
                        k_end - k_start] + [3])
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                # Compute the vector values via the passed function
                vector = func(i*factor, j*factor, k*factor)
                for dim in range(3):
                    grid_local[i - i_start,
                               j - j_start,
                               k - k_start, dim] = vector[dim]
    # Save grid to disk using parallel hdf5
    with h5py.File(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
        dset = hdf5_file.create_dataset('data', shape, dtype='float64')
        dset[i_start:i_end, j_start:j_end, k_start:k_end, :] = grid_local
    # Every process gets to know the entire grid
    Allgather(grid_local, grid)
    return grid


# Function for doing lookup in a cubic grid with vector values and
# CIC-interpolating to specified coordinates
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
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
               dim='size_t',
               gridsize_minus_1='int',
               x_lower='size_t',
               x_upper='size_t',
               y_lower='size_t',
               y_upper='size_t',
               z_lower='size_t',
               z_upper='size_t',
               )
@cython.returns('double*')
def CIC_grid2coordinates_vector(grid, x, y, z):
    """This function look up tabulated vectors in a cubic grid and interpolates 
    to (x, y, z) via the cloud in cell (CIC) method. Input arguments must be
    normalized so that 0 <= x, y, z < 1. If x, y or z is exactly equal to 1,
    they will be corrected to 1 - ϵ. It is assumed that the grid is
    nonperiodic.
    """

    # Extract the size of the regular, cubic grid
    gridsize_minus_1 = grid.shape[0] - 1
    # Correct for extreme values in the passed coordinates. This is to catch
    # inputs which are slighly larger than 1 due to numerical errors
    if x >= 1:
        x = 1 - two_machine_ϵ
    if y >= 1:
        y = 1 - two_machine_ϵ
    if z >= 1:
        z = 1 - two_machine_ϵ
    # Scale the coordinates so that 0 <= x, y, z < (gridsize - 1)
    x *= gridsize_minus_1
    y *= gridsize_minus_1
    z *= gridsize_minus_1
    # Indices of the 8 vertices (6 faces) of the grid surrounding (x, y, z)
    x_lower = int(x)
    y_lower = int(y)
    z_lower = int(z)
    x_upper = x_lower + 1
    y_upper = y_lower + 1
    z_upper = z_lower + 1
    # The linear weights according to the CIC rule W = 1 - |dist| if |dist| < 1
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
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               particles='Particles',
               grid='double[:, :, ::1]',
               domain_size_x='double',
               domain_size_y='double',
               domain_size_z='double',
               gridstart_x='double',
               gridstart_y='double',
               gridstart_z='double',
               # Locals
               posx='double*',
               posy='double*',
               posz='double*',
               gridsize_x='int',
               gridsize_y='int',
               gridsize_z='int',
               gridsize_x_minus_1='int',
               gridsize_y_minus_1='int',
               gridsize_z_minus_1='int',
               gridsize_x_minus_1_over_domain_size_x='double',
               gridsize_y_minus_1_over_domain_size_y='double',
               gridsize_z_minus_1_over_domain_size_z='double',
               i='size_t',
               j='size_t',
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
def CIC_particles2grid(particles, grid, domain_size_x,
                                        domain_size_y,
                                        domain_size_z,
                                        gridstart_x,
                                        gridstart_y,
                                        gridstart_z):
    """This function CIC-interpolates particle coordinates to grid storing
    scalar values. The passed grid should be nullified beforehand.
    """
    # Extract variables
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    gridsize_x = grid.shape[0]
    gridsize_y = grid.shape[1]
    gridsize_z = grid.shape[2]
    # The conversion factors between comoving length and grid units
    gridsize_x_minus_1 = gridsize_x - 1
    gridsize_y_minus_1 = gridsize_y - 1
    gridsize_z_minus_1 = gridsize_z - 1
    gridsize_x_minus_1_over_domain_size_x = gridsize_x_minus_1/domain_size_x
    gridsize_y_minus_1_over_domain_size_y = gridsize_y_minus_1/domain_size_y
    gridsize_z_minus_1_over_domain_size_z = gridsize_z_minus_1/domain_size_z
    # Interpolate each particle
    for j in range(100000):
        for i in range(particles.N_local):
            # Get, translate and scale the coordinates so that
            # 0 <= i < gridsize_i - 1 for i = {x, y, z}.
            x = (posx[i] - gridstart_x)*gridsize_x_minus_1_over_domain_size_x
            y = (posy[i] - gridstart_y)*gridsize_y_minus_1_over_domain_size_y
            z = (posz[i] - gridstart_z)*gridsize_z_minus_1_over_domain_size_z
            # Correct for coordinates which are exactly at an upper domain boundary
            if x == gridsize_x_minus_1:
                x -= two_machine_ϵ
            if y == gridsize_y_minus_1:
                y -= two_machine_ϵ
            if z == gridsize_z_minus_1:
                z -= two_machine_ϵ
            # Indices of the 8 vertices (6 faces) of the grid surrounding (x, y, z)
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
            grid[x_lower, y_lower, z_lower] += Wxl*Wyl*Wzl
            grid[x_lower, y_lower, z_upper] += Wxl*Wyl*Wzu
            grid[x_lower, y_upper, z_lower] += Wxl*Wyu*Wzl
            grid[x_lower, y_upper, z_upper] += Wxl*Wyu*Wzu
            grid[x_upper, y_lower, z_lower] += Wxu*Wyl*Wzl
            grid[x_upper, y_lower, z_upper] += Wxu*Wyl*Wzu
            grid[x_upper, y_upper, z_lower] += Wxu*Wyu*Wzl
            grid[x_upper, y_upper, z_upper] += Wxu*Wyu*Wzu
