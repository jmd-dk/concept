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
@cython.cdivision(True)
@cython.boundscheck(False)
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
@cython.cdivision(True)
@cython.boundscheck(False)
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
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               grid='double[:, :, ::1]',
               particles='Particles',
               # Locals
               gridsize='int',
               gridsize_over_boxsize='double',
               i='size_t',
               posx='double*',
               posy='double*',
               posz='double*',
               x='double',
               x_lower='size_t',
               x_upper='size_t',
               xl='double',
               xu='double',
               y='double',
               y_lower='size_t',
               y_upper='size_t',
               yl='double',
               yu='double',
               z='double',
               z_lower='size_t',
               z_upper='size_t',
               zl='double',
               zu='double',
               )
def CIC_coordinates2grid(grid, particles):
    """ This function CIC-interpolates particle coordinates to a grid storing
    scalar values. Input arguments should not be normalized.
    """

    # Extract variables
    posx = particles.posx
    posy = particles.posy
    posz = particles.posz
    gridsize = grid.shape[0]
    gridsize_over_boxsize = gridsize/boxsize
    # Interpolate each particle
    for i in range(particles.N):
        # Get and scale the coordinates so that 0 <= x, y, z < gridsize
        x = posx[i]*gridsize_over_boxsize
        y = posy[i]*gridsize_over_boxsize
        z = posz[i]*gridsize_over_boxsize
        # Translate coordinates so they appear to be in the first domain,
        # to ensure appropriate indexing.
        #x -= gridstart_local_x

        # Indices of the 8 vertices (6 faces) in the grid,
        # constituting the cell in which the particle is located
        x_lower = int(x)
        y_lower = int(y)
        z_lower = int(z)
        x_upper = (x_lower + 1) % gridsize
        y_upper = (y_lower + 1) % gridsize
        z_upper = (z_lower + 1) % gridsize
        # The side length of the 8 regions
        xu = x - x_lower
        yu = y - y_lower
        zu = z - z_lower
        xl = 1 - xu
        yl = 1 - yu
        zl = 1 - zu
        # Assign the weighted grid values to the vector components
        grid[x_lower, y_lower, z_lower] += xl*yl*zl
        grid[x_lower, y_lower, z_upper] += xl*yl*zu
        grid[x_lower, y_upper, z_lower] += xl*yu*zl
        grid[x_lower, y_upper, z_upper] += xl*yu*zu
        grid[x_upper, y_lower, z_lower] += xu*yl*zl
        grid[x_upper, y_lower, z_upper] += xu*yl*zu
        grid[x_upper, y_upper, z_lower] += xu*yu*zl
        grid[x_upper, y_upper, z_upper] += xu*yu*zu
