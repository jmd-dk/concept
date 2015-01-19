# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
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
               func='vector_func',
               factor='double',
               filename='str',
               # Locals
               shape='tuple',
               i_start='size_t',
               i_end='size_t',
               i='size_t',
               j_start='size_t',
               j_end='size_t',
               j='size_t',
               k_start='size_t',
               k_end='size_t',
               k='size_t',
               dim='size_t',
               grid_local='double[:, :, :, ::1]',
               grid='double[:, :, :, ::1]',
               )
@cython.returns('double[:, :, :, ::1]')
def tabulate_vectorfield(gridsize, func, factor, filename):
    """ This function creates and tabulates a cubic grid of size
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
    grid_local = empty([i_end - i_start, j_end - j_start, k_end - k_start] + [3])
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                # Compute the vector values via the passed function
                vector = func(i*factor, j*factor, k*factor)
                for dim in range(3):
                    grid_local[i - i_start, j - j_start, k - k_start, dim] = vector[dim]
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
               gridsize='int',
               two_gridsize='int',
               isnegative_x='bint',
               isnegative_y='bint',
               isnegative_z='bint',
               x_lower='size_t',
               y_lower='size_t',
               z_lower='size_t',
               x_upper='size_t',
               y_upper='size_t',
               z_upper='size_t',
               dim='size_t',
               xl='double',
               yl='double',
               zl='double',
               xu='double',
               yu='double',
               zu='double',
               )
@cython.returns('double*')
def CIC_grid2coordinates_vector(grid, x, y, z):
    """ This function look up tabulated vectors in a cubic grid and interpolates to
    (x, y, z) via the clouds-in-cell (CIC) method. Input arguments must be
    normalized so that 0 <= |x|, |y|, |z| < 1 (corresponding to boxsize = 1).
    It is assumed that only the (+++) octant of the total simulation box is
    represented by the grid, and that the same symmetries as in the Ewald method apply.
    """

    # Extract the size of the regular, cubic grid
    gridsize = grid.shape[0]
    two_gridsize = 2*gridsize
    # Shift x, y, z along the grid (in steps of the (normalized) boxsize, 1) so that the two
    # real particles are as close as they can get (only one octant of the box is tabulated).
    if x > 0.5:
        x -= 1
    elif x < -0.5:
        x += 1
    if y > 0.5:
        y -= 1
    elif y < -0.5:
        y += 1
    if z > 0.5:
        z -= 1
    elif z < -0.5:
        z += 1
    # Only the positive part of the box is tabulated
    if x > 0:
        isnegative_x = False
    else:
        x *= -1
        isnegative_x = True
    if y > 0:
        isnegative_y = False
    else:
        y *= -1
        isnegative_y = True
    if z > 0:
        isnegative_z = False
    else:
        z *= -1
        isnegative_z = True
    # It causes trouble when x, y, z is  exactly equal 0.5
    if x == 0.5:
        x -= two_machine_ϵ
    if y == 0.5:
        y -= two_machine_ϵ
    if z == 0.5:
        z -= two_machine_ϵ
    # Scale the coordinates so that 0 <= x, y, z < gridsize
    x *= two_gridsize
    y *= two_gridsize
    z *= two_gridsize
    # Indices of the 8 vertices (6 faces) of the grid
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
    for dim in range(3):
        vector[dim] = (grid[x_lower, y_lower, z_lower, dim]*xl*yl*zl +
                       grid[x_lower, y_lower, z_upper, dim]*xl*yl*zu +
                       grid[x_lower, y_upper, z_lower, dim]*xl*yu*zl +
                       grid[x_lower, y_upper, z_upper, dim]*xl*yu*zu +
                       grid[x_upper, y_lower, z_lower, dim]*xu*yl*zl +
                       grid[x_upper, y_lower, z_upper, dim]*xu*yl*zu +
                       grid[x_upper, y_upper, z_lower, dim]*xu*yu*zl +
                       grid[x_upper, y_upper, z_upper, dim]*xu*yu*zu)
    # Put the sign back in for negative input
    if isnegative_x:
        vector[0] *= -1
    if isnegative_y:
        vector[1] *= -1
    if isnegative_z:
        vector[2] *= -1
    return vector


# Function for CIC-interpolating particle coordinates to a cubic grid with scalar values
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               grid='double[:, :, ::1]',
               particles='Particles',
               # Locals
               posx='double*',
               posy='double*',
               posz='double*',
               gridsize='int',
               gridsize_over_boxsize='double',
               i='size_t',
               x='double',
               y='double',
               z='double',
               x_lower='size_t',
               y_lower='size_t',
               z_lower='size_t',
               x_upper='size_t',
               y_upper='size_t',
               z_upper='size_t',
               xl='double',
               yl='double',
               zl='double',
               xu='double',
               yu='double',
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




