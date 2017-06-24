# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
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
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from communication import communicate_domain,                             '
        '                          domain_layout_local_indices,                    '
        '                          domain_size_x,  domain_size_y,  domain_size_z,  '
        '                          domain_start_x, domain_start_y, domain_start_z, '
        '                          domain_subdivisions,                            '
        '                          get_buffer,                                     '
        '                          partition,                                      '
        '                          rank_neighboring_domain,                        '
        '                          smart_mpi,                                      '
        )



# Function for initializing and tabulating a cubic grid with
# vector values of a given dimension.
@cython.header(# Arguments
               gridsize='Py_ssize_t',
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
               size='Py_ssize_t',
               size_edge='Py_ssize_t',
               size_face='Py_ssize_t',
               size_local='Py_ssize_t',
               start_local='Py_ssize_t',
               vector_value='double*',
               ℓ_local='Py_ssize_t',
               ℓ='Py_ssize_t',
               returns='double[:, :, :, ::1]',
               )
def tabulate_vectorfield(gridsize, func, factor, filename=''):
    """This function tabulates a cubic grid of size
    gridsize*gridsize*gridsize with vector values computed by
    the function func, as
    grid[i, j, k] = func(i*factor, j*factor, k*factor).
    If filename is set, the tabulated grid is saved to a hdf5 file
    with this name.
    The grid is not distributed. Each process will end up with its own
    copy of the entire grid.
    The first point grid[0, 0, 0] corresponds to the origin of the box,
    while the last point grid[gridsize - 1, gridsize - 1, gridsize - 1]
    corresponds to the physical point ((gridsize - 1)/gridsize*boxsize,
                                       (gridsize - 1)/gridsize*boxsize,
                                       (gridsize - 1)/gridsize*boxsize).
    This means that grid[gridsize, gridsize, gridsize] is the corner
    of the box the furthers away from the origin. This point is not
    represented in the grid because of the periodicity of the box,
    which means that physically, this point is the same as the origin.
    The mapping from grid indices to physical coordinates is thus
    (i, j, k) --> (i/gridsize*boxsize,
                   j/gridsize*boxsize,
                   k/gridsize*boxsize).
    Therefore, if you wish to tabulate a global (and periodic) field,
    extending over the entire box, the factor argument should be
    proportional to boxsize/gridsize. If you wish to only tabulate say
    one octant of the box, this volume is no more periodic, and so
    the factor should be proportional to 0.5*boxsize/(gridsize - 1).
    """
    # The number of elements in each point, edge, face and bulk
    # (the total), when the grid is viewed as a 3D box of vectors.
    size_edge = gridsize*3
    size_face = gridsize*size_edge
    size      = gridsize*size_face
    # Initialize the global grid to be
    # of shape gridsize*gridsize*gridsize*3.
    # That is, grid is not really cubic, but rather four-dimensional.
    grid = empty([gridsize, gridsize, gridsize, 3], dtype=C2np['double'])
    # Partition the grid fairly among the processes.
    # Each part is defined by a linear size and starting index.
    n_vectors = gridsize**3
    start_local, n_vectors_local = partition(n_vectors)
    # Each point really consists of three elements
    start_local *= 3
    size_local = 3*n_vectors_local
    # Make a linear array for storing the local part of the grid
    grid_local = empty(size_local, dtype=C2np['double'])
    # Tabulate the local grid
    for ℓ_local in range(0, size_local, 3):
        ℓ = start_local + ℓ_local
        # The global 3D indices
        i = ℓ//size_face
        ℓ -= i*size_face
        j = ℓ//size_edge
        ℓ -= j*size_edge
        k = ℓ//3
        # Fill grid_local
        vector_value = func(i*factor, j*factor, k*factor)
        for dim in range(3):
            grid_local[ℓ_local + dim] = vector_value[dim]
    # Gather the tabulated local grid parts into the common, global grid
    smart_mpi(grid_local, grid, mpifun='allgatherv')
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
    # Correct for extreme values in the passed coordinates.
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
    # Correct for extreme values in the passed coordinates.
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
@cython.pheader(# Arguments
                gridA='double[:, :, :]',
                gridB='double[:, :, :]',
                fac='double',
                fac_grid='double[:, :, :]',
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
                use_fac_grid='bint',
                value='double',
                )
def CIC_grid2grid(gridA, gridB, fac=1, fac_grid=None):
    """This function CIC-interpolates values from one grid (gridB) onto
    another grid (gridA). The physical extend of the passed grids are
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
    If fac_grid is passed, this should be a grid of the same shape as
    gridA. An additional factor of fac_grid[i, j, k] will then be
    multiplied on the [i, j, k]'th interpolated value.
    """
    use_fac_grid = (fac_grid is not None)
    # If the two grids have the same shape, each grid point in gridA is
    # simply updated based on the equivalent grid point in gridB
    if (    gridA.shape[0] == gridB.shape[0]
        and gridA.shape[1] == gridB.shape[1]
        and gridA.shape[2] == gridB.shape[2]):
        # The two grids have equal shapes
        for         i in range(ℤ[gridA.shape[0] - 1]):
            for     j in range(ℤ[gridA.shape[1] - 1]):
                for k in range(ℤ[gridA.shape[2] - 1]):
                    value = fac*gridB[i, j, k]
                    with unswitch:
                        if use_fac_grid:
                            value *= fac_grid[i, j, k]
                    gridA[i, j, k] += value
        return
    # The two grids have different shapes. Perform CIC-interpolation.
    # Extract the shape of the grids (without the pseudo points).
    shapeA = tuple([gridA.shape[dim] - 1 for dim in range(3)])
    shapeB = tuple([gridB.shape[dim] - 1 for dim in range(3)])
    # Factors which scales grid indices in gridB
    # to (floating point) grid indices in gridA.
    scaling_i = shapeA[0]/shapeB[0]
    scaling_j = shapeA[1]/shapeB[1]
    scaling_k = shapeA[2]/shapeB[2]
    for iB in range(ℤ[shapeB[0]]):
        # The i-indices in gridA around the iB-index in gridB
        iA = iB*scaling_i
        if iA >= ℝ[shapeA[0]]:
            # The lower index must not be a boundary index
            iA = ℝ[shapeA[0]*(1 - machine_ϵ)]
        iA_lower = int(iA)
        iA_upper = iA_lower + 1
        for jB in range(ℤ[shapeB[1]]):
            # The j-indices in gridA around the jB-index in gridB
            jA = jB*scaling_j
            if jA >= ℝ[shapeA[1]]:
                # The lower index must not be a boundary index
                jA = ℝ[shapeA[1]*(1 - machine_ϵ)]
            jA_lower = int(jA)
            jA_upper = jA_lower + 1
            for kB in range(ℤ[shapeB[2]]):
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
                with unswitch:
                    if use_fac_grid:
                        gridA[iA_lower, jA_lower, kA_lower] += (
                            ℝ[value*Wil*Wjl]*Wkl*fac_grid[iA_lower, jA_lower, kA_lower])
                        gridA[iA_lower, jA_lower, kA_upper] += (
                            ℝ[value*Wil*Wjl]*Wku*fac_grid[iA_lower, jA_lower, kA_upper])
                        gridA[iA_lower, jA_upper, kA_lower] += (
                            ℝ[value*Wil*Wju]*Wkl*fac_grid[iA_lower, jA_upper, kA_lower])
                        gridA[iA_lower, jA_upper, kA_upper] += (
                            ℝ[value*Wil*Wju]*Wku*fac_grid[iA_lower, jA_upper, kA_upper])
                        gridA[iA_upper, jA_lower, kA_lower] += (
                            ℝ[value*Wiu*Wjl]*Wkl*fac_grid[iA_upper, jA_lower, kA_lower])
                        gridA[iA_upper, jA_lower, kA_upper] += (
                            ℝ[value*Wiu*Wjl]*Wku*fac_grid[iA_upper, jA_lower, kA_upper])
                        gridA[iA_upper, jA_upper, kA_lower] += (
                            ℝ[value*Wiu*Wju]*Wkl*fac_grid[iA_upper, jA_upper, kA_lower])
                        gridA[iA_upper, jA_upper, kA_upper] += (
                            ℝ[value*Wiu*Wju]*Wku*fac_grid[iA_upper, jA_upper, kA_upper])
                    else:
                        gridA[iA_lower, jA_lower, kA_lower] += ℝ[value*Wil*Wjl]*Wkl
                        gridA[iA_lower, jA_lower, kA_upper] += ℝ[value*Wil*Wjl]*Wku
                        gridA[iA_lower, jA_upper, kA_lower] += ℝ[value*Wil*Wju]*Wkl
                        gridA[iA_lower, jA_upper, kA_upper] += ℝ[value*Wil*Wju]*Wku
                        gridA[iA_upper, jA_lower, kA_lower] += ℝ[value*Wiu*Wjl]*Wkl
                        gridA[iA_upper, jA_lower, kA_upper] += ℝ[value*Wiu*Wjl]*Wku
                        gridA[iA_upper, jA_upper, kA_lower] += ℝ[value*Wiu*Wju]*Wkl
                        gridA[iA_upper, jA_upper, kA_upper] += ℝ[value*Wiu*Wju]*Wku

# Function for CIC-interpolating particles/fluid elements of
# components to a domain grid.
@cython.header(# Argument
               component_or_components='object', # Component or list of Components
               domain_grid='double[:, :, ::1]',
               quantities='list',
               # Locals
               Wxl='double',
               Wyl='double',
               Wzl='double',
               Wxu='double',
               Wyu='double',
               Wzu='double',
               amount='double',
               component='Component',
               components='list',
               domain_grid_noghosts='double[:, :, :]',
               factor='double',
               factors='double[::1]',
               fluid_quantity='double[:, :, :]',
               i='Py_ssize_t',
               interpolated_particles='bint',
               interpolations='int',
               j='Py_ssize_t',
               particle_quantity='double*',
               posx='double*',
               posy='double*',
               posz='double*',
               quantities_implemented='tuple',
               quantity='str',
               shape='tuple',
               x='double',
               x_lower='int',
               x_upper='int',
               y='double',
               y_lower='int',
               y_upper='int',
               z='double',
               z_lower='int',
               z_upper='int',
               )
def CIC_components2domain_grid(component_or_components, domain_grid, quantities):
    """This function CIC-interpolates particle/fluid elements
    to domain_grid storing scalar values. The physical extend of the
    passed domain_grid should match the domain exactly. The interpolated
    values will be added to the grid. Therefore, if the grid should
    contain the interpolated vales only, the grid must be nullified 
    beforehand.
    The quantities argument is a list, but its elements can be
    structured in different ways. If quantities = [], a particle and a
    fluid element will each contribute to the domain_grid with an amount
    of 1. This can be scaled by supplying e.g.
    quantities = [('particles', 2), ('fluid elements', 3)].
    If a specific quantity of a component should be interpolated, this
    can be specified as e.g. quantities = ['momx', 'Jx']. These can
    similarly be scaled by quantities = [('momx', 2), ('Jx', 3)].
    Finally, if each component should be scaled differently, this can
    be specified as e.g.
    quantities = [('momx', [1, 2]), ('Jx', [1, 3])].
    As a complete example, consider interpolating the comoving density:
    quantities = [('particles', [m₁/Vcell, m₂/Vcell, ..., mₙ/Vcell]),
                  ('ϱ', [a**(-3*w₁), a**(-3*w₂), ..., a**(-3*wₙ))]],
    where mᵢ are the i'th mass and Vcell is the (comoving) volume of a
    single cell of the domain grid. The order of the elements in the
    lists should match the order of components,
    and so n = len(components) even though both
    particle and fluid components are present.
    """
    if isinstance(component_or_components, list):
        components = component_or_components
    else:
        components = [component_or_components]
    # Transform the supplied quantities so that it is a list of tuples
    # of the form (str, np.ndarray), where the array is of the same
    # length as components.
    quantities = quantities.copy()
    for i, quantity_raw in enumerate(quantities):
        if isinstance(quantity_raw, str):
            quantities[i] = (quantity_raw, ones(len(components), dtype=C2np['double']))
        elif len(quantity_raw) == 2:
            try:
                quantities[i] = (quantity_raw[0],
                                 asarray([float(quantity_raw[1])]*len(components)))
            except:
                quantities[i] = (quantity_raw[0], asarray(quantity_raw[1]))
        else:
            quantities[i] = (quantity_raw[0], asarray(quantity_raw[1:]))
    # Memoryview of the domain grid without the ghost layers
    domain_grid_noghosts = domain_grid[2:(domain_grid.shape[0] - 2),
                                       2:(domain_grid.shape[1] - 2),
                                       2:(domain_grid.shape[2] - 2)]
    shape = tuple([domain_grid_noghosts.shape[dim] - 1 for dim in range(3)])
    # Do the interpolation(s)
    interpolations = 0
    interpolated_particles = False
    for i, component in enumerate(components):
        if component.representation == 'particles':
            posx = component.posx
            posy = component.posy
            posz = component.posz
            # Interpolate each particle quantity
            for quantity, factors in quantities:
                # Grab the quantity to be interpolated
                if quantity == 'particles':
                    ...  # Accept but do not assign data pointer
                elif quantity == 'posx':
                    particle_quantity = component.posx
                elif quantity == 'posy':
                    particle_quantity = component.posy
                elif quantity == 'posz':
                    particle_quantity = component.posz
                elif quantity == 'momx':
                    particle_quantity = component.momx
                elif quantity == 'momy':
                    particle_quantity = component.momy
                elif quantity == 'momz':
                    particle_quantity = component.momz
                else:
                    continue
                interpolations += 1
                interpolated_particles = True
                factor = factors[i]
                # For quantity == 'particles', each particle should
                # contribute with an amount equal to factor
                # (for quantity != 'particles', this will be overwritten
                # in the loop below).
                amount = factor
                # Interpolate each particle
                for j in range(component.N_local):
                    # Get the amount this particle contribute
                    # to the interpolated grid.
                    with unswitch(1):
                        if quantity != 'particles':
                            amount = factor*particle_quantity[j]
                    # Get, translate and scale the coordinates so that
                    # 0 <= j < shape[j] - 1 for j in (x, y, z).
                    x = (posx[j] - domain_start_x)*ℝ[shape[0]/domain_size_x]
                    y = (posy[j] - domain_start_y)*ℝ[shape[1]/domain_size_y]
                    z = (posz[j] - domain_start_z)*ℝ[shape[2]/domain_size_z]
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
                    domain_grid_noghosts[x_lower, y_lower, z_lower] += ℝ[amount*Wxl*Wyl]*Wzl
                    domain_grid_noghosts[x_lower, y_lower, z_upper] += ℝ[amount*Wxl*Wyl]*Wzu
                    domain_grid_noghosts[x_lower, y_upper, z_lower] += ℝ[amount*Wxl*Wyu]*Wzl
                    domain_grid_noghosts[x_lower, y_upper, z_upper] += ℝ[amount*Wxl*Wyu]*Wzu
                    domain_grid_noghosts[x_upper, y_lower, z_lower] += ℝ[amount*Wxu*Wyl]*Wzl
                    domain_grid_noghosts[x_upper, y_lower, z_upper] += ℝ[amount*Wxu*Wyl]*Wzu
                    domain_grid_noghosts[x_upper, y_upper, z_lower] += ℝ[amount*Wxu*Wyu]*Wzl
                    domain_grid_noghosts[x_upper, y_upper, z_upper] += ℝ[amount*Wxu*Wyu]*Wzu
        elif component.representation == 'fluid':
            # Interpolate each fluid quantity
            for quantity, factors in quantities:
                # Grab the quantity to be interpolated
                if quantity == 'fluid elements':
                    ...  # Accept but do not assign data array
                elif quantity == 'ϱ':
                    fluid_quantity = component.ϱ.grid_noghosts
                elif quantity == 'Jx':
                    fluid_quantity = component.Jx.grid_noghosts
                elif quantity == 'Jy':
                    fluid_quantity = component.Jy.grid_noghosts
                elif quantity == 'Jz':
                    fluid_quantity = component.Jz.grid_noghosts
                elif quantity == 'σxx':
                    fluid_quantity = component.σxx.grid_noghosts
                elif quantity == 'σxy':
                    fluid_quantity = component.σxy.grid_noghosts
                elif quantity == 'σxz':
                    fluid_quantity = component.σxz.grid_noghosts
                elif quantity == 'σyx':
                    fluid_quantity = component.σyx.grid_noghosts
                elif quantity == 'σyy':
                    fluid_quantity = component.σyy.grid_noghosts
                elif quantity == 'σyz':
                    fluid_quantity = component.σyz.grid_noghosts
                elif quantity == 'σzx':
                    fluid_quantity = component.σzx.grid_noghosts
                elif quantity == 'σzy':
                    fluid_quantity = component.σzy.grid_noghosts
                elif quantity == 'σzz':
                    fluid_quantity = component.σzz.grid_noghosts
                else:
                    continue
                interpolations += 1
                factor = factors[i]
                # Do the grid to grid interpolation
                CIC_grid2grid(domain_grid_noghosts,
                              fluid_quantity,
                              factor,
                              )
    # As a result of interpolating particles and/or fluid elements,
    # values of local pseudo mesh points may contribute to the lower
    # mesh points of the domain grid on other processes.
    # Do the needed communication.
    communicate_domain(domain_grid, mode='add contributions')
    # Check that each quantity got interpolated
    if interpolations != len(quantities):
        quantities_implemented = (# Particle quantities
                                  'particles', 'posx', 'posy', 'posz', 'momx', 'momy', 'momz',
                                  # Fluid quantities
                                  'fluid elements', 'ϱ', 'Jx', 'Jy', 'Jz',
                                  'σxx', 'σxy', 'σxz', 'σyx', 'σyy', 'σyz', 'σzx', 'σzy', 'σzz',
                                  )
        for quantity, factors in quantities:
            if quantity not in quantities_implemented:
                masterwarn('Could not interpolate component quantity "{}" onto grid '
                           'as this quantity is not implemented.'
                           .format(quantity))

# Function for CIC-interpolating particles of a particle component
# to fluid grids.
@cython.header(# Argument
               component='Component',
               # Locals
               Jx='double*',
               Jx_mv='double[:, :, ::1]',
               Jx_noghosts='double[:, :, :]',
               Jy='double*',
               Jy_mv='double[:, :, ::1]',
               Jy_noghosts='double[:, :, :]',
               Jz='double*',
               Jz_mv='double[:, :, ::1]',
               Jz_noghosts='double[:, :, :]',
               N_vacuum='Py_ssize_t',
               Vcell='double',
               Wlll='double',
               Wllu='double',
               Wlul='double',
               Wluu='double',
               Wull='double',
               Wulu='double',
               Wuul='double',
               Wuuu='double',
               Wxl='double',
               Wxu='double',
               Wyl='double',
               Wyu='double',
               Wzl='double',
               Wzu='double',
               dim='int',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               mass='double',
               momx='double*',
               momx_i='double',
               momy='double*',
               momy_i='double',
               momz='double*',
               momz_i='double',
               original_representation='str',
               posx='double*',
               posy='double*',
               posz='double*',
               shape='tuple',
               var='str',
               x='double',
               x_lower='int',
               x_upper='int',
               y='double',
               y_lower='int',
               y_upper='int',
               z='double',
               z_lower='int',
               z_upper='int',
               Δϱ='double',
               Δϱ_tot='double',
               ϱ_noghosts='double[:, :, :]',
               ϱ='double*',
               ϱ_mv='double[:, :, ::1]',
               returns='Py_ssize_t',
               )
def CIC_particles2fluid(component):
    """This function CIC-interpolates particle positions to fluid grids.
    The passed component should contain particle data, but not
    necessarily fluid data (any pre-existing fluid data will be
    overwritten). The particle data are then used to create the fluid
    grids (ϱ, Jx, Jy, Jz).
    The relation between the particle data and the fluid density ϱ is
    ϱ = (N*mass)/Vcell,
    where N is the number of particles in the cell, mass is the mass of
    a single particle (all particles have the same mass) and Vcell is
    the volume of a cell (all cells have the same volume). The number of
    particles N in the volume is not actually a whole number, but rather
    a fraction due to the CIC interpolation:
    N = ΣᵢWᵢ,
    where Wᵢ is the CIC weights, which sum to 1 for a single particle.
    We can then express the density field as
    ϱ = mass/Vcell*ΣᵢWᵢ.
    Note that the cell volumes (as all volumes) are comoving, meaning no
    factors of the scale factor a are needed. Remembering that
    ϱ = a**(3*(1 + w))*ρ, where ρ is the proper density, this convertion
    thus imply w = 0. This function does not however check that this
    requirement is actually fulfilled.
    The relation between particle data and fluid momentum density J is
    J = a**4*ρ*u
      = a**(-3*w)*a*ϱ*u
      = a*ϱ*u,    (w = 0)
    where u - for each fluid element - is the overall velocity of the
    fluid element. Crucially, velocities are not added, as two particles
    within the same cell volume does not lead to a fluid element with
    a velocity of their sum. Rather, velocities are averaged together
    using the same weights as above:
    u = <u>
      = 1/N*(ΣᵢWᵢ*uᵢ),
    where uᵢ are the peculiar velocities of the particles.
    The fluid momentum density then becomes
    J = a*mass/Vcell*(ΣᵢWᵢ*uᵢ).
    The relation between a particles velocity uᵢ
    and its momentum momᵢ is
    momᵢ = a*mass*uᵢ
    The final expression for J is then
    J = 1/Vcell*(ΣᵢWᵢ*momᵢ).  
    Importantly, the mass attribute of the passed component should be
    the particle mass, not the average fluid element mass. The value of
    the representation attribute does not matter and will not
    be altered. The size of the fluid grids are determined
    by component.gridsize. To save memory, the particle data will be
    freed (resized to a minimum size) during the process.
    """
    # Backup of original representation
    original_representation = component.representation
    # Instantiate fluid grids spanning the local domains.
    # The newly allocated grids will be nullified.
    component.representation = 'fluid'
    shape = tuple([component.gridsize//ds for ds in domain_subdivisions])
    if master and any([component.gridsize != domain_subdivisions[dim]*shape[dim]
                       for dim in range(3)]):
            abort('The gridsize of the {} component is {}\n'
                  'which cannot be equally shared among {} processes'
                  .format(component.name, component.gridsize, nprocs))
    component.resize(shape)  # This also nullifies all fluid grids
    # Extract fluid data variables
    ϱ           = component.ϱ .grid
    ϱ_mv        = component.ϱ .grid_mv
    ϱ_noghosts  = component.ϱ .grid_noghosts
    Jx          = component.Jx.grid
    Jx_mv       = component.Jx.grid_mv
    Jx_noghosts = component.Jx.grid_noghosts
    Jy          = component.Jy.grid
    Jy_mv       = component.Jy.grid_mv
    Jy_noghosts = component.Jy.grid_noghosts
    Jz          = component.Jz.grid
    Jz_mv       = component.Jz.grid_mv
    Jz_noghosts = component.Jz.grid_noghosts
    # Extract particle data variables
    posx = component.posx
    posy = component.posy
    posz = component.posz
    momx = component.momx
    momy = component.momy
    momz = component.momz
    # Variables used in the convertion from particle data to fluid data
    mass = component.mass
    Vcell = (boxsize/component.gridsize)**3
    # Interpolate each particle to the grids.
    # Constant factors will be multiplied on later.
    # Thus after the interpolation we will have
    # ϱ: ΣᵢWᵢ
    # J: ΣᵢWᵢ*momᵢ
    for i in range(component.N_local):
        # Get, translate and scale the coordinates so that
        # 0 <= i < shape[i] for i in (x, y, z).
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
        # The full weights
        Wlll = ℝ[Wxl*Wyl]*Wzl
        Wllu = ℝ[Wxl*Wyl]*Wzu
        Wlul = ℝ[Wxl*Wyu]*Wzl
        Wluu = ℝ[Wxl*Wyu]*Wzu
        Wull = ℝ[Wxu*Wyl]*Wzl
        Wulu = ℝ[Wxu*Wyl]*Wzu
        Wuul = ℝ[Wxu*Wyu]*Wzl
        Wuuu = ℝ[Wxu*Wyu]*Wzu
        # Assign the raw weights to the ϱ grid
        ϱ_noghosts[x_lower, y_lower, z_lower] += Wlll
        ϱ_noghosts[x_lower, y_lower, z_upper] += Wllu
        ϱ_noghosts[x_lower, y_upper, z_lower] += Wlul
        ϱ_noghosts[x_lower, y_upper, z_upper] += Wluu
        ϱ_noghosts[x_upper, y_lower, z_lower] += Wull
        ϱ_noghosts[x_upper, y_lower, z_upper] += Wulu
        ϱ_noghosts[x_upper, y_upper, z_lower] += Wuul
        ϱ_noghosts[x_upper, y_upper, z_upper] += Wuuu
        # Extract momentum of the i'th particle
        momx_i = momx[i]
        momy_i = momy[i]
        momz_i = momz[i]
        # Assign the weighted x-momentum to the Jx grid
        Jx_noghosts[x_lower, y_lower, z_lower] += Wlll*momx_i
        Jx_noghosts[x_lower, y_lower, z_upper] += Wllu*momx_i
        Jx_noghosts[x_lower, y_upper, z_lower] += Wlul*momx_i
        Jx_noghosts[x_lower, y_upper, z_upper] += Wluu*momx_i
        Jx_noghosts[x_upper, y_lower, z_lower] += Wull*momx_i
        Jx_noghosts[x_upper, y_lower, z_upper] += Wulu*momx_i
        Jx_noghosts[x_upper, y_upper, z_lower] += Wuul*momx_i
        Jx_noghosts[x_upper, y_upper, z_upper] += Wuuu*momx_i
        # Assign the weighted y-momentum to the Jy grid
        Jy_noghosts[x_lower, y_lower, z_lower] += Wlll*momy_i
        Jy_noghosts[x_lower, y_lower, z_upper] += Wllu*momy_i
        Jy_noghosts[x_lower, y_upper, z_lower] += Wlul*momy_i
        Jy_noghosts[x_lower, y_upper, z_upper] += Wluu*momy_i
        Jy_noghosts[x_upper, y_lower, z_lower] += Wull*momy_i
        Jy_noghosts[x_upper, y_lower, z_upper] += Wulu*momy_i
        Jy_noghosts[x_upper, y_upper, z_lower] += Wuul*momy_i
        Jy_noghosts[x_upper, y_upper, z_upper] += Wuuu*momy_i
        # Assign the weighted z-momentum to the Jz grid
        Jz_noghosts[x_lower, y_lower, z_lower] += Wlll*momz_i
        Jz_noghosts[x_lower, y_lower, z_upper] += Wllu*momz_i
        Jz_noghosts[x_lower, y_upper, z_lower] += Wlul*momz_i
        Jz_noghosts[x_lower, y_upper, z_upper] += Wluu*momz_i
        Jz_noghosts[x_upper, y_lower, z_lower] += Wull*momz_i
        Jz_noghosts[x_upper, y_lower, z_upper] += Wulu*momz_i
        Jz_noghosts[x_upper, y_upper, z_lower] += Wuul*momz_i
        Jz_noghosts[x_upper, y_upper, z_upper] += Wuuu*momz_i
    # The particle data is no longer needed. Free it to save memory.
    component.representation = 'particles'
    component.resize(1)
    # Values of local pseudo mesh points contribute to the lower
    # mesh points of the domain (fluid) grids on other processes.
    # Do the needed communications.
    component.representation = 'fluid'
    component.communicate_fluid_grids(mode='add contributions')
    # Multiply the missing constant factors on the
    # interpolated grid values. Here ghosts and pseudo points
    # are excluded. Vaccum elements (fluid elements not
    # interpolated to) will be assigned the vacuum density ϱ_vacuum.
    N_vacuum = 0
    for         i in range(ℤ[ϱ_noghosts.shape[0] - 1]):
        for     j in range(ℤ[ϱ_noghosts.shape[1] - 1]):
            for k in range(ℤ[ϱ_noghosts.shape[2] - 1]):
                if ϱ_noghosts[i, j, k] < ϱ_vacuum:
                    # Vacuuum element detected. Assign the vacuum
                    # density and leave the momentum at zero.
                    N_vacuum += 1
                    ϱ_noghosts[i, j, k] = ϱ_vacuum
                else:
                    ϱ_noghosts [i, j, k] *= ℝ[mass/Vcell]
                    Jx_noghosts[i, j, k] *= ℝ[1/Vcell]
                    Jy_noghosts[i, j, k] *= ℝ[1/Vcell]
                    Jz_noghosts[i, j, k] *= ℝ[1/Vcell]
    # Count up number of vacuum elements from all processes
    N_vacuum = allreduce(N_vacuum, op=MPI.SUM)
    # If any vacuum elements exist, the fact that the assigned vacuum
    # density is non-zero means that the total mass is not conserved.
    # Correct this by lowering every non-vacuum element by the same
    # amount. This act may itself produce densities lower than the
    # vacuum density, so we have to keep doing this until no sub-vaccum
    # densities exist.
    Δϱ_tot = N_vacuum*ϱ_vacuum
    while Δϱ_tot != 0:
        Δϱ_tot = 0
        Δϱ = Δϱ_tot/component.gridsize**3
        for         i in range(ℤ[ϱ_noghosts.shape[0] - 1]):
            for     j in range(ℤ[ϱ_noghosts.shape[1] - 1]):
                for k in range(ℤ[ϱ_noghosts.shape[2] - 1]):
                    if ϱ_noghosts[i, j, k] > ϱ_vacuum:
                        ϱ_noghosts[i, j, k] -= Δϱ
                        if ϱ_noghosts[i, j, k] < ϱ_vacuum:
                            Δϱ_tot += ϱ_vacuum - ϱ_noghosts[i, j, k]
                            ϱ_noghosts[i, j, k] = ϱ_vacuum
        Δϱ_tot = allreduce(Δϱ, op=MPI.SUM)
    # The local bulk of all fluid grids now hold the final values.
    # Populate pseudo and ghost points.
    component.communicate_fluid_grids(mode='populate')
    # Re-insert the original representation
    component.representation = original_representation
    # Return the number of fluid elements not interpolated to
    return N_vacuum

# Function for CIC interpolating components to the φ grid
@cython.header(# Arguments
               component_or_components='object', # Component or list of Components
               quantities='list',
               # Locals
               φ='double[:, :, ::1]',
               returns='double[:, :, ::1]',
               )
def CIC_components2φ(component_or_components, quantities):
    """Exactly what quantities of the components are interpolated to
    the global φ grid is determined by the quantities argument.
    For details on this argument,
    see the CIC_components2domain_grid function.
    """
    # If φ_gridsize is illegal, abort now
    if φ_illegal:
        abort(φ_illegal)
    # Fetch the φ grid
    φ = get_buffer(φ_shape, 'φ', nullify=True)
    # Interpolate component coordinates
    # weighted by the given quantities to φ.
    CIC_components2domain_grid(component_or_components, φ, quantities)
    return φ
# Check that φ_gridsize fulfills the requirements for FFT.
# If not, the reason why will be stored in φ_illegal.
cython.declare(φ_illegal='str')
φ_illegal = ''
if φ_gridsize%nprocs != 0:
    φ_illegal = f'A φ_gridsize = {φ_gridsize} cannot be evenly divided by {nprocs} processes.'            
else:
    if (   φ_gridsize%domain_subdivisions[0] != 0
        or φ_gridsize%domain_subdivisions[1] != 0
        or φ_gridsize%domain_subdivisions[2] != 0
        ):
        φ_illegal = (f'As φ_gridsize = {φ_gridsize}, the global φ grid have a shape of'
                     f'({φ_gridsize}, {φ_gridsize}, {φ_gridsize}), which cannot be divided '
                     f'according to the domain decomposition ({domain_subdivisions[0]}, '
                     f'{domain_subdivisions[1]}, {domain_subdivisions[2]}).'
                     )
if φ_gridsize%2 != 0:
    masterwarn(f'As φ_gridsize = {φ_gridsize} is odd, some operations may not function correctly.')
# The shape of the domain φ grid, including pseudo and ghost points
cython.declare(φ_shape='tuple')
φ_shape = tuple([φ_gridsize//domain_subdivisions[dim] + 1 + 2*2 for dim in range(3)])

# Function that compute a lot of information needed by the
# slab_decompose and domain_decompose functions.
@cython.header(# Arguments
               domain_grid='double[:, :, ::1]',
               slab='double[:, :, ::1]',
               # Locals
               N_domain2slabs_communications='Py_ssize_t',
               domain_end_i='Py_ssize_t',
               domain_end_j='Py_ssize_t',
               domain_end_k='Py_ssize_t',
               domain_grid_noghosts='double[:, :, :]',
               domain_grid_shape='tuple',
               domain_sendrecv_i_end='int[::1]',
               domain_sendrecv_i_start='int[::1]',
               domain_size_i='Py_ssize_t',
               domain_size_j='Py_ssize_t',
               domain_size_k='Py_ssize_t',
               domain_start_i='Py_ssize_t',
               domain_start_j='Py_ssize_t',
               domain_start_k='Py_ssize_t',
               domain2slabs_recvsend_ranks='int[::1]',
               index='Py_ssize_t',
               info='tuple',
               rank_recv='int',
               rank_send='int',
               recvtuple='tuple',
               sendtuple='tuple',
               slab_sendrecv_j_end='int[::1]',
               slab_sendrecv_j_start='int[::1]',
               slab_sendrecv_k_end='int[::1]',
               slab_sendrecv_k_start='int[::1]',
               slab_shape='tuple',
               slab_size_i='Py_ssize_t',
               slabs2domain_sendrecv_ranks='int[::1]',
               ℓ='Py_ssize_t',
               returns='tuple',
               )
def prepare_decomposition(domain_grid, slab):
    # Simply look up and return the needed information
    # if previously computed.
    domain_grid_shape = asarray(domain_grid).shape
    slab_shape = asarray(slab).shape
    info = decomposition_info.get((domain_grid_shape, slab_shape))
    if info:
        return info
    # Memoryview of the domain grid without the ghost layers
    domain_grid_noghosts = domain_grid[2:(domain_grid.shape[0] - 2),
                                       2:(domain_grid.shape[1] - 2),
                                       2:(domain_grid.shape[2] - 2),
                                       ]
    # The size (number of grid points) of the truly local part of the
    # domain grid, excluding both ghost layers and pseudo points,
    # for each dimension.
    domain_size_i = domain_grid_noghosts.shape[0] - 1
    domain_size_j = domain_grid_noghosts.shape[1] - 1
    domain_size_k = domain_grid_noghosts.shape[2] - 1
    # The global start and end indices of the local domain
    # in the global grid.
    domain_start_i = domain_layout_local_indices[0]*domain_size_i
    domain_start_j = domain_layout_local_indices[1]*domain_size_j
    domain_start_k = domain_layout_local_indices[2]*domain_size_k
    domain_end_i = domain_start_i + domain_size_i
    domain_end_j = domain_start_j + domain_size_j
    domain_end_k = domain_start_k + domain_size_k
    # When in real space, the slabs are distributed over the first
    # dimension. Give the size of the slab in this dimension a name.
    slab_size_i = slab.shape[0]
    # Find local i-indices to send and to which process by
    # shifting a piece of the number line in order to match
    # the communication pattern used.
    domain_sendrecv_i_start = np.roll(asarray([ℓ - domain_start_i
                                               for ℓ in range(domain_start_i,
                                                               domain_end_i,
                                                               slab_size_i)
                                               ], dtype=C2np['int']),
                                      -rank)
    domain_sendrecv_i_end = np.roll(asarray([ℓ - domain_start_i + slab_size_i
                                             for ℓ in range(domain_start_i,
                                                             domain_end_i,
                                                             slab_size_i)
                                             ], dtype=C2np['int']),
                                    -rank)
    slabs2domain_sendrecv_ranks = np.roll(asarray([ℓ//slab_size_i
                                                  for ℓ in range(domain_start_i, 
                                                                  domain_end_i,
                                                                  slab_size_i)],
                                                  dtype=C2np['int']),
                                     -rank)
    # Communicate the start and end (j, k)-indices of the slab,
    # where parts of the local domains should be received into.
    slab_sendrecv_j_start       = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_k_start       = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_j_end         = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_k_end         = empty(nprocs, dtype=C2np['int'])
    domain2slabs_recvsend_ranks = empty(nprocs, dtype=C2np['int'])
    index = 0
    for ℓ in range(nprocs):
        # Process ranks to send/receive to/from
        rank_send = mod(rank + ℓ, nprocs)
        rank_recv = mod(rank - ℓ, nprocs)
        # Send the global y and z start and end indices of the domain
        # to be send, if anything should be send to process rank_send.
        # Otherwise send None.
        sendtuple = ((domain_start_j, domain_start_k, domain_end_j, domain_end_k)
                     if rank_send in asarray(slabs2domain_sendrecv_ranks) else None)
        recvtuple = sendrecv(sendtuple, dest=rank_send, source=rank_recv)
        if recvtuple is not None:
            slab_sendrecv_j_start[index]       = recvtuple[0]
            slab_sendrecv_k_start[index]       = recvtuple[1]
            slab_sendrecv_j_end[index]         = recvtuple[2]
            slab_sendrecv_k_end[index]         = recvtuple[3]
            domain2slabs_recvsend_ranks[index] = rank_recv
            index += 1
    # Cut off the tails
    slab_sendrecv_j_start = slab_sendrecv_j_start[:index]
    slab_sendrecv_k_start = slab_sendrecv_k_start[:index]
    slab_sendrecv_j_end = slab_sendrecv_j_end[:index]
    slab_sendrecv_k_end = slab_sendrecv_k_end[:index]
    domain2slabs_recvsend_ranks = domain2slabs_recvsend_ranks[:index]
    # The maximum number of communications it takes to communicate
    # the domain grid to the slabs (or vice versa).
    N_domain2slabs_communications = np.max([slabs2domain_sendrecv_ranks.shape[0],
                                            domain2slabs_recvsend_ranks.shape[0]])
    # Store and return all the resultant information,
    # needed for communicating between the domain and the slab.
    info = (N_domain2slabs_communications,
            domain2slabs_recvsend_ranks,
            slabs2domain_sendrecv_ranks,
            domain_sendrecv_i_start,
            domain_sendrecv_i_end,
            slab_sendrecv_j_start,
            slab_sendrecv_j_end,
            slab_sendrecv_k_start,
            slab_sendrecv_k_end,
            )
    decomposition_info[domain_grid_shape, slab_shape] = info
    return info
# Cache storing results of the prepare_decomposition function.
# The keys have the format (domain_grid_shape, slab_shape).
cython.declare(decomposition_info='dict')
decomposition_info = {}

# Function for transfering data from slabs to domain grids
@cython.pheader(# Arguments
                slab='double[:, :, ::1]',
                domain_grid_or_buffer_name='object',  # double[:, :, ::1], int or str
                # Locals
                N_domain2slabs_communications='Py_ssize_t',
                buffer_name='object',  # int or str
                domain_grid='double[:, :, ::1]',
                domain_grid_noghosts='double[:, :, :]',
                domain_sendrecv_i_end='int[::1]',
                domain_sendrecv_i_start='int[::1]',
                domain2slabs_recvsend_ranks='int[::1]',
                gridsize='Py_ssize_t',
                request='object',  # mpi4py.MPI.Request
                shape='tuple',
                slab_sendrecv_j_end='int[::1]',
                slab_sendrecv_j_start='int[::1]',
                slab_sendrecv_k_end='int[::1]',
                slab_sendrecv_k_start='int[::1]',
                slabs2domain_sendrecv_ranks='int[::1]',
                ℓ='Py_ssize_t',
                returns='double[:, :, ::1]',
                )
def domain_decompose(slab, domain_grid_or_buffer_name=0):
    if slab.shape[0] > slab.shape[1]:
        masterwarn('domain_decompose was called with a slab that appears to be transposed, '
                   'i.e. in Fourier space.')
    # Determine the correct shape of the domain grid corresponding to
    # the passed slab.
    gridsize = slab.shape[1]
    shape = tuple([gridsize//domain_subdivisions[dim] + 1 + 2*2 for dim in range(3)])
    # If no domain grid is passed, fetch a buffer of the right shape
    if isinstance(domain_grid_or_buffer_name, (int, str)):
        buffer_name = domain_grid_or_buffer_name
        domain_grid = get_buffer(shape, buffer_name)
    else:
        domain_grid = domain_grid_or_buffer_name
        if asarray(domain_grid).shape != shape:
            abort('The slab and domain grid passed to domain_decompose '
                  'have incompatible shapes: {}, {}.'
                  .format(asarray(slab).shape, asarray(domain_grid).shape)
                  )
    domain_grid_noghosts = domain_grid[2:(domain_grid.shape[0] - 2),
                                       2:(domain_grid.shape[1] - 2),
                                       2:(domain_grid.shape[2] - 2)]
    # Compute needed communication variables
    (N_domain2slabs_communications,
     domain2slabs_recvsend_ranks,
     slabs2domain_sendrecv_ranks,
     domain_sendrecv_i_start,
     domain_sendrecv_i_end,
     slab_sendrecv_j_start,
     slab_sendrecv_j_end,
     slab_sendrecv_k_start,
     slab_sendrecv_k_end,
     ) = prepare_decomposition(domain_grid, slab)
    # Communicate the slabs to the domain grid
    for ℓ in range(N_domain2slabs_communications):
        # The lower ranks storing the slabs sends part of their slab
        if ℓ < domain2slabs_recvsend_ranks.shape[0]:
            # A non-blocking send is used, because the communication
            # is not pairwise.
            # In the x-dimension, the slabs are always thinner than (or
            # at least as thin as) the domain.
            request = smart_mpi(slab[:,
                                     slab_sendrecv_j_start[ℓ]:slab_sendrecv_j_end[ℓ],
                                     slab_sendrecv_k_start[ℓ]:slab_sendrecv_k_end[ℓ],
                                     ],
                                dest=domain2slabs_recvsend_ranks[ℓ],
                                mpifun='Isend')
        # The corresponding process receives the message.
        # Since the slabs extend throughout the entire yz-plane,
        # we receive into the entire yz-part of the domain grid
        # (excluding ghost and pseudo points).
        if ℓ < slabs2domain_sendrecv_ranks.shape[0]:
            smart_mpi(domain_grid_noghosts[domain_sendrecv_i_start[ℓ]:domain_sendrecv_i_end[ℓ],
                                           :ℤ[domain_grid_noghosts.shape[1] - 1],
                                           :ℤ[domain_grid_noghosts.shape[2] - 1],
                                           ],
                      source=slabs2domain_sendrecv_ranks[ℓ],
                      mpifun='Recv')
        # Wait for the non-blockind send to be complete before
        # continuing. Otherwise, data in the send buffer - which is
        # still in use by the non-blocking send - might get overwritten
        # by the next (non-blocking) send.
        request.wait()
    # The right/forward/upper boundaries (the layer of pseudo points,
    # not the ghost layer) of the domain grid should be a copy of the
    # left/backward/lower boundaries of the neighboring
    # right/forward/upper domain. Do the needed communication.
    # Also populate the ghost layers of the domain grid.
    communicate_domain(domain_grid, mode='populate')
    return domain_grid

# Function for transfering data from domain grids to slabs
@cython.pheader(# Arguments
                domain_grid='double[:, :, ::1]',
                slab_or_buffer_name='object',  # double[:, :, ::1], int or str
                prepare_fft='bint',
                # Locals
                N_domain2slabs_communications='Py_ssize_t',
                buffer_name='object',  # int or str
                domain_grid_noghosts='double[:, :, :]',
                domain_sendrecv_i_end='int[::1]',
                domain_sendrecv_i_start='int[::1]',
                domain2slabs_recvsend_ranks='int[::1]',
                gridsize='Py_ssize_t',
                request='object',  # mpi4py.MPI.Request object
                shape='tuple',
                slab='double[:, :, ::1]',
                slab_sendrecv_j_end='int[::1]',
                slab_sendrecv_j_start='int[::1]',
                slab_sendrecv_k_end='int[::1]',
                slab_sendrecv_k_start='int[::1]',
                slabs2domain_sendrecv_ranks='int[::1]',
                ℓ='Py_ssize_t',
                returns='double[:, :, ::1]',
                )
def slab_decompose(domain_grid, slab_or_buffer_name=0, prepare_fft=False):
    """This function communicates a global grid decomposed into domain
    grids into slabs. If an existing slab grid should be used it can be
    passed as the second argument. Alternatively, if a slab grid should
    be fetched from elsewhere, its name should be specified as the
    second argument. If FFT's are to be carried out on the slab,
    you must give a buffer name as the second argument and specify
    prepare_fft=True, in which case the slab will be created via FFTW.
    """
    # Determine the correct shape of the slab grid corresponding to
    # the passed domain grid.
    domain_grid_noghosts = domain_grid[2:(domain_grid.shape[0] - 2),
                                       2:(domain_grid.shape[1] - 2),
                                       2:(domain_grid.shape[2] - 2)]
    gridsize = (domain_grid_noghosts.shape[0] - 1)*domain_subdivisions[0]
    if gridsize%nprocs != 0:
        abort('A domain decomposed grid of gridsize {} was passed to the slab_decompose function. '
              'This gridsize is not evenly divisible by {} processes.'
              .format(gridsize, nprocs))
    shape = (gridsize//nprocs,  # Distributed dimension
             gridsize,
             2*(gridsize//2 + 1), # Padded dimension
             )
    # If no slab grid is passed, fetch a buffer of the right shape
    if isinstance(slab_or_buffer_name, (int, str)):
        buffer_name = slab_or_buffer_name
        if prepare_fft:
            slab = get_fftw_slab(gridsize, buffer_name)
        else:
            slab = get_buffer(shape, buffer_name)
    else:
        slab = slab_or_buffer_name
        if asarray(slab).shape != shape:
            abort('The slab and domain grid passed to slab_decompose '
                  'have incompatible shapes: {}, {}.'
                  .format(asarray(slab).shape, asarray(domain_grid).shape)
                  )
    # Compute needed communication variables
    (N_domain2slabs_communications,
     domain2slabs_recvsend_ranks,
     slabs2domain_sendrecv_ranks,
     domain_sendrecv_i_start,
     domain_sendrecv_i_end,
     slab_sendrecv_j_start,
     slab_sendrecv_j_end,
     slab_sendrecv_k_start,
     slab_sendrecv_k_end,
     ) = prepare_decomposition(domain_grid, slab)
    # Communicate the domain grid to the slabs
    for ℓ in range(N_domain2slabs_communications):
        # Send part of the local domain
        # grid to the corresponding process.
        if ℓ < slabs2domain_sendrecv_ranks.shape[0]:
            # A non-blocking send is used, because the communication
            # is not pairwise.
            # Since the slabs extend throughout the entire yz-plane,
            # we should send the entire yz-part of domain
            # (excluding ghost and pseudo points).
            request = smart_mpi(domain_grid_noghosts[
                                    domain_sendrecv_i_start[ℓ]:domain_sendrecv_i_end[ℓ],
                                    :ℤ[domain_grid_noghosts.shape[1] - 1],
                                    :ℤ[domain_grid_noghosts.shape[2] - 1],
                                                     ],
                                dest=slabs2domain_sendrecv_ranks[ℓ],
                                mpifun='Isend')
        # The lower ranks storing the slabs receives the message.
        # In the x-dimension, the slabs are always thinner than (or at
        # least as thin as) the domain.
        if ℓ < domain2slabs_recvsend_ranks.shape[0]:
            smart_mpi(slab[:,
                           slab_sendrecv_j_start[ℓ]:slab_sendrecv_j_end[ℓ],
                           slab_sendrecv_k_start[ℓ]:slab_sendrecv_k_end[ℓ]],
                      source=domain2slabs_recvsend_ranks[ℓ],
                      mpifun='Recv')
        # Wait for the non-blockind send to be complete before
        # continuing. Otherwise, data in the send buffer - which is
        # still in use by the non-blocking send - might get overwritten
        # by the next (non-blocking) send.
        request.wait()
    return slab

# Function that returns a slab decomposed grid,
# allocated by FFTW.
@cython.pheader(# Arguments
                gridsize='Py_ssize_t',
                buffer_name='object',  # int or str
                nullify='bint',
                # Locals
                as_expected='bint',
                fftw_plans_index='Py_ssize_t',
                fftw_struct='fftw_return_struct',
                plan_backward='fftw_plan',
                plan_forward='fftw_plan',
                rigor='str',
                rigor_final='str',
                shape='tuple',
                slab='double[:, :, ::1]',
                slab_address='Py_ssize_t',
                slab_ptr='double*',
                slab_size_i='Py_ssize_t',
                slab_size_j='Py_ssize_t',
                slab_start_i='Py_ssize_t',
                slab_start_j='Py_ssize_t',
                wisdom_filename='str',
                returns='double[:, :, ::1]',
                )
def get_fftw_slab(gridsize, buffer_name=0, nullify=False):
    global fftw_plans_size, fftw_plans_forward, fftw_plans_backward
    # If this slab has already been constructed, fetch it
    slab = slabs.get((gridsize, buffer_name))
    if slab is not None:
        if nullify:
            slab[...] = 0
        return slab
    # Checks on the passed gridsize
    if gridsize%nprocs != 0:
        abort('A gridsize of {} was passed to the get_fftw_slab function. '
              'This gridsize is not evenly divisible by {} processes.'
              .format(gridsize, nprocs))
    shape = (int(gridsize//nprocs),    # Distributed dimension
             int(gridsize),
             int(2*(gridsize//2 + 1)), # Padded dimension
             )
    # In pure Python mode we use NumPy, which really means that there
    # is no needed preparations. In compiled mode we use FFTW,
    # which means that the grid and its plans must be prepared.
    if not cython.compiled:
        slab = empty(shape, dtype=C2np['double'])
    else:
        # Determine what FFTW rigor to use.
        # The rigor to use will be stored as rigor_final.
        if master:
            if fftw_wisdom_reuse:
                for rigor in fftw_wisdom_rigors:
                    wisdom_filename = ('.fftw_wisdom_gridsize={}_nprocs={}_rigor={}'
                                       .format(gridsize, nprocs, rigor))
                    # At least be as rigorous as defined by
                    # the fftw_wisdom_rigor user parameter.
                    if rigor == fftw_wisdom_rigor:
                        break
                    # Use a better rigor if wisdom already exist
                    if os.path.isfile(wisdom_filename):
                        break
                rigor_final = rigor
            else:
                rigor_final = fftw_wisdom_rigor
            # If less rigorous wisdom exists for the same problem,
            # delete it.
            for rigor in reversed(fftw_wisdom_rigors):
                if rigor == rigor_final:
                    break
                wisdom_filename = ('.fftw_wisdom_gridsize={}_nprocs={}_rigor={}'
                                   .format(gridsize, nprocs, rigor))
                if os.path.isfile(wisdom_filename):
                    os.remove(wisdom_filename)
        rigor_final = bcast(rigor_final if master else None)
        wisdom_filename = ('.fftw_wisdom_gridsize={}_nprocs={}_rigor={}'
                           .format(gridsize, nprocs, rigor_final))
        # Initialize fftw_mpi, allocate the grid, initialize the
        # local grid sizes and start indices and do FFTW planning.
        # All this is handled by fftw_setup from fft.c.
        if master:
            reuse = (fftw_wisdom_reuse and os.path.isfile(wisdom_filename))
        reuse = bcast(reuse if master else None)
        if not reuse:
            masterprint('Acquiring FFTW wisdom ({}) for grid of linear size {} on {} {} ...'
                        .format(rigor_final, gridsize, nprocs,
                                'processes' if nprocs > 1 else 'process')
                        )
        fftw_struct = fftw_setup(gridsize, gridsize, gridsize,
                                 bytes(rigor_final, encoding='ascii'),
                                 reuse)
        if not reuse:
            masterprint('done')            
        # Unpack every variable from fftw_struct
        # and compare to expected values.
        slab_size_i   = int(fftw_struct.gridsize_local_i)
        slab_size_j   = int(fftw_struct.gridsize_local_j)
        slab_start_i  = int(fftw_struct.gridstart_local_i)
        slab_start_j  = int(fftw_struct.gridstart_local_j)
        plan_forward  = fftw_struct.plan_forward
        plan_backward = fftw_struct.plan_backward
        slab_ptr      = fftw_struct.grid
        as_expected = True
        if (   slab_size_i  != ℤ[shape[0]]
            or slab_size_j  != ℤ[shape[0]]
            or slab_start_i != ℤ[shape[0]*rank]
            or slab_start_j != ℤ[shape[0]*rank]
            ):
            as_expected = False
            warn(f'FFTW has distributed a slab of gridsize {gridsize} differently '
                 f'from what was expected on rank {rank}:\n'
                 f'    slab_size_i  = {slab_size_i}, expected {shape[0]},\n'
                 f'    slab_size_j  = {slab_size_j}, expected {shape[0]},\n'
                 f'    slab_start_i = {slab_start_i}, expected {shape[0]*rank},\n'
                 f'    slab_start_j = {slab_start_j}, expected {shape[0]*rank},\n'
                 )
        as_expected = allreduce(as_expected, op=MPI.LOR)
        if not as_expected:
            abort('Refusing to carry on with this non-expected decomposition.')
        # Wrap the slab pointer in a memory view. Looping over this
        # memory view should be done as noted in fft.c, but use
        # slab[i, j, k] when in real space and slab[j, i, k]
        # when in Fourier space.
        slab = cast(slab_ptr, 'double[:shape[0], :shape[1], :shape[2]]')
        # Store the plans for this slab in the global
        # fftw_plans_forward and fftw_plans_backward arrays.
        fftw_plans_index = fftw_plans_size
        fftw_plans_size += 1
        fftw_plans_forward  = realloc(fftw_plans_forward , fftw_plans_size*sizeof('fftw_plan'))
        fftw_plans_backward = realloc(fftw_plans_backward, fftw_plans_size*sizeof('fftw_plan'))
        fftw_plans_forward [fftw_plans_index] = plan_forward
        fftw_plans_backward[fftw_plans_index] = plan_backward
        # Insert mapping from the slab to the index of its plans
        # in the global fftw_plans_forward and fftw_plans_backward
        # arrays, into the global fftw_plans_mapping dict.
        slab_address = cast(cython.address(slab[0, 0, 0]), 'Py_ssize_t')
        fftw_plans_mapping[slab_address] = fftw_plans_index
    # Store and return this slab
    slabs[gridsize, buffer_name] = slab
    if nullify:
        slab[...] = 0
    return slab
# Tuple of all possible FFTW rigor levels, in descending order
cython.declare(fftw_wisdom_rigors='tuple')
fftw_wisdom_rigors = ('exhaustive', 'patient', 'measure', 'estimate')
# Cache storing slabs. The keys have the format (gridsize, buffer_name).
cython.declare(slabs='dict')
slabs = {}
# Arrays of FFTW plans
cython.declare(fftw_plans_size='Py_ssize_t',
               fftw_plans_forward ='fftw_plan*',
               fftw_plans_backward='fftw_plan*',
               )
fftw_plans_size = 0
fftw_plans_forward  = malloc(fftw_plans_size*sizeof('fftw_plan'))
fftw_plans_backward = malloc(fftw_plans_size*sizeof('fftw_plan'))
# Mapping from memory addreses of slabs to indices in
# fftw_plans_forward and fftw_plans_backward.
cython.declare(fftw_plans_mapping='dict')
fftw_plans_mapping = {}

# Function performing Fourier transformations of slab decomposed grids
@cython.header(# Arguments
               slab='double[:, :, ::1]',
               direction='str',
               # Locals
               fftw_plans_index='Py_ssize_t',
               slab_address='Py_ssize_t',
               )
def fft(slab, direction):
    """Fourier transform the given slab decomposed grid.
    For a forwards transformation from real to Fourier space, supply
    direction='forward'. Note that this is an unnormalized transform,
    as defined by FFTW. To do the normalization, divide all elements of
    the slab by gridsize**3, where gridsize is the linear gridsize
    of the cubic grid.
    For a backwards transformation from Fourier to real space, supply
    direction='backward'. Here, no further normalization is needed,
    as defined by FFTW.

    In pure Python, NumPy is used to carry out the Fourier transforms.
    To emulate the effects of FFTW perfectly, a lot of extra steps
    are needed.
    """
    if not direction in ('forward', 'backward'):
        abort('fft was called with the direction "{}", which is neither "forward" nor "backward".'
              .format(direction))
    if not cython.compiled:
        # The pure Python FFT implementation is serial.
        # Every process computes the entire FFT of the temporary
        # varaible grid_global_pure_python.
        slab_size_i = slab_size_j = slab.shape[0]
        slab_start_i = slab_size_i*rank
        slab_start_j = slab_size_j*rank
        gridsize = slab.shape[1]
        gridsize_padding = slab.shape[2]
        grid_global_pure_python = empty((gridsize, gridsize, gridsize_padding))
        Allgatherv(slab, grid_global_pure_python)
        if direction == 'forward':
            # Delete the padding on last dimension
            for i in range(gridsize_padding - gridsize):
                grid_global_pure_python = np.delete(grid_global_pure_python, -1, axis=2)
            # Do real transform via NumPy
            grid_global_pure_python = np.fft.rfftn(grid_global_pure_python)
            # FFTW transposes the first two dimensions
            grid_global_pure_python = grid_global_pure_python.transpose([1, 0, 2])
            # FFTW represents the complex array by doubles only
            tmp = empty((gridsize, gridsize, gridsize_padding))
            for i in range(gridsize_padding):
                if i % 2:
                    tmp[:, :, i] = grid_global_pure_python.imag[:, :, i//2]
                else:
                    tmp[:, :, i] = grid_global_pure_python.real[:, :, i//2]
            grid_global_pure_python = tmp
            # As in FFTW, distribute the slabs along the y-dimension
            # (which is the first dimension now, due to transposing).
            slab[...] = grid_global_pure_python[slab_start_j:(slab_start_j + slab_size_j), :, :] 
        elif direction == 'backward':
            # FFTW represents the complex array by doubles only.
            # Go back to using complex entries.
            tmp = zeros((gridsize, gridsize, gridsize_padding//2), dtype='complex128')
            for i in range(gridsize_padding):
                if i % 2:
                    tmp[:, :, i//2] += 1j*grid_global_pure_python[:, :, i]
                else:
                    tmp[:, :, i//2] += grid_global_pure_python[:, :, i]
            grid_global_pure_python = tmp
            # FFTW transposes the first
            # two dimensions back to normal.
            grid_global_pure_python = grid_global_pure_python.transpose([1, 0, 2])
            # Do real inverse transform via NumPy
            grid_global_pure_python = np.fft.irfftn(grid_global_pure_python, s=[gridsize]*3)
            # Remove the autoscaling provided by NumPy
            grid_global_pure_python *= gridsize**3
            # Add padding on last dimension, as in FFTW
            padding = empty((gridsize,
                             gridsize,
                             gridsize_padding - gridsize,
                             ))
            grid_global_pure_python = np.concatenate((grid_global_pure_python, padding), axis=2)
            # As in FFTW, distribute the slabs along the x-dimension
            slab[...] = grid_global_pure_python[slab_start_i:(slab_start_i + slab_size_i), :, :]
    else:  # Compiled mode
        # Look up the index of the FFTW plans for the passed slab.
        slab_address = cast(cython.address(slab[0, 0, 0]), 'Py_ssize_t')
        fftw_plans_index = fftw_plans_mapping[slab_address]
        # Look up the plan and let FFTW do the Fourier transformation
        if direction == 'forward':
            fftw_execute(fftw_plans_forward[fftw_plans_index])
        elif direction == 'backward':
            fftw_execute(fftw_plans_backward[fftw_plans_index])

# Function for deallocating a slab and its plans, allocated by FFTW
@cython.header(# Arguments
               gridsize='Py_ssize_t',
               buffer_name='object',  # int or str
               # Locals
               fftw_plans_index='Py_ssize_t',
               plan_forward='fftw_plan',
               plan_backward='fftw_plan',
               slab='double[:, :, ::1]',
               slab_ptr='double*',
               )
def free_fftw_slab(gridsize, buffer_name):
    # Fetch the slab from the slab cache and remove it
    slab = slabs.pop((gridsize, buffer_name))
    # Grab pointer to the slab
    slab_ptr = cython.address(slab[0, 0, 0])
    # Look up the index of the FFTW plans for the passed slab
    # and use this to look up the plans.
    slab_address = cast(slab_ptr, 'Py_ssize_t')
    fftw_plans_index = fftw_plans_mapping[slab_address]
    plan_forward  = fftw_plans_forward[fftw_plans_index]
    plan_backward = fftw_plans_backward[fftw_plans_index]
    # Let FFTW do the cleanup
    fftw_clean(slab_ptr, plan_forward, plan_backward)
    # Note that the arrays fftw_plans_forward and fftw_plans_backward
    # as well as the dict fftw_plans_mapping have not been altered.
    # Thus, accessing the pointers in fftw_plans_forward or
    # fftw_plans_backward for the newly freed plans will cause a
    # segmentation fault. As this should not ever happen, we leave
    # these as is.

# Function for checking that the slabs satisfy the required symmetry
# of a Fourier transformed real field.
@cython.pheader(# Arguments
                slab='double[:, :, ::1]',
                rel_tol='double',
                abs_tol='double',
                # Locals
                bad_pairs='set',
                global_slab='double[:, :, ::1]',
                gridsize='Py_ssize_t',
                i='Py_ssize_t',
                i_conj='Py_ssize_t',
                im1='double',
                im2='double',
                j='Py_ssize_t',
                j_conj='Py_ssize_t',
                j1='Py_ssize_t',
                j2='Py_ssize_t',
                k='Py_ssize_t',
                plane='int',
                re1='double',
                re2='double',
                slab_jik='double*',
                slab_jik_conj='double*',
                slave='int',
                t1='tuple',
                t2='tuple',
                )
def slabs_check_symmetry(slab, rel_tol=1e-9, abs_tol=machine_ϵ):
    """This function will go through the slabs and check whether they
    satisfy the symmetry condition of the Fourier transform of a
    real-valued field, namely
    field(kx, ky, kz) = field(-kx, -ky, -kz)*,
    where * is complex conjugation.
    A warning will be printed for each pair of grid points
    that does not satisfy this symmetry.
    The check is carried out in a rather ineffective manner,
    so you should not call this function during a real simulation.
    """
    masterprint('Checking the symmetry of the slabs ...')
    # Gather all slabs into global_slab on the master process
    gridsize = slab.shape[1]
    if nprocs == 1:
        global_slab = slab
    else:
        if master:
            global_slab = empty((gridsize, gridsize, slab.shape[2]), dtype=C2np['double'])
            global_slab[:slab.shape[0], :, :] = slab[...]
            for slave in range(1, nprocs):
                j1 = slab.shape[0]*slave
                j2 = j1 + slab.shape[0]
                smart_mpi(global_slab[j1:j2, :, :], source=slave, mpifun='recv')
        else:
            smart_mpi(slab, dest=master_rank, mpifun='send')
            return
    # Loop through the complete j-dimension
    bad_pairs = set()
    for j in range(gridsize):
        j_conj = 0 if j == 0 else gridsize - j
        # Loop through the complete i-dimension
        for i in range(gridsize):
            i_conj = 0 if i == 0 else gridsize - i
            # Loop through the lower (kk = 0)
            # and upper (kk = slab_size_padding//2, where
            # slab_size_padding = 2*(gridsize//2 + 1)) xy planes only.
            for plane in range(2):
                k = 0 if plane == 0 else slab.shape[2] - 2
                # Pointer to the [j, i, k]'th element and to its
                # conjugate.
                # The complex numbers is then given as e.g.
                # Re = slab_jik[0], Im = slab_jik[1].
                slab_jik      = cython.address(global_slab[j     , i     , k:])
                slab_jik_conj = cython.address(global_slab[j_conj, i_conj, k:])
                # Extract the two complex numbers,
                # which should be complex conjugates of each other
                # as required by the symmetry.
                re1 = slab_jik[0]
                im1 = slab_jik[1]
                re2 = slab_jik_conj[0]
                im2 = slab_jik_conj[1]
                # Check that the symmetry holds
                if i == i_conj and j == j_conj:
                    # Do not double count bad pairs
                    t1 = (i, j)
                    t2 = (i_conj, j_conj)
                    if (t1, t2) in bad_pairs or (t2, t1) in bad_pairs:
                        continue
                    bad_pairs.add((t1, t2))
                    bad_pairs.add((t2, t1))
                    # At origin of xy-plane.
                    # Here the value should be purely real.
                    if not isclose(im1, 0, rel_tol, abs_tol):
                        masterwarn(f'global_slab[{j}, {i}, {k}] = {complex(re1, im1)} ∉ ℝ',
                                   prefix='')
                elif (   not isclose(re1,  re2, rel_tol, abs_tol)
                      or not isclose(im1, -im2, rel_tol, abs_tol)
                      ):
                    # Do not double count bad pairs
                    t1 = (i, j)
                    t2 = (i_conj, j_conj)
                    if (t1, t2) in bad_pairs or (t2, t1) in bad_pairs:
                        continue
                    bad_pairs.add((t1, t2))
                    bad_pairs.add((t2, t1))
                    masterwarn(f'global_slab[{j}, {i}, {k}] = {complex(re1, im1)} \n'
                               '≠\n'
                               f'global_slab[{j_conj}, {i_conj}, {k}]* = {complex(re2, -im2)}',
                               prefix='')
    masterprint('done')

# Function for differentiating domain grids
@cython.pheader(# Arguments
                grid='double[:, :, ::1]',
                dim='int',
                h='double',
                buffer_or_buffer_name='object',  # double[:, :, ::1] or int or str
                order='int',
                direction='str',
                noghosts='bint',
                # Locals
                buffer='double[:, :, ::1]',
                buffer_i='Py_ssize_t',
                buffer_j='Py_ssize_t',
                buffer_k='Py_ssize_t',
                buffer_name='object',  # int or str
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
                shape='tuple',
                value='double',
                returns='double[:, :, ::1]',
                )
def diff_domain(grid, dim, h=1, buffer_or_buffer_name=0,
                order=4, direction='forward', noghosts=True):
    """This function differentiates a given grid along the dim
    dimension once. The passed grid must include psuedo and ghost
    points. The pseudo points will be differentiated along with the
    actual grid points. To achieve proper units, the physical grid
    spacing may be specified as h. If not given, grid units (h == 1)
    are used. The buffer_or_buffer_name argument can be a buffer to
    store the results, or alternatively the name of a buffer (retrieved
    via communication.get_buffer) as an int or str. If a buffer is
    supplied, the result of the differentiations will be added to this
    buffer. If a buffer should be fetched automatically, this will be
    nullify before the differentiation. Note that the buffer has to be
    contiguous. If noghosts is True, the passed buffer (if any) must not
    contain ghost points, and the returned grid will not contain ghost
    points either, though it will be contiguous. If the supplied buffer
    include ghost points, or if the returned grid should contain ghost
    points, specify noghosts=True. In this case, ghost points will be
    populated with the correct values.
    Note that a grid cannot be differentiated in-place by passing the
    grid as both the first and third argument, as the differentiation
    of each grid point requires information from the original
    (non-differentiated) grid.
    The optional order argument specifies the order of accuracy of the
    differentiation (the number of neighboring grid points used to
    approximate the derivative).
    For odd orders, the differentiation cannot be symmetric. Set the
    direction argument to either 'forward' or 'backward' to choose from
    which direction one additional grid point should be used.
    """
    # Sanity checks on input
    if master:
        if dim not in (0, 1, 2):
            abort('The dim argument should be ∈ {0, 1, 2}')
        if order not in (1, 2, 4):
            abort('The order argument should be ∈ {1, 2, 4}')
        if direction not in ('forward', 'backward'):
            abort('The direction argument should be ∈ {\'forward\', \'backward\'}')
    # If no buffer is supplied, fetch the buffer with the name
    # given by buffer_or_buffer_name.
    if isinstance(buffer_or_buffer_name, (int, str)):
        if noghosts:
            shape = tuple([grid.shape[dim] - 2*2 for dim in range(3)])
        else:
            shape = asarray(grid).shape
        buffer_name = buffer_or_buffer_name
        buffer = get_buffer(shape, buffer_name, nullify=True)
    else:
        buffer = buffer_or_buffer_name
    # Do the differentiation and add the results to the buffer
    for         i in range(2, ℤ[grid.shape[0] - 2]):
        for     j in range(2, ℤ[grid.shape[1] - 2]):
            for k in range(2, ℤ[grid.shape[2] - 2]):
                with unswitch:
                    # Differentiate along x
                    if dim == 0 and order == 1 and direction == 'forward':
                        value = ℝ[1/h]*(+ grid[ℤ[i + 1], j, k]
                                        - grid[  i     , j, k]
                                        )
                    elif dim == 0 and order == 1 and direction == 'backward':
                        value = ℝ[1/h]*(+ grid[  i     , j, k]
                                        - grid[ℤ[i - 1], j, k]
                                        )
                    elif dim == 0 and order == 2:
                        value = ℝ[1/(2*h)]*(+ grid[ℤ[i + 1], j, k]
                                            - grid[ℤ[i - 1], j, k]
                                            )
                    elif dim == 0 and order == 4:
                        value = (+ ℝ[2/(3*h)] *(+ grid[ℤ[i + 1], j, k]
                                                - grid[ℤ[i - 1], j, k]
                                                )
                                 - ℝ[1/(12*h)]*(+ grid[ℤ[i + 2], j, k]
                                                - grid[ℤ[i - 2], j, k]
                                                )
                                 )
                    # Differentiate along y
                    elif dim == 1 and order == 1 and direction == 'forward':
                        value = ℝ[1/h]*(+ grid[i, ℤ[j + 1], k]
                                        - grid[i,   j     , k]
                                        )
                    elif dim == 1 and order == 1 and direction == 'backward':
                        value = ℝ[1/h]*(+ grid[i,   j     , k]
                                        - grid[i, ℤ[j - 1], k]
                                        )
                    elif dim == 1 and order == 2:
                        value = ℝ[1/(2*h)]*(+ grid[i, ℤ[j + 1], k]
                                            - grid[i, ℤ[j - 1], k]
                                            )
                    elif dim == 1 and order == 4:
                        value = (+ ℝ[2/(3*h)] *(+ grid[i, ℤ[j + 1], k]
                                                - grid[i, ℤ[j - 1], k]
                                                )
                                 - ℝ[1/(12*h)]*(+ grid[i, ℤ[j + 2], k]
                                                - grid[i, ℤ[j - 2], k]
                                                )
                                 )
                    # Differentiate along z
                    elif dim == 2 and order == 1 and direction == 'forward':
                        value = ℝ[1/h]*(+ grid[i, j, k + 1]
                                        - grid[i, j, k    ]
                                        )
                    elif dim == 2 and order == 1 and direction == 'backward':
                        value = ℝ[1/h]*(+ grid[i, j, k    ]
                                        - grid[i, j, k - 1]
                                        )
                    elif dim == 2 and order == 2:
                        value = ℝ[1/(2*h)]*(+ grid[i, j, k + 1]
                                            - grid[i, j, k - 1]
                                            )
                    elif dim == 2 and order == 4:
                        value = (+ ℝ[2/(3*h)] *(+ grid[i, j, k + 1]
                                                - grid[i, j, k - 1]
                                                )
                                 - ℝ[1/(12*h)]*(+ grid[i, j, k + 2]
                                                - grid[i, j, k - 2]
                                                )
                                 )
                    else:
                        abort('Domain differentiation with dim = {}, order = {} '
                              'and direction = {} is not implemented'
                              .format(dim, order, direction)
                              )
                        value = 0  # Just to please the compiler
                # Update the buffer with the result
                # of the differentiation.
                with unswitch:
                    if noghosts:
                        buffer[ℤ[i - 2], ℤ[j - 2], k - 2] += value
                    else:
                        buffer[i, j, k] += value
    # If the buffer contains ghost points, these have not themselves
    # been replaced with differentiated values. Now populate these
    # ghost points with copies of their corresponding actual points.
    if not noghosts:
        communicate_domain(buffer, mode='populate')
    return buffer


# Function pointer types used in this module
pxd = """
ctypedef double* (*func_dstar_ddd)(double, double, double)
"""

# Import declarations from fft.c
pxd = """
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
                                  char*     rigor,
                                  bint fftw_wisdom_reuse)
    void fftw_execute(fftw_plan plan)
    void fftw_clean(double* grid, fftw_plan plan_forward,
                                  fftw_plan plan_backward)
"""
