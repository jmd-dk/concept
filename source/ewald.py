# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from mesh import CIC_grid2coordinates_vector, tabulate_vectorfield
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from mesh cimport CIC_grid2coordinates_vector, tabulate_vectorfield
    """

# Imports and definitions common to pure Python and Cython
from os.path import isfile
from sys import path


# Adjustable parameters for the Ewald summation. The values chosen match those listed in the article
# mentioned in the docstring of the summation function. These are also those used (in effect) in Gadget2.
cython.declare(rs='double')
cython.declare(maxdist='double')
cython.declare(maxh2='double')
cython.declare(maxh='double')
cython.declare(n_lower='int')
cython.declare(n_upper='int')
cython.declare(h_lower='int')
cython.declare(h_upper='int')
rs = 1/4  # Corresponds to alpha = 2
maxdist = 3.6
maxh2 = 10
n_lower = -int(maxdist - 1) - 1
n_upper = int(maxdist + 1) + 1
maxh = sqrt(maxh2)
h_lower = int(-maxh)
h_upper = int(maxh) + 1
# Further constants for the Ewald summation
cython.declare(rs2='double')
cython.declare(reciprocal_2rs='double')
cython.declare(reciprocal_sqrt_pi_rs='double')
cython.declare(minus_reciprocal_4rs2='double')
cython.declare(two_pi='double')
cython.declare(minus_4pi='double')
rs2 = rs**2
reciprocal_2rs = 1/(2*rs)
reciprocal_sqrt_pi_rs = 1/(sqrt(pi)*rs)
minus_reciprocal_4rs2 = -1/(4*rs**2)
two_pi = 2*pi
minus_4pi = -4*pi


# Cython function for computing Ewald correction
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               x='double',
               y='double',
               z='double',
               # Locals
               force='double*',
               force_x='double',
               force_y='double',
               force_z='double',
               sumindex_x='int',
               sumindex_y='int',
               sumindex_z='int',
               dim='Py_ssize_t',
               dist_x='double',
               dist_y='double',
               dist_z='double',
               dist2='double',
               dist='double',
               scalarpart='double',
               kx='double',
               ky='double',
               kz='double',
               )
@cython.returns('double*')
def summation(x, y, z):
    """ This function performs the Ewald summation given the distance
    x, y, z between two particles, normalized so that 0 <= |x|, |y|, |z| < 1
    (corresponding to boxsize = 1). The equation being solved corresponds to
    (8) in Ralf Klessen's 'GRAPESPH with Fully Periodic Boundary
    Conditions: Fragmentation of Molecular Clouds', though it is written
    without the normalization. The result is stored in 'force'. The
    actual Ewald force is then given by force/boxsize**2. This force will be
    the total force, not just the correction.
    """

    # The Ewald force vector and components
    force = vector
    force_x = force_y = force_z = 0
    # Two particles on top of each other: No force
    if x == y == z == 0:
        force[0] = force_x
        force[1] = force_y
        force[2] = force_z
        return force
    # The short range (real space) sum
    for sumindex_x in range(n_lower, n_upper):
        for sumindex_y in range(n_lower, n_upper):
            for sumindex_z in range(n_lower, n_upper):
                dist_x = x - sumindex_x
                dist_y = y - sumindex_y
                dist_z = z - sumindex_z
                dist2 = dist_x**2 + dist_y**2 + dist_z**2
                dist = sqrt(dist2)
                if dist > maxdist:
                    continue
                scalarpart = -dist**-3*(erfc(dist*reciprocal_2rs) +
                                        dist*reciprocal_sqrt_pi_rs*exp(dist2*minus_reciprocal_4rs2))
                force_x += dist_x*scalarpart
                force_y += dist_y*scalarpart
                force_z += dist_z*scalarpart
    # The long range (Fourier space) sum
    for sumindex_x in range(h_lower, h_upper):
        for sumindex_y in range(h_lower, h_upper):
            for sumindex_z in range(h_lower, h_upper):
                h2 = sumindex_x**2 + sumindex_y**2 + sumindex_z**2
                if h2 > maxh2 or h2 == 0:
                    continue
                kx = two_pi*sumindex_x
                ky = two_pi*sumindex_y
                kz = two_pi*sumindex_z
                k2 = kx**2 + ky**2 + kz**2
                scalarpart = minus_4pi/k2*exp(-k2*rs2)*sin(kx*x + ky*y + kz*z)
                force_x += kx*scalarpart
                force_y += ky*scalarpart
                force_z += kz*scalarpart
    # Pack and return Ewald force
    force[0] = force_x
    force[1] = force_y
    force[2] = force_z
    return force


# Cython master function of this module. Returns the Ewald force correction.
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               x='double',
               y='double',
               z='double',
               # Locals
               r3='double',
               force='double*',
               dim='size_t',
               )
@cython.returns('double*')
def ewald(x, y, z):
    """ Call this function to get the Ewald correction to the fully periodic
    gravitational force (corresponding to 1/r**2) between two particles
    seperated by x, y, z, inside a box of length boxsize.
    """

    # Look up Ewald force and do a CIC interpolation
    force = CIC_grid2coordinates_vector(grid, x/boxsize, y/boxsize, z/boxsize)
    # Scale the force to match the boxsize
    for dim in range(3):
        force[dim] /= boxsize2
    # Remove the direct force, leaving only the Ewald correction
    r3 = (x**2 + y**2 + z**2 + softening2)**1.5
    force[0] += x/r3
    force[1] += y/r3
    force[2] += z/r3
    return force

# Initialize the grid at import time
cython.declare(i='int',
               p='str',
               filepath='str',
               grid='double[:, :, :, ::1]',
               )
for i, p in enumerate(path):
    filepath = p + '/' + ewald_file
    if isfile(filepath):
        # Ewald grid already tabulated. Load it
        with h5py.File(filepath, mode='r') as hdf5_file:
            grid = hdf5_file['data'][...]
        break
    elif i == len(path) - 1:
        # No tabulated Ewald grid found. Compute it.The factor 0.5 ensures
        # that only the first octant of the box is tabulated
        if master:
            print('Tabulating Ewald grid of linear size', ewald_gridsize, '...')
        grid = tabulate_vectorfield(ewald_gridsize, summation, 0.5/ewald_gridsize, ewald_file)

