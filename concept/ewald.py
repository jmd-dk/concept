# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
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

# Cython function for computing Ewald correction
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Argument
               x='double',
               y='double',
               z='double',
               # Locals
               dim='Py_ssize_t',
               dist='double',
               dist_x='double',
               dist_y='double',
               dist_z='double',
               dist2='double',
               force='double*',
               force_x='double',
               force_y='double',
               force_z='double',
               kx='double',
               ky='double',
               kz='double',
               r3='double',
               scalarpart='double',
               sumindex_x='int',
               sumindex_y='int',
               sumindex_z='int',
               )
@cython.returns('double*')
def summation(x, y, z):
    """ This function performs the Ewald summation given the distance
    x, y, z between two particles, normalized so that 0 <= |x|, |y|, |z| < 1
    (corresponding to boxsize = 1). The equation being solved corresponds to
    (8) in Ralf Klessen's 'GRAPESPH with Fully Periodic Boundary
    Conditions: Fragmentation of Molecular Clouds', though it is written
    without the normalization. The actual Ewald force is then given by
    force/boxsize**2. What is returned is the Ewald correction,
    corresponding to (9) in the mentioned paper. That is, the return value is
    not the total force, but the force from all particle images except the
    nearest one. Note that this nearest image need not be the actual particle.
    """

    # The Ewald force vector and its components
    force = vector
    force_x = force_y = force_z = 0
    # The image is on top of the particle: No force
    if x == 0 and y == 0 and z == 0:
        force[0] = 0
        force[1] = 0
        force[2] = 0
        return force
    # Remove the direct force, as we are interested in the correction only
    r3 = (x**2 + y**2 + z**2)**1.5
    force_x += x/r3
    force_y += y/r3
    force_z += z/r3
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
                scalarpart = -dist**(-3)*(erfc(dist*recp_2rs)
                                          + dist*recp_sqrt_π_rs
                                          * exp(dist2*minus_recp_4rs2))
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
                kx = two_π*sumindex_x
                ky = two_π*sumindex_y
                kz = two_π*sumindex_z
                k2 = kx**2 + ky**2 + kz**2
                scalarpart = minus_4π/k2*exp(-k2*rs2)*sin(kx*x + ky*y + kz*z)
                force_x += kx*scalarpart
                force_y += ky*scalarpart
                force_z += kz*scalarpart
    # Pack and return Ewald force
    force[0] = force_x
    force[1] = force_y
    force[2] = force_z
    return force


# Master function of this module. Returns the Ewald force correction.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               x='double',
               y='double',
               z='double',
               # Locals
               dim='size_t',
               force='double*',
               isnegative_x='bint',
               isnegative_y='bint',
               isnegative_z='bint',
               r3='double',
               )
@cython.returns('double*')
def ewald(x, y, z):
    """This function performs a look up of the Ewald correction to the
    fully periodic gravitational force (corresponding to 1/r**2) on a particle
    due to some other particle at a position (x, y, z) relative to the first
    particle. It is important that the passed coordinates are of the nearest
    periodic image of the other particle, and not necessarily of the particle
    itself. This means that 0 <= |x|, |y|, |z| < boxsize/2. The returned value
    is thus the force arising on the first particle due to all periodic images
    of the second particle, except for the nearest one.
    """

    # Only the positive octant of the box is tabulated. Flip the sign of the
    # coordinates so that they reside inside this octant.
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
    # Look up Ewald force and do a CIC interpolation. Since the coordinates
    # is to the nearest image, they must be scaled by 2/boxsize to reside
    # in the range 0 <= x, y, z < 1.
    force = CIC_grid2coordinates_vector(grid, x*two_recp_boxsize,
                                              y*two_recp_boxsize,
                                              z*two_recp_boxsize,
                                        )
    # Put the sign back in for negative input
    if isnegative_x:
        force[0] *= -1
    if isnegative_y:
        force[1] *= -1
    if isnegative_z:
        force[2] *= -1
    # The tabulated force is for a unit box. Du rescaling
    for dim in range(3):
        force[dim] /= boxsize2
    return force


# Set parameters for the Ewald summation at import time
cython.declare(h_lower='int',
               h_upper='int',
               maxdist='double',
               maxh='double',
               maxh2='double',
               minus_recp_4rs2='double',
               n_lower='int',
               n_upper='int',
               recp_2rs='double',
               recp_sqrt_π_rs='double',
               rs='double',
               rs2='double',
               )
# The values chosen match those listed in the article mentioned in the
# docstring of the summation function. These are also those used
# (in effect) in GADGET2.
rs = 1/4  # Corresponds to alpha = 2
maxdist = 3.6
maxh2 = 10
# Derived constants
maxh = sqrt(maxh2)
h_lower = int(-maxh)  # GADGET: -4 (also the case here for maxh2=10)
h_upper = int(maxh) + 1  # GADGET: 5 (also the case here for maxh2=10)
minus_recp_4rs2 = -1/(4*rs**2)
n_lower = int(-(maxdist + 1))  # GADGET: -4 (also the case here for maxdist=3.6)
n_upper = int(maxdist + 1) + 1  # GADGET: 5 (also the case here for maxdist=3.6)
recp_2rs = 1/(2*rs)
recp_sqrt_π_rs = 1/(sqrt_π*rs)
rs2 = rs**2
# Initialize the grid at import time, if Ewald summation is to be used
cython.declare(i='int',
               p='str',
               filepath='str',
               grid='double[:, :, :, ::1]',
               )
if 'PP' in kick_algorithms.values() and use_Ewald:
    filepath = paths['concept_dir'] + '/' + ewald_file
    if isfile(filepath):
        # Ewald grid already tabulated. Load it
        with h5py.File(filepath, mode='r') as hdf5_file:
            grid = hdf5_file['data'][...]
    else:
        # No tabulated Ewald grid found. Compute it.The factor 0.5 ensures
        # that only the first octant of the box is tabulated
        if master:
            print('Tabulating Ewald grid of linear size '
                  + str(ewald_gridsize))
        grid = tabulate_vectorfield(ewald_gridsize,
                                    summation,
                                    0.5/(ewald_gridsize - 1),
                                    filepath,
                                    )
