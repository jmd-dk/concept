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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
from mesh import diff_domain
from snapshot import save
cimport('from mesh import slab,                            '
        '                 slab_size_j,                     '
        '                 slab_start_j,                    '
        '                 slabs_IFFT,                      '
        '                 slabs2φ,                         '
        '                 φ, φ_noghosts,                   '
        '                 φ_start_i, φ_start_j, φ_start_k, '
        )
cimport('from species import Component')



# Function for computing the size of the initial perturbation of φ
# in Fourier space, given as power/k².
@cython.header(# Arguments
               k2='Py_ssize_t',
               returns='double',
               )
def get_perturbation_size(k2):
    """The argument k2 should be in grid units.
    The returned value is sqrt(power)/k² in physical units.
    The following power law is used for the power spectrum:
    power = 0.13*|k/Mpc⁻¹|**(-2.1)*Mpc³.
    This is the result of a crude fit to a snapshot
    with the following cosmology:
    H0 = 70*km/(s*Mpc)
    Ωr = 0
    Ωm = 0.3
    ΩΛ = 0.7
    a  = 0.02
    The convertion from k2 in grid units to k in physical units is
    k = sqrt(k2)*2*π/boxsize
    The entire convertion from k2 to sqrt(power)/k² is then
        sqrt(power)/k² = sqrt(0.124)*(2*π/boxsize)**(-2.1/2 - 2)*Mpc**(3/2 - 2.1/2)*k2**(-2.1/4 - 1)
    In addition, a factor of boxsize**(-3/2) is needed in order to get
    the units right after the inverse FFT.
    """
    # A step-by-step implementation (assuming unit_length == 'Mpc')
    # would look like this:
    #     k = sqrt(k2)*2*π/boxsize
    #     power = 0.13*k**(-2.1)
    #     val = sqrt(power)/k**2
    #     val *= boxsize**(-1.5)     # Needed to get the units right after the inverse FFT
    #     val *= 1/sqrt(2)           # WHERE DOES THIS COME FROM? !!!
    #     return val
    return ℝ[ 1/sqrt(2)        # WHERE DOES THIS COME FROM? !!!
             *boxsize**(-3/2)  # Needed to get the units right after the inverse FFT
             *sqrt(0.13)*(2*π/boxsize)**(-2.1/2 - 2)
             *units.Mpc**(3/2 - 2.1/2)
             ]*k2**ℝ[-2.1/4 - 1]

# Function for generating a global field of Guassian perturbations in
# real space, the size of which (in Fourier space) is determined by the
# get_perturbation_size function.
@cython.header(# Locals
               i='Py_ssize_t',
               j='Py_ssize_t',
               j_global='Py_ssize_t',
               k='Py_ssize_t',
               k2='Py_ssize_t',
               ki='Py_ssize_t',
               ki2_plus_kj2='Py_ssize_t',
               kj='Py_ssize_t',
               kj2='Py_ssize_t',
               kk='Py_ssize_t',
               perturbation_size='double',
               slab_jik='double*',
               returns='double[:, :, ::1]',
               )
def generate_Gaussian_perturbations():
    """The variable φ from the mesh module, normally used for the
    gravitational field, is used to store the perturbation field.
    """
    # Begin loop over slab. As the first and second dimensions
    # are transposed (we are working in Fourier space),
    # start with the j-dimension.
    for j in range(slab_size_j):
        # The j-component of the wave vector
        j_global = slab_start_j + j
        if j_global > φ_gridsize_half:
            kj = j_global - φ_gridsize
        else:
            kj = j_global
        kj2 = kj**2
        # Loop over the entire first dimension
        for i in range(φ_gridsize):
            # The i-component of the wave vector
            if i > φ_gridsize_half:
                ki = i - φ_gridsize
            else:
                ki = i
            ki2_plus_kj2 = ki**2 + kj2
            # Loop over the entire last dimension in steps of two,
            # as contiguous pairs of elements are the real and
            # imaginary part of the same complex number.
            for k in range(0, slab_size_padding, 2):
                # The k-component of the wave vector
                kk = k//2
                # The squared magnitude of the wave vector
                k2 = ki2_plus_kj2 + kk**2
                # Zero-division (and therefore 0**(<0)) is illegal in pure Python.
                # The global [0, 0, 0] element of the slabs will be set
                # later anyway.
                if not cython.compiled:
                    if k2 == 0:
                        continue
                # The size of the perturbation at this |k|
                perturbation_size = get_perturbation_size(k2)
                # Pointer to the [j, i, k]'th element of the slab.
                # The complex number is then given as
                # Re = slab_jik[0], Im = slab_jik[1].
                slab_jik = cython.address(slab[j, i, k:])
                # Fill slab with Gaussian perturbations
                slab_jik[0] = perturbation_size*random_gaussian(0, 1)
                slab_jik[1] = perturbation_size*random_gaussian(0, 1)
    # The global [0, 0, 0] element of the slabs should be zero
    if slab_start_j == 0:
        slab[0, 0, 0] = 0  # Real part
        slab[0, 0, 1] = 0  # Imag part
    # Fourier transform the slabs to coordinate space
    slabs_IFFT()
    # Communicate the potential stored in the slabs to φ
    slabs2φ()  # This also populates pseudo and ghost points
    return φ

# Function which constructs an almost homogeneous particle Component,
# with displacements (and corresponding momenta) from homogeneity
# given by the Zel'dovich approximation
@cython.header(# Arguments
               name='str',
               species='str',
               mass='double',
               a='double',
               # Locals
               D='double',
               Ḋ='double',
               N='Py_ssize_t',
               component='Component',
               dim='int',
               displacement='double',
               h='double',
               i='Py_ssize_t',
               i_global='Py_ssize_t',
               index='Py_ssize_t',
               j='Py_ssize_t',
               j_global='Py_ssize_t',
               k='Py_ssize_t',
               k_global='Py_ssize_t',
               meshbuf_mv='double[:, :, ::1]',
               mom='double**',
               mom_dim='double*',
               pos='double**',
               pos_dim='double*',
               pos_grid='double',
               ψ='double[:, :, ::1]',
               returns='Component',
               )
def zeldovich(name, species, mass=-1, a=-1):
    # There should be exactly one particle for each grid point in φ/ψ
    N = φ_gridsize**3
    # If no mass is passed, assume that this component
    # represents all matter in the universe. 
    if mass == -1:
        mass = ϱ_mbar*boxsize**3/N
        masterprint('A particle mass of {} m☉ has been assigned '
                    'to "{}", assuming that this component alone '
                    'constitues all of Ωm = {}.'
                    .format(significant_figures(mass/units.m_sun,
                                                3,
                                                fmt='unicode',
                                                scientific=True),
                            name,
                            Ωm,
                            )
                    )
    # If no a is passed, use a_begin
    if a == -1:
        a = a_begin
    # For now, the power spectrum and the growth factor are hard coded
    # for a = 0.02, meaning that only this value of the scale factor
    # is valid. !!!
    if not isclose(a, 0.02):
        abort('Everything in ic.py only works for a = 0.02')
    # Construct a Component instance
    component = Component(name, species, N, mass)
    # Check that the supplied species correspond to a particle
    # representation, as the Zeldovich approximation
    # only makes sense for particles.
    if component.representation != 'particles':
        abort('Cannot construct initial conditions for the "{}" component as this has a '
              'representation of "{}". Initial condition generation is only possible for '
              'particle components.'
              .format(component.name, component.representation)
              )
    # Determine the number of local particles.
    # The N = φ_gridsize**3 grid points should be exactly divisible by
    # the number of processes nprocs (if not, an exception should have
    # been thrown by the mesh module).
    component.N_local = N//nprocs
    if component.N_local*nprocs != N:
        abort('A φ_gridsize of {} means that there is {}³ = {} grid points, '
              'which cannot be evenly shared by {} processes'
              .format(φ_gridsize, φ_gridsize, N, nprocs)
              )
    # Compute the growth factor
    Ḋ = 13.86/units.Gyr  # Ḋ/D hardcoded for ΛCDM, a = 0.02 !!!
    D = 1                # Ḋ/D hardcoded for ΛCDM, a = 0.02 !!!
    # Generate the initial field of Gaussian perturbations
    ψ = generate_Gaussian_perturbations()
    # Resize the component according to the size
    # of the local particle data and extract data pointers.
    component.resize(component.N_local)
    pos = component.pos
    mom = component.mom
    # Populate local particle data (particles within the local domain)
    # using the Zeldovich approximation, one dimension at a time.
    h = boxsize/φ_gridsize  # Physical grid spacing of ψ
    for dim in range(3):
        pos_dim = pos[dim]
        mom_dim = mom[dim]
        # Differentiate ψ along the dim'th dimension
        meshbuf_mv = diff_domain(ψ, dim, h, order=2)
        index = 0
        for i in range(ℤ[φ_noghosts.shape[0] - 1]):
            # The global x coordinate of this grid point
            if dim == 0:
                i_global = φ_start_i + i
                pos_grid = i_global*boxsize/φ_gridsize
            for j in range(ℤ[φ_noghosts.shape[1] - 1]):
                # The global y coordinate of this grid point
                if dim == 1:
                    j_global = φ_start_j + j
                    pos_grid = j_global*boxsize/φ_gridsize
                for k in range(ℤ[φ_noghosts.shape[2] - 1]):
                    # The global z coordinate of this grid point
                    if dim == 2:
                        k_global = φ_start_k + k
                        pos_grid = k_global*boxsize/φ_gridsize
                    # Displace the position of particle
                    # at grid point (i, j, k).
                    displacement = meshbuf_mv[i, j, k]
                    pos_dim[index] = mod(pos_grid + displacement, boxsize)
                    # Assign momentum corresponding to the displacement
                    mom_dim[index] = displacement*ℝ[Ḋ/D*mass*a**2]
                    index += 1
    return component
