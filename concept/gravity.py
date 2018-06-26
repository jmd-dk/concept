# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2018 Jeppe Mosgaard Dakin.
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
cimport('from communication import communicate_domain')
cimport('from communication import domain_size_x , domain_size_y , domain_size_z' )
cimport('from communication import domain_start_x, domain_start_y, domain_start_z')
cimport('from ewald import ewald')
cimport('from mesh import CIC_grid2grid, CIC_scalargrid2coordinates')



# Function implementing pairwise gravity
@cython.header(# Arguments
               component_1='Component',
               component_2='Component',
               rank_2='int',
               ᔑdt=dict,
               local='bint',
               mutual='bint',
               extra_args=dict,
               # Locals
               N_1='Py_ssize_t',
               N_2='Py_ssize_t',
               force_ij='double*',
               forcex_ij='double',
               forcey_ij='double',
               forcez_ij='double',
               i='Py_ssize_t',
               j='Py_ssize_t',
               j_start='Py_ssize_t',
               mass_1='double',
               mass_2='double',
               momx_1='double*',
               momx_2='double*',
               momy_1='double*',
               momy_2='double*',
               momz_1='double*',
               momz_2='double*',
               only_short_range='bint',
               periodic='bint',
               posx_1='double*',
               posx_2='double*',
               posy_1='double*',
               posy_2='double*',
               posz_1='double*',
               posz_2='double*',
               r='double',
               r_scaled='double',
               r3='double',
               shortrange_fac='double',
               softening_1='double',
               softening_2='double',
               x_ji='double',
               xi='double',
               y_ji='double',
               yi='double',
               z_ji='double',
               zi='double',
               Δmomx_2='double*',
               Δmomx_ij='double',
               Δmomy_2='double*',
               Δmomy_ij='double',
               Δmomz_2='double*',
               Δmomz_ij='double',
               returns='void',
               )
def gravity_pairwise(component_1, component_2, rank_2, ᔑdt, local, mutual, extra_args):
    if component_1.representation != 'particles' or component_2.representation != 'particles':
        abort('gravity_pairwise is only implemented for particle components')
    # Extract extra arguments
    only_short_range = extra_args.get('only_short_range', False)
    periodic         = extra_args.get('periodic',         True)
    # Extract variables from the first (the local) component
    N_1 = component_1.N_local
    mass_1 = component_1.mass
    softening_1 = component_1.softening_length
    posx_1 = component_1.posx
    posy_1 = component_1.posy
    posz_1 = component_1.posz
    momx_1 = component_1.momx
    momy_1 = component_1.momy
    momz_1 = component_1.momz
    # Extract variables from the second (the external) component
    N_2 = component_2.N_local
    mass_2 = component_2.mass
    softening_2 = component_2.softening_length
    posx_2 = component_2.posx
    posy_2 = component_2.posy
    posz_2 = component_2.posz
    momx_2 = component_2.momx
    momy_2 = component_2.momy
    momz_2 = component_2.momz
    Δmomx_2 = component_2.Δmomx
    Δmomy_2 = component_2.Δmomy
    Δmomz_2 = component_2.Δmomz
    # Loop over all pairs of particles
    for i in range(N_1):
        xi = posx_1[i]
        yi = posy_1[i]
        zi = posz_1[i]
        # If the interaction is completely local,
        # make sure not to double count.
        with unswitch:
            if local:
                j_start = i + 1
            else:
                j_start = 0
        for j in range(j_start, N_2):
            # "Vector" from particle j to particle i
            x_ji = xi - posx_2[j]
            y_ji = yi - posy_2[j]
            z_ji = zi - posz_2[j]
            # Evaluate the gravitational force in one of three ways:
            # Just the short range force, the total force with Ewald
            # corrections or the total force without Ewald corrections.
            with unswitch:
                if only_short_range:
                    # Translate coordinates so they
                    # correspond to the nearest image.
                    if x_ji > ℝ[0.5*boxsize]:
                        x_ji -= boxsize
                    elif x_ji < ℝ[-0.5*boxsize]:
                        x_ji += boxsize
                    if y_ji > ℝ[0.5*boxsize]:
                        y_ji -= boxsize
                    elif y_ji < ℝ[-0.5*boxsize]:
                        y_ji += boxsize
                    if z_ji > ℝ[0.5*boxsize]:
                        z_ji -= boxsize
                    elif z_ji < ℝ[-0.5*boxsize]:
                        z_ji += boxsize
                    r = sqrt(x_ji**2 + y_ji**2 + z_ji**2 + ℝ[(0.5*(softening_1 + softening_2))**2])
                    r_scaled = r*ℝ[1/p3m_scale_phys]
                    shortrange_fac = (  r_scaled*ℝ[1/sqrt(π)]*exp(-0.25*r_scaled**2)
                                      + erfc(0.5*r_scaled))
                    forcex_ij = x_ji*ℝ[-shortrange_fac/r**3]
                    forcey_ij = y_ji*ℝ[-shortrange_fac/r**3]
                    forcez_ij = z_ji*ℝ[-shortrange_fac/r**3]
                elif periodic:
                    # Translate coordinates so they
                    # correspond to the nearest image.
                    if x_ji > ℝ[0.5*boxsize]:
                        x_ji -= boxsize
                    elif x_ji < ℝ[-0.5*boxsize]:
                        x_ji += boxsize
                    if y_ji > ℝ[0.5*boxsize]:
                        y_ji -= boxsize
                    elif y_ji < ℝ[-0.5*boxsize]:
                        y_ji += boxsize
                    if z_ji > ℝ[0.5*boxsize]:
                        z_ji -= boxsize
                    elif z_ji < ℝ[-0.5*boxsize]:
                        z_ji += boxsize
                    # The Ewald correction force for all
                    # images except the nearest one,
                    # which might not be the actual particle.
                    force_ij = ewald(x_ji, y_ji, z_ji)
                    # Add in the force from the particle's nearest image
                    r3 = (  x_ji**2 + y_ji**2 + z_ji**2
                          + ℝ[(0.5*(softening_1 + softening_2))**2])**1.5
                    forcex_ij = force_ij[0] - x_ji*ℝ[1/r3]
                    forcey_ij = force_ij[1] - y_ji*ℝ[1/r3]
                    forcez_ij = force_ij[2] - z_ji*ℝ[1/r3]
                else:
                    # The force from the actual particle,
                    # without periodic images.
                    r3 = (  x_ji**2 + y_ji**2 + z_ji**2
                          + ℝ[(0.5*(softening_1 + softening_2))**2])**1.5
                    forcex_ij = -x_ji*ℝ[1/r3]
                    forcey_ij = -y_ji*ℝ[1/r3]
                    forcez_ij = -z_ji*ℝ[1/r3]
            # Convert force on particle i from particle j
            # to momentum change of partcicle i due to particle j.
            Δmomx_ij = forcex_ij*ℝ[G_Newton*mass_1*mass_2*ᔑdt['a**(-1)']]
            Δmomy_ij = forcey_ij*ℝ[G_Newton*mass_1*mass_2*ᔑdt['a**(-1)']]
            Δmomz_ij = forcez_ij*ℝ[G_Newton*mass_1*mass_2*ᔑdt['a**(-1)']]
            # Apply momentum change to particle i of component_1
            # (the local component).
            momx_1[i] += Δmomx_ij
            momy_1[i] += Δmomy_ij
            momz_1[i] += Δmomz_ij
            # Apply or save the momentum change of particle j
            # of component_2 (the external component).
            with unswitch:
                if local:
                    # The passed component_1 and component_2 are really one
                    # and the same, and both contain data in the
                    # local domain. Apply momentum change to particle j.
                    momx_2[j] -= Δmomx_ij
                    momy_2[j] -= Δmomy_ij
                    momz_2[j] -= Δmomz_ij
                elif mutual:
                    # Add momentum change to the external Δmom buffers
                    # of component_2.
                    Δmomx_2[j] -= Δmomx_ij
                    Δmomy_2[j] -= Δmomy_ij
                    Δmomz_2[j] -= Δmomz_ij

# Function implementing the gravitational potential (in Fouier space).
# Here k2 = k² is the squared magnitude of the wave vector,
# in physical units.
@cython.header(
    k2='double',
    returns='double',
)
def gravity_potential(k2):
    return ℝ[-4*π*G_Newton]/k2

# Function that applies the differentiated gravitational potential
# to a component.
@cython.header(# Arguments
               component='Component',
               ᔑdt=dict,
               gradφ_dim='double[:, :, ::1]',
               dim='int',
               # Locals
               J_dim='FluidScalar',
               fac='double',
               i='Py_ssize_t',
               mom_dim='double*',
               posx='double*',
               posy='double*',
               posz='double*',
               x='double',
               y='double',
               z='double',
               returns='void',
               )
def apply_gravity_potential(component, ᔑdt, gradφ_dim, dim):
    """The argument gradφ_dim is the differentiated potential [∇φ]_dim
    in physical units.
    """
    if component.representation == 'particles':
        # Extract variables from component
        posx    = component.posx
        posy    = component.posy
        posz    = component.posz
        mom_dim = component.mom[dim]
        # Update the dim momentum component of particle i
        for i in range(component.N_local):
            # The coordinates of the i'th particle,
            # transformed so that 0 <= x, y, z < 1.
            x = (posx[i] - domain_start_x)/domain_size_x
            y = (posy[i] - domain_start_y)/domain_size_y
            z = (posz[i] - domain_start_z)/domain_size_z
            # Look up the force via a CIC interpolation,
            # convert it to momentum units and subtract it from the
            # momentum of particle i (subtraction because the force is
            # the negative gradient of the potential).
            # The factor with which to multiply gradφ_dim by to get
            # momentum updates is -mass*Δt, where Δt = ᔑdt['1'].
            mom_dim[i] -= ℝ[component.mass*ᔑdt['1']]*CIC_scalargrid2coordinates(gradφ_dim, x, y, z)
    elif component.representation == 'fluid':
        # Simply scale and extrapolate the values in gradφ_dim
        # to the grid points of the dim'th component of the
        # fluid variable J.
        # First extract this fluid scalar.
        J_dim = component.J[dim]
        # As the gravitational source term is
        # -a**(-3*w_eff)*(ϱ + c⁻²𝒫)*∂ⁱφ,
        # we need to multiply each grid point in [i, j, k] in gradφ_dim
        # by (ϱ[i, j, k] + c⁻²𝒫[i, j, k]) and all grid points by the
        # same factor -a**(-3*w_eff). Actually, since what we are after
        # are the updates to the momentum density, we should also
        # multiply by Δt. Since a**(-3*we_eff) is time dependent,
        # we should then really swap -a**(-3*w_eff)*Δt
        # for -ᔑa**(-3*w_eff)dt.
        fac = -ᔑdt['a**(-3*w_eff)', component]
        CIC_grid2grid(J_dim.grid_noghosts,
                      gradφ_dim,
                      fac=fac,
                      fac_grid=component.ϱ.grid_noghosts,
                      fac2=light_speed**(-2)*fac,
                      fac_grid2=component.𝒫.grid_noghosts,
                      )
        # Communicate the pseudo and ghost points of J_dim
        communicate_domain(J_dim.grid_mv, mode='populate')
