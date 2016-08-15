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
from mesh import diff
cimport('from communication import communicate_domain_boundaries, communicate_domain_ghosts')
cimport('from graphics import plot_powerspec')
cimport('from mesh import CIC_components2slabs, '
        '                 slab,                 '
        '                 slab_size_j,          '
        '                 slab_start_j,         '
        '                 slabs_FFT,            '
        )



# Function for calculating power spectra of components
@cython.header(# Arguments
               components='list',
               filename='str',
               # Locals
               P='double',
               slab_jik='double*',
               W2='double',
               fmt='str',
               header='str',
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
               component='Component',
               power='double[::1]',
               power_fac='double',
               power_fac2='double',
               power_σ2='double[::1]',
               power_σ2_k2='double',
               recp_deconv_ijk='double',
               row_quantity='list',
               row_type='list',
               row_σ_tophat='list',
               spectrum_plural='str',
               sqrt_deconv_ij='double',
               sqrt_deconv_ijk='double',
               sqrt_deconv_j='double',
               totmass='double',
               σ_tophat='dict',
               σ_tophat_σ='dict',
               )
def powerspec(components, filename):
    global slab, mask, k_magnitudes_masked, power_N, power_dict, power_σ2_dict
    # Do not compute any power spectra if
    # powerspec_select does not contain any True values.
    if not any(powerspec_select.values()):
        return
    # Dicts storing the rms density variation and its standard deviation
    # as values, with the component names as keys.
    σ_tophat   = {}
    σ_tophat_σ = {}
    # Compute a seperate power spectrum for each component
    for component in components:
        # If component.name are not in power_dict, it means that
        # power spectra for the i'th component should not be computed,
        # or that no power spectra have been computed yet.
        if component.name not in power_dict:
            # The power spectrum of the i'th component should only be
            # computed if {component.name: True} or {'all': True} exist
            # in powerspec_select. Also, if component.name exists,
            # the value for 'all' is ignored.
            if component.name.lower() in powerspec_select:
                if not powerspec_select[component.name.lower()]:
                    continue
            elif not powerspec_select.get('all', False):
                continue
            # Power spectrum of this component should be computed!
            # Allocate arrays for the final powerspectra results
            # for the i'th component.
            power_dict[component.name]    = empty(k2_max + 1, dtype=C2np['double'])
            power_σ2_dict[component.name] = empty(k2_max + 1, dtype=C2np['double'])
        masterprint('Computing power spectrum of {} ...'.format(component.name))
        # Assign short names for the arrays storing the results
        power    = power_dict[component.name]
        power_σ2 = power_σ2_dict[component.name]
        # CIC interpolate component to the slabs
        # and do Fourier transformation.
        CIC_components2slabs([component])
        slabs_FFT()
        # Reset power, power multiplicity and power variance
        for k2 in range(k2_max):
            power[k2] = 0
            power_N[k2] = 0
            power_σ2[k2] = 0
        # Begin loop over slab. As the first and second dimensions
        # are transposed due to the FFT, start with the j-dimension.
        for j in range(slab_size_j):
            # The j-component of the wave vector
            j_global = j + slab_start_j
            if j_global > φ_gridsize_half:
                kj = j_global - φ_gridsize
            else:
                kj = j_global
            kj2 = kj**2
            # Square root of the j-component of the deconvolution
            sqrt_deconv_j = sinc(kj*ℝ[π/φ_gridsize])
            # Loop over the entire first dimension
            for i in range(φ_gridsize):
                # The i-component of the wave vector
                if i > φ_gridsize_half:
                    ki = i - φ_gridsize
                else:
                    ki = i
                ki2_plus_kj2 = ki**2 + kj2
                # Square root of the product of the i-
                # and the j-component of the deconvolution.
                sqrt_deconv_ij = sinc(ki*ℝ[π/φ_gridsize])*sqrt_deconv_j
                # Loop over the entire last dimension in steps of two,
                # as contiguous pairs of elements are the real and
                # imaginary part of the same complex number.
                for k in range(0, slab_size_padding, 2):
                    # The k-component of the wave vector
                    kk = k//2
                    # The squared magnitude of the wave vector
                    k2 = ki2_plus_kj2 + kk**2
                    # Square root of the product of
                    # all components of the deconvolution.
                    sqrt_deconv_ijk = sqrt_deconv_ij*sinc(kk*ℝ[π/φ_gridsize])
                    # The reciprocal of the product of
                    # all components of the deconvolution.
                    recp_deconv_ijk = 1/(sqrt_deconv_ijk**2)
                    # Pointer to the [j, i, k]'th element of the slab.
                    # The complex number is then given as
                    # Re = slab_jik[0], Im = slab_jik[1].
                    slab_jik = cython.address(slab[j, i, k:])
                    # Do the deconvolution
                    slab_jik[0] *= recp_deconv_ijk  # Real part
                    slab_jik[1] *= recp_deconv_ijk  # Imag part
                    # Increase the multiplicity
                    power_N[k2] += 1
                    # The power is the squared magnitude
                    # of the complex number
                    P = slab_jik[0]**2 + slab_jik[1]**2
                    # Increase the power. This is unnormalized for now.
                    power[k2] += P
                    # Increase the variance. For now, this is only the
                    # unnormalized sum of squares.
                    power_σ2[k2] += P**2
        # Sum power, power_N and power_σ2 into the master process
        Reduce(sendbuf=(MPI.IN_PLACE if master else power),
               recvbuf=(power        if master else None),
               op=MPI.SUM)
        Reduce(sendbuf=(MPI.IN_PLACE if master else power_N),
               recvbuf=(power_N      if master else None),
               op=MPI.SUM)
        Reduce(sendbuf=(MPI.IN_PLACE if master else power_σ2),
               recvbuf=(power_σ2     if master else None),
               op=MPI.SUM)
        if not master:
            continue
        # Remove the k2 == 0 elements (the background)
        # of the power arrays.
        power_N[0] = power[0] = power_σ2[0] = 0
        # Remove the k2 == k2_max elemenets of the power arrays,
        # as this comes from only one data (grid) point as is therefore
        # highly uncertain.
        power_N[k2_max] = power[k2_max] = power_σ2[k2_max] = 0
        # Boolean mask of the arrays and a masked version of the
        # k_magnitudes array. Both are identical for every
        # power spectrum in the current run.
        if not mask.shape[0]:
            mask = (asarray(power_N) != 0)
            k_magnitudes_masked = asarray(k_magnitudes)[mask]
        # Transform power from being the sum to being the mean,
        # by dividing by power_N. Also normalize to unity by dividing
        # by totmass**2 (each particle/fluid element contribute to the
        # φ grid with a total value equal to their mass. The φ grid is
        # then squared to get the power).
        # Finally, transform to physical units by multiplying by the box
        # volume. At the same time, transform power_σ2 from being the
        # sum of squares to being the actual variance,
        # using power_σ2 = Σₖpowerₖ²/N - (Σₖpowerₖ/N)².
        # Remember that as of now, power_σ2 holds the sums of
        # unnormalized squared powers.
        # Finally, divide by power_N to correct for the sample size.
        if component.representation == 'particles':
            totmass = component.N*component.mass
        elif component.representation == 'fluid':
            totmass = component.gridsize**3*component.mass
        power_fac = ℝ[boxsize**3]/totmass**2
        power_fac2 = power_fac**2
        for k2 in range(k2_max):
            if power_N[k2] != 0:
                power[k2] *= power_fac/power_N[k2]
                power_σ2_k2 = (power_σ2[k2]*power_fac2/power_N[k2] - power[k2]**2)/power_N[k2]
                # Round-off errors can lead to slightly negative
                # power_σ2_k2, which is not acceptable.
                if power_σ2_k2 > 0:
                    power_σ2[k2] = power_σ2_k2
                else:
                    power_σ2[k2] = 0
        # Compute the rms density variation σ_tophat
        # together with its standard deviation σ_tophat_σ.
        σ_tophat[component.name], σ_tophat_σ[component.name] = rms_density_variation(power,
                                                                                     power_σ2)
        masterprint('done')
    # Only the master process should write
    # power spectra to disk and do plotting.
    if not master:
        return
    # Construct the header.
    # Note that the chosen format only works well when all
    # numbers are guaranteed to be positive, as with power spectra.
    spectrum_plural = 'spectrum' if len(power_dict) == 1 else 'spectra'
    masterprint('Saving power {} to "{}" ...'.format(spectrum_plural, filename))
    header = ('# Power {} at t = {:.6g} {}{} '
              'computed with a grid of linear size {}\n#\n'
              .format(spectrum_plural,
                      universals.t,
                      unit_time,
                      ', a = {:.6g}'.format(universals.a) if enable_Hubble else '',
                      φ_gridsize)
              )
    # Header lines for component name, σ_tophat and quantity
    fmt = '{:<15}'
    row_type = [' ']
    row_σ_tophat = [' ']
    row_quantity = [unicode('k [Mpc⁻¹]')]
    for component in components:
        if component.name not in power_dict:
            continue
        fmt += '{:<2}'  # Space
        row_type.append(' ')
        row_σ_tophat.append(' ')
        row_quantity.append(' ')
        fmt += '{:^33}  '  # Either type, σ_tophat or power and σ(power)
        row_type.append(component.name)
        row_σ_tophat.append('σ' + unicode_subscript('{:.2g}'.format(R_tophat/units.Mpc))
                            + ' = {:.4g} '.format(σ_tophat[component.name]) + '±'
                            + ' {:.4g}'.format(σ_tophat_σ[component.name]))
        row_quantity.append('power [Mpc³]')
        row_quantity.append('σ(power) [Mpc³]')
    header += '# ' + fmt.format(*row_type) + '\n'
    header += '# ' + fmt.format(*row_σ_tophat) + '\n'
    header += '# ' + fmt.replace('{:^33} ', ' {:<16} {:<16}').format(*row_quantity) + '\n'
    # Write header to file
    with open(filename, 'w', encoding='utf-8') as powerspec_file:
        powerspec_file.write(unicode(header))
    # Mask the data and pack it into a list
    data_list = [k_magnitudes_masked]
    for component in components:
        if component.name not in power_dict:
            continue
        data_list.append(asarray(power_dict[component.name])[mask])
        # Take sqrt to convert power_σ2 to power_σ
        data_list.append(np.sqrt(asarray(power_σ2_dict[component.name])[mask]))
    # Write data to file
    with open(filename, 'a+b') as powerspec_file:
        np.savetxt(powerspec_file,
                   asarray(data_list).transpose(),
                   fmt=('%-13.6e' + len(power_dict)*(  7*' ' + '%-13.6e'
                                                     + 4*' ' + '%-13.6e')))
    masterprint('done')
    # Plot the power spectra
    plot_powerspec(data_list, filename, power_dict)

# Function which computes σ_tophat (the rms density variation)
# and its standard deviation σ_tophat_σ from the power spectrum.
@cython.header(# Arguments
               power='double[::1]',
               power_σ2='double[::1]',
               # Locals
               k2='Py_ssize_t',
               k2_center='Py_ssize_t',
               k2_last='Py_ssize_t',
               k2_left='Py_ssize_t',
               k2_right='Py_ssize_t',
               integrand_center='double',
               integrand_left='double',
               integrand_right='double',
               σ='double',
               σ_σ='double',
               σ2='double',
               σ2_fac='double',
               σ2_part='double',
               σ2_σ2='double',
               returns='tuple',
               )
def rms_density_variation(power, power_σ2):
    # These definitions are simply to silent compiler warnings
    k2_center = k2_last = k2_left = integrand_center = integrand_left = 0
    # Find the last data point
    for k2 in range(k2_max - 1, -1, -1):
        if power_N[k2] != 0:
            k2_last = k2
            break
    # Find the first two data points
    for k2 in range(k2_max):
        if power_N[k2] != 0:
            k2_left = k2
            integrand_left = σ2_integrand(power, k2)
            break
    for k2 in range(k2_left + 1, k2_max):
        if power_N[k2] != 0:
            k2_center = k2
            integrand_center = σ2_integrand(power, k2)
            break
    # Trapezoidally integrate the first data point
    σ2 = (k2_center - k2_left)*integrand_left
    # The variance of σ2, so far
    σ2_σ2 = (σ2/power[k2_left])**2*power_σ2[k2_left]
    # Do the integration for all other data points except the last one
    k2_right, integrand_right = k2_center, integrand_center
    for k2 in range(k2_center + 1, k2_last + 1):
        if power_N[k2] != 0:
            # Data point found to the right. Shift names
            k2_left,   integrand_left   = k2_center, integrand_center
            k2_center, integrand_center = k2_right,  integrand_right
            k2_right,  integrand_right  = k2,        σ2_integrand(power, k2)
            # Do the trapezoidal integration
            σ2_part = (k2_right - k2_left)*integrand_center
            σ2 += σ2_part
            # Update the variance
            σ2_σ2 += ((σ2_part/power[k2_center])**2*power_σ2[k2_center])
    # Trapezoidally integrate the last data point
    σ2_part = (k2_right - k2_center)*integrand_right
    σ2 += σ2_part
    # Update the variance
    σ2_σ2 = (σ2_part/power[k2_right])**2*power_σ2[k2_right]
    # Normalize σ2. According to the σ2_integrand function, the
    # integrand is missing a factor of 9/boxsize**2. In addition, the
    # trapezoidal integration above misses a factor ½.
    σ2_fac = ℝ[4.5/boxsize**2]
    σ2    *= σ2_fac
    σ2_σ2 *= σ2_fac**2
    # To get the standard deviation σ from the variance σ2, simply take
    # the square root.
    σ = sqrt(σ2)
    # To get the standard deviation of σ, σ_σ, first compute the
    # variance of σ, σ_σ2:
    #     σ_σ2 = (∂σ/∂σ2)²σ2_σ2
    #          = 1/(4*σ2)*σ2_σ2.
    # Then take the square root to get the standard deviation from the
    # variance.
    σ_σ = sqrt(1/(4*σ2)*σ2_σ2)
    return σ, σ_σ

# Function returning the integrand of σ², the square of the rms density
# variation, given an unnormalized k².
@cython.header(# Arguments
               power='double[::1]',
               k2='Py_ssize_t',
               # Locals
               kR='double',
               kR6='double',
               W2='double',
               returns='double',
               )
def σ2_integrand(power, k2):
    """
    The square of the rms density variation, σ², is given as
    σ² = ∫d³k/(2π)³ power W²
       = 1/(2π)³∫_0^∞ dk 4πk² power W²
       = 1/(2π)³∫_0^∞ dk²/(2k) 4πk² power W²
       = 1/(2π)²∫_0^∞ dk² k power W²,
    where dk² = (2π/boxsize)²
          --> 1/(2π)² dk² = 1/boxsize²
    and W = 3(sin(kR) - kR*cos(kR))/(kR)³.
    The W2 variable below is really W²/9.
    In total, the returned value is missing a factor of 9/boxsize**2.
    """
    kR = k_magnitudes[k2]*R_tophat
    kR6 = kR**6
    if kR6 > ℝ[10*machine_ϵ]:
        W2 = sin(kR) - kR*cos(kR)
        W2 = W2**2/kR6
    else:
        W2 = ℝ[1/9]
    return k_magnitudes[k2]*power[k2]*W2

# Function which can measure different quantities of a parsed component
@cython.header(# Arguments
               component='Component',
               quantity='str',
               # Locals
               N='Py_ssize_t',
               N_elements='Py_ssize_t',
               Vcell='double',
               diff_backward='double[:, :, ::1]',
               diff_forward='double[:, :, ::1]',
               diff_max='double[::1]',
               diff_max_dim='double',
               diff_size='double',
               dim='int',
               fluidscalar='FluidScalar',
               h='double',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               mass='double',
               mom='double*',
               mom_dim='double',
               mom_i='double',
               names='list',
               u_arr='object',  # np.ndarray
               ϱu_noghosts='double[:, :, :]',
               Δdiff='double',
               Δdiff_max='double[::1]',
               Δdiff_max_dim='double',
               Δdiff_max_list='list',
               Δdiff_max_normalized='double[::1]',
               Δdiff_max_normalized_list='list',
               Σmass='double',
               Σmom='double[::1]',
               Σmom_dim='double',
               Σmom2_dim='double',
               Σϱ='double',
               Σϱ2='double',
               ϱ='FluidScalar',
               ϱ_arr='object',  # np.ndarray
               ϱ_mv='double[:, :, ::1]',
               ϱ_noghosts='double[:, :, :]',
               σ2mom_dim='double',
               σ2ϱ='double',
               σmom='double[::1]',
               σmom_dim='double',
               σϱ='double',
               returns='object',
               )
def measure(component, quantity):
    """Implemented quantities are:
    'momentum'
    'mass'           (fluid quantity)
    'discontinuity'  (fluid quantity)
    """
    # Extract variables
    N = component.N
    N_elements = component.gridsize**3
    mass = component.mass
    Vcell = (boxsize/component.gridsize)**3
    ϱ = component.fluidvars['ϱ']
    ϱ_mv = ϱ.grid_mv
    ϱ_noghosts = ϱ.grid_noghosts
    # Quantities exhibited by both particle and fluid components
    if quantity == 'momentum':
        Σmom = empty(3, dtype=C2np['double'])
        σmom = empty(3, dtype=C2np['double'])
        if component.representation == 'particles':
            # Total momentum of all particles, for each dimension
            for dim in range(3):
                mom = component.mom[dim]
                Σmom_dim = Σmom2_dim = 0
                # Add up local particle momenta
                for i in range(component.N_local):
                    mom_i = mom[i]
                    Σmom_dim  += mom_i
                    Σmom2_dim += mom_i**2
                # Add up local particle momenta sums 
                Σmom_dim  = allreduce(Σmom_dim,  op=MPI.SUM)
                Σmom2_dim = allreduce(Σmom2_dim, op=MPI.SUM)
                # Compute global standard deviation
                σ2mom_dim = Σmom2_dim/N - (Σmom_dim/N)**2
                if σ2mom_dim < 0:
                    # Negative (about -machine_ϵ) σ² can happen due
                    # to round-off errors.
                    σ2mom_dim = 0
                σmom_dim = sqrt(σ2mom_dim)
                # Pack results
                Σmom[dim] = Σmom_dim
                σmom[dim] = σmom_dim
        elif component.representation == 'fluid':
            # Total momentum of all fluid elements, for each dimension
            for dim, fluidscalar in enumerate(component.fluidvars['ϱu']):
                ϱu_noghosts = fluidscalar.grid_noghosts
                Σmom_dim = Σmom2_dim = 0
                # Add up local fluid element momenta
                for         i in range(ℤ[ϱ_noghosts.shape[0] - 1]):
                    for     j in range(ℤ[ϱ_noghosts.shape[1] - 1]):
                        for k in range(ℤ[ϱ_noghosts.shape[2] - 1]):
                            # Momentum p is given by
                            # p = m*u*a
                            #   = (ϱ*Vcell)*(ϱu/ϱ)*a
                            #   = ϱu*Vcell*a
                            mom_dim = ϱu_noghosts[i, j, k]*ℝ[Vcell*universals.a]
                            Σmom_dim  += mom_dim
                            Σmom2_dim += mom_dim**2
                # Add up local fluid element momenta sums
                Σmom_dim  = allreduce(Σmom_dim,  op=MPI.SUM)
                Σmom2_dim = allreduce(Σmom2_dim, op=MPI.SUM)
                # Compute global standard deviation
                σ2mom_dim = Σmom2_dim/N_elements - (Σmom_dim/N_elements)**2
                if σ2mom_dim < 0:
                    # Negative (about -machine_ϵ) σ² can happen due
                    # to round-off errors.
                    σ2mom_dim = 0
                σmom_dim = sqrt(σ2mom_dim)
                # Pack results
                Σmom[dim] = Σmom_dim
                σmom[dim] = σmom_dim
        return Σmom, σmom
    # Fluid quantities
    elif quantity == 'ϱ':
        # Compute sum(ϱ) and std(ϱ)
        if component.representation == 'particles':
            # Particle components have no ϱ
            abort('The measure function was called with the "{}" component with '
                  'quantity=\'ϱ\', but particle components do not have ϱ'
                  .format(component.name)
                  )
        elif component.representation == 'fluid':
            # NumPy array of local part of ϱ with no pseudo points
            ϱ_arr = asarray(ϱ_noghosts[:(ϱ_noghosts.shape[0] - 1),
                                       :(ϱ_noghosts.shape[1] - 1),
                                       :(ϱ_noghosts.shape[2] - 1)])
            # Total ϱ of all fluid elements
            Σϱ = np.sum(ϱ_arr)
            # Total ϱ² of all fluid elements
            Σϱ2 = np.sum(ϱ_arr**2)
            # Add up local sums
            Σϱ  = allreduce(Σϱ,  op=MPI.SUM)
            Σϱ2 = allreduce(Σϱ2, op=MPI.SUM)
            # Compute global standard deviation
            σ2ϱ = Σϱ2/N_elements - (Σϱ/N_elements)**2
            if σ2ϱ < 0:
                # Negative (about -machine_ϵ) σ² can happen due
                # to round-off errors.
                σ2ϱ = 0
            σϱ = sqrt(σ2ϱ)
        return Σϱ, σϱ
    elif quantity == 'mass':
        if component.representation == 'particles':
            # The total mass is fixed for particle components
            masterwarn('The measure function was called with the "{}" component with '
                       'quantity=\'mass\'. This is a redundant call as the total mass of a '
                       'particle component is fixed'
                       .format(component.name)
                       )
            Σmass = component.N*mass
        elif component.representation == 'fluid':
            # Get total ϱ
            Σϱ, σϱ = measure(component, 'ϱ')
            # Convert total ϱ to total mass via
            # mass_elemet = ϱ*Vcell
            # --> Σmass = Σϱ*Vcell
            Σmass = Σϱ*Vcell
        return Σmass
    elif quantity == 'discontinuity':
        if component.representation == 'particles':
            # Particle components have no discontinuity
            abort('The measure function was called with the "{}" component with '
                  'quantity=\'discontinuity\', which is not applicable to particle componnets'
                  .format(component.name)
                  )
        elif component.representation == 'fluid':
            # Lists to store results which will be returned
            names = []
            Δdiff_max_normalized_list = []
            Δdiff_max_list = []
            # The grid spacing in physical units
            h = boxsize/component.gridsize
            # The meshbuf buffer will be used for storing the
            # backwards differentiated grid. Another grid is needed
            # for storing the forward differentiated grid.
            diff_forward = empty(component.fluidvars['shape_noghosts'], dtype=C2np['double'])
            # Find the maximum discontinuity in each fluid grid
            for fluidscalar in component.iterate_fluidscalars():
                # Store the name of the fluid scalar
                names.append('{}{}'.format('ϱ' if fluidscalar.varnum == 0 else 'ϱu',
                                           ''  if fluidscalar.varnum == 0 else 'xyz'[fluidscalar.multi_index[0]]))
                # Communicate pseudo and ghost points of the grid
                communicate_domain_boundaries(fluidscalar.grid_mv, mode=1)
                communicate_domain_ghosts(fluidscalar.grid_mv)
                # Differentiate the grid in all three directions via
                # both forward and backward difference. For each
                # direction, save the largest difference between
                # the two. Also save the largest differential in
                # each direction.
                Δdiff_max = empty(3, dtype=C2np['double'])
                diff_max = empty(3, dtype=C2np['double'])
                for dim in range(3):
                    # Nullify the forward diff buffer (the backward
                    # diff buffer is really meshbuf, which will be
                    # nullified by the diff method).
                    diff_forward[...] = 0
                    # Do the differentiations.
                    # Use diff_forward as buffer for the forwards
                    # difference and meshbuf (here called diff_backward)
                    # as buffer for the backwards dfifference.
                    diff_forward  = diff(fluidscalar.grid_mv, dim, h, diff_forward, order=1, direction='forward')
                    diff_backward = diff(fluidscalar.grid_mv, dim, h, None,         order=1, direction='backward')
                    # Find the largest difference between the results of the
                    # forward and backward difference,
                    Δdiff_max_dim = 0
                    diff_max_dim = 0
                    for         i in range(ℤ[ϱ_noghosts.shape[0] - 1]):
                        for     j in range(ℤ[ϱ_noghosts.shape[1] - 1]):
                            for k in range(ℤ[ϱ_noghosts.shape[2] - 1]):
                                # The maximum difference of the two differentials
                                Δdiff = abs(diff_forward[i, j, k] - diff_backward[i, j, k])
                                if Δdiff > Δdiff_max_dim:
                                    Δdiff_max_dim = Δdiff
                                # The maximum differential
                                diff_size = abs(diff_forward[i, j, k])
                                if diff_size > diff_max_dim:
                                    diff_max_dim = diff_size
                                diff_size = abs(diff_backward[i, j, k])
                                if diff_size > diff_max_dim:
                                    diff_max_dim = diff_size
                    # Use the global maxima
                    Δdiff_max_dim = allreduce(Δdiff_max_dim, op=MPI.MAX)
                    diff_max_dim  = allreduce(diff_max_dim,  op=MPI.MAX)
                    # Pack results into lists
                    Δdiff_max[dim] = Δdiff_max_dim
                    diff_max[dim] = diff_max_dim
                Δdiff_max_list.append(Δdiff_max)
                # Maximum discontinuity (difference between forward and
                # backward difference) normalized accoring to
                # the largest slope.
                Δdiff_max_normalized_list.append(np.array([Δdiff_max[dim]/diff_max[dim]
                                                           if Δdiff_max[dim] > 0 else 0
                                                           for dim in range(3)],
                                                          dtype=C2np['double'],
                                                          )
                                                 )
        return names, Δdiff_max_list, Δdiff_max_normalized_list
    elif master:
        abort('The measure function was called with quantity=\'{}\', which is not implemented'
              .format(quantity))

# Function for doing debugging analysis
@cython.header(# Arguments
               components='list',
               # Locals
               component='Component',
               dim='int',
               name='str',
               Δdiff_max='double[::1]',
               Δdiff_max_normalized='double[::1]',
               Σmass='double',
               Σmass_correct='double',
               Σmom='double[::1]',
               Σϱ='double',
               σmom='double[::1]',
               σϱ='double',
               )
def debug(components):
    """This function will compute many different quantities from the
    component data and print out the results. For obvious inconsistent
    results, a warning will be given.
    """
    if not enable_debugging:
        return
    # Componentwise analysis
    for component in components:
        # sum(momentum) and std(momentum) in each dimension
        Σmom, σmom = measure(component, 'momentum')
        unit = units.m_sun*units.Mpc/units.Gyr
        for dim in range(3):
            debug_print('total {}-momentum'.format('xyz'[dim]),
                        component,
                        Σmom[dim],
                        'm☉ Mpc Gyr⁻¹',
                        )
            debug_print('standard deviation of {}-momentum'.format('xyz'[dim]),
                        component,
                        σmom[dim],
                        'm☉ Mpc Gyr⁻¹',
                        )
        # sum(ϱ) and std(ϱ)
        if component.representation == 'fluid':
            Σϱ, σϱ = measure(component, 'ϱ')
            debug_print('total ϱ', component, Σϱ)
            debug_print('standard deviation of ϱ', component, σϱ)
        # The total mass
        if component.representation == 'fluid':
            Σmass = measure(component, 'mass')
            debug_print('total mass', component, Σmass, 'm☉')
            # Warn if the total mass is incorrect
            Σmass_correct = component.gridsize**3*component.mass
            if not isclose(Σmass, Σmass_correct):
                masterwarn('Component "{}" ought to have a total mass of {} m☉'
                           .format(component.name,
                                   significant_figures(Σmass_correct/units.m_sun,
                                                       12,
                                                       fmt='unicode',
                                                       incl_zeros=False,
                                                       scientific=True,
                                                       ),
                                   )
                    )
        # The maximum discontinuities in the fluid scalars,
        # for each dimension. Here, a discontinuity means a difference
        # in forward and backward difference.
        if component.representation == 'fluid':
            for name, Δdiff_max, Δdiff_max_normalized in zip(*measure(component, 'discontinuity')):
                for dim in range(3):
                    debug_print('maximum            {}-discontinuity in {}'.format('xyz'[dim], name),
                                component,
                                Δdiff_max[dim],
                                'Mpc⁻¹',
                                )
                    debug_print('maximum normalized {}-discontinuity in {}'.format('xyz'[dim], name),
                                component,
                                Δdiff_max_normalized[dim],
                                )

# Function for printing out debugging info,
# used in the debug function above.
@cython.header(# Arguments
               quantity='str',
               component='Component',
               value='double',
               unit_str='str',
               # Locals
               text='str',
               unit='double',
               value_str='str',
               )
def debug_print(quantity, component, value, unit_str='1'):
    unit = evaluate_unit(unit_str)
    value_str = significant_figures(value/unit,
                                    12,
                                    fmt='unicode',
                                    incl_zeros=False,
                                    scientific=True,
                                    )
    text = '{} {}({}) = {}{}'.format(terminal.bold_cyan('Debug info:'),
                                     quantity[0].upper() + quantity[1:],
                                     component.name,
                                     value_str,
                                     ' ' + unit_str if unit_str != '1' else '',
                                     )
    masterprint(text)


# Initialize variables used for the powerspectrum computation at import
# time, if such computation should ever take place.
cython.declare(k2_max='Py_ssize_t',
               k_magnitudes='double[::1]',
               k_magnitudes_masked='double[::1]',
               mask='object',           # numpy.ndarray
               power_N='int[::1]',
               power_dict='object',     # OrderedDict
               power_σ2_dict='object',  # OrderedDict
               )
if any(powerspec_times.values()) or special_params.get('special', '') == 'powerspec':
    # Maximum value of k squared (grid units) 
    k2_max = 3*φ_gridsize_half**2
    # Array counting the multiplicity of power data points
    power_N = empty(k2_max + 1, dtype=C2np['int'])
    # (Ordered) dictionaries with component names as keys and
    # power and power_σ2 as values.
    power_dict = collections.OrderedDict()
    power_σ2_dict = collections.OrderedDict()
    # Mask over the populated elements of power_N, power and
    # k_magnitudes. This mask is identical for every powerspectrum and
    # will be build when the first power spectrum is computed, and
    # then reused for all later power spectra.
    mask = np.array([], dtype=C2np['bint'])
    # Masked array of k_magnitudes. Will be build together with mask
    k_magnitudes_masked = np.array([], dtype=C2np['double'])
    # Create array of physical k-magnitudes
    if master:
        k_magnitudes = 2*π/boxsize*np.sqrt(arange(1 + k2_max, dtype=C2np['double']))

