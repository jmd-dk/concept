# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: you can redistribute it and/or modify
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
# The auther of CO𝘕CEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module. In the .pyx file,
# this line willbe replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from graphics import plot_powerspec
    from gravity import (PM_CIC_FFT, PM_grid, PM_gridsize_local_j,
                         PM_gridstart_local_j)
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from graphics cimport plot_powerspec
    from gravity cimport (PM_CIC_FFT, PM_grid, PM_gridsize_local_j,
                          PM_gridstart_local_j)
    """


# Calculate the power spectrum of a snapshot
@cython.header(# Arguments
               particles='Particles',
               filename='str',
               # Locals
               P='double',
               W2='double',
               file_contents='str',
               i='int',
               j='int',
               j_global='int',
               k='int',
               k2='Py_ssize_t',
               k2_next='Py_ssize_t',
               k2_prev='Py_ssize_t',
               kR='double',
               kR6='double',
               ki='int',
               kj='int',
               kk='int',
               power_fac='double',
               power_fac2='double',
               recp_deconv_ijk='double',
               sqrt_deconv_ij='double',
               sqrt_deconv_ijk='double',
               sqrt_deconv_j='double',
               subscripts='dict',
               σ='double',
               σ_σ='double',
               )
def powerspec(particles, filename):
    global PM_grid, mask, power, power_N, power_σ2
    masterprint('Calculating power spectrum and saving to "{}" ...'
                 .format(filename))
    # CIC interpolate particles to the PM mesh
    # and do Fourier transformation.
    PM_CIC_FFT(particles)
    # Reset power, power multiplicity and power variance
    for k2 in range(k2_max):
        power[k2] = 0
        power_N[k2] = 0
        power_σ2[k2] = 0
    # Begin loop over PM_grid. As the first and second dimensions are
    # transposed due to the FFT, start with the j-dimension.
    for j in range(PM_gridsize_local_j):
        # The j-component of the wave vector
        j_global = j + PM_gridstart_local_j
        if j_global > nyquist:
            kj = j_global - PM_gridsize
        else:
            kj = j_global
        # Square root of the j-component of the deconvolution
        sqrt_deconv_j = sinc(kj*π_recp_PM_gridsize)
        # Loop over the entire first dimension
        for i in range(PM_gridsize):
            # The i-component of the wave vector
            if i > nyquist:
                ki = i - PM_gridsize
            else:
                ki = i
            # Square root of the product of the i-
            # and the j-component of the deconvolution.
            sqrt_deconv_ij = sinc(ki*π_recp_PM_gridsize)*sqrt_deconv_j
            # Loop over the entire last dimension in steps of two,
            # as contiguous pairs of elements are the real and imaginary
            # part of the same complex number.
            for k in range(0, PM_gridsize_padding, 2):
                # The k-component of the wave vector
                kk = k//2
                # The squared magnitude of the wave vector
                k2 = ki**2 + kj**2 + kk**2
                if k2 == 0 or k2 == k2_max:
                    continue
                # Symmetry removing part
                if kk == 0 or kk == nyquist:
                    if kj <= 0 and ki <= 0:
                        continue
                    if ki >= 0 and kj <= 0 and abs(kj) > ki:
                        continue
                    if ki <= 0 and kj >= 0 and abs(ki) >= kj:
                        continue
                # Square root of the product of
                # all components of the deconvolution.
                sqrt_deconv_ijk = sqrt_deconv_ij*sinc(kk*π_recp_PM_gridsize)
                # The reciprocal of the product of
                # all components of the deconvolution.
                recp_deconv_ijk = 1/(sqrt_deconv_ijk**2)
                # Do the deconvolution
                PM_grid[j, i, k] *= recp_deconv_ijk
                PM_grid[j, i, k + 1] *= recp_deconv_ijk
                # Increase the multiplicity
                power_N[k2] += 1
                # The power is the squared magnitude of the complex
                # number stored as
                # Re = PM_grid[j, i, k], Im = PM_grid[j, i, k + 1].
                P = PM_grid[j, i, k]**2 + PM_grid[j, i, k + 1]**2
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
        return
    # Boolean mask of the arrays.
    # This is the same for every power spectrum in the current run.
    if not mask.shape[0]:
        mask = (asarray(power_N) != 0)
    # Transform power from being the sum to being the mean, using
    # power_N. Also normalize to unity by dividing by N**2 (each of the
    # N particles contribute with a total value of 1 to PM_grid, which
    # is then squared to get the power). Finally, transform to physical
    # units by multiplying by the box volume.
    # At the same time, transform power_σ2 from being the sum of squares
    # to being the actual variance, using
    # power_σ2 = Σₖpowerₖ²/N - (Σₖpowerₖ/N)². Remember that as of
    # now, power_σ2 holds the sums of unnormalized squared powers.
    power_fac = boxsize3/particles.N**2
    power_fac2 = power_fac**2
    for k2 in range(k2_max):
        if power_N[k2] != 0:
            power[k2] *= power_fac/power_N[k2]
            power_σ2[k2] = power_σ2[k2]*power_fac2/power_N[k2] - power[k2]**2
    # Compute the rms density variation σ
    # together with its standard deviation.
    σ, σ_σ = rms_density_variation()
    # Write to disk
    header = ('{sigma}{_R}' + ' = {:.6g} '.format(σ)
              + '{pm}' + ' {:.6g}, '.format(σ_σ)
              + 'PM_gridsize = {}, '.format(PM_gridsize)
              + 'boxsize = {:.4g} {}\n'.format(boxsize, units.length)
              + 'k [' + units.length + '{^-1}]'
              + '\tpower [' + units.length + '{^3}]'
              + '\t{sigma}(power) [' + units.length + '{^3}]')
    np.savetxt(filename,
               asarray((asarray(k_magnitudes)     [mask], 
                        asarray(power)            [mask],
                        np.sqrt(asarray(power_σ2))[mask])).transpose(),
               fmt='%.6e\t%.6e\t%.6e',
               header=header)
    # Read in and rewrite in order to insert unicode characters
    with open(filename, 'r', encoding='utf-8') as powerspec_file:
        file_contents = powerspec_file.read()
    subscripts = dict(zip('0123456789e+-',
                          [unicode(c) for c in('₀', '₁', '₂', '₃', '₄',
                                               '₅', '₆', '₇', '₈', '₉',
                                               'ₑ', '₊', '₋')]))
    file_contents = file_contents.format(**{'sigma': unicode('σ'),
                                            '_R'   : ''.join([subscripts.get(c, c)
                                                              for c in '{:.15g}'.format(R_tophat)]
                                                             ),
                                            'pm'   : unicode('±'),
                                            '^-1'  : (unicode('⁻') + unicode('¹')),
                                            '^3'   : unicode('³')})
    with open(filename, 'w', encoding='utf-8') as powerspec_file:
        powerspec_file.write(file_contents)
    masterprint('done')
    # Plot the power spectrum
    if powerspec_plot:
        plot_powerspec((filename,
                        asarray(k_magnitudes)[mask],
                        asarray(power)       [mask],
                np.sqrt(asarray(power_σ2)    [mask])))

@cython.header(# Locals
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
def rms_density_variation():
    # Find the last data point
    for k2 in range(k2_max - 1, -1, -1):
        if power_N[k2] != 0:
            k2_last = k2
            break
    # Find the first two data points
    for k2 in range(k2_max):
        if power_N[k2] != 0:
            k2_left = k2
            integrand_left = σ2_integrand(k2)
            break
    for k2 in range(k2_left + 1, k2_max):
        if power_N[k2] != 0:
            k2_center = k2
            integrand_center = σ2_integrand(k2)
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
            k2_right,  integrand_right  = k2,        σ2_integrand(k2)
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
    σ2_fac = 4.5/boxsize**2
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
    return (σ, σ_σ)

# Function returning the integrand of σ², the square of the rms density
# variation, given an unnormalized k².
@cython.header(# Arguments
               k2='Py_ssize_t',
               # Locals
               kR='double',
               kR6='double',
               W2='double',
               returns='double',
               )
def σ2_integrand(k2):
    """
    The square of the rms density variation, σ², is given as
    σ² = ∫d³k/(2π)³ power W²
       = 1/(2π)³∫_0^∞ dk 4πk² power W²
       = 1/(2π)³∫_0^∞ dk²/(2k) 4πk² power W²
       = 1/(2π)²∫_0^∞ dk² k power W²,
    where dk² = k_fac² = (2π/boxsize)²
          --> 1/(2π)² dk² = 1/boxsize²
    and W = 3(sin(kR) - kR*cos(kR))/(kR)³.
    The W2 variable below is really W²/9.
    In total, the returned value is missing a factor of 9/boxsize**2.
    """
    kR = k_magnitudes[k2]*R_tophat
    kR6 = kR**6
    if kR6 > ten_machine_ϵ:
        W2 = sin(kR) - kR*cos(kR)
        W2 = W2**2/kR6
    else:
        W2 = ℝ[1/9]
    return k_magnitudes[k2]*power[k2]*W2


# Initialize variables used for the powerspectrum computation at import
# time, if such computation should ever take place.
cython.declare(k2_max='Py_ssize_t',
               k_fac='double',
               k_magnitudes='double[::1]',
               mask='object',  # This is only ever used as a numpy array
               nyquist='int',
               power='double[::1]',
               power_N='int[::1]',
               power_σ2='double[::1]',
               )
if powerspec_times or special_params.get('special', '') == 'powerspec':
    # Maximum value of any k-component (grid units)
    nyquist = PM_gridsize//2
    # Maximum value of k squared (grid units)
    k2_max = 3*nyquist**2
    # Array storing the power values
    power = empty(k2_max, dtype=C2np['double'])
    # Array counting the multiplicity of power data points
    power_N = empty(k2_max, dtype=C2np['int'])
    # Array storing the variance of the power values
    power_σ2 = empty(k2_max, dtype=C2np['double'])
    # Mask over the populated elements of power_N, power and
    # k_magnitudes. This mask is identical for every powerspectrum and
    # will be created when the first power spectrum is computed, and
    # then reused for all later power spectra.
    mask = array([], dtype=C2np['bint'])
    # Create array of physical k-magnitudes
    if master:
        k_fac = 2*π/boxsize
        k_magnitudes = k_fac*np.sqrt(arange(k2_max, dtype=C2np['double']))
