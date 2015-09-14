# This file is part of CONCEPT, the cosmological N-body code in Python.
# Copyright (C) 2015 Jeppe Mosgard Dakin.
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CONCEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CONCEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CONCEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module. In the .pyx file,
# this line willbe replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from gravity import (PM_CIC_FFT, PM_grid, PM_gridsize_local_j,
                         PM_gridstart_local_j)
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from gravity cimport (PM_CIC_FFT, PM_grid, PM_gridsize_local_j,
                          PM_gridstart_local_j)
    """


# Calculate the power spectrum of a snapshot
@cython.header(# Arguments
               particles='Particles',
               filename='str',
               # Locals
               i='int',
               im_part='double',
               j='int',
               j_global='int',
               k='int',
               ki='int',
               kj='int',
               kk='int',
               k2='unsigned long int',
               kR='double',
               re_part='double',
               recp_deconv_ijk='double',
               sqrt_deconv_ij='double',
               sqrt_deconv_ijk='double',
               sqrt_deconv_j='double',
               wavek='double',
               sigma='double',
               power='double',
               z_loop='ptrdiff_t',
               )
def powerspectrum(particles, filename):
    # Get 
    global PM_grid, k2_multi, power_arr

    #
    PM_CIC_FFT(particles)


    masterprint('Calculating power spectrum ...')
    for k2 in range(k2_max):
        k2_multi[k2] = 0
        power_arr[k2] = 0
    for j in range(PM_gridsize_local_j):
        j_global = j + PM_gridstart_local_j
        if j_global > Nyquist:
            kj = j_global - PM_gridsize
        else:
            kj = j_global
        # Square root of the j-component of the deconvolution
        sqrt_deconv_j = sinc(kj*π_recp_PM_gridsize)
        for i in range(PM_gridsize):
            if i > Nyquist:
                ki = i - PM_gridsize
            else:
                ki = i
            # Square root of the product of the i-
            # and the j-component of the deconvolution.
            sqrt_deconv_ij = sinc(ki*π_recp_PM_gridsize)*sqrt_deconv_j
            for k in range(0, PM_gridsize_padding, 2):
                kk = k//2  # inserted
                k2 = ki**2 + kj**2 + kk**2
                if k2 == 0 or k2 == k2_max:
                    continue
                # Symmetry removing part
                if kk == 0 or kk == Nyquist:
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
                recp_deconv_ijk = 1.0/(sqrt_deconv_ijk**2)
                # Do the deconvolution
                PM_grid[j, i, k] *= recp_deconv_ijk
                PM_grid[j, i, k + 1] *= recp_deconv_ijk
                # Compute the power spectrum.
                # Increase k2 multiplicity. k2 itself is the index.
                k2_multi[k2] += 1
                re_part = PM_grid[j, i, k]      # Real part
                im_part = PM_grid[j, i, k + 1]  # Imag part
                # This is the power. Will normalize later
                power_arr[k2] += re_part**2 + im_part**2
    Allreduce(MPI.IN_PLACE, k2_multi, op=MPI.SUM)
    Allreduce(MPI.IN_PLACE, power_arr, op=MPI.SUM)
    for k2 in range(k2_max):
        power_arr[k2] = power_arr[k2]/k2_multi[k2]*pNorm
    sigma = 0
    for k2 in range(k2_max):
        if k2_multi[k2] == 0:
            continue
        wavek = kNorm*sqrt(k2)
        kR = wavek*tophat_scale
        kR = 3*(sin(kR) - kR*cos(kR))/kR**3
        sigma += power_arr[k2]*kR**2*wavek**2
    #sigma = allreduce(sigma, op=MPI.SUM)
    if not master:
        return
    sigma *= 4.0/3.0/(2*π)*tophat_scale**3 
    sigma = sqrt(sigma)
    masterprint('done')
    masterprint('Saving powerspectrum "' + filename + '" ...')
    header = ('sigma{} = {:.6e}, PM_gridsize = {}, boxsize = {:.3e} Mpc\n'
              + 'k\tmodes\tpower').format(int(round(tophat_scale/units.Mpc)),
                                          sigma,
                                          PM_gridsize,
                                          boxsize/units.Mpc)
    np.savetxt(filename,
               array((np.sqrt(arange(k2_max)),
                      k2_multi,
                      power_arr)).transpose()[array(k2_multi) != 0, :],
               fmt='%.6e\t%i\t%.6e',
               header=header)
    masterprint('done')


cython.declare(Nyquist='int',
               k2_max='int',
               k2_multi='int[::1]',
               kNorm='double',
               pNorm='double',
               power_arr='double[::1]',
               tophat_scale='double',
               )
Nyquist = PM_gridsize//2
k2_max = 3*Nyquist**2
k2_multi = empty(k2_max, dtype='int32')
kNorm = two_π/boxsize
pNorm = 1/(ϱm*boxsize**3)
power_arr = empty(k2_max, dtype='float64')
tophat_scale = 8*units.Mpc
