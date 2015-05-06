# Import everything from the commons module. In the .pyx file,
# this line willbe replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from gravity import PM_CIC_FFT, PM_grid, PM_gridsize_local_j, PM_gridstart_local_j
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from gravity cimport PM_CIC_FFT, PM_grid, PM_gridsize_local_j, PM_gridstart_local_j
    """


# Calculate the power spectrum with all allowed k2 modes,
# making a customised binning possible.
@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               particles='Particles',
               filename='str',
               # Locals not in Christian's code
               j='int',
               j_global='int',
               z_loop='ptrdiff_t',
               # Locals
               pNorm='double',
               Nyquist='int',
               k2_max='int',
               i='int',
               k='int',
               ki='int',
               kj='int',
               kk='int',
               k2='unsigned long int',
               k2_multi='int[::1]',
               power_arr='double[::1]',
               kNorm='double',
               tophat_scale='double',
               kR='double',
               re_part='double',
               im_part='double',
               wavek='double',
               sigma='double',
               power='double',
               )
def powerspectrum(particles, filename):
    global PM_grid

    pNorm = 1/(ϱm*boxsize**3)

    PM_CIC_FFT(particles)

    Nyquist = PM_gridsize//2
    # Maximum number of different k2 modes in box below Nyquist
    k2_max = 3*Nyquist**2
    zdim = PM_gridsize//2 + 1
    k2_multi = empty(k2_max, dtype='int32')
    power_arr = empty(k2_max, dtype='float64')
    if master:
        print('Calculating power spectrum')
    for k2 in range(k2_max):
        k2_multi[k2] = 0
        power_arr[k2] = 0
    kNorm = two_π/boxsize
    tophat_scale = 8*units.Mpc
    for j in range(PM_gridsize_local_j):
        j_global = j + PM_gridstart_local_j
        if j_global > Nyquist:
            kj = j_global - PM_gridsize
        else:
            kj = j_global
        # (Square root of) the j-component of the deconvolution
        deconvolution_j = sinc(kj*π_recp_PM_gridsize)
        for i in range(PM_gridsize):
            if i > Nyquist:
                ki = i - PM_gridsize
            else:
                ki = i
            # (Square root of) the product of the i- and the j-component of
            # the deconvolution.
            deconvolution_ij = sinc(ki*π_recp_PM_gridsize)*deconvolution_j
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
                # (Square root of) the product of all components of
                # the deconvolution.
                deconvolution_ijk = deconvolution_ij*sinc(kk*π_recp_PM_gridsize)
                # Do the deconvolution
                PM_grid[j, i, k] /= deconvolution_ijk**2
                PM_grid[j, i, k + 1] /= deconvolution_ijk**2
                # Compute the power spectrum
                k2_multi[k2] += 1  # Increase k2 multiplicity. k2 itself is the index
                re_part = PM_grid[j, i, k]
                im_part = PM_grid[j, i, k + 1]
                power_arr[k2] += re_part**2 + im_part**2  # This is the power. Will normalize later
    Reduce(MPI.IN_PLACE, k2_multi, op=MPI.SUM)
    Reduce(MPI.IN_PLACE, power_arr, op=MPI.SUM)
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
    sigma = reduce(sigma, op=MPI.SUM)
    sigma *= 4.0/3.0/(2*π)*tophat_scale**3 
    sigma = sqrt(sigma)
    if master:
        print('    Saving:', filename)
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