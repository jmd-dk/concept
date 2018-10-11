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



# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

Σmν = float(class_params['m_ncdm'])*float(class_params['deg_ncdm'])
n_tailcut = 25



#################################################
# Absolute neutrino power spectra, large scales #
#################################################
masterprint(f'Analyzing {this_test} (absolute) data ...')
# Read in the neutrino CO𝘕CEPT power spectra
filenames = sorted([
    os.path.basename(filename) for filename in
    glob(f'{this_dir}/output_big_linear/powerspec_a=*')
    if not filename.endswith('.png')
])
scalefactors = asarray([
    float(re.search('a=(.*)', filename).group(1))
    for filename in filenames
])
k = np.loadtxt(f'{this_dir}/output_big_linear/{filenames[0]}', usecols=0)[:-n_tailcut]
Pν_linear = [
    np.loadtxt(
        f'{this_dir}/output_big_linear/{filename}',
        usecols=3,
    )[:-n_tailcut]
    for filename in filenames
]
Pν_nonlinear = [
    np.loadtxt(
        f'{this_dir}/output_big_nonlinear/{filename}',
        usecols=3,
    )[:-n_tailcut]
    for filename in filenames
]

# Plot absolute neutrino power spectra
fig, axes = plt.subplots(len(scalefactors), sharex=True, figsize=(6.4, 4.8*0.5*len(scalefactors)))
for a, P_lin, P_nonlin, ax in zip(scalefactors, Pν_linear, Pν_nonlinear, axes):
    ax.loglog(
        k,
        k**3*P_lin,
        '-',
        label=r'CO$N$CEPT (linear $\nu$)',
    )
    ax.loglog(
        k,
        k**3*P_nonlin,
        '--',
        label=r'CO$N$CEPT (non-linear $\nu$)',
    )
    ax.set_ylabel(rf'$k^3 P_{{\nu}}$')
    ax.text(0.5, 0.85, f'$a = {a}$',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=12,
        transform=ax.transAxes,
    )
axes[0].legend(loc='best').get_frame().set_alpha(0.7)
axes[-1].set_xlabel(rf'$k\, [\mathrm{{{unit_length}}}^{{-1}}]$')
plt.tight_layout(h_pad=0)
fig.subplots_adjust(hspace=0)
fig_file = f'{this_dir}/result_abs.png'
plt.savefig(fig_file)

# Check that the linear and non-linear neutrino
# power spectra are very similar.
# The further from a = a_begin, the larger the tolerance.
rel_tol_begin, rel_tol_end = 0.05, 0.25
rel_tols = (rel_tol_begin - rel_tol_end)/(a_begin - 1)*(scalefactors - 1) + rel_tol_end
k_max = 0.05*units.Mpc**(-1)
mask = (k <= k_max)
for a, P_lin, P_nonlin, rel_tol in zip(scalefactors, Pν_linear, Pν_nonlinear, rel_tols):
    if not all(isclose(
        P_lin_i,
        P_nonlin_i,
        rel_tol=rel_tol,
    ) for P_lin_i, P_nonlin_i in zip(P_lin[mask], P_nonlin[mask])):
        abort(
            f'The linear and non-linear neutrino power spectra does not match '
            f'at large scales at a = {a}.\n'
            f'See "{fig_file}" for a visualization.'
        )

# Done analyzing absolute neurino power spectra
masterprint('done')



##############################################
# Relative total power spectra, small scales #
##############################################
masterprint(f'Analyzing {this_test} (relative) data ...')
# Read in the present total CO𝘕CEPT power spectra.
# Note that in the massless neutrino case, total ≈ matter,
# as Ων is close to 0 and the neutrino power is very small
# at the small scales we are looking at here.
filename_massless = sorted([
    filename for filename in
    glob(f'{this_dir}/output_small_massless/powerspec_a=*')
    if not filename.endswith('.png')
])[-1]
k, Ptot_massless = np.loadtxt(
    filename_massless,
    usecols=(0, 2),
    unpack=True,
)
k, Ptot_linear = np.loadtxt(
    filename_massless.replace('massless', 'massive_linear'),
    usecols=(0, 4),
    unpack=True,
)
k, Ptot_nonlinear = np.loadtxt(
    filename_massless.replace('massless', 'massive_nonlinear'),
    usecols=(0, 4),
    unpack=True,
)
k              = k             [:-n_tailcut]
Ptot_massless  = Ptot_massless [:-n_tailcut]
Ptot_linear    = Ptot_linear   [:-n_tailcut]
Ptot_nonlinear = Ptot_nonlinear[:-n_tailcut]

# Read in the present CLASS power spectra from the HDF5's
# generated by the CLASS utility.
def ζ(k):
    return (π*sqrt(2*A_s)*k**(-3/2)*(k/k_pivot)**((n_s - 1)/2)
        *exp(alpha_s/4*log(k/k_pivot)**2))
with open_hdf5(f'{this_dir}/output_small_massless/class_processed.hdf5') as f:
    globals().update(f['perturbations'].attrs)
    k_class = f['perturbations/k'][:]
    δ_tot = f['perturbations/delta_cdm+b'][-1, :]
Ptot_massless_class = (δ_tot*ζ(k_class))**2
with open_hdf5(f'{this_dir}/output_small_massive_linear/class_processed.hdf5') as f:
    δ_tot = (
        + (Ωcdm + Ωb)*f['perturbations/delta_cdm+b'  ][-1, :]
        + (Ων       )*f['perturbations/delta_ncdm[0]'][-1, :]
    )/(Ωcdm + Ωb + Ων)
Ptot_massive_class = (δ_tot*ζ(k_class))**2

# Compute the relative total power spectra
# and find their minium value.
rel_linear    = (Ptot_linear         - Ptot_massless     )/Ptot_massless
rel_nonlinear = (Ptot_nonlinear     - Ptot_massless      )/Ptot_massless
rel_class     = (Ptot_massive_class - Ptot_massless_class)/Ptot_massless_class
rel_linear_min_index    = np.argmin(rel_linear   )
rel_nonlinear_min_index = np.argmin(rel_nonlinear)
rel_linear_min    = rel_linear   [rel_linear_min_index   ]
rel_nonlinear_min = rel_nonlinear[rel_nonlinear_min_index]

# Plot relative total power spectra
plt.figure()
plt.semilogx(
    k,
    100*rel_linear,
    '-',
    label=r'CO$N$CEPT (linear $\nu$)',
)
plt.semilogx(
    k,
    100*rel_nonlinear,
    '--',
    label=r'CO$N$CEPT (non-linear $\nu$)',
)
plt.semilogx(
    k_class,
    100*rel_class,
    '-k',
    label='CLASS',
)
plt.semilogx(
    k[asarray([rel_linear_min_index - 2, rel_linear_min_index + 2])],
    [100*rel_linear_min]*2,
    '-k',
    alpha=0.5,
)
plt.xlabel(rf'$k\,[\mathrm{{{unit_length}}}^{{-1}}]$')
plt.ylabel(
    rf'$'
    rf'\frac{{'
    rf'      P_{{\mathrm{{tot}}}}^{{\Sigma m_{{\nu}} = {Σmν:g}\mathrm{{eV}}}}'
    rf'    - P_{{\mathrm{{tot}}}}^{{\Sigma m_{{\nu}} = 0.0    \mathrm{{eV}}}}'
    rf'}}{{'
    rf'      P_{{\mathrm{{tot}}}}^{{\Sigma m_{{\nu}} = 0.0    \mathrm{{eV}}}}'
    rf'}}'
    rf'\, [\%]'
    rf'$'
)
plt.title(f'$a = 1$')
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
fig_file = f'{this_dir}/result_rel.png'
plt.savefig(fig_file)

# Check whether the CLASS supression around the non-linear dip
# is about the theoretically predicted value.
rel_tol = 0.05
suppression_class_theoretical = -8*Ων/(Ωcdm + Ωb + Ων)
k_dip = k[rel_linear_min_index]
suppression_class = rel_class[np.argmin(np.abs(k_dip - k_class))]
if not isclose(
    suppression_class,
    suppression_class_theoretical,
    rel_tol=rel_tol,
):
    abort(
        f'The total power suppression predicted by CLASS ({suppression_class*100:.2f}%) '
        f'at the non-linear dip (k ≈ {k_dip}/{unit_length}) is far from being equal '
        f'to the theoretical prediction -8*Ων/(Ωcdm + Ωb + Ων) '
        f'= {suppression_class_theoretical*100:.2f}%.\n'
        f'See "{fig_file}" for a visualization.'
    )

# Check whether the CO𝘕CEPT supression around the non-linear dip
# is about that found in https://arxiv.org/pdf/0802.3700.pdf .
rel_tol = 0.10
suppression_concept_theoretical = -9.8*Ων/(Ωcdm + Ωb + Ων)
suppression_concept = rel_linear_min
if not isclose(
    suppression_concept,
    suppression_concept_theoretical,
    rel_tol=rel_tol,
):
    abort(
        f'The total power suppression predicted by CO𝘕CEPT ({suppression_concept*100:.2f}%) '
        f'at the non-linear dip (k ≈ {k_dip}/{unit_length}) is far from being equal '
        f'to the prediction -9.8*Ων/(Ωcdm + Ωb + Ων) '
        f'= {suppression_concept_theoretical*100:.2f}%.\n'
        f'See "{fig_file}" for a visualization.'
    )

# Check that the supression in the linear and non-linear neutrino
# runs are very similar.
rel_tol = 0.02
if not all(isclose(
    rel_linear_i,
    rel_nonlinear_i,
    rel_tol=rel_tol,
) for rel_linear_i, rel_nonlinear_i in zip(rel_linear, rel_nonlinear)):
    abort(
        f'The total power suppression in the CO𝘕CEPT runs using linear and non-linear '
        f'neutrinos should be more similar than they are.\n'
        f'See "{fig_file}" for a visualization.'
    )

# Done analyzing relative total power spectra
masterprint('done')
