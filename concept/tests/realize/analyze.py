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

# Read in powerspectra
powerspecs_all = {'particles': {}, 'fluid': {}}
for kind in ('particles', 'fluid'):
    for fname in sorted(glob('{}/output_{}/powerspec*'.format(this_dir, kind))):
        n = int(re.search('nprocs=(.*)_a=', fname).group(1))
        k, modes, power = np.loadtxt(fname, unpack=True)
        powerspecs_all[kind][n] = (k, power)
k_values = k
n_values = list(powerspecs_all['particles'].keys())

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Plot powerspectra from realized components
fig_file = this_dir + '/result.png'
fig = plt.figure(figsize=(8, 6))
for kind, powerspecs in powerspecs_all.items():
    for n, (k, power) in powerspecs.items():
        plt.loglog(k, power, alpha=0.7,
                   label='{}, nprocs = {}'.format(kind, n))
# Compute and plot powerspectrum from CLASS
a = a_begin
z = 1/a - 1
class_params_specialized = class_params.copy()
class_params_specialized.update({'output'         : 'mPk',
                                 'k_output_values': '{}, {}'.format(min(k_values)/units.Mpc**(-1),
                                                                    max(k_values)/units.Mpc**(-1)),
                                 'z_pk'           : str(z),
                                 })
cosmo = Class()
cosmo.set(class_params_specialized)
cosmo.compute()
power_class = asarray([cosmo.pk(k/units.Mpc**(-1), z) for k in k_values])*units.Mpc**3
plt.loglog(k_values, power_class, 'k', linewidth=2, label='CLASS')
plt.xlabel(r'$k$ $\mathrm{{[{}]}}^{{-1}}$'.format(unit_length))
plt.ylabel(r'matter power $\mathrm{{[{}^3]}}$'.format(unit_length))
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# Compare the power spectra of the realizations with
# the power spectrum from CLASS.
# For particles, only the large scale power
# should be comparable to the CLASS power due to the
# CIC deconvolution.
k_min = min(k_values)
k_max = max(k_values)
masks = {'particles': np.logical_and(k_values > k_min + 0.05*(k_max  - k_min),
                                     k_values < k_max - 0.8*(k_max  - k_min)),
         'fluid': np.logical_and(k_values > k_min + 0.05*(k_max  - k_min),
                                 k_values < k_max - 0.05*(k_max  - k_min)),
         }
k_values_trimmed = {'particles': k_values[masks['particles']],
                    'fluid'    : k_values[masks['fluid'    ]],
                    }
power_class_trimmed = {'particles': power_class[masks['particles']],
                       'fluid'    : power_class[masks['fluid'    ]],
                       }
rel_tol = {'particles': 0.04, 'fluid': 0.02}
for kind in ('particles', 'fluid'):
    for n in n_values:
        k, power = powerspecs_all[kind][n]
        power_trimmed = power[masks[kind]]
        rel_realisation_noise = mean(
            abs((power_trimmed - power_class_trimmed[kind])/power_class_trimmed[kind])
        )
        if rel_realisation_noise > rel_tol[kind]:
            abort(
                f'Power spectrum of realized matter {kind} with {n} processes '
                f'disagree with that of CLASS.\n'
                f'See "{fig_file}" for a visualization.'
            )

# Done analyzing
masterprint('done')
