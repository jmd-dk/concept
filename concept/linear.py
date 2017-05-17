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
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from communication import domain_layout_local_indices, exchange, get_buffer, smart_mpi')
cimport('from integration import Spline, hubble')
cimport('from mesh import get_fftw_slab, domain_decompose, fft')



# Class storing a classy.Class instance
# together with the corresponding |k| values
# and results retrieved from the classy.Class instance.
class CosmoResults:
    """All methods of the cosmo object used in the code, which have
    no arguments are here written as attritubes using the magick of the
    property decorator. Methods with arguments should also be defined
    in such a way that their results are cached.
    """
    def __init__(self, k_magnitudes, cosmo, gauge):
        # Store the supplied objects
        self.k_magnitudes = k_magnitudes
        self.cosmo = cosmo
        # Initiate empty dicst for each method of cosmo with arguments
        self._transfers = {}
        # Store variables as attributes
        varnames = ['A_s', 'n_s']
        for varname, value in self.cosmo.get_current_derived_parameters(varnames).items():
            setattr(self, varname, value)
        self.gauge = gauge
        self.k_pivot = class_params.get('k_pivot', 0.05)*units.Mpc**(-1)
        # Broadcast newly added attributes
        (self.A_s,
         self.n_s,
         self.gauge,
         self.k_pivot,
         ) = bcast([getattr(self, varname)
                    for varname in varnames + ['gauge', 'k_pivot']]
                    if master else None)
    @property
    def background(self):
        if not hasattr(self, '_background'):
            self._background = self.cosmo.get_background()
        return self._background
    @property
    def perturbations(self):
        if not hasattr(self, '_perturbations'):
            self._perturbations = self.cosmo.get_perturbations()['scalar']
        return self._perturbations
    @property
    def h(self, communicate=False):
        if master:
            if not hasattr(self, '_h'):
                self._h = self.cosmo.h()
        if communicate:
            self._h = bcast(self._h if master else None)
        elif not master:
            return
        return self._h
    @property
    def a_values(self):
        if not hasattr(self, '_a_values'):
            self._a_values = 1/(1 + self.background['z'])
        return self._a_values
    def transfers(self, a):
        if not hasattr(self, '_transfers'):
            self._transfers = {}
        transfers_a = self._transfers.get(a)
        if transfers_a is None:
            z = 1/a - 1
            transfers_a = self.cosmo.get_transfer(z)
            self._transfers[a] = transfers_a
        return transfers_a
    def splines(self, y):
        if not hasattr(self, '_splines'):
            self._splines = {}
        spline = self._splines.get(y)
        if spline is None:
            spline = Spline(self.a_values, self.background[y])
            self._splines[y] = spline
        return spline
    @functools.lru_cache()
    def ρ_bar(self, species_class, a, communicate=False):
        if master:
            if species_class == 'cdm+b':
                value = self.ρ_bar('cdm', a) + self.ρ_bar('b', a)
            else:
                spline = self.splines(f'(.)rho_{species_class}')
                value = spline.eval(a)*ℝ[3/(8*π*G_Newton)*(light_speed/units.Mpc)**2]
        if communicate:
            value = bcast(value if master else None)
        elif not master:
            return
        return value
    @functools.lru_cache()
    def growth_fac_f(self, a, communicate=False):
        if master:
            spline = self.splines('gr.fac. f')
            f = spline.eval(a)
        if communicate:
            f = bcast(f if master else None)
        elif not master:
            return
        return f



# Function which solves the linear cosmology using CLASS,
# from before the initial simulation time and until the present.
@cython.pheader(# Arguments
                k_min='double',
                k_max='double',
                k_gridsize='Py_ssize_t',
                gauge='str',
                # Locals
                cosmoresults='object', # CosmoResults
                extra_params='dict',
                k_gridsize_max='Py_ssize_t',
                k_magnitudes='double[::1]',
                k_magnitudes_str='str',
                returns='object',  # CosmoResults
               )
def compute_cosmo(k_min=-1, k_max=-1, k_gridsize=-1, gauge='N-body'):
    """All calls to CLASS should be done through this function.
    If no arguments are supplied, CLASS will be run with the parameters
    stored in class_params. The return type is CosmoResults, which
    stores the result of the CLASS computation.
    If k_min, k_max are given, a more in-depth computation will be
    carried out by CLASS, where transfer functions and perturbations
    are also computed.
    All results from calls to this function are cached (using the
    global variable cosmoresults_all), so you can safely call this
    function multiple times with the same arguments without it having
    to do the same CLASS computation over and over again.
    The k_min and k_max arguments specify the |k| interval on which
    the physical quantities should be tabulated. The k_gridsize specify
    the (maximum) number of |k| values at which to do this tabulation.
    The |k| values will be distributed logarithmically.
    The gauge of the transfer functions can be specified by
    the gauge argument, which can take the values 'synchronous',
    'newtonian' and 'N-body'.
    """
    # Check that the specified gauge is valid
    gauge = gauge.replace('-', '').lower()
    if gauge not in ('synchronous', 'newtonian', 'nbody'):
        abort('The compute_cosmo function was called with a gauge of "{}", '
              'which is not implemented'.format(gauge))
    # Shrink down k_gridsize if it is too large to be handled by CLASS.
    # Also use the largest allowed value as the default value,
    # when no k_gridsize is not given.
    k_gridsize_max = (class__ARGUMENT_LENGTH_MAX_ - 1)//(len(k_float2str(0)) + 1)
    if k_gridsize > k_gridsize_max or k_gridsize == -1:
        k_gridsize = k_gridsize_max
    # If this exact CLASS computation has already been carried out,
    # return the stored results.
    cosmoresults = cosmoresults_all.get((k_min, k_max, k_gridsize, gauge))
    if cosmoresults is not None:
        return cosmoresults
    # Determine whether to run CLASS "quickly" or "fully", where only
    # the latter computes the various transfer functions and
    # perturbations.
    if k_min == -1 == k_max:
        # A quick CLASS computation should be carried out,
        # using only the minial set of parameters.
        extra_params = {}
        k_magnitudes = None
    elif k_min == -1 or k_max == -1:
        abort(f'compute_cosmo was called with k_min = {k_min}, k_max = {k_max}')
    else:
        # A full CLASS computation should be carried out.
        # Array of |k| values at which to tabulate the
        # transfer functions, in both floating and str representation
        # (the argument to CLASS must be a str).
        k_magnitudes = logspace(log10((1 - 1e-2)*k_min/units.Mpc**(-1)),
                                log10((1 + 1e-2)*k_max/units.Mpc**(-1)),
                                k_gridsize)
        np_print_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=ထ)  # Avoid ellipsis in str representation
        k_magnitudes_str = np.array2string(k_magnitudes, max_line_width=ထ,
                                                         formatter={'float': k_float2str},
                                                         separator=',',
                                                         ).strip('[]')
        np.set_printoptions(threshold=np_print_threshold)
        k_magnitudes = np.fromstring(k_magnitudes_str, sep=',')*units.Mpc**(-1)
        # Specify the extra parameters with which CLASS should be run
        extra_params = {# The |k| values to tabulate the perturbations.
                        # The transfer functions computed directly by
                        # CLASS will be on a slightly denser |k| grid.
                        'k_output_values': k_magnitudes_str,
                        # Needed for transfer functions
                        'z_pk': f'{2*universals.z_begin}, 0',
                        # Needed for perturbations
                        'output': 'dTk vTk',
                        # Needed for h (the metric perturbation)
                        'extra metric transfer functions': 'yes',
                        # Set the gauge. The N-body gauge is not
                        # implemented in CLASS, and so here we choose
                        # the synchronous gauge and do the
                        # transformation to N-body gauge later.
                        'gauge': 'synchronous' if gauge == 'nbody' else gauge,
                        }
    # Call CLASS using OpenMP.
    # Only the master process will have access to the result.
    cosmo = call_class(extra_params)
    # Wrap the result in a CosmoResults object
    # and add this to the global dict.
    cosmoresults = CosmoResults(k_magnitudes, cosmo, gauge)
    cosmoresults_all[k_min, k_max, k_gridsize, gauge] = cosmoresults
    return cosmoresults
# Dict with keys of the form (k_min, k_max, k_gridsize, gauge),
# storing the results of calls to the above function as
# CosmoResults instances.
cython.declare(cosmoresults_all='dict')
cosmoresults_all = {}

# Function which can compute transfer functions
# for variables of a supplied component.
@cython.pheader(# Arguments
                component='Component',
                variables='object',  # str, int or container of str's and ints
                k_min='double',
                k_max='double',
                k_gridsize='Py_ssize_t',
                a='double',
                gauge='str',
                # Locals
                H='double',
                a_minus='double',
                a_plus='double',
                a_values='double[::1]',
                class_perturbations='list',
                class_transfers='dict',
                class_transfers_minus='dict',
                class_transfers_plus='dict',
                cosmoresults='object',  # CosmoResults
                da='double',
                h='double',
                hʹ='double[::1]',
                i='Py_ssize_t',
                k='Py_ssize_t',
                k_gridsize_class='Py_ssize_t',
                k_magnitudes='double[::1]',
                k_magnitudes_class='double[::1]',
                k_magnitudes_use='list',
                perturbation='dict',
                species_class='str',
                spline='Spline',
                transfer='double[::1]',
                transfer_splines='list',
                transfer_δ='double[::1]',
                transfer_θ='double[::1]',
                transfer_θ_tot='double[::1]',
                transfer_σ='double[::1]',
                transfer_σ_at_k_of_a='double[::1]',
                transfers='list',
                var_index='Py_ssize_t',
                var_indices='Py_ssize_t[::1]',
                w='double',
                ρ_bbar_a='double',
                ρ_cdmbar_a='double',
                ρ_mbar_a='double',
                ȧ_minus='double',
                ȧ_plus='double',
                ȧ_transfer_θ_totʹ='double[::1]',
                returns='tuple',  # (list of Spline's, CosmoResults)
                )
def compute_transfers(component, variables, k_min, k_max, k_gridsize=-1, a=-1, gauge='N-body'):
    """This function will compute the transfer functions of the
    supplied component. The returned transfer functions will be in the
    form of Spline objects. The variable(s) of which the transfer
    function should be computed is specified by variables, valid formats
    of which is determined from Component.varnames2indices.
    The k_min and k_max arguments specify the |k| interval at which the
    transfer functions should be computed, while k_gridsize specify the
    (miminum) number of |k| values at which to do the tabulation. These
    will be distributed logarithmically.
    The scale factor at which to compute the transfer functions can be
    specified by a. If not given, the current scale factor will be used.
    Finally, the gauge of the transfer functions can be specified by
    the gauge argument, which can take the values 'synchronous',
    'newtonian' and 'N-body'.

    To get the transfer functions of ϱ and J, we use the transfer
    function computed by CLASS directly. As the tranfer function of σ
    is not included here, we extract it manually from the computed
    scalar perturbations.
    """
    # Argument processing
    var_indices = component.varnames2indices(variables)
    # If a gauge is given explicitly as a CLASS parameter in the
    # parameter file, this gauge should overwrite what ever is passed
    # to this function.
    gauge = class_params.get('gauge', gauge).replace('-', '').lower()
    # Compute the cosmology via CLASS
    # (if computed previously, this simply looks up the stored result).
    cosmoresults = compute_cosmo(k_min, k_max, k_gridsize, gauge)
    k_magnitudes = cosmoresults.k_magnitudes
    # Update k_gridsize to be what ever value was settled on
    # by the compute_cosmo function.
    k_gridsize = k_magnitudes.shape[0]
    # Set the scale factor
    if a == -1:
        a = universals.a
    # The master process alone does the work of extracting transfer
    # functions from the CLASS results.
    transfers = [] if master else [None]*var_indices.shape[0]
    k_magnitudes_class = empty(1, dtype=C2np['double'])
    if master:
        h = cosmoresults.h
        H = hubble(a)
        w = component.w(a=a)
        species_class = component.species_class
        # Get the needed data from CLASS  
        if 0 in var_indices or 1 in var_indices:
            # For ϱ and J we use the transfer functions already computed
            # by CLASS directly.
            class_transfers = cosmoresults.transfers(a)
            k_magnitudes_class = class_transfers['k (h/Mpc)']*(h/units.Mpc)
            k_gridsize_class = k_magnitudes_class.shape[0]
            if species_class == 'cdm':
                transfer_δ = class_transfers['d_cdm']
                # In synchronous gauge, transfer_θ for cdm is zero by
                # definition. In this case CLASS does not
                # produce this output. In the case of N-body gauge we do
                # the transformation ourselves from synchronous gauge,
                # and so this affects N-body gauge as well.
                if gauge in ('synchronous', 'nbody'):
                    transfer_θ = zeros(transfer_δ.shape[0], dtype=C2np['double'])
                else:
                    transfer_θ = class_transfers['t_cdm']*units.Mpc**(-1)
            elif species_class == 'cdm+b':
                # Construct total matter (combined cold dark matter
                # and baryons) transfer functions.
                ρ_cdmbar_a = cosmoresults.ρ_bar('cdm', a)
                ρ_bbar_a   = cosmoresults.ρ_bar('b', a)
                ρ_mbar_a = ρ_cdmbar_a + ρ_bbar_a
                transfer_δ = (  ρ_cdmbar_a/ρ_mbar_a*class_transfers['d_cdm']
                              + ρ_bbar_a  /ρ_mbar_a*class_transfers['d_b'  ])
                # In synchronous gauge, transfer_θ for cdm is zero by
                # definition. In this case CLASS does not
                # produce this output. In the case of N-body gauge we do
                # the transformation ourselves from synchronous gauge,
                # and so this affects N-body gauge as well.
                if gauge in ('synchronous', 'nbody'):
                    transfer_θ = (  ρ_cdmbar_a/ρ_mbar_a*0
                                  + ρ_bbar_a  /ρ_mbar_a*class_transfers['t_b'])*units.Mpc**(-1)
                else:
                    transfer_θ = (  ρ_cdmbar_a/ρ_mbar_a*class_transfers['t_cdm']
                                  + ρ_bbar_a  /ρ_mbar_a*class_transfers['t_b'  ])*units.Mpc**(-1)
            else:
                transfer_δ = class_transfers[f'd_{species_class}'] 
                transfer_θ = class_transfers[f't_{species_class}']*units.Mpc**(-1)
            transfer_θ_tot = class_transfers['t_tot']*units.Mpc**(-1)
        if 2 in var_indices:
            # For σ we use the perturbations computed by CLASS
            class_perturbations = cosmoresults.perturbations
        # Get the requested transfer functions
        # and transform them to N-body gauge.
        for var_index in var_indices:
            if var_index == 0:
                # Transform the δ transfer function to N-body gauge
                if gauge == 'nbody':
                    for k in range(k_gridsize_class):
                        transfer_δ[k] += (ℝ[3*H/light_speed*(1 + w)]
                                          *transfer_θ_tot[k]/k_magnitudes_class[k]**2)
                # Done with this transfer function
                transfers.append(transfer_δ)
            elif var_index == 1:
                # Transform the θ transfer function to N-body gauge
                if gauge == 'nbody':
                    # Get the conformal time derivative
                    # of the metric perturbation.
                    hʹ = class_transfers['h_prime']*(light_speed/units.Mpc)
                    # To to this we need (a*H*T_θ) = (ȧ*T_θ)
                    # differentiated with respect to the conformal time
                    # (with T_θ the total θ transfer function) τ.
                    # We have ʹ = d/dτ = a*d/dt = a²H*d/da.
                    da = 1e-6*a  # Arbitrary but small scale factor step
                    a_plus  = a + da
                    a_minus = a - da
                    ȧ_plus  = a_plus *hubble(a_plus)
                    ȧ_minus = a_minus*hubble(a_minus)
                    class_transfers_plus  = cosmoresults.transfers(a_plus)
                    class_transfers_minus = cosmoresults.transfers(a_minus)
                    ȧ_transfer_θ_totʹ = a**2*H*0.5/da*units.Mpc**(-1)*(
                                              ȧ_plus *class_transfers_plus ['t_tot']
                                            - ȧ_minus*class_transfers_minus['t_tot'])
                    # Now do the gauge transformation
                    for k in range(k_gridsize_class):
                        transfer_θ[k] += (0.5/light_speed*hʹ[k]
                                          - ℝ[3/light_speed**2]*ȧ_transfer_θ_totʹ[k]
                                            /k_magnitudes_class[k]**2)
                # Done with this transfer function
                transfers.append(transfer_θ)
            elif var_index == 2:
                # Get the σ transfer function at the current time
                transfer_σ = empty(k_gridsize, dtype=C2np['double'])
                for k in range(k_gridsize):
                    perturbation = class_perturbations[k]
                    a_values = perturbation['a']
                    transfer_σ_at_k_of_a = perturbation[f'shear_{species_class}']  # UNITS? !!!
                    # Interpolate transfer(a_values) to the
                    # current time. As only this single interpolation is
                    # needed for this set of {a_values, transfer},
                    # use NumPy to do the interpolation (this is faster
                    # than GSL because we do not compute a spline
                    # over the entire function).
                    transfer_σ[k] = np.interp(a, a_values, transfer_σ_at_k_of_a)
                # As σ is the same in N-body and synchronous gauge,
                # we do not have to do any transformation in the case
                # of N-body gauge.
                # Done with this transfer function
                transfers.append(transfer_σ)
    # The master process is now done processing the transfer functions.
    # Broadcast the |k| grid from class.
    k_magnitudes_class = smart_mpi(k_magnitudes_class, -1, mpifun='bcast')
    # Specification of which of the two |k| arrays
    # each transfer function should be paired with.
    k_magnitudes_use = [k_magnitudes_class,  # ϱ
                        k_magnitudes_class,  # J
                        k_magnitudes      ,  # σ
                        ]
    # Broadcast the transfer functions
    # and construct a Spline object for each.
    transfer_splines = []
    for i, (var_index, transfer) in enumerate(zip(var_indices, transfers)):
        transfer = smart_mpi(transfer, i, mpifun='bcast') 
        transfer_splines.append(Spline(k_magnitudes_use[var_index], transfer))
    return transfer_splines, cosmoresults
# Helper function used in compute_transfers
def k_float2str(k):
    return f'{k:.3e}'

# Function which realises a given variable on a component
# from a supplied transfer function.
@cython.pheader(# Arguments
               component='Component',
               variable='object',  # str or int
               transfer_spline='Spline',
               cosmoresults='object',  # CosmoResults
               a='double',
               # Locals
               A_s='double',
               H='double',
               J_scalargrid='double*',
               buffer_number='Py_ssize_t',
               dim='int',
               displacement='double',
               domain_size_i='Py_ssize_t',
               domain_size_j='Py_ssize_t',
               domain_size_k='Py_ssize_t',
               domain_start_i='Py_ssize_t',
               domain_start_j='Py_ssize_t',
               domain_start_k='Py_ssize_t',
               f_growth='double',
               fluid_index='Py_ssize_t',
               fluidscalar='FluidScalar',
               fluidvar='object',  # np.ndarray
               fluidvar_name='str',
               gridsize='Py_ssize_t',
               i='Py_ssize_t',
               i_global='Py_ssize_t',
               index='Py_ssize_t',
               index0='Py_ssize_t',
               index1='Py_ssize_t',
               j='Py_ssize_t',
               j_global='Py_ssize_t',
               k='Py_ssize_t',
               k_dim0='Py_ssize_t',
               k_dim1='Py_ssize_t',
               k_factor='double',
               k_global='Py_ssize_t',
               k_gridvec='Py_ssize_t[::1]',
               k_magnitude='double',
               k_pivot='double',
               ki='Py_ssize_t',
               kj='Py_ssize_t',
               kj2='Py_ssize_t',
               kk='Py_ssize_t',
               k2='Py_ssize_t',
               k2_max='Py_ssize_t',
               mass='double',
               mom_dim='double*',
               multi_index='tuple',
               n_s='double',
               nyquist='Py_ssize_t',
               plane_dc='double[:, :,::1]',
               plane_nyquist='double[:, :,::1]',
               pos_dim='double*',
               pos_gridpoint='double',
               random_im='double',
               random_re='double',
               slab='double[:, :, ::1]',
               slab_jik='double*',
               species_class='str',
               sqrt_power='double',
               sqrt_power_common='double[::1]',
               transfer='double',
               w='double',
               ρ_bar_a='double',
               ψ_dim='double[:, :, ::1]',
               ψ_dim_noghosts='double[:, :, :]',
               ϱ='double*',
               )
def realize(component, variable, transfer_spline, cosmoresults, a=-1):
    """This function realizes a single variable of a component,
    given the transfer function as a Spline (using |k| in physical units
    as the independent variable) and the corresponding CosmoResults
    object, which carry additional information from the CLASS run that
    produced the transfer function. If you want a realization at a time
    different from the present you may specify an a.
    If a particle component is given, the Zeldovich approximation is
    used to distribute the paricles and assign momenta. This is
    done simultaneously, meaning that you cannot realize only the
    positions or only the momenta. For the particle realization to
    work correctly, you must pass the δ transfer function as
    transfer_spline. For particle components, the variable argument
    is not used.
    For both particle and fluid components it is assumed that the
    passed component is of the correct size beforehand. No resizing
    will take place in this function.
    """
    if a == -1:
        a = universals.a
    # Determine the gridsize of the grid used to do the realization
    gridsize = (component.gridsize if component.representation == 'fluid'
                                   else int(round(cbrt(component.N))))
    if gridsize%nprocs != 0:
        abort(f'The realization uses a gridsize of {gridsize}, '
              f'which is not evenly divisible by {nprocs} processes.'
              )
    # Fetch a slab decomposed grid
    slab = get_fftw_slab(gridsize)
    # Get the index of the fluid variable to be realized
    # and print out progress message.
    if component.representation == 'particles':
        # For particles, the Zeldovich approximation is used for
        # realization. This realizes both positions and momenta.
        # This means that the value of the passed variable argument
        # does not matter. To realize all three components of positions
        # and momenta, we need the fluid_index to have a value of 1
        # (corresponding to J or mom), so that multi_index takes on
        # vector values ((0, ), (1, ), (2, )).
        fluid_index = 1
        masterprint(f'Realizing particles of {component.name} ...')
    elif component.representation == 'fluid':
        fluid_index = component.varnames2indices(variable, single=True)
        fluidvar_name = component.fluid_names['ordered'][fluid_index]
        masterprint(f'Realizing fluid variable {fluidvar_name} of {component.name} ...')
    # Extract some variables
    nyquist = gridsize//2
    species_class = component.species_class
    w = component.w(a=a)
    H = hubble(a)
    A_s = cosmoresults.A_s
    n_s = cosmoresults.n_s
    k_pivot = cosmoresults.k_pivot
    # Fill array with values of the common factor
    # used in all realizations, regardless of the variable.
    k2_max = 3*(gridsize//2)**2  # Max |k|² in grid units
    buffer_number = 0
    sqrt_power_common = get_buffer(k2_max + 1, buffer_number)
    for k2 in range(1, k2_max + 1):
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        transfer = transfer_spline.eval(k_magnitude)
        sqrt_power_common[k2] = (
            # Factors from the actual realization
            k_magnitude**ℝ[0.5*n_s - 2]*transfer
            *ℝ[sqrt(2*A_s)*π*k_pivot**(0.5 - 0.5*n_s)
               # Normalization due to IFFT
               *boxsize**(-1.5)
               # Normalization of the generated random numbers:
               # <rg(0, 1)²> = 1 → <|rg(0, 1)/√2 + i*rg(0, 1)/√2|²> = 1,
               # where rg(0, 1) = random_gaussian(0, 1).
               *1/sqrt(2)
               ])
    # At |k| = 0, the power should be zero, corresponding to a
    # real-space mean value of zero of the realized variable.
    sqrt_power_common[0] = 0
    # Make the DC and Nyquist planes of random numbers,
    # respecting the complex-conjugate symmetry. These will be
    # allocated in full on all processes. A seed of master_seed + nprocs
    # (and the next, master_seed + nprocs + 1) is used, as the highest
    # process_seed will be equal to master_seed + nprocs - 1, meaning
    # that this new seed will not collide with any of the individual
    # process seeds.
    plane_dc      = create_symmetric_plane(gridsize, seed=(master_seed + nprocs + 0))
    plane_nyquist = create_symmetric_plane(gridsize, seed=(master_seed + nprocs + 1))
    # Allocate 3-vector which will store componens
    # of the k vector (grid units).
    k_gridvec = empty(3, dtype=C2np['Py_ssize_t'])
    # Loop over all fluid scalars of the fluid variable
    fluidvar = component.fluidvars[fluid_index]
    for multi_index in component.iterate_fluidscalar_indices(fluidvar):
        # Extract individual indices from multi_index
        if ℤ[len(multi_index)] > 0:
            index0 = multi_index[0]
        if ℤ[len(multi_index)] > 1:
            index1 = multi_index[1]
        # Set k_dim's. Their initial values are not important,
        # but they should compare false with the nyquist variable.
        k_dim0 = k_dim1 = -1
        # Reset the pseudo-random number generator, ensuring that
        # a given grid point always get the same random number
        # regarless of the fluid variable or scalar.
        seed_rng()
        # Loop through the local j-dimension
        for j in range(ℤ[slab.shape[0]]):
            # The j-component of the wave vector (grid units).
            # Since the slabs are distributed along the j-dimension,
            # an offset must be used.
            j_global = ℤ[slab.shape[0]*rank] + j
            if j_global > ℤ[gridsize//2]:
                kj = j_global - gridsize
            else:
                kj = j_global
            kj2 = kj**2
            # Loop through the complete i-dimension
            for i in range(gridsize):
                # The i-component of the wave vector (grid units)
                if i > ℤ[gridsize//2]:
                    ki = i - gridsize
                else:
                    ki = i
                # Loop through the complete, padded k-dimension
                # in steps of 2 (one complex number at a time).
                for k in range(0, ℤ[slab.shape[2]], 2):
                    # The k-component of the wave vector (grid units)
                    kk = k//2
                    # The squared magnitude of the wave vector
                    # (grid units).
                    k2 = ℤ[ki**2 + kj2] + kk**2
                    # Pack components of k vector (grid units)
                    k_gridvec[0] = ki
                    k_gridvec[1] = kj
                    k_gridvec[2] = kk
                    # Compute the factor which depend on the wave
                    # vector. Regardless of the variable to realize,
                    # at |k| = 0 the power should be zero.
                    with unswitch(3):
                        if ℤ[len(multi_index)] == 1:
                            # Vector quantity.
                            # The needed factor is kᵢ/k².
                            if k2 == 0:
                                k_factor = 0
                            else:
                                k_dim0 = k_gridvec[index0]
                                k_factor = (ℝ[boxsize/(2*π)]*k_dim0)/k2
                        elif ℤ[len(multi_index)] == 2:
                            # Rank 2 tensor quantity.
                            # The needed factor is 3/2(kᵢkⱼ/k² - δᵢⱼ/3).
                            if k2 == 0:
                                k_factor = 0
                            else:
                                k_dim0 = k_gridvec[index0]
                                k_dim1 = k_gridvec[index1]
                                k_factor = (1.5*k_dim0*k_dim1/k2) - 0.5*(index0 == index1)
                    # At some grid points, the complex-conjugate
                    # symmetry requires that the power vanises.
                    # We fulfill this requirement by letting
                    # k_factor vanish at these grid points.
                    if (kk == 0 or kk == nyquist) and (   k_dim0 == nyquist
                                                       or k_dim1 == nyquist):
                        k_factor = 0
                    # Draw two random numbers from a Gaussian
                    # distribution with mean 0 and spread 1.
                    # On the lowest kk (kk = 0, (DC)) and highest kk
                    # (kk = gridsize/2 (Nyquist)) planes we need to
                    # ensure that the complex-conjugate symmetry holds.
                    if kk == 0:
                        random_re = plane_dc[j_global, i, 0]
                        random_im = plane_dc[j_global, i, 1]
                    elif kk == nyquist:
                        random_re = plane_nyquist[j_global, i, 0]
                        random_im = plane_nyquist[j_global, i, 1]
                    else:
                        random_re = random_gaussian(0, 1)
                        random_im = random_gaussian(0, 1)
                    # Pointer to the [j, i, k]'th element of the slab.
                    # The complex number is then given as
                    # Re = slab_jik[0], Im = slab_jik[1].
                    slab_jik = cython.address(slab[j, i, k:])
                    # Populate slab_jik dependent on the component
                    # representation and the fluid_index (and also
                    # multi_index through the already defined k_factor).
                    with unswitch(3):
                        if component.representation == 'particles':
                            # Realize the displacement field ψ
                            sqrt_power = k_factor*sqrt_power_common[k2]
                            slab_jik[0] = -sqrt_power*random_im
                            slab_jik[1] =  sqrt_power*random_re
                        elif component.representation == 'fluid':
                            with unswitch(3):
                                if fluid_index == 0:
                                    # Realize δ
                                    sqrt_power = sqrt_power_common[k2]
                                    slab_jik[0] = sqrt_power*random_re
                                    slab_jik[1] = sqrt_power*random_im
                                elif fluid_index == 1:
                                    # Realize the component of the 
                                    # velocity field u
                                    # given by multi_index.
                                    sqrt_power = light_speed*k_factor*sqrt_power_common[k2]
                                    slab_jik[0] =  sqrt_power*random_im
                                    slab_jik[1] = -sqrt_power*random_re
                                elif fluid_index == 2:
                                    # Realize the component of the
                                    # stress tensor σ
                                    # gviven by multi_index.
                                    sqrt_power = k_factor*sqrt_power_common[k2]
                                    slab_jik[0] = sqrt_power*random_re
                                    slab_jik[1] = sqrt_power*random_im
        # Fourier transform the slabs to coordinate space.
        # Now the slabs store the realized fluid grid.
        fft(slab, 'backward')
        # Populate the fluid grids for fluid components,
        # and create the particles via the zeldovich approximation
        # for particles.
        if component.representation == 'fluid':
            # Communicate the fluid realization stored in the slabs to
            # the designated fluid scalar grid. This also populates the
            # pseudo and ghost points.
            fluidscalar = fluidvar[multi_index]
            domain_decompose(slab, fluidscalar.grid_mv)
            # Transform the realized fluid variable to the actual
            # quantity used in the fluid equations in conservation form.
            if fluid_index == 0:
                # δ → ϱ = a³⁽¹⁺ʷ⁾ρ = a³⁽¹⁺ʷ⁾ρ_bar(1 + δ)
                ρ_bar_a = cosmoresults.ρ_bar(species_class, a, communicate=True)
                ϱ = fluidscalar.grid
                for i in range(component.size):
                    ϱ[i] = ℝ[a**(3*(1 + w))*ρ_bar_a]*(1 + ϱ[i])
            elif fluid_index == 1:
                # u → J = a⁴ρu = a¹⁻³ʷϱu
                ϱ = component.ϱ.grid
                J_scalargrid = fluidscalar.grid
                for i in range(component.size):
                    J_scalargrid[i] *= ℝ[a**(1 - 3*w)]*ϱ[i]
            elif fluid_index == 2:
                # σ already in the correct form
                ...
            # Continue with the next fluidscalar
            continue
        # Below follows the Zeldovich approximation
        # for particle components.
        # Domain-decompose the realization of the displacement field
        # stored in the slabs. The resultant domain (vector) grid is
        # denoted ψ, wheres a single component of this vector field is
        # denoted ψ_dim.
        # Note that we could have skipped this and used the slab grid
        # directly. However, because a single component of the ψ grid
        # contains the information of both the positions and momenta in
        # the given direction, we minimize the needed communication by
        # communicating ψ, rather than the particles after
        # the realization.
        # Importantly, use a buffer different from the one given by
        # buffer_number, as this is already in use by sqrt_power_common.
        ψ_dim = domain_decompose(slab, buffer_number + 1)
        ψ_dim_noghosts = ψ_dim[2:(ψ_dim.shape[0] - 2),
                               2:(ψ_dim.shape[1] - 2),
                               2:(ψ_dim.shape[2] - 2)]
        # Determine and set the mass of the particles
        # if this is still unset.
        if component.mass == -1:
            ρ_bar_a = cosmoresults.ρ_bar(species_class, a, communicate=True)
            component.mass = a**3*ρ_bar_a*boxsize**3/component.N
        mass = component.mass
        # Get f_growth = H⁻¹Ḋ/D, where D is the linear growth factor
        f_growth = cosmoresults.growth_fac_f(a, communicate=True)
        # Apply the Zeldovich approximation 
        dim = multi_index[0]
        pos_dim = component.pos[dim]
        mom_dim = component.mom[dim]
        domain_size_i = ψ_dim_noghosts.shape[0] - 1
        domain_size_j = ψ_dim_noghosts.shape[1] - 1
        domain_size_k = ψ_dim_noghosts.shape[2] - 1
        domain_start_i = domain_layout_local_indices[0]*domain_size_i
        domain_start_j = domain_layout_local_indices[1]*domain_size_j
        domain_start_k = domain_layout_local_indices[2]*domain_size_k
        index = 0
        for         i in range(ℤ[ψ_dim_noghosts.shape[0] - 1]):
            for     j in range(ℤ[ψ_dim_noghosts.shape[1] - 1]):
                for k in range(ℤ[ψ_dim_noghosts.shape[2] - 1]):
                    # The global x, y or z coordinate at this grid point
                    with unswitch(3):
                        if dim == 0:
                            i_global = domain_start_i + i
                            pos_gridpoint = i_global*boxsize/gridsize
                        elif dim == 1:
                            j_global = domain_start_j + j
                            pos_gridpoint = j_global*boxsize/gridsize
                        elif dim == 2:
                            k_global = domain_start_k + k
                            pos_gridpoint = k_global*boxsize/gridsize
                    # Displace the position of particle
                    # at grid point (i, j, k).
                    displacement = ψ_dim_noghosts[i, j, k]
                    pos_dim[index] = mod(pos_gridpoint + displacement, boxsize)
                    # Assign momentum corresponding to the displacement
                    mom_dim[index] = displacement*ℝ[f_growth*H*mass*a**2]
                    index += 1
    # Done realizing this variable
    masterprint('done')
    # After realizing particles, most of them will be on the correct
    # process in charge of the domain in which they are located. Those
    # near the domain boundaries might get displaced outside of its
    # original domain, and so we do need to do an exchange.
    if component.representation == 'particles':
        exchange(component, reset_buffers=True)

# Function for creating the lower and upper Fourier xy-planes
# with complex-conjugate symmetry.
@cython.header(# Arguments
               gridsize='Py_ssize_t',
               seed='unsigned long int',
               # Locals
               plane='double[:, :, ::1]',
               i='Py_ssize_t',
               i_conj='Py_ssize_t',
               j='Py_ssize_t',
               j_conj='Py_ssize_t',
               returns='double[:, :, ::1]',
               )
def create_symmetric_plane(gridsize, seed=0):
    """If a seed is passed, the pseudo-random number generator will
    be seeded with this seed before the creation of the plane.
    """
    # Fetch plane if already computed
    plane = symmetric_planes.get((gridsize, seed))
    if plane is not None:
        return plane
    # Seed the pseudo-random number generator
    if seed != 0:
        seed_rng(seed)
    # Create the plane and populate it with Gaussian distributed
    # random numbers with mean 0 and spread 1.
    plane = empty((gridsize, gridsize, 2), dtype=C2np['double'])
    for     j in range(gridsize):
        for i in range(gridsize):
            plane[j, i, 0] = random_gaussian(0, 1)
            plane[j, i, 1] = random_gaussian(0, 1)
    # Enforce the symmetry plane[k_vec] = plane[-k_vec]*,
    # where * means complex conjugation.
    for j in range(gridsize//2 + 1):
        j_conj = 0 if j == 0 else gridsize - j
        for i in range(gridsize):
            i_conj = 0 if i == 0 else gridsize - i
            if i == i_conj and j == j_conj:
                # At origin of xy-plane.
                # Here the value should be purely real.
                plane[j, i, 1] = 0
            else:
                plane[j, i, 0] = +plane[j_conj, i_conj, 0]
                plane[j, i, 1] = -plane[j_conj, i_conj, 1]
    # Store and return the plane
    if seed != 0:
        symmetric_planes[gridsize, seed] = plane
    return plane
# Cache for symmetry planes, the key being (gridsize, seed)
cython.declare(symmetric_planes='dict')
symmetric_planes = {}



# Read in definition from CLASS source file at import time
cython.declare(class__ARGUMENT_LENGTH_MAX_='Py_ssize_t')
try:
    with open('{}/include/parser.h'.format(paths['class_dir']), 'r') as parser_file:    
        class__ARGUMENT_LENGTH_MAX_ = int(re.search('#define\s+_ARGUMENT_LENGTH_MAX_\s+(.*?)\s',
                                                    parser_file.read()
                                                    ).group(1))
except:
    class__ARGUMENT_LENGTH_MAX_ = 1024
