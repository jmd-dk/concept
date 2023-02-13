# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2023 Jeppe Mosgaard Dakin.
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
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport(
    'from communication import '
    '    exchange,             '
    '    get_buffer,           '
)
cimport('from integration import hubble')
cimport(
    'from linear import                         '
    '    compute_cosmo,                         '
    '    compute_transfer,                      '
    '    get_primordial_curvature_perturbation, '
)
cimport(
    'from mesh import         '
    '    Lattice,             '
    '    domain_decompose,    '
    '    domain_loop,         '
    '    fft,                 '
    '    fourier_curve_loop,  '
    '    fourier_loop,        '
    '    get_fftw_slab,       '
    '    get_gridshape_local, '
    '    nullify_modes,       '
    '    slab_decompose,      '
)

# Pure Python imports
from communication import get_domain_info



# Class storing the internal state for generation of pseudo-random
# numbers and implementing probability distributions.
@cython.cclass
class PseudoRandomNumberGenerator:
    # Find all bit stream generators available in NumPy,
    # e.g. 'PCG64DXSM' (Permuted Congruential Generator)
    # and 'MT19937' (Mersenne Twister).
    streams = {}
    for name, attr in vars(np.random).items():
        if attr is np.random.BitGenerator:
            continue
        try:
            if not issubclass(attr, np.random.BitGenerator):
                continue
        except TypeError:
            continue
        streams[name] = attr

    # Initialisation method
    @cython.pheader(
        # Arguments
        seed=object,  # Python int or None
        stream=str,
        cache_size='Py_ssize_t',
    )
    def __init__(self, seed=None, stream=random_generator, cache_size=2**12):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the RandomNumberGenerator type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public object seed  # Python int or None
        public str stream
        Py_ssize_t cache_size
        object bit_generator  # np.random.BitGenerator
        object generator  # np.random.Generator
        double[::1] cache_uniform
        double[::1] cache_gaussian
        double[::1] cache_rayleigh
        Py_ssize_t index_uniform
        Py_ssize_t index_gaussian
        Py_ssize_t index_rayleigh
        """
        if seed is not None:
            # Sprincle in magic number to avoid issues
            # with seed containing many zero bits.
            seed += int(π*1e+8)
        self.seed = seed
        self.stream = stream
        self.cache_size = cache_size
        # Look up requested bit stream generator
        bit_generator = self.streams.get(stream)
        if bit_generator is None and stream == 'PCG64DXSM':
            # Older versions of NumPy do not have the DXSM version
            # of PCG64. Allow falling back to the older PCG64 version.
            masterwarn(
                f'Pseudo-random bit generator "{stream}" not available in NumPy. '
                f'Falling back to "PCG64".'
            )
            stream = 'PCG64'
            bit_generator = self.streams.get(stream)
        if bit_generator is None:
            streams_str = ', '.join([f'"{stream}"' for stream in self.streams])
            abort(
                f'Pseudo-random bit generator "{stream}" not available in NumPy. '
                f'The available ones are {streams_str}.'
            )
        self.bit_generator = bit_generator(self.seed)
        # Instantiate a seeded pseudo-random number generator
        self.generator = np.random.Generator(self.bit_generator)
        # Initialise caches
        self.cache_uniform  = None
        self.cache_gaussian = None
        self.cache_rayleigh = None
        self.index_uniform  = self.cache_size - 1
        self.index_gaussian = self.cache_size - 1
        self.index_rayleigh = self.cache_size - 1

    # Uniform distribution over the half-open interval [low, high)
    @cython.header(
        # Arguments
        low='double',
        high='double',
        # Locals
        x='double',
        returns='double',
    )
    def uniform(self, low=0, high=1):
        self.index_uniform += 1
        if self.index_uniform == self.cache_size:
            self.index_uniform = 0
            # Draw new batch of uniform pseudo-random numbers
            # in the half-open interval [0, 1).
            self.cache_uniform = self.generator.uniform(0, 1, size=self.cache_size)
        # Look up in cache
        x = self.cache_uniform[self.index_uniform]
        # Transform
        x = low + x*(high - low)
        return x

    # Gaussian distribution with standard deviation
    # given by scale and mean 0.
    @cython.header(
        # Arguments
        scale='double',
        # Locals
        x='double',
        returns='double',
    )
    def gaussian(self, scale=1):
        self.index_gaussian += 1
        if self.index_gaussian == self.cache_size:
            self.index_gaussian = 0
            # Draw new batch of Gaussian pseudo-random numbers
            # with unit standard deviation and mean 0.
            self.cache_gaussian = self.generator.normal(0, 1, size=self.cache_size)
        # Look up in cache
        x = self.cache_gaussian[self.index_gaussian]
        # Transform
        x *= scale
        return x

    # Rayleigh distribution
    @cython.header(
        # Arguments
        scale='double',
        # Locals
        x='double',
        returns='double',
    )
    def rayleigh(self, scale=1):
        self.index_rayleigh += 1
        if self.index_rayleigh == self.cache_size:
            self.index_rayleigh = 0
            # Draw new batch of Rayleigh pseudo-random numbers
            # with unit scale.
            self.cache_rayleigh = self.generator.rayleigh(1, size=self.cache_size)
        # Look up in cache
        x = self.cache_rayleigh[self.index_rayleigh]
        # Transform
        x *= scale
        return x

# Instantiate pseudo-random number generator with a unique
# seed on each process, meant for general-purpose use.
# Also wrap its methods in easy to use but badly performing functions.
cython.declare(prng_general='PseudoRandomNumberGenerator')
prng_general = PseudoRandomNumberGenerator(
    random_seeds['general'] + rank,
)
@cython.header(
    # Arguments
    distribution=str,
    size=object,  # int or tuple of ints
    a='double',
    b='double',
    # Locals
    data='double[::1]',
    i='Py_ssize_t',
    shape=tuple,
    returns=object,  # double or np.ndarray
)
def random_general(distribution, size, a=0, b=0):
    shape = tuple(any2list(size))
    size = np.prod(shape)
    data = empty(size, dtype=C2np['double'])
    for i in range(size):
        with unswitch:
            if distribution == 'uniform':
                data[i] = prng_general.uniform(a, b)
            elif distribution == 'gaussian':
                data[i] = prng_general.gaussian(a)
            elif distribution == 'rayleigh':
                data[i] = prng_general.rayleigh(a)
            else:
                abort(f'random_general() got unknown distribution = "{distribution}"')
    if size == 1:
        return data[0]
    else:
        return asarray(data).reshape(shape)
def random_uniform(low=0, high=1, size=1):
    return random_general('uniform', size, low, high)
def random_gaussian(scale=1, size=1):
    return random_general('gaussian', size, scale)
def random_rayleigh(scale=1, size=1):
    return random_general('rayleigh', size, scale)

# Function for fully realizing a particle component,
# of for realising one or more variables on a fluid component.
@cython.pheader(
    # Arguments
    component='Component',
    a='double',
    a_next='double',
    variables=object,  # str or int, or sequence of strs and/or ints
    multi_indices=object,  # int, str, tuple or list
    use_gridˣ='bint',
    # Locals
    fluidvar_name=str,
    gauge=str,
    gauge_prev=str,
    multi_index=object,  # int, str or tuple
    multi_indices_var=object,  # int, str, tuple or list
    variable='int',
    returns='void',
)
def realize(
    component,
    a=-1, a_next=-1, variables=None, multi_indices=None, use_gridˣ=False,
):
    if a == -1:
        a = universals.a
    # Process the passed variables
    if variables is None:
        # Realise all variables of the component
        variables = list(arange(component.boltzmann_order + 1))
        if not variables:
            # This is a completely linear component
            # (Boltzmann order = -1). We do not realize such components
            # when no variables (only variable 0 allowed) are provided.
            return
    variables = any2list(component.varnames2indices(variables))
    # Check that the gauge to be used now is the same
    # as used for previous realisations.
    gauge = component.realization_options['gauge']
    for gauge_prev, component_names in gauges_used.items():
        if gauge != gauge_prev:
            component_names = ', '.join([
                f'"{component_name}"'
                for component_name in component_names
            ])
            masterwarn(
                f'Component "{component.name}" is to be realised in {gauge} gauge, '
                f'but the following components have already been realised '
                f'in {gauge_prev} gauge: {component_names}'
            )
            break
    gauges_used[gauge].add(component.name)
    # Delegate to representation specific realisation functions
    if component.representation == 'particles':
        if a_next != -1:
            abort(f'Cannot perform particle realization with a_next = {a_next}')
        if multi_indices is not None:
            abort('Only complete particle realization is allowed')
        # Realise particle positions and momenta in all three dimensions
        realize_particles(component, a)
        return
    for variable in variables:
        if variable > component.boltzmann_order + (component.boltzmann_closure == 'class'):
            # Requested fluid variable exceeds the
            # Boltzmann hierarchy of the component.
            continue
        multi_indices_var = multi_indices
        if multi_indices_var is None:
            multi_indices_var = []
            if variable == 2:
                multi_indices_var.append('trace')
            multi_indices_var += list(component.fluidvars[variable].multi_indices)
        if isinstance(multi_indices_var, tuple):
            multi_indices_var = [multi_indices_var]
        multi_indices_var = any2list(multi_indices_var)
        fluidvar_name = component.fluid_names['ordered'][variable]
        if variable > 0 and len(multi_indices_var) == 1:
            fluidvar_name += "['{}']".format(
                str(multi_indices_var[0]).strip('()')
            ).replace("'", "'"*isinstance(multi_indices_var[0], str))
        masterprint(
            f'Realising {fluidvar_name} of {component.name} '
            f'with grid size {component.gridsize} ...'
        )
        for multi_index in multi_indices_var:
            # Realise specific fluid variable
            realize_fluid(component, a, a_next, variable, multi_index, use_gridˣ)
        masterprint('done')
# Store keeping track of gauges within which
# components have been realised.
cython.declare(gauges_used=object)
gauges_used = collections.defaultdict(set)

# Function for realising fluid variables of fluid components
@cython.header(
    # Arguments
    component='Component',
    a='double',
    a_next='double',
    variable='int',
    multi_index=object,  # int, str or tuple
    use_gridˣ='bint',
    # Locals
    Jⁱ_ptr='double*',
    amplitudes='double[::1]',
    compound='bint',
    fluidscalar='FluidScalar',
    gridsize='Py_ssize_t',
    index='Py_ssize_t',
    nongaussianity='double',
    options=dict,
    ptr='double*',
    shape=tuple,
    slab='double[:, :, ::1]',
    value='double',
    w='double',
    w_eff='double',
    δ_min='double',
    ϱ_bar='double',
    ϱ_ptr='double*',
    ςⁱⱼ_ptr='double*',
    𝒫_ptr='double*',
    returns='void',
)
def realize_fluid(component, a, a_next, variable, multi_index, use_gridˣ=False):
    if component.representation != 'fluid':
        abort(f'realize_fluid() called with non-fluid component {component.name}')
    # Resize particle data attributes
    gridsize = component.gridsize
    shape = tuple([gridsize//domain_subdivisions[dim] for dim in range(3)])
    component.resize(shape)
    # If an approximation should be used for the realisation,
    # do so and return now.
    if realize_approximative(component, a, variable, multi_index, use_gridˣ):
        return
    # Fetch amplitudes
    amplitudes = get_amplitudes(gridsize, component, a, a_next, variable, multi_index)
    # Realise the fluid scalar variable
    slab = realize_grid(gridsize, component, a, amplitudes, variable, multi_index)
    # Communicate the fluid realisation in the slabs to the designated
    # fluid scalar grid. This also populates the ghost points.
    fluidscalar = component.fluidvars[variable][multi_index]
    domain_decompose(slab, fluidscalar.gridˣ_mv if use_gridˣ else fluidscalar.grid_mv)
    # Transform the realised fluid variable to the actual quantity used
    # in the non-linear fluid equations. Include ghost points.
    options = component.realization_options
    nongaussianity = options['nongaussianity']
    compound = (variable == component.boltzmann_order + 1 and options['compound'] == 'nonlinear')
    ϱ_bar = component.ϱ_bar
    w = component.w(a=a)
    w_eff = component.w_eff(a=a)
    ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
    if variable == 0:
        ϱ_ptr = ptr
        # δ → ϱ = ϱ_bar(1 + δ)
        δ_min = ထ
        for index in range(component.size):
            value = ϱ_ptr[index]
            if value < δ_min:
                δ_min = ℝ[ϱ_ptr[index]]
            with unswitch:
                if nongaussianity:
                    # Add non-Gaussian contribution: δ → δ + f_NL*δ²
                    value += nongaussianity*value**2
            ϱ_ptr[index] = ϱ_bar*(1 + value)
        δ_min = allreduce(δ_min, op=MPI.MIN)
        if δ_min < -1:
            masterwarn(
                f'The realised energy density field of {component.name} '
                f'has min(δ) = {δ_min:.4g} < -1'
            )
    elif variable == 1:
        # Note that the momentum grids are currently unaffected
        # by the non-Gaussianity.
        Jⁱ_ptr = ptr
        if compound:
            # uⁱ → Jⁱ = a⁴(ρ + c⁻²P)uⁱ
            #         = a**(1 - 3w_eff)(ϱ + c⁻²𝒫)uⁱ
            ϱ_ptr = component.ϱ.gridˣ if use_gridˣ else component.ϱ.grid
            𝒫_ptr = component.𝒫.gridˣ if use_gridˣ else component.𝒫.grid
            for index in range(component.size):
                Jⁱ_ptr[index] *= ℝ[a**(1 - 3*w_eff)]*(
                    ϱ_ptr[index] + ℝ[light_speed**(-2)]*𝒫_ptr[index]
                )
        else:
            # uⁱ → Jⁱ = a**4(ρ + c⁻²P)uⁱ
            #         = a**(1 - 3w_eff)(ϱ + c⁻²𝒫)uⁱ
            #         ≈ a**(1 - 3w_eff)ϱ_bar(1 + w)uⁱ
            for index in range(component.size):
                Jⁱ_ptr[index] *= ℝ[a**(1 - 3*w_eff)*ϱ_bar*(1 + w)]
    elif variable == 2 and multi_index == 'trace':
        # δP → 𝒫 = 𝒫_bar + a**(3(1 + w_eff))δP
        #        = c²wϱ_bar + a**(3(1 + w_eff))δP
        𝒫_ptr = ptr
        for index in range(component.size):
            𝒫_ptr[index] = ℝ[light_speed**2*w*ϱ_bar] + ℝ[a**(3*(1 + w_eff))]*𝒫_ptr[index]
    elif variable == 2:
        ςⁱⱼ_ptr = ptr
        if compound:
            # σⁱⱼ → ςⁱⱼ = (ϱ + c⁻²𝒫)σⁱⱼ
            ϱ_ptr = component.ϱ.gridˣ if use_gridˣ else component.ϱ.grid
            𝒫_ptr = component.𝒫.gridˣ if use_gridˣ else component.𝒫.grid
            for index in range(component.size):
               ςⁱⱼ_ptr[index] *= ϱ_ptr[index] + ℝ[light_speed**(-2)]*𝒫_ptr[index]
        else:
            # σⁱⱼ → ςⁱⱼ = (ϱ + c⁻²𝒫)σⁱⱼ
            #           ≈ ϱ_bar(1 + w)σⁱⱼ
            for index in range(component.size):
                ςⁱⱼ_ptr[index] *= ℝ[ϱ_bar*(1 + w)]
    else:
        abort(f'realize_fluid() got non-implemented variable = {variable}')

# Function for realising fluid variables of fluid components
# applying component specific approximations.
@cython.header(
    # Arguments
    component='Component',
    a='double',
    variable='int',
    multi_index=object,  # int, str or tuple
    use_gridˣ='bint',
    # Locals
    index='Py_ssize_t',
    w='double',
    𝒫_ptr='double*',
    ϱ_ptr='double*',
    returns='bint',
)
def realize_approximative(component, a, variable, multi_index, use_gridˣ=False):
    # The "P = wρ" approximation
    if (
        component.approximations['P=wρ']
        and component.representation == 'fluid'
        and variable == 2
        and multi_index == 'trace'
    ):
        # Set 𝒫 equal to the current ϱ times the current c²w
        w = component.w(a=a)
        ϱ_ptr = component.ϱ.gridˣ if use_gridˣ else component.ϱ.grid
        𝒫_ptr = component.𝒫.gridˣ if use_gridˣ else component.𝒫.grid
        for index in range(component.size):
            𝒫_ptr[index] = ϱ_ptr[index]*ℝ[light_speed**2*w]
        return True
    # No approximative realisation carried out
    return False

# Function for tabulating the amplitude of the realisation grid
# as a function of k2.
@cython.pheader(
    # Arguments
    gridsize='Py_ssize_t',
    component='Component',
    a='double',
    a_next='double',
    variable='int',
    multi_index=object,  # int, str or tuple
    factor='double',
    # Locals
    amplitudes='double[::1]',
    amplitudes_ptr='double*',
    cosmoresults=object,  # CosmoResults
    cosmoresults_δ=object,  # CosmoResults
    k2='Py_ssize_t',
    k2_max='Py_ssize_t',
    k_fundamental='double',
    k_magnitude='double',
    normalization='double',
    nyquist='Py_ssize_t',
    options=dict,
    transfer_spline='Spline',
    transfer_spline_δ='Spline',
    use_primordial='bint',
    weight=str,
    returns='double[::1]',
)
def get_amplitudes(gridsize, component, a, a_next=-1, variable=-1, multi_index=None, factor=1):
    """This function returns an array of tabulated amplitudes for use
    with realisations. Realisations from the primordial noise looks like
      ℱₓ⁻¹[T(a) ζ(k) K(k⃗) ℛ(k⃗)]
    with ℛ(k⃗) the primordial noise, T(k) = T(a, k) the transfer
    function, ζ(k) the primordial curvature perturbation and K(k⃗)
    containing any additional tensor structure. What is tabulated by
    this function is T(a, k) ζ(k) with the additional Fourier
    normalization of boxsize**(-1.5), as the grid is constructed from
    scratch in Fourier space.
    Realisations from an existing density field ϱ looks like
      ℱₓ⁻¹[T(k)/T_δϱ(k) K(k⃗) ℱₓ[δϱ(x⃗)]],
    and so in this case what is tabulated is T(k)/T_δϱ(k) with the
    additional Fourier normalization of gridsize**(-3), due to the
    forward FFT.
    """
    # Get transfer function spline. For realisations that should be
    # averaged over some time interval, we fist need to determine the
    # appropriate weighting. Note that we do not perform the averaging
    # for realisations that makes use of the non-linear structure,
    # as this structure itself is not time-averaged.
    if variable == -1:
        abort(f'get_amplitudes() called with variable = {variable}')
    options = component.realization_options
    use_primordial = (
        variable == 0
        or variable <= component.boltzmann_order
        or options['structure'] == 'primordial'
    )
    weight = None
    if use_primordial and a_next not in {-1, a}:
        if variable == 0:
            # Variable ϱ which enters the Poisson equation
            #   ∇²φ = 4πGa²ρ = 4πGa**(-3*w_eff - 1)ϱ
            weight = 'a**(-3*w_eff-1)'
        elif variable == 1:
            # Variable Jᵐ which enters the continuity equation
            #   ∂ₜϱ = -a**(3*w_eff - 2)∂ᵢJⁱ + ⋯
            weight = 'a**(3*w_eff-2)'
        elif variable == 2:
            # Variable 𝒫 or ςᵐₙ which enters the Euler equation
            #   ∂ₜJᵐ = -a**(-3*w_eff)(∂ᵐ𝒫 + ∂ⁿςᵐₙ) + ⋯
            weight = 'a**(-3*w_eff)'
        else:
            abort(f'Unknown variable "{variable}" passed to get_amplitudes()')
    transfer_spline, cosmoresults = compute_transfer(
        component, variable, gridsize, multi_index, a, a_next, options['gauge'],
        weight=weight, backscale=options['backscale']*(variable == 0),
    )
    # Get transfer function spline of δ in the case of
    # non-linear realisation.
    if not use_primordial:
        transfer_spline_δ, cosmoresults_δ = compute_transfer(
            component, 0, gridsize,
            a=a, gauge=options['gauge'],
        )
    # Fetch grid for storing amplitudes
    nyquist = gridsize//2
    k2_max = 3*(nyquist - 1)**2
    amplitudes = get_buffer(k2_max + 1, buffer_names['amplitudes'])
    amplitudes_ptr = cython.address(amplitudes[:])
    # Fourier normalization
    if use_primordial:
        normalization = boxsize**(-1.5)
    else:
        normalization = float(gridsize)**(-3)
    normalization *= factor
    # Tabulate amplitudes
    k_fundamental = ℝ[2*π/boxsize]
    amplitudes_ptr[0] = 0
    for k2 in range(1, k2_max + 1):
        k_magnitude = k_fundamental*sqrt(k2)
        with unswitch:
            if use_primordial:
                amplitudes_ptr[k2] = (
                    transfer_spline.eval(k_magnitude)
                    *get_primordial_curvature_perturbation(k_magnitude)
                    *normalization
                )
            else:
                amplitudes_ptr[k2] = (
                    transfer_spline.eval(k_magnitude)
                    /transfer_spline_δ.eval(k_magnitude)
                    *ℝ[normalization/component.ϱ_bar]
                )
    return amplitudes
# Names of buffers used in this module, specifically by the
# get_amplitudes() and displace_particles() function.
cython.declare(buffer_names='dict')
buffer_names = {
    'amplitudes'   : 0,
    'displacements': 1,
}

# Function for realising a single grid. Scalar, vector and rank-2 tensor
# realisations are supported.
@cython.pheader(
    # Arguments
    gridsize='Py_ssize_t',
    component='Component',
    a='double',
    amplitude_or_amplitudes=object,  # double or double[::1]
    variable='int',
    multi_index=object, # int, str or tuple
    lattice='Lattice',
    diff_dim='int',
    slab_structure='double[:, :, ::1]',
    nongaussianity='double',
    # Locals
    amplitude='double',
    amplitude_const='double',
    amplitudes='double[::1]',
    amplitudes_ptr='double*',
    cosθ='double',
    factor='double',
    im='double',
    index='Py_ssize_t',
    index0='int',
    index1='int',
    k2='Py_ssize_t',
    k_fundamental='double',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    kl='Py_ssize_t',
    km='Py_ssize_t',
    re='double',
    sinθ='double',
    slab='double[:, :, ::1]',
    slab_ptr='double*',
    slab_structure_ptr='double*',
    tensor_rank='int',
    θ='double',
    returns='double[:, :, ::1]',
)
def realize_grid(
    gridsize, component, a, amplitude_or_amplitudes, variable,
    multi_index=None, lattice=None, diff_dim=-1, slab_structure=None, nongaussianity=0,
):
    """Note that this function returns a slab in real space"""
    index0 = index1 = 0
    if variable == 0 or isinstance(multi_index, str):
        # We are realising either ϱ or 𝒫 (multi_index == 'trace')
        tensor_rank = 0
    else:
        if multi_index is None:
            abort(f'realize_grid() called with variable {variable} != 0 but no multi_index')
        multi_index = any2list(multi_index)
        tensor_rank = len(multi_index)
        index0 = multi_index[0%len(multi_index)]
        index1 = multi_index[1%len(multi_index)]
    if lattice is None:
        lattice = Lattice()
    if lattice.shift != (0, 0, 0):
        # The positions have been shifted by +shift, though -shift is
        # assumed by fourier_loop() when computing θ.
        lattice = Lattice(lattice, negate_shifts=True)
    # Fetch slab decomposed grid for storing
    # the complete realisation information.
    slab = get_fftw_slab(gridsize)
    slab_ptr = cython.address(slab[:, :, :])
    # Fetch slab decomposed grid for storing the underlying structure
    # (primordial noise or energy density).
    if slab_structure is None:
        slab_structure = get_slab_structure(gridsize, component, a, variable)
    slab_structure_ptr = cython.address(slab_structure[:, :, :])
    # Handle constant or varying amplitude
    amplitude_const = 0
    if isinstance(amplitude_or_amplitudes, (int, float, np.integer, np.floating)):
        amplitude_const = amplitude_or_amplitudes
    else:
        amplitudes = amplitude_or_amplitudes
        amplitudes_ptr = cython.address(amplitudes[:])
    # Populate realisation slab
    k_fundamental = ℝ[2*π/boxsize]
    for index, ki, kj, kk, factor, θ in fourier_loop(
        gridsize, skip_origin=True, interlace_lattice=lattice,
    ):
        k2 = ℤ[ℤ[ℤ[kj**2] + ki**2] + kk**2]
        with unswitch:
            if amplitude_const == 0:
                amplitude = amplitudes_ptr[k2]
            else:
                amplitude = amplitude_const
        re = slab_structure_ptr[index    ]
        im = slab_structure_ptr[index + 1]
        # Rotate the complex phase due to shift
        with unswitch:
            if lattice.shift != (0, 0, 0):
                cosθ = cos(θ)
                sinθ = sin(θ)
                re, im = (
                    re*cosθ - im*sinθ,
                    re*sinθ + im*cosθ,
                )
        # Handle the different tensor structures
        with unswitch:
            if tensor_rank == 0:
                # Scalar: K(k⃗) = 1
                pass
            elif tensor_rank == 1:
                # Vector: K(k⃗) = -ikⁱ/k²
                kl = (
                    ℤ[
                          ℤ[ℤ[-(index0 == 0)] & ki]
                        | ℤ[ℤ[-(index0 == 1)] & kj]
                    ]
                        | ℤ[ℤ[-(index0 == 2)] & kk]
                )
                amplitude *= (ℝ[-boxsize/(2*π)]*kl)/k2
                re, im = -im, re
            else:  # tensor_rank == 2
                # Rank-2 tensor: K(k⃗) = 3/2(δⁱⱼ/3 - kⁱkⱼ/k²)
                kl = (
                    ℤ[
                          ℤ[ℤ[-(index0 == 0)] & ki]
                        | ℤ[ℤ[-(index0 == 1)] & kj]
                    ]
                        | ℤ[ℤ[-(index0 == 2)] & kk]
                )
                km = (
                    ℤ[
                          ℤ[ℤ[-(index1 == 0)] & ki]
                        | ℤ[ℤ[-(index1 == 1)] & kj]
                    ]
                        | ℤ[ℤ[-(index1 == 2)] & kk]
                )
                amplitude *= ℝ[0.5*(index0 == index1)] - (1.5*kl*km)/k2
        # Possible extra differentiation (multiplication by ikⁱ)
        with unswitch:
            if diff_dim != -1:
                kl = (
                    ℤ[
                          ℤ[ℤ[-(diff_dim == 0)] & ki]
                        | ℤ[ℤ[-(diff_dim == 1)] & kj]
                    ]
                        | ℤ[ℤ[-(diff_dim == 2)] & kk]
                )
                amplitude *= k_fundamental*kl
                re, im = -im, re
        # Store results
        slab_ptr[index    ] = amplitude*re
        slab_ptr[index + 1] = amplitude*im
    # Nullify the origin and the Nyquist planes
    nullify_modes(slab, ['origin', 'nyquist'])
    # Fourier transform the slabs to coordinate space
    fft(slab, 'backward')
    # Imprint non-Gaussianity.
    # Note that this destroys the Gaussian part.
    if nongaussianity:
        for index in range(slab.shape[0]*slab.shape[1]*slab.shape[2]):
            slab_ptr[index] = nongaussianity*slab_ptr[index]**2
    return slab

# Function for fetching and populating a slab decomposed grid with the
# underlying structure for a realisation; either primordial noise or
# the density field of a component.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    component='Component',
    a='double',
    variable='int',
    use_gridˣ='bint',
    # Locals
    gridname='str',
    info=dict,
    name=str,
    reuse='bint',
    slab_structure='double[:, :, ::1]',
    use_primordial='bint',
    returns='double[:, :, ::1]',
)
def get_slab_structure(gridsize, component, a, variable, use_gridˣ=False):
    """The reusage of slabs is defined by the fourier_structure_caching
    parameter. If the given slab is not to be reused, the default slab
    is fetched. The slab is then populated with the underlying structure
    in accordance with the provided arguments. If the given slab is to
    be reused, a dedicated slab is fetched, which is not used for
    anything else in the program. The existence of such reusable slabs
    are recorded, so that they can be reused as is in future calls.
    """
    # Figure out what name to use for the structure slab
    use_primordial = (
        variable == 0
        or variable <= component.boltzmann_order
        or component.realization_options['structure'] == 'primordial'
    )
    name = None
    info = {'primordial': use_primordial}
    if use_primordial:
        if fourier_structure_caching.get('primordial'):
            name = 'slab_structure_primordial'
    else:
        if is_selected(component, fourier_structure_caching):
            name = f'slab_structure_nonlinear_{component.name}'
        info |= {
            'component': component.name,
            'a'        : a,
            'use_gridˣ': use_gridˣ,
        }
    # Can we reuse existing slab?
    reuse = False
    if name is not None:
        reuse = (slab_structure_infos.get((gridsize, name)) == info)
        # Record structure slab information for later calls
        slab_structure_infos[gridsize, name] = info
    # Fetch structure slab
    slab_structure = get_fftw_slab(gridsize, name)
    # Populate structure slab if it cannot be reused as is
    if not reuse:
        if use_primordial:
            # Populate slab_structure with primordial noise ℛ(k⃗)
            generate_primordial_noise(slab_structure)
        else:
            # Populate slab_structure with ℱₓ[ϱ(x⃗)]
            gridname = 'ϱ' + 'ˣ'*use_gridˣ
            masterprint(f'Extracting structure from {gridname} of {component.name} ...')
            slab_decompose(
                component.ϱ.gridˣ_mv if use_gridˣ else component.ϱ.grid_mv,
                slab_structure,
            )
            fft(slab_structure, 'forward')
            masterprint('done')
    # Return populated structure slab
    return slab_structure
# Dict used by the get_slab_structure() function to keep track of what
# slabs might be reused.
cython.declare(slab_structure_infos=dict)
slab_structure_infos = {}

# Function that populates a slab decomposed grid
# with primordial noise.
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    # Locals
    gridsize='Py_ssize_t',
    i_conj='Py_ssize_t',
    im='double',
    imprint='bint',
    imprint_conj='bint',
    index='Py_ssize_t',
    index_conj='Py_ssize_t',
    inside_slab='bint',
    j_conj='Py_ssize_t',
    j_global_conj='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    lower_x_zdc='bint',
    msg=list,
    prng_ampliudes='PseudoRandomNumberGenerator',
    prng_phases='PseudoRandomNumberGenerator',
    r='double',
    re='double',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_size_k='Py_ssize_t',
    slab_ptr='double*',
    θ='double',
    θ_str=str,
    returns='void',
)
def generate_primordial_noise(slab):
    """Given an already allocated slab, this function will populate it
    with Gaussian pseudo-random numbers, corresponding to primordial
    white noise. The slab grid is thought of as being in Fourier space,
    and so the random numbers are complex. We wish the variance of these
    complex numbers to equal unity, and so their real and imaginary
    parts should be drawn from a distribution with variance 1/√2,
    corresponding to
      re = gaussian(1/√2)
      im = gaussian(1/√2)
    As we further want to allow for fixing of the amplitude and shifting
    of the phase, we instead draw the numbers in polar form:
      r = rayleigh(1/√2]
      θ = uniform(-π, π)
      re = r*cos(θ)
      im = r*sin(θ)
    The 3D sequence of random numbers should be independent on the size
    of the grid, in the sense that increasing the grid size should
    amount to just populating the additional, outer "shell" with new
    random numbers, leaving the random numbers inside the inner cuboid
    the same. This has the effect that enlarging the grid leaves the
    large-scale structure invariant; one merely adds information at
    smaller scales. Additionally, the sequence of random numbers should
    be independent on the number of processes. To achieve all of this,
    all processes makes use of the same random streams (i.e. they all
    use the same seed for r and the same seed for θ, though the two
    seeds should be different). The random numbers are imprinted onto
    the slab in the order in which they are drawn, so the iteration must
    be one that starts at the origin and visits every mode
    k⃗ = (ki, kj, kk) in order according to max(|ki|, |kj|, |kk|).
    This iteration is implemented by a (Fourier) space-filling curve.
    Every process must traverse the entire grid, drawing every random
    number, though of course only imprint the drawn random numbers onto
    their local slab.
    The z DC plane (kk = 0) needs to satisfy the Hermitian symmetry of a
    Fourier transformed real field, here
        grid[+kj, +ki, kk = 0] = grid[-kj, -ki, kk = 0]*,
    where * means complex conjugation. When we visit a point
    ( ki < 0,  kj,     kk = 0), we imprint the conjugated value onto
    (-ki > 0, -kj,     kk = 0), and similarly when visiting
    ( ki = 0,  kj < 0, kk = 0). We refer to this ~half of th z DC plane
    as the 'lower x' part. We thus choose to copy the lower x part of
    the z DC plane onto the upper x part of the z DC plane.
    Note that neither the origin nor the Nyquist planes will be touched
    by this function.
    """
    slab_size_j, slab_size_i, slab_size_k = asarray(slab).shape
    gridsize = slab_size_i
    slab_ptr = cython.address(slab[:, :, :])
    # Progress message
    msg = ['Generating primordial']
    if not primordial_amplitude_fixed:
        msg.append(' Gaussian')
    msg.append(f' noise of grid size {gridsize}')
    if primordial_amplitude_fixed:
        msg.append(', fixed amplitude')
    if primordial_phase_shift != 0:
        if isclose(primordial_phase_shift, π):
            θ_str = 'π'
        else:
            θ_str = str(primordial_phase_shift)
        msg.append(f', phase shift {θ_str}')
    masterprint(''.join(msg), '...')
    # Initialize individual pseudo-random number generators for the
    # amplitudes and phases, using the same seeds on all processes.
    prng_ampliudes = PseudoRandomNumberGenerator(random_seeds['primordial amplitudes'])
    prng_phases    = PseudoRandomNumberGenerator(random_seeds['primordial phases'])
    # Burn the first random number for both generators, corresponding to
    # the origin of the grid. As the grid points are visited in the
    # order given by the Fourier space-filling curve, the sequence of
    # drawn random numbers are then perfectly copied onto the grid
    # in this same order.
    prng_ampliudes.rayleigh()
    prng_phases.uniform()
    # Traverse Fourier space from the inside out, drawing random
    # Gaussian numbers as we go and store them in the slabs.
    for index, ki, kj, kk, inside_slab in fourier_curve_loop(gridsize, skip_origin=True):
        # Draw random numbers
        r = 1
        with unswitch:
            if not primordial_amplitude_fixed:
                r = prng_ampliudes.rayleigh(1/sqrt(2))
        θ = prng_phases.uniform(-π, π)
        # Check whether the random numbers should be imprinted onto the
        # local slab, either at the current grid point (ki, kj, kk)
        # or its conjugate (-ki, -kj, -kk) in case of kk = 0.
        imprint = inside_slab
        imprint_conj = False
        if kk == 0:
            lower_x_zdc = (ki < 0) | ((ki == 0) & (kj < 0))
            imprint &= lower_x_zdc
            if lower_x_zdc:
                j_global_conj = -kj + (-(kj > 0) & gridsize)
                j_conj = j_global_conj - ℤ[slab_size_j*rank]
                imprint_conj = (0 <= j_conj < slab_size_j)
        if not imprint and not imprint_conj:
            continue
        # Finalize random noise
        with unswitch:
            if primordial_phase_shift:
                θ += primordial_phase_shift
        re = r*cos(θ)
        im = r*sin(θ)
        # Imprint onto grid point
        if imprint:
            slab_ptr[index    ] = re
            slab_ptr[index + 1] = im
        # Imprint onto conjugate grid pint
        if imprint_conj:
            i_conj = -ki + (-(ki > 0) & gridsize)
            index_conj = (j_conj*slab_size_i + i_conj)*slab_size_k  # k = 0
            slab_ptr[index_conj    ] = +re
            slab_ptr[index_conj + 1] = -im
    masterprint('done')

# Function for realising particle components
@cython.header(
    # Argumetns
    component='Component',
    a='double',
    # Locals
    amplitude='double',
    amplitudes='double[::1]',
    backscale='bint',
    cosmoresults=object,  # CosmoResults
    dim='int',
    dim0='int',
    dim1='int',
    do_2lpt='bint',
    factor='double',
    fft_factor='double',
    gridsize='Py_ssize_t',
    growth_fac_D='double',
    growth_fac_D2='double',
    growth_fac_f='double',
    growth_fac_f2='double',
    id_bgn='Py_ssize_t',
    index='Py_ssize_t',
    indexᵖ_bgn='Py_ssize_t',
    indexʳ='Py_ssize_t',
    lattice='Lattice',
    n_different_sized='Py_ssize_t',
    n_local='Py_ssize_t',
    n_particles='Py_ssize_t',
    nongaussianity='double',
    options=dict,
    particle_components=list,
    pos='double*',
    slab='double[:, :, ::1]',
    slab_2lpt='double[:, :, ::1]',
    slab_2lpt_ptr='double*',
    slab_nongaussian='double[:, :, ::1]',
    slab_ptr='double*',
    slab_xx='double[:, :, ::1]',
    slab_xx_ptr='double*',
    slab_yy='double[:, :, ::1]',
    slab_yy_arr=object,  # np.ndarray
    slab_yy_ptr='double*',
    slab_zz='double[:, :, ::1]',
    slab_zz_ptr='double*',
    variable='int',
    velocity_factor='double',
    returns='void',
)
def realize_particles(component, a):
    options = component.realization_options
    if component.representation != 'particles':
        abort(f'realize_particles() called with non-particle component {component.name}')
    if options['structure'] != 'primordial':
        abort('Can only realize particles using the primordial structure')
    # Resize particle data attributes
    if component.N%nprocs != 0:
        abort(
            f'Cannot perform realisation of {component.name} '
            f'with N = {component.N}, as N is not evenly divisible by {nprocs} processes'
        )
    component.N_local = component.N//nprocs
    component.resize(component.N_local)
    # Prepare lattice options
    if not component.preic_lattice:
        abort(
            f'Cannot initialize particle component {component.name} '
            f'with N = {component.N} on a lattice, as neither of '
            f'{{N, N/2, N/4}} is a cubic number'
        )
    if n_particles_realized['components_tally'] == 0:
        particle_components = [
            other_component for other_component in component.components_all
            if other_component.representation == 'particles' and other_component.mass == -1
        ]
        n_different_sized = (
            len({other_component.N for other_component in particle_components})
        )
        if n_different_sized == 0:
            masterwarn('Failed to detect number of particle components to be realized')
        elif n_different_sized != 1:
            masterwarn(
                'Multiple particle components with different number of particles are '
                'to be realized. This will lead to anisotropies in the initial conditions.'
            )
        n_particles_realized['components_total'] = len(particle_components)
    lattice = Lattice(component.preic_lattice)
    if component.preic_lattice == 'sc':
        gridsize = icbrt(component.N)
        if n_particles_realized['components_total'] == 2:
            lattice = Lattice('bcc', n_particles_realized['components_tally'])
        elif n_particles_realized['components_total'] == 4:
            lattice = Lattice('fcc', n_particles_realized['components_tally'])
        elif n_particles_realized['components_total'] != 1:
            masterwarn(
                f'{n_particles_realized["components_total"]} ∉ {{1, 2, 4}} particle components '
                f'are to be initialized, with at least one of them being initialized on a '
                f'simple cubic lattice. This leads to anisotropies in the initial conditions.'
            )
    elif component.preic_lattice == 'bcc':
        gridsize = icbrt(component.N//2)
        if n_particles_realized['components_total'] != 1:
            masterwarn(
                f'{n_particles_realized["components_total"]} > 1 particle components are to be '
                f'initialized, with at least one of them being initialized on a body-centered '
                f'cubic lattice. This leads to anisotropies in the initial conditions.'
            )
    elif component.preic_lattice == 'fcc':
        gridsize = icbrt(component.N//4)
        if n_particles_realized['components_total'] != 1:
            masterwarn(
                f'{n_particles_realized["components_total"]} > 1 particle components are to be '
                f'initialized, with at least one of them being initialized on a face-centered '
                f'cubic lattice. This leads to anisotropies in the initial conditions.'
            )
    else:
        abort(f'Unknown lattice {component.preic_lattice}')
    # Determine and set the particle mass if still unset
    if component.mass == -1:
        # For species with varying mass, this is the mass at a = 1
        component.mass = component.ϱ_bar*boxsize**3/component.N
    # Progress print
    backscale = options['backscale']
    masterprint(
        f'Realising {{}}{gridsize}³ particles of {component.name} {{}}...'
        .format(f'{len(lattice)}×'*(len(lattice) > 1), 'using back-scaling '*backscale)
    )
    # Get growth factors if needed
    nongaussianity = options['nongaussianity']
    do_2lpt = (options['lpt'] > 1)
    if backscale or nongaussianity or do_2lpt:
        cosmoresults = compute_cosmo(class_call_reason='in order to get growth factor')
        growth_fac_D  = cosmoresults.growth_fac_D (a)
        growth_fac_f  = cosmoresults.growth_fac_f (a)
        growth_fac_D2 = cosmoresults.growth_fac_D2(a)
        growth_fac_f2 = cosmoresults.growth_fac_f2(a)
    # Realise particles, one lattice at a time
    fft_factor = float(gridsize)**(-3)
    n_particles = gridsize**3
    indexᵖ_bgn = 0
    id_bgn = n_particles_realized['particles_tally']
    for lattice in lattice:
        # Initialize particles on the lattice
        masterprint(
            'Initializing {}particles at lattice points ...'
            .format(
                '{} set of '.format(
                    ['first', 'second', 'third', 'fourth'][lattice.index]
                )*(len(lattice) > 1)
            )
        )
        n_local = preinitialize_particles(component, n_particles, indexᵖ_bgn, id_bgn, lattice)
        masterprint('done')
        # Realise positions and momenta (1LPT)
        for variable in range(2):
            if variable == 0:
                if backscale:
                    masterprint('Displacing particle positions and boosting momenta ...')
                else:
                    masterprint('Displacing particle positions ...')
            elif variable == 1:
                masterprint('Boosting particle momenta ...')
            # Fetch δ or θ amplitudes. Note that the displacement field
            # has a sign difference relative to direct realisation of δ.
            factor = 2*variable - 1
            amplitudes = get_amplitudes(gridsize, component, a, variable=variable, factor=factor)
            # Displace positions or boost velocities
            # using the Zel'dovich approximation.
            for dim in range(3):
                slab = realize_grid(gridsize, component, a, amplitudes, 1, dim, lattice)
                displace_particles(component, slab, a, n_particles, indexᵖ_bgn, variable, dim)
                if backscale:
                    # Assign momenta using the displacement field
                    # if using back-scaling.
                    velocity_factor = a*hubble(a)*growth_fac_f
                    displace_particles(
                        component, slab, a, n_particles, indexᵖ_bgn, 1, dim, velocity_factor,
                    )
            masterprint('done')
            # Done with both positions and momenta if using back-scaling
            if backscale:
                break
        # Add non-Gaussian contributions to positions and momenta
        if nongaussianity:
            masterprint(
                'Displacing particle positions and boosting momenta '
                '(local non-Gaussianity) ...'
            )
            # Fetch δ amplitudes
            factor = -1
            amplitudes = get_amplitudes(gridsize, component, a, variable=0, factor=factor)
            # Create purely non-Gaussian δ grid. The sign aplied to the
            # Gaussian displacement field is applied here as well.
            slab = realize_grid(
                gridsize, component, a, amplitudes, 0,
                lattice=lattice, nongaussianity=factor*nongaussianity,
            )
            # Transform to Fourier space
            fft(slab, 'forward')
            slab_nongaussian = asarray(slab).copy()
            # Displace positions and boost velocities
            amplitude = fft_factor
            velocity_factor = a*hubble(a)*growth_fac_f
            for dim in range(3):
                slab = realize_grid(
                    gridsize, component, a, amplitude, 1, dim, lattice,
                    slab_structure=slab_nongaussian,
                )
                displace_particles(
                    component, slab, a, n_particles, indexᵖ_bgn, 0, dim,
                )
                displace_particles(
                    component, slab, a, n_particles, indexᵖ_bgn, 1, dim, velocity_factor,
                )
            masterprint('done')
        # Add second-order (2LPT) contributions to positions and momenta
        if do_2lpt:
            masterprint('Displacing particle positions and boosting momenta (2LPT) ...')
            # Fetch δ amplitudes
            amplitudes = get_amplitudes(gridsize, component, a, variable=0, factor=-1)
            # Create ψ_xx (= ∂ₓψₓ), ψ_yy, ψ_zz
            slab_xx = asarray(
                realize_grid(gridsize, component, a, amplitudes, 1, 0, lattice, 0)
            ).copy()
            slab_yy = slab_yy_arr = asarray(
                realize_grid(gridsize, component, a, amplitudes, 1, 1, lattice, 1)
            ).copy()
            slab_zz = (
                realize_grid(gridsize, component, a, amplitudes, 1, 2, lattice, 2)
            )
            slab_xx_ptr = cython.address(slab_xx[:, :, :])
            slab_yy_ptr = cython.address(slab_yy[:, :, :])
            slab_zz_ptr = cython.address(slab_zz[:, :, :])
            # Create 2LPT source
            #   + ψ_xx*ψ_yy + ψ_yy*ψ_zz + ψ_zz*ψ_xx
            #   - ψ_xy**2   - ψ_yz**2   - ψ_zx**2
            slab_2lpt_ptr = slab_xx_ptr
            for index in range(slab_xx.shape[0]*slab_xx.shape[1]*slab_xx.shape[2]):
                slab_2lpt_ptr[index] *= slab_yy_ptr[index] + slab_zz_ptr[index]
                slab_2lpt_ptr[index] += slab_yy_ptr[index] * slab_zz_ptr[index]
            slab_yy_arr.resize(0, refcheck=False)
            for dim0 in range(3):
                dim1 = (dim0 + 1)%3
                slab = realize_grid(gridsize, component, a, amplitudes, 1, dim0, lattice, dim1)
                slab_ptr = cython.address(slab[:, :, :])
                for index in range(slab.shape[0]*slab.shape[1]*slab.shape[2]):
                    with unswitch(1):
                        if dim0 < 2:
                            slab_2lpt_ptr[index] -= slab_ptr[index]**2
                        else:
                            slab_ptr[index] = slab_2lpt_ptr[index] - slab_ptr[index]**2
            # Transform the completed 2LPT source to Fourier space
            fft(slab, 'forward')
            slab_2lpt = asarray(slab).copy()
            slab_2lpt_ptr = cython.address(slab_2lpt[:, :, :])
            # Displace positions and boost velocities
            amplitude = fft_factor*growth_fac_D2/growth_fac_D**2
            velocity_factor = a*hubble(a)*growth_fac_f2
            for dim in range(3):
                slab = realize_grid(
                    gridsize, component, a, amplitude, 1, dim, lattice,
                    slab_structure=slab_2lpt,
                )
                displace_particles(
                    component, slab, a, n_particles, indexᵖ_bgn, 0, dim,
                )
                displace_particles(
                    component, slab, a, n_particles, indexᵖ_bgn, 1, dim, velocity_factor,
                )
            masterprint('done')
        # Prepare for next lattice
        id_bgn += n_particles
        indexᵖ_bgn += n_local
    # Done realising particles
    n_particles_realized['particles_tally'] = id_bgn
    n_particles_realized['components_tally'] += 1
    # Ensure toroidal boundaries and exchange
    # particles among the processes.
    pos = component.pos
    for indexʳ in range(3*component.N_local):
        pos[indexʳ] = mod(pos[indexʳ], boxsize)
    exchange(component)
    masterprint('done')
# Record updated by the realize_particles() function
cython.declare(n_particles_realized=dict)
n_particles_realized = {
    'components_tally': 0,
    'components_total': 0,
    'particles_tally': 0
}

# Function for pre-initialising particles, meaning placing them at
# lattice points, zeroing momenta and assigning IDs.
@cython.pheader(
    # Arguments
    component='Component',
    n_particles='Py_ssize_t',
    indexᵖ_bgn='Py_ssize_t',
    id_bgn='Py_ssize_t',
    lattice='Lattice',
    # Locals
    domain_bgn_i='Py_ssize_t',
    domain_bgn_j='Py_ssize_t',
    domain_bgn_k='Py_ssize_t',
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    ids='Py_ssize_t*',
    indexᵖ='Py_ssize_t',
    indexʳ='Py_ssize_t',
    indexˣ='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    mom='double*',
    n_local='Py_ssize_t',
    particle_id='Py_ssize_t',
    posxˣ='double*',
    posyˣ='double*',
    poszˣ='double*',
    shape=tuple,
    x='double',
    y='double',
    z='double',
    returns='Py_ssize_t',
)
def preinitialize_particles(component, n_particles=-1, indexᵖ_bgn=0, id_bgn=0, lattice=None):
    """This function will pre-initialise a passed component, meaning:
    - Set positions to lattice points.
    - Nullify momenta.
    - Assign IDs (according to lattice points).
    The lattice is a simple cubic (sc) lattice, with a total of
    n_particles = n**3 (with n ∈ ℕ) points, distributed among
    the processes. Which particles from the component to pre-initialise
    can be specified by setting th process-local index indexᵖ_bgn of the
    first particle. If the component makes use of IDs, the first
    (global, not process-local() ID can be specified as id_bgn.
    Setting shift to a 3-tuple like (½, ½, ½) displaces the lattice by
    this amount, in grid units.
    The particle data should be allocated on the component prior to
    calling this function.
    The return value is the local number of particles that
    have been pre-initialized.
    """
    if component.representation != 'particles':
        return 0
    if n_particles == -1:
        n_particles = component.N
    gridsize = icbrt(n_particles)
    if gridsize**3 != n_particles:
        abort(
            f'preinitialize_particles() called with non-cubic '
            f'number of particles {n_particles}'
        )
    if lattice is None:
        lattice = Lattice()
    # Get shape of domain (local part of lattice)
    shape = tuple(asarray(get_gridshape_local(gridsize)) - 2*nghosts)
    n_local = np.prod(shape)
    if component.N_local < indexᵖ_bgn + n_local:
        abort('Component passed to preinitialize_particles() is too small')
    domain_bgn_i = domain_layout_local_indices[0]*ℤ[shape[0]]
    domain_bgn_j = domain_layout_local_indices[1]*ℤ[shape[1]]
    domain_bgn_k = domain_layout_local_indices[2]*ℤ[shape[2]]
    # Position the particles at the lattice points (at the centre of the
    # lattice cells when running in cell centered mode), shifted in
    # accordance with the passed lattice.
    # Also assign particle IDs.
    posxˣ = component.posxˣ
    posyˣ = component.posyˣ
    poszˣ = component.poszˣ
    ids   = component.ids
    indexᵖ = indexᵖ_bgn
    indexˣ = 3*indexᵖ
    for i in range(ℤ[shape[0]]):
        x = (
            ℝ[domain_bgn_i + 0.5*cell_centered + lattice.shift[0]] + i
        )*ℝ[boxsize/gridsize]
        for j in range(ℤ[shape[1]]):
            y = (
                ℝ[domain_bgn_j + 0.5*cell_centered + lattice.shift[1]] + j
            )*ℝ[boxsize/gridsize]
            for k in range(ℤ[shape[2]]):
                z = (
                    ℝ[domain_bgn_k + 0.5*cell_centered + lattice.shift[2]] + k
                )*ℝ[boxsize/gridsize]
                posxˣ[indexˣ] = x
                posyˣ[indexˣ] = y
                poszˣ[indexˣ] = z
                indexˣ += 3
                # Set particle ID
                with unswitch:
                    if component.use_ids:
                        particle_id = (
                            + ℤ[
                                + id_bgn
                                + domain_bgn_i*gridsize**2
                                + domain_bgn_j*gridsize
                                + domain_bgn_k
                            ]
                            + ℤ[ℤ[i*ℤ[gridsize**2]] + j*gridsize]
                            + k
                        )
                        ids[indexᵖ] = particle_id
                        indexᵖ += 1
    # Nullify momenta
    mom = component.mom
    for indexʳ in range(3*indexᵖ_bgn, 3*(indexᵖ_bgn + n_local)):
        mom[indexʳ] = 0
    # Return the number of particles that have been pre-initialized
    # on this process.
    return n_local

# Function for applying a displacement field to particles;
# either displacing their positions or boosting their velocities.
@cython.header(
    # Arguments
    component='Component',
    slab='double[:, :, ::1]',
    a='double',
    n_particles='Py_ssize_t',
    indexᵖ_bgn='Py_ssize_t',
    variable='int',
    dim='int',
    factor='double',
    # Locals
    data='double*',
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    index='Py_ssize_t',
    indexʳ='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    mass='double',
    ψⁱ='double[:, :, ::1]',
    ψⁱ_ptr='double*',
    returns='void',
)
def displace_particles(component, slab, a, n_particles, indexᵖ_bgn, variable, dim, factor=1):
    if component.representation != 'particles':
        abort(f'displace_particles() called with non-particle component {component.name}')
    # Domain-decompose realised real-space grid.
    # This is either the displacement field ψⁱ or the velocity field uⁱ.
    ψⁱ = domain_decompose(slab, buffer_names['displacements'])
    ψⁱ_ptr = cython.address(ψⁱ[:, :, :])
    # Grab particle data array in accordance with the passed variable.
    # Also adapt the factor by which the grid values should
    # be multiplied before added to the particle data.
    if variable == 0:
        # Positions, Δxⁱ = ψⁱ
        data = component.pos
    elif variable == 1:
        # Momenta; momⁱ = a*m*uⁱ.
        # The current mass is the set mass (always defined a = 1),
        # scaled according to w_eff(a).
        data = component.mom
        mass = a**(-3*component.w_eff(a=a))*component.mass
        factor *= a*mass
    else:
        abort(f'displace_particles() got variable = {variable} ∉ {{0, 1}}')
    # Displace particle positions or boost particle velocities
    indexʳ = dim + 3*indexᵖ_bgn
    gridsize = slab.shape[1]
    for index, i, j, k in domain_loop(gridsize, skip_ghosts=True):
        with unswitch:
            if factor == 1:
                data[indexʳ] += ψⁱ_ptr[index]
            else:
                data[indexʳ] += factor*ψⁱ_ptr[index]
        indexʳ += 3



# Get local domain information
domain_info = get_domain_info()
cython.declare(
    domain_subdivisions='int[::1]',
    domain_layout_local_indices='int[::1]',
)
domain_subdivisions         = domain_info.subdivisions
domain_layout_local_indices = domain_info.layout_local_indices
