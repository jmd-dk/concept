# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2024 Jeppe Mosgaard Dakin.
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
    'from mesh import              '
    '    Lattice,                  '
    '    domain_decompose,         '
    '    domain_loop,              '
    '    fft,                      '
    '    fourier_curve_loop,       '
    '    fourier_curve_slice_loop, '
    '    fourier_diff,             '
    '    fourier_loop,             '
    '    free_fftw_slab,           '
    '    get_fftw_slab,            '
    '    get_gridshape_local,      '
    '    laplacian_inverse,        '
    '    resize_grid,              '
    '    nullify_modes,            '
    '    slab_decompose,           '
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
        seed=object,  # None or Python int or numpy.random.SeedSequence
        stream=str,
        cache_size='Py_ssize_t',
        salt='bint',
    )
    def __init__(self, seed=None, stream=random_generator, cache_size=2**12, salt=True):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the RandomNumberGenerator type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public object seed  # numpy.random.SeedSequence
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
        if salt and isinstance(seed, int):
            # Sprincle in magic number to avoid issues
            # with seed containing many zero bits.
            seed += int(π*1e+8)
            # Hand-picked offset leading to only small effects
            # of cosmic variance when using the default seeds
            # for the primordial noise.
            lucky_seed_offset = 137
            seed += lucky_seed_offset
        if not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(seed)
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

    # Method for spawning a child instance,
    # inheriting the seed but with an additional key mixed in.
    @cython.header(
        # Arguments
        spawn_key=object,  # int or tuple of ints
        # Locals
        child='PseudoRandomNumberGenerator',
        seed=object,  # numpy.random.SeedSequence
        returns='PseudoRandomNumberGenerator',
    )
    def spawn(self, spawn_key=0):
        spawn_key = tuple(any2list(spawn_key))
        seed = np.random.SeedSequence(
            self.seed.entropy,
            spawn_key=(self.seed.spawn_key + spawn_key),
        )
        child = type(self)(seed, self.stream, self.cache_size)
        return child

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
    slab = realize_grid(gridsize, component, a, amplitudes, variable, multi_index=multi_index)
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
        for index in range(component.size):
            value = ϱ_ptr[index]
            with unswitch:
                if nongaussianity:
                    # Add non-Gaussian contribution: δ → δ + f_NL*δ²
                    value += nongaussianity*value**2
            ϱ_ptr[index] = ϱ_bar*(1 + value)
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
    amplitudes = empty(k2_max + 1, dtype=C2np['double'])
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

# Function for realising a single grid. Scalar, vector and rank-2 tensor
# realisations are supported.
@cython.pheader(
    # Arguments
    gridsize='Py_ssize_t',
    component='Component',
    a='double',
    amplitudes='double[::1]',
    variable='int',
    buffer_or_buffer_name=object,  # double[:, :, ::1] or int or str
    multi_index=object, # int, str or tuple
    lattice='Lattice',
    slab_structure='double[:, :, ::1]',
    nongaussianity='double',
    output_space=str,
    # Locals
    amplitude='double',
    amplitudes_ptr='double*',
    cosθ='double',
    factor='double',
    im='double',
    index='Py_ssize_t',
    index0='int',
    index1='int',
    interlace_lattice='Lattice',
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
    gridsize, component, a, amplitudes, variable,
    buffer_or_buffer_name=None, multi_index=None, lattice=None,
    slab_structure=None, nongaussianity=0, output_space='real',
):
    output_space = output_space.lower()
    if output_space not in ('real', 'fourier'):
        abort(f'realize_grid() got output_space = "{output_space}" ∉ {{"real", "Fourier"}}')
    amplitudes_ptr = cython.address(amplitudes[:])
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
    if buffer_or_buffer_name is None or isinstance(buffer_or_buffer_name, (int, np.integer, str)):
        slab = get_fftw_slab(gridsize, buffer_or_buffer_name)
    else:
        slab = buffer_or_buffer_name
    slab_ptr = cython.address(slab[:, :, :])
    # Fetch slab decomposed grid for storing the underlying structure
    # (primordial noise or energy density).
    if slab_structure is None:
        slab_structure = get_slab_structure(gridsize, component, a, variable)
    slab_structure_ptr = cython.address(slab_structure[:, :, :])
    # Populate realisation slab
    k_fundamental = ℝ[2*π/boxsize]
    interlace_lattice = lattice
    for index, ki, kj, kk, factor, θ in fourier_loop(
        gridsize, skip_origin=True, interlace_lattice=interlace_lattice,
    ):
        k2 = ℤ[ℤ[ℤ[kj**2] + ki**2] + kk**2]
        amplitude = amplitudes_ptr[k2]
        re = slab_structure_ptr[index    ]
        im = slab_structure_ptr[index + 1]
        # Rotate the complex phase due to shift
        with unswitch:
            if interlace_lattice.shift != (0, 0, 0):
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
        # Store results
        slab_ptr[index    ] = amplitude*re
        slab_ptr[index + 1] = amplitude*im
    # Nullify the origin and the Nyquist planes
    nullify_modes(slab, ['origin', 'nyquist'])
    # Imprint non-Gaussianity if requested
    if nongaussianity:
        # Fourier transform the slabs to coordinate space
        fft(slab, 'backward')
        # Imprint non-Gaussianity
        for index in range(slab.shape[0]*slab.shape[1]*slab.shape[2]):
            slab_ptr[index] += nongaussianity*slab_ptr[index]**2
        # Return slabs in real or Fourier space
        if output_space == 'real':
            return slab
        fft(slab, 'forward', apply_forward_normalization=True)
        return slab
    # Return slabs in real or Fourier space
    if output_space == 'fourier':
        return slab
    # Fourier transform the slabs to coordinate space
    fft(slab, 'backward')
    return slab

# Function for fetching and populating a slab decomposed grid with the
# underlying structure for a realisation; either primordial noise or
# the density field of a component.
@cython.pheader(
    # Arguments
    gridsize='Py_ssize_t',
    component='Component',
    a='double',
    variable='int',
    use_gridˣ='bint',
    buffer_or_buffer_name=object,  # double[:, :, ::1] or int or str
    # Locals
    gridname='str',
    info=dict,
    name=str,
    options=dict,
    reuse='bint',
    slab_structure='double[:, :, ::1]',
    use_primordial='bint',
    returns='double[:, :, ::1]',
)
def get_slab_structure(
    gridsize, component, a, variable,
    use_gridˣ=False, buffer_or_buffer_name=None,
):
    """The reusage of slabs is defined by the fourier_structure_caching
    parameter. If the given slab is not to be reused, the default slab
    is fetched. The slab is then populated with the underlying structure
    in accordance with the provided arguments. If the given slab is to
    be reused, a dedicated slab is fetched, which is not used for
    anything else in the program. The existence of such reusable slabs
    are recorded, so that they can be reused as is in future calls.
    If a buffer or buffer name is passed, the reuse system is not used.
    """
    options = component.realization_options
    use_primordial = (
        variable == 0
        or variable <= component.boltzmann_order
        or options['structure'] == 'primordial'
    )
    reuse = False
    if buffer_or_buffer_name is None:
        # Figure out what name to use for the structure slab
        name = None
        info = {'primordial': use_primordial}
        if use_primordial:
            # Do not use a dedicated grid in memory for the primordial noise
            # if requested at a = a_begin (IC generation) or a = 1
            # (used with corrected power spectra), saving on memory.
            if fourier_structure_caching.get('primordial') and a not in {a_begin, 1}:
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
        if name is not None:
            reuse = (slab_structure_infos.get((gridsize, name)) == info)
            # Record structure slab information for later calls
            slab_structure_infos[gridsize, name] = info
        # Fetch structure slab
        slab_structure = get_fftw_slab(gridsize, name)
    elif isinstance(buffer_or_buffer_name, (int, np.integer, str)):
        slab_structure = get_fftw_slab(gridsize, buffer_or_buffer_name)
    else:
        slab_structure = buffer_or_buffer_name
    # Populate structure slab if it cannot be reused as is
    if not reuse:
        if use_primordial:
            # Populate slab_structure with primordial noise ℛ(k⃗)
            generate_primordial_noise(
                slab_structure,
                options['fixedamplitude'],
                options['phaseshift'],
            )
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
    fixed_amplitude='bint',
    phase_shift='double',
    # Locals
    gridsize='Py_ssize_t',
    i_conj='Py_ssize_t',
    im='double',
    im_conj='double',
    imprint='bint',
    imprint_conj='bint',
    index='Py_ssize_t',
    index_conj='Py_ssize_t',
    inside_slab='bint',
    j='Py_ssize_t',
    j_conj='Py_ssize_t',
    j_global='Py_ssize_t',
    j_global_conj='Py_ssize_t',
    ki='Py_ssize_t',
    ki_conj='Py_ssize_t',
    kj='Py_ssize_t',
    kj_conj='Py_ssize_t',
    kk='Py_ssize_t',
    lower_x_zdc='bint',
    lower_x_zdc_conj='bint',
    msg=list,
    nyquist='Py_ssize_t',
    prng_ampliudes='PseudoRandomNumberGenerator',
    prng_ampliudes_common='PseudoRandomNumberGenerator',
    prng_phases='PseudoRandomNumberGenerator',
    prng_phases_common='PseudoRandomNumberGenerator',
    r='double',
    r_conj='double',
    re='double',
    re_conj='double',
    slab_ptr='double*',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_size_k='Py_ssize_t',
    spawn_key_offset='Py_ssize_t',
    θ='double',
    θ_conj='double',
    θ_str=str,
    returns='void',
)
def generate_primordial_noise(slab, fixed_amplitude=False, phase_shift=0):
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
    be independent on the number of processes. Two different schemes for
    imprinting the primordial noise while satisfying all of the above
    are implemented:
    - The 'simple' scheme: All processes makes use of the same random
      streams (i.e. they all use the same seed for r and the same seed
      for θ, though the two seeds should be different). The random
      numbers are imprinted onto the slab in the order in which they are
      drawn, so the iteration must be one that starts at the origin and
      visits every mode k⃗ = (ki, kj, kk) in order according to
      max(|ki|, |kj|, |kk|). This iteration is implemented by a
      (Fourier) space-filling curve. Every process traverses the
      entire grid, drawing every random number, though of course only
      imprint the drawn random numbers which fall into their local slab.
      - The z DC plane (kk = 0) needs to satisfy the Hermitian symmetry
        of a Fourier transformed real field,
        grid[+kj, +ki, kk = 0] = grid[-kj, -ki, kk = 0]*,
        where * means complex conjugation. When we visit a point
        ( ki < 0,  kj,     kk = 0), we imprint the conjugated value onto
        (-ki > 0, -kj,    -kk = 0), and similarly when visiting
        ( ki = 0,  kj < 0, kk = 0). We refer to this ~half of the z DC
        plane as the 'lower x' part. We thus choose to copy the lower x
        part of the z DC plane onto the upper x part of the z DC plane.
    - The 'distributed' scheme: Each process only visits the grid points
      belonging to its local slab. The iteration is done in j-slices,
      with each slab being made up of a number of such slices, each of
      thickness 1. Within each slice, the grid points are visited in the
      same order as they would be visited in the simple scheme. Each
      slice now has its own dedicated streams for r and θ, which are
      spawned from the common r and θ streams, depending on the value of
      kj for the slice.
      - The Hermitian symmetry is implemented by simultaneously drawing
        the random numbers for the slice at kj as well as for the slice
        at -kj. Note that the points in the slice at -kj are visited in
        the same order as those in the slice at kj (not the reflected
        order). As in the simple strategy, we choose to copy the lower x
        part of the z DC plane onto the upper x part of the z DC plane.
        Note that the value of any point in the slice at kj residing
        within the upper x part should be obtained exactly from a point
        in the slice at -kj. Whenever such a point is hit in the -kj
        slice, the value is conjugated and imprinted onto the reflected
        point in the slice at kj.
    Note that the two schemes provide different realisations.
    The origin will be nullified but the Nyquist planes will be left
    untouched (these should be nullified beforehand or elsewhere).
    Both schemes visit the grid points in order of the Fourier
    space-filling curve, from the origin outwards. This enables us to
    simply use the random numbers in the order they are drawn, but at
    the cost of writing to the grid points in an order that is not
    contiguous (except for small clusters of contiguous points).
    A different technique would be to loop contiguously over the slab,
    while jumping around in the random streams. While many pseudo-random
    number generators do allow for such jumps, this cannot be used in a
    consistent fashion here, as the generation of e.g. Rayleigh
    distributed random numbers involves rejection, leading to the
    generation of one Rayleigh number really jumping the state of the
    generator two (or more) numbers ahead, which cannot be predicted
    without actually drawning the number.
    """
    slab_size_j, slab_size_i, slab_size_k = asarray(slab).shape
    gridsize = slab_size_i
    nyquist = gridsize//2
    slab_ptr = cython.address(slab[:, :, :])
    # Progress message
    msg = ['Generating primordial']
    if not fixed_amplitude:
        msg.append(' Gaussian')
    msg.append(f' noise of grid size {gridsize}')
    if fixed_amplitude:
        msg.append(', fixed amplitude')
    if phase_shift != 0:
        if isclose(phase_shift, π):
            θ_str = 'π'
        else:
            θ_str = str(phase_shift)
        msg.append(f', phase shift {θ_str}')
    masterprint(''.join(msg), '...')
    # Initialize individual pseudo-random number generators for the
    # amplitudes and phases, using the same seeds on all processes.
    prng_ampliudes_common = PseudoRandomNumberGenerator(random_seeds['primordial amplitudes'])
    prng_phases_common    = PseudoRandomNumberGenerator(random_seeds['primordial phases'])
    # Draw random numbers and imprint as noise.
    # Two possible schemes are implemented for this.
    if primordial_noise_imprinting == 'simple':
        # All processes loop over the entire Fourier space,
        # drawing random numbers at each grid point. The random numbers
        # are only imprinted onto local grid points, of course.
        # When a grid point is visited which should act as the conjugate
        # source of noise for another grid point, the conjugated noise
        # is imprinted onto this other grid point.
        for index, ki, kj, kk, inside_slab in fourier_curve_loop(gridsize, skip_origin=False):
            # Draw random numbers
            r = 1
            with unswitch:
                if not fixed_amplitude:
                    r = prng_ampliudes_common.rayleigh(1/sqrt(2))
            θ = prng_phases_common.uniform(-π, π)
            # Check whether the random numbers should be imprinted onto
            # the local slab, either at the current grid point
            # (ki, kj, kk) or its conjugate (-ki, -kj, -kk)
            # in case of kk = 0.
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
                if phase_shift:
                    θ += phase_shift
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
    elif primordial_noise_imprinting == 'distributed':
        # Each process iterates over its own slab only. This is done one
        # j-slice at a time, a slice being a slab of thickness 1.
        # When a point in the conjugate slice which should be copied
        # onto the primary slice is visited, the conjugated noise is
        # imprinted onto the reflected point in the main slice, which
        # generally is different from the current point (ki, kj, kk).
        spawn_key_offset = 2**32  # negative seeds not allowed
        if nyquist > spawn_key_offset + 1:
            # We do not expect to ever hit grids this large
            masterwarn(
                f'Primordial noise is to be imprinted on a grid of gridsize {gridsize} '
                f'using the {primordial_noise_imprinting} scheme. For grids this large, '
                f'the hard-coded spawn_key_offset of {spawn_key_offset} is too small.'
            )
        for j in range(slab_size_j):
            j_global = ℤ[slab_size_j*rank] + j
            kj = j_global - (-(j_global >= nyquist) & gridsize)
            # Skip Nyquist plane
            if kj == ℤ[-nyquist]:
                continue
            # Spawn off child pseudo-random number generators
            prng_ampliudes      = prng_ampliudes_common.spawn(spawn_key_offset + kj)
            prng_ampliudes_conj = prng_ampliudes_common.spawn(spawn_key_offset - kj)
            prng_phases         = prng_phases_common   .spawn(spawn_key_offset + kj)
            prng_phases_conj    = prng_phases_common   .spawn(spawn_key_offset - kj)
            # Traverse (ki, kk) Fourier space slice at current kj.
            # We include the origin, just to keep the order in which the
            # random numbers are drawn consistent across processes.
            kj_conj = -kj
            for index, ki, kk in fourier_curve_slice_loop(gridsize, j, skip_origin=False):
                ki_conj = +ki
                # Draw random numbers
                r = r_conj = 1
                with unswitch:
                    if not fixed_amplitude:
                        r      = prng_ampliudes     .rayleigh(1/sqrt(2))
                        r_conj = prng_ampliudes_conj.rayleigh(1/sqrt(2))
                θ      = prng_phases     .uniform(-π, π)
                θ_conj = prng_phases_conj.uniform(-π, π)
                # Check whether this grid point (ki, kj, kk) should be
                # set from the conjugate of grid point (-ki, -kj, -kk)
                # (will be done whenever we visit this point), in which
                # case imprint will be False. Also check whether the
                # noise from the point (ki_conj, kj_conj, kk_conj)
                # within the conjugate slice should be conjugated and
                # imprinted onto its reflection point in the main slice,
                # in which case imprint_conj will be True.
                imprint = True
                imprint_conj = False
                if kk == 0:
                    lower_x_zdc      = (ki      < 0) | ((ki      == 0) & (kj      < 0))
                    lower_x_zdc_conj = (ki_conj > 0) | ((ki_conj == 0) & (kj_conj > 0))
                    imprint      =     lower_x_zdc
                    imprint_conj = not lower_x_zdc_conj
                # Handle direct imprinting
                if imprint:
                    # Finalize random noise
                    with unswitch:
                        if phase_shift:
                            θ += phase_shift
                    re = r*cos(θ)
                    im = r*sin(θ)
                    # Imprint random noise onto grid point
                    slab_ptr[index    ] = re
                    slab_ptr[index + 1] = im
                # Handle conjugate imprinting
                if imprint_conj:
                    # Finalize conjugate random noise
                    with unswitch:
                        if phase_shift:
                            θ_conj += phase_shift
                    re_conj = r_conj*cos(θ_conj)
                    im_conj = r_conj*sin(θ_conj)
                    # Imprint conjugate random noise onto grid point
                    i_conj        = -ki_conj + (-(ki_conj > 0) & gridsize)
                    j_global_conj = -kj_conj + (-(kj_conj > 0) & gridsize)
                    j_conj = j_global_conj - ℤ[slab_size_j*rank]
                    index_conj = (j_conj*slab_size_i + i_conj)*slab_size_k  # k = 0
                    slab_ptr[index_conj    ] = +re_conj
                    slab_ptr[index_conj + 1] = -im_conj
    else:
        abort(f'primordial_noise_imprinting = "{primordial_noise_imprinting}" not implemented')
    # Nullify origin (random number was imprinted onto it in the above)
    nullify_modes(slab, 'origin')
    masterprint('done')

# Function for realising particle components
@cython.header(
    # Arguments
    component='Component',
    a='double',
    # Locals
    buffer_gridsize='Py_ssize_t',
    buffer_name=object,  # int or str or None
    buffer_number='int',
    cosmoresults=object,  # CosmoResults
    gridsize='Py_ssize_t',
    gridsize_dealias='Py_ssize_t',
    growth_factors=dict,
    i='int',
    id_bgn='Py_ssize_t',
    indexʳ='Py_ssize_t',
    indexᵖ_bgn='Py_ssize_t',
    lattice='Lattice',
    n_different_sized='Py_ssize_t',
    n_local='Py_ssize_t',
    n_particles='Py_ssize_t',
    nongaussianity='double',
    options=object,  # ParticleRealizationOptions
    particle_components=list,
    pos='double*',
    tmpgrid0='double[:, :, ::1]',
    tmpgrid0_dealias='double[:, :, ::1]',
    tmpgrid1='double[:, :, ::1]',
    tmpgrid1_dealias='double[:, :, ::1]',
    Φ1='double[:, :, ::1]',
    Φ2='double[:, :, ::1]',
    Φ3='double[:, :, ::1]',
    returns='void',
)
def realize_particles(component, a):
    options = ParticleRealizationOptions(
        component.realization_options['backscale'],
        component.realization_options['lpt'],
        component.realization_options['dealias'],
        component.realization_options['nongaussianity'],
        component.realization_options['structure'],
    )
    if options.lpt not in {1, 2, 3}:
        abort(
            f'realize_particles() called with attempted {options.lpt}LPT, '
            f'though only 1LPT, 2LPT and 3LPT are implemented'
        )
    if component.representation != 'particles':
        abort(f'realize_particles() called with non-particle component {component.name}')
    if options.structure != 'primordial':
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
    masterprint(
        f'Realising {{}}{gridsize}³ particles of {component.name} {{}}...'
        .format(f'{len(lattice)}×'*(len(lattice) > 1), 'using back-scaling '*options.backscale)
    )
    # Get growth factors if needed
    growth_factors = {
        'D1': NaN,
        'f1': NaN,
        'D2': NaN,
        'f2': NaN,
        'D3a': NaN,
        'f3a': NaN,
        'D3b': NaN,
        'f3b': NaN,
        'D3c': NaN,
        'f3c': NaN,
    }
    if options.backscale or options.lpt > 1:
        cosmoresults = compute_cosmo(class_call_reason='in order to get growth factors')
        for key in growth_factors:
            growth_factors[key] = getattr(cosmoresults, f'growth_fac_{key}')(a)
    # For 1LPT we need 1 potential grid and one temporary grid
    Φ1, buffer_number = get_lpt_grid(gridsize)
    tmpgrid0, buffer_number = get_lpt_grid(gridsize, buffer_number)
    tmpgrid1 = tmpgrid0
    if options.lpt >= 2:
        # For 2LPT we need 1 additional potential grid
        # and one additional temporary grid.
        Φ2, buffer_number = get_lpt_grid(gridsize, buffer_number)
        tmpgrid1, buffer_number = get_lpt_grid(gridsize, buffer_number)
    if options.lpt >= 3:
        # For 3LPT we need 1 additional potential grid
        Φ3, buffer_number = get_lpt_grid(gridsize, buffer_number)
    # If dealiasing is to be used, additionally fetch
    # two enlarged grids for use with Orszag's 3/2 rule.
    gridsize_dealias = gridsize
    tmpgrid0_dealias = tmpgrid0
    tmpgrid1_dealias = tmpgrid1
    if options.dealias:
        gridsize_dealias = (gridsize_dealias*3)//2
        gridsize_dealias += gridsize_dealias & 1
        tmpgrid0_dealias, buffer_number = get_lpt_grid(gridsize_dealias, buffer_number)
        tmpgrid1_dealias, buffer_number = get_lpt_grid(gridsize_dealias, buffer_number)
    # Realise particles, one lattice at a time
    n_particles = gridsize**3
    indexᵖ_bgn = 0
    id_bgn = n_particles_realized['particles_tally']
    for lattice in lattice:
        # Initialise particles on the lattice
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
        # Regardless of LPT order, the first step is
        # to carry out 1LPT (the Zel'dovich approximation).
        masterprint('Carrying out 1LPT ...')
        carryout_1lpt(
            component, lattice, gridsize, options, a, growth_factors, indexᵖ_bgn,
            Φ1,
            tmpgrid0, tmpgrid1,
        )
        masterprint('done')
        # We now have Φ1 in Fourier space
        notes = {}  # for keeping track of temporary grid contents
        if options.lpt >= 2:
            # Carry out 2LPT
            masterprint('Carrying out 2LPT ...')
            carryout_2lpt(
                component, a, growth_factors, indexᵖ_bgn,
                Φ1, Φ2, tmpgrid0, tmpgrid0_dealias, tmpgrid1, tmpgrid1_dealias, notes,
            )
            masterprint('done')
            # We now have Φ1 and Φ2 in Fourier space
        if options.lpt >= 3:
            masterprint('Carrying out 3LPT ...')
            # Carry out 3LPT ('a' term)
            carryout_3lpt_a(
                component, a, growth_factors, indexᵖ_bgn,
                Φ1, Φ3, tmpgrid0, tmpgrid0_dealias, tmpgrid1, tmpgrid1_dealias, notes,
            )
            # Carry out 3LPT ('b' term)
            carryout_3lpt_b(
                component, a, growth_factors, indexᵖ_bgn,
                Φ1, Φ2, Φ3, tmpgrid0, tmpgrid0_dealias, tmpgrid1, tmpgrid1_dealias, notes,
            )
            # Carry out 3LPT ('c' term)
            for i in range(3):
                carryout_3lpt_c(
                    component, a, growth_factors, indexᵖ_bgn,
                    Φ1, Φ2, Φ3, i, tmpgrid0, tmpgrid0_dealias, tmpgrid1, tmpgrid1_dealias, notes,
                )
            masterprint('done')
        # Prepare for next lattice
        id_bgn += n_particles
        indexᵖ_bgn += n_local
    # Done realising particles
    n_particles_realized['particles_tally'] = id_bgn
    n_particles_realized['components_tally'] += 1
    # Free the memory used for the potential and temporary grids
    for _ in range(len(lpt_grids_info)):
        buffer_gridsize, buffer_name = lpt_grids_info.pop()
        if buffer_name is None:
            lpt_grids_info.appendleft((buffer_gridsize, buffer_name))
        else:
            free_fftw_slab(buffer_gridsize, buffer_name)
    # Ensure toroidal boundaries and exchange
    # particles among the processes.
    pos = component.pos
    for indexʳ in range(3*component.N_local):
        pos[indexʳ] = mod(pos[indexʳ], boxsize)
    exchange(component)
    masterprint('done')
ParticleRealizationOptions = collections.namedtuple(
    'ParticleRealizationOptions',
    ['backscale', 'lpt', 'dealias', 'nongaussianity', 'structure'],
)
# Record updated by the realize_particles() function
cython.declare(n_particles_realized=dict)
n_particles_realized = {
    'components_tally': 0,
    'components_total': 0,
    'particles_tally': 0
}

# Helper function for realize_particles()
def get_lpt_grid(gridsize, buffer_name=None):
    grid = get_fftw_slab(gridsize, buffer_name)
    if (gridsize, buffer_name) not in lpt_grids_info:
        lpt_grids_info.append((gridsize, buffer_name))
    if buffer_name is None:
        buffer_name = 0
    else:
        buffer_name += 1
    return grid, buffer_name
lpt_grids_info = collections.deque()

# Function for carrying out 1LPT particle realization
@cython.header(
    # Arguments
    component='Component',
    lattice='Lattice',
    gridsize='Py_ssize_t',
    options=object,  # ParticleRealizationOptions
    a='double',
    growth_factors=dict,
    indexᵖ_bgn='Py_ssize_t',
    Φ1='double[:, :, ::1]',
    tmpgrid0='double[:, :, ::1]',
    tmpgrid1='double[:, :, ::1]',
    # Locals
    amplitudes='double[::1]',
    i='int',
    noise='double[:, :, ::1]',
    variable='int',
    velocity_factor='double',
    Ψ1ᵢ='double[:, :, ::1]',
    returns='void',
)
def carryout_1lpt(
    component, lattice, gridsize, options, a, growth_factors, indexᵖ_bgn,
    Φ1,
    tmpgrid0, tmpgrid1,
):
    """For 1LPT we need a grid for Φ1 and at least 1 temporary grid.
    If an additional temporary grid is provided, this will be used to
    store the primordial noise, as otherwise we need to generate
    this twice. Once this function returns, the Φ1 potential will be
    populated in k space.
    """
    velocity_factor = a*hubble(a)*growth_factors['f1']
    # The 1LPT potential looks like
    #   ∇²Φ⁽¹⁾ = -δ
    noise = None
    for variable in range(1 - options.backscale, -1, -1):  # first θ, then δ
        if options.backscale:
            masterprint('Building potential ...')
        else:
            masterprint(
                'Building potential ({}) ...'
                .format(['positions', 'velocities'][variable])
            )
        # Get the ampliudes from linear theory
        amplitudes = get_amplitudes(
            gridsize, component, a, variable=variable,
        )
        # Generate the primordial noise. If we only have a single
        # temporary grid, we will need to regenerate this in the next
        # iteration as well.
        if noise is None or np.shares_memory(tmpgrid0, tmpgrid1):
            noise = get_slab_structure(
                gridsize, component, a, 0,
                buffer_or_buffer_name=tmpgrid0,
            )
        # Create potential
        laplacian_inverse(
            realize_grid(
                gridsize, component, a, amplitudes, 0, Φ1,
                lattice=lattice,
                slab_structure=noise,
                # Note that the particle velocities are currently
                # unaffected by the non-Gaussianity.
                nongaussianity=(options.nongaussianity*(variable == 0)),
                output_space='Fourier',
            ),
            factor=(2*variable - 1),
        )
        masterprint('done')
        # Construct displacement field from potential,
        # displace positions and/or boost velocities.
        if options.backscale:
            masterprint('Displacing positions and boosting momenta ...')
        elif variable == 0:
            masterprint('Displacing positions ...')
        elif variable == 1:
            masterprint('Boosting velocities ...')
        for i in range(3):
            Ψ1ᵢ = diff_ifft(Φ1, tmpgrid1, tmpgrid1, i)
            displace_particles(component, Ψ1ᵢ, a, indexᵖ_bgn, variable, i)
            if options.backscale:
                displace_particles(component, Ψ1ᵢ, a, indexᵖ_bgn, 1, i, velocity_factor)
        masterprint('done')

# Function for carrying out 2LPT particle realization
@cython.header(
    # Arguments
    component='Component',
    a='double',
    growth_factors=dict,
    indexᵖ_bgn='Py_ssize_t',
    Φ1='double[:, :, ::1]',
    Φ2='double[:, :, ::1]',
    tmpgrid0='double[:, :, ::1]',
    tmpgrid0_dealias='double[:, :, ::1]',
    tmpgrid1='double[:, :, ::1]',
    tmpgrid1_dealias='double[:, :, ::1]',
    # Locals
    dealias='bint',
    fft_factor='double',
    grids=dict,
    gridsize='Py_ssize_t',
    gridsize_dealias='Py_ssize_t',
    i='int',
    notes=dict,
    potential_factor='double',
    tmpgrid='double[:, :, ::1]',
    tmpgrid_name=str,
    velocity_factor='double',
    Ψ2ᵢ='double[:, :, ::1]',
    returns='void',
)
def carryout_2lpt(
    component, a, growth_factors, indexᵖ_bgn,
    Φ1, Φ2, tmpgrid0, tmpgrid0_dealias, tmpgrid1, tmpgrid1_dealias, notes,
):
    """For 2LPT we need the pre-computed Φ1 and 2 temporary grids.
    Once this function returns, the Φ2 potential
    will be populated in k space.
    """
    gridsize = tmpgrid0.shape[1]
    gridsize_dealias = tmpgrid0_dealias.shape[1]
    dealias = (gridsize_dealias > gridsize)
    fft_factor = float(gridsize_dealias)**(-3)
    potential_factor = fft_factor*growth_factors['D2']/growth_factors['D1']**2
    velocity_factor = a*hubble(a)*growth_factors['f2']
    # The 2LPT potential looks like
    #   ∇²Φ⁽²⁾ = D⁽²⁾/(D⁽¹⁾)² (
    #       - Φ⁽¹⁾,₀₀ Φ⁽¹⁾,₁₁
    #       - Φ⁽¹⁾,₁₁ Φ⁽¹⁾,₂₂
    #       - Φ⁽¹⁾,₂₂ Φ⁽¹⁾,₀₀
    #       + Φ⁽¹⁾,₀₁²
    #       + Φ⁽¹⁾,₁₂²
    #       + Φ⁽¹⁾,₂₀²
    #   )
    # where all growth factors are taken to be positive.
    grids = {
        'Φ⁽¹⁾': Φ1,
        'Φ⁽²⁾': Φ2,
        'tmp0': (tmpgrid0, tmpgrid0_dealias),
        'tmp1': (tmpgrid1, tmpgrid1_dealias),
    }
    masterprint('Building potential ...')
    handle_lpt_term('Φ⁽²⁾  = Φ⁽¹⁾,₀₀ Φ⁽¹⁾,₁₁', grids, notes, factor=-1)
    handle_lpt_term('Φ⁽²⁾ -= Φ⁽¹⁾,₁₁ Φ⁽¹⁾,₂₂', grids, notes)
    handle_lpt_term('Φ⁽²⁾ -= Φ⁽¹⁾,₂₂ Φ⁽¹⁾,₀₀', grids, notes)
    handle_lpt_term('Φ⁽²⁾ += Φ⁽¹⁾,₀₁²',        grids, notes)
    handle_lpt_term('Φ⁽²⁾ += Φ⁽¹⁾,₁₂²',        grids, notes)
    handle_lpt_term('Φ⁽²⁾ += Φ⁽¹⁾,₂₀²',        grids, notes)
    if not dealias:
        fft(Φ2, 'forward')
    laplacian_inverse(Φ2, potential_factor)
    masterprint('done')
    # Construct displacement field from potential,
    # displace and boost particles.
    masterprint('Displacing positions and boosting momenta ...')
    tmpgrid_name, tmpgrid, _ = get_tmpgrid(grids)
    notes.pop(tmpgrid_name, None)
    for i in range(3):
        Ψ2ᵢ = diff_ifft(Φ2, tmpgrid, tmpgrid, i)
        displace_particles(component, Ψ2ᵢ, a, indexᵖ_bgn, 0, i)
        displace_particles(component, Ψ2ᵢ, a, indexᵖ_bgn, 1, i, velocity_factor)
    masterprint('done')

# Function for carrying out 3LPT ('a' term only) particle realization
@cython.header(
    # Arguments
    component='Component',
    a='double',
    growth_factors=dict,
    indexᵖ_bgn='Py_ssize_t',
    Φ1='double[:, :, ::1]',
    Φ3a='double[:, :, ::1]',
    tmpgrid0='double[:, :, ::1]',
    tmpgrid0_dealias='double[:, :, ::1]',
    tmpgrid1='double[:, :, ::1]',
    tmpgrid1_dealias='double[:, :, ::1]',
    notes=dict,
    # Locals
    dealias='bint',
    fft_factor='double',
    grids=dict,
    gridsize='Py_ssize_t',
    gridsize_dealias='Py_ssize_t',
    i='int',
    potential_factor='double',
    tmpgrid='double[:, :, ::1]',
    tmpgrid_name=str,
    velocity_factor='double',
    Ψ3aᵢ='double[:, :, ::1]',
    returns='void',
)
def carryout_3lpt_a(
    component, a, growth_factors, indexᵖ_bgn,
    Φ1, Φ3a, tmpgrid0, tmpgrid0_dealias, tmpgrid1, tmpgrid1_dealias, notes,
):
    """For 3LPT ('a' term) we need the pre-computed Φ⁽¹⁾
    and 2 temporary grids.
    """
    gridsize = tmpgrid0.shape[1]
    gridsize_dealias = tmpgrid0_dealias.shape[1]
    dealias = (gridsize_dealias > gridsize)
    fft_factor = float(gridsize_dealias)**(-3)
    potential_factor = fft_factor*growth_factors['D3a']/growth_factors['D1']**3
    velocity_factor = a*hubble(a)*growth_factors['f3a']
    # The 3LPT (a) potential looks like
    #   ∇²Φ⁽³ᵃ⁾ = D⁽³ᵃ⁾/(D⁽¹⁾)³ (
    #       + Φ⁽¹⁾,₂₀² Φ⁽¹⁾,₁₁
    #       - Φ⁽¹⁾,₁₁ Φ⁽¹⁾,₂₂ Φ⁽¹⁾,₀₀
    #       + Φ⁽¹⁾,₀₀ Φ⁽¹⁾,₁₂²
    #       - 2 Φ⁽¹⁾,₁₂ Φ⁽¹⁾,₂₀ Φ⁽¹⁾,₀₁
    #       + Φ⁽¹⁾,₀₁² Φ⁽¹⁾,₂₂
    #   )
    # where all growth factors are taken to be positive.
    grids = {
        'Φ⁽¹⁾': Φ1,
        'Φ⁽³ᵃ⁾': Φ3a,
        'tmp0': (tmpgrid0, tmpgrid0_dealias),
        'tmp1': (tmpgrid1, tmpgrid1_dealias),
    }
    masterprint('Building potential (a) ...')
    handle_lpt_term('Φ⁽³ᵃ⁾  = Φ⁽¹⁾,₂₀² Φ⁽¹⁾,₁₁',        grids, notes)
    handle_lpt_term('Φ⁽³ᵃ⁾ -= Φ⁽¹⁾,₁₁ Φ⁽¹⁾,₂₂ Φ⁽¹⁾,₀₀', grids, notes)
    handle_lpt_term('Φ⁽³ᵃ⁾ += Φ⁽¹⁾,₀₀ Φ⁽¹⁾,₁₂²',        grids, notes)
    handle_lpt_term('Φ⁽³ᵃ⁾ -= Φ⁽¹⁾,₁₂ Φ⁽¹⁾,₂₀ Φ⁽¹⁾,₀₁', grids, notes, factor=2)
    handle_lpt_term('Φ⁽³ᵃ⁾ += Φ⁽¹⁾,₀₁² Φ⁽¹⁾,₂₂',        grids, notes)
    if not dealias:
        fft(Φ3a, 'forward')
    laplacian_inverse(Φ3a, potential_factor)
    masterprint('done')
    # Construct displacement field from potential,
    # displace and boost particles.
    masterprint('Displacing positions and boosting momenta ...')
    tmpgrid_name, tmpgrid, _ = get_tmpgrid(grids)
    notes.pop(tmpgrid_name, None)
    for i in range(3):
        Ψ3aᵢ = diff_ifft(Φ3a, tmpgrid, tmpgrid, i)
        displace_particles(component, Ψ3aᵢ, a, indexᵖ_bgn, 0, i)
        displace_particles(component, Ψ3aᵢ, a, indexᵖ_bgn, 1, i, velocity_factor)
    masterprint('done')

# Function for carrying out 3LPT ('b' term only) particle realization
@cython.header(
    # Arguments
    component='Component',
    a='double',
    growth_factors=dict,
    indexᵖ_bgn='Py_ssize_t',
    Φ1='double[:, :, ::1]',
    Φ2='double[:, :, ::1]',
    Φ3b='double[:, :, ::1]',
    tmpgrid0='double[:, :, ::1]',
    tmpgrid0_dealias='double[:, :, ::1]',
    tmpgrid1='double[:, :, ::1]',
    tmpgrid1_dealias='double[:, :, ::1]',
    notes=dict,
    # Locals
    dealias='bint',
    fft_factor='double',
    grids=dict,
    gridsize='Py_ssize_t',
    gridsize_dealias='Py_ssize_t',
    i='int',
    potential_factor='double',
    tmpgrid='double[:, :, ::1]',
    tmpgrid_name=str,
    velocity_factor='double',
    Ψ3bᵢ='double[:, :, ::1]',
    returns='void',
)
def carryout_3lpt_b(
    component, a, growth_factors, indexᵖ_bgn,
    Φ1, Φ2, Φ3b, tmpgrid0, tmpgrid0_dealias, tmpgrid1, tmpgrid1_dealias, notes,
):
    """For 3LPT ('b' term) we need the pre-computed Φ⁽¹⁾ and Φ⁽²⁾
    and 2 temporary grids.
    """
    gridsize = tmpgrid0.shape[1]
    gridsize_dealias = tmpgrid0_dealias.shape[1]
    dealias = (gridsize_dealias > gridsize)
    fft_factor = float(gridsize_dealias)**(-3)
    potential_factor = fft_factor*growth_factors['D3b']/(growth_factors['D1']*growth_factors['D2'])
    velocity_factor = a*hubble(a)*growth_factors['f3b']
    # The 3LPT (b) potential looks like
    #   ∇²Φ⁽³ᵇ⁾ = D⁽³ᵇ⁾/(D⁽¹⁾D⁽²⁾) (
    #       - ½ Φ⁽¹⁾,₂₂ Φ⁽²⁾,₀₀
    #       - ½ Φ⁽²⁾,₀₀ Φ⁽¹⁾,₁₁
    #       - ½ Φ⁽¹⁾,₁₁ Φ⁽²⁾,₂₂
    #       - ½ Φ⁽²⁾,₂₂ Φ⁽¹⁾,₀₀
    #       - ½ Φ⁽¹⁾,₀₀ Φ⁽²⁾,₁₁
    #       - ½ Φ⁽²⁾,₁₁ Φ⁽¹⁾,₂₂
    #       + Φ⁽²⁾,₂₀ Φ⁽¹⁾,₂₀
    #       + Φ⁽²⁾,₀₁ Φ⁽¹⁾,₀₁
    #       + Φ⁽²⁾,₁₂ Φ⁽¹⁾,₁₂
    #   )
    # where all growth factors are taken to be positive.
    grids = {
        'Φ⁽¹⁾': Φ1,
        'Φ⁽²⁾': Φ2,
        'Φ⁽³ᵇ⁾': Φ3b,
        'tmp0': (tmpgrid0, tmpgrid0_dealias),
        'tmp1': (tmpgrid1, tmpgrid1_dealias),
    }
    masterprint('Building potential (b) ...')
    handle_lpt_term('Φ⁽³ᵇ⁾  = Φ⁽¹⁾,₂₂ Φ⁽²⁾,₀₀', grids, notes, factor=-0.5)
    handle_lpt_term('Φ⁽³ᵇ⁾ -= Φ⁽²⁾,₀₀ Φ⁽¹⁾,₁₁', grids, notes, factor=0.5)
    handle_lpt_term('Φ⁽³ᵇ⁾ -= Φ⁽¹⁾,₁₁ Φ⁽²⁾,₂₂', grids, notes, factor=0.5)
    handle_lpt_term('Φ⁽³ᵇ⁾ -= Φ⁽²⁾,₂₂ Φ⁽¹⁾,₀₀', grids, notes, factor=0.5)
    handle_lpt_term('Φ⁽³ᵇ⁾ -= Φ⁽¹⁾,₀₀ Φ⁽²⁾,₁₁', grids, notes, factor=0.5)
    handle_lpt_term('Φ⁽³ᵇ⁾ -= Φ⁽²⁾,₁₁ Φ⁽¹⁾,₂₂', grids, notes, factor=0.5)
    handle_lpt_term('Φ⁽³ᵇ⁾ += Φ⁽²⁾,₂₀ Φ⁽¹⁾,₂₀', grids, notes)
    handle_lpt_term('Φ⁽³ᵇ⁾ += Φ⁽²⁾,₀₁ Φ⁽¹⁾,₀₁', grids, notes)
    handle_lpt_term('Φ⁽³ᵇ⁾ += Φ⁽²⁾,₁₂ Φ⁽¹⁾,₁₂', grids, notes)
    if not dealias:
        fft(Φ3b, 'forward')
    laplacian_inverse(Φ3b, potential_factor)
    masterprint('done')
    # Construct displacement field from potential,
    # displace and boost particles.
    masterprint('Displacing positions and boosting momenta ...')
    tmpgrid_name, tmpgrid, _ = get_tmpgrid(grids)
    notes.pop(tmpgrid_name, None)
    for i in range(3):
        Ψ3bᵢ = diff_ifft(Φ3b, tmpgrid, tmpgrid, i)
        displace_particles(component, Ψ3bᵢ, a, indexᵖ_bgn, 0, i)
        displace_particles(component, Ψ3bᵢ, a, indexᵖ_bgn, 1, i, velocity_factor)
    masterprint('done')

# Function for carrying out 3LPT ('c' term only; A3ᵢ)
# particle realization.
@cython.header(
    # Arguments
    component='Component',
    a='double',
    growth_factors=dict,
    indexᵖ_bgn='Py_ssize_t',
    Φ1='double[:, :, ::1]',
    Φ2='double[:, :, ::1]',
    A3cᵢ='double[:, :, ::1]',
    i='int',
    tmpgrid0='double[:, :, ::1]',
    tmpgrid0_dealias='double[:, :, ::1]',
    tmpgrid1='double[:, :, ::1]',
    tmpgrid1_dealias='double[:, :, ::1]',
    notes=dict,
    # Locals
    dealias='bint',
    fft_factor='double',
    grids=dict,
    gridsize='Py_ssize_t',
    gridsize_dealias='Py_ssize_t',
    j='int',
    k='int',
    potential_factor='double',
    sign='double',
    tmpgrid='double[:, :, ::1]',
    tmpgrid_name=str,
    velocity_factor='double',
    Ψ3cⱼ='double[:, :, ::1]',
    returns='void',
)
def carryout_3lpt_c(
    component, a, growth_factors, indexᵖ_bgn,
    Φ1, Φ2, A3cᵢ, i, tmpgrid0, tmpgrid0_dealias, tmpgrid1, tmpgrid1_dealias, notes,
):
    """For 3LPT ('c' term; A3cᵢ) we need the pre-computed Φ⁽¹⁾ and Φ⁽²⁾
    and 2 temporary grids.
    """
    gridsize = tmpgrid0.shape[1]
    gridsize_dealias = tmpgrid0_dealias.shape[1]
    dealias = (gridsize_dealias > gridsize)
    fft_factor = float(gridsize_dealias)**(-3)
    potential_factor = fft_factor*growth_factors['D3c']/(growth_factors['D1']*growth_factors['D2'])
    velocity_factor = a*hubble(a)*growth_factors['f3c']
    #   ∇²Aᵢ⁽³ᶜ⁾ = D⁽³ᶜ⁾/(D⁽¹⁾D⁽²⁾) (
    #       + Φ⁽²⁾,ⱼⱼ Φ⁽¹⁾,ⱼₖ
    #       - Φ⁽¹⁾,ⱼₖ Φ⁽²⁾,ₖₖ
    #       - Φ⁽¹⁾,ᵢⱼ Φ⁽²⁾,ᵢₖ
    #       - Φ⁽¹⁾,ⱼⱼ Φ⁽²⁾,ⱼₖ
    #       + Φ⁽²⁾,ⱼₖ Φ⁽¹⁾,ₖₖ
    #       + Φ⁽²⁾,ᵢⱼ Φ⁽¹⁾,ᵢₖ
    #   )
    # The displacement field is
    #   Ψ⁽³ᶜ⁾ᵢ = Aₖ,ⱼ - Aⱼ,ₖ
    # All growth factors are taken to be positive.
    grids = {
        'Φ⁽¹⁾': Φ1,
        'Φ⁽²⁾': Φ2,
        'Aᵢ⁽³ᶜ⁾': A3cᵢ,
        'tmp0': (tmpgrid0, tmpgrid0_dealias),
        'tmp1': (tmpgrid1, tmpgrid1_dealias),
    }
    masterprint(
        'Building potential (c, {}) ...'
        .format(''.join(np.roll(list('xyz'), -i)[1:]))
    )
    j = (i + 1)%3
    k = (i + 2)%3
    handle_lpt_term('Aᵢ⁽³ᶜ⁾  = Φ⁽²⁾,ⱼⱼ Φ⁽¹⁾,ⱼₖ', grids, notes, i, j, k)
    handle_lpt_term('Aᵢ⁽³ᶜ⁾ -= Φ⁽¹⁾,ⱼₖ Φ⁽²⁾,ₖₖ', grids, notes, i, j, k)
    handle_lpt_term('Aᵢ⁽³ᶜ⁾ -= Φ⁽¹⁾,ᵢⱼ Φ⁽²⁾,ᵢₖ', grids, notes, i, j, k)
    handle_lpt_term('Aᵢ⁽³ᶜ⁾ -= Φ⁽¹⁾,ⱼⱼ Φ⁽²⁾,ⱼₖ', grids, notes, i, j, k)
    handle_lpt_term('Aᵢ⁽³ᶜ⁾ += Φ⁽²⁾,ⱼₖ Φ⁽¹⁾,ₖₖ', grids, notes, i, j, k)
    handle_lpt_term('Aᵢ⁽³ᶜ⁾ += Φ⁽²⁾,ᵢⱼ Φ⁽¹⁾,ᵢₖ', grids, notes, i, j, k)
    if not dealias:
        fft(A3cᵢ, 'forward')
    laplacian_inverse(A3cᵢ, potential_factor)
    masterprint('done')
    # Construct displacement field from potential,
    # displace and boost particles.
    masterprint('Displacing positions and boosting momenta ...')
    tmpgrid_name, tmpgrid, _ = get_tmpgrid(grids)
    notes.pop(tmpgrid_name, None)
    for j in range(3):
        if j == i:
            continue
        k = (set(range(3)) - {i, j}).pop()
        sign = 2*(k == (j + 1)%3) - 1
        Ψ3cⱼ = diff_ifft(A3cᵢ, tmpgrid, tmpgrid, k, factor=sign)
        displace_particles(component, Ψ3cⱼ, a, indexᵖ_bgn, 0, j)
        displace_particles(component, Ψ3cⱼ, a, indexᵖ_bgn, 1, j, velocity_factor)
    masterprint('done')

# Function handling each term of the full expression for an
# LPT potential, taking in a symbolic expression,
# parsing it and performing the grid operations.
@cython.pheader(
    # Arguments
    expr=str,
    grids=dict,
    notes=dict,
    i='int',
    j='int',
    k='int',
    factor='double',
    # Locals
    assignop=str,
    basegrid='double[:, :, ::1]',
    basegrid_dealias='double[:, :, ::1]',
    basegrid_dealias_ptr='double*',
    basegrid_name=str,
    basegrid_ptr='double*',
    dealias='bint',
    fft_factor='double',
    grid='double[:, :, ::1]',
    grid_dealias='double[:, :, ::1]',
    gridsize='Py_ssize_t',
    gridsize_dealias='Py_ssize_t',
    ijk_mapping=dict,
    index='Py_ssize_t',
    n='int',
    n_pot='int',
    name=str,
    name_reuse=str,
    reuse='bint',
    size='Py_ssize_t',
    size_dealias='Py_ssize_t',
    term=list,
    tmpgrid='double[:, :, ::1]',
    tmpgrid_dealias='double[:, :, ::1]',
    tmpgrid_dealias_ptr='double*',
    tmpgrid_name=str,
    tmpgrid_ptr='double*',
    Φ_in='double[:, :, ::1]',
    Φ_out='double[:, :, ::1]',
    Φ_out_ptr='double*',
    returns='void',
)
def handle_lpt_term(expr, grids, notes, i=0, j=0, k=0, factor=1):
    """A real-space expression like
      Φ⁽³ᵃ⁾ = Φ⁽¹⁾,₀₀ Φ⁽¹⁾,₁₁ Φ⁽¹⁾,₂₂
    is evaluated from the passed grids (must be in Fourier space).
    The resulting grid will be in real space unless dealiasing is used.
    Indices ᵢ, ⱼ and ₖ are replaced with the passed values.
    The differentiated grids will be constructed in turn, using the
    output and temporary grids supplied. If dealiasing is to be used,
    we will apply it after every multiplication of a pair of grids,
    i.e. twice for the above example. This is however only an
    approximation; exact dealiasing is obtained from a single dealiasing
    step but from a grid that is enlarged by a factor 2 instead of 3/2,
    which we do no implement.
    """
    # Ensure Unicode input
    expr = unicode(expr).replace(' ', '')
    for name in list(grids):
        grids[unicode(name)] = grids.pop(name)
    # Parse expr
    assignops = ['+=', '-=', '=']
    for assignop in assignops:
        if assignop in expr:
            break
    else:
        abort(f'Invalid expression passed to handle_lpt_term(): {expr}')
    lhs, rhs = expr.split(assignop)
    ijk_mapping = {
        unicode('ᵢ'): unicode_subscripts[str(i)],
        unicode('ⱼ'): unicode_subscripts[str(j)],
        unicode('ₖ'): unicode_subscripts[str(k)],
    }
    term = []
    for pot in rhs.split(unicode('Φ'))[1:]:
        pot = unicode('Φ') + pot
        for n in range(10):
            n_superscript = unicode_superscripts[str(n)]
            if pot.endswith(n_superscript):
                pot = pot.removesuffix(n_superscript)
                break
        else:
            n = 1
        pot, diff = pot.split(',')  # assumes diff in all sub-expressions
        pot = (pot, ) + tuple(sorted([
            int(unicode_subscripts_inv[ijk_mapping.get(s, s)])
            for s in diff
        ]))
        term += [pot]*n
    n_pot = len(term)
    if n_pot < 2:
        abort(f'handle_lpt_term() got expr = "{expr}" containing {n_pot} < 2 potentials')
    # General grid information
    grid, grid_dealias = grids['tmp0']
    size = grid.shape[0]*grid.shape[1]*grid.shape[2]
    size_dealias = grid_dealias.shape[0]*grid_dealias.shape[1]*grid_dealias.shape[2]
    gridsize = grid.shape[1]
    gridsize_dealias = grid_dealias.shape[1]
    dealias = gridsize_dealias > gridsize
    # If dealiasing is to be performed, we do so after every grid
    # multiplication, of which there are (n_pot - 1). The last one is
    # handled specially, as we simultaneously shrink the grid.
    # The remaining (n_pot - 2) comed with an fft_factor.
    if dealias:
        fft_factor = float(gridsize_dealias)**(-3)
        factor *= fft_factor**(n_pot - 2)
    # Get output grid in which to accumulate the evaluated term
    Φ_out = grids[lhs]
    Φ_out_ptr = cython.address(Φ_out[:, :, :])
    # We want to reuse grids from previous calls as much as possible.
    # The current contents of the temporary grids are written down in
    # the notes. If the first grid needed is present amongst the
    # temporary grids, reuse that. If not, and the second (unique) grid
    # is present, reuse that. Swap the order of grids accordingly.
    pots_unique = []
    for pot in term:
        if pot not in pots_unique:
            pots_unique.append(pot)
    for n, pot in enumerate(pots_unique[:2]):
        if pot not in notes.values():
            continue
        for name_reuse in notes:
            if notes[name_reuse] == pot:
                # If this is the first potential, place the reusable
                # grid in the front (will be used as base grid). If not,
                # place the reusable grid in the back (will be used as
                # temporary grid).
                for name in list(grids):
                    if (n == 0 and name != name_reuse) or (n != 0 and name == name_reuse):
                        grids[name] = grids.pop(name)
                break
        break
    # Obtain the first occuring potential within the expression,
    # differentiated as needed. This will be stored in the base grid.
    basegrid_name, basegrid, basegrid_dealias = get_tmpgrid(grids)
    basegrid_ptr = cython.address(basegrid[:, :, :])
    basegrid_dealias_ptr = cython.address(basegrid_dealias[:, :, :])
    pot = term[0]
    Φ_in, i, j = expand_pot(pot, grids)
    if notes.get(basegrid_name) != pot:
        diff_ifft(Φ_in, basegrid, basegrid_dealias, i, j)
    else:
        # Reuse base grid as is
        pass
    notes.pop(basegrid_name, None)
    # Obtain grid for temporary storage
    tmpgrid_name, tmpgrid, tmpgrid_dealias = get_tmpgrid(grids)
    tmpgrid_ptr = cython.address(tmpgrid[:, :, :])
    tmpgrid_dealias_ptr = cython.address(tmpgrid_dealias[:, :, :])
    # Multiply remaining grids into base grid
    for n, pot in enumerate(term[1:], 1):
        # Obtain the next grid
        if n == 1 and pot == term[0]:
            # Squared sub-expression at the beginning.
            # Copy base grid into temporary grid. We could avoid this
            # copying, but doing it allows for later reusage.
            tmpgrid_dealias[...] = basegrid_dealias
        elif notes.get(tmpgrid_name) != pot:
            # Construct grid
            Φ_in, i, j = expand_pot(pot, grids)
            diff_ifft(Φ_in, tmpgrid, tmpgrid_dealias, i, j)
        else:
            # Reuse temporary grid as is
            pass
        # Note down the contents of the temporary grid
        notes[tmpgrid_name] = pot
        # Multiply into base grid
        for index in range(size_dealias):
            basegrid_dealias_ptr[index] *= tmpgrid_dealias_ptr[index]
        # Perform explicit dealiasing if not at the last potential
        if dealias and n < n_pot - 1:
            fft(basegrid_dealias, 'forward')
            nullify_modes(basegrid_dealias, f'beyond cube of |k| < {gridsize//2}')
            fft(basegrid_dealias, 'backward')
    # Shrink the base grid if using dealiasing
    # (counts as the last dealiasing).
    if dealias:
        fft(basegrid_dealias, 'forward')
        resize_grid(
            basegrid_dealias, gridsize, 'fourier',
            output_slab_or_buffer_name=basegrid,
        )
        # Note that we stay in Fourier space
    # Move constructed term from base grid
    # to output grid and apply factor.
    for index in range(size):
        with unswitch:
            if assignop == '=':
                with unswitch:
                    if factor == 1:
                        Φ_out_ptr[index] = basegrid_ptr[index]
                    else:
                        Φ_out_ptr[index] = factor*basegrid_ptr[index]
            elif assignop == '+=':
                with unswitch:
                    if factor == 1:
                        Φ_out_ptr[index] += basegrid_ptr[index]
                    else:
                        Φ_out_ptr[index] += factor*basegrid_ptr[index]
            elif assignop == '-=':
                with unswitch:
                    if factor == 1:
                        Φ_out_ptr[index] -= basegrid_ptr[index]
                    else:
                        Φ_out_ptr[index] -= factor*basegrid_ptr[index]
# Helper functions for handle_lpt_term(()
def expand_pot(pot, grids):
    i, j = (pot + (-1, ))[1:3]
    Φ_in = grids[pot[0]]
    return Φ_in, i, j
def get_tmpgrid(grids):
    # Returns the first tmpgrid inside the grids dict.
    # This grid will be moved to the end, so that the
    # tmpgrids are being cycled through.
    for name in grids:
        if name.startswith('tmp'):
            break
    else:
        abort('get_tmpgrid(): No temporary grids available')
    grid, grid_dealias = grids.pop(name)
    grids[name] = (grid, grid_dealias)
    return name, grid, grid_dealias

# Helper function for performing differentiation and inverse Fourier
# transformation of a Fourier space slab, with enlargement for use with
# dealiasing built-in.
@cython.pheader(
    # Arguments
    Φ='double[:, :, ::1]',
    out='double[:, :, ::1]',
    out_dealias='double[:, :, ::1]',
    i='int',
    j='int',
    factor='double',
    # Locals
    dealias='bint',
    gridsize='Py_ssize_t',
    gridsize_dealias='Py_ssize_t',
    returns='double[:, :, ::1]',
)
def diff_ifft(Φ, out, out_dealias, i, j=-1, factor=1):
    gridsize = Φ.shape[1]
    gridsize_dealias = (gridsize if out_dealias is None else out_dealias.shape[1])
    dealias = gridsize_dealias > gridsize
    out = fourier_diff(Φ, i, j, slab_output=out, factor=factor)
    if dealias:
        # Enlarge grid
        resize_grid(out, gridsize_dealias, 'fourier', output_slab_or_buffer_name=out_dealias)
        out = out_dealias
    fft(out, 'backward')
    return out

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
    ψᵢ='double[:, :, ::1]',
    ψᵢ_ptr='double*',
    returns='void',
)
def displace_particles(component, slab, a, indexᵖ_bgn, variable, dim, factor=1):
    if component.representation != 'particles':
        abort(f'displace_particles() called with non-particle component {component.name}')
    # Domain-decompose realised real-space grid.
    # This is either the displacement field ψᵢ or the velocity field uᵢ.
    ψᵢ = domain_decompose(slab)
    ψᵢ_ptr = cython.address(ψᵢ[:, :, :])
    # Grab particle data array in accordance with the passed variable.
    # Also adapt the factor by which the grid values should
    # be multiplied before added to the particle data.
    if variable == 0:
        # Positions, Δxᵢ = ψᵢ
        data = component.pos
    elif variable == 1:
        # Momenta; momᵢ = a*m*uᵢ.
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
                data[indexʳ] += ψᵢ_ptr[index]
            else:
                data[indexʳ] += factor*ψᵢ_ptr[index]
        indexʳ += 3



# Get local domain information
domain_info = get_domain_info()
cython.declare(
    domain_subdivisions='int[::1]',
    domain_layout_local_indices='int[::1]',
)
domain_subdivisions         = domain_info.subdivisions
domain_layout_local_indices = domain_info.layout_local_indices
