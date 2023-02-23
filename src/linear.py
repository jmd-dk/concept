# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2021 Jeppe Mosgaard Dakin.
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
    'from communication import        '
    '    domain_layout_local_indices, '
    '    exchange,                    '
    '    get_buffer,                  '
    '    partition,                   '
    '    smart_mpi,                   '
)
cimport('from graphics import plot_detrended_perturbations')
cimport('from integration import Spline, cosmic_time, remove_doppelgängers, hubble, Ḣ, ȧ, ä')
cimport(
    'from mesh import                         '
    '    domain_decompose,                    '
    '    fft,                                 '
    '    fourier_loop,                        '
    '    get_fftw_slab,                       '
    '    interpolate_domaingrid_to_particles, '
    '    nullify_modes,                       '
    '    slab_decompose,                      '
)



# Class storing the internal state for generation of pseudo-random
# numbers and implementing probability distributions.
# NumPy is used in both compiled and pure Python mode.
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
        except:
            continue
        streams[name] = attr

    # Initialisation method
    @cython.pheader(
        # Arguments
        seed=object,  # Python int or None
        stream=str,
    )
    def __init__(self, seed=None, stream=random_generator):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the RandomNumberGenerator type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public object seed  # Python int or None
        public str stream
        object generator  # np.random.Generator
        Py_ssize_t cache_size
        double[::1] cache_uniform
        double[::1] cache_gaussian
        double[::1] cache_rayleigh
        Py_ssize_t index_uniform
        Py_ssize_t index_gaussian
        Py_ssize_t index_rayleigh
        """
        self.seed = seed
        self.stream = stream
        # Fixed size of internal distribution caches
        self.cache_size = 2**12
        # Look up requested bit stream generator
        generator = self.streams.get(stream)
        if generator is None and stream == 'PCG64DXSM':
            # Older versions of NumPy do not have the DXSM version
            # of PCG64. Allow falling back to the older PCG64 version.
            masterwarn(
                f'Pseudo-random bit generator "{stream}" not available in NumPy. '
                f'Falling back to "PCG64".'
            )
            stream = 'PCG64'
            generator = self.streams.get(stream)
        if generator is None:
            streams_str = ', '.join([f'"{stream}"' for stream in self.streams])
            abort(
                f'Pseudo-random bit generator "{stream}" not available in NumPy. '
                f'The available ones are {streams_str}.'
            )
        # Instantiate a seeded pseudo-random number generator
        self.generator = np.random.Generator(generator(self.seed))
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
prng_general = PseudoRandomNumberGenerator(1 + random_seed + rank)
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

# Class storing a classy.Class instance
# together with the corresponding |k| values
# and results retrieved from the classy.Class instance.
class CosmoResults:
    # Names of scalar attributes
    attribute_names = ('h', )
    # Class used instead of regular dict to store the CLASS
    # perturbations. The only difference is that the class below will
    # instantiate implicit perturbations missing from the CLASS output,
    # such as the squared photon sound speed perturbation "cs2_g" which
    # is always equal to 1/3.
    class PerturbationDict(dict):
        def __getitem__(self, key):
            # Attempt normal lookup
            try:
                return super().__getitem__(key)
            except KeyError:
                pass
            # Normal lookup failed; the perturbation is missing
            match = re.search('_(.*)', key)
            if not match:
                abort(
                    f'Non-existing perturbation "{key}" required. '
                    f'The CLASS species could not be determined.'
                )
            class_species = match.group(1)
            species_info = species_registered.get(
                species_canonical.get(class_species, class_species)
            )
            if species_info is None:
                abort(
                    f'Non-existing perturbation "{key}" required. The CLASS species '
                    f'"{species_info}" is not registered with linear.register_species().'
                )
            # If this perturbation can be inferred, add it
            if key.startswith('cs2_'):
                # The cs2 perturbation is zero for w = 0 and 1/3 for
                # w = 1/3. For other values of w this cannot be
                # easily determined.
                if species_info.w == 0:
                    value = zeros(self['a'].size, dtype=C2np['double'])
                elif species_info.w == 1/3:
                    value = 1/3*ones(self['a'].size, dtype=C2np['double'])
                else:
                    abort(
                        f'Non-existing perturbation "{key}" required. The CLASS species '
                        f'"{species_info}" is registered to have w = {species_info.w}, '
                        f'from which with "{key}" cannot be inferred.'
                    )
            elif key.startswith('shear_'):
                # Missing shear perturbations typically imply that this
                # is zero. Assume so hear.
                value = zeros(self['a'].size, dtype=C2np['double'])
            else:
                abort(
                    f'Non-existing perturbation "{key}" required. '
                    f'This perturbation could not be inferred.'
                )
            self[key] = value
            return value
        def get(self, key, value=None):
            try:
                value = self.__getitem__(key)
            except KeyError:
                pass
            return value
    # Initialise instance
    def __init__(self, params, k_magnitudes, cosmo=None, filename='', class_call_reason=''):
        """If no cosmo object is passed, all results should be loaded
        from disk, if possible. The first time this fails, CLASS will be
        called and a cosmo object will be produced.
        All methods of the cosmo object used in the code which have
        no arguments are here written as attributes using the magic of
        the property decorator. Methods with arguments should also be
        defined in such a way that their results are cached.
        If a filename is passed, CLASS data will be read from this file.
        Nothing will however be saved to this file.
        """
        # Only part of the computed CLASS data is needed.
        # Below, the keys corresponding to the needed fields of CLASS
        # data is written as regular expressions.
        # This dict need to be an instance variable, as it may be
        # mutated by the methods.
        gauge = (params if params else {}).get('gauge', 'synchronous').lower()
        self.needed_keys = {
            # Background data as function of time
            'background': {
                # Time
                r'^a$',
                r'^z$',
                r'^proper time \[Gyr\]$',
                r'^H \[1/Mpc\]$',
                # Density
                r'^\(\.\)rho_',
                # Pressure
                r'^\(\.\)p_',
                # Equation of state
                r'^\(\.\)w_',
                # Other
                r'^gr.fac. f$',
            },
            # Perturbations at different k as function of time.
            # Species specific perturbations will be added later.
            'perturbations': {
                # Time
                r'^a$',
                # Other
                *([r'^h_prime$'] if gauge == 'synchronous' else []),
                r'^theta_tot$',
            },
        }
        # Store the supplied objects
        self.params = params
        self.k_magnitudes = k_magnitudes
        # Store the cosmo object as a hidden attribute
        self._cosmo = cosmo
        # Determine the filename for read/write
        if filename:
            # If a filename is given, no ID is needed. Set it to None.
            self.id = None
            self.filename = filename
            if master:
                if not os.path.isfile(filename):
                    abort(f'The supplied file "{self.filename}" does not exist')
        else:
            # Unique ID and corresponding file name of this CosmoResults
            # object based on the hash of the CLASS parameters,
            # the user specified extra CLASS background quantities and
            # perturbations (if any), as well as the CLASS variables
            # _VERSION, _ARGUMENT_LENGTH_MAX_ and a_min.
            # We use a sha1 hash, which is 40 characters (hexadecimals)
            # long. For the sake of short filenames, we only use the
            # first sha_length characters.
            sha_length = 10  # 10 -> 50% chance of 1 hash collision after ~10⁶ hashes
            self.id = hashlib.sha1(str(
                tuple(sorted({str(key).replace(' ', ''): str(val).replace(' ', '').lower()
                    for key, val in self.params.items()}.items()))
                + (class__VERSION_, class__ARGUMENT_LENGTH_MAX_, class_a_min)
            ).encode('utf-8')).hexdigest()[:sha_length]
            self.filename = f'{path.reusable_dir}/class/{self.id}.hdf5'
        # Message that gets printed if and when CLASS is called
        self.class_call_reason = class_call_reason
        # Add methods which return transfer function splines for a
        # given a. The method names are those of the registered
        # transfer functions given by transferfunctions_registered.
        def construct_func(var_name):
            return (
                lambda a=-1, a_next=-1, component=None, get='as_function_of_k', weight=None:
                    self.transfer_function(var_name, component, get, a, a_next, weight)
            )
        for var_name in transferfunctions_registered:
            setattr(self, var_name, construct_func(var_name))
        # Initialise the hdf5 file on disk, if it does not
        # already exist. If it exist, 'params' and 'k_magnitudes' are
        # guaranteed to be stored there correctly already, as the
        # filename depends on the content of 'params', which also
        # include 'k_magnitudes'.
        if master and not os.path.isfile(self.filename):
            self.save('params')
            self.save('k_magnitudes')
    # Method returning a classy.Class instance, hooked into a CLASS
    # session with parameters corresponding to self.params.
    # If CLASS has not yet been called, do this now.
    @property
    def cosmo(self):
        if self._cosmo is None:
            # No actual cosmo object exists.
            # Call CLASS using OpenMP.
            # If no perturbations should be computed, the master
            # process will have access to all results.
            # If perturbations should be computed, all node masters
            # will have access to their own k modes of
            # the perturbations. All other values will be available to
            # all node masters.
            if 'k_output_values' in self.params:
                # Compute perturbations. Do this in 'MPI' mode,
                # meaning utilizing all available nodes.
                self._cosmo, self.k_node_indices = call_class(
                    self.params,
                    sleep_time=(0.1, 1),
                    mode='MPI',
                    class_call_reason=self.class_call_reason,
                )
            else:
                # Do not compute perturbations. This call should be
                # very fast and so we compute it in 'single node'
                # mode regardless of the number of nodes available.
                # (Also, MPI Class is not implemented for anything but
                # the perturbation computation).
                self._cosmo = call_class(
                    self.params,
                    mode='single node',
                    class_call_reason=self.class_call_reason,
                )
        return self._cosmo
    # Methods returning scalar attributes used in the CLASS run
    @property
    def h(self):
        if not hasattr(self, '_h'):
            if not self.load('h'):
                # Get h from CLASS
                self._h = self.cosmo.h()
                # Save to disk
                self.save('h')
            # Communicate
            self._h = bcast(self._h if master else None)
        return self._h
    @property
    def Γ_dcdm(self):
        if not hasattr(self, '_Γ_dcdm'):
            # Extract directly from the CLASS parameters
            self._Γ_dcdm = float(self.params.get('Gamma_dcdm', 0))
            # Apply unit
            self._Γ_dcdm *= units.km/(units.s*units.Mpc)
            # Communicate
            self._Γ_dcdm = bcast(self._Γ_dcdm if master else None)
        return self._Γ_dcdm
    # The background
    @property
    def background(self):
        if not hasattr(self, '_background'):
            if not self.load('background'):
                # Get background from CLASS
                self._background = self.cosmo.get_background()
                # Let the master operate on the data
                if master:
                    # Add scale factor array
                    self._background['a'] = 1/(1 + self._background['z'])
                    # Only keep the needed background variables
                    self._background = {
                        key: arr for key, arr in self._background.items()
                        if any([key == pattern or re.search(pattern, key)
                            for pattern in self.needed_keys['background'] | class_extra_background
                        ])
                    }
                    # Remove data points prior to class_a_min.
                    # A copy of the truncated data is used,
                    # making freeing the original CLASS data possible.
                    for i, a in enumerate(self._background['a']):
                        if a > class_a_min:
                            if i == 0:
                                index = 0
                            else:
                                index = i - 1
                            break
                    self._background = {
                        key: arr[index:].copy()
                        for key, arr in self._background.items()
                    }
                # Save to disk
                self.save('background')
                # Now remove the extra CLASS background variables
                # not used by this simulation.
                if master:
                    self._background = {
                        key: arr for key, arr in self._background.items()
                        if any([key == pattern or re.search(pattern, key)
                            for pattern in self.needed_keys['background'] | class_extra_background
                        ])
                    }
                    # Throw a warning if background quantities specified
                    # in class_extra_background are not present.
                    missing_backgrounds = (
                        class_extra_background
                        - set(self._background.keys())
                        - missing_background_quantities
                    )
                    if missing_backgrounds:
                        missing_background_quantities.update(missing_backgrounds)
                        quantity_plural = (
                            'quantity' if len(missing_backgrounds) == 1 else 'quantities'
                        )
                        masterwarn(
                            f'Background {quantity_plural} {{}} not available from CLASS'
                            .format(', '.join([
                                f'"{missing_background}"'
                                for missing_background in missing_backgrounds
                            ]))
                        )
            # Communicate background as
            # dict mapping str's to arrays.
            size = bcast(len(self._background) if master else None)
            if size:
                keys = bcast(tuple(self._background.keys()) if master else None)
                if not master:
                    self._background = {}
                for key in keys:
                    buffer = smart_mpi(self._background[key] if master else (), mpifun='bcast')
                    if not master:
                        self._background[key] = asarray(buffer).copy()
            else:
                self._background = {}
            # CLASS does not give the background pressure for species
            # with constant equation of state w, and for some species
            # with time varying w this is given instead of the
            # background pressure. Here we add these missing pressures.
            def get_tot_contributing_class_species():
                for key in tuple(self._background.keys()):
                    match = re.search(r'^\(\.\)rho_(.+)', key)
                    if not match:
                        continue
                    class_species = match.group(1)
                    if class_species in ('crit', 'tot'):
                        continue
                    yield class_species
            for class_species in get_tot_contributing_class_species():
                if f'(.)p_{class_species}' in self._background:
                    continue
                w = self._background.get(f'(.)w_{class_species}')
                if w is None:
                    species_info = species_registered.get(
                        species_canonical.get(class_species, class_species)
                    )
                    if species_info is not None:
                        w = species_info.w
                if w is None:
                    masterwarn(
                        f'Could not determine background pressure for CLASS species '
                        f'"{class_species}". This will be set to zero. Has this species '
                        f'been registered with linear.register_species()?'
                    )
                    w = 0
                self._background[f'(.)p_{class_species}'] = (
                    w*self._background[f'(.)rho_{class_species}']
                )
            # We need the total background density and pressure
            if '(.)rho_tot' not in self._background:
                self._background['(.)rho_tot'] = 0
                self._background['(.)p_tot'] = 0
                for class_species in get_tot_contributing_class_species():
                    self._background['(.)rho_tot'] += self._background[f'(.)rho_{class_species}']
                    self._background['(.)p_tot']   += self._background[f'(.)p_{class_species}']
            # The special "metric" CLASS species needs to be assigned
            # some background density, but since we get δρ directly
            # from CLASS and neither δ nor ρ_bar has any
            # physical meaning, this background density can be
            # chosen arbitrarily. From this, a corresponding w_eff(a) is
            # automatically constructed. A normal equation of state w(a)
            # is also needed (though its actual value does not matter).
            # Thus, we also need to assign the metric species a
            # background pressure.
            # In principle we could use any background density and
            # pressure. However, it turns out that δρ(k, a) (at least on
            # large scales) approximately follows the behaviour of the
            # other species, which simply comes about because the metric
            # is built from the other species. We thus defien the
            # background density and pressure of the metric to be the
            # sum of background densities and pressures of all physical
            # species. To ensure a nice, positive density and pressure,
            # we exclude any species with negative density or pressure.
            def get_metric_contributing_class_species():
                for class_species in get_tot_contributing_class_species():
                    species_info = species_registered.get(
                        species_canonical.get(class_species, class_species)
                    )
                    if species_info is None or not species_info.is_physical:
                        continue
                    if np.any(asarray(self._background[f'(.)rho_{class_species}']) <= 0):
                        continue
                    if np.any(asarray(self._background[f'(.)p_{class_species}']) <= 0):
                        continue
                    yield class_species
            self._background['(.)rho_metric'] = 0
            self._background['(.)p_metric'] = 0
            for class_species in get_metric_contributing_class_species():
                self._background['(.)rho_metric'] += self._background[f'(.)rho_{class_species}']
                self._background['(.)p_metric']   += self._background[f'(.)p_{class_species}']
            # The special "lapse" CLASS species needs to be assigned
            # some fictitious background density and pressure,
            # just like the "metric". Here we reuse the values
            # assigned to the "metric".
            self._background['(.)rho_lapse'] = self._background['(.)rho_metric']
            self._background['(.)p_lapse']   = self._background['(.)p_metric']
            # Remove doppelgänger values in all background variables,
            # using the scale factor array as x values.
            for key, val in self._background.items():
                if key != 'a':
                    _, self._background[key] = remove_doppelgängers(
                        self._background['a'], val, copy=True)
            _, self._background['a'] = remove_doppelgängers(
                self._background['a'], self._background['a'], copy=True)
        return self._background
    # The raw perturbations
    @property
    def perturbations(self):
        if not hasattr(self, '_perturbations'):
            # Add species specific perturbation keys to the set
            # self.needed_keys['perturbations'], based on the
            # species present in the current simulation.
            class_species_present_list = (universals_dict['class_species_present']
                .decode().replace('[', r'\[').replace(']', r'\]').split('+'))
            for class_species_present in class_species_present_list:
                if not class_species_present:
                    continue
                if class_species_present == 'metric':
                    # For the special "metric" species, what we need is
                    # the metric potentials ϕ and ψ along with the
                    # conformal time derivative of H_T in N-body gauge.
                    self.needed_keys['perturbations'] |= {r'^phi$', r'^psi$', r'^H_T_prime$'}
                elif class_species_present == 'lapse':
                    # For the special "lapse" species, what we need is
                    # the conformal time derivative of H_T
                    # in N-body gauge.
                    self.needed_keys['perturbations'] |= {r'^H_T_prime$'}
                else:
                    self.needed_keys['perturbations'] |= {
                        # Density
                        rf'^delta_{class_species_present}$',
                        # Velocity
                        rf'^theta_{class_species_present}$',
                        # # Pressure
                        rf'^cs2_{class_species_present}$',
                        # Shear stress
                        rf'^shear_{class_species_present}$',
                    }
                    # For decaying cold dark matter we perform a
                    # transformation of θ, for which the conformal time
                    # derivative of H_T in N-body gauge is required.
                    if class_species_present == 'dcdm':
                        self.needed_keys['perturbations'] |= {r'^H_T_prime$'}
            if not self.load('perturbations'):
                # Get perturbations from CLASS
                self._perturbations = self.cosmo.get_perturbations()
                # The perturbation data is distributed on
                # the node masters. Let these operate on the data.
                Barrier()
                if node_master:
                    # Only scalar perturbations are used
                    self._perturbations = self._perturbations['scalar']
                    # Only keep the needed perturbations given in the
                    # self.needed_keys['perturbations'] set, as well as
                    # any additional perturbations defined in the user
                    # parameter class_extra_perturbations. These extra
                    # perturbations are not used directly, but will be
                    # dumped along with the rest to the disk. Only the
                    # node master processes will ever store these
                    # extra perturbations. A copy of the data is used,
                    # making freeing of the original
                    # CLASS data possible.
                    self._perturbations = [
                        {
                            key: arr.copy()
                            for key, arr in perturbation.items()
                            if any([key == pattern or re.search(pattern, key) for pattern in (
                                self.needed_keys['perturbations'] | class_extra_perturbations_class
                            )])
                         }
                         for perturbation in self._perturbations
                    ]
                    if len(self.k_magnitudes) > len(self.k_node_indices):
                        # The master process needs to know which
                        # process store which k modes.
                        if master:
                            k_processes_indices = empty(len(self.k_magnitudes),
                                dtype=C2np['Py_ssize_t'])
                            k_processes_indices[self.k_node_indices] = rank
                            for rank_recv in node_master_ranks:
                                if rank_recv == rank:
                                    continue
                                k_processes_indices[recv(source=rank_recv)] = rank_recv
                        else:
                            send(asarray(self.k_node_indices), dest=master_rank)
                        # Grab perturbation keys
                        keys = sorted(list(self._perturbations[0].keys()))
                        # If no k modes at all were delegated a given
                        # node, a fake k mode will be present. Having at
                        # least one k mode on all nodes simplifies the
                        # above logic, but now it is time to get rid of
                        # this additional k mode.
                        if len(self.k_node_indices) == 0:
                            for perturbation in self._perturbations:
                                for key in keys:
                                    perturbation[key].resize(0, refcheck=False)
                                    perturbation.pop(key)
                            self._perturbations = []
                        # Gather all perturbations into the
                        # master process. Communicate these as list
                        # of dicts mapping str's to arrays.
                        if master:
                            all_perturbations = [{} for k in self.k_magnitudes]
                            for k, perturbation in zip(self.k_node_indices, self._perturbations):
                                all_perturbations[k] = perturbation
                            for rank_recv, perturbation in zip(
                                k_processes_indices,
                                all_perturbations,
                            ):
                                if rank_recv == rank:
                                    continue
                                for key in keys:
                                    size = recv(source=rank_recv)
                                    buffer = get_buffer(size, 'perturbation')
                                    Recv(buffer, source=rank_recv)
                                    perturbation[key] = asarray(buffer).copy()
                            # The master process now holds perturbations
                            # from all nodes in all_perturbations.
                            self._perturbations = all_perturbations
                        else:
                            for perturbation in self._perturbations:
                                for key in keys:
                                    send(len(perturbation[key]), dest=master_rank)
                                    Send(perturbation[key], dest=master_rank)
                                    # Once the data has been
                                    # communicated, delete it from the
                                    # slave (node master) process.
                                    perturbation[key].resize(0, refcheck=False)
                                    perturbation.pop(key)
                # The master process now holds all perturbations
                # while the other node masters do not store any.
                # Throw a warning if perturbations specified in
                # class_extra_perturbations are not present.
                if master:
                    missing_perturbations = (
                        class_extra_perturbations_class - set(self._perturbations[0].keys())
                    )
                    if missing_perturbations:
                        masterwarn(
                            'Perturbations {} not available from CLASS'
                            .format(', '.join([
                                f'"{missing_perturbation}"'
                                for missing_perturbation in missing_perturbations
                            ]))
                        )
                # Save to disk
                self.save('perturbations')
                # As perturbations comprise the vast majority of the
                # data volume of what is needed from CLASS, we might
                # as well read in any remaining bits and clean up
                # the C-space memory and delete any extra CLASS
                # perturbations (which have now been saved to disk).
                self.load_everything('perturbations')
                self.cosmo.struct_cleanup()
                # Now remove the extra CLASS perturbations not used by
                # this simulation. If we are running the CLASS utility
                # and not a simulation, keep the
                # extra perturbations around.
                if master and special_params.get('special') != 'class':
                    for key in set(self._perturbations[0].keys()):
                        if not any([key == pattern or re.search(pattern, key)
                            for pattern in class_extra_perturbations_class]
                        ):
                            continue
                        if any([key == pattern or re.search(pattern, key)
                            for pattern in self.needed_keys['perturbations']]
                        ):
                            continue
                        for perturbation in self._perturbations:
                            perturbation[key].resize(0, refcheck=False)
                            perturbation.pop(key)
            # As we only need perturbations defined within the
            # simulation timespan, a >= a_begin, we now cut off the
            # lower tail of all perturbations.
            if master:
                def find_a_min(universals_a_begin):
                    for perturbation in self._perturbations:
                        a_values = perturbation['a']
                        # Find the index in a_values which corresponds to
                        # universals.a_begin, using a binary search.
                        index_lower = 0
                        index_upper = a_values.shape[0] - 1
                        a_lower = a_values[index_lower]
                        a_upper = a_values[index_upper]
                        if a_lower > universals_a_begin:
                            msg = (
                                f'Not all perturbations are defined at '
                                f'a_begin = {universals_a_begin}.'
                            )
                            if class_a_min > 0 and universals_a_begin < class_a_min:
                                msg += (
                                    f' Not all perturbations are defined at '
                                    f'a_begin = {universals_a_begin}. Note that CLASS '
                                    f'perturbations earlier than a_min = {class_a_min} in '
                                    f'source/perturbations.c will not be used. If you really want '
                                    f'perturbations at still earlier times, decrease this a_min '
                                    f'and recompile CLASS.'
                                )
                            elif universals_a_begin < universals.a_begin:
                                msg += (
                                    f' It may help to decrease the CLASS parameter '
                                    f'"perturb_integration_stepsize" and/or '
                                    f'"perturb_sampling_stepsize".'
                                )
                            abort(msg)
                        index, a_value = 0, -1
                        while index_upper - index_lower > 1 and a_value != universals_a_begin:
                            index = (index_lower + index_upper)//2
                            a_value = a_values[index]
                            if a_value > universals_a_begin:
                                index_upper = index
                            elif a_value < universals_a_begin:
                                index_lower = index
                        # Include times slightly earlier
                        # than absolutely needed.
                        index -= 3
                        if index < 0:
                            index = 0
                        yield index, a_values[index], perturbation
                # Find the minimum scale factor value
                # needed across all k modes.
                universals_a_begin_min = universals.a_begin
                for index, universals_a_begin, perturbation in find_a_min(universals_a_begin_min):
                    if universals_a_begin < universals_a_begin_min:
                        universals_a_begin_min = universals_a_begin
                # Remove perturbations earlier than
                # universals_a_begin_min. We have to copy the data,
                # as otherwise the array will not be owning the data,
                # meaning that it cannot be freed by Python's
                # garbage collection.
                for index, universals_a_begin, perturbation in find_a_min(universals_a_begin_min):
                    for key, val in perturbation.items():
                        perturbation[key] = asarray(val[index:]).copy()
            # The perturbations stored by the master process will now be
            # distributed among all processes, each storing part of the
            # total data. We could also give every process a copy of the
            # entire data set, but as it can take up several GB, this
            # can be a waste of memory. First the master process divides
            # the k modes fairly among the processes, so that the memory
            # burden is shared amongst all processes (and hence nodes).
            n_modes = bcast(len(self._perturbations) if master else None)
            if n_modes == self.k_magnitudes.size:
                keys = bcast(tuple(self._perturbations[0].keys()) if master else None)
                # Let the master divvy up the perturbations
                if master:
                    sizes = [np.sum([val.size for val in perturbation.values()])
                        for perturbation in self._perturbations]
                    indices = arange(n_modes, dtype=C2np['Py_ssize_t'])[np.argsort(sizes)]
                    n_surplus = n_modes % nprocs
                    indices_procs_deque = collections.deque(indices[n_surplus:])
                    indices_procs = [[] for _ in range(nprocs)]
                    while indices_procs_deque:
                        for method in ('pop', 'popleft'):
                            for indices_proc in indices_procs:
                                if indices_procs_deque:
                                    indices_proc.append(getattr(indices_procs_deque, method)())
                    for index, indices_proc in zip(indices[:n_surplus], reversed(indices_procs)):
                        indices_proc.append(index)
                    indices_procs = [asarray(sorted(indices), dtype=C2np['Py_ssize_t'])
                        for indices in indices_procs]
                    for rank_other, indices in enumerate(indices_procs):
                        if rank_other == rank:
                            continue
                        # Send the global perturbation indices
                        send(indices.size, dest=rank_other)
                        Send(indices, dest=rank_other)
                        # Send the perturbation data
                        for index in indices:
                            perturbation = self._perturbations[index]
                            for key in keys:
                                send(perturbation[key].size, dest=rank_other)
                                Send(perturbation[key], dest=rank_other)
                                # Once the data has been communicated,
                                # delete it from the master process.
                                perturbation[key].resize(0, refcheck=False)
                                perturbation.pop(key)
                    self.k_indices = indices_procs[rank]
                    self._perturbations = [self._perturbations[index]
                        for index in self.k_indices]
                else:
                    # Receive the global perturbation indices
                    self.k_indices = empty(recv(source=master_rank), dtype=C2np['Py_ssize_t'])
                    Recv(self.k_indices, source=master_rank)
                    # Receive the perturbation data
                    self._perturbations = [{} for _ in range(self.k_indices.size)]
                    for perturbation in self._perturbations:
                        for key in keys:
                            perturbation[key] = empty(
                                recv(source=master_rank),
                                dtype=C2np['double'],
                            )
                            Recv(perturbation[key], source=master_rank)
                Barrier()
                # All processes should be aware of the k indices of all
                # other processes. We have this as the list of arrays
                # indices_procs on the master process, but we now store
                # it as a single array. This array will give the
                # ordering of the k modes after a call to allgatherv
                # on the perturbation data.
                if master:
                    self.k_indices_all = np.argsort(np.concatenate(indices_procs))
                else:
                    self.k_indices_all = empty(
                        self.k_magnitudes.shape[0],
                        dtype=C2np['Py_ssize_t'],
                    )
                Bcast(self.k_indices_all)
            elif n_modes == 0:
                # No perturbations exist
                self._perturbations = []
            else:
                # A wrong number of perturbations exist
                abort(
                    f'Only {n_modes} of the expected {self.k_magnitudes.size} '
                    'perturbation k modes exist.'
                )
            # Now the perturbation data is fairly distributed amongst
            # all processes.
            # As perturbations comprise the vast majority of the
            # data volume of what is needed from CLASS, we might
            # as well read in any remaining bits. Specifically, the
            # background should be read, as some tasks around the
            # perturbations require knowledge of the background,
            # and the first read-in of the background has to be done
            # in parallel.
            self.load_everything('perturbations')
            # Represent each perturbation in self._perturbations as a
            # PerturbationDict object rather than a normal dict.
            for k_local, perturbation in enumerate(self._perturbations):
                self._perturbations[k_local] = self.PerturbationDict(perturbation)
            # After the CLASS perturbations needed for the special
            # "metric" and "lapse" species has been computed/loaded,
            # we need to manually construct the corresponding
            # δ perturbations out of these.
            if 'metric' in class_species_present_list:
                self.construct_delta_metric()
            if 'lapse' in class_species_present_list:
                self.construct_delta_lapse()
        return self._perturbations
    # Method which makes sure that everything is loaded
    def load_everything(self, already_loaded=None):
        """If some attribute is already loaded, it can be specified
        as the already_loaded argument. This is crucial to specify when
        called from within one of the methods matching an attribute.
        """
        attributes = {*self.attribute_names, 'background'}
        if 'k_output_values' in self.params:
            attributes.add('perturbations')
        if already_loaded:
            attributes -= set(any2list(already_loaded))
        # Importantly, we need to iterate over the attributes in some
        # definite order, ensuring synchronization between processes.
        for attribute in sorted(attributes):
            getattr(self, attribute)
    # Method which computes and adds "delta_metric" to the perturbations
    def construct_delta_metric(self):
        """This method adds the "delta_metric" perturbation
        to self._perturbations, assuming that the ϕ and ψ potentials and
        H_Tʹ in N-body gauge already exist as perturbations.
        The strategy is as follows: For each k, we can compute the GR
        correction potential γ(a) using
        γ(a) = -(H_Tʹʹ(a) + a*H(a)*H_Tʹ(a))/k² + (ϕ(a) - ψ(a)),
        where ʹ denotes differentiation with respect to
        conformal time τ. To get H_Tʹʹ (actually ∂ₐH_Tʹ, see below) we
        construct a TransferFunction object over the H_Tʹ perturbations.
        The units of the perturbations from CLASS are as follows:
        H_Tʹ: [time⁻¹]        = [c/Mpc],
        ϕ   : [length²time⁻²] = [c²],
        ψ   : [length²time⁻²] = [c²],
        and so γ also gets units of [length²time⁻²]. Note that H_T is
        some times defined to have units of [length²]. The H_T_prime
        from CLASS follows the unitless convention of
          https://arxiv.org/abs/1708.07769
        We choose to compute k²γ, not γ by itself.
        Using ʹ = d/dτ = a*d/dt = a²H(a)*d/da, we have
        k²γ(a) = -a*H(a)(a*∂ₐH_Tʹ(a) + H_Tʹ(a)) + k²(ϕ(a) - ψ(a))
        with ˙ = d/da. The δρ(a) perturbation is now given by
        δρ(a) = 2/3*γ(a)k²/a² * 3/(8πG)
              = k²γ(a)/(4πGa²)
        where the factor 3/(8πG) = 1 in CLASS units.
        Side-note: In this form (k²γ = 4πGa²δρ) it is clear that γ
        indeed is a potential. The missing sign stems from γ being
        defined with the opposite sign of usual potentials.
        Finally, since we want δ(a), we divide by the arbitrary but
        pre-defined background density ρ_metric:
        δ(a) = k²γ(a)/(4πGa²ρ_metric).
        The δ perturbations will be in N-body gauge, the only gauge in
        which these will contain all linear GR corrections,
        and therefore the only gauge of interest when it comes to the
        "metric" species. Also, the H_T_prime from CLASS is in
        N-body gauge. Whenever a transfer function in N-body gauge
        is needed, the compute_transfer function will carry out
        this conversion, assuming that the stored transfer function
        is in synchronous gauge. With the "metric" perturbations already
        in N-body gauge, this transformation should not be carried out.
        We cannot simply add a condition inside compute_transfer,
        as this cannot work for combined species which the "metric" is
        part of. We instead need to keep all transfer functions in
        synchronous gauge, meaning that we have to transform δ from
        N-body gauge to synchronous gauge. This transformation will then
        be exactly cancelled out in the compute_transfer function.
        """
        # Check that the delta_metric perturbations
        # has not already been added.
        if self._perturbations and 'delta_metric' in self._perturbations[0]:
            return
        masterprint('Constructing metric δ perturbations ...')
        # Get the H_Tʹ(k, a) transfer functions
        transfer_H_Tʹ = self.H_Tʹ(get='object')
        # Construct the "metric" δ(a) for each k
        for k_local, perturbation in enumerate(self._perturbations):
            k = self.k_indices[k_local]
            k_magnitude = self.k_magnitudes[k]
            # Extract needed perturbations along with
            # the scale factor at which they are tabulated.
            a     = perturbation['a'        ]
            ϕ     = perturbation['phi'      ]*ℝ[light_speed**2]
            ψ     = perturbation['psi'      ]*ℝ[light_speed**2]
            H_Tʹ  = perturbation['H_T_prime']*ℝ[light_speed/units.Mpc]
            θ_tot = perturbation['theta_tot']*ℝ[light_speed/units.Mpc]
            # Compute the derivative of H_Tʹ with respect to a
            dda_H_Tʹ = asarray([transfer_H_Tʹ.eval_deriv(k_local, a_i) for a_i in a])
            # Lastly, we need the Hubble parameter and the mean density
            # of the "metric" species at the times given by a.
            H = asarray([hubble(a_i) for a_i in a])
            ρ_metric = self.ρ_bar(a, 'metric')
            # Construct the γ potential
            aH = a*H
            k_magnitude2 = k_magnitude**2
            k2γ = -aH*(a*dda_H_Tʹ + H_Tʹ) + k_magnitude2*(ϕ - ψ)
            # Construct the δ perturbation (in N-body gauge)
            δ = k2γ/(ℝ[4*π*G_Newton]*a**2*ρ_metric)
            # Transform from N-body gauge to synchronous gauge
            w_metric = asarray([self.w(a_i, 'metric') for a_i in a])
            δ -= ℝ[3/light_speed**2]*aH*(1 + w_metric)*θ_tot/k_magnitude2
            # Store the "metric" δ perturbations,
            # now in synchronous gauge.
            perturbation['delta_metric'] = δ
        masterprint('done')
    # Method which computes and adds "delta_lapse" to the perturbations
    def construct_delta_lapse(self):
        """This method adds the "delta_lapse" perturbation
        to self._perturbations, assuming that H_Tʹ in N-body gauge
        already exist as a perturbation.
        The strategy is as follows: For each k, we can compute the GR
        correction potential γ_lapse(a) using
        γ_lapse(a) = -1/(3k²)*(H_Tʹʹ(a) + (a*H(a) - Hʹ(a)/H(a))*H_Tʹ(a)),
        where ʹ denotes differentiation with respect to
        conformal time τ. To get H_Tʹʹ (actually ∂ₐH_Tʹ, see below) we
        construct a TransferFunction object over the H_Tʹ perturbations.
        The units of this perturbation from CLASS is as follows:
        H_Tʹ: [time⁻¹] = [c/Mpc],
        and so γ_lapse gets units of [length²time⁻²]. Note that H_T is
        some times defined to have units of [length²]. The H_T_prime
        from CLASS follows the unitless convention of
          https://arxiv.org/abs/1708.07769
        Using ʹ = d/dτ = a*d/dt = a²H(a)*d/da, we have
        k²γ_lapse(a) = -a/3*(a*H(a)*∂ₐH_Tʹ(a) + (H(a) - Ḣ(a)/H(a))*H_Tʹ(a))
        with ˙ = ∂ₜ. The δρ(a) perturbation is now given by
        δρ(a) = 2/3*k²γ_lapse(a)/a² * 3/(8πG)
              = k²γ_lapse(a)/(4πGa²)
        where the factor 3/(8πG) = 1 in CLASS units.
        Note that the same convention is used here as for the metric
        (not lapse) γ.
        The H_T_prime from CLASS is in N-body gauge, and so the δ
        perturbations will likewise be in N-body gauge. Whenever a
        transfer function in N-body gauge is needed,
        the compute_transfer function will carry out this conversion,
        assuming that the stored transfer function is in synchronous
        gauge. With the "lapse" perturbations already in N-body gauge,
        this transformation should not be carried out. We cannot simply
        add a condition inside compute_transfer, as this cannot work for
        combined species which the "lapse" is part of. We instead need
        to keep all transfer functions in synchronous gauge, meaning
        that we have to transform δ from N-body gauge to synchronous
        gauge. This transformation will then be exactly cancelled out in
        the compute_transfer function.
        """
        # Check that the delta_lapse perturbations
        # has not already been added.
        if self._perturbations and 'delta_lapse' in self._perturbations[0]:
            return
        masterprint('Constructing lapse δ perturbations ...')
        # Get the H_Tʹ(k, a) transfer functions
        transfer_H_Tʹ = self.H_Tʹ(get='object')
        # Construct the "lapse" δ(a) for each k
        for k_local, perturbation in enumerate(self._perturbations):
            k = self.k_indices[k_local]
            k_magnitude = self.k_magnitudes[k]
            # Extract needed perturbations along with
            # the scale factor at which they are tabulated.
            a     = perturbation['a'        ]
            H_Tʹ  = perturbation['H_T_prime']*ℝ[light_speed/units.Mpc]
            θ_tot = perturbation['theta_tot']*ℝ[light_speed/units.Mpc]
            # Compute the derivative of H_Tʹ with respect to a
            dda_H_Tʹ = asarray([transfer_H_Tʹ.eval_deriv(k_local, a_i) for a_i in a])
            # Lastly, we need the Hubble parameter, its cosmic time
            # derivative and the mean density of the "lapse" species at
            # the times given by a.
            H = asarray([hubble(a_i) for a_i in a])
            ddt_H = asarray([Ḣ(a_i) for a_i in a])
            ρ_lapse = self.ρ_bar(a, 'lapse')
            # Construct the γ_lapse potential
            aH = a*H
            k_magnitude2 = k_magnitude**2
            k2γ_lapse = -1./3.*a*(aH*dda_H_Tʹ + (H - ddt_H/H)*H_Tʹ)
            # Construct the δ perturbation (in N-body gauge)
            δ = k2γ_lapse/(ℝ[4*π*G_Newton]*a**2*ρ_lapse)
            # Transform from N-body gauge to synchronous gauge
            w_lapse = asarray([self.w(a_i, 'lapse') for a_i in a])
            δ -= ℝ[3/light_speed**2]*aH*(1 + w_lapse)*θ_tot/k_magnitude2
            # Store the "lapse" δ perturbations,
            # now in synchronous gauge.
            perturbation['delta_lapse'] = δ
        masterprint('done')
    # Method which constructs TransferFunction instances and use them
    # to compute and store transfer functions. Do not use this
    # method directly, but rather
    # call e.g. cosmoresults.δ(a, component=component).
    def transfer_function(self, var_name,
        component=None, get='object', a=-1, a_next=-1, weight=None,
    ):
        if weight in ('1', 1):
            weight = None
        if weight and get not in ('as_function_of_k', 'deriv_as_function_of_k'):
            abort(
                f'A weight was supplied to transfer_function() while get="{get}", '
                f'but get should be either "as_function_of_k" or "deriv_as_function_of_k" '
                f'when using a weight'
            )
        if not hasattr(self, '_transfer_functions'):
            self._transfer_functions = {}
        key = (component.class_species if component is not None else None, var_name)
        transfer_function = self._transfer_functions.get(key)
        if transfer_function is None:
            transfer_function = TransferFunction(self, component, var_name)
            self._transfer_functions[key] = transfer_function
        # Depending on the value of get, return either the
        # TransferFunction instance, an array of evaluated transfer
        # function values as function of k, or an array of evaluated
        # transfer function derivatives as function of k.
        # as function of k.
        if get == 'object':
            return transfer_function
        elif get == 'as_function_of_k':
            return transfer_function.as_function_of_k(a, a_next, weight)
        elif get == 'deriv_as_function_of_k':
            return transfer_function.deriv_as_function_of_k(a)
        else:
            abort(
                f'The transfer_function method was called with get = "{get}", '
                f'which is not implemented'
            )
    # Method for constructing splines of background variables
    # as function of a.
    def splines(self, y):
        if not hasattr(self, '_splines'):
            self._splines = {}
        spline = self._splines.get(y)
        if spline is None:
            # By far the most background variables are power laws in a.
            # A few exceptions are the constant pressure of the cdm, b
            # and lambda CLASS species, as well as the density, pressure
            # and equation of state w for the fld CLASS species.
            match = re.search(r'^\(\.\)(rho|p|w)_(.+)$', y)
            if y in {
                'z',
                'a',
                'H [1/Mpc]',
                'proper time [Gyr]',
                'conf. time [Mpc]',
                'gr.fac. D',
                'gr.fac. f',
            }:
                logx, logy = True, True
            elif y in {
                '(.)p_tot',
            }:
                logx = True
                logy = (not np.any(asarray(self.background[y]) <= 0))
            elif match:
                quantity = match.group(1)
                class_species = match.group(2)
                species_info = species_registered.get(
                    species_canonical.get(class_species, class_species)
                )
                if species_info is None:
                    # ρ, p or w from some non-registered species.
                    # Assume power law.
                    logx, logy = True, True
                else:
                    # ρ, p or w from registered species. Look up.
                    if quantity == 'w':
                        logx_ρ, logy_ρ = species_info.logs['rho']
                        logx_p, logy_p = species_info.logs['p']
                        logx = (logx_ρ or logx_p)
                        logy = (logy_ρ or logy_p)
                    else:
                        logx, logy = species_info.logs[quantity]
                    # If not specified, assume power law
                    if logx is None:
                        logx = True
                    if logy is None:
                        logy = True
            if logx is None or logy is None:
                logx = True
                logy = (not np.any(asarray(self.background[y]) <= 0))
                masterwarn(
                    f'A spline over the unknown CLASS background variable "{y}"(a) '
                    f'has been made with logx = {logx}, logy = {logy}. '
                    f'You should add the correct linear/log behaviour of this variable '
                    f'to the splines() method of the CosmoResults class.'
                )
            spline = Spline(self.background['a'], self.background[y], f'{y}(a)',
                logx=logx, logy=logy)
            self._splines[y] = spline
        return spline
    # Method for looking up the background density of a given
    # component/species at some specific a. If no component/species
    # is given, the critical density is returned.
    def ρ_bar(self, a, component_or_class_species='crit', apply_unit=True):
        if isinstance(component_or_class_species, str):
            class_species = component_or_class_species
        else:
            class_species = component_or_class_species.class_species
        values = 0
        for class_species in class_species.split('+'):
            spline = self.splines(f'(.)rho_{class_species}')
            # The input a may be either a scalar or an array
            with unswitch:
                if isinstance(a, (int, float)):
                    values += spline.eval(a)
                else:
                    values += asarray([spline.eval(a_i) for a_i in a])
        # Apply unit
        if apply_unit:
            values *= ℝ[3/(8*π*G_Newton)*(light_speed/units.Mpc)**2]
        return values
    # Method for looking up the background pressure of a given
    # component/species at some specific a. A component/species
    # has to be given.
    def P_bar(self, a, component_or_class_species, apply_unit=True):
        if isinstance(component_or_class_species, str):
            class_species = component_or_class_species
        else:
            class_species = component_or_class_species.class_species
        values = 0
        for class_species in class_species.split('+'):
            spline = self.splines(f'(.)p_{class_species}')
            # The input a may be either a scalar or an array
            with unswitch:
                if isinstance(a, (int, float)):
                    values += spline.eval(a)
                else:
                    values += asarray([spline.eval(a_i) for a_i in a])
        # Apply unit. Note that we define P_bar such that
        # w = c⁻²P_bar/ρ_bar.
        if apply_unit:
            values *= ℝ[3/(8*π*G_Newton)*(light_speed/units.Mpc)**2*light_speed**2]
        return values
    # Method for looking up the equation of state parameter w
    # of a given component/species at some specific a.
    def w(self, a, component_or_class_species):
        if isinstance(component_or_class_species, str):
            class_species = component_or_class_species
        else:
            class_species = component_or_class_species.class_species
        ρ_bar = P_bar = 0
        for class_species in class_species.split('+'):
            ρ_bar_spline = self.splines(f'(.)rho_{class_species}')
            P_bar_spline = self.splines(f'(.)p_{class_species}')
            # The input a may be either a scalar or an array
            with unswitch:
                if isinstance(a, (int, float)):
                    ρ_bar += ρ_bar_spline.eval(a)
                    P_bar += P_bar_spline.eval(a)
                else:
                    ρ_bar += asarray([ρ_bar_spline.eval(a_i) for a_i in a])
                    P_bar += asarray([P_bar_spline.eval(a_i) for a_i in a])
        # As we have done no unit conversion, the ratio P_bar/ρ_bar
        # gives us the unitless w.
        return P_bar/ρ_bar
    # Method for looking up the linear growth rate f_growth = H⁻¹Ḋ/D
    # (with D the linear growth factor) at some a.
    @lru_cache()
    def growth_fac_f(self, a):
        spline = self.splines('gr.fac. f')
        return spline.eval(a)
    # Method for appending a piece of raw CLASS data to the dump file
    def save(self, element):
        """You should not call this method unless you have good reason
        to believe that 'element' is not already present in the file,
        as this method will open the file in read/write ('a') mode
        regardless. This can be dangerous as HDF5 build with MPI is not
        thread-safe, and so if two running instances of CO𝘕CEPT with the
        same params run this method simultaneously, problems
        may arise. From HDF5 1.10 / H5Py 2.5.0, multiple processes can
        read from the same file, as long as it is not opened in write
        mode by any process. Thus, this complication is only relevant
        for this method. The open_hdf5 function is meant to alleviate
        this problem, but it has not been thoroughly tested.
        Note that we save regardless of the value of class_reuse.
        """
        # Do not save anything if a filename was passed,
        # in which case id is None.
        if self.id is None:
            return
        # The master process will save the given element to the file
        # given by self.filename. Importantly, the element in question
        # should be fully defined on the master process
        # before calling this method.
        if not master:
            return
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open_hdf5(self.filename, mode='a') as hdf5_file:
            # CLASS parameters as attributes on a group.
            # This should be the first element to be saved.
            if element == 'params':
                if 'params' not in hdf5_file:
                    params_h5 = hdf5_file.create_group('params')
                    for key, val in self.params.items():
                        key = key.replace('/', '__per__')
                        params_h5.attrs[key] = val
                    hdf5_file.flush()
                # Done saving to disk
                return
            # Start by checking that the params in the file match
            # those of this CosmoResults object.
            for key, val in hdf5_file['params'].attrs.items():
                key = key.replace('__per__', '/')
                if val != self.params.get(key):
                    abort(f'The CLASS dump {self.filename} contain unexpected parameter values')
            # Save the passed element
            if element in self.attribute_names:
                # Scalar attribute as attribute on the background group
                attribute = getattr(self, element)
                background_h5 = hdf5_file.require_group('background')
                background_h5.attrs[element.replace('/', '__per__')] = attribute
            elif element == 'k_magnitudes':
                # Save k_magnitudes in CLASS units (Mpc⁻¹)
                # as a dataset on the perturbations group.
                if self.k_magnitudes is not None and 'perturbations/k_magnitudes' not in hdf5_file:
                    perturbations_h5 = hdf5_file.require_group('perturbations')
                    dset = perturbations_h5.create_dataset(
                        'k_magnitudes',
                        (self.k_magnitudes.shape[0], ),
                        dtype=C2np['double'],
                    )
                    dset[:] = asarray(self.k_magnitudes)/units.Mpc**(-1)
            elif element == 'background':
                # Background arrays as data sets
                # in the 'background' group.
                background_h5 = hdf5_file.require_group('background')
                for key, val in self.background.items():
                    key = key.replace('/', '__per__')
                    if key not in background_h5:
                        dset = background_h5.create_dataset(key, (val.shape[0], ),
                                                            dtype=C2np['double'])
                        dset[:] = val
            elif element == 'perturbations':
                # Save perturbations as /perturbations/index/key
                perturbations_h5 = hdf5_file.require_group('perturbations')
                # Check whether all keys are already present in the file
                perturbations_to_store = set(self.perturbations[0].keys())
                if '0' in perturbations_h5:
                    perturbations_to_store -= {
                        key.replace('__per__', '/') for key in perturbations_h5['0'].keys()
                    }
                if perturbations_to_store:
                    # Store perturbations
                    masterprint(f'Saving CLASS perturbations to "{self.filename}" ...')
                    for index, perturbation in enumerate(self.perturbations):
                        perturbation_h5 = perturbations_h5.require_group(str(index))
                        for key in perturbations_to_store:
                            val = perturbation[key]
                            dset = perturbation_h5.create_dataset(
                                key.replace('/', '__per__'),
                                (val.shape[0], ),
                                dtype=C2np['double'],
                            )
                            dset[:] = val
                    masterprint('done')
            else:
                abort(f'CosmoResults.save was called with the unknown element of "{element}"')
            hdf5_file.flush()
    # Method for loading a piece of raw CLASS data from the dump file
    def load(self, element):
        """This method will attempt to load the element given.
        If successful, the element will be set on the instance and True
        will be returned by all processes.
        Otherwise, False will be returned by all processes.
        """
        if not class_reuse:
            return False
        if not master:
            return bcast()
        if not os.path.isfile(self.filename):
            return bcast(False)
        # The master process attempts to load the given element
        # from the file given by self.filename.
        with open_hdf5(self.filename, mode='r') as hdf5_file:
            # Start by checking that the params in the file match
            # those of this CosmoResults object. If a filename was
            # passed explicitly, this check is skipped.
            if self.id is not None:
                for key, val in hdf5_file['params'].attrs.items():
                    key = key.replace('__per__', '/')
                    if val != self.params.get(key):
                        abort(f'The CLASS dump {self.filename} contains'
                              ' unexpected parameter values')
            # Load the passed element
            if element in self.attribute_names:
                # Scalar attribute as attribute on the background group
                background_h5 = hdf5_file.get('background')
                if background_h5 is None:
                    return bcast(False)
                attribute = background_h5.attrs.get(element.replace('/', '__per__'))
                if attribute is None:
                    return bcast(False)
                setattr(self, '_' + element, attribute)
            elif element == 'k_magnitudes':
                # Load k_magnitudes as a dataset
                # on the perturbations group.
                # Remember to add CLASS units (Mpc⁻¹).
                perturbations_h5 = hdf5_file.get('perturbations')
                if perturbations_h5 is None:
                    return bcast(False)
                k_magnitudes_h5 = perturbations_h5.get('k_magnitudes')
                if k_magnitudes_h5 is None:
                    return bcast(False)
                self.k_magnitudes = k_magnitudes_h5[...]*units.Mpc**(-1)
            elif element == 'background':
                # Background arrays as data sets
                # in the 'background' group.
                background_h5 = hdf5_file.get('background')
                if background_h5 is None:
                    return bcast(False)
                self._background = {
                    key.replace('__per__', '/'): dset[...]
                    for key, dset in background_h5.items()
                    if any([key.replace('__per__', '/') == pattern
                        or re.search(pattern, key.replace('__per__', '/'))
                        for pattern in self.needed_keys['background'] | class_extra_background
                    ])
                }
                # Check that all background quantities in
                # class_extra_background were present in the file.
                background_loaded = set(self._background.keys())
                backgrounds_missing = {background_missing
                    for background_missing in class_extra_background
                    if not any([key == background_missing or re.search(background_missing, key)
                        for key in background_loaded])
                }
                if backgrounds_missing:
                    # One or more background quantities are missing.
                    # CLASS should be rerun.
                    return bcast(False)
            elif element == 'perturbations':
                # Load perturbations stored as
                # /perturbations/index/name.
                perturbations_h5 = hdf5_file.get('perturbations')
                if perturbations_h5 is None:
                    return bcast(False)
                n_modes = len(perturbations_h5)
                if 'k_magnitudes' in perturbations_h5:
                    n_modes -= 1
                if n_modes == 0:
                    return bcast(False)
                masterprint(f'Loading CLASS perturbations from "{self.filename}" ...')
                self._perturbations = [None]*len(self.k_magnitudes)
                # Check that the file contain perturbations at all
                # k modes. This is not the case if the process that
                # originally wrote the file ended prematurely. In this
                # case, no other error is necessarily detected.
                if n_modes < len(self._perturbations):
                    abort(
                        f'The file "{self.filename}" only contains perturbations for {n_modes} '
                        f'k modes, whereas it should contain perturbations for '
                        f'{len(self._perturbations)} k modes. This can happen if the creation of '
                        f'this file was ended prematurely. You should remove this file and rerun '
                        f'this simulation.'
                    )
                if n_modes > len(self._perturbations):
                    abort(
                        f'The file "{self.filename}" contains perturbations for {n_modes} '
                        f'k modes, whereas it should contain perturbations for '
                        f'{len(self._perturbations)} k modes. I cannot explain this mismatch, and '
                        f'I cannot use these perturbations.'
                    )
                # Load the perturbations
                needed_keys = self.needed_keys['perturbations'].copy()
                if special_params.get('special') == 'class':
                    needed_keys |= class_extra_perturbations_class
                for key, d in perturbations_h5.items():
                    if key == 'k_magnitudes':
                        continue
                    self._perturbations[int(key)] = {
                        key.replace('__per__', '/'): dset[...]
                        for key, dset in d.items()
                        if any([key.replace('__per__', '/') == pattern
                            or re.search(pattern, key.replace('__per__', '/'))
                            for pattern in needed_keys
                        ])
                    }
                masterprint('done')
                # Check that all needed perturbations were present
                # in the file. Some of the species specific
                # perturbations does not exist for all species
                # (e.g. "cs2" does not exist for photons). Therefore,
                # species specific perturbations are only considered
                # missing if "delta" is missing.
                perturbations_loaded = set(self.perturbations[0].keys())
                perturbations_missing = {perturbation_missing
                    for perturbation_missing in needed_keys
                    if not any([key == perturbation_missing or re.search(perturbation_missing, key)
                        for key in perturbations_loaded])
                }
                for class_species_present in (universals_dict['class_species_present']
                    .decode().replace('[', r'\[').replace(']', r'\]').split('+')):
                    perturbations_missing -= {
                        rf'^theta_{class_species_present}$',
                        rf'^cs2_{class_species_present}$',
                        rf'^shear_{class_species_present}$',
                    }
                if perturbations_missing:
                    masterprint(
                        'Not all needed perturbations were present in the file. '
                        'CLASS will be rerun.'
                    )
                    return bcast(False)
            else:
                abort(f'CosmoResults.load was called with the unknown element of "{element}"')
        # Loading of specified element completed successfully
        return bcast(True)

# Class for processing and storing transfer functions of k and a.
# The processing consists purely of data cleanup and interpolations.
# No gauge transformation etc. will be carried out.
@cython.cclass
class TransferFunction:
    # Initialisation method
    @cython.header(# Arguments
                   cosmoresults=object,  # CosmoResults
                   component='Component',
                   var_name=str,
                   )
    def __init__(self, cosmoresults, component, var_name):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the TransferFunction type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        object cosmoresults
        Component component
        str var_name
        str class_species
        double[::1] k_magnitudes
        Py_ssize_t k_gridsize
        Py_ssize_t k_gridsize_local
        Py_ssize_t[::1] k_indices
        Py_ssize_t[::1] k_indices_all
        double[::1] data
        double[::1] data_local
        double[::1] data_deriv
        double[::1] data_deriv_local
        int n_intervals
        double k_max
        double[:, ::1] factors
        double[:, ::1] exponents
        object splines  # np.ndarray of dtype object
        list a_values
        list interval_boarders
        """
        # Ensure that the cosmological perturbations has been loaded
        cosmoresults.perturbations
        # Store instance data
        self.cosmoresults = cosmoresults
        self.component = component
        self.var_name = var_name
        if self.var_name not in transferfunctions_registered:
            abort(f'Transfer function "{self.var_name}" not implemented')
        # The species (CLASS convention) of which to compute
        # transfer functions. If component is None, set the CLASS
        # species to 'tot', as this "species" do not correspond
        # to any component.
        if self.component is None:
            self.class_species = 'tot'
        else:
            self.class_species = self.component.class_species
        # The k values at which the transfer function
        # is tabulated by CLASS. Note that only the modes given by
        # self.k_indices are stored by this process.
        self.k_magnitudes = self.cosmoresults.k_magnitudes
        self.k_gridsize = self.k_magnitudes.shape[0]
        self.k_indices = self.cosmoresults.k_indices
        self.k_gridsize_local = self.k_indices.shape[0]
        self.k_indices_all = self.cosmoresults.k_indices_all
        # These will become arrays storing the transfer function and its
        # derivative with respect to the scale factor,
        # at a given k and as a function of a.
        self.data = self.data_local = self.data_deriv = self.data_deriv_local = None
        # (Maximum) number of intervals into which the scale factor
        # values should be subdivided when making the detrended splines.
        # From the manner in which interval_boarders are created in the
        # process method, one can show that the following is the maximum
        # possible number of intervals needed.
        self.n_intervals = 2*len(find_critical_times()) + 4
        # Construct splines of the transfer function as a function of a,
        # for the k modes stored by this process.
        self.k_max = class_k_max.get('all', ထ)
        self.factors   = empty((self.k_gridsize_local, self.n_intervals), dtype=C2np['double'])
        self.exponents = empty((self.k_gridsize_local, self.n_intervals), dtype=C2np['double'])
        self.splines   = empty((self.k_gridsize_local, self.n_intervals), dtype=object)
        self.a_values          = [None]*self.k_gridsize_local
        self.interval_boarders = [None]*self.k_gridsize_local
        self.process()

    # Method for processing the transfer function data from CLASS.
    # The end result is the population self.splines, self.factors
    # and self.exponents.
    @cython.header(
        # Locals
        a_values='double[::1]',
        a_values_largest_trusted_k=object,  # np.ndarray of dtype object
        any_contain_untrusted_perturbations='bint',
        approximate_P_as_wρ='bint',
        available='bint',
        class_perturbation_name=str,
        class_species=str,
        class_units='double',
        crossover='int',
        contains_untrusted_perturbations='bint',
        exponent='double',
        factor='double',
        i='Py_ssize_t',
        index='Py_ssize_t',
        index_left='Py_ssize_t',
        index_right='Py_ssize_t',
        interval_a_values='double[::1]',
        interval_perturbation_values='double[::1]',
        k='Py_ssize_t',
        k_local='Py_ssize_t',
        k_max_candidate='double',
        key=str,
        largest_trusted_k='Py_ssize_t',
        missing_perturbations_warning=str,
        n_outliers='Py_ssize_t',
        outlier='Py_ssize_t',
        outliers='Py_ssize_t[::1]',
        outliers_first='Py_ssize_t',
        outliers_last='Py_ssize_t',
        outliers_list=list,
        perturbation=object,  # np.ndarray or double
        perturbation_k=object,  # PerturbationDict
        perturbation_key=str,
        perturbation_keys=set,
        perturbation_values='double[::1]',
        perturbation_values_arr=object,  # np.ndarray
        perturbation_values_auxiliary='double[::1]',
        perturbations_available=dict,
        perturbations_detrended='double[::1]',
        perturbations_detrended_largest_trusted_k=object,  # np.ndarray of dtype object
        rank_largest_trusted_k='int',
        rank_other='int',
        size='Py_ssize_t',
        spline='Spline',
        transferfunction_info=object,  # TransferFunctionInfo
        trend=object,  # np.ndarray
        untrusted_perturbations=object,  # np.ndarray of dtype object
        weights=object,  # np.ndarray
        weights_species=dict,
        Σweights_inv=object,  # np.ndarray
    )
    def process(self):
        # Ensure that the cosmological background has been loaded
        self.cosmoresults.background
        # The processing of the perturbation / transfer function
        # depends on its type. Information about each implemented
        # transfer function is available in the global
        # transferfunctions_registered dict.
        transferfunction_info = transferfunctions_registered[self.var_name]
        class_perturbation_name = transferfunction_info.name_class
        class_units = transferfunction_info.units_class
        # Display progress message
        if self.component is None:
            if transferfunction_info.total:
                masterprint(f'Processing {self.var_name} transfer functions ...')
            else:
                masterprint(f'Processing total {self.var_name} transfer functions ...')
        else:
            masterprint(
                f'Processing {self.var_name} transfer functions '
                f'for {self.component.name} ...'
            )
        missing_perturbations_warning = ''.join([
            'The {} perturbations ',
            (f'(needed for the {self.component.name} component) '
                if self.component is not None else ''),
            'are not available'
        ])
        perturbations_available = {
            class_species: True for class_species in self.class_species.split('+')
        }
        approximate_P_as_wρ = (self.var_name == 'δP' and self.component.approximations['P=wρ'])
        # Update self.k_max dependent on the CLASS species
        perturbation_keys = set()
        for class_species in self.class_species.split('+'):
            perturbation_keys.add(class_perturbation_name.format(class_species))
        if self.var_name == 'δP':
            for class_species in self.class_species.split('+'):
                perturbation_keys.add(f'delta_{class_species}')
                if not approximate_P_as_wρ:
                    perturbation_keys.add(f'cs2_{class_species}')
        for perturbation_key in perturbation_keys:
            for key, k_max_candidate in class_k_max.items():
                if k_max_candidate < self.k_max:
                    if perturbation_key == key:
                        self.k_max = k_max_candidate
                    elif re.search(perturbation_key, key):
                        self.k_max = k_max_candidate
        # Number of additional points on each side of the interval
        # to include when doing the detrending and splining.
        crossover = 3
        # Splines should be constructed for each local k value
        largest_trusted_k = -1
        untrusted_perturbations = empty(self.k_gridsize_local, dtype=object)
        a_values_largest_trusted_k = empty(self.n_intervals, dtype=object)
        perturbations_detrended_largest_trusted_k = empty(self.n_intervals, dtype=object)
        for k_local, perturbation_k in enumerate(self.cosmoresults.perturbations):
            # The perturbation_k dict store perturbation arrays for
            # all perturbation types and CLASS species, defined at
            # times matching those of a_values.
            # The global k index corresponding to the local k index.
            k = self.k_indices[k_local]
            # Array of scale factor values at which perturbations for
            # this k mode is tabulated.
            a_values = perturbation_k['a'].copy()
            # Because a single CO𝘕CEPT species can map to multiple
            # CLASS species, we need to construct an array of
            # perturbation values as a weighted sum of perturbations
            # over the individual ('+'-separated) CLASS species.
            # These weights are constructed below.
            with unswitch:
                if 𝕊[transferfunction_info.weighting] == '1':
                    weights_species = {
                        class_species: 1
                        for class_species in self.class_species.split('+')
                    }
                elif 𝕊[transferfunction_info.weighting] == 'ρ':
                    weights_species = {
                        class_species: self.cosmoresults.ρ_bar(a_values, class_species)
                        for class_species in self.class_species.split('+')
                    }
                    Σweights_inv = 1/np.sum(tuple(weights_species.values()), axis=0)
                    for class_species in weights_species:
                        weights_species[class_species] *= Σweights_inv
                elif 𝕊[transferfunction_info.weighting] == 'ρ+P':
                    weights_species = {
                        class_species: (
                            self.cosmoresults.ρ_bar(a_values, class_species)
                            + ℝ[light_speed**(-2)]*self.cosmoresults.P_bar(a_values, class_species)
                        )
                        for class_species in self.class_species.split('+')
                    }
                    Σweights_inv = 1/np.sum(tuple(weights_species.values()), axis=0)
                    for class_species in weights_species:
                        weights_species[class_species] *= Σweights_inv
                elif 𝕊[transferfunction_info.weighting] == 'δρ':
                    weights_species = {
                        class_species: (
                            perturbation_k.get(f'delta_{class_species}')
                            *self.cosmoresults.ρ_bar(a_values, class_species)
                        )
                        for class_species in self.class_species.split('+')
                    }
                else:
                    abort(
                        f'Transfer function weighting "{transferfunction_info.weighting}" '
                        f'not implemented.'
                    )
                    weights_species = {}  # To satisfy the compiler
            # Construct the perturbation_values array from the CLASS
            # perturbations, units and weights.
            perturbation_values_arr = 0
            if approximate_P_as_wρ:
                # We are working on the δP transfer function and
                # the P=wρ approximation is enabled.
                # This means that δP/δρ = c²w.
                # The c² will be provided by class_unit.
                for class_species, weights in weights_species.items():
                    perturbation = asarray(
                        [self.component.w(a=a_value) for a_value in a_values],
                        dtype=C2np['double'],
                    )
                    perturbation_values_arr += weights*class_units*perturbation
            else:
                # We are working on a normal transfer function
                for class_species, weights in weights_species.items():
                    perturbation = perturbation_k.get(
                        class_perturbation_name.format(class_species))
                    if perturbation is None:
                        perturbations_available[class_species] = False
                    else:
                        perturbation_values_arr += weights*class_units*perturbation
            if isinstance(perturbation_values_arr, int):
                perturbation_values = asarray((), dtype=C2np['double'])
            else:
                perturbation_values = perturbation_values_arr
            # Warn or abort on missing perturbations.
            # We only do this for the first k mode
            # on the master process.
            if not approximate_P_as_wρ:
                if k_local == 0 and not all(perturbations_available.values()):
                    if len(perturbations_available) == 1:
                        abort(
                            missing_perturbations_warning
                            .format(class_perturbation_name)
                            .format(self.class_species)
                        )
                    for class_species, available in perturbations_available.items():
                        if not available:
                            masterwarn(missing_perturbations_warning
                                .format(class_perturbation_name)
                                .format(class_species)
                            )
                    if not any(perturbations_available.values()):
                        abort(
                            f'No {class_perturbation_name.format(class_species)} perturbations '
                            + ('' if self.component is None
                                else f'for the {self.component.name} component ')
                            + f'available'
                        )
            # Perform outlier rejection
            outliers_list = []
            if self.var_name == 'δP':
                # Look for outlier points which are outside the
                # legal range 0 ≤ δP/δρ ≤ c²/3. As the data is
                # directly from CLASS, c = 1.
                for class_species in weights_species:
                    if (
                        class_species not in {'g', 'ur', 'dr'}
                        and not class_species.startswith('ncdm[')
                    ):
                        continue
                    perturbation = perturbation_k.get(f'cs2_{class_species}')
                    if perturbation is not None:
                        perturbation_values_auxiliary = perturbation
                        for i in range(perturbation_values.shape[0]):
                            if not (0 <= perturbation_values_auxiliary[i] <= 1./3.):
                                outliers_list.append(i)
            if outliers_list:
                # We want to keep the points at both ends of a_values,
                # even if they are classified as outliers. In fact we
                # keep all outlier points to the left of the first
                # non-outlier, as well as all outliers to the right of
                # the last non-outlier.
                outliers = np.unique(asarray(outliers_list, dtype=C2np['Py_ssize_t']))
                outliers_first = ℤ[outliers.shape[0]]
                for i in range(ℤ[outliers.shape[0]]):
                    if outliers[i] != i:
                        outliers_first = i
                        break
                outliers_last = 0
                for i in range(ℤ[outliers.shape[0]]):
                    if outliers[ℤ[ℤ[outliers.shape[0] - 1] - i]] != ℤ[a_values.shape[0] - 1] - i:
                        outliers_last = ℤ[ℤ[outliers.shape[0] - 1] - i]
                        break
                outliers = asarray(outliers)[outliers_first:outliers_last+1]
                # Now do the removal
                n_outliers = 0
                outlier = outliers[n_outliers]
                for i in range(perturbation_values.shape[0]):
                    if i == outlier:
                        n_outliers += 1
                        if n_outliers < outliers.shape[0]:
                            outlier = outliers[n_outliers]
                    elif n_outliers:
                        index = i - n_outliers
                        a_values           [index] = a_values           [i]
                        perturbation_values[index] = perturbation_values[i]
                size = a_values.shape[0] - n_outliers
                a_values            = a_values           [:size]
                perturbation_values = perturbation_values[:size]
            # The CLASS perturbations sometime contain neighbouring
            # data points extremely close to each other.
            # Such doppelgänger points can lead to bad splines
            # later on, and so we remove them now.
            a_values, perturbation_values = remove_doppelgängers(
                a_values, perturbation_values, copy=True)
            self.a_values[k_local] = a_values
            # If k is above that of self.k_max, it means that this
            # particular perturbation is not trusted at this high
            # k value. When this is the case, we store the untrusted,
            # non-detrended data, which we then process later
            if self.k_magnitudes[k] > self.k_max:
                untrusted_perturbations[k_local] = (
                    asarray(a_values).copy(),
                    asarray(perturbation_values).copy(),
                )
                continue
            # Partition a_values into a number of intervals,
            # each to be detrended individually.
            # These intervals are determined based on "critical"
            # times in the history of the universe, e.g. matter-
            # radiation equality. We place a boundary of an interval at
            # such critical times, and also halfway (in log-space) in
            # between such critical times. Additionally, the largest
            # (in linear and log-space) intervals gets divided into
            # two intervals of half the size.
            a_criticals = find_critical_times()
            interval_boarders = a_criticals[
                (a_values[0] < a_criticals) & (a_criticals < a_values[a_values.shape[0] - 1])
            ]
            interval_boarders = np.unique(np.concatenate(
                (interval_boarders, [a_values[0], a_values[a_values.shape[0] - 1]])
            ))
            interval_loga_values = np.log(interval_boarders)
            interval_boarders = np.unique(np.concatenate(
                (
                    interval_boarders,
                    np.exp(interval_loga_values[:interval_loga_values.shape[0] - 1]
                        + 0.5*np.diff(interval_loga_values))
                )
            ))
            index = np.argmax(np.diff(interval_boarders))
            interval_loga_values = np.log(interval_boarders)
            index_log = np.argmax(np.diff(interval_loga_values))
            interval_boarders = np.unique(np.concatenate(
                (
                    interval_boarders,
                    [
                        0.5*(interval_boarders[index] + interval_boarders[index + 1]),
                        np.exp(0.5*(interval_loga_values[index_log]
                            + interval_loga_values[index_log + 1])),
                    ],
                )
            ))
            # Find indices in a_values matching the interval boarders.
            # Ensure that there is at least min_points_in_interval
            # points in each interval.
            min_points_in_interval = 16
            loga_values = np.log(a_values)
            interval_indices = asarray([
                    np.argmin(np.abs(loga_values - log(value))) for value in interval_boarders],
                dtype=C2np['Py_ssize_t'],
            )
            interval_indices[0] = 0
            interval_indices[interval_indices.shape[0] - 1] = a_values.shape[0]
            for i in range(interval_indices.shape[0] - 1, 0, -1):
                difference = interval_indices[i] - interval_indices[i-1]
                if difference < min_points_in_interval:
                    interval_indices[i-1] -= min_points_in_interval - difference
                    if interval_indices[i-1] < 0:
                        interval_indices = interval_indices[i:]
                        interval_indices[0] = 0
                        break
            if interval_indices.shape[0] == 1:
                interval_indices = asarray([0, a_values.shape[0]])
            interval_indices[interval_indices.shape[0] - 1] -= 1
            self.interval_boarders[k_local] = asarray(a_values)[interval_indices]
            interval_indices[interval_indices.shape[0] - 1] += 1
            # Perform non-linear detrending on each interval
            for i in range(interval_indices.shape[0] - 1):
                a_min = self.interval_boarders[k_local][i]
                a_max = self.interval_boarders[k_local][i+1]
                index_left = interval_indices[i] - crossover
                if index_left < 0:
                    index_left = 0
                index_right = interval_indices[i+1] + 1 + crossover
                if index_right > a_values.shape[0]:
                    index_right = a_values.shape[0]
                interval_a_values = a_values[index_left:index_right]
                interval_perturbation_values = perturbation_values[index_left:index_right]
                interval_perturbations_detrended = self.detrend(
                    interval_a_values, interval_perturbation_values, k, k_local, i,
                )
                # Take notice of the largest trusted k
                if k >= largest_trusted_k:
                    if k > largest_trusted_k:
                        a_values_largest_trusted_k[:] = None
                        perturbations_detrended_largest_trusted_k[:] = None
                    largest_trusted_k = k
                    interval_boarders_largest_trusted_k = self.interval_boarders[k_local]
                    a_values_largest_trusted_k[i] = asarray(interval_a_values).copy()
                    perturbations_detrended_largest_trusted_k[i] = (
                        asarray(interval_perturbations_detrended).copy()
                    )
                # Construct cubic spline of
                # {a, perturbations - trend}.
                spline = Spline(interval_a_values, interval_perturbations_detrended,
                    f'detrended {self.class_species} {self.var_name} perturbations '
                    f'as function of a at k = {self.k_magnitudes[k]} {unit_length}⁻¹ '
                    f'in interval {i} (a ∈ [{a_min}, {a_max}])',
                    logx=True,
                )
                self.splines[k_local, i] = spline
        # Now each process contains trends and splines for all
        # trusted perturbations owned by themselves.
        # Find the largest trusted k for all processes.
        largest_trusted_k = allreduce(largest_trusted_k, op=MPI.MAX)
        # Does this process contain untrusted perturbations
        # yet to be processed?
        contains_untrusted_perturbations = (
                self.k_indices.shape[0] > 0
            and largest_trusted_k < self.k_indices[self.k_indices.shape[0] - 1]
        )
        any_contain_untrusted_perturbations = allreduce(contains_untrusted_perturbations,
            op=MPI.LOR)
        if any_contain_untrusted_perturbations:
            # Which process holds the largest trusted perturbation?
            rank_largest_trusted_k = allreduce(
                rank + 1 if largest_trusted_k in self.k_indices else 0, op=MPI.SUM)
            rank_largest_trusted_k -= 1
            if rank_largest_trusted_k == -1:
                abort(
                    f'No trusted {self.class_species} {self.var_name} perturbations available as '
                    f'the perturbation with lowest k is at {self.k_magnitudes[0]} {unit_length}⁻¹ '
                    f'while class_k_max for this perturbation is at {self.k_max} {unit_length}⁻¹. '
                )
        # Now construct splines for untrusted perturbations,
        # if any exist on any process.
        if any_contain_untrusted_perturbations:
            masterprint('Processing untrusted transfer functions ...')
            # Untrusted perturbations exist. Communicate the data of the
            # largest trusted perturbations to all processes which
            # contain untrusted perturbations.
            if rank == rank_largest_trusted_k:
                for rank_other in range(nprocs):
                    if rank_other == rank or not recv(source=rank_other):
                        continue
                    send(interval_boarders_largest_trusted_k.shape[0], dest=rank_other)
                    Send(interval_boarders_largest_trusted_k, dest=rank_other)
                    n_intervals = 0
                    for i in range(self.n_intervals):
                        if a_values_largest_trusted_k[i] is None:
                            break
                        n_intervals += 1
                    send(n_intervals, dest=rank_other)
                    for i in range(n_intervals):
                        send(a_values_largest_trusted_k[i].shape[0], dest=rank_other)
                        Send(a_values_largest_trusted_k[i], dest=rank_other)
                        Send(perturbations_detrended_largest_trusted_k[i], dest=rank_other)
            else:
                send(contains_untrusted_perturbations, dest=rank_largest_trusted_k)
                if contains_untrusted_perturbations:
                    a_values_largest_trusted_k[:] = None
                    perturbations_detrended_largest_trusted_k[:] = None
                    interval_boarders_largest_trusted_k = empty(
                        recv(source=rank_largest_trusted_k), dtype=C2np['double'])
                    Recv(interval_boarders_largest_trusted_k, source=rank_largest_trusted_k)
                    for i in range(recv(source=rank_largest_trusted_k)):
                        size = recv(source=rank_largest_trusted_k)
                        a_values_largest_trusted_k[i] = empty(size, dtype=C2np['double'])
                        perturbations_detrended_largest_trusted_k[i] = (
                            empty(size, dtype=C2np['double']))
                        Recv(a_values_largest_trusted_k[i], source=rank_largest_trusted_k)
                        Recv(perturbations_detrended_largest_trusted_k[i],
                            source=rank_largest_trusted_k)
            for i in range(a_values_largest_trusted_k.shape[0]):
                if a_values_largest_trusted_k[i] is None:
                    a_values_largest_trusted_k = a_values_largest_trusted_k[:i]
                    perturbations_detrended_largest_trusted_k = (
                        perturbations_detrended_largest_trusted_k[:i])
                    break
            # Now all processes containing untrusted perturbations
            # have the data for the largest trusted perturbation.
            # We shall now construct splines for the untrusted
            # perturbations. We do this by reusing the detrended data
            # for the largest trusted perturbation for all untrusted
            # perturbations. Individual factors and exponents are still
            # inferred directly from the untrusted data.
            if contains_untrusted_perturbations:
                # Carry out the morphing for each
                # of the untrusted perturbations.
                for k_local in range(untrusted_perturbations.shape[0]):
                    if untrusted_perturbations[k_local] is None:
                        continue
                    k = self.k_indices[k_local]
                    self.interval_boarders[k_local] = interval_boarders_largest_trusted_k
                    a_values, perturbation_values = untrusted_perturbations[k_local]
                    for i in range(a_values_largest_trusted_k.shape[0]):
                        a_min = a_values_largest_trusted_k[i][0]
                        a_max = a_values_largest_trusted_k[i][
                            a_values_largest_trusted_k[i].shape[0] - 1]
                        # Interpolate untrusted perturbation onto the
                        # a_values for the last trusted perturbation,
                        # in the current interval.
                        interval_perturbation_values = np.interp(a_values_largest_trusted_k[i],
                            a_values, perturbation_values)
                        # Do detrending. This sets the factor and
                        # exponent on self, which is all we need.
                        interval_perturbations_detrended = self.detrend(
                            a_values_largest_trusted_k[i], interval_perturbation_values,
                            k, k_local, i,
                        )
                        # Create the spline
                        spline = Spline(
                            a_values_largest_trusted_k[i],
                            perturbations_detrended_largest_trusted_k[i],
                            f'detrended {self.class_species} {self.var_name} perturbations '
                            f'as function of a at k = {self.k_magnitudes[k]} {unit_length}⁻¹ '
                            f'in interval {i} (a ∈ [{a_min}, {a_max}]) '
                            f'(produced from the largest trusted perturbation)',
                            logx=True,
                        )
                        self.splines[k_local, i] = spline
            # Done with all untrusted perturbations
            Barrier()
            masterprint('done')
        # If the detrended perturbations should be plotted,
        # this is done by the master process, which must then receive
        # the detrended perturbations from the other processes.
        if class_plot_perturbations:
            masterprint(f'Plotting detrended transfer functions ...')
            if master:
                for rank_other in range(nprocs):
                    if rank_other == rank:
                        n_plots = self.k_gridsize_local
                    else:
                        n_plots = recv(source=rank_other)
                    factors   = empty(self.n_intervals, dtype=C2np['double'])
                    exponents = empty(self.n_intervals, dtype=C2np['double'])
                    for k_local in range(n_plots):
                        if rank_other == rank:
                            k = self.k_indices[k_local]
                            factors   = asarray(self.factors  [k_local, :]).copy()
                            exponents = asarray(self.exponents[k_local, :]).copy()
                            splines = self.splines[k_local, :]
                        else:
                            k = recv(source=rank_other)
                            Recv(factors, source=rank_other)
                            Recv(exponents, source=rank_other)
                            splines = empty(self.n_intervals, dtype=object)
                            for i in range(self.n_intervals):
                                if recv(source=rank_other):
                                    continue
                                size = recv(source=rank_other)
                                interval_a_values = get_buffer(size, 'x')
                                Recv(interval_a_values, source=rank_other)
                                interval_perturbations_detrended = get_buffer(size, 'y')
                                Recv(interval_perturbations_detrended, source=rank_other)
                                # Recreate spline at the master process
                                splines[i] = Spline(interval_a_values,
                                    interval_perturbations_detrended,
                                    f'detrended {self.class_species} {self.var_name} '
                                    f'perturbations as function of a at '
                                    f'k = {self.k_magnitudes[k]} {unit_length}⁻¹',
                                    logx=True,
                                )
                        plot_detrended_perturbations(
                            k,
                            self.k_magnitudes[k],
                            transferfunctions_registered[self.var_name],
                            self.class_species,
                            factors,
                            exponents,
                            splines,
                            self.k_magnitudes[largest_trusted_k],
                            crossover,
                        )
            else:
                send(self.k_gridsize_local, dest=master_rank)
                for k_local in range(self.k_gridsize_local):
                    k = self.k_indices[k_local]
                    send(k, dest=master_rank)
                    Send(self.factors[k_local, :], dest=master_rank)
                    Send(self.exponents[k_local, :], dest=master_rank)
                    for i in range(self.n_intervals):
                        spline = self.splines[k_local, i]
                        send(spline is None, dest=master_rank)
                        if spline is None:
                            continue
                        send(spline.x.shape[0], dest=master_rank)
                        Send(spline.x, dest=master_rank)
                        Send(spline.y, dest=master_rank)
            masterprint('done')
        # All perturbations have been processed
        Barrier()
        masterprint('done')

    # Helper functions for the process method
    @cython.header(
        # Arguments
        x='double[::1]',
        y='double[::1]',
        k='Py_ssize_t',
        k_local='Py_ssize_t',
        i='Py_ssize_t',
        # Locals
        fitted_trends=list,
        returns='double[::1]',
    )
    def detrend(self, x, y, k, k_local, i):
        import scipy.optimize
        # Maximum (absolute) allowed exponent in the trend.
        # If an exponent greater than this is found,
        # the program will terminate.
        exponent_max = 15
        # The data to be splined is in the form
        # {a, perturbation_values - trend},
        # with trend = factor*a**exponent. Here we find this
        # trend through curve fitting of perturbation_values.
        fitted_trends = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            for initial_guess in (
                (-1, 0),
                (+1, 0),
            ):
                for bounds in (
                    ([-ထ, -exponent_max], [+ထ,  0           ]),
                    ([-ထ,  0           ], [+ထ, +exponent_max]),
                ):
                    try:
                        fitted_trends.append(
                            scipy.optimize.curve_fit(
                                self.power_law,
                                asarray(x),
                                asarray(y),
                                initial_guess,
                                bounds=bounds,
                                ftol=1e-12,
                                xtol=1e-12,
                                gtol=1e-12,
                                maxfev=1_000,
                            )
                        )
                    except:
                        pass
        # The best fit is determined from the variance of
        # the exponent. Some times, bad fits gets assigned a
        # variance of exactly zero. Bump such occurrences to
        # infinity before locating the best fit.
        for fitted_trend in fitted_trends:
            if fitted_trend[1][1,1] == 0:
                fitted_trend[1][1,1] = ထ
        if fitted_trends:
            self.factors[k_local, i], self.exponents[k_local, i] = fitted_trends[
                np.argmin([fitted_trend[1][1,1] for fitted_trend in fitted_trends])
            ][0]
        else:
            warn(
                f'Failed to detrend {self.var_name} perturbations '
                + ('' if self.component is None else f'for {self.component.name} ')
                + f'at k = {self.k_magnitudes[k]} {unit_length}⁻¹. '
                f'The simulation will carry on without this detrending.'
            )
            self.factors[k_local, i], self.exponents[k_local, i] = 0, 1
        if abs(self.factors[k_local, i]) == ထ:
            abort(
                f'Error processing {self.var_name} perturbations '
                + ('' if self.component is None else f'for {self.component.name} ')
                + f'at k = {self.k_magnitudes[k]} {unit_length}⁻¹: '
                f'Detrending resulted in factor = {self.factors[k_local, i]}.'
            )
        # When the exponent is found to be 0, there is no reason
        # to keep a non-zero factor as the detrending is then
        # just a constant offset.
        if isclose(self.exponents[k_local, i], 0, rel_tol=1e-9, abs_tol=1e-6):
            self.factors[k_local, i], self.exponents[k_local, i] = 0, 1
        # Construct the trend and the detrended perturbations
        trend = self.factors[k_local, i]*asarray(x)**self.exponents[k_local, i]
        interval_perturbations_detrended = asarray(y) - trend
        return interval_perturbations_detrended
    @staticmethod
    def power_law(x, factor, exponent):
        return factor*x**exponent

    # Method which finds out which scale factor interval a given scale
    # factor value lies within, given the local perturbation index.
    @cython.header(
        # Arguments
        k_local='Py_ssize_t',
        a='double',
        # Locals
        a_max='double',
        index='Py_ssize_t',
        index_lower='Py_ssize_t',
        index_upper='Py_ssize_t',
        interval_boarders='double[::1]',
        state_changed='bint',
        returns='Py_ssize_t',
    )
    def get_interval(self, k_local, a):
        # Get all intervals for this perturbation
        interval_boarders = self.interval_boarders[k_local]
        # Find specific interval using binary search
        a_max = interval_boarders[interval_boarders.shape[0] - 1]
        if a >= a_max:
            index = interval_boarders.shape[0] - 2
        else:
            index_lower = 0
            index_upper = interval_boarders.shape[0] - 2
            state_changed = True
            while state_changed:
                state_changed = False
                index = (index_lower + index_upper)//2
                if index != index_lower and interval_boarders[index] <= a:
                    index_lower = index
                    state_changed = True
                if index != index_upper and a < interval_boarders[index + 1]:
                    index_upper = index
                    state_changed = True
            if index_upper == index_lower + 1:
                index = index_upper
        return index

    # Method for evaluating the k'th transfer function
    # at a given scale factor.
    @cython.pheader(
        # Arguments
        k_local='Py_ssize_t',
        a='double',
        # Locals
        exponent='double',
        factor='double',
        i='Py_ssize_t',
        spline='Spline',
        returns='double',
    )
    def eval(self, k_local, a):
        i = self.get_interval(k_local, a)
        factor   = self.factors  [k_local, i]
        exponent = self.exponents[k_local, i]
        spline   = self.splines  [k_local, i]
        # The spline is over transfer(a) - trend(a)
        # with trend(a) = factor*a**exponent.
        return spline.eval(a) + factor*a**exponent

    # Main method for getting the transfer function as function of k
    # at a specific value of the scale factor.
    @cython.pheader(
        # Arguments
        a='double',
        a_next='double',
        weight=str,
        # Locals
        a_i='double',
        a_values='double[::1]',
        a_values_list=list,
        fac_density='int',
        i='Py_ssize_t',
        index_min='Py_ssize_t',
        index_max='Py_ssize_t',
        k_local='Py_ssize_t',
        n_side_points='int',
        size='Py_ssize_t',
        spline='Spline',
        t='double',
        t_next='double',
        t_values='double[::1]',
        w_eff_i='double',
        weighted_transfer='double[::1]',
        weighted_transfer_arr=object,  # np.ndarray
        weights='double[::1]',
        weights_arr=object,  # np.ndarray
        returns='double[::1]',
    )
    def as_function_of_k(self, a, a_next=-1, weight=None):
        """The self.data array is used to store the transfer function
        as function of k for the given a. As this array is reused for
        all calls to this function, you cannot get two arrays of
        transfer function values at different times. If you need this,
        make sure to copy the returned array before calling this
        function again.
        If a_next and weight are passed, the transfer function will be
        averaged over the interval [a, a_next] using the time dependent
        function given by weight as weight.
        """
        if self.data is None:
            self.data       = empty(self.k_gridsize      , dtype=C2np['double'])
            self.data_local = empty(self.k_gridsize_local, dtype=C2np['double'])
        # If a weight is specified, compute the weighted average of the
        # transfer function over the interval [a, a_next]. Otherwise,
        # simply compute the transfer function at a. In the case of
        # a_next == a, the weighted average reduces to the transfer
        # function at a, and so in this case we do not do the averaging
        # even if a weight is specified.
        if weight and a_next != a:
            if a_next == -1:
                abort(
                    f'as_function_of_k() was called with a_next = {a_next}, weight = "{weight}". '
                    f'When using a weight you must also specify a_next.'
                )
            # Number of additional tabulated points to include on both
            # sides of the interval [a, a_next]. Should not exceed
            # "crossover" set in TransferFunction.process.
            n_side_points = 1
            # Number of points in the averaging integrands between each
            # pair of points in the tabulated transfer functions.
            fac_density = 10
            # Grab buffers for the integrands
            weights_arr           = self.as_function_of_k_buffers['weights']
            weighted_transfer_arr = self.as_function_of_k_buffers['weighted_transfer']
            weights, weighted_transfer = weights_arr, weighted_transfer_arr
            # The averaging integrals are over cosmic time,
            # not scale factor.
            t, t_next = cosmic_time(a), cosmic_time(a_next)
            # For each k, compute and store the averaged transfer
            # function over the time step, and also the averaged
            # weight by itself.
            for k_local in range(self.k_gridsize_local):
                # Get array of a values between a and a_next at which
                # the k'th transfer function is tabulated.
                a_values = self.a_values[k_local]
                index_min = np.searchsorted(a_values, a, 'right')
                index_max = np.searchsorted(a_values, a_next, 'left')
                index_min -= n_side_points + 1
                index_max += n_side_points
                if index_min < 0:
                    index_min = 0
                if index_max > a_values.shape[0] - 1:
                    index_max = a_values.shape[0] - 1
                a_values = linspace(
                    a_values[index_min], a_values[index_max], (index_max - index_min)*fac_density,
                )
                size = a_values.shape[0]
                # Compute weighted transfer function values
                # at the tabulated times.
                if size > weighted_transfer.shape[0]:
                    weights_arr          .resize(size, refcheck=False)
                    weighted_transfer_arr.resize(size, refcheck=False)
                    weights, weighted_transfer = weights_arr, weighted_transfer_arr
                for i in range(size):
                    a_i = a_values[i]
                    with unswitch:
                        if weight == '1':
                            weights[i] = 1.0
                        elif weight == 'a**(-3*w_eff-1)':
                            w_eff_i = self.component.w_eff(a=a_i)
                            weights[i] = a_i**(-3*w_eff_i - 1)
                        elif weight == 'a**(3*w_eff-2)':
                            w_eff_i = self.component.w_eff(a=a_i)
                            weights[i] = a_i**(3*w_eff_i - 2)
                        elif weight == 'a**(-3*w_eff)':
                            w_eff_i = self.component.w_eff(a=a_i)
                            weights[i] = a_i**(-3*w_eff_i)
                        else:
                            abort(f'weight "{weight}" not implemented in as_function_of_k()')
                    weighted_transfer[i] = weights[i]*self.eval(k_local, a_i)
                    # Replace the i'th scale factor value with the
                    # corresponding cosmic time.
                    a_values[i] = cosmic_time(a_i)
                # All scale factor values have now been replaced
                # with cosmic times.
                t_values = a_values
                # Compute and store the weighted transfer function
                # 1/(ᔑ weight(t) dt) * ᔑ weight(t)*transfer(t) dt.
                spline_weights           = Spline(t_values, weights[:size], 'weight(t)')
                spline_weighted_transfer = Spline(
                    t_values, weighted_transfer[:size], 'weight(t)*transfer(t)'
                )
                self.data_local[k_local] = (spline_weighted_transfer.integrate(t, t_next)
                    /spline_weights.integrate(t, t_next)
                )
        else:
            # For each k, compute and store the transfer function
            # at the given a.
            for k_local in range(self.k_gridsize_local):
                self.data_local[k_local] = self.eval(k_local, a)
        # Gather all local results
        smart_mpi(self.data_local, self.data, mpifun='allgatherv')
        self.data = asarray(self.data)[self.k_indices_all]
        return self.data
    # Persistent buffers used by the as_function_of_k() method,
    # shared among all instances.
    as_function_of_k_buffers = {
        'weights'          : empty(1, dtype=C2np['double']),
        'weighted_transfer': empty(1, dtype=C2np['double']),
    }

    # Method for evaluating the derivative of the k'th transfer
    # function with respect to the scale factor, at a specific value of
    # the scale factor.
    @cython.pheader(
        # Arguments
        k_local='Py_ssize_t',
        a='double',
        # Locals
        exponent='double',
        factor='double',
        i='Py_ssize_t',
        spline='Spline',
        returns='double',
    )
    def eval_deriv(self, k_local, a):
        i = self.get_interval(k_local, a)
        factor   = self.factors  [k_local, i]
        exponent = self.exponents[k_local, i]
        spline   = self.splines  [k_local, i]
        # The spline is over transfer(a) - trend(a)
        # with trend(a) = factor*a**exponent.
        # We then have to add dtrend(a)/da = factor*a**(exponent - 1)
        # to the derivative of the spline to obtain the derivative of
        # the transfer function.
        return spline.eval_deriv(a) + factor*exponent*a**(exponent - 1)

    # Method for getting the derivative of the transfer function
    # with respect to the scale factor, evaluated at a,
    # as a function of k.
    @cython.pheader(
        # Arguments
        a='double',
        # Locals
        k='Py_ssize_t',
        returns='double[::1]',
    )
    def deriv_as_function_of_k(self, a):
        """This method returns an array of derivative data for all k,
        not just the local ones. This method should then always be
        called collectively.
        The self.data_deriv array is used to store the transfer
        function derivatives as function of k for the given a. As this
        array is reused for all calls to this function, you cannot get
        two arrays of transfer function derivatives at different times.
        If you need this, make sure to copy the returned array before
        calling this function again.
        """
        # Populate the data_deriv array with derivatives of the
        # transfer_function(k) and return this array.
        if self.data_deriv is None:
            self.data_deriv       = empty(self.k_gridsize      , dtype=C2np['double'])
            self.data_deriv_local = empty(self.k_gridsize_local, dtype=C2np['double'])
        for k_local in range(self.k_gridsize_local):
            self.data_deriv_local[k_local] = self.eval_deriv(k_local, a)
        # Gather all local results
        smart_mpi(self.data_deriv_local, self.data_deriv, mpifun='allgatherv')
        self.data_deriv = asarray(self.data_deriv)[self.k_indices_all]
        return self.data_deriv
# Global set used by the above class
cython.declare(missing_background_quantities=set)
missing_background_quantities = set()

# Function which finds critical moments in the cosmic history,
# like that of matter-radiation equality.
@lru_cache(copy=True)
def find_critical_times():
    import scipy.signal
    # List storing the critical scale factor values
    a_criticals = []
    # If the CLASS background is disabled,
    # we give up computing any critical times.
    if not enable_class_background:
        return asarray(a_criticals)
    # Get the CLASS background
    cosmoresults = compute_cosmo(class_call_reason=f'in order to find critical times')
    background = cosmoresults.background
    a = background['a']
    t = background['proper time [Gyr]']
    H = background['H [1/Mpc]']
    # Find local extrema in dlog(H)/dlog(t). This usually corresponds
    # roughly to matter-radiation and darkenergy-matter equality.
    dlogH_dlogt = np.gradient(np.log(H), np.log(t))
    loga = np.log(a)
    distance = (
        + np.argmin(np.abs(loga - np.log(1.0)))
        - np.argmin(np.abs(loga - np.log(0.8)))
    )
    if distance < 5:
        distance = 5
    a_criticals += list(a[scipy.signal.find_peaks(+dlogH_dlogt, distance=distance)[0]])
    a_criticals += list(a[scipy.signal.find_peaks(-dlogH_dlogt, distance=distance)[0]])
    # Find time of average value of w(a) for each species.
    # For massive neutrinos this corresponds to the relativistic to
    # non-relativistic transition time.
    for key, arr in background.items():
        match = re.search(r'^\(\.\)rho_(.+)', key)
        if not match:
            continue
        class_species = match.group(1)
        species_info = species_registered.get(
            species_canonical.get(class_species, class_species)
        )
        if species_info is None or not species_info.is_physical:
            continue
        ρ = arr
        p = background.get(f'(.)p_{class_species}')
        if p is None:
            continue
        w = p/ρ
        w_min, w_max = np.min(w), np.max(w)
        if np.isclose(w_min, w_max):
            continue
        a_criticals.append(a[np.argmin(np.abs(w - 0.5*(w_max - w_min)))])
    # Return critical times as a sorted array or unique values
    return np.unique(a_criticals)

# Function which solves the linear cosmology using CLASS,
# from before the initial simulation time and until the present.
@cython.pheader(
    # Arguments
    gridsize='Py_ssize_t',
    gauge=str,
    filename=str,
    class_call_reason=str,
    # Locals
    cosmoresults=object, # CosmoResults
    extra_params=dict,
    k_magnitudes='double[::1]',
    k_magnitudes_str=str,
    params_specialized=dict,
    returns=object,  # CosmoResults
)
def compute_cosmo(gridsize=-1, gauge='synchronous', filename='', class_call_reason=''):
    """All calls to CLASS should be done through this function.
    If no arguments are supplied, CLASS will be run with the parameters
    stored in class_params. The return type is CosmoResults, which
    stores the result of the CLASS computation.
    If gridsize is given, a more in-depth computation will be carried
    out by CLASS, where perturbations are also computed.
    All results from calls to this function are cached (using the
    global variable cosmoresults_cache), so you can safely call this
    function multiple times with the same arguments without it having
    to do the same CLASS computation over and over again.
    The gridsize argument specify the |k| distribution on which the
    perturbations should be tabulated, as defined by get_k_magnitudes().
    The gauge of the transfer functions can be specified by
    the gauge argument, which can be any valid CLASS gauge. Note that
    N-body gauge is not implemented in CLASS.
    If a filename is given, CLASS results are loaded from this file.
    """
    # If a gauge is given explicitly as a CLASS parameter in the
    # parameter file, this gauge should overwrite what ever is passed
    # to this function.
    gauge = class_params.get('gauge', gauge).replace('-', '').lower()
    if gauge == 'nbody':
        masterwarn(
            f'The "nbody" gauge was specified in the call to compute_cosmo. '
            f'For this gauge, you should really pass in "synchronous" '
            f'and then let compute_transfer transform to N-body gauge.'
        )
    if gauge not in ('synchronous', 'newtonian'):
        abort(
            f'In compute_cosmo, gauge was set to "{gauge}" but must be '
            f'either "synchronous" or "Newtonian"'
        )
    # If this exact CLASS computation has already been carried out,
    # return the stored results.
    cosmoresults = cosmoresults_cache.get((gridsize, gauge))
    if cosmoresults is not None:
        return cosmoresults
    # Determine whether to run CLASS "quickly" or "fully",
    # where only the latter computes the perturbations.
    if gridsize == -1:
        # A quick CLASS computation should be carried out,
        # using only the minimal set of parameters.
        extra_params = {}
        k_magnitudes = None
    else:
        # A full CLASS computation should be carried out.
        # Array of |k| values at which to tabulate the perturbations,
        # in both float and str representation.
        k_magnitudes, k_magnitudes_str = get_k_magnitudes(gridsize)
        # Specify the extra parameters with which CLASS should be run
        extra_params = {
            'k_output_values': k_magnitudes_str,
            'gauge': gauge,
            # Needed for perturbation output
            'output': 'dTk vTk',
            # This is used to minimize the number of extra k values
            # inserted automatically by CLASS. With 'P_k_max_1/Mpc' set
            # to 0, only a single additional k mode is inserted,
            # and this at a very small k value.
            'P_k_max_1/Mpc': 0,
        }
    # Merge global and extra CLASS parameters
    params_specialized = class_params | extra_params
    # Transform all CLASS container parameters to str's of
    # comma-separated values. All other CLASS parameters will also
    # be converted to their str representation.
    params_specialized = stringify_dict(params_specialized)
    # Instantiate a CosmoResults object before calling CLASS,
    # in the hope that this exact CLASS call have already been
    # carried out.
    cosmoresults = CosmoResults(
        params_specialized,
        k_magnitudes,
        filename=filename,
        class_call_reason=class_call_reason,
    )
    # Add the CosmoResults object to the cache
    cosmoresults_cache[gridsize, gauge] = cosmoresults
    return cosmoresults
# Dict with keys of the form (gridsize, gauge), storing the results
# of calls to the above function as CosmoResults instances.
cython.declare(cosmoresults_cache=dict)
cosmoresults_cache = {}

# Function for computing transfer functions as function of k
@cython.pheader(
    # Arguments
    component='Component',
    variable=object,  # str or int
    gridsize='Py_ssize_t',
    specific_multi_index=object,  # tuple, int-like or str
    a='double',
    a_next='double',
    gauge=str,
    get=str,
    weight=str,
    # Locals
    H='double',
    aH_transfer_θ_totʹ='double[::1]',
    class_species=str,
    cosmoresults=object,  # CosmoResults
    k='Py_ssize_t',
    k_gridsize='Py_ssize_t',
    k_magnitudes='double[::1]',
    source='double',
    transfer='double[::1]',
    transfer_H_Tʹ='double[::1]',
    transfer_hʹ='double[::1]',
    transfer_spline='Spline',
    transfer_θ_tot='double[::1]',
    var_index='Py_ssize_t',
    w='double',
    weighted_Γ_3H='double',
    ρ_bar='double',
    θ_weight='double',
    ẇ='double',
    returns=tuple,  # (Spline, CosmoResults)
)
def compute_transfer(
    component, variable, gridsize,
    specific_multi_index=None, a=-1, a_next=-1, gauge='N-body', get='spline', weight=None,
):
    """This function calls compute_cosmo which produces a CosmoResults
    instance which can talk to CLASS. Using the δ, θ, etc. methods on
    the CosmoResults object, TransferFunction instances are
    automatically created. All this function really implements
    are then the optional gauge transformations.
    The return value is either (spline, cosmoresults) (get == 'spline')
    or (array, cosmoresults) (get == 'array').
    """
    # Argument processing
    var_index = component.varnames2indices(variable, single=True)
    if a == -1:
        a = universals.a
    gauge = gauge.replace('-', '').lower()
    if gauge not in ('synchronous', 'newtonian', 'nbody'):
        abort(
            f'Gauge was set to "{gauge}" but must be one of '
            f'"N-body", "synchronous", "Newtonian"'
        )
    get = get.lower()
    if get not in ('spline', 'array'):
        abort(
            f'The get argument of compute_transfer was "{get}", '
            f'but must be one of "spline" or "array"'
        )
    # Compute the cosmology via CLASS. As the N-body gauge is not
    # implemented in CLASS, the synchronous gauge is used in its place.
    # We do the transformation from synchronous to N-body gauge later.
    cosmoresults = compute_cosmo(
        gridsize,
        'synchronous' if gauge == 'nbody' else gauge,
        class_call_reason=f'in order to get perturbations of {component.name}',
    )
    k_magnitudes = cosmoresults.k_magnitudes
    k_gridsize = k_magnitudes.shape[0]
    # Get the requested transfer function
    # and transform to N-body gauge if requested.
    if var_index == 0:
        # Get the δ transfer function
        transfer = cosmoresults.δ(a, a_next, component=component, weight=weight)
        # Transform the δ transfer function from synchronous
        # to N-body gauge, if requested.
        if gauge == 'nbody':
            # The gauge transformation looks like
            # δᴺᵇ = δˢ + c⁻²(3aH(1 + w) - a*source/ρ_bar)θˢₜₒₜ/k²,
            # where source is any source term in the homogeneous proper
            # time continuity equation for the given CLASS species.
            source = 0
            for class_species in component.class_species.split('+'):
                species_info = species_registered.get(
                    species_canonical.get(class_species, class_species)
                )
                if species_info is None:
                    continue
                source += species_info.source_continuity(cosmoresults, a)
            # Do the gauge transformation
            ρ_bar = cosmoresults.ρ_bar(a, component)
            transfer_θ_tot = cosmoresults.θ(a)
            H = hubble(a)
            w = component.w(a=a)
            for k in range(k_gridsize):
                transfer[k] += (ℝ[light_speed**(-2)*(3*a*H*(1 + w) - a*source/ρ_bar)]
                    *transfer_θ_tot[k]/k_magnitudes[k]**2)
    elif var_index == 1:
        # Get the θ transfer function
        transfer = cosmoresults.θ(a, a_next, component=component, weight=weight)
        # Transform the θ transfer function from synchronous
        # to N-body gauge, if requested.
        if gauge == 'nbody':
            # The gauge transformation looks like
            # θᴺᵇ = θˢ + hʹ/2 - 3c⁻²(aHθˢₜₒₜ)ʹ/k²,
            # With ʹ = d/dτ being conformal time derivatives.
            transfer_hʹ = cosmoresults.hʹ(a)
            # We need (a*H*θ_tot) differentiated with respect
            # to conformal time, evaluated at the given a.
            # With ʹ = d/dτ = a*d/dt = aȧ*d/da, we have
            # (a*H*θ_tot)ʹ = a*d/dt(ȧ*θ_tot)
            #              = a*ä*θ_tot + a*ȧ*d/dt(θ_tot)
            #              = a*(ä*θ_tot + ȧ²*d/da(θ_tot)),
            # where a dot denotes differentiation
            # with respect to cosmic time.
            aH_transfer_θ_totʹ = a*(
                  ä(a)   *asarray(cosmoresults.θ(a, get='as_function_of_k'      ))
                + ȧ(a)**2*asarray(cosmoresults.θ(a, get='deriv_as_function_of_k'))
            )
            # Now do the gauge transformation
            for k in range(k_gridsize):
                transfer[k] += (0.5*transfer_hʹ[k]
                    - ℝ[3/light_speed**2]*aH_transfer_θ_totʹ[k]/k_magnitudes[k]**2)
            # In order to introduce the lapse potential for the decaying
            # cold dark matter, we have changed the velocity variable
            # away from that used by CLASS. The needed transformation is
            # θ_dcdm_CO𝘕CEPT = θ_dcdm_CLASS + Γ_dcdm/(3H)*H_Tʹ.
            # In the general case for combination species, we have
            # θ_CO𝘕CEPT = θ_CLASS + θ_weight*Γ_dcdm/(3H)*H_Tʹ,
            # θ_weight = (ρ_dcdm_bar + c⁻²P_dcdm_bar)/(
            #   ∑_α (ρ_α_bar + c⁻²P_α_bar)).
            if 'dcdm' in component.class_species.split('+'):
                θ_weight = (               cosmoresults.ρ_bar(a, 'dcdm')
                    + ℝ[light_speed**(-2)]*cosmoresults.P_bar(a, 'dcdm')
                    )/(                    cosmoresults.ρ_bar(a, component)
                    + ℝ[light_speed**(-2)]*cosmoresults.P_bar(a, component))
                weighted_Γ_3H = θ_weight*cosmoresults.Γ_dcdm/(3*hubble(a))
                transfer_H_Tʹ = cosmoresults.H_Tʹ(a)
                for k in range(k_gridsize):
                    transfer[k] += weighted_Γ_3H*transfer_H_Tʹ[k]
    elif var_index == 2 and specific_multi_index == 'trace':
        # Get the δP transfer function
        transfer = cosmoresults.δP(a, a_next, component=component, weight=weight)
        # Transform the δP transfer function from synchronous
        # to N-body gauge, if requested.
        if gauge == 'nbody':
            # The gauge transformation looks like
            # δPᴺᵇ = δPˢ + aρ_bar(3Hw(1 + w) - ẇ)θˢₜₒₜ/k²,
            # where a dot denotes differentiation
            # with respect to cosmic time.
            # Do the gauge transformation.
            transfer_θ_tot = cosmoresults.θ(a)
            ρ_bar = cosmoresults.ρ_bar(a, component)
            H = hubble(a)
            w = component.w(a=a)
            ẇ = component.ẇ(a=a)
            for k in range(k_gridsize):
                transfer[k] += ℝ[a*ρ_bar*(3*H*w*(1 + w) - ẇ)]*transfer_θ_tot[k]/k_magnitudes[k]**2
    elif (    var_index == 2
          and isinstance(specific_multi_index, tuple)
          and len(specific_multi_index) == 2
          ):
        # Get the σ transfer function
        transfer = cosmoresults.σ(a, a_next, component=component, weight=weight)
    else:
        abort(f'I do not know how to get transfer function of multi_index {specific_multi_index} '
              f'of variable number {var_index}'
              )
    # Construct a spline object over the tabulated transfer function
    if get == 'spline':
        transfer_spline = Spline(k_magnitudes, transfer,
            f'Transfer function (var_index = {var_index}) '
            f'of component {component.name} at a = {a}',
            logx=True,
            logy=False,
        )
        return transfer_spline, cosmoresults
    elif get == 'array':
        return transfer, cosmoresults

# Function which given a grid size computes an array of k values
# based on the boxsize and the k_modes_per_decade parameter.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    # Locals
    cached=object,  # tuple
    k_magnitudes_str=str,
    k_gridsize='Py_ssize_t',
    k_magnitudes='double[::1]',
    k_max='double',
    k_min='double',
    logk='double',
    logk_magnitudes=object,  # list, np.ndarray
    logk_modes_per_decade_interp=object,  # scipy.interpolate.interp1d
    logk_max='double',
    logk_min='double',
    nyquist='Py_ssize_t',
    returns=object,  # tuple
)
def get_k_magnitudes(gridsize):
    # Cache lookup
    cached = k_magnitudes_cache.get(gridsize)
    if cached is not None:
        return cached
    # As we ignore the Nyquist points, (nyquist - 1) needs to be
    # positive, requiring gridsize >= 4.
    if gridsize < 4:
        abort(f'get_k_magnitudes() got gridsize = {gridsize} < 4')
    # Minimum and maximum k
    k_min = ℝ[2*π/boxsize]
    nyquist = gridsize//2
    k_max = k_min*sqrt(3*(nyquist - 1)**2)
    k_min *= ℝ[1 - k_safety_factor]
    k_max *= ℝ[1 + k_safety_factor]
    logk_min = log10(k_min)
    logk_max = log10(k_max)
    # Starting from log10(k_min), append new log10(k)
    # using a running number of modes/decade.
    logk = logk_min
    logk_magnitudes = [logk]
    logk_modes_per_decade_interp = get_logk_modes_per_decade_interp()
    while logk <= logk_max:
        logk += 1/logk_modes_per_decade_interp(logk)
        logk_magnitudes.append(logk)
    k_gridsize = len(logk_magnitudes)
    if k_gridsize > k_gridsize_max:
        abort(
            f'Too many k modes ({k_gridsize}, for gridsize = {gridsize}) for CLASS to handle. '
            f'To allow for more k modes, you may increase the CLASS macro '
            f'_ARGUMENT_LENGTH_MAX_ in include/parser.h.'
        )
    logk_magnitudes = asarray(logk_magnitudes)
    # The last log10(k) is guaranteed to be slightly larger
    # than logk_max. Scale the tabulated log10(k) so that they exactly
    # span [log10(k_min), log10(k_max)].
    logk_magnitudes -= logk_min
    logk_magnitudes *= (logk_max - logk_min)/logk_magnitudes[k_gridsize - 1]
    logk_magnitudes += logk_min
    # Construct the |k| array
    k_magnitudes = 10**logk_magnitudes
    # Convert to CLASS units, i.e. Mpc⁻¹, which shall be the unit
    # used for the str representation of k_magnitudes.
    k_magnitudes = asarray(k_magnitudes)/units.Mpc**(-1)
    # Limit the number of decimals on each |k|,
    # also producing the str representation.
    with disable_numpy_summarization():
        k_magnitudes_str = np.array2string(
            k_magnitudes,
            max_line_width=ထ,
            formatter={'float': k_float2str},
            separator=',',
        ).strip('[]')
    k_magnitudes = np.fromstring(k_magnitudes_str, sep=',')
    if len(set(k_magnitudes)) != k_gridsize:
        abort(
            'The requested k sampling is too dense, leading to modes that are exactly equal '
            'after limiting the number of decimal places. Though this ought not to ever happen, '
            'it should help to lower the k_modes_per_decade parameter. '
            'Alternatively, you can try lowering the linear.k_safety_factor variable.'
        )
    # Convert back to the current CO𝘕CEPT unit system
    k_magnitudes = asarray(k_magnitudes)*units.Mpc**(-1)
    # Cache and return both the float and str representation
    k_magnitudes_cache[gridsize] = (k_magnitudes, k_magnitudes_str)
    return k_magnitudes_cache[gridsize]
# Cache and helper objects used by the get_k_magnitudes() function
cython.declare(
    k_magnitudes_cache=dict,
    k_str_n_decimals='int',
    k_safety_factor='double',
)
k_magnitudes_cache = {}
def get_logk_modes_per_decade_interp():
    import scipy.interpolate
    logk_modes_per_decade_interp = lambda logk, *, f=scipy.interpolate.interp1d(
        np.log10(tuple(k_modes_per_decade.keys())),
        tuple(k_modes_per_decade.values()),
        kind='linear',
        bounds_error=False,
        fill_value=(
            k_modes_per_decade[np.min(tuple(k_modes_per_decade.keys()))],
            k_modes_per_decade[np.max(tuple(k_modes_per_decade.keys()))],
        ),
    ): float(f(logk))
    return logk_modes_per_decade_interp
def k_float2str(k_float):
    k_str = 𝕊['{{:.{}e}}'.format(k_str_n_decimals)].format(k_float)
    k_str = k_str.replace('+0', '+').replace('-0', '-').replace('e+0', '')
    return k_str
k_str_n_decimals = int(ceil(log10(1 + np.max(tuple(k_modes_per_decade.values())))))
k_safety_factor = 2*10**float(-k_str_n_decimals)

# Function which realises a given variable on a component
# from a supplied transfer function.
@cython.pheader(
    # Arguments
    component='Component',
    variable=object,  # str or int
    transfer_spline='Spline',
    cosmoresults=object,  # CosmoResults
    specific_multi_index=object,  # tuple, int-like or str
    a='double',
    options=dict,
    use_gridˣ='bint',
    # Locals
    H='double',
    Jⁱ_ptr='double*',
    N_str=str,
    compound_variable='bint',
    cosmoresults_δ=object,  # CosmoResults
    deconv_order='int',
    dim='int',
    domain_start_i='Py_ssize_t',
    domain_start_j='Py_ssize_t',
    domain_start_k='Py_ssize_t',
    f_growth='double',
    factor='double',
    fluid_index='Py_ssize_t',
    fluidscalar='FluidScalar',
    fluidvar=object,  # Tensor
    fluidvar_name=str,
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    index='Py_ssize_t',
    indexʳ='Py_ssize_t',
    indexˣ='Py_ssize_t',
    indexˣʸᶻ='Py_ssize_t',
    index0='Py_ssize_t',
    index1='Py_ssize_t',
    interpolation_order='int',
    j='Py_ssize_t',
    k='Py_ssize_t',
    k_factor='double',
    k_index0='Py_ssize_t',
    k_index1='Py_ssize_t',
    k_magnitude='double',
    k2='Py_ssize_t',
    k2_max='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    mass='double',
    mom='double*',
    multi_index=object,  # tuple or str
    nyquist='Py_ssize_t',
    option_key=str,
    option_val=object,  # str or bool
    options_linear=dict,
    particle_components=list,
    particle_index='int',
    particle_shift='double',
    particle_shifts='double[::1]',
    particle_var_name=str,
    pos='double*',
    posxˣ='double*',
    posyˣ='double*',
    poszˣ='double*',
    processed_specific_multi_index=object,  # tuple or str
    reuse_slab_structure='bint',
    slab='double[:, :, ::1]',
    slab_ptr='double*',
    sqrt_power='double',
    sqrt_power_common='double[::1]',
    slab_structure='double[:, :, ::1]',
    slab_structure_info=dict,
    slab_structure_name=str,
    structure_ptr='double*',
    tensor_rank='int',
    transfer='double',
    transfer_spline_δ='Spline',
    uⁱ='double[:, :, ::1]',
    w='double',
    w_eff='double',
    x_gridpoint='double',
    y_gridpoint='double',
    z_gridpoint='double',
    Δmom='double*',
    δ_min='double',
    θ='double',
    ςⁱⱼ_ptr='double*',
    ψⁱ='double[:, :, ::1]',
    ϱ_bar='double',
    ϱ_ptr='double*',
    𝒫_ptr='double*',
)
def realize(
    component, variable, transfer_spline, cosmoresults,
    specific_multi_index=None, a=-1, options=None, use_gridˣ=False,
):
    """This function realises a single variable of a component,
    given the transfer function as a Spline (using |k⃗| in physical units
    as the independent variable) and the corresponding CosmoResults
    object, which carry additional information from the CLASS run that
    produced the transfer function. If only a single fluidscalar of the
    fluid variable should be realised, the multi_index of this
    fluidscalar may be specified. If you want a realisation at a time
    different from the present you may specify an a.
    If a particle component is given, the Zel'dovich approximation is
    used to distribute the particles and assign momenta.

    Several options has to be specified to define how the realisation is
    to be carried out. These options are contained in the "options"
    argument. By default, the options are
    options = {
        # Linear realisation options
        'velocities from displacements': False,
        # Non-linear realisation options
        'structure'     : 'primordial',
        'compound-order': 'linear',
    }
    which corresponds to linear realisation. For particle components
    (which can not be realised continually) only linear realisation is
    possible, and thus only the linear option matters. When
    'velocities from displacements' is True, the particle momenta will
    be set from the same displacement field ψⁱ as is used for the
    positions, using the linear growth rate f to convert between
    displacement and velocity. Otherwise, momenta will be constructed
    from their own velocity field uⁱ, using their own transfer function
    but the same (primordial) noise. Note that for particle components
    you must realise the momenta prior to the positions. If
    'velocities from displacements' is True, you should call this
    function once with variable = 1 (momenta), but with a
    transfer_spline for ψⁱ (corresponding to variable 0).
    Another linear option 'back-scaling' might be specified, but it is
    not used by this function.
    Taking Jⁱ as an example of a fluid variable realisation,
    linear realisation looks like
        Jⁱ(x⃗) = a**(1 - 3w_eff)ϱ_bar(1 + w)ℱₓ⁻¹[T_θ(k)ζ(k)K(k⃗)ℛ(k⃗)],
    where ζ(k) is the primordial curvature perturbation, T_θ(k) is the
    passed transfer function for θ, ℛ(k⃗) is a field of primordial noise,
    and K(k⃗) is the tensor structure (often referred to as the k factor)
    needed to convert from θ to uⁱ. For uⁱ, K(k⃗) = -ikⁱ/k². The factors
    outside the Fourier transform then converts from uⁱ to Jⁱ.
    We can instead choose to use the non-linearly evolved structure
    of ϱ, by using options['structure'] == 'non-linear'. Then the
    realisation looks like
        Jⁱ(x⃗) = a**(1 - 3w_eff)ϱ_bar(1 + w)ℱₓ⁻¹[T_θ(k)/T_δϱ(k)K(k⃗)δϱ(k⃗)],
    where δϱ(k⃗) = ℱₓ[δϱ(x⃗)] is computed from the present ϱ(x⃗) grid,
    and T_δϱ(k) is the (not passed) transfer function of δϱ.
    An orthogonal option is 'compound-order'. Setting this to
    'non-linear' signals that the multiplication which takes uⁱ to Jⁱ
    should be done using non-linear variables rather than background
    quantities. That is,
        Jⁱ(x⃗) = a**(1 - 3w_eff)(ϱ(x⃗) + c⁻²𝒫(x⃗))ℱₓ⁻¹[...].

    For both particle and fluid components it is assumed that the
    passed component is of the correct size beforehand. No resizing
    will take place in this function.
    """
    if a == -1:
        a = universals.a
    if options is None:
        options = {}
    options = {key.lower().replace(' ', '').replace('-', ''):
        (val.lower().replace(' ', '').replace('-', '') if isinstance(val, str) else val)
        for key, val in options.items()
    }
    # By default, use linear realisation options and do not construct
    # the velocities directly from the displacements.
    options_linear = {
        # Linear options
        'interpolation': 2,  # CIC
        'velocitiesfromdisplacements': False,
        # Non-linear options
        'structure'    : 'primordial',
        'compoundorder': 'linear',
    }
    for option_key, option_val in options_linear.items():
        if option_key not in options:
            options[option_key] = option_val
    for option_key in options:
        if option_key not in {
            'interpolation',
            'velocitiesfromdisplacements',
            'backscaling',
            'structure',
            'compoundorder',
        }:
            abort(f'realize() did not understand realisation option "{option_key}"')
    if options['structure'] not in {'primordial', 'nonlinear'}:
        abort(f'Unrecognised value "{options["structure"]}" for options["structure"]')
    if options['compoundorder'] not in {'linear', 'nonlinear'}:
        abort(f'Unrecognised value "{options["compoundorder"]}" for options["compound-order"]')
    options['velocitiesfromdisplacements'] = bool(options['velocitiesfromdisplacements'])
    # Get the index of the fluid variable to be realised
    # and print out progress message.
    processed_specific_multi_index = ()
    particle_var_name = 'pos'
    fluid_index = component.varnames2indices(variable, single=True)
    if component.representation == 'particles':
        if use_gridˣ:
            masterwarn(
                f'realize() was called with use_gridˣ=True '
                f'for the particle component {component.name}. '
                f'This will be ignored.'
            )
        # For particles, the only variables that exist are the positions
        # and the momenta, corresponding to a fluid_index of 0 and 1,
        # respectively.
        particle_var_name = {0: 'pos', 1: 'mom'}[fluid_index]
        # When the 'velocities from displacements' option is enabled,
        # both the positions and the momenta are constructed from the
        # displacement field ψⁱ. It is then illegal to request a position
        # realisation directly.
        if particle_var_name == 'pos' and options['velocitiesfromdisplacements']:
            abort(
                f'A realisation of particle positions for {component.name} was requested. '
                f'As this component is supposed to get its velocities from the displacements, '
                f'you should only call realize() for the momenta/velocities, which will then '
                f'realise both positions and momenta.'
            )
        if component.N > 1 and isint(ℝ[cbrt(component.N)]):
            N_str = str(int(round(ℝ[cbrt(component.N)]))) + '³'
        else:
            N_str = str(component.N)
        if specific_multi_index is None:
            masterprint(
                f'Realising {N_str} particle',
                'momenta and positions' if options['velocitiesfromdisplacements']
                    else {'pos': 'positions', 'mom': 'momenta'}[particle_var_name],
                f'of {component.name} ...'
            )
        else:
            processed_specific_multi_index = (
                component.fluidvars[fluid_index].process_multi_index(specific_multi_index)
            )
            if options['velocitiesfromdisplacements']:
                masterprint(
                    f'Realising {N_str} particle momenta[{processed_specific_multi_index[0]}] '
                    f'and positions[{processed_specific_multi_index[0]}] of {component.name} ...'
                )
            else:
                masterprint(
                    f'Realising {N_str} particle',
                    {'pos': 'positions', 'mom': 'momenta'}[particle_var_name]
                        + f'[{processed_specific_multi_index[0]}] '
                    f'of {component.name} ...'
                )
        # For particles, the Zel'dovich approximation is used for the
        # realisation. For the positions, the displacement field ψⁱ is
        # really what is realised, while for the momenta, the velocity
        # field uⁱ is what is really realised. Both of these are vector
        # fields, and so we have to set fluid_index to 1 so that
        # multi_index takes on vector values ((0, ), (1, ), (2, )).
        fluid_index = 1
    elif component.representation == 'fluid':
        fluidvar_name = component.fluid_names['ordered'][fluid_index]
        if specific_multi_index is None:
            masterprint(
                f'Realising {fluidvar_name} of {component.name} '
                f'with grid size {component.gridsize} ...'
            )
        else:
            processed_specific_multi_index = (
                component.fluidvars[fluid_index].process_multi_index(specific_multi_index)
            )
            masterprint(
                f'Realising {fluidvar_name}{{}} of {component.name} '
                f'with grid size {component.gridsize} ...'
                .format(
                    '' if fluid_index == 0 else (
                        f"['{processed_specific_multi_index}']"
                        if isinstance(processed_specific_multi_index, str) else (
                            '[{}]'.format(
                                str(processed_specific_multi_index).strip('()')
                                if len(processed_specific_multi_index) > 1
                                else processed_specific_multi_index[0]
                            )
                        )
                    )
                )
            )
    # Determine the grid size of the grid used to do the realisation
    if component.representation == 'particles':
        if not isint(ℝ[cbrt(component.N)]):
            abort(
                f'Cannot perform realisation of {component.name} '
                f'with N = {component.N}, as N is not a cubic number.'
            )
        gridsize = int(round(ℝ[cbrt(component.N)]))
    elif component.representation == 'fluid':
        gridsize = component.gridsize
    if gridsize%nprocs != 0:
        abort(
            f'The realisation uses a gridsize of {gridsize}, '
            f'which is not evenly divisible by {nprocs} processes.'
        )
    # A compound order of 'nonlinear' only makes a difference for
    # compound variables; that is, Jⁱ and ςⁱⱼ. If what we are realising
    # is another variable, switch this back to 'linear'.
    if fluid_index == 1:
        # We are realising Jⁱ
        compound_variable = True
    elif fluid_index == 2 and processed_specific_multi_index != 'trace':
        # We are realising ςⁱⱼ
        compound_variable = True
    else:
        compound_variable = False
    if not compound_variable:
        if options['compoundorder'] == 'nonlinear':
            options['compoundorder'] = 'linear'
    # Abort if the non-linear structure option was passed
    # for a particle component, as these can only be realised
    # from primordial noise.
    if (component.representation == 'particles'
        and options['structure'] != options_linear['structure']
    ):
        abort('Can only do particle realisation using primordial noise/structure')
    # When realising δ, it only makes sense to realise it linearly
    if fluid_index == 0 and options['structure'] != options_linear['structure']:
        abort('Can only do linear realisation of δ')
    # Extract various variables
    H = hubble(a)
    w = component.w(a=a)
    w_eff = component.w_eff(a=a)
    ϱ_bar = component.ϱ_bar
    # Fill 1D array with values used for the realisation.
    # These values are the k (but not k⃗) dependent values inside the
    # inverse Fourier transform, not including any additional tensor
    # structure (the k factors K(k⃗)).
    nyquist = gridsize//2
    k2_max = 3*(nyquist - 1)**2  # Max |k⃗|² in grid units
    sqrt_power_common = get_buffer(k2_max + 1,
        # Must use some buffer different from the one used to do the
        # domain decomposition of ψⁱ below.
        0,
    )
    if options['structure'] == 'nonlinear':
        # When using the non-linear structure of δϱ to do
        # the realisations, we need the transfer function of δϱ,
        # which is just ϱ_bar times the transfer function of δ.
        transfer_spline_δ, cosmoresults_δ = compute_transfer(component, 0, gridsize, a=a)
    for k2 in range(1, k2_max + 1):
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        transfer = transfer_spline.eval(k_magnitude)
        with unswitch:
            if options['structure'] == 'primordial':
                # Realise using ℱₓ⁻¹[T(k) ζ(k) K(k⃗) ℛ(k⃗)],
                # with K(k⃗) capturing any tensor structure.
                # The k⃗-independent part needed here is T(k)ζ(k),
                # with T(k) the supplied transfer function and ζ(k) the
                # primordial curvature perturbations.
                # The remaining ℛ(k⃗) is the primordial noise.
                sqrt_power_common[k2] = (
                    # T(k)
                    transfer
                    # ζ(k)
                    *ζ(k_magnitude)
                    # Fourier normalization
                    *ℝ[boxsize**(-1.5)]
                )
            else:  # options['structure'] == 'nonlinear':
                # Realise using ℱₓ⁻¹[T(k)/T_δϱ(k) K(k⃗) ℱₓ[δϱ(x⃗)]],
                # with K(k⃗) capturing any tensor structure.
                # The k⃗-independent part needed here is T(k)/T_δϱ(k),
                # with T(k) the supplied transfer function and T_δϱ(k)
                # the transfer function of δϱ.
                sqrt_power_common[k2] = (
                    # T(k)
                    transfer
                    # 1/T_δϱ(k)
                    /transfer_spline_δ.eval(k_magnitude)*ℝ[1/ϱ_bar
                        # Normalization due to FFT + IFFT
                        *float(gridsize)**(-3)
                    ]
                )
    # At |k⃗| = 0, the power should be zero, corresponding to a
    # real-space mean value of zero of the realised variable.
    sqrt_power_common[0] = 0
    # Fetch a slab decomposed grid for storing the entirety of what is
    # to be inverse Fourier transformed.
    slab = get_fftw_slab(gridsize)
    # Fetch a slab decomposed grid for storing the structure
    slab_structure_name = 'slab_structure'
    if options['structure'] == 'primordial':
        if fourier_structure_caching.get('primordial'):
            slab_structure_name += '_primordial'
        slab_structure_info = {'structure': 'primordial'}
    elif options['structure'] == 'nonlinear':
        if is_selected(component, fourier_structure_caching):
            slab_structure_name += f'_nonlinear_{component.name}'
        slab_structure_info = {
            'structure': 'nonlinear',
            'component': component.name,
            'a'        : a,
            'use_gridˣ': use_gridˣ,
        }
    reuse_slab_structure = (
        slab_structure_infos.get((gridsize, slab_structure_name)) == slab_structure_info
    )
    slab_structure_infos[gridsize, slab_structure_name] = slab_structure_info
    slab_structure = get_fftw_slab(gridsize, slab_structure_name)
    # Repopulate the slab structure if we cannot reuse it
    if not reuse_slab_structure:
        if options['structure'] == 'primordial':
            # Populate slab_structure with primordial noise ℛ(k⃗)
            generate_primordial_noise(slab_structure)
        elif options['structure'] == 'nonlinear':
            # Populate slab_structure with ℱₓ[ϱ(x⃗)]
            masterprint(
                f'Extracting structure from ϱ{"ˣ" if use_gridˣ else ""} of {component.name} ...'
            )
            slab_decompose(
                component.ϱ.gridˣ_mv if use_gridˣ else component.ϱ.grid_mv,
                slab_structure,
            )
            fft(slab_structure, 'forward')
            masterprint('done')
    # Initialise index0 and index1.
    # The actual values are not important.
    index0 = index1 = 0
    # When multiple particle components are to be realised, it is
    # preferable to not do so "on top of each other", as this leads to
    # large early forces. Below we define particle_shift to be the
    # fraction of a grid cell the current particle component should be
    # shifted relative to the default realisation grid, in all
    # directions. For a total of 1 particle components, this will be 0.
    # For a total of 2 particle components, this will be -1/4 and +1/4,
    # for the first and second particle component, respectively. For 3
    # particle components, this will be -1/3, 0, 1/3, and so on.
    # Note that this shifting trick leads to anisotropies for 3 particle
    # components and above.
    particle_shift = 0
    if component.representation == 'particles':
        particle_components = [
            other_component for other_component in component.components_all
            if other_component.representation == 'particles'
        ]
        particle_shift = 1.0/len(particle_components)
        particle_shifts = (
            linspace(particle_shift/2, 1 - particle_shift/2, len(particle_components)) -  0.5
        )
        particle_index = particle_components.index(component)
        particle_shift = particle_shifts[particle_index]
        if particle_index > 1:
            masterwarn(
                'You are realising more than 2 particle components. '
                'Note that this will lead to anisotropies in the initial conditions.'
            )
    # The realised field will be interpolated onto the shifted particle
    # positions, using the interpolation order specified in the options.
    interpolation_order = options['interpolation']
    # Preparations for the Fourier slab loop.
    # The deconvolution order is special, as we only deconvolve if the
    # particles are shifted (i.e. not on top of the grid points) or if
    # the interpolation order is more than 2 (i.e. TSC and beyond). We
    # do it like this because interpolation orders beyond NGP and CIC
    # samples more than a single grid point even in the case where the
    # particles sit on top of the grid points.
    slab_ptr      = cython.address(slab          [:, :, :])
    structure_ptr = cython.address(slab_structure[:, :, :])
    deconv_order = interpolation_order*𝔹[particle_shift or interpolation_order > 2]
    # Loop over all fluid scalars of the fluid variable
    fluidvar = component.fluidvars[fluid_index]
    for multi_index in (
        fluidvar.multi_indices if specific_multi_index is None
        else [processed_specific_multi_index]
    ):
        # Determine rank of the tensor being realised (0 for scalar
        # (i.e. ϱ), 1 for vector (i.e. J), 2 for tensor (i.e. ς)).
        if fluid_index == 0 or isinstance(multi_index, str):
            # If multi_index is a str it is 'trace', which means that
            # 𝒫 is being realised.
            # If fluid_index is 0, ϱ is being realised.
            tensor_rank = 0
        else:
            # The multi_index is a tuple of indices
            tensor_rank = len(multi_index)
        # Extract individual indices from multi_index
        if tensor_rank > 0:
            index0 = multi_index[0]
        if tensor_rank > 1:
            index1 = multi_index[1]
        # Loop over the slab
        for index, ki, kj, kk, factor, θ in fourier_loop(
            gridsize, skip_origin=True, deconv_order=deconv_order,
        ):
            k2 = ℤ[ℤ[ℤ[kj**2] + ki**2] + kk**2]
            # The square root of the power at this |k⃗|, disregarding all
            # k⃗-dependent contributions (from the k factor and the
            # non-linear structure).
            sqrt_power = sqrt_power_common[k2]
            # Apply deconvolution
            sqrt_power *= factor
            # Populate slab according to the component
            # representation and tensor_rank.
            with unswitch(5):
                if 𝔹[component.representation == 'particles']:
                    # We are realising either the displacement field ψⁱ
                    # (for the positions) or the velocity field uⁱ (for
                    # the momenta). These are constructed from the δ and
                    # θ fields, respectively, with the vector k factor
                    # K(k⃗) = ±ikⁱ/k².
                    # For fluids, fluid_index distinguish between the
                    # different variables. For particle positions and
                    # momenta, the corresponding ψⁱ and uⁱ fields are
                    # both vector variables, and so we had to set
                    # fluid_index = 1 in both cases. To distinguish
                    # between particles and momenta (and hence get the
                    # sign in the k factor correct) we instead make use
                    # of the particle_var_name variable. Also, when
                    # realising momenta with
                    # 'velocities from displacements' True, we really
                    # want to realise ψⁱ, and so we need to use the k
                    # factor for positions.
                    k_index0 = (
                          (-𝔹[index0 == 0] & ki)
                        | (-𝔹[index0 == 1] & kj)
                        | (-𝔹[index0 == 2] & kk)
                    )
                    k_factor = ℝ[
                        {
                            ('pos', True ): +1,
                            ('pos', False): +1,
                            ('mom', True ): +1,  # use 'pos' k factor
                            ('mom', False): -1,
                        }[particle_var_name, options['velocitiesfromdisplacements']]
                        *boxsize/(2*π)
                    ]*k_index0/k2
                    slab_ptr[index    ] = ℝ[sqrt_power*k_factor]*(-structure_ptr[index + 1])
                    slab_ptr[index + 1] = ℝ[sqrt_power*k_factor]*(+structure_ptr[index    ])
                elif tensor_rank == 0:  # and component.representation == 'fluid'
                    # Realise δ or δ𝒫
                    slab_ptr[index    ] = sqrt_power*structure_ptr[index    ]
                    slab_ptr[index + 1] = sqrt_power*structure_ptr[index + 1]
                elif tensor_rank == 1:  # and component.representation == 'fluid'
                    # Realise uⁱ.
                    # For vectors we have a k factor of
                    # K(k⃗) = -ikⁱ/k².
                    k_index0 = (
                          (-𝔹[index0 == 0] & ki)
                        | (-𝔹[index0 == 1] & kj)
                        | (-𝔹[index0 == 2] & kk)
                    )
                    k_factor = -(ℝ[boxsize/(2*π)]*k_index0)/k2
                    slab_ptr[index    ] = ℝ[sqrt_power*k_factor]*(-structure_ptr[index + 1])
                    slab_ptr[index + 1] = ℝ[sqrt_power*k_factor]*(+structure_ptr[index    ])
                else:  # tensor_rank == 2 and component.representation == 'fluid'
                    # Realise ςⁱⱼ.
                    # For rank 2 tensors we
                    # have a k factor of
                    # K(k⃗) = 3/2(δⁱⱼ/3 - kⁱkⱼ/k²).
                    k_index0 = (
                          (-𝔹[index0 == 0] & ki)
                        | (-𝔹[index0 == 1] & kj)
                        | (-𝔹[index0 == 2] & kk)
                    )
                    k_index1 = (
                          (-𝔹[index1 == 0] & ki)
                        | (-𝔹[index1 == 1] & kj)
                        | (-𝔹[index1 == 2] & kk)
                    )
                    k_factor = ℝ[0.5*(index0 == index1)] - (1.5*k_index0*k_index1)/k2
                    slab_ptr[index    ] = ℝ[sqrt_power*k_factor]*structure_ptr[index    ]
                    slab_ptr[index + 1] = ℝ[sqrt_power*k_factor]*structure_ptr[index + 1]
        # Ensure nullified Nyquist planes and origin
        nullify_modes(slab, 'nyquist, origin')
        # Fourier transform the slabs to coordinate space.
        # Now the slabs store the realised grid.
        fft(slab, 'backward')
        # Populate the fluid grids for fluid components,
        # or create the particles via the Zel'dovich approximation
        # for particles.
        if component.representation == 'fluid':
            # Communicate the fluid realisation stored in the slabs to
            # the designated fluid scalar grid. This also populates the
            # ghost points.
            fluidscalar = fluidvar[multi_index]
            domain_decompose(slab, fluidscalar.gridˣ_mv if use_gridˣ else fluidscalar.grid_mv)
            # Transform the realised fluid variable to the actual
            # quantity used in the non-linear fluid equations.
            if fluid_index == 0:
                # δ → ϱ = ϱ_bar(1 + δ).
                # Print a warning if min(δ) < -1.
                δ_min = ထ
                ϱ_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                for index in range(component.size):
                    if ℝ[ϱ_ptr[index]] < δ_min:
                        δ_min = ℝ[ϱ_ptr[index]]
                    ϱ_ptr[index] = ϱ_bar*(1 + ℝ[ϱ_ptr[index]])
                δ_min = allreduce(δ_min, op=MPI.MIN)
                if δ_min < -1:
                    masterwarn(f'The realised ϱ of {component.name} has min(δ) = {δ_min:.4g} < -1')
            elif fluid_index == 1:
                Jⁱ_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                if options['compoundorder'] == 'nonlinear':
                    # uⁱ → Jⁱ = a**4(ρ + c⁻²P)uⁱ
                    #         = a**(1 - 3w_eff)(ϱ + c⁻²𝒫) * uⁱ
                    ϱ_ptr  = component.ϱ.gridˣ if use_gridˣ else component.ϱ.grid
                    𝒫_ptr  = component.𝒫.gridˣ if use_gridˣ else component.𝒫.grid
                    for index in range(component.size):
                        Jⁱ_ptr[index] *= ℝ[a**(1 - 3*w_eff)]*(
                            ϱ_ptr[index] + ℝ[light_speed**(-2)]*𝒫_ptr[index]
                        )
                else:
                    # uⁱ → Jⁱ = a**4(ρ + c⁻²P)uⁱ
                    #         = a**(1 - 3w_eff)(ϱ + c⁻²𝒫) * uⁱ
                    #         ≈ a**(1 - 3w_eff)ϱ_bar(1 + w) * uⁱ
                    for index in range(component.size):
                        Jⁱ_ptr[index] *= ℝ[a**(1 - 3*w_eff)*ϱ_bar*(1 + w)]
            elif fluid_index == 2 and multi_index == 'trace':
                # δP → 𝒫 = 𝒫_bar + a**(3*(1 + w_eff)) * δP
                #        = c²*w*ϱ_bar + a**(3*(1 + w_eff)) * δP
                𝒫_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                for index in range(component.size):
                    𝒫_ptr[index] = ℝ[light_speed**2*w*ϱ_bar] + ℝ[a**(3*(1 + w_eff))]*𝒫_ptr[index]
            elif fluid_index == 2:
                ςⁱⱼ_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                if options['compoundorder'] == 'nonlinear':
                    # σⁱⱼ → ςⁱⱼ = (ϱ + c⁻²𝒫) * σⁱⱼ
                    ϱ_ptr  = component.ϱ.gridˣ if use_gridˣ else component.ϱ.grid
                    𝒫_ptr  = component.𝒫.gridˣ if use_gridˣ else component.𝒫.grid
                    for index in range(component.size):
                       ςⁱⱼ_ptr[index] *= ϱ_ptr[index] + ℝ[light_speed**(-2)]*𝒫_ptr[index]
                else:
                    # σⁱⱼ → ςⁱⱼ = (ϱ + c⁻²𝒫) * σⁱⱼ
                    #           ≈ ϱ_bar(1 + w) * σⁱⱼ
                    for index in range(component.size):
                        ςⁱⱼ_ptr[index] *= ℝ[ϱ_bar*(1 + w)]
            # Continue with the next fluidscalar
            continue
        # Domain-decompose the realised field stored in the slabs.
        # This is either the displacement field ψⁱ or the velocity
        # field uⁱ. Importantly, here we have to use a different
        # buffer from the one already used by sqrt_power_common.
        ψⁱ = uⁱ = domain_decompose(slab, 1)
        # Determine and set the mass of the particles
        # if this is still unset.
        if component.mass == -1:
            # For species with varying mass, this is the mass at a = 1
            component.mass = ϱ_bar*boxsize**3/component.N
        # The current mass is the set mass at a = 1,
        # scaled according to w_eff(a).
        mass = a**(-3*w_eff)*component.mass
        # Below follows the Zel'dovich approximation
        # for particle components.
        dim = multi_index[0]
        pos   = component.pos
        posxˣ = component.posxˣ
        posyˣ = component.posyˣ
        poszˣ = component.poszˣ
        mom   = component.mom
        Δmom  = component.Δmom
        if particle_var_name == 'mom':
            if dim == 0:
                # This is the realisation of the x momenta, which should
                # be the first variable to be realised out of
                # {x momenta, y momenta, z momenta, x positions,
                # y positions, z positions}.
                # Position the particles at the grid points,
                # possibly shifted in accordance with particle_shift.
                domain_start_i = domain_layout_local_indices[0]*(uⁱ.shape[0] - ℤ[2*nghosts])
                domain_start_j = domain_layout_local_indices[1]*(uⁱ.shape[1] - ℤ[2*nghosts])
                domain_start_k = domain_layout_local_indices[2]*(uⁱ.shape[2] - ℤ[2*nghosts])
                indexˣ = 0
                for i in range(ℤ[uⁱ.shape[0] - ℤ[2*nghosts]]):
                    x_gridpoint = (
                        ℝ[domain_start_i + 0.5*cell_centered + particle_shift] + i
                    )*ℝ[boxsize/gridsize]
                    for j in range(ℤ[uⁱ.shape[1] - ℤ[2*nghosts]]):
                        y_gridpoint = (
                            ℝ[domain_start_j + 0.5*cell_centered + particle_shift] + j
                        )*ℝ[boxsize/gridsize]
                        for k in range(ℤ[uⁱ.shape[2] - ℤ[2*nghosts]]):
                            z_gridpoint = (
                                ℝ[domain_start_k + 0.5*cell_centered + particle_shift] + k
                            )*ℝ[boxsize/gridsize]
                            posxˣ[indexˣ] = x_gridpoint
                            posyˣ[indexˣ] = y_gridpoint
                            poszˣ[indexˣ] = z_gridpoint
                            indexˣ += 3
            # Assign dim'th momenta.
            # First we nullify it.
            for indexˣʸᶻ in range(dim, 3*component.N_local, 3):
                mom[indexˣʸᶻ] = 0
            if options['velocitiesfromdisplacements']:
                # Interpolate the displacement field ψⁱ onto the particle
                # (grid) positions and assign the displacements as
                # momenta using
                #   momⁱ = a*m*uⁱ,
                #     uⁱ = a*H*f*ψⁱ,
                # with f = H⁻¹Ḋ/D being the linear growth rate.
                f_growth = cosmoresults.growth_fac_f(a)
                interpolate_domaingrid_to_particles(ψⁱ, component, 'mom', dim, interpolation_order,
                    factor=a**2*H*f_growth*mass,
                )
            else:
                # Interpolate the velocity field uⁱ onto the particle
                # (grid) positions and assign the velocities as momenta
                # using
                #   momⁱ = a*m*uⁱ
                interpolate_domaingrid_to_particles(uⁱ, component, 'mom', dim, interpolation_order,
                    factor=a*mass,
                )
        else:  # particle_var_name == 'pos'
            # Copy pos (currently containing the grid positions)
            #  into Δmom.
            for indexˣʸᶻ in range(dim, 3*component.N_local, 3):
                Δmom[indexˣʸᶻ] = pos[indexˣʸᶻ]
            # Apply displacement of dim'th positions by interpolating
            # the displacement field ψⁱ onto the particle (grid)
            # positions. The update is carried out on Δmom,
            # not pos, as this is needed for further interpolation.
            interpolate_domaingrid_to_particles(ψⁱ, component, 'Δmom', dim, interpolation_order)
            # After the z positions (dim == 2), the Δmom array contain
            # the fully displaced positions.
            # Copy these back to the pos array.
            if dim == 2:
                for indexʳ in range(3*component.N_local):
                    # Ensure toroidal boundaries
                    pos[indexʳ] = mod(Δmom[indexʳ], boxsize)
    # Done realising this variable
    masterprint('done')
    # After realising particles, most of them will be on the correct
    # process in charge of the domain in which they are located. Those
    # near the domain boundaries might however get displaced outside of
    # their original domain, and so we do need to do an exchange.
    # We can only do this exchange once both the momenta and the
    # positions have been assigned.
    if component.representation == 'particles' and (
            particle_var_name == 'pos'
        or (particle_var_name == 'mom' and options['velocitiesfromdisplacements'])
    ):
        exchange(component)
# Module level variable used by the realize() function
cython.declare(slab_structure_infos=dict)
slab_structure_infos = {}

# Function that populates the passed slab decomposed grid
# with primordial noise ℛ(k⃗).
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    # Locals
    dcplane='double[:, :, ::1]',
    dcplane_ptr='double*',
    face='int',
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    i_conj='Py_ssize_t',
    index='Py_ssize_t',
    index_dcplane='Py_ssize_t',
    index_dcplane_conj='Py_ssize_t',
    iterate='Py_ssize_t',
    j='Py_ssize_t',
    j_global='Py_ssize_t',
    j_global_conj='Py_ssize_t',
    k='Py_ssize_t',
    ki='Py_ssize_t',
    ki_start='Py_ssize_t',
    ki_step='Py_ssize_t',
    ki_stop='Py_ssize_t',
    kj='Py_ssize_t',
    kj_start='Py_ssize_t',
    kj_step='Py_ssize_t',
    kj_stop='Py_ssize_t',
    kk='Py_ssize_t',
    kk_start='Py_ssize_t',
    kk_step='Py_ssize_t',
    kk_stop='Py_ssize_t',
    noise_im='double',
    noise_re='double',
    nyquist='Py_ssize_t',
    prng='PseudoRandomNumberGenerator',
    r='double',
    shell='Py_ssize_t',
    slab_ptr='double*',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_size_k='Py_ssize_t',
    text=list,
    θ='double',
    θ_str=str,
    returns='void',
)
def generate_primordial_noise(slab):
    """Given the already allocated slab, this function will populate
    it with Gaussian pseudo-random numbers, the stream of which is
    controlled by the random_seed parameter. The slab grid is thought of
    as being in Fourier space, and so these are complex numbers. We wish
    the variance of these complex numbers to equal unity, and so their
    real and imaginary parts are drawn from a distribution
    with variance 1/√2, corresponding to
      noise_re = gaussian(1/√2)
      noise_im = gaussian(1/√2)
    As we furhter want to allow for fixing of the amplitude and shifting
    of the phase, we instead draw the numbers in the following
    equivalent manner:
      r = rayleigh(1/√2]
      θ = uniform(0, 2π)
      noise_re = r*cos(θ)
      noise_im = r*sin(θ)
    The 3D sequence of random numbers should be independent on the size
    of the grid, in the sense that increasing the grid size should
    amount to just populating the additional "shell" with new random
    numbers, but keeping the random numbers inside of the inner cuboid
    the same. This has the effect that enlarging the grid leaves the
    large-scale structure invariant; one merely adds information at
    smaller scales. Additionally, the sequence of random numbers should
    be independent on the number of processes. To achieve all of this,
    we draw the random numbers using the following scheme:
    All processes loop over the entire 3D grid in shells, starting from
    the inner most (shell = 1) and moving outwards (shell 2, 3, ...).
    Each shell consists of 5 faces:
      The kk = const face : kk =  shell, -shell ≤ ki ≤ shell, -shell ≤ kj ≤ shell
      The ki = const faces: ki = ±shell, -shell ≤ kj ≤ shell, 0 ≤ kk < shell
      The kj = const faces: kj = ±shell,  shell < ki < shell, 0 ≤ kk < shell
    In principle a shell = 0 also exists, containing just the origin
    ki = kj = kk = 0. With only a single point, the division into the 5
    faces cannot be done. We skip this shell, meaning that the origin
    will not be populated (nor will it be set to 0). The largest shell
    populated will be nyquist - 1, meaning that the three Nyquist planes
    will not be populated (nor will they be set to 0).
    At each point, all processes draw the same two random numbers
    (amounting to the real and imaginary part of the complex number),
    but only the process which owns the given point (determined by the j
    index that goes with kj) assigns the random numbers to its local
    slab.
    The z DC plane (kk = 0) needs to satisfy the complex conjugacy
    symmetry of a Fourier transformed real field, here
        plane[+kj, +ki, kk=0] = plane[-kj, -ki, kk=0]*,
    where * means complex conjugation. We enfore this symmetry by
    letting all processes tabulate this plane with random numbers in its
    entirety. After the whole 3D grid and the DC plane is filled with
    random numbers, we enforce the symmetry by looping over half of the
    DC plane and setting the inverted points in the slab equal to their
    complex conjucate partners.
    """
    slab_size_j, slab_size_i, slab_size_k = asarray(slab).shape
    gridsize = slab_size_i
    # Progress message
    text = ['Generating primordial']
    if not primordial_amplitude_fixed:
        text.append(' Gaussian')
    text.append(f' noise of grid size {gridsize}')
    if primordial_amplitude_fixed:
        text.append(', fixed amplitude')
    if primordial_phase_shift != 0:
        if isclose(primordial_phase_shift, π):
            θ_str = 'π'
        else:
            θ_str = str(primordial_phase_shift)
        text.append(f', phase shift {θ_str}')
    text.append(' ...')
    masterprint(''.join(text))
    # Allocate the entire z DC plane on all processes
    dcplane = empty((gridsize, gridsize, 2), dtype=C2np['double'])
    # Instantiate pseudo-random number generator
    # using the same seed on all processes.
    prng = PseudoRandomNumberGenerator(random_seed)
    # Extract pointers to the 3D slab and the (complex) 2D plane
    slab_ptr = cython.address(slab[:, :, :])
    dcplane_ptr = cython.address(dcplane[:, :, :])
    # Loop through all shells
    nyquist = gridsize//2
    for shell in range(1, nyquist):
        # Loop over the three types of faces
        for face in range(3):
            if face == 0:
                # The kk = const face
                ki_start, ki_stop, ki_step = ℤ[-shell], ℤ[shell + 1], 1  # -shell ≤ ki ≤ shell
                kj_start, kj_stop, kj_step = ℤ[-shell], ℤ[shell + 1], 1  # -shell ≤ kj ≤ shell
                kk_start, kk_stop, kk_step =    shell , ℤ[shell + 1], 1  #          kk = shell
            elif face == 1:
                # The two ki = const faces
                ki_start, ki_stop, ki_step = ℤ[-shell], ℤ[shell + 1], ℤ[2*shell]  #          ki = ±shell
                kj_start, kj_stop, kj_step = ℤ[-shell], ℤ[shell + 1],         1   # -shell ≤ kj ≤ shell
                kk_start, kk_stop, kk_step =        0 ,   shell     ,         1   #      0 ≤ kk < shell
            else:  # face == 2:
                # The two kj = const faces
                ki_start, ki_stop, ki_step = ℤ[-shell + 1],   shell     ,         1   # shell < ki < shell
                kj_start, kj_stop, kj_step = ℤ[-shell    ], ℤ[shell + 1], ℤ[2*shell]  #         kj = ±shell
                kk_start, kk_stop, kk_step =            0 ,   shell     ,         1   #     0 ≤ kk < shell
            # Loop over the face.
            # We want to loop like
            #   for kj in range(kj_start, kj_stop, kj_step)
            # but at least in Cython 0.29 such a loop transpiles to
            # unoptimized code (the step value needs to be known at
            # cythonisation time for proper transpilation). Below we
            # write out this loop by manually initialising and
            # incrementing the loop variable.
            kj = kj_start - kj_step
            for iterate in range(ℤ[((kj_stop - kj_start) + (kj_step - 1))//kj_step]):
                kj += kj_step
                j_global = kj + gridsize*(kj < 0)
                j = j_global - ℤ[slab_size_j*rank]
                ki = ℤ[ki_start - ki_step]
                for iterate in range(ℤ[((ki_stop - ki_start) + (ki_step - 1))//ki_step]):
                    ki += ki_step
                    i = ki + gridsize*(ki < 0)
                    kk = ℤ[kk_start - kk_step]
                    for iterate in range(ℤ[((kk_stop - kk_start) + (kk_step - 1))//kk_step]):
                        kk += kk_step
                        # Generate Gaussian random noise.
                        # In order to ensure the same random stream
                        # for both r and θ across simulations,
                        # we draw r even in the case
                        # of fixed amplitude.
                        r = prng.rayleigh(1/sqrt(2))
                        with unswitch:
                            if primordial_amplitude_fixed:
                                r = 1
                        θ = prng.uniform(0, 2*π)
                        with unswitch:
                            if primordial_phase_shift:
                                θ += primordial_phase_shift
                        noise_re = r*cos(θ)
                        noise_im = r*sin(θ)
                        # Populate the local slab
                        with unswitch(2):
                            if 0 <= j < slab_size_j:
                                k = 2*kk
                                index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                                slab_ptr[index    ] = noise_re
                                slab_ptr[index + 1] = noise_im
                        # Populate the z DC plane
                        if kk == 0:
                            index_dcplane = ℤ[(ℤ[j_global*gridsize] + i)*2]  # k = 0
                            dcplane_ptr[index_dcplane    ] = noise_re
                            dcplane_ptr[index_dcplane + 1] = noise_im
    # Enforce the complex conjugacy symmetry on the z DC plane of the
    # slabs. We do this by looping over half of the DC plane,
    # specifically -nyquist < ki ≤ 0, -nyquist < kj < nyquist, with
    # points ki = 0, 0 ≤ kj skipped. The inverted points at (-ki, -kj)
    # from the DC plane are conjugated and copied onto the slab.
    ki_start          = -nyquist + 1
    kj_start, kj_stop = -nyquist + 1, nyquist
    for kj in range(kj_start, kj_stop):
        j_global = kj + gridsize*(kj < 0)
        j = j_global - ℤ[slab_size_j*rank]
        # Each process can only change their local slab
        if not (0 <= j < slab_size_j):
            continue
        j_global_conj = -kj + gridsize*(-kj < 0)
        ki_stop = 1 - (kj >= 0)
        for ki in range(ki_start, ki_stop):
            i      =  ki + gridsize*( ki < 0)
            i_conj = -ki + gridsize*(-ki < 0)
            # Enforce conjugate symmetry for the slab
            index = (ℤ[j*slab_size_i] + i)*slab_size_k  # k = 0
            index_dcplane_conj = (ℤ[j_global_conj*gridsize] + i_conj)*2  # k = 0
            slab_ptr[index    ] = +dcplane_ptr[index_dcplane_conj    ]
            slab_ptr[index + 1] = -dcplane_ptr[index_dcplane_conj + 1]
    masterprint('done')

# Function returning the linear power spectrum of a given component
@cython.pheader(
    # Arguments
    component_or_components=object, # Component or list of Components
    k_magnitudes='double[::1]',
    a='double',
    gauge=str,
    power='double[::1]',
    # Locals
    component='Component',
    components=list,
    cosmoresults=object,  # CosmoResults
    gauge_cached=str,
    gridsize='Py_ssize_t',
    gridsize_max='Py_ssize_t',
    i='Py_ssize_t',
    k_magnitude='double',
    k_max='double',
    linear_component='Component',
    δ='double',
    δ_spline='Spline',
    returns='double[::1]',
)
def get_linear_powerspec(component_or_components, k_magnitudes, a=-1, gauge='N-body', power=None):
    """The linear power spectrum is only returned to the master process.
    """
    if isinstance(component_or_components, list):
        components = component_or_components
    else:
        components = [component_or_components]
    if a == -1:
        a = universals.a
    gauge = gauge.replace('-', '').lower()
    # Instantiate fake component with the CLASS species defined
    # as the sum of all CLASS species of the passed components.
    component = components[0]
    linear_component = type(component)(
        '',
        None,
        gridsize=2,
        boltzmann_order=-1,
        class_species='+'.join([component.class_species for component in components]),
        boltzmann_closure='class',
    )
    linear_component.name = 'linear power spectrum'
    # Get grid size for linear perturbation computation. In an attempt
    # to not rerun CLASS, we reuse any existing CosmoResults object,
    # even if this has too small a grid size, in which case the largest
    # k modes will be filled with NaN values.
    gridsize_max = -1
    for (gridsize, gauge_cached), cosmoresults in cosmoresults_cache.items():
        if gauge_cached != 𝕊['synchronous' if gauge == 'nbody' else gauge]:
            continue
        if gridsize > gridsize_max:
            gridsize_max = gridsize
    gridsize = gridsize_max
    if gridsize == -1:
        gridsize = np.max([component.powerspec_upstream_gridsize for component in components])
    # Get spline of δ transfer function for the fake component
    δ_spline, cosmoresults = compute_transfer(
        linear_component, 0, gridsize, a=a, gauge=gauge,
    )
    # Only the master process will return the linear power spectrum
    if not master:
        return power
    # Compute linear power (ζ*δ)**2
    if power is None:
        power = empty(k_magnitudes.shape[0], dtype=C2np['double'])
    k_max = δ_spline.x[len(δ_spline.x) - 1]
    for i in range(k_magnitudes.shape[0]):
        k_magnitude = k_magnitudes[i]
        if k_magnitude > k_max:
            power[i:] = NaN
            break
        δ = δ_spline.eval(k_magnitude)
        power[i] = (ζ(k_magnitude)*δ)**2
    return power

# The primordial curvature perturbation, parametrised by parameters
# in the primordial_spectrum dict.
@cython.header(
    # Arguments
    k='double',
    # Locals
    returns='double',
)
def ζ(k):
    # The parametrisation looks like
    # ζ(k) = π*sqrt(2*A_s)*k**(-3/2)*(k/pivot)**((n_s - 1)/2)
    #        *exp(α_s/4*log(k/pivot)**2).
    # See the primordial_analytic_spectrum() function in the CLASS
    # primordial.c file for a reference.
    return (
        ℝ[
            π*sqrt(2*primordial_spectrum['A_s'])
            *(1/primordial_spectrum['pivot'])**((primordial_spectrum['n_s'] - 1)/2)
        ]
        *k**ℝ[primordial_spectrum['n_s']/2 - 2]
        *exp(ℝ[primordial_spectrum['α_s']/4]*(log(k) - ℝ[log(primordial_spectrum['pivot'])])**2)
    )

# Function for registering species
def register_species(
    name, class_species, nicknames=None, *,
    w=None, Γ=0, logs=None, source_continuity=0, is_physical=True,
):
    """Calling this function will register a species globally.
    The arguments are:
    - name: Canonical CO𝘕CEPT name of the species. May contain spaces
      and should be lower-case.
    - class_species: Corresponding name used within CLASS, e.g. "b"
      for baryons. Adding species with "+", i.e. "b+cdm" for matter.
    - nicknames: List of alternative CO𝘕CEPT names which may be used
      to refer to this species. E.g. "dark matter"
      for "cold dark matter".
    - w: Equation of state parameter if this is a constant. If not a
      constant, leave it as None and it will be obtained from CLASS.
    - Γ: Function taking in the arguments (cosmoresults, a) and
      returning the decay rate of this species. Note that this is
      negative for species which acts as a sink for a decaying species.
    - logs: Dictionary of the form
      {'rho': (logx, logy), 'p': (logx, logy)}
      with each logx and logy a boolean specifying whether splines of
      the background density and pressure (as function of the scale
      factor) should be carried out logarithmically or not.
    - source_continuity: Function taking in the arguments
      (cosmoresults, a) and returning any source term in the homogeneous
      proper time continuity equation for the species.
    - is_physical: Boolean specifying whether this species is an actual
      physcial species or some fictitious species.
    """
    # Canonicalize the CLASS species if this is
    # a sum of fundamental CLASS species.
    class_species = '+'.join(sorted([
        class_species_fundamental.strip()
        for class_species_fundamental in class_species.split('+')
    ]))
    # Prepend the canonical name and append the CLASS name
    # to list of nicknames.
    if nicknames is None:
        nicknames = []
    nicknames.append(name)
    nicknames.append(class_species)
    nicknames = [  # remove duplicates and empty nicknames
        nickname for nickname in dict.fromkeys(nicknames) if nickname
    ]
    # Transform Γ to function
    if isinstance(Γ, (int, float)):
        Γ = (lambda cosmoresults, a, Γ=Γ: Γ)
    # Default log behaviour
    if logs is None:
        if w == 0:
            logs = {'rho': (True, True), 'p': (True, False)}
        elif w == 1/3:
            logs = {'rho': (True, True), 'p': (True, True)}
        else:
            logs = {'rho': (None, None), 'p': (None, None)}
    # Transform source_continuity to function
    if isinstance(source_continuity, (int, float)):
        source_continuity = (lambda cosmoresults, a, source_continuity=source_continuity: source_continuity)
    # Pack the information into a SpeciesInfo instance
    species_info = SpeciesInfo(
        name, class_species, nicknames, w, Γ, logs, source_continuity, is_physical,
    )
    # Store the species info globally
    if name in species_registered:
        abort(f'Multiple species registrations under the same name "{name}"')
    species_registered[name] = species_info
    return species_info
# Create the SpeciesInfo type used in the above function
SpeciesInfo = collections.namedtuple(
    'SpeciesInfo', (
        'name', 'class_species', 'nicknames',
        'w', 'Γ', 'logs', 'source_continuity', 'is_physical',
    ),
)
# Global dict-like container of species infos
# populated by the above function.
class SpeciesRegisteredDict(dict):
    def __getitem__(self, key):
        key = key.strip()
        # Attempt normal lookup
        try:
            return super().__getitem__(key)
        except KeyError:
            pass
        match = re.search(r'(.+?) *(\d+)$', key)
        if not match:
            # Trigger the exception again
            super().__getitem__(key)
        # Key is a numbered species, e.g. "neutrino 0".
        # Lookup base species.
        key_base, n = match.group(1), int(match.group(2))
        species_info = super().__getitem__(key_base)
        # Insert and return proper SpeciesInfo instance
        class_species = re.sub(r'\[.*?\]', '', species_info.class_species.split('+')[0]) + f'{[n]}'
        nicknames = [f'{nickname} {n}' for nickname in species_info.nicknames]
        nicknames[-1] = class_species
        species_info = SpeciesInfo(
            key, class_species, nicknames,
            species_info.w, species_info.Γ, species_info.logs,
            species_info.source_continuity, species_info.is_physical,
        )
        self[key] = species_info
        return species_info
    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True
    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default
cython.declare(species_registered=object)
species_registered = SpeciesRegisteredDict()

# Function for registering transfer functions / perturbations
def register_perturbation(
    name, name_class, name_latex=None, name_ascii=None,
    units_class=1, units_latex='',
    weighting='1',
):
    if not name.isidentifier():
        abort(
            f'Transfer function name "{name}" is illegal as it is not a valid Python identifier'
        )
    if name_latex is None:
        name_latex = rf'\mathrm{{{name_class}}}'.replace('_', ' ')
    if name_ascii is None:
        name_ascii = name_class
    name_ascii = asciify(name_ascii)
    units_class = float(units_class)
    units_latex = (units_latex
        .replace('length', rf'\mathrm{{{unit_length}}}')
        .replace('time', rf'\mathrm{{{unit_time}}}')
        .replace('mass', unit_mass.replace('1e+10*', '10^{10}').replace('m_sun', r'm_{\odot}'))
    )
    weighting = weighting.replace(' ', '')
    if weighting not in {'1', 'ρ', 'ρ+P', 'δρ'}:
        abort(f'Transfer function weighting "{weighting}" not implemented')
    # The 'total' flag, specifying whether this transfer function
    # belongs to the universe as a whole or exists
    # for several species individually.
    total = ('{}' not in name_class)
    # Pack the information into a TransferFunctionInfo instance
    transferfunction_info = TransferFunctionInfo(
        name, name_class, name_latex, name_ascii,
        units_class, units_latex,
        weighting, total,
    )
    # Store the transfer function info globally
    if name in transferfunctions_registered:
        abort(f'Multiple transfer function registrations under the same name "{name}"')
    transferfunctions_registered[name] = transferfunction_info
    return transferfunction_info
# Create the TransferFunctionInfo type used in the above function
TransferFunctionInfo = collections.namedtuple(
    'TransferFunctionInfo',
    (
        'name', 'name_class', 'name_latex', 'name_ascii',
        'units_class', 'units_latex',
        'weighting', 'total',
    ),
)
# Global dict of transfer function infos populated by the above function
cython.declare(transferfunctions_registered=dict)
transferfunctions_registered = {}



# Register all implemented species
register_species(
    'baryon', 'b', ['baryons', 'baryonic', 'baryonic matter'], w=0,
)
register_species(
    'cold dark matter', 'cdm', ['dark matter', 'dm'], w=0,
)
register_species(
    'matter', matter_class_species, ['total matter', 'm'], w=0,
)
register_species(
    'photon', 'g', ['photons', 'gamma', unicode('γ'), asciify('γ')], w=1/3,
)
register_species(
    'massless neutrino', 'ur', ['massless neutrinos'], w=1/3,
)
register_species(
    'massive neutrino', massive_neutrino_class_species, ['massive neutrinos', 'ncdm'],
    logs={'rho': (True, True), 'p': (True, True)},
)
for massive_neutrino_class_species_single in massive_neutrino_class_species.split('+'):
    match = re.search(r'\[(.*?)\]', massive_neutrino_class_species_single)
    if not match:
        continue
    nν = match.group(1)
    register_species(
        f'massive neutrino[{nν}]',
        massive_neutrino_class_species_single,
        [f'massive neutrinos[{nν}]', f'ncdm[{nν}]'],
        logs={'rho': (True, True), 'p': (True, True)},
    )
register_species(
    'neutrino', neutrino_class_species, ['neutrinos', 'nu', unicode('ν'), asciify('ν')],
    logs={'rho': (True, True), 'p': (True, True)},
)
register_species(
    'radiation', radiation_class_species, ['rad', 'r'], w=1/3,
)
register_species(
    'cosmological constant', 'lambda', [unicode('Λ'), asciify('Λ'), 'Lambda'], w=-1,
    logs={'rho': (True, False), 'p': (True, False)},
)
register_species(
    'dark energy', 'fld', ['dark energy fluid', 'dynamical dark energy'],
    logs={'rho': (False, False), 'p': (False, False)},
)
register_species(
    'decaying cold dark matter', 'dcdm', ['decaying dark matter', 'decaying matter', 'ddm'], w=0,
    Γ=(lambda cosmoresults, a: cosmoresults.Γ_dcdm),
    source_continuity=(lambda cosmoresults, a: -cosmoresults.Γ_dcdm*cosmoresults.ρ_bar(a, 'dcdm')),
)
register_species(
    'decay radiation', 'dr', ['dark radiation'], w=1/3,
    Γ=(lambda cosmoresults, a: -cosmoresults.ρ_bar(a, 'dcdm')/cosmoresults.ρ_bar(a, 'dr')*cosmoresults.Γ_dcdm),
    source_continuity=(lambda cosmoresults, a: +cosmoresults.Γ_dcdm*cosmoresults.ρ_bar(a, 'dcdm')),
)
register_species(
    'metric', 'metric', logs={'rho': (True, True), 'p': (True, True)}, is_physical=False,
)
register_species(
    'lapse', 'lapse', logs={'rho': (True, True), 'p': (True, True)}, is_physical=False,
)
register_species(
    'none', 'none', is_physical=False,
)

# Mapping from allowed species specifications to their canonical names
cython.declare(species_canonical=dict)
species_canonical = {}
for name, species_info in species_registered.items():
    for nickname in species_info.nicknames:
        if nickname in species_canonical:
            species_info2 = species_registered[species_canonical[nickname]]
            if species_info.class_species != species_info2.class_species:
                abort(
                    f'Both "{species_info2.name}" and "{name}" referred to '
                    f'as "{nickname}", but they relate to different CLASS species '
                    f'("{species_info2.class_species}" and "{species_info.class_species}", '
                    f'respectively)'
                )
            continue
        species_canonical[nickname] = name

# Register all implemented perturbations
register_perturbation(
    'δ', 'delta_{}', r'{\delta}',
    weighting='ρ',
)
register_perturbation(
    'θ', 'theta_{}', r'{\theta}',
    units_class=light_speed/units.Mpc,
    units_latex=r'time^{-1}',
    weighting='ρ + P',
)
register_perturbation(  # δP from cs2 = c⁻²δP/δρ
    'δP', 'cs2_{}', r'{\delta}P', 'deltaP_{}',
    units_class=light_speed**2,
    units_latex=r'mass\, length^{-1}\, time^{-2}',
    weighting='δρ',
)
register_perturbation(
    'σ', 'shear_{}', r'{\sigma}', 'sigma_{}',
    units_class=light_speed**2,
    units_latex=r'length^2\, time^{-2}',
    weighting='ρ + P',
)
register_perturbation(
    'θ_tot', 'theta_tot', r'{\theta}_{\mathrm{tot}}',
    units_class=light_speed/units.Mpc,
    units_latex=r'time^{-1}',
)
register_perturbation(
    'ϕ', 'phi', r'{\phi}',
    units_class=light_speed**2,
    units_latex=r'length^2\, time^{-2}',
)
register_perturbation(
    'ψ', 'psi', r'{\psi}',
    units_class=light_speed**2,
    units_latex=r'length^2\, time^{-2}',
)
register_perturbation(
    'hʹ', 'h_prime', r'h^{\prime}',
    units_class=light_speed/units.Mpc,
    units_latex=r'time^{-1}',
)
register_perturbation(
    'H_Tʹ', 'H_T_prime', r'H_{\mathrm{T}}^{\prime}',
    units_class=light_speed/units.Mpc,
    units_latex=r'time^{-1}',
)

# Create class_extra_perturbations_class, a version of
# class_extra_perturbations with keys equal to the CLASS names.
# Also perform automatic (and poor) registering of CLASS unknown
# perturbations found in class_extra_perturbations.
cython.declare(class_extra_perturbations_class=set)
class_extra_perturbations_class = set()
transferfunctions_registered_classnames = {
    transferfunction_info.name_class
    for transferfunction_info in transferfunctions_registered.values()
}
for class_extra_perturbation in class_extra_perturbations:
    for class_extra_perturbation_modified in (
        class_extra_perturbation,
        unicode(class_extra_perturbation),
        asciify(class_extra_perturbation),
    ):
        transferfunction_info = transferfunctions_registered.get(class_extra_perturbation_modified)
        if transferfunction_info is not None:
            class_extra_perturbations_class.add(transferfunction_info.name_class)
            break
    else:
        if class_extra_perturbation in transferfunctions_registered_classnames:
            class_extra_perturbations_class.add(class_extra_perturbation)
            continue
        # Unregistered perturbation from class_extra_perturbation.
        # Perform automatic registration, but throw a warning.
        masterwarn(
            f'Auto-registering CLASS perturbation "{class_extra_perturbation}". '
            f'Consider placing a proper registration in linear.py.'
        )
        perturbation_name = ''.join([
            char for char in class_extra_perturbation if f'x{char}'.isidentifier()
        ])
        for i, char in enumerate(perturbation_name):
            if char.isidentifier():
                perturbation_name = perturbation_name[i:]
                break
        else:
            perturbation_name = ''
        if not perturbation_name:
            abort(
                f'Failed to generate {esc_concept} name '
                f'for CLASS perturbation "{class_extra_perturbation}"'
            )
        transferfunction_info = register(perturbation_name, class_extra_perturbation)
        class_extra_perturbations_class.add(transferfunction_info.name_class)

# Read in definitions from CLASS source files at import time
cython.declare(
    class__VERSION_=str,
    class__ARGUMENT_LENGTH_MAX_='Py_ssize_t',
    class_a_min='double',
    k_gridsize_max='Py_ssize_t',
)
for (varname,
     filename,
     declaration_type,
     default_value) in [('_VERSION_'            , 'include/common.h'      , 'macro'   , ''   ),
                        ('_ARGUMENT_LENGTH_MAX_', 'include/parser.h'      , 'macro'   , 10000),
                        ('a_min'                , 'source/perturbations.c', 'variable', 0.001),
                        ]:
    if master:
        if declaration_type == 'macro':
            pattern = rf'(^|[^0-9a-zA-Z_])#define\s+{varname}\s+(.+?)(/\*| |//|;|\n|$)'
        elif declaration_type == 'variable':
            pattern = rf'(^|[^0-9a-zA-Z_]){varname}\s*=\s*(.*?)(/\*| |//|;|\n|$)'
        filename_abs = rf'{path.class_dir}/{filename}'
        try:
            with open_file(filename_abs, mode='r') as class_file:
                value = type(default_value)(re.search(pattern, class_file.read())
                                            .group(2).strip('"'))
        except:
            masterwarn(f'Failed to read value of {varname} from {filename_abs}')
            value = default_value
    value = bcast(value if master else None)
    if varname == '_VERSION_':
        class__VERSION_ = value
    elif varname == '_ARGUMENT_LENGTH_MAX_':
        class__ARGUMENT_LENGTH_MAX_ = value
        # This is the maximum number of k modes that CLASS can handle
        k_gridsize_max = (class__ARGUMENT_LENGTH_MAX_ - 1)//(len(k_float2str(0)) + 1)
    elif varname == 'a_min':
        class_a_min = -1.0 if special_params.get('special') == 'class' else value
