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



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from communication import partition,                   '
        '                          domain_layout_local_indices, '
        '                          exchange,                    '
        '                          get_buffer,                  '
        '                          smart_mpi,                   '
        )
cimport('from graphics import plot_detrended_perturbations')
cimport('from integration import Spline, remove_doppelgängers, hubble, ȧ, ä')
cimport('from mesh import get_fftw_slab,       '
        '                 domain_decompose,    '
        '                 slab_decompose,      '
        '                 fft,                 '
        )



# Class storing a classy.Class instance
# together with the corresponding |k| values
# and results retrieved from the classy.Class instance.
class CosmoResults:
    # Only part of the computed CLASS data is needed. Below, the keys
    # corresponding to the needed fields of CLASS data is written as
    # regular expressions.
    needed_keys = {
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
        # Species specific perturbations
        # will be added later.
        'perturbations': {
            # Time
            r'^a$',
            r'^tau \[Mpc\]$',
            # Other
            r'^h_prime$',
            r'^theta_tot$',
        },
        # Transfer functions at specific times as function
        # of k. These are no longer used by the code.
        'tranfers': {
            # Space
            r'^k \(h/Mpc\)$',
            # Densities
            r'^d_',
            # Velocity
            r'^t_',
            # Other
            r'^h_prime$',
        }
    }
    # Names of all implemented transfer function variables.
    # Functions with these names will be defined, which will return
    # the corresponding transfer function as a function of k,
    # for a given a.
    transfer_function_variable_names = ('δ', 'θ', 'δP', 'σ', 'hʹ')
    # Initialize instance
    def __init__(self, params, k_magnitudes, cosmo=None, filename=''):
        """If no cosmo object is passed, all results should be loaded
        from disk, if possible. The first time this fails, CLASS will be
        called and a cosmo object will be produced.
        All methods of the cosmo object used in the code which have
        no arguments are here written as attritubes using the magick of
        the property decorator. Methods with arguments should also be
        defined in such a way that their results are cached.
        If a filename is passed, CLASS data will be read from this file.
        Nothing will however be saved to this file.
        """
        # Store the supplied objects
        self.params = params
        self.k_magnitudes = k_magnitudes
        # Store the cosmo object as a hidden attribute
        self._cosmo = cosmo
        # Names of scalar attributes
        self.attribute_names = ('A_s', 'n_s', 'k_pivot', 'h')
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
            # object based on the hash of the CLASS parameters and the
            # CLASS variables _VERSION, _ARGUMENT_LENGTH_MAX_ and a_min.
            self.id = hashlib.sha1((str(tuple(sorted({
                str(key).replace(' ', '').lower(): str(val).replace(' ', '').lower()
                for key, val in self.params.items()
            }.items()
            )) + (
                class__VERSION_,
                class__ARGUMENT_LENGTH_MAX_,
                class_a_min,
            ))).encode()).hexdigest()
            self.filename = f'{paths["reusables_dir"]}/class/{self.id}.hdf5'
        # Add functions which returns transfer function splines
        # for a given a.
        def construct_func(var_name):
            return (
                lambda a, component=None, get='as_function_of_k':
                    self.transfer_function(a, component, var_name, get)
            )
        for var_name in self.transfer_function_variable_names:
            setattr(self, var_name, construct_func(var_name))
        # Initialize the hdf5 file on disk, if it does not
        # already exist. If it exist, 'params' and 'k_magnitudes' are
        # guarenteed to be stored there correctly already, as the
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
                self._cosmo, self.k_node_indices = call_class(self.params, mode='MPI')
            else:
                # Do not compute perturbations. This call should be
                # very fast and so we compute it in 'single node'
                # mode regardless of the number of nodes available.
                # (Also, MPI Class is not implemented for anything but
                # the perturbation computation).
                self._cosmo = call_class(self.params, mode='single node')
        return self._cosmo
    # Methods returning scalar attributes used in the CLASS run
    @property
    def A_s(self):
        if not hasattr(self, '_A_s'):
            if not self.load('A_s'):
                # Get A_s from CLASS
                self._A_s = self.cosmo.get_current_derived_parameters(['A_s'])['A_s']
                # Save to disk
                self.save('A_s')
            # Communicate
            self._A_s = bcast(self._A_s if master else None)
        return self._A_s
    @property
    def n_s(self):
        if not hasattr(self, '_n_s'):
            if not self.load('n_s'):
                # Get n_s from CLASS
                self._n_s = self.cosmo.get_current_derived_parameters(['n_s'])['n_s']
                # Save to disk
                self.save('n_s')
            # Communicate
            self._n_s = bcast(self._n_s if master else None)
        return self._n_s
    @property
    def k_pivot(self):
        if not hasattr(self, '_k_pivot'):
            if not self.load('k_pivot'):
                # Retrieve k_pivot from the CLASS params.
                # If not defined there, default to the standard CLASS
                # value of 0.05 Mpc⁻¹. We store this in CLASS units.
                self._k_pivot = self.params.get('k_pivot', 0.05)
                # Save to disk
                self.save('k_pivot')
            # Communicate
            self._k_pivot = bcast(self._k_pivot if master else None)
        # Remember to add the unit of Mpc⁻¹
        return self._k_pivot*units.Mpc**(-1)
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
                    self._background = {key: arr for key, arr in self._background.items()
                                        if any([re.search(pattern, key)
                                                for pattern in self.needed_keys['background']])}
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
                    self._background = {key: arr[index:].copy()
                                        for key, arr in self._background.items()}
                # Save to disk
                self.save('background')
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
            # CLASS does not give the background pressure for cold
            # dark matter, baryons, photons, ultra relativistic species
            # or the cosmological constant, as these are always
            # proportional to their densities with a constant
            # proportionality factor w. Here we add these missing
            # pressures explicitly.
            constant_eos_w = {
                'cdm'   : 0,
                'b'     : 0,
                'g'     : 1/3,
                'ur'    : 1/3,
                'lambda': -1,
                }
            for class_species, w in constant_eos_w.items():
                if (    f'(.)rho_{class_species}'   in self._background
                    and f'(.)p_{class_species}' not in self._background):
                    self._background[f'(.)p_{class_species}'] = (
                        w*self._background[f'(.)rho_{class_species}']
                    )
            # For the 'fld' CLASS species, '(.)p_fld' is never given.
            # For time varying equation of state, w is given
            # as '(.)w_fld', from which we construct '(.)p_fld'.
            # If neither '(.)p_fld' nor '(.)w_fld' is given, it means
            # that w = -1 throughout time.
            if '(.)rho_fld' in self._background:
                if '(.)w_fld' in self._background:
                    self._background['(.)p_fld'] = (
                        self._background['(.)w_fld']*self._background['(.)rho_fld']
                    )
                else:
                    self._background['(.)p_fld'] = (
                        -1*ones(self._background['(.)rho_fld'].shape, dtype=C2np['double'])
                    )
            # We also need to store the total background density.
            # Assuming a flat universe, we have rho_tot == rho_crit.
            if '(.)rho_crit' in self._background:
                self._background['(.)rho_tot'] = self._background['(.)rho_crit']
        return self._background
    # The raw perturbations
    @property
    def perturbations(self):
        if not hasattr(self, '_perturbations'):
            # Add species specific perturbation keys to the class
            # set self.needed_keys['perturbations'], based on the
            # species present in the current simulation.
            for class_species_present in (universals_dict['class_species_present']
                .decode().replace('[', '\[').replace(']', '\]').split('+')):
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
            if not self.load('perturbations'):
                # Get perturbations from CLASS
                masterprint('Extracting perturbations from CLASS ...')
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
                    # master process will ever store these
                    # extra perturbations. A copy of the data is used,
                    # making freeing of the original
                    # CLASS data possible.
                    self._perturbations = [
                        {
                            key: arr.copy()
                            for key, arr in perturbation.items()
                            if any([re.search(pattern, key) for pattern in (
                                self.needed_keys['perturbations'] | class_extra_perturbations   
                            )])
                         }
                         for perturbation in self._perturbations
                    ]
                    if len(self.k_magnitudes) > len(self.k_node_indices):
                        # The master process needs to know which
                        # process store which k modes.
                        if master:
                            k_processes_indices = empty(len(self.k_magnitudes), dtype=C2np['int'])
                            k_processes_indices[self.k_node_indices] = rank
                            for rank_recv in node_master_ranks:
                                if rank_recv == rank:
                                    continue
                                k_processes_indices[recv(source=rank_recv)] = rank_recv
                        else:
                            send(asarray(self.k_node_indices), dest=master_rank)
                        # Gather all perturbations into the
                        # master process. Communicate these as list
                        # of dicts mapping str's to arrays.
                        keys = sorted(list(self._perturbations[0].keys()))
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
                            # The master process now hold perturbations
                            # from all nodes in all_perturbations.
                            self._perturbations = all_perturbations
                        else:
                            for perturbation in self._perturbations:
                                for key in keys:
                                    send(len(perturbation[key]), dest=master_rank)
                                    Send(asarray(perturbation[key]), dest=master_rank)
                # Done extracting perturbations from CLASS
                masterprint('done')
                # Save to disk
                self.save('perturbations')
                # As perturbations comprise the vast majority of the
                # data volume of what is needed from CLASS, we might
                # as well read in any remaining bits and clean up
                # the C-space memory and delete any extra class
                # perturbations (which have now been saved to disk).
                self.background
                self.cosmo.struct_cleanup()
                if node_master:
                    for key in set(self._perturbations[0].keys()):
                        if not any([re.search(pattern, key)
                            for pattern in class_extra_perturbations]
                        ):
                            continue
                        if any([re.search(pattern, key)
                            for pattern in self.needed_keys['perturbations']]
                        ):
                            continue
                        # This key is for an extra class perturbation that
                        # is not used by this simulation.
                        for perturbation in self._perturbations:
                            del perturbation[key]
            # As we only need perturbations defined within the
            # simulation timespan, a > a_begin, we now cut off the
            # lower tail of all perturbations.
            if master:
                universals_a_begin = universals.a_begin
                for perturbation in self._perturbations:
                    a_values = perturbation['a']
                    # Find the index in a_values which corresponds to
                    # universals.a_begin, using a binary search.
                    index_lower = 0
                    index_upper = a_values.shape[0] - 1
                    a_lower = a_values[index_lower]
                    a_upper = a_values[index_upper]
                    if a_lower > universals_a_begin:
                        abort(
                            f'Not all perturbations are defined at '
                            f'a_begin = {universals_a_begin}. Note that CLASS perturbations '
                            f'earlier than a_min = {class_a_min} in source/perturbations.c will '
                            f'not be used. If you really want perturbations at earlier times, '
                            f'decrease this a_min.'
                        )
                    index = 0
                    while index_upper - index_lower > 1:
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
                    a_value = a_values[index]
                    # Remove perturbations earlier than a_begin.
                    # We have to copy the data, as otherwise the array
                    # owning will not be owning the data, meaning that
                    # it cannot be freed by Python's garbage collection.
                    for key, val in perturbation.items():
                        perturbation[key] = asarray(val[index:]).copy()
            # Communicate perturbations as list of dicts mapping
            # str's to arrays.
            size = bcast(len(self._perturbations) if master else None)
            if size:
                keys = bcast(tuple(self._perturbations[0].keys()) if master else None)
                if not master:
                    self._perturbations = [{} for _ in range(size)]
                for perturbation in self._perturbations:
                    for key in keys:
                        buffer = smart_mpi(perturbation[key] if master else (), mpifun='bcast')
                        if not master:
                            perturbation[key] = asarray(buffer).copy()
            else:
                self._perturbations = []
            # As perturbations comprise the vast majority of the
            # data volume of what is needed from CLASS, we might
            # as well read in any remaining bits. Specifically, the
            # background should be read, as some tasks around the
            # perturbations require knowledge of the background,
            # and the first read-in of the background has to be done
            # in parallel.
            self.background
        return self._perturbations
    # Method which constructs TransferFunction instances and use them
    # to compute and store transfer functions. Do not use this method
    # directly, but rather call e.g. cosmoresults.δ(a, component).
    # Note that the transfer functions returned by this method are those
    # gotten from get_perturbations, not get_transfer.
    def transfer_function(self, a, component, var_name, get='object'):
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
            return transfer_function.as_function_of_k(a)
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
            spline = Spline(self.background['a'], self.background[y])
            self._splines[y] = spline
        return spline
    # Method for looking up the background density of a given
    # component/species at some specific a. If no component/species
    # is given, the critical density is returned.
    def ρ_bar(self, a, component_or_class_species='crit'):
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
        values *= ℝ[3/(8*π*G_Newton)*(light_speed/units.Mpc)**2]
        return values
    # Method for looking up the background pressure of a given
    # component/species at some specific a. A component/species
    # has to be given
    def P_bar(self, a, component_or_class_species):
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
        values *= ℝ[3/(8*π*G_Newton)*(light_speed/units.Mpc)**2*light_speed**2]
        return values
    # Method for looking up f_growth = H⁻¹Ḋ/D (with D the linear
    # growth factor) at some a.
    @functools.lru_cache()
    def growth_fac_f(self, a):
        spline = self.splines('gr.fac. f')
        return spline.eval(a)
    # Method for appending a piece of raw CLASS data to the dump file
    def save(self, element, a=None):
        """You should not call this method unless you have good reason
        to believe that 'element' is not already present in the file,
        as this method will open the file in read/write ('a') mode
        regardless. This can be dangeous as HDF5 build with MPI is not
        thread-safe, and so if two running instances of CO𝘕CEPT with the
        same params run this function simultaneously, problems
        may arise. From HDF5 1.10 / H5Py 2.5.0, multiple processes can
        read from the same file, as long as it is not opened in write
        mode by any process. Thus, this complication is only relevent
        for this function. The open_hdf5 function is ment to alleviate
        this problem, but is has not been thoroughly tested.
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
                # Scalar attribute as attribute on a group
                attribute = getattr(self, element)
                if element == 'k_pivot':
                    # Convert to CLASS units
                    attribute /= units.Mpc**(-1)
                attributes_h5 = hdf5_file.require_group('attributes')
                attributes_h5.attrs[element.replace('/', '__per__')] = attribute
            elif element == 'k_magnitudes':
                # Save k_magnitudes in CLASS units (Mpc⁻¹)
                if self.k_magnitudes is not None and 'k_magnitudes' not in hdf5_file:
                    dset = hdf5_file.create_dataset('k_magnitudes',
                                                    (self.k_magnitudes.shape[0], ),
                                                    dtype=C2np['double'])
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
                # Save perturbations as
                # /perturbations/index/key.
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
                if not 'perturbations' in hdf5_file:
                    masterprint(f'Saving CLASS perturbations to "{self.filename}" ...')
                    perturbations_h5 = hdf5_file.create_group('perturbations')
                    for index, perturbation in enumerate(self.perturbations):
                        perturbation_h5 = perturbations_h5.create_group(str(index))
                        for key, val in perturbation.items():
                            key = key.replace('/', '__per__')
                            dset = perturbation_h5.create_dataset(key, (val.shape[0], ),
                                                                  dtype=C2np['double'])
                            dset[:] = val
                    masterprint('done')
            else:
                abort(f'CosmoResults.save was called with the unknown element of "{element}"')
            hdf5_file.flush()
    # Method for loading a piece of raw CLASS data from the dump file
    def load(self, element, a=None):
        """This method will attempt to load the element given.
        If successful, the element will be set on the instance and True
        will be returned by all processes.
        Otherwise, False will be returned by all processes.
        """
        if not class_reuse:
            return False
        if not master:
            return bcast(None)
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
                # Scalar attribute as attribute on a group
                attributes_h5 = hdf5_file.get('attributes')
                if attributes_h5 is None:
                    return bcast(False)
                attribute = attributes_h5.attrs.get(element.replace('/', '__per__'))
                if attribute is None:
                    return bcast(False)
                setattr(self, '_' + element, attribute)
            elif element == 'k_magnitudes':
                # Load k_magnitudes.
                # Remember to add CLASS units (Mpc⁻¹).
                k_magnitudes_h5 = hdf5_file.get('k_magnitudes')
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
                    if any([re.search(pattern, key.replace('__per__', '/'))
                        for pattern in self.needed_keys['background']])
                }
            elif element == 'perturbations':
                # Load perturbations stored as
                # /perturbations/index/name.
                perturbations_h5 = hdf5_file.get('perturbations')
                if perturbations_h5 is None:
                    return bcast(False)
                masterprint(f'Loading CLASS perturbations from "{self.filename}" ...')
                self._perturbations = [None]*len(self.k_magnitudes)
                # Check that the file contain perturbations at all
                # k modes. This is not the case if the process that
                # originally wrote the file ended prematurely. In this
                # case, no other error is necessarily detected.
                if len(perturbations_h5) < len(self._perturbations):
                    abort(
                        f'The file "{self.filename}" only contains perturbations for '
                        f'{len(perturbations_h5)} k modes, whereas it should contain '
                        f'perturbations for {len(self._perturbations)} k modes. '
                        f'This can happen if the creation of this file was ended prematurely. '
                        f'You should remove this file and rerun this simulation.'
                    )
                if len(perturbations_h5) > len(self._perturbations):
                    abort(
                        f'The file "{self.filename}" contains perturbations for '
                        f'{len(perturbations_h5)} k modes, whereas it should contain '
                        f'perturbations for {len(self._perturbations)} k modes. '
                        f'I cannot explain this mismatch, and I cannot use these perturbations.'
                    )
                # Load the perturbations
                for index, d in perturbations_h5.items():
                    self._perturbations[int(index)] = {
                        key.replace('__per__', '/'): dset[...]
                        for key, dset in d.items()
                        if any([re.search(pattern, key.replace('__per__', '/'))
                            for pattern in self.needed_keys['perturbations']])
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
                    for perturbation_missing in self.needed_keys['perturbations']
                    if not any([re.search(perturbation_missing, key)
                        for key in perturbations_loaded])
                }
                for class_species_present in (universals_dict['class_species_present']
                    .decode().replace('[', '\[').replace(']', '\]').split('+')):
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
    # Initialization method
    @cython.header(# Arguments
                   cosmoresults=object,  # CosmoResults
                   component='Component',
                   var_name=str,
                   )
    def __init__(self, cosmoresults, component, var_name):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the TransferFunction type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        object cosmoresults
        Component component
        str var_name
        str class_species
        double[::1] k_magnitudes
        Py_ssize_t k_gridsize
        double[::1] data
        double[::1] data_deriv
        double[::1] factors
        double[::1] exponents
        list splines
        """
        # Store instance data
        self.cosmoresults = cosmoresults
        self.component = component
        self.var_name = var_name
        if self.var_name not in CosmoResults.transfer_function_variable_names:
            abort(f'var_name {self.var_name} not implemented in TransferFunction')
        # The species (CLASS convention) of which to compute
        # transfer functions. If component is None, set the CLASS
        # species to 'tot', as this "species" do not correspond
        # to any component.
        if self.component is None:
            self.class_species = 'tot'
        else:
            self.class_species = self.component.class_species
        # The k values at which the transfer function
        # is tabulated by CLASS.
        self.k_magnitudes = self.cosmoresults.k_magnitudes
        self.k_gridsize = self.k_magnitudes.shape[0]
        # These will become arrays storing the transfer function and its
        # derivative with respect to the scale factor,
        # at a given k and as a function of a.
        self.data = self.data_deriv = None
        # Construct splines of the transfer function as a function of a,
        # for all k.
        self.factors   = empty(self.k_gridsize, dtype=C2np['double'])
        self.exponents = empty(self.k_gridsize, dtype=C2np['double'])
        self.splines = [None]*self.k_gridsize
        self.process()

    # Method for processing the transfer function data from CLASS.
    # The end result is the population self.splines, self.factors
    # and self.exponents.
    @cython.header(
        # Locals
        a_values='double[::1]',
        approximate_P_as_wρ='bint',
        available='bint',
        class_perturbation_name=str,
        class_species=str,
        class_units='double',
        exponent_max='double',
        fitted_trends=list,
        has_data='bint',
        i='Py_ssize_t',
        index='Py_ssize_t',
        k='Py_ssize_t',
        k_end='Py_ssize_t',
        k_send='Py_ssize_t',
        k_size='Py_ssize_t',
        k_start='Py_ssize_t',
        loga_values='double[::1]',
        loga_values_k='double[::1]',
        missing_perturbations_warning=str,
        n_outliers='Py_ssize_t',
        one_k_extra='bint',
        outlier='Py_ssize_t',
        outliers='Py_ssize_t[::1]',
        outliers_list=list,
        perturbation=object,  # np.ndarray or double
        perturbation_k=dict,
        perturbation_values='double[::1]',
        perturbation_values_arr=object,  # np.ndarray
        perturbations=list,
        perturbations_available=dict,
        perturbations_detrended='double[::1]',
        perturbations_detrended_k='double[::1]',
        rank_send='int',
        size='Py_ssize_t',
        spline='Spline',
        weights=object,  # np.ndarray
        weights_species=dict,
        Σweights=object,  # np.ndarray
    )
    def process(self):
        """
        """
        # Ensure that the cosmological background has been
        self.cosmoresults.background
        # Display progress message
        if self.component is None:
            if self.var_name == 'θ':
                masterprint(f'Processing total θ transfer functions ...')
            else:
                masterprint(f'Processing {self.var_name} transfer functions ...')
        else:
            masterprint(
                f'Processing {self.var_name} transfer functions '
                f'for {self.component.name} ...'
            )
        # Maximum (absolute) allowed exponent in the trend.
        # If an exponent greater than this is found,
        # the program will terminate.
        exponent_max = 10
        missing_perturbations_warning = ''.join([
            'The {} perturbations of CLASS species "{}" ',
            (f'(needed for the {self.component.name} component)'
                if self.component is not None else ''),
            ' are not available'
        ])
        missing_perturbations_warning_given = False
        perturbations_available = {
            class_species: True for class_species in self.class_species.split('+')
        }
        perturbations = self.cosmoresults.perturbations
        class_perturbation_name = {
            'δ' : 'delta_{}',
            'θ' : 'theta_{}',
            'δP': 'cs2_{}',  # Note that cs2 is really δP/δρ
            'σ' : 'shear_{}',
            'hʹ': 'h_prime',
        }[self.var_name]
        approximate_P_as_wρ = (self.var_name == 'δP' and self.component.approximations['P=wρ'])
        # A spline should be constructed for each k value,
        # of which there are self.k_gridsize. Fairly distribute this
        # work among the processes.
        k_start, k_size = partition(self.k_gridsize)
        k_end = k_start + k_size
        # When the work is not exactly divisible among
        # the processes, some processes will have an
        # additional k value to process.
        one_k_extra = (k_size*nprocs > self.k_gridsize)
        # Compute and store a Spline object for each k.
        # This is done in parallel. All processes are forced to
        # carry out the same number of iterations regardless of the
        # number of k values which should be processed by them.
        for k in range(k_start, k_end + (not one_k_extra)):
            # Only process if this is not the extra iteration
            has_data = (k < k_end)
            if has_data:
                perturbation_k = perturbations[k]
                a_values = perturbation_k['a'].copy()
                # The perturbation_k dict store perturbation arrays for
                # all perturbation types and CLASS species, defined at
                # times matching those of a_values.
                # Because a single CO𝘕CEPT species can map to multiple
                # CLASS species, we need to construct an array of
                # perturbation values as a weighted sum of perturbations
                # over the individual ('+'-separated) CLASS species,
                # with weights dependent on the type of
                # CLASS perturbation.
                # We also need to apply the CLASS units, which again
                # depend on the type of perturbation.
                # Finally, outlier rejection may take place by adding
                # indices to the outliers_list.
                outliers_list = []
                with unswitch:
                    if self.var_name == 'δ':
                        # For δ we have
                        # δ_tot = (δ_1*ρ_bar_1 + δ_2*ρ_bar_2 + ...)/(ρ_bar_1 + ρ_bar_2 + ...)
                        weights_species = {
                            class_species: self.cosmoresults.ρ_bar(a_values, class_species)
                            for class_species in self.class_species.split('+')
                        }
                        Σweights = np.sum(tuple(weights_species.values()), axis=0)
                        for class_species in weights_species:
                            weights_species[class_species] *= 1/Σweights
                        # We have no CLASS units to apply
                        class_units = 1
                    elif self.var_name == 'θ':
                        # For θ we have
                        # θ_tot = (θ_1*ρ_bar_1 + θ_2*ρ_bar_2 + ...)/(ρ_bar_1 + ρ_bar_2 + ...)
                        weights_species = {
                            class_species: self.cosmoresults.ρ_bar(a_values, class_species)
                            for class_species in self.class_species.split('+')
                        }
                        Σweights = np.sum(tuple(weights_species.values()), axis=0)
                        for class_species in weights_species:
                            weights_species[class_species] *= 1/Σweights
                        # We have CLASS units of [time⁻¹]
                        class_units = ℝ[light_speed/units.Mpc]
                    elif self.var_name == 'δP':
                        # CLASS does not provide the δP(k) perturbations
                        # directly. Instead it provides δP(k)/δρ(k).
                        # To get the total δP from multiple δP/δρ,
                        # we then have
                        # δP_tot = δP_1 + δP_2 + ...
                        #        = (δP/δρ)_1*δρ_1 + (δP/δρ)_2*δρ_2 + ...
                        #        = (δP/δρ)_1*δ_1*ρ_bar_1 + (δP/δρ)_2*δ_2*ρ_bar_2 + ...
                        weights_species = {
                            class_species: (
                                perturbation_k[f'delta_{class_species}']
                                *self.cosmoresults.ρ_bar(a_values, class_species)
                            )
                            for class_species in self.class_species.split('+')
                        }
                        # The CLASS units of δP/δρ are [length²time⁻²]
                        class_units = ℝ[light_speed**2]
                        # Look for oulier points which are outside the
                        # legal range 0 ≤ δP/δρ ≤ c²/3. As the data is
                        # directly from CLASS, c = 1.
                        for class_species in weights_species:
                            perturbation = perturbation_k.get(f'cs2_{class_species}')
                            if perturbation is not None:
                                perturbation_values = perturbation
                                for i in range(perturbation_values.shape[0]):
                                    if not (0 <= perturbation_values[i] <= ℝ[1/3]):
                                        outliers_list.append(i)
                    elif self.var_name == 'σ':
                        # For σ we have
                        # σ_tot = (σ_1*(ρ_bar_1 + c⁻²P_bar_1) + σ_2*(ρ_bar_2 + c⁻²P_bar_2) + ...)
                        #          /((ρ_bar_1 + c⁻²P_bar_1) + (ρ_bar_2 + c⁻²P_bar_2) + ...)
                        weights_species = {class_species: 
                                                   self.cosmoresults.ρ_bar(a_values, class_species)
                            + ℝ[light_speed**(-2)]*self.cosmoresults.P_bar(a_values, class_species)
                            for class_species in self.class_species.split('+')
                        }
                        Σweights = np.sum(tuple(weights_species.values()), axis=0)
                        for class_species in weights_species:
                            weights_species[class_species] *= 1/Σweights
                         # We have CLASS units of [length²time⁻²]
                        class_units = ℝ[light_speed**2]
                    elif self.var_name == 'hʹ':
                        # As hʹ is a species independent quantity,
                        # we do not have any weights.
                        weights_species = {class_species: 1
                            for class_species in self.class_species.split('+')
                        }
                        # We have CLASS units of [time⁻¹]
                        class_units = ℝ[light_speed/units.Mpc]
                    else:
                        abort(f'Do not know how to process transfer function "{self.var_name}"')
                        # Just to satisfy the compiler
                        weights_species, class_units = {}, 1
                # Construct the perturbation_values_arr array from the
                # CLASS perturbations matching the perturbations type
                # and CLASS species, together with the weights.
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
                            class_perturbation_name.format(class_species)
                        )
                        if perturbation is None:
                            perturbations_available[class_species] = False
                        else:
                            perturbation_values_arr += weights*class_units*perturbation
                    if k == 0 and not all(perturbations_available.values()):
                        # Warn or abort on missing perturbations.
                        # We only do this for k = 0, which is the first
                        # perturbation encountered on the master process.
                        if len(perturbations_available) == 1:
                            abort(missing_perturbations_warning.format(
                                class_perturbation_name, self.class_species,
                            ))
                        for class_species, available in perturbations_available.items():
                            if not available:
                                masterwarn(missing_perturbations_warning.format(
                                    class_perturbation_name, class_species,
                                ))
                        if not any(perturbations_available.values()):
                            abort(
                                f'No {class_perturbation_name} perturbations '
                                f'for the {self.component.name} component available'
                            )
                perturbation_values = perturbation_values_arr
                # Remove outliers
                if outliers_list:
                    outliers = asarray(outliers_list, dtype=C2np['Py_ssize_t'])
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
                a_values, perturbation_values = remove_doppelgängers(a_values, perturbation_values)
                # Perform non-linear detrending. The data to be splined
                # is in the form {log(a), perturbation_values - trend},
                # with trend = factor*a**exponent. Here we find this
                # trend trough curve fitting of perturbation_values.
                fitted_trends = [
                    scipy.optimize.curve_fit(
                        self.power_law,
                        a_values,
                        perturbation_values,
                        (1, 0),
                        bounds=bounds,
                    )
                    for bounds in (
                        (
                            [-ထ, -exponent_max],
                            [+ထ,  0           ],
                        ),
                        (
                            [-ထ,  0           ],
                            [+ထ, +exponent_max],
                        ),
                    )
                ]
                self.factors[k], self.exponents[k] = fitted_trends[
                    np.argmin([fitted_trend[1][1,1] for fitted_trend in fitted_trends])
                ][0]
                if isclose(abs(self.exponents[k]), exponent_max):
                    abort(
                        f'Error processing {self.var_name} perturbations for '
                        f'{self.component.name} at k = {self.k_magnitudes[k]} Mpc⁻¹: '
                        f'Detrending resulted in exponent = exponent_max = {exponent_max}.'
                    )
                if abs(self.exponents[k]) < ℝ[1e+3*machine_ϵ]:
                    self.exponents[k] = 0
                perturbations_detrended = (
                    asarray(perturbation_values)
                    - self.factors[k]*asarray(a_values)**self.exponents[k]
                )
                # As the spline will be over
                # {log(a), perturbation_values - trend},
                # we need log(a).
                loga_values = np.log(a_values)
            # Communicate the spline data
            for rank_send in range(nprocs):
                # Broadcast the k value belonging to the data to
                # be communicated. If no data should
                # be communicated, signal this by broadcasting -1.
                k_send = bcast(k if has_data else -1, root=rank_send)
                if k_send == -1:
                    continue
                # Broadcast the trend
                self.factors[k_send], self.exponents[k_send] = bcast(
                    (self.factors[k], self.exponents[k]),
                    root=rank_send,
                )
                # Broadcast the data
                loga_values_k = smart_mpi(
                    loga_values if rank == rank_send else None,
                    0,  # Buffer, different from the below
                    root=rank_send,
                    mpifun='bcast',
                )
                perturbations_detrended_k = smart_mpi(
                    perturbations_detrended if rank == rank_send else None,
                    1,  # Buffer, different from the above
                    root=rank_send,
                    mpifun='bcast',
                )
                # Construct cubic spline of
                # {log(a), perturbations - trend}.
                spline = Spline(loga_values_k, perturbations_detrended_k)
                self.splines[k_send] = spline
                # If class_plot_perturbations is True,
                # plot the detrended perturbation and save it to disk.
                if master and class_plot_perturbations:
                    plot_detrended_perturbations(
                        loga_values_k, perturbations_detrended_k, self, k_send,
                    )
        # Done processing transfer functions
        masterprint('done')

    # Helper function for the process method
    @staticmethod
    def power_law(x, factor, exponent):
        return factor*x**exponent

    # Method for evaluating the k'th transfer function
    # at a given scale factor.
    @cython.header(
        # Arguments
        k='Py_ssize_t',
        a='double',
        # Locals
        spline='Spline',
        value='double',
        returns='double',
    )
    def eval(self, k, a):
        # Lookup transfer(k, a) by evaluating
        # the k'th {log(a), transfer - trend} spline.
        spline = self.splines[k]
        return spline.eval(log(a)) + self.factors[k]*a**self.exponents[k]

    # Main method for getting the transfer function as function of k
    # at a specific value of the scale factor.
    @cython.pheader(
        # Arguments
        a='double',
        # Locals
        k='Py_ssize_t',
        returns='double[::1]',
    )
    def as_function_of_k(self, a):
        """The self.data array is used to store the transfer function
        as function of k for the given a. As this array is reused for
        all calls to this function, you cannot get two arrays of
        transfer function values at different times. If you need this,
        make sure to copy the returned array before calling this
        function again.
        """
        # Populate the data array with transfer_function(k)
        # and return this array.
        if self.data is None:
            self.data = empty(self.k_gridsize)
        for k in range(self.k_gridsize):
            self.data[k] = self.eval(k, a)
        return self.data

    # Method for evaluating the derivative of the k'th transfer
    # function with respect to the scale factor, at a specific value of
    # the scale factor.
    @cython.header(
        # Arguments
        k='Py_ssize_t',
        a='double',
        # Locals
        exponent='double',
        spline='Spline',
    )
    def eval_deriv(self, k, a):
        # The transfer function is splined using {x, f(x)} with
        #     x = log(a),
        #     f(x) = transfer(a) - trend(a)
        #          = transfer(a) - factor*a**exponent,
        # and so we have
        #    df/dx = df/da*da/dx
        #          = df/da*a
        #          = a*(dtransfer/da - factor*exponent*a**(exponent - 1))
        #          = a*dtransfer/da - factor*exponent*a**exponent
        # and then
        #     dtransfer/da = (df/dx)/a + factor*exponent*a**(exponent - 1).
        spline = self.splines[k]
        exponent = self.exponents[k]
        return spline.eval_deriv(log(a))/a + self.factors[k]*exponent*a**(exponent - 1)

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
        """The self.data_deriv array is used to store the transfer
        function derivatives as function of k for the given a. As this
        array is reused for all calls to this function, you cannot get
        two arrays of transfer function derivatives at different times.
        If you need this, make sure to copy the returned array before
        calling this function again.
        """
        # Populate the data_deriv array with derivatives of the
        # transfer_function(k) and return this array.
        if self.data_deriv is None:
            self.data_deriv = empty(self.k_gridsize)
        for k in range(self.k_gridsize):
            self.data_deriv[k] = self.eval_deriv(k, a)
        return self.data_deriv

# Function which solves the linear cosmology using CLASS,
# from before the initial simulation time and until the present.
@cython.pheader(# Arguments
                k_min='double',
                k_max='double',
                k_gridsize='Py_ssize_t',
                gauge=str,
                filename=str,
                # Locals
                cosmoresults=object, # CosmoResults
                extra_params=dict,
                k_gridsize_max='Py_ssize_t',
                k_magnitudes='double[::1]',
                k_magnitudes_str=str,
                params_specialized=dict,
                returns=object,  # CosmoResults
               )
def compute_cosmo(k_min=-1, k_max=-1, k_gridsize=-1, gauge='synchronous', filename=''):
    """All calls to CLASS should be done through this function.
    If no arguments are supplied, CLASS will be run with the parameters
    stored in class_params. The return type is CosmoResults, which
    stores the result of the CLASS computation.
    If k_min, k_max are given, a more in-depth computation will be
    carried out by CLASS, where transfer functions and perturbations
    are also computed.
    All results from calls to this function are cached (using the
    global variable cosmoresults_archive), so you can safely call this
    function multiple times with the same arguments without it having
    to do the same CLASS computation over and over again.
    The k_min and k_max arguments specify the |k| interval on which
    the physical quantities should be tabulated. The k_gridsize specify
    the (maximum) number of |k| values at which to do this tabulation.
    The |k| values will be distributed logarithmically.
    The gauge of the transfer functions can be specified by
    the gauge argument, which can be any valid CLASS gauge.
    If a filename is given, CLASS results are loaded from this file.
    """
    # If a gauge is given explicitly as a CLASS parameter in the
    # parameter file, this gauge should overwrite what ever is passed
    # to this function.
    gauge = class_params.get('gauge', gauge).lower()
    # Shrink down k_gridsize if it is too large to be handled by CLASS.
    # Also use the largest allowed value as the default value,
    # when no k_gridsize is given.
    k_gridsize_max = (class__ARGUMENT_LENGTH_MAX_ - 1)//(len(k_float2str(0)) + 1)
    if k_gridsize > k_gridsize_max:
        masterwarn(
            f'Reducing number of k modes from {k_gridsize} to {k_gridsize_max}. '
            f'If you really want more k modes, you need to increase the CLASS macro '
            f'_ARGUMENT_LENGTH_MAX_ in include/parser.h.'
        )
        k_gridsize = k_gridsize_max
    elif k_gridsize == -1:
        k_gridsize = k_gridsize_max
    # If this exact CLASS computation has already been carried out,
    # return the stored results.
    cosmoresults = cosmoresults_archive.get((k_min, k_max, k_gridsize, gauge))
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
        # transfer functions, in both floating and str representation.
        # This explicit stringification is needed because we have to
        # know the exact str representation of each |k| value passed to
        # CLASS, so we may turn it back into a numerical array,
        # ensuring that the values of |k| are identical
        # in both CLASS and CO𝘕CEPT.
        k_magnitudes = logspace(log10((1 - 1e-2)*k_min/units.Mpc**(-1)),
                                log10((1 + 1e-2)*k_max/units.Mpc**(-1)),
                                k_gridsize)
        with disable_numpy_summarization():
            k_magnitudes_str = np.array2string(k_magnitudes, max_line_width=ထ,
                                                             formatter={'float': k_float2str},
                                                             separator=',',
                                                             ).strip('[]')
        
        k_magnitudes = np.fromstring(k_magnitudes_str, sep=',')*units.Mpc**(-1)
        if len(set(k_magnitudes)) != k_gridsize:
            masterwarn(
                'It looks like you have requested too dense a k grid. '
                'Some of the CLASS perturbations will be computed at the same k.'
                )
        # Specify the extra parameters with which CLASS should be run
        extra_params = {# The |k| values to tabulate the perturbations.
                        # The transfer functions computed directly by
                        # CLASS will be on a slightly denser |k| grid.
                        'k_output_values': k_magnitudes_str,
                        # Needed for perturbations
                        'output': 'dTk vTk',
                        # This is used to minimize the number of extra
                        # k values inserted automatically by CLASS.
                        # With 'P_k_max_1/Mpc' set to 0, only a single
                        # additional k mode is inserted, and this at
                        # a very small k value.
                        # One could also set 'k_per_decade_for_pk' and
                        # 'k_per_decade_for_bao' to small values.
                        'P_k_max_1/Mpc': 0,
                        # Set the gauge. Note that N-body gauge
                        # is not implemented in CLASS.
                        'gauge': gauge,
                        }
    # Merge global and extra CLASS parameters
    params_specialized = class_params.copy()
    params_specialized.update(extra_params)
    # Transform all CLASS container parameters to str's of
    # comma-separated values. All other CLASS parameters will also
    # be converted to their str representation.
    params_specialized = stringify_dict(params_specialized)
    # Instantiate a CosmoResults object before calling CLASS,
    # in the hope that this exact CLASS call have already been
    # carried out.
    cosmoresults = CosmoResults(params_specialized, k_magnitudes, filename=filename)
    # Add the CosmoResults object to the global dict
    cosmoresults_archive[k_min, k_max, k_gridsize, gauge] = cosmoresults
    return cosmoresults
# Dict with keys of the form (k_min, k_max, k_gridsize, gauge),
# storing the results of calls to the above function as
# CosmoResults instances.
cython.declare(cosmoresults_archive=dict)
cosmoresults_archive = {}
# Helper function used in compute_cosmo
def k_float2str(k):
    return f'{k:.3e}'

# Function for computing transfer functions as function of k
@cython.pheader(# Arguments
                component='Component',
                variable=object,  # str, int or container of str's and ints
                k_min='double',
                k_max='double',
                k_gridsize='Py_ssize_t',
                specific_multi_index=object,  # tuple, int-like or str
                a='double',
                gauge=str,
                # Locals
                H='double',
                any_negative_values='bint',
                cosmoresults=object,  # CosmoResults
                k='Py_ssize_t',
                k_magnitudes='double[::1]',
                transfer='double[::1]',
                transfer_hʹ='double[::1]',
                transfer_spline='Spline',
                transfer_θ_tot='double[::1]',
                var_index='Py_ssize_t',
                w='double',
                ȧ_transfer_θ_totʹ='double[::1]',
                returns=tuple,  # (Spline, CosmoResults)
                )
def compute_transfer(component, variable, k_min, k_max,
                     k_gridsize=-1, specific_multi_index=None, a=-1, gauge='N-body'):
    """This function calls compute_cosmo which produces a CosmoResults
    instance which can talk to CLASS. Using the δ, θ, etc. methods on
    the CosmoResults object, TransferFunction instances are
    automatically created. All this function really implements
    are then the optional gauge transformations.
    """
    # Argument processing
    var_index = component.varnames2indices(variable, single=True)
    if a == -1:
        a = universals.a
    gauge = gauge.replace('-', '').lower()
    # Compute the cosmology via CLASS. As the N-body gauge is not
    # implemented in CLASS, the synchronous gauge is used in its place.
    # We do the transformation from synchronous to N-body gauge later.
    cosmoresults = compute_cosmo(k_min, k_max, k_gridsize,
                                 'synchronous' if gauge == 'nbody' else gauge)
    k_magnitudes = cosmoresults.k_magnitudes
    # Update k_gridsize to be what ever value was settled on
    # by the compute_cosmo function.
    k_gridsize = k_magnitudes.shape[0]
    # Get the requested transfer function
    # and transform to N-body gauge if requested.
    if var_index == 0:
        # Get the δ transfer function
        transfer = cosmoresults.δ(a, component)
        # Transform the δ transfer function from synchronous
        # to N-body gauge, if requested.
        if gauge == 'nbody':
            # To do the gauge transformation,
            # we need the total θ transfer function.
            transfer_θ_tot = cosmoresults.θ(a)
            # Do the gauge transformation
            H = hubble(a)
            w = component.w(a=a)
            for k in range(k_gridsize):
                transfer[k] += (ℝ[3*a*H/light_speed**2*(1 + w)]
                                *transfer_θ_tot[k]/k_magnitudes[k]**2)
    elif var_index == 1:
        # Get the θ transfer function
        transfer = cosmoresults.θ(a, component)
        # Transform the θ transfer function from synchronous
        # to N-body gauge, if requested.
        if gauge == 'nbody':
            # To do the gauge transformation,
            # we need the conformal time derivative
            # of the metric perturbation, hʹ.
            transfer_hʹ = cosmoresults.hʹ(a)
            # We also need (ȧ*θ_tot) differentiated with respect to
            # conformal time, evaluated at the given a.
            # With ʹ = d/dτ = a*d/dt = aȧ*d/da, we have
            # (ȧ*θ_tot)ʹ = a*d/dt(ȧ*θ_tot)
            #            = a*ä*θ_tot + a*ȧ*d/dt(θ_tot)
            #            = a*(ä*θ_tot + ȧ²*d/da(θ_tot))
            ȧ_transfer_θ_totʹ = a*(  ä(a)   *asarray(cosmoresults.θ(a,
                                                                    get='as_function_of_k'      ))
                                   + ȧ(a)**2*asarray(cosmoresults.θ(a,
                                                                    get='deriv_as_function_of_k'))
                                   )
            # Now do the gauge transformation.
            # Check for negative values, which implies that some
            # CLASS data has not converged.
            any_negative_values = False
            for k in range(k_gridsize):
                transfer[k] += (  0.5*transfer_hʹ[k]
                                - ℝ[3/light_speed**2]*ȧ_transfer_θ_totʹ[k]/k_magnitudes[k]**2
                                )
                if transfer[k] < 0:
                    any_negative_values = True
            if any_negative_values:
                masterwarn(f'The synchronous to N-body gauge transformation of the θ transfer '
                           f'function for the {component.class_species} CLASS species at '
                           f'a = {a} appears to have been carried out inaccurately, '
                           f'as negative values appear. '
                           f'You should consider cranking up the precision of CLASS. '
                           f'For now, the simulation will carry on using this possibly '
                           f'erroneous transfer function.'
                           )
    elif var_index == 2 and specific_multi_index == 'trace':
        # Get th δP transfer function
        transfer = cosmoresults.δP(a, component)
    elif (    var_index == 2
          and isinstance(specific_multi_index, tuple)
          and len(specific_multi_index) == 2
          ):
        # Get the σ transfer function
        transfer = cosmoresults.σ(a, component)
    else:
        abort(f'I do not know how to get transfer function of multi_index {specific_multi_index} '
              f'of variable number {var_index}'
              )
    # Construct a spline object over the tabulated transfer function
    transfer_spline = Spline(k_magnitudes, transfer)
    return transfer_spline, cosmoresults

# Function which given a gridsize computes k_min, k_max and k_gridsize
# which can be supplied to e.g. compute_transfer().
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    # Locals
    k_gridsize='Py_ssize_t',
    k_max='double',
    k_min='double',
    n_decades='double',
    returns=tuple,
)
def get_default_k_parameters(gridsize):
    k_min = ℝ[2*π/boxsize]
    k_max = ℝ[2*π/boxsize]*sqrt(3*(gridsize//2)**2)
    n_decades = log10(k_max/k_min)
    k_gridsize = int(round(modes_per_decade*n_decades))
    return k_min, k_max, k_gridsize

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
    scheme=dict,
    use_gridˣ='bint',
    # Locals
    A_s='double',
    H='double',
    compound_variable='bint',
    cosmoresults_δ=object,  # CosmoResults
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
    fluidvar=object,  # Tensor
    fluidvar_name=str,
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    i_conj='Py_ssize_t',
    i_global='Py_ssize_t',
    in_lower_i_half='bint',
    in_lower_j_half='bint',
    index='Py_ssize_t',
    index0='Py_ssize_t',
    index1='Py_ssize_t',
    j='Py_ssize_t',
    j_conj='Py_ssize_t',
    j_global='Py_ssize_t',
    k='Py_ssize_t',
    k_global='Py_ssize_t',
    ki='Py_ssize_t',
    ki_conj='Py_ssize_t',
    kj='Py_ssize_t',
    kj_conj='Py_ssize_t',
    kj2='Py_ssize_t',
    kk='Py_ssize_t',
    k_factor='double[::1]',
    k_factor_conj='double[::1]',
    k_gridsize='Py_ssize_t',
    k_gridvec='Py_ssize_t[::1]',
    k_gridvec_conj='Py_ssize_t[::1]',
    k_magnitude='double',
    k_max='double',
    k_min='double',
    k_pivot='double',
    k2='Py_ssize_t',
    k2_max='Py_ssize_t',
    mass='double',
    mom_dim='double*',
    multi_index=object,  # tuple or str
    n_s='double',
    nyquist='Py_ssize_t',
    phases_jik='double*',
    pos_dim='double*',
    pos_gridpoint='double',
    processed_specific_multi_index=object,  # tuple or str
    scheme_key=str,
    scheme_linear=dict,
    scheme_val=str,
    slab='double[:, :, ::1]',
    slab_jik='double*',
    slab_phases='double[:, :, ::1]',
    slab_phases_info=dict,
    sqrt_power='double',
    sqrt_power_common='double[::1]',
    tensor_rank='int',
    transfer='double',
    transfer_spline_δ='Spline',
    w='double',
    w_eff='double',
    Jⁱ_ptr='double*',
    δ_min='double',
    ρ_bar_a='double',
    ψ_dim='double[:, :, ::1]',
    ψ_dim_noghosts='double[:, :, :]',
    ςⁱⱼ_ptr='double*',
    ϱ_ptr='double*',
    𝒫_ptr='double*',   
)
def realize(component, variable, transfer_spline, cosmoresults,
            specific_multi_index=None, a=-1, scheme=None,
            use_gridˣ=False):
    """This function realizes a single variable of a component,
    given the transfer function as a Spline (using |k| in physical units
    as the independent variable) and the corresponding CosmoResults
    object, which carry additional information from the CLASS run that
    produced the transfer function. If only a single fluidscalar of the
    fluid variable should be realized, the multi_index of this
    fluidscalar may be specified. If you want a realization at a time
    different from the present you may specify an a.
    If a particle component is given, the Zeldovich approximation is
    used to distribute the paricles and assign momenta. This is
    done simultaneously, meaning that you cannot realize only the
    positions or only the momenta. For the particle realization to
    work correctly, you must pass the δ transfer function as
    transfer_spline. For particle components, the variable argument
    is not used.

    The realization can be carried out using several different schemes,
    controlled by the scheme argument. This is a dictionary with the
    keys 'phases', 'compound-order' and  'compound-space', all of which
    can take two values. The default is
    scheme = {
        'phases': 'primordial',
        'compound-order': 'linear',
        'compound-space': 'real',
    }
    which corresponds to linear realization. Taking Jⁱ as an example
    this linear realization looks like
        Jⁱ(x⃗) = a**(1 - 3w_eff)ϱ_bar(1 + w)ℱₓ⁻¹[T_θ(k)ζ(k)K(k⃗)ℛ(k⃗)],
    where ζ(k) = π*sqrt(2*A_s)*k**(-3/2)*(k/k_pivot)**((n_s - 1)/2)
    is the primordial curvature perturbation, T_θ(k) is the passed
    transfer function for θ, ℛ(k⃗) is a field of primordial phases,
    and K(k⃗) is the tensor structure (often referred to as k factor)
    needed to convet from θ to uⁱ. For uⁱ, K(k⃗) = -ikⁱ/k². The factors
    outside the Fourier transform then converts from uⁱ to Jⁱ.
    We can instead choose to use the evolved non-linear phases of ϱ,
    by using scheme['phases'] == 'non-linear'. Then the realization
    looks like
        Jⁱ(x⃗) = a**(1 - 3w_eff)ϱ_bar(1 + w)ℱₓ⁻¹[T_θ(k)/T_δϱ(k)K(k⃗)δϱ(k⃗)],
    where δϱ(k⃗) = ℱₓ[δϱ(x⃗)] is computed from the present ϱ(x⃗) grid,
    and T_δϱ(k) is the (not passed) transfer function of δϱ.
    An orthogonal option is 'compound-order'. Setting this to
    'non-linear' signals that the multiplication which takes uⁱ to Jⁱ
    should be done using non-linear variables rather than background
    quantities. That is,
        Jⁱ(x⃗) = a**(1 - 3w_eff)(ϱ(x⃗) + c⁻²𝒫(x⃗))ℱₓ⁻¹[...]
    Finally, we can move the non-linear multiplication inside of the
    Fourier transform using scheme['compound-space'] == 'fourier':
        Jⁱ(x⃗) = a**(1 - 3w_eff)ℱₓ⁻¹[(ϱ(k⃗) + c⁻²𝒫(k⃗))...]

    For both particle and fluid components it is assumed that the
    passed component is of the correct size beforehand. No resizing
    will take place in this function.
    """
    if a == -1:
        a = universals.a
    # Get the index of the fluid variable to be realized
    # and print out progress message.
    processed_specific_multi_index = ()
    if component.representation == 'particles':
        # For particles, the Zeldovich approximation is used for the
        # realization. This realizes both positions and momenta.
        # This means that the value of the passed variable argument
        # does not matter. To realize all three components of positions
        # and momenta, we need the fluid_index to have a value of 1
        # (corresponding to J or mom), so that multi_index takes on
        # vector values ((0, ), (1, ), (2, )).
        fluid_index = 1
        if specific_multi_index is not None:
            abort(f'The specific multi_index {specific_multi_index} was specified for realization '
                  f'of "{component.name}". Particle components may only be realized completely.')
        masterprint(f'Realizing particles of {component.name} ...')
    elif component.representation == 'fluid':
        fluid_index = component.varnames2indices(variable, single=True)
        fluidvar_name = component.fluid_names['ordered'][fluid_index]
        if specific_multi_index is None:
            masterprint(f'Realizing {fluidvar_name} of {component.name} ...')
        else:
            processed_specific_multi_index = ( component
                                              .fluidvars[fluid_index]
                                              .process_multi_index(specific_multi_index)
                                              )
            masterprint(
                f'Realizing {fluidvar_name}{{}} of {component.name} ...'
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
    # Determine the gridsize of the grid used to do the realization
    if component.representation == 'particles':
        if not isint(ℝ[cbrt(component.N)]):
            abort(f'Cannot perform realization of particle component "{component.name}" '
                  f'with N = {component.N}, as N is not a cubic number.'
                  )
        gridsize = int(round(ℝ[cbrt(component.N)]))
    elif component.representation == 'fluid':
        gridsize = component.gridsize
    if gridsize%nprocs != 0:
        abort(f'The realization uses a gridsize of {gridsize}, '
              f'which is not evenly divisible by {nprocs} processes.'
              )
    # Handle the scheme argument
    if scheme is None:
        scheme = {}
    scheme = {key.lower().replace(' ', '').replace('-', ''):
        val.lower().replace(' ', '').replace('-', '')
        for key, val in scheme.items()
    }
    # Use the linear realization scheme by default
    scheme_linear = {
        'phases': 'primordial',
        'compoundorder': 'linear',
        'compoundspace': 'real',
    }
    for scheme_key, scheme_val in scheme_linear.items():
        if scheme_key not in scheme:
            scheme[scheme_key] = scheme_val
    if len(scheme) != 3:
        abort('Error interpreting realization scheme')
    if scheme['phases'] not in ('primordial', 'nonlinear'):
        abort('Unreqonized value "{}" for scheme["phases"]'
            .format(scheme['phases']))
    if scheme['compoundorder'] not in ('linear', 'nonlinear'):
        abort('Unreqonized value "{}" for scheme["compound-order"]'
            .format(scheme['compoundorder']))
    if scheme['compoundspace'] not in ('real', 'fourier'):
        abort('Unreqonized value "{}" for scheme["compound-space"]'
            .format(scheme['compoundspace']))
    # When the compound order is 'linear', it does not matter whether
    # we use real or Fourier space for the compound space. We choose to
    # always use real space.
    if scheme['compoundorder'] == 'linear':
        scheme['compoundspace'] = 'real'
    # A compound order of 'nonlinear' and a compound space of 'fourier'
    # only makes a difference for compound variables;
    # that is, Jⁱ and ςⁱⱼ. If what we are realizing is another variable,
    # switch these back to 'linear' and 'real', respectively.
    if fluid_index == 1:
        # We are realizing Jⁱ
        compound_variable = True
    elif fluid_index == 2 and processed_specific_multi_index != 'trace':
        # We are realizing ςⁱⱼ
        compound_variable = True
    else:
        compound_variable = False
    if not compound_variable:
        if scheme['compoundorder'] == 'nonlinear':
            scheme['compoundorder'] = 'linear'
        if scheme['compoundspace'] == 'fourier':
            scheme['compoundspace'] = 'real'
    # Abort if a scheme was passed for a particle component, as there is
    # only one way of realizing these (linear realization).
    if component.representation == 'particles' and scheme != scheme_linear:
        abort('Can only do linear realization for particle components')
    # When realizing δ, it makes no sense to realize it using
    # non-linear phases.
    if fluid_index == 0 and scheme != scheme_linear:
        abort('Can only do linear realization of δ')
    # Extract various variables
    nyquist = gridsize//2
    H = hubble(a)
    w = component.w(a=a)
    w_eff = component.w_eff(a=a)
    ρ_bar_a = a**(-3*(1 + w_eff))*component.ϱ_bar
    if cosmoresults is not None:
        A_s = cosmoresults.A_s
        n_s = cosmoresults.n_s
        k_pivot = cosmoresults.k_pivot
    # Fill 1D array with values used for the realization.
    # These values are the k (but not k⃗) dependent values inside the
    # inverse Fourier transform, not including any additional tenstor
    # structure (the k factors K(k⃗) or non-linear grids U(k⃗).
    k2_max = 3*(gridsize//2)**2  # Max |k⃗|² in grid units
    sqrt_power_common = get_buffer(k2_max + 1,
        # Must use some buffer different from the one used to do the
        # domain decomposition of ψ below.
        0,
    )
    if scheme['phases'] == 'nonlinear':
        # When using the non-linear phases of δϱ to do the realizations,
        # we need the transfer function of δρ, which is just
        # ρ_bar_a times the transfer function of δ.
        k_min, k_max, k_gridsize = get_default_k_parameters(gridsize)
        transfer_spline_δ, cosmoresults_δ = compute_transfer(
            component, 0, k_min, k_max, k_gridsize, a=a,
        )
    for k2 in range(1, k2_max + 1):
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        transfer = transfer_spline.eval(k_magnitude)
        with unswitch:
            if scheme['phases'] == 'primordial':
                # Realize using ℱₓ⁻¹[T(k) ζ(k) K(k⃗) U(k⃗) ℛ(k⃗)],
                # with K(k⃗) and U(k⃗) capturing any tensor structure and
                # other non-linear variables to be multiplied on in
                # Fourier space, respectively. The k⃗-independent part
                # needed here is T(k)ζ(k), with T(k) the supplied
                # transfer function and
                # ζ(k) = π*sqrt(2*A_s)*k**(-3/2)*(k/k_pivot)**((n_s - 1)/2)
                # the primordial curvature perturbations. The remaining
                # ℛ(k⃗) is the primordial phases
                sqrt_power_common[k2] = (
                    # T(k)
                    transfer
                    # ζ(k)
                    *k_magnitude**ℝ[0.5*n_s - 2]*ℝ[π*sqrt(2*A_s)*k_pivot**(0.5 - 0.5*n_s)
                        # Fourier normalization
                        *boxsize**(-1.5)
                    ]
                )
            elif scheme['phases'] == 'nonlinear':
                # Realize using ℱₓ⁻¹[T(k)/T_δϱ(k) K(k⃗) U(k⃗) ℱₓ[δϱ(x⃗)]],
                # with K(k⃗) and U(k⃗) capturing any tensor structure and
                # other non-linear variables to be multiplied on in
                # Fourier space, respectively. The k⃗-independent part
                # needed here is T(k)/T_δϱ(k), with T(k) the supplied
                # transfer function and T_δϱ(k) the transfer function
                # of δϱ.
                sqrt_power_common[k2] = (
                    # T(k)
                    transfer
                    # 1/T_δϱ(k)
                    /transfer_spline_δ.eval(k_magnitude)*ℝ[1/component.ϱ_bar
                        # Normalization due to FFT + IFFT
                        *float(gridsize)**(-3)
                    ]
                )
    # At |k| = 0, the power should be zero, corresponding to a
    # real-space mean value of zero of the realized variable.
    sqrt_power_common[0] = 0
    # Fetch a slab decomposed grid for storing the phases,
    # possibly multiplied by other non-linear grids if the compound
    # space is set to Fourier. If this is the first time, the grid will
    # be allocated, otherwise the previous grid will be returned,
    # still containing the previous data.
    slab_phases = get_fftw_slab(gridsize, 'slab_phases')
    # Information about the data from the previous call
    # is stored in the module level slab_phases_previous_info dict.
    # To see if we can reuse the slab_phases as is, we compare this
    # information with that of the current realization.
    slab_phases_info = {
        'phases': scheme['phases'],
        'compoundspace': scheme['compoundspace'],
        'a': a,
        'use_gridˣ': use_gridˣ,
        'gridsize': gridsize,
    }
    if slab_phases_info['phases'] == 'primordial' and slab_phases_info['compoundspace'] == 'real':
        # The slab_phases contain no non-linear information,
        # and so it is of nu importance at what time the slab_phases
        # were made, or whether using the starred or unstarred grids.
        slab_phases_info['a'] = None
        slab_phases_info['use_gridˣ'] = None
    if slab_phases_info != slab_phases_previous_info:
        # Populate slab_phases with either ℛ(k⃗) or ℱₓ[ϱ(x⃗)]
        if scheme['phases'] == 'primordial':
            # Populate slab_phases with ℛ(k⃗)
            get_primordial_phases(slab_phases)
        elif scheme['phases'] == 'nonlinear':
            # Populate slab_phases with ℱₓ[ϱ(x⃗)]
            slab_decompose(component.ϱ.gridˣ_mv if use_gridˣ else component.ϱ.grid_mv, slab_phases)
            fft(slab_phases, 'forward')
            # Remove the k⃗ = 0⃗ mode, leaving ℱₓ[δϱ(x⃗)]
            if master:
                slab_phases[0, 0, 0] = 0  # Real part
                slab_phases[0, 0, 1] = 0  # Imag part
        # Multiply by Fourier-transformed non-linear grids U(k⃗)
        if scheme['compoundspace'] == 'fourier':
            if (   (fluid_index == 1)
                or (fluid_index == 2 and processed_specific_multi_index != 'trace')
            ):
                # We are realizing Jⁱ or ςⁱⱼ
                # !!! IMPLEMENT THIS
                ...
                abort(
                    'Realizations using scheme["compoundspace"] == "fourier" not yet implemented!'
                )
            else:
                # It should not be possible to ever reach this line
                abort(
                    f'Cannot realize variable {fluid_index}[{processed_specific_multi_index}] '
                    f'using scheme["compoundspace"] == "fourier"'
                )
    slab_phases_previous_info.update(slab_phases_info)
    # Fetch a slab decomposed grid for storing the entirety of what is
    # to be inverse Fourier transformed. As we cannot reuse data from
    # previous calls, we do not pass in a specific buffer name.
    slab = get_fftw_slab(gridsize)
    # Allocate 3-vectors which will store componens
    # of the k vectors (in grid units).
    k_gridvec      = empty(3, dtype=C2np['Py_ssize_t'])
    k_gridvec_conj = empty(3, dtype=C2np['Py_ssize_t'])
    # Allocate arrays of length 2, storing the real and imag part
    # of the k factor.
    k_factor      = empty(2, dtype=C2np['double'])
    k_factor_conj = empty(2, dtype=C2np['double'])
    # Initialize index0 and index1.
    # The actual values are not important.
    index0 = index1 = 0
    # Loop over all fluid scalars of the fluid variable
    fluidvar = component.fluidvars[fluid_index]
    for multi_index in (
        fluidvar.multi_indices if specific_multi_index is None
        else [processed_specific_multi_index]
    ):
        # Determine rank of the tensor being realized (0 for scalar
        # (i.e. ϱ), 1 for vector (i.e. J), 2 for tensor (i.e. σ)).
        if fluid_index == 0 or isinstance(multi_index, str):
            # If multi_index is a str it is 'trace', which means that
            # 𝒫 is being realized. 
            # If fluid_index is 0, ϱ is being realized.
            tensor_rank = 0
        else:
            # The multi_index is a tuple of indices
            tensor_rank = len(multi_index)
        # Extract individual indices from multi_index
        if tensor_rank > 0:
            index0 = multi_index[0]
        if tensor_rank > 1:
            index1 = multi_index[1]
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
            k_gridvec[1] = kj
            kj2 = kj**2
            in_lower_j_half = (j_global <= ℤ[gridsize//2])
            # Loop through the complete i-dimension
            for i in range(gridsize):
                # The i-component of the wave vector (grid units)
                if i > ℤ[gridsize//2]:
                    ki = i - gridsize
                else:
                    ki = i
                k_gridvec[0] = ki
                in_lower_i_half = (i < ℤ[gridsize//2])
                # Loop through the complete, padded k-dimension
                # in steps of 2 (one complex number at a time).
                for k in range(0, ℤ[slab.shape[2]], 2):
                    # The k-component of the wave vector (grid units)
                    kk = k//2
                    k_gridvec[2] = kk
                    # The squared magnitude of the wave vector
                    # (grid units).
                    k2 = ℤ[ki**2 + kj2] + kk**2
                    # Regardless of what is being realized,
                    # the |k| = 0 mode should vanish, leading to a field
                    # with zero mean.
                    if k2 == 0:  # Only ever True for master
                        slab[0, 0, 0] = 0
                        slab[0, 0, 1] = 0
                        continue
                    # Pointer to the [j, i, k]'th element of the slab.
                    # The complex number is then given as
                    # Re = slab_jik[0], Im = slab_jik[1].
                    slab_jik = cython.address(slab[j, i, k:])
                    # Pointer to the [j, i, k]'th element
                    # of the phases.
                    phases_jik = cython.address(slab_phases[j, i, k:])
                    # Compute k factor K(k⃗) depending on the rank
                    # of the tensor being realized.
                    # For scalars we do not have any k factor.
                    with unswitch(3):
                        if tensor_rank == 1:
                            # Vector
                            compute_k_factor_vector(index0, index1, k_gridvec, k2, k_factor)
                        elif tensor_rank == 2:
                            # Rank 2 tensor
                            compute_k_factor_tensor(index0, index1, k_gridvec, k2, k_factor)
                    # The phases (including the possible multiplication
                    # by the non-linear grids) are guarenteed to satisfy
                    # the complex conjugate symmetry condition.
                    # On the contrary, the k factor K(k⃗) will break the
                    # symmetry. On the DC and Nyquist planes,
                    # the complex conjugate symmetry then has to be
                    # inforced by hand. We do this by changing the
                    # k factor for the elements in the lower j half of
                    # these planes to that of their "conjugated"
                    # element, situated at the negative k vector.
                    # This is of course only needed for
                    # non-scalar variables.
                    with unswitch(3):
                        if tensor_rank > 0:
                            with unswitch(2):
                                if in_lower_j_half:
                                    if kk == 0 or kk == nyquist:
                                        # Indicies of the conjugated element.
                                        # Note that k_conj = k.
                                        j_conj = 0 if j_global == 0 else gridsize - j_global
                                        i_conj = 0 if i        == 0 else gridsize - i
                                        # Enforce complex conjugate symmetry.
                                        # For j_global == j_conj, the
                                        # conjucation is purely along i, and
                                        # so we may only edit half of the points
                                        # along this line.
                                        if i == i_conj and j_global == j_conj:
                                            # The complex number is its
                                            # own conjugate, so it has to
                                            # be purely real.
                                            k_factor[1] = 0
                                        elif j_global != j_conj or in_lower_i_half:
                                            # Fill the k_gridvec_conj vector
                                            if i_conj > ℤ[gridsize//2]:
                                                ki_conj = i_conj - gridsize
                                            else:
                                                ki_conj = i_conj
                                            if j_conj > ℤ[gridsize//2]:
                                                kj_conj = j_conj - gridsize
                                            else:
                                                kj_conj = j_conj
                                            k_gridvec_conj[0] = ki_conj
                                            k_gridvec_conj[1] = kj_conj
                                            k_gridvec_conj[2] = kk  # kk == kk_conj
                                            # Compute k_factor for the conjugate element
                                            with unswitch(3):
                                                if tensor_rank == 1:
                                                    # Vector
                                                    compute_k_factor_vector(
                                                        index0, index1, k_gridvec_conj, k2, k_factor_conj,
                                                    )
                                                elif tensor_rank == 2:
                                                    # Rank 2 tensor
                                                    compute_k_factor_tensor(
                                                        index0, index1, k_gridvec_conj, k2, k_factor_conj,
                                                    )
                                            
                                            # !!!
                                            # I do not get why the code below works, but it passes
                                            # slabs_check_symmetry. It feels wrong manipulating k_factor,
                                            # as we have to choose which of a conjugate pair to alter.
                                            # Perhaps we should just eliminate those data points where
                                            # the k_factor screws with the symmetry (equivalent to
                                            # k_factor = 0 over the entire DC and Nyquist planes).

                                            # Enforce conjugacy
                                            if tensor_rank == 1:
                                                k_factor[0] = -k_factor_conj[0]
                                                k_factor[1] = -k_factor_conj[1]
                                            elif tensor_rank == 2:
                                                k_factor[0] = +k_factor_conj[0]
                                                k_factor[1] = +k_factor_conj[1]
                    # The square root of the power at this |k⃗|,
                    # disregarding all possible k⃗-dependent quantities
                    # such as the k factor K(k⃗).
                    sqrt_power = sqrt_power_common[k2]
                    # Populate slab_jik dependent on the component
                    # representation and tensor_rank.
                    with unswitch(3):
                        if component.representation == 'particles':
                            # Realize the displacement field ψ.
                            # Because the k factor (+ikⁱ/k²) contains an
                            # i which we have not taken care of,
                            # we should perform the transformation
                            # (Re, Im) → (-Im, +Re). An additional
                            # minus sign is used because k_factor is
                            # computed with the compute_k_factor_vector
                            # function which uses the
                            # convention -ikⁱ/k².
                            slab_jik[0] = sqrt_power*k_factor[0]*(+phases_jik[1])
                            slab_jik[1] = sqrt_power*k_factor[1]*(-phases_jik[0])
                        elif component.representation == 'fluid':
                            with unswitch(3):
                                if tensor_rank == 0:
                                    # Realize δ or δ𝒫
                                    slab_jik[0] = sqrt_power*phases_jik[0]
                                    slab_jik[1] = sqrt_power*phases_jik[1]
                                elif tensor_rank == 1:
                                    # Realize uⁱ.
                                    # Because the k factor (-ikⁱ/k²)
                                    # contains an i which we have not
                                    # taken care of, we perform the
                                    # transformation
                                    # (Re, Im) → (-Im, +Re).
                                    slab_jik[0] = sqrt_power*k_factor[0]*(-phases_jik[1])
                                    slab_jik[1] = sqrt_power*k_factor[1]*(+phases_jik[0])
                                elif tensor_rank == 2:
                                    # Realize ςⁱⱼ
                                    slab_jik[0] = sqrt_power*k_factor[0]*phases_jik[0]
                                    slab_jik[1] = sqrt_power*k_factor[1]*phases_jik[1]
        # Fourier transform the slabs to coordinate space.
        # Now the slabs store the realized grid.
        fft(slab, 'backward')
        # Populate the fluid grids for fluid components,
        # or create the particles via the Zeldovich approximation
        # for particles.
        if component.representation == 'fluid':
            # Communicate the fluid realization stored in the slabs to
            # the designated fluid scalar grid. This also populates the
            # pseudo and ghost points.
            fluidscalar = fluidvar[multi_index]
            domain_decompose(slab, fluidscalar.gridˣ_mv if use_gridˣ else fluidscalar.grid_mv)
            # Transform the realized fluid variable to the actual
            # quantity used in the non-linear fluid equations.
            if fluid_index == 0:
                # δ → ϱ = ϱ_bar(1 + δ).
                # Print a warning if min(δ) < -1.
                δ_min = ထ
                ϱ_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                for i in range(component.size):
                    if ℝ[ϱ_ptr[i]] < δ_min:
                        δ_min = ℝ[ϱ_ptr[i]]
                    ϱ_ptr[i] = ℝ[component.ϱ_bar]*(1 + ℝ[ϱ_ptr[i]])
                δ_min = allreduce(δ_min, op=MPI.MIN)
                if δ_min < -1:
                    masterwarn(f'The realized ϱ of {component.name} has min(δ) = {δ_min:.4g} < -1')
            elif fluid_index == 1:
                Jⁱ_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                if scheme['compoundspace'] == 'fourier':
                    # (ϱ + c⁻²𝒫)uⁱ → Jⁱ = a**4(ρ + c⁻²P)uⁱ
                    #                   = a**(1 - 3w_eff) * (ϱ + c⁻²𝒫)uⁱ
                    for i in range(component.size):
                        Jⁱ_ptr[i] *= ℝ[a**(1 - 3*w_eff)]
                elif scheme['compoundorder'] == 'nonlinear':
                    # uⁱ → Jⁱ = a**4(ρ + c⁻²P)uⁱ
                    #         = a**(1 - 3w_eff)(ϱ + c⁻²𝒫) * uⁱ
                    ϱ_ptr  = component.ϱ.gridˣ if use_gridˣ else component.ϱ.grid
                    𝒫_ptr  = component.𝒫.gridˣ if use_gridˣ else component.𝒫.grid
                    for i in range(component.size):
                        Jⁱ_ptr[i] *= ℝ[a**(1 - 3*w_eff)]*(ϱ_ptr[i] + ℝ[light_speed**(-2)]*𝒫_ptr[i])
                else:
                    # uⁱ → Jⁱ = a**4(ρ + c⁻²P)uⁱ
                    #         = a**(1 - 3w_eff)(ϱ + c⁻²𝒫) * uⁱ
                    #         ≈ a**(1 - 3w_eff)ϱ_bar(1 + w) * uⁱ
                    for i in range(component.size):
                        Jⁱ_ptr[i] *= ℝ[a**(1 - 3*w_eff)*component.ϱ_bar*(1 + w)]
            elif fluid_index == 2 and multi_index == 'trace':
                # δP → 𝒫 = 𝒫_bar + a**(3*(1 + w_eff)) * δP
                #        = c²*w*ϱ_bar + a**(3*(1 + w_eff)) * δP
                𝒫_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                for i in range(component.size):
                    𝒫_ptr[i] = ℝ[light_speed**2*w*component.ϱ_bar] + ℝ[a**(3*(1 + w_eff))]*𝒫_ptr[i]
            elif fluid_index == 2:
                ςⁱⱼ_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                if scheme['compoundspace'] == 'fourier':
                    # What is realized is already ςⁱⱼ
                    pass
                elif scheme['compoundorder'] == 'nonlinear':
                    # σⁱⱼ → ςⁱⱼ = (ϱ + c⁻²𝒫) * σⁱⱼ
                    ϱ_ptr  = component.ϱ.gridˣ if use_gridˣ else component.ϱ.grid
                    𝒫_ptr  = component.𝒫.gridˣ if use_gridˣ else component.𝒫.grid
                    for i in range(component.size):
                       ςⁱⱼ_ptr[i] *= ϱ_ptr[i] + ℝ[light_speed**(-2)]*𝒫_ptr[i]
                else:
                    # σⁱⱼ → ςⁱⱼ = (ϱ + c⁻²𝒫) * σⁱⱼ
                    #           ≈ ϱ_bar(1 + w) * σⁱⱼ
                    for i in range(component.size):
                        ςⁱⱼ_ptr[i] *= ℝ[component.ϱ_bar*(1 + w)]
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
        # Importantly, use a buffer different from the one already in
        # use by sqrt_power_common.
        ψ_dim = domain_decompose(slab, 1)
        ψ_dim_noghosts = ψ_dim[2:(ψ_dim.shape[0] - 2),
                               2:(ψ_dim.shape[1] - 2),
                               2:(ψ_dim.shape[2] - 2)]
        # Determine and set the mass of the particles
        # if this is still unset.
        if component.mass == -1:
            component.mass = a**3*ρ_bar_a*boxsize**3/component.N
        mass = component.mass
        # Get f_growth = H⁻¹Ḋ/D, where D is the linear growth factor
        f_growth = cosmoresults.growth_fac_f(a)
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
# Module level variable used by the realize function
cython.declare(slab_phases_previous_info=dict)
slab_phases_previous_info = {}

# Functions for computing the k factors for scalars, vectors, etc.
# The passed k_factor is an array of lenght 2, representing the complex
# k factor. The functions should not return anything but just populate
# passed k_factor.
@cython.header(
    # Arguments
    index0='Py_ssize_t',
    index1='Py_ssize_t',
    k_gridvec='Py_ssize_t[::1]',
    k2='Py_ssize_t',
    k_factor='double[::1]',
    # Locals
    factor='double',
    k_dim0='Py_ssize_t',
    returns='void',
)
def compute_k_factor_vector(index0, index1, k_gridvec, k2, k_factor):
    # Vector (-ikⁱ/k²).
    # Note that we do not apply the i.
    # Thus, the following transformation is missing:
    # (Re, Im) → (-Im, +Re).
    k_dim0 = k_gridvec[index0]
    factor = -(ℝ[boxsize/(2*π)]*k_dim0)/k2
    k_factor[0] = factor  # Real
    k_factor[1] = factor  # Imag
@cython.header(
    # Arguments
    index0='Py_ssize_t',
    index1='Py_ssize_t',
    k_gridvec='Py_ssize_t[::1]',
    k2='Py_ssize_t',
    k_factor='double[::1]',
    # Locals
    factor='double',
    k_dim0='Py_ssize_t',
    k_dim1='Py_ssize_t',
    returns='void',
)
def compute_k_factor_tensor(index0, index1, k_gridvec, k2, k_factor):
    # Rank 2 tensor (3/2(δⁱⱼ/3 - kⁱkⱼ/k²))
    k_dim0 = k_gridvec[index0]
    k_dim1 = k_gridvec[index1]
    factor = 0.5*(index0 == index1) - (1.5*k_dim0*k_dim1)/k2
    k_factor[0] = factor  # Real
    k_factor[1] = factor  # Imag

# Function for creating the lower and upper random
# Fourier xy-planes with complex-conjugate symmetry.
@cython.header(
    # Arguments
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
    # Seed the pseudo-random number generator
    if seed != 0:
        seed_rng(seed)
    # Create the plane and populate it with Gaussian distributed
    # complex random numbers with mean 0 and variance 1.
    plane = empty((gridsize, gridsize, 2), dtype=C2np['double'])
    for     j in range(gridsize):
        for i in range(gridsize):
            # The real and imaginary part individually
            # have mean 0 and variance 1/√2.
            plane[j, i, 0] = random_gaussian(0, ℝ[1/sqrt(2)])
            plane[j, i, 1] = random_gaussian(0, ℝ[1/sqrt(2)])
    # Enforce the symmetry plane[k_vec] = plane[-k_vec]*,
    # where * means complex conjugation.
    # We do this by replacing the random numbers for the elements in the
    # lower j half of the plane with those of the "conjugated" element,
    # situated at the negative k vector.
    # For j == j_conj, the conjucation is purely along i, and so we may
    # only edit half of the points along this line.
    for j in range(gridsize//2 + 1):
        j_conj = 0 if j == 0 else gridsize - j
        for i in range(gridsize):
            i_conj = 0 if i == 0 else gridsize - i
            # Enforce complex conjugate symmetry
            # if necessary. For j == j_conj,
            # the conjucation is purely along i, and
            # so we may only edit half of the points
            # along this line.
            if 𝔹[j == j_conj] and i == i_conj:
                # The complex number is its own conjugate,
                # so it has to be purely real.
                plane[j, i, 1] = 0
            elif 𝔹[j != j_conj] or i < ℤ[gridsize//2]:
                # Enforce conjugacy
                plane[j, i, 0] = +plane[j_conj, i_conj, 0]
                plane[j, i, 1] = -plane[j_conj, i_conj, 1]
    return plane

# Function that populates the passed slab decomposed grid with
# primordial phases ℛ(k⃗).
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    # Locals
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    j='Py_ssize_t',
    j_global='Py_ssize_t',
    k='Py_ssize_t',
    kk='Py_ssize_t',
    nyquist='Py_ssize_t',
    plane_dc='double[:, :, ::1]',
    plane_nyquist='double[:, :, ::1]',
    random_im='double',
    random_re='double',
    shape=tuple,
)
def get_primordial_phases(slab):
    shape = asarray(slab).shape
    # The global gridsize is equal to
    # the first (1) dimension of the slab.
    gridsize = shape[1]
    nyquist = gridsize//2
    # Make the DC and Nyquist planes of random numbers,
    # respecting the complex-conjugate symmetry. These will be
    # allocated in full on all processes. A seed of master_seed + nprocs
    # (and the next, master_seed + nprocs + 1) is used, as the highest
    # process_seed will be equal to master_seed + nprocs - 1, meaning
    # that this new seed will not collide with any of the individual
    # process seeds.
    plane_dc      = create_symmetric_plane(gridsize, seed=(master_seed + nprocs + 0))
    plane_nyquist = create_symmetric_plane(gridsize, seed=(master_seed + nprocs + 1))
    # Re-seed the pseudo-random number generator
    # with the process specific seed.
    seed_rng()
    # Populate the random grid.
    # Loop through the local j-dimension.
    for j in range(ℤ[shape[0]]):
        j_global = ℤ[shape[0]*rank] + j
        # Loop through the complete i-dimension
        for i in range(gridsize):
            # Loop through the complete, padded k-dimension
            # in steps of 2 (one complex number at a time).
            for k in range(0, ℤ[shape[2]], 2):
                # The k-component of the wave vector (grid units)
                kk = k//2
                # Draw a complex random number from a Gaussian
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
                    # The real and imaginary part individually
                    # have mean 0 and variance 1/√2.
                    random_re = random_gaussian(0, ℝ[1/sqrt(2)])
                    random_im = random_gaussian(0, ℝ[1/sqrt(2)])
                # Store the two random numbers
                slab[j, i, k    ] = random_re
                slab[j, i, k + 1] = random_im



# Read in definitions from CLASS source files at import time
cython.declare(class__VERSION_=str,
               class__ARGUMENT_LENGTH_MAX_='Py_ssize_t',
               class_a_min='double',
               )
for (varname,
     filename,
     declaration_type,
     default_value) in [('_VERSION_'            , 'include/common.h'      , 'macro'   , ''   ),
                        ('_ARGUMENT_LENGTH_MAX_', 'include/parser.h'      , 'macro'   , 1024 ),
                        ('a_min'                , 'source/perturbations.c', 'variable', 0.001),
                        ]:
    if master:
        if declaration_type == 'macro':
            pattern = f'(^|[^0-9a-zA-Z_])#define\s+{varname}\s+(.+?)(/\*| |//|;|\n|$)'
        elif declaration_type == 'variable':
            pattern = f'(^|[^0-9a-zA-Z_]){varname}\s*=\s*(.*?)(/\*| |//|;|\n|$)'
        filename_abs = f'{paths["class_dir"]}/{filename}'
        try:
            with open(filename_abs, 'r') as class_file:
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
    elif varname == 'a_min':
        class_a_min = value
