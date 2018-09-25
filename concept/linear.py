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
    # Names of all implemented transfer function variables.
    # Methods with these names will be defined, which will return
    # the corresponding transfer function as a function of k,
    # for a given a.
    transfer_function_variable_names = ('δ', 'θ', 'δP', 'σ', 'hʹ', 'H_Tʹ')
    # Names of scalar attributes
    attribute_names = ('A_s', 'n_s', 'alpha_s', 'k_pivot', 'h')
    # Initialize instance
    def __init__(self, params, k_magnitudes, cosmo=None, filename='', class_call_reason=''):
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
        # Only part of the computed CLASS data is needed.
        # Below, the keys corresponding to the needed fields of CLASS
        # data is written as regular expressions.
        # This dict need to be an instance variable, as it may be
        # mutated by the methods.
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
                r'^h_prime$',
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
            # object based on the hash of the CLASS parameters and the
            # CLASS variables _VERSION, _ARGUMENT_LENGTH_MAX_ and a_min.
            self.id = hashlib.sha1(str(
                tuple(sorted({str(key).replace(' ', '').lower(): str(val).replace(' ', '').lower()
                    for key, val in self.params.items()}.items()))
                + tuple(sorted([str(val).replace(' ', '').lower()
                    for val in class_extra_background]))
                + tuple(sorted([str(val).replace(' ', '').lower()
                    for val in class_extra_perturbations]))
                + (class__VERSION_, class__ARGUMENT_LENGTH_MAX_, class_a_min)
            ).encode()).hexdigest()
            self.filename = f'{paths["reusables_dir"]}/class/{self.id}.hdf5'
        # Message that gets printed if and when CLASS is called
        self.class_call_reason = class_call_reason
        # Add methods which returns transfer function splines
        # for a given a.
        def construct_func(var_name):
            return (
                lambda a=-1, component=None, get='as_function_of_k':
                    self.transfer_function(var_name, component, get, a)
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
                self._cosmo, self.k_node_indices = call_class(
                    self.params,
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
    def alpha_s(self):
        if not hasattr(self, '_alpha_s'):
            if not self.load('alpha_s'):
                # Get alpha_s from CLASS
                self._alpha_s = self.cosmo.get_current_derived_parameters(['alpha_s'])['alpha_s']
                # Save to disk
                self.save('alpha_s')
            # Communicate
            self._alpha_s = bcast(self._alpha_s if master else None)
        return self._alpha_s
    @property
    def k_pivot(self):
        if not hasattr(self, '_k_pivot'):
            if not self.load('k_pivot'):
                # Retrieve k_pivot from the CLASS params.
                # If not defined there, default to the standard CLASS
                # value of 0.05 Mpc⁻¹. We store this in CLASS units.
                self._k_pivot = float(self.params.get('k_pivot', 0.05))
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
                if master and not special_params.get('keep_class_extra_background', False):
                    for key in set(self._background.keys()):
                        if not any([key == pattern or re.search(pattern, key)
                            for pattern in class_extra_background]
                        ):
                            continue
                        if any([key == pattern or re.search(pattern, key)
                            for pattern in self.needed_keys['background']]
                        ):
                            continue
                        del self._background[key]
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
                'cdm'   :  0,
                'b'     :  0,
                'g'     :  1/3,
                'ur'    :  1/3,
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
            # The special "metric" CLASS species needs to be assigned
            # some background density, but since we get δρ directly
            # from CLASS and neither δ nor ρ_bar has any
            # physical meaning, this background density can be
            # chosen arbitrarily. Since CO𝘕CEPT relies on
            # ϱ_bar = a**(3*(1 + w_eff(a)))*ρ_bar(a)
            # being constant for all species, we also need to assign
            # the "metric" species an effective equation of
            # state w_eff(a). This is constructed automatically from the
            # equation of state w(a), which in turn is constructed from
            # the background pressure. Thus, though the "metric" species
            # has no pressure in any meaningful sense, we need to assign
            # it the background pressure that matchea the assigned
            # background density. In principle we could steal the
            # background density and pressure from any of the other,
            # physical species, or construct a new pair of consistent
            # background densities and pressures. However, it turns out
            # that δρ(k, a) goes approximately like a⁻⁴ for a given k
            # (really it is the envelope of its oscillatins that has
            # this behaviour), which is also the behaviour of photons.
            # Choosing the photon background density and pressure
            # for the "metric" species then leads to more
            # accurate splines, which is useful to better
            # resolve the many oscillaions in δρ.
            self._background['(.)rho_metric'] = self._background['(.)rho_g']
            self._background['(.)p_metric'] = self._background['(.)p_g']
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
                            if any([key == pattern or re.search(pattern, key) for pattern in (
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
                # the C-space memory and delete any extra CLASS
                # perturbations (which have now been saved to disk).
                self.load_everything('perturbations')
                self.cosmo.struct_cleanup()
                # Now remove the extra CLASS perturbations
                # not used by this simulation.
                if node_master and not special_params.get('keep_class_extra_perturbations', False):
                    for key in set(self._perturbations[0].keys()):
                        if not any([key == pattern or re.search(pattern, key)
                            for pattern in class_extra_perturbations]
                        ):
                            continue
                        if any([key == pattern or re.search(pattern, key)
                            for pattern in self.needed_keys['perturbations']]
                        ):
                            continue
                        for perturbation in self._perturbations:
                            del perturbation[key]
            # As we only need perturbations defined within the
            # simulation timespan, a >= a_begin, we now cut off the
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
            self.load_everything('perturbations')
            # After the CLASS perturbations needed for the special
            # "metric" species has been computed/loaded, we need to
            # manually construct the corresponding δ perturbations
            # out of these.
            if 'metric' in class_species_present_list:
                self.construct_delta_metric()
        return self._perturbations
    # Method which makes sure that everything is loaded
    def load_everything(self, already_loaded=None):
        """If some attribute is already loaded, it can be specified
        as the already_loaded argument. This is crucial to specify when
        called from within one of the methods matching an attribute.
        """
        attributes = {
            *self.attribute_names,
            'background',
            'perturbations',
        }
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
        H_T' in N-body gauge already exist as perturbations.
        The strategy is as follows: For each k, we can compute the GR
        correction potential γ(a) using
        γ(a) = -(H_Tʹʹ(a) + a*H(a)*H_Tʹ(a))/k² + (ϕ(a) - ψ(a)),
        where ʹ denotes differentiation with respect to
        conformal time τ. To get H_Tʹʹ (actually Ḣ_Tʹ, see below) we
        construct a TransferFunction object over the H_Tʹ perturbations.
        This takes up the majority of the computation time and is done
        in parallel. After this, every processes will compute
        γ(a) for all k.
        The units of the perturbations from CLASS are as follows:
        H_Tʹ: [time⁻¹]        = [c/Mpc],
        ϕ   : [length²time⁻²] = [c²],
        ψ   : [length²time⁻²] = [c²],
        and so γ also gets units of [length²time⁻²]. Note that H_T is
        some times defined to have units of [length²]. The H_T_prime
        from CLASS follows the unitless convention of
        https://arxiv.org/pdf/1708.07769.pdf
        We choose to compute k²γ, not γ by itself.
        Using ʹ = d/dτ = a*d/dt = a²H(a)*d/da, we have
        k²γ(a) = -a*H(a)(a*Ḣ_Tʹ(a) + H_Tʹ(a)) + k²(ϕ(a) - ψ(a))
        with ˙ = d/da. The δρ(a) perturbation is now given by
        δρ(a) = 2/3*γ(a)k²/a² * 3/(8πG)
              = k²γ(a)/(4πGa²)
        where the factor 3/(8πG) = 1 in CLASS units.
        Side-note: In this form (k²γ = 4πGa²δρ) it is clear that γ
        indead is a potential. The missing sign comes from the CLASS
        convention (which is adopted in CO𝘕CEPT) of having δ
        (and hence δρ) transfer functions be negative.
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
        if 'delta_metric' in self._perturbations[0]:
            return
        masterprint('Constructing metric δ perturbations ...')
        # Get the H_Tʹ(k, a) transfer functions
        transfer_H_Tʹ = self.H_Tʹ(get='object')
        # Construct the "metric" δ(a) for each k
        for k, perturbation in enumerate(self._perturbations):
            k_magnitude = self.k_magnitudes[k]
            # Extract needed perturbations along with
            # the scalefactor at which they are tabulated.
            a     = perturbation['a'        ]
            ϕ     = perturbation['phi'      ]*ℝ[light_speed**2]
            ψ     = perturbation['psi'      ]*ℝ[light_speed**2]
            H_Tʹ  = perturbation['H_T_prime']*ℝ[light_speed/units.Mpc]
            θ_tot = perturbation['theta_tot']*ℝ[light_speed/units.Mpc]
            # Compute the derivative of H_Tʹ with respect to a
            Ḣ_Tʹ = asarray([transfer_H_Tʹ.eval_deriv(k, a_i) for a_i in a])
            # Lastly, we need the Hubble parameter and the mean density
            # of the "metric" species at the times given by a.
            H = asarray([hubble(a_i) for a_i in a])
            ρ_metric = self.ρ_bar(a, 'metric')
            # Construct the γ potential
            aH = a*H
            k_magnitude2 = k_magnitude**2
            k2γ = -aH*(a*Ḣ_Tʹ + H_Tʹ) + k_magnitude2*(ϕ - ψ)
            # Construct the δ perturbation (in N-body gauge)
            δ = k2γ/(ℝ[4*π*G_Newton]*a**2*ρ_metric)
            # Transform from N-body gauge to synchronous gauge
            w_metric = asarray([self.w(a_i, 'metric') for a_i in a])
            δ -= ℝ[3/light_speed**2]*aH*(1 + w_metric)*θ_tot/k_magnitude2
            # Store the "metric" δ perturbations,
            # now in synchronous gauge.
            perturbation['delta_metric'] = δ
        masterprint('done')

    # Method which constructs TransferFunction instances and use them
    # to compute and store transfer functions. Do not use this method
    # directly, but rather call e.g. cosmoresults.δ(a, component).
    # Note that the transfer functions returned by this method are those
    # gotten from get_perturbations, not get_transfer.
    def transfer_function(self, var_name, component=None, get='object', a=-1):
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
            # By far the most background variables are power laws in a.
            # A few exceptions are the constant pressure of the cdm, b
            # and lambda CLASS species, the constant density of the
            # lambda CLASS species, as well as the density, pressure
            # and equation of state w for the fld CLASS species,
            if y in {'(.)p_cdm', '(.)p_b', '(.)p_lambda', '(.)rho_lambda'}:
                logx, logy = True, False
            elif y in {'(.)rho_fld', '(.)p_fld', '(.)w_fld'}:
                logx, logy = False, False
            elif y.startswith('(.)rho_') or y.startswith('(.)p_') or y in {
                'z',
                'a',
                'H [1/Mpc]',
                'proper time [Gyr]',
                'conf. time [Mpc]',
                'gr.fac. D',
                'gr.fac. f',
            }:
                logx, logy = True, True
            else:
                logx = True
                logy = not np.any(asarray(self.background[y]) <= 0)
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
        # As we have done no unit convertion, the ratio P_bar/ρ_bar
        # gives us the unitless w.
        return P_bar/ρ_bar
    # Method for looking up the linear growth rate f_growth = H⁻¹Ḋ/D
    # (with D the linear growth factor) at some a.
    @functools.lru_cache()
    def growth_fac_f(self, a):
        spline = self.splines('gr.fac. f')
        return spline.eval(a)
    # Method for appending a piece of raw CLASS data to the dump file
    def save(self, element):
        """You should not call this method unless you have good reason
        to believe that 'element' is not already present in the file,
        as this method will open the file in read/write ('a') mode
        regardless. This can be dangeous as HDF5 build with MPI is not
        thread-safe, and so if two running instances of CO𝘕CEPT with the
        same params run this method simultaneously, problems
        may arise. From HDF5 1.10 / H5Py 2.5.0, multiple processes can
        read from the same file, as long as it is not opened in write
        mode by any process. Thus, this complication is only relevent
        for this method. The open_hdf5 function is ment to alleviate
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
                    if any([key.replace('__per__', '/') == pattern
                        or re.search(pattern, key.replace('__per__', '/'))
                        for pattern in self.needed_keys['background']
                            | (class_extra_background
                                if special_params.get('keep_class_extra_background', False)
                                else set()
                            )
                    ])
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
                        if any([key.replace('__per__', '/') == pattern
                            or re.search(pattern, key.replace('__per__', '/'))
                            for pattern in self.needed_keys['perturbations']
                                | (class_extra_perturbations
                                    if special_params.get('keep_class_extra_perturbations', False)
                                    else set()
                                )
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
                    for perturbation_missing in self.needed_keys['perturbations']
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
        double k_max
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
        self.k_max = class_k_max.get('all', ထ)
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
        a_values_k=object,  # np.ndarray
        a_values_largest_trusted_k='double[::1]',
        approximate_P_as_wρ='bint',
        available='bint',
        class_perturbation_name=str,
        class_species=str,
        class_units='double',
        exponent='double',
        exponent_max='double',
        factor='double',
        fitted_trends=list,
        has_data='bint',
        i='Py_ssize_t',
        index='Py_ssize_t',
        k='Py_ssize_t',
        k_end='Py_ssize_t',
        k_send='Py_ssize_t',
        k_size='Py_ssize_t',
        k_start='Py_ssize_t',
        largest_trusted_k='Py_ssize_t',
        loga_values_largest_trusted_k='double[::1]',
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
        perturbation_values_k=object,  # np.ndarray
        perturbations=list,
        perturbations_available=dict,
        perturbations_detrended='double[::1]',
        perturbations_detrended_largest_trusted_k='double[::1]',
        perturbations_detrended_k='double[::1]',
        perturbations_largest_trusted_k=object,  # np.ndarray
        rank_send='int',
        size='Py_ssize_t',
        spline='Spline',
        trend=object,  # np.ndarray
        untrusted_perturbations=list,
        weights=object,  # np.ndarray
        weights_species=dict,
        Σweights=object,  # np.ndarray
    )
    def process(self):
        # Ensure that the cosmological background has been loaded
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
            'The {} perturbations ',
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
            'δ'   : 'delta_{}',
            'θ'   : 'theta_{}',
            'δP'  : 'cs2_{}',  # Note that cs2 is really δP/δρ
            'σ'   : 'shear_{}',
            'hʹ'  : 'h_prime',
            'H_Tʹ': 'H_T_prime',
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
        largest_trusted_k = -1
        untrusted_perturbations = [None]*self.k_gridsize
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
                                self.get_perturbation(perturbation_k, f'delta_{class_species}')
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
                            perturbation = self.get_perturbation(
                                perturbation_k, f'cs2_{class_species}')
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
                    elif self.var_name == 'H_Tʹ':
                        # As H_Tʹ is a species independent quantity,
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
                        perturbation = self.get_perturbation(
                            perturbation_k, class_perturbation_name.format(class_species))
                        if perturbation is None:
                            perturbations_available[class_species] = False
                        else:
                            perturbation_values_arr += weights*class_units*perturbation
                if isinstance(perturbation_values_arr, int):
                    perturbation_values = np.array((), dtype=C2np['double'])
                else:
                    perturbation_values = perturbation_values_arr
                # Warn or abort on missing perturbations.
                # We only do this for k = 0, which is the first
                # perturbation encountered on the master process.
                if not approximate_P_as_wρ:
                    if k == 0 and not all(perturbations_available.values()):
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
                                f'No {class_perturbation_name} perturbations '
                                + ('' if self.component is None
                                    else f'for the {self.component.name} component ')
                                + f'available'
                            )
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
                a_values, perturbation_values = remove_doppelgängers(
                    a_values, perturbation_values, copy=True)
                # Perform non-linear detrending. The data to be splined
                # is in the form {log(a), perturbation_values - trend},
                # with trend = factor*a**exponent. Here we find this
                # trend trough curve fitting of perturbation_values.
                fitted_trends = []
                for bounds in (
                    ([-ထ, -exponent_max], [+ထ,  0           ]),
                    ([-ထ,  0           ], [+ထ, +exponent_max]),
                ):
                    try:
                        fitted_trends.append(
                            scipy.optimize.curve_fit(
                                self.power_law,
                                a_values,
                                perturbation_values,
                                (1, 0),
                                bounds=bounds,
                            )
                        )
                    except:
                        pass
                if fitted_trends:
                    self.factors[k], self.exponents[k] = fitted_trends[
                        np.argmin([fitted_trend[1][1,1] for fitted_trend in fitted_trends])
                    ][0]
                else:
                    warn(
                        f'Failed to detrend {self.var_name} perturbations '
                        + ('' if self.component is None else f'for {self.component.name} ')
                        + f'at k = {self.k_magnitudes[k]} {unit_length}⁻¹. '
                        f'The simulation will carry on without this detrending.'
                    )
                    self.factors[k], self.exponents[k] = 0, 1
                if abs(self.factors[k]) == ထ:
                    abort(
                        f'Error processing {self.var_name} perturbations '
                        + ('' if self.component is None else f'for {self.component.name} ')
                        + f'at k = {self.k_magnitudes[k]} {unit_length}⁻¹: '
                        f'Detrending resulted in factor = {self.factors[k]}.'
                    )
                # When the exponent is found to be 0, there is no reason
                # to keep a non-zero factor as the detrending is then
                # just a constant offset.
                if isclose(self.exponents[k], 0, rel_tol=1e-9, abs_tol=1e-6):
                    self.factors[k], self.exponents[k] = 0, 1
                # If the exponent is found to be equal to the maximum
                # allowed exponent, it means that the detrending has not
                # converged. This failed detrending is usually an
                # indication that CLASS has been run with too
                # low precision. For H_Tʹ and δ for the
                # "metric" species, however, this is not necessarily
                # the case, and so here we allow for such
                # extreme exponents.
                if (    isclose(abs(self.exponents[k]), exponent_max)
                    and not self.var_name == 'H_Tʹ'
                    and not (self.var_name == 'δ' and self.component.class_species == 'metric')
                ):
                    sign_str = ''
                    if self.exponents[k] < 0:
                        sign_str = '-'
                    masterwarn(self.var_name)
                    abort(
                        f'Error processing {self.var_name} perturbations '
                        + ('' if self.component is None else f'for {self.component.name} ')
                        + f'at k = {self.k_magnitudes[k]} {unit_length}⁻¹: '
                        f'Detrending resulted in exponent = '
                        f'{sign_str}exponent_max = {sign_str}{exponent_max}.'
                    )
                trend = self.factors[k]*asarray(a_values)**self.exponents[k]
                perturbations_detrended = asarray(perturbation_values) - trend
            # Communicate the spline data
            for rank_send in range(nprocs):
                # Broadcast the k value belonging to the data to
                # be communicated. If no data should be communicated,
                # signal this by broadcasting -1.
                k_send = bcast(k if has_data else -1, root=rank_send)
                if k_send == -1:
                    continue
                # Broadcast the trend
                self.factors[k_send], self.exponents[k_send] = bcast(
                    (self.factors[k], self.exponents[k]),
                    root=rank_send,
                )
                # Broadcast the data
                a_values_k = smart_mpi(
                    a_values if rank == rank_send else None,
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
                # If k_send is above that of self.k_max, it means that
                # this particular perturbation is not trusted at this
                # high k value. When this is the case, we do not
                # construct a spline object.
                self.k_max = allreduce(self.k_max, op=MPI.MIN)
                if self.k_magnitudes[k_send] > self.k_max:
                    # If not trusted, save the data. We will process it
                    # further once all the trusted perturbations have
                    # been processed.
                    untrusted_perturbations[k_send] = (
                        asarray(a_values_k).copy(),
                        asarray(perturbations_detrended_k).copy(),
                    )
                    continue
                # Take notice of the largest trusted k
                if k_send > largest_trusted_k:
                    largest_trusted_k = k_send
                    a_values_largest_trusted_k = asarray(a_values_k).copy()
                    perturbations_detrended_largest_trusted_k = (
                        asarray(perturbations_detrended_k).copy()
                    )
                # Construct cubic spline of
                # {log(a), perturbations - trend}.
                spline = Spline(np.log(a_values_k), perturbations_detrended_k,
                    f'detrended {self.class_species} {self.var_name} perturbations '
                    f'as function of log(a) at k = {self.k_magnitudes[k]} {unit_length}⁻¹'
                )
                self.splines[k_send] = spline
                # If class_plot_perturbations is True,
                # plot the detrended perturbation and save it to disk.
                if master and class_plot_perturbations:
                    plot_detrended_perturbations(
                        a_values_k, perturbations_detrended_k, self, k_send,
                    )
        # Now every process contains all trends and splines for all
        # trusted perturbations.
        for k in range(self.k_gridsize):
            if self.splines[k] is not None:
                continue
            # We are at the first untrusted perturbation.
            # If this is not equal to the largest trusted k plus 1,
            # something has gone wrong.
            if k != largest_trusted_k + 1:
                abort(
                    f'Something odd went wrong while constructing untrusted '
                    f'{self.var_name} perturbations'
                    + ('' if self.component is None else f' for {self.component.name} ')
                )
            break
        else:
            # All perturbations are trusted and have been processed
            masterprint('done')
            return
        # We shall now construct splines for the untrusted
        # perturbations. We do this by morphing the detrended data of
        # the perturbation with the largest trusted k into being as
        # similar as possible to the untrusted detrended perturbations.
        # This morphing is done via
        # perturbations_detrended_largest_trusted_k
        #     → (factor*perturbations_detrended_largest_trusted_k
        #        *a_values_largest_trusted_k**exponent),
        # where the factor and exponent are new parameters to be found
        # through minimization.
        # First, ensure that the data for the trusted perturbation with
        # the largest k starts at a = a_begin.
        for i in range(a_values_largest_trusted_k.shape[0]):
            if a_values_largest_trusted_k[i] > universals.a_begin:
                perturbations_detrended_largest_trusted_k[i - 1] = np.interp(
                    universals.a_begin,
                    a_values_largest_trusted_k,
                    perturbations_detrended_largest_trusted_k,
                )
                a_values_largest_trusted_k[i - 1] = universals.a_begin
                a_values_largest_trusted_k = a_values_largest_trusted_k[i-1:]
                perturbations_detrended_largest_trusted_k = (
                    perturbations_detrended_largest_trusted_k[i-1:])
                break
        loga_values_largest_trusted_k = np.log(a_values_largest_trusted_k)
        # Carry out the morphing for each of the untrusted perturbations
        factor, exponent = 1, 0
        for k in range(largest_trusted_k + 1, self.k_gridsize):
            # Interpolate untrusted perturbation onto the a_values for
            # the last trusted perturbation.
            perturbations_detrended_k = np.interp(
                a_values_largest_trusted_k,
                *untrusted_perturbations[k],
            )
            # Do the morphing using minimization
            factor, exponent = scipy.optimize.minimize(
                self.least_squares_morphing,
                (factor, exponent),
                (
                    asarray(a_values_largest_trusted_k),
                    asarray(perturbations_detrended_largest_trusted_k),
                    asarray(perturbations_detrended_k),
                ),
                method='nelder-mead',
            ).x
            # Create the spline
            spline = Spline(
                loga_values_largest_trusted_k,
                (factor*asarray(perturbations_detrended_largest_trusted_k)
                    *asarray(a_values_largest_trusted_k)**exponent
                ),
                f'detrended {self.class_species} {self.var_name} perturbations '
                f'as function of log(a) at k = {self.k_magnitudes[k]} {unit_length}⁻¹ '
                f'(produced from the largest trusted perturbation)'
            )
            self.splines[k] = spline
            # If class_plot_perturbations is True,
            # plot the detrended perturbation and save it to disk.
            if master and class_plot_perturbations:
                plot_detrended_perturbations(*untrusted_perturbations[k], self, k)
        # All trusted perturbations have been processed and all
        # untrusted perturbations have been constructed.
        masterprint('done')

    # Helper functions for the process method
    @staticmethod
    def power_law(a, factor, exponent):
        return factor*a**exponent
    @staticmethod
    def least_squares_morphing(x, a, y, y2):
        factor, exponent = x
        return np.sum((y2 - factor*y*a**exponent)**2)
    @cython.header(
        # Arguments
        perturbation_k=dict,
        perturbation_key=str,
        # Locals
        perturbation=object,  # np.ndarray
        k_max_candidate='double',
        key=str,
        returns=object,  # np.ndarray
    )
    def get_perturbation(self, perturbation_k, perturbation_key):
        # Get the perturbation
        perturbation = perturbation_k.get(perturbation_key)
        # If the perturbation is untrusted for large k,
        # set self.k_max to the largest trusted k if this is lower
        # than the present self.k_max.
        for key, k_max_candidate in class_k_max.items():
            if k_max_candidate < self.k_max:
                if perturbation_key == key:
                    self.k_max = k_max_candidate
                else:
                    try:
                        if re.search(perturbation_key, key):
                            self.k_max = k_max_candidate
                    except:
                        pass
        return perturbation

    # Method for evaluating the k'th transfer function
    # at a given scale factor.
    @cython.pheader(
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
            self.data = empty(self.k_gridsize, dtype=C2np['double'])
        for k in range(self.k_gridsize):
            self.data[k] = self.eval(k, a)
        return self.data

    # Method for evaluating the derivative of the k'th transfer
    # function with respect to the scale factor, at a specific value of
    # the scale factor.
    @cython.pheader(
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
            self.data_deriv = empty(self.k_gridsize, dtype=C2np['double'])
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
                class_call_reason=str,
                # Locals
                cosmoresults=object, # CosmoResults
                extra_params=dict,
                k_gridsize_max='Py_ssize_t',
                k_magnitudes='double[::1]',
                k_magnitudes_str=str,
                params_specialized=dict,
                returns=object,  # CosmoResults
               )
def compute_cosmo(k_min=-1, k_max=-1, k_gridsize=-1,
    gauge='synchronous', filename='', class_call_reason=''):
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
    gauge = class_params.get('gauge', gauge).replace('-', '').lower()
    if gauge not in ('synchronous', 'newtonian', 'nbody'):
        abort(
            f'Gauge was set to "{gauge}" but must be one of '
            f'"N-body", "synchronous", "Newtonian"'
        )
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
    # Determine whether to run CLASS "quickly" or "fully",
    # where only the latter computes the  perturbations.
    if k_min == -1 == k_max:
        # A quick CLASS computation should be carried out,
        # using only the minial set of parameters.
        extra_params = {}
        k_magnitudes = None
    elif k_min == -1 or k_max == -1:
        abort(f'compute_cosmo was called with k_min = {k_min}, k_max = {k_max}')
    else:
        # A full CLASS computation should be carried out.
        # Array of |k| values at which to tabulate the perturbations,
        # in both floating and str representation.
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
    cosmoresults = CosmoResults(
        params_specialized,
        k_magnitudes,
        filename=filename,
        class_call_reason=class_call_reason,
    )
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
    return f'{k:.3e}'.replace('+0', '+').replace('-0', '-').replace('e+0', '')

# Function for computing transfer functions as function of k
@cython.pheader(# Arguments
                component='Component',
                variable=object,  # str or int
                k_min='double',
                k_max='double',
                k_gridsize='Py_ssize_t',
                specific_multi_index=object,  # tuple, int-like or str
                a='double',
                gauge=str,
                get=str,
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
def compute_transfer(
    component, variable, k_min, k_max,
    k_gridsize=-1, specific_multi_index=None, a=-1, gauge='N-body',
    get='spline',
):
    """This function calls compute_cosmo which produces a CosmoResults
    instance which can talk to CLASS. Using the δ, θ, etc. methods on
    the CosmoResults object, TransferFunction instances are
    automatically created. All this function really implements
    are then the optional gauge transformations.
    The return value is either (spline, cosmoresults) (get == 'spline')
    or (array, cosmoresults) (get == 'array'), where spline is a Spline
    object of the array.
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
        k_min,
        k_max,
        k_gridsize,
        'synchronous' if gauge == 'nbody' else gauge,
        class_call_reason=f'in order to get "{component.name}" perturbations ',
    )
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
        # to N-body gauge, if requested. Note that the special "metric"
        # CLASS species is constructed directly in N-body gauge
        # and so does not need any transformation.
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
    options=dict,
    use_gridˣ='bint',
    # Locals
    A_s='double',
    H='double',
    Jⁱ_ptr='double*',
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
    i_global='Py_ssize_t',
    index='Py_ssize_t',
    index0='Py_ssize_t',
    index1='Py_ssize_t',
    j='Py_ssize_t',
    j_global='Py_ssize_t',
    k='Py_ssize_t',
    k_global='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kj2='Py_ssize_t',
    kk='Py_ssize_t',
    k_factor='double',
    k_gridsize='Py_ssize_t',
    k_gridvec='Py_ssize_t[::1]',
    k_magnitude='double',
    k_max='double',
    k_min='double',
    k_pivot='double',
    k2='Py_ssize_t',
    k2_max='Py_ssize_t',
    mass='double',
    momⁱ='double*',
    multi_index=object,  # tuple or str
    n_s='double',
    nyquist='Py_ssize_t',
    option_key=str,
    options_linear=dict,
    option_val=object,  # str or bool
    pariclevar_name=str,
    posⁱ='double*',
    pos_gridpoint='double',
    processed_specific_multi_index=object,  # tuple or str
    slab='double[:, :, ::1]',
    slab_jik='double*',
    sqrt_power='double',
    sqrt_power_common='double[::1]',
    slab_structure='double[:, :, ::1]',
    slab_structure_info=dict,
    structure_jik='double*',
    tensor_rank='int',
    transfer='double',
    transfer_spline_δ='Spline',
    uⁱ_noghosts='double[:, :, :]',
    w='double',
    w_eff='double',
    α_s='double',
    δ_min='double',
    ψⁱ='double[:, :, ::1]',
    ψⁱ_noghosts='double[:, :, :]',
    ςⁱⱼ_ptr='double*',
    ϱ_bar='double',
    ϱ_ptr='double*',
    𝒫_ptr='double*',
)
def realize(component, variable, transfer_spline, cosmoresults,
            specific_multi_index=None, a=-1, options=None,
            use_gridˣ=False):
    """This function realizes a single variable of a component,
    given the transfer function as a Spline (using |k⃗| in physical units
    as the independent variable) and the corresponding CosmoResults
    object, which carry additional information from the CLASS run that
    produced the transfer function. If only a single fluidscalar of the
    fluid variable should be realized, the multi_index of this
    fluidscalar may be specified. If you want a realization at a time
    different from the present you may specify an a.
    If a particle component is given, the Zel'dovich approximation is
    used to distribute the paricles and assign momenta. This is
    done simultaneously, meaning that you cannot realize only the
    positions or only the momenta. For the particle realization to
    work correctly, you must pass the δ transfer function as
    transfer_spline. For particle components, the variable argument
    is not used.

    Several options has to be specified to define how the realization is
    to be carried out. These options are contained in the "options"
    argumen. By default, the options are
    options = {
        # Linear realization options
        'velocities from displacements': False,
        # Non-linear realization options
        'structure'     : 'primordial',
        'compound-order': 'linear',
    }
    which corresponds to linear realization. For particle components
    (which can not be realized continually) only linear realization is
    possible, and thus only the linear option matters. When
    'velocities from displacements' is True, the particle momenta will
    be set from the same displacement field ψⁱ as is used for the
    positions, using the linear growth rate f to convert between
    displacement and velocity. Otherwise, momenta will be constructed
    from their own velocity field uⁱ, using their own transfer function
    but the same (primordial) noise.
    Another linear option 'back-scaling' might be specified, but it is
    not used by this function.
    Taking Jⁱ as an example of a fluid variable realization,
    linear realization looks like
        Jⁱ(x⃗) = a**(1 - 3w_eff)ϱ_bar(1 + w)ℱₓ⁻¹[T_θ(k)ζ(k)K(k⃗)ℛ(k⃗)],
    where
        ζ(k) = π*sqrt(2*A_s)*k**(-3/2)*(k/k_pivot)**((n_s - 1)/2)
                *exp(α_s/4*log(k/k_pivot)**2)
    is the primordial curvature perturbation
    (see the primordial_analytic_spectrum() function in the CLASS
    primordial.c file for a reference), T_θ(k) is the passed
    transfer function for θ, ℛ(k⃗) is a field of primordial noise,
    and K(k⃗) is the tensor structure (often referred to as the k factor)
    needed to convert from θ to uⁱ. For uⁱ, K(k⃗) = -ikⁱ/k². The factors
    outside the Fourier transform then converts from uⁱ to Jⁱ.
    We can instead choose to use the non-linearly evolved structure
    of ϱ, by using options['structure'] == 'non-linear'. Then the
    realization looks like
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
    # By default, use linear realization options and do not construct
    # the velocities directly from the displacements.
    options_linear = {
        # Linear options
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
            'velocitiesfromdisplacements',
            'backscaling',
            'structure',
            'compoundorder',
        }:
            abort(f'Did not understand realization option "{option_key}"')
    if options['structure'] not in ('primordial', 'nonlinear'):
        abort('Unrecognized value "{}" for options["structure"]'
            .format(options['structure']))
    if options['compoundorder'] not in ('linear', 'nonlinear'):
        abort('Unrecognized value "{}" for options["compound-order"]'
            .format(options['compoundorder']))
    options['velocitiesfromdisplacements'] = bool(options['velocitiesfromdisplacements'])
    # Get the index of the fluid variable to be realized
    # and print out progress message.
    processed_specific_multi_index = ()
    pariclevar_name = 'pos'
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
        pariclevar_name = {0: 'pos', 1: 'mom'}[fluid_index]
        # When the 'velocities from displacements' option is enabled,
        # both the positions and the momenta are constructed from the
        # displacement field ψⁱ. It is then illegal to request a momenta
        # realization directly.
        if pariclevar_name == 'mom' and options['velocitiesfromdisplacements']:
            abort(
                f'A realization of particle momenta for component "{component.name}" '
                f'was requested, but this component is supposed to get its velocities '
                f'from the displacements.'
            )
        if specific_multi_index is None:
            masterprint(
                'Realizing particle',
                'positions and momenta' if options['velocitiesfromdisplacements']
                    else {'pos': 'positions', 'mom': 'momenta'}[pariclevar_name],
                f'of {component.name} ...'
            )
        else:
            processed_specific_multi_index = (
                component.fluidvars[fluid_index].process_multi_index(specific_multi_index)
            )
            if options['velocitiesfromdisplacements']:
                masterprint(
                    f'Realizing particle positions[{processed_specific_multi_index[0]}] '
                    f'and momenta[{processed_specific_multi_index[0]}] of {component.name} ...'
                )
            else:
                masterprint(
                    f'Realizing particle',
                    {'pos': 'positions', 'mom': 'momenta'}[pariclevar_name]
                        + f'[{processed_specific_multi_index[0]}] '
                    f'of {component.name} ...'
                )
        # For particles, the Zel'dovich approximation is used for the
        # realization. For the positions, the displacement field ψⁱ is
        # really what is realized, while for the momenta, the velocity
        # field uⁱ is what is really realized. Both of these are vector
        # fields, and so we have to set fluid_index to 1 so that
        # multi_index takes on vector values ((0, ), (1, ), (2, )).
        fluid_index = 1
    elif component.representation == 'fluid':
        fluidvar_name = component.fluid_names['ordered'][fluid_index]
        if specific_multi_index is None:
            masterprint(f'Realizing {fluidvar_name} of {component.name} ...')
        else:
            processed_specific_multi_index = (
                component.fluidvars[fluid_index].process_multi_index(specific_multi_index)
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
    # A compound order of 'nonlinear' only makes a difference for
    # compound variables; that is, Jⁱ and ςⁱⱼ. If what we are realizing
    # is another variable, switch this back to 'linear'.
    if fluid_index == 1:
        # We are realizing Jⁱ
        compound_variable = True
    elif fluid_index == 2 and processed_specific_multi_index != 'trace':
        # We are realizing ςⁱⱼ
        compound_variable = True
    else:
        compound_variable = False
    if not compound_variable:
        if options['compoundorder'] == 'nonlinear':
            options['compoundorder'] = 'linear'
    # Abort if the non-linear structure option was passed
    # for a particle component, as these can only be realized
    # from primordial noise.
    if (component.representation == 'particles'
        and options['structure'] != options_linear['structure']
    ):
        abort('Can only do particle realization using primordial noise/structure')
    # When realizing δ, it only makes sense to realize it linearly
    if fluid_index == 0 and options['structure'] != options_linear['structure']:
        abort('Can only do linear realization of δ')
    # Extract various variables
    nyquist = gridsize//2
    H = hubble(a)
    w = component.w(a=a)
    w_eff = component.w_eff(a=a)
    ϱ_bar = component.ϱ_bar
    if cosmoresults is not None:
        A_s = cosmoresults.A_s
        n_s = cosmoresults.n_s
        α_s = cosmoresults.alpha_s
        k_pivot = cosmoresults.k_pivot
    # Fill 1D array with values used for the realization.
    # These values are the k (but not k⃗) dependent values inside the
    # inverse Fourier transform, not including any additional tenstor
    # structure (the k factors K(k⃗)).
    k2_max = 3*(gridsize//2)**2  # Max |k⃗|² in grid units
    sqrt_power_common = get_buffer(k2_max + 1,
        # Must use some buffer different from the one used to do the
        # domain decomposition of ψⁱ below.
        0,
    )
    if options['structure'] == 'nonlinear':
        # When using the non-linear structure of δϱ to do
        # the realizations, we need the transfer function of δϱ,
        # which is just ϱ_bar times the transfer function of δ.
        k_min, k_max, k_gridsize = get_default_k_parameters(gridsize)
        transfer_spline_δ, cosmoresults_δ = compute_transfer(
            component, 0, k_min, k_max, k_gridsize, a=a,
        )
    for k2 in range(1, k2_max + 1):
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        transfer = transfer_spline.eval(k_magnitude)
        with unswitch:
            if options['structure'] == 'primordial':
                # Realize using ℱₓ⁻¹[T(k) ζ(k) K(k⃗) ℛ(k⃗)],
                # with K(k⃗) capturing any tensor structure.
                # The k⃗-independent part needed here is T(k)ζ(k),
                # with T(k) the supplied transfer function and
                # ζ(k) = π*sqrt(2*A_s)*k**(-3/2)*(k/k_pivot)**((n_s - 1)/2)
                #          *exp(α_s/4*log(k/k_pivot)**2)
                # the primordial curvature perturbations.
                # The remaining ℛ(k⃗) is the primordial noise.
                sqrt_power_common[k2] = (
                    # T(k)
                    transfer
                    # ζ(k)
                    *ℝ[π*sqrt(2*A_s)*k_pivot**(0.5 - 0.5*n_s)
                        # Fourier normalization
                        *boxsize**(-1.5)
                    ]*k_magnitude**ℝ[0.5*n_s - 2]
                    *exp(ℝ[0.25*α_s]*log(k_magnitude*ℝ[1/k_pivot])**2)
                )
            elif options['structure'] == 'nonlinear':
                # Realize using ℱₓ⁻¹[T(k)/T_δϱ(k) K(k⃗) ℱₓ[δϱ(x⃗)]],
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
    # real-space mean value of zero of the realized variable.
    sqrt_power_common[0] = 0
    # Fetch a slab decomposed grid for storing the entirety of what is
    # to be inverse Fourier transformed. As we cannot reuse data from
    # previous calls, we do not pass in a specific buffer name.
    slab = get_fftw_slab(gridsize)
    # Fetch a slab decomposed grid for storing the structure. If this is
    # the first time we perform a realization of this size, the grid
    # will be allocated, otherwise the previous grid will be returned,
    # still containing the previous data.
    slab_structure = get_fftw_slab(gridsize, 'slab_structure')
    # Information about the data from the previous call
    # is stored in the module level slab_structure_previous_info dict.
    # To see if we can reuse the slab_structure as is, we compare this
    # information with that of the current realization.
    slab_structure_info = {
        'structure': options['structure'],
        'a': a,
        'use_gridˣ': use_gridˣ,
        'gridsize': gridsize,
    }
    if slab_structure_info['structure'] == 'primordial':
        # The slab_structure contain no non-linear information,
        # and so it is of no importance at what time slab_structure
        # was made, or whether using the starred or unstarred grids.
        slab_structure_info['a'] = None
        slab_structure_info['use_gridˣ'] = None
    if slab_structure_info != slab_structure_previous_info:
        # Populate slab_structure with either ℛ(k⃗) or ℱₓ[ϱ(x⃗)]
        if options['structure'] == 'primordial':
            # Populate slab_structure with primordial noise ℛ(k⃗)
            generate_primordial_noise(slab_structure)
        elif options['structure'] == 'nonlinear':
            # Populate slab_structure with ℱₓ[ϱ(x⃗)]
            masterprint(f'Extracting structure from ϱ of {component.name}')
            slab_decompose(component.ϱ.gridˣ_mv if use_gridˣ else component.ϱ.grid_mv,
                slab_structure)
            fft(slab_structure, 'forward')
        # Remove the k⃗ = 0⃗ mode, leaving ℱₓ[δϱ(x⃗)]
        if master:
            slab_structure[0, 0, 0] = 0  # Real part
            slab_structure[0, 0, 1] = 0  # Imag part
    slab_structure_previous_info.update(slab_structure_info)
    # Allocate 3-vectors which will store componens
    # of the k vector (in grid units).
    k_gridvec = empty(3, dtype=C2np['Py_ssize_t'])
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
        # (i.e. ϱ), 1 for vector (i.e. J), 2 for tensor (i.e. ς)).
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
            kj = j_global - gridsize if j_global > ℤ[gridsize//2] else j_global
            k_gridvec[1] = kj
            kj2 = kj**2
            # Loop through the complete i-dimension
            for i in range(gridsize):
                # The i-component of the wave vector (grid units)
                ki = i - gridsize if i > ℤ[gridsize//2] else i
                k_gridvec[0] = ki
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
                    # the |k⃗| = 0 mode should vanish, leading to a field
                    # with zero mean.
                    if k2 == 0:  # Only ever True for master
                        slab[0, 0, 0] = 0
                        slab[0, 0, 1] = 0
                        continue
                    # Pointer to the [j, i, k]'th element of the slab.
                    # The complex number is then given as
                    # Re = slab_jik[0], Im = slab_jik[1].
                    slab_jik = cython.address(slab[j, i, k:])
                    # When realizing a variable with a tensor structure
                    # (anything but a scalar), the multiplication by
                    # kⁱ amounts to differentiating the grid. For such
                    # Fourier space differentiations, the Nyquist
                    # mode in the dimension of differentiation has to be
                    # explicitly zeroed out for odd differentiation
                    # orders. If not, the resultant grid will not
                    # satisfy the complex conjugate symmetry, and so
                    # will not represent the Fourier transform of a
                    # real-valued grid.
                    with unswitch(3):
                        if tensor_rank == 1:
                            # Vector: First-order differentiation
                            if k_gridvec[index0] == nyquist:
                                slab_jik[0] = 0
                                slab_jik[1] = 0
                                continue
                        elif tensor_rank == 2 and index0 != index1:
                            # Rank 2 tensor with unequal indices:
                            # Two first-order differentiations.
                            if k_gridvec[index0] == nyquist or k_gridvec[index1] == nyquist:
                                slab_jik[0] = 0
                                slab_jik[1] = 0
                                continue
                    # Pointer to the [j, i, k]'th element
                    # of the structure grid.
                    structure_jik = cython.address(slab_structure[j, i, k:])
                    # The square root of the power at this |k⃗|,
                    # disregarding all k⃗-dependent contributions
                    # (from the k factor and the non-linear structure).
                    sqrt_power = sqrt_power_common[k2]
                    # Populate slab_jik dependent on the component
                    # representation and tensor_rank.
                    with unswitch(3):
                        if component.representation == 'particles':
                            # We are realizing either the displacement
                            # field ψⁱ (for the positions) or the
                            # velocity field uⁱ (for the momenta).
                            # These are constructed from the δ and θ
                            # fields, respectively, with the vector
                            # k factor
                            # K(k⃗) = ±ikⁱ/k².
                            # For fluids, fluid_index distinguishes
                            # between the different variables. For
                            # particle positions and momenta, the
                            # corresponding ψⁱ and uⁱ fields are both
                            # vector variables, and so we had to set
                            # fluid_index = 1 in both cases. To
                            # distinguish between particles and momenta
                            # (and hence get the sign in the k factor
                            # correct) we instead make use of
                            # the pariclevar_name variable.
                            k_factor = ℝ[{
                                'pos': +1,
                                'mom': -1,
                                }[pariclevar_name]
                                *boxsize/(2*π)]*k_gridvec[index0]/k2
                            slab_jik[0] = sqrt_power*k_factor*(-structure_jik[1])
                            slab_jik[1] = sqrt_power*k_factor*(+structure_jik[0])
                        elif component.representation == 'fluid':
                            with unswitch(3):
                                if tensor_rank == 0:
                                    # Realize δ or δ𝒫
                                    slab_jik[0] = sqrt_power*structure_jik[0]
                                    slab_jik[1] = sqrt_power*structure_jik[1]
                                elif tensor_rank == 1:
                                    # Realize uⁱ.
                                    # For vectors we have a k factor of
                                    # K(k⃗) = -ikⁱ/k².
                                    k_factor = -(ℝ[boxsize/(2*π)]*k_gridvec[index0])/k2
                                    slab_jik[0] = sqrt_power*k_factor*(-structure_jik[1])
                                    slab_jik[1] = sqrt_power*k_factor*(+structure_jik[0])
                                elif tensor_rank == 2:
                                    # Realize ςⁱⱼ.
                                    # For rank 2 tensors we
                                    # have a k factor of
                                    # K(k⃗) = 3/2(δⁱⱼ/3 - kⁱkⱼ/k²).
                                    k_factor = (ℝ[0.5*(index0 == index1)]
                                        - (1.5*k_gridvec[index0]*k_gridvec[index1])/k2
                                    )
                                    slab_jik[0] = sqrt_power*k_factor*structure_jik[0]
                                    slab_jik[1] = sqrt_power*k_factor*structure_jik[1]
        # Fourier transform the slabs to coordinate space.
        # Now the slabs store the realized grid.
        fft(slab, 'backward')
        # Populate the fluid grids for fluid components,
        # or create the particles via the Zel'dovich approximation
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
                    ϱ_ptr[i] = ϱ_bar*(1 + ℝ[ϱ_ptr[i]])
                δ_min = allreduce(δ_min, op=MPI.MIN)
                if δ_min < -1:
                    masterwarn(f'The realized ϱ of {component.name} has min(δ) = {δ_min:.4g} < -1')
            elif fluid_index == 1:
                Jⁱ_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                if options['compoundorder'] == 'nonlinear':
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
                        Jⁱ_ptr[i] *= ℝ[a**(1 - 3*w_eff)*ϱ_bar*(1 + w)]
            elif fluid_index == 2 and multi_index == 'trace':
                # δP → 𝒫 = 𝒫_bar + a**(3*(1 + w_eff)) * δP
                #        = c²*w*ϱ_bar + a**(3*(1 + w_eff)) * δP
                𝒫_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                for i in range(component.size):
                    𝒫_ptr[i] = ℝ[light_speed**2*w*ϱ_bar] + ℝ[a**(3*(1 + w_eff))]*𝒫_ptr[i]
            elif fluid_index == 2:
                ςⁱⱼ_ptr = fluidscalar.gridˣ if use_gridˣ else fluidscalar.grid
                if options['compoundorder'] == 'nonlinear':
                    # σⁱⱼ → ςⁱⱼ = (ϱ + c⁻²𝒫) * σⁱⱼ
                    ϱ_ptr  = component.ϱ.gridˣ if use_gridˣ else component.ϱ.grid
                    𝒫_ptr  = component.𝒫.gridˣ if use_gridˣ else component.𝒫.grid
                    for i in range(component.size):
                       ςⁱⱼ_ptr[i] *= ϱ_ptr[i] + ℝ[light_speed**(-2)]*𝒫_ptr[i]
                else:
                    # σⁱⱼ → ςⁱⱼ = (ϱ + c⁻²𝒫) * σⁱⱼ
                    #           ≈ ϱ_bar(1 + w) * σⁱⱼ
                    for i in range(component.size):
                        ςⁱⱼ_ptr[i] *= ℝ[ϱ_bar*(1 + w)]
            # Continue with the next fluidscalar
            continue
        # Below follows the Zel'dovich approximation for
        # particle components. When constructing particle positions
        # (momenta), what has just been realized is the displacement
        # (velocity) field ψⁱ (uⁱ), from which we can get the positions
        # (momenta) directly. When using the approximation of also
        # getting the velocities from ψⁱ, the linear growth rate f is
        # used to convert from displacement to velocity. We first
        # domain-decompose the realized field stored in the slabs.
        # Importantly, here we have to use a different buffer from the
        # one already in use# by sqrt_power_common.
        ψⁱ = domain_decompose(slab, 1)
        ψⁱ_noghosts = uⁱ_noghosts = ψⁱ[
            2:(ψⁱ.shape[0] - 2),
            2:(ψⁱ.shape[1] - 2),
            2:(ψⁱ.shape[2] - 2),
        ]
        # Determine and set the mass of the particles
        # if this is still unset.
        if component.mass == -1:
            component.mass = ϱ_bar*boxsize**3/component.N
        mass = component.mass
        # If we are realizing momenta directly from the displacement
        # fiel ψⁱ, get the linear growth rate f_growth = H⁻¹Ḋ/D,
        # with D the linear growth factor.
        if options['velocitiesfromdisplacements']:
            f_growth = cosmoresults.growth_fac_f(a)
        # Apply the Zel'dovich approximation
        dim = multi_index[0]
        posⁱ = component.pos[dim]
        momⁱ = component.mom[dim]
        domain_size_i = ψⁱ_noghosts.shape[0] - 1
        domain_size_j = ψⁱ_noghosts.shape[1] - 1
        domain_size_k = ψⁱ_noghosts.shape[2] - 1
        domain_start_i = domain_layout_local_indices[0]*domain_size_i
        domain_start_j = domain_layout_local_indices[1]*domain_size_j
        domain_start_k = domain_layout_local_indices[2]*domain_size_k
        index = 0
        for         i in range(ℤ[ψⁱ_noghosts.shape[0] - 1]):
            for     j in range(ℤ[ψⁱ_noghosts.shape[1] - 1]):
                for k in range(ℤ[ψⁱ_noghosts.shape[2] - 1]):
                    with unswitch(3):
                        if pariclevar_name == 'pos':
                            # The global x, y or z coordinate at this grid point
                            with unswitch(3):
                                if dim == 0:
                                    i_global = domain_start_i + i
                                    pos_gridpoint = i_global*ℝ[boxsize/gridsize]
                                elif dim == 1:
                                    j_global = domain_start_j + j
                                    pos_gridpoint = j_global*ℝ[boxsize/gridsize]
                                elif dim == 2:
                                    k_global = domain_start_k + k
                                    pos_gridpoint = k_global*ℝ[boxsize/gridsize]
                                # Displace the position of particle
                                # at grid point (i, j, k).
                                displacement = ψⁱ_noghosts[i, j, k]
                                posⁱ[index] = mod(pos_gridpoint + displacement, boxsize)
                            with unswitch(3):
                                if options['velocitiesfromdisplacements']:
                                    # Assign momentum corresponding to the displacement
                                    momⁱ[index] = displacement*ℝ[f_growth*H*mass*a**2]
                        elif pariclevar_name == 'mom':
                            momⁱ[index] = uⁱ_noghosts[i, j, k]*ℝ[mass*a]
                    index += 1
    # Done realizing this variable
    masterprint('done')
    # After realizing particles, most of them will be on the correct
    # process in charge of the domain in which they are located. Those
    # near the domain boundaries might however get displaced outside of
    # their original domain, and so we do need to do an exchange.
    # We can only do this exchange once both the positions and the
    # momenta has been assigned.
    if component.representation == 'particles' and (
        (pariclevar_name == 'pos' and options['velocitiesfromdisplacements'])
        or pariclevar_name == 'mom'
    ):
        exchange(component, reset_buffers=True)
# Module level variable used by the realize function
cython.declare(slab_structure_previous_info=dict)
slab_structure_previous_info = {}

# Function that populates the passed slab decomposed grid
# with primordial noise ℛ(k⃗).
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    # Locals
    face='int',
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    i_conj='Py_ssize_t',
    j='Py_ssize_t',
    j_global='Py_ssize_t',
    j_global_conj='Py_ssize_t',
    k='Py_ssize_t',
    ki_start='Py_ssize_t',
    ki_step='Py_ssize_t',
    ki_stop='Py_ssize_t',
    kj_start='Py_ssize_t',
    kj_step='Py_ssize_t',
    kj_stop='Py_ssize_t',
    kk_start='Py_ssize_t',
    kk_step='Py_ssize_t',
    kk_stop='Py_ssize_t',
    noise_im='double',
    noise_re='double',
    nyquist='Py_ssize_t',
    plane='double[:, :, ::1]',
    plane_dc='double[:, :, ::1]',
    plane_ji='double*',
    plane_ji_conj='double*',
    plane_nyquist='double[:, :, ::1]',
    shell='Py_ssize_t',
    slab_jik='double*',
)
def generate_primordial_noise(slab):
    """Given the already allocated slab, this function will populate
    it with Gaussian (pseudo) random numbers, the stream of which is
    controlled by the random_seed parameter. The slab grid is thought of
    as being in Fourier space, and so these are complex numbers. We wish
    the variance of these complex numbers to equal unity, and so their
    real and imaginary parts are drawn from a distribution
    with variance 1/√2.
    The 3D sequence of random numbers should be independent on the size
    of the grid, in the sense that increasing the grid size should
    amount to just populating the additional "shell" with new random
    numbers, but keeping the random numbers inside of the inner cuboid
    the same. This has the effect that enlarging the grid leaves the
    large-scale structure invariant; one merely add information at
    smaller scales. Additionally, the sequence of random numbers should
    be independent on the number of processes. To achieve all of this,
    we draw the random numbers using the following scheme:
    All processes loop over the entire 3D grid in shells, starting from
    the inner most shell (labelled shell 1) containing (amongst others)
    the (0, 0, 0) point. Since the kk-dimension is cut in half, each
    shell is only tabulated at the kk ≥ 0 half. Thus, each shell
    consists of a kk = constant face, two kj = constant faces and two
    ki = constant faces. Denoting the shell number simply by 'shell',
    the faces are defined by:
        The kk = constant face : kk = shell              , -shell + 1 ≤ ki ≤ shell, -shell < kj ≤ shell,
        The kj = constant faces: kj ∈ {-shell + 1, shell}, -shell + 1 ≤ ki ≤ shell,      0 ≤ kk < shell,
        The ki = constant faces: ki ∈ {-shell + 1, shell}, -shell + 1 < kj ≤ shell,      0 ≤ kk < shell.
    With 0 < shell ≤ nyquist, we hit all points in the 3D grid. The
    (0, 0, 0) point will be part of the kj = 0 face in shell 1. At each
    point, all processes draw the same two random numbers, but only the
    process which owns the given point (determined by the j index that
    goes with kj) assign the random numbers to its local slab.
    The DC and Nyquist planes, defined by kk = 0 and kk = nyquist,
    respectively, need to satisfy the complex conjugacy symmetry of a
    Fourier transformed real field, namely
        plane[k_vec] = plane[-k_vec]*,
    where * means complex conjugation and k_vec is a 2D vector in the
    plane. We enfore this symmetry by letting all processes tabulate
    both planes with random numbers in their entirety. After the whole
    3D grid and the two planes are filled with random numbers, we can
    enforce the symmetry by simply looping over half of the two planes
    and setting each point equal to the conjucate of the corresponding
    symmetric point.
    """
    masterprint('Generating primordial noise ...')
    # The global gridsize is equal to
    # the first (1) dimension of the slab.
    gridsize = slab.shape[1]
    nyquist = gridsize//2
    # Allocate the entire DC and Nyquist plane on all processes
    plane_dc      = empty((gridsize, gridsize, 2), dtype=C2np['double'])
    plane_nyquist = empty((gridsize, gridsize, 2), dtype=C2np['double'])
    # Seed the pseudo random number generator
    # using the same seed on all processes.
    seed_rng(random_seed)
    # Loop through all shells
    for shell in range(1, nyquist + 1):
        # Loop over the three types of faces
        for face in range(3):
            if face == 0:
                # The kk = constant face
                kj_start, kj_stop, kj_step = ℤ[-shell + 1], ℤ[shell + 1], 1
                ki_start, ki_stop, ki_step = ℤ[-shell + 1], ℤ[shell + 1], 1
                kk_start, kk_stop, kk_step =   +shell     , ℤ[shell + 1], 1
            elif face == 1:
                # The two kj = constant faces
                kj_start, kj_stop, kj_step = ℤ[-shell + 1], ℤ[shell + 1], ℤ[2*shell - 1]
                ki_start, ki_stop, ki_step = ℤ[-shell + 1], ℤ[shell + 1],             1
                kk_start, kk_stop, kk_step =            0 ,   shell     ,             1
            elif face == 2:
                # The two ki = constant faces
                kj_start, kj_stop, kj_step = ℤ[-shell + 2],   shell     ,             1
                ki_start, ki_stop, ki_step = ℤ[-shell + 1], ℤ[shell + 1], ℤ[2*shell - 1]
                kk_start, kk_stop, kk_step =            0 ,   shell     ,             1
            # Loop over the face
            for kj in range(kj_start, kj_stop, kj_step):
                j_global = kj + gridsize if kj < 0 else kj
                j = j_global - ℤ[slab.shape[0]*rank]
                for ki in range(ki_start, ki_stop, ki_step):
                    i = ki + gridsize if ki < 0 else ki
                    for kk in range(kk_start, kk_stop, kk_step):
                        # Draw the random numbers
                        noise_re = random_gaussian(0, ℝ[1/sqrt(2)])
                        noise_im = random_gaussian(0, ℝ[1/sqrt(2)])
                        # Populate the local slab
                        with unswitch(2):
                            if 0 <= j < ℤ[slab.shape[0]]:
                                k = 2*kk
                                slab_jik = cython.address(slab[j, i, k:])
                                slab_jik[0] = noise_re
                                slab_jik[1] = noise_im
                        # Populate the DC and Nyquist planes
                        if kk == 0:
                            plane_ji = cython.address(plane_dc[j_global, i, :])
                            plane_ji[0] = noise_re
                            plane_ji[1] = noise_im
                        elif kk == nyquist:
                            plane_ji = cython.address(plane_nyquist[j_global, i, :])
                            plane_ji[0] = noise_re
                            plane_ji[1] = noise_im
    # Enforce the complex conjugacy symmetry on the DC and Nyquist
    # planes. We do this by replacing the random numbers for the
    # elements in the lower j half of each plane with those of the
    # "conjugated" element, situated at the negative k vector.
    # For j_global == j_global_conj, the conjucation is purely along i,
    # and so we may only edit half of the points along this line.
    for k, plane in zip((0, 2*nyquist), (plane_dc, plane_nyquist)):
        for j_global in range(gridsize//2 + 1):
            j = j_global - ℤ[slab.shape[0]*rank]
            # Each process can only change their local slab
            if not (0 <= j < ℤ[slab.shape[0]]):
                continue
            j_global_conj = 0 if j_global == 0 else gridsize - j_global
            for i in range(gridsize):
                i_conj = 0 if i == 0 else gridsize - i
                # Enforce complex conjugate symmetry if necessary.
                # For j_global == j_global_conj, the conjucation is
                # purely along i, and so we may only edit half of the
                # points along this line.
                if 𝔹[j_global == j_global_conj] and i == i_conj:
                    # The complex number is its own conjugate,
                    # so it has to be purely real.
                    slab[j, i, k + 1] = 0
                elif 𝔹[j_global != j_global_conj] or i < ℤ[gridsize//2]:
                    # Enforce conjugacy
                    slab_jik      = cython.address(slab [j            , i     , k:])
                    plane_ji_conj = cython.address(plane[j_global_conj, i_conj,  :])
                    slab_jik[0] = +plane_ji_conj[0]
                    slab_jik[1] = -plane_ji_conj[1]
    masterprint('done')



# Read in definitions from CLASS source files at import time
cython.declare(class__VERSION_=str,
               class__ARGUMENT_LENGTH_MAX_='Py_ssize_t',
               class_a_min='double',
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
        filename_abs = rf'{paths["class_dir"]}/{filename}'
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
        class_a_min = -1.0 if special_params.get('keep_class_extra_background', False) else value
