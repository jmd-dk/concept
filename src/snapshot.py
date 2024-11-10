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
    '    partition,            '
    '    smart_mpi,            '
)
cimport(
    'from mesh import      '
    '    domain_decompose, '
    '    get_fftw_slab,    '
    '    slab_decompose,   '
)
cimport(
    'from species import         '
    '    Component,              '
    '    update_species_present, '
)

# Pure Python imports
from communication import get_domain_info



# Class storing a CO𝘕CEPT snapshot. Besides holding methods for
# saving/loading, it stores component data.
@cython.cclass
class ConceptSnapshot:
    """This class represents the CO𝘕CEPT snapshot type. Besides holding
    the components in the components list, the unit system is declared
    in the units dict. Finally, the cosmological parameters and the
    boxsize is stored in the params dict.
    """
    # The properly written name of this snapshot type
    # (only used for printing).
    name = 'CO𝘕CEPT'
    # The filename extension for this type of snapshot
    extension = '.hdf5'
    # Maximum allowed chunk size in bytes.
    # Large chunks are fine as no temporary buffer is used.
    # The maximum possible chunk size is limited by MPI, though.
    chunk_size_max = 2**30  # 1 GB

    # Class method for identifying a file to be a snapshot of this type
    @classmethod
    def is_this_type(cls, filename):
        if not os.path.isfile(filename):
            return False
        # Test for CO𝘕CEPT format by looking up the 'Ωcdm' attribute
        # in the HDF5 data structure.
        try:
            with open_hdf5(filename, mode='r', raise_exception=True) as hdf5_file:
                hdf5_file.attrs[unicode('Ωcdm')]
                return True
        except Exception:
            pass
        return False

    # Initialisation method
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the ConceptSnapshot type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public dict params
        public list components
        public dict units
        """
        # Dict containing all the parameters of the snapshot
        self.params = {}
        # List of components
        self.components = []
        # Dict containing the base units in str format
        self.units = {}

    # Method that saves the snapshot to an hdf5 file
    @cython.pheader(
        # Argument
        filename=str,
        save_all='bint',
        # Locals
        N='Py_ssize_t',
        N_local='Py_ssize_t',
        N_str=str,
        component='Component',
        end_local='Py_ssize_t',
        fluidscalar='FluidScalar',
        id_max='Py_ssize_t',
        ids_mv_unsigned=object,  # np.ndarray
        indices=object,  # int or tuple
        index='Py_ssize_t',
        multi_index=object,  # tuple or str
        name=object,  # str or int
        plural=str,
        shape=tuple,
        slab='double[:, :, ::1]',
        slab_end='Py_ssize_t',
        slab_start='Py_ssize_t',
        start_local='Py_ssize_t',
        returns=str,
    )
    def save(self, filename, save_all=False):
        # Attach missing extension to filename
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        # Print out message
        masterprint(f'Saving snapshot "{filename}" ...')
        with open_hdf5(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
            # Save used base units
            hdf5_file.attrs['unit time'  ] = self.units['time']
            hdf5_file.attrs['unit length'] = self.units['length']
            hdf5_file.attrs['unit mass'  ] = self.units['mass']
            # Save global attributes
            hdf5_file.attrs['H0']            = correct_float(self.params['H0'])
            hdf5_file.attrs['a']             = correct_float(self.params['a'])
            hdf5_file.attrs['boxsize']       = correct_float(self.params['boxsize'])
            hdf5_file.attrs[unicode('Ωb')]   = correct_float(self.params['Ωb'])
            hdf5_file.attrs[unicode('Ωcdm')] = correct_float(self.params['Ωcdm'])
            # Store each component as a separate group
            # within /components.
            for component in self.components:
                component_h5 = hdf5_file.create_group(f'components/{component.name}')
                component_h5.attrs['species'] = component.species
                if component.representation == 'particles':
                    N, N_local = component.N, component.N_local
                    N_str = get_cubenum_strrep(N)
                    plural = ('s' if N > 1 else '')
                    masterprint(
                        f'Writing out {component.name} '
                        f'({N_str} {component.species}) particle{plural} ...'
                    )
                    # Save particle attributes
                    component_h5.attrs['mass'] = correct_float(component.mass)
                    component_h5.attrs['N'] = N
                    # Get local indices of the particle data
                    start_local = int(np.sum(smart_mpi(N_local, mpifun='allgather')[:rank]))
                    end_local = start_local + component.N_local
                    # Save particle data
                    if save_all or component.snapshot_vars['save']['pos']:
                        pos_h5 = component_h5.create_dataset('pos', (N, 3), dtype=C2np['double'])
                        pos_h5[start_local:end_local, :] = component.pos_mv3[:N_local, :]
                    if save_all or component.snapshot_vars['save']['mom']:
                        mom_h5 = component_h5.create_dataset('mom', (N, 3), dtype=C2np['double'])
                        mom_h5[start_local:end_local, :] = component.mom_mv3[:N_local, :]
                    if component.use_ids:
                        # Store IDs as unsigned integers using as few
                        # bits as possible. We explicitly reinterpret
                        # the IDs as unsigned (still 64-bit) prior to
                        # writing to th efile. The convertion from
                        # unsigned 64-bit to unsigned {32, 16, 8}-bit
                        # appears to be handled by H5Py in a chunkified
                        # manner, so this operation is safe.
                        id_max = allreduce(max(component.ids_mv), op=MPI.MAX)
                        if id_max >= 2**32:
                            dtype = np.uint64
                        elif id_max >= 2**16:
                            dtype = np.uint32
                        elif id_max >= 2**8:
                            dtype = np.uint16
                        else:
                            dtype = np.uint8
                        ids_h5 = component_h5.create_dataset('ids', (N, ), dtype=dtype)
                        ids_mv_unsigned = asarray(component.ids_mv).view(np.uint64)
                        ids_h5[start_local:end_local] = ids_mv_unsigned[:N_local]
                elif component.representation == 'fluid':
                    # Write out progress message
                    masterprint(
                        f'Writing out {component.name} ({component.species} with '
                        f'gridsize {component.gridsize}, '
                        f'Boltzmann order {component.boltzmann_order}) fluid ...'
                    )
                    # Save fluid attributes
                    component_h5.attrs['gridsize'] = component.gridsize
                    component_h5.attrs['boltzmann_order'] = component.boltzmann_order
                    # Save fluid grids in groups of name
                    # "fluidvar_index", with index = 0, 1, ...
                    # The fluid scalars are then datasets within
                    # these groups, named "fluidscalar_multi_index",
                    # with multi_index (0, ), (1, ), ..., (0, 0), ...
                    shape = (component.gridsize, )*3
                    for index, fluidvar in enumerate(
                        component.fluidvars[:component.boltzmann_order + 1]
                    ):
                        if not save_all:
                            if index == 0 and not component.snapshot_vars['save']['ϱ']:
                                continue
                            if index == 1 and not component.snapshot_vars['save']['J']:
                                continue
                            if index == 2 and not (
                                   component.snapshot_vars['save']['𝒫']
                                or component.snapshot_vars['save']['ς']
                            ):
                                continue
                        fluidvar_h5 = component_h5.create_group(f'fluidvar_{index}')
                        multi_index_trace = ()
                        if 'trace' in fluidvar and fluidvar['trace'] is not None:
                            multi_index_trace = ('trace', )
                        for multi_index in fluidvar.multi_indices + multi_index_trace:
                            if not save_all and index == 2:
                                if multi_index == 'trace':
                                    if not component.snapshot_vars['save']['𝒫']:
                                        continue
                                else:
                                    if not component.snapshot_vars['save']['ς']:
                                        continue
                            fluidscalar = fluidvar[multi_index]
                            fluidscalar_h5 = fluidvar_h5.create_dataset(
                                f'fluidscalar_{multi_index}',
                                shape,
                                dtype=C2np['double'],
                            )
                            # The global fluid scalar grid is of course
                            # stored contiguously on disk. Generally
                            # though, a single process does not store a
                            # contiguous part of this global grid,
                            # as it is domain decomposed rather than
                            # slab decomposed. Here we communicate the
                            # fluid scalar to slabs before saving to
                            # disk, improving performance enormously.
                            slab = slab_decompose(fluidscalar.grid_mv)
                            slab_start = slab.shape[0]*rank
                            slab_end = slab_start + slab.shape[0]
                            fluidscalar_h5[
                                slab_start:slab_end,
                                :,
                                :,
                            ] = slab[:, :, :(slab.shape[2] - 2)]  # exclude padding
                    # Create additional names (hard links) for the fluid
                    # groups and data sets. The names from
                    # component.fluid_names will be used, except for
                    # the additional linear variable, if CLASS is used
                    # to close the Boltzmann hierarchy
                    # (hence the try/except).
                    for name, indices in component.fluid_names.items():
                        if not isinstance(name, str) or name == 'ordered':
                            continue
                        if isinstance(indices, (int, np.integer)):
                            # "name" is a fluid variable name (e.g. J,
                            # though not ϱ as this is a fluid scalar).
                            try:
                                fluidvar_h5 = component_h5[f'fluidvar_{indices}']
                                component_h5[name] = fluidvar_h5
                            except Exception:
                                pass
                        else:  # indices is a tuple
                            # "name" is a fluid scalar name (e.g. ϱ, Jx)
                            index, multi_index = indices
                            try:
                                fluidvar_h5 = component_h5[f'fluidvar_{index}']
                                fluidscalar_h5 = fluidvar_h5[f'fluidscalar_{multi_index}']
                                component_h5[name] = fluidscalar_h5
                            except Exception:
                                pass
                else:
                    abort(
                        f'Does not know how to save {component.name} '
                        f'with representation "{component.representation}"'
                    )
                # Done saving this component
                hdf5_file.flush()
                Barrier()
                masterprint('done')
        # Done saving the snapshot
        masterprint('done')
        # Return the filename of the saved file
        return filename

    # Method for loading in a CO𝘕CEPT snapshot from disk
    @cython.pheader(
        # Argument
        filename=str,
        only_params='bint',
        # Locals
        N='Py_ssize_t',
        N_local='Py_ssize_t',
        N_str=str,
        arr=object,  # np.ndarray
        boltzmann_order='Py_ssize_t',
        chunk_size='Py_ssize_t',
        component='Component',
        domain_size_i='Py_ssize_t',
        domain_size_j='Py_ssize_t',
        domain_size_k='Py_ssize_t',
        fluidscalar='FluidScalar',
        grid='double*',
        gridsize='Py_ssize_t',
        id_counter='Py_ssize_t',
        ids='Py_ssize_t*',
        index='Py_ssize_t',
        index_i='Py_ssize_t',
        index_i_file='Py_ssize_t',
        indexᵖ='Py_ssize_t',
        indexᵖ_file='Py_ssize_t',
        indexʳ='Py_ssize_t',
        mass='double',
        mom='double*',
        multi_index=tuple,
        name=str,
        plural=str,
        pos='double*',
        representation=str,
        size='Py_ssize_t',
        slab='double[:, :, ::1]',
        slab_start='Py_ssize_t',
        snapshot_unit_length='double',
        snapshot_unit_mass='double',
        snapshot_unit_time='double',
        species=str,
        start_local='Py_ssize_t',
        unit='double',
        unit_J='double',
        unit_ϱ='double',
        units_fluidvars='double[::1]',
    )
    def load(self, filename, only_params=False):
        if only_params:
            masterprint(f'Loading parameters of snapshot "{filename}" ...')
        else:
            masterprint(f'Loading snapshot "{filename}" ...')
        # Load components
        with open_hdf5(filename, mode='r', driver='mpio', comm=comm) as hdf5_file:
            # Load used base units
            self.units['time']   = hdf5_file.attrs['unit time']
            self.units['length'] = hdf5_file.attrs['unit length']
            self.units['mass']   = hdf5_file.attrs['unit mass']
            snapshot_unit_time   = eval_unit(self.units['time'])
            snapshot_unit_length = eval_unit(self.units['length'])
            snapshot_unit_mass   = eval_unit(self.units['mass'])
            # Load global attributes
            self.params['H0']      = hdf5_file.attrs['H0']*(1/snapshot_unit_time)
            self.params['a']       = hdf5_file.attrs['a']
            self.params['boxsize'] = hdf5_file.attrs['boxsize']*snapshot_unit_length
            self.params['Ωb']      = hdf5_file.attrs[unicode('Ωb')]
            self.params['Ωcdm']    = hdf5_file.attrs[unicode('Ωcdm')]
            # If a component should use IDs but none are stored in the
            # snapshot, IDs will be assigned based on the storage order.
            # In an effort to make the IDs unique even across
            # components, this counter will keep track of the largest ID
            # assigned to the previous component.
            id_counter = 0
            # Load component data
            for name, component_h5 in hdf5_file['components'].items():
                # Determine representation from the snapshot
                if 'N' in component_h5.attrs:
                    representation = 'particles'
                elif 'gridsize' in component_h5.attrs:
                    representation = 'fluid'
                else:
                    abort(
                        f'Could not determine representation of "{name}" '
                        f'in snapshot "{filename}", as neither N nor gridsize is specified'
                    )
                # Set the species based on the snapshot but overruled
                # by the select_species user parameter.
                species_h5 = component_h5.attrs['species']
                species = determine_species(name, representation, only_explicit=True)
                if not species:
                    species = species_h5
                elif species != species_h5:
                    masterwarn(
                        f'Interpreting the "{name}" component with specified '
                        f'species "{species_h5}" to instead be of species "{species}"'
                    )
                # Skip this component if it should not be loaded
                if not should_load(name, species, representation):
                    masterprint(f'Skipping {name}')
                    continue
                # Load the component
                if representation == 'particles':
                    # Construct a Component instance and append it
                    # to this snapshot's list of components.
                    N = component_h5.attrs['N']
                    mass = component_h5.attrs['mass']*snapshot_unit_mass
                    component = Component(name, species, N=N, mass=mass)
                    self.components.append(component)
                    # Done loading component attributes
                    if only_params:
                        continue
                    N_str = get_cubenum_strrep(N)
                    plural = ('s' if N > 1 else '')
                    masterprint(f'Reading in {name} ({N_str} {species}) particle{plural} ...')
                    # Extract HDF5 datasets
                    pos_h5 = None
                    if component.snapshot_vars['load']['pos']:
                        if 'pos' not in component_h5:
                            abort(f'No positions ("pos") found for component {component.name}')
                        pos_h5 = component_h5['pos']
                    mom_h5 = None
                    if component.snapshot_vars['load']['mom']:
                        if 'mom' not in component_h5:
                            abort(f'No momenta ("mom") found for component {component.name}')
                        mom_h5 = component_h5['mom']
                    ids_h5 = None
                    if component.use_ids and 'ids' in component_h5:
                        ids_h5 = component_h5['ids']
                    # Compute a fair distribution of
                    # particle data to the processes.
                    start_local, N_local = partition(N)
                    # Make sure that the particle data arrays
                    # have the correct size.
                    component.N_local = N_local
                    component.resize(N_local, only_loadable=True)
                    # Read particle data into the particle data arrays
                    dsets_arrs = []
                    if pos_h5 is not None:
                        dsets_arrs.append((pos_h5, asarray(component.pos_mv3)))
                    if mom_h5 is not None:
                        dsets_arrs.append((mom_h5, asarray(component.mom_mv3)))
                    if ids_h5 is not None:
                        # The particle IDs are stored as unsigned
                        # {64, 32, 16, 8}-bit ints in the snapshot,
                        # while they are stored as signed 64-bit ints in
                        # the code. Reinterpret the code array as
                        # unsigned 64-bit. The convertion from
                        # {32, 16, 8}-bit to 64-bit will be done on the
                        # fly by HDF5.
                        dsets_arrs.append((ids_h5, asarray(component.ids_mv).view(np.uint64)))
                    if N_local > 0:
                        for dset, arr in dsets_arrs:
                            # Load in using chunks
                            chunk_size = np.min((N_local, ℤ[self.chunk_size_max//8//3]))
                            for indexᵖ in range(0, N_local, chunk_size):
                                if indexᵖ + chunk_size > N_local:
                                    chunk_size = N_local - indexᵖ
                                indexᵖ_file = start_local + indexᵖ
                                source_sel = slice(indexᵖ_file, indexᵖ_file + chunk_size)
                                dest_sel   = slice(indexᵖ,      indexᵖ      + chunk_size)
                                if arr.ndim == 2:
                                    # Positions, momenta
                                    source_sel = (source_sel, slice(None))
                                    dest_sel   = (dest_sel,   slice(None))
                                dset.read_direct(arr, source_sel=source_sel, dest_sel=dest_sel)
                        # If the snapshot and the current run uses
                        # different systems of units, multiply the
                        # positions and momenta by the snapshot units.
                        pos = component.pos
                        mom = component.mom
                        if snapshot_unit_length != 1:
                            for indexʳ in range(3*N_local):
                                pos[indexʳ] *= snapshot_unit_length
                        unit = snapshot_unit_length/snapshot_unit_time*snapshot_unit_mass
                        if unit != 1:
                            for indexʳ in range(3*N_local):
                                mom[indexʳ] *= unit
                        # If this component should make use of particle
                        # IDs but none are stored in the snapshot,
                        # assign IDs according to the order in which
                        # they are stored in the file.
                        if component.use_ids and ids_h5 is None:
                            masterprint('Assigning particle IDs ...')
                            ids = component.ids
                            for indexᵖ in range(N_local):
                                ids[indexᵖ] = ℤ[id_counter + start_local] + indexᵖ
                            masterprint('done')
                    # Update particle ID counter
                    id_counter += N
                    # Done reading in particle component
                    masterprint('done')
                elif representation == 'fluid':
                    # Read in fluid attributes
                    gridsize = component_h5.attrs['gridsize']
                    boltzmann_order = component_h5.attrs['boltzmann_order']
                    # Construct a Component instance and append it
                    # to this snapshot's list of components.
                    component = Component(name, species,
                        gridsize=gridsize, boltzmann_order=boltzmann_order)
                    self.components.append(component)
                    # Done loading component attributes
                    if only_params:
                        continue
                    # Write out progress message
                    masterprint(
                        f'Reading in {name} ({species} with gridsize {gridsize}, '
                        f'Boltzmann order {boltzmann_order}) fluid ...'
                    )
                    # Compute local indices of fluid grids
                    domain_size_i = gridsize//domain_subdivisions[0]
                    domain_size_j = gridsize//domain_subdivisions[1]
                    domain_size_k = gridsize//domain_subdivisions[2]
                    if master and (   gridsize != domain_subdivisions[0]*domain_size_i
                                   or gridsize != domain_subdivisions[1]*domain_size_j
                                   or gridsize != domain_subdivisions[2]*domain_size_k):
                        abort(
                            f'The gridsize of the {name} component is {gridsize} '
                            f'which cannot be equally shared among {nprocs} processes'
                        )
                    # Make sure that the fluid grids
                    # have the correct size.
                    component.resize(
                        (domain_size_i, domain_size_j, domain_size_k),
                        only_loadable=True,
                    )
                    # Fluid scalars are already instantiated.
                    # Now populate them.
                    for index, fluidvar in enumerate(
                        component.fluidvars[:component.boltzmann_order + 1]
                    ):
                        if index == 0:
                            if component.snapshot_vars['load']['ϱ']:
                                if f'fluidvar_{index}' not in component_h5:
                                    abort(
                                        f'No energy density ("ϱ") found '
                                        f'for component {component.name}'
                                    )
                            else:
                                continue
                        elif index == 1:
                            if component.snapshot_vars['load']['J']:
                                if f'fluidvar_{index}' not in component_h5:
                                    abort(
                                        f'No energy density ("J") found '
                                        f'for component {component.name}'
                                    )
                            else:
                                continue
                        fluidvar_h5 = component_h5[f'fluidvar_{index}']
                        for multi_index in fluidvar.multi_indices:
                            fluidscalar_h5 = fluidvar_h5[f'fluidscalar_{multi_index}']
                            slab = get_fftw_slab(gridsize)
                            slab_start = ℤ[slab.shape[0]]*rank
                            # Load in using chunks. Large chunks are
                            # fine as no temporary buffer is used. The
                            # maximum possible chunk size is limited
                            # by MPI, though.
                            chunk_size = np.min((
                                ℤ[slab.shape[0]],
                                ℤ[self.chunk_size_max//8//gridsize**2],
                            ))
                            if chunk_size == 0:
                                masterwarn(
                                    'The input seems surprisingly large '
                                    'and may not be read in correctly'
                                )
                                chunk_size = 1
                            arr = asarray(slab)
                            for index_i in range(0, ℤ[slab.shape[0]], chunk_size):
                                if index_i + chunk_size > ℤ[slab.shape[0]]:
                                    chunk_size = ℤ[slab.shape[0]] - index_i
                                index_i_file = slab_start + index_i
                                source_sel = (
                                    slice(index_i_file, index_i_file + chunk_size),
                                    slice(None),
                                    slice(None),
                                )
                                dest_sel = (
                                    slice(index_i, index_i + chunk_size),
                                    slice(None),
                                    slice(0, slab.shape[2] - 2),  # exclude padding
                                )
                                fluidscalar_h5.read_direct(
                                    arr, source_sel=source_sel, dest_sel=dest_sel,
                                )
                            # Communicate the slabs directly to the
                            # domain decomposed fluid grids.
                            domain_decompose(
                                slab,
                                component.fluidvars[index][multi_index].grid_mv,
                                do_ghost_communication=True,
                            )
                    # If the snapshot and the current run uses different
                    # systems of units, multiply the fluid data
                    # by the snapshot units.
                    unit_ϱ = snapshot_unit_mass/snapshot_unit_length**3
                    unit_J = snapshot_unit_mass/(snapshot_unit_length**2*snapshot_unit_time)
                    units_fluidvars = asarray((unit_ϱ, unit_J), dtype=C2np['double'])
                    size = np.prod(component.shape)
                    for fluidvar, unit in zip(component.fluidvars[:component.boltzmann_order + 1],
                                              units_fluidvars):
                        if unit == 1:
                            continue
                        for fluidscalar in fluidvar:
                            grid = fluidscalar.grid
                            for index in range(size):
                                grid[index] *= unit
                    # Done reading in fluid component
                    masterprint('done')
                elif master:
                    abort(
                        f'Does not know how to load {name} with representation "{representation}"'
                    )
        # Done loading the snapshot
        masterprint('done')

    # This method populate the snapshot with component data
    # and additional parameters.
    def populate(self, components, params=None):
        if not components:
            abort(f'Cannot save a {self.name} snapshot with no components')
        if params is None:
            params = {}
        # Populated snapshot with the components
        self.components = components
        # Populate snapshot with the passed scale factor
        # and global parameters. If a params dict is passed,
        # use values from this instead.
        self.params['H0'] = params.get('H0', H0)
        if enable_Hubble:
            self.params['a'] = params.get('a', universals.a)
        else:
            self.params['a'] = universals.a
        self.params['boxsize'] = params.get('boxsize', boxsize)
        self.params['Ωb'     ] = params.get('Ωb'     , Ωb)
        self.params['Ωcdm'   ] = params.get('Ωcdm'   , Ωcdm)
        # Populate the base units with the global base units
        self.units['time'  ] = unit_time
        self.units['length'] = unit_length
        self.units['mass'  ] = unit_mass

# Class storing a GADGET snapshot. Besides holding methods for
# saving/loading, it stores particle data and the GADGET header.
@cython.cclass
class GadgetSnapshot:
    """This class represents snapshots of the "GADGET" type,
    which may be either SnapFormat = 1 or SnapFormat = 2
    native to GADGET.
    As is the case for the CO𝘕CEPT snapshot class, this class contains
    a list components (the components attribute) and dict of parameters
    (the params attribute).
    """
    # The properly written name of this snapshot type
    # (only used for printing).
    name = 'GADGET'
    # The filename extension for this type of snapshot
    extension = ''
    # Maximum allowed chunk size in bytes
    chunk_size_max = 2**23  # 8 MB
    # Names of components contained in snapshots, in order
    component_names = [
        f'GADGET {particle_type}'
        for particle_type in ['gas', 'halo', 'disk', 'bulge', 'stars', 'bndry']
    ]
    num_particle_types = len(component_names)
    # Ordered fields in the GADGET header,
    # mapped to their type and default value.
    #   † Unsigned in GADGET-2 user guide but not in source code.
    #     We go with unsigned. We never have Npart large enough for
    #     overflow of signed int when saving, but you never know when
    #     loading (except that negative Npart is nonsensical).
    #   ‡ Signed in GADGET-2 user guide but not in source code.
    GadgetHeaderField = collections.namedtuple(
        'GadgetHeaderField',
        ['fmt', 'default'],
        defaults=['', 0],
    )
    header_fields = {
        'Npart'        : GadgetHeaderField('I'),  # †
        'Massarr'      : GadgetHeaderField('d'),
        'Time'         : GadgetHeaderField('d'),
        'Redshift'     : GadgetHeaderField('d'),
        'FlagSfr'      : GadgetHeaderField('i'),
        'FlagFeedback' : GadgetHeaderField('i'),
        'Nall'         : GadgetHeaderField('I'),  # ‡
        'FlagCooling'  : GadgetHeaderField('i'),
        'NumFiles'     : GadgetHeaderField('i'),
        'BoxSize'      : GadgetHeaderField('d'),
        'Omega0'       : GadgetHeaderField('d'),
        'OmegaLambda'  : GadgetHeaderField('d'),
        'HubbleParam'  : GadgetHeaderField('d'),
        'FlagAge'      : GadgetHeaderField('i'),
        'FlagMetals'   : GadgetHeaderField('i'),
        'NallHW'       : GadgetHeaderField('I'),  # ‡
        'flag_entr_ics': GadgetHeaderField('i'),
    }
    # Ensure floating-point defaults where appropriate
    for key, val in header_fields.items():
        if val.fmt in {'f', 'd'}:
            header_fields[key] = val._replace(default=float(val.default))
    # Some fields have a value for each particle type
    for key in ['Npart', 'Massarr', 'Nall', 'NallHW']:
        val = header_fields[key]
        header_fields[key] = val._replace(
            fmt=f'{num_particle_types}{val.fmt}',
            default=[val.default]*num_particle_types,
        )
    # Block name format, head block name and size
    block_name_fmt = '4s'
    block_name_header = 'HEAD'
    headersize = 2**8
    # The maximum number of particles within a single GADGET
    # snapshot file is limited by the largest number representable
    # by an int. As we never deal with negative particle numbers, we use
    # unsigned ints in this implementation. In e.g. the GADGET-2 code
    # however, a signed int is used, and so we subtract the sign bit in
    # the calculation below, cutting the maximum number of particles per
    # snapshot roughly in half, compared to what is really needed.
    num_particles_file_max = (
        ((2**(sizesC['i']*8 - 1) - 1) - 2*sizesC['I'])
        //(
            3*(
                np.max((
                    gadget_snapshot_params['dataformat']['POS'],
                    gadget_snapshot_params['dataformat']['VEL'],
                ))//8
            )
        )
    )
    if gadget_snapshot_params['particles per file'] > 0:
        if gadget_snapshot_params['particles per file'] > num_particles_file_max:
            masterwarn(
                f'The number of particles to write to each GADGET snapshot file, '
                f'{gadget_snapshot_params["particles per file"]}, is larger than '
                f'the recommended maximum of {num_particles_file_max}.'
            )
        num_particles_file_max = gadget_snapshot_params['particles per file']

    # Class method for identifying a file to be a snapshot of this type
    @classmethod
    def is_this_type(cls, filename):
        # Construct list of possible file names
        # for the first snapshot file.
        if os.path.isdir(filename):
            filenames = glob(f'{filename}/*.0')
            if len(filenames) != 1:
                return False
        else:
            filename_stripped = filename.removesuffix('.0').rstrip('.*')
            filenames = [filename, f'{filename}.0', f'{filename_stripped}.0', filename_stripped]
        # Test for GADGET format, either SnapFormat 1 or SnapFormat 2
        for filename in filenames:
            snapformat = cls.get_snapformat(filename)
            if snapformat != -1:
                return True
        return False

    # Class method for identifying the GADGET SnapFormat of a file
    @classmethod
    def get_snapformat(cls, filename):
        """Identifies whether a file is a GADGET snapshot of SnapFormat
        1 or SnapFormat 2. If neither, -1 is returned.
        """
        if not os.path.isfile(filename):
            return -1
        # Test for SnapFormat 2 by checking for the existence
        # of the 'HEAD' identifier.
        try:
            with open_file(filename, mode='rb') as f:
                f.seek(sizesC['I'])
                if cls.block_name_header == struct.unpack(
                    cls.block_name_fmt,
                    f.read(struct.calcsize(cls.block_name_fmt)),
                )[0].decode('utf8').rstrip():
                    return 2
        except Exception:
            pass
        # Test for SnapFormat 1 by checking the size
        # of the header block.
        try:
            with open_file(filename, mode='rb') as f:
                if cls.headersize == struct.unpack('I', f.read(sizesC['I']))[0]:
                    return 1
        except Exception:
            pass
        return -1

    # Static method for distributing particles from processes to
    # files or from files to processes.
    @staticmethod
    def distribute(num_particles_files, num_local):
        """The num_particles_files is a list of lists, one for each
        file. The sublists contain the number of particles in the file
        for each type, with non-existing components excluded.
        The num_local is a list specifying the number of particles local
        to the process, for each particle type. The return value is a
        list in the same format as num_particles_files, but with values
        being the number of particles to write/read
        for the local process.
        """
        num_particles_files = deepcopy(num_particles_files)
        num_files = len(num_particles_files)
        num_components = len(num_local)
        # Inform all processes about the local particle content
        # of all other processes.
        num_locals = allgather(num_local)
        # Determine number of particles of each type
        # to write/read to/from each file.
        num_io_files = [[0]*num_components for i in range(num_files)]
        for num_particle_file, num_io_file in zip(num_particles_files, num_io_files):
            for j, num_particle in enumerate(num_particle_file):
                for rank_io in range(nprocs):
                    num_io = num_locals[rank_io][j]
                    if num_io == 0:
                        continue
                    elif num_io > num_particle:
                        num_io = num_particle
                    num_particle -= num_io
                    num_particle_file[j] = num_particle
                    num_locals[rank_io][j] -= num_io
                    if rank_io == rank:
                        num_io_file[j] = num_io
                    if num_particle_file[j] == 0:
                        break
        return num_io_files

    # Initialisation method
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the GadgetSnapshot type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public dict params
        public list components
        public int snapformat
        public dict header
        public double _unit_length
        public double _unit_velocity
        public double _unit_mass
        object block_names
        Component misnamed_halo_component
        """
        # Dict containing all the parameters of the snapshot
        self.params = {}
        # List of Component instances
        self.components = []
        # GADGET SnapFormat
        self.snapformat = -1
        # Header corresponding to the HEAD block
        self.header = {}
        # Unit system employed by this snapshot relative to the system
        # of units currently in use by CO𝘕CEPT. These will be set by the
        # corresponingly named methods.
        self._unit_length = -1
        self._unit_velocity = -1
        self._unit_mass = -1
        # Iterator yielding block names to load in case of SnapFormat 1
        self.block_names = iter(())
        # Rogue component used as the GADGET halo component
        # though it is not named accordingly.
        self.misnamed_halo_component = None
        # Check on low level type sizes
        sizes_expected = {'s': 1, 'i': 4, 'I': 4, 'Q': 8, 'f': 4, 'd': 8}
        for fmt, size_expected in sizes_expected.items():
            size = sizesC[fmt]
            if size_expected != size:
                masterwarn(
                    f'Expected C type \'{fmt2C[fmt]}\' to have a size '
                    f'of {size_expected} bytes, but its size is {size}'
                )

    # Property for the dimensionless Hubble parameter
    @property
    def h(self):
        value = self.header.get('HubbleParam', 0)
        if value == 0:
            if 'HubbleParam' in self.header and H0 > 0:
                value = correct_float(H0/(100*units.km/(units.s*units.Mpc)))
                self.header['HubbleParam'] = value
                masterwarn(
                    f'Attempted to access h on a {self.name} snapshot '
                    f'with a set value of h = 0. Will reassign h = {value}.'
                )
            else:
                abort(
                    f'Attempted to access h on a {self.name} snapshot '
                    f'prior to setting HubbleParam in the header'
                )
        return value

    # Properties for the numerical GADGET values of units
    # with respect to the unit system currently in use by CO𝘕CEPT.
    @property
    def unit_length(self):
        if self._unit_length == -1:
            self._unit_length = eval_unit(
                gadget_snapshot_params['units']['length'],
                units_dict | {'h': self.h},
            )
        return self._unit_length
    @property
    def unit_velocity(self):
        if self._unit_velocity == -1:
            self._unit_velocity = eval_unit(
                gadget_snapshot_params['units']['velocity'],
                units_dict | {'h': self.h},
            )
        return self._unit_velocity
    @property
    def unit_mass(self):
        if self._unit_mass == -1:
            self._unit_mass = eval_unit(
                gadget_snapshot_params['units']['mass'],
                units_dict | {'h': self.h},
            )
        return self._unit_mass

    # Method for saving a GADGET snapshot to disk
    @cython.pheader(
        # Arguments
        filename=str,
        save_all='bint',
        # Locals
        bits=object,  # Python int
        block=dict,
        block_fmt=str,
        block_index='Py_ssize_t',
        block_name=str,
        block_type=str,
        blocks=dict,
        boxsize_gadget_doubleprec='double',
        boxsize_gadget_singleprec='float',
        chunk_doubleprec='double[::1]',
        chunk_singleprec='float[::1]',
        chunk_size_max_needed='Py_ssize_t',
        component='Component',
        component_index='Py_ssize_t',
        fake_id_max=object,  # Python int
        file_index='Py_ssize_t',
        filename_existing=str,
        filename_existing_prefix=str,
        finalize_block='bint',
        id_max='Py_ssize_t',
        ids_mv='Py_ssize_t[::1]',
        indent='int',
        index='Py_ssize_t',
        index_left='Py_ssize_t',
        index_prev='Py_ssize_t',
        index_rght='Py_ssize_t',
        indexᵖ_bgn='Py_ssize_t',
        indexᵖ_end='Py_ssize_t',
        indices_components='Py_ssize_t[::1]',
        initialize_block='bint',
        initialize_file='bint',
        num_files='Py_ssize_t',
        num_nonlocal_prior='Py_ssize_t[::1]',
        num_write='Py_ssize_t',
        num_write_file=list,
        num_write_files=list,
        num_write_files_tot='Py_ssize_t[:, ::1]',
        num_write_max='Py_ssize_t',
        num_writeoute_jobs='Py_ssize_t',
        parallel_write='bint',
        rank_next='int',
        rank_prev='int',
        request=object,  # mpi4py.MPI.Request
        requests=list,
        save_pos='bint',
        save_vel='bint',
        writeout_job=object,  # WriteoutJob
        writeout_jobid=tuple,
        writeout_jobid_next=tuple,
        writeout_jobid_prev=tuple,
        writeout_jobids=list,
        writeout_jobids_completed=set,
        writeout_jobs=dict,
        returns=str,
    )
    def save(self, filename, save_all=False):
        # Set the GADGET SnapFormat based on user parameters
        self.snapformat = gadget_snapshot_params['snapformat']
        # Divvy up the particles between the files and processes
        num_write_files = self.divvy()
        num_write_files_tot = asarray(
            [
                allreduce(
                    asarray(num_write_file, dtype=C2np['Py_ssize_t']),
                    op=MPI.SUM,
                )
                for num_write_file in num_write_files
            ],
            dtype=C2np['Py_ssize_t'],
        )
        num_files = len(num_write_files)
        # If the snapshot is to be saved over several files,
        # create a directory for storing these.
        if master:
            if num_files == 1:
                if os.path.isdir(filename):
                    abort(
                        f'Refuses to replace directory "{filename}" with snapshot. '
                        f'Remove this directory or select different snapshot '
                        f'output directory or base name.'
                    )
            else:
                if os.path.isfile(filename):
                    os.remove(filename)
                os.makedirs(filename, exist_ok=True)
                filename_existing_prefix = f'{filename}/{output_bases["snapshot"]}.'
                for filename_existing in glob(f'{filename_existing_prefix}*'):
                    if re.fullmatch(r'\d+', filename_existing[len(filename_existing_prefix):]):
                        os.remove(filename_existing)
        # Progress messages
        msg = filename
        if num_files > 1:
            msg += f'/{output_bases["snapshot"]}.*'
        masterprint(f'Saving {self.name} snapshot "{msg}" ...')
        msg_list = []
        for component in self.components:
            N = component.N
            N_str = get_cubenum_strrep(N)
            plural = ('s' if N > 1 else '')
            msg_list.append(f'{component.name} ({N_str} {component.species}) particle{plural}')
        msg = ', '.join(msg_list)
        masterprint(f'Writing out {msg} ...')
        if num_files > 1:
            # Ensure correct printout
            masterprint(ensure_newline_after_ellipsis=False)
            Barrier()
        # Determine whether to save positions and/or velocities
        save_pos = save_vel = True
        if not save_all:
            save_pos = any([
                component.snapshot_vars['save']['pos']
                for component in self.components
            ])
            save_vel = any([
                component.snapshot_vars['save']['mom']
                for component in self.components
            ])
            if save_pos and any([
                not component.snapshot_vars['save']['pos']
                for component in self.components
            ]):
                masterwarn(
                    f'It is specified that particle positions of some component(s) '
                    f'should not be stored in snapshots, while particle positions '
                    f'of other component(s) should. For {self.name} snapshots all '
                    f'components must be treated the same. All components will be '
                    f'saved with particle positions.'
                )
            if save_vel and any([
                not component.snapshot_vars['save']['mom']
                for component in self.components
            ]):
                masterwarn(
                    f'It is specified that particle momenta/velocities of some '
                    f'component(s) should not be stored in snapshots, while particle '
                    f'velocities of other component(s) should. For {self.name} snapshots '
                    f'all components must be treated the same. All components will be '
                    f'saved with particle velocities.'
                )
        # Get information about the blocks to be written out
        blocks = self.get_blocks_info('save')
        if not save_pos:
            blocks.pop('POS', None)
        if not save_vel:
            blocks.pop('VEL', None)
        # Warn about storing particle IDs using
        # an integer type that is too small.
        block = blocks.get('ID', {})
        if block:
            block_type = block['type']
            block_fmt = block_type[len(block_type) - 1]
            bits = int(8*C2np[fmt2C[block_fmt]]().itemsize)
            if component.use_ids:
                for component, ids_mv in zip(self.components, block.get('data')):
                    id_max = allreduce(max(ids_mv[:component.N_local]), op=MPI.MAX)
                    if id_max >= 2**bits:
                        masterwarn(
                            f'Component "{component.name}" contains particle IDs '
                            f'larger than what can be stored using '
                            f'{bits}-bit unsigned integers'
                        )
            else:
                fake_id_max = -1
                for component in self.components:
                    fake_id_max += int(component.N)
                    if fake_id_max >= 2**bits:
                        masterwarn(
                            f'Component "{component.name}" will be assigned particle IDs '
                            f'larger than what can be stored using '
                            f'{bits}-bit unsigned integers'
                        )
        # If fake IDs need to be created for any of the components,
        # we need to know the total number of particles on the lower
        # ranked processes, for each component.
        if all([component.use_ids for component in self.components]):
            num_nonlocal_prior = zeros(len(self.components), dtype=C2np['Py_ssize_t'])
        else:
            num_nonlocal_prior = np.sum(
                asarray(
                    allgather([component.N_local for component in self.components]),
                    dtype=C2np['Py_ssize_t'],
                )[:rank],
                axis=0,
            )
        # Instantiate chunk buffers for particle data
        num_write_max = 0
        for num_write in itertools.chain(*num_write_files):
            if num_write > num_write_max:
                num_write_max = num_write
        chunk_size_max_needed = np.min((3*num_write_max, ℤ[self.chunk_size_max//8]))
        chunk_singleprec = None
        chunk_doubleprec = None
        if any([block['type'].endswith('f') for block in blocks.values()]):
            chunk_singleprec = empty(chunk_size_max_needed, dtype=C2np['float'])
        if any([block['type'].endswith('d') for block in blocks.values()]):
            chunk_doubleprec = empty(chunk_size_max_needed, dtype=C2np['double'])
        # The boxsize in GADGET units, used for safeguarding against
        # out-of-bounds particles after converting to GADGET units.
        boxsize_gadget_singleprec = C2np['float' ](boxsize/blocks.get('POS', 1)['unit'][0])
        boxsize_gadget_doubleprec = C2np['double'](boxsize/blocks.get('POS', 1)['unit'][0])
        # To carry out the snapshot writing in parallel, we formulate
        # the task as many smaller "writeout" jobs. Each such job has an
        # ID of the form
        #   (file_index, block_index, component_index, rank)
        # For a given file_index, all jobs with this file index must be
        # carried out in order (according to the lexicographical order
        # of the job ID), across all processes. The jobs themselves are
        # further characterized by the local indices of the particles to
        # be written. Furthermore, each job knows the ID of the preceding
        # and following job, as well as whether it is up to this
        # specific job to also initialize the file, initialize
        # the block, finalize the block.
        WriteoutJob = collections.namedtuple(
            'WriteoutJob',
            (
                'initialize_file', 'initialize_block', 'finalize_block',
                'jobid_prev', 'jobid_next',
                'file_index', 'block_name', 'component_index', 'indices',
            ),
        )
        writeout_jobs = {}
        indices_components = zeros(len(self.components), dtype=C2np['Py_ssize_t'])
        for file_index, num_write_file in enumerate(num_write_files):
            for component_index, num_write in enumerate(num_write_file):
                # Share information about the number of particles
                # to write between all processes.
                num_write_procs = asarray(allgather(num_write), dtype=C2np['Py_ssize_t'])
                if num_write == 0:
                    continue
                # Compute local particle indices
                indexᵖ_bgn = indices_components[component_index]
                indexᵖ_end = indexᵖ_bgn + num_write
                indices_components[component_index] = indexᵖ_end
                # Add (incomplete) writeout job IDs
                for block_index, block_name in enumerate(blocks):
                    writeout_jobid = (file_index, block_index, component_index, rank)
                    writeout_jobs[writeout_jobid] = WriteoutJob(
                        *[None]*5,
                        file_index, block_name, component_index, (indexᵖ_bgn, indexᵖ_end),
                    )
        # Let all processes know about all writeout job IDs
        writeout_jobids = sorted(itertools.chain(*allgather(list(writeout_jobs))))
        num_writeoute_jobs = len(writeout_jobids)
        # Find neighbour writeout job IDs and update the missing
        # fields accordingly. If we are not writing in parallel,
        # let each writeout job depend on the previous one.
        parallel_write = gadget_snapshot_params['parallel write']
        for writeout_jobid, writeout_job in writeout_jobs.items():
            index_left = 0
            index_rght = num_writeoute_jobs - 1
            index = -1
            index_prev = -1
            while True:
                index = (index_left + index_rght)//2
                if index == index_prev:
                    break
                index_prev = index
                if writeout_jobids[index] < writeout_jobid:
                    index_left = index
                elif writeout_jobids[index] > writeout_jobid:
                    index_rght = index
                else:
                    break
            if index < num_writeoute_jobs - 1 and writeout_jobids[index + 1] == writeout_jobid:
                index += 1
            writeout_jobid_prev = writeout_jobid_next = None
            initialize_file = initialize_block = finalize_block = True
            if index > 0:
                writeout_jobid_prev = writeout_jobids[index - 1]
                if writeout_jobid_prev[0] == writeout_jobid[0]:
                    # Same file as previous
                    initialize_file = False
                    if writeout_jobid_prev[1] == writeout_jobid[1]:
                        # Same block as previous
                        initialize_block = False
                elif parallel_write:
                    # Different file from previous
                    writeout_jobid_prev = None
            if index < num_writeoute_jobs - 1:
                writeout_jobid_next = writeout_jobids[index + 1]
                if writeout_jobid_next[0] == writeout_jobid[0]:
                    # Same file as next
                    if writeout_jobid_next[1] == writeout_jobid[1]:
                        # Same block as next
                        finalize_block = False
                elif parallel_write:
                    # Different file from next
                    writeout_jobid_next = None
            writeout_jobs[writeout_jobid] = writeout_job._replace(
                initialize_file=initialize_file,
                initialize_block=initialize_block,
                finalize_block=finalize_block,
                jobid_prev=writeout_jobid_prev,
                jobid_next=writeout_jobid_next,
            )
        writeout_jobids.clear()
        # Carry out each of the jobs as they become available
        indent = bcast(progressprint['indentation'])
        writeout_jobids_completed = set()
        requests = []
        while writeout_jobs:
            for writeout_jobid, writeout_job in writeout_jobs.copy().items():
                writeout_jobid_prev = writeout_job.jobid_prev
                if writeout_jobid_prev is not None:
                    rank_prev = writeout_jobid_prev[3]
                    if iprobe(source=rank_prev):
                        writeout_jobids_completed.add(recv(source=rank_prev))
                    if writeout_jobid_prev not in writeout_jobids_completed:
                        continue
                # Job ready to be carried out
                self.execute_writeout_job(
                    filename, num_write_files_tot, num_nonlocal_prior, blocks, writeout_job,
                    chunk_singleprec, chunk_doubleprec,
                    boxsize_gadget_singleprec, boxsize_gadget_doubleprec,
                    indent,
                )
                # Inform of the availability of the next job
                writeout_jobid_next = writeout_job.jobid_next
                if writeout_jobid_next is not None:
                    rank_next = writeout_jobid_next[3]
                    requests.append(isend(writeout_jobid, dest=rank_next))
                # Running cleanup
                writeout_jobids_completed.discard(writeout_jobid_prev)
                writeout_jobs.pop(writeout_jobid)
                # Start over rather than continuing on,
                # prioritising early jobs.
                break
        # For good measure, ensure that all messages have been received
        # and that all processes are synchronized.
        for request in requests:
            request.wait()
        Barrier()
        # Finalise progress messages
        masterprint('done')
        masterprint('done')
        # Return the filename of the saved snapshot. In case of multiple
        # snapshot files having been written for this single snapshot,
        # this will be a directory.
        return filename

    # Method for carrying out a single "writeout" job
    @cython.header(
        # Arguments
        filename=str,
        num_write_files_tot='Py_ssize_t[:, ::1]',
        num_nonlocal_prior='Py_ssize_t[::1]',
        blocks=dict,
        writeout_job=object,  # WriteoutJob
        chunk_singleprec='float[::1]',
        chunk_doubleprec='double[::1]',
        boxsize_gadget_singleprec='float',
        boxsize_gadget_doubleprec='double',
        indent='int',
        # Locals
        block=dict,
        block_fmt=str,
        block_name=str,
        block_size='Py_ssize_t',
        block_type=str,
        chunk='double[::1]',
        chunk_doubleprec_ptr='double*',
        chunk_ptr='double*',
        chunk_singleprec_ptr='float*',
        chunk_size='Py_ssize_t',
        component='Component',
        component_index='Py_ssize_t',
        component_prior='Component',
        data=object,  # np.ndarray
        data_value_doubleprec='double',
        data_value_singleprec='float',
        dtype=object,
        file_index='Py_ssize_t',
        id_base='Py_ssize_t',
        id_bgn='Py_ssize_t',
        id_end='Py_ssize_t',
        ids_chunk=object,  # np.ndarray
        index_chunk='Py_ssize_t',
        indexᵖ='Py_ssize_t',
        indexᵖ_bgn='Py_ssize_t',
        indexᵖ_end='Py_ssize_t',
        indexʳ='Py_ssize_t',
        indexʳ_bgn='Py_ssize_t',
        indexʳ_end='Py_ssize_t',
        num_files='Py_ssize_t',
        num_write='Py_ssize_t',
        size_write='Py_ssize_t',
        unit='double',
        returns='void',
    )
    def execute_writeout_job(
        self, filename, num_write_files_tot, num_nonlocal_prior, blocks, writeout_job,
        chunk_singleprec, chunk_doubleprec,
        boxsize_gadget_singleprec, boxsize_gadget_doubleprec,
        indent=0,
    ):
        file_index = writeout_job.file_index
        block_name = writeout_job.block_name
        component_index = writeout_job.component_index
        indexᵖ_bgn, indexᵖ_end = writeout_job.indices
        num_files = num_write_files_tot.shape[0]
        if num_files > 1:
            filename = f'{filename}/{output_bases["snapshot"]}.{file_index}'
        component = self.components[component_index]
        # Initialise the file with the HEAD block
        if writeout_job.initialize_file:
            if num_files > 1:
                indentation = ' '*indent
                fancyprint(
                    f'{indentation}Writing snapshot file {file_index}/{num_files - 1}',
                    indent=-1,
                    ensure_newline_after_ellipsis=False,
                )
            self.write_header(filename, num_write_files_tot[file_index])
        # Extract block specifics
        block = blocks[block_name]
        data = block.get('data'                            )[component_index]
        unit = block.get('unit', [1.0]*len(self.components))[component_index]
        block_type = block['type']
        block_fmt = block_type[len(block_type) - 1]
        dtype = C2np[fmt2C[block_fmt]]
        block_size = np.sum(num_write_files_tot[file_index])*struct.calcsize(block_type)
        # The first fake particle ID to assign
        # if this component does not make use of IDs.
        id_base = 0
        if not component.use_ids:
            for component_prior in self.components[:component_index]:
                id_base += component_prior.N
            id_base += num_nonlocal_prior[component_index]
        # Begin block
        if writeout_job.initialize_block:
            self.write_block_bgn(filename, block_size, block_name)
        # Write out the block contents in chunks
        chunk_singleprec_ptr = NULL
        chunk_doubleprec_ptr = NULL
        if chunk_singleprec is not None:
            chunk_singleprec_ptr = cython.address(chunk_singleprec[:])
        if chunk_doubleprec is not None:
            chunk_doubleprec_ptr = cython.address(chunk_doubleprec[:])
        num_write = indexᵖ_end - indexᵖ_bgn
        if block_name in {'POS', 'VEL'}:
            indexʳ_bgn = 3*indexᵖ_bgn
            indexʳ_end = 3*indexᵖ_end
            size_write = 3*num_write
            chunk_size = np.min((size_write, ℤ[self.chunk_size_max//8]))
            with open_file(filename, mode='ab') as f:
                indexʳ = indexʳ_bgn
                while indexʳ != indexʳ_end:
                    if indexʳ + chunk_size > indexʳ_end:
                        chunk_size = indexʳ_end - indexʳ
                    chunk = data[indexʳ:(indexʳ + chunk_size)]
                    chunk_ptr = cython.address(chunk[:])
                    indexʳ += chunk_size
                    # Copy chunk while applying unit conversion,
                    # then write this copy to the file. For positions,
                    # safeguard against round-off errors.
                    if 𝔹[block_fmt == 'f']:
                        for index_chunk in range(chunk_size):
                            data_value_singleprec = chunk_ptr[index_chunk]*ℝ[1/unit]
                            with unswitch(1):
                                if 𝔹[block_name == 'POS']:
                                    if data_value_singleprec >= boxsize_gadget_singleprec:
                                        data_value_singleprec -= boxsize_gadget_singleprec
                            chunk_singleprec_ptr[index_chunk] = data_value_singleprec
                        asarray(chunk_singleprec[:chunk_size]).tofile(f)
                    elif 𝔹[block_fmt == 'd']:
                        for index_chunk in range(chunk_size):
                            data_value_doubleprec = chunk_ptr[index_chunk]*ℝ[1/unit]
                            with unswitch(1):
                                if 𝔹[block_name == 'POS']:
                                    if data_value_doubleprec >= boxsize_gadget_doubleprec:
                                        data_value_doubleprec -= boxsize_gadget_doubleprec
                            chunk_doubleprec_ptr[index_chunk] = data_value_doubleprec
                        asarray(chunk_doubleprec[:chunk_size]).tofile(f)
                    else:
                        abort(
                            f'Block format "{block_fmt}" not implemented '
                            f'for block "{block_name}"'
                        )
        elif block_name == 'ID':
            chunk_size = np.min((num_write, ℤ[self.chunk_size_max//8]))
            with open_file(filename, mode='ab') as f:
                indexᵖ = indexᵖ_bgn
                while indexᵖ != indexᵖ_end:
                    if indexᵖ + chunk_size > indexᵖ_end:
                        chunk_size = indexᵖ_end - indexᵖ
                    if component.use_ids:
                        ids_chunk = data[indexᵖ:(indexᵖ + chunk_size)]
                    else:
                        # This component does not have particle IDs.
                        # Generate some on the fly.
                        id_bgn = id_base + indexᵖ
                        id_end = id_bgn + chunk_size
                        ids_chunk = arange(id_bgn, id_end, dtype=dtype)
                    indexᵖ += chunk_size
                    asarray(ids_chunk, dtype=dtype).tofile(f)
        else:
            abort(f'Does not know how to write {self.name} block "{block_name}"')
        # End block
        if writeout_job.finalize_block:
            self.write_block_end(filename, block_size)

    # Method for divvying up the particles of each processes
    # between the files to be written.
    def divvy(self, return_num_files=False):
        """If return_num_files is True, the method will return early
        with just the number of files.
        """
        # Total number of particles across all files
        num_particles_tot = np.sum([component.N for component in self.components])
        # Closure for doing the divvying up of particles across files,
        # given a maximum number of particles per file.
        def get_num_particle_files(num_particles_file_max):
            # Determine the number of files needed.
            # The exact number may change further down.
            num_files = num_particles_tot//num_particles_file_max + 1
            # Determine a common number of particles of each type
            # to store in each file.
            num_particles_file_common = []
            for component in self.components:
                num_particles_file = component.N//num_files
                if num_particles_file == 0:
                    num_particles_file = 1
                num_particles_file_common.append(num_particles_file)
            # Explicitly store the number of particles for each file
            num_particles_remaining = [component.N for component in self.components]
            num_particle_files = []
            while np.sum(num_particles_remaining) > 0:
                num_particle_file = []
                for j, num_particle in enumerate(num_particles_file_common):
                    num_particle = num_particles_file_common[j]
                    if num_particle > num_particles_remaining[j]:
                        num_particle = num_particles_remaining[j]
                    num_particle_file.append(num_particle)
                    num_particles_remaining[j] -= num_particle
                num_particle_files.append(num_particle_file)
            num_files = len(num_particle_files)
            # Ensure that each file is filled to the brim
            i_left = 0
            i_right = num_files - 1
            while i_left != i_right:
                num_particles_left = num_particle_files[i_left]
                num_particle_left = np.sum(num_particles_left)
                if num_particle_left == num_particles_file_max:
                    i_left += 1
                    continue
                num_particles_right = num_particle_files[i_right]
                for j, num_particle_move in enumerate(num_particles_right):
                    if num_particle_move + num_particle_left > num_particles_file_max:
                        num_particle_move = num_particles_file_max - num_particle_left
                    num_particles_right[j] -= num_particle_move
                    num_particles_left [j] += num_particle_move
                    num_particle_left      += num_particle_move
                num_particle_right = np.sum(num_particles_right)
                if num_particle_right == 0:
                    i_right -= 1
                    continue
            for i in range(num_files - 1, 0, -1):
                if np.sum(num_particle_files[i]) > 0:
                    break
                num_particle_files.pop()
            return num_particle_files
        # Divvy up the particles across files, filling each file with
        # the maximum possible number of particles.
        num_particle_files = get_num_particle_files(self.num_particles_file_max)
        num_files = len(num_particle_files)
        # The number of files is now finally determined
        if return_num_files:
            return num_files
        # Do the divvying up again, using as low a particle number per
        # file as possible, keeping the total number of files the same.
        # This distributes the particles evenly across the files.
        num_particles_file_max = num_particles_tot//num_files
        num_particles_file_max += (num_particles_file_max*num_files < num_particles_tot)
        num_particle_files = get_num_particle_files(num_particles_file_max)
        # Sort in order of most total particles
        # and particles of lower type.
        num_particle_files.sort(
            key=(lambda num_particle_file: (np.sum(num_particle_file), num_particle_file)),
            reverse=True,
        )
        # Sanity checks
        num_particle_files_tot = [
            np.sum(num_particle_file)
            for num_particle_file in num_particle_files
        ]
        if (
            len(num_particle_files) != num_files
            or np.sum(num_particle_files_tot) != num_particles_tot
            or np.max(num_particle_files_tot) > self.num_particles_file_max
        ):
            abort(f'Something went wrong divvying up the particles')
        # Distribute particles within the files across the processes
        num_write_files = self.distribute(
            num_particle_files,
            [component.N_local for component in self.components],
        )
        return num_write_files

    # Method returning information about required file blocks
    def get_blocks_info(self, io):
        # All blocks except the header
        blocks = {
            'POS': {
                'data': [
                    (None if component is None else component.pos_mv)
                    for component in self.components
                ],
                # Three-dimensional positions.
                # Data type will be added below.
                'type': '3',
                # Comoving coordinates
                'unit': [
                    self.unit_length
                    for component in self.components
                ],
            },
            'VEL': {
                'data': [
                    (None if component is None else component.mom_mv)
                    for component in self.components
                ],
                # Three-dimensional velocities.
                # Data type will be added below.
                'type': '3',
                # Peculiar velocities u = a*dx/dt divided by sqrt(a)
                'unit': [
                    (
                        1 if component is None
                        else self.unit_velocity*component.mass*self.header['Time']**1.5
                    )
                    for component in self.components
                ],
            },
            'ID': {
                'data': [
                    (None if component is None else component.ids_mv)
                    for component in self.components
                ],
                # Scalar. Data type will be added below.
                'type': '1',
            },
        }
        # Add data types to format specifications
        for block_name, block in blocks.items():
            num_bits = gadget_snapshot_params['dataformat'][block_name]
            if num_bits == 'automatic':
                if block_name != 'ID':
                    abort(
                        f'Got gadget_snapshot_params["dataformat"][{block_name}] = "{num_bits}", '
                        f'but this can not be determined automatically'
                    )
                # Determine whether to use 32- or 64-bit unsigned
                # integers for the IDs.
                num_bits = 32
                num_particles_tot = np.sum([
                    component.N
                    for component in self.components
                    if component is not None
                ])
                if num_particles_tot > 2**32:
                    num_bits = 64
            elif isinstance(num_bits, str):
                abort(
                    f'Could not understand '
                    f'gadget_snapshot_params["dataformat"][{block_name}] = {num_bits}'
                )
            num_bytes = num_bits//8
            if block_name in {'POS', 'VEL'}:  # floating
                fmts = ['f', 'd']
            elif block_name in {'ID', }:  # unsigned integral
                fmts = ['I', 'Q']
            else:
                abort(f'Block "{block_name}" not given a data type in get_blocks_info()')
            for fmt in fmts:
                if sizesC[fmt] == num_bytes:
                    block['type'] += fmt
                    break
            else:
                abort(f'No {num_bytes} byte type found in {fmts} (needed for "{block_name}")')
        # Return required blocks
        if io == 'names':
            # Return names of all blocks (including the header) in order
            block_names = [self.block_name_header] + list(blocks.keys())
            return block_names
        elif io == 'save':
            block_names = ['POS', 'VEL', 'ID']
        elif io == 'load':
            block_names = ['POS', 'VEL']
            # We only need to load the ID block if some of the
            # components make use of IDs.
            if any([component.use_ids for component in self.components]):
                block_names.append('ID')
        else:
            abort(f'get_blocks_info() got io = "{io}" ∉ {{"save", "load"}}')
        blocks = {block_name: blocks[block_name] for block_name in block_names}
        return blocks

    # Method returning the GADGET snapshot index
    # of a component based on its name.
    def get_component_index(self, component, fail_on_error=True):
        index = -1
        if component.name in self.component_names:
            index = self.component_names.index(component.name)
        elif (
            self.misnamed_halo_component is not None
            and self.misnamed_halo_component.name == component.name
        ):
            # Though the named of the passed component does not match
            # any of the GADGET component names, it is intended for use
            # as the GADGET halo component, i.e. particle type 1.
            index = 1
        elif fail_on_error:
            msg = ', '.join([f'"{name}"' for name in self.component_names])
            abort(
                f'Component name "{component.name}" does not match any of the names '
                f'required for use in {self.name} snapshots. Available names are: {msg}.'
            )
        return index

    # Method for writing out the initial HEAD block
    # of a GADGET snapshot file.
    def write_header(self, filename, num_particles_file_tot):
        # Create header for this particular file
        num_particles_header = [0]*self.num_particle_types
        indices = [self.get_component_index(component) for component in self.components]
        for index, num_particles_file in zip(indices, num_particles_file_tot):
            num_particles_header[index] = num_particles_file
        header = self.header.copy()
        header['Npart'] = num_particles_header
        # Initialize file with HEAD block
        block_size = self.headersize
        with open_file(filename, mode='wb') as f:
            # Start the HEAD block
            self.write_block_bgn(f, block_size, self.block_name_header)
            # Write out header, tallying up its size
            size = 0
            for key, val in self.header_fields.items():
                size += self.write(f, val.fmt, header[key])
            if size > block_size:
                abort(
                    f'The "{self.block_name_header}" block took up {size} bytes '
                    f'but was specified as {block_size}'
                )
            # Pad the header with zeros to fill out its specified size
            size_padding = block_size - size
            self.write(f, 'b', [0]*size_padding)
            # Close the HEAD block
            self.write_block_end(f, block_size)

    # Method for initialising a block on disk
    def write_block_bgn(self, f, block_size, block_name):
        """The passed f may be either a file name
        or a file object to an already opened file.
        """
        block_name_length = int(self.block_name_fmt.rstrip('s'))
        if len(block_name) > block_name_length:
            abort(f'Block name "{block_name}" larger than {block_name_length} characters')
        # Closure for doing the actual writing
        def writeout(f):
            # The initial block meta data
            if self.snapformat == 2:
                self.write(f, 'I', block_name_length*sizesC['s'] + sizesC['I'])
                self.write(f, 's', block_name.ljust(block_name_length).encode('ascii'))
                self.write(f, 'I', sizesC['I'] + block_size + sizesC['I'])
                self.write(f, 'I', sizesC['I'] + block_name_length*sizesC['s'])
            self.write(f, 'I', block_size)
        # Call writeout() in accordance with the supplied f
        if isinstance(f, str):
            filename = f
            with open_file(filename, mode='ab') as f:
                writeout(f)
        else:
            writeout(f)

    # Method for finalising a block on disk
    def write_block_end(self, f, block_size):
        """The passed f may be either a file name
        or a file object to an already opened file.
        """
        # Closure for doing the actual writing
        def writeout(f):
            # The closing int
            self.write(f, 'I', block_size)
        # Call writeout() in accordance with the supplied f
        if isinstance(f, str):
            filename = f
            with open_file(filename, mode='ab') as f:
                writeout(f)
        else:
            writeout(f)

    # Method for writing a series of bytes to the snapshot file
    def write(self, f, fmt, data):
        # Check the type format
        fmt_types = re.findall(r'\D|\?', fmt)
        if len(fmt_types) == 0:
            abort(f'Missing type information in format string "{fmt}"')
        elif len(fmt_types) > 1:
            abort(f'Can only handle a single type in the format string (got "{fmt}")')
        fmt_type = fmt_types[0]
        # Add quantifier if missing
        data = any2list(data)
        if isinstance(data[0], (str, bytes)):
            size = len(data[0])
        else:
            size = len(data)
        if size == 0:
            return 0
        if size > 1 and not re.search(r'\d', fmt[0]):
            fmt = f'{size}{fmt}'
        # Correct floating-point data if double-precision
        if fmt_type == 'd':
            data = correct_float(data)
        # Write data to disk
        f.write(struct.pack(fmt, *data))
        # Return the number of bytes written
        return struct.calcsize(fmt)

    # Method for loading in a GADGET snapshot from disk
    @cython.pheader(
        # Arguments
        filename=str,
        only_params='bint',
        # Locals
        N='Py_ssize_t',
        N_local='Py_ssize_t',
        N_str=str,
        block=dict,
        block_name=str,
        block_size='Py_ssize_t',
        block_type=str,
        blocks=dict,
        blocks_required=set,
        bytes_per_particle='int',
        bytes_per_particle_dim='int',
        check='int',
        chunk='double[::1]',
        chunk_arr=object,  # np.ndarray
        chunk_doubleprec='double[::1]',
        chunk_doubleprec_ptr='double*',
        chunk_ptr='double*',
        chunk_singleprec='float[::1]',
        chunk_singleprec_ptr='float*',
        chunk_size='Py_ssize_t',
        component='Component',
        components_skipped_names=list,
        data='double[::1]',
        data_components=list,
        data_value='double',
        dtype=object,
        end_of_file='bint',
        filename_candidate=str,
        filename_glob=str,
        filename_i=str,
        filenames=list,
        header=dict,
        header_backup=dict,
        header_i=dict,
        index_chunk='Py_ssize_t',
        indexᵖ='Py_ssize_t',
        indexʳ='Py_ssize_t',
        i='Py_ssize_t',
        id_counters='Py_ssize_t[::1]',
        ids_arr=object,  # np.ndarray
        ids_mv='Py_ssize_t[::1]',
        j='Py_ssize_t',
        j_populated=list,
        key=str,
        load_pos='bint',
        load_vel='bint',
        mass='double',
        msg=str,
        msg_list=list,
        name=str,
        ndim='int',
        num_files='Py_ssize_t',
        num_local=list,
        num_particles_component='Py_ssize_t',
        num_particles_component_header=object,  # Python int
        num_particles_components=list,
        num_particles_file='Py_ssize_t',
        num_particles_files=list,
        num_particles_proc='Py_ssize_t',
        num_particles_tot=list,
        num_particle_files=list,
        num_read='Py_ssize_t',
        num_read_file=list,
        num_read_file_locals=list,
        num_read_files=list,
        offset='Py_ssize_t',
        offset_header='Py_ssize_t',
        offset_nextblock='Py_ssize_t',
        plural=str,
        representation=str,
        size_read='Py_ssize_t',
        species=object,  # str or None
        start_local='Py_ssize_t',
        unit='double',
        unit_components=list,
    )
    def load(self, filename, only_params=False):
        # Determine which files are part of this snapshot
        if master:
            if (
                not filename.endswith('*')
                and os.path.isfile(filename)
            ) and (
                not filename.endswith('.0')
                or allow_snapshot_multifile_singleload
            ):
                filenames = [filename]
            else:
                if os.path.isdir(filename):
                    filenames = [
                        filename_candidate
                        for filename_candidate in glob(f'{filename}/*.0')
                        if self.is_this_type(filename_candidate)
                    ]
                    if len(filenames) != 1:
                        msg = ', '.join([f'"{filename}"' for filename in filenames])
                        abort(
                            f'Found several candidates for the '
                            f'first {self.name} snapshot file: {msg}'
                        )
                    filename = filenames[0]
                filename = filename.removesuffix('.0').rstrip('.*')
                filenames = []
                for i in itertools.count():
                    filename_candidate = f'{filename}.{i}'
                    if (
                            os.path.isfile(filename_candidate)
                        and self.is_this_type(filename_candidate)
                    ):
                        filenames.append(filename_candidate)
                    else:
                        break
                if not filenames:
                    abort(f'Could not locate {self.name} snapshot "{filename}"')
        filenames = bcast(filenames if master else None)
        # Progress message
        msg = ''
        if only_params:
            msg = 'parameters of '
        if len(filenames) == 1:
            filename = filenames[0]
            masterprint(f'Loading {msg}snapshot "{filename}" ...')
        else:
            filename_glob = filenames[0].removesuffix('.0') + '*'
            masterprint(f'Loading {msg}snapshot "{filename_glob}" ...')
        # Determine the GADGET SnapFormat of the first file
        # (we assume that they are all stored in the same SnapFormat).
        self.snapformat = bcast(self.get_snapformat(filenames[0]) if master else None)
        if self.snapformat not in (1, 2):
            abort(f'Could not determine GADGET SnapFormat of "{filename}"')
        if self.snapformat == 1:
            # For SnapFormat 1 the block names are left out of the
            # snapshot, but they occur in a specific order.
            self.block_names = iter(self.get_blocks_info('names'))
            # Consume the header block name
            next(self.block_names)
        # Read in the header of each file in the snapshot and check that
        # they are consistent. Keep information about the number of
        # particles of each type within each snapshot.
        num_particles_files = []
        if master:
            num_files = -1
            for i, filename in enumerate(filenames):
                if i == num_files:
                    break
                with open_file(filename, mode='rb') as f:
                    # Read in and store header
                    header_i, offset_header = self.read_header(f)
                    num_particles_files.append(header_i['Npart'])
                    if i == 0:
                        # Check on number of snapshot files
                        header = header_i
                        num_files = header['NumFiles']
                        if not allow_snapshot_multifile_singleload:
                            if num_files > len(filenames):
                                msg = (
                                    f'Could only locate {len(filenames)} of the supposed '
                                    f'{num_files} files making up the snapshot.'
                                )
                                match = re.search(r'\.(\d+)$', filename)
                                if match:
                                    if int(match.group(1)) != 0:
                                        msg += f' Is "{filename}" not the first file of the snapshot?'
                                abort(msg)
                            elif num_files < len(filenames):
                                masterwarn(
                                    f'The snapshot supposedly consists of {num_files} files, '
                                    f'but {len(filenames)} files were found. '
                                    f'The last {len(filenames) - num_files} will be ignored.'
                                )
                    else:
                        # Compare i'th header against the first
                        for key, val in header.items():
                            val_i = header_i.get(key)
                            if key != 'Npart' and val != val_i:
                                masterwarn(
                                    f'Disagreement between "{filenames[0]}" and "{filename}": '
                                    f'{key} = {val} vs {val_i}. The first value will be used.'
                                )
            filenames = filenames[:num_files]
        # Broadcast results
        filenames           = bcast(filenames           if master else None)  # re-broadcast
        header              = bcast(header              if master else None)
        offset_header       = bcast(offset_header       if master else None)
        num_particles_files = bcast(num_particles_files if master else None)
        num_files = len(filenames)
        # Check whether the particle count matches the header record
        num_particles_components = list(
            asarray(num_particles_files, dtype=C2np['Py_ssize_t']).sum(axis=0)
        )
        header_backup = header
        for check in range(2):
            if check == 1:
                # Assume that the N-genIC convention
                # of storing NallHW[1] in Nall[2] is used.
                # Create copy of header with the standard convention.
                header = deepcopy(header)
                header['NallHW'][1] = header['Nall'][2]
                header['Nall'][2] = 0
            for j, num_particles_component in enumerate(num_particles_components):
                num_particles_component_header = header['Nall'][j] + 2**32*header['NallHW'][j]
                if num_particles_component_header != num_particles_component:
                    # Disagreement found between the ('Nall', 'NallHW')
                    # of the header and of the {'Npart'} of the
                    # snapshot files.
                    break
            else:
                break
        else:
            if num_files == header['NumFiles']:
                masterwarn(
                    f'Inconsistent particle counts in header. Got Nall = {header_backup["Nall"]} '
                    f'and NallHW = {header_backup["NallHW"]} while Npart summed over all files '
                    f'is {num_particles_components}. Will use the values from Npart.'
                )
        header = header_backup
        self.header = header
        self.h  # To ensure non-zero h
        # From now on the header stays the same.
        # Set parameters.
        self.params['H0'     ] = self.header['HubbleParam']*(100*units.km/(units.s*units.Mpc))
        self.params['a'      ] = self.header['Time']
        self.params['boxsize'] = self.header['BoxSize']*self.unit_length
        self.params['Ωm'     ] = self.header['Omega0']
        self.params['ΩΛ'     ] = self.header['OmegaLambda']
        # Divvy up the particles so that each process gets the same
        # number of each type of particle. Initialise components.
        num_local = []
        self.components.clear()
        components_skipped_names = []
        for j, num_particles_component in enumerate(num_particles_components):
            if num_particles_component == 0:
                continue
            # Determine local number of particles
            num_particles_proc = num_particles_component//nprocs
            num_particles_proc += (rank < num_particles_component - num_particles_proc*nprocs)
            num_local.append(num_particles_proc)
            # Get basic component information
            name = self.component_names[j]
            representation = 'particles'
            species = determine_species(name, representation)
            # Skip this component if it should not be loaded
            if not should_load(name, species, representation):
                # Skip by appending None as a placeholder
                self.components.append(None)
                components_skipped_names.append(name)
                continue
            # Instantiate component
            mass = self.header['Massarr'][j]
            if mass <= 0:
                masterwarn(f'Mass of "{name}" particles is {mass}×10¹⁰ h⁻¹ m☉')
            mass *= self.unit_mass
            component = Component(name, species, N=num_particles_component, mass=mass)
            self.components.append(component)
        # Done loading component attributes
        if only_params:
            self.components = [
                component
                for component in self.components
                if component is not None
            ]
            masterprint('done')
            return
        # If no components are to be read, return now
        if len(self.components) == len(components_skipped_names):
            msg = ', '.join(components_skipped_names)
            masterprint(f'Skipping {msg}')
            self.components = [
                component
                for component in self.components
                if component is not None
            ]
            masterprint('done')
            return
        # Only keep particle counts for non-empty components
        j_populated = [
            j
            for j, num_particles_component in enumerate(num_particles_components)
            if num_particles_component > 0
        ]
        for i in range(num_files):
            num_particle_files = num_particles_files[i]
            num_particles_files[i] = [num_particle_files[j] for j in j_populated]
        # Distribute particles within the files across the processes
        num_read_files = self.distribute(num_particles_files, num_local)
        # Progress message
        msg_list = []
        for component in self.components:
            if component is None:
                continue
            N = component.N
            N_str = get_cubenum_strrep(N)
            plural = ('s' if N > 1 else '')
            msg_list.append(f'{component.name} ({N_str} {component.species}) particle{plural}')
        msg = ', '.join(msg_list)
        if components_skipped_names:
            msg_list = [msg]
            msg = ', '.join(components_skipped_names)
            msg_list.append(f'(skipping {msg})')
            msg = ' '.join(msg_list)
        masterprint(f'Reading in {msg} ...')
        # Determine whether to load any positions and/or velocities
        load_pos = load_vel = False
        for component in self.components:
            if component is None:
                continue
            if component.snapshot_vars['load']['pos']:
                load_pos = True
            if component.snapshot_vars['load']['mom']:
                load_vel = True
        # Construct ID counters for each component.
        # Only used in case the snapshot does not include IDs.
        num_particles_tot = [
            (0 if component is None else component.N)
            for component in self.components
        ]
        id_counters = np.concatenate((
            [0],
            np.cumsum(num_particles_tot, dtype=C2np['Py_ssize_t']),
        ))[:len(num_particles_tot)]
        # Enlarge components in order to accommodate particles
        for component, N_local in zip(self.components, num_local):
            if component is None:
                continue
            component.N_local = N_local
            if component.N_allocated < N_local:
                component.resize(N_local, only_loadable=True)
        # Get information about the blocks to be read in
        blocks = self.get_blocks_info('load')
        if not load_pos:
            blocks.pop('POS', None)
        if not load_vel:
            blocks.pop('VEL', None)
        # Read in each file in turn
        for i, filename_i in enumerate(filenames):
            if num_files > 1:
                masterprint(f'Reading snapshot file {i}/{num_files - 1} ...')
            num_particles_file = np.sum(num_particles_files[i])
            num_read_file = num_read_files[i]
            # Iterate over required blocks. The order is not important
            # and will be determined from the file. All files should
            # contain all of the required blocks.
            offset_nextblock = offset_header
            blocks_required = set(blocks.keys())
            end_of_file = False
            if self.snapformat == 1:
                # For SnapFormat 1 the block names are left out of the
                # snapshot, but they occur in a specific order.
                # The header block has already been read in.
                self.block_names = iter(self.get_blocks_info('names'))
                next(self.block_names)
            while True:
                # Find next required block
                if rank == 0:
                    with open_file(filename_i, mode='rb') as f:
                        while True:
                            # Seek to next block
                            offset_nextblock, block_size, block_name = (
                                self.read_block_bgn(f, offset_nextblock)
                            )
                            if offset_nextblock == -1:
                                end_of_file = True
                                break
                            if block_name not in blocks:
                                masterprint(f'Skipping block "{block_name}"')
                                continue
                            if block_name not in blocks_required:
                                masterwarn(f'Skipping repeated block "{block_name}"')
                                continue
                            if block_size%num_particles_file:
                                abort(
                                    f'File {filename_i} contains {num_particles_file} particles '
                                    f'but its "{block_name}" block has a size of {block_size} '
                                    f'bytes, which does not divide the particle number.'
                                )
                            # Arrived at required block
                            offset = f.tell()
                            break
                    bcast(end_of_file)
                    if end_of_file:
                        if blocks_required == {'ID'}:
                            # IDs are required but not found within the
                            # file. We shall generate these ourselves.
                            break
                        elif blocks_required:
                            plural = ('s' if len(blocks_required) > 1 else '')
                            abort(
                                f'Could not find required block{plural}',
                                ', '.join([f'"{block_name}"' for block_name in blocks_required]),
                            )
                        break
                    bcast((block_name, block_size))
                else:
                    end_of_file = bcast()
                    if end_of_file:
                        break
                    block_name, block_size = bcast()
                blocks_required.remove(block_name)
                block = blocks[block_name]
                data_components = block.get('data')
                unit_components = block.get('unit')
                block_type = block['type']
                # Figure out the size of the data type
                # used by this block.
                bytes_per_particle = block_size//num_particles_file
                ndim = int(re.search(r'^\d+', block_type).group())
                bytes_per_particle_dim = bytes_per_particle//ndim
                if bytes_per_particle_dim*ndim != bytes_per_particle:
                    abort(
                        f'Block "{block_name}" stores {ndim}-dimensional data '
                        f'but contains {bytes_per_particle} bytes per particle, '
                        f'which is not divisible by {ndim}.'
                    )
                # Iterate over all components. The block is organised
                # so that all data belonging to a given component
                # is provided consecutively.
                if block_name in {'POS', 'VEL'}:
                    if bytes_per_particle_dim == 4:
                        # Single-precision floating point format
                        dtype = C2np['float']
                    elif bytes_per_particle_dim == 8:
                        # Double-precision floating point format
                        dtype = C2np['double']
                    else:
                        abort(
                            f'No data format with a size of {bytes_per_particle_dim} bytes '
                            f'implemented for block "{block_name}"'
                        )
                    for j, (num_read, component, data, unit) in enumerate(
                        zip(num_read_file, self.components, data_components, unit_components)
                    ):
                        size_read = 3*num_read
                        # Get file offset from previous process
                        if rank > 0 or (nprocs > 1 and j > 0):
                            offset = recv(source=mod(rank - 1, nprocs))
                        # Read in block data
                        if component is not None and num_read > 0:
                            with open_file(filename_i, mode='rb') as f:
                                # Seek to where the previous
                                # process left off.
                                f.seek(offset)
                                # Read in using chunks
                                chunk_size = np.min((size_read, ℤ[self.chunk_size_max//8]))
                                for indexʳ in range(0, size_read, chunk_size):
                                    if indexʳ + chunk_size > size_read:
                                        chunk_size = size_read - indexʳ
                                    chunk = data[indexʳ:(indexʳ + chunk_size)]
                                    chunk_ptr = cython.address(chunk[:])
                                    # Read in chunk, then copy it to
                                    # double precision chunk while
                                    # applying unit conversion. In the
                                    # case of positions, safeguard
                                    # against round-off errors.
                                    chunk_arr = np.fromfile(
                                        f,
                                        dtype=dtype,
                                        count=chunk_size,
                                    )
                                    if chunk_arr.shape[0] < chunk_size:
                                        abort(f'Ran out of bytes in block "{block_name}"')
                                    if 𝔹[dtype is C2np['float']]:
                                        chunk_singleprec = chunk_arr
                                        chunk_singleprec_ptr = cython.address(chunk_singleprec[:])
                                    else:  # dtype is C2np['double']:
                                        chunk_doubleprec = chunk_arr
                                        chunk_doubleprec_ptr = cython.address(chunk_doubleprec[:])
                                    # Copy single-precision chunk
                                    # into chunk while applying
                                    # unit conversion.
                                    for index_chunk in range(chunk_size):
                                        with unswitch(1):
                                            if 𝔹[dtype is C2np['float']]:
                                                data_value = chunk_singleprec_ptr[index_chunk]*unit
                                            else:  # dtype is C2np['double']
                                                data_value = chunk_doubleprec_ptr[index_chunk]*unit
                                        # In the case of positions,
                                        # safeguard against
                                        # round-off errors.
                                        with unswitch(3):
                                            if block_name == 'POS':
                                                if data_value >= boxsize:
                                                    data_value -= boxsize
                                        chunk_ptr[index_chunk] = data_value
                            # Crop the populated part of the data away
                            # from the memory view. Note that this
                            # changes the content of the block object
                            # returned by get_blocks_info().
                            data_components[j] = data[size_read:]
                        # Update offset, to be used by the next process
                        offset += size_read*dtype().itemsize
                        # Inform the next process about the file offset
                        if rank < nprocs - 1 or (nprocs > 1 and j < len(self.components) - 1):
                            send(offset, dest=mod(rank + 1, nprocs))
                elif block_name == 'ID':
                    # The particle IDs are stored as 32- or 64-bit
                    # unsigned ints. We read them in as signed ints,
                    # as they are stored in the code as signed.
                    if bytes_per_particle_dim == 4:
                        dtype = np.int32
                    elif bytes_per_particle_dim == 8:
                        dtype = np.int64
                    else:
                        abort(
                            f'No data format with a size of {bytes_per_particle_dim} bytes '
                            f'implemented for block "{block_name}"'
                        )
                    for j, (num_read, component, ids_mv) in enumerate(
                        zip(num_read_file, self.components, data_components)
                    ):
                        # Get file offset from previous process
                        if rank > 0 or (nprocs > 1 and j > 0):
                            offset = recv(source=mod(rank - 1, nprocs))
                        # Read in block data
                        if component is not None and component.use_ids and num_read > 0:
                            with open_file(filename_i, mode='rb') as f:
                                # Seek to where the previous
                                # process left off.
                                f.seek(offset)
                                # Read in using chunks
                                chunk_size = np.min((num_read, ℤ[self.chunk_size_max//8]))
                                for indexᵖ in range(0, num_read, chunk_size):
                                    if indexᵖ + chunk_size > num_read:
                                        chunk_size = num_read - indexᵖ
                                    # Read in chunk and copy it into the
                                    # component data array, implicitly
                                    # converting to 64-bit as necessary.
                                    chunk_arr = np.fromfile(
                                        f,
                                        dtype=dtype,
                                        count=chunk_size,
                                    )
                                    if chunk_arr.shape[0] < chunk_size:
                                        abort(f'Ran out of bytes in block "{block_name}"')
                                    ids_arr = asarray(ids_mv[indexᵖ:(indexᵖ + chunk_size)])
                                    ids_arr[:] = chunk_arr
                            # Crop the populated part of the data away
                            # from the memory view. Note that this
                            # changes the content of the block object
                            # returned by get_blocks_info().
                            data_components[j] = ids_mv[num_read:]
                        # Update offset, to be used by the next process
                        offset += num_read*dtype().itemsize
                        # Inform the next process about the file offset
                        if rank < nprocs - 1 or (nprocs > 1 and j < len(self.components) - 1):
                            send(offset, dest=mod(rank + 1, nprocs))
                else:
                    abort(f'Does not know how to read {self.name} block "{block_name}"')
                # Let all the processes catch up,
                # ensuring that the file is closed.
                Barrier()
            # If any of the components should make use of particle IDs
            # but none are stored in the snapshot file, assign IDs
            # according to the order in which the particles
            # are stored within the file.
            if blocks_required == {'ID'}:
                masterprint('Assigning particle IDs ...')
                block = blocks['ID']
                data_components = block.get('data')
                # Get number of particles of each type
                # which should have been read by each process.
                num_read_file_locals = allgather(num_read_file)
                # Loop as when reading in, but do so in parallel
                for j, (num_read, component, ids_mv) in enumerate(
                    zip(num_read_file, self.components, data_components)
                ):
                    if component is None or not component.use_ids:
                        continue
                    # Get local starting index
                    start_local = np.sum(
                        [
                            num_read_file_local[j]
                            for num_read_file_local in num_read_file_locals[:rank]
                        ],
                        dtype=C2np['Py_ssize_t'],
                    )
                    # Assign consecutive IDs to num_read particles
                    for indexᵖ in range(num_read):
                        ids_mv[indexᵖ] = ℤ[id_counters[j] + start_local] + indexᵖ
                    # Crop the populated part of the data away from the
                    # memory view. Note that this changes the content of
                    # the block object returned by get_blocks_info().
                    data_components[j] = ids_mv[num_read:]
                    # Update ID counter, accounting for all processes
                    id_counters[j] += np.sum(
                        [
                            num_read_file_local[j]
                            for num_read_file_local in num_read_file_locals
                        ],
                        dtype=C2np['Py_ssize_t'],
                    )
                masterprint('done')
            # Done loading this snapshot file
            if len(filenames) > 1:
                masterprint('done')
        # Done loading entire snapshot
        self.components = [
            component
            for component in self.components
            if component is not None
        ]
        masterprint('done')
        masterprint('done')

    # Method for reading in the initial HEAD block
    # of a GADGET snapshot file.
    def read_header(self, f):
        """This method read the HEAD block into a dict.
        The passed f may be either a file name or a file object to an
        already opened file. The heading information will be returned
        as is, with no unit conversion performed.
        """
        # Closure for doing the actual reading
        def readin(f):
            header = {}
            offset = 0
            f.seek(offset)
            offset, size, block_name = self.read_block_bgn(f, offset, self.block_name_header)
            if size == -1:
                abort(
                    f'Expected block "{self.block_name_header}" at the '
                    f'beginning of the file but found nothing'
                )
            if block_name != self.block_name_header:
                abort(
                    f'Expected block "{self.block_name_header}" at the '
                    f'beginning of the file but found "{block_name}"'
                )
            size_expected = self.headersize
            if size != size_expected:
                warn(
                    f'Block "{self.block_name_header}" has size {size} '
                    f'but expected {size_expected}'
                )
            for key, val in self.header_fields.items():
                header[key] = self.read(f, val.fmt)
            return header, offset
        # Call readin() in accordance with the supplied f
        if isinstance(f, str):
            filename = f
            with open_file(filename, mode='rb') as f:
                header, offset = readin(f)
        else:
            header, offset = readin(f)
        return header, offset

    # Method for reading in the name and size of a block,
    # given the offset to where it begins.
    def read_block_bgn(self, f, offset, block_name=''):
        def read_size(f, offset):
            f.seek(offset)
            # Each block is bracketed with an unsigned int
            # containing the size of the block.
            size = self.read(f, 'I')
            if size is None:
                offset = size = -1
            else:
                offset += sizesC['I'] + size + sizesC['I']
            return offset, size
        offset, size = read_size(f, offset)
        if self.snapformat == 1:
            # In the case of SnapFormat 1 we are already done.
            # Which block is up is not specified explicitly in the file,
            # so we rely on the ordered block_names.
            if not block_name:
                try:
                    block_name = next(self.block_names)
                except StopIteration:
                    # No more blocks to read in
                    offset = size = -1
            return offset, size, block_name
        size_bare = -1
        if offset != -1:
            block_name = self.read(f, self.block_name_fmt).decode('utf8').rstrip()
            size = self.read(f, 'I')
            offset, size_bare = read_size(f, offset)
            size_bare_2 = size - 2*sizesC['I']
            if size_bare != size_bare_2:
                # The two sizes do not agree.
                # Pick one according to the 'settle'
                # gadget_snapshot_params parameter.
                msg = (
                    f'Size of block "{block_name}" not consistent: '
                    f'{size} - {2*sizesC["I"]} = {size_bare_2} ≠ {size_bare}. '
                )
                if gadget_snapshot_params['settle'] == 0:
                    msg += (
                        f'Choosing to go with a block size of {size_bare}. '
                        f'To instead pick {size_bare_2}, rerun with '
                        f'gadget_snapshot_params["settle"] = 1.'
                    )
                else:
                    msg += (
                        f'Choosing to go with a block size of {size_bare_2}. '
                        f'To instead pick {size_bare}, rerun with '
                        f'gadget_snapshot_params["settle"] = 0.'
                    )
                    offset += size_bare_2 - size_bare
                    size_bare = size_bare_2
                warn(msg)
        return offset, size_bare, block_name

    # Method for reading series of bytes from the snapshot file
    def read(self, f, fmt):
        # Convert bytes to Python objects and store them in a tuple
        payload = f.read(struct.calcsize(fmt))
        if not payload:
            return None
        t = struct.unpack(fmt, payload)
        # If the tuple contains just a single element,
        # return this element rather than the tuple.
        if len(t) == 1:
            return t[0]
        # It is nicer to use mutable lists than immutable tuples
        return list(t)

    # This method populate the snapshot with component data
    # and additional header information.
    def populate(self, components, params=None):
        if not components:
            abort(f'Cannot save a {self.name} snapshot with no components')
        if params is None:
            params = {}
        # Only particle components with names matching the pre-defined
        # GADGET component names may be used. If however just a single
        # matter or cold dark matter particle component is supplied,
        # use this as the GADGET halo component.
        num_valid_halo_components = np.sum([
            component.representation == 'particles'
            and component.species in {'matter', 'cold dark matter'}
            for component in components
        ])
        components_possible = [None]*self.num_particle_types
        for component in components:
            if component.representation != 'particles':
                masterwarn(
                    f'Leaving out the {component.representation} component '
                    f'"{component.name}" from the {self.name} snapshot as only '
                    f'particle components are supported'
                )
                continue
            index = self.get_component_index(component, fail_on_error=False)
            if index == -1:
                if (
                    num_valid_halo_components == 1
                    and component.species in {'matter', 'cold dark matter'}
                ):
                    # Use this component as the GADGET halo component
                    index = 1
                    self.misnamed_halo_component = component
                    masterprint(
                        f'Mapping "{component.name}" component '
                        f'to "{self.component_names[index]}"'
                    )
                else:
                    msg = ', '.join([f'"{name}"' for name in self.component_names])
                    masterwarn(
                        f'Leaving out the "{component.name}" component '
                        f'from the {self.name} snapshot, as its name '
                        f'is not a valid {self.name} component name. '
                        f'Valid names are: {msg}.'
                    )
                    continue
            components_possible[index] = component
        # While the components_possible keeps entries for
        # non-existing components, this should not be so
        # for the components attribute.
        self.components = [
            component
            for component in components_possible
            if component is not None
        ]
        if not self.components:
            abort(f'No components left to store in the {self.name} snapshot')
        # Populate snapshot with the passed scale factor
        # and global parameters. If a params dict is passed,
        # use values from this instead.
        self.params['H0'] = params.get('H0', H0)
        if enable_Hubble:
            self.params['a'] = params.get('a', universals.a)
        else:
            self.params['a'] = universals.a
        self.params['boxsize'] = params.get('boxsize', boxsize)
        self.params['Ωm'] = params.get('Ωm', Ωm)
        ΩΛ = 1 - self.params['Ωm']  # Flat universe with only matter and cosmological constant
        self.params['ΩΛ'] = params.get('ΩΛ', ΩΛ)
        # Build the GADGET header.
        # Fields not relevant will be left with a default value of 0.
        # Though relevant, the Npart field will be left at 0 as this is
        # individual to each file of the snapshot.
        # The Nall and NallHW fields are set so that the total number
        # of particles is Nall + 2**32*NallHW.
        num_particles = [
            (0 if component is None else component.N)
            for component in components_possible
        ]
        num_particles_hw = [
            num//2**32
            for num in num_particles
        ]
        num_particles_lw = [
            num - 2**32*num_hw
            for num, num_hw in zip(num_particles, num_particles_hw)
        ]
        if gadget_snapshot_params['Nall high word'] == 'Nall':
            # The "Nall" convention should be used in place of the
            # standard "NallHW" convention for storing the highest 32
            # bits of the total number of particles. In this scheme,
            # Nall[j] = 0 for all j but j = 1, which then store the
            # lower 32 bits of the total number of particles of type 1.
            # The 32 higher bits of particle type 1 is then stored
            # in Nall[2]. This scheme is used by at least some versions
            # of N-genIC.
            for j, num in enumerate(num_particles):
                if num != 0 and j != 1:
                    msg = ', '.join([
                        f'{num} particles of type {j} ({self.component_names[j]})'
                        for j, num in enumerate(num_particles)
                        if num != 0
                    ])
                    abort(
                        f'The "Nall" convention for storing the higher 32 bits '
                        f'can only be used when you have a single component, '
                        f'specifically type 1 ({self.component_names[1]}), '
                        f'but you have {msg}.'
                    )
            num_particles_lw[2] = num_particles_hw[1]
            num_particles_hw[1] = 0
        self.header.clear()
        for key, val in self.header_fields.items():
            self.header[key] = deepcopy(val.default)
        self.header['HubbleParam'] = (  # should be set before using units
            self.params['H0']/(100*units.km/(units.s*units.Mpc))
        )
        for j, component in enumerate(components_possible):
            if component is not None:
                self.header['Massarr'][j] = component.mass/self.unit_mass
        self.header['Time'       ] = self.params['a']
        self.header['Redshift'   ] = 1/self.params['a'] - 1
        self.header['Nall'       ] = num_particles_lw
        self.header['NumFiles'   ] = self.get_num_files()
        self.header['BoxSize'    ] = self.params['boxsize']/self.unit_length
        self.header['Omega0'     ] = self.params['Ωm']
        self.header['OmegaLambda'] = self.params['ΩΛ']
        self.header['NallHW'     ] = num_particles_hw
        # Overwrite header fields according to user specifications
        def transform(key):
            return key.lower().replace(' ', '').replace('-', '').replace('_', '')
        for key, val in gadget_snapshot_params['header'].items():
            key_in_header = key
            if key not in self.header:
                key_simple = transform(key)
                for key_in_header in self.header.keys():
                    key_in_header_simple = transform(key_in_header)
                    if key_simple == key_in_header_simple:
                        break
                else:
                    masterwarn(f'Unknown {self.name} header field "{key}" will not be set')
                    continue
            val_in_header = self.header[key_in_header]
            if isinstance(val_in_header, list):
                val = any2list(val)
                if len(val) < len(val_in_header):
                    val.extend(val_in_header[len(val):])
                elif len(val) > len(val_in_header):
                    masterwarn(
                        f'Too many elements in {self.name} header field "{key}": {val}. '
                        f'Only the first {len(val_in_header)} elements will be used.'
                    )
                    val = val[:len(val_in_header)]
                val = [type(el_in_header)(el) for el, el_in_header in zip(val, val_in_header)]
                overwriting_set_value = any([
                    el != el_in_header != el_in_field
                    for el, el_in_header, el_in_field in zip(
                        val, val_in_header, self.header_fields[key_in_header].default,
                    )
                ])
            else:
                val = type(val_in_header)(val)
                overwriting_set_value = (
                    val != val_in_header != self.header_fields[key_in_header].default
                )
            if overwriting_set_value:
                masterwarn(
                    f'Overwriting header field "{key_in_header}": '
                    f'{val_in_header} → {val}'
                )
            self.header[key_in_header] = val

    # Method for getting the number of files over which
    # this snapshot is or will be distributed.
    def get_num_files(self):
        num_files = self.header.get('NumFiles', 0)
        if num_files == 0:
            num_files = self.divvy(return_num_files=True)
        return num_files

# Class storing a TIPSY snapshot. Besides holding methods for
# saving/loading, it stores particle data and the TIPSY header.
@cython.cclass
class TipsySnapshot:
    """This class represents snapshots of the "TIPSY" type.
    As is the case for the CO𝘕CEPT snapshot class, this class contains
    a list components (the components attribute) and dict of parameters
    (the params attribute).
    """
    # The properly written name of this snapshot type
    # (only used for printing).
    name = 'TIPSY'
    # The filename extension for this type of snapshot
    extension = ''
    # Maximum allowed chunk size in bytes
    chunk_size_max = 2**23  # 8 MB
    # Names of components contained in snapshots, in order
    component_names = [
        f'TIPSY {particle_type}'
        for particle_type in ['sph', 'dark', 'star']
    ]
    num_particle_types = len(component_names)
    # Ordered fields in the TIPSY header,
    # mapped to their type and default value.
    TipsyField = collections.namedtuple(
        'TipsyHeaderField',
        ['fmt', 'default'],
        defaults=['', 0],
    )
    header_fields = {
        'time'   : TipsyField('d'),
        'nbodies': TipsyField('I'),
        'ndim'   : TipsyField('I'),
        'nsph'   : TipsyField('I'),
        'ndark'  : TipsyField('I'),
        'nstar'  : TipsyField('I'),
    }
    headersize = 0
    for val in header_fields.values():
        headersize += struct.calcsize(val.fmt)
    # Ensure floating-point defaults where appropriate
    for key, val in header_fields.items():
        if val.fmt in {'f', 'd'}:
            header_fields[key] = val._replace(default=float(val.default))
    # Ordered fields in the TIPSY particle structures,
    # mapped to their type and default value.
    particle_fields = {
        'sph': {
            'mass'   : TipsyField('f'),
            'pos'    : TipsyField('{ndim}f'),
            'vel'    : TipsyField('{ndim}f'),
            'rho'    : TipsyField('f'),
            'temp'   : TipsyField('f'),
            'hsmooth': TipsyField('f'),
            'metals' : TipsyField('f'),
            'phi'    : TipsyField('f'),
        },
        'dark': {
            'mass': TipsyField('f'),
            'pos' : TipsyField('{ndim}f'),
            'vel' : TipsyField('{ndim}f'),
            'eps' : TipsyField('f'),
            'phi' : TipsyField('f'),
        },
        'star': {
            'mass'  : TipsyField('f'),
            'pos'   : TipsyField('{ndim}f'),
            'vel'   : TipsyField('{ndim}f'),
            'metals': TipsyField('f'),
            'tform' : TipsyField('f'),
            'eps'   : TipsyField('f'),
            'phi'   : TipsyField('f'),
        },
    }
    # Ensure floating-point defaults where appropriate
    for fields in particle_fields.values():
        for key, val in fields.items():
            for c in {'f', 'd'}:
                if val.fmt.endswith(c):
                    fields[key] = val._replace(default=float(val.default))
                    break

    # Class method for identifying a file to be a snapshot of this type
    @classmethod
    def is_this_type(cls, filename):
        if not os.path.isfile(filename):
            return False
        # Test for TIPSY by checking the 'ndim' field of the header
        try:
            header, endianness = cls.read_header(filename)
            if header.get('ndim') in {1, 2, 3}:
                return True
        except Exception:
           pass
        return False

    # Initialisation method
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the TipsySnapshot type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public dict params
        public list components
        public dict header
        public str endianness
        """
        # Dict containing all the parameters of the snapshot
        self.params = {}
        # List of Component instances
        self.components = []
        # Header
        self.header = {}
        # The endianness of the binary data
        self.endianness = '@'  # '@' → native, '<' → little, '>' → big
        # Check on low level type sizes
        sizes_expected = {'I': 4, 'f': 4, 'd': 8}
        for fmt, size_expected in sizes_expected.items():
            size = sizesC[fmt]
            if size_expected != size:
                masterwarn(
                    f'Expected C type \'{fmt2C[fmt]}\' to have a size '
                    f'of {size_expected} bytes, but its size is {size}'
                )

    # Method for reading in the header of a TIPSY snapshot file
    @classmethod
    def read_header(cls, f):
        def _read_header(f):
            for endianness in '<>':
                f.seek(0)
                header = {
                    field: struct.unpack(
                        f'{endianness}{val.fmt}',
                        f.read(struct.calcsize(val.fmt)),
                    )[0]
                    for field, val in cls.header_fields.items()
                }
                if header.get('ndim') in {1, 2, 3}:
                    break
            else:
                endianness = '?'
            return header, endianness
        if isinstance(f, str):
            with open_file(f, mode='rb') as f:
                return _read_header(f)
        else:
            return _read_header(f)

    # Method that saves the snapshot to a TIPSY file
    @cython.pheader(
        # Argument
        filename=str,
        save_all='bint',
        # Locals
        returns=str,
    )
    def save(self, filename, save_all=False):
        abort('Saving snapshots in TIPSY format is not implemented')
        return filename

    # Method for loading in a TIPSY snapshot from disk
    @cython.pheader(
        # Argument
        filename=str,
        only_params='bint',
        # Locals
        N='Py_ssize_t',
        indexʳ='Py_ssize_t',
        mom='double*',
        pos='double*',
        unit='double',
    )
    def load(self, filename, only_params=False):
        if only_params:
            masterprint(f'Loading parameters of snapshot "{filename}" ...')
        else:
            masterprint(f'Loading snapshot "{filename}" ...')
        # Read in the header
        self.header, self.endianness = self.read_header(filename)
        num_particles = {
            particle_type: self.header[f'n{particle_type}']
            for particle_type in self.particle_fields.keys()
        }
        if self.header['nbodies'] != np.sum(list(num_particles.values())):
            masterwarn(
                f'The total number of particles ({self.header["nbodies"]}) does not match '
                f'the individual particle counts: {num_particles}'
            )
        # Populate params dict.
        # Only the time is actually stored in the file.
        self.params['H0'     ] = H0
        self.params['a'      ] = self.header['time']
        self.params['boxsize'] = boxsize
        self.params['Ωm'     ] = Ωm
        self.params['ΩΛ'     ] = 1 - Ωm
        # Get file size
        with open_file(filename, mode='rb') as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
        # Create particle data types
        ndim_max = 3
        paddings = {0, 4}
        for ndim in (self.header['ndim'], ndim_max):
            particle_dtypes = {
                particle_type: np.dtype([
                    (field, f'{self.endianness}{val.fmt}'.format(ndim=ndim))
                    for field, val in fields.items()
                ])
                for particle_type, fields in self.particle_fields.items()
            }
            datasize = np.sum([
                dtype.itemsize*self.header[f'n{particle_type}']
                for particle_type, dtype in particle_dtypes.items()
            ])
            padding = filesize - (self.headersize + datasize)
            if padding in paddings:
                break
        else:
            abort('Could not determine data layout of TIPSY file')
        if ndim != 3:
            abort(
                f'The file contains {ndim}-dimensional data, '
                f'but only 3-dimensional data is allowed'
            )
        # Create component informations
        components_info = []
        components_skipped_names = []
        for name, particle_type in zip(self.component_names, self.particle_fields.keys()):
            N = self.header[f'n{particle_type}']
            if N == 0:
                continue
            # Basic component information
            representation = 'particles'
            species = determine_species(name, representation)
            # Skip this component if it should not be loaded
            if not should_load(name, species, representation):
                components_skipped_names.append(name)
                continue
            components_info.append({
                'name': name,
                'species': species,
                'N': N,
                'mass': -1,  # still unknown
            })
        # If no components are to be read, return now
        if only_params or not components_info:
            if not components_info:
                msg = ', '.join(components_skipped_names)
                masterprint(f'Skipping {msg}')
            self.components = [
                Component(
                    component_info['name'],
                    component_info['species'],
                    N=component_info['N'],
                    mass=component_info['mass'],
                )
                for component_info in components_info
            ]
            masterprint('done')
            return
        # Progress message
        msg_list = []
        for component_info in components_info:
            N = component_info['N']
            N_str = get_cubenum_strrep(N)
            plural = ('s' if N > 1 else '')
            msg_list.append(
                f'{component_info["name"]} ({N_str} {component_info["species"]}) particle{plural}'
            )
        msg = ', '.join(msg_list)
        if components_skipped_names:
            msg_list = [msg]
            msg = ', '.join(components_skipped_names)
            msg_list.append(f'(skipping {msg})')
            msg = ' '.join(msg_list)
        masterprint(f'Reading in {msg} ...')
        # Load components
        self.components.clear()
        components_info_iter = iter(components_info)
        with open_file(filename, mode='rb') as f:
            f.seek(-datasize, os.SEEK_END)
            for name, (particle_type, dtype) in zip(self.component_names, particle_dtypes.items()):
                count = self.header[f'n{particle_type}']
                if count == 0:
                    continue
                species = determine_species(name, representation)
                if not should_load(name, species, representation):
                    f.seek(count*dtype.itemsize)
                    continue
                # Only the master is used for reading in the data
                mass = -1
                if master:
                    # Read in data
                    data = np.fromfile(f, dtype=dtype, count=count)
                    # Get mass
                    masses = data['mass']
                    mass = masses[0]
                    if np.unique(masses).size > 1:
                        mass = np.mean(masses)
                        masterwarn(
                            f'Particles of component {component_info["name"]} have '
                            f'independent masses. Will use the mean particle mass.'
                        )
                    unit = 3*self.params['H0']**2/(8*π*G_Newton)*self.params['boxsize']**3
                    mass *= unit
                mass = bcast(mass)
                # Instantiate component
                component_info = next(components_info_iter)
                component_info['mass'] = mass
                component = Component(
                    component_info['name'],
                    component_info['species'],
                    N=component_info['N'],
                    mass=component_info['mass'],
                )
                self.components.append(component)
                # Only the master has read in the data
                if not master:
                    component.N_local = 0
                    continue
                # Populate data attributes
                N = component_info['N']
                component.N_local = N
                component.resize(N, only_loadable=True)
                if component.snapshot_vars['load']['pos']:
                    asarray(component.pos_mv)[:] = data['pos'].flatten()
                    pos = component.pos
                    unit = boxsize
                    for indexʳ in range(3*N):
                        pos[indexʳ] = (0.5 + pos[indexʳ])*unit
                if component.snapshot_vars['load']['mom']:
                    asarray(component.mom_mv)[:] = data['vel'].flatten()
                    mom = component.mom
                    unit = (
                        self.params['boxsize']*self.params['H0']
                        *sqrt(3/(8*π))*self.params['a']**2*mass
                    )
                    for indexʳ in range(3*N):
                        mom[indexʳ] *= unit
                # Assign particle IDs corresponding to their order
                # within the snapshot.
                if component.use_ids:
                    component.ids_mv[:] = arange(N, dtype=C2np['double'])
        # Done loading snapshot
        masterprint('done')
        masterprint('done')

    # This method populate the snapshot with component data
    # and additional parameters.
    def populate(self, components, params=None):
        if not components:
            abort(f'Cannot save a {self.name} snapshot with no components')
        if params is None:
            params = {}
        # Populated snapshot with the components
        self.components = components
        # Populate snapshot with the passed scale factor
        # and global parameters. If a params dict is passed,
        # use values from this instead.
        self.params['H0'] = params.get('H0', H0)
        if enable_Hubble:
            self.params['a'] = params.get('a', universals.a)
        else:
            self.params['a'] = universals.a
        self.params['boxsize'] = params.get('boxsize', boxsize)
        self.params['Ωm'] = params.get('Ωm', Ωm)
        ΩΛ = 1 - self.params['Ωm']  # Flat universe with only matter and cosmological constant
        self.params['ΩΛ'] = params.get('ΩΛ', ΩΛ)
        # Build the TIPSY header
        # (assign all particles to the "dark" particle type).
        self.header.clear()
        for key, val in self.header_fields.items():
            self.header[key] = deepcopy(val.default)
        self.header['time'] = self.params['a']
        self.header['nbodies'] = self.header['ndark'] = np.sum(
            [
                component.N for component in components
                if component.representation == 'particles'
            ],
            dtype=C2np['Py_ssize_t']
        )

# Function that saves the current state of the simulation
# (consisting of global parameters as well as the list
# of components) to a snapshot file.
# Note that since we want this function to be
# exposed to pure Python, a pheader is used.
@cython.pheader(
    # Argument
    one_or_more_components=object,  # Component or container of Components
    filename=str,
    params=dict,
    snapshot_type=str,
    save_all='bint',
    # Locals
    component='Component',
    components=list,
    components_selected=list,
    snapshot=object,  # Any implemented snapshot type
    returns=str,
)
def save(
    one_or_more_components, filename,
    params=None, snapshot_type=snapshot_type, save_all=False,
):
    """The type of snapshot to be saved may be given as the
    snapshot_type argument. If not given, it defaults to the value
    given by the of the snapshot_type parameter.
    Should you wish to replace the global parameters with
    something else, you may pass new parameters as the params argument.
    The components to include in the snapshot files are determined by
    the snapshot_vars component attribute. If you wish to overrule this
    and force every component to be included fully,
    set save_all to True.
    """
    if not filename:
        abort('An empty filename was passed to snapshot.save()')
    if params is None:
        params = {}
    # Filter out the components which should be saved
    if isinstance(one_or_more_components, Component):
        components = [one_or_more_components]
    else:
        components = list(one_or_more_components)
    if not components:
        abort('snapshot.save() called with no components')
    if save_all:
        components_selected = components
    else:
        components_selected = [
            component
            for component in components
            if component.snapshot_vars['save']['any']
        ]
        if not components_selected:
            msg = f't = {universals.t} {unit_time}'
            if enable_Hubble:
                msg += f', a = {universals.a}'
            abort(
                f'You have specified snapshot output at {msg}, but none '
                f'of the components present are selected for snapshot output. '
                f'Check the snapshot_select["save"] parameter.'
            )
    # Instantiate snapshot of the appropriate type
    snapshot = eval(snapshot_type.capitalize() + 'Snapshot()')
    # Populate the snapshot with data
    snapshot.populate(components_selected, params)
    # Make sure that the directory of the snapshot exists
    if master:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    Barrier()
    # Save the snapshot to disk.
    # The (maybe altered) filename is returned,
    # which should also be the return value of this function.
    return snapshot.save(filename, save_all)

# Function that loads a snapshot file.
# The type of snapshot can be any of the implemented.
# Note that since we want this function to be
# exposed to pure Python, a pheader is used.
@cython.pheader(
    # Argument
    filename=str,
    compare_params='bint',
    only_params='bint',
    only_components='bint',
    do_exchange='bint',
    compare_boxsize_on_exchange='bint',
    as_if=str,
    # Locals
    component='Component',
    input_type=str,
    snapshot=object,          # Some snapshot type
    snapshot_newtype=object,  # Some snapshot type
    returns=object,           # Snapshot or list
)
def load(
    filename,
    compare_params=True, only_params=False, only_components=False,
    do_exchange=True, compare_boxsize_on_exchange=True, as_if='',
):
    """When only_params is False and only_components is False,
    the return type is simply a snapshot object containing all the
    data in the snapshot on disk.
    When only_components is True, a list of components within
    the snapshot will be returned.
    When only_params is True, a snapshot object will be returned,
    containing both parameters (.params) and components (.components),
    just as when only_params is False. These components will have
    correctly specified attributes, but no actual component data.
    """
    # If no snapshot should be loaded, return immediately
    if not filename:
        return
    # Determine snapshot type
    input_type = get_snapshot_type(filename)
    if input_type is None:
        abort(
            f'Cannot recognise "{filename}" as one of the implemented snapshot types ({{}})'
            .format(', '.join([snapshot_class.name for snapshot_class in snapshot_classes]))
        )
    # Instantiate snapshot of the appropriate type
    snapshot = eval(input_type.capitalize() + 'Snapshot()')
    # Load the snapshot from disk
    snapshot.load(filename, only_params=only_params)
    # Populate universals_dict['species_present']
    # and universals_dict['class_species_present'].
    update_species_present(snapshot.components)
    # Check if the parameters of the snapshot matches those of the
    # current simulation run. Display a warning if they do not.
    if compare_params:
        compare_parameters(snapshot, filename)
    # Check that all particles are positioned within the box.
    # Particles exactly on the upper boundaries will be moved to the
    # physically equivalent lower boundaries.
    if not only_params:
        for component in snapshot.components:
            out_of_bounds_check(component, snapshot.params['boxsize'])
    # Scatter particles within particle components to the correct
    # domain-specific process. Note that ghost points of fluid variables
    # within fluid components are already properly communicated.
    if not only_params and do_exchange:
        # Do exchanges for all components
        for component in snapshot.components:
            if not component.snapshot_vars['load']['pos']:
                # Only components with loaded positions can be exchanged
                continue
            if not compare_params and compare_boxsize_on_exchange:
                # The exchange() function may crash if the snapshot uses
                # a boxsize different (larger) than the one currently
                # set within the program. This mismatching boxsize is
                # almost certainly not what the user wants.
                # A proper warning is already printed in the case
                # where compare_params is True. If not,
                # we print the warning now.
                compare_parameters(snapshot, filename, only_boxsize=True)
            exchange(
                component,
                component.snapshot_vars['load']['mom'],
                progress_msg=True,
            )
    # If the caller is interested in the components only,
    # return the list of components.
    if only_components:
        return snapshot.components
    # If a specific snapshot type is required, build this snapshot
    # and populate it with the loaded data.
    if as_if and as_if != input_type:
        snapshot_newtype = eval(as_if.capitalize() + 'Snapshot()')
        snapshot_newtype.populate(snapshot.components, snapshot.params)
        return snapshot_newtype
    # Return the loaded snapshot
    return snapshot

# Function for determining the snapshot type of a file
@cython.header(
    # Arguments
    filename=str,
    collective='bint',
    # Locals
    returns=str,
)
def get_snapshot_type(filename, collective=True):
    """Call the 'is_this_type' class method of each snapshot class until
    the file is recognised as a specific snapshot type.
    The returned name of the snapshot type is in the same format as the
    explicit name of the snapshot class, but with the "Snapshot" suffix
    removed and all characters are converted to lower-case.
    If the file is not recognised as any snapshot type at all,
    do not throw an error but simply return None.
    Only the master process will do the work. When collective is True,
    the result will be broadcasted, and so this function should be
    called by all processes. If called only by the master,
    set collective to False, which will leave out the broadcast.
    """
    # Return None if the file is not a valid snapshot
    determined_type = None
    # Get the snapshot type by asking each snapshot class whether they
    # recognise the file. As this is a file operation, only the master
    # does the check.
    if master:
        for snapshot_class in snapshot_classes:
            if snapshot_class.is_this_type(filename):
                determined_type = snapshot_class.__name__.removesuffix('Snapshot').lower()
                break
        if determined_type is None and not os.path.exists(filename):
            abort(f'The snapshot file "{filename}" does not exist')
    if collective:
        determined_type = bcast(determined_type)
    return determined_type

# Function which takes in a dict of parameters and compares their
# values to those of the current run. If any disagreement is found,
# a warning is emitted.
@cython.header(
    # Arguments
    snapshot=object,
    filename=str,
    only_boxsize='bint',
    # Locals
    a='double',
    component='Component',
    component_group=list,
    components=list,
    factor='double',
    factors=list,
    line_fmt=str,
    msg=str,
    msg_list=list,
    params=dict,
    rel_tol='double',
    unit='double',
    ρ_bar_background='double',
    ρ_bar_backgrounds=list,
    ρ_bar_component='double',
)
def compare_parameters(snapshot, filename, only_boxsize=False):
    params = snapshot.params
    components = snapshot.components
    # The relative tolerance by which the parameters are compared
    rel_tol = 1e-6
    # Format strings
    line_fmt = '    {{}}: {{:.{num}g}} vs {{:.{num}g}}'.format(num=int(1 - log10(rel_tol)))
    msg_list = [f'Mismatch between current parameters and those in the snapshot "{filename}":']
    # Compare parameters one by one
    if not isclose(boxsize, float(params['boxsize']), rel_tol):
        msg_list.append(
            line_fmt.format('boxsize', boxsize, params['boxsize']) + f' [{unit_length}]'
        )
    if not only_boxsize:
        if enable_Hubble and not isclose(universals.a, float(params['a']), rel_tol):
            msg_list.append(line_fmt.format('a', universals.a, params['a']))
        if not isclose(H0, float(params['H0']), rel_tol):
            unit = units.km/(units.s*units.Mpc)
            msg_list.append(
                line_fmt.format('H0', H0/unit, params['H0']/unit) + ' [km s⁻¹ Mpc⁻¹]'
            )
        if not isclose(Ωb, float(params.get('Ωb', Ωb)), rel_tol):
            msg_list.append(line_fmt.format('Ωb', Ωb, params['Ωb']))
        if not isclose(Ωcdm, float(params.get('Ωcdm', Ωcdm)), rel_tol):
            msg_list.append(line_fmt.format('Ωcdm', Ωcdm, params['Ωcdm']))
        if not isclose(Ωm, float(params.get('Ωm', Ωm)), rel_tol):
            msg_list.append(line_fmt.format('Ωm', Ωm, params['Ωm']))
        # Check if the total mass of each species within components
        # adds up to the correct value as set by the CLASS background.
        if enable_class_background:
            # One species may be distributed over several components.
            # Group components together according to their species.
            species_components = collections.defaultdict(list)
            for component in components:
                species_components[component.species].append(component)
            # Do the check for each species
            a = correct_float(params['a'])
            for component_group in species_components.values():
                factors = [a**(-3*(1 + component.w_eff(a=a))) for component in component_group]
                if len(set(factors)) > 1:
                    # Different w_eff despite same species.
                    # This is presumably caused by the user having some
                    # weird specifications. Skip check for this species.
                    continue
                ρ_bar_backgrounds = [
                    factor*component.ϱ_bar
                    for component, factor in zip(component_group, factors)
                ]
                if len(set(ρ_bar_backgrounds)) > 1:
                    # Different ρ_bar_background despite same species.
                    # This is presumably caused by the user having some
                    # weird specifications. Skip check for this species.
                    continue
                factor = factors[0]
                ρ_bar_background = ρ_bar_backgrounds[0]
                ϱ_bar_component = 0
                for component in component_group:
                    if component.representation == 'particles':
                        ϱ_bar_component += component.N*component.mass/boxsize**3
                    elif component.representation == 'fluid':
                        ϱ_bar_component += (
                            allreduce(np.sum(component.ϱ.grid_noghosts), op=MPI.SUM)
                            /component.gridsize**3
                        )
                    else:
                        continue
                ρ_bar_component = factor*ϱ_bar_component
                if not isclose(ρ_bar_background, ρ_bar_component, rel_tol):
                    msg = ', '.join([f'"{component.name}"' for component in component_group])
                    if len(component_group) > 1:
                        msg = f'{{{msg}}}'
                    msg_list.append(
                        line_fmt.format(
                            f'̅ρ(a = {a}) of {msg} (species: {component.species})',
                            ρ_bar_background,
                            ρ_bar_component,
                        ) + f' [{unit_mass} {unit_length}⁻³]'
                    )
    # Print out accumulated warning messages
    if len(msg_list) > 1:
        masterwarn('\n'.join(msg_list))

# Function which does a sanity check of particle components,
# ensuring that they are within the box.
@cython.header(
    # Arguments
    component='Component',
    snapshot_boxsize='double',
    # Locals
    indexᵖ='Py_ssize_t',
    indexʳ='Py_ssize_t',
    indexˣ='Py_ssize_t',
    pos='double*',
    value='double',
    x='double',
    y='double',
    z='double',
    returns='void',
)
def out_of_bounds_check(component, snapshot_boxsize=-1):
    """Unless the snapshot_wrap parameter is True, any particles outside
    of the box will cause the program to terminate. Particles located
    exactly at the upper box boundaries are allowed but will be moved to
    the (physically equivalent) lower boundaries.
    When snapshot_wrap is True, particles will be wrapped around the
    periodic box, ending up inside it.
    Note that no communication will be performed. Therefore, you should
    always call the exchange() function after calling this function.
    """
    # Only components with particle representation can be out of bounds,
    # as these are the only ones with explicit positions.
    if component.representation != 'particles':
        return
    # If no boxsize is passed, use the global boxsize
    if snapshot_boxsize == -1:
        snapshot_boxsize = boxsize
    pos = component.pos
    for indexʳ in range(3*component.N_local):
        value = pos[indexʳ]
        with unswitch:
            if snapshot_wrap:
                # Wrap particle around the periodic box
                pos[indexʳ] = mod(value, snapshot_boxsize)
            else:
                # Fail if any particle is not within the box.
                # Allow for but correct particle positions exactly at
                # the upper box boundaries.
                if value == snapshot_boxsize:
                    pos[indexʳ] = 0
                elif not (0 <= value < snapshot_boxsize):
                    indexᵖ = indexʳ//3
                    indexˣ = 3*indexᵖ
                    x = pos[indexˣ + 0]
                    y = pos[indexˣ + 1]
                    z = pos[indexˣ + 2]
                    abort(
                        f'Particle number {indexᵖ} of {component.name} has position '
                        f'({x}, {y}, {z}) {unit_length}, '
                        f'which is outside of the cubic box '
                        f'of side length {snapshot_boxsize} {unit_length}. '
                        f'If this is not due to erroneous snapshot data or a mismatch '
                        f'between snapshot and simulation parameters, '
                        f'you can set the snapshot_wrap parameter to True in order to '
                        f'have the particle positions wrapped around the periodic box '
                        f'upon snapshot read-in.'
                    )

# Function that either loads existing initial conditions from a snapshot
# or produces the initial conditions itself.
@cython.pheader(
    # Arguments
    initial_conditions_touse=object,  # dict, str or Nonetype
    do_realization='bint',
    # Locals
    component='Component',
    components=list,
    initial_condition_specifications=list,
    initial_conditions_list=list,
    n_components_from_snapshot='Py_ssize_t',
    name=str,
    path_or_specifications=object,  # str or dict
    species=str,
    specifications=dict,
    returns=list,
)
def get_initial_conditions(initial_conditions_touse=None, do_realization=True):
    if initial_conditions_touse is None:
        # Use initial_conditions user parameter
        initial_conditions_touse = initial_conditions
    if not initial_conditions_touse:
        return []
    # The initial_conditions_touse variable should be a list or tuple of
    # initial conditions, each of which can be a str (path to snapshot)
    # or a dict describing a component to be realised.
    # If the initial_conditions_touse variable itself is a str or dict,
    # wrap it in a list.
    if isinstance(initial_conditions_touse, (str, dict)):
        initial_conditions_list = [initial_conditions_touse]
    else:
        initial_conditions_list = list(initial_conditions_touse)
    # Now parse the list of initial conditions
    components = []
    initial_condition_specifications = []
    for path_or_specifications in initial_conditions_list:
        if isinstance(path_or_specifications, str):
            # Initial condition snapshot is given. Load it.
            components += load(sensible_path(path_or_specifications), only_components=True)
        elif isinstance(path_or_specifications, dict):
            # A component to realise is given. Remember this.
            initial_condition_specifications.append(path_or_specifications.copy())
        else:
            abort(f'Error parsing initial_conditions of type {type(path_or_specifications)}')
    n_components_from_snapshot = len(components)
    # Instantiate the component(s) given as
    # initial condition specifications.
    for specifications in initial_condition_specifications:
        species = str(specifications.pop('species', None))
        if 'name' in specifications:
            name = str(specifications.pop('name'))
        else:
            # Let the name default to the species
            name = species
        # Instantiate
        specifications = {key.replace(' ', '_'): value for key, value in specifications.items()}
        component = Component(name, species, **specifications)
        components.append(component)
    # Populate universals_dict['species_present']
    # and universals_dict['class_species_present'].
    update_species_present(components)
    # Realise all components instantiated from
    # initial condition specifications.
    if do_realization:
        for component in components[n_components_from_snapshot:]:
            component.realize()
    return components

# Function for determining the species of a component
# by looking in the select_species user parameter.
def determine_species(name, representation, only_explicit=False):
    """The species is taken from the select_species user parameter.
    If only_explicit is True, the component name must appear
    in select_species, whereas otherwise it can be matched on
    weaker grounds such as its representation and generic strings
    such as 'default'.
    """
    # The species is determined from the select_species user parameter
    # and found through the is_selected() function. In order to use this
    # function though, the component must already exist. As we do not
    # instantiate a component before knowing its species, we have
    # a problem. To tackle this we create a temporary mock
    # version of the component from the given information
    component_mock = ComponentMock(name, 'none', representation)
    select_species_copy = select_species.copy()
    if only_explicit:
        keys = ['default', 'all', 'particles', 'fluid']
        for key in keys:
            select_species_copy.pop(key, None)
    species = is_selected([component_mock], select_species_copy)
    if not species:
        species = ''
    return species

# Function for determining whether a given component
# is to be read in from snapshot or not.
def should_load(name, species, representation):
    component_mock = ComponentMock(name, species, representation)
    data_load = is_selected(
        [component_mock],
        snapshot_select['load'],
        default={},
    )
    if not data_load:
        return False
    if representation == 'particles':
        return (data_load.get('pos') or data_load.get('mom'))
    else:  # representation == 'fluid'
        return (
               data_load.get('ϱ') or data_load.get('J')
            or data_load.get('𝒫') or data_load.get('ς')
        )

# Simple mock of the Component type used by
# the determine_species() and should_load() functions.
ComponentMock = collections.namedtuple(
    'ComponentMock',
    ['name', 'species', 'representation'],
)



# Construct tuple of possible filename extensions for snapshots
# by simply grabbing the 'extension' class variable off of all
# classes defined in this module with the name '...Snapshot'.
cython.declare(
    snapshot_classes=tuple,
    snapshot_extensions=tuple,
)
snapshot_classes = tuple([
    var for name, var in globals().items()
    if (
            hasattr(var, '__module__')
        and var.__module__ == 'snapshot'
        and inspect.isclass(var)
        and name.endswith('Snapshot')
    )
])
snapshot_extensions = tuple(set([
    snapshot_class.extension
    for snapshot_class in snapshot_classes
    if snapshot_class.extension
]))

# Get local domain information
domain_info = get_domain_info()
cython.declare(
    domain_subdivisions='int[::1]',
    domain_layout_local_indices='int[::1]',
)
domain_subdivisions         = domain_info.subdivisions
domain_layout_local_indices = domain_info.layout_local_indices
