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
cimport('from communication import partition,                   '
        '                          domain_layout_local_indices, '
        '                          domain_subdivisions,         '
        '                          exchange,                    '
        '                          smart_mpi,                   '
        )
cimport('from mesh import domain_decompose, get_fftw_slab, slab_decompose')
cimport('from species import Component, FluidScalar, update_species_present')

# Pure Python imports
import struct



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
        except:
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
        # Locals
        N='Py_ssize_t',
        N_lin='double',
        N_local='Py_ssize_t',
        N_str=str,
        component='Component',
        end_local='Py_ssize_t',
        fluidscalar='FluidScalar',
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
    def save(self, filename):
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
                    N_lin = cbrt(N)
                    if N > 1 and isint(N_lin):
                        N_str = str(int(round(N_lin))) + '³'
                    else:
                        N_str = str(N)
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
                    pos_h5 = component_h5.create_dataset('pos', (N, 3), dtype=C2np['double'])
                    mom_h5 = component_h5.create_dataset('mom', (N, 3), dtype=C2np['double'])
                    pos_h5[start_local:end_local, :] = component.pos_mv3[:N_local, :]
                    mom_h5[start_local:end_local, :] = component.mom_mv3[:N_local, :]
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
                        fluidvar_h5 = component_h5.create_group('fluidvar_{}'.format(index))
                        for multi_index in fluidvar.multi_indices:
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
                            ] = slab[:, :, :component.gridsize]
                    # Create additional names (hard links) for the fluid
                    # groups and data sets. The names from
                    # component.fluid_names will be used, except for
                    # the additional linear variable, if CLASS is used
                    # to close the Boltzmann hierarchy
                    # (hence the try/except).
                    for name, indices in component.fluid_names.items():
                        if not isinstance(name, str) or name == 'ordered':
                            continue
                        if isinstance(indices, int):
                            # "name" is a fluid variable name (e.g. J,
                            # though not ϱ as this is a fluid scalar).
                            try:
                                fluidvar_h5 = component_h5['fluidvar_{}'.format(indices)]
                                component_h5[name] = fluidvar_h5
                            except:
                                pass
                        else:  # indices is a tuple
                            # "name" is a fluid scalar name (e.g. ϱ, Jx)
                            index, multi_index = indices
                            try:
                                fluidvar_h5 = component_h5['fluidvar_{}'.format(index)]
                                fluidscalar_h5 = fluidvar_h5['fluidscalar_{}'.format(multi_index)]
                                component_h5[name] = fluidscalar_h5
                            except:
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
        N_lin='double',
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
        # Load all components
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
                #  Load the component
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
                    N_lin = cbrt(N)
                    if N > 1 and isint(N_lin):
                        N_str = str(int(round(N_lin))) + '³'
                    else:
                        N_str = str(N)
                    plural = ('s' if N > 1 else '')
                    masterprint(f'Reading in {name} ({N_str} {species}) particle{plural} ...')
                    # Extract HDF5 datasets
                    pos_h5 = component_h5['pos']
                    mom_h5 = component_h5['mom']
                    # Compute a fair distribution of
                    # particle data to the processes.
                    start_local, N_local = partition(N)
                    # Make sure that the particle data arrays
                    # have the correct size.
                    component.N_local = N_local
                    component.resize(N_local)
                    # Read particle data directly into
                    # the particle data arrays.
                    if N_local > 0:
                        for dset, arr in [
                            (pos_h5, asarray(component.pos_mv3)),
                            (mom_h5, asarray(component.mom_mv3)),
                        ]:
                            # Load in using chunks
                            chunk_size = np.min((N_local, ℤ[self.chunk_size_max//8//3]))
                            for indexᵖ in range(0, N_local, chunk_size):
                                if indexᵖ + chunk_size > N_local:
                                    chunk_size = N_local - indexᵖ
                                indexᵖ_file = start_local + indexᵖ
                                dset.read_direct(
                                    arr,
                                    source_sel=np.s_[indexᵖ_file:(indexᵖ_file + chunk_size), :],
                                    dest_sel=np.s_[indexᵖ:(indexᵖ + chunk_size), :],
                                )
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
                    component.resize((domain_size_i, domain_size_j, domain_size_k))
                    # Fluid scalars are already instantiated.
                    # Now populate them.
                    for index, fluidvar in enumerate(
                        component.fluidvars[:component.boltzmann_order + 1]
                    ):
                        fluidvar_h5 = component_h5[f'fluidvar_{index}']
                        for multi_index in fluidvar.multi_indices:
                            fluidscalar_h5 = fluidvar_h5[f'fluidscalar_{multi_index}']
                            slab = get_fftw_slab(gridsize)
                            slab_start = slab.shape[0]*rank
                            # Load in using chunks. Large chunks are
                            # fine as no temporary buffer is used. The
                            # maximum possible chunk size is limited
                            # by MPI, though.
                            chunk_size = np.min((
                                ℤ[slab.shape[0]],
                                ℤ[self.chunk_size_max//8//gridsize**2],
                            ))
                            if chunk_size == 0:
                                masterwarn('The input seems surprisingly large and may not be read in correctly')
                                chunk_size = 1
                            arr = asarray(slab)
                            for index_i in range(0, ℤ[slab.shape[0]], chunk_size):
                                if index_i + chunk_size > ℤ[slab.shape[0]]:
                                    chunk_size = ℤ[slab.shape[0]] - index_i
                                index_i_file = slab_start + index_i
                                fluidscalar_h5.read_direct(
                                    arr,
                                    source_sel=np.s_[index_i_file:(index_i_file + chunk_size), :, :],
                                    dest_sel=np.s_[index_i:(index_i + chunk_size), :, :gridsize],
                                )
                            # Communicate the slabs directly to the
                            # domain decomposed fluid grids.
                            domain_decompose(slab, component.fluidvars[index][multi_index].grid_mv)
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
    # Low level types and their sizes
    fmts = {
        's': 'signed char',
        'i': 'int',
        'I': 'unsigned int',
        'Q': 'unsigned long long int',
        'f': 'float',
        'd': 'double',
    }
    sizes = {fmt: struct.calcsize(fmt) for fmt in fmts}
    # Block name format, head block name and size
    block_name_fmt = '4s'
    block_name_header = 'HEAD'
    sizes['header'] = 2**8
    # The maximum number of particles within a single GADGET
    # snapshot file is limited by the largest number representable
    # by an int. As we never deal with negative particle numbers, we use
    # unsigned ints in this implementation. In e.g. the GADGET-2 code
    # however, a signed int is used, and so we subtract the sign bit in
    # the calculation below, cutting the maximum number of particles per
    # snapshot roughly in half, compared to what is really needed.
    num_particles_file_max = (
        ((2**(sizes['i']*8 - 1) - 1) - 2*sizes['I'])
        //(
            3*(
                np.max((
                    gadget_snapshot_params['dataformat']['POS'],
                    gadget_snapshot_params['dataformat']['VEL'],
                ))//8
            )
        )
    )

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
                f.seek(cls.sizes['I'])
                if cls.block_name_header == struct.unpack(
                    cls.block_name_fmt,
                    f.read(struct.calcsize(cls.block_name_fmt)),
                )[0].decode('utf8').rstrip():
                    return 2
        except:
            pass
        # Test for SnapFormat 1 by checking the size
        # of the header block.
        try:
            with open_file(filename, mode='rb') as f:
                if cls.sizes['header'] == struct.unpack('I', f.read(cls.sizes['I']))[0]:
                    return 1
        except:
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
        Py_ssize_t current_block_size
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
        # Size of the current block in bytes, when writing
        self.current_block_size = -1
        # Check on low level type sizes
        for fmt, size in self.sizes.items():
            size_expected = {'s': 1, 'i': 4, 'I': 4, 'Q': 8, 'f': 4, 'd': 8}.get(fmt)
            if size_expected is not None and size != size_expected:
                masterwarn(
                    f'Expected C type \'{fmt}\' to be {size_expected} bytes large, '
                    f'but it is {size}'
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
        # Locals
        N='Py_ssize_t',
        N_lin='double',
        N_str=str,
        block=dict,
        block_fmt=str,
        block_name=str,
        block_size='Py_ssize_t',
        block_type=str,
        blocks=dict,
        boxsize_gadget_doubleprec='double',
        boxsize_gadget_singleprec='float',
        chunk='double[::1]',
        chunk_doubleprec='double[::1]',
        chunk_doubleprec_ptr='double*',
        chunk_ptr='double*',
        chunk_singleprec='float[::1]',
        chunk_singleprec_ptr='float*',
        chunk_size='Py_ssize_t',
        chunk_size_max_needed='Py_ssize_t',
        component='Component',
        data='double[::1]',
        data_components=list,
        data_value_doubleprec='double',
        data_value_singleprec='float',
        doubleprec_needed='bint',
        filename_i=str,
        i='Py_ssize_t',
        id_counter='Py_ssize_t',
        id_counters='Py_ssize_t[::1]',
        indexᵖ='Py_ssize_t',
        indexᵖ_bgn='Py_ssize_t',
        indexᵖ_end='Py_ssize_t',
        indexʳ='Py_ssize_t',
        j='Py_ssize_t',
        msg=str,
        msg_list=list,
        num_files='Py_ssize_t',
        num_particle_file_tot='Py_ssize_t',
        num_particles_file_tot='Py_ssize_t[::1]',
        num_write_file=list,
        num_write_files=list,
        num_write_max='Py_ssize_t',
        num_write='Py_ssize_t',
        plural=str,
        rank_writer='int',
        singleprec_needed='bint',
        size_write='Py_ssize_t',
        unit='double',
        unit_components=list,
        returns=str,
    )
    def save(self, filename):
        # Set the GADGET SnapFormat based on user parameters
        self.snapformat = gadget_snapshot_params['snapformat']
        # Divvy up the particles between the files and processes
        num_write_files = self.divvy()
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
        # Progress messages
        msg = filename
        if num_files > 1:
            msg += f'/{output_bases["snapshot"]}.*'
        masterprint(f'Saving {self.name} snapshot "{msg}" ...')
        msg_list = []
        for component in self.components:
            N = component.N
            N_lin = cbrt(N)
            if N > 1 and isint(N_lin):
                N_str = str(int(round(N_lin))) + '³'
            else:
                N_str = str(N)
            plural = ('s' if N > 1 else '')
            msg_list.append(f'{component.name} ({N_str} {component.species}) particle{plural}')
        msg = ', '.join(msg_list)
        masterprint(f'Writing out {msg} ...')
        # Get information about the blocks to be written out
        blocks = self.get_blocks_info('save')
        # The boxsize in GADGET units, used for safeguarding against
        # out-of-bounds particles after converting to GADGET units.
        boxsize_gadget_singleprec = C2np['float' ](boxsize/blocks['POS']['unit'][0])
        boxsize_gadget_doubleprec = C2np['double'](boxsize/blocks['POS']['unit'][0])
        # Instantiate chunk buffers for particle data
        num_write_max = 0
        for num_write in itertools.chain(*num_write_files):
            if num_write > num_write_max:
                num_write_max = num_write
        chunk_size_max_needed = np.min((3*num_write_max, ℤ[self.chunk_size_max//8]))
        singleprec_needed = any([block['type'].endswith('f') for block in blocks.values()])
        doubleprec_needed = any([block['type'].endswith('d') for block in blocks.values()])
        if singleprec_needed:
            chunk_singleprec = empty(chunk_size_max_needed, dtype=C2np['float'])
            chunk_singleprec_ptr = cython.address(chunk_singleprec[:])
        if doubleprec_needed:
            chunk_doubleprec = empty(chunk_size_max_needed, dtype=C2np['double'])
            chunk_doubleprec_ptr = cython.address(chunk_doubleprec[:])
        # Counters keeping track of the unique particles ID's of each
        # particle type. The particle ID's are generated consecutively,
        # with all particles of a given type receiving ID's following
        # each other with no gaps.
        id_counters = zeros(len(self.components), dtype=C2np['Py_ssize_t'])
        id_counter = 0
        for j in range(1, len(self.components)):
            id_counter += self.components[j - 1].N
            id_counters[j] = id_counter
        # Write out each file in turn
        for i, num_write_file in enumerate(num_write_files):
            if num_files == 1:
                filename_i = filename
            else:
                masterprint(f'Writing snapshot file {i}/{num_files - 1} ...')
                filename_i = f'{filename}/{output_bases["snapshot"]}.{i}'
            # Initialise the file with the HEAD block
            self.write_header(filename_i, num_write_file)
            # The number of particles of each type to be written
            # to this file by all processes.
            num_particles_file_tot = reduce(
                asarray(num_write_file, dtype=C2np['Py_ssize_t']),
                op=MPI.SUM,
            )
            # Write out the data blocks
            for block_name, block in blocks.items():
                data_components = block.get('data')
                unit_components = block.get('unit')
                block_type = block['type']
                block_fmt = block_type[len(block_type) - 1]
                # Begin block
                if master:
                    block_size = np.sum(num_particles_file_tot)*struct.calcsize(block_type)
                    self.write_block_bgn(filename_i, block_size, block_name)
                # Write out the block contents
                if block_name in {'POS', 'VEL'}:
                    # Iterate over each component
                    for j, (num_write, data, unit) in enumerate(
                        zip(num_write_file, data_components, unit_components)
                    ):
                        size_write = 3*num_write
                        chunk_size = np.min((size_write, ℤ[self.chunk_size_max//8]))
                        # Write the block in serial,
                        # one process at a time.
                        for rank_writer in range(nprocs):
                            Barrier()
                            if rank != rank_writer:
                                continue
                            if num_write == 0:
                                continue
                            # Write out data in chunks
                            with open_file(filename_i, mode='ab') as f:
                                for indexʳ in range(0, size_write, chunk_size):
                                    if indexʳ + chunk_size > size_write:
                                        chunk_size = size_write - indexʳ
                                    chunk = data[indexʳ:(indexʳ + chunk_size)]
                                    chunk_ptr = cython.address(chunk[:])
                                    # Copy chunk while applying unit
                                    # conversion, then write this copy
                                    # to the file. For positions,
                                    # safeguard against
                                    # round-off errors.
                                    if 𝔹[block_fmt == 'f']:
                                        for index_chunk in range(chunk_size):
                                            data_value_singleprec = chunk_ptr[index_chunk]*ℝ[1/unit]
                                            with unswitch(4):
                                                if block_name == 'POS':
                                                    if data_value_singleprec >= boxsize_gadget_singleprec:
                                                        data_value_singleprec -= boxsize_gadget_singleprec
                                            chunk_singleprec_ptr[index_chunk] = data_value_singleprec
                                        asarray(chunk_singleprec[:chunk_size]).tofile(f)
                                    elif 𝔹[block_fmt == 'd']:
                                        for index_chunk in range(chunk_size):
                                            data_value_doubleprec = chunk_ptr[index_chunk]*ℝ[1/unit]
                                            with unswitch(4):
                                                if block_name == 'POS':
                                                    if data_value_doubleprec >= boxsize_gadget_doubleprec:
                                                        data_value_doubleprec -= boxsize_gadget_doubleprec
                                            chunk_doubleprec_ptr[index_chunk] = data_value_doubleprec
                                        asarray(chunk_doubleprec[:chunk_size]).tofile(f)
                                    else:
                                        abort(
                                            f'Block format "{block_fmt}" not implemented '
                                            f'for block "{block_name}"'
                                        )
                            # Crop the now written data
                            # away from the memory view.
                            data_components[j] = data[size_write:]
                elif block_name == 'ID':
                    # We generate the particles ID's on the fly.
                    # As these do not correspond to actual data,
                    # they can be handled by a single process.
                    if master:
                        for j, num_particle_file_tot in enumerate(num_particles_file_tot):
                            if num_particle_file_tot == 0:
                                continue
                            # Get and update ID counters
                            id_counter = id_counters[j]
                            id_counters[j] += num_particle_file_tot
                            # Generate and write the ID's
                            chunk_size = np.min((num_particle_file_tot, ℤ[self.chunk_size_max//8]))
                            indexᵖ_bgn = id_counter
                            indexᵖ_end = id_counter + num_particle_file_tot
                            with open_file(filename_i, mode='ab') as f:
                                for indexᵖ in range(indexᵖ_bgn, indexᵖ_end, chunk_size):
                                    if indexᵖ + chunk_size > indexᵖ_end:
                                        chunk_size = indexᵖ_end - indexᵖ
                                    arange(
                                        indexᵖ,
                                        indexᵖ + chunk_size,
                                        dtype=𝕆[C2np[self.fmts[block_fmt]]],
                                    ).tofile(f)
                else:
                    abort(f'Does not know how to write {self.name} block "{block_name}"')
                # End block
                Barrier()
                self.write_block_end(filename_i)
            # Done saving this snapshot file
            if num_files > 1:
                masterprint('done')
        # Finalise progress messages
        masterprint('done')
        masterprint('done')
        # Return the filename of the saved snapshot. In case of multiple
        # snapshot files having been written for this single snapshot,
        # this will be a directory.
        return filename

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
                # Determine whether to use 32 or 64 bit unsigned
                # integers for the ID's.
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
                if self.sizes[fmt] == num_bytes:
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
    def write_header(self, filename, num_write_file):
        """Though the majority of the work is carried out by the master
        process only, this method needs to be called by all processes.
        On the master process, the return value is the number of
        particles to be written to this file.
        """
        # Get the total number of particles of each type in this
        # snapshot file, after which only the master continues.
        num_particles_file_tot = reduce(
            asarray(num_write_file, dtype=C2np['Py_ssize_t']),
            op=MPI.SUM,
        )
        if not master:
            return
        # Create header for this particular file
        num_particles_header = [0]*self.num_particle_types
        indices = [self.get_component_index(component) for component in self.components]
        for index, num_particles_file in zip(indices, num_particles_file_tot):
            num_particles_header[index] = num_particles_file
        header = self.header.copy()
        header['Npart'] = num_particles_header
        # Initialize file with HEAD block
        with open_file(filename, mode='wb') as f:
            # Start the HEAD block
            self.write_block_bgn(f, self.sizes['header'], self.block_name_header)
            # Write out header, tallying up its size
            size = 0
            for key, val in self.header_fields.items():
                size += self.write(f, val.fmt, header[key])
            if size > self.current_block_size:
                abort(
                    f'The "{self.block_name_header}" block took up {size} bytes '
                    f'but was specified to {self.current_block_size}'
                )
            # Pad the header with zeros to fill out its specified size
            size_padding = self.current_block_size - size
            self.write(f, 'b', [0]*size_padding)
            # Close the HEAD block
            self.write_block_end(f)

    # Method for initialising a block on disk
    def write_block_bgn(self, f, block_size, block_name):
        """Though safe to call by all processes, the work is carried out
        by the master process only. The passed f may be either a
        file name or a file object to an already opened file.
        """
        if not master:
            return
        block_name_length = int(self.block_name_fmt.rstrip('s'))
        if len(block_name) > block_name_length:
            abort(f'Block name "{block_name}" larger than {block_name_length} characters')
        # Closure for doing the actual writing
        def writeout(f):
            # The initial block meta data
            if self.snapformat == 2:
                self.write(f, 'I', block_name_length*self.sizes['s'] + self.sizes['I'])
                self.write(f, 's', block_name.ljust(block_name_length).encode('ascii'))
                self.write(f, 'I', self.sizes['I'] + block_size + self.sizes['I'])
                self.write(f, 'I', self.sizes['I'] + block_name_length*self.sizes['s'])
            self.write(f, 'I', block_size)
        # Call writeout() in accordance with the supplied f
        if isinstance(f, str):
            filename = f
            with open_file(filename, mode='ab') as f:
                writeout(f)
        else:
            writeout(f)
        # Store block size for use with write_block_end()
        self.current_block_size = block_size

    # Method for finalising a block on disk
    def write_block_end(self, f):
        """Though safe to call by all processes, the work is carried out
        by the master process only. The passed f may be either a
        file name or a file object to an already opened file.
        """
        if not master:
            return
        if self.current_block_size == -1:
            abort(
                f'write_block_end() was called though '
                f'it seems that write_block_bgn() was never called'
            )
        # Closure for doing the actual writing
        def writeout(f):
            # The closing int
            self.write(f, 'I', self.current_block_size)
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
        N_lin='double',
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
        indexʳ='Py_ssize_t',
        i='Py_ssize_t',
        j='Py_ssize_t',
        j_populated=list,
        key=str,
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
        num_particle_files=list,
        num_read='Py_ssize_t',
        num_read_file=list,
        num_read_files=list,
        offset='Py_ssize_t',
        offset_header='Py_ssize_t',
        offset_nextblock='Py_ssize_t',
        plural=str,
        representation=str,
        size_read='Py_ssize_t',
        species=object,  # str or None
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
            msg = ', '.join([name for name in components_skipped_names])
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
            N_lin = cbrt(N)
            if N > 1 and isint(N_lin):
                N_str = str(int(round(N_lin))) + '³'
            else:
                N_str = str(N)
            plural = ('s' if N > 1 else '')
            msg_list.append(f'{component.name} ({N_str} {component.species}) particle{plural}')
        msg = ', '.join(msg_list)
        if components_skipped_names:
            msg_list = [msg]
            msg = ', '.join(components_skipped_names)
            msg_list.append(f'(skipping {msg})')
            msg = ' '.join(msg_list)
        masterprint(f'Reading in {msg} ...')
        # Enlarge components in order to accommodate particles
        for component, N_local in zip(self.components, num_local):
            if component is None:
                continue
            component.N_local = N_local
            if component.N_allocated < N_local:
                component.resize(N_local)
        # Get information about the blocks to be read in
        blocks = self.get_blocks_info('load')
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
                        if blocks_required:
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
                            # Crop the populated part of the data
                            # away from the memory view.
                            data_components[j] = data[size_read:]
                        # Update offset, to be used by the next process
                        offset += size_read*dtype().itemsize
                        # Inform the next process about the file offset
                        if rank < nprocs - 1 or (nprocs > 1 and j < len(self.components) - 1):
                            send(offset, dest=mod(rank + 1, nprocs))
                else:
                    abort(f'Does not know how to read {self.name} block "{block_name}"')
                # Let all the processes catch up,
                # ensuring that the file is closed.
                Barrier()
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
            size_expected = self.sizes['header']
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
                offset += self.sizes['I'] + size + self.sizes['I']
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
            size_bare_2 = size - 2*self.sizes['I']
            if size_bare != size_bare_2:
                # The two sizes do not agree.
                # Pick one according to the 'settle'
                # gadget_snapshot_params parameter.
                msg = (
                    f'Size of block "{block_name}" not consistent: '
                    f'{size} - {2*self.sizes["I"]} = {size_bare_2} ≠ {size_bare}. '
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
    save_all_components='bint',
    # Locals
    component='Component',
    components=list,
    components_selected=list,
    snapshot=object,  # Any implemented snapshot type
    returns=str,
)
def save(
    one_or_more_components, filename,
    params=None, snapshot_type=snapshot_type, save_all_components=False,
):
    """The type of snapshot to be saved may be given as the
    snapshot_type argument. If not given, it defaults to the value
    given by the of the snapshot_type parameter.
    Should you wish to replace the global parameters with
    something else, you may pass new parameters as the params argument.
    The components to include in the snapshot files are determined by
    the snapshot_select user parameter. If you wish to overrule this
    and force every component to be included,
    set save_all_components to True.
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
    if save_all_components:
        components_selected = components
    else:
        components_selected = [
            component
            for component in components
            if is_selected(component, snapshot_select['save'])
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
    return snapshot.save(filename)

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
    do_exchange=True, as_if='',
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
    # Scatter particles to the correct domain-specific process.
    # Also communicate ghost points of fluid variables.
    if not only_params and do_exchange:
        # Do exchanges for all components
        for component in snapshot.components:
            exchange(component, progress_msg=True)
        # Communicate the ghost points of all fluid variables
        # in fluid components.
        for component in snapshot.components:
            component.communicate_fluid_grids('=')
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
@cython.header(filename=str, returns=str)
def get_snapshot_type(filename):
    """Call the 'is_this_type' class method of each snapshot class until
    the file is recognised as a specific snapshot type.
    The returned name of the snapshot type is in the same format as the
    explicit name of the snapshot class, but with the "Snapshot" suffix
    removed and all characters are converted to lower-case.
    If the file is not recognised as any snapshot type at all,
    do not throw an error but simply return None.
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
    return bcast(determined_type)

# Function which takes in a dict of parameters and compares their
# values to those of the current run. If any disagreement is found,
# a warning is emitted.
@cython.header(
    # Arguments
    snapshot=object,
    filename=str,
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
def compare_parameters(snapshot, filename):
    params = snapshot.params
    components = snapshot.components
    # The relative tolerance by which the parameters are compared
    rel_tol = 1e-6
    # Format strings
    line_fmt = '    {{}}: {{:.{num}g}} vs {{:.{num}g}}'.format(num=int(1 - log10(rel_tol)))
    msg_list = [f'Mismatch between current parameters and those in the snapshot "{filename}":']
    # Compare parameters one by one
    if enable_Hubble and not isclose(universals.a, float(params['a']), rel_tol):
        msg_list.append(line_fmt.format('a', universals.a, params['a']))
    if not isclose(boxsize, float(params['boxsize']), rel_tol):
        msg_list.append(line_fmt.format('boxsize', boxsize, params['boxsize']) + f' [{unit_length}]')
    if not isclose(H0, float(params['H0']), rel_tol):
        unit = units.km/(units.s*units.Mpc)
        msg_list.append(line_fmt.format('H0', H0/unit, params['H0']/unit) + ' [km s⁻¹ Mpc⁻¹]')
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
            abort(f'Error parsing initial_conditions of type {type(path_or_dict)}')
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
    return is_selected(
        [component_mock],
        snapshot_select['load'],
        default=False,
    )
# Simple mock of the Component type used by
# the determine_species() and should_load() functions.
ComponentMock = collections.namedtuple(
    'CompnentMock',
    ['name', 'species', 'representation'],
)



# Construct tuple of possible filename extensions for snapshots
# by simply grabbing the 'extension' class variable off of all
# classes defined in this module with the name '...Snapshot'.
cython.declare(snapshot_classes=tuple,
               snapshot_extensions=tuple,
               )
snapshot_classes = tuple([var for name, var in globals().items()
                          if (    hasattr(var, '__module__')
                              and var.__module__ == 'snapshot'
                              and inspect.isclass(var)
                              and name.endswith('Snapshot')
                              )
                          ])
snapshot_extensions = tuple(set([snapshot_class.extension for snapshot_class in snapshot_classes
                                 if snapshot_class.extension]))
