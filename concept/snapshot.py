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



# Class storing a standard snapshot. Besides holding methods for
# saving/loading, it stores component data.
@cython.cclass
class StandardSnapshot:
    """This class represents the standard snapshot type. Besides holding
    the components in the components list, the unit system is declared
    in the units dict. Finally, the cosmological parameters and the
    boxsize is stored in the params dict.
    """
    # The properly written name of this snapshot type
    # (only used for printing).
    name = 'standard'
    # The filename extension for this type of snapshot
    extension = '.hdf5'

    # Static method for identifying a file to be a snapshot of this type
    @staticmethod
    def is_this_type(filename):
        # Test for standard format by looking up the 'Ωcdm' attribute
        # in the HDF5 data structure.
        try:
            with open_hdf5(filename, mode='r') as hdf5_file:
                hdf5_file.attrs[unicode('Ωcdm')]
                return True
        except:
            pass
        return False

    # Initialisation method
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the StandardSnapshot type.
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
        masterprint(f'Saving standard snapshot "{filename}" ...')
        with open_hdf5(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
            # Save used base units
            hdf5_file.attrs['unit time']   = self.units['time']
            hdf5_file.attrs['unit length'] = self.units['length']
            hdf5_file.attrs['unit mass']   = self.units['mass']
            # Save global attributes
            hdf5_file.attrs['H0']            = correct_float(self.params['H0'])
            hdf5_file.attrs['a']             = correct_float(self.params['a'])
            hdf5_file.attrs['boxsize']       = correct_float(self.params['boxsize'])
            hdf5_file.attrs[unicode('Ωcdm')] = correct_float(self.params['Ωcdm'])
            hdf5_file.attrs[unicode('Ωb')]   = correct_float(self.params['Ωb'])
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
                    masterprint(
                        f'Writing out {component.name} '
                        f'({N_str} {component.species} particles) ...'
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
                        f'Boltzmann order {component.boltzmann_order}) ...'
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

    # Method for loading in a standard snapshot from disk
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
        indexʳ='Py_ssize_t',
        indexˣ='Py_ssize_t',
        indexˣ_file='Py_ssize_t',
        mass='double',
        mom='double*',
        multi_index=tuple,
        name=str,
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
            self.params['Ωcdm']    = hdf5_file.attrs[unicode('Ωcdm')]
            self.params['Ωb']      = hdf5_file.attrs[unicode('Ωb')]
            # Load component data
            for name, component_h5 in hdf5_file['components'].items():
                species = component_h5.attrs['species']
                if 'N' in component_h5.attrs:
                    representation = 'particles'
                elif 'gridsize' in component_h5.attrs:
                    representation = 'fluid'
                else:
                    abort(
                        f'Could not determine representation of {name} '
                        f'in snapshot "{filename}", as neither N nor gridsize is specified'
                    )
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
                    masterprint(f'Reading in {name} ({N_str} {species} particles) ...')
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
                            # Load in using chunks. Large chunks are
                            # fine as no temporary buffer is used.
                            # The maximum possible chunk size
                            # is limited by MPI, though.
                            chunk_size = pairmin(N_local, 2**30//8//3)  # max a GB of doubles
                            for indexˣ in range(0, N_local, chunk_size):
                                if indexˣ + chunk_size > N_local:
                                    chunk_size = N_local - indexˣ
                                indexˣ_file = start_local + indexˣ
                                dset.read_direct(
                                    arr,
                                    source_sel=np.s_[indexˣ_file:(indexˣ_file + chunk_size), :],
                                    dest_sel=np.s_[indexˣ:(indexˣ + chunk_size), :],
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
                        f'Boltzmann order {boltzmann_order}) ...'
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
                            chunk_size = pairmin(ℤ[slab.shape[0]], ℤ[2**30//8//gridsize**2])  # max a GB of doubles
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
    @cython.pheader(# Arguments
                    components=list,
                    params=dict,
                    )
    def populate(self, components, params=None):
        if params is None:
            params = {}
        # Populated snapshot with the components
        self.components = components
        # Populate snapshot with the passed scalefactor
        # and global parameters. If a params dict is passed,
        # use values from this instead.
        self.params['H0']      = params.get('H0',      H0)
        if enable_Hubble:
            self.params['a']   = params.get('a',       universals.a)
        else:
            self.params['a']   = universals.a
        self.params['boxsize'] = params.get('boxsize', boxsize)
        self.params['Ωcdm']    = params.get('Ωcdm'   , Ωcdm)
        self.params['Ωb']      = params.get('Ωb'     , Ωb)
        # Populate the base units with the global base units
        self.units['time']   = unit_time
        self.units['length'] = unit_length
        self.units['mass']   = unit_mass

# Class storing a GADGET-2 snapshot. Besides holding methods for
# saving/loading, it stores particle data (positions, momenta, mass)
# and also GADGET-2 ID's and the GADGET-2 header.
@cython.cclass
class Gadget2Snapshot:
    """This class represents snapshots of the "GADGET-2" type, meaning
    the second type of snapshot native to GADGET-2. Only GADGET-2 type 1
    (halo) particles, corresponding to cold dark matter particles, are
    supported. It is possible to save a particle component with any species
    as Gadget2Snapshot. When loading a Gadget2Snapshot, a component of
    species "cold dark matter" is produced.
    As is the case for the standard snapshot class, this class contains
    a list components (the components attribute) and dict of parameters
    (the params attribute). Besides holding the cosmological parameters
    and the boxsize, the params dict also contains a "header" key, the
    item of which is the GADGET-2 header, represented as an ordered dict.
    This class does not have a units attribute, as no global unit system
    is used by GADGET-2 snapshots.
    As only a single component (GADGET-2 halos) are supported, the
    components list will always contain this single component only. For
    ease of access, the component attribute is also defined, referring
    directly to this component. Finally, the ID attribute holds the
    GADGET-2 ID's of particles. When constructing a Gadget2Snapshot
    instance by other means than by loading from a snapshot on disk,
    these are generated in a somewhat arbitrary (but consistent)
    fashion.
    """
    # The properly written name of this snapshot type
    # (only used for printing).
    name = 'GADGET-2'
    # The filename extension for this type of snapshot
    extension = ''
    # Sizes of low level types and the total header, in bytes
    sizes = {'I': 4, 'i': 4, 'f': 4, 'd': 8, 's': 1, 'header': 256}

    # Static method for identifying a file to be a snapshot of this type
    @staticmethod
    def is_this_type(filename):
        # Test for GADGET-2 format by checking the existence
        # of the 'HEAD' identifier.
        try:
            with open(filename, 'rb') as f:
                f.seek(4)
                head = struct.unpack('4s', f.read(struct.calcsize('4s')))
                if head[0] == b'HEAD':
                    return True
        except:
            pass
        return False

    # Initialisation method
    @cython.header
    def __init__(self):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Gadget2Snapshot type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public dict params
        public list components
        Component component
        unsigned int[::1] ID
        """
        # Dict containing all the parameters of the snapshot
        self.params = {}
        # List of Component instances (will only ever hold
        # self.component, which can only be GADGET-2 halos).
        self.components = []
        # The actual component data
        self.component = None
        # The ID of each particle (not used by the CO𝘕CEPT code)
        self.ID = None

    # Method for saving a GADGET-2 snapshot of type 2 to disk
    @cython.pheader(
        # Arguments
        filename=str,
        # Locals
        N='Py_ssize_t',
        N_lin='double',
        N_local='Py_ssize_t',
        N_str=str,
        block=dict,
        block_name=str,
        blocks=dict,
        boxsize_gadget='float',
        component='Component',
        chunk='double[::1]',
        chunk_ptr='double*',
        chunk_singleprec='float[::1]',
        chunk_singleprec_ptr='float*',
        chunk_size='Py_ssize_t',
        data=object,  # np.ndarray
        data_mv='double[::1]',
        index_chunk='Py_ssize_t',
        indexʳ='Py_ssize_t',
        itemsize='int',
        ndim='int',
        rank_writer='int',
        sizes=dict,
        unit='double',
        returns=str,
    )
    def save(self, filename):
        """The snapshot data (positions and velocities) are stored in
        single-precision. Only GADGET-2 type 1 (halo) particles,
        corresponding to cold dark matter particles, are supported.
        """
        masterprint(f'Saving GADGET-2 snapshot "{filename}" ...')
        component = self.component
        if component.representation != 'particles':
            abort(
                f'The GAGDET-2 snapshot type can only store particles, '
                f'but {component.name} is a {component.representation} component.'
            )
        N, N_local = component.N, component.N_local
        header = self.params['header']
        # The master process write the HEAD block
        sizes = self.sizes
        if master:
            with open(filename, 'wb') as f:
                f.write(struct.pack('I', 4*sizes['s'] + sizes['I']))
                f.write(struct.pack('4s', b'HEAD'))
                f.write(struct.pack('I', sizes['I'] + sizes['header'] + sizes['I']))
                f.write(struct.pack('I', sizes['I'] + 4*sizes['s']))
                f.write(struct.pack('I', sizes['header']))
                f.write(struct.pack('6I', *             (header['Npart'        ])))
                f.write(struct.pack('6d', *correct_float(header['Massarr'      ])))
                f.write(struct.pack('d',   correct_float(header['Time'         ])))
                f.write(struct.pack('d',   correct_float(header['Redshift'     ])))
                f.write(struct.pack('i',                (header['FlagSfr'      ])))
                f.write(struct.pack('i',                (header['FlagFeedback' ])))
                f.write(struct.pack('6i', *             (header['Nall'         ])))
                f.write(struct.pack('i',                (header['FlagCooling'  ])))
                f.write(struct.pack('i',                (header['Numfiles'     ])))
                f.write(struct.pack('d',   correct_float(header['BoxSize'      ])))
                f.write(struct.pack('d',   correct_float(header['Omega0'       ])))
                f.write(struct.pack('d',   correct_float(header['OmegaLambda'  ])))
                f.write(struct.pack('d',   correct_float(header['HubbleParam'  ])))
                f.write(struct.pack('i',                (header['FlagAge'      ])))
                f.write(struct.pack('i',                (header['FlagMetals'   ])))
                f.write(struct.pack('6i', *             (header['NallHW'       ])))
                f.write(struct.pack('i',                (header['flag_entr_ics'])))
                # Padding to fill out the 256 bytes
                f.write(struct.pack('60s', b' '*60))
                f.write(struct.pack('I', sizes['header']))
        # Write out the position, velocity and ID blocks
        N_lin = cbrt(N)
        if N > 1 and isint(N_lin):
            N_str = str(int(round(N_lin))) + '³'
        else:
            N_str = str(N)
        masterprint(f'Writing out {component.name} ({N_str} {component.species} particles) ...')
        blocks = {
            'POS': {
                'data': component.pos_mv3,
                # Comoving coordinates in kpc/h
                'unit': units.kpc/header['HubbleParam'],
            },
            'VEL': {
                'data': component.mom_mv3,
                # Peculiar velocities u=a*dx/dt
                # divided by sqrt(a), in km/s.
                'unit': units.km/units.s*component.mass*header['Time']**1.5,
            },
            'ID': {'data': self.ID},
        }
        chunk_size = pairmin(ℤ[3*N_local], 2**20)
        chunk_singleprec = empty(chunk_size, dtype=C2np['float'])
        chunk_singleprec_ptr = cython.address(chunk_singleprec[:])
        boxsize_gadget = boxsize/blocks['POS']['unit']
        for block_name, block in blocks.items():
            data = asarray(block['data'])
            unit = block.get('unit', 1)
            # Size in bytes of each element
            itemsize = data.itemsize
            if block_name in {'POS', 'VEL'}:
                itemsize = asarray(chunk_singleprec).itemsize
            if itemsize != 4:
                masterwarn(
                    f'Expected "{block_name}" elements to be of size 4 '
                    f'but they are of size {itemsize}'
                )
            # Get dimensionality of data and then flatten it
            ndim = 1
            if data.ndim == 2:
                ndim = 3
            data = data.ravel()[:ndim*N_local]
            # Write the block in serial, one process at a time
            for rank_writer in range(nprocs):
                Barrier()
                if rank != rank_writer:
                    continue
                with open(filename, 'ab') as f:
                    # The identifier
                    if rank_writer == 0:
                        f.write(struct.pack('I', 4*sizes['s'] + sizes['I']))
                        f.write(struct.pack('4s', block_name.ljust(4).encode('ascii')))
                        f.write(struct.pack('I', 2*sizes['I'] + ndim*N*itemsize))
                        f.write(struct.pack('I', sizes['I'] + 4*sizes['s']))
                        f.write(struct.pack('I', ndim*N*itemsize))
                    # The data
                    if block_name in {'POS', 'VEL'}:
                        data_mv = data
                        # Write positions and velocities in chunks
                        chunk_size = pairmin(ℤ[3*N_local], 2**20)
                        for indexʳ in range(0, ℤ[3*N_local], chunk_size):
                            if indexʳ + chunk_size > ℤ[3*N_local]:
                                chunk_size = ℤ[3*N_local] - indexʳ
                            chunk = data_mv[indexʳ:(indexʳ + chunk_size)]
                            chunk_ptr = cython.address(chunk[:])
                            # Copy chunk into single-precision chunk
                            # while applying unit conversion.
                            for index_chunk in range(chunk_size):
                                chunk_singleprec_ptr[index_chunk] = (
                                    chunk_ptr[index_chunk]*ℝ[1/unit]
                                )
                                # In the case of positions,
                                # safeguard against round-off errors.
                                with unswitch(2):
                                    if block_name == 'POS':
                                        if chunk_singleprec_ptr[index_chunk] >= boxsize_gadget:
                                            chunk_singleprec_ptr[index_chunk] -= boxsize_gadget
                            # Write out chunk
                            asarray(chunk_singleprec[:chunk_size]).tofile(f)
                    elif block_name == 'ID':
                        data.tofile(f)
                    else:
                        abort(f'Does not know how to write GADGET block "{block_name}"')
                    # The closing int
                    if rank_writer == nprocs - 1:
                        f.write(struct.pack('I', ndim*N*itemsize))
        # Finalize progress messages
        masterprint('done')
        masterprint('done')
        # Return the filename of the saved file
        return filename

    # Method for loading in a GADGET-2 snapshot of type 2 from disk
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
        blocks=dict,
        blocks_read=set,
        chunk='double[::1]',
        chunk_ptr='double*',
        chunk_singleprec='float[::1]',
        chunk_singleprec_ptr='float*',
        chunk_size='Py_ssize_t',
        component='Component',
        data=object,  # np.ndarray
        data_mv='double[::1]',
        eof='bint',
        header=dict,
        index_chunk='Py_ssize_t',
        indexʳ='Py_ssize_t',
        mass='double',
        name=str,
        offset='Py_ssize_t',
        rank_reader='int',
        size='unsigned int',
        size_expected='unsigned int',
        sizes=dict,
        species=str,
        start_local='Py_ssize_t',
        unit='double',
    )
    def load(self, filename, only_params=False):
        """It is assumed that the snapshot on the disk is a GADGET-2
        snapshot of type 2 and that it uses single-precision. The
        Gadget2Snapshot instance stores the data (positions and
        velocities) in double-precision. Only GADGET-2 type 1 (halo)
        particles, corresponding to cold dark matter particles,
        are supported.
        """
        if only_params:
            masterprint(f'Loading parameters of snapshot "{filename}" ...')
        else:
            masterprint(f'Loading snapshot "{filename}" ...')
        # Only type 1 (halo) particles are supported. Since GADGET-2
        # wants Ωm = Ωcdm + Ωb (what GADGET-2 calls Omega0) to be
        # accounted for fully by the particles, we should make the
        # species of the particles 'matter', as using 'cold dark matter'
        # would suggest that the baryons are missing.
        name = 'GADGET-2 halos'
        species = 'matter'
        # Read in the snapshot
        offset = 0
        with open(filename, 'rb') as f:
            # Read the HEAD block into a params['header'] dict.
            # All processes read this, but in turn as to not overload
            # the file system.
            # No unit conversion will be done.
            for rank_reader in range(nprocs):
                Barrier()
                if rank != rank_reader:
                    continue
                offset = self.new_block(f, offset)
                block_name = self.read(f, '4s').decode('utf8').rstrip()
                if block_name != 'HEAD':
                    abort(f'Expected block "HEAD" but found "{block_name}"')
                size = self.read(f, 'I')
                sizes = self.sizes
                size_expected = sizes['I'] + sizes['header'] + sizes['I']
                if size != size_expected:
                    masterwarn(f'The "HEAD" block has size {size} but expected {size_expected}')
                offset = self.new_block(f, offset)
                header = {}
                self.params['header'] = header
                header['Npart']         = self.read(f, '6I')
                header['Massarr']       = self.read(f, '6d')
                header['Time']          = self.read(f, 'd')
                header['Redshift']      = self.read(f, 'd')
                header['FlagSfr']       = self.read(f, 'i')
                header['FlagFeedback']  = self.read(f, 'i')
                header['Nall']          = self.read(f, '6i')
                header['FlagCooling']   = self.read(f, 'i')
                header['NumFiles']      = self.read(f, 'i')
                header['BoxSize']       = self.read(f, 'd')
                header['Omega0']        = self.read(f, 'd')
                header['OmegaLambda']   = self.read(f, 'd')
                header['HubbleParam']   = self.read(f, 'd')
                header['FlagAge']       = self.read(f, 'i')
                header['FlagMetals']    = self.read(f, 'i')
                header['NallHW']        = self.read(f, '6i')
                header['flag_entr_ics'] = self.read(f, 'i')
            # Also include some of the header fields as parameters
            # directly in the params dict. These are the same as
            # those included in the params dict of
            # standard type snapshots.
            unit = 100*units.km/(units.s*units.Mpc)
            self.params['H0']      = header['HubbleParam']*unit
            self.params['a']       = header['Time']
            unit = units.kpc/header['HubbleParam']
            self.params['boxsize'] = header['BoxSize']*unit
            self.params['Ωm']      = header['Omega0']
            self.params['ΩΛ']      = header['OmegaLambda']
            # Construct a Component instance and pack it
            # into this snapshot's list of components.
            N = header['Npart'][1]
            unit = 1e+10*units.m_sun/header['HubbleParam']
            mass = header['Massarr'][1]*unit
            component = Component(name, species, N=N, mass=mass)
            self.component = component
            self.components = [component]
            # Done loading component attributes
            if only_params:
                masterprint('done')
                return
            N_lin = cbrt(N)
            if N > 1 and isint(N_lin):
                N_str = str(int(round(N_lin))) + '³'
            else:
                N_str = str(N)
            masterprint(f'Reading in {name} ({N_str} {species} particles) ...')
            # Compute a fair distribution
            # of component data to the processes.
            start_local, N_local = partition(N)
            component.N_local = N_local
            if component.N_allocated < N_local:
                component.resize(N_local)
            # Read in blocks. We again do this one process at a time.
            blocks_read = {'HEAD'}
            blocks = {
                'POS': {
                    'data': component.pos_mv,
                    # Comoving coordinates in kpc/h
                    'unit': units.kpc/header['HubbleParam'],
                },
                'VEL': {
                    'data': component.mom_mv,
                    # Peculiar velocities u=a*dx/dt
                    # divided by sqrt(a), in km/s.
                    'unit': units.km/units.s*mass*header['Time']**1.5,
                },
                'ID': {'data': self.ID},
            }
            eof = False
            while not eof:
                for rank_reader in range(nprocs):
                    Barrier()
                    if rank != rank_reader:
                        continue
                    if eof:
                        continue
                    offset = self.new_block(f, offset)
                    if offset == -1:
                        # End of file
                        eof = True
                        continue
                    block_name = self.read(f, '4s').decode('utf8').rstrip()
                    if block_name in blocks_read:
                        continue
                    size = self.read(f, 'I')
                    offset = self.new_block(f, offset)
                    block = blocks.get(block_name, {})
                    data = asarray(block.get('data', ()))
                    unit = block.get('unit', 1)
                    if block_name in {'POS', 'VEL'}:
                        blocks_read.add(block_name)
                        f.seek(3*sizes['f']*start_local, 1)
                        data_mv = data
                        chunk_size = pairmin(ℤ[3*N_local], 2**20)
                        for indexʳ in range(0, ℤ[3*N_local], chunk_size):
                            if indexʳ + chunk_size > ℤ[3*N_local]:
                                chunk_size = ℤ[3*N_local] - indexʳ
                            chunk = data_mv[indexʳ:(indexʳ + chunk_size)]
                            chunk_ptr = cython.address(chunk[:])
                            # Read in single-precision chunk
                            chunk_singleprec = np.fromfile(
                                f,
                                dtype=C2np['float'],
                                count=chunk_size,
                            )
                            chunk_singleprec_ptr = cython.address(chunk_singleprec[:])
                            # Copy single-precision chunk into chunk
                            # while applying unit conversion.
                            for index_chunk in range(chunk_size):
                                chunk_ptr[index_chunk] = chunk_singleprec_ptr[index_chunk]*unit
                                # In the case of positions,
                                # safeguard against round-off errors.
                                with unswitch(2):
                                    if block_name == 'POS':
                                        if chunk_ptr[index_chunk] >= boxsize:
                                            chunk_ptr[index_chunk] -= boxsize
                    elif block_name == 'ID':
                        blocks_read.add(block_name)
                        f.seek(sizes['I']*start_local, 1)
                        self.ID = np.fromfile(f, dtype=C2np['unsigned int'], count=N_local)
                    else:
                        masterprint(f'Skipping block {block_name}')
                        continue
            # Done reading in particles
            if 'POS' not in blocks_read:
                masterwarn('No POS block found')
            if 'VEL' not in blocks_read:
                masterwarn('No VEL block found')
            masterprint('done')
        # Done loading the snapshot
        masterprint('done')

    # This method populate the snapshot with component data
    # as well as ID's (which are not used by this code) and
    # additional header information.
    @cython.pheader(
        # Arguments
        components=list,
        params=dict,
        # Locals
        component='Component',
        start_local='Py_ssize_t',
        ΩΛ='double',
    )
    def populate(self, components, params=None):
        """The following header fields depend on the particles:
            Npart, Massarr, Nall.
        The following header fields depend on the current time:
            Time, Redshift.
        The following header fields correspond to the parameters
        used in the current run:
            BoxSize, Omega0, OmegaLambda, HubbleParam.
        All other fields get generic values.
        """
        if params is None:
            params = {}
        # Populate snapshot with the GADGET halos
        if len(components) > 1:
            abort(
                f'The GAGDET2 snapshot type can only store a single component, '
                f'but you are trying to populate a single such snapshot with',
                [component.name for component in components],
            )
        component = components[0]
        self.component = component
        self.components = [component]
        # The ID's of the local particles, generated such that
        # the process with the lowest rank has the lowest ID's.
        start_local = int(np.sum(smart_mpi(component.N_local, mpifun='allgather')[:rank]))
        self.ID = arange(start_local, start_local + component.N_local, dtype=C2np['unsigned int'])
        # Populate snapshot with the passed scalefactor
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
        # Build the GADGET-2 header
        self.update_header()

    # Method for constructing the GADGET-2 header from the other
    # parameters in the params dict.
    @cython.header(
        # Locals
        component='Component',
        h='double',
        header=dict,
        params=dict,
        unit='double',
    )
    def update_header(self):
        # Extract variables
        component = self.component
        params = self.params
        # The GADGET-2 header is constructed from scratch
        params['header'] = {}
        header = params['header']
        # Fill the header
        header['Npart'] = [0, component.N, 0, 0, 0, 0]
        unit = 100*units.km/(units.s*units.Mpc)
        h = params['H0']/unit
        unit = 1e+10*units.m_sun/h
        header['Massarr']       = [0.0, component.mass/unit, 0.0, 0.0, 0.0, 0.0]
        header['Time']          = params['a']
        header['Redshift']      = 1/params['a'] - 1
        header['FlagSfr']       = 0
        header['FlagFeedback']  = 0
        header['Nall']          = [0, component.N, 0, 0, 0, 0]
        header['FlagCooling']   = 0
        header['Numfiles']      = 1
        unit = units.kpc/h
        header['BoxSize']       = params['boxsize']/unit
        header['Omega0']        = params['Ωm']
        header['OmegaLambda']   = params['ΩΛ']
        header['HubbleParam']   = h
        header['FlagAge']       = 0
        header['FlagMetals']    = 0
        header['NallHW']        = [0, 0, 0, 0, 0, 0]
        header['flag_entr_ics'] = 1

    # Method used for reading series of bytes from the snapshot file
    @cython.header(
        # Arguments
        f=object,
        fmt=str,
        # Locals
        payload=object,  # bytes
        t=tuple,
        returns=object,
    )
    def read(self, f, fmt):
        # Convert bytes to Python objects and store them in a tuple
        payload = f.read(struct.calcsize(fmt))
        if not payload:
            return None
        t = struct.unpack(fmt, payload)
        # If the tuple contains just a single element, return this
        # element rather than the tuple.
        if len(t) == 1:
            return t[0]
        # It is nicer to use mutable lists than immutable tuples
        return list(t)

    # Method that handles the file object's position in the snapshot
    # file during loading. Call it when the next block should be read.
    @cython.header(
        # Arguments
        f=object,
        offset='Py_ssize_t',
        # Locals
        size=object,  # unsigned int or NoneType,
        returns='Py_ssize_t',
    )
    def new_block(self, f, offset):
        # Set the current position in the file
        f.seek(offset)
        # Each block is bracketed with an unsigned int
        # containing the size of the block.
        size = self.read(f, 'I')
        if size is None:
            return -1
        offset += self.sizes['I'] + size + self.sizes['I']
        return offset

# Function that saves the current state of the simulation
# - consisting of global parameters as well as the list of components -
# to a snapshot file. Note that since we want this function to be
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
def save(one_or_more_components, filename, params=None, snapshot_type=snapshot_type,
         save_all_components=False):
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
    if save_all_components:
        components_selected = components
    else:
        components_selected = [
            component for component in components if is_selected(component, snapshot_select)
        ]
        if not components_selected:
            abort(
                'You have specified snapshot(s) to be dumped, but none of the components present '
                'are selected for snapshot output. Check the snapshot_select parameter.'
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
    if master and input_type is None:
        abort(
            'Cannot recognise "{}" as one of the implemented snapshot types ({})'
            .format(
                filename,
                ', '.join([snapshot_class.name for snapshot_class in snapshot_classes]),
            )
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
        compare_parameters(snapshot.params, filename)
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
        if not os.path.isfile(filename):
            abort(f'The snapshot file "{filename}" does not exist')
        for snapshot_class in snapshot_classes:
            if snapshot_class.is_this_type(filename):
                determined_type = rstrip_exact(snapshot_class.__name__, 'Snapshot').lower()
                break
    return bcast(determined_type)

# Function which takes in a dict of parameters and compare their
# values to those of the current run. If any disagreement is found,
# write a warning message.
@cython.header(# Arguments
               params=dict,
               filename=str,
               # Locals
               indent_str=str,
               msg=str,
               rel_tol='double',
               unit='double',
               vs=str,
               )
def compare_parameters(params, filename):
    """Specifically, the following parameters are compared:
    a (compared against a_begin)
    boxsize
    H0
    Ωcdm
    Ωb
    """
    # The relative tolerance by which the parameters are compared
    rel_tol = 1e-6
    # Format strings
    vs = '{{:.{num}g}} vs {{:.{num}g}}'.format(num=int(1 - log10(rel_tol)))
    indent_str = '\n    '
    msg = ''
    # Do the comparisons one by one
    if enable_Hubble and not isclose(a_begin, float(params['a']), rel_tol):
        msg += '{}a_begin: {}'.format(indent_str, vs.format(a_begin, params['a']))
    if not isclose(boxsize, float(params['boxsize']), rel_tol):
        msg += '{}boxsize: {} [{}]'.format(
            indent_str, vs.format(boxsize, params['boxsize']), unit_length,
        )
    if not isclose(H0, float(params['H0']), rel_tol):
        unit = units.km/(units.s*units.Mpc)
        msg += '{}H0: {} [km s⁻¹ Mpc⁻¹]'.format(indent_str, vs.format(H0/unit, params['H0']/unit))
    if 'Ωb' in params:
        if not isclose(Ωb, float(params['Ωb']), rel_tol):
            msg += '{}Ωb: {}'.format(indent_str, vs.format(Ωb, params['Ωb']))
    if 'Ωcdm' in params:
        if not isclose(Ωcdm, float(params['Ωcdm']), rel_tol):
            msg += '{}Ωcdm: {}'.format(indent_str, vs.format(Ωcdm, params['Ωcdm']))
    if 'Ωm' in params:
        if not isclose(Ωm, float(params['Ωm']), rel_tol):
            msg += '{}Ωm: {}'.format(indent_str, vs.format(Ωm, params['Ωm']))
    if msg:
        msg = f'Mismatch between current parameters and those in the snapshot "{filename}":{msg}'
        masterwarn(msg, skipline=False)

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
)
def out_of_bounds_check(component, snapshot_boxsize=-1):
    """If any particles are outside of the box, the program
    will terminate. Particles located exactly at the upper box
    boundaries will be moved to the (physically equivalent) lower
    boundaries. Note that no communication will be performed! Therefore,
    you should always call the exchange function after this function.
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
                f'of side length {snapshot_boxsize} {unit_length}'
            )

# Function that either loads existing initial conditions from a snapshot
# or produces the initial conditions itself.
@cython.header(
    # Arguments
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
def get_initial_conditions(do_realization=True):
    if not initial_conditions:
        return []
    # The initial_conditions parameter should be a list or tuple of
    # initial conditions, each of which can be a str (path to snapshot)
    # or a dict describing a component to be realised.
    # If the initial_conditions parameter itself is a str or dict,
    # wrap it in a list.
    if isinstance(initial_conditions, (str, dict)):
        initial_conditions_list = [initial_conditions]
    else:
        initial_conditions_list = list(initial_conditions)
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
