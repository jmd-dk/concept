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
    '    communicate_ghosts,   '
    '    exchange,             '
    '    smart_mpi,            '
    '    rung_indices_arr,     '
)
cimport(
    'from fluid import                     '
    '    kurganov_tadmor,                  '
    '    kurganov_tadmor_internal_sources, '
    '    maccormack,                       '
    '    maccormack_internal_sources,      '
)
cimport('from ic import realize')
cimport(
    'from integration import '
    '    Spline,             '
    '    cosmic_time,        '
    '    scale_factor,       '
    '    ȧ,                  '
)
cimport(
    'from linear import      '
    '    compute_cosmo,      '
    '    species_canonical,  '
    '    species_registered, '
)

# Pure Python imports
from communication import get_domain_info



# Class which serves as the data structure for fluid variables,
# efficiently and elegantly implementing symmetric
# multi-dimensional arrays. The actual fluid (scalar) grids are then
# the elements of these tensors.
class Tensor:
    """With respect to indexing and looping, you may threat instances of
    this class just like NumPy arrays.
    """
    def __init__(self, component, varnum, shape, symmetric=False, active=True):
        """If disguised_scalar is True, every element of the tensor
        will always point to the same object.
        """
        # Empty and otherwise falsy shapes are understood as rank 0
        # tensors (scalars). For this reason,
        # empty tensors are not representable.
        if not shape:
            shape = 1
        # Store initialisation arguments as instance variables
        self.component = component
        self.varnum = varnum
        self.shape = tuple(any2list(shape))
        self.symmetric = symmetric
        self.active = active
        # Store other array-like attributes
        self.size = np.prod(self.shape)
        self.ndim = len(self.shape)
        self.dtype = 'object'
        # Is this tensor really just a scalar in disguise?
        self.disguised_scalar = (self.component.boltzmann_order + 1 == self.varnum)
        # Should this fluid variable do realisations when iterating
        # with the iterate method?
        self.iterative_realizations = (
            self.disguised_scalar and self.component.boltzmann_closure == 'class'
        )
        # Only "square" tensors can be symmetric
        if self.symmetric and len(set(self.shape)) != 1:
            abort('A {} tensor cannot be made symmetric'.format('×'.join(self.shape)))
        # Compute and store all multi_indices
        self.multi_indices = tuple(sorted(set([self.process_multi_index(multi_index)
                                               for multi_index
                                               in itertools.product(*[range(size)
                                                                      for size in self.shape])])))
        # Initialise tensor data
        self.data = {multi_index: None for multi_index in self.multi_indices}
        # Additional (possible) elements
        self.additional_dofs = ('trace', )
        for additional_dof in self.additional_dofs:
            self.data[additional_dof] = None

    # Method for processing multi_indices.
    # This is where the symmetry of the indices is implemented.
    def process_multi_index(self, multi_index):
        # The case of an extra degree of freedom
        if isinstance(multi_index, str):
            return multi_index.lower()
        if (    isinstance(multi_index, tuple)
            and len(multi_index) == 1
            and isinstance(multi_index[0], str)):
            return multi_index[0].lower()
        # The normal case
        if self.symmetric:
            multi_index = tuple(sorted(any2list(multi_index), reverse=True))
        else:
            multi_index = tuple(any2list(multi_index))
        if len(multi_index) != self.ndim:
            # The tensor has been indexed with a non-existing index
            raise KeyError(f'An attempt was made to index {self} using {multi_index}')
        return multi_index

    # Methods for indexing and membership testing
    def __getitem__(self, multi_index):
        multi_index = self.process_multi_index(multi_index)
        return self.data[multi_index]
    def __setitem__(self, multi_index, value):
        multi_index = self.process_multi_index(multi_index)
        if multi_index not in self.data:
            abort('Attempted to access multi_index {} of {} tensor'
                  .format(multi_index, '×'.join(self.shape))
                  )
        self.data[multi_index] = value
        # If only a single element should exist in memory,
        # point every multi_index to the newly set element.
        if self.disguised_scalar and multi_index not in self.additional_dofs:
            for other_multi_index in self.multi_indices:
                self.data[other_multi_index] = value
    def __contains__(self, multi_index):
        try:
           multi_index = self.process_multi_index(multi_index)
        except (IndexError, KeyError):
            return False
        return multi_index in self.data

    # Iteration
    def __iter__(self):
        """By default, iteration yields all elements, except for
        the case of disguised scalars, where only the first element
        is yielded.
        Elements corresponding to additional degrees of freedom
        are not included.
        Inactive tensors do not yield anything.
        """
        if self.active:
            N_elements = 1 if self.disguised_scalar else len(self.multi_indices)
        else:
            N_elements = 0
        return (self.data[multi_index] for multi_index in self.multi_indices[:N_elements])
    def iterate(self, *attributes, multi_indices=False, a_next=-1):
        """This generator yields all normal elements of the tensor
        (that is, the additional degrees of freedom are not included).
        For disguised scalars, all logical elements are realised
        before they are yielded.
        What attribute(s) of the elements (fluidscalars) should be
        yielded is controlled by the attributes argument. For a value of
        'fluidscalar', the entire fluidscalar is returned. This is also
        the default behaviour when no attributes are passed.
        If multi_index is True, both the multi_index and the fluidscalar
        will be yielded, in that order.
        """
        if not attributes:
            attributes = ('fluidscalar', )
        for multi_index in self.multi_indices:
            with unswitch:
                if self.iterative_realizations:
                    self.component.realize_if_linear(self.varnum, multi_index, a_next=a_next)
            fluidscalar = self.data[multi_index]
            values = []
            for attribute in attributes:
                if attribute == 'fluidscalar':
                    values.append(fluidscalar)
                else:
                    values.append(getattr(fluidscalar, attribute))
            with unswitch:
                if multi_indices:
                    yield (multi_index, *values)
                else:
                    if len(values) == 1:
                        yield values[0]
                    else:
                        yield tuple(values)

    # String representation
    def __repr__(self):
        return f'<Tensor({self.component}, {self.varnum}, {self.shape}>'
    def __str__(self):
        return self.__repr__()



# Class which serves as the data structure for a scalar fluid grid
# (each component of a fluid variable is stored as a collection of
# scalar grids). Each scalar fluid has its own name, e.g.
# ϱ    (varnum == 0, multi_index == 0),
# J[0] (varnum == 1, multi_index == 0),
# J[1] (varnum == 1, multi_index == 1),
# J[2] (varnum == 1, multi_index == 2),
@cython.cclass
class FluidScalar:
    # Initialisation method
    @cython.header(
        # Arguments
        varnum='int',
        multi_index=object,  # tuple or int-like
        is_linear='bint',
    )
    def __init__(self, varnum, multi_index=(), is_linear=False):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the FluidScalar type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        # Fluid variable number and index of fluid scalar
        public int varnum
        public tuple multi_index
        # The is_linear flag
        public bint is_linear
        # Meta data
        public tuple shape
        public tuple shape_noghosts
        public Py_ssize_t size
        public Py_ssize_t size_noghosts
        # The data itself
        double* grid
        public double[:, :, ::1] grid_mv
        public double[:, :, :] grid_noghosts
        # The starred buffer
        double* gridˣ
        public double[:, :, ::1] gridˣ_mv
        public double[:, :, :] gridˣ_noghosts
        # The Δ buffer
        double* Δ
        public double[:, :, ::1] Δ_mv
        public double[:, :, :] Δ_noghosts
        """
        # Number and index of fluid variable
        self.varnum = varnum
        self.multi_index = tuple(any2list(multi_index))
        # Flag indicating whether this FluidScalar is linear or not.
        # For a linear FluidScalar, the starred and unstarred grids
        # point to the same memory.
        self.is_linear = is_linear
        # Minimal starting layout
        self.shape = (1, 1, 1)
        self.shape_noghosts = (1, 1, 1)
        self.size = np.prod(self.shape)
        self.size_noghosts = np.prod(self.shape_noghosts)
        # The data itself
        self.grid = malloc(self.size*sizeof('double'))
        self.grid_mv = cast(self.grid, 'double[:self.shape[0], :self.shape[1], :self.shape[2]]')
        self.grid_noghosts = self.grid_mv[:, :, :]
        # The starred buffer
        if self.is_linear:
            self.gridˣ          = self.grid
            self.gridˣ_mv       = self.grid_mv
            self.gridˣ_noghosts = self.grid_noghosts
        else:
            self.gridˣ = malloc(self.size*sizeof('double'))
            self.gridˣ_mv = cast(
                self.gridˣ, 'double[:self.shape[0], :self.shape[1], :self.shape[2]]',
            )
            self.gridˣ_noghosts = self.gridˣ_mv[:, :, :]
        # Due to the Unicode NFKC normalization done by pure Python,
        # attributes with a ˣ in their name need to be set in following
        # way in order for dynamical lookup to function.
        if not cython.compiled:
            setattr(self, 'gridˣ'         , self.gridˣ         )
            setattr(self, 'gridˣ_mv'      , self.gridˣ_mv      )
            setattr(self, 'gridˣ_noghosts', self.gridˣ_noghosts)
        # The Δ buffer
        self.Δ = malloc(self.size*sizeof('double'))
        self.Δ_mv = cast(self.Δ, 'double[:self.shape[0], :self.shape[1], :self.shape[2]]')
        self.Δ_noghosts = self.Δ_mv[:, :, :]

    # Method for resizing all grids of this scalar fluid
    @cython.header(shape_noghosts=tuple)
    def resize(self, shape_noghosts):
        """After resizing the fluid scalar,
        all fluid elements will be nullified.
        """
        # The full shape and size of the grid with ghost points
        self.shape = tuple([s + 2*nghosts for s in shape_noghosts])
        self.size = np.prod(self.shape)
        # The shape and size of the grid with no ghost points
        self.shape_noghosts = shape_noghosts
        self.size_noghosts = np.prod(self.shape_noghosts)
        # The data itself
        self.grid = realloc(self.grid, self.size*sizeof('double'))
        self.grid_mv = cast(self.grid, 'double[:self.shape[0], :self.shape[1], :self.shape[2]]')
        self.grid_noghosts = self.grid_mv[
            nghosts:(self.grid_mv.shape[0] - nghosts),
            nghosts:(self.grid_mv.shape[1] - nghosts),
            nghosts:(self.grid_mv.shape[2] - nghosts),
        ]
        # Nullify the newly allocated data grid
        self.nullify_grid()
        # The starred buffer
        if self.is_linear:
            self.gridˣ          = self.grid
            self.gridˣ_mv       = self.grid_mv
            self.gridˣ_noghosts = self.grid_noghosts
        else:
            self.gridˣ = realloc(self.gridˣ, self.size*sizeof('double'))
            self.gridˣ_mv = cast(
                self.gridˣ, 'double[:self.shape[0], :self.shape[1], :self.shape[2]]',
            )
            self.gridˣ_noghosts = self.gridˣ_mv[
                nghosts:(self.gridˣ_mv.shape[0] - nghosts),
                nghosts:(self.gridˣ_mv.shape[1] - nghosts),
                nghosts:(self.gridˣ_mv.shape[2] - nghosts),
            ]
        # Due to the Unicode NFKC normalization done by pure Python,
        # attributes with a ˣ in their name need to be set in following
        # way in order for dynamical lookup to function.
        if not cython.compiled:
            setattr(self, 'gridˣ'         , self.gridˣ         )
            setattr(self, 'gridˣ_mv'      , self.gridˣ_mv      )
            setattr(self, 'gridˣ_noghosts', self.gridˣ_noghosts)
        # Nullify the newly allocated starred buffer
        if not self.is_linear:
            self.nullify_gridˣ()
        # The Δ buffer
        self.Δ = realloc(self.Δ, self.size*sizeof('double'))
        self.Δ_mv = cast(self.Δ, 'double[:self.shape[0], :self.shape[1], :self.shape[2]]')
        self.Δ_noghosts = self.Δ_mv[
            nghosts:(self.Δ_mv.shape[0] - nghosts),
            nghosts:(self.Δ_mv.shape[1] - nghosts),
            nghosts:(self.Δ_mv.shape[2] - nghosts),
        ]
        # Nullify the newly allocated Δ buffer
        self.nullify_Δ()

    # Method for scaling the data grid
    @cython.pheader(
        # Argument
        a='double',
        # Locals
        index='Py_ssize_t',
        grid='double*',
    )
    def scale_grid(self, a):
        # Extract data pointer
        grid = self.grid
        # Scale data buffer
        for index in range(self.size):
            grid[index] *= a

    # Method for nullifying the data grid
    @cython.pheader()
    def nullify_grid(self):
        self.grid_mv[...] = 0

    # Method for nullifying the starred grid
    @cython.pheader()
    def nullify_gridˣ(self):
        self.gridˣ_mv[...] = 0

    # Method for nullifying the Δ buffer
    @cython.pheader()
    def nullify_Δ(self):
        self.Δ_mv[...] = 0

    # Method for copying the content of grid into gridˣ
    @cython.pheader(
        # Arguments
        operation=str,
        # Locals
        index='Py_ssize_t',
        grid='double*',
        gridˣ='double*',
    )
    def copy_grid_to_gridˣ(self, operation='='):
        grid, gridˣ = self.grid, self.gridˣ
        for index in range(self.size):
            with unswitch:
                if operation == '=':
                    gridˣ[index] = grid[index]
                else:  # operation == '+='
                    gridˣ[index] += grid[index]

    # Method for copying the content of gridˣ into grid
    @cython.pheader(
        # Arguments
        operation=str,
        # Locals
        index='Py_ssize_t',
        grid='double*',
        gridˣ='double*',
    )
    def copy_gridˣ_to_grid(self, operation='='):
        grid, gridˣ = self.grid, self.gridˣ
        for index in range(self.size):
            with unswitch:
                if operation == '=':
                    grid[index] = gridˣ[index]
                else:  # operation == '+='
                    grid[index] += gridˣ[index]

    # This method is automatically called when a FluidScalar instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        free(self.grid)
        if not self.is_linear:
            free(self.gridˣ)
        free(self.Δ)

    # String representation
    def __repr__(self):
        return '<fluidscalar {}[{}]>'.format(self.varnum,
                                             ', '.join([str(mi) for mi in self.multi_index]))
    def __str__(self):
        return self.__repr__()



# Class providing the data structure for particle tiling.
# Though the variable naming only refer to tiles (within a domain),
# subtiles (within a tile) makes use of this data structure as well.
@cython.cclass
class Tiling:
    # Initialisation method
    @cython.header(
        # Arguments
        tiling_name=str,
        component='Component',
        shape=object,  # sequence of length 3 or int-like
        extent='double[::1]',
        initial_rung_size=object,  # sequence of length N_rungs or int-like
        refinement_period='Py_ssize_t',
        # Locals
        dim='int',
        i='Py_ssize_t',
        initial_rung_sizes='Py_ssize_t[::1]',
        j='Py_ssize_t',
        k='Py_ssize_t',
        rung='Py_ssize_t*',
        rung_index='signed char',
        rungs_N='Py_ssize_t*',
        rungs_N_mv='Py_ssize_t[::1]',
        rungs_sizes='Py_ssize_t*',
        rungs_sizes_mv='Py_ssize_t[::1]',
        tile='Py_ssize_t**',
        tile_index='Py_ssize_t',
        tile_index3D='Py_ssize_t[::1]',
        tiles_rungs_N_mv='Py_ssize_t[::1]',
        tiles_rungs_sizes_mv='Py_ssize_t[::1]',
    )
    def __init__(self, tiling_name, component, shape, extent,
        initial_rung_size, refinement_period):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Tiling type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public str            name
        bint                  is_trivial
        Component             component
        Py_ssize_t[::1]       shape
        Py_ssize_t            size
        Py_ssize_t[:, :, ::1] layout
        Py_ssize_t[:, ::1]    layout_1Dto3D
        Py_ssize_t***         tiles
        Py_ssize_t*           tiles_rungs_N_data
        Py_ssize_t**          tiles_rungs_N
        Py_ssize_t*           tiles_rungs_sizes_data
        Py_ssize_t**          tiles_rungs_sizes
        signed char*          contain_particles
        double[::1]           location
        double[::1]           extent
        double[::1]           tile_extent
        public Py_ssize_t     refinement_period
        public Py_ssize_t     refinement_offset
        double                computation_time
        double                computation_time_total
        """
        # Remember the name of this tiling
        self.name = tiling_name
        # The tiling with the name 'trivial' is special.
        # Note whether this tiling is the trivial tiling.
        self.is_trivial = (self.name == 'trivial')
        # A separate Tiling instance should be used for each component
        self.component = component
        # The shape of this tiling.
        # If a single number is provided, a cubic tiling is created.
        shape = any2list(shape)
        if len(shape) == 1:
            shape *= 3
        elif len(shape) != 3:
            abort(f'Tilings need a 3D shape, but shape = {shape} is given')
        shape = asarray(shape, dtype=C2np['Py_ssize_t'])
        if np.any(shape < 1):
            abort(
                f'Tilings must have a size of at least 1 in each dimension, '
                f'but a tiling of shape {shape} is about to be created.'
            )
        self.shape = shape
        self.size = np.prod(self.shape)
        # The tiling layout; mapping from 3D integer tile indices
        # to 1D integer tile indices.
        self.layout = arange(self.size, dtype=C2np['Py_ssize_t']
            ).reshape(tuple(self.shape))
        # Mapping from 1D tile indices to 3D tile indices.
        # This is implemented as a 2D array.
        self.layout_1Dto3D = empty((self.size, 3), dtype=C2np['Py_ssize_t'])
        for         i in range(self.shape[0]):
            for     j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    tile_index = self.layout[i, j, k]
                    tile_index3D = self.layout_1Dto3D[tile_index, :]
                    tile_index3D[0] = i
                    tile_index3D[1] = j
                    tile_index3D[2] = k
        # Create array of initial rung sizes. If the passed
        # initial_rung_size is a single number, this will be used for
        # all rungs.
        initial_rung_size = any2list(initial_rung_size)
        if len(initial_rung_size) == 1:
            initial_rung_size *= N_rungs
        elif len(initial_rung_size) != N_rungs:
            abort(
                f'A Tiling with initial rung sizes of {initial_rung_size} is about '
                f'to be initialised, but we need one size for each of the {N_rungs} rungs.'
            )
        initial_rung_sizes = asarray(initial_rung_size, dtype=C2np['Py_ssize_t'])
        # The tiles themselves. This is a triple pointer,
        # indexed in the following way:
        # tile = tiles[tile_index],
        # rung = tile[rung_index],
        # indexᵖ = rung[rung_particle_index],
        # where indexᵖ is the particle index.
        # The number of tiles is fixed for any given tiling,
        # and the number of rungs is always given by N_rungs.
        # The number of particles within a rung is not constant,
        # however, and so we additionally need to keep track of the
        # allocation and occupation size of each rung.
        # These are denoted rungs_sizes and rungs_N, and are similarly
        # indexed as
        # rungs_sizes = tiles_rungs_sizes[tile_index],
        # rung_size = rungs_sizes[rung_index],
        # and
        # rungs_N = tiles_rungs_N[tile_index],
        # rung_N = rungs_N[rung_index].
        # Thus, rung_particle_index goes from 0 to rung_N - 1.
        # We furthermore have the contain_particles array, which
        # indicate the content of a given tile:
        # contain_particles[tile_index] == 0
        #   → No particles at all
        # contain_particles[tile_index] == 1
        #   → Only inactive particles
        # contain_particles[tile_index] == 2
        #   → At least 1 active particle
        # contain_particles[tile_index] == 3
        #   → At least 1 active particle,
        #     at least 1 of which have an upcoming rung jump
        self.tiles                  = malloc(self.size        *sizeof('Py_ssize_t**'))
        self.tiles_rungs_sizes_data = malloc(self.size*N_rungs*sizeof('Py_ssize_t'))
        self.tiles_rungs_sizes      = malloc(self.size        *sizeof('Py_ssize_t*'))
        self.tiles_rungs_N_data     = malloc(self.size*N_rungs*sizeof('Py_ssize_t'))
        self.tiles_rungs_N          = malloc(self.size        *sizeof('Py_ssize_t*'))
        self.contain_particles      = malloc(self.size        *sizeof('signed char'))
        tiles_rungs_sizes_mv = cast(self.tiles_rungs_sizes_data, 'Py_ssize_t[:self.size*N_rungs]')
        tiles_rungs_N_mv     = cast(self.tiles_rungs_N_data,     'Py_ssize_t[:self.size*N_rungs]')
        for tile_index in range(self.size):
            tile = malloc(N_rungs*sizeof('Py_ssize_t*'))
            self.tiles[tile_index] = tile
            for rung_index in range(N_rungs):
                rung = malloc(initial_rung_sizes[rung_index]*sizeof('Py_ssize_t'))
                tile[rung_index] = rung
            rungs_sizes_mv = tiles_rungs_sizes_mv[tile_index*N_rungs:(tile_index + 1)*N_rungs]
            rungs_sizes = cython.address(rungs_sizes_mv[:])
            self.tiles_rungs_sizes[tile_index] = rungs_sizes
            for rung_index in range(N_rungs):
                rungs_sizes[rung_index] = initial_rung_sizes[rung_index]
            rungs_N_mv = tiles_rungs_N_mv[tile_index*N_rungs:(tile_index + 1)*N_rungs]
            rungs_N = cython.address(rungs_N_mv[:])
            self.tiles_rungs_N[tile_index] = rungs_N
            for rung_index in range(N_rungs):
                rungs_N[rung_index] = 0
            self.contain_particles[tile_index] = 0
        # When sorting particles into tiles, we need to know the spatial
        # location of each tile. For this, we need the position of the
        # beginning of the tiling (the left, backward, lower corner of
        # the [0, 0, 0] tile) as well as the size of the complete
        # tiling. While the extent is supplied to this initialiser
        # method, the location should be adjusted afterwards using
        # the relocate method.
        self.extent = extent
        self.tile_extent = empty(3, dtype=C2np['double'])
        for dim in range(3):
            self.tile_extent[dim] = self.extent[dim]/self.shape[dim]
        self.location = zeros(3, dtype=C2np['double'])
        # If this is a subtiling and it uses automatic refinement,
        # it needs a refinement period (measured in base time steps).
        # This is set by the subtiling_refinement_period user parameter.
        # However, this is for subtilings with cubic subtiles, i.e. subtilings
        # where the subtiles have the same physical extent in each
        # direction (not necessarily a shape of the form [n]*3).
        # Such cubic subtiles will get all three dimensions refined
        # simultaneously, and thus stay cubic. For non-cubic
        # subtiles, the refinement works on only one or two
        # directions at a time, and hence these are in need of a
        # shorter refinement period.
        if refinement_period > 0:
            refinement_period = int(round(refinement_period/len(set(self.tile_extent))))
            if refinement_period < subtiling_refinement_period_min:
                refinement_period = subtiling_refinement_period_min
        self.refinement_period = refinement_period
        self.refinement_offset = 0
        # The running total computation time,
        # measured over interactions between tile pairs.
        # Both the computation_time and computation_time_total
        # attributes are set by the interactions module.
        # The main module looks up computation_time_total,
        # and nullifies it at the beginning of each time step.
        self.computation_time = 0
        self.computation_time_total = 0

    # Method for spatially relocating the tiling
    @cython.header(
        # Arguments
        location='double[::1]',
        # Locals
        returns='void',
    )
    def relocate(self, location):
        self.location = location

    # Method for converting a 1D/linear tile index
    # into its 3D equivalent.
    @cython.header(
        # Arguments
        tile_index='Py_ssize_t',
        # Locals
        tile_index3D='Py_ssize_t*',
        returns='Py_ssize_t*',
    )
    def tile_index3D(self, tile_index):
        tile_index3D = cython.address(self.layout_1Dto3D[tile_index, :])
        return tile_index3D

    # Method for sorting particles into tiles. If the arguments
    # coarse_tiling and coarse_tiling_index are left out,
    # all particles within the attached component will be
    # tile sorted. Otherwise, only the subset of the particles
    # given by these additional arguments will be taken into account
    # when performing the tile sorting.
    @cython.header(
        # Arguments
        coarse_tiling='Tiling',
        coarse_tile_index='Py_ssize_t',
        already_reset='bint',
        # Locals
        coarse_rung='Py_ssize_t*',
        coarse_rung_N='Py_ssize_t',
        coarse_rung_index='signed char',
        coarse_rung_particle_index='Py_ssize_t',
        coarse_rungs_N='Py_ssize_t*',
        coarse_tile='Py_ssize_t**',
        component='Component',
        contain_particles='signed char*',
        contains='signed char',
        i='Py_ssize_t',
        index='Py_ssize_t',
        indexᵖ='Py_ssize_t',
        indexˣ='Py_ssize_t',
        j='Py_ssize_t',
        k='Py_ssize_t',
        lowest_active_rung='signed char',
        posxˣ='double*',
        posyˣ='double*',
        poszˣ='double*',
        rung='Py_ssize_t*',
        rung_N='Py_ssize_t',
        rung_growth='double',
        rung_index='signed char',
        rung_index_jumped='signed char',
        rung_indices='signed char*',
        rung_indices_jumped='signed char*',
        rung_size='Py_ssize_t',
        tile='Py_ssize_t**',
        tile_index='Py_ssize_t',
        tiles='Py_ssize_t***',
        tiles_rungs_N='Py_ssize_t**',
        tiles_rungs_N_data='Py_ssize_t*',
        tiles_rungs_sizes_data='Py_ssize_t*',
        returns='void',
    )
    def sort(self, coarse_tiling=None, coarse_tile_index=-1, already_reset=False):
        # Extract variables
        component = self.component
        rung_indices        = component.rung_indices
        rung_indices_jumped = component.rung_indices_jumped
        lowest_active_rung  = component.lowest_active_rung
        posxˣ = component.posxˣ
        posyˣ = component.posyˣ
        poszˣ = component.poszˣ
        tiles                  = self.tiles
        tiles_rungs_sizes_data = self.tiles_rungs_sizes_data
        tiles_rungs_N_data     = self.tiles_rungs_N_data
        tiles_rungs_N          = self.tiles_rungs_N
        contain_particles      = self.contain_particles
        # If only the particles within a coarser tile should be sorted
        # (into finer tiles), extract this coarse tile.
        if 𝔹[coarse_tiling is not None]:
            # If this is the trivial tiling, the only allowed coarse
            # tiling is itself. It is thus already sorted.
            if self.is_trivial:
                return
            coarse_tile = coarse_tiling.tiles[coarse_tile_index]
            coarse_rungs_N = coarse_tiling.tiles_rungs_N[coarse_tile_index]
        # Reset particle information if not already done
        if not already_reset:
            # Reset the particle count for each rung within every tile
            for index in range(self.size*N_rungs):
                tiles_rungs_N_data[index] = 0
            # Reset particle content within each tile
            for tile_index in range(self.size):
                contain_particles[tile_index] = 0
        # Place each particle into a tile. If any of the particles
        # are outside of the tiling, this will fail.
        coarse_rung_N = 1      # Needed when not using a coarse tile
        rung_index = 0         # Needed when not using rungs
        rung_index_jumped = 0  # Needed when not using rungs
        tile_index = 0         # Used if this is the trivial tiling
        for indexᵖ in range(component.N_local if coarse_tiling is None else N_rungs):
            # When a coarse tile is in use, the indexᵖ variable is
            # really the rung_index for the coarse tile.
            # Use it to pick out the coarse rung.
            with unswitch:
                if 𝔹[coarse_tiling is not None]:
                    coarse_rung_index = indexᵖ
                    coarse_rung   = coarse_tile   [coarse_rung_index]
                    coarse_rung_N = coarse_rungs_N[coarse_rung_index]
            # Loop over the particles in this coarse rung. When not
            # using a coarse tile, this is a one-iteration loop.
            for coarse_rung_particle_index in range(coarse_rung_N):
                with unswitch:
                    if 𝔹[coarse_tiling is not None]:
                        indexᵖ = coarse_rung[coarse_rung_particle_index]
                # Determine the tile within which this particle
                # is located. For tilings of shape (1, 1, 1) (e.g. the
                # trivial tiling) we already know the answer.
                with unswitch:
                    if not (self.shape[0] == self.shape[1] == self.shape[2] == 1):
                        indexˣ = 3*indexᵖ
                        # Guard against round-off errors for positions
                        # at or very near the higher edge of the
                        # (sub)tile. We have found that introducing
                        # 2*machine_ϵ both as a translation of the
                        # numerator and as a scaling of the denominator
                        # works best, and that 1*machine_ϵ is not
                        # always enough. Note that this translation is
                        # OK at the lower edge, as casting a
                        # floating-point value only slightly below 0.0
                        # to an int results in 0.
                        i = cast((posxˣ[indexˣ] - ℝ[self.location[0]*(1 + 2*machine_ϵ)])
                            *ℝ[(1/self.tile_extent[0])*(1 - 2*machine_ϵ)], 'Py_ssize_t')
                        j = cast((posyˣ[indexˣ] - ℝ[self.location[1]*(1 + 2*machine_ϵ)])
                            *ℝ[(1/self.tile_extent[1])*(1 - 2*machine_ϵ)], 'Py_ssize_t')
                        k = cast((poszˣ[indexˣ] - ℝ[self.location[2]*(1 + 2*machine_ϵ)])
                            *ℝ[(1/self.tile_extent[2])*(1 - 2*machine_ϵ)], 'Py_ssize_t')
                        # The tile_index is given by
                        # self.layout[i, j, k], but as an optimization
                        # we compute it ourselves.
                        tile_index = (
                            + i*ℤ[self.shape[2]*self.shape[1]]
                            + j*ℤ[self.shape[2]]
                            + k
                        )
                # Record particle content within the tile using the
                # mapping documented in Tiling.__init__.
                # The possible values are:
                #   1: Particle sits on an inactive rung.
                #   2: Particle sits on an active rung
                #      and is not flagged to jump.
                #   3: Particle sits on active rung
                #      and is flagged to jump.
                with unswitch:
                    if component.use_rungs:
                        rung_index = rung_indices[indexᵖ]
                        if 𝔹[rung_index >= lowest_active_rung]:
                            rung_index_jumped = rung_indices_jumped[indexᵖ]
                            contains = 2 + (rung_index_jumped >= N_rungs)
                        else:
                            contains = 1
                    else:
                        contains = 2
                if contain_particles[tile_index] < contains:
                    contain_particles[tile_index] = contains
                # Resize this rung within the tile, if needed
                tile = tiles[tile_index]
                rung = tile[rung_index]
                index = tile_index*ℤ[N_rungs] + rung_index
                rung_N = tiles_rungs_N_data[index]
                rung_size = tiles_rungs_sizes_data[index]
                if rung_size == rung_N:
                    rung_growth = 1.2
                    rung_size = cast(rung_growth*rung_size, 'Py_ssize_t') + 1
                    rung = realloc(rung, rung_size*sizeof('Py_ssize_t'))
                    tile[rung_index] = rung
                    tiles_rungs_sizes_data[index] = rung_size
                # Add this particle to the rung within the tile
                rung[rung_N] = indexᵖ
                tiles_rungs_N_data[index] += 1

    # This method is automatically called when a Tiling instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        cython.declare(
            rung_index='signed char',
            tile_index='Py_ssize_t',
        )
        for tile_index in range(self.size):
            for rung_index in range(N_rungs):
                free(self.tiles[tile_index][rung_index])
            free(self.tiles[tile_index])
        free(self.tiles)
        free(self.tiles_rungs_sizes_data)
        free(self.tiles_rungs_sizes)
        free(self.tiles_rungs_N_data)
        free(self.tiles_rungs_N)
        free(self.contain_particles)

    # String representation
    def __repr__(self):
        return f'<tiling "{self.name}" of "{self.component.name}" with shape {tuple(self.shape)}>'
    def __str__(self):
        return self.__repr__()


# The class governing any component of the universe
@cython.cclass
class Component:
    """An instance of this class represents either a collection of
    particles or a grid of fluid values. A Component instance should be
    present on all processes.
    """

    # Initialisation method
    @cython.pheader(
        # Arguments
        name=str,
        species=str,
        N='Py_ssize_t',
        gridsize='Py_ssize_t',
        mass='double',
        boltzmann_order='Py_ssize_t',
        boltzmann_closure=str,
        # Locals
        tile_index='signed char',
        index='Py_ssize_t',
        indexᵖ='Py_ssize_t',
    )
    def __init__(self, name, species, *,
        N=-1,
        gridsize=-1,
        mass=-1,
        boltzmann_order=-2,
        boltzmann_closure=None,
    ):
        # The keyword-only arguments are passed from dicts in the
        # initial_conditions user parameter. If not specified there
        # (None passed) they will be set through other parameters.
        # Of special interest is the fluid parameters boltzmann_order,
        # boltzmann_closure and approximations. Together, these control
        # the degree to which a fluid component will behave non-
        # linearly. Below is listed an overview of all allowed
        # combinations of boltzmann_order and boltzmann_closure,
        # together with the accompanying fluid variable behavoir.
        # Note that for particle components, only boltzmann_order = 1
        # is allowed.
        #
        # boltzmann_order = -1, boltzmann_closure = 'class':
        #     linear ϱ  (Realised continuously, affects other components gravitationally)
        #
        # boltzmann_order = 0, boltzmann_closure = 'truncate':
        #     non-linear ϱ  (Though "non-linear", ϱ is frozen in time as no J exist.
        #                    Also, unlike when boltzmann_order = -1 and
        #                    boltzmann_closure = 'class', ϱ will only be realised
        #                    at the beginning of the simulation.)
        #
        # boltzmann_order = 0, boltzmann_closure = 'class':
        #     non-linear ϱ
        #         linear J  (realised continuously)
        #         linear 𝒫  (P=wρ approximation enforced)
        #
        # boltzmann_order: 1, boltzmann_closure = 'truncate':
        #     non-linear ϱ
        #     non-linear J
        #         linear 𝒫  (P=wρ approximation enforced)
        #
        # boltzmann_order = 1, boltzmann_closure = 'class':
        #     non-linear ϱ
        #     non-linear J
        #         linear 𝒫  (realised continuously)
        #         linear ς  (realised continuously)
        #
        # boltzmann_order = 2, boltzmann_closure = 'truncate':
        #     non-linear ϱ
        #     non-linear J
        #     non-linear 𝒫  (Though "non-linear", 𝒫 is frozen in time since the evolution equation
        #                    for 𝒫 is not implemented.
        #                    Also, unlike when boltzmann_order = 1 and boltzmann_closure = 'class',
        #                    𝒫 will only be realised at the beginning of the simulation.)
        #     non-linear ς  (Though "non-linear", ς is frozen in time since the evolution equation
        #                    for ς is not implemented.
        #                    Also, unlike when boltzmann_order = 1 and boltzmann_closure = 'class',
        #                    ς will only be realised at the beginning of the simulation.)
        #
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Component type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        # General component attributes
        public str name
        public str species
        public str representation
        public dict forces
        public dict potential_gridsizes
        public dict potential_differentiations
        public str class_species
        public dict snapshot_vars
        public tuple life
        # Particle attributes
        public Py_ssize_t N
        public Py_ssize_t N_allocated
        public Py_ssize_t N_local
        public double mass
        public double softening_length
        public Py_ssize_t powerspec_upstream_gridsize
        public Py_ssize_t bispec_upstream_gridsize
        public Py_ssize_t render2D_upstream_gridsize
        public Py_ssize_t render3D_upstream_gridsize
        public str preic_lattice
        # Particle data
        double* pos
        public double[::1] pos_mv
        public double[:, ::1] pos_mv3
        public double[:] posx
        public double[:] posy
        public double[:] posz
        double* posxˣ
        double* posyˣ
        double* poszˣ
        double* mom
        public double[::1] mom_mv
        public double[:, ::1] mom_mv3
        public double[:] momx
        public double[:] momy
        public double[:] momz
        double* momxˣ
        double* momyˣ
        double* momzˣ
        double* Δmom
        public double[::1] Δmom_mv
        public double[:, ::1] Δmom_mv3
        public double[:] Δmomx
        public double[:] Δmomy
        public double[:] Δmomz
        double* Δmomxˣ
        double* Δmomyˣ
        double* Δmomzˣ
        # Particle IDs
        public bint use_ids
        Py_ssize_t* ids
        public Py_ssize_t[::1] ids_mv
        # Short-range rungs
        public bint use_rungs
        signed char lowest_active_rung
        signed char lowest_populated_rung
        signed char highest_populated_rung
        Py_ssize_t* rungs_N
        signed char* rung_indices
        signed char[::1] rung_indices_mv
        signed char* rung_indices_jumped
        signed char[::1] rung_indices_jumped_mv
        # Dict used for storing Tiling instances
        public dict tilings
        public object n_interactions  # collections.defaultdict
        # Fluid attributes
        public Py_ssize_t gridsize
        public tuple shape
        public tuple shape_noghosts
        public Py_ssize_t size
        public Py_ssize_t size_noghosts
        public dict fluid_names
        public Py_ssize_t boltzmann_order
        public str boltzmann_closure
        public dict realization_options
        public str w_type
        public double w_constant
        public double[:, ::1] w_tabulated
        public str w_expression
        Spline w_spline
        public str w_eff_type
        Spline w_eff_spline
        public dict approximations
        public double _ϱ_bar
        # Fluid data
        public list fluidvars
        FluidScalar ϱ
        public object J  # Tensor
        FluidScalar Jx
        FluidScalar Jy
        FluidScalar Jz
        public object ς  # Tensor
        FluidScalar ςxx
        FluidScalar ςxy
        FluidScalar ςxz
        FluidScalar ςyx
        FluidScalar ςyy
        FluidScalar ςyz
        FluidScalar ςzx
        FluidScalar ςzy
        FluidScalar ςzz
        FluidScalar 𝒫
        # Other
        public list components_all
        """
        # A reference to each and every Component instance ever created,
        # including internally defined ones, is stored in the global
        # components_all list. We also store a reference to
        # components_all on this Component instance.
        components_all.append(self)
        self.components_all = components_all
        # Check that the name does not conflict with
        # one of the special names used internally,
        # and that the name has not already been used.
        name = name.strip()
        if name in internally_defined_names:
            masterwarn(
                f'A component by the name of "{name}" is to be created. '
                f'As this name is used internally by the code, '
                f'this may lead to erroneous behaviour.'
            )
        elif not allow_similarly_named_components and name in component_names:
            masterwarn(
                f'A component with the name of "{name}" has already '
                f'been instantiated. Instantiating multiple components '
                f'with the same name may lead to undesired behaviour.'
            )
        for char in r',{}':
            if char in name:
                masterwarn(
                    f'A component by the name of "{name}" is to be created. '
                    f'As this name contains a "{char}" character, '
                    f'this may lead to erroneous behaviour.'
                )
        if name:
            component_names.add(name)
        # General attributes
        self.name = name
        if N != -1:
            self.N = N
            self.gridsize = 1
            self.representation = 'particles'
        elif gridsize != -1:
            self.gridsize = gridsize
            self.N = 1
            self.representation = 'fluid'
            if self.gridsize%2 != 0:
                masterwarn(
                    f'Fluid component {self.name} has an odd grid size ({self.gridsize}). '
                    f'Some operations may not function correctly.'
                )
        else:
            abort(f'Neither N nor gridsize set for {self.name}')
        if species is None or species.lower() == 'none':
            self.species = ''
            species = is_selected(self, select_species)
        self.species = ''
        for single_species in str(species).split('+'):
            single_species = single_species.replace('[', ' ').replace(']', ' ').strip()
            match = re.search(r'(.+?) *(\d+) *$', single_species)
            if match:
                single_species = match.group(1)
            self.species += '+' + species_canonical.get(single_species.lower(), '')
            if match:
                self.species += ' ' + match.group(2)
            if self.species.endswith('+'):
                abort(
                    f'Species "{single_species}" not supported. The supported species are:',
                    sorted(set(species_canonical.values())),
                )
        self.species = self.species.strip(' +')
        # Determine pre-initial condition lattice type
        self.preic_lattice = ''
        if self.representation == 'particles':
            if iscubic(self.N):
                # Use simple cubic lattice
                self.preic_lattice = 'sc'
            elif self.N%2 == 0 and iscubic(self.N//2):
                # Use body-centered cubic lattice
                self.preic_lattice = 'bcc'
            elif self.N%4 == 0 and iscubic(self.N//4):
                # Use face-centered cubic lattice
                self.preic_lattice = 'fcc'
        # Set forces (and force methods)
        forces = is_selected(self, select_forces, accumulate=True)
        if not forces:
            forces = {}
        self.forces = forces
        # Check that needed short-range parameters are set
        for force, method in self.forces.items():
            if method == 'p3m':
                if shortrange_params[force]['scale'] < 0:
                    abort(
                        f'It is specified that {self.name} should use P³M {force}, '
                        f'but the grid size to use could not be determined. Please also '
                        f'specify the grid size to use in '
                        f'potential_options["gridsize"]["global"]["{force}"]["p3m"], '
                        f'and/or specify shortrange_params["{force}"]["scale"].'
                    )
        # Function for converting expressions involving
        # 'N' and 'gridsize' to floats.
        def to_float(s):
            s = unicode(str(s))
            if self.representation == 'particles':
                # Here Ñ = n³ is a shorthand for one of {N, N/2 or N/4}
                # depending on whether we are realizing N = n³ = Ñ,
                # N = 2n³ = 2Ñ or N = 4n³ = 4Ñ particles.
                Ñ = self.N//{'sc': 1, 'bcc': 2, 'fcc': 4}.get(self.preic_lattice, 1)
                s = (
                    s
                    .replace('N', str(self.N))
                    .replace('gridsize', str(cbrt(Ñ)))
                    .replace(unicode('Ñ'), str(Ñ))
                )
            else:  # self.representation == 'fluid':
                s = (
                    s
                    .replace('N', str(self.gridsize**3))
                    .replace('gridsize', str(self.gridsize))
                    .replace(unicode('Ñ'), str(self.gridsize**3))
                )
            s = s.replace('nprocs', str(nprocs))
            return eval(s, globals(), units_dict)
        # Set upstream and downstream potential grid sizes
        # for each force and method.
        self.potential_gridsizes = {
            key_force: dict_method_cleaned
            for key_force, dict_method in is_selected(
                self, potential_options['gridsize'], accumulate=True, default={},
            ).items()
            if (
                dict_method_cleaned := {
                    key_method: gridsizes
                    for key_method, gridsizes in dict_method.items()
                    if -1 not in gridsizes
                }
            )
        }
        for force, method in self.forces.items():
            self.potential_gridsizes.setdefault(force, {})
            methods = [method]
            if method == 'p3m':
                # If P³M is to be used, also set up potential grid sizes
                # for PM, as P³M will be switched out for PM in the case
                # of fluid components.
                methods.append('pm')
            for method_extra in methods:
                if self.representation == 'fluid':
                    # Fluids should have upstream and downstream
                    # grid sizes equal to that of the fluid grids.
                    gridsizes_default = self.gridsize
                else:
                    gridsizes_default = (
                        potential_options['gridsize']['global'].get(force, {}).get(method_extra)
                    )
                    if not gridsizes_default:
                        continue
                gridsizes = self.potential_gridsizes[force].get(method_extra, gridsizes_default)
                if gridsizes == -1:
                    if self.representation == 'particles':
                        # For particle components, we choose the default
                        # potential grid size to equal cbrt(N) (for both
                        # upstream and downstream), except when using
                        # P³M where having a large grid is very
                        # important for performance, in which case we
                        # use 2*cbrt(N). For pre-initial condition
                        # lattices other than the simple cubic
                        # (bocy-centered cubic, face-centered cubic),
                        # we use the same grid size as for the
                        # simple cubic.
                        gridsizes = ('cbrt(Ñ)', )*2
                        if method_extra == 'p3m':
                            gridsizes = ('2*cbrt(Ñ)', )*2
                    elif self.representation == 'fluid':
                        # Fluids should have upstream and downstream
                        # grid sizes equal to that of the fluid grids.
                        gridsizes = ('gridsize', 'gridsize')
                gridsizes = any2list(gridsizes)
                if len(gridsizes) == 1:
                    gridsizes *= 2
                self.potential_gridsizes[force][method_extra] = (
                    PotentialGridsizesComponent(*gridsizes)
                )
        for force, dict_method in self.potential_gridsizes.items():
            for method, gridsizes in dict_method.items():
                gridsizes_transformed = [
                    int(round(to_float(gridsize_str)))
                    for gridsize_str in gridsizes
                ]
                dict_method[method] = PotentialGridsizesComponent(*gridsizes_transformed)
        # Set potential differentiation order for each force and method.
        self.potential_differentiations = is_selected(
            self, potential_options['differentiation'], default={},
        )
        for force, method in self.forces.items():
            if method not in ('pm', 'p3m'):
                continue
            self.potential_differentiations.setdefault(force, {})
            methods = [method]
            if method == 'p3m':
                # If P³M is to be used, also set up potential
                # differentiation order for PM, as P³M will be switched
                # out for PM in the case of fluid components.
                methods.append('pm')
            for method_extra in methods:
                self.potential_differentiations[force].setdefault(
                    method_extra,
                    potential_options['differentiation']['default'][force][method_extra],
                )
        # Check that fluid components have upstream and downstream
        # potential grid sizes equal to their fluid grid size.
        if self.representation == 'fluid' and self.name:
            gridsizes_ought = PotentialGridsizesComponent(self.gridsize, self.gridsize)
            for force, dict_method in self.potential_gridsizes.items():
                for method, gridsizes in dict_method.items():
                    if gridsizes != gridsizes_ought:
                        # As PM is the only method available for fluid
                        # components, we do not count wrong
                        # upstream/downstream potential grid size
                        # specifications of other methods as errors.
                        if method == 'pm':
                            abort(
                                f'Upstream and downstream potential grid sizes of fluid component '
                                f'"{self.name}" for force "{force}" with method "{method}" was '
                                f'set to {tuple(dict_method[method])} but both need to equal the '
                                f'fluid grid size {self.gridsize}'
                            )
                        else:
                            # Allow wrong specification
                            # but overwrite for consistency.
                            dict_method[method] = gridsizes_ought
        # Mapping from component names to number of
        # (instantaneous) interactions that have taken place.
        self.n_interactions = collections.defaultdict(int)
        # Set the CLASS species
        if self.species == 'none':
            if self.name:
                abort(
                    f'Neither "species" nor "class species" '
                    f'specified for component "{self.name}"'
                )
            else:
                abort(
                    f'Neither "species" nor "class species" '
                    f'specified for unnamed component'
                )
        class_species = ''
        for single_species in self.species.split('+'):
            if single_species in species_registered:
                class_species += '+' + species_registered[single_species].class_species
            else:
                abort(
                    f'Default CLASS species assignment failed because '
                    f'the species "{single_species}" does not map to any CLASS species'
                )
        self.class_species = class_species.strip(' +')
        # Set closure rule for the Boltzmann hierarchy
        if boltzmann_closure is None:
            boltzmann_closure = is_selected(self, select_boltzmann_closure)
        if not boltzmann_closure:
            boltzmann_closure = ''
        self.boltzmann_closure = boltzmann_closure.lower()
        if self.representation == 'fluid' and self.boltzmann_closure not in ('truncate', 'class'):
            abort(
                f'{self.name} was initialised '
                f'with an unknown Boltzmann closure of "{self.boltzmann_closure}"'
            )
        # Set life
        life = is_selected(self, select_lives)
        life = tuple([float(el) for el in sorted(any2list(life))])
        if len(life) != 2:
            abort(f'life = {life} of "{self.name}" not understood')
        self.life = life
        # Set approximations. Ensure that all implemented approximations
        # get set to either True or False. If an approximation is not
        # set for this component, its value defaults to False.
        # Also, specific circumstances may force some approximations to
        # have a specific value.
        approximations = is_selected(self, select_approximations, accumulate=True)
        if not approximations:
            approximations = {}
        approximations_transformed = {}
        for approximation, value in approximations.items():
            # General transformations
            approximation = unicode(approximation)
            for char in unicode(' *×^'):
                approximation = approximation.replace(char, '')
            for n in range(10):
                approximation = approximation.replace(unicode_superscript(str(n)), str(n))
            # The P=wρ approximation
            approximation_transformed = approximation
            for s in ('\rho', r'\rho', 'rho'):
                approximation_transformed = approximation_transformed.replace(s, unicode('ρ'))
            if approximation_transformed in {
                    unicode('P=wρ'),
                    unicode('P=ρw'),
                    unicode('wρ=P'),
                    unicode('ρw=P'),
            }:
                approximation_transformed = unicode('P=wρ')
            approximations_transformed[approximation_transformed] = bool(value)
        approximations = approximations_transformed
        for approximation, value in approximations.copy().items():
            if unicode(approximation) not in approximations_implemented:
                abort(
                    f'{self.name} was initialised '
                    f'with the unknown approximation "{approximation}"'
                )
            approximations[asciify(approximation)] = value
            approximations[unicode(approximation)] = value
        for approximation in approximations_implemented:
            value = approximations.get(approximation, False)
            approximations[asciify(approximation)] = value
            approximations[unicode(approximation)] = value
        if self.representation == 'particles':
            approximations[unicode('P=wρ')] = True
            approximations[asciify('P=wρ')] = True
        self.approximations = approximations
        # Set softening length
        softening_length = is_selected(self, select_softening_length)
        if isinstance(softening_length, str):
            # Evaluate softening_length if it's a str.
            # Replace 'N' with the number of particles of this component
            # and 'gridsize' with the grid size of this component.
            softening_length = to_float(softening_length)
        if not softening_length:
            if self.representation == 'particles':
                # If no name is given, this is an internally
                # used component, in which case it is OK not to have
                # any softening length set.
                if self.name:
                    masterwarn(f'No softening length set for {self.name}')
            softening_length = 0
        self.softening_length = float(softening_length)
        # Set upstream grid size for power spectra,
        # bispectra and 2D renders.
        for output_type, options in {
            'powerspec': powerspec_options,
            'bispec'   : bispec_options,
            'render2D' : render2D_options,
            'render3D' : render3D_options,
        }.items():
            upstream_gridsize = -1
            if self.name:
                upstream_gridsize = is_selected(
                    self, options['upstream gridsize'],
                    default=upstream_gridsize,
                )
            if upstream_gridsize == -1:
                if self.representation == 'fluid':
                    upstream_gridsize = 'gridsize'
                elif self.representation == 'particles':
                    upstream_gridsize = {
                        'powerspec': '2*cbrt(Ñ)',
                        'bispec'   : '2*cbrt(Ñ)',
                        'render2D' : '1*cbrt(Ñ)',
                        'render3D' : '1*cbrt(Ñ)',
                    }[output_type]
            upstream_gridsize = int(round(to_float(upstream_gridsize)))
            setattr(self, f'{output_type}_upstream_gridsize', upstream_gridsize)
            # Check that fluid components have upstream grid sizes
            # equal to their fluid grid size.
            if self.representation == 'fluid' and upstream_gridsize != self.gridsize and self.name:
                abort(
                    f'Upstream {output_type} grid size of fluid component "{self.name}" '
                    f'was set to {upstream_gridsize} but needs to equal the fluid '
                    f'grid size {self.gridsize}'
                )
        # This attribute will store the conserved mean density
        # of this component. It is set by the ϱ_bar method.
        self._ϱ_bar = -1
        # Particle attributes
        self.mass = mass
        self.N_allocated = 1
        self.N_local = 1
        # Particle data
        self.pos = malloc(3*self.N_allocated*sizeof('double'))
        self.pos_mv = cast(self.pos, 'double[:3*self.N_allocated]')
        self.pos_mv3 = cast(self.pos, 'double[:self.N_allocated, :3]')
        self.posx = self.pos_mv3[:, 0]
        self.posy = self.pos_mv3[:, 1]
        self.posz = self.pos_mv3[:, 2]
        self.posxˣ = cython.address(self.pos_mv[0:])
        self.posyˣ = cython.address(self.pos_mv[1:])
        self.poszˣ = cython.address(self.pos_mv[2:])
        self.mom = malloc(3*self.N_allocated*sizeof('double'))
        self.mom_mv = cast(self.mom, 'double[:3*self.N_allocated]')
        self.mom_mv3 = cast(self.mom, 'double[:self.N_allocated, :3]')
        self.momx = self.mom_mv3[:, 0]
        self.momy = self.mom_mv3[:, 1]
        self.momz = self.mom_mv3[:, 2]
        self.momxˣ = cython.address(self.mom_mv[0:])
        self.momyˣ = cython.address(self.mom_mv[1:])
        self.momzˣ = cython.address(self.mom_mv[2:])
        self.Δmom = malloc(3*self.N_allocated*sizeof('double'))
        self.Δmom_mv = cast(self.Δmom, 'double[:3*self.N_allocated]')
        self.Δmom_mv3 = cast(self.Δmom, 'double[:self.N_allocated, :3]')
        self.Δmomx = self.Δmom_mv3[:, 0]
        self.Δmomy = self.Δmom_mv3[:, 1]
        self.Δmomz = self.Δmom_mv3[:, 2]
        self.Δmomxˣ = cython.address(self.Δmom_mv[0:])
        self.Δmomyˣ = cython.address(self.Δmom_mv[1:])
        self.Δmomzˣ = cython.address(self.Δmom_mv[2:])
        self.Δmom_mv[:3*self.N_allocated] = 0
        # Particle IDs
        self.use_ids = bool(is_selected(self, select_particle_id))
        self.ids = malloc(self.N_allocated*sizeof('Py_ssize_t'))
        self.ids_mv = cast(self.ids, 'Py_ssize_t[:self.N_allocated]')
        # Short-range rungs
        self.use_rungs = bool(
            N_rungs > 1
            and self.representation == 'particles'
            and ({'ppnonperiodic', 'pp', 'p3m'} & set(self.forces.values()))
        )
        self.lowest_active_rung = 0
        self.lowest_populated_rung = 0
        self.highest_populated_rung = 0
        self.rungs_N = malloc(N_rungs*sizeof('Py_ssize_t'))
        for tile_index in range(N_rungs):
            self.rungs_N[tile_index] = 0
        self.rung_indices = malloc(self.N_local*sizeof('signed char'))
        self.rung_indices_mv = cast(self.rung_indices, 'signed char[:self.N_local]')
        for indexᵖ in range(self.N_local):
            self.rung_indices[indexᵖ] = 0
        self.rung_indices_jumped = malloc(self.N_local*sizeof('signed char'))
        for indexᵖ in range(self.N_local):
            self.rung_indices_jumped[indexᵖ] = 0
        self.rung_indices_jumped_mv = cast(self.rung_indices_jumped, 'signed char[:self.N_local]')
        # Dict used for storing Tiling instances
        self.tilings = {}
        # Fluid attributes
        if boltzmann_order == -2:
            boltzmann_order = is_selected(self, select_boltzmann_order)
        self.boltzmann_order = boltzmann_order
        if self.representation == 'particles':
            if self.boltzmann_order != 1:
                abort(
                    f'Particle components must have boltzmann_order = 1, '
                    f'but boltzmann_order = {self.boltzmann_order} was specified for {self.name}'
                )
        elif self.representation == 'fluid':
            if self.boltzmann_order < -1:
                abort(
                    f'Having boltzmann_order < -1 is nonsensical, '
                    f'but boltzmann_order = {self.boltzmann_order} was specified for {self.name}'
                )
            if self.boltzmann_order == -1 and self.boltzmann_closure == 'truncate':
                abort(
                    f'The {self.name} (a fluid component) has no non-linear and no '
                    f'linear fluid variables, and so practically it does not exist. '
                    f'Such components are disallowed.'
                )
            if self.boltzmann_order == 2 and self.boltzmann_closure == 'class':
                abort(
                    f'{self.name} wants to close its Boltzmann hierarchy using '
                    f'the linear variable after ς from CLASS, which is not implemented. '
                    f'You need to lower its Boltzmann order to '
                    f'1 or use boltzmann_closure = "truncate".'
                )
            if self.boltzmann_order > 2:
                abort(
                    f'Fluids with boltzmann_order > 2 are not implemented, '
                    f'but boltzmann_order = {self.boltzmann_order} was specified for {self.name}'
                )
        self.shape = (1, 1, 1)
        self.shape_noghosts = (1, 1, 1)
        self.size = np.prod(self.shape)
        self.size_noghosts = np.prod(self.shape_noghosts)
        # Components with Boltzmann order -1 cannot receive forces
        # but may still take part in interactions as suppliers.
        # Any such component should not specify a particular method
        # for its forces.
        if self.boltzmann_order == -1:
            self.forces = {key: '' for key in self.forces}
        # Implement species specific restrictions on the forces below
        if self.species == 'lapse':
            # Only allow a component of the special "lapse" species to
            # participate in the lapse interaction.
            self.forces = {
                force: method for force, method in self.forces.items() if force == 'lapse'
            }
        # Fluid components may only receive a force using the PM method
        if (
            self.representation == 'fluid'
            and self.name and self.name not in internally_defined_names
        ):
            for force, method in self.forces.items():
                if method not in {'pm', ''}:
                    abort(
                        f'{self.name} wants to receive the {force} force '
                        f'using the {method} method, but only the pm method is allowed '
                        f'for fluid components.'
                    )
        # Set realisation options
        self.realization_options = {
            key: is_selected(self, option)
            for key, option in realization_options.items()
        }
        if (
            self.representation == 'particles'
            or self.boltzmann_order == -1
            or self.boltzmann_closure == 'truncate'
        ):
            self.realization_options['structure'] = 'primordial'
        self.realization_options['fixedamplitude'] = primordial_amplitude_fixed
        self.realization_options['phaseshift'] = primordial_phase_shift
        # Set snapshot variables
        varnames_all = {
            'particles': ('pos', 'mom'),
            'fluid': ('ϱ', 'J', '𝒫', 'ς'),
        }
        varnames = varnames_all[self.representation]
        wrong_varname_sets = {
            'pos': {'x', 'position', 'positions', 'Position', 'Positions'},
            'mom': {'momentum', 'momenta', 'Momentum', 'Momenta'},
            'ϱ': {'r', 'rho', 'ρ'},
            'J': {'j'},
            '𝒫': {'P', 'δP', 'δ𝒫', 'p', 'δp'},
            'ς': {'s', 'sigma', 'Sigma', 'σ', 'Σ'},
        }
        snapshot_vars = {}
        for key in ('save', 'load'):
            snapshot_vars[key] = {}
            snapshot_vars_selected = is_selected(self, snapshot_select[key])
            for varname in varnames:
                if unicode(varname) in snapshot_vars_selected:
                    snapshot_vars[key][varname] = snapshot_vars_selected[unicode(varname)]
                elif asciify(varname) in snapshot_vars_selected:
                    snapshot_vars[key][varname] = snapshot_vars_selected[asciify(varname)]
                else:
                    wrong_varnames = wrong_varname_sets[varname]
                    for wrong_varname in wrong_varnames:
                        if unicode(wrong_varname) in snapshot_vars_selected:
                            snapshot_vars[key][varname] = (
                                snapshot_vars_selected[unicode(wrong_varname)]
                            )
                            break
                        elif asciify(wrong_varname) in snapshot_vars_selected:
                            snapshot_vars[key][varname] = (
                                snapshot_vars_selected[asciify(wrong_varname)]
                            )
                            break
            for varname in varnames:
                snapshot_vars[key].setdefault(varname, False)
            for varname, val in snapshot_vars[key].copy().items():
                snapshot_vars[key][varname] = bool(val)
            # Add special 'any' flag, signalling wehther any data at all
            # from this component should be saved/loaded.
            if self.representation == 'particles':
                snapshot_vars[key]['any'] = any([
                    snapshot_vars[key][varname]
                    for varname in ('pos', 'mom')
                ])
            elif self.representation == 'fluid':
                snapshot_vars[key]['any'] = any([
                    snapshot_vars[key][varname]
                    for varname in {
                        -1: [],  # completely linear component
                         0: ['ϱ'],
                         1: ['ϱ', 'J'],
                         2: ['ϱ', 'J', '𝒫', 'ς'],
                    }[self.boltzmann_order]
                ])
            # Add missing variable names from the other representation,
            # needed when converting components
            # between the two representations.
            for varname in itertools.chain(*varnames_all.values()):
                snapshot_vars[key].setdefault(varname, True)
        self.snapshot_vars = snapshot_vars
        # Set the equation of state parameter w
        w = is_selected(self, select_eos_w)
        self.init_w(w)
        self.init_w_eff()
        # Fluid data.
        # Create the (boltzmann_order + 1) non-linear fluid variables
        # and store them in the fluidvars list. This is done even for
        # particle components, as the fluidscalars are all instantiated
        # with a grid size of 1. The is_linear argument specifies
        # whether the FluidScalar will be a linear or non-linear
        # variable, where a non-linear variable is one that is updated
        # non-linearly, as opposed to a linear variable which is only
        # updated through continuous realisation. Currently, only ϱ and
        # J is implemented as non-linear variables. It is still allowed
        # to have boltzmann_order == 2, in which case ς (and 𝒫) is also
        # specified as being non-linear, although no non-linear
        # evolution is implemented, meaning that these will then be
        # constant in time. Note that the 𝒫 fluid variable is
        # treated specially, as it really lives on the same tensor as
        # the ς fluid scalars. Therefore, the 𝒫 fluid scalar
        # is added later.
        self.fluidvars = []
        for index in range(self.boltzmann_order + 1):
            # Instantiate fluid variable as a 3×3×...×3 (index times)
            # symmetric tensor.
            fluidvar = Tensor(self, index, (3, )*index, symmetric=True)
            # Populate the tensor with fluid scalar fields
            for multi_index in fluidvar.multi_indices:
                fluidvar[multi_index] = FluidScalar(index, multi_index, is_linear=False)
            # Add the fluid variable to the list
            self.fluidvars.append(fluidvar)
        # If CLASS should be used to close the Boltzmann hierarchy,
        # we need one additional fluid variable. This should act like
        # a symmetric tensor of rank boltzmann_order, but really only a
        # single element of this tensor need to exist in memory.
        # For boltzmann_order == 1, ς is the additional fluid variable.
        # Instantiate the scalar element but disguised as a
        # 3×3×...×3 ((boltzmann_order + 1) times) symmetric tensor.
        # Importantly, this fluid variable is always considered linear.
        if self.boltzmann_closure == 'class':
            disguised_scalar = Tensor(
                self,
                self.boltzmann_order + 1,
                (3, )*(self.boltzmann_order + 1),
                symmetric=True,
            )
            # Populate the tensor with a fluidscalar
            multi_index = disguised_scalar.multi_indices[0]
            disguised_scalar[multi_index] = FluidScalar(
                self.boltzmann_order + 1, multi_index, is_linear=True,
            )
            # Add this additional fluid variable to the list
            self.fluidvars.append(disguised_scalar)
        # Ensure that the approximation P=wρ is set to True
        # for fluid components which have either a linear J
        # fluid variable, or a non-linear J fluid variable but with the
        # non-linear Boltzmann hierarchy truncated right after J.
        if not self.approximations['P=wρ']:
            if (   self.boltzmann_order < 0
                or (self.boltzmann_order == 0 and self.boltzmann_closure == 'truncate')):
                # The 𝒫 fluid scalar does not exist at all for
                # this component, and so whether the P=wρ approximation
                # is True or not does not make much sense.
                # We set it to True, reflecting the fact that
                # 𝒫 certainly is not a non-linear variable.
                self.approximations[asciify('P=wρ')] = True
                self.approximations[unicode('P=wρ')] = True
            elif self.boltzmann_order == 0 and self.boltzmann_closure == 'class':
                masterwarn(
                    f'The P=wρ approximation has been switched on for {self.name} '
                    f'because Jⁱ = a⁴(ρ + c⁻²P)uⁱ is a linear fluid variable.'
                )
                self.approximations[asciify('P=wρ')] = True
                self.approximations[unicode('P=wρ')] = True
            elif self.boltzmann_order == 1 and self.boltzmann_closure == 'truncate':
                masterwarn(
                    f'The P=wρ approximation has been switched on for {self.name} '
                    f'because the non-linear Boltzmann hierarchy is truncated after the '
                    f'non-linear fluid variable Jⁱ, while 𝒫 is part of the next fluid variable.'
                )
                self.approximations[asciify('P=wρ')] = True
                self.approximations[unicode('P=wρ')] = True
        # When the P=wρ approximation is False, the fluid variable 𝒫
        # has to follow the structure of ϱ closely. Otherwise, spurious
        # features will develop in ϱ. Display a warning if the
        # realisation option for 𝒫 is set so that its structure does not
        # match that of ϱ.
        if (
            not self.approximations['P=wρ']
            and self.realization_options['structure'] == 'primordial'
        ):
            masterwarn(
                f'The 𝒫 fluid variable of {self.name} will be realised using the '
                f'primordial structure throughout time. This is known to '
                f'generates spurious features.'
            )
        # When the P=wρ approximation is True, the 𝒫 fluid variable is
        # superfluous. Yet, as it is used in the definition of J,
        # J = a⁴(ρ + P)u, P = a**(-3*(1 + w_eff))*𝒫, it is simplest to
        # just always instantiate a complete 𝒫 fluid variable,
        # regardless of whether 𝒫 appears in the closed
        # Boltzmann hierarchy. We place 𝒫 on ς, since 𝒫 is the trace
        # missing from ς. The only time we do not instantiate 𝒫 is for
        # a fluid without any J variable, be it linear or non-linear.
        if not (    self.boltzmann_order < 0
                or (self.boltzmann_order == 0 and self.boltzmann_closure == 'truncate')):
            # We need a 𝒫 fluid scalar
            if (   (self.boltzmann_order == 0 and self.boltzmann_closure == 'class')
                or (self.boltzmann_order == 1 and self.boltzmann_closure == 'truncate')
                ):
                # The ς tensor on which 𝒫 lives does not yet exist.
                # Instantiate a fake ς tensor, used only to store 𝒫.
                self.fluidvars.append(Tensor(self, 2, (), symmetric=True, active=False))
            # Add the 𝒫 fluid scalar to the ς tensor
            self.fluidvars[2]['trace'] = FluidScalar(0, 0,
                is_linear=(self.boltzmann_order < 2 or self.approximations['P=wρ']),
            )
        # Construct mapping from names of fluid variables (e.g. J)
        # to their indices in self.fluidvars, and also from names of
        # fluid scalars (e.g. ϱ, Jx) to tuple of the form
        # (index, multi_index). The fluid scalar is then given
        # by self.fluidvars[index][multi_index].
        # Also include trivial mappings from indices to themselves,
        # and the special "reverse" mapping from indices to names
        # given by the 'ordered' key.
        self.fluid_names = {
            'ordered': fluidvar_names[:
                self.boltzmann_order + (1 if self.boltzmann_closure == 'truncate' else 2)
            ]
        }
        for index, (fluidvar, fluidvar_name) in enumerate(
            zip(self.fluidvars, self.fluid_names['ordered'])
        ):
            # The fluid variable
            self.fluid_names[asciify(fluidvar_name)] = index
            self.fluid_names[unicode(fluidvar_name)] = index
            self.fluid_names[index                 ] = index
            # The fluid scalars
            for multi_index in fluidvar.multi_indices:
                fluidscalar_name = fluidvar_name
                if index > 0:
                    fluidscalar_name += ''.join(['xyz'[mi] for mi in multi_index])
                self.fluid_names[asciify(fluidscalar_name)] = (index, multi_index)
                self.fluid_names[unicode(fluidscalar_name)] = (index, multi_index)
                self.fluid_names[index, multi_index       ] = (index, multi_index)
        # Additional fluid scalars
        # due to additional degrees of freedom.
        if len(self.fluidvars) > 2:
            # The 𝒫 fluid scalar. Also, if the ς fluid variable exists
            # but is solely used to store 𝒫, mappings for it will not
            # exist yet. Add these as well.
            self.fluid_names[asciify('𝒫')] = (2, 'trace')
            self.fluid_names[unicode('𝒫')] = (2, 'trace')
            self.fluid_names[2, 'trace'  ] = (2, 'trace')
            self.fluid_names[asciify('ς')] = 2
            self.fluid_names[unicode('ς')] = 2
            self.fluid_names[2           ] = 2
        # Also include particle variable names in the fluid_names dict
        self.fluid_names['pos'] = 0
        self.fluid_names['mom'] = 1
        # Assign the fluid variables and scalars as conveniently
        # named attributes on the Component instance.
        # Use the same naming scheme as above.
        try:
            if len(self.fluidvars) > 0:
                self.ϱ   = self.fluidvars[0][0]
            if len(self.fluidvars) > 1:
                self.J   = self.fluidvars[1]
                self.Jx  = self.fluidvars[1][0]
                self.Jy  = self.fluidvars[1][1]
                self.Jz  = self.fluidvars[1][2]
            if len(self.fluidvars) > 2:
                self.ς   = self.fluidvars[2]
                self.𝒫   = self.fluidvars[2]['trace']
                self.ςxx = self.fluidvars[2][0, 0]
                self.ςxy = self.fluidvars[2][0, 1]
                self.ςxz = self.fluidvars[2][0, 2]
                self.ςyx = self.fluidvars[2][1, 0]
                self.ςyy = self.fluidvars[2][1, 1]
                self.ςyz = self.fluidvars[2][1, 2]
                self.ςzx = self.fluidvars[2][2, 0]
                self.ςzy = self.fluidvars[2][2, 1]
                self.ςzz = self.fluidvars[2][2, 2]
        except (IndexError, KeyError):
            pass

    # Function which returns the constant background density
    # ϱ_bar = a**(3*(1 + w_eff(a)))*ρ_bar(a).
    @property
    def ϱ_bar(self):
        # If already set
        if self._ϱ_bar != -1:
            return self._ϱ_bar
        # If the CLASS background is available
        if enable_class_background:
            cosmoresults = compute_cosmo(
                class_call_reason=f'in order to determine ̅ϱ of {self.name}',
            )
            self._ϱ_bar = cosmoresults.ρ_bar(1, self.class_species)
            return self._ϱ_bar
        # Handle simple matter species
        if not (set(self.class_species.split('+')) - {'b', 'cdm'}):
            self._ϱ_bar = 0
            for class_species in self.class_species.split('+'):
                self._ϱ_bar += {'b': Ωb, 'cdm': Ωcdm}[class_species]*ρ_crit
            return self._ϱ_bar
        # Set from assigned particle mass
        if self.representation == 'particles':
            if self.mass == -1:
                abort(
                    f'Cannot determine ϱ_bar for {self.name} '
                    f'with CLASS species {self.class_species} because its (particle) '
                    f'mass is not (yet?) set and enable_class_background is False'
                )
            self._ϱ_bar = self.N*self.mass/boxsize**3
            return self._ϱ_bar
        # Set from assigned fluid densities
        if self.representation == 'fluid':
            if self.mass == -1:
                self._ϱ_bar = (
                    allreduce(np.sum(self.ϱ.grid_noghosts), op=MPI.SUM)
                    /self.gridsize**3
                )
                if self._ϱ_bar == 0:
                    masterwarn(
                        f'Failed to measure ̅ϱ of {self.name}. '
                        f'Try specifying the (fluid element) mass.'
                    )
            else:
                self._ϱ_bar = (self.gridsize/boxsize)**3*self.mass
            return self._ϱ_bar
        abort(
            f'Cannot determine ϱ_bar for {self.name} '
            f'with CLASS species {self.class_species}'
        )

    # Method which returns the decay rate of this component
    @cython.header(
        # Arguments
        a='double',
        # Locals
        class_species=str,
        class_species_canonical=str,
        Γ_class_species='double',
        Γρ_bar='double',
        ρ_bar='double',
        ρ_bar_class_species='double',
        returns='double',
    )
    def Γ(self, a=-1):
        """For components consisting of a combination of
        (CLASS) species, their Γ is defined as
        Γ(a) = (Γ_1*ρ_1_bar(a) + Γ_2*ρ_2_bar(a))/(ρ_1_bar(a) + ρ_2_bar(a)),
        and so Γ is in general time dependent.
        """
        if a == -1:
            a = universals.a
        if self.w_eff_type == 'constant':
            # A constant w_eff implies that the particle/fluid element
            # mass generally given by self.mass*a**(-3*self.w_eff(a=a))
            # is either constant (w = 0) or a power law in a. This is
            # not the behaviour we expect for decaying species.
            return 0
        if not enable_class_background:
            # We cannot compute Γ without the CLASS background
            return 0
        # Compute CLASS background
        cosmoresults = compute_cosmo(class_call_reason=f'in order to determine Γ of {self.name}')
        # Sum up ρ_bar and Γ*ρ_bar
        ρ_bar = Γρ_bar = 0
        for class_species in self.class_species.split('+'):
            class_species_canonical = species_canonical.get(class_species, class_species)
            species_info = species_registered.get(class_species_canonical)
            if species_info is None:
                continue
            Γ_class_species = species_info.Γ(cosmoresults, a)
            ρ_bar_class_species = cosmoresults.ρ_bar(a, class_species)
            ρ_bar += ρ_bar_class_species
            Γρ_bar += Γ_class_species*ρ_bar_class_species
        if Γρ_bar == 0:
            return 0
        return Γρ_bar/ρ_bar

    # This method populate the Component pos/mom arrays (for a
    # particles representation) or the fluid scalar grids (for a
    # fluid representation) with data. It is deliberately designed so
    # that you have to make a call for each scalar quantity (posx, posy,
    # ... for particle components, ϱ, Jx, Jy, ... for fluid components).
    # You should construct the data array within the call itself,
    # as this will minimize memory usage. This data array is 1D for
    # particle data and 3D for fluid data.
    @cython.pheader(
        # Arguments
        data=object,  # 1D/3D (particles/fluid) memoryview
        var=object,   # int-like or str
        multi_index=object,  # tuple, int-like or str
        buffer='bint',
        # Locals
        component_mv='double[:]',
        fluid_indices=object,  # tuple or int-like
        fluidscalar='FluidScalar',
        index='Py_ssize_t',
        mv1D='double[:]',
        mv3D='double[:, :, :]',
        prefix=str,
        suffix=str,
    )
    def populate(self, data, var, multi_index=0, buffer=False):
        """For fluids, the data should not include ghost points.
        If buffer is True, the Δ buffers will be populated
        instead of the data arrays.
        """
        if self.representation == 'particles':
            mv1D = data
            self.N_local = mv1D.shape[0]
            # Enlarge data attributes if necessary
            if self.N_allocated < self.N_local:
                self.resize(self.N_local)
            # Update the data corresponding to the passed string
            prefix, suffix = var[:(len(var) - 1)], var[len(var) - 1]
            component_mv = getattr(self, f'{prefix}_mv3')[:, 'xyz'.index(suffix)]
            component_mv[:mv1D.shape[0]] = mv1D
        elif self.representation == 'fluid':
            mv3D = data
            # The fluid scalar will be given as
            # self.fluidvars[index][multi_index],
            # where index is an int and multi_index is a tuple of ints.
            # These may be given directly as var and multi_index.
            # Alternatively, var may be a str, in which case it can be
            # the name of a fluid variable or a fluid scalar.
            # For each possibility, find index and multi_index.
            if isinstance(var, (int, np.integer)):
                if not (0 <= var < len(self.fluidvars)):
                    abort(
                        f'{self.name} does not have a fluid variable with index {var}'
                    )
                # The fluid scalar is given as
                # self.fluidvars[index][multi_index].
                index = var
            if isinstance(var, str):
                if var not in self.fluid_names:
                    abort(
                        f'{self.name} does not contain a fluid variable '
                        f'with the name "{var}"'
                    )
                # Lookup the fluid indices corresponding to var.
                # This can either be a tuple of the form
                # (index, multi_index) (for a passed fluid scalar name)
                # or just an index (for a passed fluid name).
                fluid_indices = self.fluid_names[var]
                if isinstance(fluid_indices, (int, np.integer)):
                    index = fluid_indices
                else:  # fluid_indices is a tuple
                    if multi_index != 0:
                        masterwarn('Overwriting passed multi_index ({}) with {}, '
                                   'deduced from the passed var = {}.'
                                   .format(multi_index, fluid_indices[1], var)
                                   )
                    index, multi_index = fluid_indices
            # Type check on the multi_index
            if not isinstance(multi_index, (int, np.integer, tuple, str)):
                abort(
                    f'A multi_index of type "{type(multi_index)}" was supplied. '
                    f'This should be either an int, a tuple or a str.'
                )
            # Reallocate fluid grids if necessary
            self.resize(asarray(mv3D).shape)
            # Populate the scalar grid given by index and multi_index
            # with the passed data. This data should not
            # include ghost points.
            fluidscalar = self.fluidvars[index][multi_index]
            if buffer:
                fluidscalar.Δ_noghosts[
                    :mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]
            else:
                fluidscalar.grid_noghosts[
                    :mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]
            # Populate ghost points
            if buffer:
                communicate_ghosts(fluidscalar.Δ_mv, '=')
            else:
                communicate_ghosts(fluidscalar.grid_mv, '=')

    # This method will grow/shrink the data attributes.
    # Note that it will update N_allocated but not N_local.
    @cython.pheader(
        # Arguments
        size_or_shape_noghosts=object,  # Py_ssize_t or tuple
        only_loadable='bint',
        # Locals
        fluidscalar='FluidScalar',
        i='Py_ssize_t',
        s='Py_ssize_t',
        shape_noghosts=tuple,
        size='Py_ssize_t',
        size_old='Py_ssize_t',
        s_old='Py_ssize_t',
    )
    def resize(self, size_or_shape_noghosts, only_loadable=False):
        if self.representation == 'particles':
            size = np.prod(any2list(size_or_shape_noghosts))
            size_old = self.N_allocated
            if size != size_old:
                self.N_allocated = size
                if self.N_allocated == 0:
                    self.N_allocated = 1
                # Reallocate particle data
                if not only_loadable or self.snapshot_vars['load']['pos']:
                    self.pos = realloc(self.pos, 3*self.N_allocated*sizeof('double'))
                    self.pos_mv = cast(self.pos, 'double[:3*self.N_allocated]')
                    self.pos_mv3 = cast(self.pos, 'double[:self.N_allocated, :3]')
                    self.posx = self.pos_mv3[:, 0]
                    self.posy = self.pos_mv3[:, 1]
                    self.posz = self.pos_mv3[:, 2]
                    self.posxˣ = cython.address(self.pos_mv[0:])
                    self.posyˣ = cython.address(self.pos_mv[1:])
                    self.poszˣ = cython.address(self.pos_mv[2:])
                if not only_loadable or self.snapshot_vars['load']['mom']:
                    self.mom = realloc(self.mom, 3*self.N_allocated*sizeof('double'))
                    self.mom_mv = cast(self.mom, 'double[:3*self.N_allocated]')
                    self.mom_mv3 = cast(self.mom, 'double[:self.N_allocated, :3]')
                    self.momx = self.mom_mv3[:, 0]
                    self.momy = self.mom_mv3[:, 1]
                    self.momz = self.mom_mv3[:, 2]
                    self.momxˣ = cython.address(self.mom_mv[0:])
                    self.momyˣ = cython.address(self.mom_mv[1:])
                    self.momzˣ = cython.address(self.mom_mv[2:])
                    self.Δmom = realloc(self.Δmom, 3*self.N_allocated*sizeof('double'))
                    self.Δmom_mv = cast(self.Δmom, 'double[:3*self.N_allocated]')
                    self.Δmom_mv3 = cast(self.Δmom, 'double[:self.N_allocated, :3]')
                    self.Δmomx = self.Δmom_mv3[:, 0]
                    self.Δmomy = self.Δmom_mv3[:, 1]
                    self.Δmomz = self.Δmom_mv3[:, 2]
                    self.Δmomxˣ = cython.address(self.Δmom_mv[0:])
                    self.Δmomyˣ = cython.address(self.Δmom_mv[1:])
                    self.Δmomzˣ = cython.address(self.Δmom_mv[2:])
                    self.Δmom_mv[3*size_old:3*self.N_allocated] = 0
                # Particle IDs
                if self.use_ids:
                    self.ids = realloc(self.ids, self.N_allocated*sizeof('Py_ssize_t'))
                    self.ids_mv = cast(self.ids, 'Py_ssize_t[:self.N_allocated]')
                # Reallocate indices of rungs and jumps
                if self.use_rungs:
                    self.rung_indices = realloc(
                        self.rung_indices,
                        self.N_allocated*sizeof('signed char'),
                    )
                    self.rung_indices_mv = cast(
                        self.rung_indices,
                        'signed char[:self.N_allocated]',
                    )
                    self.rung_indices_mv[size_old:size] = 0  # new particles default to rung 0
                    self.rung_indices_jumped = realloc(
                        self.rung_indices_jumped,
                        self.N_allocated*sizeof('signed char'),
                    )
                    self.rung_indices_jumped_mv = cast(
                        self.rung_indices_jumped,
                        'signed char[:self.N_allocated]',
                    )
                    self.rung_indices_jumped_mv[size_old:size] = 0  # no jumps
        elif self.representation == 'fluid':
            shape_noghosts = tuple(any2list(size_or_shape_noghosts))
            if len(shape_noghosts) == 1:
                shape_noghosts *= 3
            if not any([
                s + 2*nghosts != s_old
                for s, s_old in zip(shape_noghosts, self.shape)
            ]):
                return
            if any([s < 1 for s in shape_noghosts]):
                abort(
                    f'Attempted to resize fluid grids of {self.name} '
                    f'to a shape of {shape_noghosts}'
                )
            # Recalculate and reassign meta data
            self.shape          = tuple([s + 2*nghosts for s in shape_noghosts])
            self.shape_noghosts = shape_noghosts
            self.size           = np.prod(self.shape)
            self.size_noghosts  = np.prod(self.shape_noghosts)
            # Reallocate fluid data
            for fluidscalar in self.iterate_fluidscalars():
                if only_loadable:
                    if fluidscalar.varnum == 0 and not self.snapshot_vars['load']['ϱ']:
                        continue
                    if fluidscalar.varnum == 1 and not self.snapshot_vars['load']['J']:
                        continue
                fluidscalar.resize(shape_noghosts)

    # Method for realisation of one or more variables of the component
    def realize(self, a=-1, a_next=-1, variables=None, multi_indices=None, use_gridˣ=False):
        if not self.is_active(a):
            return
        # Delegate work to the realize() function
        realize(self, a, a_next, variables, multi_indices, use_gridˣ)

    # Method for realising a fluid scalar if it is linear
    def realize_if_linear(self, variable=0, multi_index=None, a=-1, a_next=-1, use_gridˣ=False):
        """If the fluid scalar is not linear or does not exist at all,
        no realisation will be performed and no exception will
        be raised.
        """
        if self.representation == 'particles':
            return
        # Check that the fluid variable exists
        try:
            variable = self.varnames2indices(variable, single=True)
        except (IndexError, KeyError):
            return
        # For all variables other than ϱ (variable == 0),
        # a multi_index has to have been passed.
        if multi_index is None:
            if variable == 0:
                multi_index = 0
            else:
                abort(
                    f'The realize_if_linear() method was called with '
                    f'variable = {variable} ≠ 0 but without any multi_index'
                )
        # Check that the fluid scalar exist
        if multi_index not in self.fluidvars[variable]:
            return
        # Check that the fluid scalar is linear
        if not self.is_linear(variable, multi_index):
            return
        # Do the realisation
        self.realize(a, a_next, variable, multi_index, use_gridˣ)

    # Method for checking whether a given fluid variable
    # or fluid scalar is linear or non-linear.
    def is_linear(self, variable, specific_multi_index=None):
        """When no specific_multi_index is passed, it as assumed that it
        does not matter which fluid scalar of the variable we check
        for linearity (this is not necessarily the case for ς which may
        store the additional "trace" (𝒫) fluid scalar).
        If a variable is passed that does not exist on the component at
        all, this method will return True. Crucially then, the caller
        must never rely on the fact that the variable is linear
        (only that it is not non-linear), unless it is certain that the
        variable does indeed exist.
        """
        if self.representation == 'particles':
            abort(
                f'The is_linear() method was called on the {self.name} particle component, '
                f'for which the concept of linearity does not make sense'
            )
        # Lookup the variable. Return True if it does not exist.
        try:
            variable = self.varnames2indices(variable, single=True)
        except (IndexError, KeyError):
            return True
        # If no specific_multi_index is passed, construct one
        if specific_multi_index is None:
            if variable == 0:
                specific_multi_index = 0
            else:
                specific_multi_index = (0,)*variable
        # Check the linearity
        return self.fluidvars[variable][specific_multi_index].is_linear

    # Method for integrating particle positions/fluid values
    # forward in time.
    # For fluid components, source terms are not included.
    @cython.header(
        # Arguments
        ᔑdt=dict,
        a_next='double',
        # Locals
        indexʳ='Py_ssize_t',
        mom='double*',
        pos='double*',
        rk_order='int',
        scheme=str,
        Δt_over_mass='double',
    )
    def drift(self, ᔑdt, a_next=-1):
        if self.representation == 'particles':
            # The factor a**(3*w_eff) is included below to account for
            # decaying particles, for which the mass is given by
            # a**(-3*w_eff)*self.mass. We should not include this
            # time-varying factor inside the integral, as the reciprocal
            # factor hides inside the momentum (as it is proportional to
            # the mass). Furthermore we should not evaluate this factor
            # at different values dependent on which short-range rung is
            # being drifted, as the factor inside the momentum is only
            # applied once for every base time step (i.e. it is
            # considered a long-range "force" / source term).
            Δt_over_mass = ᔑdt['a**(-2)']*universals.a**(3*self.w_eff(a=universals.a))/self.mass
            # Update positions, taking care of toroidal boundaries
            pos = self.pos
            mom = self.mom
            for indexʳ in range(3*self.N_local):
                pos[indexʳ] = mod(pos[indexʳ] + mom[indexʳ]*Δt_over_mass, boxsize)
            # Some particles may have drifted out of the local domain.
            # Exchange particles between processes.
            exchange(self)
        elif self.representation == 'fluid':
            # Evolve the fluid due to flux terms using the scheme
            # specified in the user parameters.
            scheme = is_selected(self, fluid_scheme_select)
            if scheme == 'maccormack':
                # For the MacCormack scheme to do anything,
                # the J variable must exist.
                if not (
                        self.boltzmann_order == -1
                    or (self.boltzmann_order == 0 and self.boltzmann_closure == 'truncate')
                ):
                    masterprint(
                        f'Evolving fluid variables (flux terms, using the MacCormack scheme) '
                        f'of {self.name} ...'
                    )
                    maccormack(self, ᔑdt, a_next)
                    masterprint('done')
            elif scheme == 'kurganovtadmor':
                # For the Kurganov-Tadmor scheme to do anything,
                # the J variable must exist.
                if not (
                        self.boltzmann_order == -1
                    or (self.boltzmann_order == 0 and self.boltzmann_closure == 'truncate')
                ):
                    rk_order = is_selected(
                        self, fluid_options['kurganovtadmor']['rungekuttaorder'])
                    masterprint(
                        f'Evolving fluid variables (flux terms, using the Kurganov-Tadmor scheme) '
                        f'of {self.name} ...'
                    )
                    kurganov_tadmor(self, ᔑdt, rk_order=rk_order)
                    masterprint('done')
            else:
                abort(
                    f'It was specified that {self.name} should be evolved using '
                    f'the "{scheme}" scheme, which is not implemented.'
                )

    # Method for updating mom due to Δmom
    @cython.header(
        # Arguments
        only_active='bint',
        # Locals
        dim='Py_ssize_t',
        indexᵖ='Py_ssize_t',
        indexʳ='Py_ssize_t',
        indexˣ='Py_ssize_t',
        lowest_active_rung='signed char',
        mom='double*',
        rung_indices='signed char*',
        Δmom='double*',
        returns='void',
    )
    def apply_Δmom(self, only_active=True):
        mom  = self .mom
        Δmom = self.Δmom
        rung_indices = self.rung_indices
        lowest_active_rung = self.lowest_active_rung
        for indexᵖ in range(self.N_local):
            with unswitch:
                if only_active and self.use_rungs:
                    if rung_indices[indexᵖ] < lowest_active_rung:
                        continue
            indexˣ = 3*indexᵖ
            for dim in range(3):
                indexʳ = indexˣ + dim
                mom[indexʳ] += Δmom[indexʳ]

    # Method for converting momentum updates stored in the Δmom buffer
    # to acceleration. The conversion is done in-place.
    @cython.header(
        # Arguments
        ᔑdt_rungs=dict,
        any_rung_jumps='bint',
        # Locals
        conversion_factor='double',
        conversion_factors='double[::1]',
        conversion_factors_ptr='double*',
        dim='Py_ssize_t',
        indexᵖ='Py_ssize_t',
        indexˣ='Py_ssize_t',
        lowest_active_rung='signed char',
        rung_index='signed char',
        rung_index_jumped='signed char',
        rung_indices='signed char*',
        rung_indices_jumped='signed char*',
        w_eff='double',
        Δmom='double*',
        returns='void',
    )
    def convert_Δmom_to_acc(self, ᔑdt_rungs, any_rung_jumps=False):
        # The acceleration is only used for the rung assignment,
        # and so we skip the conversion if this component
        # does not make use of rungs.
        if not self.use_rungs:
            return
        # By "acceleration" is meant
        #   acc = (∂mom/∂t)/(a**2*mass),
        #       → Δmom/(a**(-3*w_eff)*mass*ᔑdt['a**2'])
        #       = Δmom*a**(3*w_eff)/(mass*ᔑdt['a**2']),
        # where the a**(-3*w_eff) takes care of decaying particle mass.
        # As the momentum has the same scaling a**(-3*w_eff), this
        # should not be included under the integral.
        w_eff = self.w_eff(a=universals.a)
        conversion_factors = universals.a**(3*w_eff)/(
            self.mass*(machine_ϵ + ᔑdt_rungs['a**2'])
        )
        conversion_factors_ptr = cython.address(conversion_factors[:])
        # Convert momentum updates to accelerations
        Δmom = self.Δmom
        rung_indices        = self.rung_indices
        rung_indices_jumped = self.rung_indices_jumped
        lowest_active_rung = self.lowest_active_rung
        for indexᵖ in range(self.N_local):
            rung_index = rung_indices[indexᵖ]
            if rung_index < lowest_active_rung:
                continue
            with unswitch:
                if any_rung_jumps:
                    rung_index_jumped = rung_indices_jumped[indexᵖ]
                else:
                    rung_index_jumped = rung_index
            conversion_factor = conversion_factors_ptr[rung_index_jumped]
            indexˣ = 3*indexᵖ
            for dim in range(3):
                Δmom[indexˣ + dim] *= conversion_factor

    # This method computes the rung that particle i should be on,
    # given the acceleration stored in the Δmom buffers.
    @cython.header(
        # Arguments
        indexᵖ='Py_ssize_t',
        rung_factor='double',
        # Locals
        acc2='double',
        indexˣ='Py_ssize_t',
        rung_index_float='double',
        rung_index='signed char',
        returns='signed char',
    )
    def get_rung(self, indexᵖ, rung_factor):
        # The rung_factor argument should be obtained
        # by calling get_rung_factor().
        # This is supplied as an argument rather than called from within
        # this method as it is independent of the particle index.
        indexˣ = 3*indexᵖ
        acc2 = self.Δmomxˣ[indexˣ]**2 + self.Δmomyˣ[indexˣ]**2 + self.Δmomzˣ[indexˣ]**2
        # In the case of zero acceleration, we have no information to
        # determine the rung. Return the rung currently assigned.
        if acc2 == 0:
            return self.rung_indices[indexᵖ]
        # Get the rung index according to the formula
        # stated in get_rung_factor().
        rung_index_float = rung_factor + 0.25*log2(acc2)
        if rung_index_float < 0:
            rung_index = 0
        elif rung_index_float > ℤ[N_rungs - 1]:
            rung_index = ℤ[N_rungs - 1]
        else:
            rung_index = 1 + cast(rung_index_float, 'signed char')
        return rung_index

    # Helper method used for calling the get_rung() method.
    # Specifically, this method returns the rung_factor argument which
    # should be passed to get_rung().
    @cython.header(
        # Arguments
        Δt='double',
        fac_softening='double',
        # Locals
        returns='double',
    )
    def get_rung_factor(self, Δt, fac_softening):
        # Within a full time step Δt, particles on rung 0 may not have
        # an acceleration so large that they accelerate across more than
        # a fraction of their softening length. If so, they should jump
        # up a rung. That is, for particles on rung 0,
        #   0.5*acc*Δt**2 < fac_softening*softening_length.
        # Here the acceleration acc is ∝ (∂mom/∂t)/mass. The exact
        # acc is implemented in convert_Δmom_to_acc(), where momentum
        # updated Δmom and time step integrals ᔑdt[...] are used.
        # Given acc, the largest allowed Δt_rung is then
        #   Δt_rung < sqrt(2*fac_softening*softening_length/acc),
        # where Δt_rung = Δt/2**rung_index is the time step size
        # for rung number rung_index.
        # The rung_index at which a particle ought to be is then
        #   rung_index = log2(Δt/sqrt(2*fac_softening*softening_length/acc))
        #              = 0.5*log2(Δt**2/(2*fac_softening*softening_length))
        #                + 0.25*log2(acc**2)
        #              = rung_factor + 0.25*log2(acc**2),
        # which define rung_factor, the return value of this function.
        # Doubling the acceleration then increases the rung_index by
        # exactly 1. The acceleration then has to change by a factor
        # of 4 for a particle to jump rung.
        return 0.5*log2(Δt**2/(2*fac_softening*self.softening_length))

    # Method for assigning rungs to all particles based on accelerations
    # stored in the Δmom buffer, together with the current time step
    # size Δt and a tuning factor fac_softening. The current rungs will
    # not be taken into consideration, i.e. a particle may "jump"
    # several rungs. Existing values of jumped rung indices are then
    # rendered meaningless and are thus reset.
    @cython.header(
        # Arguments
        Δt='double',
        fac_softening='double',
        # Locals
        indexᵖ='Py_ssize_t',
        rung_factor='double',
        rung_index='signed char',
        rung_indices='signed char*',
        rung_indices_jumped='signed char*',
        rungs_N='Py_ssize_t*',
        returns='void',
    )
    def assign_rungs(self, Δt, fac_softening):
        if not self.use_rungs:
            # When not using rungs, all particles occupy rung 0.
            # The rung indices themselves are not needed.
            self.rungs_N[0] = self.N_local
            return
        rung_indices        = self.rung_indices
        rung_indices_jumped = self.rung_indices_jumped
        rungs_N = self.rungs_N
        # Nullify the rung count
        for rung_index in range(N_rungs):
            rungs_N[rung_index] = 0
        # Assign each particle to a rung
        rung_factor = self.get_rung_factor(Δt, fac_softening)
        for indexᵖ in range(self.N_local):
            rung_index = self.get_rung(indexᵖ, rung_factor)
            rung_indices       [indexᵖ] = rung_index
            rung_indices_jumped[indexᵖ] = rung_index  # no jump
            rungs_N[rung_index] += 1
        # Flag the lowest and highest populated rungs
        self.set_lowest_highest_populated_rung()

    # Method for flagging particle rung jumps to come
    @cython.header(
        # Arguments
        Δt='double',
        Δt_jump_fac='double',
        fac_softening='double',
        ᔑdt_rungs=dict,
        # Locals
        any_rung_jumps='bint',
        indexᵖ='Py_ssize_t',
        integrals='double[::1]',
        integrals_ptr='double*',
        jump_down='signed char',
        jump_up='signed char',
        lowest_active_rung='signed char',
        rung_factor_down='double',
        rung_factor_up='double',
        rung_index='signed char',
        rung_index_ought='signed char',
        rung_indices='signed char*',
        rung_indices_jumped='signed char*',
        rung_jump_down='signed char',
        returns='bint',
    )
    def flag_rung_jumps(self, Δt, Δt_jump_fac, fac_softening, ᔑdt_rungs):
        """This method sets self.rung_indices_jumped[:] equal to
        self.rung_indics[:] in case of no jumps. For jumping up one
        rung, a value of 2*N_rungs is added. For jumping down one rung,
        a value of N_rungs is added. This is chosen so that
        ᔑdt_rungs[integrand][rung_jump] maps to the correct integral.
        """
        any_rung_jumps = False
        if not self.use_rungs:
            return any_rung_jumps
        integrals = ᔑdt_rungs['1']
        integrals_ptr = cython.address(integrals[:])
        rung_indices        = self.rung_indices
        rung_indices_jumped = self.rung_indices_jumped
        lowest_active_rung = self.lowest_active_rung
        rung_factor_up   = self.get_rung_factor(Δt*Δt_jump_fac, fac_softening)
        rung_factor_down = self.get_rung_factor(Δt/Δt_jump_fac, fac_softening)
        jump_up   = ℤ[2*N_rungs]
        jump_down =     N_rungs
        for indexᵖ in range(self.N_local):
            rung_index = rung_indices[indexᵖ]
            # Skip this particle if it is on an inactive rung
            with unswitch:
                if lowest_active_rung > 0:
                    if rung_index < lowest_active_rung:
                        continue
            # A rung may be effectively inactive even if it is above
            # the lowest active rung, due to it being ahead and at
            # a synchronization point. If so it will have a
            # time step size of zero.
            if integrals_ptr[rung_index] == 0:
                continue
            # A particle is always allowed to jump up, but may only jump
            # down every second kick. When a down-jump is disallowed,
            # integrals[rung_index + N_rungs] is set to -1.
            rung_index_ought = self.get_rung(indexᵖ, rung_factor_up)
            if rung_index_ought > rung_index:
                # Signal to jump up
                any_rung_jumps = True
                rung_indices_jumped[indexᵖ] = rung_index + jump_up
            else:
                rung_jump_down = rung_index + jump_down
                if integrals_ptr[rung_jump_down] == -1:
                    continue
                rung_index_ought = self.get_rung(indexᵖ, rung_factor_down)
                if rung_index_ought < rung_index:
                    # Signal to jump down
                    any_rung_jumps = True
                    rung_indices_jumped[indexᵖ] = rung_jump_down
        return any_rung_jumps

    # Method for applying particle rung jumps
    @cython.header(
        # Locals
        indexᵖ='Py_ssize_t',
        rung_index='signed char',
        rung_index_jumped='signed char',
        rung_indices='signed char*',
        rung_indices_jumped='signed char*',
        rungs_N='Py_ssize_t*',
        returns='void',
    )
    def apply_rung_jumps(self):
        if not self.use_rungs:
            return
        # Here, "applying" rung jumps means updating rung_indices
        # and rungs_N based on rung_indices_jumped.
        # Also, rung_indices_jumped is set equal to (the updated)
        # run_indices, signalling no further jumps.
        rung_indices        = self.rung_indices
        rung_indices_jumped = self.rung_indices_jumped
        rungs_N = self.rungs_N
        for indexᵖ in range(self.N_local):
            rung_index_jumped = rung_indices_jumped[indexᵖ]
            if rung_index_jumped < N_rungs:
                continue
            # Particle has undergone a rung jump
            rung_index = rung_indices[indexᵖ]
            rungs_N[rung_index] -= 1
            rung_index += 2*(rung_index_jumped >= ℤ[2*N_rungs]) - 1  # += 1 (jump up), += -1 (jump down)
            rungs_N[rung_index] += 1
            rung_indices       [indexᵖ] = rung_index
            rung_indices_jumped[indexᵖ] = rung_index  # done with jump
        # Flag the lowest and highest populated rungs
        self.set_lowest_highest_populated_rung()

    # Method for setting rungs_N from rung_indices
    @cython.header(
        # Locals
        indexᵖ='Py_ssize_t',
        rung_index='signed char',
        rung_indices='signed char*',
        rungs_N='Py_ssize_t*',
        returns='void',
    )
    def set_rungs_N(self):
        rung_indices = self.rung_indices
        rungs_N = self.rungs_N
        for rung_index in range(N_rungs):
            rungs_N[rung_index] = 0
        for indexᵖ in range(self.N_local):
            rung_index = rung_indices[indexᵖ]
            rungs_N[rung_index] += 1
        # Flag the lowest and highest populated rungs
        self.set_lowest_highest_populated_rung()

    # Method for setting the (local) lowest and highest populated rung
    @cython.header(
        # Locals
        highest_populated_rung='signed char',
        lowest_populated_rung='signed char',
        rung_index='signed char',
        returns='void',
    )
    def set_lowest_highest_populated_rung(self):
        lowest_populated_rung = ℤ[N_rungs - 1]
        highest_populated_rung = 0
        for rung_index in range(N_rungs):
            if self.rungs_N[rung_index] > 0:
                lowest_populated_rung = rung_index
                break
        for rung_index in range(N_rungs - 1, -1, -1):
            if self.rungs_N[rung_index] > 0:
                highest_populated_rung = rung_index
                break
        self.lowest_populated_rung = lowest_populated_rung
        self.highest_populated_rung = highest_populated_rung

    # Method for initialising a tiling
    @cython.header(
        # Arguments
        tiling_name=str,
        initial_rung_size=object,  # sequence of length N_rungs or int-like
        # Locals
        tiling='Tiling',
        returns='Tiling',
    )
    def init_tiling(self, tiling_name, initial_rung_size=-1):
        # Do nothing if the (sub)tiling is already initialised
        # on this component.
        tiling = self.tilings.get(tiling_name)
        if tiling is not None:
            return tiling
        # Initialise the (sub)tiling on this component
        tiling = init_tiling(self, tiling_name, initial_rung_size)
        self.tilings[tiling_name] = tiling
        return tiling

    # Method for sorting particles according to short-range tiles
    @cython.header(
        # Arguments
        tiling_name=str,
        subtiling_name=str,
        # Locals
        N_subtiles='Py_ssize_t',
        count='Py_ssize_t',
        data_quantity='double*',
        dim='int',
        dim_quantity='Py_ssize_t',
        highest_populated_rung='signed char',
        ids='Py_ssize_t*',
        indexᵖ='Py_ssize_t',
        indexʳ='Py_ssize_t',
        indexˣ='Py_ssize_t',
        indexˣ_tmp='Py_ssize_t',
        lowest_populated_rung='signed char',
        quantity='int',
        quantityx='double*',
        quantityy='double*',
        quantityz='double*',
        rung='Py_ssize_t*',
        rung_indices='signed char*',
        rung_indices_jumped='signed char*',
        rung_particle_index='Py_ssize_t',
        rungs_N='Py_ssize_t*',
        subtile='Py_ssize_t**',
        subtile_index='Py_ssize_t',
        subtiles='Py_ssize_t***',
        subtiles_contain_particles='signed char*',
        subtiles_rungs_N='Py_ssize_t**',
        subtiling='Tiling',
        tile_extent='double[::1]',
        tile_index='Py_ssize_t',
        tile_index3D='Py_ssize_t*',
        tiles_contain_particles='signed char*',
        tiling='Tiling',
        tiling_location='double[::1]',
        tiling_plural=str,
        tiling_names=object,  # list, collections.Counter, str
        tmp_ids='Py_ssize_t*',
        tmp_ids_mv='Py_ssize_t[::1]',
        tmp_rung_indices='signed char*',
        tmp_rung_indices_mv='signed char[::1]',
        tmp_quantity='double*',
        returns='void',
    )
    def tile_sort(self, tiling_name, subtiling_name=''):
        # Get the tiling from the passed tiling_name
        tiling = self.tilings.get(tiling_name)
        if tiling is None:
            # This tiling has not yet been instantiated
            # on this component. Do it now.
            tiling = self.init_tiling(tiling_name)
        # Perform tile sort
        tiling.sort()
        # When a subtiling_name is supplied, this signals an in-memory
        # sorting of the data arrays of the particles. The final memory
        # order will match that of the particle visiting order when
        # iterating over the tiles and subtiles.
        if not subtiling_name:
            return
        tiling_names = gather(tiling_name)
        if master:
            tiling_names = collections.Counter(tiling_names)
            tiling_names = sorted(tiling_names,
                key=(lambda key, tiling_names=tiling_names: (-tiling_names[key], key)),
            )
            tiling_plural = '' if len(tiling_names) == 1 else 's'
            tiling_names = '/'.join([
                tiling_name_i.removesuffix(' (tiles)') for tiling_name_i in tiling_names
            ])
            masterprint(
                f'Reordering {self.name} particles in memory according to '
                f'the {tiling_names} tiling{tiling_plural} ...',
            )
        # Extract variables
        lowest_populated_rung  = self.lowest_populated_rung
        highest_populated_rung = self.highest_populated_rung
        tiles_contain_particles = tiling.contain_particles
        tiling_location         = tiling.location
        tile_extent             = tiling.tile_extent
        subtiling = self.tilings.get(subtiling_name)
        if subtiling is None:
            subtiling = self.init_tiling(subtiling_name)
        subtiles                   = subtiling.tiles
        N_subtiles                 = subtiling.size
        subtiles_contain_particles = subtiling.contain_particles
        # Iterate over the tiles and subtiles while keeping a counter
        # keeping track of the particle visiting number. We copy the
        # particle data (positions, momenta, IDs, rung indices) to a
        # temporary buffer using the visiting order. After the iteration
        # we then copy this sorted buffer into the original data arrays.
        # For the 64-bit data (positions, momenta, IDs) we use Δmom as
        # the temporary buffer, as this should not store any data at the
        # time of calling this method. For the rungs, we use the
        # rung_indices_arr buffer from the communication module, which
        # we enlarge if needed. As the Δmom buffer can only store one
        # quantity at a time (i.e. not all positions and all momenta
        # and all IDs), we iterate over the tiles and subtiles several
        # times: 2 times if not using IDs and 3 times if using IDs,
        # with the first iteration handling the IDs. Importantly, the
        # positions and rung indices must be sorted during the last
        # iteration, as the iteration itself depends on these
        # through subtiling.sort().
        tmp_quantity = self.Δmom
        tmp_ids_mv = asarray(self.Δmom_mv).view(C2np['Py_ssize_t'])
        tmp_ids = cython.address(tmp_ids_mv[:])
        ids = self.ids
        rung_indices        = self.rung_indices
        rung_indices_jumped = self.rung_indices_jumped
        if self.use_rungs and rung_indices_arr.shape[0] < self.N_local:
            rung_indices_arr.resize(self.N_local, refcheck=False)
        tmp_rung_indices_mv = rung_indices_arr
        tmp_rung_indices = cython.address(tmp_rung_indices_mv[:])
        for quantity in range(int(not self.use_ids), 3):
            if quantity == 1:
                data_quantity = self.mom
            elif quantity == 2:
                data_quantity = self.pos
            count = 0
            # Loop over all tiles
            for tile_index in range(tiling.size):
                if tiles_contain_particles[tile_index] == 0:
                    continue
                # Sort particles within the tile into subtiles
                tile_index3D = tiling.tile_index3D(tile_index)
                for dim in range(3):
                    tile_location[dim] = tiling_location[dim] + tile_index3D[dim]*tile_extent[dim]
                subtiling.relocate(tile_location)
                subtiling.sort(tiling, tile_index)
                subtiles_rungs_N = subtiling.tiles_rungs_N
                # Loop over all subtiles in the tile
                for subtile_index in range(N_subtiles):
                    if subtiles_contain_particles[subtile_index] == 0:
                        continue
                    subtile = subtiles[subtile_index]
                    rungs_N = subtiles_rungs_N[subtile_index]
                    # Loop over all rungs in the subtile
                    for rung_index in range(lowest_populated_rung, ℤ[highest_populated_rung + 1]):
                        rung_N = rungs_N[rung_index]
                        if rung_N == 0:
                            continue
                        rung = subtile[rung_index]
                        # Loop over all particles in the rung
                        for rung_particle_index in range(rung_N):
                            indexᵖ = rung[rung_particle_index]
                            # Copy the data to the temporary buffers
                            with unswitch(4):
                                if quantity == 0:
                                    # IDs
                                    tmp_ids[count] = ids[indexᵖ]
                                else:
                                    # Momenta and positions
                                    indexˣ = 3*indexᵖ
                                    indexˣ_tmp = 3*count
                                    for dim_quantity in range(3):
                                        tmp_quantity[indexˣ_tmp + dim_quantity] = (
                                            data_quantity[indexˣ + dim_quantity]
                                        )
                                    # Rungs
                                    with unswitch(4):
                                        if quantity == 2 and self.use_rungs:
                                            tmp_rung_indices[count] = rung_indices[indexᵖ]
                            count += 1
            # Copy the sorted data back into the data arrays
            if quantity == 0:
                # IDs
                for indexᵖ in range(self.N_local):
                    ids[indexᵖ] = tmp_ids[indexᵖ]
            else:
                # Momenta and positions
                for indexʳ in range(3*self.N_local):
                    data_quantity[indexʳ] = tmp_quantity[indexʳ]
            if quantity == 2 and self.use_rungs:
                # Rungs
                for indexᵖ in range(self.N_local):
                    rung_index = tmp_rung_indices[indexᵖ]
                    rung_indices       [indexᵖ] = rung_index
                    rung_indices_jumped[indexᵖ] = rung_index  # no jump
        # Finally we need to re-sort the tiling
        tiling.sort()
        masterprint('done')

    # Method for integrating fluid values forward in time
    # due to "internal" source terms, meaning source terms that do not
    # result from interacting with other components.
    @cython.header(
        # Arguments
        ᔑdt=dict,
        a_next='double',
        # Locals
        J_dim='double*',
        J_dim_fluidscalar='FluidScalar',
        a='double',
        dim='int',
        index='Py_ssize_t',
        indexʳ='Py_ssize_t',
        mom='double*',
        mom_decay_factor='double',
        scheme=str,
    )
    def apply_internal_sources(self, ᔑdt, a_next=-1):
        # Decaying components should have their momentum reduced due
        # to loss of mass. As the mass at time a is given by
        # self.mass*a**(-3*self.w_eff(a)),
        # the mass (and hence momentum) reduction over the time step
        # can be written as the ratio of this expression evaluated
        # at both ends of the time step.
        # Note that this will keep the velocity
        # au = a²ẋ = a²dx/dt = q/m
        # constant with respect to the decay (external forces like
        # gravity will of course change this velocity).
        # For fluid components, the conserved momentum is
        # J = a⁴(ρ + P)u = (ϱ + c⁻²𝒫)*a**(-3*w_eff)*au.
        # For species with w_eff = 0 (and hence w = 0), we indeed have
        # constant au. When w_eff != 0 due to decay, J is still
        # conserved while au is not, which is wrong. Thus we need the
        # same correction factor on fluids.
        # Importantly, this prescription breaks down for stable
        # components which nonetheless have non-zero w_eff. This ought
        # to be sorted out, which would require separating the two
        # effects (changing equation of state and decay) so that they
        # are not both relayed by w_eff. For now we check this explicitly
        # and emit a warning if an inconsistent component is used.
        # Currently the only decaying CLASS species implemented is dcdm.
        # We really should include dr as well, which now gains
        # mass/energy rather than loose it. It however has
        # w_eff = 1/3 != 0 and so we cannot distinguish between changes
        # in mass due to decay of dcdm and due to redshifting.
        a = universals.a
        if a_next == -1:
            mom_decay_factor = 1
        else:
            mom_decay_factor = a**(3*self.w_eff(a=a))/a_next**(3*self.w_eff(a=a_next))
        if mom_decay_factor != 1 and 'dcdm' in self.class_species.split('+'):
            # Check that the species of this component are legal
            legal_class_species = {'dcdm', 'cdm', 'b'}
            if not all([class_species in legal_class_species
                for class_species in self.class_species.split('+')]):
                masterwarn(
                    f'Currently you must not mix species with non-zero w_eff together with '
                    f'decaying species in a single component. This is the case for '
                    f'{component.name} with CLASS species "{self.class_species}".'
                )
            # Reduce all momenta, corresponding to the loss of mass
            if self.representation == 'particles':
                mom = self.mom
                for indexʳ in range(3*self.N_local):
                    mom[indexʳ] *= mom_decay_factor
            elif self.representation == 'fluid' and not self.is_linear(1):
                for dim in range(3):
                    J_dim_fluidscalar = self.J[dim]
                    J_dim = J_dim_fluidscalar.grid
                    for index in range(self.size):
                        J_dim[index] *= mom_decay_factor
        # Representation specific internal source terms below
        if self.representation == 'particles':
            # Particle components have no special internal source terms
            pass
        elif self.representation == 'fluid':
            # Below follow internal source terms not taken into account
            # by the MacCormack/Kurganov-Tadmor methods.
            scheme = is_selected(self, fluid_scheme_select)
            if scheme == 'maccormack':
                # For the MacCormack scheme we have three internal
                # source terms: The Hubble term in the continuity
                # equation and the pressure and shear terms in
                # the Euler equation.
                if (
                    (   # The Hubble term
                            self.boltzmann_order > -1
                        and not self.approximations['P=wρ']
                        and enable_Hubble
                    )
                    or
                    (   # The pressure term
                            self.boltzmann_order > 0
                        and not (self.w_type == 'constant' and self.w_constant == 0)
                    )
                    or
                    (
                        # The shear term
                            self.boltzmann_order > 1
                        or (self.boltzmann_order == 1 and self.boltzmann_closure == 'class')
                    )
                ):
                    masterprint(
                        f'Evolving fluid variables (internal source terms) of {self.name} ...'
                    )
                    maccormack_internal_sources(self, ᔑdt, a_next)
                    masterprint('done')
            elif scheme == 'kurganovtadmor':
                # Only the Hubble term in the continuity equation
                # exist as an internal source term when using
                # the Kurganov-Tadmor scheme.
                if (
                    # The Hubble term
                        self.boltzmann_order > -1
                    and not self.approximations['P=wρ']
                    and enable_Hubble
                ):
                    masterprint(
                        f'Evolving fluid variables (internal source terms) of {self.name} ...'
                    )
                    kurganov_tadmor_internal_sources(self, ᔑdt)
                    masterprint('done')
            else:
                abort(
                    f'It was specified that the {self.name} component should be evolved using '
                    f'the "{scheme}" scheme, which is not implemented.'
                )

    # Method for testing whether this component is
    # constituted solely of matter-like species.
    def is_matter_like(self):
        return not (
            {
                class_species.strip()
                for class_species in self.class_species.split('+')
            } - {
                class_species.strip()
                for class_species in matter_class_species.split('+')
            }
        )

    # Method for computing the equation of state parameter w
    # at a certain time t or value of the scale factor a.
    # This has to be a pure Python function, otherwise it cannot
    # be called as w(a=a).
    def w(self, *, t=-1, a=-1):
        """This method should not be called before w has been
        initialised by the init_w method.
        """
        # If no time or scale factor value is passed,
        # use the current time and scale factor value.
        if t == -1 == a:
            t = universals.t
            a = universals.a
        # Compute w dependent on its type
        if self.w_type == 'constant':
            value = self.w_constant
        elif self.w_type == 'tabulated (t)':
            if t == -1:
                t = cosmic_time(a)
            value = self.w_spline.eval(t)
        elif self.w_type == 'tabulated (a)':
            if a == -1:
                a = scale_factor(t)
            value = self.w_spline.eval(a)
        elif self.w_type == 'expression':
            if t == -1:
                t = cosmic_time(a)
            elif a == -1:
                a = scale_factor(t)
            units_dict['t'] = t
            units_dict['a'] = a
            value = eval_unit(self.w_expression, units_dict)
            units_dict.pop('t')
            units_dict.pop('a')
        else:
            abort(f'Did not recognise w type "{self.w_type}"')
        # For components with a non-linear evolution of ϱ,
        # we cannot handle w ≤ -1, as (ϱ + c⁻²𝒫) becomes non-positive.
        # This really should not be a problem, but the current fluid
        # implementation computes J/(ϱ + c⁻²𝒫) while solving the
        # continuity equation. If what is being run is not a simulation
        # but the class utility, this is not a problem as the system
        # is not to be evolved.
        if value <= -1 and special_params.get('special') != 'class':
            if (
                    (   self.boltzmann_order > 0
                     or (self.boltzmann_order == 0 and self.boltzmann_closure == 'class'))
                and (a > universals.a_begin or t > universals.t_begin)
            ):
                if t == -1:
                    t = cosmic_time(a)
                elif a == -1:
                    a = scale_factor(t)
                abort(
                    f'The equation of state parameter w for {self.name} took on the value '
                    f'{value} ≤ -1 at t = {t} {unit_time}, a = {a}. '
                    f'Such phantom w is not currently allowed for components with non-linear ϱ.'
                )
        # For components with a non-linear evolution of J,
        # we cannot handle w < 0, as the sound speed c*sqrt(w) becomes
        # negative. If what is being run is not a simulation
        # but the class utility, this is not a problem as the system
        # is not to be evolved.
        if value < 0 and special_params.get('special') != 'class':
            if self.boltzmann_order > 0 and (a > universals.a_begin or t > universals.t_begin):
                if t == -1:
                    t = cosmic_time(a)
                elif a == -1:
                    a = scale_factor(t)
                abort(
                    f'The equation of state parameter w for {self.name} took on the value '
                    f'{value} < 0 at t = {t} {unit_time}, a = {a}. '
                    f'This is disallowed for components with non-linear J.'
                )
        return value

    # Method for computing the effective equation of state parameter
    # w_eff at a certain time t or value of the scale factor a.
    # This has to be a pure Python function,
    # otherwise it cannot be called as w_eff(a=a).
    def w_eff(self, *, t=-1, a=-1):
        """This method should not be called before w_eff has been
        initialised by the init_w_eff method.
        """
        # For constant w_eff, w_eff = w
        if self.w_eff_type == 'constant':
            return self.w(t=t, a=a)
        elif self.w_eff_type == 'tabulated (a)':
            # If no time or scale factor value is passed,
            # use the current time and scale factor value.
            if t == -1 == a:
                t = universals.t
                a = universals.a
            # Compute w_eff
            if a == -1:
                a = scale_factor(t)
            return self.w_eff_spline.eval(a)
        abort(f'Did not recognise w_eff type "{self.w_eff_type}"')

    # Method for computing the proper time derivative
    # of the equation of state parameter w
    # at a certain time t or value of the scale factor a.
    # This has to be a pure Python function,
    # otherwise it cannot be called as ẇ(a=a).
    def ẇ(self, *, t=-1, a=-1):
        """This method should not be called before w has been
        initialised by the init_w method.
        """
        # If no time or scale factor value is passed,
        # use the current time and scale factor value.
        if t == -1 == a:
            t = universals.t
            a = universals.a
        # Compute ẇ dependent on its type
        if self.w_type == 'constant':
            return 0
        if self.w_type == 'tabulated (t)':
            if t == -1:
                t = cosmic_time(a)
            return self.w_spline.eval_deriv(t)
        if self.w_type == 'tabulated (a)':
            # Here we use dw/dt = da/dt*dw/da
            if a == -1:
                a = scale_factor(t)
            return ȧ(a)*self.w_spline.eval_deriv(a)
        if self.w_type == 'expression':
            # Approximate the derivative via symmetric difference
            if t == -1:
                t = cosmic_time(a)
            elif a == -1:
                a = scale_factor(t)
            Δx = 1e+6*machine_ϵ
            units_dict['t'] = t - Δx
            units_dict['a'] = a - Δx
            w_before = eval_unit(self.w_expression, units_dict)
            units_dict['t'] = t + Δx
            units_dict['a'] = a + Δx
            w_after = eval_unit(self.w_expression, units_dict)
            units_dict.pop('t')
            units_dict.pop('a')
            return (w_after - w_before)/(2*Δx)
        abort(f'Did not recognise w type "{self.w_type}"')

    # Method for computing the proper time derivative
    # of the effective equation of state parameter w_eff
    # at a certain time t or value of the scale factor a.
    # This has to be a pure Python function,
    # otherwise it cannot be called as ẇ_eff(a=a).
    def ẇ_eff(self, *, t=-1, a=-1):
        """This method should not be called before w_eff has been
        initialised by the init_w_eff method.
        """
        # Compute ẇ_eff dependent on its type
        if self.w_eff_type == 'constant':
            return 0
        if self.w_eff_type == 'tabulated (a)':
            # If no time or scale factor value is passed,
            # use the current time and scale factor value.
            if t == -1 == a:
                t = universals.t
                a = universals.a
            # Compute ẇ_eff. Here we use dw_eff/dt = da/dt*dw_eff/da.
            if a == -1:
                a = scale_factor(t)
            return ȧ(a)*self.w_eff_spline.eval_deriv(a)
        abort(f'Did not recognise w_eff type "{self.w_type}"')

    # Method which initialises the equation of state parameter w.
    # Call this before calling the w and ẇ methods.
    @cython.header(
        # Arguments
        w=object,  # float-like, str or dict
        # Locals
        char=str,
        char_last=str,
        class_species=str,
        delim_left=str,
        delim_right=str,
        done_reading_w='bint',
        i='int',
        i_tabulated='double[:]',
        key=str,
        line=str,
        p_tabulated=object,  # np.ndarray
        pattern=str,
        unit='double',
        w_data='double[:, :]',
        w_list=list,
        w_ori=str,
        w_tabulated='double[:]',
        w_values='double[::1]',
        ρ_tabulated=object,  # np.ndarray
        returns='Spline',
    )
    def init_w(self, w):
        """The w argument can be one of the following (Python) types:
        - float-like: Designates a constant w.
                      The w will be stored in self.w_constant and
                      self.w_type will be set to 'constant'.
        - str       : If w is given as a str, it can mean any of
                      four things:
                      - w may be the string 'CLASS'.
                      - w may be the string 'default'.
                      - w may be a filename.
                      - w may be some analytical expression.
                      If w == 'CLASS', CLASS should be used to compute
                      w throughout time. The value of class_species
                      will be used to pick out w(a) for the correct
                      species. The result from CLASS are tabulated
                      w(a), where the a values will be stored as
                      self.w_tabulated[0, :], while the tabulated values
                      of w will be stored as self.w_tabulated[1, :].
                      The self.w_type will be set to 'tabulated (a)'.
                      If w == 'default', a constant w will be assigned
                      based on the species.
                      If w is a filename, the file should contain
                      tabulated values of w and either t or a. The file
                      must be in a format understood by np.loadtxt,
                      and a header must be present stating whether w(t)
                      or w(a) is tabulated. For w(t), the header should
                      also specify the units for the t column.
                      Legal formats for the header are:
                      # a    w(a)
                      # a    w
                      # t [Gyr]    w
                      # t [Gyr]    w(t)
                      In addition, the two columns may be specified in
                      the reverse order, parentheses and brackets may be
                      replaced with any of (), [], {}, <> and the number
                      of spaces/tabs does not matter.
                      The tabulated values for t or a will be stored as
                      self.w_tabulated as described above when using
                      CLASS, though self.w_type will be set to
                      'tabulated (t)' or 'tabulated (a)' depending on
                      whether t or a is used.
                      If w is some arbitrary expression,
                      this should include either t or a.
                      The w will be stored in self.w_expression and
                      self.w_type will be set to 'expression'.
        - dict      : The dict has to be of the form
                      {'t': iterable, 'w': iterable} or
                      {'a': iterable, 'w': iterable}, where the
                      iterables are some iterables of matching
                      tabulated values.
                      The tabulated values for t or a will be stored as
                      self.w_tabulated[0, :], the tabulated values for w
                      as self.w_tabulated[1, :] and self.w_type will be
                      set to 'tabulated (t)' or 'tabulated (a)'.
        """
        # Initialise w dependent on its type
        try:
            w = float(w)
        except Exception:
            pass
        if isinstance(w, (float, np.floating)):
            # Assign passed constant w
            self.w_type = 'constant'
            self.w_constant = w
        elif isinstance(w, str) and w.lower() == 'class':
            # Get w as P_bar/ρ_bar from CLASS
            if enable_class_background:
                self.w_type = 'tabulated (a)'
                cosmoresults = compute_cosmo(
                    class_call_reason=f'in order to determine w(a) of {self.name}',
                )
                background = cosmoresults.background
                i_tabulated = background['a']
                # For combination species it is still true that
                # w = c⁻²P_bar/ρ_bar, with P_bar and ρ_bar the sum of
                # individual background pressures and densities.
                # Note that the quantities in the background dict is
                # given in CLASS units, specifically c = 1.
                ρ_tabulated = 0
                p_tabulated = 0
                for class_species in self.class_species.split('+'):
                    key = f'(.)rho_{class_species}'
                    if key not in background:
                        abort(
                            f'No background density {key} for CLASS species "{class_species}" '
                            f'present in the CLASS background.'
                        )
                    ρ_tabulated += background[key]
                    key = f'(.)p_{class_species}'
                    if key not in background:
                        abort(
                            f'No background pressure {key} for CLASS species "{class_species}" '
                            f'present in the CLASS background.'
                        )
                    p_tabulated += background[key]
                w_tabulated = p_tabulated/ρ_tabulated
            elif self.is_matter_like():
                # Directly assign w = 0 for matter-like components
                # when not using the CLASS background.
                self.w_type = 'constant'
                self.w_constant = 0
            else:
                abort(
                    f'Attempted to call CLASS to get the equation of state parameter w for '
                    f'{self.name} with CLASS species "{self.class_species}", '
                    f'but enable_class_background is False.'
                )
        elif isinstance(w, str) and w.lower() == 'default':
            # Assign w a constant value based on the species.
            # For combination species, the combined w is a weighted sum
            # of the individual w's with the individual background
            # densities as weight, or equivalently, the ratio of the sum
            # of the individual background pressures and the sum of the
            # individual background densities. To do it this proper way,
            # w should be passed in as 'class'. For w == 'default',
            # we simply do not handle this case.
            self.w_type = 'constant'
            try:
                self.w_constant = float(species_registered[self.species].w)
            except Exception:
                if self.representation == 'particles':
                    self.w_constant = 0
                else:
                    abort(f'No default, constant w is defined for species "{self.species}"')
        elif isinstance(w, str) and os.path.isfile(w):
            # Load tabulated w from file
            self.w_type = 'tabulated (?)'
            # Let only the master process read in the file
            if master:
                w_data = np.loadtxt(w)
                # Transpose w_data so that it consists of two rows
                w_data = asarray(w_data).T
                # For varying w it is crucial to know whether w is a
                # function of t or a.
                # This should be written in the header of the file.
                pattern = r'[a-zA-Z_]*\s*(?:\(\s*[a-zA-Z0-9\._]+\s*\))?'
                done_reading_w = False
                with open_file(w, mode='r', encoding='utf-8') as w_file:
                    while True:
                        line = w_file.readline().lstrip()
                        if line and not line.startswith('#'):
                            break
                        line = line.replace('#', '').replace('\t', ' ').replace('\n', ' ')
                        for delim_left, delim_right in zip('[{<', ']}>'):
                            line = line.replace(delim_left , '(')
                            line = line.replace(delim_right, ')')
                        match = re.search(
                            r'\s*({pattern})\s+({pattern})\s*(.*)'.format(pattern=pattern),
                            line,
                        )
                        if match and not match.group(3):
                            # Header line containing the relevant
                            # information found.
                            for i in range(2):
                                var = match.group(1 + i)
                                if var[0] == 't':
                                    self.w_type = 'tabulated (t)'
                                    unit_match = re.search(r'\((.*)\)', var)  # Applied later
                                elif var[0] == 'a':
                                    self.w_type = 'tabulated (a)'
                                elif var[0] == 'w':
                                    # Extract the two rows
                                    w_tabulated = w_data[i, :]
                                    i_tabulated = w_data[(i + 1) % 2, :]
                                    done_reading_w = True
                        if done_reading_w and '(?)' not in self.w_type:
                            break
                # Multiply unit on the time
                if '(t)' in self.w_type:
                    if unit_match and unit_match.group(1) in units_dict:
                        unit = eval_unit(unit_match.group(1))
                        i_tabulated = asarray(i_tabulated)*unit
                    elif unit_match:
                        abort(
                            f'Time unit "{unit_match.group(1)}" in header of "{w}" not understood'
                        )
                    else:
                        abort(f'Could not find time unit in header of "{w}"')
        elif isinstance(w, str):
            # Some expression for w was passed.
            # Insert '*' between all numbers and letters as well as all
            # between numbers and opening parentheses and between
            # closing parentheses and numbers/letters.
            w_ori = w
            w = w.lower().replace(' ', '').replace('_', '')
            w_list = []
            char_last = ''
            for char in w:
                if (
                    (
                        re.search(r'[0-9.]', char_last)
                    and (re.search(r'\w', char) and not re.search(r'[0-9.]', char))
                    )
                    or
                    (
                        re.search(r'[0-9.]', char)
                    and (re.search(r'\w', char_last) and not re.search(r'[0-9.]', char_last))
                    )
                    or
                    (
                        re.search(r'[0-9.]', char_last) and re.search(r'\(', char)
                    )

                    or
                    (
                        re.search(r'\)', char_last) and re.search(r'\w', char)
                    )
                ):
                    w_list.append('*')
                w_list.append(char)
                char_last = char
            w = ''.join(w_list)
            # Save the passed expression for w
            self.w_type = 'expression'
            self.w_expression = w
            # Test that the expression is parsable
            try:
                self.w()
            except Exception:
                abort(
                    f'Cannot parse w = "{w_ori}" as an expression '
                    f'and no file with that name can be found either.'
                )
        elif isinstance(w, dict):
            # Use the tabulated w given by the two dict key-value pairs
            self.w_type = 'tabulated (?)'
            for key, val in w.items():
                if key.startswith('t'):
                    self.w_type = 'tabulated (t)'
                    i_tabulated = asarray(val, dtype=C2np['double'])
                elif key.startswith('a'):
                    self.w_type = 'tabulated (a)'
                    i_tabulated = asarray(val, dtype=C2np['double'])
                elif key.startswith('w'):
                    w_tabulated = asarray(val, dtype=C2np['double'])
        else:
            abort('Cannot handle w of type {}'.format(type(w)))
        # In the case of a tabulated w, only the master process
        # have the needed i_tabulated and w_tabulated. Broadcast these
        # to the slaves and instantiate a spline on all processes.
        if 'tabulated' in self.w_type:
            if master:
                # Check that tabulated values have been found
                if '(?)' in self.w_type:
                    abort(
                        'Could not detect the independent variable in tabulated '
                        'w data (should be \'a\' or \'t\')'
                    )
                # Make sure that the values of i_tabulated
                # are in increasing order.
                order = np.argsort(i_tabulated)
                i_tabulated = asarray(i_tabulated)[order]
                w_tabulated = asarray(w_tabulated)[order]
                # Pack the two rows together
                self.w_tabulated = empty((2, w_tabulated.shape[0]), dtype=C2np['double'])
                self.w_tabulated[0, :] = i_tabulated
                self.w_tabulated[1, :] = w_tabulated
            # Broadcast the tabulated w
            self.w_type = bcast(self.w_type)
            self.w_tabulated = smart_mpi(self.w_tabulated if master else (), mpifun='bcast')
            # If the tabulated w is constant, treat it as such
            w_values = asarray(tuple(set(self.w_tabulated[1, :])))
            if isclose(min(w_values), max(w_values), abs_tol=1e-9):
                self.w_type = 'constant'
                self.w_constant = self.w_tabulated[1, 0]
            else:
                # Construct a Spline object from the tabulated data.
                # For most physical species, w(a) is approximately a
                # power law in a (and thus also approximately in t)
                # and so a log-log spline should be used.
                logx, logy = True, True
                if np.any(asarray(self.w_tabulated[1, :]) <= 0):
                    logy = False
                if self.class_species == 'fld':
                    # The CLASS dark energy fluid (fld) uses
                    # the {w_0, w_a} parametrisation, and so a linear
                    # spline should be used.
                    logx, logy = False, False
                self.w_spline = Spline(self.w_tabulated[0, :], self.w_tabulated[1, :],
                    f'w{self.w_type[len(self.w_type) - 3:]} of {self.name}',
                    logx=logx, logy=logy)

    # Method which initialises the effective
    # equation of state parameter w_eff.
    # Call this before calling the w_eff method,
    # but after calling the init_w method.
    @cython.header(
        # Locals
        a='double',
        a_min='double',
        a_tabulated=object,  # np.ndarray
        integrand_spline='Spline',
        integrand_tabulated='double[::1]',
        t='double',
        t_tabulated='double[::1]',
        ρ_bar_tabulated=object,  # np.ndarray
    )
    def init_w_eff(self):
        """This method initialises the effective equation of state
        parameter w_eff by defining the w_eff_spline attribute,
        which is used by the w_eff method to get w_eff(a).
        Only future times compared to universals.a will be included.
        The definition of w_eff is
        ϱ = a**(3(1 + w_eff))ρ
        such that mean(ϱ) = ϱ_bar is constant in time. This leads to
        w_eff(a) = log(ρ_bar(a=1)/ρ_bar)/(3*log(a)) - 1,
        which is well behaved for any positive definite ρ_bar.
        """
        # If the CLASS background is disabled,
        # only constant w = w_eff is allowed.
        if not enable_class_background:
            # With the CLASS background disabled we do not have access
            # to the evolution of background densities. We can get by
            # as long as we have a constant w, in which case w_eff = w.
            # Note that this does not hold
            # for interacting/decaying species.
            if self.w_type != 'constant':
                abort(
                    f'Cannot construct w_eff of {self.name} '
                    f'without access to the CLASS background evolution.'
                )
            self.w_eff_type = 'constant'
            return
        # Construct a tabulated array of scale factor values.
        # If w is already tabulated at certain values, reuse these.
        self.w_eff_type = 'tabulated (a)'
        if self.w_type == 'tabulated (a)':
            a_tabulated = self.w_tabulated[0, :]
            a_min = a_tabulated[0]
        elif self.w_type == 'tabulated (t)':
            t_tabulated = self.w_tabulated[0, :]
            a_tabulated = asarray([scale_factor(t) for t in t_tabulated])
            a_min = a_tabulated[0]
        else:
            a_min = -1
        cosmoresults = compute_cosmo(
            class_call_reason=f'in order to determine w_eff(a) of {self.name}',
        )
        if a_min == -1:
            a_tabulated = cosmoresults.background['a']
        a_tabulated = a_tabulated.copy()  # Needed as we mutate a_tabulated below
        ρ_bar_tabulated = cosmoresults.ρ_bar(a_tabulated, self.class_species, apply_unit=False)
        ρ_bar_0 = ρ_bar_tabulated[ρ_bar_tabulated.shape[0] - 1]
        # At a = 1 a division by 0 error occurs due to 1/log(a).
        # Here we use linear extrapolation to obtain the end point.
        a_tabulated_end = a_tabulated[a_tabulated.shape[0] - 1]
        a_tabulated[a_tabulated.shape[0] - 1] = a_tabulated[a_tabulated.shape[0] - 2]
        w_eff_tabulated = np.log(ρ_bar_0/ρ_bar_tabulated)/(3*np.log(a_tabulated)) - 1
        if (  np.max(w_eff_tabulated[:w_eff_tabulated.shape[0] - 1])
            - np.min(w_eff_tabulated[:w_eff_tabulated.shape[0] - 1]) < 1e+6*machine_ϵ):
            # The effective equation of state is really constant.
            # Treat it as so.
            self.w_eff_type = 'constant'
            return
        a_tabulated[a_tabulated.shape[0] - 1] = a_tabulated_end
        # For most physical species, w_eff(a) is approximately a
        # power law in a and so log-log inter-/extrapolation
        # should be used.
        logx, logy = True, True
        if np.any(asarray(w_eff_tabulated[:w_eff_tabulated.shape[0] - 1]) <= 0):
            logy = False
        if self.class_species == 'fld':
            # The CLASS dark energy fluid (fld) uses
            # the {w_0, w_a} parametrisation. It turns out that the
            # best spline is achieved from log(a) but linear w_eff.
            logx, logy = True, False
        # Extrapolate to get the value at a = 1
        import scipy.interpolate
        w_eff_tabulated_end = scipy.interpolate.interp1d(
            np.log(     a_tabulated    [:a_tabulated.shape[0] - 1]) if logx else
                asarray(a_tabulated    [:a_tabulated.shape[0] - 1]),
            np.log(     w_eff_tabulated[:a_tabulated.shape[0] - 1]) if logy else
                asarray(w_eff_tabulated[:a_tabulated.shape[0] - 1]),
            'linear',
            fill_value='extrapolate',
        )(log(a_tabulated_end) if logx else a_tabulated_end)
        if logy:
            w_eff_tabulated_end = exp(w_eff_tabulated_end)
        w_eff_tabulated[a_tabulated.shape[0] - 1] = w_eff_tabulated_end
        # Instantiate the w_eff spline object
        self.w_eff_spline = Spline(a_tabulated, w_eff_tabulated, f'w_eff(a) of {self.name}',
            logx=logx, logy=logy)

    # Method which convert named fluid/particle
    # variable names to indices.
    @cython.header(
        # Arguments
        varnames=object,  # str, int or container of str's and ints
        single='bint',
        # Locals
        N_vars='Py_ssize_t',
        i='Py_ssize_t',
        indices='Py_ssize_t[::1]',
        varname=object, # str or int
        varnames_list=list,
        returns=object,  # Py_ssize_t[::1] or Py_ssize_t
    )
    def varnames2indices(self, varnames, single=False):
        """This method conveniently transform any reasonable input
        to an array of variable indices. Some examples:

        varnames2indices('ϱ') → asarray([0])
        varnames2indices(['J', 'ϱ']) → asarray([1, 0])
        varnames2indices(['pos', 'mom']) → asarray([0, 1])
        varnames2indices(2) → asarray([2])
        varnames2indices(['ς', 1]) → asarray([2, 1])
        varnames2indices('ϱ', single=True) → 0
        """
        if isinstance(varnames, str):
            # Single variable name given
            indices = asarray([self.fluid_names[varnames]], dtype=C2np['Py_ssize_t'])
        else:
            # One or more variable names/indices given
            varnames_list = any2list(varnames)
            indices = empty(len(varnames_list), dtype=C2np['Py_ssize_t'])
            for i, varname in enumerate(varnames_list):
                indices[i] = self.fluid_names[varname]
        if single:
            N_vars = indices.shape[0]
            if N_vars > 1:
                abort(f'The varnames2indices method was called '
                      'with {N_vars} variables while single is True')
            return indices[0]
        return indices

    # Method which extracts a fluid variable or fluidscalar from its
    # name, index or index and multi_index.
    def varname2var(self, varname):
        indices = self.fluid_names[varname]
        if isinstance(indices, tuple):
            # Return fluidscalar
            index, multi_indices = indices
            return self.fluidvars[index][multi_index]
        else:
            # Return fluid variable
            index = indices
            return self.fluidvars[index]

    # Generator for looping over all
    # scalar fluid grids within the component.
    def iterate_fluidscalars(self, include_disguised_scalar=True, include_additional_dofs=True):
        for i, fluidvar in enumerate(self.fluidvars):
            if include_disguised_scalar or i < self.boltzmann_order + 1:
                yield from fluidvar
                if include_additional_dofs:
                    for additional_dof in fluidvar.additional_dofs:
                        fluidscalar = fluidvar[additional_dof]
                        if fluidscalar is not None:
                            yield fluidscalar

    # Generator for looping over all non-linear
    # scalar fluid grids within the component.
    def iterate_nonlinear_fluidscalars(self):
        for fluidvar in self.fluidvars:
            for i, fluidscalar in enumerate(fluidvar):
                if fluidscalar is not None:
                    if i == 0 and fluidscalar.is_linear:
                        break
                    yield fluidscalar
            # Also yield the additional degrees of freedom
            # (e.g. 𝒫 corresponding to 'trace' on ς).
            for additional_dof in fluidvar.additional_dofs:
                fluidscalar = fluidvar[additional_dof]
                if fluidscalar is not None:
                    if fluidscalar.is_linear:
                        continue
                    yield fluidscalar

    # Method for communicating ghost points of all fluid variables
    @cython.header(
        # Arguments
        operation=str,
        # Locals
        fluidscalar='FluidScalar',
    )
    def communicate_fluid_grids(self, operation):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_ghosts(fluidscalar.grid_mv, operation)

    # Method for communicating ghost points
    # of all starred fluid variables.
    @cython.header(
        # Arguments
        operation=str,
        # Locals
        fluidscalar='FluidScalar',
    )
    def communicate_fluid_gridsˣ(self, operation):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_ghosts(fluidscalar.gridˣ_mv, operation)

    # Method for communicating ghost points
    # of all non-linear fluid variables.
    @cython.header(
        # Arguments
        operation=str,
        # Locals
        fluidscalar='FluidScalar',
    )
    def communicate_nonlinear_fluid_grids(self, operation):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            communicate_ghosts(fluidscalar.grid_mv, operation)

    # Method for communicating ghost points
    # of all starred non-linear fluid variables.
    @cython.header(
        # Arguments
        operation=str,
        # Locals
        fluidscalar='FluidScalar',
    )
    def communicate_nonlinear_fluid_gridsˣ(self, operation):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            communicate_ghosts(fluidscalar.gridˣ_mv, operation)

    # Method for communicating ghost points of all fluid Δ buffers
    @cython.header(
        # Arguments
        operation=str,
        # Locals
        fluidscalar='FluidScalar',
    )
    def communicate_fluid_Δ(self, operation):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_ghosts(fluidscalar.Δ_mv, operation)

    # Method which calls scale_grid on all non-linear fluid scalars
    @cython.header(
        # Arguments
        a='double',
        # Locals
        fluidscalar='FluidScalar',
    )
    def scale_nonlinear_fluid_grids(self, a):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            fluidscalar.scale_grid(a)

    # Method which calls the nullify_grid
    # on all non-linear fluid scalars.
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_nonlinear_fluid_grids(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            fluidscalar.nullify_grid()

    # Method which calls the nullify_gridˣ
    # on all non-linear fluid scalars.
    @cython.header(# Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_nonlinear_fluid_gridsˣ(self):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            fluidscalar.nullify_gridˣ()

    # Method for nullifying Δ buffers
    @cython.pheader(
        # Arguments
        specifically=object,  # str og container of str's
        only_active='bint',
        # Locals
        fluidscalar='FluidScalar',
        dim='Py_ssize_t',
        indexᵖ='Py_ssize_t',
        indexˣ='Py_ssize_t',
        lowest_active_rung='signed char',
        rung_indices='signed char*',
        variable=str,
        Δmom='double*',
        returns='void',
    )
    def nullify_Δ(self, specifically=None, only_active=True):
        """For fluid components, the nullification is delegated to the
        nullify_Δ() method on each fluid scalar.
        For particle components, a specific variable should be given by
        the "specifically" argument. Currently only "mom" is valid.
        """
        if self.representation == 'particles':
            if specifically is None:
                abort(
                    'You must specify "specifically" when calling Component.nullify_Δ() '
                    'for particle components.'
                )
            for variable in any2list(specifically):
                if variable == 'mom':
                    # We only nullify Δmom for active particles
                    Δmom = self.Δmom
                    rung_indices = self.rung_indices
                    lowest_active_rung = self.lowest_active_rung
                    for indexᵖ in range(self.N_local):
                        with unswitch:
                            if only_active and self.use_rungs:
                                if rung_indices[indexᵖ] < lowest_active_rung:
                                    continue
                        indexˣ = 3*indexᵖ
                        for dim in range(3):
                            Δmom[indexˣ + dim] = 0
                else:
                    abort(f'Component.nullify_Δ(): specifically = {specifically} not supported')
        elif self.representation == 'fluid':
            if specifically is not None:
                abort(
                    f'Component.nullify_Δ(): specifically = {specifically} not supported '
                    f'for fluid components'
                )
            for fluidscalar in self.iterate_fluidscalars():
                fluidscalar.nullify_Δ()

    # Method for checking whether the component is active
    @cython.header(
        # Arguments
        a='double',
        # Locals
        returns='bint',
    )
    def is_active(self, a=-1):
        """A component exists in one of three states:
          - "passive"   : Before it is activated, a < life[0].
          - "active"    : Once it has been activated but before
                          it is terminated, life[0] <= a < life[1].
          - "terminated": Once it has been terminated
                          after a period of activation, a >= life[1].
        In the normal case, the entire simulation takes place within the
        active period, and so no (explicit) activation
        and termination takes place.
        """
        if a == -1:
            a = universals.a
        return self.life[0] <= a < self.life[1]

    # Method which copies the content of all unstarred non-linear grids
    # into the corresponding starred grids.
    @cython.header(fluidscalar='FluidScalar', operation=str)
    def copy_nonlinear_fluid_grids_to_gridsˣ(self, operation='='):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            fluidscalar.copy_grid_to_gridˣ(operation)

    # Method which copies the content of all starred non-linear grids
    # into the corresponding unstarred grids.
    @cython.header(fluidscalar='FluidScalar', operation=str)
    def copy_nonlinear_fluid_gridsˣ_to_grids(self, operation='='):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            fluidscalar.copy_gridˣ_to_grid(operation)

    # Method for cleaning up the memory associated with this component,
    # without explicitly freeing it.
    def cleanup(self):
        # Cleanup Tiling instances on the component.
        # This will free the associated memory, as no other references
        # to the tilings should exist.
        self.tilings.clear()
        # We want to give back the allocated particle/fluid data memory
        # of this component to the system, but as lots of objects
        # potentially hold a reference to this component we cannot
        # simply free its memory (that and freeing this memory
        # explicitly does not play nicely with the automatic garbage
        # collection implemented via Component.__dealloc__ and
        # FluidScalar.__dealloc__ methods). Instead we resize the data,
        # resulting in effective freeing via a call to realloc().
        self.resize(1)

    # This method is automatically called when a Component instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        free(self. pos)
        free(self. mom)
        free(self.Δmom)
        free(self.rungs_N)
        free(self.rung_indices)
        free(self.rung_indices_jumped)

    # String representation
    def __repr__(self):
        return f'<component "{self.name}" of species "{self.species}">'
    def __str__(self):
        return self.__repr__()

# Array used by the Component.tile_sort() method
cython.declare(tile_location='double[::1]')
tile_location = empty(3, dtype=C2np['double'])

# Function for adding species to the universals_dict,
# recording the presence of any species in use.
@cython.header(
    # Arguments
    components=list,
    # Locals
    class_species_present=set,
    class_species_present_bytes=bytes,
    class_species_previously_present=str,
    i='Py_ssize_t',
    species_present=set,
    species_present_bytes=bytes,
    species_previously_present=str,
)
def update_species_present(components):
    """We cannot change the species_present and class_species_present
    fields of the universals structure, as this would require sticking
    in a new char*. Instead we use the universals_dict dict.
    For consistency, we store the species as bytes objects.
    """
    components = components.copy()
    for i, component in enumerate(components):
        if component.name in internally_defined_names:
            components[i] = None
    components = [component for component in components if component is not None]
    if not components:
        return
    # Species present (CO𝘕CEPT convention)
    species_present = {component.species for component in components}
    species_previously_present = universals_dict['species_present'].decode()
    if species_previously_present:
        species_present |= set(species_previously_present.split('+'))
    species_present_bytes = '+'.join(species_present).encode('utf-8')
    universals_dict['species_present'] = species_present_bytes
    # Species present (CLASS convention)
    class_species_present = {component.class_species for component in components}
    class_species_previously_present = universals_dict['class_species_present'].decode()
    if class_species_previously_present:
        class_species_present |= set(class_species_previously_present.split('+'))
    class_species_present_bytes = (
        # A component.class_species may be a combination of several CLASS species
        '+'.join(set('+'.join(class_species_present).split('+'))).encode('utf-8')
    )
    universals_dict['class_species_present'] = class_species_present_bytes

# Function for initialising a tiling on a component
@cython.header(
    # Arguments
    component='Component',
    tiling_name=str,
    initial_rung_size=object,  # sequence of length N_rungs or int-like
    # Locals
    extent='double[::1]',
    force=str,
    location='double[::1]',
    rung_index='signed char',
    shape=object, # sequence of length 3 of int-like
    shortrange_params_force=dict,
    tiling='Tiling',
    returns='Tiling',
)
def init_tiling(component, tiling_name, initial_rung_size=-1):
    """In general the tiling_name should be of the form
        '<force_name> (tiles)'
    or
        '<force_name> (subtiles)'
    In addition, the special
        'trivial'
    tiling_name is valid as well.
    """
    # Handle the special case of a trivial tiling
    if tiling_name == 'trivial':
        # This tiling spans the box using a single tile,
        # resulting in no actual tiling. It is useful since the
        # rung-ordering of particles is done at the tile level.
        shape = (1, 1, 1)
        tiling_shapes[tiling_name] = asarray(shape, dtype=C2np['Py_ssize_t'])
        # If not already specified, the rungs within the tile start out
        # with half of the mean required memory per rung.
        if initial_rung_size == -1:
            initial_rung_size = [
                component.rungs_N[rung_index]//2
                for rung_index in range(N_rungs)
            ]
        # The extent of the entire tiling, i.e. the extent of the box
        extent = asarray([boxsize]*3, dtype=C2np['double'])
        # The position of the beginning of the tiling,
        # i.e. the left, backward, lower corner of the box.
        location = zeros(3, dtype=C2np['double'])
        # Instantiate Tiling instance
        return Tiling(tiling_name, component, shape, extent, initial_rung_size,
            refinement_period=0)
    # Delegate subtiling initialisation
    if ' (subtiles)' in tiling_name:
        return init_subtiling(component, tiling_name, initial_rung_size)
    # Extract the name of the force
    match = re.search(r'(.+) \(tiles\)', tiling_name)
    if not match:
        abort(f'init_tiling() called with tiling_name = "{tiling_name}"')
    force = match.group(1)
    # Extract the short-range parameters for the force
    shortrange_params_force = shortrange_params.get(force)
    if shortrange_params_force is None:
        abort(f'Force "{force}" not specified in shortrange_params')
    # The same tiling is applied to all domains. The tile
    # decomposition on a domain will have a general shape
    # of shape[0]×shape[1]×shape[2], with shape[dim] determined
    # by the criterion that a tile must be at least as large as
    # the given tilesize length, in all directions.
    # At the same time, we want to maximize the number of tiles.
    shape = tiling_shapes.get(tiling_name)
    if shape is None:
        shape = asarray(
            (boxsize/asarray(domain_subdivisions))/shortrange_params_force['tilesize']
            *(1 + machine_ϵ),
            dtype=C2np['Py_ssize_t'],
        )
        masterprint(f'Tile decomposition ({force}): {shape[0]}×{shape[1]}×{shape[2]}')
        tiling_shapes[tiling_name] = shape
    # The tiling needs to have a minimum number of tiles
    # across each dimension. The minimum criteria are:
    # - The logic used for the tile pairing assumes that all domain
    #   tilings have immediate neighbour tiles which are different
    #   from each other (and the tile in question itself).
    #   We thus need at least 3 tiles along each dimension per tiling.
    #   This restriction is really for the entire, global tiling,
    #   not for the individual domain tilings.
    #   - The logic used for the subtile pairings comes with a similar
    #     restriction, applying the above restriction at the
    #     domain tiling level.
    # - The shortest path between pairs of particles in different tiles
    #   may either be the "direct" path or a path through the periodic
    #   boundary of the box. With 3 tiles (or lower) along each
    #   dimension, it is possible to have a mixture of these two
    #   shortest path types for pairs of particles within a given
    #   pair of tiles. That is, the tile locations alone cannot
    #   generally tell us whether to use the "direct" or the periodic
    #   path, but the implemented logic for the particle periodicity in
    #   fact assumes that it can. We thus really need at least 4 tiles
    #   along each dimension of the global tiling, not just 3.
    if np.min(asarray(shape)*asarray(domain_subdivisions)) < 4:
        abort(
            f'The global {force} tiling needs to have at least 4 tiles across the box in '
            f'every direction. Consider lowering shortrange_params["{force}"]["tilesize"].'
        )
    if np.min(shape) < 3:
        msg = (
            f'The {force} domain tiling needs a subdivision of at least 3 in every direction. '
            f'Consider lowering shortrange_params["{force}"]["tilesize"].'
        )
        if 1 != nprocs and not iscubic(nprocs):
            msg += ' It may also help to choose a lower and/or cubic number of processes.'
        abort(msg)
    # If not already specified, the rungs within each tile start out
    # with half of the mean required memory per rung.
    if initial_rung_size == -1:
        initial_rung_size = [
            component.rungs_N[rung_index]//(2*np.prod(shape))
            for rung_index in range(N_rungs)
        ]
    # The extent of the entire tiling, i.e. the extent of the domain
    extent = asarray((domain_size_x, domain_size_y, domain_size_z), dtype=C2np['double'])
    # The position of the beginning of the tiling,
    # i.e. the left, backward, lower corner of this domain.
    location = asarray((domain_bgn_x, domain_bgn_y, domain_bgn_z), dtype=C2np['double'])
    # Instantiate Tiling instance
    tiling = Tiling(tiling_name, component, shape, extent, initial_rung_size, refinement_period=0)
    # Relocate the tiling
    tiling.relocate(location)
    return tiling
# Mapping from tiling names to shapes of all tilings instantiated
# across all components.
cython.declare(tiling_shapes=dict)
tiling_shapes = {}

# Function for initialising a subtiling on a component
@cython.header(
    # Arguments
    component='Component',
    subtiling_name=str,
    initial_rung_size=object,  # sequence of length N_rungs or int-like
    # Locals
    coarse_tiling='Tiling',
    extent='double[::1]',
    force=str,
    i='int',
    j='int',
    k='int',
    location='double[::1]',
    particles_per_subtile='double',
    particles_per_subtile_max='double',
    particles_per_subtile_min='double',
    ratio='double',
    refine='bint',
    refinement_period='Py_ssize_t',
    rung_index='signed char',
    shape=object,  # sequence of length 3 or 2 or int-like
    shortrange_params_force=dict,
    returns='Tiling',
)
def init_subtiling(component, subtiling_name, initial_rung_size=-1):
    """The subtiling_name should be of the form
        '<force_name> (subtiles)'
    """
    # Extract the name of the force
    match = re.search(r'(.+) \(subtiles\)', subtiling_name)
    if not match:
        abort(f'init_subtiling() called with subtiling_name = "{subtiling_name}"')
    force = match.group(1)
    # Extract the short-range parameters for the force
    shortrange_params_force = shortrange_params.get(force)
    if shortrange_params_force is None:
        abort(f'Force "{force}" not specified in shortrange_params')
    # Grab the constant refinement period,
    # if automatic subtiling refinement is enabled.
    refine = False
    refinement_period = 0
    shape = shortrange_params_force['subtiling']
    if shape[0] == 'automatic':
        refine = True
        refinement_period = shape[1]
    # The entire (sub)tiling currently being initialised lives
    # within a single tile of the "<force> (tiles)" tiling.
    # Get this coarser tiling.
    coarse_tiling = component.tilings.get(f'{force} (tiles)')
    if coarse_tiling is None:
        abort(
            f'Cannot initialise the "{force} (subtiles)" tiling '
            f'without first having the "{force} (tiles)" tiling initialised'
        )
    # Get the shape of the subtiling
    shape = tiling_shapes.get(subtiling_name)
    if shape is None:
        shape = shortrange_params_force['subtiling']
        if refine:
            # The subtiling shape is to be determined
            # automatically. It is optimal to have the subtiles
            # be as cubic as possible. Additionally, we have
            # found that having (on average) ~8–14
            # particles/subtile is optimal, when the particles
            # are not too clustered. Here we pick the most cubic
            # choice of all possible subtiling shapes which
            # leads to subtiles of a volume comparable to the
            # above stated number of particles.
            particles_per_subtile_min, particles_per_subtile_max = 8, 14
            tiling_global_shape = asarray(
                asarray(domain_subdivisions)*asarray(coarse_tiling.shape),
                dtype=C2np['double'],
            )
            shape_candidates = []
            for particles_per_subtile in (
                particles_per_subtile_min, particles_per_subtile_max,
            ):
                shape = np.prod(tiling_global_shape)/tiling_global_shape
                shape *= (
                    float(component.N)/(
                        nprocs*np.prod(asarray(coarse_tiling.shape)*shape)*particles_per_subtile
                    )
                )**(1./3.)
                shape = asarray(np.round(shape), dtype=C2np['Py_ssize_t'])
                shape[shape == 0] = 1
                shape_candidates.append(shape)
            shape_diff = shape_candidates[0] - shape_candidates[1]
            shape_base = shape_candidates[1]
            shape_candidates = {}
            for         i in range(shape_diff[0] + 1):
                for     j in range(shape_diff[1] + 1):
                    for k in range(shape_diff[2] + 1):
                        shape = shape_base + asarray((i, j, k))
                        subtiling_global_shape = tiling_global_shape*shape
                        particles_per_subtile = (
                            float(component.N)/np.prod(subtiling_global_shape)
                        )
                        # Construct tuple key to be used to
                        # store this shape. The lower the value
                        # of each element in the key, the better
                        # we consider this key to be.
                        key = []
                        noncubicness = np.max((
                            np.max(subtiling_global_shape)**3
                                /np.prod(subtiling_global_shape),
                            np.prod(subtiling_global_shape)
                                /np.min(subtiling_global_shape)**3,
                        ))
                        key.append(noncubicness)
                        ratio = particles_per_subtile*ℝ[
                            1/sqrt(particles_per_subtile_min*particles_per_subtile_max)
                        ]
                        if ratio < 1:
                            ratio = 1/ratio
                        key.append(ratio)
                        shape_candidates[tuple(key)] = shape
            # Pick the shape with the smallest key
            shape = shape_candidates[sorted(shape_candidates)[0]]
        shape = asarray(shape, dtype=C2np['Py_ssize_t'])
        masterprint(f'Subtile decomposition ({force}): {shape[0]}×{shape[1]}×{shape[2]}')
        tiling_shapes[subtiling_name] = shape
    # If not already specified, the rungs within each subtile start out
    # with half of the mean required memory per rung.
    if initial_rung_size == -1:
        initial_rung_size = [
            component.rungs_N[rung_index]//(2*int(np.prod(shape)*np.prod(coarse_tiling.shape)))
            for rung_index in range(N_rungs)
        ]
    # The extent of the entire subtiling,
    # i.e. the extent of a coarse tile.
    extent = coarse_tiling.tile_extent
    # Instantiate Tiling instance.
    # Note that since this same subtiling will be used for all of the
    # coarse tiles, the location of this subtiling is not static.
    # We thus do not care about the initial value of the location.
    return Tiling(subtiling_name, component, shape, extent, initial_rung_size, refinement_period)

# Function which refines the subtiling corresponding to the given
# interaction_name, on all instantiated components. The original
# subtilings are not deleted, and so may later be substituted back in
# if desired.
@cython.header(
    # Arguments
    interaction_name=str,
    # Locals
    component='Component',
    computation_time_total='double',
    dim='int',
    key=tuple,
    key2=tuple,
    refinement_offset='Py_ssize_t',
    refinement_threshold='double',
    shape='Py_ssize_t[::1]',
    subtile_extent='double[::1]',
    subtile_extent_max='double',
    subtiling='Tiling',
    subtiling_2='Tiling',
    subtiling_name=str,
    subtiling_name_2=str,
    subtiling_rejected='Tiling',
    subtiling_rejected_2='Tiling',
    returns='void',
)
def tentatively_refine_subtiling(interaction_name):
    subtiling_name   = f'{interaction_name} (subtiles)'
    subtiling_name_2 = f'{interaction_name} (subtiles 2)'
    shape = None
    for component in components_all:
        subtiling = component.tilings.pop(subtiling_name, None)
        if subtiling is None:
            continue
        # Component with the right subtiling found.
        # Take a copy of its computation_time_total and
        # refinement_offset attributes, as we need to copy these over
        # to the refined subtiling.
        computation_time_total = subtiling.computation_time_total
        refinement_offset      = subtiling.refinement_offset
        # Remove the subtiling (including its second version
        # "subtiles 2") from the component and store it away.
        key = (component, subtiling_name)
        stored_subtilings[key] = subtiling
        subtiling_2 = component.tilings.pop(subtiling_name_2, None)
        if subtiling_2 is not None:
            key2 = (component, subtiling_name_2)
            stored_subtilings[key2] = subtiling_2
        # If we have already attempted this refinement before,
        # the new, rejected subtilings are stored
        # in rejected_subtilings. Reuse these if available.
        subtiling_rejected = rejected_subtilings.pop(key, None)
        if subtiling_rejected is not None:
            subtiling_rejected.computation_time_total = computation_time_total
            subtiling_rejected.refinement_offset      = refinement_offset
            component.tilings[subtiling_name] = subtiling_rejected
            if subtiling_2 is not None:
                subtiling_rejected_2 = rejected_subtilings.pop(key2, None)
                if subtiling_rejected_2 is not None:
                    component.tilings[subtiling_name_2] = subtiling_rejected_2
            tiling_shapes[subtiling_name] = asarray(subtiling_rejected.shape).copy()
            continue
        # Refine the subtiling shape. The dimension with the largest
        # length will be divided into one more section than before. For
        # very eccentric subtile shapes this is what we want. Usually
        # though, subtiles are either exactly cubic or close to cubic.
        # In the exactly cubic case, all dimensions are refined, and so
        # the subtiles stay cubic. In the case of near cubic subtiles,
        # refining only the largest dimension will lead to eccentric
        # subtiles, and so it is preferable to refine all dimensions
        # with lengths close to the maximum length. The threshold is set
        # by the refinement_threshold constant.
        refinement_threshold = 0.85
        if shape is None:
            shape = tiling_shapes[subtiling_name]
            subtile_extent = subtiling.tile_extent
            subtile_extent_max = max(subtile_extent)
            for dim in range(3):
                if subtile_extent[dim] > refinement_threshold*subtile_extent_max:
                    shape[dim] += 1
        # Initialise new subtiling
        subtiling = component.init_tiling(subtiling_name)
        subtiling.computation_time_total = computation_time_total
        subtiling.refinement_offset      = refinement_offset
        if subtiling_2 is not None:
            component.tilings.pop(subtiling_name)
            subtiling_2 = component.init_tiling(subtiling_name)
            component.tilings[subtiling_name  ] = subtiling
            component.tilings[subtiling_name_2] = subtiling_2
# Global containers temporarily storing subtilings for each component
cython.declare(stored_subtilings=dict, rejected_subtilings=dict)
stored_subtilings = {}
rejected_subtilings = {}

# Function which either accepts or rejects the tentative subtile
# refining carried out by tentatively_refine_subtiling().
@cython.header(
    # Arguments
    interaction_name=str,
    computation_times_sum='double[::1]',
    computation_times_sqsums='double[::1]',
    computation_times_N='Py_ssize_t[::1]',
    # Locals
    N='Py_ssize_t',
    component='Component',
    computation_time_total_new='double',
    computation_time_total_old='double',
    index='Py_ssize_t',
    key=tuple,
    keys=list,
    name=str,
    sigmas='double',
    shape='Py_ssize_t[::1]',
    subtiling='Tiling',
    subtiling_name=str,
    subtiling_name_2=str,
    subtiling_names=set,
    subtiling_rejected='Tiling',
    returns='Py_ssize_t[::1]',
)
def accept_or_reject_subtiling_refinement(
    interaction_name, computation_times_sum, computation_times_sqsums, computation_times_N,
):
    """
    All three computation_times_* arrays are indexed as follows:
    computation_times_sum[rung_index] → new computation times with
    lowest_active_rung == rung_index, "new" meaning with the subtiling
    refinement.
    computation_times_sum[N_rungs + rung_index] → old computation
    times with, "old" meaning with the original subtiling refinement.
    """
    # Compute mean of all computation times
    for index in range(ℤ[2*N_rungs]):
        N = computation_times_N[index]
        computation_times_mean[index] = (
            computation_times_sum[index]/computation_times_N[index]
        ) if N > 0 else 0
    # Compute std of old computation times
    for index in range(N_rungs, ℤ[2*N_rungs]):
        N = computation_times_N[index]
        computation_times_std[index] = (
            sqrt(computation_times_sqsums[index]/N - computation_times_mean[index]**2)
        ) if N > 0 else 0
    # Compute the total time it took to carry out all of the old
    # computations together. To exaggerate a bit (encouraging early
    # subtile refinement), we use mean + sigmas*std rather than just
    # the mean, where sigmas sets the number of standard deviations
    # we wish to exaggerate with.
    subtiling_name   = f'{interaction_name} (subtiles)'
    subtiling_name_2 = f'{interaction_name} (subtiles 2)'
    subtiling_names = {subtiling_name, subtiling_name_2}
    for key, subtiling in stored_subtilings.items():
        if key[1] in subtiling_names:
            # Assign sigmas a value using some appropriate sigmoid,
            # starting out large for small subtilings and decreasing
            # for already large subtilings.
            sigmas = 0.7 - 0.5*erf(0.42*cbrt(np.prod(subtiling.shape)) - 2.3)
            break
    else:
        # For some reason, the original subtiling was not found.
        # Use a generic value for sigmas.
        sigmas = 0.25
    computation_time_total_old = 0
    for index in range(N_rungs, ℤ[2*N_rungs]):
        if computation_times_N[index] == 0 or computation_times_N[-N_rungs + index] == 0:
            continue
        computation_time_total_old += computation_times_N[index]*(
            computation_times_mean[index] + sigmas*computation_times_std[index]
        )
    # Compute the total time it would take the new subtiling to perform
    # the work recorded by the old subtiling.
    computation_time_total_new = 0
    for index in range(N_rungs):
        if computation_times_N[index] == 0 or computation_times_N[N_rungs + index] == 0:
            continue
        computation_time_total_new += (
            computation_times_N[N_rungs + index]*computation_times_mean[index]
        )
    # Accept the recent subtiling refinement if it outperforms the
    # old one. Otherwise reject it.
    if computation_time_total_new < computation_time_total_old:
        # Accept recent subtiling refinement.
        # The old subtilings should be cleaned up. Here we remove the
        # only references to the old subtilings, after which their
        # memory is available for garbage collection.
        for key in tuple(stored_subtilings.keys()):
            if key[1] in subtiling_names:
                stored_subtilings.pop(key)
        # Fast forward the refinement cycle so that we immediately
        # begin collecting computation times.
        for component in components_all:
            subtiling = component.tilings.get(subtiling_name)
            if subtiling is None:
                continue
            subtiling.refinement_offset += (
                subtiling.refinement_period - subtiling_refinement_period_min
        )
        # Return the new subtiling decomposition
        return tiling_shapes[subtiling_name]
    else:
        # Reject recent subtiling refinement.
        # Move the stored subtilings back on to their respective
        # components and insert rejected subtilings into the global
        # rejected_subtilings dict.
        keys = []
        for key, subtiling in stored_subtilings.items():
            component, name = key
            if name in subtiling_names:
                keys.append(key)
                shape = subtiling.shape
                subtiling_rejected = component.tilings[name]
                subtiling.computation_time_total = subtiling_rejected.computation_time_total
                subtiling.refinement_offset      = subtiling_rejected.refinement_offset
                rejected_subtilings[key] = subtiling_rejected
                component.tilings[name] = subtiling
        tiling_shapes[subtiling_name] = asarray(shape).copy()
        # Remove the old references in the stored_subtilings dict
        for key in keys:
            stored_subtilings.pop(key)
        # Return zeros, indicating rejection
        return subtiling_shape_rejected
# Arrays used by the accept_or_reject_subtiling_refinement() function
cython.declare(
    computation_times_mean='double[::1]',
    computation_times_std='double[::1]',
    subtiling_shape_rejected='Py_ssize_t[::1]',
)
computation_times_mean = zeros(2*N_rungs, dtype=C2np['double'])
computation_times_std  = zeros(2*N_rungs, dtype=C2np['double'])
subtiling_shape_rejected = zeros(3, dtype=C2np['Py_ssize_t'])



# Set of all approximations implemented on Component objects
cython.declare(approximations_implemented=set)
approximations_implemented = {
    unicode('P=wρ'),
}
# Set of all component names used internally by the code,
# and which the user should generally avoid.
cython.declare(internally_defined_names=set)
internally_defined_names = {
    'all',
    'all combinations',
    'buffer',
    'default',
    'fake',
    'global',
    'linear power spectrum',
    'tmp',
    'total',
}
# Names of all implemented fluid variables in order.
# Note that 𝒫 is not considered a separate fluid variable,
# but rather a fluid scalar that lives on ς.
cython.declare(fluidvar_names=tuple)
fluidvar_names = ('ϱ', 'J', 'ς')
# Flag specifying whether a warning should be given if multiple
# components with the same name are instantiated, and a set of names of
# all instantiated components.
cython.declare(allow_similarly_named_components='bint', component_names=set)
allow_similarly_named_components = False
component_names = set()
# Set of all instantiated components
cython.declare(components_all=list)
components_all = []

# Get local domain information
domain_info = get_domain_info()
cython.declare(
    domain_subdivisions='int[::1]',
    domain_size_x='double',
    domain_size_y='double',
    domain_size_z='double',
    domain_bgn_x='double',
    domain_bgn_y='double',
    domain_bgn_z='double',
)
domain_subdivisions = domain_info.subdivisions
domain_size_x       = domain_info.size_x
domain_size_y       = domain_info.size_y
domain_size_z       = domain_info.size_z
domain_bgn_x        = domain_info.bgn_x
domain_bgn_y        = domain_info.bgn_y
domain_bgn_z        = domain_info.bgn_z
