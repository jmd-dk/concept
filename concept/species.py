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
cimport('from analysis import measure')
cimport('from communication import communicate_domain, domain_subdivisions, exchange, smart_mpi')
cimport('from fluid import maccormack, maccormack_internal_sources, '
    'kurganov_tadmor, kurganov_tadmor_internal_sources'
)
cimport('from integration import Spline, cosmic_time, scale_factor, ȧ')
cimport('from linear import compute_cosmo, compute_transfer, get_default_k_parameters, realize')



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
        # Store initialization arguments as instance variables
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
        self.disguised_scalar = (self.component.boltzmann_order == self.varnum)
        # Should this fluid variable do realizations when iterating
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
        # Initialize tensor data
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
        if not multi_index in self.data:
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
    def iterate(self, *attributes, multi_indices=False):
        """This generator yields all normal elements of the tensor
        (that is, the additional degrees of freedom are not included).
        For disguised scalars, all logical elements are realized
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
                    self.component.realize_if_linear(self.varnum, specific_multi_index=multi_index)
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
    # Initialization method
    @cython.header(# Arguments
                   varnum='int',
                   multi_index=object,  # tuple or int-like
                   is_linear='bint',
                   )
    def __init__(self, varnum, multi_index=(), is_linear=False):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the FluidScalar type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
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
    @cython.header(# Arguments
                   shape_nopseudo_noghost=tuple,
                   )
    def resize(self, shape_nopseudo_noghost):
        """After resizing the fluid scalar,
        all fluid elements will be nullified.
        """
        # The full shape and size of the grid,
        # with pseudo and ghost points.
        self.shape = tuple([2 + s + 1 + 2 for s in shape_nopseudo_noghost])
        self.size = np.prod(self.shape)
        # The shape and size of the grid
        # with no ghost points but with pseudo points.
        self.shape_noghosts = tuple([s + 1 for s in shape_nopseudo_noghost])
        self.size_noghosts = np.prod(self.shape_noghosts)
        # The data itself
        self.grid = realloc(self.grid, self.size*sizeof('double'))
        self.grid_mv = cast(self.grid, 'double[:self.shape[0], :self.shape[1], :self.shape[2]]')
        self.grid_noghosts = self.grid_mv[
            2:(self.grid_mv.shape[0] - 2),
            2:(self.grid_mv.shape[1] - 2),
            2:(self.grid_mv.shape[2] - 2),
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
                2:(self.gridˣ_mv.shape[0] - 2),
                2:(self.gridˣ_mv.shape[1] - 2),
                2:(self.gridˣ_mv.shape[2] - 2),
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
            2:(self.Δ_mv.shape[0] - 2),
            2:(self.Δ_mv.shape[1] - 2),
            2:(self.Δ_mv.shape[2] - 2),
        ]
        # Nullify the newly allocated Δ buffer
        self.nullify_Δ()

    # Method for scaling the data grid
    @cython.pheader(# Argument
                    a='double',
                    # Locals
                    i='Py_ssize_t',
                    grid='double*',
                    )
    def scale_grid(self, a):
        # Extract data pointer
        grid = self.grid
        # Scale data buffer
        for i in range(self.size):
            grid[i] *= a

    # Method for nullifying the data grid
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    grid='double*',
                    )
    def nullify_grid(self):
        # Extract data pointer
        grid = self.grid
        # Nullify data buffer
        for i in range(self.size):
            grid[i] = 0

    # Method for nullifying the starred grid
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    gridˣ='double*',
                    )
    def nullify_gridˣ(self):
        # Extract starred buffer pointer
        gridˣ = self.gridˣ
        # Nullify starred buffer
        for i in range(self.size):
            gridˣ[i] = 0

    # Method for nullifying the Δ buffer
    @cython.pheader(# Locals
                    i='Py_ssize_t',
                    Δ='double*',
                    )
    def nullify_Δ(self):
        # Extract Δ buffer pointer
        Δ = self.Δ
        # Nullify Δ buffer
        for i in range(self.size):
            Δ[i] = 0

    # Method for copying the content of grid into gridˣ
    @cython.pheader(
        # Arguments
        operation=str,
        # Locals
        i='Py_ssize_t',
        grid='double*',
        gridˣ='double*',
    )
    def copy_grid_to_gridˣ(self, operation='='):
        grid, gridˣ = self.grid, self.gridˣ
        for i in range(self.size):
            with unswitch:
                if operation == '=':
                    gridˣ[i] = grid[i]
                elif operation == '+=':
                    gridˣ[i] += grid[i]

    # Method for copying the content of gridˣ into grid
    @cython.pheader(
        # Arguments
        operation=str,
        # Locals
        i='Py_ssize_t',
        grid='double*',
        gridˣ='double*',
    )
    def copy_gridˣ_to_grid(self, operation='='):
        grid, gridˣ = self.grid, self.gridˣ
        for i in range(self.size):
            with unswitch:
                if operation == '=':
                    grid[i] = gridˣ[i]
                elif operation == '+=':
                    grid[i] += gridˣ[i]

    # This method is automaticlly called when a FluidScalar instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        free(self.grid)
        if not self.is_linear:
            free(self.gridˣ)

    # String representation
    def __repr__(self):
        return '<fluidscalar {}[{}]>'.format(self.varnum,
                                             ', '.join([str(mi) for mi in self.multi_index]))
    def __str__(self):
        return self.__repr__()



# The class governing any component of the universe
@cython.cclass
class Component:
    """An instance of this class represents either a collection of
    particles or a grid of fluid values. A Component instance should be
    present on all processes.
    """

    # Initialization method
    @cython.pheader(
        # Arguments
        name=str,
        species=str,
        N_or_gridsize='Py_ssize_t',
        mass='double',
        boltzmann_order='Py_ssize_t',
        forces=dict,
        class_species=object,  # str or container of str's
        w=object,  # NoneType, float, int, str or dict
        boltzmann_closure=str,
        approximations=dict,
        softening_length=object,  # float or str
        realization_options=dict,
    )
    def __init__(self, name, species, N_or_gridsize, *,
        mass=-1,
        boltzmann_order=2,
        forces=None,
        class_species=None,
        realization_options=None,
        w=None,
        boltzmann_closure=None,
        approximations=None,
        softening_length=None,
    ):
        # The keyword-only arguments are passed from dicts in the
        # initial_conditions user parameter. If not specified there
        # (None passed) they will be set trough other parameters.
        # Of special interest is the fluid parameters boltzmann_order,
        # boltzmann_closure and approximations. Together, these control
        # the degree to which a fluid component will behave non-
        # linearly. Below is listed an overview of all allowed
        # combinations of boltzmann_order and boltzmann_closure,
        # together with the accompanying fluid variable behavoir.
        # Note that for particle components, only boltzmann_order = 2
        # is allowed.
        #
        # boltzmann_order = 0, boltzmann_closure = 'class':
        #     linear ϱ  (Realized continuously, affects other components gravitationally)
        #
        # boltzmann_order = 1, boltzmann_closure = 'truncate':
        #     non-linear ϱ  (Though "non-linear", ϱ is frozen in time as no J exist.
        #                    Also, unlike when boltzmann_order = 1 and boltzmann_closure = 'class',
        #                    ϱ will only be realized at the beginning of the simulation.)
        #
        # boltzmann_order = 1, boltzmann_closure = 'class':
        #     non-linear ϱ
        #         linear J  (realized continuously)
        #         linear 𝒫  (P=wρ approximation enforced)
        #
        # boltzmann_order: 2, boltzmann_closure = 'truncate':
        #     non-linear ϱ
        #     non-linear J
        #         linear 𝒫  (P=wρ approximation enforced)
        #
        # boltzmann_order = 2, boltzmann_closure = 'class':
        #     non-linear ϱ
        #     non-linear J
        #         linear 𝒫  (realized continuously)
        #         linear ς  (realized continuously)
        #
        # boltzmann_order = 3, boltzmann_closure = 'truncate':
        #     non-linear ϱ
        #     non-linear J
        #     non-linear 𝒫  (Though "non-linear", 𝒫 is frozen in time since the evolution equation
        #                    for 𝒫 is not implemented.
        #                    Also, unlike when boltzmann_order = 2 and boltzmann_closure = 'class',
        #                    𝒫 will only be realized at the beginning of the simulation.)
        #     non-linear ς  (Though "non-linear", ς is frozen in time since the evolution equation
        #                    for ς is not implemented.
        #                    Also, unlike when boltzmann_order = 2 and boltzmann_closure = 'class',
        #                    ς will only be realized at the beginning of the simulation.)
        #
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Component type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        # General component attributes
        public str name
        public str species
        public str representation
        public dict forces
        public str class_species
        # Particle attributes
        public Py_ssize_t N
        public Py_ssize_t N_allocated
        public Py_ssize_t N_local
        public double mass
        public double softening_length
        # Particle data
        double* posx
        double* posy
        double* posz
        double* momx
        double* momy
        double* momz
        double** pos
        double** mom
        public double[::1] posx_mv
        public double[::1] posy_mv
        public double[::1] posz_mv
        public double[::1] momx_mv
        public double[::1] momy_mv
        public double[::1] momz_mv
        public list pos_mv
        public list mom_mv
        # Particle Δ buffers
        double* Δposx
        double* Δposy
        double* Δposz
        double* Δmomx
        double* Δmomy
        double* Δmomz
        double** Δpos
        double** Δmom
        public double[::1] Δposx_mv
        public double[::1] Δposy_mv
        public double[::1] Δposz_mv
        public double[::1] Δmomx_mv
        public double[::1] Δmomy_mv
        public double[::1] Δmomz_mv
        public list Δpos_mv
        public list Δmom_mv
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
        """
        # Check that the name does not conflict with
        # one of the special names used internally,
        # and that the name has not already been used.
        if name in internally_defined_names:
            masterwarn(
                f'A species by the name of "{name}" is to be created. '
                f'As this name is used internally by the code, '
                f'this may lead to erroneous behaviour.'
            )
        elif not allow_similarly_named_components and name in component_names:
            masterwarn(
                f'A component with the name of "{name}" has already '
                f'been instantiated. Instantiating multiple components '
                f'with the same name may lead to undesired behaviour.'
            )
        for char in ',{}':
            if char in name:
                masterwarn(
                    f'A species by the name of "{name}" is to be created. '
                    f'As this name contains a "{char}" character, '
                    f'this may lead to erroneous behaviour.'
                )
        if name:
            component_names.add(name)
        # General attributes
        self.name    = name
        self.species = species
        self.representation = get_representation(self.species)
        if self.representation == 'particles':
            self.N = N_or_gridsize
            self.gridsize = 1
        elif self.representation == 'fluid':
            self.gridsize = N_or_gridsize
            self.N = 1
            if self.gridsize%2 != 0:
                masterwarn(
                    f'The fluid component "{self.name}" has an odd gridsize ({self.gridsize}). '
                    f'Some operations may not function correctly.'
                )
        # Set forces (and force methods)
        if forces is None:
            forces = is_selected(self, select_forces, accumulate=True)
        if not forces:
            forces = {}
        self.forces = forces
        # Set the CLASS species
        if class_species is None:
            class_species = is_selected(self, select_class_species)
        elif not isinstance(class_species, str):
            class_species = '+'.join(class_species)
        if class_species == 'default':
            if self.species in default_class_species:
                class_species = default_class_species[self.species]
            else:
                abort(
                    f'Default CLASS species assignment failed because '
                    f'the species "{self.species}" does not map to any CLASS species'
                )
        self.class_species = class_species
        # Set closure rule for the Boltzmann hierarchy
        if boltzmann_closure is None:
            boltzmann_closure = is_selected(self, select_boltzmann_closure)
        if not boltzmann_closure:
            boltzmann_closure = ''
        self.boltzmann_closure = boltzmann_closure.lower()
        if self.representation == 'fluid' and self.boltzmann_closure not in ('truncate', 'class'):
            abort(
                f'The component "{self.name}" was initialized '
                f'with an unknown Boltzmann closure of "{self.boltzmann_closure}"'
            )
        # Set realization options
        if realization_options is None:
            realization_options = {}
        realization_options_selected = is_selected(
            self, select_realization_options, accumulate=True)
        realization_options_selected.update(realization_options)
        realization_options = realization_options_selected
        realization_options_all = realization_options.get('all', {})
        for key, val in realization_options.copy().items():
            if not isinstance(val, dict):
                realization_options_all[key] = val
                del realization_options[key]
        realization_options_all = {
            key.lower().replace(' ', '').replace('-', '').replace('_', ''):
                (val.lower().replace(' ', '').replace('-', '').replace('_', '') if
                isinstance(val, str) else val)
            for key, val in realization_options_all.items()}
        varnames = {
            'particles': ('pos', 'mom'),
            'fluid': ('ϱ', 'J', '𝒫', 'ς'),
        }[self.representation]
        wrong_varname_sets = {
            'pos': {'x', 'position', 'positions', 'Position', 'Positions'},
            'mom': {'momentum', 'momenta', 'Momentum', 'Momenta'},
            'ϱ': {'r', 'rho', 'ρ'},
            'J': {'j'},
            '𝒫': {'P', 'δP', 'δ𝒫', 'p', 'δp'},
            'ς': {'s', 'sigma', 'Sigma', 'σ', 'Σ'},
        }
        for varname in varnames:
            wrong_varnames = wrong_varname_sets[varname]
            for wrong_varname in wrong_varnames:
                realization_options_varname = (
                       realization_options.get(unicode(wrong_varname))
                    or realization_options.get(asciify(wrong_varname))
                )
                if realization_options_varname:
                    realization_options[varname] = realization_options_varname
                    break
        realization_options_default = {
            # Linear realization options
            'velocitiesfromdisplacements': realization_options_all.get(
                'velocitiesfromdisplacements', False),
            'backscaling': realization_options_all.get('backscaling', False),
            # Non-linear realization options
            'structure'    : realization_options_all.get('structure', 'nonlinear'),
            'compoundorder': realization_options_all.get('compoundorder', 'linear'),
        }
        for varname in varnames:
            realization_options_default_copy = realization_options_default.copy()
            for varname_encoding in (unicode(varname), asciify(varname)):
                if varname_encoding in realization_options:
                    realization_options_default_copy.update(realization_options[varname_encoding])
            realization_options[varname] = realization_options_default_copy
        for varname, realization_options_varname in realization_options.copy().items():
            realization_options[unicode(varname)] = realization_options_varname.copy()
            realization_options[asciify(varname)] = realization_options_varname.copy()
        realization_options = {
            varname: {
                key.lower().replace(' ', '').replace('-', '').replace('_', ''):
                    (val.lower().replace(' ', '').replace('-', '').replace('_', '') if
                    isinstance(val, str) else val)
                for key, val in realization_options[varname].items()}
            for varname in varnames
        }
        for varname, realization_options_varname in realization_options.copy().items():
            realization_options[unicode(varname)] = realization_options_varname.copy()
            realization_options[asciify(varname)] = realization_options_varname.copy()
            for realization_option_varname in realization_options_varname:
                if realization_option_varname not in {
                    # Linear realization options
                    'velocitiesfromdisplacements',
                    'backscaling',
                    # Non-linear realization options
                    'structure',
                    'compoundorder',
                }:
                    abort(
                        f'Realization option "{realization_option_varname}" (specified for '
                        f'component "{self.name}" not recognized.'
                    )
        if self.representation == 'particles':
            # None of the non-linear relization options
            # makes sense for particle components.
            for realization_options_varname in realization_options.values():
                del realization_options_varname['structure']
                del realization_options_varname['compoundorder']
        elif self.representation == 'fluid':
            # None of the non-linear relization options
            # makes sense for ϱ.
            for realization_options_varname in (
                realization_options[unicode('ϱ')],
                realization_options[asciify('ϱ')],
            ):
                del realization_options_varname['structure']
                del realization_options_varname['compoundorder']
        for varname, realization_options_varname in realization_options.items():
            if varname != 'mom':
                if realization_options_varname['velocitiesfromdisplacements']:
                    masterwarn(
                        f'The "velocities from displacements" realization option was set to True '
                        f'for the "{varname}" variable of the "{self.name}" component, but in only '
                        f'makes sense for the "mom" variable'
                        + ('' if self.representation == 'particles' else
                            ' (and only for particle components)')
                    )
                del realization_options_varname['velocitiesfromdisplacements']
        self.realization_options = realization_options
        # Set approximations. Ensure that all implemented approximations
        # get set to either True or False. If an approximation is not
        # set for this component, its value defaults to False.
        # Also, specific circumstances may force some approximations to
        # have a specific value.
        if approximations is None:
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
                    f'The component "{self.name}" was initialized '
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
        if softening_length is None:
            softening_length = is_selected(self, select_softening_length)
        if isinstance(softening_length, str):
            # Evaluate softening_length if it's a str.
            # Replace 'N' with the number of particles of this component
            # and 'gridsize' with the gridsize of this component.
            if self.representation == 'particles':
                softening_length = softening_length.replace('N', str(self.N))
                softening_length = softening_length.replace('gridsize', str(cbrt(self.N)))
            elif self.representation == 'fluid':
                softening_length = softening_length.replace('N', str(self.gridsize**3))
                softening_length = softening_length.replace('gridsize', str(self.gridsize))
            softening_length = eval(softening_length, globals(), units_dict)
        if not softening_length:
            if self.representation == 'particles':
                # If no name is given, this is an internally
                # used component, in which case it is OK not to have
                # any softening lenth set.
                if self.name:
                    masterwarn(f'No softening length set for particles component "{self.name}"')
            softening_length = 0
        self.softening_length = float(softening_length)
        # This attribute will store the conserved mean density
        # of this component. It is set by the ϱ_bar method.
        self._ϱ_bar = -1
        # Particle attributes
        self.mass = mass
        self.N_allocated = 1
        self.N_local = 1
        # Particle data
        self.posx = malloc(self.N_allocated*sizeof('double'))
        self.posy = malloc(self.N_allocated*sizeof('double'))
        self.posz = malloc(self.N_allocated*sizeof('double'))
        self.momx = malloc(self.N_allocated*sizeof('double'))
        self.momy = malloc(self.N_allocated*sizeof('double'))
        self.momz = malloc(self.N_allocated*sizeof('double'))
        self.posx_mv = cast(self.posx, 'double[:self.N_allocated]')
        self.posy_mv = cast(self.posy, 'double[:self.N_allocated]')
        self.posz_mv = cast(self.posz, 'double[:self.N_allocated]')
        self.momx_mv = cast(self.momx, 'double[:self.N_allocated]')
        self.momy_mv = cast(self.momy, 'double[:self.N_allocated]')
        self.momz_mv = cast(self.momz, 'double[:self.N_allocated]')
        # Pack particle data into a pointer array of pointers
        # and a list of memoryviews.
        self.pos = malloc(3*sizeof('double*'))
        self.pos[0] = self.posx
        self.pos[1] = self.posy
        self.pos[2] = self.posz
        self.mom = malloc(3*sizeof('double*'))
        self.mom[0] = self.momx
        self.mom[1] = self.momy
        self.mom[2] = self.momz
        self.pos_mv = [self.posx_mv, self.posy_mv, self.posz_mv]
        self.mom_mv = [self.momx_mv, self.momy_mv, self.momz_mv]
        # Particle data buffers
        self.Δposx = malloc(self.N_allocated*sizeof('double'))
        self.Δposy = malloc(self.N_allocated*sizeof('double'))
        self.Δposz = malloc(self.N_allocated*sizeof('double'))
        self.Δmomx = malloc(self.N_allocated*sizeof('double'))
        self.Δmomy = malloc(self.N_allocated*sizeof('double'))
        self.Δmomz = malloc(self.N_allocated*sizeof('double'))
        self.Δposx_mv = cast(self.Δposx, 'double[:self.N_allocated]')
        self.Δposy_mv = cast(self.Δposy, 'double[:self.N_allocated]')
        self.Δposz_mv = cast(self.Δposz, 'double[:self.N_allocated]')
        self.Δmomx_mv = cast(self.Δmomx, 'double[:self.N_allocated]')
        self.Δmomy_mv = cast(self.Δmomy, 'double[:self.N_allocated]')
        self.Δmomz_mv = cast(self.Δmomz, 'double[:self.N_allocated]')
        # Pack particle data buffers into a pointer array of pointers
        # and a list of memoryviews.
        self.Δpos = malloc(3*sizeof('double*'))
        self.Δpos[0] = self.Δposx
        self.Δpos[1] = self.Δposy
        self.Δpos[2] = self.Δposz
        self.Δmom = malloc(3*sizeof('double*'))
        self.Δmom[0] = self.Δmomx
        self.Δmom[1] = self.Δmomy
        self.Δmom[2] = self.Δmomz
        self.Δpos_mv = [self.Δposx_mv, self.Δposy_mv, self.Δposz_mv]
        self.Δmom_mv = [self.Δmomx_mv, self.Δmomy_mv, self.Δmomz_mv]
        # Fluid attributes
        self.boltzmann_order = boltzmann_order
        if self.representation == 'particles':
            if self.boltzmann_order != 2:
                abort(
                    f'Particle components must have boltzmann_order = 2, '
                    f'but boltzmann_order = {self.boltzmann_order} was specified for "{self.name}"'
                )
        elif self.representation == 'fluid':
            if self.boltzmann_order < 0:
                abort(
                    f'Having boltzmann_order < 0 are nonsensical, '
                    f'but boltzmann_order = {self.boltzmann_order} was specified for "{self.name}"'
                )
            if self.boltzmann_order == 0 and self.boltzmann_closure == 'truncate':
                abort(
                    f'The fluid component "{self.name}" has no non-linear and no '
                    f'linear fluid variables, and so practically it does not exist. '
                    f'Such components are disallowed.'
                )
            if self.boltzmann_order == 3 and self.boltzmann_closure == 'class':
                abort(
                    f'The "{self.name}" component wants to close the Boltzmann hierarchy using '
                    f'the linear variable after ς from class, which is not implemented'
                )
            if self.boltzmann_order > 3:
                abort(
                    f'Fluids with boltzmann_order > 3 are not implemented, '
                    f'but boltzmann_order = {self.boltzmann_order} was specified for "{self.name}"'
                )
        self.shape = (1, 1, 1)
        self.shape_noghosts = (1, 1, 1)
        self.size = np.prod(self.shape)
        self.size_noghosts = np.prod(self.shape_noghosts)
        # Set the equation of state parameter w
        if w is None:
            w = is_selected(self, select_eos_w)
        self.initialize_w(w)
        self.initialize_w_eff()
        # Fluid data.
        # Create the boltzmann_order non-linear fluid variables and
        # store them in the fluidvars list. This is done even for
        # particle components, as the fluidscalars are all instantiated
        # with a gridsize of 1. The is_linear argument specifies whether
        # the FluidScalar will be a linear or non-linear variable,
        # where a non-linear variable is one that is updated non-
        # linearly, as opposed to a linear variable which is only
        # updated through continuous realization. Currently, only ϱ and
        # J is implemented as non-linear variables. It is still allowed
        # to have boltzmann_order == 3, in which case ς (and 𝒫) is also
        # specified as being non-linear, although no non-linear
        # evolution is implemented, meaning that these will then be
        # constant in time. Note that the 𝒫 fuid variable is
        # treated specially, as it really lives on the same tensor as
        # the ς fluid scalars. Therefore, the 𝒫 fluid scalar is added later.
        self.fluidvars = []
        for i in range(self.boltzmann_order):
            # Instantiate the i'th fluid variable
            # as a 3×3×...×3 (i times) symmetric tensor.
            fluidvar = Tensor(self, i, (3, )*i, symmetric=True)
            # Populate the tensor with fluid scalar fields
            for multi_index in fluidvar.multi_indices:
                fluidvar[multi_index] = FluidScalar(i, multi_index, is_linear=False)
            # Add the fluid variable to the list
            self.fluidvars.append(fluidvar)
        # If CLASS should be used to close the Boltzmann hierarchy,
        # we need one additional fluid variable. This should act like
        # a symmetric tensor of rank boltzmann_order, but really only a
        # single element of this tensor need to exist in memory.
        # For boltzmann_order == 2, ς is the additional fluid variable.
        # Instantiate the scalar element but disguised as a
        # 3×3×...×3 (boltzmann_order times) symmetric tensor.
        # Importantly, this fluid variabe is always considered linear.
        if self.boltzmann_closure == 'class':
            disguised_scalar = Tensor(
                self,
                self.boltzmann_order,
                (3, )*self.boltzmann_order,
                symmetric=True,
            )
            # Populate the tensor with a fluidscalar
            multi_index = disguised_scalar.multi_indices[0]
            disguised_scalar[multi_index] = FluidScalar(
                self.boltzmann_order, multi_index, is_linear=True,
            )
            # Add this additional fluid variable to the list
            self.fluidvars.append(disguised_scalar)
        # Ensure that the approximation P=wρ is set to True
        # for fluid components which have either a linear J
        # fluid variable, or a non-linear J fluid variable but with the
        # non-linear Boltzmann hierarchy truncated right after J.
        if not self.approximations['P=wρ']:
            if (   self.boltzmann_order < 1
                or (self.boltzmann_order == 1 and self.boltzmann_closure == 'truncate')):
                # The 𝒫 fluid scalar does not exist at all for
                # this component, and so whether the P=wρ is True or not
                # does not make much sense. We set it to True,
                # reflecting the fact that 𝒫 certainly is not a
                # non-linear variable.
                self.approximations[asciify('P=wρ')] = True
                self.approximations[unicode('P=wρ')] = True
            elif self.boltzmann_order == 1 and self.boltzmann_closure == 'class':
                masterwarn(
                    f'The P=wρ approximation has been switched on for the "{self.name}" component '
                    f'because Jⁱ = a⁴(ρ + c⁻²P)uⁱ is a linear fluid variable.'
                )
                self.approximations[asciify('P=wρ')] = True
                self.approximations[unicode('P=wρ')] = True
            elif self.boltzmann_order == 2 and self.boltzmann_closure == 'truncate':
                masterwarn(
                    f'The P=wρ approximation has been switched on for the "{self.name}" component '
                    f'because the non-linear Boltzmann hierarchy is truncated after the second '
                    f'non-linear fluid variable Jⁱ, while 𝒫 is part of the third fluid variable.'
                )
                self.approximations[asciify('P=wρ')] = True
                self.approximations[unicode('P=wρ')] = True
        # When the P=wρ approximation is False, the fluid variable 𝒫
        # has to follow the structure of ϱ closely. Otherwise, spurious
        # features will develop in ϱ. Display a warning if the
        # realization option for 𝒫 is set so that its structure does not
        # match that of ϱ.
        if not self.approximations['P=wρ']:
            if self.realization_options['𝒫']['structure'] == 'primordial':
                masterwarn(
                    f'It is specified that the 𝒫 fluid variable of the "{self.name}" component '
                    f'should be realized using the primordial structure throughout time. '
                    f'It is known that this generates spurious features.'
                )
        # When the P=wρ approximation is True, the 𝒫 fluid variable is
        # superfluous. Yet, as it is used in the definition of J,
        # J = a⁴(ρ + P)u, P = a**(-3*(1 + w_eff))*𝒫, it is simplest to
        # just always instantiate a complete 𝒫 fluid variable,
        # regardless of whether 𝒫 appears in the closed
        # Boltzmann hierarchy. We place 𝒫 on ς, since 𝒫 is the trace
        # missing from ς. The only time we do not instantiate 𝒫 is for
        # a fluid without any J variable, be it linear or non-linear.
        if not (   self.boltzmann_order < 1
                or (self.boltzmann_order == 1 and self.boltzmann_closure == 'truncate')):
            # We need a 𝒫 fluid scalar
            if (   (self.boltzmann_order == 1 and self.boltzmann_closure == 'class')
                or (self.boltzmann_order == 2 and self.boltzmann_closure == 'truncate')
                ):
                # The ς tensor on which 𝒫 lives does not yet exist.
                # Instantiate a fake ς tensor, used only to store 𝒫.
                self.fluidvars.append(Tensor(self, 2, (), symmetric=True, active=False))
            # Add the 𝒫 fluid scalar to the ς tensor
            self.fluidvars[2]['trace'] = FluidScalar(0, 0,
                is_linear=(self.boltzmann_order < 3 or self.approximations['P=wρ']),
            )
        # Construct mapping from names of fluid variables (e.g. J)
        # to their indices in self.fluidvars, and also from names of
        # fluid scalars (e.g. ϱ, Jx) to tuple of the form
        # (index, multi_index). The fluid scalar is then given
        # by self.fluidvars[index][multi_index].
        # Also include trivial mappings from indices to themselves,
        # and the special "reverse" mapping from indices to names
        # given by the 'ordered' key.
        self.fluid_names = {'ordered': fluidvar_names[:
                self.boltzmann_order + (0 if self.boltzmann_closure == 'truncate' else 1)
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
        # Aditional fluid scalars
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
        if self._ϱ_bar == -1:
            if self.representation == 'particles':
                # For particles, ϱ_bar = a**3*ρ_bar because w = 0,
                # which can be easily evaluated at the present time
                # if the mass is set.
                # Otherwise, ask CLASS for ρ_bar at the present time.
                if self.mass == -1:
                    if enable_class_background:
                        cosmoresults = compute_cosmo(
                            class_call_reason=f'in order to determine ̅ϱ of {self.name} ',
                        )
                        self._ϱ_bar = cosmoresults.ρ_bar(1, self.class_species)
                    else:
                        abort(
                            f'Cannot determine ϱ_bar for particle component "{self.name}" because '
                            f'its mass is not (yet?) set and enable_class_background is False'
                            )
                else:
                    self._ϱ_bar = self.N*self.mass/boxsize**3
            elif self.representation == 'fluid':
                # For fluids, ask CLASS for ρ_bar at the present time.
                # If CLASS background computations are disabled,
                # measure ϱ_bar directly from the component data.
                if self.mass == -1:
                    if enable_class_background:
                        cosmoresults = compute_cosmo(
                            class_call_reason=f'in order to determine ̅ϱ of {self.name} ',
                        )
                        self._ϱ_bar = cosmoresults.ρ_bar(1, self.class_species)
                    else:
                        self._ϱ_bar, σϱ, ϱ_min = measure(self, 'ϱ')
                        if self._ϱ_bar == 0:
                            masterwarn(
                                f'Failed to measure ̅ϱ of {self.name}. '
                                f'Try specifying the (fluid element) mass.'
                            )
                else:
                    self._ϱ_bar = (self.gridsize/boxsize)**3*self.mass
        return self._ϱ_bar

    # This method populate the Component pos/mom arrays (for a
    # particles representation) or the fluid scalar grids (for a
    # fluid representation) with data. It is deliberately designed so
    # that you have to make a call for each attribute (posx, posy, ...
    # for particle components, ϱ, Jx, Jy, ... for fluid components).
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
        fluid_indices=object,  # tuple or int-like
        fluidscalar='FluidScalar',
        index='Py_ssize_t',
        mv1D='double[::1]',
        mv3D='double[:, :, :]',
    )
    def populate(self, data, var, multi_index=0, buffer=False):
        """For fluids, the data should not include pseudo
        or ghost points.
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
            if var == 'posx':
                if buffer:
                    self.Δposx_mv[:self.N_local] = mv1D[:]
                else:
                    self.posx_mv [:self.N_local] = mv1D[:]
            elif var == 'posy':
                if buffer:
                    self.Δposy_mv[:self.N_local] = mv1D[:]
                else:
                    self.posy_mv [:self.N_local] = mv1D[:]
            elif var == 'posz':
                if buffer:
                    self.Δposz_mv[:self.N_local] = mv1D[:]
                else:
                    self.posz_mv [:self.N_local] = mv1D[:]
            elif var == 'momx':
                if buffer:
                    self.Δmomx_mv[:self.N_local] = mv1D[:]
                else:
                    self.momx_mv [:self.N_local] = mv1D[:]
            elif var == 'momy':
                if buffer:
                    self.Δmomy_mv[:self.N_local] = mv1D[:]
                else:
                    self.momy_mv [:self.N_local] = mv1D[:]
            elif var == 'momz':
                if buffer:
                    self.Δmomz_mv[:self.N_local] = mv1D[:]
                else:
                    self.momz_mv [:self.N_local] = mv1D[:]
            elif master:
                abort('Wrong component attribute name "{}"!'.format(var))
        elif self.representation == 'fluid':
            mv3D = data
            # The fluid scalar will be given as
            # self.fluidvars[index][multi_index],
            # where index is an int and multi_index is a tuple of ints.
            # These may be given directly as var and multi_index.
            # Alternatively, var may be a str, in which case it can be
            # the name of a fluid variable or a fluid scalar.
            # For each possibility, find index and multi_index.
            if isinstance(var, int):
                if not (0 <= var < len(self.fluidvars)):
                    abort('The "{}" component does not have a fluid variable with index {}'
                          .format(self.name, var))
                # The fluid scalar is given as
                # self.fluidvars[index][multi_index].
                index = var
            if isinstance(var, str):
                if var not in self.fluid_names:
                    abort('The "{}" component does not contain a fluid variable with the name "{}"'
                          .format(self.name, var))
                # Lookup the fluid indices corresponding to var.
                # This can either be a tuple of the form
                # (index, multi_index) (for a passed fluid scalar name)
                # or just an index (for a passed fluid name).
                fluid_indices = self.fluid_names[var]
                if isinstance(fluid_indices, int):
                    index = fluid_indices
                else:  # fluid_indices is a tuple
                    if multi_index != 0:
                        masterwarn('Overwriting passed multi_index ({}) with {}, '
                                   'deduced from the passed var = {}.'
                                   .format(multi_index, fluid_indices[1], var)
                                   )
                    index, multi_index = fluid_indices
            # Type check on the multi_index
            if not isinstance(multi_index, (int, tuple, str)):
                abort(
                    f'A multi_index of type "{type(multi_index)}" was supplied. '
                    f'This should be either an int, a tuple or a str.'
                )
            # Reallocate fluid grids if necessary
            self.resize(asarray(mv3D).shape)
            # Populate the scalar grid given by index and multi_index
            # with the passed data. This data should not
            # include pseudo or ghost points.
            fluidscalar = self.fluidvars[index][multi_index]
            if buffer:
                fluidscalar.Δ_noghosts[
                    :mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]
            else:
                fluidscalar.grid_noghosts[
                    :mv3D.shape[0], :mv3D.shape[1], :mv3D.shape[2]] = mv3D[...]
            # Populate pseudo and ghost points
            if buffer:
                communicate_domain(fluidscalar.Δ_mv   , mode='populate')
            else:
                communicate_domain(fluidscalar.grid_mv, mode='populate')

    # This method will grow/shrink the data attributes.
    # Note that it will update N_allocated but not N_local.
    @cython.pheader(# Arguments
                    size_or_shape_nopseudo_noghosts=object,  # Py_ssize_t or tuple
                    # Locals
                    fluidscalar='FluidScalar',
                    s='Py_ssize_t',
                    shape_nopseudo_noghosts=tuple,
                    size='Py_ssize_t',
                    s_old='Py_ssize_t',
                    )
    def resize(self, size_or_shape_nopseudo_noghosts):
        if self.representation == 'particles':
            size = size_or_shape_nopseudo_noghosts
            if size != self.N_allocated:
                self.N_allocated = size
                # Reallocate particle data
                self.posx = realloc(self.posx, self.N_allocated*sizeof('double'))
                self.posy = realloc(self.posy, self.N_allocated*sizeof('double'))
                self.posz = realloc(self.posz, self.N_allocated*sizeof('double'))
                self.momx = realloc(self.momx, self.N_allocated*sizeof('double'))
                self.momy = realloc(self.momy, self.N_allocated*sizeof('double'))
                self.momz = realloc(self.momz, self.N_allocated*sizeof('double'))
                # Reassign particle data memory views
                self.posx_mv = cast(self.posx, 'double[:self.N_allocated]')
                self.posy_mv = cast(self.posy, 'double[:self.N_allocated]')
                self.posz_mv = cast(self.posz, 'double[:self.N_allocated]')
                self.momx_mv = cast(self.momx, 'double[:self.N_allocated]')
                self.momy_mv = cast(self.momy, 'double[:self.N_allocated]')
                self.momz_mv = cast(self.momz, 'double[:self.N_allocated]')
                # Repack particle data into pointer arrays of pointers
                # and lists of memoryviews.
                self.pos[0], self.pos[1], self.pos[2] = self.posx, self.posy, self.posz
                self.mom[0], self.mom[1], self.mom[2] = self.momx, self.momy, self.momz
                self.pos_mv = [self.posx_mv, self.posy_mv, self.posz_mv]
                self.mom_mv = [self.momx_mv, self.momy_mv, self.momz_mv]
                # Reallocate particle buffers
                # (commented as these are not currently used).
                #self.Δposx = realloc(self.Δposx, self.N_allocated*sizeof('double'))
                #self.Δposy = realloc(self.Δposy, self.N_allocated*sizeof('double'))
                #self.Δposz = realloc(self.Δposz, self.N_allocated*sizeof('double'))
                self.Δmomx = realloc(self.Δmomx, self.N_allocated*sizeof('double'))
                self.Δmomy = realloc(self.Δmomy, self.N_allocated*sizeof('double'))
                self.Δmomz = realloc(self.Δmomz, self.N_allocated*sizeof('double'))
                # Reassign particle buffer memory views
                # (commented as these are not currently used).
                #self.Δposx_mv = cast(self.Δposx, 'double[:self.N_allocated]')
                #self.Δposy_mv = cast(self.Δposy, 'double[:self.N_allocated]')
                #self.Δposz_mv = cast(self.Δposz, 'double[:self.N_allocated]')
                self.Δmomx_mv = cast(self.Δmomx, 'double[:self.N_allocated]')
                self.Δmomy_mv = cast(self.Δmomy, 'double[:self.N_allocated]')
                self.Δmomz_mv = cast(self.Δmomz, 'double[:self.N_allocated]')
                # Repack particle buffers into pointer arrays of
                # pointers and lists of memoryviews.
                self.Δpos[0], self.Δpos[1], self.Δpos[2] = self.Δposx, self.Δposy, self.Δposz
                self.Δmom[0], self.Δmom[1], self.Δmom[2] = self.Δmomx, self.Δmomy, self.Δmomz
                self.Δpos_mv = [self.Δposx_mv, self.Δposy_mv, self.Δposz_mv]
                self.Δmom_mv = [self.Δmomx_mv, self.Δmomy_mv, self.Δmomz_mv]
                # Nullify the newly allocated Δ buffer
                self.nullify_Δ()
        elif self.representation == 'fluid':
            shape_nopseudo_noghosts = size_or_shape_nopseudo_noghosts
            # The allocated shape of the fluid grids are 5 points
            # (one layer of pseudo points and two layers of ghost points
            # before and after the local grid) longer than the local
            # shape, in each direction.
            if not any([2 + s + 1 + 2 != s_old for s, s_old in zip(shape_nopseudo_noghosts,
                                                                   self.shape)]):
                return
            if any([s < 1 for s in shape_nopseudo_noghosts]):
                abort('Attempted to resize fluid grids of the {} component'
                      'to a shape of {}'.format(self.name, shape_nopseudo_noghosts))
            # Recalculate and reassign meta data
            self.shape          = tuple([2 + s + 1 + 2 for s in shape_nopseudo_noghosts])
            self.shape_noghosts = tuple([    s + 1     for s in shape_nopseudo_noghosts])
            self.size           = np.prod(self.shape)
            self.size_noghosts  = np.prod(self.shape_noghosts)
            # Reallocate fluid data
            for fluidscalar in self.iterate_fluidscalars():
                fluidscalar.resize(shape_nopseudo_noghosts)

    # Method for 3D realisation of linear transfer functions.
    # As all arguments are optional,
    # this has to be a pure Python method.
    def realize(self, variables=None,
                      transfer_spline=None,
                      cosmoresults=None,
                      specific_multi_index=None,
                      a=-1,
                      gauge='N-body',
                      options=None,
                      use_gridˣ=False,
                      ):
        """This method will realise a given fluid/particle variable from
        a given transfer function. Any existing data for the variable
        in question will be lost.
        The variables argument specifies which variable(s) of the
        component to realise. Valid formats of this argument can be seen
        in varnames2indices. If no variables argument is passed,
        transfer functions for each variable will be computed via CLASS
        and all of them will be realized.
        If a specific_multi_index is passed, only the single fluidscalar
        of the variable(s) with the corresponding multi_index
        will be realized. If no specific_multi_index is passed,
        all fluidscalars of the variable(s) will be realized.
        The transfer_spline is a Spline object of the transfer function
        of the variable which should be realised.
        The cosmoresults argument is a linear.CosmoResults object
        containing all results from the CLASS run which produced the
        transfer function, from which further information
        can be obtained.
        Specify the scale factor a if you want to realize the variables
        at a time different from the present time.
        If neither the transfer_spline nor the cosmoresults argument is
        given, these will be produced by calling CLASS.
        You can supply multiple variables in one go,
        but then you have to leave the transfer_spline and cosmoresults
        arguments unspecified (as you can only pass in a
        single transfer_spline).
        The gauge and options arguments are passed on to
        linear.compute_transfer and linear.realize, respectively.
        See these functions for further detail.
        The use_gridˣ argument is passed on to linear.relize and
        determines whether the unstarred or starred grids should be used
        when doing the realization.
        """
        if a == -1:
            a = universals.a
        if options is None:
            options = {}
        options = {key.lower().replace(' ', '').replace('-', ''):
            (val.lower().replace(' ', '').replace('-', '') if isinstance(val, str) else val)
            for key, val in options.items()
        }
        # Define the gridsize used by the realization (gridsize for
        # fluid components and ∛N for particle components) and resize
        # the data attributes if needed.
        # Also do some particles-only checks.
        if self.representation == 'particles':
            if self.N%nprocs != 0:
                abort(f'Cannot perform realization of particle component "{self.name}" '
                      f'with N = {self.N}, as N is not evenly divisible by {nprocs} processes.'
                      )
            if not isint(ℝ[cbrt(self.N)]):
                abort(f'Cannot perform realization of particle component "{self.name}" '
                      f'with N = {self.N}, as N is not a cubic number.'
                      )
            gridsize = int(round(ℝ[cbrt(self.N)]))
            self.N_local = self.N//nprocs
            self.resize(self.N_local)
        elif self.representation == 'fluid':
            gridsize = self.gridsize
            shape = tuple([gridsize//domain_subdivisions[dim] for dim in range(3)])
            self.resize(shape)
        # Check that the gridsize fulfills the requirements for FFT
        # and therefore for realizations.
        if gridsize%nprocs != 0:
            abort(f'Cannot perform realization of component "{self.name}" '
                  f'with gridsize = {gridsize}, as gridsize is not '
                  f'evenly divisible by {nprocs} processes.'
                  )
        for dim in range(3):
            if gridsize%domain_subdivisions[dim] != 0:
                abort(f'Cannot perform realization of component "{self.name}" '
                      f'with gridsize = {gridsize}, as the global grid of shape '
                      f'({gridsize}, {gridsize}, {gridsize}) cannot be divided '
                      f'according to the domain decomposition ({domain_subdivisions[0]}, '
                      f'{domain_subdivisions[1]}, {domain_subdivisions[2]}).'
                      )
        # Argument processing
        if transfer_spline is None and cosmoresults is not None:
            abort('The realize method was called with cosmoresults but no transfer_spline')
        if variables is None:
            if transfer_spline is not None:
                masterwarn('The realize method was called without specifying a variable, '
                           'though a transfer_spline is passed. '
                           'This transfer_spline will be ignored.')
            if cosmoresults is not None:
                masterwarn('The realize method was called without specifying a variable, '
                           'though a cosmoresults is passed. This cosmoresults will be ignored.')
            # Realize all variables
            variables = arange(self.boltzmann_order)
        else:
            # Realize one or more variables
            variables = any2list(variables)
            N_vars = len(variables)
            if N_vars > 1:
                # Realize multiple variables
                if transfer_spline is not None:
                    abort(f'The realize method was called with {N_vars} variables '
                          'while a transfer_spline was supplied as well')
                if cosmoresults is not None:
                    abort(f'The realize method was called with {N_vars} variables '
                          'while cosmoresults was supplied as well')
        variables = any2list(variables)
        # Prepare arguments to compute_transfer,
        # if no transfer_spline is passed.
        if transfer_spline is None:
            k_min, k_max, k_gridsize = get_default_k_parameters(gridsize)
        # Realize each of the variables in turn
        options_passed = options.copy()
        for variable in variables:
            options = options_passed.copy()
            # The special "realization" of 𝒫 when using
            # the P=wρ approximation.
            if (   self.representation == 'fluid'
                and variable == 2
                and specific_multi_index == 'trace'
                and transfer_spline is None
                and self.approximations['P=wρ']
                ):
                self.realize_𝒫(a, use_gridˣ)
            else:
                # Normal realization
                if (self.representation == 'particles'
                    and 'velocitiesfromdisplacements' not in options
                ):
                    # The 'velocities from displacements' option
                    options['velocitiesfromdisplacements'] = self.realization_options['mom'][
                        'velocitiesfromdisplacements']
                    # For particles, the Boltzmann order is always 2,
                    # corresponding to positions and momenta. However,
                    # when velocities are set to be realized from
                    # displacements, the momenta (proportional to the
                    # velocity field uⁱ) are constructed from the
                    # displacement field ψⁱ (using the linear growth
                    # rate f) during the Zel'dovich approximation. Thus,
                    # from a single realization of ψⁱ, both the
                    # positions and the momenta are constructed. We
                    # should then pass only the positions as the
                    # variable to be realized (the realize function will
                    # realize both positions and momenta when velocities
                    # are to be realized from displacements).
                    if options['velocitiesfromdisplacements'] and variable == 1:
                        break
                # The back-scaling option
                if 'backscaling' not in options:
                    if variable == 0:
                        options['backscaling'] = self.realization_options[
                            {'particles': 'pos', 'fluid': 'ϱ'}[self.representation]
                        ]['backscaling']
                    elif variable == 1:
                        options['backscaling'] = self.realization_options[
                            {'particles': 'mom', 'fluid': 'J'}[self.representation]
                        ]['backscaling']
                    elif variable == 2 and specific_multi_index == 'trace':
                        options = self.realization_options['𝒫']['backscaling']
                    elif variable == 2:
                        options = self.realization_options['ς']['backscaling']
                # Get transfer function if not passed
                if transfer_spline is None:
                    transfer_spline, cosmoresults = compute_transfer(
                        self,
                        variable,
                        k_min, k_max, k_gridsize,
                        specific_multi_index,
                        a,
                        gauge,
                    )
                # Do the realization
                realize(
                    self,
                    variable,
                    transfer_spline,
                    cosmoresults,
                    specific_multi_index,
                    a,
                    options,
                    use_gridˣ,
                )
                # Reset transfer_spline to None so that a transfer
                # function will be computed for the next variable.
                transfer_spline = None

    # Method for realizing a linear fluid scalar
    def realize_if_linear(
        self,
        variable,
        transfer_spline=None,
        cosmoresults=None,
        specific_multi_index=None,
        a=-1,
        gauge='N-body',
        options=None,
        use_gridˣ=False,
    ):
        """If the fluid scalar is not linear or does not exist at all,
        no realization will be performed and no exception will
        be raised.
        """
        if self.representation == 'particles':
            return
        # Check that the fluid variable exist
        try:
            variable = self.varnames2indices(variable, single=True)
        except (IndexError, KeyError):
            return
        # For all variables other than ϱ (variable == 0),
        # a specific_multi_index has to have been passed.
        if specific_multi_index is None:
            if variable == 0:
                specific_multi_index = 0
            else:
                abort(
                    f'The realize_if_linear function was called with variable = {variable} ≠ 0 '
                    f'but without any specific_multi_index'
                )
        # Check that the fluid scalar exist
        if specific_multi_index not in self.fluidvars[variable]:
            return
        # Get the non-linear realization options
        if options is None:
            if variable == 0:
                options = self.realization_options['ϱ']
            elif variable == 1:
                options = self.realization_options['J']
            elif variable == 2 and specific_multi_index == 'trace':
                options = self.realization_options['𝒫']
            elif variable == 2:
                options = self.realization_options['ς']
            else:
                abort(
                    f'Do not know how to extract realization options '
                    f'for fluid variable {variable}[{specific_multi_index}]'
                )
        # Do the realization if the passed variable really is linear
        if self.is_linear(variable, specific_multi_index):
            self.realize(
                variable,
                transfer_spline,
                cosmoresults,
                specific_multi_index,
                a,
                gauge,
                options,
                use_gridˣ,
            )
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
        (only that it is not non-linear), unless it is sure that the
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

    # Method for realizing 𝒫 when the P=wρ approximation is enabled
    @cython.header(
        # Arguments
        a='double',
        use_gridˣ='bint',
        # Locals
        i='Py_ssize_t',
        ϱ_ptr='double*',
        𝒫_ptr='double*',
    )
    def realize_𝒫(self, a=-1, use_gridˣ=False):
        """This method applies 𝒫 = c²wϱ if the P=wρ approximation
        is enabled. If not, an exception will be thrown. This method
        is called from the more general realize method. It is not the
        intend that this method should be called from anywhere else.
        """
        if a == -1:
            a = universals.a
        if self.approximations['P=wρ']:
            # Set 𝒫 equal to the current ϱ times the current c²w
            if use_gridˣ:
                ϱ_ptr = self.ϱ.gridˣ
                𝒫_ptr = self.𝒫.gridˣ
            else:
                ϱ_ptr = self.ϱ.grid
                𝒫_ptr = self.𝒫.grid
            for i in range(self.size):
                𝒫_ptr[i] = ϱ_ptr[i]*ℝ[light_speed**2*self.w(a=a)]
        else:
            abort(
                f'The realize_𝒫 method was called on the {self.name} component wich have P ≠ wρ. '
                f'You should call the more general realize method instead.'
            )

    # Method for integrating particle positions/fluid values
    # forward in time.
    # For fluid components, source terms are not included.
    @cython.header(# Arguments
                   ᔑdt=dict,
                   # Locals
                   i='Py_ssize_t',
                   momx='double*',
                   momy='double*',
                   momz='double*',
                   posx='double*',
                   posy='double*',
                   posz='double*',
                   rk_order='int',
                   scheme=str,
                   )
    def drift(self, ᔑdt):
        if self.representation == 'particles':
            masterprint('Drifting {} ...'.format(self.name))
            posx = self.posx
            posy = self.posy
            posz = self.posz
            momx = self.momx
            momy = self.momy
            momz = self.momz
            # Update positions
            for i in range(self.N_local):
                posx[i] += momx[i]*ℝ[ᔑdt['a**(-2)']/self.mass]
                posy[i] += momy[i]*ℝ[ᔑdt['a**(-2)']/self.mass]
                posz[i] += momz[i]*ℝ[ᔑdt['a**(-2)']/self.mass]
                # Toroidal boundaries
                posx[i] = mod(posx[i], boxsize)
                posy[i] = mod(posy[i], boxsize)
                posz[i] = mod(posz[i], boxsize)
            masterprint('done')
            # Some partiles may have drifted out of the local domain.
            # Exchange particles to the correct processes.
            exchange(self)
        elif self.representation == 'fluid':
            # Evolve the fluid due to flux terms using the scheme
            # specified in the user parameters.
            scheme = is_selected(self, fluid_scheme_select)
            if scheme == 'maccormack':
                # For the MacCormack scheme to do anything,
                # the J variable must exist.
                if not (
                        self.boltzmann_order == 0
                    or (self.boltzmann_order == 1 and self.boltzmann_closure == 'truncate')
                ):
                    masterprint(
                        f'Evolving fluid variables (flux terms, using the MacCormack scheme) '
                        f'of {self.name} ...'
                    )
                    maccormack(self, ᔑdt)
                    masterprint('done')
            elif scheme == 'kurganovtadmor':
                # For the Kurganov-Tadmor scheme to do anything,
                # the J variable must exist.
                if not (
                        self.boltzmann_order == 0
                    or (self.boltzmann_order == 1 and self.boltzmann_closure == 'truncate')
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
                    f'It was specified that the {self.name} component should be evolved using '
                    f'the "{scheme}" scheme, which is not implemented.'
                )

    # Method for integrating fluid values forward in time
    # due to "internal" source terms, meaning source terms that do not
    # result from interacting with other components.
    @cython.header(# Arguments
                   ᔑdt=dict,
                   # Locals
                   scheme=str,
                   )
    def apply_internal_sources(self, ᔑdt):
        if self.representation == 'particles':
            return
        scheme = is_selected(self, fluid_scheme_select)
        if scheme == 'maccormack':
            # For the MacCormack scheme we have three internal
            # source terms: The Hubble term in the continuity equation
            # and the pressure and shear term in the Euler equation.
            if (
                (   # The Hubble term
                        self.boltzmann_order > 0
                    and not self.approximations['P=wρ']
                    and enable_Hubble
                )
                or
                (   # The pressure term
                        self.boltzmann_order > 1
                    and not (self.w_type == 'constant' and self.w_constant == 0)
                )
                or
                (
                    # The shear term
                        self.boltzmann_order > 2
                    or (self.boltzmann_order == 2 and self.boltzmann_closure == 'class')
                )
            ):
                masterprint(f'Evolving fluid variables (internal source terms) of {self.name} ...')
                maccormack_internal_sources(self, ᔑdt)
                masterprint('done')
        elif scheme == 'kurganovtadmor':
            # Only the Hubble term in the continuity equation
            # exist as an internal source term when using
            # the Kurganov Tadmor scheme.
            if (
                # The Hubble term
                    self.boltzmann_order > 0
                and not self.approximations['P=wρ']
                and enable_Hubble
            ):
                masterprint(f'Evolving fluid variables (internal source terms) of {self.name} ...')
                kurganov_tadmor_internal_sources(self, ᔑdt)
                masterprint('done')
        else:
            abort(f'It was specified that the {self.name} component should be evolved using '
                f'the "{scheme}" scheme, which is not implemented.'
            )

    # Method for computing the equation of state parameter w
    # at a certain time t or value of the scale factor a.
    # This has to be a pure Python function, otherwise it cannot
    # be called as w(a=a).
    def w(self, *, t=-1, a=-1):
        """This method should not be called before w has been
        initialized by the initialize_w method.
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
            abort(f'Did not recognize w type "{self.w_type}"')
        # For components with a non-linear evolution of ϱ,
        # we cannot handle w ≤ -1, as (ϱ + c⁻²𝒫) becomes non-positive.
        # This really should not be a problem, but the current fluid
        # implementation computes J/(ϱ + c⁻²𝒫) while solving the
        # continuity equation. If what is being run is not a simulation
        # but the CLASS utility, this is not a problem as the system
        # is not to be evolved.
        if value <= -1 and special_params.get('special') != 'CLASS':
            if (
                    (   self.boltzmann_order > 1
                     or (self.boltzmann_order == 1 and self.boltzmann_closure == 'class'))
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
        # but the CLASS utility, this is not a problem as the system
        # is not to be evolved.
        if value < 0 and special_params.get('special') != 'CLASS':
            if self.boltzmann_order > 1 and (a > universals.a_begin or t > universals.t_begin):
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
        initialized by the initialize_w_eff method.
        """
        # For constant w, w_eff = w
        if self.w_type == 'constant':
            return self.w(t=t, a=a)
        # If no time or scale factor value is passed,
        # use the current time and scale factor value.
        if t == -1 == a:
            t = universals.t
            a = universals.a
        # Compute w_eff
        if a == -1:
            a = scale_factor(t)
        value = self.w_eff_spline.eval(a)
        return value

    # Method for computing the proper time derivative
    # of the equation of state parameter w
    # at a certain time t or value of the scale factor a.
    # This has to be a pure Python function,
    # otherwise it cannot be called as ẇ(a=a).
    def ẇ(self, *, t=-1, a=-1):
        """This method should not be called before w has been
        initialized by the initialize_w method.
        """
        # If no time or scale factor value is passed,
        # use the current time and scale factor value.
        if t == -1 == a:
            t = universals.t
            a = universals.a
        # Compute the current ẇ dependent on its type
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
        abort('Did not recognize w type "{}"'.format(self.w_type))

    # Method which initializes the equation of state parameter w.
    # Call this before calling the w and ẇ methods.
    @cython.header(# Arguments
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
                   w_constant='double',
                   w_data='double[:, :]',
                   w_list=list,
                   w_ori=str,
                   w_tabulated='double[:]',
                   w_values='double[::1]',
                   ρ_tabulated=object,  # np.ndarray
                   returns='Spline',
                   )
    def initialize_w(self, w):
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
                      based on default_w and the species.
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
                      this should include eather t or a.
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
        # Initialize w dependent on its type
        try:
            w = float(w)
        except:
            pass
        if isinstance(w, float):
            # Assign passed constant w
            self.w_type = 'constant'
            self.w_constant = w
        elif isinstance(w, str) and w.lower() == 'class':
            # Get w as P_bar/ρ_bar from CLASS
            if not enable_class_background:
                abort(
                    f'Attempted to call CLASS to get the equation of state parameter w for the '
                    f'"{self.name}" component of CLASS species "{self.class_species}", '
                    f'but enable_class_background is False.'
                )
            self.w_type = 'tabulated (a)'
            cosmoresults = compute_cosmo(
                class_call_reason=f'in order to determine w(a) of {self.name} ',
            )
            background = cosmoresults.background
            i_tabulated = background['a']
            # For combination species it is still true that
            # w = c⁻²P_bar/ρ_bar, with P_bar and ρ_bar the sum of
            # individual background pressures and densities. Note that
            # the quantities in the background dict is given in CLASS
            # units, specifically c = 1.
            ρ_tabulated = 0
            p_tabulated = 0
            for class_species in self.class_species.split('+'):
                ρ_tabulated += background[f'(.)rho_{class_species}']
                p_tabulated += background[f'(.)p_{class_species}']
            w_tabulated = p_tabulated/ρ_tabulated
        elif isinstance(w, str) and w.lower() == 'default':
            # Assign w a constant value based on the species.
            # For combination species, the combined w is a weighted sum
            # of the invidual w's with the individual background
            # densities as weight, or equivalently, the ratio of the sum
            # of the invividual background pressures and the sum of the
            # individual background densities. To do it this proper way,
            # w should be passed in as 'class'. For w == 'default',
            # we simply do not handle this case.
            self.w_type = 'constant'
            w_constant = is_selected(self, default_w)
            try:
                self.w_constant = float(w_constant)
            except:
                abort(f'No default, constant w is defined for the "{self.species}" species')
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
                with open(w, 'r', encoding='utf-8') as w_file:
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
                                    unit_match = re.search('\((.*)\)', var)  # Applied later
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
                        abort('Time unit "{}" in header of "{}" not understood'
                            .format(unit_match.group(1), w))
                    else:
                        abort('Could not find time unit in header of "{}"'.format(w))
        elif isinstance(w, str):
            # Some expression for w was passed.
            # Insert '*' between all numbers and letters as well as all
            # bewteen numbers and opening parentheses and between
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
            except:
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
                self.w_tabulated = empty((2, w_tabulated.shape[0]))
                self.w_tabulated[0, :] = i_tabulated
                self.w_tabulated[1, :] = w_tabulated
            # Broadcast the tabulated w
            self.w_type = bcast(self.w_type)
            self.w_tabulated = smart_mpi(self.w_tabulated if master else (), mpifun='bcast')
            # If the tabulated w is constant, treat it as such
            w_values = asarray(tuple(set(self.w_tabulated[1, :])))
            if isclose(min(w_values), max(w_values)):
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
                    # the {w_0, w_a} parameterization, and so a linear
                    # spline should be used.
                    logx, logy = False, False
                self.w_spline = Spline(self.w_tabulated[0, :], self.w_tabulated[1, :],
                    f'w{self.w_type[len(self.w_type) - 3:]} of {self.name}',
                    logx=logx, logy=logy)

    # Method which initializes the effective
    # equation of state parameter w_eff.
    # Call this before calling the w_eff method,
    # but after calling the initialize_w method.
    @cython.header(# Locals
                   a='double',
                   a_min='double',
                   a_tabulated='double[::1]',
                   integrand_spline='Spline',
                   integrand_tabulated='double[::1]',
                   n_points='Py_ssize_t',
                   t='double',
                   t_tabulated='double[::1]',
                   w='double',
                   w_tabulated='double[::1]',
                   w_eff_tabulated_list=list,
                   )
    def initialize_w_eff(self):
        """This method initializes the effective equation of state
        parameter w_eff by defining the w_eff_spline attribute,
        which is used by the w_eff method to get w_eff(a).
        Only future times compared to universals.a will be included.
        The definition of w_eff is
        w_eff(a) = 1/log(a)∫₁ᵃ(w/a)da.
        """
        # For constant w, w_eff = w and so we do not need to do anything
        if self.w_type == 'constant':
            return
        masterprint(f'Tabulating effective equation of state parameter for {self.name} ...')
        # Construct tabulated arrays of matching scale factor and w.
        # If w is already tabulated at certain values, reuse these.
        if self.w_type == 'tabulated (a)':
            a_tabulated = self.w_tabulated[0, :]
            a_min = a_tabulated[0]
            w_tabulated = self.w_tabulated[1, :]
        elif self.w_type == 'tabulated (t)':
            t_tabulated = self.w_tabulated[0, :]
            a_tabulated = asarray([scale_factor(t) for t in t_tabulated])
            a_min = a_tabulated[0]
            w_tabulated = self.w_tabulated[1, :]
        else:
            a_min = universals.a
            n_points = 1000
            a_tabulated = logspace(log10(a_min), log10(1), n_points)
            w_tabulated = asarray([self.w(a=a) for a in a_tabulated])
        # Tabulate the integrand w/a
        integrand_tabulated = asarray([w/a for a, w in zip(a_tabulated, w_tabulated)])
        # For each tabulated point, find w_eff by doing the integral.
        # To do this we utilize the integrate method of the
        # Spline class. This is somewhat wasteful as the same initial
        # piece of the integral is computed over and over, but as this
        # is only ever done once per component we do not need to worry.
        # At a = 1 a division by 0 error occurs due to 1/log(a).
        # Here, we use the analytic result w_eff(a=1) = w(1=a).
        integrand_spline = Spline(a_tabulated, integrand_tabulated, f'w(a)/a of {self.name}')
        w_eff_tabulated_list = [integrand_spline.integrate(1, a)/log(a)
                                for a in a_tabulated[:(a_tabulated.size - 1)]]
        w_eff_tabulated_list.append(self.w(a=1))
        w_eff_tabulated = asarray(w_eff_tabulated_list)
        # Instantiate the w_eff spline object.
        # For most physical species, w_eff(a) is approximately a
        # power law in a and so a log-log spline should be used.
        logx, logy = True, True
        if np.any(asarray(w_eff_tabulated) <= 0):
            logy = False
        if self.class_species == 'fld':
            # The CLASS dark energy fluid (fld) uses
            # the {w_0, w_a} parameterization. It turns out that the
            # best spline is achieved from log(a) but linear w_eff.
            logx, logy = True, False
        self.w_eff_spline = Spline(a_tabulated, w_eff_tabulated, f'w_eff(a) of {self.name}',
            logx=logx, logy=logy)
        masterprint('done')

    # Method which convert named fluid/particle
    # variable names to indices.
    @cython.header(# Arguments
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
            if include_disguised_scalar or i < self.boltzmann_order:
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

    # Method for communicating pseudo and ghost points
    # of all fluid variables.
    @cython.header(# Arguments
                   mode=str,
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_fluid_grids(self, mode=''):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_domain(fluidscalar.grid_mv, mode=mode)

    # Method for communicating pseudo and ghost points
    # of all starred fluid variables.
    @cython.header(# Arguments
                   mode=str,
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_fluid_gridsˣ(self, mode=''):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_domain(fluidscalar.gridˣ_mv, mode=mode)

    # Method for communicating pseudo and ghost points
    # of all non-linear fluid variables.
    @cython.header(# Arguments
                   mode=str,
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_nonlinear_fluid_grids(self, mode=''):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            communicate_domain(fluidscalar.grid_mv, mode=mode)

    # Method for communicating pseudo and ghost points
    # of all starred non-linear fluid variables.
    @cython.header(# Arguments
                   mode=str,
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_nonlinear_fluid_gridsˣ(self, mode=''):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_nonlinear_fluidscalars():
            communicate_domain(fluidscalar.gridˣ_mv, mode=mode)

    # Method for communicating pseudo and ghost points
    # of all fluid Δ buffers.
    @cython.header(# Arguments
                   mode=str,
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def communicate_fluid_Δ(self, mode=''):
        if self.representation != 'fluid':
            return
        for fluidscalar in self.iterate_fluidscalars():
            communicate_domain(fluidscalar.Δ_mv, mode=mode)

    # Method which calls scale_grid on all non-linear fluid scalars
    @cython.header(# Arguments
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

    # Method which calls the nullify_Δ on all fluid scalars
    @cython.header(# Arguments
                   specifically=object,  # str og container of str's
                   # Locals
                   fluidscalar='FluidScalar',
                   )
    def nullify_Δ(self, specifically=None):
        if self.representation == 'particles':
            if specifically is None:
                self.Δposx_mv[...] = 0
                self.Δposy_mv[...] = 0
                self.Δposz_mv[...] = 0
                self.Δmomx_mv[...] = 0
                self.Δmomy_mv[...] = 0
                self.Δmomz_mv[...] = 0
            else:
                if 'pos' in specifically:
                    self.Δposx_mv[...] = 0
                    self.Δposy_mv[...] = 0
                    self.Δposz_mv[...] = 0
                if 'mom' in specifically:
                    self.Δmomx_mv[...] = 0
                    self.Δmomy_mv[...] = 0
                    self.Δmomz_mv[...] = 0
        elif self.representation == 'fluid':
            for fluidscalar in self.iterate_fluidscalars():
                fluidscalar.nullify_Δ()

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

    # This method is automaticlly called when a Component instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        # Free particle data
        # (fluid data lives on FluidScalar instances).
        free(self.pos)
        free(self.posx)
        free(self.posy)
        free(self.posz)
        free(self.mom)
        free(self.momx)
        free(self.momy)
        free(self.momz)

    # String representation
    def __repr__(self):
        return '<component "{}" of species "{}">'.format(self.name, self.species)
    def __str__(self):
        return self.__repr__()



# Function for getting the component representation based on the species
@cython.header(# Arguments
               species=str,
               # Locals
               key=tuple,
               representation=str,
               returns=str
               )
def get_representation(species):
    for key, representation in representation_of_species.items():
        if species in key:
            return representation
    abort('Species "{}" not implemented'.format(species))

# Function for adding species to the universals_dict,
# recording the presence of any species in use.
@cython.header(# Arguments
               components=list,
               # Locals
               class_species_present=set,
               class_species_present_bytes=bytes,
               class_species_previously_present=str,
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
    if not components:
        return
    # Species present (CO𝘕CEPT convention)
    species_present = {component.species for component in components}
    species_previously_present = universals_dict['species_present'].decode()
    if species_previously_present:
        species_present |= set(species_previously_present.split('+'))
    species_present_bytes = '+'.join(species_present).encode()
    universals_dict['species_present'] = species_present_bytes
    # Species present (CLASS convention)
    class_species_present = {component.class_species for component in components}
    class_species_previously_present = universals_dict['class_species_present'].decode()
    if class_species_previously_present:
        class_species_present |= set(class_species_previously_present.split('+'))
    class_species_present_bytes = (
        # A component.class_species may be a combination of several CLASS species
        '+'.join(set('+'.join(class_species_present).split('+'))).encode()
    )
    universals_dict['class_species_present'] = class_species_present_bytes



# Mapping from species to their representations
cython.declare(representation_of_species=dict)
representation_of_species = {
    ('baryons',
     'dark energy particles',
     'dark matter particles',
     'matter particles',
     'neutrinos',
     'photons',
     'particles',
     ): 'particles',
    ('baryon fluid',
     'dark energy fluid',
     'dark matter fluid',
     'matter fluid',
     'metric',
     'neutrino fluid',
     'photon fluid',
     'fluid',
     ): 'fluid',
}
# Mapping from (CO𝘕CEPT) species names to default
# CLASS species names. Note that combination species
# (e.g. matter) is expressed as e.g. 'cdm+b'.
cython.declare(default_class_species=dict)
default_class_species = {
    'baryon fluid'         : 'b',
    'baryons'              : 'b',
    'dark energy fluid'    : 'fld',
    'dark energy particles': 'fld',
    'dark matter fluid'    : 'cdm',
    'dark matter particles': 'cdm',
    'matter fluid'         : 'cdm+b',
    'matter particles'     : 'cdm+b',
    'metric'               : 'metric',
    'neutrino fluid'       : 'ncdm[0]',
    'neutrinos'            : 'ncdm[0]',
    'photon fluid'         : 'g',
    'photons'              : 'g',
}
# Mapping from species and representations to default w values
cython.declare(default_w=dict)
default_w = {
    'baryon fluid'         :  0,
    'baryons'              :  0,
    'dark energy fluid'    : -1,
    'dark energy particles': -1,
    'dark matter fluid'    :  0,
    'dark matter particles':  0,
    'matter fluid'         :  0,
    'matter particles'     :  0,
    'metric'               :  0,
    'neutrino fluid'       :  1/3,
    'neutrinos'            :  1/3,
    'particles'            :  0,
    'photons'              :  1/3,
    'photon fluid'         :  1/3,
}
# Set of all approximations implemented on Component objects
cython.declare(approximations_implemented=set)
approximations_implemented = {
    unicode('P=wρ'),
}
# Set of all component names used internally by the code,
# and which the user should generally avoid.
cython.declare(internally_defined_names=set)
internally_defined_names = {'all', 'all combinations', 'buffer', 'default', 'total'}
# Names of all implemented fluid variables in order.
# Note that 𝒫 is not considered a seperate fluid variable,
# but rather a fluid scalar that lives on ς.
cython.declare(fluidvar_names=tuple)
fluidvar_names = ('ϱ', 'J', 'ς')
# Flag specifying whether a warning should be given if multiple
# components with the same name are instantiated, and a set of names of
# all instantiated componenets.
cython.declare(allow_similarly_named_components='bint', component_names=set)
allow_similarly_named_components = False
component_names = set()

