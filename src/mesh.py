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
    '    get_buffer,           '
    '    partition,            '
    '    smart_mpi,            '
)

# Pure Python imports
from communication import get_domain_info

# Function pointer types used in this module
pxd('ctypedef double* (*func_dstar_ddd)(double, double, double)')

# Import declarations from fft.c
pxd("""
# FFT functionality via FFTW from fft.c
cdef extern from "fft.c":
    # The fftw_plan type
    ctypedef struct fftw_plan_struct:
        pass
    ctypedef fftw_plan_struct *fftw_plan
    # The returned struct of fftw_setup
    struct fftw_return_struct:
        ptrdiff_t gridsize_local_i
        ptrdiff_t gridsize_local_j
        ptrdiff_t gridstart_local_i
        ptrdiff_t gridstart_local_j
        double* grid
        fftw_plan plan_forward
        fftw_plan plan_backward
    # Functions
    fftw_return_struct fftw_setup(ptrdiff_t gridsize_i,
                                  ptrdiff_t gridsize_j,
                                  ptrdiff_t gridsize_k,
                                  char*     rigor,
                                  bint      fftw_wisdom_reuse,
                                  char*     wisdom_filename,
                                  )
    void fftw_execute(fftw_plan plan)
    void fftw_clean(double* grid, fftw_plan plan_forward,
                                  fftw_plan plan_backward)
    void fftw_free(double* grid)
""")



# Class representing one of the three lattice types
# (simple cubic, body-centered cubic, face-centered cubic).
@cython.cclass
class Lattice:

    # The shifts ∈ {0, ±½} in grid units of the different kinds of
    # lattices. The sign convention is such that the shifts are to be
    # applied to particles, not grids.
    # For cell-centred, the shifts are negative.
    # For cell-vertex, the shifts are positive.
    shift_amount = (1 - 2*cell_centered)*0.5
    shifts_all = {
        'sc': [  # simple cubic lattice
            (0, 0, 0),
        ],
        'bcc': [  # body-centered cubic lattice; 2 primitive sc lattices
            (    0,                   0,            0),
            (shift_amount, shift_amount, shift_amount),
        ],
        'fcc': [  # face-centered cubic lattice; 4 primitive sc lattices
            (           0,            0,            0),
            (           0, shift_amount, shift_amount),
            (shift_amount,            0, shift_amount),
            (shift_amount, shift_amount,            0),
        ],
    }

    # Initialisation method
    def __init__(self, kind='sc', single_primitive=None, negate_shifts=False):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Lattice type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        public str kind
        public list shifts
        public tuple shift
        public int index
        """
        # If a lattice is passed, copy its attributes
        if isinstance(kind, type(self)):
            lattice = kind
            self.kind   = lattice.kind
            self.shifts = lattice.shifts.copy()
            self.shift  = lattice.shift
            self.index  = lattice.index
            if negate_shifts:
                self.negate_shifts()
            return
        if not isinstance(kind, str):
            abort('The lattice kind must be given as a str')
        # The lattice type
        kind = kind.lower()
        if 'simple' in kind or not kind:
            kind = 'sc'  # simple cubic lattice
        elif 'body' in kind:
            kind = 'bcc'  # body-centered cubic lattice
        elif 'face' in kind:
            kind = 'fcc'  # face-centered cubic lattice
        if kind not in self.shifts_all:
            abort(f'Unrecognized lattice "{kind}" ∉ {set(self.shifts_all)}')
        self.kind = kind
        self.shifts = self.shifts_all[self.kind]
        # If only a single primitive sc lattice should be used,
        # forget about the others.
        if single_primitive is not None:
            self.shifts = [self.shifts[single_primitive%len(self)]]
        # Initialise current shift / primitive sc lattice
        self.reset()
        if negate_shifts:
            self.negate_shifts()

    # Method for resetting the current shift / primitive sc lattice
    def reset(self):
        self.index = 0
        self.shift = self.shifts[self.index]

    # Method for negating all shifts
    def negate_shifts(self):
        def negate_shift(shift):
            return tuple([-s for s in shift])
        self.shifts = [negate_shift(shift) for shift in self.shifts]
        self.shift = negate_shift(self.shift)

    # Iterating over a Lattice instance amounts to iterating over the
    # shifts / primitive sc lattices, with both the shift and index
    # attributes being updated. Note that what is yielded is the
    # instance itself, meaning that iteration should be done like
    #   for lattice in lattice:
    #       ...
    # or
    #   for _ in lattice:
    #       ...
    def __iter__(self):
        for self.index, self.shift in enumerate(self.shifts):
            yield self

    # The size of the lattice instance is considered to be the number of
    # primitive sc lattices in the full lattice. Note that this has
    # nothing to do with the grid size.
    def __len__(self):
        return len(self.shifts)

    # String representation
    def __repr__(self):
        return f'<lattice "{self.kind}" with shifts {self.shifts} at index {self.index}>'
    def __str__(self):
        return self.__repr__()


# Function for initialising and tabulating a cubic grid
# with vector values.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    func=func_dstar_ddd,
    factor='double',
    filename=str,
    # Locals
    dim='Py_ssize_t',
    grid='double[:, :, :, ::1]',
    grid_local='double[::1]',
    i='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    n_vectors='Py_ssize_t',
    n_vectors_local='Py_ssize_t',
    size='Py_ssize_t',
    size_edge='Py_ssize_t',
    size_face='Py_ssize_t',
    size_local='Py_ssize_t',
    start_local='Py_ssize_t',
    vector_value='double*',
    ℓ_local='Py_ssize_t',
    ℓ='Py_ssize_t',
    returns='double[:, :, :, ::1]',
)
def tabulate_vectorgrid(gridsize, func, factor, filename=''):
    """This function tabulates a cubic grid of size
    gridsize*gridsize*gridsize with vector values computed by
    the function func, as
    grid[i, j, k] = func(i*factor, j*factor, k*factor).
    If filename is set, the tabulated grid is saved to a hdf5 file
    with this name.
    The grid is not distributed. Each process will end up with its own
    copy of the entire grid.
    The first point grid[0, 0, 0] corresponds to the origin of the box,
    while the last point grid[gridsize - 1, gridsize - 1, gridsize - 1]
    corresponds to the physical point ((gridsize - 1)/gridsize*boxsize,
                                       (gridsize - 1)/gridsize*boxsize,
                                       (gridsize - 1)/gridsize*boxsize).
    This means that grid[gridsize, gridsize, gridsize] is the corner
    of the box the furthers away from the origin. This point is not
    represented in the grid because of the periodicity of the box,
    which means that physically, this point is the same as the origin.
    The mapping from grid indices to physical coordinates is thus
      (i, j, k) → (
        i/gridsize*boxsize,
        j/gridsize*boxsize,
        k/gridsize*boxsize,
      )
    Therefore, if you wish to tabulate a global (and periodic) field,
    extending over the entire box, the factor argument should be
    proportional to boxsize/gridsize. If you wish to only tabulate say
    one octant of the box, this volume is no more periodic, and so
    the factor should be proportional to 0.5*boxsize/(gridsize - 1).
    """
    # The number of elements in each point, edge, face and bulk
    # (the total), when the grid is viewed as a 3D box of vectors.
    size_edge = gridsize*3
    size_face = gridsize*size_edge
    size      = gridsize*size_face
    # Initialise the global grid to be
    # of shape gridsize*gridsize*gridsize*3.
    # That is, grid is not really cubic, but rather four-dimensional.
    grid = empty([gridsize, gridsize, gridsize, 3], dtype=C2np['double'])
    # Partition the grid fairly among the processes.
    # Each part is defined by a linear size and starting index.
    n_vectors = gridsize**3
    start_local, n_vectors_local = partition(n_vectors)
    # Each point really consists of three elements
    start_local *= 3
    size_local = 3*n_vectors_local
    # Make a linear array for storing the local part of the grid
    grid_local = empty(size_local, dtype=C2np['double'])
    # Tabulate the local grid
    for ℓ_local in range(0, size_local, 3):
        ℓ = start_local + ℓ_local
        # The global 3D indices
        i = ℓ//size_face
        ℓ -= i*size_face
        j = ℓ//size_edge
        ℓ -= j*size_edge
        k = ℓ//3
        # Fill grid_local
        vector_value = func(i*factor, j*factor, k*factor)
        for dim in range(3):
            grid_local[ℓ_local + dim] = vector_value[dim]
    # Gather the tabulated local grid parts into the common, global grid
    smart_mpi(grid_local, grid, mpifun='allgatherv')
    # Save grid to disk using parallel HDF5
    if filename:
        if master:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        Barrier()
        with open_hdf5(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
            dset = hdf5_file.create_dataset('data', (size, ), dtype=C2np['double'])
            dset[start_local:(start_local + size_local)] = grid_local
    return grid

# Function for doing lookup in a grid with vector values
@cython.header(
    # Argument
    grid='double[:, :, :, ::1]',
    x='double',
    y='double',
    z='double',
    order='int',
    factor='double',
    # Locals
    dim='Py_ssize_t',
    grid_ptr='double*',
    index='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    weight='double',
    returns='double*',
)
def interpolate_in_vectorgrid(grid, x, y, z, order, factor=1):
    """This function looks up tabulated vectors in a grid and
    interpolates to (x, y, z).
    Input arguments must be normalized so that
      0 <= x < grid.shape[0] - 1,
      0 <= y < grid.shape[1] - 1,
      0 <= z < grid.shape[2] - 1.
    It is assumed that the grid is non-periodic (that is, the first and
    the last gridpoint in any dimension are physically distinct). The
    grid is not (necessarily) a domain grid and has no ghost points.
    This means that the interpolation will fail if using higher than
    second-order (CIC) interpolation.
    """
    size_j, size_k = grid.shape[1], grid.shape[2]
    grid_ptr = cython.address(grid[:, :, :, :])
    for dim in range(3):
        vector[dim] = 0
    # Assign the weighted grid values to the vector components
    if order == 1:  # NGP interpolation
        for index, weight in particle_interpolation_loop_NGP(
            x, y, z, size_j, size_k,
            factor, apply_factor=True,
        ):
            index *= 3
            for dim in range(3):
                vector[dim] += grid_ptr[index + dim]*weight
    else:  # order == 2:  # CIC interpolation
        for index, weight in particle_interpolation_loop_CIC(
            x, y, z, size_j, size_k,
            factor, apply_factor=True,
        ):
            index *= 3
            for dim in range(3):
                vector[dim] += grid_ptr[index + dim]*weight
    return vector
# Vector used as the return value of the
# interpolate_in_vectorgrid() function.
cython.declare(vector='double*')
vector = malloc(3*sizeof('double'))

# Function for doing lookup in a grid with scalar values and
# interpolating to specified coordinates.
@cython.pheader(
    # Argument
    grid='double[:, :, ::1]',
    component='Component',
    variable=str,
    dim='int',
    order='int',
    lattice='Lattice',
    factor='double',
    # Locals
    cellsize='double',
    grid_ptr='double*',
    index='Py_ssize_t',
    indexˣ='Py_ssize_t',
    mv_dim='double[::1]',
    offset_x='double',
    offset_y='double',
    offset_z='double',
    posxˣ='double*',
    posyˣ='double*',
    poszˣ='double*',
    ptr_dim='double*',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    value='double',
    weight='double',
    x='double',
    y='double',
    z='double',
    returns='void',
)
def interpolate_domaingrid_to_particles(
    grid, component, variable, dim, order, lattice=None, factor=1,
):
    """This function updates the dim'th dimension of variable ('pos',
    'mom' or 'Δmom') of the component, through interpolation in the grid
    of a given order. If the grid values should be multiplied by a
    factor prior to adding them to the variable, this may be specified.
    """
    if not (1 <= order <= 4):
        abort(
            f'interpolate_domaingrid_to_particles() called '
            f'with order = {order} ∉ {{1, 2, 3, 4}}'
        )
    if lattice is None:
        lattice = Lattice()
    # Extract pointer to particle data,
    # indexed as ptr_dim[indexˣ] regardless of dim.
    if variable == 'pos':
        mv_dim = component.pos_mv[dim:]
    elif variable == 'mom':
        mv_dim = component.mom_mv[dim:]
    elif variable == 'Δmom':
        mv_dim = component.Δmom_mv[dim:]
    else:
        abort(
            f'interpolate_domaingrid_to_particles() called with variable = "{variable}" '
            f'∉ {{"pos", "mom"}}'
        )
    ptr_dim = cython.address(mv_dim[:])
    # Offsets needed for the interpolation.
    # Note that here we employ the reverse sign on the shifs compared
    # to in interpolate_particles().
    cellsize = domain_size_x/(grid.shape[0] - ℤ[2*nghosts])  # we have cubic grid cells
    offset_x = (
        + domain_bgn_x
        - (1 + machine_ϵ)*(nghosts - 0.5*cell_centered + lattice.shift[0])*cellsize
    )
    offset_y = (
        + domain_bgn_y
        - (1 + machine_ϵ)*(nghosts - 0.5*cell_centered + lattice.shift[1])*cellsize
    )
    offset_z = (
        + domain_bgn_z
        - (1 + machine_ϵ)*(nghosts - 0.5*cell_centered + lattice.shift[2])*cellsize
    )
    # Interpolate onto each particle
    posxˣ = component.posxˣ
    posyˣ = component.posyˣ
    poszˣ = component.poszˣ
    size_j, size_k = grid.shape[1], grid.shape[2]
    grid_ptr = cython.address(grid[:, :, :])
    for indexˣ in range(0, 3*component.N_local, 3):
        # Get, translate and scale the coordinates so that
        # nghosts - ½ < r < shape[r] - nghosts - ½ for r ∈ {x, y, z}.
        x = (posxˣ[indexˣ] - offset_x)*ℝ[(1/cellsize)*(1 - machine_ϵ)]
        y = (posyˣ[indexˣ] - offset_y)*ℝ[(1/cellsize)*(1 - machine_ϵ)]
        z = (poszˣ[indexˣ] - offset_z)*ℝ[(1/cellsize)*(1 - machine_ϵ)]
        # Carry out the interpolation according to the order
        value = 0
        with unswitch:
            if order == 1:  # NGP interpolation
                for index, weight in particle_interpolation_loop_NGP(
                    x, y, z, size_j, size_k,
                ):
                    value += grid_ptr[index]*weight
            elif order == 2:  # CIC interpolation
                for index, weight in particle_interpolation_loop_CIC(
                    x, y, z, size_j, size_k,
                ):
                    value += grid_ptr[index]*weight
            elif order == 3:  # TSC interpolation
                for index, weight in particle_interpolation_loop_TSC(
                    x, y, z, size_j, size_k,
                ):
                    value += grid_ptr[index]*weight
            else:  # order == 4  # PCS interpolation
                for index, weight in particle_interpolation_loop_PCS(
                    x, y, z, size_j, size_k,
                ):
                    value += grid_ptr[index]*weight
        with unswitch:
            if factor != 1:
                value *= factor
        ptr_dim[indexˣ] += value

# Function for interpolating a certain quantity from components
# (particles and fluids) onto global domain grids using intermediate
# upstream grids.
@cython.pheader(
    # Arguments
    components=list,
    gridsizes_upstream=list,
    gridsize_global='Py_ssize_t',
    quantity=str,
    order='int',
    ᔑdt=dict,
    deconvolve='bint',
    interlace=str,
    output_space=str,
    output_as_slabs='bint',
    do_ghost_communication='bint',
    # Locals
    component='Component',
    fft_factor='double',
    fluid_components=list,
    grid_global='double[:, :, ::1]',
    grid_upstream='double[:, :, ::1]',
    gridshape_upstream_local=tuple,
    gridsize_upstream='Py_ssize_t',
    group=dict,
    groups=dict,
    lattice='Lattice',
    particle_components=list,
    slab_global='double[:, :, ::1]',
    returns='double[:, :, ::1]',
)
def interpolate_upstream(
    components, gridsizes_upstream, gridsize_global, quantity, order,
    ᔑdt=None, deconvolve=True, interlace='sc', output_space='real',
    output_as_slabs=False, do_ghost_communication=True,
):
    """Given a list of components, a list of corresponding upstream grid
    sizes and a single global grid size, this function interpolates the
    components onto a global grid by first interpolating directly onto
    the upstream grids, which are then added together in Fourier space
    to obtain the global grid.

    The (particle) interpolation order is given by the order argument.
    No upstream interpolation takes place for fluid components,
    i.e. the passed υpstream grid sizes are assumed to equal the
    fluid grid size for fluid components.

    Whether to apply particle deconvolution and interlacing can be
    specified by deconvolve and interlace. The deconvolution order will
    be the same as the interpolation order (the order argument).

    The returned grid may either be a real space domain grid or a
    Fourier space slab, depending on the value of output_space
    (either 'real' or 'Fourier'). If 'real', the domain grids will have
    properly populated ghost points if do_ghost_communication is True.

    The quantity argument determines what should be interpolated onto
    the grid(s). Valid values are:
    'ρ': The returned grid(s) will hold physical densities. Note that
        ρ = a**(-3(1 + w_eff))*ϱ. Note that this physical 'ρ' is always
        preferable to the conserved 'ϱ' when multiple components are to
        be interpolated together, as only 'ρ' is additive across
        components/species. Each particle will contribute with
        a**(-3*w_eff)*mass/V_cell, a**(-3*w_eff)*mass being the current
        mass of the particle (a**(-3*w_eff) taking decay into account)
        and V_cell = (a*boxsize/gridsize)**3 being the physical grid
        cell volume. In total, each particle contribute with
        a**(-3*(1 + w_eff))*(gridsize/boxsize)**3*mass. Each fluid cell
        will contribute with a**(-3*(1 + w_eff))*ϱᵢⱼₖ*V_cell_fluid/V_cell,
        where a**(-3*(1 + w_eff))*ϱᵢⱼₖ = ρᵢⱼₖ is the physical density of
        fluid cell [i, j, k] and
        V_cell_fluid = (a*boxsize/gridsize_fluid)**3 is the physical
        cell volume of the fluid grid. In total, each fluid cell
        contribute with
        a**(-3*(1 + w_eff))*(gridsize/gridsize_fluid)**3*ϱᵢⱼₖ
    'a²ρ': The returned grid(s) will hold physical densities times the
        square of the scale factor. From the 'ρ' entry above, we then
        have that each particle will contribute with
        a**(-3*w_eff - 1)*(gridsize/boxsize)**3*mass and that each fluid
        cell will contribute with
        a**(-3*w_eff - 1)*(gridsize/gridsize_fluid)**3*ϱᵢⱼₖ.
    'ϱ': The returned grid(s) will hold the conserved densities. From
        the 'ρ' entry above, we then have that each particle will
        contribute with ϱ = (gridsize/boxsize)**3*mass and that each
        fluid cell will contribute with (gridsize/gridsize_fluid)**3*ϱᵢⱼₖ.
    'Jx', 'Jy', 'Jz'; described for Jx: The returned grid(s) will hold
        the conserved momentum density Jₓ = a**4*(ρ + c⁻²P)*uₓ. As this
        is built from physical quantities, this is additive across
        components/species. For particles we set P = 0, leaving
        Jₓ = a**4*ρ*uₓ. The canonical momentum momₓ and peculiar velocity
        uₓ is related by momₓ = a*(a**(-3*w_eff)*mass)*uₓ, and so from
        the particle construction of ρ above we get that each particle
        contribute with (gridsize/boxsize)**3*momₓ. As each fluid cell
        already stores Jₓ, they contribute by
        V_cell_fluid/V_cell*Jₓ = (gridsize/gridsize_fluid)**3*Jₓ.
    In all of the above, expressions involving a = a(t) will be
    evaluated at the current universal time, unless a dict ᔑdt of time
    step integrals is passed, in which case the expressions will be
    integrated over the time step.
    """
    output_space = output_space.lower()
    if output_space not in ('real', 'fourier'):
        abort(
            f'interpolate_upstream() got output_space = "{output_space}" ∉ {{"real", "Fourier"}}'
        )
    lattice = Lattice(interlace)
    # Group components according to their upstream grid size.
    # The order does not matter, except that we want the group with the
    # downstream grid size equal to the global grid size to be
    # the first, if indeed such a group exist.
    groups = group_components(components, gridsizes_upstream, [gridsize_global, ...])
    # The global Fourier slabs will be properly initialised below
    slab_global = None
    # For each group, interpolate the components onto upstream grids,
    # Fourier transform these upstream grids and add them
    # to the global slabs.
    for gridsize_upstream, group in groups.items():
        gridshape_upstream_local = get_gridshape_local(gridsize_upstream)
        particle_components = group.get('particles', [])
        fluid_components    = group.get('fluid',     [])
        # Normalization factor needed after a forward FFT
        fft_factor = float(gridsize_upstream)**(-3)
        # Fluid components
        if fluid_components:
            # Copy fluid components onto upstream grid
            grid_upstream = get_buffer(
                gridshape_upstream_local, 'grid_updownstream',
                nullify=False,
            )
            combine_fluids(fluid_components, grid_upstream, quantity, ᔑdt, fft_factor)
            # Transform the upstream grid to Fourier space
            # and add it to the global Fourier slabs.
            slab_global = add_upstream_to_global_slabs(
                grid_upstream, slab_global, gridsize_upstream, gridsize_global,
            )
        # Particle components
        if particle_components:
            for lattice in lattice:
                # Interpolate particle components onto upstream grid
                grid_upstream = get_buffer(
                    gridshape_upstream_local, 'grid_updownstream',
                    nullify=True,
                )
                for component in particle_components:
                    interpolate_particles(
                        component, gridsize_upstream, grid_upstream, quantity, order, ᔑdt,
                        lattice, fft_factor, do_ghost_communication=False,
                    )
                communicate_ghosts(grid_upstream, '+=')
                # Transform the upstream grid to Fourier space,
                # perform deconvolution and interlacing and add
                # the result to the global Fourier slabs.
                slab_global = add_upstream_to_global_slabs(
                    grid_upstream, slab_global, gridsize_upstream, gridsize_global,
                    deconv_order=ℤ[deconvolve*order], lattice=lattice,
                )
    # Global Fourier slabs complete. Note that we do not have to nullify
    # the Nyquist planes of these global slabs as we have nullified the
    # Nyquist planes of all upstream slabs from which the global slabs
    # are built.
    # Return now if Fourier space is what we want.
    if output_space == 'fourier':
        return slab_global
    # Fourier transform the global slabs to real space
    fft(slab_global, 'backward')
    # Return real space slabs if requested
    if output_as_slabs:
        return slab_global
    # Domain-decompose and return the global real space grid
    grid_global = get_buffer(get_gridshape_local(gridsize_global), 'grid_global')
    domain_decompose(
        slab_global, grid_global,
        do_ghost_communication=do_ghost_communication,
    )
    return grid_global

# Helper function for interpolate_upstream(), taking in a real space
# upstream grid and adding its Fourier modes to the global slabs.
@cython.header(
    # Arguments
    grid_upstream='double[:, :, ::1]',
    slab_global='double[:, :, ::1]',
    gridsize_upstream='Py_ssize_t',
    gridsize_global='Py_ssize_t',
    deconv_order='int',
    lattice='Lattice',
    # Locals
    nullification=str,
    operation=str,
    slab_upstream='double[:, :, ::1]',
    use_upstream_as_global='bint',
    returns='double[:, :, ::1]',
)
def add_upstream_to_global_slabs(
    grid_upstream, slab_global, gridsize_upstream, gridsize_global,
    deconv_order=0, lattice=None,
):
    # If the global slabs have yet to be initialised, we must do so
    # within this function. If at the same time the global and upstream
    # grid sizes are equal, we will use the upstream slabs directly as
    # the global slabs, rather than performing a copy.
    use_upstream_as_global = (slab_global is None and gridsize_global == gridsize_upstream)
    # Transform upstream slabs to Fourier space,
    # using the 'slab_global' or 'slab_updownstream' buffer.
    slab_upstream = slab_decompose(
        grid_upstream,
        None if use_upstream_as_global else 'slab_updownstream',
        prepare_fft=True,
    )
    fft(slab_upstream, 'forward')
    # Ensure nullified Nyquist planes
    nullify_modes(slab_upstream, 'nyquist')
    # Perform deconvolution and interlacing on the upstream slabs
    # and assign/update the global slabs to/with the results.
    if use_upstream_as_global:
        # Perform in-place deconvolution and interlacing
        fourier_operate(slab_upstream, deconv_order, lattice)
        # The upstream slabs are really the global slabs
        slab_global = slab_upstream
    else:
        # Add upstream slabs to the global slabs,
        # performing deconvolution and interlacing on the way.
        # If the global slabs have yet to be initialised,  we set the
        # values directly, rather than adding to existing ones.
        operation = '+='
        if slab_global is None:
            operation = '='
            # Fetch slab to be used as the global one. Depending on
            # whether this is constructed without scaling or through
            # downscaling or upscaling of the υpstream slab, different
            # sets of elements needs to be nullified. We could of course
            # just always perform a complete nullification.
            if gridsize_global == gridsize_upstream:
                nullification = 'all'
            elif gridsize_global < gridsize_upstream:
                # Global slabs constructed through downscaling of the
                # upstream slabs. All elements of slab_global will be
                # set except the Nyquist planes, which then have
                # to be nullified.
                nullification = 'nyquist'
            else:
                # Global slabs constructed through upscaling of the
                # upstream slabs. Only elements within a cube centred
                # at the origin and of size gridsize_upstream will be
                # set. We then need to nullify all elements beyond
                # this cube.
                nullification = f'beyond cube of |k| < {gridsize_upstream//2}'
            slab_global = get_fftw_slab(gridsize_global, nullify=nullification)
        copy_modes(slab_upstream, slab_global, deconv_order, lattice, operation)
    return slab_global

# Function for grouping components according to
# associated grid sizes and their representation.
def group_components(components, gridsizes, gridsizes_order=(), split_representations=True):
    """Given a list of components and a list of corresponding grid
    sizes, this function returns a dict of components, grouped by their
    grid size and representation, in the format
        {
            <gridsize 0>: {
                'particles': [<component 0>],
            },
            <gridsize 2>: {
                'particles': [<component 4>, <component 2>],
                'fluid'    : [<component 1>],
            },
            <gridsize 1>: {
                'fluid': [<component 5>, <component 3>],
            },
        }
    A specific ordering of the grid sizes can be enforced by specifying
    gridsizes_order. The above example can be obtained by one of
        gridsizes_order = [..., <gridsize 2>, <gridsize 1>]
        gridsizes_order = [<gridsize 0>, ..., <gridsize 1>]
        gridsizes_order = [<gridsize 0>, <gridsize 2>, ...]
    The ellipsis (...) signals any grid sizes not explicitly listed.
    If the passed gridsizes_order does not contain an ellipsis, one will
    be placed at the end.
    """
    # Look up previous result in cache
    key = (tuple(components), tuple(gridsizes), tuple(gridsizes_order), split_representations)
    groups = component_groups_cache.get(key)
    if groups is not None:
        return groups
    # Perform the grouping
    if split_representations:
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        for component, gridsize in zip(components, gridsizes):
            groups[gridsize][component.representation].append(component)
    else:
        groups = collections.defaultdict(list)
        for component, gridsize in zip(components, gridsizes):
            groups[gridsize].append(component)
    # Create list of ordered grid sizes
    gridsizes_order = any2list(gridsizes_order)
    n_ellipses = gridsizes_order.count(...)
    if n_ellipses == 0:
        gridsizes_order += [...]
    elif n_ellipses > 1:
        abort(
            f'group_components() got gridsizes_order = {gridsizes_order}, '
            f'but this must contain at most one ellipsis (...)'
        )
    index = gridsizes_order.index(...)
    gridsizes_first = gridsizes_order[:index]
    gridsizes_last = gridsizes_order[index + 1:]
    gridsizes_ordered = gridsizes_first.copy()
    for gridsize in gridsizes:
        if gridsize in gridsizes_ordered or gridsize in gridsizes_last:
            continue
        gridsizes_ordered.append(gridsize)
    gridsizes_ordered += gridsizes_last
    # Enforce correct ordering of the groups based on their grid sizes
    groups = {
        gridsize: dict(groups[gridsize]) if split_representations else list(groups[gridsize])
        for gridsize in gridsizes_ordered
        if gridsize in gridsizes
    }
    # Cache and return result
    component_groups_cache[key] = groups
    return groups
# Cache used by the group_components() function
cython.declare(component_groups_cache=dict)
component_groups_cache = {}

# Function for resizing a real space domain grid
# or Fourier space slabs, i.e. change the grid size.
@cython.pheader(
    # Arguments
    grid_or_slab='double[:, :, ::1]',
    gridsize_new='Py_ssize_t',
    input_space=str,
    output_space=str,
    internal_slab_or_buffer_name=object,  # double[:, :, ::1] or int or str
    output_grid_or_buffer_name=object,    # double[:, :, ::1] or int or str
    output_slab_or_buffer_name=object,    # double[:, :, ::1] or int or str
    inplace='bint',
    apply_forward_fft_normalization='bint',
    do_ghost_communication='bint',
    # Locals
    grid='double[:, :, ::1]',
    grid_new='double[:, :, ::1]',
    gridsize='Py_ssize_t',
    nullification=str,
    slab='double[:, :, ::1]',
    slab_new='double[:, :, ::1]',
    returns='double[:, :, ::1]',
)
def resize_grid(
    grid_or_slab, gridsize_new, input_space,
    output_space='same', internal_slab_or_buffer_name=None,
    output_grid_or_buffer_name=None, output_slab_or_buffer_name=None,
    inplace=True, apply_forward_fft_normalization=True, do_ghost_communication=True,
):
    """Given a real space domain grid or Fourier space slabs (specified
    by input_space = 'real' or input_space = 'Fourier') and a new grid
    size (gridsize_new), the grid/slab data will be copied over to a new
    grid/slab with the new grid size. Whether to return a real space
    grid or Fourier space slabs is specified by output_space.
    If the output space is different from the input space, a new
    grid/slab is needed to store the resized result. The grid/slab to
    use for this is specified via output_grid_or_buffer_name (for real
    space output) and output_slab_or_buffer_name (for Fourier space
    output).
    When a resizing is to be performed (the original and new grid size
    are different) and the input is in real space, an additional
    internal slab is needed, as the resizing is carried out in Fourier
    space. The slab to use for this is specified as
    internal_slab_or_buffer_name.
    If the input and output space are the same and the original and new
    grid size are also the same, the passed grid/slab is returned back
    as is. No additional grid/slab will be used. Also, if the input
    space is Fourier, the output space is real and the original and new
    grid size are the same, the input slab will be transformed in-place.
    To guarantee that the passed data is left untouched and that a new
    grid/slab is returned, set inplace = False.
    """
    if internal_slab_or_buffer_name is None:
        internal_slab_or_buffer_name = 'slab_updownstream'
    if output_grid_or_buffer_name is None:
        output_grid_or_buffer_name = 'grid_global'
    if output_slab_or_buffer_name is None:
        output_slab_or_buffer_name = 'slab_global'
    # Handle the input and output space
    input_space = input_space.lower()
    if input_space == 'real':
        grid = grid_or_slab
        gridsize = (grid.shape[0] - ℤ[2*nghosts])*domain_subdivisions[0]
    elif input_space == 'fourier':
        slab = grid_or_slab
        gridsize = slab.shape[1]
    else:
        abort(f'resize_grid() got input_space = "{input_space}" ∉ {{"real", "Fourier"}}')
    output_space = output_space.lower()
    if output_space == 'same':
        output_space = input_space
    elif output_space not in ('real', 'fourier'):
        abort(f'resize_grid() got output_space = "{output_space}" ∉ {{"real", "Fourier"}}')
    # No resizing is required if the original and new grid size are the
    # same. If the input and output space are also the same, just return
    # the passed grid. Otherwise do the required FFT.
    if gridsize == gridsize_new:
        if input_space == output_space:
            if inplace:
                # In-place real to real or Fourier to Fourier
                return grid_or_slab
            elif input_space == 'real':
                # Out-of-place real to real
                if isinstance(output_grid_or_buffer_name, (int, np.integer, str)):
                    grid_new = get_buffer(get_gridshape_local(gridsize), output_grid_or_buffer_name)
                else:
                    grid_new = output_grid_or_buffer_name
                grid_new[...] = grid
                return grid_new
            else:  # input_space == 'fourier'
                # Out-of-place Fourier to Fourier
                if isinstance(output_slab_or_buffer_name, (int, np.integer, str)):
                    slab_new = get_fftw_slab(gridsize, output_slab_or_buffer_name)
                else:
                    slab_new = output_slab_or_buffer_name
                slab_new[...] = slab
                return slab_new
        elif output_space == 'fourier':
            # From real to Fourier
            slab_new = slab_decompose(grid, output_slab_or_buffer_name, prepare_fft=True)
            fft(slab_new, 'forward', apply_forward_fft_normalization)
            return slab_new
        else:  # output_space == 'real'
            # From Fourier to real
            if not inplace:
                if isinstance(internal_slab_or_buffer_name, (int, np.integer, str)):
                    slab_internal = get_fftw_slab(gridsize, internal_slab_or_buffer_name)
                else:
                    slab_internal = internal_slab_or_buffer_name
                slab_internal[...] = slab
                slab = slab_internal
            fft(slab, 'backward')
            grid_new = domain_decompose(
                slab, output_grid_or_buffer_name,
                do_ghost_communication=do_ghost_communication,
            )
            return grid_new
    # Resizing should take place. Go to Fourier space.
    if input_space == 'real':
        slab = slab_decompose(grid, internal_slab_or_buffer_name, prepare_fft=True)
        fft(slab, 'forward', apply_forward_fft_normalization)
    # Perform the resizing by copying the slab values into another
    # slab decomposed grid with the new grid size.
    if isinstance(output_slab_or_buffer_name, (int, np.integer, str)):
        slab_new = get_fftw_slab(gridsize_new, output_slab_or_buffer_name)
    else:
        slab_new = output_slab_or_buffer_name
    if gridsize_new < gridsize:
        # The grid is to be downscaled. All modes except the Nyquist
        # ones will be set and so these need to be explicitly nullified.
        nullification = 'nyquist'
    else:  # gridsize_new > gridsize
        # The grid is to be upscaled. New modes beyond the ones in the
        # original grid will not be set and so these need to be
        # explicitly nullified.
        nullification = f'beyond cube of |k| < {gridsize//2}'
    nullify_modes(slab_new, nullification)
    copy_modes(slab, slab_new, operation='=')
    # Return the slab as is or transform to real space and return the
    # resulting grid, depending on the output space.
    if output_space == 'real':
        fft(slab_new, 'backward')
        grid_new = domain_decompose(
            slab_new, output_grid_or_buffer_name,
            do_ghost_communication=do_ghost_communication,
        )
        return grid_new
    else:  # output_space == 'fourier'
        return slab_new

# Function for adding the Fourier modes of one slab to another
@cython.pheader(
    # Arguments
    slab_from='double[:, :, ::1]',
    slab_onto='double[:, :, ::1]',
    deconv_order='int',
    lattice='Lattice',
    operation=str,
    # Locals
    cosθ='double',
    factor='double',
    gridsize_from='Py_ssize_t',
    gridsize_large='Py_ssize_t',
    gridsize_onto='Py_ssize_t',
    gridsize_small='Py_ssize_t',
    half_subslab_recv='double[:, :, ::1]',
    i_from_bgn='Py_ssize_t',
    i_from_end='Py_ssize_t',
    i_onto_bgn='Py_ssize_t',
    i_onto_end='Py_ssize_t',
    i_ranges_from=list,
    i_ranges_large=list,
    i_ranges_onto=list,
    i_ranges_small=list,
    i_small_bgn='Py_ssize_t',
    i_small_end='Py_ssize_t',
    im='double',
    index='Py_ssize_t',
    index_from='Py_ssize_t',
    index_large='Py_ssize_t',
    index_onto='Py_ssize_t',
    index_recv='Py_ssize_t',
    index_small='Py_ssize_t',
    interlace_lattice='Lattice',
    j_from_bgn='Py_ssize_t',
    j_from_end='Py_ssize_t',
    j_global_large='Py_ssize_t',
    j_onto_bgn='Py_ssize_t',
    j_onto_end='Py_ssize_t',
    j_small_bgn='Py_ssize_t',
    j_small_end='Py_ssize_t',
    k_end='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    local='bint',
    nyquist_large='Py_ssize_t',
    nyquist_small='Py_ssize_t',
    rank_from='int',
    rank_onto='int',
    re='double',
    recv='bint',
    request=object,  # mpi4py.MPI.Request
    send='bint',
    sinθ='double',
    slab_from_ptr='double*',
    slab_from_size_i='Py_ssize_t',
    slab_from_size_j='Py_ssize_t',
    slab_from_size_k='Py_ssize_t',
    slab_large='double[:, :, ::1]',
    slab_large_size_i='Py_ssize_t',
    slab_large_size_j='Py_ssize_t',
    slab_large_size_k='Py_ssize_t',
    slab_onto_ptr='double*',
    slab_onto_size_i='Py_ssize_t',
    slab_onto_size_j='Py_ssize_t',
    slab_onto_size_k='Py_ssize_t',
    slab_read_ptr='double*',
    slab_small='double[:, :, ::1]',
    slab_small_size_i='Py_ssize_t',
    slab_small_size_j='Py_ssize_t',
    slab_small_size_k='Py_ssize_t',
    subslabs_dict_from=dict,
    subslabs_dict_large=dict,
    subslabs_dict_onto=dict,
    subslabs_dict_small=dict,
    subslabs_ranges_from=list,
    subslabs_ranges_onto=list,
    upscaling='bint',
    θ='double',
    θ_total='double',
    returns='double[:, :, ::1]',
)
def copy_modes(slab_from, slab_onto, deconv_order=0, lattice=None, operation='='):
    """The Fourier modes of slab_from will be copied into slab_onto.
    Deconvolution will be performed according to deconv_order.
    Set operation = '+=' to add the modes from slab_from to slab_onto,
    rather than overwriting them. By default no interlacing is
    performed. To carry out interlacing you need to call this function
    once for every sub-lattice with the (shifted) grid as slab_from,
    with all calls (except possibly the first) using operation = '+=' to
    not overwrite the results from the first call.
    """
    if lattice is None:
        lattice = Lattice()
    # Extract pointers
    slab_onto_ptr = cython.address(slab_onto[:, :, :])
    slab_from_ptr = cython.address(slab_from[:, :, :])
    # Extract grid sizes
    slab_onto_size_j, slab_onto_size_i, slab_onto_size_k = asarray(slab_onto).shape
    slab_from_size_j, slab_from_size_i, slab_from_size_k = asarray(slab_from).shape
    gridsize_from, gridsize_onto = slab_from_size_i, slab_onto_size_i
    # Consider the easy case of equal grid sizes separately
    if gridsize_from == gridsize_onto:
        # As slab_from and slab_onto have identical grid sizes,
        # we can copy the Fourier modes locally,
        # i.e. without any MPI communication.
        if deconv_order == 0 and lattice.shift == (0, 0, 0):
            # Deconvolution should not be performed and no phase shift
            # should be performed due to interlacing.
            # Combine directly.
            for index in range(slab_onto_size_j*slab_onto_size_i*slab_onto_size_k):
                # Set the factor due to interlacing
                factor = 1
                with unswitch:
                    if len(lattice) > 1:
                        factor = ℝ[1/len(lattice)]
                # Write to slab_onto
                with unswitch:
                    if 𝔹[operation == '=']:
                        slab_onto_ptr[index] = factor*slab_from_ptr[index]
                    else:  # operation == '+='
                        slab_onto_ptr[index] += factor*slab_from_ptr[index]
        else:
            # At least one of deconvolution or the non-trivial part of
            # interlacing is to be performed.
            # For this, we need proper iteration over ki, kj, kk.
            interlace_lattice = lattice
            for index, ki, kj, kk, factor, θ in fourier_loop(
                gridsize_onto,
                deconv_order=deconv_order, interlace_lattice=interlace_lattice,
            ):
                # Extract real and imag part of this
                # Fourier mode of slab_from.
                re = slab_from_ptr[index    ]
                im = slab_from_ptr[index + 1]
                # Rotate the complex phase due to interlacing
                with unswitch:
                    if interlace_lattice.shift != (0, 0, 0):
                        cosθ = cos(θ)
                        sinθ = sin(θ)
                        re, im = (
                            re*cosθ - im*sinθ,
                            re*sinθ + im*cosθ,
                        )
                # Apply factor from deconvolution and interlacing
                re *= factor
                im *= factor
                # Write to slab_onto
                with unswitch:
                    if 𝔹[operation == '=']:
                        slab_onto_ptr[index    ] = re
                        slab_onto_ptr[index + 1] = im
                    else:  # operation == '+='
                        slab_onto_ptr[index    ] += re
                        slab_onto_ptr[index + 1] += im
        # Easy case of equal grid sizes complete
        return slab_onto
    # The two grid sizes are not equal, and so the grid is to be
    # scaled up or down. Assign some variables according to
    # which of the two scenarios we are in.
    upscaling = (gridsize_from < gridsize_onto)
    if upscaling:
        slab_small, slab_large = slab_from, slab_onto
    else:  # downscaling
        slab_small, slab_large = slab_onto, slab_from
    slab_small_size_j, slab_small_size_i, slab_small_size_k = asarray(slab_small).shape
    slab_large_size_j, slab_large_size_i, slab_large_size_k = asarray(slab_large).shape
    gridsize_small, gridsize_large = slab_small_size_i, slab_large_size_i
    nyquist_small, nyquist_large = gridsize_small//2, gridsize_large//2
    # As slab_from and slab_onto have different grid sizes, a given
    # Fourier mode situated within slab_from on process p may be
    # situated outside slab_onto on process p, but instead within
    # slab_onto on some other process q. Generally then, different grid
    # sizes calls for interprocess communication. To do this
    # communication, we subdivide both slab_from and slab_onto into
    # subslabs, i.e. thinner (in the y direction only) slabs which
    # together constitute a full slab. For this subslab decomposition to
    # be of any use, we require that the Fourier modes within a given
    # subslab of slab_small on p is fully contained within just a single
    # subslab of slab_large on q (which may or may not equal p), and in
    # fact that these two corresponding subslabs span the same kj. Some
    # freedom remains regarding the subslab decomposition, but the
    # optimal choice is that which minimizes the number of subslabs.
    # Find this subslab decomposition now (does not mutate any data).
    subslabs_dict_small, subslabs_dict_large = get_subslabs(gridsize_small, gridsize_large)
    if upscaling:
        subslabs_dict_from, subslabs_dict_onto = subslabs_dict_small, subslabs_dict_large
    else:  # downscaling
        subslabs_dict_from, subslabs_dict_onto = subslabs_dict_large, subslabs_dict_small
    # While the j range is set by the subslab in question, they all span
    # the entire slab in the i and k dimension, but we are only
    # interested in the region corresponding to slab_small, as Fourier
    # modes beyond it cannot be copied between the two slabs. Set i and
    # k ranges. Note that each subslab consists of two i ranges, i.e.
    # two half subslabs. This is needed as these two halves are not
    # contiguous (due to the skipped larger Fourier modes in the large
    # slab and the skipped Nyquist modes in the small slab). Also note
    # that we implicitly have k_bgn = 0.
    i_ranges_small = [(0, nyquist_small), (gridsize_small - nyquist_small + 1, gridsize_small)]
    i_ranges_large = [(0, nyquist_small), (gridsize_large - nyquist_small + 1, gridsize_large)]
    if upscaling:
        i_ranges_from, i_ranges_onto = i_ranges_small, i_ranges_large
    else:  # downscaling
        i_ranges_from, i_ranges_onto = i_ranges_large, i_ranges_small
    k_end = slab_small_size_k - 2
    # Loop over processes with which to communicate subslabs
    for (
        (rank_onto, subslabs_ranges_from),
        (rank_from, subslabs_ranges_onto),
    ) in zip(
        subslabs_dict_from.items(),
        subslabs_dict_onto.items(),
    ):
        # Flag specifying whether this is a purely local subslab,
        # meaning that it resides entirely on the local process in
        # both slab_from and slab_onto.
        local = (rank_onto == rank)
        # Loop over subslabs
        for (
            (j_from_bgn, j_from_end),
            (j_onto_bgn, j_onto_end),
        ) in zip(
            subslabs_ranges_from,
            subslabs_ranges_onto,
        ):
            # Flags specifying whether we should
            # send/receive the subslab.
            send = (not local and rank_onto >= 0 and j_from_bgn != -1)
            recv = (not local and rank_from >= 0 and j_onto_bgn != -1)
            # Loop over half subslabs (positive and negative ki)
            for (
                (i_from_bgn, i_from_end),
                (i_onto_bgn, i_onto_end),
            ) in zip(
                i_ranges_from,
                i_ranges_onto,
            ):
                # Send half subslab
                if send:
                    request = smart_mpi(
                        slab_from[j_from_bgn:j_from_end, i_from_bgn:i_from_end, :k_end],
                        dest=rank_onto,
                        mpifun='Isend',
                    )
                # Receive half subslab
                if recv:
                    half_subslab_recv = get_buffer(
                        (j_onto_end - j_onto_bgn, i_onto_end - i_onto_bgn, k_end),
                        'subslab',
                    )
                    smart_mpi(
                        half_subslab_recv,
                        source=rank_from,
                        mpifun='Recv',
                    )
                # Wait for the non-blocking send to complete
                if send:
                    request.wait()
                # Done with subslab communication. Only carry on if the
                # half subslab should be copied into the local slab_onto
                # (i.e. if we are not at a local subslab which should
                # only be send).
                if not (local or recv):
                    continue
                # To copy the (local or received) Fourier modes over to
                # slab_onto, we now need to loop over this slab, or
                # rather some (half) subslab of it. However, we only
                # want to set the modes within the small grid (even in
                # the case of slab_onto = slab_large, as modes beyond it
                # cannot be set from slab_from = slab_small), and so we
                # need to loop only over the j range of the
                # small subslab. Get this j range from
                # j_onto_bgn and j_onto_end.
                if upscaling:
                    # When upscaling we have slab_onto = slab_large
                    # and so the known j range of the onto slab is for
                    # the subslab of the large slab, not the small.
                    # Convert the bgn index.
                    j_global_large = ℤ[slab_large_size_j*rank] + j_onto_bgn
                    kj = j_global_large - (-(j_global_large >= nyquist_large) & gridsize_large)
                    j_small_bgn = kj + (-(kj < 0) & gridsize_small) - ℤ[slab_small_size_j*rank]
                    # Convert the end index, remembering to temporarily
                    # treat it as inclusive rather than exclusive.
                    j_global_large = ℤ[slab_large_size_j*rank] + j_onto_end - 1
                    kj = j_global_large - (-(j_global_large >= nyquist_large) & gridsize_large)
                    j_small_end = kj + (-(kj < 0) & gridsize_small) - ℤ[slab_small_size_j*rank] + 1
                else:  # downscaling
                    # When downscaling we have slab_onto = slab_small
                    # and so the known j range of the onto slab is the
                    # same as the j range of the small slab.
                    j_small_bgn, j_small_end = j_onto_bgn, j_onto_end
                # We further need the i range of the half subslab of the
                # small slab. Here we can always just pick the correct
                # pre-computed such range.
                if upscaling:
                    i_small_bgn, i_small_end = i_from_bgn, i_from_end
                else:  # downscaling
                    i_small_bgn, i_small_end = i_onto_bgn, i_onto_end
                # Pointer to the slab from which
                # the data should be copied.
                if local:
                    slab_read_ptr = slab_from_ptr
                else:
                    slab_read_ptr = cython.address(half_subslab_recv[:, :, :])
                # Loop over the Fourier modes of the half small subslab,
                # evaluating deconvolution and interlacing as if they
                # belong to a grid of size gridsize_from
                # (which they do).
                interlace_lattice = lattice
                for index_small, ki, kj, kk, factor, θ in fourier_loop(
                    gridsize_small, gridsize_from,
                    i_small_bgn, i_small_end,
                    j_small_bgn, j_small_end,
                    deconv_order=deconv_order, interlace_lattice=interlace_lattice,
                ):
                    # Corresponding index into the large subslab
                    index_large = (
                        ℤ[
                            (
                                ℤ[
                                    (
                                        kj + (-(kj < 0) & gridsize_large) - ℤ[slab_large_size_j*rank]  # j_large
                                    )*slab_large_size_i
                                ]
                                + ki + (-(ki < 0) & gridsize_large)  # i_large
                            )*slab_large_size_k
                        ]
                        + 2*kk  # k_large
                    )
                    # Corresponding index into the received half subslab
                    index_recv = (
                        ℤ[
                            (
                                ℤ[
                                    (
                                        kj + (-(kj < 0) & gridsize_small) - ℤ[slab_small_size_j*rank + j_small_bgn]  # j_recv
                                    )*ℤ[i_small_end - i_small_bgn]
                                ]
                                + ki + (-(ki < 0) & gridsize_small) - i_small_bgn  # i_recv
                            )*k_end
                        ]
                        + 2*kk  # k_recv
                    )
                    # Set indices into the read and write slab
                    with unswitch(5):
                        if 𝔹[local and upscaling]:
                            # Local upscaling
                            index_from = index_small
                            index_onto = index_large
                        elif 𝔹[local and not upscaling]:
                            # Local downscaling
                            index_from = index_large
                            index_onto = index_small
                        elif upscaling:
                            # Non-local upscaling
                            index_from = index_recv
                            index_onto = index_large
                        else:
                            # Non-local downscaling
                            index_from = index_recv
                            index_onto = index_small
                    # Extract real and imag part of this Fourier mode
                    re = slab_read_ptr[index_from    ]
                    im = slab_read_ptr[index_from + 1]
                    # The total complex phase shift due to both change
                    # of grid size and interlacing.
                    θ_total = ℝ[π/gridsize_onto - π/gridsize_from]*ℤ[ℤ[ki + kj] + kk]
                    with unswitch(5):
                        if interlace_lattice.shift != (0, 0, 0):
                            θ_total += θ
                    # Apply factor and phase shift from deconvolution,
                    # interlacing and change of grid size.
                    cosθ = cos(θ_total)
                    sinθ = sin(θ_total)
                    re, im = (
                        factor*(re*cosθ - im*sinθ),
                        factor*(re*sinθ + im*cosθ),
                    )
                    # Write to slab_onto
                    with unswitch(5):
                        if 𝔹[operation == '=']:
                            slab_onto_ptr[index_onto    ] = re
                            slab_onto_ptr[index_onto + 1] = im
                        else:  # operation == '+=':
                            slab_onto_ptr[index_onto    ] += re
                            slab_onto_ptr[index_onto + 1] += im
    return slab_onto

# Function for dividing up a pair of different sized slabs into
# overlapping subslabs. The resulting subslab layout is returned,
# while no slab is mutated.
def get_subslabs(gridsize_small, gridsize_large):
    """Given two grid sizes, one smaller than the other, this function
    returns two corresponding dicts mapping process ranks to lists of
    overlapping subslab ranges between the local process and the process
    rank in question.
    An example will illustrate the format. Name the two returned dicts
    subslabs_dict_j_small and subslabs_dict_j_large and say we see
    things from the perspective of process 1 while running with 4
    processes and having gridsize_small = 12, gridsize_large = 20,
    the result would then be
        subslabs_dict_j_small = {1: [(2, 3)], 0: [(0, 2)]}  # process 1
        subslabs_dict_j_large = {1: [(0, 1)]}               # process 1
    This means that given these two grid sizes, the local (to process 1)
    small slab is divided into two subslabs slab_small[j=0:2, i=:, k=:]
    and slab_small[j=2:3, i=:, k=:], whereas only a single local subslab
    slab_large[j=0:1, i=:, k=:] results from the local large slab. Note
    that the union of all subslabs given may not result in the complete
    slab, as some subslabs may be irrelevant (they contain no overlap
    between the small and large slabs, or contain only Nyquist points
    which are disregarded). The small local subslab
    slab_small[j=0:2, i=:, k=:] overlaps with a large subslab on process
    0 (note that by construction two small or two large subslabs may
    never overlap). The details of this non-local overlapping large
    subslab is not known to process 1, but is contained (among other
    things) in the corresponding result
        subslabs_dict_j_large = {0: [(0, 3)], 1: [(3, 5)]}  # process 0
    on process 0. We see that the other small subslab
    slab_small[j=2:3, i=:, k=:] local to process 1 overlaps with a large
    subslab on process 1 itself. Indeed, looking at
    subslabs_dict_j_large on process 1 we see this large subslab.
    """
    # Look up previous result in cache
    key = (gridsize_small, gridsize_large)
    subslabs_j_both = subslabs_cache.get(key)
    if subslabs_j_both is not None:
        return subslabs_j_both
    # Sanity check on grid sizes
    if gridsize_small > gridsize_large:
        masterwarn(
            f'get_subslabs() called with gridsize_small = {gridsize_small} '
            f'> gridsize_large = {gridsize_large}. These arguments have '
            f'been switched around.'
        )
        gridsize_smal, gridsize_large = gridsize_large, gridsize_small
        return get_subslabs(gridsize_small, gridsize_large)
    for varname, gridsize in {
        'gridsize_small': gridsize_small,
        'gridsize_large': gridsize_large,
    }.items():
        if gridsize%2:
            abort(
                f'get_subslabs() got {varname} = {gridsize}, but this must be even'
            )
        if gridsize%nprocs:
            abort(
                f'get_subslabs() got {varname} = {gridsize} '
                f'which is not divisible by {nprocs} processes'
            )
    # Collect all local kj
    def get_kj_sets(gridsize):
        nyquist = gridsize//2
        slab_size_j = gridsize//nprocs
        kj_sets = []
        for rank_other in range(nprocs):
            kj_set = set()
            for j in range(slab_size_j):
                j_global = slab_size_j*rank_other + j
                kj = j_global - (-(j_global >= nyquist) & gridsize)
                # Ignore Nyquist planes
                if kj == -nyquist:
                    continue
                kj_set.add(kj)
            kj_sets.append(kj_set)
        return kj_sets
    kj_sets_small = get_kj_sets(gridsize_small)
    kj_sets_large = get_kj_sets(gridsize_large)
    # Find groups of consecutive local kj, i.e. subslabs
    def construct_subslabs_kj_dict(kj_set, kj_sets, sign):
        subslabs_dict_kj = {}
        for ℓ in range(nprocs):
            rank_other = (rank + sign*ℓ)%nprocs  # determines rank communication order
            other_kj_set = kj_sets[rank_other]
            overlap = sorted(kj_set & other_kj_set)
            if not overlap:
                continue
            subslabs_kj_ranges = []
            subslabs_dict_kj[rank_other] = subslabs_kj_ranges
            for _, groups in itertools.groupby(enumerate(overlap), lambda el: el[1] - el[0]):
                subrange = [group[1] for group in groups]
                kj_bgn, kj_end = subrange[0], subrange[-1]
                # A subslab may not contain both
                # negative and non-negative kj.
                kj_bgn, kj_end = np.min((kj_bgn, kj_end)), np.max((kj_bgn, kj_end))
                if kj_bgn < 0 and kj_end >= 0:
                    subslabs_kj_ranges.append((kj_bgn, -1))
                    subslabs_kj_ranges.append((0, kj_end))
                else:
                    subslabs_kj_ranges.append((kj_bgn, kj_end))
        return subslabs_dict_kj
    subslabs_dict_kj_small = construct_subslabs_kj_dict(kj_sets_small[rank], kj_sets_large, +1)
    subslabs_dict_kj_large = construct_subslabs_kj_dict(kj_sets_large[rank], kj_sets_small, -1)
    # Convert from kj to j
    def convert_subslabs_kj_to_j(subslabs_kj_dict, gridsize):
        slab_size_j = gridsize//nprocs
        def kj_to_j(kj):
            # NumPy Boolean scalars do not support operator '-'
            kj = int(kj)
            return kj + (-(kj < 0) & gridsize) - slab_size_j*rank
        subslabs_j_dict = {
            rank_other: [
                (kj_to_j(kj_bgn), kj_to_j(kj_end) + 1)  # use exclusive end point
                for kj_bgn, kj_end in subslabs_kj_ranges
            ]
            for rank_other, subslabs_kj_ranges in subslabs_kj_dict.items()
        }
        return subslabs_j_dict
    subslabs_dict_j_small = convert_subslabs_kj_to_j(subslabs_dict_kj_small, gridsize_small)
    subslabs_dict_j_large = convert_subslabs_kj_to_j(subslabs_dict_kj_large, gridsize_large)
    # Pad the small and large subslab dicts such that
    # they contain the same number of items, with pairs of j ranges
    # which also contain the same number of elements.
    def pad_dict(subslabs_j, length):
        fake_rank = -1
        while len(subslabs_j) < length:
            subslabs_j[fake_rank] = []
            fake_rank -= 1
    pad_dict(subslabs_dict_j_small, len(subslabs_dict_j_large))
    pad_dict(subslabs_dict_j_large, len(subslabs_dict_j_small))
    def pad_ranges(ranges, length):
        while len(ranges) < length:
            ranges.append((-1, -1))
    for subslabs_j_ranges_small, subslabs_j_ranges_large in zip(
        subslabs_dict_j_small.values(), subslabs_dict_j_large.values()
    ):
        length = np.max([len(subslabs_j_ranges_small), len(subslabs_j_ranges_large)])
        pad_ranges(subslabs_j_ranges_small, length)
        pad_ranges(subslabs_j_ranges_large, length)
    # Cache results and return
    subslabs_dict_j_both = (subslabs_dict_j_small, subslabs_dict_j_large)
    subslabs_cache[key] = subslabs_dict_j_both
    return subslabs_dict_j_both
# Cache used by the get_subslabs() function
cython.declare(subslabs_cache=dict)
subslabs_cache = {}

# Function for interpolating a certain quantity from a particle
# component onto a supplied domain grid.
@cython.pheader(
    # Arguments
    component='Component',
    gridsize='Py_ssize_t',
    grid='double[:, :, ::1]',
    quantity=str,
    order='int',
    ᔑdt=dict,
    lattice='Lattice',
    factor='double',
    do_ghost_communication='bint',
    # Locals
    a='double',
    cellsize='double',
    constant_contribution='bint',
    contribution='double',
    contribution_factor='double',
    contribution_mv='double[::1]',
    contribution_ptr='double*',
    contribution_weighted='double',
    dim='int',
    grid_ptr='double*',
    index='Py_ssize_t',
    indexˣ='Py_ssize_t',
    offset_x='double',
    offset_y='double',
    offset_z='double',
    posxˣ='double*',
    posyˣ='double*',
    poszˣ='double*',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    w_eff='double',
    x='double',
    y='double',
    z='double',
    returns='void',
)
def interpolate_particles(
    component, gridsize, grid, quantity, order,
    ᔑdt=None, lattice=None, factor=1, do_ghost_communication=True,
):
    """The given quantity of the particle component will be added to
    current content of the local grid with global grid size given by
    gridsize. For info about the quantity argument, see the
    interpolate_upstream() function.
    Time dependent factors in the quantity are evaluated at the current
    time as defined by the universals struct. If ᔑdt is passed as a
    dict containing time step integrals, these factors will be
    integrated over the time step.
    If a lattice is supplied, the particle positions will be shifted by
    the negative lattice shift before interpolated to the grid.
    Setting the factor != 1 scales the interpolated quantity.
    The supplied grid should contain ghost layers, as the interpolation
    will populate these. To communicate and add the resulting values in
    the ghost cells to their physical cells, set do_ghost_communication
    to True. Note that even with do_ghost_communication set to True, the
    ghost cells will not end up with copies of the boundary values.
    """
    if not (1 <= order <= 4):
        abort(
            f'interpolate_particles() called with order = {order} '
            f'∉ {{1 (NGP), 2 (CIC), 3 (TSC), 4 (PCS)}}'
        )
    # Always use the current time
    a = universals.a
    w_eff = component.w_eff(a=a)
    # Determine the contribution of each particle based on the quantity
    contribution = 1
    if quantity == 'ρ':
        constant_contribution = True
        if ᔑdt:
            contribution = ᔑdt['a**(-3*(1+w_eff))', component.name]/ᔑdt['1']
        else:
            contribution = a**(-3*(1 + w_eff))
        contribution *= component.mass
    elif quantity == 'a²ρ':
        constant_contribution = True
        if ᔑdt:
            contribution = ᔑdt['a**(-3*w_eff-1)', component.name]/ᔑdt['1']
        else:
            contribution = a**(-3*w_eff - 1)
        contribution *= component.mass
    elif quantity == 'ϱ':
        constant_contribution = True
        contribution = component.mass
    elif quantity in {'Jx', 'Jy', 'Jz'}:
        constant_contribution = False
        # Extract pointer to momentum data,
        # indexed as contribution_ptr[indexˣ] regardless of dim.
        dim = 'xyz'.index(quantity[1])
        contribution_mv = component.mom_mv[dim:]
        contribution_ptr = cython.address(contribution_mv[:])
    else:
        abort(
            f'interpolate_particles() called with '
            f'quantity = "{quantity}" ∉ {{"ρ", "a²ρ", "ϱ", "Jx", "Jy", "Jz"}}'
        )
    contribution_factor = factor*(gridsize/boxsize)**3
    contribution *= contribution_factor
    # Offsets and scalings needed for the interpolation
    if lattice is None:
        lattice = Lattice()
    cellsize = boxsize/gridsize
    offset_x = (
        + domain_bgn_x
        - (1 + machine_ϵ)*(nghosts - 0.5*cell_centered - lattice.shift[0])*cellsize
    )
    offset_y = (
        + domain_bgn_y
        - (1 + machine_ϵ)*(nghosts - 0.5*cell_centered - lattice.shift[1])*cellsize
    )
    offset_z = (
        + domain_bgn_z
        - (1 + machine_ϵ)*(nghosts - 0.5*cell_centered - lattice.shift[2])*cellsize
    )
    # Interpolate each particle
    posxˣ = component.posxˣ
    posyˣ = component.posyˣ
    poszˣ = component.poszˣ
    size_j, size_k = grid.shape[1], grid.shape[2]
    grid_ptr = cython.address(grid[:, :, :])
    for indexˣ in range(0, 3*component.N_local, 3):
        # Get the total contribution from this particle
        with unswitch:
            if not constant_contribution:
                contribution = contribution_factor*contribution_ptr[indexˣ]
        # Get, translate and scale the coordinates so that
        #   nghosts - ½ < r < shape[r] - nghosts - ½ for r ∈ {x, y, z}
        # (in the case of no shifting).
        x = (posxˣ[indexˣ] - offset_x)*ℝ[(1/cellsize)*(1 - machine_ϵ)]
        y = (posyˣ[indexˣ] - offset_y)*ℝ[(1/cellsize)*(1 - machine_ϵ)]
        z = (poszˣ[indexˣ] - offset_z)*ℝ[(1/cellsize)*(1 - machine_ϵ)]
        # Carry out the interpolation according to the order
        with unswitch:
            if order == 1:  # NGP interpolation
                for index, contribution_weighted in particle_interpolation_loop_NGP(
                    x, y, z, size_j, size_k,
                    contribution, apply_factor=True,
                ):
                    grid_ptr[index] += contribution_weighted
            elif order == 2:  # CIC interpolation
                for index, contribution_weighted in particle_interpolation_loop_CIC(
                    x, y, z, size_j, size_k,
                    contribution, apply_factor=True,
                ):
                    grid_ptr[index] += contribution_weighted
            elif order == 3:  # TSC interpolation
                for index, contribution_weighted in particle_interpolation_loop_TSC(
                    x, y, z, size_j, size_k,
                    contribution, apply_factor=True,
                ):
                    grid_ptr[index] += contribution_weighted
            else:  # order == 4  # PCS interpolation
                for index, contribution_weighted in particle_interpolation_loop_PCS(
                    x, y, z, size_j, size_k,
                    contribution, apply_factor=True,
                ):
                    grid_ptr[index] += contribution_weighted
    # All particles interpolated. Some may have gotten interpolated
    # partly onto ghost points, which then need to be communicated.
    if do_ghost_communication:
        communicate_ghosts(grid, '+=')

# Function for adding together a certain quantity
# from several fluid components.
@cython.header(
    # Arguments
    components=list,
    grid='double[:, :, ::1]',
    quantity=str,
    ᔑdt=dict,
    factor='double',
    # Locals
    component='Component',
)
def combine_fluids(components, grid, quantity, ᔑdt, factor=1):
    """The given quantity of the components (all of which are assumed to
    be fluid components) will be added together into the grid, with the
    current contents of the grid overwritten.
    """
    if not components:
        return grid
    add_fluid_to_grid(components[0], grid, quantity, ᔑdt, factor, operation='=')
    for component in components[1:]:
        add_fluid_to_grid(component, grid, quantity, ᔑdt, factor, operation='+=')
    return grid

# Function for adding a certain quantity from a fluid
# component onto a supplied domain grid.
@cython.pheader(
    # Arguments
    component='Component',
    grid='double[:, :, ::1]',
    quantity=str,
    ᔑdt=dict,
    factor='double',
    operation=str,
    # Locals
    a='double',
    dim='int',
    fluidscalar='FluidScalar',
    grid_fluid='double[:, :, ::1]',
    grid_fluid_ptr='double*',
    grid_ptr='double*',
    gridshape=tuple,
    gridshape_fluid=tuple,
    index='Py_ssize_t',
    w_eff='double',
    returns='double[:, :, ::1]',
)
def add_fluid_to_grid(component, grid, quantity, ᔑdt, factor=1, operation='+='):
    """The given quantity of the fluid component will be added to
    current content of the local grid with global grid size given by
    gridsize. For info about the quantity argument, see the
    interpolate_upstream() function.
    Time dependent factors in the quantity are evaluated at the current
    time as defined by the universals struct. If ᔑdt is passed as a
    dict containing time step integrals, these factors will be
    integrated over the time step.
    Setting the factor != 1 scales the interpolated quantity.
    The supplied grid should contain ghost layers. If these are properly
    populated when this function is called, they will remain so after
    this function has finished.
    """
    # The component must have identical grid size to the passed grid
    gridshape = asarray(grid).shape
    gridshape_fluid = get_gridshape_local(component.gridsize)
    if gridshape != gridshape_fluid:
        abort(
            f'add_fluid_to_grid() got component with local grid shape {gridshape_fluid} '
            f'(global grid size {component.gridsize}) and non-matching grid '
            f'of local shape {gridshape}.'
        )
    # Always use the current time
    a = universals.a
    w_eff = component.w_eff(a=a)
    # Get the fluid grid and determine the contribution factor of each
    # fluid cell, based on the quantity.
    if quantity == 'ρ':
        fluidscalar = component.ϱ
        if ᔑdt:
            factor *= ᔑdt['a**(-3*(1+w_eff))', component.name]/ᔑdt['1']
        else:
            factor *= a**(-3*(1 + w_eff))
    elif quantity == 'a²ρ':
        fluidscalar = component.ϱ
        if ᔑdt:
            factor *= ᔑdt['a**(-3*w_eff-1)', component.name]/ᔑdt['1']
        else:
            factor *= a**(-3*w_eff - 1)
    elif quantity == 'ϱ':
        fluidscalar = component.ϱ
    elif quantity in {'Jx', 'Jy', 'Jz'}:
        dim = 'xyz'.index(quantity[1])
        fluidscalar = component.J[dim]
    else:
        abort(
            f'add_fluid_to_grid() called with '
            f'quantity = "{quantity}" ∉ {{"ρ", "a²ρ", "ϱ", "Jx", "Jy", "Jz"}}'
        )
    grid_fluid = fluidscalar.grid_mv
    # Copy the values of the fluid grid onto the buffer grid
    grid_fluid_ptr = cython.address(grid_fluid[:, :, :])
    grid_ptr = cython.address(grid[:, :, :])
    for index in range(grid.shape[0]*grid.shape[1]*grid.shape[2]):
        with unswitch:
            if operation == '=':
                with unswitch:
                    if factor == 1:
                        grid_ptr[index] = grid_fluid_ptr[index]
                    else:
                        grid_ptr[index] = grid_fluid_ptr[index]*factor
            else:  # operation == '+=''
                with unswitch:
                    if factor == 1:
                        grid_ptr[index] += grid_fluid_ptr[index]
                    else:
                        grid_ptr[index] += grid_fluid_ptr[index]*factor
    return grid

# Function for converting particles of a particle component to fluid
# grids, effectively changing the representation of the component.
@cython.header(
    # Arguments
    component='Component',
    order='int',
    interlace=str,
    # Locals
    J_dim='FluidScalar',
    N_vacuum='Py_ssize_t',
    N_vacuum_originally='Py_ssize_t',
    dim='int',
    do_interlacing='bint',
    fft_factor='double',
    fields=dict,
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    lattice='Lattice',
    mv='double[:, :, ::1]',
    original_representation=str,
    quantity=str,
    shape=tuple,
    slab='double[:, :, ::1]',
    vacuum_sweep='Py_ssize_t',
    Δϱ_each='double',
    ϱ='double[:, :, ::1]',
    returns='Py_ssize_t',
)
def convert_particles_to_fluid(component, order, interlace='sc'):
    """This function interpolates particle positions and momenta onto
    fluid grids, effectively converting from a 'particles'
    representation to a 'fluid' representation. The mass attribute of
    the passed component should be the particle mass, not the average
    fluid element mass. The value of the representation attribute does
    not matter and will not be altered. The size of the fluid grids are
    determined by component.gridsize. To save memory, the particle data
    will be freed (resized to a minimum size) during the process.
    """
    # Backup of original representation
    original_representation = component.representation
    # Instantiate fluid grids spanning the local domains.
    # The newly allocated grids will be nullified.
    component.representation = 'fluid'
    shape = tuple([component.gridsize//ds for ds in domain_subdivisions])
    if any([component.gridsize != domain_subdivisions[dim]*shape[dim] for dim in range(3)]):
        abort(
            f'The grid size of {component.name} is {component.gridsize} '
            f'which cannot be equally shared among {nprocs} processes'
        )
    component.resize(shape)
    # Do the particle → fluid interpolation
    gridsize = component.gridsize
    fields = {'ϱ': component.ϱ.grid_mv}
    fields |= {'J' + 'xyz'[dim]: component.J[dim].grid_mv for dim in range(3)}
    lattice = Lattice(interlace)
    do_interlacing = (len(lattice) > 1 or lattice.shift != (0, 0, 0))
    fft_factor = (float(gridsize)**(-3) if do_interlacing else 1)
    for quantity, mv in fields.items():
        if do_interlacing:
            slab = None
            for lattice in lattice:
                mv[...] = 0
                interpolate_particles(
                    component, gridsize, mv, quantity, order,
                    lattice=lattice, factor=fft_factor,
                )
                slab = add_upstream_to_global_slabs(
                    mv, slab, gridsize, gridsize, order, lattice,
                )
            fft(slab, 'backward')
            domain_decompose(slab, mv, do_ghost_communication=False)
        else:
            interpolate_particles(component, gridsize, mv, quantity, order)
        communicate_ghosts(mv, '=')
    # The interpolation may have left some cells empty. Count up the
    # number of such vacuum cells and add to each a density of
    # ρ_vacuum, while leaving the momentum at zero. This will increase
    # the total mass, which then has to be lowered again, which we do
    # by subtracting a constant amount from each cell. This subtraction
    # may itself produce vacuum cells, and so we need to repeat until
    # no vacuum is detected.
    # Note that this vacuum correction scheme does not work well if
    # interlacing has been used above, as this often produce cells that
    # are not just zero but (slightly) negative.
    ϱ = component.ϱ.grid_mv
    for vacuum_sweep in range(gridsize):
        # Count up and assign to vacuum cells
        N_vacuum = 0
        for         i in range(nghosts, ℤ[ϱ.shape[0] - nghosts]):
            for     j in range(nghosts, ℤ[ϱ.shape[1] - nghosts]):
                for k in range(nghosts, ℤ[ϱ.shape[2] - nghosts]):
                    if ϱ[i, j, k] < ρ_vacuum:
                        N_vacuum += 1
                        ϱ[i, j, k] += ρ_vacuum
        N_vacuum = allreduce(N_vacuum, op=MPI.SUM)
        # Remember the original number of vacuum cells
        if vacuum_sweep == 0:
            N_vacuum_originally = N_vacuum
        # We are done when no vacuum is left
        if N_vacuum == 0:
            break
        # Ensure mass conservation
        Δϱ_each = N_vacuum*ℝ[ρ_vacuum/gridsize**3]
        for         i in range(nghosts, ℤ[ϱ.shape[0] - nghosts]):
            for     j in range(nghosts, ℤ[ϱ.shape[1] - nghosts]):
                for k in range(nghosts, ℤ[ϱ.shape[2] - nghosts]):
                    ϱ[i, j, k] -= Δϱ_each
    else:
        # Failed to remove vacuum
        masterwarn(
            'The convert_particles_to_fluid() function was unable to '
            'get rid of vacuum cells in the fluid after interpolation'
        )
    # Populate ghost points of all fluid grids
    component.communicate_fluid_grids('=')
    # The particle data is no longer needed. Free it to save memory.
    component.representation = 'particles'
    component.resize(1)
    # Re-insert the original representation and return
    # the original number of vacuum cells.
    component.representation = original_representation
    return N_vacuum_originally

# Function for getting the shape of a local grid, which is part of a
# global, cubic grid with some gridsize.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    # Locals
    dim='int',
    gridshape_local=tuple,
    returns=tuple,
)
def get_gridshape_local(gridsize):
    # Cache lookup
    gridshape_local = gridshape_local_cache.get(gridsize)
    if gridshape_local is not None:
        return gridshape_local
    # The global grid will be cut into domains according to the
    # domain_subdivisions. The cut along each dimension has to leave
    # the local grids with integer gridsize.
    for dim in range(3):
        if gridsize%domain_subdivisions[dim] != 0:
            abort(
                f'A grid of global gridsize {gridsize} is to be distributed '
                f'across the processes, but {gridsize}×{gridsize}×{gridsize} '
                f'cannot be divided according to the domain decomposition '
                f'{domain_subdivisions[0]}×{domain_subdivisions[1]}×{domain_subdivisions[2]}.'
            )
    # We have nghosts ghost points on both sides of the local grid,
    # for all dimensions.
    gridshape_local = tuple([
        gridsize//domain_subdivisions[dim] + ℤ[2*nghosts] for dim in range(3)
    ])
    for dim in range(3):
        if gridshape_local[dim] < ℤ[4*nghosts]:
            abort(
                f'A grid of shape {gridshape_local} (or '
                f'{asarray(gridshape_local) - 2*nghosts} without ghosts) was encountered '
                f'in get_gridshape_local(), but all domain grids must have at least twice '
                f'as many grid points across each dimension as the number of ghost layers '
                f'nghosts = {nghosts}.'
            )
    # Store result in cache and return
    gridshape_local_cache[gridsize] = gridshape_local
    return gridshape_local
# Cache used by the get_local_local function
cython.declare(gridshape_local_cache=dict)
gridshape_local_cache = {}

# Function for getting the shape of a slab
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    # Locals
    shape=tuple,
    returns=tuple,
)
def get_slabshape_local(gridsize):
    shape = (
        (gridsize//nprocs),  # distributed dimension
        gridsize,
        # Explicit int cast necessary for some reason
        int(2*(gridsize//2 + 1)),  # padded dimension
    )
    return shape

# Function that compute a lot of information needed by the
# slab_decompose and domain_decompose functions.
@cython.header(
    # Arguments
    domain_grid='double[:, :, ::1]',
    slab='double[:, :, ::1]',
    # Locals
    N_domain2slabs_communications='Py_ssize_t',
    domain_bgn_i='Py_ssize_t',
    domain_bgn_j='Py_ssize_t',
    domain_bgn_k='Py_ssize_t',
    domain_end_i='Py_ssize_t',
    domain_end_j='Py_ssize_t',
    domain_end_k='Py_ssize_t',
    domain_grid_noghosts='double[:, :, :]',
    domain_grid_shape=tuple,
    domain_sendrecv_i_end='int[::1]',
    domain_sendrecv_i_start='int[::1]',
    domain_size_i='Py_ssize_t',
    domain_size_j='Py_ssize_t',
    domain_size_k='Py_ssize_t',
    domain2slabs_recvsend_ranks='int[::1]',
    index='Py_ssize_t',
    info=tuple,
    rank_recv='int',
    rank_send='int',
    recvtuple=tuple,
    sendtuple=tuple,
    slab_sendrecv_j_end='int[::1]',
    slab_sendrecv_j_start='int[::1]',
    slab_sendrecv_k_end='int[::1]',
    slab_sendrecv_k_start='int[::1]',
    slab_shape=tuple,
    slab_size_i='Py_ssize_t',
    slabs2domain_sendrecv_ranks='int[::1]',
    ℓ='Py_ssize_t',
    returns=tuple,
)
def prepare_decomposition(domain_grid, slab):
    # Simply look up and return the needed information
    # if previously computed.
    domain_grid_shape = asarray(domain_grid).shape
    slab_shape = asarray(slab).shape
    info = decomposition_info.get((domain_grid_shape, slab_shape))
    if info:
        return info
    # Memoryview of the domain grid without the ghost layers
    domain_grid_noghosts = domain_grid[
        nghosts:(domain_grid.shape[0] - nghosts),
        nghosts:(domain_grid.shape[1] - nghosts),
        nghosts:(domain_grid.shape[2] - nghosts),
    ]
    # The size (number of grid points) of the truly local part of the
    # domain grid, excluding ghost layers, for each dimension.
    domain_size_i = domain_grid_noghosts.shape[0]
    domain_size_j = domain_grid_noghosts.shape[1]
    domain_size_k = domain_grid_noghosts.shape[2]
    # The global start and end indices of the local domain
    # in the global grid.
    domain_bgn_i = domain_layout_local_indices[0]*domain_size_i
    domain_bgn_j = domain_layout_local_indices[1]*domain_size_j
    domain_bgn_k = domain_layout_local_indices[2]*domain_size_k
    domain_end_i = domain_bgn_i + domain_size_i
    domain_end_j = domain_bgn_j + domain_size_j
    domain_end_k = domain_bgn_k + domain_size_k
    # When in real space, the slabs are distributed over the first
    # dimension. Give the size of the slab in this dimension a name.
    slab_size_i = slab.shape[0]
    # Find local i-indices to send and to which process by
    # shifting a piece of the number line in order to match
    # the communication pattern used.
    domain_sendrecv_i_start = np.roll(
        asarray(
            [
                ℓ - domain_bgn_i
                for ℓ in range(domain_bgn_i, domain_end_i, slab_size_i)
            ],
            dtype=C2np['int'],
        ),
        -rank,
    )
    domain_sendrecv_i_end = np.roll(
        asarray(
            [
                ℓ - domain_bgn_i + slab_size_i
                for ℓ in range(domain_bgn_i, domain_end_i, slab_size_i)
            ],
            dtype=C2np['int'],
        ),
        -rank,
    )
    slabs2domain_sendrecv_ranks = np.roll(
        asarray(
            [
                ℓ//slab_size_i
                for ℓ in range(domain_bgn_i, domain_end_i, slab_size_i)
            ],
            dtype=C2np['int'],
        ),
        -rank,
    )
    # Communicate the start and end (j, k)-indices of the slab,
    # where parts of the local domains should be received into.
    slab_sendrecv_j_start       = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_k_start       = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_j_end         = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_k_end         = empty(nprocs, dtype=C2np['int'])
    domain2slabs_recvsend_ranks = empty(nprocs, dtype=C2np['int'])
    index = 0
    for ℓ in range(nprocs):
        # Process ranks to send/receive to/from
        rank_send = mod(rank + ℓ, nprocs)
        rank_recv = mod(rank - ℓ, nprocs)
        # Send the global y and z start and end indices of the domain
        # to be send, if anything should be send to process rank_send.
        # Otherwise send None.
        sendtuple = (
            (domain_bgn_j, domain_bgn_k, domain_end_j, domain_end_k)
            if rank_send in asarray(slabs2domain_sendrecv_ranks) else None
        )
        recvtuple = sendrecv(sendtuple, dest=rank_send, source=rank_recv)
        if recvtuple is not None:
            slab_sendrecv_j_start[index]       = recvtuple[0]
            slab_sendrecv_k_start[index]       = recvtuple[1]
            slab_sendrecv_j_end[index]         = recvtuple[2]
            slab_sendrecv_k_end[index]         = recvtuple[3]
            domain2slabs_recvsend_ranks[index] = rank_recv
            index += 1
    # Cut off the tails
    slab_sendrecv_j_start = slab_sendrecv_j_start[:index]
    slab_sendrecv_k_start = slab_sendrecv_k_start[:index]
    slab_sendrecv_j_end = slab_sendrecv_j_end[:index]
    slab_sendrecv_k_end = slab_sendrecv_k_end[:index]
    domain2slabs_recvsend_ranks = domain2slabs_recvsend_ranks[:index]
    # The maximum number of communications it takes to communicate
    # the domain grid to the slabs (or vice versa).
    N_domain2slabs_communications = np.max([slabs2domain_sendrecv_ranks.shape[0],
                                            domain2slabs_recvsend_ranks.shape[0]])
    # Store and return all the resultant information,
    # needed for communicating between the domain and the slab.
    info = (N_domain2slabs_communications,
            domain2slabs_recvsend_ranks,
            slabs2domain_sendrecv_ranks,
            domain_sendrecv_i_start,
            domain_sendrecv_i_end,
            slab_sendrecv_j_start,
            slab_sendrecv_j_end,
            slab_sendrecv_k_start,
            slab_sendrecv_k_end,
            )
    decomposition_info[domain_grid_shape, slab_shape] = info
    return info
# Cache storing results of the prepare_decomposition function.
# The keys have the format (domain_grid_shape, slab_shape).
cython.declare(decomposition_info=dict)
decomposition_info = {}

# Function for transferring data from slabs to domain grids
@cython.pheader(
    # Arguments
    slab='double[:, :, ::1]',
    grid_or_buffer_name=object,  # double[:, :, ::1], int or str
    do_ghost_communication='bint',
    do_ghost_nullification='bint',
    # Locals
    N_domain2slabs_communications='Py_ssize_t',
    buffer_name=object,  # int or str
    chunk_recv_bgn='Py_ssize_t',
    chunk_recv_end='Py_ssize_t',
    chunk_send_bgn='Py_ssize_t',
    chunk_send_end='Py_ssize_t',
    domain_sendrecv_i_end='int[::1]',
    domain_sendrecv_i_start='int[::1]',
    domain2slabs_recvsend_ranks='int[::1]',
    grid='double[:, :, ::1]',
    grid_noghosts='double[:, :, :]',
    gridsize='Py_ssize_t',
    i_chunk='Py_ssize_t',
    n_chunks='Py_ssize_t',
    rank_recv='int',
    rank_send='int',
    request=object,  # mpi4py.MPI.Request
    shape=tuple,
    should_recv='bint',
    should_send='bint',
    slab_sendrecv_j_end='int[::1]',
    slab_sendrecv_j_start='int[::1]',
    slab_sendrecv_k_end='int[::1]',
    slab_sendrecv_k_start='int[::1]',
    slabs2domain_sendrecv_ranks='int[::1]',
    thickness_chunk='Py_ssize_t',
    ℓ='Py_ssize_t',
    returns='double[:, :, ::1]',
)
def domain_decompose(
    slab,
    grid_or_buffer_name=0,
    do_ghost_communication=True,
    do_ghost_nullification=False,
):
    if slab is None:
        return None
    if slab.shape[0] > slab.shape[1]:
        masterwarn(
            'domain_decompose() was called with a slab that appears to be transposed, '
            'i.e. in Fourier space.'
        )
    # Determine the correct shape of the domain grid corresponding to
    # the passed slab.
    gridsize = slab.shape[1]
    shape = get_gridshape_local(gridsize)
    # If no domain grid is passed, fetch a buffer of the right shape
    if isinstance(grid_or_buffer_name, (int, np.integer, str)):
        buffer_name = grid_or_buffer_name
        grid = get_buffer(shape, buffer_name)
    else:
        grid = grid_or_buffer_name
        if asarray(grid).shape != shape:
            abort(
                f'domain_decompose(): The slab and domain grid have '
                f'incompatible shapes: {asarray(slab).shape}, {asarray(grid).shape}.'
            )
    grid_noghosts = grid[
        nghosts:(grid.shape[0] - nghosts),
        nghosts:(grid.shape[1] - nghosts),
        nghosts:(grid.shape[2] - nghosts),
    ]
    # Compute needed communication variables
    (
        N_domain2slabs_communications,
        domain2slabs_recvsend_ranks,
        slabs2domain_sendrecv_ranks,
        domain_sendrecv_i_start,
        domain_sendrecv_i_end,
        slab_sendrecv_j_start,
        slab_sendrecv_j_end,
        slab_sendrecv_k_start,
        slab_sendrecv_k_end,
    ) = prepare_decomposition(grid, slab)
    # Communicate the slabs to the domain grid in chunks
    n_chunks, thickness_chunk = get_slab_domain_decomposition_chunk_size(slab, grid_noghosts)
    for ℓ in range(N_domain2slabs_communications):
        should_send = (ℓ < ℤ[domain2slabs_recvsend_ranks.shape[0]])
        should_recv = (ℓ < ℤ[slabs2domain_sendrecv_ranks.shape[0]])
        if not should_send and not should_recv:
            continue
        if should_send:
            rank_send = domain2slabs_recvsend_ranks[ℓ]
        if should_recv:
            rank_recv = slabs2domain_sendrecv_ranks[ℓ]
        # Communicate the chunks
        for i_chunk in range(n_chunks):
            # The lower ranks storing the slabs sends part of their slab
            if should_send:
                # A non-blocking send is used, because the communication
                # is not pairwise.
                # In the x-dimension, the slabs are always thinner than
                # (or at least as thin as) the domain.
                chunk_send_bgn = i_chunk*thickness_chunk
                chunk_send_end = chunk_send_bgn + thickness_chunk
                if chunk_send_end > ℤ[slab.shape[0]]:
                    chunk_send_end = ℤ[slab.shape[0]]
                request = smart_mpi(
                    slab[
                        chunk_send_bgn:chunk_send_end,
                        ℤ[slab_sendrecv_j_start[ℓ]]:ℤ[slab_sendrecv_j_end[ℓ]],
                        ℤ[slab_sendrecv_k_start[ℓ]]:ℤ[slab_sendrecv_k_end[ℓ]],
                    ],
                    dest=rank_send,
                    mpifun='Isend',
                )
            # The corresponding process receives the message.
            # Since the slabs extend throughout the entire yz-plane,
            # we receive into the entire yz-part of the domain grid
            # (excluding ghost points).
            if should_recv:
                chunk_recv_bgn = ℤ[domain_sendrecv_i_start[ℓ]] + i_chunk*thickness_chunk
                chunk_recv_end = chunk_recv_bgn + thickness_chunk
                if chunk_recv_end > ℤ[domain_sendrecv_i_end[ℓ]]:
                    chunk_recv_end = ℤ[domain_sendrecv_i_end[ℓ]]
                smart_mpi(
                    grid_noghosts[
                        chunk_recv_bgn:chunk_recv_end,
                        :ℤ[grid_noghosts.shape[1]],
                        :ℤ[grid_noghosts.shape[2]],
                    ],
                    source=rank_recv,
                    mpifun='Recv',
                )
            # Wait for the non-blocking send to be complete before
            # continuing. Otherwise, data in the send buffer - which is
            # still in use by the non-blocking send - might get
            # overwritten by the next (non-blocking) send.
            if should_send:
                request.wait()
    # Populate ghost layers. Nullification takes precedence.
    if do_ghost_nullification:
        nullify_ghosts(grid)
    elif do_ghost_communication:
        communicate_ghosts(grid, '=')
    return grid

# Function for transferring data from domain grids to slabs
@cython.pheader(
    # Arguments
    grid='double[:, :, ::1]',
    slab_or_buffer_name=object,  # double[:, :, ::1], int or str
    prepare_fft='bint',
    # Locals
    N_domain2slabs_communications='Py_ssize_t',
    buffer_name=object,  # int or str
    chunk_recv_bgn='Py_ssize_t',
    chunk_recv_end='Py_ssize_t',
    chunk_send_bgn='Py_ssize_t',
    chunk_send_end='Py_ssize_t',
    domain_sendrecv_i_end='int[::1]',
    domain_sendrecv_i_start='int[::1]',
    domain2slabs_recvsend_ranks='int[::1]',
    grid_noghosts='double[:, :, :]',
    grids=dict,
    gridsize='Py_ssize_t',
    i_chunk='Py_ssize_t',
    n_chunks='Py_ssize_t',
    rank_recv='int',
    rank_send='int',
    request=object,  # mpi4py.MPI.Request
    shape=tuple,
    should_send='bint',
    should_recv='bint',
    slab='double[:, :, ::1]',
    slab_arr=object,
    slab_sendrecv_j_end='int[::1]',
    slab_sendrecv_j_start='int[::1]',
    slab_sendrecv_k_end='int[::1]',
    slab_sendrecv_k_start='int[::1]',
    slabs2domain_sendrecv_ranks='int[::1]',
    thickness_chunk='Py_ssize_t',
    ℓ='Py_ssize_t',
    returns='double[:, :, ::1]',
)
def slab_decompose(grid, slab_or_buffer_name=None, prepare_fft=False):
    """This function communicates a global domain decomposed grid into
    a global slab decomposed grid. If an existing slab grid should be
    used it can be passed as the second argument.
    Alternatively, if a slab grid should be fetched from elsewhere,
    its name should be specified as the second argument.

    If FFTs are to be carried out on a slab fetched by name,
    you must specify prepare_fft=True, in which case the slab will be
    created via FFTW.

    Note that ghost points will not be read by this function,
    and so the passed domain grid need not have these set properly.
    """
    if grid is None:
        return None
    if slab_or_buffer_name is None:
        slab_or_buffer_name = 'slab_global'
    # Determine the correct shape of the slab grid corresponding to
    # the passed domain grid.
    grid_noghosts = grid[
        nghosts:(grid.shape[0] - nghosts),
        nghosts:(grid.shape[1] - nghosts),
        nghosts:(grid.shape[2] - nghosts),
    ]
    gridsize = grid_noghosts.shape[0]*domain_subdivisions[0]
    if gridsize%nprocs != 0:
        abort(
            f'A domain decomposed grid of size {gridsize} was passed to the slab_decompose() '
            f'function. This grid size is not evenly divisible by {nprocs} processes.'
        )
    shape = get_slabshape_local(gridsize)
    # If no slab grid is passed, fetch a buffer of the right shape
    if isinstance(slab_or_buffer_name, (int, np.integer, str)):
        buffer_name = slab_or_buffer_name
        if prepare_fft:
            slab = get_fftw_slab(gridsize, buffer_name)
        else:
            slab = get_buffer(shape, buffer_name)
    else:
        slab = slab_or_buffer_name
        if asarray(slab).shape != shape:
            abort(
                f'slab_decompose(): The slab and domain grid have '
                f'incompatible shapes: {asarray(slab).shape}, {asarray(grid).shape}.'
            )
    # Nullify the additional elements in the padded dimension, which may
    # contain junk. This is not needed if an FFT is to be done.
    # However, some places the code skips the FFT because it knows that
    # the slab consists purely of zeros. Any junk in the additional
    # elements ruin this.
    # Due to a bug in cython, we need the below operation to be done by
    # NumPy, not directly on the memory view. See
    #   https://github.com/cython/cython/issues/2941
    slab_arr = asarray(slab)
    slab_arr[:, :, gridsize:] = 0
    # Compute needed communication variables
    (
        N_domain2slabs_communications,
        domain2slabs_recvsend_ranks,
        slabs2domain_sendrecv_ranks,
        domain_sendrecv_i_start,
        domain_sendrecv_i_end,
        slab_sendrecv_j_start,
        slab_sendrecv_j_end,
        slab_sendrecv_k_start,
        slab_sendrecv_k_end,
    ) = prepare_decomposition(grid, slab)
    # Communicate the domain grid to the slabs in chunks
    n_chunks, thickness_chunk = get_slab_domain_decomposition_chunk_size(
        slab, grid_noghosts,
    )
    for ℓ in range(N_domain2slabs_communications):
        should_send = (ℓ < ℤ[slabs2domain_sendrecv_ranks.shape[0]])
        should_recv = (ℓ < ℤ[domain2slabs_recvsend_ranks.shape[0]])
        if not should_send and not should_recv:
            continue
        if should_send:
            rank_send = slabs2domain_sendrecv_ranks[ℓ]
        if should_recv:
            rank_recv = domain2slabs_recvsend_ranks[ℓ]
        # Communicate the chunks
        for i_chunk in range(n_chunks):
            # Send part of the local domain
            # grid to the corresponding process.
            if should_send:
                # A non-blocking send is used, because the communication
                # is not pairwise.
                # Since the slabs extend throughout the entire yz-plane,
                # we should send the entire yz-part of domain
                # (excluding ghost points).
                chunk_send_bgn = ℤ[domain_sendrecv_i_start[ℓ]] + i_chunk*thickness_chunk
                chunk_send_end = chunk_send_bgn + thickness_chunk
                if chunk_send_end > ℤ[domain_sendrecv_i_end[ℓ]]:
                    chunk_send_end = ℤ[domain_sendrecv_i_end[ℓ]]
                request = smart_mpi(
                    grid_noghosts[
                        chunk_send_bgn:chunk_send_end,
                        :ℤ[grid_noghosts.shape[1]],
                        :ℤ[grid_noghosts.shape[2]],
                    ],
                    dest=rank_send,
                    mpifun='Isend',
                )
            # The lower ranks storing the slabs receives the message.
            # In the x-dimension, the slabs are always thinner than
            # (or at least as thin as) the domain.
            if should_recv:
                chunk_recv_bgn = i_chunk*thickness_chunk
                chunk_recv_end = chunk_recv_bgn + thickness_chunk
                if chunk_recv_end > ℤ[slab.shape[0]]:
                    chunk_recv_end = ℤ[slab.shape[0]]
                smart_mpi(
                    slab[
                        chunk_recv_bgn:chunk_recv_end,
                        ℤ[slab_sendrecv_j_start[ℓ]]:ℤ[slab_sendrecv_j_end[ℓ]],
                        ℤ[slab_sendrecv_k_start[ℓ]]:ℤ[slab_sendrecv_k_end[ℓ]],
                    ],
                    source=rank_recv,
                    mpifun='Recv',
                )
            # Wait for the non-blocking send to be complete before
            # continuing. Otherwise, data in the send buffer - which is
            # still in use by the non-blocking send - might get
            # overwritten by the next (non-blocking) send.
            if should_send:
                request.wait()
    return slab

# Helper function for slab and domain decomposition
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    grid_noghosts='double[:, :, :]',
    # Locals
    area='Py_ssize_t',
    n_chunks='Py_ssize_t',
    n_send='Py_ssize_t',
    n_send_max_allowed='Py_ssize_t',
    thickness='Py_ssize_t',
    thickness_chunk='Py_ssize_t',
    returns=tuple,
)
def get_slab_domain_decomposition_chunk_size(slab, grid_noghosts):
    """The communicating of the grid/slab data for slab/domain
    decomposition is done in chunks. These chunks are the full domain
    size in the y and z direction, and smaller or equal to the slab size
    in the x direction. This function computes and returns the number of
    chunks and their size along long x direction for such communication.
    """
    # Maximum number of elements (grid values) to communicate at a time
    n_send_max_allowed = 2**23  # 64 MB
    # Compute number of chunks and their thickness
    thickness = slab.shape[0]
    area = grid_noghosts.shape[1]*grid_noghosts.shape[2]
    n_send = thickness*area
    if n_send <= n_send_max_allowed:
        n_chunks = 1
        thickness_chunk = thickness
    else:
        n_chunks = n_send//n_send_max_allowed
        thickness_chunk = thickness//n_chunks
        if thickness_chunk*area > n_send_max_allowed:
            n_chunks += 1
            thickness_chunk = thickness//n_chunks
        if thickness_chunk < 1:
            thickness_chunk = 1
            n_chunks = thickness
        else:
            n_chunks = thickness//thickness_chunk
            if n_chunks*thickness_chunk < thickness:
                n_chunks += 1
    return n_chunks, thickness_chunk

# Iterator implementing looping over domain grids.
# The yielded values are the linear index as well as the
# 3D indices into the grid.
# If skip_ghosts is True, only non-ghost points will be yielded.
@cython.iterator(
    depends=(
        # Functions used by domain_loop()
        'get_gridshape_local',
   ),
)
def domain_loop(
    gridsize,
    *,
    skip_ghosts=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        gridsize='Py_ssize_t',
        skip_ghosts='bint',
        # Locals
        _i_bgn='Py_ssize_t',
        _i_end='Py_ssize_t',
        _index_i='Py_ssize_t',
        _index_ij='Py_ssize_t',
        _j_bgn='Py_ssize_t',
        _j_end='Py_ssize_t',
        _k_bgn='Py_ssize_t',
        _k_end='Py_ssize_t',
        _shape='Py_ssize_t[::1]',
        _nskip='Py_ssize_t',
        # Yielded
        index='Py_ssize_t',
        i='Py_ssize_t',
        j='Py_ssize_t',
        k='Py_ssize_t',
    )
    _shape = asarray(get_gridshape_local(gridsize), dtype=C2np['Py_ssize_t'])
    _nskip = nghosts*skip_ghosts
    _i_bgn = _nskip
    _i_end = _shape[0] - _nskip
    _j_bgn = _nskip
    _j_end = _shape[1] - _nskip
    _k_bgn = _nskip
    _k_end = _shape[2] - _nskip
    _index_i = _shape[1]*_shape[2]*(_nskip - 1)
    for i in range(_i_bgn, _i_end):
        _index_i += ℤ[_shape[1]*_shape[2]]
        _index_ij = _index_i + ℤ[_shape[2]*(_nskip - 1)]
        for j in range(_j_bgn, _j_end):
            _index_ij += ℤ[_shape[2]]
            index = _index_ij + ℤ[_nskip - 1]
            for k in range(_k_bgn, _k_end):
                index += 1
                yield index, i, j, k

# Iterator implementing looping over slabs in real space.
# For Fourier space slabs, see the fourier_loop() iterator instead.
# The yielded values are the linear index as well as the
# 3D indices into the slab.
# If skip_ghosts is True, only non-ghost points will be yielded.
@cython.iterator(
    depends=(
        # Functions used by slab_loop()
        'get_slabshape_local',
   ),
)
def slab_loop(
    gridsize,
    *,
    skip_data=False, skip_padding=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        gridsize='Py_ssize_t',  # int or array-like
        skip_data='bint',
        skip_padding='bint',
        # Locals
        _i_end='Py_ssize_t',
        _index_i='Py_ssize_t',
        _index_ij='Py_ssize_t',
        _k_bgn='Py_ssize_t',
        _k_end='Py_ssize_t',
        _shape='Py_ssize_t[::1]',
        _size_i='Py_ssize_t',
        _size_j='Py_ssize_t',
        _size_k='Py_ssize_t',
        _size_padding='Py_ssize_t',
        # Yielded
        index='Py_ssize_t',
        i='Py_ssize_t',
        j='Py_ssize_t',
        k='Py_ssize_t',
    )
    # Get slab shape
    _shape = asarray(get_slabshape_local(gridsize), dtype=C2np['Py_ssize_t'])
    _size_i = _shape[0]
    _size_j = _shape[1]
    _size_k = _size_j
    _size_padding = _shape[2]
    # Set up iteration limits
    _i_end = _size_i
    _k_bgn = 0
    _k_end = _size_padding
    if skip_data:
        if skip_padding:
            _i_end = 0
        _k_bgn = _size_k
    elif skip_padding:
        _k_end = _size_k
    # Perform iteration
    _index_i = _k_bgn - _size_padding*(_size_j + 1) - 1
    for i in range(_i_end):
        _index_i += ℤ[_size_j*_size_padding]
        _index_ij = _index_i
        for j in range(_size_j):
            _index_ij += _size_padding
            index = _index_ij
            for k in range(_k_bgn, _k_end):
                index += 1
                yield index, i, j, k

# Iterator implementing looping over Fourier space slabs.
# The yielded values are the linear index into the slab, the physical
# ki, kj, kk (in grid units), the combined factor due to deconvolution
# and interlacing, as well as the angle by which to rotate a shifted
# slab for interlacing.
# The iteration is determined from the passed gridsize. By default the
# same grid size is used in the computations of deconvolution and
# interlacing, though this may be specified separately
# through gridsize_corrections.
# If sparse is True, only unique points in the z DC plane will
# be visited.
# If skip_origin is True, the ki = kj = kk = 0 point will be excluded
# from the iteration.
# By default, all points except those on Nyquist planes are included in
# the iteration. If you only want points for which
#   ki**2 + kj**2 + kk**2 < k2_max,
# specify this k2_max (in grid units). For the largest sphere possible
# (i.e. excluding "corner modes"), still excluding points on Nyquist
# planes, set k2_max = (nyquist - 1)**2 = (gridsize//2 - 1)**2.
# If the returned factor is to be used for deconvolution, specify the
# deconvolution (interpolation) order as deconv_order. If the factor and
# θ are to be used for interlacing, specify the lattice defining the
# shifts for the interlacing. Both deconv_order and lattice may be
# specified simultaneously.
@cython.iterator(
    depends=(
        # Classes used by fourier_loop()
        'Lattice',
    ),
)
def fourier_loop(
    gridsize, gridsize_corrections=-1,
    i_bgn=0, i_end=None,
    j_bgn=0, j_end=None,
    *,
    sparse=False, skip_origin=False, k2_max=-1,
    deconv_order=0, interlace_lattice=None,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        gridsize='Py_ssize_t',
        gridsize_corrections='Py_ssize_t',
        i_bgn='Py_ssize_t',
        i_end=object,
        j_bgn='Py_ssize_t',
        j_end=object,
        sparse='bint',
        skip_origin='bint',
        k2_max='Py_ssize_t',
        deconv_order='int',
        interlace_lattice='Lattice',
        # Locals
        _deconv_i_denom='double',
        _deconv_i_numer='double',
        _deconv_ij_denom='double',
        _deconv_ij_numer='double',
        _deconv_j_denom='double',
        _deconv_j_numer='double',
        _deconv_k_denom='double',
        _deconv_k_numer='double',
        _gridsize='Py_ssize_t',
        _i_chunk='int',
        _i_chunk_bgn='Py_ssize_t',
        _i_chunk_end='Py_ssize_t',
        _i_end='Py_ssize_t',
        _index_ij='Py_ssize_t',
        _index_j='Py_ssize_t',
        _j_chunk='int',
        _j_chunk_bgn='Py_ssize_t',
        _j_chunk_end='Py_ssize_t',
        _j_end='Py_ssize_t',
        _j_global='Py_ssize_t',
        _j_global_bgn='Py_ssize_t',
        _j_global_chunk_bgn='Py_ssize_t',
        _j_global_chunk_end='Py_ssize_t',
        _j_global_end='Py_ssize_t',
        _ki_bgn='Py_ssize_t',
        _ki_chunk_bgn='Py_ssize_t',
        _ki_chunk_end='Py_ssize_t',
        _ki_end='Py_ssize_t',
        _ki_max='Py_ssize_t',
        _kj_bgn='Py_ssize_t',
        _kj_chunk_bgn='Py_ssize_t',
        _kj_chunk_end='Py_ssize_t',
        _kj_end='Py_ssize_t',
        _kj_max='Py_ssize_t',
        _kk_bgn='Py_ssize_t',
        _kk_end='Py_ssize_t',
        _nyquist='Py_ssize_t',
        _offset_j='Py_ssize_t',
        _slab_size_i='Py_ssize_t',
        _slab_size_j='Py_ssize_t',
        _slab_size_k='Py_ssize_t',
        # Yielded
        index='Py_ssize_t',
        ki='Py_ssize_t',
        kj='Py_ssize_t',
        kk='Py_ssize_t',
        factor='double',
        θ='double',
    )
    if interlace_lattice is None:
        interlace_lattice = Lattice()
    # Set up slab shape. Avoid NumPy ints in pure Python mode.
    _gridsize = int(gridsize)
    _nyquist = _gridsize//2
    _slab_size_j = _gridsize//nprocs
    _slab_size_i = _gridsize
    _slab_size_k = _gridsize + 2
    if gridsize_corrections == -1:
        gridsize_corrections = _gridsize
    # Set default end indices if not given. After this,
    # _j_end and _i_end should be used instead of j_end and i_end.
    if j_end is None:
        # Default to entire local length along this dimension
        _j_end = _slab_size_j
    else:
        _j_end = j_end
    if i_end is None:
        # Default to entire length along this dimension
        _i_end = _slab_size_i
    else:
        _i_end = i_end
    # The slabs are distributed along the j-dimension,
    # introducing a global offset individual to each process.
    _offset_j = _slab_size_j*rank
    _j_global_bgn = _offset_j +  j_bgn
    _j_global_end = _offset_j + _j_end
    # Begin iterating over slab. As the first and second dimensions
    # are transposed due to the FFT, the j-dimension is first.
    # We loop through the j-dimension in two chunks, skipping the
    # Nyquist point (kj = -_nyquist ⇒ _j_global = _nyquist) in between.
    for _j_chunk in range(2):
        if _j_chunk == 0:
            _j_chunk_bgn = j_bgn
            _j_global_chunk_end = pairmin(_j_global_end, _nyquist)
            _j_chunk_end = _j_global_chunk_end - _offset_j
        else:  # _j_chunk == 1
            _j_global_chunk_bgn = pairmax(_j_global_bgn, ℤ[_nyquist + 1])
            _j_chunk_bgn = _j_global_chunk_bgn - _offset_j
            _j_chunk_end = _j_end
        _kj_chunk_bgn = _offset_j + _j_chunk_bgn - ℤ[-_j_chunk & _gridsize]
        _kj_chunk_end = _offset_j + _j_chunk_end - ℤ[-_j_chunk & _gridsize]
        # We similarly loop through the i-dimension in two chunks,
        # skipping the Nyquist point (ki = -_nyquist ⇒ _i = _nyquist)
        # in between.
        for _i_chunk in range(2):
            if _i_chunk == 0:
                _i_chunk_bgn = i_bgn
                _i_chunk_end = pairmin(_i_end, _nyquist)
            else:  # _i_chunk == 1
                _i_chunk_bgn = pairmax(i_bgn, ℤ[_nyquist + 1])
                _i_chunk_end = _i_end
            _ki_chunk_bgn = _i_chunk_bgn - ℤ[-_i_chunk & _gridsize]
            _ki_chunk_end = _i_chunk_end - ℤ[-_i_chunk & _gridsize]
            # Loop over the j chunk
            _kj_bgn = _kj_chunk_bgn
            _kj_end = _kj_chunk_end
            _index_j = ℤ[_j_chunk_bgn*_slab_size_i - _slab_size_i - 1]
            with unswitch(2):
                if k2_max != -1:
                    _kj_max = ℤ[isqrt(k2_max*(k2_max != -1))]
                    _kj_bgn = pairmax(_kj_bgn, -_kj_max)
                    _kj_end = pairmin(_kj_end, _kj_max + 1)
                    _index_j -= (
                        (
                            -_j_chunk & -(_kj_chunk_bgn < -_kj_max)
                        ) & (_kj_max + _kj_chunk_bgn)
                    )*_slab_size_i
            for kj in range(_kj_bgn, _kj_end):
                _index_j += _slab_size_i
                # The j-component of the 1D NGP deconvolution factor.
                # This is given by
                #   deconv_1D_NGP = 1/sinc(ky*Δx/2)
                #                 = 1/sinc(kj*π/gridsize)
                # with
                #   sinc(x) ≡ sin(x)/x
                # and
                #   sinc(0) = 1.
                # Here ky = 2*π/boxsize*kj is the y component of the
                # physical wave vector and Δx = boxsize/gridsize is the
                # grid spacing. The 3D deconvolution factor is then
                # given as a product over the three 1D deconvolution
                # factors, and the final higher-order deconvolution
                # factor is obtained through exponentiation with the
                # power given by the deconvolution order (**1 → NGP,
                # **2 → CIC, **3 → TSC, **4 → PCS).
                _deconv_j_numer = kj*ℝ[π/gridsize_corrections] + machine_ϵ
                _deconv_j_denom = sin(_deconv_j_numer)
                # Loop over the i chunk
                _ki_bgn = _ki_chunk_bgn
                _ki_end = _ki_chunk_end
                _index_ij = _i_chunk_bgn + _index_j
                with unswitch(3):
                    if k2_max != -1:
                        _ki_max = isqrt(k2_max - ℤ[kj**2])
                        _ki_bgn = pairmax(_ki_bgn, -_ki_max)
                        _ki_end = pairmin(_ki_end, _ki_max + 1)
                        _index_ij -= (
                            -_i_chunk & -(_ki_chunk_bgn < -_ki_max)
                            & (_ki_max + _ki_chunk_bgn)
                        )
                _index_ij *= _slab_size_k
                for ki in range(_ki_bgn, _ki_end):
                    _index_ij += _slab_size_k
                    # The product of the i- and the j-components
                    # of the 1D NGP deconvolution factor.
                    _deconv_i_numer = ki*ℝ[π/gridsize_corrections] + machine_ϵ
                    _deconv_i_denom = sin(_deconv_i_numer)
                    _deconv_ij_numer = _deconv_i_numer*_deconv_j_numer
                    _deconv_ij_denom = _deconv_i_denom*_deconv_j_denom
                    # The origin is the first element encountered on the
                    # master process. If the partial index _index_ij has
                    # a value of 0, then this is the first i iteration,
                    # meaning that the origin (with kk = 0) is included
                    # in the upcoming kk iteration. If the origin should
                    # be skipped, we do so by excluding kk = 0 from this
                    # next iteration.
                    _kk_bgn = 0
                    with unswitch(4):
                        if skip_origin:
                            with unswitch(4):
                                if master:
                                    _kk_bgn = (_index_ij == 0)
                    # The z DC plane consists of complex conjugate pairs
                    # of points. When looping sparsely we only want to
                    # hit one point from each pair. To do this, we
                    # choose to skip points with positive ki and also
                    # points with positive kj and ki == 0. Note that we
                    # could switch the roles of ki and kj here, but this
                    # produces a work load imbalance as skipping all
                    # positive kj points means that half of the
                    # processes do not participate.
                    with unswitch(4):
                        if sparse:
                            # Remember to not disregard a possible
                            # value of _kk_bgn = 1 from the above
                            # skip of the origin.
                            _kk_bgn |= (ki > 0) | ((ki == 0) & 𝔹[kj > 0])
                    # Only the non-negative part of the k-dimension
                    # exists. Loop through this half, one complex number
                    # at a time, looping directly over kk instead of
                    # k == 2*kk. Here the Nyquist point kk = _nyquist is
                    # the last element along this dimension, and so we
                    # skip it by simply not including it in the range.
                    _kk_end = _nyquist
                    with unswitch(4):
                        if k2_max != -1:
                            _kk_end = pairmin(_kk_end, 1 + isqrt(k2_max - ℤ[ℤ[kj**2] + ℤ[ki**2]]))
                    index = _index_ij + 2*(_kk_bgn - 1)
                    for kk in range(_kk_bgn, _kk_end):
                        # Index into (real part of) complex Fourier mode
                        index += 2
                        # Combined factor for deconvolution
                        # and interlacing.
                        factor = 1
                        # The full deconvolution factor
                        with unswitch(5):
                            if deconv_order:
                                # The 3D NGP deconvolution factor
                                _deconv_k_numer = kk*ℝ[π/gridsize_corrections] + machine_ϵ
                                _deconv_k_denom = sin(_deconv_k_numer)
                                factor = (
                                     (_deconv_ij_numer*_deconv_k_numer)
                                    /(_deconv_ij_denom*_deconv_k_denom)
                                )
                                # The full deconvolution factor
                                factor **= deconv_order
                        factor *= ℝ[1/len(interlace_lattice)]
                        # Include factor from interlacing
                        # and compute interlacing angle.
                        θ = 0
                        with unswitch(5):
                            if interlace_lattice.shift != (0, 0, 0):
                                # The angle by which to rotate the
                                # complex phase of the shifted grid at
                                # this k⃗, before ("harmonically")
                                # averaging together with the other
                                # grid(s). This angle is given by
                                #   θ =             (kx*shiftx + ky*shifty + kz*shiftz)
                                #     = 2π/boxsize *(ki*shiftx + kj*shifty + kk*shiftz)
                                #     = 2π/gridsize*(ki*shifti + kj*shiftj + kk*shiftk)
                                # We put in a sign, meaning that θ
                                # computed matches particles shifted
                                # by (-shifti, -shiftj, -shiftk).
                                θ = (
                                    ℝ[
                                        + ℝ[
                                            ki*ℝ[-2*π/gridsize_corrections
                                            *interlace_lattice.shift[0]]
                                        ]
                                        + ℝ[
                                            kj*ℝ[-2*π/gridsize_corrections
                                            *interlace_lattice.shift[1]]
                                        ]
                                    ]
                                        + ℝ[
                                            kk*ℝ[-2*π/gridsize_corrections
                                            *interlace_lattice.shift[2]]
                                        ]
                                )
                        # Yield the needed variables
                        yield index, ki, kj, kk, factor, θ

# Iterator implementing looping over Fourier space, using the
# space-filling curve implemented by get_fourier_curve_coords().
# The iteration will be over the full Fourier space (i.e. all slabs) on
# all processes. The yielded values are the linear index into the slab,
# the physical ki, kj, kk (in grid units) and a Boolean inside_slab,
# specifying whether the current point is within the local slab or not.
# The iteration is determined from the passed gridsize.
# All points except those on Nyquist planes are included in
# the iteration.
# If skip_origin is True, the ki = kj = kk = 0 point will be excluded
# from the iteration.
@cython.iterator(
    depends=(
        # Functions used by fourier_curve_loop()
        'get_fourier_curve_coords',
    ),
)
def fourier_curve_loop(
    gridsize,
    *,
    skip_origin=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        gridsize='Py_ssize_t',
        skip_origin='bint',
        # Locals
        _coords='Py_ssize_t*',
        _i='Py_ssize_t',
        _j='Py_ssize_t',
        _j_global='Py_ssize_t',
        _k='Py_ssize_t',
        _key='Py_ssize_t',
        _n_nyquist='Py_ssize_t',
        _n_total='Py_ssize_t',
        _nyquist='Py_ssize_t',
        _slab_size_i='Py_ssize_t',
        _slab_size_j='Py_ssize_t',
        _slab_size_k='Py_ssize_t',
        # Yielded
        index='Py_ssize_t',
        ki='Py_ssize_t',
        kj='Py_ssize_t',
        kk='Py_ssize_t',
        inside_slab='bint',
    )
    # Set up slab shape
    _nyquist = gridsize//2
    _slab_size_j = gridsize//nprocs
    _slab_size_i = gridsize
    _slab_size_k = gridsize + 2
    _n_total = gridsize*gridsize*(_nyquist + 1)
    _n_nyquist = gridsize**2 + _nyquist*(2*gridsize - 1)
    # Iterate over Fourier space using the space-filling curve
    for _key in range(skip_origin, _n_total - _n_nyquist):
        # Lookup Fourier curve key
        _coords = get_fourier_curve_coords(_key)
        ki, kj, kk = _coords[0], _coords[1], _coords[2]
        # Convert (ki, kj, kk) to slab index
        _j_global = kj + (-(kj < 0) & gridsize)
        _j = _j_global - ℤ[_slab_size_j*rank]
        _i = ki + (-(ki < 0) & gridsize)
        _k = 2*kk
        index = (_j*_slab_size_i + _i)*_slab_size_k + _k
        # Boolean specifying whether this point is inside the local slab
        inside_slab = (0 <= _j < _slab_size_j)
        # Yield the needed variables
        yield index, ki, kj, kk, inside_slab

# Iterator implementing looping over a single j-slice of thickness 1 of
# Fourier space, with the iteration order being that of the
# space-filling curve implemented by get_fourier_curve_coords(), though
# for the slice with kj = 0 regardless of the j provided. This means
# that the visiting order is the same for all slices.
# The yielded values are the linear index into the slab, as well as the
# physical ki and kk (in grid units). Note that
# kj = j_global - (-(j_global >= nyquist) & gridsize) with
# j_global = slab_size_j*rank + j is constant during the iteration over
# the slice and so is not yielded. The iteration is determined from the
# passed gridsize. All points except those on Nyquist planes are
# included in the iteration.
# If skip_origin is True, the ki = kj = kk = 0 point will be excluded
# from the iteration.
@cython.iterator(
    depends=(
        # Functions used by fourier_curve_slice_loop()
        'get_fourier_curve_coords',
    ),
)
def fourier_curve_slice_loop(
    gridsize, j=0,
    *,
    skip_origin=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        gridsize='Py_ssize_t',
        j='Py_ssize_t',
        skip_origin='bint',
        # Locals
        _coords='Py_ssize_t*',
        _f='Py_ssize_t',
        _g='Py_ssize_t',
        _i='Py_ssize_t',
        _k='Py_ssize_t',
        _key='Py_ssize_t',
        _key_bgn='Py_ssize_t',
        _num='Py_ssize_t',
        _nyquist='Py_ssize_t',
        _s='Py_ssize_t',
        _slab_size_i='Py_ssize_t',
        _slab_size_k='Py_ssize_t',
        _step='Py_ssize_t',
        # Yielded
        index='Py_ssize_t',
        ki='Py_ssize_t',
        kk='Py_ssize_t',
    )
    # Set up slab shape
    _nyquist = gridsize//2
    _slab_size_i = gridsize
    _slab_size_k = gridsize + 2
    # Iterate over Fourier space using slice of space-filling curve
    for _s in range(_nyquist):
        for _f in range(1 + (-(_s < ℤ[_nyquist - 1]) & 2)):  # avoids Nyquist planes
            _key_bgn = (
                + _f**2 + (_f == 2)
                + (1 + 6*_f - (_f == 1))*_s
                + (5 + 4*_f - (_f == 2))*ℤ[_s**2]
                + ℤ[4*_s**3]
            )
            _num = -(
                𝔹[𝔹[not (skip_origin and master and j == 0)] | (_s != 0)] | (_f != 0)
            ) & (ℤ[_s + 1]*(1 + (_f == 2)))
            _step = 1 + (-(_f == 2) & (_num - 1))
            _key = _key_bgn - _step
            for _g in range(_num):
                _key += _step
                # Lookup Fourier curve key
                _coords = get_fourier_curve_coords(_key)
                ki, kk = _coords[0], _coords[2]  # kj == 0
                # Convert (ki, kk) to slab index
                _i = ki + (-(ki < 0) & gridsize)
                _k = 2*kk
                index = (ℤ[j*_slab_size_i] + _i)*_slab_size_k + _k
                # Yield the needed variables
                yield index, ki, kk

# Functions implementing a 3D Fourier space-filling curve,
# with get_fourier_curve_key() and get_fourier_curve_coords() being
# inverses of each other. The curve starts at the origin,
# get_fourier_curve_coords(key=0) = (0, 0, 0), with larger keys
# gradually mapping all of Fourier space, in a spiral-like pattern.
# For a given |k⃗|**2 = ki**2 + kj**2 + kk**2, all Nyquist points have a
# larger key than non-Nyquist points.
@cython.header(
    # Arguments
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    # Locals
    ki_abs='Py_ssize_t',
    kj_abs='Py_ssize_t',
    kl='Py_ssize_t',
    kl_abs='Py_ssize_t',
    s='Py_ssize_t',
    returns='Py_ssize_t',
)
def get_fourier_curve_key(ki, kj, kk):
    """The output domain (image) of this function is all of the integers
      0 ≤ i < gridsize*gridsize*(gridsize/2 + 1)
    when called with all integer triplets (ki, kj, kk)
    from the input domain
      -gridsize/2 ≤ ki < gridsize/2
      -gridsize/2 ≤ kj < gridsize/2
                0 ≤ kk ≤ gridsize/2
    for any value of gridsize.
    """
    ki_abs, kj_abs = abs(ki), abs(kj)
    if ki_abs > kj_abs:
        kl = ki
    elif ki_abs < kj_abs:
        kl = kj
    else:
        kl = pairmax(ki, kj)
    kl_abs = abs(kl)
    if kl_abs < kk:
        s = kk
    else:
        s = kl_abs + (kl > 0)
    if kk == s:
        return kj + s*(2*ki + s*(4*s + 2) + 1)
    if kj == -s:
        return kk + s*(ki + s*(4*s - 1))
    if ki == -s:
        return kk + s*(kj + s*(4*s - 3))
    if ki != s - 1:
        return kk + s*(ki + s*(4*s - 5) + 2)
    return kk + s*(kj + s*(4*s - 7) + 3)
@cython.header(
    # Arguments
    key='Py_ssize_t',
    # Locals
    f0_size='Py_ssize_t',
    f1_size='Py_ssize_t',
    f2_size='Py_ssize_t',
    f3_size='Py_ssize_t',
    g='Py_ssize_t',
    s='Py_ssize_t',
    returns='Py_ssize_t*',
)
def get_fourier_curve_coords(key):
    """The output domain (image) of this function is all of the
    integer triplets (ki, kj, kk) with
      -gridsize/2 ≤ ki < gridsize/2
      -gridsize/2 ≤ kj < gridsize/2
                0 ≤ kk ≤ gridsize/2
    when called with all integers from the input domain
      0 ≤ i < gridsize*gridsize*(gridsize/2 + 1)
    for any value of gridsize.
    """
    if not cython.compiled:
        # This function makes use of the trick that (-True & i) == i,
        # but the negative operator '-' is not supported
        # on NumPy Boolean scalars.
        key = int(key)
    g = icbrt(2*key)
    g += g & 1
    g += -(g**2*(g//2 + 1) <= key) & 2
    s = g//2
    key -= (g - 2)**2*s
    f0_size = ℤ[2*s**2] - s
    if key < f0_size:
        fourier_coords[0] = +s - 1
        fourier_coords[1] = -s + 1 + key//s
        fourier_coords[2] = key%s
        return fourier_coords
    f1_size = ℤ[2*s**2] - 2*s + f0_size
    if key < f1_size:
        key -= f0_size
        fourier_coords[0] = -s + 1 + key//s
        fourier_coords[1] = +s - 1
        fourier_coords[2] = key%s
        return fourier_coords
    f2_size = f0_size + f1_size
    if key < f2_size:
        key -= f1_size
        fourier_coords[0] = -s
        fourier_coords[1] = -s + 1 + key//s
        fourier_coords[2] = key%s
        return fourier_coords
    f3_size = ℤ[2*s**2] + f2_size
    if key < f3_size:
        key -= f2_size
        fourier_coords[0] = -s + key//s
        fourier_coords[1] = -s
        fourier_coords[2] = key%s
        return fourier_coords
    key -= f3_size
    fourier_coords[0] = -s + key//(2*s)
    fourier_coords[1] = -s + key%(2*s)
    fourier_coords[2] = +s
    return fourier_coords
# Global array used as the return value of get_fourier_curve_coords()
cython.declare(
    fourier_coords_mv='Py_ssize_t[::1]',
    fourier_coords='Py_ssize_t*',
)
fourier_coords_mv = empty(3, dtype=C2np['Py_ssize_t'])
fourier_coords = cython.address(fourier_coords_mv[:])

# Iterator implementing looping over spherical shells within
# Fourier space slabs.
# The yielded values are the linear index into the slab as well as the
# physical ki, kj, kk (in grid units).
# The iteration is determined from the passed gridsize, as well as
# k_min (the minimum floating-point k in grid units) and
# k_max (the maximum floating-point k in grid units), both of which are
# assumed positive. With cell (0, 0, 0) placed at the origin, it covers
# a range from |k| = 0 to |k| = sqrt(3*½**2) = sqrt(3)/2.
# If skip_origin is True, the ki = kj = kk = 0 point will be excluded
# from the iteration.
# If skip_negative_ki is True, only the ki ≥ 0 quarter of the shell will
# be visited (remember that kk < 0 is always skipped, as this half is
# not represented in memory). For a given point (ki, kj, kk),
# the reflection (-ki, kj, kk) always recides on the same process,
# hence why we choose skip_negative_ki rather than skip_negative_kj.
@cython.iterator
def fourier_shell_loop(
    gridsize, k_min, k_max,
    *,
    skip_origin=False, skip_negative_ki=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        gridsize='Py_ssize_t',
        k_min='double',
        k_max='double',
        skip_origin='bint',
        skip_negative_ki='bint',
        # Locals
        _i_bgn='Py_ssize_t',
        _i_chunk='int',
        _index_ij='Py_ssize_t',
        _index_j='Py_ssize_t',
        _j_chunk='int',
        _k_max_1D='Py_ssize_t',
        _k2_max='double',
        _k2_max_2D='Py_ssize_t',
        _k2_min='double',
        _ki_bgn='Py_ssize_t',
        _ki_end='Py_ssize_t',
        _ki_lim='Py_ssize_t',
        _ki2_corner_max='double',
        _ki2_corner_min='double',
        _kj_bgn='Py_ssize_t',
        _kj_end='Py_ssize_t',
        _kj_lim='Py_ssize_t',
        _kj2_corner_max='double',
        _kj2_corner_min='double',
        _kk_bgn='Py_ssize_t',
        _kk_end='Py_ssize_t',
        _kk2_corner_max='double',
        _kk2_corner_min='double',
        _nyquist='Py_ssize_t',
        _offset_j='Py_ssize_t',
        _slab_size_i='Py_ssize_t',
        _slab_size_j='Py_ssize_t',
        _slab_size_k='Py_ssize_t',
        # Yielded
        index='Py_ssize_t',
        ki='Py_ssize_t',
        kj='Py_ssize_t',
        kk='Py_ssize_t',
    )
    # Set up slab shape
    _nyquist = gridsize//2
    _slab_size_j = gridsize//nprocs
    _slab_size_i = gridsize
    _slab_size_k = gridsize + 2
    # Min and max floating k²
    _k2_min = k_min**2
    _k2_max = k_max**2
    # Min and max integer k and k² values for grid points in
    # (or very near) the shell.
    _k_max_1D  = int(    ((k_max + 0.5*sqrt(1))   ))
    _k2_max_2D = int(    ((k_max + 0.5*sqrt(2))**2))
    _k_max_1D  = pairmin(_k_max_1D,  1*(_nyquist - 1)   )
    _k2_max_2D = pairmin(_k2_max_2D, 2*(_nyquist - 1)**2)
    # Carry out the iteration
    _kj_lim = _k_max_1D
    _offset_j = -_slab_size_j*rank
    for _j_chunk in range(2):
        if _j_chunk == 0:
            _kj_bgn = ℤ[_slab_size_j*rank]
            _kj_end = pairmin(_kj_lim + 1, ℤ[_slab_size_j*(rank + 1)])
        else:  # _j_chunk == 1
            _kj_bgn = pairmax(-_kj_lim, ℤ[_slab_size_j*rank] - gridsize)
            _kj_end = pairmin(0, ℤ[_slab_size_j*(rank + 1)] - gridsize)
            _offset_j += gridsize
        for _i_chunk in range(2 - skip_negative_ki):
            _index_j = ℤ[(_kj_bgn + _offset_j)*_slab_size_i - ℤ[_slab_size_i + 1]]
            for kj in range(_kj_bgn, _kj_end):
                _index_j += _slab_size_i
                _ki_lim = isqrt(_k2_max_2D - ℤ[kj**2])
                _ki_lim = pairmin(_ki_lim, _k_max_1D)
                with unswitch(1):
                    if _i_chunk == 0:
                        _ki_bgn = 0
                        _ki_end = _ki_lim + 1
                        _i_bgn = _ki_bgn
                    else:  # _i_chunk == 1
                        _ki_bgn = -_ki_lim
                        _ki_end = 0
                        _i_bgn = _ki_bgn + gridsize
                _kj2_corner_min = (ℝ[abs(kj)] + 0.5*𝔹[kj != 0])**2
                _kj2_corner_max = (ℝ[abs(kj)] - 0.5*𝔹[kj != 0])**2
                _index_ij = (_i_bgn + _index_j)*_slab_size_k
                for ki in range(_ki_bgn, _ki_end):
                    _index_ij += _slab_size_k
                    _ki2_corner_min = (ℝ[abs(ki)] + 0.5*𝔹[ki != 0])**2
                    _ki2_corner_max = (ℝ[abs(ki)] - 0.5*𝔹[ki != 0])**2
                    _kk2_corner_min = _k2_min - _kj2_corner_min - _ki2_corner_min
                    _kk2_corner_max = _k2_max - _kj2_corner_max - _ki2_corner_max
                    _kk_bgn = int(sqrt(_kk2_corner_min*(_kk2_corner_min > 0)) + 0.5)
                    with unswitch(4):
                        if skip_origin:
                            with unswitch(4):
                                if master:
                                    _kk_bgn += (_index_ij == 0) & (_kk_bgn == 0)
                    _kk_end = int(sqrt(_kk2_corner_max*(_kk2_corner_max > 0)) + 0.5)
                    _kk_end = pairmin(_kk_end, _k_max_1D)
                    index = _index_ij + 2*(_kk_bgn - 1)
                    for kk in range(_kk_bgn, _kk_end + 1):
                        index += 2
                        yield index, ki, kj, kk

# Function performing in-place deconvolution and/or interlacing
# and/or differentiation of Fourier slabs.
@cython.pheader(
    # Arguments
    slab='double[:, :, ::1]',
    deconv_order='int',
    lattice='Lattice',
    diff_dim='int',
    # Locals
    cosθ='double',
    factor='double',
    gridsize='Py_ssize_t',
    im='double',
    index='Py_ssize_t',
    interlace_lattice='Lattice',
    k_fundamental='double',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    kl='Py_ssize_t',
    re='double',
    sinθ='double',
    slab_ptr='double*',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_size_k='Py_ssize_t',
    θ='double',
    returns='double[:, :, ::1]',
)
def fourier_operate(slab, deconv_order=0, lattice=None, diff_dim=-1):
    """Deconvolution will be performed according to deconv_order.
    To do an interlacing you need to call this function once for each
    sub-lattice. To perform differentiation along x, y or z,
    pass diff_dim = 0, 1 or 2, respectively. Note that this convention
    ignores the fact that the first two dimensions are transposed
    in Fourier space.
    """
    if lattice is None:
        lattice = Lattice()
    # Bail out if neither deconvolution nor interlacing
    # nor differentiation is to be performed.
    if (
        deconv_order == 0
        and (len(lattice) == 1 and lattice.shift == (0, 0, 0))
        and diff_dim == -1
    ):
        return slab
    if not (-1 <= diff_dim < 3):
        abort(f'fourier_operate() called with diff_dim = {diff_dim} ∉ {{-1, 0, 1, 2}}')
    # Extract slab shape and pointer
    slab_size_j, slab_size_i, slab_size_k = asarray(slab).shape
    slab_ptr = cython.address(slab[:, :, :])
    # If no deconvolution, no differentiation and only the first
    # (scaling only) stage of interlacing is to be performed, we can do
    # this directly, without use of the Fourier loop.
    if (
        deconv_order == 0
        and (len(lattice) > 1 and lattice.shift == (0, 0, 0))
        and diff_dim == -1
    ):
        for index in range(slab_size_j*slab_size_i*slab_size_k):
            slab_ptr[index] *= ℝ[1/len(lattice)]
        return slab
    # Perform the deconvolution and interlacing
    k_fundamental = ℝ[2*π/boxsize]
    gridsize = slab_size_i
    interlace_lattice = lattice
    for index, ki, kj, kk, factor, θ in fourier_loop(
        gridsize, deconv_order=deconv_order, interlace_lattice=interlace_lattice,
    ):
        # Extract real and imag part of slab
        re = slab_ptr[index    ]
        im = slab_ptr[index + 1]
        # Rotate the complex phase due to interlacing
        with unswitch:
            if interlace_lattice.shift != (0, 0, 0):
                cosθ = cos(θ)
                sinθ = sin(θ)
                re, im = (
                    re*cosθ - im*sinθ,
                    re*sinθ + im*cosθ,
                )
        # Differentiate by multiplying by the imaginary unit and the
        # given component of k⃗ in physical units.
        with unswitch:
            if 𝔹[diff_dim != -1]:
                kl = (
                    ℤ[
                          ℤ[ℤ[-(diff_dim == 0)] & ki]
                        | ℤ[ℤ[-(diff_dim == 1)] & kj]
                    ]
                        | ℤ[ℤ[-(diff_dim == 2)] & kk]
                )
                factor *= k_fundamental*kl
                re, im = -im, re
        # Apply factor from deconvolution,
        # interlacing and differentiation.
        re *= factor
        im *= factor
        # Store updated values back in slabs
        slab_ptr[index    ] = re
        slab_ptr[index + 1] = im
    return slab

# Function for in-place inverting the Laplacian in
#   ∇²Φ = source
@cython.pheader(
    # Arguments
    source='double[:, :, ::1]',
    factor='double',
    # Locals
    amplitude='double',
    gridsize='Py_ssize_t',
    index='Py_ssize_t',
    k2='Py_ssize_t',
    k_fundamental='double',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    scaling='double',
    source_ptr='double*',
    θ='double',
    returns='double[:, :, ::1]',
)
def laplacian_inverse(source, factor=1):
    """This function will operate on the source field in-place.
    It is assumed that this is already in Fourier space,
    and the result will be returned in Fourier space as well.
    """
    gridsize = source.shape[1]
    source_ptr = cython.address(source[:, :, :])
    # Invert Laplacian by dividing by -k²
    k_fundamental = ℝ[2*π/boxsize]
    scaling = factor  # rename to not clash with fourier_loop()
    for index, ki, kj, kk, factor, θ in fourier_loop(gridsize, skip_origin=True):
        k2 = ℤ[ℤ[ℤ[kj**2] + ki**2] + kk**2]
        amplitude = ℝ[-scaling/k_fundamental**2]/k2
        source_ptr[index    ] *= amplitude
        source_ptr[index + 1] *= amplitude
    return source

# Function for differentiation of a Fourier space slab, once or twice
@cython.pheader(
    # Argumens
    slab='double[:, :, ::1]',
    dim0='int',
    dim1='int',
    buffer_name=object,  # int or str
    slab_output='double[:, :, ::1]',
    factor='double',
    # Locals
    amplitude='double',
    gridsize='Py_ssize_t',
    im='double',
    index='Py_ssize_t',
    k_fundamental='double',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    kl0='Py_ssize_t',
    kl1='Py_ssize_t',
    re='double',
    scaling='double',
    slab_output_ptr='double*',
    slab_ptr='double*',
    θ='double',
    returns='double[:, :, ::1]',
)
def fourier_diff(slab, dim0, dim1=-1, buffer_name=None, slab_output=None, factor=1):
    """This function expects slab to be in Fourier space.
    The return value will be a new slab (unless slab_output is passed)
    also in Fourier space.
    """
    gridsize = slab.shape[1]
    slab_ptr = cython.address(slab[:, :, :])
    # Get new slab for storing the dim'th component of the vector field
    if slab_output is None:
        slab_output = get_fftw_slab(gridsize, buffer_name)
    slab_output_ptr = cython.address(slab_output[:, :, :])
    nullify_modes(slab_output, ['origin', 'nyquist'])
    # Do the differentiation in Fourier space by multiplying by ikᵢ
    # for each dim specified.
    k_fundamental = ℝ[2*π/boxsize]
    scaling = factor  # rename to not clash with fourier_loop()
    for index, ki, kj, kk, factor, θ in fourier_loop(gridsize, skip_origin=True):
        re = slab_ptr[index    ]
        im = slab_ptr[index + 1]
        # Differentiation (multiplication by ikᵢ)
        kl0 = (
            ℤ[
                  ℤ[ℤ[-(dim0 == 0)] & ki]
                | ℤ[ℤ[-(dim0 == 1)] & kj]
            ]
                | ℤ[ℤ[-(dim0 == 2)] & kk]
        )
        re, im = -im, re
        with unswitch:
            if dim1 == -1:
                amplitude = ℝ[scaling*k_fundamental]*kl0
            else:
                kl1 = (
                    ℤ[
                          ℤ[ℤ[-(dim1 == 0)] & ki]
                        | ℤ[ℤ[-(dim1 == 1)] & kj]
                    ]
                        | ℤ[ℤ[-(dim1 == 2)] & kk]
                )
                re, im = -im, re
                amplitude = ℝ[scaling*k_fundamental**2]*kl0*kl1
        # Store results
        slab_output_ptr[index    ] = amplitude*re
        slab_output_ptr[index + 1] = amplitude*im
    return slab_output

# Function for nullifying sets of modes of Fourier space slabs
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    nullifications=object,  # None, bint, str or list of str's
    # Locals
    gridsize='Py_ssize_t',
    index='Py_ssize_t',
    i='Py_ssize_t',
    j='Py_ssize_t',
    j_bgn='Py_ssize_t',
    j_end='Py_ssize_t',
    j_global='Py_ssize_t',
    k='Py_ssize_t',
    k_min='Py_ssize_t',
    ki='Py_ssize_t',
    ki_bgn='Py_ssize_t',
    ki_end='Py_ssize_t',
    kj='Py_ssize_t',
    kj_bgn='Py_ssize_t',
    kj_end='Py_ssize_t',
    kk='Py_ssize_t',
    kk_bgn='Py_ssize_t',
    kk_end='Py_ssize_t',
    match=object,  # re.Match
    nullification=str,
    nyquist='Py_ssize_t',
    slab_ptr='double*',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_size_k='Py_ssize_t',
    returns='void',
)
def nullify_modes(slab, nullifications):
    """The nullifications argument can be a bool, a str of
    comma-separated words, or alternatively a list of str's, each being
    a single word. The words specify which types of modes to nullify:
    - False, None: Do not perform any nullification.
    - True, "all": Nullify the entire slab.
    - "origin": Nullify the origin ki = kj = kk = 0.
    - "Nyquist": Nullify the three Nyquist planes:
        ki = -nyquist, -nyquist ≤ kj < nyquist, 0 ≤ kk ≤ nyquist.
        kj = -nyquist, -nyquist ≤ ki < nyquist, 0 ≤ kk ≤ nyquist.
        kk = +nyquist, -nyquist ≤ ki < nyquist, -nyquist ≤ kj < nyquist.
    - "beyond cube of |k| < num" with num some positive int literal:
        Nullifies everything except points (ki, kj, kk) with at least
        one |kl| < num, leaving an un-nullified cube
        centred at the origin.
    """
    if slab is None:
        return
    slab_ptr = cython.address(slab[:, :, :])
    # Parse nullifications
    if isinstance(nullifications, str):
        nullifications = nullifications.split(',')
    else:
        nullifications = any2list(nullifications)
    nullifications = [
        str(nullification_obj).strip().lower()
        for nullification_obj in nullifications
    ]
    # Get slab dimensions
    slab_size_j, slab_size_i, slab_size_k = asarray(slab).shape
    gridsize = slab_size_i
    nyquist = gridsize//2
    # Perform nullifications
    for nullification in nullifications:
        if not nullification or nullification in {'false', 'none'}:
            # Do not perform any nullification
            continue
        elif nullification in {'true', 'all'}:
            # Nullify entire slab
            slab[...] = 0
        elif nullification == 'origin':
            # Nullify the origin point ki == kj == kk == 0. This is
            # always located as the first element on the master process.
            if master:
                slab_ptr[0] = 0  # real part
                slab_ptr[1] = 0  # imag part
        elif nullification == 'nyquist':
            # Nullify the three Nyquist planes ki = -nyquist,
            # kj = -nyquist and kk = +nyquist. These planes overlap
            # pairwise at the edges and so a little effort can be spared
            # by not nullifying these edges twice. We take this into
            # account for the two edges in the kk = +nyquist plane but
            # not for the remaining ki = kj = -nyquist edge, as skipping
            # the Nyquist point along the i or j direction (unlike along
            # the k direction) requires additional logic.
            ki = -nyquist
            i = ki + (-(ki < 0) & gridsize)
            for j in range(slab_size_j):
                for k in range(0, gridsize, 2):  # exclude k = gridsize (kk = nyquist)
                    index = ℤ[(j*slab_size_i + i)*slab_size_k] + k
                    slab_ptr[index    ] = 0  # real part
                    slab_ptr[index + 1] = 0  # imag part
            kj = -nyquist
            j_global = kj + (-(kj < 0) & gridsize)
            j = j_global - ℤ[slab_size_j*rank]
            if 0 <= j < slab_size_j:
                for i in range(gridsize):
                    for k in range(0, gridsize, 2):  # exclude k = gridsize (kk = nyquist)
                        index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                        slab_ptr[index    ] = 0  # real part
                        slab_ptr[index + 1] = 0  # imag part
            kk = +nyquist
            k = 2*kk
            for j in range(slab_size_j):
                for i in range(gridsize):
                    index = (ℤ[j*slab_size_i] + i)*slab_size_k + k
                    slab_ptr[index    ] = 0  # real part
                    slab_ptr[index + 1] = 0  # imag part
        elif nullification.startswith('beyond cube'):
            # Nullify everything except for a central cube with
            # half-width k_min determined from the nullification str.
            # As only half of the full 3D Fourier space is represented
            # in memory, the region to be nullified can be partitioned
            # into 2×2 + 1 = 5 regions;
            #   ki <= -k_min, k_min <= ki,
            #   kj <= -k_min, k_min <= kj,
            #   k_min <= kk.
            # Below these 2×2 + 1 nullifications are performed
            # such that overlapping parts of the 5 regions are not
            # nullified more than once.
            match = re.search(r'^beyond cube of \|k\| < (\d+)$', nullification)
            if not match:
                abort(f'nullify_modes(): wrongly formatted nullification "{nullification}"')
            k_min = int(match.group(1))
            for ki_bgn, ki_end in zip((-nyquist, k_min), (-k_min + 1, nyquist)):
                for j in range(slab_size_j):
                    for ki in range(ki_bgn, ki_end):
                        i = ki + (-(ki < 0) & gridsize)
                        for k in range(0, slab_size_k, 2):
                            index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                            slab_ptr[index    ] = 0  # real part
                            slab_ptr[index + 1] = 0  # imag part
            ki_bgn, ki_end = -k_min + 1, k_min
            for kj_bgn, kj_end in zip((-nyquist, k_min), (-k_min + 1, nyquist)):
                j_bgn = kj_bgn + ℤ[(-(kj_bgn < 0) & gridsize) - ℤ[slab_size_j*rank]]
                j_end = kj_end + ℤ[(-(kj_bgn < 0) & gridsize) - ℤ[slab_size_j*rank]]
                j_bgn = pairmax(j_bgn, 0)
                j_end = pairmin(j_end, slab_size_j)
                for j in range(j_bgn, j_end):
                    for ki in range(ki_bgn, ki_end):
                        i = ki + (-(ki < 0) & gridsize)
                        for k in range(0, slab_size_k, 2):
                            index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                            slab_ptr[index    ] = 0  # real part
                            slab_ptr[index + 1] = 0  # imag part
            ki_bgn, ki_end = -k_min + 1, k_min
            kk_bgn, kk_end = k_min, nyquist + 1
            for kj_bgn, kj_end in zip((-k_min + 1, 0), (0, k_min)):
                j_bgn = kj_bgn + ℤ[(-(kj_bgn < 0) & gridsize) - ℤ[slab_size_j*rank]]
                j_end = kj_end + ℤ[(-(kj_bgn < 0) & gridsize) - ℤ[slab_size_j*rank]]
                j_bgn = pairmax(j_bgn, 0)
                j_end = pairmin(j_end, slab_size_j)
                for j in range(j_bgn, j_end):
                    for ki in range(ki_bgn, ki_end):
                        i = ki + (-(ki < 0) & gridsize)
                        for k in range(ℤ[2*kk_bgn], ℤ[2*kk_end], 2):
                            index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                            slab_ptr[index    ] = 0  # real part
                            slab_ptr[index + 1] = 0  # imag part
        else:
            abort(f'nullify_modes(): nullification "{nullification}" not understood')

# Function for nullifying ghost points
@cython.header(
    # Arguments
    grid='double[:, :, ::1]',
    # Locals
    i='Py_ssize_t',
    i_bgn='Py_ssize_t',
    i_end='Py_ssize_t',
    j='Py_ssize_t',
    j_bgn='Py_ssize_t',
    j_end='Py_ssize_t',
    k='Py_ssize_t',
    k_bgn='Py_ssize_t',
    k_end='Py_ssize_t',
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    returns='void',
)
def nullify_ghosts(grid):
    """This function will nullify the ghost points of the supplied grid.
    The (ghost) planes referred to below all have thickness nghosts.
    """
    if grid is None:
        return
    size_i = grid.shape[0]
    size_j = grid.shape[1]
    size_k = grid.shape[2]
    # The entire x planes
    for i_bgn, i_end in zip((0, ℤ[size_i - nghosts]), (nghosts, size_i)):
        for         i in range(i_bgn, i_end):
            for     j in range(size_j):
                for k in range(size_k):
                    grid[i, j, k] = 0
    # The y planes except their overlap with the x planes
    for j_bgn, j_end in zip((0, ℤ[size_j - nghosts]), (nghosts, size_j)):
        for         i in range(nghosts, ℤ[size_i - nghosts]):
            for     j in range(j_bgn, j_end):
                for k in range(size_k):
                    grid[i, j, k] = 0
    # The z planes except their overlap with the x and y planes
    for k_bgn, k_end in zip((0, size_k - nghosts), (nghosts, size_k)):
        for         i in range(nghosts, ℤ[size_i - nghosts]):
            for     j in range(nghosts, ℤ[size_j - nghosts]):
                for k in range(k_bgn, k_end):
                    grid[i, j, k] = 0

# Function for populating the padding region of slabs with a given value
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    value='double',
    # Locals
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    index='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    slab_ptr='double*',
    returns='void',
)
def fill_slab_padding(slab, value):
    gridsize = slab.shape[1]
    slab_ptr = cython.address(slab[:, :, :])
    for index, i, j, k in slab_loop(gridsize, skip_data=True):
        slab_ptr[index] = value

# Function that returns a slab decomposed grid,
# allocated by FFTW.
@cython.pheader(
    # Arguments
    gridsize='Py_ssize_t',
    buffer_name=object,  # int or str or None
    nullify=object,  # bint, str or list of str's
    # Locals
    acquire='bint',
    as_expected='bint',
    fftw_plans_index='Py_ssize_t',
    fftw_struct=fftw_return_struct,
    plan_backward=fftw_plan,
    plan_forward=fftw_plan,
    shape=tuple,
    slab='double[:, :, ::1]',
    slab_address='Py_ssize_t',
    slab_ptr='double*',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_start_i='Py_ssize_t',
    slab_start_j='Py_ssize_t',
    wisdom_filename=str,
    returns='double[:, :, ::1]',
)
def get_fftw_slab(gridsize, buffer_name=None, nullify=False):
    global fftw_plans_size, fftw_plans_forward, fftw_plans_backward
    if buffer_name is None:
        buffer_name = 'slab_global'
    # If this slab has already been constructed, fetch it
    slab = slabs.get((gridsize, buffer_name))
    if slab is not None:
        nullify_modes(slab, nullify)
        return slab
    # Checks on the passed gridsize
    if gridsize%nprocs != 0:
        abort(
            f'A grid size of {gridsize} was passed to the get_fftw_slab() function. '
            f'This grid size is not evenly divisible by {nprocs} processes.'
        )
    if gridsize%2 != 0:
        masterwarn(
            f'An odd grid size ({gridsize}) was passed to the get_fftw_slab() function. '
            f'Some operations may not function correctly.'
    )
    shape = get_slabshape_local(gridsize)
    # In pure Python mode we use NumPy, which really means that there
    # is no needed preparations. In compiled mode we use FFTW,
    # which means that the grid and its plans must be prepared.
    if not cython.compiled:
        slab = empty(shape, dtype=C2np['double'])
    else:
        # Get path to FFTW wisdom file
        wisdom_filename = get_wisdom_filename(gridsize)
        # Initialise fftw_mpi, allocate the grid, initialise the
        # local grid sizes and start indices and do FFTW planning.
        acquire = False
        if master:
            os.makedirs(os.path.dirname(wisdom_filename), exist_ok=True)
            if gridsize not in wisdom_acquired and not os.path.isfile(wisdom_filename):
                acquire = True
                masterprint(
                    f'Acquiring FFTW wisdom ({fftw_wisdom_rigor}) for grid size {gridsize} ...'
                )
        fftw_struct = fftw_setup(
            gridsize, gridsize, gridsize,
            bytes(fftw_wisdom_rigor, encoding='ascii'),
            fftw_wisdom_reuse,
            bytes(wisdom_filename, encoding='ascii'),
        )
        if acquire:
            masterprint('done')
        wisdom_acquired[gridsize] = True
        # Unpack every variable from fftw_struct
        # and compare to expected values.
        slab_size_i   = int(fftw_struct.gridsize_local_i)
        slab_size_j   = int(fftw_struct.gridsize_local_j)
        slab_start_i  = int(fftw_struct.gridstart_local_i)
        slab_start_j  = int(fftw_struct.gridstart_local_j)
        plan_forward  = fftw_struct.plan_forward
        plan_backward = fftw_struct.plan_backward
        slab_ptr      = fftw_struct.grid
        as_expected = True
        if (
               slab_size_i  != ℤ[shape[0]]
            or slab_size_j  != ℤ[shape[0]]
            or slab_start_i != ℤ[shape[0]*rank]
            or slab_start_j != ℤ[shape[0]*rank]
        ):
            as_expected = False
            warn(
                f'FFTW has distributed a slab of grid size {gridsize} differently '
                f'from what was expected on rank {rank}:\n'
                f'    slab_size_i  = {slab_size_i}, expected {shape[0]},\n'
                f'    slab_size_j  = {slab_size_j}, expected {shape[0]},\n'
                f'    slab_start_i = {slab_start_i}, expected {shape[0]*rank},\n'
                f'    slab_start_j = {slab_start_j}, expected {shape[0]*rank},\n'
            )
        as_expected = allreduce(as_expected, op=MPI.LOR)
        if not as_expected:
            abort('Refusing to carry on with this non-expected decomposition')
        # Wrap the slab pointer in a memory view. Looping over this
        # memory view should be done as noted in fft.c, but use
        # slab[i, j, k] when in real space and slab[j, i, k]
        # when in Fourier space.
        slab = cast(slab_ptr, 'double[:shape[0], :shape[1], :shape[2]]')
        # Store the plans for this slab in the global
        # fftw_plans_forward and fftw_plans_backward arrays.
        fftw_plans_index = fftw_plans_size
        fftw_plans_size += 1
        fftw_plans_forward  = realloc(fftw_plans_forward , fftw_plans_size*sizeof('fftw_plan'))
        fftw_plans_backward = realloc(fftw_plans_backward, fftw_plans_size*sizeof('fftw_plan'))
        fftw_plans_forward [fftw_plans_index] = plan_forward
        fftw_plans_backward[fftw_plans_index] = plan_backward
        # Insert mapping from the slab to the index of its plans
        # in the global fftw_plans_forward and fftw_plans_backward
        # arrays, into the global fftw_plans_mapping dict.
        slab_address = cast(cython.address(slab[:, :, :]), 'Py_ssize_t')
        fftw_plans_mapping[slab_address] = fftw_plans_index
    # Store and return this slab
    slabs[gridsize, buffer_name] = slab
    nullify_modes(slab, nullify)
    return slab
# Cache storing slabs. The keys have the format (gridsize, buffer_name).
cython.declare(slabs=dict)
slabs = {}
# Arrays of FFTW plans
cython.declare(
    fftw_plans_size='Py_ssize_t',
    fftw_plans_forward ='fftw_plan*',
    fftw_plans_backward='fftw_plan*',
)
fftw_plans_size = 0
fftw_plans_forward  = malloc(fftw_plans_size*sizeof('fftw_plan'))
fftw_plans_backward = malloc(fftw_plans_size*sizeof('fftw_plan'))
# Mapping from memory addresses of slabs to indices in
# fftw_plans_forward and fftw_plans_backward.
cython.declare(fftw_plans_mapping=dict)
fftw_plans_mapping = {}
# Dict keeping track of what FFTW wisdom has already been acquired
cython.declare(wisdom_acquired=dict)
wisdom_acquired = {}

# Function that frees the memory of grid allocated by FFTW
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    buffer_name=object,  # int or str or None
    # Locals
    slab='double[:, :, ::1]',
    slab_ptr='double*',
    returns='void',
)
def free_fftw_slab(gridsize, buffer_name):
    if buffer_name is None:
        buffer_name = 'slab_global'
    # If this slab has already been constructed, fetch it
    slab = slabs.pop((gridsize, buffer_name), None)
    if slab is None:
        abort(
            f'free_fftw_slab(): No slab with '
            f'gridsize = {gridsize}, buffer_name = {buffer_name}'
        )
    if not cython.compiled:
        try:
            [slab].pop().resize(0, refcheck=False)
        except Exception:
            pass
    else:
        slab_ptr = cython.address(slab[:, :, :])
        fftw_free(slab_ptr)
        # Note that we leave the FFTW plans as is

# Helper function for the get_fftw_slab() function,
# which construct the absolute path to the wisdom file to use.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    # Locals
    content=str,
    fftw_pkgconfig_filename=str,
    filename=str,
    index='Py_ssize_t',
    match=object,  # re.Match
    node_process_count=object,  # collections.Counter
    other_node='int',
    other_node_name=str,
    primary_nodes=list,
    process_count='Py_ssize_t',
    process_count_max='Py_ssize_t',
    returns=str,
)
def get_wisdom_filename(gridsize):
    """The FFTW wisdom file name is built as a hash of several things:
    - The passed grid size.
    - The total number of processes.
    - The global FFTW wisdom rigour.
    - The FFTW version.
    - The name of the node "owning" the wisdom in the case of
      fftw_wisdom_share being False. Here a node is said to own the
      wisdom if it hosts the majority of the processes. A more elaborate
      key like the complete MPI layout is of no use, as FFTW wisdom is
      really generated on each process, after which the wisdom of one is
      chosen arbitrarily as the wisdom to stick with.
      When fftw_wisdom_share is True, this part of the key is constant.
    """
    global fftw_version, wisdom_owner
    # The master process constructs the file name
    # and then broadcasts it.
    if not master:
        return bcast()
    # Get the version of FFTW in use
    if not fftw_version:
        fftw_version = '<unknown>'
        fftw_pkgconfig_filename = f'{path.fftw_dir}/lib/pkgconfig/fftw3.pc'
        if os.path.exists(fftw_pkgconfig_filename):
            with open_file(fftw_pkgconfig_filename, mode='r') as fftw_pkgconfig_file:
                content = fftw_pkgconfig_file.read()
            match = re.search('Version.*?([0-9].*)', content)
            if match:
                fftw_version = match.group(1)
            else:
                masterwarn('Failed to determine FFTW version from fftw3.pc')
        else:
            masterwarn('Could not find the fftw3.pc file needed to determine the FFTW version')
    # Get the name of the node owning the wisdom
    if not wisdom_owner:
        if fftw_wisdom_share:
            wisdom_owner = '<shared>'
        else:
            node_process_count = collections.Counter()
            for other_node in range(nnodes):
                other_node_name = node_numbers2names[other_node]
                node_process_count[other_node_name] += np.sum(asarray(nodes) == other_node)
            primary_nodes = node_process_count.most_common(len(node_process_count))
            process_count_max = primary_nodes[0][1]
            for index, (other_node_name, process_count) in enumerate(primary_nodes):
                if process_count < process_count_max:
                    primary_nodes = primary_nodes[:index]
                    break
            wisdom_owner = sorted([  # guarantees deterministic outcome in case of ties
                other_node_name
                for other_node_name, process_count in primary_nodes
            ])[0]
    # The full path to the wisdom file
    filename = get_reusable_filename(
        'fftw',
        gridsize, nprocs, fftw_wisdom_rigor, fftw_version, wisdom_owner,
        extension='wisdom',
    )
    # Broadcast and return result
    return bcast(filename)
# Constant strings set and used by the get_wisdom_filename function
cython.declare(fftw_version=str, wisdom_owner=str)
fftw_version = ''
wisdom_owner = ''

# Function performing Fourier transformations of slab decomposed grids
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    direction=str,
    apply_forward_normalization='bint',
    # Locals
    fftw_plans_index='Py_ssize_t',
    slab_address='Py_ssize_t',
    returns='void',
)
def fft(slab, direction, apply_forward_normalization=False):
    """Fourier transform the given slab decomposed grid.
    For a forwards transformation from real to Fourier space, supply
    direction='forward'. By default this is an unnormalised transform,
    as defined by FFTW. To do the normalization, divide all elements of
    the slab by gridsize**3, where gridsize is the linear grid size
    of the cubic grid. This will be done if you further set
    apply_forward_normalization to True.
    For a backwards transformation from Fourier to real space, supply
    direction='backward'. Here, no further normalization is needed,
    as defined by FFTW.

    In pure Python, NumPy is used to carry out the Fourier transforms.
    To emulate the effects of FFTW perfectly, a lot of extra steps
    are needed.
    """
    if slab is None:
        return
    if direction not in ('forward', 'backward'):
        abort(
            f'fft() was called with the direction "{direction}", '
            f'which is neither "forward" nor "backward".'
        )
    if not cython.compiled:
        # Do to floating-point inaccuracies, the order in which the
        # three dimensions are handled by the FFT matters slightly
        # (this is the case for both NumPy and FFTW). By setting
        # do_all_axis_permutations to True, the FFT will be carried out
        # using all 6 axis permutation and then averaged. This should
        # only be used for testing.
        do_all_axis_permutations = False
        if do_all_axis_permutations:
            permutations = list(itertools.permutations(range(3)))
        else:
            permutations = [tuple(range(3))]
        # The pure Python FFT implementation is serial.
        # Every process computes the entire FFT of the temporary
        # variable grid_global_pure_python.
        slab_size_i = slab_size_j = slab.shape[0]
        slab_start_i = slab_size_i*rank
        slab_start_j = slab_size_j*rank
        gridsize = slab.shape[1]
        gridsize_padding = slab.shape[2]
        grid_global_pure_python = empty(
            (gridsize, gridsize, gridsize_padding),
            dtype=C2np['double'],
        )
        Allgatherv(slab, grid_global_pure_python)
        if direction == 'forward':
            # Delete the padding on the last dimension
            grid_global_pure_python = grid_global_pure_python[:, :, :gridsize]
            # Do Fourier transform via NumPy
            if do_all_axis_permutations:
                # We do not make use of the real transform,
                # as we then cannot permute the axes.
                grid_global_pure_python_fft = 0
                for permutation in permutations:
                    grid_global_pure_python_fft += np.transpose(
                        np.fft.fftn(
                            np.transpose(grid_global_pure_python, permutation),
                            norm='backward',
                        ),
                        [permutation.index(dim) for dim in range(3)],
                    )[:, :, :gridsize_padding//2]
                grid_global_pure_python = grid_global_pure_python_fft/len(permutations)
            else:
                grid_global_pure_python = np.fft.rfftn(grid_global_pure_python, norm='backward')
            # FFTW transposes the first two dimensions
            grid_global_pure_python = grid_global_pure_python.transpose((1, 0, 2))
            # FFTW represents the complex array by doubles only
            tmp = empty((gridsize, gridsize, gridsize_padding), dtype=C2np['double'])
            for i in range(gridsize_padding):
                if i%2:
                    tmp[:, :, i] = grid_global_pure_python.imag[:, :, i//2]
                else:
                    tmp[:, :, i] = grid_global_pure_python.real[:, :, i//2]
            grid_global_pure_python = tmp
            # As in FFTW, distribute the slabs along the y-dimension
            # (which is the first dimension now, due to transposing).
            asarray(slab)[...] = grid_global_pure_python[
                slab_start_j:(slab_start_j + slab_size_j), :, :,
            ]
        else:  # direction == 'backward':
            # FFTW represents the complex array by doubles only.
            # Switch to using complex entries.
            tmp = zeros((gridsize, gridsize, gridsize_padding//2), dtype=np.complex128)
            for i in range(gridsize_padding):
                tmp[:, :, i//2] += (1j if i & 1 else 1)*grid_global_pure_python[:, :, i]
            grid_global_pure_python = tmp
            # FFTW transposes the first
            # two dimensions back to normal.
            grid_global_pure_python = grid_global_pure_python.transpose((1, 0, 2))
            # Do inverse Fourier transform via NumPy
            if do_all_axis_permutations:
                # The full grid is needed to permute the axes.
                # Construct the full grid using the Hermitian symmetry.
                tmp = empty((gridsize, gridsize, gridsize), dtype=np.complex128)
                tmp[:gridsize, :gridsize, :gridsize_padding//2] = grid_global_pure_python
                for i in range(gridsize):
                    i_conj = -i + (-(i > 0) & gridsize)
                    for j in range(gridsize):
                        j_conj = -j + (-(j > 0) & gridsize)
                        for k in range(gridsize_padding//2, gridsize):
                            k_conj = -k + (-(k > 0) & gridsize)
                            tmp[i, j, k] = grid_global_pure_python[i_conj, j_conj, k_conj].conj()
                grid_global_pure_python = tmp
                # We do not make use of the real transform,
                # as we then cannot permute the axes.
                grid_global_pure_python_fft = 0
                for permutation in permutations:
                    grid_global_pure_python_fft += np.transpose(
                        np.fft.ifftn(
                            np.transpose(grid_global_pure_python, permutation),
                            s=[gridsize]*3, norm='forward',
                        ),
                        [permutation.index(dim) for dim in range(3)],
                    )
                grid_global_pure_python = grid_global_pure_python_fft.real/len(permutations)
            else:
                grid_global_pure_python = np.fft.irfftn(
                    grid_global_pure_python, s=[gridsize]*3, norm='forward',
                )
            # Add padding on last dimension, as in FFTW
            padding = empty(
                (gridsize, gridsize, gridsize_padding - gridsize),
                dtype=C2np['double'],
            )
            grid_global_pure_python = np.concatenate((grid_global_pure_python, padding), axis=2)
            # As in FFTW, distribute the slabs along the x-dimension
            asarray(slab)[...] = grid_global_pure_python[
                slab_start_i:(slab_start_i + slab_size_i), :, :,
            ]
    else:  # Compiled mode
        # Look up the index of the FFTW plans for the passed slab.
        slab_address = cast(cython.address(slab[:, :, :]), 'Py_ssize_t')
        if slab_address not in fftw_plans_mapping:
            abort('slab passed to fft() not obtained through get_fftw_slab()')
        fftw_plans_index = fftw_plans_mapping[slab_address]
        # Look up the plan and let FFTW do the Fourier transformation
        if 𝔹[direction == 'forward']:
            fftw_execute(fftw_plans_forward[fftw_plans_index])
        else:  # direction == 'backward':
            fftw_execute(fftw_plans_backward[fftw_plans_index])
    # Apply normalization after forward transform, if specified
    if 𝔹[direction == 'forward'] and apply_forward_normalization:
        fft_normalize(slab)

# Function for normalizing Fourier slabs,
# needed after a forward FFT.
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    # Locals
    factor='double',
    gridsize='Py_ssize_t',
    index='Py_ssize_t',
    slab_ptr='double*',
    returns='double[:, :, ::1]',
)
def fft_normalize(slab):
    gridsize = slab.shape[1]
    factor = float(gridsize)**(-3)
    slab_ptr = cython.address(slab[:, :, :])
    for index in range(slab.shape[0]*slab.shape[1]*slab.shape[2]):
        slab_ptr[index] *= factor
    return slab

# Function for checking that the slabs satisfy the required symmetry
# of a Fourier transformed real field.
@cython.remove
def slabs_check_symmetry(
    slab, nullified_nyquist=False, nullified_origin=False,
    gridsize=-1, allow_zeros=False, pure_embedding=True, count_information=True,
    test_fft=False, rel_tol=1e-12, abs_tol=machine_ϵ,
):
    """This function checks and reports on the symmetries that a Fourier
    transformed 3D grid (passed as FFTW slabs) of real data should obey.
    Two distinct symmetries exist:
    - Bulk inversion through the origin, i.e. grid point
      [ki, kj, kk] should be the complex conjugate of [-ki, -kj, -kk].
    - Plane inversion through the centre of each Nyquist plane,
      e.g. the grid point [-nyquist, kj, kk] should be the complex
      conjugate of [-nyquist, -kj, -kk].
    In the above, {ki, kj, kk} are the components of the physical k
    vector (in grid units in the code below), not indices i, j, k.
    Also, nyquist = gridsize//2.
    A standard 3D grid is tabulated at
      ki ∈ [-nyquist, -nyquist + 1, ..., nyquist - 1],
      kj ∈ [-nyquist, -nyquist + 1, ..., nyquist - 1],
      kk ∈ [0, 1, ..., nyquist],
    i.e. the z direction is cut roughly in half and the positive x and
    y Nyquist planes are not part of the tabulation. The surviving
    Nyquist planes are then the negative x and y and the positive z.

    From the bulk and Nyquist plane inversion symmetries it follows that
    (see the docstring of get_purely_reals() below) the grid point at
    the origin, grid points at centres of Nyquist planes, at centres of
    Nyquist edges and at corners must all be real. This is also checked.

    An example of a Nyquist edge could be [-nyquist, kj, nyquist]. Such
    edges are subject to three of the four symmetries. Applying all
    three, we have
        grid[-nyquist, +kj, +nyquist]
      → grid[+nyquist, -kj, -nyquist]*  (bulk inversion)
      → grid[+nyquist, +kj, +nyquist]   (x Nyquist plane inversion)
      → grid[-nyquist, -kj, +nyquist]*  (z Nyquist plane inversion)
    i.e. we effectively have a new symmetry; edge inversion, again under
    complex conjugation. Though this follows logically from the basic
    bulk and plane inversion symmetries, it is possible to satisfy these
    individually but not combined. Thus, a separate check for this edge
    symmetry is carried out. Note that all relevant edges lie in the
    positive z Nyquist plane, and that only two (one at ki = -nyquist
    and one at kj = -nyquist) are within the tabulated region.

    If a gridsize is further supplied to this function, it signals that
    the passed grid contains within it a smaller, embedded grid of the
    given size. This brings the positive x and y Nyquist planes within
    the tabulated region as well, and so the symmetry conditions will
    also be checked here. Note that the symmetry of plane inversion for
    the positive x and y Nyquist planes are not distinct symmetries, but
    follows from bulk inversion and the negative Nyquist plane
    symmetries. Thus, even in the case of an embedded grid, checking the
    bulk symmetry (now including ki, kj = +nyquist) and the three
    original Nyquist plane symmetries suffice. Similarly, combining the
    bulk symmetry with one of the Nyquist plane symmetries, a "new"
    symmetry is produced, akin to the aforementioned edge symmetries:
        grid[-nyquist, +kj, +kk]
      → grid[+nyquist, -kj, -kk]*  (bulk inversion)
      → grid[+nyquist, +kj, +kk]   (x Nyquist plane inversion)
    i.e. pairs of parallel Nyquist planes are identical copies of one
    another, without doing plane inversion or conjugation. Again, since
    this symmetry is implied by the basic bulk and three Nyquist plane
    symmetries (and automatically enforced once each of these are
    satisfied — unlike the edge symmetry), this will not be checked
    explicitly. Finally, note that the edge symmetries at kk = +nyquist
    and ki = +nyquist or kj = +nyquist are also guaranteed to be
    satisfied if the bulk symmetry and the corresponding ki = -nyquist
    and kj = -nyquist symmetries are satisfied, and so these two extra
    edge symmetries will also not be explicitly checked.

    For further explanation of symmetries and data layout,
    see the comments and especially docstrings below.

    An easy way to falsely satisfy the symmetry conditions is by having
    grid cells being equal to zero. When allow_zeros is False, any zeros
    found within the embedded grid will be interpreted as failures.

    If an embedded grid is passed and pure_embedding is True,
    it is further checked that all modes outside of the small grid
    are exactly zero.

    Too much symmetry — as in wrongly copied data points — is also bad.
    When count_information is True, all (real and imaginary) numbers in
    the grid are compared against one another in order to find the total
    information content of the grid. This is then compared to what it
    should be for a Fourier transformed grid which had no symmetries at
    all in real space, i.e. each Fourier mode is independent. If the
    input grid is the Fourier transform of some tabulated analytical
    function, you should probably set count_information to zero.

    If nullified_nyquist is True, it signals that the grid is supposed
    to have nullified Nyquist planes. This will be taken into account
    when searching for (non-)zeros and when counting the information
    content. Also, a check that the Nyquist planes really are nullified
    will be added.

    If nullified_origin is True, it signals that the grid is supposed
    to have a nullified origin. This will be taken into account
    when searching for (non-)zeros and when counting the information
    content. Also, a check that the Nyquist planes really are nullified
    will be added.

    Note that this function is not written with performance in mind
    and should not be called during actual simulation, and never with
    large grids.
    """
    # Get grid size
    gridsize_large = slab.shape[1]
    if gridsize == -1:
        gridsize = gridsize_large
        pure_embedding = False
    if gridsize%2:
        abort(f'Cannot check symmetry of grid of odd grid size {gridsize}')
    if gridsize_large%2:
        abort(
            f'Cannot check symmetry of grid embedded within a larger grid '
            f'of odd grid size {gridsize_large}'
        )
    if gridsize_large < gridsize:
        abort(
            f'The passed gridsize ({gridsize}) should be less than the '
            f'grid size of the grid ({gridsize_large})'
        )
    nyquist = gridsize//2
    nyquist_large = gridsize_large//2
    masterprint(f'Checking slab symmetries of grid of size {gridsize} ...')
    # Gather all slabs into global grid on the master process
    if master:
        grid = empty((gridsize, gridsize, slab.shape[2]), dtype=C2np['double'])
        grid[:slab.shape[0], :, :] = slab[...]
        for slave in range(1, nprocs):
            j_bgn = slab.shape[0]*slave
            j_end = j_bgn + slab.shape[0]
            smart_mpi(grid[j_bgn:j_end, :, :], source=slave, mpifun='recv')
    else:
        smart_mpi(slab, dest=master_rank, mpifun='send')
        return
    # Perform and compare real and complex FFT
    if test_fft:
        grid_r = zeros((gridsize, gridsize, slab.shape[2]//2), dtype=np.complex128)
        for i in range(slab.shape[2]):
            grid_r[:, :, i//2] += (1j if i & 1 else 1)*grid[:, :, i]
        grid_c = empty((gridsize, gridsize, gridsize), dtype=np.complex128)
        grid_c[:gridsize, :gridsize, :slab.shape[2]//2] = grid_r
        for i in range(gridsize):
            i_conj = -i + (-(i > 0) & gridsize)
            for j in range(gridsize):
                j_conj = -j + (-(j > 0) & gridsize)
                for k in range(slab.shape[2]//2, gridsize):
                    k_conj = -k + (-(k > 0) & gridsize)
                    grid_c[i, j, k] = grid_r[i_conj, j_conj, k_conj].conj()
        grid_rf = np.fft.irfftn(grid_r, s=[gridsize]*3, norm='forward')
        grid_cf = np.fft.ifftn (grid_c, s=[gridsize]*3, norm='forward')
        if not np.isclose(grid_rf, grid_cf, rtol=rel_tol, atol=1e+5*abs_tol).all():
            mask = ~np.isclose(grid_rf, grid_cf, rtol=rel_tol, atol=1e+5*abs_tol)
            print((grid_rf - grid_cf)[mask])
            fancyprint(
                'Hermitian symmetry seems to be violated '
                'as real and complex transforms disagree',
                wrap=False,
                file=sys.stderr,
            )
    # Create set of [ki, kj, kk] where grid points
    # ought to be purely real.
    def get_purely_reals(nyquist):
        """Purely real grid points occur when an odd number of
        applications of some of the basic symmetry inversions results in
        the identity transformation (as each inversion comes with a
        complex conjugation, having an odd number of these bringing a
        grid point back to itself implies that it should equal its own
        complex conjugate and hence be real).
        As an example, consider bulk inversion followed by x Nyquist
        plane inversion followed by y Nyquist plane inversion,
        on the grid point [ki = ±nyquist, kj = ±nyquist, kk = m]:
            grid[±nyquist, ±nyquist, +m]
          → grid[∓nyquist, ∓nyquist, -m]*  (bulk inversion)
          → grid[∓nyquist, ±nyquist, +m]   (x Nyquist plane inversion)
          → grid[±nyquist, ±nyquist, -m]*  (y Nyquist plane inversion)
        from which we see that [ki = ±nyquist, kj = ±nyquist, kk = m]
        should be real for m = 0.
        Similarly, [ki = ±nyquist, kj = ±nyquist, kk = ±nyquist] can be
        found to be real by applying all three Nyquist plane inversions.
        More trivially, applying just bulk inversions leaves [0, 0, 0]
        invariant, and applying just e.g. the x Nyquist plane inversions
        leaves [±nyquist, 0, 0] invariant. In total, all possible
        triplets [ki, kj, kk] with ki, kj, kk ∈  {0, ±nyquist} should
        be purely real. This corresponds to the centre or the bulk
        (the origin), the centre of each of the 6 Nyquist faces,
        the centre of each of the 12 Nyquist edges
        and the 8 Nyquist corners.
        """
        return set(itertools.product(*[(0, +nyquist, -nyquist)]*3))
    purely_reals = get_purely_reals(nyquist)
    # Generator for looping over the grid
    def visit(nyquist=nyquist, sparse=False):
        """When sparse is False, all points within |nyquist| will be
        visited except kk < 0. This includes points with ki or kj equal
        to +nyquist. When sparse is True, we only visit points that are
        tabulated (i.e. we exclude ki and kj equal to +nyquist), and we
        also only visits one grid point from each conjugate grid point
        pair. To perform this sparse iteration, note that all conjugate
        pairs with both members in the tabulated region are situated
        either in the positive z Nyquist plane (kk == nyquist) or on the
        z DC plane (kk == 0). Both of these planes are symmetric with
        respect to inversion through their centre (and complex
        conjugation), with the DC plane inheriting this symmetry from
        the bulk inversion symmetry. To only hit half of the points in
        these planes, we skip points with ki > 0. Furthermore, when
        ki == 0, we skip points with kj > 0, which reduces the remaining
        four symmetric lines to half lines. To better understand these
        four lines, consider:
          z DC plane:
            - grid[       0, kj, 0] == grid[       0, -kj, 0]* (bulk inversion)
            - grid[-nyquist, kj, 0] == grid[-nyquist, -kj, 0]* (x Nyquist plane inversion)
          z Nyquist plane:
            - grid[       0, kj, nyquist] == grid[       0, -kj, nyquist]* (z Nyquist plane inversion)
            - grid[-nyquist, kj, nyquist] == grid[-nyquist, -kj, nyquist]* (all three above inversions together)
        As a bonus, this sparse sampling allows us to calculate the
        number of independent grid points. Defining
          n_bulk      = (gridsize//2 + 1)*gridsize*gridsize
          n_halfplane = (gridsize//2 - 1)*gridsize
          n_halfline   = (gridsize//2 - 1)
        as the number of grid points in the bulk, in one of the half
        planes that we skip, in one of the half lines that we skip,
        respectively, the total number comes out to be
          n_independent = n_bulk - 2*(n_halfplane + 2*n_halfline)
                        = gridsize**3//2 + 4
        """
        if not sparse:
            for         ki in range(-nyquist, nyquist + 1):
                for     kj in range(-nyquist, nyquist + 1):
                    for kk in range(       0, nyquist + 1):
                        yield ki, kj, kk
        else:
            for         ki in range(-nyquist, nyquist    ):
                for     kj in range(-nyquist, nyquist    ):
                    for kk in range(       0, nyquist + 1):
                        if kk == 0 or kk == nyquist:
                            if ki > 0:
                                continue
                            if (ki == 0 or ki == -nyquist) and kj > 0:
                                continue
                        yield ki, kj, kk
    # Function for checking that [ki, kj, kk] is within
    # the tabulated region.
    def is_within_tabulation(ki, kj, kk):
        if not (-nyquist_large <= ki < nyquist_large):
            return False
        if not (-nyquist_large <= kj < nyquist_large):
            return False
        if not (0 <= kk <= nyquist_large):
            return False
        return True
    # Function for looking up a complex grid point
    def lookup(ki, kj, kk):
        i = ki + (-(ki < 0) & gridsize_large)
        j = kj + (-(kj < 0) & gridsize_large)
        k = 2*kk
        return complex(*grid[i, j, k:k+2])
    # Function checking the reality condition
    # which applies for certain grid points.
    def check_reality(ki, kj, kk):
        """A return value of True signals that
        a non-zero imaginary part was found.
        """
        if not is_within_tabulation(ki, kj, kk):
            return False
        # Skip if this grid point is not one of those
        # that should be purely real.
        if (ki, kj, kk) not in purely_reals:
            return False
        # Check for non-zero imaginary part
        c = lookup(ki, kj, kk)
        if np.abs(c.imag) > abs_tol:
            name = {
                0: 'centre of bulk (origin)',
                1: 'centre of face',
                2: 'centre of edge',
                3: 'corner',
            }[(np.abs(ki) == nyquist) + (np.abs(kj) == nyquist) + (np.abs(kk) == nyquist)]
            fancyprint(
                f'Should be real ({name}): [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                wrap=False,
                file=sys.stderr,
            )
            return True
        return False
    # Function checking for existence of zeros inside embedded grid
    def check_zero_within_embedding(ki, kj, kk):
        """A return value of True signals that
        a zero has been found.
        """
        if not is_within_tabulation(ki, kj, kk):
            return False
        if nullified_nyquist and (nyquist in np.abs((ki, kj, kk))):
            return False
        if nullified_origin and (ki, kj, kk) == (0, 0, 0):
            return False
        # Check for zero
        c = lookup(ki, kj, kk)
        found_zero = False
        should_be_real = ((ki, kj, kk) in purely_reals)
        if should_be_real:
            if c.real == 0:
                found_zero = True
                fancyprint(
                    f'Found zero (should be real): [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                    wrap=False,
                    file=sys.stderr,
                )
        else:
            if c.real == 0 or c.imag == 0:
                found_zero = True
                fancyprint(
                    f'Found zero: [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                    wrap=False,
                    file=sys.stderr,
                )
        return found_zero
    # Function checking for existence of non-zeros outside embedded grid
    def check_nonzero_beyond_embedding(ki, kj, kk):
        """A return value of True signals that
        a non-zero has been found.
        """
        if not is_within_tabulation(ki, kj, kk):
            return False
        # Skip if [ki, kj, kk] is within the embedding
        if np.abs(ki) <= nyquist or np.abs(kj) <= nyquist or np.abs(kk) <= nyquist:
            return False
        # Check for non-zero (exact)
        c = lookup(ki, kj, kk)
        if c != 0:
            fancyprint(
                f'Found non-zero outside embedding: [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                wrap=False,
                file=sys.stderr,
            )
            return True
        return False
    # Function checking for existence of non-zeros at Nyquist planes
    def check_nonzero_at_nyquist(ki, kj, kk):
        """A return value of True signals that
        a non-zero has been found.
        """
        if not is_within_tabulation(ki, kj, kk):
            return False
        # Skip non-Nyquist points
        if nyquist not in np.abs((ki, kj, kk)):
            return False
        # Check for non-zero (exact)
        c = lookup(ki, kj, kk)
        if c != 0:
            fancyprint(
                f'Found non-zero at Nyquist plane: [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                wrap=False,
                file=sys.stderr,
            )
            return True
        return False
    # Function checking for existence of non-zeros at the origin
    def check_nonzero_at_origin(ki, kj, kk):
        """A return value of True signals that
        a non-zero has been found.
        """
        # Skip non-origin points
        if (ki, kj, kk) != (0, 0, 0):
            return False
        # Check for non-zero (exact)
        c = lookup(ki, kj, kk)
        if c != 0:
            fancyprint(
                f'Found non-zero at origin: [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                wrap=False,
                file=sys.stderr,
            )
            return True
        return False
    # Function factory for generating inversion functions
    # capable of checking conjugate symmetries.
    def generate_check(accept, invert, name):
        def check(ki, kj, kk, *, visited=set()):
            """A return value of True signals broken symmetry"""
            # Skip if [ki, kj, kk] not within region of interest
            if not accept(ki, kj, kk):
                return False
            # Skip if [ki, kj, kk] not within tabulated region
            if not is_within_tabulation(ki, kj, kk):
                return False
            # Skip if inverted [ki, kj, kk] not within tabulated region
            ki_inv, kj_inv, kk_inv = invert(ki, kj, kk)
            if not is_within_tabulation(ki_inv, kj_inv, kk_inv):
                return False
            # Skip if [ki, kj, kk] has already been visited.
            # If not, add [ki, kj, kk] and its inversion
            # as visited sites.
            if (ki, kj, kk) in visited:
                return False
            visited.add((ki,     kj,     kk     ))
            visited.add((ki_inv, kj_inv, kk_inv))
            # If [ki, kj, kk] and its inversion are one and the same
            # point, the conjugate symmetry implies that this point
            # should be real. We could also just look up (ki, kj, kk)
            # in the purely_reals set.
            should_be_real = ((ki, kj, kk) == (ki_inv, kj_inv, kk_inv))
            # Check symmetry
            c     = lookup(ki,     kj,     kk    )
            c_inv = lookup(ki_inv, kj_inv, kk_inv)
            symmetry_broken = False
            if not (
                    np.isclose(c.real,  c_inv.real, rel_tol, abs_tol)
                and np.isclose(c.imag, -c_inv.imag, rel_tol, abs_tol)
            ):
                symmetry_broken = True
                if should_be_real:
                    # This grid point has a non-zero imaginary part
                    # while it ought to be purely real. Though this
                    # is counted as an error, we do not print anything,
                    # as a separate reality check will also
                    # be carried out.
                    pass
                else:
                    fancyprint(
                        f'Should be conjugate pair ({name}): '
                        f'[ki = {ki}, kj = {kj}, kk = {kk}] → {c} vs. '
                        f'[ki = {ki_inv}, kj = {kj_inv}, kk = {kk_inv}] → {c_inv}',
                        wrap=False,
                        file=sys.stderr,
                    )
            return symmetry_broken
        return check
    checks = []
    # Symmetry of inverting everything through the origin.
    # Tack on checking of zeros.
    checks.append(
        generate_check(
            lambda ki, kj, kk: True,
            lambda ki, kj, kk: (-ki, -kj, -kk),
            'inversion through origin',
        )
    )
    # Symmetry of inverting the negative x Nyquist plane
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (ki == -nyquist),
            lambda ki, kj, kk: (+ki, -kj, -kk),
            'inversion through centre of ki = -nyquist plane',
        )
    )
    # Symmetry of inverting the negative y Nyquist plane
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kj == -nyquist),
            lambda ki, kj, kk: (-ki, +kj, -kk),
            'inversion through centre of kj = -nyquist plane',
        )
    )
    # Symmetry of inverting the positive z Nyquist plane
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kk == +nyquist),
            lambda ki, kj, kk: (-ki, -kj, +kk),
            'inversion through centre of kk = +nyquist plane',
        )
    )
    # Symmetry of inverting the negative x, positive z Nyquist edge
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (ki == -nyquist and kk == +nyquist),
            lambda ki, kj, kk: (+ki, -kj, +kk),
            'inversion through centre of ki = -nyquist, kk = +nyquist edge',
        )
    )
    # Symmetry of inverting the negative y, positive z Nyquist edge
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kj == -nyquist and kk == +nyquist),
            lambda ki, kj, kk: (-ki, +kj, +kk),
            'inversion through centre of kj = -nyquist, kk = +nyquist edge',
        )
    )
    # Reality of special points
    checks.append(check_reality)
    # Check symmetries throughout the small grid
    symmetry_broken = False
    for check in checks:
        for ki, kj, kk in visit():
            symmetry_broken |= check(ki, kj, kk)
    # Check for zeros in the bulk, if requested
    if not allow_zeros:
        for ki, kj, kk in visit():
            symmetry_broken |= check_zero_within_embedding(ki, kj, kk)
    # Check that non-tabulated modes are nullified, if requested
    if pure_embedding:
        for ki, kj, kk in visit(nyquist_large):
            symmetry_broken |= check_nonzero_beyond_embedding(ki, kj, kk)
    # Check that the Nyquist planes are nullified, if requested
    if nullified_nyquist:
        for ki, kj, kk in visit(nyquist_large):
            symmetry_broken |= check_nonzero_at_nyquist(ki, kj, kk)
    # Check that the origin is nullified, if requested
    if nullified_origin:
        for ki, kj, kk in [(0, 0, 0)]:
            symmetry_broken |= check_nonzero_at_origin(ki, kj, kk)
    # Tally up the total amount of information and compare
    # against the expected value, if requested.
    if count_information:
        # Count up the number of different real numbers within the grid,
        # including both the real and imaginary part
        # and disregarding signs.
        decimals = int(round(-log10(rel_tol)))
        information_full = len(set(np.round(
              list(np.abs(grid[:, :, 0::2].flatten()))
            + list(np.abs(grid[:, :, 1::2].flatten())),
            decimals,
        )))
        if gridsize_large > gridsize:
            if not pure_embedding:
                # The grid is embedded within a larger non-zero grid,
                # meaning that the above count contains lots of points
                # outside of the small grid. Discard the count.
                information_full = -1
        # We also iterate over the grid in a way that samples each
        # unique complex grid point once, recording the real values
        # (from both the real and imaginary parts).
        information = set()
        for ki, kj, kk in visit(sparse=True):
            c = lookup(ki, kj, kk)
            information.add(np.round(np.abs(c.real), decimals))
            information.add(np.round(np.abs(c.imag), decimals))
        information = len(information)
        # A general real, cubic grid consists of gridsize**3
        # unique real numbers, and so we expect the same number to
        # exist in the Fourier transformed grid. As some complex grid
        # points must be real, the number 0 is guaranteed to exist
        # in the Fourier grid. Disregarding the off change that some
        # complex mode is 0 (as in 0 + 0j), this ups the information
        # count by 1.
        # If we have nullified Nyquist planes, we need to count more
        # careful. Ignoring the Nyquist planes as well as the z DC
        # plane leaves us with points which do not have a conjugate
        # partner within the tabulation. This block of points has
        # gridsize - 1 points along the x and y directions and
        # nyquist - 1 points along the z direction. That this, it has
        #   block = 2*(nyquist - 1)*(gridsize - 1)**2
        # unique real numbers. The remaining z DC plane consists purely
        # of conjugate pairs. That is, it has (gridsize - 1)**2 complex
        # grid points, exactly half of which are redundant. The
        # exception is the origin, which we should count as contributing
        # two real numbers (though one of them is a zero). That is, the
        # z DC plane contains
        #   plane = (gridsize - 1)**2 + 1
        # unique real numbers. In total, we have
        #   block + plane = gridsize**3 + 3*gridsize*(1 - gridsize)
        # unique real numbers.
        if nullified_nyquist:
            information_expected = gridsize**3 + 3*gridsize*(1 - gridsize)
            nullified_origin_str = ''
            if nullified_origin:
                information_expected -= 1
                nullified_origin_str = '- 1 '
            information_expected_str = (
                f'{gridsize}³ + 3*{gridsize}(1 - {gridsize}) '
                f'{nullified_origin_str}= {information_expected}'
            )
        else:
            information_expected = gridsize**3 + 1
            nullified_origin_str = '+ 1 '
            if nullified_origin:
                information_expected -= 1
                nullified_origin_str = ''
            information_expected_str = f'{gridsize}³ {nullified_origin_str}= {information_expected}'
        if len({information_full, information, information_expected} - {-1}) != 1:
            symmetry_broken = True
            msg_information_full = ''
            if information_full != -1:
                msg_information_full = f'{information_full} (counted over entire grid), '
            fancyprint(
                f'Dispute about number of unique absolute values '
                f'of real and imaginary parts of grid points: '
                f'{msg_information_full}'
                f'{information} (found through sparse iteration), '
                f'while we expect {information_expected_str} for a random grid.',
                wrap=False,
                file=sys.stderr,
            )
        # As a further check, we want to tally up the number of unique
        # complex number in the grid, with complex conjugate pairs
        # deemed non-unique. Count up number of unique complex number
        # in the entire grid.
        n_unique_full = len(set(np.round(
            np.sqrt((grid[:, :, 0::2]**2 + grid[:, :, 1::2]**2).flatten()),
            decimals,
        )))
        if gridsize_large > gridsize:
            if pure_embedding:
                # The grid is embedded purely, i.e. the grid contains
                # zeros outside of the Nyquist frequencies. These all
                # count as an additional unique complex grid point.
                n_unique_full -= 1
            else:
                # The grid is embedded within a larger non-zero grid,
                # meaning that the above count contains lots of points
                # outside of the small grid. Discard the count.
                n_unique_full = -1
        # We also iterate over the grid in a way that samples each
        # unique complex grid point once, recording the absolute values.
        absolutes = set()
        for ki, kj, kk in visit(sparse=True):
            absolutes.add(np.abs(lookup(ki, kj, kk)))
        n_unique = len(absolutes)
        # The Fourier transform of a 3D random real grid has exactly
        # gridsize**3//2 + 4 complex grid points with unique absolute
        # values (see the docstring of visit() for details).
        # If the Nyquist planes are nullified, the z DC plane contains
        # ((gridsize - 1)**2 - 1)//2 unique complex points, plus the
        # origin. The remaining block contains
        # (nyquist - 1)*(gridsize - 1)**2 unique points.
        # In total we have
        #    (gridsize**3 + 3*gridsize*(1 - gridsize))//2
        # unique absolute values.
        if nullified_nyquist or nullified_origin:
            n_unique_full -= 1  # ignore zeros at Nyquist points and/or origin
            n_unique      -= 1  # ignore zeros at Nyquist points and/or origin
        if nullified_nyquist:
            n_unique_expected = (gridsize**3 + 3*gridsize*(1 - gridsize))//2
            n_unique_expected_str = f'({gridsize}³ + 3*{gridsize}(1 - {gridsize}))//2 '
        else:
            n_unique_expected = gridsize**3//2 + 4
            n_unique_expected_str = f'{gridsize}³//2 + 4 '
        if nullified_origin:
            n_unique_expected -= 1
            n_unique_expected_str += '- 1 '
        n_unique_expected_str += f'= {n_unique_expected}'
        if len({n_unique_full, n_unique, n_unique_expected} - {-1}) != 1:
            symmetry_broken = True
            msg_n_unique_full = ''
            if n_unique_full != -1:
                msg_n_unique_full = f'{n_unique_full} (counted over entire grid), '
            fancyprint(
                f'Dispute about number of unique absolute values of complex grid points: '
                f'{msg_n_unique_full}'
                f'{n_unique} (found through sparse iteration), '
                f'while we expect {n_unique_expected} for a random grid.',
                wrap=False,
                file=sys.stderr,
            )
    masterprint('done')
    return not symmetry_broken

# Function for differentiating domain grids
@cython.pheader(
    # Arguments
    grid='double[:, :, ::1]',
    dim='int',
    order='int',
    Δx='double',
    buffer_or_buffer_name=object,  # double[:, :, ::1] or int or str
    direction=str,
    do_ghost_communication='bint',
    # Locals
    grid_mv='double[::1]',
    grid_ptr_lower_1='double*',
    grid_ptr_lower_2='double*',
    grid_ptr_lower_3='double*',
    grid_ptr_lower_4='double*',
    grid_ptr_upper_1='double*',
    grid_ptr_upper_2='double*',
    grid_ptr_upper_3='double*',
    grid_ptr_upper_4='double*',
    index='Py_ssize_t',
    index_i='Py_ssize_t',
    index_i_end='Py_ssize_t',
    index_j='Py_ssize_t',
    index_j_end='Py_ssize_t',
    offset_1='Py_ssize_t',
    offset_2='Py_ssize_t',
    offset_3='Py_ssize_t',
    offset_4='Py_ssize_t',
    offset_ghosts='Py_ssize_t',
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    ᐁgrid_dim='double[:, :, ::1]',
    ᐁgrid_dim_mv='double[::1]',
    ᐁgrid_dim_ptr='double*',
    returns='double[:, :, ::1]',
)
def diff_domaingrid(
    grid, dim, order,
    Δx=1, buffer_or_buffer_name=0, direction='forward', do_ghost_communication=True,
):
    """This function differentiates a given domain grid along the dim'th
    dimension once through finite differencing. The passed grid must
    include correctly populated ghost points, and the returned grid will
    contain correctly populated ghost points as well unless
    do_ghost_communication is False.
    The order argument specifies the order of the differentiation,
    meaning the number of neighbouring grid points used to approximate
    the first derivative. For odd orders, the differentiation cannot be
    symmetric, and so the direction should be specified as either
    'forward' or 'backward'.
    To achieve proper units, the physical grid spacing may be specified
    as Δx. If not given, grid units (Δx == 1) are used. The
    buffer_or_buffer_name argument can be a (contiguous) buffer to store
    the results, or alternatively the name of a buffer to use. In either
    case, existing values will be overwritten.
    Note that a grid cannot be differentiated in-place by passing the
    grid as both the first and fifth argument, as the differentiation
    of each grid point requires information from the original
    (non-differentiated) grid.
    """
    # Sanity checks on input
    if dim not in (0, 1, 2):
        abort(f'diff_domaingrid() called with dim = {dim} ∉ {{0, 1, 2}}')
    if order not in (1, 2, 4, 6, 8):
        abort(f'diff_domaingrid() called with order = {order} ∉ {{1, 2, 4, 6, 8}}')
    if order == 1 and direction not in ('forward', 'backward'):
        abort(f'diff_domaingrid() called with direction = {direction} ∉ {{"forward", "backward"}}')
    # If no buffer is supplied, fetch the buffer with the name
    # given by buffer_or_buffer_name.
    if isinstance(buffer_or_buffer_name, (int, np.integer, str)):
        ᐁgrid_dim = get_buffer(asarray(grid).shape, buffer_or_buffer_name, nullify=False)
    else:
        ᐁgrid_dim = buffer_or_buffer_name
        if asarray(ᐁgrid_dim).shape != asarray(grid).shape:
            abort(
                f'diff_domaingrid() called with grid of shape {asarray(grid).shape}'
                f'and buffer of different shape {asarray(ᐁgrid_dim).shape}'
            )
    # Extract pointer of ᐁgrid_dim, offset to take care of ghost points
    size_i, size_j, size_k = asarray(grid).shape
    offset_ghosts = nghosts*((size_j + 1)*size_k + 1)
    ᐁgrid_dim_mv = asarray(ᐁgrid_dim).ravel()
    ᐁgrid_dim_ptr = cython.address(ᐁgrid_dim_mv[offset_ghosts:])
    # Extract pointers of grid, offset to take care of ghost points and
    # steps along direction dim.
    grid_mv = asarray(grid).ravel()
    offset_1 = (1*ℤ[(dim == 0)*size_j] + 1*ℤ[dim == 1])*size_k + 1*ℤ[dim == 2]
    offset_2 = (2*ℤ[(dim == 0)*size_j] + 2*ℤ[dim == 1])*size_k + 2*ℤ[dim == 2]
    offset_3 = (3*ℤ[(dim == 0)*size_j] + 3*ℤ[dim == 1])*size_k + 3*ℤ[dim == 2]
    offset_4 = (4*ℤ[(dim == 0)*size_j] + 4*ℤ[dim == 1])*size_k + 4*ℤ[dim == 2]
    if order == 1:
        # Order 1 (odd)
        grid_ptr_upper_1 = cython.address(
            grid_mv[(offset_ghosts + offset_1*ℤ[direction == 'forward']):]
        )
        grid_ptr_lower_1 = cython.address(
            grid_mv[(offset_ghosts - offset_1*ℤ[direction == 'backward']):]
        )
    else:
        # Even orders
        grid_ptr_upper_1 = cython.address(grid_mv[(offset_ghosts + offset_1):])
        grid_ptr_lower_1 = cython.address(grid_mv[(offset_ghosts - offset_1):])
        if order >= 4:
            grid_ptr_upper_2 = cython.address(grid_mv[(offset_ghosts + offset_2):])
            grid_ptr_lower_2 = cython.address(grid_mv[(offset_ghosts - offset_2):])
            if order >= 6:
                grid_ptr_upper_3 = cython.address(grid_mv[(offset_ghosts + offset_3):])
                grid_ptr_lower_3 = cython.address(grid_mv[(offset_ghosts - offset_3):])
                if order >= 8:
                    grid_ptr_upper_4 = cython.address(grid_mv[(offset_ghosts + offset_4):])
                    grid_ptr_lower_4 = cython.address(grid_mv[(offset_ghosts - offset_4):])
    # Loop over the 3D grid. Cython does not produce optimal code when
    # using a stepped range, so we write out the looping by hand.
    index_i = 0
    index_i_end = (size_i - ℤ[2*nghosts + 1])*ℤ[size_j*size_k]
    while True:
        index_j     = index_i
        index_j_end = index_i + ℤ[(size_j - ℤ[2*nghosts + 1])*size_k]
        while True:
            index     = index_j
            index_end = index_j + ℤ[size_k - ℤ[2*nghosts + 1]]
            while True:
                with unswitch:
                    if order == 1:
                        ᐁgrid_dim_ptr[index] = ℝ[1/Δx]*(
                            + grid_ptr_upper_1[index]
                            - grid_ptr_lower_1[index]
                        )
                    elif order == 2:
                        ᐁgrid_dim_ptr[index] = ℝ[(1/2)/Δx]*(
                            + grid_ptr_upper_1[index]
                            - grid_ptr_lower_1[index]
                        )
                    elif order == 4:
                        ᐁgrid_dim_ptr[index] = (
                            + ℝ[(2/3)/Δx]*(
                                + grid_ptr_upper_1[index]
                                - grid_ptr_lower_1[index]
                            )
                            - ℝ[(1/12)/Δx]*(
                                + grid_ptr_upper_2[index]
                                - grid_ptr_lower_2[index]
                            )
                        )
                    elif order == 6:
                        ᐁgrid_dim_ptr[index] = (
                            ℝ[(3/4)/Δx]*(
                                + grid_ptr_upper_1[index]
                                - grid_ptr_lower_1[index]
                            )
                            - ℝ[(3/20)/Δx]*(
                                + grid_ptr_upper_2[index]
                                - grid_ptr_lower_2[index]
                            )
                            + ℝ[(1/60)/Δx]*(
                                + grid_ptr_upper_3[index]
                                - grid_ptr_lower_3[index]
                            )
                        )
                    else:  # order == 8
                        ᐁgrid_dim_ptr[index] = (
                            ℝ[(4/5)/Δx]*(
                                + grid_ptr_upper_1[index]
                                - grid_ptr_lower_1[index]
                            )
                            - ℝ[(1/5)/Δx]*(
                                + grid_ptr_upper_2[index]
                                - grid_ptr_lower_2[index]
                            )
                            + ℝ[(4/105)/Δx]*(
                                + grid_ptr_upper_3[index]
                                - grid_ptr_lower_3[index]
                            )
                            - ℝ[(1/280)/Δx]*(
                                + grid_ptr_upper_4[index]
                                - grid_ptr_lower_4[index]
                            )
                        )
                # Breakouts and loop counter incrementations
                if index == index_end:
                    break
                index += 1
            if index_j == index_j_end:
                break
            index_j += size_k
        if index_i == index_i_end:
            break
        index_i += ℤ[size_j*size_k]
    # Populate the ghost points with copies of their
    # corresponding actual points.
    if do_ghost_communication:
        communicate_ghosts(ᐁgrid_dim, '=')
    return ᐁgrid_dim

# Below we define iterators for looping over grid cells within particle
# interpolation regions. Each function takes in coordinates which must
# be pre-scaled so that they lie in the interval
#   nghosts - ½ < {x, y z} < shape[dim] - nghosts - ½
# where shape is the full shape (with ghost layers) of the grid taking
# part in the interpolation. This j (y) and k (z) size of the grid must
# also be given. The iterators return the linear index into the grid as
# well as the weight of the given grid point. If the weight should be
# applied to (multiplied by) some number, you can pass this number as an
# optional argument and specify apply_factor=True.
#
# Nearest grid point (NGP) interpolation (order 1)
@cython.iterator(
    depends=(
        # Global variables used by particle_interpolation_loop_NGP()
        'weights_x',
        'weights_y',
        'weights_z',
    ),
)
def particle_interpolation_loop_NGP(
    x, y, z, size_j, size_k, multiplier=1,
    *,
    apply_factor=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        x='double',
        y='double',
        z='double',
        size_j='Py_ssize_t',
        size_k='Py_ssize_t',
        multiplier='double',
        apply_factor='bint',
        # Locals
        _i='Py_ssize_t',
        _index_weights_i='Py_ssize_t',
        _index_weights_j='Py_ssize_t',
        _index_weights_k='Py_ssize_t',
        # Yielded
        _index='Py_ssize_t',
        weight='double',
    )
    # Set interpolation weights and get grid indices
    _index_weights_i = set_weights_NGP(x, weights_x)
    _index_weights_j = set_weights_NGP(y, weights_y)
    _index_weights_k = set_weights_NGP(z, weights_z)
    # "Iterate" over the single interpolation point,
    # yielding the grid index and associated weight.
    for _i in range(1):
        with unswitch(1):
            if apply_factor:
                weight = multiplier
            else:
                weight = 1
        _index = (_index_weights_i*size_j + _index_weights_j)*size_k + _index_weights_k
        yield _index, weight
# Cloud-in-cell (CIC) interpolation (order 2)
@cython.iterator(
    depends=(
        # Global variables used by particle_interpolation_loop_CIC()
        'weights_x',
        'weights_y',
        'weights_z',
    ),
)
def particle_interpolation_loop_CIC(
    x, y, z, size_j, size_k, multiplier=1,
    *,
    apply_factor=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        x='double',
        y='double',
        z='double',
        size_j='Py_ssize_t',
        size_k='Py_ssize_t',
        multiplier='double',
        apply_factor='bint',
        # Locals
        _i='Py_ssize_t',
        _index_i='Py_ssize_t',
        _index_j='Py_ssize_t',
        _index_weights_i='Py_ssize_t',
        _index_weights_j='Py_ssize_t',
        _index_weights_k='Py_ssize_t',
        _j='Py_ssize_t',
        _k='Py_ssize_t',
        _weight_i='double',
        # Yielded
        _index='Py_ssize_t',
        weight='double',
    )
    # Set interpolation weights and get grid indices
    _index_weights_i = set_weights_CIC(x, weights_x)
    _index_weights_j = set_weights_CIC(y, weights_y)
    _index_weights_k = set_weights_CIC(z, weights_z)
    # Iterate efficiently over the interpolation region,
    # yielding the grid index and associated weight.
    _index_i = (
        ((_index_weights_i - 1)*size_j + (_index_weights_j - 1))*size_k
        + _index_weights_k - 1
    )
    for _i in range(2):
        _weight_i = weights_x[_i]
        with unswitch(1):
            if apply_factor:
                _weight_i *= multiplier
        _index_i += ℤ[size_j*size_k]
        _index_j = _index_i
        for _j in range(2):
            _index_j += size_k
            _index = _index_j
            for _k in range(2):
                _index += 1
                weight = ℝ[_weight_i*weights_y[_j]]*weights_z[_k]
                yield _index, weight
# Triangular-shaped cloud (TSC) interpolation (order 3)
@cython.iterator(
    depends=(
        # Global variables used by particle_interpolation_loop_TSC()
        'weights_x',
        'weights_y',
        'weights_z',
    ),
)
def particle_interpolation_loop_TSC(
    x, y, z, size_j, size_k, multiplier=1,
    *,
    apply_factor=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        x='double',
        y='double',
        z='double',
        size_j='Py_ssize_t',
        size_k='Py_ssize_t',
        multiplier='double',
        apply_factor='bint',
        # Locals
        _i='Py_ssize_t',
        _index_i='Py_ssize_t',
        _index_j='Py_ssize_t',
        _index_weights_i='Py_ssize_t',
        _index_weights_j='Py_ssize_t',
        _index_weights_k='Py_ssize_t',
        _j='Py_ssize_t',
        _k='Py_ssize_t',
        _weight_i='double',
        # Yielded
        _index='Py_ssize_t',
        weight='double',
    )
    # Set interpolation weights and get grid indices
    _index_weights_i = set_weights_TSC(x, weights_x)
    _index_weights_j = set_weights_TSC(y, weights_y)
    _index_weights_k = set_weights_TSC(z, weights_z)
    # Iterate efficiently over the interpolation region,
    # yielding the grid index and associated weight.
    _index_i = (
        ((_index_weights_i - 1)*size_j + (_index_weights_j - 1))*size_k
        + _index_weights_k - 1
    )
    for _i in range(3):
        _weight_i = weights_x[_i]
        with unswitch(1):
            if apply_factor:
                _weight_i *= multiplier
        _index_i += ℤ[size_j*size_k]
        _index_j = _index_i
        for _j in range(3):
            _index_j += size_k
            _index = _index_j
            for _k in range(3):
                _index += 1
                weight = ℝ[_weight_i*weights_y[_j]]*weights_z[_k]
                yield _index, weight
# Piecewise cubic spline (PCS) interpolation (order 4)
@cython.iterator(
    depends=(
        # Global variables used by particle_interpolation_loop_PCS()
        'weights_x',
        'weights_y',
        'weights_z',
    ),
)
def particle_interpolation_loop_PCS(
    x, y, z, size_j, size_k, multiplier=1,
    *,
    apply_factor=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        x='double',
        y='double',
        z='double',
        size_j='Py_ssize_t',
        size_k='Py_ssize_t',
        multiplier='double',
        apply_factor='bint',
        # Locals
        _i='Py_ssize_t',
        _index_i='Py_ssize_t',
        _index_j='Py_ssize_t',
        _index_weights_i='Py_ssize_t',
        _index_weights_j='Py_ssize_t',
        _index_weights_k='Py_ssize_t',
        _j='Py_ssize_t',
        _k='Py_ssize_t',
        _weight_i='double',
        # Yielded
        _index='Py_ssize_t',
        weight='double',
    )
    # Set interpolation weights and get grid indices
    _index_weights_i = set_weights_PCS(x, weights_x)
    _index_weights_j = set_weights_PCS(y, weights_y)
    _index_weights_k = set_weights_PCS(z, weights_z)
    # Iterate efficiently over the interpolation region,
    # yielding the grid index and associated weight.
    _index_i = (
        ((_index_weights_i - 1)*size_j + (_index_weights_j - 1))*size_k
        + _index_weights_k - 1
    )
    for _i in range(4):
        _weight_i = weights_x[_i]
        with unswitch(1):
            if apply_factor:
                _weight_i *= multiplier
        _index_i += ℤ[size_j*size_k]
        _index_j = _index_i
        for _j in range(4):
            _index_j += size_k
            _index = _index_j
            for _k in range(4):
                _index += 1
                weight = ℝ[_weight_i*weights_y[_j]]*weights_z[_k]
                yield _index, weight

# Below we define weight functions for one-dimensional particle
# interpolation. Each function takes in a coordinate x which must be
# pre-scaled so that it lies in the interval
#   nghosts - ½ < x < shape[dim] - nghosts - ½
# where shape is the full shape (with ghost layers) of the grid taking
# part in the interpolation. The second argument is an array for storing
# the resulting weights.
# The return value is an index into the grid, such that the weight
# associated with that grid point is weights[0]. The full weighted sum
# is then weights[0]*grid[index] + weights[1]*grid[index + 1] + ...
#
# Nearest grid point (NGP) interpolation (order 1)
@cython.header(
    # Arguments
    x='double',
    weights='double*',
    # Locals
    index='Py_ssize_t',
    returns='Py_ssize_t',
)
def set_weights_NGP(x, weights):
    index = int(x + 0.5)
    weights[0] = 1
    return index
# Cloud-in-cell (CIC) interpolation (order 2)
@cython.header(
    # Arguments
    x='double',
    weights='double*',
    # Locals
    dist='double',
    index='Py_ssize_t',
    returns='Py_ssize_t',
)
def set_weights_CIC(x, weights):
    index = int(x)
    dist = x - index  # Distance between leftmost grid point and x; 0 <= dist < 1
    weights[0] = 1 - dist
    weights[1] = dist
    return index
# Triangular-shaped cloud (TSC) interpolation (order 3)
@cython.header(
    # Arguments
    x='double',
    weights='double*',
    # Locals
    dist='double',
    dist2='double',
    index='Py_ssize_t',
    weight0='double',
    weight1='double',
    returns='Py_ssize_t',
)
def set_weights_TSC(x, weights):
    index = int(x + 0.5)
    dist = x - index  # Distance between centre grid point and x; -0.5 <= dist < 0.5
    index -= 1
    dist2 = dist**2
    weight0 = 0.125 + 0.5*(dist2 - dist)
    weight1 = 0.75 - dist2
    weights[0] = weight0
    weights[1] = weight1
    weights[2] = 1 - weight0 - weight1
    return index
# Piecewise cubic spline (PCS) interpolation (order 4)
@cython.header(
    # Arguments
    x='double',
    weights='double*',
    # Locals
    dist='double',
    index='Py_ssize_t',
    tmp='double',
    tmp2='double',
    tmp3='double',
    weight0='double',
    weight2='double',
    weight3='double',
    returns='Py_ssize_t',
)
def set_weights_PCS(x, weights):
    index = int(x)
    index -= 1
    dist = x - index  # Distance between leftmost grid point and x; 1 <= dist < 2
    tmp = 2 - dist
    tmp2 = tmp**2
    tmp3 = tmp*tmp2
    weight0 = 1./6.*tmp3
    weight2 = 2./3. - tmp2 + 0.5*tmp3
    weight3 = 1./6.*(dist - 1)**3
    weights[0] = weight0
    weights[1] = 1 - weight0 - weight2 - weight3
    weights[2] = weight2
    weights[3] = weight3
    return index
# Allocate global weights arrays to be used with the above functions
cython.declare(
    highest_implemented_interpolation_order='int',
    weights_x='double*',
    weights_y='double*',
    weights_z='double*',
)
highest_interpolation_order_implemented = 4  # PCS
weights_x = malloc(highest_interpolation_order_implemented*sizeof('double'))
weights_y = malloc(highest_interpolation_order_implemented*sizeof('double'))
weights_z = malloc(highest_interpolation_order_implemented*sizeof('double'))



# Get local domain information
domain_info = get_domain_info()
cython.declare(
    domain_subdivisions='int[::1]',
    domain_layout_local_indices='int[::1]',
    domain_size_x='double',
    domain_size_y='double',
    domain_size_z='double',
    domain_bgn_x='double',
    domain_bgn_y='double',
    domain_bgn_z='double',
)
domain_subdivisions         = domain_info.subdivisions
domain_layout_local_indices = domain_info.layout_local_indices
domain_size_x               = domain_info.size_x
domain_size_y               = domain_info.size_y
domain_size_z               = domain_info.size_z
domain_bgn_x                = domain_info.bgn_x
domain_bgn_y                = domain_info.bgn_y
domain_bgn_z                = domain_info.bgn_z
