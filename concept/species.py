# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from gravity import PP, PM, P3M')
cimport('from communication import exchange')



# The class governing any component of the universe
@cython.cclass
class Component:
    """An instance of this class represents either a collection of
    particles or a grid of fluid values. A Component instance should be
    present on all processes.
    """

    # Initialization method
    @cython.header(# Arguments
                   name='str',
                   species='str',
                   representation='str',
                   mass='double',
                   # Locals
                   shape='Py_ssize_t[::1]',
                   size='Py_ssize_t',
                   )
    def __init__(self, name, species, representation, mass):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Component type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        # General component attributes
        public str name
        public str species
        public str representation
        public double mass
        # Particle component attributes
        public Py_ssize_t N
        Py_ssize_t N_allocated
        Py_ssize_t N_local
        double softening
        # Particle data
        double* posx
        double* posy
        double* posz
        double* momx
        double* momy
        double* momz
        double[::1] posx_mv
        double[::1] posy_mv
        double[::1] posz_mv
        double[::1] momx_mv
        double[::1] momy_mv
        double[::1] momz_mv
        # Fluid component attributes
        public Py_ssize_t gridsize
        # Fluid data
        double* δ
        double* ux
        double* uy
        double* uz
        double[:, :, ::1] δ_mv
        double[:, :, ::1] ux_mv
        double[:, :, ::1] uy_mv
        double[:, :, ::1] uz_mv
        double[:, :, :] δ_noghosts
        double[:, :, :] ux_noghosts
        double[:, :, :] uy_noghosts
        double[:, :, :] uz_noghosts
        """
        # General component attributes
        self.name           = name
        self.species        = species
        self.representation = representation
        self.mass           = mass
        # Particle component attributes
        self.N           = 1
        self.N_allocated = 1
        self.N_local     = 1
        self.softening   = 1      
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
        # Fluid component attributes
        self.gridsize = 1
        # Fluid data
        shape = ones(3, dtype=C2np['Py_ssize_t'])
        size = np.prod(shape)
        self.δ  = malloc(size*sizeof('double'))
        self.ux = malloc(size*sizeof('double'))
        self.uy = malloc(size*sizeof('double'))
        self.uz = malloc(size*sizeof('double'))
        self.δ_mv  = cast(self.δ,  'double[:shape[0], :shape[1], :shape[2]]')
        self.ux_mv = cast(self.ux, 'double[:shape[0], :shape[1], :shape[2]]')
        self.uy_mv = cast(self.ux, 'double[:shape[0], :shape[1], :shape[2]]')
        self.uz_mv = cast(self.ux, 'double[:shape[0], :shape[1], :shape[2]]')
        self.δ_noghosts = self.δ_mv[:, :, :]
        self.ux_noghosts = self.ux_mv[:, :, :]
        self.uy_noghosts = self.uy_mv[:, :, :]
        self.uz_noghosts = self.uz_mv[:, :, :]

    # This method populate the Component pos/mom arrays (for a
    # particles representation) or the δ/u arrays (for a fluid
    # representation) with data. It is deliberately designed so that
    # you have to make a call for each attribute. You should construct
    # the data array within the call itself, as this will minimize
    # memory usage. This data array is 1D for particle data and 3D
    # for fluid data.
    @cython.header(# Arguments
                   data='object',  # 1D/3D (particles/fluid) memoryview
                   var='str',
                   # Locals
                   mv1D='double[::1]',
                   mv3D='double[:, :, ::1]',
                   )
    def populate(self, data, var):
        if self.representation == 'particles':
            mv1D = data
            self.N_local = mv1D.shape[0]
            # Enlarge data attributes if necessary
            if self.N_allocated < self.N_local:
                self.resize(self.N_local)
            # Update the data corresponding to the parsed string
            if var == 'posx':
                self.posx_mv[...] = mv1D[...]
            elif var == 'posy':
                self.posy_mv[...] = mv1D[...]
            elif var == 'posz':
                self.posz_mv[...] = mv1D[...]
            elif var == 'momx':
                self.momx_mv[...] = mv1D[...]
            elif var == 'momy':
                self.momy_mv[...] = mv1D[...]
            elif var == 'momz':
                self.momz_mv[...] = mv1D[...]
            elif master:
                abort('Wrong component attribute name "{}"!'.format(var))
        elif self.representation == 'fluid':
            mv3D = data
            # Reallocate data attributes if necessary
            self.resize(asarray(mv3D).shape)
            # Update the data corresponding to the parsed string
            if var == 'δ':
                self.δ_mv[:mv3D.shape[0],
                          :mv3D.shape[1],
                          :mv3D.shape[2]] = mv3D[...]
            elif var == 'ux':
                self.ux_mv[:mv3D.shape[0],
                           :mv3D.shape[1],
                           :mv3D.shape[2]] = mv3D[...]
            elif var == 'uy':
                self.uy_mv[:mv3D.shape[0],
                           :mv3D.shape[1],
                           :mv3D.shape[2]] = mv3D[...]
            elif var == 'uz':
                self.uz_mv[:mv3D.shape[0],
                           :mv3D.shape[1],
                           :mv3D.shape[2]] = mv3D[...]
            else:
                abort('Wrong component attribute name "{}"!'.format(var))

    # This method will grow/shrink the data attributes.
    # Note that it will not update the N_local attribute.
    @cython.header(# Arguments
                   shape='object',  # Py_ssize_t or tuple
                   # Locals
                   N_allocated='Py_ssize_t',
                   msg='str',
                   size='Py_ssize_t',
                   )
    def resize(self, shape):
        if self.representation == 'particles':
            if shape != self.N_allocated:
                self.N_allocated = shape
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
        elif self.representation == 'fluid':
            # The allocated shape of the fluid grids are 5 points
            # (one layer of pseudo points and two layers of ghost points
            # before and after the logical grid) longer than the logical
            # shape, in each direction.
            shape = tuple([s + 1 + 2*2 for s in shape])
            if (   shape[0] != self.δ_mv.shape[0]
                or shape[1] != self.δ_mv.shape[1]
                or shape[2] != self.δ_mv.shape[2]):
                if any([s - (1 + 2*2) < 1 for s in shape]):
                    msg = ('Attempted to resize fluid grids of the {} component\n'
                           'to a shape of {}. All dimensions must be > 5.').format(self.name,
                                                                                   shape)
                    abort(msg)
                # Reallocate fluid data
                size = np.prod(shape)
                self.δ  = realloc(self.δ,  size*sizeof('double'))
                self.ux = realloc(self.ux, size*sizeof('double'))
                self.uy = realloc(self.uy, size*sizeof('double'))
                self.uz = realloc(self.uz, size*sizeof('double'))
                # Reassign fluid data memory views
                self.δ_mv =  cast(self.δ,  'double[:shape[0], :shape[1], :shape[2]]')
                self.ux_mv = cast(self.ux, 'double[:shape[0], :shape[1], :shape[2]]')
                self.uy_mv = cast(self.uy, 'double[:shape[0], :shape[1], :shape[2]]')
                self.uz_mv = cast(self.uz, 'double[:shape[0], :shape[1], :shape[2]]')
                self.δ_noghosts = self.δ_mv[2:(self.δ_mv.shape[0] - 2),
                                            2:(self.δ_mv.shape[1] - 2),
                                            2:(self.δ_mv.shape[2] - 2)]
                self.ux_noghosts = self.ux_mv[2:(self.ux_mv.shape[0] - 2),
                                              2:(self.ux_mv.shape[1] - 2),
                                              2:(self.ux_mv.shape[2] - 2)]
                self.uy_noghosts = self.uy_mv[2:(self.uy_mv.shape[0] - 2),
                                              2:(self.uy_mv.shape[1] - 2),
                                              2:(self.uy_mv.shape[2] - 2)]
                self.uz_noghosts = self.uz_mv[2:(self.uz_mv.shape[0] - 2),
                                              2:(self.uz_mv.shape[1] - 2),
                                              2:(self.uz_mv.shape[2] - 2)]

    # Method for integrating particle positions/fluid values
    # forward in time.
    @cython.header(# Arguments
                   Δt='double',
                   # Locals
                   fac='double',
                   posx='double*',
                   posy='double*',
                   posz='double*',
                   momx='double*',
                   momy='double*',
                   momz='double*',
                   i='Py_ssize_t',
                   )
    def drift(self, Δt):
        """Note that the time step size
        Δt is really ∫_t^(t + Δt) dt/a**2.
        """
        masterprint('Drifting', self.name, '...')
        if self.representation == 'particles':
            # Particle drift
            posx = self.posx
            posy = self.posy
            posz = self.posz
            momx = self.momx
            momy = self.momy
            momz = self.momz
            # The factor 1/mass*∫_t^(t + Δt) dt/a**2
            fac = Δt/self.mass
            # Update positions
            for i in range(self.N_local):
                posx[i] += momx[i]*fac
                posy[i] += momy[i]*fac
                posz[i] += momz[i]*fac
                # Toroidal boundaries
                posx[i] = mod(posx[i], boxsize)
                posy[i] = mod(posy[i], boxsize)
                posz[i] = mod(posz[i], boxsize) 
        elif self.representation == 'fluid':
            # Fluid drift
            pass
        masterprint('done')
        # Some partiles may have drifted out of the local domain.
        # Exchange particles to the correct processes.
        exchange(self)

    # Method for updating particle momenta/fluid velocity
    @cython.header(# Arguments
                   Δt='double',
                   # Locals
                   kick_algorithm='str',
                   )
    def kick(self, Δt):
        """Note that the time step size Δt is really ∫_t^(t + Δt) dt/a.
        """
        kick_algorithm = kick_algorithms[self.species]
        masterprint('Kicking ({}) {} ...'.format(kick_algorithm, self.name))
        # Delegate the work to the appropriate function based on species
        if kick_algorithm == 'PP':
            PP(self, Δt)
        elif kick_algorithm == 'PM':
            PM(self, Δt)
        elif kick_algorithm == 'P3M':
            P3M(self, Δt)
        elif master:
            abort(('Species "{}" has been assigned the kick algorithm "{}", '
                   + 'which is not implemented!').format(self.species, kick_algorithm))
        masterprint('done')

    # This method is automaticlly called when a Component instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        if self.posx:
            free(self.posx)
        if self.posy:
            free(self.posy)
        if self.posz:
            free(self.posz)
        if self.momx:
            free(self.momx)
        if self.momy:
            free(self.momy)
        if self.momz:
            free(self.momz)
        if self.δ:
            free(self.δ)



# Constructor function for Component instances representing particles
@cython.header(# Argument
               name='str',
               species='str',
               N='Py_ssize_t',
               mass='double',
               # Locals
               component='Component',
               returns='Component',
               )
def construct_particles(name, species, N, mass):
    # Instantiate Component instance
    component = Component(name, species, 'particles', mass)
    # Attach particle attributes
    component.N = N
    if species in softeningfactors:
        component.softening = softeningfactors[species]*boxsize/(N**ℝ[1/3])
    elif master:
        abort('Species "{}" do not have an assigned softening length!'.format(species))
    return component

# Constructor function for Component instances representing a fluid
@cython.header(# Argument
               name='str',
               species='str',
               gridsize='Py_ssize_t',
               mass='double',
               # Locals
               component='Component',
               returns='Component',
               )
def construct_fluid(name, species, gridsize, mass):
    # Instantiate Component instance
    component = Component(name, species, 'fluid', mass)
    # Attach particle attributes
    component.gridsize = gridsize
    return component


