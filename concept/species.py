# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: you can redistribute it and/or modify
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
# The auther of CO𝘕CEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    from gravity import PP, PM, P3M
    from communication import exchange
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    from gravity cimport PP, PM, P3M
    from communication cimport exchange
    """


# The class representing a collection of particles of a given type
@cython.cclass
class Particles:
    """An instance of this class represents a collection of particles
    of a definite type. Only one instance of any Particles type may be
    instantiated in a run. A Particles instance of a given type should
    be present on all processes.
    All species share the same class (this one). The difference is
    purely in their "type" and "species" attributes. The "species"
    attribute is used as a flag to allow different species to behave
    differently.
    """

    # Initialization method
    @cython.header(N='Py_ssize_t')
    def __init__(self, N):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Particles type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        Py_ssize_t N
        Py_ssize_t N_local
        Py_ssize_t N_allocated
        double mass
        double softening
        str species
        str type
        double[::1] posx_mv
        double[::1] posy_mv
        double[::1] posz_mv
        double[::1] momx_mv
        double[::1] momy_mv
        double[::1] momz_mv
        double* posx
        double* posy
        double* posz
        double* momx
        double* momy
        double* momz
        """
        # Store particle meta data
        self.N = N
        self.N_allocated = 1
        self.N_local = 1
        self.mass = 1
        self.softening = 1
        self.species = 'generic species'
        self.type = 'generic particles'
        # Manually allocate memory for particle data
        self.posx = malloc(self.N_allocated*sizeof('double'))
        self.posy = malloc(self.N_allocated*sizeof('double'))
        self.posz = malloc(self.N_allocated*sizeof('double'))
        self.momx = malloc(self.N_allocated*sizeof('double'))
        self.momy = malloc(self.N_allocated*sizeof('double'))
        self.momz = malloc(self.N_allocated*sizeof('double'))
        # Memory views around the allocated data
        self.posx_mv = cast(self.posx, 'double[:self.N_allocated]')
        self.posy_mv = cast(self.posy, 'double[:self.N_allocated]')
        self.posz_mv = cast(self.posz, 'double[:self.N_allocated]')
        self.momx_mv = cast(self.momx, 'double[:self.N_allocated]')
        self.momy_mv = cast(self.momy, 'double[:self.N_allocated]')
        self.momz_mv = cast(self.momz, 'double[:self.N_allocated]')

    # This method populate the Particles pos/mom attributes with data.
    # It is deliberately designed so that you have to make a call for
    # each attribute. You should construct the mv array within the call
    # itself, as this will minimize memory usage.
    @cython.header(# Arguments
                   mv='double[::1]',
                   coord='str',
                   )
    def populate(self, mv, coord):
        self.N_allocated = mv.size
        self.N_local = self.N_allocated
        # Update the attribute corresponding to the passed string
        if coord == 'posx':
            self.posx = realloc(self.posx, self.N_allocated*sizeof('double'))
            self.posx_mv = cast(self.posx, 'double[:self.N_local]')
            self.posx_mv[...] = mv[...]
        elif coord == 'posy':
            self.posy = realloc(self.posy, self.N_allocated*sizeof('double'))
            self.posy_mv = cast(self.posy, 'double[:self.N_local]')
            self.posy_mv[...] = mv[...]
        elif coord == 'posz':
            self.posz = realloc(self.posz, self.N_allocated*sizeof('double'))
            self.posz_mv = cast(self.posz, 'double[:self.N_local]')
            self.posz_mv[...] = mv[...]
        elif coord == 'momx':
            self.momx = realloc(self.momx, self.N_allocated*sizeof('double'))
            self.momx_mv = cast(self.momx, 'double[:self.N_local]')
            self.momx_mv[...] = mv[...]
        elif coord == 'momy':
            self.momy = realloc(self.momy, self.N_allocated*sizeof('double'))
            self.momy_mv = cast(self.momy, 'double[:self.N_local]')
            self.momy_mv[...] = mv[...]
        elif coord == 'momz':
            self.momz = realloc(self.momz, self.N_allocated*sizeof('double'))
            self.momz_mv = cast(self.momz, 'double[:self.N_local]')
            self.momz_mv[...] = mv[...]
        elif master:
            raise ValueError('Wrong attribute name "{}"!'.format(coord))

    # This method will grow/shrink the data attributes.
    # Note that it will not update the N_local attribute.
    @cython.header(N_allocated='Py_ssize_t')
    def resize(self, N_allocated):
        if N_allocated != self.N_allocated:
            self.N_allocated = N_allocated
            # Reallocate data
            self.posx = realloc(self.posx, self.N_allocated*sizeof('double'))
            self.posy = realloc(self.posy, self.N_allocated*sizeof('double'))
            self.posz = realloc(self.posz, self.N_allocated*sizeof('double'))
            self.momx = realloc(self.momx, self.N_allocated*sizeof('double'))
            self.momy = realloc(self.momy, self.N_allocated*sizeof('double'))
            self.momz = realloc(self.momz, self.N_allocated*sizeof('double'))
            # Reassign memory views
            self.posx_mv = cast(self.posx, 'double[:self.N_allocated]')
            self.posy_mv = cast(self.posy, 'double[:self.N_allocated]')
            self.posz_mv = cast(self.posz, 'double[:self.N_allocated]')
            self.momx_mv = cast(self.momx, 'double[:self.N_allocated]')
            self.momy_mv = cast(self.momy, 'double[:self.N_allocated]')
            self.momz_mv = cast(self.momz, 'double[:self.N_allocated]')

    # Method for integrating particle positions forward in time
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
        masterprint('Drifting', self.type, '...')
        # Extracting variables
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
        # Some partiles may have drifted out of the local domain.
        # Exchange particles to the correct processes.
        masterprint('done')
        exchange(self)

    # Method for updating particle momenta
    @cython.header(# Arguments
                   Δt='double',
                   # Locals
                   kick_algorithm='str',
                   )
    def kick(self, Δt):
        """Note that the time step size Δt is really ∫_t^(t + Δt) dt/a.
        """
        kick_algorithm = kick_algorithms[self.species]
        masterprint('Kicking ({}) {} ...'.format(kick_algorithm,
                                                 self.type))
        # Delegate the work to the appropriate function based on species
        if kick_algorithm == 'PP':
            PP(self, Δt)
        elif kick_algorithm == 'PM':
            PM(self, Δt)
        elif kick_algorithm == 'P3M':
            P3M(self, Δt)
        elif master:
            raise ValueError(('Species "{}" has been assigned the kick'
                              + 'algorithm "{}", which is not implemented!'
                              ).format(self.species, kick_algorithm))
        masterprint('done')

    # This method is automaticlly called when a Particles instance
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


# Constructor function for Particles instances
@cython.header(# Argument
               particle_type='str',
               particle_species='str',
               mass='double',
               N='Py_ssize_t',
               # Locals
               particles='Particles',
               returns='Particles',
               )
def construct(particle_type, particle_species, mass, N):
    # Instantiate Particles instance
    particles = Particles(N)
    # Attach information to the particles
    particles.type = particle_type
    particles.species = particle_species
    particles.mass = mass
    if particle_species in softeningfactors:
        particles.softening = softeningfactors[particle_species]*boxsize/(N**ℝ[1/3])
    elif master:
        raise ValueError('Species "{}" do not have an assigned softening length!'
                         .format(particle_species))
    return particles
