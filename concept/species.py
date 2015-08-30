# Copyright (C) 2015 Jeppe Mosgard Dakin
#
# This file is part of CONCEPT, the cosmological N-body code in Python
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.



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
    @cython.header(N='size_t')
    def __init__(self, N):
        # The triple quoted string below serves as the type declaration
        # for the Particles type. It will get picked up by the pyxpp
        # script and indluded in the .pxd file.
        """
        # Data attributes
        size_t N
        size_t N_local
        size_t N_allocated
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
        # Methods
        drift(self, double Δt)
        kick(self, double Δt)
        resize(self, size_t N_allocated)
        populate(self, double[::1] mv, str coord)
        """
        # Store particle meta data
        self.N = N
        self.N_allocated = 1
        self.N_local = 1
        self.mass = 1*units.m_sun
        self.softening = 1*units.kpc
        self.species = 'generic particle species'
        self.type = 'generic particle type'
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
        else:
            raise ValueError('Wrong attribute name "' + coord + '"!')

    # This method will grow/shrink the data attributes.
    # Note that it will not update the N_local attribute.
    @cython.header(N_allocated='size_t')
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
                   i='size_t',
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
        masterprint('Kicking (' + kick_algorithm + ')', self.type, '...')
        # Delegate the work to the appropriate function based on species
        if kick_algorithm == 'PP':
            PP(self, Δt)
        elif kick_algorithm == 'PM':
            PM(self, Δt)
        elif kick_algorithm == 'P3M':
            P3M(self, Δt)
        else:
            raise ValueError('Species "' + self.species
                             + '" has been assigned the kick algorithm "'
                             + kick_algorithm + '", which is not implemented!')
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
               type_name='str',
               species_name='str',
               mass='double',
               N='size_t',
               # Locals
               particles='Particles',
               returns='Particles',
               )
def construct(type_name, species_name, mass, N):
    # Instantiate Particles instance
    particles = Particles(N)
    # Attach information to the particles
    particles.mass = mass
    particles.species = species_name
    particles.type = type_name
    if species_name in softeningfactors:
        particles.softening = (softeningfactors[species_name]
                               *boxsize/(N**one_third))
    else:
        raise ValueError('Species "' + species_name
                         + '" do not have an assigned softening length!')
    return particles


# Function that constructs a Particles instance with random
# positions, momenta and masses. The particle data is
# scattered fair among the processes.
@cython.header(# Argument
               type_name='str',
               species_name='str',
               N='size_t',
               # Locals
               N_local='size_t',
               N_locals='tuple',
               mass='double',
               mom_max='double',
               particles='Particles',
               returns='Particles',
               )
def construct_random(type_name, species_name, N):
    # Print out message
    masterprint('Initializes particles of type "' + type_name + '" ...')
    # Minimum and maximum mass and maximum
    # momenta (in any of the three directions)
    mass = Ωm*ϱ*boxsize3/N
    mom_max = 1.5e+3*units.km/units.s*mass
    # Compute a fair distribution of particle data to the processes
    N_locals = ((N//nprocs, )*(nprocs - (N % nprocs))
                + (N//nprocs + 1, )*(N % nprocs))
    N_local = N_locals[rank]
    # Construct a Particles instance
    particles = construct(type_name,
                          species_name,
                          mass=mass,
                          N=N,
                          )
    # Populate the Particles instance with random data
    particles.populate(random(N_local)*boxsize, 'posx')
    particles.populate(random(N_local)*boxsize, 'posy')
    particles.populate(random(N_local)*boxsize, 'posz')
    particles.populate((2*random(N_local) - 1)*mom_max, 'momx')
    particles.populate((2*random(N_local) - 1)*mom_max, 'momy')
    particles.populate((2*random(N_local) - 1)*mom_max, 'momz')
    masterprint('done')
    return particles
