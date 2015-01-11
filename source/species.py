# Import everything from the _params_active module (including cython).
# In the .pyx file, this line will be replaced by the content of _params_active.py itself 
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """


@cython.cclass
class Particles:
    """An instance of this class represents a collection of particles
    of a definite species. 
    """

    ########################################
    # NOTE! Allocation should happen here. Later, the data can be set to whatever needed. Free is important! Maybe alloc with C's alloc, not Pythons! See http://docs.cython.org/src/tutorial/memory_allocation.html
    ######################################## 


    # Initialization method. Note that data attributes are declared in the .pxd file!
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(# Arguments
                   pos='double[:, ::1]',
                   vel='double[:, ::1]',
                   mass='double',
                   )
    def __init__(self, pos, vel, mass):
        """As always, the last dimension (of pos, vel) are chosen as the
        contiguous one (the C standard). For fast access to a specific
        coordinate of all particles, the particles are distributed over
        the last dimension, while the coordinates are distributed over
        the first. That is, posx = pos[0, :] and so on.
        """
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.N = pos.shape[1]
        self.posx = cython.address(pos[0, :])
        self.posy = cython.address(pos[1, :])
        self.posz = cython.address(pos[2, :])
        self.velx = cython.address(vel[0, :])
        self.vely = cython.address(vel[1, :])
        self.velz = cython.address(vel[2, :])

    # Method for integrating particles forward in time
    @cython.cfunc
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(posx='double*',
                   posy='double*',
                   posz='double*',
                   velx='double*',
                   vely='double*',
                   velz='double*',
                   i='size_t',
                   )
    def drift(self):
        # Extracting variables
        posx = self.posx
        posy = self.posy
        posz = self.posz
        velx = self.velx
        vely = self.vely
        velz = self.velz
        # Update positions
        for i in range(self.N):
            posx[i] += velx[i]*dt
            posy[i] += vely[i]*dt
            posz[i] += velz[i]*dt
            # Toroidal boundaries
            posx[i] %= boxsize
            posy[i] %= boxsize
            posz[i] %= boxsize

