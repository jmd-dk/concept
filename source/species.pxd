cdef class Particles:
    # Data attributes
    cdef double mass
    cdef size_t N
    cdef size_t N_local
    cdef size_t N_allocated
    cdef double[::1] posx_mw
    cdef double[::1] posy_mw
    cdef double[::1] posz_mw
    cdef double[::1] velx_mw
    cdef double[::1] vely_mw
    cdef double[::1] velz_mw
    cdef double* posx
    cdef double* posy
    cdef double* posz
    cdef double* velx
    cdef double* vely
    cdef double* velz
    cdef str type
    cdef str species
    # Methods
    cdef drift(self, double dt)
    cdef kick(self, double dt)
    cdef resize(self, size_t N_allocated)
    cdef populate(self, double[::1] mw, str coord)

cdef Particles construct(str type_name, str species_name, double mass, size_t N)

cdef Particles construct_random(str type_name, str species_name, size_t N)
