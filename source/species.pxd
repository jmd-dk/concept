cdef class Particles:
    # Data attributes
    cdef double mass
    cdef size_t N
    cdef size_t N_local
    cdef size_t N_allocated
    cdef double[::1] posx_mw
    cdef double[::1] posy_mw
    cdef double[::1] posz_mw
    cdef double[::1] momx_mw
    cdef double[::1] momy_mw
    cdef double[::1] momz_mw
    cdef double* posx
    cdef double* posy
    cdef double* posz
    cdef double* momx
    cdef double* momy
    cdef double* momz
    cdef str type
    cdef str species
    # Methods
    cdef drift(self, double __ASCII_repr_of_unicode__greek_Deltat)
    cdef kick(self, double __ASCII_repr_of_unicode__greek_Deltat)
    cdef resize(self, size_t N_allocated)
    cdef populate(self, double[::1] mw, str coord)

cdef Particles construct(str type_name, str species_name, double mass, size_t N)

cdef Particles construct_random(str type_name, str species_name, size_t N)
