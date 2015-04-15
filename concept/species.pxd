cdef class Particles:
    cdef:
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
        drift(self, double __ASCII_repr_of_unicode__greek_Deltat)
        kick(self, double __ASCII_repr_of_unicode__greek_Deltat)
        resize(self, size_t N_allocated)
        populate(self, double[::1] mv, str coord)

cdef:
    Particles construct(str type_name, str species_name, double mass, size_t N)
    Particles construct_random(str type_name, str species_name, size_t N)
