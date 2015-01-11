cdef class Particles:
    cdef double[:, ::1] pos
    cdef double[:, ::1] vel
    cdef double mass
    cdef size_t N
    cdef double* posx
    cdef double* posy
    cdef double* posz
    cdef double* velx
    cdef double* vely
    cdef double* velz
    cdef drift(self)
