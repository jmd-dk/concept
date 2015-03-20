from species cimport Particles
cdef exchange(Particles particles, bint reset_buffers=*)

cdef size_t[::1] find_N_recv(size_t[::1] N_send)

cdef list cutout_domains(int n, bint basecall=*)
