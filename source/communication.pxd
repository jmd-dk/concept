from species cimport Particles
cdef exchange_all(Particles particles)

cdef size_t[::1] find_N_recv(size_t[::1] N_send)
