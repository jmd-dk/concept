from species cimport Particles

cdef:
    size_t[::1] find_N_recv(size_t[::1] N_send)
    exchange(Particles particles, bint reset_buffers=*)
    list cutout_domains(int n, bint basecall=*)
    int domain(double x, double y, double z)
    dict neighboring_ranks()
