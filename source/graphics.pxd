from species cimport Particles

cdef:
    animate(Particles particles, size_t timestep)
    timestep_message(int timestep, double t_iter, double a, double t)
