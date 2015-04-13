from species cimport Particles

cdef:
    animate(Particles particles, size_t timestep, double a, double a_snapshot)
    str significant_figures(double f, int n, int just=*, bint scientific=*)
