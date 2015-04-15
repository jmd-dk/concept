ctypedef double  (*func_d_dd_pxd)   (double, double)

cdef:
    double __ASCII_repr_of_unicode__dot_a(double t, double a)
    double rkf45(func_d_dd_pxd __ASCII_repr_of_unicode__dot_f, double f_start, double t_start, double t_end, double __ASCII_repr_of_unicode__greek_delta, double __ASCII_repr_of_unicode__greek_epsilon, bint save_intermediate)
    double expand(double a, double t, double __ASCII_repr_of_unicode__greek_Deltat)
    double scalefactor_integral(int power)
    double cosmic_time(double a, double a_lower=*, double t_lower=*, double t_upper=*)
