ctypedef double (*func_d_dd_pxd)(double, double)

cdef:
    double expand(double a, double t, double __ASCII_repr_of_unicode__greek_Deltat)
    double cosmic_time(double a, double a_lower=*, double t_lower=*, double t_upper=*)
    double scalefactor_integral(int power)
