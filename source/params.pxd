cdef:
    # Numerical parameters
    double boxsize
    int ewald_gridsize
    ptrdiff_t PM_gridsize
    double softening

    # Cosmological parameters
    double H0
    double __ASCII_repr_of_unicode__greek_Omegam
    double __ASCII_repr_of_unicode__greek_Omega__ASCII_repr_of_unicode__greek_Lambda
    double a_begin
    double a_end

    # Graphics
    str image_format
    str framefolder
    str liveframe
    str scp_liveframe
    size_t framespace
