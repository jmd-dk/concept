cdef:
    # Numerical parameters
    double boxsize
    int ewald_gridsize
    ptrdiff_t PM_gridsize

    # Cosmological parameters
    double H0
    double __ASCII_repr_of_unicode__greek_varrho
    double __ASCII_repr_of_unicode__greek_Omegam
    double __ASCII_repr_of_unicode__greek_Omega__ASCII_repr_of_unicode__greek_Lambda
    double a_begin
    double a_end

    # Graphics
    str framefolder
    str liveframe
    str image_format
    size_t framespace
    str remote_liveframe
    str protocol
