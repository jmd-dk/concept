cdef:
    # Input/output
    str IC_file
    str output_dir
    str output_type
    str snapshot_base
    tuple outputtimes

    # Numerical parameters
    double boxsize
    int ewald_gridsize
    ptrdiff_t PM_gridsize
    dict softeningfactors
    double __ASCII_repr_of_unicode__greek_Deltat_factor

    # Cosmological parameters
    double H0
    double __ASCII_repr_of_unicode__greek_varrho
    double __ASCII_repr_of_unicode__greek_Omegam
    double __ASCII_repr_of_unicode__greek_Omega__ASCII_repr_of_unicode__greek_Lambda
    double a_begin

    # Graphics
    str framefolder
    str liveframe
    str image_format
    size_t framespace
    str remote_liveframe
    str protocol

    # Simulation options
    bint use_Ewald
    dict kick_algorithms
