# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from communication import communicate_domain_boundaries, communicate_domain_ghosts')
cimport('from mesh import diff')

cimport('from analysis import debug')


# This function implements the Hubble parameter H(a)=ȧ/a,
# as described by the Friedmann equation.
# The Hubble parameter is only ever written here. Every time the Hubble
# parameter is used in the code, this function should be called.
@cython.header(# Arguments
               a='double',
               # Locals
               returns='double',
               )
def hubble(a):
    # Currently, only matter (Ωm) and a cosmological constant (ΩΛ)
    # is implemented in the Friedmann equation.
    if enable_Hubble:
        return H0*sqrt(Ωm/(a**3 + machine_ϵ) + ΩΛ)
    return 0

# Function returning the time differentiated scale factor,
# used to integrate the scale factor forwards in time.
@cython.header(# Argumetns
               t='double',
               a='double',
               # Locals
               returns='double',
               )
def ȧ(t, a):
    return a*hubble(a)

# Function for solving ODEs of the form ḟ(t, f)
@cython.header(# Arguments
               ḟ='func_d_dd',
               f_start='double',
               t_start='double',
               t_end='double',
               abs_tol='double',
               rel_tol='double',
               save_intermediate='bint',
               # Locals
               error='double',
               f='double',
               f4='double',
               f5='double',
               h='double',
               h_max='double',
               i='Py_ssize_t',
               k1='double',
               k2='double',
               k3='double',
               k4='double',
               k5='double',
               k6='double',
               tolerence='double',
               Δt='double',
               returns='double',
               )
def rkf45(ḟ, f_start, t_start, t_end, abs_tol, rel_tol, save_intermediate=False):
    """ḟ(t, f) is the derivative of f with respect to t. Initial values
    are given by f_start and t_start. ḟ will be integrated from t_start
    to t_end. That is, the returned value is f(t_end). The absolute and
    relative accuracies are given by abs_tol, rel_tol.
    If save_intermediate is True, intermediate values optained during
    the integration will be kept in t_tab, f_tab.
    """
    global alloc_tab, f_tab, f_tab_mv, integrand_tab, integrand_tab_mv
    global size_tab, t_tab, t_tab_mv
    # The maximum and minimum step size
    Δt = t_end - t_start
    h_max = 0.1*Δt + machine_ϵ
    h_min = 10*machine_ϵ
    # Initial values
    h = h_max*rel_tol
    i = 0
    f = f_start
    t = t_start
    # Drive the method
    while t < t_end:
        # The embedded Runge–Kutta–Fehlberg 4(5) step
        k1 = h*ḟ(t,        f)
        k2 = h*ḟ(t + c2*h, f + a21*k1)
        k3 = h*ḟ(t + c3*h, f + a31*k1 + a32*k2)
        k4 = h*ḟ(t + c4*h, f + a41*k1 + a42*k2 + a43*k3)
        k5 = h*ḟ(t + c5*h, f + a51*k1 + a52*k2 + a53*k3 + a54*k4)
        k6 = h*ḟ(t + c6*h, f + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
        f5 = f               +  b1*k1          +  b3*k3 +  b4*k4 +  b5*k5 + b6*k6
        f4 = f               +  d1*k1          +  d3*k3 +  d4*k4 +  d5*k5
        # The error estimate
        error = abs(f5 - f4) + machine_ϵ
        # The local tolerence
        tolerence = (rel_tol*abs(f5) + abs_tol)*sqrt(h/Δt)
        if error < tolerence:
            # Step accepted
            t += h
            f = f5
            # Save intermediate t and f values
            if save_intermediate:
                t_tab[i] = t
                f_tab[i] = f
                i += 1
                # If necessary, t_tab and f_tab get resized (doubled)
                if i == alloc_tab:
                    alloc_tab *= 2
                    t_tab = realloc(t_tab, alloc_tab*sizeof('double'))
                    f_tab = realloc(f_tab, alloc_tab*sizeof('double'))
                    integrand_tab = realloc(integrand_tab, alloc_tab*sizeof('double'))
                    t_tab_mv = cast(t_tab, 'double[:alloc_tab]')
                    f_tab_mv = cast(f_tab, 'double[:alloc_tab]')
                    integrand_tab_mv = cast(integrand_tab, 'double[:alloc_tab]')
        # Updating step size
        h *= 0.95*(tolerence/error)**0.25
        if h > h_max:
            h = h_max
        elif h < h_min:
            h = h_min
        if t + h > t_end:
            h = t_end - t
    if save_intermediate:
        size_tab = i
    return f
# Initialize the Butcher tableau for the above Runge–Kutta–Fehlberg
# method at import time.
cython.declare(a21='double',
               a31='double',
               a41='double',
               a51='double',
               a61='double',
               a32='double',
               a42='double',
               a52='double',
               a62='double',
               a43='double',
               a53='double',
               a63='double',
               a54='double',
               a64='double',
               a65='double',
               b1='double',
               b3='double',
               b4='double',
               b5='double',
               b6='double',
               c2='double',
               c3='double',
               c4='double',
               c5='double',
               c6='double',
               d1='double',
               d3='double',
               d4='double',
               d5='double',
               )
a21 = 1/4;
a31 = 3/32;        a32 = 9/32;
a41 = 1932/2197;   a42 = -7200/2197;  a43 = 7296/2197;
a51 = 439/216;     a52 = -8;          a53 = 3680/513;    a54 = -845/4104;
a61 = -8/27;       a62 = 2;           a63 = -3544/2565;  a64 = 1859/4104;  a65 = -11/40;
b1  = 16/135;      b3  = 6656/12825;  b4  = 28561/56430; b5  = -9/50;      b6  = 2/55;
c2  = 1/4;         c3  = 3/8;         c4  = 12/13;       c5  = 1;          c6  = 1/2;
d1  = 25/216;      d3  = 1408/2565;   d4  = 2197/4104;   d5  = -1/5;
# Allocate t_tab, f_tab and integrand_tab at import time.
# t_tab and f_tab are used to store intermediate values of t, f,
# in the Runge-Kutta-Fehlberg method. integrand_tab stores the
# associated values of the integrand in ∫_t^(t + Δt) integrand dt.
cython.declare(alloc_tab='Py_ssize_t',
               f_tab='double*',
               f_tab_mv='double[::1]',
               integrand_tab='double*',
               integrand_tab_mv='double[::1]',
               size_tab='Py_ssize_t',
               t_tab='double*',
               t_tab_mv='double[::1]'
               )
alloc_tab = 100
size_tab = 0
t_tab = malloc(alloc_tab*sizeof('double'))
f_tab = malloc(alloc_tab*sizeof('double'))
integrand_tab = malloc(alloc_tab*sizeof('double'))
t_tab_mv = cast(t_tab, 'double[:alloc_tab]')
f_tab_mv = cast(f_tab, 'double[:alloc_tab]')
integrand_tab_mv = cast(integrand_tab, 'double[:alloc_tab]')

# Function for updating the scale factor
@cython.header(# Arguments
               a='double',
               t='double',
               Δt='double',
               returns='double',
               )
def expand(a, t, Δt):
    """Integrates the Friedmann equation from t to t + Δt,
    where the scale factor at time t is given by a. Returns a(t + Δt).
    """
    return rkf45(ȧ, a, t, t + Δt, abs_tol=1e-9, rel_tol=1e-9, save_intermediate=True)

# Function for calculating integrals of the sort
# ∫_t^(t + Δt) integrand(a) dt.
@cython.header(# Arguments
               integrand='str',
               # Locals
               acc='gsl_interp_accel*',
               i='Py_ssize_t',
               integral='double',
               spline='gsl_spline*',
               returns='double',
               )
def scalefactor_integral(integrand):
    """This function returns the factor
    ∫_t^(t + Δt) integrand(a) dt
    used in the drift and kick operations. The integrand is parsed
    as a string, which may only be one of these implemented values:
    integrand ∈ {'a⁻¹', 'a⁻²', 'ȧ/a'}
    It is important that the expand function expand(a, t, Δt) has been
    called prior to calling this function, as expand generates the
    values needed in the integration. You can call this function
    multiple times (and with different integrands) without calling
    expand in between.
    """
    # If expand has been called as it should, f_tab now stores
    # tabulated values of a. Compute the integrand.
    if integrand == 'a⁻¹':
        for i in range(size_tab):
            integrand_tab[i] = 1/f_tab[i]
    elif integrand == 'a⁻²':
        for i in range(size_tab):
            integrand_tab[i] = 1/f_tab[i]**2
    elif integrand == 'ȧ/a':
        for i in range(size_tab):
            integrand_tab[i] = hubble(f_tab[i])
    elif master:
        abort('The scalefactor integral with "{}" as the integrand is not implemented'
              .format(integrand))
    # Integrate integrand_tab in pure Python or Cython
    if not cython.compiled:
        integral = np.trapz(integrand_tab_mv[:size_tab], t_tab_mv[:size_tab])
    else:
        if size_tab < 10:
            # Use NumPy for small integrations
            integral = np.trapz(integrand_tab_mv[:size_tab], t_tab_mv[:size_tab])
        else:
            # Use GSL for larger integrations.
            # Allocate an interpolation accelerator
            # and a cubic spline object.
            acc = gsl_interp_accel_alloc()
            spline = gsl_spline_alloc(gsl_interp_cspline, size_tab)
            # Initialize spline
            gsl_spline_init(spline, t_tab, integrand_tab, size_tab)
            # Integrate the splined function
            integral = gsl_spline_eval_integ(spline, t_tab[0], t_tab[size_tab - 1], acc)
            # Free the accelerator and the spline object
            gsl_spline_free(spline)
            gsl_interp_accel_free(acc)
    return integral

# Function for computing the cosmic time t at some given scale factor a
@cython.header(# Arguments
               a='double',
               a_lower='double',
               t_lower='double',
               t_upper='double',
               # Locals
               a_test='double',
               a_test_prev='double',
               abs_tol='double',
               rel_tol='double',
               t='double',
               t_max='double',
               t_min='double',
               returns='double',
               )
def cosmic_time(a, a_lower=machine_ϵ, t_lower=machine_ϵ, t_upper=-1):
    """Given lower and upper bounds on the cosmic time, t_lower and
    t_upper, and the scale factor at time t_lower, a_lower,
    this function finds the future time at which the scale
    factor will have the value a.
    """
    global t_max_ever
    # This function only works when Hubble expansion is enabled
    if not enable_Hubble:
        abort('A mapping a(t) cannot be constructed when enable_Hubble == False.')
    if t_upper == -1:
        # If no explicit t_upper is parsed, use t_max_ever
        t_upper = t_max_ever
    elif t_upper > t_max_ever:
        # If parsed t_upper exceeds t_max_ever,
        # set t_max_ever to this larger value.
        t_max_ever = t_upper 
    # Tolerences
    abs_tol = 1e-9
    rel_tol = 1e-9
    # Saves copies of extreme t values
    t_min, t_max = t_lower, t_upper
    # Compute the cosmic time at which the scale factor had the value a,
    # using a binary search.
    a_test = a_test_prev = t = -1
    while (    not isclose(a_test,  a,       0, ℝ[2*machine_ϵ])
           and not isclose(t_upper, t_lower, 0, ℝ[2*machine_ϵ])):
        t = 0.5*(t_upper + t_lower)
        a_test = rkf45(ȧ, a_lower, t_min, t, abs_tol, rel_tol)
        if a_test == a_test_prev:
            if not isclose(a_test, a):
                if isclose(t, t_max):
                    # Integration stopped at t == t_max.
                    # Break out so that this function is called
                    # recursively, this time with a highter t_upper.
                    break
                else:
                    # Integration halted for whatever reason
                    abort('Integration halted.', a_test, a, t_max, t)
            break
        a_test_prev = a_test
        if a_test > a:
            t_upper = t
        else:
            t_lower = t
    # If the result is equal to t_max, it means that the t_upper
    # argument was too small! Call recursively with double t_upper.
    if isclose(t, t_max):
        return cosmic_time(a, a_test, t_max, 2*t_max)
    return t
# Initialize t_max_ever, a cosmic time later than what will
# ever be reached (if exceeded, it dynamically grows).
cython.declare(t_max_ever='double')
t_max_ever = 20*units.Gyr

# Function which evolves a fluid component forwards by one time step,
# the size of which is defined by the scale factor integrals given.
# The first-order Runge-Kutta (Euler) method is used.
@cython.header(# Arguments
               component='Component',
               a_integrals='dict',
               # Locals
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               size_i='Py_ssize_t',
               size_j='Py_ssize_t',
               size_k='Py_ssize_t',
               ux_noghosts='double[:, :, :]',
               ux_Δ_noghosts='double[:, :, :]',
               uy_noghosts='double[:, :, :]',
               uy_Δ_noghosts='double[:, :, :]',
               uz_noghosts='double[:, :, :]',
               uz_Δ_noghosts='double[:, :, :]',
               δ_noghosts='double[:, :, :]',
               δ_Δ_noghosts='double[:, :, :]',
               )
def rk1_euler_fluid(component, a_integrals):
    if component.representation != 'fluid':
        abort('Cannot integrate fluid variables of component "{}" with representation "{}"'
              .format(component.name, component.representation))
    # Extract scalar grids
    δ_noghosts  = component.fluidvars['δ'].grid_noghosts
    ux_noghosts = component.fluidvars['ux'].grid_noghosts
    uy_noghosts = component.fluidvars['uy'].grid_noghosts
    uz_noghosts = component.fluidvars['uz'].grid_noghosts
    δ_Δ_noghosts = component.fluidvars['δ'].Δ_noghosts
    ux_Δ_noghosts = component.fluidvars['ux'].Δ_noghosts
    uy_Δ_noghosts = component.fluidvars['uy'].Δ_noghosts
    uz_Δ_noghosts = component.fluidvars['uz'].Δ_noghosts
    # Sizes of the grids
    size_i = δ_noghosts.shape[0]
    size_j = δ_noghosts.shape[1]
    size_k = δ_noghosts.shape[2]
    # Nullify the Δ buffers on all fluid variables
    component.nullify_fluid_Δ()
    # Compute whole step from the start, storing the step in the
    # Δ buffers (that is, use grid as RHS and Δ as LHS).
    # Ideally, the grid should be used as both LHS and RHS,
    # removing the need for the Δ buffers. With the way LHS and RHS
    # are implemented, this is not currently possible.
    component.swap_fluid_LHS_RHS()
    evolve_fluid(component, a_integrals, step_frac=1)
    component.swap_fluid_LHS_RHS()
    # Take the whole step
    for i in range(size_i):
        for j in range(size_j):
            for k in range(size_k):
                δ_noghosts[i, j, k]  +=  δ_Δ_noghosts[i, j, k] 
                ux_noghosts[i, j, k] += ux_Δ_noghosts[i, j, k] 
                uy_noghosts[i, j, k] += uy_Δ_noghosts[i, j, k] 
                uz_noghosts[i, j, k] += uz_Δ_noghosts[i, j, k]

# Function which evolves a fluid component forwards by one time step,
# the size of which is defined by the scale factor integrals given.
# The second-order Runge-Kutta midpoint method is used.
@cython.header(# Arguments
               component='Component',
               a_integrals='dict',
               # Locals
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               size_i='Py_ssize_t',
               size_j='Py_ssize_t',
               size_k='Py_ssize_t',
               ux_noghosts='double[:, :, :]',
               ux_Δ_noghosts='double[:, :, :]',
               uy_noghosts='double[:, :, :]',
               uy_Δ_noghosts='double[:, :, :]',
               uz_noghosts='double[:, :, :]',
               uz_Δ_noghosts='double[:, :, :]',
               δ_noghosts='double[:, :, :]',
               δ_Δ_noghosts='double[:, :, :]',
               )
def rk2_midpoint_fluid(component, a_integrals):
    if component.representation != 'fluid':
        abort('Cannot integrate fluid variables of component "{}" with representation "{}"'
              .format(component.name, component.representation))
    # Extract scalar grids
    δ_noghosts    = component.fluidvars['δ'].grid_noghosts
    δ_Δ_noghosts  = component.fluidvars['δ'].Δ_noghosts
    ux_noghosts   = component.fluidvars['ux'].grid_noghosts
    uy_noghosts   = component.fluidvars['uy'].grid_noghosts
    uz_noghosts   = component.fluidvars['uz'].grid_noghosts
    ux_Δ_noghosts = component.fluidvars['ux'].Δ_noghosts
    uy_Δ_noghosts = component.fluidvars['uy'].Δ_noghosts
    uz_Δ_noghosts = component.fluidvars['uz'].Δ_noghosts
    # Sizes of the grids
    size_i = δ_noghosts.shape[0] - 1
    size_j = δ_noghosts.shape[1] - 1
    size_k = δ_noghosts.shape[2] - 1
    # Nullify the Δ buffers on all fluid variables
    component.nullify_fluid_Δ()
    # Compute half step from the start, storing the step in the
    # Δ buffers (that is, use grid as RHS and Δ as LHS).
    component.swap_fluid_LHS_RHS()
    evolve_fluid(component, a_integrals, step_frac=0.5)
    # Take the half step,
    # storing the resulting fluid in the Δ buffers.
    for i in range(size_i):
        for j in range(size_j):
            for k in range(size_k):
                δ_Δ_noghosts[i, j, k]  +=  δ_noghosts[i, j, k]
                ux_Δ_noghosts[i, j, k] += ux_noghosts[i, j, k]
                uy_Δ_noghosts[i, j, k] += uy_noghosts[i, j, k]
                uz_Δ_noghosts[i, j, k] += uz_noghosts[i, j, k]
    # Compute whole step from the midpoint (the Δ buffers) and apply it
    # at the beginning (the grids). That is, use Δ as the RHS and grid
    # as the LHS.
    component.swap_fluid_LHS_RHS()
    evolve_fluid(component, a_integrals, step_frac=1)

# Function which updates a fluid component using both the actual data
# (grid) and the update buffer (Δ) of fluid scalars, controlled by
# the LHS and RHS references on the fluid scalars.
@cython.header(# Arguments
               component='Component',
               a_integrals='dict',
               step_frac='double',
               # Locals
               LHS='double[:, :, :]',
               RHS='double[:, :, :]',
               RHS_mv='double[:, :, ::1]',
               diffx_mv='double[:, :, ::1]',
               diffy_mv='double[:, :, ::1]',
               diffz_mv='double[:, :, ::1]',
               fluidscalar='FluidScalar',
               h='double',
               i='Py_ssize_t',
               integrals='dict',
               j='Py_ssize_t',
               k='Py_ssize_t',
               size_i='Py_ssize_t',
               size_j='Py_ssize_t',
               size_k='Py_ssize_t',
               ux_RHS='double[:, :, :]',
               uy_RHS='double[:, :, :]',
               uz_RHS='double[:, :, :]',
               ux_diffx_mv='double[:, :, ::1]',
               ux_diffy_mv='double[:, :, ::1]',
               ux_diffz_mv='double[:, :, ::1]',
               uy_diffx_mv='double[:, :, ::1]',
               uy_diffy_mv='double[:, :, ::1]',
               uy_diffz_mv='double[:, :, ::1]',
               uz_diffx_mv='double[:, :, ::1]',
               uz_diffy_mv='double[:, :, ::1]',
               uz_diffz_mv='double[:, :, ::1]',
               Δgravity_mv='double[:, :, ::1]',
               δ_RHS='double[:, :, :]',
               δ_diffx_mv='double[:, :, ::1]',
               δ_diffy_mv='double[:, :, ::1]',
               δ_diffz_mv='double[:, :, ::1]',
               δ_noghosts='double[:, :, :]',
               δ_Δ_noghosts='double[:, :, :]',
               )
def evolve_fluid(component, a_integrals, step_frac):
    """Do not call this function directly. Instead call e.g. the
    rk2_midpoint_fluid function which in turn calls this function.
    """
    if component.representation != 'fluid':
        abort('Cannot evolve fluid variables of component "{}" with representation "{}"'
              .format(component.name, component.representation))
    # Version of a_integrals, scaled according to step_fac
    integrals = {key: step_frac*val for key, val in a_integrals.items()}    
    # Pre-tabulate all three differentiations of each fluid
    # scalar and store the results in the
    # designated diff buffers.
    # The physical grid spacing h is the same in all directions.
    h = boxsize/component.gridsize
    for fluidscalar in component.iterate_fluidvars():
        # Extract memoryview of the grid
        RHS_mv = fluidscalar.RHS_mv
        # Communicate pseudo and ghost points of fluid grid
        communicate_domain_boundaries(RHS_mv, mode=1)
        communicate_domain_ghosts(RHS_mv)
        # Do the differentiations
        fluidscalar.nullify_diff()
        diff(RHS_mv, 0, h, fluidscalar.diffx_mv, order=2)
        diff(RHS_mv, 1, h, fluidscalar.diffy_mv, order=2)
        diff(RHS_mv, 2, h, fluidscalar.diffz_mv, order=2)
    # Extract RHS and diff scalar grids
    δ_RHS       = component.fluidvars['δ'].RHS_noghosts
    δ_diffx_mv  = component.fluidvars['δ'].diffx_mv
    δ_diffy_mv  = component.fluidvars['δ'].diffy_mv
    δ_diffz_mv  = component.fluidvars['δ'].diffz_mv
    ux_RHS      = component.fluidvars['ux'].RHS_noghosts
    ux_diffx_mv = component.fluidvars['ux'].diffx_mv
    ux_diffy_mv = component.fluidvars['ux'].diffy_mv
    ux_diffz_mv = component.fluidvars['ux'].diffz_mv
    uy_RHS      = component.fluidvars['uy'].RHS_noghosts
    uy_diffx_mv = component.fluidvars['uy'].diffx_mv
    uy_diffy_mv = component.fluidvars['uy'].diffy_mv
    uy_diffz_mv = component.fluidvars['uy'].diffz_mv
    uz_RHS      = component.fluidvars['uz'].RHS_noghosts
    uz_diffx_mv = component.fluidvars['uz'].diffx_mv
    uz_diffy_mv = component.fluidvars['uz'].diffy_mv
    uz_diffz_mv = component.fluidvars['uz'].diffz_mv
    # Sizes of the grids
    size_i = δ_RHS.shape[0] - 1
    size_j = δ_RHS.shape[1] - 1
    size_k = δ_RHS.shape[2] - 1
    # Update the δ fluid variable
    LHS = component.fluidvars['δ'].LHS_noghosts
    for i in range(size_i):
        for j in range(size_j):
            for k in range(size_k):
                # ∂ₜδ = -a⁻¹∇·([1 + δ]u)
                #     = -a⁻¹(∇δ·u + (1 + δ)∇·u)
                LHS[i, j, k] += -ℝ[integrals['a⁻¹']]*(+ δ_diffx_mv[i, j, k]*ux_RHS[i, j, k]
                                                      + δ_diffy_mv[i, j, k]*uy_RHS[i, j, k]
                                                      + δ_diffz_mv[i, j, k]*uz_RHS[i, j, k]
                                                      + (1 + δ_RHS[i, j, k])*(+ ux_diffx_mv[i, j, k]
                                                                              + uy_diffy_mv[i, j, k]
                                                                              + uz_diffz_mv[i, j, k]
                                                                              )
                                                      )
    # Update the u fluid variable
    for fluidscalar in component.fluidvars['u']:
        LHS = fluidscalar.LHS_noghosts
        RHS = fluidscalar.RHS_noghosts
        Δgravity_mv = fluidscalar.Δgravity_mv
        diffx_mv    = fluidscalar.diffx_mv
        diffy_mv    = fluidscalar.diffy_mv
        diffz_mv    = fluidscalar.diffz_mv
        for i in range(size_i):
            for j in range(size_j):
                for k in range(size_k):
                    # ∂ₜu = -a⁻²∇φ - a⁻¹u·∇u - (ȧ/a)u
                    LHS[i, j, k] += (# Gravitational term
                                     step_frac*Δgravity_mv[i, j, k]
                                     # Other terms
                                     - ℝ[integrals['a⁻¹']]*(+ ux_RHS[i, j, k]*diffx_mv[i, j, k]
                                                            + uy_RHS[i, j, k]*diffy_mv[i, j, k]
                                                            + uz_RHS[i, j, k]*diffz_mv[i, j, k]
                                                            )
                                     - ℝ[integrals['ȧ/a']]*RHS[i, j, k]
                                     )
