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
    if enable_Hubble:
        return H0*sqrt(+ Ωr/(a**4 + machine_ϵ)  # Radiation
                       + Ωm/(a**3 + machine_ϵ)  # Matter
                       + ΩΛ                     # Cosmological constant
                       )
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
                    abort('Integration of scale factor a(t) halted')
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

# Function which sets the value of universals.a and universals.t
# based on the user parameters a_begin and t_begin together with the
# cosmology if enable_Hubble == True.
@cython.header
def initiate_time():
    if enable_Hubble:
        # Hubble expansion enabled.
        # A specification of initial scale factor or
        # cosmic time is needed.
        if 'a_begin' in user_params:
            # a_begin specified
            if 't_begin' in user_params:
                # t_begin also specified
                masterwarn('Ignoring t_begin = {}*{} becuase enable_Hubble == True\n'
                           'and a_begin is specified'.format(t_begin, unit_time))
            universals.a = a_begin
            universals.t = cosmic_time(universals.a)
        elif 't_begin' in user_params:
            # a_begin not specified, t_begin specified
            universals.t = t_begin
            universals.a = expand(machine_ϵ, machine_ϵ, universals.t)
        else:
            # Neither a_begin nor t_begin is specified.
            # One or the other is needed when enable_Hubble == True.
            abort('No initial scale factor (a_begin) or initial cosmic time (t_begin) specified. '
                  'A specification of one or the other is needed when enable_Hubble == True.')
    else:
        # Hubble expansion disabled.
        # Values of the scale factor (and therefore a_begin)
        # are meaningless.
        # Set universals.a to unity, effectively ignoring its existence.
        universals.a = 1
        if 'a_begin' in user_params:
            masterwarn('Ignoring a_begin = {} becuase enable_Hubble == False'.format(a_begin))
        # Use universals.t = t_begin, which defaults to 0 when not
        # supplied by the user, as specified in commons.py.
        universals.t = t_begin

@cython.header(# Arguments
               component='Component',
               ᔑdt='dict',
               # Locals
               shape='tuple',
               ϱ='double[:, :, ::1]',
               ϱux='double[:, :, ::1]',
               ϱuy='double[:, :, ::1]',
               ϱuz='double[:, :, ::1]',
               ϱˣ='double[:, :, ::1]',
               ϱuxˣ='double[:, :, ::1]',
               ϱuyˣ='double[:, :, ::1]',
               ϱuzˣ='double[:, :, ::1]',
               ϱ_ijk='double',
               ϱux_ijk='double',
               ϱuy_ijk='double',
               ϱuz_ijk='double',
               h='double',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               ϱux_source='double[:, :, ::1]',
               ϱuy_source='double[:, :, ::1]',
               ϱuz_source='double[:, :, ::1]',
               indices_local_start='Py_ssize_t[::1]',
               indices_local_end='Py_ssize_t[::1]',
               indices_start='Py_ssize_t[::1]',
               indices_end='Py_ssize_t[::1]',
               step='int',
               steps='int[::1]',
               i_step='Py_ssize_t',
               step_order='str',
               ϱ_flux='double',
               ϱux_flux='double',
               ϱuy_flux='double',
               ϱuz_flux='double',
               ϱ_sjk ='double',
               ϱ_isk ='double',
               ϱ_ijs ='double',
               Σϱu_ijk='double',
               Σu_ijk='double',
               ux_sjk='double',
               uy_isk='double',
               uz_ijs='double',
               ϱux_sjk='double',
               ϱuy_sjk='double',
               ϱuz_sjk='double',
               ϱux_isk='double',
               ϱuy_isk='double',
               ϱuz_isk='double',
               ϱux_ijs='double',
               ϱuy_ijs='double',
               ϱuz_ijs='double',
               )
def maccormack(component, ᔑdt):
    """First forward differencing and then backward differencing.
    """
    # Parameters
    step_order = 'forward, backward'
    shape = component.fluidvars['shape']
    h = boxsize/component.gridsize
    # Arrays of start and end indices for the local part of the
    # fluid grids, meaning disregarding pseudo points and ghost points.
    # We have 2 ghost points in the beginning and 1 pseudo point and
    # 2 ghost points in the end.
    indices_local_start = asarray([2, 2, 2], dtype=C2np['Py_ssize_t'])
    indices_local_end   = asarray(shape    , dtype=C2np['Py_ssize_t']) - 2 - 1
    if step_order == 'forward, backward':
        steps = asarray([+1, -1], dtype=C2np['int'])
    elif step_order == 'backward, forward':
        steps = asarray([-1, +1], dtype=C2np['int'])
    # Extract fluid grids
    ϱ = component.fluidvars['ϱ'].grid_mv
    ϱux = component.fluidvars['ϱux'].grid_mv
    ϱuy = component.fluidvars['ϱuy'].grid_mv
    ϱuz = component.fluidvars['ϱuz'].grid_mv
    # Extract starred fluid grids
    ϱˣ = component.fluidvars['ϱ'].gridˣ_mv
    ϱuxˣ = component.fluidvars['ϱux'].gridˣ_mv
    ϱuyˣ = component.fluidvars['ϱuy'].gridˣ_mv
    ϱuzˣ = component.fluidvars['ϱuz'].gridˣ_mv
    # Extract needed source term grids
    ϱux_source = component.fluidvars['ϱux'].source_mv
    ϱuy_source = component.fluidvars['ϱuy'].source_mv
    ϱuz_source = component.fluidvars['ϱuz'].source_mv
    # Add source terms.
    # In addition to local grid points, loop over 
    # one layer of grid points in both directions.
    for         i in range(ℤ[indices_local_start[0] - 1], ℤ[indices_local_end[0] + 1]):
        for     j in range(ℤ[indices_local_start[1] - 1], ℤ[indices_local_end[1] + 1]):
            for k in range(ℤ[indices_local_start[2] - 1], ℤ[indices_local_end[2] + 1]):
                ϱux[i, j, k] += ℝ[ᔑdt['a⁻²']]*ϱux_source[i, j, k]
                ϱuy[i, j, k] += ℝ[ᔑdt['a⁻²']]*ϱuy_source[i, j, k]
                ϱuz[i, j, k] += ℝ[ᔑdt['a⁻²']]*ϱuz_source[i, j, k]
    # Nullify the grids of the starred variables
    component.nullify_fluid_gridˣ()
    # The two MacCormack steps. Source terms will be added later.
    for i_step in range(2):
        step = steps[i_step]
        # Determine which part of the grids to loop over
        if i_step == 0:
            # First step
            if step == +1:  # forward, backward
                # In addition to local grid points, loop over 
                # one layer of grid points in the backward directions.
                indices_start = asarray(indices_local_start) - 1
                indices_end   = indices_local_end
            elif step == -1:  # backward, forward
                # In addition to local grid points, loop over 
                # one layer of grid points in the forward directions.
                indices_start = indices_local_start
                indices_end   = asarray(indices_local_end) + 1
        elif i_step == 1:
            # Second step.
            # Loop over local grid points only.
            indices_start = indices_local_start
            indices_end   = indices_local_end
        # Loop which compute the starred variables from the unstarred
        # (first step) or update the unstarred variables from the
        # starred (second step).
        for         i in range(ℤ[indices_start[0]], ℤ[indices_end[0]]):
            for     j in range(ℤ[indices_start[1]], ℤ[indices_end[1]]):
                for k in range(ℤ[indices_start[2]], ℤ[indices_end[2]]):
                    # Density at this point
                    ϱ_ijk = ϱ[i, j, k]
                    # Density at forward (backward) points
                    ϱ_sjk = ϱ[i + step, j       , k       ]
                    ϱ_isk = ϱ[i       , j + step, k       ]
                    ϱ_ijs = ϱ[i       , j       , k + step]
                    # Momentum densities at this point
                    ϱux_ijk = ϱux[i, j, k]
                    ϱuy_ijk = ϱuy[i, j, k]
                    ϱuz_ijk = ϱuz[i, j, k]
                    Σϱu_ijk = ϱux_ijk + ϱuy_ijk + ϱuz_ijk
                    # Momentum densities at forward (backward) points
                    ϱux_sjk = ϱux[i + step, j       , k       ]
                    ϱux_isk = ϱux[i       , j + step, k       ]
                    ϱux_ijs = ϱux[i       , j       , k + step]
                    ϱuy_sjk = ϱuy[i + step, j       , k       ]
                    ϱuy_isk = ϱuy[i       , j + step, k       ]
                    ϱuy_ijs = ϱuy[i       , j       , k + step]
                    ϱuz_sjk = ϱuz[i + step, j       , k       ]
                    ϱuz_isk = ϱuz[i       , j + step, k       ]
                    ϱuz_ijs = ϱuz[i       , j       , k + step]
                    # Velocity sum at this point
                    Σu_ijk = Σϱu_ijk/ϱ_ijk
                    # Velocities at forward (backward) points
                    ux_sjk = ϱux_sjk/ϱ_sjk
                    uy_isk = ϱuy_isk/ϱ_isk
                    uz_ijs = ϱuz_ijs/ϱ_ijs
                    # Flux of ϱ (ϱ*u)
                    ϱ_flux = step*(# Forward fluxes
                                   + ϱux_sjk
                                   + ϱuy_isk
                                   + ϱuz_ijs
                                   # Local fluxes
                                   - Σϱu_ijk
                                   )
                    # Flux of ϱux (ϱux*u)
                    ϱux_flux = step*(# Forward fluxes
                                     + ϱux_sjk*ux_sjk
                                     + ϱux_isk*uy_isk
                                     + ϱux_ijs*uz_ijs
                                     # Local fluxes
                                     - ϱux_ijk*Σu_ijk
                                     )
                    # Flux of ϱuy (ϱuy*u)
                    ϱuy_flux = step*(# Forward fluxes
                                     + ϱuy_sjk*ux_sjk
                                     + ϱuy_isk*uy_isk
                                     + ϱuy_ijs*uz_ijs
                                     # Local fluxes
                                     - ϱuy_ijk*Σu_ijk
                                     )
                    # Flux of ϱuz (ϱuz*u)
                    ϱuz_flux = step*(# Forward fluxes
                                     + ϱuz_sjk*ux_sjk
                                     + ϱuz_isk*uy_isk
                                     + ϱuz_ijs*uz_ijs
                                     # Local fluxes
                                     - ϱuz_ijk*Σu_ijk
                                     )
                    # Update ϱ
                    ϱˣ[i, j, k] += (# Initial value
                                    + ϱ_ijk
                                    # Flux
                                    - ℝ[ᔑdt['a⁻¹']/h]*ϱ_flux
                                    )
                    # Update ϱux
                    ϱuxˣ[i, j, k] += (# Initial value
                                      + ϱux_ijk
                                      # Flux
                                      - ℝ[ᔑdt['a⁻¹']/h]*ϱux_flux
                                      # Hubble drag
                                      - ℝ[ᔑdt['ȧ/a']]*ϱux_ijk
                                      )
                    # Update ϱuy
                    ϱuyˣ[i, j, k] += (# Initial value
                                      + ϱuy_ijk
                                      # Flux
                                      - ℝ[ᔑdt['a⁻¹']/h]*ϱuy_flux
                                      # Hubble drag
                                      - ℝ[ᔑdt['ȧ/a']]*ϱuy_ijk
                                      )
                    # Update ϱuz
                    ϱuzˣ[i, j, k] += (# Initial value
                                      + ϱuz_ijk
                                      # Flux
                                      - ℝ[ᔑdt['a⁻¹']/h]*ϱuz_flux
                                      # Hubble drag
                                      - ℝ[ᔑdt['ȧ/a']]*ϱuz_ijk
                                      )
        # Swap the role of the fluid variable grids and buffers
        ϱ  , ϱˣ   = ϱˣ  , ϱ
        ϱux, ϱuxˣ = ϱuxˣ, ϱux
        ϱuy, ϱuyˣ = ϱuyˣ, ϱuy
        ϱuz, ϱuzˣ = ϱuzˣ, ϱuz
    # Because the fluid variables have been updated twice
    # (the two steps above), the values of the fluid grids
    # need to be halved.
    for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
        for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
            for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                ϱ  [i, j, k] *= 0.5
                ϱux[i, j, k] *= 0.5
                ϱuy[i, j, k] *= 0.5
                ϱuz[i, j, k] *= 0.5
