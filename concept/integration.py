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
    h_max = 0.1*Δt
    h_min = 10*machine_ϵ
    # Initial values
    h = h_max*rel_tol
    i = 0
    f = f_start
    t = t_start
    # Drive the method
    while t < t_end:
        # The embedded Runge–Kutta–Fehlberg 4(5) step
        k1 = h*ḟ(t, f)
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
def cosmic_time(a, a_lower=machine_ϵ, t_lower=machine_ϵ, t_upper=20*units.Gyr):
    """Given lower and upper bounds on the cosmic time, t_lower and
    t_upper, and the scale factor at time t_lower, a_lower,
    this function finds the future time at which the scale
    factor will have the value a.
    """
    if not enable_Hubble:
        abort('A mapping a(t) cannot be constructed when enable_Hubble == False.')
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
                abort('Integration halted.')
            break
        a_test_prev = a_test
        if a_test > a:
            t_upper = t
        else:
            t_lower = t
    # If the result is equal to t_max, it means that the t_upper
    # argument was too small! Call recursively with double t_upper.
    if t == t_max:
        return cosmic_time(a, a_test, t_max, 2*t_max)
    return t

# Initialize the Butcher tableau for the Runge–Kutta–Fehlberg
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
a41 = 1932.0/2197; a42 = -7200/2197;  a43 = 7296/2197;
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

