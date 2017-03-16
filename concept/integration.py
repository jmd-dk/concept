# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
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



# Class used for  1D interpolation.
# In compiled mode, cubic spline interpolation is used.
# In pure Python mode, linear interpolation is used.
@cython.cclass
class Spline:
    # Initialization method
    @cython.header(# Arguments
                   x='double[::1]',
                   y='double[::1]',
                   size='Py_ssize_t',
                   )
    def __init__(self, x, y, size=-1):
        # Here x and y = y(x) are the tabulated data.
        # Optionally, the size of the data to use may be given.
        #
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Spline type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        double min
        double max
        Py_ssize_t size
        gsl_interp_accel* acc
        gsl_spline* spline
        """
        # The size of the data to use may be given as
        # the 'size' argument. If not explicitly given,
        # use the entirety of the arrays.
        if size == -1:
            size = x.shape[0]
        self.size = size
        # Allocate an interpolation accelerator
        # and a (cubic) spline object (no-op in pure Python).
        self.acc = gsl_interp_accel_alloc()
        self.spline = gsl_spline_alloc(gsl_interp_cspline, size)
        # Populate the spline
        self.tabulate(x, y, size)

    # Method for populating the spline with tabulated values
    @cython.header(# Arguments
                   x='double[::1]',
                   y='double[::1]',
                   size='Py_ssize_t',
                   returns='double',
                   )
    def tabulate(self, x, y, size=-1):
        """This method is invoked on instance initialization,
        but it may be re-invoked later to swap out the tabulated values.
        The new tabulation should however be
        of the same size as the original.
        """
        # The size of the data to use may be given as
        # the 'size' argument. If not explicitly given,
        # use the entirety of the arrays.
        if size == -1:
            size = x.shape[0]
        # Checks on the supplied size as well as the size of x and y
        if size != self.size:
            abort('Spline object cannot change size under retabulation.')
        if size < 3:
            abort('Too few tabulated values ({}) were given for cubic spline interpolation. '
                  'At least 3 is needed'.format(size))
        if x.shape[0] < size:
            abort('Error in constructing spline: The input x havs a size of {} '
                  'but a size of {} was requested.'
                  .format(x.shape[0], size))
        if y.shape[0] < size:
            abort('Error in constructing spline: The input y havs a size of {} '
                  'but a size of {} was required.'
                  .format(y.shape[0], size))
        # Store meta data
        self.min = x[0]
        self.max = x[size - 1]
        # Use NumPy in pure Python and GSL when compiled
        if not cython.compiled:
            # Simply store the parsed arrays
            self.x = x
            self.y = y
        else:
            # Initialize the spline
            gsl_spline_init(self.spline,
                            cython.address(x[:]),
                            cython.address(y[:]),
                            size)

    # Method for doing spline evaluation
    @cython.header(x='double',
                   returns='double',
                   )
    def eval(self, x):
        # Check that the supplied x is within the interpolation interval
        if not (self.min <= x <= self.max):
            abort('Could not interpolate {} because it is outside the interval [{}, {}].'
                  .format(x, self.min, self.max)
                  )
        # Use NumPy in pure Python and GSL when compiled
        if not cython.compiled:
            return np.interp(x, self.x, self.y)
        else:
            return gsl_spline_eval(self.spline, x, self.acc)

    # Method for doing spline derivative evaluation
    @cython.header(x='double',
                   returns='double',
                   )
    def eval_deriv(self, x):
        # Check that the supplied x is within the interpolation interval
        if not (self.min <= x <= self.max):
            abort('Could not interpolate {} because it is outside the interval [{}, {}].'
                  .format(x, self.min, self.max)
                  )
        # Use NumPy in pure Python and GSL when compiled
        if not cython.compiled:
            if x == self.min:
                i0, i1 = 0, 1
            elif x == self.max:
                i0, i1 = -2, -1
            else:
                i0 = np.where(self.x <= x)[0][-1]
                i1 = np.where(self.x >= x)[0][ 0]
            return (self.y[i1] - self.y[i0])/(self.x[i1] - self.x[i0])
        else:
            return gsl_spline_eval_deriv(self.spline, x, self.acc)

    # Method for computing the definite integral over some
    # interval [a, b] of the splined function.
    @cython.header(# Arguments
                   a='double',
                   b='double',
                   # Locals
                   sign='int',
                   ᔑ='double',
                   returns='double',
                   )
    def integrate(self, a, b):
        # Check that the supplied limits are
        # within the interpolation interval.
        if not (self.min <= a <= self.max):
            abort('Error integrating spline from {} to {}: '
                  'The first limit is outside the interval [{}, {}].'
                  .format(a, b, self.min, self.max)
                  )
        if not (self.min <= b <= self.max):
            abort('Error integrating spline from {} to {}: '
                  'The second limit is outside the interval [{}, {}].'
                  .format(a, b, self.min, self.max)
                  )
        # The function gsl_spline_eval_integ fails for a > b.
        # Take care of this manually by switching a and b and note
        # down a sign change.
        sign = +1
        if a > b:
            a, b = b, a
            sign = -1
        # Use NumPy in pure Python and GSL when compiled
        if not cython.compiled:
            # Create arrays of tabulated points with interpolated
            # values at both ends.
            mask = np.logical_and(a <= self.x, self.x <= b)
            x = np.concatenate(([          a ], self.x[mask], [          b ]))
            y = np.concatenate(([self.eval(a)], self.y[mask], [self.eval(b)]))
            # Do trapezoidal integration over the arrays
            ᔑ = np.trapz(y, x)
        else:
            ᔑ = gsl_spline_eval_integ(self.spline, a, b, self.acc)
        # Remember the sign change for a > b
        ᔑ *= sign
        return ᔑ

    # This method is automaticlly called when a Spline instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        # Free the accelerator and the spline object
        gsl_spline_free(self.spline)
        gsl_interp_accel_free(self.acc)



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

# Function returning the proper time differentiated scale factor.
# Because this function is used by rkf45, it needs to take in both
# t and a as arguments, even though t is not used.
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
    h_min = 10*machine_ϵ
    h_max = 0.01*Δt + h_min
    # Initial values
    h = h_max*rel_tol
    i = 0
    f = f_start
    t = t_start
    # Drive the method
    while t < t_end:
        # The embedded Runge-Kutta-Fehlberg 4(5) step
        k1 = h*ḟ(t             , f)
        k2 = h*ḟ(t + ℝ[1/4  ]*h, f + ℝ[1/4      ]*k1)
        k3 = h*ḟ(t + ℝ[3/8  ]*h, f + ℝ[3/32     ]*k1 + ℝ[9/32      ]*k2)
        k4 = h*ḟ(t + ℝ[12/13]*h, f + ℝ[1932/2197]*k1 + ℝ[-7200/2197]*k2 + ℝ[7296/2197 ]*k3)
        k5 = h*ḟ(t +          h, f + ℝ[439/216  ]*k1 + ℝ[-8        ]*k2 + ℝ[3680/513  ]*k3 + ℝ[-845/4104  ]*k4)
        k6 = h*ḟ(t + ℝ[1/2  ]*h, f + ℝ[-8/27    ]*k1 + ℝ[2         ]*k2 + ℝ[-3544/2565]*k3 + ℝ[1859/4104  ]*k4 + ℝ[-11/40]*k5)
        f5 =                     f + ℝ[16/135   ]*k1                    + ℝ[6656/12825]*k3 + ℝ[28561/56430]*k4 + ℝ[-9/50 ]*k5 + ℝ[2/55]*k6
        f4 =                     f + ℝ[25/216   ]*k1                    + ℝ[1408/2565 ]*k3 + ℝ[2197/4104  ]*k4 + ℝ[-1/5  ]*k5
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
               i='Py_ssize_t',
               spline='Spline',
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
    for i in range(size_tab):
        with unswitch:
            if integrand == 'a⁻¹':
                integrand_tab[i] = 1/f_tab[i]
            elif integrand == 'a⁻²':
                integrand_tab[i] = 1/f_tab[i]**2
            elif integrand == 'ȧ/a':
                integrand_tab[i] = hubble(f_tab[i])
            elif master:
                abort('The scalefactor integral with "{}" as the integrand is not implemented'
                      .format(integrand))
    # Do the integration over the entire tabulated integrand
    spline = Spline(t_tab_mv, integrand_tab_mv, size_tab)
    return spline.integrate(t_tab[0], t_tab[size_tab - 1])

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
