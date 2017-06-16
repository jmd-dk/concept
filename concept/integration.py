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
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from communication import smart_mpi')



# Class used for 1D interpolation.
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
                  'At least 3 is needed.'.format(size))
        if x.shape[0] < size:
            abort('Error in constructing spline: The input x has a size of {} '
                  'but a size of {} is required.'
                  .format(x.shape[0], size))
        if y.shape[0] < size:
            abort('Error in constructing spline: The input y has a size of {} '
                  'but a size of {} is required.'
                  .format(y.shape[0], size))
        # Store meta data
        self.min = x[0]
        self.max = x[size - 1]
        if self.min > self.max:
            abort('A tabulation of a Spline instance was attempted '
                  'with x values that are not strictly increasing')
        # Use NumPy in pure Python and GSL when compiled
        if not cython.compiled:
            # Simply store the passed data.
            # In compiled mode the passed data is not used by the Spline
            # object instantiation, and so it can safely be freed.
            # To guarantee the same safety in pure Python mode, we do
            # not store the data directly, but rather copies thereof.
            self.x = x.copy()
            self.y = y.copy()
        else:
            # Initialize the spline
            gsl_spline_init(self.spline, cython.address(x[:]), cython.address(y[:]), size)

    # Method for doing spline evaluation
    @cython.pheader(x='double',
                    returns='double',
                    )
    def eval(self, x):
        # Check that the supplied x is within the interpolation interval
        x = self.in_interval(x, 'interpolate to')
        # Use NumPy in pure Python and GSL when compiled
        if not cython.compiled:
            return np.interp(x, self.x, self.y)
        else:
            x = gsl_spline_eval(self.spline, x, self.acc)
            return x

    # Method for doing spline derivative evaluation
    @cython.header(x='double',
                   returns='double',
                   )
    def eval_deriv(self, x):
        # Check that the supplied x is within the interpolation interval
        x = self.in_interval(x, 'differentiate at')
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
        a = self.in_interval(a, 'integrate from')
        b = self.in_interval(b, 'integrate to')
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

    # Method for checking whether a given number
    # is within the tabulated interval.
    @cython.header(# Arguments
                   x='double',
                   action='str',
                   # Locals
                   abs_tol='double',
                   rel_tol='double',
                   returns='double',
                   )
    def in_interval(self, x, action='interpolate to'):
        # Check that the supplied x is within the
        # interpolation interval. If it is just barely outside of it,
        # move it to the boundary.
        rel_tol = 1e-9
        abs_tol = rel_tol*(self.max - self.min)
        if x < self.min:
            if x > self.min - abs_tol:
                x = self.min
            else:
                abort('Could not {} {} because it is outside the tabulated interval [{}, {}].'
                      .format(action, x, self.min, self.max)
                      )
        elif x > self.max:
            if x < self.max + abs_tol:
                x = self.max
            else:
                abort('Could not {} {} because it is outside the tabulated interval [{}, {}].'
                      .format(action, x, self.min, self.max)
                      )
        return x

    # This method is automaticlly called when a Spline instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        # Free the accelerator and the spline object
        gsl_spline_free(self.spline)
        gsl_interp_accel_free(self.acc)

# Function for updating the scale factor
@cython.header(# Arguments
               t='double',
               # Locals
               returns='double',
               )
def scale_factor(t=-1):
    if not enable_Hubble:
        return 1
    if t == -1:
        t = universals.t
    # When using CLASS, simply look up a(t) in the tabulated data
    if enable_class:
        if spline_t_a is None:
            abort('The function a(t) has not been tabulated. Have you called initiate_time?')
        return spline_t_a.eval(t)
    # Not using CLASS.
    # Integrate the Friedmann equation from the beginning of time
    # to the requested time.
    return rkf45(ȧ, machine_ϵ, machine_ϵ, t, abs_tol=1e-9, rel_tol=1e-9)

# Function for computing the cosmic time t at some given scale factor a
@cython.pheader(# Arguments
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
def cosmic_time(a=-1, a_lower=machine_ϵ, t_lower=machine_ϵ, t_upper=-1):
    """Given lower and upper bounds on the cosmic time, t_lower and
    t_upper, and the scale factor at time t_lower, a_lower,
    this function finds the future time at which the scale
    factor will have the value a.
    """
    global t_max_ever
    # This function only works when Hubble expansion is enabled
    if not enable_Hubble:
        abort('The cosmic_time function was called. '
              'A mapping from a to t is only meaningful when Hubble expansion is enabled.')
    if a == -1:
        a = universals.a
    # When using CLASS, simply look up t(a) in the tabulated data
    if enable_class:
        if spline_a_t is None:
            abort('The function t(a) has not been tabulated. Have you called initiate_time?')
        return spline_a_t.eval(a)
    # Not using CLASS
    if t_upper == -1:
        # If no explicit t_upper is passed, use t_max_ever
        t_upper = t_max_ever
    elif t_upper > t_max_ever:
        # If passed t_upper exceeds t_max_ever,
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

# This function implements the Hubble parameter H(a)=ȧ/a
# The Hubble parameter is only ever written here. Every time the Hubble
# parameter is used in the code, this function should be called.
@cython.header(# Arguments
               a='double',
               # Locals
               ΩΛ='double',
               returns='double',
               )
def hubble(a=-1):
    if not enable_Hubble:
        return 0
    if a == -1:
        a = universals.a
    # When using CLASS, simply look up H(a) in the tabulated data.
    # The largest tabulated a is 1, but it may happen that this function
    # gets called with an a that is sightly larger. In this case,
    # compute H directly via the simple Friedmann equation, just as when
    # CLASS is not being used.
    if enable_class and a <= 1:
        if spline_a_H is None:
            abort('The function H(a) has not been tabulated. Have you called initiate_time?')
        return spline_a_H.eval(a)
    # CLASS not enabled. Assume that the universe is flat and consists
    # purely of matter and cosmological constant
    ΩΛ = 1 - Ωm
    return H0*sqrt(Ωm/(a**3 + machine_ϵ) + ΩΛ)

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
    the integration will be kept in the global variables t_tab, f_tab.
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
    while t < t_end - 1e+3*machine_ϵ:
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
# t_tab and f_tab are used to store intermediate values of t and f
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

# Function for calculating integrals of the sort
# ∫_t^(t + Δt) integrand(a) dt.
@cython.header(# Arguments
               key='object',  # str or tuple
               # Locals
               a='double',
               component='Component',
               i='Py_ssize_t',
               integrand='str',
               spline='Spline',
               t='double',
               w='double',
               returns='double',
               )
def scalefactor_integral(key):
    """This function returns the factor
    ∫_t^(t + Δt) integrand(a) dt
    used in the drift and kick operations. The integrand is passed
    as the key argument, which may be a string (e.g. 'a⁻¹') or a tuple
    in the format (string, component), where again the string is really
    the integrand. This second form is used when the integrand is
    component specific, e.g. 'a⁻³ʷ'.
    It is important that the expand function expand(a, t, Δt) has been
    called prior to calling this function, as expand generates the
    values needed in the integration. You can call this function
    multiple times (and with different integrands) without calling
    expand in between.
    """
    # Extract the integrand from the passed key
    if isinstance(key, str):
        integrand = key
    else:  # tuple key
        integrand, component = key
    # If expand has been called as it should, t_tab stores the tabulated
    # values of t while f_tab now stores the tabulated values of a.
    # Compute the integrand for these tabulated values.
    for i in range(size_tab):
        a = f_tab[i]
        t = t_tab[i]
        with unswitch:
            if integrand == '1' or integrand == '':
                integrand_tab[i] = 1
            elif integrand == 'a⁻¹':
                integrand_tab[i] = 1/a
            elif integrand == 'a⁻²':
                integrand_tab[i] = 1/a**2
            elif integrand == 'ȧ/a':
                integrand_tab[i] = hubble(a)
            elif integrand == 'a⁻³ʷ':
                w = component.w(t=t, a=a)
                integrand_tab[i] = a**(-3*w)
            elif integrand == 'a⁻³ʷ⁻¹':
                w = component.w(t=t, a=a)
                integrand_tab[i] = a**(-3*w - 1)
            elif integrand == 'a³ʷ⁻²':
                w = component.w(t=t, a=a)
                integrand_tab[i] = a**(3*w - 2)
            elif integrand == 'a⁻³ʷw/(1+w)':
                w = component.w(t=t, a=a)
                integrand_tab[i] = a**(-3*w)*w/(1 + w)
            elif integrand == 'a³ʷ⁻²(1+w)':
                w = component.w(t=t, a=a)
                integrand_tab[i] = a**(3*w - 2)*(1 + w)
            elif integrand == 'ẇ/(1+w)':
                w = component.w(t=t, a=a)
                ẇ = component.ẇ(t=t, a=a)
                integrand_tab[i] = ẇ/(1 + w)
            elif integrand == 'ẇlog(a)':
                ẇ = component.ẇ(t=t, a=a)
                integrand_tab[i] = ẇ*log(a)
            elif master:
                abort('The scalefactor integral with "{}" as the integrand is not implemented'
                      .format(integrand))
    # Do the integration over the entire tabulated integrand
    spline = Spline(t_tab_mv, integrand_tab_mv, size_tab)
    a = spline.integrate(t_tab[0], t_tab[size_tab - 1])
    return a

# Function which sets the value of universals.a and universals.t
# based on the user parameters a_begin and t_begin together with the
# cosmology if enable_Hubble == True. The functions t(a), a(t) and H(a)
# will also be tabulated and stored in the module namespace in the
# form of spline_a_t, spline_t_a and spline_a_H.
@cython.pheader(# Locals
                H_values='double[::1]',
                a_begin_correct='double',
                a_values='double[::1]',
                background='dict',
                t_values='double[::1]',
                t_begin_correct='double',
                )
def initiate_time():
    global spline_a_t, spline_t_a, spline_a_H
    if enable_Hubble:
        # Hubble expansion enabled.
        # If CLASS should be used to compute the evolution of
        # the background throughout time, run CLASS now.
        if enable_class:
            # Ideally we would call CLASS via compute_cosmo from the
            # linear module, as this would preserve all results for any
            # future needs. However, the linear module imports functions
            # from this module (integration), so importing from
            # linear would create a cyclic import. Instead we do the
            # call ourselves via the "raw" call_class function from the
            # commons module.
            # Note that only the master have access to the results
            # from the CLASS computation.
            cosmo = call_class()
            background = cosmo.get_background()
            # What we need to store is the cosmic time t and the Hubble
            # parametert H, both as functions of the scale factor a.
            # We do this by defining global Spline objects.
            a_values = smart_mpi(1/(background['z'] + 1)                      , 0, mpifun='bcast')
            t_values = smart_mpi(background['proper time [Gyr]']*units.Gyr    , 1, mpifun='bcast') 
            H_values = smart_mpi(background['H [1/Mpc]']*light_speed/units.Mpc, 2, mpifun='bcast') 
            spline_a_t = Spline(a_values, t_values)
            spline_t_a = Spline(t_values, a_values)
            spline_a_H = Spline(a_values, H_values)
        # A specification of initial scale factor or
        # cosmic time is needed.
        if 'a_begin' in user_params:
            # a_begin specified
            if 't_begin' in user_params:
                # t_begin also specified
                masterwarn('Ignoring t_begin = {}*{} becuase enable_Hubble == True '
                           'and a_begin is specified'.format(t_begin, unit_time))
            a_begin_correct = a_begin
            t_begin_correct = cosmic_time(a_begin_correct)
        elif 't_begin' in user_params:
            # a_begin not specified, t_begin specified
            t_begin_correct = t_begin
            a_begin_correct = scale_factor(t_begin_correct)
        else:
            # Neither a_begin nor t_begin is specified.
            # One or the other is needed when enable_Hubble == True.
            abort('No initial scale factor (a_begin) or initial cosmic time (t_begin) specified. '
                  'A specification of one or the other is needed when enable_Hubble == True.')
    else:
        # Hubble expansion disabled
        t_begin_correct = t_begin
        # Values of the scale factor (and therefore a_begin)
        # are meaningless. Set a_begin to unity,
        # effectively ignoring its existence.
        a_begin_correct = 1
        if 'a_begin' in user_params:
            masterwarn('Ignoring a_begin = {} because enable_Hubble == False'.format(a_begin))
    # Now t_begin_correct and a_begin_correct are defined and store
    # the actual values of the initial time and scale factor.
    # Assign these correct values to the corresponding
    # universal variables.
    universals.t_begin = t_begin_correct
    universals.a_begin = a_begin_correct
    universals.z_begin = 1/a_begin_correct - 1
    # Initiate the current universal time and scale factor
    universals.t = t_begin_correct
    universals.a = a_begin_correct
    
    

# Global Spline objects defined by initiate_time
cython.declare(spline_a_t='Spline', spline_t_a='Spline', spline_a_H='Spline')
spline_a_t = None
spline_t_a = None
spline_a_H = None

# Function pointer types used in this module
pxd = """
ctypedef double (*func_d_dd) (double, double)
"""
