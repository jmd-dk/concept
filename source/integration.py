# Import everything from the commons module.
# In the .pyx file, this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    pass
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    """


# The Friedmann equation, used to integrate
# the scale factor forwards in time
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(t='double',
               a='double',
               )
@cython.returns('double')
def ȧ(t, a):
    return a*H0*sqrt(Ωm/(a**3 +  machine_ϵ) + ΩΛ)

# Function for solving ODEs of the form ḟ(t, f)
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               ḟ='func_d_dd',
               f_start='double',
               t_start='double',
               t_end='double',
               δ='double',
               ϵ='double',
               save_intermediate='bint',
               # Locals
               error='double',
               f='double',
               f4='double',
               f5='double',
               h='double',
               h_max='double',
               i='int',
               k1='double',
               k2='double',
               k3='double',
               k4='double',
               k5='double',
               k6='double',
               tolerence='double',
               Δt='double',
               )
@cython.returns('double')
def rkf45(ḟ, f_start, t_start, t_end, δ, ϵ, save_intermediate):
    """ḟ(t, f) is the derivative of f with respect to t. Initial values
    are given by f_start and t_start. ḟ will be integrated from t_start
    to t_end. That is, the returned value is f(t_end). The absolute and
    relative accuracies are given by δ, ϵ. If save_intermediate is True,
    intermediate values optained during the integration will be kept in
    t_cum, f_cum.
    """
    global alloc_cum, f_cum, f_cum_mw, integrand_cum, integrand_cum_mw, size_cum, t_cum, t_cum_mw
    # The maximum step size
    Δt = t_end - t_start
    h_max = Δt/10
    # Initial values
    h = h_max*ϵ
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
        f5 = f     + b1*k1            +  b3*k3 +  b4*k4 +  b5*k5 + b6*k6
        f4 = f     + d1*k1            +  d3*k3 +  d4*k4 +  d5*k5
        # The error estimate
        error = abs(f5 - f4) + machine_ϵ
        # The local tolerence
        tolerence = (ϵ*abs(f5) + δ)*sqrt(h/Δt)
        if error < tolerence:
            # Step accepted
            t += h
            f = f5
            # Save intermediate t and f values
            if save_intermediate:
                t_cum[i] = t
                f_cum[i] = f
                i += 1
                # If necessary, t_cum and f_cum get resized (doubled)
                if i == alloc_cum:
                    alloc_cum *= 2
                    t_cum = realloc(t_cum, alloc_cum*sizeof('double'))
                    f_cum = realloc(f_cum, alloc_cum*sizeof('double'))
                    integrand_cum = realloc(integrand_cum, alloc_cum*sizeof('double'))
                    t_cum_mw = cast(t_cum, 'double[:alloc_cum]')
                    f_cum_mw = cast(f_cum, 'double[:alloc_cum]')
                    integrand_cum_mw = cast(integrand_cum, 'double[:alloc_cum]')
        # Updating step size
        h *= 0.95*(tolerence/error)**0.25
        if h > h_max:
            h = h_max
        if t + h > t_end:
            h = t_end - t
    if save_intermediate:
        size_cum = i
    return f

# Function for updating the scale factor
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               a='double',
               t='double',
               Δt='double',
               )
@cython.returns('double')
def expand(a, t, Δt):
    """Integrates the Friedmann equation from t to t + Δt,
    where the scale factor at time t is given by a. Returns a(t + Δt).
    """
    return rkf45(ȧ, a, t, t + Δt, δ=1e-9, ϵ=1e-9, save_intermediate=True)

# Function for calculating integrals of the sort ∫_t^(t + Δt) a^power dt
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               power='int',
               # Locals
               acc='gsl_interp_accel*',
               integral='double',
               spline='gsl_spline*',
               )
@cython.returns('double')
def scalefactor_integral(power):
    """This function returns the factor ∫_t^(t + Δt) a^power dt used
    in the drift (power = -2) and kick (power = -1) operations.
    It is important that the expand function expand(a, t, Δt) has been
    called prior to calling this function, as expand generates the a
    values needed in the integration. You can call this function multiple
    times (and with different values of power) without calling expand
    in between.
    """
    # If expand has been called as it should, f_cum now stores
    # accumulated values of a. Compute the integrand a^power.
    global integrand_cum
    if power == -1:
        for i in range(size_cum):
            integrand_cum[i] = 1/f_cum[i]
    elif power == -2:
        for i in range(size_cum):
            integrand_cum[i] = 1/f_cum[i]**2
    # Integrate integrand_cum in pure Python or Cython
    if not cython.compiled:
        integral = trapz(integrand_cum_mw[:size_cum], t_cum_mw[:size_cum])
    else:
        # Allocate an interpolation accelerator and a cubic spline object
        acc = gsl_interp_accel_alloc()
        spline = gsl_spline_alloc(gsl_interp_cspline, size_cum)
        # Initialize spline
        gsl_spline_init(spline, t_cum, integrand_cum, size_cum)
        # Integrate the splined function
        integral = gsl_spline_eval_integ(spline, t_cum[0], t_cum[size_cum - 1], acc)
        # Free the accelerator and the spline object
        gsl_spline_free(spline)
        gsl_interp_accel_free(acc)
    return integral

# Function for computing the cosmic time t at some given scale factor a
@cython.cfunc
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.locals(# Arguments
               a='double',
               a_lower='double',
               t_lower='double',
               t_upper='double',
               # Locals
               a_test='double',
               t='double',
               t_lowest='double',
               )
@cython.returns('double')
def cosmic_time(a, a_lower=machine_ϵ, t_lower=machine_ϵ, t_upper=20*units.Gyr):
    """Given lower and upper bounds on the cosmic time, t_lower and t_upper,
    and the scale factor at time t_lower, a_lower, this function finds
    the future time at which the scale factor will have the value a.
    """
    # Saves a copy of t_lower, the time at which the scale factor
    # had a value of a_lower
    t_lowest = t_lower
    # Compute the cosmic time at which the scale factor had the value a,
    # using a binary search
    a_test = t = -1
    while abs(a_test - a) > machine_ϵ and (t_upper - t_lower) > machine_ϵ:
        t = (t_upper + t_lower)/2
        a_test = rkf45(ȧ, a_lower, t_lowest, t, δ=1e-9, ϵ=1e-9, save_intermediate=False)
        if a_test > a:
            t_upper = t
        else:
            t_lower = t
    return t

# Initialize the Butcher tableau for the Runge–Kutta–Fehlberg
# method at import time
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
c2 = 1.0/4;   a21 = 1.0/4
c3 = 3.0/8;   a31 = 3.0/32;      a32 = 9.0/32
c4 = 12.0/13; a41 = 1932.0/2197; a42 = -7200.0/2197; a43 = 7296.0/2197
c5 = 1;       a51 = 439.0/216;   a52 = -8;           a53 = 3680.0/513;   a54= -845.0/4104
c6 = 1.0/2;   a61 = -8.0/27;     a62 = 2;            a63 = -3544.0/2565; a64 = 1859.0/4104;  a65 = -11.0/40
b1 = 16.0/135;                                        b3 = 6656.0/12825;  b4 = 28561.0/56430; b5 = -9.0/50;  b6 = 2.0/55
d1 = 25.0/216;                                        d3 = 1408.0/2565;   d4 = 2197.0/4104;   d5 = -1.0/5
# Allocate t_cum, f_cum and integrand_cum at import time.
# t_cum and f_cum are used to store intermediate values of t, f,
# in the Runge–Kutta–Fehlberg method. integrand_cum stores the associated
# values of the integrand in ∫_t^(t + Δt) 1/a dt and ∫_t^(t + Δt) 1/a^2 dt.
cython.declare(alloc_cum='int',
               f_cum='double*',
               f_cum_mw='double[::1]',
               integrand_cum='double*',
               integrand_cum_mw='double[::1]',
               size_cum='int',
               t_cum='double*',
               t_cum_mw='double[::1]'
               )
alloc_cum = 100
size_cum = 0
t_cum = malloc(alloc_cum*sizeof('double'))
f_cum = malloc(alloc_cum*sizeof('double'))
integrand_cum = malloc(alloc_cum*sizeof('double'))
t_cum_mw = cast(t_cum, 'double[:alloc_cum]')
f_cum_mw = cast(f_cum, 'double[:alloc_cum]')
integrand_cum_mw = cast(integrand_cum, 'double[:alloc_cum]')
