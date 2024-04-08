# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2024 Jeppe Mosgaard Dakin.
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
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport(
    'from communication import '
    '    get_buffer,           '
    '    smart_mpi,            '
)



# Class used for 1D, cubic interpolation.
# GSL is used when in compiled mode, while SciPy is used in
# pure Python mode. The behaviour is identical.
@cython.cclass
class Spline:
    size_min = 3

    # Initialisation method
    @cython.header(
        # Arguments
        x='double[::1]',
        y='double[::1]',
        name=str,
        logx='bint',
        logy='bint',
        # Locals
        contains_negative='bint',
        contains_positive='bint',
        contains_zero='bint',
        i='Py_ssize_t',
    )
    def __init__(self, x, y, name='', *, logx=False, logy=False):
        # Here x and y = y(x) are the tabulated data.
        # The values in x must be in increasing order.
        # If logx (logy) is True, the log will be taken of the x (y)
        # data before it is splined.
        #
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Spline type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        double[::1] x
        double[::1] y
        str name
        bint logx
        bint logy
        bint negativey
        double xmin
        double xmax
        double abs_tol_min
        double abs_tol_max
        AcceleratorContainer acc_container
        gsl_interp_accel* acc
        gsl_spline* spline
        """
        # The name is only for clearer error messages
        self.name = name
        # The size of the data
        if x.shape[0] != y.shape[0]:
            abort(
                f'Spline "{self.name}": '
                f'A Spline object cannot be tabulated using arrays of different lengths. '
                f'Arrays of lengths {x.shape[0]} and {y.shape[0]} were passed.'
            )
        if x.shape[0] < self.size_min:
            abort(
                f'Spline "{self.name}": '
                f'Too few tabulated values ({x.shape[0]}) were given for '
                f'cubic spline interpolation. At least {self.size_min} is needed.'
            )
        # Take the log, if requested.
        # Note that a copy of the input data is used,
        # we do not mutate the input data.
        self.logx, self.logy = logx, logy
        if self.logx:
            for i in range(x.shape[0]):
                if x[i] <= 0:
                    self.logx = False
                    warn(
                        f'Spline "{self.name}": '
                        f'Could not take log of spline x data '
                        f'as it contains non-positive values'
                    )
                    break
        if self.logx:
            x = np.log(x)
        self.negativey = False  # only relevant for logy
        if self.logy:
            contains_negative = False
            contains_zero     = False
            contains_positive = False
            for i in range(y.shape[0]):
                contains_negative |= (y[i] <  0)
                contains_zero     |= (y[i] == 0)
                contains_positive |= (y[i] >  0)
            if contains_zero:
                self.logy = False
                warn(
                    f'Spline "{self.name}": '
                    f'Could not take log of spline y data '
                    f'as it contains a zero'
                )
            elif contains_negative and contains_positive:
                self.logy = False
                warn(
                    f'Spline "{self.name}": '
                    f'Could not take log of spline y data '
                    f'as it contains both positive and negative values'
                )
            elif contains_negative:
                # Purely negative
                self.negativey = True
                y = -asarray(y)
        if self.logy:
            y = np.log(y)
        # Check that the passed x values are strictly increasing and
        # remove doppelgänger points. Note that this does not mutate the
        # passed x and y arrays, and so the new x and y refer to
        # different underlying data.
        x, y = remove_doppelgängers(x, y)
        # Store a copy of the non-logged, non-negated,
        # doppelgänger free data.
        self.x =                        np.exp(x) if self.logx else asarray(x).copy()
        self.y = (1 - 2*self.negativey)*np.exp(y) if self.logy else asarray(y).copy()
        # Store meta data
        self.xmin = x[0]
        self.xmax = x[x.shape[0] - 1]
        abs_tol = 1e-9*(self.xmax - self.xmin) + machine_ϵ
        self.abs_tol_min = abs_tol + 0.5*(x[1] - self.xmin)
        self.abs_tol_max = abs_tol + 0.5*(self.xmax - x[x.shape[0] - 2])
        # Use SciPy in pure Python and GSL when compiled
        if not cython.compiled:
            # Initialise the spline.
            # Here we simply overwrite the spline attribute.
            # The boundary condition type is set to 'natural'
            # as to match the default of GSL.
            import scipy.interpolate
            self.spline = scipy.interpolate.CubicSpline(
                asarray(x).copy(), asarray(y).copy(),
                bc_type='natural',
            )
        else:
            # Allocate and initialise cubic spline object
            self.spline = gsl_spline_alloc(gsl_interp_cspline, x.shape[0])
            gsl_spline_init(self.spline, cython.address(x[:]), cython.address(y[:]), x.shape[0])
            # Fetch existing interpolation accelerator over x
            # or create a new one if missing.
            self.acc_container = fetch_acc(x)
            self.acc = self.acc_container.acc

    # Method for doing spline evaluation
    @cython.pheader(
        # Arguments
        x_in='double',
        # Locals
        x='double',
        y='double',
        returns='double',
    )
    def eval(self, x_in):
        x = log(x_in) if self.logx else x_in
        # Check that x is within the interpolation interval
        x = self.in_interval(x, 'interpolate to')
        # Use SciPy in pure Python and GSL when compiled
        if not cython.compiled:
            y = float(self.spline(x))
        else:
            y = gsl_spline_eval(self.spline, x, self.acc)
        # Undo the log
        if self.logy:
            y = exp(y)
            # Undo the negation
            if self.negativey:
                y *= -1
        return y

    # Method for doing spline derivative evaluation
    @cython.pheader(
        # Arguments
        x_in='double',
        # Locals
        x='double',
        ẏ='double',
        returns='double',
    )
    def eval_deriv(self, x_in):
        x = log(x_in) if self.logx else x_in
        # Check that x is within the interpolation interval
        x = self.in_interval(x, 'differentiate at')
        # Use SciPy in pure Python and GSL when compiled
        if not cython.compiled:
            ẏ = self.spline(x, 1)
        else:
            ẏ = gsl_spline_eval_deriv(self.spline, x, self.acc)
        # Undo the log
        if self.logx and self.logy:
            # ∂ₓy(x) = y(x)/x*∂ₗₙ₍ₓ₎ln(y(x))
            ẏ *= self.eval(x_in)/x_in
        elif self.logx:
            # ∂ₓy(x) = x⁻¹*∂ₗₙ₍ₓ₎y(x)
            ẏ /= x_in
        elif self.logy:
            # ∂ₓy(x) = y(x)*∂ₓln(y(x))
            ẏ *= self.eval(x_in)
        # Undo the negation
        if self.negativey:
            ẏ *= -1
        return ẏ

    # Method for computing the definite integral over some
    # interval [a, b] of the splined function.
    @cython.pheader(
        # Arguments
        a='double',
        b='double',
        # Locals
        sign_flip='bint',
        ᔑ='double',
        returns='double',
    )
    def integrate(self, a, b):
        if self.logx or self.logy:
            abort(f'Spline "{self.name}": Spline integration not possible for logged data')
        # Check that the supplied limits are
        # within the interpolation interval.
        a = self.in_interval(a, 'integrate from')
        b = self.in_interval(b, 'integrate to')
        # The function gsl_spline_eval_integ fails for a > b.
        # Take care of this manually by switching a and b and note
        # the sign change.
        sign_flip = False
        if a > b:
            sign_flip = True
            a, b = b, a
        # Use SciPy in pure Python and GSL when compiled
        if not cython.compiled:
            ᔑ = self.spline.integrate(a, b)
        else:
            ᔑ = gsl_spline_eval_integ(self.spline, a, b, self.acc)
        # Remember the sign change for a > b
        if sign_flip:
            ᔑ *= -1
        # Undo the negation
        if self.negativey:
            ᔑ *= -1
        return ᔑ

    # Method for checking whether a given number
    # is within the tabulated interval.
    @cython.header(
        # Arguments
        x='double',
        action=str,
        # Locals
        returns='double',
    )
    def in_interval(self, x, action='interpolate to'):
        # Check that the supplied x is within the
        # interpolation interval. If it is just barely outside of it,
        # move it to the boundary.
        if x < self.xmin:
            if x > self.xmin - self.abs_tol_min:
                x = self.xmin
            else:
                abort(
                    f'Spline "{self.name}": '
                    f'Could not {action} {x} because it is outside the tabulated interval '
                    f'[{self.xmin}, {self.xmax}]'
                )
        elif x > self.xmax:
            if x < self.xmax + self.abs_tol_max:
                x = self.xmax
            else:
                abort(
                    f'Spline "{self.name}": '
                    f'Could not {action} {x} because it is outside the tabulated interval '
                    f'[{self.xmin}, {self.xmax}]'
                )
        return x

    # This method is automatically called when a Spline instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        # Free the spline object
        gsl_spline_free(self.spline)
        # Decrease accelerator counter and remove accelerator from
        # global store if the count reach zero. No references should
        # then remain to the accelerator and it will be cleaned up.
        digest = self.acc_container.digest
        acc_counter[digest] -= 1
        if acc_counter[digest] == 0:
            acc_store.pop(digest, None)
            acc_counter.pop(digest, None)

    # String representation
    def __repr__(self):
        return f'<Spline "{self.name}">'
    def __str__(self):
        return self.__repr__()

# Container class for storing GSL interpolation accelerator objects
@cython.cclass
class AcceleratorContainer:
    @cython.header(x='double[::1]')
    def __init__(self, x):
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the AcceleratorContainer type.
        # It will get picked up by the pyxpp script
        # and included in the .pxd file.
        """
        str digest
        gsl_interp_accel* acc
        """
        # Allocate new accelerator
        self.acc = gsl_interp_accel_alloc()
        # Add instance to global store
        self.digest = self.hash_interpolation_domain(x)
        if self.digest in acc_store:
            abort(f'AcceleratorContainer {self.digest} instantiated twice')
        acc_store[self.digest] = self
        acc_counter[self.digest] += 1
    @staticmethod
    def hash_interpolation_domain(x):
        return hashlib.sha1(asarray(x)).hexdigest()
    def __dealloc__(self):
        gsl_interp_accel_free(self.acc)

# Storage and retrieval mechanism for reusage
# of AcceleratorContainer instances.
@cython.header(
    # Arguments
    x='double[::1]',
    # Locals
    acc_container='AcceleratorContainer',
    digest=str,
    returns='AcceleratorContainer'
)
def fetch_acc(x):
    digest = AcceleratorContainer.hash_interpolation_domain(x)
    acc_container = acc_store.get(digest)
    if acc_container is None:
        # Instantiate new accelerator container
        acc_container = AcceleratorContainer(x)
    else:
        # Reuse old accelerator; bump its count
        acc_counter[digest] += 1
    return acc_container
acc_store = {}
acc_counter = collections.Counter()

# Function for cleaning up arrays of points possibly containing
# duplicate points.
@cython.pheader(
    # Arguments
    xs='object',  # double[::1] or tuple or list of double[::1]
    y='double[::1]',
    rel_tol='double',
    copy='bint',
    # Locals
    accepted_indices='Py_ssize_t[::1]',
    i='Py_ssize_t',
    index='Py_ssize_t',
    j='Py_ssize_t',
    multiple_x_passed='bint',
    size='Py_ssize_t',
    sizes=tuple,
    x='double[::1]',
    x_arrays=list,
    x_cleaned='double[::1]',
    x_cleaned_arrays=list,
    x_prev='double',
    xdiff='double',
    xdiff_prev='double',
    y_cleaned='double[::1]',
    returns=tuple,
)
def remove_doppelgängers(xs, y, rel_tol=1e-1, copy=False):
    """Given arrays of x and y values, this function checks for
    doppelgängers in the x values, meaning consecutive x values that are
    exactly or very nearly equal. New arrays with doppelgängers
    reduced to single points will be returned.
    You may have as many x arrays as you like.
    The relative tolerance used in determining doppelgängers is given
    by rel_tol.
    For this function to work, the x values must be in increasing order.
    The passed x and y will not be modified. Note that this function
    reuses the same buffer for all returned arrays, meaning that if you
    call this function multiple times, the returned arrays will point to
    the same underlying data. If you want the returned arrays to be
    freshly allocated, use copy=True.
    """
    global accepted_indices_ptr, accepted_indices_size
    # Pack xs into a list of x arrays
    if isinstance(xs, (tuple, list)):
        x_arrays = list(xs)
        multiple_x_passed = True
    else:
        x_arrays = [xs]
        multiple_x_passed = False
    # All the x arrays should have the same size,
    # and the y array must be at least as large.
    sizes = tuple({x.shape[0] for x in x_arrays})
    if len(sizes) > 1:
        abort(
            f'The x arrays passed to remove_doppelgängers() do not all have '
            f'the same size {sizes}'
        )
    size = sizes[0]
    if y.shape[0] < size:
        abort(
            f'The y array passed to remove_doppelgängers() must have at least the same number '
            f'of elements as that of the passed x array(s) ({size}), '
            f'but it only has {y.shape[0]} elements'
        )
    y = y[:size]
    # Fetch buffers for storing the cleaned up versions of the arrays
    x_cleaned_arrays = [get_buffer(size, f'remove_doppelgängers (x_cleaned {i})')
        for i in range(len(x_arrays))
    ]
    y_cleaned = get_buffer(size, 'remove_doppelgängers (y_cleaned)')
    # No cleanup is required if a single
    # or no points exist in the passed data.
    if size < 2:
        for x, x_cleaned in zip(x_arrays, x_cleaned_arrays):
            x_cleaned[:] = x
        y_cleaned[:] = y
        if multiple_x_passed:
            if copy:
                return ([asarray(x_cleaned).copy() for x_cleaned in x_cleaned_arrays],
                    asarray(y_cleaned).copy())
            else:
                return [asarray(x_cleaned) for x_cleaned in x_cleaned_arrays], asarray(y_cleaned)
        else:
            if copy:
                return asarray(x_cleaned).copy(), asarray(y_cleaned).copy()
            else:
                return asarray(x_cleaned), asarray(y_cleaned)
    # Check that the x values are in increasing order
    for x in x_arrays:
        x_prev = x[0]
        for i in range(1, size):
            if ℝ[x[i]] < x_prev:
                abort(
                    'The values in (one of) the x array(s) passed to remove_doppelgängers() '
                    'are not in increasing order'
                )
            x_prev = ℝ[x[i]]
    # Loop from right to left, checking for doppelgängers
    # using the first passed x array.
    if accepted_indices_size < size:
        accepted_indices_ptr = realloc(accepted_indices_ptr, size*sizeof('Py_ssize_t'))
        accepted_indices_size = size
    accepted_indices = cast(accepted_indices_ptr, 'Py_ssize_t[:size]')
    for index in range(size - 1, size - 3, -1):
        accepted_indices[index] = index
    x = x_arrays[0]
    x_prev = x[index]
    xdiff_prev = x[size - 1] - x_prev
    index -= 1
    for i in range(size - 3, -1, -1):
        xdiff = x_prev - x[i]
        if xdiff > rel_tol*xdiff_prev:
            # Accept this point
            accepted_indices[index] = i
            index -= 1
            x_prev = x[i]
            xdiff_prev = xdiff
    # Copy accepted points to the cleaned arrays
    accepted_indices = accepted_indices[index+1 :]
    size = accepted_indices.shape[0]
    for j, (x, x_cleaned) in enumerate(zip(x_arrays, x_cleaned_arrays)):
        for i in range(size):
            x_cleaned[i] = x[accepted_indices[i]]
        x_cleaned_arrays[j] = x_cleaned[:size]
    for i in range(size):
        y_cleaned[i] = y[accepted_indices[i]]
    y_cleaned = y_cleaned[:size]
    # Always include the first point
    for x, x_cleaned in zip(x_arrays, x_cleaned_arrays):
        x_cleaned[0] = x[0]
    y_cleaned[0] = y[0]
    # Let x and y point to the cleaned up versions of these arrays
    for i, x_cleaned in enumerate(x_cleaned_arrays):
        x_arrays[i] = x_cleaned
    y = y_cleaned
    # Loop from left to right, checking for doppelgängers
    # using the first passed x array.
    for index in range(2):
        accepted_indices[index] = index
    x = x_arrays[0]
    x_prev = x[1]
    xdiff_prev = x_prev - x[0]
    index += 1
    for i in range(2, x.shape[0]):
        xdiff =  ℝ[x[i]] - x_prev
        if xdiff > rel_tol*xdiff_prev:
            # Accept this point
            accepted_indices[index] = i
            index += 1
            x_prev = ℝ[x[i]]
            xdiff_prev = xdiff
    # Copy accepted points to the cleaned arrays
    size = index
    accepted_indices = accepted_indices[:size]
    for j, x_cleaned in enumerate(x_cleaned_arrays):
        for i in range(size):
            x_cleaned[i] = x_cleaned[accepted_indices[i]]
        x_cleaned_arrays[j] = x_cleaned[:size]
    for i in range(size):
        y_cleaned[i] = y_cleaned[accepted_indices[i]]
    y_cleaned = y_cleaned[:size]
    # Always include the last point
    for x, x_cleaned in zip(x_arrays, x_cleaned_arrays):
        x_cleaned[ℤ[size - 1]] = x[ℤ[y.shape[0] - 1]]
    y_cleaned[ℤ[size - 1]] = y[ℤ[y.shape[0] - 1]]
    # Return the cleaned arrays
    if multiple_x_passed:
        if copy:
            return ([asarray(x_cleaned).copy() for x_cleaned in x_cleaned_arrays],
                asarray(y_cleaned).copy())
        else:
            return [asarray(x_cleaned) for x_cleaned in x_cleaned_arrays], asarray(y_cleaned)
    else:
        if copy:
            return asarray(x_cleaned_arrays[0]).copy(), asarray(y_cleaned).copy()
        else:
            return asarray(x_cleaned_arrays[0]), asarray(y_cleaned)
# Pointer used by the remove_doppelgängers function
cython.declare(accepted_indices_ptr='Py_ssize_t*', accepted_indices_size='Py_ssize_t')
accepted_indices_size = 3
accepted_indices_ptr = malloc(accepted_indices_size*sizeof('Py_ssize_t'))

# This function implements the Hubble parameter H(a) = ȧ/a,
# The Hubble parameter is only ever written here. Every time the Hubble
# parameter is used in the code, this function should be called.
@cython.header(
    # Arguments
    a='double',
    # Locals
    spline='Spline',
    ΩΛ='double',
    returns='double',
)
def hubble(a=-1):
    if not enable_Hubble:
        return 0
    if a == -1:
        a = universals.a
    # When using CLASS for the background computation, simply look up
    # H(a) in the tabulated data. The largest tabulated a is 1, but it
    # may happen that this function gets called with an a that is
    # sightly larger. In this case, compute H directly via the simple
    # Friedmann equation, just as when CLASS is not being used for the
    # background computation.
    if enable_class_background and a <= 1:
        spline = temporal_splines.a_H
        if spline is None:
            abort('The function H(a) has not been tabulated. Have you called init_time?')
        return spline.eval(a)
    # CLASS not enabled. Assume that the universe is flat
    # and consists purely of matter and Λ.
    # If we write a**(±3) directly,
    # -ffast-math degrades the power computation.
    ΩΛ = 1 - Ωm
    return H0*sqrt(Ωm*np.power(a, -3) + ΩΛ)

# Function for computing the scale factor a,
# given a value for the cosmic time.
@cython.header(
    # Arguments
    t='double',
    # Locals
    spline='Spline',
    returns='double',
)
def scale_factor(t=-1):
    if not enable_Hubble:
        return 1
    if t == -1:
        t = universals.t
    spline = temporal_splines.t_a
    if spline is None:
        abort('The function a(t) has not been tabulated. Have you called init_time?')
    return spline.eval(t)

# Function for computing the cosmic time t,
# given a value for the scale factor.
@cython.pheader(
    # Arguments
    a='double',
    # Locals
    spline='Spline',
    returns='double',
)
def cosmic_time(a=-1):
    if not enable_Hubble:
        abort(
            'The cosmic_time() function was called. '
            'A mapping from a to t is only meaningful when Hubble expansion is enabled.'
        )
    if a == -1:
        a = universals.a
    spline = temporal_splines.a_t
    if spline is None:
        abort('The function t(a) has not been tabulated. Have you called init_time?')
    return spline.eval(a)

# Function returning the proper time differentiated scale factor,
# given a value for the scale factor.
@cython.header(
    # Arguments
    a='double',
    # Locals
    returns='double',
)
def ȧ(a=-1):
    if not enable_Hubble:
        return 0
    if a == -1:
        a = universals.a
    return a*hubble(a)

# Function returning the second proper time derivative of the
# scale factor, given a value for the scale factor.
@cython.header(
    # Arguments
    a='double',
    # Locals
    returns='double',
)
def ä(a=-1):
    if not enable_Hubble:
        return 0
    if a == -1:
        a = universals.a
    return a*(hubble(a)**2 + Ḣ(a))

# Function returning the proper time differentiated Hubble parameter,
# given a value for the scale factor.
@cython.header(
    # Arguments
    a='double',
    # Locals
    spline='Spline',
    returns='double',
)
def Ḣ(a=-1):
    if not enable_Hubble:
        return 0
    if a == -1:
        a = universals.a
    spline = temporal_splines.a_H
    if spline is None:
        abort('The function H(a) has not been tabulated. Have you called init_time?')
    return ȧ(a)*spline.eval_deriv(a)

# Function for calculating integrals of the sort
# ᔑ_t_start^t_end integrand(a(t)) dt.
@cython.header(
    # Arguments
    key=object,  # str or tuple
    t_start='double',
    t_eval='double',
    all_components=list,
    # Locals
    a='double',
    a_tab_spline='double[::1]',
    component='Component',
    component_0='Component',
    component_1='Component',
    component_name=str,
    component_names=list,
    components=list,
    i='Py_ssize_t',
    integrand=str,
    integrand_tab_spline='double[::1]',
    size='Py_ssize_t',
    spline='Spline',
    t='double',
    t_tab_spline='double[::1]',
    w_eff='double',
    w_eff_0='double',
    w_eff_1='double',
    returns='double',
)
def scalefactor_integral(key, t_start, t_end, all_components):
    """This function returns the integral
    ᔑ_t_start^t_end integrand(a(t)) dt.
    The integrand is passed as the key argument, which may be a string
    (e.g. 'a**(-1)') or a tuple in the format (string, component.name),
    (string, component_0.name, component_1.name) etc., where again the
    first string is really the integrand. The tuple form is used when
    the integrand is component specific, e.g. 'a**(-3*w_eff)'.
    """
    if t_start == t_end:
        return 0
    # Extract the integrand from the passed key
    components = []
    if 𝔹[isinstance(key, str)]:
        # Global integrand
        integrand, component = key, None
    else:
        # Component integrand
        integrand, *component_names = key
        for component_name in component_names:
            for component in all_components:
                if component.name == component_name:
                    components.append(component)
                    break
    # Lookup spline and use it for the integration
    spline = spline_t_integrands.get(key)
    if spline is not None:
        return spline.integrate(t_start, t_end)
    # A spline has yet to be made for this integrand.
    # Get tabulated a(t).
    if enable_Hubble:
        for component in components:
            if component is not None and component.w_eff_type != 'constant':
                a_tab_spline = component.w_eff_spline.x
                t_tab_spline = asarray([temporal_splines.a_t.eval(a) for a in a_tab_spline])
                break
        else:
            a_tab_spline = temporal_splines.a_t.x
            t_tab_spline = temporal_splines.a_t.y
    else:
        # Construct dummy a(t) table
        t_tab_spline = linspace(t_start, t_end, Spline.size_min)
        a_tab_spline = asarray([scale_factor(t) for t in t_tab_spline])
    size = t_tab_spline.shape[0]
    integrand_tab_spline = empty(size, dtype=C2np['double'])
    # Do the tabulation
    for i in range(size):
        a = a_tab_spline[i]
        t = t_tab_spline[i]
        with unswitch:
            if 𝔹[isinstance(key, str)]:
                # Global integrands
                with unswitch:
                    if integrand == '1' or integrand == '':
                        integrand_tab_spline[i] = 1
                    elif integrand == 'a**2':
                        integrand_tab_spline[i] = a**2
                    elif integrand == 'a**(-1)':
                        integrand_tab_spline[i] = 1/a
                    elif integrand == 'a**(-2)':
                        integrand_tab_spline[i] = 1/a**2
                    elif integrand == 'ȧ/a':
                        integrand_tab_spline[i] = hubble(a)
                    elif master:
                        abort(
                            f'The scale factor integral with "{integrand}" as the integrand '
                            f'is not implemented'
                        )
            elif ℤ[len(key)] == 2:
                # Single-component integrands
                with unswitch:
                    if integrand == 'a**(-3*w_eff)':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(-3*w_eff)
                    elif integrand == 'a**(-3*(1+w_eff))':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(-3*(1 + w_eff))
                    elif integrand == 'a**(-3*w_eff-1)':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(-3*w_eff - 1)
                    elif integrand == 'a**(3*w_eff-2)':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(3*w_eff - 2)
                    elif integrand == 'a**(-3*w_eff)*Γ/H':
                        with unswitch:
                            if enable_Hubble:
                                w_eff = component.w_eff(t=t, a=a)
                                integrand_tab_spline[i] = a**(-3*w_eff)*component.Γ(a)/hubble(a)
                            else:
                                integrand_tab_spline[i] = component.Γ(a)
                    elif master:
                        abort(
                            f'The scale factor integral with "{integrand}" as the integrand '
                            f'is not implemented'
                        )
            elif ℤ[len(key)] == 3:
                # Two-component integrands
                component_0, component_1 = components
                with unswitch:
                    if integrand == 'a**(-3*w_eff₀-3*w_eff₁-1)':
                        w_eff_0 = component_0.w_eff(t=t, a=a)
                        w_eff_1 = component_1.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(-3*w_eff_0 - 3*w_eff_1 - 1)
                    elif master:
                        abort(
                            f'The scale factor integral with "{integrand}" as the integrand '
                            f'is not implemented'
                        )
            else:
                abort(f'scalefactor_integral(): Invalid length ({len(key)}) of key {key}')
    # Create and store the spline
    spline = Spline(t_tab_spline, integrand_tab_spline, integrand)
    if enable_Hubble:
        spline_t_integrands[key] = spline
    # Perform the integration
    return spline.integrate(t_start, t_end)
# Global dict of Spline objects defined by scalefactor_integral()
cython.declare(spline_t_integrands=dict)
spline_t_integrands = {}

# Function which sets the value of universals.a and universals.t
# based on the user parameters a_begin and t_begin together with the
# cosmology if enable_Hubble is True. The functions t(a), a(t) and H(a)
# will also be tabulated and stored as spline attributes on the module
# level temporal_splines object. If enable_class_background is False,
# D(a), f(a), D2(a), f2(a), D3a(a), f3a(a), D3b(a), f3b(a),
# D3c(a), f3c(a), will be added as well.
@cython.pheader(
    # Arguments
    reinitialize='bint',
    # Locals
    a_begin_correct='double',
    a_today='double',
    a_values='double[::1]',
    background=dict,
    cosmo=object,  # classy.Class
    D2_values='double[::1]',
    D3a_values='double[::1]',
    D3b_values='double[::1]',
    D3c_values='double[::1]',
    D_values='double[::1]',
    f2_values='double[::1]',
    f3a_values='double[::1]',
    f3b_values='double[::1]',
    f3c_values='double[::1]',
    f_values='double[::1]',
    H_values='double[::1]',
    n_bg='Py_ssize_t',
    t_begin_correct='double',
    t_values='double[::1]',
    returns='void',
)
def init_time(reinitialize=False):
    if temporal_splines.initialized and not reinitialize:
        return
    a_today = 1
    if enable_Hubble:
        # Hubble expansion enabled.
        # If CLASS should be used to compute the evolution of the
        # background throughout time, we run CLASS now.
        # Otherwise we solve the simplified background implemented
        # by hubble() ourselves.
        a_values = t_values = H_values = None
        D_values = f_values = None
        D2_values = f2_values = None
        D3a_values = f3a_values = None
        D3b_values = f3b_values = None
        D3c_values = f3c_values = None
        if enable_class_background:
            # Ideally we would call CLASS via compute_cosmo() from the
            # linear module, as this would preserve all results for any
            # future needs. However, the linear module imports functions
            # from this module (integration), so importing from
            # linear would create a cyclic import. Instead we do the
            # call ourselves via the "raw" call_class function from the
            # commons module.
            # Note that only the master have access to the results
            # from the CLASS computation.
            cosmo = call_class(class_call_reason='in order to set the cosmic clock')
            background = cosmo.get_background()
            a_values = 1/(background['z'] + 1)
            t_values = background['proper time [Gyr]']*units.Gyr
            H_values = background['H [1/Mpc]']*(light_speed/units.Mpc)
        else:
            # Use the simplified matter + Λ background.
            # Here we futher compute the growth factors (normally
            # computed within CLASS using the full background and
            # handled within the linear module;
            # see e.g. linear.CosmoResults.growth_fac_D()).
            (
                a_values, t_values, H_values,
                D_values, f_values,
                D2_values, f2_values,
                D3a_values, f3a_values,
                D3b_values, f3b_values,
                D3c_values, f3c_values,
            ) = solve_matterΛ_background(a_today)
        # Ensure that the last scale factor and Hubble value
        # are set to their current values.
        if master:
            n_bg = a_values.shape[0]
            if not np.isclose(a_values[n_bg - 1], a_today, rtol=1e-6, atol=0):
                masterwarn(
                    f'Expected the last scale factor value in the '
                    f'tabulated background to be {a_today}, '
                    f'but found {a_values[n_bg - 1]}.'
                )
            a_values[n_bg - 1] = a_today
            if not np.isclose(H_values[n_bg - 1], H0, rtol=1e-6, atol=0):
                unit = units.km/(units.s*units.Mpc)
                masterwarn(
                    f'Expected the last Hubble value in the '
                    f'tabulated background to be {H0/unit} km s⁻¹ Mpc⁻¹, '
                    f'but found {H_values[n_bg - 1]/unit} km s⁻¹ Mpc⁻¹.'
                )
            H_values[n_bg - 1] = H0
        # Communicate results and create splines
        a_values = smart_mpi(a_values, mpifun='bcast').copy()
        t_values = smart_mpi(t_values, mpifun='bcast')
        temporal_splines.a_t = Spline(a_values, t_values, 't(a)', logx=True, logy=True)
        temporal_splines.t_a = Spline(t_values, a_values, 'a(t)', logx=True, logy=True)
        H_values = smart_mpi(H_values, mpifun='bcast')
        temporal_splines.a_H = Spline(a_values, H_values, 'H(a)', logx=True, logy=True)
        if not enable_class_background:
            # Growth factors computed with the simplified background
            D_values = smart_mpi(D_values, mpifun='bcast')
            temporal_splines.a_D = Spline(a_values, D_values, 'D(a)', logx=True, logy=True)
            f_values = smart_mpi(f_values, mpifun='bcast')
            temporal_splines.a_f = Spline(a_values, f_values, 'f(a)', logx=True, logy=False)
            D2_values = smart_mpi(D2_values, mpifun='bcast')
            temporal_splines.a_D2 = Spline(a_values, D2_values, 'D2(a)', logx=True, logy=True)
            f2_values = smart_mpi(f2_values, mpifun='bcast')
            temporal_splines.a_f2 = Spline(a_values, f2_values, 'f2(a)', logx=True, logy=False)
            D3a_values = smart_mpi(D3a_values, mpifun='bcast')
            temporal_splines.a_D3a = Spline(a_values, D3a_values, 'D3a(a)', logx=True, logy=True)
            f3a_values = smart_mpi(f3a_values, mpifun='bcast')
            temporal_splines.a_f3a = Spline(a_values, f3a_values, 'f3a(a)', logx=True, logy=False)
            D3b_values = smart_mpi(D3b_values, mpifun='bcast')
            temporal_splines.a_D3b = Spline(a_values, D3b_values, 'D3b(a)', logx=True, logy=True)
            f3b_values = smart_mpi(f3b_values, mpifun='bcast')
            temporal_splines.a_f3b = Spline(a_values, f3b_values, 'f3b(a)', logx=True, logy=False)
            D3c_values = smart_mpi(D3c_values, mpifun='bcast')
            temporal_splines.a_D3c = Spline(a_values, D3c_values, 'D3c(a)', logx=True, logy=True)
            f3c_values = smart_mpi(f3c_values, mpifun='bcast')
            temporal_splines.a_f3c = Spline(a_values, f3c_values, 'f3c(a)', logx=True, logy=False)
        # A specification of initial scale factor or
        # cosmic time is needed.
        if 'a_begin' in user_params_keys_raw:
            # a_begin specified
            if 't_begin' in user_params_keys_raw:
                # t_begin also specified
                masterwarn(
                    f'Ignoring t_begin = {t_begin}*{unit_time} because '
                    f'enable_Hubble is True and a_begin is specified'
                )
            a_begin_correct = a_begin
            t_begin_correct = cosmic_time(a_begin_correct)
        elif 't_begin' in user_params_keys_raw:
            # a_begin not specified, t_begin specified
            if t_begin == 0:
                abort(
                    'You have specified t_begin = 0 while having Hubble expansion enabled. '
                    'Please specify some finite starting time or disable Hubble expansion.'
                )
            t_begin_correct = t_begin
            a_begin_correct = scale_factor(t_begin_correct)
        else:
            # Neither a_begin nor t_begin is specified.
            # One or the other is needed when enable_Hubble is True.
            abort('No initial scale factor (a_begin) or initial cosmic time (t_begin) specified. '
                  'A specification of one or the other is needed when enable_Hubble is True.')
    else:
        # Hubble expansion disabled
        t_begin_correct = t_begin
        # Values of the scale factor (and therefore a_begin)
        # are meaningless. Set a_begin to unity,
        # effectively ignoring its existence.
        a_begin_correct = 1.0
        if 'a_begin' in user_params_keys_raw and a_begin != 1:
            masterwarn(f'Ignoring a_begin = {a_begin} because enable_Hubble is False')
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

# Container class storing background splines to be set by init_time()
@cython.cclass
class TemporalSplines:
    @cython.header()
    def __init__(self):
        """
        Spline a_t
        Spline t_a
        Spline a_H
        Spline a_D
        Spline a_f
        Spline a_D2
        Spline a_f2
        Spline a_D3a
        Spline a_f3a
        Spline a_D3b
        Spline a_f3b
        Spline a_D3c
        Spline a_f3c
        """
        self.a_t = None
        self.t_a = None
        self.a_H = None
        self.a_D = None
        self.a_f = None
        self.a_D2 = None
        self.a_f2 = None
        self.a_D3a = None
        self.a_f3a = None
        self.a_D3b = None
        self.a_f3b = None
        self.a_D3c = None
        self.a_f3c = None
    @property
    def initialized(self):
        return self.a_t is not None
cython.declare(temporal_splines='TemporalSplines')
temporal_splines = TemporalSplines()

# Function for solving the simplified matter + Λ background
def solve_matterΛ_background(a_today=1):
    """Note that the results are only returned by the master process"""
    a_values = t_values = H_values = None
    D_values = f_values = None
    D2_values = f2_values = None
    D3a_values = f3a_values = None
    D3b_values = f3b_values = None
    D3c_values = f3c_values = None
    if not master:
        return (
            a_values, t_values, H_values,
            D_values, f_values,
            D2_values, f2_values,
            D3a_values, f3a_values,
            D3b_values, f3b_values,
            D3c_values, f3c_values,
        )
    # Load from cache
    filename = get_reusable_filename('background', Ωm, a_today, unit_time)
    if os.path.exists(filename):
        return tuple(map(np.ascontiguousarray, np.loadtxt(filename, unpack=True)))
    # Solve matter + Λ background
    masterprint('Solving matter + Λ background ...')
    import scipy.integrate
    # Get the age of the universe t(a=1)
    a_begin_bg = 1e-14  # as in CLASS
    solve_ivp_kwargs = dict(
        method='DOP853',
        rtol=1e-12,
        atol=0,
    )
    t_begin_bg = 2/(3*hubble(a_begin_bg))  # assumes matter domination
    def event(logt, loga):
        return loga[0] - ℝ[log(a_today)]
    event.terminal = True
    t_today = exp(
        scipy.integrate.solve_ivp(
            dloga_dlogt,
            (log(t_begin_bg), ထ),
            asarray([log(a_begin_bg)]),
            events=event,
            **solve_ivp_kwargs,
        ).t_events[0][0]
    )
    # Tabulate a(t) from t_begin_bg to today
    n_bg = int(log(a_today/a_begin_bg)/7e-3)  # as in CLASS
    logt_values = linspace(log(t_begin_bg), log(t_today), n_bg)
    t_values = np.exp(logt_values)
    a_values = np.exp(
        scipy.integrate.solve_ivp(
            dloga_dlogt,
            (log(t_begin_bg), log(t_today)),
            [log(a_begin_bg)],
            t_eval=logt_values,
            **solve_ivp_kwargs,
        ).y[0]
    )
    t_values[0], t_values[-1] = t_begin_bg, t_today
    a_values[0], a_values[-1] = a_begin_bg, a_today
    # Tabulate H on the same interval
    H_values = asarray([hubble(a) for a in a_values])
    # Solve growth factors in a-space,
    # using matter dominated initial conditions.
    C = 1  # arbitrary
    D_begin_bg       =  1./ 1.*C**1*a_begin_bg**1
    dD_da_begin_bg   =  1./ 1.*C**1*a_begin_bg**0
    D2_begin_bg      =  3./ 7.*C**2*a_begin_bg**2
    dD2_da_begin_bg  =  6./ 7.*C**2*a_begin_bg**1
    D3a_begin_bg     =  1./ 3.*C**3*a_begin_bg**3
    dD3a_da_begin_bg =  1./ 1.*C**3*a_begin_bg**2
    D3b_begin_bg     = 10./21.*C**3*a_begin_bg**3
    dD3b_da_begin_bg = 10./ 7.*C**3*a_begin_bg**2
    D3c_begin_bg     =  1./ 7.*C**3*a_begin_bg**3
    dD3c_da_begin_bg =  3./ 7.*C**3*a_begin_bg**2
    growth_solution = scipy.integrate.solve_ivp(
        dgrowth_da,
        (a_begin_bg, a_today),
        [
            D_begin_bg, dD_da_begin_bg,
            D2_begin_bg, dD2_da_begin_bg,
            D3a_begin_bg, dD3a_da_begin_bg,
            D3b_begin_bg, dD3b_da_begin_bg,
            D3c_begin_bg, dD3c_da_begin_bg,
        ],
        t_eval=a_values,
        **solve_ivp_kwargs,
    )
    # Extract growth rates f = a/D*dD/da
    # and normalize growth factors to D(a=1) = 1.
    (
        D_values, f_values,
        D2_values, f2_values,
        D3a_values, f3a_values,
        D3b_values, f3b_values,
        D3c_values, f3c_values,
    ) = map(np.ascontiguousarray, growth_solution.y)
    f_values *= a_values/D_values
    f2_values *= a_values/D2_values
    f3a_values *= a_values/D3a_values
    f3b_values *= a_values/D3b_values
    f3c_values *= a_values/D3c_values
    D_normalization = 1/D_values[-1]
    D_values *= D_normalization
    D_values[-1] = 1
    D2_values  *= D_normalization**2
    D3a_values *= D_normalization**3
    D3b_values *= D_normalization**3
    D3c_values *= D_normalization**3
    # Cache background to disk
    results = (
        a_values, t_values, H_values,
        D_values, f_values,
        D2_values, f2_values,
        D3a_values, f3a_values,
        D3b_values, f3b_values,
        D3c_values, f3c_values,
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(
        filename,
        asarray(results).T,
        header='\n'.join([
            unicode(f'Matter + Λ background with Ωm = {Ωm}, ΩΛ = 1 - Ωm = {1 - Ωm}'),
            '',
            ' '.join([
                f'{title:<22}'
                for title in map(
                    unicode,
                    [
                        'a', f't [{unit_time}]', f'H [{unit_time}⁻¹]',
                        'D⁽¹⁾', 'f⁽¹⁾',
                        'D⁽²⁾', 'f⁽²⁾',
                        'D⁽³ᵃ⁾', 'f⁽³ᵃ⁾',
                        'D⁽³ᵇ⁾', 'f⁽³ᵇ⁾',
                        'D⁽³ᶜ⁾', 'f⁽³ᶜ⁾',
                    ],
                )
            ]).rstrip(),
        ]),
        encoding='utf-8',
    )
    masterprint('done')
    return results

# Function returning ∂log(a)/∂log(t), given both log(t) and log(a).
# The function is written as to be
# plugged into scipy.integrate.solve_ivp().
@cython.pheader(
    # Arguments
    logt='double',
    loga='double[::1]',
    # Locals
    a='double',
    t='double',
    returns='double',
)
def dloga_dlogt(logt, loga):
    t = exp(logt)
    a = exp(loga[0])
    return t*hubble(a)

# Function that takes in a and the vector
#   [D⁽¹⁾, ∂D⁽¹⁾/∂a, D⁽²⁾, ∂D⁽²⁾/∂a, D⁽³ᵃ⁾, ∂D⁽³ᵃ⁾/∂a, D⁽³ᵇ⁾, ∂D⁽³ᵇ⁾/∂a, D⁽³ᶜ⁾, ∂D⁽³ᶜ⁾/∂a]
# and returns the derivative of this vector with respect to a.
# The function is written as to be
# plugged into scipy.integrate.solve_ivp().
@cython.pheader(
    # Arguments
    a='double',
    y='double[::1]',
    # Locals
    D='double',
    D2='double',
    D3a='double',
    D3b='double',
    D3c='double',
    d2D2_da2='double',
    d2D3a_da2='double',
    d2D3b_da2='double',
    d2D3c_da2='double',
    d2D_da2='double',
    dD2_da='double',
    dD3a_da='double',
    dD3b_da='double',
    dD3c_da='double',
    dD_da='double',
    dH_da_over_H='double',
    returns='double[::1]',
)
def dgrowth_da(a, y):
    # Extract
    (
        D, dD_da,
        D2, dD2_da,
        D3a, dD3a_da,
        D3b, dD3b_da,
        D3c, dD3c_da,
    ) = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]  # generates better code than implicit unpacking
    # Compute derivatives.
    # Here we assume the simplified matter + Λ Hubble parameter
    # H(a) = H0*sqrt(Ωm/a**3 + (1 - Ωm)), as also used in hubble()
    # when enable_class_background is False.
    dH_da_over_H = -1.5*Ωm*(H0/hubble(a))**2/a**4
    d2D_da2   = -(3/a + dH_da_over_H)*dD_da   - dH_da_over_H/a*D
    d2D2_da2  = -(3/a + dH_da_over_H)*dD2_da  - dH_da_over_H/a*(D2 + D**2)
    d2D3a_da2 = -(3/a + dH_da_over_H)*dD3a_da - dH_da_over_H/a*(D3a + 2*D**3)
    d2D3b_da2 = -(3/a + dH_da_over_H)*dD3b_da - dH_da_over_H/a*(D3b + 2*D*D2 + 2*D**3)
    d2D3c_da2 = -(3/a + dH_da_over_H)*dD3c_da - dH_da_over_H/a*D**3
    # Pack and return
    dgrowth_da_returnvec[0] = y[1]
    dgrowth_da_returnvec[1] = d2D_da2
    dgrowth_da_returnvec[2] = y[3]
    dgrowth_da_returnvec[3] = d2D2_da2
    dgrowth_da_returnvec[4] = y[5]
    dgrowth_da_returnvec[5] = d2D3a_da2
    dgrowth_da_returnvec[6] = y[7]
    dgrowth_da_returnvec[7] = d2D3b_da2
    dgrowth_da_returnvec[8] = y[9]
    dgrowth_da_returnvec[9] = d2D3c_da2
    return dgrowth_da_returnvec
cython.declare(dgrowth_da_returnvec='double[::1]')
dgrowth_da_returnvec = empty(10, dtype=C2np['double'])
