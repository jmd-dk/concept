Numerics
--------
The 'numerics' parameter category contains parameters specifying various
options regarding the numerical methods used within the simulation, as well as
some numerical resolutions and length scales.

.. contents::
   :local:
   :depth: 1



------------------------------------------------------------------------------



.. _boxsize:

``boxsize``
...........
== =============== == =
\  **Description** \  Specifies size of simulation box
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         512*Mpc

-- --------------- -- -
\  **Elaboration** \  All CO\ *N*\ CEPT simulations take place within a cubic
                      box of constant comoving side length
                      :math:`L_{\text{box}}`. This parameter sets this length.
-- --------------- -- -
\  **Example 0**   \  Use a box size of :math:`1024\,\mathrm{Mpc}`:

                      .. code-block:: python3

                         boxsize = 1024*Mpc

-- --------------- -- -
\  **Example 1**   \  Use a box size of
                      :math:`1\,\mathrm{Gpc} = 1000\,\mathrm{Mpc}`:

                      .. code-block:: python3

                         boxsize = 1*Gpc

-- --------------- -- -
\  **Example 2**   \  Use a box size of :math:`1024\,\mathrm{Mpc}/h`:

                      .. code-block:: python3

                         boxsize = 1024*Mpc/h

                      With e.g.
                      :math:`H_0 = 64\,\mathrm{km}\, \mathrm{s}^{-1}\, \mathrm{Mpc}^{-1}`,
                      this is equivalent to

                      .. code-block:: python3

                         boxsize = 1600*Mpc

                      .. note::
                         See the ``H0`` :ref:`parameter <H0>` for details on
                         ``h``

== =============== == =



------------------------------------------------------------------------------



.. _potential_options:

``potential_options``
.....................
== =============== == =
\  **Description** \  Specifications for potential computations
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'gridsize': {
                                 'global': {},
                                 'particles': {
                                     'gravity': {
                                         'pm' : (  'cbrt(N)',   'cbrt(N)')
                                         'p3m': ('2*cbrt(N)', '2*cbrt(N)'),
                                     },
                                     'lapse': {
                                         'pm': ('cbrt(N)', 'cbrt(N)'),
                                     },
                                 },
                                 'fluid': {
                                     'gravity': {
                                         'pm': ('gridsize', 'gridsize'),
                                     },
                                     'lapse': {
                                         'pm': ('gridsize', 'gridsize'),
                                     },
                                 },
                             },
                             'interpolation': {
                                 'gravity': {
                                     'pm' : 'CIC',
                                     'p3m': 'CIC',
                                 },
                                 'lapse': {
                                     'pm': 2,
                                 },
                             },
                             'deconvolve': {
                                 'gravity': {
                                     'pm' : (True, True),
                                     'p3m': (True, True),
                                 },
                                 'lapse': {
                                     'pm': (True, True),
                                 },
                             },
                             'interlace': {
                                 'gravity': {
                                     'pm' : False,
                                     'p3m': False,
                                 },
                                 'lapse': {
                                     'pm': False,
                                 },
                             },
                             'differentiation': {
                                 'default': {
                                     'gravity': {
                                         'pm' : 2,
                                         'p3m': 4,
                                     },
                                     'lapse': {
                                         'pm': 2,
                                     },
                                 },
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This parameter is a ``dict`` of several individual
                      sub-parameters, specifying the details of how
                      potentials are constructed and applied to components.
                      Below you will find a summary of how potentials are
                      implemented in CO\ *N*\ CEPT, after which the
                      sub-parameters are described.

                      Potentials are discretised on grids in order to
                      implement long-range interactions.

                      .. note::
                         Currently, two distinct long-range
                         interactions/forces are implemented; ``'gravity'``
                         (via the PM or P³M method, the latter of which
                         further has a short-range part) and ``'lapse'``
                         (via the PM method). The ``'lapse'`` interaction is
                         relatively obscure and has to do with gravitational
                         time dilation. See the
                         :ref:`decaying cold dark matter part of the tutorial<decaying_cold_dark_matter>`
                         as well as the paper on
                         ":doc:`Fully relativistic treatment of decaying cold dark matter in 𝘕-body simulations </publications>`
                         for details on the ``'lapse'`` interaction/potential.
                         For specifications of forces and methods, see the
                         ``select_forces`` :ref:`parameter <select_forces>`.
                         Though we shall focus on ``'gravity'`` below, all
                         long-range interactions and their potentials
                         behave similarly.

                      In CO\ *N*\ CEPT a long-range interaction takes place
                      between two groups of
                      :ref:`components <components_and_selections>`;
                      the *suppliers* and the *receivers*, which respectively
                      build the potential and receive a force due to it. In
                      the simplest case of self-gravity between particles of
                      a component, this component is both the sole supplier
                      and receiver component.

                      CO\ *N*\ CEPT uses an '*upstream* :math:`\rightarrow`
                      *global* :math:`\rightarrow` *downstream* scheme' for
                      potentials, as outlined below:

                      1. Interpolate each supplier component onto individual
                         'upstream' grids, the size of which are individual to
                         each component. More precisely, a specific quantity
                         of the supplier components is interpolated, e.g. the
                         mass (particle components) or energy density (fluid
                         components) in the case of gravity.
                      2. Transform each upstream grid to Fourier space.
                      3. Optionally perform deconvolution and interlacing of
                         upstream grids constructed from particle suppliers
                         (see
                         "`Computer simulation using particles <https://dx.doi.org/10.1201/9780367806934>`_").
                      4. Add upstream Fourier grids together, producing a
                         'global' Fourier grid, with the global grid size
                         being a free parameter.

                         .. note::
                            Grids of different sizes can be seamlessly added
                            together in Fourier space, as all
                            :math:`\boldsymbol{k}` modes (grid cells) of
                            smaller grids are contained within larger grids.
                            One need to be careful though, as complex phase
                            shifts are needed to correct for the change of
                            grid size.

                      5. Convert the values within the global grid to
                         potential values. For gravity, this amounts to
                         solving the Poisson equation.
                      6. For each receiver component, produce a 'downstream'
                         version of the global Fourier potential, by copying
                         (and shifting) the modes onto grids individual to
                         each receiver.
                      7. Optionally perform (another) deconvolution for
                         downstream grids of particle receivers, due to the
                         upcoming interpolation back to the particle
                         positions.

                      For receivers obtaining the force grid from the
                      downstream real-space potential grid:

                         8. Transform downstream potential to real space.
                         9. Obtain force grid by differentiating the
                            downstream real-space potential using some finite
                            difference method.

                      For receivers obtaining the force grid from the
                      downstream Fourier-space potential:

                         8. Obtain Fourier-space force grid by differentiating
                            the downstream Fourier-space potential through
                            multiplication by
                            :math:`\mathrm{i}\boldsymbol{k}`.
                         9. Transform the force grid to real space.

                      10. Apply each force grid by interpolating it back onto
                          the corresponding receiver.

                      .. note::
                         Though the above recipe is conceptually faithful to
                         the actual implementation in CO\ *N*\ CEPT, typically
                         many of the steps can be combined, reducing the
                         total number of steps (and grids in memory)
                         significantly. In all cases, CO\ *N*\ CEPT collapses
                         the above procedure to its minimal equivalence,
                         which in simple (typical) cases among other things
                         leaves out the upstream and downstream potentials
                         completely and combines the two deconvolutions into
                         one.

                      With the implemented potential scheme outlined above,
                      the different sub-parameters of the
                      ``potential_options`` parameter are presented below:

                      * ``'gridsize'``: This is a ``dict`` which functions as
                        a
                        :ref:`component selection <components_and_selections>`,
                        mapping components to ``dict``\ s specifying the
                        various grid sizes, e.g.

                        .. code-block:: python3

                           'gridsize': {
                               'matter': {
                                   'gravity': {
                                       'pm' : (gridsize_pm_upstream,  gridsize_pm_downstream),
                                       'p3m': (gridsize_p3m_upstream, gridsize_p3m_downstream),
                                   },
                               },
                           }

                        where ``'matter'`` is used as a stand-in for some
                        component and ``'gravity'`` could be any long-range
                        interaction. The ``gridsize_*`` variables are then
                        integers specifying the upstream and downstream grid
                        sizes (i.e. the number of grid cells along each
                        dimension) for the given interaction and component.

                        .. note::
                           One can refer to the number of particles :math:`N`
                           by using ``'N'`` within a ``str``, as used in the
                           above default specifications. Similarly,
                           ``'gridsize'`` can be used to refer to the
                           intrinsic 'fluid grid size' of fluid components.
                           Importantly, fluid components must have upstream
                           and downstream grid sizes equal to their fluid grid
                           size.

                        A special key ``'global'`` is further specified within
                        the ``'gridsize'`` sub-\ ``dict``, with a value of the
                        form

                        .. code-block:: python3

                           'gridsize': {
                               'global': {
                                   'gravity': {
                                       'pm' : gridsize_pm_global,
                                       'p3m': gridsize_p3m_global,
                                   },
                               },
                           }

                        where again ``'gravity'`` could be any long-range
                        interaction, with the ``gridsize_*_global`` parameters
                        specifying the global grid sizes to use for this
                        interaction.

                        .. note::
                           The ``'global'`` ``dict`` within the ``'gridsize'``
                           sub-parameter does not have a default value, as
                           appropriate grid sizes depend on the components
                           used within the simulation.

                      * ``'interpolation'``: This is a ``dict`` of the form

                        .. code-block:: python3

                           'interpolation': {
                               'gravity': {
                                   'pm' : order_interp_pm,
                                   'p3m': order_interp_p3m,
                               },
                           }

                        where the ``order_interp_*`` variables specify the
                        interpolation order to use when interpolating
                        particles to/from upstream/downstream grids. The
                        implemented interpolations are:

                        * ``'NPG'``: 'Nearest grid point', order ``1``.
                        * ``'CIC'``: 'Cloud in cell', order ``2``.
                        * ``'TSC'``: 'Triangular shaped cloud', order ``3``.
                        * ``'PCS'``: 'Piecewise cubic spline', order ``4``.

                        These may be specified using either their name (e.g.
                        ``'CIC'``) or integer order (e.g. ``2``). See the
                        paper on
                        ":doc:`The cosmological simulation code CO𝘕CEPT 1.0</publications>`"
                        for details on these.

                      * ``'deconvolve'``: This is a ``dict`` of the form

                        .. code-block:: python3

                           'deconvolve': {
                               'gravity': {
                                   'pm' : (deconv_pm_upstream,  deconv_pm_downstream),
                                   'p3m': (deconv_p3m_upstream, deconv_p3m_downstream),
                               },
                           }

                        with the ``deconv_*`` variables specifying whether
                        deconvolution due to particle upstream and downstream
                        interpolation should take place.

                      * ``'interlace'``: This is a ``dict`` of the form

                        .. code-block:: python3

                           'dinterlace': {
                               'gravity': {
                                   'pm' : interlace_pm,
                                   'p3m': interlace_p3m,
                               },
                           }

                        with the ``interlace_*`` variables specifying whether
                        grid interlacing should be used when interpolating
                        particles onto upstream grids.

                      * ``'differentiation'``: This is a
                        :ref:`component selection <components_and_selections>`
                        ``dict`` of the form

                        .. code-block:: python3

                           'differentiation': {
                               'matter': {
                                   'gravity': {
                                       'pm' : order_diff_pm,
                                       'p3m': order_diff_p3m,
                                   },
                               },
                           }

                        with the ``order_diff_*`` variables specifying the
                        order of differentiation to use to construct the force
                        grid(s) from the (downstream) potential grid(s). For
                        real-space differentiation, symmetric finite
                        differencing of orders ``2``, ``4``, ``6`` and ``8``
                        are implemented (see the paper on
                        ":doc:`The cosmological simulation code CO𝘕CEPT 1.0</publications>`" for details).
                        If instead you wish to make use of Fourier-space
                        differentiation, set the order to either ``'Fourier'``
                        or ``0``.

-- --------------- -- -
\  **Example 0**   \  Use default potential options, but set the global
                      gravitational P³M potential grid size to :math:`128`:

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': {
                                 'global': {
                                     'gravity': {
                                         'p3m': 128,
                                     },
                                 },
                             },
                         }

                      This can be shortened to

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': {
                                 'global': {
                                     'gravity': 128,
                                 },
                             },
                         }

                      though now both the P³M and PM method of gravity gets a
                      global grid size of :math:`128` (if the simulation does
                      not make use of PM, this does not matter). The above can
                      be further shortened to

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': {
                                 'global': 128,
                             },
                         }

                      Here it will be detected that no force/interaction has
                      been specified, in which case the default one
                      --- ``'gravity'`` --- will be used. The above can be
                      further shortened to

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': 128,
                         }

                      though this now also sets ``128`` as the upstream and
                      downstream grid sizes for ``'all'`` components. In
                      simple cases with e.g. only one component, this is
                      typically what we want. Finally, this can be simplified
                      to just

                      .. code-block:: python3

                         potential_options = 128

-- --------------- -- -
\  **Example 1**   \  Use default potential options, but set the
                      gravitational P³M potential grid size to :math:`128`,
                      for the global potential as well as for the upstream
                      and downstream potential grids for the component with a
                      name/species of ``'matter'``. For PM gravity, use a
                      global potential grid of size :math:`64`:

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': {
                                 'global': {
                                     'gravity': {
                                         'pm' :  64,
                                         'p3m': 128,
                                     },
                                 },
                                 'matter': {
                                     'gravity': {
                                         'p3m': (128, 128),
                                     },
                                 },
                             },
                         }

                      .. note::
                         As the upstream and downstream grid size within the
                         2-tuple are identical, this can be shortened to

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': {
                                 'global': {
                                     'gravity': {
                                         'pm' :  64,
                                         'p3m': 128,
                                     },
                                 },
                                 'matter': {
                                     'gravity': {
                                         'p3m': 128,
                                     },
                                 },
                             },
                         }

                      .. note::

                         This further assigns ``64`` as the upstream and
                         downstream gravity PM grid sizes for all components
                         for which these are unset, including ``'matter'``.

-- --------------- -- -
\  **Example 2**   \  For the gravitational interaction, be it PM or P³M, make
                      use of the highest available order for particle
                      interpolation (PCS), as well as interlacing. Use
                      :math:`80` as the size of all potential grids.

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': 80,
                             'interpolation': {
                                 'gravity': 'PCS',
                             },
                             'interlace': {
                                 'gravity': True,
                             }
                         }

-- --------------- -- -
\  **Example 3**   \  Use :math:`128` as the size of all potential grids, use
                      order-6 real-space finite differencing to obtain the
                      force from the gravitational PM potential and use
                      Fourier-space differentiation to obtain the force from
                      the gravitational P³M potential:

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': 128,
                             'differentiation': {
                                 'all': {
                                     'gravity': {
                                         'pm' : 6,
                                         'p3m': 'Fourier',
                                     },
                                 },
                             },
                         }

== =============== == =



------------------------------------------------------------------------------



.. _shortrange_params:

``shortrange_params``
.....................
== =============== == =
\  **Description** \  Specifications for short-range forces
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'gravity': {
                                 'scale'    : '1.25*boxsize/gridsize',
                                 'range'    : '4.5*scale',
                                 'tilesize' : 'range',
                                 'subtiling': ('automatic', 16),
                                 'tablesize': 4096,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This parameter is a ``dict`` which maps
                      short-range interactions to sub-parameter ``dict``\ s,
                      which specifies sub-parameters for each short-range
                      force. Currently, ``'gravity'`` is the only short-range
                      force implemented. For detailed information about the
                      short-range force --- including the short-/long-range
                      force split scale :math:`x_{\mathrm{s}}` and cut-off
                      :math:`x_{\mathrm{r}}` as well as tiles, subtiles and
                      automatic refinement thereof --- we refer to the paper
                      on
                      ":doc:`The cosmological simulation code CO𝘕CEPT 1.0</publications>`".

                      Each of the sub-parameters are described below.

                      * ``'scale'``: This is the short-/long-range force split
                        scale :math:`x_{\mathrm{s}}`, specifying the scale at
                        which the short- and long-range P³M forces are
                        comparable. The default value corresponds to a scale
                        of :math:`1.25` grid cell widths, as ``'gridsize'`` is
                        dynamically substituted for the global grid size of
                        the long-range potential grid. See the
                        ``potential_options``
                        :ref:`parameter <potential_options>`
                        for details on this grid.

                      * ``'range'``: This is the cut-off scale
                        :math:`x_{\mathrm{r}}`, specifying the range of the
                        short-range force. The default value is set to
                        :math:`4.5 x_{\mathrm{s}}`, as ``'scale'`` is being
                        dynamically substituted.

                      * ``'tilesize'``: This sets the (minimum) width of the
                        tiles. To ensure that all particle pairs within
                        :math:`x_{\mathrm{r}}` of each other gets paired up,
                        this should be set no lower than
                        :math:`x_{\mathrm{r}}`, i.e. ``'range'``. At the same
                        time, there is typically no reason to set a larger
                        value, as this just slows down the short-range
                        interaction without actually increasing the number of
                        particle-particle interactions.

                      * ``'subtiling'``: This specifies the subtile
                        decomposition. It may be defined as a triplet of
                        integers specifying a fixed decomposition or a single
                        integer specifying a fixed cubic decomposition.
                        Alternatively, it may be defined to be the ``str``
                        ``'automatic'``, which specifies automatic and dynamic
                        subtile refinement, or a 2-tuple with ``'automatic'``
                        as the first element and the length of the refinement
                        period (in number of time steps) as the second
                        element.

                        .. caution::
                           The automatic subtile refinement is based on CPU
                           timing measurements within the simulation and so
                           breaks strict deterministic behaviour.

                      * ``'tablesize'``: The gravitational short-range force
                        between two particles is a complicated expression, and
                        so it is pre-tabulated, with actual forces found
                        through cheap (1D) NGP lookups in this table. As we do
                        not even use linear interpolation, this table needs to
                        be rather large. Exactly how large is controlled by
                        the ``'tablesize'`` sub-parameter.

-- --------------- -- -
\  **Example 0**   \  Extend :math:`x_{\mathrm{r}}` all the way to
                      :math:`5.5 x_{\mathrm{s}}`, for the gravitational
                      short-range interaction:

                      .. code-block:: python3

                         shortrange_params = {
                             'gravity': {
                                 'scale': '1.25*boxsize/gridsize',
                                 'range': '5.5*scale',
                             },
                         }

                      We can shorten the above to

                      .. code-block:: python3

                         shortrange_params = {
                             'scale': '1.25*boxsize/gridsize',
                             'range': '5.5*scale',
                         }

                      in which case ``'gravity'`` will be set as the
                      force/interaction automatically, as it is the default
                      force/interaction.

                      .. note::
                         In both cases, the remaining sub-parameters will
                         receive default values

-- --------------- -- -
\  **Example 1**   \  Use a fixed subtile decomposition of
                      :math:`(3, 3, 3)` for the gravitational short-range
                      interaction:

                      .. code-block:: python3

                         shortrange_params = {
                             'subtiling': (3, 3, 3),
                         }

                      As all three subdivisions are equal, this can be
                      shortened to

                      .. code-block:: python3

                         shortrange_params = {
                             'subtiling': 3,
                         }

                      While probably a bit slower than using automatic subtile
                      refinement, this subtile decomposition has the benefit of
                      being static and thus deterministic.
-- --------------- -- -
\  **Example 2**   \  Use dynamic subtile refinement with a period
                      of :math:`8` time steps for the gravitational
                      short-range interaction:

                      .. code-block:: python3

                         shortrange_params = {
                             'subtiling': ('automatic', 8),
                         }

== =============== == =



------------------------------------------------------------------------------



.. _powerspec_options:

``powerspec_options``
.....................
== =============== == =
\  **Description** \  Specifications for power spectrum computations
                      and dumps
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'upstream gridsize': {
                                 'particles': '2*cbrt(N)',
                                 'fluid'    : 'gridsize',
                             },
                             'global gridsize': {},
                             'interpolation': {
                                 'default': 'PCS',
                             },
                             'deconvolve': {
                                 'default': True,
                             },
                             'interlace': {
                                 'default': True,
                             },
                             'k_max': {
                                 'default': 'nyquist',
                             },
                             'binsize': {
                                 'default': {
                                     '1*k_min':   π/boxsize,
                                     '5*k_min': 2*π/boxsize,
                                 },
                             },
                             'tophat': {
                                 'default': 8*Mpc/h,
                             },
                             'significant figures': {
                                 'default': 8,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` of several individual sub-parameters,
                      specifying details of how to compute and save power
                      spectra. All sub-parameters are themselves
                      :ref:`component selections <components_and_selections>`.

                      For computing a power spectrum, one or more components
                      are first interpolated onto individual *upstream* grids,
                      possibly using deconvolution and interlacing, after
                      which they are added together in Fourier space,
                      producing a *global* grid. This scheme is similar to the
                      one used for potentials, except here we never go back to
                      real space, neither do we interpolate anything back to
                      the particles. See the ``potential_options``
                      :ref:`parameter <potential_options>` for a walk-through
                      of the scheme.

                      Each sub-parameter is described below:

                      * ``'upstream gridsize'``: Specifies the upstream grid
                        sizes to use for each component.

                      * ``'global gridsize'``: Specifies the global grid size
                        to use for each power spectrum. Which power spectra to
                        compute are in turn specified by the
                        ``powerspec_select``
                        :ref:`parameter <powerspec_select>`.

                        .. note::
                           This has no default value, as a proper global grid
                           size depends on the components within the
                           simulation. If this is not set when a power
                           spectrum is to be computed, a value equal to the
                           largest of the upstream grid sizes (in use for this
                           particular power spectrum) will be used.

                      * ``'interpolation'``: Specifies the interpolation order
                        to use when interpolating particles to upstream grids.
                        The implemented interpolations are:

                        * ``'NPG'``: 'Nearest grid point', order ``1``.
                        * ``'CIC'``: 'Cloud in cell', order ``2``.
                        * ``'TSC'``: 'Triangular shaped cloud', order ``3``.
                        * ``'PCS'``: 'Piecewise cubic spline', order ``4``.

                      * ``'deconvolve'``: Specifies whether to apply
                        deconvolution for upstream particle interpolations.

                      * ``'interlace'``: Specifies whether to use interlacing
                        for upstream particle interpolations.

                      * ``'k_max'``: Specifies the largest :math:`k` mode to
                        include in the power spectrum output files (data files
                        and plots). If given as a ``str``, ``'nyquist'`` is
                        dynamically substituted for the Nyquist mode,

                        .. math::

                           \require{upgreek}
                           k_{\mathrm{Nyquist}} = \biggl\lfloor\frac{n_{\mathrm{ps}}}{2}\biggr\rfloor \times \frac{2\uppi}{L_{\mathrm{box}}}\, ,

                        with :math:`L_{\mathrm{box}}` given by ``boxsize`` and
                        :math:`n_{\mathrm{ps}}` the global grid size used for
                        the power spectrum.

                        .. note::
                           Setting ``'k_max'`` above ``'sqrt(3)*nyquist'``
                           will not yield further data points, as no further
                           data is available in the 3D grid.

                      * ``'binsize'``: Specifies the width of the power
                        spectrum bins. By specifying different widths at
                        different :math:`k`, a running (logarithmically
                        interpolated) bin size is obtained. When the
                        :math:`k`\ 's are given as ``str``\ s, the following
                        sub\ ``str``\ s are dynamically substituted:

                        * ``'k_min'``: ``2*π/boxsize``.
                        * ``'k_max'``: Value set for ``'k_max'``, though at
                          most ``sqrt(3)*gridsize//2*k_min``.
                        * ``'gridsize'``: ``gridsize``.
                        * ``'nyquist'``: ``gridsize//2*k_min``.

                      * ``'tophat'``: Included in the power spectrum data
                        files is also the root-mean-square density variation
                        :math:`\sigma_R`, smoothed with a spherical
                        top-hat filter :math:`W(s)` of radius :math:`R`:

                        .. math::

                           \require{upgreek}
                           \sigma_R &= \frac{1}{2\uppi^2} \left( \int k^2 W^2(kR) P(k)\,\mathrm{d}k \right)^{1/2}\, , \\
                           W(s) &= \frac{3}{s^3}\bigl(\sin(s) - s\cos(s)\bigr)\, ,

                        with :math:`P(k)` the power spectrum.

                        .. note::
                           Though the integral should really be over all
                           :math:`k`, naturally we can only use those within
                           the tabulated power spectrum

                        The radius :math:`R` is controlled by this
                        sub-parameter. The default value of ``8*Mpc/h``
                        results in the standard :math:`\sigma_8`.

                      * ``'significant figures'``: Specifies the number of
                        significant figures to use for the data in the power
                        spectrum data files.

                      .. note::
                         For all sub-parameters above except
                         ``'upstream gridsize'``, the keys used within the
                         :ref:`component selection <components_and_selections>`
                         sub-\ ``dict``\ s may refer to either a single
                         component or a combination of components. In the case
                         of the latter, this designates a combined (auto)
                         power spectrum.

-- --------------- -- -
\  **Example 0**   \  Use an upstream grid size of :math:`256` for the
                      component with a name/species of ``'matter'``:

                      .. code-block:: python3

                         powerspec_options = {
                             'upstream gridsize': {
                                 'matter': 256,
                             },
                         }

                      As we did not specify a global grid size, this will
                      similarly be :math:`256`. We can also set both
                      explicitly by using

                      .. code-block:: python3

                         powerspec_options = {
                             'gridsize': {
                                 'matter': 256,
                             },
                         }

                      Here ``'gridsize'`` is not a real sub-parameter of
                      ``powerspec_options``. It merely acts as a convenient
                      shortcut for setting both ``'upstream gridsize'`` and
                      ``'global gridsize'``.
-- --------------- -- -
\  **Example 1**   \  Use small bins of the same size for all :math:`k` of
                      every power spectrum:

                      .. code-block:: python3

                         powerspec_options = {
                             'binsize': {
                                 'all'             : 0.5*π/boxsize,
                                 'all combinations': ...,
                             },
                         }

                      This may be shortened to

                      .. code-block:: python3

                         powerspec_options = {
                             'binsize': 0.5*π/boxsize,
                         }

-- --------------- -- -
\  **Example 2**   \  Use CIC interpolation and disable interlacing for power
                      spectra of the component with a name/species of
                      ``'matter'``:

                      .. code-block:: python3

                         powerspec_options = {
                             'interpolation': {
                                 'matter': 'CIC',
                             },
                             'interlace': {
                                 'matter': False,
                             }
                         }

== =============== == =



------------------------------------------------------------------------------



``k_modes_per_decade``
......................
== =============== == =
\  **Description** \  Number of Fourier modes :math:`k` per decade in
                      CLASS computations
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             3e-3/Mpc: 10,
                             3e-2/Mpc: 30,
                             3e-1/Mpc: 30,
                                1/Mpc: 10,
                         }

-- --------------- -- -
\  **Elaboration** \  This parameter determines the number of CLASS
                      perturbations to compute, by specifying the density
                      of Fourier modes :math:`k` as a (running) number per
                      decade. The resulting tabulation of :math:`k` is
                      employed for all CLASS perturbations, which are used
                      for e.g. initial conditions. The default choice yields a
                      relatively sparse tabulation of perturbations except
                      around baryon acoustic oscillations, where more detail
                      is added.
-- --------------- -- -
\  **Example 0**   \  Use a constant :math:`20` modes per decade, i.e.
                      :math:`20` values of :math:`k` placed logarithmically
                      equidistant between
                      :math:`k = 10^{-3}\,\mathrm{Mpc}^{-1}` and
                      :math:`k = 10^{-2}\,\mathrm{Mpc}^{-1}`, between
                      :math:`k = 10^{-2}\,\mathrm{Mpc}^{-1}` and
                      :math:`k = 10^{-1}\,\mathrm{Mpc}^{-1}`, etc:

                      .. code-block:: python3

                         k_modes_per_decade = 20

-- --------------- -- -
\  **Example 1**   \  Use :math:`50` modes per decade around
                      :math:`k = 10^{-4}\,\mathrm{Mpc}^{-1}` and :math:`10`
                      modes per decade around
                      :math:`k = 10^1\,\mathrm{Mpc}^{-1}`, with a running
                      number per decade in between found through logarithmic
                      interpolation (resulting in e.g. :math:`42` modes per
                      decade around :math:`k = 10^{-3}\,\mathrm{Mpc}^{-1}`):

                      .. code-block:: python3

                         k_modes_per_decade = {
                             1e-4/Mpc: 50,
                             1e+1/Mpc: 10,
                         }

== =============== == =

