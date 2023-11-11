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
\  **Example 0**   \  Use a box size of :math:`1024\,\text{Mpc}`:

                      .. code-block:: python3

                         boxsize = 1024*Mpc
-- --------------- -- -
\  **Example 1**   \  Use a box size of
                      :math:`1\,\text{Gpc} = 1000\,\text{Mpc}`:

                      .. code-block:: python3

                         boxsize = 1*Gpc
-- --------------- -- -
\  **Example 2**   \  Use a box size of :math:`1024\,\text{Mpc}/h`:

                      .. code-block:: python3

                         boxsize = 1024*Mpc/h

                      With e.g.
                      :math:`H_0 = 64\,\text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1}`,
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
                                         'pm' : (  'cbrt(Ñ)',   'cbrt(Ñ)'),
                                         'p3m': ('2*cbrt(Ñ)', '2*cbrt(Ñ)'),
                                     },
                                     'lapse': {
                                         'pm': ('cbrt(Ñ)', 'cbrt(Ñ)'),
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
                                     'pm' : (False, False),
                                     'p3m': (False, False),
                                 },
                                 'lapse': {
                                     'pm': (False, False),
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
                         ':ref:`Fully relativistic treatment of decaying cold dark matter in 𝘕-body simulations <fully_relativistic_treatment_of_decaying_cold_dark_matter_in_nbody_simulations>`'
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
                      a single component, this component is both the sole
                      supplier and receiver component.

                      CO\ *N*\ CEPT uses an '*upstream* :math:`\rightarrow`
                      *global* :math:`\rightarrow` *downstream* scheme' for
                      potentials, as outlined below:

                      1. Interpolate each supplier component onto individual
                         'upstream' grids, the size of which are individual to
                         each component. More precisely, a specific quantity
                         of the supplier components is interpolated, e.g. the
                         mass (particle components) or energy density (fluid
                         components) in the case of gravity. For particle
                         components, the interpolation may be interlaced,
                         meaning carried out multiple times with shifted
                         grids.
                      2. Transform each upstream grid to Fourier space.
                      3. Optionally perform deconvolution and/or shifting of
                         the complex phase (due to interlacing) of upstream
                         grids constructed from particle suppliers (see
                         "`Computer simulation using particles <https://dx.doi.org/10.1201/9780367806934>`__").
                      4. Add upstream Fourier grids together, producing a
                         'global' Fourier grid, with the global grid size
                         being a free parameter.

                         .. note::
                            Grids of different sizes can be seamlessly added
                            together in Fourier space, as all
                            :math:`\boldsymbol{k}` modes (grid cells) of
                            smaller grids are contained within larger grids.
                            Complex phase shifts are needed to correct for the
                            change of grid size, all taken care of
                            by CO\ *N*\ CEPT.

                      5. Convert the values within the global grid to
                         potential values. For gravity, this amounts to
                         solving the Poisson equation.
                      6. For each receiver component, produce a 'downstream'
                         version of the global Fourier potential by copying
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
                          the corresponding receiver. This downstream
                          interpolation may again be carried out using
                          interlacing, requiring steps 8--9 to be carried out
                          multiple times as well as shifting of the
                          complex phases.

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
                           size. For particle components with :math:`N = 2n^3`
                           or :math:`N = 4n^3` particles (pre-initialised on
                           interleaved lattices, see the ``initial_conditions``
                           :ref:`parameter <initial_conditions>`), you can
                           refer to just the cubic part as ``'Ñ'``. That is,
                           regardless of whether the number of particles is
                           :math:`N = n^3`, :math:`N = 2n^3` or
                           :math:`N = 4n^3`, we always have
                           :math:`\widetilde{N} = n^3`.

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
                        ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'
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
                                   'pm' : (interlace_pm_upstream,  interlace_pm_downstream),
                                   'p3m': (interlace_p3m_upstream, interlace_p3m_downstream),
                               },
                           }

                        with the ``interlace_*`` variables specifying whether
                        grid interlacing should be used for particle upstream
                        and downstream interpolation.  Possible values are
                        ``'sc'`` (or ``False``) for a simple cubic lattice
                        (meaning no interlacing), ``'bcc'`` (or ``True``) for
                        a body-centered cubic lattice (meaning standard
                        interlacing involving two relatively shifted particle
                        interpolations) or ``'fcc'`` for a face-centered cubic
                        lattice (meaning interlacing involving four relatively
                        shifted particle interpolations).

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
                        ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'
                        for details). If instead you wish to make use of
                        Fourier-space differentiation, set the order to either
                        ``'Fourier'`` or ``0``.
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
                      interpolation (PCS), as well as both upstream and
                      downstream (standard) interlacing. Use :math:`80` as the
                      size of all potential grids.

                      .. code-block:: python3

                         potential_options = {
                             'gridsize': 80,
                             'interpolation': {
                                 'gravity': 'PCS',
                             },
                             'interlace': {
                                 'gravity': (True, True),  # or ('bcc', 'bcc')
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
                      force split scale :math:`x_{\text{s}}` and cut-off
                      :math:`x_{\text{r}}` as well as tiles, subtiles and
                      automatic refinement thereof --- we refer to the paper
                      on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'.

                      Each of the sub-parameters are described below.

                      * ``'scale'``: This is the short-/long-range force split
                        scale :math:`x_{\text{s}}`, specifying the scale at
                        which the short- and long-range P³M forces are
                        comparable. The default value corresponds to a scale
                        of :math:`1.25` grid cell widths, as ``'gridsize'`` is
                        dynamically substituted for the global grid size of
                        the long-range potential grid. See the
                        ``potential_options``
                        :ref:`parameter <potential_options>`
                        for details on this grid.

                      * ``'range'``: This is the cut-off scale
                        :math:`x_{\text{r}}`, specifying the range of the
                        short-range force. The default value is set to
                        :math:`4.5 x_{\text{s}}`, as ``'scale'`` is being
                        dynamically substituted.

                      * ``'tilesize'``: This sets the (minimum) width of the
                        tiles. To ensure that all particle pairs within
                        :math:`x_{\text{r}}` of each other gets paired up,
                        this should be set no lower than
                        :math:`x_{\text{r}}`, i.e. ``'range'``. At the same
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
\  **Example 0**   \  Extend :math:`x_{\text{r}}` all the way to
                      :math:`5.5 x_{\text{s}}`, for the gravitational
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
                                 'particles': '2*cbrt(Ñ)',
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
                             'realization correction': {
                                 'default': True,
                             },
                             'k_max': {
                                 'default': 'nyquist',
                             },
                             'bins per decade': {
                                 'default': {
                                     '  4*k_min':  4,
                                     '100*k_min': 40,
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
                        sizes to use for each component. See the
                        ``potential_options``
                        :ref:`parameter <potential_options>` for the use of
                        the ``'Ñ'`` notation.

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
                        for upstream particle interpolations. Possible values
                        are ``'sc'`` (or ``False``) for a simple cubic lattice
                        (meaning no interlacing), ``'bcc'`` (or ``True``) for
                        a body-centered cubic lattice (meaning standard
                        interlacing involving two relatively shifted particle
                        interpolations) or ``'fcc'`` for a face-centered cubic
                        lattice (meaning interlacing involving four relatively
                        shifted particle interpolations).

                      * ``'realization correction'``: Specifies whether to
                        also correct for the current realisation
                        (cosmic variance) when computing *corrected* power
                        spectra (see the ``powerspec_select``
                        :ref:`parameter <powerspec_select>`), as opposed to
                        only correcting for noise stemming from the binning
                        procedure.

                      * ``'k_max'``: Specifies the largest :math:`k` mode to
                        include in the power spectrum output files (data files
                        and plots). If given as a ``str``, ``'nyquist'`` is
                        dynamically substituted for the Nyquist mode,

                        .. math::

                           \require{upgreek}
                           k_{\text{Nyquist}} = \biggl\lfloor\frac{n_{\text{ps}}}{2}\biggr\rfloor \times \frac{2\uppi}{L_{\text{box}}}\, ,

                        with :math:`L_{\text{box}}` given by ``boxsize`` and
                        :math:`n_{\text{ps}}` the global grid size used for
                        the power spectrum.

                        .. note::
                           Setting ``'k_max'`` above ``'sqrt(3)*nyquist'``
                           will not yield further data points, as no further
                           data is available in the 3D grid.

                      * ``'bins per decade'``: Specifies the number of power
                        spectrum bins per decade in :math:`k`. By specifying
                        different numbers at different :math:`k`, a running
                        (logarithmically interpolated) number of bins is
                        obtained. When the :math:`k`\ 's are given as
                        ``str``\ s, the following sub\ ``str``\ s are
                        dynamically substituted:

                        * ``'k_min'``: ``2*π/boxsize``.
                        * ``'k_max'``: Value set for ``'k_max'``, though at
                          most ``sqrt(3)*(gridsize//2)*k_min``.
                        * ``'gridsize'``: ``gridsize``.
                        * ``'nyquist'``: ``gridsize//2*k_min``.
                        * ``'k_fundamental'``/``'k_f'``: ``2*π/boxsize``.

                      * ``'tophat'``: Included in the power spectrum data
                        files is also the root-mean-square density variation
                        :math:`\sigma_R`, smoothed with a spherical
                        top-hat filter :math:`W(s)` of radius :math:`R`:

                        .. math::

                           \require{upgreek}
                           \sigma_R &= \frac{1}{2\uppi^2} \left( \int k^2 W^2(kR) P(k)\,\mathrm{d}k \right)^{\frac{1}{2}}\, , \\
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
\  **Example 1**   \  Employ many (small) bins of the same (logarithmic) size
                      for all :math:`k` of every power spectrum:

                      .. code-block:: python3

                         powerspec_options = {
                             'bins per decade': {
                                 'all'             : 100,
                                 'all combinations': ...,
                             },
                         }

                      This may be shortened to

                      .. code-block:: python3

                         powerspec_options = {
                             'bins per decade': 100,
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
                                 'matter': False,  # or 'sc'
                             }
                         }

== =============== == =



------------------------------------------------------------------------------



.. _bispec_options:

``bispec_options``
..................
== =============== == =
\  **Description** \  Specifications for bispectrum computations
                      and dumps
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'configuration': {
                                 'default': ('equilateral', 20),
                             },
                             'shellthickness': {
                                 'default': [
                                     {
                                         '1*k_fundamental': '0.25*k_fundamental',
                                         '4*k_fundamental': 'max(3*k_fundamental, 1/20*log(10)*k)',
                                     },
                                     {
                                         '1*k_fundamental': '0.25*k_fundamental',
                                         '4*k_fundamental': 'max(3*k_fundamental, 1/20*log(10)*k)',
                                     },
                                     {
                                         '1*k_fundamental': '0.25*k_fundamental',
                                         '4*k_fundamental': 'max(3*k_fundamental, 1/20*log(10)*k)',
                                     },
                                 ]
                             },
                             'upstream gridsize': {
                                 'particles': '2*cbrt(Ñ)',
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
                             'significant figures': {
                                 'default': 8,
                             },
                         }
-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` of several individual sub-parameters,
                      specifying details of how to compute and save bispectra.
                      All sub-parameters are themselves
                      :ref:`component selections <components_and_selections>`.

                      For computing a bispectrum, one or more components are
                      first interpolated onto individual *upstream* grids,
                      possibly using deconvolution and interlacing, after
                      which they are added together in Fourier space,
                      producing a *global* grid. This scheme is similar to the
                      one used for potentials, except here we never go back to
                      real space, neither do we interpolate anything back to
                      the particles. See the ``potential_options``
                      :ref:`parameter <potential_options>` for a walk-through
                      of the scheme.

                      Many of the sub-parameters are the same as within the
                      ``powerspec_options``
                      :ref:`parameter <powerspec_options>`, and have the same
                      meaning. These will not be reiterated here. The
                      remaining sub-parameters are described below:

                      * ``'configuration'``: Specifies the triangle
                        configurations for which to compute the bispectrum for
                        each component. Rather than writing the bispectrum as
                        a function of the length of the three wave vectors
                        :math:`B(k_1, k_2, k_3)`, CO\ *N*\ CEPT makes use of
                        parametrisation :math:`B(k, t, \mu)`, where

                        .. math::

                           k &\equiv k_1\,,\\
                           t &\equiv \frac{k_2}{k_1}\,,\\
                           \mu &\equiv \frac{k_1^2 + k_2^2 - k_3^2}{2k_1k_2}\,,

                        or conversely

                        .. math::

                           k_1 &\equiv k\,,\\
                           k_2 &\equiv tk_1\,,\\
                           k_3 &\equiv \sqrt{k_1^2 + k_2^2 - 2\mu k_1k_2}\,.

                        Using this parametrisation, the configurations to use
                        may be specified in multiple different ways:

                        * ``(k, t, μ)``: Single configuration, corresponding
                          to one bispectrum bin. Note that ``k``, ``t`` and
                          ``μ`` must each be a single number, given in that
                          order. The bin represents the average of triangles
                          with :math:`(k, t, \mu)` in the vicinity of the
                          values specified, in accordance with the
                          ``'shellthickness'`` sub-parameter (see below).
                        * ``{'k': <...>, 't': <...>, 'μ': <...>}``: Multiple
                          configurations spanning a chunk of configuration
                          space. Here each ``<...>`` is stand-in for either a
                          single number or multiple numbers (e.g. within in a
                          ``list``). All configurations in the Cartesian
                          product of the three ``<...>`` will be included.
                        * ``'<named-configuration>'``: Multiple configurations
                          spanning a named portion of configuration space.
                          Many typical configurations are known to
                          CO\ *N*\ CEPT by name. These include:

                          * ``'equilateral'``: :math:`t = 1`,
                            :math:`\mu = \frac{1}{2}`,
                            :math:`\,(k_1 = k_2 = k_3).`
                          * ``'stretched'``: :math:`t = \frac{1}{2}`,
                            :math:`\mu = 1`, :math:`\,(k_1 = 2k_2 = 2k_3).`
                          * ``'squeezed'``: :math:`t = 1`, :math:`\mu = 0.99`,
                            :math:`\,(k_1 = k_2, k_3 \approx 0).`
                          * ``'isosceles right'``: :math:`t = 1/\sqrt{2}`,
                            :math:`\mu = 1/\sqrt{2}`,
                            :math:`\,(k_1 = \sqrt{2}k_2 = \sqrt{2}k_3).`
                          * ``'L-isosceles'``: :math:`t = 1`,
                            :math:`\frac{1}{2} \le \mu < 1`,
                            :math:`\,(k_1 = k_2 \ge k_3).`
                          * ``'S-isosceles'``:
                            :math:`\frac{1}{2} \le t \le 1`,
                            :math:`\mu = (2t)^{-1}`,
                            :math:`\,(k_1 \ge k_2 = k_3).`
                          * ``'elongated'``/``'flattened'``/``'folded'``/``'linear'``:
                            :math:`\frac{1}{2} \le t < 1`, :math:`\mu = 1`,
                            :math:`\,(k_1 = k_2 + k_3).`
                          * ``'right'``: :math:`1/\sqrt{2} \le t = \mu < 1`
                            :math:`\,(k_1^2 = k_2^2 + k_3^2).`
                          * ``'acute'``: :math:`1/\sqrt{2} \le t \le 1`,
                            :math:`(2t)^{-1} \le \mu < 1`,
                            :math:`\,(k_1^2 \le k_2^2 + k_3^2).`
                          * ``'obtuse'`` :math:`1/\sqrt{2} \le \mu < 1`,
                            :math:`(2\mu)^{-1} \le t < \mu`,
                            :math:`\,(k_1^2 \ge k_2^2 + k_3^2).`
                          * ``'all'`` :math:`\frac{1}{2} \le t \le 1`,
                            :math:`(2t)^{-1} \le \mu < 1`
                            :math:`\,(k_1 \ge k_2 \ge k_3).`

                          In all of th above named configurations,
                          :math:`k = k_1` goes from ``5*k_f`` to
                          ``2/3*nyquist``, with ``k_f = 2*π/boxsize`` and
                          ``nyquist = gridsize//2*k_f``.

                          For all of the bove named configurations,
                          the convention that :math:`k_1 \ge k_2 \ge k_3` is
                          used, corresponding to

                          .. math::

                             \frac{1}{2} &\le t \le 1\,, \\
                             \frac{1}{2} &\le \mu \le 1\,, \\
                             \frac{1}{2} &\le t\mu\,.

                          Note that this restriction is not imposed when
                          manually specifying :math:`k`, :math:`t`,
                          :math:`\mu`.

                        * (``'<named-configuration>'``, n): Named portion of
                          configuration space like above, with the number of
                          bins set through the integer ``n``. This is
                          interpreted as the number of bins per decade along
                          the :math:`k = k_1` dimension, with the total number
                          further depending on the global grid size in use for
                          the bispectrum measurement. The :math:`t` and
                          :math:`\mu` dimensions are likewise subdivided,
                          using the same total number of cuts as for the
                          :math:`k` dimension.

                          When ``n`` is not specified (corresponding to the
                          previous bullet point), ``n`` defaults to ``20``.

                          .. tip::
                             From the above, it can be shown that the total
                             number of bispectrum bins in the named
                             configurations grows approximately like
                             :math:`\mathcal{O}[(n^d(\ln (n_{\text{bs}}) - 1)^d]`,
                             with :math:`n_{\text{bs}}` the global grid size
                             used for the bispectrum computation and :math:`d`
                             the dimensionality of the selected chunk of
                             parameter space:

                             * :math:`d = 1`: ``'equilateral'``,
                               ``'stretched'``, ``'squeezed'``,
                               ``'isosceles right'``.
                             * :math:`d = 2`: ``'L-isosceles'``,
                               ``'S-isosceles'``, ``'elongated'``,
                               ``'right'``.
                             * :math:`d = 3`: ``'acute'``, ``'obtuse'``,
                               ``'all'``.

                        * Finally, multiple specifications of any of the above
                          kinds can be simultaneously selected by writing them
                          together in a ``list``.

                      * ``'shellthickness'``: Specifies the thickness of the
                        three :math:`k` shells (one for each of :math:`k_1`,
                        :math:`k_2`, :math:`k_3`) making up each bispectrum
                        bin. By specifying different thicknesses at different
                        :math:`k`, a running (logarithmically interpolated)
                        thickness is obtained. Both the thickness values
                        themselves and the :math:`k` values (shell radii) at
                        which they apply can be given as either numbers (in
                        units of inverse length) or as ``str``\ s. In the
                        latter case, the following sub\ ``str``\ s are
                        dynamically substituted:

                        * ``'k_min'``: ``2*π/boxsize``.
                        * ``'k_max'``: ``sqrt(3)*(gridsize//2)*k_min``.
                        * ``'gridsize'``: ``gridsize``.
                        * ``'nyquist'``: ``gridsize//2*k_min``.
                        * ``'k_fundamental'``/``'k_f'``: ``2*π/boxsize``.

                        For specifying individual shell thicknesses for the
                        :math:`k_1`, :math:`k_2` and :math:`k_3` shells,
                        ``'shellthickness'`` should be specified as a ``list``
                        of three ``dict``\ s. If instead ``'shellthickness'``
                        is given as just a single such ``dict``, the same
                        shell thickness will be used for all three shells.

                      .. note::
                         For all sub-parameters except
                         ``'upstream gridsize'``, the keys used within the
                         :ref:`component selection <components_and_selections>`
                         sub-\ ``dict``\ s may refer to either a single
                         component or a combination of components. In the case
                         of the latter, this designates a combined (auto)
                         bispectrum.
-- --------------- -- -
\  **Example 0**   \  Use the equilateral configurations for bispectra of the
                      component with a name/species of ``'matter'``:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': 'equilateral',
                             },
                         }

                      If we want more control over the number of bispectrum
                      bins (the density of sampling points), we can specify
                      the number of bins/subdivisions per decade along the
                      :math:`k = k_1` dimension (with the number of
                      subdivisions along the :math:`t` and :math:`\mu`
                      dimensions also adapting accordingly):

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': ('equilateral', 30),
                             },
                         }

                      Instead of using the predefined name ``'equilateral'``,
                      we can specify such configurations --- defined by having
                      :math:`t = 1` and :math:`\mu = \frac{1}{2}`
                      --- manually:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': {
                                     'k': 'logspace(log10(k_f), log10(nyquist), int(30*log10(nyquist/k_f)))',
                                     't': 1,
                                     'μ': 0.5,
                                 },
                             },
                         }

                      The above specification for ``'k'`` results in 30 values
                      per decade, placed logarithmically equidistant between
                      ``k_f`` and ``nyquist``; a wider parameter range than
                      what is used by the built-in ``'equilateral'``
                      configuration.

                      If we prefer, we might instead list each equilateral bin
                      individually:

                      .. code-block:: python3

                         _gridsize = 128  # global grid size for bispectrum
                         bispec_options = {
                             'configuration': {
                                 'matter': [
                                     (k, 1, 0.5)
                                     for k in logspace(
                                         log10(2*π/boxsize),
                                         log10(2*π/boxsize*(_gridsize//2)),
                                         int(30*log10(_gridsize//2)),
                                     )
                                 ],
                             },
                         }

                      Note that in the above we have to reference the grid
                      size explicitly, whereas previously this was encoded in
                      ``'nyquist'``. Likewise we previously made use of the
                      fundamental frequency ``'k_f'``, whereas just above we
                      write this out explicitly as ``2*π/boxsize``. We can
                      reintroduce ``'k_f'`` to (perhaps) simplify the above:

                      .. code-block:: python3

                         _gridsize = 128  # global grid size for bispectrum
                         bispec_options = {
                             'configuration': {
                                 'matter': [
                                     (f'{q}*k_f', 1, 0.5)
                                     for q in logspace(
                                         0,
                                         log10(_gridsize//2),
                                         int(30*log10(_gridsize//2)),
                                     )
                                 ],
                             },
                         }
-- --------------- -- -
\  **Example 1**   \  Use a constant shell thickness equal to three times the
                      fundamental frequency, for shell :math:`k_1`,
                      shell :math:`k_2` and shell :math:`k_3`, for
                      every bispectrum:

                      .. code-block:: python3

                         bispec_options = {
                             'shellthickness': {
                                 'all'             : ['3*k_f', '3*k_f', '3*k_f'],
                                 'all combinations': ...,
                             },
                         }

                      This may be shortened to

                      .. code-block:: python3

                         bispec_options = {
                             'shellthickness': ['3*k_f', '3*k_f', '3*k_f'],
                         }

                      or indeed

                      .. code-block:: python3

                         bispec_options = {
                             'shellthickness': '3*k_f',
                         }
-- --------------- -- -
\  **Example 2**   \  Use the equilateral configurations for bispectra of the
                      component with a name/species of ``'matter'``, with 30
                      bins per decade along the :math:`k` dimension.

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': ('equilateral', 30),
                             },
                         }

                      Ideally, the concentric shells of different bins should
                      not overlap, as otherwise the same underlying modes will
                      each contribute to multiple bins. At the same time, we
                      would like to include all available modes within our
                      bins, and so together they should cover all of
                      configuration space, i.e. we want there to be no gaps
                      between the concentric shells. Given the 30
                      logarithmically equidistant shell radii above, we can
                      match the shell thicknesses in order to achieve this:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': ('equilateral', 30),
                             },
                             'shellthickness': {
                                 'matter': '1/30*log(10)*k',
                             },
                         }

                      Note that with the growing shell thickness above,
                      the usual shell thickness of :math:`3k_{\text{f}}` is not
                      achieved until :math:`k \sim 40k_{\text{f}}`. We may
                      introduce a minimum shell thickness --- say of
                      :math:`1.5k_{\text{f}}` --- like so:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': ('equilateral', 30),
                             },
                             'shellthickness': {
                                 'matter': 'max(1.5*k_f, 1/30*log(10)*k)',
                             },
                         }

                      The above minimum shell thickness of
                      :math:`1.5k_{\text{f}}` applies for
                      :math:`k \lesssim 20k_{\text{f}}`, though for the very
                      lowest :math:`k` we might want a smaller value. We can
                      introduce a lowest thickness of e.g.
                      :math:`0.25k_{\text{f}}` below e.g.
                      :math:`1k_{\text{f}}`, with the previous thickness
                      specification being applied above e.g.
                      :math:`10k_{\text{f}}`:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': ('equilateral', 30),
                             },
                             'shellthickness': {
                                 'matter': {
                                      '1*k_f': '0.25*k_f',
                                     '10*k_f': 'max(1.5*k_f, 1/30*log(10)*k)',
                                 },
                             },
                         }

                      Between the two control points :math:`1k_{\text{f}}`
                      and :math:`10k_{\text{f}}`, the shell thickness is
                      obtained through logarithmic interpolation. For example,
                      the shell thickness at :math:`4k_{\text{f}}` becomes
                      :math:`\sim 1.0k_{\text{f}}`, given the above
                      specification.

                      Note the similarity with the above shell thickness
                      settings and the default settings.
-- --------------- -- -
\  **Example 3**   \  Measure the bispectrum for all isosceles configurations
                      --- i.e. both L-isosceles and S-isosceles --- for the
                      component with a name/species of ``'matter'``.

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': ['L-isosceles', 'S-isosceles'],
                             },
                         }

                      We might want to explicitly control the number of bins
                      for each of two configuration subspaces:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': [('L-isosceles', 15), ('S-isosceles', 25)],
                             },
                         }

                      .. caution::
                         Due to the many possible configuration
                         specifications, the parsing of these specifications
                         is less robust than what is usually the case. In
                         particular, swapping ``list``\ s for ``tuple``\ s
                         or vice versa can lead to erroneous parsing,
                         ultimately crashing the program. That is, while

                         .. code-block:: python3

                            ['L-isosceles', 'S-isosceles']

                         is a valid specification, this is not:

                         .. code-block:: python3

                            ('L-isosceles', 'S-isosceles')  # ❌

                      Note that the two sets of bins are written without
                      separation in the output data file. However, the order
                      in which the bins appear will always match the order in
                      which they are specified.

                      .. tip::
                         For the case above (and most others), the transition
                         from one configuration subset to another within a
                         bispectrum data file can easily be located as the row
                         where :math:`k` *de*\ creases in relation to the
                         previous row.
-- --------------- -- -
\  **Example 4**   \  Use a squeezed configurations for bispectra of the
                      component with a name/species of ``'matter'``:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': 'squeezed',  # or e.g. (squeezed, 20)
                             },
                         }

                      The squeezed configurations are more tricky than other
                      configurations, as the :math:`k_3` shell vanishes in
                      the squeezed limit :math:`t\rightarrow 1`,
                      :math:`\mu\rightarrow 0`, while a reliable bispectrum
                      measurement requires each shell to contain a substantial
                      amount of grid points. For the built-in ``'squeezed'``,
                      he values ``t = 1``, ``μ = 0.99`` are chosen. We can use
                      a less restrictive ``μ = 0.95`` --- leading to a less
                      noisy bispectrum --- by specifying the configurations
                      manually:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': {
                                     'k': (
                                         'logspace(log10(5*k_f), '
                                         'log10(2/3*nyquist), '
                                         'int(20*log10(2/3*nyquist/(5*k_f))))'
                                     ),
                                     't': 1,
                                     'μ': 0.95,
                                 },
                             },
                         }

                      In the above we have kept the :math:`k = k_1` range the
                      same as the default; 20 points logarithmically
                      equidistant between 5 times the fundamental frequency
                      and two-thirds times the Nyquist frequency. Note however
                      that some of these configurations are rejected when
                      using the built-in ``'squeezed'``, as :math:`k_3` is
                      deemed too small.

                      With :math:`k_3` much smaller than :math:`k_1` and
                      :math:`k_2` it might makes sense to assign the
                      :math:`k_3` shell a thickness that is different from
                      the one used by the :math:`k_1` and :math:`k_2` shell:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': {
                                     'k': (
                                         'logspace(log10(5*k_f), '
                                         'log10(2/3*nyquist), '
                                         'int(20*log10(2/3*nyquist/(5*k_f))))'
                                     ),
                                     't': 1,
                                     'μ': 0.95,
                                 },
                             },
                             'shellthickness': {
                                 'matter': ['3*k_f', '3*k_f', '1.5*k_f'],
                             },
                         }
-- --------------- -- -
\  **Example 5**   \  Imitate the way
                      `Pylians <https://pylians3.readthedocs.io/>`__ computes
                      bispectra:

                      * Specify :math:`k_1 = |\boldsymbol{k}_1|` and
                        :math:`k_2 = |\boldsymbol{k}_2|` as well as the angle
                        :math:`\theta = \cos^{-1}(\hat{\boldsymbol{k}}_1 \cdot \hat{\boldsymbol{k}}_2)`
                        (opposite sign convention compared to CO\ *N*\ CEPT,
                        making :math:`\mu = -\cos\theta`).
                      * Use a constant shell thickness of twice the
                        fundamental frequency.

                      Use :math:`k_1 = 1.1\, h\, \text{Mpc}^{-1}`,
                      :math:`k_2 = 0.8\, h\, \text{Mpc}^{-1}` and :math:`50`
                      values of :math:`\theta` between :math:`0` and
                      :math:`\require{upgreek}\uppi`. Apply all of this for
                      the component with a name/species of ``'matter'``:

                      .. code-block:: python3

                         _k1 = 1.1*h/Mpc
                         _k2 = 0.8*h/Mpc
                         _θ = linspace(0, π, 50)
                         bispec_options = {
                             'configuration': {
                                 'matter': {
                                     'k': _k1,
                                     't': _k2/_k1,
                                     'μ': -cos(_θ),
                                 },
                             },
                             'shellthickness': {
                                 'matter': '2*k_f',
                             },
                         }

                      .. note::
                         In order to fully reduce the bispectrum computation
                         within CO\ *N*\ CEPT to one consistent with that of
                         `Pylians <https://pylians3.readthedocs.io/>`__, one
                         should further:

                         * Disable
                           :ref:`shell anti-aliasing <bispec_antialiasing>`.
                         * Disable interlacing (specified trough
                           ``bispec_options['interlace']``).
                         * Switch to :ref:`cell-vertex <cell_centered>` grid
                           discretisation.
-- --------------- -- -
\  **Example 6**   \  Sample the full configuration space for the component
                      with a name/species of ``'matter'``:

                      .. code-block:: python3

                         bispec_options = {
                             'configuration': {
                                 'matter': ('all', 5),
                             },
                         }

                      We recall that the full configuration space is
                      parametrised as

                      .. math::

                         \frac{1}{2} &\le t \le 1\,, \\
                         \frac{1}{2} &\le \mu \le 1\,, \\
                         \frac{1}{2} &\le t\mu\,,

                      which uniquely covers each configuration (triangle
                      shape) for a given :math:`k` (triangle size). While the
                      built-in ``'all'`` samples configuration space evenly
                      (logarithmically in :math:`k`, linear in :math:`t` and
                      :math:`\mu`), we might instead wish to sample randomly.

                      .. code-block:: python3

                         np.random.seed(42)  # for consistent bins
                         def _sample():
                             r = np.random.random(3)
                             k = f'k_min*(k_max/k_min)**{r[0]}'
                             t = (1 + r[1])/2
                             μ = (1 + r[2])/2
                             if t*μ < 0.5:
                                 return _sample()
                             return k, t, μ
                         bispec_options = {
                             'configuration': {
                                 'matter': [_sample() for _ in range(100)],
                             },
                         }

                      Note that the above samples the entire possible
                      :math:`k = k_1` range, whereas this range is reduced
                      for the built-in ``'all'`` (as well as the other named
                      configurations).
== =============== == =



------------------------------------------------------------------------------



.. _bispec_antialiasing:

``bispec_antialiasing``
.......................
== =============== == =
\  **Description** \  Specifies whether to enable anti-aliasing for
                      bispectrum shells
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True
-- --------------- -- -
\  **Elaboration** \  Numerically, each bispectrum shell is a collection of
                      Fourier grid cells, each positioned a distance (measured
                      from their centres) of about :math:`k` from the origin
                      (with :math:`k = k_1` for the first shell,
                      :math:`k = k_2` for the second shell, :math:`k = k_3`
                      for the third shell). Call this distance the radial
                      coordinate of the cell. With shell anti-aliasing
                      disabled (not the default), a cell is included in the
                      shell if its radial coordinate lies between
                      :math:`k - s/2` and :math:`k + s/2`, :math:`s` being the
                      shell thickness of the shell with radius :math:`k`.

                      The simple scheme for cell inclusion/exclusion within a
                      shell described above comes with a number of drawbacks,
                      arising from allowing the discrete nature of the
                      underlying grid to spread into the shell. A more
                      sophisticated approach is to treat the shells as being
                      truly spherical and continuous, with cells on the
                      boundary being taken into account in proportion to their
                      volumetric overlap with the shell. This is what we refer
                      to as shell anti-aliasing (enabled by default).

                      Note that with shell anti-aliasing, the inner and outer
                      shell radii --- related by their difference equalling
                      the shell thickness :math:`s` --- can now be chosen such
                      that the average (weighted) radial coordinate of the
                      cells within the shell exactly equals the radius
                      :math:`k` of the shell, i.e. a measured bispectrum value
                      :math:`B(k_1, k_2, k_3)` really do belong *exactly* to
                      the given :math:`k_1`, :math:`k_2`, :math:`k_3`. This is
                      unlike the non-anti-aliased case where small changes to
                      :math:`k_1`, :math:`k_2`, :math:`k_3` do not alter the
                      measured value :math:`B(k_1, k_2, k_3)`.

                      .. note::
                         For all but the smallest :math:`k`, CO\ *N*\ CEPT in
                         fact chooses the inner and outer shell radii not so
                         that the average radial coordinate of the cells
                         results in the radius :math:`k` of the shell, but
                         instead so that the average of the logarithm of the
                         radial coordinates of the cells results in
                         :math:`\log k`.
-- --------------- -- -
\  **Example 0**   \  Turn off bispectrum shell anti-aliasing:

                      .. code-block:: python3

                         bispec_antialiasing = False

                      Though generally not preferable, this is useful if the
                      resulting bispectra are to be compared with ones
                      computed by a different code which does not have this
                      feature.
== =============== == =



------------------------------------------------------------------------------



.. _class_dedicated_spectra:

``class_dedicated_spectra``
...........................
== =============== == =
\  **Description** \  Specifies whether to carry out a dedicated CLASS
                      computation for use with perturbation theory spectra
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         False
-- --------------- -- -
\  **Elaboration** \  When ``False`` (the default), rather than running CLASS,
                      already obtained CLASS results are reused whenever a
                      perturbation theory spectrum (a linear power spectrum or
                      a tree-level bispectrum) is requested, even if this
                      does not cover the entire :math:`k` region in question.
                      This often leads to a few missing (``NaN``) values in
                      the perturbation theory columns of output spectra, but
                      saves time as fewer invocations of CLASS are needed.

                      .. note::
                         This will only affect the perturbation theory output
                         in power spectrum and bispectrum data files; the
                         CLASS data used internally for simulation purposes
                         will always be complete.
-- --------------- -- -
\  **Example 0**   \  Always rerun CLASS as necessary, ensuring fully
                      populated perturbation theory data in spectral output:

                      .. code-block:: python3

                         class_dedicated_spectra = True
== =============== == =



------------------------------------------------------------------------------



.. _class_modes_per_decade:

``class_modes_per_decade``
..........................
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
                      relatively sparse tabulation of perturbations, except
                      around baryon acoustic oscillations, where more detail
                      is added.

                      If you seek highly resolved linear power spectra or
                      tree-level, make sure to not just increase
                      ``class_modes_per_decade``, but also enable the
                      ``class_dedicated_spectra``
                      :ref:`parameter <class_dedicated_spectra>`.
-- --------------- -- -
\  **Example 0**   \  Use a constant :math:`20` modes per decade, i.e.
                      :math:`20` values of :math:`k` placed logarithmically
                      equidistant between
                      :math:`k = 10^{-3}\,\text{Mpc}^{-1}` and
                      :math:`k = 10^{-2}\,\text{Mpc}^{-1}`, between
                      :math:`k = 10^{-2}\,\text{Mpc}^{-1}` and
                      :math:`k = 10^{-1}\,\text{Mpc}^{-1}`, etc:

                      .. code-block:: python3

                         class_modes_per_decade = 20
-- --------------- -- -
\  **Example 1**   \  Use :math:`50` modes per decade around
                      :math:`k = 10^{-4}\,\text{Mpc}^{-1}` and :math:`10`
                      modes per decade around
                      :math:`k = 10^1\,\text{Mpc}^{-1}`, with a running
                      number per decade in between found through logarithmic
                      interpolation (resulting in e.g. :math:`42` modes per
                      decade around :math:`k = 10^{-3}\,\text{Mpc}^{-1}`):

                      .. code-block:: python3

                         class_modes_per_decade = {
                             1e-4/Mpc: 50,
                             1e+1/Mpc: 10,
                         }

== =============== == =

