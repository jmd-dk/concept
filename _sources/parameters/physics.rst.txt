Physics
-------
The 'physics' parameter category contains parameters specifying various
physical models and schemes to be used.

.. contents::
   :local:
   :depth: 1



------------------------------------------------------------------------------



.. _select_forces:

``select_forces``
.................
== =============== == =
\  **Description** \  Specifies which forces to enable
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {}  # no forces enabled

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      ``dict``, mapping components to sub-\ ``dict``\ s of the
                      form

                      .. code-block:: python3

                         {force: method}

                      specifying which forces/interactions affect the given
                      component, as well as what method/implementation to use
                      for the given forces. Currently, the only forces in
                      CO\ *N*\ CEPT are ``'gravity'`` and ``'lapse'``, with the
                      latter having to do with gravitational
                      :ref:`time dilation <decaying_cold_dark_matter>`. The
                      implemented methods for the forces are:

                      * ``'gravity'``:

                        * ``'pp'`` (PP, particle-particle)
                        * ``'pm'`` (PM, particle-mesh)
                        * ``'p3m'`` (P³M, particle-particle-mesh)

                      * ``'lapse'``:

                        * ``'pm'`` (PM, particle-mesh)

                      For details on these methods, consult the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'.
                      As ``select_forces`` is empty by default, no forces (not
                      even gravity) are enabled. If however a global potential
                      grid size is specified within the ``potential_options``
                      :ref:`parameter <potential_options>`, this places the
                      following default entries within ``select_forces``:

                      .. code-block:: python3

                         {
                             'particles': {
                                 'gravity': 'p3m',
                             },
                             'fluid': {
                                 'gravity': 'pm',
                             },
                         }

                      so that particle components will participate in the
                      gravitational interaction using P³M, while fluid
                      components will participate in the gravitational
                      interaction using PM.

                      .. note::
                         All interactions for fluid components may only ever
                         use the PM method

                      .. note::
                         With the gravitational force specified for particle
                         and fluid components as above, gravity will really be
                         applied to such components in the following manner:

                         * Gravity applied to particle components:

                            * From particle components:

                               * P³M (long-range)
                               * P³M (short-range)

                            * From fluid components:

                               * PM

                         * Gravity applied to fluid components:

                            * From particle and fluid components:

                               * PM

                         Besides P³M being split up into a long-range and a
                         short-range part, we see that the interactions
                         between particle and fluid components make use of
                         PM, even though PM is only selected for the fluid
                         components.

                      For more information on these
                      mesh-based methods, see the ``potential_options``
                      :ref:`parameter <potential_options>`. For a more casual
                      walk-through of how to specify forces and methods within
                      CO\ *N*\ CEPT, see the
                      :doc:`tutorial </tutorial/gravity>`.

-- --------------- -- -
\  **Example 0**   \  Explicitly specify the component with a name/species of
                      ``'matter`` to be under the influence of gravity, using
                      the P³M method:

                      .. code-block:: python3

                         select_forces = {
                             'matter': {
                                 'gravity': 'p3m',
                             },
                         }

                      As P³M is the default gravitational method, we can
                      shorten this to

                      .. code-block:: python3

                         select_forces = {
                             'matter': 'gravity',
                         }

                      As mentioned above, this can be shortened further by
                      simply removing any mention of ``select_forces`` from
                      the parameter file, provided that a global grid size is
                      set within the ``potential_options``
                      :ref:`parameter <potential_options>`.

-- --------------- -- -
\  **Example 1**   \  Explicitly specify the component with a name/species of
                      ``'matter`` to be under the influence of gravity, using
                      the PM method:

                      .. code-block:: python3

                         select_forces = {
                             'matter': {
                                 'gravity': 'pm',
                             },
                         }

== =============== == =


------------------------------------------------------------------------------



.. _select_species:

``select_species``
..................
== =============== == =
\  **Description** \  Specifies the species of components read in from snapshots
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'default': 'matter',
                         }

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      ``dict`` for mapping components to species. While the
                      ``'species'`` of a component is explicitly specified in
                      the ``initial_conditions``
                      :ref:`parameter <initial_conditions>`, no species are
                      necessarily defined for components read in from
                      snapshots. For components within snapshots of the
                      ``'concept'`` :ref:`type <snapshot_type>`, their species
                      are explicitly included within the snapshots. For
                      components read from ``'gadget'``
                      :ref:`type <snapshot_type>` snapshots, their GADGET
                      particle type (see table 3 of the
                      `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__)
                      is known, but this needs to be mapped to a species in
                      CO\ *N*\ CEPT. This is what this parameter is for.

                      .. note::
                         Though a species is explicitly defined for each
                         component within a ``'concept'`` snapshot, the species
                         to use within a simulation may be overruled by
                         setting this parameter

-- --------------- -- -
\  **Example 0**   \  Explicitly map the GADGET particle type 1 (``halo``) to
                      correspond to the ``'matter'`` species within
                      CO\ *N*\ CEPT:

                      .. code-block:: python3

                         select_species = {
                             'GADGET halo': 'matter',
                         }

                      .. note::
                         The name ``'GADGET halo'`` will automatically be
                         assigned to the component containing the particles of
                         type 1. See table 3 of the
                         `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__
                         for other names.

-- --------------- -- -
\  **Example 1**   \  Map the GADGET particle type 2 (``disk``) to the
                      ``'neutrino'`` species within CO\ *N*\ CEPT, while
                      mapping everything else to ``'matter'``:

                      .. code-block:: python3

                         select_species = {
                             'GADGET disk': 'neutrino',
                             'all'        : 'matter',
                         }

== =============== == =



------------------------------------------------------------------------------



.. _realization_options:

``realization_options``
.......................
== =============== == =
\  **Description** \  Specifies how to realise different components
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'gauge': {
                                 'default': 'N-body',
                             },
                             'back-scale': {
                                'default': False,
                             },
                             'LPT': {
                                 'default': 1,
                             },
                             'non-Gaussianity': {
                                 'default': 0.0,
                             },
                             'structure': {
                                 'default': 'non-linear',
                             },
                             'compound': {
                                 'default': 'linear',
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` of several individual sub-parameters,
                      specifying how individual components are to be
                      *realised*. All sub-parameters are themselves
                      :ref:`component selections <components_and_selections>`.
                      A realisation refers to the generation of
                      initial conditions --- i.e. particle positions and
                      momenta for particle components and fluid grids like
                      energy and momentum densities for fluid components ---
                      but also late-time re-realisation of fluid grids for
                      fluid components.

                      Each sub-parameter is described below:

                      * ``'gauge'``: Sets the general relativistic gauge in
                        which to perform the realisation, i.e. the gauge of
                        the transfer functions :math:`T(a, k)` to use.
                        Available gauges are ``'N-body'``
                        (:math:`N`-body gauge), ``'synchronous'``
                        (synchronous gauge) or ``'Newtonian'`` (conformal
                        Newtonian / longitudinal gauge).

                      * ``'back-scale'``: Specifies whether to do particle
                        realisation using a back-scaled density transfer
                        function, rather than using the transfer function
                        as is. That is, without back-scaling, the raw density
                        transfer function :math:`T_\delta(a, k)` is used
                        for realisation at time (scale factor value) :math:`a`
                        of particle positions, and similarly the raw velocity
                        transfer function :math:`T_\theta(a, k)` is used for
                        the particle velocities. With back-scaling, the
                        density transfer function is substituted by a
                        scaled-back version of itself at time :math:`a = 1`:

                        .. math::

                           T_\delta(a, k) \rightarrow \frac{D(a)}{D(a=1)} T_\delta(a=1, k)\,,

                        where :math:`D(a)` is the linear, Newtonian growth
                        factor. Also, the velocity transfer function
                        :math:`T_\theta(a, k)` is no longer used at all,
                        but replaced with a Newtonian approximation

                        .. math::

                           T_\theta(a, k) \rightarrow -aH(a)f(a) \biggl(\frac{D(a)}{D(a=1)} T_\delta(a=1, k) \biggr)\,,

                        where
                        :math:`f(a) \equiv \mathrm{d}\ln D(a) / \mathrm{d}\ln a`
                        is the linear, Newtonian growth rate. Back-scaling is
                        only possible for particle components.

                      * ``'LPT'``: Specifies the order of Lagrangian
                        perturbation theory to use when realising particle
                        components. Orders :math:`1` and :math:`2` are
                        available. While the first order (1LPT) is carried out
                        relativistically (if not using back-scaling), the
                        second-order (2LPT) contributions are always
                        constructed in a Newtonian fashion (though they are
                        build from the (relativistic) 1LPT results).

                      * ``'non-Gaussianity'``: Sets the amount of local
                        non-Gaussianity to include in realisations. This is
                        simply the number :math:`f_{\text{NL}}` in the
                        transformation

                        .. math::

                           \delta(\boldsymbol{x}) \rightarrow \delta(\boldsymbol{x}) + f_{\text{NL}}\delta^2(\boldsymbol{x})\,,

                        where the :math:`\delta^2(\boldsymbol{x})` term
                        introduces local non-Gaussianity into the otherwise
                        Gaussian density contrast field
                        :math:`\delta(\boldsymbol{x})`.

                        For fluid components, only :math:`\delta` (the energy
                        density) will be affected. For particle components,
                        both the positions and velocities are affected, with
                        the non-Gaussian velocity contribution obtained from
                        :math:`f_{\text{NL}}\delta^2(\boldsymbol{x})` using
                        :math:`aH(a)f(a)` as a conversion factor, similar to
                        how the velocities are obtained when using
                        back-scaling (see above).

                      * ``'structure'``: Specifies the underlying 3D structure
                        to use for a realisation. This can be either the
                        primordial noise (``'primordial'``)
                        :math:`\zeta(k)\mathcal{R}(\boldsymbol{k})`
                        (:ref:`random noise <random_seeds>`
                        :math:`\mathcal{R}(\boldsymbol{k})` with
                        :ref:`amplitude <primordial_spectrum>`
                        :math:`\zeta(k)`) or the structure extracted from the
                        non-linearly evolved energy density of the component
                        (``'non-linear'``), i.e.
                        :math:`\delta(a, \boldsymbol{k})/T_\delta(a, k)`.
                        Realisation happening at the initial time of the
                        simulation (i.e. initial condition generation) will
                        always make use of the primordial noise, though at
                        this time the two options are mathematically
                        equivalent. As late-time realisation is not possible
                        for particle components, this realisation option only
                        affects fluid components.

                      * ``'compound'``: When performing late-time realisations
                        for fluid components using the non-linearly evolved
                        energy density as the source of structure, some fluid
                        variables can be defined in several ways. Consider
                        the (conserved) shear stress fluid variable
                        :math:`\varsigma^i_j(a, \boldsymbol{x})`,
                        realised through

                        .. math::

                           \varsigma^i_j(a, \boldsymbol{x}) = \bigl(\varrho + c^{-2}\mathcal{P}\bigr)\sigma^i_j(a, \boldsymbol{x})\,,

                        with :math:`\sigma^i_j(a, \boldsymbol{x})` itself
                        realised from the scalar anisotropic stress. With the
                        ``'compound'`` realisation option set to ``'linear'``,
                        the (conserved) energy density and pressure in the
                        parenthesis above will be evaluated at the background
                        level,
                        :math:`\varrho + c^{-2}\mathcal{P} \rightarrow (1 + w)\bar{\varrho}(a)`.
                        With the ``'compound'`` realisation option set to
                        ``'non-linear'``, the full non-linearly evolved fluid
                        grids are used,
                        :math:`\varrho + c^{-2}\mathcal{P} \rightarrow \varrho(a, \boldsymbol{x}) + c^{-2}\mathcal{P}(a, \boldsymbol{x})`.
                        Using the non-linear option injects additional
                        non-linearity into the realised fluid variable,
                        which can be desirable. See the paper on
                        ':ref:`νCO𝘕CEPT: Cosmological neutrino simulations from the non-linear Boltzmann hierarchy <nuconcept_cosmological_neutrino_simulations_from_the_nonlinear_boltzmann_hierarchy>`'
                        for further details.

-- --------------- -- -
\  **Example 0**   \  Do all realisations in synchronous gauge. Also, use
                      back-scaling when realising components with a
                      name/species of ``'matter``:

                      .. code-block:: python3

                         realization_options = {
                             'gauge': {
                                 'all': 'synchronous',
                             },
                             'back-scaling': {
                                 'matter': True,
                             },
                         }

                      .. note::

                         When using back-scaling, the :math:`\delta` transfer
                         function is used not only for the particle positions
                         but also for the velocities. This is crucial when
                         using the synchronous gauge, as the velocity transfer
                         function in this gauge is not well suited for
                         :math:`N`-body initial conditions.

-- --------------- -- -
\  **Example 1**   \  Make use of second-order Lagrangian perturbation
                      theory (2LPT) when realising particle components:

                      .. code-block:: python3

                         realization_options = {
                             'LPT': {
                                 'particles': 2,
                             },
                         }

                      As the ``'LPT'`` specification only applies to particle
                      components anyway, we can also do

                      .. code-block:: python3

                         realization_options = {
                             'LPT': {
                                 'all': 2,
                             },
                         }

                      which can be more succinctly expressed as

                      .. code-block:: python3

                         realization_options = {
                             'LPT': 2,
                         }

-- --------------- -- -
\  **Example 2**   \  Always perform realisations using the primordial noise
                      as the underlying source of structure, regardless of the
                      time of realisation:

                      .. code-block:: python3

                         realization_options = {
                             'structure': {
                                 'all': 'primordial',
                             },
                         }

== =============== == =



------------------------------------------------------------------------------



``select_lives``
................
== =============== == =
\  **Description** \  Specifies life spans of components
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'default': (0, ထ),
                         }

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      ``dict``, specifying the *life span* of individual
                      components. Here, a life span is a 2-tuple in the format

                      .. code-block:: python3

                         (a_activate, a_terminate)

                      specifying the values of the scale factor :math:`a` at
                      which components should be *activated* and *terminated*,
                      respectively. Prior to its activation time, a given
                      component will not take part in the simulation. At the
                      activation time, it will then be realised and included
                      in the simulation. At termination time, the component
                      will be removed from the simulation, and it cannot be
                      reintroduced. The default specifications of this
                      parameter ensures that --- by default --- all components
                      are active from the beginning to the end of the
                      simulation.

-- --------------- -- -
\  **Example 0**   \  Let the component with the name ``'linear neutrino'``
                      participate actively in the simulation until
                      :math:`a = 0.1`, at which point the component with the
                      name ``'non-linear neutrino'`` should take its place:

                      .. code-block:: python3

                         select_lives = {
                                 'linear neutrino': (0,   0.1),
                             'non-linear neutrino': (0.1,   ထ),
                         }

                      Assuming the two components both represent the
                      ``'neutrino'`` species, but makes use of linear and
                      non-linear evolution, this then effectively sets
                      :math:`a = 0.1` to be the time at which the neutrino
                      species should be treated non-linearly. This can spare
                      the simulation from a lot of unnecessary computation.
                      See the :ref:`tutorial <nonlinear_massive_neutrinos>`
                      for a similar example use case.
== =============== == =



------------------------------------------------------------------------------



.. _softening_kernel:

``softening_kernel``
....................
== =============== == =
\  **Description** \  The kernel to use for particle softening
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         'spline'

-- --------------- -- -
\  **Elaboration** \  For direct particle-particle forces (PP gravity,
                      short-range P³M gravity), the force needs to be
                      *softened* at short distances in order to avoid
                      numerical problems with insufficient time resolution for
                      two-body encounters. See the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'
                      for details. This parameter sets the type of softening
                      to use for all such forces and components. All softening
                      kernels may be described as a transformation on
                      :math:`|\boldsymbol{x}|^{-3}` in the force

                      .. math::

                         \boldsymbol{f} \propto |\boldsymbol{x}|^{-3}\boldsymbol{x}\, .

                      The implemented softening kernels are:

                      * ``'none'``: Do not use any softening;

                        .. math::

                           |\boldsymbol{x}|^{-3} \rightarrow |\boldsymbol{x}|^{-3}\, .

                      * ``'plummer'``: Make use of the Plummer softening
                        kernel;

                        .. math::

                           |\boldsymbol{x}|^{-3} \rightarrow \bigl(|\boldsymbol{x}|^2 + \epsilon^2\bigr)^{-\frac{3}{2}}\, ,

                        with :math:`\epsilon` the
                        :ref:`softening length <select_softening_length>`.

                      * ``'spline'``: Make use of the cubic spline softening
                        kernel, also used by e.g. GADGET;

                        .. math::

                           |\boldsymbol{x}|^{-3} \rightarrow \begin{cases}
                               \displaystyle \frac{32}{\epsilon_{\text{B}}^3}\biggl( x_{\text{B}}^3 - \frac{6}{5}x_{\text{B}}^2 + \frac{1}{3} \biggr) & \displaystyle x_{\text{B}} < \frac{1}{2} \\
                               \displaystyle \frac{32}{\epsilon_{\text{B}}^3}\biggl( -\frac{1}{3}x_{\text{B}}^3 + \frac{6}{5}x_{\text{B}}^2 - \frac{3}{2}x_{\text{B}} + \frac{2}{3} - \frac{1}{480}x_{\text{B}}^{-3} \biggr) & \displaystyle \frac{1}{2} \leq x_{\text{B}} < 1 \\
                               \displaystyle |\boldsymbol{x}|^{-3} & \displaystyle 1 \leq x_{\text{B}}\, ,
                           \end{cases}

                        with
                        :math:`x_{\text{B}} \equiv |\boldsymbol{x}|/\epsilon_{\text{B}}`
                        and :math:`\epsilon_{\text{B}} = 2.8\epsilon`, with
                        :math:`\epsilon` being the specified
                        "Plummer-equivalent"
                        :ref:`softening length <select_softening_length>`.

-- --------------- -- -
\  **Example 0**   \  Use simple Plummer softening:

                      .. code-block:: python3

                         softening_kernel = 'plummer'

== =============== == =



------------------------------------------------------------------------------



.. _select_softening_length:

``select_softening_length``
...........................
== =============== == =
\  **Description** \  Specifies particle softening lengths
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'default': '0.025*boxsize/cbrt(N)',
                         }

-- --------------- -- -
\  **Elaboration** \  Direct particle-particle forces are softened according
                      to the chosen
                      :ref:`softening kernel <softening_kernel>`. Regardless
                      of the chosen kernel, the associated length scale is
                      expressed in terms of the "Plummer-equivalent" softening
                      length :math:`\epsilon`. This parameter is a
                      :ref:`component selection <components_and_selections>`
                      ``dict`` specifying this :math:`\epsilon` for the
                      particle components within the simulation (fluid
                      components have no need for softening).

                      The default value sets :math:`\epsilon` to be
                      :math:`2.5\,\%` of the mean inter-particle distance,
                      within each particle component separately. Here ``'N'``
                      is dynamically substituted for the number of particles
                      :math:`N` within the given component.

                      .. note::
                         The specified softening lengths are always comoving

-- --------------- -- -
\  **Example 0**   \  Use a softening length of :math:`10\,\mathrm{kpc}/h` for
                      all components:

                      .. code-block:: python3

                         select_softening_length = {
                             'all': 10*kpc/h,
                         }

-- --------------- -- -
\  **Example 1**   \  Use a softening length of :math:`1/10\,000` of the box
                      size for the component with a name/species of
                      ``'cold dark matter'`` and a softening length of
                      :math:`2\,\%` of the mean inter-particle distance
                      (between particles of each given component) for all other
                      components:

                      .. code-block:: python3

                         select_softening_length = {
                             'cold dark matter': boxsize/10_000,
                             'all'             : '0.02*boxsize/cbrt(N)',
                         }

                      .. note::
                         Parameters like ``boxsize`` may be used either
                         directly or referred to inside ``str``\ s.
                         Component-specific attributes like ``N`` must be
                         specified within a ``str``.

== =============== == =

