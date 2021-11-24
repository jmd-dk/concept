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
                      ":doc:`The cosmological simulation code CO𝘕CEPT 1.0</publications>`".
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
                      `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`_)
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
                         `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`_
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
                      ":doc:`The cosmological simulation code CO𝘕CEPT 1.0</publications>`"
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

                           |\boldsymbol{x}|^{-3} \rightarrow \bigl(|\boldsymbol{x}|^2 + \epsilon^2\bigr)^{-3/2}\, ,

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
                             'particles': '0.03*boxsize/cbrt(N)',
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
                      :math:`3\,\%` of the mean inter-particle distance,
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

