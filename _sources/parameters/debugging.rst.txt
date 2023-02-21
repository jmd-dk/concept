Debugging
---------
The 'debugging' parameter category contains parameters for toggling standard
features as well as additional information on or off.

.. contents::
   :local:
   :depth: 1



------------------------------------------------------------------------------



``print_load_imbalance``
........................
== =============== == =
\  **Description** \  Controls how the work load imbalance should be
                      displayed, if at all
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True

-- --------------- -- -
\  **Elaboration** \  As described in the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`',
                      each CPU core / MPI rank are responsible for a
                      designated region of the box, called its *domain*. Over
                      time, inhomogeneities naturally lead to some domains
                      requiring a greater computational work load than others
                      in order to evolve its contents through a time step.
                      This load imbalance may be reported after each time
                      step. By default, only the worst (largest) load
                      imbalance between all CPU cores is shown.
-- --------------- -- -
\  **Example 0**   \  Do not print out the load imbalance:

                      .. code-block:: python3

                         print_load_imbalance = False

-- --------------- -- -
\  **Example 1**   \  Print out load imbalance for all CPU cores:

                      .. code-block:: python3

                         print_load_imbalance = 'full'

== =============== == =



------------------------------------------------------------------------------



.. _particle_reordering:

``particle_reordering``
.......................
== =============== == =
\  **Description** \  Specifies whether particles within a component are
                      allowed to be periodically reordered in memory
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True

-- --------------- -- -
\  **Elaboration** \  After a reordering, the particle data in memory is
                      sorted so that it mimics the particle visiting order
                      when traversing the tiles and subtiles of the
                      short-range force. This improves CPU caching and can
                      lead to a substantial speed-ups.
-- --------------- -- -
\  **Example 0**   \  Disable periodic in-memory reordering of particles:

                      .. code-block:: python3

                         particle_reordering = False

-- --------------- -- -
\  **Example 1**   \  In case of several tilings, the reordering will be done
                      with respect to the tiling belong to the short-range
                      force with the largest computation time. Though
                      preferable for performance, this introduces a slight
                      indeterminacy which you may not want. To enable particle
                      reordering but always done with respect to the same
                      short-range force, use

                      .. code-block:: python3

                         particle_reordering = 'deterministic'

                      .. note::
                         As CO\ *N*\ CEPT currently only has a single
                         short-range force implemented (``'gravity'``), this
                         is --- for the moment --- equivalent to the default

                         .. code-block:: python3

                            particle_reordering = True

== =============== == =



------------------------------------------------------------------------------



.. _enable_Hubble:

``enable_Hubble``
.................
== =============== == =
\  **Description** \  Enable the universal Hubble expansion of the background
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True

-- --------------- -- -
\  **Elaboration** \  With ``enable_Hubble = True`` (the default), the Hubble
                      expansion is turned on, so that the scale factor
                      :math:`a(t)` evolves according to the specified
                      cosmology. For standard cosmological simulations,
                      this is what we want. By default, CLASS is used to
                      provide CO\ *N*\ CEPT with :math:`a(t)`, though this
                      can be :ref:`disabled <enable_class_background>`.
-- --------------- -- -
\  **Example 0**   \  Turn off the Hubble expansion of:

                      .. code-block:: python3

                         enable_Hubble = False

                      Note that this is really only intended for use with
                      internal testing. A lot basic functionality will not
                      work in this mode.
== =============== == =



------------------------------------------------------------------------------



.. _enable_class_background:

``enable_class_background``
...........................
== =============== == =
\  **Description** \  Specifies whether to use CLASS to solve the cosmological
                      background evolution
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True

-- --------------- -- -
\  **Elaboration** \  The `CLASS <http://class-code.net/>`__ code is fully
                      integrated within CO\ *N*\ CEPT and provides the
                      cosmological background (e.g. :math:`a(t)`) as well as
                      linear perturbations, the latter used for e.g.
                      generation of initial conditions. The background
                      computation within CLASS takes all species into account.
                      Even in the most basic case of not specifying any
                      :ref:`CLASS parameters <class_params>`, this includes
                      radiation (photons and massless neutrinos), matter
                      (baryons and cold dark matter) and a cosmological
                      constant. Alternatively, CO\ *N*\ CEPT implements a
                      simplified background containing just matter and a
                      cosmological constant.
-- --------------- -- -
\  **Example 0**   \  Swap the full CLASS background with a simplified
                      background containing just matter and a cosmological
                      constant:

                      .. code-block:: python3

                         enable_class_background = False

                      This is useful when seeking compatibility with other
                      simulation codes which use this simplified background,
                      but note that some features (e.g. initial condition
                      generation) require the proper CLASS background.
== =============== == =

