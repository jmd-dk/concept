Simulation
----------
The 'simulation' parameter category contains parameters specifying the
accuracy of the time integration, as well as parameters having to do with the
random numbers used for the primordial noise.

.. contents::
   :local:
   :depth: 1



------------------------------------------------------------------------------



.. _Deltat_base_background_factor:

``Δt_base_background_factor``
.............................
== =============== == =
\  **Description** \  This scales all background time step limiters
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         1

-- --------------- -- -
\  **Elaboration** \  The global time step size :math:`\Delta t` within
                      CO\ *N*\ CEPT depends on various *limiters* as described
                      in the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'.
                      This parameter acts as a scaling factor on all
                      *background limiters*.
-- --------------- -- -
\  **Example 0**   \  Scale the background limiters by a factor :math:`1/2`,
                      typically cutting the time step size :math:`\Delta t`
                      roughly in half at early times compared to its normal
                      size:

                      .. code-block:: python3

                         Δt_base_background_factor = 0.5

== =============== == =



------------------------------------------------------------------------------



.. _Deltat_base_nonlinear_factor:

``Δt_base_nonlinear_factor``
............................
== =============== == =
\  **Description** \   This scales all non-linear time step limiters
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         1

-- --------------- -- -
\  **Elaboration** \  The global time step size :math:`\Delta t` within
                      CO\ *N*\ CEPT depends on various *limiters* as described
                      in the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'.
                      This parameter acts as a scaling factor on all
                      *non-linear limiters*.
-- --------------- -- -
\  **Example 0**   \  Scale the non-linear limiters by a factor :math:`1/3`,
                      typically reducing the time step size :math:`\Delta t`
                      to about a third of its normal value at late times:

                      .. code-block:: python3

                         Δt_base_nonlinear_factor = 1/3

-- --------------- -- -
\  **Example 1**   \  Effectively remove the non-linear limiters by scaling
                      them by :math:`\infty`, so that the global time step size
                      :math:`\Delta t` is controlled solely by the background
                      limiters:

                      .. code-block:: python3

                         Δt_base_nonlinear_factor = ထ

== =============== == =



------------------------------------------------------------------------------



``Δt_increase_max_factor``
..........................
== =============== == =
\  **Description** \  Specifies a limit to the allowed relative increase of
                      the global time step size
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         ထ  # no limit

-- --------------- -- -
\  **Elaboration** \  As described in the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`',
                      the global time step size :math:`\Delta t` is
                      periodically allowed to increase. This parameter sets
                      an upper limit for the allowed relative change of
                      :math:`\Delta t` each time it is increased.
-- --------------- -- -
\  **Example 0**   \  Allow a maximum relative increase to :math:`\Delta t` of
                      :math:`15\,\%`:

                      .. code-block:: python3

                         Δt_increase_max_factor = 1.15

== =============== == =



------------------------------------------------------------------------------



.. _Deltat_rung_factor:

``Δt_rung_factor``
..................
== =============== == =
\  **Description** \  This scales the rung time step size required for
                      particles
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         1

-- --------------- -- -
\  **Elaboration** \  As described in the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`',
                      adaptive particle time-stepping is achieved through
                      *rungs*; power-of-two subdivisions of the global time
                      step size, :math:`\Delta t/2^{\ell}`. The rung
                      :math:`\ell \in \mathbb{N}_0` on which a given particle
                      belongs depends on the (short-range) acceleration
                      :math:`\boldsymbol{a}` of the particle, as

                      .. math::

                         \frac{\Delta t}{2^{\ell}} \underset{\sim}{\propto} \frac{1}{\sqrt{|\boldsymbol{a}|}}\, ,

                      with the proportionality scaled on the right by this
                      parameter.

-- --------------- -- -
\  **Example 0**   \  Decrease the rung time step sizes by a factor of
                      :math:`2`, effectively pushing the particles up one
                      rung, resulting in particle time-stepping which is
                      "twice as fine" as usual:

                      .. code-block:: python3

                         Δt_rung_factor = 0.5

== =============== == =



------------------------------------------------------------------------------



.. _Deltaa_max_early:

``Δa_max_early``
................
== =============== == =
\  **Description** \  Specifies the maximum allowed change to the scale factor
                      at early times
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         0.00153

-- --------------- -- -
\  **Elaboration** \  As described in the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`',
                      one of the time step limiters include the condition
                      that --- at early times --- the global time step size
                      :math:`\Delta t` must not be so large as to correspond
                      to an absolute change in scale factor :math:`\Delta a`
                      larger than some value. This parameter sets this value.
-- --------------- -- -
\  **Example 0**   \  Allow only a much smaller change :math:`\Delta a` at
                      early times, typically leading to finer time-stepping at
                      such times:

                      .. code-block:: python3

                         Δa_max_early = 0.0008

-- --------------- -- -
\  **Example 1**   \  Effectively disregard this limiter completely by setting
                      it equal to its
                      :ref:`late-time version <Deltaa_max_late>`:

                      .. code-block:: python3

                         Δa_max_early = Δa_max_late

                      .. note::
                         For this to work, you must explicitly set a value for
                         the ``Δa_max_late``
                         :ref:`parameter <Deltaa_max_late>`.

                      .. note::
                         As the instantaneous Hubble time :math:`H^{-1}` is
                         part of the same limiter as the early-time
                         :math:`\Delta a` (again, see the paper on
                         ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'),
                         this also removes any explicit limitation of the
                         global time step size due to :math:`H^{-1}`.

== =============== == =



------------------------------------------------------------------------------



.. _Deltaa_max_late:

``Δa_max_late``
...............
== =============== == =
\  **Description** \  Specifies the maximum allowed change to the scale factor
                      at late times
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         0.022

-- --------------- -- -
\  **Elaboration** \  As described in the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`',
                      one of the time step limiters include the condition
                      that --- at late times --- the global time step size
                      :math:`\Delta t` must not be so large as to correspond
                      to an absolute change in scale factor :math:`\Delta a`
                      larger than some value. This parameter sets this value.
-- --------------- -- -
\  **Example 0**   \  Significantly decrease the allowed :math:`\Delta a` at
                      late times, leading to finer late-time time-stepping
                      for simulations with low resolution (in simulations of
                      high resolution, non-linear limiters probably dominate
                      at late times regardless of this):

                      .. code-block:: python3

                         Δa_max_late = 0.01

-- --------------- -- -
\  **Example 1**   \  Effectively disregard this limiter by setting it equal
                      to :math:`\infty`:

                      .. code-block:: python3

                         Δa_max_late = ထ

== =============== == =



------------------------------------------------------------------------------



``N_rungs``
...........
== =============== == =
\  **Description** \  Specifies the number of available rungs
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         8

-- --------------- -- -
\  **Elaboration** \  As described in the paper on
                      ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`',
                      adaptive particle time-stepping is achieved by placing
                      the particles on *rungs* :math:`\ell \in \mathbb{N}_0`
                      according to their acceleration. This parameter
                      specifies the maximum number :math:`n_{\text{rung}}` of
                      these rungs, with :math:`0 \leq \ell < n_{\text{rung}}`.
                      Particles with accelerations large enough for them to
                      belong to rung :math:`\ell \geq n_{\text{rung}}` (see
                      the ``Δt_rung_factor``
                      :ref:`parameter <Deltat_rung_factor>`) will remain on
                      rung :math:`\ell = n_{\text{rung}} - 1`.
-- --------------- -- -
\  **Example 0**   \  Increase the number of available rungs to :math:`10`:

                      .. code-block:: python3

                         N_rungs = 10

                      .. note::
                         As demonstrated in
                         ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`',
                         the time step limiters (see the
                         ``Δt_base_background_factor``
                         :ref:`parameter <Deltat_base_background_factor>`, the
                         ``Δt_base_nonlinear_factor``
                         :ref:`parameter <Deltat_base_nonlinear_factor>`, the
                         ``Δa_max_early``
                         :ref:`parameter <Deltaa_max_early>`, the
                         ``Δa_max_late`` :ref:`parameter <Deltaa_max_late>`)
                         serve to keep the highest occupied rung rather low,
                         and so typically we do not gain anything from
                         increasing ``N_rungs``. It does however come with a
                         hefty performance penalty if ``N_rungs`` is increased
                         significantly.

== =============== == =



------------------------------------------------------------------------------



.. _fftw_wisdom_rigor:

``fftw_wisdom_rigor``
.....................
== =============== == =
\  **Description** \  Specifies the rigour level to use when gathering FFTW
                      wisdom
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         'measure'

-- --------------- -- -
\  **Elaboration** \  The FFTW library used by CO\ *N*\ CEPT to compute
                      distributed FFTs knows of a multitude of different
                      methods for computing a given FFT. Which method is the
                      fastest is not obvious, and so FFTW
                      `plan <https://www.fftw.org/fftw3_doc/Planner-Flags.html>`_
                      the FFT ahead of time, deciding on some strategy.
                      The knowledge gathered by this planning step is called
                      *wisdom*. This parameter sets the rigour level to use
                      when gathering said wisdom, with a higher level taking
                      longer to gather, but hopefully also resulting in a
                      superior plan, saving time in the long run. The
                      available rigour levels --- from lowest to highest ---
                      are:

                      * ``'estimate'``
                      * ``'measure'``
                      * ``'patient'``
                      * ``'exhaustive'``

                      .. caution::
                         Only wisdom gathering with the lowest-level rigour
                         ``'estimate'`` will result in deterministic plans.
                         By reusing the wisdom between simulations (see the
                         ``fftw_wisdom_reuse``
                         :ref:`parameter <fftw_wisdom_reuse>` and the
                         ``fftw_wisdom_share`` :ref:`parameter
                         <fftw_wisdom_share>`) you can still reliably have
                         deterministic simulations.

-- --------------- -- -
\  **Example 0**   \  Run with the highest possible level of FFTW wisdom
                      rigour, producing optimal FFTW plans at the cost of
                      slow planning:

                      .. code-block:: python3

                         fftw_wisdom_rigor = 'exhaustive'

== =============== == =



------------------------------------------------------------------------------



.. _fftw_wisdom_reuse:

``fftw_wisdom_reuse``
.....................
== =============== == =
\  **Description** \  Specifies whether to reuse FFTW wisdom between
                      simulations
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True

-- --------------- -- -
\  **Elaboration** \  The gathered FFTW wisdom (see the ``fftw_wisdom_rigor``
                      :ref:`parameter <fftw_wisdom_rigor>`) can be reused
                      between simulations by
                      `saving it to disk <https://www.fftw.org/fftw3_doc/Words-of-Wisdom_002dSaving-Plans.html>`_
                      (specifically to the ``.reusable/fftw`` directory). When
                      ``fftw_wisdom_reuse`` is ``True``, any gathered wisdom
                      will be saved. Any already existing wisdom will be read
                      and used, providing it matches the given problem (grid
                      size, number of processes) and rigour level. See the
                      ``fftw_wisdom_share``
                      :ref:`parameter <fftw_wisdom_share>` for further
                      restrictions on reused wisdom.

                      .. caution::
                         Depending on the
                         :ref:`rigour level <fftw_wisdom_rigor>`, the
                         gathering of wisdom may not be a deterministic
                         process. When not reusing wisdom between simulations,
                         they may then end up with different FFTW plans for
                         the same problem, introducing indeterminism.

-- --------------- -- -
\  **Example 0**   \  Always gather FFTW wisdom anew, disregarding available
                      stored wisdom:

                      .. code-block:: python3

                         fftw_wisdom_reuse = False

                      The gathered wisdom will similarly not be stored on the
                      disk.
== =============== == =



------------------------------------------------------------------------------



.. _fftw_wisdom_share:

``fftw_wisdom_share``
.....................
== =============== == =
\  **Description** \  Specifies whether to reuse FFTW wisdom between
                      simulations running on a different set of nodes
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         False

-- --------------- -- -
\  **Elaboration** \  The gathered FFTW wisdom (see the ``fftw_wisdom_rigor``
                      :ref:`parameter <fftw_wisdom_rigor>`) can be reused
                      between simulations (see the ``fftw_wisdom_reuse``
                      :ref:`parameter <fftw_wisdom_reuse>`) by caching it to
                      disk. Even for the same problem (grid size, number of
                      processes) the optimal plan may be highly dependent on
                      the process configuration, i.e. how the MPI processes
                      are mapped to physical CPU cores on the nodes. For this
                      reason, it may not be ideal to reuse wisdom, unless we
                      know that it was generated using a similar process
                      configuration to what is used this time around, i.e.
                      with processes belonging to the same nodes as currently.
                      This parameter specifies whether wisdom should be reused
                      regardless of mismatches between process configurations.

                      .. caution::
                         Not sharing FFTW wisdom (the default) comes at the
                         price of indeterminism, as different simulations
                         might now end up using different plans for the same
                         problem.

-- --------------- -- -
\  **Example 0**   \  Allow sharing of FFTW wisdom between simulations running
                      on different nodes, guaranteeing deterministic behaviour
                      regardless of how the processes of the MPI job gets
                      distributed across the compute nodes of a cluster:

                      .. code-block:: python3

                         fftw_wisdom_share = True

                      .. note::
                         This does not have any effect if the FFTW wisdom is
                         not :ref:`shared <fftw_wisdom_share>` in the first
                         place.

== =============== == =



------------------------------------------------------------------------------



.. _random_seed:

``random_seed``
...............
== =============== == =
\  **Description** \  Number with which to seed the pseudo-random number
                      generator, used for generating noise for initial
                      conditions
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         0

-- --------------- -- -
\  **Elaboration** \  The initial, linear density field (from which e.g. particle
                      positions are constructed) is given as

                      .. math::

                         \delta(a_{\text{begin}}, \boldsymbol{x}) \propto \underset{\hspace{-0.5em}\boldsymbol{x}\leftarrow\boldsymbol{k}}{\mathcal{F}^{-1}}\bigl[\mathcal{R}(\boldsymbol{k}) \zeta(|\boldsymbol{k}|) T_\delta(a_{\text{begin}}, |\boldsymbol{k}|)\bigr]\, ,

                      with :math:`\zeta(k)` and :math:`T_{\delta}(a, k)` the
                      primordial curvature perturbation and :math:`\delta`
                      (density) transfer function (see the ``class_params``
                      :ref:`parameter <class_params>`), respectively, and
                      :math:`\underset{\hspace{-0.5em}\boldsymbol{x}\leftarrow\boldsymbol{k}}{\mathcal{F}^{-1}}`
                      being the (inverse) Fourier transform from Fourier
                      :math:`\boldsymbol{k}` space to real
                      :math:`\boldsymbol{x}` space. Finally,
                      :math:`\mathcal{R}(\boldsymbol{k})` is the primordial
                      noise, which is a field of uncorrelated random complex
                      numbers drawn from a Gaussian (normal) distribution with
                      mean 0 and standard deviation :math:`1`
                      (i.e. :math:`1/\sqrt{2}` for the real and imaginary
                      components separately). The sequence of pseudo-random
                      numbers with which to populate
                      :math:`\mathcal{R}(\boldsymbol{k})` is generated by a
                      pseudo-random number generator, seeded with a
                      non-negative integer value of ``random_seed``. To change
                      the underlying primordial random noise, change the value
                      for ``random_seed``. For running multiple simulations
                      with the same noise, use the same ``random_seed``.

                      .. tip::
                         The random seed is common for all processes in the
                         simulation, implemented in such a way as to ensure
                         the same random noise regardless of the number of
                         processes used to run the simulation and thus for
                         generating the initial conditions.

                      .. tip::
                         When running successive simulations with different
                         resolution (e.g. number of particles) but fixed box
                         size, the random noise will be identical for the
                         :math:`k` modes shared between the simulations, with
                         new values used only for the higher :math:`k` not
                         available in the lower-resolution simulations. This
                         allows for e.g. convergence tests without fear of
                         changes to the random realisation.

-- --------------- -- -
\  **Example 0**   \  Use some particular random sequence for the initial
                      conditions:

                      .. code-block:: python3

                         random_seed = 42

-- --------------- -- -
\  **Example 1**   \  Have each successive simulation use a unique
                      random realisation of the primordial noise:

                      .. code-block:: python3

                         random_seed = jobid

== =============== == =



------------------------------------------------------------------------------



.. _primordial_amplitude_fixed:

``primordial_amplitude_fixed``
..............................
== =============== == =
\  **Description** \  Specifies whether to use fixed primordial amplitudes
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         False

-- --------------- -- -
\  **Elaboration** \  See the ``random_seed`` :ref:`parameter <random_seed>`
                      for how the primordial complex Gaussian random noise is
                      normally defined. For each mode :math:`\boldsymbol{k}`,
                      instead of drawing the real and imaginary components of
                      :math:`\mathcal{R}(\boldsymbol{k})` from a Gaussian
                      distribution separately, we may fix the amplitude of the
                      complex number to its mean value (:math:`1`), and draw
                      a random complex phase from a uniform distribution.
                      While not physically sound, statistic such as the
                      power spectrum become much smoother by doing this, as
                      the discretised Fourier modes now all have amplitudes
                      corresponding to the average amplitude taken
                      over an infinite ensemble. See
                      `this paper <https://academic.oup.com/mnrasl/article/462/1/L1/2589516>`_
                      for more information about such '*fixed* simulations'.

                      .. tip::
                         The random complex numbers generated when using fixed
                         amplitudes will have the same phases as when not using
                         fixed amplitudes. Though not equal, the underlying
                         random noise of two sets of initial conditions,
                         one generated with fixed amplitudes and the other
                         without, will then be similar (provided they use the
                         same value for the ``random_seed`` :ref:`parameter
                         <random_seed>`).

-- --------------- -- -
\  **Example 0**   \  Use fixed primordial amplitudes, leading to smoother
                      power spectra:

                      .. code-block:: python3

                         primordial_amplitude_fixed = True

== =============== == =



------------------------------------------------------------------------------



.. _primordial_phase_shift:

``primordial_phase_shift``
..........................
== =============== == =
\  **Description** \  Specifies a constant offset to use for the primordial
                      phases
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         0

-- --------------- -- -
\  **Elaboration** \  See the ``random_seed`` :ref:`parameter <random_seed>`
                      for how the primordial complex Gaussian random noise is
                      normally defined. The phase of each random complex
                      number will be shifted by the value of this parameter.

                      .. note::
                         This is typically used together with
                         :ref:`fixed amplitudes <primordial_amplitude_fixed>`,
                         but it does not have to be.

-- --------------- -- -
\  **Example 0**   \  Use completely out-of-phase primordial noise compared to
                      the default:

                      .. code-block:: python3

                         primordial_phase_shift = π

                      By taking the average of e.g. power spectra from this
                      simulation and a simulation with
                      ``primordial_phase_shift = 0``, much of the statistical
                      noise coming about due to the finite simulation size
                      will vanish. This is typically used together with
                      :ref:`fixed amplitudes <primordial_amplitude_fixed>`.
                      See
                      `this paper <https://academic.oup.com/mnrasl/article/462/1/L1/2589516>`_
                      for more information about such '*paired*-and-*fixed*
                      simulations'.
== =============== == =



------------------------------------------------------------------------------



.. _cell_centered:

``cell_centered``
.................
== =============== == =
\  **Description** \  Specifies whether to use cell-centred as opposed to
                      cell-vertex grid discretisation
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True

-- --------------- -- -
\  **Elaboration** \  The geometry of a CO\ *N*\ CEPT simulation is that of a
                      fixed (in comoving coordinates), periodic, cubic box of
                      size length :math:`L_{\mathrm{box}}` (corresponding to
                      the ``boxsize`` :ref:`parameter <boxsize>`), with one
                      corner at :math:`(0, 0, 0)` and another at
                      :math:`(L_{\mathrm{box}}, L_{\mathrm{box}}, L_{\mathrm{box}})`.
                      This allows us to talk about e.g. the absolute position
                      of individual particles. Various global, Cartesian,
                      periodic grids --- such as
                      :ref:`potential grids <potential_options>` --- need to be
                      embedded within this box. We do this in the obvious way,
                      with the corner of one of the grid cells at the box
                      origin :math:`(0, 0, 0)`, and with cell faces aligned
                      with the faces of the box. With each extended cell
                      storing a single value, corresponding to a discrete
                      tabulation of some underlying continuous field, we must
                      now decide on exactly where within the cell volume this
                      single value supposedly matches that of the underlying
                      field. We refer to this choice as the grid
                      discretisation strategy. Typical choices for this
                      strategy are those of **cell-centred** or
                      **cell-vertex** discretisation, corresponding to
                      choosing the cell centre or (lower) corner as the
                      point of tabulation. If we think of the cell value as
                      an average of the underlying field over a cell volume,
                      the cell-centred strategy simply corresponds to
                      averaging over the cell itself (for a sufficiently
                      smooth underlying field), while the cell-vertex
                      strategy corresponds to averaging over a cell-shaped
                      volume centred at the (lower) cell corner.

                      In effect, the two choices of grid discretisation come
                      down to whether they tabulate the underlying field at
                      :math:`(L_{\text{c}}/2, 3L_{\text{c}}/2, 5L_{\text{c}}/2, \dots)`
                      (cell-centred) or
                      :math:`(0, L_{\text{c}}, 2L_{\text{c}}, \dots)`
                      (cell-vertex), with :math:`L_{\text{c}}` the cell width
                      and the tabulations applying for each of the :math:`3`
                      dimensions separately. Ideally the discretisation
                      strategy should not alter the simulation results
                      noticeably, but for low-resolution simulations
                      :doc:`this is not so </tutorial/gadget>`.

-- --------------- -- -
\  **Example 0**   \  Swap out the default cell-centred grid discretisation
                      strategy for cell-vertex, as used by e.g. GADGET:

                      .. code-block:: python3

                         cell_centered = False

== =============== == =



------------------------------------------------------------------------------



``class_reuse``
...............
== =============== == =
\  **Description** \  Specifies whether to reuse results of CLASS computations
                      between simulations
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True

-- --------------- -- -
\  **Elaboration** \   CO\ *N*\ CEPT delegates the computation of background
                       quantities (unless the ``enable_class_background``
                       :ref:`parameter <enable_class_background>` is set to
                       ``False``) as well as linear perturbations to the
                       `CLASS <http://class-code.net/>`_ code. Though
                       typically inexpensive, running precision simulations
                       with e.g.
                       :ref:`non-linear massive neutrinos <nonlinear_massive_neutrinos>`
                       tend to increase the CLASS computation time
                       substantially. All CLASS results are always cached to
                       disk (specifically to the ``.reusable/class``
                       directory). If a CLASS computation is about to be run
                       for which the results are already cached, these will be
                       reused if this parameter is ``True``.
-- --------------- -- -
\  **Example 0**   \  Do not make use any pre-existing CLASS results:

                      .. code-block:: python3

                         class_reuse = False

                      .. note::
                         Even when not making use of cached CLASS results,
                         new CLASS runs will still be cached to disk

== =============== == =

