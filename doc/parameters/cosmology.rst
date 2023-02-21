Cosmology
---------
The 'cosmology' parameter category contains parameters specifying the
cosmology, as well as the starting time for the simulation.

.. contents::
   :local:
   :depth: 1



------------------------------------------------------------------------------



.. _H0:

``H0``
......
== =============== == =
\  **Description** \  The Hubble constant :math:`H_0`, i.e. the value of the
                      Hubble parameter at the present time :math:`a = 1`
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         67*km/(s*Mpc)

-- --------------- -- -
\  **Elaboration** \  Setting ``H0`` also defines the *inferred* parameter
                      ``h`` as ``h = H0/(100*km/(s*Mpc))``. With e.g.

                      .. code-block:: python3

                         H0 = 67*km/(s*Mpc)

                      we then have ``h = 0.67``. This ``h`` is available for
                      other parameters to use, see e.g. the ``boxsize``
                      :ref:`parameter <boxsize>` for an example.

                      .. caution::

                         The value of ``h`` will not be properly set unless
                         you explicitly define ``H0``.

-- --------------- -- -
\  **Example 0**   \  Use a value of
                      :math:`H_0 = 74\,\text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1}`
                      for the Hubble constant:

                      .. code-block:: python3

                         H0 = 74*km/(s*Mpc)

== =============== == =



------------------------------------------------------------------------------



.. _Omegab:

``Ωb``
......
== =============== == =
\  **Description** \  Density parameter for baryonic matter
                      :math:`\Omega_{\text{b}}` at the present
                      time :math:`a = 1`
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         0.049

-- --------------- -- -
\  **Elaboration** \  Within a simulation CO\ *N*\ CEPT treats baryonic matter
                      as though it were cold dark matter, i.e. it has no gas
                      physics. Typically, the species used for the particles in
                      a simulation is (total) 'matter', meaning combined
                      baryonic and cold dark matter. As the baryons and the
                      cold dark matter have different linear perturbations
                      (from CLASS), a distinction must still be made within
                      CO\ *N*\ CEPT as they have different initial conditions.
                      This is why we need both the ``Ωb`` parameter and the
                      ``Ωcdm`` :ref:`parameter <Omegacdm>`.
-- --------------- -- -
\  **Example 0**   \  Specify :math:`\Omega_{\text{b}} = 0.05`:

                      .. code-block:: python3

                         Ωb = 0.05

-- --------------- -- -
\  **Example 1**   \  Specify :math:`\Omega_{\text{b}}` via known value of
                      :math:`h^2\Omega_{\text{b}} \equiv \omega_{\text{b}} = 0.022`:

                      .. code-block:: python3

                         Ωb = 0.022/h**2

== =============== == =



------------------------------------------------------------------------------



.. _Omegacdm:

``Ωcdm``
........
== =============== == =
\  **Description** \  Density parameter for cold dark matter
                      :math:`\Omega_{\text{cdm}}` at the present
                      time :math:`a = 1`
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         0.27

-- --------------- -- -
\  **Elaboration** \  Typically, the species used for the particles in a
                      simulation is (total) 'matter', meaning combined
                      :ref:`baryonic <Omegab>` and cold dark matter.
-- --------------- -- -
\  **Example 0**   \  Specify :math:`\Omega_{\text{cdm}} = 0.28`:

                      .. code-block:: python3

                         Ωcdm = 0.28

-- --------------- -- -
\  **Example 1**   \  Specify :math:`\Omega_{\text{cdm}}` via known value of
                      :math:`h^2\Omega_{\text{cdm}} \equiv \omega_{\text{cdm}} = 0.12`:

                      .. code-block:: python3

                         Ωcdm = 0.12/h**2

-- --------------- -- -
\  **Example 2**   \  Specify :math:`\Omega_{\text{cdm}}` such that
                      :math:`\Omega_{\text{m}} \equiv \Omega_{\text{b}} + \Omega_{\text{cdm}} + \Omega_{\nu} = 0.30`:

                      .. code-block:: python3

                         Ωcdm = 0.30 - Ωb - Ων

                      Here ``Ων`` is the current total density parameter for
                      massive neutrinos, automatically inferred from the
                      ``class_params`` :ref:`parameter <class_params>`.
== =============== == =



------------------------------------------------------------------------------



.. _a_begin:

``a_begin``
...........
== =============== == =
\  **Description** \  Scale factor value :math:`a` at the start of the
                      simulation
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         1

                      You should not rely on this default, as the code detects
                      if you do not specify ``a_begin`` explicitly.
-- --------------- -- -
\  **Elaboration** \  If you
                      :ref:`start the simulation from a snapshot <initial_conditions>`
                      you should make sure to set ``a_begin`` to the value of
                      :math:`a` set in the snapshot.

                      Alternatively, you may :ref:`define <t_begin>`
                      ``t_begin``, the cosmic time :math:`t` at the start of
                      the simulation.
-- --------------- -- -
\  **Example 0**   \  Begin simulation at :math:`a = 0.01`:

                      .. code-block:: python3

                         a_begin = 0.01

== =============== == =



------------------------------------------------------------------------------



.. _t_begin:

``t_begin``
...........
== =============== == =
\  **Description** \  Cosmic time :math:`t` at the start of the simulation
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         0

                      If unset and running a *cosmological* simulation
                      (the :ref:`default <enable_Hubble>`), the initial
                      :math:`t` will really be set in accordance with the
                      :ref:`initial scale factor <a_begin>` and the cosmology.
-- --------------- -- -
\  **Elaboration** \  You may not set both ``t_begin`` and ``a_begin``.
-- --------------- -- -
\  **Example 0**   \  Begin simulation at time :math:`t = 10\,\text{Myr}`:

                      .. code-block:: python3

                         t_begin = 10*Myr

== =============== == =



------------------------------------------------------------------------------



.. _primordial_spectrum:

``primordial_spectrum``
.......................
== =============== == =
\  **Description** \  Parameters for specifying the primordial curvature
                      perturbation (power spectrum)
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'A_s'  : 2.1e-9,    # amplitude
                             'n_s'  : 0.96,      # tilt / spectral index
                             'α_s'  : 0,         # tilt running
                             'pivot': 0.05/Mpc,  # pivot scale
                         }

-- --------------- -- -
\  **Elaboration** \  Linear power spectra can be written as
                      :math:`P_{\text{lin}}(a, k) = \zeta^2(k) T^2(a, k)`,
                      with :math:`T(a, k)` some transfer function (obtained
                      from CLASS) and :math:`\zeta(k)` the primordial
                      curvature perturbation. CO\ *N*\ CEPT adopts the
                      parametrisation

                      .. math::

                         \require{upgreek}
                         \zeta(k) = \uppi \sqrt{2A_{\text{s}}} k^{-\frac{3}{2}} \biggl(\frac{k}{k_{\text{pivot}}}\biggr)^{\frac{n_{\text{s}} - 1}{2}} \exp\biggl[\frac{\alpha_{\text{s}}}{4} \ln\biggl(\frac{k}{k_{\text{pivot}}}\biggr)^2\biggr]\, ,

                      where :math:`A_{\text{s}}` (``'A_s'``) is the amplitude,
                      :math:`n_{\text{s}}` (``'n_s'``) is the 'tilt' or
                      'spectral index', :math:`\alpha_{\text{s}}` (``'α_s'``)
                      is the 'tilt running' and :math:`k_{\text{pivot}}`
                      (``'pivot'``) is the 'pivot scale'.

                      Note that though these exists as CLASS parameters, you
                      should never set any of these primordial parameters in
                      :ref:`class_params`. CLASS is only responsible for the
                      transfer functions :math:`T(a, k)`, with CO\ *N*\ CEPT
                      performing the scaling by :math:`\zeta(k)`.

-- --------------- -- -
\  **Example 0**   \  Enhance the primordial perturbations, leading to
                      stronger clustering:

                      .. code-block:: python3

                         primordial_spectrum = {
                             'A_s': 2.3e-9,
                         }

                      .. note::
                         Non-specified items in ``primordial_spectrum`` will
                         be set to their default values.

== =============== == =



------------------------------------------------------------------------------



.. _class_params:

``class_params``
................
== =============== == =
\  **Description** \  Parameters to pass on to the CLASS code
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {}  # default CLASS parameters except H0, Omega_b, Omega_cdm

-- --------------- -- -
\  **Elaboration** \  CO\ *N*\ CEPT relies on the
                      `CLASS <http://class-code.net/>`__ code for computing the
                      cosmological background (unless
                      :ref:`enable_class_background` is ``False``) as well as
                      linear perturbations. As such it is vital that CLASS is
                      run with the correct cosmological parameters. These
                      parameters are exactly the ones specified in
                      ``class_params``, with the addition of the ``H0``
                      :ref:`parameter <H0>`, the ``Ωb``
                      :ref:`parameter <Omegab>` and the ``Ωcdm``
                      :ref:`parameter <Omegacdm>`. That is, specifying
                      ``class_params = {}`` (the default) really amounts to

                      .. code-block:: python3

                         class_params = {
                             'H0'       : H0/(km/(s*Mpc)),
                             'Omega_b'  : Ωb,
                             'Omega_cdm': Ωcdm,
                         }

                      You thus never have to (and should not) include any of
                      these three parameters in ``class_params``.

                      Any CLASS parameters not specified in ``class_params``
                      (besides ``'H0'``, ``'Omega_b'`` and ``'Omega_cdm'``)
                      will take on default values as defined by CLASS.
                      To find out what they are, consult e.g. the
                      `explanatory.ini <https://github.com/lesgourg/class_public/blob/v2.7.2/explanatory.ini>`__
                      example CLASS parameter file.

                      .. note::
                         Note that the most basic CLASS cosmology obtained
                         from ``class_params = {}`` contains photons,
                         massless neutrinos, baryons, cold dark matter and a
                         cosmological constant. If you want to run simulations
                         with no radiation included in the background,
                         see the :ref:`enable_class_background` parameter.

-- --------------- -- -
\  **Example 0**   \  Use a cosmology with a slightly warmer CMB:

                      .. code-block:: python3

                         class_params = {
                             'T_cmb': 2.8,  # in K
                         }

                      .. note::
                         Unlike CO\ *N*\ CEPT, CLASS do assume its input
                         parameters to be specified using particular units.
                         As the ``class_params`` are fed more or less directly
                         to CLASS as is, these unit conventions must be
                         respected when adding items to ``class_params``.

-- --------------- -- -
\  **Example 1**   \  Use a cosmology with three massive neutrinos with a
                      total mass of :math:`\sum m_\nu = 0.1\, \text{eV}`:

                      .. code-block:: python3

                         class_params = {
                             'N_ur'    : 0,      # no massless neutrinos
                             'N_ncdm'  : 1,      # one massive neutrino
                             'deg_ncdm': 3,      # three times degenerate
                             'm_ncdm'  : 0.1/3,  # Σmν = 0.1 eV
                         }

                      .. note::

                         Using a cosmology with massive neutrinos also defines
                         the *inferred* parameter ``Ων``, the total density
                         parameter for all massive neutrino species
                         :math:`\Omega_{\nu}` at the present time
                         :math:`a = 1`. This may then be used to set other
                         parameters (see the ``Ωcdm``
                         :ref:`parameter <Omegacdm>`).

                      See the :ref:`tuturial <massive_neutrinos>` for a
                      walk-through of using a cosmology with massive
                      neutrinos.
-- --------------- -- -
\  **Example 2**   \  Use a cosmology with massive neutrinos of (possibly)
                      different masses:

                      .. code-block:: python3

                         # List of neutrino (ν) masses, possibly containing duplicates
                         _mν = [0*eV, 8.7e-3*eV, 5.0e-2*eV]  # any number of masses allowed

                         _N_eff = 3.046  # effective number of ν species
                         class_params = {
                             'N_ur'    : 0,              # no massless ν
                             'N_ncdm'  : len(set(_mν)),  # number of ν species with unique masses
                             'deg_ncdm': [               # number of ν species for each mass
                                 _mν.count(mν) for mν in sorted(set(_mν))
                             ],
                             'm_ncdm'  : [               # ν masses (avoid exact value of 0.0)
                                 max(mν/eV, 1e-100) for mν in sorted(set(_mν))
                             ],
                             'T_ncdm'  : [               # ν temperatures
                                 (4/11)**(1/3)*(_N_eff/len(_mν))**(1/4)
                             ]*len(set(_mν)),
                         }

                      In the above we additionally specify the neutrino
                      temperature (equal for all neutrino species),
                      parametrised via the effective number of neutrinos
                      :math:`N_{\text{eff}}`, as

                      .. math::

                         T_{\nu} = \biggl(\frac{4}{11}\biggr)^{\frac{1}{3}} \biggl(\frac{N_{\text{eff}}}{N_{\nu}}\biggr)^{\frac{1}{4}} T_{\gamma}\, ,

                      with :math:`N_{\nu}` the integer number of massive
                      neutrino species and :math:`T_{\gamma}` the photon
                      temperature ``T_cmb`` (not explicitly referenced in the
                      above ``class_params``).

                      .. note::
                         In the above parameter specification we use
                         ``list``\ s for multi-valued items within
                         ``class_params``. When using standard CLASS, such
                         values must be specified as single ``str``\ s of
                         comma-separated values. CO\ *N*\ CEPT takes care of
                         the required conversion.

                      .. note::
                         The above parameter specification is more involved
                         than it needs to be. What one gains from this is a
                         more efficient CLASS computation, as specifying
                         multiple neutrinos with equal mass in ``_mν`` here is
                         picked up and used to properly set the
                         degeneracy ``'deg_ncdm'``.

-- --------------- -- -
\  **Example 3**   \  By default, CLASS adjusts the density parameter
                      :math:`\Omega_{\Lambda}` of the cosmological constant
                      :math:`\Lambda` to ensure a flat universe, given the
                      specified amounts of matter, photons, neutrinos, etc.
                      If we instead wish to have *dynamic* dark energy, with
                      equation of state parametrised as
                      :math:`w(a) = w_0 + (1 - a)w_a`, we can request it
                      like so:

                      .. code-block:: python3

                         class_params = {
                             'Omega_Lambda':    0,  # disable cosmological constant
                             'w0_fld'      : -0.8,
                             'wa_fld'      :  0.1,
                         }

                      See the :ref:`tuturial <dynamical_dark_energy>` for a
                      walk-through of using a cosmology with dynamical dark
                      energy.
== =============== == =

