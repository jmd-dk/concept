Units
-----
The 'units' parameter category contains parameters specifying the unit system
used by CO\ *N*\ CEPT. This system of units is used both internally and
for output.

.. contents::
   :local:
   :depth: 1



------------------------------------------------------------------------------



``unit_length``
...............
== =============== == =
\  **Description** \  Specifies the base length unit to use
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         'Mpc'

-- --------------- -- -
\  **Elaboration** \  Internally, all dimensional values (be it lengths or
                      otherwise) are represented as bare floating-point
                      numbers. The number ``1.0`` then corresponds to some
                      length. This length is what this parameter specifies.
-- --------------- -- -
\  **Example 0**   \  Use kiloparsec as the base length unit:

                      .. code-block:: python3

                         unit_length = 'kpc'

                      .. caution::
                         Note that unlike when using units when defining other
                         parameters, here ``'kpc'`` is used as a ``str``. This
                         is because here we want the symbolic meaning of a
                         'kiloparsec', whereas in other places what we are
                         after is its numeric value (which really is not
                         defined prior to specifying ``unit_length``).

-- --------------- -- -
\  **Example 1**   \  Many other codes make use of units like
                      :math:`\text{Mpc}/h` or :math:`\text{kpc}/h`, where
                      :math:`h \equiv H_0/(100\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1})`
                      and :math:`H_0` is the :ref:`Hubble constant <H0>`.
                      Specifying simply ``'Mpc/h'`` as ``unit_length`` will
                      not work however, since ``h`` itself can only be
                      inferred from ``H0`` once the numeric value of ``Mpc``
                      has been established, which itself depends on
                      ``unit_length``. To bypass this issue, we can use

                      .. code-block:: python3

                         unit_length = 'Mpc/0.67'

                      given that ``h = 0.67``, corresponding to
                      ``H0 = 67*km/(s*Mpc)``.

== =============== == =



------------------------------------------------------------------------------



``unit_time``
.............
== =============== == =
\  **Description** \  Specifies the base time unit to use
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         'Gyr'

-- --------------- -- -
\  **Elaboration** \  Internally, all dimensional values (be it times or
                      otherwise) are represented as bare floating-point
                      numbers. The number ``1.0`` then corresponds to some
                      time. This time is what this parameter specifies.
-- --------------- -- -
\  **Example 0**   \  Use seconds as the base time unit:

                      .. code-block:: python3

                         unit_time = 's'

-- --------------- -- -
\  **Example 1**   \  Use the same base time unit as is (implicitly) used in
                      CLASS:

                      .. code-block:: python3

                         unit_time = 'Mpc/c'

== =============== == =



------------------------------------------------------------------------------



``unit_mass``
.............
== =============== == =
\  **Description** \  Specifies the base mass unit to use
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         '10¹⁰ m☉'

-- --------------- -- -
\  **Elaboration** \  Internally, all dimensional values (be it masses or
                      otherwise) are represented as bare floating-point
                      numbers. The number ``1.0`` then corresponds to some
                      mass. This mass is what this parameter specifies.
-- --------------- -- -
\  **Example 0**   \  Use the same base mass unit as is (implicitly) used in
                      CLASS (at least in the background module):

                      .. code-block:: python3

                         unit_mass = '3/(8 π G) c² Mpc'

                      .. note::
                         In specifying the symbolic unit, we are allowed to be
                         fancy as above. We could equivalently use e.g.

                         .. code-block:: python3

                            unit_mass = '119366.2073189215*light_speed**2*pc/G_Newton'

== =============== == =

