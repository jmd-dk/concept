Input/output
------------
The 'input/output' parameter category contains parameters specifying the
initial conditions as well as the wanted results to be outputted from the
CO\ *N*\ CEPT run.

.. contents::
   :local:
   :depth: 1



------------------------------------------------------------------------------



.. _initial_conditions:

``initial_conditions``
......................
== =============== == =
\  **Description** \  Specifies the initial conditions of the simulation
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         ''  # no initial conditions

-- --------------- -- -
\  **Elaboration** \  There are two kinds of initial condition specifications:

                      * Component specifications, describing a component for
                        which initial conditions should be generated.
                      * Paths to snapshot files containing the initial
                        conditions.

                      Several such specifications may be used together when
                      defining ``initial_conditions``, and the two kinds may
                      be used together.
-- --------------- -- -
\  **Example 0**   \  Generate initial conditions consisting of a single
                      component comprised of :math:`128^3` matter particles.
                      The particle positions and momenta will be set according
                      to the combined baryonic and cold dark matter transfer
                      functions:

                      .. code-block:: python3

                         initial_conditions = {
                            'species': 'matter',
                            'N'      : 128**3,
                         }

-- --------------- -- -
\  **Example 1**   \  Load initial conditions from the snapshot ``snapshot``
                      in the ``ic`` directory:

                      .. code-block:: python3

                         initial_conditions = f'{path.ic}/snapshot'

                      .. note::
                         If the snapshot is distributed over multiple files,
                         use the path to either the first file, the directory
                         containing the files (and nothing else) or use
                         globbing as in ``f'{path.ic}/snapshot*'``.

-- --------------- -- -
\  **Example 2**   \  Generate initial conditions where baryonic and cold dark
                      matter particles are realised separately as individual
                      components, each comprised of :math:`64^3` particles:

                      .. code-block:: python3

                         initial_conditions = [
                             {
                                 'species': 'cold dark matter',
                                 'N'      : 64**3,
                             },
                             {
                                 'species': 'baryon',
                                 'N'      : 64**3,
                             },
                         ]
-- --------------- -- -
\  **Example 3**   \  Generate initial conditions consisting of a combined
                      matter component with :math:`64^3` particles, as well as
                      a fluid component containing the combined linear energy
                      density perturbations of the photons, the neutrinos and
                      the metric, having a linear grid size of :math:`64` (and
                      thus :math:`64^3` fluid elements):

                      .. code-block:: python3

                         initial_conditions = [
                             {
                                 'species': 'matter',
                                 'N'      : 64**3,
                             },
                             {
                                 'name'           : 'linear component',
                                 'species'        : 'photon + neutrino + metric',
                                 'gridsize'       : 64,
                                 'boltzmann order': -1,  # completely linear
                             },
                         ]

                      .. note::
                         The ``'name'`` assigned to a component is used only
                         for referencing by other parameters and may generally
                         be omitted. If so, this will be set equal to the
                         value of ``'species'``.

-- --------------- -- -
\  **Example 4**   \  Use combined initial conditions from the two snapshots
                      ``snapshot_b`` and ``snapshot_cdm`` in the ``ic``
                      directory, supplemented by a non-linear neutrino
                      component generated on the fly:

                      .. code-block:: python3

                         initial_conditions = [
                             f'{path.ic}/snapshot_b',
                             f'{path.ic}/snapshot_cdm',
                             {
                                 'species'        : 'neutrino',
                                 'gridsize'       : 128,
                                 'boltzmann order': +1,  # non-linear
                             },
                         ]

== =============== == =



------------------------------------------------------------------------------



.. _output_dirs:

``output_dirs``
...............
== =============== == =
\  **Description** \  Directories for storing output
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'snapshot' : path.output_dir,
                             'powerspec': path.output_dir,
                             'render2D' : path.output_dir,
                             'render3D' : path.output_dir,
                             'autosave' : f'{path.ic_dir}/autosave',
                         }

-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` with the keys ``'snapshot'``,
                      ``'powerspec'``, ``'render2D'``, ``'render3D'`` and
                      ``'autosave'``, mapping to directory paths to use for
                      snapshot outputs, power spectrum outputs, 2D render
                      outputs, 3D render outputs and autosaves, respectively.
-- --------------- -- -
\  **Example 0**   \  Dump power spectra to a directory with a name that
                      reflects the name of the parameter file:

                      .. code-block:: python3

                         output_dirs = {
                             'powerspec': f'{path.output_dir}/{param}',
                         }

                      .. note::
                         Unspecified entries will take on their
                         default values

-- --------------- -- -
\  **Example 1**   \  Use the same directory for all output, and let its name
                      reflect the ID of the running job:

                      .. code-block:: python3

                         output_dirs = {
                             'snapshot' : f'{path.output_dir}/{jobid}',
                             'powerspec': ...,
                             'render2D' : ...,
                             'render3D' : ...,
                         }

== =============== == =



------------------------------------------------------------------------------



``output_bases``
................
== =============== == =
\  **Description** \  File base names for output
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'snapshot' : 'snapshot',
                             'powerspec': 'powerspec',
                             'render2D' : 'render2D',
                             'render3D' : 'render3D',
                         }

-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` with the keys ``'snapshot'``,
                      ``'powerspec'``, ``'render2D'`` and ``'render3D'``,
                      mapping to file base names of the respective
                      output types.

                      The file name of e.g. a power spectrum output at scale
                      factor :math:`a = 1.0` will be
                      ``output_bases['powerspec'] + '_a=1.0'``. The directory
                      of this file is :ref:`controlled by <output_dirs>`
                      ``output_dirs['powerspec']``.

-- --------------- -- -
\  **Example 0**   \  Use a shorter name for power spectrum files:

                      .. code-block:: python3

                         output_bases = {
                             'powerspec': 'p',
                         }

                      .. note::
                         Unspecified entries will take on their
                         default values

== =============== == =



------------------------------------------------------------------------------



``output_times``
................
== =============== == =
\  **Description** \  Times at which to dump output
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {}  # no output times

-- --------------- -- -
\  **Elaboration** \  In its simplest form this is a ``dict`` with the keys
                      ``'snapshot'``, ``'powerspec'``, ``'render2D'`` and
                      ``'render3D'``, mapping to scale factor values :math:`a`
                      at which to dump the respective outputs.

                      Alternatively, such ``dict``\ s can be used as values
                      within an outer ``dict`` with keys ``'a'`` and ``'t'``,
                      for specifying output times at either scale factor
                      values :math:`a` or cosmic times :math:`t`.

-- --------------- -- -
\  **Example 0**   \  Specify a single power spectrum output at :math:`a = 1`:

                      .. code-block:: python3

                         output_times = {
                             'powerspec': 1,
                         }

-- --------------- -- -
\  **Example 1**   \  Specify snapshot outputs at :math:`a = 0.1`,
                      :math:`a = 0.3` and :math:`a = 1`:

                      .. code-block:: python3

                         output_times = {
                             'snapshot': [0.1, 0.3, 1],
                         }

-- --------------- -- -
\  **Example 2**   \  Specify 8 power spectrum outputs between the initial
                      :math:`a = a_{\mathrm{begin}}` and final :math:`a = 1`,
                      placed logarithmically equidistant:

                      .. code-block:: python3

                         output_times = {
                             'powerspec': logspace(log10(a_begin), log10(1), 8),
                         }

-- --------------- -- -
\  **Example 3**   \  Specify a series of power spectrum outputs and use these
                      same values for 2D renders:

                      .. code-block:: python3

                         output_times = {
                             'powerspec': [0.03, 0.1, 0.3, 1],
                             'render2D' : ...,
                         }

-- --------------- -- -
\  **Example 4**   \  Specify snapshots at cosmic times
                      :math:`t = 1729\,\mathrm{Myr}` and
                      :math:`t = 13\,\mathrm{Gyr}`, as well as at scale factor
                      :math:`a = 1`.

                      .. code-block:: python3

                         output_times = {
                             't': {
                                 'snapshot': [1729*Myr, 13*Gyr],
                             },
                             'a': {
                                 'snapshot': 1,
                             },
                         }

== =============== == =



------------------------------------------------------------------------------



``autosave_interval``
.....................
== =============== == =
\  **Description** \  Time interval between successive automated saves of the
                      simulation to disk
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         ထ  # never autosave

-- --------------- -- -
\  **Elaboration** \  Setting this to some finite time will periodically dump
                      a snapshot, intended for use with restarting the
                      simulation in case of crashes or similar. The autosaved
                      snapshot is written to a subdirectory
                      :ref:`of <output_dirs>` ``output_dirs['autosave']`` and
                      named in accordance with the parameter file in use.

                      Starting a simulation with the same parameter file will
                      pick up on such an autosaved snapshot, if it exists.

                      When autosaving, the previous autosave will be
                      overwritten (in a fail-safe manner), so that only the
                      newest autosave remains.
-- --------------- -- -
\  **Example 0**   \  Autosave about every hour:

                      .. code-block:: python3

                         autosave_interval = 1*hr

-- --------------- -- -
\  **Example 1**   \  Autosave 5 times a day:

                      .. code-block:: python3

                         autosave_interval = day/5

-- --------------- -- -
\  **Example 2**   \  Disabling autosaving, including starting from an
                      existing autosave on disk:

                      .. code-block:: python3

                         autosave_interval = 0

                      .. note::
                         This is different from having an infinitely long
                         autosave interval,

                         .. code-block:: python3

                            autosave_interval = ထ

                         as this still makes use of already existing
                         autosaves on disk.
== =============== == =



------------------------------------------------------------------------------



.. _snapshot_select:

``snapshot_select``
...................
== =============== == =
\  **Description** \  Specifies which components to include when reading and
                      writing snapshots
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'save': {
                                 'all': True,
                             },
                             'load': {
                                 'all': True,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  The sub\ ``dict``\ s ``snapshot_select['save']`` and
                      ``snapshot_select['load']`` are
                      :ref:`component selections <components_and_selections>`
                      determining which components to include when writing
                      and reading snapshots, respectively.
-- --------------- -- -
\  **Example 0**   \  Only include the component with a name/species of
                      ``'matter'``, for both reading and writing:

                      .. code-block:: python3

                         snapshot_select = {
                             'save': {
                                 'matter': True,
                             },
                             'load': {
                                 'matter': True,
                             },
                         }

                      Equivalently, but a bit shorter:

                      .. code-block:: python3

                         snapshot_select = {
                             'save': {
                                 'matter': True,
                             },
                             'load': ...,
                         }

                      Even shorter still:

                      .. code-block:: python3

                         snapshot_select = {
                             'matter': True,
                         }

-- --------------- -- -
\  **Example 1**   \  Exclude any (and only) fluid components when
                      writing snapshots:

                      .. code-block:: python3

                         snapshot_select = {
                             'save': {
                                 'all'  : True,
                                 'fluid': False,
                             },
                         }

== =============== == =



------------------------------------------------------------------------------



.. _powerspec_select:

``powerspec_select``
....................
== =============== == =
\  **Description** \  Specifies the kind of power spectrum output to include
                      for different components
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'default': {
                                 'data'  : True,
                                 'linear': True,
                                 'plot'  : True,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      determining which components participate in power
                      spectrum output, as well as what kind of power
                      spectrum outputs to include.

                      Here ``'data'`` refers to text files containing
                      tabulated values of the (auto) power spectrum
                      :math:`P(k)`. A separate data column within these files
                      containing the corresponding linear-theory power
                      spectrum is added if ``'linear'`` is also selected.
                      Selecting ``'plot'`` results in a plot of the selected
                      (non-)linear data, stored as a PNG file.

                      .. note::
                         As CO\ *N*\ CEPT runs in *N*\ -body gauge, the output
                         power spectra will also be in this gauge.
                         This includes the linear power spectra.

                      To tune the specifics of how power spectra are computed,
                      see the ``powerspec_options``
                      :ref:`parameter <powerspec_options>`.
-- --------------- -- -
\  **Example 0**   \  Dump power spectrum data files containing spectra for all
                      components, including both non-linear and linear data.
                      Do not dump any plots of this data:

                      .. code-block:: python3

                         powerspec_select = {
                             'all': {
                                 'data'  : True,
                                 'linear': True,
                                 'plot'  : False,
                             },
                         }

-- --------------- -- -
\  **Example 1**   \  Leave out the linear power spectrum for every component
                      except the one with a name/species of ``'matter'``,
                      and do not make any plots:

                      .. code-block:: python3

                         powerspec_select = {
                             'all': {
                                 'data': True,
                             },
                             'matter': {
                                 'data'  : True,
                                 'linear': True,
                             },
                         }

                      .. note::
                         Unspecified values are assigned ``False``

-- --------------- -- -
\  **Example 2**   \  Do not create any power spectrum outputs except plots
                      of the component with a name/species of ``'matter'``:

                      .. code-block:: python3

                         powerspec_select = {
                             'all'   : False,
                             'matter': {
                                 'plot': True,
                             },
                         }

-- --------------- -- -
\  **Example 3**   \  Create full (auto) power spectrum outputs for
                      all components, as well as for the combined
                      ``'matter'`` and ``'neutrino'`` components:

                      .. code-block:: python3

                         powerspec_select = {
                             'all'                 : True,
                             ('matter', 'neutrino'): True,
                         }

== =============== == =



------------------------------------------------------------------------------



.. _render2D_select:

``render2D_select``
....................
== =============== == =
\  **Description** \  Specifies the kind of 2D render output to include
                      for different components
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'default': {
                                 'data'          : True,
                                 'image'         : True,
                                 'terminal image': True,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      determining which components participate in 2D
                      render outputs, as well as what kind of 2D render
                      outputs to include.

                      Here ``'data'`` refers to HDF5 files containing the
                      values of the 2D projection, while ``'image'`` refers to
                      an actual rendered image, stored as a PNG file.
                      Finally, ``'terminal image'`` refers to colour renders
                      printed directly in the terminal, which thus become part
                      of the job log.

                      To tune the specifics of how 2D renders are created,
                      see the ``render2D_options``
                      :ref:`parameter <render2D_options>`.
-- --------------- -- -
\  **Example 0**   \  Store 2D renders as image files for all components,
                      and also display these in the terminal. Do not store
                      the raw 2D render data.

                      .. code-block:: python3

                         render2D_select = {
                             'all': {
                                 'data'          : False,
                                 'image'         : True,
                                 'terminal image': True,
                             },
                         }

-- --------------- -- -
\  **Example 1**   \  Dump 2D render images for all components, but only show
                      the ones for the component with a name/species of
                      ``'neutrino'`` in the terminal:

                      .. code-block:: python3

                         render2D_select = {
                             'all': {
                                 'image': True,
                             },
                             'neutrino': {
                                 'image'         : True,
                                 'terminal image': True,
                             },
                         }

                      .. note::
                         Unspecified values are assigned ``False``

-- --------------- -- -
\  **Example 2**   \  Create full 2D render outputs for the combined
                      ``'matter'`` and ``'neutrino'`` components,
                      and nothing else:

                      .. code-block:: python3

                         render2D_select = {
                             ('matter', 'neutrino'): True,
                         }

== =============== == =



------------------------------------------------------------------------------



``render3D_select``
...................
== =============== == =
\  **Description** \  Specifies which components to include in 3D
                      render outputs
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {'all': True}

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      determining which components participate in 3D
                      render outputs. These are stored as PNG files.

                      Note that you cannot use component combinations as keys
                      in ``render3D_select``. If multiple components are 3D
                      rendered (separately), one additional, total 3D render
                      will also be produced, combining all into one.
-- --------------- -- -
\  **Example 0**   \  Only do 3D renders of the component with a name/species
                      of ``'matter'``:

                      .. code-block:: python3

                         render3D_select = {
                             'matter': True,
                         }

== =============== == =



------------------------------------------------------------------------------



.. _snapshot_type:

``snapshot_type``
.................
== =============== == =
\  **Description** \  Specifies the snapshot format to use when dumping
                      snapshots
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         'concept'

-- --------------- -- -
\  **Elaboration**  \ CO\ *N*\ CEPT understands two snapshot
                      formats; ``'concept'``, which is its own,
                      well-structured
                      `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
                      format, and ``'gadget'``, which is the binary, non-HDF5
                      format of
                      `GADGET <https://wwwmpa.mpa-garching.mpg.de/gadget/>`_.
                      Note that the value of ``snapshot_type`` does
                      not affect which snapshots may be *read*, e.g. used
                      within the ``initial_conditions``
                      :ref:`parameter <initial_conditions>`.
-- --------------- -- -
\  **Example 0**   \  Dump output snapshots in GADGET format:

                      .. code-block:: python3

                         snapshot_type = 'gadget'

                      .. note::

                         Though which components to include in/from snapshots
                         are generally determined by the ``snapshot_select``
                         :ref:`parameter <snapshot_select>`, additional
                         information is needed to map components to/from the
                         *particle types* of GADGET (see table 3 of the
                         `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`_).
                         When loading in a GADGET snapshot, the available
                         particle types will be read into separate components,
                         with names matching the particle type, e.g.
                         ``'GADGET halo'`` (particle type ``1``),
                         ``'GADGET disk'`` (particle type ``2``), etc. To
                         similarly map components in CO\ *N*\ CEPT to specific
                         particle types when writing GADGET snapshots, simply
                         use the appropriate names for the components within
                         CO\ *N*\ CEPT (i.e. set the ``'name'`` according to
                         the GADGET particle type when defining the
                         ``initial_conditions``
                         :ref:`parameter <initial_conditions>`). If a single
                         particle component is to be saved and its name does
                         not correspond to a GADGET particle type, the ``halo``
                         type will be used.

                         By default, the species of all components read from
                         GADGET snapshots will be ``'matter'``. This can be
                         changed through the ``select_species``
                         :ref:`parameter <select_species>`.

                      To adjust the specifics of the GADGET format to
                      your needs, see the ``gadget_snapshot_params``
                      :ref:`parameter <gadget_snapshot_params>`.
== =============== == =



------------------------------------------------------------------------------



.. _gadget_snapshot_params:

``gadget_snapshot_params``
..........................
== =============== == =
\  **Description** \  Specifies various details for reading and writing of
                      GADGET snapshots
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'snapformat': 2,
                             'dataformat': {
                                 'POS': 32,
                                 'VEL': 32,
                                 'ID' : 'automatic',
                             },
                             'Nall high word': 'NallHW',
                             'header': {},
                             'settle': 0,
                             'units': {
                                 'length'  : 'kpc/h',
                                 'velocity': 'km/s',
                                 'mass'    : '10¹⁰ m☉/h',
                             },
                         }

-- --------------- -- -
\  **Elaboration**  \ This parameter is a ``dict`` of several individual
                      sub-parameters, each of which is described below.

                      Sub-parameters which affect the *writing* of
                      GADGET-snapshots:

                      * ``'snapformat'``: Specifies whether GADGET snapshots
                        should use a ``SnapFormat`` of ``1`` or ``2``. Note
                        that ``SnapFormat`` ``3`` (the HDF5 format) is not
                        available. See section 5.1 in the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`_
                        for more information.
                      * ``'dataformat'``: This is a ``dict`` specifying the
                        data type sizes to use when writing out particle
                        positions, velocities and ID's. The corresponding keys
                        are ``'POS'``, ``'VEL'`` and ``'ID'``, which may all
                        have a value of either ``32`` or ``64``, specifying
                        the size in bits (corresponding to single- or double-
                        precision for ``'POS'`` and ``'VEL'``, and 4- or
                        8-byte unsigned integers (typically corresponding to
                        ``unsigned int`` and ``unsigned long long`` in C) for
                        ``'ID'``). In addition, the value of ``'ID'`` may also
                        be set to ``'automatic'``, in which case 32 bits will
                        be used if this is enough to uniquely label each
                        particle (:math:`N \leq 2^{32}`). If not, 64 bits will
                        be used.
                      * ``'Nall high word'``: The ``Nall`` field of the header
                        (see table 4 of the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`_)
                        is meant to store the total number :math:`N` of
                        particles (of each type) within the snapshot, summed
                        over all files in case of the snapshot being
                        distributed over several files. Unfortunately, this is
                        a 32-bit field, and so cannot store :math:`N` in the
                        case of :math:`N > 2^{32}`. To overcome this
                        limitation, another 32-bit field ``NallHW`` exists,
                        meant to contain the "high word" part of a now
                        distributed 64-integer, with ``Nall`` supplying the
                        "low word" part. This is the behaviour given the
                        default

                        .. code-block:: python3

                           'Nall high word': 'NallHW'

                        A separate convention (used by at least some versions
                        of
                        `NGenIC <https://www.h-its.org/2014/11/05/ngenic-code/>`_)
                        is to only allow for particle type 1 (``halo``
                        particles, corresponding to (cold dark) matter; see
                        table 3 of the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`_)
                        and store all 64 bits within ``Nall``, overflowing into
                        the (now unused) slot usually designated to particle
                        type 2. This convention can be chosen by specifying

                        .. code-block:: python3

                           'Nall high word': 'Nall'

                      * ``'header'``: The contents of the GADGET header (the
                        ``HEAD`` block) will match the specifications in table
                        4 of the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`_.
                        You may overwrite the values of the various fields by
                        specifying them in the ``'header'`` sub-\ ``dict``,
                        e.g.

                        .. code-block:: python3

                           'header': {
                               'HubbleParam': 0.7,
                               'FlagSfr'    : 1,
                           }

                        Here ``'HubbleParam'``
                        (:math:`h \equiv H_0/(100\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1})`)
                        will be set to ``0.7``, disregarding the value of the
                        :ref:`Hubble constant <H0>` ``H0`` actually in use.
                        The ``FlagSfr`` field does not mean anything to
                        CO\ *N*\ CEPT and as so will always be set equal to
                        ``0``. This is changed to ``1`` by the above parameter
                        specification.

                      .. note::
                         CO\ *N*\ CEPT does not provide a way to specify the
                         number of files over which to distribute each GADGET
                         snapshot. It simply writes as few files as possible,
                         with the maximum number of particles per file
                         (assuming single-precision data) being
                         :math:`178\,956\,969 \approx 563^3`, the exact number
                         coming about due to the details of the GADGET format.
                         When using double-precision, this number is adjusted
                         accordingly.

                      Sub-parameters which affect the *reading* of
                      GADGET-snapshots:

                      * ``'settle'``: If a GADGET snapshot is stored in
                        ``SnapFormat`` ``2`` (see the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`_),
                        the size of a block is effectively stored twice before
                        its data begins. If these two sizes disagree, we need
                        to settle for one of them. A value of ``0`` for
                        ``'settle'`` picks the first size, while a value of
                        ``1`` picks the second size. Regardless, a warning
                        will be given if the two sizes disagree.

                      Sub-parameters which affect *both* the reading
                      and writing of GADGET-snapshots:

                      * ``'units'``: While the units used for the data within
                        GADGET snapshots are typically as shown above, i.e.

                        .. code-block:: python3

                           'units': {
                               'length'  : 'kpc/h',
                               'velocity': 'km/s',
                               'mass'    : '10¹⁰ m☉/h',
                           }

                        this is not guaranteed. Information about the unit
                        system actually in use is however not stored within
                        the snapshot. Thus, this sub-parameter allows you to
                        specify the units used within existing snapshots, as
                        well as what units should be used when writing out new
                        snapshots.
-- --------------- -- -
\  **Example 0**   \  Use ``SnapFormat`` ``1`` with all data being stored in
                      64-bit, and using ``Mpc/h`` for the base length unit:

                      .. code-block:: python3

                         gadget_snapshot_params = {
                             'snapformat': 1,
                             'dataformat': {
                                 'POS': 64,
                                 'VEL': ...,
                                 'ID' : ...
                             },
                             'units': {
                                 'length': 'Mpc/h',
                             },
                         }

== =============== == =

