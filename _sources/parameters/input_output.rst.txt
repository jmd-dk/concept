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

                      When realising two particle components with the same
                      number of particles :math:`N`, the two particle
                      distributions will be pre-initialised on relatively
                      shifted lattices, amounting to using a body-centered
                      cubic (bcc) lattice for the combined particle system.
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

-- --------------- -- -
\  **Example 5**   \  Generate initial conditions consisting of a single
                      component comprised of :math:`2\times 64^3` matter
                      particles:

                      .. code-block:: python3

                         initial_conditions = {
                            'species': 'matter',
                            'N'      : 2*64**3,
                         }

                      As the number of particles is of the form
                      :math:`N = 2n^3` rather than the standard
                      :math:`N = n^3`, the particles will be pre-initialised
                      on lattice points of a body-centered (bcc) lattice
                      rather than a simple cubic (sc) lattice. Similarly,
                      components with :math:`N = 4n^3` (e.g. ``'N': 4*64**3``)
                      particles will be pre-initialised on a face-centered
                      cubic (fcc) lattice.

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
                             'bispec'   : path.output_dir,
                             'render2D' : path.output_dir,
                             'render3D' : path.output_dir,
                             'autosave' : f'{path.ic_dir}/autosave',
                         }

-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` with the keys ``'snapshot'``,
                      ``'powerspec'``, ``'bispec'``, ``'render2D'``,
                      ``'render3D'`` and ``'autosave'``, mapping to directory
                      paths to use for snapshot outputs, power spectrum
                      outputs, bispectrum outputs, 2D render outputs,
                      3D render outputs and autosaves, respectively.
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
                             'bispec'   : ...,
                             'render2D' : ...,
                             'render3D' : ...,
                         }

-- --------------- -- -
\  **Example 2**   \  Dump all output (even autosaves) to the directory
                      containing the parameter file currently in use:

                      .. code-block:: python3

                         output_dirs = {
                             'snapshot' : param.dir,
                             'powerspec': ...,
                             'bispec'   : ...,
                             'render2D' : ...,
                             'render3D' : ...,
                             'autosave' : ...,
                         }

                      When all the different outputs should go to the same
                      directory (like above), we may instead specify this as
                      simply

                      .. code-block:: python3

                         output_dirs = param.dir

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
                             'bispec'   : 'bispec',
                             'render2D' : 'render2D',
                             'render3D' : 'render3D',
                         }

-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` with the keys ``'snapshot'``,
                      ``'powerspec'``, ``'bispec'``, ``'render2D'`` and
                      ``'render3D'``, mapping to file base names of the
                      respective output types.

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
                      ``'snapshot'``, ``'powerspec'``, ``'bispec'``,
                      ``'render2D'`` and ``'render3D'``, mapping to scale
                      factor values :math:`a` at which to dump the respective
                      outputs.

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
                      :math:`a = a_{\text{begin}}` and final :math:`a = 1`,
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
                      :math:`t = 1729\,\text{Myr}` and
                      :math:`t = 13\,\text{Gyr}`, as well as at scale factor
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
\  **Description** \  Specifies what data of which components to include when
                      reading and writing snapshots
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'save': {
                                 'default': {
                                     'pos': True,
                                     'mom': True,
                                     'ϱ'  : True,
                                     'J'  : True,
                                 },
                             },
                             'load': {
                                 'default': {
                                     'pos': True,
                                     'mom': True,
                                     'ϱ'  : True,
                                     'J'  : True,
                                 },
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  The sub\ ``dict``\ s ``snapshot_select['save']`` and
                      ``snapshot_select['load']`` are
                      :ref:`component selections <components_and_selections>`
                      determining what data of which components to include
                      when writing and reading snapshots, respectively.
                      Here ``'pos'`` and ``'mom'`` are particle positions and
                      momenta, respectively, while ``'ϱ'`` and ``'J'`` are
                      fluid energy and momentum densities, respectively.
-- --------------- -- -
\  **Example 0**   \  Only include the component with a name/species of
                      ``'matter'``, for both reading and writing.
                      Include all data, which is generally desirable:

                      .. code-block:: python3

                         snapshot_select = {
                             'save': {
                                 'matter': {
                                     'pos': True,
                                     'mom': True,
                                     'ϱ'  : True,
                                     'J'  : True,
                                 },
                             },
                             'load': {
                                 'matter': {
                                     'pos': True,
                                     'mom': True,
                                     'ϱ'  : True,
                                     'J'  : True,
                                 },
                             },
                         }

                      When all data is to be included,
                      the above can be simplified to

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

                      Components not captured by any specification defaults
                      to ``True``, so the above may be shortened to

                      .. code-block:: python3

                         snapshot_select = {
                             'save': {
                                 'fluid': False,
                             },
                         }

-- --------------- -- -
\  **Example 2**   \  Only read in positions when loading particles,
                      i.e. ignore momenta:

                      .. code-block:: python3

                         snapshot_select = {
                             'load': {
                                 'particles': {
                                     'pos': True,
                                     'mom': False,
                                 },
                             },
                         }

                      Data variables left out defaults to ``False``,
                      so the above may be shortened to

                      .. code-block:: python3

                         snapshot_select = {
                             'load': {
                                 'particles': {
                                     'pos': True,
                                 },
                             },
                         }

                      .. caution::
                         Leaving out certain data when reading in snapshots
                         will result in components not being fully
                         initialized, e.g. in this example all particles
                         loaded from disk will not have any momenta assigned
                         (not even :math:`0`). Running a simulation with such
                         a partially initialised component will result in a
                         crash.

                      The usefulness of this example is found when using e.g.
                      the :doc:`/utilities/powerspec` utility, where reading
                      in the momentum information only wastes time and memory.

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
                                 'data'            : True,
                                 'linear'          : True,
                                 'linear imprinted': False,
                                 'plot'            : True,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      determining which components participate in power
                      spectrum output, as well as what kind of power
                      spectrum output to include.

                      Here ``'data'`` refers to text files containing
                      tabulated values of the (auto) power spectrum
                      :math:`P(k)`. A separate column within these files
                      containing the corresponding linear-theory power
                      spectrum is added if ``'linear'`` is also selected.
                      The ``'linear imprinted'`` output is again the
                      linear-theory power spectrum, but computed by first
                      realising a full 3D density field and then computing its
                      power spectrum, leaving the
                      :ref:`primordial noise <random_seeds>` imprinted on
                      the spectrum.
                      Selecting ``'plot'`` results in a plot of the specified
                      data, stored as a PNG file.

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
                             'all': False,
                             'matter': {
                                 'plot': True,
                             },
                         }

-- --------------- -- -
\  **Example 3**   \  Output imprinted linear power spectra rather than pure
                      linear power spectra, for all components:

                      .. code-block:: python3

                         powerspec_select = {
                             'all': {
                                 'data'            : True,
                                 'linear'          : False,
                                 'linear imprinted': True,
                                 'plot'            : True,
                             },
                         }

                      .. tip::
                         As the "imprinted" version of the linear power
                         spectrum shares both the
                         :ref:`primordial random noise <random_seeds>` and the
                         binning with the non-linear simulation power
                         spectrum, ratios between non-linear and linear power
                         spectra become less noisy when using imprinted
                         linear spectra.

-- --------------- -- -
\  **Example 4**   \  Create full (auto) power spectrum outputs for
                      all components, as well as for the combined
                      ``'matter'`` and ``'neutrino'`` components:

                      .. code-block:: python3

                         powerspec_select = {
                             'all'                 : True,
                             ('matter', 'neutrino'): True,
                         }

== =============== == =



------------------------------------------------------------------------------



.. _bispec_select:

``bispec_select``
.................
== =============== == =
\  **Description** \  Specifies the kind of bispectrum output to include
                      for different components
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'default': {
                                 'data'      : True,
                                 'reduced'   : True,
                                 'tree-level': True,
                                 'plot'      : True,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      determining which components participate in bispectrum
                      output, as well as what kind of bispectrum output
                      to include.

                      Here ``'data'`` refers to text files containing
                      tabulated values of the (auto) bispectrum
                      :math:`B(k, t, \mu) = B(k_1, k_2, k_3)`. If
                      ``'reduced'`` is also selected, a separate column
                      containing the reduced bispectrum
                      :math:`Q(k_1, k_2, k_3) \equiv B(k_1, k_2, k_3)/[P(k_1)P(k_2) + P(k_2)P(k_3) + P(k_3)P(k_1)]`
                      will be included within the text file. When
                      ``'tree-level'`` is selected, a separate column
                      containing the perturbative tree-level prediction of
                      :math:`B` is added, and likewise for :math:`Q` if
                      ``'reduced'`` is simultaneously selected. Selecting
                      ``'plot'`` results in a plot of the specified data,
                      stored as a PNG file.

                      .. note::
                         Tree-level predictions are available for matter-like
                         species only, and are
                         `given by <https://arxiv.org/abs/1602.05933>`__

                         .. math::

                            B_{\text{tree-level}}(k_1, k_2, k_3) = 2 [ \quad &\mathcal{K}(k_1, k_2, k_3)P_{\text{L}}(k_1)P_{\text{L}}(k_2) \\
                            + &\mathcal{K}(k_2, k_3, k_1)P_{\text{L}}(k_2)P_{\text{L}}(k_3) \\
                            + &\mathcal{K}(k_3, k_1, k_2)P_{\text{L}}(k_3)P_{\text{L}}(k_1) ]\,,

                         .. math::

                            \mathcal{K}(k_1, k_2, k_3) = 1 - \frac{1}{2}\biggl(1 + \frac{D^{(2)}}{D^2}\biggr) (1 - \mu^2) - \frac{\mu}{2}\biggl( \frac{k_1}{k_2} + \frac{k_2}{k_1} \biggr)\,,

                         .. math::

                            \mu = -\hat{\boldsymbol{k}}_1\cdot\hat{\boldsymbol{k}}_2 = \frac{k_1^2 + k_2^2 - k_3^2}{2k_1k_2}\,,

                         where :math:`P_{\text{L}}(k)` is the linear power
                         spectrum while :math:`D` and :math:`D^{(2)}` are the
                         first- and second-order growth factors.

                      To tune the specifics of how bispectra are computed, see
                      the ``bispec_options`` :ref:`parameter <bispec_options>`.
-- --------------- -- -
\  **Example 0**   \  Dump bispectrum data files containing spectra for all
                      components, including both non-linear and tree-level
                      data, for both the full and reduced bispectrum. Do not
                      dump any plots of this data:

                      .. code-block:: python3

                         bispec_select = {
                             'all': {
                                 'data'      : True,
                                 'reduced'   : True,
                                 'tree-level': True,
                                 'plot'      : False,
                             },
                         }

-- --------------- -- -
\  **Example 1**   \  Leave out the reduced bispectrum for every component
                      except the one with a name/species of ``'matter'``.
                      Do not include any tree-level predictions and do not
                      make any plots:

                      .. code-block:: python3

                         bispec_select = {
                             'all': {
                                 'data': True,
                             },
                             'matter': {
                                 'data'   : True,
                                 'reduced': True,
                             },
                         }

                      .. note::
                         Unspecified values are assigned ``False``

-- --------------- -- -
\  **Example 2**   \  Do not create any bispectrum outputs except plots
                      of the component with a name/species of ``'matter'``.
                      Include both the full and reduced bispectrum as well as
                      their tree-level predictions in the plots:

                      .. code-block:: python3

                         bispec_select = {
                             'all'   : False,
                             'matter': {
                                 'reduced'   : True,
                                 'tree-level': True,
                                 'plot'      : True,
                             },
                         }

-- --------------- -- -
\  **Example 3**   \  Create full (auto) bispectrum outputs for all
                      components, as well as for the combined
                      ``'matter'`` and ``'neutrino'`` components:

                      .. code-block:: python3

                         bispec_select = {
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

                         {
                             'default': True,
                         }

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      determining which components participate in 3D
                      render outputs. These are stored as PNG files.

                      To tune the specifics of how 3D renders are created,
                      see the ``render3D_options``
                      :ref:`parameter <render3D_options>`.
-- --------------- -- -
\  **Example 0**   \  Only do 3D renders of the component with a name/species
                      of ``'matter'``:

                      .. code-block:: python3

                         render3D_select = {
                             'matter': True,
                         }

                      In a manner similar to the specifications within e.g.
                      the ``render2D_select``
                      :ref:`parameter <render2D_select>`
                      we may also specify this as

                      .. code-block:: python3

                         render3D_select = {
                             'matter': {'image': True},
                         }

-- --------------- -- -
\  **Example 1**   \  Create 3D render outputs for the combined
                      ``'matter'`` and ``'neutrino'`` components,
                      and nothing else:

                      .. code-block:: python3

                         render3D_select = {
                             ('matter', 'neutrino'): True,
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
                      `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__
                      format, and ``'gadget'``, which is the binary, non-HDF5
                      format of
                      `GADGET <https://wwwmpa.mpa-garching.mpg.de/gadget/>`__.
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
                         `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__).
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
                             'particles per file': 'automatic',
                             'parallel write': True,
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
                      GADGET snapshots:

                      * ``'snapformat'``: Specifies whether GADGET snapshots
                        should use a ``SnapFormat`` of ``1`` or ``2``. Note
                        that ``SnapFormat`` ``3`` (the HDF5 format) is not
                        available. See section 5.1 in the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__
                        for more information.
                      * ``'dataformat'``: This is a ``dict`` specifying the
                        data type sizes to use when writing out particle
                        positions, velocities and
                        :ref:`IDs <select_particle_id>`. The corresponding
                        keys are ``'POS'``, ``'VEL'`` and ``'ID'``, which may
                        all have a value of either ``32`` or ``64``,
                        specifying the size in bits (corresponding to single-
                        or double-precision for ``'POS'`` and ``'VEL'``, and
                        4- or 8-byte unsigned integers (typically corresponding
                        to ``unsigned int`` and ``unsigned long long`` in C)
                        for ``'ID'``). In addition, the value of ``'ID'`` may
                        also be set to ``'automatic'``, in which case 32 bits
                        will be used if this is enough to uniquely label each
                        particle (:math:`N < 2^{32}`). If not, 64 bits will
                        be used.
                      * ``'particles per file'``: This specifies the maximum
                        number of total particles (across all particle types)
                        to save within each file of a distributed (multi-file)
                        GADGET snapshot. When set to ``'automatic'``
                        (the default), CO\ *N*\ CEPT will tune this number to
                        be as large as possible, given the limitations of the
                        file format. For single-precision data, this comes out
                        to be :math:`178\,956\,969 \approx 563^3` particles
                        per file. By manually setting ``'particles per file'``
                        to some number, you can control the maximum file size.
                      * ``'parallel write'``: Boolean specifying whether to
                        write out snapshot files in parallel for distributed
                        (multi-file) GADGET snapshots.
                      * ``'Nall high word'``: The ``Nall`` field of the header
                        (see table 4 of the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__)
                        is meant to store the total number :math:`N` of
                        particles (of each type) within the snapshot, summed
                        over all files in case of the snapshot being
                        distributed over several files. Unfortunately, this is
                        a 32-bit field, and so cannot store :math:`N` in the
                        case of :math:`N \ge 2^{32}`. To overcome this
                        limitation, another 32-bit field ``NallHW`` exists,
                        meant to contain the "high word" part of a now
                        distributed 64-integer, with ``Nall`` supplying the
                        "low word" part. This is the behaviour given the
                        default

                        .. code-block:: python3

                           'Nall high word': 'NallHW'

                        A separate convention (used by at least some versions
                        of
                        `NGenIC <https://www.h-its.org/2014/11/05/ngenic-code/>`__)
                        is to only allow for particle type 1 (``halo``
                        particles, corresponding to (cold dark) matter; see
                        table 3 of the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__)
                        and store all 64 bits within ``Nall``, overflowing into
                        the (now unused) slot usually designated to particle
                        type 2. This convention can be chosen by specifying

                        .. code-block:: python3

                           'Nall high word': 'Nall'

                      * ``'header'``: The contents of the GADGET header (the
                        ``HEAD`` block) will match the specifications in table
                        4 of the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__.
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

                      Sub-parameters which affect the *reading* of
                      GADGET snapshots:

                      * ``'settle'``: If a GADGET snapshot is stored in
                        ``SnapFormat`` ``2`` (see the
                        `user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__),
                        the size of a block is effectively stored twice before
                        its data begins. If these two sizes disagree, we need
                        to settle for one of them. A value of ``0`` for
                        ``'settle'`` picks the first size, while a value of
                        ``1`` picks the second size. Regardless, a warning
                        will be given if the two sizes disagree.

                      Sub-parameters which affect *both* the reading
                      and writing of GADGET snapshots:

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

-- --------------- -- -
\  **Example 1**   \  Fill each GADGET snapshot file with at most
                      :math:`420^3` particles when writing snapshot files to
                      disk, thus limiting the size of each file to
                      :math:`\sim 2\, \text{GB}` (assuming
                      single-precision):

                      .. code-block:: python3

                         gadget_snapshot_params = {
                             'particles per file': 420**3,
                         }

== =============== == =



------------------------------------------------------------------------------



``snapshot_wrap``
.................
== =============== == =
\  **Description** \  Specifies whether or not to wrap out-of-bounds particles
                      around the periodic box when reading snapshots
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         False

-- --------------- -- -
\  **Elaboration** \  All particles should have positions
                      :math:`0 \leq x, y, z < L_{\text{box}}`, with
                      :math:`L_{\text{box}}` corresponding to the
                      ``boxsize`` :ref:`parameter <boxsize>`.
                      During simulation, particles drifting out of the cubic
                      box is immediately wrapped around.

                      When reading particles from a snapshot, some particles
                      may be erroneously located outside of the box. If a
                      particle is positioned exactly on the upper boundary
                      :math:`L_{\text{box}}`, this is silently wrapped back
                      to :math:`0`. Positions beyond this as well as
                      negative positions are counted as out-of-bounds.

                      If this parameter is set to ``False`` (the default), any
                      out-of-bounds particles found within a snapshot will
                      cause CO\ *N*\ CEPT to terminate with an error message.
                      Setting this parameter to ``True``, all out-of-bounds
                      particles read from snapshots will be silently wrapped
                      around, placing them within the box.
-- --------------- -- -
\  **Example 0**   \  Allow and correct for out-of-bounds particles read from
                      snapshots:

                      .. code-block:: python3

                         snapshot_wrap = True

== =============== == =



------------------------------------------------------------------------------



.. _select_particle_id:

``select_particle_id``
......................
== =============== == =
\  **Description** \  Specifies components that should keep track of particle
                      IDs
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'default': False,
                         }

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      specifying particle components that should make use of
                      particle IDs, i.e. unique integer labels, one for each
                      particle.

                      When a component using particle IDs are saved to a
                      snapshot, the IDs are saved as well.

                      .. note::

                         When writing GADGET snapshots, IDs will be written
                         even for components that do not make use of paticle
                         IDs. In this case, some IDs are simply made up when
                         the snapshot is written. Thus, the IDs in GADGET
                         snapshots should not be relied upon in such cases.

                      When saving a GADGET snapshot, the data type used for
                      the IDs is determined by the ``gadget_snapshot_params``
                      :ref:`parameter <gadget_snapshot_params>`.

                      When saving a CO\ *N*\ CEPT snapshot, the data type used
                      for the IDs is automatically determined to be an
                      unsigned 8-, 16-, 32- or 64-bit integer.

                      When a component that should use particle IDs are loaded
                      from a CO\ *N*\ CEPT snapshot that does not contain such
                      IDs, new IDs are assigned.

-- --------------- -- -
\  **Example 0**   \  Use particle IDs for all particle components:

                      .. code-block:: python3

                         select_particle_id = {
                             'particles': True,
                         }

== =============== == =



------------------------------------------------------------------------------



.. _class_plot_perturbations:

``class_plot_perturbations``
............................
== =============== == =
\  **Description** \  Specifies whether to plot CLASS perturbations used
                      within the CO\ *N*\ CEPT run
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         False

-- --------------- -- -
\  **Elaboration** \  When enabled, all `CLASS <http://class-code.net/>`__
                      perturbations used within the CO\ *N*\ CEPT run will be
                      plotted and saved to image files. This is primarily
                      intended for visual checks of convergence of CLASS
                      computations.

                      Two new directories will be created --- both within the
                      specified power spectrum
                      :ref:`output directory <output_dirs>`
                      ``output_dirs['powerspec']`` --- containing
                      subdirectories for the various perturbations. The two
                      directories are:

                      * ``class_perturbations``: The plots within this
                        directory show the evolution of each of the computed
                        :math:`k` modes through time :math:`a`, for each type
                        of perturbation. The time axis is cut into regions,
                        within each of which the perturbations are
                        'detrended', meaning that a trend-line in the form of
                        a power law in :math:`a` has been fitted and
                        subsequently subtracted. This detrending is done prior
                        to any spline interpolation, greatly increasing the
                        accuracy of interpolation.

                        Perturbations used only indirectly (e.g. if they enter
                        in a used gauge transformation) are plotted here
                        as well.

                      * ``class_perturbations_processed``: The plots within
                        this directory are of the final, processed
                        perturbations, as they are used by CO\ *N*\ CEPT
                        for e.g. initial condition generation. These are
                        plotted as functions of :math:`k`, for various
                        :math:`a`.

                      .. note::
                         This feature is primarily meant to be used with the
                         :doc:`class utility </utilities/class>`.

-- --------------- -- -
\  **Example 0**   \  Plot all CLASS perturbations used within the
                      CO\ *N*\ CEPT run:

                      .. code-block:: python3

                         class_plot_perturbations = True

== =============== == =



------------------------------------------------------------------------------



.. _class_extra_background:

``class_extra_background``
..........................
== =============== == =
\  **Description** \  Specifies additional CLASS background quantities to
                      include as part of the CLASS data
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         set()

-- --------------- -- -
\  **Elaboration** \  Only a subset of the available
                      `CLASS <http://class-code.net/>`__ background quantities
                      are used by CO\ *N*\ CEPT, and so only these are
                      retrieved from CLASS computations. This also means that
                      only these specific background quantities end up in the
                      CLASS data stored on disk, be it the automatic CLASS
                      disk cache or the data files generated by the
                      :doc:`class utility </utilities/class>`.

                      To include extra CLASS background quantities --- not
                      used by CO\ *N*\ CEPT --- within these files, specify
                      them within the ``class_extra_background`` parameter.
                      You can refer to the extra quantities by their name as
                      defined by CLASS. In addition, the following easier
                      names are provided by CO\ *N*\ CEPT:

                      * ``τ`` or ``tau``: The conformal time :math:`\tau` (in
                        CLASS called ``conf. time [Mpc]``).
                      * ``D``: The linear growth factor :math:`D` (in CLASS
                        called ``gr.fac. D``).
                      * ``f``: The linear growth rate
                        :math:`f \equiv \mathrm{d}\ln D / \mathrm{d}\ln a`
                        (in CLASS called ``gr.fac. f``).
                      * ``D2``: The second-order growth factor :math:`D^{(2)}`
                        (in CLASS called ``gr.fac. D2``).
                      * ``f2``: The second-order growth rate
                        :math:`f^{(2)} \equiv \mathrm{d}\ln D^{(2)} / \mathrm{d}\ln a`
                        (in CLASS called ``gr.fac. f2``).

                      .. note::
                         When running the
                         :doc:`class utility </utilities/class>`, the
                         ``class_extra_background`` parameter is by default
                         set to

                         .. code-block:: python3

                            {'tau', 'D', 'f', 'D2', 'f2'}

-- --------------- -- -
\  **Example 0**   \  Include the conformal time :math:`\tau` among the
                      CLASS background quantities when dumping these to disk,
                      e.g. when running the CO\ *N*\ CEPT
                      :doc:`class utility </utilities/class>`:

                      .. code-block:: python3

                         class_extra_background = 'τ'

                      We can also refer to :math:`\tau` using its CLASS name:

                      .. code-block:: python3

                         class_extra_background = 'conf. time [Mpc]'

-- --------------- -- -
\  **Example 1**   \  Include the linear growth factor :math:`D` and rate
                      :math:`f` among the background quantities when dumping
                      these to disk, e.g. when running the CO\ *N*\ CEPT
                      :doc:`class utility </utilities/class>`:

                      .. code-block:: python3

                         class_extra_background = {'D', 'f'}

== =============== == =



------------------------------------------------------------------------------



.. _class_extra_perturbations:

``class_extra_perturbations``
.............................
== =============== == =
\  **Description** \  Specifies additional CLASS perturbations to include as
                      part of the CLASS data
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         set()

-- --------------- -- -
\  **Elaboration** \  Only a subset of the available
                      `CLASS <http://class-code.net/>`__ perturbations are
                      used by CO\ *N*\ CEPT, and so only these are retrieved
                      from CLASS computations. This also means that only these
                      specific perturbations end up in the CLASS data stored
                      on disk, be it the automatic CLASS disk cache or the
                      data files generated by the
                      :doc:`class utility </utilities/class>`.

                      To include extra CLASS perturbations --- not used by
                      CO\ *N*\ CEPT --- within these files, specify them
                      within the ``class_extra_perturbation`` parameter. You
                      can refer to the extra quantities by their name as
                      defined by CLASS. In addition, the following fancy
                      names are provided by CO\ *N*\ CEPT:

                      * ``θ_tot``: The total velocity divergence
                        :math:`\theta_{\text{tot}}` from all species (in CLASS
                        called ``theta_tot``).
                      * ``ϕ``: The spatial metric perturbation :math:`\phi` in
                        conformal Newtonian gauge (in CLASS called ``phi``).
                      * ``ψ``: The temporal metric perturbation :math:`\psi`
                        in conformal Newtonian gauge (in CLASS called ``psi``).
                      * ``hʹ``: The conformal time derivative of the trace of
                        the spatial metric perturbation in synchronous gauge,
                        :math:`\partial_{\tau} h` (in CLASS called
                        ``h_prime``).
                      * ``H_Tʹ``: The conformal time derivative of the
                        trace-free component of the spatial metric in *N*-body
                        gauge, :math:`\partial_{\tau} H_{\text{T}}` (in CLASS
                        called ``H_T_prime``).

-- --------------- -- -
\  **Example 0**   \  Include the two conformal Newtonian metric potentials
                      :math:`\phi` and :math:`\psi` among the CLASS
                      perturbations when dumping these to disk,
                      e.g. when running the CO\ *N*\ CEPT
                      :doc:`class utility </utilities/class>`:

                      .. code-block:: python3

                         class_extra_perturbations = {'ϕ', 'ψ'}

                      We can also refer to these using their CLASS names:

                      .. code-block:: python3

                         class_extra_background = {'phi', 'psi'}

== =============== == =

