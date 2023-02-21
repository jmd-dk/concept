.. |bgn-h3| raw:: html

     <h3>

.. |end-h3| raw:: html

     </h3>



Parameters
==========
The following sections list the various parameters accepted by
CO\ *N*\ CEPT, along with explanations, default and example values. Each
section deals with a specific category of parameters.

To learn how to *use* parameters and parameter files, see the
:doc:`tutorial </tutorial/parameter_files>` as well as the ``-p`` and ``-c``
:doc:`command-line options </command_line_options>`.

Parameters are specified as Python 3 variables. As such, it is helpful to be
familiar with basic Python syntax and
`data types <https://docs.python.org/3/library/stdtypes.html>`__, such as
``str``\ ings, ``list``\ s and ``dict``\ ionaries.

Below you will find the parameter categories, corresponding to the sections
ahead. Besides bringing some organization to the large set of parameters,
these categories have no meaning.

.. toctree::
   :maxdepth: 1

   input_output
   numerics
   cosmology
   physics
   simulation
   graphics
   units
   debugging

Besides these sections, you may also look in the
`provided example parameter files <https://github.com/jmd-dk/concept/tree/master/param>`__.
In particular, ``example_explanatory`` defines close to all supported
parameters. Though this makes it impractically large for actual use, it serves
to briefly showcase the many possible parameters.

The rest of this page contains information regarding parameter specification
generally, independent of the specific parameters.



|bgn-h3| Parameter files as Python scripts |end-h3|

You may think of parameter files as (glorified) Python scripts, and use any
valid Python 3 statements to formulate or compute your parameters,
including ``print()`` for debugging.

Even without any explicit ``import`` statement present in the parameter file,
all of ``numpy`` is readily available. As an example, consider the following
specification of 7 power spectrum outputs placed logarithmically equidistant
in scale factor :math:`a`, from :math:`a = 0.01` to :math:`a = 1`:

.. code-block:: python3

   # Input/output
   output_times = {'powerspec': logspace(log10(0.01), log10(1), 7)}



|bgn-h3| Units and constants |end-h3|

For dimensional parameters, it is important to explicitly tack the unit onto
the value, as no default units are ever assumed by CO\ *N*\ CEPT. For example,

.. code-block:: python3

   # Numerics
   boxsize = 512      # ❌
   boxsize = 512*Mpc  # OK

   # Cosmology
   H0 = 67             # ❌
   H0 = 67*km/(s*Mpc)  # OK

A large set of units is understood:

* ``[mck]m`` (meter), ``AU`` (astronomical unit), ``[kMG]ly`` (light-year),
  ``[kMG]pc`` (parsec).
* ``s`` (second), ``minutes``, ``hr`` (hour), ``day`` (24 hours),
  ``[kMG]yr`` (Julian year).
* ``[k]g`` (gram), ``[kMG]m☉`` or ``[kMG]m_sun`` (solar mass).
* ``[mkMGT]eV`` (electron volt), ``J`` (Joule).

with brackets ``[...]`` indicating allowed SI prefixes.

CO\ *N*\ CEPT always expresses angles in terms of radians.

The following physical constants are further available:

* ``c`` or ``light_speed`` (speed of light in vacuum).
* ``G`` or ``G_Newton`` (gravitational constant).
* ``ħ`` or ``h_bar`` (reduced Planck constant).

.. note::
   The precise relations between the above units and constants are adopted
   from the
   `2019 redefinition of the SI base units <https://en.wikipedia.org/wiki/2019_redefinition_of_the_SI_base_units#Redefinition>`__,
   exact `IAU definitions <https://arxiv.org/abs/1605.09788>`__ and
   experimental
   `2020 values from the Particle Data Group <https://doi.org/10.1093/ptep/ptaa104>`__.

A few mathematical constants are similarly readily available:

* ``π`` or ``pi`` (half-circle constant).
* ``τ`` or ``tau`` (circle constant).
* ``e`` (Euler's number).
* ``ထ`` or ``inf`` (infinity).
* ``machine_ϵ`` or ``eps`` (double-precision machine epsilon).

.. tip::
   For more exotic mathematical constants you are encouraged to make use of
   NumPy/SciPy:

   .. code-block:: python3

      import scipy.special
      scipy.special.zeta(3)  # Riemann ζ(3)

.. note::
   The actual numerical value resulting from e.g. ``512*Mpc`` depends on the
   internal unit system employed by CO\ *N*\ CEPT. The unit system itself is
   not statically defined, but specified on a per-simulation basis via
   :doc:`parameters <units>`. Though the vast majority of users have no need
   to manually specify the internal unit system, it is useful for e.g.
   testing against other simulation codes that use some other, fixed system
   of units.



|bgn-h3| Non-linear parsing of parameter file |end-h3|

Parameters may be defined in terms of each other. Unlike with regular scripts,
the definition order does not matter (i.e. you may reference a variable
*before* it is defined). The magic that makes this work is cyclic execution of
the parameter file, so that a variable that is needed on some line but only
defined by some later line, is correctly set at the second execution. This
non-linear variable dependence may continue arbitrarily, requiring many more
execution passes.

In the following example, the initial scale factor value ``a_begin`` is
included in the list of scale factor values at which to dump power
spectrum output, even though ``a_begin`` is not defined before further down:

.. code-block:: python3

   # Input/output
   output_times = {'powerspec': [a_begin, 0.3, 1.0]}

   # Cosmology
   a_begin = 0.01



|bgn-h3| The :raw-html:`<code class="docutils literal notranslate">path</code>`, :raw-html:`<code class="docutils literal notranslate">param</code>` and :raw-html:`<code class="docutils literal notranslate">jobid</code>` objects |end-h3|

Available for use in parameter files is the convenient ``path`` object, which
holds absolute paths to various useful directories and files. For example,
``path.output_dir`` evaluates to the output directory, which by default is
``output``. If you prefer, you may instead use the ``dict``-like
syntax ``path['output_dir']``. The contents of the ``path`` variable is
supplied by the ``.path`` file.

Another convenience object available within parameter files is ``param``,
which evaluates to the name of the parameter file in which it is used. As an
example, consider this nice way of ensuring that the power spectrum output
gets dumped in a subdirectory of the output directory, labelled according to
the parameter file:

.. code-block:: python3

   # Input/output
   output_dirs = {'powerspec': f'{path.output_dir}/{param}'}

If used within the parameter file ``param/my_param``,
``f'{path.output_dir}/{param}'`` then evaluates to ``'output/my_param'``.

The ``param`` object is not just a plain ``str`` though:

* ``param.name``: Evaluates to the name of the parameter file (equivalent to
  to using ``param`` on its own within a ``str``\ -context).
* ``param.path``: Evaluates to the absolute path of the parameter file.
* ``param.dir``: Evaluates to the absolute path of the directory containing
  the parameter file.

Lastly, the ``jobid`` variable is also available within parameter files. This
is the unique integer ID labelling the current job. To e.g. direct power
spectrum output to a directory named after this ID, one can use

.. code-block:: python3

   # Input/output
   output_dirs = {'powerspec': f'{path.output_dir}/{jobid}'}

This way, previous outputs are not overwritten when running multiple
simulations using the same parameter file.



|bgn-h3| Custom non-parameter variables |end-h3|

Using custom non-parameter variables in parameter files can help provide
better organisation. For example, you may want to define both the ``boxsize``
and the resolution of the gravitational potential grid in terms of a common
variable:

.. code-block:: python3

   # Non-parameter helper variable used to control the size of the simulation
   _size = 256

   # Numerics
   boxsize = _size*Mpc
   potential_options = 2*_size

Both the ``boxsize`` and the potential grid size given by
``potential_options`` can now be simultaneously updated through the newly
introduced ``_size`` variable, which itself is not a parameter. When defined
in a parameter file, CO\ *N*\ CEPT treats any variable whose name does not
begin with an underscore '``_``' as a proper parameter; hence '``_size``' and
not '``size``'.

.. note::
   The ``potential_options`` parameter is in fact much more involved than what
   it would appear from the above example. The ``potential_options`` is an
   example of a "nested" parameter, with many "sub-parameters" definable
   within it. For your convenience, CO\ *N*\ CEPT allows for a plethora of
   different non-complete specifications of such nested parameters (with
   non-specified sub-parameters being auto-assigned), simplifying the writing
   of parameter files. See the :ref:`full specification <potential_options>`
   of the ``potential_options`` parameter for details.



|bgn-h3| Inferred parameters |end-h3|

Some parameters are automatically defined based on the values set for other
parameters. These may then be used when defining further parameters.
Currently, the only two such *inferred* parameters are ``h`` and ``Ων``:

* ``h`` is simply defined through
  :math:`h \equiv H_0/(100\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1})`,
  with the Hubble constant :math:`H_0` being a normal :ref:`parameter <H0>`,
  ``H0``, defined as e.g.

  .. code-block:: python3

     # Cosmology
     H0 = 67*km/(s*Mpc)

  in which case ``h`` is set equal to ``0.67``. Having ``h`` available is
  useful if you want to use the common unit of :math:`\text{Mpc}/h`, e.g. when
  defining the box size:

  .. code-block:: python3

     # Numerics
     boxsize = 256*Mpc/h

* ``Ων`` is the total density parameter :math:`\Omega_\nu` for all *massive*
  neutrino species. It is set based on the massive neutrino parameters defined
  by the ``class_params`` :ref:`parameter <class_params>`. As the computation
  of :math:`\Omega_\nu` is non-trivial, this is nice to have available for
  :ref:`simulations with massive neutrinos <massive_neutrinos>` where the sum
  :math:`\Omega_{\text{cdm}} + \Omega_\nu` is constrained. With e.g.
  :math:`\Omega_{\text{cdm}} + \Omega_\nu = 0.27`, one would set

  .. code-block:: python3

     # Cosmology
     Ωcdm = 0.27 - Ων

  in the parameter file. As ``Ων`` is non-trivial to compute from the
  user-defined parameters (``class_params``), its value is printed at the
  beginning of the CO\ *N*\ CEPT run (if running with massive neutrinos).

.. caution::

   You must not yourself explicitly define any inferred parameters within
   parameter files.



.. _dynamic_value_insertion_using_ellipses:

|bgn-h3| Dynamic value insertion using ellipses |end-h3|

Several of the parameters are ``dict``\ s with which one often wants the
values to be identical for multiple keys. Instead of typing out the same
value multiple times, this may be copied over dynamically from the placement
of ellipses '``...``' like so:

.. code-block:: python3

   # Input/output
   output_dirs = {
       'snapshot' : f'{path.output_dir}/{param}',
       'powerspec': ...,
       'bispec'   : ...,
       'render2D' : ...,
       'render3D' : ...,
   }

Here, the keys ``'powerspec'``, ``'bispec'``, ``'render2D'`` and
``'render3D'`` will all map to the same value as ``'snapshot'``. More
generally, an ellipsis is replaced by the first non-ellipsis value encountered
when looking back up the key-value definitions, wrapping around if necessary.
Thus,

.. code-block:: python3

   # Input/output
   output_times = {
       'snapshot' : ...,
       'powerspec': 1,
       'bispec'   : ...,
       'render2D' : ...,
       'render3D' : [0.1, 0.3, 1],
   }

results in ``'snapshot'`` being mapped to ``[0.1, 0.3, 1]`` while ``'bispec'``
and ``'render2D'`` are being mapped to ``1``.

.. note::
   Another takeaway from the above example is the fact that CO\ *N*\ CEPT
   rarely is picky about the exact data types used when defining parameters.
   We see that the keys of the ``output_times`` ``dict`` may be either single
   numbers or ``list``\ s of numbers, which again may be either ``int``\ s or
   ``float``\ s. Furhtermore, the container used did not have to be
   a ``list``, but might instead have been e.g. a ``tuple`` ``(0.1, 0.3, 1)``
   or a ``set`` ``{0.1, 0.3, 1}``.



.. _checking_parameter_specifications:

|bgn-h3| Checking parameter specifications |end-h3|

With all the possible magic surrounding parameter specifications, one might
feel the need to explicitly check exactly how a given parameter ends up being
defined. For this, you can do either of:

* Run CO\ *N*\ CEPT with a printout of specific parameters (here
  ``parameter0`` and ``parameter1``) as a substitute for the usual
  :ref:`main entry point <main_entry_point>`, e.g.

  .. code-block:: bash

     ./concept --pure-python \
         [-p param] [-c cmd-param] \
         -m "from commons import *; \
             print(f'{parameter0 = }'); \
             print(f'{parameter1 = }'); \
         "

* Run CO\ *N*\ CEPT in :ref:`interactive mode <interactive>` like so:

  .. code-block:: bash

     ./concept --pure-python -i -m "from commons import *;" \
         [-p param] [-c cmd-param]

  and then check the values of the parameters on the interactive Python
  prompt:

  .. code-block:: python3

     >>> parameter0
     >>> parameter1
     >>> exit()

Here brackets ``[...]`` indicate optional command-line arguments, with
``param`` the :ref:`parameter file <parameter_file>` and ``cmd-param`` any
:ref:`command-line parameters <command_line_parameters>`.

Remember that all dimensional parameters are expressed in accordance with the
:doc:`employed unit system </parameters/units>`.



.. _components_and_selections:

|bgn-h3| Components and selections |end-h3|

In CO\ *N*\ CEPT, the different species (e.g. matter, neutrinos) within a
simulation are each represented by a *component*. Typically, the matter
component consists of some number :math:`N` of particles. All particles within
a component are identical, so that they have the same mass, softening length,
etc. Besides *particle components*, CO\ *N*\ CEPT also implements
*fluid components*. Conceptually, such components consist of a grid of cells,
where again each cell share the same properties (e.g. equation of state).
The type of a component --- either 'particles' or 'fluid' --- is called
its *representation*.

Though all components must have an assigned species, a component is not
identical to its species: A species is a particular (usually physical)
substance inhabiting particular traits. A component is a given numerical
implementation of such a species, with additional properties on top.
One such property is a unique name, which however typically is just set
equal to the name of the species. See the
:ref:`specification of initial conditions <initial_conditions>` for how
to separate the component name from its species.

.. note::
   To get some practical experience with both particle and fluid components
   of various species, check out the :doc:`tutorial </tutorial/tutorial>`,
   especially the large section going
   :doc:`beyond matter-only simulations </tutorial/beyond_matter_only>`.

Several parameters either are or contain so-called component *selections*.
These are ``dict``\ s which assign some property (``dict`` value) to the
component(s) (``dict`` key).

An example of a component selection is the ``select_softening_length``
:ref:`parameter <select_softening_length>`. Say we have two particle
components; one named ``'spam'`` of species ``'baryon'`` and one named
``'ham'`` of species ``'cold dark matter'``. We can then assign separate
softening lengths to particles within each of these components like so:

.. code-block:: python3

   select_softening_length = {
       'spam': 25*kpc,
       'ham' : 15*kpc,
   }

The component selection system is however much more flexible. Beside names,
components may also be referenced using their species:

.. code-block:: python3

   select_softening_length = {
       'baryon'          : 25*kpc,
       'cold dark matter': 15*kpc,
   }

If we wish to employ the same softening length for all components, we can use
the short-hand

.. code-block:: python3

   select_softening_length = {
       'all': 25*kpc,
   }

.. note::
   Often, properties like the softening length can be described as an
   expression common to all or multiple components, though the evaluated value
   may differ. Consider

   .. code-block:: python3

      select_softening_length = {
          'all': '0.025*boxsize/cbrt(N)',
      }

   which sets the softening length to be :math:`2.5\,\%` of the mean
   inter-particle distance within each component separately. That is,
   while ``boxsize`` is a global constant, ``N`` is understood to be an
   attribute specific to each component, here its number of particles
   :math:`N`.

We can also refer to components using their representation, like so:

.. code-block:: python3

   select_softening_length = {
       'particles': 25*kpc,
       'fluid'    : -1,
   }

which again would assign :math:`25\,\text{kpc}` as the softening length to
both ``'spam'`` and ``'ham'``, assuming these are particle components. As
fluid components do not have an associated softening length, the value set for
``'fluid'`` does not matter for this particular component selection, even in
the case where fluid components are present within the simulation.

Component selections support the
:ref:`ellipsis syntax <dynamic_value_insertion_using_ellipses>`, as in

.. code-block:: python3

   select_softening_length = {
       'spam': 25*kpc,
       'ham' : ...,
       'all' : 15*kpc,
   }

which assigns :math:`25\,\text{kpc}` as the softening length to the
``'spam'`` and ``'ham'`` components and :math:`15\,\text{kpc}` to
all others.

There is also a ``'default'`` key --- which works similar to ``'all'`` ---
but which should not be set by the user. Instead, this is set internally by
the code, intended to catch otherwise unset components within a selection.

Some component selection parameters further support component *combinations*.
An example is the ``powerspec_select`` :ref:`parameter <powerspec_select>`.
The following specifies that we want power spectrum outputs for both
``'spam'`` and ``'ham'`` individually, as well as their combined (auto)
power spectrum:

.. code-block:: python3

   powerspec_select = {
       'spam'         : True,
       'ham'          : True,
       ('spam', 'ham'): True,
   }

The order in which the components are listed within a combination has
no significance, so ``('spam', 'ham')`` is equivalent to ``('ham', 'spam')``.

.. note::
   While it would be nice to use a ``set`` ``{'spam', 'ham'}`` to express this
   lack of ordering, we cannot do so as ``set``\ s are not hashable and thus
   cannot be used as keys in a ``dict``.

Any number of components may be paired together, not just two. Finally, all
possible combinations of components may be specified as

.. code-block:: python3

   powerspec_select = {
       'all combinations': True,
   }



.. _ensuring_determinism:

|bgn-h3| Ensuring determinism |end-h3|

Reproducibility is a highly desirable feature of any complex software system.
Running the same simulation twice, we of course expect very similar outcomes
for the two runs. However, due to the non-associativity of floating-point
arithmetic, two supposedly equivalent simulations generally do not produce
exactly the same results (though any such differences ought to be much smaller
than the inherent precision of the simulations).

Below we list the parameter settings needed in order to guarantee fully
deterministic simulations, always yielding bitwise identical outputs
(snapshots included) when run several times over. Note that this does come at
a performance penalty.

* ``random_seeds``: CO\ *N*\ CEPT makes use of explicit (pseudo-)randomness
  for generating initial conditions, specifically the primordial noise. For
  the sake of reproducibility, this random noise is generated using particular
  random seeds, specified by the
  ``random_seeds`` :ref:`parameter <random_seeds>`.

  For deterministic behaviour, you should then keep the values of these seeds
  fixed between simulations, e.g.

  .. code-block:: python3

     random_seeds = {
         'primordial amplitudes': 1_000,  # keep fixed between runs
         'primordial phases'    : 2_000,  # keep fixed between runs
     }

* ``shortrange_params``: By default, the decomposition of tiles into subtiles
  is dynamically updated throughout the simulation, based on CPU timing
  measurements (see the paper on
  ':ref:`The cosmological simulation code CO𝘕CEPT 1.0<the_cosmological_simulation_code_concept_10>`'
  for details). This naturally introduces indeterminism, as timing
  measurements are affected by whatever else is going on within the system, as
  well as the real-world physical circumstances regarding the hardware. The
  subtile decomposition is controlled by the ``'subtiling'`` sub-parameter of
  the ``shortrange_params`` :ref:`parameter <shortrange_params>`.

  For deterministic behaviour, you should set some static subtile
  decomposition, e.g.

  .. code-block:: python3

     shortrange_params = {
         'subtiling': 2,
     }

* ``particle_reordering``: The order in which particles of a component appear
  in memory ultimately determines the order in which direct particle-particle
  interactions --- typically short-range gravity --- are computed,
  corresponding to a specific ordering of the sum

  .. math::

     \boldsymbol{f}_i = -G m^2 \sum_{j} \frac{\boldsymbol{x}_i - \boldsymbol{x}_j}{|\boldsymbol{x}_i - \boldsymbol{x}_j|^3}\, ,

  for the force on particle :math:`i` at :math:`\boldsymbol{x}_i` due to all
  other particles :math:`j` at :math:`\boldsymbol{x}_j`. As this sum consists
  of floating-point numbers it is not associative, and so the order matters
  for determinism. By default, the particles in memory are periodically
  re-sorted for performance reasons (see the ``particle_reordering``
  :ref:`parameter <particle_reordering>` for details).

  For deterministic behaviour, you should restrict the in-memory reordering of
  particles to be done in a deterministic fashion,

  .. code-block:: python3

     particle_reordering = 'deterministic'

  or disable the reordering entirely:

  .. code-block:: python3

     particle_reordering = False


* ``fftw_wisdom_*``: The FFTW library used for the distributed FFTs is able
  to compute these in a multitude of different ways, with the results not
  being exactly identical to machine precision, introducing non-determinism
  into the simulations. Controlling the FFTW behaviour we have the
  ``fftw_wisdom_rigor`` :ref:`parameter <fftw_wisdom_rigor>`, the
  ``fftw_wisdom_reuse`` :ref:`parameter <fftw_wisdom_reuse>` and the
  ``fftw_wisdom_share`` :ref:`parameter <fftw_wisdom_share>`.

  For deterministic behaviour, we can specify this set of parameters in a few
  different ways, depending on the circumstances.

  * If running CO\ *N*\ CEPT locally or always on a single, specific node of a
    cluster, every simulation should make use of the same FFTW plans, stored
    as
    `FFTW wisdom <https://www.fftw.org/fftw3_doc/Words-of-Wisdom_002dSaving-Plans.html>`__:

    .. code-block:: python3

       fftw_wisdom_reuse = True

    Note that this is the default.

  * If running CO\ *N*\ CEPT simulations on a cluster, possibly using a
    different set of nodes for different simulations, one should further allow
    the different nodes to share the same FFTW wisdom:

    .. code-block:: python3

       fftw_wisdom_reuse = True
       fftw_wisdom_share = True

    Note that this is *not* the default.

  * With the above solutions for determinism within FFTW, the
    :ref:`rigour level <fftw_wisdom_rigor>` does not matter. An alternative to
    the above is to always stick with the lowest-level rigour, which estimates
    a good FFTW plan without running any performance tests, always resulting
    in the same plan for the same problem:

    .. code-block:: python3

       fftw_wisdom_rigor = 'estimate'

    Note that this is *not* the default.

Besides the above parameter settings, the way in which CO\ *N*\ CEPT is run
may also break strict determinism:

* **Fixed set of parameters**: On top of using specific parameter values as
  described above, the total set of parameters used should of course be
  exactly the same across the simulations, if exactly identical outputs are
  required.
* **Number of processes**: The
  :ref:`number of processes <number_of_processes>` used to run the simulation
  directly dictates the domain decomposition and thus the order of various
  operations, again degrading the determinism due to the non-associativity of
  floating-point arithmetic. For deterministic results, always run with the
  same number of processes. How these processes are distributed across nodes
  does not matter, at least if the ``fftw_wisdom_*`` parameters are set
  according to the above.
* **Different builds**: Different systems (exact compiler version,
  operating system, etc.) may compile the CO\ *N*\ CEPT source code into
  different machine code, which can lead to different results for otherwise
  equivalent simulations, performed on different systems. As usual, such
  differences should only be observed around the level of machine precision.
  For deterministic CO\ *N*\ CEPT runs on a cluster, :ref:`build <build>` the
  code once and make use of this build for all runs.
* **Different hardware**: Running the same build of CO\ *N*\ CEPT on different
  hardware can lead to slight differences in the results. If you work on a
  cluster with nodes of e.g. different CPU architectures, you should make sure
  to only and always use nodes of identical CPU architecture, if strict
  determinism is desirable.

