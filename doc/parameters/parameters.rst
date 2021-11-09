Parameters
==========
The following sections list the various parameters accepted by
CO\ *N*\ CEPT, along with explanations, default and example values. Each
section deals with a specific category of parameters.

To learn how to *use* parameters and parameter files, see the
:doc:`tutorial </tutorial/parameter_files>` as well as the ``-p`` and ``-c``
:doc:`command-line options </command_line_options>`.

Parameters are specified as live Python 3 variables. As such, it is helpful to
be familiar with basic Python syntax and knowledge of
`data types <https://docs.python.org/3/library/stdtypes.html>`_, such as
``str``\ ings, ``list``\ s and ``dict``\ ionaries.

Below you will find the parameter categories, corresponding to the sections
ahead. Besides bringing some organization to the large set of parameters,
these categories have no meaning.

.. toctree::
   :maxdepth: 1

   input_output
   numerical_parameters
   cosmology
   physics
   simulation_options
   graphics
   system_of_units
   debugging_options

The rest of this page contains helpful information regarding parameter
specification generally, independent of the specific parameters.



.. raw:: html

   <h3>Units</h3>

For dimensional parameters, it is important to explicitly tack the unit onto
the value, as no default units are ever assumed by CO\ *N*\ CEPT. For example,
you may define the ``boxsize`` as

.. code-block:: python3

   boxsize = 512*Mpc

but you should never leave out the unit,

.. code-block:: python3

   boxsize = 512  # Wrong!

A vast set of units are understood, like ``km``, ``pc``, ``kpc``, ``Mpc``,
``Gpc``, ``s``, ``yr``, to name a few.

.. note::
   The actual numerical value resulting from e.g. ``512*Mpc`` depends on the
   internal unit system employed by CO\ *N*\ CEPT. The unit system itself is
   not statically defined, but specified on a per-simulation basis via
   :doc:`parameters <system_of_units>`. Though the vast majority of users
   have no need to manually specify the internal unit system, it is useful
   for e.g. testing against other simulation codes that use some other,
   fixed system of units.



.. raw:: html

   <h3>Parameter files as Python scripts</h3>

You may think of parameter files as Python scripts, and use any valid Python 3
statements to formulate or compute your parameters, including ``print()`` for
debugging. Even without any explicit ``import`` statement present in the
parameter file, all of ``numpy`` is readily available. As an example, consider
the following specification of 7 power spectra output placed logarithmically
equidistant in scale factor :math:`a`, from :math:`a = 0.01` to :math:`a = 1`:

.. code-block:: python3

   output_times = {'powerspec': logspace(log10(0.01), log10(1), 7)}



.. raw:: html

   <h4>The <code class="docutils literal notranslate">path</code> and <code class="docutils literal notranslate">param</code> objects</h4>

Also readily available for use in parameter files is the convenient ``path``
object, which holds absolute paths to various useful directories and files.
For example, ``path.output_dir`` evaluates to the output directory, which by
default is ``output``. If you prefer, you may instead use the ``dict``-like
syntax ``path['output_dir']``. The contents of the ``path`` variable is
supplied by the ``.path`` file.

Another convenience object available within parameter files is ``param``,
which evaluates to the name of the parameter file in which it is used. As an
example, consider this nice way of ensuring that the power spectrum output
gets dumped in a subdirectory of the output directory, labelled according to
the parameter file:

.. code-block:: python3

   output_dirs = {'powerspec': f'{path.output_dir}/{param}'}

If used within the parameter file ``param/my_param``, ``f'{path.output_dir}/{param}'`` then evaluates to ``'output/my_param'``.

The ``param`` object is not a plain ``str`` though:

* ``param.name``: Evaluates to the name of the parameter file (equivalent to
  to using ``param`` on its own within a ``str``-context).
* ``param.path``: Evaluates to the absolute path of the parameter file.
* ``param.dir``: Evaluates to the absolute path to the directory containing
  the parameter file.



.. raw:: html

   <h3>Parameter files as <em>glorified</em> Python scripts</h3>

Beyond being equipped with convenience variables, the parameter files
themselves are additionally much more powerful than regular Python scripts.
The additional super powers are described below.



.. raw:: html

   <h4>Non-linear parsing of parameter file content</h4>

Parameters may be defined in terms of each other. Unlike with regular scripts,
the definition order does not matter (i.e. you may reference a variable
*before* it is defined). The magic that makes this work is cyclic execution of
the parameter file, so that a variable that is needed on some line but only
defined by some later line, is correctly set at the second execution. This
non-linear variable dependence may continue arbitrarily, requiring many more
execution passes.



.. raw:: html

   <h4>Custom non-parameter variables</h4>

Using custom non-parameter variables in parameter files can help provide
better organisation. For example, you may want to define both the ``boxsize``
and the resolution of the gravitational potential grid in terms of a common
variable:

.. code-block:: python3

   # Non-parameter helper variable used to control the size of the simulation
   _size = 256

   # Numerical parameters
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
   non-specified sub-parameters being auto-assigned), simplicying the writing
   of parameter files. For the :ref:`full specification <potential_options>`
   of the ``potential_options`` parameter for details.



.. raw:: html

   <h4>Dynamic value insertion using ellipses</h4>

Several of the parameters are ``dict``'s with which one often want the
values to be identical for multiple keys. Instead of typing out the same
value multiple times, this may be inferred dynamically from the placement
of ellipses '``...``' like so:

.. code-block:: python3

   output_dirs = {
       'snapshot' : f'{path.output_dir}/{param}',
       'powerspec': ...,
       'render2D' : ...,
       'render3D' : ...,
   }

Here, the keys ``'powerspec'``, ``'render2D'`` and ``'render3D'`` will map
to the same value as ``'snapshot'``. More generally, an ellipsis is replaced
by the first non-ellipsis value encountered when looking back up the key-value
definitions, wrapping around if necessary. Thus,

.. code-block:: python3

   output_times = {
       'snapshot' : ...,
       'powerspec': 1,
       'render2D' : ...,
       'render3D' : [0.1, 0.3, 1],
   }

results in ``'snapshot'`` being mapped to ``[0.1, 0.3, 1]`` and ``'render2D'``
to ``1``.

.. note::
   Another takeaway from the above example is the fact that CO\ *N*\ CEPT
   rarely is picky about the data types used when defining parameters. We see
   that the keys of the ``output_times`` ``dict`` may be either single numbers
   or ``list``\ s of numbers, which again may be either ``int``\ s or
   ``float``\ s. Furhtermore, the container used did not have been a ``list``,
   but might instead have been e.g. a ``tuple`` ``(0.1, 0.3, 1)`` or a
   ``set`` ``{0.1, 0.3, 1}``.



.. raw:: html

   <h4>Inferred parameters</h4>

Some parameters are inferred from other parameters. The value of these may be
used to define other parameters, but they should not themselves be explicitly
specified. Currently, the only two such *inferred* parameters are ``h``
and ``Ων``:

* ``h`` is simply defined through
  :math:`h \equiv H_0/(100\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1})`,
  with the Hubble constant :math:`H_0` being a normal :ref:`parameter <H0>`,
  ``H0``, defined as e.g.

  .. code-block:: python3

     H0 = 67*km/(s*Mpc)

  in which case ``h`` is set equal to ``0.67``. Having ``h`` available is
  useful if you want to use the common unit of :math:`\text{Mpc}/h`, e.g. when
  defining the box size:

  .. code-block:: python3

     boxsize = 256*Mpc/h

* ``Ων`` is the total density parameter :math:`\Omega_\nu` for all *massive*
  neutrino species. It is set based on the massive neutrino parameters defined
  by the ``class_params`` :ref:`parameter <class_params>`. As the computation
  of :math:`\Omega_\nu` is non-trivial, this is nice to have available for
  :ref:`simulations with massive neutrinos <massive_neutrinos>` where the sum
  :math:`\Omega_{\text{cdm}} + \Omega_\nu` is constrained. With e.g.
  :math:`\Omega_{\text{cdm}} + \Omega_\nu = 0.27`, one would set

  .. code-block:: python3

     Ωcdm = 0.27 - Ων

in the parameter file. As ``Ων`` unlike ``h`` is non-trivial to compute from
the user-defined parameters, its value will be printed at the beginning of the
CO\ *N*\ CEPT run (when running with massive neutrinos).

