Working with parameter files
----------------------------
Specifying the many parameters to the ``concept`` script via the ``-c`` option
quickly becomes tiresome. A better solution is to write all parameters in a
text file.

.. tip::
   The ``param`` directory is meant as a dedicated place to store your
   parameter files. It comes with a few example parameter files included.

Copy the content below to a file named e.g. ``param/tutorial-3``:

.. code-block:: python3
   :caption: param/tutorial-3
   :name: param-parameter-files

   # Non-parameter helper variable used to control the size of the simulation
   _size = 64

   # Input/output
   initial_conditions = {
       'species': 'matter',
       'N'      : _size**3,
   }
   output_dirs = {
       'powerspec': f'{path.output_dir}/{param}',
   }
   output_times = {
       'powerspec': [a_begin, 0.3, 1],
   }

   # Numerics
   boxsize = 4*_size*Mpc/h
   potential_options = 2*_size

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.02

To run CO\ *N*\ CEPT using this parameter file, do

.. code-block:: bash

   ./concept \
       -p param/tutorial-3 \
       -n 4

The simulation specified by the above parameters is quite similar to the
simulations of the previous section, though we've now been more explicit about
the cosmology and the size of the simulation box, and also refined the output
specifications. In doing so, we've made use of several helpful tricks:

* The comments / section headings are there for organisational purposes only.
  These may be omitted, and the order in which the parameters are specified
  does not matter.

* Parameter files are glorified Python scripts, and so we may utilize the full
  Python (3) language when defining the parameters. We can also define
  helper variables that are not themselves parameters, like ``_size`` above,
  which here is used to simultaneously adjust the number of particles ``'N'``,
  the side length of the simulation box ``boxsize`` and the size of the
  potential grid via ``potential_options``. Though not strictly necessary,
  it is preferable for such helper variables to be named with a leading
  underscore ``_``, to separate them from actual parameters.

* We have explicitly specified the directory for power spectrum output in the
  ``output_dirs`` parameter. The value is constructed using the magic ``path``
  and ``param`` variables, available to all parameter files. The ``path``
  variable holds absolute paths specified in the ``.path`` file.
  If you want to take a look at this ``.path`` file, do e.g.

  .. code-block:: bash

     cat .path

  Writing ``path.output_dir`` then maps to whatever value is specified for
  ``output_dir`` in the ``.path`` file, which will be
  ``'/path/to/concept/output'``. When used as is, the ``param`` variable
  maps to the file name of the parameter file in which it is used, in this
  case ``'tutorial-3'``. In total, the power spectrum output directory then
  gets set to ``'/path/to/concept/output/tutorial-3'``. We could have gotten
  away with just writing ``'output/tutorial-3'`` out statically. Such relative
  paths in parameter files should be given with respect to the
  installation directory.

* We've specified multiple times at which to dump power spectra in
  ``output_times``. Note that we use the parameter ``a_begin`` (creating a
  power spectrum of the initial conditions), the value of which is set further
  down. Generally, the order of variable definitions --- even when one depends
  on another --- is of no importance inside parameter files.

* The simulation takes place in a cubic box, the side length of which is set
  by ``boxsize``. In the parameter specification above, we've let ``boxsize``
  be proportional to ``_size``, which has the effect of keeping the particle
  density constant when varying ``_size``: If e.g. ``_size`` is increased by
  a factor of 2, ``boxsize`` increases by a factor of 2 as well, meaning that
  the simulation volume goes up by a factor of 8, which is compensated by a
  similar increase in the number of particles by a factor :math:`2^3 = 8`.

  No default units are ever assumed by CO\ *N*\ CEPT when it reads parameters,
  and so it's critical that you explicitly tack on units on all dimensional
  parameters. For ``boxsize``, the extra fancy unit of :math:`\text{Mpc}/h` is
  used above, with
  :math:`h \equiv H_0/(100\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1})`
  inferred dynamically from the Hubble constant ``H0`` set further down.

* The parameters ``Ωb`` and ``Ωcdm`` of course set the present amounts of
  baryons and cold dark matter, respectively. Together, these otherwise
  distinct species are collectively referred to as just *matter*. Thus,
  declaring the species to be ``'matter'`` in the ``initial_conditions``
  implies that the particles will represent both the baryonic
  and cold dark matter.

The ``-p`` (parameter file) option can be mixed with the ``-c`` (command-line
parameter) option. As an example, consider leaving out the definition of
``_size`` from the parameter file and instead supplying it when running the
code:

.. code-block:: bash

   ./concept \
       -p param/tutorial-3 \
       -c "_size = 64" \
       -n 4

If you forget to specify ``_size`` --- or any other variable referenced by the
parameter file --- CO\ *N*\ CEPT will exit with an error, letting you know.



.. raw:: html

   <h3>Logged job information</h3>


For each CO\ *N*\ CEPT run (or *job*) a lot of information is logged. Each new
job gets a unique integer ID, which is stated in the beginning and end of
the run (and further included in the header of the power spectrum data files).
Each job gets its own subdirectory within the ``job`` directory, containing
at least:

* ``param``: A copy of the parameter file used for the job.

  .. note::
     When mixing ``-p`` and ``-c``, the combined parameters are what's stored
     in the copied parameter file.

* ``log``: A record of the information printed to the screen during the job.

  .. tip::
     To view the logged output of e.g. CO\ *N*\ CEPT job 1 with proper
     colouring, use

     .. code-block:: bash

        less -rf job/1/log

     Arrow keys to navigate, ``Q`` to quit.

So far we've introduced only the most essential parameters. The remaining
sections of this tutorial will introduce further parameters --- and expand on
already encountered ones --- as needed. For more focused documentation on any
specific parameter, consult :doc:`Parameters </parameters/parameters>`.

