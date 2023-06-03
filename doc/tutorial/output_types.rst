Other kinds of output
---------------------
So far, the result of any simulation has been power spectra, though several
other types of output are available, as exemplified by the below parameter
file:

.. code-block:: python3
   :caption: param/tutorial-5
   :name: param-output-types
   :emphasize-lines: 10, 12-14, 17, 19-21, 23-34, 46-68

   # Non-parameter variable used to control the size of the simulation
   _size = 64

   # Input/output
   initial_conditions = {
       'species': 'matter',
       'N'      : _size**3,
   }
   output_dirs = {
       'snapshot' : f'{path.output_dir}/{param}',
       'powerspec': ...,
       'bispec'   : ...,
       'render2D' : ...,
       'render3D' : ...,
   }
   output_times = {
       'snapshot' : 0.1,
       'powerspec': [a_begin, 0.3, 1],
       'bispec'   : ...,
       'render3D' : ...,
       'render2D' : logspace(log10(a_begin), log10(1), 15),
   }
   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'linear imprinted': True, 'plot': True},
   }
   bispec_select = {
       'matter': {'data': True, 'reduced': True, 'tree-level': True, 'plot': True},
   }
   render2D_select = {
       'matter': {'data': False, 'image': True, 'terminal image': True},
   }
   render3D_select = {
       'matter': {'image': True},
   }

   # Numerics
   boxsize = 128*Mpc
   potential_options = 2*_size

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.02

   # Graphics
   render2D_options = {
       'gridsize': {
           'matter': _size,
       },
       'terminal resolution': {
           'matter': 80,
       },
       'colormap': {
           'matter': 'inferno',
       },
   }
   render3D_options = {
       'color': {
           'matter': 'inferno',
       },
       'background': {
           'matter': 'black',
       },
       'resolution': {
           'matter': 640,
       },
   }

Run a simulation using the :ref:`above <param-output-types>` parameters, e.g.
by saving them to ``param/tutorial-5`` and executing

.. code-block:: bash

   ./concept \
       -p param/tutorial-5 \
       -n 4

This will take a few minutes. You may read along in the meantime.

We see that besides power spectra, we now have *snapshots*, *bispectra* and
*renders*, the latter of which comes in a 2D and a 3D version. The ellipses
(``...``) used above in e.g. ``output_dirs`` indicate that we want all kinds
of output to go to the same directory.

For the ``output_times``, different values are given for three of the output
types, while ``'bispec'`` and ``'render3D'`` are set to use the same times as
the output just above, i.e. that of ``'powerspec'``. For ``'render2D'``, we've
specified 15 outputs spaced logarithmically equidistant between
:math:`a = a_{\text{begin}} = 0.02` and :math:`a = 1`.

Among the new parameters introduced are ``powerspec_select``, in which we have
specified that we want the usual data files and plots, including the linear
theory predictions. A new kind of power spectrum output,
``'linear imprinted'``, is also specified. This is once again the linear
theory prediction, but with the random realisation noise imprinted onto it,
effectively incorporating cosmic variance. From the power spectrum plots, it
is clear that this kind of linear power spectrum is a much closer match to
the simulation power spectrum at the lower :math:`k`. For taking ratios of
non-linear and linear power spectra, such "imprinted" linear spectra are thus
to be preferred.



Bispectra
.........
The output directory will be populated with a large number of output files.
The ones with names beginning with ``bispec`` are the bispectrum files,
analogous to the by-now familiar power spectrum files. Looking at the
bispectrum plots, we see that they show both the bispectrum :math:`B` as well
as its reduced version :math:`Q`, and even the perturbative tree-level
prediction. We get all of these as they are enabled in the ``bispec_select``
parameter.

For the first bispectrum plot at :math:`a = a_{\text{begin}}`, the coloured
simulation :math:`B` line is mostly dashed, indicating negative values, which
usually means that the bispectrum signal present in the simulation is too
small compared to the noise. Indeed, the later plots show the simulation
:math:`B` as a full line, indicating positive, proper values.

The bispectrum data files are very analogous to the power spectrum files,
though the number of columns contained within them is usually much greater.

.. tip::
   One way to comfortably view wide text files (e.g.
   ``output/tutorial-5/bispec_a=1.00``) in the terminal is to use

   .. code-block:: python3

      less -S output/tutorial-5/bispec_a=1.00

   You can then scroll both horizontally and vertically with the arrow keys.
   Press ``Q`` to quit.

The specific bispectrum that has been measured is the *equilateral*
bispectrum. A :doc:`later section </tutorial/bispectra>` of this tutorial is
dedicated to the bispectrum, including other configurations.



3D renders
..........
You will find other image files in the output directory with names that begin
with ``render3D``. These are --- unsurprisingly --- the 3D renders.

In the ``render3D_select`` parameter we've specified that we want 3D renders
of the ``'matter'`` component, whereas specifics of the 3D renders are given
in ``render3D_options``. Here we've set the colour of the matter particles to
be ``'inferno'``, which Matplotlib recognises as a colormap. This colormap
will be used to map colours to regions of the box depending on the local
density.

.. tip::
   The available colormaps can be viewed
   `here <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`__.
   A single colour value may also be given, either as an RGB tuple (each value
   ranging from 0 to 1) or as a
   `named colour <https://matplotlib.org/stable/gallery/color/named_colors.html>`__.

We also specify the background colour, as well as the resolution (height and
width) of the image, in pixels.



2D renders
..........
The 2D renders show the particle configuration projected along one of the axes
of the box.

In the ``render2D_select`` parameter we've specified that we want images as
well as terminal images, but no data. Here, *images* refer to the 2D render
image files you see in the output directory. *Terminal images* are rendered
directly in the terminal as part of the printed output, as you have probably
noticed. If you turn on the *data* output, the 2D render data will further be
dumped to HDF5 files.

The options for the 2D renders are collected in the ``render2D_options``
parameter. Here ``gridsize`` sets the resolution of the cubic grid onto which
the particles are interpolated in order to produce the render. The height and
width of the image files (in pixels) are then equal to ``gridsize``. The
terminal image is produced by resizing the interpolation grid to match the
resolution given in ``'terminal resolution'``. The terminal image is then
displayed using one character slot per grid cell, i.e. the terminal render
will be ``'terminal resolution'`` (80) characters wide. Since the terminal
render is constructed from the original 2D render, this does not show more
detail even though the resolution is higher (80 vs. 64).

Also available through ``render2D_options`` is the colormap to use.



.. raw:: html

   <h3>The play utility</h3>

For this next trick, the simulation needs to have finished, and we need to
know its job ID.

.. tip::
   To grab the job ID from a power spectrum (or bispectrum) data file, you can
   do e.g.

   .. code-block:: bash

      grep -o "job [0-9]*" output/tutorial-5/powerspec_a=1.00

With the job ID at hand, try the following:

.. code-block:: bash

   ./concept -u play <ID>  # replace <ID> with job ID number

You should see a nice animation of the evolution of the large-scale structure,
playing out right in the terminal! The animation is produced from the terminal
images stored in the log file ``job/<ID>/log``.

The ``-u`` option to the ``concept`` script signals CO\ *N*\ CEPT to start up
a *utility* rather than running a simulation. These utilities are handy (and
in this case goofy) side programs baked into CO\ *N*\ CEPT. You are encouraged
to play around with the options to the play utility, a brief overview of which
gets printed by

.. code-block:: bash

   ./concept -u play -h

Another such utility --- the *info utility* --- is encountered just below,
and we will encounter others in later sections of the tutorial. To see the
available utilities, do .e.g

.. code-block:: bash

   ls util

For focused documentation on the individual utilities,
consult :doc:`Utilities </utilities/utilities>`.



Snapshots
.........
Snapshots are raw dumps of the simulated system, in this case the positions
and momenta of all :math:`N = 64^3` particles. By default, CO\ *N*\ CEPT uses
its own snapshot format, which is simply a well-structured HDF5 file.

.. tip::
   For a great graphical tool to explore HDF5 files in general, check out
   `ViTables <https://vitables.org/>`__. If you encounter problems viewing
   HDF5 files produced by CO\ *N*\ CEPT, check that you are using ViTables 3.

   You can install ViTables as part of the CO\ *N*\ CEPT installation via

   .. code-block:: bash

      (source concept && $python -m pip install vitables)

   after which you can run it by e.g. doing

   .. code-block:: bash

      (source concept && $python -m vitables &)

   If ViTables is unable to start, it might be due to Qt 5 not being installed
   on your system. If you're on a Debian/Ubuntu-like system (and have root
   privileges), you can install Qt 5 by

   .. code-block:: bash

      sudo apt install qtbase5-dev

   It might also help to install the master version rather than the latest
   release:

   .. code-block:: bash

      (source concept && $python -m pip uninstall -y vitables)
      (source concept && $python -m pip install git+https://github.com/uvemas/ViTables.git)

Such snapshots are useful if you want to process the raw data using some
external program. You can also initialise a simulation from a snapshot, instead
of generating initial conditions from scratch. To try this, redefine the
initial conditions to simply be the path to the snapshot produced by the
simulation you just ran:

.. code-block:: python3

   initial_conditions = f'{path.output_dir}/{param}/snapshot_a=0.10.hdf5'

Also, you should change ``a_begin`` to be ``0.1`` as to comply with the time at
which the snapshot was dumped. Finally, before rerunning the simulation
starting from the snapshot, you should probably comment out at least the
``'render2D'`` ``output_times``, as to not clutter up the output directory too
heavily.

If you forget to correct ``a_begin``, a warning will be emitted. The same goes
for other obvious inconsistencies between the parameter file and the snapshot,
like if ``boxsize`` or ``Ωcdm`` is wrong. To be able to do this, some meta data
about the cosmology and numerical setup is stored in the snapshot as well.

.. note::
   Warnings are shown in red bold text, easily distinguishable from the main
   output text. As warnings (as opposed to errors) results from non-fatal
   issues, CO\ *N*\ CEPT will continue running. A warning emitted during the
   simulation may hint that something has gone wrong, meaning that the results
   perhaps should not be trusted. To make sure that no warnings go unnoticed,
   CO\ *N*\ CEPT will notify you at the end of the simulation. A separate error
   log, ``job/<ID>/log_err``, containing just warnings and error messages,
   will also be present.

To generate a warning and error log file, try wrongly specifying e.g.
``Ωb = 0.05``. Once the simulation has completed, check out the error
log file.

If you intend to run many simulations using the same initial conditions,
it might be worthwhile to initialise these from a common snapshot. To produce
such an initial snapshot, simply set ``output_times = {'snapshot': a_begin}``,
in which case CO\ *N*\ CEPT will exit right after the snapshot has been dumped
at the initial time, without doing any simulation. An ``ic`` directory should
already exist within your CO\ *N*\ CEPT installation, the purpose of which is
to hold such initial condition snapshots. To dump snapshots to this directory,
set the ``'snapshot'`` entry in ``output_dirs`` to ``path.ic_dir``. We can
achieve both without even altering the parameter file:

.. code-block:: python3

   ./concept \
       -p param/tutorial-5 \
       -c "output_times = {'snapshot': a_begin}" \
       -c "output_dirs = {'snapshot': path.ic_dir}" \
       -n 4

You may also want to use CO\ *N*\ CEPT purely as an initial condition
generator, and perform the actual simulation using some other code. If so, the
default CO\ *N*\ CEPT snapshot format is of little use. To this end,
CO\ *N*\ CEPT also supports the binary Fortran format of
`GADGET <https://wwwmpa.mpa-garching.mpg.de/gadget/>`__, which is understood
by many other simulation codes and tools. To use this snapshot format in place
of the CO\ *N*\ CEPT format, add ``snapshot_type = 'gadget'`` to your
parameter file.



.. raw:: html

   <h3>The info utility</h3>

We mentioned `ViTables <https://vitables.org/>`__ as a great way to peek inside
the default CO\ *N*\ CEPT (HDF5) snapshots. It would be nice to have a general
tool which worked for the supported GADGET snapshots as well. Luckily,
CO\ *N*\ CEPT comes with just such a tool: the *info utility*. To try it out,
simply do

.. code-block:: bash

   ./concept -u info output/tutorial-5

or

.. code-block:: bash

   ./concept -u info ic

The content of all snapshots --- CO\ *N*\ CEPT (HDF5) or GADGET format --- in
the given directory will now be printed to the screen. Should you want
information about just a specific snapshot, simply provide its entire path.

