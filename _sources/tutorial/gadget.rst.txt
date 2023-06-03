Comparison with GADGET-2
------------------------
In this section we shall perform equivalent simulations with CO\ *N*\ CEPT and
the well-known `GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/>`__ code.
Due to the two codes utilising different numerical schemes, as well as the
fact that CO\ *N*\ CEPT --- unlike GADGET-2 --- is designed to be consistent
with general relativistic perturbation theory by default, some subtleties need
to be taken into account in order to perform simulations with CO\ *N*\ CEPT
that are truly equivalent to those of GADGET-2.

If you are unfamiliar with GADGET-2 or have no interest in running GADGET-2
equivalent simulations with CO\ *N*\ CEPT, you may skip this section. However,
many of the tricks to come are not exclusively related to GADGET-2.



.. raw:: html

   <h3>CONCEPT</h3>

As always, we start with a CO\ *N*\ CEPT parameter file, which you should save
as e.g. ``param/tutorial-9``:

.. code-block:: python3
   :caption: param/tutorial-9
   :name: param-gadget
   :emphasize-lines: 4, 21, 34-38, 41

   # Input/output
   if _gen:
       initial_conditions = {
           'name'   : 'GADGET halo',
           'species': 'matter',
           'N'      : _size**3,
       }
   else:
       initial_conditions = f'{path.ic_dir}/{param}_a={a_begin}'
   output_dirs = {
       'snapshot': path.ic_dir if _gen else f'{path.output_dir}/{param}',
       'powerspec': ...,
   }
   output_bases = {
      'snapshot': param if _gen else 'snapshot',
   }
   output_times = {
       'snapshot' : a_begin if _gen else '',
       'powerspec': '' if _gen else [0.1, 0.3, 1.0],
   }
   snapshot_type = 'gadget'

   # Numerics
   boxsize = 512*Mpc/h
   potential_options = 2*_size

   # Cosmology
   H0 = 67*km/(s*Mpc)
   Ωb = 0.049
   Ωcdm = 0.27
   a_begin = 0.02

   # Physics
   realization_options = {
       'gauge'     : 'synchronous',
       'back-scale': True,
   }

   # Non-parameter helper variables
   _size = 64
   _gen = False

To perform the same simulation with both CO\ *N*\ CEPT and GADGET-2, we shall
want to start from a common initial snapshot. The
:ref:`parameter file <param-gadget>` is set up to produce this initial
snapshot when ``_gen`` is ``True``. To create the snapshot then, do

.. code-block:: bash

   ./concept \
       -p param/tutorial-9 \
       -c "_gen = True"

(as always, feel free to tack on the number of processes you want
with ``-n``). The snapshot will be dumped to the ``ic`` directory.

As we have ``snapshot_type = 'gadget'``, the snapshot will be in
GADGET format. Furthermore, naming the matter component ``'GADGET halo'``
ensures that this component gets mapped to GADGET particle type 1
(conventionally used for cold dark matter, see table 3 in the
`user guide for GADGET-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__
).

Starting from this snapshot we can now perform the CO\ *N*\ CEPT simulation by
running with ``-c "_gen = False"``. As this is also the value set in the
:ref:`parameter file <param-gadget>`, we may in fact start the simulation
simply by

.. code-block:: bash

   ./concept -p param/tutorial-9

Note that the ``initial_conditions`` is set to the path of the generated
snapshot when ``_gen`` is ``False``.

The power spectrum outputs of the CO\ *N*\ CEPT simulation will be dumped to
``output/tutorial-9`` as usual.



.. raw:: html

   <h3>GADGET-2</h3>

For GADGET-2, the first thing we need is the source code itself. If you have
installed CO\ *N*\ CEPT the easy way, you already have it! Let's make a copy
just for use with this tutorial:

.. code-block:: bash

   (source concept && cp -r "${Gadget2_dir}" "${output_dir}"/tutorial-9/)

You now have the complete GADGET-2 code in ``output/tutorial-9/Gadget2``.

The ``Makefile`` of GADGET-2 needs to be set up with correct path information
for its dependencies. Furthermore, various options need to be set in order
for the GADGET-2 simulation to come to be equivalent to the CO\ *N*\ CEPT
simulation. Last but not least, we need a GADGET-2 parameter file equivalent
to the :ref:`CONCEPT parameter file <param-gadget>`. All of this can be
conveniently achieved using the *gadget utility* included with CO\ *N*\ CEPT:

.. code-block:: bash

   ./concept \
       -u gadget output/tutorial-9/Gadget2 \
       -p param/tutorial-9

The ``output/tutorial-9/Gadget2`` directory now has a properly set up
``Makefile`` and a parameter file called ``param``. The output times of the
:ref:`original parameter file <param-gadget>` have also been copied to an
``outputlist`` file, similarly placed in ``output/tutorial-9/Gadget2``.

.. note::
   The parameters specified in the GADGET-2 ``Makefile`` and ``param`` file
   include the cosmology, the grid size of the PM grid, the particle softening
   lengths and more, all set to match those used by the specific CO\ *N*\ CEPT
   simulation as specified by the :ref:`parameter file <param-gadget>`
   together with default CO\ *N*\ CEPT values.

   The GADGET-2 TreePM gravitational method is used, which is quite similar in
   essence to the P³M method used by the CO\ *N*\ CEPT simulation, the
   difference being that the tree approximates the gravitational short-range
   force by grouping particles together.

   Lastly, the values of physical constants employed by CO\ *N*\ CEPT and
   GADGET-2 differ slightly, which will make a small difference for the
   comparison. If the GADGET-2 source code you are using is the one that goes
   with the CO\ *N*\ CEPT installation, all such constants of the GADGET-2
   source code have been patched to match the values used by CO\ *N*\ CEPT.

With the various GADGET-2 files in place, let's build the code and launch
the simulation:

.. code-block:: bash

   (source concept \
       && cd output/tutorial-9/Gadget2 \
       && make clean \
       && make \
       && $mpiexec -n 4 ./Gadget2 param)

where you may wish to change the specified number of processes deployed.

As GADGET-2 is not able to produce power spectra, the results from the
simulation are raw snapshots, placed along with other files in
``output/tutorial-9/Gadget2/output``. To investigate these snapshots, we can
make use of the info utility:

.. code-block:: bash

   ./concept -u info output/tutorial-9/Gadget2/output

You should find that the snapshots indeed have the same values of the scale
factor :math:`a` as specified in the :ref:`parameter file <param-gadget>`
(perhaps with small numerical errors).

To measure the power spectra of the particle distributions contained within
the snapshots, we may make use of the *powerspec utility*:

.. code-block:: bash

   ./concept \
       -u powerspec output/tutorial-9/Gadget2/output \
       -p param/tutorial-9

(as always, you may choose to throw more processes at the task with ``-n``).



.. raw:: html

   <h3>Comparison and adjustments</h3>

You should now have CO\ *N*\ CEPT power spectra in ``output/tutorial-9`` and
corresponding GADGET-2 power spectra in ``/output/tutorial-9/Gadget2/output``.
We can of course look at each pair of plots, but for a proper comparison we
should plot the power spectra together in a single plot. You may use the
script below:

.. code-block:: python3
   :caption: output/tutorial-9/plot.py
   :name: plot-gadget

   import glob, os, re
   import numpy as np
   import matplotlib; matplotlib.use('agg')
   import matplotlib.pyplot as plt

   # Particle Nyquist frequency
   Mpc = 1
   h = 0.67
   boxsize = 512*Mpc/h
   num_particles = 64**3
   k_nyquist = 2*np.pi/boxsize*np.cbrt(num_particles)/2

   # Read in CO𝘕CEPT data
   def read(dirname):
       P = {}
       for filename in sorted(glob.glob(f'{dirname}/powerspec*'), key=os.path.getmtime):
           if filename.endswith('.png'):
               continue
           with open(filename, mode='r', encoding='utf-8') as f:
               header = f.readline()
           a = float(re.search('a = (.+)', header).group(1).rstrip('.'))
           k, P[float(f'{a:.3f}')] = np.loadtxt(filename, usecols=(0, 2), unpack=True)
       return k, P
   this_dir = os.path.dirname(os.path.realpath(__file__))
   k, P_concept = read(this_dir)

   # Read in GADGET-2 data
   k, P_gadget = read(f'{this_dir}/Gadget2/output')

   # Plot
   fig, axes = plt.subplots(2, sharex=True)
   for a, P in P_concept.items():
       axes[0].loglog(k, P, label=f'$a = {a}$ (CO$N$CEPT)')
       if a in P_gadget:
           axes[0].loglog(k, P_gadget[a], 'k--')
           axes[1].semilogx(k, 100*(P/P_gadget[a] - 1))
   for ax in axes:
       ylim = ax.get_ylim()
       ax.plot([k_nyquist]*2, ylim, ':', color='grey')
       ax.set_ylim(ylim)
   ax.text(
       k_nyquist,
       ylim[0] + 0.15*np.diff(ylim),
       r'$k_{\mathrm{Nyquist}}\rightarrow$',
       ha='right',
   )
   axes[0].set_xlim(k[0], k[-1])
   axes[1].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   axes[0].set_ylabel(r'$P\, [\mathrm{Mpc}^3]$')
   axes[1].set_ylabel(r'$P_{\mathrm{CO}N\mathrm{CEPT}}/P_{\mathrm{GADGET}} - 1\, [\%]$')
   axes[0].legend()
   axes[0].tick_params('x', direction='inout', which='both')
   axes[1].set_zorder(-1)
   fig.tight_layout()
   fig.subplots_adjust(hspace=0)
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Store the script as e.g. ``output/tutorial-9/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial-9/plot.py

The upper subplot --- with absolute power spectra --- of the generated
``output/tutorial-9/plot.png`` should show a good qualitative match between the
two codes, whereas the lower subplot --- with relative power spectra ---
should reveal a disagreement of several percent. This disagreement stems from
both physical and numerical differences between the codes, which we will now
alleviate by adjusting the CO\ *N*\ CEPT simulation.



.. raw:: html

   <h4>Cell-centred vs. cell-vertex discretisation</h4>

The initial particle distribution, stored within the generated initial
snapshot file ``ic/tutorial-9_a=0.02``, should ideally be close to homogeneous
and isotropic. We can check this qualitatively by plotting the distribution
in 3D. For this, let's use the *render3D utility*:

.. code-block:: bash

   ./concept \
       -u render3D ic/tutorial-9_a=0.02 \
       -p param/tutorial-9 \
       -c "render3D_options = { \
           'interpolation': 0, \
           'color': 'black', \
           'background': 'white', \
           'resolution': 2000}"

This produces an image file in the ``ic`` directory. Zooming in on one of the
corners of the box, it should be clear that while we do have a close to
homogeneous system, it is far from isotropic.

.. note::
   Adding the ``render3D_options`` parameter above tunes the render so that it
   more clearly shows what we're after. Indeed, a more interesting render is
   obtained by not adding this parameter.

The strong anisotropy of the particles is inherited from the grid on which
they are (pre-)initialised. This grid-like structure of the particle
distribution makes the system sensitive towards other grids used during the
simulation, e.g. the potential grid. In CO\ *N*\ CEPT, a default choice of
using 'cell-centred' grid discretisation is made, whereas GADGET-2 uses
'cell-vertex' grid discretisation. In effect this means that the potential
grids of the two codes are shifted by half a grid cell relative to each other,
in all three dimensions (see the ``cell_centered``
:ref:`parameter <cell_centered>` for more information). This detail oughtn't
matter much, but the grid structure imprinted on the particle distribution,
together with the fact that our simulations are rather small
(:math:`N = 64^3`), makes for an exaggerated effect.

We can force CO\ *N*\ CEPT to use cell-vertex discretisation by specifying

.. code-block:: python3

   cell_centered = False

Introduce this new parameter --- either by editing the parameter file or
supplying it with ``-c`` --- and rerun the CO\ *N*\ CEPT simulation. After
also updating the comparison plot, you should find that the relative error is
now much smaller, though still a few percent.

.. note::
   Though we now have better agreement, we should not be tempted to conclude
   that cell-vertex discretisation is superior to its cell-centred cousin.
   The improvement came from using the *same* discretisation.

   The observed difference in results for the two discretisation schemes would
   be smaller had the initial conditions not contained the strong
   anisotropies. CO\ *N*\ CEPT does allow for the creation of more isotropic
   initial condition by initially placing the particles on either a
   body-centered cubic (bcc) lattice or a face-centered cubic (fcc) lattice,
   rather than the standard simple cubic (sc) lattice. See the
   ``initial_conditions`` :ref:`parameter <initial_conditions>`
   for more information.

   Though undesirable when comparing against GADGET-2, CO\ *N*\ CEPT
   additionally effectively allows for simultaneous usage of both
   discretisation schemes by carrying out both of them followed by an
   *interlacing* step to combine the results. This technique effectively
   lowers the discretisation scale, reducing the numerical artefacts. For
   more information about potential interlacing, see the ``potential_options``
   :ref:`parameter <potential_options>`.



.. raw:: html

   <h4>Radiation in the background</h4>

A key difference between CO\ *N*\ CEPT and GADGET-2 is that the former
--- unlike the latter --- tries to incorporate all species of a given
cosmology, as demonstrated in the
:doc:`previous section <beyond_matter_only>`. As such, CO\ *N*\ CEPT uses
a full background evolution obtained from the
`CLASS <http://class-code.net/>`__ code. By default, this CLASS background
includes matter, :math:`\Lambda` and radiation, whereas the background of
GADGET-2 only consists of matter and :math:`\Lambda`.

One cannot remove radiation from the CLASS background, as a cosmology without
radiation is not supported by CLASS. Instead, we can deactivate the CLASS
background entirely, which will turn on an internal background solver of
CO\ *N*\ CEPT, which like that of GADGET-2 only includes matter and
:math:`\Lambda`. To do so, add

.. code-block:: python3

   enable_class_background = False

to the parameters (while also still keeping ``cell_centered = False``) and
rerun the CO\ *N*\ CEPT simulation.

Updating the comparison plot, the codes should now agree to within a percent.



.. raw:: html

   <h4>Increased short-range precision</h4>

The gravitational force of the P³M method in CO\ *N*\ CEPT and the TreePM
method of GADGET-2 is split into a long-range and a short-range part at a
scale :math:`x_{\text{s}}`, which in both CO\ *N*\ CEPT and GADGET-2 has a
default value of :math:`x_{\text{s}} = 1.25 \Delta x`, with :math:`\Delta x`
the cell size of the potential grid. The short-range force between pairs of
particles separated by a distance somewhat larger than :math:`x_{\text{s}}`
falls off exponentially with the distance. Particle pairs with a separation
:math:`|\boldsymbol{x}_i - \boldsymbol{x}_j| > x_{\text{r}}` may then be
neglected, provided :math:`x_{\text{r}} \gg x_{\text{s}}`.

In both CO\ *N*\ CEPT and GADGET-2, the range of the short-range force has a
default value of :math:`x_{\text{r}} = 4.5x_{\text{s}}`. Due to the grouping
of particles by the tree in GADGET-2, the short-range forces between some
particles separated by distances somewhat larger than :math:`x_{\text{r}}` are
being taken into account as well. We can then hope to obtain still better
agreement with GADGET-2 by increasing the value of :math:`x_{\text{r}}` in
CO\ *N*\ CEPT slightly. Try rerunning CO\ *N*\ CEPT with
:math:`x_{\text{r}} = 5.5x_{\text{s}}` with the added parameter

.. code-block:: python3

   shortrange_params = {'range': '5.5*scale'}

and updating the comparison plot. You should find that the two codes now agree
well within 1%.

For still better agreement, you may try similarly upgrading
:math:`x_{\text{r}}` to :math:`5.5x_{\text{s}}` in GADGET-2 (either by
including the updated ``shortrange_params`` when running the CO\ *N*\ CEPT
``gadget`` utility or by manually setting ``RCUT`` in
``output/tutorial-9/Gadget2/Makefile``) and then re-compiling and -running
GADGET-2 and computing the power spectra from the new snapshots. Improved
agreement between the two codes is also achieved by increasing the
simulation size.



.. raw:: html

   <h4>Final notes</h4>

CO\ *N*\ CEPT strives to be consistent with general relativistic perturbation
theory. It does so by using the full background from CLASS, and adding in
gravitational effects from linear perturbations of other species (when
:doc:`enabled <beyond_matter_only>`), ensuring that the simulation stays in
the so-called *N*-body gauge. All this fanciness is hardly needed when
running without massive neutrinos or other exotic species, which GADGET-2 does
not support.

GADGET-2 only allows for the simulation of matter (though besides the standard
cold dark matter particles, it further supports (smoothed-particle)
hydrodynamical baryons), and always uses a background containing just matter
and :math:`\Lambda`. Sticking to even the simplest ΛCDM cosmologies (which
*do* include photons), neglecting the gravitational tug from radiation is
unacceptable for precision simulations. To resolve this problem, the usual
trick used with Newtonian *N*-body codes is to generate initial conditions at
:math:`a = a_{\text{begin}}` using the results of general relativistic
perturbations theory (which also accounts for radiation) at :math:`a = 1`, but
scaled back in time using the Newtonian growth factor. Thus, only the
:math:`a = 1` results of such *back-scaled* simulations are really physical.
The ``realization_options`` :ref:`parameter <realization_options>` in the
:ref:`parameter file used for this section <param-gadget>` specifies that
back-scaling should be used when generating the initial conditions, and also
that this should be done within the synchronous gauge (as opposed to the
default *N*-body gauge), as is typically used with back-scaling.

When using back-scaling, the particle velocities are obtained from the same
(:math:`\delta`) transfer function as the positions, using a Newtonian
approximation. The velocities are thus not the actual synchronous gauge
velocities, but are in fact closer to the velocities within the *N*-body
gauge, as obtained from the :math:`\theta` transfer function and no
back-scaling. As the synchronous and *N*-body gauge :math:`\delta` transfer
functions are quite similar, the back-scaled synchronous gauge initial
conditions --- used within this section --- are then in fact very similar to
the non-back-scaled *N*-body gauge initial conditions used within
CO\ *N*\ CEPT by default. The specifications within ``realization_options`` in
the :ref:`parameter file <param-gadget>` are thus not very crucial for the
results, but mostly put there to comply with the spirit of Newtonian
*N*-body simulations and GADGET-2. We note again that the real difference
between the *N*-body gauge approach and the back-scaled, synchronous gauge
approach is to be found outside basic ΛCDM cosmologies and matter-only
simulations, where the Newtonian approximations used for the back-scaling
break down.

