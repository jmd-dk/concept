Beyond matter-only simulations
------------------------------
If you've followed this tutorial so far, you're now fully capable of using
CO\ *N*\ CEPT for running matter-only simulations. More exotic simulations ---
taking additional species into account --- are also possible, as this section
will demonstrate.

To assess the effects resulting from including non-matter species, we shall
perform a slew of simulations throughout this section. As most effects are
small, we shall also learn how to improve the precision in different
circumstances, introducing several new parameters. If you have no interest in
any of this, feel free to skip this section, as it's rather lengthy and more
technical than the previous sections of this tutorial.

Each subsection below tackles a separate species/subject. Though you might be
interested in only one of these, it's highly recommended to go through them
in order, as they build upon each other. Though each subsection deals with
a particular species in isolation, these may be mixed and matched at will when
running simulations.

.. contents::
   :local:
   :depth: 1



.. _radiation:

Radiation
.........

.. raw:: html

   <h3>Introduction</h3>

Though universes of interest may contain several different species, in
cosmological *N*-body simulations we are typically interested in following the
evolution of the matter component only, as this is the only component which is
non-linear at cosmological scales. Even so, we may wish to not neglect the
small gravitational effects on matter from the other, linear species. We can
do this by solving the given cosmology (including the matter) in linear
perturbation theory ahead of time, and then feed the linear gravitational
field from all species but matter to the *N*-body simulation, applying the
otherwise missing gravity as an external force.

CO\ *N*\ CEPT uses the `CLASS <http://class-code.net/>`__ code to solve the
equations of linear perturbation theory. For details on how the linear
perturbations are applied to the *N*-body particles during the simulation, we
refer to the paper on
':ref:`Fully relativistic treatment of light neutrinos in 𝘕-body simulations <fully_relativistic_treatment_of_light_neutrinos_in_nbody_simulations>`'.

We begin our exploration by performing a standard matter-only simulation, as
specified by the below parameter file:

.. code-block:: python3
   :caption: param/tutorial-7.1 :math:`\,` (radiation)
   :name: param-radiation
   :emphasize-lines: 12-22, 36-43, 53

   # Non-parameter helper variable used to control the size of the simulation
   _size = 192

   # Input/output
   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
   ]
   for _species in _lin.split(','):
       if not _species:
           continue
       initial_conditions.append(
           # Linear fluid component(s)
           {
               'species'        : _species,
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': f'{path.output_dir}/{param}',
   }
   output_bases = {
       'powerspec': f'powerspec{_lin=}'
           .replace(' ', '').replace('"', '').replace("'", ''),
   }
   output_times = {
       'powerspec': 2*a_begin,
   }

   # Numerics
   boxsize = 4*Gpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'pm' :   _size,
               'p3m': 2*_size,
           },
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.01

   # Non-parameter helper variable used to specify linear components.
   # Should be supplied as a command-line parameter.
   _lin = ''

As usual, save the parameters as e.g. ``param/tutorial-7.1``, then run the
simulation:

.. code-block:: bash

   ./concept -p param/tutorial-7.1

possibly with the addition of ``-n 4`` or some other number of processes.

.. note::
   The remainder of this section of the tutorial leaves out explicit mention
   of the ``-n`` option to ``concept`` invocations. Please add whatever number
   of processes you would like yourself. To always use e.g. 4 processes,
   you may set the ``CONCEPT_nprocs``
   :ref:`environment variable <number_of_processes>`:

   .. code-block:: bash

      export CONCEPT_nprocs=4

A relatively large number of particles :math:`N = 192^3` is used in order to
increase the precision of the simulation. Our goal is to investigate the
effects from radiation perturbations, which are most pronounced at very large
scales and early times --- hence the large ``boxsize`` and early output time.
The unfamiliar parameter specifications will be explained in due time.

.. note::
   Do not fret if :ref:`the above parameter file <param-radiation>` --- and
   those to come further down in this section --- looks complicated. They
   *are* more complicated than normal parameter files, because here a single
   parameter file is designed to be used for several different simulations,
   necessitating parameter definitions being dependent on various flags.



.. raw:: html

   <h3>The hunt for high precision</h3>

Investigating the resulting ``output/tutorial-7.1/powerspec_lin=_a=0.02.png``
you should see a familiar looking simulation power spectrum: decently looking
at intermediary :math:`k`, inaccurate at small :math:`k` and with obvious
numerical artefacts at large :math:`k`. As we are interested in fine details
at low :math:`k`, we need to improve the precision here. We can do so by
adding

.. code-block:: python3

   primordial_amplitude_fixed = True

to the parameter file and rerunning the simulation. This has the effect of
replacing the uncorrelated random amplitudes of the primordial noise used to
generate the initial conditions with amplitudes that are all of the same size.
With this change, ``powerspec_lin=_a=0.02.png`` should look much better at
low :math:`k` (larger :math:`k` are not very affected by this, as here many
more :math:`\boldsymbol{k}` with the same magnitude :math:`|\boldsymbol{k}|=k`
goes into producing each data point (recorded in the ``modes`` column in the
power spectrum data file), reducing errors arising due to small
number statistics).

We expect the evolution of the *N*-body particles to be completely linear at
these large scales and early times, and so we may use the difference between
the simulation and linear power spectrum as a measure for the error in the
simulation. To better see this difference, we shall make use of the below
plotting script:

.. code-block:: python3
   :caption: output/tutorial-7.1/plot.py :math:`\,` (radiation)
   :name: plot-radiation

   import glob, os, re
   import numpy as np
   import matplotlib; matplotlib.use('agg')
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   ks = {}
   P_sims = {}
   P_lins = {}
   P_lins_imprinted = {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', os.path.basename(filename))
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           exec(f'{var} = "{val}"')
       P_lin_imprinted = None
       try:
           k, P_sim, P_lin, P_lin_imprinted = np.loadtxt(
               filename, usecols=(0, 2, 3, 4), unpack=True,
           )
       except ValueError:
           k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       ks[len(k)] = k[mask]
       P_lins[len(k)] = P_lin[mask]
       if P_lin_imprinted is not None:
           P_lins_imprinted[len(k)] = P_lin_imprinted[mask]
       P_sims[len(k), lin] = P_sim[mask]
   # Plot
   fig, axes = plt.subplots(2, sharex=True)
   def darken(color):
       return 0.65*np.asarray(matplotlib.colors.ColorConverter().to_rgb(color))
   linestyles = ['-', '--', ':', '-.']
   for i, ((lenk, lin), P_sim) in enumerate(P_sims.items()):
       linestyle = linestyles[
           sum(np.allclose(line.get_ydata()[:30], P_sim[:30], 1e-3) for line in axes[0].lines)
           %len(linestyles)
       ]
       k = ks[lenk]
       P_lin = P_lins[lenk]
       axes[0].loglog(k, P_sim, linestyle, color=f'C{i}', label=f'simulation: {lin = }')
       axes[1].semilogx(k, (P_sim/P_lin - 1)*100, linestyle, color=f'C{i}')
       P_lin_imprinted = P_lins_imprinted.get(lenk)
       if P_lin_imprinted is not None:
           axes[1].semilogx(
               k, (P_sim/P_lin_imprinted - 1)*100, linestyle,
               color=darken(f'C{i}'),
           )
   lenk = max(ks)
   k = ks[lenk]
   P_lin = P_lins[lenk]
   axes[0].loglog(k, P_lin, 'k--', label='linear', linewidth=1)
   axes[1].semilogx(k, (P_lin/P_lin - 1)*100, 'k--', linewidth=1)
   k_max = 0.5*k[-1]
   axes[0].set_xlim(k[0], k_max)
   axes[0].set_ylim(0.95*min(P_lin[k < k_max]), 1.05*max(P_lin[k < k_max]))
   axes[1].set_ylim(-1, 1)
   if P_lins_imprinted:
       axes[1].plot(
           0.5, 0.5, '-',
           color='grey', label='raw linear', transform=axes[1].transAxes,
       )
       axes[1].plot(
           0.5, 0.5, '-',
           color=darken('grey'), label='imprinted linear', transform=axes[1].transAxes,
       )
       axes[1].legend()
   axes[1].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   axes[0].set_ylabel(r'$P\, [\mathrm{Mpc}^3]$')
   axes[1].set_ylabel(r'$P_{\mathrm{simulation}}/P_{\mathrm{linear}} - 1\, [\%]$')
   axes[0].legend(fontsize=9)
   axes[0].tick_params('x', direction='inout', which='both')
   axes[1].set_zorder(-1)
   fig.tight_layout()
   fig.subplots_adjust(hspace=0)
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Save the script as e.g. ``output/tutorial-7.1/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial-7.1/plot.py

This will produce ``output/tutorial-7.1/plot.png``, where the bottom panel
shows the relative error between the simulated power spectrum and that
computed using purely linear theory. They should agree to within a percent at
the lowest :math:`k`. At higher :math:`k` the agreement is worse. Though this
can be remedied by increasing the resolution of the simulation (e.g. by
increasing ``_size``), we shall not do so here, as we focus on the lower
:math:`k` only.

The power spectra outputted by the simulation are binned logarithmically in
:math:`k`. This is usually desirable, though higher precision at the lowest
:math:`k` can be achieved by leaving out this binning. The number of bins per
decade --- among many other specifics --- is controlled through the
``powerspec_options`` parameter. Disabling the binning can be achieved by
requesting an infinite number of bins per decade. Add

.. code-block:: python3

   powerspec_options = {
       'bins per decade': inf,
   }

to the parameter file, then rerun the simulation and the plotting script.
Though the simulation power spectrum generally becomes more jagged, you should
observe better agreement with linear theory at low :math:`k`.



.. raw:: html

   <h3>Including linear species</h3>

With our high-precision setup established, we are ready to start experimenting
with adding in the missing species to the simulation, hopefully leading to
better agreement with linear theory on the largest scales. To keep the clutter
within ``output/tutorial-7.1`` to a minimum, go ahead and add

.. code-block:: python3

   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'plot': False},
   }

to the parameter file before continuing.

The inhomogeneities in the CMB causes a slight gravitational tug on matter,
perturbing its evolution. To add this effect to the simulation, we need to add
a photon component. This could look like (do not change the parameter file)

.. code-block:: python3

   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
       # Linear photon fluid
       {
           'species'        : 'photon',
           'gridsize'       : _size,
           'boltzmann order': -1,
       },
   ]

We do not want to model the photons using *N*-body particles, but rather as a
collection of spatially fixed grids, storing the energy density, momentum
density, etc. This is referred to as the *fluid representation* --- as opposed
to the *particle representation* --- and is generally preferable for
linear components. To represent the photon component as a fluid, we specify
``'gridsize'`` in place of ``'N'``, where ``'gridsize'`` is the number of grid
cells along each dimension, for the cubic fluid grids. Finally, the number of
fluid quantities --- and corresponding grids --- to take into account is
implicitly specified by the *Boltzmann order*. As is
`customary <https://arxiv.org/abs/astro-ph/9506072>`__ in linear perturbation
theory, we transform the Boltzmann equation for a given species into an
infinite hierarchy of multipole moments :math:`\ell \geq 0`. We then partition
this hierarchy in two; a non-linear part :math:`\ell \leq \ell_{\text{nl}}`
and a linear part :math:`\ell > \ell_{\text{nl}}`, where
:math:`\ell_{\text{nl}}` is precisely the Boltzmann order. A few examples
shall illuminate this concept:

- **Boltzmann order 0**: The evolution equation for the lowest moment (i.e.
  the continuity equation for the energy density) is solved non-linearly
  during the simulation, while higher moments like momentum density (on which
  the energy density depends) are solved in pure linear theory.
- **Boltzmann order 1**: The evolution equations for the two lowest moments
  (i.e. the continuity equation for the energy density and the Euler equation
  for the momentum density) are solved non-linearly during the simulation,
  while higher moments like pressure and shear (on which the energy and
  momentum density depends) are solved in pure linear theory.
- **Boltzmann order -1**: None of the moments are treated non-linearly, i.e.
  this results in a purely linear component. Though the evolution of such a
  component is independent of the simulation, a purely linear component may
  still act with a force on other, non-linear components during
  the simulation.

.. note::
   Though higher Boltzmann orders are well-defined, the largest Boltzmann
   order currently implemented in CO\ *N*\ CEPT is :math:`1`.

We shall look into Boltzmann order :math:`+1` in the subsection on
:ref:`non-linear massive neutrinos <nonlinear_massive_neutrinos>`. For now we
shall keep to the purely linear case of Boltzmann order :math:`-1`.

For fluid components (whether linear or not), the only sensible gravitational
method is that of PM. Looking at :ref:`the parameter file <param-radiation>`,
we see that a grid size of ``_size`` is specified for ``'pm'``. This *must*
match the *fluid grid size*, i.e. the value specified for ``'gridsize'`` for
the fluid component in ``initial_conditions``, so that the geometry of the
grid(s) internal to the fluid component matches that of the potential grid.
Now the simulation has *two* potential grids; one for P³M self-gravity of
the matter particles (of grid size ``2*_size``) and one for PM one-way gravity
from the linear photon fluid to the matter particles (of grid size ``_size``).

.. note::
   As the PM grid size has to match the fluid grid size, you do in fact not
   need to specify the ``'pm'`` item of ``potential_options`` in
   :ref:`the parameter file <param-radiation>`. You still need to have the
   explicit specifications of ``'gridsize'``, ``'gravity'`` and ``'p3m'``,
   though. Specifying simply ``potential_options = 2*_size`` sets the P³M
   *and* PM grid size to ``2*_size``, which fails for our fluid with grid
   size ``_size``.

The parameter file has already been set up to include optional linear fluid
components, using the ``_lin`` command-line parameter. To perform a simulation
with the inclusion of linear photons, run

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.1 \
       -c "_lin = 'photon'"

.. tip::
   Note that the ``_lin`` helper variable is defined at the bottom of the
   parameter file to have an empty value (leading to no linear species being
   included). As this is placed after all actual parameters, this defines a
   default value which is used when ``_lin`` is not given as a command-line
   parameter. As ``_size`` is defined at the top, supplying ``_size`` as a
   command-line parameter will have no effect.

Now redo the plot, and the results of both the matter-only and the matter +
photon simulation should appear. The plot will show that --- sadly ---
including the photons does *not* lead to better large-scale behaviour.

CO\ *N*\ CEPT delegates all linear (and background) computations to the
`CLASS <http://class-code.net/>`__ code. Though we have specified :math:`H_0`,
:math:`\Omega_{\text{b}}` and :math:`\Omega_{\text{cdm}}` in the parameter file,
many cosmological parameters are still left unspecified. Here the default CLASS
parameters are used, which in addition to baryons, cold dark matter and photons
also contain massless neutrinos. With our hope renewed, let's run a simulation
which includes both linear photons and linear massless neutrinos:

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.1 \
       -c "_lin = 'photon, massless neutrino'"

Redoing the plot, we discover that including the neutrinos made the
disagreement between the simulation and linear theory even larger!

The gravity applied to the non-linear matter particles from the linear photon
and neutrino fluids is the *Newtonian* gravity, i.e. that which results from
their energy densities. By contrast, the linear theory computation includes
full general relativistic gravity, meaning that we have still to account for
the gravitational effects due to the momentum density, pressure and shear of
the photons and neutrinos. As this part of gravity amounts to a general
relativistic correction, we shall refer to it as the *metric* contribution.
That is, we invent a new numerical species, the metric, containing the
collective non-Newtonian gravitational effects due to all physical species. As
demonstrated in the paper on
':ref:`Fully relativistic treatment of light neutrinos in 𝘕-body simulations <fully_relativistic_treatment_of_light_neutrinos_in_nbody_simulations>`',
this metric species might be numerically realised as a (fictitious) linear
energy density field, the *Newtonian* gravity from which implements exactly
the missing general relativistic corrections.

.. note::
   For the metric species to be able to supply the correct force, the entire
   simulation must be performed in a particular gauge; the *N*-body gauge.
   That is, initial conditions for non-linear species as well as linear input
   during the simulation must all be in this gauge. This is the default gauge
   employed by CO\ *N*\ CEPT, though
   :ref:`other gauges are available as well <realization_options>`. Note that
   all outputs are similarly in this gauge, including non-linear
   (CO\ *N*\ CEPT) and linear (CLASS) power spectra. Direct comparison to
   output from other *N*-body codes (which often do not define a gauge at all)
   is generally perfectly doable, as the choice of gauge only becomes aparrent
   at very large scales.

To finally run a simulation which includes the gravitational effects from
photons and neutrinos in their entirety, run

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.1 \
       -c "_lin = 'photon, massless neutrino, metric'"

Re-plotting, you should see a much better behaved simulation power spectrum at
large scales, agreeing with linear theory to well within 0.1%.



.. raw:: html

   <h3>Combining species</h3>

If you've read along in the terminal output during the simulations, you may
have noticed that the energy density, :math:`\varrho`, of each of the linear
species are realised in turn. We can save some time and memory by treating all
linear species as a single, collective component. To specify this, we would
normally write e.g.

.. code-block:: python3

   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
       # Linear fluid component
       {
           'species'        : 'photon + massless neutrino + metric',
           'gridsize'       : _size,
           'boltzmann order': -1,
       },
   ]

Using :ref:`our clever parameter file <param-radiation>` however, we may
specify this directly at the command-line using

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.1 \
       -c "_lin = 'photon + massless neutrino + metric'"

This idea of combining species is embraced fully by CO\ *N*\ CEPT. As such,
the species ``'photon + massless neutrino'`` may be collectively referred to
simply as ``'radiation'``. Thus,

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.1 \
       -c "_lin = 'radiation + metric'"

works just as well. You are encouraged to run at least one of the above and
check that you obtain the same result as before.

You are in fact already familiar with the idea of combining species, as
``'matter'`` really means ``'baryon + cold dark matter'``.

.. tip::
   When performing simulations in a cosmology without massless neutrinos,
   specifying ``'photon + massless neutrino'`` as the species of a component
   will produce an error. However, specifying ``'radiation'`` is always safe,
   as this dynamically maps to the set of all radiation species present in the
   current cosmology, whatever this may be. Similarly, ``'matter'`` is safe to
   use even in a cosmology without e.g. cold dark matter.



.. raw:: html

   <h3>Imprinted linear power spectra</h3>

With confidence in the linear corrections, let us now go back to using binned
power spectra. That is, remove the ``powerspec_options`` parameter --- which
you introduced earlier --- from the parameter file. Further add imprinted
linear power spectra to the desired output, by changing
``powerspec_select`` to

.. code-block:: python3

   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'linear imprinted': True, 'plot': False},
   }

inside the parameter file. Now perform one last simulation including the
necessary linear corrections;

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.1 \
       -c "_lin = 'radiation + metric'"

After updating the plot yet again, you should find that you are able to obtain
excellent agreement with linear theory with the binned spectra as well, when
using the "imprinted" version of the linear theory prediction.

To better disentangle the effects of the linear corrections from those of the
imprinted linear power spectrum, you may further wish to rerun the simulation
with\ *out* any linear corrections, keeping the imprinted linear output enabled.



.. _massive_neutrinos:

Massive neutrinos
.................
The previous subsection demonstrated how simulations of matter can be made to
agree extremely well with linear theory at linear scales, if we include the
gravitational contributions from the otherwise missing species, which were
treated linearly. We did this by comparing the simulated power spectrum
directly to the linear one, for the same cosmology.

With confidence in the strategy of including linear species, let's now look at
the relative difference in matter power between two separate cosmologies, with
and without the inclusion of linear species. As dividing one simulated power
spectrum by another cancels out much of the numerical noise, this time we can
obtain high accuracy without using any of the tricks from the previous
subsection.

Our aim shall be to compute the effect on the matter power spectrum caused by
neglecting the fact that neutrinos really do have mass, albeit small. If you
wish to study the underlying theory as well as the implementation in
CO\ *N*\ CEPT, we refer to the paper on
':ref:`Fully relativistic treatment of light neutrinos in 𝘕-body simulations <fully_relativistic_treatment_of_light_neutrinos_in_nbody_simulations>`'.



.. raw:: html

   <h3>Adding massive neutrinos to the background cosmology</h3>

To compute the effect on the matter power spectrum caused by neglecting the
fact that neutrinos do have some mass, we shall make use of the below
parameter file:

.. code-block:: python3
   :caption: param/tutorial-7.2 :math:`\,` (massive neutrinos)
   :name: param-massive-neutrinos
   :emphasize-lines: 50, 52-59, 63

   # Non-parameter helper variable used to control the size of the simulation
   _size = 80

   # Input/output
   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
   ]
   for _species in _lin.split(','):
       if not _species:
           continue
       initial_conditions.append(
           # Linear fluid component(s)
           {
               'species'        : _species,
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': f'{path.output_dir}/{param}',
   }
   output_bases = {
       'powerspec': f'powerspec{_mass=}{_lin=}'
           .replace(' ', '').replace('"', '').replace("'", ''),
   }
   output_times = {
       'powerspec': 1,
   }
   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'plot': False},
   }

   # Numerics
   boxsize = 1.4*Gpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'p3m': 2*_size,
           },
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27 - Ων
   a_begin = 0.01
   class_params = {
       # Disable massless neutrinos
       'N_ur': 0,
       # Add 3 massive neutrinos of equal mass
       'N_ncdm'  : 1,
       'deg_ncdm': 3,
       'm_ncdm'  : max(_mass/3, 1e-100),  # avoid exact value of 0.0
   }

   # Non-parameter helper variables which should
   # be supplied as command-line parameters.
   _mass = 0   # sum of neutrino masses in eV
   _lin  = ''  # linear species to include

You may want to save this as e.g. ``param/tutorial-7.2`` and get a simulation
going --- of course using

.. code-block:: bash

    ./concept -p param/tutorial-7.2

--- while you read on.

The new elements appearing in the parameter file are:

- The ``class_params`` :ref:`parameter <class_params>` has been added. Items
  defined within ``class_params`` are passed onto CLASS and are thus used for
  the background and linear computations. That is, ``class_params`` is used to
  change the cosmology deployed within the CO\ *N*\ CEPT simulation away from
  the default cosmology as defined by CLASS.

  As with CO\ *N*\ CEPT itself, a vast number of CLASS parameters exist. The
  best source for exploring these is probably the
  `explanatory.ini <https://github.com/lesgourg/class_public/blob/v2.7.2/explanatory.ini>`__
  example CLASS parameter file, which also lists default values.

  .. caution::
     As :math:`H_0` (``H0``), :math:`\Omega_{\text{b}}` (``Ωb``) and
     :math:`\Omega_{\text{cdm}}` (``Ωcdm``) already exists as CO\ *N*\ CEPT
     parameters, these should never be specified explicitly within
     ``class_params``.

  Of interest to us now are ``'N_ur'`` and ``'N_ncdm'``; the number of
  **u**\ ltra-\ **r**\ elativistic species (massless neutrinos) and
  **n**\ on-\ **c**\ old **d**\ ark **m**\ atter species (massive neutrinos).
  In the above parameter specifications, we switch out the default use of
  massless neutrinos with one (``'N_ncdm': 1``) 3-times degenerate
  (``'deg_ncdm': 3``) massive neutrino, which really amounts to three separate
  neutrinos but all of the same mass (``'m_ncdm'``).

- Besides ``_lin``, another command-line parameter ``_mass`` is now in play.
  This is the sum of neutrino masses :math:`\sum m_\nu`, in eV. As we have
  three neutrinos of equal mass, the neutrino mass ``'m_ncdm'`` is set to
  ``_mass/3``.

  We can in fact obtain massless neutrinos in CLASS using the 'ncdm' species
  by setting ``'m_ncdm'`` to zero. To avoid potential safeguards employed by
  CLASS, we ensure that a specified value of exactly ``0.0`` gets replaced by
  a tiny but non-zero number (here :math:`10^{-100}`).

- As you may gather from the name 'non-cold dark matter', the massive
  neutrinos behave like unrelativistic dark matter during most if not all of
  the simulation time span (unless ``_mass`` is set very low). When specifying
  the amount of dark matter in the cosmology, one may then choose to state
  :math:`\Omega_{\text{cdm}} + \Omega_\nu` instead of
  :math:`\Omega_{\text{cdm}}` alone. Since :math:`\Omega_\nu` is implicitly
  fixed by the choice of neutrino masses (and a few other parameters), this
  means that :math:`\Omega_{\text{cdm}}` can no longer be chosen freely.
  Rather, if we want the total dark matter energy density parameter to equal
  e.g. 0.27, :math:`\Omega_{\text{cdm}} + \Omega_{\nu} = 0.27`, we must
  specify ``Ωcdm = 0.27 - Ων``, as is done in the
  :ref:`above parameter file <param-massive-neutrinos>`. Just like ``h`` is
  automatically inferred from ``H0``, so is ``Ων`` automatically inferred
  from ``class_params``. As this latter inference is non-trivial, the
  resulting ``Ων`` is written to the terminal at the beginning of the
  simulation.

Once the first simulation --- with a cosmology including three "massive"
neutrinos of zero mass --- is done, run a simulation with e.g.
:math:`\sum m_\nu = 0.1\, \text{eV}`:

.. code-block:: bash

    ./concept \
       -p param/tutorial-7.2 \
       -c "_mass = 0.1"

With both simulations done, we can plot their relative power spectrum. To do
this, you should make use of the following script:

.. code-block:: python3
   :caption: output/tutorial-7.2/plot.py :math:`\,` (massive neutrinos)
   :name: plot-massive-neutrinos

   import glob, os, re
   import numpy as np
   import matplotlib; matplotlib.use('agg')
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   P_sims, P_lins = {}, {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', os.path.basename(filename))
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           try:
               exec(f'{var} = {val}')
           except Exception:
               exec(f'{var} = "{val}"')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       P_sims[mass, lin] = P_sim[mask]
       P_lins[mass     ] = P_lin[mask]
   k = k[mask]

   # Plot
   fig, ax = plt.subplots()
   for (mass, lin), P_sim in P_sims.items():
       if not mass:
           continue
       mass_nonzero = mass
       P_sim_ref = P_sims.get((0, lin))
       if P_sim_ref is not None:
           ax.semilogx(k, (P_sim/P_sim_ref - 1)*100,
               label=f'simulation: {mass = } eV, {lin = }',
           )
   ax.semilogx(k, (P_lins[mass_nonzero]/P_lins[0] - 1)*100, 'k--',
       label='linear', linewidth=1,
   )
   ax.set_xlim(k[0], k[-1])
   ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   ax.set_ylabel(
       rf'$P_{{\Sigma m_\nu = {mass_nonzero}\, \mathrm{{eV}}}}'
       rf'/P_{{\Sigma m_\nu = 0}} - 1\, [\%]$'
   )
   ax.legend(fontsize=9)
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

As usual, to run the script, save it as e.g. ``output/tutorial-7.2/plot.py``
and invoke

.. code-block:: bash

   ./concept -m output/tutorial-7.2/plot.py

The resulting ``output/tutorial-7.2/plot.png`` should show that letting the
neutrinos have mass results in a few percent suppression of the matter power
spectrum. At intermediary :math:`k` the simulation and linear relative power
spectra agree, whereas they do not for the smallest and largest :math:`k`.
In the case of large :math:`k`, you should see that the non-linear solution
forms a trough below the linear one, before rising up above it near the
largest :math:`k` shown. This is the well-known non-linear suppression dip,
the low-:math:`k` end of which marks the beginning of the non-linear regime.
We thus trust the simulated results at the high-:math:`k` end of the plot,
while we trust the linear results at the low-:math:`k` end.



.. raw:: html

   <h3>Adding gravitation from massive neutrinos</h3>

The hope is now to be able to correct the simulated relative power spectrum at
low :math:`k` by including the missing species to the simulation, without this
altering the high-:math:`k` behaviour. Besides ``'massive neutrino'``, we
should not forget about ``'photon'`` and ``'metric'``. Note that
``'massive neutrino'`` is not considered part of ``'radiation'``. We can
however just write ``'neutrino'``, as this refers to all neutrinos (massive
('ncdm') as well as massless ('ur')) present in the cosmology. To rerun both
cosmologies with all linear species included, we might call ``concept`` within
a Bash for-loop:

.. code-block:: bash

   for mass in 0 0.1; do
       ./concept \
           -p param/tutorial-7.2 \
           -c "_mass = $mass" \
           -c "_lin = 'photon + neutrino + metric'"
   done

Once completed, redo the plot. You should find that including the linear
species did indeed correct the large-scale behaviour while leaving the
small-scale behaviour intact.



.. raw:: html

   <h3>Tweaking the CLASS computation</h3>

Though better agreement with linear theory is achieved after the inclusion of
the linear species, the relative spectrum is now also more jagged. This added
noise stems from the massive neutrinos, the evolution of which is not solved
perfectly by CLASS. A large set of general and massive neutrino specific CLASS
precision parameters exist, which can remedy this problem.

Here we shall look at just one such CLASS parameter; ``'evolver'``. This sets
the ODE solver to be used by CLASS, and may be either ``0`` (Runge-Kutta
Cash-Karp) or ``1`` (ndf15, the default). For high-precision CLASS
computations used for *N*-body simulations, it is generally preferable to
switch to the Runge-Kutta solver. To do this, just add

.. code-block:: python3

   # Use the Runge-Kutta evolver
   'evolver': 0,

to the ``class_params`` in the parameter file.

With this change, rerun the two simulations with linear species included (you
may also rerun all four simulations, but the matter-only ones are hardly
affected by the change to ``'evolver'``). After re-plotting, the simulated
relative power spectrum should have been smoothed out at low :math:`k`,
showing excellent agreement with the linear prediction.



.. _dynamical_dark_energy:

Dynamical dark energy
.....................
This subsection investigates how to perform simulations where dark energy is
dynamic, specifically using the equation of state
:math:`w(a) = w_0 + (1 - a)w_a`. Beyond just changing the background
evolution, having :math:`w \neq -1` also causes perturbations within the dark
energy, leading to an additional gravitational tug. If you're interested in
the physics of dark energy perturbations as well as their implementation in
CO\ *N*\ CEPT, we refer to the paper on
':ref:`Dark energy perturbations in 𝘕-body simulations <dark_energy_perturbations_in_nbody_simulations>`'.



.. raw:: html

   <h3>Cosmological constant <span class="math notranslate nohighlight">\(\Lambda\)</span></h3>

So far, this tutorial has mentioned nothing about dark energy, but really it
has been there all along, as a cosmological constant :math:`\Lambda` affecting
the background evolution.

CO\ *N*\ CEPT assumes the universe to be flat;
:math:`\sum_\alpha \Omega_\alpha = 1`, :math:`\alpha` running over all
species. Written out in the standard cosmologies we have looked at thus far,
this looks like

.. math::

     \Omega_{\text{b}}
   + \Omega_{\text{cdm}}
   + \Omega_{\gamma}
   + \Omega_{\nu}
   + \Omega_{\Lambda}
   = 1\, ,

where :math:`\Omega_{\text{b}}` and :math:`\Omega_{\text{cdm}}` are defined
through the ``Ωb`` and ``Ωcdm`` CO\ *N*\ CEPT parameters, while
:math:`\Omega_{\gamma}` (photons) and :math:`\Omega_{\nu}` (massless and
massive neutrinos) are defined through CLASS parameters (typically,
:math:`\Omega_{\gamma}` is defined implicitly through the CMB temperature
``class_params['T_cmb']`` while :math:`\Omega_{\nu}` is defined implicitly
through the effective number of massless neutrino species
``class_params['N_ur']``, the number of massive neutrino species
``class_params['N_ncdm']`` and ``class_params['deg_ncdm']``, their masses
``class_params['m_ncdm']`` and temperatures ``class_params['T_ncdm']``). The
remaining dark energy density :math:`\Omega_{\Lambda}` is simply chosen as to
ensure a flat universe.

Though :math:`\Lambda` is present, it does not tug on the matter (or anything
else) as it remains completely homogeneous throughout time, which is why we
never need to include it as a linear species.



.. raw:: html

   <h3>Dynamical dark energy</h3>

We can set :math:`w_0` and :math:`w_a` through the CLASS parameters
``'w0_fld'`` and ``'wa_fld'``. In CLASS, the cosmological constant is
implemented as a separate species, rather than as the special case
:math:`w_0 = -1`, :math:`w_a = 0` of the dynamical dark energy species (in
CLASS called 'fld' for dark energy **fl**\ ui\ **d**). To disable the
cosmological constant, set the ``'Omega_Lambda'`` CLASS parameter to ``0``. In
total, specifying dynamical dark energy could then look like

.. code-block:: python3

   class_params = {
       # Disable cosmological constant
       'Omega_Lambda': 0,
       # Dark energy fluid parameters
       'w0_fld': -0.7,
       'wa_fld': 0,
   }

To test the effect on the matter from switching from :math:`\Lambda` to
dynamical dark energy (here :math:`w_0 = -1\, \rightarrow\, w_0 = -0.7`), we
shall make use of the following parameter file, which you should save as e.g.
``param/tutorial-7.3``:

.. code-block:: python3
   :caption: param/tutorial-7.3 :math:`\,` (dynamical dark energy)
   :name: param-dynamical-dark-energy
   :emphasize-lines: 52-60, 63-65, 69

   # Non-parameter helper variable used to control the size of the simulation
   _size = 128

   # Input/output
   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
   ]
   for _species in _lin.split(','):
       if not _species:
           continue
       initial_conditions.append(
           # Linear fluid component(s)
           {
               'species'        : _species,
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': f'{path.output_dir}/{param}',
   }
   output_bases = {
       'powerspec': f'powerspec{_de=}{_lin=}'
           .replace(' ', '').replace('"', '').replace("'", ''),
   }
   output_times = {
       'powerspec': 1,
   }
   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'plot': False},
   }

   # Numerics
   boxsize = 3*Gpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'p3m': 2*_size,
           },
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.1
   if _de != 'Lambda':
       class_params = {
           # Disable cosmological constant
           'Omega_Lambda': 0,
           # Dark energy fluid parameters
           'w0_fld' : -0.7,
           'wa_fld' : 0,
           'cs2_fld': 0.01,
       }

   # Simulation
   Δt_base_background_factor = 2
   Δt_base_nonlinear_factor  = 2
   N_rungs                   = 1

   # Non-parameter helper variables which should
   # be supplied as command-line parameters.
   _de  = 'Lambda'  # type of dark energy
   _lin = ''        # linear species to include

The parameter file is set up to use :math:`\Lambda` by default, while
dynamical dark energy is enabled by supplying ``-c "_de = 'dynamical'"``. One
can also supply ``-c "_de = 'Lambda'"`` to explicitly select :math:`\Lambda`.
Perform a simulation using both types of dark energy using

.. code-block:: bash

   for de in Lambda dynamical; do
       ./concept \
           -p param/tutorial-7.3 \
           -c "_de = '$de'"
   done

The parameter specifications ``Δt_base_background_factor = 2`` and
``Δt_base_nonlinear_factor = 2`` double the allowable time step size, while
``N_rungs = 1`` effectively disables the adaptive time-stepping. In addition,
we start the simulation rather late at ``a_begin = 0.1``, as the effects from
dark energy show up only at late times. All of this is just to speed up the
simulations, as we do not require excellent precision.

.. note::
   The (global) time step size during the simulation is limited by a set of
   conditions/limiters, each classified as either a 'background' or a
   'non-linear' condition. The maximum allowed time step size within each
   category is scaled by the ``Δt_base_background_factor`` and
   ``Δt_base_nonlinear_factor`` parameter, respectively.

.. note::
   The adaptive particle time-stepping is a feature enabled by default when
   using the P³M method, which assigns separate time step sizes to the
   different particles, allowing for small time steps in dense regions and
   large time steps in less dense regions, achieving both accuracy and
   numerical efficiency. The possible particle time step sizes are exactly the
   base time step size divided by :math:`2^n`, where
   :math:`n \in \{0, 1, 2, \dots\}` is referred to as the *rung*. The number
   of available rungs (and thus the minimum allowed particle time step) is
   determined through the ``N_rungs`` parameter. With ``N_rungs = 1``, all
   particles are kept fixed at rung 0, i.e. the base time step, and so no
   adaptive time-stepping takes place. The distribution of particles across
   all rungs is printed at the start of each time step, for ``N_rungs > 1``.

To make the usual plot of the relative power spectrum --- this time comparing
the matter spectrum within a cosmology with a cosmological constant
(:math:`w = -1`) to one with dynamical dark energy (here :math:`w = -0.7`) ---
we shall make use of the following plotting script:

.. code-block:: python3
   :caption: output/tutorial-7.3/plot.py :math:`\,` (dynamical dark energy)
   :name: plot-dynamical-dark-energy

   import glob, os, re
   import numpy as np
   import matplotlib; matplotlib.use('agg')
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   P_sims, P_lins = {}, {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', os.path.basename(filename))
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           exec(f'{var} = "{val}"')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       P_sims[de, lin] = P_sim[mask]
       P_lins[de     ] = P_lin[mask]
   k = k[mask]

   # Plot
   fig, ax = plt.subplots()
   linestyles = ['-', '--', ':', '-.']
   for (de, lin), P_sim in P_sims.items():
       if de == 'Lambda':
           continue
       lin_ref = lin
       for lin_ignore in ('darkenergy', 'fld'):
           lin_ref = (lin_ref.replace(lin_ignore, '')
               .replace('++', '+').replace(',,', ',')
               .replace('+,', ',').replace(',+', ',')
               .strip('+,')
           )
       P_sim_ref = P_sims.get(('Lambda', lin_ref))
       label = f'simulation: {lin = }'
       if P_sim_ref is None:
           P_sim_ref = P_sims['Lambda', '']
           label += ' (dynamical sim only)'
       P_rel = (P_sim/P_sim_ref - 1)*100
       linestyle = linestyles[
           sum(np.allclose(line.get_ydata(), P_rel, 5e-3) for line in ax.lines)
           %len(linestyles)
       ]
       ax.semilogx(k, P_rel, linestyle, label=label)
   ax.semilogx(k, (P_lins['dynamical']/P_lins['Lambda'] - 1)*100, 'k--',
       label='linear', linewidth=1,
   )
   ax.set_xlim(k[0], k[-1])
   ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   ax.set_ylabel(r'$P_{\mathrm{dynamical}}/P_{\Lambda} - 1\, [\%]$')
   ax.legend(fontsize=10)
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Save this to e.g. ``output/tutorial-7.3/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial-7.3/plot.py

The generated plot should show that the matter power is reduced quite a bit
when switching to using the dynamical dark energy. At large :math:`k`, we see
the usual non-linear suppression dip. At low/linear :math:`k`, the power
suppression is larger in the simulation power spectrum than in the linear one.
This is due to inhomogeneities forming in the dark energy species itself, the
tug on matter we have not incorporated into the simulation. This effect is
enlarged as we :ref:`have specified <param-dynamical-dark-energy>` a low dark
energy sound speed ``'cs2_fld'`` (given in units of the speed of light
squared, :math:`c^2`).



.. raw:: html

   <h3>Adding gravitation from dark energy perturbations</h3>

As usual, the missing gravity can be incorporated into the simulation by
including the missing species in the simulation as a linear component. The
parameter file has once again been set up to be able to do this via the
``_lin`` command-line parameter. To run both cosmologies again, this time
including all linear species, do e.g.

.. code-block:: bash

   for de in Lambda dynamical; do
       lin="radiation + metric"
       [ $de == dynamical ] && lin+=" + dark energy"
       ./concept \
           -p param/tutorial-7.3 \
           -c "_de = '$de'" \
           -c "_lin = '$lin'"
   done

where 'radiation' includes the photons and massless neutrinos supplied by
CLASS by default.

Notice that we do not include 'dark energy' when running with :math:`\Lambda`,
as here there are no dark energy perturbations.

After re-plotting, you should see that the simulation spectrum now matches
the linear prediction at low :math:`k`.

Though including all species --- i.e. also photons and neutrinos --- is what
should be done for production runs, it can be educational to run with fewer
linear species, to separate out their individual effects on the matter
spectrum. Besides being small, the effects from radiation perturbations should
be very close to identical between the two cosmologies. We thus expect effects
from radiation to be completely negligible for the relative power
spectrum. To test this, perform a simulation with dynamical dark energy,
including only dark energy as a linear species:

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.3 \
       -c "_de = 'dynamical'" \
       -c "_lin = 'dark energy'"

If you now redo the plot, a relative spectrum between the newly run simulation
and the :math:`\Lambda` simulation without any linear species will be added.
You will see that though it's close, it has a bit too much power suppression
at low :math:`k`.

The missing power is not due to the missing radiation, but rather the missing
gravity from the rather large pressure perturbations in the dark energy, which
is accounted for by the fictitious metric species. Performing a simulation
including the metric as well,

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.3 \
       -c "_de = 'dynamical'" \
       -c "_lin = 'dark energy + metric'"

and re-plotting, we see that we indeed achieve nearly exactly the same result
as when running with radiation.

.. note::
   As the metric always contains the total gravitational contribution from
   momentum, pressure and shear perturbations of all species, it is not
   possible to completely separate out the gravitational effects from each
   species using this set-up. For example, the last simulation above *does*
   include some photon and neutrino gravity, since the metric still contains
   contributions from their momentum, pressure and shear perturbations. The
   :math:`\Lambda` simulation with which it is paired up for the plot does
   not, however, as here the simulation is performed without including
   the metric.



.. _decaying_cold_dark_matter:

Decaying cold dark matter
.........................
This subsection deals with the case of :bolditalic:`d`\ *ecaying* **c**\ old
**d**\ ark **m**\ atter ('dcdm'), in particular the scenario where it decays
to some new form of massless non-interacting radiation; what we might call
**d**\ ark **r**\ adiation or **d**\ ecay **r**\ adiation ('dr'). While
the decay radiation is handled in exactly the same way as was done for other
linear species in the previous subsections, the decaying cold dark matter
itself is simulated using particles. To keep things general, we allow for
having both stable and unstable cold dark matter within the same simulation,
and so we think of dcdm as an entirely new species, separate from the usual,
stable cdm.

If you're interested in the details of the physics of decaying cold dark
matter as well as its implementation in CO\ *N*\ CEPT, we refer to the paper
on
':ref:`Fully relativistic treatment of decaying cold dark matter in 𝘕-body simulations <fully_relativistic_treatment_of_decaying_cold_dark_matter_in_nbody_simulations>`'.



.. raw:: html

   <h3>Particle-only simulations</h3>

We denote the rate of decay for the new decaying cold dark matter species by
:math:`\Gamma_{\text{dcdm}}`. We further wish to specify the amount of dcdm
today, :math:`\Omega_{\text{dcdm}}`, but as this depends highly on the decay
rate :math:`\Gamma_{\text{dcdm}}`, this is not a good parameter. Instead we
make use of :math:`\widetilde{\Omega}_{\text{dcdm}}`, which we define to be
the energy density parameter that dcdm *would* have had, had
:math:`\Gamma_{\text{dcdm}} = 0` (in which case dcdm and cdm would be
indistinguishable). Finally, rather than specifying
:math:`\Omega_{\text{cdm}}` and :math:`\widetilde{\Omega}_{\text{dcdm}}`
separately, we re-parametrise these as the *total* (stable and decaying) cold
dark matter energy density
:math:`(\Omega_{\text{cdm}} + \widetilde{\Omega}_{\text{dcdm}})`, as well as
the fraction of this which is of the decaying kind;

.. math::

   f_{\text{dcdm}} \equiv
         \frac{\widetilde{\Omega}_{\text{dcdm}}}{\Omega_{\text{cdm}}
       + \widetilde{\Omega}_{\text{dcdm}}}\, .

Below you'll find a parameter file set up to run simulations with dcdm, which
you should save as e.g. ``param/tutorial-7.4``:

.. code-block:: python3
   :caption: param/tutorial-7.4 :math:`\,` (decaying cold dark matter)
   :name: param-decaying-cold-dark-matter
   :emphasize-lines: 4-6, 13-17, 26-27, 30-35, 60, 76, 78-83, 90-94, 102-103

   # Non-parameter helper variable used to control the size of the simulation
   _size = 96

   # Non-parameter helper variables used to control the dcdm cosmology
   _Ω_cdm_plus_dcdm = 0.27  # total amount of stable and decaying cold dark matter
   _Γ = 80*km/(s*Mpc)       # decay rate

   # Input/output
   if _combine:
       initial_conditions = [
           # Non-linear (total) matter particles
           {
               'name'   : 'total matter',
               'species': (
                   'baryon + cold dark matter'
                   + (' + decaying cold dark matter' if _frac else '')
               ),
               'N'      : _size**3,
           }
       ]
   else:
       # Assume 0 < _frac < 1
       initial_conditions = [
           # Non-linear baryon and (stable) cold dark matter particles
           {
               'name'   : 'stable matter',
               'species': 'baryon + cold dark matter',
               'N'      : _size**3,
           },
           # Non-linear decaying cold dark matter particles
           {
               'name'   : 'decaying matter',
               'species': 'decaying cold dark matter',
               'N'      : _size**3,
           },
       ]
   for _species in _lin.split(','):
       if not _species:
           continue
       initial_conditions.append(
           # Linear fluid component(s)
           {
               'species'        : _species,
               'gridsize'       : _size,
               'boltzmann order': -1,
           }
       )
   output_dirs = {
       'powerspec': f'{path.output_dir}/{param}',
   }
   output_bases = {
       'powerspec': f'powerspec_{boxsize=}{_frac=}{_lin=}{_combine=}'
           .replace(' ', '').replace('"', '').replace("'", ''),
   }
   output_times = {
       'powerspec': 1,
   }
   powerspec_select = {
       'total matter': {'data': True, 'linear': True, 'plot': False},
       ('stable matter', 'decaying matter'): ...,
   }

   # Numerics
   boxsize = 1*Gpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'p3m': 2*_size,
           },
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = (1 - _frac)*_Ω_cdm_plus_dcdm
   a_begin = 0.02
   if _frac:
       class_params = {
           # Decaying cold dark matter parameters
           'Omega_ini_dcdm': _frac*_Ω_cdm_plus_dcdm,
           'Gamma_dcdm'    : _Γ/(km/(s*Mpc)),
       }

   # Physics
   select_forces = {
       'particles': {'gravity': 'p3m'},
       'fluid'    : {'gravity': 'pm'},
   }
   if 'lapse' in _lin:
       select_forces |= {
           'decaying matter': 'lapse',
           'total matter'   : 'lapse',
       }

   # Simulation
   primordial_amplitude_fixed = True

   # Non-parameter helper variables which should
   # be supplied as command-line parameters.
   _lin     = ''    # linear species to include
   _frac    = 0     # fraction of total cold dark matter which is decaying
   _combine = True  # combine decaying and stable matter into a single component?

Begin by running this without any additional command-line parameters;

.. code-block:: bash

   ./concept -p param/tutorial-7.4

which performs a standard simulation with just stable matter (baryons and cold
dark matter).

In the parameter file, the dcdm parameters :math:`\Gamma_{\text{dcdm}}`,
:math:`(\Omega_{\text{cdm}} + \widetilde{\Omega}_{\text{dcdm}})` and
:math:`f_{\text{dcdm}}` are called ``_Γ``, ``_Ω_cdm_plus_dcdm`` and
``_frac``, respectively. A rather extreme value of
:math:`\Gamma_{\text{dcdm}} = 80\, \text{km}\, \text{s}^{-1}\, \text{Mpc}^{-1}`
is used, corresponding to a dcdm mean particle lifetime
:math:`1/\Gamma_{\text{dcdm}}` comparable to the age of the universe, meaning
that the majority of the primordial dcdm population has decayed away at
:math:`a = 1`.

To run a simulation where some of the cold dark matter is decaying, say 70%,
specify ``_frac``:

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.4 \
       -c "_frac = 0.7"

This new simulation still consists of just a single particle component, now
with a species of
``'baryon + cold dark matter + decaying cold dark matter'``. The decay is
taken into effect by continuously reducing the mass of each *N*-body particle
according to the decay rate, without changing the particle velocity. As the
component now represents three fundamental species, the effective "*N*-body
decay rate" used is

.. math::

   \Gamma_{\text{b}+\text{cdm}+\text{dcdm}}(a) =
       \frac{\bar{\rho}_{\text{dcdm}}(a)\Gamma_{\text{dcdm}}}{
             \bar{\rho}_{\text{b}}(a)
           + \bar{\rho}_{\text{cdm}}(a)
           + \bar{\rho}_{\text{dcdm}}(a)
       }\, .

To plot the usual relative power spectrum, this time between a cosmology with
and without decaying cold dark matter, make use of the below script:

.. code-block:: python3
   :caption: output/tutorial-7.4/plot.py :math:`\,` (decaying cold dark matter)
   :name: plot-decaying-cold-dark-matter

   import glob, os, re
   import numpy as np
   import matplotlib; matplotlib.use('agg')
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   boxsizes = set()
   ks, P_sims, P_lins = {}, {}, {}
   for filename in sorted(glob.glob(f'{this_dir}/powerspec*'), key=os.path.getmtime):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', os.path.basename(filename))
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           try:
               exec(f'{var} = {val}')
           except Exception:
               exec(f'{var} = "{val}"')
       boxsizes.add(boxsize)
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       ks[boxsize] = k[mask]
       P_sims[boxsize, frac, lin, combine] = P_sim[mask]
       P_lins[boxsize, frac              ] = P_lin[mask]

   # Plot
   fig, ax = plt.subplots()
   label = 'linear'
   for (boxsize, frac), P_lin in P_lins.items():
       if frac == 0:
           continue
       P_lin_ref = P_lins.get((boxsize, 0))
       if P_lin_ref is None:
           continue
       k = ks[boxsize]
       ax.semilogx(k, (P_lin/P_lin_ref - 1)*100, 'k--',
           zorder=np.inf, label=label, linewidth=1,
       )
       label = None
   linestyles = ['-', '--', ':', '-.']
   colours = {}
   for (boxsize, frac, lin, combine), P_sim in P_sims.items():
       if frac == 0:
           continue
       frac_nonzero = frac
       lin_ref = lin
       for lin_ignore in (*['decayradiation', 'darkradiation', 'dr'], 'lapse'):
           lin_ref = (lin_ref.replace(lin_ignore, '')
               .replace('++', '+').replace(',,', ',')
               .replace('+,', ',').replace(',+', ',')
               .strip('+,')
           )
       P_sim_ref = P_sims.get((boxsize, 0, lin_ref, True))
       if P_sim_ref is None:
           continue
       k = ks[boxsize]
       colour, label = colours.get((lin, combine)), None
       if colour is None:
           colour = colours[lin, combine] = f'C{len(colours)%10}'
           label = f'simulation: {lin = }, {combine = }'
       P_rel = (P_sim/P_sim_ref - 1)*100
       linestyle = linestyles[
           sum(np.allclose(line.get_ydata(), P_rel, 1e-2) for line in ax.lines)
           %len(linestyles)
       ]
       if boxsize > min(boxsizes):
           ylim = ax.get_ylim()
       ax.semilogx(k, P_rel, f'{colour}{linestyle}', label=label)
       if boxsize > min(boxsizes):
           ax.set_ylim(ylim)
   xdata = np.concatenate([line.get_xdata() for line in ax.lines])
   ax.set_xlim(min(xdata), max(xdata))
   ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
   ax.set_ylabel(
       rf'$P_{{f_{{\mathrm{{dcdm}}}} = {frac_nonzero}}}'
       rf'/P_{{f_{{\mathrm{{dcdm}}}} = 0}} - 1\, [\%]$'
   )
   ax.legend(fontsize=8)
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Save this script as e.g. ``output/tutorial-7.4/plot.py`` and run it using

.. code-block:: bash

   ./concept -m output/tutorial-7.4/plot.py

The resulting ``plot.png`` should show prominently the familiar non-linear
suppression dip on top of an already substantial drop in power due to the
decayed matter.



.. raw:: html

   <h3>Decay radiation</h3>

The plot resulting from the first two simulations shows a familiar discrepancy
between the linear and non-linear result at low :math:`k`. As usual, we may
try to fix this by including the missing species as linear components during
the simulation:

.. code-block:: bash

   for frac in 0 0.7; do
       ./concept \
           -p param/tutorial-7.4 \
           -c "_lin = 'photon + neutrino + metric'" \
           -c "_frac = $frac"
   done

Once the above two simulations are complete, redo the plot. Adding the
photons, neutrinos and metric perturbations supplied about half of the missing
large-scale power needed to reach agreement with the linear prediction.

The remaining missing power should be supplied by further including the decay
radiation, of course only applicable for the dcdm simulation:

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.4 \
       -c "_lin = 'photon + neutrino + decay radiation + metric'" \
       -c "_frac = 0.7"

Re-plotting after running the above, you should now see excellent agreement
with the linear result at large scales.

Studying :ref:`the parameter file <param-decaying-cold-dark-matter>`, we see
that the ``'species'`` of the ``'total matter'`` component gets set to
``'baryon + cold dark matter'`` when ``_frac`` equals ``0`` (corresponding to
unset) and ``'baryon + cold dark matter + decaying cold dark matter'``
otherwise. (Do not worry about the case of the variable ``_combine`` being
falsy. We shall make use of this special flag later.) We are used to
``'matter'`` being an alias for ``'baryon + cold dark matter'``, but really
it functions as a stand-in for *all* matter within the given cosmology,
including decaying cold dark matter. Go ahead and replace this needlessly
complicated expression for the ``'species'`` of ``'total matter'`` in
:ref:`the parameter file <param-decaying-cold-dark-matter>` with just
``'matter'``. Likewise, ``'radiation'`` includes not just ``'photon'`` and
(massless) ``'neutrino'``, but also ``'decay radiation'``, when present. With
the aforementioned change to
:ref:`the parameter file <param-decaying-cold-dark-matter>` in place, try
rerunning both the dcdm and the reference simulation using simply

.. code-block:: bash

   for frac in 0 0.7; do
       ./concept \
           -p param/tutorial-7.4 \
           -c "_lin = 'radiation + metric'" \
           -c "_frac = $frac"
   done

Updating the plot, we see that the new simulation results are indeed identical
to the previous ones.



.. raw:: html

   <h3>Additional general relativistic effects and multiple particle components</h3>

For most work involving dcdm simulations, the story ends here. For simulations
using very large boxes, or possibly extreme values of :math:`f_{\text{dcdm}}`
and :math:`\Gamma_{\text{dcdm}}`, additional effects from general relativity
show up, which one might wish to include in the simulations.

To begin the exploration of this extreme regime, run the same dcdm and
reference simulation as before, but in a much larger box:

.. code-block:: bash

   for frac in 0 0.7; do
       ./concept \
           -p param/tutorial-7.4 \
           -c "boxsize = 30*Gpc" \
           -c "_lin = 'radiation + metric'" \
           -c "_frac = $frac"
   done

Rerunning ``plot.py``, we see that having the cold dark matter decay actually
leads to an *in*\ crease in power at very large scales. This increase in power
is replicated by the new dcdm simulation, though not quite enough to match the
linear result.

The continuous decay of all *N*-body particles happen in unison, according to
the set decay rate and the flow of cosmic time. Really though, each particle
should decay away in accordance with its own individually experienced flow of
proper time, which is affected by the local gravitational field. At linear
order, this general relativistic effect may be implemented as a correction
force applied to all decaying particles, with a strength proportional to their
decay rate. This force can be described as arising from a potential, which in
CO\ *N*\ CEPT is implemented as an energy density field from a fictitious
species --- much like the metric species --- called the *lapse* species. For
details on the physics of this lapse potential, see the paper on
':ref:`Fully relativistic treatment of decaying cold dark matter in 𝘕-body simulations <fully_relativistic_treatment_of_decaying_cold_dark_matter_in_nbody_simulations>`'.

Just like the metric species needs to be assigned to a linear fluid component
in order to exist during the simulation, so does the lapse species. Simply
appending ``'+ lapse'`` to our ``_lin`` string of linear species is no good
though, as this would include the lapse potential as part of gravity,
affecting *all* non-linear components, decaying or not (and with a force not
satisfying the required proportionality of the decay rate). Instead, what we
need is to let lapse be its own separate linear fluid component. As we've seen
before, :ref:`the parameter file <param-decaying-cold-dark-matter>` has been
set up to allow separating linear components using '``,``'. That is, to
properly include the lapse component, we should use
``_lin = 'radiation + metric, lapse'``.

We further need to assign the new lapse force to the decaying matter
component. Studying the specification of ``select_forces`` in
:ref:`the parameter file <param-decaying-cold-dark-matter>`, we see that the
lapse force is already being assigned whenever ``_lin`` contains the substring
``'lapse'``. To run the large-box dcdm simulation with the lapse force
included then, simply do

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.4 \
       -c "boxsize = 30*Gpc" \
       -c "_lin = 'radiation + metric, lapse'" \
       -c "_frac = 0.7"

Re-plotting after completion of the above run, we see that the lapse force
indeed managed to supply the necessary power boost, and only at very large
scales, as required.

.. note::
   In the region of :math:`k` where the relative power spectra from the
   simulations in the small and large box meet, we would of course like them
   to agree, whereas in fact the large-box simulations are slightly off. This
   is simply a lack of resolution and can be corrected by running larger
   simulations (i.e. increasing ``_size``).

Being perhaps overly critical, we may conclude that the lapse force in fact
overdid its job, with the spectrum from the dcdm simulation now having
slightly too much power at very large scales. This small error arises from our
choice of combining the dcdm species together with the stable matter species
into a single particle component. Doing so in fact introduces new general
relativistic correction terms into the equations of motion for the particles,
which are not incorporated into CO\ *N*\ CEPT. For the physics of these
additional correction terms, we once again refer to the paper on
':ref:`Fully relativistic treatment of decaying cold dark matter in 𝘕-body simulations <fully_relativistic_treatment_of_decaying_cold_dark_matter_in_nbody_simulations>`'.

To tackle this problem --- or at least confirm that it is indeed caused by
combining decaying and stable matter --- we may run a simulation which makes
use of two separate particle components; one for stable matter
(``'baryon + cold dark matter'``) and one for decaying matter
(``'decaying cold dark matter'``). This is done simply by listing each
particle component separately in the ``initial_conditions`` parameter in
:ref:`the parameter file <param-decaying-cold-dark-matter>`. Specifying
``_combine = False``, we see that
:ref:`our parameter file <param-decaying-cold-dark-matter>` file does exactly
this. We further want the produced power spectrum data file to contain the
combined power of the two particle components, rather than simply listing the
power spectra of each component separately. Looking at the specification of
``powerspec_select`` in
:ref:`the parameter file <param-decaying-cold-dark-matter>`, we see that
power spectra are to be produced of ``'total matter'`` and
``('stable matter', 'decaying matter')``. We are used to having these
specifications refer to our non-linear component through its species (usually
``'matter'``), but here we've chosen to refer by *name*, where each
(arbitrary) name is set as part of the component specification within
``initial_conditions``. The tuple syntax
``(<component 0>, <component 1>, ...)`` used within ``powerspec_select``
specifies the combined, total power spectrum of the listed components.

.. note::
   The same use of name referencing is also used when assigning the lapse
   force within ``select_forces``, as here we do not wish to also assign this
   force to ``'stable matter'``. It was this use of name referencing which
   earlier enabled us to reformulate the ``'species'`` of the total matter
   component, without this having any effect on the component selections
   within :ref:`the parameter file <param-decaying-cold-dark-matter>`.

As everything is already handled within
:ref:`the parameter file <param-decaying-cold-dark-matter>`, running the
two-particle-component simulation is then as simple as

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.4 \
       -c "boxsize = 30*Gpc" \
       -c "_lin = 'radiation + metric, lapse'" \
       -c "_frac = 0.7" \
       -c "_combine = False"

This of course increases the computation time drastically, as we now have
twice the number of particles and several times the number of force
evaluations. Once completed, update the plot once again. You should now see
better agreement with linear theory on very large scales, but at the cost of
noise at "small" (relative to the enormous box) scales.

The noise in the relative spectrum arise from having :math:`96^3 + 96^3`
particles (``'stable matter'`` plus ``'decaying matter'``) in the dcdm
simulation and just :math:`96^3` particles (``'total matter'``) in the
reference simulation. When (pre-)initialising two particle components
containing the same number of particles, CO\ *N*\ CEPT places the two sets of
particles on interleaved lattices, corresponding to
`the two simple lattices constituting a body-centered lattice <https://en.wikipedia.org/wiki/Cubic_crystal_system#Caesium_chloride_structure>`__.
The same body-centered cubic lattice arrangement is used for the particles
within a single component when its number of particles :math:`N = 2n^3` with
:math:`n\in\mathbb{N}`. For the final run, then, try doubling the number of
particles for the ``'total matter'`` component within the parameter file from
``_size**3`` to ``2*_size**3``, then rerun the large-box reference simulation:

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.4 \
       -c "boxsize = 30*Gpc" \
       -c "_lin = 'radiation + metric'"

After updating the plot, the noise previously seen in the relative spectrum
using separate stable and decaying components should now be gone, leaving a
spectrum that has great agreement with linear theory all the way down to where
the smaller-box spectra begins.

.. note::
   As the large-box reference simulation now contains :math:`N = 2\times 96^3`
   particles while the large-box "combined" dcdm simulations contain
   :math:`N = 96^3` particles, the corresponding relative spectra now perform
   worse than previously.



.. _nonlinear_massive_neutrinos:

Non-linear massive neutrinos
............................
Here we'll explore simulations with massive neutrinos, solved *non*-linearly
within CO\ *N*\ CEPT. You should have already completed the *linear*
:ref:`massive neutrinos <massive_neutrinos>` subsection of this tutorial
before carrying on with this subsection.



.. raw:: html

   <h3>Linear neutrinos</h3>

The goal of this subsection is to upgrade the massive neutrinos within the
simulations from being a simple linear density field to be a non-linearly
evolved fluid. For this we shall make use of the below parameter file, which
you should save as e.g. ``param/tutorial-7.5``:

.. code-block:: python3
   :caption: param/tutorial-7.5 :math:`\,` (non-linear massive neutrinos)
   :name: param-nonlinear-massive-neutrinos
   :emphasize-lines: 19-35, 75-79, 83-88, 96-97

   # Non-parameter helper variable used to control the size of the simulation
   _size = 80

   # Input/output
   initial_conditions = [
       # Non-linear matter particles
       {
           'species': 'matter',
           'N'      : _size**3,
       },
       # Linear fluid component
       {
           'name'           : 'linear',
           'species'        : 'photon + neutrino + metric',
           'gridsize'       : _size,
           'boltzmann order': -1,
       },
   ]
   if _nunonlin:
       initial_conditions += [
           # Non-linear neutrino fluid
           {
               'name'           : 'non-linear neutrino',
               'species'        : 'neutrino',
               'gridsize'       : _size,
               'boltzmann order': +1,
           },
           # Linear fluid component with neutrinos left out
           {
               'name'           : 'linear (no neutrino)',
               'species'        : 'photon + metric',
               'gridsize'       : _size,
               'boltzmann order': -1,
           },
       ]
   output_dirs = {
       'powerspec': f'{path.output_dir}/{param}',
   }
   output_bases = {
       'powerspec': f'powerspec{_mass=}{_nunonlin=}{_nutrans=}',
   }
   output_times = {
       'powerspec': 1,
   }
   powerspec_select = {
       'matter'             : {'data': True, 'linear': True, 'plot': True},
       'non-linear neutrino': ...,
   }

   # Numerics
   boxsize = 400*Mpc
   potential_options = {
       'gridsize': {
           'gravity': {
               'pm' :   _size,
               'p3m': 2*_size,
           },
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27 - Ων
   a_begin = 0.02
   class_params = {
       # Disable massless neutrinos
       'N_ur': 0,
       # Add 3 massive neutrinos of equal mass
       'N_ncdm'  : 1,
       'deg_ncdm': 3,
       'm_ncdm'  : max(_mass/3, 1e-100),  # avoid exact value of 0.0
       # General precision parameters
       'evolver': 0,
       # Neutrino precision parameters
       'l_max_ncdm'              : 200,
       'Number of momentum bins' : 50,
       'Quadrature strategy'     : 2,
       'ncdm_fluid_approximation': 3,
   }

   # Physics
   if _nunonlin:
       select_lives = {
           'linear'              : (0, _nutrans),
           'linear (no neutrino)': (_nutrans, inf),
           'non-linear neutrino' : (_nutrans, inf),
       }

   # Simulation
   primordial_amplitude_fixed = True

   # Non-parameter helper variables which should
   # be supplied as command-line parameters.
   _mass = 0          # sum of neutrino masses in eV
   _nunonlin = False  # use non-linear neutrinos?
   _nutrans = 0       # scale factor at which to transition to non-linear neutrinos

Start by running this parameter file as is,

.. code-block:: bash

   ./concept -p param/tutorial-7.5

which will perform a simulation with three mass\ *less* neutrinos, with these
as well as photons and the metric included in a combined, linear component.

Once done, also perform a simulation with three massive (but still linear)
neutrinos, say with :math:`\sum m_\nu = 0.5\, \text{eV}`.
:ref:`The parameter file <param-nonlinear-massive-neutrinos>` has been set up
so that this can be achieved by supplying ``_mass`` as a command-line
parameter:

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.5 \
       -c "_mass = 0.5"

With both the linear massless and the linear massive neutrino run done, plot
the results using the following plotting script:

.. code-block:: python3
   :caption: output/tutorial-7.5/plot.py :math:`\,` (non-linear massive neutrinos)
   :name: plot-nonlinear-massive-neutrinos

   import collections, glob, os, re
   import numpy as np
   import matplotlib; matplotlib.use('agg')
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   inf = np.inf
   P_sims, P_lins = collections.defaultdict(dict), collections.defaultdict(dict)
   for filename in glob.glob(f'{this_dir}/powerspec*'):
       matches = re.findall(r'(?=_(.*?)=(.*?)_)', os.path.basename(filename))
       if not matches or filename.endswith('.png'):
           continue
       for var, val in matches:
           exec(f'{var} = {val}')
       k, P_sim, P_lin = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
       mask = ~np.isnan(P_lin)
       k = k[mask]
       P_sims['matter'][mass, nunonlin, nutrans] = P_sim[mask]
       P_lins['matter'][mass                   ] = P_lin[mask]
       try:
           k_ν, P_sim, P_lin = np.loadtxt(filename, usecols=(4, 6, 7), unpack=True)
       except Exception:
           continue
       mask_ν = ~np.isnan(P_lin)
       k_ν = k_ν[mask_ν]
       P_sims['neutrino'][mass, nunonlin, nutrans] = P_sim[mask_ν]
       P_lins['neutrino'][mass                   ] = P_lin[mask_ν]
   for species in ['matter', 'neutrino']:
       P_sims[species] = {
           key: P_sims[species][key]
           for key in sorted(P_sims[species].keys())
       }
       P_lins[species] = {
           key: P_lins[species][key]
           for key in sorted(P_lins[species].keys())
       }

   # Plot
   fig = plt.figure()
   gs = matplotlib.gridspec.GridSpec(2, 2, figure=fig)
   ax_w  = fig.add_subplot(gs[:, 0])
   ax_ne = fig.add_subplot(gs[0, 1])
   ax_se = fig.add_subplot(gs[1, 1])
   linestyles = ['-', '--', ':', '-.']
   fmts = {}
   i = -1
   for (mass, nunonlin, nutrans), P_sim in P_sims['matter'].items():
       if not mass:
           continue
       i += 1
       for nutrans_ref in (nutrans, 0):
           P_sim_ref = P_sims['matter'].get((0, False, nutrans_ref))
           if P_sim_ref is not None:
               break
       y = (P_sim/P_sim_ref - 1)*100
       linestyle = linestyles[
           sum(np.allclose(line.get_ydata()[:30], y[:30], 3e-2) for line in ax_w.lines)
           %len(linestyles)
       ]
       fmt = f'C{i%10}{linestyle}'
       if nunonlin:
           fmts[mass, nutrans] = fmt
       label = rf'simulation: $\Sigma m_\nu = {mass}$ eV, '
       if nunonlin:
           label += rf'$a_{{\nu\mathrm{{trans}}}} = {nutrans:.2g}$'
       else:
           label += r'linear $\nu$'
       ax_w.semilogx(k, y, fmt, label=label)
   mass_nonzeros = sorted({mass for mass, *_ in P_sims['matter'].keys() if mass})
   label = 'linear'
   for mass_nonzero in mass_nonzeros:
       ax_w.semilogx(
           k, (P_lins['matter'][mass_nonzero]/P_lins['matter'][0] - 1)*100, 'k--',
           label=label, linewidth=1, zorder=-1,
       )
       label = None
   for mass_nonzero in mass_nonzeros:
       P_sim_ref = P_sims['matter'].get((mass_nonzero, False, 0))
       if P_sim_ref is None:
           continue
       for (mass, nunonlin, nutrans), P_sim in P_sims['matter'].items():
           if mass != mass_nonzero or not nunonlin:
               continue
           ax_ne.semilogx(
               k, (P_sim/P_sim_ref - 1)*100, fmts[mass, nutrans],
               label=(
                   rf'simulation: $\Sigma m_\nu = {mass}$ eV, '
                   rf'$a_{{\nu\mathrm{{trans}}}} = {nutrans:.2g}$'
               ),
           )
   for (mass, nunonlin, nutrans), P_sim in P_sims['neutrino'].items():
       ax_se.loglog(
           k_ν, k_ν**3*P_sim, fmts[mass, nutrans],
           label=(
               rf'simulation: $\Sigma m_\nu = {mass}$ eV, '
               rf'$a_{{\nu\mathrm{{trans}}}} = {nutrans:.2g}$'
           ),
       )
   ax_w .set_xlim(k[0], k[-1])
   ax_ne.set_xlim(k[0], k[-1])
   label = 'linear'
   for mass_nonzero in mass_nonzeros:
       P_lin = P_lins['neutrino'].get(mass_nonzero)
       if P_lin is None:
           continue
       ax_se.loglog(k_ν, k_ν**3*P_lin, 'k--',
           label=label, linewidth=1, zorder=-1,
       )
       label = None
       ax_se.set_xlim(k_ν[0], k_ν[-1])
   for ax in [ax_w, ax_ne, ax_se]:
       ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
       if ax.lines:
           ax.legend(fontsize=7)
   ax_w.set_ylabel(
       rf'$P_{{\Sigma m_\nu > 0}}'
       rf'/P_{{\Sigma m_\nu = 0}} - 1\, [\%]$'
   )
   ax_ne.set_ylabel(
       rf'$P_{{\mathrm{{non‐linear}}\,\nu}}'
       rf'/P_{{\mathrm{{linear}}\,\nu}} - 1\, [\%]$'
   )
   ax_se.set_ylabel(r'$k^3P_{\nu}$')
   ax_ne.set_ylim(bottom=0)
   fig.tight_layout()
   fig.savefig(f'{this_dir}/plot.png', dpi=150)

Save the plotting script to e.g. ``output/tutorial-7.5/plot.py`` and run
it using

.. code-block:: bash

   ./concept -m output/tutorial-7.5/plot.py

The resulting ``plot.png`` shows the relative matter power spectrum between
the massive and massless neutrino cosmology on the left. You should see the
usual non-linear suppression dip. The corresponding result from linear theory
is shown as well, which matches at large scales. Be careful not to confuse
completely linear theory with non-linear matter simulations containing linear
neutrinos, which again is different from the upcoming simulations treating
both matter and neutrinos non-linearly.



.. raw:: html

   <h3>Non-linear neutrinos</h3>

Studying :ref:`the parameter file <param-nonlinear-massive-neutrinos>` it
should be clear that setting ``_nunonlin = True`` will result in two
new components within the simulation, namely ``'non-linear neutrino'`` and
``'linear (no neutrino)'``, the latter of which supplies the simulation with
photons and the metric. To not double count these species, the old component
named ``'linear'`` should now no longer be used. This is handled by the newly
introduced ``select_lives`` parameter, which we shall look into shortly. What
makes the ``'non-linear neutrino'`` component non-linear is its Boltzmann
order of :math:`+1`, meaning that its energy and momentum density are treated
as non-linear fields --- evolved within CO\ *N*\ CEPT --- with pressure and
shear still treated linearly. See the :ref:`radiation <radiation>` subsection
for further explanation about the Boltzmann order. For details on how the
non-linear fluid evolution is implemented in CO\ *N*\ CEPT, we refer to the
paper
':ref:`νCO𝘕CEPT: Cosmological neutrino simulations from the non-linear Boltzmann hierarchy <nuconcept_cosmological_neutrino_simulations_from_the_nonlinear_boltzmann_hierarchy>`'.

Run a simulation with non-linear massive neutrinos with the same mass as used
for the linear massive neutrino run:

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.5 \
       -c "_mass = 0.5" \
       -c "_nunonlin = True"

Reading the output in the terminal during the simulation, it's clear that each
time step is now quite a bit more involved. Beyond having to additionally
evolve the energy and momentum density of the non-linear neutrinos --- which
in turn require continual realisation of the linear pressure (written as
``ς['trace']``) and shear (written as ``ς[i, j]``) --- the number of
gravitational interactions also increases. Below we give an overview of the
different kinds of gravitational interactions at play:

* Gravity applied to ``'matter'``:

   * From ``'matter'``:

      * P³M (long-range)
      * P³M (short-range)

   * From ``'non-linear neutrino'`` + ``'linear (no neutrino)'``:

      * PM

* Gravity applied to ``'non-linear neutrino'``:

   * From ``matter`` + ``'non-linear neutrino'`` + ``'linear (no neutrino)'``:

      * PM

All of this makes non-linear neutrino simulation slower than simulations with
linear neutrinos. Even more significant is the fact that non-linear neutrinos
require a small time step size due to the
`Courant condition <https://en.wikipedia.org/wiki/Courant-Friedrichs-Lewy_condition>`__.
As no rung-like system exists for fluids in CO\ *N*\ CEPT, the single most
extreme fluid cell typically dictates the global time step size, slowing down
the entire simulation.

Once the non-linear neutrino simulation has completed, update the plot by
rerunning the plotting script. A new line will appear in the left panel, which
shows that running with non-linear neutrinos makes the non-linear suppression
dip slightly shallower. That is, treating the neutrinos non-linearly slightly
increases the matter power. This is caused by the enhanced clustering of the
neutrinos. Exactly how much of an increase in matter power we get by
substituting linear neutrinos for non-linear neutrinos is shown in the upper
right panel. For :math:`\sum m_\nu = 0.5\, \text{eV}` it should amount to a
few per mille around the :math:`k`'s of the non-linear suppression dip. The
lower right panel shows the power spectrum of the neutrino component, where
:math:`k^3P_{\nu}` rather than just :math:`P_{\nu}` is plotted, as this
results in a better view of the data. Both the non-linearly evolved neutrinos
as well as linear ones are shown.

.. note::

   Larger neutrino masses lead to stronger neutrino clustering, in turn
   leading to further increased matter power. Though non-linear clustering in
   the neutrino component itself is clearly visible in the neutrino power
   spectrum even for small masses, the accompanying increase in matter power
   becomes very hard to see for :math:`\sum m_\nu \lesssim 0.1\, \text{eV}`.

.. caution::

   The default fluid solver used within CO\ *N*\ CEPT is that of
   `MacCormack <https://en.wikipedia.org/wiki/MacCormack_method>`__. We have
   found this method to be bad at handling strong clustering, and so our
   implementation includes a crude fix to avoid the generation of cells
   with negative densities. For large neutrino masses and/or high resolution
   this may not be sufficient, in which case the code will abort, stating that
   it gives up trying to repair the erroneous negative cells. We thus
   recommend that you only run CO\ *N*\ CEPT with non-linear neutrinos for
   :math:`\sum m_\nu \lesssim 0.6\, \text{eV}` (3 degenerate neutrinos)
   or :math:`m_\nu \lesssim 0.2\, \text{eV}` (per neutrino).



.. raw:: html

   <h3>Transitioning from linear to non-linear neutrinos</h3>

As you've now experienced first-hand, non-linear neutrino simulations can take
a long time to complete. As written above, a major reason for this is the
small global time step size required by the non-linear neutrino fluid, leading
to many more time steps than usual. This is especially true at early times and
for light neutrinos, as here the neutrino fluid has a high pressure and thus a
high speed of sound.

.. note::

   At each time step of the simulation the equation of state (EoS) parameter
   :math:`w` of the non-linear neutrino component is shown. By following the
   evolution of the EoS you can get an idea about how far the neutrino
   component is on its path towards becoming non-relativistic and hence
   matter-like, with only a small pressure.

As the non-linear neutrino component doesn't significantly deviate from linear
behaviour at early times, we can save on computational resources by running
with linear neutrinos in the beginning and then transition to using non-linear
neutrinos at some scale factor value :math:`a_{\nu\text{trans}}`.
:ref:`The parameter file <param-nonlinear-massive-neutrinos>` is set up to do
exactly this by specifying :math:`a_{\nu\text{trans}}` as the ``_nutrans``
variable. To rerun the non-linear neutrino simulation, though using linear
neutrinos for :math:`a < a_{\nu\text{trans}} = 0.1`, do

.. code-block:: bash

   ./concept \
       -p param/tutorial-7.5 \
       -c "_mass = 0.5" \
       -c "_nunonlin = True" \
       -c "_nutrans = 0.1"

The transitioning works by terminating the ``'linear'`` component (containing
neutrinos, photons and metric) at :math:`a = a_{\nu\text{trans}}`, while
simultaneously activating both the ``'linear (no neutrino)'`` (containing
photons and metric) and the ``'non-linear neutrino'`` component. This is
achieved through the ``select_lives`` parameter in the
:ref:`the parameter file <param-nonlinear-massive-neutrinos>`, which maps
components to life spans in the format
:math:`(a_{\text{activate}}, a_{\text{terminate}})`. When ``_nunonlin`` is
``False`` (default), ``select_lives`` is not set at all, and so all specified
components (here only ``'matter'`` and ``'linear'``) live for the entirety of
the simulation. With ``_nunonlin = True`` we see that ``'linear'`` is only
active until the time of ``_nutrans``, whereas ``'linear (no neutrino)'`` and
``'non-linear neutrino'`` are first activated at the time of ``_nutrans`` and
then continues to be active throughout time.

You should find that this simulation completes significantly faster than the
previous one. Once completed and with the plot updated, you should find that
running with linear neutrinos at early times doesn't change the results much.
In fact, the non-linear neutrino power spectra at :math:`a = 1` looks very
close to identical. It does make a noticeable difference at high :math:`k` for
the matter spectrum, as seen from the top right panel of the plot. However,
these (sub per mille) changes come about mostly due to the difference in
global time-stepping between the simulations, and so should not be ascribed to
early non-linearity of the neutrinos.



.. raw:: html

   <h3>Static time-stepping</h3>

To confirm that the excess matter power at high :math:`k` observed for the
simulation where the neutrino fluid is treated non-linear right from the
beginning is caused by the finer time-stepping (and as such is not directly
related to the neutrinos), let's now rerun the non-linear neutrino simulations
where we force them to use identical time-stepping. To do this, we make use of
the ``static_timestepping`` :ref:`parameter <static_timestepping>` to record
the fine time-stepping of the simulation using non-linear neutrinos from the
beginning, and to apply this time-stepping to the simulation using the linear
:math:`\rightarrow` non-linear transition:

.. code-block:: python3

   for nutrans in 0 0.1; do
       ./concept \
           -p param/tutorial-7.5 \
           -c "_mass = 0.5" \
           -c "_nunonlin = True" \
           -c "_nutrans = $nutrans" \
           -c "static_timestepping = f'{path.output_dir}/{param}/timestepping'"
   done

After updating the plot, the upper right panel indeed confirms that the two
non-linear neutrino simulations lead to a very similar increase in matter
power, when compared to the simulation using linear neutrinos. Note however
that a significant fraction of the speed-up gained by employing the linear
:math:`\rightarrow` non-linear transition precisely arose due to the coarser
time-stepping, which has now been reverted back to the finer time-stepping.

To completely eliminate any effects caused by different time-stepping from the
upper right panel, we further need to run the *linear* neutrino simulation
using the same time-stepping as the non-linear ones. Feel free to do so.

.. tip::

   As stated previously, running with non-linear neutrinos require continual
   realisation of the linear pressure and shear. For the non-linear neutrino
   evolution within CO\ *N*\ CEPT to turn out correct, the linear inputs from
   CLASS then need to be nicely converged. For this to be the case, CLASS must
   be run with much higher precision in the neutrino sector than is used by
   default, hence the neutrino precision parameters specified in
   ``class_params`` in
   :ref:`the parameter file <param-nonlinear-massive-neutrinos>`. Stated
   briefly, ``'l_max_ncdm'`` sets the size of the linear massive neutrino
   Boltzmann hierarchy, while ``'Number of momentum bins'`` specifies the
   number of discrete neutrino momenta to employ for mapping the neutrino
   distribution function. Setting ``'Quadrature strategy'`` to ``2`` ensures
   that momentum integrals over the distribution function are integrated all
   the way to infinity. Lastly, setting ``'ncdm_fluid_approximation'`` to
   ``3`` disables the fluid approximation used by default for
   massive neutrinos.

   For simulations with higher resolution or lower neutrino masses, still
   higher values for ``'l_max_ncdm'`` and ``'Number of momentum bins'`` may be
   required, which can make the usually quick CLASS computation take up hours.
   As CO\ *N*\ CEPT runs CLASS in a distributed manner (even across compute
   nodes of a cluster) this is not an issue in practice. Furthermore,
   CO\ *N*\ CEPT caches all CLASS results to disk by default, so that such
   expensive CLASS computations do not have to be redone repeatedly.

You may play around with different neutrino masses ``_mass`` and/or transition
times ``_nutrans`` --- with or without ``static_timestepping`` --- while
continuing to use
:ref:`the plotting script <plot-nonlinear-massive-neutrinos>`, as it can
handle output of multiple masses and transition times at the same time.

