Bispectra and 2LPT
------------------
The focus of this section is on bispectrum measurements from simulations. We
shall also see how using 2LPT initial conditions improves the initial
bispectrum of the particle distribution.



.. raw:: html

   <h3>Equilateral bispectra</h3>

We shall make use of the following parameter file throughout this section.
Save it as e.g. ``param/tutorial-8``:

.. code-block:: python3
   :caption: param/tutorial-8
   :name: param-bispec
   :emphasize-lines: 12, 16, 23-25, 30-34, 43-45, 48-50, 52, 56-59

   # Non-parameter helper variable used to control the size of the simulation
   _size = 64

   # Input/output
   initial_conditions = {
       'species': 'matter',
       'N'      : _size**3,
   }
   output_dirs = {
       'snapshot' : f'{path.output_dir}/{param}/{_lpt}LPT/seed{_seed}_shift{_shift}',
       'powerspec': ...,
       'bispec'   : f'{path.output_dir}/{param}/{_lpt}LPT/seed{_seed}_shift{_shift}/{_conf}',
   }
   output_times = {
       'powerspec': [a_begin, 0.3, 1],
       'bispec'   : ...,
   }
   if _lpt == 2:
       output_times |= {'snapshot': ...}
   powerspec_select = {
       'matter': {'data': True, 'linear': True, 'plot': True},
   }
   bispec_select = {
       'matter': {'data': True, 'reduced': True, 'tree-level': True, 'plot': True},
   }

   # Numerics
   boxsize = 512*Mpc
   potential_options = 2*_size
   bispec_options = {
       'configuration': {
           'matter': _conf,
       },
   }

   # Cosmology
   H0      = 67*km/(s*Mpc)
   Ωb      = 0.049
   Ωcdm    = 0.27
   a_begin = 0.02

   # Physics
   realization_options = {
       'LPT': _lpt,
   }

   # Simulation
   random_seeds = {
       'primordial phases': 2_000 + _seed,
   }
   primordial_amplitude_fixed = False
   primordial_phase_shift = _shift

   # Non-parameter helper variables which should
   # be supplied as command-line parameters.
   _conf  = 'equilateral'  # triangle configurations for bispectra
   _seed  = 0              # seed offset used for the phases of the primordial noise
   _shift = 0              # phase shift for primordial noise
   _lpt   = 1              # order of Lagrangian perturbation theory

We start by running this parameter file as is:

.. code-block:: bash

   ./concept -p param/tutorial-8

(feel free to run this in parallel by further supplying ``-n``). This will run
a simple simulation and measure the matter bispectrum (as well as the power
spectrum) at three different times along the evolution, as specified in
``output_times``. The bispectra (data files and plots) are dumped into a
nested set of subdirectories within ``output/tutorial-8`` (for organisational
purposes, as many more bispectrum measurements are to come). You can have a
look at these, but what we really want is to plot the different bispectra
together in a single plot, for which we make use of the plotting script below:

.. code-block:: python3
   :caption: output/tutorial-8/plot.py
   :name: plot-bispec

   import collections, glob, os, re, sys
   import numpy as np
   import matplotlib; matplotlib.use('agg')
   import matplotlib.pyplot as plt

   # Read in data
   this_dir = os.path.dirname(os.path.realpath(__file__))
   data = collections.defaultdict(lambda: collections.defaultdict(float))
   confs = set()
   lpts = set()
   a_values = set()
   for dirname in glob.glob(f'{this_dir}/*LPT'):
       lpt = int(re.search(r'(\d)LPT$', dirname).group(1))
       lpts.add(lpt)
       for dirname in glob.glob(f'{dirname}/seed*_shift*'):
           for dirname in glob.glob(f'{dirname}/*'):
               if not os.path.isdir(dirname):
                   continue
               conf = os.path.basename(dirname)
               confs.add(conf)
               a_files = set()
               for filename in glob.glob(f'{dirname}/bispec*'):
                   if filename.endswith('.png'):
                       continue
                   with open(filename, mode='r', encoding='utf-8') as f:
                       a = float(re.search(r'a = (.+)', f.readline()).group(1))
                   a_values.add(a)
                   if a in a_files:
                       print(
                           f'Found multiple bispectrum data files'
                           f'for a = {a} in {dirname}',
                           file=sys.stderr,
                       )
                   a_files.add(a)
                   k, t, μ, B, B_tree, Q, Q_tree = np.loadtxt(
                       filename, usecols=(0, 1, 2, 5, 6, 7, 8), unpack=True,
                   )
                   data[conf, lpt, a]['k'     ] = k
                   data[conf, lpt, a]['t'     ] = t
                   data[conf, lpt, a]['μ'     ] = μ
                   data[conf, lpt, a]['B_tree'] = B_tree
                   data[conf, lpt, a]['Q_tree'] = Q_tree
                   data[conf, lpt, a]['B'     ] += B
                   data[conf, lpt, a]['Q'     ] += Q
                   data[conf, lpt, a]['n'     ] += 1
   for d in data.values():
       d['B'] /= d['n']
       d['Q'] /= d['n']
   confs    = sorted(confs)
   lpts     = sorted(lpts)
   a_values = sorted(a_values)

   # Plotting functions
   def plot1D(conf):
       def get_color(a, lpt=1, color=None):
           if color is None:
               color = f'C{a_values.index(a)%10}'
           color = np.asarray(matplotlib.colors.ColorConverter().to_rgb(color))
           color /= (1 + 0.4*(lpt - 1))
           return color
       def get_linestyle(lpt):
           return {1: '-', 2: '--', 3: ':'}.get(lpt, '-.')
       fig, axes = plt.subplots(2, sharex=True)
       for lpt in lpts:
           linestyle = get_linestyle(lpt)
           for a in a_values:
               subdata = data.get((conf, lpt, a))
               if subdata is None:
                   continue
               k = subdata['k']
               axes[0].loglog  (k, subdata['B'], linestyle, color=get_color(a, lpt))
               axes[1].semilogx(k, subdata['Q'], linestyle, color=get_color(a, lpt))
               axes[0].loglog  (k, subdata['B_tree'], 'k--', linewidth=1)
               axes[1].semilogx(k, subdata['Q_tree'], 'k--', linewidth=1)
               axes[0].set_xlim(k[0], k[-1])
       axes[1].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
       axes[0].set_ylabel(r'$B\, [\mathrm{Mpc}^6]$')
       axes[1].set_ylabel(r'$Q$')
       axes[0].tick_params('x', direction='inout', which='both')
       axes[1].set_zorder(-1)
       # Legends
       for a in a_values:
           axes[0].plot(
               0.5, 0.5, '-',
               transform=axes[0].transAxes,
               color=get_color(a), label=rf'$a = {a}$',
           )
       for lpt in lpts:
           axes[1].plot(
               0.5, 0.5, get_linestyle(lpt),
               color=get_color(1, lpt, 'grey'),
               transform=axes[1].transAxes,
               label=f'simulation: {lpt}LPT',
           )
       axes[1].plot(
           0.5, 0.5, 'k--',
           transform=axes[1].transAxes,
           linewidth=1, label='tree-level',
       )
       for ax in axes:
           ax.legend(fontsize=9)
       return fig
   def plot2D(conf, y):
       def get_logticks(logk, y, B):
           fig_tmp, ax_tmp = plt.subplots()
           k = 10**logk
           ax_tmp.tripcolor(k, y, B)
           ax_tmp.set_xscale('log')
           x_min, x_max = k[0], k[-1]
           ticks = np.log10(ax_tmp.get_xticks())
           plt.close(fig_tmp)
           return ticks
       fig, axes = plt.subplots(
           2, len(a_values),
           figsize=(6.4*(1 + 0.5*(len(a_values) - 1)), 4.8),
           sharex=True,
       )
       for lpt in lpts:
           for i, a in enumerate(a_values):
               subdata = data.get((conf, lpt, a))
               if subdata is None:
                   continue
               logk = np.log10(subdata['k'])
               B, B_tree = subdata['B'], subdata['B_tree']
               Q, Q_tree = subdata['Q'], subdata['Q_tree']
               # B
               ax = axes[0, i]
               logc = (np.max(B[B > 0])/np.min(B[B > 0]) > 1e+2)
               pc = ax.tripcolor(
                   logk, subdata[y], B,
                   norm=(matplotlib.colors.LogNorm() if logc else None),
                   shading='gouraud',
               )
               cbar = fig.colorbar(pc, ax=ax, shrink=0.9)
               cbar.ax.tick_params(labelsize=8)
               cbar.set_label(r'$B\, [\mathrm{Mpc}^6]$')
               if logc:
                   cbar.ax.set_yscale('log')
               ax.tricontour(
                   logk, subdata[y], B_tree,
                   norm=pc.norm, linewidths=1,
               )
               logc_tree = (np.max(B_tree)/np.min(B_tree) > 1e+2)
               tc = axes[0, i].tricontour(
                   logk, subdata[y], B_tree,
                   norm=(matplotlib.colors.LogNorm() if logc_tree else None),
                   colors='k', linestyles='dashed', linewidths=1,
               )
               fmt = matplotlib.ticker.LogFormatterSciNotation()
               fmt.create_dummy_axis()
               ax.clabel(tc, tc.levels, fmt=fmt)
               ticks = get_logticks(logk, subdata[y], B)
               # Q
               ax = axes[1, i]
               pc = ax.tripcolor(logk, subdata[y], Q, shading='gouraud')
               cbar = fig.colorbar(pc, ax=ax, shrink=0.9)
               cbar.ax.tick_params(labelsize=8)
               cbar.set_label(r'$Q$')
               ax.tricontour(
                   logk, subdata[y], Q_tree,
                   norm=pc.norm, linewidths=1,
               )
               tc = ax.tricontour(
                   logk, subdata[y], Q_tree,
                   colors='k', linestyles='dashed', linewidths=1,
               )
               ax.clabel(tc, tc.levels)
               axes[0, i].set_title(rf'$a = {a}$')
               for j, ax in enumerate(axes[:, i]):
                   ax.set_ylim(np.min(subdata[y]), 1 - 1e-6*j)
       for ax in axes[1, :]:
           ax.set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$')
           ax.set_xticks(ticks)
           ax.xaxis.set_major_formatter(
               matplotlib.ticker.FuncFormatter(
                   lambda val, pos=None: matplotlib.ticker.LogFormatterSciNotation()(
                       10**val, pos,
                   )
               )
           )
           ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
       for ax in axes[:, 0]:
           ax.set_ylabel({'t': r'$t$', 'μ': r'$\mu$'}[y])
       for ax in axes.flatten():
           ax.tick_params(axis='both', which='both', labelsize=8)
       for ax in axes[0, :]:
           ax.tick_params('x', direction='inout', which='both')
       for ax in axes[:, 1:].flatten():
           ax.tick_params('y', direction='inout', which='both')
       for i in range(axes.shape[0]):
           for j in range(axes.shape[1]):
               axes[i, j].set_zorder(j - i)
       # Legend
       ax = axes[1, 0]
       markerfacecolors = getattr(
           matplotlib.cm, matplotlib.rcParams['image.cmap'],
       )([0.2, 0.8])[:, :3]
       ylim = ax.get_ylim()
       ax.plot(
           0.5, 2, 's',
           transform=ax.transAxes,
           markersize=13, markeredgecolor='none', fillstyle='top',
           markerfacecolor=markerfacecolors[1], markerfacecoloralt=markerfacecolors[0],
           label='simulation',
       )
       ax.set_ylim(ylim)
       ax.plot(
           0.5, 0.5, 'k--',
           transform=ax.transAxes,
           linewidth=1,
           label='tree-level',
       )
       ax.legend(fontsize=9)
       return fig

   # Plot
   for conf in confs:
       conf_simplified = (
           conf
           .replace('-', '')
           .replace('_', '')
           .replace(' ', '')
           .lower()
       )
       if conf_simplified in {'equilateral', 'stretched', 'squeezed', 'isoscelesright'}:
           fig = plot1D(conf)
       elif conf_simplified in {'lisosceles', 'sisosceles'}:
           fig = plot2D(conf, 'μ')
       elif conf_simplified in {'elongated', 'linear', 'right'}:
           fig = plot2D(conf, 't')
       else:
           print(
               f'Configuration {conf} not known to plotting script',
               file=sys.stderr,
           )
           continue
       fig.suptitle(conf)
       fig.tight_layout()
       fig.subplots_adjust(hspace=0)
       fig.savefig(f'{this_dir}/plot_{conf}.png', dpi=150)
       plt.close(fig)

Do not worry about the large size of the plotting script; though we shall make
use of it, we shall not study its content in any detail. Store the plotting
script as e.g. ``output/tutorial-8/plot.py``. With the first simulation
completed, run this script using

.. code-block:: bash

   ./concept -m output/tutorial-8/plot.py

This will produce a file named ``plot_equilateral.png`` in the
``output/tutorial-8`` directory. Here, 'equilateral' refers to the particular
bispectrum configuration measured within the simulation, as specified within
the ``bispec_options`` parameter. In our :ref:`parameter file <param-bispec>`
this value is set through the helper variable ``_conf``, which indeed is given
a default value of ``'equilateral'``. The plot shows the full bispectrum
:math:`B` in the upper panel and the reduced bispectrum :math:`Q` in the lower
panel. Both the simulation results and the theoretical tree-level predictions
are shown for both :math:`B` and :math:`Q`. All of these are made available in
the bispectrum data files due to them being enabled in the ``bispec_select``
parameter in the :ref:`parameter file <param-bispec>`.

While the intermediary (:math:`a = 0.3`) bispectrum seems to agree reasonably
well with the tree-level prediction, the final (:math:`a = 1`) bispectrum has
grown beyond the tree-level prediction as a result of the non-linear gravity
applied throughout the simulation. The initial
(:math:`a = a_{\text{begin}} = 0.02`) bispectrum looks very bad in comparison,
which is caused by the low bispectrum signal mostly drowning in noise.

.. note::
   It may appear that only a single tree-level prediction is shown for the
   reduced bispectrum :math:`Q` in the lower panel, but this is really due to
   this value being very close to time-independent (and approximately equal to
   :math:`\frac{4}{7}` for the case of the equilateral bispectrum).



.. raw:: html

   <h4>Precision through averages</h4>

To help with the noise issue (especially noticeable at early times), we can
fix the amplitudes for the primordial random noise to all have the value of
the ensemble average. For this we use the ``primordial_amplitude_fixed``
:ref:`parameter <primordial_amplitude_fixed>`, already present in the
:ref:`parameter file <param-bispec>` but set to ``False``. Change this to
``True``, rerun the simulation and update the plot.

While simply fixing the primordial amplitudes works wonders for the *power*
spectrum, its usefulness for the *bi*\ spectrum may not be so evident. The true
strength of the fixed amplitudes comes when we combine two such simulations,
having primordial noise which is completely out of phase with each other. That
is, we now want to run a partner-simulation also with fixed amplitudes, but
with the phases shifted by :math:`\require{upgreek}\uppi`. For this we can use
the ``primordial_phase_shift`` :ref:`parameter <primordial_phase_shift>`, which
in the :ref:`parameter file above <param-bispec>` is controlled through the
``_shift`` helper variable:

.. code-block:: bash

   ./concept \
       -p param/tutorial-8 \
       -c "_shift = π"

Running the above does not overwrite the existing results, but dumps the new
output into a separate subdirectory in accordance with the shift. If you now
compare the *automatically generated* bispectrum plot at the initial time
(``output/tutorial-8/1LPT/seed0_shift0/equilateral/bispec_a=0.02.png``) of the
non-shifted simulation with that of the shifted simulation, you will find that
the two bispectra almost look like each others negative (dashed pieces of the
coloured line indicate negative :math:`B`). Averaging them together will thus
cancel out much of the noise, resulting in a much better measurement of the
true matter bispectrum inherent to the cosmology under consideration. In fact,
the :ref:`plotting script <plot-bispec>` is set up to do exactly that! Rerun
the plotting script to now make ``plot_equilateral.png`` much nicer,
especially at early times.

While the primordial amplitudes are kept fixed, the phases are still being
randomly drawn. We can thus further reduce the noise in the bispectrum by
increasing the number of phases. We could do this by increasing the simulation
size, but we may instead choose to carry out the simulations multiple times
over, each time making use of a different set of random primordial phases,
again with the intent of averaging together their individual bispectra. The
various random seeds used by CO\ *N*\ CEPT live within the ``random_seeds``
:ref:`parameter <random_seeds>`, with ``'primordial phases'`` being the seed
of interest currently. Furthermore, our :ref:`parameter file <param-bispec>`
is set up to construct this seed based on a ``_seed`` helper variable. To
increase the precision of our final bispectrum significantly, let us perform
three additional paired simulations, i.e. make use of three additional seeds,
with two anti-correlated ("paired") simulations being carried out for each
such seed:

.. code-block:: bash

   for seed in 1 2 3; do
       for shift in 0 π; do
           ./concept \
               -p param/tutorial-8 \
               -c "_seed = $seed" \
               -c "_shift = $shift"
       done
   done

Once all simulations are complete, update the plot. The bispectrum should now
have become decently smooth at all three times, possibly with some numerical
artefacts at the high and/or low :math:`k` end. It is now very clear that the
initial bispectrum falls short of the tree-level value, implying that the
particle system is not initialised in a manner that respects this theoretical
prediction.



.. raw:: html

   <h3>Second-order Lagrangian perturbation theory initial conditions</h3>

The reason for the mismatch between the simulation initial conditions and the
perturbative tree-level prediction is that the former makes use of first-order
(Lagrangian) perturbation theory while the latter is a second-order result.
By default, CO\ *N*\ CEPT uses first-order Lagrangian perturbation theory
(1LPT) to assign initial particle positions and momenta (through what is often
referred to as the Zel'dovich approximation), though second-order (2LPT)
corrections can optionally be applied on top. The order of LPT to use is
specified by the ``'LPT'`` sub-parameter of the ``realization_options``
:ref:`parameter <realization_options>`, which defaults to ``1``. In
:ref:`our parameter file <param-bispec>`, this is controlled through the
``_lpt`` helper variable.

To see what difference switching out 1LPT for 2LPT has on the initial as well
as the later bispectra, we should run all :math:`4\times 2` simulations again,
this time with ``_lpt = 2``:

.. code-block:: bash

   for seed in 0 1 2 3; do
       for shift in 0 π; do
           ./concept \
               -p param/tutorial-8 \
               -c "_seed = $seed" \
               -c "_shift = $shift" \
               -c "_lpt = 2"
       done
   done

Once again, these new results will not overwrite any of the old. While you can
in fact update the plot continually while the above simulations are busy being
carried out, you should suspend any judgement until they are all complete. The
2LPT results will appear as dashed lines.

With the plot updated following the completion of every simulation, we see that
opting for 2LPT indeed leads to excellent agreement with the tree-level
prediction at the initial time. At later times, the (relative) difference in
the simulation bispectra for 1LPT and 2LPT becomes less pronounced. We also
see that the simulation bispectrum follows the tree-level prediction closely
up until rather late times, before outgrowing it at the larger
:math:`k` values.

Since using 1LPT lands us on the desired linear *power* spectrum at the
initial time, we might fear that opting for 2LPT ruins this behaviour. We can
check this by comparing the automatically generated ``powerspec_a=0.02.png``
plot within any of the subdirectories of ``output/tutorial-8/1LPT`` with one
from the subdirectories of ``output/tutorial-8/2LPT`` (the seed and shift does
not matter much for the initial power spectrum, due to
``primordial_amplitude_fixed = True``). They should appear completely
indistinguishable, demonstrating that the 2LPT corrections leaves the 1LPT
power spectrum invariant (at least for the large/intermediary scales we are
working with here).

If you would like to see still better convergence of the bispectrum, feel free
to run even more paired simulations with different seeds (using 2LPT, 1LPT,
or both).

.. note::
   CO\ *N*\ CEPT further implements 3LPT, which given our
   :ref:`parameter file <param-bispec>` you can enable via ``_lpt = 3``.
   The changes to the bispectrum when upgrading from 2LPT to 3LPT are however
   much less dramatic than the changes from 1LPT to 2LPT. In fact, given the
   parameters set in our :ref:`parameter file <param-bispec>`, the difference
   will be virtually impossible to see from the plots generated by our
   :ref:`plotting script <plot-bispec>`. For 3LPT to be a significant
   improvement upon 2LPT, we need to initialise later and/or simulate
   smaller scales than what we have just done. For example, changing to
   ``a_begin = 0.08`` and ``boxsize = 128*Mpc`` and then rerunning everything
   using ``_lpt = 1``, ``_lpt = 2`` and ``_lpt = 3``, differences between 2LPT
   and 3LPT will show up in the generated plot at high :math:`k`.



.. raw:: html

   <h3>Other bispectrum configurations</h3>

Let us now consider bispectrum configurations different from the equilateral
case. Another commonly studied case is that of the "squeezed" bispectrum,
where one of the triangle legs (one of the three :math:`k` values) is much
smaller than the other two. The configuration to use is controlled by the
``'configuration'`` sub-parameter of the ``bispec_options``
:ref:`parameter <bispec_options>`, which in the
:ref:`above parameter file <param-bispec>` is controlled through the ``_conf``
helper variable. We could rerun the many simulations in a manner similar to
what we did above (now with ``_conf = 'squeezed'`` included), but we can save
time by computing the bispectra directly off of the snapshots dumped during
the 2LPT runs (that's right; all of the 2LPT simulations sneakily dumped
snapshots as well!).

In order to compute bispectra directly from the snapshots, we shall make use
of the *bispec utility*:

.. code-block:: bash

   ./concept \
       -u bispec \
       -p param/tutorial-8 \
       -c "_conf = 'squeezed'" \
       output/tutorial-8/2LPT/*

(possibly with the inclusion of ``-n``). Note that the above will compute the
squeezed bispectrum for all snapshots produced by the 2LPT runs. For a given
snapshot, the output files produced by the bispec utility will be placed in
the directory containing said snapshot. That is, they will not all be neatly
collected into a separate subdirectory named ``squeezed``, next to the already
existing ``equilateral`` subdirectory. Let's organise the squeezed bispectra
appropriately ourselves, then:

.. code-block:: bash

   organise() {
       (
           cd output/tutorial-8/2LPT \
           && for d in *; do \
               cd $d \
               && mkdir -p $1 \
               && mv bispec* $1/ \
               && cd ..
           done
       )
   }
   organise squeezed

If you now take a look at the contents of the ``output/tutorial-8/2LPT``
directory, each subdirectory should have equilateral and squeezed bispectrum
files neatly sorted into separate further subdirectories.

Rerunning the plotting script, we now additionally get ``plot_squeezed.png``,
showing only the 2LPT (as we did not compute the squeezed bispectra for the
1LPT simulations) results together with their tree-level predictions. Looking
at the lower :math:`Q` panel, the agreement between the simulation measurements
and the tree-level predictions does not appear to be as good as for the
equilateral configurations. From this it is clear that we cannot generally
expect our simulations to be initialised with a bispectrum matching the
tree-level prediction for a general triangle configuration, even when
using 2LPT. The striking agreement we found for the equilateral configurations
then appears even more incredible. Looking at the upper :math:`B` panel of
``plot_squeezed.png``, we see that the simulation and theory results *do*
manage to closely follow each other over many orders of magnitude.

.. note::
   To specify a given triangle configuration, CO\ *N*\ CEPT uses the
   :math:`(k, t, \mu)` parametrisation, related to the side lengths of the
   triangle :math:`(k_1, k_2, k_3)` through

   .. math::

      k &\equiv k_1\,,\\
      t &\equiv \frac{k_2}{k_1}\,,\\
      \mu &\equiv \frac{k_1^2 + k_2^2 - k_3^2}{2k_1k_2}\,.

   Here :math:`k` specifies the overall size of the triangle, while :math:`t`
   and :math:`\mu` together specify the shape. The equilateral configuration
   :math:`k_1 = k_2 = k_3` thus corresponds to :math:`t = 1`,
   :math:`\mu = \frac{1}{2}`, leaving :math:`k` to be varied. The squeezed
   limit is defined by :math:`k_3 = 0` and thus :math:`k_1 = k_2`,
   corresponding to :math:`t = 1`, :math:`\mu = 1`. At this very limit, the
   :math:`k_3` shell contains no modes and so no measurement of the bispectrum
   can be performed. A slight deviation from the true squeezed limit thus has
   to be made, with  CO\ *N*\ CEPT choosing to use :math:`\mu = 0.99`.

   The first three columns of the bispectrum data files contain :math:`k`,
   :math:`t` and :math:`\mu`, respectively.

   For further details on the :math:`(k, t, \mu)` parametrisation used for the
   various pre-defined configurations in CO\ *N*\ CEPT, see the
   ``bispec_options`` :ref:`parameter <bispec_options>`.

Using the same snapshots as before, we can compute the bispectrum for yet
another class of configurations. Let's choose "S-isosceles":

.. code-block:: bash

   conf=S-isosceles
   ./concept \
       -u bispec \
       -p param/tutorial-8 \
       -c "_conf = '$conf'" \
       output/tutorial-8/2LPT/* \
   && organise $conf \
   && ./concept -m output/tutorial-8/plot.py

Note that the above includes the bispectrum measurements, file organisation
and subsequent plotting. Once completed, ``plot_S-isosceles.png`` will have
appeared. This plot is qualitatively different from the equilateral and
squeezed plots, as now the configuration subspace is two-dimensional, with
both :math:`k` and :math:`\mu` varying. In fact, "S-isosceles" is defined
through :math:`\frac{1}{2} \le t \le 1`, :math:`\mu = (2t)^{-1}` (or
equivalently :math:`k_1 \ge k_2 = k_3`), meaning that :math:`t` varies as
well, but in a manner dependent on :math:`\mu`. The simulation results are
shown in this plot as shaded regions, with the colours corresponding to
:math:`B` and :math:`Q` values as indicated by the colorbars. The tree-level
predictions are shown on top with dashed, black contour lines. Though it can
be difficult to see, each dashed tree-level line further consists of a full,
coloured line (drawn underneath), with the colour again corresponding to the
value via the colorbars. For the lower :math:`Q` panels, the colours of the
tree-level lines appear more clearly on top of the shaded regions (especially
toward high :math:`\mu`), indicating a significant difference between the
simulation and tree-level :math:`Q` value. Conversely, the fact that the
colour of the tree-level lines does not stand out on top of the upper shaded
regions tells us that the tree-level and simulation results for :math:`B`
agree well. As expected, the simulation results start to clearly deviate from
the tree-level prediction at late times, as seen in the upper right panel.

Let us measure the bispectrum for yet another two-dimensional configuration
subspace, say that of "elongated" triangles, meaning triangles with no
enclosed area; :math:`\frac{1}{2} \le t \le 1`, :math:`\mu = 1` (or
equivalently :math:`k_1 = k_2 + k_3`).

.. note::
   In the literature, "elongated" configurations are also sometimes referred to
   as "flattened", "folded" or "linear".

Perform the bispectrum measurements in the usual manner:

.. code-block:: bash

   conf=elongated
   ./concept \
       -u bispec \
       -p param/tutorial-8 \
       -c "_conf = '$conf'" \
       output/tutorial-8/2LPT/* \
   && organise $conf \
   && ./concept -m output/tutorial-8/plot.py

The resulting ``plot_elongated.png`` will again show two-dimensional subplots,
though this time with :math:`t` as the second independent variable. We again
see reasonable agreement between the simulation and tree-level results,
though not quite as stunning as for S-isosceles. Most strikingly though, a
significant chunk of parameter space is missing, corresponding to low
:math:`k` and high :math:`t`. Though this part in fact belongs to the
elongated subspace of the full triangle configuration space, it has been
excluded due to :math:`k_3` being dangerously small, leading to measurements
of :math:`B` being much less accurate here.

You are encouraged to explore even more triangle configurations on your own.
The :ref:`plotting script <plot-bispec>` will recognize the following
configurations:

* elongated
* equilateral
* isosceles-right
* L-isosceles
* right
* S-isosceles
* squeezed
* stretched

To learn more about the different bispectrum configurations available in
CO\ *N*\ CEPT --- including completely general, manual configuration
specification --- consult the documentation for the ``bispec_options``
:ref:`parameter <bispec_options>`.

