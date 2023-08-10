## CO*N*CEPT changelog

<br/>



### 🚀 1.1.0 — *Still under development*

#### ✨ Features added
- Overhaul of initial condition generation:
  - Both 1LPT and **2LPT** now available.
    - Second-order growth factor and rate implemented via CLASS.
  - **BCC** and **FCC** lattices for pre-initial conditions.
  - Local **non-Gaussianity** now available.
  - Besides the *N*-body gauge, the synchronous and the Newtonian gauge
    are now available as well.
  - Optional back-scaling (as opposed to direct realisation).
- **Bispectrum** functionality:
  - Various configurations (equilateral, squeezed, stretched, isosceles, ...).
  - On-the-fly bispectrum measurements.
  - A `bispec` utility for computing bispectra from snapshots.
  - Reduced bispectra.
  - Novel anti-aliasing for bispectrum shells.
  - Perturbation theory (tree-level) predictions.
- Particle **IDs**.
- Linear power spectra **imprinted** with realisation noise.
- Improved and generalized 3D renders.
- Interlacing is now implemented through the new lattice system, meaning that
  we can now use either BCC (standard) or FCC interlacing. For potentials,
  we now have independent upstream and downstream interlacing.
- Support for loading of TIPSY snapshots.
- `CONCEPT_*` environment variables corresponding to command-line options.
- The `class` utility is easier to work with, owing to the new `--kmin`,
  `--kmax` and `--modes` options. The `--times` option can now also be used to
  specify explicit scale factor values to use.

#### ⚡ Optimizations
- The random numbers used for the primordial noise are now drawn in a
  distributed fashion.
- Snapshot data can now be saved and loaded partially (e.g. only particle
  positions).
- Multi-file GADGET snapshots can now be written in parallel.
- Faster detrending of perturbations.

#### 👌 Other changes
- Some command-line options are renamed. Boolean command-line options may now
  be supplied with an optional Boolean value.
- The binning of power spectra is now done logarithmically in wave number.
  The `powerspec_options['binsize']` parameter has accordingly been
  substituted for `powerspec_options['bins per decade']`.
- Default particle softening length changed to 2.5% of the mean
  inter-particle distance.
- The `install` script has been made more robust, in particular with regards
  to pre-installed MPI distributions, non-GNU compilers and Python packages,
  in particular NumPy/SciPy.
- Python >= 3.11 is now required.

[Commits since 1.0.1](https://github.com/jmd-dk/concept/compare/v1.0.1...master)

---
<br/><br/>



### 🚀 [1.0.1](https://github.com/jmd-dk/concept/releases/tag/v1.0.1) — 2021-12-07

#### 🐛 Bugs fixed
- Now safe to install CO*N*CEPT into non-empty directory.
- The code now refuses to run if the domain tile decomposition
  is less than `(3, 3, 3)`, as this otherwise leads to incorrect
  pairing when using non-trivial subtiling.
- Running a CLASS perturbation computation across more compute nodes
  than Fourier modes is now possible.

#### 📖 Documentation
- Copying code blocks containing prompts now leaves out the prompts.

#### 👌 Other changes
- The `install` script is now more user-friendly and less error prone
  when it comes to choosing an installation directory.

[Commits since 1.0.0](https://github.com/jmd-dk/concept/compare/v1.0.0...v1.0.1)

---
<br/><br/>



### 🚀 [1.0.0](https://github.com/jmd-dk/concept/releases/tag/v1.0.0) — 2021-12-03

#### ✨ Features added
- [**Documentation**](https://jmd-dk.github.io/concept/).
- [**Docker** images](https://hub.docker.com/r/jmddk/concept/).
- **Continuous integration** and **deployment** through GitHub workflows.
- Proper **example parameter files**.
- Proper **P³M** via tiles and subtiles.
- **Adaptive particle time-stepping**.
- Overhaul of grid implementation:
  - Grids used for PM, power spectra and 2D renders are now separate.
  - Addition of grid quantities from components are now carried out in an
    upstream → global → downstream scheme. Only the global grid is common
    to all components. Different sized upstream grids are added to the
    global grid in Fourier space.
  - Nyquist planes are explicitly zeroed.
  - Switched to cell-centred values.
  - NGP, CIC, TSC, **PCS** interpolation.
  - Differentiation orders 2, 4, 6, 8, Fourier space differentiation.
  - Toggleable **interlacing**.
  - Toggleable deconvolution.
- Full **GADGET snapshot** support.
- B-spline particle softening and only soften the Newtonian force.
- History-independent time step size.
- Allow for static global time-stepping.
- **'Paired-and-fixed'** initial conditions.
- Linear *N*-body gauge power spectrum output.
- Allow finite life times of components.
- Display of load imbalance.
- The `--job-name` option to `concept`.
- The `--no-lto` option to `concept`.
- The `--rebuild` option to `concept`.
- The `--repeats` option to the `play` utility.

#### 🐛 Bugs fixed
- Guard against particles with positions right at the edge of the box,
  domains, etc.

#### ⚡ Optimizations
- Particle positions/momenta now stored contiguously in memory.
- Chunkified domain/slab decompositions.
- Chunkified and linear complexity particle exchange.
- Chunkified snapshot read and write.
- More performant grid differentiation.
- More aggressive compile time optimization.
- New `--native-optimizations` option to `concept`.
- Fine-grained inlining.
- Basic invocation of `concept` and `make` sped up.
- Disabling of Cython assertions.
- Transpile-time insertions of `NAN` and `INFINITY` literals.

#### 🤖 Tests
- Test of `update` utility.
- Test of example parameter files.

#### 👌 Other changes
- New and more organised directory structure.
- Generalised build system:
  - Multiple simultaneous builds possible.
  - Build from submitted job possible.
- Convenient `path` and `param` variables in parameter files.
- Generalised species.
- More tunable power spectrum settings.
- Allow for deterministic [FFTW wisdom](https://www.fftw.org/fftw3_doc/Wisdom.html).
- Switched to the [PCG64 DXSM](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html) pseudo-random number generator.
- Explicit registration of CLASS perturbations.
- Improved robustness of the `install` script.
- 'Slim builds' now available.

[Commits since 0.3.0](https://github.com/jmd-dk/concept/compare/v0.3.0...v1.0.0)

---
<br/><br/>



### 🚀 [0.3.0](https://github.com/jmd-dk/concept/releases/tag/v0.3.0) — 2019-04-29

#### ✨ Features added
- **Decaying dark matter**, including dark radiation
  and the 'lapse component'.
- Additional perturbations available from the `class` utility (e.g. ϕ, ψ).

#### 🐛 Bugs fixed
- PPF dark energy pressure perturbation corrected in CLASS.

#### ⚡ Optimizations
- Link time optimizations.
- GSL linked to OpenBLAS.

#### 👌 Other changes
- Several improvements to the computation of
  dark energy perturbations in CLASS.
- Added missing *N*-body gauge transformation for pressure perturbations.
- Detrending of CLASS perturbations now done in intervals.
- Broader compiler support by the `install` script.
- Tests no longer performed by default by the `install` script
  (instead, supply `-t`).
- More robust integration with pre-installed MPI libraries.

[Commits since 0.2.1](https://github.com/jmd-dk/concept/compare/v0.2.1...v0.3.0)

---
<br/><br/>



### 🚀 [0.2.1](https://github.com/jmd-dk/concept/releases/tag/v0.2.1) — 2019-02-11

#### ✨ Features added
- Linear, **dynamical dark energy** (fluid and PPF).
- The `update` utility can now update CLASS as well.

#### 🐛 Bugs fixed
- Erroneous sound speed computed in Kurganov Tadmor method.

#### 👌 Other changes
- Primordial spectrum separated from CLASS parameters.
- CLASS perturbations are now MPI distributed.

[Commits since 0.2.0](https://github.com/jmd-dk/concept/compare/v0.2.0...v0.2.1)

---
<br/><br/>



### 🚀 [0.2.0](https://github.com/jmd-dk/concept/releases/tag/v0.2.0) — 2018-11-05

#### ✨ Features added
- Linear **neutrinos**, **photons** and the '**metric** component'.
- The `class` utility.
- Command-line parameters (`-c`) and job directives (`-j`)
  as options to `concept`.

#### 🐛 Bugs fixed
- Erroneous time-stepping limiter for fluid components.

#### 👌 Other changes
- Random numbers for initial condition generation now drawn in shells.

[Commits since 0.1.0](https://github.com/jmd-dk/concept/compare/v0.1.0...v0.2.0)

---
<br/><br/>



### 🚀 [0.1.0](https://github.com/jmd-dk/concept/releases/tag/v0.1.0) — 2018-01-16

#### ✨ Features
- **Installation script**.
- [Cython](https://cython.org/) framework.
- [CLASS](http://class-code.net/) integration.
- **Initial condition generation**, adopting the *N*-body gauge.
- **Particle** (matter) components with Plummer softening.
- **Fluid** (massive neutrino) components, focusing on non-linear evolution.
- **Gravity**:
  - PP.
  - PM.
  - P³M (inefficient).
- **Snapshots**:
  - Native CO*N*CEPT format.
  - GADGET format (limited).
- **Power spectra**.
- 3D and 2D **renders**.
- **Utilities**.
- **Test** suite.

