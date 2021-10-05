## CO*N*CEPT changelog



### 🚀 1.0.0 — *Still under development*

#### ✨ Features added
- [Documentation](https://jmd-dk.github.io/concept/)
- Proper P³M via tiles and subtiles.
- Adaptive particle time-stepping.
- Overhaul of grid implementation:
  - Grids used for for PM, power spectra and 2D renders are now separate.
  - Addition of grid quantities from components are now carried out in an
    upstream → global → downstream scheme. Only the global grid is common
    to all components. Different sized upstream grids are added to the
    global grid in Fourier space.
  - Nyquist planes are explicitly zeroed.
  - Switched to cell-centred values.
  - NGP, CIC, TSC, PCS interpolation.
  - Differentiation orders 2, 4, 6, 8, Fourier space differentiation.
  - Toggleable interlacing.
  - Toggleable deconvolution.
- Full GADGET snapshot support.
- Switch to B-spline particle softening and only soften the Newtonian force.
- History-independent time step size.
- [Docker images](https://hub.docker.com/r/jmddk/concept/).
- 'Paired-and-fixed' initial conditions.
- Linear *N*-body gauge power spectrum output.
- Allow finite life times of components.
- Allow for static global time-stepping.
- Display of load imbalance.
- The `--no-lto` option to `concept`.
- The `--no-recompilation` option to `concept`.

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

#### 👌 Other changes
- Generalised species.
- More tunable power spectrum settings.
- Allow for deterministic FFTW wisdom.
- Switched to the PCG64DXSM pseudo-random number generator.
- Explicit registration of CLASS perturbations.
- Improved robustness of the `installer`.
- 'Slim builds' now available.

[Commits since 0.3.0](https://github.com/jmd-dk/concept/compare/v0.3.0...master)

---
<br/><br/>



### 🚀 [0.3.0](https://github.com/jmd-dk/concept/releases/tag/v0.3.0) — 2019-04-29

#### ✨ Features added
- Decaying dark matter, including dark radiation and the 'lapse component'.
- Additional perturbations available from the `CLASS` utility (e.g. ϕ, ψ).

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
- Broader compiler support by the `installer`.
- Tests no longer performed by default by the `instaler`
  (instead, supply `-t`).
- More robust integration with pre-installed MPI libraries.

[Commits since 0.2.1](https://github.com/jmd-dk/concept/compare/v0.2.1...v0.3.0)

---
<br/><br/>



### 🚀 [0.2.1](https://github.com/jmd-dk/concept/releases/tag/v0.2.1) — 2019-02-11

#### ✨ Features added
- Linear, dynamical dark energy (fluid and PPF).
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
- Linear neutrinos, photons and the 'metric component'.
- The `CLASS` utility.
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

First release, containing the following:
- Installation script.
- [Cython](https://cython.org/) framework.
- [CLASS](http://class-code.net/) integration.
- Initial condition generation, adopting the *N*-body gauge.
- Particle (matter) components with Plummer softening.
- Fluid (massive neutrino) components, focusing on non-linear evolution.
- Gravity: PP, PM, (inefficient) P³M.
- Snapshots: CO*N*CEPT and (limited) GADGET formats.
- Power spectra: Basic (using the PM grid).
- Renders: 3D and 2D (using the PM grid).
- Utilities.
- Test suite.

