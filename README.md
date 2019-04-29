CO*N*CEPT
=========
<img align="right" height="300" src="http://users-phys.au.dk/jmd/github/concept/render2D.png"/>

CO*N*CEPT (**CO**smological ***N***-body **C**od**E** in **P**y**T**hon)
is a free and open-source code for cosmological structure formation.
The code should run on any Linux system, though it is primarily intended for
massively parallel computer clusters with distributed memory.
The code is written almost exclusively in Python, but achieves C-like
performance thanks to Cython.

CO*N*CEPT is capable of simultaneously evolving matter particles and fluids
such as (massless or massive) neutrinos at different orders of non-linearity.
Completely linear perturbations in all species and the metric itself are
fully supported, allowing for simulations consistent with
general relativity (GR).
The [CLASS](https://github.com/lesgourg/class_public) code is fully integrated
into CO*N*CEPT, making generation of e.g. initial conditions and GR corrections
easily available.

For academic papers produced using CO*N*CEPT, see:

- [Relativistic implementation of decaying dark matter](https://arxiv.org/abs/1904.11773)
  - (available as of release 0.3.0)
- [Linear dark energy perturbations](https://arxiv.org/abs/1904.05210)
  - (available as of release 0.3.0)
- [Linear photon, massive neutrino and GR corrections](https://arxiv.org/abs/1811.00904)
  - (available as of release 0.2.0)
- [Non-linear massive neutrinos](https://arxiv.org/abs/1712.03944)
  - (available as of release 0.1.0)


Installation instructions
-------------------------
As CO*N*CEPT has a lot of dependencies, it comes with an
[installation script](installer) which installs all of these
(and CO*N*CEPT itself) into a single location.
The path to the CO*N*CEPT installation directory may be given
as an argument. If not, the installer will prompt you for a directory.
You may further supply the `--tests` option which enables tests for each
dependency and CO*N*CEPT itself during the installation.
Without tests, the installation takes about an hour on modern hardware.

You can download and invoke the installer in one go by

    concept_version=v0.3.0
    bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer) [/path/to/concept] [--tests]

where brackets indicate optional arguments. Note that the initial
"`bash`" is required regardless of which shell you are using.
You may adjust the value of `concept_version` to your liking. For the
absolute newest (and unstable!) version, use `concept_version=master`.

Should the installation process end prematurely (e.g. due to network
failure), simply rerun the installation commands and it will pick up
from where it was.

Note that the above will install *all* dependencies into
`/path/to/concept`, regardless of whether these are already installed
somewhere else on the system. One of these dependencies is an
MPI 3 library, which can be preferable not to include when installing
CO*N*CEPT on a cluster, but instead make use of an already installed
MPI 3 library (any implementation should do). This is achieved by
setting the `mpi_dir` variable, e.g.

    concept_version=v0.3.0
    mpi_dir=/path/to/mpi bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer) [/path/to/concept] [--tests]

This trick may also be used should you wish to use
some other pre-installed dependency.

For further help with the installation process,
run the [installer](installer) with the `--help` option.


Running the code
----------------
To run a small sample simulation, navigate to the `concept` directory
and invoke

    ./concept -p params/example_params -n 2 --local

This will run the simulation defined by the provided `example_params`
parameter file using 2 processes. If omitting `--local` when on
a cluster, no simulation will be run. Instead, a matching job script
will be produced. For automatic submission of this job script to the
job scheduler of the cluster, supply the queue name using
the `-q QUEUE` option.

For additional options, run

    ./concept -h

To learn about the many parameters which can be specified in a
parameter file, study the `concept/params/example_params` parameter file.
It is much larger than typical parameter files because it contains
(almost) every possible parameter.


Further documentation
---------------------
- For the physics and numerical methods implemented, you may consult my [PhD thesis](https://tildeweb.au.dk/au282038/github/concept/phd_thesis-b5.pdf).
- No additional up-to-date documentation on how to actually *use* CO*N*CEPT exists.
  A very out-of-date [user's guide](https://arxiv.org/abs/1510.07621) from back
  when CO*N*CEPT was a pure particle code, as well as my [master's thesis](http://users-phys.au.dk/jmd/github/concept/masters_thesis.pdf)
  for which CO*N*CEPT was originally written, are available.

Please direct any questions and issues to the author: dakin(at)phys.au.dk

