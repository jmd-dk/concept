CO*N*CEPT
=========
<img align="right" height="255" src="http://users-phys.au.dk/jmd/github/concept/render2D.png"/>

CO*N*CEPT (**CO**smological ***N***-body **C**od**E** in **P**y**T**hon)
is a free and open-source code for cosmological structure formation.
The code should run on any Linux system, though it is primarily intended
for massively parallel computer clusters with distributed memory.
The code is written almost exclusively in Python, but achieves C-like
performance thanks to Cython.

CO*N*CEPT is a mixed particle and grid code. Current development
focuses on non-linear neutrino simulations, as described in
the [νCO*N*CEPT paper](https://arxiv.org/abs/1712.03944).
The 0.1.0 release corresponds to the version of the code used
for this paper.


Installation instructions
-------------------------
As CO*N*CEPT has a lot of dependencies, it comes with an
[installation script](installer) which installs all of these
(and CO*N*CEPT itself) into a single location.
The path to the CO*N*CEPT installation directory may be given
as an argument. If not, the installer will prompt you for a directory.
The installation will take a couple of hours, depending on the hardware.
To speed up the installation you may supply the optional `--fast`
option which skips all tests (not recommended).
You can download and invoke the installer in one go by

    concept_version=v0.1.0
    bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer) [/path/to/concept] [--fast]

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

    concept_version=0.1.0
    mpi_dir=/path/to/mpi bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/${concept_version}/installer) [/path/to/concept] [--fast]

This trick may also be used should you wish to use
some other pre-installed dependency.


Running the code
----------------
To run a small sample simulation, navigate to the `concept` directory
and invoke

    ./concept -p params/example -n 2 --local

This will run the simulation defined by the provided `example`
parameter file using 2 processes. If omitting `--local` when on
a cluster, no simulation will be run. Instead, a matching job script
will be produced. For automatic submission of this job script to the
job scheduler of the cluster, supply the queue name using
the `-q QUEUE` option.

For additional options, run

    ./concept -h

To learn about the many parameters which can be specified in a
parameter file, study the `concept/params/example` parameter file.
It is much larger than typical parameter files because it contains
(almost) every possible parameter.


Further documentation
---------------------
Unfortunately, no additional up-to-date documentation exists.
An out-of-date [user guide](https://arxiv.org/abs/1510.07621) from back
when CO*N*CEPT was a pure particle code,
as well as the [master's thesis](http://users-phys.au.dk/jmd/github/concept/masters_thesis.pdf)
for which CO*N*CEPT was originally written, are available.

Please direct any questions and issues to the author: dakin(at)phys.au.dk

