## CO*N*CEPT <img align="right" src="https://github.com/jmd-dk/concept/workflows/build/badge.svg"/>

 &nbsp; <a href="https://github.com/jmd-dk/concept/tree/v1.0.0/">:rocket:</a> &nbsp; <a href="https://github.com/jmd-dk/concept/tree/v1.0.0/">Latest release: 1.0.0</a> \
 &nbsp; <a href="https://jmd-dk.github.io/concept/">:book:</a> &nbsp; <a href="https://jmd-dk.github.io/concept/">Documentation</a>



### Introduction

<img align="right" height="250" src="/doc/static/render2D.png"/>

CO*N*CEPT (**CO**smological ***N***-body **C**od**E** in **P**y**T**hon) is a
free and open-source simulation code for cosmological structure formation. The
code should run on any Linux system, from massively parallel computer clusters
to laptops. The code is written almost exclusively in Python, but achieves
C-like performance through code transformation using a custom
transpiler/optimizer and [Cython]. While highly competitive regarding both
performance and accuracy, CO*N*CEPT further strives for ease of use.

CO*N*CEPT is capable of simulating matter particles evolving under
self-gravity in an expanding background. It has multiple gravitational solvers
to choose from, and has adaptive time integration built in. In addition to
particles, the code is further able to evolve fluids at various levels of
non-linearity, providing the means for the inclusion of more exotic species
such as massive neutrinos, as well as for simulations consistent with general
relativistic perturbation theory. Various non-standard species &mdash; such as
decaying cold dark matter &mdash; are fully supported.

CO*N*CEPT comes with a sophisticated initial condition generator built in, and
can output snapshots, power spectra and several kinds of renders.

The [CLASS] code is fully integrated into CO*N*CEPT, supplying the needed
information for e.g. initial condition generation and
general relativistic corrections.



### Code paper
The primary paper on CO*N*CEPT is
‘[The cosmological simulation code CO*N*CEPT 1.0](https://arxiv.org/pdf/2112.01508)’. \
Cite this paper if you make use of CO*N*CEPT in a publication.



### Getting Started
To get started with CO*N*CEPT, walking through the [tutorial] is highly
recommended. That said, installation can be as simple as

```bash
bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/v1.0.0/install)
```

which installs CO*N*CEPT along with all of its dependencies into a single
directory. The installation takes a couple of hours on modern hardware.
Should the installation process end prematurely, simply rerun the installation
command and it will pick up from where it was.

To run a small sample simulation, navigate to the directory where CO*N*CEPT
is installed and invoke

```bash
./concept -p param/example_basic -n 2 --local
```

This will run the simulation defined by the provided `example_basic`
parameter file using 2 processes.

Consult the [tutorial] and the rest of the [documentation]
for further guidance.



[Cython]: https://cython.org/
[CLASS]: http://class-code.net/
[documentation]: https://jmd-dk.github.io/concept/
[tutorial]: https://jmd-dk.github.io/concept/tutorial/tutorial.html

