## CO*N*CEPT

![test suite](https://github.com/jmd-dk-raspberrypi/concept/workflows/test%20suite/badge.svg)

<img align="right" height="300" src="/concept/docs/_static/render2D.png"/>

CO*N*CEPT (**CO**smological ***N***-body **C**od**E** in **P**y**T**hon)
is a free and open-source simulation code for cosmological structure
formation. The code should run on any Linux system, from massively parallel
computer clusters to laptops. The code is written almost exclusively in
Python, but achieves C-like performance thanks to Cython.

CO*N*CEPT is capable of simulating matter particles evolving under
self-gravity in an expanding background. It has multiple gravitational
solvers to choose from, and has adaptive time integration built in. In
addition to particles, fluids such as neutrinos are also implemented,
and can be simulated at various levels of non-linearity. Completely linear
perturbations in all species and the metric itself are fully supported,
allowing for simulations consistent with general relativity.

CO*N*CEPT comes with a sophisticated initial condition generator built in,
and can output snapshots, power spectra and several kinds of renders.

The [CLASS] code is fully integrated into CO*N*CEPT, supplying the needed
information for e.g. initial condition generation and
general relativistic corrections.



### Getting Started
To get started with CO*N*CEPT, walking through the [tutorial] is highly
recommended. That said, installation can be as simple as

```bash
bash <(wget -O- https://raw.githubusercontent.com/jmd-dk/concept/master/installer)
```

which installs CO*N*CEPT along with all of its dependencies into a single
directory. The installation takes about an hour or two on modern hardware.
Should the installation process end prematurely, simply rerun the installation
command and it will pick up from where it was.

To run a small sample simulation, navigate to the `concept` directory
and invoke

```bash
./concept -p params/example_params -n 2 --local
```

This will run the simulation defined by the provided `example_params`
parameter file using 2 processes.

Consult the [tutorial] and the rest of the [documentation]
for further guidance.



[CLASS]: http://class-code.net/
[documentation]: https://jmd-dk.github.io/concept/
[tutorial]: https://jmd-dk.github.io/concept/tutorial/tutorial.html

