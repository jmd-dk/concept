class
-----
Through the 'class' utility, CO\ *N*\ CEPT can be used to generate linear
perturbations, tabulated on a common :math:`(a, k)` grid and written to an
HDF5 file. CLASS is used to solve the linear cosmology, while CO\ *N*\ CEPT
itself transforms the raw CLASS results to *N*\ -body gauge (unless some other
gauge is specified) before writing them to the file. The CLASS background is
included in the HDF5 as well. This HDF5 file is then intended to contain all
information required for cosmological simulation codes to generate general
relativistic initial conditions and corrections.

For a brief description of how to use the class utility, run

.. code-block:: bash

   ./concept -u class -h

