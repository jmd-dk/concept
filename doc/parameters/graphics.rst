Graphics
--------
The 'graphics' parameter category contains parameters specifying the look of
2D and 3D renders, as well as of the general terminal output.

.. contents::
   :local:
   :depth: 1



------------------------------------------------------------------------------



.. _terminal_width:

``terminal_width``
..................
== =============== == =
\  **Description** \  Specifies the maximum width of the terminal output
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         80

-- --------------- -- -
\  **Elaboration** \  Long lines of terminal output from CO\ *N*\ CEPT is
                      generally wrapped into multiple lines before they are
                      printed. This is preferable to having the wrapping done
                      automatically by the terminal emulator, as these often
                      do not care about e.g. splitting sentences at word
                      boundaries. This parameter sets the width of the
                      terminal in characters, beyond which line-wrapping
                      should occur.
-- --------------- -- -
\  **Example 0**   \  Use a larger, more modern terminal width:

                      .. code-block:: python3

                         terminal_width = 120

== =============== == =



------------------------------------------------------------------------------



.. _enable_terminal_formatting:

``enable_terminal_formatting``
..............................
== =============== == =
\  **Description** \  Specifies whether to use formatted text in terminal
                      output
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         True

-- --------------- -- -
\  **Elaboration** \  By default, CO\ *N*\ CEPT makes use of various
                      formatting when printing to the terminal, e.g. bold red
                      text for warnings and errors. This can be disabled by
                      this parameter.
-- --------------- -- -
\  **Example 0**   \  Do not make use of terminal formatting:

                      .. code-block:: python3

                         enable_terminal_formatting = False

                      This can be helpful if you e.g. want to programmatically
                      search the generated log files.

                      .. note::
                         Though this disables the majority of the terminal
                         formatting, a few bits and pieces remain

== =============== == =



------------------------------------------------------------------------------



.. _render2D_options:

``render2D_options``
....................
== =============== == =
\  **Description** \  Specifications for 2D render computations
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'upstream gridsize': {
                                 'particles': 'cbrt(N)',
                                 'fluid'    : 'gridsize',
                             },
                             'global gridsize': {},
                             'terminal resolution': {},
                             'interpolation': {
                                 'default': 'PCS',
                             },
                             'deconvolve': {
                                 'default': False,
                             },
                             'interlace': {
                                 'default': False,
                             },
                             'axis': {
                                 'default': 'z',
                             },
                             'extent': {
                                 'default': (0, 0.1*boxsize),
                             },
                             'colormap': {
                                 'default': 'inferno',
                             },
                             'enhance': {
                                 'default': True,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` of several individual sub-parameters,
                      specifying details of how to compute 2D renders. All
                      sub-parameters are themselves
                      :ref:`component selections <components_and_selections>`.

                      For constructing a 2D render, one or more components (or
                      more precisely their energy densities) are first
                      interpolated onto individual *upstream* grids, possibly
                      using deconvolution and interlacing, after which they
                      are added together, producing a *global* grid. This
                      scheme is similar to the one used for potentials, except
                      here we stay in real space (though with a round-trip to
                      Fourier space if upstream grids of different sizes are
                      to be added together) and we do not interpolate anything
                      back to the particles. See the ``potential_options``
                      :ref:`parameter <potential_options>` for a walk-through
                      of the scheme. A slab of a given thickness along a
                      particular axis is then projected along this axis,
                      resulting in a greyscale image. The contrast within the
                      image is then possibly enhanced, and colours applied
                      according to a given colormap.

                      Each sub-parameter is described below:

                      * ``'upstream gridsize'``: Specifies the upstream grid
                        sizes to use for each component.

                      * ``'global gridsize'``: Specifies the global grid size
                        to use for each 2D render. Which 2D renders to compute
                        are in turn specified by the
                        ``render2D_select``
                        :ref:`parameter <render2D_select>`.

                        .. note::
                           This has no default value, as a proper global grid
                           size depends on the components within the
                           simulation. If this is not set when a 2D render is
                           to be computed, a value equal to the largest of the
                           upstream grid sizes (in use for this particular 2D
                           render) will be used.

                      * ``'terminal resolution'``: Specifies the resolution (in
                        characters) to use for the 2D terminal render, if
                        :ref:`selected <render2D_select>` as an output.

                        .. note::
                           This has no default value, as a proper terminal
                           resolution depends on the components within the
                           simulation. If this is not set when a 2D render is
                           to be computed, a value equal to the minimum of
                           ``'global gridsize'`` (in use for this particular
                           2D render) and the ``terminal_width``
                           :ref:`parameter <terminal_width>` will be used.

                      * ``'interpolation'``: Specifies the interpolation order
                        to use when interpolating particles to upstream grids.
                        The implemented interpolations are:

                        * ``'NPG'``: 'Nearest grid point', order ``1``.
                        * ``'CIC'``: 'Cloud in cell', order ``2``.
                        * ``'TSC'``: 'Triangular shaped cloud', order ``3``.
                        * ``'PCS'``: 'Piecewise cubic spline', order ``4``.

                      * ``'deconvolve'``: Specifies whether to apply
                        deconvolution for upstream particle interpolations.

                      * ``'interlace'``: Specifies whether to use interlacing
                        for upstream particle interpolations.

                      * ``'axis'``: Specifies the projection axis. Valid axes
                        are ``'x'``, ``'y'``, ``'z'``.

                      * ``'extent'``: Specifies the thickness of a slab
                        along the given axis to project, or a specific
                        interval for said slab. Only data within this slab
                        will contribute to the projected image.

                      * ``'colormap'``: The
                        `colormap <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
                        to apply to the image.

                      * ``'enhance'``: Specifies whether to apply non-linear
                        contrast enhancement to the image.

                      .. note::
                         For all sub-parameters above except
                         ``'upstream gridsize'``, the keys used within the
                         :ref:`component selection <components_and_selections>`
                         sub-\ ``dict``\ s may refer to either a single
                         component or a combination of components.

-- --------------- -- -
\  **Example 0**   \  Do the projection along the ``'x'`` axis and include a
                      quarter of the simulation box in the projection, for
                      every 2D render:

                      .. code-block:: python3

                         render2D_options = {
                             'axis': {
                                 'all'             : 'x',
                                 'all combinations': ...,
                             },
                             'extent': {
                                 'all'             : boxsize/4,
                                 'all combinations': ...,
                             },
                         }

                      .. note::
                         When only specifying a single number (the slab
                         thickness) for ``'extent'`` as above, the actual
                         extent will be ``(0, boxsize/4)``

-- --------------- -- -
\  **Example 1**   \  Apply different colormaps to the two components with
                      names/species of ``'matter'`` and ``'neutrino'``,
                      and only enhance the 2D render of the latter:

                      .. code-block:: python3

                         render2D_options = {
                             'colormap': {
                                 'matter'  : 'cool',
                                 'neutrino': 'hot',
                             },
                             'enhance': {
                                 'matter'  : False,
                                 'neutrino': True,
                             },
                         }

                      .. note::
                         The colormaps are applied for both the saved PNG and
                         for the terminal renders. The colours in the terminal
                         are however defined using global state, meaning that
                         only a single colormap can be applied within the
                         terminal at any given time. When playing back the 2D
                         terminal renders of a given component from a simulation
                         using the :doc:`play utility </utilities/play>`, the
                         proper colormap will be applied.

== =============== == =



------------------------------------------------------------------------------



``render3D_colors``
...................
== =============== == =
\  **Description** \  Specifies the colour in which to 3D render components
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {}  # use default Matpotlib colours

-- --------------- -- -
\  **Elaboration** \  This is a
                      :ref:`component selection <components_and_selections>`
                      ``dict`` for specifying the colour to use for each
                      component within 3D renders. The specified colours may
                      be
                      `any of the recognised Matplotlib colours <https://matplotlib.org/stable/gallery/color/named_colors.html>`_,
                      a 3-tuple of RGB values or a single number for
                      greyscale (valued 0–1). In all cases, a
                      default alpha (opacity) value of ``0.2`` will
                      additionally be assigned. To further set this alpha
                      yourself, define this parameter as a tuple with the
                      alpha as the last element.
-- --------------- -- -
\  **Example 0**   \  Set the component with a name/species of ``'matter'`` to
                      be 3D rendered in turquoise and the component with a
                      name/species of ``'neutrino'`` to be 3D rendered in
                      violet, with the latter being less transparent than the
                      default:

                      .. code-block:: python3

                         render3D_colors = {
                             'matter'  : 'turquoise',
                             'neutrino': ('violet', 0.3),
                         }

== =============== == =



------------------------------------------------------------------------------



``render3D_bgcolor``
....................
== =============== == =
\  **Description** \  Specifies the background colour for 3D renders
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         'black'

-- --------------- -- -
\  **Elaboration** \  All 3D renders within a simulation will have the
                      background colour as specified by this parameter, with
                      the particles or fluid cells of the rendered
                      component(s) (as well as time stamps) on top. The
                      specified colour may be
                      `any colour recognised by Matplotlib <https://matplotlib.org/stable/gallery/color/named_colors.html>`_,
                      a 3-tuple of RGB values or a single greyscale number
                      (valued 0–1).
-- --------------- -- -
\  **Example 0**   \  Use a light background colour, though not quite white:

                      .. code-block:: python3

                         render3D_bgcolor = 'ivory'

                      .. note::
                         The text (time stamps) on the 3D renders will now be
                         in black rather than their usual white, maximizing the
                         contrast

== =============== == =



------------------------------------------------------------------------------



``render3D_resolution``
.......................
== =============== == =
\  **Description** \  Specifies the size of 3D renders
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         1080

-- --------------- -- -
\  **Elaboration** \  The 3D renders are always stored as square images
                      (PNG files). This parameter specifies the height and
                      width of these images, in pixels.
-- --------------- -- -
\  **Example 0**   \  Generate high-resolution '4K' 3D renders:

                      .. code-block:: python3

                         render3D_resolution = 4096

== =============== == =

