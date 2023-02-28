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
                                 'particles': 'cbrt(Ñ)',
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

                      For constructing a 2D render, the energy density of one
                      or more components is first interpolated onto individual
                      *upstream* grids, possibly using deconvolution and
                      interlacing, after which they are added together,
                      producing a *global* grid. This scheme is similar to the
                      one used for potentials, except that here nothing is
                      interpolate back to the particles. See the
                      ``potential_options``
                      :ref:`parameter <potential_options>` for a walk-through
                      of the scheme. A slab of a given thickness along a
                      particular axis is then projected along this axis,
                      resulting in a greyscale image. The contrast within the
                      image is then possibly enhanced, and colours applied
                      according to a given colormap.

                      Each sub-parameter is described below:

                      * ``'upstream gridsize'``: Specifies the upstream grid
                        sizes to use for each component. See the
                        ``potential_options``
                        :ref:`parameter <potential_options>` for the use of
                        the ``'Ñ'`` notation.

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
                        for upstream particle interpolations. Possible values
                        are ``'sc'`` (or ``False``) for a simple cubic lattice
                        (meaning no interlacing), ``'bcc'`` (or ``True``) for
                        a body-centered cubic lattice (meaning standard
                        interlacing involving two relatively shifted particle
                        interpolations) or ``'fcc'`` for a face-centered cubic
                        lattice (meaning interlacing involving four relatively
                        shifted particle interpolations).

                      * ``'axis'``: Specifies the projection axis. Valid axes
                        are ``'x'``, ``'y'``, ``'z'``.

                      * ``'extent'``: Specifies the thickness of a slab
                        along the given axis to project, or a specific
                        interval for said slab. Only data within this slab
                        will contribute to the projected image.

                      * ``'colormap'``: The
                        `colormap <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`__
                        to apply to the image.

                      * ``'enhance'``: Specifies whether to apply non-linear
                        contrast enhancement to the image.

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



.. _render3D_options:

``render3D_options``
....................
== =============== == =
\  **Description** \  Specifications for 3D render computations
-- --------------- -- -
\  **Default**     \  .. code-block:: python3

                         {
                             'upstream gridsize': {
                                 'particles': 'cbrt(Ñ)',
                                 'fluid'    : 'gridsize',
                             },
                             'global gridsize': {},
                             'interpolation': {
                                 'default': 'PCS',
                             },
                             'deconvolve': {
                                 'default': False,
                             },
                             'interlace': {
                                 'default': False,
                             },
                             'elevation': {
                                 'default': π/6,
                             },
                             'azimuth': {
                                 'default': -π/3,
                             },
                             'roll': {
                                 'default': 0,
                             },
                             'zoom': {
                                 'default': 1.05,
                             },
                             'projection': {
                                 'default': 'perspective',
                             },
                             'color': {},
                             'depthshade': {
                                 'default': True,
                             },
                             'enhancement': {
                                 'default': {
                                     'contrast'  : 0.5,
                                     'clip'      : lambda a: (0.45, 1 - 1e-6*a),
                                     'α'         : 1,
                                     'brightness': 0.35,
                                 },
                             },
                             'background': {
                                 'default': 'black',
                             },
                             'fontsize': {
                                 'default': '0.022*resolution',
                             },
                             'resolution': {
                                 'default': 1080,
                             },
                         }

-- --------------- -- -
\  **Elaboration** \  This is a ``dict`` of several individual sub-parameters,
                      specifying details of how to compute 3D renders. All
                      sub-parameters are themselves
                      :ref:`component selections <components_and_selections>`.

                      For constructing a 3D render, the energy density of one
                      or more components is first interpolated onto individual
                      *upstream* grids, possibly using deconvolution and
                      interlacing, after which they are added together,
                      producing a *global* grid. This scheme is similar to the
                      one used for potentials, except that here nothing is
                      interpolate back to the particles. See the
                      ``potential_options``
                      :ref:`parameter <potential_options>` for a walk-through
                      of the scheme. A 3D scatter plot is then constructed
                      from the grid positions and values, using a desired
                      colormap. Image enhancement is applied to bring out the
                      details. Several components can be rendered together,
                      in which case single-component renders are first
                      constructed and then blended together. Lastly, the 3D
                      render is given a solid background as well as labels
                      stating the current (cosmic) time :math:`t` and scale
                      factor :math:`a`.

                      Most of the sub-parameters dealing with the
                      interpolation of particles onto the grid for the
                      ``render3D_options`` parameter are identical to those of
                      the ``render2D_options``
                      :ref:`parameter <render2D_options>` and will not be
                      reiterated here. The remaining sub-parameters are
                      described below:

                      * ``'interpolation'``: Specifies the interpolation order
                        to use when interpolating particles to upstream grids,
                        similar to the ``'interpolation'`` sub-parameter of
                        the ``render2D_options``
                        :ref:`parameter <render2D_options>`:

                        * ``'NPG'``: 'Nearest grid point', order ``1``.
                        * ``'CIC'``: 'Cloud in cell', order ``2``.
                        * ``'TSC'``: 'Triangular shaped cloud', order ``3``.
                        * ``'PCS'``: 'Piecewise cubic spline', order ``4``.

                        In addition to the usual interpolation orders above,
                        one additional option exists:

                        * ``0``: Do not perform interpolation at all. That is,
                          instead of using the fixed grid cell positions with
                          the scatter plot, use the particle positions as is.
                          As no density information now exists, all particles
                          of a given component will be assigned the same
                          colour, rather than drawing individual colours from
                          a colormap. Also, only the ``'brightness'`` image
                          enhancement will be performed (see further down).

                      * ``'elevation'``, ``'azimuth'`` and ``'roll'``:
                        Together, these specify the camera viewing angles and
                        rotation. A description can be found
                        `here <https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html>`__.
                        Note however that CO\ *N*\ CEPT uses radians for all
                        angles, not degrees.

                        These can all be specified as numbers, or as
                        functions of :math:`t` or :math:`a` returning a
                        number. The latter case can be used to animate the
                        camera.

                      * ``'zoom'``: Specifies the camera zoom, either as a
                        number or a function of :math:`t` or :math:`a`.

                      * ``'projection'``: Specifies the (3D
                        :math:`\rightarrow` 2D) projection type to use. This
                        can be either of ``'orthographic'`` or
                        ``'perspective'``. In the case of the latter, a focal
                        length may optionally be specified, as in
                        ``('perspective', 1)`` (``1`` being the default).
                        Furthermore, the focal length may be given as a
                        function of :math:`t` or :math:`a`. A description of
                        the projection types can be found
                        `here <https://matplotlib.org/stable/gallery/mplot3d/projections.html>`__.

                      * ``'color'``: The
                        `colormap <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`__
                        or single
                        `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`__
                        to use for each component across all renders. In the
                        case of the former, the colormap range may further be
                        limited by also specifying a range
                        ``(c_lower, c_upper)``, with
                        :math:`c_{\text{lower}} \ge 0`,
                        :math:`c_{\text{upper}} \le 1`. It is also possible
                        to specify the range as a function (or two separate
                        functions) of :math:`t` or :math:`a`.

                        Though no defaults are explicitly stated above,
                        CO\ *N*\ CEPT in fact automatically assigns colours
                        --- to components that have not been given a colour
                        by the user --- using he following prescription:

                        0. First component with no colour specified:

                           .. code-block:: python3

                              ('inferno', lambda a: (0.10, 0.75 + 0.25*a))

                        1. Second component with no colour specified:

                           .. code-block:: python3

                              ('viridis', lambda a: (0.15, 0.85 + 0.15*a))

                        2. Any further components with no colour specified
                           will be assigned single colours taken from the
                           `default Matplotlib colour cycle <https://matplotlib.org/stable/gallery/color/color_cycle_default.html>`__.

                        .. note::
                           In the above, we see examples of how sub-parameters
                           of ``render3D_options`` can be specified as
                           functions, here functions of the scale factor
                           :math:`a`. Such functions may instead be written as
                           depending on the (cosmic) time :math:`t`.
                           CO\ *N*\ CETP will figure out which of these is
                           used through introspection and evaluate the given
                           function accordingly.

                      * ``'depthshade'``: Specifies whether to apply "depth
                        shading", meaning lowering the opacity of particles in
                        proportion to their distance from the camera, helping
                        to construct the illusion of depth within the image.

                      * ``'enhancement'``: Specifies the amounts of different
                        kinds of image enhancement to perform, specified via
                        four sub-sub-parameters:

                        * ``'contrast'``: The interpolated density values are
                          first linearly transformed to lie in the interval
                          ``[0, 1]``, then non-linearly transformed such that
                          the most common value is that specified by
                          ``'contrast'``. A value of ``'contrast'`` around
                          ``0.5`` serves to normalise the density values,
                          enhancing the contrast of the generated image. This
                          value should be in the interval :math:`[0, 1]`, or
                          :math:`-1` to not apply this enhancement at all.

                        * ``'clip'``: Specifies minimum and maximum value for
                          the densities (now in the interval ``[0, 1]``),
                          beyond which they are clipped/saturated. Having a
                          lower ``'clip'`` value significantly above ``0``
                          helps to remove dull, semi-homogeneous features,
                          whereas having an upper ``'clip'`` value somewhat
                          below ``1`` helps to highlight small, very overdense
                          regions (halos).

                        * ``'α'``: Each marker in the scatter plot is
                          assigned an α (opacity) value based on its density
                          value, but many other factors come into play. One
                          such factor is this parameter, which acts as a
                          scaling applied to all markers. This value should be
                          in the interval :math:`[0, \infty)`, or :math:`-1`
                          to not apply this enhancement at all.

                        * ``'brightness'``: After the 3D :math:`\rightarrow`
                          2D projection, the image is brightened or dimmed so
                          that its brightness matches this specification. This
                          value should be in the interval :math:`[0, 1]`, or
                          :math:`-1` to not apply this enhancement at all.

                        All four of the sub-sub-parameters above may be given
                        as functions of :math:`t` or :math:`a`.

                      * ``'background'``: Solid background
                        `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`__
                        to use. If instead you want a completely transparent
                        background, you can use a value of ``None``.

                      * ``'fontsize'``: Font size to use for the :math:`t` and
                        :math:`a` labels. Can be given as a number or as a
                        ``str`` expression involving the ``resolution``. Set
                        to ``0`` to remove the text labels.

                      * ``'resolution'``: The height and width of the square
                        render image, in pixels.

                      The above sub-parameters are
                      :ref:`component selections <components_and_selections>`
                      and may be specified for a single or multiple
                      components. However, a given component will look the
                      same across all 3D renders, meaning that certain options
                      apply directly to the components and not the render. To
                      illustrate this, consider the below specification of 3D
                      renders for a ``'matter'`` and a ``'neutrino'``
                      component:

                      .. code-block:: python3

                         render3D_select = {
                             'matter'  : True,              # single-component render
                             'neutrino': True,              # single-component render
                             ('matter', 'neutrino'): True,  # multi-component render
                         }
                         render3D_options = {
                             'color': {
                                 'matter'              : 'red',
                                 'neutrino'            : 'green',
                                 ('matter', 'neutrino'): 'blue',  # ❌
                             },
                             'background': {
                                 'matter'              : 'black',
                                 'neutrino'            : 'white',
                                 ('matter', 'neutrino'): 'silver',
                             },
                         }

                      The above specifies that we want three renders; one with
                      just ``'matter'``, one with just ``'neutrino'``, and one
                      with the two components together. Each component is
                      given a colour, but so is the two components together.
                      This additional colour specification will be ignored, as
                      colours belong to the components, not the renders.
                      Separate background colours are similarly given for all
                      three renders, but since the background colour is a
                      property of the render itself, none of the background
                      specifications will be ignored. We can even remove e.g.
                      ``'neutrino'`` from ``render3D_select`` (setting its
                      value to ``False``), leaving the ``'matter'``\ -only
                      render as well as the multi-component render. In the
                      multi-component render, the ``'neutrino'`` component
                      will still be rendered with the ``'green'`` colour.

                      The full list of sub-parameters can thus be split into
                      two categories, depending on whether they behave as
                      *render attributes* or *component attributes*,
                      where the latter should only be assigned to single
                      components:

                      * Render attributes: ``'elevation'``, ``'azimuth'``,
                        ``'roll'``, ``'zoom'``, ``'projection'``,
                        ``'depthshade'``, ``'background'``, ``'fontsize'``,
                        ``'resolution'``.

                      * Component attributes: ``'upstream gridsize'``,
                        ``'global gridsize'``, ``'interpolation'``,
                        ``'deconvolve'``, ``'interlace'``, ``'color'``,
                        ``'enhancement'``.

                      An exception from the above clean categorization is the
                      ``'brightness'`` sub-sub-parameter within the
                      ``'enhancement'`` sub-parameter, which really belongs to
                      both categories. That is, though each component has an
                      associated (relative) brightness across all renders, the
                      total brightness of multi-component renders may be
                      specified as well.

-- --------------- -- -
\  **Example 0**   \  Give the component with a name/species of ``'matter'``
                      the ``'lawngreen'`` colour, while using the ``'cool'``
                      colormap for the component with a name/species of
                      ``'neutrino'``:

                      .. code-block:: python3

                         render3D_options = {
                             'color': {
                                 'matter'  : 'lawngreen',
                                 'neutrino': 'cool',
                             },
                         }

                      We might instead specify the single colour as an RGB
                      value:

                      .. code-block:: python3

                         render3D_options = {
                             'color': {
                                 'matter'  : (124/255, 252/255, 0),
                                 'neutrino': 'cool',
                             },
                         }

                      .. tip::
                        To quickly convert any Matplotlib colour to the RGB
                        representation, you can use

                        .. code-block:: bash

                           python -c "import matplotlib; \
                               print(matplotlib.colors.ColorConverter().to_rgb('lawngreen'))"

-- --------------- -- -
\  **Example 1**   \  Use a white background:

                      .. code-block:: python3

                         render3D_options = {
                             'background': {
                                 'all': 'white',
                             },
                         }

                      .. note::
                         The text labels for the time :math:`t` and scale
                         factor :math:`a` will always be either white or
                         black; white for dark backgrounds and black for light
                         backgrounds.

                      The above ``'all'`` specification only captures
                      single-component renders. We can further apply the white
                      background to all multi-component renders:

                      .. code-block:: python3

                         render3D_options = {
                             'background': {
                                 'all'             : 'white',
                                 'all combinations': ...,
                             },
                         }

-- --------------- -- -
\  **Example 2**   \  For renders of the component with a name/species of
                      ``'matter'``, let the camera rotate around the box,
                      completing one full revolution during the simulation
                      time span:

                      .. code-block:: python3

                         render3D_options = {
                             'azimuth': {
                                 'matter': lambda a: 2*π*(a - a_begin)/(1 - a_begin),
                             },
                         }

                      Combined with frequent renders linearly spaced in
                      :math:`a`, the resulting renders can be combined into a
                      nifty animation.
-- --------------- -- -
\  **Example 3**   \  Do not interpolate particles of the component with a
                      name/species of ``'matter'`` onto a grid for the 3D
                      renders:

                      .. code-block:: python3

                         render3D_options = {
                             'interpolation': {
                                 'matter': 0,
                             },
                         }

                      Though this disables the colormap and enhancement
                      features, this has the benefit of being able to better
                      represent small structures, as the particle resolution
                      is not limited to the grid spacing.
== =============== == =

