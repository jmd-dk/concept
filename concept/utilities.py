# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2018 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see http://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports.
# Functions from the 'analysis' and 'graphics' modules are not dumped
# directly into the global namespace of this module, as functions with
# identical names are defined here.
cimport('import analysis')
cimport('from analysis import measure')
cimport('from communication import domain_subdivisions, exchange, partition, smart_mpi')
cimport('import graphics')
cimport('from integration import initiate_time, remove_doppelgängers')
cimport('from linear import compute_cosmo, compute_transfer, get_default_k_parameters')
cimport('from mesh import CIC_particles2fluid')
cimport('from snapshot import get_snapshot_type, snapshot_extensions')
cimport('import species')
cimport('from species import get_representation')
cimport('from snapshot import get_initial_conditions, load, save')



# Entry point of this module.
# Call this function to perform a special operation,
# defined in the special_params dict.
@cython.header()
def delegate():
    eval(special_params['special'] + '()')

# Context manager which temporaly sets the
# allow_similarly_named_components flag in the species module to True,
# allowing for the initialization of many component instances
# with the same name.
@contextlib.contextmanager
def allow_similarly_named_components():
    # Backup the current state of the flag
    allowed = species.allow_similarly_named_components
    # Make sure that it is allowed to instantiate
    # multiple components with the same name.
    species.allow_similarly_named_components = True
    # Yield control back to the caller
    yield
    # Reset flag
    species.allow_similarly_named_components = allowed

# Function which convert all snapshots in the
# special_params['snapshot_filenames'] parameter to the snapshot type
# given in the snapshot_type parameter.
@cython.pheader(# Locals
                N_vacuum='Py_ssize_t',
                a='double',
                component='Component',
                dim='int',
                ext=str,
                index='int',
                snapshot=object,
                snapshot_filename=str,
                converted_snapshot_filename=str,
                params=dict,
                attribute_str=str,
                attributes=object,  # collections.defaultdict
                attribute=str,
                key=str,
                value=object,  # double, str or NoneType
                mass='double',
                name=str,
                names=list,
                names_lower=list,
                original_mass='double',
                original_representation=str,
                rel_tol='double',
                unit_str=str,
                σmom_fluid='double[::1]',
                σmom_particles='double[::1]',
                Σmass_fluid='double',
                Σmass_particles='double',
                Σmom_fluid='double[::1]',
                Σmom_particles='double[::1]',
                )
def convert():
    """This function will convert the snapshot given in the
    special_params['snapshot_filename'] parameter to the type
    specified by the snapshot_type parameter.
    If special_params['attributes'] is not empty, it contains
    information about global parameters and individual componen
    attributes which should be changed.
    """
    # Create dict of global parameters (params) and (default)dict of
    # component attributes (attributes) from the passed attributes.
    params = {}
    attributes = collections.defaultdict(dict)
    for attribute_str in special_params['attributes']:
        index = attribute_str.index('=')
        key = asciify(attribute_str[:index].strip())
        # Numerical value, possibly with units
        value = eval_unit(attribute_str[(index + 1):], fail_on_error=False)
        if value is None:
            # String value
            value = attribute_str[(index + 1):].strip()
        if '.' in key:
            index = key.index('.')
            if index + 1 < len(key) and key[index + 1] not in '0123456789':
                # Component attribute
                name, attribute = key.split('.')
                attributes[name.strip()][attribute.strip()] = value
            else:
                # Global parameter
                params[key] = value
        else:
            # Global parameter
            params[key] = value
    # The filename of the snapshot to read in
    snapshot_filename = special_params['snapshot_filename']
    # Read snapshot on disk into the requested type
    snapshot = load(
        snapshot_filename,
        compare_params=True,  # Warn the user of non-matching params
        do_exchange=False,    # Exchanges happen later, if needed
        as_if=snapshot_type,
    )
    # Some of the functions used later use the value of universals.a.
    # Set this equal to the scale factor value in the snapshot.
    # In the end of this function, the original value of
    # universals.a will be reassigned.
    a = universals.a
    universals.a = snapshot.params['a']
    # Warn the user of specified changes to component attributes
    # of non-existing components. Allow for components written in a
    # different case.
    names = [component.name for component in snapshot.components]
    names_lower = [name.lower() for name in names]
    for name in dict(attributes):  # New dict neeed as keys are removed during iteration
        if name not in names:
            # Specified component name not present.
            # Maybe the problem is due to lower/upper case.
            if name.lower() in names_lower:
                # The component name is written in a different case.
                # Move specified attributes over to the properly
                # written name and delete the wrongly written name key
                # from the attributes.
                attributes[names[names_lower.index(name.lower())]].update(attributes[name])
                del attributes[name]
            else:
                masterwarn('The following attributes are specified for the "{}" component, '
                           'which does not exist:\n{}'.format(name, attributes[name]))
    # Overwrite parameters in the snapshot with those from the
    # parameter file (those which are currently loaded as globals).
    # If paremters are passed directly, these should take precedence
    # over those from the parameter file.
    snapshot.populate(snapshot.components, params)
    # Edit individual components if component attributes are passed
    for component in snapshot.components:
        # The (original) name of this component
        name = component.name
        # Backup of original representation and mass
        original_representation = component.representation
        original_mass = component.mass
        # Edit component attributes
        for key, val in attributes[name].items():
            if key in ('w', 'eos_w'):
                # An equation of state parameter w is given.
                # As this is not just a single attribute, we need to
                # handle this case on its own.
                component.initialize_w(val)
                continue
            if not hasattr(component, key):
                # A non-existing attribute was specified. As this is
                # nonsensical and leads to an error in compiled mode
                # but not in pure Python mode, do an explicit abort.
                abort('The following non-existing attribute was specified for\n'
                      'the "{}" component: {}'.format(component.name, key))
            setattr(component, key, val)
        component.representation = get_representation(component.species)
        # Apply particles --> fluid convertion, if necessary
        if original_representation == 'particles' and component.representation == 'fluid':
            # To do the convertion, the particles need to be
            # distributed according to which domain they are in.
            component.representation = 'particles'
            exchange(component)
            # The total particle mass and momentum
            Σmass_particles = component.mass*component.N
            Σmom_particles, σmom_particles = measure(component, 'momentum')
            # Done treating component as particles.
            # Reassign the fluid representation
            component.representation = 'fluid'
            # The mass attribute shall now be the average mass of a
            # fluid element. Since the total mass of the component
            # should be the same as before, the mass and the gridsize
            # are related, as
            # mass = totmass/gridsize**3
            #      = N*original_mass/gridsize**3.
            # If either mass or gridsize is given by the user,
            # use this to determine the other.
            if 'gridsize' in attributes[name] and 'mass' not in attributes[name]:
                component.mass *= float(component.N)/component.gridsize**3
            elif 'mass' in attributes[name] and 'gridsize' not in attributes[name]:
                component.gridsize = int(round(cbrt(original_mass/component.mass*component.N)))
                Σmass_fluid = component.mass*component.gridsize**3
                if not isclose(Σmass_particles, Σmass_fluid, 1e-6):
                    if Σmass_fluid > Σmass_particles:
                        masterwarn('The specified mass for the "{}" component\n'
                                   'leads to a relative increase of {:.9g}\n'
                                   'for the total mass of this component.'
                                   'Note that for fluids, the specified mass should be\n'
                                   'the average mass of a fluid element.'
                                   .format(component.name, Σmass_fluid/Σmass_particles - 1))
                    else:
                        masterwarn('The specified mass for the "{}" component\n'
                                   'leads to a relative decrease of {:.9g}\n'
                                   'for the total mass of this component.\n'
                                   'Note that for fluids, the specified mass should be\n'
                                   'the average mass of a fluid element.'
                                   .format(component.name, 1 - Σmass_fluid/Σmass_particles))
            elif 'gridsize' not in attributes[name] and 'mass' not in attributes[name]:
                # If neither the gridsize nor the mass is specified,
                # the number of gridpoints in the fluid
                # representation should equal the number of
                # particles in the particle representation.
                component.gridsize = int(round(cbrt(component.N)))
                # If component.N is not a cube number, the number
                # of fluid elements will not be exactly equal to the
                # number of particles. Adjust the mass accordingly.
                component.mass *= component.N/component.gridsize**3
            # CIC-interpolate particle data to fluid data. Temporarily
            # let the mass attribute be the original particle mass.
            mass = component.mass
            component.mass = original_mass
            N_vacuum = CIC_particles2fluid(component)
            component.mass = mass
            # Measure the total mass and momentum of the fluid
            Σmass_fluid = measure(component, 'mass')
            Σmom_fluid, σmom_fluid  = measure(component, 'momentum')
            # Warn the user about changes in the total mass
            rel_tol = 1e-9
            if not isclose(Σmass_particles, Σmass_fluid, rel_tol):
                masterwarn('Interpolation of particles to fluid did not preserve mass:\n'
                           'Total particle mass: {{:.{num}g}}\n'
                           'Total fluid mass:    {{:.{num}g}}'
                           .format(num=int(ceil(-log10(rel_tol))))
                           .format(Σmass_particles, Σmass_fluid)
                     )
            # Warn the user about changes in the
            # total momentum after interpolation.
            if not all([isclose(Σmom_particles[dim], Σmom_fluid[dim],
                                rel_tol=rel_tol,
                                abs_tol=rel_tol*component.gridsize**3*(+ σmom_particles[dim]
                                                                       + σmom_fluid[dim]))
                        for dim in range(3)]):
                unit_str = '{} {} {}⁻¹'.format(unit_mass, unit_length, unit_time)
                masterwarn('Interpolation of particles to fluid did not preserve momentum:\n'
                           'Total particle momentum: [{{:.{num}g}}, {{:.{num}g}}, {{:.{num}g}}] {{}}\n'
                           'Total fluid momentum:    [{{:.{num}g}}, {{:.{num}g}}, {{:.{num}g}}] {{}}'
                           .format(num=int(ceil(-log10(rel_tol))))
                           .format(*Σmom_particles, unit_str,
                                   *Σmom_fluid,     unit_str)
                     )
            # If the particle number equal the number of grid points
            # and every fluid elements was interpolated to,
            # then (roughly) one grid point corresponds to one particle.
            # In this case, the convertion from particles to fluid should
            # preserve the momentum distribution. For this particular case,
            # warn the user about changes in the
            # standard deviation of the momentum after interpolation.
            if component.gridsize**3 == component.N and N_vacuum == 0:
                rel_tol = 0.6
                if not all([isclose(σmom_particles[dim], σmom_fluid[dim], rel_tol)
                            for dim in range(3)]):
                    unit_str = '{} {} {}⁻¹'.format(unit_mass, unit_length, unit_time)
                    masterwarn('Interpolation of particles to fluid did not preserve momentum spread:\n'
                               'σ(particle momentum): [{{:.{num}e}}, {{:.{num}e}}, {{:.{num}e}}] {{}}\n'
                               'σ(fluid momentum):    [{{:.{num}e}}, {{:.{num}e}}, {{:.{num}e}}] {{}}'
                               .format(num=(1 + int(ceil(-log10(rel_tol)))))
                               .format(*σmom_particles, unit_str,
                                       *σmom_fluid,     unit_str)
                         )
        elif original_representation == 'fluid' and component.representation == 'particles':
            abort('Cannot convert fluid to particles')
    # Remove original file extension
    # (the correct extension will be added by the save function).
    converted_snapshot_filename = snapshot_filename
    for ext in snapshot_extensions:
        if converted_snapshot_filename.endswith(ext):
            index = len(converted_snapshot_filename) - len(ext)
            converted_snapshot_filename = converted_snapshot_filename[:index]
            break
    # Append string to the filename,
    # signalling that this is the output of the conversion.
    converted_snapshot_filename += '_converted'
    # Save the converted snapshot
    snapshot.save(converted_snapshot_filename)
    # Reassign the original value of universals.a
    universals.a = a

# Function for finding all snapshots in a directory
@cython.pheader(# Arguments
                path=str,
                # Locals
                filenames=list,
                msg=str,
                snapshot_filenames=list,
                returns=list,
                )
def locate_snapshots(path):
    # Get all files from the path
    if master and not os.path.exists(path):
        msg = 'Path "{}" does not exist!'.format(path)
        abort(msg)
    if os.path.isdir(path):
        filenames = [os.path.join(path, filename)
                     for filename in os.listdir(path)
                     if os.path.isfile(os.path.join(path, filename))]
    else:
        filenames = [path]
    # Only use snapshots
    snapshot_filenames = [filename for filename in filenames if get_snapshot_type(filename)]
    # Abort if none of the files where snapshots
    if master and not snapshot_filenames:
        if os.path.isdir(path):
            msg = 'The directory "{}" does not contain any snapshots.'.format(path)
        else:
            msg = 'The file "{}" is not a valid snapshot.'.format(path)
        abort(msg)
    return snapshot_filenames

# Function that produces a power spectrum of the file
# specified by the special_params['snapshot_filename'] parameter.
@cython.pheader(# Locals
                basename=str,
                index='int',
                ext=str,
                output_dir=str,
                output_filename=str,
                snapshot=object,
                snapshot_filename=str,
                )
def powerspec():
    # Initial cosmic time universals.t
    # and scale factor a(universals.t) = universals.a.
    initiate_time()
    # Extract the snapshot filename
    snapshot_filename = special_params['snapshot_filename']
    # Read in the snapshot
    snapshot = load(snapshot_filename, compare_params=False)
    # Construct output filename based on the snapshot filename.
    # Importantly, remove any file extension signalling a snapshot.
    output_dir, basename = os.path.split(snapshot_filename)
    for ext in snapshot_extensions:
        if basename.endswith(ext):
            index = len(basename) - len(ext)
            basename = basename[:index]
            break
    output_filename = '{}/{}{}{}'.format(output_dir,
                                         output_bases['powerspec'],
                                         '_' if output_bases['powerspec'] else '',
                                         basename)
    # Prepend 'powerspec_' to filename if it
    # is identical to the snapshot filename.
    if output_filename == snapshot_filename:
        output_filename = '{}/powerspec_{}'.format(output_dir, basename)
    # Produce power spectrum of the snapshot
    analysis.powerspec(snapshot.components, output_filename)

# Function which produces a 3D render of the file
# specified by the special_params['snapshot_filename'] parameter.
@cython.pheader(# Locals
                basename=str,
                index='int',
                ext=str,
                output_dir=str,
                output_filename=str,
                snapshot=object,
                snapshot_filename=str,
                )
def render3D():
    # Initial cosmic time universals.t
    # and scale factor a(universals.t) = universals.a.
    initiate_time()
    # Extract the snapshot filename
    snapshot_filename = special_params['snapshot_filename']
    # Read in the snapshot
    snapshot = load(snapshot_filename, compare_params=True)
    # Construct output filename based on the snapshot filename.
    # Importantly, remove any file extension signalling a snapshot.
    output_dir, basename = os.path.split(snapshot_filename)
    for ext in snapshot_extensions:
        if basename.endswith(ext):
            index = len(basename) - len(ext)
            basename = basename[:index]
            break
    output_filename = '{}/{}{}{}'.format(output_dir,
                                         output_bases['render3D'],
                                         '_' if output_bases['render3D'] else '',
                                         basename)
    # Attach missing extension to filename
    if not output_filename.endswith('.png'):
        output_filename += '.png'
    # Prepend 'render3D_' to filename if it
    # is identical to the snapshot filename.
    if output_filename == snapshot_filename:
        output_filename = '{}/render3D_{}'.format(output_dir, basename)
    # Render the snapshot
    graphics.render3D(snapshot.components, output_filename,
                    True, '.renders3D_{}'.format(basename))

# Function for printing all informations within a snapshot
@cython.pheader(# Locals
                alt_str=str,
                component='Component',
                h='double',
                heading=str,
                index='int',
                eos_info=str,
                ext=str,
                param_num='int',
                parameter_filename=str,
                params=dict,
                path=str,
                paths=list,
                snapshot=object,
                snapshot_filenames=list,
                snapshot_type=str,
                unit='double',
                value='double',
                Σmom='double[::1]',
                σmom='double[::1]',
                )
def info():
    # Extract the paths to snapshot(s)
    paths = special_params['paths']
    # Get list of all snapshots
    snapshot_filenames = [snapshot_filename for path in paths
                                            for snapshot_filename in locate_snapshots(path)]
    # Print out information about each snapshot
    for snapshot_filename in snapshot_filenames:
        # Load parameters from the snapshot
        with allow_similarly_named_components():
            snapshot = load(snapshot_filename, compare_params=False,
                                               only_params=(not special_params['stats']),
                                               do_exchange=False,
                                               )
        params = snapshot.params
        snapshot_type = get_snapshot_type(snapshot_filename)
        # If a parameter file should be generated from the snapshot,
        # print out the content which should be placed in parameter file
        # to stdout and directly to a new parameter file.
        # The value of special_params['generate params'] is either a
        # directory path where the parameter file should be placed,
        # or False if no parameter file should be generated.
        if special_params['generate params']:
            # Make sure that the params directory exist
            if master:
                os.makedirs(special_params['generate params'], exist_ok=True)
            # The filename of the new parameter file
            parameter_filename = '{}/{}'.format(special_params['generate params'],
                                                os.path.basename(snapshot_filename))
            for ext in snapshot_extensions:
                if parameter_filename.endswith(ext):
                    index = len(parameter_filename) - len(ext)
                    parameter_filename = parameter_filename[:index]
                    break
            parameter_filename += '.params'
            # Do not overwrite an existing parameter file.
            # Append increasing number
            # until a non-existing file is reached.
            if os.path.isfile(parameter_filename):
                param_num = 0
                while os.path.isfile(parameter_filename + str(param_num)):
                    param_num += 1
                parameter_filename += str(param_num)
            # As the following printed information should be parsable,
            # wrapping is deactivated on every call to masterprint.
            # Do not edit the text in the heading,
            # as it is grepped for by several of the Bash utilities.
            heading = '\nParameters of "{}"'.format(sensible_path(snapshot_filename))
            masterprint(terminal.bold(heading), wrap=False)
            with open(parameter_filename, 'w') as pfile:
                masterprint('# Auto-generated parameter file for the snapshot\n# "{}"\n'
                            .format(snapshot_filename), file=pfile, wrap=False)
                # Loop over stdout and the new parameter file
                for file in (sys.stdout, pfile):
                    masterprint('# Input/output', file=file, wrap=False)
                    masterprint("initial_conditions = '{}'".format(sensible_path(snapshot_filename)),
                                file=file, wrap=False)
                    if hasattr(snapshot, 'units'):
                        masterprint('# System of units', file=file, wrap=False)
                        masterprint("unit_length = '{}'".format(snapshot.units['length']),
                                    file=file, wrap=False)
                        masterprint("unit_time = '{}'".format(snapshot.units['time']),
                                    file=file, wrap=False)
                        masterprint("unit_mass = '{}'".format(snapshot.units['mass']),
                                    file=file, wrap=False)
                    masterprint('# Numerical parameters', file=file, wrap=False)
                    unit = 100*units.km/(units.s*units.Mpc)
                    h = params['H0']/unit
                    value = params['boxsize']*h
                    if isint(value):
                        masterprint('boxsize = {}/{:.12g}*{}'.format(int(round(value)),
                                                                     h,
                                                                     unit_length),
                                    file=file, wrap=False)
                    else:
                        masterprint('boxsize = {:.12g}*{}'.format(params['boxsize'], unit_length),
                                    file=file, wrap=False)
                    masterprint('# Cosmological parameters', file=file, wrap=False)
                    unit = units.km/(units.s*units.Mpc)
                    masterprint('H0 = {:.12g}*km/(s*Mpc)'.format(params['H0']/unit),
                                file=file, wrap=False)
                    if enable_Hubble:
                        masterprint('a_begin = {:.12g}'.format(params['a']), file=file, wrap=False)
                    if snapshot_type == 'standard':
                        masterprint('Ωcdm = {:.12g}'.format(params['Ωcdm']), file=file, wrap=False)
                        masterprint('Ωb = {:.12g}'.format(params['Ωb']), file=file, wrap=False)
            # Do not edit the printed text below,
            # as it is grepped for by several of the Bash utilities.
            masterprint('\nThe above parameters have been written to "{}"'
                        .format(parameter_filename),
                        wrap=False)
            # Done writing out parameters. The code below which prints
            # out information about the snapshot should not be reached.
            continue
        # Print out heading stating the filename
        heading = '\nInformation about "{}"'.format(sensible_path(snapshot_filename))
        masterprint(terminal.bold(heading))
        # Print out snapshot type
        masterprint('{:<20} {}'.format('Snapshot type', snapshot_type))
        # Print out unit system for standard snapshots
        if snapshot_type == 'standard':
            masterprint('{:<20} {}'.format('unit_length', snapshot.units['length']))
            masterprint('{:<20} {}'.format('unit_time',   snapshot.units['time']))
            # The mass is typically some large number written in
            # exponential notation. Print it out nicely.
            mass_num = eval_unit(snapshot.units['mass'])/units.m_sun
            mass_basicunit = 'm☉'
            mass_num_fmt = significant_figures(float(mass_num), 6, fmt='unicode', incl_zeros=False)
            masterprint('{:<20} {}'.format('unit_mass', '{} {}'.format(mass_num_fmt,
                                                                       mass_basicunit)))
        # Print out global parameters
        unit = units.km/(units.s*units.Mpc)
        masterprint('{:<20} {:.12g} km s⁻¹ Mpc⁻¹'.format('H0', params['H0']/unit))
        masterprint('{:<20} {:.12g}'.format('a', params['a']))
        # The boxsize should also be printed as boxsize/h, if integer
        unit = 100*units.km/(units.s*units.Mpc)
        h = params['H0']/unit
        value = params['boxsize']*h
        alt_str = ''
        if isint(value) and not isint(params['boxsize']):
            alt_str = ' = {:.12g}/{:.12g} {}'.format(int(round(value)), h, unit_length)
        masterprint('{:<20} {:.12g} {}{}'.format('boxsize', params['boxsize'], unit_length, alt_str))
        # Print out the cosmological density parameters Ωcdm and Ωb.
        # These are only present in the standard snapshots. In gadget2
        # snapshots, instead we have ΩΛ and Ωm. We do not print these
        # out here, as these will be printed as part
        # of the GADGET header.
        if snapshot_type == 'standard':
            masterprint('{:<20} {:.12g}'.format(unicode('Ωcdm'), params['Ωcdm']))
            masterprint('{:<20} {:.12g}'.format(unicode('Ωb'), params['Ωb']))
        # Print out component information
        for component in snapshot.components:
            masterprint('{}:'.format(component.name))
            masterprint('{:<16} {}'.format('species', component.species), indent=4)
            # Representation-specific attributes
            if component.representation == 'particles':
                # Print the particle number N
                if isint(ℝ[cbrt(component.N)]):
                    # When N is cube number, print also the cube root
                    masterprint('{:<16} {} = {:.0f}³'.format('N',
                                                             component.N,
                                                             ℝ[cbrt(component.N)]),
                                indent=4)
                else:
                    masterprint('{:<16} {}'.format('N', component.N), indent=4)
                masterprint('{:<16} {} m☉'.format('mass',
                                                  significant_figures(component.mass/units.m_sun,
                                                                      6,
                                                                      fmt='unicode',
                                                                      incl_zeros=False)
                                                  ),
                            indent=4)
            elif component.representation == 'fluid':
                masterprint('{:<16} {}'.format('gridsize', component.gridsize), indent=4)
                masterprint(
                    '{:<16} {}'.format('boltzmann_order', component.boltzmann_order),
                    indent=4,
                )
                if component.w_type == 'constant':
                    eos_info = significant_figures(component.w_constant, 6,
                                                   fmt='unicode', incl_zeros=False,
                                                   )
                elif component.w_type == 'tabulated (t)':
                    eos_info = 'tabulated w(t)'
                elif component.w_type == 'tabulated (a)':
                    eos_info = 'tabulated w(a)'
                elif component.w_type == 'expression':
                    eos_info = component.w_expression
                else:
                    eos_info = 'not understood'
                masterprint('{:<16} {}'.format('w', eos_info), indent=4)
            # Component statistics
            if special_params['stats']:
                Σmom, σmom = measure(component, 'momentum')
                masterprint('{:<16} [{}, {}, {}] {}'.format('momentum sum',
                                                            *significant_figures(asarray(Σmom)/units.m_sun,
                                                                                 6,
                                                                                 fmt='unicode',
                                                                                 scientific=True),
                                                            'm☉ {} {}⁻¹'.format(unit_length, unit_time)),
                            indent=4)
                masterprint('{:<16} [{}, {}, {}] {}'.format('momentum spread',
                                                            *significant_figures(asarray(σmom)/units.m_sun,
                                                                                 6,
                                                                                 fmt='unicode',
                                                                                 scientific=True),
                                                            'm☉ {} {}⁻¹'.format(unit_length, unit_time)),
                            indent=4)
        # Print out GADGET header for GADGET2 snapshots
        if snapshot_type == 'gadget2':
            masterprint('GADGET header:')
            for key, val in params['header'].items():
                masterprint('{:<16} {}'.format(key, val), indent=4)
        # End of information
        masterprint('')

# Function that saves the processed CLASS background
# and perturbations to an hdf5 file.
@cython.pheader(
    # Locals
    a='double',
    a_first='double',
    a_min='double',
    a_size='Py_ssize_t',
    a_start='Py_ssize_t',
    a_values='double[::1]',
    all_a_values='double[::1]',
    arr='double[::1]',
    class_species=str,
    complete_transfer='double[:, ::1]',
    component='Component',
    component_variables=dict,
    components=list,
    filename=str,
    i='Py_ssize_t',
    index='Py_ssize_t',
    k_gridsize='Py_ssize_t',
    k_magnitudes='double[::1]',
    k_max='double',
    k_min='double',
    max_a_values='double',
    max_a_values_str=str,
    perturbations=dict,
    size='Py_ssize_t',
    transfer='double[:, ::1]',
    transfer_of_k='double[::1]',
    var_name=str,
    var_name_ascii=str,
    variable_specifications=list,
)
def CLASS():
    # Suppress warning about the total energy density of the components
    # being too high, as the components are not used to perform a
    # simulation anyway.
    suppress_output['err'].add('the energy density of the components add up to')
    # Initialize components, but do not realize them
    initiate_time()
    components = get_initial_conditions(do_realization=False)
    # Do CLASS computation
    k_min, k_max, k_gridsize = get_default_k_parameters(φ_gridsize)
    cosmoresults = compute_cosmo(
        k_min,
        k_max,
        k_gridsize,
        class_call_reason='in order to get perturbations ',
    )
    cosmoresults.load_everything()
    k_magnitudes = cosmoresults.k_magnitudes
    # Store all CLASS parameters, the unit system in use
    # and the processed background in a new hdf5 file.
    if master:
        filename = output_dirs['powerspec'] + '/class_processed.hdf5'
        os.makedirs(output_dirs['powerspec'], exist_ok=True)
        with open_hdf5(filename, mode='w') as hdf5_file:
            # Store CLASS parameters as attributes on the
            # "class_params" group. If you need to know furthe
            # parameters used by CLASS (i.e. default values),
            # you should specify these explicitly in the class_params
            # user parameter. No unit convertion will take place.
            params_h5 = hdf5_file.require_group('class_params')
            for key, val in cosmoresults.params.items():
                key = key.replace('/', '__per__')
                params_h5.attrs[key] = bytes(str(val), encoding='ascii')
            # Store the unit system in use. This is important as all
            # variables stored below will be stored in these units.
            units_h5 = hdf5_file.require_group('units')
            for unit_name, unit_val in {
                'unit time': unit_time,
                'unit length': unit_length,
                'unit mass': unit_mass,
                }.items():
                try:
                    units_h5.attrs[unit_name] = bytes(unit_val, encoding='ascii')
                except UnicodeEncodeError:
                    units_h5.attrs[unit_name] = unit_val
            # Store background variables present in the
            # cosmoresults.background dict. Here we convert to the
            # current unit system in use.
            background_h5 = hdf5_file.require_group('background')
            for key, arr in cosmoresults.background.items():
                key = key.replace('/', '__per__')
                if key.startswith('(.)rho_'):
                    # The "(.)" notation is CLASS syntax reminding us
                    # that we need to multiply by 3/(8πG).
                    # We do this and convert to the proper units
                    # using the ρ_bar method.
                    class_species = key.split('(.)rho_')[1]
                    arr = cosmoresults.ρ_bar(cosmoresults.background['a'], class_species)
                    # Now, the "(.)" prefix should be dropped
                    key = key.lstrip('(.)')
                elif key.startswith('(.)p'):
                    # The "(.)" notation is CLASS syntax reminding us
                    # that we need to multiply by 3/(8πG).
                    # We do this and convert to the proper units
                    # using the P_bar method.
                    class_species = key.split('(.)p_')[1]
                    arr = cosmoresults.P_bar(cosmoresults.background['a'], class_species)
                    # Now, the "(.)" prefix should be dropped
                    key = key.lstrip('(.)')
                elif key in {'a', 'z'}:
                    # Unitless
                    pass
                elif key == 'proper time [Gyr]':
                    arr = asarray(arr)*units.Gyr
                    key = 't'
                elif key == 'conf. time [Mpc]':
                    arr = asarray(arr)*(units.Mpc/light_speed)
                    key = 'tau'
                elif key in {'H [1/Mpc]', 'H [1__per__Mpc]'}:
                    arr = asarray(arr)*(light_speed/units.Mpc)
                    key = 'H'
                elif key == 'gr.fac. D':
                    # Unitless
                    key = 'D1'
                elif key == 'gr.fac. f':
                    # Unitless
                    key = 'f1'
                else:
                    masterwarn(
                        f'Unrecognized CLASS background variable "{key}". '
                        f'Unit convertion could not be carried out.'
                    )
                dset = background_h5.create_dataset(key, (arr.shape[0],), dtype=C2np['double'])
                dset[:] = arr
            # Also store ρ_bar and P_bar for the class species of the
            # components, if these are combination species.
            for component in components:
                if '+' not in component.class_species:
                    continue
                key = f'rho_{component.class_species}'
                arr = cosmoresults.ρ_bar(cosmoresults.background['a'], component)
                dset = background_h5.create_dataset(key, (arr.shape[0],), dtype=C2np['double'])
                dset[:] = arr
                key = f'p_{component.class_species}'
                arr = cosmoresults.P_bar(cosmoresults.background['a'], component)
                dset = background_h5.create_dataset(key, (arr.shape[0],), dtype=C2np['double'])
                dset[:] = arr
    # Create dict mapping components to lists of
    # (variable, specific_multi_index, var_name), specifying which
    # transfer functions to store in the hdf5 file.
    component_variables = {}
    for component in components:
        # Create list of (variable, specific_multi_index, var_name)
        variable_specifications = [(0, None, 'δ')]
        if component.representation == 'particles':
            if not component.realization_options['mom'].get('velocitiesfromdisplacements', False):
                variable_specifications.append((1, None, 'θ'))
        elif component.representation == 'fluid':
            if component.boltzmann_order > 1 or (
                component.boltzmann_order == 1 and component.boltzmann_closure == 'class'):
                variable_specifications.append((1, None, 'θ'))
            if component.boltzmann_order > 2 or (
                component.boltzmann_order == 2 and component.boltzmann_closure == 'class'):
                variable_specifications.append((2, (0, 0), 'σ'))
            if not component.approximations['P=wρ']:
                variable_specifications.append((2, 'trace', 'δP'))
        component_variables[component] = variable_specifications
    # Construct array of a values at which to tabulate the
    # transfer functions. This is done by merging all of the a arrays
    # for the individual k modes, ensuring that all perturbations will
    # be smooth on this common grid of a values.
    a_min = 0
    size = 0
    for perturbations in cosmoresults.perturbations:
        a_values = perturbations['a']
        a_first = a_values[0]
        if a_first > a_min:
            a_min = a_first
        size += a_values.shape[0]
    all_a_values = np.empty(size, dtype=C2np['double'])
    index = 0
    for perturbations in cosmoresults.perturbations:
        a_values = perturbations['a']
        size = a_values.shape[0]
        all_a_values[index:index+size] = a_values
        index += size
    asarray(all_a_values).sort()
    for index in range(all_a_values.shape[0]):
        if all_a_values[index] == a_min:
            all_a_values = all_a_values[index:]
            break
    all_a_values, _ = remove_doppelgängers(all_a_values, all_a_values, rel_tol=0.5)
    all_a_values = asarray(all_a_values).copy()
    # If too many a values are given, evenly select the amount given by
    # the "max_a_values" utility argument.
    max_a_values_str = str(special_params['max_a_values'])
    if max_a_values_str in {'inf', 'np.inf', 'numpy.inf'}:
        max_a_values = ထ
    else:
        try:
            max_a_values = float(max_a_values_str)
        except:
            try:
                max_a_values = float(eval(max_a_values_str))
            except:
                abort(f'Could not interpret max_a_values = {max_a_values_str}')
    if all_a_values.shape[0] > max_a_values:
        step = float(all_a_values.shape[0])/(max_a_values - 1)
        all_a_values_selected = np.empty(int(max_a_values), dtype=C2np['double'])
        for i in range(int(max_a_values) - 1):
            all_a_values_selected[i] = all_a_values[cast(int(i*step), 'Py_ssize_t')]
        all_a_values_selected[int(max_a_values) - 1] = all_a_values[all_a_values.shape[0] - 1]
        all_a_values = all_a_values_selected
    # Store the a and k values at which the perturbations are tabulated,
    # as well as the primordial parameters needed to convert transfer
    # functions into power spectra and the gauge.
    gauge = special_params['gauge']
    if master:
        with open_hdf5(filename, mode='a') as hdf5_file:
            perturbations_h5 = hdf5_file.require_group('perturbations')
            dset = perturbations_h5.create_dataset(
                'a', (all_a_values.shape[0], ), dtype=C2np['double'],
            )
            dset[:] = all_a_values
            dset = perturbations_h5.create_dataset(
                'k', (k_magnitudes.shape[0], ), dtype=C2np['double'],
            )
            dset[:] = k_magnitudes
            for primordial_key in {'A_s', 'n_s', 'alpha_s', 'k_pivot'}:
                perturbations_h5.attrs[primordial_key] = getattr(cosmoresults, primordial_key)
            perturbations_h5.attrs['gauge'] = bytes(
                gauge.replace('-', '').lower(), encoding='ascii')
    # Get transfer functions of k for each a.
    # Partition the work across the a values.
    # Collect the results into the 2D transfer array.
    # Once a complete transfer function (given at all a and k values)
    # has been constructed, it is saved to disk and possibly plotted.
    # For the next transfer function, we reuse the same 2D arrays,
    # as all transfer functions are tabulated at the same a and k.
    a_start, a_size = partition(all_a_values.shape[0])
    transfer = np.empty((a_size, k_gridsize), dtype=C2np['double'])
    if master:
        complete_transfer = np.empty((all_a_values.shape[0], k_gridsize), dtype=C2np['double'])
    for component, variable_specifications in component_variables.items():
        for variable, specific_multi_index, var_name in variable_specifications:
            for i in range(a_size):
                a = all_a_values[a_start + i]
                transfer_of_k, _ = compute_transfer(
                    component, variable, k_min, k_max,
                    k_gridsize, specific_multi_index, a, gauge,
                    get='array',
                )
                transfer[i, :] = transfer_of_k
            # Gather transfer function data into the master
            smart_mpi(transfer, complete_transfer if master else None, mpifun='gatherv')
            if not master:
                continue
            # Save transfer function to disk
            masterprint(
                f'Saving processed {var_name} {component.class_species} '
                f'transfer functions ...'
            )
            var_name_ascii = var_name
            for key, val in {
                'δ': 'delta',
                'θ': 'theta',
                'σ': 'shear',
            }.items():
                var_name_ascii = var_name_ascii.replace(key, val)
            with open_hdf5(filename, mode='a') as hdf5_file:
                perturbations_h5 = hdf5_file.require_group('perturbations')
                dset = perturbations_h5.create_dataset(
                    f'{var_name_ascii}_{component.class_species}',
                    asarray(complete_transfer).shape,
                    dtype=C2np['double'],
                )
                dset[...] = complete_transfer
            masterprint('done')
            # Plot transfer functions
            if class_plot_perturbations:
                graphics.plot_processed_perturbations(
                    all_a_values,
                    k_magnitudes,
                    complete_transfer,
                    var_name,
                    component.class_species,
                )
    # Done writing processed CLASS output
    if master:
        masterprint(f'All processed CLASS output has been saved to "{filename}"')
