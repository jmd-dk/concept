# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2017 Jeppe Mosgaard Dakin.
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
cimport('from communication import domain_subdivisions, exchange')
cimport('import graphics')
cimport('from integration import initiate_time')
cimport('from mesh import CIC_particles2fluid')
cimport('from snapshot import get_snapshot_type, snapshot_extensions')
cimport('from species import get_representation')
cimport('from snapshot import load, save')



# Entry point of this module.
# Call this function to perform a special operation,
# defined in the special_params dict.
@cython.header()
def delegate():
    eval(special_params['special'] + '()')

# Function which convert all snapshots in the
# special_params['snapshot_filenames'] parameter to the snapshot type
# given in the snapshot_type parameter.
@cython.pheader(# Locals
                N_vacuum='Py_ssize_t',
                a='double',
                component='Component',
                dim='int',
                ext='str',
                index='int',
                snapshot='object',
                snapshot_filename='str',
                converted_snapshot_filename='str',
                params='dict',
                attribute_str='str',
                attributes='object',  # collections.defaultdict
                attribute='str',
                key='str',
                value='object',  # double, str or NoneType
                mass='double',
                name='str',
                names='list',
                names_lower='list',
                original_mass='double',
                original_representation='str',
                rel_tol='double',
                unit_str='str',
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
    snapshot = load(snapshot_filename, compare_params=True,  # Warn the user of non-matching params
                                       do_exchange=False,    # Exchanges happen later, if needed
                                       as_if=snapshot_type)
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
                path='str',
                # Locals
                filenames='list',
                msg='str',
                snapshot_filenames='list',
                returns='list',
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
                basename='str',
                index='int',
                ext='str',
                output_dir='str',
                output_filename='str',
                snapshot='object',
                snapshot_filename='str',
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

# Function that produces a render of the file
# specified by the special_params['snapshot_filename'] parameter.
@cython.pheader(# Locals
                basename='str',
                index='int',
                ext='str',
                output_dir='str',
                output_filename='str',
                snapshot='object',
                snapshot_filename='str',
                )
def render():
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
                                         output_bases['render'],
                                         '_' if output_bases['render'] else '',
                                         basename)
    # Attach missing extension to filename
    if not output_filename.endswith('.png'):
        output_filename += '.png'
    # Prepend 'render_' to filename if it
    # is identical to the snapshot filename.
    if output_filename == snapshot_filename:
        output_filename = '{}/render_{}'.format(output_dir, basename)
    # Render the snapshot
    graphics.render(snapshot.components, output_filename,
                    True, '.renders_{}'.format(basename))

# Function for printing all informations within a snapshot
@cython.pheader(# Locals
                alt_str='str',
                component='Component',
                h='double',
                heading='str',
                index='int',
                eos_info='str',
                ext='str',
                param_num='int',
                parameter_filename='str',
                params='dict',
                path='str',
                paths='list',
                snapshot='object',
                snapshot_filenames='list',
                snapshot_type='str',
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
                if isint(ℝ[cbrt(component.N)]):
                    # Print both the particle number N and its
                    # cube root, if integer.
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
                masterprint('{:<16} {}'.format('N_fluidvars', len(component.fluidvars)), indent=4)
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
