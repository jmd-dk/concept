# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2024 Jeppe Mosgaard Dakin.
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
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('import analysis')
cimport('from analysis import measure')
cimport('from communication import exchange')
cimport('import graphics')
cimport(
    'from integration import   '
    '    cosmic_time,          '
    '    init_time,            '
    '    remove_doppelgängers, '
)
cimport(
    'from linear import                   '
    '    class_extra_perturbations_class, '
    '    compute_cosmo,                   '
    '    compute_transfer,                '
    '    get_k_magnitudes,                '
    '    transferfunctions_registered,    '
)
cimport('from mesh import convert_particles_to_fluid')
cimport(
    'from snapshot import     '
    '    compare_parameters,  '
    '    get_snapshot_type,   '
    '    snapshot_extensions, '
)
cimport('import species')
cimport(
    'from snapshot import        '
    '    get_initial_conditions, '
    '    load,                   '
    '    save,                   '
)



# Entry point of this module.
# Call this function to perform a special operation,
# defined in the special_params dict.
@cython.header()
def delegate():
    utility = special_params['special']
    utility = {
        'class': 'class_',
    }.get(utility, utility)
    eval(f'{utility}()')

# Context manager which temporarily sets the
# allow_similarly_named_components flag in the species module to True,
# allowing for the initialisation of many component instances
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
@cython.pheader(
    # Locals
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
    information about global parameters and individual component
    attributes which should be changed.
    """
    init_time()
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
        compare_params=False,  # Postpone parameter comparison
        do_exchange=False,     # Exchanges happen later, if needed
        as_if=snapshot_type,
    )
    # Some of the functions used later use the value of universals.a.
    # Set this equal to the scale factor value in the snapshot.
    # In the end of this function, the original value of
    # universals.a will be reassigned.
    a = universals.a
    universals.a = snapshot.params['a']
    # Now do the parameter comparison
    compare_parameters(snapshot, snapshot_filename)
    # Warn the user of specified changes to component attributes
    # of non-existing components. Allow for components written in a
    # different case.
    names = [component.name for component in snapshot.components]
    names_lower = [name.lower() for name in names]
    for name in dict(attributes):  # New dict needed as keys are removed during iteration
        if name not in names:
            # Specified component name not present.
            # Maybe the problem is due to lower-/upper-case.
            if name.lower() in names_lower:
                # The component name is written in a different case.
                # Move specified attributes over to the properly
                # written name and delete the wrongly written name key
                # from the attributes.
                attributes[names[names_lower.index(name.lower())]].update(attributes[name])
                attributes.pop(name)
            else:
                masterwarn(
                    f'The following attributes are specified for {name}, '
                    f'which does not exist:\n{attributes[name]}'
                )
    # Overwrite parameters in the snapshot with those from the
    # parameter file (those which are currently loaded as globals).
    # If parameters are passed directly, these should take precedence
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
                component.init_w(val)
                continue
            if not hasattr(component, key):
                # A non-existing attribute was specified. As this is
                # nonsensical and leads to an error in compiled mode
                # but not in pure Python mode, do an explicit abort.
                abort(
                    f'The following non-existing attribute was specified for '
                    f'{component.name}: {key}'
                )
            setattr(component, key, val)
        # If both N and gridsize is specified for this component, it
        # means that particles should be converted to a fluid (the other
        # way around is not supported).
        if component.N > 1 and component.gridsize > 1:
            component.representation = 'fluid'
        # Apply particles → fluid conversion, if necessary
        if original_representation == 'particles' and component.representation == 'fluid':
            # To do the conversion, the particles need to be
            # distributed according to which domain they are in.
            component.representation = 'particles'
            exchange(component)
            # The total particle mass and momentum
            Σmass_particles = measure(component, 'mass')
            Σmom_particles, σmom_particles = measure(component, 'momentum')
            # Done treating component as particles.
            # Reassign the fluid representation
            component.representation = 'fluid'
            # The mass attribute shall now be the average mass of a
            # fluid element. Since the total mass of the component
            # should be the same as before, the mass (at a = 1) and the
            # grid size are related by
            # mass = (a**(3*w_eff)*Σmass)/gridsize**3
            #      = original_mass*a**(3*w_eff)*N/gridsize**3.
            # If either mass or gridsize is given by the user,
            # use this to determine the other.
            w_eff = component.w_eff(a=universals.a)
            if 'gridsize' in attributes[name] and 'mass' not in attributes[name]:
                component.mass *= universals.a**(3*w_eff)*float(component.N)/component.gridsize**3
            elif 'mass' in attributes[name] and 'gridsize' not in attributes[name]:
                component.gridsize = int(round(cbrt(
                    universals.a**(3*w_eff)*original_mass/component.mass*component.N)))
                Σmass_fluid = universals.a**(-3*w_eff)*component.mass*component.gridsize**3
                if not isclose(Σmass_particles, Σmass_fluid, 1e-6):
                    if Σmass_fluid > Σmass_particles:
                        masterwarn(
                            f'The specified mass for {component.name} leads to a relative '
                            f'increase of {(Σmass_fluid/Σmass_particles - 1):.9g} '
                            f'for the total mass of this component. '
                            f'Note that for fluids, the specified mass should be '
                            f'the average mass of a fluid element.'
                        )
                    else:
                        masterwarn(
                            f'The specified mass for {component.name} leads to a relative '
                            f'decrease of {(1 - Σmass_fluid/Σmass_particles):.9g} '
                            f'for the total mass of this component. '
                            f'Note that for fluids, the specified mass should be '
                            f'the average mass of a fluid element.'
                        )
            elif 'gridsize' not in attributes[name] and 'mass' not in attributes[name]:
                # If neither the grid size nor the mass is specified,
                # the number of grid points in the fluid
                # representation should equal the number of
                # particles in the particle representation.
                component.gridsize = int(round(cbrt(component.N)))
                # If component.N is not a cube number, the number
                # of fluid elements will not be exactly equal to the
                # number of particles. Adjust the mass accordingly.
                component.mass *= component.N/component.gridsize**3
            # Interpolate particle data to fluid data. Temporarily
            # let the mass attribute be the original particle mass.
            mass = component.mass
            component.mass = original_mass
            N_vacuum = convert_particles_to_fluid(
                component,
                4,  # PCS
            )
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
            # In this case, the conversion from particles to fluid should
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
@cython.pheader(
    # Arguments
    path_or_paths_snapshot=object,  # str or list
    warn_individual='bint',
    exit_individual='bint',
    warn_all='bint',
    exit_all='bint',
    # Locals
    filenames=list,
    msg=str,
    snapshot_filenames=list,
    returns=object,  # list or str; always list when called externally
)
def locate_snapshots(
    path_or_paths_snapshot,
    warn_individual=True,
    exit_individual=False,
    warn_all=False,
    exit_all=True,
    initial_call=True,
):
    if not master:
        return bcast()
    if isinstance(path_or_paths_snapshot, list):
        if len(path_or_paths_snapshot) == 1:
            return locate_snapshots(
                path_or_paths_snapshot[0],
                warn_individual,
                exit_individual,
                warn_all,
                exit_all,
                initial_call=True,
            )
        # Handle list of strings
        snapshot_filenames = []
        for path_snapshot in path_or_paths_snapshot:
            snapshot_filenames += locate_snapshots(
                path_snapshot,
                warn_individual,
                exit_individual,
                initial_call=False,
            )
    else:
        # Handle single string (pointing to file or directory)
        path_snapshot = path_or_paths_snapshot
        # Get all files and directories from the path
        if get_snapshot_type(path_snapshot, collective=False):
            filenames = [path_snapshot]
        elif os.path.isdir(path_snapshot):
            filenames = []
            for filename in sorted(os.listdir(path_snapshot)):
                filename = os.path.join(path_snapshot, filename)
                if os.path.isfile(filename) or os.path.isdir(filename):
                    filenames.append(filename)
        else:
            filenames = [path_snapshot]
        # Only use snapshots
        snapshot_filenames = [
            filename
            for filename in filenames
            if get_snapshot_type(filename, collective=False)
        ]
        if not snapshot_filenames:
            if os.path.isdir(path_snapshot):
                msg = f'The directory "{path_snapshot}" does not contain any snapshots'
            elif os.path.exists(path_snapshot):
                msg = f'The file "{path_snapshot}" is not recognized as a snapshot'
            else:
                msg = f'Path "{path_snapshot}" does not exist'
            if exit_individual:
                abort(msg)
            elif warn_individual:
                warn(msg)
    if not initial_call:
        return snapshot_filenames
    if not snapshot_filenames:
        if isinstance(path_or_paths_snapshot, list):
            msg = 'Could not find any snapshots in {}'.format(
                ', '.join([
                    f'"{path_snapshot}"'
                    for path_snapshot in path_or_paths_snapshot
            ]))
        else:
            if os.path.isdir(path_or_paths_snapshot):
                msg = f'The directory "{path_or_paths_snapshot}" does not contain any snapshots'
            elif os.path.exists(path_or_paths_snapshot):
                msg = f'The file "{path_or_paths_snapshot}" is not recognized as a snapshot'
            else:
                msg = f'Path "{path_or_paths_snapshot}" does not exist'
        if exit_all:
            abort(msg)
        elif warn_all:
            warn(msg)
    return bcast(snapshot_filenames)

# Function that produces a power spectrum of the file
# specified by the special_params['snapshot_filename'] parameter.
@cython.pheader(
    # Locals
    basename=str,
    index='int',
    ext=str,
    output_dir=str,
    output_filename=str,
    snapshot=object,
    snapshot_filename=str,
)
def powerspec():
    init_time()
    # Extract the snapshot filename
    snapshot_filename = special_params['snapshot_filename']
    # Read in the snapshot, postponing the parameter comparison
    snapshot = load(snapshot_filename, compare_params=False)
    # Set universal scale factor and cosmic time and to match
    # that of the snapshot.
    universals.a = snapshot.params['a']
    if enable_Hubble:
        universals.t = cosmic_time(universals.a)
    # Now do the parameter comparison
    compare_parameters(snapshot, snapshot_filename)
    # Construct output filename based on the snapshot filename.
    # Importantly, remove any file extension signalling a snapshot.
    output_dir, basename = os.path.split(snapshot_filename)
    for ext in snapshot_extensions:
        if basename.endswith(ext):
            index = len(basename) - len(ext)
            basename = basename[:index]
            break
    output_filename = '{}/{}{}{}'.format(
        output_dir,
        output_bases['powerspec'],
        '_' if output_bases['powerspec'] else '',
        basename,
    )
    # Prepend 'powerspec_' to filename if it
    # is identical to the snapshot filename.
    if output_filename == snapshot_filename:
        output_filename = f'{output_dir}/powerspec_{basename}'
    # Produce power spectrum of the snapshot
    analysis.powerspec(snapshot.components, output_filename)

# Function that produces a bispectrum of the file
# specified by the special_params['snapshot_filename'] parameter.
@cython.pheader(
    # Locals
    basename=str,
    index='int',
    ext=str,
    output_dir=str,
    output_filename=str,
    snapshot=object,
    snapshot_filename=str,
)
def bispec():
    init_time()
    # Extract the snapshot filename
    snapshot_filename = special_params['snapshot_filename']
    # Read in the snapshot, postponing the parameter comparison
    snapshot = load(snapshot_filename, compare_params=False)
    # Set universal scale factor and cosmic time and to match
    # that of the snapshot.
    universals.a = snapshot.params['a']
    if enable_Hubble:
        universals.t = cosmic_time(universals.a)
    # Now do the parameter comparison
    compare_parameters(snapshot, snapshot_filename)
    # Construct output filename based on the snapshot filename.
    # Importantly, remove any file extension signalling a snapshot.
    output_dir, basename = os.path.split(snapshot_filename)
    for ext in snapshot_extensions:
        if basename.endswith(ext):
            index = len(basename) - len(ext)
            basename = basename[:index]
            break
    output_filename = '{}/{}{}{}'.format(
        output_dir,
        output_bases['bispec'],
        '_' if output_bases['bispec'] else '',
        basename,
    )
    # Prepend 'bispec_' to filename if it
    # is identical to the snapshot filename.
    if output_filename == snapshot_filename:
        output_filename = f'{output_dir}/bispec_{basename}'
    # Produce bispectrum of the snapshot
    analysis.bispec(snapshot.components, output_filename)

# Function which produces a 3D render of the file
# specified by the special_params['snapshot_filename'] parameter.
@cython.pheader(
    # Locals
    basename=str,
    index='int',
    ext=str,
    output_dir=str,
    output_filename=str,
    snapshot=object,
    snapshot_filename=str,
)
def render3D():
    init_time()
    # Extract the snapshot filename
    snapshot_filename = special_params['snapshot_filename']
    # Read in the snapshot, postponing the parameter comparison
    snapshot = load(snapshot_filename, compare_params=False)
    # Set universal scale factor and cosmic time and to match
    # that of the snapshot.
    universals.a = snapshot.params['a']
    if enable_Hubble:
        universals.t = cosmic_time(universals.a)
    # Now do the parameter comparison
    compare_parameters(snapshot, snapshot_filename)
    # Construct output filename based on the snapshot filename.
    # Importantly, remove any file extension signalling a snapshot.
    output_dir, basename = os.path.split(snapshot_filename)
    for ext in snapshot_extensions:
        if basename.endswith(ext):
            index = len(basename) - len(ext)
            basename = basename[:index]
            break
    output_filename = '{}/{}{}{}'.format(
        output_dir,
        output_bases['render3D'],
        '_' if output_bases['render3D'] else '',
        basename,
    )
    # Attach missing extension to filename
    if not output_filename.endswith('.png'):
        output_filename += '.png'
    # Prepend 'render3D_' to filename if it
    # is identical to the snapshot filename.
    if output_filename == snapshot_filename:
        output_filename = f'{output_dir}/render3D_{basename}'
    # Render the snapshot
    graphics.render3D(snapshot.components, output_filename)

# Function for printing all informations within a snapshot
@cython.pheader(
    # Locals
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
    paths=list,
    snapshot=object,
    snapshot_filename=str,
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
    snapshot_filenames = locate_snapshots(
        paths,
        warn_individual=False,
        exit_individual=False,
        warn_all=True,
        exit_all=False,
    )
    # Print out information about each snapshot
    for snapshot_filename in snapshot_filenames:
        # Load parameters from the snapshot
        with allow_similarly_named_components():
            snapshot = load(
                snapshot_filename,
                compare_params=False,
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
        generate_param = special_params.get('generate param')
        if generate_param:
            if os.path.basename(generate_param) == '__together_with_snapshot__':
                generate_param = os.path.dirname(snapshot_filename)
            # Make sure that the params directory exists
            if master:
                os.makedirs(generate_param, exist_ok=True)
            # The filename of the new parameter file
            parameter_filename = f'{generate_param}/{os.path.basename(snapshot_filename)}'
            for ext in snapshot_extensions:
                if parameter_filename.endswith(ext):
                    index = len(parameter_filename) - len(ext)
                    parameter_filename = parameter_filename[:index]
                    break
            parameter_filename += '.param'
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
            with open_file(parameter_filename, mode='w', encoding='utf-8') as pfile:
                masterprint(
                    f'# Auto-generated parameter file for the snapshot\n# "{snapshot_filename}"\n',
                    file=pfile, wrap=False,
                )
                # Loop over stdout and the new parameter file
                for file in (sys.stdout, pfile):
                    masterprint('# Input/output', file=file, wrap=False)
                    masterprint(
                        "initial_conditions = '{}'".format(sensible_path(snapshot_filename)),
                        file=file, wrap=False,
                    )
                    if hasattr(snapshot, 'units'):
                        masterprint('# System of units', file=file, wrap=False)
                        masterprint(
                            "unit_length = '{}'".format(snapshot.units['length']),
                            file=file, wrap=False,
                        )
                        masterprint(
                            "unit_time = '{}'".format(snapshot.units['time']),
                            file=file, wrap=False,
                        )
                        masterprint(
                            "unit_mass = '{}'".format(snapshot.units['mass']),
                            file=file, wrap=False,
                        )
                    masterprint('# Numerical parameters', file=file, wrap=False)
                    unit = 100*units.km/(units.s*units.Mpc)
                    h = params['H0']/unit
                    value = params['boxsize']*h
                    if isint(value):
                        masterprint(
                            'boxsize = {}/{}*{}'
                            .format(int(round(value)), correct_float(h), unit_length),
                            file=file, wrap=False,
                        )
                    else:
                        masterprint(
                            'boxsize = {}*{}'
                            .format(correct_float(params['boxsize']), unit_length),
                            file=file, wrap=False,
                        )
                    masterprint('# Cosmological parameters', file=file, wrap=False)
                    unit = units.km/(units.s*units.Mpc)
                    masterprint(
                        'H0 = {}*km/(s*Mpc)'.format(correct_float(params['H0']/unit)),
                        file=file, wrap=False,
                    )
                    if snapshot_type == 'concept':
                        masterprint(
                            'Ωb = {}'.format(correct_float(params['Ωb'])),
                            file=file, wrap=False,
                        )
                        masterprint(
                            'Ωcdm = {}'.format(correct_float(params['Ωcdm'])),
                            file=file, wrap=False,
                        )
                    elif snapshot_type == 'gadget':
                        # Gadget snapshots only store Ωm = Ωb + Ωcdm.
                        # Use the global value of Ωb (from the parameters)
                        # to get {Ωb, Ωcdm}.
                        masterprint(
                            f'Ωb = {{}}  # from parameters, not the snapshot'
                            .format(correct_float(Ωb)),
                            file=file, wrap=False,
                        )
                        masterprint(
                            f'Ωcdm = {{}}  # using above Ωb and Ωb + Ωcdm = Ωm = {{}}'
                            .format(correct_float(params['Ωm'] - Ωb), correct_float(params['Ωm'])),
                            file=file, wrap=False,
                        )
                    if enable_Hubble:
                        masterprint(
                            'a_begin = {}'.format(correct_float(params['a'])),
                            file=file, wrap=False,
                        )
            # Do not edit the printed text below,
            # as it is grepped for by several of the Bash utilities.
            masterprint(
                f'\nThe above parameters have been written to "{parameter_filename}"',
                wrap=False,
            )
            # Done writing out parameters. The code below which prints
            # out information about the snapshot should not be reached.
            continue
        # Print out heading stating the filename
        heading = '\nInformation about "{}"'.format(sensible_path(snapshot_filename))
        masterprint(terminal.bold(heading))
        # Print out snapshot type
        masterprint('{:<20} {}'.format('Snapshot type', snapshot_type))
        if snapshot_type == 'gadget':
            # Also print out SnapFormat in case of GADGET snapshot
            masterprint('{:<20} {}'.format('GADGET SnapFormat', snapshot.snapformat))
        # Print out unit system for CO𝘕CEPT snapshots
        if snapshot_type == 'concept':
            masterprint('{:<20} {}'.format('unit_length', snapshot.units['length']))
            masterprint('{:<20} {}'.format('unit_time',   snapshot.units['time']))
            # The mass is typically some large number written in
            # exponential notation. Print it out nicely.
            mass_num = eval_unit(snapshot.units['mass'])/units.m_sun
            mass_basicunit = 'm☉'
            mass_num_fmt = significant_figures(float(mass_num), 6, fmt='unicode', incl_zeros=False)
            masterprint(f'{{:<20}} {mass_num_fmt} {mass_basicunit}'.format('unit_mass'))
        # Print out global parameters
        unit = units.km/(units.s*units.Mpc)
        masterprint('{:<20} {} km s⁻¹ Mpc⁻¹'.format('H0', correct_float(params['H0']/unit)))
        masterprint('{:<20} {}'.format('a', correct_float(params['a'])))
        # The boxsize should also be printed as boxsize/h, if integer
        unit = 100*units.km/(units.s*units.Mpc)
        h = params['H0']/unit
        value = params['boxsize']*h
        alt_str = ''
        if isint(value) and not isint(params['boxsize']):
            alt_str = ' = {} {}/h'.format(int(round(value)), unit_length)
        masterprint(
            '{:<20} {} {}{}'
            .format('boxsize', correct_float(params['boxsize']), unit_length, alt_str)
        )
        # Print out the cosmological density parameters Ωcdm and Ωb.
        # These are only present in the CO𝘕CEPT snapshots. In GADGET
        # snapshots, instead we have ΩΛ and Ωm. We do not print these
        # out here, as these will be printed as part
        # of the GADGET header.
        if snapshot_type == 'concept':
            masterprint('{:<20} {}'.format(unicode('Ωb'), correct_float(params['Ωb'])))
            masterprint('{:<20} {}'.format(unicode('Ωcdm'), correct_float(params['Ωcdm'])))
        # Print out GADGET header for GADGET snapshots.
        # Note that the header structure of GADGET-2
        # specifically is assumed.
        if snapshot_type == 'gadget':
            masterprint('GADGET header:')
            for key, val in snapshot.header.items():
                masterprint(f'{key:<16} {val}', indent=4)
        # Print out TIPSY header for TIPSY snapshots
        if snapshot_type == 'tipsy':
            masterprint('TIPSY header:')
            for key, val in snapshot.header.items():
                masterprint(f'{key:<16} {val}', indent=4)
        # Print out component information
        for component in snapshot.components:
            masterprint(f'{component.name}:')
            masterprint('{:<16} {}'.format('species', component.species), indent=4)
            # Representation-specific attributes
            if component.representation == 'particles':
                # Print the particle number N
                masterprint(
                    '{:<16} {} = {}'.format('N', component.N, get_cubenum_strrep(component.N)),
                    indent=4,
                )
                masterprint(
                    '{:<16} {} m☉'.format(
                        'mass',
                        significant_figures(
                            component.mass/units.m_sun, 6, fmt='unicode', incl_zeros=False,
                        )
                    ) if component.mass != -1 else '{:<16} ?'.format('mass'),
                    indent=4,
                )
            elif component.representation == 'fluid':
                masterprint('{:<16} {}'.format('gridsize', component.gridsize), indent=4)
                masterprint(
                    '{:<16} {}'.format('boltzmann_order', component.boltzmann_order),
                    indent=4,
                )
                if component.w_type == 'constant':
                    eos_info = significant_figures(
                        component.w_constant, 6, fmt='unicode', incl_zeros=False,
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
                masterprint(
                    '{:<16} [{}, {}, {}] {}'.format(
                        'momentum sum',
                        *significant_figures(
                            asarray(Σmom)/units.m_sun, 6, fmt='unicode', force_scientific=True,
                        ),
                        f'm☉ {unit_length} {unit_time}⁻¹',
                    ),
                    indent=4,
                )
                masterprint(
                    '{:<16} [{}, {}, {}] {}'.format(
                        'momentum spread',
                        *significant_figures(
                            asarray(σmom)/units.m_sun, 6, fmt='unicode', force_scientific=True,
                        ),
                        f'm☉ {unit_length} {unit_time}⁻¹',
                    ),
                    indent=4,
                )
        # End of information
        masterprint('')

# Function that saves the processed CLASS background
# and perturbations to an hdf5 file.
@cython.pheader(
    # Locals
    a='double',
    a_first='double',
    a_min='double',
    a_values='double[::1]',
    all_a_values='double[::1]',
    arr='double[::1]',
    boltzmann_order='Py_ssize_t',
    class_modes_per_decade_ori=dict,
    class_species=str,
    component='Component',
    component_variables=dict,
    components=list,
    compute_perturbations='bint',
    convenience_attributes=dict,
    fac='double',
    fac_max='double',
    fac_min='double',
    filename=str,
    gauge=str,
    gauge_str=str,
    gridsize='Py_ssize_t',
    gridsize_i='Py_ssize_t',
    gridsize_or_k_magnitudes=object,  # Py_ssize_t or np.ndarray
    i='Py_ssize_t',
    index='Py_ssize_t',
    k_magnitudes='double[::1]',
    k_max='double',
    k_min='double',
    key=str,
    modes=object,  # int or np.ndarray
    perturbations=object,  # PerturbationDict
    rank_other='int',
    size='Py_ssize_t',
    times=object,  # int or np.ndarray
    transfer='double[:, ::1]',
    transfer_of_k='double[::1]',
    transferfunction_info=object,  # TransferFunctionInfo
    var_name=str,
    variable_specifications=list,
    ρ_bars=dict,
)
def class_():
    # Error out if any species has a Boltzmann order
    # outside the implemented range.
    for d in initial_conditions:
        if not isinstance(d, dict):
            continue
        boltzmann_order = int(d.get('boltzmann_order', 0)) + 1
        class_species = d.get('species', d.get('name', ''))
        if boltzmann_order < 0:
            abort(f'Species {class_species} has Boltzmann order {boltzmann_order} < 0')
        if boltzmann_order > 2:
            abort(f'Species {class_species} has Boltzmann order {boltzmann_order} > 2')
    # Suppress warning about the total energy density of the components
    # being too high, as the components are not used to perform a
    # simulation anyway.
    suppress_output['err'].add('the energy density of the components add up to')
    # Initialise components, but do not realise them
    init_time()
    components = get_initial_conditions(do_realization=False)
    # Should we compute and store perturbations (or only background)?
    compute_perturbations = bool(components or class_extra_perturbations)
    if compute_perturbations:
        # Get the grid size corresponding to the maximum k.
        # Note that the minimum k is implicitly set by the boxsize.
        gridsize = -1
        if compute_perturbations:
            for gridsize_i in powerspec_options.get('global gridsize', {'default': -1}).values():
                gridsize_i = int(round(float(gridsize_i)))
                if gridsize_i > gridsize:
                    gridsize = gridsize_i
            if gridsize <= 2:
                abort(
                    'You should specify the maximum k mode to compute, either explicitly '
                    'via the --kmax command-line option to the class utility, or implicitly '
                    'by specifying the powerspec_options["global gridsize"] parameter, e.g.\n'
                    'powerspec_options = {"global gridsize": 512}'
                )
        # If a maximum k is explicitly given, make sure that this is in
        # fact included in the k range.
        k_min = float(special_params['kmin'])
        k_max = float(special_params['kmax'])
        while k_max != -1:
            k_magnitudes, _ = get_k_magnitudes(gridsize, n_pad=0, use_cache=False)
            if k_magnitudes[k_magnitudes.shape[0] - 1] >= k_max:
                break
            gridsize += 2
        # Get either the number of modes at which to tabulate
        # the perturbations, or the exact k values of these modes.
        # A value of -1 means that no specific number of modes has been
        # requested, in which case we should use the number obtained
        # from class_modes_per_decade as is.
        modes = bcast(
            handle_class_arg(special_params['modes'], 'modes', k_min, k_max)
            if master else None
        )
        # Get the k values at which to tabulate the perturbations
        gridsize_or_k_magnitudes = modes
        if isinstance(modes, (int, np.integer)):
            gridsize_or_k_magnitudes = gridsize
            if modes == 0:
                abort(
                    'You have specified 0 modes. If you do not want any perturbation output, '
                    'omit the perturbation argument to the class utility'
                )
            elif modes < -1:
                abort(f'Invalid number of modes: {modes}')
            elif modes == 1:
                # Set a specific mode
                if 0 < k_min < ထ and 0 < k_max < ထ:
                    gridsize_or_k_magnitudes = asarray(
                        [sqrt(k_min*k_max)],
                        dtype=C2np['double'],
                    )
                else:
                    k_magnitudes, _ = get_k_magnitudes(gridsize, n_pad=0, use_cache=False)
                    gridsize_or_k_magnitudes = asarray(
                        [np.prod(k_magnitudes)**(1/k_magnitudes.size)],
                        dtype=C2np['double'],
                    )
            elif modes != -1:
                # Adjust the global user parameter
                # class_modes_per_decade so that the exact requested
                # number of k modes is obtained.
                class_modes_per_decade_ori = class_modes_per_decade.copy()
                k_magnitudes, _ = get_k_magnitudes(gridsize, n_pad=0, use_cache=False)
                fac = 1
                while k_magnitudes.shape[0] < modes:
                    fac *= 1.1
                    for k_magnitude, val in class_modes_per_decade_ori.items():
                        class_modes_per_decade[k_magnitude] = fac*val
                    k_magnitudes, _ = get_k_magnitudes(gridsize, n_pad=0, use_cache=False)
                fac_max = fac
                while k_magnitudes.shape[0] > modes:
                    fac *= 0.9
                    for k_magnitude, val in class_modes_per_decade_ori.items():
                        class_modes_per_decade[k_magnitude] = fac*val
                    k_magnitudes, _ = get_k_magnitudes(gridsize, n_pad=0, use_cache=False)
                fac_min = fac
                while True:
                    fac = sqrt(fac_min*fac_max)
                    for k_magnitude, val in class_modes_per_decade_ori.items():
                        class_modes_per_decade[k_magnitude] = fac*val
                    k_magnitudes, _ = get_k_magnitudes(gridsize, n_pad=0, use_cache=False)
                    if isclose(fac_min, fac_max):
                        break
                    if k_magnitudes.shape[0] < modes:
                        fac_min = fac
                    elif k_magnitudes.shape[0] > modes:
                        fac_max = fac
                    else:
                        break
                gridsize_or_k_magnitudes = k_magnitudes
    # Do CLASS computation
    if compute_perturbations:
        gauge = special_params['gauge'].replace('-', '').lower()
        cosmoresults = compute_cosmo(
            gridsize_or_k_magnitudes,
            'synchronous' if gauge == 'nbody' else gauge,
            class_call_reason=(
                'in order to get {} gauge perturbations'
                .format(
                    {
                        'nbody'      : '𝘕-body',
                        'synchronous': 'synchronous',
                        'newtonian'  : 'Newtonian',
                    }.get(gauge)
                )
            ),
        )
        k_magnitudes = cosmoresults.k_magnitudes
    else:
        cosmoresults = compute_cosmo(class_call_reason='in order to get background')
    cosmoresults.load_everything()
    # If perturbations should be plotted, we want to include all
    # perturbations used as an intermediate step as well (e.g. for gauge
    # transformation) in the detrended plots. We ensure this by simply
    # retrieving all used perturbations as TransferFunction instances,
    # which triggers calls to graphics.plot_detrended_perturbations().
    if (
            compute_perturbations
        and class_plot_perturbations
    ):
        # If the number of processes exceeds the number of k modes,
        # not all processes know about the perturbations in use, and so
        # we need to broadcast this information.
        bcast_root = allreduce(
            rank + nprocs*(len(getattr(cosmoresults, '_perturbations', [])) == 0),
            op=MPI.MIN,
        )
        if bcast_root < nprocs:
            names_class = bcast(
                (
                    list(cosmoresults._perturbations[0].keys())
                    if rank == bcast_root else None
                ),
                bcast_root,
            )
            for name_class in names_class:
                for transferfunction_info in transferfunctions_registered.values():
                    if transferfunction_info.name_class == name_class:
                        getattr(
                            cosmoresults,
                            transferfunction_info.name.removesuffix('_tot'),
                        )(get='object')
    # Store all CLASS parameters, the unit system in use,
    # the processed background and a few convenience attributes
    # in a new hdf5 file.
    filename = output_dirs['powerspec'] + '/class_processed.hdf5'
    if master:
        os.makedirs(output_dirs['powerspec'], exist_ok=True)
        with open_hdf5(filename, mode='w') as hdf5_file:
            # Store CLASS parameters as attributes on the
            # "class_params" group. If you need to know further
            # parameters used by CLASS (i.e. default values),
            # you should specify these explicitly in the class_params
            # user parameter. No unit conversion will take place.
            params_h5 = hdf5_file.require_group('class_params')
            for key, val in cosmoresults.params.items():
                key = key.replace('/', '__per__')
                params_h5.attrs[key] = bytes(str(val), encoding='ascii')
            # Store the unit system in use. This is important as all
            # variables stored below will be stored in these units.
            units_h5 = hdf5_file.require_group('units')
            for unit_name, unit_val in {
                'unit time'  : unit_time,
                'unit length': unit_length,
                'unit mass'  : unit_mass,
            }.items():
                try:
                    units_h5.attrs[unit_name] = bytes(unit_val, encoding='ascii')
                except UnicodeEncodeError:
                    units_h5.attrs[unit_name] = unit_val
            # Store background variables present in the
            # cosmoresults.background dict. Here we convert to the
            # current unit system in use. We also add any (present)
            # background densities we come across to the ρ_bars dict.
            ρ_bars = {}
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
                    key = key.removeprefix('(.)')
                    # Add the present background density to ρ_bars
                    ρ_bars[class_species] = arr[arr.shape[0] - 1]
                elif key.startswith('(.)p'):
                    # The "(.)" notation is CLASS syntax reminding us
                    # that we need to multiply by 3/(8πG).
                    # We do this and convert to the proper units
                    # using the P_bar method.
                    class_species = key.split('(.)p_')[1]
                    arr = cosmoresults.P_bar(cosmoresults.background['a'], class_species)
                    # Now, the "(.)" prefix should be dropped
                    key = key.removeprefix('(.)')
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
                    key = 'D'
                elif key == 'gr.fac. f':
                    # Unitless
                    key = 'f'
                elif key == 'gr.fac. D2':
                    # Unitless
                    key = 'D2'
                elif key == 'gr.fac. f2':
                    # Unitless
                    key = 'f2'
                elif key == 'gr.fac. D3a':
                    # Unitless
                    key = 'D3a'
                elif key == 'gr.fac. f3a':
                    # Unitless
                    key = 'f3a'
                elif key == 'gr.fac. D3b':
                    # Unitless
                    key = 'D3b'
                elif key == 'gr.fac. f3b':
                    # Unitless
                    key = 'f3b'
                elif key == 'gr.fac. D3c':
                    # Unitless
                    key = 'D3c'
                elif key == 'gr.fac. f3c':
                    # Unitless
                    key = 'f3c'
                elif key == '(.)w_fld':
                    # Unitless
                    key = 'w_fld'
                else:
                    masterwarn(
                        f'Unrecognised CLASS background variable "{key}". '
                        f'Unit conversion could not be carried out.'
                    )
                dset = background_h5.create_dataset(key, (arr.shape[0],), dtype=C2np['double'])
                dset[:] = arr
            # Also store ρ_bar and P_bar for the class species of the
            # components, if these are combination species.
            for component in components:
                if '+' not in component.class_species:
                    continue
                # Store the background density
                key = f'rho_{component.class_species}'
                arr = cosmoresults.ρ_bar(cosmoresults.background['a'], component)
                dset = background_h5.create_dataset(key, (arr.shape[0],), dtype=C2np['double'])
                dset[:] = arr
                # Add the present background density to ρ_bars
                ρ_bars[component.class_species] = arr[arr.shape[0] - 1]
                # Store the background pressure
                key = f'p_{component.class_species}'
                arr = cosmoresults.P_bar(cosmoresults.background['a'], component)
                dset = background_h5.create_dataset(key, (arr.shape[0],), dtype=C2np['double'])
                dset[:] = arr
            # Store a few convenience attributes on the background
            # group, specifying the cosmology. These convenience
            # attributes include h ≡ H0/(100 km s⁻¹ Mpc⁻¹), density
            # parameters (Ω) for all the CLASS species present
            # (including combination species) and the w_0 and w_a
            # parameters in the case of dynamical dark energy.
            # Note that these convenience attributes do not add
            # information; they could be derived from the data already
            # present in the HDF5 file.
            convenience_attributes = {'h': H0/(100*units.km/(units.s*units.Mpc))}
            for class_species, ρ_bar in ρ_bars.items():
                convenience_attributes[f'Omega_{class_species}'] = ρ_bar/ρ_bars['crit']
            if 'w0_fld' in class_params:
                convenience_attributes['w_0'] = float(class_params['w0_fld'])
            if 'wa_fld' in class_params:
                convenience_attributes['w_a'] = float(class_params['wa_fld'])
            for convenience_name, convenience_val in convenience_attributes.items():
                background_h5.attrs[convenience_name] = convenience_val
    # Done writing CLASS background to file
    if not compute_perturbations:
        masterprint(f'All processed CLASS output has been saved to "{filename}"')
        return
    # Create dict mapping components to lists of
    # (variable, specific_multi_index, var_name), specifying which
    # transfer functions to store in the hdf5 file.
    component_variables = {}
    for component in components:
        # Create list of (variable, specific_multi_index, var_name)
        variable_specifications = [(0, None, 'δ')]
        if component.representation == 'particles':
            if not component.realization_options['backscale']:
                variable_specifications.append((1, None, 'θ'))
        elif component.representation == 'fluid':
            if component.boltzmann_order > 0 or (
                component.boltzmann_order == 0 and component.boltzmann_closure == 'class'):
                variable_specifications.append((1, None, 'θ'))
            if component.boltzmann_order > 1 or (
                component.boltzmann_order == 1 and component.boltzmann_closure == 'class'):
                variable_specifications.append((2, 'trace', 'δP'))
                variable_specifications.append((2, (0, 0), 'σ'))
        component_variables[component] = variable_specifications
    # Add any extra perturbations specified in the
    # class_extra_perturbations user parameter.
    component_variables[None] = []
    for class_extra_perturbation in sorted(class_extra_perturbations_class):
        for transferfunction_info in transferfunctions_registered.values():
            if not transferfunction_info.total:
                continue
            if transferfunction_info.name_class == class_extra_perturbation:
                component_variables[None].append((None, None, transferfunction_info.name))
                break
    if not component_variables[None]:
        component_variables.pop(None)
    # Construct array of a values at which to tabulate the
    # transfer functions. This is done by merging all of the a arrays
    # for the individual k modes, ensuring that all perturbations will
    # be smooth on this common grid of a values.
    a_min = -1
    size = 0
    for perturbations in cosmoresults.perturbations:
        a_values = perturbations['a']
        a_first = a_values[0]
        if a_first > a_min:
            a_min = a_first
        size += a_values.shape[0]
    if master:
        a_min = reduce(a_min, op=MPI.MAX)
        size = reduce(size, op=MPI.SUM)
        all_a_values = empty(size, dtype=C2np['double'])
        index = 0
        # The a values of the master itself
        for perturbations in cosmoresults.perturbations:
            a_values = perturbations['a']
            size = a_values.shape[0]
            all_a_values[index:index+size] = a_values
            index += size
        # The a values of the slaves
        for rank_other in range(nprocs):
            if rank_other == rank:
                continue
            while True:
                size = recv(source=rank_other)
                if size == 0:
                    break
                Recv(all_a_values[index:], source=rank_other)
                index += size
        # Sort and remove duplicate a values
        asarray(all_a_values).sort()
        for index in range(all_a_values.shape[0]):
            if all_a_values[index] == a_min:
                all_a_values = all_a_values[index:]
                break
        all_a_values, _ = remove_doppelgängers(all_a_values, all_a_values, rel_tol=0.5)
        all_a_values = asarray(all_a_values).copy()
        # Get either the number of times at which to tabulate
        # the perturbations, or the exact scale factor values at which
        # to do so. A value of -1 means include all values from CLASS.
        times = handle_class_arg(special_params['times'], 'times')
        # Get the scale factor values at which
        # to tabulate the perturbations.
        if isinstance(times, (int, np.integer)):
            # If too many a values are given, evenly select the
            # requested amount from the available times.
            if all_a_values.shape[0] > times and times != -1:
                step = float(all_a_values.shape[0])/(times - 1)
                all_a_values_selected = empty(times, dtype=C2np['double'])
                for i in range(times - 1):
                    all_a_values_selected[i] = all_a_values[cast(int(i*step), 'Py_ssize_t')]
                all_a_values_selected[times - 1] = all_a_values[all_a_values.shape[0] - 1]
                all_a_values = all_a_values_selected
        else:
            # Use explicit values given
            if times[0] < all_a_values[0]:
                if isclose(cast(times[0], 'double'), cast(all_a_values[0], 'double')):
                    times[0] = all_a_values[0]
                else:
                    abort(
                        f'Cannot tabulate perturbations at a = {times[0]} as the '
                        f'earliest scale factor value tabulated from the CLASS computation '
                        f'is {all_a_values[0]}. Try lowering the a_begin parameter.'
                    )
            if times[times.shape[0] - 1] > all_a_values[all_a_values.shape[0] - 1]:
                if isclose(
                    cast(times[times.shape[0] - 1], 'double'),
                    cast(all_a_values[all_a_values.shape[0] - 1], 'double'),
                ):
                    times[times.shape[0] - 1] = all_a_values[all_a_values.shape[0] - 1]
                else:
                    abort(
                        f'Cannot tabulate perturbations at a = {times[times.shape[0] - 1]} as the '
                        f'last scale factor value tabulated by CLASS '
                        f'is {all_a_values[all_a_values.shape[0] - 1]}'
                    )
            all_a_values = times
        # Broadcast the a values to the slave processes
        bcast(all_a_values.shape[0])
        Bcast(all_a_values)
    else:
        reduce(a_min, op=MPI.MAX)
        reduce(size, op=MPI.SUM)
        # Send a values of local perturbations to the master process
        for perturbations in cosmoresults.perturbations:
            a_values = perturbations['a']
            size = a_values.shape[0]
            send(size, dest=master_rank)
            Send(a_values, dest=master_rank)
        send(0, dest=master_rank)
        # Receive processed a values from the master process
        all_a_values = empty(bcast(), dtype=C2np['double'])
        Bcast(all_a_values)
    # Store the a and k values at which the perturbations are tabulated.
    # Also store the gauge.
    if component_variables and master:
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
            perturbations_h5.attrs['gauge'] = bytes(gauge, encoding='ascii')
    # Get transfer functions of k for each a.
    # Partition the work across the a values.
    # Collect the results into the 2D transfer array.
    # Once a complete transfer function (given at all a and k values)
    # has been constructed, it is saved to disk and possibly plotted.
    # For the next transfer function, we reuse the same 2D arrays,
    # as all transfer functions are tabulated at the same a and k.
    gauge_str = {
        'nbody'      : '𝘕-body',
        'synchronous': 'synchronous',
        'newtonian'  : 'Newtonian',
    }.get(gauge, gauge)
    if master:
        transfer = empty((all_a_values.shape[0], k_magnitudes.shape[0]), dtype=C2np['double'])
    for component, variable_specifications in component_variables.items():
        if component is None:
            class_species = 'tot'
        else:
            class_species = component.class_species
        for variable, specific_multi_index, var_name in variable_specifications:
            transferfunction_info = transferfunctions_registered[var_name]
            if transferfunction_info.total:
                masterprint(f'Working on {var_name} perturbations ...')
            else:
                masterprint(
                    f'Working on {var_name} {class_species} '
                    f'{gauge_str} gauge perturbations ...'
                )
            for i in range(all_a_values.shape[0]):
                a = all_a_values[i]
                if component is None:
                    transfer_of_k = getattr(cosmoresults, var_name)(a)
                else:
                    transfer_of_k, _ = compute_transfer(
                        component, variable, gridsize_or_k_magnitudes, specific_multi_index, a,
                        -1,  # the a_next argument
                        gauge, get='array',
                    )
                if master:
                    transfer[i, :] = transfer_of_k
            if not master:
                continue
            # Save perturbations to disk
            if transferfunction_info.total:
                masterprint(f'Saving processed {var_name} perturbations ...')
            else:
                masterprint(f'Saving processed {var_name} {class_species} perturbations ...')
            with open_hdf5(filename, mode='a') as hdf5_file:
                perturbations_h5 = hdf5_file.require_group('perturbations')
                dset_name = transferfunction_info.name_ascii.format(class_species)
                dset = perturbations_h5.create_dataset(
                    dset_name,
                    asarray(transfer).shape,
                    dtype=C2np['double'],
                )
                dset[...] = transfer
            masterprint('done')
            # Plot perturbations
            if class_plot_perturbations:
                graphics.plot_processed_perturbations(
                    all_a_values,
                    k_magnitudes,
                    transfer,
                    transferfunction_info,
                    class_species,
                )
            # Completely done with this requested transfer function
            masterprint('done')
    # Done writing processed CLASS output
    masterprint(f'All processed CLASS output has been saved to "{filename}"')

# Helper function for the class_() function
def handle_class_arg(arg, kind, vmin=0, vmax=ထ):
    if not isinstance(arg, str):
        abort(f'handle_class_arg() called with type {type(arg)}')
    if vmin == -1:
        vmin = 0
    if vmax == -1:
        vmax = ထ
    kind = {'times': 'a', 'modes': 'k'}.get(kind, kind)
    if os.path.isfile(arg):
        if arg.endswith('.hdf5'):
            with open_hdf5(arg, mode='r') as hdf5_file:
                values = hdf5_file[f'perturbations/{kind}'][:]
                # Apply units of inverse length in the case of k modes
                if kind == 'k':
                    unit_length_hdf5 = hdf5_file['units'].attrs['unit length']
                    unit = 1/eval_unit(unit_length_hdf5)
                    values *= unit
        else:
            values = np.loadtxt(arg)
            if values.ndim > 1:
                abort(f'Got several arrays of data from file {arg}')
    else:
        values = eval_unit(arg, units_dict)
    if isinstance(values, (int, float, np.integer, np.floating)):
        is_pure_number = True
        try:
            float(arg)
        except Exception:
            is_pure_number = False
        if is_pure_number:
            if values == ထ:
                values = -1
            elif np.isclose(values, round(values), atol=10*machine_ϵ):
                values = int(round(values))
            else:
                values = [float(values)]
        else:
            values = [values]
    if not isinstance(values, (int, float, np.integer, np.floating)):
        values = asarray(values, dtype=C2np['double'])
        if values.shape == ():
            values = asarray([values[()]])
        values = asarray(sorted(set(values)), dtype=C2np['double'])
        values = values[(vmin <= values) & (values <= vmax)]
    return values
