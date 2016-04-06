# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015-2016 Jeppe Mosgaard Dakin.
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
# The auther of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports.
# Functions from the 'analysis' and 'graphics' modules are not dumped
# directly into the global namespace of this module, as functions with
# identical names are defined here.
from communication import smart_mpi
from snapshot import load
cimport('import analysis, graphics')
cimport('from communication import domain_subdivisions, exchange')
cimport('from mesh import CIC_particles2fluid')
cimport('from snapshot import get_snapshot_type, snapshot_extensions')
cimport('from species import get_representation')



# Entry point to this module.
# Call this function to perform a special operation,
# defined by the special_params dict.
@cython.header()
def delegate():
    eval(special_params['special'] + '()')

# Function which convert all snapshots in the
# special_params['snapshot_filenames'] parameter to the snapshot type
# given in the snapshot_type parameter.
@cython.pheader(# Locals
                component='Component',
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
                value='object',  # double or str
                mass='double',
                name='str',
                original_mass='double',
                original_representation='str',
                original_tot_mass='double',
                tot_mass='double',
                )
def convert():
    """This function will convert all snapshots listed in the
    special_params['snapshot_filenames'] parameter to the type
    specified by the snapshot_type parameter.
    If special_params['component attributes'] is not empty, it contains
    information about individual component attributes which should be
    changed.
    """
    # Create dict of global parameters (params) and (default)dict of
    # component attributes (attributes) from the parsed attributes.
    params = {}
    attributes = collections.defaultdict(dict)
    for attribute_str in special_params['attributes']:
        index = attribute_str.index('=')
        key = (attribute_str[:index].strip().replace(unicode('Ω'), 'Ω')
                                            .replace(unicode('Λ'), 'Λ'))
        try:
            # Numerical value, possibly with units
            value = eval(attribute_str[(index + 1):], units_dict)
        except:
            # String value
            value = attribute_str[(index + 1):].strip()
        if '.' in key:
            name, attribute = key.split('.')
            attributes[name.strip()][attribute.strip()] = value
        else:
            params[key] = value
    # The filename of the snapshot to read in
    snapshot_filename = special_params['snapshot_filename']
    # Read snapshot on disk into the requested type
    snapshot = load(snapshot_filename, compare_params=True,  # Warn the user of non-matching params
                                       do_exchange=False,    # Exchanges happen later, if needed
                                       as_if=snapshot_type)
    # Overwrite parameters in the snapshot with those from the
    # parameter file (those which are currently loaded as globals).
    snapshot.populate(snapshot.components, a_begin)
    # If specific parameters are parsed as attributes,
    # update the snapshot parameters accordingly.
    snapshot.params.update(params)
    # For GADGET snapshot, also update the GADGET header
    if snapshot_type == 'gadget2':
        snapshot.update_header()
    # Edit individual components if component attributes are parsed
    for component in snapshot.components:
        # The (original) name of this component
        name = component.name
        # Backup of original representation and mass
        original_representation = component.representation
        original_mass = component.mass
        # Edit component attributes
        for key, val in attributes[name].items():
            setattr(component, key, val)
        component.representation = get_representation(component.species)
        # Apply particles <--> fluid convertion, if necessary
        if original_representation == 'particles' and component.representation == 'fluid':
            # To do the convertion, the particles need to be
            # distributed according to which domain they are in.
            component.representation = 'particles'
            exchange(component)
            component.representation = 'fluid'
            # The mass attribute is now the average mass of a fluid
            # element. Since the total mass of the component should be
            # the same as before, the mass and the gridsize are related,
            # as mass = totmass/gridsize**3
            #         = N*original_mass/gridsize**3.
            # If either mass or gridsize is given by the user,
            # use this to determine the other.
            if 'gridsize' in attributes[name] and 'mass' not in attributes[name]:
                component.mass *= float(component.N)/component.gridsize**3
            elif 'mass' in attributes[name] and 'gridsize' not in attributes[name]:
                component.gridsize = int(round((original_mass/component.mass*component.N)**ℝ[1/3]))
                original_tot_mass = original_mass*component.N
                tot_mass = component.mass*component.gridsize**3
                if not isclose(original_tot_mass, tot_mass, 1e-6):
                    if tot_mass > original_tot_mass:
                        masterwarn('The specified mass for the "{}" component\n'
                                   'leads to a relative increase of {:.9g}\n'
                                   'for the total mass of this component.'
                                   'Note that for fluids, the specified mass should be\n'
                                   'the average mass of a fluid element.'
                                   .format(component.name, tot_mass/original_tot_mass - 1))
                    else:
                        masterwarn('The specified mass for the "{}" component\n'
                                   'leads to a relative decrease of {:.9g}\n'
                                   'for the total mass of this component.\n'
                                   'Note that for fluids, the specified mass should be\n'
                                   'the average mass of a fluid element.'
                                   .format(component.name, 1 - tot_mass/original_tot_mass))
            elif 'gridsize' not in attributes[name] and 'mass' not in attributes[name]:
                # If neither the gridsize nor the mass is specified,
                # the number of gridpoints in the fluid
                # representation should equal the number of
                # particles in the particle representation.
                component.gridsize = int(round(component.N**ℝ[1/3]))
                # If component.N is not a cube number, the number
                # of fluid elements will not be exactly equal to the
                # number of particles. Adjust the mass accordingly.
                component.mass *= component.N/component.gridsize**3
            # CIC-interpolate particle data to fluid data
            mass = component.mass
            component.mass = original_mass
            CIC_particles2fluid(component, snapshot.params['a'])
            component.mass = mass
        elif original_representation == 'fluid' and component.representation == 'particles':
            pass
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
    snapshot_filenames = [filename for filename in filenames
                          if get_snapshot_type(filename)]
    # Abort if none of the files where snapshots
    if master and not snapshot_filenames:
        if os.path.isdir(path):
            msg = 'The directory "{}" does not contain any snapshots.'.format(path)
        else:
            msg = 'The file "{}" is not a valid snapshot.'.format(path)
        abort(msg)
    return snapshot_filenames

# Function that produces a powerspectrum of the file
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
    # Produce powerspectrum of the snapshot
    analysis.powerspec(snapshot.components, snapshot.params['a'], output_filename)

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
    graphics.render(snapshot.components,
                    snapshot.params['a'],
                    output_filename)

# Function for printing all informations within a snapshot
@cython.pheader(# Locals
                component='Component',
                cube_root_N='double',
                h='double',
                heading='str',
                params='dict',
                path='str',
                snapshot='object',
                snapshot_filenames='list',
                snapshot_type='str',
                unit='double',
                unit_str='str',
                value='double',
                index='int',
                ext='str',
                parameter_filename='str',
                param_num='int',
                )
def info():
    # This function should not run in parallel
    if not master:
        return
    # Extract the path to snapshot(s)
    path = special_params['path']
    # Get list of snapshots
    snapshot_filenames = locate_snapshots(path)
    # Print out information about each snapshot
    for snapshot_filename in snapshot_filenames:
        # Load parameters from the snapshot
        snapshot = load(snapshot_filename, compare_params=False, only_params=True)
        params = snapshot.params
        # If a parameter file should be generated from the snapshot,
        # print out the content which should be placed in parameter file
        # to stdout and directly to a new parameter file.
        if special_params['generate paramsfile']:
            # Make sure that the params dir exist
            if master:
                os.makedirs(paths['params_dir'], exist_ok=True)
            # The filename of the new parameter file
            parameter_filename = '{}/{}'.format(paths['params_dir'],
                                                os.path.basename(snapshot_filename))
            for ext in snapshot_extensions:
                if parameter_filename.endswith(ext):
                    index = len(parameter_filename) - len(ext)
                    parameter_filename = parameter_filename[:index]
                    break
            parameter_filename += '.params'
            # Do not overwrite an existing parameter file.
            # Append increasing number
            # until a non-existing name is reached.
            param_num = 0
            if os.path.isfile(parameter_filename):
                param_num = 0
                while os.path.isfile(parameter_filename + str(param_num)):
                    param_num += 1
                parameter_filename += str(param_num)
            heading = '\nParameters of "{}"'.format(sensible_path(snapshot_filename))
            masterprint(terminal.bold(heading))
            with open(parameter_filename, 'w') as pfile:
                masterprint('# Auto-generated parameter file for the snapshot\n# "{}"\n'
                            .format(snapshot_filename), file=pfile)
                # Loop over stdout and the new parameter file
                for file in (sys.stdout, pfile):
                    masterprint('# Input/output', file=file)
                    masterprint("IC_file = '{}'".format(sensible_path(snapshot_filename)),
                                file=file)
                    masterprint('# Unit system', file=file)
                    masterprint("unit_length = '{}'".format(snapshot.units['length']), file=file)
                    masterprint("unit_time = '{}'".format(snapshot.units['time']), file=file)
                    masterprint("unit_mass = '{}'".format(snapshot.units['mass']), file=file)
                    masterprint('# Numerical parameters', file=file)
                    unit = 100*units.km/(units.s*units.Mpc)
                    h = params['H0']/unit
                    value = params['boxsize']*h
                    if isint(value):
                        masterprint('boxsize = {}*{:.12g}*{}'.format(int(round(value)),
                                                                          h, unit_length),
                                    file=file)
                    else:
                        masterprint('boxsize = {:.12g}*{}'.format(params['boxsize'], unit_length),
                                    file=file)
                    masterprint('# Cosmological parameters', file=file)
                    unit = units.km/(units.s*units.Mpc)
                    unit_str = unicode('km s⁻¹ Mpc⁻¹')
                    masterprint('H0 = {:.12g}*km/(s*Mpc)'.format(params['H0']/unit), file=file)
                    masterprint('a_begin = {:.12g}'.format(params['a']), file=file)
                    masterprint('{} = {:.12g}'.format(unicode('Ωm'), params['Ωm']), file=file)
                    masterprint('{} = {:.12g}'.format(unicode('ΩΛ'), params['ΩΛ']), file=file)
            masterprint('\nThe above parameters have been written to\n"{}"'.format(parameter_filename))
            # Done writing out parameters
            continue
        # Print out heading stating the filename
        heading = '\nInformation about "{}"'.format(sensible_path(snapshot_filename))
        masterprint(terminal.bold(heading))
        # Print out snapshot type
        snapshot_type = get_snapshot_type(snapshot_filename)
        masterprint('{:<19} {}'.format('Snapshot type', snapshot_type))
        # Print out unit system for standard snapshots
        if snapshot_type == 'standard':
            masterprint('{:<19} {}'.format('unit_length', snapshot.units['length']))
            masterprint('{:<19} {}'.format('unit_time',   snapshot.units['time']))
            # The mass is typically some large number written in
            # exponential notation. Print it out nicely.
            mass_num, mass_basicunit = (snapshot.units['mass'].replace('m_sun', 'm☉')
                                                              .replace(' ', '')
                                                              .replace('*m', ' m').split(' '))
            mass_num_fmt = significant_figures(float(mass_num), 6, fmt='unicode', incl_zeros=False)
            masterprint('{:<19} {}'.format('unit_mass', '{} {}'.format(mass_num_fmt,
                                                                       mass_basicunit)))
        # Print out global parameters
        unit = units.km/(units.s*units.Mpc)
        unit_str = unicode('km s⁻¹ Mpc⁻¹')
        masterprint('{:<19} {:.12g} {}'.format('H0', params['H0']/unit, unit_str))
        masterprint('{:<19} {:.12g}'.format('a', params['a']))
        # The boxsize should also be printed as boxsize/h, if integer
        unit = 100*units.km/(units.s*units.Mpc)
        h = params['H0']/unit
        value = params['boxsize']*h
        alt_str = ''
        if isint(value) and not isint(params['boxsize']):
            alt_str = ' = {:.12g}/{:.12g} {}'.format(int(round(value)), h, unit_length)
        masterprint('{:<19} {:.12g} {}{}'.format('boxsize',
                                                 params['boxsize'],
                                                 unit_length, alt_str))
        masterprint('{:<19} {:.12g}'.format(unicode('Ωm'), params['Ωm']))
        masterprint('{:<19} {:.12g}'.format(unicode('ΩΛ'), params['ΩΛ']))
        # Print out component information
        for component in snapshot.components:
            masterprint(component.name + ':')
            # General attributes
            masterprint('{:<15} {}'.format('species', component.species), indent=4)
            value = component.mass/units.m_sun
            masterprint('{:<15} {} {}'.format('mass',
                                              significant_figures(value, 6,
                                                                  fmt='unicode', incl_zeros=False),
                                              unicode('m☉')), indent=4)
            # Representation-specific attributes
            if component.representation == 'particles':
                cube_root_N = float(component.N)**ℝ[1/3]
                if isint(cube_root_N):
                    # Print both the particle number N and its
                    # cube root, if integer.
                    masterprint('{:<15} {} = {}{}'.format('N',
                                                          component.N,
                                                          int(round(cube_root_N)), unicode('³')),
                                indent=4)
                else:
                    masterprint('{:<15} {}'.format('N', component.N), indent=4)
            elif component.representation == 'fluid':
                masterprint('{:<15} {}'.format('gridsize', component.gridsize), indent=4)
        # Print out GADGET header for GADGET snapshots
        if snapshot_type == 'gadget2':
            masterprint('GADGET header:')
            for key, val in params['header'].items():
                masterprint('{:<15} {}'.format(key, val), indent=4)

