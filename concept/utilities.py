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
cimport('from species import construct_particles, construct_fluid')



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
    If the special_params['params_given'] paramater is True,
    all parameters in the snapshot will be overwritten with the
    values given by the parameter file in use.
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
    # Convert each snapshot sequentially
    for snapshot_filename in special_params['snapshot_filenames']:
        # Read snapshot on disk into the requested type
        snapshot = load(snapshot_filename, compare_params=False,
                                           do_exchange=False,
                                           as_if=snapshot_type)
        # If a parameter file was specified by the user,
        # overwrite parameters in the snapshot with those from that
        # parameter file (those which are currently loaded as globals).
        if special_params['params_given']:
            snapshot.populate(snapshot.components, a_begin)
        # If specific parameters are parsed as attributes, update the
        # snapshot accordingly.
        snapshot.params.update(params)
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
            # Apply particles <--> fluid convertion, if necessary
            if original_representation == 'particles' and component.representation == 'fluid':
                # To do the convertion, the particles need to be
                # distributed according to which domain they are in.
                component.representation = 'particles'
                exchange(component)
                component.representation = 'fluid'
                # The mass attribute is now the mass of each fluid
                # element. Since the total mass of the component should
                # be the same as before, the mass and the gridsize are
                # related. If either mass or gridsize is given by the
                # user, use this to determine the other.
                if 'gridsize' in attributes[name] and 'mass' not in attributes[name]:
                    component.mass *= float(component.N)/component.gridsize**3
                elif 'mass' in attributes[name] and 'gridsize' not in attributes[name]:
                    component.gridsize = int(round((original_mass/component.mass*component.N)
                                                    **ℝ[1/3]))
                    original_tot_mass = original_mass*component.N
                    tot_mass = component.mass*component.gridsize**3
                    if not isclose(original_tot_mass, tot_mass, 1e-6):
                        if tot_mass > original_tot_mass:
                            masterwarn('The specified mass for the "{}" component\n'
                                       'leads to a relative increase of {:.9g}\n'
                                       'for the total mass of this component.'
                                       .format(component.name, tot_mass/original_tot_mass - 1))
                        else:
                            masterwarn('The specified mass for the "{}" component\n'
                                       'leads to a relative decrease of {:.9g}\n'
                                       'for the total mass of this component.'
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
                path='str',
                snapshot='object',
                snapshot_filenames='list',
                snapshot_type='str',
                unit='double',
                unit_str='str',
                value='double',
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
        # Print out heading stating the filename
        heading = ('\nInformation about "{}"'.format(sensible_path(snapshot_filename)))
        masterprint(terminal.bold(heading))
        # Print out snapshot type
        snapshot_type = get_snapshot_type(snapshot_filename)
        masterprint('{:<19} {}'.format('Snapshot type', snapshot_type))
        # Load parameters from the snapshot
        snapshot = load(snapshot_filename, compare_params=False, only_params=True)
        params = snapshot.params
        # Print out global parameters
        masterprint('{:<19} {:.12g}'.format('a', params['a']))
        # The boxsize should also be printed as boxsize*h, if integer
        unit = 100*units.km/(units.s*units.Mpc)
        h = params['H0']/unit
        value = params['boxsize']*h
        alt_str = ''
        if isint(value):
            alt_str = ' = {:.12g}{}{:.12g} {}'.format(int(value), unicode('×'), h, base_length)
        masterprint('{:<19} {:.12g} {}{}'.format('boxsize',
                                                 params['boxsize'],
                                                 base_length, alt_str))
        unit = units.km/(units.s*units.Mpc)
        unit_str = unicode('km s⁻¹ Mpc⁻¹')
        masterprint('{:<19} {:.12g} {}'.format('H0', params['H0']/unit, unit_str))
        masterprint('{:<19} {:.12g}'.format(unicode('Ωm'), params['Ωm']))
        masterprint('{:<19} {:.12g}'.format(unicode('ΩΛ'), params['ΩΛ']))
        # Print out component information
        for component in snapshot.components:
            masterprint(component.name + ':')
            masterprint('{:<15} {}'.format('species', component.species), indent=4)
            masterprint('{:<15} {}'.format('representation', component.representation), indent=4)
            # The cube root of the particle number N
            # should also be printed, if integer.
            cube_root_N = float(component.N)**ℝ[1/3]
            if isint(cube_root_N):
                masterprint('{:<15} {} = {}{}'.format('N',
                                                      component.N,
                                                      int(round(cube_root_N)), unicode('³')),
                            indent=4)
            else:
                masterprint('{:<15} {}'.format('N', component.N), indent=4)
            value = component.mass/units.m_sun
            masterprint('{:<15} {:.12e} {}'.format('mass', value, 'm_sun'), indent=4)
        # Print out GADGET header for GADGET snapshots
        if snapshot_type == 'gadget2':
            masterprint('GADGET header:')
            for key, val in params['header'].items():
                masterprint('{:<15} {}'.format(key, val), indent=4)

