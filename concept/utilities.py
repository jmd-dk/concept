# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: you can redistribute it and/or modify
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
# The auther of CO𝘕CEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    import graphics, analysis
    from snapshot import get_snapshot_type, load_into_standard, load_params
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    cimport graphics, analysis
    from snapshot cimport get_snapshot_type, load_into_standard, load_params
    """



# Entry point to this module when calling from within CO𝘕CEPT itself.
# Call this function to perform a special operation,
# defined by the special_params dict.
@cython.header()
def delegate():
    if special_params['special'] == 'powerspec':
        # Powerspectra of one or more snapshots should be made
        powerspec()
    elif special_params['special'] == 'render':
        # A snapshot should be rendered
        render()
    elif special_params['special'] == 'snapshot_info':
        # Information about one or more snapshots should be printed
        snapshot_info()
    else:
        masterwarn('Special flag "{}" not recognized!'
                   .format(special_params['special']))

# Function for finding all snapshots in special_params['snapshot_filename']
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
        raise Exception(msg)
    if os.path.isdir(path):
        filenames = [os.path.join(path, filename)
                     for filename in os.listdir(path)
                     if os.path.isfile(os.path.join(path, filename))]
    else:
        filenames = [path]
    # Only use snapshots
    snapshot_filenames = [filename for filename in filenames
                          if get_snapshot_type(filename)]
    # If none of the files where snapshots, throw an exception
    if master and not snapshot_filenames:
        if os.path.isdir(path):
            msg = ('The directory "{}" does not contain any snapshots.'
                   ).format(path)
        else:
            msg = ('The file "{}" is not a valid snapshot.'
                   ).format(path)
        raise Exception(msg)
    return snapshot_filenames

# Function that produces a powerspectrum of the file
# specified by the special_params['snapshot_filename'] parameter.
@cython.header(# Locals
               basename='str',
               output_dir='str',
               output_filename='str',
               snapshot_filename='str',
               snapshot='StandardSnapshot',
               )
def powerspec():
    # Extract the snapshot filename
    snapshot_filename = special_params['snapshot_filename']
    # Read in the snapshot
    snapshot = load_into_standard(snapshot_filename, compare_params=False)
    # Output file
    output_dir, basename = os.path.split(snapshot_filename)
    output_filename = '{}/{}{}{}'.format(output_dir,
                                         output_bases['powerspec'],
                                         '_' if output_bases['powerspec'] else '',
                                         basename)
    # Produce powerspectrum of the snapshot
    analysis.powerspec(snapshot.particles,
                       output_filename)

# Function that produces a render of the file
# specified by the special_params['snapshot_filename'] parameter.
@cython.header(# Locals
               basename='str',
               output_dir='str',
               output_filename='str',
               snapshot_filename='str',
               snapshot='StandardSnapshot',
               )
def render():
    # Extract the snapshot filename
    snapshot_filename = special_params['snapshot_filename']
    # Read in the snapshot
    snapshot = load_into_standard(snapshot_filename, compare_params=False)
    # Output file
    output_dir, basename = os.path.split(snapshot_filename)
    output_filename = '{}/{}{}{}'.format(output_dir,
                                         output_bases['render'],
                                         '_' if output_bases['render'] else '',
                                         basename)
    # Render the snapshot
    graphics.render(snapshot.particles,
                    snapshot.params['a'],
                    output_filename)

# Function for printing all informations within a snapshot
@cython.pheader(# Locals
                heading='str',
                particle_attribute='dict',
                path='str',
                snapshot_filenames='list',
                snapshot_type='str',
                unit='double',
                unit_str='str',
                value='double',
                )
def snapshot_info():
    # This function should not run in parallel
    if not master:
        return
    # Extract the path to snapshot(s)
    path = special_params['path']
    # Get list of snapshots
    snapshot_filenames = locate_snapshots(path)
    # Print out information about each snapshot
    for i, snapshot_filename in enumerate(snapshot_filenames):
        # Print out heading stating the filename
        heading = ('{}Information about "{}"'.format('' if i == 0 else '\n',
                                                     sensible_path(snapshot_filename)))
        masterprint(terminal.bold(heading))
        # Print out snapshot type
        snapshot_type = get_snapshot_type(snapshot_filename)
        masterprint('{:<18} {}'.format('Snapshot type', snapshot_type))
        # Load parameters from the snapshot
        params = load_params(snapshot_filename, compare_params=False)
        # Print out global parameters
        masterprint('{:<18} {:.12g}'.format('a', params['a']))
        masterprint('{:<18} {:.12g} {}'.format('boxsize', params['boxsize'], units.length))
        unit = units.km/(units.s*units.Mpc)
        unit_str = 'km s' + unicode('⁻') + unicode('¹') + ' Mpc' + unicode('⁻') + unicode('¹')
        masterprint('{:<18} {:.12g} {}'.format('H0', params['H0']/unit, unit_str))
        masterprint('{:<18} {:.12g}'.format(unicode('Ω') + 'm', params['Ωm']))
        masterprint('{:<18} {:.12g}'.format(unicode('Ω') + unicode('Λ'), params['ΩΛ']))
        # Print out particle information
        for particle_type in params['particle_attributes']:
            masterprint(particle_type + ':')
            particle_attribute = params['particle_attributes'][particle_type]
            masterprint('{:<14} {}'.format('species', particle_attribute['species']), indent=4)
            masterprint('{:<14} {}'.format('N', particle_attribute['N']), indent=4)
            value = particle_attribute['mass']/units.m_sun
            masterprint('{:<14} {:.12e} {}'.format('mass', value, 'm_sun'), indent=4)
        # Print out GADGET header for GADGET snapshots
        if not 'header' in params:
            continue
        masterprint('GADGET header:')
        for key, val in params['header'].items():
            masterprint('{:<14} {}'.format(key, val), indent=4)
