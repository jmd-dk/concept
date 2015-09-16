# This file is part of CONCEPT, the cosmological N-body code in Python.
# Copyright (C) 2015 Jeppe Mosgard Dakin.
#
# CONCEPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CONCEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CONCEPT. If not, see http://www.gnu.org/licenses/
#
# The auther of CONCEPT can be contacted at
# jeppe.mosgaard.dakin(at)post.au.dk
# The latest version of CONCEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module. In the .pyx file,
# this line will be replaced by the content of commons.py itself.
from commons import *

# Seperate but equivalent imports in pure Python and Cython
if not cython.compiled:
    import graphics, analysis
    from IO import get_snapshot_type, load_into_standard
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    cimport graphics, analysis
    from IO cimport get_snapshot_type, load_into_standard
    """



# Entry point to this module. Call this function to perform a special
# operation, defined by the special_params dict.
@cython.header()
def delegate():
    if special_params['special'] == 'powerspectrum':
        # Powerspectra of one or more snapshots should be made
        powerspec()
    elif special_params['special'] == 'render':
        # A snapshot should be rendered
        render()
    else:
        masterwarn('Special flag "{}" not recognized!'
                   .format(special_params['special']))

# Function for finding all snapshots in special_params['path']
@cython.header(# Locals
               filenames='list',
               msg='str',
               snapshot_filenames='list',
               returns='list',
               )
def get_snapshots():
    # Get all files from the path
    if master and not os.path.exists(special_params['path']):
        msg = 'Path "{}" does not exist!'.format(special_params['path'])
        raise Exception(msg)
    if os.path.isdir(special_params['path']):
        filenames = [os.path.join(special_params['path'], filename)
                     for filename in os.listdir(special_params['path'])
                     if os.path.isfile(os.path.join(special_params['path'],
                                                    filename))]
    else:
        filenames = [special_params['path']]
    # Only use snapshots
    snapshot_filenames = [filename for filename in filenames
                          if get_snapshot_type(filename)]
    # If none of the files where snapshots, throw an exception
    if master and not snapshot_filenames:
        if os.path.isdir(special_params['path']):
            msg = ('The directory "{}" does not contain any snapshots.'
                   ).format(special_params['path'])
        else:
            msg = ('The file "{}" is not a valid snapshot.'
                   ).format(special_params['path'])
        raise Exception(msg)
    return snapshot_filenames

# Function that produces powerspectra according to the
# special_params['path'] parameter.
@cython.header(# Locals
               filename='str',
               filenames='list',
               snapshot='StandardSnapshot',
               )
def powerspec():
    # Get list of snapshots
    filenames = get_snapshots()
    # Produce a powerspectrum of each snapshot
    for filename in filenames:
        # Read in the snapshot
        snapshot = load_into_standard(filename, write_msg=False)
        # Compute powerspectrum of the snapshot
        analysis.powerspectrum(snapshot.particles,
                               filename + '_powerspec')

# Function that produces renders according to the
# special_params['path'] parameter.
@cython.header(# Locals
               filename='str',
               filenames='list',
               snapshot='StandardSnapshot',
               )
def render():
    # Get list of snapshots
    filenames = get_snapshots()
    # Produce a render of each snapshot
    for filename in filenames:
        # Read in the snapshot
        snapshot = load_into_standard(filename, write_msg=False)
        # Render the snapshot
        graphics.render(snapshot.particles,
                        snapshot.params['a'],
                        filename + '.png',
                        boxsize=snapshot.params['boxsize'])
