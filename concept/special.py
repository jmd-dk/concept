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
    import graphics
    from IO import get_snapshot_type, load_into_standard
else:
    # Lines in triple quotes will be executed in the .pyx file.
    """
    cimport graphics
    from IO cimport get_snapshot_type, load_into_standard
    """



# Entry point to this module. Call this function to perform a special
# operation, defined by the special_params dict.
@cython.header()
def delegate():
    if special_params['special'] == 'powerspectrum':
        pass
        # Powerspectra of one or more snapshots should be made.
        # Load initial conditions
        #particles = load(IC_file, write_msg=False)
        # Load initial conditions
        #powerspectrum(particles, powerspec_dir + '/' + powerspec_base + '_'
        #                         + basename(IC_file))
    elif special_params['special'] == 'render':
        # A snapshot should be rendered
        render()
    else:
        masterwarn('Special flag "{}" not recognized!'
                   .format(special_params['special']))

# Function that produces renders according to the
# special_params['path'] parameter. If this is the path of a snapshot,
# it will be rendered. If it is a directory, all snapshots in the
# directory will be rendered.
@cython.header(# Locals
               msg='str',
               filename='str',
               filenames='list',
               
               )
def render():
    # Create list of potential files to render
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
    # Loop over all files. Produce renders of those files which happen
    # to be snapshots.
    N_renders = 0
    for filename in filenames:
        # Get the snapshot type
        snapshot_format = get_snapshot_type(filename)
        if not snapshot_format:
            continue
        N_renders += 1
        # Read in the snapshot
        snapshot = load_into_standard(filename, write_msg=False)
        # Render the snapshot
        graphics.render(snapshot.particles,
                        snapshot.params['a'],
                        filename + '.png',
                        boxsize=snapshot.params['boxsize'])
    # If none of the files where snapshots, throw an exception
    if master and N_renders == 0:
        if len(filenames) == 1:
            msg = ('The file "{}" is not a valid snapshot.'
                   ).format(special_params['path'])
        else:
            if len(filenames) == 0:
                msg = ('The directory "{}" does not contain any files.'
                       ).format(special_params['path'])
            else:
                msg = ('None of the files in "{}" are valid snapshots.'
                       ).format(special_params['path'])
        raise Exception(msg)

