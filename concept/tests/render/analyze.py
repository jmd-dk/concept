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



# This file has to be run in pure Python mode!

# Include the concept_dir in the searched paths and get directory of this file
import sys, os
sys.path.append(os.environ['concept_dir'])
this_dir = os.path.dirname(os.path.realpath(__file__))

# Imports from the CONCEPT code
from commons import *

# Read in the three images
import matplotlib.pyplot as plt
render_path   = this_dir + '/output/snapshot.png'
render_0_path = this_dir + '/output/subdir/snapshot_0.png'
render_1_path = this_dir + '/output/subdir/snapshot_1.png'
render   = plt.imread(render_path)
render_0 = plt.imread(render_0_path)
render_1 = plt.imread(render_1_path)

# Printout error message for unsuccessful test
if not np.all(render_0 == render_1):
    masterwarn(('The renders "{}" and "{}" are not identical!'
                ).format(render_0, render_1))
    sys.exit(1)




tol = 1e-2
#if np.mean(dist/boxsize) > tol:
#    masterwarn('The results from CONCEPT disagree with those from GADGET.\n'
#               + 'See "{}" for a visualization.'.format(fig_file))
#    sys.exit(1)

