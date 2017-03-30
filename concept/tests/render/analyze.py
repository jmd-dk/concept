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



# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Read in the three images
render_path   = this_dir + '/output/render_snapshot.png'
render_0_path = this_dir + '/output/subdir/snapshot_0.png'
render_1_path = this_dir + '/output/subdir/snapshot_1.png'
render   = plt.imread(render_path)
render_0 = plt.imread(render_0_path)
render_1 = plt.imread(render_1_path)

# The two identical renders should be exacty equal
if not np.all(render_0 == render_1):
    abort('The renders "{}" and "{}" are not identical!'.format(render_0, render_1))

# The dimensions of the images should be as stated in
# render.params_0 and render.params_1.
for r, path, params in zip((render, render_0),
                           (render_path, render_0_path),
                           (this_dir + '/render.params_0', this_dir + '/render.params_1')):
    module_dict = imp.load_source('params', params).__dict__
    shape = r.shape[:2]
    if shape[0] != shape[1] or shape[0] != module_dict['resolution']:
        masterprint('done')
        abort('The render "{}" is not of size {}x{}!'
              .format(path, module_dict['resolution'], module_dict['resolution']))

# There should be some completely black pixels in the first render
# and some completely white pixels in the second (and third) render
# due to the text.
if not np.any(render[:, :, :3] > [0.99]*3):
    abort('The scalefactor text do not seem to '
          'be white on render "{}".'.format(render_path))
if not np.any(render_0[:, :, :3] < [0.01]*3):
    abort('The scalefactor text do not seem to '
          'be black on render "{}".'.format(render_0_path))

# Done analyzing
masterprint('done')

