# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
plt = get_matplotlib().pyplot

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Begin analysis
masterprint(f'Analysing {this_dir} data ...')

# Read in the three 3D render images
render3D_path   = f'{this_dir}/output/render3D_snapshot.png'
render3D_0_path = f'{this_dir}/output/subdir/snapshot_0.png'
render3D_1_path = f'{this_dir}/output/subdir/snapshot_1.png'
render3D   = plt.imread(render3D_path)
render3D_0 = plt.imread(render3D_0_path)
render3D_1 = plt.imread(render3D_1_path)

# The two identical 3D renders should be exactly equal
if not np.all(render3D_0 == render3D_1):
    abort('The 3D renders "{}" and "{}" are not identical!'.format(render3D_0, render3D_1))

# The dimensions of the images should be as stated in
# render3D.param_0 and render3D.param_1.
for r, p, param_i in zip(
    (render3D, render3D_0),
    (render3D_path, render3D_0_path),
    (f'{this_dir}/render3D.param_0', f'{this_dir}/render3D.param_1'),
):
    module_dict = load_source('param', param_i).__dict__
    shape = r.shape[:2]
    if shape[0] != shape[1] or shape[0] != module_dict['render3D_resolution']:
        masterprint('done')
        abort('The 3D render "{}" is not of size {}x{}!'
              .format(p, module_dict['render3D_resolution'], module_dict['render3D_resolution']))

# There should be some completely black pixels in the first 3D render
# and some completely white pixels in the second (and third) 3D render
# due to the text.
if not np.any(render3D[:, :, :3] > [0.99]*3):
    abort('The scale factor text do not seem to '
          'be white on 3D render "{}".'.format(render3D_path))
if not np.any(render3D_0[:, :, :3] < [0.01]*3):
    abort('The scale factor text do not seem to '
          'be black on 3D render "{}".'.format(render3D_0_path))

# Done analysing
masterprint('done')
