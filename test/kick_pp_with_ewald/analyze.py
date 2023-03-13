# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species
plt = get_matplotlib().pyplot

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Read in data from the CO𝘕CEPT snapshots
species.allow_similarly_named_components = True
a = []
x0 = []
x0_std = []
x1 = []
x1_std = []
for fname in sorted(
    glob(f'{this_dir}/output/snapshot_a=*'),
    key=(lambda s: s[(s.index('=') + 1):]),
):
    snapshot = load(fname, compare_params=False)
    posx = snapshot.components[0].posx
    a.append(snapshot.params['a'])
    x0.append(np.mean(posx[:4]))
    x0_std.append(np.std(posx[:4])/np.mean(posx[:4]))
    x1.append(np.mean(posx[4:]))
    x1_std.append(np.std(posx[4:])/np.mean(posx[4:]))
N_snapshots = len(a)

# Read in data from the GADGET snapshots
x0_gadget = []
x0_std_gadget = []
x1_gadget = []
x1_std_gadget = []
order = None
for fname in sorted(glob(f'{this_dir}/Gadget2/output/snapshot_*'))[:N_snapshots]:
    snapshot = load(fname, compare_params=False)
    posx_gadget = snapshot.components[0].posx
    if order is None:
        order = np.argsort(posx_gadget)
    posx_gadget = posx_gadget[order]
    x0_gadget.append(np.mean(posx_gadget[:4]))
    x0_std_gadget.append(np.std(posx_gadget[:4])/np.mean(posx_gadget[:4]))
    x1_gadget.append(np.mean(posx_gadget[4:]))
    x1_std_gadget.append(np.std(posx_gadget[4:])/np.mean(posx_gadget[4:]))

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# Plot
fig_file = f'{this_dir}/result.png'
fig, ax = plt.subplots()
ax.plot(
    np.concatenate((a, a)), np.concatenate((x0, x1)),
    '.',
    markersize=15,
    alpha=0.7,
    label='CO$N$CEPT',
)
ax.plot(
    np.concatenate((a, a)),
    np.concatenate((x0_gadget, x1_gadget)),
    '.',
    markersize=10,
    alpha=0.7,
    label='GADGET',
)
ax.set_xlabel('$a$')
ax.set_ylabel(rf'$x\,\mathrm{{[{unit_length}]}}$')
ax.set_ylim(0, boxsize)
ax.legend()
fig.tight_layout()
fig.savefig(fig_file, dpi=150)

# There should be no variance on the x positions
reltol = 1e-6
if max(x0_std_gadget) > reltol or max(x1_std_gadget) > reltol:
    abort(
        'Unequal x-positions for the 2*4 particles in the GADGET-2 snapshots.\n'
        'It is no good to compare the CO𝘕CEPT results to these.'
    )
if max(x0_std) > reltol or max(x1_std) > reltol:
    abort(
        'Unequal x-positions for the 2*4 particles in the snapshots.\n'
        'The symmetric initial conditions has produced non-symmetrical results!'
    )

# Printout error message for unsuccessful test
reltol = 1e-2
if (   max(abs(asarray(x0)/asarray(x0_gadget) - 1)) > reltol
    or max(abs(asarray(x1)/asarray(x1_gadget) - 1)) > reltol):
    abort(
        f'The results from CO𝘕CEPT disagree with those from GADGET-2.\n'
        f'See "{fig_file}" for a visualization.'
    )

# Done analysing
masterprint('done')
