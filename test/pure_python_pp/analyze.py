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
nprocs_list = sorted(
    int(dname[(dname.index('python_') + 7):])
    for dname in [
        os.path.basename(dname)
        for dname in glob(f'{this_dir}/output_python_*')
    ]
)
components = {
    'cython': {n: [] for n in nprocs_list},
    'python': {n: [] for n in nprocs_list},
}
for cp in components.keys():
    for n in nprocs_list:
        for fname in sorted(
            glob(f'{this_dir}/output_{cp}_{n}/snapshot_a=*'),
            key=(lambda s: s[(s.index('=') + 1):]),
        ):
            snapshot = load(fname, compare_params=False)
            if cp == 'cython' and n == 1:
                a.append(snapshot.params['a'])
            components[cp][n].append(snapshot.components[0])
N_snapshots = len(a)

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# Using the particle order of the cython snapshot as the standard, find the corresponding
# ID's in the python snapshots and order these particles accordingly.
N = components['cython'][1][0].N
D2 = zeros(N, dtype=float)
ID = zeros(N, dtype=int)
for i in range(N_snapshots):
    for n in nprocs_list:
        x_cython = components['cython'][n][i].posx
        y_cython = components['cython'][n][i].posy
        z_cython = components['cython'][n][i].posz
        x_python = components['python'][n][i].posx
        y_python = components['python'][n][i].posy
        z_python = components['python'][n][i].posz
        for j in range(N):
            for k in range(N):
                dx = x_cython[j] - x_python[k]
                if dx > 0.5*boxsize:
                    dx -= boxsize
                elif dx < -0.5*boxsize:
                    dx += boxsize
                dy = y_cython[j] - y_python[k]
                if dy > 0.5*boxsize:
                    dy -= boxsize
                elif dy < -0.5*boxsize:
                    dy += boxsize
                dz = z_cython[j] - z_python[k]
                if dz > 0.5*boxsize:
                    dz -= boxsize
                elif dz < -0.5*boxsize:
                    dz += boxsize
                D2[k] = dx**2 + dy**2 + dz**2
            ID[j] = np.argmin(D2)
        components['python'][n][i].posx[:] = components['python'][n][i].posx[ID]
        components['python'][n][i].posy[:] = components['python'][n][i].posy[ID]
        components['python'][n][i].posz[:] = components['python'][n][i].posz[ID]
        components['python'][n][i].momx[:] = components['python'][n][i].momx[ID]
        components['python'][n][i].momy[:] = components['python'][n][i].momy[ID]
        components['python'][n][i].momz[:] = components['python'][n][i].momz[ID]

# Compute distance between particles in the two snapshots
dist = collections.OrderedDict((n, []) for n in nprocs_list)
for i in range(N_snapshots):
    x = {(cp, n): components[cp][n][i].posx for cp in ('cython', 'python') for n in nprocs_list}
    y = {(cp, n): components[cp][n][i].posy for cp in ('cython', 'python') for n in nprocs_list}
    z = {(cp, n): components[cp][n][i].posz for cp in ('cython', 'python') for n in nprocs_list}
    for n in nprocs_list:
        dist[n].append(sqrt(asarray([
            min([
                + (x['cython', n][j] - x['python', n][j] + xsgn*boxsize)**2
                + (y['cython', n][j] - y['python', n][j] + ysgn*boxsize)**2
                + (z['cython', n][j] - z['python', n][j] + zsgn*boxsize)**2
                for xsgn in (-1, 0, +1)
                for ysgn in (-1, 0, +1)
                for zsgn in (-1, 0, +1)
            ])
            for j in range(N)
        ])))

# Plot
fig_file = f'{this_dir}/result.png'
fig, axes = plt.subplots(len(nprocs_list), sharex=True, sharey=True)
for n, d, ax in zip(dist.keys(), dist.values(), axes):
    for i in range(N_snapshots):
        ax.semilogy(
            machine_ϵ + asarray(d[i])/boxsize,
            '.',
            alpha=0.7,
            label=f'$a={a[i]}$',
            zorder=-i,
        )
    ax.set_ylabel(
        rf'$|\mathbf{{x}}_{{\mathrm{{pp}}{n}}} - \mathbf{{x}}_{{\mathrm{{c}}{n}}}|'
        rf'/\mathrm{{boxsize}}$'
    )
axes[ 0].set_xlim(0, N - 1)
axes[-1].set_xlabel('Particle number')
fig.subplots_adjust(hspace=0)
plt.setp([ax.get_xticklabels() for ax in axes[:-1]], visible=False)
axes[0].legend()
fig.tight_layout()
fig.savefig(fig_file, dpi=150)

# Printout error message for unsuccessful test
tol = 1e-10
if any(np.mean(asarray(d)/boxsize) > tol for d in dist.values()):
    abort(
        f'Some or all pure Python runs with nprocs = {nprocs_list} yielded results '
        f'different from their compiled counterparts!\n'
        f'See "{fig_file}" for a visualization.'
    )

# Compare the two tabulated grids
ewald_grid = {}
for cp in ('cython', 'python'):
    with open_hdf5(f'{this_dir}/ewald_{cp}.hdf5', mode='r') as hdf5_file:
        ewald_grid[cp] = hdf5_file['data'][...]
δ, ϵ = 1e-10, 1e-10
if not all(np.isclose(ewald_grid['cython'], ewald_grid['python'], ϵ, δ)):
    abort(
        'The two tabulated Ewald grids "{}" and "{}" are far from being numerically identical!'
        .format(*[f'{this_dir}/ewald_{cp}.hdf5' for cp in ('cython', 'python')])
    )

# Done analysing
masterprint('done')
