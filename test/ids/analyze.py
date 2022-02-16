# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species

# Absolute path and name of this test
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(os.path.dirname(this_dir))

# Analyze the realization subtest data
masterprint(f'Analysing {this_test} data ...')
species.allow_similarly_named_components = True
def read(subtest, keyfun, snapname='snap', **load_kwargs):
    load_kwargs.setdefault('only_components', True)
    components = {}
    for pure_python in (True, False):
        pure_python_dirname = 'purepython' if pure_python else 'compiled'
        for directory in glob(f'{this_dir}/{subtest}/{pure_python_dirname}/nprocs*'):
            n = int(os.path.basename(directory).removeprefix('nprocs'))
            if os.path.isdir(glob(f'{directory}/*')[0]):
                directories = glob(f'{directory}/*')
            else:
                directories = [directory]
            for directory in directories:
                snaptype = os.path.basename(directory)
                for component in load(
                    glob(f'{directory}/{snapname}*')[0],
                    **load_kwargs,
                ):
                    components[keyfun(pure_python, n, snaptype, component)] = component
    return components
def compare(subtest, components, ϵ=1e-12):
    def build_label(key):
        pure_python = key[0]
        n = key[1]
        label = (('pure Python' if pure_python else 'compiled'), f'nprocs{n}')
        if len(key) == 3:
            label = label + (key[2], )
        return label
    reference = None
    for key, component in components.items():
        # Reorder according to IDs
        ordering = np.argsort(component.ids)
        pos = component.pos_mv3[ordering, :]
        mom = component.mom_mv3[ordering, :]
        if reference is None:
            reference = build_label(key)
            pos_ref = pos
            mom_ref = mom
            mom_std = np.std(mom_ref)
            continue
        if (
               not np.all(np.isclose(pos, pos_ref, ϵ, ϵ*boxsize))
            or not np.all(np.isclose(mom, mom_ref, ϵ, ϵ*mom_std))
        ):
            current = build_label(key)
            abort(
                f'Subtest "{subtest}": Particle IDs of snapshot {current} '
                f'did not match those of {reference}'
            )
subtest = 'realize'
components = read(
    subtest,
    lambda pure_python, n, snaptype, component: (pure_python, n, component.species),
)
ids_species = collections.defaultdict(set)
for s in ('baryon', 'cold dark matter'):
    subcomponents = {
        key: component
        for key, component in components.items()
        if component.species == s
    }
    compare(subtest, subcomponents)
    for component in subcomponents.values():
        ids_species[s] |= set(component.ids)
N = component.N
index = 0
for s, ids in ids_species.items():
    if list(arange(index, index + N)) != list(ids):
        abort(
            f'Subtest "realization": The IDs of the two particle components '
            f'were not assigned in a distinct, contiguous fashion'
        )
    index += N

# Analyze the snapshot subtest data
subtest = 'snapshot'
components = read(
    subtest,
    lambda pure_python, n, snaptype, component: (pure_python, n, snaptype),
    'B',
)
compare(subtest, components)

# Analyze the made-up subtest data
subtest = 'madeup'
components = read(
    subtest,
    lambda pure_python, n, snaptype, component: (pure_python, n, snaptype),
)
for snaptype in ('gadget', 'concept'):
    subcomponents = {
        (pure_python, n, st): component
        for (pure_python, n, st), component in components.items()
        if st == snaptype
    }
    if snaptype == 'gadget':
        ns = sorted({n for pure_python, n, st in subcomponents.keys()})
        for n_ in ns:
            subsubcomponents = {
                (pure_python, n, st): component
                for (pure_python, n, st), component in subcomponents.items()
                if n == n_
            }
            compare(subtest, subsubcomponents)
    else:
        compare(subtest, subcomponents)

# Analyze the evolution subtest data
subtest = 'evolution'
components = read(
    subtest,
    lambda pure_python, n, snaptype, component: (pure_python, n),
    compare_params=False,
)
compare(subtest, components)

# Done analysing
masterprint('done')

