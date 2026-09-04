import argparse
import itertools
import textwrap

"""
LRAN hyperparameter sweep -- job script generator for OSC (SLURM).

EDIT THIS FILE to configure your sweep:
  DATASET, GRID, SEEDS, FIXED

Then run:
    python create_script.py       # generates .sh files + submit_batch*.sh
    bash submit_batch1.sh         # on the cluster login node

check_results.py imports GRID, SEEDS, and result_dir from here automatically.
"""

parser = argparse.ArgumentParser(description='LRAN model grid')

parser.add_argument('--data_name',    default='Isothermal_Plasticity',
                        help='label used for data in folder')
parser.add_argument('--mode',         default='sweep',
                        help='data ablation, hyperparameter sweep, or cross-validation')
parser.add_argument('--n_seeds', type = int, default=1,
                        help='number of seeds in sweep')

args = parser.parse_args()

# Dataset
X_KEY  = 'X'
U_KEY  = 'U'

def create_grid(mode):
    if mode == 'sweep':
        # Hyperparameter grid (all combinations are swept)
        GRID = {
            'gamma_id':   [1.0],
            'gamma_fwd':  [0.5, 1.0, 2.0, 4.0],
            'gamma_lin':  [0.5, 1.0, 2.0, 4.0],
            'gamma_eig':  [0.0, 0.5, 1.0, 2.0],
            'train_num':  [210], # TODO: change this to whatever ideal value
            'shift_frac': [0.0]
        }
        SEEDS = list(range(args.n_seeds))

        # Fixed hyperparameters (held constant across all jobs)
        FIXED = {
            'n_z':        512,
            'n_h':        4,
            'alpha':      32,
            'activation': 'LeakyReLU',
            'steps':      10,
            'lr':         1e-4,
            'wd':         1e-4,
            'batch_size': 128,
            'epochs':     500,
            'valid_num':  30, 
            'test_num':   60
        }

    elif mode == 'ablation':
        # Hyperparameter grid (all combinations are swept)
        GRID = {
            'gamma_id':   [1.0],
            'gamma_fwd':  [1.0],
            'gamma_lin':  [1.0],
            'gamma_eig':  [0.0],
            'train_num':  [800, 400, 200, 100, 50, 25, 12, 6, 3],
            'shift_frac': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        SEEDS = list(range(args.n_seeds))

        # Fixed hyperparameters (held constant across all jobs)
        FIXED = {
            'n_z':        512,
            'n_h':        4,
            'alpha':      32,
            'activation': 'LeakyReLU',
            'steps':      10,
            'lr':         1e-4,
            'wd':         1e-4,
            'batch_size': 128,
            'epochs':     500,
            'valid_num':  100, 
            'test_num':   0
        }

    elif mode == 'cross-valid':
        # Hyperparameter grid (all combinations are swept)
        GRID = {
            'gamma_id':   [1.0], # TODO: change all these to whatever ideal value
            'gamma_fwd':  [1.0],
            'gamma_lin':  [1.0],
            'gamma_eig':  [0.0],
            'train_num':  [210],
            'shift_frac': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        SEEDS = list(range(args.n_seeds))

        # Fixed hyperparameters (held constant across all jobs)
        FIXED = {
            'n_z':        512,
            'n_h':        4,
            'alpha':      32,
            'activation': 'LeakyReLU',
            'steps':      10,
            'lr':         1e-4,
            'wd':         1e-4,
            'batch_size': 128,
            'epochs':     500,
            'valid_num':  30,
            'test_num':   60
        }
    return GRID, FIXED, SEEDS

# SLURM configuration 
ACCOUNT        = 'PAS3353'
EMAIL          = 'frazier.626@osu.edu'
JOB_TIME       = '72:00:00'
NODES          = 1
TASKS_PER_NODE = 48
CONDA          ='jax-fem-env'
JOBS_PER_BATCH = 990


#  Helpers (module-level so check_results.py can import them)

def _fmt(v):
    """Format a value for use in directory names and shell args."""
    if isinstance(v, float) and v != 0 and (abs(v) < 1 or abs(v) >= 1e4):
        return f'{v:.0e}'
    return str(v)


def all_combos(GRID):
    keys = list(GRID.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


def _tag(p, seed):
    return (f'{args.data_name}_{args.mode}_tnum{p["train_num"]}_sfac{p["shift_frac"]}_gid{p["gamma_id"]}'
            f'_gfwd{p["gamma_fwd"]}_glat{p["gamma_lin"]}_geig{p["gamma_eig"]}_seed{seed}')


def result_dir(p, seed):
    return f'metrics_lran_{_tag(p, seed)}'


def _slurm_header(job_name):
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -J ondemand/sys/myjobs/basic_python_serial
        #SBATCH --job-name={job_name}
        #SBATCH --time={JOB_TIME}
        #SBATCH --nodes={NODES} --ntasks-per-node={TASKS_PER_NODE}
        #SBATCH --account={ACCOUNT}

        module load miniconda3/24.1.2-py310
        source activate {CONDA}
        export PYTHONPATH="/users/PAS3353/peterfrazier/.local/bin:$PYTHONPATH"

        cd $SLURM_SUBMIT_DIR

        """)


def _submit_header(batch_name):
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={batch_name}
        #SBATCH --time=0:30:00
        #SBATCH --nodes=1 --ntasks-per-node=1 --mem=2G
        #SBATCH --account={ACCOUNT}
        #SBATCH --mail-type=BEGIN,END,FAIL
        #SBATCH --mail-user={EMAIL}

        cd $SLURM_SUBMIT_DIR

        """)


def _fixed_arg_str():
    return ' '.join(f'--{k} {v}' for k, v in FIXED.items())


# Job generation 

if __name__ == '__main__':
    GRID, FIXED, SEEDS = create_grid(args.mode)

    job_files = []

    for p in all_combos(GRID):
        for seed in SEEDS:
            out   = result_dir(p, seed)
            name  = f'lran_{_tag(p, seed)}'
            fname = f'{name}.sh'

            sweep_args = (
                f'--data_name {args.data_name} '
                f'--train_num {p["train_num"]} --shift_frac {p["shift_frac"]} '
                f'--gamma_id {p["gamma_id"]} --gamma_fwd {p["gamma_fwd"]} '
                f'--gamma_lin {p["gamma_lin"]} --gamma_eig {p["gamma_eig"]} '
                f'--seed {seed} --out_dir {out}'
            )

            with open(fname, 'w') as fh:
                fh.write(_slurm_header(name))
                fh.write(f'python3 -u driver.py {sweep_args} {_fixed_arg_str()}\n')

            job_files.append(fname)

    n_batches = (len(job_files) + JOBS_PER_BATCH - 1) // JOBS_PER_BATCH
    for b in range(n_batches):
        batch = job_files[b * JOBS_PER_BATCH : (b + 1) * JOBS_PER_BATCH]
        bname = f'submit_batch{b + 1}'
        with open(f'{bname}.sh', 'w') as fh:
            fh.write(_submit_header(bname))
            for jf in batch:
                fh.write(f'dos2unix {jf}\n')
                fh.write(f'sbatch {jf}\n')

    print(f'Created {len(job_files)} job scripts-> {n_batches} submission batch(es).')
    print('On the cluster: sbatch submit_batch1.sh')
