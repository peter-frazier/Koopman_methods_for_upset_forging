import argparse
import os
import h5py
import json
import torch
import numpy as onp
import matplotlib.pyplot as plt
from read_dataset import denormalize
from jax_fem.generate_mesh import cylinder_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem_checkpoint import logger
from jax_fem_checkpoint.utils import save_sol
from jax_fem_checkpoint.fe_new import FiniteElement


parser = argparse.ArgumentParser(description='Retrieve Test Error Propagation from model')

# Folders
parser.add_argument('--data_folder',              default='Data',                         help='folder containing data')
parser.add_argument('--dataset',                  default='Isothermal_Plasticity.mat',    help='data being processed')
parser.add_argument('--test'   ,                  default='linear_test.npy',              help='numpy file of model test simulations')
parser.add_argument('--metrics',                  default='metrics',                      help='folder containing data metrics')

# Visuals Details
parser.add_argument('--mode',                     default='Paraview',                     help='what kind of visual to make')
parser.add_argument('--sim',       type=int,      default=1,                              help='which test sim to make a plot of')
parser.add_argument('--cell',      type=int,      default=150,                            help='which mesh cell to make a plot of')

args = parser.parse_args()


# Make Paths
dir        = os.path.dirname(os.path.dirname(__file__))
folder     = os.path.dirname(__file__)
truth_path = os.path.join(dir, args.data_folder, args.dataset)
test_path  = os.path.join(folder, args.metrics, args.test)
model_path = os.path.join(folder, args.metrics, 'model.pt')

def unpack_states(state):
    num_cells   = 1600
    cell_vector = state[:num_cells*12]
    node_vector = state[num_cells*12:]

    cell_states = cell_vector.reshape((12,-1), order='F')
    node_states = node_vector.reshape((-1, 3), order='C')
    return cell_states, node_states

def save_sim(dir, fe, cell_states, node_states):
    save_sol(fe, node_states, dir, cell_infos=[('e11', cell_states[0,:]),
                                               ('e12', cell_states[1,:]),
                                               ('e13', cell_states[2,:]),
                                               ('e22', cell_states[3,:]),
                                               ('e23', cell_states[4,:]),
                                               ('e33', cell_states[5,:]),
                                               ('s11', cell_states[6,:]),
                                               ('s12', cell_states[7,:]),
                                               ('s13', cell_states[8,:]),
                                               ('s22', cell_states[9,:]),
                                               ('s23', cell_states[10,:]),
                                               ('s33', cell_states[11,:])])


# Prepare Mesh
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
R, H, rect_ratio = 5., 10., 0.4
circle_mesh, hight_mesh = 5, 20
meshio_mesh = cylinder_mesh_gmsh(data_dir=os.path.join(dir, args.data_folder), 
                                R=R, 
                                H=H, 
                                circle_mesh=circle_mesh, 
                                hight_mesh=hight_mesh, 
                                rect_ratio=rect_ratio)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
fe = FiniteElement(mesh=mesh, vec=3, dim=3, ele_type=ele_type, gauss_order=None, dirichlet_bc_info=None) # just for saving visual data (.vtu)


# Load and Convert Data
logger.debug('Loading data...')

with open(os.path.join(dir, args.data_folder, 'Simulation_Info.json'), 'r') as f:
    sim_info = json.load(f)

num_sim = sim_info['num_sim']
num_steps = sim_info['num_steps']
test_sims = onp.array(sim_info['test_idx'])

with h5py.File(truth_path, 'r') as f:
    X = f['X'][:] 
    U = f['U'][:]

X, U = X[test_sims,:,:] , U[test_sims,:,:]
X_te = onp.load(test_path)


# Denormalize Train Data
logger.debug('Denormalizing data...')
model_dict = torch.load(model_path, map_location='cuda', weights_only=False)
scale      = model_dict['scale']
X_te       = denormalize(X_te, scale['x_lo'], scale['x_rng'])


# Make .vtu Files for Paraview
logger.debug('Making testing visualizations...')
if args.mode == 'Paraview':
    for i, sim in enumerate(test_sims):
        for j in range(100):
            vtk_path = os.path.join(folder, args.metrics, 'sims', f'BLRAN_test_sim{sim:03d}_step{j:03d}.vtu')
            cell_states, node_states = unpack_states(X_te[i, j, :])
            save_sim(dir=vtk_path, fe=fe, cell_states=cell_states, node_states=node_states)
        logger.info(f'Saved {i+1} of {len(test_sims)}')

elif args.mode == 'plot':
    sim      = test_sims[args.sim]
    cell_idx = (args.cell-1)*12+11
    
    steps = onp.arange(num_steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize =(10, 7))
    ax1.set_xticks(onp.linspace(0, num_steps, 11, endpoint=True))
    ax1.set_xticklabels(onp.linspace(0, num_steps, 11, endpoint=True, dtype='int32'))
    ax2.set_xticks(onp.linspace(0, num_steps, 11, endpoint=True))
    ax2.set_xticklabels(onp.linspace(0, num_steps, 11, endpoint=True, dtype='int32'))

    ax1.plot(steps,         X[args.sim, :, cell_idx],  color='blue',      label=f'Actual z stress')
    ax1.plot(steps,      X_te[args.sim, :, cell_idx],  color='lightblue', label=f'Predicted z stress')
    ax2.plot(steps[:-1],    U[args.sim, :, :],         color='black',     label='Vertical Displacement')

    plt.xlabel(f'Step', fontsize=20)
    ax1.legend()
    ax1.minorticks_on()
    ax1.set_title(f'Simulation {sim} Cell {args.cell} Tracking', fontsize=20)
    ax1.set_ylabel(f'Stress [MPa]', fontsize=20)
    ax2.minorticks_on()
    ax2.set_ylabel(f'Input', fontsize=20)

    plt.tight_layout()
    fig.savefig(os.path.join(folder, args.metrics, 'pics', f'Sim {sim} Cell {args.cell} Tracking.png'))
    plt.show()