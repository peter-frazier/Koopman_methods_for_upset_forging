import sys
sys.path.insert(0, "/home/frazier.626/Simple_Koopman_Forging") # TODO: change to folder containing jax-fem-checkpoint

import jax
import jax.numpy as jnp
import numpy as onp
import os
import time
import json
import h5py
import argparse
from plasticity import Plasticity
from jax_fem_checkpoint.solver import solver
from jax_fem_checkpoint.generate_mesh import cylinder_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem_checkpoint.funcs import save_sim, pack_states
from jax_fem_checkpoint import logger

# -- Arguments -----------------------------------------------------------------
parser = argparse.ArgumentParser(description='Data Generation')

# Data
parser.add_argument('--data_name',  default='15-5PH_SS_20C',
                    help='label used for data in folder')
parser.add_argument('--x_key',      default='X',   help='key for states in .mat file')
parser.add_argument('--u_key',      default='U',   help='key for controls in .mat file')

# Mesh
parser.add_argument('--R',          type=float,   default=5.,
                    help='radius of cylinder mesh')
parser.add_argument('--H',          type=float,   default=10.,
                    help='height of cylinder mesh')
parser.add_argument('--circle_mesh',type=int,     default=5,
                    help='number of meshes in circle lines')
parser.add_argument('--height_mesh',type=int,     default=20,
                    help='number of meshes in height')
parser.add_argument('--rect_ratio', type=float,   default=0.4,
                    help='rect length/R')

# Material parameters
parser.add_argument('--E',          type=float,   default=196.e3,
                    help='Youngs Modulus [MPa]')
parser.add_argument('--sig0',       type=float,   default=1172.,
                    help='Initial yield stress [MPa]')
parser.add_argument('--Q',          type=float,   default=145.,
                    help='Hardening saturation [MPa]')
parser.add_argument('--b',          type=float,   default=25.,
                    help='Hardening rate')

# Simulation
parser.add_argument('--n_traj',     type=int,   default=1000,
                    help='number of trajectories to generate')
parser.add_argument('--traj_len',   type=int,   default=101,
                    help='number of snapshots per trajectory')
parser.add_argument('--disp_range', type=float, default=0.01,
                    help='bounds of relative displacement')
parser.add_argument('--IC_range',   type=float, default=4e-4,
                    help='bounds of nodal IC displacement')
parser.add_argument('--seed',       type=int,   default=101)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from jax import config
config.update("jax_enable_x64", True)

# -- Problem -------------------------------------------------------------------

class Plasticity_isotropic(Plasticity):
    def set_params(self, params):
        int_vars, scale, sol, rho_ini = params
        scale = jax.lax.stop_gradient(scale)
        int_vars = jax.lax.stop_gradient(int_vars)
        sol = jax.lax.stop_gradient(sol)
        self.scale = scale
        a1, a2, a3, a4, a5, a6 = int_vars

        full_params = jnp.ones((self.fe.num_cells, len(rho_ini)))
        full_params = full_params.at[self.fe.flex_inds].set(rho_ini)
        self.thetas = jnp.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.internal_vars = [a1, a2, a3, a4, a5, self.thetas]

# -- Random Inputs/Initial Conditions-------------------------------------------

def get_random_inputs(key):
    inputs = onp.zeros(shape=(traj_len,), dtype=onp.float32)
    i = 0

    while i < traj_len:
        key_new = jax.random.fold_in(key, data=i)
        j = 0
        disp = jax.random.uniform(key_new, shape=(), minval=-args.disp_range, maxval=args.disp_range, dtype=jnp.float32)
        hold = jax.random.randint(key_new, shape=(), minval=1, maxval=10)

        while i < traj_len and j < hold:
            if i == 0:
                inputs[0] = disp
            else:
                inputs[i] = inputs[i-1] + disp
            i += 1
            j += 1

    inputs = jnp.array(inputs)
    return inputs

def get_initial_conditions(key):
    sol_IC = jax.random.uniform(key, shape=mesh.points.shape, minval=-args.IC_range, maxval=args.IC_range, dtype=jnp.float32)
    return sol_IC

# -- Simulation -----------------------------------------------------------------

def runSimulations(key):
    start_date = time.strftime('%B %d %Y %-I:%M:%S %p')
    keys = jax.random.split(key, n_traj)

    # Initialize data snapshot matrices
    n_u   = 1
    n_x   = 12*len(mesh.cells)+3*len(mesh.points)
    U     = onp.zeros((n_traj, traj_len-1, n_u), dtype=onp.float32)  # inputs
    X     = onp.zeros((n_traj, traj_len,   n_x), dtype=onp.float32)  # vector states
    alpha = onp.zeros((n_traj, traj_len, len(mesh.cells)), dtype=onp.float32)  # alpha hardening parameter (currently unused)
    times = onp.zeros((n_traj,), dtype=onp.float32)                  # time to run through inputs

    for i in range(n_traj):
        # Initialize random displacements 
        disps  = get_random_inputs(keys[i])       #(traj_len,)
        sol_IC = get_initial_conditions(keys[i])  #(n_nodes, dim)

        # Problem definition
        location_fns = [bottom, top]
        value_fns = [dirichlet_val_bottom, get_dirichlet_top(0)]
        vecs = [2, 2]
        dirichlet_bc_info = [location_fns, vecs, value_fns]

        int_vars = problem.internal_vars
        rho_ini = jnp.array([1.,1.,1.,1.]) # dummy parameter
        sol = sol_IC                       # for initial guess
        int_vars = problem.update_int_vars_gp(sol, int_vars) # update sigmas, log strains based on IC

        # Run simulation
        sim_snapshots = onp.zeros((traj_len, n_x), dtype=onp.float32)
        try:
            for j, disp in enumerate(disps):
                print(f"\nSimulation {i+1} of {n_traj}")
                print(f"Step {j} in {len(disps)}, disp = {disp}")

                if j==1:
                    start = time.time() # solving for first saved state not included in time

                # Update Dirichlet BC to next displacement
                dirichlet_bc_info[-1][-1] = get_dirichlet_top(disp)
                problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)

                # Find solution
                problem.set_params([int_vars, 0, sol, rho_ini])
                solver_options_jax = {'jax_solver': {}, 'initial_guess': sol, 'line_search_flag': False}
                sol = solver(problem, solver_options_jax)[0]

                # Update internal variables
                a1, a2, a3, a4, a5, a6 = int_vars
                int_vars = problem.update_int_vars_gp(sol, int_vars) # update sigmas, log strains based on sol
                _, _, _, _, a5_updated, _ = int_vars
                int_vars_copy =(a1, a2, a3, a4, a5_updated, a6) #a5_updated

                # Calculate cell-wise Cauchy Stress, Log Strain Tensors (weighted average by Gauss points)
                log_strain = problem.compute_log_strain(sol, int_vars_copy) # (num_cells, num_quads, vec, dim)
                log_strain = jnp.sum(log_strain * problem.fes[0].JxW[:, :, None, None], 1) / jnp.sum(problem.fes[0].JxW, axis=1)[:, None, None] 

                cell_sigma = problem.compute_stress(sol, int_vars_copy)
                cell_sigma = jnp.sum(cell_sigma * problem.fes[0].JxW[:, :, None, None], 1) / jnp.sum(problem.fes[0].JxW, axis=1)[:, None, None]

                a = jnp.sum(a3 * problem.fes[0].JxW, 1) / jnp.sum(problem.fes[0].JxW, axis=1)
                alpha[i, j, :] = a

                # Separate out tensor components and save solution
                vtk_path = os.path.join(sim_path, f'{args.data_name}_{i:03d}_step{j:03d}.vtu')
                cell_states = jnp.stack([log_strain[:,0,0],
                                         log_strain[:,0,1],
                                         log_strain[:,0,2],
                                         log_strain[:,1,1],
                                         log_strain[:,1,2],
                                         log_strain[:,2,2],
                                         cell_sigma[:,0,0],
                                         cell_sigma[:,0,1],
                                         cell_sigma[:,0,2],
                                         cell_sigma[:,1,1],
                                         cell_sigma[:,1,2],
                                         cell_sigma[:,2,2]], axis=0)
                cell_dict = [('log strain XX',    cell_states[0,:]),
                             ('log strain XY',    cell_states[1,:]),
                             ('log strain XZ',    cell_states[2,:]),
                             ('log strain YY',    cell_states[3,:]),
                             ('log strain YZ',    cell_states[4,:]),
                             ('log strain ZZ',    cell_states[5,:]),
                             ('Cauchy stress XX', cell_states[6,:]),
                             ('Cauchy stress XY', cell_states[7,:]),
                             ('Cauchy stress XZ', cell_states[8,:]),
                             ('Cauchy stress YY', cell_states[9,:]),
                             ('Cauchy stress YZ', cell_states[10,:]),
                             ('Cauchy stress ZZ', cell_states[11,:])]
                save_sim(dir=vtk_path, fe=problem.fes[0], cell_dict=cell_dict, node_states=sol)

                # Add solution to state vector, state vector to array
                sim_snapshots[j,:] = pack_states(cell_states, sol)

            end = time.time()
            run = end - start
            times[i] = run
            logger.debug(f'Simulation {i+1} of {n_traj} ran in {run} seconds.')         
        except Exception as e:
            logger.debug(f'Simulation {i+1} of {n_traj} failed at step {j+1} of {traj_len}.')
            print(e)
            continue
    
        # Add simulation data to snapshot collections
        U[i, :, :] = jnp.expand_dims(disps[1:], axis=1)
        X[i, :, :] = sim_snapshots

    end_date = time.strftime('%B %d %Y %-I:%M:%S %p')

    # Find where the simulations have failed
    fail_idx = onp.where(onp.all(X==0, axis=(1, 2)))[0]
    times = onp.delete(times, fail_idx)

    return U, X, alpha, times, fail_idx, start_date, end_date

# -- Main Code -----------------------------------------------------------------

if __name__ == "__main__":
    # Define folder paths
    top_dir  = os.path.dirname(__file__)               # Repository folder
    data_dir = os.path.join(top_dir, 'data')           # Data folder
    sim_path = os.path.join(data_dir, args.data_name)  # Folder of JAX-FORGE ParaView simulations

    # Define simulation parameters
    n_traj   = args.n_traj    # Simulations being run
    traj_len = args.traj_len  # Snapshots per simulation
    key      = jax.random.key(args.seed)
    
    # Create cylinder mesh
    ele_type  = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)

    R, H, rect_ratio = args.R, args.H, args.rect_ratio
    circle_mesh, hight_mesh = args.circle_mesh, args.height_mesh
    meshio_mesh = cylinder_mesh_gmsh(data_dir=data_dir, 
                                    R=R, 
                                    H=H, 
                                    circle_mesh=circle_mesh, 
                                    hight_mesh=hight_mesh, 
                                    rect_ratio=rect_ratio)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Define boundary locations.
    def top(point):
        return jnp.isclose(point[2], H, atol=1e-5)

    def bottom(point):
        return jnp.isclose(point[2], 0., atol=1e-5)

    # Define Dirichlet boundary values.
    def dirichlet_val_bottom(point):
        return 0.

    def get_dirichlet_top(disp):
        def val_fn(point):
            return disp
        return val_fn 

    # Initialize problem
    problem = Plasticity_isotropic(mesh=mesh, 
                                   ele_type=ele_type, 
                                   vec=3, 
                                   dim=3, 
                                   dirichlet_bc_info=None, 
                                   p_bounds=False,
                                   E=args.E,
                                   sig0=args.sig0,
                                   Q=args.Q,
                                   b=args.b)

    logger.info(f'\nRunning {n_traj} simulations of length {traj_len}'
                f'\n E [MPa]: {args.E}'
                f'\n sig0 [MPa]: {args.sig0}'
                f'\n Q [MPa]: {args.Q}'
                f'\n b: {args.b}')

    # Simulate data
    U, X, alpha, times, fail_idx, start_date, end_date = runSimulations(key)

    logger.info(f'{n_traj-len(fail_idx)} simulation(s) ran in {onp.sum(times)} seconds')
    logger.info(f'{len(fail_idx)} simulation(s) lost to failure')
    
    # Save metadata
    info_path = os.path.join(data_dir, args.data_name+'.txt')
    with open(info_path, 'w') as f:
        f.write(f'Data collection started: {start_date}\n')
        f.write(f'Data collection ended {end_date}\n\n')
        f.write(f'Youngs modulus [MPa]: {args.E}\n')
        f.write(f'Initial yield stress [MPa]: {args.sig0}\n')
        f.write(f'Hardening saturation [MPa]: {args.Q}\n')
        f.write(f'Hardening rate: {args.b}\n\n')
        f.write(f'Number of simulations: {n_traj}\n')
        f.write(f'Number of snapshots per simulation: {traj_len}\n')
        f.write(f'Number of failed simulations: {len(fail_idx)}\n')
        f.write(f'Failed simulations: {fail_idx}\n')
        f.write(f'Mean simulation time [s]: {onp.mean(times):.3f}\n')
        f.write(f'Std simulation time [s]: {onp.std(times):.3f}\n')

    info_path = os.path.join(data_dir, args.data_name+'.json')
    sim_info = {
        'n_traj': n_traj,
        'traj_len': traj_len,
        'fail_idx': fail_idx.tolist(),
        'sim_times': times.tolist()
    }
    with open(info_path, 'w') as f:
        json.dump(sim_info, f)

    # Save data, failed sim numbers, times
    with h5py.File(os.path.join(data_dir, args.data_name+'.mat'), 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('U', data=U)
        f.create_dataset('alpha', data=alpha)