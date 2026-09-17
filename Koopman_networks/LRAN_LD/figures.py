import sys
sys.path.insert(0, "/home/frazier.626/Koopman_methods_for_upset_forging") # TODO: change to folder containing jax-fem-checkpoint

import argparse
import os
import h5py
import json
import torch
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat
from model import LRAN_LD
from read_dataset import normalize, denormalize
from driver import evaluate_model
from jax_fem_checkpoint.generate_mesh import cylinder_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem_checkpoint import logger
from jax_fem_checkpoint.funcs import save_sim, unpack_states
from jax_fem_checkpoint.fe_new import FiniteElement


parser = argparse.ArgumentParser(description='LRAN_LD figures')

# Folders
parser.add_argument('--data_name',    default='Isothermal_Plasticity',        help='data being processed')
parser.add_argument('--metrics',      default='metrics',                      help='folder containing data metrics')

# Visuals Details
parser.add_argument('--ablation',     action='store_true',                    help='plot ablation study')

parser.add_argument('--history',      action='store_true',                    help='plot loss function histories from training')
parser.add_argument('--structure',    action='store_true',                    help='plot A and B matrices of model, C if applicable')
parser.add_argument('--eigenvalues',  action='store_true',                    help='plot A matrix eigenvalues')

parser.add_argument('--prediction',   action='store_true',                    help='run model and plot model prediction errors')
parser.add_argument('--sims',         nargs='+',         type=int,            help='which test sims to make time trace of',
                    default=[])
parser.add_argument('--cells',        nargs='+',         type=int,            help='which mesh cells to make time trace of',
                    default=[0, 150, 179, 710])
parser.add_argument('--nodes',        nargs='+',         type=int,            help='which mesh nodes to make time trace of',
                    default=[0, 1118, 54, 159])

args = parser.parse_args()


# Make Paths
data_dir    = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
data_file   = os.path.join(data_dir, args.data_name+'.mat')
meta_file   = os.path.join(data_dir, args.data_name+'.json')
metric_path = os.path.join(os.path.dirname(__file__), args.metrics)
model_file  = os.path.join(metric_path, 'model.pt')
train_file  = os.path.join(metric_path, 'train_metrics.mat')


if args.ablation:
    dict = {}

    for tfac in [0.004, 0.007, 0.013, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]:
        for sfac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            file_path = os.path.join(os.path.dirname(__file__), f'metrics_lran_ld_15-5PH_SS_682C_ablation_tfac{tfac}_sfac{sfac}_gid1.0_gfwd1.0_glat1.0_geig0.0_seed0', 'train_metrics.mat')
            if os.path.exists(file_path):
                metrics = loadmat(file_path)
                NRMSE = float(metrics['step_NRMSE'])
                n_train = int(metrics['n_train'])

                if n_train in list(dict.keys()):
                    dict[n_train].append(NRMSE)
                else:
                    dict[n_train] = [NRMSE]

    for key in list(dict.keys()):
        dict[key] = sum(dict[key])/len(dict[key])

    n_trains = list(dict.keys())
    NRMSEs = list(dict.values())

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.scatter(n_trains, NRMSEs, marker='o', s=200, color='gold')
    ax.set_xlabel('Number of Training Trajectories')
    ax.set_xscale('log')
    ax.set_ylabel('Validation Set NRMSE')
    ax.set_title('LRAN-LD Data Ablation Study')

    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), f'LRAN-LD {args.data_name} Data Ablation Study.png'))


if args.history:
    metrics  = loadmat(train_file)
    loss     = metrics['loss'].squeeze()
    loss_id  = metrics['loss_id'].squeeze()
    loss_fwd = metrics['loss_fwd'].squeeze()
    loss_lin = metrics['loss_lin'].squeeze()
    loss_eig = metrics['loss_eig'].squeeze()
    epochs   = onp.arange(len(loss)) + 1

    fig, axes = plt.subplots(5, 1, figsize=(9,10), sharex=True)

    axes[0].plot(epochs, loss, linewidth=2, color='black', label='Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].set_yscale('log')
    axes[1].plot(epochs, loss_id, color='red', label='Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_yscale('log')
    axes[2].plot(epochs, loss_fwd, color='blue', label='Forward Loss')
    axes[2].set_title('Forward Loss')
    axes[2].set_yscale('log')
    axes[3].plot(epochs, loss_lin, color='green', label='Latent Loss')
    axes[3].set_title('Latent Loss')
    axes[3].set_yscale('log')
    axes[4].plot(epochs, loss_eig, color='purple', label='Stability Loss')
    axes[4].set_title('Stability Loss')
    axes[4].set_xlabel('Epoch')

    fig.savefig(os.path.join(metric_path, f'training_loss.png'))


if args.structure or args.eigenvalues or args.prediction:
    # Load model
    logger.debug('Loading model...')
    model_dict = torch.load(model_file, map_location='cuda', weights_only=False)
    state_dict = model_dict['state_dict']
    args_dict  = model_dict['args']
    n_x        = model_dict['n_x']
    n_u        = model_dict['n_u']
    test_sims  = model_dict['test_sims']
    model      = LRAN_LD(n_x, n_u, args_dict['n_z'], args_dict['n_h'], args_dict['activation'], args_dict['alpha'], args_dict['init_scale'])
    model.load_state_dict(state_dict)
    model.eval()
    model.to('cuda')


if args.structure:
    A = onp.abs(model.A.weight.detach().cpu().numpy())
    B = onp.abs(model.B.weight.detach().cpu().numpy())
    C = onp.abs(model.C.weight.detach().cpu().numpy())

    fig1, ax1 = plt.subplots(figsize=(8,6))
    im1 = ax1.imshow(A, cmap='Blues', norm=LogNorm(vmin=1e-6, vmax=max(onp.max(A),onp.max(B),onp.max(C))))
    ax1.set_title('A Matrix Pattern')
    ax1.set_xticks([])
    ax1.set_yticks([])
    cbar1 = fig1.colorbar(im1)
    cbar1.set_label('Magnitude')
    fig1.savefig(os.path.join(metric_path, f'A_matrix.png'))

    fig2, ax2 = plt.subplots(figsize=(6,6))
    im2 = ax2.imshow(B, cmap='Blues', norm=LogNorm(vmin=1e-6, vmax=max(onp.max(A),onp.max(B),onp.max(C))), aspect='auto')
    ax2.set_title('B Matrix Pattern')
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar2 = fig2.colorbar(im2)
    cbar2.set_label('Magnitude')
    fig2.savefig(os.path.join(metric_path, f'B_matrix.png'))

    fig3, ax3 = plt.subplots(figsize=(6,6))
    im3 = ax3.imshow(C, cmap='Blues', norm=LogNorm(vmin=1e-6, vmax=max(onp.max(A),onp.max(B),onp.max(C))), aspect='auto')
    ax3.set_title('C Matrix Pattern')
    ax3.set_xticks([])
    ax3.set_yticks([])
    cbar3 = fig3.colorbar(im3)
    cbar3.set_label('Magnitude')
    fig3.savefig(os.path.join(metric_path, f'C_matrix.png'))


if args.eigenvalues:
    A = model.A.weight.detach().cpu().numpy()
    eig = onp.linalg.eigvals(A)
    onp.save(os.path.join(metric_path, 'eigenvalues.npy'), eig)
    maxi = max(onp.max(eig.real), -onp.min(eig.real), onp.max(eig.imag), -onp.min(eig.imag))
    fig, ax = plt.subplots(figsize=(8, 10))
    circle = plt.Circle((0.,0.),1., color='r', fill=False)
    ax.add_patch(circle)
    plt.scatter(eig.real, eig.imag)
    plt.xlim(-maxi,maxi)
    plt.ylim(-maxi,maxi)
    plt.grid()
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.title('A matrix eigenvalues')
    plt.tight_layout()
    fig.savefig(os.path.join(metric_path, f'eigenvalues.png'))


if args.prediction:
    plt.close('all')
    # Prepare Mesh
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    R, H, rect_ratio = 5., 10., 0.4
    circle_mesh, hight_mesh = 5, 20
    meshio_mesh = cylinder_mesh_gmsh(data_dir=data_dir, 
                                    R=R, 
                                    H=H, 
                                    circle_mesh=circle_mesh, 
                                    hight_mesh=hight_mesh, 
                                    rect_ratio=rect_ratio)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    fe = FiniteElement(mesh=mesh, vec=3, dim=3, ele_type=ele_type, gauss_order=None, dirichlet_bc_info=None) # just for saving visual data (.vtu)

    # Load and Convert Data
    logger.debug('Loading data...')

    with open(meta_file, 'r') as f:
        sim_info = json.load(f)
    traj_len = sim_info['traj_len']

    with h5py.File(data_file, 'r') as f:
        X = f['X'][:]
        U = f['U'][:]

    # Normalize
    logger.debug('Normalizing data on [-1, 1]...')
    X_n, U_n, scale = normalize(X, U)

    # Run testing evaluation
    logger.debug('Running testing evaluation...')
    X_te, U_te = X_n[test_sims,:,:] , U_n[test_sims,:,:]
    X_te = torch.from_numpy(X_te)
    U_te = torch.from_numpy(U_te)
    test_set_stats = evaluate_model(model, X_te, U_te, test_sims, device='cuda')

    X_pred    = test_set_stats['X_pred']
    errors    = test_set_stats['errors']
    RE        = test_set_stats['RE']
    NRMSE     = test_set_stats['NRMSE']
    NRMSE_sim = test_set_stats['NRMSE_sim']
    times     = test_set_stats['times']
    worst_sims = onp.argmax(NRMSE_sim, axis=0)
    best_sims  = onp.argmin(NRMSE_sim, axis=0)

    logger.info(f'Avg test simulation time [s]: {onp.mean(times)}')
    logger.info(f'Std test simulation time [s]: {onp.std(times)}')
    logger.info(f'{args_dict["steps"]}-step test simulation NRMSE: {NRMSE[args_dict["steps"]]}')

    with open(os.path.join(metric_path, 'test_metrics.txt'), 'w') as f:
        f.write(f'Number of test simulations: {len(times)}\n')
        f.write(f'Mean test simulations time [s]: {onp.mean(times):.6f}\n')
        f.write(f'Std test simulations time [s]: {onp.std(times):.6f}\n')
        f.write(f'{args_dict["steps"]}-step test simulation NRMSE: {NRMSE[args_dict["steps"]]:.6f}')

    # Denormalize Train Data
    logger.debug('Denormalizing data...')
    U_te   = denormalize(U_te,   scale['u_lo'], scale['u_rng'])
    X_te   = denormalize(X_te,   scale['x_lo'], scale['x_rng'])
    X_pred = denormalize(X_pred, scale['x_lo'], scale['x_rng'])
    errors = denormalize(errors, scale['x_lo'], scale['x_rng'])
    
    # Save testing metrics
    logger.debug('Saving testing metrics...')
    with h5py.File(os.path.join(metric_path, 'test_metrics.mat'), 'w') as f:
        f.create_dataset('X_pred', data=X_pred)
        f.create_dataset('errors', data=errors)
        f.create_dataset('RE', data=RE)
        f.create_dataset('NRMSE', data=NRMSE)
        f.create_dataset('NRMSE_sim', data=NRMSE_sim)
        f.create_dataset('times', data=times)
        f.create_dataset('best_sims', data=best_sims)
        f.create_dataset('worst_sims', data=worst_sims)

    # Make .vtu Files for ParaView
    logger.debug('Making testing visualizations...')
    for i, sim in enumerate(test_sims):
        for j in range(traj_len):
            vtk_path = os.path.join(metric_path, f'{args.data_name}_test_sims', f'LRAN_LD_{args.data_name}_test_sim{sim:03d}_step{j:03d}.vtu')
            err_path = os.path.join(metric_path, f'{args.data_name}_test_errs', f'LRAN_LD_{args.data_name}_errs_sim{sim:03d}_step{j:03d}.vtu')
            cell_states, node_states = unpack_states(X_pred[i, j, :], 1600)
            cell_errors, node_errors = unpack_states(errors[i, j, :], 1600)
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
            error_cell_dict = [('error log strain XX',    onp.abs(cell_errors[0,:])),
                               ('error log strain XY',    onp.abs(cell_errors[1,:])),
                               ('error log strain XZ',    onp.abs(cell_errors[2,:])),
                               ('error log strain YY',    onp.abs(cell_errors[3,:])),
                               ('error log strain YZ',    onp.abs(cell_errors[4,:])),
                               ('error log strain ZZ',    onp.abs(cell_errors[5,:])),
                               ('error Cauchy stress XX', onp.abs(cell_errors[6,:])),
                               ('error Cauchy stress XY', onp.abs(cell_errors[7,:])),
                               ('error Cauchy stress XZ', onp.abs(cell_errors[8,:])),
                               ('error Cauchy stress YY', onp.abs(cell_errors[9,:])),
                               ('error Cauchy stress YZ', onp.abs(cell_errors[10,:])),
                               ('error Cauchy stress ZZ', onp.abs(cell_errors[11,:]))]
            save_sim(dir=vtk_path, fe=fe, cell_dict=cell_dict, node_states=node_states)
            save_sim(dir=err_path, fe=fe, cell_dict=error_cell_dict, node_states=node_errors)

        logger.info(f'Saved {i+1} of {len(test_sims)}')
    
    # Plot cell/node time traces
    logger.debug('Plotting reference tracking time traces...')
    assert len(args.cells)==len(args.nodes)

    lines  = ('solid', 'dashed')
    blues  = ('blue', 'lightblue')
    reds   = ('red', 'pink')
    greens = ('green', 'lightgreen')
    blacks = ('black', 'dimgray')

    for n in range(len(args.cells)):
        for q in [args_dict['steps'], traj_len-1]:
            if not len(args.sims)==0:
                sims = args.sims
            else:
                worst_sim = worst_sims[q]
                best_sim  = best_sims[q]
                sims = [best_sim, worst_sim]

            fig, ax = plt.subplots(4, 1, figsize =(10, 12))

            for i, s in enumerate(sims):
                sim        = test_sims[s]
                stress_idx = (args.cells[n]+1)*12-1
                strain_idx = (args.cells[n]+1)*12-7
                disp_idx   = 1600*12 + (args.nodes[n]+1)*3-1
                
                steps = onp.arange(q+1)

                for p in range(4):
                    ax[p].set_xlim(0,q)
                    ax[p].set_xticks(onp.linspace(0, q, 11, endpoint=True))
                    ax[p].set_xticklabels(onp.linspace(0, q, 11, endpoint=True, dtype='int32'))

                ax[0].plot(steps,   X_te[s, :q+1, stress_idx], color=blues[i%2],  linestyle=lines[0], label=f'Sim {sim} Actual z stress')
                ax[0].plot(steps, X_pred[s, :q+1, stress_idx], color=blues[i%2],  linestyle=lines[1], label=f'Sim {sim} Predicted z stress')

                ax[1].plot(steps,   X_te[s, :q+1, strain_idx], color=reds[i%2],   linestyle=lines[0], label=f'Sim {sim} Actual z strain')
                ax[1].plot(steps, X_pred[s, :q+1, strain_idx], color=reds[i%2],   linestyle=lines[1], label=f'Sim {sim} Predicted z strain')

                ax[2].plot(steps,   X_te[s, :q+1, disp_idx],   color=greens[i%2], linestyle=lines[0], label=f'Sim {sim} Actual z displacement')
                ax[2].plot(steps, X_pred[s, :q+1, disp_idx],   color=greens[i%2], linestyle=lines[1], label=f'Sim {sim} Predicted z displacement')

                ax[3].plot(steps[:-1], U_te[s, :q, :],       color=blacks[i%2], linestyle=lines[0], label=f'Sim {sim} Vertical Displacement')

                plt.xlabel(f'Step', fontsize=20)

                for i in range(4):
                    ax[i].legend()
                    ax[i].minorticks_on()
                    
                ax[0].set_title(f'Cell {args.cells[n]} Node {args.nodes[n]} Tracking', fontsize=20)
                ax[0].set_ylabel(f'Stress [MPa]', fontsize=20)
                ax[1].set_ylabel(f'Strain', fontsize=20)
                ax[2].set_ylabel(f'Displacement', fontsize=20)
                ax[3].set_ylabel(f'Input', fontsize=20)

                plt.tight_layout()
                fig.savefig(os.path.join(metric_path, f'LRAN_LD Cell {args.cells[n]}, Node {args.nodes[n]} Tracking ({q} steps).png'))

    logger.debug('Plotting RE and NRMSE distributions...')
    plt.style.use('_mpl-gallery')

    colors = ('darkgray', 'purple')
    types = ('Instantaneous', 'Cumulative')
    errs = ('Relative Error', 'NRMSE')

    for scale in ['log', 'linear']:
        for i, data in enumerate([RE, NRMSE_sim]):
            for step in [args_dict['steps'], traj_len-1]:
                fig, ax = plt.subplots(figsize =(10, 7))
                bp = ax.boxplot(data[:, :step+1], patch_artist=True, positions=list(range(step+1)),
                                boxprops     = dict(facecolor=colors[i], color='black'), 
                                capprops     = dict(color='black'),
                                whiskerprops = dict(color='black'),
                                flierprops   = dict(color='black', markeredgecolor='black'),
                                medianprops  = dict(color='black'),
                                showfliers = True)
                ax.set_xticks(onp.linspace(0, step, 11, endpoint=True))
                ax.set_xticklabels(onp.linspace(0, step, 11, endpoint=True, dtype='int32'))
                plt.minorticks_on()
                plt.xlabel(f'Snapshot Number', fontsize=20)
                plt.ylabel(errs[i], fontsize=20)
                plt.yscale(scale)
                plt.title(f'{types[i]} Testing Errors', fontsize=20)
                plt.tight_layout()
                fig.savefig(os.path.join(metric_path, f'Testing Error Propagation {step} steps ({types[i]}, {scale} scale).png'))

plt.close('all')