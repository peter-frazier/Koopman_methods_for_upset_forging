import sys
sys.path.insert(0, "/users/PAS3353/peterfrazier/Simple_Koopman_Forging") # TODO: change to folder containing jax-fem-checkpoint

import os
import numpy as onp
import jax.numpy as jnp
from jax_fem_checkpoint.utils import save_sol


# Save array to files
def save_array(dir, array, name):
    os.makedirs(dir, exist_ok=True)
    file = os.path.join(dir, f'{name}')
    onp.save(file, array)


# Saves simulation data to files
def save_sim(dir, fe, cell_dict, node_states, node_errors=None):
    if node_errors is not None:
        save_sol(fe, node_states, dir, cell_infos=cell_dict, point_infos=node_errors)
    else:
        save_sol(fe, node_states, dir, cell_infos=cell_dict)

  
def pack_states(cell_states, node_states):
    cell_vector = jnp.ravel(cell_states, order ='F')
    node_vector = jnp.ravel(node_states, order='C')
    return jnp.concatenate((cell_vector, node_vector))


def unpack_states(state, num_cells):
    assert num_cells == 1600
    cell_vector = state[:num_cells*12]
    node_vector = state[num_cells*12:]

    cell_states = cell_vector.reshape((12,-1), order='F')
    node_states = node_vector.reshape((-1, 3), order='C')
    return cell_states, node_states