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
                                               ('s33', cell_states[11,:]),
                                               ('alpha', cell_states[12,:])])
    
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