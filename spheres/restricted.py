import pdb
from pathlib import Path
import unittest
from tqdm import tqdm
import utils
import potentials
from jax import (
    random,
    vmap,
    hessian,
    jacfwd,
    jit,
    value_and_grad,
    grad,
    lax,
    checkpoint,
    clear_backends,
)
import os
import optax
from jaxopt import implicit_diff, GradientDescent
from checkpoint import checkpoint_scan
import functools
import jax.numpy as jnp
from jax_md import energy, space, simulate
from jax_md import rigid_body as orig_rigid_body
import potentials
import jax_transformations3d as jts
from jax.config import config

config.update("jax_enable_x64", True)
import itertools
import numpy as np
import jax.numpy as jnp
import unittest
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R

euler_scheme = "sxyz"
V = 54000.0

SEED = 42
main_key = random.PRNGKey(SEED)

init_params = jnp.array(
    [2.0, 2.5, 1e-12, 1.0, 0.5]
)  # morse_eps, morse_alpha, rep_A, rep_alpha, kbT
displacement_fn, shift_fn = space.free()


def quat_to_euler(quaternions):
    """
    Converts a batch of normalized quaternions to Euler angles (3-2-1 sequence: roll, pitch, yaw).

    Args:
        quaternions (np.ndarray): An array of shape (N, 4) where each row represents a quaternion [w, x, y, z].

    Returns:
        np.ndarray: An array of shape (N, 3) where each row represents Euler angles [roll, pitch, yaw] in radians.
    """
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=1)


file_path_quaternion = "rb_orientation_vec.npy"
file_path_xyz_coordinate = "rb_center.npy"
file_path_shape = "vertex_shape_points.npy"

file_paths = [file_path_quaternion, file_path_xyz_coordinate, file_path_shape]


def load_rb_orientation_vec(file_paths):
    """
    Load the rigid body orientation vector from an .npy file.

    Args:
        file_path (str): Path to the .npy file.

    Returns:
        jnp.ndarray: Loaded array as a JAX array.
    """
    rb_orientation_vec = jnp.load(file_path_quaternion).astype(jnp.float64)
    rb_center_vec = jnp.load(file_path_xyz_coordinate).astype(jnp.float64)
    rb_shape_vec = jnp.load(file_path_shape).astype(jnp.float64)

    orientation_sets = rb_orientation_vec.reshape(-1, 4)
    center_sets = rb_center_vec.reshape(-1, 3)

    # euler_from_quaternion_fn = vmap(lambda quat: jts.euler_from_quaternion(quat, euler_scheme))
    euler_orientations = quat_to_euler(np.array(orientation_sets))

    combined = np.hstack([np.array(center_sets), euler_orientations])
    # combined = np.hstack([np.array(center_sets), orientation_sets])
    full_shell = jnp.array(combined)
    shapes = rb_shape_vec

    return full_shell, shapes


def get_icos_shape_and_species(size):

    base_shape = load_rb_orientation_vec(file_paths)[1]
    base_species = jnp.array([0, 1, 1, 1, 1, 1])

    # base_shape = base_shape.reshape(6, 3)

    return jnp.array([base_shape for _ in range(size)]), jnp.array(
        [base_species for _ in range(size)]
    )


def are_blocks_connected_rb(vertex_coords, vertex_radius=2.0):
    """
    Check connectivity between all vertices of an icosahedron based on positional data.

    Args:
        vertex_coords (np.ndarray): An array of vertex positions and orientations (12, 6).
        vertex_radius (float): Radius of the icosahedron.

    Returns:
        np.ndarray: A binary adjacency matrix where 1 indicates connectivity.
    """
    # Extract only the positional data (first three columns)
    positions = vertex_coords[:, :3]

    # Compute pairwise distances
    distances = distance_matrix(positions, positions)

    # Determine edge length threshold (connectivity distance)
    phi = (1 + np.sqrt(5)) / 2
    edge_length = np.sqrt(2 * (1 + phi)) * vertex_radius

    # Generate adjacency matrix
    adjacency_matrix = (distances <= edge_length).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix


def is_configuration_connected_rb(indices, adj_matrix):
    """
    Check if a configuration of vertices is connected.

    Args:
        indices (jnp.ndarray): List of vertex indices in the configuration.
        adj_matrix (np.ndarray): Adjacency matrix of the full graph.

    Returns:
        bool: True if the configuration is connected, False otherwise.
    """
    indices = list(map(int, indices))

    visited = set()
    to_visit = {indices[0]}
    while to_visit:
        current = to_visit.pop()
        visited.add(current)
        neighbors = set(np.where(adj_matrix[current] == 1)[0])
        to_visit.update(neighbors & set(indices) - visited)

    return visited == set(indices)


def generate_connected_subsets_rb(vertex_coords, adj_matrix):
    """
    Iteratively remove vertices while maintaining connectivity, preserving orientations.

    Args:
        vertex_coords (np.ndarray): The full set of vertex coordinates and orientations (12, 6).
        adj_matrix (np.ndarray): Adjacency matrix of the full graph.

    Returns:
        list: A list of connected configurations (each configuration is an array of shape (N, 6)).
    """
    # Initialize variables
    num_vertices = len(vertex_coords)
    all_configs = [vertex_coords.copy()]

    current_config = vertex_coords
    current_adj_matrix = adj_matrix

    # Iteratively remove vertices
    for n in range(num_vertices - 1, 0, -1):  # From full structure down to 1 vertex
        for i in range(len(current_config)):
            remaining_indices = list(range(len(current_config)))
            del remaining_indices[i]  # Remove the i-th vertex
            remaining_indices = jnp.array(remaining_indices)

            if is_configuration_connected_rb(remaining_indices, current_adj_matrix):
                new_config = current_config[remaining_indices]
                all_configs.append(new_config)

                current_config = new_config
                current_adj_matrix = current_adj_matrix[
                    jnp.ix_(remaining_indices, remaining_indices)
                ]
                break
    all_configs = all_configs[::-1]

    return all_configs


def generate_all_4_monomer_configs(vertex_coords, adjacency_matrix, target_size=4):
    """
    Generate all possible unique connected configurations of size `target_size` 
    starting from a single monomer.

    Args:
        vertex_coords (np.ndarray): Coordinates of the vertices.
        adjacency_matrix (np.ndarray): Adjacency matrix representing connectivity.
        target_size (int): Desired size of the connected configurations (default is 4).

    Returns:
        np.ndarray: An array of unique connected configurations of size `target_size`.
    """
    def explore(current_config, remaining_vertices):
        # If we reached the target size, save this configuration
        if len(current_config) == target_size:
            # Sort for canonical representation to avoid duplicates
            current_config_sorted = tuple(sorted(current_config))
            unique_configs.add(current_config_sorted)
            return

        # Explore neighbors of the current configuration
        for vertex in current_config:
            neighbors = set(np.where(adjacency_matrix[vertex] == 1)[0])
            for neighbor in neighbors:
                if neighbor in remaining_vertices:  # Only consider unvisited vertices
                    explore(current_config + [neighbor], remaining_vertices - {neighbor})

    # Initialize
    unique_configs = set()

    # Start from each vertex as the initial configuration
    for start_vertex in range(len(vertex_coords)):
        explore([start_vertex], set(range(len(vertex_coords))) - {start_vertex})

    # Build the final configurations using vertex coordinates
    final_configs = [vertex_coords[np.array(list(config))] for config in unique_configs]

    return jnp.array(final_configs)

full_shell_coord = load_rb_orientation_vec(file_paths)[0]
adjacency_matrix = are_blocks_connected_rb(full_shell_coord, 2)

rb_2 = generate_connected_subsets_rb(full_shell_coord, adjacency_matrix)[1]
configs_4 = generate_all_4_monomer_configs(full_shell_coord, adjacency_matrix, target_size=4)

def invert_monomer_orientation_relative_to_patch(configs_2, connecting_patch_idx):
    """
    Invert the orientation of the second monomer relative to a specified connecting patch.

    Args:
        configs_2 (np.ndarray): Array of shape (2, 6) representing two monomers.
        connecting_patch_idx (int): Index of the patch to rotate around (0-5 for a 6-patch system).

    Returns:
        np.ndarray: Updated configuration with the second monomer's orientation and position inverted.
    """
    configs_2 = configs_2.copy()  # Ensure we don't modify the input array directly
    first_monomer = configs_2[0]
    second_monomer = configs_2[1]

    # Extract positional and orientation data
    first_vertex = first_monomer[:3]
    second_vertex = second_monomer[:3]
    orientation = second_monomer[3:]  # Orientation of the second monomer

    # Vector pointing from the first vertex to the second vertex
    connection_vector = second_vertex - first_vertex

    # Normalize the connection vector
    if np.linalg.norm(connection_vector) == 0:
        raise ValueError("The monomers are at the same position; cannot define a connection vector.")
    connection_vector /= np.linalg.norm(connection_vector)

    # Define a rotation matrix for 180 degrees around the connection vector
    rotation_matrix = R.from_rotvec(np.pi * connection_vector).as_matrix()

    # Apply rotation to the second monomer's vertex and orientation
    rotated_vertex = first_vertex + rotation_matrix @ (second_vertex - first_vertex)
    rotated_orientation = rotation_matrix @ orientation

    # Update the second monomer in configs_2
    configs_2[1, :3] = rotated_vertex
    configs_2[1, 3:] = rotated_orientation

    return configs_2



# Example Usage
configs_2 = generate_all_4_monomer_configs(full_shell_coord, adjacency_matrix, target_size=2)[1]
print("Before inversion:")
print(configs_2)

# Invert the orientation of the second monomer relative to the first patch
#configs_2_inv = invert_monomer_orientation_relative_to_patch(configs_2, connecting_patch_idx=0)

print("After inversion:")
print(configs_2.shape)
#configs_2_tot = jnp.array([rb_2, configs_2_inv])

#configs_2 = generate_all_4_monomer_configs(full_shell_coord, adjacency_matrix, target_size=2)[0]


configs_true = configs_4[:4]
print("shape 4", configs_4.shape)
#configs_2_inv = jnp.array(configs_2_inv)
#configs_2_flat = [config.flatten() for config in configs_2_tot]

#print(configs_2_flat[1].shape)
   
#configs_4 = np.array(configs_4)

vertex_species = 0
n_species = 2
vertex_radius = 2.0
small_value = 1e-12

rep_rmax_table = jnp.full((n_species, n_species), 2 * vertex_radius)


def make_tables(opt_params):
    morse_eps_table = jnp.full((n_species, n_species), opt_params[0])
    morse_eps_table = morse_eps_table.at[0, :].set(small_value)
    morse_eps_table = morse_eps_table.at[:, 0].set(small_value)

    morse_narrow_alpha = opt_params[1]
    morse_alpha_table = jnp.full((n_species, n_species), morse_narrow_alpha)
    # rep_A_table = (jnp.full((n_species, n_species), opt_params[2]).at[vertex_species, vertex_species].set(small_value))
    rep_A_table = (
        jnp.full((n_species, n_species), small_value)
        .at[vertex_species, vertex_species]
        .set(opt_params[2])
    )
    rep_alpha_table = jnp.full((n_species, n_species), opt_params[3])

    return morse_eps_table, morse_alpha_table, rep_A_table, rep_alpha_table


def pairwise_morse(ipos, jpos, i_species, j_species, opt_params):
    morse_eps_table = make_tables(opt_params)[0]
    morse_d0 = morse_eps_table[i_species, j_species]
    morse_alpha = make_tables(opt_params)[1][i_species, j_species]
    morse_r0 = 0.0
    morse_rcut = 8.0 / morse_alpha + morse_r0
    dr = space.distance(ipos - jpos)
    return potentials.morse_x(
        dr,
        rmin=morse_r0,
        rmax=morse_rcut,
        D0=morse_d0,
        alpha=morse_alpha,
        r0=morse_r0,
        ron=morse_rcut / 2.0,
    )


morse_func = vmap(
    vmap(pairwise_morse, in_axes=(None, 0, None, 0, None)),
    in_axes=(0, None, 0, None, None),
)


def pairwise_repulsion(ipos, jpos, i_species, j_species, opt_params):
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = make_tables(opt_params)[2][i_species, j_species]
    rep_alpha = make_tables(opt_params)[3][i_species, j_species]
    dr = space.distance(ipos - jpos)
    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)


inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0, None))
rep_func = vmap(inner_rep, in_axes=(0, None, 0, None, None))


def get_nmer_energy_fn(n):
    pairs = jnp.array(list(itertools.combinations(np.arange(n), 2)))

    def nmer_energy_fn(q, pos, species, opt_params):
        positions = utils.get_positions(q, pos)
        pos_slices = [(i * 6, (i + 1) * 6) for i in range(n)]
        species_slices = [(i * 6, (i + 1) * 6) for i in range(n)]

        all_pos = jnp.stack([positions[start:end] for start, end in pos_slices])
        species = jnp.concatenate(
            [jnp.array(s).reshape(-1) for s in species]
        )  # Ensure species elements are properly reshaped
        all_species = jnp.stack([species[start:end] for start, end in species_slices])

        def pairwise_energy(pair):
            i, j = pair
            morse_energy = morse_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j], opt_params
            ).sum()
            rep_energy = rep_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j], opt_params
            ).sum()
            return morse_energy + rep_energy

        all_pairwise_energies = vmap(pairwise_energy)(pairs)
        total_energy = all_pairwise_energies.sum()

        # Print intermediate values for debugging

        return total_energy

    return nmer_energy_fn


def hess(energy_fn, q, pos, species, opt_params):
    H = hessian(energy_fn)(q, pos, species, opt_params)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def compute_zvib(energy_fn, q, pos, species, opt_params):
    evals, _ = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(
        jnp.sqrt(2.0 * jnp.pi / ((opt_params[4]) * (jnp.abs(evals[6:]) + 1e-12)))
    )
    return zvib


def compute_zrot_mod_sigma(energy_fn, q, pos, species, opt_params, key, nrandom=10000):
    Nbb = len(pos)
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)

    def set_nu_random(key):
        quat = jts.random_quaternion(None, key)
        angles = jnp.array(jts.euler_from_quaternion(quat, euler_scheme))
        nu0 = jnp.full((Nbb * 6,), 0.0)
        return nu0.at[3:6].set(angles)

    def ftilde(nu):
        nu = nu.astype(jnp.float32)
        q_tilde = jnp.matmul(evecs.T[6:].T, nu[6:])
        nu_tilde = jnp.reshape(jnp.array([nu[:6] for _ in range(Nbb)]), nu.shape)
        return utils.add_variables_all(q_tilde, nu_tilde)

    key, *splits = random.split(key, nrandom + 1)
    nus = vmap(set_nu_random)(jnp.array(splits))
    nu_fn = lambda nu: jnp.abs(jnp.linalg.det(jacfwd(ftilde)(nu)))
    Js = vmap(nu_fn)(nus)
    J = jnp.mean(Js)
    Jtilde = 8.0 * (jnp.pi**2) * J
    return Jtilde, Js, key


def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma, V=V):
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib

def load_sigmas(file_path):
    sigmas = {}
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            try:
                shell, sigma = line.strip().split(',')
                size = int(shell.split('_')[1].split('.')[0])
                sigmas[size] = int(sigma)
            except (IndexError, ValueError) as e:
                print(f"Error parsing line: {line.strip()} - {e}")
    return sigmas

def adjust_sigmas(sigmas):
    adjusted_sigmas = {}
    for size, sigma in sigmas.items():
        if size > 1:
            adjusted_sigmas[size] = sigma * (5 ** (size-1))
        else:
            adjusted_sigmas[size] = 1
    return adjusted_sigmas


def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma, V=V):
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib


sigmas_ext = load_sigmas("symmetry_numbers.txt")
sigmas = adjust_sigmas(sigmas_ext)


full_shell_coord = load_rb_orientation_vec(file_paths)[0]
# print(full_shell_coord)
# print(full_shell_coord.shape)
adj_ma = are_blocks_connected_rb(full_shell_coord)
rbs = generate_connected_subsets_rb(full_shell_coord, adj_ma)[1]
rbs = [rb.flatten() for rb in rbs]
shapes_species = [get_icos_shape_and_species(size) for size in range(1, 13)]
shapes, species = zip(*shapes_species)



import os

# Create the directory if it doesn't exist
output_dir = "pos_4_files"
os.makedirs(output_dir, exist_ok=True)

configs_true = configs_4[:4]
#pdb.set_trace()
configs_true_flat = [config.flatten() for config in configs_true]


for inx in range(4):
    pos72 = utils.get_positions(configs_true_flat[inx], shapes_species[3][0])
    vertex_radius = 2.0

    box_size = 30.0
    patch_radius = 0.5
    vertex_color = "43a5be"
    patch_color = "4fb06d"
    body_pos = pos72.reshape(-1, 3)
    assert len(body_pos.shape) == 2
    assert body_pos.shape[0] % 6 == 0
    n_vertices = body_pos.shape[0] // 6
    if n_vertices != 12:
        print(f"WARNING: writing shell body with only {n_vertices} vertices")
    assert body_pos.shape[1] == 3

    box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}"
    vertex_def = f'def V "sphere {vertex_radius*2} {vertex_color}"'
    patch_def = f'def P "sphere {patch_radius*2} {patch_color}"'

    position_lines = list()
    for num_vertex in range(n_vertices):
        vertex_start_idx = num_vertex * 6

        # vertex center
        vertex_center_pos = body_pos[vertex_start_idx]
        vertex_line = (
            f"V {vertex_center_pos[0]} {vertex_center_pos[1]} {vertex_center_pos[2]}"
        )
        position_lines.append(vertex_line)

        for num_patch in range(5):
            patch_pos = body_pos[vertex_start_idx + num_patch + 1]
            patch_line = f"P {patch_pos[0]} {patch_pos[1]} {patch_pos[2]}"
            position_lines.append(patch_line)

    all_lines = [box_def, vertex_def, patch_def] + position_lines + ["eof"]
    
    # Write to a new .pos file for each index
    output_file = os.path.join(output_dir, f"shell_{inx}.pos")
    with open(output_file, "w+") as of:
        of.write("\n".join(all_lines))

pdb.set_trace()

energy_fns = {12: jit(get_nmer_energy_fn(12))}

mon_energy_fn = lambda q, pos, species, opt_params: 0.0


zrot_mod_sigma_1, _, main_key = compute_zrot_mod_sigma(
    mon_energy_fn, rbs[0], shapes[0], species[0], init_params, main_key
)
zvib_1 = 1.0
boltzmann_weight = 1.0

z_1 = jnp.array([compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1, sigmas[1])])
log_z_1 = jnp.log(z_1)

zrot_mod_sigma, Js, main_key = compute_zrot_mod_sigma(
    energy_fns[12],
    rbs[11],  # Size 12 corresponds to index 11 in zero-based indexing
    shapes[11],
    species[11],
    init_params,
    main_key,
)
zrot_mod_sigma_values = [zrot_mod_sigma]



def get_log_z_all(opt_params):
    energy_fn = energy_fns[12]
    shape = shapes[11]
    rb = rbs[11]
    specie = species[11]
    sigma = sigmas[12]
    zrot_mod_sigma = zrot_mod_sigma_values[0]
    zvib = compute_zvib(energy_fn, rb, shape, specie, opt_params)
    e0 = energy_fn(rb, shape, specie, opt_params)
    boltzmann_weight = jnp.exp(-e0 / opt_params[4])
    log_z = (
        (-e0 / opt_params[4])
        + jnp.log(zvib)
        + jnp.log(zrot_mod_sigma)
        + jnp.log(V)
        - jnp.log(sigma)
    )
    log_z_all = jnp.array([log_z])
    log_zs = jnp.concatenate([log_z_1, log_z_all], axis=0)

    return log_zs


nper_structure = jnp.array([1, 12])
init_conc = jnp.array([0.001])


def loss_fn(log_concs_struc, log_z_list):
    tot_conc = init_conc
    log_mon_conc = jnp.log(tot_conc)

    mon_val = jnp.log(jnp.dot(nper_structure, jnp.exp(log_concs_struc)))
    mon_loss = jnp.sqrt((mon_val - log_mon_conc) ** 2).squeeze()


    def struc_loss_fn(struc_idx):
        log_vcs = jnp.log(V) + log_concs_struc[struc_idx]

        n_sa = nper_structure[struc_idx]
        log_vca = jnp.log(V) + log_concs_struc[struc_idx]
        vcs_denom = n_sa * log_vca
        log_zs = log_z_list[struc_idx]

        log_zalpha = log_z_list[struc_idx]
        z_denom = n_sa * log_zalpha

        # Compute loss
        loss = jnp.sqrt((log_vcs - vcs_denom - log_zs + z_denom) ** 2)

        return loss

    
    struc_loss = vmap(struc_loss_fn)(jnp.arange(1, 13))
    combined_loss = jnp.concatenate([jnp.array([mon_loss]), struc_loss])  # Add mon_loss to combined_loss as a 1D array
    loss_var = jnp.var(combined_loss)

    loss_var = jnp.var(combined_loss)
    # loss_max = jnp.max(combined_loss)

    tot_loss = jnp.linalg.norm(combined_loss) + loss_var
    return tot_loss, combined_loss, loss_var


def optimality_fn(log_concs_struc, log_z_list):
    return grad(
        lambda log_concs_struc, log_z_list: loss_fn(log_concs_struc, log_z_list)[0]
    )(log_concs_struc, log_z_list)


@implicit_diff.custom_root(optimality_fn)
def inner_solver(init_guess, log_z_list):
    gd = GradientDescent(
        fun=lambda log_concs_struc, log_z_list: loss_fn(log_concs_struc, log_z_list)[0],
        maxiter=80000,
        implicit_diff=True,
    )
    sol = gd.run(init_guess, log_z_list)

    final_params = sol.params
    final_loss, combined_losses, _ = loss_fn(final_params, log_z_list)
    max_loss = jnp.max(combined_losses)
    second_max_loss = jnp.partition(combined_losses, -2)[-2]

    return final_params


def ofer(opt_params):
    log_z_list = get_log_z_all(opt_params)
    tot_conc = init_conc
    struc_concs_guess = jnp.full(2, jnp.log(0.001 / 2))
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list)
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / jnp.sum(fin_concs)
    target_yield = jnp.log(yields[-1])
    return target_yield


def ofer_grad_fn(opt_params, desired_yield_val):
    target_yield = ofer(opt_params)
    loss = (desired_yield_val - jnp.exp(target_yield)) ** 2
    return loss


num_params = len(init_params)
mask = jnp.zeros(num_params)
#mask = mask.at[0].set(1.0)
mask = mask.at[-1].set(1.0)

def masked_grads(grads):
    return grads * mask


our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=False))
params = init_params
outer_optimizer = optax.adam(1e-2)
opt_state = outer_optimizer.init(params)

n_outer_iters = 200
outer_losses = []


param_names = [f"morse_eps"]
param_names += [f"morse_alpha"]
param_names += [f"rep_A"]
param_names += [f"rep_alpha"]
param_names += [f"kbT"]



desired_yields_range = jnp.array([0.5])



os.makedirs("Temp", exist_ok=True)

with open("Temp/restricted.txt", "w") as f:
    for des_yield in desired_yields_range:

        for i in tqdm(range(n_outer_iters)):
            loss, grads = our_grad_fn(params, des_yield)
            # outer_losses.append(loss)
            grads = masked_grads(grads)
            updates, opt_state = outer_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            #params = project(params)
            print("Updated Parameters:")
            for name, value in {
                name: params[idx] for idx, name in enumerate(param_names)
            }.items():
                print(f"{name}: {value}")
            print(params)
            fin_yield = ofer(params)
            fin_yield = jnp.exp(fin_yield)
            print(f"Desired Yield: {des_yield}, Yield: {fin_yield}")

        final_params = params
        fin_yield = ofer(params)
        final_target_yields = jnp.exp(fin_yield)

        f.write(f"{des_yield},{final_target_yields},{params[-1]}\n")
        #f.write(f"{des_yield}, {final_target_yields}, {params[0]}, {params[-1]}\n")
        f.flush() 


print("All results saved.")





