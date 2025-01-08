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
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm

euler_scheme = "sxyz"
V = 54000.0

SEED = 42
main_key = random.PRNGKey(SEED)

init_params = jnp.array(
    [2.0, 2.5, 5.0, 1.0, 1.0]
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


# adj_matrix_rb = are_blocks_connected_rb(rb_data, vertex_radius=1.0)
# configs_rb = generate_connected_subsets_rb(rb_data, adj_matrix_rb)

# print(configs_rb[0])
vertex_species = 0
n_species = 2
vertex_radius = 2.0
small_value = 1e-14

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
        jnp.sqrt(2.0 * jnp.pi / ((1 / opt_params[4]) * (jnp.abs(evals[6:]) + 1e-12)))
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


def remove_outliers(Js):
    # Convert Js to a numpy array for statsmodels
    Js_np = np.array(Js)
    # Perform outlier test
    test = sm.OLS(Js_np, sm.add_constant(np.arange(len(Js_np)))).fit()
    test_outliers = test.outlier_test()
    # Identify non-outliers
    non_outliers = test_outliers['bonf(p)'] > 0.05
    # Return cleaned Js
    return Js_np[non_outliers]


full_shell_coord = load_rb_orientation_vec(file_paths)[0]
adj_ma = are_blocks_connected_rb(full_shell_coord)
rbs = generate_connected_subsets_rb(full_shell_coord, adj_ma)
rbs = [rb.flatten() for rb in rbs]
shapes_species = [get_icos_shape_and_species(size) for size in range(1, 13)]
shapes, species = zip(*shapes_species)

nrandom = 45000
key = random.PRNGKey(0)
energy_fn = get_nmer_energy_fn(10)
q = rbs[9]
pos = shapes[9]
species = species[9]
opt_params = init_params

Jtilde, Js, key = compute_zrot_mod_sigma(energy_fn, q, pos, species, opt_params, key, nrandom)

# Optional: Uncomment to use outlier removal
# Js_clean = remove_outliers(Js)
# if len(Js_clean) == 0:
#     print(f"No non-outlier values for nrandom = {nrandom}")
#     exit()

# Directly calculate mean from `Js` (or `Js_clean` if using outlier removal)
Jmean = float(jnp.mean(Js))  # Ensure Jmean is a scalar

# Determine bins for histogram
max_js = int(jnp.max(Js)) + 500  # Adjust range dynamically
bins = list(range(0, max_js, 500))

# Plot histogram
plt.hist(Js, bins=bins, alpha=0.75)
nrandom = 45000
key = random.PRNGKey(0)
energy_fn = get_nmer_energy_fn(10)
q = rbs[9]
pos = shapes[9]
species = species[9]
opt_params = init_params

Jtilde, Js, key = compute_zrot_mod_sigma(energy_fn, q, pos, species, opt_params, key, nrandom)

# Optional: Uncomment to use outlier removal
# Js_clean = remove_outliers(Js)
# if len(Js_clean) == 0:
#     print(f"No non-outlier values for nrandom = {nrandom}")
#     exit()

# Directly calculate mean from `Js` (or `Js_clean` if using outlier removal)
Jmean = float(jnp.mean(Js))  # Ensure Jmean is a scalar


print(f"Max value in Js: {jnp.max(Js)}")
print(f"Min value in Js: {jnp.min(Js)}")

# Cap the maximum value dynamically
MAX_ALLOWED_BIN = 10_000  # Prevent range overflow with an upper limit
max_js = int(min(jnp.max(Js) + 500, MAX_ALLOWED_BIN))  # Apply cap here

if max_js > 0:  # Ensure max_js is valid
    bins = list(range(0, max_js, 500))
    plt.hist(Js, bins=bins, alpha=0.75)
    plt.xlabel("Values of Js")
    plt.ylabel("Frequency")
    plt.title("Histogram of Js Values")
    plt.show()
else:
    print("Invalid range for histogram bins. Skipping plot.")
