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
    clear_backends
)
from checkpoint import checkpoint_scan
import functools
import jax.numpy as jnp
from jax_md import energy, space, simulate
from jax_md import rigid_body as orig_rigid_body
import potentials
import jax_transformations3d as jts
# from jax.config import config
import jax
jax.config.update('jax_enable_x64', True)
import itertools
import numpy as np
import jax.numpy as jnp
import unittest
from scipy.spatial import distance_matrix
euler_scheme = "sxyz"
V = 1.0

SEED = 42
main_key = random.PRNGKey(SEED)

init_params = jnp.array([7., 2.5, 5.0, 500, 1.0])
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


def get_icos_rb(ico_radius=1.0):
    """
    Generate the 12 vertices of an icosahedron along with their orientations in Euler angles.

    Args:
        ico_radius (float): Radius of the icosahedron.

    Returns:
        np.ndarray: An array of shape (12, 6) with [x, y, z, roll, pitch, yaw] for each building block.
    """
    phi = 0.5 * (1 + jnp.sqrt(5))
    vertex_coords = ico_radius * jnp.array([
        [phi, 1.0, 0.0],
        [phi, -1.0, 0.0],
        [-1 * phi, 1.0, 0.0],
        [-1 * phi, -1.0, 0.0],
        [1.0, 0.0, phi],
        [-1.0, 0.0, phi],
        [1.0, 0.0, -phi],
        [-1.0, 0.0, -phi],
        [0.0, phi, 1.0],
        [0.0, -phi, 1.0],
        [0.0, phi, -1.0],
        [0.0, -phi, -1.0]
    ])

    central_point = jnp.mean(vertex_coords, axis=0)  # Center of the icosahedron
    d = vmap(lambda pos, center: pos - center, (0, None))
    reoriented_vectors = d(vertex_coords, central_point)
    norm = jnp.linalg.norm(reoriented_vectors, axis=1).reshape(-1, 1)
    reoriented_vectors /= norm

    orig_vec = reoriented_vectors[0]
    crossed = vmap(jnp.cross, (None, 0))(orig_vec, reoriented_vectors)
    dotted = vmap(jnp.dot, (0, None))(reoriented_vectors, orig_vec)

    theta = jnp.arccos(dotted)
    cos_part = jnp.cos(theta / 2).reshape(-1, 1)
    sin_part = crossed * jnp.sin(theta / 2).reshape(-1, 1)
    orientation = jnp.concatenate([cos_part, sin_part], axis=1)
    orientation /= jnp.linalg.norm(orientation, axis=1).reshape(-1, 1)

    # Convert quaternions to Euler angles
    euler_angles = quat_to_euler(np.array(orientation))

    # Combine vertex coordinates and Euler angles
    combined = np.hstack([np.array(vertex_coords), euler_angles])
    combined = jnp.array(combined)

    return combined

def get_icos_shape_and_species(vertex_coords, vertex_radius, size):
    # Get the vertex shape (i.e. the coordinates of a vertex for defining a rigid body)

    anchor = vertex_coords[0]
    d = vmap(displacement_fn, (0, None))

    # Compute all pairwise distances
    dists = space.distance(d(vertex_coords, anchor))

    # Mask the diagonal
    self_distance_tolerance = 1e-5
    large_mask_distance = 100.0
    dists = jnp.where(dists < self_distance_tolerance, large_mask_distance, dists) # mask the diagonal

    # Find nearest neighbors
    # note: we use min because the distances to the nearest neighbors are all the same (they should be 1 diameter away)
    # note: this step is not differentiable, but that's fine: we keep the icosahedron fixed for the optimization
    nbr_ids = jnp.where(dists == jnp.min(dists))[0]
    nbr_coords = vertex_coords[nbr_ids]

    # Compute displacements to neighbors to determine patch positions
    vec = d(nbr_coords, anchor)
    norm = jnp.linalg.norm(vec, axis=1).reshape(-1, 1)
    vec /= norm
    patch_pos = anchor - vertex_radius * vec

    # Collect shape in an array and return
    base_shape = jnp.concatenate([jnp.array([anchor]), patch_pos]) - anchor
    base_species = jnp.array([0, 1, 1, 1, 1, 1])

    base_shape = base_shape.reshape(6, 3)

    return jnp.array([base_shape for _ in range(size)]), jnp.array([base_species for _ in range(size)])

def are_blocks_connected_rb(vertex_coords, vertex_radius=1.0):
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
                current_adj_matrix = current_adj_matrix[jnp.ix_(remaining_indices, remaining_indices)]
                break
    all_configs = all_configs[::-1]

    return all_configs


#adj_matrix_rb = are_blocks_connected_rb(rb_data, vertex_radius=1.0)
#configs_rb = generate_connected_subsets_rb(rb_data, adj_matrix_rb)

#print(configs_rb[0])
vertex_species = 0
n_species = 2
vertex_radius = 1.0
small_value = 1e-14

rep_rmax_table = jnp.full((n_species, n_species), 2 * vertex_radius)

def make_tables(
    opt_params):
    morse_eps_table = jnp.full((n_species, n_species), opt_params[0])
    morse_eps_table = morse_eps_table.at[0, :].set(small_value)
    morse_eps_table = morse_eps_table.at[:, 0].set(small_value)

    morse_narrow_alpha = opt_params[1]
    morse_alpha_table = jnp.full((n_species, n_species), morse_narrow_alpha)
    #rep_A_table = (jnp.full((n_species, n_species), opt_params[2]).at[vertex_species, vertex_species].set(small_value))
    rep_A_table = (jnp.full((n_species, n_species), small_value).at[vertex_species, vertex_species].set(opt_params[2]))
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
        species = jnp.concatenate([jnp.array(s).reshape(-1) for s in species])  # Ensure species elements are properly reshaped
        all_species = jnp.stack(
            [species[start:end] for start, end in species_slices]
        )

        def pairwise_energy(pair):
            i, j = pair
            morse_energy = morse_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j], opt_params
            ).sum()
            rep_energy = rep_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j],
            opt_params).sum()
            return morse_energy + rep_energy

        all_pairwise_energies = vmap(pairwise_energy)(pairs)
        total_energy = all_pairwise_energies.sum()

        # Print intermediate values for debugging
        """
        print(f"n = {n}")
        print(f"positions = {positions}")
        print(f"all_pos = {all_pos}")
        print(f"species = {species}")
        print(f"all_species = {all_species}")
        print(f"all_pairwise_energies = {all_pairwise_energies}")
        print(f"total_energy = {total_energy}")
        """

        return total_energy

    return nmer_energy_fn

def hess(energy_fn, q, pos, species, opt_params):
    H = hessian(energy_fn)(q, pos, species, opt_params)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def compute_zvib(energy_fn, q, pos, species, opt_params):
    evals, _ = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(jnp.sqrt(2.0 * jnp.pi / (opt_params[4]*(jnp.abs(evals[6:]) + 1e-12))))
    return zvib


def compute_zrot_mod_sigma(energy_fn, q, pos, species, opt_params, key, nrandom=10000):
    Nbb = len(pos)
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)

    def set_nu_random(key):
        quat = jts.random_quaternion(None, key)
        angles = jnp.array(jts.euler_from_quaternion(quat, euler_scheme))
        nu0 = jnp.full((Nbb * 6,), 0.)
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
    Jtilde = 8.0 * (jnp.pi ** 2) * J
    return Jtilde, Js, key


def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma, V=V):
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib

def get_sigma(size):

    Nbb = size

    if  Nbb  == 2 or  Nbb == 4:
        s = 2
    if  Nbb == 3:
        s = 3
    elif  Nbb  == 12:
        s = 12
    else:
        s = 1

    return s

def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma, V=V):
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib


sigmas = [get_sigma(size) for size in range(1, 13)]
coo_shell_full = get_icos_rb(ico_radius=1.0)
adj_ma = are_blocks_connected_rb(coo_shell_full)
rbs = generate_connected_subsets_rb(coo_shell_full, adj_ma)
rbs = [rb.flatten() for rb in rbs]
shapes_species = [get_icos_shape_and_species(coo_shell_full, vertex_radius, size) for size in range(1, 13)]


## START RK EDITS


pos72 = utils.get_positions(rbs[-1], shapes_species[-1][0])
vertex_radius = 1.0


# note: body is only a single state, not a trajectory
box_size = 30.0
patch_radius=0.5
vertex_color="43a5be"
patch_color="4fb06d"
body_pos = pos72.reshape(-1, 3)
assert(len(body_pos.shape) == 2)
assert(body_pos.shape[0] % 6 == 0)
n_vertices = body_pos.shape[0] // 6
if n_vertices != 12:
    print(f"WARNING: writing shell body with only {n_vertices} vertices")
assert(body_pos.shape[1] == 3)

box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}"
vertex_def = f"def V \"sphere {vertex_radius*2} {vertex_color}\""
patch_def = f"def P \"sphere {patch_radius*2} {patch_color}\""

position_lines = list()
for num_vertex in range(n_vertices):
    vertex_start_idx = num_vertex*6

    # vertex center
    vertex_center_pos = body_pos[vertex_start_idx]
    vertex_line = f"V {vertex_center_pos[0]} {vertex_center_pos[1]} {vertex_center_pos[2]}"
    position_lines.append(vertex_line)

    for num_patch in range(5):
        patch_pos = body_pos[vertex_start_idx+num_patch+1]
        patch_line = f"P {patch_pos[0]} {patch_pos[1]} {patch_pos[2]}"
        position_lines.append(patch_line)

all_lines = [box_def, vertex_def, patch_def] + position_lines + ["eof"]
with open('my_test.pos', 'w+') as of:
    of.write('\n'.join(all_lines))
pdb.set_trace()


## END RK EDITS
shapes, species = zip(*shapes_species)

energy_fns = {size: jit(get_nmer_energy_fn(size)) for size in range(2, 12+1)}

mon_energy_fn = lambda q, pos, species, opt_params: 0.0


zrot_mod_sigma_1,_, main_key = compute_zrot_mod_sigma(mon_energy_fn, rbs[0], shapes[0],  species[0], init_params, main_key)
zvib_1 = 1.0
boltzmann_weight = 1.0

z_1 = jnp.array([compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1, sigmas[0])])
log_z_1 = jnp.log(z_1)

zrot_mod_sigma_values = []

for size in range(2, 12 + 1):
    zrot_mod_sigma, Js, main_key = compute_zrot_mod_sigma(
        energy_fns[size],
        rbs[size-1],
        shapes[size-1],
        species[size-1],
        init_params,
        main_key
    )
    zrot_mod_sigma_values.append(zrot_mod_sigma)

# print(zrot_mod_sigma_values)

def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))

'''
def get_log_z_all(opt_params):
    def compute_log_z(size):
        energy_fn = energy_fns[size]
        shape = shapes[size-1]
        rb = rbs[size-1]
        specie = species[size-1]
        sigma = sigmas[size-1]
        zrot_mod_sigma = zrot_mod_sigma_values[size-2]  # Adjusted index to size-2
        zvib = compute_zvib(energy_fn, rb, shape, specie, opt_params)
        zvib = jnp.maximum(zvib, 1e-12)
        e0 = energy_fn(rb, shape, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 /opt_params[4])
        z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib, sigma)
        z = jnp.maximum(z, 1e-12)
        return safe_log(z)

    log_z_all = []

    for size in range(2, 12 + 1):
        log_z = compute_log_z(size)
        ###
        if size <= 4:
            log_z = compute_log_z(size)
        else:
            compute_log_z_ckpt = checkpoint(lambda: compute_log_z(size))
            xs = jnp.array(sigmas)

            def scan_fn(carry, x):
                result = compute_log_z_ckpt()
                return carry, result

            checkpoint_freq = 10
            scan_with_ckpt = functools.partial(checkpoint_scan, checkpoint_every=checkpoint_freq)
            _, log_z = scan_with_ckpt(scan_fn, None, xs)
            log_z = jnp.array(log_z)
        ###
        log_z_all.append(log_z)
    log_z_all = jnp.array(log_z_all)
    #log_z_all = jnp.concatenate(log_z_all, axis=0)
    print(log_z_all.shape)
    log_z_all = jnp.concatenate([log_z_1, log_z_all], axis=0)

    return log_z_all
'''
def get_log_z_all(opt_params):
    def compute_log_z(size):
        energy_fn = energy_fns[size]
        shape = shapes[size-1]
        # print(f"shape: {shape}")
        rb = rbs[size-1]
        # print(f"rb: {rb}")
        specie = species[size-1]
        sigma = sigmas[size-1]
        zrot_mod_sigma = zrot_mod_sigma_values[size-2]
        zvib = compute_zvib(energy_fn, rb, shape, specie, opt_params)
        zvib = jnp.maximum(zvib, 1e-12)
        e0 = energy_fn(rb, shape, species, opt_params)
        # print(f"e0: {e0}")
        boltzmann_weight = jnp.exp(-e0 / opt_params[4])
        z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib, sigma)
        z = jnp.maximum(z, 1e-12)


        log_z = safe_log(z)

        return log_z

    log_z_all = []

    for size in tqdm(range(2, 12 + 1)):
        log_z = compute_log_z(size)
        log_z_all.append(log_z)
        print(f"size {size}: {log_z}")

    log_z_all = jnp.array(log_z_all)
    # print(f"log_z_all shape: {log_z_all.shape}")
    log_z_all = jnp.concatenate([log_z_1, log_z_all], axis=0)

    return log_z_all


Z_test = get_log_z_all(init_params)
print(Z_test)
'''
nper_structure = jnp.arange(1, 13)
init_conc = jnp.array([0.001])

def loss_fn(log_concs_struc, log_z_list, opt_params):
    m_conc = opt_params[-n:]
    tot_conc = init_conc
    log_mon_conc = safe_log(m_conc)

    def mon_loss_fn(mon_idx):
        mon_val = safe_log(jnp.dot(nper_structure[mon_idx], jnp.exp(log_concs_struc)))
        return jnp.sqrt((mon_val - log_mon_conc[mon_idx])**2)

    def struc_loss_fn(struc_idx):
        log_vcs = jnp.log(V) + log_concs_struc[struc_idx]

        def get_vcs_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_vca = jnp.log(V) + log_concs_struc[mon_idx]
            return n_sa * log_vca

        vcs_denom = vmap(get_vcs_denom)(jnp.arange(num_monomers)).sum()
        log_zs = log_z_list[struc_idx]

        def get_z_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_zalpha = log_z_list[mon_idx]
            return n_sa * log_zalpha

        z_denom = vmap(get_z_denom)(jnp.arange(num_monomers)).sum()

        return jnp.sqrt((log_vcs - vcs_denom - log_zs + z_denom)**2)

    mon_loss = vmap(mon_loss_fn)(jnp.arange(num_monomers))
    struc_loss = vmap(struc_loss_fn)(jnp.arange(num_monomers, tot_num_structures))
    combined_loss = jnp.concatenate([mon_loss, struc_loss])
    loss_var = jnp.var(combined_loss)
    loss_max = jnp.max(combined_loss)

    tot_loss = jnp.linalg.norm(combined_loss) + loss_var
    return tot_loss, combined_loss, loss_var


coo_shell_full = get_icos_rb(ico_radius=1.0)
pos2, species2 = get_icos_shape_and_species(coo_shell_full, vertex_radius, 2)
sigma2 = get_sigma(pos2)
adj_matrix = are_blocks_connected_rb(coo_shell_full, vertex_radius=1.0)
q2 = generate_connected_subsets_rb(coo_shell_full, adj_matrix)[-2]


SEED = 42
main_key = random.PRNGKey(SEED)

q2 = q2.flatten()
print(q2.shape)
testing = compute_zrot_mod_sigma(get_nmer_energy_fn(2), q2, pos2, species2, params, main_key, nrandom=100000)
print(testing)
'''
