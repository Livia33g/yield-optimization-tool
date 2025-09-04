#!/usr/bin/env python
import pdb
import os

os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/foo"
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
    device_get,
    block_until_ready,
)
import sys

sys.stdout.flush()
import optax
from jaxopt import implicit_diff, GradientDescent, LBFGS
from checkpoint import checkpoint_scan
import functools
import jax.numpy as jnp
from jax_md import energy, space, simulate
from jax_md.energy import morse, soft_sphere
from jax_md import rigid_body as orig_rigid_body
import potentials
import jax_transformations3d as jts
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

# --- Set JAX configuration; disable JIT to prevent verbose tracer printing ---
##config.update("jax_disable_jit", True)  # Disabling jit to avoid jax tracing output in prints.
# config.update("jax_log_compiles", False)

import itertools
import numpy as np
from scipy.spatial import distance_matrix
from jax import debug
import argparse
from jax.scipy.special import logsumexp

parser = argparse.ArgumentParser(
    description="Run the rigid-body simulation with adjustable parameters."
)
# Using new defaults so that typical free energy differences are more moderate.
parser.add_argument("--kt", type=float, default=5.0, help="Initial temperature (kbT)")
parser.add_argument(
    "--init_morse", type=float, default=12.0, help="Initial Morse epsilon"
)
parser.add_argument(
    "--rep_alpha", type=float, default=1.0, help="Initial repulsion alpha"
)
parser.add_argument(
    "--init_conc", type=float, default=0.001, help="Initial concentration"
)
parser.add_argument("--number_mon", type=int, default=343, help="Number of monomers")
parser.add_argument("--desired_yield", type=float, default=1.0, help="Desired yield")
args = parser.parse_args()

euler_scheme = "sxyz"
V = args.number_mon * args.init_conc

SEED = 42
main_key = random.PRNGKey(SEED)
# Parameter order: [morse_eps, morse_alpha, rep_A, rep_alpha, kbT]
init_params = jnp.array([args.init_morse, 2.0, 10000.0, args.rep_alpha, args.kt])
displacement_fn, shift_fn = space.free()


# --- Utility: Quaternion to Euler conversion ---
def quat_to_euler(quaternions):
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


# --- Geometry Loading Functions ---
oct_dir = Path("octahedron")
file_path_quaternion = oct_dir / "rb_orientation_vec.npy"
file_path_xyz_coordinate = oct_dir / "rb_center.npy"
file_path_shape = oct_dir / "vertex_shape_points.npy"
file_path_species = oct_dir / "vertex_shape_point_species.npy"
file_paths = [file_path_quaternion, file_path_xyz_coordinate, file_path_shape]


def load_rb_orientation_vec():
    rb_orientation_vec = jnp.load(file_path_quaternion).astype(jnp.float64)
    rb_center_vec = jnp.load(file_path_xyz_coordinate).astype(jnp.float64)
    rb_shape_vec = jnp.load(file_path_shape).astype(jnp.float64)
    rb_species_vec = jnp.load(file_path_species).astype(jnp.int32)
    euler_orientations = quat_to_euler(np.array(rb_orientation_vec))
    full_shell = jnp.hstack([rb_center_vec, euler_orientations])
    return full_shell, rb_shape_vec, rb_species_vec


def get_icos_shape_and_species(size):
    base_shape = load_rb_orientation_vec()[1]
    base_species = jnp.array([0, 1, 2, 1, 2])
    return jnp.array([base_shape for _ in range(size)]), jnp.array(
        [base_species for _ in range(size)]
    )


def load_rb_orientation_vec():
    rb_orientation_vec = jnp.load(file_path_quaternion).astype(jnp.float64)
    rb_center_vec = jnp.load(file_path_xyz_coordinate).astype(jnp.float64)
    rb_shape_vec = jnp.load(file_path_shape).astype(jnp.float64)
    rb_species_vec = jnp.load(file_path_species).astype(jnp.int32)
    euler_orientations = quat_to_euler(np.array(rb_orientation_vec))
    full_shell = jnp.hstack([rb_center_vec, euler_orientations])
    return full_shell, rb_shape_vec, rb_species_vec


def get_octa_shape_and_species(size):
    _, base_shape, _ = load_rb_orientation_vec()
    base_species = jnp.array([0, 1, 2, 1, 2])
    return jnp.array([base_shape for _ in range(size)]), jnp.array(
        [base_species for _ in range(size)]
    )


# --- Connectivity Utilities ---
def are_blocks_connected_rb(vertex_coords, vertex_radius=2.0, tolerance=0.2):
    positions = vertex_coords[:, :3]
    distances = distance_matrix(positions, positions)
    edge_length = (2.0 + tolerance) * vertex_radius
    adjacency_matrix = (distances <= edge_length).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix


def is_configuration_connected_rb(indices, adj_matrix):
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
    num_vertices = len(vertex_coords)
    configs = [np.array(vertex_coords)]
    current_indices = list(range(num_vertices))
    current_adj_matrix = adj_matrix.copy()
    while len(current_indices) > 1:
        for i in range(len(current_indices)):
            test_indices = current_indices[:i] + current_indices[i + 1 :]
            if is_configuration_connected_rb(test_indices, current_adj_matrix):
                current_indices = test_indices
                current_config = vertex_coords[jnp.array(current_indices)]
                configs.append(current_config)
                current_adj_matrix = current_adj_matrix[
                    np.ix_(current_indices, current_indices)
                ]
                break
        else:
            break
    connected_subsets = configs[::-1]
    one_mer = [vertex_coords[0:1]]
    two_mer = []
    min_dist = float("inf")
    for i in range(vertex_coords.shape[0]):
        for j in range(i + 1, vertex_coords.shape[0]):
            dist = np.linalg.norm(vertex_coords[i, :3] - vertex_coords[j, :3])
            if dist < 2.2 * 2.0 and dist < min_dist:
                two_mer = [np.vstack([vertex_coords[i], vertex_coords[j]])]
                min_dist = dist
    all_config = one_mer + two_mer + connected_subsets
    return all_config


vertex_species = 0
n_species = 3
vertex_radius = 2.1
small_value = 1e-12
rep_rmax_table = jnp.full((n_species, n_species), 2 * vertex_radius)


# --- Potential Functions ---
def make_tables(opt_params):
    morse_eps = jnp.zeros((n_species, n_species))
    morse_alpha = jnp.zeros((n_species, n_species))
    soft_eps = jnp.zeros((n_species, n_species))
    soft_sigma = jnp.ones((n_species, n_species)) * 1e-5
    morse_eps = morse_eps.at[1, 1].set(opt_params[0])
    morse_alpha = morse_alpha.at[1, 1].set(opt_params[1])
    morse_eps = morse_eps.at[2, 2].set(opt_params[0])
    morse_alpha = morse_alpha.at[2, 2].set(opt_params[1])
    soft_eps = soft_eps.at[0, 0].set(opt_params[2])
    soft_sigma = soft_sigma.at[0, 0].set(opt_params[-2])
    return morse_eps, morse_alpha, soft_eps, soft_sigma


def pairwise_morse(ipos, jpos, i_species, j_species, opt_params):
    morse_eps, morse_alpha, _, _ = make_tables(opt_params)
    eps = morse_eps[i_species, j_species]
    alpha = morse_alpha[i_species, j_species]
    sigma = 1e-12
    r_onset = 10.0
    r_cutoff = 12.0
    dr = space.distance(ipos - jpos)
    return morse(
        dr, epsilon=eps, alpha=alpha, r_onset=r_onset, r_cutoff=r_cutoff, sigma=sigma
    )


morse_func = vmap(
    vmap(pairwise_morse, in_axes=(None, 0, None, 0, None)),
    in_axes=(0, None, 0, None, None),
)


def pairwise_repulsion(ipos, jpos, i_species, j_species, opt_params):
    _, _, soft_eps, soft_sigma = make_tables(opt_params)
    eps = soft_eps[i_species, j_species]
    sigma = soft_sigma[i_species, j_species]
    dr = space.distance(ipos - jpos)
    return soft_sphere(dr, sigma=sigma, epsilon=eps)


inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0, None))
rep_func = vmap(inner_rep, in_axes=(0, None, 0, None, None))

# --- Energy Function Generator ---


def get_nmer_energy_fn(n):
    pos_slices = [(i * 5, (i + 1) * 5) for i in range(n)]
    species_slices = [(i * 5, (i + 1) * 5) for i in range(n)]
    pairs = jnp.array(list(itertools.combinations(np.arange(n), 2)))

    def nmer_energy_fn(q, pos, species, opt_params):
        positions = utils.get_positions(q, pos)
        all_pos = jnp.stack([positions[start:end] for start, end in pos_slices])
        species_flat = jnp.concatenate([jnp.array(s).reshape(-1) for s in species])
        all_species = jnp.stack(
            [species_flat[start:end] for start, end in species_slices]
        )

        def pairwise_energy(pair):
            i, j = pair
            morse_energy = morse_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j], opt_params
            ).sum()
            rep_energy = rep_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j], opt_params
            ).sum()
            return morse_energy + rep_energy

        total_energy = vmap(pairwise_energy)(pairs).sum()
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
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            shell, sigma_str = line.strip().split(",")
            size_str = shell.split("_")[-1]
            size = int(size_str.replace("size", "").replace(".pos", ""))
            sigmas[size] = float(sigma_str)
    return sigmas


def adjust_sigmas(sigmas):
    adjusted_sigmas = {}
    for size, sigma in sigmas.items():
        if size > 1:
            adjusted_sigmas[size] = sigma  # * (5 ** size)
        else:
            adjusted_sigmas[size] = 5.0
    return adjusted_sigmas


sigmas_ext = load_sigmas("symmetry_numbers_oct.txt")
sigmas = adjust_sigmas(sigmas_ext)

# --- Prepare Rigid-Body and Shape Data ---
full_shell = load_rb_orientation_vec()[0]
adj_ma = are_blocks_connected_rb(full_shell, vertex_radius=2.1, tolerance=0.2)
rbs = generate_connected_subsets_rb(full_shell, adj_ma)
rbs = [rb.flatten() for rb in rbs]
shapes_species = [get_icos_shape_and_species(size) for size in range(1, 7)]
shapes, species = zip(*shapes_species)

# --- Precompute Energy Functions ---
energy_fns = {size: jit(get_nmer_energy_fn(size)) for size in range(2, 6 + 1)}
mon_energy_fn = lambda q, pos, species, opt_params: 0.0

# --- Compute Rotational Contributions ---
zrot_mod_sigma_1, _, main_key = compute_zrot_mod_sigma(
    mon_energy_fn, rbs[0], shapes[0], species[0], init_params, main_key
)
zvib_1 = 1.0
boltzmann_weight = 1.0

z_1 = jnp.array([compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1, sigmas[1])])
log_z_1 = jnp.log(z_1)

zrot_mod_sigma_values = []

for size in range(2, 6 + 1):
    zrot_mod_sigma, Js, main_key = compute_zrot_mod_sigma(
        energy_fns[size],
        rbs[size - 1],
        shapes[size - 1],
        species[size - 1],
        init_params,
        main_key,
    )
    zrot_mod_sigma_values.append(zrot_mod_sigma)


def get_log_z_all(opt_params):
    def compute_log_z(size):
        energy_fn = energy_fns[size]
        shape = shapes[size - 1]
        rb = rbs[size - 1]
        specie = species[size - 1]
        sigma = sigmas[size]
        zrot_mod_sigma = zrot_mod_sigma_values[size - 2]
        zvib = compute_zvib(energy_fn, rb, shape, specie, opt_params)
        e0 = energy_fn(rb, shape, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / opt_params[4])
        z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib, sigma)
        log_z = jnp.log(z)

        return log_z

    log_z_struc = []

    for size in range(2, 6 + 1):
        log_z = compute_log_z(size)
        log_z_struc.append(log_z)

    log_z_struc = jnp.array(log_z_struc)

    log_z_all = jnp.concatenate([log_z_1, log_z_struc], axis=0)

    return log_z_all


log_z_list = get_log_z_all(init_params)

# --- Outer Optimization Setup ---
nper_structure = jnp.arange(1, 7)
init_conc_val = args.init_conc
init_conc = jnp.array([init_conc_val])


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
        return jnp.sqrt((log_vcs - vcs_denom - log_zs + z_denom) ** 2)

    struc_losses = vmap(struc_loss_fn)(jnp.arange(2, 6))  # sizes 2–6
    combined = jnp.concatenate([jnp.array([mon_loss]), struc_losses])  # (6,)

    # --- apply a big weight to the monomer term only ---
    weights = jnp.array([10.0] + [1.0] * 4)
    weighted = combined * weights

    # --- variance penalty on the *unweighted* combined losses ---
    loss_var = jnp.var(combined)

    # --- total objective is norm + weighted variance penalty ---
    tot_loss = jnp.linalg.norm(weighted) + 50 * loss_var

    return tot_loss, combined, loss_var


def optimality_fn(log_concs_struc, log_z_list):
    return grad(lambda x, z: loss_fn(x, z)[0])(log_concs_struc, log_z_list)


def inner_solver(init_guess, log_z_list):
    lbfgs = LBFGS(
        fun=lambda x, z: loss_fn(x, z)[0],
        maxiter=600,  # you can raise/lower this
        tol=1e-6,  # stop when ‖∇loss‖<1e-6
        jit=True,
    )
    sol = lbfgs.run(init_guess, log_z_list)
    return sol.params


def safe_exp(x, lower_bound=-709.0, upper_bound=709.0):

    clipped_x = jnp.clip(x, a_min=lower_bound, a_max=upper_bound)

    return jnp.exp(clipped_x)


def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))


def ofer(opt_params):
    log_z_list = get_log_z_all(opt_params)
    tot_conc = init_conc_val
    struc_concs_guess = jnp.full(6, safe_log(init_conc_val / 6))
    # struc_concs_guess = struc_concs_guess.at[-1].set(jnp.log(0.2))
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list)
    total_log = logsumexp(fin_log_concs)
    log_yield = fin_log_concs[-1] - total_log
    return log_yield


def ofer_grad_fn(opt_params, desired_yield_val):
    log_yield = ofer(opt_params)
    target = jnp.log(desired_yield_val)
    energy_penalty = sum(
        [
            energy_fns[size](
                rbs[size - 1], shapes[size - 1], species[size - 1], opt_params
            )
            for size in range(2, 7)
        ]
    )
    return (
        (desired_yield_val - jnp.exp(log_yield)) ** 2
        + 0.1 * (log_yield - target) ** 2
        + 0.05 * energy_penalty
    )


num_params = len(init_params)
mask = jnp.zeros(num_params)

mask = mask.at[0].set(1.0)
mask = mask.at[-2].set(1.0)
# mask = mask.at[0].set(1.0)


def masked_grads(grads):
    return grads * mask


def enforce_param_bounds(params):
    # Ensure morse_eps >= 0.5
    return params.at[0].set(jnp.maximum(params[0], 0.5))


our_grad_fn = jit(value_and_grad(lambda p, y: (y - jnp.exp(ofer(p))) ** 2))
params = init_params
outer_optimizer = optax.adam(5e-2)
opt_state = outer_optimizer.init(params)

n_outer_iters = 200
outer_losses = []


param_names = [f"morse_eps"]
param_names += [f"morse_alpha"]
param_names += [f"rep_A"]
param_names += [f"rep_alpha"]
param_names += [f"kbT"]


desired_yield_val = args.desired_yield


@jit
def outer_step(params, opt_state, desired_yield):
    loss, grads = our_grad_fn(params, desired_yield)
    grads = masked_grads(grads)
    updates, opt_state = outer_optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    new_params = enforce_param_bounds(new_params)
    return new_params, opt_state, loss


os.makedirs("Paper/optimized", exist_ok=True)
kt_val = args.kt
params = init_params
opt_state = outer_optimizer.init(params)
with open(f"Paper/optimized/{kt_val}.txt", "w") as f:
    for i in range(n_outer_iters):
        params, opt_state, loss = outer_step(params, opt_state, desired_yield_val)
        print(f"iter {i:3d}   loss={loss:.6f}")
        print("Updated Parameters:")
        for name, value in {
            name: params[idx] for idx, name in enumerate(param_names)
        }.items():
            print(f"{name}: {value}")
        print(params)
        fin_yield = ofer(params)
        fin_yield = jnp.exp(fin_yield)
        print(f"Desired Yield: {desired_yield_val}, Yield: {fin_yield}")
        final_params = params
        fin_yield = ofer(params)
        final_target_yields = jnp.exp(fin_yield)

    f.write(
        f"{desired_yield_val},{final_target_yields},{params[0]},{params[-2]},{params[-1]}\n"
    )
    # f.write(f"{des_yield}, {final_target_yields}, {params[0]}, {params[-1]}\n")
    f.flush()
