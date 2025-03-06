import argparse
import numpy as onp
import pickle
import time
import jax.numpy as jnp
import optax
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
    linearize,
)
from tqdm import tqdm
from jax_md import space
import potentials
import utils
from jax_transformations3d import jax_transformations3d as jts
from jaxopt import implicit_diff, GradientDescent
from checkpoint import checkpoint_scan
import pdb
import functools
import itertools
from itertools import permutations
from functools import wraps
import matplotlib.pyplot as plt
from jax.config import config
from jax.flatten_util import ravel_pytree
import gc
import os
import networkx as nx
from itertools import product

# Set up argument parsing
parser = argparse.ArgumentParser(description="Simulation with adjustable parameters.")
parser.add_argument(
    "--eps_weak",
    type=float,
    default=7.0,
    help="weak epsilon values (attraction strengths).",
)
parser.add_argument(
    "--eps_init",
    type=float,
    default=7.0,
    help="init strong epsilon values (attraction strengths).",
)
parser.add_argument(
    "--kt", type=float, default=1.0, help="Thermal energy (kT). Default is 1.0."
)
parser.add_argument(
    "--init_conc",
    type=float,
    default=0.007,
    help="Initial concentration. Default is 0.001.",
)
parser.add_argument(
    "--desired_yield", type=float, default=0.4, help="desired yield of target."
)
parser.add_argument(
    "--sim_particles",
    type=int,
    default=300,
    help="total number of particles wanted in the simulation.",
)
parser.add_argument("--mon_type", type=int, default=3, help="number of monomer types.")

args = parser.parse_args()

V = args.sim_particles / args.init_conc
# Define constants
a = 1.0  # Radius placeholder
b = 0.3
separation = 2.0
noise = 1e-14

use_custom_pairs = True

custom_pairs = [(4, 5), (2, 3), (1, 6), (1, 2), (1, 4), (2, 5), (3, 4), (3, 6), (5, 6)]


def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))


def poly_chain_decorator(func):
    @functools.wraps(func)
    def wrapper(size, *args, poly_chain=False, repeats=1, manual_chain=None, **kwargs):
        # If poly_chain is True and a manual_chain is provided,
        # use the manual chain repeated 'repeats' times.
        if poly_chain and manual_chain is not None:
            chain = jnp.array(manual_chain)
            return jnp.concatenate([chain] * repeats)
        else:
            # Otherwise, generate the default target sequence.
            target = func(size, *args, **kwargs)
            if poly_chain:
                target = jnp.concatenate([target] * repeats)
            return target

    return wrapper


@poly_chain_decorator
def produce_target(size):
    target = jnp.array([1, 0, 2])
    if size == 1:
        return target
    else:
        for i in range(2, size + 1):
            target = jnp.concatenate([target, jnp.array([i * 2 - 1, 0, i * 2])])
    return target


target = produce_target(3, poly_chain=True, repeats=2)


def produce_custom_pair(target):
    """
    Given a target 1D array, produce a list of pairs where two nonzero numbers
    occur consecutively in the target. Each pair is added only once.

    Parameters:
        target (jnp.ndarray): 1D array containing integer elements.

    Returns:
        List[Tuple[int, int]]: List of unique pairs of adjacent nonzero numbers.
    """
    pairs = []
    for i in range(len(target) - 1):
        if target[i] != 0 and target[i + 1] != 0:
            pair = (int(target[i]), int(target[i + 1]))
            if pair not in pairs:
                pairs.append(pair)
    return pairs


# custom_pairs = produce_custom_pair(target)


def load_species_combinations(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


data = load_species_combinations("arvind_36.pkl")

data_tetr = load_species_combinations("extracted_7.pkl")

num_monomers = max(
    int(k.split("_")[0]) for k in data.keys() if k.endswith("_pc_species")
)


species_data = {}
tot_num_structures = 0

for i in range(1, num_monomers + 1):
    key = f"{i}_pc_species"
    species_data[key] = data[key]
    tot_num_structures += species_data[key].shape[0]


# Function to find the index of a target structure
def indx_of_target(target, species_data):
    target_reversed = target[::-1]
    num_monomers = len(species_data)

    offset = 0
    for i in range(1, num_monomers + 1):
        key = f"{i}_pc_species"
        current_species = species_data[key]
        for j in range(current_species.shape[0]):
            if jnp.array_equal(current_species[j], target) or jnp.array_equal(
                current_species[j], target_reversed
            ):
                return j + offset
        offset += current_species.shape[0]

    return None


target_idx = indx_of_target(target, species_data)

euler_scheme = "sxyz"

SEED = 42
main_key = random.PRNGKey(SEED)
kT = jnp.array([args.kt])
n = args.mon_type

# Shape and energy helper functions
a = 1.0  # distance of the center of the spheres from the BB COM
b = 0.3  # distance of the center of the patches from the BB COM
separation = 2.0
noise = 1e-14
vertex_radius = a
patch_radius = 0.2 * a
small_value = 1e-12
vertex_species = 0


n_patches = n * 2  # 2 species of patches per monomer type
n_species = n_patches + 1  # plus the common vertex species 0

custom_pairs_n = len(custom_pairs)
patchy_vals_intermediate = jnp.array([args.eps_init])
patchy_vals_weak = jnp.full(
    custom_pairs_n - 3, args.eps_weak
)  # FIXME for optimization over specific attraction strengths
patchy_vals_strong = jnp.full(2, args.eps_init)

init_conc = args.init_conc
m_conc = init_conc / n
init_concs = jnp.full(n, m_conc)
# init_params = jnp.concatenate([patchy_vals, init_concs])
# init_params = patchy_vals

patchy_vals = jnp.concatenate(
    [patchy_vals_strong, patchy_vals_intermediate, patchy_vals_weak]
)
init_params = jnp.concatenate([patchy_vals, kT, init_concs])


def make_shape(size):
    base_shape = jnp.array(
        [
            [-a, 0.0, b],  # first patch
            [-a, b * jnp.cos(jnp.pi / 6.0), -b * jnp.sin(jnp.pi / 6.0)],  # second patch
            [-a, -b * jnp.cos(jnp.pi / 6.0), -b * jnp.sin(jnp.pi / 6.0)],
            [0.0, 0.0, a],
            [
                0.0,
                a * jnp.cos(jnp.pi / 6.0),
                -a * jnp.sin(jnp.pi / 6.0),
            ],  # second sphere
            [
                0.0,
                -a * jnp.cos(jnp.pi / 6.0),
                -a * jnp.sin(jnp.pi / 6.0),
            ],  # third sphere
            [a, 0.0, b],  # first patch
            [a, b * jnp.cos(jnp.pi / 6.0), -b * jnp.sin(jnp.pi / 6.0)],  # second patch
            [a, -b * jnp.cos(jnp.pi / 6.0), -b * jnp.sin(jnp.pi / 6.0)],  # third patch
        ],
        dtype=jnp.float64,
    )
    return jnp.array([base_shape for _ in range(size)])


def make_rb(size, key, separation=2.0, noise=1e-14):
    if size == 1:
        return jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64), key  # Return key

    key, subkey = random.split(key)  # Split the main_key properly
    rand_vals = random.normal(subkey, shape=(size,))

    rb = []
    half_size = size // 2

    for i in range(size):
        if size % 2 == 0:
            if i < half_size:
                rb.extend(
                    [
                        -separation / 2.0 * (size - 1 - 2 * i),
                        rand_vals[i] * noise,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
            else:
                rb.extend(
                    [
                        separation / 2.0 * (size - 1 - 2 * (size - 1 - i)),
                        rand_vals[i] * noise,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
        else:
            if i == half_size:
                rb.extend([0, 0, 0, 0, 0, 0])
            elif i < half_size:
                rb.extend(
                    [-separation * (half_size - i), rand_vals[i] * noise, 0, 0, 0, 0]
                )
            else:
                rb.extend(
                    [separation * (i - half_size), rand_vals[i] * noise, 0, 0, 0, 0]
                )

    return jnp.array(rb, dtype=jnp.float64), key


rep_rmax_table = jnp.full((n_species, n_species), 2 * vertex_radius)
rep_A_table = (
    jnp.full((n_species, n_species), small_value)
    .at[vertex_species, vertex_species]
    .set(500.0)
)
rep_alpha_table = jnp.full((n_species, n_species), 2.5)

morse_narrow_alpha = 5.0
morse_alpha_table = jnp.full((n_species, n_species), morse_narrow_alpha)


def generate_idx_pairs(n_species):
    idx_pairs = []
    for i in range(1, n_species):
        for j in range(i + 1, n_species):
            idx_pairs.append((i, j))
    return idx_pairs


generated_idx_pairs = generate_idx_pairs(n_species)


def make_tables(opt_params, use_custom_pairs=True, custom_pairs=custom_pairs):
    # morse_eps_table = jnp.full((n_species, n_species), args.eps_weak)
    morse_eps_table = jnp.full((n_species, n_species), 1e-12)
    morse_eps_table = morse_eps_table.at[0, :].set(small_value)
    morse_eps_table = morse_eps_table.at[:, 0].set(small_value)

    if use_custom_pairs and custom_pairs is not None:
        idx_pairs = custom_pairs
    else:
        idx_pairs = generated_idx_pairs

    # Set off-diagonal elements
    for i, (idx1, idx2) in enumerate(idx_pairs):
        morse_eps_table = morse_eps_table.at[idx1, idx2].set(opt_params[i])
        morse_eps_table = morse_eps_table.at[idx2, idx1].set(opt_params[i])

    # Set diagonal elements excluding (0,0)
    if not use_custom_pairs:
        diagonal_start_idx = len(idx_pairs)
        for i in range(1, n_species):
            morse_eps_table = morse_eps_table.at[i, i].set(
                opt_params[diagonal_start_idx + i - 1]
            )

    return morse_eps_table


def pairwise_morse(ipos, jpos, i_species, j_species, opt_params):
    morse_eps_table = make_tables(opt_params)
    morse_d0 = morse_eps_table[i_species, j_species]
    morse_alpha = morse_alpha_table[i_species, j_species]
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


def pairwise_repulsion(ipos, jpos, i_species, j_species):
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)
    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)


inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))


def get_nmer_energy_fn(n):
    pairs = jnp.array(list(itertools.combinations(onp.arange(n), 2)))

    def nmer_energy_fn(q, pos, species, opt_params):
        positions = utils.get_positions(q, pos)
        pos_slices = [(i * 9, (i + 1) * 9) for i in range(n)]
        species_slices = [(i * 3, (i + 1) * 3) for i in range(n)]

        all_pos = jnp.stack([positions[start:end] for start, end in pos_slices])
        all_species = jnp.stack(
            [jnp.repeat(species[start:end], 3) for start, end in species_slices]
        )

        def pairwise_energy(pair):
            i, j = pair
            morse_energy = morse_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j], opt_params
            ).sum()
            rep_energy = rep_func(
                all_pos[i], all_pos[j], all_species[i], all_species[j]
            ).sum()
            return morse_energy + rep_energy

        all_pairwise_energies = vmap(pairwise_energy)(pairs)
        return all_pairwise_energies.sum()

    return nmer_energy_fn


def hess(energy_fn, q, pos, species, opt_params):
    # Only differentiate with respect to q (the rigid-body degrees of freedom).
    # The other inputs are treated as constants.
    flat_q, unravel_q = ravel_pytree(q)

    def energy_flat(x):
        # Reconstruct q from the flattened vector.
        new_q = unravel_q(x)
        # Call energy_fn with new_q while keeping pos, species, and opt_params fixed.
        return energy_fn(new_q, pos, species, opt_params)

    # Get the gradient function with respect to the flattened q.
    grad_energy = grad(energy_flat)
    # Use linearize to get a function that computes Hessian-vector products.
    _, hvp = linearize(grad_energy, flat_q)

    # Build the full Hessian by applying hvp to each basis vector.
    basis = jnp.eye(flat_q.shape[0])
    H_cols = [hvp(b) for b in basis]
    H = jnp.stack(H_cols).reshape(flat_q.shape[0], flat_q.shape[0])

    # Compute the eigenvalues and eigenvectors of the Hessian.
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def compute_zvib(energy_fn, q, pos, species, opt_params):
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(
        jnp.sqrt(2.0 * jnp.pi / (opt_params[-4] * jnp.abs(evals[6:]) + 1e-12))
    )
    return zvib


def compute_zrot_mod_sigma(energy_fn, q, pos, species, opt_params, key, nrandom=100000):
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


sizes = range(1, num_monomers + 1)

rbs = {}

for size in sizes:
    main_key, subkey = random.split(main_key)
    rb, subkey = make_rb(size, subkey)
    rbs[size] = rb
    main_key = subkey

shapes = {size: make_shape(size) for size in sizes}
sigmas = {size: data[f"{size}_sigma"] for size in sizes if f"{size}_sigma" in data}
energy_fns = {
    size: jit(get_nmer_energy_fn(size)) for size in range(2, num_monomers + 1)
}

e_plus_1_fn = get_nmer_energy_fn(7)


main_key, subkey = random.split(main_key)
rb_plus_1, subkey = make_rb(7, subkey)
main_key = subkey
main_key, subkey = random.split(main_key)


shape_plus_1 = make_shape(7)


rb1 = rbs[1]
shape1 = shapes[1]
# sigma1 = data["1_sigma"]
mon_energy_fn = lambda q, pos, species, opt_params: 0.0


zrot_mod_sigma_1, _, main_key = compute_zrot_mod_sigma(
    mon_energy_fn, rb1, shape1, jnp.array([1, 0, 2]), patchy_vals, main_key
)
zvib_1 = 1.0
boltzmann_weight = 1.0

z_1 = compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1, sigmas[1])
z_1s = jnp.full(n, z_1)
log_z_1 = jnp.log(z_1s)

zrot_mod_sigma_values = {}

for size in range(2, 7):

    zrot_mod_sigma, Js, main_key = compute_zrot_mod_sigma(
        energy_fns[size],
        rbs[size],
        shapes[size],
        jnp.array([1, 0, 2] * size),
        patchy_vals,
        main_key,
    )

    zrot_mod_sigma_values[size] = zrot_mod_sigma


def get_log_z_all(opt_params):
    def compute_log_z(size, species, sigma):
        energy_fn = energy_fns[size]
        shape = shapes[size]
        rb = rbs[size]
        zrot_mod_sigma = zrot_mod_sigma_values[size]
        zvib = compute_zvib(energy_fn, rb, shape, species, opt_params)
        e0 = energy_fn(rb, shape, species, opt_params)
        boltzmann_weight = jnp.exp(-e0 / opt_params[-4])
        z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib, sigma)
        return safe_log(z)

    log_z_all = []

    for size in range(2, num_monomers + 1):
        species = data[f"{size}_pc_species"]
        sigma = data[f"{size}_sigma"]

        # Repeat sigma for each structure in species of the current size
        sigma_array = jnp.full(species.shape[0], sigma)

        if size <= 4:
            log_z = vmap(lambda sp, sg: compute_log_z(size, sp, sg))(
                species, sigma_array
            )
        else:
            compute_log_z_ckpt = checkpoint(lambda sp, sg: compute_log_z(size, sp, sg))
            flat_species = species.reshape(species.shape[0], -1)
            xs = jnp.concatenate([flat_species, sigma_array[:, None]], axis=-1)

            def scan_fn(carry, x):
                flat_species, sigma_val = x[:-1], x[-1]
                species_new = flat_species.reshape(species.shape[1:])
                result = compute_log_z_ckpt(species_new, sigma_val)
                return carry, result

            checkpoint_freq = 10
            scan_with_ckpt = functools.partial(
                checkpoint_scan, checkpoint_every=checkpoint_freq
            )
            _, log_z = scan_with_ckpt(scan_fn, None, xs)
            log_z = jnp.array(log_z)

        log_z_all.append(log_z)

    log_z_all = jnp.concatenate(log_z_all, axis=0)
    print(log_z_all.shape)
    log_z_all = jnp.concatenate([log_z_1, log_z_all], axis=0)

    return log_z_all


# Example monomer counts
monomer_counts = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    counts_list = []
    for i in range(1, 7):
        key = f"{letter}_{i}_counts"
        if key in data:
            counts_list.append(data[key])
    if counts_list:
        monomer_counts.append(jnp.concatenate(counts_list))

nper_structure = jnp.array(monomer_counts)
species_tetr = data_tetr[f"7_pc_species"]

tetramer_counts = {
    key.split("_")[0]: data_tetr[key]
    for key in data_tetr.keys()
    if key.endswith("_7_counts")
}

tetramer_counts_array = jnp.array(list(tetramer_counts.values()))
# tetramer_counts_array = jnp.array([tetramer_counts[key] for key in sorted(tetramer_counts.keys())])
tetramer_counts_array = jnp.array(list(tetramer_counts.values()))
# tetramer_counts_array = jnp.array([tetramer_counts[key] for key in sorted(tetramer_counts.keys())])


def loss_fn(log_concs_struc, log_z_list, opt_params):
    # m_conc = init_concs  # opt_params[-n:] init_concs
    m_conc = opt_params[-n:]
    log_mon_conc = safe_log(m_conc)

    def mon_loss_fn(mon_idx):
        mon_val = safe_log(jnp.dot(nper_structure[mon_idx], jnp.exp(log_concs_struc)))
        return jnp.sqrt((mon_val - log_mon_conc[mon_idx]) ** 2)

    def struc_loss_fn(struc_idx):
        log_vcs = jnp.log(V) + log_concs_struc[struc_idx]

        def get_vcs_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_vca = jnp.log(V) + log_concs_struc[mon_idx]
            return n_sa * log_vca

        vcs_denom = vmap(get_vcs_denom)(jnp.arange(n)).sum()
        log_zs = log_z_list[struc_idx]

        def get_z_denom(mon_idx):
            n_sa = nper_structure[mon_idx][struc_idx]
            log_zalpha = log_z_list[mon_idx]
            return n_sa * log_zalpha

        z_denom = vmap(get_z_denom)(jnp.arange(n)).sum()

        return jnp.sqrt((log_vcs - vcs_denom - log_zs + z_denom) ** 2)

    final_mon_conc = jnp.exp(log_concs_struc[:n]).sum()

    def log_massact_loss_fn(opt_params, struc_idx):

        def mon_sum(mon_idx):

            conc_mon_cont = (
                tetramer_counts_array[mon_idx, struc_idx] * log_concs_struc[mon_idx]
            )

            return conc_mon_cont

        e_plus_1 = e_plus_1_fn(
            rb_plus_1, shape_plus_1, species_tetr[struc_idx], opt_params
        )

        presum_mon = vmap(mon_sum)(jnp.arange(n))

        mass_act_loss = (
            -1 / opt_params[custom_pairs_n] * e_plus_1
            + jnp.sum(presum_mon)
            + jnp.log(n + 1)
        )

        return mass_act_loss

    mon_loss = vmap(mon_loss_fn)(jnp.arange(n))
    struc_loss = vmap(struc_loss_fn)(jnp.arange(n, tot_num_structures))
    """
    mass_act_loss_fun = vmap(log_massact_loss_fn, in_axes=(None, 0))
    mass_act_loss = jnp.sum(mass_act_loss_fun(opt_params, jnp.arange(species_tetr.shape[0])))

    mass_act_loss_log = jnp.array([ mass_act_loss])
    """
    loss_var = jnp.var(jnp.concatenate([mon_loss, struc_loss]))
    # combined_losses = jnp.concatenate([mon_loss, struc_loss, mass_act_loss_log])
    combined_losses = jnp.concatenate([mon_loss, struc_loss])
    combined_loss = jnp.linalg.norm(combined_losses)

    tot_loss = combined_loss + loss_var

    return tot_loss, combined_loss, loss_var


def optimality_fn(log_concs_struc, log_z_list, opt_params):
    return grad(
        lambda log_concs_struc, log_z_list, opt_params: loss_fn(
            log_concs_struc, log_z_list, opt_params
        )[0]
    )(log_concs_struc, log_z_list, opt_params)


@implicit_diff.custom_root(optimality_fn)
def inner_solver(init_guess, log_z_list, opt_params):
    gd = GradientDescent(
        fun=lambda log_concs_struc, log_z_list, opt_params: loss_fn(
            log_concs_struc, log_z_list, opt_params
        )[0],
        maxiter=50000,
        implicit_diff=True,
    )
    sol = gd.run(init_guess, log_z_list, opt_params)

    final_params = sol.params
    final_loss, combined_losses, loss_var = loss_fn(
        final_params, log_z_list, opt_params
    )

    return final_params


#########################

"""
def return_both_yields(func):
    @wraps(func)
    def wrapper(*args, return_both=False, **kwargs):
        target_yield, mon_yield_tot, tots = func(*args, **kwargs)
        if return_both:
            return target_yield, mon_yield_tot, tots
        return target_yield

    return wrapper


@return_both_yields
def ofer(opt_params):
    log_z_list = get_log_z_all(opt_params)
    tot_conc = opt_params[-n:].sum()
    struc_concs_guess = jnp.full(
        tot_num_structures, safe_log(tot_conc / tot_num_structures)
    )
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list, opt_params)
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / jnp.sum(fin_concs)
    target_yield = safe_log(yields[target_idx])
    mon_yield_tot = jnp.sum(yields[:n])
    # return target_yield, jnp.sum(fin_concs)  # mon_yield_tot
    # return target_yield, mon_yield_tot
    return target_yield, fin_concs[:n], fin_concs.sum()
"""


def safe_exp(x, lower_bound=-709.0, upper_bound=709.0):

    clipped_x = jnp.clip(x, a_min=lower_bound, a_max=upper_bound)

    return jnp.exp(clipped_x)


def ofer(opt_params):
    log_z_list = get_log_z_all(opt_params)
    tot_conc = init_conc
    struc_concs_guess = jnp.full(
        tot_num_structures, safe_log(tot_conc / tot_num_structures)
    )
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list, opt_params)
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / jnp.sum(fin_concs)
    target_yield = jnp.log(yields[target_idx])
    return target_yield, fin_concs[:n], fin_concs.sum()


def ofer_grad_fn(opt_params, desired_yield_val):
    target_yield, mon_concs, fin_conc = ofer(opt_params)

    def log_massact_loss_fn(opt_params, struc_idx):

        def mon_sum(mon_idx):

            conc_mon_cont = jnp.sum(
                tetramer_counts_array[mon_idx, struc_idx] * safe_log(mon_concs[mon_idx])
            )

            return conc_mon_cont

        e_plus_1 = e_plus_1_fn(
            rb_plus_1, shape_plus_1, species_tetr[struc_idx], opt_params
        )

        presum_mons = vmap(mon_sum)(jnp.arange(n))

        mass_act_loss = (
            -1 / opt_params[custom_pairs_n] * e_plus_1
            + jnp.sum(presum_mons)
            + jnp.log(n + 1)
        )

        return mass_act_loss

    mass_act_loss_logs = vmap(log_massact_loss_fn, in_axes=(None, 0))
    # mass_act_loss = jnp.sqrt((init_conc-fin_conc+jnp.sum(mass_act_loss_fun(opt_params, species_tetr)))**2)

    mass_act_loss = jnp.sum(
        mass_act_loss_logs(opt_params, jnp.arange(species_tetr.shape[0]))
    )
    # loss = 10000 * target_yield + mass_act_loss
    # loss = jnp.linalg.norm(desired_yield_val - jnp.exp(target_yield))
    # loss = 10000 * (abs(jnp.log(desired_yield_val) - target_yield)) ** 2 + mass_act_loss
    loss = -100 * target_yield + jnp.exp(mass_act_loss)
    # mass_act_loss = 0.0
    # loss = (abs(jnp.log(desired_yield_val)- target_yield))**2

    return loss, jnp.exp(mass_act_loss)


def abs_array(par):
    return jnp.abs(par)


def normalize(arr):
    sum_arr = jnp.sum(arr)
    new_arr = arr / sum_arr
    return new_arr


def normalize_logits(logits, total_concentration):
    norm_logits = normalize(logits)

    concentrations = norm_logits * total_concentration
    # Note: concentrations now sums to total_concentration
    return concentrations


num_params = len(init_params)

mask = jnp.full(num_params, 1.0)
mask = mask.at[-4].set(0.0)


def masked_grads(grads):
    return grads * mask


def project(params):
    conc_min = 1e-6
    concs = jnp.clip(params[-n:], a_min=conc_min)
    return jnp.concatenate([params[:-n], concs])


our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=True))
params = init_params
outer_optimizer = optax.adam(1e-2)
opt_state = outer_optimizer.init(params)

n_outer_iters = 300
outer_losses = []


if use_custom_pairs and custom_pairs is not None:
    param_names = [f"Eps({i},{j})" for i, j in custom_pairs]
else:
    param_names = [f"Eps({i},{j})" for i, j in generated_idx_pairs]
    param_names += [f"Eps({i},{i})" for i in range(1, n_patches + 1)]

param_names += ["kT"]

param_names += [f"conc_{chr(ord('A') + i)}" for i in range(n)]

final_results = []

desired_yield = args.desired_yield
directory_name = "True_Mass_inter"
# directory_name = "No_Tetramer"
file_name = f"yield_{kT}.txt"
# file_name = f"kt{kT}_epsS_6_epsW_1.txt"
output_file_path = os.path.join(directory_name, file_name)

# Ensure the directory exists

os.makedirs(directory_name, exist_ok=True)
with open(output_file_path, "w") as f:

    for i in tqdm(range(n_outer_iters)):
        (loss, mass_act_loss), grads = our_grad_fn(params, args.desired_yield)
        grads = masked_grads(grads)
        print(f"Iteration {i + 1}, Loss: {loss}")
        print(f"Mass action loss: {mass_act_loss}")
        print(f"Gradients: {grads}")
        print(f"Parameters: {params}")
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        conc_params = normalize_logits
        params = abs_array(params)
        norm_conc = normalize_logits(params[-n:], init_conc)
        params = jnp.concatenate([params[:-n], norm_conc])
        params = project(params)
        print("Updated Parameters:")
        for name, value in {
            name: params[idx] for idx, name in enumerate(param_names)
        }.items():
            print(f"{name}: {value}")
        print(params)
        fin_yield = ofer(params)
        fin_yield = jnp.exp(fin_yield[0])
        print(f"Desired Yield: {args.desired_yield}, Final Yield: {fin_yield}")

    final_params = params
    fin_yield = ofer(params)
    final_target_yields = jnp.exp(fin_yield[0])

    num_params = len(params)
    params_str = ",".join([f"{params[i]}" for i in range(num_params)])
    output_str = (
        f"{args.desired_yield},{final_target_yields},{mass_act_loss},{params_str}\n"
    )
    f.write(output_str)
    f.flush()
