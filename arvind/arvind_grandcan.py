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
import matplotlib.pyplot as plt
from jax.config import config
import gc
import os
import networkx as nx
from itertools import product 
import argparse

SEED = 42
main_key = random.PRNGKey(SEED)

import jax.numpy as jnp
from itertools import permutations
from jax import random

# Define constants
a = 1.0  # Radius placeholder
b = .3
separation = 2.0
noise = 1e-14






def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))
# Define target as a JAX array directly
#target = jnp.array([1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10, 11, 0, 12, 13, 0, 14])
#target = jnp.array([1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10])
#target = jnp.array([1, 0, 2, 3, 0, 4, 5, 0, 6, 7,0,8])
target = jnp.array([1, 0, 2, 3, 0, 4, 5, 0, 6])

use_custom_pairs = True
#custom_pairs = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
#custom_pairs = [(2, 3), (4, 5), (6, 7)]
custom_pairs = [(2, 3), (4, 5)]
#custom_pairs = [(2, 3), (4, 5), (6, 7), (8, 9)]

def load_species_combinations(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

data = load_species_combinations("arvind_3.pkl")

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

parser = argparse.ArgumentParser(description="Simulation with adjustable parameters.")
parser.add_argument(
    "--eps_weak",
    type=float,
    default=1.0,
    help="weak epsilon values (attraction strengths).",
)
parser.add_argument(
    "--eps_init",
    type=float,
    default=6.,
    help="init strong epsilon values (attraction strengths).",
)
parser.add_argument(
    "--kt", type=float, default=1.0, help="Thermal energy (kT). Default is 1.0."
)
parser.add_argument(
    "--init_conc",
    type=float,
    default=0.001,
    help="Initial concentration. Default is 0.001.",
)
parser.add_argument(
    "--desired_yield", type=float, default=0.4, help="desired yield of target."
)

args = parser.parse_args()



V =  54000.0
kT = args.kt
n = num_monomers  

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

n_morse_vals = (
    n_patches * (n_patches - 1) // 2 + n_patches
)  # all possible pair permutations plus same patch attraction (i,i)
patchy_vals = jnp.full(
    n-1, args.eps_init) # FIXME for optimization over specific attraction strengths

init_conc = args.init_conc/n
init_concs = jnp.full(n, init_conc)
weak_eps = jnp.array([args.eps_weak])
init_params = jnp.concatenate([patchy_vals, weak_eps, init_concs])


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
    morse_eps_table = jnp.full((n_species, n_species), opt_params[2])
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
    H = hessian(energy_fn)(q, pos, species, opt_params)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def compute_zvib(energy_fn, q, pos, species, opt_params):
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(jnp.sqrt(2.0 * jnp.pi / (kT * jnp.abs(evals[6:]) + 1e-12)))
    return zvib


def compute_zrot_mod_sigma(energy_fn, q, pos, species, opt_params, key, nrandom=100000):
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

sizes = range(1, n+1)


rbs = {}

for size in sizes:
    main_key, subkey = random.split(main_key)
    rb, subkey = make_rb(size, subkey)
    rbs[size] = rb
    main_key = subkey
    
shapes = {size: make_shape(size) for size in sizes}
sigmas = {size: data[f'{size}_sigma'] for size in sizes if f'{size}_sigma' in data}
energy_fns = {size: jit(get_nmer_energy_fn(size)) for size in range(2, n+1)}

rb1 = rbs[1]
shape1 = shapes[1]
#sigma1 = data["1_sigma"]
mon_energy_fn = lambda q, pos, species, opt_params: 0.0



zrot_mod_sigma_1,_, main_key = compute_zrot_mod_sigma(mon_energy_fn, rb1, shape1,  jnp.array([1, 0, 2]), patchy_vals, main_key)
zvib_1 = 1.0
boltzmann_weight = 1.0

z_1 = compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1, sigmas[1])
z_1s = jnp.full(n, z_1)
log_z_1 = jnp.log(z_1s)

zrot_mod_sigma_values = {}

for size in range(2, n + 1):

    zrot_mod_sigma, Js, main_key = compute_zrot_mod_sigma(
        energy_fns[size], 
        rbs[size], 
        shapes[size], 
        jnp.array([1, 0, 2] * size), 
        patchy_vals, 
        main_key  
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
        boltzmann_weight = jnp.exp(-e0 / kT)
        z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib, sigma)
        return jnp.log(z)
    
    log_z_all = []
    
    for size in range(2, n + 1):
        species = data[f'{size}_pc_species']
        sigma = data[f'{size}_sigma']
        
        # Repeat sigma for each structure in species of the current size
        sigma_array = jnp.full(species.shape[0], sigma)
        
        if size <= 4:
            log_z = vmap(lambda sp, sg: compute_log_z(size, sp, sg))(species, sigma_array)
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
            scan_with_ckpt = functools.partial(checkpoint_scan, checkpoint_every=checkpoint_freq)
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
    for i in range(1, n+1):  
        key = f"{letter}_{i}_counts"
        if key in data:
            counts_list.append(data[key])
    if counts_list:  
        monomer_counts.append(jnp.concatenate(counts_list))


nper_structure = jnp.array(monomer_counts)

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

def optimality_fn(log_concs_struc, log_z_list, opt_params):
    return grad(lambda log_concs_struc, log_z_list, opt_params: loss_fn(log_concs_struc, log_z_list, opt_params)[0])(log_concs_struc, log_z_list, opt_params)


@implicit_diff.custom_root(optimality_fn)
def inner_solver(init_guess, log_z_list, opt_params):
    gd = GradientDescent(fun=lambda log_concs_struc, log_z_list, opt_params: loss_fn(log_concs_struc, log_z_list, opt_params)[0], maxiter=50000, implicit_diff=True)
    sol = gd.run(init_guess, log_z_list, opt_params)
    
    final_params = sol.params
    final_loss, combined_losses, loss_var = loss_fn(final_params, log_z_list, opt_params)
    max_loss = jnp.max(combined_losses)
    second_max_loss = jnp.partition(combined_losses, -2)[-2]
    
    return final_params


#########################

def ofer(opt_params):
    log_z_list = get_log_z_all(opt_params[:n])
    tot_conc = init_conc
    struc_concs_guess = jnp.full(tot_num_structures, safe_log(tot_conc / tot_num_structures))
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list, opt_params)
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / jnp.sum(fin_concs)
    target_yield = safe_log(yields[target_idx])
    return target_yield

def ofer_grad_fn(opt_params, desired_yield_val):
    target_yield = ofer(opt_params)
    #loss = (desired_yield_val - jnp.exp(target_yield))**4
    loss = (abs(jnp.log(desired_yield_val)- target_yield))**2
    return loss

def ofer_grad_max(opt_params):
    target_yield = ofer(opt_params)
    loss = - target_yield
    return loss

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

mask = jnp.full(num_params, 0.0)
#mask = mask.at[-n:].set(1.0)


def masked_grads(grads):
    return grads * mask

def project(params):
    conc_min = 1e-6
    concs = jnp.clip(params[-n:], a_min=conc_min)
    return jnp.concatenate([params[0:n], concs])

our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=False))
our_grad_max = jit(value_and_grad(ofer_grad_max, has_aux=False))
#our_grad_fn = value_and_grad(ofer_grad_fn, has_aux=False)
params = init_params
print("init params are:", params)
outer_optimizer = optax.adam(1e-2)
opt_state = outer_optimizer.init(params)

n_outer_iters = 2
outer_losses = []

if use_custom_pairs and custom_pairs is not None:
    param_names = [f"Eps({i},{j})" for i, j in custom_pairs]
else:
    param_names = [f"Eps({i},{j})" for i, j in generated_idx_pairs]
    param_names += [f"Eps({i},{i})" for i in range(1, n_patches + 1)]

param_names += [f"A conc:" ]
param_names += [f"B conc:" ]
param_names += [f"C conc:" ]
#param_names += [f"D conc:" ]

final_results = []

    
#desired_yields_range = jnp.arange(0.1,0.3, 0.1)


"""  
for i in tqdm(range(n_outer_iters)):
    # Compute loss and gradients
    loss, grads = our_grad_max(params`)
    grads = masked_grads(grads)

    # Print loss and gradients
    print(f"Iteration {i + 1}: Loss = {loss}")

    # Update parameters and print updated parameters
    updates, opt_state = outer_optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    params = abs_array(params)
    norm_conc = normalize_logits(params[-n:], 0.001)
    params = jnp.concatenate([params[:-n], norm_conc])
    params = project(params)
    print("Updated Parameters:")
    #for name, value in {name: params[idx] for idx, name in enumerate(param_names)}.items():
       # print(f"{name}: {value}")
    print(params)

    # Compute yield (assuming fin_yield is defined here)
    fin_yield = jnp.exp(ofer(params))   
    print(f"Yield: {fin_yield}")

"""    
desired_yield = args.desired_yield

directory_name = "Arvind_Stoc"
#file_name = f"yield_{desired_yield}_kt{kT}.txt"  
file_name = f"kt{kT}_epsS_6_epsW_1.txt"
output_file_path = os.path.join(directory_name, file_name)

# Ensure the directory exists

os.makedirs(directory_name, exist_ok=True)
with open(output_file_path, "w") as f:

    for i in tqdm(range(n_outer_iters)):
        loss, grads = our_grad_fn(params, args.desired_yield)
        #loss, grads = our_grad_max(params)
        grads = masked_grads(grads)
        print(f"Iteration {i + 1}, Loss: {loss}")    
            
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        conc_params = normalize_logits
        params = abs_array(params)
        norm_conc = normalize_logits(params[-n:], args.init_conc)
        params = jnp.concatenate([params[:-n], norm_conc])
        params = project(params)
        print("Updated Parameters:")
        for name, value in {
            name: params[idx] for idx, name in enumerate(param_names)
        }.items():
            print(f"{name}: {value}")
        print(params)
        fin_yield = ofer(params)
        fin_yield = jnp.exp(fin_yield)
        print(f"Desired Yield: {args.desired_yield}, Final Yield: {fin_yield}")

    final_params = params
    fin_yield = ofer(params)
    final_target_yields = jnp.exp(fin_yield)

    f.write(
        f"{args.desired_yield},{final_target_yields},{params[0]},{params[1]},{params[2]},{params[3]},{params[4]},{params[5]}\n"
    )
    f.flush()
