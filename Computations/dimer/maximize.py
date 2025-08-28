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
import jax_transformations3d as jts
from jaxopt import implicit_diff, GradientDescent
from checkpoint import checkpoint_scan
import pdb
import functools
from functools import partial
import itertools
import matplotlib.pyplot as plt
from jax.config import config
import gc
import os
import argparse


SEED = 42
main_key = random.PRNGKey(SEED)


@partial(jit, static_argnums=(1,))
def safe_mask(mask, fn, operand, placeholder=0):
    masked = jnp.where(mask, operand, 0)
    return jnp.where(mask, fn(masked), placeholder)


def distance(dR):
    dr = jnp.sum(dR**2, axis=-1)
    return safe_mask(dr > 0, jnp.sqrt, dr)


# dist_fn = jnp.linalg.norm
dist_fn = distance


euler_scheme = "sxyz"


def safe_log(x, eps=1e-10):
    return jnp.log(jnp.clip(x, a_min=eps, a_max=None))


# Shape and energy helper functions
a = 1.0  # distance of the center of the spheres from the BB COM
b = 0.3
n = 2  # distance of the center of the patches from the BB COM
separation = 2.0
noise = 1e-14
vertex_radius = a
patch_radius = 0.2 * a
small_value = 1e-12
vertex_species = 0
n_species = 4
V = 54000.0


parser = argparse.ArgumentParser(description="Optimization script with argparse")
parser.add_argument(
    "--init_patchy_vals", type=float, required=True, help="Initial patchy values"
)
parser.add_argument(
    "--init_kt", type=float, required=True, help="Initial temperature value"
)
parser.add_argument(
    "--init_conc_val", type=float, required=True, help="Initial concentration value"
)
args = parser.parse_args()

# Extract command-line arguments
init_patchy_vals = args.init_patchy_vals
init_kT = args.init_kt
init_conc_val = args.init_conc_val

# Set up parameters
patchy_vals = jnp.full(3, init_patchy_vals)
init_conc = jnp.full(2, init_conc_val)
init_kT = jnp.array([init_kT])
init_params = jnp.concatenate([patchy_vals, init_kT, init_conc])


# mon_shapes = [sp.make_shape(1, a, b, edge_patches="right"), sp.make_shape(1, a, b, edge_patches="right")]

a = 1  # distance of the center of the spheres from the BB COM
b = 0.3  # distance of the center of the patches from the BB COM
mon_shape1 = onp.array(
    [
        [0.0, 0.0, a],  # first sphere
        [0.0, a * onp.cos(onp.pi / 6.0), -a * onp.sin(onp.pi / 6.0)],  # second sphere
        [0.0, -a * onp.cos(onp.pi / 6.0), -a * onp.sin(onp.pi / 6.0)],  # third sphere
        [a, 0.0, b],  # first patch
        [a, b * onp.cos(onp.pi / 6.0), -b * onp.sin(onp.pi / 6.0)],  # second patch
        [a, -b * onp.cos(onp.pi / 6.0), -b * onp.sin(onp.pi / 6.0)],  # third patch
    ]
)

mon_shape2 = jts.matrix_apply(
    jts.reflection_matrix(
        jnp.array([0, 0, 0], dtype=jnp.float64), jnp.array([1, 0, 0], dtype=jnp.float64)
    ),
    mon_shape1,
)
mon_shape2 = jts.matrix_apply(
    jts.reflection_matrix(
        jnp.array([0, 0, 0], dtype=jnp.float64), jnp.array([0, 1, 0], dtype=jnp.float64)
    ),
    mon_shape2,
)

dimer_shape = jnp.array([mon_shape1, mon_shape2])


mon_species_1 = onp.array([0, 0, 0, 1, 2, 3])
mon_species_2 = onp.array([0, 0, 0, 1, 3, 2])
dimer_species = onp.array([0, 0, 0, 1, 3, 2, 0, 0, 0, 1, 2, 3])


rb_1 = jnp.array([-separation / 2.0, 1e-15, 0, 0, 0, 0])
rb_2 = jnp.array(
    [-separation / 2.0, 1e-15, 0, 0, 0, 0, separation / 2.0, 0, 0, 0, 0, 0],
    dtype=jnp.float64,
)


def get_positions(q, ppos):
    Mat = []
    for i in range(2):
        qi = i * 6
        Mat.append(utils.convert_to_matrix(q[qi : qi + 6]))

    real_ppos = []
    for i in range(2):
        real_ppos.append(jts.matrix_apply(Mat[i], ppos[i]))

    return real_ppos


rep_rmax_table = jnp.full((n_species, n_species), small_value)
rep_rmax_table = rep_rmax_table.at[jnp.diag_indices(n_species)].set(2 * vertex_radius)
rep_rmax_table = rep_rmax_table.at[0, 0].set(small_value)

rep_alpha_table = jnp.full((n_species, n_species), small_value)
rep_alpha_table = rep_alpha_table.at[jnp.diag_indices(n_species)].set(2.5)
rep_alpha_table = rep_alpha_table.at[0, 0].set(small_value)

rep_A_table = jnp.full((n_species, n_species), 500.0)
rep_A_table = rep_A_table.at[0, 0].set(small_value)


morse_narrow_alpha = 5.0
morse_alpha_table = jnp.full((n_species, n_species), small_value)
morse_alpha_table = morse_alpha_table.at[jnp.diag_indices(n_species)].set(
    morse_narrow_alpha
)
morse_alpha_table = morse_alpha_table.at[0, 0].set(small_value)


main_key, subkey = random.split(main_key)
"""
morse_eps_table = small_value * random.uniform(subkey, shape=(n_species, n_species))


def make_tables(opt_params):
    diag_indices = jnp.arange(1, min(n_species, len(opt_params) + 1))  
    morse_eps_table = morse_eps_table.at[(diag_indices, diag_indices)].set(opt_params)
    return morse_eps_table


def pairwise_repulsion(ipos, jpos, i_species, j_species):
  
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)

    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)
               
                     
def pairwise_morse(ipos, jpos, i_species, j_species, opt_params):
    morse_eps_table = make_tables(opt_params)                 
    morse_d0 = morse_eps_table[i_species, j_species]
    morse_alpha = morse_alpha_table[i_species, j_species]
    morse_alpha = jnp.clip(morse_alpha, 1e-6, None)
    morse_r0 = 0.0                                   
    morse_rcut = 8. / morse_alpha + morse_r0
    dr = space.distance(ipos - jpos)
                     
    return potentials.morse_x(dr, rmin=morse_r0, rmax=morse_rcut, D0=morse_d0, 
                   alpha=morse_alpha, r0=morse_r0, ron=morse_rcut/2.)   


"""


@jit
def get_energy_fns(q, pos, species, opt_params):

    Nbb = 2

    morse_rcut = 8.0 / 5.0

    sphere_radius = 1.0
    patch_radius = 0.2 * sphere_radius

    tot_energy = jnp.float64(0)
    real_ppos = get_positions(q, pos)

    def j_repulsive_fn(j, pos1):
        pos2 = real_ppos[1][j]
        r = dist_fn(pos1 - pos2)
        return potentials.repulsive(
            r, rmin=0, rmax=sphere_radius * 2, A=500.0, alpha=2.5
        )

    def i_repulsive_fn(i):
        pos1 = real_ppos[0][i]
        all_j_terms = vmap(j_repulsive_fn, (0, None))(jnp.arange(3), pos1)
        return jnp.sum(all_j_terms)

    repulsive_sm = jnp.sum(vmap(i_repulsive_fn)(jnp.arange(3)))
    tot_energy += repulsive_sm

    pos1 = real_ppos[0][3]
    pos2 = real_ppos[1][3]

    r = dist_fn(pos1 - pos2)
    tot_energy += potentials.morse_x(
        r,
        rmin=0,
        rmax=morse_rcut,
        D0=opt_params[0],
        alpha=5.0,
        r0=0.0,
        ron=morse_rcut / 2.0,
    )

    pos1 = real_ppos[0][5]
    pos2 = real_ppos[1][4]
    r = dist_fn(pos1 - pos2)
    tot_energy += potentials.morse_x(
        r,
        rmin=0,
        rmax=morse_rcut,
        D0=opt_params[1],
        alpha=5.0,
        r0=0.0,
        ron=morse_rcut / 2.0,
    )

    pos1 = real_ppos[0][4]
    pos2 = real_ppos[1][5]
    r = dist_fn(pos1 - pos2)
    tot_energy += potentials.morse_x(
        r,
        rmin=0,
        rmax=morse_rcut,
        D0=opt_params[2],
        alpha=5.0,
        r0=0.0,
        ron=morse_rcut / 2.0,
    )

    return tot_energy


"""
def get_energy(q, pos, species, opt_params):
    
    positions = get_positions(q, pos)

    pos1 = positions[0]  
    pos2 = positions[1]  

    species1 = species[:6]  
    species2 = species[6:]  

    morse_func = vmap(vmap(pairwise_morse, in_axes=(None, 0, None, 0, None)), in_axes=(0, None, 0, None, None))
    tot_energy = jnp.sum(morse_func(pos1, pos2, species1, species2, opt_params))
    
    inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
    rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))
    tot_energy += jnp.sum(rep_func(pos1, pos2, species1, species2))
    
    return tot_energy
"""


def add_variables(ma, mb):
    """
    given two vectors of length (6,) corresponding to x,y,z,alpha,beta,gamma,
    convert to transformation matrixes, 'add' them via matrix multiplication,
    and convert back to x,y,z,alpha,beta,gamma

    note: add_variables(ma,mb) != add_variables(mb,ma)
    """

    Ma = utils.convert_to_matrix(ma)
    Mb = utils.convert_to_matrix(mb)
    Mab = jnp.matmul(Mb, Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))

    return jnp.concatenate((trans, angles))


def add_variables_all(mas, mbs):
    """
    Given two vectors of length (6*n,), 'add' them per building block according
    to add_variables().
    """

    mas_temp = jnp.reshape(mas, (mas.shape[0] // 6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0] // 6, 6))

    return jnp.reshape(
        vmap(add_variables, in_axes=(0, 0))(mas_temp, mbs_temp), mas.shape
    )


"""
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

# Use `vmap` over the required axis
morse_func = vmap(pairwise_morse, in_axes=(0, None, 0, None, None))







def pairwise_repulsion(ipos, jpos, i_species, j_species):
    rep_rmax = rep_rmax_table[i_species, j_species]
    rep_a = rep_A_table[i_species, j_species]
    rep_alpha = rep_alpha_table[i_species, j_species]
    dr = space.distance(ipos - jpos)
    return potentials.repulsive(dr, rmin=0, rmax=rep_rmax, A=rep_a, alpha=rep_alpha)


inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))


def get_nmer_energy_fn(n):
    pairs = jnp.array(onp.array(list(itertools.combinations(onp.arange(n), 2))))

    def nmer_energy_fn(q, pos, species, opt_params):
        positions = utils.get_positions(q, pos)
        pos_slices = [(i * 6, (i + 1) * 6) for i in range(n)]
        species_slices = [(i * 6, (i + 1) * 6) for i in range(n)]

        all_pos = jnp.stack([positions[start:end] for start, end in pos_slices])
        all_species = jnp.stack(
            [jnp.repeat(species[start:end], 1) for start, end in species_slices]
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

def get_energy(q, pos, species, opt_params):
    
    positions = get_positions(q, pos)

    pos1 = positions[0]  
    pos2 = positions[1]  

    species1 = species[:6]  
    species2 = species[6:12]  

    morse_func = vmap(vmap(pairwise_morse, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))
    tot_energy = jnp.sum(morse_func(pos1, pos2, species1, species2, opt_params))
    
    inner_rep = vmap(pairwise_repulsion, in_axes=(None, 0, None, 0))
    rep_func = vmap(inner_rep, in_axes=(0, None, 0, None))
    tot_energy += jnp.sum(rep_func(pos1, pos2, species1, species2))

    return tot_energy
"""


def hess(energy_fn, q, pos, species, opt_params):
    H = hessian(energy_fn)(q, pos, species, opt_params)
    evals, evecs = jnp.linalg.eigh(H)
    return evals, evecs


def compute_zvib(energy_fn, q, pos, species, opt_params):
    evals, evecs = hess(energy_fn, q, pos, species, opt_params)
    zvib = jnp.prod(
        jnp.sqrt(2.0 * jnp.pi / (opt_params[3] * (jnp.abs(evals[6:]) + 1e-12)))
    )
    return zvib


def compute_zrot_mod_sigma(
    energy_fn, q, pos, species, opt_params, key, size, nrandom=100000
):
    Nbb = size
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


def compute_zc(boltzmann_weight, z_rot_mod_sigma, z_vib, sigma=3, V=V):
    z_trans = V
    z_rot = z_rot_mod_sigma / sigma
    return boltzmann_weight * z_trans * z_rot * z_vib


mon_energy_fn = lambda q, pos, species, opt_params: 0.0

zrot_mod_sigma_1, _, main_key = compute_zrot_mod_sigma(
    mon_energy_fn, rb_1, mon_shape1, mon_species_1, patchy_vals, main_key, 1
)
zvib_1 = 1.0
boltzmann_weight = 1.0

z_1 = compute_zc(boltzmann_weight, zrot_mod_sigma_1, zvib_1)
z_1s = jnp.full(n, z_1)
log_z_1 = safe_log(z_1s)


def get_log_z_all(opt_params, key, rb=rb_2, shape=dimer_shape, species=dimer_species):
    dim_energy_fn = get_energy_fns
    zvib = compute_zvib(dim_energy_fn, rb, shape, species, opt_params)
    e0 = dim_energy_fn(rb, shape, species, opt_params)
    boltzmann_weight = jnp.exp(-jnp.clip(e0 / opt_params[3], a_min=-100, a_max=100))
    zrot_mod_sigma, _, new_key = compute_zrot_mod_sigma(
        dim_energy_fn, rb, shape, species, opt_params, key, 2
    )
    z = compute_zc(boltzmann_weight, zrot_mod_sigma, zvib)
    z_log = safe_log(z)
    z_log_2 = jnp.array([z_log])
    z_all = jnp.concatenate([log_z_1, z_log_2], axis=0)
    return z_all, new_key


nper_structure = nper_structure = jnp.array([[0, 1, 1], [1, 0, 1]])


def loss_fn(log_concs_struc, log_z_list, opt_params):
    m_conc = opt_params[-n:]
    tot_conc = jnp.sum(m_conc)
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

    mon_loss = vmap(mon_loss_fn)(jnp.arange(n))
    struc_loss = vmap(struc_loss_fn)(jnp.arange(n, 3))
    combined_loss = jnp.concatenate([mon_loss, struc_loss])
    loss_var = jnp.var(combined_loss)
    loss_max = jnp.max(combined_loss)
    tot_loss = jnp.linalg.norm(combined_loss) + loss_var
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
    max_loss = jnp.max(combined_losses)
    second_max_loss = jnp.partition(combined_losses, -2)[-2]

    return final_params


def ofer(opt_params, key):
    log_z_list, new_key = get_log_z_all(opt_params, key)
    tot_conc = jnp.sum(opt_params[-n:])
    struc_concs_guess = jnp.full(3, safe_log(tot_conc / 3))
    fin_log_concs = inner_solver(struc_concs_guess, log_z_list, opt_params)
    fin_concs = jnp.exp(fin_log_concs)
    yields = fin_concs / jnp.sum(fin_concs)
    target_yield = safe_log(yields[2])
    return target_yield, new_key


def ofer_grad_fn(opt_params, key):
    target_yield, new_key = ofer(opt_params, key)
    loss = target_yield
    return -loss, new_key


def project(params):
    conc_min, conc_max = 1e-6, 3.0
    kbt_min, kbt_max = 1e-6, 3.0
    kbt_idx = 3
    concs = jnp.clip(params[-n:], a_min=conc_min, a_max=conc_max)
    kbt_val = jnp.clip(params[kbt_idx], a_min=kbt_min, a_max=kbt_max)
    kbt = jnp.array([kbt_val])
    return jnp.concatenate([params[:kbt_idx], kbt, concs])


num_params = len(init_params)
mask = jnp.zeros(num_params)
mask = mask.at[:3].set(1.0)


def masked_grads(grads):
    return grads * mask


our_grad_fn = jit(value_and_grad(ofer_grad_fn, has_aux=True))
params = init_params
outer_optimizer = optax.adam(1e-2)
opt_state = outer_optimizer.init(params)

n_outer_iters = 400
outer_losses = []

use_custom_pairs = True
custom_pairs = [[1, 1], [2, 2], [3, 3]]
n_patches = 3

if use_custom_pairs and custom_pairs is not None:
    param_names = [f"Eps({i},{j})" for i, j in custom_pairs]
else:
    param_names = [f"Eps({i},{j})" for i, j in generated_idx_pairs]
    param_names += [f"Eps({i},{i})" for i in range(1, n_patches + 1)]

param_names += ["kT"]

param_names += [f"conc_{chr(ord('A') + i)}" for i in range(n)]

final_results = []


os.makedirs("Maximized_agnese", exist_ok=True)
output_filename = f"Maximized_agnese/kbt{init_kT[0]:.2f}.txt"
#os.makedirs("Nonmaximized_agnese", exist_ok=True)
#output_filename = f"Nonmaximized_agnese/kbt{init_kT[0]:.2f}.txt"

with open(output_filename, "w") as f:


    for i in tqdm(range(n_outer_iters)):
        (loss, main_key), grads = our_grad_fn(params, main_key)
        # outer_losses.append(loss)
        grads = masked_grads(grads)
        print(f"Iteration {i + 1}: Loss = {loss}")
        updates, opt_state = outer_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # params = project(params)
        print("Updated Parameters:")
        for name, value in {
            name: params[idx] for idx, name in enumerate(param_names)
        }.items():
            print(f"{name}: {value}")
        print(params)
        fin_yield, main_key = ofer(params, main_key)
        fin_yield = jnp.exp(fin_yield)
        print(f"Yield: {fin_yield}")

    final_params = params
    fin_yield, main_key = ofer(params, main_key)
    final_target_yields = jnp.exp(fin_yield)

    f.write(
        f"{final_target_yields},{params[0]},{params[1]},{params[2]}\n"
    )
    # f.write(f"{des_yield}, {final_target_yields}, {params[3]}\n")
    f.flush()


print("All results saved.")
