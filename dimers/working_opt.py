import pdb
from tqdm import tqdm
import argparse
import numpy as onp
import scipy as osp
import csv
import matplotlib.pyplot as plt
import time
from random import randint
from jax import lax
from functools import partial

from jax import random
from jax import jit, grad, vmap, value_and_grad, hessian, jacfwd, jacrev, tree_util 
import jax.numpy as jnp
import optax

import potentials
from jax_transformations3d import jax_transformations3d as jts
from utils import euler_scheme, convert_to_matrix, ref_ppos, ref_q0

from jax.config import config
config.update("jax_enable_x64", True)



some_big_number = 100
factorial_table = jnp.array([osp.special.factorial(x) for x in range(some_big_number)])
comb_table = onp.zeros((some_big_number, some_big_number))
for i in range(some_big_number):
    for j in range(some_big_number):
        if i >= j:
            comb_table[i, j] = osp.special.comb(i, j)
comb_table = jnp.array(comb_table)




def check_for_nans(variable, name):
    if jnp.isnan(variable).any():
        print(f"{name} contains NaNs: {variable}")
        raise ValueError(f"{name} contains NaNs")  # Raise an error to stop the program if NaNs are found


def get_energy_fns(args):

    Nbb = 2

    def monomer_energy(q, ppos):
        # assert(q.shape[0] == 6)
        return jnp.float64(0)


    sphere_radius = 1.0
    patch_radius = 0.2 * sphere_radius
    # types: ['A', 'B1', 'B2', 'G1', 'G2', 'R1', 'R2']
    # BBt[0].typeids: array([0, 0, 0, 1, 5, 3])
    # BBt[1].typeids: array([0, 0, 0, 2, 4, 6])

    morse_rcut = 8. / args['morse_a'] + args['morse_r0']
    def cluster_energy(q, ppos):
        # convert the building block coordinates to a tranformation
        # matrix
        Mat = []
        for i in range(Nbb):
            qi = i*6
            Mat.append(convert_to_matrix(q[qi:qi+6]))

        # apply building block matrix to spheres positions
        real_ppos = []
        for i in range(Nbb):
            real_ppos.append(jts.matrix_apply(Mat[i], ppos[i]))

        tot_energy = jnp.float64(0)

        # Add repulsive interaction between spheres
        for i in range(3):
            pos1 = real_ppos[0][i]
            for j in range(3):
                pos2 = real_ppos[1][j]
                r = jnp.linalg.norm(pos1-pos2)
                tot_energy += potentials.repulsive(
                    r, rmin=0, rmax=sphere_radius*2,
                    A=args['rep_A'], alpha=args['rep_alpha'])

        # Add attraction b/w blue patches
        pos1 = real_ppos[0][3]
        pos2 = real_ppos[1][3]
        r = jnp.linalg.norm(pos1-pos2)
        tot_energy += potentials.morse_x(
            r, rmin=0, rmax=morse_rcut,
            D0=args['morse_d0']*args['morse_d0_b'],
            alpha=args['morse_a'], r0=args['morse_r0'],
            ron=morse_rcut/2.)

        # Add attraction b/w green patches
        pos1 = real_ppos[0][5]
        pos2 = real_ppos[1][4]
        r = jnp.linalg.norm(pos1-pos2)
        tot_energy += potentials.morse_x(
            r, rmin=0, rmax=morse_rcut,
            D0=args['morse_d0']*args['morse_d0_g'],
            alpha=args['morse_a'], r0=args['morse_r0'],
            ron=morse_rcut/2.)

        # Add attraction b/w red patches
        pos1 = real_ppos[0][4]
        pos2 = real_ppos[1][5]
        r = jnp.linalg.norm(pos1-pos2)
        tot_energy += potentials.morse_x(
            r, rmin=0, rmax=morse_rcut,
            D0=args['morse_d0']*args['morse_d0_r'],
            alpha=args['morse_a'], r0=args['morse_r0'],
            ron=morse_rcut/2.)

        # Note: no repulsion between identical patches, as in Agnese's code. May affect simulations.
        return tot_energy

    return monomer_energy, cluster_energy

def add_variables(ma, mb):
    """
    given two vectors of length (6,) corresponding to x,y,z,alpha,beta,gamma,
    convert to transformation matrixes, 'add' them via matrix multiplication,
    and convert back to x,y,z,alpha,beta,gamma

    note: add_variables(ma,mb) != add_variables(mb,ma)
    """

    Ma = convert_to_matrix(ma)
    Mb = convert_to_matrix(mb)
    Mab = jnp.matmul(Mb,Ma)
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

    return jnp.reshape(vmap(add_variables, in_axes=(0, 0))(
        mas_temp, mbs_temp), mas.shape)


def setup_variable_transformation(energy_fn, q0, ppos):
    """
    Args:
    energy_fn: function to calculate the energy:
    E = energy_fn(q, euler_scheme, ppos)
    q0: initial coordinates (positions and orientations) of the building blocks
    ppos: "patch positions", array of shape (N_bb, N_patches, dimension)

    Returns: function f that defines the coordinate transformation, as well as
    the number of zero modes (which should be 6) and Z_vib

    Note: we assume without checking that ppos is defined such that all the euler
    angels in q0 are initially 0.
    """

    Nbb = q0.shape[0] // 6 # Number of building blocks
    assert(Nbb*6 == q0.shape[0])
    assert(len(ppos.shape) == 3)
    assert(ppos.shape[0] == Nbb)
    assert(ppos.shape[2] == 3)

    E = energy_fn(q0, ppos)
    G = grad(energy_fn)(q0, ppos)
    H = hessian(energy_fn)(q0, ppos)

    evals, evecs = jnp.linalg.eigh(H)


    zeromode_thresh = 1e-8
    num_zero_modes = jnp.sum(jnp.where(evals < zeromode_thresh, 1, 0))

    if Nbb == 1:
        zvib = 1.0
    else:
        zvib = jnp.prod(jnp.sqrt(2.*jnp.pi/(jnp.abs(evals[6:])+1e-12)))

    def ftilde(nu):
        return jnp.matmul(evecs.T[6:].T, nu[6:])

    def f_multimer(nu, addq0=True):
        # q0+ftilde
        dq_tilde = ftilde(nu)

        q_tilde = jnp.where(addq0, add_variables_all(q0, dq_tilde), dq_tilde)

        nu_bar_repeat = jnp.reshape(jnp.array([nu[:6] for _ in range(Nbb)]), nu.shape)
        return add_variables_all(q_tilde, nu_bar_repeat)

    def f_monomer(nu, addq0=True):
        return nu

    if Nbb == 1:
        f = f_monomer
    else:
        f = f_multimer

    return jit(f), num_zero_modes, zvib
    
def standard_error(data):
    mean = jnp.mean(data, axis=0)

    # Calculate the standard error using the formula: std(data) / sqrt(N)
    std_dev = jnp.std(data, axis=0)
    sqrt_n = jnp.sqrt(data.shape[0])
    std_error = std_dev / sqrt_n

    return std_error


def calc_jmean(f, key, nrandom=100000):
    def random_euler_angles(key):
        quat = jts.random_quaternion(None, key)
        return jnp.array(jts.euler_from_quaternion(quat, euler_scheme))

    def set_nu(angles):
        nu0 = jnp.full((12,), 0.0)
        return nu0.at[3:6].set(angles)

    def set_nu_random(key):
        return set_nu(random_euler_angles(key))

    key, *splits = random.split(key, nrandom+1)
    nus = vmap(set_nu_random)(jnp.array(splits))

    nu_fn = jit(lambda nu: jnp.abs(jnp.linalg.det(jacfwd(f)(nu, False))))
    Js = vmap(nu_fn)(nus)
    # pdb.set_trace()
    mean = jnp.mean(Js)
    # error = osp.stats.sem(Js)
    # error = osp.stats.sem(Js)
    error = standard_error(Js)

    return mean, error


def calculate_zc(key, energy_fn, all_q0, all_ppos, sigma, kBT, V):
    # Check for NaNs in inputs
    check_for_nans(all_q0, "all_q0")
    check_for_nans(all_ppos, "all_ppos")

    # Set up the variable transformation
    f, num_zero_modes, zvib = setup_variable_transformation(energy_fn, all_q0, all_ppos)

    # Calculate the mean of the Jacobian determinant
    Js_mean, Js_error = calc_jmean(f, key)
    Jtilde = 8.0 * (jnp.pi**2) * Js_mean
    check_for_nans(Jtilde, "Jtilde")

    # Compute the energy at the initial configuration
    E0 = energy_fn(all_q0, all_ppos)
    check_for_nans(E0, "E0")

    # Boltzmann weight calculation
    boltzmann_weight = jnp.exp(-E0 / (kBT + 1e-8))
    check_for_nans(boltzmann_weight, "boltzmann_weight")

    # Check zvib for NaNs
    check_for_nans(zvib, "zvib")

    # Final Zc calculation
    Zc = boltzmann_weight * V * (Jtilde / sigma) * zvib
    check_for_nans(Zc, "Zc")

    return Zc



def Calculate_pc_list(N_mon, Zc_monomer, Zc_dimer, exact=False):
    # nd_fact = jax_factorial(N_mon_real)
    N_mon_real = N_mon
    def Mc(Nd):
        return comb_table[N_mon_real, Nd] * comb_table[N_mon_real, Nd] * factorial_table[Nd]

    def Pc(Nd):
        return Mc(Nd) * (Zc_dimer**Nd) * (Zc_monomer**(N_mon_real-Nd)) * (Zc_monomer**(N_mon_real-Nd))

    pc_list = vmap(Pc)(jnp.arange(N_mon_real+1))
    return pc_list / jnp.sum(pc_list)

def Calculate_yield_can(Nb, Nr, pc_list):
    Y_list = jnp.array([Nd / (Nb+Nr-Nd) for Nd in range(len(pc_list))])
    return jnp.dot(Y_list, pc_list)

def run(args, kBT=1, n=27, seed=0):

    key = random.PRNGKey(seed)

    monomer_energy, dimer_energy = get_energy_fns(args)

    Nblue, Nred = n, n

    conc = args['conc']
    Ntot = 2*n
    V = Ntot / conc
    
    split1, split2 = random.split(key)
    Zc_dimer = calculate_zc(
        split1, dimer_energy, ref_q0, ref_ppos,
        sigma=1, kBT=kBT, V=V)
    Zc_monomer = calculate_zc(
        split2, monomer_energy,
        ref_q0[:6], jnp.array([ref_ppos[0]]),
        sigma=1, kBT=kBT, V=V)

    pc_list = Calculate_pc_list(Nblue, Zc_monomer, Zc_dimer)
    Y_dimer = Calculate_yield_can(Nblue, Nred, pc_list)

    return Y_dimer, pc_list

def get_argparse():
    parser = argparse.ArgumentParser(description='Compute the yield of a simple dimer system')

    # System setup
    parser.add_argument('-c', '--conc', type=float,  default=0.001, help='Monomer concentration')
        # Morse interaction
    parser.add_argument('--morse-d0', type=float,  default=7.202929758352057,
                        help='d0 parameter for Morse interaction')
    parser.add_argument('--morse-d0-r', type=float,  default=1.2100346259989965,
                        help='Scalar for d0 for red patches')
    parser.add_argument('--morse-d0-g', type=float,  default=1.2099136420636807,
                        help='Scalar for d0 for green patches')
    parser.add_argument('--morse-d0-b', type=float,  default=1.2100346259989965,
                        help='Scalar for d0 for blue patches')
    parser.add_argument('-n', '--num-monomer', type=int,  default=27,
                        help='Number of each kind of monomer')

    # Repulsive interaction
    parser.add_argument('--rep-A', type=float,  default=500.0,
                        help='A parameter for repulsive interaction')
    parser.add_argument('--rep-alpha', type=float,  default=2.5,
                        help='alpha parameter for repulsive interaction')


    parser.add_argument('--morse-a', type=float,  default=5.0,
                        help='alpha parameter for Morse interaction')
    parser.add_argument('--morse-r0', type=float,  default=0.0,
                        help='r0 parameter for Morse interaction')

    return parser


if __name__ == "__main__":

    parser = get_argparse()
    args = vars(parser.parse_args())

    
    def yield_fn(d0, dr, db, dg, args, seed):
        args['morse_d0'] = d0
        args['morse_d0_r'] = dr
        args['morse_d0_g'] = dg
        args['morse_d0_b'] = db
        yi, _ = run(args, seed)
        return yi


    #def loss_fn(params, args, target_yield, seed):
    def loss_fn(params, args, seed):
        d0 = params['d0']
        dr = params['dr']
        dg = params['dg']
        db = params['db']
        yi = yield_fn(d0, dr, db, dg, args, seed)
        loss = (yi - 0.5) ** 2
        return loss, yi
                      
    d0 = 10.0
    dr = 1.0 
    db = 1.0   
    dg = 1.0    


    num_iters = 100
    learning_rate = 0.01
    optimizer = optax.adam(learning_rate)
    params = {'d0': d0,  'dr': dr,  'dg': dg,  'db': db}
                  
    grad_yield = jit(jacrev(loss_fn, has_aux=True))                  

    opt_state = optimizer.init(params)
 
    yield_path = "yield.txt"
    grad_path = "grads.txt"
    d0_path = "d0s.txt"
    
    open(yield_path, 'w').close()
    open(grad_path, 'w').close()
    open(d0_path, 'w').close()

    for i in tqdm(range(num_iters)):
        print(f"Iteration {i}:")

        #grads, curr_yield = grad_yield(params, args, target_yield, seed=9)
        grads, curr_yield = grad_yield(params, args, seed=9)
        # grads = grads['d0']
        # grads_numeric = grads.item()
        curr_d0_grad = grads['d0']
        curr_dr_grad = grads['dr']
        curr_dg_grad = grads['dg']
        curr_db_grad = grads['db']

        print(f"\t- d0 grad: {float(curr_d0_grad)}")
        print(f"\t- dr grad: {float(curr_dr_grad)}")
        print(f"\t- dg grad: {float(curr_dg_grad)}")
        print(f"\t- db grad: {float(curr_db_grad)}")
      
        print(f"\t- current yield: {curr_yield}")

        curr_d0 = params['d0']
        curr_dr = params['dr']
        curr_dg = params['dg']
        curr_db = params['db']             
        print(f"\t- current d0s: {float(curr_d0)}, {float(curr_dr)}, {float(curr_db)}, {float(curr_dg)}")

        with open(yield_path, "a") as f:
            f.write(f"{curr_yield}\n")
        with open(grad_path, "a") as f:
            f.write(f"{curr_d0_grad},{curr_dr_grad},{curr_dg_grad},{curr_db_grad}\n")
        with open(d0_path, "a") as f:
            f.write(f"{curr_d0},{curr_dr},{curr_dg},{curr_db}\n")

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
  
    pdb.set_trace()

    print("done")

