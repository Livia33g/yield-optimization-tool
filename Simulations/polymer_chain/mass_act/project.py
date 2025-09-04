import numpy as np
import jax.numpy as jnp
from jax import vmap, grad
import signac
from flow import FlowProject
from flow import directives
import flow.environments
import json
import os
import gsd.hoomd
import hoomd
from hoomd import md
from scipy.spatial.transform import Rotation
from pair_potentials import set_pair_potentials_params


class Project(FlowProject):
    pass


@Project.label
def preint(job):
    return True


@Project.label
def initialized(job):
    return os.path.isfile(job.fn("init.gsd"))


@Project.post(initialized)
@Project.operation(directives={"walltime": 1, "nranks": 1})
def initialize(job):
    """Initialize a system of rigid monomers randomly distributed in a cubic box."""

    box_length = job.sp.box_L
    monomer_counts = job.sp.monomer_counts
    total_monomers = sum(monomer_counts.values())

    r = job.sp.r
    a = job.sp.a
    b = job.sp.b

    typesA = ["AP1", "AP1", "AP1", "AM", "AM", "AM", "AP2", "AP2", "AP2"]
    typesB = ["BP1", "BP1", "BP1", "BM", "BM", "BM", "BP2", "BP2", "BP2"]
    typesC = ["CP1", "CP1", "CP1", "CM", "CM", "CM", "CP2", "CP2", "CP2"]

    monomer_positions = np.array(
        [
            [-a, 0.0, b],
            [-a, b * np.cos(np.pi / 6.0), -b * np.sin(np.pi / 6.0)],
            [-a, -b * np.cos(np.pi / 6.0), -b * np.sin(np.pi / 6.0)],
            [0.0, 0.0, a],
            [0.0, a * np.cos(np.pi / 6.0), -a * np.sin(np.pi / 6.0)],
            [0.0, -a * np.cos(np.pi / 6.0), -a * np.sin(np.pi / 6.0)],
            [a, 0.0, b],
            [a, b * np.cos(np.pi / 6.0), -b * np.sin(np.pi / 6.0)],
            [a, -b * np.cos(np.pi / 6.0), -b * np.sin(np.pi / 6.0)],
        ],
        dtype=np.float64,
    )

    monomer_centers = []
    orientations = []
    monomer_type_ids = []
    type_name_to_id_mapping = {"A": 0, "B": 1, "C": 2}

    def is_valid_monomer_position(new_cm_pos):
        """Ensure that monomer centers stay inside the box and do not overlap."""
        min_distance = 1.5
        buffer = max(a, b)
        lower_bound, upper_bound = -box_length / 2 + buffer, box_length / 2 - buffer

        if not (
            lower_bound <= new_cm_pos[0] <= upper_bound
            and lower_bound <= new_cm_pos[1] <= upper_bound
            and lower_bound <= new_cm_pos[2] <= upper_bound
        ):
            return False  # Prevent placement near boundaries

        return all(
            np.linalg.norm(new_cm_pos - cm_pos) > min_distance
            for cm_pos in monomer_centers
        )

    for monomer_type, count in monomer_counts.items():
        for _ in range(count):
            while True:
                new_cm_pos = np.random.uniform(-box_length / 2, box_length / 2, size=3)
                if is_valid_monomer_position(new_cm_pos):
                    monomer_centers.append(new_cm_pos)
                    orientations.append(
                        Rotation.random().as_quat(scalar_first=True)
                    )  # Scalar first to go along with proper HOOMD convention.
                    monomer_type_ids.append(type_name_to_id_mapping[monomer_type])
                    break

    monomer_centers = np.array(monomer_centers)
    orientations = np.array(orientations)

    expected_particles = total_monomers
    assert (
        len(monomer_type_ids) == expected_particles
    ), f"Shape mismatch! Expected {expected_particles}, got {len(monomer_type_ids)}"

    device = hoomd.device.GPU()
    simulation = hoomd.Simulation(device=device, seed=job.sp.seed)

    snapshot = hoomd.Snapshot()

    snapshot.configuration.box = [box_length, box_length, box_length, 0, 0, 0]

    snapshot.particles.types = [
        "A",
        "B",
        "C",
        "AP1",
        "AM",
        "AP2",
        "BP1",
        "BM",
        "BP2",
        "CP1",
        "CM",
        "CP2",
    ]

    snapshot.particles.N = total_monomers  # Ensure HOOMD knows how many particles exist

    snapshot.particles.typeid[:] = monomer_type_ids

    snapshot.particles.position[:] = monomer_centers

    snapshot.particles.orientation[:] = orientations

    MOI = np.zeros(3)

    for i, pos in enumerate(monomer_positions):
        mass = 1.0 if i in [3, 4, 5] else 0.2
        MOI[0] += mass * (
            pos[1] ** 2 + pos[2] ** 2
        )  # I_xx = Sum(m_i * (y_i^2 + z_i^2))
        MOI[1] += mass * (
            pos[0] ** 2 + pos[2] ** 2
        )  # I_yy = Sum(m_i * (x_i^2 + z_i^2))
        MOI[2] += mass * (
            pos[0] ** 2 + pos[1] ** 2
        )  # I_zz = Sum(m_i * (x_i^2 + y_i^2))

    moi_core = MOI.tolist()

    # moi_map = {
    #     'AP1': [0, 0, 0], 'AP2': [0, 0, 0], 'AM': moi_core,
    #     'BP1': [0, 0, 0], 'BP2': [0, 0, 0], 'BM': moi_core,
    #     'CP1': [0, 0, 0], 'CP2': [0, 0, 0], 'CM': moi_core
    # }

    snapshot.particles.moment_inertia[:] = np.array([moi_core] * total_monomers)

    # diameter_map = {
    #     'AP1': 2 * b, 'AP2': 2 * b, 'AM': 2 * a,
    #     'BP1': 2 * b, 'BP2': 2 * b, 'BM': 2 * a,
    #     'CP1': 2 * b, 'CP2': 2 * b, 'CM': 2 * a
    # }
    #
    # snapshot.particles.diameter[:] = np.array([
    #     diameter_map[type_id_to_name[t]] for t in expanded_type_ids
    # ]) # Can play with this later, but this only affects things for visualization purposes.

    snapshot.particles.charge[:] = np.array([0.0]) * total_monomers

    rigid = md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": typesA,
        "positions": monomer_positions,
        "orientations": np.array([[1, 0, 0, 0]] * 9, dtype=np.float64),
    }
    rigid.body["B"] = {
        "constituent_types": typesB,
        "positions": monomer_positions,
        "orientations": np.array([[1, 0, 0, 0]] * 9, dtype=np.float64),
    }
    rigid.body["C"] = {
        "constituent_types": typesC,
        "positions": monomer_positions,
        "orientations": np.array([[1, 0, 0, 0]] * 9, dtype=np.float64),
    }

    simulation.create_state_from_snapshot(snapshot)
    rigid.create_bodies(
        simulation.state
    )  # Adds consituents body particles to the system.

    np.save(job.fn("positions.npy"), monomer_positions)
    np.save(job.fn("orientations.npy"), np.array([[1, 0, 0, 0]] * 9))
    np.save(job.fn("types.npy"), np.array([typesA, typesB, typesC]))

    print(
        f"Initialized system with {total_monomers} rigid bodies in a {box_length}³ box."
    )
    nl = hoomd.md.nlist.Cell(buffer=0, exclusions=["body"])
    morse = md.pair.Morse(default_r_cut=job.sp.r_cut, nlist=nl)
    table = md.pair.Table(nlist=nl, default_r_cut=job.sp.rep_r_cut)

    def smooth_step(r, rmin, rmax, steepness=10):
        x = (r - rmin) / (rmax - rmin)
        return jnp.clip(1 / (1 + jnp.exp(-steepness * (x - 0.5))), 0, 1)

    def repulsive_potential(rmin, rmax, A, alpha):
        def _V(r):
            epsilon = 1e-6
            base = jnp.maximum(rmax - r, epsilon)
            potential = (A / (alpha * rmax)) * base**alpha
            return jnp.where(r < rmax, potential * smooth_step(r, rmin, rmax), 0.0)

        return _V

    repulsive = repulsive_potential(
        rmin=job.sp.rep_r_min,
        rmax=job.sp.rep_r_max,
        A=job.sp.rep_A,
        alpha=job.sp.rep_alpha,
    )
    tabulated_repulsive = (
        np.array(repulsive(np.linspace(0, job.sp.rep_r_cut, 1001))),
        np.array(-1 * vmap(grad(repulsive))(jnp.linspace(0, job.sp.rep_r_cut, 1001))),
    )

    # Use simulation.state.types instead of system.particles.types
    set_pair_potentials_params(
        job, morse, table, tabulated_repulsive, snapshot.particles.types
    )

    """ Options for potential shifting:
    none - No shifting; potentials are cut off abruptly.
    shift - Applies a constant shift so potential is 0 at cutoff.
    xplor - Smooth function applied for a gradual force and potential decrease.
    """
    morse.mode = "shift"

    # Set up integrator
    integrator = md.Integrator(dt=job.sp.dt, integrate_rotational_dof=True)
    simulation.operations.integrator = integrator
    integrator.rigid = rigid

    rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
    md_int = md.methods.ConstantVolume(
        filter=rigid_centers_and_free,
        thermostat=md.methods.thermostats.MTTK(kT=2.0 + job.sp.kT, tau=job.sp.tau),
    )
    integrator.methods.append(md_int)
    integrator.forces.append(morse)
    integrator.forces.append(table)

    # Thermalize system
    simulation.state.thermalize_particle_momenta(
        filter=rigid_centers_and_free, kT=2.0 + job.sp.kT
    )

    # Set up logging
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    simulation.operations.computes.append(thermodynamic_properties)
    logger = hoomd.logging.Logger(categories=["scalar", "sequence"])
    logger.add(simulation, quantities=["timestep", "walltime"])
    logger.add(
        thermodynamic_properties,
        quantities=[
            "pressure",
            "potential_energy",
            "translational_kinetic_energy",
            "rotational_kinetic_energy",
        ],
    )

    hdf5_writer = hoomd.write.HDF5Log(
        trigger=hoomd.trigger.Periodic(int(job.sp.log_period)),
        filename=job.fn("init_log.h5"),
        mode="w",
        logger=logger,
    )
    simulation.operations.writers.append(hdf5_writer)

    # Adjust the system's box if needed
    concentration = job.sp.concentration
    total_N = total_monomers * 9  # No more uc.N * job.sp.Nx * job.sp.Ny * job.sp.Nz
    final_L = (total_N / concentration) ** (1 / 3)

    inverse_volume_ramp = hoomd.variant.box.InverseVolumeRamp(
        initial_box=simulation.state.box,
        final_volume=final_L**3,
        t_start=simulation.timestep,
        t_ramp=int(job.sp.equil_step),
    )
    box_resize = hoomd.update.BoxResize(
        trigger=hoomd.trigger.Periodic(10),
        box=inverse_volume_ramp,
    )
    simulation.operations.updaters.append(box_resize)
    simulation.run(job.sp.equil_step)
    simulation.operations.updaters.remove(box_resize)
    simulation.run(1e6)

    # Save final system state
    hoomd.write.GSD.write(
        state=simulation.state, mode="xb", filename=job.fn("init.gsd")
    )
    hoomd.write.GSD.write(
        state=simulation.state, mode="wb", filename=job.fn("init_pos.gsd")
    )


@Project.label
def dumped(job):
    if job.isfile("dump.gsd"):
        with job:
            with gsd.hoomd.open(name="dump.gsd", mode="r") as gsd_file:
                if len(gsd_file) == 2100:
                    return True
    return False


@Project.pre.after(initialize)
@Project.post(dumped)
@Project.operation(directives={"walltime": 48, "nranks": 1})
def equilibrate(job):
    import hoomd
    from hoomd import md
    from pair_potentials import set_pair_potentials_params

    types = np.load(job.fn("types.npy"), allow_pickle=True).tolist()
    typesA, typesB, typesC = list(types[0]), list(types[1]), list(types[2])

    positions = np.load(job.fn("positions.npy"))
    orientations = np.load(job.fn("orientations.npy"))

    # device = hoomd.device.CPU(notice_level=10) # Run on CPU
    device = hoomd.device.GPU()  # Run on gpu
    simulation = hoomd.Simulation(device=device, seed=job.sp.seed)
    simulation.create_state_from_gsd(filename=job.fn("init.gsd"))

    rigid = md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": typesA,
        "positions": positions,
        "orientations": orientations,
    }
    rigid.body["B"] = {
        "constituent_types": typesB,
        "positions": positions,
        "orientations": orientations,
    }
    rigid.body["C"] = {
        "constituent_types": typesC,
        "positions": positions,
        "orientations": orientations,
    }
    # rigid.create_bodies(simulation.state) # I believe this is redundant, but anyhow...

    # Set up pair potentials
    nl = hoomd.md.nlist.Cell(buffer=0, exclusions=["body"])
    morse = md.pair.Morse(default_r_cut=job.sp.r_cut, nlist=nl)
    table = md.pair.Table(nlist=nl, default_r_cut=job.sp.rep_r_cut)

    def smooth_step(r, rmin, rmax, steepness=10):
        x = (r - rmin) / (rmax - rmin)
        return jnp.clip(1 / (1 + jnp.exp(-steepness * (x - 0.5))), 0, 1)

    def repulsive_potential(rmin, rmax, A, alpha):
        def _V(r):
            epsilon = 1e-6
            base = jnp.maximum(rmax - r, epsilon)
            # smoothing_factor = smooth_step(r, rmin, rmax)
            potential = (A / (alpha * rmax)) * base**alpha
            return jnp.where(r < rmax, potential * smooth_step(r, rmin, rmax), 0.0)

        return _V

    repulsive = repulsive_potential(
        rmin=job.sp.rep_r_min,
        rmax=job.sp.rep_r_max,
        A=job.sp.rep_A,
        alpha=job.sp.rep_alpha,
    )
    tabulated_repulsive = (
        np.array(repulsive(np.linspace(0, job.sp.rep_r_cut, 1001))),
        np.array(-1 * vmap(grad(repulsive))(jnp.linspace(0, job.sp.rep_r_cut, 1001))),
    )

    set_pair_potentials_params(
        job, morse, table, tabulated_repulsive, simulation.state.types["particle_types"]
    )

    """ Options here are the following:
    none - No shifting is performed and potentials are abruptly cut off
    shift - A constant shift is applied to the entire potential so that it is 0 at the cutoff
    xplor - A smoothing function is applied to gradually decrease both the force and potential to 0 at the cutoff when ron < rcut, and shifts the potential to 0 at the cutoff when ron >= rcut.
    """
    morse.mode = "shift"

    integrator = md.Integrator(dt=job.sp.dt, integrate_rotational_dof=True)
    simulation.operations.integrator = integrator
    integrator.rigid = rigid

    rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
    md_int = md.methods.ConstantVolume(
        filter=rigid_centers_and_free,
        thermostat=md.methods.thermostats.MTTK(kT=2.0 + job.sp.kT, tau=job.sp.tau),
    )
    integrator.methods.append(md_int)
    integrator.forces.append(morse)
    integrator.forces.append(table)

    simulation.state.thermalize_particle_momenta(
        filter=rigid_centers_and_free, kT=2.0 + job.sp.kT
    )

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    simulation.operations.computes.append(thermodynamic_properties)
    logger = hoomd.logging.Logger(categories=["scalar", "sequence"])
    logger.add(simulation, quantities=["timestep", "walltime"])
    logger.add(
        thermodynamic_properties,
        quantities=[
            "pressure",
            "potential_energy",
            "translational_kinetic_energy",
            "rotational_kinetic_energy",
        ],
    )
    hdf5_writer = hoomd.write.HDF5Log(
        trigger=hoomd.trigger.Periodic(int(job.sp.log_period)),
        filename=job.fn("dump_log.h5"),
        mode="x",
        logger=logger,
    )
    simulation.operations.writers.append(hdf5_writer)

    # pos = deprecated.dump.pos(filename = job.fn('dump.pos'), unwrap_rigid = True, period = job.sp.dump_period*10)
    # pos.set_def('A', 'sphere 1 a6a6a6')
    # pos.set_def('D', 'dipole 1 0 b31b1b f7f7f7')
    # pos.set_def('B', 'sphere 1 a6a6a6')

    gsd = hoomd.write.GSD(filename=job.fn("dump.gsd"), trigger=int(job.sp.dump_period))
    simulation.operations.writers.append(gsd)

    for kT in np.arange(2.0 + job.sp.kT, job.sp.kT, -0.1):
        md_int.thermostat.kT = kT
        simulation.run(5e5)

    md_int.thermostat.kT = job.sp.kT
    simulation.run(job.sp.run_step)


if __name__ == "__main__":
    Project().main()
