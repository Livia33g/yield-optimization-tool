import signac
import numpy as np
import random
import math

project = signac.init_project()

# System parameters
replicas = [1, 2, 3]
alpha = 5.0

# Target total concentration
target_total_concentration = 0.001

# File containing simulation parameters
parameter_file = "all_params.txt"

simulation_parameters = []

with open(parameter_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        values = line.split(",")
        try:
            target_yield = float(values[1])
            # Extract d_types as values[3] through values[17] (15 values)
            d_types = [float(v) for v in values[3:-4]]
            kT = float(values[-4])
            conc_A = float(values[-3])
            conc_B = float(values[-2])
            conc_C = float(values[-1])
        except ValueError:
            print(f"Warning: Skipping line with invalid numeric values: {line}")
            continue

        monomer_concentrations = {"A": conc_A, "B": conc_B, "C": conc_C}

        simulation_parameters.append(
            (target_yield, d_types, monomer_concentrations, kT)
        )

# Step 1: Initial guesses
initial_total_volume = 300000  # Initial estimate of system volume
initial_side_length = 12.0
max_iterations = 100
tolerance = 0.00005


def refine_system(volume, target_concentration, monomer_concentrations):
    """Adjust the number of particles and system volume to match concentration constraints."""
    total_monomers = sum(monomer_concentrations.values())
    computed_concentration = total_monomers / volume
    return total_monomers, volume, computed_concentration


# Iterate over parameters and create Signac jobs
for (target_yield, d_types, monomer_concentrations, kT) in simulation_parameters:
    for replica in replicas:
        # Adjust monomer counts to match concentration constraints
        target_total_concentration = sum(monomer_concentrations.values())
        base_concentration = min(monomer_concentrations.values())
        scaled_ratios = {
            k: max(1, round(v / base_concentration))
            for k, v in monomer_concentrations.items()
        }

        # Scale to meet the approximate total concentration
        scaling_factor = target_total_concentration / sum(monomer_concentrations.values())
        monomer_counts = {
            k: max(1, round(v * scaling_factor * initial_total_volume))
            for k, v in monomer_concentrations.items()
        }

        # Iteratively refine box size and particle count
        current_volume = initial_total_volume
        for _ in range(max_iterations):
            total_monomers, adjusted_volume, computed_concentration = refine_system(
                current_volume, target_total_concentration, monomer_counts
            )
            if abs(computed_concentration - target_total_concentration) <= tolerance:
                break
            current_volume *= target_total_concentration / computed_concentration

        # Compute final box size
        final_side_length = round(adjusted_volume ** (1 / 3), 2)

        # Debugging output
        print(f"Final monomer counts: {monomer_counts}")
        print(f"Final box size: {final_side_length:.2f}³")
        print(f"Final total concentration: {computed_concentration:.6f}")
        print(f"Temperature (kT): {kT}")
        print(f"d_types (potentials from values[3:18]): {d_types}")

        # Pass parameters to Signac jobs
        sp = {
            # System Information
            "box_L": final_side_length,  # Box side length
            "seed": random.randint(1, 65535),  # Random seed for reproducibility
            "replica": replica,  # Replica ID
            "equil_step": 5e4,  # Equilibration steps
            "concentration": target_total_concentration,  # Target total concentration
            "monomer_counts": monomer_counts,  # Number of each monomer type
            "scale": 0.99,
            # Simulation Setup
            "kT": kT,  # Temperature parameter
            "a": 1.0,
            "b": 0.1,
            "r": 1.1,
            "separation": 2.0,
            "alpha": alpha,
            "r0": 0.0,
            "r_cut": 8.0 / alpha,
            "rep_A": 500.0,
            "rep_alpha": 2.5,
            "rep_r_min": 0.0,
            "rep_r_max": 2.0,
            "rep_r_cut": 6,
            "dt": 0.001,
            "tau": 1.0,
            "run_step": 2e8,
            "dump_period": 1e5,
            "log_period": 1e4,
            # Interaction Parameters
            "D0": 1.0,
            "d_types": d_types,  # Now contains the 15 potential values from the file.
            "target_yield": target_yield,
        }
        job = project.open_job(sp)
        job.init()
