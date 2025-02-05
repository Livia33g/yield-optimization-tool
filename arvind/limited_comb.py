import itertools
import pickle
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor


# Define the types of monomers
def create_monomers(num_monomers):  # num_monomers is not really used now
    monomers = {"A": [0, 2], "B": [3, 0, 4], "C": [5, 0]}
    return monomers


# Check if a combination meets the even-odd condition
def valid_even_odd_sequence(monomer_combination_list):
    if len(monomer_combination_list) <= 1:
        return True  # Single monomer is always valid

    for i in range(len(monomer_combination_list) - 1):
        current_monomer_patches = monomer_combination_list[i]
        next_monomer_patches = monomer_combination_list[i + 1]

        last_patch_current = current_monomer_patches[-1]
        first_patch_next = next_monomer_patches[0]

        if last_patch_current == 0 or first_patch_next == 0:
            return False  # 0 cannot attach to anything

        if last_patch_current != 0 and first_patch_next != 0:
            if last_patch_current % 2 == first_patch_next % 2:
                return False
    return True


# Generate unique combinations of monomers of all sizes up to max_struc_size
def generate_combinations(monomers, max_struc_size):
    monomers_prime = {f"{k}'": v[::-1] for k, v in monomers.items()}
    all_monomers = {**monomers, **monomers_prime}

    all_combinations = []
    for r in range(1, max_struc_size + 1):
        all_combinations.extend(itertools.product(all_monomers.keys(), repeat=r))

    unique_combinations = []
    seen = set()
    for comb in all_combinations:
        numeric_combination_list = get_numeric_combination_list(
            combination_to_string(comb), all_monomers
        )  # Get list of lists
        if valid_even_odd_sequence(numeric_combination_list):  # Pass list of lists
            mirrored_comb = tuple(
                [k + "'" if k[-1] != "'" else k[:-1] for k in comb][::-1]
            )
            if mirrored_comb not in seen:
                unique_combinations.append(comb)
                seen.add(comb)
    return unique_combinations, all_monomers


def combination_to_string(comb):
    return " ".join(comb)


def get_numeric_combination_list(comb_str, all_monomers):  # Returns list of lists
    monomer_names = comb_str.split()
    numeric_combination_list = [all_monomers[name] for name in monomer_names]
    return numeric_combination_list


def get_numeric_combination(
    comb_str, all_monomers
):  # Returns flattened list (original format)
    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])
    return numeric_combination


# Count the occurrences of each monomer in the combinations
def count_monomers(combinations, monomer_name):
    monomer_counts = []
    for comb in combinations:
        count = sum(
            1 for mon in comb if mon == monomer_name or mon == f"{monomer_name}'"
        )
        monomer_counts.append(count)
    return monomer_counts


# Calculate sigma based on structure size
def calculate_sigma(size):
    return 3 ** (size)


# Process combinations and calculate sigma for each size
def process_combinations(unique_combinations, all_monomers, monomers, max_struc_size):
    species_dict = {}
    counts = {}
    sigmas = {}

    for size in range(1, max_struc_size + 1):
        structure_key = f"{size}_pc_species"
        current_combinations = [
            comb for comb in unique_combinations if len(comb) == size
        ]

        # Store species (as list of lists - NOT jnp.array)
        species_dict[structure_key] = [  # Removed jnp.array conversion
            get_numeric_combination(combination_to_string(comb), all_monomers)
            for comb in current_combinations
        ]

        # Store monomer counts
        for letter in monomers.keys():
            count_key = f"{letter}_{size}_counts"
            counts[count_key] = jnp.array(count_monomers(current_combinations, letter))

        # Store sigma value
        sigma_key = f"{size}_sigma"
        sigma_value = calculate_sigma(size)
        sigmas[sigma_key] = sigma_value
        print(f"Sigma for size {size}: {sigma_value}")

    return {**species_dict, **counts, **sigmas}


def save_results(data):
    with open("limited.pkl", "wb") as f:
        pickle.dump(data, f)


def main(num_monomers, max_struc_size):
    monomers = create_monomers(num_monomers)
    unique_combinations, all_monomers = generate_combinations(monomers, max_struc_size)

    print("Unique Combinations:")
    for comb in unique_combinations:
        print(comb)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            process_combinations,
            unique_combinations,
            all_monomers,
            monomers,
            max_struc_size,
        )
        result = future.result()

    save_results(result)

    print("Species combinations, counts, and sigma values saved successfully.")
    print(sum(len(result[f"{i}_pc_species"]) for i in range(1, max_struc_size + 1)))


if __name__ == "__main__":
    num_monomers = 3  # Adjust the number of monomers
    max_struc_size = 3  # Adjust the max size structure in the ensemble
    main(num_monomers, max_struc_size)
