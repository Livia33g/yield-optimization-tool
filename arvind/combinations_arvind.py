import itertools
import pickle
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor

# Define the types of monomers 
def create_monomers(num_monomers):
    monomers = {}
    for i in range(num_monomers):
        letter = chr(ord('A') + i)
        monomers[letter] = [2 * i + 1, 0, 2 * i + 2]
    return monomers

# Check if a combination meets the even-odd condition
def valid_even_odd_sequence(numeric_combination):
    for i in range(1, len(numeric_combination) - 1):
        if numeric_combination[i] != 0 and numeric_combination[i + 1] != 0:
            if numeric_combination[i] % 2 == numeric_combination[i + 1] % 2:
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
        numeric_combination = sum([all_monomers[name] for name in comb], [])
        if valid_even_odd_sequence(numeric_combination):
            mirrored_comb = tuple([k + "'" if k[-1] != "'" else k[:-1] for k in comb][::-1])
            if mirrored_comb not in seen:
                unique_combinations.append(comb)
                seen.add(comb)
    return unique_combinations, all_monomers

def combination_to_string(comb):
    return ' '.join(comb)

def get_numeric_combination(comb_str, all_monomers):
    monomer_names = comb_str.split()
    numeric_combination = sum([all_monomers[name] for name in monomer_names], [])
    return numeric_combination

# Count the occurrences of each monomer in the combinations
def count_monomers(combinations, monomer_name):
    monomer_counts = []
    for comb in combinations:
        count = sum(1 for mon in comb if mon == monomer_name or mon == f"{monomer_name}'")
        monomer_counts.append(count)
    return monomer_counts


def process_combinations(unique_combinations, all_monomers, monomers, max_struc_size):
    species_dict = {}
    counts = {}
    
    for size in range(1, max_struc_size + 1):
        structure_key = f'{size}_pc_species'
        current_combinations = [comb for comb in unique_combinations if len(comb) == size]
        species_dict[structure_key] = jnp.array([get_numeric_combination(combination_to_string(comb), all_monomers) for comb in current_combinations])
        
        
        for letter in monomers.keys():
            count_key = f'{letter}_{size}_counts'
            counts[count_key] = jnp.array(count_monomers(current_combinations, letter))
    
    return {
        **species_dict,
        **counts
    }

def save_results(data):
    with open('arvind_4.pkl', 'wb') as f:
        pickle.dump(data, f)

def main(num_monomers, max_struc_size):
    monomers = create_monomers(num_monomers)
    unique_combinations, all_monomers = generate_combinations(monomers, max_struc_size)
    
    with ThreadPoolExecutor() as executor:
        future = executor.submit(process_combinations, unique_combinations, all_monomers, monomers, max_struc_size)
        result = future.result()
    
    save_results(result)

    
    print("Species combinations and counts saved successfully.")
    print(sum(len(result[f'{i}_pc_species']) for i in range(1, max_struc_size + 1)))

if __name__ == "__main__":
    num_monomers = 4  # Fixme to the number of monomers
    max_struc_size = 4  # Fixme to the max size structure in the ensemble 
    main(num_monomers, max_struc_size)
