import pickle
import jax.numpy as jnp


def extract_polymer_info(input_file, output_file, min_polymer=4, max_polymer=6):
    """
    Extract polymer information (from min_polymer to max_polymer) from the input file and save it to a separate file.

    Parameters:
        input_file (str): Path to the input .pkl file containing species information.
        output_file (str): Path to save the extracted polymer information.
        min_polymer (int): Minimum polymer size to extract (inclusive).
        max_polymer (int): Maximum polymer size to extract (inclusive).
    """
    # Load the existing data
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    polymer_data = {}

    # Iterate over the range of polymer sizes
    for polymer_size in range(min_polymer, max_polymer + 1):
        species_key = f"{polymer_size}_pc_species"
        sigma_key = f"{polymer_size}_sigma"
        amount_structures_key = f"{polymer_size}_amount_structures" # Key for amount of structures

        # Extract species information
        if species_key in data:
            polymer_data[species_key] = data[species_key]
        else:
            raise KeyError(f"{species_key} not found in the input file.")

        # Extract sigma information
        if sigma_key in data:
            polymer_data[sigma_key] = data[sigma_key]
        else:
            raise KeyError(f"{sigma_key} not found in the input file.")

        # Extract monomer counts for the current polymer size
        count_keys = [
            key for key in data.keys() if key.endswith(f"_{polymer_size}_counts")
        ]
        for key in count_keys:
            polymer_data[key] = data[key]

        # Extract amount_structures information
        if amount_structures_key in data: # Extract amount_structures
            polymer_data[amount_structures_key] = data[amount_structures_key]
        else:
            raise KeyError(f"{amount_structures_key} not found in the input file.")


    # Save the polymer-specific information
    with open(output_file, "wb") as f:
        pickle.dump(polymer_data, f)

    print(
        f"Polymer information (sizes {min_polymer}-{max_polymer}) saved to {output_file}."
    )


if __name__ == "__main__":
    input_file = "arvind_63.pkl"  # Path to the file containing all species data
    output_file = "polymer_extracted.pkl"  # File to save polymer-specific data
    extract_polymer_info(input_file, output_file, min_polymer=4, max_polymer=6)