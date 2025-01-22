import pickle
import jax.numpy as jnp


def extract_tetramer_info(input_file, output_file):
    """
    Extract tetramer information from the input file and save it to a separate file.

    Parameters:
        input_file (str): Path to the input .pkl file containing species information.
        output_file (str): Path to save the extracted tetramer information.
    """
    # Load the existing data
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    # Extract tetramer-specific information
    tetramer_species_key = "4_pc_species"
    tetramer_sigma_key = "4_sigma"
    tetramer_data = {}

    if tetramer_species_key in data:
        tetramer_data[tetramer_species_key] = data[tetramer_species_key]
    else:
        raise KeyError(f"{tetramer_species_key} not found in the input file.")

    if tetramer_sigma_key in data:
        tetramer_data[tetramer_sigma_key] = data[tetramer_sigma_key]
    else:
        raise KeyError(f"{tetramer_sigma_key} not found in the input file.")

    # Extract monomer counts for tetramers
    monomer_keys = [key for key in data.keys() if key.endswith("_4_counts")]
    for key in monomer_keys:
        tetramer_data[key] = data[key]

    # Save the tetramer-specific information
    with open(output_file, "wb") as f:
        pickle.dump(tetramer_data, f)

    print(f"Tetramer information saved to {output_file}.")


if __name__ == "__main__":
    input_file = "arvind_43.pkl"  # Path to the file containing all species data
    output_file = "tetramers.pkl"  # File to save tetramer-specific data
    extract_tetramer_info(input_file, output_file)



