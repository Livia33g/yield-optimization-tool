import numpy as np
from scipy.spatial.transform import Rotation as R 
import os

def compute_center_of_mass(coordinates):
    """Compute the center of mass for the structure."""
    return np.mean(coordinates, axis=0)

def generate_icosahedral_rotations():
    """Generate rotation matrices for the full icosahedral symmetry group. this holds all 60 rotations."""
    from scipy.spatial.transform import Rotation as R

    icosahedral_group = R.create_group('I')  # 'I' stands for icosahedral group
    return icosahedral_group.as_matrix()


def validate_unique_rotations(rotations):
    """Ensure the rotation group forms a complete set of unique rotations. (each generated matrix is unique)"""
    unique_rotations = set()
    for rotation in rotations:
        key = tuple(np.round(rotation.flatten(), decimals=6)) 
        unique_rotations.add(key)
    return len(unique_rotations), unique_rotations

def align_coordinates(coords):
    """Align coordinates to a canonical form for comparison."""
    return np.round(np.sort(coords, axis=0), decimals=6)

def compute_symmetry_number(coordinates, rotations, atol=0.53):
    """Compute the symmetry number of the structure based on immutability under rotations."""
    com = compute_center_of_mass(coordinates)
    centered_coords = coordinates - com

    symmetry_count = 0
    aligned_original = align_coordinates(centered_coords)

    for rotation in rotations:
        rotated_coords = centered_coords @ rotation.T
        aligned_rotated = align_coordinates(rotated_coords)

        if np.allclose(aligned_rotated, aligned_original, atol=atol):
            symmetry_count += 1

    return symmetry_count

def load_coordinates(file_path):
    """Load coordinates from a .pos file."""
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith(('boxMatrix', 'def', 'eof')):
                parts = line.split()
                if len(parts) == 4 and parts[0] in ('V', 'P'):
                    coordinates.append(list(map(float, parts[1:])))
    return np.array(coordinates)

if __name__ == "__main__":
    pos_files_folder = "pos_files"  # Update with your folder path
    output_file = "symmetry_numbers.txt"  # File to save the results

    rotations = generate_icosahedral_rotations()

    # Open the output file for writing
    with open(output_file, "w") as f:
        f.write("File Name,Symmetry Number\n")  # Header line

        for pos_file in os.listdir(pos_files_folder):
            file_path = os.path.join(pos_files_folder, pos_file)
            coords = load_coordinates(file_path)
            if coords.size == 0:
                f.write(f"{pos_file},No vertices found\n")
                continue

            symmetry_number = compute_symmetry_number(coords, rotations)
            print(f"{pos_file}: Symmetry Number = {symmetry_number}")
            # Write the result to the file
            f.write(f"{pos_file},{symmetry_number}\n")
