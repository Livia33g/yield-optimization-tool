import numpy as np
from scipy.spatial.transform import Rotation as R
import os


def compute_center_of_mass(coordinates):
    return np.mean(coordinates, axis=0)


def generate_octahedral_rotations():
    """Generate rotation matrices for the octahedral (O) group — 24 rotations."""
    return R.create_group("O").as_matrix()


def validate_unique_rotations(rotations):
    unique_rotations = set()
    for rotation in rotations:
        key = tuple(np.round(rotation.flatten(), decimals=6))
        unique_rotations.add(key)
    return len(unique_rotations), unique_rotations


def align_coordinates(coords):
    return np.round(np.sort(coords, axis=0), decimals=6)


def compute_symmetry_number(coordinates, rotations, atol=0.558670741):
    com = compute_center_of_mass(coordinates)
    centered_coords = coordinates - com
    aligned_original = align_coordinates(centered_coords)

    symmetry_count = 0
    for rotation in rotations:
        rotated_coords = centered_coords @ rotation.T
        aligned_rotated = align_coordinates(rotated_coords)
        if np.allclose(aligned_rotated, aligned_original, atol=atol):
            symmetry_count += 1

    return symmetry_count


def load_coordinates(file_path):
    coordinates = []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip() and not line.startswith(("boxMatrix", "def", "eof")):
                parts = line.split()
                if len(parts) == 4 and parts[0] in ("V", "P"):
                    coordinates.append(list(map(float, parts[1:])))
    return np.array(coordinates)


if __name__ == "__main__":
    pos_files_folder = (
        "oct_files"  # folder with .pos files from your octahedron subsets
    )
    output_file = "symmetry_numbers_oct.txt"

    rotations = generate_octahedral_rotations()

    with open(output_file, "w") as f:
        f.write("File Name,Symmetry Number\n")
        for pos_file in sorted(os.listdir(pos_files_folder)):
            if not pos_file.endswith(".pos"):
                continue
            file_path = os.path.join(pos_files_folder, pos_file)
            coords = load_coordinates(file_path)
            if coords.size == 0:
                f.write(f"{pos_file},No vertices found\n")
                continue
            symmetry_number = compute_symmetry_number(coords, rotations)
            print(f"{pos_file}: Symmetry Number = {symmetry_number}")
            f.write(f"{pos_file},{symmetry_number}\n")
