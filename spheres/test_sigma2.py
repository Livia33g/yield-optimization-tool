import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_pos_file(file_path):
    vertices, patches = [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] == 'V':
                vertices.append(list(map(float, parts[1:])))
            elif parts[0] == 'P':
                patches.append(list(map(float, parts[1:])))
    return np.array(vertices), np.array(patches)

def center_of_mass(coords):
    return np.mean(coords, axis=0)

def translate_to_center(coords, center):
    return coords - center

def align_to_principal_axes(coords):
    cov_matrix = np.cov(coords.T)
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    return coords @ eigenvectors

def normalize_coordinates(coords):
    max_norm = np.linalg.norm(coords, axis=1).max()
    return coords / max_norm

def are_structures_equal(coords1, coords2, rtol=1e-3, atol=0.02):
    scale_factor = np.max(np.linalg.norm(coords1, axis=1))
    atol = atol * scale_factor
    rtol = rtol * scale_factor
    return np.allclose(
        np.sort(coords1, axis=0),
        np.sort(coords2, axis=0),
        rtol=rtol,
        atol=atol
    )

def estimate_symmetries_monte_carlo(vertices, num_samples):
    """
    Monte Carlo simulation to estimate the number of symmetries.
    
    Args:
        vertices (np.ndarray): Array of vertex coordinates for the structure.
        num_samples (int): Number of random rotations to sample.

    Returns:
        float: Estimated number of unique symmetries.
    """
    successful_matches = 0  # Count of rotations that leave the structure unchanged
    random_rotations = R.random(num_samples)  # Generate random rotations
    
    for rotation in random_rotations:
        rotated_vertices = rotation.apply(vertices)
        if are_structures_equal(vertices, rotated_vertices):
            successful_matches += 1

    # Estimate total number of unique symmetries
    symmetry_fraction = successful_matches / num_samples
    estimated_symmetries = symmetry_fraction * num_samples  # Scaled estimate

    return estimated_symmetries, symmetry_fraction

# Main logic
file_path = 'shell_5.pos'
vertices, patches = parse_pos_file(file_path)

# Preprocessing
com = center_of_mass(vertices)
vertices = translate_to_center(vertices, com)
vertices = align_to_principal_axes(vertices)
vertices = normalize_coordinates(vertices)

# Monte Carlo simulation
num_samples = 1000000  # Number of random rotations to sample
estimated_symmetries, symmetry_fraction = estimate_symmetries_monte_carlo(vertices, num_samples)

# Output the results
print(f"\nEstimated number of symmetries: {estimated_symmetries:.2f}")
print(f"Fraction of successful matches: {symmetry_fraction:.4f}")
if estimated_symmetries >= 59 and estimated_symmetries <= 61:
    print("Close to the expected number of Icosahedral symmetries (60)!")
else:
    print("Estimate may be off. Consider increasing the number of samples.")
