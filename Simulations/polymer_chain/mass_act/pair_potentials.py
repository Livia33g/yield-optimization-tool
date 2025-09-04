import numpy as np


def set_pair_potentials_params(job, morse, table, table_func, types):
    # set Morse potential parameters

# Mapping: 1→AP1 (index 3), 2→AP2 (index 5), 3→BP1 (index 6),
# 4→BP2 (index 8), 5→CP1 (index 9), 6→CP2 (index 11)
    mapping = {1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 11}

# The ordered list of pairs (using your numbering)
    d_mores_pairs = [
        (2, 3), (4, 5),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 4), (2, 5), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 6),
        (5, 6),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)
    ]

    # Assume job.sp.d_types is a sequence (e.g., list or np.array) of potentials
    # with length equal to len(d_mores_pairs).
    # Initialize the potential matrix for 12 types.
    morse_d0s = np.zeros((12, 12))

    # Assign potentials in order according to the pairs list.
    for (a, b), potential in zip(d_mores_pairs, job.sp.d_types):
        i, j = mapping[a], mapping[b]
        # Make sure we fill the lower triangle (i >= j) for later consistency.
        if i < j:
            i, j = j, i
        morse_d0s[i, j] = job.sp.D0 * potential

    # Then set the Morse potential parameters using the populated matrix.
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                morse.params[(type_i, type_j)] = dict(
                    alpha=job.sp.alpha,
                    D0=morse_d0s[i, j],
                    r0=job.sp.r0
                )


                # ['A', 'B', 'C', 'AP1', 'AM', 'AP2', 'BP1', 'BM', 'BP2', 'CP1', 'CM', 'CP2']
    morse_r_cuts = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        ]
    )

    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                if morse_r_cuts[i, j] == 0.0:
                    morse.r_cut[(type_i, type_j)] = 0

    # Set table parameters
    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                table.params[(type_i, type_j)] = dict(
                    r_min=0, U=table_func[0], F=table_func[1]
                )

                # ['A', 'B', 'C', 'AP1', 'AM', 'AP2', 'BP1', 'BM', 'BP2', 'CP1', 'CM', 'CP2']
    table_r_cuts = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    for i, type_i in enumerate(types):
        for j, type_j in enumerate(types):
            if i >= j:
                if table_r_cuts[i, j] == 0.0:
                    table.params[(type_i, type_j)] = dict(r_min=0, U=[0], F=[0])
                    table.r_cut[(type_i, type_j)] = 0
