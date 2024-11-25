import jax.numpy as jnp
from jax import vmap
import jax_transformations3d as jts

euler_scheme = "sxyz"


def convert_to_matrix(mi):
    if len(mi) < 6:
        raise ValueError(f"Input mi must have at least 6 elements, but got {len(mi)}")
    T = jts.translation_matrix(mi[:3])
    R = jts.euler_matrix(mi[3], mi[4], mi[5], axes=euler_scheme)
    return jnp.matmul(T, R)


def get_positions(q, ppos):
    Mat = []
    for i in range(len(ppos)):
        qi = i * 6
        Mat.append(convert_to_matrix(q[qi : qi + 6]))

    real_ppos = []
    for i, mat in enumerate(Mat):
        real_ppos.append(jts.matrix_apply(mat, ppos[i]))

    real_ppos = jnp.array(real_ppos)
    real_ppos = real_ppos.reshape(-1, 3)

    return real_ppos


def convert_to_matrix_quat(mi):
    """
    Convert a 6-element vector to a transformation matrix.
    The first 3 elements are the translation vector, and the last 4 elements are the quaternion.

    Args:
        mi (jnp.ndarray): Array of length 7 where:
                          mi[:3] is the translation vector.
                          mi[3:] is the quaternion (w, x, y, z).
    Returns:
        jnp.ndarray: 4x4 transformation matrix.
    """
    if len(mi) != 7:
        raise ValueError(
            f"Input mi must have 7 elements (3 for translation, 4 for quaternion), but got {len(mi)}"
        )

    T = jts.translation_matrix(mi[:3])  # Translation matrix
    R = jts.quaternion_matrix(mi[3:])  # Rotation matrix from quaternion
    return jnp.matmul(T, R)


def get_positions_quat(q, ppos):
    """
    Apply transformations defined by the input `q` (translations + quaternions)
    to a set of points `ppos`.

    Args:
        q (jnp.ndarray): Array of shape (n * 7,) where every 7 elements represent:
                         - 3 elements for translation.
                         - 4 elements for quaternion (w, x, y, z).
        ppos (jnp.ndarray): Array of shape (n, m, 3) representing `m` points for each of `n` rigid bodies.

    Returns:
        jnp.ndarray: Transformed positions of shape (n * m, 3).
    """
    Mat = []
    for i in range(len(ppos)):
        qi = i * 7  # Every transformation has 7 elements
        Mat.append(convert_to_matrix_quat(q[qi : qi + 7]))

    real_ppos = []
    for i, mat in enumerate(Mat):
        real_ppos.append(
            jts.matrix_apply(mat, ppos[i])
        )  # Apply the transformation matrix to points

    real_ppos = jnp.array(real_ppos).reshape(-1, 3)
    return real_ppos


def add_variables(ma, mb):
    Ma = convert_to_matrix(ma)
    Mb = convert_to_matrix(mb)
    Mab = jnp.matmul(Mb, Ma)
    trans = jnp.array(jts.translation_from_matrix(Mab))
    angles = jnp.array(jts.euler_from_matrix(Mab, euler_scheme))
    return jnp.concatenate((trans, angles))


def add_variables_all(mas, mbs):
    mas_temp = jnp.reshape(mas, (mas.shape[0] // 6, 6))
    mbs_temp = jnp.reshape(mbs, (mbs.shape[0] // 6, 6))
    return jnp.reshape(
        vmap(add_variables, in_axes=(0, 0))(mas_temp, mbs_temp), mas.shape
    )
