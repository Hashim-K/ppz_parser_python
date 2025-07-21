import numpy as np
import pandas as pd


def eulers_of_quat(
    quat_df: pd.DataFrame,
    q_w: str = "qi",
    q_x: str = "qx",
    q_y: str = "qy",
    q_z: str = "qz",
) -> pd.DataFrame:
    """
    Converts a DataFrame with quaternion columns into ZYX Euler angles.

    Args:
        quat_df: DataFrame containing the quaternion data. Must include a 'timestamp' column.
        q_w: The column name for the scalar component (w).
        q_x: The column name for the x component.
        q_y: The column name for the y component.
        q_z: The column name for the z component.

    Returns:
        A new DataFrame with 'timestamp', 'phi', 'theta', and 'psi' columns.
    """
    # Extract the quaternion components as numpy arrays for efficient calculation.
    w = quat_df[q_w].to_numpy()
    x = quat_df[q_x].to_numpy()
    y = quat_df[q_y].to_numpy()
    z = quat_df[q_z].to_numpy()

    # --- ZYX Euler Angle Conversion ---
    # This is the standard aerospace sequence for roll, pitch, and yaw.

    # Roll (phi): rotation about the x-axis.
    phi = np.arctan2(2 * (x * y + w * z), w**2 + x**2 - y**2 - z**2)

    # Pitch (theta): rotation about the y-axis.
    # Use np.clip to avoid errors with values slightly outside the [-1, 1] range due to float precision.
    pitch_arg = np.clip(-2 * (x * z - w * y), -1.0, 1.0)
    theta = np.arcsin(pitch_arg)

    # Yaw (psi): rotation about the z-axis.
    psi = np.arctan2(2 * (y * z + w * x), w**2 - x**2 - y**2 + z**2)

    # Create a new DataFrame to hold the results.
    eulers_df = pd.DataFrame(
        {"timestamp": quat_df["timestamp"], "phi": phi, "theta": theta, "psi": psi}
    )

    return eulers_df


def quat_inv(q: np.ndarray) -> np.ndarray:
    """
    Returns the inverse of a quaternion.
    Expects q as a numpy array [w, x, y, z].
    """
    # The inverse of a quaternion is its conjugate divided by its squared magnitude.
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    q_mag_sq = np.sum(q**2)
    if q_mag_sq == 0:
        # Return a zero quaternion or handle as an error if magnitude is zero.
        return np.zeros_like(q)
    return q_conj / q_mag_sq


def quat_comp(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Computes the composition of two quaternions (Hamilton product q1 * q2).
    Expects q1 and q2 as numpy arrays [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Hamilton product formulas
    w_comp = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_comp = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_comp = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_comp = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w_comp, x_comp, y_comp, z_comp])


def quat_inv_comp(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Computes the composition of the inverse of q1 with q2 (q1_inv * q2).
    Expects q1 and q2 as numpy arrays [w, x, y, z].
    """
    q1_inv = quat_inv(q1)
    return quat_comp(q1_inv, q2)


def rmat_of_quat(q: np.ndarray) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix corresponding to a quaternion.
    Expects q as a numpy array [w, x, y, z].
    """
    w, x, y, z = q

    # Pre-calculate squared terms
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z

    # Pre-calculate cross-terms
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Create the rotation matrix
    R = np.array(
        [
            [w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), w2 - x2 + y2 - z2, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), w2 - x2 - y2 + z2],
        ]
    )

    return R


def quat_of_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Returns the quaternion corresponding to a rotation of 'angle' radians
    around the given 'axis'.
    Expects axis as a 3-element numpy array.
    """
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)

    # Calculate half angle
    half_angle = angle / 2.0

    # Calculate sin and cos of half angle
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    # Create the quaternion [w, x, y, z]
    w = cos_half
    x = axis[0] * sin_half
    y = axis[1] * sin_half
    z = axis[2] * sin_half

    return np.array([w, x, y, z])
