import numpy as np
import pytest
from cassini.kinematics import Quaternion, euler_to_quat, quat_to_mrp, mrp_to_quat

def test_quaternion_norm_constraint():
    """Ensures quaternion propagation maintains unit norm."""
    q = Quaternion([0.5, 0.5, 0.5, 0.5]) # Valid unit quaternion
    assert np.isclose(q.norm(), 1.0)

    # Test integration
    omega = np.array([0.1, 0.0, 0.0])
    q_next = q.integrate(omega=omega, dt=0.1)

    assert np.isclose(q_next.norm(), 1.0)

def test_euler_to_quat():
    """Test Euler angle conversion."""
    # 90 degrees yaw (z-axis)
    # q = [cos(45), 0, 0, sin(45)] = [0.707, 0, 0, 0.707]
    q = euler_to_quat(0, 0, np.pi/2)
    assert np.allclose(q.q, [0.70710678, 0, 0, 0.70710678])

def test_mrp_conversion():
    """Test MRP <-> Quaternion round trip."""
    sigma = np.array([0.1, 0.2, 0.3])
    q = mrp_to_quat(sigma)
    sigma_back = quat_to_mrp(q)

    assert np.allclose(sigma, sigma_back)

def test_dcm_conversion():
    """Test Quaternion to DCM."""
    # Identity quaternion
    q = Quaternion([1, 0, 0, 0])
    dcm = q.to_dcm()
    assert np.allclose(dcm, np.eye(3))

    # 90 deg rotation about Z
    q_z = Quaternion([0.70710678, 0, 0, 0.70710678])
    dcm_z = q_z.to_dcm()
    expected = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    # The convention in my code seems to be Body to Inertial or Inertial to Body?
    # Let's check `to_dcm` implementation again.
    # C[0, 1] = 2 * (q1*q2 + q0*q3)
    # If q = [c, 0, 0, s], q0=c, q3=s
    # C[0, 1] = 2 * (0 + c*s) = 2cs = sin(theta) = 1
    # C[1, 0] = 2 * (0 - c*s) = -1
    # So it looks like rotation matrix R_z(90) = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    # This corresponds to a Passive Rotation (Coordinate Transformation) or Active?
    # R_z(theta) = [[c, s, 0], [-s, c, 0], [0, 0, 1]] usually transforms vector from inertial to body?
    # Or body to inertial?
    # If v_b = C * v_i
    # For a frame rotated by 90 deg around Z (X_b aligns with Y_i, Y_b aligns with -X_i),
    # v_i = [1, 0, 0] -> v_b = [0, -1, 0]
    # My matrix gives [0, 1, 0] * [1, 0, 0]^T = [0, -1, 0]
    # Wait, [0, 1, 0] means x_new = y_old.
    # Let's check my matrix calc:
    # C[0,0] = c^2 - s^2 = 0
    # C[0,1] = 2cs = 1
    # C[1,0] = -2cs = -1
    # C[1,1] = c^2 - s^2 = 0
    # So C = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    # v_new = C * v_old
    # v_old = [1, 0, 0] -> v_new = [0, -1, 0].
    # If Frame B is rotated +90 deg wrt Frame A about Z.
    # X_A is [1,0,0]. In Frame B, X_A points along -Y_B. So [0, -1, 0].
    # So my matrix transforms A to B. Inertial to Body.

    assert np.allclose(dcm_z, expected)
