import numpy as np
import pytest
from cassini.dynamics import RigidBody
from cassini.kinematics import Quaternion

def test_rigid_body_init():
    rb = RigidBody([1, 2, 3])
    assert np.allclose(rb.J, np.diag([1, 2, 3]))

    rb_mat = RigidBody(np.diag([1, 2, 3]))
    assert np.allclose(rb_mat.J, np.diag([1, 2, 3]))

def test_euler_equations():
    # Symmetric body, spinning about principal axis
    rb = RigidBody([1, 1, 1])
    omega = np.array([0, 0, 1])
    torque = np.array([0, 0, 0])

    # w x Jw = [0,0,1] x [0,0,1] = 0
    # w_dot = inv(J) * (0 - 0) = 0
    omega_dot = rb.euler_equations(omega, torque)
    assert np.allclose(omega_dot, [0, 0, 0])

    # Asymmetric body, spinning about intermediate axis (unstable, but initially w_dot depends on w)
    rb = RigidBody([1, 2, 3])
    omega = np.array([0, 1, 0]) # Y-axis spin

    # Jw = [0, 2, 0]
    # w x Jw = [0, 1, 0] x [0, 2, 0] = 0
    # So w_dot should be zero initially if exactly on principal axis
    omega_dot = rb.euler_equations(omega, torque)
    assert np.allclose(omega_dot, [0, 0, 0])

    # Off-axis spin
    omega = np.array([1, 1, 1])
    # Jw = [1, 2, 3]
    # w x Jw = [1, 1, 1] x [1, 2, 3] = [1*3 - 1*2, 1*1 - 1*3, 1*2 - 1*1] = [1, -2, 1]
    # w_dot = inv(J) * (-[1, -2, 1]) = [ -1/1, 2/2, -1/3 ] = [-1, 1, -0.333]
    expected = np.array([-1.0, 1.0, -1/3])
    omega_dot = rb.euler_equations(omega, torque)
    assert np.allclose(omega_dot, expected)

def test_gravity_gradient():
    # Spherical body: no gravity gradient torque
    rb = RigidBody([100, 100, 100])
    q = Quaternion([1, 0, 0, 0])
    r = np.array([7000e3, 0, 0])
    gg_torque = rb.gravity_gradient_torque(q, r)
    assert np.allclose(gg_torque, [0, 0, 0])

    # Asymmetric body
    rb = RigidBody([100, 200, 300])
    # Nadir vector is [-1, 0, 0] in inertial frame.
    # Body aligned with inertial (q=identity).
    # So c3 in body frame is [-1, 0, 0].
    # J * c3 = [-100, 0, 0].
    # c3 x J*c3 = [-1, 0, 0] x [-100, 0, 0] = 0.
    # Should be zero torque if principal axis points to nadir.
    gg_torque = rb.gravity_gradient_torque(q, r)
    assert np.allclose(gg_torque, [0, 0, 0])

    # Now rotate body 45 deg about Z
    # q = [0.9239, 0, 0, 0.3827] (approx for 45 deg? No, 22.5. Wait. 45 deg = pi/4)
    # q for 45 deg Z rot: cos(22.5), 0, 0, sin(22.5)
    # Nadir (Inertial) = [-1, 0, 0]
    # Rotated Body Frame: X_b is 45 deg from X_i.
    # So Nadir vector in Body Frame should be:
    # C_bi = [ [c, s, 0], [-s, c, 0], [0, 0, 1] ] for theta=45
    # Nadir_i = [-1, 0, 0]^T
    # Nadir_b = C_bi * [-1, 0, 0]^T = [-c, s, 0]^T = [-0.707, 0.707, 0]
    # J = diag([100, 200, 300])
    # J * Nadir_b = [-70.7, 141.4, 0]
    # Nadir_b x (J * Nadir_b) = [0, 0, (-0.707 * 141.4) - (0.707 * -70.7)]
    # = [0, 0, -100 + 50] = [0, 0, -50] approx.
    # So torque should be around Z axis.

    # Let's verify code calculates non-zero torque
    # q for 45 deg Z rotation
    q_rot = Quaternion([np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)])
    gg_torque = rb.gravity_gradient_torque(q_rot, r)

    assert not np.allclose(gg_torque, [0, 0, 0])
    assert np.abs(gg_torque[2]) > 1e-5 # Should be significant (order of 1e-4)
    assert np.isclose(gg_torque[0], 0)
    assert np.isclose(gg_torque[1], 0)
