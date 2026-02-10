import numpy as np
import pytest
from cassini.control import LyapunovController
from cassini.kinematics import Quaternion

def test_lyapunov_convergence():
    """Simple check that control torque opposes error."""
    # Target: Identity
    target = Quaternion([1, 0, 0, 0])

    # Current: Rotated 90 deg about Z
    current = Quaternion([0.7071, 0, 0, 0.7071])

    # Omega: Spinning +Z
    omega = np.array([0, 0, 0.1])

    # Controller
    kp = 1.0
    kd = 1.0
    controller = LyapunovController(kp, kd)

    torque = controller.compute_control(current, omega, target)

    # Expect torque to be negative Z to counter both error (requires -Z rotation) and rate (+Z spin)
    # q_err = target* . current = I . q_curr = q_curr
    # q_vec = [0, 0, 0.7071]
    # P-term = kp * 0.7071 = 0.7071
    # D-term = kd * 0.1 = 0.1
    # Torque = -0.7071 - 0.1 = -0.8071 (in Z)

    assert torque[2] < 0
    assert np.isclose(torque[0], 0)
    assert np.isclose(torque[1], 0)

def test_lyapunov_shortest_path():
    """Check if controller takes shortest path (q vs -q)."""
    target = Quaternion([1, 0, 0, 0])

    # Current: Rotated -10 deg about Z. q = [cos(-5), 0, 0, sin(-5)]
    # cos(-5) ~ 0.996, sin(-5) ~ -0.087
    # q = [0.996, 0, 0, -0.087]
    # But q and -q are same. Let's force negative q0.
    # q_neg = [-0.996, 0, 0, 0.087]

    q_neg = Quaternion([-0.996, 0, 0, 0.087])
    # Note: My Quaternion class normalizes on init. It keeps sign.
    # Let's double check if normalize enforces q0 > 0? No, it just divides by norm.

    omega = np.array([0, 0, 0])
    controller = LyapunovController(1.0, 1.0)

    torque = controller.compute_control(q_neg, omega, target)

    # Rotation is -10 deg. We want torque +Z to correct it.
    # If controller naively used q_vec=0.087 (from q_neg), torque = -Kp * 0.087 = negative.
    # This would push it further away (long way).
    # Correct logic should flip q to [0.996, 0, 0, -0.087].
    # q_vec = -0.087.
    # Torque = -Kp * (-0.087) = +0.087. Positive.

    assert torque[2] > 0
