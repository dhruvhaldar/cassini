import numpy as np
from .kinematics import Quaternion

class RigidBody:
    def __init__(self, inertia):
        """
        Initialize Rigid Body with Inertia Tensor (diagonal or full matrix).
        inertia: list or array (3,) or (3,3)
        """
        self.J = np.array(inertia, dtype=float)
        if self.J.shape == (3,):
            self.J = np.diag(self.J)
        elif self.J.shape != (3, 3):
            raise ValueError("Inertia must be 3-element list or 3x3 matrix")

        self.J_inv = np.linalg.inv(self.J)

    def euler_equations(self, omega, torque):
        """
        Computes angular acceleration (omega_dot) using Euler's equations.
        J * w_dot + w x (J * w) = T
        w_dot = J_inv * (T - w x (J * w))
        """
        Jw = self.J @ omega
        gyroscopic = np.cross(omega, Jw)

        omega_dot = self.J_inv @ (torque - gyroscopic)
        return omega_dot

    def gravity_gradient_torque(self, q, r_orbit, mu=3.986e14):
        """
        Computes gravity gradient torque.
        q: Quaternion (Inertial to Body)
        r_orbit: position vector of spacecraft in inertial frame (m)
        mu: Gravitational parameter (m^3/s^2), default Earth
        """
        R = np.linalg.norm(r_orbit)
        if R < 1e-3:
            return np.zeros(3)

        # Nadir vector in inertial frame (pointing to Earth center)
        nadir_inertial = -r_orbit / R

        # Transform nadir vector to Body frame
        # q.to_dcm() returns C_bi (Inertial to Body) based on my previous verification
        C_bi = q.to_dcm()
        c3 = C_bi @ nadir_inertial

        # 3 * mu / R^3
        coeff = 3 * mu / (R**3)

        Jc3 = self.J @ c3
        torque_gg = coeff * np.cross(c3, Jc3)

        return torque_gg
