import numpy as np

class LyapunovController:
    def __init__(self, kp, kd):
        """
        kp: Proportional gain (scalar or 3x3 matrix)
        kd: Derivative gain (scalar or 3x3 matrix)
        """
        self.kp = kp
        self.kd = kd

    def compute_control(self, current_quat, current_omega, target_quat):
        """
        Computes control torque to align current_quat with target_quat.
        u = -Kp * q_err_vec * sign(q_err_scalar) - Kd * omega
        """
        # Error quaternion: q_err = q_target* . q_current
        # This represents the rotation needed to go from Target to Current?
        # Standard: q_err = q_target.conjugate().multiply(current_quat)
        # Let's check: if current == target, q_err = q* . q = Identity [1, 0, 0, 0].
        # q_vec = 0. Torque = -Kd * omega. If omega=0, torque=0. Correct.

        q_err = target_quat.conjugate().multiply(current_quat)

        # Shortest path logic: if q0 < 0, negate quaternion
        sign = np.sign(q_err.q[0]) if q_err.q[0] != 0 else 1
        if sign < 0:
            q_vec = -q_err.q[1:]
        else:
            q_vec = q_err.q[1:]

        # Handle scalar vs matrix gains
        if np.isscalar(self.kp):
            p_term = self.kp * q_vec
        else:
            p_term = self.kp @ q_vec

        if np.isscalar(self.kd):
            d_term = self.kd * current_omega
        else:
            d_term = self.kd @ current_omega

        # Torque = -Kp * q_vec - Kd * omega
        # Note: If q_err represents rotation FROM target TO body,
        # then q_vec is axis of rotation. We want to apply torque OPPOSITE to that.
        torque = -p_term - d_term
        return torque

class BdotController:
    def __init__(self, gain):
        self.k = gain

    def compute_control(self, b_field_body, b_dot_body):
        """
        Computes magnetic dipole moment and resulting torque.
        m = -k * b_dot
        Torque = m x b
        """
        m = -self.k * b_dot_body
        torque = np.cross(m, b_field_body)
        return torque

class RateDamper:
    """
    Simple detumbling controller that applies torque opposite to angular velocity.
    Idealized B-dot.
    """
    def __init__(self, gain):
        self.k = gain

    def compute_control(self, omega):
        return -self.k * omega
