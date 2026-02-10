import numpy as np

class Quaternion:
    def __init__(self, q):
        """
        Initialize quaternion [q0, q1, q2, q3] where q0 is scalar part.
        """
        self.q = np.array(q, dtype=float)
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.q)
        if norm > 1e-6:
            self.q /= norm
        else:
            # Handle zero quaternion edge case (identity)
            self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def norm(self):
        return np.linalg.norm(self.q)

    def to_list(self):
        return self.q.tolist()

    def __repr__(self):
        return f"Quaternion({self.q})"

    def conjugate(self):
        return Quaternion([self.q[0], -self.q[1], -self.q[2], -self.q[3]])

    def multiply(self, other):
        """
        Quaternion multiplication.
        """
        q1 = self.q
        q2 = other.q

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return Quaternion([w, x, y, z])

    def to_dcm(self):
        """
        Converts quaternion to Direction Cosine Matrix (Inertial to Body).
        v_body = C * v_inertial
        """
        q0, q1, q2, q3 = self.q

        C = np.zeros((3, 3))

        C[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
        C[0, 1] = 2 * (q1*q2 + q0*q3)
        C[0, 2] = 2 * (q1*q3 - q0*q2)

        C[1, 0] = 2 * (q1*q2 - q0*q3)
        C[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
        C[1, 2] = 2 * (q2*q3 + q0*q1)

        C[2, 0] = 2 * (q1*q3 + q0*q2)
        C[2, 1] = 2 * (q2*q3 - q0*q1)
        C[2, 2] = q0**2 - q1**2 - q2**2 + q3**2

        return C

    def derivative(self, omega):
        """
        Computes q_dot = 0.5 * B(q) * omega
        omega is angular velocity vector [wx, wy, wz] in body frame.
        """
        wx, wy, wz = omega

        q0, q1, q2, q3 = self.q

        q_dot = 0.5 * np.array([
            -q1*wx - q2*wy - q3*wz,
             q0*wx - q3*wy + q2*wz,
             q3*wx + q0*wy - q1*wz,
            -q2*wx + q1*wy + q0*wz
        ])

        return q_dot

    def integrate(self, omega, dt):
        """
        Simple Euler integration for quaternion.
        q_next = q + q_dot * dt
        Normalize afterwards.
        """
        q_dot = self.derivative(omega)
        q_next = self.q + q_dot * dt
        return Quaternion(q_next)

def euler_to_quat(roll, pitch, yaw, sequence='321'):
    """
    Converts Euler angles to quaternion.
    Currently only supporting 3-2-1 sequence (Yaw-Pitch-Roll).
    Inputs in radians.
    """
    c1 = np.cos(yaw / 2)
    s1 = np.sin(yaw / 2)
    c2 = np.cos(pitch / 2)
    s2 = np.sin(pitch / 2)
    c3 = np.cos(roll / 2)
    s3 = np.sin(roll / 2)

    if sequence == '321':
        q0 = c1*c2*c3 + s1*s2*s3
        q1 = c1*c2*s3 - s1*s2*c3
        q2 = c1*s2*c3 + s1*c2*s3
        q3 = s1*c2*c3 - c1*s2*s3
        return Quaternion([q0, q1, q2, q3])
    else:
        raise NotImplementedError("Only 3-2-1 sequence is currently supported")

def quat_to_mrp(q):
    """
    Converts Quaternion to Modified Rodrigues Parameters (MRP).
    sigma_i = q_i / (1 + q_0) for i=1,2,3
    """
    q0 = q.q[0]
    # Check for shadow set condition (near -360 rotation)
    if q0 < 0:
        # Use shadow set: q -> -q. This keeps q0 positive and avoids singularity at -1.
        # sigma_i = -q_i / (1 - q_0)
        # But generally, we want to map to minimal set.
        # If q0 is close to -1, sigma goes to infinity.
        pass

    if np.isclose(q0, -1.0):
        # Singularity
        return np.array([0.0, 0.0, 0.0])

    return q.q[1:] / (1 + q0)

def mrp_to_quat(sigma):
    """
    Converts MRP to Quaternion.
    """
    sigma = np.array(sigma)
    s2 = np.dot(sigma, sigma)
    q0 = (1 - s2) / (1 + s2)
    q_vec = 2 * sigma / (1 + s2)
    return Quaternion([q0, q_vec[0], q_vec[1], q_vec[2]])
