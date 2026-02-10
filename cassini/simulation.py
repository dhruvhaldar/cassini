import numpy as np
from .kinematics import Quaternion
from .dynamics import RigidBody

class Simulation:
    def __init__(self, rigid_body, initial_omega, initial_quat):
        self.rb = rigid_body
        self.omega = np.array(initial_omega, dtype=float)
        self.quat = Quaternion(initial_quat)
        self.time = 0.0

        # History
        self.history = {
            'time': [],
            'omega': [],
            'quat': [],
            'energy': []
        }
        # Record initial state
        self._record()

    def _state_derivative(self, omega, torque):
        """
        Returns omega_dot
        """
        return self.rb.euler_equations(omega, torque)

    def step(self, dt, torque=np.zeros(3)):
        """
        Propagate state by dt using RK4.
        State: [q, omega]
        """
        # 1. k1
        k1_q_dot = self.quat.derivative(self.omega)
        k1_w_dot = self._state_derivative(self.omega, torque)

        # 2. k2
        q2 = Quaternion(self.quat.q + 0.5 * dt * k1_q_dot)
        w2 = self.omega + 0.5 * dt * k1_w_dot
        k2_q_dot = q2.derivative(w2)
        k2_w_dot = self._state_derivative(w2, torque) # Assume constant torque over step

        # 3. k3
        q3 = Quaternion(self.quat.q + 0.5 * dt * k2_q_dot)
        w3 = self.omega + 0.5 * dt * k2_w_dot
        k3_q_dot = q3.derivative(w3)
        k3_w_dot = self._state_derivative(w3, torque)

        # 4. k4
        q4 = Quaternion(self.quat.q + dt * k3_q_dot)
        w4 = self.omega + dt * k3_w_dot
        k4_q_dot = q4.derivative(w4)
        k4_w_dot = self._state_derivative(w4, torque)

        # Update
        self.quat.q += (dt / 6.0) * (k1_q_dot + 2*k2_q_dot + 2*k3_q_dot + k4_q_dot)
        self.quat.normalize()
        self.omega += (dt / 6.0) * (k1_w_dot + 2*k2_w_dot + 2*k3_w_dot + k4_w_dot)

        self.time += dt
        self._record()

    def _record(self):
        self.history['time'].append(self.time)
        self.history['omega'].append(self.omega.copy())
        self.history['quat'].append(self.quat.q.copy())

        # Kinetic Energy: 0.5 * w^T * J * w
        ke = 0.5 * np.dot(self.omega, self.rb.J @ self.omega)
        self.history['energy'].append(ke)

    def run_open_loop(self, duration, dt=0.1, torque=None):
        if torque is None:
            torque = np.zeros(3)
        steps = int(duration / dt)
        for _ in range(steps):
            self.step(dt, torque)
        return self.history

    def run_closed_loop(self, controller, target_quat=None, duration=10.0, dt=0.1):
        steps = int(duration / dt)
        for _ in range(steps):
            if hasattr(controller, 'compute_control'):
                try:
                    if target_quat is not None:
                         # For Lyapunov
                         u = controller.compute_control(self.quat, self.omega, target_quat)
                    else:
                         # For RateDamper
                         u = controller.compute_control(self.omega)
                except TypeError:
                     # Fallback
                     u = controller.compute_control(self.omega)
            else:
                u = np.zeros(3)

            self.step(dt, u)
        return self.history
