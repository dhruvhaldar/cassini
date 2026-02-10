import numpy as np
import matplotlib.pyplot as plt
from cassini.dynamics import RigidBody
from cassini.simulation import Simulation
from cassini.control import LyapunovController
from cassini.kinematics import Quaternion

def main():
    # Setup
    rb = RigidBody([100, 150, 200])
    omega_init = [0.1, 0.1, 0.1]
    # Initial attitude: 90 deg about X
    q_init = [0.707, 0.707, 0, 0]

    sim = Simulation(rb, omega_init, q_init)

    # Controller
    # Gains
    kp = 50.0
    kd = 100.0
    controller = LyapunovController(kp, kd)

    # Target: Identity
    target = Quaternion([1, 0, 0, 0])

    # Run
    res = sim.run_closed_loop(controller, target, duration=60.0, dt=0.1)

    # Plot
    time = res['time']
    omega = np.array(res['omega'])
    quat = np.array(res['quat'])

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(time, omega)
    axs[0].set_title('Angular Velocity')
    axs[0].set_ylabel('rad/s')
    axs[0].legend(['x', 'y', 'z'])
    axs[0].grid(True)

    axs[1].plot(time, quat)
    axs[1].set_title('Quaternion')
    axs[1].set_ylabel('Value')
    axs[1].legend(['q0', 'q1', 'q2', 'q3'])
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('lyapunov_plot.png')
    print("Saved lyapunov_plot.png")

if __name__ == "__main__":
    main()
