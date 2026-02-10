import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cassini.dynamics import RigidBody
from cassini.simulation import Simulation

def main():
    # Asymmetric spacecraft
    rb = RigidBody([100, 150, 200])

    # Unstable intermediate axis spin (perturbed)
    omega_init = [0.1, 2.0, 0.1]
    q_init = [1, 0, 0, 0]

    sim = Simulation(rb, omega_init, q_init)

    res = sim.run_open_loop(duration=100.0, dt=0.05)

    omega = np.array(res['omega'])

    # Plot Polhode (wx vs wy vs wz)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(omega[:,0], omega[:,1], omega[:,2])
    ax.set_xlabel('$\omega_x$')
    ax.set_ylabel('$\omega_y$')
    ax.set_zlabel('$\omega_z$')
    ax.set_title('Polhode Plot (Torque Free Motion)')

    plt.savefig('polhode_plot.png')
    print("Saved polhode_plot.png")

if __name__ == "__main__":
    main()
