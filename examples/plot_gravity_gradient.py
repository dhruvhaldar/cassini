import numpy as np
import matplotlib.pyplot as plt
from cassini.dynamics import RigidBody
from cassini.simulation import Simulation
from cassini.kinematics import Quaternion

def main():
    # Asymmetric body optimized for gravity gradient stabilization (I_x < I_y < I_z)
    # Pitch axis (Y) should be stable/oscillatory.
    # Let's choose inertia: J = diag([100, 300, 400])
    rb = RigidBody([100, 300, 400])
    
    # Circular orbit at 7000 km
    R = 7000e3
    mu = 3.986004418e14
    omega_orbit = np.sqrt(mu / (R**3)) # ~0.00108 rad/s
    
    # Small initial deviation
    q_init = Quaternion([0.996, 0.05, 0.05, 0.05])
    q_init.normalize()
    omega_init = [0.0001, omega_orbit + 0.0005, 0.0001]
    
    sim = Simulation(rb, omega_init, q_init.q)
    
    duration = 15000.0 # seconds
    dt = 2.0
    steps = int(duration / dt)
    
    times = []
    pitch_angles = []
    roll_angles = []
    
    for i in range(steps):
        t = sim.time
        theta = omega_orbit * t
        r_orbit = R * np.array([np.cos(theta), np.sin(theta), 0.0])
        
        # Calculate GG torque
        torque_gg = rb.gravity_gradient_torque(sim.quat, r_orbit, mu=mu)
        
        # Propagate one step
        sim.step(dt, torque_gg)
        
        # Calculate pitch and roll relative to local vertical
        C_bi = sim.quat.to_dcm()
        nadir_i = -r_orbit / R
        nadir_b = C_bi @ nadir_i
        
        # Roll and pitch angle estimation
        pitch = np.degrees(np.arctan2(nadir_b[1], nadir_b[0]))
        roll = np.degrees(np.arctan2(nadir_b[2], nadir_b[0]))
        
        times.append(t)
        pitch_angles.append(pitch)
        roll_angles.append(roll)

    plt.figure(figsize=(10, 6))
    plt.plot(times, pitch_angles, label='Pitch', color='#3b82f6')
    plt.plot(times, roll_angles, label='Roll', color='#ef4444')
    plt.xlabel('Time (s)')
    plt.ylabel('Deviation from local vertical (degrees)')
    plt.title('Gravity Gradient Stabilization (Pitch & Roll Oscillation)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('gravity_gradient_plot.png')
    print("Saved gravity_gradient_plot.png")

if __name__ == "__main__":
    main()
