import numpy as np
from cassini.simulation import Simulation
from cassini.dynamics import RigidBody
from cassini.control import RateDamper

def test_detumbling_decay():
    """Verify angular velocity decays with RateDamper."""
    rb = RigidBody([100, 200, 300])
    omega_init = [1.0, 1.0, 1.0]
    q_init = [1, 0, 0, 0]

    sim = Simulation(rb, omega_init, q_init)

    # Gain such that torque = -10 * omega
    # If J=300 (largest axis), alpha ~ -0.033 * omega.
    # Time constant ~ 30s.
    controller = RateDamper(gain=10.0)

    res = sim.run_closed_loop(controller, target_quat=None, duration=50.0, dt=0.1)

    # Check energy decay
    e_start = res['energy'][0]
    e_end = res['energy'][-1]

    assert e_end < e_start
    # Should decay significantly
    assert e_end < 0.1 * e_start
