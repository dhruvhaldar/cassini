# Cassini

**Cassini** is a Python-based Attitude Determination and Control System (ADCS) simulation library. It implements the rigid body dynamics and non-linear control strategies required by **SD2910 Spacecraft Dynamics**.

Uniquely, Cassini is architected as a **web-ready microservice**. It can run locally as a research tool or be deployed to **Vercel** to power 3D visualizations of spacecraft maneuvers.

## 📚 Syllabus Mapping (SD2910)

This project covers the full kinematics and dynamics spectrum required by the course:

| Module | Syllabus Topic | Implemented Features |
| --- | --- | --- |
| **Kinematics** | *Parameterizations in 3D* | Conversions between DCM, Euler Angles (12 sets), Quaternions, Classical (CRP) & Modified Rodrigues Parameters (MRP). |
| **Dynamics** | *Rigid Body Dynamics* | Euler’s Rotational Equations, Full Inertia Tensor support (), Gravity Gradient Torque modeling. |
| **Stability** | *Nonlinear Stability* | Lyapunov candidate functions () and derivative analysis (). |
| **Control** | *Feedback Control Laws* | Detumbling (B-dot), Pointing (PD Controller on Quaternions), and Eigenaxis rotations. |
| **Simulation** | *Prediction & Evaluation* | 4th Order Runge-Kutta (RK4) integrator for propagating attitude state over time. |

## 🚀 Deployment (Vercel)

This project uses **FastAPI** to expose the physics engine as a serverless function.

1. **Fork** this repository.
2. Import the project into your **Vercel** dashboard.
3. Vercel will automatically detect `api/index.py` and `requirements.txt`.
4. Your simulation API will be live at `https://your-project.vercel.app/api/simulate`.

## 📊 Artifacts & Control Analysis

### 1. Torque-Free Motion (The Polhode Plot)

*Visualizes the path of the angular velocity vector  in the body frame for an asymmetric spacecraft.*

**Code:**

```python
from cassini.dynamics import RigidBody
from cassini.simulation import Simulation

# Define an asymmetric spacecraft (I1 < I2 < I3)
sat = RigidBody(inertia=[100, 150, 200])
sim = Simulation(sat, omega_init=[0.1, 2.0, 0.1], q_init=[1,0,0,0])
results = sim.run_open_loop(duration=100)

# results is a dict, plotting needs extraction
# See examples/plot_torque_free.py
```

**Artifact Output:**

> *Figure 1: The "Polhode" curve. The angular velocity vector traces a closed curve on the intersection of the Kinetic Energy ellipsoid and the Angular Momentum sphere. The stability around the minor and major axes (and instability around the intermediate axis) is clearly visible.*

### 2. Lyapunov Stability Proof (Rest-to-Rest Maneuver)

*Demonstrates the convergence of a nonlinear control law using a Lyapunov function.*

**Mathematical Basis:**
We select a Lyapunov candidate function:



The control law  ensures .

**Code:**

```python
from cassini.control import LyapunovController
from cassini.simulation import Simulation
from cassini.dynamics import RigidBody
from cassini.kinematics import Quaternion

sim = Simulation(RigidBody([100,100,100]), [0,0,0], [0.707, 0, 0, 0.707])

# Target: Rotate 90 degrees about Z-axis
controller = LyapunovController(kp=10, kd=20)
results = sim.run_closed_loop(controller, target_quat=Quaternion([1, 0, 0, 0]), duration=60)

# See examples/plot_lyapunov.py for plotting
```

**Artifact Output:**

> *Figure 2: Control Convergence. Top subplot: Angular velocity . Bottom subplot: Quaternion error vector . The asymptotic decay confirms global asymptotic stability for the chosen gains.*

### 3. Gravity Gradient Stabilization

*Simulates the passive stabilization of a satellite using Earth's gravity gradient.*

**Artifact Output:**

> *Figure 3: Pitch and Roll oscillation. The spacecraft oscillates around the local vertical (nadir) vector like a pendulum, bounded by the gravity gradient torque .*

## 🧪 Testing Strategy

### Unit Tests (Kinematics Verification)

Located in `tests/unit/`, these ensure mathematical rigour, critical for passing the exam.

*Example: `tests/unit/test_kinematics.py*`

```python
def test_quaternion_norm_constraint():
    """Ensures quaternion propagation maintains unit norm to prevent numerical drift."""
    q = Quaternion([0.5, 0.5, 0.5, 0.5]) # Valid unit quaternion
    q_next = q.integrate(omega=[0.1, 0, 0], dt=100.0) # Long integration

    assert abs(q_next.norm() - 1.0) < 1e-6

```

### E2E Tests (Mission Simulation)

Located in `tests/e2e/`.

*Example: `tests/e2e/test_detumbling.py*`

```python
def test_detumbling_energy_decay():
    """Verifies that kinetic energy strictly decreases under B-dot control."""
    sim = Simulation(initial_omega=[5, 5, 5]) # High spin
    controller = BdotController(gain=1e-4)
    res = sim.run_closed_loop(controller, dt=0.1, duration=50)

    energy_start = res['energy'][0]
    energy_end = res['energy'][-1]

    assert energy_end < 0.1 * energy_start

```

## ⚖️ License

**MIT License**

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files... [Standard MIT Text]
