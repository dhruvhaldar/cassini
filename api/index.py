import sys
import os
# Add project root to sys.path to allow imports from cassini/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from cassini.dynamics import RigidBody
from cassini.simulation import Simulation
from cassini.control import LyapunovController, RateDamper
from cassini.kinematics import Quaternion

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class SimulationRequest(BaseModel):
    inertia: list[float]
    omega_init: list[float]
    q_init: list[float]
    duration: float = 10.0
    dt: float = 0.1
    control_mode: str = "none" # none, detumble, pointing
    gains: list[float] = [1.0, 1.0] # [kp, kd] or [k]
    target_quat: list[float] = [1.0, 0.0, 0.0, 0.0]

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    rb = RigidBody(req.inertia)
    sim = Simulation(rb, req.omega_init, req.q_init)

    if req.control_mode == "detumble":
        controller = RateDamper(gain=req.gains[0])
        history = sim.run_closed_loop(controller, target_quat=None, duration=req.duration, dt=req.dt)
    elif req.control_mode == "pointing":
        kp, kd = req.gains[0], req.gains[1]
        controller = LyapunovController(kp, kd)
        target = Quaternion(req.target_quat)
        history = sim.run_closed_loop(controller, target_quat=target, duration=req.duration, dt=req.dt)
    else:
        history = sim.run_open_loop(duration=req.duration, dt=req.dt)

    # Convert numpy arrays to lists for JSON serialization
    return {
        "time": history['time'],
        "omega": [arr.tolist() for arr in history['omega']],
        "quat": [arr.tolist() for arr in history['quat']],
        "energy": history['energy']
    }
