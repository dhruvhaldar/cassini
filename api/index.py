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

from fastapi.staticfiles import StaticFiles

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

@app.get("/api/debug")
def debug_info():
    import os
    possible_paths = {
        "cwd": os.getcwd(),
        "__file__": __file__,
        "cwd_public": os.path.abspath("public"),
        "cwd_public_exists": os.path.exists(os.path.abspath("public")),
        "file_parent_public": os.path.join(os.path.dirname(os.path.abspath(__file__)), "public"),
        "file_parent_public_exists": os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")),
        "file_parent_parent_public": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public"),
        "file_parent_parent_public_exists": os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")),
    }
    try:
        possible_paths["cwd_contents"] = os.listdir(os.getcwd())
    except Exception as e:
        possible_paths["cwd_contents_error"] = str(e)
    try:
        possible_paths["file_parent_contents"] = os.listdir(os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        possible_paths["file_parent_contents_error"] = str(e)
    return possible_paths

# Mount static files at root as fallback after API routes are defined
possible_dirs = [
    os.path.join(os.getcwd(), "public"),
    os.path.abspath("public"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "public"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public"),
]
public_dir = None
for d in possible_dirs:
    if os.path.exists(d) and os.path.isdir(d):
        public_dir = d
        break

if public_dir:
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")


