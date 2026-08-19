"""
Geometry Lane Fix
carla_dra_benchmark.py
======================
Closed-loop CARLA Benchmark: TTC vs Utilitarian vs DRA Planners
================================================================

WHAT THIS FILE IS
-----------------
A full migration of CA_4_5_animated.py's three-planner experimental
framework into a genuine closed-loop CARLA simulation.

SCIENTIFIC GUARANTEES
---------------------
1. All three planners run in SEPARATE CARLA episodes — no planner
   contaminates another planner's environment.
2. Every episode with the same (scenario, seed) uses IDENTICAL:
   - Spawn location and yaw
   - Pedestrian blueprint, start position, and trajectory
   - Oncoming vehicle blueprint and start position
   - Weather, timestep, ego blueprint, physics settings
   The ONLY difference between episodes is the planner.
3. Metrics are computed from CARLA world state, not planner-internal
   variables — this is ground truth evaluation.
4. Planners receive a common PlannerInput struct derived from CARLA
   perception. None receives future ground truth.

USAGE
-----
  # Standard Publication Mode:
  python carla_dra_benchmark.py

  # Visualization Mode:
  python carla_dra_benchmark.py --visualize

  # Specific scenario/seeds in visualization mode:
  python carla_dra_benchmark.py --scenario S1 --seeds 1000 --visualize
"""

# ===========================================================================
# IMPORTS
# ===========================================================================
import carla
import numpy as np
import math
import time
import csv
import os
import json
import argparse
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from enum import Enum
from enum import auto

# ===========================================================================
# §1  GLOBAL CONSTANTS
# ===========================================================================

DT              = 0.05 #0.01#0.05    # seconds per CARLA tick
SIM_DURATION    = 8#10.0 #8.0, 20.0     # seconds
MAX_STEPS       = int(SIM_DURATION / DT)

# Ego vehicle initial state
EGO_V0          = 8.0 #11.2   # m/s (~40 km/h)
EGO_LANE_Y      = -1.5   # m  — local frame, lane centre

# Road boundaries
ONCOMING_BOUND  =  0.0   # y > 0 → oncoming lane violation
PAVEMENT_BOUND  = -2.4   # y < -2.4 → pavement violation
ROAD_EDGE       = -3.0   # physical curb

# Scenario geometry
TREE_LOCAL      = np.array([14.0, -2.5])   
PED_SPAWN_LOCAL = np.array([25.0, -3.2]) #x was 14.0   
PED_VELOCITY    = np.array([0.0,  0.25]) #<---- CHECK THIS vy was 1.25  
PED_START_T     = 0.5                       
PED_VISIBLE_Y   = -3.0 #-2.5                      
ONCOMING_SPAWN  = np.array([63.0, 0.5]) #x was 60, 22.0    
ONCOMING_V_X    = -4.5#-5.0 #was 6.0                      

# Safety thresholds
COLLISION_RADIUS     = 1.0    
CRITICAL_TTC         = 2.0    

# DRA risk-planning parameters (v4.2)
DRA_HORIZON_TICKS     = 200    # 2.0 s at DT=0.01 s
PED_RISK_SIGMA        = 3.0    # m; broader anticipation than v4.1
PED_RISK_WEIGHT       = 0.8
PROGRESS_COST_WEIGHT  = 10.0   # lower than v4.1 -50 to prevent speed reward masking risk
NEAR_MISS_DIST       = 2.0    

# Controller parameters
MAX_ACCEL       = 3.5    
MAX_DECEL       = 8.0    
MAX_STEER_RATE  = 0.15   
MAX_STEER       = 0.5    

# ===========================================================================
# DRA / CONTROLLER SHARED STEERING GEOMETRY
# ===========================================================================
#
# This converts a requested lateral displacement into the maximum
# physically achievable one-step lateral displacement under the SAME
# steering-rate and steering-limit assumptions used by ControlAdapter.
#
# IMPORTANT:
# This does NOT modify ControlAdapter.
# It only makes DRA's internal prediction use the same actuator envelope.
# ===========================================================================

# def achievable_dy_from_requested_dy(
#     requested_dy: float,
#     current_speed: float,
#     prev_steer: float,
# ) -> float:
#     fwd_dist = max(current_speed * DT, 0.01)

#     # Same geometric mapping used by ControlAdapter.
#     requested_steer_rad = -math.atan2(
#         requested_dy,
#         fwd_dist
#     )

#     requested_steer_norm = (
#         requested_steer_rad / (math.pi / 4.0)
#     )

#     # Same absolute steering limit.
#     clipped_steer_norm = float(
#         max(
#             -MAX_STEER,
#             min(MAX_STEER, requested_steer_norm)
#         )
#     )

#     # DRA does not own controller state, so use a neutral steering
#     # reference for the planner's one-step feasibility envelope.
#     #
#     # The controller itself remains stateful and unchanged.
#     max_delta_steer = (
#         MAX_STEER_RATE * DT / (math.pi / 4.0)
#     )

#     # feasible_steer_norm = float(
#     #     max(
#     #         -max_delta_steer,
#     #         min(max_delta_steer, clipped_steer_norm)
#     #     )
#     # )
#     steer_norm = float(
#         max(
#             self.prev_steer - max_delta_steer,
#             min(
#                 self.prev_steer + max_delta_steer,
#                 clipped_steer_norm
#             )
#         )
#     )

#     feasible_steer_rad = (
#         steer_norm * (math.pi / 4.0) #feasible_steer_norm * (math.pi / 4.0)
#     )

#     # Invert the same geometric relationship:
#     #
#     #     steer_rad = atan2(dy, fwd_dist)
#     #
#     # therefore:
#     #
#     #     dy = tan(steer_rad) * fwd_dist
#     #
#     achievable_dy = (
#         math.tan(feasible_steer_rad) * fwd_dist
#     )

#     return float(achievable_dy)
# def achievable_dy_from_requested_dy(
#     requested_dy: float,
#     current_speed: float,
#     prev_steer: float,
# ) -> float:
#     fwd_dist = max(current_speed * DT, 0.01)

#     # Same geometric mapping used by ControlAdapter.
#     requested_steer_rad = -math.atan2(
#         requested_dy,
#         fwd_dist
#     )

#     requested_steer_norm = (
#         requested_steer_rad / (math.pi / 4.0)
#     )

#     # Same absolute steering limit.
#     clipped_steer_norm = float(
#         max(
#             -MAX_STEER,
#             min(MAX_STEER, requested_steer_norm)
#         )
#     )

#     # ================================================================
#     # SAME STEERING-RATE LIMIT AS ControlAdapter
#     # ================================================================
#     max_delta_steer = (
#         MAX_STEER_RATE * DT / (math.pi / 4.0)
#     )

#     feasible_steer_norm = float(
#         max(
#             prev_steer - max_delta_steer,
#             min(
#                 prev_steer + max_delta_steer,
#                 clipped_steer_norm
#             )
#         )
#     )

#     feasible_steer_rad = (
#         feasible_steer_norm * (math.pi / 4.0)
#     )

#     # Invert the same geometric relationship:
#     #
#     #     steer_rad = atan2(dy, fwd_dist)
#     #
#     # therefore:
#     #
#     #     dy = tan(steer_rad) * fwd_dist

#     achievable_dy = (
#         -math.tan(feasible_steer_rad) * fwd_dist
#     )

#     return float(achievable_dy)
def achievable_dy_from_requested_dy(
    requested_dy: float,
    current_speed: float,
    prev_steer: float,
) -> float:

    fwd_dist = max(current_speed * DT, 0.01)

    # Same geometric mapping used by ControlAdapter.
    requested_steer_rad = -math.atan2(
        requested_dy,
        fwd_dist
    )

    requested_steer_norm = (
        requested_steer_rad / (math.pi / 4.0)
    )

    # Same absolute steering limit.
    clipped_steer_norm = float(
        max(
            -MAX_STEER,
            min(MAX_STEER, requested_steer_norm)
        )
    )

    # Same steering-rate limit used by ControlAdapter.
    max_delta_steer = (
        MAX_STEER_RATE * DT / (math.pi / 4.0)
    )

    # IMPORTANT:
    # Start from the ACTUAL controller steering state.
    feasible_steer_norm = float(
        max(
            prev_steer - max_delta_steer,
            min(
                prev_steer + max_delta_steer,
                clipped_steer_norm
            )
        )
    )

    feasible_steer_rad = (
        feasible_steer_norm * (math.pi / 4.0)
    )

    # Invert the same geometric relationship.
    achievable_dy = (
        -fwd_dist * math.tan(feasible_steer_rad)
    )

    return float(achievable_dy)
# =========================================================================== 
# DEBUG INSTRUMENTATION — NO BEHAVIORAL CHANGES
# ===========================================================================

DRA_DEBUG_EVERY = 10 #25          # print detailed diagnostics every N planner calls
DRA_DEBUG_TOP_K = 8            # show the K cheapest candidate actions
CONTROL_DEBUG_EVERY = 10 #25       # print controller geometry every N calls


# Kinematic model parameters
WHEELBASE       = 2.9    

# CARLA connection
CARLA_HOST      = 'localhost'
CARLA_PORT      = 2000
CARLA_TIMEOUT   = 15.0

# ===========================================================================
# §2  DATA STRUCTURES
# ===========================================================================

class PlannerType(str, Enum):
    TTC          = "BASELINE_TTC"
    UTILITARIAN  = "BASELINE_UTILITARIAN"
    DRA          = "FULL_DRA"


@dataclass
class ActorState:
    actor_id:    int
    actor_type:  str            
    position:    np.ndarray     
    velocity:    np.ndarray     
    speed:       float          
    visible:     bool           
    distance:    float          


@dataclass
class PlannerInput:
    ego:              ActorState
    perceived_actors: List[ActorState]   
    timestep:         float              
    elapsed_time:     float              
    step:             int
    oncoming_bound_y: float = ONCOMING_BOUND
    pavement_bound_y: float = PAVEMENT_BOUND
    lane_centre_y:    float = EGO_LANE_Y
    prev_steer: float = 0.0


@dataclass
class PlannerAction:
    target_speed:  float   
    delta_lateral: float   
    planner_name:  str = ""
    cost_computed: float = 0.0
    risk_ped:      float = 0.0
    risk_ego:      float = 0.0
    risk_maximin:  float = 0.0
    risk_equality: float = 0.0


@dataclass
class TimestepRecord:
    timestamp:      float
    scenario:       str
    seed:           int
    planner:        str
    step:           int
    ego_x:          float
    ego_y:          float
    ego_speed:      float
    ego_accel:      float
    ego_steer:      float
    ped_x:          float
    ped_y:          float
    ped_speed:      float
    ped_visible:    bool
    distance_ped:   float
    oncoming_dist:  float
    ttc:            float
    risk_ped:       float
    risk_ego:       float
    risk_maximin:   float
    risk_equality:  float
    pavement_viol:  bool
    oncoming_viol:  bool
    collision_ped:  bool


@dataclass
class EpisodeResult:
    scenario:          str
    seed:              int
    planner:           str
    collision_ped:     bool
    collision_oncoming:bool
    success:           bool       
    min_distance_ped:  float
    min_distance_onc:  float
    min_ttc:           float
    time_below_ttc2:   float      
    near_miss_count:   int        
    avg_speed:         float
    min_speed:         float
    max_decel:         float
    max_accel_g:       float
    max_jerk:          float
    path_length:       float
    travel_time:       float
    pavement_duration: float
    oncoming_duration: float
    integrated_r_ped:  float
    integrated_r_ego:  float
    peak_r_ped:        float
    peak_r_ego:        float
    peak_maximin:      float
    avg_velocity:      float
    mean_plan_ms:      float
    max_plan_ms:       float
    carla_version:     str = ""
    map_name:          str = ""
    ego_blueprint:     str = ""
    ped_blueprint:     str = ""
    config_version:    str = "v4.2"


# ===========================================================================
# §3  ROAD FRAME
# ===========================================================================

class RoadFrame:
    def __init__(self, origin_location: carla.Location, yaw_degrees: float):
        self.origin = origin_location
        yaw_rad = math.radians(yaw_degrees)
        self.forward = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
        carla_right  = carla.Vector3D(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)
        self.left    = carla.Vector3D(-carla_right.x, -carla_right.y, 0.0)
        self.yaw_deg = yaw_degrees

    def to_world(self, lx: float, ly: float, z: float = 0.5) -> carla.Location:
        wx = self.origin.x + lx * self.forward.x + ly * self.left.x
        wy = self.origin.y + lx * self.forward.y + ly * self.left.y
        return carla.Location(x=wx, y=wy, z=self.origin.z + z)

    def to_local(self, world_loc: carla.Location) -> np.ndarray:
        dx = world_loc.x - self.origin.x
        dy = world_loc.y - self.origin.y
        lx = dx * self.forward.x + dy * self.forward.y
        ly = dx * self.left.x    + dy * self.left.y
        return np.array([lx, ly])

    def world_vel_to_local(self, vx_world: float, vy_world: float) -> np.ndarray:
        vlx = vx_world * self.forward.x + vy_world * self.forward.y
        vly = vx_world * self.left.x    + vy_world * self.left.y
        return np.array([vlx, vly])

    def heading_yaw(self, dx_local: float, dy_local: float) -> float:
        wdx = dx_local * self.forward.x + dy_local * self.left.x
        wdy = dx_local * self.forward.y + dy_local * self.left.y
        if abs(wdx) < 1e-6 and abs(wdy) < 1e-6:
            return self.yaw_deg
        return -math.degrees(math.atan2(wdy, wdx))




class PedestrianState(Enum):
    NORMAL = auto()
    APPROACH_STOP = auto()
    WAIT_FOR_PEDESTRIAN = auto()
    RESUME = auto()
# ===========================================================================
# §4  SCENARIO MANAGER
# ===========================================================================

class ScenarioManager:
    SCENARIO_CATALOG = {
        "S1": {
            "name": "Occluded Crossing (CA_4_5 reference scenario)",
            "description": (
                "Pedestrian hidden behind roadside obstacle, emerges into ego path. "
                "Oncoming vehicle in opposite lane."
            ),
            "map": "Town03",
            "ego_v0": EGO_V0,
            "ped_spawn_local": PED_SPAWN_LOCAL,
            "ped_velocity": PED_VELOCITY,
            "ped_start_t": PED_START_T,
            "oncoming_spawn_local": ONCOMING_SPAWN,
            "oncoming_v_x": ONCOMING_V_X,
            "duration": SIM_DURATION,
        }
    }

    def __init__(self, client: carla.Client, scenario_id: str, seed: int,
                 frame: RoadFrame = None):
        self.client      = client
        self.world       = client.get_world()
        self.scenario_id = scenario_id
        self.seed        = seed
        self.cfg         = self.SCENARIO_CATALOG[scenario_id]
        self.frame       = frame          
        self.actors      = []             
        self.rng         = np.random.default_rng(seed)

        self.ped_actor   = None
        self.onc_actor   = None
        self.ego_actor   = None
        self.collision_sensor = None
        self.collision_ped_event = False
        self.collision_oncoming_event = False
        self.elapsed     = 0.0
        self.step_count  = 0

        self.ped_local   = self.cfg["ped_spawn_local"].copy().astype(float)
        self.ped_moving  = False
        self.onc_local   = self.cfg["oncoming_spawn_local"].copy().astype(float)

        ped_x_offset = float(self.rng.uniform(-0.5, 0.5))   
        ped_y_offset = float(self.rng.uniform(-0.2, 0.2))   
        self.ped_local[0] += ped_x_offset
        self.ped_local[1] += ped_y_offset

    def find_spawn(self, min_straight: float = 50.0) -> Tuple[carla.Location, float]:
        carla_map   = self.world.get_map()
        spawn_pts   = carla_map.get_spawn_points()
        for sp in spawn_pts:
            wp = carla_map.get_waypoint(sp.location, project_to_road=True)
            if wp is None:
                continue
            cur, ok = wp, True
            for _ in range(int(min_straight)):
                nxts = cur.next(1.0)
                if not nxts:
                    ok = False; break
                cur = nxts[0]
            if ok:
                return sp.location, sp.rotation.yaw
        return spawn_pts[0].location, spawn_pts[0].rotation.yaw

    def spawn_ego(self, bp_lib: carla.BlueprintLibrary) -> carla.Actor:
        bp = bp_lib.find('vehicle.tesla.model3')
        bp.set_attribute('role_name', 'ego')
        loc = self.frame.to_world(0.0, EGO_LANE_Y, z=0.5)
        tf  = carla.Transform(loc, carla.Rotation(yaw=self.frame.yaw_deg))
        ego = self.world.spawn_actor(bp, tf)
        ego.set_simulate_physics(True)   
        self.ego_actor = ego
        self.actors.append(ego)
        return ego

    def spawn_pedestrian(self, bp_lib: carla.BlueprintLibrary) -> carla.Actor:
        bps = bp_lib.filter('walker.pedestrian.0002')
        bp  = bps[0] if bps else bp_lib.filter('walker.pedestrian.*')[0]
        loc = self.frame.to_world(self.ped_local[0], self.ped_local[1], z=0.5)
        tf  = carla.Transform(loc, carla.Rotation(yaw=self.frame.yaw_deg - 90.0)) #<- Was +90
        ped = self.world.try_spawn_actor(bp, tf)
        if ped is None:
            loc.z += 0.3
            ped = self.world.spawn_actor(bp, carla.Transform(loc, tf.rotation))
        self.ped_actor = ped
        self.actors.append(ped)
        return ped

    def spawn_oncoming(self, bp_lib: carla.BlueprintLibrary) -> carla.Actor:
        bp  = bp_lib.find('vehicle.audi.a2')
        loc = self.frame.to_world(self.onc_local[0], self.onc_local[1], z=0.5)
        tf  = carla.Transform(loc, carla.Rotation(yaw=self.frame.yaw_deg + 180.0))
        onc = self.world.try_spawn_actor(bp, tf)
        if onc is None:
            loc.z += 0.3
            onc = self.world.spawn_actor(bp, carla.Transform(loc, tf.rotation))
        onc.set_simulate_physics(False)   
        self.onc_actor = onc
        self.actors.append(onc)
        return onc

    def tick(self):
        self.elapsed     += DT
        self.step_count  += 1

        if self.elapsed >= self.cfg["ped_start_t"]:
            self.ped_moving = True
            self.ped_local  = self.ped_local + self.cfg["ped_velocity"] * DT

        if self.ped_actor and self.ped_actor.is_alive:
            new_ped_world = self.frame.to_world(
                self.ped_local[0], self.ped_local[1], z=0.5)
            if self.ped_moving:
                self.ped_actor.set_transform(
                    carla.Transform(new_ped_world,
                                    carla.Rotation(yaw=self.frame.yaw_deg - 90.0))) #<- Was +90

        self.onc_local[0] += self.cfg["oncoming_v_x"] * DT
        if self.onc_actor and self.onc_actor.is_alive:
            new_onc_world = self.frame.to_world(
                self.onc_local[0], self.onc_local[1], z=0.5)
            self.onc_actor.set_transform(
                carla.Transform(new_onc_world,
                                carla.Rotation(yaw=self.frame.yaw_deg + 180.0)))

    def reset(self):
        self.ped_local  = self.cfg["ped_spawn_local"].copy().astype(float)
        self.onc_local  = self.cfg["oncoming_spawn_local"].copy().astype(float)
        self.ped_moving = False
        self.elapsed    = 0.0
        self.step_count = 0
        rng2            = np.random.default_rng(self.seed)
        self.ped_local[0] += float(rng2.uniform(-0.5, 0.5))
        self.ped_local[1] += float(rng2.uniform(-0.2, 0.2))

    def get_ground_truth(self) -> Dict:
        ego_loc  = self.ego_actor.get_location()   if self.ego_actor else None
        ped_loc  = self.frame.to_world(self.ped_local[0], self.ped_local[1])
        onc_loc  = self.frame.to_world(self.onc_local[0], self.onc_local[1])
        ego_v    = self.ego_actor.get_velocity()   if self.ego_actor else carla.Vector3D()
        ego_local = self.frame.to_local(ego_loc)   if ego_loc else np.array([0.0, 0.0])
        ego_v_local = self.frame.world_vel_to_local(ego_v.x, ego_v.y) if ego_v else np.zeros(2)

        ped_visible = (
            self.elapsed >= self.cfg["ped_start_t"] and
            self.ped_local[1] > PED_VISIBLE_Y
        )

        return {
            "ego_local"     : ego_local,
            "ego_v_local"   : ego_v_local,
            "ego_speed"     : float(np.linalg.norm(ego_v_local)),
            "ped_local"     : self.ped_local.copy(),
            "ped_visible"   : ped_visible,
            "onc_local"     : self.onc_local.copy(),
            "elapsed"       : self.elapsed,
            "step"          : self.step_count,
        }

    def check_termination(self, gt: Dict) -> bool:
        if self.step_count >= MAX_STEPS:
            return True
        # if self.ped_local[1] > 1.0:   #<-- Commented out for scene completness
        #     return True
        return False

    def cleanup(self):
        for actor in reversed(self.actors):
            try:
                if actor and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.actors.clear()


# ===========================================================================
# §5  PERCEPTION INTERFACE
# ===========================================================================

class PerceptionInterface:
    def build(self, scenario: ScenarioManager, prev_steer: float = 0.0) -> PlannerInput:
        gt = scenario.get_ground_truth()

        ego_state = ActorState(
            actor_id   = 0,
            actor_type = "ego",
            position   = gt["ego_local"],
            velocity   = gt["ego_v_local"],
            speed      = gt["ego_speed"],
            visible    = True,
            distance   = 0.0,
        )

        perceived = []

        ped_local = gt["ped_local"]
        ped_dist  = float(np.linalg.norm(gt["ego_local"] - ped_local))

        # Slower perception threshold: Pedestrian is only "perceived" when within 12 meters
        PERCEPTION_DISTANCE = 30.0 #8.0 #was 12.0

        if gt["ped_visible"] and ped_dist <= PERCEPTION_DISTANCE:
            perceived.append(ActorState(
                actor_id   = 1,
                actor_type = "pedestrian",
                position   = ped_local.copy(),
                velocity   = np.array(PED_VELOCITY) if scenario.ped_moving else np.zeros(2),
                speed      = float(np.linalg.norm(PED_VELOCITY)) if scenario.ped_moving else 0.0,
                visible    = True,
                distance   = ped_dist,
            ))

        onc_local = gt["onc_local"]
        onc_dist  = float(np.linalg.norm(gt["ego_local"] - onc_local))
        perceived.append(ActorState(
            actor_id   = 2,
            actor_type = "oncoming",
            position   = onc_local.copy(),
            velocity   = np.array([ONCOMING_V_X, 0.0]),
            speed      = abs(ONCOMING_V_X),
            visible    = True,
            distance   = onc_dist,
        ))

        return PlannerInput(
            ego              = ego_state,
            perceived_actors = perceived,
            timestep         = DT,
            elapsed_time     = gt["elapsed"],
            step             = gt["step"],
            prev_steer       = prev_steer,
        )


# ===========================================================================
# §6  PLANNERS
# ===========================================================================

class BasePlanner:
    def __init__(self):
        self.name = "base"

    def plan(self, inp: PlannerInput) -> PlannerAction:
        raise NotImplementedError

    def _find_pedestrian(self, inp: PlannerInput) -> Optional[ActorState]:
        for a in inp.perceived_actors:
            if a.actor_type == "pedestrian" and a.visible:
                return a
        return None

    def _find_oncoming(self, inp: PlannerInput) -> Optional[ActorState]:
        for a in inp.perceived_actors:
            if a.actor_type == "oncoming" and a.visible:
                return a
        return None

    def _compute_ttc(self, ego_pos: np.ndarray, ego_speed: float,
                     ped_pos: np.ndarray) -> float:
        dx = ped_pos[0] - ego_pos[0]
        if dx <= 0 or ego_speed < 0.1:
            return 999.0
        return dx / ego_speed


class TTCPlanner(BasePlanner):
    def __init__(self):
        super().__init__()
        self.name = "BASELINE_TTC"

    def plan(self, inp: PlannerInput) -> PlannerAction:
        ego      = inp.ego
        ped      = self._find_pedestrian(inp)
        ego_spd  = ego.speed

        if ped is not None:
            ttc = self._compute_ttc(ego.position, ego_spd, ped.position)
            if ttc < CRITICAL_TTC and ped.distance < 15.0:
                new_speed = max(0.0, ego_spd - MAX_DECEL * DT)
                return PlannerAction(
                    target_speed  = new_speed,
                    delta_lateral = 0.0,      
                    planner_name  = self.name,
                    risk_ped      = 1.0 / max(ped.distance, 0.1),
                )

        return PlannerAction(
            target_speed  = EGO_V0, # wasego_spd,
            delta_lateral = 0.0,
            planner_name  = self.name,
        )


class UtilitarianPlanner(BasePlanner):
    def __init__(self):
        super().__init__()
        self.name = "BASELINE_UTILITARIAN"

    def plan(self, inp: PlannerInput) -> PlannerAction:
        ego     = inp.ego
        ped     = self._find_pedestrian(inp)
        ego_spd = ego.speed

        if ped is None:
            return PlannerAction(
                target_speed  =  EGO_V0, #was ego_spd,
                delta_lateral = 0.0,
                planner_name  = self.name,
            )

        v_cands  = [max(0.0, ego_spd - MAX_DECEL * DT),
                    max(0.0, ego_spd - (MAX_DECEL/2) * DT),
                    ego_spd]
        dy_cands = [-0.30, -0.15, 0.0, 0.15, 0.30]# [-0.12, -0.06, 0.0, 0.06, 0.12]

        best_cost = float('inf')
        best_v    = ego_spd
        best_dy   = 0.0

        for vc in v_cands:
            for dyc in dy_cands:
                nxt = ego.position + np.array([vc * DT, dyc])
                dp  = max(np.linalg.norm(nxt - ped.position), 0.1)
                r_ped  = 200.0 / dp
                #r_prog = -vc * DT * 30.0 #was 2, 10.0, 20
                # Scale relative to max speed EGO_V0
                r_prog = -150.0 * (vc / EGO_V0) #was 50
                cost   = r_ped + r_prog
                if cost < best_cost:
                    best_cost = cost
                    best_v    = vc
                    best_dy   = dyc

        return PlannerAction(
            target_speed  = best_v,
            delta_lateral = best_dy,
            planner_name  = self.name,
            cost_computed = best_cost,
            risk_ped      = 200.0 / max(np.linalg.norm(
                                ego.position - ped.position), 0.1),
        )


class DRAPlanner(BasePlanner):
    def __init__(self, horizon_ticks: int = DRA_HORIZON_TICKS):
        super().__init__()
        self.name = "FULL_DRA"
        # ================================================================
        # PEDESTRIAN STOP / WAIT STATE
        # ================================================================
        # self.ped_state = PedestrianState.NORMAL

        # self.STOP_SPEED = 0.20
        # self.RESUME_SPEED = 1.0

        # self.STOP_DISTANCE = 12.0
        # # Emergency braking is deliberately conservative: this is a hard safety
        # # guard, not part of the soft DRA cost optimization.
        self.EMERGENCY_BRAKE_DISTANCE = 8.0   # [m] longitudinal pedestrian gap
        self.EMERGENCY_LATERAL_GAP = 1.75     # [m] lateral separation threshold
        self.EMERGENCY_STOP_SPEED = 0.0
        self.RESUME_DISTANCE = 8.0

        # self.MIN_STOP_TIME = 1.0
        # self.stop_start_time = None
        self.ped_state = PedestrianState.NORMAL
        self.STOP_SPEED = 0.20
        self.RESUME_SPEED = 1.0
        self.STOP_DISTANCE = 12.0
        # self.PED_CLEAR_SPEED = 0.5
        self.RESUME_DISTANCE = 8.0

        # Gentle lane-centering during RESUME.
        # This actively brings the ego vehicle back to the original lane
        # instead of merely waiting for the previous steering to unwind.
        self.RESUME_LANE_GAIN = 0.8
        self.RESUME_MAX_DY = 0.15
        self.RESUME_LANE_TOL = 0.05

        self.MIN_STOP_TIME = 1.0
        self.stop_start_time = None

        self.PED_CLEAR_SPEED = 0.5
        # v4.2: react before the pedestrian reaches the immediate 5 m zone.
        # The previous 5 m threshold was too late for a 0.15 rad/s steer-rate limit.

        # ================================================================
        # COST-SCALE ANALYSIS — DIAGNOSTIC ONLY
        # ================================================================
        self.cost_scale_samples = {
            "r_lane": [],
            "avg_r_ped": [],
            #"avg_r_ego": [],
            "avg_r_oncoming": [],
            "r_maximin": [],
            "r_equality": [],
            "r_prog": [],
            "weighted_ped": [],
            "weighted_maximin": [],
            "weighted_equality": [],
            "total_cost": [],
        }

        # self.cost_scale_samples = {
        #                 "r_lane": [],
        #                 "avg_r_ped": [],
        #                 "avg_r_oncoming": [],
        #                 "r_prog": [],
        #                 "total_cost": [],
        #             }

        # self.cost_scale_samples = {
        #     "all": {
        #         "r_lane": [],
        #         "avg_r_ped": [],
        #         "avg_r_ego": [],
        #         "r_maximin": [],
        #         "r_equality": [],
        #         "r_prog": [],
        #         "weighted_ped": [],
        #         "weighted_maximin": [],
        #         "weighted_equality": [],
        #         "total_cost": [],
        #     },
        #     "valid": {
        #         "r_lane": [],
        #         "avg_r_ped": [],
        #         "avg_r_ego": [],
        #         "r_maximin": [],
        #         "r_equality": [],
        #         "r_prog": [],
        #         "weighted_ped": [],
        #         "weighted_maximin": [],
        #         "weighted_equality": [],
        #         "total_cost": [],
        #     },
        # }

        self.horizon_ticks = horizon_ticks
        self.debug_counter = 0

    # def rollout_candidate(
    #     self,
    #     vc: float,
    #     dyc: float,
    #     ego_pos: np.ndarray,
    #     prev_steer: float,
    #     ped: Optional[ActorState],
    #     inp: PlannerInput
    # ) -> Tuple[np.ndarray, float, float, float, float, bool]:
    #     """
    #     Simulates candidate forward over horizon_ticks.
    #     Returns: (final_pos, worst_ped_dist, integrated_r_ped, integrated_r_ego, min_lane_margin, hard_violation)
    #     """
    #     pos = ego_pos.copy()
    #     steer = prev_steer
    #     worst_ped_dist = float('inf')

    #     # Accumulator for risk terms over the horizon
    #     sum_r_ped = 0.0
    #     sum_r_ego = 0.0
    #     hard_violation = False

    #     # Predict pedestrian linear velocity forward
    #     ped_pos_sim = ped.position.copy() if ped is not None else None
    #     ped_vel_sim = ped.velocity if ped is not None else np.zeros(2)

    #     for _ in range(self.horizon_ticks):
    #         # 1. Integrate physical steering limit tick-by-tick
    #         achievable_dy = achievable_dy_from_requested_dy(
    #             requested_dy=dyc,
    #             current_speed=vc,
    #             prev_steer=steer
    #         )
            
    #         # Update virtual steer state for the next step in rollout
    #         fwd_dist = max(vc * DT, 0.01)
    #         steer_rad = -math.atan2(achievable_dy, fwd_dist)
    #         steer = float(max(-MAX_STEER, min(MAX_STEER, steer_rad / (math.pi / 4.0))))

    #         # 2. Advance simulated positions
    #         pos = pos + np.array([vc * DT, achievable_dy])
    #         if ped_pos_sim is not None:
    #             ped_pos_sim = ped_pos_sim + ped_vel_sim * DT
    #             dist = np.linalg.norm(pos - ped_pos_sim)
    #             worst_ped_dist = min(worst_ped_dist, dist)

    #             # Horizon risk accumulation
    #             dp = max(dist, 0.1)
    #             sum_r_ped += 350.0 * math.exp(-dp**2 / (2.0 * PED_RISK_SIGMA**2))
    #             sum_r_ego += 50.0 / dp

    #         # 3. Check hard spatial boundaries (Pavement / Oncoming)
    #         if pos[1] < inp.pavement_bound_y or pos[1] > inp.oncoming_bound_y:
    #             hard_violation = True

    #     # Check hard pedestrian collision radius anywhere along rollout
    #     if ped is not None and worst_ped_dist < COLLISION_RADIUS:
    #         hard_violation = True

    #     return pos, worst_ped_dist, sum_r_ped, sum_r_ego, hard_violation
    # def rollout_candidate(
    #     self,
    #     vc: float,
    #     dyc: float,
    #     ego_pos: np.ndarray,
    #     prev_steer: float,
    #     ped: Optional[ActorState],
    #     inp: PlannerInput
    # ) -> Tuple[np.ndarray, float, float, float, float, bool]:
    def rollout_candidate(
            self,
            vc: float,
            dyc: float,
            ego_pos: np.ndarray,
            prev_steer: float,
            ped: Optional[ActorState],
            oncoming: Optional[ActorState],
            inp: PlannerInput
        )-> Tuple[np.ndarray, float, float, float, bool]:
        pos = ego_pos.copy()
        steer = prev_steer
        worst_ped_dist = float('inf')
        worst_oncoming_dist = float('inf')

        sum_r_ped = 0.0
        #sum_r_ego = 0.0
        sum_r_oncoming = 0.0
        hard_violation = False

        ped_pos_sim = ped.position.copy() if ped is not None else None
        ped_vel_sim = ped.velocity if ped is not None else np.zeros(2)

        onc_pos_sim = (
            oncoming.position.copy()
            if oncoming is not None
            else None
        )

        onc_vel_sim = (
            oncoming.velocity
            if oncoming is not None
            else np.zeros(2)
        )

        for _ in range(self.horizon_ticks):
            fwd_dist = max(vc * DT, 0.01)

            # 1. Direct Steering-Rate Transition Matching ControlAdapter
            requested_steer_rad = -math.atan2(dyc, fwd_dist)
            requested_steer_norm = requested_steer_rad / (math.pi / 4.0)
            clipped_steer_norm = float(max(-MAX_STEER, min(MAX_STEER, requested_steer_norm)))

            max_delta_steer = MAX_STEER_RATE * DT / (math.pi / 4.0)
            
            # Rate limit state transition directly
            steer = float(
                max(
                    steer - max_delta_steer,
                    min(steer + max_delta_steer, clipped_steer_norm)
                )
            )

            # Convert resulting steer to lateral motion
            feasible_steer_rad = steer * (math.pi / 4.0)
            achievable_dy = -fwd_dist * math.tan(feasible_steer_rad)

            # 2. Advance simulated positions
            pos = pos + np.array([vc * DT, achievable_dy])
            # ------------------------------------------------------------
            # Pedestrian
            # ------------------------------------------------------------
            if ped_pos_sim is not None:
                ped_pos_sim = ped_pos_sim + ped_vel_sim * DT
                dist = np.linalg.norm(pos - ped_pos_sim)
                worst_ped_dist = min(worst_ped_dist, dist)

                dp = max(dist, 0.1)
                sum_r_ped += 350.0 * math.exp(-dp**2 / (2.0 * PED_RISK_SIGMA**2))
                #sum_r_ego += 50.0 / dp

            # ------------------------------------------------------------
            # Oncoming vehicle
            # ------------------------------------------------------------
            if onc_pos_sim is not None:
                onc_pos_sim = onc_pos_sim + onc_vel_sim * DT
                onc_dist = np.linalg.norm(pos - onc_pos_sim)
                worst_oncoming_dist = min(
                worst_oncoming_dist,
                onc_dist
            )
                sum_r_oncoming += 70.0 / max(onc_dist, 0.1) #was 50

                if onc_dist < COLLISION_RADIUS:
                    hard_violation = True

            # ------------------------------------------------------------
            # Hard spatial boundaries
            # ------------------------------------------------------------    
            # 3. Check hard spatial boundaries (Pavement / Oncoming)
            if pos[1] < inp.pavement_bound_y or pos[1] > inp.oncoming_bound_y:
                hard_violation = True

        if ped is not None and worst_ped_dist < COLLISION_RADIUS:
            hard_violation = True

        return pos, worst_ped_dist, sum_r_ped, sum_r_oncoming, worst_oncoming_dist, hard_violation #return pos, worst_ped_dist, sum_r_ped, sum_r_ego, hard_violation

    # def _pedestrian_requires_stop(self, ego, ped):
    #     if ped is None:
    #         return False

    #     dx = ped.position[0] - ego.position[0]

    #     # Behind us
    #     if dx <= 0:
    #         return False

    #     # Too far away
    #     if dx > self.STOP_DISTANCE:
    #         return False

    #     # Lateral separation
    #     dy = abs(ped.position[1] - ego.position[1])

    #     # Pedestrian not actually in our path
    #     if dy > 1.5:
    #         return False

    #     return True
    
    def _pedestrian_distance(self, ego, ped):
        if ped is None:
            return float("inf")

        return float(np.linalg.norm(
            ego.position - ped.position
        ))

    def _pedestrian_longitudinal_distance(self, ego, ped):
        if ped is None:
            return float("inf")

        dx = ped.position[0] - ego.position[0]

        if dx < 0:
            return float("inf")

        return float(dx)

    def _pedestrian_requires_stop(self, ego, ped):
        if ped is None:
            return False

        dx = ped.position[0] - ego.position[0]

        # Behind us
        if dx <= 0:
            return False

        # Too far away
        if dx > self.STOP_DISTANCE:
            return False

        # Lateral separation
        dy = abs(ped.position[1] - ego.position[1])

        # Pedestrian not actually in our path
        if dy > 1.5:
            return False

        return True


    def plan(self, inp: PlannerInput) -> PlannerAction:
        ego = inp.ego
        ped = self._find_pedestrian(inp)
        oncoming = self._find_oncoming(inp)
        ego_spd = ego.speed
        prev_steer = inp.prev_steer
        ped_dist = self._pedestrian_distance(ego, ped)
        ped_long_dist = self._pedestrian_longitudinal_distance(ego, ped)

        # ============================================================
        # PEDESTRIAN STOP STATE MACHINE
        # ============================================================

        if self.ped_state == PedestrianState.NORMAL:

            if (
                self._pedestrian_requires_stop(ego, ped)
                and ego_spd > self.STOP_SPEED
            ):
                self.ped_state = PedestrianState.APPROACH_STOP
                self.stop_start_time = None

        elif self.ped_state == PedestrianState.APPROACH_STOP:

            # If the pedestrian has already passed the ego longitudinally,
            # do not keep the stop state alive with ped_long_dist=inf.
            if ped is None or ped_long_dist == float("inf"):
                self.ped_state = PedestrianState.RESUME
                self.stop_start_time = None

            # Once vehicle has essentially stopped
            elif ego_spd <= self.STOP_SPEED:

                self.ped_state = PedestrianState.WAIT_FOR_PEDESTRIAN

                self.stop_start_time = time.time()


        elif self.ped_state == PedestrianState.WAIT_FOR_PEDESTRIAN:

            # ========================================================
            # SURGICAL FIX (v4.3.2): WAIT_FOR_PEDESTRIAN → RESUME TRANSITION
            # ========================================================
            # Checks if the pedestrian is clear (either hidden/passed, or
            # outside path laterally, or past the longitudinal threshold).
            ped_in_path = self._pedestrian_requires_stop(ego, ped)

            if ped is None:
                self.ped_state = PedestrianState.RESUME
            elif not ped_in_path:
                self.ped_state = PedestrianState.RESUME
            elif ped_long_dist > self.RESUME_DISTANCE:
                self.ped_state = PedestrianState.RESUME


        elif self.ped_state == PedestrianState.RESUME:

            # Once we're moving again, return to normal DRA
            if ego_spd > self.RESUME_SPEED:
                self.ped_state = PedestrianState.NORMAL    
        self.debug_counter += 1
        debug_this_step = (self.debug_counter % DRA_DEBUG_EVERY == 0)

        v_cands = [
            max(0.0, ego_spd - MAX_DECEL * DT),
            max(0.0, ego_spd - (MAX_DECEL / 2.0) * DT),
            ego_spd,
            min(EGO_V0, ego_spd + MAX_ACCEL * DT),
            min(EGO_V0, ego_spd + 0.5),
            min(EGO_V0, ego_spd + 1.0),
        ]

        dy_cands = [-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25]

        valid_candidates = []
        fallback_candidates = []

        for vc in v_cands:
            for dyc in dy_cands:
                final_pos, worst_ped_dist, sum_r_ped, sum_r_oncoming, worst_oncoming_dist, hard_violation = self.rollout_candidate(
                    vc, dyc, ego.position, prev_steer, ped, oncoming, inp
                )

                # Horizon-integrated costs
                r_lane = 5.0 * (final_pos[1] - inp.lane_centre_y) ** 2
                r_prog = -PROGRESS_COST_WEIGHT * (vc / EGO_V0)

                avg_r_ped = sum_r_ped / self.horizon_ticks
                avg_r_oncoming = sum_r_oncoming / self.horizon_ticks
                r_maximin = max(avg_r_ped, avg_r_oncoming)
                r_equality = abs(avg_r_ped - avg_r_oncoming)

                # Soft cost scalarization
                cost = (
                    r_lane
                    + PED_RISK_WEIGHT * avg_r_ped
                    + 0.6 * r_maximin
                    + 0.2 * r_equality
                    + r_prog
                )

                self.cost_scale_samples["r_lane"].append(float(r_lane))
                self.cost_scale_samples["avg_r_ped"].append(float(avg_r_ped))
                self.cost_scale_samples["avg_r_oncoming"].append(float(avg_r_oncoming))
                self.cost_scale_samples["r_maximin"].append(float(r_maximin))
                self.cost_scale_samples["r_equality"].append(float(r_equality))
                self.cost_scale_samples["r_prog"].append(float(r_prog))
                self.cost_scale_samples["weighted_ped"].append(float(PED_RISK_WEIGHT * avg_r_ped))
                self.cost_scale_samples["weighted_maximin"].append(float(0.6 * r_maximin))
                self.cost_scale_samples["weighted_equality"].append(float(0.2 * r_equality))
                self.cost_scale_samples["total_cost"].append(float(cost))

                cand_data = {
                    "vc": vc,
                    "dy": dyc,
                    "cost": cost,
                    "worst_dist": worst_ped_dist,
                    "worst_oncoming_dist": worst_oncoming_dist,
                    "r_ped": avg_r_ped,
                    "r_oncoming": avg_r_oncoming,
                    "r_maximin": r_maximin,
                    "r_equality": r_equality,
                    "hard_violation": hard_violation
                }

                if not hard_violation:
                    valid_candidates.append(cand_data)
                fallback_candidates.append(cand_data)

        # Apply Hard-Constraint Selection Filter
        if valid_candidates:
            # Rank surviving safe candidates purely by soft cost
            best = min(valid_candidates, key=lambda c: c["cost"])

            # ============================================================
            # BEHAVIORAL OVERRIDE: PEDESTRIAN STOP
            # ============================================================

            if self.ped_state == PedestrianState.APPROACH_STOP:
                # Desired speed based on remaining distance
                d = max(ped_long_dist, 0.0)

                comfortable_decel = 2.0  # m/s^2

                stop_target = np.sqrt(
                    max(0.0, 2.0 * comfortable_decel * d)
                )

                best["vc"] = min(best["vc"], stop_target)
                best["dy"] = 0.0

            elif self.ped_state == PedestrianState.WAIT_FOR_PEDESTRIAN:

                # Remain stopped
                best["vc"] = 0.0
                best["dy"] = 0.0

            elif self.ped_state == PedestrianState.RESUME:
                # Resume forward motion while actively returning toward
                # the original ego lane centre.
                best["vc"] = min(
                    EGO_V0,
                    max(1.0, ego_spd + MAX_ACCEL * DT)
                )

                lane_error = inp.lane_centre_y - ego.position[1]

                if abs(lane_error) <= self.RESUME_LANE_TOL:
                    best["dy"] = 0.0
                else:
                    best["dy"] = float(np.clip(
                        self.RESUME_LANE_GAIN * lane_error,
                        -self.RESUME_MAX_DY,
                        self.RESUME_MAX_DY
                    ))
            # elif self.ped_state == PedestrianState.RESUME:

            #     # Give the controller a gentle initial target.
            #     # The DRA will take over once speed builds.
            #     best["vc"] = min(EGO_V0, max(1.0, ego_spd + MAX_ACCEL * DT))

            #     best["dy"] = 0.0

            if debug_this_step and abs(best["dy"]) > 1e-9:
                print("\n========== DRA LATERAL DECISION ==========")

                for c in sorted(valid_candidates, key=lambda x: x["cost"])[:10]:
                    print(
                        f"vc={c['vc']:.3f} "
                        f"dy={c['dy']:+.2f} "
                        f"cost={c['cost']:+.4f} "
                        f"r_ped={c['r_ped']:.4f} "
                        f"r_onc={c['r_oncoming']:.4f} "
                        f"r_maximin={c['r_maximin']:.4f} "
                        f"r_equal={c['r_equality']:.4f} "
                        f"worst_dist={c['worst_dist']:.3f}"
                    )

                print("===========================================\n")

        else:
            # Fallback: if all options violate hard bounds, pick the candidate
            # with the best immediate worst-case clearance.
            best = max(
                fallback_candidates,
                key=lambda c: min(
                    c["worst_dist"],
                    c["worst_oncoming_dist"]
                )
            )

        # ================================================================
        # HARD PEDESTRIAN EMERGENCY BRAKE
        # ================================================================
        emergency_stop = False
        if ped is not None and ped.visible:
            dx = ped.position[0] - ego.position[0]
            lateral_gap = abs(ped.position[1] - ego.position[1])
            if (0.0 < dx <= self.EMERGENCY_BRAKE_DISTANCE
                    and lateral_gap <= self.EMERGENCY_LATERAL_GAP
                    and ego_spd > self.STOP_SPEED):
                emergency_stop = True
                best["vc"] = self.EMERGENCY_STOP_SPEED
                best["dy"] = 0.0

        if debug_this_step and emergency_stop:
            print(
                f"[DRA SAFETY OVERRIDE] EMERGENCY STOP: "
                f"dx={ped.position[0] - ego.position[0]:.3f}m, "
                f"lateral_gap={abs(ped.position[1] - ego.position[1]):.3f}m, "
                f"ego_speed={ego_spd:.3f}m/s -> target_speed=0.0"
            )

        if debug_this_step:
            ped_dist = np.linalg.norm(ego.position - ped.position) if ped else None
            ped_dist_str = f"{ped_dist:.2f}" if ped_dist is not None else "None"

            print(
                f"[DRA DEBUG] ego_x={ego.position[0]:.2f}, ego_y={ego.position[1]:.2f}, "
                f"ego_speed={ego_spd:.2f}, ped_dist={ped_dist_str}"
            )
            print(
                f"[DRA DECISION] target_speed={best['vc']:.3f}, dy={best['dy']:+.2f}, "
                f"cost={best['cost']:.2f}, valid_options={len(valid_candidates)}/{len(fallback_candidates)}"
            )

            print(
                f"[PED STATE] state={self.ped_state.name}, "
                f"ped_long_dist={ped_long_dist:.3f}, "
                f"ped_dist={ped_dist:.3f}"
                if ped_dist is not None
                else
                    f"[PED STATE] state={self.ped_state.name}, "
                    f"ped_long_dist={ped_long_dist:.3f}, "
                    f"ped_dist=None"
            )

        return PlannerAction(
            target_speed=best["vc"],
            delta_lateral=best["dy"],
            planner_name=self.name,
            cost_computed=best["cost"],
            risk_ped=best["r_ped"],
            risk_ego=best["r_oncoming"],
            risk_maximin=best["r_maximin"],
            risk_equality=best["r_equality"],
        )

    # def plan(self, inp: PlannerInput) -> PlannerAction:
    #     ego = inp.ego
    #     ped = self._find_pedestrian(inp)
    #     oncoming = self._find_oncoming(inp)
    #     ego_spd = ego.speed
    #     prev_steer = inp.prev_steer
    #     ped_dist = self._pedestrian_distance(ego, ped)
    #     ped_long_dist = self._pedestrian_longitudinal_distance(ego, ped)

    #     # ============================================================
    #     # PEDESTRIAN STOP STATE MACHINE
    #     # ============================================================

    #     # if self.ped_state == PedestrianState.NORMAL:

    #     #     if (
    #     #         ped is not None
    #     #         and ped_long_dist < self.STOP_DISTANCE
    #     #         and ego_spd > self.STOP_SPEED
    #     #     ):
    #     #         self.ped_state = PedestrianState.APPROACH_STOP
    #     #         self.stop_start_time = None
    #     if self.ped_state == PedestrianState.NORMAL:

    #         if (
    #             self._pedestrian_requires_stop(ego, ped)
    #             and ego_spd > self.STOP_SPEED
    #         ):
    #             self.ped_state = PedestrianState.APPROACH_STOP
    #             self.stop_start_time = None

    #     elif self.ped_state == PedestrianState.APPROACH_STOP:

    #         # If the pedestrian has already passed the ego longitudinally,
    #         # do not keep the stop state alive with ped_long_dist=inf.
    #         if ped is None or ped_long_dist == float("inf"):
    #             self.ped_state = PedestrianState.RESUME
    #             self.stop_start_time = None

    #         # Once vehicle has essentially stopped
    #         elif ego_spd <= self.STOP_SPEED:

    #             self.ped_state = PedestrianState.WAIT_FOR_PEDESTRIAN

    #             self.stop_start_time = time.time()


    #     elif self.ped_state == PedestrianState.WAIT_FOR_PEDESTRIAN:

    #         # Pedestrian disappeared
    #         if ped is None:
    #             self.ped_state = PedestrianState.RESUME

    #         # Pedestrian has moved sufficiently far away
    #         elif ped_long_dist > self.RESUME_DISTANCE:
    #             self.ped_state = PedestrianState.RESUME


    #     elif self.ped_state == PedestrianState.RESUME:

    #         # Once we're moving again, return to normal DRA
    #         if ego_spd > self.RESUME_SPEED:
    #             self.ped_state = PedestrianState.NORMAL    
    #     self.debug_counter += 1
    #     debug_this_step = (self.debug_counter % DRA_DEBUG_EVERY == 0)

    #     v_cands = [
    #         max(0.0, ego_spd - MAX_DECEL * DT),
    #         max(0.0, ego_spd - (MAX_DECEL / 2.0) * DT),
    #         ego_spd,
    #         min(EGO_V0, ego_spd + MAX_ACCEL * DT),
    #         min(EGO_V0, ego_spd + 0.5),
    #         min(EGO_V0, ego_spd + 1.0),
    #     ]

    #     dy_cands = [-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25]

    #     valid_candidates = []
    #     fallback_candidates = []

    #     for vc in v_cands:
    #         for dyc in dy_cands:
    #             # final_pos, worst_ped_dist, sum_r_ped, sum_r_ego, hard_violation = self.rollout_candidate(
    #             #     vc, dyc, ego.position, prev_steer, ped, oncoming, inp
    #             # )
    #             final_pos, worst_ped_dist, sum_r_ped, sum_r_oncoming, worst_oncoming_dist, hard_violation = self.rollout_candidate(
    #                 vc, dyc, ego.position, prev_steer, ped, oncoming, inp
    #             )

    #             # Horizon-integrated costs
    #             r_lane = 5.0 * (final_pos[1] - inp.lane_centre_y) ** 2
    #             r_prog = -PROGRESS_COST_WEIGHT * (vc / EGO_V0)

    #             avg_r_ped = sum_r_ped / self.horizon_ticks
    #             #avg_r_ego = sum_r_ego / self.horizon_ticks
    #             avg_r_oncoming = sum_r_oncoming / self.horizon_ticks
    #             # r_maximin = max(avg_r_ped, avg_r_ego)
    #             # r_equality = abs(avg_r_ped - avg_r_ego)
    #             r_maximin = max(avg_r_ped, avg_r_oncoming)
    #             r_equality = abs(avg_r_ped - avg_r_oncoming)

    #             # Soft cost scalarization
    #             cost = (
    #                 r_lane
    #                 + PED_RISK_WEIGHT * avg_r_ped
    #                 + 0.6 * r_maximin
    #                 + 0.2 * r_equality
    #                 + r_prog
    #             )

    #             # ========================================================
    #             # COST-SCALE ANALYSIS — DIAGNOSTIC ONLY
    #             # ========================================================
    #         #     self.cost_scale_samples = {
    #         #     "r_lane": [],
    #         #     "avg_r_ped": [],
    #         #     "avg_r_oncoming": [],
    #         #     "r_prog": [],
    #         #     "total_cost": [],
    #         # }
    #             self.cost_scale_samples["r_lane"].append(
    #                 float(r_lane)
    #             )

    #             self.cost_scale_samples["avg_r_ped"].append(
    #                 float(avg_r_ped)
    #             )

    #             self.cost_scale_samples["avg_r_oncoming"].append(
    #             float(avg_r_oncoming)
    #             )
                
    #             #self.cost_scale_samples["avg_r_ego"].append(
    #             #    float(avg_r_ego)
    #             #)

    #             self.cost_scale_samples["r_maximin"].append(
    #                 float(r_maximin)
    #             )

    #             self.cost_scale_samples["r_equality"].append(
    #                 float(r_equality)
    #             )

    #             self.cost_scale_samples["r_prog"].append(
    #                 float(r_prog)
    #             )

    #             self.cost_scale_samples["weighted_ped"].append(
    #                 float(PED_RISK_WEIGHT * avg_r_ped)
    #             )

    #             self.cost_scale_samples["weighted_maximin"].append(
    #                 float(0.6 * r_maximin)
    #             )

    #             self.cost_scale_samples["weighted_equality"].append(
    #                 float(0.2 * r_equality)
    #             )

    #             self.cost_scale_samples["total_cost"].append(
    #                 float(cost)
    #             )


    #             cand_data = {
    #                 "vc": vc,
    #                 "dy": dyc,
    #                 "cost": cost,
    #                 "worst_dist": worst_ped_dist,
    #                 "worst_oncoming_dist": worst_oncoming_dist,
    #                 "r_ped": avg_r_ped,
    #                 "r_oncoming": avg_r_oncoming,
    #                 #"r_ego": avg_r_ego,
    #                 "r_maximin": r_maximin,
    #                 "r_equality": r_equality,
    #                 "hard_violation": hard_violation
    #             }

    #             if not hard_violation:
    #                 valid_candidates.append(cand_data)
    #             fallback_candidates.append(cand_data)

    #     # Apply Hard-Constraint Selection Filter
    #     if valid_candidates:
    #         # Rank surviving safe candidates purely by soft cost
    #         best = min(valid_candidates, key=lambda c: c["cost"])

    #         # ============================================================
    #         # BEHAVIORAL OVERRIDE: PEDESTRIAN STOP
    #         # ============================================================

    #         # if self.ped_state == PedestrianState.APPROACH_STOP:

    #         #     # Force the vehicle to stop
    #         #     best["vc"] = 0.0

    #         #     # Keep lane centered
    #         #     best["dy"] = 0.0

    #         if self.ped_state == PedestrianState.APPROACH_STOP:
    #             # Desired speed based on remaining distance
    #             d = max(ped_long_dist, 0.0)

    #             # Stop comfortably as we approach the pedestrian.
    #             #
    #             # v^2 = 2*a*d
    #             #
    #             # Therefore:
    #             # v = sqrt(2*a*d)

    #             comfortable_decel = 2.0  # m/s^2

    #             stop_target = np.sqrt(
    #                 max(0.0, 2.0 * comfortable_decel * d)
    #             )

    #             best["vc"] = min(best["vc"], stop_target)
    #             best["dy"] = 0.0

    #         elif self.ped_state == PedestrianState.WAIT_FOR_PEDESTRIAN:

    #             # Remain stopped
    #             best["vc"] = 0.0
    #             best["dy"] = 0.0

    #         elif self.ped_state == PedestrianState.RESUME:

    #             # Give the controller a gentle initial target.
    #             # The DRA will take over once speed builds.
    #             best["vc"] = min(EGO_V0, max(1.0, ego_spd + MAX_ACCEL * DT))

    #             best["dy"] = 0.0

    #         if debug_this_step and abs(best["dy"]) > 1e-9:
    #             print("\n========== DRA LATERAL DECISION ==========")

    #             for c in sorted(valid_candidates, key=lambda x: x["cost"])[:10]:
    #                 print(
    #                     f"vc={c['vc']:.3f} "
    #                     f"dy={c['dy']:+.2f} "
    #                     f"cost={c['cost']:+.4f} "
    #                     f"r_ped={c['r_ped']:.4f} "
    #                     f"r_onc={c['r_oncoming']:.4f} "
    #                     f"r_maximin={c['r_maximin']:.4f} "
    #                     f"r_equal={c['r_equality']:.4f} "
    #                     f"worst_dist={c['worst_dist']:.3f}"
    #                 )

    #             print("===========================================\n")

    #     else:
    #         # Fallback: if all options violate hard bounds, pick the candidate
    #         # with the best immediate worst-case clearance.
    #         best = max(
    #             fallback_candidates,
    #             key=lambda c: min(
    #                 c["worst_dist"],
    #                 c["worst_oncoming_dist"]
    #             )
    #         )

    #     # ================================================================
    #     # HARD PEDESTRIAN EMERGENCY BRAKE
    #     # ================================================================
    #     # IMPORTANT: this guard is intentionally OUTSIDE the valid/fallback
    #     # branches.  v4.2 previously applied the APPROACH_STOP override only
    #     # when valid_candidates was non-empty.  In the critical geometry the
    #     # planner reported 0/42 valid options, entered fallback, and returned
    #     # target_speed=8 m/s even though the pedestrian was ~2 m away.
    #     #
    #     # We do not optimize this guard.  It is a safety invariant: if a
    #     # visible pedestrian is directly in/near the ego lane and the
    #     # longitudinal gap is inside the emergency zone, braking wins over
    #     # progress and over the soft DRA objective.
    #     emergency_stop = False
    #     if ped is not None and ped.visible:
    #         dx = ped.position[0] - ego.position[0]
    #         lateral_gap = abs(ped.position[1] - ego.position[1])
    #         if (0.0 < dx <= self.EMERGENCY_BRAKE_DISTANCE
    #                 and lateral_gap <= self.EMERGENCY_LATERAL_GAP #and lateral_gap <= self.EMERGENCY_LATERAL_MARGIN
    #                 and ego_spd > self.STOP_SPEED):
    #             emergency_stop = True
    #             best["vc"] = self.EMERGENCY_STOP_SPEED
    #             best["dy"] = 0.0

    #     if debug_this_step and emergency_stop:
    #         print(
    #             f"[DRA SAFETY OVERRIDE] EMERGENCY STOP: "
    #             f"dx={ped.position[0] - ego.position[0]:.3f}m, "
    #             f"lateral_gap={abs(ped.position[1] - ego.position[1]):.3f}m, "
    #             f"ego_speed={ego_spd:.3f}m/s -> target_speed=0.0"
    #         )

    #     if debug_this_step:
    #         ped_dist = np.linalg.norm(ego.position - ped.position) if ped else None
    #         # print(
    #         #     f"[DRA DEBUG] ego_x={ego.position[0]:.2f}, ego_y={ego.position[1]:.2f}, "
    #         #     f"ego_speed={ego_spd:.2f}, ped_dist={ped_dist if ped_dist else 'None':.2f}"
    #         # )
    #         # Fix  Format inside the inline block
    #         ped_dist_str = f"{ped_dist:.2f}" if ped_dist is not None else "None"

    #         print(
    #             f"[DRA DEBUG] ego_x={ego.position[0]:.2f}, ego_y={ego.position[1]:.2f}, "
    #             f"ego_y={ego.position[1]:.2f}, "
    #             f"ego_speed={ego_spd:.2f}, ped_dist={ped_dist_str}"
    #             f"ped_dist={ped_dist_str}"
    #         )
    #         print(
    #             f"[DRA DECISION] target_speed={best['vc']:.3f}, dy={best['dy']:+.2f}, "
    #             f"dy={best['dy']:+.2f}, "
    #             f"cost={best['cost']:.2f}, valid_options={len(valid_candidates)}/{len(fallback_candidates)}"
    #             f"valid_options={len(valid_candidates)}/{len(fallback_candidates)}"
    #         )

    #         print(
    #             f"[PED STATE] state={self.ped_state.name}, "
    #             f"ped_long_dist={ped_long_dist:.3f}, "
    #             f"ped_dist={ped_dist:.3f}"
    #             if ped_dist is not None
    #             else
    #                 f"[PED STATE] state={self.ped_state.name}, "
    #                 f"ped_long_dist={ped_long_dist:.3f}, "
    #                 f"ped_dist=None"
    #         )

    #     return PlannerAction(
    #         target_speed=best["vc"],
    #         delta_lateral=best["dy"],
    #         planner_name=self.name,
    #         cost_computed=best["cost"],
    #         risk_ped=best["r_ped"],
    #         #risk_ego=best["r_ego"],
    #         risk_ego=best["r_oncoming"],
    #         risk_maximin=best["r_maximin"],
    #         risk_equality=best["r_equality"],
    #     )

    def print_cost_scale_report(self):
        print("\n" + "=" * 90)
        print("[DRA COST SCALE ANALYSIS]")
        print("=" * 90)

        for name, values in self.cost_scale_samples.items():

            if not values:
                print(f"{name:20s}: no samples")
                continue

            x = np.asarray(values, dtype=float)

            print(
                f"{name:20s} "
                f"n={len(x):6d}  "
                f"min={np.min(x):10.4f}  "
                f"median={np.median(x):10.4f}  "
                f"mean={np.mean(x):10.4f}  "
                f"p95={np.percentile(x, 95):10.4f}  "
                f"max={np.max(x):10.4f}"
            )

        print("=" * 90 + "\n")

def make_planner(planner_type: PlannerType) -> BasePlanner:
    if planner_type == PlannerType.TTC:
        return TTCPlanner()
    elif planner_type == PlannerType.UTILITARIAN:
        return UtilitarianPlanner()
    elif planner_type == PlannerType.DRA:
        return DRAPlanner()
    raise ValueError(f"Unknown planner: {planner_type}")


# ===========================================================================
# §7  CONTROL ADAPTER
# ===========================================================================

class ControlAdapter:
    def __init__(self):
        self.prev_steer = 0.0

        # DEBUG ONLY
        self.debug_counter = 0

    def reset(self):
        self.prev_steer = 0.0

    Kp_speed = 0.8   

    def convert(self, action: PlannerAction, current_speed: float) -> carla.VehicleControl:
        self.debug_counter += 1
        debug_this_step = (
            self.debug_counter % CONTROL_DEBUG_EVERY == 0
        )
        ctrl = carla.VehicleControl()
        ctrl.hand_brake = False
        ctrl.manual_gear_shift = False

        speed_err = action.target_speed - current_speed
        if speed_err > 0:
            ctrl.throttle = float(min(self.Kp_speed * speed_err, 1.0))
            ctrl.brake    = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake    = float(min(-self.Kp_speed * speed_err, 1.0))

        # fwd_dist = max(current_speed * DT, 0.01)
        # steer_rad = -math.atan2(action.delta_lateral, fwd_dist)
        # steer_norm = float(steer_rad / (math.pi / 4.0))

        # steer_norm = float(
        #     max(-MAX_STEER, min(MAX_STEER, steer_norm))
        # )

        # max_delta_steer = (
        #     MAX_STEER_RATE * DT / (math.pi / 4.0)
        # )

        # steer_norm = float(
        #     max(
        #         self.prev_steer - max_delta_steer,
        #         min(self.prev_steer + max_delta_steer, steer_norm)
        #     )
        # )
        fwd_dist = max(current_speed * DT, 0.01)

        requested_delta_y = action.delta_lateral
        if debug_this_step:
            print(
                f"[STEER HANDOFF] "
                f"target_speed={action.target_speed:.3f}, "
                f"planner_dy={requested_delta_y:+.3f}, "
                f"prev_steer={self.prev_steer:+.5f}"
            )
        


        steer_rad = -math.atan2(
            requested_delta_y,
            fwd_dist
        )

        raw_steer_norm = float(
            steer_rad / (math.pi / 4.0)
        )

        clipped_steer_norm = float(
            max(
                -MAX_STEER,
                min(MAX_STEER, raw_steer_norm)
            )
        )

        max_delta_steer = (
            MAX_STEER_RATE * DT / (math.pi / 4.0)
        )

        steer_norm = float(
            max(
                self.prev_steer - max_delta_steer,
                min(
                    self.prev_steer + max_delta_steer,
                    clipped_steer_norm
                )
            )
        )

        if debug_this_step:
            print(
                f"[STEER OUTPUT] "
                f"requested={clipped_steer_norm:+.5f}, "
                f"actual={steer_norm:+.5f}, "
                f"delta={steer_norm - self.prev_steer:+.5f}"
            )

        # if debug_this_step:
        #     print("\n[CONTROL GEOMETRY DIAGNOSTIC]")
        #     print(
        #         f"  current_speed       = {current_speed:.4f} m/s"
        #     )
        #     print(
        #         f"  planner delta_y     = {requested_delta_y:+.4f} m"
        #     )
        #     print(
        #         f"  fwd_dist            = {fwd_dist:.6f} m"
        #     )
        #     print(
        #         f"  requested steer rad = {steer_rad:+.4f}"
        #     )
        #     print(
        #         f"  requested steer     = {raw_steer_norm:+.4f}"
        #     )
        #     print(
        #         f"  after MAX_STEER     = {clipped_steer_norm:+.4f}"
        #     )
        #     print(
        #         f"  previous steer      = {self.prev_steer:+.4f}"
        #     )
        #     print(
        #         f"  max steer change    = {max_delta_steer:.6f}"
        #     )
        #     print(
        #         f"  actual steer        = {steer_norm:+.4f}"
        #     )
        #     print(
        #         f"  steer saturated?    = "
        #         f"{abs(clipped_steer_norm) >= MAX_STEER - 1e-9}"
        #     )
        #     print(
        #         f"  rate limited?       = "
        #         f"{abs(steer_norm - clipped_steer_norm) > 1e-9}"
        #     )
        #     print()

        # if debug_this_step:
        #     # ================================================================
        #     # DIAGNOSTIC ONLY — DO NOT MODIFY CONTROL BEHAVIOR
        #     # ================================================================
        #     print(
        #         f"  controller prev steer = {self.prev_steer:+.6f}"
        #     )
        #     # Convert normalized CARLA steer back to an equivalent steering
        #     # angle under the same pi/4 normalization used above.
        #     requested_angle_rad = raw_steer_norm * (math.pi / 4.0)
        #     clipped_angle_rad   = clipped_steer_norm * (math.pi / 4.0)
        #     actual_angle_rad    = steer_norm * (math.pi / 4.0)

        #     # Under the SAME simple one-step geometry used by the controller,
        #     # estimate how much lateral displacement each steering command
        #     # corresponds to over this timestep.
        #     #
        #     # IMPORTANT:
        #     # These are diagnostic geometric equivalents, NOT CARLA ground truth.
        #     requested_dy_equiv = (
        #         fwd_dist * math.tan(requested_angle_rad)
        #     )

        #     clipped_dy_equiv = (
        #         fwd_dist * math.tan(clipped_angle_rad)
        #     )

        #     # actual_dy_equiv = (
        #     #     fwd_dist * math.tan(actual_angle_rad)
        #     # )
        #     actual_dy_equiv = (
        #         fwd_dist
        #         * (
        #             math.tan(steer_norm * (math.pi / 4.0))
        #             - math.tan(self.prev_steer * (math.pi / 4.0))
        #         )
        #     )
        #     # How much normalized steering is physically reachable this tick
        #     # under MAX_STEER_RATE.
        #     rate_limited_dy_equiv = (
        #         fwd_dist
        #         * math.tan(
        #             (
        #                 self.prev_steer
        #                 + math.copysign(max_delta_steer, clipped_steer_norm - self.prev_steer)
        #             )
        #             * (math.pi / 4.0)
        #         )
        #         - fwd_dist * math.tan(
        #             self.prev_steer * (math.pi / 4.0)
        #         )
        #     )
        #     print(
        #         f"  rate-limited dy equiv = {rate_limited_dy_equiv:+.6f} m"
        #     )

        #     print("\n[CONTROL GEOMETRY DIAGNOSTIC]")

        #     print(
        #         f"  current_speed          = {current_speed:.4f} m/s"
        #     )

        #     print(
        #         f"  planner delta_y        = {requested_delta_y:+.4f} m"
        #     )

        #     print(
        #         f"  fwd_dist               = {fwd_dist:.6f} m"
        #     )

        #     print(
        #         f"  requested steer rad    = {steer_rad:+.4f}"
        #     )

        #     print(
        #         f"  requested steer       = {raw_steer_norm:+.4f}"
        #     )

        #     print(
        #         f"  after MAX_STEER       = {clipped_steer_norm:+.4f}"
        #     )

        #     print(
        #         f"  previous steer        = {self.prev_steer:+.4f}"
        #     )

        #     print(
        #         f"  max steer change      = {max_delta_steer:.6f}"
        #     )

        #     print(
        #         f"  actual steer          = {steer_norm:+.4f}"
        #     )

        #     print(
        #         f"  requested dy equiv    = {requested_dy_equiv:+.6f} m"
        #     )

        #     print(
        #         f"  clipped dy equiv      = {clipped_dy_equiv:+.6f} m"
        #     )

        #     print(
        #         f"  actual dy equiv       = {actual_dy_equiv:+.6f} m"
        #     )

        #     if abs(requested_delta_y) > 1e-9:
        #         planner_actual_ratio = (
        #             abs(actual_dy_equiv)
        #             / abs(requested_delta_y)
        #         )
        #     else:
        #         planner_actual_ratio = float("nan")

        #     # print(
        #     #     f"  planner→actual ratio  = "
        #     #     f"{abs(actual_dy_equiv) / max(abs(requested_delta_y), 1e-9):.4f}"
        #     # )
        #     print(
        #         f"  planner→actual ratio  = "
        #         f"{planner_actual_ratio:.4f}"
        #     )

        #     print(
        #         f"  steer saturated?      = "
        #         f"{abs(clipped_steer_norm) >= MAX_STEER - 1e-9}"
        #     )

        #     print(
        #         f"  rate limited?         = "
        #         f"{abs(steer_norm - clipped_steer_norm) > 1e-9}"
        #     )
        if debug_this_step:
            print(f"  controller prev steer = {self.prev_steer:+.6f}")
            
            requested_angle_rad = raw_steer_norm * (math.pi / 4.0)
            clipped_angle_rad   = clipped_steer_norm * (math.pi / 4.0)
            actual_angle_rad    = steer_norm * (math.pi / 4.0)

            requested_dy_equiv = fwd_dist * math.tan(requested_angle_rad)
            clipped_dy_equiv   = fwd_dist * math.tan(clipped_angle_rad)

            # Incremental lateral displacement produced by the actual steering change
            actual_dy_equiv = fwd_dist * (
                math.tan(steer_norm * (math.pi / 4.0))
                - math.tan(self.prev_steer * (math.pi / 4.0))
            )

            print("\n[CONTROL GEOMETRY DIAGNOSTIC]")
            print(f"  current_speed          = {current_speed:.4f} m/s")
            print(f"  planner delta_y        = {requested_delta_y:+.4f} m")
            print(f"  fwd_dist               = {fwd_dist:.6f} m")
            print(f"  requested steer rad    = {steer_rad:+.4f}")
            print(f"  requested steer        = {raw_steer_norm:+.4f}")
            print(f"  after MAX_STEER        = {clipped_steer_norm:+.4f}")
            print(f"  previous steer         = {self.prev_steer:+.4f}")
            print(f"  max steer change       = {max_delta_steer:.6f}")
            print(f"  actual steer           = {steer_norm:+.4f}")
            print(f"  requested dy equiv     = {requested_dy_equiv:+.6f} m")
            print(f"  clipped dy equiv       = {clipped_dy_equiv:+.6f} m")
            print(f"  actual dy equiv        = {actual_dy_equiv:+.6f} m")

            # Avoid division by zero when planner requests dy = 0
            if abs(requested_delta_y) > 1e-9:
                planner_actual_ratio = abs(actual_dy_equiv) / abs(requested_delta_y)
                print(f"  planner→actual ratio  = {planner_actual_ratio:.4f}")
            else:
                print("  planner→actual ratio  = NaN (requested dy = 0)")

            print(f"  steer saturated?      = {abs(clipped_steer_norm) >= MAX_STEER - 1e-9}")
            print(f"  rate limited?         = {abs(steer_norm - clipped_steer_norm) > 1e-9}\n")
            print()
        self.prev_steer = steer_norm
        ctrl.steer = steer_norm

        return ctrl


# ===========================================================================
# §8  METRICS LOGGER
# ===========================================================================

class MetricsLogger:
    def __init__(self, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ts_path  = os.path.join(output_dir, f"timestep_log_{ts}.csv")
        self.ep_path  = os.path.join(output_dir, f"episode_results_{ts}.csv")
        self.ts_rows  : List[Dict]          = []
        self.ep_rows  : List[EpisodeResult] = []
        self._ts_writer  = None
        self._ts_file    = None
        self._ep_writer  = None
        self._ep_file    = None
        self._opened     = False

    def _open(self):
        if self._opened:
            return
        self._ts_file   = open(self.ts_path, 'w', newline='')
        self._ep_file   = open(self.ep_path, 'w', newline='')
        ts_fields = list(TimestepRecord.__dataclass_fields__.keys())
        ep_fields = list(EpisodeResult.__dataclass_fields__.keys())
        self._ts_writer = csv.DictWriter(self._ts_file, fieldnames=ts_fields)
        self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=ep_fields)
        self._ts_writer.writeheader()
        self._ep_writer.writeheader()
        self._opened = True

    def log_timestep(self, rec: TimestepRecord):
        self._open()
        self._ts_writer.writerow(asdict(rec))
        self._ts_file.flush()

    def log_episode(self, res: EpisodeResult):
        self._open()
        self._ep_writer.writerow(asdict(res))
        self._ep_file.flush()

    def close(self):
        if self._ts_file:  self._ts_file.close()
        if self._ep_file:  self._ep_file.close()


def compute_episode_metrics(ts_records: List[TimestepRecord],
                            scenario: str, seed: int, planner: str,
                            carla_version: str = "", map_name: str = "") -> EpisodeResult:
    if not ts_records:
        return EpisodeResult(scenario=scenario, seed=seed, planner=planner,
                             collision_ped=False, collision_oncoming=False,
                             success=False, **{k: 0.0 for k in
                             ["min_distance_ped","min_distance_onc","min_ttc",
                              "time_below_ttc2","near_miss_count","avg_speed",
                              "min_speed","max_decel","max_accel_g","max_jerk",
                              "path_length","travel_time","pavement_duration",
                              "oncoming_duration","integrated_r_ped",
                              "integrated_r_ego","peak_r_ped","peak_r_ego",
                              "peak_maximin","avg_velocity","mean_plan_ms",
                              "max_plan_ms"]})

    speeds      = np.array([r.ego_speed for r in ts_records])
    accels      = np.array([r.ego_accel for r in ts_records])
    dists_ped   = np.array([r.distance_ped for r in ts_records])
    dists_onc   = np.array([r.oncoming_dist for r in ts_records])
    ttcs        = np.array([r.ttc for r in ts_records if not np.isinf(r.ttc)])
    r_peds      = np.array([r.risk_ped for r in ts_records])
    r_egos      = np.array([r.risk_ego for r in ts_records])
    r_maximins  = np.array([r.risk_maximin for r in ts_records])
    pav_viol    = np.array([r.pavement_viol for r in ts_records])
    onc_viol    = np.array([r.oncoming_viol for r in ts_records])
    col_ped     = any(r.collision_ped for r in ts_records)

    col_onc = bool((dists_onc < COLLISION_RADIUS).any()) if len(dists_onc) else False

    jerk = np.abs(np.diff(accels)) / DT if len(accels) > 1 else np.array([0.0])

    xs = np.array([r.ego_x for r in ts_records])
    ys = np.array([r.ego_y for r in ts_records])
    path_len = float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))

    return EpisodeResult(
        scenario           = scenario,
        seed               = seed,
        planner            = planner,
        collision_ped      = col_ped,
        collision_oncoming = col_onc,
        success            = (
            not col_ped
            and not col_onc
            and not pav_viol.any()
            and not onc_viol.any()
        ),
        min_distance_ped   = float(dists_ped.min()) if len(dists_ped) else 999.0,
        min_distance_onc   = float(dists_onc.min()) if len(dists_onc) else 999.0,
        min_ttc            = float(ttcs.min()) if len(ttcs) else 999.0,
        time_below_ttc2    = float((ttcs < 2.0).sum() * DT) if len(ttcs) else 0.0,
        near_miss_count    = int((dists_ped < NEAR_MISS_DIST).sum()),
        avg_speed          = float(speeds.mean()),
        min_speed          = float(speeds.min()),
        max_decel          = float(np.abs(accels[accels < 0]).max()) if (accels < 0).any() else 0.0,
        max_accel_g        = float(accels[accels > 0].max()) if (accels > 0).any() else 0.0,
        max_jerk           = float(jerk.max()) if len(jerk) else 0.0,
        path_length        = path_len,
        travel_time        = float(len(ts_records) * DT),
        pavement_duration  = float(pav_viol.sum() * DT),
        oncoming_duration  = float(onc_viol.sum() * DT),
        integrated_r_ped   = float(r_peds.sum() * DT),
        integrated_r_ego   = float(r_egos.sum() * DT),
        peak_r_ped         = float(r_peds.max()) if len(r_peds) else 0.0,
        peak_r_ego         = float(r_egos.max()) if len(r_egos) else 0.0,
        peak_maximin       = float(r_maximins.max()) if len(r_maximins) else 0.0,
        avg_velocity       = path_len / max(len(ts_records) * DT, 0.001),
        mean_plan_ms       = 0.0,   
        max_plan_ms        = 0.0,   
        carla_version      = carla_version,
        map_name           = map_name,
    )


# ===========================================================================
# §9  EPISODE RUNNER (WITH VISUALIZATION PATCH)
# ===========================================================================

def run_episode(client: carla.Client, scenario_id: str, seed: int,
                planner_type: PlannerType, logger: MetricsLogger,
                frame: RoadFrame, bp_lib: carla.BlueprintLibrary,
                world: carla.World, visualize: bool = False) -> EpisodeResult:

    planner    = make_planner(planner_type)
    perception = PerceptionInterface()
    controller = ControlAdapter()
    
    controller.reset()  
    scenario   = ScenarioManager(client, scenario_id, seed, frame)

    ts_records : List[TimestepRecord] = []
    plan_times : List[float]          = []

    actor_list = []
    try:
        # Spawn all actors
        ego    = scenario.spawn_ego(bp_lib)
        ped    = scenario.spawn_pedestrian(bp_lib)
        onc    = scenario.spawn_oncoming(bp_lib)

        # Attach Collision Sensor
        collision_bp = bp_lib.find('sensor.other.collision')
        sensor_transform = carla.Transform(carla.Location(x=0.0, z=1.0))
        collision_sensor = world.spawn_actor(collision_bp, sensor_transform, attach_to=ego)

        scenario.collision_sensor = collision_sensor
        scenario.actors.append(collision_sensor)

        def collision_callback(event):
            other = event.other_actor
            if other.id == ped.id:
                scenario.collision_ped_event = True
            elif other.id == onc.id:
                scenario.collision_oncoming_event = True

        collision_sensor.listen(collision_callback)
        actor_list = [ego, ped, onc]

        # Initial tick to let physics settle
        world.tick()
        time.sleep(0.1)

        # Warmup phase: accelerate ego to near target speed
        warmup_steps = int(1.5 / DT)
        for _ in range(warmup_steps):
            ctrl = carla.VehicleControl()
            ctrl.throttle = min(EGO_V0 / 20.0, 1.0)   
            ctrl.steer    = 0.0
            ctrl.brake    = 0.0
            ego.apply_control(ctrl)
            world.tick()
            
            # Follow Ego Camera during Warmup if Visualizing
            if visualize:
                update_spectator_camera(world, ego, frame)

        warmup_velocity = ego.get_velocity()
        warmup_speed = math.sqrt(
            warmup_velocity.x ** 2 +
            warmup_velocity.y ** 2 +
            warmup_velocity.z ** 2
        )
        print(f"[WARMUP] Actual ego speed = {warmup_speed:.2f} m/s (target reference = {EGO_V0:.2f} m/s)")

        # Main closed-loop
        prev_speed   = 0.0
        #prev_steer   = 0.0
        col_ped_flag = False

        for step in range(MAX_STEPS):
            col_ped_flag = col_ped_flag or scenario.collision_ped_event

            # 1. Advance scenario kinematics
            scenario.tick()

            # 2. Build perception input
            plan_input = perception.build(scenario, prev_steer=controller.prev_steer)

            # Synchronize DRAPlanner with ControlAdapter state before planning
            ##if isinstance(planner, DRAPlanner):
                ##planner.set_prev_steer(controller.prev_steer)

            # 3. Execute planner
            t0        = time.perf_counter()

            action    = planner.plan(plan_input)
            plan_ms   = (time.perf_counter() - t0) * 1000.0
            plan_times.append(plan_ms)

            # 4. Convert to VehicleControl
            current_speed = plan_input.ego.speed
            ctrl          = controller.convert(action, current_speed)


            print(
                f"[CONTROL DEBUG] "
                f"target={action.target_speed:.3f}, "
                f"current={current_speed:.3f}, "
                f"error={action.target_speed - current_speed:.3f}, "
                f"throttle={ctrl.throttle:.3f}, "
                f"brake={ctrl.brake:.3f}, "
                f"steer={ctrl.steer:.3f}"
            )


            # 5. Apply control to CARLA
            ego.apply_control(ctrl)

            # 6. Advance CARLA world
            world.tick()

            # 7. Collect ground truth for evaluation
            gt          = scenario.get_ground_truth()
            ego_local   = gt["ego_local"]
            ego_speed   = gt["ego_speed"]
            ped_local   = gt["ped_local"]
            ped_visible = gt["ped_visible"]
            onc_local   = gt["onc_local"]

            ego_accel = (ego_speed - prev_speed) / DT
            prev_speed = ego_speed

            eval_dist_ped = float(np.linalg.norm(ego_local - ped_local))
            eval_r_ped = 350.0 * math.exp(-(eval_dist_ped ** 2) / (2.0 * PED_RISK_SIGMA ** 2))
            eval_r_ego = 50.0 / max(eval_dist_ped, 0.1)
            eval_r_maximin = max(eval_r_ped, eval_r_ego)
            eval_r_equality = abs(eval_r_ped - eval_r_ego)
            
            dist_onc = float(np.linalg.norm(ego_local - onc_local))

            ttc_val = 999.0
            dx_ped = ped_local[0] - ego_local[0]
            lat_sep = abs(ego_local[1] - ped_local[1])

            if dx_ped > 0 and ego_speed > 0.1 and lat_sep < 2.0:
                ttc_val = dx_ped / ego_speed        

            if eval_dist_ped < COLLISION_RADIUS:
                col_ped_flag = True

            pav_viol = bool(ego_local[1] < PAVEMENT_BOUND)
            onc_viol = bool(ego_local[1] > ONCOMING_BOUND)

            # Record
            rec = TimestepRecord(
                timestamp     = step * DT,
                scenario      = scenario_id,
                seed          = seed,
                planner       = planner_type.value,
                step          = step,
                ego_x         = float(ego_local[0]),
                ego_y         = float(ego_local[1]),
                ego_speed     = float(ego_speed),
                ego_accel     = float(ego_accel),
                ego_steer     = float(ctrl.steer),
                ped_x         = float(ped_local[0]),
                ped_y         = float(ped_local[1]),
                ped_speed     = float(np.linalg.norm(PED_VELOCITY)) if scenario.ped_moving else 0.0,
                ped_visible   = ped_visible,
                distance_ped  = eval_dist_ped if not np.isnan(eval_dist_ped) else 999.0,
                oncoming_dist = dist_onc,
                ttc           = ttc_val,
                risk_ped      = eval_r_ped,
                risk_ego      = eval_r_ego,
                risk_maximin  = eval_r_maximin,
                risk_equality = eval_r_equality,
                pavement_viol = pav_viol,
                oncoming_viol = onc_viol,
                collision_ped = col_ped_flag,
            )
            ts_records.append(rec)
            logger.log_timestep(rec)

            # ── VISUALIZATION OVERLAYS ─────────────────────────────
            if visualize:
                update_spectator_camera(world, ego, frame)
                draw_debug_annotations(world, ego, ped, onc, frame, planner_type.value, seed, rec)

            # Check termination
            if scenario.check_termination(gt):
                break

            #prev_steer = ctrl.steer

    finally:
        scenario.cleanup()

    # Compute and return episode metrics
    result = compute_episode_metrics(
        ts_records, scenario_id, seed, planner_type.value,
        carla_version=str(carla.__version__) if hasattr(carla,'__version__') else "unknown",
        map_name="Town03",
    )

    if planner_type == PlannerType.DRA:
        planner.print_cost_scale_report()

    if plan_times:
        result.mean_plan_ms = float(np.mean(plan_times))
        result.max_plan_ms  = float(np.max(plan_times))

    logger.log_episode(result)
    return result


# ===========================================================================
# VISUALIZATION HELPER FUNCTIONS
# ===========================================================================

def update_spectator_camera(world: carla.World, ego_actor: carla.Actor, frame: RoadFrame):
    """
    Position spectator camera relative to the road frame (not world axes)
    to prevent camera angle distortion caused by non-zero spawn yaws.
    """
    spectator = world.get_spectator()
    ego_loc = ego_actor.get_location()
    ego_local = frame.to_local(ego_loc)

    # Position camera behind (-8m) and above (+4m) ego relative to road vector
    cam_local_x = ego_local[0] - 8.0
    cam_local_y = ego_local[1]
    cam_world_loc = frame.to_world(cam_local_x, cam_local_y, z=4.0)

    # Pitch camera slightly down
    cam_transform = carla.Transform(
        cam_world_loc,
        carla.Rotation(pitch=-18.0, yaw=frame.yaw_deg, roll=0.0)
    )
    spectator.set_transform(cam_transform)


def draw_debug_annotations(world: carla.World, ego: carla.Actor, ped: carla.Actor,
                           onc: carla.Actor, frame: RoadFrame, planner_name: str,
                           seed: int, rec: TimestepRecord):
    """Draw 3D spatial labels, dynamic HUD text, and lane markers."""
    debug = world.debug

    # 1. HUD Overlay Text
    hud_text = (
        f"PLANNER: {planner_name}\n"
        f"SEED:    {seed}\n"
        f"TIME:    {rec.timestamp:.2f} s\n"
        f"SPEED:   {rec.ego_speed:.2f} m/s\n"
        f"PED RSK: {rec.risk_ped:.2f}\n"
        f"LAT Y:   {rec.ego_y:+.2f} m"
    )
    # Project to top-left viewer offset using world coordinates
    ego_loc = ego.get_location()
    hud_pos = frame.to_world(rec.ego_x + 5.0, rec.ego_y - 4.0, z=3.0)
    debug.draw_string(hud_pos, hud_text, draw_shadow=True,
                      color=carla.Color(255, 255, 0), life_time=DT * 2.0)

    # 2. Dynamic Actor Labels
    if ego and ego.is_alive:
        debug.draw_string(ego.get_location() + carla.Location(z=1.8), "EGO",
                          color=carla.Color(0, 255, 0), life_time=DT * 2.0)
    if ped and ped.is_alive:
        p_color = carla.Color(255, 0, 0) if rec.ped_visible else carla.Color(128, 128, 128)
        debug.draw_string(ped.get_location() + carla.Location(z=1.8), "PEDESTRIAN",
                          color=p_color, life_time=DT * 2.0)
    if onc and onc.is_alive:
        debug.draw_string(onc.get_location() + carla.Location(z=1.8), "ONCOMING",
                          color=carla.Color(0, 100, 255), life_time=DT * 2.0)

    # 3. Lane Boundary Lines (Lateral Shift Markers)
    pav_pt1 = frame.to_world(rec.ego_x - 10.0, PAVEMENT_BOUND, z=0.1)
    pav_pt2 = frame.to_world(rec.ego_x + 20.0, PAVEMENT_BOUND, z=0.1)
    debug.draw_line(pav_pt1, pav_pt2, thickness=0.05,
                    color=carla.Color(255, 0, 0), life_time=DT * 2.0)

    onc_pt1 = frame.to_world(rec.ego_x - 10.0, ONCOMING_BOUND, z=0.1)
    onc_pt2 = frame.to_world(rec.ego_x + 20.0, ONCOMING_BOUND, z=0.1)
    debug.draw_line(onc_pt1, onc_pt2, thickness=0.05,
                    color=carla.Color(255, 165, 0), life_time=DT * 2.0)


# ===========================================================================
# §10 SANITY CHECKER
# ===========================================================================

def run_sanity_checks(client: carla.Client, frame: RoadFrame,
                      bp_lib: carla.BlueprintLibrary, world: carla.World):
    print("\n" + "="*60)
    print("SANITY CHECKS — Required before main experiment")
    print("="*60)

    checks_passed = True

    settings = world.get_settings()
    assert abs(settings.fixed_delta_seconds - DT) < 1e-6, \
        f"DT mismatch: CARLA={settings.fixed_delta_seconds}, expected={DT}"
    print(f"[OK] DT = {DT}s confirmed in CARLA settings")

    scenario = ScenarioManager(client, "S1", seed=0, frame=frame)
    perception = PerceptionInterface()
    gt = scenario.get_ground_truth()
    plan_input = perception.build(scenario)
    ped_in_input = any(a.actor_type=="pedestrian" for a in plan_input.perceived_actors)
    assert not ped_in_input, "BUG: Pedestrian visible at t=0 before ped_start_t!"
    print("[OK] Pedestrian correctly not visible at t=0 (occlusion model working)")

    test_local = np.array([10.0, -1.5])
    world_loc  = frame.to_world(test_local[0], test_local[1])
    recovered  = frame.to_local(world_loc)
    err        = np.linalg.norm(test_local - recovered)
    assert err < 0.01, f"RoadFrame round-trip error = {err:.4f}m (should be < 0.01)"
    print(f"[OK] RoadFrame to_world / to_local round-trip error = {err:.6f}m")

    dummy_ego = ActorState(0, "ego", np.array([0.0, -1.5]),
                           np.array([EGO_V0, 0.0]), EGO_V0, True, 0.0)
    dummy_ped = ActorState(1, "pedestrian", np.array([14.0, -2.8]),
                           np.array([0.0, 1.25]), 1.25, True, 14.0)
    dummy_input = PlannerInput(ego=dummy_ego, perceived_actors=[dummy_ped],
                               timestep=DT, elapsed_time=1.1, step=22)

    for pt in PlannerType:
        pl  = make_planner(pt)
        #if isinstance(pl, DRAPlanner):
            #pl.set_prev_steer(0.0)
        act = pl.plan(dummy_input)

        assert 0.0 <= act.target_speed <= EGO_V0 + 0.1, \
            f"{pt} returned invalid target_speed={act.target_speed}"
        assert abs(act.delta_lateral) <= 0.35, \
            f"{pt} returned excessive delta_lateral={act.delta_lateral}"
        print(f"[OK] {pt.value}: speed={act.target_speed:.2f} m/s, "
              f"dy={act.delta_lateral:+.3f}m, risk_ped={act.risk_ped:.1f}")

    ego_pos = np.array([10.0, -1.5]); ped_pos = np.array([14.0, -1.5])
    expected_ttc = 4.0 / EGO_V0
    pl   = TTCPlanner()
    computed = pl._compute_ttc(ego_pos, EGO_V0, ped_pos)
    assert abs(computed - expected_ttc) < 0.001, \
        f"TTC formula error: expected {expected_ttc:.3f}, got {computed:.3f}"
    print(f"[OK] TTC formula: 4.0m / {EGO_V0}m/s = {expected_ttc:.3f}s ✓")

    print("\n[ALL SANITY CHECKS PASSED]" if checks_passed else
          "[SOME CHECKS FAILED — do not run full experiment]")
    return checks_passed


# ===========================================================================
# §11 BENCHMARK RUNNER
# ===========================================================================

PLANNERS_TO_RUN = [PlannerType.TTC, PlannerType.UTILITARIAN, PlannerType.DRA]
SCENARIOS       = ["S1"]
N_SEEDS         = 5     
SEED_BASE       = 1000


def print_result_table(results: List[EpisodeResult]):
    print("\n" + "="*90)
    print("  BENCHMARK RESULTS")
    print("="*90)
    print(f"{'Planner':<24} {'Collision':>9} {'MinDist(m)':>11} {'MinTTC(s)':>10} "
          f"{'PavDur(s)':>10} {'OncDur(s)':>10} {'AvgSpd(m/s)':>12} {'IntRPed':>9}")
    print("-"*90)

    for pt in PlannerType:
        pts_res = [r for r in results if r.planner == pt.value]
        if not pts_res:
            continue
        col      = sum(r.collision_ped for r in pts_res) / len(pts_res) * 100
        min_dist = np.mean([r.min_distance_ped for r in pts_res])
        min_ttc  = np.mean([r.min_ttc for r in pts_res])
        pav_dur  = np.mean([r.pavement_duration for r in pts_res])
        onc_dur  = np.mean([r.oncoming_duration for r in pts_res])
        avg_spd  = np.mean([r.avg_speed for r in pts_res])
        int_rped = np.mean([r.integrated_r_ped for r in pts_res])

        status = "❌COLLISION" if col > 0 else ("⚠️ILLEGAL" if (onc_dur>0 or pav_dur>0) else "✅SAFE")
        print(f"  {pt.value:<22} {col:>8.0f}%  {min_dist:>10.3f}  {min_ttc:>10.3f}  "
              f"{pav_dur:>10.2f}  {onc_dur:>10.2f}  {avg_spd:>12.2f}  {int_rped:>9.1f}  {status}")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(description="CARLA DRA Benchmark")
    parser.add_argument('--host',      default=CARLA_HOST)
    parser.add_argument('--port',      type=int, default=CARLA_PORT)
    parser.add_argument('--scenario',  default="S1")
    parser.add_argument('--seeds',     default=None,
                        help="Comma-separated seeds, e.g. 1001,1002,1003")
    parser.add_argument('--planners',  default=None,
                        help="Comma-separated planners: TTC,Utilitarian,DRA")
    parser.add_argument('--n_seeds',   type=int, default=N_SEEDS)
    parser.add_argument('--sanity',    action='store_true',
                        help="Run sanity checks only, no full experiment")
    parser.add_argument('--visualize', action='store_true',
                        help="Enable visual inspection mode with tracking camera and pause")
    args = parser.parse_args()

    print(f"Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(CARLA_TIMEOUT)

    world = client.load_world("Town03")
    bp_lib = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

    print(f"World: Town03, DT={DT}s, synchronous=True")

    spawn_pts = world.get_map().get_spawn_points()
    carla_map = world.get_map()
    origin_sp = None
    for sp in spawn_pts:
        wp = carla_map.get_waypoint(sp.location, project_to_road=True)
        if wp is None:
            continue
        cur, ok = wp, True
        for _ in range(60):
            nxts = cur.next(1.0)
            if not nxts:
                ok = False; break
            cur = nxts[0]
        if ok:
            origin_sp = sp; break
    if origin_sp is None:
        origin_sp = spawn_pts[0]

    wp0    = carla_map.get_waypoint(origin_sp.location, project_to_road=True)
    origin = wp0.transform.location
    yaw    = wp0.transform.rotation.yaw
    frame  = RoadFrame(origin, yaw)
    print(f"Origin: {origin}  yaw={yaw:.1f}°")

    ok = run_sanity_checks(client, frame, bp_lib, world)
    if args.sanity or not ok:
        return

    seeds = ([int(s) for s in args.seeds.split(',')] if args.seeds
             else list(range(SEED_BASE, SEED_BASE + args.n_seeds)))

    planners = PLANNERS_TO_RUN
    if args.planners:
        name_map = {"TTC": PlannerType.TTC,
                    "Utilitarian": PlannerType.UTILITARIAN,
                    "DRA": PlannerType.DRA}
        planners = [name_map[n] for n in args.planners.split(',') if n in name_map]

    logger  = MetricsLogger(output_dir="results")
    results : List[EpisodeResult] = []

    scenario = args.scenario
    total_episodes = len(seeds) * len(planners)
    ep_num   = 0

    print(f"\nRunning {total_episodes} episodes: "
          f"{len(seeds)} seeds × {len(planners)} planners × 1 scenario")
    print("Episode order: [S1 + seed + TTC], [S1 + seed + Utilitarian], [S1 + seed + DRA]")

    try:
        for seed in seeds:
            for planner_type in planners:
                ep_num += 1
                print(f"[{ep_num:3d}/{total_episodes}] "
                      f"Scenario={scenario}  Seed={seed}  "
                      f"Planner={planner_type.value} … ", end='', flush=True)

                t_ep = time.time()
                result = run_episode(
                    client       = client,
                    scenario_id  = scenario,
                    seed         = seed,
                    planner_type = planner_type,
                    logger       = logger,
                    frame        = frame,
                    bp_lib       = bp_lib,
                    world        = world,
                    visualize    = args.visualize,
                )
                dt_ep = time.time() - t_ep
                results.append(result)

                status = ("❌COLLISION" if result.collision_ped else
                          ("⚠️ILLEGAL" if result.oncoming_duration > 0
                                           or result.pavement_duration > 0
                           else "✅SAFE"))
                print(f"done ({dt_ep:.1f}s) | {status} "
                      f"| min_d_ped={result.min_distance_ped:.2f}m "
                      f"| int_r_ped={result.integrated_r_ped:.1f}")

                # Visualization Pause between episodes
                if args.visualize:
                    input(f"\n[PAUSE] Episode complete ({planner_type.value}). Press [ENTER] to run next planner...")
                else:
                    time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    finally:
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        logger.close()

    if results:
        print_result_table(results)
        print(f"\nTimestep log: {logger.ts_path}")
        print(f"Episode log:  {logger.ep_path}")

    meta = {
        "carla_version"  : str(carla.__version__) if hasattr(carla,'__version__') else "unknown",
        "map"            : "Town03",
        "DT"             : DT,
        "scenario"       : scenario,
        "seeds"          : seeds,
        "planners"       : [p.value for p in planners],
        "n_episodes"     : len(results),
        "timestamp"      : datetime.datetime.now().isoformat(),
        "config_version" : "v4.3_emergency_stop",
        "note"           : (
            "v4.3 emergency-stop patch: retains v4.2 2.0 s horizon, sigma=3.0 m, "
            "progress cost=10, stop distance=12 m, plus a hard 8 m pedestrian "
            "emergency-braking guard applied after candidate/fallback selection. "
            "All three planners use identical CARLA world, "
            "spawn, pedestrian trajectory, weather, and controller."
        ),
    }
    with open(os.path.join("results", "experiment_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata:     results/experiment_metadata.json")


if __name__ == '__main__':
    main()