"""
Sim2Sim script for G1 Parkour task (Instinct-Parkour-Target-Amp-G1).
Transfers a trained policy from Isaac Lab (ONNX) to MuJoCo for validation.

Usage:
    python source/instinctlab/instinctlab/tasks/parkour/scripts/sim2sim.py \
        --model_dir /path/to/logs/.../exported/ \
        --mujoco_model /path/to/g1_mujoco.xml

===========================================================================
Observation Structure (Policy Input, flattened with history_length=8)
===========================================================================
The policy uses a depth-encoder + actor architecture.

Proprioception (fed directly to actor after concatenation with depth latent):
    [0:24]      base_ang_vel       × 8 history × 0.25 scale   = 24
    [24:48]     projected_gravity  × 8 history                 = 24
    [48:72]     velocity_commands  × 8 history                 = 24
    [72:304]    joint_pos_rel      × 8 history (29 joints)     = 232
    [304:536]   joint_vel_rel      × 8 history × 0.05 scale    = 232
    [536:768]   last_action        × 8 history (29 joints)     = 232
    Total proprio: 768

Depth Image (fed through depth encoder → latent vector):
    Shape depends on encoder (typically 8 frames × 16×16 from crop+resize)
    Depth encoder output is concatenated with proprio to form actor input.

Action Space (29 joints):
    target_pos = default_pos + clip(action, ±100) × action_scale
    torque = kp × (target_pos - q) - kd × dq
    torque = clip(torque, ±effort_limit)

    action_scale[j] = 0.25 × effort_limit[j] / stiffness[j]  (BeyondMimic formula)

===========================================================================
Isaac Sim Joint Order (29 DOF, torsoBase popsicle URDF)
===========================================================================
 0  left_shoulder_pitch_joint      1  right_shoulder_pitch_joint
 2  waist_pitch_joint              3  left_shoulder_roll_joint
 4  right_shoulder_roll_joint      5  waist_roll_joint
 6  left_shoulder_yaw_joint        7  right_shoulder_yaw_joint
 8  waist_yaw_joint                9  left_elbow_joint
10  right_elbow_joint             11  left_hip_pitch_joint
12  right_hip_pitch_joint         13  left_wrist_roll_joint
14  right_wrist_roll_joint        15  left_hip_roll_joint
16  right_hip_roll_joint          17  left_wrist_pitch_joint
18  right_wrist_pitch_joint       19  left_hip_yaw_joint
20  right_hip_yaw_joint           21  left_wrist_yaw_joint
22  right_wrist_yaw_joint         23  left_knee_joint
24  right_knee_joint              25  left_ankle_pitch_joint
26  right_ankle_pitch_joint       27  left_ankle_roll_joint
28  right_ankle_roll_joint
===========================================================================
"""

import argparse
import math
import numpy as np
import os
from collections import deque
import cv2

try:
    import mujoco
    import mujoco_viewer
except ImportError:
    print("Please install mujoco and mujoco_viewer: pip install mujoco mujoco_viewer")
    exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime")
    exit(1)


# ============================================================
# G1 29-DOF Joint Order (Isaac Sim internal order)
# ============================================================
G1_JOINT_NAMES = [
    "left_shoulder_pitch_joint",  # 0
    "right_shoulder_pitch_joint",  # 1
    "waist_pitch_joint",  # 2
    "left_shoulder_roll_joint",  # 3
    "right_shoulder_roll_joint",  # 4
    "waist_roll_joint",  # 5
    "left_shoulder_yaw_joint",  # 6
    "right_shoulder_yaw_joint",  # 7
    "waist_yaw_joint",  # 8
    "left_elbow_joint",  # 9
    "right_elbow_joint",  # 10
    "left_hip_pitch_joint",  # 11
    "right_hip_pitch_joint",  # 12
    "left_wrist_roll_joint",  # 13
    "right_wrist_roll_joint",  # 14
    "left_hip_roll_joint",  # 15
    "right_hip_roll_joint",  # 16
    "left_wrist_pitch_joint",  # 17
    "right_wrist_pitch_joint",  # 18
    "left_hip_yaw_joint",  # 19
    "right_hip_yaw_joint",  # 20
    "left_wrist_yaw_joint",  # 21
    "right_wrist_yaw_joint",  # 22
    "left_knee_joint",  # 23
    "right_knee_joint",  # 24
    "left_ankle_pitch_joint",  # 25
    "right_ankle_pitch_joint",  # 26
    "left_ankle_roll_joint",  # 27
    "right_ankle_roll_joint",  # 28
]
NUM_JOINTS = 29
JOINT_NAME_TO_IDX = {name: i for i, name in enumerate(G1_JOINT_NAMES)}

# ============================================================
# BeyondMimic PD gains computation
# (from instinctlab.assets.unitree_g1)
# ============================================================
PI = math.pi
NATURAL_FREQ = 10.0 * 2.0 * PI  # 10 Hz natural frequency
DAMPING_RATIO = 2.0

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  # ≈ 14.25
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2  # ≈ 40.18
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2  # ≈ 99.10
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2  # ≈ 16.78

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ  # ≈ 0.91
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # ≈ 2.56
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # ≈ 6.31
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ  # ≈ 1.07

# Build per-joint KP, KD, effort_limit arrays (Isaac joint order)
_KP = np.zeros(NUM_JOINTS, dtype=np.float64)
_KD = np.zeros(NUM_JOINTS, dtype=np.float64)
_EFFORT = np.zeros(NUM_JOINTS, dtype=np.float64)

# Legs: hip_pitch, hip_yaw → 7520-14; hip_roll, knee → 7520-22
for jn in ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint"]:
    i = JOINT_NAME_TO_IDX[jn]
    _KP[i] = STIFFNESS_7520_14
    _KD[i] = DAMPING_7520_14
    _EFFORT[i] = 88.0

for jn in ["left_hip_roll_joint", "right_hip_roll_joint", "left_knee_joint", "right_knee_joint"]:
    i = JOINT_NAME_TO_IDX[jn]
    _KP[i] = STIFFNESS_7520_22
    _KD[i] = DAMPING_7520_22
    _EFFORT[i] = 139.0

# Feet: ankle_pitch, ankle_roll → 2×5020
for jn in [
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]:
    i = JOINT_NAME_TO_IDX[jn]
    _KP[i] = 2.0 * STIFFNESS_5020
    _KD[i] = 2.0 * DAMPING_5020
    _EFFORT[i] = 50.0

# Waist: pitch/roll → 2×5020; yaw → 7520-14
for jn in ["waist_pitch_joint", "waist_roll_joint"]:
    i = JOINT_NAME_TO_IDX[jn]
    _KP[i] = 2.0 * STIFFNESS_5020
    _KD[i] = 2.0 * DAMPING_5020
    _EFFORT[i] = 50.0

_KP[JOINT_NAME_TO_IDX["waist_yaw_joint"]] = STIFFNESS_7520_14
_KD[JOINT_NAME_TO_IDX["waist_yaw_joint"]] = DAMPING_7520_14
_EFFORT[JOINT_NAME_TO_IDX["waist_yaw_joint"]] = 88.0

# Arms: shoulder/elbow → 5020; wrist_roll → 5020; wrist_pitch/yaw → 4010
for jn in [
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]:
    i = JOINT_NAME_TO_IDX[jn]
    _KP[i] = STIFFNESS_5020
    _KD[i] = DAMPING_5020
    _EFFORT[i] = 25.0

for jn in ["left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint"]:
    i = JOINT_NAME_TO_IDX[jn]
    _KP[i] = STIFFNESS_4010
    _KD[i] = DAMPING_4010
    _EFFORT[i] = 5.0

# Action scale: 0.25 * effort / stiffness  (BeyondMimic formula)
ACTION_SCALE = 0.25 * _EFFORT / _KP

# Default joint positions (from G1_29DOF_TORSOBASE_POPSICLE_CFG init_state)
DEFAULT_JOINT_POS = np.zeros(NUM_JOINTS, dtype=np.float64)
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["left_hip_pitch_joint"]] = -0.312
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["right_hip_pitch_joint"]] = -0.312
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["left_knee_joint"]] = 0.669
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["right_knee_joint"]] = 0.669
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["left_ankle_pitch_joint"]] = -0.363
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["right_ankle_pitch_joint"]] = -0.363
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["left_elbow_joint"]] = 0.6
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["right_elbow_joint"]] = 0.6
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["left_shoulder_roll_joint"]] = 0.2
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["left_shoulder_pitch_joint"]] = 0.2
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["right_shoulder_roll_joint"]] = -0.2
DEFAULT_JOINT_POS[JOINT_NAME_TO_IDX["right_shoulder_pitch_joint"]] = 0.2

# Observation constants
HISTORY_LENGTH = 8
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_JOINT_VEL = 0.05
CLIP_OBS = 100.0
CLIP_ACTIONS = 100.0

# Proprio size per step: ang_vel(3) + proj_grav(3) + cmd(3) + jpos(29) + jvel(29) + act(29) = 96
PROPRIO_PER_STEP = 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS  # 96
PROPRIO_TOTAL = PROPRIO_PER_STEP * HISTORY_LENGTH  # 768


# ============================================================
# Utility functions
# ============================================================
def quat_to_projected_gravity(quat_wxyz):
    """Compute projected gravity from quaternion (w,x,y,z).
    
    Projects [0, 0, -1] (gravity in world frame) into body frame.
    This matches Isaac Lab's `projected_gravity` observation.
    """
    w, x, y, z = quat_wxyz
    q_vec = np.array([x, y, z])
    g = np.array([0.0, 0.0, -1.0])
    a = g * (2.0 * w * w - 1.0)
    b = 2.0 * w * np.cross(q_vec, g)
    c = 2.0 * q_vec * np.dot(q_vec, g)
    return a - b + c


class ObsBuffer:
    """Rolling history buffer for proprioceptive observations."""

    def __init__(self, history_length=HISTORY_LENGTH):
        self.history_length = history_length
        self.buffer = deque(maxlen=history_length)

    def reset(self, initial_obs):
        """Fill buffer with initial observation repeated."""
        self.buffer.clear()
        for _ in range(self.history_length):
            self.buffer.append(initial_obs.copy())

    def append(self, obs):
        self.buffer.append(obs)

    def get_flattened(self):
        """Return [obs(t-H+1), ..., obs(t)] concatenated."""
        return np.concatenate(list(self.buffer))


def build_proprio_obs(ang_vel, proj_gravity, commands, joint_pos_rel, joint_vel_rel, last_action):
    """Build single-step proprioceptive observation (96-dim)."""
    return np.concatenate([
        ang_vel * OBS_SCALE_ANG_VEL,       # (3,)
        proj_gravity,                        # (3,)
        commands,                            # (3,)
        joint_pos_rel,                       # (29,)
        joint_vel_rel * OBS_SCALE_JOINT_VEL, # (29,)
        last_action,                         # (29,)
    ]).astype(np.float32)


# ============================================================
# ONNX model loading and inference
# ============================================================
def load_onnx_models(model_dir):
    """Load depth encoder and actor ONNX models."""
    providers = ort.get_available_providers()
    encoder_path = os.path.join(model_dir, "0-depth_encoder.onnx")
    actor_path = os.path.join(model_dir, "actor.onnx")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Depth encoder not found: {encoder_path}")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor not found: {actor_path}")

    encoder = ort.InferenceSession(encoder_path, providers=providers)
    actor = ort.InferenceSession(actor_path, providers=providers)

    print("[INFO] Depth encoder inputs:", [(i.name, i.shape) for i in encoder.get_inputs()])
    print("[INFO] Depth encoder outputs:", [(i.name, i.shape) for i in encoder.get_outputs()])
    print("[INFO] Actor inputs:", [(i.name, i.shape) for i in actor.get_inputs()])
    print("[INFO] Actor outputs:", [(i.name, i.shape) for i in actor.get_outputs()])

    return encoder, actor


def get_far_depth_latent(encoder):
    """Run depth encoder with ones to get a constant latent (using very-far assumption)."""
    inp = encoder.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    far_depth = np.ones(shape, dtype=np.float32)
    latent = encoder.run(None, {inp.name: far_depth})[0]
    print(f"[INFO] Far-depth latent shape: {latent.shape} (using very-far assumption)")
    return latent

# def get_real_depth_latent(renderer, data, encoder):
#     # 1. 更新並渲染深度圖
#     renderer.update_scene(data, camera="head_depth")
#     depth_image = renderer.render()
    
#     # 2. 裁剪中心 16x16 (原圖是 64x36)
#     cropped_depth = depth_image[0:16, 18:34]
    
#     # 2. 裁剪中心 16x16 (原圖是 64x36)
#     # y 軸從 0 開始取 16，x 軸從 18 開始取 16 (對應 crop_region=(18, 0, 16, 16))
#     cropped_depth = depth_image[0:16, 18:34] 
    
#     # 3. 截斷與正規化 (0.0~2.5m -> 0.0~1.0)
#     processed_depth = np.clip(cropped_depth, 0.0, 2.5) / 2.5
    
#     # 4. 變形為 ONNX 需要的張量形狀 [1, 1, 16, 16]
#     depth_input = processed_depth.reshape(1, 1, 16, 16).astype(np.float32)
    
#     # 5. 通過 Encoder 得到 Latent
#     latent = encoder.run(None, {encoder.get_inputs()[0].name: depth_input})[0]
#     return latent

def get_processed_depth(renderer, data):
    # 1. 更新並渲染深度圖
    renderer.update_scene(data, camera="head_depth")
    depth_image = renderer.render()
    
    # 2. Resize 到 32x18 (Width=32, Height=18)
    # cv2.resize 的尺寸格式要求為 (width, height)
    resized_depth = cv2.resize(depth_image, (32, 18), interpolation=cv2.INTER_AREA)
    
    # 3. 截斷與正規化 (0.0~2.5m -> 0.0~1.0)
    processed_depth = np.clip(resized_depth, 0.0, 2.5) / 2.5
    
    return processed_depth.astype(np.float32) # 回傳形狀為 (18, 32)


def run_inference(actor, proprio_flat, depth_latent):
    """Run actor: [proprio | depth_latent] → actions."""
    actor_input = np.concatenate([
        proprio_flat.reshape(1, -1),
        depth_latent,
    ], axis=1).astype(np.float32)
    action = actor.run(None, {actor.get_inputs()[0].name: actor_input})[0]
    return action.squeeze(0)


# ============================================================
# MuJoCo joint mapping
# ============================================================
def build_joint_mapping(model):
    """Build Isaac joint order ↔ MuJoCo qpos index mapping.
    
    MuJoCo free joint: qpos[0:3]=pos, qpos[3:7]=quat, qpos[7:]=joints
                       qvel[0:3]=lin_vel, qvel[3:6]=ang_vel, qvel[6:]=joint_vel
    """
    mujoco_joint_names = [model.joint(i).name for i in range(model.njnt)]
    print(f"[INFO] MuJoCo joints ({len(mujoco_joint_names)}): {mujoco_joint_names}")

    isaac_to_mujoco = {}  # isaac_idx → mujoco_joint_idx (0-based among non-free joints)
    mujoco_to_isaac = {}

    for isaac_idx, isaac_name in enumerate(G1_JOINT_NAMES):
        # Strip "_joint" suffix for flexible matching
        isaac_base = isaac_name.replace("_joint", "")
        for mj_idx, mj_name in enumerate(mujoco_joint_names):
            if isaac_name == mj_name or isaac_base in mj_name or mj_name in isaac_base:
                isaac_to_mujoco[isaac_idx] = mj_idx
                mujoco_to_isaac[mj_idx] = isaac_idx
                break

    unmapped = [G1_JOINT_NAMES[i] for i in range(NUM_JOINTS) if i not in isaac_to_mujoco]
    if unmapped:
        print(f"[WARN] Unmapped Isaac joints: {unmapped}")

    return isaac_to_mujoco, mujoco_to_isaac


def read_mujoco_state(data, isaac_to_mujoco):
    """Read MuJoCo state and return in Isaac joint order."""
    q = np.zeros(NUM_JOINTS, dtype=np.float64)
    dq = np.zeros(NUM_JOINTS, dtype=np.float64)
    for isaac_idx, mujoco_idx in isaac_to_mujoco.items():
        q[isaac_idx] = data.qpos[7 + mujoco_idx]
        dq[isaac_idx] = data.qvel[6 + mujoco_idx]
    return q, dq


def apply_mujoco_torques(data, tau, isaac_to_mujoco):
    """Apply torques (Isaac order) to MuJoCo actuators."""
    for isaac_idx, mujoco_idx in isaac_to_mujoco.items():
        data.ctrl[mujoco_idx] = tau[isaac_idx]


def set_mujoco_init(data, isaac_to_mujoco):
    """Set MuJoCo initial joint positions to default."""
    for isaac_idx, mujoco_idx in isaac_to_mujoco.items():
        data.qpos[7 + mujoco_idx] = DEFAULT_JOINT_POS[isaac_idx]


# ============================================================
# Main sim2sim loop
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Sim2Sim for G1 Parkour (InstinctLab)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to exported ONNX model directory (contains actor.onnx + 0-depth_encoder.onnx)")
    parser.add_argument("--mujoco_model", type=str, required=True,
                        help="Path to MuJoCo XML/URDF model of G1 robot")
    parser.add_argument("--sim_duration", type=float, default=30.0, help="Simulation duration (seconds)")
    parser.add_argument("--dt", type=float, default=0.005, help="Physics timestep (default: 0.005 = 200Hz)")
    parser.add_argument("--decimation", type=int, default=4,
                        help="Policy decimation (dt × decimation = policy freq, default: 4 → 50Hz)")
    parser.add_argument("--vx", type=float, default=0.0, help="X velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Y velocity command (m/s)")
    parser.add_argument("--vyaw", type=float, default=0.0, help="Yaw velocity command (rad/s)")
    args = parser.parse_args()

    # --- Load ONNX models ---
    print("=" * 60)
    print("[INFO] Loading ONNX models...")
    encoder, actor = load_onnx_models(args.model_dir)
    depth_latent_one = get_far_depth_latent(encoder)

    # Verify sizes
    actor_input_size = actor.get_inputs()[0].shape
    expected = PROPRIO_TOTAL + depth_latent_one.shape[1]
    if isinstance(actor_input_size[1], int):
        assert actor_input_size[1] == expected, (
            f"Actor input size {actor_input_size[1]} != proprio({PROPRIO_TOTAL}) + latent({depth_latent_one.shape[1]})"
        )
    print(f"[INFO] Actor input: proprio={PROPRIO_TOTAL} + depth_latent={depth_latent_one.shape[1]} = {expected}")

    # --- Load MuJoCo ---
    print("=" * 60)
    print(f"[INFO] Loading MuJoCo model: {args.mujoco_model}")
    model = mujoco.MjModel.from_xml_path(args.mujoco_model)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    
    depth_renderer = mujoco.Renderer(model, height=36, width=64)
    depth_renderer.enable_depth_rendering()

    isaac_to_mujoco, mujoco_to_isaac = build_joint_mapping(model)
    set_mujoco_init(data, isaac_to_mujoco)
    mujoco.mj_forward(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data)

    # --- Initialize ---
    action = np.zeros(NUM_JOINTS, dtype=np.float64)
    target_q = DEFAULT_JOINT_POS.copy()
    obs_buffer = ObsBuffer(HISTORY_LENGTH)
    commands = np.array([args.vx, args.vy, args.vyaw], dtype=np.float32)

    # Build initial obs
    q, dq = read_mujoco_state(data, isaac_to_mujoco)
    quat = data.qpos[3:7]  # (w, x, y, z)
    proj_grav = quat_to_projected_gravity(quat)
    ang_vel = data.qvel[3:6]  # body angular velocity (MuJoCo free joint convention)
    joint_pos_rel = q - DEFAULT_JOINT_POS
    initial_obs = build_proprio_obs(ang_vel, proj_grav, commands, joint_pos_rel, dq, action)
    obs_buffer.reset(initial_obs)
    
    depth_history_buffer = deque(maxlen=8)
    initial_depth = get_processed_depth(depth_renderer, data)
    for _ in range(8):
        depth_history_buffer.append(initial_depth)

    # --- Print config ---
    policy_freq = 1.0 / (args.dt * args.decimation)
    print("=" * 60)
    print(f"[INFO] Sim config: dt={args.dt}, decimation={args.decimation}, policy_freq={policy_freq:.1f}Hz")
    print(f"[INFO] Commands: vx={args.vx}, vy={args.vy}, vyaw={args.vyaw}")
    print(f"[INFO] Duration: {args.sim_duration}s")
    print(f"[INFO] Default joint pos (first 10): {np.round(DEFAULT_JOINT_POS[:10], 3)}")
    print(f"[INFO] Action scale (first 10): {np.round(ACTION_SCALE[:10], 4)}")
    print(f"[INFO] KP (first 10): {np.round(_KP[:10], 2)}")
    print(f"[INFO] KD (first 10): {np.round(_KD[:10], 4)}")
    print(f"[INFO] Effort limits (first 10): {_EFFORT[:10]}")
    print(f"[INFO] NOTE: Using ONE depth input (far ground assumption)")
    print("=" * 60)

    # --- Sim loop ---
    count = 0
    try:
        for step in range(int(args.sim_duration / args.dt)):
            # Read state
            q, dq = read_mujoco_state(data, isaac_to_mujoco)
            quat = data.qpos[3:7]
            proj_grav = quat_to_projected_gravity(quat)
            ang_vel = data.qvel[3:6]

            # Policy step
            if count % args.decimation == 0:
                joint_pos_rel = q - DEFAULT_JOINT_POS
                proprio = build_proprio_obs(ang_vel, proj_grav, commands, joint_pos_rel, dq, action)
                obs_buffer.append(proprio)

                proprio_flat = obs_buffer.get_flattened()
                proprio_flat = np.clip(proprio_flat, -CLIP_OBS, CLIP_OBS).astype(np.float32)
                # real_depth_latent = get_real_depth_latent(depth_renderer, data, encoder)
                # ⭐️ 1. 讀取當前深度並推入緩衝區
                current_depth = get_processed_depth(depth_renderer, data)
                depth_history_buffer.append(current_depth)
                
                # ⭐️ 2. 將 8 幀堆疊並轉換形狀為 ONNX 期望的 (1, 8, 18, 32)
                depth_input = np.stack(depth_history_buffer, axis=0) # 形狀: (8, 18, 32)
                depth_input = np.expand_dims(depth_input, axis=0)    # 形狀: (1, 8, 18, 32)
                
                # ⭐️ 3. 取得 Latent Vector
                real_depth_latent = encoder.run(None, {encoder.get_inputs()[0].name: depth_input})[0]
                action_onnx = run_inference(actor, proprio_flat, real_depth_latent)
                # action_onnx = run_inference(actor, proprio_flat, depth_latent_one)
                action = np.clip(action_onnx, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float64)

                target_q = DEFAULT_JOINT_POS + action * ACTION_SCALE

            # PD control: τ = kp × (target - q) - kd × dq
            tau = _KP * (target_q - q) - _KD * dq
            tau = np.clip(tau, -_EFFORT, _EFFORT)

            apply_mujoco_torques(data, tau, isaac_to_mujoco)
            mujoco.mj_step(model, data)

            if count % 10 == 0:
                viewer.render()
            count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        viewer.close()
        print("[INFO] Simulation finished.")


if __name__ == "__main__":
    main()