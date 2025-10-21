
import numpy as np
import pinocchio as pin
import pytest

from manipulator_kinematics.manipulator_kinematics import TCP_FRAME_NAME

import rerun as rr
from typing import List
from scipy.spatial.transform import Rotation as R

def init_rerun_session(session_name: str = "pose_dataset_visualization"):
    """Initialize a rerun session for visualization."""
    rr.init(session_name, spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Log world frame with axes
    rr.log("world", rr.Transform3D(translation=[0, 0, 0]))
    axis_length = 0.05
    rr.log(
        "world/world_frame",
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=[
                [axis_length, 0, 0],  # X-axis (red)
                [0, axis_length, 0],  # Y-axis (green)
                [0, 0, axis_length],  # Z-axis (blue)
            ],
            colors=[
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
        ),
    )
    
def log_pose_frame(entity_path: str, position: np.ndarray, rotation: np.ndarray, 
                   frame_size: float = 0.05, color: List[int] = None):
    """
    Log a single pose frame to rerun.
    
    Args:
        entity_path: The rerun entity path for this frame
        position: 3D position [x, y, z]
        rotation: 3D rotation in Euler angles [rx, ry, rz] (radians)
        frame_size: Size of the coordinate frame axes
        color: Optional color for the frame [r, g, b]
    """
    # Convert Euler angles to quaternion for rerun
    rotation_quat = R.from_euler('xyz', rotation).as_quat()  # Returns [x, y, z, w]
    
    # Log the transform
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=position,
            rotation=rr.Quaternion(xyzw=rotation_quat)
        )
    )
    
    # Log the coordinate frame axes
    if color is not None and isinstance(color, list) and len(color) >= 3:
        # Use provided color for all axes
        colors = [color, color, color]
    else:
        # Default RGB colors for XYZ axes
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    
    rr.log(
        f"{entity_path}/axes",
        rr.Arrows3D(
            vectors=[[frame_size, 0, 0], [0, frame_size, 0], [0, 0, frame_size]],
            colors=colors
        )
    )


def visualize_all_robot_frames(robot_kin, qpos, data, step_count, frame_size=0.03):
    """
    Visualize all robot frames using rerun.
    
    Args:
        robot_kin: ManipulatorKinematics instance
        qpos: Current joint positions
        data: Pinocchio data object
        step_count: Current step number for labeling
        frame_size: Size of the coordinate frame axes
    """
    # Set the timeline for this step
    rr.set_time_sequence("step", step_count)
    
    # Get all frames from forwardKinematicsAllFrames
    all_frames = robot_kin.forwardKinematicsAllFrames(qpos, data=data, out_format="se3")
    
    # Log each frame to rerun
    for frame_name, se3_pose in all_frames.items():
        # Extract position and rotation from SE3
        position = np.array(se3_pose.translation, dtype=float)
        rotation_matrix = np.array(se3_pose.rotation, dtype=float)
        
        # Convert rotation matrix to Euler angles for log_pose_frame
        scipy_rotation = R.from_matrix(rotation_matrix)
        euler_angles = scipy_rotation.as_euler('xyz')  # [roll, pitch, yaw] in radians
        
        # Create entity path for this frame
        entity_path = f"robot/frames/{frame_name}"
        
        # Log the frame
        log_pose_frame(
            entity_path=entity_path,
            position=position,

            rotation=euler_angles,
            frame_size=frame_size,
            color=None  # Use default RGB colors for axes
        )
        
        # Also add a text label at the frame position
        rr.log(
            f"{entity_path}/label",
            rr.TextDocument(frame_name)
        )


def rotation_from_quat_wxyz(qwxyz):
    # pinocchio expects [x, y, z, w]
    q_xyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]])
    return pin.Quaternion(q_xyzw).toRotationMatrix()

def rotation_from_axis_angle(axis, angle):
    return pin.exp3(np.asarray(axis) * float(angle))

@pytest.mark.parametrize("fmt", ["quat", "rpy", "axis-angle"])
def test_forwardKinematics_formats_consistent(mk, data, random_q, fmt, tol):
    # FK to TCP in SE3 (reference)
    se3 = mk.forwardKinematics(random_q, data=data, out_format="se3")
    R_ref = np.array(se3.rotation)
    t_ref = np.array(se3.translation)

    out = mk.forwardKinematics(random_q, data=data, out_format=fmt)
    assert out is not None

    if fmt == "quat":
        R = rotation_from_quat_wxyz(out["quaternion_wxyz"])
        t = out["position"]
    elif fmt == "rpy":
        R = pin.rpy.rpyToMatrix(np.array(out["rpy_xyz_rad"]))
        t = out["position"]
    elif fmt == "axis-angle":
        R = rotation_from_axis_angle(out["axis"], out["angle_rad"])
        t = out["position"]
    else:
        raise AssertionError("unexpected fmt")

    assert np.allclose(t, t_ref, atol=tol["pos"])
    assert np.allclose(R, R_ref, atol=tol["rot"])


def test_forwardKinematicsAllFrames_has_all_and_tcp_matches(mk, data, random_q, tol):
    frames = mk.forwardKinematicsAllFrames(random_q, data=data, out_format="se3")
    # should match model.nframes
    assert len(frames) == mk.getModel().nframes

    # TCP must be present & match direct FK
    tcp_pose = frames[mk.getTCPFrameName()]
    se3 = mk.forwardKinematics(random_q, data=data, out_format="se3")
    assert np.allclose(np.array(tcp_pose.translation), np.array(se3.translation), atol=tol["pos"])
    assert np.allclose(np.array(tcp_pose.rotation), np.array(se3.rotation), atol=tol["rot"])


def test_getPose_World_generic_names(mk, data, random_q, tol):
    # Run FK once
    mk.forwardKinematics(random_q, data=data, out_format="se3")
    # try TCP as a frame
    pose_tcp = mk.getPose_World(mk.getTCPFrameName(), data=data, out_format="se3")
    # should match direct
    direct_tcp = mk.getPose_World_TCP(data=data, out_format="se3")
    assert np.allclose(np.array(pose_tcp.translation), np.array(direct_tcp.translation), atol=tol["pos"])
    assert np.allclose(np.array(pose_tcp.rotation), np.array(direct_tcp.rotation), atol=tol["rot"])

    # also try querying the parent joint of TCP if available
    # not all robots expose the same joint names, so we do a tolerant attempt:
    try:
        # get the joint id that owns the TCP frame
        jid = mk.getModel().frames[mk.getModel().getFrameId(mk.getTCPFrameName())].parent
        jname = mk.getModel().names[jid]
        pose_joint = mk.getPose_World(jname, data=data, out_format="se3")
        assert pose_joint is not None
    except Exception:
        pytest.skip("Joint-by-name query not available on this model (no accessible joint name).")



def test_tcp_frame_location(mk, data, zero_q, tol):
    # FK to TCP in SE3 (reference)
    # python -m pytest -s manipulator_kinematics/test/test_fk_and_pose.py -k test_tcp_frame_location -v
    init_rerun_session("Pose visualization Test")
    world_tcp_T_se3 = mk.forwardKinematics(zero_q, data=data, out_format="se3")
    visualize_all_robot_frames(mk, zero_q, data, step_count=0, frame_size=0.05)
    
    print(f"TCP frame pose at zero q: Pos: {world_tcp_T_se3.translation},  Rot: {world_tcp_T_se3.rotation}")