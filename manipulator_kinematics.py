from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Literal

import numpy as np
import pinocchio as pin

# ---------------------------- constants & defaults ----------------------------
TCP_FRAME_NAME: str = "tcp" # Fixed link called tcp. 

VECTOR_WORLD_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
_EYE6 = np.eye(6, dtype=float)

PoseFormat = Literal["se3", "quat", "rpy", "axis-angle",
                      "axis-angle_array", "quat_array", "rpy_array"]

# --------------------------------- dataclasses --------------------------------

@dataclass
class KinematicsSolverParameters:
    eps: float = 1e-3
    translationErrorThreshold: float = 1e-3
    rotationErrorThreshold: float = np.pi / 180.0 * 1e-5
    maxIterationsCount: int = 10000
    timeStep: float = 1e-3
    damping: float = 1e-6


# --------------------------------- utilities ----------------------------------

def _solve_spd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b with A SPD using Cholesky; b can be shape (6,) or (6,k)."""
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def _quat_xyzw_from_R(R: np.ndarray) -> np.ndarray:
    """
    Quaternion (w,x,y,z) from rotation matrix using Pinocchio's Eigen wrapper, which
    yields coeffs() in [x, y, z, w].
    """
    q_xyzw = pin.Quaternion(R).coeffs()      # [x, y, z, w]
    return np.array([q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]], dtype=float)


def _rpy_xyz_from_R(R: np.ndarray) -> np.ndarray:
    """
    Roll-Pitch-Yaw (X-Y-Z extrinsic; radians). Pinocchio provides this directly.
    Returns [roll_x, pitch_y, yaw_z].
    """
    return np.array(pin.rpy.matrixToRpy(R), dtype=float)


def _axis_angle_from_R(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Axis-angle via Lie log: v = log3(R) = axis * angle.
    Returns (axis[3], angle_rad). If angle ~ 0, axis = [1,0,0].
    """
    v = np.array(pin.log3(R), dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    return v / angle, angle


def _format_pose(se3: pin.SE3, fmt: PoseFormat):
    """
    Format pose as requested.
      - "se3": returns pin.SE3
      - "quat": dict {"position": (x,y,z), "quaternion_xyzw": (x,y,z,w)}
      - "rpy":  dict {"position": (x,y,z), "rpy_xyz_rad": (roll_x,pitch_y,yaw_z)}
      - "axis-angle": dict {"position": (x,y,z), "axis": (ax,ay,az), "angle_rad": a}
    """
    if fmt == "se3":
        return se3
    t = np.array(se3.translation, dtype=float)
    R = np.array(se3.rotation, dtype=float)
    if fmt == "quat":
        return {"position": t, "quaternion_xyzw": _quat_xyzw_from_R(R)}
    if fmt == "rpy":
        return {"position": t, "rpy_xyz_rad": _rpy_xyz_from_R(R)}
    if fmt == "axis-angle":
        axis, angle = _axis_angle_from_R(R)
        return {"position": t, "axis": axis, "angle_rad": float(angle)}
    if (fmt == "axis-angle_array"):
        axis, angle = _axis_angle_from_R(R)
        return np.hstack([se3.translation, axis, angle])
    if fmt == "quat_array":
        return np.hstack([se3.translation, _quat_xyzw_from_R(R)])
    if fmt == "rpy_array":
        return np.hstack([se3.translation, _rpy_xyz_from_R(R)])
    raise ValueError(f"Unknown pose format: {fmt}")

# --------------------------------- main class ---------------------------------

class ManipulatorKinematics:
    """
    Concurrency-friendly Pinocchio kinematics/wrench utilities.

    ⚠️ Concurrency model:
      - Every method that needs Pinocchio Data accepts an optional `data: pin.Data`.
      - If you pass your own `pin.Data()` per thread, calls are thread-safe.
      - If you pass None, a shared internal `self._data` is used (NOT thread-safe).
      - No locks are used.

    Pose formats:
      - "quat": quaternion order **(w, x, y, z)**
      - "rpy":  roll-pitch-yaw **(x, y, z)** in **radians**
      - "axis-angle": unit axis (ax,ay,az) and angle (rad)
    """

    def __init__(
        self,
        urdf_filename: str,
        parameters: Optional[KinematicsSolverParameters] = None,
        tcp_frame_name: str = TCP_FRAME_NAME,
    ) -> None:
        self._model: pin.Model = pin.buildModelFromUrdf(urdf_filename)
        self._data: pin.Data = pin.Data(self._model)  # shared (non-thread-safe) fallback

        self._params = parameters or KinematicsSolverParameters()

        if not self._model.existFrame(tcp_frame_name):
            raise RuntimeError("[ManipulatorKinematics] TCP frame not found in robot model")
        if not self._model.existFrame(ft_frame_name):
            # Warn and set to None
            print("[ManipulatorKinematics] Warning: FT frame not found in robot model")
            ft_frame_name = None

        self._tcp_frame_name: str = tcp_frame_name
        self._tcp_frame_id: int = int(self._model.getFrameId(tcp_frame_name))
        self._ft_frame_name: str = ft_frame_name
        if (ft_frame_name is None) or (not self._model.existFrame(ft_frame_name)):
            self._ft_frame_id: int = -1
        else:
            self._ft_frame_id: int = int(self._model.getFrameId(ft_frame_name))

        self.setPose_EE_TCP_T(self._model.frames[self._tcp_frame_id].placement)

    # ----------------------------------- model ----------------------------------

    def getModel(self) -> pin.Model:
        return self._model

    def getTCPFrameName(self) -> str:
        return self._tcp_frame_name

    # -------------------------------- model properties ---------------------------

    @property
    def nq(self) -> int:
        """Number of position coordinates."""
        return self._model.nq

    @property
    def nv(self) -> int:
        """Number of velocity coordinates (degrees of freedom)."""
        return self._model.nv

    def get_neutral_q0(self) -> np.ndarray:
        """
        Returns the neutral configuration of the robot.
        This is the configuration where all joints are at their neutral/zero position.
        """
        return pin.neutral(self._model)

    def setPose_EE_TCP_T(self, pose_J6_TCP: pin.SE3) -> None:
        """
        Sets the placement of the TCP frame relative to its parent.
        Mutates the model's frame placement (global for the instance).
        """
        self._model.frames[self._tcp_frame_id].placement = pose_J6_TCP
        self._T_parent_TCP = pose_J6_TCP

    def getPose_J6_TCP(self) -> pin.SE3:
        return self._T_parent_TCP

    # --------------------------- center of mass & dynamics -----------------------

    def com(
        self,
        q: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        *,
        data: Optional[pin.Data] = None,
    ):
        """
        Compute center of mass position, velocity, and/or acceleration.
        
        Args:
            q: joint positions. If None, uses current data state.
            v: joint velocities. If provided, also computes vcom.
            a: joint accelerations. If provided (with v), also computes acom.
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
        
        Returns:
            - If q is None: returns com[0] from current data
            - If only q: returns com position (3,)
            - If q and v: returns (com, vcom)
            - If q, v, and a: returns (com, vcom, acom)
        """
        d = data if data is not None else self._data
        
        if q is None:
            pin.centerOfMass(self._model, d)
            return np.array(d.com[0], dtype=float)
        
        if v is not None:
            if a is None:
                pin.centerOfMass(self._model, d, q, v)
                return np.array(d.com[0], dtype=float), np.array(d.vcom[0], dtype=float)
            pin.centerOfMass(self._model, d, q, v, a)
            return (
                np.array(d.com[0], dtype=float),
                np.array(d.vcom[0], dtype=float),
                np.array(d.acom[0], dtype=float),
            )
        
        return pin.centerOfMass(self._model, d, q)

    def vcom(
        self,
        q: np.ndarray,
        v: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
    ) -> np.ndarray:
        """
        Compute center of mass velocity.
        
        Args:
            q: joint positions
            v: joint velocities
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
        
        Returns:
            vcom: center of mass velocity (3,)
        """
        d = data if data is not None else self._data
        pin.centerOfMass(self._model, d, q, v)
        return np.array(d.vcom[0], dtype=float)

    def acom(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
    ) -> np.ndarray:
        """
        Compute center of mass acceleration.
        
        Args:
            q: joint positions
            v: joint velocities
            a: joint accelerations
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
        
        Returns:
            acom: center of mass acceleration (3,)
        """
        d = data if data is not None else self._data
        pin.centerOfMass(self._model, d, q, v, a)
        return np.array(d.acom[0], dtype=float)

    def centroidalMomentum(
        self,
        q: np.ndarray,
        v: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
    ) -> np.ndarray:
        """
        Compute centroidal momentum.
        
        Args:
            q: joint positions
            v: joint velocities
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
        
        Returns:
            centroidal momentum (6,)
        """
        d = data if data is not None else self._data
        return pin.computeCentroidalMomentum(self._model, d, q, v)

    def mass(
        self,
        q: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
    ) -> np.ndarray:
        """
        Compute the joint space mass matrix (composite rigid body algorithm).
        
        Args:
            q: joint positions
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
        
        Returns:
            M: mass matrix (nv, nv)
        """
        d = data if data is not None else self._data
        return pin.crba(self._model, d, q)

    def nle(
        self,
        q: np.ndarray,
        v: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
    ) -> np.ndarray:
        """
        Compute nonlinear effects (Coriolis + gravity).
        
        Args:
            q: joint positions
            v: joint velocities
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
        
        Returns:
            nle: nonlinear effects vector (nv,)
        """
        d = data if data is not None else self._data
        return pin.nonLinearEffects(self._model, d, q, v)

    def gravity(
        self,
        q: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
    ) -> np.ndarray:
        """
        Compute generalized gravity forces.
        
        Args:
            q: joint positions
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
        
        Returns:
            g: gravity forces vector (nv,)
        """
        d = data if data is not None else self._data
        return pin.computeGeneralizedGravity(self._model, d, q)

    # --------------------------- FK (TCP or arbitrary) ---------------------------

    def forwardKinematics(
        self,
        q: np.ndarray,
        v: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        *,
        data: Optional[pin.Data] = None,
        frame_or_joint_name: Optional[str] = None,
        out_format: PoseFormat = "se3",
    ):
        """
        Compute FK at configuration q and return the pose in desired format.
        Optionally compute velocities and accelerations if v and a are provided.

        Args:
          q: joint configuration (nq or nv depending on model)
          v: joint velocities (optional)
          a: joint accelerations (optional, requires v)
          data: Pinocchio Data to use; if None, uses internal (not thread-safe)
          frame_or_joint_name: if None -> TCP; else resolve frame FIRST, then joint
          out_format: "se3" | "quat" | "rpy" | "axis-angle"

        Returns: pose formatted according to out_format (see _format_pose)
        """
        d = data if data is not None else self._data

        # Call appropriate forward kinematics based on provided arguments
        if v is not None:
            if a is not None:
                pin.forwardKinematics(self._model, d, q, v, a)
            else:
                pin.forwardKinematics(self._model, d, q, v)
        else:
            pin.forwardKinematics(self._model, d, q)

        # if a specific entity is requested
        if frame_or_joint_name is not None:
            try:
                fid = int(self._model.getFrameId(frame_or_joint_name))
                pin.updateFramePlacement(self._model, d, fid)
                pose = d.oMf[fid]
                return _format_pose(pose, out_format)
            except Exception:
                # not a frame? try joint
                try:
                    jid = int(self._model.getJointId(frame_or_joint_name))
                except Exception as e:
                    raise RuntimeError(f"Unknown frame or joint name: {frame_or_joint_name}") from e
                # joint world placement lives in oMi[jid]
                # no extra update call needed beyond forwardKinematics
                pose = d.oMi[jid]
                return _format_pose(pose, out_format).copy()

        # default: TCP
        pin.updateFramePlacement(self._model, d, self._tcp_frame_id)
        pose_tcp = d.oMf[self._tcp_frame_id]
        return _format_pose(pose_tcp, out_format).copy()

    def forwardKinematicsAllFrames(
        self,
        q: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
        out_format: PoseFormat = "se3",
    ):
        """
        Compute FK and return all frames' world poses in chosen format (dict name -> pose).
        """
        d = data if data is not None else self._data
        pin.forwardKinematics(self._model, d, q)
        pin.updateFramePlacements(self._model, d)

        result: Dict[str, object] = {}
        for fid in range(self._model.nframes):
            se3 = d.oMf[fid]
            result[self._model.frames[fid].name] = _format_pose(se3, out_format)
        return result

    def velocity(
        self,
        q: np.ndarray,
        v: np.ndarray,
        *,
        frame_name: Optional[str] = None,
        data: Optional[pin.Data] = None,
        update_kinematics: bool = True,
        reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL,
    ) -> pin.Motion:
        """
        Compute velocity at a frame.
        
        Args:
            q: joint positions
            v: joint velocities
            frame_name: frame name; if None, uses TCP frame
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
            update_kinematics: whether to update kinematics before computing velocity
            reference_frame: LOCAL, LOCAL_WORLD_ALIGNED, or WORLD
        
        Returns:
            velocity as pin.Motion
        """
        d = data if data is not None else self._data
        
        if update_kinematics:
            pin.forwardKinematics(self._model, d, q, v)
        
        frame_idx = self._resolve_frame_index(frame_name)
        return pin.getVelocity(self._model, d, frame_idx, reference_frame)

    def acceleration(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        *,
        frame_name: Optional[str] = None,
        data: Optional[pin.Data] = None,
        update_kinematics: bool = True,
        reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL,
    ) -> pin.Motion:
        """
        Compute acceleration at a frame.
        
        Args:
            q: joint positions
            v: joint velocities
            a: joint accelerations
            frame_name: frame name; if None, uses TCP frame
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
            update_kinematics: whether to update kinematics before computing acceleration
            reference_frame: LOCAL, LOCAL_WORLD_ALIGNED, or WORLD
        
        Returns:
            acceleration as pin.Motion
        """
        d = data if data is not None else self._data
        
        if update_kinematics:
            pin.forwardKinematics(self._model, d, q, v, a)
        
        frame_idx = self._resolve_frame_index(frame_name)
        return pin.getAcceleration(self._model, d, frame_idx, reference_frame)

    # ------------------------------- pose getters --------------------------------

    def getPose_World_TCP(
        self,
        *,
        data: Optional[pin.Data] = None,
        out_format: PoseFormat = "se3",
    ):
        d = data if data is not None else self._data
        # assumes FK already computed into d
        return _format_pose(d.oMf[self._tcp_frame_id], out_format)

    def getPose_World_FtFrame(
        self,
        *,
        data: Optional[pin.Data] = None,
        out_format: PoseFormat = "se3",
    ):
        d = data if data is not None else self._data
        pin.updateFramePlacement(self._model, d, self._ft_frame_id)
        return _format_pose(d.oMf[self._ft_frame_id], out_format)

    def getPose_World(
        self,
        name: str,
        *,
        data: Optional[pin.Data] = None,
        out_format: PoseFormat = "se3",
    ):
        """
        Pose of any frame or joint by name (world).
        Requires FK to have been run into `data` (or pass q to forwardKinematics).
        """
        d = data if data is not None else self._data
        # try frame
        try:
            fid = int(self._model.getFrameId(name))
            pin.updateFramePlacement(self._model, d, fid)
            return _format_pose(d.oMf[fid], out_format)
        except Exception:
            # try joint
            try:
                jid = int(self._model.getJointId(name))
            except Exception as e:
                raise RuntimeError(f"Unknown frame or joint name: {name}") from e
            return _format_pose(d.oMi[jid], out_format)

    def framePlacement(
        self,
        q: np.ndarray,
        *,
        frame_name: Optional[str] = None,
        data: Optional[pin.Data] = None,
        update_kinematics: bool = True,
    ) -> pin.SE3:
        """
        Compute frame placement (pose).
        
        Args:
            q: joint positions
            frame_name: frame name; if None, updates all frame placements
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
            update_kinematics: whether to update kinematics first
        
        Returns:
            frame placement as pin.SE3 (if frame_name provided), or None (if updating all)
        """
        d = data if data is not None else self._data
        
        if update_kinematics:
            pin.forwardKinematics(self._model, d, q)
        
        # Update specific frame
        frame_idx = self._resolve_frame_index(frame_name)
        return pin.updateFramePlacement(self._model, d, frame_idx)

    def frameVelocity(
        self,
        q: np.ndarray,
        v: np.ndarray,
        *,
        frame_name: Optional[str] = None,
        data: Optional[pin.Data] = None,
        update_kinematics: bool = True,
        reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL,
    ) -> pin.Motion:
        """
        Compute velocity at a frame.
        
        Args:
            q: joint positions
            v: joint velocities
            frame_name: frame name; if None, uses TCP frame
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
            update_kinematics: whether to update kinematics before computing velocity
            reference_frame: LOCAL, LOCAL_WORLD_ALIGNED, or WORLD
        
        Returns:
            velocity as pin.Motion
        """
        d = data if data is not None else self._data
        
        if update_kinematics:
            pin.forwardKinematics(self._model, d, q, v)
        
        frame_idx = self._resolve_frame_index(frame_name)
        return pin.getFrameVelocity(self._model, d, frame_idx, reference_frame)

    def frameTwist(
        self,
        q: np.ndarray,
        v: np.ndarray,
        *,
        frame_name: Optional[str] = None,
        data: Optional[pin.Data] = None,
        update_kinematics: bool = True,
        reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL,
    ) -> pin.Motion:
        """
        Compute twist (velocity) at a frame. Semantic alias for frameVelocity.
        
        Args:
            q: joint positions
            v: joint velocities
            frame_name: frame name; if None, uses TCP frame
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
            update_kinematics: whether to update kinematics before computing velocity
            reference_frame: LOCAL, LOCAL_WORLD_ALIGNED, or WORLD
        
        Returns:
            twist as pin.Motion
        """
        return self.frameVelocity(
            q, v,
            frame_name=frame_name,
            data=data,
            update_kinematics=update_kinematics,
            reference_frame=reference_frame,
        )

    def frameAcceleration(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        *,
        frame_name: Optional[str] = None,
        data: Optional[pin.Data] = None,
        update_kinematics: bool = True,
        reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL,
    ) -> pin.Motion:
        """
        Compute acceleration at a frame.
        
        Args:
            q: joint positions
            v: joint velocities
            a: joint accelerations
            frame_name: frame name; if None, uses TCP frame
            data: Pinocchio Data to use; if None, uses internal (not thread-safe)
            update_kinematics: whether to update kinematics before computing acceleration
            reference_frame: LOCAL, LOCAL_WORLD_ALIGNED, or WORLD
        
        Returns:
            acceleration as pin.Motion
        """
        d = data if data is not None else self._data
        
        if update_kinematics:
            pin.forwardKinematics(self._model, d, q, v, a)
        
        frame_idx = self._resolve_frame_index(frame_name)
        return pin.getFrameAcceleration(self._model, d, frame_idx, reference_frame)

    # ------------------------------ inverse kinematics ---------------------------

    def inverseKinematics(
        self,
        qInitial: np.ndarray,
        pose_World_DesTCP: pin.SE3,
        *,
        data: Optional[pin.Data] = None,
    ) -> Tuple[bool, np.ndarray]:
        """
        Closed-loop IK for TCP using damped least squares. Returns (success, q).
        Uses provided `data` if given, else internal one (not thread-safe).
        """
        d = data if data is not None else self._data
        q = qInitial.copy()

        for _ in range(self._params.maxIterationsCount + 1):
            pin.forwardKinematics(self._model, d, q)
            pin.updateFramePlacement(self._model, d, self._tcp_frame_id)

            pose_World_CalcTCP = d.oMf[self._tcp_frame_id]
            err_vec = pin.log6(pose_World_CalcTCP.actInv(pose_World_DesTCP)).vector

            if np.linalg.norm(err_vec) < self._params.eps:
                return True, q

            J = pin.computeFrameJacobian(self._model, d, q, self._tcp_frame_id, pin.ReferenceFrame.LOCAL)
            M = J @ J.T + self._params.damping * _EYE6
            dq = J.T @ _solve_spd(M, err_vec)

            q = pin.integrate(self._model, q, self._params.timeStep * dq)

        return False, q

    # --------------------------- differential inverse kinematics ----------------

    def inverseDifferentialKinematics(
        self,
        q: np.ndarray,
        spatialVelocity: pin.Motion,
        *,
        data: Optional[pin.Data] = None,
    ) -> np.ndarray:
        J = self.computeJacobianLocalFrame(q, data=data)
        M = J @ J.T + self._params.damping * _EYE6
        dq = J.T @ _solve_spd(M, spatialVelocity.vector)
        return dq

    # --------------------------------- Jacobians --------------------------------

    def computeJacobianLWAFrame(
        self,
        q: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
        frame_or_joint_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Jacobian in LOCAL_WORLD_ALIGNED. If frame_or_joint_name is None, uses TCP.
        For a joint name, we first map the joint to its associated frame (placement).
        """
        d = data if data is not None else self._data
        target_fid = self._resolve_frame_index(frame_or_joint_name)
        return pin.computeFrameJacobian(self._model, d, q, target_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def computeJacobianLocalFrame(
        self,
        q: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
        frame_or_joint_name: Optional[str] = None,
    ) -> np.ndarray:
        d = data if data is not None else self._data
        target_fid = self._resolve_frame_index(frame_or_joint_name)
        return pin.computeFrameJacobian(self._model, d, q, target_fid, pin.ReferenceFrame.LOCAL)

    # --------------------------------- twists -----------------------------------

    def computeTwistInEEFrame(
        self,
        q: np.ndarray,
        qDot: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
        frame_or_joint_name: Optional[str] = None,
    ) -> pin.Motion:
        d = data if data is not None else self._data
        target_fid = self._resolve_frame_index(frame_or_joint_name)
        pin.forwardKinematics(self._model, d, q, qDot)
        pin.updateFramePlacement(self._model, d, target_fid)
        return pin.getFrameVelocity(self._model, d, target_fid, pin.ReferenceFrame.LOCAL)

    def computeTwistInLocalWorldAlignedFrame(
        self,
        q: np.ndarray,
        qDot: np.ndarray,
        *,
        data: Optional[pin.Data] = None,
        frame_or_joint_name: Optional[str] = None,
    ) -> pin.Motion:
        d = data if data is not None else self._data
        target_fid = self._resolve_frame_index(frame_or_joint_name)
        pin.forwardKinematics(self._model, d, q, qDot)
        pin.updateFramePlacement(self._model, d, target_fid)
        return pin.getFrameVelocity(self._model, d, target_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    # -------------------------- load/wrench compensation -------------------------

    def estimateAndCompensateWrench(
        self,
        jointPositions: np.ndarray,
        wrenchMeasured_FtFrame_AllLoad: pin.Force,
        *,
        data: Optional[pin.Data] = None,
        transformToTCP: bool = False,
        rf: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL,
    ) -> pin.Force:
        """
        Compensate raw FT wrench for known load and (optionally) express at TCP.
        Output reference frame rf: LOCAL / LOCAL_WORLD_ALIGNED / WORLD.
        """
        d = data if data is not None else self._data

        # FK for both TCP and FT frames
        pin.forwardKinematics(self._model, d, jointPositions)
        pin.updateFramePlacement(self._model, d, self._tcp_frame_id)
        pin.updateFramePlacement(self._model, d, self._ft_frame_id)

        pose_World_TCP = d.oMf[self._tcp_frame_id]
        pose_World_FT = d.oMf[self._ft_frame_id]

        # predict static load; subtract from measured to get compensated (external/dynamic) wrench
        R_world_ft = pose_World_FT.rotation
        wrench_est_ft = self._predict_static_load_wrench_in_ft(R_world_ft)
        wrench_comp_ft = pin.Force(
            wrenchMeasured_FtFrame_AllLoad.linear - wrench_est_ft.linear,
            wrenchMeasured_FtFrame_AllLoad.angular - wrench_est_ft.angular,
        )

        # choose "local" expression (FT or TCP)
        if transformToTCP and (self._ft_frame_id != self._tcp_frame_id):
            pose_TCP_FT = pose_World_TCP.inverse() * pose_World_FT
            wrench_local = pose_TCP_FT.act(wrench_comp_ft)  # dual action for Force
            pose_World_Local = pose_World_TCP
        else:
            wrench_local = wrench_comp_ft
            pose_World_Local = pose_World_FT

        # convert to requested reference frame
        if rf == pin.ReferenceFrame.LOCAL:
            return wrench_local
        if rf == pin.ReferenceFrame.LOCAL_WORLD_ALIGNED:
            T = pin.SE3(pose_World_Local.rotation, np.zeros(3))
            return T.act(wrench_local)
        if rf == pin.ReferenceFrame.WORLD:
            return pose_World_Local.act(wrench_local)
        raise RuntimeError("Unknown or unsupported reference frame for wrench transformation.")

    def getRawWrenchBias(
        self,
        jointPositions: np.ndarray,
        wrenchMeasured_FtFrame_AllLoad: pin.Force,
        *,
        data: Optional[pin.Data] = None,
    ) -> pin.Force:
        d = data if data is not None else self._data
        pin.forwardKinematics(self._model, d, jointPositions)
        pin.updateFramePlacement(self._model, d, self._ft_frame_id)
        pose_World_FT = d.oMf[self._ft_frame_id]
        load_FT = self._get_ft_frame_load(pose_World_FT.rotation)
        return pin.Force(
            wrenchMeasured_FtFrame_AllLoad.linear - load_FT.linear,
            wrenchMeasured_FtFrame_AllLoad.angular - load_FT.angular,
        )

    def getJointPositionLimits(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(self._model.lowerPositionLimit).copy()
        upper = np.asarray(self._model.upperPositionLimit).copy()
        if lower.size != self._model.nv or upper.size != self._model.nv:
            raise RuntimeError("Joint limits size does not match the number of joints in the model (nv).")
        return lower, upper

    def getJointVelocityLimits(self) -> np.ndarray:
        return np.asarray(self._model.velocityLimit).copy()

    def getJointTorqueLimits(self) -> np.ndarray:
        return np.asarray(self._model.effortLimit).copy()

    # ------------------------------ internals (frame resolution) -----------------

    def _resolve_frame_index(self, frame_name: Optional[str]) -> int:
        """
        Resolve frame name to frame index.
        
        Args:
            frame_name: frame or joint name; if None, returns TCP frame index
        
        Returns:
            frame index (int)
        
        Raises:
            RuntimeError: if frame/joint name cannot be resolved
        """
        if frame_name is None:
            return self._tcp_frame_id
        
        try:
            return int(self._model.getFrameId(frame_name))
        except Exception:
            # Try as joint name
            try:
                jid = int(self._model.getJointId(frame_name))
                return int(self._model.getJointFrameId(jid))
            except Exception as e:
                raise RuntimeError(f"Cannot resolve frame or joint name: {frame_name}") from e

