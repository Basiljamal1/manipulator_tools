# ManipulatorTools

**(This is still work in progress)**

A **lightweight, thread-safe, and dependency-minimal** Python module for robotic manipulator kinematics, built on top of [Pinocchio](https://stack-of-tasks.github.io/pinocchio/).

Unlike heavy robotic libraries that bring large frameworks and dependencies, `ManipulatorKinematics` provides just what you need — forward/inverse kinematics, Jacobians, dynamics, and wrench compensation; all wrapped in a clean, modern Python API.

---

## 🌟 Key Features

* **Thread-safe** design — each thread can safely use its own `pin.Data()` instance. If not, an internal data is used.
* **Minimal overhead** — no ROS, no large frameworks, only `numpy` and `pinocchio`.
* **Flexible pose formats** — choose between SE(3), quaternions, roll-pitch-yaw, or axis-angle.
* **Custom TCP offset** — redefine your **Tool Center Point (TCP)** dynamically. Useful for control in **camera**, **gripper**, or **tool** frames.
* **Full dynamics support** — compute mass matrix, nonlinear effects, gravity, centroidal momentum, and center of mass.
* **Convenient utilities** — clean access to joint limits, velocities, accelerations, and wrenches.
* **Inverse and differential IK** — damped least-squares solvers with configurable tolerances for now. 

---

## 🧠 Concept

Robotic libraries often come with unnecessary complexity. This package focuses on **what matters most** for robotics developers — **kinematics, dynamics, and frame transformations**, all in a **simple and efficient form**.

Every robot model is defined by its URDF, which **must contain a fixed link named `tcp`**. This link acts as the default Tool Center Point (TCP) but can be redefined at runtime via `setPose_EE_TCP_T()` to shift the control frame relative to the robot’s end-effector.

---

## ⚙️ Requirements

* Python ≥ 3.8
* [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)
* NumPy

Your URDF **must** define a fixed frame named `tcp`.
For example:

```xml
<link name="tcp">
  <visual> ... </visual>
</link>
```

---

## 🚀 Quick Start

### 1. Load Your Robot

```python
from manipulator_kinematics import ManipulatorKinematics

kin = ManipulatorKinematics("path/to/robot.urdf")
```

### 2. Forward Kinematics

```python
import numpy as np

q = np.zeros(kin.nq)
pose = kin.forwardKinematics(q, out_format="rpy_array") # Other output formats such as quaternion, and axis angle are supported. 
print(pose) # Prints the pose of the TCP
```

### 3. Inverse Kinematics

```python
from pinocchio import SE3

desired_pose = SE3(np.eye(3), np.array([0.4, 0.2, 0.5]))
success, q_sol = kin.inverseKinematics(q, desired_pose)
```

### 4. Change the TCP (Tool Center Point)

```python
from pinocchio import SE3

# Offset the TCP by 5 cm in z
T_offset = SE3(np.eye(3), np.array([0.0, 0.0, 0.05]))
kin.setPose_EE_TCP_T(T_offset)
```

### 5. Thread-Safe Computations

When using in multi-threaded contexts:

```python
data = pin.Data(kin.getModel())
pose = kin.forwardKinematics(q, data=data)
```

Each thread should allocate its own `pin.Data()` instance for concurrency.

---

## 🧩 API Overview

| Category                | Methods                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| **Model & Properties**  | `getModel()`, `nq`, `nv`, `getJointPositionLimits()`, `getJointVelocityLimits()`                 |
| **Forward Kinematics**  | `forwardKinematics()`, `forwardKinematicsAllFrames()`                                            |
| **Inverse Kinematics**  | `inverseKinematics()`, `inverseDifferentialKinematics()`                                         |
| **Dynamics**            | `mass()`, `nle()`, `gravity()`, `centroidalMomentum()`, `com()`                                  |
| **Twists & Jacobians**  | `computeJacobianLWAFrame()`, `computeTwistInEEFrame()`, `computeTwistInLocalWorldAlignedFrame()` |
| **Wrench Compensation** | `estimateAndCompensateWrench()`, `getRawWrenchBias()`                                            |
| **TCP & Frame Access**  | `setPose_EE_TCP_T()`, `getPose_World_TCP()`, `getPose_World(name)`                               |

---

## 🧵 Concurrency Model

`ManipulatorKinematics` is **lock-free** by design.

* ✅ **Thread-safe:** if each thread passes its own `pin.Data()` instance.
* ⚠️ **Not thread-safe:** if using the shared internal `self._data`.

This design keeps it **fast and predictable** for real-time control or multi-threaded visualization pipelines.

---

## 🛠️ Example Use Case

**Visualize a robot’s TCP motion in a background thread:**

```python
import threading
import time

def visualize():
    data = pin.Data(kin.getModel())
    while True:
        q = get_current_joint_state()  # hypothetical function
        pose = kin.forwardKinematics(q, data=data, out_format="quat")
        visualize_pose(pose)           # user-defined visualization
        time.sleep(0.01)

threading.Thread(target=visualize, daemon=True).start()
```

This allows your visualization thread to safely compute forward kinematics while another thread runs the control loop.

---

## 📐 Pose Formats

Supported output formats from `forwardKinematics()` or `getPose_*()`:

| Format         | Description               | Output                                                      |
| -------------- | ------------------------- | ----------------------------------------------------------- |
| `"se3"`        | Pinocchio SE3 object      | `pin.SE3`                                                   |
| `"quat"`       | Quaternion                | `{"position": (x,y,z), "quaternion_xyzw": (x,y,z,w)}`       |
| `"rpy"`        | Roll-pitch-yaw (radians)  | `{"position": (x,y,z), "rpy_xyz_rad": (r,p,y)}`             |
| `"axis-angle"` | Axis-angle representation | `{"position": (x,y,z), "axis": (ax,ay,az), "angle_rad": a}` |


Add  an `"_array"` post format (e.g `quat_array`) returns the translation and orientation in a single array. 

---

## 🧮 Kinematics Solver Parameters

Tuning parameters for inverse kinematics and DLS-based solvers are encapsulated in:

```python
@dataclass
class KinematicsSolverParameters:
    eps: float = 1e-3
    translationErrorThreshold: float = 1e-3
    rotationErrorThreshold: float = np.pi / 180.0 * 1e-5
    maxIterationsCount: int = 10000
    timeStep: float = 1e-3
    damping: float = 1e-6
```

Pass custom parameters at construction:

```python
params = KinematicsSolverParameters(eps=1e-4, damping=1e-5)
kin = ManipulatorKinematics("robot.urdf", parameters=params)
```

---

## 🧰 Dependencies

* `numpy`
* `pinocchio`

Install via pip:

```bash
pip install numpy pin
```