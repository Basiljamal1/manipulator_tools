
import numpy as np
import pinocchio as pin
import pytest

from manipulator_kinematics.manipulator_kinematics import TCP_FRAME_NAME

def norm_se3_error(Ta, Tb):
    return np.linalg.norm(pin.log6(Ta.actInv(Tb)).vector)

def test_computeTwistInEEFrame_matches_Jqdot(mk, data, random_q, small_qdot, tol):
    # twist in LOCAL frame
    twist = mk.computeTwistInEEFrame(random_q, small_qdot, data=data)
    J = mk.computeJacobianLocalFrame(random_q, data=data)
    twist_from_jac = J @ small_qdot
    assert np.allclose(twist.vector, twist_from_jac, atol=1e-5)


def test_inverseDifferentialKinematics_reproduces_twist(mk, data, random_q, small_qdot, tol):
    # Desired spatial velocity from J*qdot
    J = mk.computeJacobianLocalFrame(random_q, data=data)
    desired_twist_vec = J @ small_qdot
    desired_twist = pin.Motion(desired_twist_vec[:3], desired_twist_vec[3:])

    dq_est = mk.inverseDifferentialKinematics(random_q, desired_twist, data=data)
    # Check that J*dq_est ~= desired_twist (more robust than direct dq_est ~ qdot for redundancies)
    twist_from_dq = J @ dq_est
    assert np.allclose(twist_from_dq, desired_twist_vec, atol=2e-5)


def test_inverseKinematics_reaches_pose(mk, data, tol):
    model = mk.getModel()

    # choose a "ground truth" configuration
    q_true = pin.randomConfiguration(model)

    # target pose = FK(q_true) at TCP
    T_target = mk.forwardKinematics(q_true, data=data, out_format="se3")

    # start from a nearby seed: integrate small delta
    delta = 0.1 * (2*np.random.rand(model.nv) - 1.0)
    q0 = pin.integrate(model, q_true, delta)

    success, q_sol = mk.inverseKinematics(q0, T_target, data=data)
    assert success, "IK did not converge"

    # verify pose error
    mk.forwardKinematics(q_sol, data=data, out_format="se3")
    T_sol = mk.getPose_World_TCP(data=data, out_format="se3")
    err = norm_se3_error(T_sol, T_target)
    assert err < 1e-4