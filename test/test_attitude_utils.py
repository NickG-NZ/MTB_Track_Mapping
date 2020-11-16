
import unittest
from attitude_utils import *
import pdb


class TestingAttitudeMethods(unittest.TestCase):

    def compare_vecs(self, v1, v2):
        v1 = v1.flatten()
        v2 = v2.flatten()
        self.assertEqual(len(v1), len(v2))
        for idx in range(len(v1)):
            self.assertAlmostEqual(v1[idx], v2[idx], places=4)

    def test_angleAxis2Quat(self):
        # identity
        angle = 0
        q1 = np.array([1, 0, 0, 0]).reshape(-1, 1)
        q2 = angleAxis2Quat(angle, np.random.rand(3, 1))
        self.compare_vecs(q1, q2)

        # 90 degree z rotation
        angle = np.pi / 2
        axis = np.array([0, 0, 1]).reshape(-1, 1)
        q1 = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        q2 = angleAxis2Quat(angle, axis)
        self.compare_vecs(q1, q2)
        
    def test_quatMultiply(self):
        axis1 = np.array([0, 0, 1])
        angle1 = np.pi / 2
        q1 = angleAxis2Quat(angle1, axis1)
        axis2 = np.array([0, 0, 1])
        angle2 = np.array(np.pi / 2)
        q2 = angleAxis2Quat(angle2, axis2)
        q3 = quatMultiply(q1, q2)

    def test_quatActiveRot(self):
        # rotation around z-axis by 90 degrees
        angle = np.pi / 2
        axis = np.array([0, 0, 1])
        q = angleAxis2Quat(angle, axis)
        v0 = np.array([1, 1, 1]).reshape(-1, 1)
        v1 = quatActiveRot(q, v0)
        v1_true = np.array([-1, 1, 1]).reshape(-1, 1)
        self.compare_vecs(v1, v1_true)

        # 2-axis rotation axis in x-y plane by 90 degrees
        angle = np.pi / 2 
        axis = np.array([1, 1, 0]) / np.sqrt(2)
        q = angleAxis2Quat(angle, axis)
        v0 = np.array([1, 0, 0]).reshape(-1, 1)
        v1 = quatActiveRot(q, v0)
        v1_true = np.array([0.5, 0.5, -np.cos(np.pi/4)])
        self.compare_vecs(v1, v1_true)

        # Frame transformation from body to inertial
        v_B = np.array([-np.sqrt(2), 0, 0]).reshape(-1, 1)
        angle = -np.pi / 4
        axis = np.array([0, 0, 1])
        q_B_to_N = angleAxis2Quat(angle, axis)
        v_N = quatActiveRot(q_B_to_N, v_B)
        v_N_true = np.array([-1, 1, 0])
        self.compare_vecs(v_N, v_N_true)

        # compound rotation body to inertial about Bz then new Bx
        v_B = np.array([0, 1, 0]).reshape(-1, 1)  # By
        angle1 = -np.pi / 2
        axis1 = np.array([0, 0, 1])  # Bz (or Nz)
        angle2 = -np.pi
        axis2 = np.array([1, 0, 0])  # new Bx (or Nx)
        q1 = angleAxis2Quat(angle1, axis1)
        q2 = angleAxis2Quat(angle2, axis2)
        q3 = quatMultiply(q1, q2)
        v_N = quatActiveRot(q3, v_B)
        v_N_true = np.array([-1, 0, 0])
        self.compare_vecs(v_N, v_N_true)

        # same example (body to inertial) but rotating N into B about FIXED inertial axes
        v_B = np.array([0, 1, 0]).reshape(-1, 1)  # By
        angle1 = -np.pi / 2
        axis1 = np.array([0, 0, 1])  # Nz
        angle2 = -np.pi
        axis2 = np.array([0, 1, 0])  # old Ny
        q1 = angleAxis2Quat(angle1, axis1)
        q2 = angleAxis2Quat(angle2, axis2)
        q3 = quatMultiply(q2, q1)  # swapped to "nice" ordering
        v_N = quatActiveRot(q3, v_B)
        v_N_true = np.array([-1, 0, 0])
        self.compare_vecs(v_N, v_N_true)

    def test_quatPassiveRot(self):
        # Frame transformation from inertial to body
        v_N = np.array([-1, 1, 0]).reshape(-1, 1)
        angle = -np.pi / 4
        axis = np.array([0, 0, 1])
        q_N_to_B = angleAxis2Quat(angle, axis)
        v_B = quatPassiveRot(q_N_to_B, v_N)
        v_B_true = np.array([-np.sqrt(2), 0, 0])
        self.compare_vecs(v_B, v_B_true)

        # compound rotation inertial to body about moving axes
        v_N = np.array([-1, 0, 0]).reshape(-1, 1)  # -Nx
        angle1 = -np.pi / 2
        axis1 = np.array([0, 0, 1])  # Nz
        angle2 = -np.pi
        axis2 = np.array([1, 0, 0])  # new Nx
        q1 = angleAxis2Quat(angle1, axis1)
        q2 = angleAxis2Quat(angle2, axis2)
        q3 = quatMultiply(q1, q2)
        v_B = quatActiveRot(q3, v_N)
        v_B_true = np.array([0, 1, 0])
        self.compare_vecs(v_B, v_B_true)

    def test_quatKinematics(self):
        # Inertial to body
        angle = 90 * np.pi / 180
        axis = np.array([0, 0, 1])
        q_N_to_B = angleAxis2Quat(angle, axis)
        omega_body = np.array([0, 0, 1])  # om_b/n in body frame
        om_q = vec2Quat(omega_body)
        q_dot = 0.5 * quatMultiply(q_N_to_B, om_q)
        print(q_N_to_B)
        print(q_dot)

        # Body to Inertial
        q_B_to_N = angleAxis2Quat(angle, axis)
        omega_body = np.array([0, 0, -1])  # om_n/b in body frame
        om_q = vec2Quat(omega_body)
        q_dot = 0.5 * quatMultiply(q_N_to_B, om_q)
        print(q_N_to_B)
        print(q_dot)




if __name__ == "__main__":
    unittest.main()