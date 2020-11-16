""""
Utilities for working with attitude represenations
"""
import numpy as np


def angleAxis2Quat(angle, axis):
	ax = axis.flatten() / np.linalg.norm(axis)
	s = np.array([np.cos(angle / 2)])
	v = (np.sin(angle / 2) * ax).reshape(-1, 1)
	q = np.vstack((s, v))
	return q


def vec2Quat(v):
	q = np.zeros((4,1))
	q[1:, 0] = v
	return q / np.linalg.norm(q)


def quatMultiply(q1, q2):
	"""
	Multiplies two quaternions according to
	Hamilton's convention
	q3 = q1 * q2
	"""
	q1 = q1.flatten()
	q2 = q2.flatten()
	q3 = np.zeros(4)
	q3[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
	q3[1:] = (q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:]))
	return (q3 / np.linalg.norm(q3)).reshape(-1, 1)


def skewMat(v):
	"""
	skew symmetric cross product matrix 
	"""
	v = v.flatten()
	v_hat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return v_hat


def quatRightMat(q):
	"""
	Matrix equivalent to right multiplication
	of quaternion. Quaternion in form [s, v1, v2, v3]
	"""
	s = q[0]
	v = q[1:].reshape(-1,)
	R = np.zeros((4, 4))
	R[0, 0] = s
	R[0, 1:] = -v
	R[1:, 0] = v
	R[1:, 1:] = s*np.eye(3) - skewMat(v)
	return R


def quatLeftMat(q):
	"""
	Matrix equivalent to left multiplication
	of quaternion. Quaternion in form [s, v1, v2, v3].T
	"""
	s = q[0]
	v = q[1:].reshape(-1,)
	L = np.zeros((4, 4))
	L[0, 0] = s
	L[0, 1:] = -v
	L[1:, 0] = v
	L[1:, 1:] = s*np.eye(3) + skewMat(v)
	return L


def quatInv(q):
	"""
	Inverts a quaternion
	"""
	q_inv = np.zeros((4, 1))
	q_inv[0] = q[0]
	q_inv[1:] = -q[1:]
	return q_inv


def quatActiveRot(q, v):
	"""
	Rotates a vector, v by a quaternion, q
	"""
	v_q = np.zeros((4, 1))
	v_q[1:] = v
	v_qnew = quatLeftMat(q) @ quatRightMat(q).T @ v_q
	return v_qnew[1:]


def quatPassiveRot(q, v):
	"""
	Transforms a vector, v into a different frame
	with the transformation represented as a quaternion
	"""
	v_q = np.zeros((4, 1))
	v_q[1:] = v
	v_qnew = quatLeftMat(q).T @ quatRightMat(q) @ v_q
	return v_qnew[1:]


def quat2DCM(q):
	"""
	Transforms an active quaternion, q into a DCM
	"""
	DCM = quatLeftMat(q) @ quatRightMat(q).T
	DCM = DCM[1:, 1:]
	return DCM


def qMethod(g_b, g_n, m_b, m_n):
	"""
	Uses the q-method to estimate the optimal attitude quaternion, q_b2n
	relating magentometer and accelerometer measurements in the
	body and inertial frames
	"""
	B = g_b @ g_n.T + m_b @ m_n.T
	Z = (np.cross(g_b.flatten(), g_n.flatten()) + np.cross(m_b.flatten(), m_n.flatten())).reshape(-1, 1)
	K = np.block([[B + B.T - np.trace(B) * np.eye(3), Z],  # quadratic cost max qTKq
				  [Z.T, np.trace(B)]])
	w, v = np.linalg.eig(K)
	q_ = v[:, np.argmax(w), np.newaxis]  # maximum eigenvector
	q_ /= np.linalg.norm(q_)
	q_b2n = np.zeros((4, 1))  # convert unit quat from [v s] to [s v]
	q_b2n[0, 0] = q_[-1, 0]
	q_b2n[1:, 0] = q_[:-1, 0]
	return q_b2n


def quat2Euler(q_b2n):
	"""
	Converts a quaternion into Euler angles: q_b2n -> yaw, pitch roll from NED frame
	"""
	nCb = quat2DCM(q_b2n)
	bCn = nCb.T  # NED to body
	psi = np.arctan2(bCn[0, 1], bCn[0, 0])
	# theta = -np.arctan2(bCn[0, 2] * np.sin(psi), bCn[0, 1])
	theta = -np.arcsin(bCn[0, 2])
	phi = np.arctan2(bCn[1, 2], bCn[2, 2])
	angles = [psi, theta, phi]
	# for i in range(3):
	# 	if angles[i] < 0:
	# 		angles[i] = 2 * np.pi + angles[i]
	psi, theta, phi = angles
	return psi, theta, phi



