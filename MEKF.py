"""
A Multiplicative Extended Kalman Filter for position and attitude estimation with INS and GNSS
"""
from coordinate_utils import *
from attitude_utils import *


class MEKF:

    def __init__(self, map_config, sensor_config, init_state, init_covariance):
        self.motion_model = MotionModel(map_config, sensor_config, init_state)
        self.gnss_accel_model = GnssAccelMeasurementModel(map_config, sensor_config, init_state)
        self.mag_model = MagMeasurementModel(map_config, sensor_config, init_state)
        self.state = init_state
        self.covariance = init_covariance

    def step(self, gyro_meas, accel_meas, mag_meas, gnss_meas):
        """
        Performs an update step using the avilable measurements
        GNSS measurement update takes precidence over magnetometer updates due to the lower rate
        """
        state_pred, cov_pred = self.motion_model.step(self.state, self.covariance, gyro_meas, accel_meas)

        state = state_pred
        covariance = cov_pred
        if not np.isnan(gnss_meas[0, 0]):
            error_state, covariance = self.gnss_accel_model.step(state_pred, cov_pred, gnss_meas, accel_meas)
            state = self._error_correction(state_pred, error_state)
        elif not np.isnan(mag_meas[0, 0]):
            error_state, covariance = self.mag_model.step(state_pred, cov_pred, mag_meas)
            state = self._error_correction(state_pred, error_state)

        if np.isnan(state[0]):  # Measurement update failed
            state = state_pred
            covariance = cov_pred
        self.state = state
        self.covariance = covariance
        self._enforce_unit_quaternion()
        return self.state, self.covariance

    def _error_correction(self, state_pred, error_state):
        """
        Uses the error state returned by the measurement model to correct the state
        prediction from the measurement model
        """
        state = np.zeros((13, 1))
        if np.linalg.norm(error_state[:3]) > 1:
            print("WARNING: Rotation error too large")
            return np.full(13, np.nan)
        error_quat = np.vstack((np.sqrt(1 - error_state[:3].T @ error_state[:3]), error_state[:3])).reshape(-1, 1)
        state[:4] = quatMultiply(state_pred[:4], error_quat)
        state[4:] = state_pred[4:] + error_state[3:]
        return state

    def _enforce_unit_quaternion(self):
        """
        Checks the error on the quaternion norm and normalizes if neccesary
        """
        tol = 1e-5
        quat_norm = np.linalg.norm(self.state[:4])
        if abs(1 - quat_norm) > tol:
            self.state[:4] /= quat_norm
            print("Re-normalized quaternion")


class MotionModel:
    """
    A motion model for gyroscope and accelerometer inputs
    """
    def __init__(self, map_config, sensor_config, initial_state):
        self.map_config = map_config
        self.sensor_config = sensor_config
        self.state = initial_state
        self.covariance = None
        self.jacobian = None
        self.motion_covariance = None  # Q
        self.omega_bn_b = None  # [rad/s]  body wrt NED in body coord (adjusted for bias)
        self.omega_bi_b = np.zeros((3,1))  # The gyro measurement
        self.omega_ne_n = None
        self.omega_ei_n = None
        self.f_b = np.zeros((3,1))  # Accelerometer measurement
        self.accel_n = None  # [m/s^2] total accel in body frame
        self.input_quat = None  # Sk
        self.dT = map_config["predict_rate"]  # [s]

    def step(self, state, covariance, gyro_meas, accel_meas):
        self.state = state
        self.covariance = covariance
        self._update_angular_vels(gyro_meas)
        self._update_accel(accel_meas)
        self._update_motion_covariance()
        state_pred = self._nonlinear_step()
        covariance_pred = self._covariance_step()
        return state_pred, covariance_pred

    def _nonlinear_step(self):
        """
        Compute the new state
        """
        # Attitude
        theta = np.linalg.norm(self.omega_bn_b) * self.dT
        axis = self.omega_bn_b * self.dT / theta
        self.input_quat = np.vstack((np.cos(theta/2), axis * np.sin(theta/2)))
        q_new = quatMultiply(self.state[:4], self.input_quat)
        # bias
        b_new = self.state[4:7]
        # velocity
        v_dot = self.accel_n - skewMat(2 * self.omega_ei_n + self.omega_ne_n) @ self.state[7:10]
        v_new = self.state[7:10] + v_dot * self.dT
        # position
        p_dot = velNed2Geo(self.state[7:10], self.state[10:])
        p_new = self.state[10:] + p_dot * self.dT
        state_new = np.vstack((q_new, b_new, v_new, p_new))
        return state_new

    def _covariance_step(self):
        """
        Compute the new covariance
        """
        self._update_jacobian()
        cov_new = self.jacobian @ self.covariance @ self.jacobian.T + self.motion_covariance
        return cov_new

    def _update_jacobian(self):
        """
        Find the MEKF jacobian Ak for the nonlinear system
        """
        A = np.zeros((12, 12))
        V = np.zeros((4,3))
        V[1:, :] = np.eye(3)
        A[:3, :3] = V.T @ quatLeftMat(self.input_quat).T @ quatRightMat(self.input_quat) @ V  # dphi/dphi
        A[:3, 3:6] = -(1 / 2) * np.eye(3) * self.dT  # dphi/db
        A[3:6, 3:6] = np.eye(3)  # db/db
        A[6:9, :3] = -2 * quat2DCM(self.state[:4]) @ skewMat(self.f_b) * self.dT  # dV/dphi
        A[6:9, 6:9] = np.eye(3) - (2 * self.omega_ei_n + self.omega_ne_n) * self.dT  # dV/dV
        A[9:, 6:9] = velNed2GeoMatrix(self.state[10:]) * self.dT  # dP/dV

        lat = self.state[10, 0]
        alt = self.state[12, 0]
        M = meridian_radius(lat)
        N = prime_vertical_radius(lat)
        V = self.state[7:10, 0]
        a = WGS84['a']
        e2 = WGS84['e'] ** 2
        dlatdh = -V[0] / ((M + alt) ** 2)
        dlatdlat = 3 * dlatdh * (a * (1 - e2)) * np.sin(lat) * np.cos(lat) / ((1 - e2 * np.sin(lat) ** 2) ** (5 / 2))
        dlondh = -V[1] / (np.cos(lat) * (N + alt) ** 2)
        dlondlat = dlondh * a * e2 * np.sin(lat) * np.cos(lat) / ((1 - e2 * np.sin(lat) ** 2) ** (3 / 2)) + \
                   2 * V[1] * np.sin(lat) / ((N + alt) * (np.cos(2 * lat) + 1))
        dpdotdp = np.array([[dlatdlat, 0, dlatdh], [dlondlat, 0, dlondh], [0, 0, 0]])
        A[9:, 9:] = np.eye(3) + dpdotdp * self.dT  # dP/dP
        self.jacobian = A

    def _update_angular_vels(self, omega_bi_b):
        """
        Find useful angular-velocities from raw gyro measurement and state
        """
        bCn = quat2DCM(self.state[:4]).T  # DCM, NED to body
        omega_ei_e = np.array([0, 0, WGS84['omega_earth']]).reshape(-1, 1)
        self.omega_ei_n = ecef2Ned(omega_ei_e, self.state[10:])
        self.omega_ne_n = angularRateNedECEF(self.state[7:10], self.state[10:])
        bias = self.state[4:7]
        if all(abs(omega_bi_b) < self.map_config['max_angular_rate']):  # only use new measurement if within limits
            self.omega_bi_b = omega_bi_b - bias
        else:
            print("Rejected gyroscope measurement")
        self.omega_bn_b = self.omega_bi_b - bCn @ self.omega_ei_n - bCn @ self.omega_ne_n

    def _update_accel(self, f_b):
        """
        Convert raw accelerometer measurement into total accel in NED frame
        """
        if all(abs(f_b) < self.map_config['max_specific_force']):  # only use new measurement if within limits
            self.f_b = f_b
        else:
            print("Rejected accelerometer measurement")
        gravity_n = np.array([0, 0, self.map_config['gravity']]).reshape(-1, 1)
        self.accel_n = quatActiveRot(self.state[:4], self.f_b) + gravity_n

    def _update_motion_covariance(self):
        """
        Computes the time dependent motion model covariance matrix Q
        """
        # TODO: Use proper allan variance for bias drift
        self.motion_covariance = np.zeros((12, 12))
        self.motion_covariance[:3, :3] = self.sensor_config[2]['covariance'] * self.dT  # phi
        self.motion_covariance[3:6, 3:6] = self.sensor_config[2]['bias_covariance'] * self.dT  # gyro bias
        self.motion_covariance[6:9, 6:9] = self.sensor_config[1]['covariance'] * self.dT  # velocity
        D = velNed2GeoMatrix(self.state[10:])
        self.motion_covariance[9:, 9:] = D @ self.motion_covariance[6:9, 6:9] @ D.T  # position


class GnssAccelMeasurementModel:
    """ Model for combined GNSS and Accelerometer update """

    def __init__(self, map_config, sensor_config, initial_state):
        self.map_config = map_config
        self.sensor_config = sensor_config
        self.state = initial_state
        self.covariance = None
        self.jacobian = None
        self.measurement_covariance = None
        self.gravity_b = None  # normalized gravity measured in body co-ords
        self.gravity_b_expected = None  # normalized expected measurement
        self.gnss_meas = None
        self.dT = map_config["predict_rate"]  # [s]

    def step(self, state, covariance, gnss_meas, accel_meas):
        self._update_meas(gnss_meas, accel_meas)
        self._update_meas_covariance()
        self.state = state
        self.covariance = covariance
        self._update_jacobian()
        kalman_gain = self._compute_kalman_gain()
        error_state = self._innovation_step(kalman_gain)
        new_covariance = self._covariance_step(kalman_gain)
        return error_state, new_covariance

    def _innovation_step(self, kalman_gain):
        """
        Compute the error vector using innovation between gravity measurement and known
        gravity in NED. Normalize both values as only correcting attitude
        """
        meas_vec = np.vstack((self.gravity_b, self.gnss_meas))  # [g, v, p].T
        expected_vec = np.vstack((self.gravity_b_expected, self.state[7:]))
        innovation = meas_vec - expected_vec
        error_state = kalman_gain @ innovation
        return error_state

    def _covariance_step(self, kalman_gain):
        M = np.eye(12) - kalman_gain @ self.jacobian
        cov_new = M @ self.covariance @ M.T + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        return cov_new

    def _compute_kalman_gain(self):
        innov_cov = self.jacobian @ self.covariance @ self.jacobian.T + self.measurement_covariance
        kalman_gain = self.covariance @ self.jacobian.T @ np.linalg.inv(innov_cov)
        return kalman_gain

    def _update_jacobian(self):
        """
        Find the MEKF measurement Jacobian Ck for the accelerometer and gnss measurements
        """
        C = np.zeros((9, 12))
        gravity_n_expected = np.array([0, 0, 1]).reshape(-1, 1)
        q_n2b = quatInv(self.state[:4])
        self.gravity_b_expected = quatActiveRot(q_n2b, gravity_n_expected)
        C[:3, :3] = 2 * skewMat(self.gravity_b_expected)  # gravity
        C[3:6, 6:9] = np.eye(3)  # vel
        C[6:, 9:] = np.eye(3)  # pos
        self.jacobian = C

    def _update_meas(self, gnss_meas, accel_meas):
        """
        Extract the gravity unit vector from the accelerometer using GNSS for velocity and accel
        """
        omega_ei_e = np.array([0, 0, WGS84['omega_earth']]).reshape(-1, 1)
        self.omega_ei_n = ecef2Ned(omega_ei_e, self.state[10:])
        self.omega_ne_n = angularRateNedECEF(self.state[7:10], self.state[10:])
        Vdot = (gnss_meas[:3] - self.state[7:10]) / self.dT
        bCn = quat2DCM(self.state[:4]).T
        gravity_b = -accel_meas + bCn @ (skewMat(self.omega_ne_n + 2 * self.omega_ei_n) @ gnss_meas[:3] + Vdot)
        self.gravity_b = gravity_b / np.linalg.norm(gravity_b)
        self.gnss_meas = gnss_meas

    def _update_meas_covariance(self):
        """
        Compute the measurement covariance matrix R
        """
        self.measurement_covariance = np.zeros((9, 9))
        cov_gnss = self.sensor_config[4]['covariance']
        # cov_gnss_accel = cov_gnss[:3, :3] / self.dT  TODO :Removed this
        cov_accelerometer = self.sensor_config[1]['covariance']

        # jacobian between gravity_b meas and (f_b, V, Vdot)
        # P = np.hstack((-np.eye(3), skewMat(self.omega_ne_n + 2 * self.omega_ei_n), np.eye(3)))
        # zeta = np.block([[cov_accelerometer, np.zeros((3, 6))],
        #                  [np.zeros((3, 3)), cov_gnss[:3,:3], np.zeros((3, 3))],
        #                  [np.zeros((3, 6)), cov_gnss_accel]])
        P = np.hstack((-np.eye(3), skewMat(self.omega_ne_n + 2 * self.omega_ei_n)))
        zeta = np.block([[cov_accelerometer, np.zeros((3, 3))],
                         [np.zeros((3, 3)), cov_gnss[:3,:3]]])
        cov_gravity = P @ zeta @ P.T
        self.measurement_covariance[:3, :3] = cov_gravity
        self.measurement_covariance[3:, 3:] = cov_gnss


class MagMeasurementModel:
    """ Model for Magnetometer update"""

    def __init__(self, map_config, sensor_config, initial_state):
        self.map_config = map_config
        self.sensor_config = sensor_config
        self.state = initial_state
        self.covariance = None
        self.jacobian = None
        self.measurement_covariance = sensor_config[3]['covariance']
        self.mag_b = None  # normalized magnetic field measured in body frame
        self.mag_b_expected = None

    def step(self, state, covariance, mag_meas):
        self.state = state
        self.covariance = covariance
        self._update_mag_meas(mag_meas)
        self._update_jacobian()
        kalman_gain = self._compute_kalman_gain()
        error_state = self._innovation_step(kalman_gain)
        new_covariance = self._covariance_step(kalman_gain)
        return error_state, new_covariance

    def _innovation_step(self, kalman_gain):
        """
        Compute the error vector using innovation between magnetometer measurement and known
        Earth magnetic field. Normalize both values as only correcting attitude
        """
        innovation = self.mag_b - self.mag_b_expected
        error_state = kalman_gain @ innovation
        return error_state

    def _covariance_step(self, kalman_gain):
        M = np.eye(12) - kalman_gain @ self.jacobian
        cov_new = M @ self.covariance @ M.T + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        return cov_new

    def _compute_kalman_gain(self):
        innov_cov = self.jacobian @ self.covariance @ self.jacobian.T + self.measurement_covariance
        kalman_gain = self.covariance @ self.jacobian.T @ np.linalg.inv(innov_cov)
        return kalman_gain

    def _update_jacobian(self):
        """
        Find the MEKF measurement Jacobian Ck for the magnetometer measurement
        """
        C = np.zeros((3, 12))
        dec = self.map_config['mag_field_decl']
        inc = self.map_config['mag_field_incl']
        mag_n_expected = np.array([[np.cos(inc) * np.cos(dec)],
                                   [np.cos(inc) * np.sin(dec)],
                                   [-np.sin(inc)]]) # unit vec. expected mag field in NED
        q_n2b = quatInv(self.state[:4])
        self.mag_b_expected = quatActiveRot(q_n2b, mag_n_expected)
        C[:, :3] = 2 * skewMat(self.mag_b_expected)
        self.jacobian = C

    def _update_mag_meas(self, mag_meas):
        """
        Normalizes the magnetic field measured in the body frame
        """
        self.mag_b = mag_meas / np.linalg.norm(mag_meas)
    
    


