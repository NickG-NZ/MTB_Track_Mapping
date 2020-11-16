"""
The top level tools available to the user for the track mapping application
"""
import csv
import pandas as pd

from MEKF import MEKF
import data_utils as dutils
from config_parsers import load_sensor_config, load_mapping_config
from coordinate_utils import posGeodetic2Ecef, ecef2Ned
from attitude_utils import *


class TrackMapper:

    def __init__(self, track_map, processed_data, map_config_file, sensor_config_file):
        self.map = track_map
        self.data = processed_data
        self.map_config = load_mapping_config(map_config_file)
        self.sensor_config = load_sensor_config(sensor_config_file)
        init_state, init_covariance = self._initialize_state()
        self.filter = MEKF(self.map_config, self.sensor_config, init_state, init_covariance)

    def run_mapping(self, no_updates=False):
        time_idx = 0
        complete = False
        print("\nRunning Map Builder\n...")
        while not complete:
            time_idx += 1
            try:
                t_now = self.data['time'][time_idx]  # [s]
            except IndexError as e:
                complete = True
                continue
            accel_meas = self.data['accel'][time_idx].reshape(-1, 1)
            gyro_meas = self.data['gyro'][time_idx].reshape(-1, 1)
            mag_meas = self.data['mag'][time_idx].reshape(-1, 1)
            vel_meas = self.data['vel'][time_idx].reshape(-1, 1)
            pos_meas = self.data['pos'][time_idx].reshape(-1, 1)
            gnss_meas = np.vstack((vel_meas, pos_meas))
            if no_updates:
                gnss_meas = np.full((6,1), np.nan, dtype=float)
                mag_meas = np.full((3,1), np.nan, dtype=float)

            # Step MEKF
            state, covariance = self.filter.step(gyro_meas, accel_meas, mag_meas, gnss_meas)
            self._update_map(state, covariance, t_now)
        print("Map Building Complete")

    def _initialize_state(self):
        """
        Estimates the initial state:
         - attitude quaternion (body to NED) using Wahba's/q-method algorithm
         - gyro bias (set to 0)
         - geodetic location (set to first gnss meas)
         - NED velocity (set to 0)
         """
        # Magnetometer
        dec = self.map_config['mag_field_decl']
        inc = self.map_config['mag_field_incl']
        mag_n = np.array([[np.cos(inc) * np.cos(dec)],
                          [np.cos(inc) * np.sin(dec)],
                          [-np.sin(inc)]])  # unit vec. expected mag field in NED
        mag_b_measurements = np.vstack(self.data['mag'][:3])
        mag_b = np.mean(mag_b_measurements, axis=0).reshape(-1, 1)  # avg. first 3 magnetometer measurements
        mag_b = mag_b / np.linalg.norm(mag_b)

        # Accelerometer
        g_n = np.array([0, 0, 1]).reshape(-1, 1)  # unit vec gravity in NED
        g_b_measurements = np.vstack(self.data['accel'][:5])
        g_b = -np.mean(g_b_measurements, axis=0).reshape(-1, 1)  # avg. first 5 accelerometer measurements
        g_b = g_b / np.linalg.norm(g_b)

        # q-method optimization for quaternion
        q_b2n = qMethod(g_b, g_n, mag_b, mag_n)

        # Set initial state
        initial_state = np.zeros((13, 1))
        initial_state[:4] = q_b2n
        initial_state[10:] = self.data['pos'][0].reshape(-1, 1)

        # set covariance and scale appropriately for units
        initial_covariance = np.eye(12)
        initial_covariance[:3, :3] *= 0.5  # [rad]
        initial_covariance[3:6, 3:6] *= 0.2  # [rad/s]
        initial_covariance[6:9, 6:9] *= 1  # [m/s]
        initial_covariance[9:, 9:] *= 5  # [m]

        # initialize map
        pos_ecef = posGeodetic2Ecef(initial_state[10:])
        self.map.start_pos_ecef = pos_ecef
        self.map.start_pos_geodetic = initial_state[10:]
        self._update_map(initial_state, initial_covariance, 0)
        return initial_state, initial_covariance

    def _update_map(self, state, covariance, time):
        self.map.states.append(state)
        self.map.covariances.append(covariance)
        self.map.map_times.append(time)
        # Derived states
        rel_pos_ecef = posGeodetic2Ecef(state[10:]) - self.map.start_pos_ecef
        rel_pos_ned = ecef2Ned(rel_pos_ecef, self.map.start_pos_geodetic)  # relative to initial NED frame
        self.map.positions_ecef.append(rel_pos_ecef)
        self.map.positions_ned.append(rel_pos_ned)
        self.map.bike_attitudes.append(self._compute_bike_attitude(state))

    def _compute_bike_attitude(self, state):
        """
        Compute orientation of bike in NED frame by accounting for the device mounting orientation,
        specified as yaw, roll, pitch sequence from bike frame to sensor frame
        Bike coord frame is denoted by {C}
        """
        euler_c2b = self.map_config["sensor_orientation"]  # yaw, roll, pitch
        yaw = np.array([np.cos(euler_c2b[0, 0] / 2), 0, 0, -np.sin(euler_c2b[0, 0] / 2)]).reshape(-1, 1)
        roll = np.array([np.cos(euler_c2b[1, 0] / 2), -np.sin(euler_c2b[1, 0] / 2), 0, 0]).reshape(-1, 1)
        pitch = np.array([np.cos(euler_c2b[2, 0] / 2), 0, -np.sin(euler_c2b[2, 0] / 2), 0]).reshape(-1, 1)
        q_c2b = quatMultiply(pitch, quatMultiply(roll, yaw))
        q_c2n = quatMultiply(state[:4], q_c2b)
        return q_c2n

    def _compute_total_meters_descended(self):
        pass

    def _compute_total_meters_ascended(self):
        pass

    def _computes_elevation_extremes(self):
        pass


class TrackMap:

    def __init__(self):
        self.start_pos_ecef = None
        self.start_pos_geodetic = None
        self.states = []
        self.covariances = []
        self.map_times = []
        self.positions_ned = []
        self.positions_ecef = []
        self.bike_attitudes = []
        self.total_descent = None
        self.total_ascsent =None

    @property
    def states_array(self):
        return np.concatenate(self.states, axis=1).T

    @property
    def bike_attitudes_array(self):
        return np.concatenate(self.bike_attitudes, axis=1).T

    @property
    def positions_ned_array(self):
        return np.concatenate(self.positions_ned, axis=1).T

    @property
    def positions_ecef_array(self):
        return np.concatenate(self.positions_ecef, axis=1).T


class TrackDataParser:
    """
    A tool for taking raw IMU and GNSS data files and
    creating datasets for the track mapping application
    """
    def __init__(self, map_config_file, sensor_config_file):
        self.map_config = load_mapping_config(map_config_file)
        self.sensor_config = load_sensor_config(sensor_config_file)
        self.dT = self.map_config['predict_rate']  # [s]
        self.processed_data = {'time': [],  # [s]
                               'accel': [],  # [m/s^2]
                               'gyro': [],  # [rad/s]
                               'mag': [],  # [muT]
                               'vel': [],  # [m/s]
                               'pos': []}  # [rad, m]

    def save_to_file(self, out_file):
        header = ["Time[s]", "Accel[m/s^2]", "Gyro[rad/s]", "Mag[muT]", "NED-Vel[m/s]", "Geodetic-Pos[rad,m]"]
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(header)
            for idx, time in enumerate(self.processed_data['time']):
                line_to_write = [time]
                measurements = np.array([])
                for key, val in self.processed_data.items():
                    if key == 'time': continue
                    measurements = np.hstack((measurements, val[idx]))
                line_to_write.extend(list(measurements))
                writer.writerow(line_to_write)
        print(f"Saved Processed Measurement Data to {out_file.split('/')[-1]}")

    def parse_raw_data(self, imu_file, gnss_file):
        # Load raw sensor data
        data_raw = {}
        for id_num in range(1, 5):
            if id_num != 4:
                data_raw[id_num] = dutils.extract_single_imu_sensor_data(imu_file, id_num, time_in_seconds=True)
            else:
                data_raw[id_num] = dutils.extract_gnss_sensor_data(gnss_file, time_in_seconds=True)

        # Apply sensor calibration to IMU data
        accel_data, gyro_data, mag_data, gnss_data = self._calibrate_data(data_raw)

        # Align data times
        t_start, t_end = self._find_start_end_times(accel_data, gyro_data, mag_data, gnss_data)
        a_idx = np.where(accel_data[:, 0] >= t_start)[0][0]
        g_idx = np.where(gyro_data[:, 0] >= t_start)[0][0]
        m_idx = np.where(mag_data[:, 0] >= t_start)[0][0]
        gns_idx = np.where(gnss_data[:, 0] >= t_start)[0][0]

        # Sample data at specified rate
        idces = [a_idx, g_idx, m_idx, gns_idx]
        data_sets = [accel_data, gyro_data, mag_data, gnss_data]
        new_meas = [False, False, False, False]
        prev_idces = [-1, -1 ,-1, -1]

        self.processed_data['time'].append(0)
        self.processed_data['accel'].append(accel_data[a_idx, 1:])
        self.processed_data['gyro'].append(gyro_data[g_idx, 1:])
        self.processed_data['mag'].append(mag_data[m_idx, 1:])
        self.processed_data['vel'].append(np.zeros(3))
        self.processed_data['pos'].append(gnss_data[gns_idx, 1:4])

        time_stamp = self.dT
        parsing_complete = False
        while not parsing_complete:

            # Find closest measurement from each sensor prior to time_stamp
            for i, sensor_t_idx in enumerate(idces):
                try:
                    sensor_time = data_sets[i][sensor_t_idx + 1, 0] - t_start
                except IndexError:
                    parsing_complete = True
                    continue
                if sensor_time < time_stamp:  # skip measurements according to specified time step
                    new_meas[i] = False
                    idces[i] += 1
                elif sensor_t_idx == prev_idces[i]: # sensor value hasn't changed (don't record twice)
                    new_meas[i] = False
                else:
                    new_meas[i] = True

            if new_meas[0] and new_meas[1]:  # accel and gyro
                self.processed_data['accel'].append(accel_data[idces[0], 1:])
                self.processed_data['gyro'].append(gyro_data[idces[1], 1:])
                self.processed_data['time'].append(time_stamp)
                time_stamp = round(time_stamp + self.dT, 5)
                if time_stamp >= t_end:
                    parsing_complete = True
                if new_meas[3]:  # gnss
                    gns_idx = idces[3]
                    pos = gnss_data[gns_idx, 1:4]
                    pos_prev = gnss_data[gns_idx-1, 1:4]
                    speed = gnss_data[gns_idx, 4]
                    speed_prev = gnss_data[gns_idx-1, 4]
                    vel = dutils.compute_gnss_velocity(pos_prev, speed_prev, pos, speed)
                else:
                    pos = np.full(3, np.nan, dtype='float')
                    vel = np.full(3, np.nan, dtype='float')
                if new_meas[2]:  # mag
                    mag = mag_data[idces[2], 1:]
                else:
                    mag = np.full(3, np.nan, dtype='float')
                self.processed_data['mag'].append(mag)
                self.processed_data['vel'].append(vel)
                self.processed_data['pos'].append(pos)
                prev_idces = idces.copy()
        print("Calibration Downsampling and Alignment of Sensor Data Complete")

    def _calibrate_data(self, data_raw):
        """
        Returns calibrated data sets as numpy arrays
        """
        data_calibrated = []
        for key, data in data_raw.items():
            data_array = pd.DataFrame.from_dict(data).to_numpy()
            if key != 4:  # not gnss
                biases = self.sensor_config[key]['biases']
                scale_inv = self.sensor_config[key]['scale']
                data_array = dutils.batch_measurement_correction(data_array, biases, scale_inv)
            data_calibrated.append(data_array)
        acc_c, gyro_c, mag_c, gns_c = data_calibrated
        return acc_c, gyro_c, mag_c, gns_c

    @staticmethod
    def _find_start_end_times(acc, gyro, mag, gns):
        starts = []
        ends = []
        for dataset in [acc, gyro, mag, gns]:
            starts.append(dataset[0, 0])
            ends.append(dataset[-1, 0])
        start_time = max(starts)
        end_time = min(ends)
        return start_time, end_time

