import os
import pandas as pd
import numpy as np
from data_utils import *
from config_parsers import update_sensor_config


GYRO_ID = 2
CALIBRATION_FILE = "Data/Calibration/Gyro_Calibration_20201026T1940.txt"
# NOISE_FILE = "Data/Calibration/Sensor_Noise_20201113T1842.txt"
NOISE_FILE = "Data/RAW_IMU_Log/test_20201104T191036_imu.txt"
SAVE_FILE = "Configs/sensor_config_20201029T1909.xml"
TEST_BED_SPEED = 45  # RPM

# Data Clip Points (ms)
bias_start = 1000
bias_end = 12000
x_start = 58000
x_end = 70000
y_start = 96000
y_end = 106000
z_start = 17000
z_end = 28000

# Noise clip times
# noise_start = 1e5
# noise_end = 5.5e6
noise_start = 100
noise_end = 400e3


def compute_bias(gyro_data):
    x_bias = np.mean(gyro_data["x"])
    y_bias = np.mean(gyro_data["y"])
    z_bias = np.mean(gyro_data["z"])
    return np.array([x_bias, y_bias, z_bias]).reshape(-1, 1)


def compute_scale_misalignment(measured_rates, biases):
    """
    Models sensor as y = (I + T)x + b + v
    T is the combined scaling and misalignment matrix
    solve for P = (I + T)
    """
    test_speed = TEST_BED_SPEED * np.pi / 30
    true_rates = ([-test_speed, 0, 0],
                  [0, -test_speed, 0],
                  [0, 0, -test_speed])
    coeffs = np.zeros((9, 9))
    meas_vector = []
    for orientation, meas_data in enumerate(measured_rates):
        for axis in range(3):
            avg_value = np.mean(meas_data.iloc[:, axis+1])
            meas_vector.append(avg_value - biases[axis])
            r_idx = orientation * 3 + axis
            c_idx = axis * 3
            coeffs[r_idx, c_idx:c_idx+3] = true_rates[orientation]
    P = np.linalg.solve(coeffs, np.array(meas_vector))
    return P.reshape(3, 3)


def compute_covariance(noise_data):
    """
    Commpute the variance for each axis and form a
    diagonal covariance matrix
    """
    try:
        noise_data = noise_data.to_numpy()
    except AttributeError:
        noise_data = np.array(noise_data)
    variances = np.std(noise_data[:, 1:], axis=0) ** 2
    covariance = np.diag(variances)
    return covariance


def compute_bias_covariance(noise_data):
    return compute_covariance(noise_data)


def gyroscope_calibration(visualize=False):
    """
    Perform a gyro calibration using a single datafile
    which contains 4 segments:
        1) stationary data for computing biases
        2), 3), 4) constant -ve rotation about x, y, z
    """
    print("\nRunning Gyroscope Calibration")
    calibration_dict = {}

    # Load and visualize calibration data
    gyro_dict = extract_single_imu_sensor_data(CALIBRATION_FILE, GYRO_ID, zero_times=True)
    gyro_data = pd.DataFrame.from_dict(gyro_dict)
    time_steps, rates = compute_logging_rates(gyro_dict)
    print(f"Log size: {gyro_data.shape[0]}")
    print(f"Time step (ms): min:{min(time_steps)}, max:{max(time_steps)}, avg:{np.mean(time_steps):.2f}")
    print(f"Average logging rate (Hz): {1000*np.mean(rates):.2f}")
    if visualize: visualize_3axis_timeseries(gyro_data, GYRO_ID)

    # Calibration
    bias_dataset = get_subsets_between_times(gyro_data, [(bias_start, bias_end)])
    biases = compute_bias(bias_dataset)
    clip_points = [(x_start, x_end), (y_start, y_end), (z_start, z_end)]
    orientation_datasets = get_subsets_between_times(gyro_data, clip_points)
    scaling = compute_scale_misalignment(orientation_datasets, biases)
    scale_inv = np.linalg.inv(scaling)
    calibration_dict["scale"] = scale_inv
    calibration_dict["biases"] = biases

    # Load and visualize noise data
    noise_data = extract_single_imu_sensor_data(NOISE_FILE, GYRO_ID, zero_times=True)
    noise_data = pd.DataFrame.from_dict(noise_data)
    if visualize: visualize_3axis_timeseries(noise_data, GYRO_ID)

    # Noise
    noise_data = get_subsets_between_times(noise_data, [(noise_start, noise_end)])
    covariance = compute_covariance(noise_data)
    bias_covariance = compute_bias_covariance(noise_data)
    calibration_dict["covariance"] = covariance
    calibration_dict["bias_covariance"] = bias_covariance

    # Results Visualization
    corrected_data = batch_measurement_correction(gyro_data, biases, scale_inv)
    corrected_orientation_sets = get_subsets_between_times(corrected_data, clip_points)
    if visualize:
        for idx in range(3):
            raw = orientation_datasets[idx]
            corrected = corrected_orientation_sets[idx]
            visualize_3axis_timeseries(raw, GYRO_ID)
            visualize_3axis_timeseries(corrected, GYRO_ID)
            print(f"rotation rate: {np.mean(corrected.iloc[:, idx+1]) * 30 / np.pi:.2f} RPM")
    print(f"covariance {covariance}")

    # Saving
    save_params = input("Save calibration params?: ")
    if save_params.lower() in ("y","yes"):
        update_sensor_config(SAVE_FILE, GYRO_ID, calibration_dict)
        print("Params saved")

def main():
    gyroscope_calibration()


if __name__ == "__main__":
    main()