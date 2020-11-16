import pandas as pd
import numpy as np
from data_utils import *
from config_parsers import update_sensor_config


ACCEL_ID = 1
CALIBRATION_FILE = "Data/Calibration/Accel_Calibration_20201026T2216.txt"
# NOISE_FILE = "Data/Calibration/Sensor_Noise_20201113T1842.txt"
NOISE_FILE = "Data/RAW_IMU_Log/test_20201104T191036_imu.txt"
SAVE_FILE = "Configs/sensor_config_20201029T1909.xml"
GRAVITY = 9.8025 # [m/s^2] for Wellington


# Calibration data clip points (ms)
x_pos1 = 50000
x_pos2 = 58000
y_pos1 = 117500
y_pos2 = 135000
z_pos1 = 10
z_pos2 = 15000
x_neg1 = 70000
x_neg2 = 82500
y_neg1 = 92500
y_neg2 = 102500
z_neg1 = 22500
z_neg2 = 37500

# Noise clip times
# noise_start = 1e5
# noise_end = 5.5e6
noise_start = 100
noise_end = 400e3


def compute_calibration_params(measured_accels):
    """
    Models sensor as y = (I + T)x + b + v
    T is the combined scaling and misalignment matrix
    solve for P = (I + T) and b simultaneously
    """
    g = GRAVITY
    true_accels = ([g, 0, 0], [0, g, 0], [0, 0, g],
                   [-g, 0, 0], [0, -g, 0], [0, 0, -g])
    coeffs = np.zeros((18, 12))
    coeffs[:, 9:12] = np.tile(np.eye(3), (6, 1))  # bias
    meas_vector = []
    for orientation, meas_data in enumerate(measured_accels):
        for axis in range(3):
            avg_value = np.mean(meas_data.iloc[:, axis+1])
            meas_vector.append(avg_value)
            r_idx = orientation * 3 + axis
            c_idx = axis * 3
            coeffs[r_idx, c_idx:c_idx+3] = true_accels[orientation]
    cal_params = np.linalg.lstsq(coeffs, np.array(meas_vector), rcond=None)[0]
    P = cal_params[:9].reshape(3, 3)
    biases = cal_params[9:].reshape(-1, 1)
    return P, biases


def compute_covariance(noise_data):
    """
    Commpute the average variance across all 3 axes and form a
    diagonal covariance matrix
    """
    try:
        noise_data = noise_data.to_numpy()
    except AttributeError:
        noise_data = np.array(noise_data)
    variances = np.zeros(3)
    for axis in range(3):
        variances[axis] = np.std(noise_data[:, axis + 1])
    avg_variance = np.mean(variances)
    covariance = np.eye(3) * (avg_variance ** 2)
    return covariance


def acclerometer_calibration(visualize=True):
    """
    Perform an accelerometer calibration using a single datafile
    which contains 6 segments:
        Negative and positive stationary measurement for each axis
    """ 
    print("\nRunning Accelerometer Calibration")
    calibration_dict = {}

    # Load and visualize calibration data
    accel_dict = extract_single_imu_sensor_data(CALIBRATION_FILE, ACCEL_ID, zero_times=True)
    accel_data = pd.DataFrame.from_dict(accel_dict)
    time_steps, rates = compute_logging_rates(accel_dict)
    print(f"Log size: {accel_data.shape[0]}")
    print(f"Time step (ms): min:{min(time_steps)}, max:{max(time_steps)}, avg:{np.mean(time_steps):.2f}")
    print(f"Average logging rate (Hz): {1000*np.mean(rates):.2f}")
    if visualize: visualize_3axis_timeseries(accel_data, ACCEL_ID)

    # Calibration
    clip_points = [(x_pos1, x_pos2), (y_pos1, y_pos2), (z_pos1, z_pos2),
                   (x_neg1, x_neg2), (y_neg1, y_neg2), (z_neg1, z_neg2)]
    orientation_datasets = get_subsets_between_times(accel_data, clip_points)
    scaling, biases = compute_calibration_params(orientation_datasets)
    scale_inv = np.linalg.inv(scaling)
    calibration_dict["scale"] = scale_inv
    calibration_dict["biases"] = biases

    # Load and visualize noise data
    noise_data = extract_single_imu_sensor_data(NOISE_FILE, ACCEL_ID, zero_times=True)
    noise_data = pd.DataFrame.from_dict(noise_data)
    if visualize: visualize_3axis_timeseries(noise_data, ACCEL_ID)

    # Noise
    noise_data = get_subsets_between_times(noise_data, [(noise_start, noise_end)])
    covariance = compute_covariance(noise_data)
    calibration_dict["covariance"] = covariance

    # Results Visualization
    corrected_data = batch_measurement_correction(accel_data, biases, scale_inv)
    corrected_orientation_sets = get_subsets_between_times(corrected_data, clip_points)
    if visualize:
        for idx in range(6):
            raw = orientation_datasets[idx]
            corrected = corrected_orientation_sets[idx]
            visualize_3axis_timeseries(raw, ACCEL_ID)
            visualize_3axis_timeseries(corrected, ACCEL_ID)
            print(f"calibrated acceleration: {np.mean(corrected.iloc[:, idx%3+1]):.2f} m/s^2")
    print(f"covariance {covariance}")

    # Saving
    save_params = input("Save calibration params?: ")
    if save_params.lower() in ("y","yes"):
        update_sensor_config(SAVE_FILE, ACCEL_ID, calibration_dict)
        print("Params saved")


def main():
    acclerometer_calibration()
    


if __name__ == "__main__":
    main()