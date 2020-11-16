import pandas as pd
import numpy as np
from data_utils import *
from config_parsers import update_sensor_config
from coordinate_utils import WGS84


NOISE_FILE = "Data/RAW_GNSS_Log/test_20201104T152440_gnss.txt"
SAVE_FILE = "Configs/sensor_config_20201029T1909.xml"
GNSS_ID = 4


def compute_covariance(noise_data):
    """
    Compute GNSS covariance matrix diag([vel, pos])
    """
    acc = noise_data['accuracy'].to_numpy()
    acc_avg = np.mean(acc)
    geodetic_acc = (acc_avg ** 2) / WGS84['a']  # rad^2
    covariance = np.diag([acc_avg, acc_avg, acc_avg, geodetic_acc, geodetic_acc, acc_avg])
    return covariance


def gnss_calibration(visualize=True):
    """
    Perform gnss calibration using the accuracy data supplied with fixes.
    """
    print("\nRunning GNSS Calibration")
    calibration_dict = {}

    # Load and visualize noise data
    noise_data = extract_gnss_sensor_data(NOISE_FILE, zero_times=False, time_in_seconds=True)
    noise_data = pd.DataFrame.from_dict(noise_data)
    if visualize: visualize_3axis_timeseries(noise_data, GNSS_ID)

    # Noise
    covariance = compute_covariance(noise_data)
    calibration_dict["covariance"] = covariance
    print(f"covariance {covariance}")

    # Saving
    save_params = input("Save calibration params?: ")
    if save_params.lower() in ("y", "yes"):
        update_sensor_config(SAVE_FILE, GNSS_ID, calibration_dict)
        print("Params saved")


def main():
    gnss_calibration()


if __name__ == "__main__":
    main()